package com.cameragestures

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Stage 6 Android binding for HandGestureRecognizing (3-phase pipeline).
 *
 * Thread-safety: all public methods are internally serialised with a lock so that
 * processShot (called from a MediaPipe result thread) and tickTimers (called from a
 * timer loop) never race on the underlying C++ recognizer.
 *
 * Usage:
 *   val rec = HandGestureRecognizing()
 *   rec.onGestureDetected = { name, confidence -> ... }
 *   rec.create(gestureModel)
 *   // per frame:
 *   rec.processShot(handShot)
 *   // periodically (~10 ms):
 *   rec.tickTimers(System.currentTimeMillis() / 1000.0)
 *   // on teardown:
 *   rec.destroy()
 */
class HandGestureRecognizing {

    var onGestureDetected: ((name: String, confidence: Float) -> Unit)? = null

    private val lock   = ReentrantLock()
    private var handle = 0L
    val isCreated: Boolean get() = handle != 0L

    /** Create the recognizer backed by an already-loaded GestureModel. */
    fun create(model: GestureModel): Boolean {
        if (model.nativeHandle() == 0L) return false
        lock.withLock {
            if (handle != 0L) destroyLocked()
            handle = RecognizerNative.create(model.nativeHandle())
        }
        return handle != 0L
    }

    /** Push one HandShot through the 3-phase pipeline. Call on every camera frame. */
    fun processShot(shot: HandShot) {
        val lm = FloatArray(63) { i ->
            val pt = shot.landmarks[i / 3]
            when (i % 3) { 0 -> pt.x; 1 -> pt.y; else -> pt.z }
        }
        val handednessCode = when (shot.handedness) {
            Handedness.LEFT  -> 0
            Handedness.RIGHT -> 1
            else             -> 2
        }
        val pending: String?
        lock.withLock {
            if (handle == 0L) return
            RecognizerNative.processShot(
                handle, lm, shot.timestamp, handednessCode,
                if (shot.isAbsent) 1 else 0
            )
            pending = RecognizerNative.pollGesture(handle)
        }
        pending?.let { encoded ->
            val idx = encoded.lastIndexOf('|')
            if (idx >= 0) {
                val name       = encoded.substring(0, idx)
                val confidence = encoded.substring(idx + 1).toFloatOrNull() ?: 0f
                if (name.isNotEmpty() && name != "_none") {
                    onGestureDetected?.invoke(name, confidence)
                }
            }
        }
    }

    /** Advance T_commit / T_min_buffer timers. Call every ~10 ms from any thread. */
    fun tickTimers(nowSeconds: Double) {
        lock.withLock {
            if (handle != 0L) RecognizerNative.tickTimers(handle, nowSeconds)
        }
    }

    /** Reset Phase-1 gate and Phase-2 state. */
    fun reset() {
        lock.withLock { if (handle != 0L) RecognizerNative.reset(handle) }
    }

    fun destroy() {
        lock.withLock { destroyLocked() }
    }

    private fun destroyLocked() {
        if (handle != 0L) {
            RecognizerNative.destroy(handle)
            handle = 0L
        }
    }
}

internal object RecognizerNative {
    init { System.loadLibrary("cameragestures") }

    @JvmStatic external fun create(modelHandle: Long): Long
    @JvmStatic external fun destroy(handle: Long)
    @JvmStatic external fun processShot(
        handle: Long, landmarks: FloatArray,
        timestamp: Double, handedness: Int, isAbsent: Int
    )
    @JvmStatic external fun tickTimers(handle: Long, now: Double): Int
    @JvmStatic external fun reset(handle: Long)
    /** Returns "gesture_name|confidence" string, or null if no gesture pending. */
    @JvmStatic external fun pollGesture(handle: Long): String?
}

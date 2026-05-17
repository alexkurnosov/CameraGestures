package com.cameragestures

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

/**
 * Stage 3 Android binding for HandsRecognizing.
 *
 * Mirrors the iOS V1 HandsRecognizing API surface.
 * MediaPipe tasks-vision handles landmark detection; results are pushed to
 * the C++ film buffer via JNI (HandsRecognizerNative).
 */
class HandsRecognizing(private val context: Context) {

    // ---- Callbacks (mirrors iOS API) ----------------------------------------

    var handshotCallback: ((HandShot) -> Unit)? = null

    // ---- Config / state -----------------------------------------------------

    private var config: HandsRecognizingConfig = HandsRecognizingConfig()
    private var nativeHandle: Long = 0L
    private var landmarker: HandLandmarker? = null

    // ---- Lifecycle ----------------------------------------------------------

    fun initialize(config: HandsRecognizingConfig) {
        this.config   = config
        nativeHandle  = HandsRecognizerNative.create()

        val modelPath = "hand_landmarker.task"   // bundled as Android asset

        val baseOpts = BaseOptions.builder()
            .setModelAssetPath(modelPath)
            .build()

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOpts)
            .setNumHands(if (config.detectBothHands) 2 else 1)
            .setMinHandDetectionConfidence(config.minDetectionConfidence)
            .setMinTrackingConfidence(config.minTrackingConfidence)
            .setMinHandPresenceConfidence(config.minDetectionConfidence)
            .setRunningMode(com.google.mediapipe.tasks.vision.core.RunningMode.LIVE_STREAM)
            .setResultListener(::onLandmarkerResult)
            .setErrorListener { e -> android.util.Log.e("CameraGestures", "MediaPipe error", e) }
            .build()

        landmarker = HandLandmarker.createFromOptions(context, options)
    }

    fun stop() {
        landmarker?.close()
        landmarker = null
        if (nativeHandle != 0L) {
            HandsRecognizerNative.reset(nativeHandle)
        }
    }

    fun destroy() {
        stop()
        if (nativeHandle != 0L) {
            HandsRecognizerNative.destroy(nativeHandle)
            nativeHandle = 0L
        }
    }

    /** Push a camera frame (BGRA8 or YUV) to MediaPipe for async processing. */
    fun pushFrame(image: Image, timestampMs: Long) {
        val landmarker = landmarker ?: return
        // Convert YUV_420_888 → Bitmap → MPImage
        val bitmap  = image.toBitmap()
        val mpImage = BitmapImageBuilder(bitmap).build()
        landmarker.detectAsync(mpImage, timestampMs)
    }

    /** Harvest the accumulated HandFilm and reset the buffer. */
    fun harvestHandfilm(): HandFilm {
        if (nativeHandle == 0L) return HandFilm(emptyList(), 0.0)
        val raw = HandsRecognizerNative.harvest(nativeHandle)
        return parseHandFilm(raw)
    }

    fun resetHandfilm() {
        if (nativeHandle != 0L) HandsRecognizerNative.reset(nativeHandle)
    }

    // ---- Private — MediaPipe result handler ---------------------------------

    private fun onLandmarkerResult(result: HandLandmarkerResult, input: com.google.mediapipe.framework.image.MPImage, timestampMs: Long) {
        val ts = timestampMs / 1000.0

        if (result.landmarks().isEmpty()) {
            pushAbsent(ts)
            return
        }

        for ((handIndex, lmList) in result.landmarks().withIndex()) {
            val pts = lmList.map { Triple(it.x(), it.y(), it.z()) }

            // Treat all-zero detections as absent (same as iOS V1).
            val (wx, wy, wz) = pts[0]; val (lx, ly, lz) = pts[9]
            val dx = lx - wx; val dy = ly - wy; val dz = lz - wz
            if (dx*dx + dy*dy + dz*dz < 1e-12f) { pushAbsent(ts); continue }

            val handednessCode: Int = result.handedness().getOrNull(handIndex)
                ?.firstOrNull()?.categoryName()
                ?.let { if (it.lowercase() == "left") 0 else 1 } ?: 1

            val landmarks = FloatArray(63)
            for ((i, pt) in pts.withIndex()) {
                landmarks[i * 3 + 0] = pt.first
                landmarks[i * 3 + 1] = pt.second
                landmarks[i * 3 + 2] = pt.third
            }

            HandsRecognizerNative.pushHandshot(
                nativeHandle, landmarks, ts, handednessCode, 0
            )

            handshotCallback?.invoke(
                HandShot(pts.map { Point3D(it.first, it.second, it.third) },
                    ts, if (handednessCode == 0) Handedness.LEFT else Handedness.RIGHT)
            )
        }
    }

    private fun pushAbsent(ts: Double) {
        HandsRecognizerNative.pushHandshot(
            nativeHandle, FloatArray(63), ts, 2 /* unknown */, 1 /* absent */
        )
    }

    // ---- Flat float-array → HandFilm ----------------------------------------

    private fun parseHandFilm(raw: FloatArray): HandFilm {
        if (raw.size < 2) return HandFilm(emptyList(), 0.0)
        val startTime  = raw[0].toDouble()
        val shotCount  = raw[1].toInt()
        val shots      = mutableListOf<HandShot>()
        for (i in 0 until shotCount) {
            val base       = 2 + i * 66
            val ts         = raw[base + 0].toDouble()
            val handedness = raw[base + 1].toInt()
            val isAbsent   = raw[base + 2].toInt() != 0
            val pts        = (0 until 21).map { j ->
                Point3D(raw[base + 3 + j * 3], raw[base + 4 + j * 3], raw[base + 5 + j * 3])
            }
            shots += HandShot(pts, ts,
                when (handedness) { 0 -> Handedness.LEFT; 1 -> Handedness.RIGHT; else -> Handedness.UNKNOWN },
                isAbsent)
        }
        return HandFilm(shots, startTime)
    }

    // ---- YUV → Bitmap -------------------------------------------------------

    private fun Image.toBitmap(): Bitmap {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val ySize   = yBuffer.remaining()
        val uSize   = uBuffer.remaining()
        val vSize   = vBuffer.remaining()
        val nv21    = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out      = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val jpegBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }
}

// ---- Data classes (Kotlin equivalents of C/Swift types) --------------------

data class Point3D(val x: Float, val y: Float, val z: Float)

enum class Handedness { LEFT, RIGHT, UNKNOWN }

data class HandShot(
    val landmarks:   List<Point3D>,
    val timestamp:   Double,
    val handedness:  Handedness,
    val isAbsent:    Boolean = false
)

data class HandFilm(
    val frames:    List<HandShot>,
    val startTime: Double
) {
    val endTime:        Double get() = frames.lastOrNull()?.timestamp ?: startTime
    val gestureDuration: Double get() = endTime - startTime
    val inViewDuration: Double get() {
        val visible = frames.filter { !it.isAbsent }
        if (visible.size < 2) return 0.0
        return visible.zipWithNext().sumOf { (a, b) -> b.timestamp - a.timestamp }
    }
    val inViewFrameCount: Int get() = frames.count { !it.isAbsent }
}

data class HandsRecognizingConfig(
    val detectBothHands:        Boolean = true,
    val minDetectionConfidence: Float   = 0.5f,
    val minTrackingConfidence:  Float   = 0.5f,
    val targetFPS:              Int     = 30
)

// ---- JNI native declarations -----------------------------------------------

internal object HandsRecognizerNative {
    init { System.loadLibrary("cameragestures") }

    @JvmStatic external fun create(): Long
    @JvmStatic external fun destroy(handle: Long)
    @JvmStatic external fun pushHandshot(
        handle: Long, landmarks: FloatArray,
        timestamp: Double, handedness: Int, isAbsent: Int)
    @JvmStatic external fun harvest(handle: Long): FloatArray
    @JvmStatic external fun reset(handle: Long)
}

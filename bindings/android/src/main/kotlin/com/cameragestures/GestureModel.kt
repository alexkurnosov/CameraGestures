package com.cameragestures

/**
 * Stage 6 Android binding for GestureModel.
 *
 * Wraps the C ABI cg_gesture_model_* functions. All methods are thin pass-throughs;
 * threading is the caller's responsibility (must stay on a single serial queue).
 */
class GestureModel {

    private var handle: Long = 0L
    val isLoaded: Boolean get() = handle != 0L

    /**
     * Load the Phase-3 gesture MLP using the server's own class list. PREFERRED.
     *
     * [gestureIds] comes from the `gesture_ids.json` sidecar shipped with the
     * model and is used verbatim — its order is the model's output order. Only
     * the server knows how its model was trained; deriving the list on-device
     * is a guess that broke every client on 2026-09-09, when the server stopped
     * emitting the `_none` class. Returns true on success.
     */
    fun loadWithIds(tflitePath: String, gestureIds: List<String>): Boolean {
        if (gestureIds.isEmpty()) return false
        if (handle != 0L) destroy()
        handle = GestureModelNative.loadWithIds(tflitePath, gestureIds.toTypedArray())
        return handle != 0L
    }

    /**
     * Load the Phase-3 gesture MLP, deriving the class list from the registry. LEGACY.
     *
     * Reconstructs a contract the server owns: registry IDs plus `_none`,
     * sorted. Use [loadWithIds] whenever the sidecar is available.
     */
    fun load(tflitePath: String, registryPath: String): Boolean {
        if (handle != 0L) destroy()
        handle = GestureModelNative.load(tflitePath, registryPath)
        return handle != 0L
    }

    /** Optionally load the Phase-2 pose MLP. Returns true on success. */
    fun loadPose(tflitePath: String, manifestPath: String): Boolean {
        if (handle == 0L) return false
        return GestureModelNative.loadPose(handle, tflitePath, manifestPath) != 0
    }

    fun destroy() {
        if (handle != 0L) {
            GestureModelNative.destroy(handle)
            handle = 0L
        }
    }

    internal fun nativeHandle(): Long = handle
}

internal object GestureModelNative {
    init { System.loadLibrary("cameragestures") }

    @JvmStatic external fun load(tflitePath: String, registryPath: String): Long
    @JvmStatic external fun loadWithIds(tflitePath: String, gestureIds: Array<String>): Long
    @JvmStatic external fun destroy(handle: Long)
    @JvmStatic external fun loadPose(handle: Long, tflitePath: String, manifestPath: String): Int
}

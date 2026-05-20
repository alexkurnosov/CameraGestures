package com.cameragestures.demo

import android.content.Context
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.cameragestures.GestureModel
import com.cameragestures.HandGestureRecognizing
import com.cameragestures.HandsRecognizing
import com.cameragestures.HandsRecognizingConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

private const val TAG = "GestureViewModel"

sealed class ModelState {
    object Loading : ModelState()
    object Ready : ModelState()
    data class Error(val message: String) : ModelState()
}

class GestureViewModel(private val context: Context) : ViewModel() {

    val detectedGesture = MutableStateFlow<String?>(null)
    val modelState      = MutableStateFlow<ModelState>(ModelState.Loading)

    // Single serial executor for all recognizer operations.
    private val pipelineExecutor = Executors.newSingleThreadExecutor()

    private val handsRecognizing     = HandsRecognizing(context)
    private val gestureModel         = GestureModel()
    private val gestureRecognizing   = HandGestureRecognizing()
    private val initialized          = AtomicBoolean(false)

    init {
        viewModelScope.launch(Dispatchers.IO) { initPipeline() }
        startTimerLoop()
    }

    // ---- Init ---------------------------------------------------------------

    private fun initPipeline() {
        val tfliteFile   = copyAsset("gesture_model.tflite")
        val registryFile = copyAsset("gestures.json")

        if (tfliteFile == null || registryFile == null) {
            modelState.value = ModelState.Error(
                "gesture_model.tflite or gestures.json not found.\n" +
                "Copy them to apps/demo-android/app/src/main/assets/ and rebuild."
            )
            return
        }

        val loaded = gestureModel.load(tfliteFile.absolutePath, registryFile.absolutePath)
        if (!loaded) {
            modelState.value = ModelState.Error("Failed to load gesture model (TFLite error).")
            return
        }

        val created = gestureRecognizing.create(gestureModel)
        if (!created) {
            modelState.value = ModelState.Error("Failed to create recognizer.")
            return
        }

        // Wire handshot callback: MediaPipe result thread → pipelineExecutor.
        gestureRecognizing.onGestureDetected = { name, confidence ->
            Log.d(TAG, "Gesture: $name (${String.format("%.2f", confidence)})")
            viewModelScope.launch {
                detectedGesture.value = name
                delay(3_000)
                detectedGesture.compareAndSet(name, null)
            }
        }

        handsRecognizing.handshotCallback = { shot ->
            pipelineExecutor.execute { gestureRecognizing.processShot(shot) }
        }

        handsRecognizing.initialize(HandsRecognizingConfig())
        initialized.set(true)
        modelState.value = ModelState.Ready
    }

    // ---- Camera frames ------------------------------------------------------

    /** Called from the CameraX ImageAnalysis executor on every frame. */
    fun onFrame(imageProxy: ImageProxy) {
        val image = imageProxy.image
        if (image != null && initialized.get()) {
            // MediaPipe detectAsync is thread-safe; call directly from the analyzer thread.
            val tsMs = imageProxy.imageInfo.timestamp / 1_000_000L
            handsRecognizing.pushFrame(image, tsMs)
        }
        imageProxy.close()
    }

    // ---- Timer loop ---------------------------------------------------------

    private fun startTimerLoop() {
        viewModelScope.launch {
            while (isActive) {
                delay(10)
                if (initialized.get()) {
                    val nowSec = System.currentTimeMillis() / 1000.0
                    pipelineExecutor.execute { gestureRecognizing.tickTimers(nowSec) }
                }
            }
        }
    }

    // ---- Cleanup ------------------------------------------------------------

    override fun onCleared() {
        initialized.set(false)
        handsRecognizing.destroy()
        gestureRecognizing.destroy()
        gestureModel.destroy()
        pipelineExecutor.shutdown()
    }

    // ---- Helpers ------------------------------------------------------------

    private fun copyAsset(name: String): File? = try {
        val file = File(context.filesDir, name)
        if (!file.exists()) {
            context.assets.open(name).use { input ->
                file.outputStream().use { input.copyTo(it) }
            }
        }
        file
    } catch (e: Exception) {
        Log.w(TAG, "Asset not found: $name — ${e.message}")
        null
    }
}

class GestureViewModelFactory(private val context: Context) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T =
        GestureViewModel(context.applicationContext) as T
}

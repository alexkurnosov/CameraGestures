package com.cameragestures.demo

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.isGranted
import com.google.accompanist.permissions.rememberPermissionState
import com.google.accompanist.permissions.shouldShowRationale
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color    = Color.Black
                ) {
                    DemoScreen()
                }
            }
        }
    }
}

// ---- Root screen ------------------------------------------------------------

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun DemoScreen(
    vm: GestureViewModel = viewModel(
        factory = GestureViewModelFactory(LocalContext.current)
    )
) {
    val cameraPermission  = rememberPermissionState(Manifest.permission.CAMERA)
    val detectedGesture   by vm.detectedGesture.collectAsState()
    val modelState        by vm.modelState.collectAsState()

    LaunchedEffect(Unit) {
        if (!cameraPermission.status.isGranted) {
            cameraPermission.launchPermissionRequest()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {

        when {
            !cameraPermission.status.isGranted -> {
                PermissionPlaceholder(
                    showRationale = cameraPermission.status.shouldShowRationale,
                    onRequest     = { cameraPermission.launchPermissionRequest() }
                )
            }
            else -> {
                CameraPreview(
                    onFrame  = vm::onFrame,
                    modifier = Modifier.fillMaxSize()
                )

                // Status / error overlay (loading or error states)
                when (val state = modelState) {
                    is ModelState.Loading -> LoadingOverlay()
                    is ModelState.Error   -> ErrorOverlay(state.message)
                    is ModelState.Ready   -> { /* no status overlay */ }
                }

                // Gesture result overlay
                if (modelState is ModelState.Ready) {
                    GestureOverlay(
                        gesture  = detectedGesture,
                        modifier = Modifier.align(Alignment.BottomCenter)
                    )
                }
            }
        }
    }
}

// ---- Camera preview ---------------------------------------------------------

@Composable
fun CameraPreview(onFrame: (androidx.camera.core.ImageProxy) -> Unit, modifier: Modifier = Modifier) {
    val context        = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val executor       = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(Unit) { onDispose { executor.shutdown() } }

    AndroidView(
        factory = { ctx ->
            val previewView = PreviewView(ctx)

            val future = ProcessCameraProvider.getInstance(ctx)
            future.addListener({
                val provider = future.get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(executor, onFrame) }

                provider.unbindAll()
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    analysis
                )
            }, ContextCompat.getMainExecutor(ctx))

            previewView
        },
        modifier = modifier
    )
}

// ---- Overlays ---------------------------------------------------------------

@Composable
fun GestureOverlay(gesture: String?, modifier: Modifier = Modifier) {
    Box(
        modifier          = modifier
            .fillMaxWidth()
            .padding(bottom = 48.dp),
        contentAlignment  = Alignment.Center
    ) {
        val label = gesture ?: ""
        if (label.isNotEmpty()) {
            Text(
                text      = label,
                fontSize  = 36.sp,
                color     = Color.White,
                textAlign = TextAlign.Center,
                modifier  = Modifier
                    .background(Color.Black.copy(alpha = 0.55f), MaterialTheme.shapes.medium)
                    .padding(horizontal = 24.dp, vertical = 12.dp)
            )
        }
    }
}

@Composable
fun LoadingOverlay() {
    Box(
        modifier         = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.6f)),
        contentAlignment = Alignment.Center
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            CircularProgressIndicator(color = Color.White)
            Spacer(Modifier.height(16.dp))
            Text("Loading gesture model…", color = Color.White)
        }
    }
}

@Composable
fun ErrorOverlay(message: String) {
    Box(
        modifier         = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.8f))
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text      = message,
            color     = Color(0xFFFF6B6B),
            fontSize  = 16.sp,
            textAlign = TextAlign.Center
        )
    }
}

@Composable
fun PermissionPlaceholder(showRationale: Boolean, onRequest: () -> Unit) {
    Column(
        modifier            = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text      = if (showRationale)
                "Camera access is required to detect hand gestures."
            else
                "Waiting for camera permission…",
            color     = Color.White,
            textAlign = TextAlign.Center
        )
        if (showRationale) {
            Spacer(Modifier.height(24.dp))
            Button(onClick = onRequest) { Text("Grant Permission") }
        }
    }
}

# demo-android

Minimal Jetpack Compose app that proves `CameraGestures` works on Android.

**Stage 3 status:** scaffold (bindings/android) created. Full Compose demo UI is populated in Stage 6.

## What it will do (Stage 6)

- Camera permission request via CameraX
- Live camera preview
- MediaPipe hand landmark detection via `HandsRecognizing`
- Overlay of 21 landmark dots drawn on a `Canvas`

## Build prerequisites

1. Android NDK r25+ — set `ANDROID_NDK` env var.
2. From repo root: `./core/scripts/build-android.sh`
   → produces `bindings/android/src/main/cpp/prebuilt/{abi}/libCameraGestures.a`
3. Open `apps/demo-android/` in Android Studio → Sync Gradle → Run on device (API 24+).

## Key dependencies

| Library | Version |
|---|---|
| `com.google.mediapipe:tasks-vision` | 0.10.14 |
| `cameragestures` AAR (local, `bindings/android/`) | local |
| AndroidX CameraX | 1.3.x |
| Jetpack Compose BOM | 2024.02.00 |

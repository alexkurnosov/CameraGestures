# CameraGestures — core C++ library

Cross-platform gesture-recognition library. Exposes a C ABI consumed by platform
bindings in `../bindings/`.

## Build — iOS (Stage 3+)

```sh
./scripts/build-ios.sh
# Output: bindings/ios/XCFramework/CameraGestures.xcframework
# Then: cd apps/training-ios && pod install
```

## Build — Android (Stage 3+)

```sh
export ANDROID_NDK=/path/to/ndk
./scripts/build-android.sh
# Output: bindings/android/src/main/cpp/prebuilt/{abi}/libCameraGestures.a
# Then open apps/demo-android/ in Android Studio
```

## macOS status — SPIKE NOT YET STARTED

macOS support is a stretch goal (plan §10 Stage 3 step 6).

The spike entry point is `src/hands_recognizing/macos/HandLandmarkerLiteRT.cpp`.
It must implement a two-stage LiteRT-only decode of `hand_landmarker.task`
(palm detector → ROI crop → landmark model) and reach parity with the iOS
MediaPipe Tasks output within tolerance (mean Euclidean distance < 0.01
normalized across a fixture set).

**Current decision: PENDING** — spike not started. macOS build target is
disabled by default (`BUILD_MACOS=OFF`). This file will be updated once the
spike concludes with either "macOS supported" or "macOS deferred to follow-up plan".

## Third-party prebuilts

See `third_party/tflite/README.md`.

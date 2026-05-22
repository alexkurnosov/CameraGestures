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

## Build — macOS (Stage 7)

```sh
# Step 1: vendor TFLite (requires TensorFlowLiteC CocoaPod in cache)
./scripts/vendor-tflite-macos.sh

# Step 2: build universal framework
./scripts/build-macos.sh
# Output: bindings/macos/CameraGestures.framework
# Then: open apps/demo-macos in Xcode (see apps/demo-macos/README.md)
```

## macOS status — SPIKE IMPLEMENTED (Stage 7)

macOS support uses a LiteRT-only two-stage hand detection pipeline:
`src/hands_recognizing/macos/HandLandmarkerLiteRT.cpp`

The pipeline:
1. Resize BGRA frame to 192×192, run `hand_detector.tflite` (SSD palm detection)
2. Decode anchors + NMS → ROI per hand
3. Affine-warp ROI to 224×224, run `hand_landmarks_detector.tflite`
4. Project 21 landmarks back to image coordinates

TFLite library: re-platformed from iOS Simulator xcframework 2.17.0 via
`scripts/vendor-tflite-macos.sh`. See `third_party/tflite/macos/README.md`.

**Parity status:** spike implemented; formal parity validation (fixture comparison
against iOS MediaPipe Tasks output) is the next verification step. See
`apps/demo-macos/README.md` for the parity test command once fixture data is recorded.

## Third-party prebuilts

See `third_party/tflite/README.md`.

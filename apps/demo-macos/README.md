# Demo App — macOS

Minimal SwiftUI app: camera preview + detected-gesture overlay.
Uses the LiteRT-only HandLandmarker (no MediaPipe framework).

## Prerequisites

1. **Build the macOS framework** (only needed once or after core/ changes):
   ```sh
   # Vendor TFLite (requires TensorFlowLiteC pod to be in CocoaPods cache)
   ./core/scripts/vendor-tflite-macos.sh

   # Build CameraGestures.framework
   ./core/scripts/build-macos.sh
   ```

2. **Prepare assets** — copy the bundled model files into the Xcode target:
   - `gesture_model.tflite` (from `apps/demo-android/app/src/main/assets/`)
   - `gestures.json` (same location)
   These must be added as Bundle Resources in the Xcode target.

## Setting up the Xcode project

The `DemoMac/` directory contains the Swift source files.
Create a new Xcode project ("macOS › App") and:

1. Add all `.swift` files from `DemoMac/` to the target.
2. Under **Build Settings**:
   - `Other Linker Flags`: add
     `-L$(REPO_ROOT)/bindings/macos/CameraGestures.framework/Versions/Current`  
     `-L$(REPO_ROOT)/core/third_party/tflite/macos`  
     `-lTensorFlowLiteC -lc++`
   - `Header Search Paths`: `$(REPO_ROOT)/core/include`
3. Under **Build Phases → Link Binary With Libraries**:
   - Add `CameraGestures.framework` from `bindings/macos/`
4. Under **Build Phases → Copy Bundle Resources**:
   - Add `gesture_model.tflite`, `gestures.json`, `hand_landmarker.task` (from `core/assets/`)
5. In **Signing & Capabilities**:
   - Enable **Camera** in App Sandbox (or add `NSCameraUsageDescription` to Info.plist).
6. Add `Info.plist` from `DemoMac/Info.plist` (or merge NSCameraUsageDescription into the target's plist).

## What the demo does

- Opens the camera (front or built-in).
- Runs the LiteRT-only hand detector + landmark model on each frame.
- Feeds landmarks through the gesture recognition pipeline.
- Displays the recognized gesture name as a centered overlay.
- No server contact at runtime.

## Parity note (Stage 3 go/no-go spike)

The LiteRT HandLandmarker should produce landmarks within mean Euclidean
distance < 0.01 (normalized) versus the iOS MediaPipe Tasks implementation,
measured over the recorded HandFilm corpus.

Run the parity test with:
```sh
# (fixture-based test — see core/tests/macos_parity_test.cpp once implemented)
cmake --build build-macos --target macos_parity_test && ./build-macos/macos_parity_test
```

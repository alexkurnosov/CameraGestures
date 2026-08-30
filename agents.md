# CameraGestures — Architecture Reference

## Project Overview

CameraGestures is a cross-platform C++ library for real-time dynamic hand gesture recognition. It captures hand movements through a camera and emits gesture events to the host application. The core library is written in C++17 with a C ABI; platform bindings (Swift for iOS/macOS, Kotlin for Android) wrap that ABI.

## Design Principles

- **Inference-only library.** The C++ core loads a `.tflite` model, runs the recognition pipeline, and fires callbacks. Server I/O lives in the Training App, not the library.
- **Camera-agnostic.** The library accepts pre-captured BGRA8 frames from platform code; it does not own the camera.
- **C ABI surface.** One header (`core/include/CameraGestures/CameraGestures.h`) using `cg_*` prefixed opaque handles and callbacks. Platform wrappers call only this ABI.
- **Vendored third-party.** LiteRT prebuilts are checked in under `core/third_party/tflite/`. No Bazel, no download scripts.

## System Architecture

```
Camera frames (BGRA8, supplied by platform code)
       │
       ▼
HandsRecognizing   ─── iOS:     MediaPipeTasksVision (CocoaPod)
                   ─── Android: com.google.mediapipe:tasks-vision (AAR via JNI)
                   ─── macOS:   Apple Vision VNDetectHumanHandPoseRequest
       │
       ▼  HandShots → accumulated into HandFilms
HandGestureRecognizing  (3-phase orchestration)
   ├── MotionGate      — gates motion start/end
   ├── HoldDetector    — detects steady poses within a moving sequence
   ├── PrefixMatcher   — early high-confidence predictions
   └── GestureModel    — TFLite inference (LiteRT C++ runtime, vendored)
           └── FeaturePreprocessor — HandFilm → (60×126) + 256-feature vectors
       │
       ▼
DetectedGesture callbacks → application layer
```

## Module Descriptions

### HandGestureTypes (`core/src/types/`)
POD structs shared by all modules: `Point3D`, `HandShot`, `HandFilm`, `GestureDefinition`, `GesturePrediction`, `DetectedGesture`. Also contains `GestureRegistry` — JSON-persisted list of `GestureDefinition`. Format matches `<AppSupport>/gestures.json` from the iOS Training App.

**Dependencies:** nlohmann/json only.

### HandsRecognizing (`core/src/hands_recognizing/`)
Converts raw BGRA8 camera frames into streams of `HandShot` structs (21 3D landmarks per hand). Platform-specific implementations selected at CMake build time:

| Platform | Implementation | Approach |
|---|---|---|
| iOS | `ios/HandLandmarkerIOS.mm` | Obj-C++ shim over `MPHandLandmarker` |
| Android | `android/HandLandmarkerAndroid.cpp` | JNI bridge to Kotlin MediaPipe wrapper |
| macOS | `macos/` (Swift binding) | Apple Vision `VNDetectHumanHandPoseRequest` |
| common | `common/HandsRecognizing.cpp` | Frame-rate limiting, `is_absent` computation |

### GestureModel (`core/src/gesture_model/`)
Loads a server-trained `.tflite` and classifies `HandFilm` sequences. Components:

- `FeaturePreprocessor` — produces a `(60, 126)` sequence array and a 256-feature summary. Bit-for-bit equivalent to `CameraGestures-server/server/ml/preprocessor.py`.
- `PoseManifest` — JSON-defined geometric features.
- `TFLiteBackend` — wraps `tflite::Interpreter` from vendored LiteRT.
- `GestureModel` — top-level classifier: `classify(handfilm) → GesturePrediction`.

**Dependencies:** vendored LiteRT (`core/third_party/tflite/`).

### HandGestureRecognizing (`core/src/hand_gesture_recognizing/`)
The 3-phase orchestration layer. Drives the camera → landmark → buffer → gate → classify → callback pipeline. Contains:

- `HandGestureRecognizing` — top-level C++ class; exposed through the C ABI as `cg_recognizer_*`.
- `MotionGate` — Phase 1: decides when a motion segment is significant.
- `HoldDetector` — Phase 2: detects steady poses within a moving segment.
- `PrefixMatcher` — fires early predictions on high-confidence prefixes.

Configuration parameters are defined in `CameraGestures-server/docs/architecture/parameters.md` (canonical source).

**Dependencies:** all three lower modules plus nlohmann/json.

---

## Repository Layout

```
CameraGestures/
├── core/                        # C++ library
│   ├── include/CameraGestures/  # Public C ABI headers
│   ├── src/                     # Implementation (types/, hands_recognizing/, gesture_model/, hand_gesture_recognizing/)
│   ├── third_party/
│   │   ├── tflite/              # Vendored LiteRT prebuilts per platform
│   │   └── nlohmann_json/       # Header-only JSON
│   ├── assets/
│   │   └── hand_landmarker.task # MediaPipe model bundle (used on iOS/Android)
│   ├── tests/                   # gtest unit tests + replay rig CLI
│   └── scripts/
│       ├── build-ios.sh         # → bindings/ios/XCFramework/CameraGestures.xcframework
│       ├── build-android.sh     # → bindings/android/ AAR
│       └── build-macos.sh       # → bindings/macos/CameraGestures.framework
│
├── bindings/                    # Platform wrappers (call the C ABI only)
│   ├── ios/                     # CameraGestures CocoaPod + Swift wrappers
│   │   ├── CameraGestures.podspec
│   │   ├── CameraGestures/      # HandGestureTypes.swift, HandsRecognizing.swift, …
│   │   └── XCFramework/         # Prebuilt CameraGestures.xcframework
│   ├── android/                 # Gradle module (AAR) + Kotlin wrappers + JNI bridge
│   └── macos/                   # Swift wrappers + CameraGestures.framework
│
└── apps/                        # Apps that consume the library
    ├── training-ios/            # Training App v2 — sole pod dependency: CameraGestures
    ├── demo-ios/                # Minimal iOS demo (camera + gesture overlay)
    ├── demo-android/            # Minimal Android demo
    └── demo-macos/              # Minimal macOS demo
```

---

## Platform Implementations

### iOS (`bindings/ios/`)

- **Pod:** `CameraGestures` (unified, replaces the four V1 pods).
- **Linkage:** XCFramework statically linked; `MediaPipeTasksVision` pulled by CocoaPods at consumer build time.
- **Swift surface:** `HandGestureTypes.swift`, `HandsRecognizing.swift`, `GestureModel.swift`, `HandGestureRecognizing.swift` — same class names as the V1 pods.
- **Training App v2:** `apps/training-ios/ModelTraining/ModelTrainingApp.xcworkspace`. Podfile lists only `pod 'CameraGestures'`. Server I/O (registration, example upload, training trigger, model download) is in Swift in `apps/training-ios/ModelTraining/Networking/`.
- **Min deployment:** iOS 16.0.

### Android (`bindings/android/`)

- **Artifact:** `cameragestures-release.aar`.
- **Hand detection:** `com.google.mediapipe:tasks-vision` Gradle dependency; Kotlin wrapper calls it and posts results back to C++ via JNI.
- **Inference:** `libcameragestures.so` (arm64-v8a, armeabi-v7a, x86_64) with LiteRT statically linked.
- **Min API:** 24 (Android 7.0).
- **Demo:** `apps/demo-android/` — Jetpack Compose, camera permission, gesture overlay.

### macOS (`bindings/macos/`)

- **Artifact:** `CameraGestures.framework` (universal arm64 + x86_64).
- **Hand detection:** Apple Vision `VNDetectHumanHandPoseRequest` — no MediaPipe SDK.
- **Inference:** LiteRT statically linked.
- **Min macOS:** 13 (Ventura).
- **Demo:** `apps/demo-macos/DemoMac/` — SwiftUI, `AVCaptureSession`, gesture overlay.

---

## Python Training Server

The training server lives in the sibling repo **`CameraGestures-server`**. It receives labelled `HandFilm` examples from the iOS Training App, trains a Keras MLP (Phase 1) or LSTM (Phase 2) model, and exports a `.tflite` for download. See that repo for setup, API docs, and deployment instructions.

### ML Pipeline (key files in `CameraGestures-server/server/`)

- `ml/preprocessor.py` — `HandFilm` → `(60, 126)` numpy array + 256-feature summary. The C++ `FeaturePreprocessor` must match this output bit-for-bit.
- `ml/trainer_rf_mlp.py` — Phase 1: shallow MLP on stat features → `.tflite`.
- `ml/trainer_lstm.py` — Phase 2: LSTM on full sequence → `.tflite`.

---

## Glossary

| Term | Definition |
|------|-----------|
| **HandShot** | 21 3D landmarks captured at one instant |
| **HandFilm** | Time-ordered sequence of HandShots representing one gesture motion |
| **GestureDefinition** | `{id, name, description}` struct identifying a gesture at runtime |
| **GestureRegistry** | JSON-persisted list of GestureDefinitions (`gestures.json`) |
| **MotionGate** | Phase 1 component: decides when motion is significant enough to start/end a candidate segment |
| **HoldDetector** | Phase 2 component: detects steady poses inside a moving segment |
| **PrefixMatcher** | Early-exit component: fires high-confidence predictions before a full HandFilm closes |
| **FeaturePreprocessor** | Converts a HandFilm into numeric feature arrays for the TFLite model |
| **LiteRT / TFLite** | TensorFlow Lite C++ runtime, vendored in `core/third_party/tflite/` |
| **hand_landmarker.task** | MediaPipe model bundle (palm detector + landmark model) used on iOS and Android |

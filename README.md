# CameraGestures

A cross-platform C++ library for real-time hand gesture recognition, with platform bindings for iOS, Android, and macOS.

## Overview

CameraGestures captures hand movements via camera and translates them into recognizable dynamic gestures. The core library is written in C++17 and exposes a C ABI so that iOS (Swift), Android (Kotlin), and macOS (Swift) can each wrap it with native idioms.

## Repository Layout

```
CameraGestures/
├── core/                    # C++17 library — the cross-platform engine
│   ├── include/CameraGestures/   # Public C ABI headers (CameraGestures.h, Types.h)
│   ├── src/
│   │   ├── types/               # HandGestureTypes: POD structs, GestureRegistry JSON
│   │   ├── hands_recognizing/   # HandsRecognizing: platform impls (ios/, android/, common/)
│   │   ├── gesture_model/       # GestureModel: TFLite inference + FeaturePreprocessor
│   │   └── hand_gesture_recognizing/  # 3-phase orchestrator, MotionGate, HoldDetector, PrefixMatcher
│   ├── third_party/
│   │   ├── tflite/              # Vendored LiteRT prebuilts (ios/, android/, macos/, include/)
│   │   └── nlohmann_json/       # Header-only JSON
│   ├── assets/
│   │   └── hand_landmarker.task # MediaPipe hand-landmark model bundle
│   ├── tests/                   # gtest unit tests + replay rig
│   └── scripts/
│       ├── build-ios.sh         # → bindings/ios/XCFramework/CameraGestures.xcframework
│       ├── build-android.sh     # → bindings/android/ AAR
│       └── build-macos.sh       # → bindings/macos/CameraGestures.framework
│
├── bindings/                # Platform wrappers around the C ABI
│   ├── ios/                 # CocoaPod "CameraGestures" + Swift wrappers
│   ├── android/             # Gradle module + Kotlin wrappers + JNI bridge
│   └── macos/               # Swift wrappers (uses Apple Vision for hand detection)
│
└── apps/                    # Apps that consume the library
    ├── training-ios/        # Training App v2 — iOS SwiftUI app (sole dependency: CameraGestures pod)
    ├── demo-ios/            # Minimal iOS demo: camera + gesture overlay
    ├── demo-android/        # Minimal Android demo
    └── demo-macos/          # Minimal macOS demo
```

## Building

### Prerequisites

- CMake 3.16+, C++17 compiler
- iOS/macOS: Xcode 14+
- Android: Android NDK r25+

### iOS XCFramework

```bash
./core/scripts/build-ios.sh
# Output: bindings/ios/XCFramework/CameraGestures.xcframework
```

### Android AAR

```bash
./core/scripts/build-android.sh
# Output: bindings/android/build/outputs/aar/cameragestures-release.aar
```

### macOS Framework

```bash
./core/scripts/build-macos.sh
# Output: bindings/macos/CameraGestures.framework
```

### C++ Unit Tests (host/macOS)

```bash
mkdir -p core/build-test && cd core/build-test
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . && ctest --output-on-failure
```

## Training App (iOS)

Open `apps/training-ios/ModelTraining/ModelTrainingApp.xcworkspace` in Xcode. The app depends only on `pod 'CameraGestures'` (the unified V2 pod). It handles gesture data collection, server upload, training trigger, and model download in Swift.

## Demo Apps

Each demo bundles a server-trained `gesture_model.tflite` and `gestures.json` and requires no server contact at runtime.

| App | Location | Stack |
|-----|----------|-------|
| iOS demo | `apps/demo-ios/` | SwiftUI |
| Android demo | `apps/demo-android/` | Jetpack Compose |
| macOS demo | `apps/demo-macos/` | SwiftUI on macOS |

## Training Server

The Python FastAPI training server lives in the sibling repo **`CameraGestures-server`**. It trains `.tflite` models from uploaded hand gesture examples. See that repo for setup, deployment, and API documentation.

## Versioning

The repo-root `VERSION` file (`MAJOR.MINOR.BUILD`) is the single source of truth for the library version. A pre-commit hook bumps `BUILD` on each commit. The training server repo carries its own independent `VERSION`. Enable once per clone:

```bash
git config core.hooksPath .githooks
```

## Architecture

```
Camera frames (BGRA8, platform-supplied)
       │
       ▼
HandsRecognizing  ─── iOS: MediaPipeTasksVision (pod)
                  ─── Android: tasks-vision AAR via JNI
                  ─── macOS: Apple Vision VNDetectHumanHandPoseRequest
       │
       ▼  HandShots / HandFilms
HandGestureRecognizing (3-phase orchestrator)
   ├── MotionGate   — decides when motion is significant
   ├── HoldDetector — detects steady poses mid-sequence
   ├── PrefixMatcher— early high-confidence predictions
   └── GestureModel — TFLite inference via LiteRT C++ runtime
       │
       ▼
DetectedGesture callbacks → application layer
```

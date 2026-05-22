# demo-android

Minimal Jetpack Compose app that proves the full CameraGestures pipeline works on Android.
Camera permission → hand landmark detection (MediaPipe) → 3-phase gesture recognition → on-screen label.

## What it does

- Requests camera permission at launch
- Streams the front camera through CameraX
- MediaPipe `hand_landmarker.task` detects hand landmarks per frame
- `HandGestureRecognizing` (3-phase: MotionGate → HoldDetector → GestureModel) classifies gestures
- Detected gesture name is shown as a text overlay at the bottom of the screen (clears after 3 s)

## Build prerequisites

### 1 — Android NDK r25+

Set the `ANDROID_NDK` environment variable to your NDK root.

### 2 — Build the core C++ library

```bash
# From the repo root:
./core/scripts/build-android.sh
# Produces: bindings/android/src/main/cpp/prebuilt/{abi}/libCameraGestures.a
```

This step requires CMake 3.21+ and the Android NDK.

### 3 — Copy model assets

Put these two files in `apps/demo-android/app/src/main/assets/`:

| File | Source |
|---|---|
| `gesture_model.tflite` | Latest model from the training server |
| `gestures.json` | GestureRegistry matching the model |

`hand_landmarker.task` is bundled automatically from `core/assets/`.

### 4 — Open in Android Studio

Open `apps/demo-android/` as a Gradle project → Sync → Run on a physical device (API 24+, camera required).

Android Studio will download the Gradle wrapper on first sync.

## Project structure

```
apps/demo-android/
├── app/
│   ├── build.gradle.kts
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/          ← gesture_model.tflite + gestures.json go here
│       ├── kotlin/com/cameragestures/demo/
│       │   ├── MainActivity.kt    (Compose UI: camera preview + gesture overlay)
│       │   └── GestureViewModel.kt (pipeline lifecycle + state)
│       └── res/values/
├── build.gradle.kts
├── settings.gradle.kts      (includes :app + :cameragestures from bindings/android/)
└── gradle.properties
```

## Follow-ups / Known Issues

| # | Area | Description |
|---|------|-------------|
| 1 | Gesture sensitivity | Gestures are detected but require exaggerated, energetic movements. Tune the pipeline parameters: lower `MotionGate` threshold, reduce `HoldDetector` min-in-view duration, or relax confidence threshold in `HandGestureRecognizing.kt` (`name != "_none"` filter). May also need to retrain the model with more varied (slower) examples. |

---

## Key dependencies

| Library | Version |
|---|---|
| `com.google.mediapipe:tasks-vision` | 0.10.14 |
| `:cameragestures` (local module, `bindings/android/`) | local |
| AndroidX CameraX | 1.3.2 |
| Jetpack Compose BOM | 2024.02.00 |
| Accompanist Permissions | 0.34.0 |

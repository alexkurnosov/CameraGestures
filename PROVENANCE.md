# Provenance

Copyright © 2026 Aleksei Kurnosov. All rights reserved pending selection of an
open-source licence for the client library (see "Licensing status" below).

This note records who wrote CameraGestures, when, on what equipment, and from which
components. It is kept in the repository so that the record is dated by version
control. Entries are added, not rewritten, when facts change.

## Authorship

- Sole author: Aleksei Kurnosov, committing under the GitHub accounts
  `alexkurnosov` and `KurnosoviOS`.
- AI coding assistants (Claude, via Claude Code) were used as tools under the
  author's direction for parts of the code, tests and documentation. All design
  decisions, the pipeline architecture, the data collection and the training
  methodology are the author's.
- No other person has contributed code, models or data.

## Timeline (from git history)

| Date | Milestone |
|---|---|
| 2026-02-14 | Initial commit: dynamic gesture recognition prototype |
| 2026-02-24 | Switched hand tracking to Google MediaPipe HandLandmarker; first CocoaPods integration |
| 2026-03-06 | First implementation of the training server (client–server training) |
| 2026-03-18 | MediaPipe pinned as a git submodule (commit 9e4f898) |
| 2026-04-22 to 2026-05-09 | Three-phase recognition pipeline designed and implemented in stages |
| 2026-05-17 | Rework into `core/` (C++17, C ABI), `bindings/` and `apps/` |
| 2026-05-20 | Android demo |
| 2026-05-22 to 2026-05-26 | macOS demo, first with TFLite hand detection, then with Apple Vision |
| 2026-08-30 | Training server split into the separate `CameraGestures-server` repository |

## Development conditions

- Personal equipment only: a personal MacBook Air (MacBookAir10,1, macOS 26),
  personal iOS and Android test devices [list models], and a personal VPS for the
  training server.
- Personal accounts only: personal GitHub account, personal Apple and Google
  developer accounts, personal cloud and email.
- Personal time only, outside any employment working hours.
- No equipment, network, account, code, data, document, model or other resource of
  any employer or client was used at any point.

## Components and their origins

| Component | Origin | Licence | Where |
|---|---|---|---|
| Hand landmark detection | Google MediaPipe Tasks Vision 0.10.14 | Apache-2.0 | iOS: `MediaPipeTasksVision` pod; Android: `com.google.mediapipe:tasks-vision` |
| Hand landmark model | Google MediaPipe `hand_landmarker.task` | Apache-2.0 | `core/assets/` |
| Inference runtime | LiteRT / TensorFlow Lite 2.17.0 prebuilts | Apache-2.0 | `core/third_party/tflite/` |
| JSON | nlohmann/json (header-only) | MIT | `core/third_party/nlohmann_json/` |
| iOS/macOS CMake toolchain | leetal/ios-cmake | BSD-3-Clause | `core/cmake/ios-cmake/` |
| Unit test framework | GoogleTest (fetched at build time) | BSD-3-Clause | `core/tests/` |
| macOS hand detection | Apple Vision framework | Apple SDK terms | `bindings/macos/` |
| Android camera and UI | AndroidX CameraX, Compose, Accompanist, Guava | Apache-2.0 | `apps/demo-android/`, `bindings/android/` |

Everything not listed above is original work by the author: the C++17 core
(`types/`, `hands_recognizing/`, `gesture_model/`, `hand_gesture_recognizing/`
including the three-phase orchestrator, MotionGate, HoldDetector and
PrefixMatcher), the public C ABI, the Swift and Kotlin wrappers and JNI bridge, the
demo and training apps, the gesture definitions, and the trained gesture models.

## Training data and models

- All gesture recordings ("hand films") used for training were recorded by the
  author on personal devices [confirm: author only, or named consenting
  participants].
- No third-party dataset was used.
- Trained models (`gesture_model.tflite` and successors) are derived only from
  those recordings and from the author's training code.

## Independence

CameraGestures was conceived and developed independently. It was not created at
the request or under the instruction of any employer or client, was not created in
the performance of any employment duties, and does not contain or draw on any
code, model, data, design, documentation or confidential information belonging to
any employer or client. Its design references are public: the MediaPipe and
TensorFlow Lite documentation and the publicly available hand-tracking solutions
surveyed in `hand_gesture_recognition_solutions.rtf`.

## Licensing status

- No licence has been granted to anyone as of this note. The repository is public
  for review; all rights are reserved by the author.
- An open-source licence (Apache-2.0) for the client library is planned; the
  training server, dataset and trained production models are to remain private.
- Known cleanup before any release: `bindings/ios/CameraGestures.podspec` still
  carries placeholder `author` and `license` fields from scaffolding and must be
  corrected to the author's name and the chosen licence.

## Log

| Date | Entry |
|---|---|
| 2026-09-14 | Provenance note created. |

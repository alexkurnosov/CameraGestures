// macOS HandLandmarker implementation — Stage 3 spike entry point.
//
// SPIKE STATUS: NOT YET STARTED.
//
// Goal: decode hand_landmarker.task (palm detector + landmark TFLite models +
// metadata) using only the vendored LiteRT C++ runtime — no MediaPipe framework.
//
// Spike budget (see plan §10 Stage 3 step 6): implement two-stage decode:
//   1. Palm detection inference  → ROI crop
//   2. Hand landmark inference   → 21 × (x, y, z) normalized landmarks
//   3. NMS / simple tracker for multi-hand stability
//
// Go/no-go gate: run fixture BGRA frames through both this impl and the iOS
// MediaPipe Tasks impl, compare landmark outputs. Parity threshold: mean
// Euclidean distance < 0.01 (normalized) across the fixture set.
//
// If parity is not reachable within the spike budget, set BUILD_MACOS=OFF in
// core/CMakeLists.txt, document deferral in core/README.md, and skip Stage 7.
//
// Reference:
//   hand_landmarker.task bundle layout: see MediaPipe source
//   palm_detection_lite.tflite  — input: 192×192 BGRA → anchors → palm ROIs
//   hand_landmark_lite.tflite   — input: 224×224 cropped hand → 21 keypoints
//   Metadata JSON               — anchor config, normalization params
//
// Files to create when spike starts:
//   HandLandmarkerLiteRT.hpp   — IHandLandmarker implementation
//   TaskBundleParser.cpp/hpp   — parse .task zip bundle
//   AnchorDecoder.cpp/hpp      — palm-detect anchor generation + NMS
//   HandROICropper.cpp/hpp     — perspective-correct crop from palm ROI

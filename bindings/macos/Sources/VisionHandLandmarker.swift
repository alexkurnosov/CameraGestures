// VisionHandLandmarker.swift
// Wraps Apple Vision's VNDetectHumanHandPoseRequest and converts results to
// cg_handshot so the unchanged C++ recognizer can consume them.
//
// Replaces the broken TFLite two-stage palm-detector → landmark pipeline.
// Vision provides reliable 21-landmark tracking on macOS 11+ with no model files.
//
// Coordinate-system notes:
//   - Vision uses bottom-left origin (Cocoa convention). The gesture model
//     expects top-left origin (MediaPipe convention). Flip: y_out = 1 - vision_y.
//   - Vision is 2D; z is set to 0 for every landmark.
//   - Mirroring is handled at the overlay layer (HandOverlayView mirrored:true).
//     Landmark coords are emitted in raw un-mirrored image space.

import Vision
import AVFoundation
import CameraGesturesC

// MARK: - Joint → MediaPipe index mapping

/// Vision joint names in MediaPipe canonical order (index 0–20).
/// Source: VNHumanHandPoseObservation.JointName documentation vs MediaPipe hand topology.
private let kJointOrder: [VNHumanHandPoseObservation.JointName] = [
    .wrist,                                         // 0
    .thumbCMC, .thumbMP, .thumbIP, .thumbTip,       // 1–4
    .indexMCP, .indexPIP, .indexDIP, .indexTip,     // 5–8
    .middleMCP, .middlePIP, .middleDIP, .middleTip, // 9–12
    .ringMCP, .ringPIP, .ringDIP, .ringTip,         // 13–16
    .littleMCP, .littlePIP, .littleDIP, .littleTip, // 17–20
]

// MARK: - VisionHandLandmarker

/// Runs one `VNDetectHumanHandPoseRequest` per sample buffer and pushes a
/// `cg_handshot` into the C++ recognizer.
///
/// **Threading**: call `process(_:timestamp:)` on the capture queue
/// (`com.cameragestures.macos.processing`). Vision inference is synchronous
/// and runs on the same thread — no extra queue is created per frame.
final class VisionHandLandmarker {

    private let recognizerRef: cg_hands_recognizer_ref
    /// Callback fired after each push so HandsRecognizing can forward it.
    var handshotCallback: ((HandShot) -> Void)?

    // Re-using a sequence handler lets Vision apply temporal smoothing.
    private let sequenceHandler = VNSequenceRequestHandler()

    init(recognizerRef: cg_hands_recognizer_ref) {
        self.recognizerRef = recognizerRef
    }

    // MARK: - Per-frame entry point

    /// Process one camera frame. Performs Vision inference synchronously on the
    /// calling thread and immediately pushes the resulting `cg_handshot`.
    func process(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) {
        let request = VNDetectHumanHandPoseRequest()
        request.maximumHandCount = 1

        do {
            try sequenceHandler.perform([request], on: pixelBuffer,
                                        orientation: .up)
        } catch {
            // Vision error — treat as absent frame.
            pushAbsent(timestamp: timestamp)
            return
        }

        guard let observations = request.results, !observations.isEmpty,
              let obs = observations.first else {
            pushAbsent(timestamp: timestamp)
            return
        }

        pushObservation(obs, timestamp: timestamp)
    }

    // MARK: - Private helpers

    private func pushObservation(_ obs: VNHumanHandPoseObservation,
                                 timestamp: TimeInterval) {
        var landmarks: [Point3D] = []
        landmarks.reserveCapacity(21)

        for jointName in kJointOrder {
            if let point = try? obs.recognizedPoint(jointName),
               point.confidence > 0.1 {
                // Vision origin is bottom-left; flip Y to top-left.
                landmarks.append(Point3D(x: Float(point.x),
                                         y: Float(1.0 - point.y),
                                         z: 0.0))
            } else {
                // Low-confidence or missing joint — use a neutral zero value.
                landmarks.append(Point3D(x: 0.0, y: 0.0, z: 0.0))
            }
        }

        let chirality = obs.chirality
        let handedness: LeftOrRight =
            chirality == .left  ? .left  :
            chirality == .right ? .right : .unknown

        let shot = HandShot(landmarks:   landmarks,
                            timestamp:   timestamp,
                            leftOrRight: handedness,
                            isAbsent:    false)
        push(shot)
    }

    private func pushAbsent(timestamp: TimeInterval) {
        let shot = HandShot(landmarks:   Array(repeating: Point3D(x: 0, y: 0, z: 0), count: 21),
                            timestamp:   timestamp,
                            leftOrRight: .unknown,
                            isAbsent:    true)
        push(shot)
    }

    private func push(_ shot: HandShot) {
        // toCHandshot() is defined in HandGestureRecognizing.swift (internal).
        var cShot = shot.toCHandshot()
        cg_hands_recognizer_push_handshot(recognizerRef, &cShot)
        handshotCallback?(shot)
    }
}

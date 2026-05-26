import Foundation
import Combine
import AVFoundation
// HandGestureRecognizing, HandsRecognizing, GestureModel, HandGestureTypes are
// compiled directly into this target from bindings/macos/Sources/.

enum PipelineState {
    case loading, ready, running, error(String)
}

@MainActor
class GestureViewModel: ObservableObject {

    // Confirmed gesture (confidence >= kConfirmedThreshold); cleared after 3 s.
    @Published var detectedGesture: String? = nil
    // All-class probabilities from the latest cycle end (updated without cooldown).
    @Published var latestPredictions: [GesturePrediction] = []
    // Most recent non-absent handshot landmarks (21 points in normalized [0,1]
    // image space). Powers the on-screen hand skeleton overlay so we can see
    // exactly what the landmark model is producing each frame.
    @Published var latestLandmarks: [Point3D] = []

    @Published var pipelineState: PipelineState = .loading

    // Use confidenceThreshold: 0.65 for the "confirmed" green badge.
    private let recognizer = HandGestureRecognizing(
        config: HandGestureRecognizingConfig(
            confidenceThreshold: 0.65,
            motionGateConfig: .defaultConfig
        )
    )
    private var clearTask: Task<Void, Never>?

    init() {
        Task { await setup() }
    }

    // MARK: Setup

    private func setup() async {
        do {
            let modelPath    = try bundledAssetPath("gesture_model.tflite")
            let registryPath = try bundledAssetPath("gestures.json")

            try await recognizer.initialize()
            try recognizer.loadModel(from: modelPath, registryPath: registryPath)

            // Raw predictions: fires on every cycle end regardless of confidence.
            // Drives the always-visible probability panel.
            recognizer.rawPredictionsCallback = { [weak self] predictions in
                Task { @MainActor [weak self] in
                    self?.latestPredictions = predictions
                }
            }

            // Confirmed gesture: fires only when confidence >= 0.65 (Swift-side gate).
            recognizer.gestureDetectionCallback = { [weak self] detected in
                Task { @MainActor [weak self] in
                    self?.showGesture(detected.prediction.gestureName)
                }
            }

            // Per-frame handshot: feeds the live landmark overlay. Skip absent
            // shots so the overlay vanishes when no detection is emitted.
            recognizer.handshotCallback = { [weak self] shot in
                Task { @MainActor [weak self] in
                    if shot.isAbsent {
                        self?.latestLandmarks = []
                    } else {
                        self?.latestLandmarks = shot.landmarks
                    }
                }
            }

            pipelineState = .ready
        } catch {
            pipelineState = .error(error.localizedDescription)
        }
    }

    /// Forward the session so ContentView can show a live preview.
    var previewSession: AVCaptureSession? { recognizer.previewSession }

    // MARK: Camera

    func startCamera() async {
        guard await HandsRecognizing.requestCameraPermission() else {
            pipelineState = .error("Camera permission denied")
            return
        }
        do {
            try await recognizer.start()
            pipelineState = .running
        } catch {
            pipelineState = .error(error.localizedDescription)
        }
    }

    func stopCamera() {
        recognizer.stop()
        pipelineState = .ready
    }

    // MARK: Private

    private func showGesture(_ name: String) {
        clearTask?.cancel()
        detectedGesture = name
        clearTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            if !Task.isCancelled { detectedGesture = nil }
        }
    }

    private func bundledAssetPath(_ name: String) throws -> String {
        if let path = Bundle.main.path(forResource: (name as NSString).deletingPathExtension,
                                       ofType: (name as NSString).pathExtension) {
            return path
        }
        throw NSError(domain: "DemoMac", code: 1,
                      userInfo: [NSLocalizedDescriptionKey:
                          "\(name) not found in bundle. Add it to the Xcode target."])
    }
}

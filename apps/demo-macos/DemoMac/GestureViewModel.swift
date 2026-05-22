import Foundation
import AVFoundation
import CameraGesturesMac  // macOS binding module (HandsRecognizing, GestureModel, HandGestureRecognizing)

enum PipelineState {
    case loading, ready, running, error(String)
}

@MainActor
class GestureViewModel: ObservableObject {

    @Published var detectedGesture: String? = nil
    @Published var pipelineState: PipelineState = .loading

    private let recognizer = HandGestureRecognizing()
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
            try recognizer.loadModel(from: modelPath)

            recognizer.gestureDetectionCallback = { [weak self] detected in
                Task { @MainActor [weak self] in
                    self?.showGesture(detected.prediction.gestureName)
                }
            }

            pipelineState = .ready
        } catch {
            pipelineState = .error(error.localizedDescription)
        }
    }

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
        // Try main bundle (copied into .app at build time)
        if let path = Bundle.main.path(forResource: (name as NSString).deletingPathExtension,
                                       ofType: (name as NSString).pathExtension) {
            return path
        }
        throw NSError(domain: "DemoMac", code: 1,
                      userInfo: [NSLocalizedDescriptionKey:
                          "\(name) not found in bundle. Add it to the Xcode target."])
    }
}

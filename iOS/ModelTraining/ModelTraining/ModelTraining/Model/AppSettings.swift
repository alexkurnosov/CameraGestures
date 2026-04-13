import SwiftUI
import Combine
import HandGestureRecognizingFramework
import GestureModelModule

class AppSettings: ObservableObject {
    @Published var colorScheme: ColorScheme? = nil
    @Published var preferredCamera: Int = 0
    @Published var targetFPS: Int = 30
    @Published var confidenceThreshold: Float = 0.7
    @Published var enableHapticFeedback = true
    @Published var showDebugInfo = false

    @Published var cameraConfig = HandsRecognizingConfig.defaultConfig
    @Published var modelConfig = GestureModelConfig.defaultConfig

    // MARK: - In-view threshold

    private static let minInViewDurationKey = "minInViewDuration"
    private static let isThresholdLockedKey = "isThresholdLocked"

    /// Minimum seconds the hand must be visible within a capture window for the
    /// resulting HandFilm to be accepted as a training example.
    /// Defaults to 1.2s; locked after the first successful training job.
    @Published var minInViewDuration: TimeInterval {
        didSet { UserDefaults.standard.set(minInViewDuration, forKey: Self.minInViewDurationKey) }
    }

    /// Once `true`, `minInViewDuration` cannot be changed from the UI.
    /// Locked when the first `POST /train` succeeds.
    @Published var isThresholdLocked: Bool {
        didSet { UserDefaults.standard.set(isThresholdLocked, forKey: Self.isThresholdLockedKey) }
    }

    init() {
        let stored = UserDefaults.standard.double(forKey: Self.minInViewDurationKey)
        minInViewDuration = stored > 0 ? stored : 1.2
        isThresholdLocked = UserDefaults.standard.bool(forKey: Self.isThresholdLockedKey)
    }

    /// Call after the first training job fires to permanently lock the threshold.
    func lockThresholdIfNeeded() {
        guard !isThresholdLocked else { return }
        isThresholdLocked = true
    }

    func updateCameraConfig() {
        cameraConfig = HandsRecognizingConfig(
            cameraIndex: preferredCamera,
            targetFPS: targetFPS,
            detectBothHands: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        )
    }

    func updateModelConfig() {
        let tfliteURL = defaultTFLiteModelURL()
        let modelPath = FileManager.default.fileExists(atPath: tfliteURL.path)
            ? tfliteURL.path
            : nil
        modelConfig = GestureModelConfig(
            modelPath: modelPath,
            backendType: modelPath != nil ? .tensorFlow : .mock,
            predictionThreshold: confidenceThreshold,
            maxPredictions: 5
        )
    }

    func defaultTFLiteModelURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("GestureModel/gesture_model.tflite")
    }
}

import SwiftUI
import Combine
import HandGestureRecognizingFramework
import GestureModelModule

enum BalanceStrategy: String, CaseIterable, Identifiable {
    case classWeight = "class_weight"
    case jitter
    case none

    var id: String { rawValue }

    var title: String {
        switch self {
        case .classWeight: return "Class weight"
        case .jitter:      return "Jitter oversample"
        case .none:        return "None"
        }
    }

    var caption: String {
        switch self {
        case .classWeight: return "Weight loss by inverse class frequency. Safe default."
        case .jitter:      return "Oversample minority classes with noisy copies (train only)."
        case .none:        return "No balancing — baseline for comparison."
        }
    }
}

class AppSettings: ObservableObject {
    @Published var colorScheme: ColorScheme? = nil
    @Published var preferredCamera: Int = 0
    @Published var targetFPS: Int = 30
    @Published var confidenceThreshold: Float = 0.7
    @Published var enableHapticFeedback = true
    @Published var showDebugInfo = false

    // MARK: - Diagnostics

    private static let enhancedPredictionModeKey = "enhancedPredictionMode"
    private static let bypassPhase2FilterKey = "bypassPhase2Filter"

    /// Shows per-phase telemetry overlay and the bypass toggle on the Camera screen.
    @Published var enhancedPredictionMode: Bool {
        didSet { UserDefaults.standard.set(enhancedPredictionMode, forKey: Self.enhancedPredictionModeKey) }
    }

    /// When true, Phase 3 runs unrestricted (ignores the Phase 2 candidate set).
    /// Predictions are not uploaded to the server. For diagnostics only.
    @Published var bypassPhase2Filter: Bool {
        didSet { UserDefaults.standard.set(bypassPhase2Filter, forKey: Self.bypassPhase2FilterKey) }
    }

    @Published var cameraConfig = HandsRecognizingConfig.defaultConfig
    @Published var modelConfig = GestureModelConfig.defaultConfig

    // MARK: - In-view threshold

    private static let minInViewDurationKey = "minInViewDuration"
    private static let isThresholdLockedKey = "isThresholdLocked"
    private static let balanceStrategyKey = "balanceStrategy"

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

    /// Strategy the server uses to counter class-imbalance during training.
    /// Sent with every `POST /train`.
    @Published var balanceStrategy: BalanceStrategy {
        didSet { UserDefaults.standard.set(balanceStrategy.rawValue, forKey: Self.balanceStrategyKey) }
    }

    init() {
        let stored = UserDefaults.standard.double(forKey: Self.minInViewDurationKey)
        minInViewDuration = stored > 0 ? stored : 1.2
        isThresholdLocked = UserDefaults.standard.bool(forKey: Self.isThresholdLockedKey)
        let storedStrategy = UserDefaults.standard.string(forKey: Self.balanceStrategyKey) ?? ""
        balanceStrategy = BalanceStrategy(rawValue: storedStrategy) ?? .classWeight
        enhancedPredictionMode = UserDefaults.standard.bool(forKey: Self.enhancedPredictionModeKey)
        bypassPhase2Filter = UserDefaults.standard.bool(forKey: Self.bypassPhase2FilterKey)
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

    func defaultGestureIdsURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("GestureModel/gesture_ids.json")
    }

    func defaultPreprocessorURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("GestureModel/preprocessor.js")
    }

    func defaultPoseModelURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("GestureModel/pose_model.tflite")
    }

    func defaultPoseManifestURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("GestureModel/pose_manifest.json")
    }
}

import SwiftUI
import Combine
import CameraGestures
import CameraGestures

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
    private static let geomCoefKey = "geomCoef"

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

    /// Multiplier applied to the 20 geometric extras (distances + angles) in the
    /// preprocessor before the model sees them. Sent with `POST /train` and
    /// `POST /train/pose`. The server bakes the value into the served preprocessor.js,
    /// so iOS inference picks it up automatically on next model download.
    @Published var geomCoef: Double {
        didSet { UserDefaults.standard.set(geomCoef, forKey: Self.geomCoefKey) }
    }

    // MARK: - Device model timestamps

    private static let gestureModelLoadedAtKey = "gestureModelLoadedAt"
    private static let poseModelLoadedAtKey = "poseModelLoadedAt"

    /// When the Phase 3 gesture model was last downloaded and loaded onto this device.
    @Published var gestureModelLoadedAt: Date? {
        didSet {
            if let date = gestureModelLoadedAt {
                UserDefaults.standard.set(date.timeIntervalSince1970, forKey: Self.gestureModelLoadedAtKey)
            } else {
                UserDefaults.standard.removeObject(forKey: Self.gestureModelLoadedAtKey)
            }
        }
    }

    /// When the Phase 2 pose model was last downloaded and loaded onto this device.
    @Published var poseModelLoadedAt: Date? {
        didSet {
            if let date = poseModelLoadedAt {
                UserDefaults.standard.set(date.timeIntervalSince1970, forKey: Self.poseModelLoadedAtKey)
            } else {
                UserDefaults.standard.removeObject(forKey: Self.poseModelLoadedAtKey)
            }
        }
    }

    init() {
        let stored = UserDefaults.standard.double(forKey: Self.minInViewDurationKey)
        minInViewDuration = stored > 0 ? stored : 1.2
        isThresholdLocked = UserDefaults.standard.bool(forKey: Self.isThresholdLockedKey)
        let storedStrategy = UserDefaults.standard.string(forKey: Self.balanceStrategyKey) ?? ""
        balanceStrategy = BalanceStrategy(rawValue: storedStrategy) ?? .classWeight
        let storedCoef = UserDefaults.standard.double(forKey: Self.geomCoefKey)
        geomCoef = storedCoef > 0 ? storedCoef : 1.0
        enhancedPredictionMode = UserDefaults.standard.bool(forKey: Self.enhancedPredictionModeKey)
        bypassPhase2Filter = UserDefaults.standard.bool(forKey: Self.bypassPhase2FilterKey)
        let gestureTS = UserDefaults.standard.double(forKey: Self.gestureModelLoadedAtKey)
        gestureModelLoadedAt = gestureTS > 0 ? Date(timeIntervalSince1970: gestureTS) : nil
        let poseTS = UserDefaults.standard.double(forKey: Self.poseModelLoadedAtKey)
        poseModelLoadedAt = poseTS > 0 ? Date(timeIntervalSince1970: poseTS) : nil
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

    // MARK: - Effective device model dates

    /// Explicit stamp when set; falls back to the .tflite file's modification date.
    var effectiveGestureModelLoadedAt: Date? {
        if let d = gestureModelLoadedAt { return d }
        let url = defaultTFLiteModelURL()
        return (try? FileManager.default.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
    }

    /// Explicit stamp when set; falls back to the pose .tflite file's modification date.
    var effectivePoseModelLoadedAt: Date? {
        if let d = poseModelLoadedAt { return d }
        let url = defaultPoseModelURL()
        return (try? FileManager.default.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
    }

    func updateModelConfig() {
        let tfliteURL = defaultTFLiteModelURL()
        let modelPath = FileManager.default.fileExists(atPath: tfliteURL.path)
            ? tfliteURL.path
            : nil
        modelConfig = GestureModelConfig(
            modelPath: modelPath,
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

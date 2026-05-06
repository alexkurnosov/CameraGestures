import Foundation
import Combine
import HandGestureRecognizingFramework
import HandGestureTypes

@MainActor
class GestureRecognizerWrapper: ObservableObject {
    let recognizer: HandGestureRecognizing

    // Combine publishers — callbacks are set ONCE; multiple views subscribe
    let gestureDetected = PassthroughSubject<DetectedGesture, Never>()
    let handTrackingUpdate = PassthroughSubject<HandShot, Never>()
    let statusChanged = PassthroughSubject<GestureRecognizingStatus, Never>()

    @Published var isRecognizing: Bool = false
    @Published var currentGesture: String?
    @Published var confidence: Float = 0.0
    @Published var lastError: String?

    // Phase 1 gate state (updated via motionGateUpdateCallback)
    @Published var motionGateState: MotionGateState = .closed
    @Published var gateBufferCount: Int = 0
    var gateBufferCap: Int { recognizer.getConfig().gestureBufferSize }

    // Phase 2 telemetry (updated via holdsModeTelemetryCallback)
    @Published var holdsTelemetry: HoldsTelemetry = HoldsTelemetry()

    /// Version strings for the currently loaded models (used in confidence-log entries).
    /// Set by the training view when a new model is downloaded.
    var handfilmModelVersion: String = ""
    var poseModelVersion: String = ""

    init(recognizer: HandGestureRecognizing) {
        self.recognizer = recognizer
    }

    /// Wire up recognizer callbacks once. Each callback publishes to the
    /// corresponding Combine subject so multiple subscribers can react.
    func setupCallbacks() {
        recognizer.gestureDetectionCallback = { [weak self] gesture in
            DispatchQueue.main.async { self?.gestureDetected.send(gesture) }
        }
        recognizer.handTrackingUpdateCallback = { [weak self] handshot in
            DispatchQueue.main.async { self?.handTrackingUpdate.send(handshot) }
        }
        recognizer.statusChangeCallback = { [weak self] status in
            DispatchQueue.main.async { self?.statusChanged.send(status) }
        }
        recognizer.motionGateUpdateCallback = { [weak self] state, count in
            DispatchQueue.main.async {
                self?.motionGateState = state
                self?.gateBufferCount = count
            }
        }
        recognizer.holdsModeTelemetryCallback = { [weak self] telemetry in
            DispatchQueue.main.async {
                self?.holdsTelemetry = telemetry
            }
        }
    }

    /// Build a config and initialize the underlying recognizer.
    func initialize(appSettings: AppSettings) async throws {
        setupCallbacks()

        let config = HandGestureRecognizingConfig(
            handsRecognizingConfig: appSettings.cameraConfig,
            gestureModelConfig: appSettings.modelConfig,
            enableRealTimeProcessing: true,
            gestureBufferSize: 30,
            confidenceThreshold: appSettings.confidenceThreshold,
            motionGateConfig: .defaultConfig,
            holdsConfig: .defaultConfig
        )
        try await recognizer.initialize(config: config)
        loadPoseModelIfAvailable(appSettings: appSettings)
    }

    /// Loads pose_model.tflite + pose_manifest.json if both exist on disk.
    func loadPoseModelIfAvailable(appSettings: AppSettings) {
        let tflite = appSettings.defaultPoseModelURL()
        let manifest = appSettings.defaultPoseManifestURL()
        guard FileManager.default.fileExists(atPath: tflite.path),
              FileManager.default.fileExists(atPath: manifest.path) else { return }
        do {
            try recognizer.loadPoseModel(tflitePath: tflite.path, manifestPath: manifest.path)
        } catch {
            print("[GestureRecognizerWrapper] Failed to load pose model: \(error)")
        }
    }
}

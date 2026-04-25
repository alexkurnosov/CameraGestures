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
    }

    /// Build a config and initialize the underlying recognizer.
    func initialize(appSettings: AppSettings) async throws {
        setupCallbacks()

        let config = HandGestureRecognizingConfig(
            handsRecognizingConfig: appSettings.cameraConfig,
            gestureModelConfig: appSettings.modelConfig,
            enableRealTimeProcessing: true,
            gestureBufferSize: 30,
            confidenceThreshold: appSettings.confidenceThreshold
        )
        try await recognizer.initialize(config: config)
    }
}

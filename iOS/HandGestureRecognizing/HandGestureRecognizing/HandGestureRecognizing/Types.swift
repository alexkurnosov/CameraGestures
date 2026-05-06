import Foundation
@_exported import HandsRecognizingModule
import GestureModelModule
import HandGestureTypes

// MARK: - Configuration

/// Configuration for HandGestureRecognizing system
public struct HandGestureRecognizingConfig {
    public let handsRecognizingConfig: HandsRecognizingConfig
    public let gestureModelConfig: GestureModelConfig
    public let enableRealTimeProcessing: Bool
    public let gestureBufferSize: Int
    public let confidenceThreshold: Float
    /// When non-nil, `gateEnabled` on the recognizer activates Phase 1 motion gate.
    public let motionGateConfig: MotionGateConfig?
    /// When non-nil, enables Phase 2 hold detection and prefix matching in Holds mode.
    public let holdsConfig: HoldsConfig?

    public init(
        handsRecognizingConfig: HandsRecognizingConfig = .defaultConfig,
        gestureModelConfig: GestureModelConfig = .defaultConfig,
        enableRealTimeProcessing: Bool = true,
        gestureBufferSize: Int = 30,
        confidenceThreshold: Float = 0.7,
        motionGateConfig: MotionGateConfig? = nil,
        holdsConfig: HoldsConfig? = nil
    ) {
        self.handsRecognizingConfig = handsRecognizingConfig
        self.gestureModelConfig = gestureModelConfig
        self.enableRealTimeProcessing = enableRealTimeProcessing
        self.gestureBufferSize = gestureBufferSize
        self.confidenceThreshold = confidenceThreshold
        self.motionGateConfig = motionGateConfig
        self.holdsConfig = holdsConfig
    }

    public static let defaultConfig = HandGestureRecognizingConfig()
}

// MARK: - Holds Config (Phase 2 parameters)

/// Runtime parameters for Phase 2 hold detection and prefix matching.
/// Values calibrated from the 166-film corpus (2026-04-30).
public struct HoldsConfig {
    /// Smoothed energy threshold below which a run is counted as a hold.
    public let tHold: Float
    /// Minimum hold duration in milliseconds (≈100 ms = 3 frames at 30 fps).
    public let kHoldMs: TimeInterval
    /// Smoothing window duration in milliseconds for energy (≈100 ms).
    public let smoothKMs: TimeInterval
    /// Post-hold wait for a longer prefix before committing (milliseconds).
    public let tCommitMs: TimeInterval
    /// Minimum buffer duration since gate-open before any commit is allowed (milliseconds).
    public let tMinBufferMs: TimeInterval
    /// Minimum pose-classifier confidence; holds below this are rejected.
    public let tauPoseConfidence: Float
    /// Phase 3 confidence threshold for the masked-argmax output.
    public let tauPhase3Confidence: Float

    public init(
        tHold: Float = 2.10,
        kHoldMs: TimeInterval = 100,
        smoothKMs: TimeInterval = 100,
        tCommitMs: TimeInterval = 300,
        tMinBufferMs: TimeInterval = 200,
        tauPoseConfidence: Float = 0.6,
        tauPhase3Confidence: Float = 0.7
    ) {
        self.tHold = tHold
        self.kHoldMs = kHoldMs
        self.smoothKMs = smoothKMs
        self.tCommitMs = tCommitMs
        self.tMinBufferMs = tMinBufferMs
        self.tauPoseConfidence = tauPoseConfidence
        self.tauPhase3Confidence = tauPhase3Confidence
    }

    public static let defaultConfig = HoldsConfig()
}

// MARK: - Holds Mode Telemetry

/// Snapshot of Phase 2 state for the Holds-mode overlay in CameraView.
public struct HoldsTelemetry {
    /// Most recently detected hold's pose prediction (nil before first hold).
    public let lastPoseId: Int?
    public let lastPoseLabel: String?
    public let lastPoseConfidence: Float?
    public let lastPoseKind: String?
    /// Current observed sequence (list of pose ids since gate opened).
    public let observedSequence: [Int]
    /// Gesture id of the current matched template, or nil if no match.
    public let matchedGesture: String?

    public init(
        lastPoseId: Int? = nil,
        lastPoseLabel: String? = nil,
        lastPoseConfidence: Float? = nil,
        lastPoseKind: String? = nil,
        observedSequence: [Int] = [],
        matchedGesture: String? = nil
    ) {
        self.lastPoseId = lastPoseId
        self.lastPoseLabel = lastPoseLabel
        self.lastPoseConfidence = lastPoseConfidence
        self.lastPoseKind = lastPoseKind
        self.observedSequence = observedSequence
        self.matchedGesture = matchedGesture
    }
}

/// Callback fired on the main thread whenever Phase 2 processes a hold in Holds mode.
public typealias HoldsModeTelemetryCallback = (HoldsTelemetry) -> Void

// MARK: - Motion Gate

/// Parameters for the Phase 1 hysteresis motion gate.
/// tOpen/tClose are empirically tuned starting values (2026-05-02); replace with corpus-calibrated
/// seeds once idle-capture films are available for Stage 1 re-calibration.
public struct MotionGateConfig {
    /// Energy threshold to open the gate (sum of per-landmark L2 deltas in wrist-relative, scale-normalised coords).
    public let tOpen: Float
    /// Gate opens after energy exceeds tOpen for this many milliseconds (≈1 frame at 30 fps).
    public let kOpenMs: TimeInterval
    /// Energy threshold to close the gate (fallback; absent-hand is the primary close path).
    public let tClose: Float
    /// Gate closes after energy stays below tClose for this many milliseconds (≈30 frames at 30 fps).
    public let kCloseMs: TimeInterval
    /// Duration of post-cycle cooldown in milliseconds. Emission is suppressed; most-recent commit queued.
    public let cooldownMs: TimeInterval

    public init(
        tOpen: Float = 1,
        kOpenMs: TimeInterval = 33,
        tClose: Float = 0.5,
        kCloseMs: TimeInterval = 200,
        cooldownMs: TimeInterval = 1000
    ) {
        self.tOpen = tOpen
        self.kOpenMs = kOpenMs
        self.tClose = tClose
        self.kCloseMs = kCloseMs
        self.cooldownMs = cooldownMs
    }

    public static let defaultConfig = MotionGateConfig()
}

/// Current state of the Phase 1 motion gate.
public enum MotionGateState: Equatable {
    case closed
    case open

    public var displayName: String {
        switch self {
        case .closed: return "Closed"
        case .open:   return "Open"
        }
    }
}

/// Callback fired on the main thread whenever the gate state or gate-buffer count changes.
/// Arguments: (state, bufferFrameCount)
public typealias MotionGateUpdateCallback = (MotionGateState, Int) -> Void

// MARK: - Gesture Detection Events

/// Detected gesture with context
public struct DetectedGesture {
    public let prediction: GesturePrediction
    public let handfilm: HandFilm
    public let handedness: LeftOrRight
    public let detectionTimestamp: TimeInterval
    public let processingLatency: TimeInterval
    /// Number of candidate gestures passed to Phase 3 masked-argmax (nil in unrestricted handfilm mode).
    public let candidateSetSize: Int?

    public init(
        prediction: GesturePrediction,
        handfilm: HandFilm,
        handedness: LeftOrRight,
        detectionTimestamp: TimeInterval = Date().timeIntervalSince1970,
        processingLatency: TimeInterval = 0.0,
        candidateSetSize: Int? = nil
    ) {
        self.prediction = prediction
        self.handfilm = handfilm
        self.handedness = handedness
        self.detectionTimestamp = detectionTimestamp
        self.processingLatency = processingLatency
        self.candidateSetSize = candidateSetSize
    }
}

/// Statistics for gesture recognition performance
public struct GestureRecognizingStats {
    public let totalGesturesDetected: Int
    public let averageProcessingLatency: TimeInterval
    public let averageConfidence: Float
    public let gesturesByType: [String: Int]
    public let uptime: TimeInterval
    public let fps: Float
    
    public init(
        totalGesturesDetected: Int = 0,
        averageProcessingLatency: TimeInterval = 0.0,
        averageConfidence: Float = 0.0,
        gesturesByType: [String: Int] = [:],
        uptime: TimeInterval = 0.0,
        fps: Float = 0.0
    ) {
        self.totalGesturesDetected = totalGesturesDetected
        self.averageProcessingLatency = averageProcessingLatency
        self.averageConfidence = averageConfidence
        self.gesturesByType = gesturesByType
        self.uptime = uptime
        self.fps = fps
    }
}

// MARK: - Error Types

/// Errors specific to HandGestureRecognizing
public enum HandGestureRecognizingError: Error {
    case notInitialized
    case alreadyRunning
    case notRunning
    case handsRecognizingError(Error)
    case gestureModelError(Error)
    case configurationError(String)
    case cameraPermissionDenied
    case processingError(String)
    
    public var localizedDescription: String {
        switch self {
        case .notInitialized:
            return "Hand gesture recognizing not initialized"
        case .alreadyRunning:
            return "Hand gesture recognizing already running"
        case .notRunning:
            return "Hand gesture recognizing not running"
        case .handsRecognizingError(let error):
            return "Hands recognizing error: \(error.localizedDescription)"
        case .gestureModelError(let error):
            return "Gesture model error: \(error.localizedDescription)"
        case .configurationError(let message):
            return "Configuration error: \(message)"
        case .cameraPermissionDenied:
            return "Camera permission denied"
        case .processingError(let message):
            return "Processing error: \(message)"
        }
    }
}

// MARK: - Callback Types

/// Callback for detected gestures
public typealias GestureDetectionCallback = (DetectedGesture) -> Void

/// Callback for real-time hand tracking updates
public typealias HandTrackingUpdateCallback = (HandShot) -> Void

/// Callback for system status changes
public typealias StatusChangeCallback = (GestureRecognizingStatus) -> Void


// MARK: - System Status

/// Current status of the gesture recognizing system
public enum GestureRecognizingStatus: Equatable {
    case idle
    case initializing
    case running
    case paused
    case error(String)
    case stopping
    
    public var isActive: Bool {
        switch self {
        case .running:
            return true
        default:
            return false
        }
    }
    
    public var displayName: String {
        switch self {
        case .idle:
            return "Idle"
        case .initializing:
            return "Initializing"
        case .running:
            return "Running"
        case .paused:
            return "Paused"
        case .error:
            return "Error"
        case .stopping:
            return "Stopping"
        }
    }
}

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
    
    public init(
        handsRecognizingConfig: HandsRecognizingConfig = .defaultConfig,
        gestureModelConfig: GestureModelConfig = .defaultConfig,
        enableRealTimeProcessing: Bool = true,
        gestureBufferSize: Int = 10,
        confidenceThreshold: Float = 0.7
    ) {
        self.handsRecognizingConfig = handsRecognizingConfig
        self.gestureModelConfig = gestureModelConfig
        self.enableRealTimeProcessing = enableRealTimeProcessing
        self.gestureBufferSize = gestureBufferSize
        self.confidenceThreshold = confidenceThreshold
    }
    
    public static let defaultConfig = HandGestureRecognizingConfig()
}

// MARK: - Gesture Detection Events

/// Detected gesture with context
public struct DetectedGesture {
    public let prediction: GesturePrediction
    public let handfilm: HandFilm
    public let handedness: LeftOrRight
    public let detectionTimestamp: TimeInterval
    public let processingLatency: TimeInterval
    
    public init(
        prediction: GesturePrediction,
        handfilm: HandFilm,
        handedness: LeftOrRight,
        detectionTimestamp: TimeInterval = Date().timeIntervalSince1970,
        processingLatency: TimeInterval = 0.0
    ) {
        self.prediction = prediction
        self.handfilm = handfilm
        self.handedness = handedness
        self.detectionTimestamp = detectionTimestamp
        self.processingLatency = processingLatency
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

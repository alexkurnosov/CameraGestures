import Foundation
import HandGestureTypes
import MediaPipeTasksVision

/// Error codes for framework operations
public enum HandsRecognizingError: Error {
    case cameraNotAvailable
    case invalidConfiguration
    case initializationFailed
    case processingError
    
    public var localizedDescription: String {
        switch self {
        case .cameraNotAvailable:
            return "Camera not available"
        case .invalidConfiguration:
            return "Invalid configuration"
        case .initializationFailed:
            return "Initialization failed"
        case .processingError:
            return "Processing error"
        }
    }
}

// MARK: - Configuration

/// Configuration for HandsRecognizing
public struct HandsRecognizingConfig {
    public let cameraIndex: Int
    public let targetFPS: Int
    public let detectBothHands: Bool
    public let minDetectionConfidence: Float
    public let minTrackingConfidence: Float
    public let handfilmMaxDuration: TimeInterval
    
    public init(
        cameraIndex: Int = 0,
        targetFPS: Int = 30,
        detectBothHands: Bool = true,
        minDetectionConfidence: Float = 0.5,
        minTrackingConfidence: Float = 0.5,
        handfilmMaxDuration: TimeInterval = 2.0
    ) {
        self.cameraIndex = cameraIndex
        self.targetFPS = targetFPS
        self.detectBothHands = detectBothHands
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence
        self.handfilmMaxDuration = handfilmMaxDuration
    }
    
    public static let defaultConfig = HandsRecognizingConfig()
    
    public func getHandLandmarkerOptions() -> HandLandmarkerOptions {
        let options = HandLandmarkerOptions()
        
        // Configure base options with model file
        let baseOptions = BaseOptions()
        
        // Load model from vendored MediaPipeModel framework
        let frameworkBundle = Bundle(for: HandsRecognizing.self)
        if let modelPath = frameworkBundle.path(forResource: "hand_landmarker", ofType: "task") {
            baseOptions.modelAssetPath = modelPath
        }
        options.baseOptions = baseOptions
        
        // Configure number of hands to detect
        options.numHands = detectBothHands ? 2 : 1
        
        // Configure confidence thresholds
        options.minHandDetectionConfidence = minDetectionConfidence
        options.minTrackingConfidence = minTrackingConfidence
        options.minHandPresenceConfidence = minDetectionConfidence
        
        // Set running mode to live stream for real-time processing
        options.runningMode = .liveStream
        
        return options
    }
}

// MARK: - Callback Types

/// Callback for individual handshot detection
public typealias HandShotCallback = (HandShot) -> Void

/// Callback for completed handfilm sequences
public typealias HandFilmCallback = (HandFilm) -> Void

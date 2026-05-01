//
//  HandGestureTypes.swift
//  HandGestureTypes
//
//  Created by Алексей Курносов on 27.01.2026.
//

import Foundation

// MARK: - Core Data Types

/// 3D point coordinates
public struct Point3D: Equatable {
    public let x: Float
    public let y: Float
    public let z: Float

    public init(x: Float, y: Float, z: Float) {
        self.x = x
        self.y = y
        self.z = z
    }
}

/// Single frame of hand landmarks.
/// When `isAbsent` is `true` the hand was not detected in this frame;
/// `landmarks` will be 21 zero-valued points used as a placeholder.
public struct HandShot: Equatable {
    public let landmarks: [Point3D]
    public let timestamp: TimeInterval
    public let leftOrRight: LeftOrRight
    /// `true` when no hand was detected in this frame (all landmarks are zero).
    public let isAbsent: Bool

    public init(
        landmarks: [Point3D],
        timestamp: TimeInterval,
        leftOrRight: LeftOrRight,
        isAbsent: Bool = false
    ) {
        self.landmarks = landmarks
        self.timestamp = timestamp
        self.leftOrRight = leftOrRight
        self.isAbsent = isAbsent
    }

    /// A placeholder HandShot representing a frame where no hand was visible.
    public static func absent(timestamp: TimeInterval) -> HandShot {
        let zeroPoints = (0..<21).map { _ in Point3D(x: 0, y: 0, z: 0) }
        return HandShot(landmarks: zeroPoints, timestamp: timestamp, leftOrRight: .unknown, isAbsent: true)
    }
}

/// Sequence of handshots representing a gesture.
/// Frames where the hand left the camera view are stored with `isAbsent = true`
/// so the temporal structure of the session is preserved.
public struct HandFilm {
    public var frames: [HandShot]
    public let startTime: TimeInterval

    public var endTime: TimeInterval {
        frames.last?.timestamp ?? startTime
    }

    /// Total wall-clock length of the session (first frame → last frame).
    public var gestureDuration: TimeInterval {
        endTime - startTime
    }

    /// Backward-compatible alias for `gestureDuration`.
    public var duration: TimeInterval { gestureDuration }

    /// Total time the hand was actually visible (sum of inter-frame intervals
    /// between consecutive non-absent frames).
    public var inViewDuration: TimeInterval {
        let visible = frames.filter { !$0.isAbsent }
        guard visible.count >= 2 else {
            return visible.isEmpty ? 0 : 0
        }
        var total: TimeInterval = 0
        for i in 1..<visible.count {
            total += visible[i].timestamp - visible[i - 1].timestamp
        }
        return total
    }

    /// Number of frames where the hand was detected.
    public var inViewFrameCount: Int {
        frames.filter { !$0.isAbsent }.count
    }

    public init(startTime: TimeInterval = Date().timeIntervalSince1970) {
        self.frames = []
        self.startTime = startTime
    }

    public mutating func addFrame(_ handshot: HandShot) {
        frames.append(handshot)
    }

    public mutating func clear() {
        frames.removeAll()
    }
}

/// Hand identification (left or right)
public enum LeftOrRight: Equatable {
    case left
    case right
    case unknown
}

/// Represents a recognized gesture with confidence score
public struct GesturePrediction {
    public let gestureId: String
    public let gestureName: String
    public let confidence: Float
    public let timestamp: TimeInterval
    
    public init(gestureId: String, gestureName: String, confidence: Float, timestamp: TimeInterval = Date().timeIntervalSince1970) {
        self.gestureId = gestureId
        self.gestureName = gestureName
        self.confidence = confidence
        self.timestamp = timestamp
    }
}


// MARK: - Gesture Types

/// A user-defined gesture with a name and description
public struct GestureDefinition: Codable, Identifiable, Equatable {
    /// Slug identifier derived from the name (e.g. "thumbs_up")
    public let id: String
    /// Human-readable display name (e.g. "Thumbs Up")
    public let name: String
    /// Description of how to perform this gesture
    public let description: String

    public init(id: String, name: String, description: String) {
        self.id = id
        self.name = name
        self.description = description
    }
}

// MARK: - Training Data

/// Training example for gesture recognition
public struct TrainingExample: Identifiable {
    public let id: UUID
    public let handfilm: HandFilm
    /// ID of the gesture being demonstrated (matches `GestureDefinition.id`)
    public var gestureId: String
    public let userId: String?
    public let sessionId: String
    public let timestamp: TimeInterval
    
    public init(id: UUID = UUID(), handfilm: HandFilm, gestureId: String, userId: String? = nil, sessionId: String, timestamp: TimeInterval = Date().timeIntervalSince1970) {
        self.id = id
        self.handfilm = handfilm
        self.gestureId = gestureId
        self.userId = userId
        self.sessionId = sessionId
        self.timestamp = timestamp
    }
}

/// Collection of training examples
public struct TrainingDataset {
    public var examples: [TrainingExample]
    public let name: String
    public let createdAt: TimeInterval
    
    public init(name: String) {
        self.examples = []
        self.name = name
        self.createdAt = Date().timeIntervalSince1970
    }
    
    public mutating func addExample(_ example: TrainingExample) {
        examples.append(example)
    }

    public mutating func removeExample(id: UUID) {
        examples.removeAll { $0.id == id }
    }

    public mutating func relabelExample(id: UUID, newGestureId: String) {
        if let idx = examples.firstIndex(where: { $0.id == id }) {
            let old = examples[idx]
            examples[idx] = TrainingExample(
                id: old.id,
                handfilm: old.handfilm,
                gestureId: newGestureId,
                userId: old.userId,
                sessionId: old.sessionId
            )
        }
    }

    /// Number of examples recorded per gesture ID
    public var gestureCount: [String: Int] {
        return Dictionary(grouping: examples) { $0.gestureId }
            .mapValues { $0.count }
    }
}

// MARK: - HandFilm Validation

/// Why a captured HandFilm was rejected as a training example.
///
/// Stored as a raw string so older clients can deserialise values introduced by
/// newer app versions without crashing — unknown strings become `.unknown(rawValue)`.
public enum HandFilmFailureReason: Equatable {
    case insufficientInViewDuration
    case unknown(String)

    public var rawValue: String {
        switch self {
        case .insufficientInViewDuration: return "insufficientInViewDuration"
        case .unknown(let s):             return s
        }
    }

    public init(rawValue: String) {
        switch rawValue {
        case "insufficientInViewDuration": self = .insufficientInViewDuration
        default:                           self = .unknown(rawValue)
        }
    }

    /// Human-readable description shown in the UI.
    public var displayName: String {
        switch self {
        case .insufficientInViewDuration: return "Hand not visible long enough"
        case .unknown(let s):             return "Unknown reason (\(s))"
        }
    }
}

extension HandFilmFailureReason: Codable {
    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self.init(rawValue: raw)
    }
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

/// A HandFilm that did not meet the quality threshold for training.
/// Stored locally only; never uploaded to the server.
public struct FailedHandFilm: Identifiable {
    public let id: UUID
    public let handfilm: HandFilm
    public let gestureId: String
    public let failureReason: HandFilmFailureReason
    /// Human-readable detail, e.g. "0.4s in-view, need ≥1.2s"
    public let failureDetail: String
    public let timestamp: TimeInterval
    /// `true` when the user has manually overridden the failure and promoted
    /// this film to a valid training example.
    public var isManuallyValidated: Bool

    public init(
        id: UUID = UUID(),
        handfilm: HandFilm,
        gestureId: String,
        failureReason: HandFilmFailureReason,
        failureDetail: String,
        isManuallyValidated: Bool = false
    ) {
        self.id = id
        self.handfilm = handfilm
        self.gestureId = gestureId
        self.failureReason = failureReason
        self.failureDetail = failureDetail
        self.timestamp = Date().timeIntervalSince1970
        self.isManuallyValidated = isManuallyValidated
    }
}

// MARK: - Model Statistics

/// Performance metrics for gesture model
public struct ModelMetrics {
    public let accuracy: Float
    public let precision: Float
    public let recall: Float
    public let f1Score: Float
    public let confusionMatrix: [[Int]]
    public let trainingTime: TimeInterval
    public let validationTime: TimeInterval
    
    public init(
        accuracy: Float,
        precision: Float,
        recall: Float,
        f1Score: Float,
        confusionMatrix: [[Int]],
        trainingTime: TimeInterval,
        validationTime: TimeInterval
    ) {
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1Score = f1Score
        self.confusionMatrix = confusionMatrix
        self.trainingTime = trainingTime
        self.validationTime = validationTime
    }
}

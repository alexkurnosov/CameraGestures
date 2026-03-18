//
//  HandGestureTypes.swift
//  HandGestureTypes
//
//  Created by Алексей Курносов on 27.01.2026.
//

import Foundation

// MARK: - Core Data Types

/// 3D point coordinates
public struct Point3D {
    public let x: Float
    public let y: Float
    public let z: Float
    
    public init(x: Float, y: Float, z: Float) {
        self.x = x
        self.y = y
        self.z = z
    }
}

/// Single frame of hand landmarks
public struct HandShot {
    public let landmarks: [Point3D]
    public let timestamp: TimeInterval
    public let leftOrRight: LeftOrRight
    
    public init(landmarks: [Point3D], timestamp: TimeInterval, leftOrRight: LeftOrRight) {
        self.landmarks = landmarks
        self.timestamp = timestamp
        self.leftOrRight = leftOrRight
    }
}

/// Sequence of handshots representing a gesture
public struct HandFilm {
    public var frames: [HandShot]
    public let startTime: TimeInterval
    public var endTime: TimeInterval {
        frames.last?.timestamp ?? startTime
    }
    
    public var duration: TimeInterval {
        endTime - startTime
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
public enum LeftOrRight {
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
    
    public init(id: UUID = UUID(), handfilm: HandFilm, gestureId: String, userId: String? = nil, sessionId: String) {
        self.id = id
        self.handfilm = handfilm
        self.gestureId = gestureId
        self.userId = userId
        self.sessionId = sessionId
        self.timestamp = Date().timeIntervalSince1970
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

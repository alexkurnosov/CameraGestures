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

/// Known gesture types
public enum GestureType: String, CaseIterable {
    case openHand = "open_hand"
    case closedFist = "closed_fist"
    case pointing = "pointing"
    case peace = "peace"
    case wave = "wave"
    case grab = "grab"
    case swipeLeft = "swipe_left"
    case swipeRight = "swipe_right"
    case thumbsUp = "thumbs_up"
    case thumbsDown = "thumbs_down"
    
    public var displayName: String {
        switch self {
        case .openHand: return "Open Hand"
        case .closedFist: return "Closed Fist"
        case .pointing: return "Pointing"
        case .peace: return "Peace Sign"
        case .wave: return "Wave"
        case .grab: return "Grab"
        case .swipeLeft: return "Swipe Left"
        case .swipeRight: return "Swipe Right"
        case .thumbsUp: return "Thumbs Up"
        case .thumbsDown: return "Thumbs Down"
        }
    }
}

// MARK: - Training Data

/// Training example for gesture recognition
public struct TrainingExample {
    public let handfilm: HandFilm
    public let gestureType: GestureType
    public let userId: String?
    public let sessionId: String
    public let timestamp: TimeInterval
    
    public init(handfilm: HandFilm, gestureType: GestureType, userId: String? = nil, sessionId: String) {
        self.handfilm = handfilm
        self.gestureType = gestureType
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
    
    public var gestureCount: [GestureType: Int] {
        return Dictionary(grouping: examples) { $0.gestureType }
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

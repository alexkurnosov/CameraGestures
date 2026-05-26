// Swift surface for HandGestureTypes — matches the V1 HandGestureTypes pod API exactly
// so Training App v2 compiles without source changes after the pod swap.
//
// All value types (Point3D, HandShot, HandFilm, etc.) are pure Swift structs;
// they mirror the C ABI but don't wrap handles — they are converted to/from
// C ABI types when crossing the boundary to lower modules.
//
// GestureRegistry wraps cg_registry_ref.

import Foundation
import Combine

// MARK: - Core value types

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

public enum LeftOrRight: Equatable {
    case left
    case right
    case unknown
}

public struct HandShot: Equatable {
    public let landmarks: [Point3D]
    public let timestamp: TimeInterval
    public let leftOrRight: LeftOrRight
    public let isAbsent: Bool

    public init(landmarks: [Point3D], timestamp: TimeInterval,
                leftOrRight: LeftOrRight, isAbsent: Bool = false) {
        self.landmarks   = landmarks
        self.timestamp   = timestamp
        self.leftOrRight = leftOrRight
        self.isAbsent    = isAbsent
    }

    public static func absent(timestamp: TimeInterval) -> HandShot {
        let zeros = (0..<21).map { _ in Point3D(x: 0, y: 0, z: 0) }
        return HandShot(landmarks: zeros, timestamp: timestamp,
                        leftOrRight: .unknown, isAbsent: true)
    }
}

public struct HandFilm {
    public var frames: [HandShot]
    public let startTime: TimeInterval

    public var endTime: TimeInterval { frames.last?.timestamp ?? startTime }

    public var gestureDuration: TimeInterval { endTime - startTime }

    public var duration: TimeInterval { gestureDuration }

    public var inViewDuration: TimeInterval {
        let visible = frames.filter { !$0.isAbsent }
        guard visible.count >= 2 else { return 0 }
        var total: TimeInterval = 0
        for i in 1..<visible.count {
            total += visible[i].timestamp - visible[i - 1].timestamp
        }
        return total
    }

    public var inViewFrameCount: Int { frames.filter { !$0.isAbsent }.count }

    public init(startTime: TimeInterval = Date().timeIntervalSince1970) {
        self.frames    = []
        self.startTime = startTime
    }

    public mutating func addFrame(_ handshot: HandShot) { frames.append(handshot) }
    public mutating func clear() { frames.removeAll() }
}

public struct GesturePrediction {
    public let gestureId: String
    public let gestureName: String
    public let confidence: Float
    public let timestamp: TimeInterval

    public init(gestureId: String, gestureName: String, confidence: Float,
                timestamp: TimeInterval = Date().timeIntervalSince1970) {
        self.gestureId   = gestureId
        self.gestureName = gestureName
        self.confidence  = confidence
        self.timestamp   = timestamp
    }
}

// MARK: - Gesture definition

public struct GestureDefinition: Codable, Identifiable, Equatable {
    public let id: String
    public let name: String
    public let description: String

    public init(id: String, name: String, description: String) {
        self.id          = id
        self.name        = name
        self.description = description
    }
}

// MARK: - Training data  (pure Swift; not in C ABI)

public struct TrainingExample: Identifiable {
    public let id: UUID
    public let handfilm: HandFilm
    public var gestureId: String
    public let userId: String?
    public let sessionId: String
    public let timestamp: TimeInterval

    public init(id: UUID = UUID(), handfilm: HandFilm, gestureId: String,
                userId: String? = nil, sessionId: String,
                timestamp: TimeInterval = Date().timeIntervalSince1970) {
        self.id        = id
        self.handfilm  = handfilm
        self.gestureId = gestureId
        self.userId    = userId
        self.sessionId = sessionId
        self.timestamp = timestamp
    }
}

public struct TrainingDataset {
    public var examples: [TrainingExample]
    public let name: String
    public let createdAt: TimeInterval

    public init(name: String) {
        self.examples  = []
        self.name      = name
        self.createdAt = Date().timeIntervalSince1970
    }

    public mutating func addExample(_ example: TrainingExample) {
        examples.append(example)
    }

    public mutating func removeExample(id: UUID) {
        examples.removeAll { $0.id == id }
    }

    public mutating func relabelExample(id: UUID, newGestureId: String) {
        guard let idx = examples.firstIndex(where: { $0.id == id }) else { return }
        let old = examples[idx]
        examples[idx] = TrainingExample(id: old.id, handfilm: old.handfilm,
                                        gestureId: newGestureId,
                                        userId: old.userId,
                                        sessionId: old.sessionId)
    }

    public var gestureCount: [String: Int] {
        Dictionary(grouping: examples) { $0.gestureId }.mapValues { $0.count }
    }
}

// MARK: - HandFilm validation

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
        var c = encoder.singleValueContainer()
        try c.encode(rawValue)
    }
}

public struct FailedHandFilm: Identifiable {
    public let id: UUID
    public let handfilm: HandFilm
    public let gestureId: String
    public let failureReason: HandFilmFailureReason
    public let failureDetail: String
    public let timestamp: TimeInterval
    public var isManuallyValidated: Bool

    public init(id: UUID = UUID(), handfilm: HandFilm, gestureId: String,
                failureReason: HandFilmFailureReason, failureDetail: String,
                isManuallyValidated: Bool = false) {
        self.id                  = id
        self.handfilm            = handfilm
        self.gestureId           = gestureId
        self.failureReason       = failureReason
        self.failureDetail       = failureDetail
        self.timestamp           = Date().timeIntervalSince1970
        self.isManuallyValidated = isManuallyValidated
    }
}

// MARK: - Model metrics

public struct ModelMetrics {
    public let accuracy: Float
    public let precision: Float
    public let recall: Float
    public let f1Score: Float
    public let confusionMatrix: [[Int]]
    public let trainingTime: TimeInterval
    public let validationTime: TimeInterval

    public init(accuracy: Float, precision: Float, recall: Float, f1Score: Float,
                confusionMatrix: [[Int]], trainingTime: TimeInterval,
                validationTime: TimeInterval) {
        self.accuracy         = accuracy
        self.precision        = precision
        self.recall           = recall
        self.f1Score          = f1Score
        self.confusionMatrix  = confusionMatrix
        self.trainingTime     = trainingTime
        self.validationTime   = validationTime
    }
}

// MARK: - GestureRegistry (wraps C ABI cg_registry_ref)

import CameraGesturesC   // module map exposes the C ABI headers

public final class GestureRegistry: ObservableObject {

    @Published public private(set) var gestures: [GestureDefinition] = []

    private let ref: cg_registry_ref

    public init(storageURL: URL? = nil) {
        let url: URL
        if let given = storageURL {
            url = given
        } else {
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask).first!
            url = appSupport.appendingPathComponent("gestures.json")
        }
        ref = cg_registry_create(url.path)
        gestures = fetchAll()
    }

    deinit { cg_registry_destroy(ref) }

    // MARK: Public API

    @discardableResult
    public func addGesture(name: String, description: String) -> GestureDefinition? {
        var def = cg_gesture_definition()
        guard cg_registry_add(ref, name, description, &def) != 0 else { return nil }
        let result = GestureDefinition(id: string(def.id), name: string(def.name),
                                       description: string(def.description))
        gestures = fetchAll()
        return result
    }

    public func removeGesture(id: String) {
        cg_registry_remove(ref, id)
        gestures = fetchAll()
    }

    public static func slug(from name: String) -> String {
        name.lowercased()
            .replacingOccurrences(of: " ", with: "_")
            .filter { $0.isLetter || $0.isNumber || $0 == "_" }
    }

    // MARK: Private

    private func fetchAll() -> [GestureDefinition] {
        let count = cg_registry_count(ref)
        return (0..<count).compactMap { i in
            var def = cg_gesture_definition()
            guard cg_registry_get(ref, i, &def) != 0 else { return nil }
            return GestureDefinition(id: string(def.id), name: string(def.name),
                                     description: string(def.description))
        }
    }

    private func string<T>(_ tuple: T) -> String {
        withUnsafeBytes(of: tuple) { buf in
            let ptr = buf.baseAddress!.assumingMemoryBound(to: CChar.self)
            return String(cString: ptr)
        }
    }
}

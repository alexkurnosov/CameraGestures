import Foundation

// MARK: - Cluster Kind

public enum ClusterKind: String {
    case regular
    case idle
    case unconfirmed
}

// MARK: - Pose Manifest types (mirrors server pose_manifest.json)

public struct PoseCluster: Codable {
    public let label: String
    public let kind: String
    public let suspectedIdle: Bool
    public let nSamples: Int
    public let centroid: [Float]
    
    public init(
        label: String,
        kind: String,
        suspectedIdle: Bool,
        nSamples: Int,
        centroid: [Float]
    ) {
        self.label = label
        self.kind = kind
        self.suspectedIdle = suspectedIdle
        self.nSamples = nSamples
        self.centroid = centroid
    }

    enum CodingKeys: String, CodingKey {
        case label, kind
        case suspectedIdle = "suspected_idle"
        case nSamples = "n_samples"
        case centroid
    }
}

public struct PoseManifestParameters: Codable {
    public let tHold: Float?
    public let kHoldFrames: Int?
    public let smoothK: Int?
    public let edgeTrimFraction: Float?
    public let tCommitMs: Double?
    public let epsilon: Float?
    public let tauPoseConfidence: Float?

    enum CodingKeys: String, CodingKey {
        case tHold = "t_hold"
        case kHoldFrames = "k_hold_frames"
        case smoothK = "smooth_k"
        case edgeTrimFraction = "edge_trim_fraction"
        case tCommitMs = "t_commit_ms"
        case epsilon
        case tauPoseConfidence = "tau_pose_confidence"
    }
}

public struct PoseManifest: Codable {
    public let version: Int
    public let poseClusters: [String: PoseCluster]
    public let idlePoses: [Int]
    public let gestureTemplates: [String: [Int]]
    public let parameters: PoseManifestParameters?

    enum CodingKeys: String, CodingKey {
        case version
        case poseClusters = "pose_clusters"
        case idlePoses = "idle_poses"
        case gestureTemplates = "gesture_templates"
        case parameters
    }

    public func clusterKind(for id: Int) -> ClusterKind {
        guard let cluster = poseClusters[String(id)] else { return .unconfirmed }
        return ClusterKind(rawValue: cluster.kind) ?? .unconfirmed
    }

    public func clusterLabel(for id: Int) -> String {
        poseClusters[String(id)]?.label ?? "pose_\(id)"
    }
}

// MARK: - Pose Prediction

public struct PosePrediction {
    public let poseId: Int
    public let confidence: Float
    public let kind: ClusterKind
    public let clusterLabel: String

    public init(poseId: Int, confidence: Float, kind: ClusterKind, clusterLabel: String) {
        self.poseId = poseId
        self.confidence = confidence
        self.kind = kind
        self.clusterLabel = clusterLabel
    }
}

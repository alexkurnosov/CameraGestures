import Foundation
import Combine
import HandGestureTypes

// MARK: - Response Types

struct UploadExampleResponse: Codable {
    let id: String
    let totalForGesture: Int

    enum CodingKeys: String, CodingKey {
        case id
        case totalForGesture = "total_for_gesture"
    }
}

struct ExampleStatsResponse: Codable {
    struct GestureStat: Codable {
        let gestureId: String
        let count: Int

        enum CodingKeys: String, CodingKey {
            case gestureId = "gesture_id"
            case count
        }
    }
    let gestures: [GestureStat]
    let total: Int
}

struct TrainingJobResponse: Codable {
    let jobId: String
    let status: String

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case status
    }
}

struct ModelInfoResponse: Codable {
    let modelId: String
    let trainer: String
    let trainedOn: Int
    let trainedAt: TimeInterval
    let gestureIds: [String]
    let accuracy: Double?
    let f1: Double?
    let confusionMatrix: [[Int]]?
    let minInViewDuration: Double?

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case trainer
        case trainedOn = "trained_on"
        case trainedAt = "trained_at"
        case gestureIds = "gesture_ids"
        case accuracy
        case f1
        case confusionMatrix = "confusion_matrix"
        case minInViewDuration = "min_in_view_duration"
    }
}

struct ModelStatusResponse: Codable {
    let status: String          // "idle" | "training" | "ready" | "failed"
    let accuracy: Double?
    let trainedOn: Int
    let gestureIds: [String]
    let trainedAt: TimeInterval?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case status
        case accuracy
        case trainedOn = "trained_on"
        case gestureIds = "gesture_ids"
        case trainedAt = "trained_at"
        case error
    }
}

struct UpdateServerResponse: Codable {
    let status: String
}

// MARK: - Detailed Metrics (public /model/metrics endpoints)

struct PerClassMetric: Codable, Identifiable {
    let gestureId: String
    let precision: Double
    let recall: Double
    let f1: Double
    let supportVal: Int
    let supportTrain: Int

    var id: String { gestureId }

    enum CodingKeys: String, CodingKey {
        case gestureId = "gesture_id"
        case precision
        case recall
        case f1
        case supportVal = "support_val"
        case supportTrain = "support_train"
    }
}

struct ConfidenceByClass: Codable, Identifiable {
    let gestureId: String
    let count: Int
    let mean: Double?
    let p10: Double?
    let p50: Double?
    let p90: Double?

    var id: String { gestureId }

    enum CodingKeys: String, CodingKey {
        case gestureId = "gesture_id"
        case count, mean, p10, p50, p90
    }
}

struct ThresholdPoint: Codable, Identifiable {
    let threshold: Double
    let coverage: Double
    let precision: Double?
    let fires: Int

    var id: Double { threshold }
}

struct NoneAwareMetrics: Codable {
    let noneFalsePositiveRate: Double?
    let noneSupportVal: Int?
    let realAccuracy: Double?
    let realSupportVal: Int?

    enum CodingKeys: String, CodingKey {
        case noneFalsePositiveRate = "none_false_positive_rate"
        case noneSupportVal = "none_support_val"
        case realAccuracy = "real_accuracy"
        case realSupportVal = "real_support_val"
    }
}

struct AucMetrics: Codable {
    let rocAucMacro: Double?
    let prAucMacro: Double?

    enum CodingKeys: String, CodingKey {
        case rocAucMacro = "roc_auc_macro"
        case prAucMacro = "pr_auc_macro"
    }
}

struct ModelMetricsResponse: Codable, Identifiable {
    let modelId: String
    let trainer: String
    let trainedAt: TimeInterval
    let trainedOn: Int
    let gestureIds: [String]
    let balanceStrategy: String?
    let accuracy: Double?
    let f1Weighted: Double?
    let confusionMatrix: [[Int]]?
    let valSize: Int?
    let trainSize: Int?
    let perClass: [PerClassMetric]
    let noneAware: NoneAwareMetrics
    let confidenceByClass: [ConfidenceByClass]
    let thresholdCurves: [ThresholdPoint]
    let auc: AucMetrics
    let confidenceCurvePhase3: [ConfidenceCurvePoint]

    var id: String { modelId }

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case trainer
        case trainedAt = "trained_at"
        case trainedOn = "trained_on"
        case gestureIds = "gesture_ids"
        case balanceStrategy = "balance_strategy"
        case accuracy
        case f1Weighted = "f1_weighted"
        case confusionMatrix = "confusion_matrix"
        case valSize = "val_size"
        case trainSize = "train_size"
        case perClass = "per_class"
        case noneAware = "none_aware"
        case confidenceByClass = "confidence_by_class"
        case thresholdCurves = "threshold_curves"
        case auc
        case confidenceCurvePhase3 = "confidence_curve_phase3"
    }
}

struct ModelMetricsSummary: Codable, Identifiable {
    let modelId: String
    let trainedAt: TimeInterval
    let trainedOn: Int
    let accuracy: Double?
    let f1Weighted: Double?

    var id: String { modelId }

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case trainedAt = "trained_at"
        case trainedOn = "trained_on"
        case accuracy
        case f1Weighted = "f1_weighted"
    }
}

struct ModelMetricsListResponse: Codable {
    let models: [ModelMetricsSummary]
}

struct VersionResponse: Codable {
    let version: String
}

// MARK: - Sync Response Types

struct ServerExampleResponse: Codable {
    let id: String
    let gestureId: String
    let sessionId: String
    let userId: String?
    let handFilm: ServerHandFilmResponse
    let createdAt: Double

    enum CodingKeys: String, CodingKey {
        case id
        case gestureId = "gesture_id"
        case sessionId = "session_id"
        case userId = "user_id"
        case handFilm = "hand_film"
        case createdAt = "created_at"
    }
}

struct ServerHandFilmResponse: Codable {
    let frames: [ServerHandShotResponse]
    let startTime: Double

    enum CodingKeys: String, CodingKey {
        case frames
        case startTime = "start_time"
    }
}

struct ServerHandShotResponse: Codable {
    let landmarks: [ServerPoint3DResponse]
    let timestamp: Double
    let leftOrRight: String
    let isAbsent: Bool?

    enum CodingKeys: String, CodingKey {
        case landmarks
        case timestamp
        case leftOrRight = "left_or_right"
        case isAbsent = "is_absent"
    }
}

struct ServerPoint3DResponse: Codable {
    let x: Float
    let y: Float
    let z: Float
}

struct ExampleListResponse: Codable {
    let examples: [ServerExampleResponse]
    let total: Int
}

// MARK: - Stage 4: Pose model status

struct PoseTrainingJobResponse: Codable {
    let jobId: String
    let status: String  // "started" | "already_running"

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case status
    }
}

struct PoseReclusterSignals: Codable {
    let outOfEpsilonFraction: Double?
    let suggestRecluster: Bool
    let signalMessages: [String]

    enum CodingKeys: String, CodingKey {
        case outOfEpsilonFraction = "out_of_epsilon_fraction"
        case suggestRecluster = "suggest_recluster"
        case signalMessages = "signal_messages"
    }
}

struct PoseTrainingStatusResponse: Codable {
    let status: String
    let trainedOn: Int
    let nClusters: Int
    let trainedAt: TimeInterval?
    let error: String?
    let reclusterSignals: PoseReclusterSignals?

    enum CodingKeys: String, CodingKey {
        case status
        case trainedOn = "trained_on"
        case nClusters = "n_clusters"
        case trainedAt = "trained_at"
        case error
        case reclusterSignals = "recluster_signals"
    }
}

// MARK: - Stage 8: Phase 2 metrics

struct PosePerClassMetric: Codable, Identifiable {
    let clusterId: Int
    let precision: Double
    let recall: Double
    let f1: Double
    let supportVal: Int
    let supportTrain: Int

    var id: Int { clusterId }

    enum CodingKeys: String, CodingKey {
        case clusterId = "cluster_id"
        case precision, recall, f1
        case supportVal = "support_val"
        case supportTrain = "support_train"
    }
}

struct PoseLayer2Metrics: Codable {
    let perClass: [PosePerClassMetric]
    let confusionMatrix: [[Int]]
    let valSize: Int
    let trainSize: Int
    let nClusters: Int

    enum CodingKeys: String, CodingKey {
        case perClass = "per_class"
        case confusionMatrix = "confusion_matrix"
        case valSize = "val_size"
        case trainSize = "train_size"
        case nClusters = "n_clusters"
    }
}

struct PoseLengthDistEntry: Codable {
    let length: Int
    let count: Int
}

struct PosePerGestureMetric: Codable, Identifiable {
    let gestureId: String
    let nFilms: Int
    let commitRate: Double
    let commitCorrectRate: Double
    let precision: Double?
    let recall: Double?
    let f1: Double?

    var id: String { gestureId }

    enum CodingKeys: String, CodingKey {
        case gestureId = "gesture_id"
        case nFilms = "n_films"
        case commitRate = "commit_rate"
        case commitCorrectRate = "commit_correct_rate"
        case precision, recall, f1
    }
}

struct PoseNonModalImpactEntry: Codable, Identifiable {
    let gestureId: String
    let recallAllFilms: Double?
    let recallModalFilmsOnly: Double?
    let nAll: Int
    let nModal: Int

    var id: String { gestureId }

    enum CodingKeys: String, CodingKey {
        case gestureId = "gesture_id"
        case recallAllFilms = "recall_all_films"
        case recallModalFilmsOnly = "recall_modal_films_only"
        case nAll = "n_all"
        case nModal = "n_modal"
    }
}

struct PoseNonModalImpact: Codable {
    let perClass: [PoseNonModalImpactEntry]
    let note: String

    enum CodingKeys: String, CodingKey {
        case perClass = "per_class"
        case note
    }
}

struct PoseLayer3Metrics: Codable {
    let nFilms: Int
    let commitRate: Double
    let commitCorrectRate: Double
    let noPrefixRate: Double
    let prematureIdleRate: Double
    let idleWhileLivePrefixRate: Double
    let idleWhileLivePrefixSuccessRate: Double?
    let lengthDistributionBySignal: [String: [PoseLengthDistEntry]]
    let perGesture: [PosePerGestureMetric]
    let nonModalExclusionImpact: PoseNonModalImpact?

    enum CodingKeys: String, CodingKey {
        case nFilms = "n_films"
        case commitRate = "commit_rate"
        case commitCorrectRate = "commit_correct_rate"
        case noPrefixRate = "no_prefix_rate"
        case prematureIdleRate = "premature_idle_rate"
        case idleWhileLivePrefixRate = "idle_while_live_prefix_rate"
        case idleWhileLivePrefixSuccessRate = "idle_while_live_prefix_success_rate"
        case lengthDistributionBySignal = "length_distribution_by_signal"
        case perGesture = "per_gesture"
        case nonModalExclusionImpact = "non_modal_exclusion_impact"
    }
}

// MARK: - Stage 10: Migration report + bootstrap stability

struct ExclusionMigrationEntry: Codable, Identifiable {
    let filmId: String
    let oldOrdinal: Int?
    let oldRepFrame: Int?
    let newHoldOrdinals: [Int]
    let newOrdinal: Int?

    var id: String { "\(filmId)_\(oldOrdinal ?? -1)" }

    enum CodingKeys: String, CodingKey {
        case filmId = "film_id"
        case oldOrdinal = "old_ordinal"
        case oldRepFrame = "old_rep_frame"
        case newHoldOrdinals = "new_hold_ordinals"
        case newOrdinal = "new_ordinal"
    }
}

struct ClusterMigrationEntry: Codable, Identifiable {
    let `case`: String   // "inherited" | "new" | "split" | "merge" | "lost_review"
    let newId: Int?
    let oldId: Int?
    let oldIds: [Int]
    let newIds: [Int]
    let distance: Double?
    let oldKind: String?

    var id: String { "\(`case`)_\(oldId ?? -1)_\(newId ?? -1)" }

    enum CodingKeys: String, CodingKey {
        case `case` = "case"
        case newId = "new_id"
        case oldId = "old_id"
        case oldIds = "old_ids"
        case newIds = "new_ids"
        case distance
        case oldKind = "old_kind"
    }
}

struct MigrationReport: Codable {
    let exclusionClean: [ExclusionMigrationEntry]
    let exclusionSplit: [ExclusionMigrationEntry]
    let exclusionMerge: [ExclusionMigrationEntry]
    let exclusionLost: [ExclusionMigrationEntry]
    let clusterMigration: [ClusterMigrationEntry]

    enum CodingKeys: String, CodingKey {
        case exclusionClean = "exclusion_clean"
        case exclusionSplit = "exclusion_split"
        case exclusionMerge = "exclusion_merge"
        case exclusionLost = "exclusion_lost"
        case clusterMigration = "cluster_migration"
    }

    var hasExclusionIssues: Bool {
        !exclusionSplit.isEmpty || !exclusionMerge.isEmpty || !exclusionLost.isEmpty
    }
}

struct BootstrapStabilityResult: Codable {
    let p95Drift: Double?
    let isStable: Bool
    let nStableClusters: Int
    let nResamples: Int
    let warning: String?

    enum CodingKeys: String, CodingKey {
        case p95Drift = "p95_drift"
        case isStable = "is_stable"
        case nStableClusters = "n_stable_clusters"
        case nResamples = "n_resamples"
        case warning
    }
}

struct PoseMetricsResponse: Codable {
    let modelId: String
    let trainedAt: TimeInterval
    let trainedOn: Int
    let layer2: PoseLayer2Metrics?
    let layer3: PoseLayer3Metrics?
    let confidenceCurvePose: [ConfidenceCurvePoint]
    let migrationReport: MigrationReport?
    let bootstrapStability: BootstrapStabilityResult?

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case trainedAt = "trained_at"
        case trainedOn = "trained_on"
        case layer2
        case layer3
        case confidenceCurvePose = "confidence_curve_pose"
        case migrationReport = "migration_report"
        case bootstrapStability = "bootstrap_stability"
    }
}

// MARK: - Stage 5: Pose manifest, corrections, cluster holds

struct PoseClusterInfo: Codable {
    let label: String
    let kind: String          // "idle" | "regular" | "unconfirmed"
    let suspectedIdle: Bool
    let nSamples: Int
    let centroid: [Double]    // 63 floats

    enum CodingKeys: String, CodingKey {
        case label, kind
        case suspectedIdle = "suspected_idle"
        case nSamples = "n_samples"
        case centroid
    }
}

struct PoseManifestResponse: Codable {
    let version: Int
    let poseClusters: [String: PoseClusterInfo]
    let idlePoses: [Int]
    let gestureTemplates: [String: [[Int]]]
    let templateFractions: [String: [Double]]?

    enum CodingKeys: String, CodingKey {
        case version
        case poseClusters = "pose_clusters"
        case idlePoses = "idle_poses"
        case gestureTemplates = "gesture_templates"
        case templateFractions = "template_fractions"
    }
}

struct ExcludedHoldEntry: Codable {
    let filmId: String
    let holdOrdinal: Int
    let repFrame: Int
    let startFrame: Int
    let endFrame: Int
    let paramsHash: String

    enum CodingKeys: String, CodingKey {
        case filmId = "film_id"
        case holdOrdinal = "hold_ordinal"
        case repFrame = "rep_frame"
        case startFrame = "start_frame"
        case endFrame = "end_frame"
        case paramsHash = "params_hash"
    }
}

struct PoseCorrectionsResponse: Codable {
    let clusterKinds: [String: String]
    let excludedHolds: [ExcludedHoldEntry]
    let extraTemplates: [String: [[Int]]]

    enum CodingKeys: String, CodingKey {
        case clusterKinds = "cluster_kinds"
        case excludedHolds = "excluded_holds"
        case extraTemplates = "extra_templates"
    }
}

struct PoseCorrectionsRequest: Encodable {
    let clusterKinds: [String: String]
    let excludedHolds: [ExcludedHoldEntry]
    let extraTemplates: [String: [[Int]]]

    enum CodingKeys: String, CodingKey {
        case clusterKinds = "cluster_kinds"
        case excludedHolds = "excluded_holds"
        case extraTemplates = "extra_templates"
    }
}

struct ClusterHoldEntry: Codable, Identifiable {
    let filmId: String
    let gestureId: String
    let clusterId: Int
    let ordinal: Int
    let isEdge: Bool
    let positionFraction: Double
    let distanceFromCentroid: Double
    let coords: [Double]     // 63 floats — wrist-relative normalised coords

    var id: String { "\(filmId)_\(ordinal)" }

    enum CodingKeys: String, CodingKey {
        case filmId = "film_id"
        case gestureId = "gesture_id"
        case clusterId = "cluster_id"
        case ordinal
        case isEdge = "is_edge"
        case positionFraction = "position_fraction"
        case distanceFromCentroid = "distance_from_centroid"
        case coords
    }
}

struct ClusterHoldsResponse: Codable {
    let holds: [ClusterHoldEntry]
    let paramsHash: String

    enum CodingKeys: String, CodingKey {
        case holds
        case paramsHash = "params_hash"
    }
}

// MARK: - Stage 2: /analyze/holds

struct HoldInfo: Codable, Identifiable {
    let ordinal: Int
    let startFrame: Int
    let endFrame: Int
    let repFrame: Int
    let isEdge: Bool
    let positionFraction: Double

    var id: Int { ordinal }

    enum CodingKeys: String, CodingKey {
        case ordinal
        case startFrame = "start_frame"
        case endFrame = "end_frame"
        case repFrame = "rep_frame"
        case isEdge = "is_edge"
        case positionFraction = "position_fraction"
    }
}

struct AnalyzeHoldsResponse: Codable {
    let holds: [HoldInfo]
    let paramsHash: String

    enum CodingKeys: String, CodingKey {
        case holds
        case paramsHash = "params_hash"
    }
}

// MARK: - Stage 9: Confidence logging

enum ConfidenceLogEntry: Encodable {
    case pose(PoseConfidenceLogEntry)
    case phase3(Phase3ConfidenceLogEntry)

    func encode(to encoder: Encoder) throws {
        switch self {
        case .pose(let e):   try e.encode(to: encoder)
        case .phase3(let e): try e.encode(to: encoder)
        }
    }
}

struct PoseConfidenceLogEntry: Encodable {
    let phase: String = "pose"
    let modelVersion: String
    let predictedPoseId: Int
    let confidence: Double
    let reviewerLabel: String?   // "correct" | "wrong" | "skip" | nil
    let timestamp: TimeInterval
    let filmId: String?

    enum CodingKeys: String, CodingKey {
        case phase
        case modelVersion   = "model_version"
        case predictedPoseId = "predicted_pose_id"
        case confidence
        case reviewerLabel  = "reviewer_label"
        case timestamp
        case filmId         = "film_id"
    }
}

struct Phase3ConfidenceLogEntry: Encodable {
    let phase: String = "phase3"
    let modelVersion: String
    let candidateSetSize: Int
    let predictedClass: String
    let confidence: Double
    let reviewerLabel: String?
    let timestamp: TimeInterval
    let filmId: String?

    enum CodingKeys: String, CodingKey {
        case phase
        case modelVersion    = "model_version"
        case candidateSetSize = "candidate_set_size"
        case predictedClass  = "predicted_class"
        case confidence
        case reviewerLabel   = "reviewer_label"
        case timestamp
        case filmId          = "film_id"
    }
}

struct ConfidenceLogBatchRequest: Encodable {
    let entries: [ConfidenceLogEntry]
}

struct ConfidenceLogResponse: Codable {
    let accepted: Int
}

struct ConfidenceCurvePoint: Codable, Identifiable {
    let tau: Double
    let acceptanceRate: Double
    let conditionalAccuracy: Double?
    let nAccepted: Int

    var id: Double { tau }

    enum CodingKeys: String, CodingKey {
        case tau
        case acceptanceRate   = "acceptance_rate"
        case conditionalAccuracy = "conditional_accuracy"
        case nAccepted        = "n_accepted"
    }
}

struct ConfidenceCurvesResponse: Codable {
    let poseOffline: [ConfidenceCurvePoint]
    let poseOnline: [ConfidenceCurvePoint]
    let phase3Offline: [ConfidenceCurvePoint]
    let phase3Online: [ConfidenceCurvePoint]
    let nPoseSamples: Int
    let nPhase3Samples: Int

    enum CodingKeys: String, CodingKey {
        case poseOffline   = "pose_offline"
        case poseOnline    = "pose_online"
        case phase3Offline = "phase3_offline"
        case phase3Online  = "phase3_online"
        case nPoseSamples  = "n_pose_samples"
        case nPhase3Samples = "n_phase3_samples"
    }
}

// MARK: - Upload State

enum UploadState: Equatable {
    case idle
    case uploading
    case uploaded(total: Int)
    case failed(String)
}

// MARK: - API Client

/// HTTP client for the gesture recognition training server.
/// Set IS_MOCKING_SERVER = true to return hard-coded stubs instead of hitting the network.
class GestureModelAPIClient: ObservableObject {

    // MARK: - Configuration

    @Published var baseURL: URL {
        didSet {
            UserDefaults.standard.set(baseURL.absoluteString, forKey: Self.baseURLKey)
        }
    }

    @Published var registrationToken: String {
        didSet {
            UserDefaults.standard.set(registrationToken, forKey: Self.registrationTokenKey)
        }
    }

    /// Last fetched server version, or nil if never fetched / unreachable.
    @Published var serverVersion: String?

    private static let IS_MOCKING_SERVER = false
    private static let baseURLKey = "GestureModelAPIClient.baseURL"
    private static let defaultBaseURL = "http://192.168.0.107:8000"
    private static let registrationTokenKey = "GestureModelAPIClient.registrationToken"
    private static let deviceIdKey = "GestureModelAPIClient.deviceId"

    /// Stable UUID identifying this app installation. Generated once and persisted.
    private let deviceId: String

    private let tokenStorage = TokenStorage()

    private let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 120
        return URLSession(configuration: config)
    }()

    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys]
        return e
    }()

    private let decoder = JSONDecoder()

    init() {
        let stored = UserDefaults.standard.string(forKey: Self.baseURLKey) ?? Self.defaultBaseURL
        baseURL = URL(string: stored) ?? URL(string: Self.defaultBaseURL)!

        registrationToken = UserDefaults.standard.string(forKey: Self.registrationTokenKey) ?? ""

        if let existingId = UserDefaults.standard.string(forKey: Self.deviceIdKey) {
            deviceId = existingId
        } else {
            let newId = UUID().uuidString
            UserDefaults.standard.set(newId, forKey: Self.deviceIdKey)
            deviceId = newId
        }
    }

    // MARK: - Auth

    /// Clears the stored JWT so the next request triggers fresh registration.
    func clearToken() {
        tokenStorage.delete()
    }

    /// Ensures a valid JWT is stored in the Keychain.
    /// If no token exists, calls POST /auth/register and saves the returned token.
    private func ensureAuthenticated() async throws {
        guard tokenStorage.load() == nil else { return }

        guard !registrationToken.isEmpty else {
            throw APIError.missingRegistrationToken
        }

        var request = URLRequest(url: baseURL.appendingPathComponent("auth/register"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: String] = ["device_id": deviceId, "registration_token": registrationToken]
        request.httpBody = try encoder.encode(body)

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            let detail = (try? JSONDecoder().decode(ServerErrorDetail.self, from: data))?.detail
                ?? "Registration failed (HTTP \(code))"
            throw APIError.httpError(statusCode: code, detail: detail)
        }
        struct TokenResponse: Decodable {
            let accessToken: String
            enum CodingKeys: String, CodingKey { case accessToken = "access_token" }
        }
        let tokenResponse = try JSONDecoder().decode(TokenResponse.self, from: data)
        tokenStorage.save(tokenResponse.accessToken)
    }

    // MARK: - Upload Example

    func uploadExample(_ example: TrainingExample) async throws -> UploadExampleResponse {
        if Self.IS_MOCKING_SERVER {
            simulateLog("POST", path: "/examples")
            return UploadExampleResponse(id: UUID().uuidString, totalForGesture: 1)
        }

        let payload = TrainingExamplePayload(from: example)
        var request = URLRequest(url: baseURL.appendingPathComponent("examples"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(payload)

        return try await perform(request, decoding: UploadExampleResponse.self)
    }

    // MARK: - Download Examples

    func downloadExamples(gestureId: String) async throws -> ExampleListResponse {
        if Self.IS_MOCKING_SERVER {
            simulateLog("GET", path: "/examples?gesture_id=\(gestureId)")
            return ExampleListResponse(examples: [], total: 0)
        }

        var components = URLComponents(url: baseURL.appendingPathComponent("examples"), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "gesture_id", value: gestureId)]
        let request = URLRequest(url: components.url!)
        return try await perform(request, decoding: ExampleListResponse.self)
    }

    // MARK: - Update Example (Relabel)

    func updateExample(id: String, gestureId: String) async throws {
        if Self.IS_MOCKING_SERVER {
            simulateLog("PUT", path: "/examples/\(id)")
            return
        }

        var request = URLRequest(url: baseURL.appendingPathComponent("examples/\(id)"))
        request.httpMethod = "PUT"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body = ["gesture_id": gestureId]
        request.httpBody = try encoder.encode(body)

        let _: [String: String] = try await perform(request, decoding: [String: String].self)
    }

    // MARK: - Delete Single Example

    func deleteExample(id: String) async throws {
        if Self.IS_MOCKING_SERVER {
            simulateLog("DELETE", path: "/examples/\(id)")
            return
        }

        var request = URLRequest(url: baseURL.appendingPathComponent("examples/\(id)"))
        request.httpMethod = "DELETE"

        struct DeleteResponse: Decodable {
            let id: String
            let deleted: Bool
        }
        let _: DeleteResponse = try await perform(request, decoding: DeleteResponse.self)
    }

    // MARK: - Example Stats

    func fetchExampleStats() async throws -> ExampleStatsResponse {
        if Self.IS_MOCKING_SERVER {
            simulateLog("GET", path: "/examples/stats")
            return ExampleStatsResponse(gestures: [], total: 0)
        }

        let request = URLRequest(url: baseURL.appendingPathComponent("examples/stats"))
        return try await perform(request, decoding: ExampleStatsResponse.self)
    }

    // MARK: - Trigger Training

    func triggerTraining(
        minInViewDuration: Double = 1.2,
        balanceStrategy: String = "class_weight"
    ) async throws -> TrainingJobResponse {
        if Self.IS_MOCKING_SERVER {
            simulateLog("POST", path: "/train")
            return TrainingJobResponse(jobId: UUID().uuidString, status: "started")
        }

        var request = URLRequest(url: baseURL.appendingPathComponent("train"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "min_in_view_duration": minInViewDuration,
            "balance_strategy": balanceStrategy,
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        return try await perform(request, decoding: TrainingJobResponse.self)
    }

    // MARK: - Model Info

    func fetchModelInfo() async throws -> ModelInfoResponse {
        if Self.IS_MOCKING_SERVER {
            simulateLog("GET", path: "/model/info")
            return ModelInfoResponse(
                modelId: "mock",
                trainer: "rf_mlp",
                trainedOn: 0,
                trainedAt: Date().timeIntervalSince1970,
                gestureIds: [],
                accuracy: nil,
                f1: nil,
                confusionMatrix: nil,
                minInViewDuration: nil
            )
        }
        let request = URLRequest(url: baseURL.appendingPathComponent("model/info"))
        return try await perform(request, decoding: ModelInfoResponse.self)
    }

    // MARK: - Model Metrics (public / no auth)

    func fetchLatestMetrics() async throws -> ModelMetricsResponse {
        let request = URLRequest(url: baseURL.appendingPathComponent("model/metrics"))
        return try await performUnauthenticated(request, decoding: ModelMetricsResponse.self)
    }

    func fetchMetricsList() async throws -> ModelMetricsListResponse {
        let request = URLRequest(url: baseURL.appendingPathComponent("model/metrics/list"))
        return try await performUnauthenticated(request, decoding: ModelMetricsListResponse.self)
    }

    func fetchMetrics(modelId: String) async throws -> ModelMetricsResponse {
        let request = URLRequest(url: baseURL.appendingPathComponent("model/metrics/\(modelId)"))
        return try await performUnauthenticated(request, decoding: ModelMetricsResponse.self)
    }

    // MARK: - Model Status

    func fetchModelStatus() async throws -> ModelStatusResponse {
        if Self.IS_MOCKING_SERVER {
            simulateLog("GET", path: "/model/status")
            return ModelStatusResponse(
                status: "idle",
                accuracy: nil,
                trainedOn: 0,
                gestureIds: [],
                trainedAt: nil,
                error: nil
            )
        }

        let request = URLRequest(url: baseURL.appendingPathComponent("model/status"))
        return try await perform(request, decoding: ModelStatusResponse.self)
    }

    // MARK: - Download Preprocessor

    /// Download preprocessor.js from the server, save it alongside the model file,
    /// and immediately load it into JSPreprocessorWrapper.
    func downloadPreprocessor() async throws -> URL {
        try await ensureAuthenticated()

        var request = URLRequest(url: baseURL.appendingPathComponent("model/preprocessor"))
        if let token = tokenStorage.load() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (tmpURL, response) = try await session.download(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            throw APIError.httpError(statusCode: code, detail: "model/preprocessor download failed")
        }

        let destURL = preprocessorURL()
        let destDir = destURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        if FileManager.default.fileExists(atPath: destURL.path) {
            try FileManager.default.removeItem(at: destURL)
        }
        try FileManager.default.moveItem(at: tmpURL, to: destURL)

        print("[GestureModelAPIClient] Preprocessor saved to \(destURL.path)")
        return destURL
    }

    func preprocessorURL() -> URL {
        tfliteModelURL()
            .deletingLastPathComponent()
            .appendingPathComponent("preprocessor.js")
    }

    // MARK: - Download Model

    /// Download the latest .tflite model from the server.
    /// Writes gesture_model.tflite to Documents/GestureModel/, then fetches
    /// /model/status and writes gesture_ids.json sidecar alongside it.
    /// Returns the URL of the saved .tflite file.
    func downloadModel() async throws -> URL {
        if Self.IS_MOCKING_SERVER {
            simulateLog("GET", path: "/model/download")
            let dest = tfliteModelURL()
            print("[GestureModelAPIClient] Mock: would write model to \(dest.path)")
            return dest
        }

        // 1. Download the binary
        try await ensureAuthenticated()
        var downloadRequest = URLRequest(url: baseURL.appendingPathComponent("model/download"))
        if let token = tokenStorage.load() {
            downloadRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (tmpURL, response) = try await session.download(for: downloadRequest)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard httpResponse.statusCode == 200 else {
            throw APIError.httpError(statusCode: httpResponse.statusCode, detail: "model/download failed")
        }

        // 2. Move to Documents/GestureModel/gesture_model.tflite
        let destURL = tfliteModelURL()
        let destDir = destURL.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)

        if FileManager.default.fileExists(atPath: destURL.path) {
            try FileManager.default.removeItem(at: destURL)
        }
        try FileManager.default.moveItem(at: tmpURL, to: destURL)

        // 3. Fetch status and write gesture_ids.json sidecar
        let status = try await fetchModelStatus()
        let sidecarURL = destDir.appendingPathComponent("gesture_ids.json")
        let sidecarData = try JSONEncoder().encode(status.gestureIds)
        try sidecarData.write(to: sidecarURL, options: .atomic)

        print("[GestureModelAPIClient] Model saved to \(destURL.path), gesture_ids: \(status.gestureIds)")
        return destURL
    }

    // MARK: - Update Server

    /// POST /admin/update — triggers git pull + docker compose up --build -d on the VPS.
    /// The server will restart shortly after; the response arrives before it goes down.
    func updateServer() async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent("admin/update"))
        request.httpMethod = "POST"
        let _: UpdateServerResponse = try await perform(request, decoding: UpdateServerResponse.self)
    }

    // MARK: - Pose Metrics (Stage 8)

    func fetchPoseMetrics() async throws -> PoseMetricsResponse {
        let request = URLRequest(url: baseURL.appendingPathComponent("model/pose/metrics"))
        return try await perform(request, decoding: PoseMetricsResponse.self)
    }

    // MARK: - Confidence log (Stage 9)

    /// POST /confidence-log — upload a batch of per-hold pose or per-cycle Phase 3 log entries.
    @discardableResult
    func postConfidenceLog(entries: [ConfidenceLogEntry]) async throws -> ConfidenceLogResponse {
        var request = URLRequest(url: baseURL.appendingPathComponent("confidence-log"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(ConfidenceLogBatchRequest(entries: entries))
        return try await perform(request, decoding: ConfidenceLogResponse.self)
    }

    /// GET /confidence-log/curves — offline + on-device τ-sweep curves for Phase 2 and Phase 3.
    func fetchConfidenceCurves() async throws -> ConfidenceCurvesResponse {
        let request = URLRequest(url: baseURL.appendingPathComponent("confidence-log/curves"))
        return try await perform(request, decoding: ConfidenceCurvesResponse.self)
    }

    // MARK: - Pose Model (Stage 4 / Stage 5)

    func triggerPoseTraining() async throws -> PoseTrainingJobResponse {
        var request = URLRequest(url: baseURL.appendingPathComponent("train/pose"))
        request.httpMethod = "POST"
        return try await perform(request, decoding: PoseTrainingJobResponse.self)
    }

    func fetchPoseTrainingStatus() async throws -> PoseTrainingStatusResponse {
        let request = URLRequest(url: baseURL.appendingPathComponent("model/pose/status"))
        return try await perform(request, decoding: PoseTrainingStatusResponse.self)
    }

    func fetchPoseManifest() async throws -> PoseManifestResponse {
        let request = URLRequest(url: baseURL.appendingPathComponent("model/pose/manifest"))
        return try await perform(request, decoding: PoseManifestResponse.self)
    }

    func fetchPoseCorrections() async throws -> PoseCorrectionsResponse {
        let request = URLRequest(url: baseURL.appendingPathComponent("pose/corrections"))
        return try await perform(request, decoding: PoseCorrectionsResponse.self)
    }

    func putPoseCorrections(_ body: PoseCorrectionsRequest) async throws -> PoseCorrectionsResponse {
        var request = URLRequest(url: baseURL.appendingPathComponent("pose/corrections"))
        request.httpMethod = "PUT"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(body)
        return try await perform(request, decoding: PoseCorrectionsResponse.self)
    }

    func fetchClusterHolds() async throws -> ClusterHoldsResponse {
        let request = URLRequest(url: baseURL.appendingPathComponent("model/pose/cluster-holds"))
        return try await perform(request, decoding: ClusterHoldsResponse.self)
    }

    // MARK: - Download Pose Model (Stage 7)

    /// Download pose_model.tflite and pose_manifest.json, saving them alongside the
    /// handfilm model in Documents/GestureModel/.
    /// Returns the URL of the saved .tflite file.
    func downloadPoseModel() async throws -> URL {
        let dir = tfliteModelURL().deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // Download tflite
        try await ensureAuthenticated()
        var tfliteRequest = URLRequest(url: baseURL.appendingPathComponent("model/pose/download"))
        if let token = tokenStorage.load() {
            tfliteRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let (tmpTflite, tfliteResponse) = try await session.download(for: tfliteRequest)
        guard let http = tfliteResponse as? HTTPURLResponse, http.statusCode == 200 else {
            throw APIError.httpError(statusCode: (tfliteResponse as? HTTPURLResponse)?.statusCode ?? 0,
                                     detail: "model/pose/download failed")
        }
        let tfliteDest = dir.appendingPathComponent("pose_model.tflite")
        if FileManager.default.fileExists(atPath: tfliteDest.path) {
            try FileManager.default.removeItem(at: tfliteDest)
        }
        try FileManager.default.moveItem(at: tmpTflite, to: tfliteDest)

        // Download manifest
        let manifest = try await fetchPoseManifest()
        let manifestData = try JSONEncoder().encode(manifest)
        let manifestDest = dir.appendingPathComponent("pose_manifest.json")
        try manifestData.write(to: manifestDest, options: .atomic)

        print("[GestureModelAPIClient] Pose model saved to \(tfliteDest.path)")
        return tfliteDest
    }

    func poseModelURL() -> URL {
        tfliteModelURL().deletingLastPathComponent().appendingPathComponent("pose_model.tflite")
    }

    func poseManifestURL() -> URL {
        tfliteModelURL().deletingLastPathComponent().appendingPathComponent("pose_manifest.json")
    }

    // MARK: - Analyze Holds (Stage 2)

    func analyzeHolds(film: HandFilm) async throws -> AnalyzeHoldsResponse {
        if Self.IS_MOCKING_SERVER {
            simulateLog("POST", path: "/analyze/holds")
            return AnalyzeHoldsResponse(holds: [], paramsHash: "sha256:mock")
        }

        let payload = HandFilmAnalyzePayload(from: film)
        var request = URLRequest(url: baseURL.appendingPathComponent("analyze/holds"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(payload)

        return try await perform(request, decoding: AnalyzeHoldsResponse.self)
    }

    // MARK: - Server Version

    /// GET /version — unauthenticated. Updates `serverVersion` on success,
    /// sets it to nil on failure (shown as "unknown" in the UI).
    @discardableResult
    func fetchServerVersion() async -> String? {
        let request = URLRequest(url: baseURL.appendingPathComponent("version"))
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                await MainActor.run { self.serverVersion = nil }
                return nil
            }
            let decoded = try decoder.decode(VersionResponse.self, from: data)
            await MainActor.run { self.serverVersion = decoded.version }
            return decoded.version
        } catch {
            await MainActor.run { self.serverVersion = nil }
            return nil
        }
    }

    // MARK: - Wipe Model

    /// Sends DELETE /model to the server, then removes the local .tflite and sidecar files.
    func wipeModel() async throws {
        if Self.IS_MOCKING_SERVER {
            simulateLog("DELETE", path: "/model")
            deleteLocalModelFiles()
            return
        }

        var request = URLRequest(url: baseURL.appendingPathComponent("model"))
        request.httpMethod = "DELETE"

        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let detail = (try? JSONDecoder().decode(ServerErrorDetail.self, from: data))?.detail
                ?? HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
            throw APIError.httpError(statusCode: httpResponse.statusCode, detail: detail)
        }

        deleteLocalModelFiles()
    }

    private func deleteLocalModelFiles() {
        let modelURL = tfliteModelURL()
        let sidecarURL = modelURL.deletingLastPathComponent().appendingPathComponent("gesture_ids.json")
        for url in [modelURL, sidecarURL] {
            if FileManager.default.fileExists(atPath: url.path) {
                try? FileManager.default.removeItem(at: url)
            }
        }
        print("[GestureModelAPIClient] Local model files removed")
    }

    // MARK: - Helpers

    func tfliteModelURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("GestureModel", isDirectory: true)
            .appendingPathComponent("gesture_model.tflite")
    }

    private func perform<T: Decodable>(_ request: URLRequest, decoding type: T.Type) async throws -> T {
        try await ensureAuthenticated()

        var authedRequest = request
        if let token = tokenStorage.load() {
            authedRequest.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (data, response) = try await session.data(for: authedRequest)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        if httpResponse.statusCode == 401 {
            tokenStorage.delete()
            throw APIError.unauthorized
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let detail = (try? JSONDecoder().decode(ServerErrorDetail.self, from: data))?.detail ?? HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
            throw APIError.httpError(statusCode: httpResponse.statusCode, detail: detail)
        }

        return try decoder.decode(T.self, from: data)
    }

    /// Performs a request without attaching any auth header — used for endpoints
    /// that are intentionally public (e.g. /model/metrics).
    private func performUnauthenticated<T: Decodable>(_ request: URLRequest, decoding type: T.Type) async throws -> T {
        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let detail = (try? JSONDecoder().decode(ServerErrorDetail.self, from: data))?.detail
                ?? HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
            throw APIError.httpError(statusCode: httpResponse.statusCode, detail: detail)
        }

        return try decoder.decode(T.self, from: data)
    }

    private func simulateLog(_ method: String, path: String) {
        print("[GestureModelAPIClient] MOCK \(method) \(baseURL)\(path)")
    }
}

// MARK: - API Errors

enum APIError: LocalizedError {
    case httpError(statusCode: Int, detail: String)
    case unauthorized
    case missingRegistrationToken

    var errorDescription: String? {
        switch self {
        case .httpError(let code, let detail):
            return "Server error \(code): \(detail)"
        case .unauthorized:
            return "Token rejected by server. Please re-register in Settings."
        case .missingRegistrationToken:
            return "No registration token set. Enter it in Settings → Server."
        }
    }
}

private struct ServerErrorDetail: Decodable {
    let detail: String
}

// MARK: - Request Payload

/// Codable wrapper for TrainingExample to send over the network.
/// HandGestureTypes structs are not Codable by design; this wrapper lives in the app target.
private struct TrainingExamplePayload: Encodable {
    let id: String
    let handFilm: HandFilmPayload
    let gestureId: String
    let sessionId: String
    let userId: String?

    enum CodingKeys: String, CodingKey {
        case id
        case handFilm = "hand_film"
        case gestureId = "gesture_id"
        case sessionId = "session_id"
        case userId = "user_id"
    }

    init(from example: TrainingExample) {
        id = example.id.uuidString
        handFilm = HandFilmPayload(from: example.handfilm)
        gestureId = example.gestureId
        sessionId = example.sessionId
        userId = example.userId
    }
}

private struct HandFilmPayload: Encodable {
    let frames: [HandShotPayload]
    let startTime: TimeInterval

    enum CodingKeys: String, CodingKey {
        case frames
        case startTime = "start_time"
    }

    init(from film: HandFilm) {
        frames = film.frames.map { HandShotPayload(from: $0) }
        startTime = film.startTime
    }
}

private struct HandShotPayload: Encodable {
    let landmarks: [Point3DPayload]
    let timestamp: TimeInterval
    let leftOrRight: String

    enum CodingKeys: String, CodingKey {
        case landmarks
        case timestamp
        case leftOrRight = "left_or_right"
    }

    init(from shot: HandShot) {
        landmarks = shot.landmarks.map { Point3DPayload(from: $0) }
        timestamp = shot.timestamp
        leftOrRight = {
            switch shot.leftOrRight {
            case .left: return "left"
            case .right: return "right"
            case .unknown: return "unknown"
            }
        }()
    }
}

private struct Point3DPayload: Encodable {
    let x: Float
    let y: Float
    let z: Float

    init(from point: Point3D) {
        x = point.x; y = point.y; z = point.z
    }
}

// Analyze-only payload — includes is_absent so the JS preprocessor can filter absent frames.
private struct HandFilmAnalyzePayload: Encodable {
    let frames: [HandShotAnalyzePayload]
    let startTime: TimeInterval

    enum CodingKeys: String, CodingKey {
        case frames
        case startTime = "start_time"
    }

    init(from film: HandFilm) {
        frames = film.frames.map { HandShotAnalyzePayload(from: $0) }
        startTime = film.startTime
    }
}

private struct HandShotAnalyzePayload: Encodable {
    let landmarks: [Point3DPayload]
    let timestamp: TimeInterval
    let leftOrRight: String
    let isAbsent: Bool

    enum CodingKeys: String, CodingKey {
        case landmarks
        case timestamp
        case leftOrRight = "left_or_right"
        case isAbsent = "is_absent"
    }

    init(from shot: HandShot) {
        landmarks = shot.landmarks.map { Point3DPayload(from: $0) }
        timestamp = shot.timestamp
        leftOrRight = {
            switch shot.leftOrRight {
            case .left: return "left"
            case .right: return "right"
            case .unknown: return "unknown"
            }
        }()
        isAbsent = shot.isAbsent
    }
}

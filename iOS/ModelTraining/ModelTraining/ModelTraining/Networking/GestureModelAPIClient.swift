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

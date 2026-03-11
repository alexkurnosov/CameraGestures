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

    private static let IS_MOCKING_SERVER = false
    private static let baseURLKey = "GestureModelAPIClient.baseURL"
    private static let defaultBaseURL = "http://192.168.0.107:8000"

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

    func triggerTraining() async throws -> TrainingJobResponse {
        if Self.IS_MOCKING_SERVER {
            simulateLog("POST", path: "/train")
            return TrainingJobResponse(jobId: UUID().uuidString, status: "started")
        }

        var request = URLRequest(url: baseURL.appendingPathComponent("train"))
        request.httpMethod = "POST"

        return try await perform(request, decoding: TrainingJobResponse.self)
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
        let downloadURL = baseURL.appendingPathComponent("model/download")
        let (tmpURL, response) = try await session.download(from: downloadURL)

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

    // MARK: - Helpers

    func tfliteModelURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("GestureModel", isDirectory: true)
            .appendingPathComponent("gesture_model.tflite")
    }

    private func perform<T: Decodable>(_ request: URLRequest, decoding type: T.Type) async throws -> T {
        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            let detail = (try? JSONDecoder().decode(ServerErrorDetail.self, from: data))?.detail ?? HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
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

    var errorDescription: String? {
        switch self {
        case .httpError(let code, let detail):
            return "Server error \(code): \(detail)"
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
    let handFilm: HandFilmPayload
    let gestureId: String
    let sessionId: String
    let userId: String?

    enum CodingKeys: String, CodingKey {
        case handFilm = "hand_film"
        case gestureId = "gesture_id"
        case sessionId = "session_id"
        case userId = "user_id"
    }

    init(from example: TrainingExample) {
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

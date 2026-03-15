import SwiftUI
import HandGestureTypes
import GestureModelModule
import HandGestureRecognizingFramework
import Combine

@main
struct ModelTrainingApp: App {

    // MARK: - State Management

    @StateObject private var gestureRecognizer = GestureRecognizerWrapper(recognizer: HandGestureRecognizing())
    @StateObject private var trainingDataManager = TrainingDataManager()
    @StateObject private var appSettings = AppSettings()
    @StateObject private var gestureRegistry = GestureRegistry()
    @StateObject private var apiClient = GestureModelAPIClient()

    // MARK: - Scene Configuration

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(gestureRecognizer)
                .environmentObject(trainingDataManager)
                .environmentObject(appSettings)
                .environmentObject(gestureRegistry)
                .environmentObject(apiClient)
                .preferredColorScheme(appSettings.colorScheme)
                .onAppear {
                    trainingDataManager.apiClient = apiClient
                }
        }
    }
}

// MARK: - Training Data Manager

class TrainingDataManager: ObservableObject {
    @Published var currentDataset: TrainingDataset?
    @Published var trainingExamples: [TrainingExample] = []
    @Published var isCollecting = false
    @Published var currentGestureId: String?
    @Published var trainingState: TrainingState = .idle
    @Published var uploadState: UploadState = .idle

    /// Currently selected gesture for training (shared between TrainingView and CameraView).
    @Published var selectedGesture: GestureDefinition?

    /// Examples collected locally and not yet sent to the server.
    @Published var pendingExamples: [TrainingExample] = []
    @Published var isSendingToServer = false

    weak var apiClient: GestureModelAPIClient?

    // MARK: - Data Collection

    func startDataCollection(for gesture: GestureDefinition) {
        currentGestureId = gesture.id
        isCollecting = true
    }

    func stopDataCollection() {
        isCollecting = false
        currentGestureId = nil
    }

    /// Add an example to local collections. Does NOT auto-upload.
    func addTrainingExample(_ example: TrainingExample) {
        trainingExamples.append(example)
        pendingExamples.append(example)
        if currentDataset != nil {
            currentDataset?.addExample(example)
        }
        objectWillChange.send()
    }

    /// Upload all pending examples to the server (currently mocked).
    func sendPendingToServer() {
        guard !pendingExamples.isEmpty, !isSendingToServer, let client = apiClient else { return }
        isSendingToServer = true
        uploadState = .uploading
        let batch = pendingExamples
        pendingExamples = []
        Task.detached { [weak self, weak client] in
            guard let client else { return }
            var lastTotal = 0
            for example in batch {
                do {
                    let response = try await client.uploadExample(example)
                    lastTotal = response.totalForGesture
                } catch {
                    await MainActor.run {
                        self?.uploadState = .failed(error.localizedDescription)
                        self?.isSendingToServer = false
                    }
                    return
                }
            }
            await MainActor.run {
                self?.uploadState = .uploaded(total: lastTotal)
                self?.isSendingToServer = false
            }
        }
    }

    func createNewDataset(name: String) {
        var dataset = TrainingDataset(name: name)
        for example in trainingExamples {
            dataset.addExample(example)
        }
        currentDataset = dataset
    }

    // MARK: - Persistence

    /// Serialize the current dataset (and all training examples) to Documents.
    func saveDataset() {
        guard let dataset = currentDataset else { return }
        let url = datasetFileURL(name: dataset.name)
        do {
            let dto = TrainingDatasetDTO(from: dataset)
            let data = try JSONEncoder().encode(dto)
            let dir = url.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try data.write(to: url, options: .atomic)
        } catch {
            print("TrainingDataManager: failed to save dataset — \(error)")
        }
    }

    /// Load a previously saved dataset from Documents.
    func loadDataset(name: String) {
        let url = datasetFileURL(name: name)
        guard let data = try? Data(contentsOf: url),
              let dto = try? JSONDecoder().decode(TrainingDatasetDTO.self, from: data) else {
            return
        }
        let dataset = dto.toTrainingDataset()
        currentDataset = dataset
        trainingExamples = dataset.examples
    }

    /// All dataset names currently saved on disk.
    func savedDatasetNames() -> [String] {
        let dir = datasetsDirectory()
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: nil,
            options: .skipsHiddenFiles
        ) else { return [] }
        return files
            .filter { $0.pathExtension == "json" }
            .map { $0.deletingPathExtension().lastPathComponent }
            .sorted()
    }

    // MARK: - File Paths

    private func datasetsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("TrainingDatasets", isDirectory: true)
    }

    private func datasetFileURL(name: String) -> URL {
        datasetsDirectory().appendingPathComponent("\(name).json")
    }
}

// MARK: - Training State

enum TrainingState: Equatable {
    case idle
    case training
    case done(ModelMetrics)
    case failed(String)

    static func == (lhs: TrainingState, rhs: TrainingState) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle), (.training, .training): return true
        case (.done, .done): return true
        case (.failed(let a), .failed(let b)): return a == b
        default: return false
        }
    }
}

// MARK: - App Settings

class AppSettings: ObservableObject {
    @Published var colorScheme: ColorScheme? = nil
    @Published var preferredCamera: Int = 0
    @Published var targetFPS: Int = 30
    @Published var confidenceThreshold: Float = 0.7
    @Published var enableHapticFeedback = true
    @Published var showDebugInfo = false

    @Published var cameraConfig = HandsRecognizingConfig.defaultConfig
    @Published var modelConfig = GestureModelConfig.defaultConfig

    func updateCameraConfig() {
        cameraConfig = HandsRecognizingConfig(
            cameraIndex: preferredCamera,
            targetFPS: targetFPS,
            detectBothHands: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        )
    }

    func updateModelConfig() {
        let tfliteURL = defaultTFLiteModelURL()
        let modelPath = FileManager.default.fileExists(atPath: tfliteURL.path)
            ? tfliteURL.path
            : nil
        modelConfig = GestureModelConfig(
            modelPath: modelPath,
            backendType: modelPath != nil ? .tensorFlow : .mock,
            predictionThreshold: confidenceThreshold,
            maxPredictions: 5
        )
    }

    func defaultTFLiteModelURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("GestureModel/gesture_model.tflite")
    }
}

// MARK: - Gesture Recognizer Wrapper

@MainActor
class GestureRecognizerWrapper: ObservableObject {
    let recognizer: HandGestureRecognizing

    @Published var isRecognizing: Bool = false
    @Published var currentGesture: String?
    @Published var confidence: Float = 0.0
    @Published var lastError: String?

    init(recognizer: HandGestureRecognizing) {
        self.recognizer = recognizer
    }
}

// MARK: - Training Series Coordinator

/// Drives a repeating series of HandFilm captures for training data collection.
/// Each iteration: countdown → recording window → pause → repeat.
/// Subscribes to handshotCallback/handfilmCallback on the shared HandGestureRecognizing instance.
@MainActor
class TrainingSeriesCoordinator: ObservableObject {

    // MARK: - Configurable timing
    var captureWindow: TimeInterval = 1.0
    var pauseInterval: TimeInterval = 5.0
    var countdownDuration: Int = 3

    // MARK: - Capture phase
    enum Phase: Equatable {
        case idle
        case countdown(remaining: Int)
        case recording
        case pause(remaining: Int)
    }

    @Published var phase: Phase = .idle
    @Published var capturedCount: Int = 0
    @Published var handTrackingPoints: [Point3D] = []

    private weak var gestureRecognizer: HandGestureRecognizing?
    private var seriesTask: Task<Void, Never>?
    private var latestFilm = HandFilm()
    private var filmReady = false
    private var onFilmCaptured: ((HandFilm) -> Void)?

    var isRunning: Bool { phase != .idle }

    // MARK: - Start / Stop

    func start(
        using recognizer: HandGestureRecognizing,
        captureWindow: TimeInterval,
        pauseInterval: TimeInterval,
        onFilmCaptured: @escaping (HandFilm) -> Void
    ) {
        stop()
        self.gestureRecognizer = recognizer
        self.captureWindow = captureWindow
        self.pauseInterval = pauseInterval
        self.onFilmCaptured = onFilmCaptured
        capturedCount = 0

        recognizer.handshotCallback = { [weak self] handshot in
            DispatchQueue.main.async {
                self?.handTrackingPoints = handshot.landmarks
            }
        }
        recognizer.handfilmCallback = { [weak self] film in
            DispatchQueue.main.async {
                self?.latestFilm = film
                self?.filmReady = true
            }
        }

        seriesTask = Task { await self.runLoop() }
    }

    func stop() {
        seriesTask?.cancel()
        seriesTask = nil
        gestureRecognizer?.handshotCallback = nil
        gestureRecognizer?.handfilmCallback = nil
        gestureRecognizer = nil
        phase = .idle
        handTrackingPoints = []
    }

    // MARK: - Series Loop

    private func runLoop() async {
        while !Task.isCancelled {
            // --- Countdown ---
            for remaining in stride(from: countdownDuration, through: 1, by: -1) {
                guard !Task.isCancelled else { return }
                phase = .countdown(remaining: remaining)
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            guard !Task.isCancelled else { return }

            // --- Recording ---
            phase = .recording
            filmReady = false

            let deadline = Date().addingTimeInterval(captureWindow + 0.5)
            while !filmReady && Date() < deadline && !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 100_000_000) // poll every 100ms
            }
            guard !Task.isCancelled else { return }

            if filmReady {
                capturedCount += 1
                onFilmCaptured?(latestFilm)
            }

            // --- Pause ---
            let pauseSecs = Int(pauseInterval)
            for remaining in stride(from: pauseSecs, through: 1, by: -1) {
                guard !Task.isCancelled else { return }
                phase = .pause(remaining: remaining)
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
        phase = .idle
    }
}

// MARK: - DTOs for JSON Serialization

/// DTO for Point3D (HandGestureTypes is not Codable to keep it dependency-free).
struct Point3DDTO: Codable {
    let x: Float
    let y: Float
    let z: Float

    init(from point: Point3D) {
        x = point.x; y = point.y; z = point.z
    }

    func toPoint3D() -> Point3D { Point3D(x: x, y: y, z: z) }
}

struct HandShotDTO: Codable {
    let landmarks: [Point3DDTO]
    let timestamp: TimeInterval
    let leftOrRight: String   // "left" | "right" | "unknown"

    init(from handShot: HandShot) {
        landmarks = handShot.landmarks.map { Point3DDTO(from: $0) }
        timestamp = handShot.timestamp
        leftOrRight = {
            switch handShot.leftOrRight {
            case .left: return "left"
            case .right: return "right"
            case .unknown: return "unknown"
            }
        }()
    }

    func toHandShot() -> HandShot {
        let side: LeftOrRight = leftOrRight == "left" ? .left : leftOrRight == "right" ? .right : .unknown
        return HandShot(
            landmarks: landmarks.map { $0.toPoint3D() },
            timestamp: timestamp,
            leftOrRight: side
        )
    }
}

struct HandFilmDTO: Codable {
    let frames: [HandShotDTO]
    let startTime: TimeInterval

    init(from handFilm: HandFilm) {
        frames = handFilm.frames.map { HandShotDTO(from: $0) }
        startTime = handFilm.startTime
    }

    func toHandFilm() -> HandFilm {
        var film = HandFilm(startTime: startTime)
        frames.map { $0.toHandShot() }.forEach { film.addFrame($0) }
        return film
    }
}

struct TrainingExampleDTO: Codable {
    let handfilm: HandFilmDTO
    let gestureId: String
    let userId: String?
    let sessionId: String
    let timestamp: TimeInterval

    init(from example: TrainingExample) {
        handfilm = HandFilmDTO(from: example.handfilm)
        gestureId = example.gestureId
        userId = example.userId
        sessionId = example.sessionId
        timestamp = example.timestamp
    }

    func toTrainingExample() -> TrainingExample {
        TrainingExample(
            handfilm: handfilm.toHandFilm(),
            gestureId: gestureId,
            userId: userId,
            sessionId: sessionId
        )
    }
}

struct TrainingDatasetDTO: Codable {
    let name: String
    let createdAt: TimeInterval
    let examples: [TrainingExampleDTO]

    init(from dataset: TrainingDataset) {
        name = dataset.name
        createdAt = dataset.createdAt
        examples = dataset.examples.map { TrainingExampleDTO(from: $0) }
    }

    func toTrainingDataset() -> TrainingDataset {
        var dataset = TrainingDataset(name: name)
        examples.map { $0.toTrainingExample() }.forEach { dataset.addExample($0) }
        return dataset
    }
}

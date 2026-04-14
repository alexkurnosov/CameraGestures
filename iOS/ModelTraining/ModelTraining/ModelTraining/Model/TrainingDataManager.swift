import Foundation
import Combine
import UIKit
import HandGestureTypes
import HandGestureRecognizingFramework
import GestureModelModule

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

    /// Per-gesture example counts fetched from the server (already-uploaded examples).
    @Published var serverExampleCounts: [String: Int] = [:]

    /// Non-nil when the last attempt to fetch server counts failed.
    @Published var serverSyncError: String?

    /// Films that failed the quality threshold. Stored locally only; never uploaded.
    @Published var failedExamples: [FailedHandFilm] = []

    // MARK: - Collection Progress

    @Published var currentSamples = 0
    @Published var collectionProgress: Double = 0.0
    @Published var targetSamples = 20

    // MARK: - Dependencies

    weak var apiClient: GestureModelAPIClient? {
        didSet { Task { await fetchServerExampleCounts() } }
    }

    weak var gestureRecognizer: GestureRecognizerWrapper? {
        didSet { setupCollectionSubscription() }
    }

    weak var appSettings: AppSettings?

    private var collectionCancellables = Set<AnyCancellable>()

    // MARK: - Init

    init() {
        loadPendingExamples()
        loadFailedExamples()
    }

    // MARK: - Data Collection

    func startDataCollection(for gesture: GestureDefinition) {
        currentGestureId = gesture.id
        currentSamples = 0
        collectionProgress = 0.0
        isCollecting = true

        if let recognizer = gestureRecognizer, !recognizer.recognizer.isActive {
            Task {
                try? await recognizer.recognizer.start()
            }
        }
    }

    func stopDataCollection() {
        isCollecting = false
        currentGestureId = nil

        if appSettings?.enableHapticFeedback == true {
            let feedback = UINotificationFeedbackGenerator()
            feedback.notificationOccurred(.success)
        }
    }

    // MARK: - Collection Subscription

    private func setupCollectionSubscription() {
        collectionCancellables.removeAll()
        guard let gestureRecognizer else { return }

        gestureRecognizer.gestureDetected
            .receive(on: DispatchQueue.main)
            .sink { [weak self] detectedGesture in
                self?.handleTrainingGesture(detectedGesture)
            }
            .store(in: &collectionCancellables)
    }

    func handleTrainingGesture(_ gesture: DetectedGesture) {
        guard isCollecting,
              let selected = selectedGesture,
              currentGestureId == selected.id else { return }

        let example = TrainingExample(
            handfilm: gesture.handfilm,
            gestureId: selected.id,
            userId: "current_user",
            sessionId: UUID().uuidString
        )

        addTrainingExample(example)

        currentSamples += 1
        collectionProgress = Double(currentSamples) / Double(targetSamples)

        if currentSamples >= targetSamples {
            stopDataCollection()
        }

        if appSettings?.enableHapticFeedback == true {
            let impactFeedback = UIImpactFeedbackGenerator(style: .light)
            impactFeedback.impactOccurred()
        }
    }

    // MARK: - Local Model Training

    func startLocalTraining() {
        guard let dataset = currentDataset, let appSettings else { return }

        trainingState = .training

        let modelConfig = GestureModelConfig(
            modelPath: nil,
            backendType: .tensorFlow,
            predictionThreshold: appSettings.confidenceThreshold,
            maxPredictions: 5
        )

        Task.detached { [weak self] in
            do {
                let gestureModel = GestureModel(config: modelConfig)
                let metrics = try await gestureModel.trainAsync(dataset: dataset)

                await MainActor.run {
                    self?.trainingState = .done(metrics)
                    appSettings.updateModelConfig()
                }
            } catch {
                await MainActor.run {
                    self?.trainingState = .failed(error.localizedDescription)
                }
            }
        }
    }

    /// Add an example to local collections. Does NOT auto-upload.
    func addTrainingExample(_ example: TrainingExample) {
        trainingExamples.append(example)
        pendingExamples.append(example)
        if currentDataset != nil {
            currentDataset?.addExample(example)
        }
        objectWillChange.send()
        savePendingExamples()
    }

    // MARK: - Failed Film Management

    /// Store a film that did not meet quality requirements. Never added to `pendingExamples`.
    func addFailedFilm(_ film: FailedHandFilm) {
        failedExamples.append(film)
        saveFailedExamples()
    }

    /// Promote a failed film to a valid training example (manual override).
    func validateFailedFilm(id: UUID) {
        guard let idx = failedExamples.firstIndex(where: { $0.id == id }) else { return }
        var film = failedExamples[idx]
        film.isManuallyValidated = true
        failedExamples.remove(at: idx)
        saveFailedExamples()
        let example = TrainingExample(
            handfilm: film.handfilm,
            gestureId: film.gestureId,
            userId: nil,
            sessionId: UUID().uuidString
        )
        addTrainingExample(example)
    }

    /// Permanently delete a failed film.
    func deleteFailedFilm(id: UUID) {
        failedExamples.removeAll { $0.id == id }
        saveFailedExamples()
    }

    /// Upload all pending examples to the server.
    func sendPendingToServer() {
        guard !pendingExamples.isEmpty, !isSendingToServer, let client = apiClient else { return }
        isSendingToServer = true
        uploadState = .uploading
        let batch = pendingExamples
        pendingExamples = []
        savePendingExamples()
        print("TrainingDataManager: uploading \(batch.count) example(s) to \(client.baseURL)")
        Task.detached { [weak self, weak client] in
            guard let client else { return }
            var lastTotal = 0
            for example in batch {
                do {
                    let response = try await client.uploadExample(example)
                    lastTotal = response.totalForGesture
                    print("TrainingDataManager: uploaded example id=\(response.id) gesture=\(example.gestureId) → server total for gesture: \(response.totalForGesture)")
                } catch {
                    print("TrainingDataManager: upload failed — \(error)")
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
            await self?.fetchServerExampleCounts()
        }
    }

    // MARK: - Server Sync

    /// Fetch per-gesture example counts from the server and update `serverExampleCounts`.
    /// Called automatically when `apiClient` is set and after a successful upload.
    func fetchServerExampleCounts() async {
        guard let client = apiClient else {
            print("TrainingDataManager: fetchServerExampleCounts skipped — apiClient is nil")
            return
        }
        do {
            let stats = try await client.fetchExampleStats()
            print("TrainingDataManager: server stats — total=\(stats.total), gestures=\(stats.gestures.map { "\($0.gestureId):\($0.count)" })")
            let counts = Dictionary(uniqueKeysWithValues: stats.gestures.map { ($0.gestureId, $0.count) })
            await MainActor.run {
                self.serverExampleCounts = counts
                self.serverSyncError = nil
            }
        } catch {
            print("TrainingDataManager: fetchServerExampleCounts error — \(error)")
            await MainActor.run { self.serverSyncError = error.localizedDescription }
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

    // MARK: - Example Mutation

    func deleteExample(id: UUID) {
        trainingExamples.removeAll { $0.id == id }
        pendingExamples.removeAll { $0.id == id }
        currentDataset?.removeExample(id: id)
        savePendingExamples()
    }

    func relabelExample(id: UUID, newGestureId: String) {
        if let idx = trainingExamples.firstIndex(where: { $0.id == id }) {
            let old = trainingExamples[idx]
            trainingExamples[idx] = TrainingExample(
                id: old.id,
                handfilm: old.handfilm,
                gestureId: newGestureId,
                userId: old.userId,
                sessionId: old.sessionId
            )
        }
        if let idx = pendingExamples.firstIndex(where: { $0.id == id }) {
            let old = pendingExamples[idx]
            pendingExamples[idx] = TrainingExample(
                id: old.id,
                handfilm: old.handfilm,
                gestureId: newGestureId,
                userId: old.userId,
                sessionId: old.sessionId
            )
        }
        currentDataset?.relabelExample(id: id, newGestureId: newGestureId)
        savePendingExamples()
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

    // MARK: - Pending & Failed Persistence

    func savePendingExamples() {
        let url = pendingExamplesFileURL()
        do {
            let dtos = pendingExamples.map { TrainingExampleDTO(from: $0) }
            let data = try JSONEncoder().encode(dtos)
            try data.write(to: url, options: .atomic)
        } catch {
            print("TrainingDataManager: failed to save pending examples — \(error)")
        }
    }

    private func loadPendingExamples() {
        let url = pendingExamplesFileURL()
        guard let data = try? Data(contentsOf: url),
              let dtos = try? JSONDecoder().decode([TrainingExampleDTO].self, from: data) else {
            return
        }
        let examples = dtos.map { $0.toTrainingExample() }
        pendingExamples = examples
        trainingExamples = examples
    }

    func saveFailedExamples() {
        let url = failedExamplesFileURL()
        do {
            let dtos = failedExamples.map { FailedHandFilmDTO(from: $0) }
            let data = try JSONEncoder().encode(dtos)
            try data.write(to: url, options: .atomic)
        } catch {
            print("TrainingDataManager: failed to save failed examples — \(error)")
        }
    }

    private func loadFailedExamples() {
        let url = failedExamplesFileURL()
        guard let data = try? Data(contentsOf: url),
              let dtos = try? JSONDecoder().decode([FailedHandFilmDTO].self, from: data) else {
            return
        }
        failedExamples = dtos.map { $0.toFailedHandFilm() }
    }

    // MARK: - File Paths

    private func datasetsDirectory() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("TrainingDatasets", isDirectory: true)
    }

    private func datasetFileURL(name: String) -> URL {
        datasetsDirectory().appendingPathComponent("\(name).json")
    }

    private func pendingExamplesFileURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("pending_examples.json")
    }

    private func failedExamplesFileURL() -> URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("failed_examples.json")
    }
}

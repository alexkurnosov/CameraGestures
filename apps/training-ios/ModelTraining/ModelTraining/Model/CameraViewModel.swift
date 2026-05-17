import Foundation
import Combine
import HandGestureTypes
import HandGestureRecognizingFramework
import GestureModelModule

@MainActor
class CameraViewModel: ObservableObject {

    // MARK: - Recognition state

    @Published var isRecognitionActive = false
    @Published var currentGesture: DetectedGesture?
    @Published var recentGestures: [DetectedGesture] = []
    @Published var recognitionHandPoints: [Point3D] = []
    @Published var stats = GestureRecognizingStats()
    // MARK: - Permissions & banners

    @Published var cameraPermissionGranted = false
    @Published var showModelNotTrainedBanner = false

    // MARK: - Training series config

    @Published var captureWindow: TimeInterval = 2.0
    @Published var pauseInterval: TimeInterval = 2.0

    // MARK: - Series coordinator

    let seriesCoordinator = TrainingSeriesCoordinator()

    // MARK: - Dependencies (set via configure)

    private(set) weak var gestureRecognizer: GestureRecognizerWrapper?
    private(set) weak var trainingDataManager: TrainingDataManager?
    private(set) weak var appSettings: AppSettings?
    private(set) weak var gestureRegistry: GestureRegistry?

    private var cancellables = Set<AnyCancellable>()
    private var statsTask: Task<Void, Never>?
    private var isConfigured = false

    // MARK: - Init

    init() {
        // SwiftUI does not observe nested ObservableObjects automatically.
        // Forward seriesCoordinator changes so the view redraws.
        seriesCoordinator.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)
    }

    // MARK: - Computed Properties

    var displayPoints: [Point3D] {
        seriesCoordinator.isRunning
            ? seriesCoordinator.handTrackingPoints
            : recognitionHandPoints
    }

    var previewIsActive: Bool { isRecognitionActive || seriesCoordinator.isRunning }

    var isModelTrained: Bool {
#if DEBUG
        //Test:
        return true
#else
#error ("test")
#endif
        guard let path = appSettings?.modelConfig.modelPath else { return false }
        return FileManager.default.fileExists(atPath: path)
    }

    var canStartTraining: Bool {
        cameraPermissionGranted && trainingDataManager?.selectedGesture != nil
    }

    // MARK: - Configuration

    func configure(
        recognizer: GestureRecognizerWrapper,
        dataManager: TrainingDataManager,
        settings: AppSettings,
        registry: GestureRegistry,
        apiClient: GestureModelAPIClient
    ) {
        guard !isConfigured else { return }
        isConfigured = true

        self.gestureRecognizer = recognizer
        self.trainingDataManager = dataManager
        self.appSettings = settings
        self.gestureRegistry = registry

        settings.updateModelConfig()
        showModelNotTrainedBanner = !isModelTrained

        if dataManager.selectedGesture == nil {
            dataManager.selectedGesture = registry.gestures.first
        }
        dataManager.apiClient = apiClient

        setupGestureSubscriptions()
    }

    // MARK: - Actions

    func startRecognition() {
        guard let gestureRecognizer else { return }
        guard isModelTrained else {
            showModelNotTrainedBanner = true
            return
        }
        recentGestures.removeAll()
        gestureRecognizer.recognizer.gateEnabled = true
        gestureRecognizer.recognizer.bypassPhase2Filter = appSettings?.bypassPhase2Filter ?? false
        gestureRecognizer.recognizer.resetGateState()
        Task {
            do {
                try await gestureRecognizer.recognizer.start()
                isRecognitionActive = true
            } catch {
                print("Failed to start recognition: \(error)")
            }
        }
    }

    func startTrainingSeries() {
        guard let gestureRecognizer, let trainingDataManager, let appSettings else { return }
        guard let gesture = trainingDataManager.selectedGesture else { return }
        trainingDataManager.startDataCollection(for: gesture)

        Task {
            if !gestureRecognizer.recognizer.isActive {
                do {
                    try await gestureRecognizer.recognizer.start()
                } catch {
                    print("Failed to start recognizer for training: \(error)")
                    trainingDataManager.stopDataCollection()
                    return
                }
            }
            seriesCoordinator.start(
                using: gestureRecognizer.recognizer,
                captureWindow: captureWindow,
                pauseInterval: pauseInterval,
                minInViewDuration: appSettings.minInViewDuration,
                gestureId: gesture.id,
                onFilmCaptured: { film in
                    let example = TrainingExample(
                        handfilm: film,
                        gestureId: gesture.id,
                        userId: "current_user",
                        sessionId: UUID().uuidString
                    )
                    trainingDataManager.addTrainingExample(example)
                },
                onFilmFailed: { failedFilm, _ in
                    trainingDataManager.addFailedFilm(failedFilm)
                }
            )
        }
    }

    func stopAll() {
        if seriesCoordinator.isRunning {
            seriesCoordinator.stop()
            trainingDataManager?.stopDataCollection()
        }
        gestureRecognizer?.recognizer.stop()
        isRecognitionActive = false
    }

    func clearGestures() {
        recentGestures.removeAll()
        currentGesture = nil
        gestureRecognizer?.recognizer.clearHistory()
    }

    func checkCameraPermission() {
        Task {
            let permission = await HandsRecognizing.requestCameraPermission()
            cameraPermissionGranted = permission
        }
    }

    func handleGestureRegistryChange(_ gestures: [GestureDefinition]) {
        guard let trainingDataManager else { return }
        if let current = trainingDataManager.selectedGesture, !gestures.contains(current) {
            trainingDataManager.selectedGesture = gestures.first
        } else if trainingDataManager.selectedGesture == nil {
            trainingDataManager.selectedGesture = gestures.first
        }
    }

    // MARK: - Subscriptions

    private func setupGestureSubscriptions() {
        guard let gestureRecognizer else { return }

        gestureRecognizer.gestureDetected
            .receive(on: DispatchQueue.main)
            .sink { [weak self] gesture in
                self?.currentGesture = gesture
                self?.recentGestures.append(gesture)
                if (self?.recentGestures.count ?? 0) > 50 {
                    self?.recentGestures.removeFirst()
                }
            }
            .store(in: &cancellables)

        gestureRecognizer.handTrackingUpdate
            .receive(on: DispatchQueue.main)
            .sink { [weak self] handshot in
                self?.recognitionHandPoints = handshot.landmarks
            }
            .store(in: &cancellables)

        statsTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard let self, self.isRecognitionActive, let recognizer = self.gestureRecognizer else { continue }
                self.stats = recognizer.recognizer.getStatistics()
            }
        }
    }

    deinit {
        statsTask?.cancel()
    }
}

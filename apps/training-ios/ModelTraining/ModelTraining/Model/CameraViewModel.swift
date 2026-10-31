import Foundation
import Combine
import CameraGestures

@MainActor
class CameraViewModel: ObservableObject {

    // MARK: - Recognition state

    @Published var isRecognitionActive = false
    @Published var currentGesture: DetectedGesture?
    @Published var recentGestures: [DetectedGesture] = []
    @Published var recognitionHandPoints: [Point3D] = []
    @Published var stats = GestureRecognizingStats()
    /// Landmark-path cadence, polled once a second while recognition runs.
    /// Surfaced only behind the debug switch; see `CameraView.frameCadencePanel`.
    @Published var frameRateStats = FrameRateStats()
    /// Added to every cadence log line of the current run, so a scripted
    /// measurement can name its configurations. Nil for an ordinary run.
    var cadenceLabel: String?
    // MARK: - Permissions & banners

    @Published var cameraPermissionGranted = false
    @Published var showModelNotTrainedBanner = false

    // MARK: - Training series config

    @Published var captureWindow: TimeInterval = 2.0
    @Published var pauseInterval: TimeInterval = 2.0

    #if CG_SESSION_CAPTURE
    // MARK: - Session capture (Show Debug Info only)

    /// Armed for the next Start Prediction. Not saved, so a relaunch disarms it.
    @Published var recordSession = false
    @Published var recordWithVideo = false
    /// Why the last recording did not start or ended early. Cleared on the next start.
    @Published var sessionCaptureMessage: String?
    #endif

    // MARK: - Series coordinator

    let seriesCoordinator = TrainingSeriesCoordinator()

    // MARK: - Dependencies (set via configure)

    private(set) weak var gestureRecognizer: GestureRecognizerWrapper?
    private(set) weak var trainingDataManager: TrainingDataManager?
    private(set) weak var appSettings: AppSettings?
    private(set) weak var gestureRegistry: GestureRegistry?

    private var cancellables = Set<AnyCancellable>()
    private var statsTask: Task<Void, Never>?

    // Cadence log state for the current prediction run (see CadenceLog).
    private var cadenceRun: String?
    private var cadenceThermalSeen: [String] = []
    private var cadenceSeconds = 0
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
                cadenceRun = ISO8601DateFormatter().string(from: Date())
                cadenceThermalSeen = [CadenceLog.name(ProcessInfo.processInfo.thermalState)]
                cadenceSeconds = 0
                logCadence("start")
            } catch {
                print("Failed to start recognition: \(error)")
                #if CG_SESSION_CAPTURE
                discardUnstartedSession()
                #endif
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
        // Before the recognizer stops, while the counters still describe the run.
        if isRecognitionActive { logCadence("stop") }
        if seriesCoordinator.isRunning {
            seriesCoordinator.stop()
            trainingDataManager?.stopDataCollection()
        }
        gestureRecognizer?.recognizer.stop()
        isRecognitionActive = false
        #if CG_SESSION_CAPTURE
        // stop() has ended the session.
        gestureRecognizer?.recordingSessionURL = nil
        #endif
    }

    #if CG_SESSION_CAPTURE
    /// Start Prediction with Record session armed. The recorder starts right
    /// before prediction, so the session spans exactly this run. Declining
    /// consent starts nothing.
    func startRecognitionWithSession() {
        guard let gestureRecognizer, let appSettings, isModelTrained else { return }
        let includesVideo = recordWithVideo
        Task {
            guard let consent = await SessionCaptureConsent.request(includesVideo: includesVideo) else {
                sessionCaptureMessage = "Consent declined. Nothing was started or recorded."
                return
            }
            do {
                gestureRecognizer.recordingSessionURL = try SessionRecorder(recognizer: gestureRecognizer.recognizer)
                    .start(consent: consent,
                           modelFiles: appSettings.sessionModelFiles(),
                           minimumFreeBytes: appSettings.sessionMinimumFreeBytes)
            } catch SessionCaptureError.notEnoughSpace(let available, let minimum) {
                sessionCaptureMessage = "Not started: \(Self.bytes(available)) free, "
                    + "below the \(Self.bytes(minimum)) limit in Settings."
                return
            } catch {
                sessionCaptureMessage = "Recording could not start: \(error)"
                return
            }
            sessionCaptureMessage = nil
            startRecognition()
        }
    }

    /// Prediction failed to start after the recorder did: drop the empty session.
    private func discardUnstartedSession() {
        guard let gestureRecognizer, let url = gestureRecognizer.recordingSessionURL else { return }
        SessionRecorder(recognizer: gestureRecognizer.recognizer).stop()
        try? FileManager.default.removeItem(at: url)
        gestureRecognizer.recordingSessionURL = nil
        sessionCaptureMessage = "Prediction did not start, so the recording was discarded."
    }

    /// A session ended on its own. It spans one run, so the run ends too.
    private func handleSessionCaptureStopped(_ reason: SessionCaptureStopReason) {
        if isRecognitionActive { stopAll() }
        gestureRecognizer?.recordingSessionURL = nil
        switch reason {
        case .lowDiskSpace(let available):
            sessionCaptureMessage = "Recording stopped: \(Self.bytes(available)) free, below the limit "
                + "in Settings. The session is saved up to this point."
        case .writeFailed(let error):
            sessionCaptureMessage = "Recording stopped: a write failed (\(error)). "
                + "The session is readable up to the last part that was saved."
        }
    }

    private static func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
    }
    #endif

    /// Restarts the cadence measurement window without restarting the camera —
    /// used to discard the warm-up frames at the start of a baseline run.
    func resetFrameRateStats() {
        gestureRecognizer?.recognizer.resetFrameRateStats()
        frameRateStats = FrameRateStats()
        if isRecognitionActive { logCadence("reset") }
    }

    /// One cadence log line: the counters since start or the last reset, the
    /// thermal state, and the camera exposure, which shows when low light
    /// stretches the frames.
    private func logCadence(_ kind: String) {
        guard let recognizer = gestureRecognizer?.recognizer else { return }
        var fields: [String: Any] = [
            "kind":         kind,
            "run":          cadenceRun ?? "",
            "thermal":      CadenceLog.name(ProcessInfo.processInfo.thermalState),
            "thermal_seen": cadenceThermalSeen,
        ]
        if let cadenceLabel { fields["label"] = cadenceLabel }
        fields.merge(CadenceLog.fields(recognizer.getFrameRateStats())) { a, _ in a }
        fields.merge(CadenceLog.fields(recognizer.getCameraExposure())) { a, _ in a }
        CadenceLog.shared.write("CG:Cadence", fields)
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

        #if CG_SESSION_CAPTURE
        gestureRecognizer.sessionCaptureStopped
            .receive(on: DispatchQueue.main)
            .sink { [weak self] reason in self?.handleSessionCaptureStopped(reason) }
            .store(in: &cancellables)
        #endif

        statsTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard let self, self.isRecognitionActive, let recognizer = self.gestureRecognizer else { continue }
                self.stats = recognizer.recognizer.getStatistics()
                self.frameRateStats = recognizer.recognizer.getFrameRateStats()
                self.cadenceSeconds += 1
                if self.cadenceSeconds % 10 == 0 { self.logCadence("tick") }
            }
        }

        NotificationCenter.default.publisher(for: ProcessInfo.thermalStateDidChangeNotification)
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                guard let self, self.isRecognitionActive else { return }
                let state = CadenceLog.name(ProcessInfo.processInfo.thermalState)
                if self.cadenceThermalSeen.last != state { self.cadenceThermalSeen.append(state) }
                self.logCadence("thermal")
            }
            .store(in: &cancellables)
    }

    deinit {
        statsTask?.cancel()
    }
}

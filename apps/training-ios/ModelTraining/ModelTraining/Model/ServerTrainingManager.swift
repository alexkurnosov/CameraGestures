import Foundation
import Combine
import CameraGestures
import CameraGestures
import CameraGestures

class ServerTrainingManager: ObservableObject {

    // MARK: - Published State

    @Published var serverStatus: ModelStatusResponse?
    @Published var isPollingStatus = false
    @Published var isDownloadingModel = false
    @Published var isWipingModel = false
    @Published var serverActionError: String?

    // Last-training metrics fetched from the server (refreshed on start, after train, after download)
    @Published var phase3Metrics: ModelMetricsResponse?
    @Published var poseMetrics: PoseMetricsResponse?

    // MARK: - Dependencies

    private(set) weak var apiClient: GestureModelAPIClient?
    private(set) weak var appSettings: AppSettings?
    private(set) weak var gestureRecognizer: GestureRecognizerWrapper?

    private var statusPollingTask: Task<Void, Never>?

    // MARK: - Configuration

    func configure(
        apiClient: GestureModelAPIClient,
        appSettings: AppSettings,
        gestureRecognizer: GestureRecognizerWrapper
    ) {
        self.apiClient = apiClient
        self.appSettings = appSettings
        self.gestureRecognizer = gestureRecognizer
    }

    // MARK: - Server Actions

    func refreshServerStatus() {
        guard let apiClient else { return }
        Task {
            do {
                let status = try await apiClient.fetchModelStatus()
                await MainActor.run { serverStatus = status }
                if status.status == "training" {
                    startPollingStatus()
                }
            } catch {
                print("[ServerTrainingManager] fetchModelStatus failed: \(error)")
            }
            // Run sequentially so auth is already established by fetchModelStatus above.
            refreshAllMetrics()
        }
    }

    func refreshAllMetrics() {
        guard let apiClient else { return }
        Task {
            // Start both concurrently; handle each error independently so one
            // 404 (e.g. no pose model yet) doesn't suppress the other result.
            async let p3Fetch = apiClient.fetchLatestMetrics()
            async let p2Fetch = apiClient.fetchPoseMetrics()
            var m3: ModelMetricsResponse? = nil
            var m2: PoseMetricsResponse? = nil
            do { m3 = try await p3Fetch } catch { print("[ServerTrainingManager] fetchLatestMetrics: \(error)") }
            do { m2 = try await p2Fetch } catch { print("[ServerTrainingManager] fetchPoseMetrics: \(error)") }
            await MainActor.run {
                if let m3 { phase3Metrics = m3 }
                if let m2 { poseMetrics = m2 }
            }
        }
    }

    func triggerServerTraining() {
        guard let apiClient, let appSettings else { return }
        Task {
            do {
                let job = try await apiClient.triggerTraining(
                    minInViewDuration: appSettings.minInViewDuration,
                    balanceStrategy: appSettings.balanceStrategy.rawValue,
                    geomCoef: appSettings.geomCoef
                )
                print("[ServerTrainingManager] Training job started: \(job.jobId)")
                appSettings.lockThresholdIfNeeded()
                startPollingStatus()
            } catch {
                serverActionError = error.localizedDescription
            }
        }
    }

    func downloadModelFromServer() {
        guard let apiClient, let appSettings, let gestureRecognizer else { return }
        isDownloadingModel = true
        Task {
            do {
                async let modelURL = apiClient.downloadModel()
                async let preprocessorURL = apiClient.downloadPreprocessor()

                let (mURL, pURL) = try await (modelURL, preprocessorURL)

                // Load preprocessor first so JS-exported constants (POSE_VECTOR_SIZE etc.)
                // are up-to-date before the model's tensor shapes are validated.
                try JSPreprocessorWrapper.shared.load(from: pURL)
                print("[ServerTrainingManager] Preprocessor loaded: version=\(JSPreprocessorWrapper.shared.preprocVersion) poseVectorSize=\(JSPreprocessorWrapper.shared.poseVectorSize) summaryFeaturesCount=\(JSPreprocessorWrapper.shared.summaryFeaturesCount)")

                appSettings.updateModelConfig()
                let sidecarURL = mURL.deletingLastPathComponent().appendingPathComponent("gesture_ids.json")
                let gestureIds = (try? JSONDecoder().decode([String].self, from: Data(contentsOf: sidecarURL))) ?? []
                // loadModel validates tensor input shape against summaryFeaturesCount — throws if mismatched.
                try gestureRecognizer.recognizer.loadModel(from: mURL.path, gestureIds: gestureIds)
                await MainActor.run { appSettings.gestureModelLoadedAt = Date() }

                // Pose model — report failure but don't abort the main model update.
                do {
                    try await apiClient.downloadPoseModel()
                    gestureRecognizer.loadPoseModelIfAvailable(appSettings: appSettings)
                    await MainActor.run { appSettings.poseModelLoadedAt = Date() }
                } catch {
                    serverActionError = "Pose model: \(error.localizedDescription)"
                }
            } catch {
                serverActionError = error.localizedDescription
            }
            isDownloadingModel = false
            refreshAllMetrics()
        }
    }

    func wipeServerModel() {
        guard let apiClient, let appSettings else { return }
        isWipingModel = true
        Task {
            do {
                try await apiClient.wipeModel()
                serverStatus = nil
                appSettings.updateModelConfig()
                appSettings.isThresholdLocked = false
            } catch {
                serverActionError = error.localizedDescription
            }
            isWipingModel = false
        }
    }

    func startPollingStatus() {
        guard let apiClient else { return }
        statusPollingTask?.cancel()
        isPollingStatus = true
        statusPollingTask = Task {
            while !Task.isCancelled {
                do {
                    try await Task.sleep(nanoseconds: 3_000_000_000)
                    guard !Task.isCancelled else { break }
                    let status = try await apiClient.fetchModelStatus()
                    await MainActor.run { serverStatus = status }
                    if status.status != "training" {
                        refreshAllMetrics()
                        break
                    }
                } catch {
                    break
                }
            }
            await MainActor.run { isPollingStatus = false }
        }
    }

    func stopPolling() {
        statusPollingTask?.cancel()
        statusPollingTask = nil
    }

    deinit {
        statusPollingTask?.cancel()
    }
}

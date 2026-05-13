import Foundation
import Combine
import HandGestureTypes
import HandGestureRecognizingFramework
import GestureModelModule

class ServerTrainingManager: ObservableObject {

    // MARK: - Published State

    @Published var serverStatus: ModelStatusResponse?
    @Published var isPollingStatus = false
    @Published var isDownloadingModel = false
    @Published var isWipingModel = false
    @Published var serverActionError: String?

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
                serverStatus = status
                if status.status == "training" {
                    startPollingStatus()
                }
            } catch {
                print("[ServerTrainingManager] fetchModelStatus failed: \(error)")
            }
        }
    }

    func triggerServerTraining() {
        guard let apiClient, let appSettings else { return }
        Task {
            do {
                let job = try await apiClient.triggerTraining(
                    minInViewDuration: appSettings.minInViewDuration,
                    balanceStrategy: appSettings.balanceStrategy.rawValue
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

                // Pose model — report failure but don't abort the main model update.
                do {
                    try await apiClient.downloadPoseModel()
                    gestureRecognizer.loadPoseModelIfAvailable(appSettings: appSettings)
                } catch {
                    serverActionError = "Pose model: \(error.localizedDescription)"
                }
            } catch {
                serverActionError = error.localizedDescription
            }
            isDownloadingModel = false
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
                    serverStatus = status
                    if status.status != "training" { break }
                } catch {
                    break
                }
            }
            isPollingStatus = false
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

import Foundation
import Combine
import HandGestureTypes
import HandGestureRecognizingFramework

/// Drives a repeating series of HandFilm captures for training data collection.
/// Each iteration: countdown → recording window → pause → repeat.
///
/// At the end of each recording window the accumulated film is harvested and
/// evaluated against `minInViewDuration`. Films that pass are delivered to
/// `onFilmCaptured`; films that fail are delivered to `onFilmFailed`.
@MainActor
class TrainingSeriesCoordinator: ObservableObject {

    // MARK: - Configurable timing
    var captureWindow: TimeInterval = 1.0
    var pauseInterval: TimeInterval = 5.0
    var countdownDuration: Int = 3
    /// Minimum seconds of non-absent frames required for a film to be valid.
    var minInViewDuration: TimeInterval = 1.2

    // MARK: - Capture phase
    enum Phase: Equatable {
        case idle
        case countdown(remaining: Int)
        case recording
        case pause(remaining: Int)
    }

    @Published var phase: Phase = .idle
    @Published var capturedCount: Int = 0
    @Published var failedCount: Int = 0
    @Published var handTrackingPoints: [Point3D] = []
    /// `true` during the recording phase when the most recent frame had no hand.
    @Published var isHandAbsent: Bool = false

    private weak var gestureRecognizer: HandGestureRecognizing?
    private var seriesTask: Task<Void, Never>?
    private var onFilmCaptured: ((HandFilm) -> Void)?
    private var onFilmFailed: ((FailedHandFilm, String) -> Void)?

    var isRunning: Bool { phase != .idle }

    // MARK: - Start / Stop

    func start(
        using recognizer: HandGestureRecognizing,
        captureWindow: TimeInterval,
        pauseInterval: TimeInterval,
        minInViewDuration: TimeInterval,
        gestureId: String,
        onFilmCaptured: @escaping (HandFilm) -> Void,
        onFilmFailed: @escaping (FailedHandFilm, String) -> Void
    ) {
        stop()
        self.gestureRecognizer = recognizer
        self.captureWindow = captureWindow
        self.pauseInterval = pauseInterval
        self.minInViewDuration = minInViewDuration
        self.onFilmCaptured = onFilmCaptured
        self.onFilmFailed = onFilmFailed
        capturedCount = 0
        failedCount = 0

        recognizer.handshotCallback = { [weak self] handshot in
            DispatchQueue.main.async {
                self?.handTrackingPoints = handshot.landmarks
                self?.isHandAbsent = handshot.isAbsent
            }
        }

        seriesTask = Task { await self.runLoop(gestureId: gestureId) }
    }

    func stop() {
        seriesTask?.cancel()
        seriesTask = nil
        gestureRecognizer?.handshotCallback = nil
        gestureRecognizer?.handfilmCallback = nil
        gestureRecognizer = nil
        phase = .idle
        handTrackingPoints = []
        isHandAbsent = false
    }

    // MARK: - Series Loop

    private func runLoop(gestureId: String) async {
        while !Task.isCancelled {
            // --- Countdown ---
            for remaining in stride(from: countdownDuration, through: 1, by: -1) {
                guard !Task.isCancelled else { return }
                phase = .countdown(remaining: remaining)
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
            guard !Task.isCancelled else { return }

            // --- Recording ---
            // Reset the buffer so countdown/pause frames don't contaminate the capture.
            gestureRecognizer?.resetHandfilm()
            phase = .recording
            isHandAbsent = false

            // Sleep for the full capture window, then harvest whatever was recorded.
            let windowNs = UInt64(captureWindow * 1_000_000_000)
            try? await Task.sleep(nanoseconds: windowNs)
            guard !Task.isCancelled else { return }

            let film = gestureRecognizer?.harvestHandfilm() ?? HandFilm()
            evaluateAndDispatch(film: film, gestureId: gestureId)

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

    private func evaluateAndDispatch(film: HandFilm, gestureId: String) {
        let inView = film.inViewDuration
        if inView >= minInViewDuration {
            capturedCount += 1
            onFilmCaptured?(film)
        } else {
            failedCount += 1
            let detail = String(
                format: "%.1fs in-view, need ≥%.1fs (gesture: %.1fs total)",
                inView, minInViewDuration, film.gestureDuration
            )
            let failed = FailedHandFilm(
                handfilm: film,
                gestureId: gestureId,
                failureReason: .insufficientInViewDuration,
                failureDetail: detail
            )
            onFilmFailed?(failed, gestureId)
        }
    }
}

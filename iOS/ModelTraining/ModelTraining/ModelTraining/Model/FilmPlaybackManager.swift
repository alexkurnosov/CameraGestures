import Foundation
import Combine
import HandGestureTypes

@MainActor
class FilmPlaybackManager: ObservableObject {

    // MARK: - Published State

    @Published var currentFrameIndex: Int = 0
    @Published var isPlaying: Bool = false
    @Published var currentIndex: Int = 0
    @Published var filterGestureId: String? = nil

    // MARK: - Dependencies (set via configure)

    private(set) weak var trainingDataManager: TrainingDataManager?

    private var playTimer: Timer?
    private var isConfigured = false

    // MARK: - Configuration

    func configure(dataManager: TrainingDataManager) {
        guard !isConfigured else { return }
        isConfigured = true
        self.trainingDataManager = dataManager
    }

    // MARK: - Computed Properties

    var filteredExamples: [TrainingExample] {
        guard let trainingDataManager else { return [] }
        return trainingDataManager.trainingExamples
            .filter { filterGestureId == nil || $0.gestureId == filterGestureId }
            .sorted { $0.handfilm.startTime > $1.handfilm.startTime }
    }

    var currentExample: TrainingExample? {
        let examples = filteredExamples
        guard !examples.isEmpty, currentIndex < examples.count else { return nil }
        return examples[currentIndex]
    }

    var currentFilm: HandFilm? { currentExample?.handfilm }

    var frameCount: Int { currentFilm?.frames.count ?? 0 }

    var currentPoints: [Point3D] {
        guard let film = currentFilm, !film.frames.isEmpty else { return [] }
        let safe = min(currentFrameIndex, film.frames.count - 1)
        return film.frames[safe].landmarks
    }

    var currentHandedness: String {
        guard let film = currentFilm, let first = film.frames.first else { return "—" }
        switch first.leftOrRight {
        case .left: return "Left"
        case .right: return "Right"
        case .unknown: return "Unknown"
        }
    }

    // MARK: - Playback

    func startPlayback() {
        guard frameCount > 1 else { return }
        isPlaying = true
        playTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 24.0, repeats: true) { [weak self] _ in
            DispatchQueue.main.async {
                guard let self else { return }
                if self.currentFrameIndex < self.frameCount - 1 {
                    self.currentFrameIndex += 1
                } else {
                    self.stopPlayback()
                }
            }
        }
    }

    func stopPlayback() {
        isPlaying = false
        playTimer?.invalidate()
        playTimer = nil
    }

    // MARK: - Navigation

    func goToPrevious() {
        stopPlayback()
        if currentIndex > 0 {
            currentIndex -= 1
        }
    }

    func goToNext() {
        stopPlayback()
        let count = filteredExamples.count
        if currentIndex < count - 1 {
            currentIndex += 1
        }
    }

    func setFilter(_ gestureId: String?) {
        stopPlayback()
        filterGestureId = gestureId
        currentIndex = 0
        currentFrameIndex = 0
    }

    func onCurrentIndexChanged() {
        stopPlayback()
        currentFrameIndex = 0
    }

    // MARK: - Actions

    func deleteCurrentExample() {
        guard let example = currentExample else { return }
        let total = filteredExamples.count
        trainingDataManager?.deleteExample(id: example.id)
        if currentIndex >= total - 1 {
            currentIndex = max(0, total - 2)
        }
        currentFrameIndex = 0
    }
}

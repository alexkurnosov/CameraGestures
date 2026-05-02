import Combine
import SwiftUI
import HandGestureTypes

struct HandFilmsView: View {
    @EnvironmentObject var trainingDataManager: TrainingDataManager
    @EnvironmentObject var gestureRegistry: GestureRegistry
    @EnvironmentObject var apiClient: GestureModelAPIClient

    @StateObject private var playbackManager = FilmPlaybackManager()
    @StateObject private var holdVM = HoldInspectorViewModel()

    @State private var showingRelabelSheet = false
    @State private var showingDeleteAlert = false
    @State private var showFailedFilms = false
    @State private var failedFilmToValidate: FailedHandFilm? = nil
    @State private var showingValidateAlert = false
    @State private var failedFilmToDelete: FailedHandFilm? = nil
    @State private var showingFailedDeleteAlert = false
    @State private var showingClearAllFailedAlert = false

    // MARK: - Display helpers

    private var currentGestureName: String {
        guard let example = playbackManager.currentExample else { return "—" }
        return gestureRegistry.gestures.first { $0.id == example.gestureId }?.name ?? example.gestureId
    }

    private var filterLabel: String {
        guard let id = playbackManager.filterGestureId else { return "All" }
        return gestureRegistry.gestures.first { $0.id == id }?.name ?? id
    }

    // MARK: - Body

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                if playbackManager.filteredExamples.isEmpty {
                    emptyState
                } else {
                    skeletonSection
                    Divider()
                    bottomPanel
                    holdInspectorSection
                }

                failedFilmsSection
            }
        }
        .navigationTitle("Collected Films")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                filterMenu
            }
        }
        .onAppear {
            playbackManager.configure(dataManager: trainingDataManager)
        }
        .onDisappear {
            playbackManager.stopPlayback()
            holdVM.clear()
        }
        .task(id: playbackManager.currentExample?.id) {
            guard let film = playbackManager.currentFilm,
                  let filmId = playbackManager.currentExample?.id.uuidString else {
                holdVM.clear()
                return
            }
            await holdVM.fetch(film: film, filmId: filmId, using: apiClient)
        }
        .sheet(isPresented: $showingRelabelSheet) {
            RelabelSheet(
                currentGestureId: playbackManager.currentExample?.gestureId ?? "",
                onSelect: { newId in
                    if let example = playbackManager.currentExample {
                        trainingDataManager.relabelExample(id: example.id, newGestureId: newId)
                    }
                }
            )
            .environmentObject(gestureRegistry)
        }
        .alert("Delete Film?", isPresented: $showingDeleteAlert) {
            Button("Delete", role: .destructive) { playbackManager.deleteCurrentExample() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This HandFilm will be permanently removed from the local collection.")
        }
        .alert("Mark as Valid?", isPresented: $showingValidateAlert) {
            Button("Mark as Valid", role: .none) {
                if let film = failedFilmToValidate {
                    trainingDataManager.validateFailedFilm(id: film.id)
                    failedFilmToValidate = nil
                }
            }
            Button("Cancel", role: .cancel) { failedFilmToValidate = nil }
        } message: {
            Text("This film will be added to your training collection despite not meeting the quality threshold.")
        }
        .alert("Clear All Failed Films?", isPresented: $showingClearAllFailedAlert) {
            Button("Clear All", role: .destructive) {
                trainingDataManager.deleteAllFailedFilms()
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("All failed films will be permanently removed.")
        }
        .alert("Delete Failed Film?", isPresented: $showingFailedDeleteAlert) {
            Button("Delete", role: .destructive) {
                if let film = failedFilmToDelete {
                    trainingDataManager.deleteFailedFilm(id: film.id)
                    failedFilmToDelete = nil
                }
            }
            Button("Cancel", role: .cancel) { failedFilmToDelete = nil }
        } message: {
            Text("This failed film will be permanently removed.")
        }
    }

    // MARK: - Skeleton Section

    private var skeletonSection: some View {
        VStack(spacing: 8) {
            ZStack {
                Color.black.opacity(0.85)
                    .cornerRadius(12)
                HandSkeletonView(points: playbackManager.currentPoints)
                    .cornerRadius(12)

                if playbackManager.frameCount == 0 {
                    Text("No frames")
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .frame(minHeight: 260)
            .padding(.horizontal)
            .padding(.top, 8)

            if !holdVM.holds.isEmpty {
                GeometryReader { geo in
                    holdTimelineView(in: geo.size.width)
                }
                .frame(height: 20)
                .padding(.horizontal)
            }

            playerControls
                .padding(.horizontal)
                .padding(.bottom, 8)
        }
    }

    private func holdTimelineView(in width: CGFloat) -> some View {
        let n = CGFloat(max(playbackManager.frameCount - 1, 1))
        return ZStack(alignment: .leading) {
            Capsule()
                .fill(Color.gray.opacity(0.2))
                .frame(height: 6)

            ForEach(holdVM.holds) { hold in
                let x = width * CGFloat(hold.startFrame) / n
                let endX = width * CGFloat(hold.endFrame) / n
                Capsule()
                    .fill((hold.isEdge ? Color.orange : Color.blue).opacity(0.65))
                    .frame(width: max(endX - x, 4), height: 6)
                    .offset(x: x)
            }

            ForEach(holdVM.holds) { hold in
                let cx = width * CGFloat(hold.repFrame) / n
                Circle()
                    .fill(hold.isEdge ? Color.orange : Color.blue)
                    .frame(width: 10, height: 10)
                    .offset(x: cx - 5)
            }

            let px = min(width * CGFloat(playbackManager.currentFrameIndex) / n, width - 2)
            RoundedRectangle(cornerRadius: 1)
                .fill(Color.white.opacity(0.9))
                .frame(width: 2, height: 16)
                .offset(x: px)
        }
        .frame(height: 20)
    }

    private var playerControls: some View {
        VStack(spacing: 6) {
            if playbackManager.frameCount > 1 {
                Slider(
                    value: Binding(
                        get: { Double(playbackManager.currentFrameIndex) },
                        set: { newValue in
                            playbackManager.currentFrameIndex = Int(newValue)
                            if playbackManager.isPlaying { playbackManager.stopPlayback() }
                        }
                    ),
                    in: 0...Double(max(playbackManager.frameCount - 1, 1)),
                    step: 1
                )
            }

            HStack(spacing: 16) {
                Text("\(playbackManager.currentFrameIndex + 1) / \(playbackManager.frameCount)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .monospacedDigit()

                Spacer()

                Button {
                    playbackManager.isPlaying ? playbackManager.stopPlayback() : playbackManager.startPlayback()
                } label: {
                    Image(systemName: playbackManager.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title3)
                }
                .disabled(playbackManager.frameCount < 2)

                Spacer()

                if let film = playbackManager.currentFilm {
                    VStack(alignment: .trailing, spacing: 1) {
                        Text(String(format: "%.2fs total", film.gestureDuration))
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(String(format: "%.2fs in-view", film.inViewDuration))
                            .font(.caption2)
                            .foregroundColor(film.inViewDuration < film.gestureDuration * 0.6 ? .orange : .secondary)
                    }
                }
            }
        }
    }

    // MARK: - Bottom Panel

    private var bottomPanel: some View {
        VStack(spacing: 12) {
            // Prev / counter / Next
            HStack {
                Button {
                    playbackManager.goToPrevious()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                        Text("Prev")
                    }
                }
                .disabled(playbackManager.currentIndex <= 0)

                Spacer()

                Text("\(playbackManager.currentIndex + 1) / \(playbackManager.filteredExamples.count)")
                    .font(.subheadline.monospacedDigit())
                    .foregroundColor(.secondary)

                Spacer()

                Button {
                    playbackManager.goToNext()
                } label: {
                    HStack(spacing: 4) {
                        Text("Next")
                        Image(systemName: "chevron.right")
                    }
                }
                .disabled(playbackManager.currentIndex >= playbackManager.filteredExamples.count - 1)
            }
            .padding(.horizontal)

            Divider()

            // Gesture + action buttons
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Gesture")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(currentGestureName)
                        .font(.subheadline.weight(.medium))
                        .lineLimit(1)
                }

                Spacer()

                Button("Change Gesture") {
                    showingRelabelSheet = true
                }
                .font(.subheadline)
                .buttonStyle(.bordered)

                Button(role: .destructive) {
                    showingDeleteAlert = true
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.bordered)
                .tint(.red)
            }
            .padding(.horizontal)

            // Metadata row
            if let example = playbackManager.currentExample, let film = playbackManager.currentFilm {
                HStack(spacing: 16) {
                    metaChip(label: "Frames", value: "\(film.frames.count)")
                    metaChip(label: "Hand", value: playbackManager.currentHandedness)
                    metaChip(label: "Captured", value: relativeTime(example.timestamp))
                }
                .padding(.horizontal)
            }
        }
        .padding(.vertical, 12)
    }

    private func metaChip(label: String, value: String) -> some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(value)
                .font(.caption.weight(.medium))
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
        .background(Color.gray.opacity(0.08))
        .cornerRadius(8)
    }

    // MARK: - Hold Inspector (Stage 2)

    private var holdInspectorSection: some View {
        VStack(spacing: 0) {
            Divider()

            HStack(spacing: 6) {
                Text("Holds")
                    .font(.subheadline.weight(.semibold))
                if holdVM.isLoading {
                    ProgressView().scaleEffect(0.7)
                } else {
                    Text(holdVM.holds.isEmpty ? "None detected" : "(\(holdVM.holds.count))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                if let err = holdVM.error {
                    Text(err)
                        .font(.caption2)
                        .foregroundColor(.red)
                        .lineLimit(1)
                }
            }
            .padding(.horizontal)
            .padding(.vertical, 8)

            if !holdVM.holds.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(holdVM.holds) { hold in
                            holdThumbnail(hold: hold)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 12)
                }
            }
        }
    }

    private func holdThumbnail(hold: HoldInfo) -> some View {
        VStack(spacing: 4) {
            ZStack {
                Color.black.opacity(0.75)
                if let film = playbackManager.currentFilm, hold.repFrame < film.frames.count {
                    HandSkeletonView(points: film.frames[hold.repFrame].landmarks)
                }
                if hold.isEdge {
                    VStack {
                        HStack {
                            Spacer()
                            Text("edge")
                                .font(.system(size: 8, weight: .semibold))
                                .foregroundColor(.white)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 2)
                                .background(Color.orange.opacity(0.85))
                                .cornerRadius(4)
                                .padding(3)
                        }
                        Spacer()
                    }
                }
            }
            .frame(width: 80, height: 80)
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(hold.isEdge ? Color.orange.opacity(0.6) : Color.blue.opacity(0.5), lineWidth: 1.5)
            )

            Text("#\(hold.ordinal)")
                .font(.caption.weight(.medium))
            Text("f\(hold.repFrame)")
                .font(.caption2)
                .foregroundColor(.secondary)

            let isExcluded = holdVM.excludedOrdinals.contains(hold.ordinal)
            Button(isExcluded ? "Excluded" : "Exclude") {
                guard !isExcluded,
                      let filmId = playbackManager.currentExample?.id.uuidString else { return }
                Task {
                    await holdVM.excludeHold(hold, filmId: filmId, using: apiClient)
                }
            }
            .font(.caption2)
            .foregroundColor(isExcluded ? .secondary : .red)
            .buttonStyle(.bordered)
            .tint(isExcluded ? .gray : .red)
            .controlSize(.mini)
            .disabled(isExcluded || holdVM.isSaving)
        }
    }

    // MARK: - Filter Menu

    private var filterMenu: some View {
        Menu {
            Button {
                playbackManager.setFilter(nil)
            } label: {
                HStack {
                    Text("All")
                    if playbackManager.filterGestureId == nil { Image(systemName: "checkmark") }
                }
            }

            Divider()

            ForEach(gestureRegistry.gestures) { gesture in
                Button {
                    playbackManager.setFilter(gesture.id)
                } label: {
                    HStack {
                        Text(gesture.name)
                        if playbackManager.filterGestureId == gesture.id { Image(systemName: "checkmark") }
                    }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "line.3.horizontal.decrease.circle")
                Text(filterLabel)
                    .font(.subheadline)
            }
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "film.stack")
                .font(.system(size: 56))
                .foregroundColor(.gray)
            Text(playbackManager.filterGestureId == nil ? "No films collected yet" : "No films for this gesture")
                .font(.headline)
                .foregroundColor(.secondary)
            if playbackManager.filterGestureId != nil {
                Button("Show All") { playbackManager.setFilter(nil) }
                    .buttonStyle(.bordered)
            }
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Failed Films Section

    private var failedFilmsSection: some View {
        let failed = trainingDataManager.failedExamples
            .filter { playbackManager.filterGestureId == nil || $0.gestureId == playbackManager.filterGestureId }

        return VStack(spacing: 0) {
            if !failed.isEmpty {
                Divider().padding(.top, 8)

                HStack {
                    Button {
                        withAnimation { showFailedFilms.toggle() }
                    } label: {
                        HStack {
                            Image(systemName: showFailedFilms ? "chevron.down" : "chevron.right")
                                .font(.caption.weight(.semibold))
                                .foregroundColor(.secondary)
                            Text("Failed Films (\(failed.count))")
                                .font(.subheadline.weight(.medium))
                                .foregroundColor(.orange)
                            Spacer()
                            Text("Not sent to server")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                    .buttonStyle(.plain)

                    Button {
                        showingClearAllFailedAlert = true
                    } label: {
                        Text("Clear All")
                            .font(.caption.weight(.semibold))
                            .foregroundColor(.red)
                    }
                    .buttonStyle(.plain)
                    .padding(.trailing, 16)
                }
                .padding(.horizontal)
                .padding(.vertical, 10)

                if showFailedFilms {
                    ForEach(failed) { film in
                        FailedFilmRow(
                            film: film,
                            gestureName: gestureRegistry.gestures.first { $0.id == film.gestureId }?.name ?? film.gestureId
                        ) {
                            failedFilmToValidate = film
                            showingValidateAlert = true
                        } onDelete: {
                            failedFilmToDelete = film
                            showingFailedDeleteAlert = true
                        }
                        Divider().padding(.leading)
                    }
                }
            }
        }
    }

    // MARK: - Helpers

    private func relativeTime(_ timestamp: TimeInterval) -> String {
        let diff = Date().timeIntervalSince1970 - timestamp
        if diff < 60 { return "\(Int(diff))s ago" }
        if diff < 3600 { return "\(Int(diff / 60))m ago" }
        if diff < 86400 { return "\(Int(diff / 3600))h ago" }
        return "\(Int(diff / 86400))d ago"
    }
}

// MARK: - Hold Inspector View Model

@MainActor
private final class HoldInspectorViewModel: ObservableObject {
    @Published var holds: [HoldInfo] = []
    @Published var paramsHash: String = ""
    @Published var isLoading: Bool = false
    @Published var isSaving: Bool = false
    @Published var error: String? = nil
    @Published var excludedOrdinals: Set<Int> = []

    private var cachedExcludedHolds: [ExcludedHoldEntry] = []
    private var cachedClusterKinds: [String: String] = [:]
    private var currentFilmId: String? = nil

    func fetch(film: HandFilm, filmId: String, using client: GestureModelAPIClient) async {
        isLoading = true
        error = nil
        holds = []
        excludedOrdinals = []
        currentFilmId = filmId
        do {
            async let holdsTask = client.analyzeHolds(film: film)
            async let correctionsTask = client.fetchPoseCorrections()
            let (holdsResponse, corrections) = try await (holdsTask, correctionsTask)

            holds = holdsResponse.holds
            paramsHash = holdsResponse.paramsHash
            cachedExcludedHolds = corrections.excludedHolds
            cachedClusterKinds = corrections.clusterKinds

            // Mark ordinals already excluded for this film + params
            excludedOrdinals = Set(
                corrections.excludedHolds
                    .filter { $0.filmId == filmId && $0.paramsHash == holdsResponse.paramsHash }
                    .map { $0.holdOrdinal }
            )
        } catch {
            self.error = error.localizedDescription
        }
        isLoading = false
    }

    func excludeHold(_ hold: HoldInfo, filmId: String, using client: GestureModelAPIClient) async {
        isSaving = true
        let entry = ExcludedHoldEntry(
            filmId: filmId,
            holdOrdinal: hold.ordinal,
            repFrame: hold.repFrame,
            startFrame: hold.startFrame,
            endFrame: hold.endFrame,
            paramsHash: paramsHash
        )
        let newExclusions = cachedExcludedHolds + [entry]
        let body = PoseCorrectionsRequest(
            clusterKinds: cachedClusterKinds,
            excludedHolds: newExclusions
        )
        do {
            let updated = try await client.putPoseCorrections(body)
            cachedExcludedHolds = updated.excludedHolds
            cachedClusterKinds = updated.clusterKinds
            excludedOrdinals.insert(hold.ordinal)
        } catch {
            self.error = "Exclude failed: \(error.localizedDescription)"
        }
        isSaving = false
    }

    func clear() {
        holds = []
        paramsHash = ""
        error = nil
        isLoading = false
        isSaving = false
        excludedOrdinals = []
        cachedExcludedHolds = []
        cachedClusterKinds = [:]
        currentFilmId = nil
    }
}

// MARK: - Failed Film Row

private struct FailedFilmRow: View {
    let film: FailedHandFilm
    let gestureName: String
    let onValidate: () -> Void
    let onDelete: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            // Orange failure badge
            VStack(spacing: 2) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.orange)
                    .font(.subheadline)
                Text("Failed")
                    .font(.caption2)
                    .foregroundColor(.orange)
            }
            .frame(width: 44)

            VStack(alignment: .leading, spacing: 3) {
                Text(gestureName)
                    .font(.subheadline.weight(.medium))
                Text(film.failureReason.displayName)
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(film.failureDetail)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }

            Spacer()

            // Actions
            VStack(spacing: 6) {
                Button(action: onValidate) {
                    Text("Validate")
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.green)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(Color.green.opacity(0.12))
                        .cornerRadius(8)
                }
                .buttonStyle(.plain)

                Button(action: onDelete) {
                    Text("Delete")
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.red)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(Color.red.opacity(0.10))
                        .cornerRadius(8)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }
}

// MARK: - Relabel Sheet

private struct RelabelSheet: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var gestureRegistry: GestureRegistry

    let currentGestureId: String
    let onSelect: (String) -> Void

    var body: some View {
        NavigationView {
            List(gestureRegistry.gestures) { gesture in
                Button {
                    onSelect(gesture.id)
                    dismiss()
                } label: {
                    HStack {
                        Text(gesture.name)
                            .foregroundColor(.primary)
                        Spacer()
                        if gesture.id == currentGestureId {
                            Image(systemName: "checkmark")
                                .foregroundColor(.blue)
                        }
                    }
                }
            }
            .navigationTitle("Change Gesture")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}

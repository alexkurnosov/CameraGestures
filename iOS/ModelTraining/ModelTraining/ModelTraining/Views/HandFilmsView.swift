import SwiftUI
import HandGestureTypes

struct HandFilmsView: View {
    @EnvironmentObject var trainingDataManager: TrainingDataManager
    @EnvironmentObject var gestureRegistry: GestureRegistry

    @State private var filterGestureId: String? = nil
    @State private var currentIndex: Int = 0
    @State private var showingRelabelSheet = false
    @State private var showingDeleteAlert = false
    @State private var currentFrameIndex: Int = 0
    @State private var isPlaying = false
    @State private var playTimer: Timer? = nil
    @State private var showFailedFilms = false
    @State private var failedFilmToValidate: FailedHandFilm? = nil
    @State private var showingValidateAlert = false
    @State private var failedFilmToDelete: FailedHandFilm? = nil
    @State private var showingFailedDeleteAlert = false

    // MARK: - Derived data

    private var filteredExamples: [TrainingExample] {
        trainingDataManager.trainingExamples
            .filter { filterGestureId == nil || $0.gestureId == filterGestureId }
            .sorted { $0.handfilm.startTime > $1.handfilm.startTime }
    }

    private var currentExample: TrainingExample? {
        guard !filteredExamples.isEmpty, currentIndex < filteredExamples.count else { return nil }
        return filteredExamples[currentIndex]
    }

    private var currentFilm: HandFilm? { currentExample?.handfilm }

    private var frameCount: Int { currentFilm?.frames.count ?? 0 }

    private var currentPoints: [Point3D] {
        guard let film = currentFilm, !film.frames.isEmpty else { return [] }
        let safe = min(currentFrameIndex, film.frames.count - 1)
        return film.frames[safe].landmarks
    }

    private var currentHandedness: String {
        guard let film = currentFilm, let first = film.frames.first else { return "—" }
        switch first.leftOrRight {
        case .left: return "Left"
        case .right: return "Right"
        case .unknown: return "Unknown"
        }
    }

    private var currentGestureName: String {
        guard let example = currentExample else { return "—" }
        return gestureRegistry.gestures.first { $0.id == example.gestureId }?.name ?? example.gestureId
    }

    private var filterLabel: String {
        guard let id = filterGestureId else { return "All" }
        return gestureRegistry.gestures.first { $0.id == id }?.name ?? id
    }

    // MARK: - Body

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                if filteredExamples.isEmpty {
                    emptyState
                } else {
                    skeletonSection
                    Divider()
                    bottomPanel
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
        .onChange(of: filterGestureId) { _ in
            stopPlayback()
            currentIndex = 0
            currentFrameIndex = 0
        }
        .onChange(of: currentIndex) { _ in
            stopPlayback()
            currentFrameIndex = 0
        }
        .onDisappear { stopPlayback() }
        .sheet(isPresented: $showingRelabelSheet) {
            RelabelSheet(
                currentGestureId: currentExample?.gestureId ?? "",
                onSelect: { newId in
                    if let example = currentExample {
                        trainingDataManager.relabelExample(id: example.id, newGestureId: newId)
                    }
                }
            )
            .environmentObject(gestureRegistry)
        }
        .alert("Delete Film?", isPresented: $showingDeleteAlert) {
            Button("Delete", role: .destructive) { deleteCurrentExample() }
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
                HandSkeletonView(points: currentPoints)
                    .cornerRadius(12)

                if frameCount == 0 {
                    Text("No frames")
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity)
            .frame(minHeight: 260)
            .padding(.horizontal)
            .padding(.top, 8)

            playerControls
                .padding(.horizontal)
                .padding(.bottom, 8)
        }
    }

    private var playerControls: some View {
        VStack(spacing: 6) {
            if frameCount > 1 {
                Slider(
                    value: Binding(
                        get: { Double(currentFrameIndex) },
                        set: { newValue in
                            currentFrameIndex = Int(newValue)
                            if isPlaying { stopPlayback() }
                        }
                    ),
                    in: 0...Double(max(frameCount - 1, 1)),
                    step: 1
                )
            }

            HStack(spacing: 16) {
                Text("\(currentFrameIndex + 1) / \(frameCount)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .monospacedDigit()

                Spacer()

                Button {
                    isPlaying ? stopPlayback() : startPlayback()
                } label: {
                    Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                        .font(.title3)
                }
                .disabled(frameCount < 2)

                Spacer()

                if let film = currentFilm {
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
                    stopPlayback()
                    currentIndex -= 1
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                        Text("Prev")
                    }
                }
                .disabled(currentIndex <= 0)

                Spacer()

                Text("\(currentIndex + 1) / \(filteredExamples.count)")
                    .font(.subheadline.monospacedDigit())
                    .foregroundColor(.secondary)

                Spacer()

                Button {
                    stopPlayback()
                    currentIndex += 1
                } label: {
                    HStack(spacing: 4) {
                        Text("Next")
                        Image(systemName: "chevron.right")
                    }
                }
                .disabled(currentIndex >= filteredExamples.count - 1)
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
            if let example = currentExample, let film = currentFilm {
                HStack(spacing: 16) {
                    metaChip(label: "Frames", value: "\(film.frames.count)")
                    metaChip(label: "Hand", value: currentHandedness)
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

    // MARK: - Filter Menu

    private var filterMenu: some View {
        Menu {
            Button {
                filterGestureId = nil
            } label: {
                HStack {
                    Text("All")
                    if filterGestureId == nil { Image(systemName: "checkmark") }
                }
            }

            Divider()

            ForEach(gestureRegistry.gestures) { gesture in
                Button {
                    filterGestureId = gesture.id
                } label: {
                    HStack {
                        Text(gesture.name)
                        if filterGestureId == gesture.id { Image(systemName: "checkmark") }
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
            Text(filterGestureId == nil ? "No films collected yet" : "No films for this gesture")
                .font(.headline)
                .foregroundColor(.secondary)
            if filterGestureId != nil {
                Button("Show All") { filterGestureId = nil }
                    .buttonStyle(.bordered)
            }
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Playback

    private func startPlayback() {
        guard frameCount > 1 else { return }
        isPlaying = true
        playTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 24.0, repeats: true) { _ in
            DispatchQueue.main.async {
                if currentFrameIndex < frameCount - 1 {
                    currentFrameIndex += 1
                } else {
                    stopPlayback()
                }
            }
        }
    }

    private func stopPlayback() {
        isPlaying = false
        playTimer?.invalidate()
        playTimer = nil
    }

    // MARK: - Actions

    private func deleteCurrentExample() {
        guard let example = currentExample else { return }
        let total = filteredExamples.count
        trainingDataManager.deleteExample(id: example.id)
        if currentIndex >= total - 1 {
            currentIndex = max(0, total - 2)
        }
        currentFrameIndex = 0
    }

    // MARK: - Failed Films Section

    private var failedFilmsSection: some View {
        let failed = trainingDataManager.failedExamples
            .filter { filterGestureId == nil || $0.gestureId == filterGestureId }

        return VStack(spacing: 0) {
            if !failed.isEmpty {
                Divider().padding(.top, 8)

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
                    .padding(.horizontal)
                    .padding(.vertical, 10)
                }
                .buttonStyle(.plain)

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

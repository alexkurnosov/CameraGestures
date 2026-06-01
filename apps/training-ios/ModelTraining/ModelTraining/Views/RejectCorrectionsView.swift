import Combine
import SwiftUI
import CameraGestures

struct RejectCorrectionsView: View {
    @EnvironmentObject var apiClient: GestureModelAPIClient
    @EnvironmentObject var gestureRegistry: GestureRegistry

    @StateObject private var vm = RejectCorrectionsViewModel()

    @State private var showingRelabelSheet = false
    @State private var showingDeleteAlert = false

    // MARK: - Body

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                if vm.isLoading {
                    loadingState
                } else if let error = vm.error {
                    errorState(error)
                } else if vm.filteredItems.isEmpty {
                    emptyState
                } else {
                    skeletonSection
                    Divider()
                    bottomPanel
                }
            }
        }
        .navigationTitle("Corrections")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                HStack(spacing: 4) {
                    filterMenu
                    Button {
                        Task { await vm.load(phase: vm.phaseFilter, using: apiClient) }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(vm.isLoading)
                }
            }
        }
        .task {
            await vm.load(phase: nil, using: apiClient)
        }
        .sheet(isPresented: $showingRelabelSheet) {
            if let item = vm.currentItem {
                RelabelSheet(
                    currentGestureId: item.gestureId ?? "",
                    gestureRegistry: gestureRegistry,
                    onSelect: { newId in
                        Task { await vm.relabel(to: newId, using: apiClient) }
                    }
                )
            }
        }
        .alert("Delete Correction?", isPresented: $showingDeleteAlert) {
            Button("Delete", role: .destructive) {
                Task { await vm.deleteCurrent(using: apiClient) }
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This reject correction and its hard-negative example will be permanently removed.")
        }
    }

    // MARK: - Skeleton Section

    private var skeletonSection: some View {
        VStack(spacing: 8) {
            ZStack {
                Color.black.opacity(0.85).cornerRadius(12)
                HandSkeletonView(points: vm.currentPoints)
                    .cornerRadius(12)
                if vm.frameCount == 0 {
                    Text("No frames").foregroundColor(.secondary)
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
            if vm.frameCount > 1 {
                Slider(
                    value: Binding(
                        get: { Double(vm.currentFrameIndex) },
                        set: { newValue in
                            vm.currentFrameIndex = Int(newValue)
                            if vm.isPlaying { vm.stopPlayback() }
                        }
                    ),
                    in: 0...Double(max(vm.frameCount - 1, 1)),
                    step: 1
                )
            }

            HStack(spacing: 16) {
                Text("\(vm.currentFrameIndex + 1) / \(max(vm.frameCount, 1))")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .monospacedDigit()

                Spacer()

                Button {
                    vm.isPlaying ? vm.stopPlayback() : vm.startPlayback()
                } label: {
                    Image(systemName: vm.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title3)
                }
                .disabled(vm.frameCount < 2)

                Spacer()

                Text("\(vm.currentIndex + 1) / \(vm.filteredItems.count)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .monospacedDigit()
            }
        }
    }

    // MARK: - Bottom Panel

    private var bottomPanel: some View {
        VStack(spacing: 12) {
            HStack {
                Button {
                    vm.goToPrevious()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.left")
                        Text("Prev")
                    }
                }
                .disabled(vm.currentIndex <= 0)

                Spacer()

                Button {
                    vm.goToNext()
                } label: {
                    HStack(spacing: 4) {
                        Text("Next")
                        Image(systemName: "chevron.right")
                    }
                }
                .disabled(vm.currentIndex >= vm.filteredItems.count - 1)
            }
            .padding(.horizontal)

            Divider()

            metadataChips

            Divider()

            HStack(spacing: 10) {
                Button("Relabel") {
                    showingRelabelSheet = true
                }
                .font(.subheadline)
                .buttonStyle(.bordered)
                .disabled(vm.isSaving)

                Spacer()

                Button(role: .destructive) {
                    showingDeleteAlert = true
                } label: {
                    Image(systemName: "trash")
                }
                .buttonStyle(.bordered)
                .tint(.red)
                .disabled(vm.isSaving)
            }
            .padding(.horizontal)
            .padding(.bottom, 12)
        }
        .padding(.top, 12)
    }

    private var metadataChips: some View {
        VStack(spacing: 8) {
            if let item = vm.currentItem {
                HStack(spacing: 12) {
                    metaChip(label: "Predicted", value: predictedLabel(item.originalPredictedClass))
                    metaChip(label: "Conf", value: String(format: "%.2f", item.originalConfidence))
                    metaChip(label: "Phase", value: item.phase)
                }
                .padding(.horizontal)

                HStack(spacing: 12) {
                    metaChip(label: "None-reject", value: item.isNoneReject ? "✓" : "✗")
                    metaChip(
                        label: "Candidates",
                        value: item.candidateSetSize.map { "\($0)" } ?? "—"
                    )
                    metaChip(
                        label: "Corrected",
                        value: item.gestureId.map { gestureName($0) } ?? "—"
                    )
                }
                .padding(.horizontal)
            }
        }
    }

    private func metaChip(label: String, value: String) -> some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(value)
                .font(.caption.weight(.medium))
                .lineLimit(1)
                .minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
        .background(Color.gray.opacity(0.08))
        .cornerRadius(8)
    }

    private func gestureName(_ id: String) -> String {
        gestureRegistry.gestures.first { $0.id == id }?.name ?? id
    }

    private func predictedLabel(_ predicted: String) -> String {
        // For pose clusters: look up which gestures use this cluster and show their names.
        if let gestureIds = vm.poseGestureMap[predicted], !gestureIds.isEmpty {
            let names = gestureIds.map { gestureName($0) }.joined(separator: ", ")
            return "\(predicted) (\(names))"
        }
        // For phase3 (gesture ID directly): map to name.
        return gestureName(predicted)
    }


    // MARK: - Filter Menu

    private var filterMenu: some View {
        Menu {
            Button {
                Task { await vm.load(phase: nil, using: apiClient) }
            } label: {
                HStack {
                    Text("All")
                    if vm.phaseFilter == nil { Image(systemName: "checkmark") }
                }
            }
            Button {
                Task { await vm.load(phase: "pose", using: apiClient) }
            } label: {
                HStack {
                    Text("Pose")
                    if vm.phaseFilter == "pose" { Image(systemName: "checkmark") }
                }
            }
            Button {
                Task { await vm.load(phase: "phase3", using: apiClient) }
            } label: {
                HStack {
                    Text("Phase 3")
                    if vm.phaseFilter == "phase3" { Image(systemName: "checkmark") }
                }
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "line.3.horizontal.decrease.circle")
                Text(vm.phaseFilter ?? "All")
                    .font(.subheadline)
            }
        }
    }

    // MARK: - Empty / Loading / Error

    private var loadingState: some View {
        VStack(spacing: 16) {
            Spacer()
            ProgressView()
            Text("Loading corrections…").foregroundColor(.secondary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 60)
    }

    private func errorState(_ message: String) -> some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 48))
                .foregroundColor(.orange)
            Text(message)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            Button("Retry") {
                Task { await vm.load(phase: vm.phaseFilter, using: apiClient) }
            }
            .buttonStyle(.bordered)
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 60)
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "hand.raised.slash")
                .font(.system(size: 56))
                .foregroundColor(.gray)
            Text(vm.phaseFilter == nil ? "No corrections stored" : "No \(vm.phaseFilter!) corrections")
                .font(.headline)
                .foregroundColor(.secondary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 60)
    }
}

// MARK: - Relabel Sheet

private struct RelabelSheet: View {
    @Environment(\.dismiss) private var dismiss
    let currentGestureId: String
    let gestureRegistry: GestureRegistry
    let onSelect: (String) -> Void

    var body: some View {
        NavigationView {
            List {
                Section {
                    Button {
                        onSelect("_none")
                        dismiss()
                    } label: {
                        HStack {
                            Text("_none")
                                .foregroundColor(.primary)
                                .font(.body.monospaced())
                            Spacer()
                            if currentGestureId == "_none" {
                                Image(systemName: "checkmark").foregroundColor(.blue)
                            }
                        }
                    }
                }
                Section {
                    ForEach(gestureRegistry.gestures) { gesture in
                        Button {
                            onSelect(gesture.id)
                            dismiss()
                        } label: {
                            HStack {
                                Text(gesture.name).foregroundColor(.primary)
                                Spacer()
                                if gesture.id == currentGestureId {
                                    Image(systemName: "checkmark").foregroundColor(.blue)
                                }
                            }
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

// MARK: - View Model

@MainActor
private final class RejectCorrectionsViewModel: ObservableObject {
    @Published var items: [RejectCorrectionResponse] = []
    @Published var phaseFilter: String? = nil
    @Published var currentIndex: Int = 0
    @Published var currentFrameIndex: Int = 0
    @Published var isPlaying: Bool = false
    @Published var isLoading: Bool = false
    @Published var isSaving: Bool = false
    @Published var error: String? = nil

    private var playTimer: Timer?
    private var currentFilm: HandFilm? = nil
    // "pose_N" → sorted gesture IDs that use that cluster
    private(set) var poseGestureMap: [String: [String]] = [:]

    var filteredItems: [RejectCorrectionResponse] { items }

    var currentItem: RejectCorrectionResponse? {
        guard !filteredItems.isEmpty, currentIndex < filteredItems.count else { return nil }
        return filteredItems[currentIndex]
    }

    var frameCount: Int { currentFilm?.frames.count ?? 0 }

    var currentPoints: [Point3D] {
        guard let film = currentFilm, !film.frames.isEmpty else { return [] }
        let safe = min(currentFrameIndex, film.frames.count - 1)
        return film.frames[safe].landmarks
    }

    func load(phase: String?, using client: GestureModelAPIClient) async {
        stopPlayback()
        isLoading = true
        error = nil
        phaseFilter = phase
        do {
            let response = try await client.fetchRejectCorrections(phase: phase)
            items = response.corrections
            if let manifest = try? await client.fetchPoseManifest() {
                poseGestureMap = buildPoseGestureMap(from: manifest)
            }
            currentIndex = 0
            currentFrameIndex = 0
            rebuildFilm()
        } catch {
            self.error = error.localizedDescription
        }
        isLoading = false
    }

    private func buildPoseGestureMap(from manifest: PoseManifestResponse) -> [String: [String]] {
        var map: [String: Set<String>] = [:]
        for (gestureId, templates) in manifest.gestureTemplates {
            for sequence in templates {
                for poseId in sequence {
                    map["pose_\(poseId)", default: []].insert(gestureId)
                }
            }
        }
        return map.mapValues { $0.sorted() }
    }

    func deleteCurrent(using client: GestureModelAPIClient) async {
        guard let item = currentItem else { return }
        isSaving = true
        do {
            try await client.deleteRejectCorrection(exampleId: item.id)
            let total = filteredItems.count
            items.removeAll { $0.id == item.id }
            if currentIndex >= total - 1 {
                currentIndex = max(0, total - 2)
            }
            currentFrameIndex = 0
            rebuildFilm()
        } catch {
            self.error = "Delete failed: \(error.localizedDescription)"
        }
        isSaving = false
    }

    func relabel(to gestureId: String, using client: GestureModelAPIClient) async {
        guard let item = currentItem, let idx = items.firstIndex(where: { $0.id == item.id }) else { return }
        isSaving = true
        do {
            try await client.updateExample(id: item.id, gestureId: gestureId)
            // Rebuild with updated gestureId locally
            let updated = RejectCorrectionResponse(
                id: item.id,
                gestureId: gestureId,
                sessionId: item.sessionId,
                phase: item.phase,
                originalPredictedClass: item.originalPredictedClass,
                originalConfidence: item.originalConfidence,
                modelVersion: item.modelVersion,
                isNoneReject: item.isNoneReject,
                candidateSetSize: item.candidateSetSize,
                handFilm: item.handFilm,
                createdAt: item.createdAt
            )
            items[idx] = updated
        } catch {
            self.error = "Relabel failed: \(error.localizedDescription)"
        }
        isSaving = false
    }

    func goToPrevious() {
        stopPlayback()
        if currentIndex > 0 { currentIndex -= 1 }
        currentFrameIndex = 0
        rebuildFilm()
    }

    func goToNext() {
        stopPlayback()
        if currentIndex < filteredItems.count - 1 { currentIndex += 1 }
        currentFrameIndex = 0
        rebuildFilm()
    }

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

    private func rebuildFilm() {
        guard let item = currentItem else { currentFilm = nil; return }
        currentFilm = HandFilm(server: item.handFilm)
    }
}

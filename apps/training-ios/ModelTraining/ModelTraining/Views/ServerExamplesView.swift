import SwiftUI
import HandGestureTypes

/// Screen for downloading, reviewing, relabeling, and deleting server examples for a gesture.
struct ServerExamplesView: View {
    let gesture: GestureDefinition

    @EnvironmentObject var trainingDataManager: TrainingDataManager
    @EnvironmentObject var gestureRegistry: GestureRegistry
    @EnvironmentObject var apiClient: GestureModelAPIClient

    @State private var isDownloading = false
    @State private var downloadError: String?
    @State private var hasDownloaded = false
    @State private var showingRelabelSheet = false
    @State private var relabelTarget: TrainingExample?
    @State private var showingDeleteAlert = false
    @State private var deleteTarget: TrainingExample?

    private var examples: [TrainingExample] {
        trainingDataManager.trainingExamples
            .filter { $0.gestureId == gesture.id }
            .sorted { $0.timestamp > $1.timestamp }
    }

    private var hasPendingChanges: Bool {
        let hasRelabels = !trainingDataManager.pendingRelabels.isEmpty
        let hasDeletions = !trainingDataManager.pendingDeletions.isEmpty
        let hasPending = !trainingDataManager.pendingExamples.isEmpty
        return hasRelabels || hasDeletions || hasPending
    }

    var body: some View {
        VStack(spacing: 0) {
            if !hasDownloaded && examples.isEmpty {
                downloadPrompt
            } else if examples.isEmpty && hasDownloaded {
                emptyState
            } else {
                examplesList
            }
        }
        .navigationTitle("Server Examples")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    downloadExamples()
                } label: {
                    if isDownloading {
                        ProgressView()
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "arrow.clockwise")
                    }
                }
                .disabled(isDownloading)
            }
        }
        .sheet(isPresented: $showingRelabelSheet) {
            if let target = relabelTarget {
                RelabelSheet(
                    currentGestureId: target.gestureId,
                    gestureRegistry: gestureRegistry,
                    onSelect: { newId in
                        trainingDataManager.relabelExample(id: target.id, newGestureId: newId)
                    }
                )
            }
        }
        .alert("Delete Example?", isPresented: $showingDeleteAlert) {
            Button("Delete", role: .destructive) {
                if let target = deleteTarget {
                    trainingDataManager.deleteExample(id: target.id)
                    deleteTarget = nil
                }
            }
            Button("Cancel", role: .cancel) { deleteTarget = nil }
        } message: {
            Text("This example will be deleted from the server on the next sync.")
        }
    }

    // MARK: - Download Prompt

    private var downloadPrompt: some View {
        VStack(spacing: 20) {
            Spacer()

            Image(systemName: "arrow.down.circle")
                .font(.system(size: 56))
                .foregroundColor(.blue)

            Text("Download Examples")
                .font(.title2)
                .fontWeight(.medium)

            Text("Fetch all examples for \"\(gesture.name)\" from the server to review, relabel, or delete them.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)

            if let serverCount = trainingDataManager.serverExampleCounts[gesture.id], serverCount > 0 {
                Text("\(serverCount) example(s) on server")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 5)
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(8)
            }

            Button {
                downloadExamples()
            } label: {
                if isDownloading {
                    HStack(spacing: 8) {
                        ProgressView()
                            .tint(.white)
                        Text("Downloading...")
                    }
                    .frame(maxWidth: .infinity)
                } else {
                    Label("Download from Server", systemImage: "arrow.down.circle.fill")
                        .frame(maxWidth: .infinity)
                }
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal, 32)
            .disabled(isDownloading)

            if let error = downloadError {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 32)
            }

            Spacer()
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "tray")
                .font(.system(size: 56))
                .foregroundColor(.gray)
            Text("No examples on server")
                .font(.headline)
                .foregroundColor(.secondary)
            Text("No examples found for \"\(gesture.name)\" on the server.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
            Spacer()
        }
    }

    // MARK: - Examples List

    private var examplesList: some View {
        List {
            Section {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("\(examples.count) example(s)")
                            .font(.subheadline.weight(.medium))
                        if hasPendingChanges {
                            Text("Unsaved changes pending sync")
                                .font(.caption)
                                .foregroundColor(.orange)
                        }
                    }
                    Spacer()
                    if let serverCount = trainingDataManager.serverExampleCounts[gesture.id] {
                        Text("\(serverCount) on server")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }

            Section("Examples") {
                ForEach(examples) { example in
                    ServerExampleRow(
                        example: example,
                        gestureName: gestureRegistry.gestures.first { $0.id == example.gestureId }?.name ?? example.gestureId,
                        isPendingRelabel: trainingDataManager.pendingRelabels[example.id] != nil,
                        onRelabel: {
                            relabelTarget = example
                            showingRelabelSheet = true
                        },
                        onDelete: {
                            deleteTarget = example
                            showingDeleteAlert = true
                        }
                    )
                }
            }
        }
    }

    // MARK: - Actions

    private func downloadExamples() {
        isDownloading = true
        downloadError = nil
        Task {
            do {
                let count = try await trainingDataManager.downloadExamplesFromServer(gestureId: gesture.id)
                await MainActor.run {
                    isDownloading = false
                    hasDownloaded = true
                    print("ServerExamplesView: downloaded \(count) example(s)")
                }
            } catch {
                await MainActor.run {
                    isDownloading = false
                    downloadError = error.localizedDescription
                }
            }
        }
    }
}

// MARK: - Server Example Row

private struct ServerExampleRow: View {
    let example: TrainingExample
    let gestureName: String
    let isPendingRelabel: Bool
    let onRelabel: () -> Void
    let onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(gestureName)
                            .font(.subheadline.weight(.medium))
                        if isPendingRelabel {
                            Text("relabeled")
                                .font(.caption2)
                                .foregroundColor(.orange)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(Color.orange.opacity(0.15))
                                .cornerRadius(4)
                        }
                    }
                    Text(example.id.uuidString.prefix(8) + "...")
                        .font(.caption2.monospaced())
                        .foregroundColor(.secondary)
                }

                Spacer()

                Button("Relabel") {
                    onRelabel()
                }
                .font(.caption)
                .buttonStyle(.bordered)

                Button(role: .destructive) {
                    onDelete()
                } label: {
                    Image(systemName: "trash")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .tint(.red)
            }

            HStack(spacing: 12) {
                metaChip(label: "Frames", value: "\(example.handfilm.frames.count)")
                metaChip(label: "Duration", value: String(format: "%.2fs", example.handfilm.duration))
                metaChip(label: "In-view", value: String(format: "%.2fs", example.handfilm.inViewDuration))
                metaChip(label: "Created", value: relativeTime(example.timestamp))
            }
        }
        .padding(.vertical, 4)
    }

    private func metaChip(label: String, value: String) -> some View {
        VStack(spacing: 1) {
            Text(label)
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(value)
                .font(.caption2.weight(.medium))
        }
        .frame(maxWidth: .infinity)
    }

    private func relativeTime(_ timestamp: TimeInterval) -> String {
        let diff = Date().timeIntervalSince1970 - timestamp
        if diff < 60 { return "\(Int(diff))s ago" }
        if diff < 3600 { return "\(Int(diff / 60))m ago" }
        if diff < 86400 { return "\(Int(diff / 3600))h ago" }
        return "\(Int(diff / 86400))d ago"
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

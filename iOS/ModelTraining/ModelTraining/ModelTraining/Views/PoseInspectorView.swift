import Combine
import SwiftUI
import HandGestureTypes

// MARK: - PoseInspectorView

/// Per-cluster grid view (Stage 5.2 / 5.3).
/// Pulls the pose manifest, corrections, and cluster holds from the server,
/// renders a grid of hold thumbnails grouped by cluster, and lets the reviewer
/// mark each cluster as idle / regular / unconfirmed.
struct PoseInspectorView: View {
    @EnvironmentObject var apiClient: GestureModelAPIClient

    @StateObject private var vm = PoseInspectorViewModel()

    var body: some View {
        Group {
            if vm.isLoading && vm.clusters.isEmpty {
                ProgressView("Loading inspector…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let err = vm.error, vm.clusters.isEmpty {
                errorState(err)
            } else if vm.clusters.isEmpty {
                Text("No pose model trained yet.")
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                clusterList
            }
        }
        .navigationTitle("Pose Inspector")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button {
                    Task { await vm.load(using: apiClient) }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .disabled(vm.isLoading)
            }
        }
        .task {
            await vm.load(using: apiClient)
        }
        .overlay {
            if vm.isSaving {
                Color.black.opacity(0.2).ignoresSafeArea()
                ProgressView("Saving…")
                    .padding(20)
                    .background(.regularMaterial)
                    .cornerRadius(12)
            }
        }
        .alert("Error", isPresented: Binding(
            get: { vm.saveError != nil },
            set: { if !$0 { vm.saveError = nil } }
        )) {
            Button("OK", role: .cancel) { vm.saveError = nil }
        } message: {
            Text(vm.saveError ?? "")
        }
    }

    // MARK: - Cluster list

    private var clusterList: some View {
        List {
            ForEach(vm.clusters) { cluster in
                Section {
                    clusterHoldsRow(cluster: cluster)
                } header: {
                    clusterHeader(cluster: cluster)
                }
            }
        }
        .listStyle(.insetGrouped)
        .refreshable { await vm.load(using: apiClient) }
    }

    // MARK: - Cluster header

    private func clusterHeader(cluster: ClusterViewModel) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                // Suspected-idle badge
                if cluster.suspectedIdle {
                    Label("Suspected idle", systemImage: "hand.raised.slash")
                        .font(.caption2.weight(.semibold))
                        .foregroundColor(.orange)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.orange.opacity(0.15))
                        .cornerRadius(6)
                }

                // Kind badge
                kindBadge(kind: cluster.currentKind)

                Spacer()

                Text("n=\(cluster.nSamples)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Text("Cluster \(cluster.id)")
                .font(.subheadline.weight(.semibold))

            // Gesture composition bar
            if !cluster.gestureComposition.isEmpty {
                gestureCompositionRow(cluster.gestureComposition, total: cluster.nSamples)
            }

            // Mark actions
            HStack(spacing: 8) {
                kindButton(label: "Idle", kind: "idle", cluster: cluster)
                kindButton(label: "Regular", kind: "regular", cluster: cluster)
                kindButton(label: "Unconfirmed", kind: "unconfirmed", cluster: cluster)
            }
            .padding(.top, 2)
        }
        .padding(.vertical, 4)
        .textCase(nil)
    }

    private func kindBadge(kind: String) -> some View {
        let (color, icon): (Color, String) = {
            switch kind {
            case "idle":         return (.purple, "hand.raised.slash")
            case "regular":      return (.green,  "checkmark.circle")
            default:             return (.gray,   "questionmark.circle")
            }
        }()
        return Label(kind, systemImage: icon)
            .font(.caption2.weight(.medium))
            .foregroundColor(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.12))
            .cornerRadius(6)
    }

    private func gestureCompositionRow(_ comp: [String: Int], total: Int) -> some View {
        HStack(spacing: 4) {
            ForEach(comp.sorted(by: { $0.value > $1.value }), id: \.key) { id, count in
                let pct = total > 0 ? Double(count) / Double(total) : 0
                Text("\(id) \(Int(pct * 100))%")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }

    private func kindButton(label: String, kind: String, cluster: ClusterViewModel) -> some View {
        let isSelected = cluster.currentKind == kind
        return Button(label) {
            Task { await vm.setKind(kind, for: cluster.id, using: apiClient) }
        }
        .font(.caption.weight(isSelected ? .semibold : .regular))
        .foregroundColor(isSelected ? .white : .primary)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(isSelected ? kindColor(kind) : Color.gray.opacity(0.12))
        .cornerRadius(8)
        .buttonStyle(.plain)
        .disabled(vm.isSaving)
    }

    private func kindColor(_ kind: String) -> Color {
        switch kind {
        case "idle":    return .purple
        case "regular": return .green
        default:        return .gray
        }
    }

    // MARK: - Cluster holds grid

    @ViewBuilder
    private func clusterHoldsRow(cluster: ClusterViewModel) -> some View {
        if cluster.holds.isEmpty {
            Text("No non-edge holds")
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(cluster.holds) { hold in
                        holdThumbnail(hold: hold)
                    }
                }
                .padding(.vertical, 6)
            }
        }
    }

    private func holdThumbnail(hold: HoldViewModel) -> some View {
        VStack(spacing: 4) {
            ZStack {
                Color.black.opacity(0.80)
                HandSkeletonView(points: hold.landmarks)
            }
            .frame(width: 72, height: 72)
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(hold.isEdge ? Color.orange.opacity(0.6) : Color.blue.opacity(0.4), lineWidth: 1.5)
            )

            Text(hold.gestureId)
                .font(.system(size: 8))
                .foregroundColor(.secondary)
                .lineLimit(1)
            Text(String(format: "d=%.2f", hold.distanceFromCentroid))
                .font(.system(size: 8, design: .monospaced))
                .foregroundColor(.secondary)
        }
    }

    // MARK: - Error state

    private func errorState(_ message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .foregroundColor(.orange)
            Text("Failed to load inspector")
                .font(.headline)
            Text(message)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            Button("Retry") { Task { await vm.load(using: apiClient) } }
                .buttonStyle(.borderedProminent)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - View Models

private struct ClusterViewModel: Identifiable {
    let id: String              // cluster id string key
    let nSamples: Int
    let suspectedIdle: Bool
    let gestureComposition: [String: Int]  // gesture_id → hold count
    var currentKind: String
    var holds: [HoldViewModel]  // non-edge holds sorted by distance from centroid
}

private struct HoldViewModel: Identifiable {
    let id: String
    let gestureId: String
    let isEdge: Bool
    let distanceFromCentroid: Double
    let landmarks: [Point3D]    // decoded from 63-float coords
}

@MainActor
private final class PoseInspectorViewModel: ObservableObject {
    @Published var clusters: [ClusterViewModel] = []
    @Published var isLoading = false
    @Published var isSaving = false
    @Published var error: String? = nil
    @Published var saveError: String? = nil

    private var pendingKinds: [String: String] = [:]  // cluster_id → kind (accumulated edits)
    private var cachedCorrections: PoseCorrectionsResponse? = nil

    func load(using client: GestureModelAPIClient) async {
        isLoading = true
        error = nil
        do {
            async let manifestTask = client.fetchPoseManifest()
            async let correctionsTask = client.fetchPoseCorrections()
            async let holdsTask = client.fetchClusterHolds()

            let (manifest, corrections, clusterHolds) = try await (manifestTask, correctionsTask, holdsTask)
            cachedCorrections = corrections
            pendingKinds = corrections.clusterKinds

            clusters = buildClusters(manifest: manifest, corrections: corrections, clusterHolds: clusterHolds)
        } catch {
            self.error = error.localizedDescription
        }
        isLoading = false
    }

    func setKind(_ kind: String, for clusterId: String, using client: GestureModelAPIClient) async {
        // Optimistic update
        if let idx = clusters.firstIndex(where: { $0.id == clusterId }) {
            clusters[idx].currentKind = kind
        }
        pendingKinds[clusterId] = kind

        isSaving = true
        saveError = nil
        do {
            let body = PoseCorrectionsRequest(
                clusterKinds: pendingKinds,
                excludedHolds: cachedCorrections?.excludedHolds ?? []
            )
            let updated = try await client.putPoseCorrections(body)
            cachedCorrections = updated
            pendingKinds = updated.clusterKinds
        } catch {
            saveError = error.localizedDescription
            // Revert optimistic update on failure
            if let cached = cachedCorrections,
               let idx = clusters.firstIndex(where: { $0.id == clusterId }) {
                clusters[idx].currentKind = cached.clusterKinds[clusterId] ?? "unconfirmed"
            }
        }
        isSaving = false
    }

    private func buildClusters(
        manifest: PoseManifestResponse,
        corrections: PoseCorrectionsResponse,
        clusterHolds: ClusterHoldsResponse
    ) -> [ClusterViewModel] {
        // Group holds by cluster_id
        var holdsByCluster: [String: [ClusterHoldEntry]] = [:]
        for hold in clusterHolds.holds {
            let key = String(hold.clusterId)
            holdsByCluster[key, default: []].append(hold)
        }

        // Build gesture composition per cluster from holds
        var gestureComp: [String: [String: Int]] = [:]
        for hold in clusterHolds.holds where !hold.isEdge {
            let key = String(hold.clusterId)
            gestureComp[key, default: [:]][hold.gestureId, default: 0] += 1
        }

        return manifest.poseClusters
            .sorted { a, b in
                // Sort: suspected_idle first, then by n_samples descending
                let ai = a.value.suspectedIdle ? 0 : 1
                let bi = b.value.suspectedIdle ? 0 : 1
                if ai != bi { return ai < bi }
                return a.value.nSamples > b.value.nSamples
            }
            .map { (cid, info) in
                let kind = corrections.clusterKinds[cid] ?? "unconfirmed"
                let rawHolds = holdsByCluster[cid] ?? []
                let holdVMs = rawHolds
                    .filter { !$0.isEdge }
                    .sorted { $0.distanceFromCentroid < $1.distanceFromCentroid }
                    .map { entry -> HoldViewModel in
                        HoldViewModel(
                            id: entry.id,
                            gestureId: entry.gestureId,
                            isEdge: entry.isEdge,
                            distanceFromCentroid: entry.distanceFromCentroid,
                            landmarks: decodeLandmarks(entry.coords)
                        )
                    }
                return ClusterViewModel(
                    id: cid,
                    nSamples: info.nSamples,
                    suspectedIdle: info.suspectedIdle,
                    gestureComposition: gestureComp[cid] ?? [:],
                    currentKind: kind,
                    holds: holdVMs
                )
            }
    }
}

// MARK: - Landmark decode

private func decodeLandmarks(_ coords: [Double]) -> [Point3D] {
    guard coords.count == 63 else { return [] }
    return stride(from: 0, to: 63, by: 3).map { i in
        Point3D(x: Float(coords[i]), y: Float(coords[i + 1]), z: Float(coords[i + 2]))
    }
}

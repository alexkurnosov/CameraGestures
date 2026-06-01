import SwiftUI
import CameraGestures

/// Modal sheet shown when the reviewer taps ✗ on a high-confidence pose prediction.
/// Lets the reviewer pick a replacement pose label (or _none) and posts the correction.
struct PoseCorrectionView: View {
    let capture: PendingPoseCapture
    let gestureRecognizer: GestureRecognizerWrapper
    let gestureRegistry: GestureRegistry
    let appSettings: AppSettings
    let apiClient: GestureModelAPIClient

    @Environment(\.dismiss) private var dismiss

    // Pose clusters loaded from the on-device manifest.
    @State private var poseOptions: [(id: String, label: String)] = [("_none", "_none")]
    // Maps "pose_<n>" → sorted gesture IDs that include that pose in any template.
    @State private var poseGestureMap: [String: [String]] = [:]
    // Maps "pose_<n>" → centroid landmarks for skeleton preview.
    @State private var poseClusterMap: [String: PoseClusterInfo] = [:]
    // class_labels ordering (index → cluster ID), needed to map allScores → cluster key.
    @State private var classLabels: [Int] = []
    // Model softmax confidence per cluster key ("pose_<n>"), fetched on appear.
    @State private var poseScores: [String: Float] = [:]
    @State private var selectedReplacement: String? = nil
    @State private var isSending = false
    @State private var errorMessage: String? = nil

    private var skeletonPoints: [Point3D] {
        guard let coords = capture.normalizedCoords, coords.count == 63 else {
            return capture.repShot?.landmarks ?? []
        }
        return stride(from: 0, to: 63, by: 3).map { i in
            Point3D(x: coords[i], y: coords[i+1], z: coords[i+2])
        }
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                // Skeleton
                HoldSkeletonView(points: skeletonPoints)
                    .frame(height: 180)
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    .padding(.horizontal)

                // Original prediction info
                VStack(spacing: 4) {
                    Text("Original: pose_\(capture.logEntry.predictedPoseId)")
                        .font(.subheadline.weight(.semibold))
                    Text(String(format: "%.0f%% · model %@",
                                capture.logEntry.confidence * 100,
                                capture.logEntry.modelVersion))
                        .font(.caption)
                        .foregroundColor(.secondary)
                    let originalKey = "pose_\(capture.logEntry.predictedPoseId)"
                    if let gestureIds = poseGestureMap[originalKey], !gestureIds.isEmpty {
                        let names = gestureIds.map { id in
                            gestureRegistry.gestures.first { $0.id == id }?.name ?? id
                        }
                        Text("in: \(names.joined(separator: ", "))")
                            .font(.caption)
                            .foregroundColor(.orange)
                    }
                }

                Divider()

                // Replacement picker
                Text("Select replacement label")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal)

                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(sortedPoseOptions, id: \.id) { option in
                            Button {
                                selectedReplacement = option.id
                            } label: {
                                PoseCandidateRow(
                                    option: option,
                                    clusterInfo: poseClusterMap[option.id],
                                    gestures: poseGestureMap[option.id],
                                    score: poseScores[option.id],
                                    isSelected: selectedReplacement == option.id
                                )
                            }
                            .buttonStyle(.plain)
                            Divider().padding(.leading)
                        }
                    }
                }

                if let err = errorMessage {
                    Text(err)
                        .font(.caption)
                        .foregroundColor(.red)
                        .padding(.horizontal)
                }

                // Action buttons
                HStack(spacing: 12) {
                    Button("Close") {
                        resumeAndDismiss()
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(Color(.systemGray5))
                    .cornerRadius(10)

                    Button {
                        Task { await sendCorrection() }
                    } label: {
                        if isSending {
                            ProgressView().frame(maxWidth: .infinity)
                        } else {
                            Text("Send Correction")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .padding(.vertical, 12)
                    .background(selectedReplacement != nil ? Color.blue : Color.gray.opacity(0.4))
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .disabled(selectedReplacement == nil || isSending)
                }
                .padding(.horizontal)
                .padding(.bottom, 8)
            }
            .navigationTitle("Pose Correction")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                loadPoseOptions()
                loadPoseScores()
            }
        }
    }

    // MARK: - Sorting

    private var sortedPoseOptions: [(id: String, label: String)] {
        let none = poseOptions.filter { $0.id == "_none" }
        let rest = poseOptions.filter { $0.id != "_none" }
            .sorted { (poseScores[$0.id] ?? 0) > (poseScores[$1.id] ?? 0) }
        return none + rest
    }

    // MARK: - Data loading

    private func loadPoseOptions() {
        let manifestURL = appSettings.defaultPoseManifestURL()
        guard FileManager.default.fileExists(atPath: manifestURL.path),
              let data = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(PoseManifestResponse.self, from: data)
        else {
            poseOptions = [("_none", "_none")]
            return
        }
        let sorted = manifest.poseClusters
            .sorted { a, b in (Int(a.key) ?? 0) < (Int(b.key) ?? 0) }
            .map { (id: "pose_\($0.key)", label: $0.value.label) }
        poseOptions = [("_none", "_none")] + sorted
        poseGestureMap = buildPoseGestureMap(from: manifest)
        poseClusterMap = Dictionary(uniqueKeysWithValues:
            manifest.poseClusters.map { ("pose_\($0.key)", $0.value) })
        classLabels = manifest.classLabels ?? []
    }

    private func loadPoseScores() {
        guard let shot = capture.repShot else { return }
        Task {
            guard let prediction = try? gestureRecognizer.recognizer
                .predictPoseFromShot(shot, allScores: true),
                  let scores = prediction.allScores
            else { return }
            var mapped: [String: Float] = [:]
            for (i, clusterId) in classLabels.enumerated() {
                if let score = scores[i] { mapped["pose_\(clusterId)"] = score }
            }
            await MainActor.run { poseScores = mapped }
        }
    }

    private func buildPoseGestureMap(from manifest: PoseManifestResponse) -> [String: [String]] {
        var map: [String: Set<String>] = [:]
        for (gestureId, templates) in manifest.gestureTemplates {
            for sequence in templates {
                for poseId in sequence {
                    let key = "pose_\(poseId)"
                    map[key, default: []].insert(gestureId)
                }
            }
        }
        return map.mapValues { $0.sorted() }
    }

    // MARK: - Send

    private func sendCorrection() async {
        guard let replacement = selectedReplacement else { return }
        isSending = true
        errorMessage = nil

        let correctionId = UUID().uuidString
        let isNone = (replacement == "_none")

        let correctionPayload = RejectCorrectionPayloadDTO(
            id: correctionId,
            originalPredictedClass: "pose_\(capture.logEntry.predictedPoseId)",
            originalConfidence: capture.logEntry.confidence,
            modelVersion: capture.logEntry.modelVersion,
            isNoneReject: isNone,
            candidateSetSize: nil
        )

        // Build a one-frame HandFilm from the repShot (§8.4: reuse hand_film_json column).
        let handFilm: HandFilmPayload
        if let shot = capture.repShot {
            var film = HandFilm(startTime: shot.timestamp)
            film.addFrame(shot)
            handFilm = HandFilmPayload(from: film)
        } else {
            let ts = Date().timeIntervalSince1970
            var film = HandFilm(startTime: ts)
            film.addFrame(HandShot.absent(timestamp: ts))
            handFilm = HandFilmPayload(from: film)
        }

        let example = RejectTrainingExampleDTO(
            id: UUID().uuidString,
            handFilm: handFilm,
            gestureId: isNone ? nil : replacement,
            sessionId: UUID().uuidString,
            userId: "current_user",
            phase: "pose",
            correction: correctionPayload
        )

        let item = RejectCorrectionItemDTO(rejectCorrection: correctionPayload, example: example)

        do {
            try await apiClient.postRejectCorrections(items: [item])
            resumeAndDismiss()
        } catch {
            errorMessage = error.localizedDescription
        }
        isSending = false
    }

    // MARK: - Lifecycle

    private func resumeAndDismiss() {
        Task {
            try? await gestureRecognizer.recognizer.resumeFromCorrection()
        }
        dismiss()
    }
}

// MARK: - Shared candidate row

struct PoseCandidateRow: View {
    let option: (id: String, label: String)
    let clusterInfo: PoseClusterInfo?
    let gestures: [String]?
    let score: Float?
    let isSelected: Bool

    private var centroidPoints: [Point3D] {
        guard let c = clusterInfo?.centroid, c.count >= 63 else { return [] }
        return stride(from: 0, to: 63, by: 3).map { i in
            Point3D(x: Float(c[i]), y: Float(c[i+1]), z: Float(c[i+2]))
        }
    }

    var body: some View {
        HStack(spacing: 10) {
            if !centroidPoints.isEmpty {
                HoldSkeletonView(points: centroidPoints)
                    .frame(width: 64, height: 64)
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
            } else {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(.systemGray5))
                    .frame(width: 64, height: 64)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(option.id)
                    .font(.subheadline.weight(.medium))
                    .foregroundColor(.primary)
                if !option.label.isEmpty && option.label != option.id {
                    Text(option.label)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                if let gestures, !gestures.isEmpty {
                    Text("in: \(gestures.joined(separator: ", "))")
                        .font(.caption)
                        .foregroundColor(.orange)
                }
            }

            Spacer()

            if let score {
                Text(String(format: "%.0f%%", score * 100))
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.secondary)
                    .frame(minWidth: 36, alignment: .trailing)
            }

            if isSelected {
                Image(systemName: "checkmark")
                    .foregroundColor(.blue)
                    .font(.subheadline.weight(.semibold))
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
        .background(isSelected ? Color.blue.opacity(0.08) : Color.clear)
    }
}

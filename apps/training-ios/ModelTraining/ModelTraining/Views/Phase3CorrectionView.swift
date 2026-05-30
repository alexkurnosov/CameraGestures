import SwiftUI
import CameraGestures

/// Modal sheet shown when the reviewer taps ✗ on a high-confidence Phase 3 gesture.
/// Shows hold thumbnails from the recently-captured hold telemetry, lets the reviewer
/// pick a replacement gesture and optionally relabel individual holds, then posts all
/// corrections atomically.
struct Phase3CorrectionView: View {
    let capture: PendingPhase3Capture
    let gestureRecognizer: GestureRecognizerWrapper
    let gestureRegistry: GestureRegistry
    let appSettings: AppSettings
    let apiClient: GestureModelAPIClient

    @Environment(\.dismiss) private var dismiss

    @State private var selectedGesture: String? = nil
    // Per-hold relabels keyed by index in capture.recentHolds.
    // Value is the replacement pose ID string (e.g. "pose_3") or "_none".
    @State private var holdRelabels: [Int: String] = [:]
    @State private var relabelTarget: HoldRelabelTarget? = nil
    @State private var poseOptions: [(id: String, label: String)] = []
    @State private var isSending = false
    @State private var errorMessage: String? = nil

    private var gesture: DetectedGesture { capture.gesture }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Hold thumbnail strip — shows holds from the telemetry window
                if !capture.recentHolds.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 10) {
                            ForEach(Array(capture.recentHolds.enumerated()), id: \.offset) { idx, hold in
                                HoldThumbnailCell(
                                    hold: hold,
                                    index: idx,
                                    relabeledAs: holdRelabels[idx]
                                ) {
                                    relabelTarget = HoldRelabelTarget(index: idx)
                                }
                            }
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                    }
                    .background(Color(.systemGray6))
                }

                Divider()

                ScrollView {
                    VStack(spacing: 16) {
                        // Original prediction info
                        VStack(spacing: 4) {
                            Text("Original: \(gesture.prediction.gestureId)")
                                .font(.subheadline.weight(.semibold))
                            let candidateText = gesture.candidateSetSize.map { "candidates: \($0) · " } ?? ""
                            Text(String(format: "%@%.0f%% · model %@",
                                        candidateText,
                                        Double(gesture.prediction.confidence) * 100,
                                        gestureRecognizer.handfilmModelVersion))
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding(.top, 12)

                        Divider()

                        Text("Select replacement gesture")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal)

                        VStack(spacing: 0) {
                            ForEach(gestureOptions, id: \.self) { gestureId in
                                Button {
                                    selectedGesture = gestureId
                                } label: {
                                    HStack {
                                        Text(gestureId == "_none" ? "_none" : gestureName(for: gestureId))
                                            .font(.subheadline.weight(.medium))
                                            .foregroundColor(.primary)
                                        Spacer()
                                        if selectedGesture == gestureId {
                                            Image(systemName: "checkmark")
                                                .foregroundColor(.blue)
                                                .font(.subheadline.weight(.semibold))
                                        }
                                    }
                                    .padding(.horizontal)
                                    .padding(.vertical, 10)
                                    .background(selectedGesture == gestureId
                                                ? Color.blue.opacity(0.08)
                                                : Color.clear)
                                }
                                .buttonStyle(.plain)
                                Divider().padding(.leading)
                            }
                        }
                        .background(Color(.systemBackground))
                        .cornerRadius(10)
                        .padding(.horizontal)

                        if let err = errorMessage {
                            Text(err)
                                .font(.caption)
                                .foregroundColor(.red)
                                .padding(.horizontal)
                        }
                    }
                }

                HStack(spacing: 12) {
                    Button("Close") { resumeAndDismiss() }
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
                            Text("Send Correction").frame(maxWidth: .infinity)
                        }
                    }
                    .padding(.vertical, 12)
                    .background(selectedGesture != nil ? Color.blue : Color.gray.opacity(0.4))
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .disabled(selectedGesture == nil || isSending)
                }
                .padding(.horizontal)
                .padding(.bottom, 8)
            }
            .navigationTitle("Phase 3 Correction")
            .navigationBarTitleDisplayMode(.inline)
        }
        .sheet(item: $relabelTarget) { target in
            RelabelHoldSheet(
                holdIndex: target.index,
                hold: capture.recentHolds[target.index],
                poseOptions: poseOptions,
                currentLabel: holdRelabels[target.index]
            ) { replacement in
                holdRelabels[target.index] = replacement
            }
        }
        .onAppear { loadPoseOptions() }
    }

    // MARK: - Helpers

    private var gestureOptions: [String] {
        gestureRegistry.gestures.map(\.id) + ["_none"]
    }

    private func gestureName(for id: String) -> String {
        gestureRegistry.gestures.first { $0.id == id }?.name ?? id
    }

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
            .sorted { (Int($0.key) ?? 0) < (Int($1.key) ?? 0) }
            .map { (id: "pose_\($0.key)", label: $0.value.label) }
        poseOptions = sorted + [("_none", "_none")]
    }

    // MARK: - Send

    private func sendCorrection() async {
        guard let replacementGesture = selectedGesture else { return }
        isSending = true
        errorMessage = nil

        let correctionId = UUID().uuidString
        let isNone = (replacementGesture == "_none")

        let phase3Correction = RejectCorrectionPayloadDTO(
            id: correctionId,
            originalPredictedClass: gesture.prediction.gestureId,
            originalConfidence: Double(gesture.prediction.confidence),
            modelVersion: gestureRecognizer.handfilmModelVersion,
            isNoneReject: isNone,
            candidateSetSize: gesture.candidateSetSize
        )

        let phase3Example = RejectTrainingExampleDTO(
            id: UUID().uuidString,
            handFilm: HandFilmDTO(from: gesture.handfilm),
            gestureId: isNone ? nil : replacementGesture,
            sessionId: UUID().uuidString,
            userId: "current_user",
            phase: "phase3",
            correction: phase3Correction
        )

        var items: [RejectCorrectionItemDTO] = [
            RejectCorrectionItemDTO(rejectCorrection: phase3Correction, example: phase3Example)
        ]

        // Per-hold pose corrections for relabeled holds (§4.3)
        for (holdIdx, replacementPose) in holdRelabels {
            guard holdIdx < capture.recentHolds.count else { continue }
            let holdCapture = capture.recentHolds[holdIdx]
            let holdIsNone = (replacementPose == "_none")

            let holdCorrectionId = UUID().uuidString
            let originalClass = "pose_\(holdCapture.logEntry.predictedPoseId)"

            let holdCorrection = RejectCorrectionPayloadDTO(
                id: holdCorrectionId,
                originalPredictedClass: originalClass,
                originalConfidence: holdCapture.logEntry.confidence,
                modelVersion: holdCapture.logEntry.modelVersion,
                isNoneReject: holdIsNone,
                candidateSetSize: nil
            )

            let holdFilm: HandFilmDTO
            if let shot = holdCapture.repShot {
                var film = HandFilm(startTime: shot.timestamp)
                film.addFrame(shot)
                holdFilm = HandFilmDTO(from: film)
            } else {
                let absent = HandShot(landmarks: [], timestamp: Date().timeIntervalSince1970,
                                      leftOrRight: .unknown, isAbsent: true)
                var film = HandFilm(startTime: absent.timestamp)
                film.addFrame(absent)
                holdFilm = HandFilmDTO(from: film)
            }

            let holdExample = RejectTrainingExampleDTO(
                id: UUID().uuidString,
                handFilm: holdFilm,
                gestureId: holdIsNone ? nil : replacementPose,
                sessionId: UUID().uuidString,
                userId: "current_user",
                phase: "pose",
                correction: holdCorrection
            )

            items.append(RejectCorrectionItemDTO(rejectCorrection: holdCorrection, example: holdExample))
        }

        do {
            try await apiClient.postRejectCorrections(items: items)
            resumeAndDismiss()
        } catch {
            errorMessage = error.localizedDescription
        }
        isSending = false
    }

    private func resumeAndDismiss() {
        Task { try? await gestureRecognizer.recognizer.resumeFromCorrection() }
        dismiss()
    }
}

// MARK: - Hold thumbnail cell

private struct HoldThumbnailCell: View {
    let hold: PendingPoseCapture
    let index: Int
    let relabeledAs: String?
    let onTap: () -> Void

    private var skeletonPoints: [Point3D] {
        guard let coords = hold.normalizedCoords, coords.count == 63 else {
            return hold.repShot?.landmarks ?? []
        }
        return stride(from: 0, to: 63, by: 3).map { i in
            Point3D(x: coords[i], y: coords[i+1], z: coords[i+2])
        }
    }

    var body: some View {
        Button(action: onTap) {
            VStack(spacing: 4) {
                ZStack(alignment: .topTrailing) {
                    HoldSkeletonView(points: skeletonPoints)
                        .frame(width: 80, height: 80)
                        .background(Color(.systemBackground))
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(relabeledAs != nil ? Color.orange : Color.clear, lineWidth: 2)
                        )
                    if relabeledAs != nil {
                        Image(systemName: "pencil.circle.fill")
                            .foregroundColor(.orange)
                            .font(.caption)
                            .padding(2)
                    }
                }
                Text("pose_\(hold.logEntry.predictedPoseId)")
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                Text(String(format: "%.0f%%", hold.logEntry.confidence * 100))
                    .font(.system(size: 9))
                    .foregroundColor(.secondary)
            }
            .frame(width: 88)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Per-hold relabel sheet

private struct HoldRelabelTarget: Identifiable {
    let index: Int
    var id: Int { index }
}

struct RelabelHoldSheet: View {
    let holdIndex: Int
    let hold: PendingPoseCapture
    let poseOptions: [(id: String, label: String)]
    let currentLabel: String?
    let onApply: (String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var selected: String?

    private var skeletonPoints: [Point3D] {
        guard let coords = hold.normalizedCoords, coords.count == 63 else {
            return hold.repShot?.landmarks ?? []
        }
        return stride(from: 0, to: 63, by: 3).map { i in
            Point3D(x: coords[i], y: coords[i+1], z: coords[i+2])
        }
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                HoldSkeletonView(points: skeletonPoints)
                    .frame(height: 160)
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    .padding(.horizontal)

                Text(String(format: "Original pose: pose_%d · %.0f%%",
                            hold.logEntry.predictedPoseId,
                            hold.logEntry.confidence * 100))
                    .font(.caption)
                    .foregroundColor(.secondary)

                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(poseOptions, id: \.id) { option in
                            Button {
                                selected = option.id
                            } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(option.id)
                                            .font(.subheadline.weight(.medium))
                                            .foregroundColor(.primary)
                                        if !option.label.isEmpty && option.label != option.id {
                                            Text(option.label)
                                                .font(.caption)
                                                .foregroundColor(.secondary)
                                        }
                                    }
                                    Spacer()
                                    if selected == option.id {
                                        Image(systemName: "checkmark")
                                            .foregroundColor(.blue)
                                            .font(.subheadline.weight(.semibold))
                                    }
                                }
                                .padding(.horizontal)
                                .padding(.vertical, 10)
                                .background(selected == option.id
                                            ? Color.blue.opacity(0.08)
                                            : Color.clear)
                            }
                            .buttonStyle(.plain)
                            Divider().padding(.leading)
                        }
                    }
                }

                HStack(spacing: 12) {
                    Button("Cancel") { dismiss() }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(Color(.systemGray5))
                        .cornerRadius(10)

                    Button("Apply") {
                        if let s = selected { onApply(s) }
                        dismiss()
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(selected != nil ? Color.blue : Color.gray.opacity(0.4))
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .disabled(selected == nil)
                }
                .padding(.horizontal)
                .padding(.bottom, 8)
            }
            .navigationTitle("Relabel Hold \(holdIndex + 1)")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear { selected = currentLabel }
        }
    }
}

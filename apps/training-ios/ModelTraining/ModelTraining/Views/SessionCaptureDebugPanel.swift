// TEMPORARY (session capture Stage 3): a debug panel that starts a session
// capture, so Stage 3 can be tested on a device. Stage 6 replaces it with the
// "Record session" toggle and the session list, and deletes this file.

#if CG_SESSION_CAPTURE

import SwiftUI
import CameraGestures

struct SessionCaptureDebugPanel: View {
    let recognizer: HandGestureRecognizing
    let appSettings: AppSettings
    /// Recording can only be armed while prediction is stopped.
    let isRecognitionActive: Bool

    @State private var message = "Not recording."
    @State private var sessions: [StoredSession] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "record.circle")
                    .foregroundColor(.secondary)
                Text("Session capture (temporary)")
                    .font(.subheadline.weight(.semibold))
                Spacer()
            }

            Button("Record the next prediction run") { arm() }
                .disabled(isRecognitionActive)
            Text(message)
                .font(.caption)
                .foregroundColor(.secondary)

            Divider()

            HStack {
                Text("Stored sessions: \(sessions.count)")
                    .font(.caption.weight(.semibold))
                Spacer()
                Button("Refresh") { sessions = SessionRecorder.storedSessions() }
                    .font(.caption)
            }
            ForEach(sessions.suffix(3), id: \.url) { session in
                Text(describe(session))
                    .font(.caption2.monospaced())
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(8)
        .onAppear { sessions = SessionRecorder.storedSessions() }
        .onChange(of: isRecognitionActive) { active in
            // The recognizer ends the session when prediction stops.
            if !active {
                message = "Not recording."
                sessions = SessionRecorder.storedSessions()
            }
        }
    }

    private func arm() {
        Task { @MainActor in
            guard let consent = await SessionCaptureConsent.request(includesVideo: false) else {
                message = "Consent declined. Nothing is recorded."
                sessions = SessionRecorder.storedSessions()
                return
            }
            do {
                let url = try SessionRecorder(recognizer: recognizer)
                    .start(consent: consent, modelFiles: modelFiles())
                message = "Armed: \(url.lastPathComponent). Tap Start Prediction, then Stop to finish."
            } catch {
                message = "Could not start: \(error)"
            }
        }
    }

    private func modelFiles() -> SessionModelFiles {
        func existing(_ url: URL) -> URL? {
            FileManager.default.fileExists(atPath: url.path) ? url : nil
        }
        return SessionModelFiles(
            gestureModel: existing(appSettings.defaultTFLiteModelURL()),
            gestureIds:   existing(appSettings.defaultGestureIdsURL()),
            poseModel:    existing(appSettings.defaultPoseModelURL()),
            poseManifest: existing(appSettings.defaultPoseManifestURL()),
            preprocessor: existing(appSettings.defaultPreprocessorURL()))
    }

    private func describe(_ s: StoredSession) -> String {
        let state = s.isComplete ? "complete" : (s.isTruncated ? "interrupted, truncated" : "interrupted")
        return "\(s.url.lastPathComponent)\n  \(state), \(String(format: "%.0f", s.duration)) s, "
             + "\(s.shotCount) shots, \(s.telemetryCount) rows, \(s.eventCount) events"
    }
}

#endif

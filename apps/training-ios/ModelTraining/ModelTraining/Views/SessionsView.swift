// Sessions tab: the sessions stored on the device, import of a .cgsession
// from Files, and the timeline player.
//
// The session reader ships only in the capture variant of the library, so the
// whole tab is compiled only with CG_SESSION_CAPTURE.

#if CG_SESSION_CAPTURE

import SwiftUI
import UniformTypeIdentifiers
import CameraGestures

extension UTType {
    /// Declared in the app's Info.plist.
    static let cgSession = UTType(exportedAs: "ru.akurnosov.cameragestures.session")
}

struct SessionsView: View {
    @State private var sessions: [StoredSession] = []
    @State private var player: SessionPlayer?
    @State private var showingImporter = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationView {
            List(sessions.reversed(), id: \.url) { session in
                Button { open(session) } label: { row(session) }
            }
            .overlay {
                if sessions.isEmpty {
                    Text("No sessions.\nImport a .cgsession from Files.")
                        .multilineTextAlignment(.center)
                        .foregroundColor(.secondary)
                }
            }
            .navigationTitle("Sessions")
            .toolbar {
                Button("Import") { showingImporter = true }
            }
        }
        .onAppear { sessions = SessionRecorder.storedSessions() }
        // A bundle is a folder. Files shows it as one document only once it
        // knows the type, so a plain folder can be picked as well.
        .fileImporter(isPresented: $showingImporter,
                      allowedContentTypes: [.cgSession, .folder]) { result in
            do {
                try Self.importSession(from: result.get())
                sessions = SessionRecorder.storedSessions()
            } catch {
                errorMessage = "Import failed: \(error.localizedDescription)"
            }
        }
        .fullScreenCover(item: $player) { player in
            SessionPlayerView(player: player)
        }
        .alert("Sessions", isPresented: Binding(get: { errorMessage != nil },
                                                set: { if !$0 { errorMessage = nil } })) {
            Button("OK") { }
        } message: {
            Text(errorMessage ?? "")
        }
    }

    private func row(_ session: StoredSession) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(session.url.deletingPathExtension().lastPathComponent)
                .foregroundColor(.primary)
            Text("\(formatTime(session.duration)), \(session.shotCount) shots"
                 + (session.isComplete ? "" : ", interrupted"))
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    private func open(_ session: StoredSession) {
        do {
            player = try SessionPlayer(url: session.url)
        } catch {
            errorMessage = "Could not open the session: \(error)"
        }
    }

    /// Copies a bundle into the app's sessions directory. Fails when a session
    /// with the same name is already there, or when the reader cannot open it.
    static func importSession(from source: URL) throws {
        let scoped = source.startAccessingSecurityScopedResource()
        defer { if scoped { source.stopAccessingSecurityScopedResource() } }

        let name = source.deletingPathExtension().lastPathComponent + ".cgsession"
        let destination = try SessionRecorder.sessionsDirectory().appendingPathComponent(name)
        try FileManager.default.copyItem(at: source, to: destination)
        do {
            _ = try SessionReader(url: destination)
        } catch {
            try? FileManager.default.removeItem(at: destination)
            throw error
        }
    }
}

// MARK: - Player

struct SessionPlayerView: View {
    @ObservedObject var player: SessionPlayer
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        let shot = player.currentShot

        VStack(spacing: 12) {
            HStack {
                Text(player.name)
                    .font(.headline)
                    .lineLimit(1)
                Spacer()
                Button("Done") { dismiss() }
            }

            if player.tracks.count > 1 {
                Picker("Hand", selection: $player.selectedTrack) {
                    ForEach(player.tracks, id: \.self) { track in
                        Text("Hand \(track)").tag(track)
                    }
                }
                .pickerStyle(.segmented)
            }

            ZStack {
                Color.black.opacity(0.85)
                HandSkeletonView(points: shot?.landmarks ?? [])
                if shot == nil {
                    Text(player.tracks.isEmpty ? "No hand in this session" : "Hand not in view")
                        .foregroundColor(.secondary)
                }
            }
            .cornerRadius(12)
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            Text(handedness(shot))
                .font(.caption)
                .foregroundColor(.secondary)

            overview
                .frame(height: CGFloat(max(player.tracks.count, 1)) * 10)

            Slider(value: Binding(get: { player.time }, set: { player.seek(to: $0) }),
                   in: 0...max(player.duration, 0.001))

            HStack {
                Button {
                    player.isPlaying ? player.pause() : player.play()
                } label: {
                    Image(systemName: player.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title2)
                        .frame(width: 44, height: 44)
                }
                Spacer()
                Text("\(formatTime(player.time)) / \(formatTime(player.duration))")
                    .font(.body.monospacedDigit())
            }
        }
        .padding()
        .onDisappear { player.pause() }
    }

    /// One band per hand track showing where that hand is in view, with the
    /// playhead on top. The selected track is highlighted.
    private var overview: some View {
        Canvas { context, size in
            let rowHeight = size.height / CGFloat(max(player.overview.count, 1))
            for (rowIndex, row) in player.overview.enumerated() {
                let bucketWidth = size.width / CGFloat(row.count)
                var path = Path()
                for (bucket, present) in row.enumerated() where present {
                    path.addRect(CGRect(x: CGFloat(bucket) * bucketWidth,
                                        y: CGFloat(rowIndex) * rowHeight + 1,
                                        width: bucketWidth,
                                        height: rowHeight - 2))
                }
                let selected = player.tracks[rowIndex] == player.selectedTrack
                context.fill(path, with: .color(selected ? .blue : .gray.opacity(0.5)))
            }
            if player.duration > 0 {
                let x = size.width * CGFloat(player.time / player.duration)
                context.fill(Path(CGRect(x: x - 0.5, y: 0, width: 1, height: size.height)),
                             with: .color(.primary))
            }
        }
        .background(Color.gray.opacity(0.15))
    }

    private func handedness(_ shot: HandShot?) -> String {
        switch shot?.leftOrRight {
        case .left:    return "Left hand"
        case .right:   return "Right hand"
        case .unknown: return "Handedness unknown"
        case nil:      return " "
        }
    }
}

/// Minutes, seconds and tenths, e.g. "9:59.4".
private func formatTime(_ seconds: TimeInterval) -> String {
    let tenths = Int((seconds * 10).rounded(.down))
    return String(format: "%d:%02d.%d", tenths / 600, tenths / 10 % 60, tenths % 10)
}

#endif

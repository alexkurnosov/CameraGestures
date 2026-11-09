// Sessions tab: the sessions stored on the device with their size and length,
// import of a .cgsession from Files, export through the share sheet, deletion,
// and the timeline player.
//
// The session reader ships only in the capture variant of the library, so the
// whole tab is compiled only with CG_SESSION_CAPTURE.

#if CG_SESSION_CAPTURE

import SwiftUI
import UniformTypeIdentifiers
import AVFoundation
import CameraGestures

extension UTType {
    /// Declared in the app's Info.plist.
    static let cgSession = UTType(exportedAs: "ru.akurnosov.cameragestures.session")
}

struct SessionsView: View {
    @EnvironmentObject var gestureRecognizer: GestureRecognizerWrapper

    @State private var sessions: [StoredSession] = []
    /// Bytes on disk per bundle, measured when the list is loaded.
    @State private var sizes: [URL: Int64] = [:]
    @State private var player: SessionPlayer?
    @State private var showingImporter = false
    @State private var errorMessage: String?
    @State private var sharing: SharedSession?
    @State private var pendingDeletion: StoredSession?

    var body: some View {
        NavigationView {
            List(sessions.reversed(), id: \.url) { session in
                let recording = isBeingRecorded(session)
                Button { open(session) } label: { row(session, recording: recording) }
                    .swipeActions(edge: .trailing) {
                        // The Camera tab is still writing it.
                        if !recording {
                            Button("Delete", role: .destructive) { pendingDeletion = session }
                            Button("Share") { sharing = SharedSession(url: session.url) }
                                .tint(.blue)
                        }
                    }
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
        .onAppear { reload() }
        // A recording that has just ended is now complete.
        .onChange(of: gestureRecognizer.recordingSessionURL) { _ in reload() }
        // A bundle is a folder. Files shows it as one document only once it
        // knows the type, so a plain folder can be picked as well.
        .fileImporter(isPresented: $showingImporter,
                      allowedContentTypes: [.cgSession, .folder]) { result in
            do {
                try Self.importSession(from: result.get())
                reload()
            } catch {
                errorMessage = "Import failed: \(error.localizedDescription)"
            }
        }
        .fullScreenCover(item: $player) { player in
            SessionPlayerView(player: player)
        }
        .sheet(item: $sharing) { shared in
            ActivityView(items: [shared.url])
        }
        .alert("Delete this session?",
               isPresented: Binding(get: { pendingDeletion != nil },
                                    set: { if !$0 { pendingDeletion = nil } }),
               presenting: pendingDeletion) { session in
            Button("Delete", role: .destructive) { delete(session) }
            Button("Cancel", role: .cancel) { }
        } message: { session in
            Text("\(session.url.deletingPathExtension().lastPathComponent) will be removed from this device. "
                 + "A recording cannot be made again.")
        }
        .alert("Sessions", isPresented: Binding(get: { errorMessage != nil },
                                                set: { if !$0 { errorMessage = nil } })) {
            Button("OK") { }
        } message: {
            Text(errorMessage ?? "")
        }
    }

    private func row(_ session: StoredSession, recording: Bool) -> some View {
        let size = sizes[session.url].map {
            ByteCountFormatter.string(fromByteCount: $0, countStyle: .file)
        } ?? "?"
        return VStack(alignment: .leading, spacing: 2) {
            Text(session.url.deletingPathExtension().lastPathComponent)
                .foregroundColor(.primary)
            if recording {
                Text("Recording")
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.red)
            } else {
                Text("\(formatTime(session.duration)), \(size), \(session.shotCount) shots"
                     + (session.isComplete ? "" : ", interrupted"))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }

    private func reload() {
        sessions = SessionRecorder.storedSessions()
        sizes = Dictionary(uniqueKeysWithValues: sessions.map { ($0.url, Self.bundleSize($0.url)) })
    }

    private func isBeingRecorded(_ session: StoredSession) -> Bool {
        // By name: both are in the sessions directory, but one URL may carry
        // a trailing slash or a /private prefix the other lacks.
        session.url.lastPathComponent == gestureRecognizer.recordingSessionURL?.lastPathComponent
            && SessionRecorder(recognizer: gestureRecognizer.recognizer).isRecording
    }

    private func delete(_ session: StoredSession) {
        do {
            try FileManager.default.removeItem(at: session.url)
        } catch {
            errorMessage = "Could not delete the session: \(error.localizedDescription)"
        }
        reload()
    }

    /// Sum of the file sizes in a bundle.
    static func bundleSize(_ url: URL) -> Int64 {
        let files = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey])
        var total: Int64 = 0
        while let file = files?.nextObject() as? URL {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        }
        return total
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

// MARK: - Share sheet

/// A bundle handed to the share sheet; `sheet(item:)` needs an identity.
private struct SharedSession: Identifiable {
    let url: URL
    var id: URL { url }
}

/// The bundle goes out as the .cgsession folder, so the receiver gets the
/// same bundle the reader opens.
private struct ActivityView: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}

// MARK: - Player

struct SessionPlayerView: View {
    @ObservedObject var player: SessionPlayer
    @Environment(\.dismiss) private var dismiss

    /// Panel width as a share of the page width, set by pinching the panel.
    @State private var panelScale: CGFloat = 1

    var body: some View {
        VStack(spacing: 12) {
            HStack {
                Text(player.name)
                    .font(.headline)
                    .lineLimit(1)
                Spacer()
                Button("Done") { dismiss() }
            }

            // The panel can take the full width, so the charts below scroll.
            GeometryReader { page in
                ScrollView {
                    VStack(spacing: 12) {
                        if player.tracks.count > 1 {
                            HandPicker(tracks: player.tracks) { player.trackFilter = $0 }
                                .equatable()
                        }

                        panel
                            .frame(width: page.size.width * panelScale)

                        Text(handedness(player.currentShots))
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

                        SessionLayersView(player: player)
                    }
                }
            }
        }
        .padding()
        .onDisappear { player.pause() }
    }

    /// Video and skeletons at the video's own shape. Pinching sets the panel's
    /// size up to the page width and zooms into the picture beyond it. A drag
    /// pans while zoomed, and a double tap shows the whole frame again.
    private var panel: some View {
        let shots = player.currentShots
        let shape = uprightVideoSize(player.video) ?? CGSize(width: 3, height: 4)

        return ZStack {
            Color.black.opacity(0.85)
            ZoomableView(panelScale: $panelScale, content: ZStack {
                if let avPlayer = player.avPlayer, player.videoState == .showing {
                    SessionVideoView(player: avPlayer)
                }
                SessionHandsOverlay(hands: shots.map { $0.shot.landmarks }, video: player.video)
            })
            Group {
                if shots.isEmpty, player.videoState != .showing {
                    Text(player.tracks.isEmpty ? "No hand in this session" : "Hand not in view")
                        .foregroundColor(.secondary)
                }
                if let note = videoNote {
                    Text(note)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(8)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                }
            }
            // The notes cover the panel; touches go through to the zoom.
            .allowsHitTesting(false)
        }
        .aspectRatio(shape, contentMode: .fit)
        .cornerRadius(12)
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
                let selected = player.shownTracks.contains(player.tracks[rowIndex])
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

    /// What the panel says about the video, if anything.
    private var videoNote: String? {
        switch player.videoState {
        case .unavailable: return "Video unavailable (recorded without video)"
        case .failed:      return "Video could not be loaded: \(player.videoError ?? "")"
        case .loading:     return "Loading video"
        case .hidden:      return nil
        case .outOfRange:  return "No video at this time"
        case .showing:     return player.video?.mirrored == true ? "Mirrored video" : nil
        }
    }

    /// Each shown hand's track and MediaPipe's label for it, e.g. "Hand 0: left".
    private func handedness(_ shots: [(track: Int, shot: HandShot)]) -> String {
        guard !shots.isEmpty else { return " " }
        return shots.map { hand in
            switch hand.shot.leftOrRight {
            case .left:    return "Hand \(hand.track): left"
            case .right:   return "Hand \(hand.track): right"
            case .unknown: return "Hand \(hand.track): handedness unknown"
            }
        }.joined(separator: ", ")
    }
}

// MARK: - Hand picker

/// A view of its own that redraws only when the tracks change. While playing,
/// the playhead redraws the player many times a second, and a segmented picker
/// redrawn that often dropped taps.
private struct HandPicker: View, Equatable {
    let tracks: [Int]
    let onSelect: (Int?) -> Void

    /// The track shown; nil shows every hand.
    @State private var selection: Int?

    static func == (lhs: HandPicker, rhs: HandPicker) -> Bool {
        lhs.tracks == rhs.tracks
    }

    var body: some View {
        Picker("Hand", selection: $selection) {
            Text("Both").tag(Int?.none)
            ForEach(tracks, id: \.self) { track in
                Text("Hand \(track)").tag(Int?.some(track))
            }
        }
        .pickerStyle(.segmented)
        .onChange(of: selection) { onSelect($0) }
    }
}

// MARK: - Video and hand overlay

/// Pinch zoom and pan from UIScrollView, which works inside the page's
/// ScrollView where a SwiftUI pinch gesture did not. A pinch that ends below
/// the panel's size resizes the panel instead, so the size stays. A double tap
/// zooms out.
private struct ZoomableView<Content: View>: UIViewRepresentable {
    @Binding var panelScale: CGFloat
    let content: Content

    /// The smallest panel, as a share of the page width.
    private static var minPanelScale: CGFloat { 0.3 }

    func makeCoordinator() -> Coordinator {
        Coordinator(host: UIHostingController(rootView: content))
    }

    func makeUIView(context: Context) -> UIScrollView {
        let scrollView = UIScrollView()
        scrollView.delegate = context.coordinator
        scrollView.maximumZoomScale = 6
        scrollView.showsHorizontalScrollIndicator = false
        scrollView.showsVerticalScrollIndicator = false
        scrollView.backgroundColor = .clear

        let hosted: UIView = context.coordinator.host.view
        hosted.backgroundColor = .clear
        hosted.frame = scrollView.bounds
        hosted.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        scrollView.addSubview(hosted)

        let doubleTap = UITapGestureRecognizer(target: context.coordinator,
                                               action: #selector(Coordinator.zoomOut(_:)))
        doubleTap.numberOfTapsRequired = 2
        scrollView.addGestureRecognizer(doubleTap)
        return scrollView
    }

    func updateUIView(_ scrollView: UIScrollView, context: Context) {
        context.coordinator.host.rootView = content
        context.coordinator.panelScale = $panelScale
        // Zooming out below 1 shrinks the picture; the panel follows when the
        // pinch ends.
        scrollView.minimumZoomScale = Self.minPanelScale / panelScale
    }

    final class Coordinator: NSObject, UIScrollViewDelegate {
        let host: UIHostingController<Content>
        var panelScale: Binding<CGFloat>?

        init(host: UIHostingController<Content>) {
            self.host = host
        }

        func viewForZooming(in scrollView: UIScrollView) -> UIView? {
            host.view
        }

        /// Keeps a picture smaller than the panel in its centre.
        func scrollViewDidZoom(_ scrollView: UIScrollView) {
            let dx = max((scrollView.bounds.width - scrollView.contentSize.width) / 2, 0)
            let dy = max((scrollView.bounds.height - scrollView.contentSize.height) / 2, 0)
            scrollView.contentInset = UIEdgeInsets(top: dy, left: dx, bottom: dy, right: dx)
        }

        func scrollViewDidEndZooming(_ scrollView: UIScrollView, with view: UIView?, atScale scale: CGFloat) {
            guard let panelScale else { return }
            let current = panelScale.wrappedValue
            // Below the page width, the pinch sets the panel's size and the
            // picture fills the panel again.
            guard scale < 1 || current < 1 else { return }
            let newScale = min(max(current * scale, ZoomableView.minPanelScale), 1)
            scrollView.setZoomScale(1, animated: false)
            scrollView.contentInset = .zero
            panelScale.wrappedValue = newScale
        }

        @objc func zoomOut(_ gesture: UITapGestureRecognizer) {
            (gesture.view as? UIScrollView)?.setZoomScale(1, animated: true)
        }
    }
}

/// The session video, fitted to the view and turned upright by the file's own
/// rotation.
private struct SessionVideoView: UIViewRepresentable {
    let player: AVPlayer

    func makeUIView(context: Context) -> PlayerView {
        let view = PlayerView()
        view.playerLayer.videoGravity = .resizeAspect
        view.playerLayer.player = player
        return view
    }

    func updateUIView(_ view: PlayerView, context: Context) {}

    final class PlayerView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        var playerLayer: AVPlayerLayer { layer as! AVPlayerLayer }
    }
}

/// Hand skeletons drawn where the video shows its frames. Landmarks are
/// normalised to the stored frame, so they get the same rotation and fit as the
/// video. The mirrored flag flips neither: the frames and the landmarks are
/// both as MediaPipe saw them. Without video they fill the view, as on the
/// Camera tab.
private struct SessionHandsOverlay: View {
    let hands: [[Point3D]]
    let video: SessionVideoInfo?

    private static let connections: [(Int, Int)] = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20)
    ]

    var body: some View {
        Canvas { context, size in
            let rect = contentRect(in: size)
            let rotation = video?.rotationDegrees ?? 0
            for points in hands where points.count >= 21 {
                let placed = points.map { point -> CGPoint in
                    let p = Self.upright(point, rotation: rotation)
                    return CGPoint(x: rect.minX + p.x * rect.width, y: rect.minY + p.y * rect.height)
                }
                var bones = Path()
                for (a, b) in Self.connections {
                    bones.move(to: placed[a])
                    bones.addLine(to: placed[b])
                }
                context.stroke(bones, with: .color(.blue), lineWidth: 2)
                for p in placed {
                    context.stroke(Path(ellipseIn: CGRect(x: p.x - 3, y: p.y - 3, width: 6, height: 6)),
                                   with: .color(.green), lineWidth: 2)
                }
            }
        }
    }

    /// Where an aspect fit puts the upright video, as AVPlayerLayer does.
    private func contentRect(in size: CGSize) -> CGRect {
        let bounds = CGRect(origin: .zero, size: size)
        guard let upright = uprightVideoSize(video) else { return bounds }
        return AVMakeRect(aspectRatio: upright, insideRect: bounds)
    }

    /// A stored-frame point in the upright frame, both normalised, for a
    /// clockwise rotation.
    private static func upright(_ point: Point3D, rotation: Int) -> CGPoint {
        let x = CGFloat(point.x), y = CGFloat(point.y)
        switch rotation {
        case 90:  return CGPoint(x: 1 - y, y: x)
        case 180: return CGPoint(x: 1 - x, y: 1 - y)
        case 270: return CGPoint(x: y, y: 1 - x)
        default:  return CGPoint(x: x, y: y)
        }
    }
}

/// The video frame's size once turned upright; nil without video.
private func uprightVideoSize(_ video: SessionVideoInfo?) -> CGSize? {
    guard let video else { return nil }
    return video.rotationDegrees % 180 != 0 ? CGSize(width: video.height, height: video.width)
                                            : CGSize(width: video.width, height: video.height)
}

/// Minutes, seconds and tenths, e.g. "9:59.4".
private func formatTime(_ seconds: TimeInterval) -> String {
    let tenths = Int((seconds * 10).rounded(.down))
    return String(format: "%d:%02d.%d", tenths / 600, tenths / 10 % 60, tenths % 10)
}

#endif

// Plays one .cgsession on the session's own time axis.
//
// FilmPlaybackManager steps a frame index on a fixed 24 fps timer. A session
// has several tracks with uneven frame intervals, so this player keeps a
// playhead in seconds and looks up the shot that was current at that time.
//
// With video, the video leads: while the playhead is inside the video and the
// video is shown, the playhead is read from the video's position through the
// session clock. Elsewhere the playhead follows the wall clock.

#if CG_SESSION_CAPTURE

import Foundation
import Combine
import QuartzCore
import AVFoundation
import CameraGestures

@MainActor
final class SessionPlayer: NSObject, ObservableObject, Identifiable {

    /// Buckets in the overview, over the whole session.
    nonisolated static let overviewBuckets = 600
    /// A hand track has no shots while its hand is out of view. A shot older
    /// than this is not shown, so a hand that left does not stay on screen.
    private static let staleAfter: TimeInterval = 0.2
    /// Shots kept in memory around the playhead.
    private static let windowSize = 256

    let name: String
    /// Seconds between the first and the last shot.
    let duration: TimeInterval
    /// MediaPipe hand indices with shots.
    let tracks: [Int]

    /// Playhead, in seconds from the first shot.
    @Published private(set) var time: TimeInterval = 0
    @Published private(set) var isPlaying = false
    /// The hand shown; nil shows every hand.
    @Published var trackFilter: Int?
    /// One row per entry of `tracks`: whether the hand has a shot in each
    /// bucket. Empty until the load-time pass over the session finishes.
    @Published private(set) var overview: [[Bool]] = []
    /// The library's decisions, on the playhead's time axis. Nil until loaded.
    @Published private(set) var layers: SessionLayers?

    /// Nil when the session has no video.
    let video: SessionVideoInfo?
    let avPlayer: AVPlayer?
    @Published var showsVideo = true {
        didSet { if showsVideo != oldValue { seek(to: time) } }
    }
    /// The playhead positions the video covers. Nil until the video and the
    /// session clock are loaded.
    @Published private(set) var videoRange: Range<TimeInterval>?
    /// Why the video could not be loaded; nil while loading or once loaded.
    @Published private(set) var videoError: String?

    private let reader: SessionReader
    private let startTime: TimeInterval
    private var displayLink: CADisplayLink?
    private var anchorWallTime: CFTimeInterval = 0
    private var anchorTime: TimeInterval = 0
    /// Per track, the shots kept around the playhead.
    private var windows: [Int: (first: Int, shots: [HandShot])] = [:]

    private var timing: VideoTiming?
    /// The video is playing and sets the playhead.
    private var videoDriving = false
    private var seekInFlight = false
    private var pendingSeek: CMTime?

    init(url: URL) throws {
        let reader = try SessionReader(url: url)
        let range = reader.timeRange ?? 0...0
        let tracks = reader.handTracks
        self.reader = reader
        self.name = url.deletingPathExtension().lastPathComponent
        self.startTime = range.lowerBound
        self.duration = range.upperBound - range.lowerBound
        self.tracks = tracks
        if let video = reader.video, FileManager.default.fileExists(atPath: video.url.path) {
            self.video = video
            let avPlayer = AVPlayer(url: video.url)
            avPlayer.automaticallyWaitsToMinimizeStalling = false
            self.avPlayer = avPlayer
        } else {
            self.video = nil
            self.avPlayer = nil
        }
        super.init()

        let start = startTime, duration = duration
        Task {
            // A second reader, because a reader must stay on one thread.
            overview = await Task.detached {
                SessionPlayer.makeOverview(url: url, tracks: tracks, start: start, duration: duration)
            }.value
        }
        Task {
            layers = await Task.detached {
                SessionLayers(url: url, start: start)
            }.value
        }
        if let video, let item = avPlayer?.currentItem {
            Task { await loadVideo(video, item: item, url: url) }
        }
    }

    // MARK: - Current shots

    /// The tracks the hand filter shows.
    var shownTracks: [Int] { trackFilter.map { [$0] } ?? tracks }

    /// The shown hands' shots at the playhead. A hand out of view has none.
    var currentShots: [(track: Int, shot: HandShot)] {
        shownTracks.compactMap { track in currentShot(track: track).map { (track, $0) } }
    }

    private func currentShot(track: Int) -> HandShot? {
        let now = startTime + time
        let count = reader.shotCount(track: track)
        var index = reader.firstShotIndex(track: track, atOrAfter: now)
        // The shot on screen is the latest one at or before the playhead.
        if index == count || shot(track: track, at: index).timestamp > now { index -= 1 }
        guard index >= 0 else { return nil }
        let shot = shot(track: track, at: index)
        return now - shot.timestamp <= Self.staleAfter ? shot : nil
    }

    private func shot(track: Int, at index: Int) -> HandShot {
        if let window = windows[track],
           index >= window.first, index < window.first + window.shots.count {
            return window.shots[index - window.first]
        }
        let first = index - index % Self.windowSize
        let shots = reader.shots(track: track, from: first, count: Self.windowSize)
        windows[track] = (first, shots)
        return shots[index - first]
    }

    // MARK: - Video

    enum VideoState {
        case unavailable, failed, loading, hidden, outOfRange, showing
    }

    var videoState: VideoState {
        guard video != nil else { return .unavailable }
        guard videoError == nil else { return .failed }
        guard let videoRange else { return .loading }
        guard showsVideo else { return .hidden }
        return videoRange.contains(time) ? .showing : .outOfRange
    }

    /// Video duration and the session clock, which places the playhead on it.
    private struct VideoTiming {
        let clock: SessionClock
        let anchorPTS: TimeInterval
        let startTime: TimeInterval
        let duration: TimeInterval

        func videoTime(atPlayhead t: TimeInterval) -> TimeInterval {
            clock.pts(atTimestamp: startTime + t) - anchorPTS
        }

        func playhead(atVideoTime v: TimeInterval) -> TimeInterval {
            clock.timestamp(atPTS: anchorPTS + v) - startTime
        }
    }

    private var usesVideo: Bool { timing != nil && showsVideo }

    private func loadVideo(_ video: SessionVideoInfo, item: AVPlayerItem, url: URL) async {
        let clock = await Task.detached { () -> SessionClock? in
            // A second reader, because a reader must stay on one thread.
            guard let reader = try? SessionReader(url: url) else { return nil }
            return SessionClock(reader: reader)
        }.value
        guard let clock else {
            videoError = "no shot has a capture timestamp"
            return
        }
        let duration: TimeInterval
        do {
            duration = try await item.asset.load(.duration).seconds
        } catch {
            videoError = "duration: \(error.localizedDescription)"
            return
        }
        guard duration > 0 else {
            videoError = "the video is empty"
            return
        }
        // Seeks before the item is ready do not show a frame. Polled, because
        // a KVO stream drops a change that arrives while no value is awaited,
        // and a short video became ready in that gap and was never seen.
        while item.status == .unknown {
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        if item.status == .failed {
            videoError = item.error?.localizedDescription ?? "the player failed"
            return
        }
        let timing = VideoTiming(clock: clock, anchorPTS: video.anchorPTS,
                                 startTime: startTime, duration: duration)
        self.timing = timing
        videoRange = timing.playhead(atVideoTime: 0)..<timing.playhead(atVideoTime: duration)
        seek(to: time)
    }

    /// Exact seeks, one at a time. While one runs, a newer target waits and
    /// replaces any older one.
    private func seekVideo(toPlayhead t: TimeInterval) {
        guard let timing else { return }
        // A millisecond past the frame's own time, so rounding never lands on
        // the frame before it.
        let v = min(max(timing.videoTime(atPlayhead: t), 0), timing.duration) + 0.001
        let target = CMTime(seconds: v, preferredTimescale: 1_000_000)
        if seekInFlight {
            pendingSeek = target
        } else {
            startSeek(target)
        }
    }

    private func startSeek(_ target: CMTime) {
        seekInFlight = true
        avPlayer?.seek(to: target, toleranceBefore: .zero, toleranceAfter: .zero) { [weak self] _ in
            Task { @MainActor in self?.seekFinished() }
        }
    }

    private func seekFinished() {
        seekInFlight = false
        if let next = pendingSeek {
            pendingSeek = nil
            startSeek(next)
            return
        }
        guard isPlaying else { return }
        if usesVideo, videoRange?.contains(time) == true {
            avPlayer?.play()
            videoDriving = true
        } else {
            setAnchor()
        }
    }

    // MARK: - Transport

    func play() {
        guard !isPlaying, duration > 0 else { return }
        let link = CADisplayLink(target: self, selector: #selector(tick))
        link.add(to: .main, forMode: .common)
        displayLink = link
        isPlaying = true
        // Puts the video on the playhead's frame before either moves.
        seek(to: time >= duration ? 0 : time)
    }

    /// Also releases the display link, which holds the player.
    func pause() {
        if videoDriving, let timing, let avPlayer {
            time = min(timing.playhead(atVideoTime: avPlayer.currentTime().seconds), duration)
        }
        videoDriving = false
        avPlayer?.pause()
        displayLink?.invalidate()
        displayLink = nil
        isPlaying = false
    }

    func seek(to newTime: TimeInterval) {
        time = min(max(newTime, 0), duration)
        setAnchor()
        if videoDriving {
            videoDriving = false
            avPlayer?.pause()
        }
        if usesVideo { seekVideo(toPlayhead: time) }
    }

    private func setAnchor() {
        anchorWallTime = CACurrentMediaTime()
        anchorTime = time
    }

    @objc private func tick() {
        // The playhead waits while the video seeks, so the two start together.
        if seekInFlight {
            setAnchor()
            return
        }
        if videoDriving, let timing, let avPlayer {
            let v = avPlayer.currentTime().seconds
            if avPlayer.rate != 0, v < timing.duration {
                advance(to: timing.playhead(atVideoTime: v))
                return
            }
            // The video ended. The wall clock carries on from its last frame.
            videoDriving = false
            time = min(videoRange?.upperBound ?? time, duration)
            setAnchor()
        }
        // The playhead follows the wall clock, not the number of ticks, so a
        // slow frame does not slow the session down.
        let newTime = anchorTime + CACurrentMediaTime() - anchorWallTime
        advance(to: newTime)
        if isPlaying, usesVideo, videoRange?.contains(newTime) == true {
            // Entering the video. It takes over once it is on the frame.
            seekVideo(toPlayhead: newTime)
        }
    }

    private func advance(to newTime: TimeInterval) {
        if newTime >= duration {
            time = duration
            pause()
        } else {
            time = newTime
        }
    }

    // MARK: - Overview

    nonisolated private static func makeOverview(url: URL, tracks: [Int],
                                                 start: TimeInterval,
                                                 duration: TimeInterval) -> [[Bool]] {
        guard let reader = try? SessionReader(url: url) else { return [] }
        let scale = Double(overviewBuckets) / max(duration, .leastNonzeroMagnitude)
        return tracks.map { track in
            var row = [Bool](repeating: false, count: overviewBuckets)
            let count = reader.shotCount(track: track)
            var first = 0
            while first < count {
                let shots = reader.shots(track: track, from: first, count: 512)
                if shots.isEmpty { break }
                for shot in shots {
                    let bucket = Int((shot.timestamp - start) * scale)
                    row[min(max(bucket, 0), overviewBuckets - 1)] = true
                }
                first += shots.count
            }
            return row
        }
    }
}

#endif

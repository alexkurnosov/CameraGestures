// Plays one .cgsession on the session's own time axis.
//
// FilmPlaybackManager steps a frame index on a fixed 24 fps timer. A session
// has several tracks with uneven frame intervals, so this player keeps a
// playhead in seconds and looks up the shot that was current at that time.

#if CG_SESSION_CAPTURE

import Foundation
import Combine
import QuartzCore
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
    @Published var selectedTrack: Int
    /// One row per entry of `tracks`: whether the hand has a shot in each
    /// bucket. Empty until the load-time pass over the session finishes.
    @Published private(set) var overview: [[Bool]] = []

    private let reader: SessionReader
    private let startTime: TimeInterval
    private var displayLink: CADisplayLink?
    private var anchorWallTime: CFTimeInterval = 0
    private var anchorTime: TimeInterval = 0
    private var window: (track: Int, first: Int, shots: [HandShot]) = (-1, 0, [])

    init(url: URL) throws {
        let reader = try SessionReader(url: url)
        let range = reader.timeRange ?? 0...0
        let tracks = reader.handTracks
        self.reader = reader
        self.name = url.deletingPathExtension().lastPathComponent
        self.startTime = range.lowerBound
        self.duration = range.upperBound - range.lowerBound
        self.tracks = tracks
        self.selectedTrack = tracks.first ?? 0
        super.init()

        let start = startTime, duration = duration
        Task {
            // A second reader, because a reader must stay on one thread.
            overview = await Task.detached {
                SessionPlayer.makeOverview(url: url, tracks: tracks, start: start, duration: duration)
            }.value
        }
    }

    // MARK: - Current shot

    /// The selected hand's shot at the playhead; nil while that hand is out of view.
    var currentShot: HandShot? {
        let now = startTime + time
        let count = reader.shotCount(track: selectedTrack)
        var index = reader.firstShotIndex(track: selectedTrack, atOrAfter: now)
        // The shot on screen is the latest one at or before the playhead.
        if index == count || shot(at: index).timestamp > now { index -= 1 }
        guard index >= 0 else { return nil }
        let shot = shot(at: index)
        return now - shot.timestamp <= Self.staleAfter ? shot : nil
    }

    private func shot(at index: Int) -> HandShot {
        if window.track != selectedTrack
            || index < window.first || index >= window.first + window.shots.count {
            let first = index - index % Self.windowSize
            window = (selectedTrack, first,
                      reader.shots(track: selectedTrack, from: first, count: Self.windowSize))
        }
        return window.shots[index - window.first]
    }

    // MARK: - Transport

    func play() {
        guard !isPlaying, duration > 0 else { return }
        if time >= duration { time = 0 }
        setAnchor()
        let link = CADisplayLink(target: self, selector: #selector(tick))
        link.add(to: .main, forMode: .common)
        displayLink = link
        isPlaying = true
    }

    /// Also releases the display link, which holds the player.
    func pause() {
        displayLink?.invalidate()
        displayLink = nil
        isPlaying = false
    }

    func seek(to newTime: TimeInterval) {
        time = min(max(newTime, 0), duration)
        setAnchor()
    }

    private func setAnchor() {
        anchorWallTime = CACurrentMediaTime()
        anchorTime = time
    }

    @objc private func tick() {
        // The playhead follows the wall clock, not the number of ticks, so a
        // slow frame does not slow the session down.
        let newTime = anchorTime + CACurrentMediaTime() - anchorWallTime
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

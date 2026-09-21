// Frame-cadence instrumentation for the landmark path.
//
// Motion energy is a per-frame delta with no dt normalisation, so the gate's
// behaviour depends on how fast and how evenly handshots arrive. Anything that
// steals time from the camera → MediaPipe → pipeline path (session capture, a
// video writer, thermal throttling) makes the gate measurably hotter.
//
// This monitor is the instrument that measurement is taken with. It lives in
// the library rather than in an app so that a baseline recorded by one build is
// comparable with a run recorded by another — the numbers come off the same
// counters either way.
//
// Cost when nobody reads it: four integer increments and one array append per
// frame, under a lock that is never contended for more than that.

import Foundation

// MARK: - FrameRateStats

/// A snapshot of the landmark path's cadence since the last reset.
///
/// Counters are cumulative over the window; the interval percentiles describe
/// the gaps between consecutive *frames* (distinct handshot timestamps), which
/// is the cadence the motion gate actually sees.
public struct FrameRateStats: Equatable {

    /// Seconds since the monitor was last reset (monotonic clock).
    public let elapsed: TimeInterval

    /// Sample buffers delivered by `AVCaptureVideoDataOutput`.
    public let cameraFrames: Int
    /// Sample buffers `AVFoundation` discarded before the delegate saw them.
    public let cameraDrops: Int
    /// Detection callbacks returned by MediaPipe.
    public let landmarkerResults: Int

    /// Real (non-absent) handshots pushed into the pipeline. With two hands in
    /// frame this is roughly twice `frames`.
    public let handshots: Int
    /// Absent-hand placeholders pushed into the pipeline.
    public let absentShots: Int
    /// Distinct handshot timestamps — i.e. frames that produced at least one shot.
    public let frames: Int

    /// Inter-frame intervals in milliseconds, as percentiles over `intervalCount`
    /// samples. Zero when fewer than two frames have been seen.
    public let p50IntervalMs: Double
    public let p90IntervalMs: Double
    public let p99IntervalMs: Double
    public let maxIntervalMs: Double
    public let intervalCount: Int

    public init(elapsed: TimeInterval = 0,
                cameraFrames: Int = 0,
                cameraDrops: Int = 0,
                landmarkerResults: Int = 0,
                handshots: Int = 0,
                absentShots: Int = 0,
                frames: Int = 0,
                p50IntervalMs: Double = 0,
                p90IntervalMs: Double = 0,
                p99IntervalMs: Double = 0,
                maxIntervalMs: Double = 0,
                intervalCount: Int = 0) {
        self.elapsed           = elapsed
        self.cameraFrames      = cameraFrames
        self.cameraDrops       = cameraDrops
        self.landmarkerResults = landmarkerResults
        self.handshots         = handshots
        self.absentShots       = absentShots
        self.frames            = frames
        self.p50IntervalMs     = p50IntervalMs
        self.p90IntervalMs     = p90IntervalMs
        self.p99IntervalMs     = p99IntervalMs
        self.maxIntervalMs     = maxIntervalMs
        self.intervalCount     = intervalCount
    }

    /// Camera frames per second.
    public var cameraRate: Double { elapsed > 0 ? Double(cameraFrames) / elapsed : 0 }

    /// Frames per second that reached the pipeline — the number to compare
    /// across builds, since it is independent of how many hands are in view.
    public var frameRate: Double { elapsed > 0 ? Double(frames) / elapsed : 0 }

    /// Real handshots per second. Hand-count dependent; reported alongside
    /// `frameRate` rather than instead of it.
    public var handshotRate: Double { elapsed > 0 ? Double(handshots) / elapsed : 0 }

    /// Camera frames that produced no detection callback. MediaPipe's
    /// live-stream mode discards frames silently while it is busy, so this is
    /// the channel a capture-induced stall shows up on first.
    ///
    /// One or two frames may be in flight when the snapshot is taken, so treat
    /// small non-zero values over a short window as noise.
    public var landmarkerDrops: Int { max(0, cameraFrames - landmarkerResults) }

    /// Fraction of camera frames lost either before or inside MediaPipe.
    public var dropFraction: Double {
        cameraFrames > 0 ? Double(cameraDrops + landmarkerDrops) / Double(cameraFrames) : 0
    }
}

// MARK: - FrameRateMonitor

/// Thread-safe counter set behind `FrameRateStats`.
///
/// Every `note*` method may be called from the capture queue, the MediaPipe
/// delegate queue, or both; `snapshot()` is typically called from the main
/// thread on a timer.
public final class FrameRateMonitor {

    /// Cap on retained interval samples. 32768 covers ~18 minutes at 30 fps;
    /// beyond that the oldest samples are dropped, which biases the percentiles
    /// toward the end of the run rather than truncating it.
    private static let maxIntervalSamples = 32_768

    private let lock = NSLock()

    private var startedAt: TimeInterval = ProcessInfo.processInfo.systemUptime
    private var cameraFrames = 0
    private var cameraDrops = 0
    private var landmarkerResults = 0
    private var handshots = 0
    private var absentShots = 0
    private var frames = 0

    private var lastShotTimestamp: TimeInterval?
    private var intervalsMs: [Double] = []

    public init() {
        intervalsMs.reserveCapacity(4_096)
    }

    /// Clears every counter and restarts the window. Called on each capture start.
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        startedAt         = ProcessInfo.processInfo.systemUptime
        cameraFrames      = 0
        cameraDrops       = 0
        landmarkerResults = 0
        handshots         = 0
        absentShots       = 0
        frames            = 0
        lastShotTimestamp = nil
        intervalsMs.removeAll(keepingCapacity: true)
    }

    public func noteCameraFrame() {
        lock.lock(); cameraFrames += 1; lock.unlock()
    }

    public func noteCameraDrop() {
        lock.lock(); cameraDrops += 1; lock.unlock()
    }

    public func noteLandmarkerResult() {
        lock.lock(); landmarkerResults += 1; lock.unlock()
    }

    /// Records one handshot pushed into the pipeline.
    ///
    /// - Parameters:
    ///   - timestamp: the shot's own timestamp, so intervals are measured on the
    ///     same clock the gate computes energy against.
    ///   - isAbsent: absent placeholders count as frames — the gate processes
    ///     them — but not as handshots.
    public func noteShot(timestamp: TimeInterval, isAbsent: Bool) {
        lock.lock()
        defer { lock.unlock() }

        if isAbsent { absentShots += 1 } else { handshots += 1 }

        // Several shots share a timestamp when more than one hand is in frame;
        // only the first of them starts a new frame.
        guard timestamp != lastShotTimestamp else { return }
        if let previous = lastShotTimestamp {
            let deltaMs = (timestamp - previous) * 1000.0
            // A non-monotonic timestamp would poison the distribution.
            if deltaMs > 0 {
                intervalsMs.append(deltaMs)
                if intervalsMs.count > Self.maxIntervalSamples {
                    intervalsMs.removeFirst(intervalsMs.count - Self.maxIntervalSamples)
                }
            }
        }
        lastShotTimestamp = timestamp
        frames += 1
    }

    public func snapshot() -> FrameRateStats {
        lock.lock()
        defer { lock.unlock() }

        let sorted = intervalsMs.sorted()
        return FrameRateStats(
            elapsed:           ProcessInfo.processInfo.systemUptime - startedAt,
            cameraFrames:      cameraFrames,
            cameraDrops:       cameraDrops,
            landmarkerResults: landmarkerResults,
            handshots:         handshots,
            absentShots:       absentShots,
            frames:            frames,
            p50IntervalMs:     Self.percentile(sorted, 0.50),
            p90IntervalMs:     Self.percentile(sorted, 0.90),
            p99IntervalMs:     Self.percentile(sorted, 0.99),
            maxIntervalMs:     sorted.last ?? 0,
            intervalCount:     sorted.count)
    }

    /// Nearest-rank percentile over an already-sorted array.
    private static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let rank = Int((p * Double(sorted.count)).rounded(.up))
        return sorted[min(max(rank, 1), sorted.count) - 1]
    }
}

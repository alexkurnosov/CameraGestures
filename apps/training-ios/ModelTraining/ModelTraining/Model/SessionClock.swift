// Converts between the shot clock and the camera clock of one session.
//
// The player's playhead and the layers are on the shot clock
// (HandShot.timestamp), which follows the wall clock and moved by up to 67 ms
// against the camera within a session (Stage 4). The video is on the camera
// clock. Every shot carries both, so the table of pairs places each shot on
// exactly its own video frame, and interpolates between shots.

#if CG_SESSION_CAPTURE

import Foundation
import CameraGestures

struct SessionClock: Sendable {
    private let timestamps: [TimeInterval]
    private let pts: [TimeInterval]

    /// Reads every shot of every track, frames with no hand included. Nil when
    /// no shot has a capture timestamp.
    init?(reader: SessionReader) {
        var pairs: [(timestamp: TimeInterval, pts: TimeInterval)] = []
        for track in reader.handTracks + [SessionReader.absentTrack] {
            let count = reader.shotCount(track: track)
            var first = 0
            while first < count {
                let times = reader.shotTimes(track: track, from: first, count: 512)
                if times.isEmpty { break }
                for time in times {
                    if let pts = time.pts { pairs.append((time.timestamp, pts)) }
                }
                first += times.count
            }
        }
        pairs.sort { $0.pts < $1.pts }

        // Both columns must rise for lookups in either direction. Hands of one
        // frame share a capture timestamp, and a wall clock that stepped back
        // would break the order of the shot column.
        var timestamps: [TimeInterval] = [], pts: [TimeInterval] = []
        for pair in pairs {
            if let lastPTS = pts.last, let lastTimestamp = timestamps.last,
               pair.pts <= lastPTS || pair.timestamp < lastTimestamp { continue }
            timestamps.append(pair.timestamp)
            pts.append(pair.pts)
        }
        guard !pts.isEmpty else { return nil }
        self.timestamps = timestamps
        self.pts = pts
    }

    func pts(atTimestamp t: TimeInterval) -> TimeInterval {
        Self.interpolate(t, from: timestamps, to: pts)
    }

    func timestamp(atPTS p: TimeInterval) -> TimeInterval {
        Self.interpolate(p, from: pts, to: timestamps)
    }

    /// Linear between the two pairs around `x`; beyond either end, the end
    /// pair's offset.
    private static func interpolate(_ x: Double, from xs: [Double], to ys: [Double]) -> Double {
        // Index of the first pair with xs > x.
        var low = 0, high = xs.count
        while low < high {
            let mid = (low + high) / 2
            if xs[mid] <= x { low = mid + 1 } else { high = mid }
        }
        if low == 0 { return ys[0] + x - xs[0] }
        let i = low - 1
        if i == xs.count - 1 { return ys[i] + x - xs[i] }
        let fraction = (x - xs[i]) / (xs[i + 1] - xs[i])
        return ys[i] + fraction * (ys[i + 1] - ys[i])
    }
}

#endif

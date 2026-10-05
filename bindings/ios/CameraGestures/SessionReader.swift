// Session reader: open a .cgsession bundle and read its shots on demand.
//
// The whole file is compiled only with CG_SESSION_CAPTURE: the C reader ships
// in the capture variant of the library together with the recorder.
//
// A reader is not thread-safe. Use each one from a single thread or actor; open
// a second reader on the same bundle for background work.

#if CG_SESSION_CAPTURE

import Foundation
import CameraGesturesC

public enum SessionReaderError: Error {
    case openFailed(String)
}

public final class SessionReader {

    private let ref: cg_session_reader_ref

    /// Fails on a missing or unparsable manifest and on a format version newer
    /// than this library reads. An interrupted session opens normally.
    public init(url: URL) throws {
        var message = [CChar](repeating: 0, count: 256)
        guard let ref = cg_session_reader_open(url.path, &message, message.count) else {
            throw SessionReaderError.openFailed(String(cString: message))
        }
        self.ref = ref
    }

    deinit {
        cg_session_reader_close(ref)
    }

    /// The MediaPipe hand indices that have shots, ascending. Frames with no
    /// hand are on a track of their own, which is not listed here.
    public var handTracks: [Int] {
        (0..<cg_session_reader_track_count(ref))
            .map { Int(cg_session_reader_track_at(ref, $0)) }
            .filter { $0 != Int(CG_SESSION_ABSENT_TRACK) }
    }

    /// Earliest and latest shot timestamp over all tracks; nil with no shots.
    public var timeRange: ClosedRange<TimeInterval>? {
        var start = 0.0, end = 0.0
        guard cg_session_reader_time_range(ref, &start, &end) != 0 else { return nil }
        return start...end
    }

    public func shotCount(track: Int) -> Int {
        cg_session_reader_shot_count(ref, Int32(track))
    }

    /// Up to `count` shots of one track, starting at index `first`. Only the
    /// chunks that hold them are read from disk.
    public func shots(track: Int, from first: Int, count: Int) -> [HandShot] {
        var buffer = [cg_session_shot](repeating: cg_session_shot(), count: count)
        let read = cg_session_reader_read_shots(ref, Int32(track), first, count, &buffer)
        return buffer.prefix(read).map { HandShot(fromCStruct: $0.shot) }
    }

    /// Index of the first shot with a timestamp at or after `time`;
    /// `shotCount(track:)` when there is none.
    public func firstShotIndex(track: Int, atOrAfter time: TimeInterval) -> Int {
        cg_session_reader_find_shot(ref, Int32(track), time)
    }
}

private extension HandShot {
    init(fromCStruct c: cg_handshot) {
        var landmarks: [Point3D] = []
        landmarks.reserveCapacity(21)
        withUnsafeBytes(of: c.landmarks) { buffer in
            for point in buffer.bindMemory(to: cg_point3d.self) {
                landmarks.append(Point3D(x: point.x, y: point.y, z: point.z))
            }
        }
        self.init(landmarks: landmarks,
                  timestamp: c.timestamp,
                  leftOrRight: c.handedness == CG_HAND_LEFT ? .left
                             : c.handedness == CG_HAND_RIGHT ? .right : .unknown,
                  isAbsent: c.is_absent != 0)
    }
}

#endif

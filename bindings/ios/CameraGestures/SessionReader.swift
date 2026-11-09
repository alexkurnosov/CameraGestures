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

/// The recognizer settings a session was recorded with, from its manifest.
public struct SessionRecognizerConfig: Sendable {
    public let gateEnabled: Bool
    public let tOpen: Float
    public let tClose: Float
    public let holdsEnabled: Bool
    public let bypassPhase2: Bool
    public let tHold: Float
    public let tauPhase3Confidence: Float
    public let confidenceThreshold: Float
}

/// One telemetry row: what the library reported for one shot. A value the
/// library did not compute for this shot is nil, so a gap stays a gap.
public struct SessionTelemetryRow {
    public let timestamp: TimeInterval
    public let rawEnergy: Float?
    public let smoothedEnergy: Float?
    public let gateOpen: Bool
    public let bufferCount: Int
    /// Pending deadlines after this shot, on the shot clock.
    public let commitDeadline: TimeInterval?
    public let minBufferDeadline: TimeInterval?
}

/// The video beside the shots, from the manifest.
public struct SessionVideoInfo: Sendable {
    public let url: URL
    /// Stored frame size in pixels, before rotation. Landmarks are normalised
    /// to this frame.
    public let width: Int
    public let height: Int
    /// Clockwise rotation that shows a stored frame upright: 0, 90, 180 or 270.
    public let rotationDegrees: Int
    /// The stored frames are mirrored relative to the scene.
    public let mirrored: Bool
    /// Capture timestamp of video time 0. A shot's frame is at video time
    /// `pts - anchorPTS`.
    public let anchorPTS: TimeInterval
}

/// One line of events.jsonl. `fields` is the whole JSON object.
public struct SessionEvent {
    public let timestamp: TimeInterval
    public let name: String
    public let reason: String
    public let fields: [String: Any]
}

public final class SessionReader {

    private let ref: cg_session_reader_ref
    private let url: URL

    /// The track of frames with no hand; not listed in `handTracks`.
    public static let absentTrack = Int(CG_SESSION_ABSENT_TRACK)

    /// Fails on a missing or unparsable manifest and on a format version newer
    /// than this library reads. An interrupted session opens normally.
    public init(url: URL) throws {
        var message = [CChar](repeating: 0, count: 256)
        guard let ref = cg_session_reader_open(url.path, &message, message.count) else {
            throw SessionReaderError.openFailed(String(cString: message))
        }
        self.ref = ref
        self.url = url
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

    /// Timestamp and capture timestamp of up to `count` shots of one track,
    /// starting at index `first`. The capture timestamp is nil for a shot
    /// recorded without one.
    public func shotTimes(track: Int, from first: Int, count: Int)
        -> [(timestamp: TimeInterval, pts: TimeInterval?)] {
        var buffer = [cg_session_shot](repeating: cg_session_shot(), count: count)
        let read = cg_session_reader_read_shots(ref, Int32(track), first, count, &buffer)
        return buffer.prefix(read).map { ($0.shot.timestamp, $0.has_pts != 0 ? $0.pts : nil) }
    }

    /// Nil when the session has no video, or when no frame reached the video
    /// file, so there is nothing to place shots on.
    public var video: SessionVideoInfo? {
        var provenance = cg_session_provenance()
        cg_session_reader_get_provenance(ref, &provenance)
        let v = provenance.video
        guard v.has_video != 0, v.has_anchor != 0 else { return nil }
        return SessionVideoInfo(url: url.appendingPathComponent("video.mov"),
                                width: Int(v.width), height: Int(v.height),
                                rotationDegrees: Int(v.rotation_degrees),
                                mirrored: v.mirrored != 0,
                                anchorPTS: v.anchor_pts)
    }

    /// Nil when the session was recorded without a recognizer.
    public var config: SessionRecognizerConfig? {
        var c = cg_recognizer_config()
        var bypass: Int32 = 0
        guard cg_session_reader_get_config(ref, &c, &bypass) != 0 else { return nil }
        return SessionRecognizerConfig(gateEnabled: c.gate_enabled != 0,
                                       tOpen: c.motion_gate.t_open,
                                       tClose: c.motion_gate.t_close,
                                       holdsEnabled: c.holds_enabled != 0,
                                       bypassPhase2: bypass != 0,
                                       tHold: c.holds.t_hold,
                                       tauPhase3Confidence: c.holds.tau_phase3_confidence,
                                       confidenceThreshold: c.confidence_threshold)
    }

    /// Whether a pose model file was loaded when the session was recorded.
    public var hasPoseModel: Bool {
        var files = cg_session_file_records()
        cg_session_reader_get_model_files(ref, &files)
        return files.pose_model.name != nil
    }

    public var telemetryCount: Int {
        cg_session_reader_telemetry_count(ref)
    }

    /// Up to `count` telemetry rows, starting at index `first`.
    public func telemetry(from first: Int, count: Int) -> [SessionTelemetryRow] {
        var buffer = [cg_frame_telemetry](repeating: cg_frame_telemetry(), count: count)
        let read = cg_session_reader_read_telemetry(ref, first, count, &buffer)
        return buffer.prefix(read).map { r in
            SessionTelemetryRow(timestamp: r.timestamp,
                                rawEnergy: r.raw_energy_valid != 0 ? r.raw_energy : nil,
                                smoothedEnergy: r.smoothed_energy_valid != 0 ? r.smoothed_energy : nil,
                                gateOpen: r.gate_open != 0,
                                bufferCount: Int(r.buffer_count),
                                commitDeadline: r.commit_deadline != 0 ? r.commit_deadline : nil,
                                minBufferDeadline: r.min_buffer_deadline != 0 ? r.min_buffer_deadline : nil)
        }
    }

    /// Every event of the session, in recorded order.
    public func events() -> [SessionEvent] {
        (0..<cg_session_reader_event_count(ref)).compactMap { i in
            guard let line = cg_session_reader_event_json(ref, i),
                  let object = try? JSONSerialization.jsonObject(with: Data(String(cString: line).utf8)),
                  let fields = object as? [String: Any],
                  let time = fields["t"] as? Double,
                  let name = fields["event"] as? String else { return nil }
            return SessionEvent(timestamp: time, name: name,
                                reason: fields["reason"] as? String ?? "", fields: fields)
        }
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

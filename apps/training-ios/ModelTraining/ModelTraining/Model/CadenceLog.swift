// Cadence log: one "[CG:Cadence] {json}" line per event of a prediction run,
// printed to the Xcode console and appended to Documents/cadence.log.
//
// The file lets a long measurement (session capture Stage 5) be pulled off the
// phone with `xcrun devicectl device copy from` instead of copying numbers by
// hand. A line every 10 s costs nothing measurable: the view model already
// takes the same snapshot once a second for the debug panel.

import Foundation
import CameraGestures

@MainActor
final class CadenceLog {

    static let shared = CadenceLog()

    /// Past this size at launch the log moves to cadence.log.old. A line is
    /// about 600 bytes, so one file holds about 20 hours of prediction.
    private static let rotateBytes = 5_000_000

    let url: URL

    private init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        url = docs.appendingPathComponent("cadence.log")
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        if size > Self.rotateBytes {
            let old = docs.appendingPathComponent("cadence.log.old")
            try? FileManager.default.removeItem(at: old)
            try? FileManager.default.moveItem(at: url, to: old)
        }
    }

    /// Writes one line. `marker` is the bracketed tag, e.g. "CG:Cadence".
    func write(_ marker: String, _ fields: [String: Any]) {
        // JSONSerialization throws on NaN and infinity. The camera reports an
        // invalid exposure while it restarts between runs.
        var fields = fields.mapValues { value -> Any in
            if let number = value as? Double, !number.isFinite { return NSNull() }
            return value
        }
        fields["wall"] = Date().timeIntervalSince1970
        guard let data = try? JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8) else { return }
        let line = "[\(marker)] \(json)\n"
        print(line, terminator: "")

        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        try? handle.write(contentsOf: Data(line.utf8))
    }

    // MARK: - Field helpers

    static func fields(_ fr: FrameRateStats) -> [String: Any] {
        [
            "elapsed_s":        fr.elapsed,
            "frame_rate_hz":    fr.frameRate,
            "handshot_rate_hz": fr.handshotRate,
            "camera_rate_hz":   fr.cameraRate,
            "interval_ms_p50":  fr.p50IntervalMs,
            "interval_ms_p90":  fr.p90IntervalMs,
            "interval_ms_p99":  fr.p99IntervalMs,
            "interval_ms_max":  fr.maxIntervalMs,
            "interval_samples": fr.intervalCount,
            "frames":           fr.frames,
            "handshots":        fr.handshots,
            "absent_shots":     fr.absentShots,
            "camera_frames":    fr.cameraFrames,
            "camera_drops":     fr.cameraDrops,
            "mediapipe_drops":  fr.landmarkerDrops,
            "drop_fraction":    fr.dropFraction,
        ]
    }

    static func fields(_ exposure: CameraExposure?) -> [String: Any] {
        guard let exposure else { return [:] }
        return [
            "exposure_ms":      exposure.exposureDuration * 1000,
            "iso":              Double(exposure.iso),
            "min_frame_ms":     exposure.minFrameDuration * 1000,
            "max_frame_ms":     exposure.maxFrameDuration * 1000,
        ]
    }

    static func name(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal:    return "nominal"
        case .fair:       return "fair"
        case .serious:    return "serious"
        case .critical:   return "critical"
        @unknown default: return "unknown"
        }
    }
}

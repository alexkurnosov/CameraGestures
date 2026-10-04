// Session capture: record one prediction run as a .cgsession bundle.
//
// The whole file is compiled only with CG_SESSION_CAPTURE, which the
// `CameraGestures/SessionCapture` subspec sets. That subspec also ships the
// capture variant of the C library; the standard variant has no recorder.
//
// Order of use:
//   1. `SessionCaptureConsent.request(includesVideo:)` asks the user.
//   2. `SessionRecorder.start(consent:modelFiles:)`, before the recognizer starts.
//   3. The session ends when the recognizer stops, or at `stop()`.

#if CG_SESSION_CAPTURE

import Foundation
import UIKit
import CameraGesturesC

public enum SessionCaptureError: Error {
    /// A consent covers one session. Ask again for the next one.
    case consentAlreadyUsed
    case recognizerNotInitialized
    case alreadyRecording
    case startFailed(String)
}

/// The files that shape the recognizer's decisions. The library records each
/// one's name, size and SHA-256 in the session manifest. `nil` = not loaded.
public struct SessionModelFiles {
    public var gestureModel: URL?
    public var gestureIds:   URL?
    public var poseModel:    URL?
    public var poseManifest: URL?
    public var preprocessor: URL?

    public init(gestureModel: URL? = nil, gestureIds: URL? = nil,
                poseModel: URL? = nil, poseManifest: URL? = nil,
                preprocessor: URL? = nil) {
        self.gestureModel = gestureModel
        self.gestureIds   = gestureIds
        self.poseModel    = poseModel
        self.poseManifest = poseManifest
        self.preprocessor = preprocessor
    }
}

// --------------------------------------------------------------------------
// MARK: - Consent
// --------------------------------------------------------------------------

/// The user's agreement to record one session. Only `request` creates one, and
/// only after the user accepted the library's own prompt, so an app cannot
/// start a capture the user was not asked about.
public final class SessionCaptureConsent {
    public let includesVideo: Bool
    public let grantedAt: Date
    private var used = false

    private init(includesVideo: Bool) {
        self.includesVideo = includesVideo
        self.grantedAt     = Date()
    }

    /// What the prompt says.
    public static func text(includesVideo: Bool) -> String {
        let content = includesVideo
            ? "the positions of your hands in each camera frame, the gesture decisions made from them, and the camera video"
            : "the positions of your hands in each camera frame and the gesture decisions made from them. No video is recorded"
        return "This session will be recorded. The recording contains \(content). "
             + "It is saved on this device, and it can be exported from the app as a file and shared."
    }

    /// Shows the prompt over the app's frontmost screen. Returns nil when the
    /// user declines, or when there is no screen to show the prompt on.
    @MainActor
    public static func request(includesVideo: Bool) async -> SessionCaptureConsent? {
        let scene = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first { $0.activationState == .foregroundActive }
        guard var presenter = scene?.windows.first(where: \.isKeyWindow)?.rootViewController else {
            return nil
        }
        while let presented = presenter.presentedViewController { presenter = presented }

        let granted: Bool = await withCheckedContinuation { continuation in
            let alert = UIAlertController(title: "Record this session?",
                                          message: text(includesVideo: includesVideo),
                                          preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "Don't Record", style: .cancel) { _ in
                continuation.resume(returning: false)
            })
            alert.addAction(UIAlertAction(title: "Record", style: .default) { _ in
                continuation.resume(returning: true)
            })
            presenter.present(alert, animated: true)
        }
        return granted ? SessionCaptureConsent(includesVideo: includesVideo) : nil
    }

    fileprivate func consume() -> Bool {
        if used { return false }
        used = true
        return true
    }
}

// --------------------------------------------------------------------------
// MARK: - Recorder
// --------------------------------------------------------------------------

/// A session found in `SessionRecorder.sessionsDirectory()`.
public struct StoredSession {
    public let url: URL
    /// False when the app was killed while recording. The session is still
    /// readable up to the last chunk that reached the disk.
    public let isComplete: Bool
    public let isTruncated: Bool
    public let shotCount: Int
    public let telemetryCount: Int
    public let eventCount: Int
    /// Seconds between the first and the last recorded shot.
    public let duration: TimeInterval
}

public final class SessionRecorder {

    private let recognizer: HandGestureRecognizing

    public init(recognizer: HandGestureRecognizing) {
        self.recognizer = recognizer
    }

    public var isRecording: Bool { recognizer.isCapturingSession }

    /// Starts recording and returns the bundle's location. Call it before
    /// `HandGestureRecognizing.start()`; the session ends when the recognizer
    /// stops.
    @discardableResult
    public func start(consent: SessionCaptureConsent,
                      modelFiles: SessionModelFiles = SessionModelFiles()) throws -> URL {
        guard consent.consume() else { throw SessionCaptureError.consentAlreadyUsed }

        let name = ISO8601DateFormatter.string(from: Date(), timeZone: TimeZone(identifier: "UTC")!,
                                               formatOptions: [.withYear, .withMonth, .withDay,
                                                               .withTime, .withTimeZone])
        let bundle = try Self.sessionsDirectory().appendingPathComponent(name + ".cgsession")
        try recognizer.startSessionCapture(
            bundlePath: bundle.path,
            modelFiles: modelFiles,
            extra: [
                "consent_granted_at":     ISO8601DateFormatter().string(from: consent.grantedAt),
                "consent_includes_video": consent.includesVideo ? "true" : "false",
            ])
        return bundle
    }

    /// Ends the session before the recognizer stops.
    public func stop() {
        recognizer.stopSessionCapture()
    }

    /// Where sessions are kept: inside the app's container, not visible in the
    /// Files app, and left out of iCloud and device backups.
    public static func sessionsDirectory() throws -> URL {
        var dir = try FileManager.default
            .url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("CameraGestures/Sessions", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try dir.setResourceValues(values)
        return dir
    }

    /// Every readable session on the device, oldest first, including ones cut
    /// short by a kill.
    public static func storedSessions() -> [StoredSession] {
        guard let dir = try? sessionsDirectory(),
              let urls = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        else { return [] }

        return urls
            .filter { $0.pathExtension == "cgsession" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .compactMap { url in
                guard let reader = cg_session_reader_open(url.path, nil, 0) else { return nil }
                defer { cg_session_reader_close(reader) }
                var shots = 0
                for i in 0..<cg_session_reader_track_count(reader) {
                    shots += cg_session_reader_shot_count(reader, cg_session_reader_track_at(reader, i))
                }
                var start = 0.0, end = 0.0
                cg_session_reader_time_range(reader, &start, &end)
                return StoredSession(
                    url:            url,
                    isComplete:     cg_session_reader_is_complete(reader) != 0,
                    isTruncated:    cg_session_reader_is_truncated(reader) != 0,
                    shotCount:      shots,
                    telemetryCount: cg_session_reader_telemetry_count(reader),
                    eventCount:     cg_session_reader_event_count(reader),
                    duration:       end - start)
            }
    }
}

#endif

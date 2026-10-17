// Session capture: the video beside the shots (video.mov in the bundle).
//
// The camera frames go to MediaPipe as before. This writer gets a copy of each
// one and encodes it on its own queue. A frame is left out of the video when
// the encoder is behind; the camera queue never waits for it.
//
// Compiled only with CG_SESSION_CAPTURE.

#if CG_SESSION_CAPTURE

import Foundation
import AVFoundation

/// How the stored frames relate to the scene, for the session manifest.
struct SessionVideoFormat {
    let width: Int32
    let height: Int32
    /// Clockwise rotation that shows a stored frame upright.
    let rotationDegrees: Int32
    let mirrored: Bool
}

/// Something the writer has to tell the session: a line for events.jsonl.
/// `video_started` also carries the clock anchor (`pts`).
struct SessionVideoNote {
    let event: String
    /// On the shot clock (`HandShot.timestamp`).
    let timestamp: Double
    let fields: [String: Any]
}

final class SessionVideoWriter {

    /// Frames copied and not yet released by the encoder. A frame that would
    /// exceed this is skipped, which bounds both the memory and the backlog.
    private static let maxFramesInFlight = 4

    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let pool: CVPixelBufferPool
    private let queue = DispatchQueue(label: "com.cameragestures.session-video", qos: .utility)

    // Writer queue only.
    private var sessionStarted = false
    private var framesWritten = 0
    private var framesSkipped = 0
    private var skipRun: (firstTimestamp: Double, lastTimestamp: Double, firstPTS: Double, count: Int)?

    private let notesLock = NSLock()
    private var notes: [SessionVideoNote] = []

    init(url: URL, format: SessionVideoFormat) throws {
        writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        // Fragments keep the file playable up to the last one if the app is killed.
        writer.movieFragmentInterval = CMTime(seconds: 2, preferredTimescale: 600)

        input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey:  AVVideoCodecType.h264,
            AVVideoWidthKey:  Int(format.width),
            AVVideoHeightKey: Int(format.height),
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey:             2_000_000,
                AVVideoMaxKeyFrameIntervalDurationKey: 1.0,
                // Reordering makes the encoder hold frames back.
                AVVideoAllowFrameReorderingKey:       false,
            ] as [String: Any],
        ])
        input.expectsMediaDataInRealTime = true
        // The frames are stored as the camera delivers them; players rotate.
        input.transform = CGAffineTransform(rotationAngle: CGFloat(format.rotationDegrees) * .pi / 180)

        let bufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String:     kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String:               Int(format.width),
            kCVPixelBufferHeightKey as String:              Int(format.height),
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ]
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
                                                       sourcePixelBufferAttributes: bufferAttributes)
        var made: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, bufferAttributes as CFDictionary, &made)
        guard let made, writer.canAdd(input) else {
            throw SessionCaptureError.startFailed("cannot set up the video writer")
        }
        pool = made
        writer.add(input)
        guard writer.startWriting() else {
            throw SessionCaptureError.startFailed(
                "cannot write \(url.lastPathComponent): \(writer.error?.localizedDescription ?? "unknown error")")
        }
    }

    /// Call on the camera queue, for every frame given to MediaPipe.
    /// `timestamp` is the frame's time on the shot clock.
    func append(_ frame: CVPixelBuffer, pts: CMTime, timestamp: Double) {
        let copy = copyOf(frame)
        queue.async { [self] in
            if let copy {
                write(copy, pts: pts, timestamp: timestamp)
            } else {
                noteSkipped(pts: pts, timestamp: timestamp)
            }
        }
    }

    /// The notes collected since the last call.
    func takeNotes() -> [SessionVideoNote] {
        notesLock.lock()
        defer { notesLock.unlock() }
        let taken = notes
        notes = []
        return taken
    }

    /// Closes the file and returns the remaining notes. Blocks until the file
    /// is complete. Call once, after the last `append`.
    func finish() -> [SessionVideoNote] {
        queue.sync {
            closeSkipRun()
            if sessionStarted {
                input.markAsFinished()
                let done = DispatchSemaphore(value: 0)
                writer.finishWriting { done.signal() }
                done.wait()
            } else {
                writer.cancelWriting()
            }
            var fields: [String: Any] = ["frames_written": framesWritten, "frames_skipped": framesSkipped]
            if let error = writer.error { fields["error"] = error.localizedDescription }
            add(SessionVideoNote(event: "video_finished", timestamp: Date().timeIntervalSince1970, fields: fields))
        }
        return takeNotes()
    }

    // MARK: - Camera queue

    /// nil when `maxFramesInFlight` copies are still in use.
    private func copyOf(_ frame: CVPixelBuffer) -> CVPixelBuffer? {
        let limit = [kCVPixelBufferPoolAllocationThresholdKey: Self.maxFramesInFlight] as CFDictionary
        var made: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBufferWithAuxAttributes(nil, pool, limit, &made)
        guard let copy = made else { return nil }

        CVPixelBufferLockBaseAddress(frame, .readOnly)
        CVPixelBufferLockBaseAddress(copy, [])
        defer {
            CVPixelBufferUnlockBaseAddress(copy, [])
            CVPixelBufferUnlockBaseAddress(frame, .readOnly)
        }
        guard let from = CVPixelBufferGetBaseAddress(frame),
              let to = CVPixelBufferGetBaseAddress(copy) else { return nil }
        let fromStride = CVPixelBufferGetBytesPerRow(frame)
        let toStride   = CVPixelBufferGetBytesPerRow(copy)
        let rows       = min(CVPixelBufferGetHeight(frame), CVPixelBufferGetHeight(copy))
        if fromStride == toStride {
            memcpy(to, from, rows * fromStride)
        } else {
            for row in 0..<rows {
                memcpy(to + row * toStride, from + row * fromStride, min(fromStride, toStride))
            }
        }
        return copy
    }

    // MARK: - Writer queue

    private func write(_ frame: CVPixelBuffer, pts: CMTime, timestamp: Double) {
        if !sessionStarted {
            // Video time 0 is this frame, whether or not the encoder takes it.
            writer.startSession(atSourceTime: pts)
            sessionStarted = true
            add(SessionVideoNote(event: "video_started", timestamp: timestamp, fields: ["pts": pts.seconds]))
        }
        guard input.isReadyForMoreMediaData, adaptor.append(frame, withPresentationTime: pts) else {
            noteSkipped(pts: pts, timestamp: timestamp)
            return
        }
        closeSkipRun()
        framesWritten += 1
    }

    private func noteSkipped(pts: CMTime, timestamp: Double) {
        framesSkipped += 1
        if skipRun == nil {
            skipRun = (timestamp, timestamp, pts.seconds, 0)
        }
        skipRun?.lastTimestamp = timestamp
        skipRun?.count += 1
    }

    /// One event per run of consecutive skipped frames.
    private func closeSkipRun() {
        guard let run = skipRun else { return }
        skipRun = nil
        add(SessionVideoNote(event: "video_frames_skipped", timestamp: run.firstTimestamp, fields: [
            "count":     run.count,
            "last_t":    run.lastTimestamp,
            "first_pts": run.firstPTS,
        ]))
    }

    private func add(_ note: SessionVideoNote) {
        notesLock.lock()
        notes.append(note)
        notesLock.unlock()
    }
}

#endif

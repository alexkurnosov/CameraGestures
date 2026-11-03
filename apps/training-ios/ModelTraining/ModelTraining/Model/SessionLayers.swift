// What the library decided during one session, prepared for the metadata
// layers: telemetry rows and events turned into marks on the player's time axis.
//
// A whole session is read once. At 56 bytes a row, a 10-minute two-hand session
// costs about 2 MB, so a moving window like the shots' would save little.
//
// Nothing here is recomputed from the shots. The layers show the library's own
// values, and a value it did not report stays missing.

#if CG_SESSION_CAPTURE

import Foundation
import CameraGestures

struct SessionLayers: Sendable {

    struct Row {
        let time: TimeInterval
        let rawEnergy: Float?
        let smoothedEnergy: Float?
        let gateOpen: Bool
        let bufferCount: Int
    }

    struct Span {
        let start: TimeInterval
        let end: TimeInterval
    }

    struct Hold {
        let start: TimeInterval
        let end: TimeInterval
        let poseId: Int
        let confidence: Float
        let accepted: Bool
        let action: String
    }

    struct Prediction {
        let time: TimeInterval
        let accepted: Bool
        let restricted: Bool
        /// The model's best candidate. Nil for a rejected prediction in a session
        /// recorded before the library kept it.
        let gestureId: String?
        let confidence: Float
        let candidates: [String]
    }

    struct Note {
        let time: TimeInterval
        let text: String
    }

    struct Cooldown {
        let start: TimeInterval
        let end: TimeInterval
        let gestureId: String
    }

    struct Suppressed {
        /// When the gesture was detected, which is what the mark shows. The event
        /// itself is written later, when another gesture replaces this one.
        let detectedAt: TimeInterval
        let gestureId: String
        let confidence: Float
        let replacedBy: String
    }

    let rows: [Row]
    let config: SessionRecognizerConfig?

    let gateSpans: [Span]
    let cycleEnds: [Note]
    /// Why the gate layer has nothing to show; nil when it has.
    let gateUnavailable: String?

    let holds: [Hold]
    /// Why the hold layer has nothing to show; nil when it has.
    let holdUnavailable: String?

    let predictions: [Prediction]
    let cycleSkips: [Note]

    let commitWaits: [Span]
    let minBufferWaits: [Span]
    let commits: [Note]
    let cooldowns: [Cooldown]
    let suppressed: [Suppressed]

    /// `start` is the player's time zero, on the shot clock. Nil when the reader
    /// cannot open the bundle.
    init?(url: URL, start: TimeInterval) {
        guard let reader = try? SessionReader(url: url) else { return nil }

        var sourceRows: [SessionTelemetryRow] = []
        sourceRows.reserveCapacity(reader.telemetryCount)
        while sourceRows.count < reader.telemetryCount {
            let batch = reader.telemetry(from: sourceRows.count, count: 512)
            if batch.isEmpty { break }
            sourceRows += batch
        }
        rows = sourceRows.map {
            Row(time: $0.timestamp - start, rawEnergy: $0.rawEnergy, smoothedEnergy: $0.smoothedEnergy,
                gateOpen: $0.gateOpen, bufferCount: $0.bufferCount)
        }
        let config = reader.config
        self.config = config

        gateSpans = Self.gateSpans(rows)
        commitWaits = Self.waits(sourceRows, start: start) { $0.commitDeadline }
        minBufferWaits = Self.waits(sourceRows, start: start) { $0.minBufferDeadline }

        if rows.isEmpty {
            gateUnavailable = "no telemetry in this session"
        } else if config?.gateEnabled == false {
            gateUnavailable = "the gate was off"
        } else {
            gateUnavailable = nil
        }

        // Decided from the values, so a model that failed to load is covered too.
        // The manifest only supplies the reason.
        if rows.contains(where: { $0.smoothedEnergy != nil }) {
            holdUnavailable = nil
        } else if !reader.hasPoseModel {
            holdUnavailable = "no pose model in this session"
        } else if config?.holdsEnabled == false {
            holdUnavailable = "holds mode was off"
        } else if config?.bypassPhase2 == true {
            holdUnavailable = "Phase 2 was bypassed"
        } else {
            holdUnavailable = "no smoothed energy recorded"
        }

        let events = reader.events()
        func relative(_ event: SessionEvent) -> TimeInterval { event.timestamp - start }
        func number(_ event: SessionEvent, _ key: String) -> Double { event.fields[key] as? Double ?? 0 }
        func text(_ event: SessionEvent, _ key: String) -> String { event.fields[key] as? String ?? "" }
        func flag(_ event: SessionEvent, _ key: String) -> Bool { event.fields[key] as? Bool ?? false }
        func named(_ name: String) -> [SessionEvent] {
            events.filter { $0.name == name }.sorted { $0.timestamp < $1.timestamp }
        }

        cycleEnds = named("cycle_ended").map { Note(time: relative($0), text: $0.reason) }
        cycleSkips = named("cycle_skipped").map { Note(time: relative($0), text: $0.reason) }
        commits = named("commit_fired").map { Note(time: relative($0), text: $0.reason) }

        holds = named("hold_completed").map {
            Hold(start: number($0, "hold_start_time") - start, end: relative($0),
                 poseId: $0.fields["pose_id"] as? Int ?? -1,
                 confidence: Float(number($0, "confidence")),
                 accepted: flag($0, "accepted"), action: $0.reason)
        }
        predictions = named("phase3_prediction").map {
            let top = text($0, "top_gesture_id")
            let emitted = text($0, "gesture_id")
            return Prediction(time: relative($0), accepted: flag($0, "accepted"),
                              restricted: flag($0, "restricted"),
                              gestureId: !top.isEmpty ? top : !emitted.isEmpty ? emitted : nil,
                              confidence: Float(number($0, "confidence")),
                              candidates: $0.fields["candidates"] as? [String] ?? [])
        }
        suppressed = named("gesture_suppressed").map {
            Suppressed(detectedAt: number($0, "detected_at") - start,
                       gestureId: text($0, "gesture_id"),
                       confidence: Float(number($0, "confidence")),
                       replacedBy: text($0, "replaced_by"))
        }.sorted { $0.detectedAt < $1.detectedAt }

        let marks = events.filter { $0.name == "cooldown_started" || $0.name == "cooldown_expired" }
                          .sorted { $0.timestamp < $1.timestamp }
        cooldowns = marks.indices.compactMap { i in
            guard marks[i].name == "cooldown_started" else { return nil }
            // A cooldown the recognizer gave up on has no expiry event, so its
            // own duration caps the window.
            let cap = marks[i].timestamp + number(marks[i], "duration")
            let end = i + 1 < marks.count ? min(marks[i + 1].timestamp, cap) : cap
            return Cooldown(start: relative(marks[i]), end: end - start,
                            gestureId: text(marks[i], "gesture_id"))
        }
    }

    // MARK: - Lookup

    /// Index of the last item at or before `time` in an array sorted by time.
    static func lastIndex<T>(in items: [T], atOrBefore time: TimeInterval,
                             by key: (T) -> TimeInterval) -> Int? {
        var low = 0, high = items.count
        while low < high {
            let middle = (low + high) / 2
            if key(items[middle]) <= time { low = middle + 1 } else { high = middle }
        }
        return low > 0 ? low - 1 : nil
    }

    // MARK: - Derived spans

    private static func gateSpans(_ rows: [Row]) -> [Span] {
        var spans: [Span] = []
        var openedAt: TimeInterval?
        for row in rows {
            if row.gateOpen, openedAt == nil {
                openedAt = row.time
            } else if !row.gateOpen, let start = openedAt {
                spans.append(Span(start: start, end: row.time))
                openedAt = nil
            }
        }
        if let start = openedAt, let last = rows.last {
            spans.append(Span(start: start, end: last.time))
        }
        return spans
    }

    /// The times a deadline was pending. A wait ends at its deadline, or at the
    /// row where the deadline is gone if the cycle ended sooner. A deadline
    /// already past when it first appears gives a wait of zero length.
    private static func waits(_ rows: [SessionTelemetryRow], start: TimeInterval,
                              deadline: (SessionTelemetryRow) -> TimeInterval?) -> [Span] {
        var spans: [Span] = []
        var open: (start: TimeInterval, deadline: TimeInterval)?
        func close(at time: TimeInterval) {
            guard let wait = open else { return }
            let end = max(wait.start, min(wait.deadline, time))
            spans.append(Span(start: wait.start - start, end: end - start))
            open = nil
        }
        for row in rows {
            let current = deadline(row)
            if current != open?.deadline {
                close(at: row.timestamp)
                if let current { open = (row.timestamp, current) }
            }
        }
        if let wait = open { close(at: wait.deadline) }
        return spans
    }
}

#endif

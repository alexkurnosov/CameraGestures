import Foundation
import HandGestureTypes

/// Pure Phase 1 motion-gate state machine.
///
/// No threading, no callbacks, no I/O — drive it by feeding `HandShot`s in order,
/// inspect the returned `Event` and the public `state` / `bufferCount`. Designed for
/// unit tests: tests construct synthetic `HandShot` sequences with controlled
/// timestamps and assert state transitions.
///
/// `HandGestureRecognizing` owns one `MotionGate` instance and forwards each
/// per-frame handshot to it on its private serial queue.
public final class MotionGate {

    public enum Event: Equatable {
        /// Closed → closed, no transition this frame.
        case stillClosed
        /// Closed → open this frame.
        case opened
        /// Open → open, frame appended to buffer.
        case stillOpen(bufferCount: Int)
        /// Open → closed this frame. The captured buffer is the cycle's payload.
        case cycleEnded(buffer: [HandShot])
    }

    public private(set) var state: MotionGateState = .closed
    public var bufferCount: Int { gateBuffer.count }

    private let config: MotionGateConfig
    private let bufferCap: Int

    private var prevNormalizedCoords: [Float]? = nil
    private var aboveThresholdSince: TimeInterval? = nil
    private var belowThresholdSince: TimeInterval? = nil
    private var gateBuffer: [HandShot] = []

    public init(config: MotionGateConfig, bufferCap: Int = 30) {
        self.config = config
        self.bufferCap = bufferCap
    }

    /// Reset the gate to its initial state. Drops any in-flight buffer.
    public func reset() {
        state = .closed
        prevNormalizedCoords = nil
        aboveThresholdSince = nil
        belowThresholdSince = nil
        gateBuffer.removeAll()
    }

    /// Feed one frame in and observe the state transition this frame produced.
    @discardableResult
    public func process(_ handshot: HandShot) -> Event {
        let now = handshot.timestamp

        // Absent frame → immediate close (with cycle end if was open)
        if handshot.isAbsent {
            let wasOpen = (state == .open)
            let captured = gateBuffer
            gateBuffer.removeAll()
            state = .closed
            prevNormalizedCoords = nil
            aboveThresholdSince = nil
            belowThresholdSince = nil
            return wasOpen ? .cycleEnded(buffer: captured) : .stillClosed
        }

        // Compute energy against previous frame (if any)
        let currCoords = MotionGate.normalize(handshot)
        let energy: Float
        if let prev = prevNormalizedCoords, let curr = currCoords {
            energy = MotionGate.energy(current: curr, previous: prev)
        } else {
            energy = 0
        }
        prevNormalizedCoords = currCoords

        switch state {
        case .closed:
            if energy > config.tOpen {
                if aboveThresholdSince == nil { aboveThresholdSince = now }
                let durationMs = (now - aboveThresholdSince!) * 1000
                if durationMs >= config.kOpenMs {
                    state = .open
                    aboveThresholdSince = nil
                    belowThresholdSince = nil
                    gateBuffer.removeAll()
                    return .opened
                }
            } else {
                aboveThresholdSince = nil
            }
            return .stillClosed

        case .open:
            if gateBuffer.count < bufferCap {
                gateBuffer.append(handshot)
            }

            var shouldClose = false
            if energy < config.tClose {
                if belowThresholdSince == nil { belowThresholdSince = now }
                let durationMs = (now - belowThresholdSince!) * 1000
                if durationMs >= config.kCloseMs {
                    shouldClose = true
                }
            } else {
                belowThresholdSince = nil
            }
            if gateBuffer.count >= bufferCap {
                shouldClose = true
            }

            if shouldClose {
                let captured = gateBuffer
                gateBuffer.removeAll()
                state = .closed
                aboveThresholdSince = nil
                belowThresholdSince = nil
                return .cycleEnded(buffer: captured)
            }
            return .stillOpen(bufferCount: gateBuffer.count)
        }
    }

    // MARK: - Pure helpers (static so tests can call without instantiating a gate)

    /// Wrist-relative, scale-normalised 63-dim coord vector for the shot, or nil if absent/degenerate.
    /// Scale reference is the wrist→middle-finger-MCP distance.
    public static func normalize(_ shot: HandShot) -> [Float]? {
        guard shot.landmarks.count == 21 else { return nil }
        let wrist = shot.landmarks[0]
        let middleMCP = shot.landmarks[9]
        let dx = middleMCP.x - wrist.x
        let dy = middleMCP.y - wrist.y
        let dz = middleMCP.z - wrist.z
        let scaleRef = sqrtf(dx*dx + dy*dy + dz*dz)
        guard scaleRef > 1e-6 else { return nil }
        var coords = [Float](repeating: 0, count: 63)
        for i in 0..<21 {
            let lm = shot.landmarks[i]
            coords[i*3]   = (lm.x - wrist.x) / scaleRef
            coords[i*3+1] = (lm.y - wrist.y) / scaleRef
            coords[i*3+2] = (lm.z - wrist.z) / scaleRef
        }
        return coords
    }

    /// Sum of per-landmark L2 distances between two normalised 63-dim coord vectors.
    public static func energy(current: [Float], previous: [Float]) -> Float {
        var energy: Float = 0
        for i in 0..<21 {
            let dx = current[i*3]   - previous[i*3]
            let dy = current[i*3+1] - previous[i*3+1]
            let dz = current[i*3+2] - previous[i*3+2]
            energy += sqrtf(dx*dx + dy*dy + dz*dz)
        }
        return energy
    }
}

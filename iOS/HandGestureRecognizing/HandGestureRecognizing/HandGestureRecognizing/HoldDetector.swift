import Foundation
import HandGestureTypes

/// Pure Phase 2 hold-detection state machine.
///
/// No threading, no callbacks, no I/O. Feed HandShots in order (only while the
/// Phase 1 gate is open), inspect the returned Event. Reset when the gate opens
/// or an absent frame arrives.
///
/// A hold is a maximal run of smoothed energy < `tHold` that lasts at least
/// `kHoldMs`. The representative frame is the argmin of smoothed energy within
/// the run. `edge_trim` is not applied at runtime (plan §Phase 2 §Key-pose
/// extraction pipeline, plan:387).
public final class HoldDetector {

    public enum Event: Equatable {
        case noHold
        case holdDetected(repShot: HandShot, startTime: TimeInterval, endTime: TimeInterval)
    }

    // MARK: - Config

    public struct Config {
        public let tHold: Float
        public let kHoldMs: TimeInterval
        public let smoothKMs: TimeInterval

        public init(tHold: Float = 2.10,
                    kHoldMs: TimeInterval = 100,
                    smoothKMs: TimeInterval = 100) {
            self.tHold = tHold
            self.kHoldMs = kHoldMs
            self.smoothKMs = smoothKMs
        }

        public static let defaultConfig = Config()
    }

    // MARK: - Private state

    private let config: Config

    private struct Frame {
        let shot: HandShot
        let rawEnergy: Float
    }

    private var history: [Frame] = []
    private var prevCoords: [Float]? = nil

    private var inHold = false
    private var holdStartIdx = 0
    private var holdArgminIdx = 0
    private var holdArgminEnergy: Float = Float.infinity
    // Prevents re-emitting the same hold if it continues past kHoldMs
    private var holdEmitted = false

    public init(config: Config = .defaultConfig) {
        self.config = config
    }

    // MARK: - Public API

    public func reset() {
        history.removeAll()
        prevCoords = nil
        inHold = false
        holdStartIdx = 0
        holdArgminIdx = 0
        holdArgminEnergy = .infinity
        holdEmitted = false
    }

    @discardableResult
    public func process(_ shot: HandShot) -> Event {
        guard !shot.isAbsent else {
            let event = finishCurrentHold()
            reset()
            return event ?? .noHold
        }

        // Compute raw energy vs previous frame
        let currCoords = MotionGate.normalize(shot)
        let rawEnergy: Float
        if let prev = prevCoords, let curr = currCoords {
            rawEnergy = MotionGate.energy(current: curr, previous: prev)
        } else {
            rawEnergy = 0
        }
        prevCoords = currCoords
        history.append(Frame(shot: shot, rawEnergy: rawEnergy))

        // Prune history older than 5s to bound memory
        let cutoffTime = shot.timestamp - 5.0
        let pruneCount = history.prefix(while: { $0.shot.timestamp < cutoffTime }).count
        if pruneCount > 0 {
            history.removeFirst(pruneCount)
            if inHold {
                holdStartIdx = max(0, holdStartIdx - pruneCount)
                holdArgminIdx = max(0, holdArgminIdx - pruneCount)
            }
        }

        let currentIdx = history.count - 1
        let smoothed = smoothedEnergy(at: currentIdx)

        if smoothed < config.tHold {
            if !inHold {
                inHold = true
                holdEmitted = false
                holdStartIdx = currentIdx
                holdArgminIdx = currentIdx
                holdArgminEnergy = smoothed
            } else if smoothed < holdArgminEnergy {
                holdArgminEnergy = smoothed
                holdArgminIdx = currentIdx
            }
            // Emit once as soon as the hold has lasted kHoldMs
            if !holdEmitted {
                let startTime = history[holdStartIdx].shot.timestamp
                let durationMs = (shot.timestamp - startTime) * 1000
                if durationMs >= config.kHoldMs {
                    holdEmitted = true
                    let endTime = shot.timestamp
                    let repShot = history[holdArgminIdx].shot
                    return .holdDetected(repShot: repShot, startTime: startTime, endTime: endTime)
                }
            }
        } else {
            if inHold {
                let event = finishCurrentHold()
                inHold = false
                holdArgminEnergy = .infinity
                holdEmitted = false
                return event ?? .noHold
            }
        }

        return .noHold
    }

    // MARK: - Private helpers

    private func finishCurrentHold() -> Event? {
        guard inHold, !holdEmitted else { return nil }

        let startTime = history[holdStartIdx].shot.timestamp
        let endIdx = history.count - 1
        let endTime = history[endIdx].shot.timestamp
        let durationMs = (endTime - startTime) * 1000

        guard durationMs >= config.kHoldMs else { return nil }

        let repShot = history[holdArgminIdx].shot
        return .holdDetected(repShot: repShot, startTime: startTime, endTime: endTime)
    }

    private func smoothedEnergy(at index: Int) -> Float {
        let targetTime = history[index].shot.timestamp
        let windowStart = targetTime - config.smoothKMs / 1000.0
        var sum: Float = 0
        var count = 0
        for i in stride(from: index, through: 0, by: -1) {
            let frame = history[i]
            if frame.shot.timestamp < windowStart { break }
            sum += frame.rawEnergy
            count += 1
        }
        return count > 0 ? sum / Float(count) : 0
    }

    // MARK: - Testable helpers

    /// Smoothed energy of the most-recently processed frame. Useful in unit tests.
    public var lastSmoothedEnergy: Float {
        history.isEmpty ? 0 : smoothedEnergy(at: history.count - 1)
    }
}

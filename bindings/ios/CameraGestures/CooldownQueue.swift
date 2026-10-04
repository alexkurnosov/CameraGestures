// The cooldown between emitted gestures, as plain state with no timer and no
// clock of its own, so it can be tested alone (bindings/ios/Tests). It is
// generic for the same reason: the test builds this one file, without the
// library's types.
//
// A gesture that arrives outside a cooldown window is emitted and opens a
// window. A gesture that arrives inside a window is held until the window ends.
// Only one gesture is held: a second one inside the same window replaces the
// first, which is never emitted.

import Foundation

struct CooldownQueue<Gesture> {

    enum Submission {
        /// Emit the gesture now. A cooldown window has started.
        case emit
        /// The gesture is held until the window ends. `suppressed` is the
        /// gesture it replaced, which is now discarded.
        case queued(suppressed: Gesture?)
    }

    private(set) var cooldownEnd: TimeInterval?
    private(set) var duration: TimeInterval = 0
    private var pending: Gesture?

    mutating func submit(_ gesture: Gesture, now: TimeInterval, cooldown: TimeInterval) -> Submission {
        if let end = cooldownEnd, now < end {
            let suppressed = pending
            pending = gesture
            return .queued(suppressed: suppressed)
        }
        cooldownEnd = now + cooldown
        duration    = cooldown
        return .emit
    }

    /// The window ended. Returns the held gesture, which the caller emits now;
    /// emitting it starts the next window, of the same duration.
    mutating func expire(now: TimeInterval) -> Gesture? {
        guard let gesture = pending else {
            cooldownEnd = nil
            return nil
        }
        pending     = nil
        cooldownEnd = now + duration
        return gesture
    }
}

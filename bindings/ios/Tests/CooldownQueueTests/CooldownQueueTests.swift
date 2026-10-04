import XCTest
@testable import CooldownQueue

final class CooldownQueueTests: XCTestCase {

    private func isEmit(_ s: CooldownQueue<String>.Submission) -> Bool {
        if case .emit = s { return true }
        return false
    }

    /// Two gestures inside one cooldown window: the second replaces the first,
    /// so the window produces one emission and one suppression.
    func testTwoGesturesInsideOneWindowGiveOneEmissionAndOneSuppression() {
        var queue = CooldownQueue<String>()
        var emitted: [String] = []
        var suppressed: [String] = []

        func submit(_ gesture: String, at now: TimeInterval) {
            switch queue.submit(gesture, now: now, cooldown: 1.0) {
            case .emit:               emitted.append(gesture)
            case .queued(let old?):   suppressed.append(old)
            case .queued(nil):        break
            }
        }

        submit("opens the window", at: 10.0)
        XCTAssertEqual(emitted, ["opens the window"])
        emitted.removeAll()

        submit("first", at: 10.2)
        submit("second", at: 10.6)
        XCTAssertEqual(emitted, [])
        XCTAssertEqual(suppressed, ["first"])

        if let held = queue.expire(now: 11.0) { emitted.append(held) }
        XCTAssertEqual(emitted, ["second"])
        XCTAssertEqual(suppressed, ["first"])
    }

    func testEmittingTheHeldGestureStartsANewWindow() {
        var queue = CooldownQueue<String>()
        _ = queue.submit("a", now: 0.0, cooldown: 1.0)
        _ = queue.submit("b", now: 0.5, cooldown: 1.0)
        XCTAssertEqual(queue.expire(now: 1.0), "b")
        XCTAssertEqual(queue.cooldownEnd, 2.0)

        // Still inside the new window, so this one is held, not emitted.
        XCTAssertFalse(isEmit(queue.submit("c", now: 1.5, cooldown: 1.0)))
    }

    func testAWindowWithNothingHeldEndsTheCooldown() {
        var queue = CooldownQueue<String>()
        _ = queue.submit("a", now: 0.0, cooldown: 1.0)
        XCTAssertNil(queue.expire(now: 1.0))
        XCTAssertNil(queue.cooldownEnd)
        XCTAssertTrue(isEmit(queue.submit("b", now: 1.1, cooldown: 1.0)))
    }
}

import XCTest
import HandGestureTypes
@testable import HandGestureRecognizingFramework

final class MotionGateTests: XCTestCase {

    // MARK: - Hand-generation helpers

    /// Build a 21-landmark stationary "hand" anchored at `wrist` with a fixed bone layout.
    private func makeHand(wrist: (Float, Float, Float) = (0, 0, 0),
                          scale: Float = 0.10) -> [Point3D] {
        var points = [Point3D]()
        points.append(Point3D(x: wrist.0, y: wrist.1, z: wrist.2))
        for i in 1..<21 {
            let f = Float(i)
            let dx = scale * sinf(f)
            let dy = (i == 9) ? scale : scale * cosf(f) * 0.5
            let dz = scale * sinf(f * 0.7) * 0.3
            points.append(Point3D(x: wrist.0 + dx, y: wrist.1 + dy, z: wrist.2 + dz))
        }
        return points
    }

    private func translateHand(_ landmarks: [Point3D], by delta: (Float, Float, Float)) -> [Point3D] {
        landmarks.map { Point3D(x: $0.x + delta.0, y: $0.y + delta.1, z: $0.z + delta.2) }
    }

    private func scaleHand(_ landmarks: [Point3D], by factor: Float) -> [Point3D] {
        let wrist = landmarks[0]
        return landmarks.map {
            Point3D(
                x: wrist.x + ($0.x - wrist.x) * factor,
                y: wrist.y + ($0.y - wrist.y) * factor,
                z: wrist.z + ($0.z - wrist.z) * factor
            )
        }
    }

    private func shot(_ landmarks: [Point3D], at t: TimeInterval) -> HandShot {
        HandShot(landmarks: landmarks, timestamp: t, leftOrRight: .right, isAbsent: false)
    }

    /// Returns a HandShot at "step" `n`. Each step shifts landmarks 1..8 and
    /// 10..20 by `(delta, delta, 0)` while landmarks 0 (wrist) and 9 (middle MCP)
    /// stay anchored — keeping the scale reference stable. Consecutive steps
    /// thus produce a deterministic, continuous motion energy.
    /// With default delta=0.005 and scale=0.1, per-frame energy ≈ 1.34
    /// (well above typical T_open=0.5; well above T_close).
    private func movingFrame(step n: Int,
                             at t: TimeInterval,
                             delta: Float = 0.005,
                             scale: Float = 0.1) -> HandShot {
        var points = [Point3D]()
        for i in 0..<21 {
            if i == 0 {
                points.append(Point3D(x: 0, y: 0, z: 0))
            } else if i == 9 {
                points.append(Point3D(x: 0, y: scale, z: 0))
            } else {
                let f = Float(i)
                let baseX = scale * sinf(f)
                let baseY = scale * cosf(f) * 0.5
                let baseZ = scale * sinf(f * 0.7) * 0.3
                let shift = Float(n) * delta
                points.append(Point3D(x: baseX + shift, y: baseY + shift, z: baseZ))
            }
        }
        return HandShot(landmarks: points, timestamp: t, leftOrRight: .right, isAbsent: false)
    }

    // MARK: - Energy / normalisation

    func testEnergyIsZeroForStationaryHand() {
        let h = makeHand()
        let n = MotionGate.normalize(shot(h, at: 0))!
        XCTAssertEqual(MotionGate.energy(current: n, previous: n), 0, accuracy: 1e-6)
    }

    func testEnergyIsTranslationInvariant() {
        let h1 = makeHand()
        let h2 = translateHand(h1, by: (0.5, 0, 0))
        let n1 = MotionGate.normalize(shot(h1, at: 0))!
        let n2 = MotionGate.normalize(shot(h2, at: 1.0/30))!
        XCTAssertEqual(MotionGate.energy(current: n2, previous: n1), 0, accuracy: 1e-5)
    }

    func testEnergyIsScaleInvariant() {
        let h1 = makeHand()
        let h2 = scaleHand(h1, by: 2.0)
        let n1 = MotionGate.normalize(shot(h1, at: 0))!
        let n2 = MotionGate.normalize(shot(h2, at: 1.0/30))!
        XCTAssertEqual(MotionGate.energy(current: n2, previous: n1), 0, accuracy: 1e-4)
    }

    func testEnergyIsPositiveForFingerMotion() {
        var h2 = makeHand()
        let h1 = h2
        h2[4] = Point3D(x: h1[4].x + 0.05, y: h1[4].y, z: h1[4].z)
        let n1 = MotionGate.normalize(shot(h1, at: 0))!
        let n2 = MotionGate.normalize(shot(h2, at: 1.0/30))!
        XCTAssertGreaterThan(MotionGate.energy(current: n2, previous: n1), 0.1)
    }

    func testNormalizeReturnsNilForDegenerateHand() {
        let zeros = (0..<21).map { _ in Point3D(x: 0, y: 0, z: 0) }
        XCTAssertNil(MotionGate.normalize(shot(zeros, at: 0)))
    }

    // MARK: - State machine

    private func config(tOpen: Float = 0.5,
                        kOpenMs: TimeInterval = 33,
                        tClose: Float = 0.05,
                        kCloseMs: TimeInterval = 100,
                        cooldownMs: TimeInterval = 1000) -> MotionGateConfig {
        MotionGateConfig(
            tOpen: tOpen, kOpenMs: kOpenMs,
            tClose: tClose, kCloseMs: kCloseMs,
            cooldownMs: cooldownMs
        )
    }

    func testGateStartsClosed() {
        let gate = MotionGate(config: config(), bufferCap: 30)
        XCTAssertEqual(gate.state, .closed)
        XCTAssertEqual(gate.bufferCount, 0)
    }

    func testGateOpensAfterSustainedAboveThreshold() {
        let gate = MotionGate(config: config(tOpen: 0.5, kOpenMs: 33), bufferCap: 30)
        // Frame 1 establishes prev coords; energy=0.
        XCTAssertEqual(gate.process(movingFrame(step: 0, at: 0)), .stillClosed)
        // Frame 2: energy > T_open, but aboveThresholdSince just set → durationMs=0.
        XCTAssertEqual(gate.process(movingFrame(step: 1, at: 1.0/30)), .stillClosed)
        // Frame 3: durationMs = 33 ≥ K_open → opens.
        XCTAssertEqual(gate.process(movingFrame(step: 2, at: 2.0/30)), .opened)
        XCTAssertEqual(gate.state, .open)
    }

    func testGateStaysClosedBelowThreshold() {
        let gate = MotionGate(config: config(tOpen: 1.0), bufferCap: 30)
        // Tiny per-step shift → energy well below T_open=1.0
        for i in 0..<10 {
            let event = gate.process(movingFrame(step: i, at: Double(i) / 30, delta: 0.0001))
            XCTAssertEqual(event, .stillClosed, "frame \(i)")
        }
        XCTAssertEqual(gate.state, .closed)
    }

    func testGateClosesAfterSustainedBelowTClose() {
        let gate = MotionGate(
            config: config(tOpen: 0.5, kOpenMs: 33, tClose: 0.05, kCloseMs: 100),
            bufferCap: 30
        )
        // Open the gate
        _ = gate.process(movingFrame(step: 0, at: 0))
        _ = gate.process(movingFrame(step: 1, at: 1.0/30))
        _ = gate.process(movingFrame(step: 2, at: 2.0/30))
        XCTAssertEqual(gate.state, .open)

        // Stationary frames: keep feeding step=2 so consecutive-frame energy=0,
        // which is below T_close=0.05. Should close once 100 ms has elapsed.
        var t = 3.0/30
        var closed = false
        for _ in 0..<8 {
            let event = gate.process(movingFrame(step: 2, at: t))
            t += 1.0/30
            if case .cycleEnded = event { closed = true; break }
        }
        XCTAssertTrue(closed, "Gate should have closed within 8 stationary frames")
        XCTAssertEqual(gate.state, .closed)
    }

    func testAbsentFrameClosesOpenGateImmediately() {
        let gate = MotionGate(config: config(), bufferCap: 30)
        // Open the gate; the opening frame itself doesn't accumulate, so feed
        // one extra moving frame so the buffer has something to emit.
        _ = gate.process(movingFrame(step: 0, at: 0))
        _ = gate.process(movingFrame(step: 1, at: 1.0/30))
        _ = gate.process(movingFrame(step: 2, at: 2.0/30))   // opens
        _ = gate.process(movingFrame(step: 3, at: 3.0/30))   // appended
        XCTAssertEqual(gate.state, .open)
        XCTAssertEqual(gate.bufferCount, 1)

        let event = gate.process(HandShot.absent(timestamp: 4.0/30))
        if case .cycleEnded(let buffer) = event {
            XCTAssertEqual(buffer.count, 1)
        } else {
            XCTFail("Expected .cycleEnded on absent frame, got \(event)")
        }
        XCTAssertEqual(gate.state, .closed)
        XCTAssertEqual(gate.bufferCount, 0)
    }

    func testAbsentFrameWhileClosedIsNoOp() {
        let gate = MotionGate(config: config(), bufferCap: 30)
        let event = gate.process(HandShot.absent(timestamp: 0))
        XCTAssertEqual(event, .stillClosed)
        XCTAssertEqual(gate.state, .closed)
    }

    func testBufferCapTriggersCycleEnd() {
        let cap = 5
        // T_close very small so per-step energy stays above it; K_close enormous
        // so only the buffer cap can trigger a cycle end.
        let gate = MotionGate(
            config: config(tOpen: 0.5, kOpenMs: 33, tClose: 0.0001, kCloseMs: 100_000),
            bufferCap: cap
        )
        _ = gate.process(movingFrame(step: 0, at: 0))
        _ = gate.process(movingFrame(step: 1, at: 1.0/30))
        _ = gate.process(movingFrame(step: 2, at: 2.0/30))
        XCTAssertEqual(gate.state, .open)

        // Continuous motion. Each post-open frame appends one to the buffer.
        // The cap-th appended frame triggers cycleEnd.
        var lastEvent: MotionGate.Event = .stillClosed
        for i in 0..<(cap + 5) {
            lastEvent = gate.process(movingFrame(step: 3 + i, at: Double(3 + i) / 30))
            if case .cycleEnded = lastEvent { break }
        }
        if case .cycleEnded(let buffer) = lastEvent {
            XCTAssertEqual(buffer.count, cap)
        } else {
            XCTFail("Expected cycleEnded after buffer cap, got \(lastEvent)")
        }
    }

    func testResetReturnsGateToInitialState() {
        let gate = MotionGate(config: config(), bufferCap: 30)
        _ = gate.process(movingFrame(step: 0, at: 0))
        _ = gate.process(movingFrame(step: 1, at: 1.0/30))
        _ = gate.process(movingFrame(step: 2, at: 2.0/30))
        XCTAssertEqual(gate.state, .open)

        gate.reset()
        XCTAssertEqual(gate.state, .closed)
        XCTAssertEqual(gate.bufferCount, 0)
    }
}

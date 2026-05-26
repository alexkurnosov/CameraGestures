// Stage 5: HandGestureRecognizing Swift binding.
// Public API is identical to the V1 HandGestureRecognizingFramework so Training App v2
// compiles without source changes after the pod swap in Stage 5.
//
// Threading: all C++ calls are serialised on `pipelineQueue` — exactly as V1 did
// with its private `handshotQueue`.
//
// Timer model: a DispatchSourceTimer ticks `cg_recognizer_tick_timers` every 10 ms
// while the gate is running, replacing the Swift Structured Concurrency Tasks used
// in V1.

import Foundation
import AVFoundation
import CameraGesturesC

// MARK: - Re-export types that V1 HandGestureRecognizingFramework exposed

// HandGestureRecognizingConfig, MotionGateConfig, HoldsConfig, MotionGateState,
// DetectedGesture, GestureRecognizingStats, GestureRecognizingStatus,
// HoldsTelemetry, and all callback typedefs are declared below, matching the
// V1 Types.swift from HandGestureRecognizingFramework verbatim.

// --------------------------------------------------------------------------
// MARK: - Configuration
// --------------------------------------------------------------------------

public struct MotionGateConfig {
    public let tOpen:      Float
    public let kOpenMs:    TimeInterval
    public let tClose:     Float
    public let kCloseMs:   TimeInterval
    public let cooldownMs: TimeInterval

    public init(tOpen:      Float        = 1.0,
                kOpenMs:    TimeInterval = 33.0,
                tClose:     Float        = 0.5,
                kCloseMs:   TimeInterval = 200.0,
                cooldownMs: TimeInterval = 1000.0) {
        self.tOpen      = tOpen
        self.kOpenMs    = kOpenMs
        self.tClose     = tClose
        self.kCloseMs   = kCloseMs
        self.cooldownMs = cooldownMs
    }

    public static let defaultConfig = MotionGateConfig()
}

public struct HoldsConfig {
    public let tHold:              Float
    public let kHoldMs:            TimeInterval
    public let smoothKMs:          TimeInterval
    public let tCommitMs:          TimeInterval
    public let tMinBufferMs:       TimeInterval
    public let tauPoseConfidence:  Float
    public let tauPhase3Confidence: Float

    public init(tHold:              Float        = 2.10,
                kHoldMs:            TimeInterval = 100,
                smoothKMs:          TimeInterval = 100,
                tCommitMs:          TimeInterval = 300,
                tMinBufferMs:       TimeInterval = 200,
                tauPoseConfidence:  Float        = 0.6,
                tauPhase3Confidence: Float       = 0.7) {
        self.tHold              = tHold
        self.kHoldMs            = kHoldMs
        self.smoothKMs          = smoothKMs
        self.tCommitMs          = tCommitMs
        self.tMinBufferMs       = tMinBufferMs
        self.tauPoseConfidence  = tauPoseConfidence
        self.tauPhase3Confidence = tauPhase3Confidence
    }

    public static let defaultConfig = HoldsConfig()
}

public struct HandGestureRecognizingConfig {
    public let handsRecognizingConfig: HandsRecognizingConfig
    public let gestureModelConfig:     GestureModelConfig
    public let enableRealTimeProcessing: Bool
    public let gestureBufferSize:      Int
    public let confidenceThreshold:    Float
    public let motionGateConfig:       MotionGateConfig?
    public let holdsConfig:            HoldsConfig?

    public init(handsRecognizingConfig: HandsRecognizingConfig = .defaultConfig,
                gestureModelConfig:     GestureModelConfig     = .defaultConfig,
                enableRealTimeProcessing: Bool = true,
                gestureBufferSize:      Int   = 30,
                confidenceThreshold:    Float = 0.7,
                motionGateConfig:       MotionGateConfig? = nil,
                holdsConfig:            HoldsConfig?      = nil) {
        self.handsRecognizingConfig  = handsRecognizingConfig
        self.gestureModelConfig      = gestureModelConfig
        self.enableRealTimeProcessing = enableRealTimeProcessing
        self.gestureBufferSize       = gestureBufferSize
        self.confidenceThreshold     = confidenceThreshold
        self.motionGateConfig        = motionGateConfig
        self.holdsConfig             = holdsConfig
    }

    public static let defaultConfig = HandGestureRecognizingConfig()
}

// --------------------------------------------------------------------------
// MARK: - Status
// --------------------------------------------------------------------------

public enum MotionGateState: Equatable {
    case closed, open
    public var displayName: String { self == .open ? "Open" : "Closed" }
}

public enum GestureRecognizingStatus: Equatable {
    case idle, initializing, running, paused, stopping
    case error(String)
    public var isActive: Bool { self == .running }
    public var displayName: String {
        switch self {
        case .idle:         return "Idle"
        case .initializing: return "Initializing"
        case .running:      return "Running"
        case .paused:       return "Paused"
        case .stopping:     return "Stopping"
        case .error:        return "Error"
        }
    }
}

// --------------------------------------------------------------------------
// MARK: - Data types
// --------------------------------------------------------------------------

public struct DetectedGesture {
    public let prediction:          GesturePrediction
    public let handfilm:            HandFilm
    public let handedness:          LeftOrRight
    public let detectionTimestamp:  TimeInterval
    public let processingLatency:   TimeInterval
    public let candidateSetSize:    Int?

    public init(prediction:         GesturePrediction,
                handfilm:           HandFilm,
                handedness:         LeftOrRight,
                detectionTimestamp: TimeInterval = Date().timeIntervalSince1970,
                processingLatency:  TimeInterval = 0,
                candidateSetSize:   Int?         = nil) {
        self.prediction         = prediction
        self.handfilm           = handfilm
        self.handedness         = handedness
        self.detectionTimestamp = detectionTimestamp
        self.processingLatency  = processingLatency
        self.candidateSetSize   = candidateSetSize
    }
}

public struct GestureRecognizingStats {
    public let totalGesturesDetected:   Int
    public let averageProcessingLatency: TimeInterval
    public let averageConfidence:        Float
    public let gesturesByType:          [String: Int]
    public let uptime:                  TimeInterval
    public let fps:                     Float

    public init(totalGesturesDetected:   Int           = 0,
                averageProcessingLatency: TimeInterval  = 0,
                averageConfidence:        Float         = 0,
                gesturesByType:          [String: Int]  = [:],
                uptime:                  TimeInterval   = 0,
                fps:                     Float          = 0) {
        self.totalGesturesDetected   = totalGesturesDetected
        self.averageProcessingLatency = averageProcessingLatency
        self.averageConfidence       = averageConfidence
        self.gesturesByType          = gesturesByType
        self.uptime                  = uptime
        self.fps                     = fps
    }
}

public struct HoldsTelemetry: Equatable {
    public let lastPoseId:          Int?
    public let lastPoseLabel:       String?
    public let lastPoseConfidence:  Float?
    public let lastPoseKind:        String?
    public let observedSequence:    [Int]
    public let matchedGesture:      String?

    public init(lastPoseId:         Int?    = nil,
                lastPoseLabel:      String? = nil,
                lastPoseConfidence: Float?  = nil,
                lastPoseKind:       String? = nil,
                observedSequence:   [Int]   = [],
                matchedGesture:     String? = nil) {
        self.lastPoseId         = lastPoseId
        self.lastPoseLabel      = lastPoseLabel
        self.lastPoseConfidence = lastPoseConfidence
        self.lastPoseKind       = lastPoseKind
        self.observedSequence   = observedSequence
        self.matchedGesture     = matchedGesture
    }
}

// --------------------------------------------------------------------------
// MARK: - Errors
// --------------------------------------------------------------------------

public enum HandGestureRecognizingError: Error {
    case notInitialized
    case alreadyRunning
    case notRunning
    case handsRecognizingError(Error)
    case gestureModelError(Error)
    case configurationError(String)
    case cameraPermissionDenied
    case processingError(String)
}

// --------------------------------------------------------------------------
// MARK: - Callback typedefs
// --------------------------------------------------------------------------

public typealias GestureDetectionCallback    = (DetectedGesture) -> Void
public typealias RawPredictionsCallback      = ([GesturePrediction]) -> Void
public typealias HandTrackingUpdateCallback  = (HandShot) -> Void
public typealias StatusChangeCallback        = (GestureRecognizingStatus) -> Void
public typealias MotionGateUpdateCallback    = (MotionGateState, Int) -> Void
public typealias HoldsModeTelemetryCallback  = (HoldsTelemetry) -> Void

// --------------------------------------------------------------------------
// MARK: - HandGestureRecognizing
// --------------------------------------------------------------------------

public class HandGestureRecognizing {

    // MARK: Public callbacks

    public var gestureDetectionCallback:    GestureDetectionCallback?
    /// Fires on every cycle end with ALL gesture probabilities, regardless of
    /// confidence threshold and without cooldown gating.  Use for diagnostics.
    public var rawPredictionsCallback:      RawPredictionsCallback?
    public var handTrackingUpdateCallback:  HandTrackingUpdateCallback?
    public var statusChangeCallback:        StatusChangeCallback?
    public var motionGateUpdateCallback:    MotionGateUpdateCallback?
    public var holdsModeTelemetryCallback:  HoldsModeTelemetryCallback?
    public var handshotCallback:            HandShotCallback?
    public var handfilmCallback:            HandFilmCallback?

    // MARK: Runtime flags

    public var gateEnabled: Bool {
        get { _gateEnabled }
        set {
            _gateEnabled = newValue
            pipelineQueue.async { [weak self] in
                guard let self else { return }
                cg_recognizer_set_gate_enabled(self.recognizerRef, newValue ? 1 : 0)
            }
        }
    }

    public var bypassPhase2Filter: Bool {
        get { _bypassPhase2 }
        set {
            _bypassPhase2 = newValue
            pipelineQueue.async { [weak self] in
                guard let self else { return }
                cg_recognizer_set_bypass_phase2(self.recognizerRef, newValue ? 1 : 0)
            }
        }
    }

    // MARK: Properties

    private var config: HandGestureRecognizingConfig
    private var handsRecognizer: HandsRecognizing
    private var gestureModel: GestureModel

    private var isInitialized = false
    private var isRunning     = false
    private var startTime: TimeInterval = 0

    // C++ recognizer handle
    private var recognizerRef: cg_recognizer_ref?

    // Serial pipeline queue (same role as V1 handshotQueue)
    private let pipelineQueue = DispatchQueue(
        label: "com.cameragestures.pipeline", qos: .userInteractive)

    // Timer tick for T_commit / T_min_buffer
    private var timerSource: DispatchSourceTimer?

    // Stats
    private var detectedGestures:  [DetectedGesture]  = []
    private var processingTimes:   [TimeInterval]      = []
    private var confidenceScores:  [Float]             = []
    private var recentHandshots:   [HandShot]          = []

    // Cooldown (main thread)
    private var cooldownEndTime: TimeInterval?  = nil
    private var cooldownDuration: TimeInterval  = 1.0
    private var pendingGesture: DetectedGesture? = nil
    private var cooldownGen: Int = 0

    private var _gateEnabled  = false
    private var _bypassPhase2 = false

    private var currentStatus: GestureRecognizingStatus = .idle {
        didSet {
            DispatchQueue.main.async { [weak self] in
                self?.statusChangeCallback?(self?.currentStatus ?? .idle)
            }
        }
    }

    // MARK: Initialisation

    public init() {
        self.config          = .defaultConfig
        self.handsRecognizer = HandsRecognizing()
        self.gestureModel    = GestureModel()
    }

    public init(config: HandGestureRecognizingConfig) {
        self.config          = config
        self.handsRecognizer = HandsRecognizing()
        self.gestureModel    = GestureModel(config: config.gestureModelConfig)
    }

    deinit {
        stopTimerTick()
        if let ref = recognizerRef { cg_recognizer_destroy(ref) }
    }

    // MARK: Configuration

    public func initialize(config: HandGestureRecognizingConfig? = nil) async throws {
        guard !isInitialized else { return }
        currentStatus = .initializing
        if let c = config { self.config = c }

        do {
            try handsRecognizer.initialize(config: self.config.handsRecognizingConfig)
            setupHandsRecognizingCallbacks()
            try gestureModel.initialize(config: self.config.gestureModelConfig)
            buildRecognizer()
            isInitialized = true
            currentStatus = .idle
        } catch {
            currentStatus = .error(error.localizedDescription)
            throw HandGestureRecognizingError.configurationError(error.localizedDescription)
        }
    }

    // MARK: Lifecycle

    public func start() async throws {
        guard isInitialized else { throw HandGestureRecognizingError.notInitialized }
        guard !isRunning     else { throw HandGestureRecognizingError.alreadyRunning }
        guard await HandsRecognizing.requestCameraPermission() else {
            throw HandGestureRecognizingError.cameraPermissionDenied
        }
        do {
            currentStatus = .initializing
            try handsRecognizer.start()
            isRunning = true
            startTime = Date().timeIntervalSince1970
            resetStats()
            startTimerTick()
            currentStatus = .running
        } catch {
            currentStatus = .error(error.localizedDescription)
            throw HandGestureRecognizingError.handsRecognizingError(error)
        }
    }

    public func stop() {
        guard isRunning else { return }
        currentStatus = .stopping
        handsRecognizer.stop()
        stopTimerTick()
        isRunning = false
        currentStatus = .idle
    }

    public func pause() {
        guard isRunning else { return }
        handsRecognizer.stop()
        stopTimerTick()
        currentStatus = .paused
    }

    public func resume() async throws {
        guard currentStatus == .paused else { return }
        try handsRecognizer.start()
        startTimerTick()
        currentStatus = .running
    }

    // MARK: Stats / state

    public func getStatus()     -> GestureRecognizingStatus { currentStatus }
    public func getConfig()     -> HandGestureRecognizingConfig { config }
    public var  isReady: Bool   { isInitialized && gestureModel.isLoaded }
    public var  isActive: Bool  { isRunning && currentStatus.isActive }

    public func getStatistics() -> GestureRecognizingStats {
        let uptime  = isRunning ? Date().timeIntervalSince1970 - startTime : 0
        let avgLat  = processingTimes.isEmpty  ? 0 : processingTimes.reduce(0,+) / Double(processingTimes.count)
        let avgConf = confidenceScores.isEmpty ? 0 : confidenceScores.reduce(0,+) / Float(confidenceScores.count)
        var byType: [String: Int] = [:]
        for g in detectedGestures { byType[g.prediction.gestureName, default: 0] += 1 }
        let fps = uptime > 0 ? Float(recentHandshots.count) / Float(uptime) : 0
        return GestureRecognizingStats(
            totalGesturesDetected:   detectedGestures.count,
            averageProcessingLatency: avgLat,
            averageConfidence:       avgConf,
            gesturesByType:          byType,
            uptime:                  uptime,
            fps:                     fps)
    }

    public func getRecentGestures(limit: Int = 10) -> [DetectedGesture] {
        Array(detectedGestures.suffix(limit))
    }

    public func clearHistory() { resetStats() }

    // MARK: Model management

    public func loadModel(from path: String,
                          registryPath: String? = nil,
                          gestureIds: [String] = []) throws {
        try gestureModel.loadModel(from: path, registryPath: registryPath)
        if !gestureIds.isEmpty { gestureModel.setSupportedGestures(gestureIds) }
        pipelineQueue.async { [weak self] in
            guard let self, let ref = self.recognizerRef else { return }
            cg_recognizer_set_gesture_model(ref, self.gestureModel.modelRef)
        }
    }

    public func loadPoseModel(tflitePath: String, manifestPath: String) throws {
        try gestureModel.loadPoseModel(tflitePath: tflitePath, manifestPath: manifestPath)
        // Rebuild recognizer so PrefixMatcher picks up the new manifest.
        pipelineQueue.async { [weak self] in self?.rebuildRecognizerOnQueue() }
    }

    public func resetHandfilm() { handsRecognizer.resetHandfilm() }
    public func harvestHandfilm() -> HandFilm { handsRecognizer.harvestHandfilm() }

    /// The underlying AVCaptureSession — pass to CameraPreviewView for live video display.
    public var previewSession: AVCaptureSession? { handsRecognizer.previewSession }

    public func updateConfig(_ newConfig: HandGestureRecognizingConfig) throws {
        let wasRunning = isRunning
        if wasRunning { stop() }
        self.config = newConfig
        try handsRecognizer.initialize(config: newConfig.handsRecognizingConfig)
        try gestureModel.initialize(config: newConfig.gestureModelConfig)
        buildRecognizer()
        if wasRunning { Task { try await start() } }
    }

    public func resetGateState() {
        pipelineQueue.async { [weak self] in
            guard let self, let ref = self.recognizerRef else { return }
            cg_recognizer_reset_gate(ref)
        }
    }

    // MARK: Convenience

    public func quickStart() async throws {
        try await initialize()
        try await start()
    }

    // MARK: - Private: build C++ recognizer

    private func buildRecognizer() {
        if let old = recognizerRef { cg_recognizer_destroy(old); recognizerRef = nil }

        var c = cg_recognizer_default_config()
        c.gate_enabled        = config.motionGateConfig != nil ? 1 : 0
        c.gesture_buffer_size = Int32(config.gestureBufferSize)
        // Always pass 0.0 to the C++ layer so every cycle end fires the callback.
        // Swift-side threshold (config.confidenceThreshold) gates the main
        // gestureDetectionCallback; rawPredictionsCallback fires unconditionally.
        c.confidence_threshold = 0.0

        if let mg = config.motionGateConfig {
            c.motion_gate.t_open     = mg.tOpen
            c.motion_gate.k_open_ms  = mg.kOpenMs
            c.motion_gate.t_close    = mg.tClose
            c.motion_gate.k_close_ms = mg.kCloseMs
            c.motion_gate.cooldown_ms = mg.cooldownMs
        }
        if let h = config.holdsConfig {
            c.holds_enabled              = 1
            c.holds.t_hold               = h.tHold
            c.holds.k_hold_ms            = h.kHoldMs
            c.holds.smooth_k_ms          = h.smoothKMs
            c.holds.t_commit_ms          = h.tCommitMs
            c.holds.t_min_buffer_ms      = h.tMinBufferMs
            c.holds.tau_pose_confidence  = h.tauPoseConfidence
            c.holds.tau_phase3_confidence = h.tauPhase3Confidence
        }

        recognizerRef = cg_recognizer_create(&c, gestureModel.modelRef)
        wireCallbacks()
    }

    private func rebuildRecognizerOnQueue() {
        buildRecognizer()
    }

    private func wireCallbacks() {
        guard let ref = recognizerRef else { return }

        // Gesture callback
        let selfPtr = Unmanaged.passRetained(self)
        cg_recognizer_set_gesture_callback(ref, { ctx, predPtr, film, candidateSize in
            guard let ctx, let predPtr else { return }
            let hgr = Unmanaged<HandGestureRecognizing>.fromOpaque(ctx).takeUnretainedValue()
            hgr.handleGestureDetected(predPtr: predPtr, film: film, candidateSetSize: Int(candidateSize))
        }, selfPtr.toOpaque())

        // Gate update callback
        cg_recognizer_set_gate_update_callback(ref, { ctx, gateOpen, count in
            guard let ctx else { return }
            let hgr = Unmanaged<HandGestureRecognizing>.fromOpaque(ctx).takeUnretainedValue()
            let state: MotionGateState = gateOpen != 0 ? .open : .closed
            DispatchQueue.main.async { hgr.motionGateUpdateCallback?(state, Int(count)) }
        }, selfPtr.toOpaque())

        // Holds telemetry callback
        cg_recognizer_set_holds_telemetry_callback(ref, { ctx, poseId, conf, seqPtr, seqLen, matchedPtr in
            guard let ctx else { return }
            let hgr = Unmanaged<HandGestureRecognizing>.fromOpaque(ctx).takeUnretainedValue()
            var seq: [Int] = []
            if let p = seqPtr { seq = (0..<Int(seqLen)).map { Int(p[$0]) } }
            let matched = matchedPtr.flatMap { String(cString: $0).nilIfEmpty() }
            let telemetry = HoldsTelemetry(
                lastPoseId:         Int(poseId),
                lastPoseLabel:      nil,
                lastPoseConfidence: conf,
                lastPoseKind:       nil,
                observedSequence:   seq,
                matchedGesture:     matched)
            DispatchQueue.main.async { hgr.holdsModeTelemetryCallback?(telemetry) }
        }, selfPtr.toOpaque())

        // Release the retained reference when done wiring — the C++ side holds a raw
        // pointer, so we keep the Swift object alive through the recognizer's lifetime.
        // The deinit destroys the C++ recognizer before releasing self, which is safe.
        selfPtr.release()
    }

    // MARK: - Private: camera callbacks

    private func setupHandsRecognizingCallbacks() {
        handsRecognizer.handshotCallback = { [weak self] shot in
            self?.handleHandshot(shot)
            self?.handshotCallback?(shot)
        }
        handsRecognizer.handfilmCallback = { [weak self] film in
            self?.handfilmCallback?(film)
            // In gate mode, recognition fires from the C++ pipeline — not from handfilms.
        }
    }

    private func handleHandshot(_ shot: HandShot) {
        pipelineQueue.async { [weak self] in
            guard let self else { return }
            self.recentHandshots.append(shot)
            if self.recentHandshots.count > self.config.gestureBufferSize {
                self.recentHandshots.removeFirst()
            }
            DispatchQueue.main.async { self.handTrackingUpdateCallback?(shot) }

            guard let ref = self.recognizerRef else { return }
            var c = shot.toCHandshot()
            cg_recognizer_process_shot(ref, &c)
        }
    }

    // MARK: - Private: gesture received from C++ pipeline

    private func handleGestureDetected(predPtr: UnsafePointer<cg_gesture_prediction>,
                                        film: cg_handfilm_ref?,
                                        candidateSetSize: Int) {
        // Called on pipelineQueue from the C++ callback.
        let t0 = Date().timeIntervalSince1970
        let pred = GesturePrediction(fromCStruct: predPtr.pointee)
        let handfilm = handfilmFromRef(film)
        let handedness = handfilm.frames.first?.leftOrRight ?? .unknown

        // Always fire raw predictions (all classes, no threshold, no cooldown).
        let allPreds = gestureModel.predictAllGestures(handfilm: handfilm)
        let top = allPreds.first
        print("[CG:Swift] handleGestureDetected  top=\(top?.gestureName ?? "-")  conf=\(String(format: "%.3f", top?.confidence ?? 0))  threshold=\(config.confidenceThreshold)")
        Task { @MainActor [weak self] in
            self?.rawPredictionsCallback?(allPreds)
        }

        // Main callback only fires when confidence meets the Swift-side threshold.
        guard pred.confidence >= config.confidenceThreshold else { return }

        let detected = DetectedGesture(
            prediction:         pred,
            handfilm:           handfilm,
            handedness:         handedness,
            detectionTimestamp: Date().timeIntervalSince1970,
            processingLatency:  Date().timeIntervalSince1970 - t0,
            candidateSetSize:   candidateSetSize >= 0 ? candidateSetSize : nil)

        let cooldownSec = (config.motionGateConfig?.cooldownMs ?? 1000) / 1000.0
        Task { @MainActor [weak self] in
            self?.emitOrQueueGated(detected, cooldown: cooldownSec)
        }
    }

    // MARK: - Private: cooldown (MainActor)

    @MainActor
    private func emitOrQueueGated(_ gesture: DetectedGesture, cooldown: TimeInterval) {
        let now = Date().timeIntervalSince1970
        if let end = cooldownEndTime, now < end {
            pendingGesture = gesture
        } else {
            processDetectedGesture(gesture)
            startCooldown(duration: cooldown)
        }
    }

    @MainActor
    private func startCooldown(duration: TimeInterval) {
        cooldownGen += 1
        let gen = cooldownGen
        cooldownEndTime = Date().timeIntervalSince1970 + duration
        cooldownDuration = duration
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
            self?.cooldownExpired(generation: gen)
        }
    }

    @MainActor
    private func cooldownExpired(generation: Int) {
        guard generation == cooldownGen else { return }
        cooldownEndTime = nil
        guard let g = pendingGesture else { return }
        pendingGesture = nil
        processDetectedGesture(g)
        startCooldown(duration: cooldownDuration)
    }

    @MainActor
    private func processDetectedGesture(_ gesture: DetectedGesture) {
        detectedGestures.append(gesture)
        processingTimes.append(gesture.processingLatency)
        confidenceScores.append(gesture.prediction.confidence)
        if detectedGestures.count > 1000 { detectedGestures.removeFirst(detectedGestures.count - 1000) }
        gestureDetectionCallback?(gesture)
    }

    // MARK: - Private: timer tick

    private func startTimerTick() {
        let src = DispatchSource.makeTimerSource(queue: pipelineQueue)
        src.schedule(deadline: .now(), repeating: .milliseconds(10))
        src.setEventHandler { [weak self] in
            guard let self, let ref = self.recognizerRef else { return }
            let now = Date().timeIntervalSince1970
            _ = cg_recognizer_tick_timers(ref, now)
        }
        src.resume()
        timerSource = src
    }

    private func stopTimerTick() {
        timerSource?.cancel()
        timerSource = nil
    }

    // MARK: - Private: helpers

    private func resetStats() {
        detectedGestures.removeAll()
        processingTimes.removeAll()
        confidenceScores.removeAll()
        recentHandshots.removeAll()
    }

    private func handfilmFromRef(_ ref: cg_handfilm_ref?) -> HandFilm {
        guard let ref else { return HandFilm() }
        let count = cg_handfilm_shot_count(ref)
        var film = HandFilm(startTime: cg_handfilm_start_time(ref))
        for i in 0..<count {
            var shot = cg_handshot()
            if cg_handfilm_get_shot(ref, i, &shot) != 0 {
                film.addFrame(HandShot(fromCStruct: shot))
            }
        }
        return film
    }
}

// --------------------------------------------------------------------------
// MARK: - Helpers
// --------------------------------------------------------------------------

// Internal (not private) so HandsRecognizing.swift and VisionHandLandmarker.swift
// can call init(fromCStruct:) without a redeclaration conflict.
extension HandShot {
    func toCHandshot() -> cg_handshot {
        var c        = cg_handshot()
        c.timestamp  = self.timestamp
        c.handedness = self.leftOrRight == .left  ? CG_HAND_LEFT
                     : self.leftOrRight == .right ? CG_HAND_RIGHT : CG_HAND_UNKNOWN
        c.is_absent  = self.isAbsent ? 1 : 0
        withUnsafeMutableBytes(of: &c.landmarks) { buf in
            let typed = buf.bindMemory(to: cg_point3d.self)
            for (i, pt) in self.landmarks.prefix(21).enumerated() {
                typed[i] = cg_point3d(x: pt.x, y: pt.y, z: pt.z)
            }
        }
        return c
    }

    init(fromCStruct c: cg_handshot) {
        var lms: [Point3D] = []
        withUnsafeBytes(of: c.landmarks) { buf in
            let typed = buf.bindMemory(to: cg_point3d.self)
            for i in 0..<21 { lms.append(Point3D(x: typed[i].x, y: typed[i].y, z: typed[i].z)) }
        }
        self.init(landmarks: lms,
                  timestamp: c.timestamp,
                  leftOrRight: c.handedness == CG_HAND_LEFT ? .left
                             : c.handedness == CG_HAND_RIGHT ? .right : .unknown,
                  isAbsent: c.is_absent != 0)
    }
}

private extension GesturePrediction {
    init(fromCStruct c: cg_gesture_prediction) {
        var c = c
        let id   = withUnsafeBytes(of: &c.gesture_id)   { String(bytes: $0.prefix(while: { $0 != 0 }), encoding: .utf8) ?? "" }
        let name = withUnsafeBytes(of: &c.gesture_name) { String(bytes: $0.prefix(while: { $0 != 0 }), encoding: .utf8) ?? "" }
        self.init(gestureId: id, gestureName: name, confidence: c.confidence, timestamp: c.timestamp)
    }
}

private extension String {
    func nilIfEmpty() -> String? { isEmpty ? nil : self }
}


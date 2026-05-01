import Foundation
import UIKit
import HandGestureTypes
import HandsRecognizingModule
import GestureModelModule

/// Production-ready gesture recognition module that orchestrates hand tracking and gesture classification
public class HandGestureRecognizing {

    // MARK: - Properties

    private var config: HandGestureRecognizingConfig
    private var handsRecognizer: HandsRecognizing
    private var gestureModel: GestureModel

    private var isInitialized = false
    private var isRunning = false
    private var startTime: TimeInterval = 0

    // Statistics tracking
    private var stats = GestureRecognizingStats()
    private var detectedGestures: [DetectedGesture] = []
    private var processingTimes: [TimeInterval] = []
    private var confidenceScores: [Float] = []

    // Rolling buffer used for legacy real-time recognition and FPS stats.
    private var recentHandshots: [HandShot] = []
    private let handshotQueue = DispatchQueue(label: "com.cameragestures.handshot", qos: .userInteractive)

    // MARK: - Motion Gate (Phase 1)
    // All gate state is accessed only on handshotQueue.

    /// Set to `true` to route incoming frames through the Phase 1 motion gate instead of
    /// continuous per-frame recognition. Requires `config.motionGateConfig` to be non-nil.
    public var gateEnabled: Bool = false

    /// Fired on the main thread whenever the gate state or buffer count changes.
    public var motionGateUpdateCallback: MotionGateUpdateCallback?

    private var motionGate: MotionGate? = nil

    // MARK: - Cooldown state (main thread only)

    private var cooldownEndTime: TimeInterval? = nil
    private var cooldownDuration: TimeInterval = 1.0
    private var pendingGesture: DetectedGesture? = nil
    private var cooldownGenerationCount: Int = 0

    // MARK: - Callbacks

    public var gestureDetectionCallback: GestureDetectionCallback?
    public var handTrackingUpdateCallback: HandTrackingUpdateCallback?
    public var statusChangeCallback: StatusChangeCallback?

    /// Called for every individual handshot received from the camera.
    public var handshotCallback: HandShotCallback?

    /// Called whenever a completed handfilm is produced.
    public var handfilmCallback: HandFilmCallback?

    private var currentStatus: GestureRecognizingStatus = .idle {
        didSet {
            DispatchQueue.main.async { [weak self] in
                self?.statusChangeCallback?(self?.currentStatus ?? .idle)
            }
        }
    }

    // MARK: - Initialization

    public init() {
        self.config = .defaultConfig
        self.handsRecognizer = HandsRecognizing()
        self.gestureModel = GestureModel()
    }

    public init(config: HandGestureRecognizingConfig) {
        self.config = config
        self.handsRecognizer = HandsRecognizing()
        self.gestureModel = GestureModel(config: config.gestureModelConfig)
    }

    // MARK: - Configuration

    /// Initialize the gesture recognition system
    public func initialize(config: HandGestureRecognizingConfig? = nil) async throws {
        guard !isInitialized else { return }

        currentStatus = .initializing

        if let newConfig = config {
            self.config = newConfig
        }

        do {
            try handsRecognizer.initialize(config: self.config.handsRecognizingConfig)
            setupHandsRecognizingCallbacks()
            try gestureModel.initialize(config: self.config.gestureModelConfig)
            if let gateConfig = self.config.motionGateConfig {
                motionGate = MotionGate(config: gateConfig, bufferCap: self.config.gestureBufferSize)
            }
            isInitialized = true
            currentStatus = .idle
        } catch {
            currentStatus = .error(error.localizedDescription)
            throw HandGestureRecognizingError.configurationError(error.localizedDescription)
        }
    }

    // MARK: - Lifecycle

    /// Start gesture recognition
    public func start() async throws {
        guard isInitialized else {
            throw HandGestureRecognizingError.notInitialized
        }
        guard !isRunning else {
            throw HandGestureRecognizingError.alreadyRunning
        }

        let hasPermission = await HandsRecognizing.requestCameraPermission()
        guard hasPermission else {
            throw HandGestureRecognizingError.cameraPermissionDenied
        }

        do {
            currentStatus = .initializing
            try handsRecognizer.start()
            isRunning = true
            startTime = Date().timeIntervalSince1970
            resetStats()
            currentStatus = .running
        } catch {
            currentStatus = .error(error.localizedDescription)
            throw HandGestureRecognizingError.handsRecognizingError(error)
        }
    }

    /// Stop gesture recognition
    public func stop() {
        guard isRunning else { return }
        currentStatus = .stopping
        handsRecognizer.stop()
        isRunning = false
        currentStatus = .idle
    }

    /// Pause gesture recognition
    public func pause() {
        guard isRunning else { return }
        handsRecognizer.stop()
        currentStatus = .paused
    }

    /// Resume gesture recognition from paused state
    public func resume() async throws {
        guard currentStatus == .paused else { return }
        try handsRecognizer.start()
        currentStatus = .running
    }

    // MARK: - Status and Statistics

    public func getStatus() -> GestureRecognizingStatus { currentStatus }

    public func getStatistics() -> GestureRecognizingStats {
        updateStatistics()
        return stats
    }

    public func getRecentGestures(limit: Int = 10) -> [DetectedGesture] {
        Array(detectedGestures.suffix(limit))
    }

    public func clearHistory() {
        detectedGestures.removeAll()
        processingTimes.removeAll()
        confidenceScores.removeAll()
        resetStats()
    }

    // MARK: - Configuration Updates

    /// Discard the in-progress handfilm buffer and start fresh.
    public func resetHandfilm() {
        handsRecognizer.resetHandfilm()
    }

    /// Return the accumulated handfilm and reset the buffer atomically.
    public func harvestHandfilm() -> HandFilm {
        handsRecognizer.harvestHandfilm()
    }

    /// Load (or reload) the gesture model from a file path without restarting the recognizer.
    public func loadModel(from path: String, gestureIds: [String] = []) throws {
        try gestureModel.loadModel(from: path)
        if !gestureIds.isEmpty {
            gestureModel.setSupportedGestures(gestureIds)
        }
    }

    /// Update configuration while running
    public func updateConfig(_ newConfig: HandGestureRecognizingConfig) throws {
        let wasRunning = isRunning
        if wasRunning { stop() }
        self.config = newConfig
        try handsRecognizer.initialize(config: newConfig.handsRecognizingConfig)
        try gestureModel.initialize(config: newConfig.gestureModelConfig)
        if wasRunning {
            Task { try await start() }
        }
    }

    public func getConfig() -> HandGestureRecognizingConfig { config }

    /// Reset all gate state. Safe to call from any thread.
    public func resetGateState() {
        handshotQueue.async { [weak self] in
            guard let self else { return }
            self.motionGate?.reset()
            self.reportGateUpdate()
        }
    }

    // MARK: - Private Setup

    private func setupHandsRecognizingCallbacks() {
        handsRecognizer.handshotCallback = { [weak self] handshot in
            self?.handleHandshot(handshot)
            self?.handshotCallback?(handshot)
        }
        handsRecognizer.handfilmCallback = { [weak self] handfilm in
            self?.handleHandfilm(handfilm)
            self?.handfilmCallback?(handfilm)
        }
    }

    // MARK: - Per-Frame Routing

    private func handleHandshot(_ handshot: HandShot) {
        handshotQueue.async { [weak self] in
            guard let self else { return }

            // FPS-tracking buffer (always maintained)
            self.recentHandshots.append(handshot)
            if self.recentHandshots.count > self.config.gestureBufferSize {
                self.recentHandshots.removeFirst()
            }

            DispatchQueue.main.async {
                self.handTrackingUpdateCallback?(handshot)
            }

            if self.gateEnabled, let gate = self.motionGate, let gateConfig = self.config.motionGateConfig {
                self.handleHandshotWithGate(handshot, gate: gate, gateConfig: gateConfig)
            } else if self.config.enableRealTimeProcessing && self.recentHandshots.count >= 3 {
                self.performRealTimeGestureRecognition()
            }
        }
    }

    private func handleHandfilm(_ handfilm: HandFilm) {
        // In gate mode recognition fires from triggerCycleEnd, not from the handfilm stream.
        guard !gateEnabled else { return }
        Task { [weak self] in
            await self?.performGestureRecognition(on: handfilm)
        }
    }

    // MARK: - Gate Logic (called on handshotQueue; delegates to MotionGate)

    private func handleHandshotWithGate(_ handshot: HandShot, gate: MotionGate, gateConfig: MotionGateConfig) {
        let event = gate.process(handshot)
        switch event {
        case .stillClosed, .stillOpen, .opened:
            break
        case .cycleEnded(let buffer):
            handleCycleEnd(buffer: buffer, gateConfig: gateConfig)
        }
        reportGateUpdate(state: gate.state, count: gate.bufferCount)
    }

    private func handleCycleEnd(buffer: [HandShot], gateConfig: MotionGateConfig) {
        let cooldownSec = gateConfig.cooldownMs / 1000.0
        Task { @MainActor [weak self] in
            self?.startCooldown(duration: cooldownSec)
        }
        guard !buffer.isEmpty, gestureModel.isLoaded else { return }
        let film = makeHandFilm(from: buffer)
        Task { [weak self] in
            await self?.recognizeAndEmitGated(film)
        }
    }

    private func reportGateUpdate(state: MotionGateState, count: Int) {
        DispatchQueue.main.async { [weak self] in
            self?.motionGateUpdateCallback?(state, count)
        }
    }

    private func reportGateUpdate() {
        let state = motionGate?.state ?? .closed
        let count = motionGate?.bufferCount ?? 0
        reportGateUpdate(state: state, count: count)
    }

    private func makeHandFilm(from shots: [HandShot]) -> HandFilm {
        guard let first = shots.first else { return HandFilm() }
        var film = HandFilm(startTime: first.timestamp)
        for shot in shots { film.addFrame(shot) }
        return film
    }

    // MARK: - Gated Recognition

    private func recognizeAndEmitGated(_ film: HandFilm) async {
        let t0 = Date().timeIntervalSince1970
        do {
            let predictions = try await gestureModel.predictTopK(handfilm: film, k: 3)
            for prediction in predictions where prediction.confidence >= config.confidenceThreshold {
                let detected = DetectedGesture(
                    prediction: prediction,
                    handfilm: film,
                    handedness: film.frames.first?.leftOrRight ?? .unknown,
                    detectionTimestamp: Date().timeIntervalSince1970,
                    processingLatency: Date().timeIntervalSince1970 - t0
                )
                await MainActor.run { [weak self] in
                    self?.emitOrQueueGated(detected)
                }
                break
            }
        } catch {
            print("Gate recognition error: \(error)")
        }
    }

    // MARK: - Cooldown (MainActor-isolated)

    @MainActor
    private func startCooldown(duration: TimeInterval) {
        cooldownGenerationCount += 1
        let gen = cooldownGenerationCount
        cooldownEndTime = Date().timeIntervalSince1970 + duration
        cooldownDuration = duration
        pendingGesture = nil
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
            self?.cooldownExpired(generation: gen)
        }
    }

    @MainActor
    private func cooldownExpired(generation: Int) {
        guard generation == cooldownGenerationCount else { return }
        cooldownEndTime = nil
        guard let gesture = pendingGesture else { return }
        pendingGesture = nil
        processDetectedGesture(gesture)
        // Fresh cooldown starts from this emission.
        startCooldown(duration: cooldownDuration)
    }

    @MainActor
    private func emitOrQueueGated(_ gesture: DetectedGesture) {
        let now = Date().timeIntervalSince1970
        if let endTime = cooldownEndTime, now < endTime {
            pendingGesture = gesture  // most-recent wins
        } else {
            processDetectedGesture(gesture)
        }
    }

    // MARK: - Legacy Real-Time Recognition

    private func performRealTimeGestureRecognition() {
        guard gestureModel.isLoaded else { return }
        Task { [weak self] in
            guard let self else { return }
            do {
                let predictions = try await self.gestureModel.predictStreaming(handshots: self.recentHandshots)
                for prediction in predictions where prediction.confidence >= self.config.confidenceThreshold {
                    let detected = DetectedGesture(
                        prediction: prediction,
                        handfilm: HandFilm(),
                        handedness: self.recentHandshots.last?.leftOrRight ?? .unknown,
                        detectionTimestamp: Date().timeIntervalSince1970,
                        processingLatency: 0.0
                    )
                    await self.processDetectedGesture(detected)
                }
            } catch {
                print("Real-time gesture recognition error: \(error)")
            }
        }
    }

    private func performGestureRecognition(on handfilm: HandFilm) async {
        let t0 = Date().timeIntervalSince1970
        do {
            print("<<--prediction-->>prediction start")
            let predictions = try await gestureModel.predictTopK(handfilm: handfilm, k: 3)
            print("<<--prediction-->>predictions: \(predictions)")
            for prediction in predictions where prediction.confidence >= config.confidenceThreshold {
                let detected = DetectedGesture(
                    prediction: prediction,
                    handfilm: handfilm,
                    handedness: handfilm.frames.first?.leftOrRight ?? .unknown,
                    detectionTimestamp: Date().timeIntervalSince1970,
                    processingLatency: Date().timeIntervalSince1970 - t0
                )
                await processDetectedGesture(detected)
                break
            }
        } catch {
            print("Gesture recognition error: \(error)")
        }
    }

    @MainActor
    private func processDetectedGesture(_ gesture: DetectedGesture) {
        detectedGestures.append(gesture)
        processingTimes.append(gesture.processingLatency)
        confidenceScores.append(gesture.prediction.confidence)
        if detectedGestures.count > 1000 {
            detectedGestures.removeFirst(detectedGestures.count - 1000)
        }
        gestureDetectionCallback?(gesture)
    }

    private func updateStatistics() {
        let currentTime = Date().timeIntervalSince1970
        let uptime = isRunning ? currentTime - startTime : 0
        let avgLatency = processingTimes.isEmpty ? 0 :
            processingTimes.reduce(0, +) / Double(processingTimes.count)
        let avgConfidence = confidenceScores.isEmpty ? 0 :
            confidenceScores.reduce(0, +) / Float(confidenceScores.count)
        var gesturesByType: [String: Int] = [:]
        for gesture in detectedGestures {
            let key = gesture.prediction.gestureName
            gesturesByType[key] = (gesturesByType[key] ?? 0) + 1
        }
        let fps: Float = uptime > 0 ? Float(recentHandshots.count) / Float(uptime) : 0
        stats = GestureRecognizingStats(
            totalGesturesDetected: detectedGestures.count,
            averageProcessingLatency: avgLatency,
            averageConfidence: avgConfidence,
            gesturesByType: gesturesByType,
            uptime: uptime,
            fps: fps
        )
    }

    private func resetStats() {
        stats = GestureRecognizingStats()
        detectedGestures.removeAll()
        processingTimes.removeAll()
        confidenceScores.removeAll()
        recentHandshots.removeAll()
    }
}

// MARK: - Convenience Extensions

extension HandGestureRecognizing {

    public func quickStart() async throws {
        try await initialize()
        try await start()
    }

    public var isReady: Bool { isInitialized && gestureModel.isLoaded }
    public var isActive: Bool { isRunning && currentStatus.isActive }
}

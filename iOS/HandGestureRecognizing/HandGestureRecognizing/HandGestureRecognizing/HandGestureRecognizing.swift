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
    
    // Gesture buffering for real-time processing
    private var recentHandshots: [HandShot] = []
    private let handshotQueue = DispatchQueue(label: "com.cameragestures.handshot", qos: .userInteractive)
    
    // Callbacks
    public var gestureDetectionCallback: GestureDetectionCallback?
    public var handTrackingUpdateCallback: HandTrackingUpdateCallback?
    public var statusChangeCallback: StatusChangeCallback?
    
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
            // Initialize hands recognizing
            try handsRecognizer.initialize(config: self.config.handsRecognizingConfig)
            
            // Set up hands recognizing callbacks
            setupHandsRecognizingCallbacks()
            
            // Initialize gesture model
            try gestureModel.initialize(config: self.config.gestureModelConfig)
            
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
        
        // Check camera permission
        let hasPermission = await HandsRecognizing.requestCameraPermission()
        guard hasPermission else {
            throw HandGestureRecognizingError.cameraPermissionDenied
        }
        
        do {
            currentStatus = .initializing
            
            // Start hands recognizing
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
    
    /// Get current system status
    public func getStatus() -> GestureRecognizingStatus {
        return currentStatus
    }
    
    /// Get current statistics
    public func getStatistics() -> GestureRecognizingStats {
        updateStatistics()
        return stats
    }
    
    /// Get recent detected gestures
    public func getRecentGestures(limit: Int = 10) -> [DetectedGesture] {
        return Array(detectedGestures.suffix(limit))
    }
    
    /// Clear gesture history
    public func clearHistory() {
        detectedGestures.removeAll()
        processingTimes.removeAll()
        confidenceScores.removeAll()
        resetStats()
    }
    
    // MARK: - Configuration Updates
    
    /// Update configuration while running
    public func updateConfig(_ newConfig: HandGestureRecognizingConfig) throws {
        let wasRunning = isRunning
        
        if wasRunning {
            stop()
        }
        
        self.config = newConfig
        
        // Re-initialize with new config
        try handsRecognizer.initialize(config: newConfig.handsRecognizingConfig)
        try gestureModel.initialize(config: newConfig.gestureModelConfig)
        
        if wasRunning {
            Task {
                try await start()
            }
        }
    }
    
    /// Get current configuration
    public func getConfig() -> HandGestureRecognizingConfig {
        return config
    }
    
    // MARK: - Private Methods
    
    private func setupHandsRecognizingCallbacks() {
        // Handle individual handshots
        handsRecognizer.handshotCallback = { [weak self] handshot in
            self?.handleHandshot(handshot)
        }
        
        // Handle completed handfilms
        handsRecognizer.handfilmCallback = { [weak self] handfilm in
            self?.handleHandfilm(handfilm)
        }
    }
    
    private func handleHandshot(_ handshot: HandShot) {
        handshotQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Add to recent handshots buffer
            self.recentHandshots.append(handshot)
            
            // Maintain buffer size
            if self.recentHandshots.count > self.config.gestureBufferSize {
                self.recentHandshots.removeFirst()
            }
            
            // Notify callback
            DispatchQueue.main.async {
                self.handTrackingUpdateCallback?(handshot)
            }
            
            // Perform real-time gesture recognition if enabled
            if self.config.enableRealTimeProcessing && self.recentHandshots.count >= 3 {
                self.performRealTimeGestureRecognition()
            }
        }
    }
    
    private func handleHandfilm(_ handfilm: HandFilm) {
        Task { [weak self] in
            await self?.performGestureRecognition(on: handfilm)
        }
    }
    
    private func performRealTimeGestureRecognition() {
        guard gestureModel.isLoaded else { return }
        
        Task { [weak self] in
            guard let self = self else { return }
            
            do {
                let predictions = try await self.gestureModel.predictStreaming(handshots: self.recentHandshots)
                
                // Process predictions above threshold
                for prediction in predictions {
                    if prediction.confidence >= self.config.confidenceThreshold {
                        let detectedGesture = DetectedGesture(
                            prediction: prediction,
                            handfilm: HandFilm(), // Empty for real-time
                            handedness: self.recentHandshots.last?.leftOrRight ?? .unknown,
                            detectionTimestamp: Date().timeIntervalSince1970,
                            processingLatency: 0.0 // Would be calculated in real implementation
                        )
                        
                        await self.processDetectedGesture(detectedGesture)
                    }
                }
                
            } catch {
                // Handle error silently for real-time processing
                print("Real-time gesture recognition error: \(error)")
            }
        }
    }
    
    private func performGestureRecognition(on handfilm: HandFilm) async {
        let startTime = Date().timeIntervalSince1970
        
        do {
            // Get gesture predictions
            let predictions = try await gestureModel.predictTopK(handfilm: handfilm, k: 3)
            
            // Process each prediction above threshold
            for prediction in predictions {
                if prediction.confidence >= config.confidenceThreshold {
                    let processingLatency = Date().timeIntervalSince1970 - startTime
                    
                    let detectedGesture = DetectedGesture(
                        prediction: prediction,
                        handfilm: handfilm,
                        handedness: handfilm.frames.first?.leftOrRight ?? .unknown,
                        detectionTimestamp: Date().timeIntervalSince1970,
                        processingLatency: processingLatency
                    )
                    
                    await processDetectedGesture(detectedGesture)
                    break // Only process the highest confidence gesture
                }
            }
            
        } catch {
            print("Gesture recognition error: \(error)")
        }
    }
    
    @MainActor
    private func processDetectedGesture(_ gesture: DetectedGesture) {
        // Add to history
        detectedGestures.append(gesture)
        
        // Track statistics
        processingTimes.append(gesture.processingLatency)
        confidenceScores.append(gesture.prediction.confidence)
        
        // Maintain history size
        if detectedGestures.count > 1000 {
            detectedGestures.removeFirst(detectedGestures.count - 1000)
        }
        
        // Notify callback
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
    
    /// Quick start with default configuration
    public func quickStart() async throws {
        try await initialize()
        try await start()
    }
    
    /// Check if system is ready for processing
    public var isReady: Bool {
        return isInitialized && gestureModel.isLoaded
    }
    
    /// Check if system is actively running
    public var isActive: Bool {
        return isRunning && currentStatus.isActive
    }
}

import Foundation
import AVFoundation
import UIKit
import HandGestureTypes
import MediaPipeTasksVision

/// Hand tracking and gesture recognition module (stub implementation)
public class HandsRecognizing: NSObject {
    
    // MARK: - Properties
    
    private var config: HandsRecognizingConfig
    private var isRunning = false
    private var currentHandfilm = HandFilm()
    
    // Callbacks
    public var handshotCallback: HandShotCallback?
    public var handfilmCallback: HandFilmCallback?
    
    // MediaPipe components
    private var landmarker: HandLandmarker?
    
    // Camera components
    private var captureSession: AVCaptureSession?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var processingQueue: DispatchQueue?
    
    // MARK: - Initialization
    
    public override init() {
        self.config = .init(detectBothHands: false)
        super.init()
    }
    
    // MARK: - Configuration
    
    /// Initialize with configuration
    public func initialize(config: HandsRecognizingConfig) throws {
        self.config = config
        
        // Stub: Just validate basic parameters
        guard config.targetFPS > 0 && config.targetFPS <= 120 else {
            throw HandsRecognizingError.invalidConfiguration
        }
        
        guard config.minDetectionConfidence >= 0.0 && config.minDetectionConfidence <= 1.0 else {
            throw HandsRecognizingError.invalidConfiguration
        }
        
        guard config.minTrackingConfidence >= 0.0 && config.minTrackingConfidence <= 1.0 else {
            throw HandsRecognizingError.invalidConfiguration
        }
        
        // TODO: Investigate custom MediaPipe model for better accuracy
        var options = config.getHandLandmarkerOptions()
        
        // Set up live stream callback
        options.handLandmarkerLiveStreamDelegate = self
        
        do {
            landmarker = try HandLandmarker(options: options)
        }
        catch {
            print("HandLandmarker init error: \(error)")
        }
        
        // Set up camera session
        try setupCameraSession()
    }
    
    // MARK: - Lifecycle
    
    /// Start hand tracking
    public func start() throws {
        guard !isRunning else { return }
        
        // Distinguish between "not yet asked" and "denied/restricted"
        // .notDetermined means the system prompt hasn't appeared yet — the session
        // will silently produce no frames rather than throwing, so surface it explicitly.
        let cameraStatus = AVCaptureDevice.authorizationStatus(for: .video)
        switch cameraStatus {
        case .authorized:
            break
        case .notDetermined:
            throw HandsRecognizingError.cameraPermissionNotDetermined
        default:
            throw HandsRecognizingError.cameraNotAvailable
        }
        
        try startCameraCapture()
    }
    
    /// Stop hand tracking
    public func stop() {
        isRunning = false
        stopCameraCapture()
        
        // Complete any pending handfilm
        if !currentHandfilm.frames.isEmpty {
            handfilmCallback?(currentHandfilm)
            currentHandfilm.clear()
        }
    }
    
    // MARK: - Status
    
    /// Check if tracking is currently running
    public var isTracking: Bool {
        return isRunning
    }
    
    /// Get current configuration
    public func getConfig() -> HandsRecognizingConfig {
        return config
    }
    
    // MARK: - Private Methods
    
    private func setupCameraSession() throws {
        captureSession = AVCaptureSession()
        guard let captureSession = captureSession else {
            throw HandsRecognizingError.initializationFailed
        }
        
        // Configure session
        captureSession.beginConfiguration()
        
        // TODO: check the difference between different levels of sessionPreset
        captureSession.sessionPreset = .medium
        
        // Add camera input
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            throw HandsRecognizingError.cameraNotAvailable
        }
        
        let cameraInput = try AVCaptureDeviceInput(device: camera)
        guard captureSession.canAddInput(cameraInput) else {
            throw HandsRecognizingError.cameraNotAvailable
        }
        captureSession.addInput(cameraInput)
        
        // Add video output — configure settings before adding to session,
        // and attach the delegate only after addOutput so the session owns the output first
        videoOutput = AVCaptureVideoDataOutput()
        guard let videoOutput = videoOutput else {
            throw HandsRecognizingError.initializationFailed
        }
        
        videoOutput.alwaysDiscardsLateVideoFrames = false
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        
        guard captureSession.canAddOutput(videoOutput) else {
            throw HandsRecognizingError.initializationFailed
        }
        captureSession.addOutput(videoOutput)
        
        // Set the delegate after the output is added to the session
        processingQueue = DispatchQueue(label: "com.cameragestures.processing", qos: .userInitiated)
        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        
        captureSession.commitConfiguration()
    }
    
    private func startCameraCapture() throws {
        guard let captureSession = captureSession else {
            throw HandsRecognizingError.initializationFailed
        }
        
        isRunning = true
        DispatchQueue.global(qos: .userInitiated).async {
            captureSession.startRunning()
        }
    }
    
    private func stopCameraCapture() {
        captureSession?.stopRunning()
    }
    
    private func createPixelBuffer(from data: Data, width: Int, height: Int, channels: Int) throws -> CVPixelBuffer {
        let bytesPerRow = width * channels
        var pixelBuffer: CVPixelBuffer?
        
        let status = CVPixelBufferCreateWithBytes(
            nil,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            UnsafeMutableRawPointer(mutating: data.withUnsafeBytes { $0.bindMemory(to: UInt8.self).baseAddress! }),
            bytesPerRow,
            nil,
            nil,
            nil,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw HandsRecognizingError.processingError
        }
        
        return buffer
    }
    
    private func convertMediaPipeResults(_ result: HandLandmarkerResult, timestamp: TimeInterval) {
        // Process each detected hand
        for (handIndex, landmarks) in result.landmarks.enumerated() {
            // Convert MediaPipe landmarks to our Point3D format
            let convertedLandmarks = landmarks.map { landmark in
                Point3D(
                    x: landmark.x,
                    y: landmark.y,
                    z: landmark.z ?? 0.0 // MediaPipe z might be nil
                )
            }
            
            // Determine hand side (left/right)
            let handedness: LeftOrRight
            if handIndex < result.handedness.count,
               let firstHandedness = result.handedness[handIndex].first, let categoryName = firstHandedness.categoryName {
                handedness = categoryName.lowercased() == "left" ? .left : .right
            } else {
                handedness = .right // Default fallback
            }
            
            let handshot = HandShot(
                landmarks: convertedLandmarks,
                timestamp: timestamp,
                leftOrRight: handedness
            )
            
            processHandshot(handshot)
        }
    }
    
    private func processHandshot(_ handshot: HandShot) {
        // Call handshot callback
        handshotCallback?(handshot)
        
        // Add to current handfilm
        currentHandfilm.addFrame(handshot)
        
        // Check if handfilm is complete
        if currentHandfilm.duration >= config.handfilmMaxDuration {
            handfilmCallback?(currentHandfilm)
            currentHandfilm.clear()
        }
    }
    
    // MARK: - Camera Utilities
    
    /// Request camera permission
    public static func requestCameraPermission() async -> Bool {
        return await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .video) { granted in
                continuation.resume(returning: granted)
            }
        }
    }
    
    /// Check if camera is available
    public static func isCameraAvailable() -> Bool {
        return !AVCaptureDevice.devices(for: .video).isEmpty
    }
    
    /// Get available cameras
    public static func getAvailableCameras() -> [AVCaptureDevice] {
        return AVCaptureDevice.devices(for: .video)
    }
}

// MARK: - Extensions

extension HandsRecognizing {
    
    /// Convenience method to start with default config
    public func startWithDefaultConfig() throws {
        try initialize(config: .defaultConfig)
        try start()
    }
    
    /// Get current handfilm (for debugging)
    public func getCurrentHandfilm() -> HandFilm {
        return currentHandfilm
    }
}

// MARK: - HandLandmarkerLiveStreamDelegate

extension HandsRecognizing: HandLandmarkerLiveStreamDelegate {
    public func handLandmarker(_ handLandmarker: HandLandmarker, didFinishDetection result: HandLandmarkerResult?, timestampInMilliseconds: Int, error: Error?) {
        // Handle MediaPipe errors
        if let error = error {
            print("MediaPipe detection error: \(error)")
            return
        }
        
        // Process results if available
        if let result = result {
            let timestamp = TimeInterval(timestampInMilliseconds) / 1000.0
            convertMediaPipeResults(result, timestamp: timestamp)
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension HandsRecognizing: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard isRunning,
              let landmarker = landmarker,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        do {
            // Convert to MPImage
            let mpImage = try MPImage(pixelBuffer: pixelBuffer)
            let timestamp = Int(Date().timeIntervalSince1970 * 1000)
            
            // Process with MediaPipe
            try landmarker.detectAsync(image: mpImage, timestampInMilliseconds: timestamp)
        } catch {
            print("Frame processing error: \(error)")
        }
    }
}

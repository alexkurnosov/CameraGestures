// Stage 3: HandsRecognizing Swift wrapper.
// Public API is identical to the V1 HandsRecognizingModule so Training App v2
// compiles without source changes after the pod swap.
//
// Differences from V1:
// - Film buffer is managed by cg_hands_recognizer_ref (C ABI in the XCFramework).
// - harvestHandfilm() / resetHandfilm() delegate to the C layer.
// - No other behavioral changes.

import Foundation
import AVFoundation
import UIKit
import MediaPipeTasksVision
import CameraGesturesC

// MARK: - Error / Config / Callback types (mirror V1 Types.swift)

public enum HandsRecognizingError: Error {
    case cameraNotAvailable
    case cameraPermissionNotDetermined
    case invalidConfiguration
    case initializationFailed
    case processingError

    public var localizedDescription: String {
        switch self {
        case .cameraNotAvailable:            return "Camera not available or permission denied"
        case .cameraPermissionNotDetermined: return "Camera permission not yet requested — call requestCameraPermission() first"
        case .invalidConfiguration:          return "Invalid configuration"
        case .initializationFailed:          return "Initialization failed"
        case .processingError:               return "Processing error"
        }
    }
}

public struct HandsRecognizingConfig {
    public let cameraIndex:              Int
    public let targetFPS:                Int
    public let detectBothHands:          Bool
    public let minDetectionConfidence:   Float
    public let minTrackingConfidence:    Float
    public let handfilmMaxDuration:      TimeInterval

    public init(
        cameraIndex:            Int          = 0,
        targetFPS:              Int          = 30,
        detectBothHands:        Bool         = true,
        minDetectionConfidence: Float        = 0.5,
        minTrackingConfidence:  Float        = 0.5,
        handfilmMaxDuration:    TimeInterval = 2.0
    ) {
        self.cameraIndex            = cameraIndex
        self.targetFPS              = targetFPS
        self.detectBothHands        = detectBothHands
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence  = minTrackingConfidence
        self.handfilmMaxDuration    = handfilmMaxDuration
    }

    public static let defaultConfig = HandsRecognizingConfig()

    public func getHandLandmarkerOptions() -> HandLandmarkerOptions {
        let options = HandLandmarkerOptions()
        let baseOptions = BaseOptions()
        let frameworkBundle = Bundle(for: HandsRecognizing.self)
        if let modelPath = frameworkBundle.path(forResource: "hand_landmarker", ofType: "task") {
            baseOptions.modelAssetPath = modelPath
        }
        options.baseOptions          = baseOptions
        options.numHands             = detectBothHands ? 2 : 1
        options.minHandDetectionConfidence = minDetectionConfidence
        options.minTrackingConfidence       = minTrackingConfidence
        options.minHandPresenceConfidence   = minDetectionConfidence
        options.runningMode                 = .liveStream
        return options
    }
}

public typealias HandShotCallback  = (HandShot) -> Void
public typealias HandFilmCallback  = (HandFilm) -> Void

// MARK: - HandsRecognizing

public class HandsRecognizing: NSObject {

    // MARK: Properties

    private var config: HandsRecognizingConfig
    private var isRunning = false

    // C++ film buffer
    private let recognizerRef: cg_hands_recognizer_ref

    // Callbacks
    public var handshotCallback: HandShotCallback?
    public var handfilmCallback: HandFilmCallback?

    // MediaPipe
    private var landmarker: HandLandmarker?

    // Camera
    private var captureSession:  AVCaptureSession?
    private var videoOutput:     AVCaptureVideoDataOutput?
    private var processingQueue: DispatchQueue?

    // MARK: Initialization

    public override init() {
        self.config        = .init(detectBothHands: false)
        self.recognizerRef = cg_hands_recognizer_create()
        super.init()
    }

    deinit {
        cg_hands_recognizer_destroy(recognizerRef)
    }

    // MARK: Configuration

    public func initialize(config: HandsRecognizingConfig) throws {
        self.config = config

        guard config.targetFPS > 0 && config.targetFPS <= 120 else {
            throw HandsRecognizingError.invalidConfiguration
        }
        guard config.minDetectionConfidence >= 0.0 && config.minDetectionConfidence <= 1.0 else {
            throw HandsRecognizingError.invalidConfiguration
        }
        guard config.minTrackingConfidence >= 0.0 && config.minTrackingConfidence <= 1.0 else {
            throw HandsRecognizingError.invalidConfiguration
        }

        var options = config.getHandLandmarkerOptions()
        options.handLandmarkerLiveStreamDelegate = self

        do {
            landmarker = try HandLandmarker(options: options)
        } catch {
            print("HandLandmarker init error: \(error)")
        }

        try setupCameraSession()
    }

    // MARK: Lifecycle

    public func start() throws {
        guard !isRunning else { return }

        let cameraStatus = AVCaptureDevice.authorizationStatus(for: .video)
        switch cameraStatus {
        case .authorized:         break
        case .notDetermined:      throw HandsRecognizingError.cameraPermissionNotDetermined
        default:                  throw HandsRecognizingError.cameraNotAvailable
        }

        cg_hands_recognizer_reset(recognizerRef)
        try startCameraCapture()
    }

    public func stop() {
        isRunning = false
        stopCameraCapture()
        cg_hands_recognizer_reset(recognizerRef)
    }

    public var isTracking: Bool { isRunning }
    public func getConfig() -> HandsRecognizingConfig { config }

    // MARK: Camera utilities

    public static func requestCameraPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .video) { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    public static func isCameraAvailable() -> Bool {
        !AVCaptureDevice.devices(for: .video).isEmpty
    }

    public static func getAvailableCameras() -> [AVCaptureDevice] {
        AVCaptureDevice.devices(for: .video)
    }

    // MARK: Private — camera setup

    private func setupCameraSession() throws {
        captureSession = AVCaptureSession()
        guard let captureSession else { throw HandsRecognizingError.initializationFailed }

        captureSession.beginConfiguration()
        captureSession.sessionPreset = .medium

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            throw HandsRecognizingError.cameraNotAvailable
        }

        let cameraInput = try AVCaptureDeviceInput(device: camera)
        guard captureSession.canAddInput(cameraInput) else {
            throw HandsRecognizingError.cameraNotAvailable
        }
        captureSession.addInput(cameraInput)

        videoOutput = AVCaptureVideoDataOutput()
        guard let videoOutput else { throw HandsRecognizingError.initializationFailed }

        videoOutput.alwaysDiscardsLateVideoFrames = false
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]

        guard captureSession.canAddOutput(videoOutput) else {
            throw HandsRecognizingError.initializationFailed
        }
        captureSession.addOutput(videoOutput)

        processingQueue = DispatchQueue(label: "com.cameragestures.processing", qos: .userInitiated)
        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)

        captureSession.commitConfiguration()
    }

    private func startCameraCapture() throws {
        guard let captureSession else { throw HandsRecognizingError.initializationFailed }
        isRunning = true
        DispatchQueue.global(qos: .userInitiated).async { captureSession.startRunning() }
    }

    private func stopCameraCapture() {
        captureSession?.stopRunning()
    }

    // MARK: Private — MediaPipe result conversion

    private func convertMediaPipeResults(_ result: HandLandmarkerResult, timestamp: TimeInterval) {
        if result.landmarks.isEmpty {
            pushAbsentIfActive(timestamp: timestamp)
            return
        }

        for (handIndex, landmarks) in result.landmarks.enumerated() {
            let pts = landmarks.map { Point3D(x: $0.x, y: $0.y, z: $0.z ?? 0.0) }

            // MediaPipe occasionally returns an all-zero detection instead of an
            // empty result — treat as absent to avoid corrupting hold detection.
            let wrist = pts[0], lm9 = pts[9]
            let dx = lm9.x - wrist.x, dy = lm9.y - wrist.y, dz = lm9.z - wrist.z
            if dx*dx + dy*dy + dz*dz < 1e-12 {
                pushAbsentIfActive(timestamp: timestamp)
                continue
            }

            let handedness: LeftOrRight
            if handIndex < result.handedness.count,
               let first = result.handedness[handIndex].first,
               let name  = first.categoryName {
                handedness = name.lowercased() == "left" ? .left : .right
            } else {
                handedness = .right
            }

            let shot = HandShot(landmarks: pts, timestamp: timestamp, leftOrRight: handedness)
            pushHandshot(shot)
        }
    }

    private func pushAbsentIfActive(timestamp: TimeInterval) {
        let absent = HandShot.absent(timestamp: timestamp)
        // cg_hands_recognizer silently drops absent shots with no prior real frame
        let cShot = absent.toCStruct()
        cg_hands_recognizer_push_handshot(recognizerRef, [cShot])
        handshotCallback?(absent)
    }

    private func pushHandshot(_ shot: HandShot) {
        var cShot = shot.toCStruct()
        cg_hands_recognizer_push_handshot(recognizerRef, &cShot)
        handshotCallback?(shot)
    }
}

// MARK: - Convenience extensions

extension HandsRecognizing {

    public func startWithDefaultConfig() throws {
        try initialize(config: .defaultConfig)
        try start()
    }

    public func getCurrentHandfilm() -> HandFilm {
        // Peek without harvesting: harvest + push back is not ideal;
        // for debug use only. Use harvestHandfilm() for production.
        let ref = cg_hands_recognizer_harvest(recognizerRef)
        defer { cg_handfilm_destroy(ref) }
        return HandFilm(fromCRef: ref)
    }

    public func resetHandfilm() {
        cg_hands_recognizer_reset(recognizerRef)
    }

    public func harvestHandfilm() -> HandFilm {
        let ref = cg_hands_recognizer_harvest(recognizerRef)
        defer { cg_handfilm_destroy(ref) }
        return HandFilm(fromCRef: ref)
    }
}

// MARK: - HandLandmarkerLiveStreamDelegate

extension HandsRecognizing: HandLandmarkerLiveStreamDelegate {
    public func handLandmarker(
        _ handLandmarker: HandLandmarker,
        didFinishDetection result: HandLandmarkerResult?,
        timestampInMilliseconds: Int,
        error: Error?)
    {
        if let error {
            print("MediaPipe detection error: \(error)")
            return
        }
        if let result {
            let ts = TimeInterval(timestampInMilliseconds) / 1000.0
            convertMediaPipeResults(result, timestamp: ts)
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension HandsRecognizing: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection)
    {
        guard isRunning,
              let landmarker,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        do {
            let mpImage   = try MPImage(pixelBuffer: pixelBuffer)
            let timestamp = Int(Date().timeIntervalSince1970 * 1000)
            try landmarker.detectAsync(image: mpImage, timestampInMilliseconds: timestamp)
        } catch {
            print("Frame processing error: \(error)")
        }
    }
}

// MARK: - HandShot ↔ cg_handshot conversion

private extension HandShot {
    func toCStruct() -> cg_handshot {
        var c       = cg_handshot()
        c.timestamp = self.timestamp
        c.handedness = self.leftOrRight == .left  ? CG_HAND_LEFT  :
                       self.leftOrRight == .right ? CG_HAND_RIGHT : CG_HAND_UNKNOWN
        c.is_absent  = self.isAbsent ? 1 : 0
        withUnsafeMutableBytes(of: &c.landmarks) { buf in
            let typed = buf.bindMemory(to: cg_point3d.self)
            for (i, pt) in self.landmarks.prefix(21).enumerated() {
                typed[i] = cg_point3d(x: pt.x, y: pt.y, z: pt.z)
            }
        }
        return c
    }
}

// MARK: - HandFilm from cg_handfilm_ref

private extension HandFilm {
    init(fromCRef ref: cg_handfilm_ref?) {
        guard let ref else { self.init(); return }
        let startTime = cg_handfilm_start_time(ref)
        self.init(startTime: startTime)
        let count = cg_handfilm_shot_count(ref)
        for i in 0..<count {
            var cShot = cg_handshot()
            guard cg_handfilm_get_shot(ref, i, &cShot) != 0 else { continue }
            addFrame(HandShot(fromCStruct: cShot))
        }
    }
}

private extension HandShot {
    init(fromCStruct c: cg_handshot) {
        var pts: [Point3D] = []
        pts.reserveCapacity(21)
        withUnsafeBytes(of: c.landmarks) { buf in
            let typed = buf.bindMemory(to: cg_point3d.self)
            for pt in typed { pts.append(Point3D(x: pt.x, y: pt.y, z: pt.z)) }
        }
        self.init(
            landmarks:    pts,
            timestamp:    c.timestamp,
            leftOrRight:  c.handedness == CG_HAND_LEFT ? .left :
                          c.handedness == CG_HAND_RIGHT ? .right : .unknown,
            isAbsent:     c.is_absent != 0
        )
    }
}

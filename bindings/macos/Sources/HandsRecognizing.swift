// macOS HandsRecognizing — Vision-based (no TFLite / no MediaPipe dependency).
//
// Hand landmark detection is performed by VisionHandLandmarker, which wraps
// Apple's VNDetectHumanHandPoseRequest. This replaces the broken two-stage
// TFLite pipeline (palm_detection_full + hand_landmarks_detector).
//
// Everything downstream of cg_hands_recognizer_push_handshot — the motion gate,
// gesture model, and HandGestureRecognizing — is unchanged.

import Foundation
import AVFoundation
import CameraGesturesC

// MARK: - Error / Config / Callback types

public enum HandsRecognizingError: Error {
    case cameraNotAvailable
    case cameraPermissionNotDetermined
    case invalidConfiguration
    case initializationFailed
    case modelLoadFailed(String)

    public var localizedDescription: String {
        switch self {
        case .cameraNotAvailable:            return "Camera not available or permission denied"
        case .cameraPermissionNotDetermined: return "Camera permission not yet determined"
        case .invalidConfiguration:          return "Invalid configuration"
        case .initializationFailed:          return "Initialization failed"
        case .modelLoadFailed(let m):        return "Model load failed: \(m)"
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
        detectBothHands:        Bool         = false,
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
}

public typealias HandShotCallback = (HandShot) -> Void
public typealias HandFilmCallback = (HandFilm) -> Void

// MARK: - HandsRecognizing

public class HandsRecognizing: NSObject {

    // MARK: Properties

    private var config: HandsRecognizingConfig = .defaultConfig
    private var isRunning = false

    private let recognizerRef: cg_hands_recognizer_ref
    private var visionLandmarker: VisionHandLandmarker?

    public var handshotCallback: HandShotCallback?
    public var handfilmCallback: HandFilmCallback?

    private var captureSession:  AVCaptureSession?
    private var videoOutput:     AVCaptureVideoDataOutput?

    // MARK: Init

    public override init() {
        self.recognizerRef = cg_hands_recognizer_create()
        super.init()
    }

    deinit {
        cg_hands_recognizer_destroy(recognizerRef)
    }

    // MARK: Configuration

    public func initialize(config: HandsRecognizingConfig = .defaultConfig) throws {
        self.config = config

        // Create Vision-based landmarker. It will push handshots into recognizerRef.
        let lm = VisionHandLandmarker(recognizerRef: recognizerRef)
        lm.handshotCallback = { [weak self] shot in
            self?.handshotCallback?(shot)
        }
        visionLandmarker = lm

        try setupCameraSession()
    }

    // MARK: Lifecycle

    public func start() throws {
        guard !isRunning else { return }
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized: break
        case .notDetermined: throw HandsRecognizingError.cameraPermissionNotDetermined
        default: throw HandsRecognizingError.cameraNotAvailable
        }
        cg_hands_recognizer_reset(recognizerRef)
        guard let session = captureSession else { throw HandsRecognizingError.initializationFailed }
        isRunning = true
        DispatchQueue.global(qos: .userInitiated).async { session.startRunning() }
    }

    public func stop() {
        isRunning = false
        captureSession?.stopRunning()
        cg_hands_recognizer_reset(recognizerRef)
    }

    public var isTracking: Bool { isRunning }
    public func getConfig() -> HandsRecognizingConfig { config }

    /// The underlying AVCaptureSession — pass to CameraPreviewView for live video display.
    public var previewSession: AVCaptureSession? { captureSession }

    // MARK: Camera permission

    public static func requestCameraPermission() async -> Bool {
        await withCheckedContinuation { cont in
            AVCaptureDevice.requestAccess(for: .video) { cont.resume(returning: $0) }
        }
    }

    public func resetHandfilm() { cg_hands_recognizer_reset(recognizerRef) }

    public func harvestHandfilm() -> HandFilm {
        let ref = cg_hands_recognizer_harvest(recognizerRef)
        defer { cg_handfilm_destroy(ref) }
        return HandFilm(fromCRef: ref)
    }

    // MARK: Private — camera setup

    private func setupCameraSession() throws {
        let session = AVCaptureSession()
        session.beginConfiguration()
        session.sessionPreset = .medium

        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: .front)
        guard let camera = discovery.devices.first
                           ?? AVCaptureDevice.default(for: .video) else {
            throw HandsRecognizingError.cameraNotAvailable
        }

        let input = try AVCaptureDeviceInput(device: camera)
        guard session.canAddInput(input) else { throw HandsRecognizingError.cameraNotAvailable }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        // Vision accepts kCVPixelFormatType_32BGRA natively.
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        guard session.canAddOutput(output) else { throw HandsRecognizingError.initializationFailed }
        session.addOutput(output)

        let queue = DispatchQueue(label: "com.cameragestures.macos.processing", qos: .userInitiated)
        output.setSampleBufferDelegate(self, queue: queue)

        session.commitConfiguration()
        captureSession = session
        videoOutput    = output
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension HandsRecognizing: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection)
    {
        guard isRunning, let lm = visionLandmarker else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ts = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        lm.process(pixelBuffer, timestamp: ts)
    }
}

// MARK: - HandFilm from cg_handfilm_ref

private extension HandFilm {
    init(fromCRef ref: cg_handfilm_ref?) {
        guard let ref else { self.init(); return }
        self.init(startTime: cg_handfilm_start_time(ref))
        let count = cg_handfilm_shot_count(ref)
        for i in 0..<count {
            var cShot = cg_handshot()
            guard cg_handfilm_get_shot(ref, i, &cShot) != 0 else { continue }
            addFrame(HandShot(fromCStruct: cShot))
        }
    }
}

// macOS HandsRecognizing — LiteRT-only (no MediaPipe dependency).
//
// Differences from the iOS version:
// - No MediaPipeTasksVision import.
// - Landmark detection runs via CG_BUILD_MACOS_LANDMARKER C ABI
//   (cg_hand_landmarker_lrt_*), which calls our LiteRT two-stage decoder.
// - AVCaptureSession setup and BGRA8 pipeline are identical to iOS.

import Foundation
import AVFoundation
import CameraGesturesC

// MARK: - Error / Config / Callback types (identical to iOS)

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
        case .modelLoadFailed(let m):        return "LiteRT model load failed: \(m)"
        }
    }
}

public struct HandsRecognizingConfig {
    public let targetFPS:                Int
    public let detectBothHands:          Bool
    public let minDetectionConfidence:   Float
    public let minTrackingConfidence:    Float
    // Path to hand_landmarker.task; if nil, the bundle is looked up from the
    // framework bundle and then the main bundle.
    public let taskBundlePath:           String?

    public init(
        targetFPS:              Int    = 30,
        detectBothHands:        Bool   = false,
        minDetectionConfidence: Float  = 0.5,
        minTrackingConfidence:  Float  = 0.5,
        taskBundlePath:         String? = nil
    ) {
        self.targetFPS              = targetFPS
        self.detectBothHands        = detectBothHands
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence  = minTrackingConfidence
        self.taskBundlePath         = taskBundlePath
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
    private var landmarkerRef: cg_hand_landmarker_lrt_ref?

    public var handshotCallback: HandShotCallback?
    public var handfilmCallback: HandFilmCallback?

    private var captureSession:  AVCaptureSession?
    private var videoOutput:     AVCaptureVideoDataOutput?
    private var processingQueue: DispatchQueue?

    // MARK: Init

    public override init() {
        self.recognizerRef = cg_hands_recognizer_create()
        super.init()
        wireHandshotCallback()
    }

    deinit {
        if let lm = landmarkerRef { cg_hand_landmarker_lrt_destroy(lm) }
        cg_hands_recognizer_destroy(recognizerRef)
    }

    // MARK: Configuration

    public func initialize(config: HandsRecognizingConfig = .defaultConfig) throws {
        self.config = config

        // Resolve hand_landmarker.task path
        let taskPath = config.taskBundlePath ?? resolveTaskBundlePath()
        guard let taskPath else {
            throw HandsRecognizingError.modelLoadFailed("hand_landmarker.task not found in bundle")
        }

        // Create LiteRT landmarker (loads both TFLite models from the .task bundle)
        if let old = landmarkerRef { cg_hand_landmarker_lrt_destroy(old); landmarkerRef = nil }
        landmarkerRef = cg_hand_landmarker_lrt_create(taskPath, recognizerRef)
        guard landmarkerRef != nil else {
            throw HandsRecognizingError.modelLoadFailed("cg_hand_landmarker_lrt_create failed for: \(taskPath)")
        }

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

        // On macOS, prefer a built-in camera
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
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        guard session.canAddOutput(output) else { throw HandsRecognizingError.initializationFailed }
        session.addOutput(output)

        let queue = DispatchQueue(label: "com.cameragestures.macos.processing", qos: .userInitiated)
        output.setSampleBufferDelegate(self, queue: queue)

        session.commitConfiguration()
        captureSession  = session
        videoOutput     = output
        processingQueue = queue
    }

    // MARK: Private — handshot callback wiring

    private func wireHandshotCallback() {
        // The C++ recognizer pushes handshots from the LiteRT landmarker.
        // We forward them to the Swift callback.
        let selfPtr = Unmanaged.passRetained(self)
        cg_hands_recognizer_set_callback(recognizerRef, { shotPtr, ctx in
            guard let ctx, let shotPtr else { return }
            let hr = Unmanaged<HandsRecognizing>.fromOpaque(ctx).takeUnretainedValue()
            let shot = HandShot(fromCStruct: shotPtr.pointee)
            hr.handshotCallback?(shot)
        }, selfPtr.toOpaque())
        selfPtr.release()
    }

    // MARK: Private — task bundle path resolution

    private func resolveTaskBundlePath() -> String? {
        // Check the CameraGesturesMac framework bundle, then main bundle.
        let bundles: [Bundle] = [Bundle(for: HandsRecognizing.self), .main]
        for bundle in bundles {
            // Check inside a CameraGesturesAssets.bundle resource bundle
            if let assetURL = bundle.url(forResource: "CameraGesturesAssets", withExtension: "bundle"),
               let assetBundle = Bundle(url: assetURL),
               let path = assetBundle.path(forResource: "hand_landmarker", ofType: "task") {
                return path
            }
            // Direct resource in the bundle
            if let path = bundle.path(forResource: "hand_landmarker", ofType: "task") {
                return path
            }
        }
        return nil
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension HandsRecognizing: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection)
    {
        guard isRunning, let lm = landmarkerRef else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
        let width  = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let stride = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let ts     = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))

        cg_hand_landmarker_lrt_push_frame(
            lm,
            base.assumingMemoryBound(to: UInt8.self),
            Int32(width), Int32(height), Int32(stride),
            ts)
    }
}

// MARK: - HandShot ↔ cg_handshot

private extension HandShot {
    init(fromCStruct c: cg_handshot) {
        var pts: [Point3D] = []
        pts.reserveCapacity(21)
        withUnsafeBytes(of: c.landmarks) { buf in
            let typed = buf.bindMemory(to: cg_point3d.self)
            for pt in typed { pts.append(Point3D(x: pt.x, y: pt.y, z: pt.z)) }
        }
        self.init(
            landmarks:   pts,
            timestamp:   c.timestamp,
            leftOrRight: c.handedness == CG_HAND_LEFT  ? .left
                       : c.handedness == CG_HAND_RIGHT ? .right : .unknown,
            isAbsent:    c.is_absent != 0)
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

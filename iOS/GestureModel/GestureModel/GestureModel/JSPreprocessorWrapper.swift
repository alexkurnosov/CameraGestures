import Foundation
import JavaScriptCore
import HandGestureTypes

// MARK: - Error

enum PreprocessorError: Error, LocalizedError {
    case notLoaded
    case loadFailed(String)

    var errorDescription: String? {
        switch self {
        case .notLoaded:
            return "Preprocessor JS not loaded. Call load(from:) first."
        case .loadFailed(let detail):
            return "Failed to load preprocessor.js: \(detail)"
        }
    }
}

// MARK: - Wrapper

/// Executes preprocessor.js via JavaScriptCore.
///
/// Usage:
///   1. Call `JSPreprocessorWrapper.shared.load(from:)` once after downloading the JS file.
///   2. Then `FeaturePreprocessor` delegates all calls here automatically.
public final class JSPreprocessorWrapper {

    public static let shared = JSPreprocessorWrapper()

    private let context = JSContext()!
    private(set) var isLoaded = false

    // JS-exported constants (read after load; fall back to v1 defaults before load)
    public private(set) var preprocVersion: Int = 1
    public private(set) var poseVectorSize: Int = 63
    public private(set) var featuresPerFrame: Int = 126
    public private(set) var summaryFeaturesCount: Int = 256

    private init() {
        context.exceptionHandler = { _, exception in
            print("[JSPreprocessorWrapper] JS exception: \(exception?.toString() ?? "unknown")")
        }
    }

    // MARK: - Loading

    /// Evaluate preprocessor.js from disk. Safe to call again to hot-reload after an update.
    /// Reads PREPROC_VERSION, POSE_VECTOR_SIZE, FEATURES_PER_FRAME from the JS context after loading.
    public func load(from url: URL) throws {
        let source = try String(contentsOf: url, encoding: .utf8)
        context.evaluateScript(source)
        if let exception = context.exception {
            throw PreprocessorError.loadFailed(exception.toString() ?? "unknown JS exception")
        }
        preprocVersion     = Int(context.evaluateScript("PREPROC_VERSION")?.toInt32() ?? 1)
        poseVectorSize     = Int(context.evaluateScript("POSE_VECTOR_SIZE")?.toInt32() ?? 63)
        featuresPerFrame   = Int(context.evaluateScript("FEATURES_PER_FRAME")?.toInt32() ?? 126)
        let coordsPerFrame = Int(context.evaluateScript("COORDS_PER_FRAME")?.toInt32() ?? 63)
        summaryFeaturesCount = poseVectorSize * 2 + coordsPerFrame * 2 + 4
        isLoaded = true
    }

    // MARK: - Public API

    /// Flat array of length 60 × featuresPerFrame.
    func featureMatrix(from handfilm: HandFilm) -> [Double] {
        callDoubleArray("featureMatrix", handfilm: handfilm)
    }

    /// Array of 60 rows, each of length featuresPerFrame.
    func featureRows(from handfilm: HandFilm) -> [[Double]] {
        guard isLoaded else { return [] }
        guard let result = context.evaluateScript("featureRows(\(handfilm.preprocessorJSON))"),
              result.isArray,
              let outer = result.toArray() else { return [] }
        return outer.compactMap { element -> [Double]? in
            guard let inner = element as? [Any] else { return nil }
            return inner.compactMap { ($0 as? NSNumber)?.doubleValue }
        }
    }

    /// summaryFeaturesCount-element Float vector used as Phase 3 MLP input.
    func summaryFeatures(from handfilm: HandFilm) -> [Float] {
        callDoubleArray("summaryFeatures", handfilm: handfilm).map { Float($0) }
    }

    /// poseVectorSize-element Float vector for a single frame (63 normalised coords + 20 geometric extras).
    /// Used as Phase 2 pose-MLP input.
    public func poseVector(from handshot: HandShot) -> [Float] {
        guard isLoaded else { return [] }
        let side = handshot.leftOrRight.jsString
        let frameJSON = handshot.frameJSON(leftOrRight: side)
        guard let result = context.evaluateScript("poseVector(\(frameJSON), \"\(side)\")"),
              result.isArray,
              let arr = result.toArray() else { return [] }
        return arr.compactMap { ($0 as? NSNumber)?.floatValue }
    }

    // MARK: - Private

    private func callDoubleArray(_ name: String, handfilm: HandFilm) -> [Double] {
        guard isLoaded else { return [] }
        guard let result = context.evaluateScript("\(name)(\(handfilm.preprocessorJSON))"),
              result.isArray,
              let arr = result.toArray() else { return [] }
        return arr.compactMap { ($0 as? NSNumber)?.doubleValue }
    }
}

// MARK: - HandFilm → JSON

private extension HandFilm {
    /// Serialise to the JSON format expected by preprocessor.js.
    var preprocessorJSON: String {
        var frameStrings: [String] = []
        frameStrings.reserveCapacity(frames.count)

        for frame in frames {
            frameStrings.append(frame.frameJSON(leftOrRight: frame.leftOrRight.jsString))
        }

        return "{\(q("frames")):[\(frameStrings.joined(separator: ","))],\(q("start_time")):\(startTime)}"
    }

    /// Wrap a string in JSON double-quotes.
    private func q(_ s: String) -> String { "\"\(s)\"" }
}

// MARK: - HandShot → single-frame JSON

extension HandShot {
    /// Serialise one frame to the format expected by poseVector() in preprocessor.js.
    fileprivate func frameJSON(leftOrRight: String) -> String {
        let lms = landmarks.map { lm in
            "{\"x\":\(lm.x),\"y\":\(lm.y),\"z\":\(lm.z)}"
        }.joined(separator: ",")
        return "{\"landmarks\":[\(lms)],\"timestamp\":\(timestamp),\"left_or_right\":\"\(leftOrRight)\",\"is_absent\":\(isAbsent ? "true" : "false")}"
    }
}

// MARK: - LeftOrRight → JS string

extension LeftOrRight {
    fileprivate var jsString: String {
        switch self {
        case .left:    return "left"
        case .right:   return "right"
        case .unknown: return "unknown"
        }
    }
}

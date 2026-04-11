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
///   1. Call `JSPreprocessorWrapper.shared.load(from: url)` once after downloading the JS file.
///   2. Then `FeaturePreprocessor` delegates all calls here automatically.
public final class JSPreprocessorWrapper {

    public static let shared = JSPreprocessorWrapper()

    private let context = JSContext()!
    private(set) var isLoaded = false

    private init() {
        context.exceptionHandler = { _, exception in
            print("[JSPreprocessorWrapper] JS exception: \(exception?.toString() ?? "unknown")")
        }
    }

    // MARK: - Loading

    /// Evaluate preprocessor.js from disk. Safe to call again to hot-reload after an update.
    public func load(from url: URL) throws {
        let source = try String(contentsOf: url, encoding: .utf8)
        context.evaluateScript(source)
        if let exception = context.exception {
            throw PreprocessorError.loadFailed(exception.toString() ?? "unknown JS exception")
        }
        isLoaded = true
    }

    // MARK: - Public API

    /// Flat array of length 60 × 126 = 7 560.
    func featureMatrix(from handfilm: HandFilm) -> [Double] {
        callDoubleArray("featureMatrix", handfilm: handfilm)
    }

    /// Array of 60 rows, each of length 126.
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

    /// 256-element Float vector used as MLP input.
    func summaryFeatures(from handfilm: HandFilm) -> [Float] {
        callDoubleArray("summaryFeatures", handfilm: handfilm).map { Float($0) }
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
            let lms = frame.landmarks.map { lm in
                "{\(q("x")):\(lm.x),\(q("y")):\(lm.y),\(q("z")):\(lm.z)}"
            }.joined(separator: ",")

            let hand: String
            switch frame.leftOrRight {
            case .left:    hand = "left"
            case .right:   hand = "right"
            case .unknown: hand = "unknown"
            }

            frameStrings.append(
                "{\(q("landmarks")):[\(lms)],\(q("timestamp")):\(frame.timestamp),\(q("left_or_right")):\(q(hand))}"
            )
        }

        return "{\(q("frames")):[\(frameStrings.joined(separator: ","))],\(q("start_time")):\(startTime)}"
    }

    /// Wrap a string in JSON double-quotes.
    private func q(_ s: String) -> String { "\"\(s)\"" }
}

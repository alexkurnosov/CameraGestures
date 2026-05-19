// Stage 5: JSPreprocessorWrapper compatibility shim.
//
// In V1 (GestureModelModule) this class loaded preprocessor.js via JavaScriptCore
// and delegated all feature-computation calls to the JS runtime.
//
// In V2 the feature computation is done entirely in C++ (FeaturePreprocessor.cpp).
// The JS file is no longer needed at runtime. This shim preserves the same public
// API so Training App v2 compiles unchanged:
//
//   • load(from:)    — no-op; the C++ preprocessor needs no external JS file.
//   • preprocVersion — hardcoded to match the C++ preprocessor version.
//   • poseVectorSize, featuresPerFrame, summaryFeaturesCount — match FeaturePreprocessor.hpp.
//   • isLoaded       — always true (C++ preprocessor is always ready).
//
// If the server ships a new preprocessor.js that changes constants, the C++ library
// must be rebuilt to match — there is no hot-reload path in V2.

import Foundation

// MARK: - Error (kept for API compat)

public enum PreprocessorError: Error, LocalizedError {
    case notLoaded
    case loadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notLoaded:           return "Preprocessor not loaded."
        case .loadFailed(let msg): return "Preprocessor load failed: \(msg)"
        }
    }
}

// MARK: - Shim

public final class JSPreprocessorWrapper {

    public static let shared = JSPreprocessorWrapper()
    private init() {}

    // C++ FeaturePreprocessor.hpp constants (CG_POSE_VECTOR_SIZE, CG_FEATURES_PER_FRAME,
    // CG_SUMMARY_FEATURES). Must stay in sync with the compiled XCFramework.
    public private(set) var preprocVersion:       Int = 2
    public private(set) var poseVectorSize:       Int = 83   // CG_POSE_VECTOR_SIZE
    public private(set) var featuresPerFrame:     Int = 146  // CG_FEATURES_PER_FRAME
    public private(set) var summaryFeaturesCount: Int = 296  // CG_SUMMARY_FEATURES
    public var isLoaded: Bool { true }

    /// No-op in V2: the C++ preprocessor is compiled into the XCFramework and requires
    /// no external JS file. The downloaded preprocessor.js is silently ignored at runtime.
    /// The method still `throw`s on actual I/O errors so callers using `try` still compile.
    public func load(from url: URL) throws {
        // Validate the file is accessible so callers get a real error if the server
        // download failed to produce the file, but otherwise ignore the JS content.
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw PreprocessorError.loadFailed("file not found at \(url.path)")
        }
        // Log the version embedded in the JS file for diagnostic purposes only.
        if let source = try? String(contentsOf: url, encoding: .utf8),
           let range = source.range(of: #"PREPROC_VERSION\s*=\s*(\d+)"#,
                                     options: .regularExpression) {
            let match = source[range]
            print("[JSPreprocessorWrapper] JS file found (ignored at runtime): \(match). "
                + "C++ preprocessor v\(preprocVersion) is active.")
        }
    }
}

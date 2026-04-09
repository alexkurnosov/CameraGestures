import Foundation
import HandGestureTypes

/// Converts a HandFilm into feature vectors for TensorFlow Lite inference.
///
/// All logic lives in preprocessor.js and is executed via JSPreprocessorWrapper.
/// Call `JSPreprocessorWrapper.shared.load(from:)` before using any method here.
enum FeaturePreprocessor {

    static let frameCount       = 60
    static let landmarkCount    = 21
    static let coordsPerFrame   = landmarkCount * 3       // 63
    static let featuresPerFrame = coordsPerFrame * 2      // 126

    // MARK: - Public API

    /// Flat array of length frameCount × featuresPerFrame (7 560).
    static func featureMatrix(from handfilm: HandFilm) -> [Double] {
        JSPreprocessorWrapper.shared.featureMatrix(from: handfilm)
    }

    /// Array of frameCount rows, each of length featuresPerFrame.
    static func featureRows(from handfilm: HandFilm) -> [[Double]] {
        JSPreprocessorWrapper.shared.featureRows(from: handfilm)
    }

    /// 256-element Float vector used as MLP input.
    static func summaryFeatures(from handfilm: HandFilm) -> [Float] {
        JSPreprocessorWrapper.shared.summaryFeatures(from: handfilm)
    }
}

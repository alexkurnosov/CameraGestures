import Foundation
import HandGestureTypes

/// Converts a HandFilm into feature vectors for TensorFlow Lite inference.
///
/// All logic lives in preprocessor.js and is executed via JSPreprocessorWrapper.
/// Call `JSPreprocessorWrapper.shared.load(from:)` before using any method here.
enum FeaturePreprocessor {

    static let frameCount       = 60
    static let landmarkCount    = 21
    static let coordsPerFrame   = landmarkCount * 3   // 63 — landmark coords only

    /// Per-frame pose vector: 63 landmark coords + 20 geometric extras = 83 (post preprocessor v2).
    /// Reads the live value from the loaded JS; falls back to 63 if JS not yet loaded.
    static var poseVectorSize: Int { JSPreprocessorWrapper.shared.poseVectorSize }

    /// Per-frame feature width: poseVectorSize + coordsPerFrame (pose + velocity).
    static var featuresPerFrame: Int { JSPreprocessorWrapper.shared.featuresPerFrame }

    /// Length of the Phase 3 summary feature vector (296 post preprocessor v2).
    static var summaryFeaturesCount: Int { JSPreprocessorWrapper.shared.summaryFeaturesCount }

    // MARK: - Public API

    /// Flat array of length frameCount × featuresPerFrame.
    static func featureMatrix(from handfilm: HandFilm) -> [Double] {
        JSPreprocessorWrapper.shared.featureMatrix(from: handfilm)
    }

    /// Array of frameCount rows, each of length featuresPerFrame.
    static func featureRows(from handfilm: HandFilm) -> [[Double]] {
        JSPreprocessorWrapper.shared.featureRows(from: handfilm)
    }

    /// summaryFeaturesCount-element Float vector used as Phase 3 MLP input.
    static func summaryFeatures(from handfilm: HandFilm) -> [Float] {
        JSPreprocessorWrapper.shared.summaryFeatures(from: handfilm)
    }
}

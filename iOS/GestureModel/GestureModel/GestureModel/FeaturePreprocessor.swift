import Foundation
import HandGestureTypes

/// Converts a HandFilm into a fixed-length feature matrix for TensorFlow Lite inference.
///
/// Output shape: 60 frames × 126 features/frame
///   [0..62]  — 21 landmark (x, y, z) triplets normalized relative to wrist (landmark 0)
///   [63..125] — frame-to-frame velocity of those same 63 coordinates (zero for first frame)
enum FeaturePreprocessor {

    static let frameCount = 60
    static let landmarkCount = 21
    static let coordsPerFrame = landmarkCount * 3       // 63
    static let featuresPerFrame = coordsPerFrame * 2    // 126

    // MARK: - Public API

    /// Convert a HandFilm to a [frameCount × featuresPerFrame] Double matrix.
    /// - Returns: Flat row-major array of length frameCount × featuresPerFrame.
    static func featureMatrix(from handfilm: HandFilm) -> [Double] {
        let normalizedFrames = buildNormalizedFrames(from: handfilm)
        let velocityFrames = buildVelocityFrames(from: normalizedFrames)

        var result = [Double]()
        result.reserveCapacity(frameCount * featuresPerFrame)

        for i in 0..<frameCount {
            result.append(contentsOf: normalizedFrames[i])
            result.append(contentsOf: velocityFrames[i])
        }
        return result
    }

    /// Convenience: returns the matrix as a nested array (rows = frames, cols = features).
    static func featureRows(from handfilm: HandFilm) -> [[Double]] {
        let flat = featureMatrix(from: handfilm)
        return (0..<frameCount).map { i in
            Array(flat[(i * featuresPerFrame)..<((i + 1) * featuresPerFrame)])
        }
    }

    /// Mirrors preprocessor.py `summary_features()`.
    ///
    /// Produces a 256-element Float vector used as input to the rf_mlp TFLite model:
    ///   - mean of 63 normalised coord dims across frames  →  63
    ///   - std  of 63 normalised coord dims across frames  →  63
    ///   - mean of 63 velocity dims across frames          →  63
    ///   - std  of 63 velocity dims across frames          →  63
    ///   - net raw wrist displacement (xyz)                →   3
    ///   - dominant motion axis magnitude                  →   1
    ///   Total: 256
    static func summaryFeatures(from handfilm: HandFilm) -> [Float] {
        let mat = featureMatrix(from: handfilm)  // length: frameCount * featuresPerFrame

        // Split into coords (cols 0–62) and velocities (cols 63–125) per frame
        var coords = [[Double]]()  // frameCount × coordsPerFrame
        var vels   = [[Double]]()  // frameCount × coordsPerFrame
        coords.reserveCapacity(frameCount)
        vels.reserveCapacity(frameCount)

        for i in 0..<frameCount {
            let base = i * featuresPerFrame
            coords.append(Array(mat[base ..< base + coordsPerFrame]))
            vels.append(Array(mat[base + coordsPerFrame ..< base + featuresPerFrame]))
        }

        // Column-wise mean and std helpers
        func colMean(_ m: [[Double]]) -> [Double] {
            (0..<coordsPerFrame).map { j in
                m.reduce(0.0) { $0 + $1[j] } / Double(frameCount)
            }
        }
        func colStd(_ m: [[Double]], means: [Double]) -> [Double] {
            (0..<coordsPerFrame).map { j in
                let variance = m.reduce(0.0) { $0 + pow($1[j] - means[j], 2) } / Double(frameCount)
                return sqrt(variance)
            }
        }

        let coordMean = colMean(coords)
        let coordStd  = colStd(coords, means: coordMean)
        let velMean   = colMean(vels)
        let velStd    = colStd(vels, means: velMean)

        // Net raw wrist displacement (first vs last frame, before normalisation)
        var displacement = [Double](repeating: 0.0, count: 3)
        let frames = handfilm.frames
        if frames.count >= 2 {
            let first = frames.first!.landmarks[0]
            let last  = frames.last!.landmarks[0]
            displacement[0] = Double(last.x - first.x)
            displacement[1] = Double(last.y - first.y)
            displacement[2] = Double(last.z - first.z)
        }

        // Dominant motion axis: max |mean velocity| across x/y/z averaged over all landmarks
        // vels shape: frameCount × (landmarkCount * 3), reshape to frameCount × landmarkCount × 3
        var axisSum = [Double](repeating: 0.0, count: 3)
        for frameVel in vels {
            for lmIdx in 0..<landmarkCount {
                let base = lmIdx * 3
                axisSum[0] += frameVel[base + 0]
                axisSum[1] += frameVel[base + 1]
                axisSum[2] += frameVel[base + 2]
            }
        }
        let total = Double(frameCount * landmarkCount)
        let dominant = axisSum.map { abs($0 / total) }.max() ?? 0.0

        var result = [Double]()
        result.reserveCapacity(256)
        result.append(contentsOf: coordMean)
        result.append(contentsOf: coordStd)
        result.append(contentsOf: velMean)
        result.append(contentsOf: velStd)
        result.append(contentsOf: displacement)
        result.append(dominant)

        return result.map { Float($0) }
    }

    // MARK: - Private Helpers

    /// Extract and normalize landmark coords for each frame, pad/trim to frameCount.
    private static func buildNormalizedFrames(from handfilm: HandFilm) -> [[Double]] {
        let frames = handfilm.frames
        var normalized = [[Double]]()
        normalized.reserveCapacity(frameCount)

        // Use last `frameCount` frames if the film is longer; pad with zeros if shorter.
        let startIndex = max(0, frames.count - frameCount)

        for i in startIndex..<frames.count {
            normalized.append(normalizeFrame(frames[i]))
        }

        let zeroFrame = [Double](repeating: 0.0, count: coordsPerFrame)
        while normalized.count < frameCount {
            normalized.append(zeroFrame)
        }

        return normalized
    }

    /// Normalize a single HandShot's 21 landmarks relative to wrist (landmark 0).
    private static func normalizeFrame(_ handshot: HandShot) -> [Double] {
        let landmarks = handshot.landmarks
        guard landmarks.count == landmarkCount else {
            return [Double](repeating: 0.0, count: coordsPerFrame)
        }

        let wrist = landmarks[0]
        var coords = [Double]()
        coords.reserveCapacity(coordsPerFrame)

        for lm in landmarks {
            coords.append(Double(lm.x - wrist.x))
            coords.append(Double(lm.y - wrist.y))
            coords.append(Double(lm.z - wrist.z))
        }
        return coords
    }

    /// Compute frame-to-frame velocity; first frame velocity is all zeros.
    private static func buildVelocityFrames(from normalizedFrames: [[Double]]) -> [[Double]] {
        let zeroFrame = [Double](repeating: 0.0, count: coordsPerFrame)
        var velocity = [zeroFrame]

        for i in 1..<normalizedFrames.count {
            var delta = [Double](repeating: 0.0, count: coordsPerFrame)
            for j in 0..<coordsPerFrame {
                delta[j] = normalizedFrames[i][j] - normalizedFrames[i - 1][j]
            }
            velocity.append(delta)
        }
        return velocity
    }
}

import Foundation
import HandGestureTypes
import TensorFlowLite

/// Gesture ID used by the server trainer for the synthetic negative class.
/// Predictions for this class are filtered out — they indicate "not a known gesture".
private let noneGestureID = "_none"

/// Neural network abstraction layer for gesture classification.
public class GestureModel {

    // MARK: - Properties

    private var config: GestureModelConfig
    private var isModelLoaded = false
    private var mockBackend: MockGestureBackend?
    private var supportedGestureIds: [String] = []
    private var tfliteInterpreter: Interpreter?

    // MARK: - Pose model (Phase 2 single-frame classifier)

    private var poseInterpreter: Interpreter?
    private var _poseManifest: PoseManifest?
    /// Ordered list of cluster id strings matching the pose MLP output indices.
    private var poseClusterIds: [String] = []

    public var isPoseModelLoaded: Bool { poseInterpreter != nil && _poseManifest != nil }
    public var poseManifest: PoseManifest? { _poseManifest }

    // MARK: - Initialization

    public init() {
        self.config = .defaultConfig
        setupBackend()
    }

    public init(config: GestureModelConfig) {
        self.config = config
        setupBackend()
    }

    // MARK: - Configuration

    public func initialize(config: GestureModelConfig) throws {
        self.config = config
        setupBackend()

        if let modelPath = config.modelPath {
            try loadModel(from: modelPath)
        }
    }

    // MARK: - Model Management

    /// Load model from file path (.tflite file or mock file).
    public func loadModel(from path: String) throws {
        guard FileManager.default.fileExists(atPath: path) else {
            throw GestureModelError.invalidModelPath
        }

        switch config.backendType {
        case .tensorFlow:
            try loadTensorFlowModel(from: path)
        case .mock:
            try loadMockModel(from: path)
        }

        isModelLoaded = true
    }

    /// Save model to a destination path.
    public func saveModel(to path: String) throws {
        guard isModelLoaded else {
            throw GestureModelError.modelNotLoaded
        }

        switch config.backendType {
        case .tensorFlow:
            try saveTensorFlowModel(to: path)
        case .mock:
            try saveMockModel(to: path)
        }
    }

    public var isLoaded: Bool { isModelLoaded }

    // MARK: - Pose Model Management (Phase 2 single-frame classifier)

    /// Load the pose MLP (.tflite) and its manifest from the given file paths.
    /// Cluster ids in the manifest are sorted numerically to derive the output-index mapping.
    public func loadPoseModel(tflitePath: String, manifestPath: String) throws {
        guard FileManager.default.fileExists(atPath: tflitePath) else {
            throw GestureModelError.invalidModelPath
        }
        let manifestData = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
        let manifest = try JSONDecoder().decode(PoseManifest.self, from: manifestData)

        let interpreter = try Interpreter(modelPath: tflitePath)
        try interpreter.allocateTensors()

        let inputDim = (try? interpreter.input(at: 0))?.shape.dimensions.last ?? 0
        let expected = FeaturePreprocessor.poseVectorSize
        guard inputDim == expected else {
            print("[GestureModel] Pose model input dim \(inputDim) ≠ preprocessor poseVectorSize \(expected). Reload preprocessor.js and model together.")
            throw GestureModelError.invalidInput
        }

        _poseManifest = manifest
        poseInterpreter = interpreter
        poseClusterIds = manifest.poseClusters.keys.sorted {
            (Int($0) ?? 0) < (Int($1) ?? 0)
        }
    }

    /// Predict the pose cluster for a single frame given its pose vector
    /// (output of JSPreprocessorWrapper.poseVector — poseVectorSize floats, currently 83).
    public func predictPose(normalizedCoords: [Float]) throws -> PosePrediction? {
        guard isPoseModelLoaded else { throw GestureModelError.modelNotLoaded }
        guard normalizedCoords.count == FeaturePreprocessor.poseVectorSize else { throw GestureModelError.invalidInput }
        guard let interpreter = poseInterpreter, let manifest = _poseManifest else {
            throw GestureModelError.modelNotLoaded
        }

        var coords = normalizedCoords
        let inputData = Data(bytes: &coords, count: coords.count * MemoryLayout<Float>.size)

        do {
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()

            let outputTensor = try interpreter.output(at: 0)
            let probabilities: [Float] = outputTensor.data.withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }

            guard probabilities.count == poseClusterIds.count else {
                throw GestureModelError.predictionFailed
            }

            guard let maxIdx = probabilities.indices.max(by: { probabilities[$0] < probabilities[$1] }) else {
                return nil
            }

            let clusterIdStr = poseClusterIds[maxIdx]
            let clusterId = Int(clusterIdStr) ?? maxIdx
            let cluster = manifest.poseClusters[clusterIdStr]
            let kind = ClusterKind(rawValue: cluster?.kind ?? "unconfirmed") ?? .unconfirmed
            let label = cluster?.label ?? "pose_\(clusterId)"

            return PosePrediction(
                poseId: clusterId,
                confidence: probabilities[maxIdx],
                kind: kind,
                clusterLabel: label
            )
        } catch let e as GestureModelError {
            throw e
        } catch {
            throw GestureModelError.predictionFailed
        }
    }

    // MARK: - Prediction

    public func predict(handfilm: HandFilm) throws -> GesturePrediction? {
        guard isModelLoaded else { throw GestureModelError.modelNotLoaded }
        guard !handfilm.frames.isEmpty else { throw GestureModelError.invalidInput }
        return try predictTopK(handfilm: handfilm, k: 1).first
    }

    public func predictTopK(handfilm: HandFilm, k: Int) throws -> [GesturePrediction] {
        guard isModelLoaded else { throw GestureModelError.modelNotLoaded }
        guard !handfilm.frames.isEmpty else { throw GestureModelError.invalidInput }

        let maxK = min(k, config.maxPredictions)

        switch config.backendType {
        case .tensorFlow:
            return try predictWithTensorFlow(handfilm: handfilm, k: maxK)
        case .mock:
            return try predictWithMock(handfilm: handfilm, k: maxK)
        }
    }

    /// Phase 3 restricted prediction (plan §Phase 3 Output restriction to candidate set S).
    ///
    /// Runs Phase 3 normally, then takes the argmax only over classes in `candidateGestures`.
    /// Reports the **pre-mask, unrenormalised** softmax probability of the chosen class so
    /// τ_phase3_confidence is invariant to |S|.
    public func predictRestrictedToSet(handfilm: HandFilm, candidateGestures: Set<String>) throws -> GesturePrediction? {
        guard isModelLoaded else { throw GestureModelError.modelNotLoaded }
        guard !handfilm.frames.isEmpty else { throw GestureModelError.invalidInput }
        guard !candidateGestures.isEmpty, !supportedGestureIds.isEmpty else { return nil }

        switch config.backendType {
        case .tensorFlow:
            return try predictMaskedArgmaxTensorFlow(handfilm: handfilm, candidateGestures: candidateGestures)
        case .mock:
            return nil
        }
    }

    public func predictStreaming(handshots: [HandShot]) throws -> [GesturePrediction] {
        guard isModelLoaded else { throw GestureModelError.modelNotLoaded }

        guard config.enableTemporal else {
            guard let latestHandshot = handshots.last else { throw GestureModelError.invalidInput }
            var handfilm = HandFilm()
            handfilm.addFrame(latestHandshot)
            return try predict(handfilm: handfilm).map { [$0] } ?? []
        }

        let currentTime = Date().timeIntervalSince1970
        let windowStart = currentTime - config.temporalWindow
        let recentHandshots = handshots.filter { $0.timestamp >= windowStart }

        guard !recentHandshots.isEmpty else { return [] }

        var handfilm = HandFilm(startTime: recentHandshots.first!.timestamp)
        recentHandshots.forEach { handfilm.addFrame($0) }

        return try predictTopK(handfilm: handfilm, k: config.maxPredictions)
    }

    // MARK: - Training (local mock only; real training happens server-side)

    public func train(dataset: TrainingDataset) throws -> ModelMetrics {
        guard !dataset.examples.isEmpty else { throw GestureModelError.insufficientData }

        supportedGestureIds = Array(Set(dataset.examples.map { $0.gestureId })).sorted()

        switch config.backendType {
        case .tensorFlow:
            return try trainWithTensorFlow(dataset: dataset)
        case .mock:
            return try trainWithMock(dataset: dataset)
        }
    }

    public func trainAsync(dataset: TrainingDataset) async throws -> ModelMetrics {
        guard !dataset.examples.isEmpty else { throw GestureModelError.insufficientData }

        supportedGestureIds = Array(Set(dataset.examples.map { $0.gestureId })).sorted()

        switch config.backendType {
        case .tensorFlow:
            return try trainWithTensorFlow(dataset: dataset)
        case .mock:
            return try await Task.detached(priority: .userInitiated) { [weak self] in
                guard let self else { throw GestureModelError.trainingFailed }
                return try self.trainWithMock(dataset: dataset)
            }.value
        }
    }

    public func evaluate(testDataset: TrainingDataset) throws -> ModelMetrics {
        guard isModelLoaded else { throw GestureModelError.modelNotLoaded }
        guard !testDataset.examples.isEmpty else { throw GestureModelError.insufficientData }
        return MockData.mockModelMetrics(gestureCount: supportedGestureIds.count)
    }

    // MARK: - Status and Info

    public func getConfig() -> GestureModelConfig { config }

    public func getSupportedGestures() -> [String] { supportedGestureIds }

    public func setSupportedGestures(_ ids: [String]) { supportedGestureIds = ids }

    public func getModelInfo() -> [String: Any] {
        [
            "backend": config.backendType.rawValue,
            "loaded": isModelLoaded,
            "modelPath": config.modelPath ?? "none",
            "supportedGestures": supportedGestureIds,
            "temporalEnabled": config.enableTemporal,
            "predictionThreshold": config.predictionThreshold
        ]
    }

    // MARK: - Private Setup

    private func setupBackend() {
        switch config.backendType {
        case .mock:
            mockBackend = MockGestureBackend()
        case .tensorFlow:
            break
        }
    }

    // MARK: - TensorFlow Lite Backend

    private func loadTensorFlowModel(from path: String) throws {
        do {
            let interpreter = try Interpreter(modelPath: path)
            try interpreter.allocateTensors()

            let inputDim = (try? interpreter.input(at: 0))?.shape.dimensions.last ?? 0
            let expected = FeaturePreprocessor.summaryFeaturesCount
            guard inputDim == expected else {
                print("[GestureModel] Gesture model input dim \(inputDim) ≠ preprocessor summaryFeaturesCount \(expected). Reload preprocessor.js and model together.")
                throw GestureModelError.invalidInput
            }

            tfliteInterpreter = interpreter
        } catch let e as GestureModelError {
            throw e
        } catch {
            throw GestureModelError.predictionFailed
        }
    }

    private func saveTensorFlowModel(to path: String) throws {}

    private func predictWithTensorFlow(handfilm: HandFilm, k: Int) throws -> [GesturePrediction] {
        guard let interpreter = tfliteInterpreter else {
            throw GestureModelError.modelNotLoaded
        }
        guard !supportedGestureIds.isEmpty else { return [] }

        var floatFeatures = FeaturePreprocessor.summaryFeatures(from: handfilm)

        let inputByteCount = floatFeatures.count * MemoryLayout<Float>.size
        let inputData = Data(bytes: &floatFeatures, count: inputByteCount)

        do {
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()

            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data
            let outputCount = outputData.count / MemoryLayout<Float>.size

            let probabilities: [Float] = outputData.withUnsafeBytes { rawBuffer in
                Array(rawBuffer.bindMemory(to: Float.self))
            }

            guard outputCount == supportedGestureIds.count else {
                throw GestureModelError.predictionFailed
            }

            let timestamp = Date().timeIntervalSince1970
            let predictions = probabilities.enumerated().compactMap { (index, confidence) -> GesturePrediction? in
                guard confidence >= config.predictionThreshold else { return nil }
                let gestureId = supportedGestureIds[index]
                // Skip the synthetic negative class — it means "not a known gesture".
                guard gestureId != noneGestureID else { return nil }
                return GesturePrediction(gestureId: gestureId, gestureName: gestureId, confidence: confidence, timestamp: timestamp)
            }

            return Array(predictions.sorted { $0.confidence > $1.confidence }.prefix(k))
        } catch let e as GestureModelError {
            throw e
        } catch {
            throw GestureModelError.predictionFailed
        }
    }

    private func predictMaskedArgmaxTensorFlow(handfilm: HandFilm, candidateGestures: Set<String>) throws -> GesturePrediction? {
        guard let interpreter = tfliteInterpreter else { throw GestureModelError.modelNotLoaded }
        guard !supportedGestureIds.isEmpty else { return nil }

        var floatFeatures = FeaturePreprocessor.summaryFeatures(from: handfilm)
        let inputData = Data(bytes: &floatFeatures, count: floatFeatures.count * MemoryLayout<Float>.size)

        do {
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()

            let outputTensor = try interpreter.output(at: 0)
            let probabilities: [Float] = outputTensor.data.withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }

            guard probabilities.count == supportedGestureIds.count else {
                throw GestureModelError.predictionFailed
            }

            // Masked argmax: best probability only among candidateGestures
            var bestIdx: Int? = nil
            var bestProb: Float = -1
            for (i, gestureId) in supportedGestureIds.enumerated() {
                guard candidateGestures.contains(gestureId), gestureId != noneGestureID else { continue }
                if probabilities[i] > bestProb {
                    bestProb = probabilities[i]
                    bestIdx = i
                }
            }

            guard let idx = bestIdx, bestProb >= config.predictionThreshold else { return nil }

            let gestureId = supportedGestureIds[idx]
            return GesturePrediction(
                gestureId: gestureId,
                gestureName: gestureId,
                confidence: bestProb,
                timestamp: Date().timeIntervalSince1970
            )
        } catch let e as GestureModelError {
            throw e
        } catch {
            throw GestureModelError.predictionFailed
        }
    }

    private func trainWithTensorFlow(dataset: TrainingDataset) throws -> ModelMetrics {
        return MockData.mockModelMetrics(gestureCount: supportedGestureIds.count)
    }

    // MARK: - Mock Backend

    private func loadMockModel(from path: String) throws {
        mockBackend?.loadModel(path: path)
        isModelLoaded = true
    }

    private func saveMockModel(to path: String) throws {
        try mockBackend?.saveModel(path: path)
    }

    private func predictWithMock(handfilm: HandFilm, k: Int) throws -> [GesturePrediction] {
        guard let backend = mockBackend else { throw GestureModelError.unsupportedBackend }
        return backend.predict(handfilm: handfilm, k: k, threshold: config.predictionThreshold)
    }

    private func trainWithMock(dataset: TrainingDataset) throws -> ModelMetrics {
        guard let backend = mockBackend else { throw GestureModelError.unsupportedBackend }
        return backend.train(dataset: dataset)
    }
}

// MARK: - Mock Backend Implementation

private class MockGestureBackend {
    private var modelLoaded = false
    private var modelPath: String?

    func loadModel(path: String) {
        self.modelPath = path
        self.modelLoaded = true
    }

    func saveModel(path: String) throws {
        guard modelLoaded else { throw GestureModelError.modelNotLoaded }
        try "Mock model data".write(toFile: path, atomically: true, encoding: .utf8)
    }

    func predict(handfilm: HandFilm, k: Int, threshold: Float) -> [GesturePrediction] {
        return []
    }

    func train(dataset: TrainingDataset) -> ModelMetrics {
        Thread.sleep(forTimeInterval: 2.0)
        let gestureCount = Set(dataset.examples.map { $0.gestureId }).count
        return MockData.mockModelMetrics(gestureCount: gestureCount)
    }
}

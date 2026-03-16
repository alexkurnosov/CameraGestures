import Foundation
import HandGestureTypes
import TensorFlowLite

/// Neural network abstraction layer for gesture classification.
public class GestureModel {

    // MARK: - Properties

    private var config: GestureModelConfig
    private var isModelLoaded = false
    private var mockBackend: MockGestureBackend?
    private var supportedGestureIds: [String] = []
    private var tfliteInterpreter: Interpreter?

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
            tfliteInterpreter = interpreter
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
                return GesturePrediction(gestureId: gestureId, gestureName: gestureId, confidence: confidence, timestamp: timestamp)
            }

            return Array(predictions.sorted { $0.confidence > $1.confidence }.prefix(k))
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

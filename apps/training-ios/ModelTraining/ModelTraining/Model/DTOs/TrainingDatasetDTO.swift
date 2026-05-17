import Foundation
import HandGestureTypes

struct TrainingDatasetDTO: Codable {
    let name: String
    let createdAt: TimeInterval
    let examples: [TrainingExampleDTO]

    init(from dataset: TrainingDataset) {
        name = dataset.name
        createdAt = dataset.createdAt
        examples = dataset.examples.map { TrainingExampleDTO(from: $0) }
    }

    func toTrainingDataset() -> TrainingDataset {
        var dataset = TrainingDataset(name: name)
        examples.map { $0.toTrainingExample() }.forEach { dataset.addExample($0) }
        return dataset
    }
}

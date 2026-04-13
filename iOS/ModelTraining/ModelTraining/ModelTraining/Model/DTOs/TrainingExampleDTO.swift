import Foundation
import HandGestureTypes

struct TrainingExampleDTO: Codable {
    let id: UUID
    let handfilm: HandFilmDTO
    let gestureId: String
    let userId: String?
    let sessionId: String
    let timestamp: TimeInterval

    init(from example: TrainingExample) {
        id = example.id
        handfilm = HandFilmDTO(from: example.handfilm)
        gestureId = example.gestureId
        userId = example.userId
        sessionId = example.sessionId
        timestamp = example.timestamp
    }

    func toTrainingExample() -> TrainingExample {
        TrainingExample(
            id: id,
            handfilm: handfilm.toHandFilm(),
            gestureId: gestureId,
            userId: userId,
            sessionId: sessionId
        )
    }
}

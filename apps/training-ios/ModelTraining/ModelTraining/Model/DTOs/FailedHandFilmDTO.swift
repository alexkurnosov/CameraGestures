import Foundation
import HandGestureTypes

struct FailedHandFilmDTO: Codable {
    let id: UUID
    let handfilm: HandFilmDTO
    let gestureId: String
    let failureReason: HandFilmFailureReason
    let failureDetail: String
    let timestamp: TimeInterval
    let isManuallyValidated: Bool

    init(from film: FailedHandFilm) {
        id = film.id
        handfilm = HandFilmDTO(from: film.handfilm)
        gestureId = film.gestureId
        failureReason = film.failureReason
        failureDetail = film.failureDetail
        timestamp = film.timestamp
        isManuallyValidated = film.isManuallyValidated
    }

    func toFailedHandFilm() -> FailedHandFilm {
        FailedHandFilm(
            id: id,
            handfilm: handfilm.toHandFilm(),
            gestureId: gestureId,
            failureReason: failureReason,
            failureDetail: failureDetail,
            isManuallyValidated: isManuallyValidated
        )
    }
}

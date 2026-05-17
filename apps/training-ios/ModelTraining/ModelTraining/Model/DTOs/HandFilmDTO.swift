import Foundation
import HandGestureTypes

struct HandFilmDTO: Codable {
    let frames: [HandShotDTO]
    let startTime: TimeInterval

    init(from handFilm: HandFilm) {
        frames = handFilm.frames.map { HandShotDTO(from: $0) }
        startTime = handFilm.startTime
    }

    func toHandFilm() -> HandFilm {
        var film = HandFilm(startTime: startTime)
        frames.map { $0.toHandShot() }.forEach { film.addFrame($0) }
        return film
    }
}

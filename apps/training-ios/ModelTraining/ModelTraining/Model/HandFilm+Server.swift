import CameraGestures

extension HandFilm {
    init(server film: ServerHandFilmResponse) {
        var result = HandFilm(startTime: film.startTime)
        for shot in film.frames {
            let landmarks = shot.landmarks.map { Point3D(x: $0.x, y: $0.y, z: $0.z) }
            let side: LeftOrRight = {
                switch shot.leftOrRight {
                case "left": return .left
                case "right": return .right
                default: return .unknown
                }
            }()
            result.addFrame(HandShot(
                landmarks: landmarks,
                timestamp: shot.timestamp,
                leftOrRight: side,
                isAbsent: shot.isAbsent ?? false
            ))
        }
        self = result
    }
}

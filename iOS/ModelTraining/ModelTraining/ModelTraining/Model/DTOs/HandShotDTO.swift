import Foundation
import HandGestureTypes

struct HandShotDTO: Codable {
    let landmarks: [Point3DDTO]
    let timestamp: TimeInterval
    let leftOrRight: String   // "left" | "right" | "unknown"
    let isAbsent: Bool

    init(from handShot: HandShot) {
        landmarks = handShot.landmarks.map { Point3DDTO(from: $0) }
        timestamp = handShot.timestamp
        isAbsent = handShot.isAbsent
        leftOrRight = {
            switch handShot.leftOrRight {
            case .left: return "left"
            case .right: return "right"
            case .unknown: return "unknown"
            }
        }()
    }

    func toHandShot() -> HandShot {
        let side: LeftOrRight = leftOrRight == "left" ? .left : leftOrRight == "right" ? .right : .unknown
        return HandShot(
            landmarks: landmarks.map { $0.toPoint3D() },
            timestamp: timestamp,
            leftOrRight: side,
            isAbsent: isAbsent
        )
    }
}

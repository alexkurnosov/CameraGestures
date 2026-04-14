import Foundation
import HandGestureTypes

/// DTO for Point3D (HandGestureTypes is not Codable to keep it dependency-free).
struct Point3DDTO: Codable {
    let x: Float
    let y: Float
    let z: Float

    init(from point: Point3D) {
        x = point.x; y = point.y; z = point.z
    }

    func toPoint3D() -> Point3D { Point3D(x: x, y: y, z: z) }
}

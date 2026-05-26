import SwiftUI

/// Draws the 21 MediaPipe hand landmarks + skeleton connections on top of the
/// camera preview, so we can visually confirm whether the landmark model is
/// actually tracking the user's hand.
///
/// Landmarks arrive in normalized [0,1] image-space coordinates. The camera
/// preview uses .resizeAspectFill, so the displayed image is scaled to fill
/// the view (cropping the longer dimension). We mirror that transform here so
/// the skeleton lands on the visible portion of the frame.
struct HandOverlayView: View {
    let landmarks: [Point3D]          // 21 normalized points (or empty)
    let cameraWidth:  CGFloat          // source frame width  in pixels
    let cameraHeight: CGFloat          // source frame height in pixels
    let mirrored:     Bool             // front-facing camera is shown mirrored

    /// MediaPipe hand-skeleton edges.
    private static let connections: [(Int, Int)] = [
        // Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        // Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        // Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        // Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        // Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        // Palm web
        (5, 9), (9, 13), (13, 17),
    ]

    var body: some View {
        GeometryReader { geo in
            if landmarks.count == 21 {
                let viewPoints = landmarks.map { lm in
                    Self.projectAspectFill(
                        nx: CGFloat(lm.x), ny: CGFloat(lm.y),
                        camW: cameraWidth, camH: cameraHeight,
                        viewSize: geo.size,
                        mirrored: mirrored)
                }
                ZStack {
                    // Skeleton edges
                    Path { p in
                        for (a, b) in Self.connections {
                            p.move(to: viewPoints[a])
                            p.addLine(to: viewPoints[b])
                        }
                    }
                    .stroke(Color.green, lineWidth: 2)

                    // Landmark dots
                    Path { p in
                        for pt in viewPoints {
                            p.addEllipse(in: CGRect(x: pt.x - 4, y: pt.y - 4,
                                                    width: 8,    height: 8))
                        }
                    }
                    .fill(Color.green)
                }
                .allowsHitTesting(false)
            }
        }
    }

    /// Map a normalized image-space point to view-space coordinates, accounting
    /// for AVCaptureVideoPreviewLayer's .resizeAspectFill cropping and the
    /// horizontal mirroring applied to front-facing camera previews.
    private static func projectAspectFill(
        nx: CGFloat, ny: CGFloat,
        camW: CGFloat, camH: CGFloat,
        viewSize: CGSize,
        mirrored: Bool
    ) -> CGPoint {
        let scale     = max(viewSize.width / camW, viewSize.height / camH)
        let scaledW   = camW * scale
        let scaledH   = camH * scale
        let offsetX   = (viewSize.width  - scaledW) / 2
        let offsetY   = (viewSize.height - scaledH) / 2

        // Apply horizontal mirror in image space if the preview is mirrored.
        let displayNx = mirrored ? (1.0 - nx) : nx
        let x = displayNx * scaledW + offsetX
        let y = ny        * scaledH + offsetY
        return CGPoint(x: x, y: y)
    }
}

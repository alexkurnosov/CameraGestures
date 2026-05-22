import SwiftUI
import AVFoundation
import AppKit

// NSViewRepresentable wrapper for AVCaptureVideoPreviewLayer on macOS.
struct CameraPreviewView: NSViewRepresentable {
    let session: AVCaptureSession?

    func makeNSView(context: Context) -> CameraPreviewNSView {
        let view = CameraPreviewNSView()
        if let session { view.setSession(session) }
        return view
    }

    func updateNSView(_ nsView: CameraPreviewNSView, context: Context) {
        if let session { nsView.setSession(session) }
    }
}

final class CameraPreviewNSView: NSView {
    private var previewLayer: AVCaptureVideoPreviewLayer?

    override func layout() {
        super.layout()
        previewLayer?.frame = bounds
    }

    func setSession(_ session: AVCaptureSession) {
        if previewLayer?.session === session { return }
        previewLayer?.removeFromSuperlayer()
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        layer.frame = bounds
        wantsLayer = true
        self.layer?.addSublayer(layer)
        previewLayer = layer
    }
}

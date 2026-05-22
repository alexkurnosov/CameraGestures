import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var vm = GestureViewModel()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // Camera preview
            if case .running = vm.pipelineState {
                CameraPreviewView(session: nil)   // session provided by HandsRecognizing
                    .ignoresSafeArea()
            }

            // Gesture overlay
            VStack {
                Spacer()

                if let gesture = vm.detectedGesture {
                    Text(gesture)
                        .font(.system(size: 48, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                        .padding(.horizontal, 32)
                        .padding(.vertical, 16)
                        .background(
                            RoundedRectangle(cornerRadius: 16)
                                .fill(Color.black.opacity(0.6))
                        )
                        .transition(.scale.combined(with: .opacity))
                        .animation(.spring(response: 0.3), value: gesture)
                }

                // Status row
                statusBadge
                    .padding(.bottom, 24)
            }
            .padding()
        }
        .frame(width: 640, height: 480)
        .task {
            if case .ready = vm.pipelineState {
                await vm.startCamera()
            }
        }
    }

    @ViewBuilder
    private var statusBadge: some View {
        switch vm.pipelineState {
        case .loading:
            Label("Loading model…", systemImage: "hourglass")
                .font(.caption)
                .foregroundColor(.yellow)
        case .ready:
            Label("Ready", systemImage: "checkmark.circle")
                .font(.caption)
                .foregroundColor(.green)
        case .running:
            Label("Recognizing gestures", systemImage: "waveform")
                .font(.caption)
                .foregroundColor(.white.opacity(0.7))
        case .error(let msg):
            Label(msg, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundColor(.red)
                .multilineTextAlignment(.center)
        }
    }
}

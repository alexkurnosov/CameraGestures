import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var vm = GestureViewModel()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            // Camera preview + landmark overlay
            if case .running = vm.pipelineState {
                CameraPreviewView(session: vm.previewSession)
                    .ignoresSafeArea()
                // Hand skeleton overlay — confirms whether the landmark model
                // is actually tracking the user's hand or hallucinating points.
                // Camera frame is 1280×720 (.medium preset); front-facing
                // preview is mirrored by AVCaptureVideoPreviewLayer by default.
                HandOverlayView(
                    landmarks:    vm.latestLandmarks,
                    cameraWidth:  1280,
                    cameraHeight: 720,
                    mirrored:     true)
                    .ignoresSafeArea()
            }

            VStack(spacing: 0) {
                Spacer()
                predictionsPanel
                    .padding(.horizontal, 16)
                    .padding(.bottom, 8)
                statusBadge
                    .padding(.bottom, 20)
            }
        }
        .frame(width: 640, height: 480)
        .task {
            if case .ready = vm.pipelineState {
                await vm.startCamera()
            }
        }
    }

    // MARK: - Predictions panel (reads vm directly via @StateObject)

    @ViewBuilder
    private var predictionsPanel: some View {
        let predictions = vm.latestPredictions
        let noneProb    = max(0.0, 1.0 - predictions.reduce(0) { $0 + $1.confidence })

        VStack(alignment: .leading, spacing: 6) {

            // ── Top row: best prediction ──────────────────────────────────
            // Only show the top gesture label when it's more confident than
            // "_none". This stops the panel flashing random low-confidence
            // gestures driven by noise cycles (false-positive palm detections
            // feeding garbage landmarks into the gesture model).
            HStack(alignment: .firstTextBaseline) {
                if let top = predictions.first {
                    let isDetected = top.confidence > noneProb
                    if isDetected && vm.detectedGesture != nil {
                        Image(systemName: "checkmark.seal.fill")
                            .foregroundColor(.green)
                            .font(.title3)
                    }
                    Text(isDetected ? top.gestureName : "—")
                        .font(.system(size: 30, weight: .bold, design: .rounded))
                        .foregroundColor(
                            !isDetected ? .white.opacity(0.25) :
                            top.confidence >= 0.65 ? .green :
                            top.confidence >= 0.35 ? .orange : .white.opacity(0.45))
                        .animation(.easeInOut(duration: 0.2), value: isDetected ? top.gestureName : "—")
                    Spacer()
                    Text(isDetected ? "\(Int(top.confidence * 100))%" : "—")
                        .font(.system(size: 24, weight: .semibold).monospacedDigit())
                        .foregroundColor(
                            !isDetected ? .white.opacity(0.35) :
                            top.confidence >= 0.65 ? .green : .white.opacity(0.65))
                } else {
                    Text("—")
                        .font(.system(size: 30, weight: .bold, design: .rounded))
                        .foregroundColor(.white.opacity(0.25))
                    Spacer()
                    Text("waiting…")
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.25))
                }
            }

            Divider().background(Color.white.opacity(0.25))

            // ── Per-gesture probability bars ──────────────────────────────
            ForEach(predictions, id: \.gestureId) { pred in
                ProbRow(
                    name:       pred.gestureName,
                    confidence: pred.confidence,
                    accent:     pred.confidence >= 0.65 ? .green
                              : pred.confidence >= 0.35 ? .orange
                              : .white
                )
            }

            // ── _none bar ─────────────────────────────────────────────────
            ProbRow(name: "_none", confidence: noneProb, accent: .gray)
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 14).fill(Color.black.opacity(0.78)))
    }

    // MARK: - Status badge

    @ViewBuilder
    private var statusBadge: some View {
        switch vm.pipelineState {
        case .loading:
            Label("Loading model…", systemImage: "hourglass")
                .font(.caption).foregroundColor(.yellow)
        case .ready:
            Label("Ready", systemImage: "checkmark.circle")
                .font(.caption).foregroundColor(.green)
        case .running:
            Label("Recognizing gestures", systemImage: "waveform")
                .font(.caption).foregroundColor(.white.opacity(0.7))
        case .error(let msg):
            Label(msg, systemImage: "exclamationmark.triangle")
                .font(.caption).foregroundColor(.red)
                .multilineTextAlignment(.center)
        }
    }
}

// MARK: - Probability Row

/// Leaf display component — `let` is correct here: it holds the value
/// it was constructed with and has no need to observe anything.
private struct ProbRow: View {
    let name:       String
    let confidence: Float
    let accent:     Color

    var body: some View {
        HStack(spacing: 8) {
            Text(name)
                .font(.caption.monospaced())
                .foregroundColor(accent.opacity(confidence > 0.04 ? 1.0 : 0.35))
                .frame(width: 120, alignment: .leading)
                .lineLimit(1)

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3)
                        .fill(Color.white.opacity(0.08))
                    RoundedRectangle(cornerRadius: 3)
                        .fill(accent.opacity(confidence > 0.04 ? 0.75 : 0.25))
                        .frame(width: geo.size.width * CGFloat(max(0, confidence)))
                        .animation(.easeOut(duration: 0.15), value: confidence)
                }
            }
            .frame(height: 10)

            Text("\(Int(confidence * 100))%")
                .font(.caption.monospacedDigit())
                .foregroundColor(accent.opacity(confidence > 0.04 ? 1.0 : 0.35))
                .frame(width: 36, alignment: .trailing)
        }
    }
}

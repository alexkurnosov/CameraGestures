// Scripted observer-effect measurement (session capture Stage 5).
//
// "Run observer-effect experiment" asks for the six session consents up front,
// warms the phone up to thermal state serious, then runs three rounds of
// capture off / capture without video / capture with video, 120 s each after
// a 5 s warm-up, with no input needed. "Run 1-minute capture check" records
// one session without video, for checking the capture-degraded event in a
// given light.
//
// Results: "[CG:Cadence]" lines from CameraViewModel (one "stop" line per run
// holds its counters) and "[CG:Experiment]" lines from here, all in
// Documents/cadence.log. Kept for re-measuring after a library change, so it
// is not part of the temporary SessionCaptureDebugPanel.

#if CG_SESSION_CAPTURE

import SwiftUI
import Combine
import UIKit
import CameraGestures

@MainActor
final class ObserverEffectExperiment: ObservableObject {

    enum Capture: String {
        case off
        case noVideo = "no_video"
        case video
    }

    struct Stopped: Error, CustomStringConvertible {
        let description: String
    }

    static let rounds         = 3
    static let warmUpSeconds  = 5
    static let measureSeconds = 120
    static let checkSeconds   = 60

    @Published private(set) var status = ""
    @Published private(set) var isRunning = false

    private var task: Task<Void, Never>?

    func runExperiment(viewModel: CameraViewModel, recognizer: HandGestureRecognizing, settings: AppSettings) {
        run(viewModel: viewModel, recognizer: recognizer) { [self] in
            var consents: [Capture: [SessionCaptureConsent]] = [:]
            for capture in [Capture.noVideo, .video] {
                for _ in 0..<Self.rounds {
                    status = "Asking for consent"
                    guard let consent = await SessionCaptureConsent.request(includesVideo: capture == .video) else {
                        throw Stopped(description: "consent declined")
                    }
                    consents[capture, default: []].append(consent)
                }
            }
            log("experiment_start", ["rounds": Self.rounds, "measure_s": Self.measureSeconds])
            for round in 1...Self.rounds {
                for capture in [Capture.off, .noVideo, .video] {
                    try await waitForSerious(viewModel)
                    try await measure(label: "r\(round)-\(capture.rawValue)", capture: capture,
                                      consent: consents[capture]?.removeFirst(),
                                      seconds: Self.measureSeconds, stopWhenCritical: true,
                                      viewModel: viewModel, recognizer: recognizer, settings: settings)
                }
            }
            log("experiment_end")
        }
    }

    func runCaptureCheck(viewModel: CameraViewModel, recognizer: HandGestureRecognizing, settings: AppSettings) {
        run(viewModel: viewModel, recognizer: recognizer) { [self] in
            status = "Asking for consent"
            guard let consent = await SessionCaptureConsent.request(includesVideo: false) else {
                throw Stopped(description: "consent declined")
            }
            try await measure(label: "check", capture: .noVideo, consent: consent,
                              seconds: Self.checkSeconds, stopWhenCritical: false,
                              viewModel: viewModel, recognizer: recognizer, settings: settings)
        }
    }

    func cancel() {
        task?.cancel()
    }

    // MARK: - Steps

    private func run(viewModel: CameraViewModel, recognizer: HandGestureRecognizing,
                     _ body: @escaping @MainActor () async throws -> Void) {
        guard !isRunning else { return }
        isRunning = true
        // The phone is left alone for half an hour; a lock would stop the camera.
        UIApplication.shared.isIdleTimerDisabled = true
        task = Task { @MainActor in
            do {
                try await body()
                status = "Done"
            } catch {
                status = "Stopped: \(error)"
                log("experiment_stopped", ["reason": "\(error)"])
            }
            if viewModel.isRecognitionActive { viewModel.stopAll() }
            // A session armed for a run that never started.
            SessionRecorder(recognizer: recognizer).stop()
            viewModel.cadenceLabel = nil
            UIApplication.shared.isIdleTimerDisabled = false
            isRunning = false
        }
    }

    /// Returns once the phone is at thermal state serious. Below it, keeps a
    /// prediction run going to heat the phone; at critical, stops the camera
    /// until the phone cools back down.
    private func waitForSerious(_ viewModel: CameraViewModel) async throws {
        var lastState: ProcessInfo.ThermalState?
        while true {
            let state = ProcessInfo.processInfo.thermalState
            if state == .serious { return }
            if state != lastState {
                log("thermal_wait", ["thermal": CadenceLog.name(state)])
                lastState = state
            }
            if state == .critical {
                if viewModel.isRecognitionActive { viewModel.stopAll() }
                status = "Critical: camera off until serious"
            } else {
                if !viewModel.isRecognitionActive {
                    viewModel.cadenceLabel = "warmup"
                    viewModel.startRecognition()
                }
                status = "Warming up (\(CadenceLog.name(state)))"
            }
            try await Task.sleep(nanoseconds: 5_000_000_000)
        }
    }

    private func measure(label: String, capture: Capture, consent: SessionCaptureConsent?,
                         seconds: Int, stopWhenCritical: Bool,
                         viewModel: CameraViewModel, recognizer: HandGestureRecognizing,
                         settings: AppSettings) async throws {
        if viewModel.isRecognitionActive { viewModel.stopAll() }

        var session: Any = NSNull()
        if let consent {
            session = try SessionRecorder(recognizer: recognizer)
                .start(consent: consent, modelFiles: settings.sessionModelFiles())
                .lastPathComponent
        }
        viewModel.cadenceLabel = label
        log("run_start", ["label": label, "capture": capture.rawValue, "session": session])

        viewModel.startRecognition()
        var waited = 0
        while !viewModel.isRecognitionActive && waited < 20 {
            try await Task.sleep(nanoseconds: 500_000_000)
            waited += 1
        }
        guard viewModel.isRecognitionActive else {
            throw Stopped(description: "prediction did not start for \(label)")
        }

        status = "\(label): warm-up"
        try await Task.sleep(nanoseconds: UInt64(Self.warmUpSeconds) * 1_000_000_000)
        viewModel.resetFrameRateStats()

        var stoppedCritical = false
        for second in 0..<seconds {
            status = "\(label): \(second) of \(seconds) s"
            try await Task.sleep(nanoseconds: 1_000_000_000)
            if stopWhenCritical && ProcessInfo.processInfo.thermalState == .critical {
                stoppedCritical = true
                break
            }
        }
        viewModel.stopAll() // writes the run's "stop" cadence line
        viewModel.cadenceLabel = nil
        log("run_end", ["label": label, "stopped_critical": stoppedCritical])
    }

    private func log(_ event: String, _ fields: [String: Any] = [:]) {
        var fields = fields
        fields["event"] = event
        CadenceLog.shared.write("CG:Experiment", fields)
    }
}

/// The two buttons and the progress line, shown in the frame cadence panel.
struct ObserverEffectExperimentControls: View {
    @ObservedObject var viewModel: CameraViewModel
    let recognizer: HandGestureRecognizing
    let appSettings: AppSettings

    @StateObject private var experiment = ObserverEffectExperiment()

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button("Run observer-effect experiment") {
                experiment.runExperiment(viewModel: viewModel, recognizer: recognizer, settings: appSettings)
            }
            .disabled(experiment.isRunning || viewModel.isRecognitionActive)
            Button("Run 1-minute capture check") {
                experiment.runCaptureCheck(viewModel: viewModel, recognizer: recognizer, settings: appSettings)
            }
            .disabled(experiment.isRunning || viewModel.isRecognitionActive)
            if experiment.isRunning {
                Button("Stop experiment", role: .destructive) { experiment.cancel() }
            }
            if !experiment.status.isEmpty {
                Text(experiment.status)
                    .font(.caption.monospaced())
                    .foregroundColor(.secondary)
            }
        }
        .font(.caption)
        .buttonStyle(.bordered)
    }
}

#endif

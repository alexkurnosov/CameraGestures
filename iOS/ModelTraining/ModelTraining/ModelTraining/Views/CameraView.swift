import SwiftUI
import AVFoundation
import GestureModelModule
import HandGestureTypes
import HandGestureRecognizingFramework

struct CameraView: View {
    @EnvironmentObject var gestureRecognizer: GestureRecognizerWrapper
    @EnvironmentObject var appSettings: AppSettings
    @EnvironmentObject var trainingDataManager: TrainingDataManager
    @EnvironmentObject var gestureRegistry: GestureRegistry
    @EnvironmentObject var apiClient: GestureModelAPIClient

    @StateObject private var seriesCoordinator = TrainingSeriesCoordinator()

    // Recognition state
    @State private var isRecognitionActive = false
    @State private var currentGesture: DetectedGesture?
    @State private var recentGestures: [DetectedGesture] = []
    @State private var recognitionHandPoints: [Point3D] = []
    @State private var stats = GestureRecognizingStats()

    // Permissions & banners
    @State private var showingPermissionAlert = false
    @State private var cameraPermissionGranted = false
    @State private var showModelNotTrainedBanner = false

    // Training series config
    @State private var captureWindow: TimeInterval = 2.0
    @State private var pauseInterval: TimeInterval = 2.0

    // Which hand tracking points to show on the preview
    private var displayPoints: [Point3D] {
        seriesCoordinator.isRunning
            ? seriesCoordinator.handTrackingPoints
            : recognitionHandPoints
    }

    private var previewIsActive: Bool { isRecognitionActive || seriesCoordinator.isRunning }

    private var isModelTrained: Bool {
#if DEBUG
        //Test:
        return true
#else
#error ("test")
#endif
        guard let path = appSettings.modelConfig.modelPath else { return false }
        return FileManager.default.fileExists(atPath: path)
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Model not trained banner
                if showModelNotTrainedBanner {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("No model downloaded yet. Go to Training → Update Model.")
                            .font(.caption)
                            .foregroundColor(.primary)
                        Spacer()
                        Button {
                            showModelNotTrainedBanner = false
                        } label: {
                            Image(systemName: "xmark")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(10)
                    .background(Color.orange.opacity(0.15))
                    .cornerRadius(8)
                    .padding(.horizontal)
                    .padding(.top, 4)
                }

                // Camera preview with capture overlay
                ZStack {
                    CameraPreviewView(
                        handTrackingPoints: .constant(displayPoints),
                        isActive: .constant(previewIsActive)
                    )
                    .cornerRadius(12)

                    capturePhaseOverlay
                }
                .frame(maxHeight: 400)
                .padding(.horizontal)
                .padding(.top, 8)

                ScrollView {
                    VStack(spacing: 12) {
                        // Recognised gesture (prediction mode)
                        if let currentGesture, !seriesCoordinator.isRunning {
                            CurrentGestureView(gesture: currentGesture)
                                .padding()
                                .background(Color.blue.opacity(0.1))
                                .cornerRadius(8)
                        }

                        // Controls row
                        controlsSection

                        // Recent gestures list (prediction mode)
                        if !recentGestures.isEmpty && !seriesCoordinator.isRunning {
                            recentGesturesSection
                        }

                        // Send to server (shown when there are pending examples)
                        if !trainingDataManager.pendingExamples.isEmpty || trainingDataManager.isSendingToServer {
                            sendToServerSection
                        }
                    }
                    .padding()
                }
            }
            .navigationTitle("Camera")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Clear") { clearGestures() }
                        .disabled(recentGestures.isEmpty)
                }
            }
        }
        .onAppear {
            checkCameraPermission()
            setupGestureCallbacks()
            appSettings.updateModelConfig()
            showModelNotTrainedBanner = !isModelTrained
            if trainingDataManager.selectedGesture == nil {
                trainingDataManager.selectedGesture = gestureRegistry.gestures.first
            }
            trainingDataManager.apiClient = apiClient
        }
        .onChange(of: gestureRegistry.gestures) { gestures in
            if let current = trainingDataManager.selectedGesture, !gestures.contains(current) {
                trainingDataManager.selectedGesture = gestures.first
            } else if trainingDataManager.selectedGesture == nil {
                trainingDataManager.selectedGesture = gestures.first
            }
        }
        .alert("Camera Permission Required", isPresented: $showingPermissionAlert) {
            Button("Settings") { openAppSettings() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Please enable camera access in Settings to use gesture recognition.")
        }
        .onDisappear {
            seriesCoordinator.stop()
        }
    }

    // MARK: - Capture Phase Overlay

    @ViewBuilder
    private var capturePhaseOverlay: some View {
        switch seriesCoordinator.phase {
        case .idle:
            EmptyView()

        case .countdown(let remaining):
            VStack(spacing: 6) {
                Text("Get ready!")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.white)
                Text("\(remaining)")
                    .font(.system(size: 72, weight: .bold, design: .rounded))
                    .foregroundColor(.yellow)
            }
            .padding(24)
            .background(.black.opacity(0.55))
            .cornerRadius(16)

        case .recording:
            HStack(spacing: 10) {
                Circle()
                    .fill(Color.red)
                    .frame(width: 12, height: 12)
                    .opacity(recPulse ? 1 : 0.25)
                    .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: recPulse)
                    .onAppear { recPulse = true }
                    .onDisappear { recPulse = false }
                Text("REC — Perform gesture")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.white)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 10)
            .background(Color.red.opacity(0.75))
            .cornerRadius(12)
            .padding(.bottom, 12)
            .frame(maxHeight: .infinity, alignment: .bottom)

        case .pause(let remaining):
            VStack(spacing: 4) {
                Text("Next capture in")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.75))
                Text("\(remaining)s")
                    .font(.system(size: 40, weight: .semibold, design: .rounded))
                    .foregroundColor(.white)
                Text("Captured: \(seriesCoordinator.capturedCount)")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.75))
            }
            .padding(18)
            .background(.black.opacity(0.55))
            .cornerRadius(14)
        }
    }

    @State private var recPulse = false

    // MARK: - Controls Section

    private var controlsSection: some View {
        VStack(spacing: 10) {
            // Gesture picker (always visible)
            gesturePicker

            if seriesCoordinator.isRunning {
                // Running: show capture count + Stop button
                HStack(spacing: 10) {
                    Image(systemName: "film.stack")
                        .foregroundColor(.blue)
                    Text("Captured: \(seriesCoordinator.capturedCount)")
                        .font(.subheadline.weight(.medium))
                    Spacer()
                    phasePill
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.blue.opacity(0.08))
                .cornerRadius(10)

                stopButton

            } else if isRecognitionActive {
                // Prediction running: Stop button only
                stopButton

            } else {
                // Idle: timing config + both start buttons
                timingConfig

                HStack(spacing: 12) {
                    startPredictionButton
                    startTrainingButton
                }
            }
        }
    }

    private var gesturePicker: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Training gesture")
                .font(.caption)
                .foregroundColor(.secondary)

            if gestureRegistry.gestures.isEmpty {
                Text("No gestures defined — add them in the Training tab")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.vertical, 4)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(gestureRegistry.gestures) { gesture in
                            let isSelected = trainingDataManager.selectedGesture == gesture
                            Button {
                                if !seriesCoordinator.isRunning {
                                    trainingDataManager.selectedGesture = gesture
                                }
                            } label: {
                                Text(gesture.name)
                                    .font(.caption.weight(isSelected ? .semibold : .regular))
                                    .foregroundColor(isSelected ? .white : .primary)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 6)
                                    .background(isSelected ? Color.blue : Color.gray.opacity(0.15))
                                    .clipShape(Capsule())
                            }
                            .buttonStyle(.plain)
                            .disabled(seriesCoordinator.isRunning)
                        }
                    }
                }
            }
        }
    }

    private var timingConfig: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Capture")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                HStack(spacing: 4) {
                    Text("\(Int(captureWindow))s")
                        .font(.subheadline.weight(.medium))
                        .frame(width: 28, alignment: .leading)
                    Stepper("", value: $captureWindow, in: 1...5, step: 1)
                        .labelsHidden()
                }
            }
            Divider().frame(height: 32)
            VStack(alignment: .leading, spacing: 2) {
                Text("Pause")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                HStack(spacing: 4) {
                    Text("\(Int(pauseInterval))s")
                        .font(.subheadline.weight(.medium))
                        .frame(width: 28, alignment: .leading)
                    Stepper("", value: $pauseInterval, in: 2...15, step: 1)
                        .labelsHidden()
                }
            }
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.gray.opacity(0.07))
        .cornerRadius(10)
    }

    private var startPredictionButton: some View {
        Button(action: startRecognition) {
            HStack(spacing: 6) {
                Image(systemName: "play.fill")
                Text("Start Prediction")
            }
            .font(.subheadline.weight(.semibold))
            .foregroundColor(.white)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity)
            .background(isModelTrained && cameraPermissionGranted ? Color.green : Color.gray)
            .cornerRadius(10)
        }
        .disabled(!isModelTrained || !cameraPermissionGranted)
    }

    private var startTrainingButton: some View {
        Button(action: startTrainingSeries) {
            HStack(spacing: 6) {
                Image(systemName: "record.circle")
                Text("Start Training")
            }
            .font(.subheadline.weight(.semibold))
            .foregroundColor(.white)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity)
            .background(canStartTraining ? Color.red : Color.gray)
            .cornerRadius(10)
        }
        .disabled(!canStartTraining)
    }

    private var stopButton: some View {
        Button(action: stopAll) {
            HStack(spacing: 6) {
                Image(systemName: "stop.fill")
                Text("Stop")
            }
            .font(.subheadline.weight(.semibold))
            .foregroundColor(.white)
            .padding(.vertical, 12)
            .frame(maxWidth: .infinity)
            .background(Color.orange)
            .cornerRadius(10)
        }
    }

    @ViewBuilder
    private var phasePill: some View {
        switch seriesCoordinator.phase {
        case .countdown: Text("Countdown").font(.caption).foregroundColor(.orange)
        case .recording: Text("Recording").font(.caption.weight(.semibold)).foregroundColor(.red)
        case .pause:     Text("Pausing").font(.caption).foregroundColor(.secondary)
        case .idle:      EmptyView()
        }
    }

    // MARK: - Recent Gestures

    private var recentGesturesSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Recent Gestures")
                .font(.headline)

            ScrollView {
                LazyVStack(spacing: 4) {
                    ForEach(recentGestures.reversed().indices, id: \.self) { index in
                        RecentGestureRow(gesture: recentGestures.reversed()[index])
                    }
                }
            }
            .frame(maxHeight: 150)
        }
    }

    // MARK: - Send to Server

    private var sendToServerSection: some View {
        VStack(spacing: 6) {
            Button(action: { trainingDataManager.sendPendingToServer() }) {
                HStack {
                    if trainingDataManager.isSendingToServer {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .scaleEffect(0.8)
                            .tint(.white)
                    } else {
                        Image(systemName: "arrow.up.circle.fill")
                    }
                    Text(trainingDataManager.isSendingToServer
                         ? "Sending…"
                         : "Send to Server (\(trainingDataManager.pendingExamples.count))")
                }
                .font(.subheadline.weight(.semibold))
                .foregroundColor(.white)
                .padding(.vertical, 12)
                .frame(maxWidth: .infinity)
                .background(!trainingDataManager.pendingExamples.isEmpty && !trainingDataManager.isSendingToServer
                             ? Color.blue : Color.gray)
                .cornerRadius(10)
            }
            .disabled(trainingDataManager.pendingExamples.isEmpty || trainingDataManager.isSendingToServer)

            uploadStatusRow

            NavigationLink(destination: HandFilmsView()
                .environmentObject(trainingDataManager)
                .environmentObject(gestureRegistry)
            ) {
                HStack(spacing: 6) {
                    Image(systemName: "film.stack")
                    Text("View Collected Films (\(trainingDataManager.trainingExamples.count))")
                        .font(.subheadline)
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .foregroundColor(.primary)
                .padding(.vertical, 8)
                .padding(.horizontal, 12)
                .background(Color.gray.opacity(0.08))
                .cornerRadius(10)
            }
        }
    }

    @ViewBuilder
    private var uploadStatusRow: some View {
        switch trainingDataManager.uploadState {
        case .idle:
            EmptyView()
        case .uploading:
            HStack(spacing: 6) {
                ProgressView().scaleEffect(0.75)
                Text("Uploading…").font(.caption).foregroundColor(.secondary)
            }
        case .uploaded(let total):
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill").foregroundColor(.green).font(.caption)
                Text("Sent — server has \(total) example(s) for this gesture")
                    .font(.caption).foregroundColor(.secondary)
            }
        case .failed(let msg):
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.circle.fill").foregroundColor(.orange).font(.caption)
                Text("Upload failed: \(msg)").font(.caption).foregroundColor(.secondary).lineLimit(2)
            }
        }
    }

    // MARK: - Computed Properties

    private var canStartTraining: Bool {
        cameraPermissionGranted && trainingDataManager.selectedGesture != nil
    }

    // MARK: - Actions

    private func startRecognition() {
        guard isModelTrained else {
            showModelNotTrainedBanner = true
            return
        }
        Task {
            do {
                try await gestureRecognizer.recognizer.start()
                await MainActor.run { isRecognitionActive = true }
            } catch {
                print("Failed to start recognition: \(error)")
            }
        }
    }

    private func startTrainingSeries() {
        guard let gesture = trainingDataManager.selectedGesture else { return }
        trainingDataManager.startDataCollection(for: gesture)

        Task {
            if !gestureRecognizer.recognizer.isActive {
                do {
                    try await gestureRecognizer.recognizer.start()
                } catch {
                    print("Failed to start recognizer for training: \(error)")
                    trainingDataManager.stopDataCollection()
                    return
                }
            }
            seriesCoordinator.start(using: gestureRecognizer.recognizer, captureWindow: captureWindow, pauseInterval: pauseInterval) { film in
                let example = TrainingExample(
                    handfilm: film,
                    gestureId: gesture.id,
                    userId: "current_user",
                    sessionId: UUID().uuidString
                )
                trainingDataManager.addTrainingExample(example)
            }
        }
    }

    private func stopAll() {
        if isRecognitionActive {
            gestureRecognizer.recognizer.stop()
            isRecognitionActive = false
        }
        if seriesCoordinator.isRunning {
            seriesCoordinator.stop()
            trainingDataManager.stopDataCollection()
        }
    }

    private func clearGestures() {
        recentGestures.removeAll()
        currentGesture = nil
        gestureRecognizer.recognizer.clearHistory()
    }

    // MARK: - Setup

    private func setupGestureCallbacks() {
        gestureRecognizer.recognizer.gestureDetectionCallback = { gesture in
            DispatchQueue.main.async {
                currentGesture = gesture
                recentGestures.append(gesture)
                if recentGestures.count > 50 { recentGestures.removeFirst() }
            }
        }

        gestureRecognizer.recognizer.handTrackingUpdateCallback = { handshot in
            let callbackTime = Date().timeIntervalSince1970
            DispatchQueue.main.async {
                //print(String(format: "<<--render_timing-->> handshot_received=%.4f", callbackTime))
                recognitionHandPoints = handshot.landmarks
            }
        }

        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            if isRecognitionActive {
                stats = gestureRecognizer.recognizer.getStatistics()
            }
        }
    }

    private func checkCameraPermission() {
        Task {
            let permission = await HandsRecognizing.requestCameraPermission()
            await MainActor.run {
                cameraPermissionGranted = permission
                if !permission { showingPermissionAlert = true }
            }
        }
    }

    private func openAppSettings() {
        if let url = URL(string: UIApplication.openSettingsURLString) {
            UIApplication.shared.open(url)
        }
    }
}

// MARK: - Camera Preview View

struct CameraPreviewView: UIViewControllerRepresentable {
    @Binding var handTrackingPoints: [Point3D]
    @Binding var isActive: Bool
    
    func makeUIViewController(context: Context) -> CameraPreviewController {
        return CameraPreviewController()
    }
    
    func updateUIViewController(_ uiViewController: CameraPreviewController, context: Context) {
        uiViewController.updateHandTrackingPoints(handTrackingPoints)
        
        if isActive {
            uiViewController.startCamera()
        } else {
            uiViewController.stopCamera()
        }
    }
}

class CameraPreviewController: UIViewController {
    private var captureSession: AVCaptureSession?
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var handTrackingOverlay: HandTrackingOverlayView?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupOverlay()
    }
    
    private func setupOverlay() {
        handTrackingOverlay = HandTrackingOverlayView()
        handTrackingOverlay?.backgroundColor = .clear
        
        if let overlay = handTrackingOverlay {
            view.addSubview(overlay)
            overlay.translatesAutoresizingMaskIntoConstraints = false
            NSLayoutConstraint.activate([
                overlay.topAnchor.constraint(equalTo: view.topAnchor),
                overlay.leadingAnchor.constraint(equalTo: view.leadingAnchor),
                overlay.trailingAnchor.constraint(equalTo: view.trailingAnchor),
                overlay.bottomAnchor.constraint(equalTo: view.bottomAnchor)
            ])
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }
    
    func startCamera() {
        DispatchQueue.global(qos: .background).async { [weak self] in
            self?.captureSession?.startRunning()
        }
    }
    
    func stopCamera() {
        DispatchQueue.global(qos: .background).async { [weak self] in
            self?.captureSession?.stopRunning()
        }
    }
    
    func updateHandTrackingPoints(_ points: [Point3D]) {
        DispatchQueue.main.async { [weak self] in
            self?.handTrackingOverlay?.updatePoints(points)
        }
    }
}

// MARK: - Hand Skeleton SwiftUI View

/// Reusable SwiftUI wrapper around HandTrackingOverlayView.
/// Pass 21 Point3D landmarks; pass an empty array to show a blank canvas.
struct HandSkeletonView: UIViewRepresentable {
    var points: [Point3D]

    func makeUIView(context: Context) -> HandTrackingOverlayView {
        let view = HandTrackingOverlayView()
        view.backgroundColor = .clear
        return view
    }

    func updateUIView(_ uiView: HandTrackingOverlayView, context: Context) {
        uiView.updatePoints(points)
    }
}

// MARK: - Hand Tracking Overlay

class HandTrackingOverlayView: UIView {
    private var handPoints: [Point3D] = []
    private var pointsReceivedTime: TimeInterval = 0
    private var framesRendered: Int = 0

    func updatePoints(_ points: [Point3D]) {
        handPoints = points
        pointsReceivedTime = Date().timeIntervalSince1970
        setNeedsDisplay()
    }
    
    override func draw(_ rect: CGRect) {
        let drawTime = Date().timeIntervalSince1970
        guard let context = UIGraphicsGetCurrentContext(), !handPoints.isEmpty else { return }
        framesRendered += 1
        //print(String(format: "<<--render_timing-->> frame=%d  draw_start=%.4f  render_lag=%.4f s", framesRendered, drawTime, drawTime - pointsReceivedTime))
        
        context.setStrokeColor(UIColor.green.cgColor)
        context.setLineWidth(2.0)
        
        // Draw hand landmarks
        for point in handPoints {
            let x = CGFloat(point.x) * rect.width
            let y = CGFloat(point.y) * rect.height
            let adjustedPoint = CGPoint(x: x, y: y)
            
            context.addEllipse(in: CGRect(
                x: adjustedPoint.x - 3,
                y: adjustedPoint.y - 3,
                width: 6,
                height: 6
            ))
        }
        
        context.strokePath()
        
        drawHandConnections(context: context, rect: rect)
    }
    
    private func drawHandConnections(context: CGContext, rect: CGRect) {
        guard handPoints.count >= 21 else { return }
        
        let connections: [(Int, Int)] = [
            // Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            // Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            // Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            // Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            // Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        context.setStrokeColor(UIColor.blue.cgColor)
        context.setLineWidth(1.0)
        
        for (start, end) in connections {
            if start < handPoints.count && end < handPoints.count {
                let startPoint = handPoints[start]
                let endPoint = handPoints[end]
                
                let startCG = CGPoint(
                    x: CGFloat(startPoint.x) * rect.width,
                    y: CGFloat(startPoint.y) * rect.height
                )
                let endCG = CGPoint(
                    x: CGFloat(endPoint.x) * rect.width,
                    y: CGFloat(endPoint.y) * rect.height
                )
                
                context.move(to: startCG)
                context.addLine(to: endCG)
            }
        }
        
        context.strokePath()
    }
}

// MARK: - Current Gesture View

struct CurrentGestureView: View {
    let gesture: DetectedGesture
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: "hand.raised.fill")
                    .foregroundColor(.blue)
                
                Text(gesture.prediction.gestureName)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Spacer()
                
                Text("\(Int(gesture.prediction.confidence * 100))%")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.green)
            }
            
            Text("Latency: \(String(format: "%.1f", gesture.processingLatency * 1000))ms")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Recent Gesture Row

struct RecentGestureRow: View {
    let gesture: DetectedGesture
    
    var body: some View {
        HStack {
            Circle()
                .fill(confidenceColor)
                .frame(width: 8, height: 8)
            
            Text(gesture.prediction.gestureName)
                .font(.system(.body, design: .monospaced))
            
            Spacer()
            
            Text("\(Int(gesture.prediction.confidence * 100))%")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text(timeAgo)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal)
        .padding(.vertical, 2)
    }
    
    private var confidenceColor: Color {
        if gesture.prediction.confidence > 0.8 {
            return .green
        } else if gesture.prediction.confidence > 0.6 {
            return .orange
        } else {
            return .red
        }
    }
    
    private var timeAgo: String {
        let interval = Date().timeIntervalSince1970 - gesture.detectionTimestamp
        if interval < 60 {
            return "\(Int(interval))s"
        } else {
            return "\(Int(interval / 60))m"
        }
    }
}

// MARK: - Preview

struct CameraView_Previews: PreviewProvider {
    static var previews: some View {
        CameraView()
            .environmentObject(GestureRecognizerWrapper(recognizer: HandGestureRecognizing()))
            .environmentObject(AppSettings())
    }
}

import SwiftUI
import HandGestureTypes
import GestureModelModule
import HandGestureRecognizingFramework

struct SettingsView: View {
    @EnvironmentObject var appSettings: AppSettings
    @EnvironmentObject var gestureRecognizer: GestureRecognizerWrapper
    @EnvironmentObject var apiClient: GestureModelAPIClient

    @State private var showingResetAlert = false
    @State private var showingAbout = false
    
    var body: some View {
        NavigationView {
            Form {
                // Server Settings Section
                serverSection

                // Camera Settings Section
                cameraSettingsSection
                
                // Recognition Settings Section
                recognitionSettingsSection
                
                // UI Settings Section
                uiSettingsSection
                
                // Model Settings Section
                modelSettingsSection
                
                // Data Management Section
                dataManagementSection
                
                // About Section
                aboutSection
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
        }
        .alert("Reset Settings", isPresented: $showingResetAlert) {
            Button("Reset", role: .destructive) {
                resetAllSettings()
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This will reset all settings to their default values. This action cannot be undone.")
        }
        .sheet(isPresented: $showingAbout) {
            AboutView()
        }
    }
    
    // MARK: - Settings Sections

    private var serverSection: some View {
        Section("Server") {
            HStack {
                Image(systemName: "network")
                    .foregroundColor(.blue)
                    .frame(width: 24)
                TextField("Server URL", text: Binding(
                    get: { apiClient.baseURL.absoluteString },
                    set: { if let url = URL(string: $0) { apiClient.baseURL = url } }
                ))
                .keyboardType(.URL)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)
            }

            HStack {
                Image(systemName: "key.fill")
                    .foregroundColor(.orange)
                    .frame(width: 24)
                SecureField("Registration Token", text: $apiClient.registrationToken)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
            }

            Button("Re-register Device") {
                apiClient.clearToken()
            }
            .foregroundColor(.red)
        }
    }

    private var cameraSettingsSection: some View {
        Section("Camera") {
            // Preferred Camera
            HStack {
                Image(systemName: "camera.fill")
                    .foregroundColor(.blue)
                    .frame(width: 24)
                
                Text("Preferred Camera")
                
                Spacer()
                
                Picker("Camera", selection: $appSettings.preferredCamera) {
                    Text("Front").tag(1)
                    Text("Back").tag(0)
                }
                .pickerStyle(.segmented)
                .frame(width: 120)
            }
            
            // Target FPS
            HStack {
                Image(systemName: "speedometer")
                    .foregroundColor(.orange)
                    .frame(width: 24)
                
                VStack(alignment: .leading) {
                    Text("Target FPS")
                    Text("\(appSettings.targetFPS) fps")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Stepper("", value: $appSettings.targetFPS, in: 15...60, step: 15)
                    .labelsHidden()
            }
            
            // Detection Settings
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: "hand.raised.fill")
                        .foregroundColor(.green)
                        .frame(width: 24)
                    
                    Text("Detection Confidence")
                    
                    Spacer()
                    
                    Text("\(Int(appSettings.cameraConfig.minDetectionConfidence * 100))%")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Slider(
                    value: Binding(
                        get: { appSettings.cameraConfig.minDetectionConfidence },
                        set: { newValue in
                            appSettings.cameraConfig = HandsRecognizingConfig(
                                cameraIndex: appSettings.cameraConfig.cameraIndex,
                                targetFPS: appSettings.cameraConfig.targetFPS,
                                detectBothHands: appSettings.cameraConfig.detectBothHands,
                                minDetectionConfidence: Float(newValue),
                                minTrackingConfidence: appSettings.cameraConfig.minTrackingConfidence
                            )
                        }
                    ),
                    in: 0.1...1.0,
                    step: 0.1
                )
            }
        }
    }
    
    private var recognitionSettingsSection: some View {
        Section("Recognition") {
            // Confidence Threshold
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: "brain.head.profile")
                        .foregroundColor(.purple)
                        .frame(width: 24)
                    
                    Text("Confidence Threshold")
                    
                    Spacer()
                    
                    Text("\(Int(appSettings.confidenceThreshold * 100))%")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Slider(
                    value: $appSettings.confidenceThreshold,
                    in: 0.3...0.95,
                    step: 0.05
                )
            }
            
            // Model Backend
            HStack {
                Image(systemName: "cpu.fill")
                    .foregroundColor(.red)
                    .frame(width: 24)
                
                Text("Model Backend")
                
                Spacer()
                
                // TODO: implement backend selection
                /*
                Picker("Backend", selection: $appSettings.modelConfig.backendType) {
                    ForEach(BackendType.allCases, id: \.self) { backend in
                        Text(backend.displayName).tag(backend)
                    }
                }
                .pickerStyle(.menu)*/
            }
            
            // Real-time Processing Toggle
            HStack {
                Image(systemName: "bolt.fill")
                    .foregroundColor(.yellow)
                    .frame(width: 24)
                
                Text("Real-time Processing")
                
                Spacer()
                
                Toggle("", isOn: .constant(true))
                    .labelsHidden()
                    .disabled(true) // Always enabled in this version
            }
        }
    }
    
    private var uiSettingsSection: some View {
        Section("Interface") {
            // Color Scheme
            HStack {
                Image(systemName: "paintbrush.fill")
                    .foregroundColor(.indigo)
                    .frame(width: 24)
                
                Text("Appearance")
                
                Spacer()
                
                Picker("Appearance", selection: $appSettings.colorScheme) {
                    Text("System").tag(nil as ColorScheme?)
                    Text("Light").tag(ColorScheme.light as ColorScheme?)
                    Text("Dark").tag(ColorScheme.dark as ColorScheme?)
                }
                .pickerStyle(.menu)
            }
            
            // Haptic Feedback
            HStack {
                Image(systemName: "iphone.and.arrow.forward")
                    .foregroundColor(.pink)
                    .frame(width: 24)
                
                Text("Haptic Feedback")
                
                Spacer()
                
                Toggle("", isOn: $appSettings.enableHapticFeedback)
                    .labelsHidden()
            }
            
            // Debug Info
            HStack {
                Image(systemName: "info.circle.fill")
                    .foregroundColor(.cyan)
                    .frame(width: 24)
                
                Text("Show Debug Info")
                
                Spacer()
                
                Toggle("", isOn: $appSettings.showDebugInfo)
                    .labelsHidden()
            }
        }
    }
    
    private var modelSettingsSection: some View {
        Section("Model") {
            // Current Model Info
            NavigationLink(destination: ModelInfoView()) {
                HStack {
                    Image(systemName: "brain.fill")
                        .foregroundColor(.purple)
                        .frame(width: 24)
                    
                    VStack(alignment: .leading) {
                        Text("Current Model")
                        Text("Mock Model v1.0")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            
            // Model Performance
            HStack {
                Image(systemName: "gauge.medium")
                    .foregroundColor(.green)
                    .frame(width: 24)
                
                VStack(alignment: .leading) {
                    Text("Performance Mode")
                    Text("Balanced")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Picker("Performance", selection: .constant("balanced")) {
                    Text("Fast").tag("fast")
                    Text("Balanced").tag("balanced")
                    Text("Accurate").tag("accurate")
                }
                .pickerStyle(.menu)
                .disabled(true) // Not implemented in stub
            }
        }
    }
    
    private var dataManagementSection: some View {
        Section("Data") {
            // Storage Info
            HStack {
                Image(systemName: "internaldrive.fill")
                    .foregroundColor(.gray)
                    .frame(width: 24)
                
                VStack(alignment: .leading) {
                    Text("Storage Used")
                    Text("~2.5 MB")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                Button("Manage") {
                    // TODO: Show storage management
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            
            // Export Data
            Button(action: exportAllData) {
                HStack {
                    Image(systemName: "square.and.arrow.up.fill")
                        .foregroundColor(.blue)
                        .frame(width: 24)
                    
                    Text("Export Training Data")
                        .foregroundColor(.blue)
                }
            }
            
            // Clear Data
            Button(action: { showingResetAlert = true }) {
                HStack {
                    Image(systemName: "trash.fill")
                        .foregroundColor(.red)
                        .frame(width: 24)
                    
                    Text("Clear All Data")
                        .foregroundColor(.red)
                }
            }
        }
    }
    
    private var aboutSection: some View {
        Section("About") {
            Button(action: { showingAbout = true }) {
                HStack {
                    Image(systemName: "info.circle.fill")
                        .foregroundColor(.blue)
                        .frame(width: 24)
                    
                    Text("About CameraGestures")
                        .foregroundColor(.blue)
                }
            }
            
            // Version Info
            HStack {
                Image(systemName: "number.circle.fill")
                    .foregroundColor(.gray)
                    .frame(width: 24)
                
                Text("Version")
                
                Spacer()
                
                Text("1.0.0 (1)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            // System Info
            HStack {
                Image(systemName: "iphone")
                    .foregroundColor(.gray)
                    .frame(width: 24)
                
                Text("iOS Version")
                
                Spacer()
                
                Text(UIDevice.current.systemVersion)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }
    
    // MARK: - Actions
    
    private func resetAllSettings() {
        appSettings.colorScheme = nil
        appSettings.preferredCamera = 0
        appSettings.targetFPS = 30
        appSettings.confidenceThreshold = 0.7
        appSettings.enableHapticFeedback = true
        appSettings.showDebugInfo = false
        
        appSettings.updateCameraConfig()
        appSettings.updateModelConfig()
    }
    
    private func exportAllData() {
        // In a real implementation, this would export all training data
        print("Exporting all training data...")
        
        // Would show activity view controller with export options
        let activityVC = UIActivityViewController(
            activityItems: ["Training data would be exported here"],
            applicationActivities: nil
        )
        
        if let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let window = windowScene.windows.first {
            window.rootViewController?.present(activityVC, animated: true)
        }
    }
}

// MARK: - Model Info View

struct ModelInfoView: View {
    @EnvironmentObject var gestureRecognizer: GestureRecognizerWrapper
    @EnvironmentObject var gestureRegistry: GestureRegistry

    var body: some View {
        List {
            Section("Model Details") {
                InfoRow(title: "Name", value: "Mock Gesture Model")
                InfoRow(title: "Version", value: "1.0.0")
                InfoRow(title: "Backend", value: "Mock Backend")
                InfoRow(title: "Size", value: "~1.2 MB")
            }

            Section("Defined Gestures (\(gestureRegistry.gestures.count))") {
                if gestureRegistry.gestures.isEmpty {
                    Text("No gestures defined yet")
                        .foregroundColor(.secondary)
                } else {
                    ForEach(gestureRegistry.gestures) { gesture in
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)

                            VStack(alignment: .leading, spacing: 2) {
                                Text(gesture.name)
                                if !gesture.description.isEmpty {
                                    Text(gesture.description)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                        .lineLimit(1)
                                }
                            }
                        }
                    }
                }
            }

            Section("Performance") {
                InfoRow(title: "Average Latency", value: "~45ms")
                InfoRow(title: "Accuracy", value: "~87%")
                InfoRow(title: "Supported FPS", value: "15-60")
            }
        }
        .navigationTitle("Model Info")
        .navigationBarTitleDisplayMode(.large)
    }
}

// MARK: - About View

struct AboutView: View {
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // App Icon
                    Image(systemName: "hand.raised.app.fill")
                        .font(.system(size: 80))
                        .foregroundColor(.blue)
                    
                    // App Name
                    Text("CameraGestures")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Model Training")
                        .font(.title3)
                        .foregroundColor(.secondary)
                    
                    // Description
                    VStack(alignment: .leading, spacing: 12) {
                        Text("About")
                            .font(.headline)
                        
                        Text("CameraGestures is a modular dynamic gesture recognition system that captures hand movements through a camera and translates them into recognizable gestures for application control.")
                            .font(.body)
                            .multilineTextAlignment(.leading)
                        
                        Text("This ModelTraining app allows you to collect gesture data, train custom models, and test recognition performance.")
                            .font(.body)
                            .multilineTextAlignment(.leading)
                    }
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                    
                    // Features
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Features")
                            .font(.headline)
                        
                        FeatureRow(icon: "camera.fill", text: "Real-time hand tracking")
                        FeatureRow(icon: "brain.head.profile", text: "Machine learning gesture recognition")
                        FeatureRow(icon: "chart.bar.fill", text: "Training data collection")
                        FeatureRow(icon: "gear", text: "Customizable settings")
                    }
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                    
                    // Technical Info
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Technology")
                            .font(.headline)
                        
                        InfoRow(title: "Hand Tracking", value: "MediaPipe Hands")
                        InfoRow(title: "ML Framework", value: "Core ML / TensorFlow")
                        InfoRow(title: "UI Framework", value: "SwiftUI")
                        InfoRow(title: "Minimum iOS", value: "15.0")
                    }
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                    
                    Spacer()
                }
                .padding()
            }
            .navigationTitle("About")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

// MARK: - Helper Views

struct InfoRow: View {
    let title: String
    let value: String
    
    var body: some View {
        HStack {
            Text(title)
                .foregroundColor(.primary)
            
            Spacer()
            
            Text(value)
                .foregroundColor(.secondary)
                .font(.system(.body, design: .monospaced))
        }
    }
}

struct FeatureRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(.blue)
                .frame(width: 20)
            
            Text(text)
                .font(.subheadline)
            
            Spacer()
        }
    }
}

// MARK: - Preview

struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        SettingsView()
            .environmentObject(AppSettings())
            .environmentObject(GestureRecognizerWrapper(recognizer: HandGestureRecognizing()))
            .environmentObject(GestureRegistry())
            .environmentObject(GestureModelAPIClient())
    }
}

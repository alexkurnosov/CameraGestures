import SwiftUI
import Combine
import HandGestureTypes
import GestureModelModule
import HandGestureRecognizingFramework

struct TrainingView: View {
    @EnvironmentObject var gestureRecognizer: GestureRecognizerWrapper
    @EnvironmentObject var trainingDataManager: TrainingDataManager
    @EnvironmentObject var appSettings: AppSettings
    @EnvironmentObject var gestureRegistry: GestureRegistry
    @EnvironmentObject var apiClient: GestureModelAPIClient
    @EnvironmentObject var serverManager: ServerTrainingManager

    // UI-only state
    @State private var showingNewDatasetAlert = false
    @State private var newDatasetName = ""
    @State private var showingAddGestureSheet = false
    @State private var showingMetricsSheet = false
    @State private var completedMetrics: ModelMetrics?
    @State private var showingTrainingError = false
    @State private var showingServerError = false
    @State private var showingWipeModelAlert = false

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    datasetInfoSection
                    gestureSelectionSection

                    if trainingDataManager.isCollecting {
                        collectionProgressSection
                    }

                    collectionControlsSection
                    trainingDataSummarySection
                    trainingControlsSection
                    inViewThresholdSection
                    serverControlsSection
                    dangerZoneSection
                }
                .padding()
            }
            .navigationTitle("Training Data")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button {
                        showingAddGestureSheet = true
                    } label: {
                        Image(systemName: "plus")
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    HStack(spacing: 4) {
                        NavigationLink(destination: HandFilmsView()
                            .environmentObject(trainingDataManager)
                            .environmentObject(gestureRegistry)
                        ) {
                            Image(systemName: "film.stack")
                        }
                        Button("New Dataset") {
                            showingNewDatasetAlert = true
                        }
                    }
                }
            }
        }
        .onAppear {
            if trainingDataManager.selectedGesture == nil {
                trainingDataManager.selectedGesture = gestureRegistry.gestures.first
            }
            serverManager.refreshServerStatus()
        }
        .onChange(of: gestureRegistry.gestures) { gestures in
            if let current = trainingDataManager.selectedGesture, !gestures.contains(current) {
                trainingDataManager.selectedGesture = gestures.first
            } else if trainingDataManager.selectedGesture == nil {
                trainingDataManager.selectedGesture = gestures.first
            }
        }
        .onChange(of: trainingDataManager.trainingState) { newState in
            if case .done(let metrics) = newState {
                completedMetrics = metrics
                showingMetricsSheet = true
            }
        }
        .onChange(of: serverManager.serverActionError) { error in
            if error != nil {
                showingServerError = true
            }
        }
        .sheet(isPresented: $showingAddGestureSheet) {
            AddGestureSheet()
                .environmentObject(gestureRegistry)
        }
        .sheet(isPresented: $showingMetricsSheet) {
            if let metrics = completedMetrics {
                TrainingMetricsSheet(
                    metrics: metrics,
                    gestureIds: gestureRegistry.gestures.map { $0.id }
                )
            }
        }
        .alert("New Dataset", isPresented: $showingNewDatasetAlert) {
            TextField("Dataset Name", text: $newDatasetName)
            Button("Create") { createNewDataset() }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("Enter a name for the new training dataset")
        }
        .alert("Training Failed", isPresented: $showingTrainingError) {
            Button("OK", role: .cancel) { }
        } message: {
            if case .failed(let msg) = trainingDataManager.trainingState {
                Text(msg)
            } else {
                Text("An unknown error occurred.")
            }
        }
        .alert("Server Error", isPresented: $showingServerError) {
            Button("OK", role: .cancel) {
                serverManager.serverActionError = nil
            }
        } message: {
            Text(serverManager.serverActionError ?? "An unknown error occurred.")
        }
        .alert("Wipe Server Model?", isPresented: $showingWipeModelAlert) {
            Button("Wipe Model", role: .destructive) {
                serverManager.wipeServerModel()
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("This will delete the trained model. This action cannot be undone.")
        }
        .onDisappear {
            serverManager.stopPolling()
        }
    }

    // MARK: - UI Sections

    private var datasetInfoSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Current Dataset")
                .font(.headline)

            if let dataset = trainingDataManager.currentDataset {
                HStack {
                    VStack(alignment: .leading) {
                        Text(dataset.name)
                            .font(.title2)
                            .fontWeight(.medium)

                        Text("\(dataset.examples.count) examples")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()

                    Button("Save") {
                        trainingDataManager.saveDataset()
                    }
                    .buttonStyle(.bordered)
                }
                .padding()
                .background(Color.blue.opacity(0.1))
                .cornerRadius(8)
            } else {
                Text("No dataset selected")
                    .foregroundColor(.secondary)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
            }
        }
    }

    private var gestureSelectionSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Select Gesture")
                    .font(.headline)
                Spacer()
                Button {
                    showingAddGestureSheet = true
                } label: {
                    Label("Add", systemImage: "plus.circle")
                        .font(.caption)
                }
            }

            if gestureRegistry.gestures.isEmpty {
                Text("No gestures defined yet. Tap + to add one.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        ForEach(gestureRegistry.gestures) { gesture in
                            GestureSelectionCard(
                                gesture: gesture,
                                isSelected: gesture == trainingDataManager.selectedGesture,
                                sampleCount: getSampleCount(for: gesture)
                            ) {
                                trainingDataManager.selectedGesture = gesture
                            }
                        }
                    }
                    .padding(.horizontal)
                }
            }
        }
    }

    private var collectionProgressSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Collecting: \(trainingDataManager.selectedGesture?.name ?? "")")
                    .font(.headline)

                Spacer()

                Text("\(trainingDataManager.currentSamples)/\(trainingDataManager.targetSamples)")
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(.blue)
            }

            ProgressView(value: trainingDataManager.collectionProgress, total: 1.0)
                .progressViewStyle(LinearProgressViewStyle(tint: .green))
                .scaleEffect(y: 2.0)

            Text("Perform the gesture in front of the camera")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color.green.opacity(0.1))
        .cornerRadius(8)
    }

    private var collectionControlsSection: some View {
        VStack(spacing: 12) {
            if !trainingDataManager.isCollecting {
                Button(action: {
                    guard let gesture = trainingDataManager.selectedGesture else { return }
                    trainingDataManager.startDataCollection(for: gesture)
                }) {
                    HStack {
                        Image(systemName: "record.circle")
                        Text("Start Collecting")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.red)
                    .cornerRadius(8)
                }
                .disabled(trainingDataManager.currentDataset == nil || trainingDataManager.selectedGesture == nil)

                HStack {
                    Text("Target Samples:")

                    Spacer()

                    Stepper(value: $trainingDataManager.targetSamples, in: 5...50, step: 5) {
                        Text("\(trainingDataManager.targetSamples)")
                            .font(.title3)
                            .fontWeight(.medium)
                    }
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(8)

            } else {
                Button(action: { trainingDataManager.stopDataCollection() }) {
                    HStack {
                        Image(systemName: "stop.circle")
                        Text("Stop Collecting")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.orange)
                    .cornerRadius(8)
                }
            }
        }
    }

    private var trainingDataSummarySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Training Data Summary")
                .font(.headline)

            if let dataset = trainingDataManager.currentDataset, !dataset.examples.isEmpty {
                LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 8) {
                    ForEach(gestureRegistry.gestures) { gesture in
                        let count = getSampleCount(for: gesture)

                        HStack {
                            Text(gesture.name)
                                .font(.caption)

                            Spacer()

                            Text("\(count)")
                                .font(.caption)
                                .fontWeight(.medium)
                                .foregroundColor(count >= 10 ? .green : .orange)
                        }
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(4)
                    }
                }
            } else {
                Text("No training data collected yet")
                    .foregroundColor(.secondary)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
            }
        }
    }

    private var trainingControlsSection: some View {
        VStack(spacing: 12) {
            switch trainingDataManager.trainingState {
            case .idle:
                Button(action: { trainingDataManager.startLocalTraining() }) {
                    HStack {
                        Image(systemName: "brain")
                        Text("Train Model")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(canStartTraining ? Color.blue : Color.gray)
                    .cornerRadius(8)
                }
                .disabled(!canStartTraining)

            case .training:
                VStack(spacing: 8) {
                    ProgressView()
                        .progressViewStyle(.circular)
                        .scaleEffect(1.2)

                    Text("Training model…")
                        .font(.subheadline)
                        .foregroundColor(.secondary)

                    Text("This may take a few minutes.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color.blue.opacity(0.08))
                .cornerRadius(8)

            case .done(let metrics):
                VStack(spacing: 8) {
                    HStack {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text("Model trained")
                            .font(.headline)
                            .foregroundColor(.green)
                        Spacer()
                        Text(String(format: "%.1f%%", metrics.accuracy * 100))
                            .font(.title3)
                            .fontWeight(.bold)
                            .foregroundColor(.green)
                    }

                    HStack(spacing: 12) {
                        Button("View Metrics") {
                            completedMetrics = metrics
                            showingMetricsSheet = true
                        }
                        .buttonStyle(.bordered)

                        Button(action: { trainingDataManager.startLocalTraining() }) {
                            Label("Retrain", systemImage: "arrow.clockwise")
                        }
                        .buttonStyle(.bordered)
                        .disabled(!canStartTraining)
                    }
                }
                .padding()
                .background(Color.green.opacity(0.08))
                .cornerRadius(8)

            case .failed(let message):
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.red)
                        Text("Training failed")
                            .font(.headline)
                            .foregroundColor(.red)
                    }

                    Text(message)
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Button(action: { trainingDataManager.startLocalTraining() }) {
                        Label("Retry", systemImage: "arrow.clockwise")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                    .disabled(!canStartTraining)
                }
                .padding()
                .background(Color.red.opacity(0.08))
                .cornerRadius(8)
            }

            if let dataset = trainingDataManager.currentDataset {
                Text("Total examples: \(dataset.examples.count)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }

    // MARK: - In-View Threshold Section

    private var inViewThresholdSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Recording Quality")
                    .font(.headline)
                Spacer()
                if appSettings.isThresholdLocked {
                    Label("Locked", systemImage: "lock.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.orange)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color.orange.opacity(0.12))
                        .cornerRadius(6)
                }
            }

            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Min. in-view duration")
                        .font(.subheadline)
                    Text("Minimum seconds the hand must be visible in a capture window")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
                HStack(spacing: 6) {
                    Text(String(format: "%.1fs", appSettings.minInViewDuration))
                        .font(.subheadline.weight(.medium))
                        .foregroundColor(appSettings.isThresholdLocked ? .secondary : .primary)
                        .frame(width: 38, alignment: .trailing)
                    Stepper(
                        "",
                        value: $appSettings.minInViewDuration,
                        in: 0.2...10.0,
                        step: 0.1
                    )
                    .labelsHidden()
                    .disabled(appSettings.isThresholdLocked)
                }
            }

            if appSettings.isThresholdLocked {
                Text("Locked after first training job. Wipe the server model to unlock.")
                    .font(.caption)
                    .foregroundColor(.orange)
            } else {
                Text("This value will be locked when you start server training.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color.yellow.opacity(0.06))
        .cornerRadius(8)
    }

    // MARK: - Server Controls Section

    private var serverControlsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Server Training")
                    .font(.headline)
                Spacer()
                if let status = serverManager.serverStatus {
                    ServerStatusBadge(status: status.status)
                }
            }

            // Server URL
            HStack(spacing: 8) {
                Image(systemName: "network")
                    .foregroundColor(.secondary)
                    .frame(width: 20)
                Text(apiClient.baseURL.absoluteString)
                    .font(.caption.monospaced())
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            // Upload state indicator
            uploadStatusRow

            // Action buttons
            HStack(spacing: 12) {
                Button(action: { serverManager.triggerServerTraining() }) {
                    Label("Train on Server", systemImage: "brain.head.profile")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.blue)
                .disabled(serverManager.isPollingStatus)

                Button(action: { serverManager.downloadModelFromServer() }) {
                    if serverManager.isDownloadingModel {
                        ProgressView()
                            .frame(maxWidth: .infinity)
                    } else {
                        Label("Update Model", systemImage: "arrow.down.circle")
                            .frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
                .disabled(serverManager.isDownloadingModel)
            }

            // Training progress / status details
            if let status = serverManager.serverStatus {
                serverStatusDetailView(status: status)
            }
        }
        .padding()
        .background(Color.purple.opacity(0.06))
        .cornerRadius(8)
    }

    private var dangerZoneSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Danger Zone")
                .font(.headline)
                .foregroundColor(.red)

            Button(action: { showingWipeModelAlert = true }) {
                HStack {
                    if serverManager.isWipingModel {
                        ProgressView()
                            .scaleEffect(0.8)
                            .frame(width: 20, height: 20)
                    } else {
                        Image(systemName: "trash.fill")
                            .frame(width: 20, height: 20)
                    }
                    Text(serverManager.isWipingModel ? "Wiping…" : "Wipe Server Model")
                }
                .font(.subheadline)
                .foregroundColor(.red)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 10)
                .background(Color.red.opacity(0.08))
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.red.opacity(0.3), lineWidth: 1)
                )
            }
            .disabled(serverManager.isWipingModel)
        }
        .padding()
        .background(Color.red.opacity(0.04))
        .cornerRadius(8)
    }

    @ViewBuilder
    private var uploadStatusRow: some View {
        switch trainingDataManager.uploadState {
        case .idle:
            EmptyView()
        case .uploading:
            HStack(spacing: 6) {
                ProgressView()
                    .scaleEffect(0.75)
                Text("Uploading example…")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        case .uploaded(let total):
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundColor(.green)
                    .font(.caption)
                Text("Uploaded — server has \(total) example(s) for this gesture")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        case .failed(let msg):
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.circle.fill")
                    .foregroundColor(.orange)
                    .font(.caption)
                Text("Upload failed: \(msg)")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }
        }
    }

    @ViewBuilder
    private func serverStatusDetailView(status: ModelStatusResponse) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if status.status == "training" {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.85)
                    Text("Training in progress…")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            if let accuracy = status.accuracy {
                HStack {
                    Text("Server accuracy:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.1f%%", accuracy * 100))
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(.green)
                }
            }

            if status.status == "ready", !status.gestureIds.isEmpty {
                Text("Gestures: \(status.gestureIds.joined(separator: ", "))")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }

            if let err = status.error {
                Text("Error: \(err)")
                    .font(.caption2)
                    .foregroundColor(.red)
                    .lineLimit(3)
            }
        }
    }

    // MARK: - Computed Properties

    private var canStartTraining: Bool {
        guard let dataset = trainingDataManager.currentDataset,
              !gestureRegistry.gestures.isEmpty else { return false }
        let gestureCount = dataset.gestureCount
        return gestureRegistry.gestures.allSatisfy { gesture in
            (gestureCount[gesture.id] ?? 0) >= 5
        }
    }

    // MARK: - Helper Methods

    private func getSampleCount(for gesture: GestureDefinition) -> Int {
        return trainingDataManager.currentDataset?.gestureCount[gesture.id] ?? 0
    }

    private func createNewDataset() {
        guard !newDatasetName.isEmpty else { return }
        trainingDataManager.createNewDataset(name: newDatasetName)
        newDatasetName = ""
    }
}

// MARK: - Server Status Badge

struct ServerStatusBadge: View {
    let status: String

    private var color: Color {
        switch status {
        case "ready": return .green
        case "training": return .orange
        case "failed": return .red
        default: return .secondary
        }
    }

    private var label: String {
        switch status {
        case "ready": return "Ready"
        case "training": return "Training…"
        case "failed": return "Failed"
        default: return "Idle"
        }
    }

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 7, height: 7)
            Text(label)
                .font(.caption)
                .foregroundColor(color)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(color.opacity(0.12))
        .clipShape(Capsule())
    }
}

// MARK: - Gesture Selection Card

struct GestureSelectionCard: View {
    let gesture: GestureDefinition
    let isSelected: Bool
    let sampleCount: Int
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Image(systemName: "hand.raised")
                    .font(.title2)
                    .foregroundColor(isSelected ? .white : .primary)

                Text(gesture.name)
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .foregroundColor(isSelected ? .white : .primary)

                if sampleCount > 0 {
                    Text("\(sampleCount)")
                        .font(.caption2)
                        .fontWeight(.bold)
                        .foregroundColor(isSelected ? .blue : .white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(isSelected ? .white : Color.blue)
                        .clipShape(Capsule())
                }
            }
            .padding()
            .frame(width: 100, height: 80)
            .background(isSelected ? Color.blue : Color.gray.opacity(0.1))
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(isSelected ? Color.blue : Color.clear, lineWidth: 2)
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Add Gesture Sheet

struct AddGestureSheet: View {
    @EnvironmentObject var gestureRegistry: GestureRegistry
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var description = ""
    @State private var showDuplicateError = false

    private var slug: String {
        GestureRegistry.slug(from: name)
    }

    private var isNameValid: Bool {
        !slug.isEmpty && !gestureRegistry.gestures.contains(where: { $0.id == slug })
    }

    var body: some View {
        NavigationView {
            Form {
                Section("Gesture Name") {
                    TextField("e.g. Thumbs Up", text: $name)
                        .autocorrectionDisabled()

                    if !name.isEmpty {
                        HStack {
                            Text("ID:")
                                .foregroundColor(.secondary)
                                .font(.caption)
                            Text(slug.isEmpty ? "—" : slug)
                                .font(.caption.monospaced())
                                .foregroundColor(isNameValid ? .secondary : .red)
                        }

                        if !isNameValid && !slug.isEmpty {
                            Text("A gesture with this ID already exists.")
                                .font(.caption)
                                .foregroundColor(.red)
                        }
                    }
                }

                Section("Description") {
                    TextField("Describe how to perform this gesture", text: $description, axis: .vertical)
                        .lineLimit(3...6)
                }
            }
            .navigationTitle("Add Gesture")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Save") {
                        save()
                    }
                    .disabled(!isNameValid)
                }
            }
        }
    }

    private func save() {
        gestureRegistry.addGesture(name: name.trimmingCharacters(in: .whitespaces),
                                   description: description.trimmingCharacters(in: .whitespaces))
        dismiss()
    }
}

// MARK: - Training Metrics Sheet

struct TrainingMetricsSheet: View {
    let metrics: ModelMetrics
    let gestureIds: [String]
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            List {
                Section("Summary") {
                    MetricRow(label: "Accuracy", value: String(format: "%.1f%%", metrics.accuracy * 100))
                    MetricRow(label: "Precision", value: String(format: "%.1f%%", metrics.precision * 100))
                    MetricRow(label: "Recall", value: String(format: "%.1f%%", metrics.recall * 100))
                    MetricRow(label: "F1 Score", value: String(format: "%.3f", metrics.f1Score))
                }

                Section("Timing") {
                    MetricRow(
                        label: "Training Time",
                        value: formatDuration(metrics.trainingTime)
                    )
                    MetricRow(
                        label: "Validation Time",
                        value: formatDuration(metrics.validationTime)
                    )
                }

                if !metrics.confusionMatrix.isEmpty && !gestureIds.isEmpty {
                    Section("Per-Class Accuracy") {
                        ForEach(gestureIds.indices, id: \.self) { i in
                            if i < metrics.confusionMatrix.count {
                                let row = metrics.confusionMatrix[i]
                                let total = row.reduce(0, +)
                                let correct = i < row.count ? row[i] : 0
                                let pct = total > 0 ? Float(correct) / Float(total) : 0
                                HStack {
                                    Text(gestureIds[i])
                                        .font(.subheadline)
                                    Spacer()
                                    Text("\(correct)/\(total)")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    Text(String(format: "%.0f%%", pct * 100))
                                        .font(.subheadline)
                                        .fontWeight(.medium)
                                        .foregroundColor(pct >= 0.8 ? .green : pct >= 0.6 ? .orange : .red)
                                        .frame(width: 44, alignment: .trailing)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Training Results")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        if seconds < 60 {
            return String(format: "%.1fs", seconds)
        } else {
            let mins = Int(seconds) / 60
            let secs = Int(seconds) % 60
            return "\(mins)m \(secs)s"
        }
    }
}

struct MetricRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
        }
    }
}

// MARK: - Preview

struct TrainingView_Previews: PreviewProvider {
    static var previews: some View {
        TrainingView()
            .environmentObject(GestureRecognizerWrapper(recognizer: HandGestureRecognizing()))
            .environmentObject(TrainingDataManager())
            .environmentObject(AppSettings())
            .environmentObject(GestureRegistry())
            .environmentObject(GestureModelAPIClient())
            .environmentObject(ServerTrainingManager())
    }
}

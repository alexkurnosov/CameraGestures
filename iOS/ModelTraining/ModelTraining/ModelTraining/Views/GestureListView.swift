import SwiftUI
import HandGestureTypes
import GestureModelModule

struct GestureListView: View {
    @EnvironmentObject var trainingDataManager: TrainingDataManager
    @EnvironmentObject var gestureRegistry: GestureRegistry

    @State private var selectedGesture: GestureDefinition?
    @State private var showingGestureDetail = false
    @State private var showingAddGestureSheet = false
    @State private var searchText = ""

    var body: some View {
        NavigationView {
            VStack {
                if trainingDataManager.trainingExamples.isEmpty && gestureRegistry.gestures.isEmpty {
                    emptyStateView
                } else {
                    gestureListContent
                }
            }
            .navigationTitle("Gestures")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button {
                        showingAddGestureSheet = true
                    } label: {
                        Image(systemName: "plus")
                    }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Export") {
                        exportTrainingData()
                    }
                    .disabled(trainingDataManager.trainingExamples.isEmpty)
                }
            }
        }
        .sheet(isPresented: $showingGestureDetail) {
            if let gesture = selectedGesture {
                GestureDetailView(gesture: gesture)
                    .environmentObject(trainingDataManager)
            }
        }
        .sheet(isPresented: $showingAddGestureSheet) {
            AddGestureSheet()
                .environmentObject(gestureRegistry)
        }
        .overlay(alignment: .bottom) {
            if let error = trainingDataManager.serverSyncError {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    Text("Server sync failed: \(error)")
                        .font(.caption)
                        .foregroundColor(.primary)
                        .lineLimit(2)
                }
                .padding(10)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
                .padding()
            }
        }
    }

    // MARK: - UI Components

    private var emptyStateView: some View {
        VStack(spacing: 20) {
            Image(systemName: "hand.raised.slash")
                .font(.system(size: 60))
                .foregroundColor(.gray)

            Text("No Gestures Yet")
                .font(.title2)
                .fontWeight(.medium)

            Text("Add a gesture definition to get started")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            Button {
                showingAddGestureSheet = true
            } label: {
                Label("Add Gesture", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var gestureListContent: some View {
        List {
            Section("Overview") {
                StatisticRow(
                    title: "Total Examples",
                    value: "\(trainingDataManager.trainingExamples.count + trainingDataManager.serverExampleCounts.values.reduce(0, +))",
                    icon: "chart.bar.fill",
                    color: .blue
                )

                StatisticRow(
                    title: "Defined Gestures",
                    value: "\(gestureRegistry.gestures.count)",
                    icon: "hand.raised.fill",
                    color: .green
                )

                StatisticRow(
                    title: "Training Sessions",
                    value: "\(uniqueSessions.count)",
                    icon: "clock.fill",
                    color: .orange
                )
            }

            Section("Gesture Definitions") {
                ForEach(filteredGestures) { gesture in
                    GestureTypeRow(
                        gesture: gesture,
                        exampleCount: getExampleCount(for: gesture),
                        lastRecorded: getLastRecorded(for: gesture)
                    ) {
                        selectedGesture = gesture
                        showingGestureDetail = true
                    }
                }
                .onDelete { offsets in
                    deleteGestures(at: offsets)
                }

                Button {
                    showingAddGestureSheet = true
                } label: {
                    Label("Add Gesture", systemImage: "plus.circle")
                }
            }

            if !recentExamples.isEmpty {
                Section("Recent Examples") {
                    ForEach(recentExamples.indices, id: \.self) { index in
                        ExampleRow(example: recentExamples[index], gestureRegistry: gestureRegistry)
                    }
                }
            }
        }
        .searchable(text: $searchText, prompt: "Search gestures...")
    }

    // MARK: - Computed Properties

    private var filteredGestures: [GestureDefinition] {
        if searchText.isEmpty {
            return gestureRegistry.gestures
        } else {
            return gestureRegistry.gestures.filter {
                $0.name.localizedCaseInsensitiveContains(searchText) ||
                $0.description.localizedCaseInsensitiveContains(searchText)
            }
        }
    }

    private var uniqueSessions: [String] {
        Array(Set(trainingDataManager.trainingExamples.map { $0.sessionId }))
    }

    private var recentExamples: [TrainingExample] {
        Array(trainingDataManager.trainingExamples
            .sorted { $0.timestamp > $1.timestamp }
            .prefix(5))
    }

    // MARK: - Helper Methods

    private func getExampleCount(for gesture: GestureDefinition) -> Int {
        let pending = trainingDataManager.trainingExamples.filter { $0.gestureId == gesture.id }.count
        let server = trainingDataManager.serverExampleCounts[gesture.id] ?? 0
        return pending + server
    }

    private func getLastRecorded(for gesture: GestureDefinition) -> Date? {
        let timestamps = trainingDataManager.trainingExamples
            .filter { $0.gestureId == gesture.id }
            .map { $0.timestamp }
        guard let max = timestamps.max() else { return nil }
        return Date(timeIntervalSince1970: max)
    }

    private func deleteGestures(at offsets: IndexSet) {
        for index in offsets {
            let gesture = filteredGestures[index]
            gestureRegistry.removeGesture(id: gesture.id)
        }
    }

    private func exportTrainingData() {
        let exportData = trainingDataManager.trainingExamples.map { example in
            [
                "gestureId": example.gestureId,
                "timestamp": example.timestamp,
                "sessionId": example.sessionId,
                "frameCount": example.handfilm.frames.count,
                "duration": example.handfilm.duration
            ] as [String: Any]
        }
        print("Export data prepared: \(exportData.count) examples")
    }
}

// MARK: - Statistic Row

struct StatisticRow: View {
    let title: String
    let value: String
    let icon: String
    let color: Color

    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(color)
                .frame(width: 24, height: 24)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                Text(value)
                    .font(.title3)
                    .fontWeight(.medium)
            }

            Spacer()
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Gesture Type Row

struct GestureTypeRow: View {
    let gesture: GestureDefinition
    let exampleCount: Int
    let lastRecorded: Date?
    let action: () -> Void

    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter
    }()

    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: "hand.raised")
                    .foregroundColor(.blue)
                    .frame(width: 30, height: 30)
                    .background(Color.blue.opacity(0.1))
                    .clipShape(Circle())

                VStack(alignment: .leading, spacing: 4) {
                    Text(gesture.name)
                        .font(.headline)
                        .foregroundColor(.primary)

                    if !gesture.description.isEmpty {
                        Text(gesture.description)
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }

                    HStack {
                        Text("\(exampleCount) examples")
                            .font(.caption)
                            .foregroundColor(.secondary)

                        if let date = lastRecorded {
                            Text("•")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Text("Last: \(dateFormatter.string(from: date))")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }

                Spacer()

                QualityIndicator(count: exampleCount)

                Image(systemName: "chevron.right")
                    .foregroundColor(.secondary)
                    .font(.caption)
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Quality Indicator

struct QualityIndicator: View {
    let count: Int

    var body: some View {
        Circle()
            .fill(qualityColor)
            .frame(width: 12, height: 12)
            .overlay(
                Text(qualityLevel)
                    .font(.system(size: 8, weight: .bold))
                    .foregroundColor(.white)
            )
    }

    private var qualityColor: Color {
        if count >= 20 { return .green }
        else if count >= 10 { return .orange }
        else { return .red }
    }

    private var qualityLevel: String {
        if count >= 20 { return "✓" }
        else if count >= 10 { return "~" }
        else { return "!" }
    }
}

// MARK: - Example Row

struct ExampleRow: View {
    let example: TrainingExample
    let gestureRegistry: GestureRegistry

    private let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter
    }()

    private var gestureName: String {
        gestureRegistry.gestures.first(where: { $0.id == example.gestureId })?.name ?? example.gestureId
    }

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(gestureName)
                    .font(.subheadline)
                    .fontWeight(.medium)

                HStack {
                    Text(timeFormatter.string(from: Date(timeIntervalSince1970: example.timestamp)))
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Text("•")
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Text("\(example.handfilm.frames.count) frames")
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Text("•")
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Text(String(format: "%.1fs", example.handfilm.duration))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()

            Text(example.sessionId.prefix(8))
                .font(.system(.caption2, design: .monospaced))
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Gesture Detail View

struct GestureDetailView: View {
    let gesture: GestureDefinition
    @EnvironmentObject var trainingDataManager: TrainingDataManager
    @Environment(\.dismiss) private var dismiss

    private var examples: [TrainingExample] {
        trainingDataManager.trainingExamples
            .filter { $0.gestureId == gesture.id }
            .sorted { $0.timestamp > $1.timestamp }
    }

    var body: some View {
        NavigationView {
            List {
                Section("Gesture Info") {
                    LabeledContent("Name", value: gesture.name)
                    LabeledContent("ID", value: gesture.id)
                    if !gesture.description.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Description")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(gesture.description)
                                .font(.subheadline)
                        }
                        .padding(.vertical, 2)
                    }
                }

                Section("Summary") {
                    StatisticRow(
                        title: "Total Examples",
                        value: "\(examples.count)",
                        icon: "chart.bar.fill",
                        color: .blue
                    )

                    if !examples.isEmpty {
                        StatisticRow(
                            title: "Average Duration",
                            value: String(format: "%.1fs", averageDuration),
                            icon: "clock.fill",
                            color: .orange
                        )

                        StatisticRow(
                            title: "Average Frame Count",
                            value: "\(Int(averageFrameCount))",
                            icon: "film.fill",
                            color: .green
                        )
                    }
                }

                Section("Examples") {
                    ForEach(examples.indices, id: \.self) { index in
                        ExampleDetailRow(example: examples[index], index: index + 1)
                    }
                }
            }
            .navigationTitle(gesture.name)
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private var averageDuration: Double {
        guard !examples.isEmpty else { return 0 }
        return examples.map { $0.handfilm.duration }.reduce(0, +) / Double(examples.count)
    }

    private var averageFrameCount: Double {
        guard !examples.isEmpty else { return 0 }
        return Double(examples.map { $0.handfilm.frames.count }.reduce(0, +)) / Double(examples.count)
    }
}

// MARK: - Example Detail Row

struct ExampleDetailRow: View {
    let example: TrainingExample
    let index: Int

    private let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .medium
        return formatter
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Example #\(index)")
                    .font(.subheadline)
                    .fontWeight(.medium)

                Spacer()

                Text(dateFormatter.string(from: Date(timeIntervalSince1970: example.timestamp)))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            HStack {
                Label("\(example.handfilm.frames.count) frames", systemImage: "film")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Label(String(format: "%.2fs", example.handfilm.duration), systemImage: "clock")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                Spacer()
            }
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Search Bar (kept for backward compat if referenced elsewhere)

struct SearchBar: View {
    @Binding var text: String

    var body: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundColor(.secondary)

            TextField("Search gestures...", text: $text)
                .textFieldStyle(RoundedBorderTextFieldStyle())
        }
    }
}

// MARK: - Preview

struct GestureListView_Previews: PreviewProvider {
    static var previews: some View {
        GestureListView()
            .environmentObject(TrainingDataManager())
            .environmentObject(GestureRegistry())
    }
}

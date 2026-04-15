import SwiftUI
import HandGestureTypes
import GestureModelModule

struct GestureListView: View {
    @EnvironmentObject var trainingDataManager: TrainingDataManager
    @EnvironmentObject var gestureRegistry: GestureRegistry
    @EnvironmentObject var apiClient: GestureModelAPIClient

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
                    .environmentObject(gestureRegistry)
                    .environmentObject(apiClient)
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

// MARK: - Preview

struct GestureListView_Previews: PreviewProvider {
    static var previews: some View {
        GestureListView()
            .environmentObject(TrainingDataManager())
            .environmentObject(GestureRegistry())
            .environmentObject(GestureModelAPIClient())
    }
}

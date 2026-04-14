import SwiftUI
import HandGestureTypes
import GestureModelModule

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

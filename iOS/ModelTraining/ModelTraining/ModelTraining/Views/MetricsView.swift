import SwiftUI

/// Model interpretation page. Fetches /model/metrics from the server (public,
/// no auth) and presents each group in plain language so non-technical users
/// can read it.
struct MetricsView: View {
    @EnvironmentObject var apiClient: GestureModelAPIClient

    @State private var metrics: ModelMetricsResponse?
    @State private var history: [ModelMetricsSummary] = []
    @State private var selectedModelId: String?
    @State private var isLoading = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationView {
            Group {
                if isLoading && metrics == nil {
                    ProgressView("Loading metrics…")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let message = errorMessage, metrics == nil {
                    errorState(message)
                } else if let metrics = metrics {
                    metricsList(metrics)
                } else {
                    Text("No trained model yet.")
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .navigationTitle("Model Metrics")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button {
                        Task { await refresh() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(isLoading)
                }
            }
            .task {
                await refresh()
            }
        }
    }

    // MARK: - States

    private func errorState(_ message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .foregroundColor(.orange)
            Text("Failed to load metrics")
                .font(.headline)
            Text(message)
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            Button("Retry") {
                Task { await refresh() }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    @ViewBuilder
    private func metricsList(_ m: ModelMetricsResponse) -> some View {
        List {
            modelPickerSection
            overviewSection(m)
            perClassSection(m)
            falseAlarmSection(m)
            confidenceSection(m)
            thresholdSection(m)
            aucSection(m)
            confusionMatrixSection(m)
        }
        .listStyle(.insetGrouped)
        .refreshable { await refresh() }
    }

    // MARK: - Sections

    @ViewBuilder
    private var modelPickerSection: some View {
        if history.count > 1 {
            Section {
                Picker("Model", selection: Binding(
                    get: { selectedModelId ?? history.first?.modelId ?? "" },
                    set: { newValue in
                        selectedModelId = newValue
                        Task { await loadMetrics(for: newValue) }
                    }
                )) {
                    ForEach(history) { summary in
                        Text(pickerLabel(for: summary))
                            .tag(summary.modelId)
                    }
                }
            } header: {
                Text("Version")
            } footer: {
                Text("Switch to a previous training run to compare metrics.")
            }
        }
    }

    private func overviewSection(_ m: ModelMetricsResponse) -> some View {
        Section {
            metricRow("Overall accuracy", value: percent(m.accuracy))
            metricRow("Weighted F1", value: format(m.f1Weighted))
            metricRow("Trained on", value: "\(m.trainedOn) examples")
            if let val = m.valSize, let train = m.trainSize {
                metricRow("Train / validation split", value: "\(train) / \(val)")
            }
            metricRow("Trained at", value: formatDate(m.trainedAt))
            if let strategy = m.balanceStrategy {
                metricRow("Balance strategy", value: strategy)
            }
        } header: {
            Text("Overview")
        } footer: {
            Text("Overall accuracy is the single aggregate score. It includes synthetic 'nothing' examples that are easy to reject, so it may look optimistic — see the next sections for a more honest breakdown.")
        }
    }

    @ViewBuilder
    private func perClassSection(_ m: ModelMetricsResponse) -> some View {
        if !m.perClass.isEmpty {
            Section {
                ForEach(m.perClass) { row in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(row.gestureId)
                            .font(.headline)
                        HStack {
                            labelValue("Precision", percent(row.precision))
                            Spacer()
                            labelValue("Recall", percent(row.recall))
                            Spacer()
                            labelValue("F1", format(row.f1))
                        }
                        .font(.caption)
                        Text("train: \(row.supportTrain)   •   val: \(row.supportVal)")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 2)
                }
            } header: {
                Text("Per gesture")
            } footer: {
                Text("Precision = 'when the model says X, how often is it actually X?'. Recall = 'out of real X gestures, how many did we catch?'. Low precision means false alarms; low recall means missed gestures. 'support' is how many examples of this class were used.")
            }
        }
    }

    @ViewBuilder
    private func falseAlarmSection(_ m: ModelMetricsResponse) -> some View {
        let n = m.noneAware
        if n.noneFalsePositiveRate != nil || n.realAccuracy != nil {
            Section {
                if let fpr = n.noneFalsePositiveRate {
                    metricRow("False-alarm rate", value: percent(fpr))
                }
                if let support = n.noneSupportVal {
                    metricRow("Synthetic negatives evaluated", value: "\(support)")
                }
                if let acc = n.realAccuracy {
                    metricRow("Real-gesture accuracy", value: percent(acc))
                }
                if let support = n.realSupportVal {
                    metricRow("Real examples evaluated", value: "\(support)")
                }
            } header: {
                Text("False alarms")
            } footer: {
                Text("The model is trained to reject 'nothing happening' by adding synthetic negatives. The false-alarm rate is the fraction of those negatives that got mistaken for a real gesture — this is closer to what the user feels in daily use. Real-gesture accuracy excludes the easy synthetic rejections.")
            }
        }
    }

    @ViewBuilder
    private func confidenceSection(_ m: ModelMetricsResponse) -> some View {
        if !m.confidenceByClass.isEmpty {
            Section {
                ForEach(m.confidenceByClass) { c in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(c.gestureId)
                            .font(.headline)
                        Text("predicted \(c.count) times")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                        HStack {
                            labelValue("p10", percent(c.p10))
                            Spacer()
                            labelValue("median", percent(c.p50))
                            Spacer()
                            labelValue("p90", percent(c.p90))
                            Spacer()
                            labelValue("mean", percent(c.mean))
                        }
                        .font(.caption)
                    }
                    .padding(.vertical, 2)
                }
            } header: {
                Text("Confidence when predicted")
            } footer: {
                Text("For each class, the confidence distribution when the model chose it. p10 means '10% of predictions were below this value'. If the median is low, the model is guessing; if p90 is close to 1.0, the model is usually very sure.")
            }
        }
    }

    @ViewBuilder
    private func thresholdSection(_ m: ModelMetricsResponse) -> some View {
        if !m.thresholdCurves.isEmpty {
            Section {
                ForEach(m.thresholdCurves) { t in
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Confidence ≥ \(format(t.threshold))")
                            .font(.headline)
                        HStack {
                            labelValue("Coverage", percent(t.coverage))
                            Spacer()
                            labelValue("Precision", percent(t.precision))
                            Spacer()
                            labelValue("Fires", "\(t.fires)")
                        }
                        .font(.caption)
                    }
                    .padding(.vertical, 2)
                }
            } header: {
                Text("Threshold trade-off")
            } footer: {
                Text("If you only fire the recogniser when softmax confidence exceeds the threshold: 'Precision' is how often those fires were correct, 'Coverage' is what fraction of examples passed the threshold. Higher threshold → fewer firings but more reliable.")
            }
        }
    }

    @ViewBuilder
    private func aucSection(_ m: ModelMetricsResponse) -> some View {
        if m.auc.rocAucMacro != nil || m.auc.prAucMacro != nil {
            Section {
                if let roc = m.auc.rocAucMacro {
                    metricRow("ROC-AUC (macro)", value: format(roc))
                }
                if let pr = m.auc.prAucMacro {
                    metricRow("PR-AUC (macro)", value: format(pr))
                }
            } header: {
                Text("Threshold-independent AUC")
            } footer: {
                Text("Both scores go from 0 to 1, higher is better. They summarise how well the model ranks correct answers above wrong ones, without picking a specific threshold. Above ~0.9 is good, below ~0.7 is poor.")
            }
        }
    }

    @ViewBuilder
    private func confusionMatrixSection(_ m: ModelMetricsResponse) -> some View {
        if let matrix = m.confusionMatrix, !matrix.isEmpty {
            Section {
                ScrollView(.horizontal, showsIndicators: true) {
                    ConfusionMatrixGrid(labels: m.gestureIds, matrix: matrix)
                        .padding(.vertical, 4)
                }
            } header: {
                Text("Confusion matrix")
            } footer: {
                Text("Rows are the true gesture, columns are the model's prediction. The diagonal is correct; off-diagonal cells show which gestures get mixed up.")
            }
        }
    }

    // MARK: - Helpers

    private func metricRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .foregroundColor(.secondary)
                .font(.system(.body, design: .monospaced))
        }
    }

    private func labelValue(_ label: String, _ value: String) -> some View {
        VStack(spacing: 2) {
            Text(label)
                .foregroundColor(.secondary)
            Text(value)
                .font(.system(.caption, design: .monospaced))
        }
    }

    private func pickerLabel(for summary: ModelMetricsSummary) -> String {
        let date = formatDate(summary.trainedAt)
        if let acc = summary.accuracy {
            return "\(date) — \(percent(acc))"
        }
        return date
    }

    private func format(_ value: Double?) -> String {
        guard let v = value else { return "—" }
        return String(format: "%.3f", v)
    }

    private func percent(_ value: Double?) -> String {
        guard let v = value else { return "—" }
        return String(format: "%.1f%%", v * 100)
    }

    private func formatDate(_ ts: TimeInterval) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter.string(from: Date(timeIntervalSince1970: ts))
    }

    // MARK: - Loading

    private func refresh() async {
        isLoading = true
        errorMessage = nil
        do {
            let list = try await apiClient.fetchMetricsList()
            history = list.models

            let id = selectedModelId ?? list.models.first?.modelId
            if let id = id {
                let detail = try await apiClient.fetchMetrics(modelId: id)
                metrics = detail
                selectedModelId = detail.modelId
            } else {
                metrics = nil
            }
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }

    private func loadMetrics(for modelId: String) async {
        isLoading = true
        errorMessage = nil
        do {
            metrics = try await apiClient.fetchMetrics(modelId: modelId)
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}

// MARK: - Confusion Matrix Grid

private struct ConfusionMatrixGrid: View {
    let labels: [String]
    let matrix: [[Int]]

    private var rowMax: [Int] {
        matrix.map { $0.max() ?? 0 }
    }

    var body: some View {
        let cell: CGFloat = 44
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 0) {
                Text("")
                    .frame(width: cell, height: cell)
                ForEach(labels.indices, id: \.self) { i in
                    Text(short(labels[i]))
                        .font(.caption2)
                        .rotationEffect(.degrees(-45))
                        .frame(width: cell, height: cell)
                }
            }
            ForEach(matrix.indices, id: \.self) { r in
                HStack(spacing: 0) {
                    Text(short(labels[safe: r] ?? ""))
                        .font(.caption2)
                        .frame(width: cell, height: cell, alignment: .trailing)
                    ForEach(matrix[r].indices, id: \.self) { c in
                        let value = matrix[r][c]
                        let intensity = rowMax[r] > 0 ? Double(value) / Double(rowMax[r]) : 0
                        Text("\(value)")
                            .font(.caption2)
                            .frame(width: cell, height: cell)
                            .background(cellColor(intensity: intensity, isDiagonal: r == c))
                            .border(Color.gray.opacity(0.2), width: 0.5)
                    }
                }
            }
        }
    }

    private func cellColor(intensity: Double, isDiagonal: Bool) -> Color {
        let base: Color = isDiagonal ? .green : .red
        return base.opacity(max(0.08, min(intensity, 1.0) * 0.6))
    }

    private func short(_ s: String) -> String {
        s.count <= 6 ? s : String(s.prefix(6))
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

// MARK: - Preview

struct MetricsView_Previews: PreviewProvider {
    static var previews: some View {
        MetricsView()
            .environmentObject(GestureModelAPIClient())
    }
}

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
    @State private var reclusterSignals: PoseReclusterSignals? = nil
    @State private var poseMetrics: PoseMetricsResponse? = nil
    @State private var confidenceCurves: ConfidenceCurvesResponse? = nil
    @State private var pipelineMetrics: PipelineMetricsResponse? = nil

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
                ToolbarItem(placement: .navigationBarLeading) {
                    if let signals = reclusterSignals, signals.suggestRecluster {
                        NavigationLink {
                            PoseInspectorView()
                                .environmentObject(apiClient)
                        } label: {
                            Label("Re-cluster suggested", systemImage: "exclamationmark.triangle.fill")
                                .foregroundColor(.orange)
                                .font(.caption.weight(.semibold))
                        }
                    }
                }
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
            reclusterSection
            overviewSection(m)
            perClassSection(m)
            falseAlarmSection(m)
            confidenceSection(m)
            thresholdSection(m)
            aucSection(m)
            confusionMatrixSection(m)
            if let pm = poseMetrics {
                phase2Section(pm)
            }
            if let curves = confidenceCurves {
                confidenceCurvesSection(curves)
            }
            if let pl = pipelineMetrics {
                pipelineSection(pl)
            }
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

    @ViewBuilder
    private var reclusterSection: some View {
        if let signals = reclusterSignals, signals.suggestRecluster {
            Section {
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundColor(.orange)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Re-cluster suggested")
                            .font(.subheadline.weight(.semibold))
                            .foregroundColor(.orange)
                        ForEach(signals.signalMessages, id: \.self) { msg in
                            Text(msg)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding(.vertical, 2)

                NavigationLink("Open Pose Inspector") {
                    PoseInspectorView()
                        .environmentObject(apiClient)
                }
                .font(.subheadline)
            } header: {
                Text("Pose model warning")
            } footer: {
                Text("One or more re-cluster signals tripped. Review the cluster inspector and run POST /train/pose to rebuild.")
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

    // MARK: - Phase 2 section

    @ViewBuilder
    private func phase2Section(_ pm: PoseMetricsResponse) -> some View {
        if let l2 = pm.layer2 {
            Section {
                metricRow("Clusters", value: "\(l2.nClusters)")
                metricRow("Train holds", value: "\(l2.trainSize)")
                metricRow("Val holds", value: "\(l2.valSize)")
                ForEach(l2.perClass) { row in
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Cluster \(row.clusterId)")
                            .font(.headline)
                        HStack {
                            labelValue("Prec", percent(row.precision))
                            Spacer()
                            labelValue("Recall", percent(row.recall))
                            Spacer()
                            labelValue("F1", format(row.f1))
                        }
                        .font(.caption)
                        Text("train: \(row.supportTrain)  •  val: \(row.supportVal)")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .padding(.vertical, 2)
                }
            } header: {
                Text("Phase 2 — Pose MLP (Layer 2)")
            } footer: {
                Text("Per-cluster precision/recall of the pose MLP classifier on the validation hold set.")
            }
        }

        if let l3 = pm.layer3 {
            Section {
                metricRow("Films evaluated", value: "\(l3.nFilms)")
                metricRow("Commit rate", value: percent(l3.commitRate))
                HStack {
                    Text("Commit-correct rate").bold()
                    Spacer()
                    Text(percent(l3.commitCorrectRate))
                        .bold()
                        .foregroundColor(.secondary)
                        .font(.system(.body, design: .monospaced))
                }
                metricRow("No-prefix rate", value: percent(l3.noPrefixRate))
                metricRow("Premature idle rate", value: percent(l3.prematureIdleRate))
                metricRow("Idle-while-live-prefix", value: percent(l3.idleWhileLivePrefixRate))
                if let s = l3.idleWhileLivePrefixSuccessRate {
                    metricRow("  └ success rate", value: percent(s))
                }
            } header: {
                Text("Phase 2 — End-to-end pipeline (Layer 3)")
            } footer: {
                Text("Commit-correct rate: fraction of corpus films that committed to the correct gesture. No-prefix: Phase 2 discarded the capture entirely.")
            }

            if !l3.perGesture.isEmpty {
                Section {
                    ForEach(l3.perGesture) { g in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(g.gestureId)
                                .font(.headline)
                            HStack {
                                labelValue("Commit%", percent(g.commitRate))
                                Spacer()
                                labelValue("Correct%", percent(g.commitCorrectRate))
                                Spacer()
                                labelValue("Recall", percent(g.recall))
                                Spacer()
                                labelValue("F1", format(g.f1))
                            }
                            .font(.caption)
                            Text("\(g.nFilms) films")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        .padding(.vertical, 2)
                    }
                } header: {
                    Text("Phase 2 — Per gesture")
                } footer: {
                    Text("Commit% = fraction of films with any commit. Correct% = fraction committed to the right gesture.")
                }
            }

            if let nmi = l3.nonModalExclusionImpact, !nmi.perClass.isEmpty {
                Section {
                    ForEach(nmi.perClass) { entry in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(entry.gestureId)
                                .font(.headline)
                            HStack {
                                labelValue("All films", percent(entry.recallAllFilms))
                                Spacer()
                                labelValue("Modal only", percent(entry.recallModalFilmsOnly))
                                Spacer()
                                labelValue("n_all", "\(entry.nAll)")
                                Spacer()
                                labelValue("n_modal", "\(entry.nModal)")
                            }
                            .font(.caption)
                        }
                        .padding(.vertical, 2)
                    }
                } header: {
                    Text("Phase 2 — Non-modal exclusion impact")
                } footer: {
                    Text(nmi.note)
                }
            }
        }
    }

    // MARK: - Stage 9: Confidence curves

    @ViewBuilder
    private func confidenceCurvesSection(_ curves: ConfidenceCurvesResponse) -> some View {
        confidenceCurveRows(
            title: "Phase 2 τ-pose curve",
            offline: curves.poseOffline,
            online: curves.poseOnline,
            nOnline: curves.nPoseSamples,
            footer: "Acceptance rate and conditional accuracy at each confidence threshold for the pose classifier. Offline = validation fold from last training run. On-device = entries logged from the device (\(curves.nPoseSamples) samples)."
        )

        confidenceCurveRows(
            title: "Phase 3 τ-phase3 curve",
            offline: curves.phase3Offline,
            online: curves.phase3Online,
            nOnline: curves.nPhase3Samples,
            footer: "Acceptance rate and conditional accuracy at each confidence threshold for the handfilm classifier. Offline = validation fold. On-device = entries logged from device (\(curves.nPhase3Samples) samples). Phase 3 fires once per committed cycle, so the on-device curve stabilises more slowly."
        )
    }

    @ViewBuilder
    private func confidenceCurveRows(
        title: String,
        offline: [ConfidenceCurvePoint],
        online: [ConfidenceCurvePoint],
        nOnline: Int,
        footer: String
    ) -> some View {
        if !offline.isEmpty || !online.isEmpty {
            Section {
                // Table header
                HStack {
                    Text("τ")
                        .frame(width: 36, alignment: .leading)
                    Spacer()
                    Text("Accept")
                        .frame(width: 54, alignment: .trailing)
                    Text("Acc")
                        .frame(width: 48, alignment: .trailing)
                    if !online.isEmpty {
                        Text("↕ Accept")
                            .frame(width: 60, alignment: .trailing)
                            .foregroundColor(.blue)
                        Text("↕ Acc")
                            .frame(width: 48, alignment: .trailing)
                            .foregroundColor(.blue)
                    }
                }
                .font(.caption2.weight(.semibold))
                .foregroundColor(.secondary)
                .padding(.vertical, 2)

                ForEach(offlineCurveRows(offline: offline, online: online)) { row in
                    HStack {
                        Text(String(format: "%.2f", row.tau))
                            .font(.system(.caption, design: .monospaced))
                            .frame(width: 36, alignment: .leading)
                        Spacer()
                        Text(percent(row.acceptOffline))
                            .font(.system(.caption, design: .monospaced))
                            .frame(width: 54, alignment: .trailing)
                        Text(percent(row.accOffline))
                            .font(.system(.caption, design: .monospaced))
                            .frame(width: 48, alignment: .trailing)
                        if !online.isEmpty {
                            Text(percent(row.acceptOnline))
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.blue)
                                .frame(width: 60, alignment: .trailing)
                            Text(percent(row.accOnline))
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.blue)
                                .frame(width: 48, alignment: .trailing)
                        }
                    }
                    .padding(.vertical, 1)
                }
            } header: {
                Text(title)
            } footer: {
                Text(footer)
            }
        }
    }

    private struct CurveTableRow: Identifiable {
        var id: Double { tau }
        let tau: Double
        let acceptOffline: Double?
        let accOffline: Double?
        let acceptOnline: Double?
        let accOnline: Double?
    }

    private func offlineCurveRows(
        offline: [ConfidenceCurvePoint],
        online: [ConfidenceCurvePoint]
    ) -> [CurveTableRow] {
        let offMap = Dictionary(uniqueKeysWithValues: offline.map { ($0.tau, $0) })
        let onMap = Dictionary(uniqueKeysWithValues: online.map { ($0.tau, $0) })
        let taus = Set(offline.map(\.tau)).union(online.map(\.tau)).sorted()
        return taus.map { tau in
            CurveTableRow(
                tau: tau,
                acceptOffline: offMap[tau]?.acceptanceRate,
                accOffline: offMap[tau]?.conditionalAccuracy,
                acceptOnline: onMap[tau]?.acceptanceRate,
                accOnline: onMap[tau]?.conditionalAccuracy
            )
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

    // MARK: - Pipeline section (Stage 11.3)

    @ViewBuilder
    private func pipelineSection(_ pl: PipelineMetricsResponse) -> some View {
        let trimImpactLarge = abs(pl.gateTrimImpact) > 0.05
        Section {
            LabeledContent("Films evaluated", value: "\(pl.nFilms)")
            LabeledContent("Gate open rate",  value: percent(pl.gateOpenRate))
            LabeledContent("Gate miss rate",  value: percent(pl.gateMissRate))
            if let buf = pl.avgBufferFraction {
                LabeledContent("Avg buffer / film", value: percent(buf))
            }
            Divider()
            LabeledContent("1+2 commit-correct",   value: percent(pl.commitCorrect12))
            LabeledContent("1+2+3 commit-correct", value: percent(pl.commitCorrect123))
            HStack {
                Text("Phase 3 lift")
                Spacer()
                Text(String(format: "%+.1f pp", pl.phase3Lift * 100))
                    .foregroundColor(pl.phase3Lift >= 0 ? .green : .red)
            }
            Divider()
            HStack {
                Text("Gate-trim impact")
                Spacer()
                Text(String(format: "%+.1f pp", pl.gateTrimImpact * 100))
                    .foregroundColor(trimImpactLarge ? .red : .secondary)
            }
            if trimImpactLarge {
                Text("⚠ Gate-trim impact > 5 pp — check T_open / K_open thresholds.")
                    .font(.caption)
                    .foregroundColor(.red)
            }
            LabeledContent("Full-film baseline", value: percent(pl.fullFilmCommitCorrect))
            if let ts = pl.evaluatedAt {
                LabeledContent("Evaluated", value: formatDate(ts))
            }
        } header: {
            Text("Pipeline (1+2+3) — Layer 4")
        }

        if !pl.perGesture.isEmpty {
            Section("Per gesture — pipeline") {
                Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 8, verticalSpacing: 4) {
                    GridRow {
                        Text("Gesture").font(.caption2).foregroundColor(.secondary)
                        Text("Gate%").font(.caption2).foregroundColor(.secondary)
                        Text("1+2%").font(.caption2).foregroundColor(.secondary)
                        Text("1+2+3%").font(.caption2).foregroundColor(.secondary)
                    }
                    ForEach(pl.perGesture) { g in
                        GridRow {
                            Text(g.gestureId).font(.caption2)
                            Text(percent(g.gateOpenRate)).font(.caption2.monospacedDigit())
                            Text(percent(g.commitCorrect12)).font(.caption2.monospacedDigit())
                            Text(percent(g.commitCorrect123))
                                .font(.caption2.monospacedDigit())
                                .foregroundColor(g.commitCorrect123 >= 0.85 ? .green : .orange)
                        }
                    }
                }
                .padding(.vertical, 4)
            }
        }

        Section {
            Button("Re-evaluate pipeline") {
                Task { try? await apiClient.triggerPipelineEvaluation() }
            }
        } footer: {
            Text("Runs Phase 1 gate simulation + Phase 2 + Phase 3 on the stored corpus. "
                 + "Requires gate_calibration.json, pose model, and Phase 3 model.")
                .font(.caption)
        }
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
            async let listTask = apiClient.fetchMetricsList()
            async let poseStatusTask = apiClient.fetchPoseTrainingStatus()

            let (list, poseStatus) = try await (listTask, poseStatusTask)
            history = list.models
            reclusterSignals = poseStatus.reclusterSignals

            let id = selectedModelId ?? list.models.first?.modelId
            if let id = id {
                let detail = try await apiClient.fetchMetrics(modelId: id)
                metrics = detail
                selectedModelId = detail.modelId
            } else {
                metrics = nil
            }

            // Load Phase 2 metrics (Stage 8.3)
            poseMetrics = try? await apiClient.fetchPoseMetrics()

            // Load confidence curves (Stage 9.3)
            confidenceCurves = try? await apiClient.fetchConfidenceCurves()

            // Load pipeline metrics (Stage 11.3)
            pipelineMetrics = try? await apiClient.fetchPipelineMetrics()
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

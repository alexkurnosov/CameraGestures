// The metadata layers under the session player's skeleton: gate, hold, Phase 3
// and commit, each with its own toggle. The row of toggles also holds the
// video's, which the player panel uses.
//
// All four charts show the same fixed window centred on the playhead, and
// dragging on them moves the playhead. In a 10-minute session the slider moves
// about 1.5 s per point, which is too coarse to land on a gate transition.

#if CG_SESSION_CAPTURE

import SwiftUI
import UIKit
import CameraGestures

private enum SessionLayerKind: String, CaseIterable, Identifiable {
    case gate = "Gate", hold = "Hold", phase3 = "Phase 3", commit = "Commit"
    var id: String { rawValue }
}

/// Colours checked for colour-blind separation as one palette, light and dark.
private enum LayerColor {
    static let energy   = dynamic(0x2a78d6, 0x3987e5)
    static let smoothed = dynamic(0xeb6834, 0xd95926)
    static let gate     = dynamic(0x1baf7a, 0x199e70)
    static let cooldown = dynamic(0xeda100, 0xc98500)
    static let emitted  = dynamic(0x6250d6, 0x9085e9)
    static let critical = dynamic(0xd03b3b, 0xe66767)

    private static func dynamic(_ light: UInt32, _ dark: UInt32) -> Color {
        Color(UIColor { $0.userInterfaceStyle == .dark ? uiColor(dark) : uiColor(light) })
    }

    private static func uiColor(_ rgb: UInt32) -> UIColor {
        UIColor(red: CGFloat(rgb >> 16 & 0xff) / 255,
                green: CGFloat(rgb >> 8 & 0xff) / 255,
                blue: CGFloat(rgb & 0xff) / 255,
                alpha: 1)
    }
}

struct SessionLayersView: View {
    @ObservedObject var player: SessionPlayer

    @State private var visible = Set(SessionLayerKind.allCases)
    @State private var dragStartTime: TimeInterval?
    @State private var scrollingDrag = false

    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 6) {
                ForEach(SessionLayerKind.allCases) { kind in
                    toggle(kind.rawValue, on: visible.contains(kind)) {
                        if visible.contains(kind) { visible.remove(kind) } else { visible.insert(kind) }
                    }
                }
                // A session without video says so in the player panel.
                toggle("Video", on: player.video != nil && player.showsVideo) {
                    player.showsVideo.toggle()
                }
                .disabled(player.video == nil)
            }
            if let layers = player.layers {
                GeometryReader { proxy in
                    VStack(spacing: 3) {
                        ForEach(SessionLayerKind.allCases.filter(visible.contains)) { kind in
                            chart(kind, layers)
                        }
                    }
                    .contentShape(Rectangle())
                    // Beside the page's scrolling, which a plain gesture would block.
                    .simultaneousGesture(scrub(width: proxy.size.width))
                }
                .frame(height: CGFloat(visible.count) * Self.rowHeight)
            } else {
                ProgressView()
                    .frame(height: CGFloat(visible.count) * Self.rowHeight)
            }
        }
    }

    private static let chartHeight: CGFloat = 54
    /// Chart, its heading and the gap between rows.
    private static let rowHeight: CGFloat = chartHeight + 14 + 3

    private func toggle(_ title: String, on: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.caption.weight(on ? .semibold : .regular))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 4)
                .background(on ? Color.primary : Color.clear)
                .foregroundColor(on ? Color(.systemBackground) : .secondary)
                .overlay(Capsule().stroke(Color.secondary.opacity(0.5)))
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(on ? .isSelected : [])
    }

    private func scrub(width: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: 8)
            .onChanged { drag in
                if scrollingDrag { return }
                if dragStartTime == nil,
                   abs(drag.translation.height) > abs(drag.translation.width) {
                    // A mostly vertical drag scrolls the page.
                    scrollingDrag = true
                    return
                }
                let start = dragStartTime ?? player.time
                dragStartTime = start
                player.seek(to: start - Double(drag.translation.width / width) * LayerScale.window)
            }
            .onEnded { _ in
                dragStartTime = nil
                scrollingDrag = false
            }
    }

    private func chart(_ kind: SessionLayerKind, _ layers: SessionLayers) -> some View {
        let time = player.time
        return VStack(spacing: 0) {
            HStack(spacing: 8) {
                Text(kind.rawValue)
                    .font(.system(size: 11, weight: .semibold))
                Spacer(minLength: 0)
                Text(readout(kind, layers, time))
                    .font(.system(size: 11).monospacedDigit())
                    .foregroundColor(.secondary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
            }
            .frame(height: 14)
            Canvas { context, size in
                let scale = LayerScale(time: time, width: size.width)
                switch kind {
                case .gate:   drawGate(layers, scale, size, &context)
                case .hold:   drawHold(layers, scale, size, &context)
                case .phase3: drawPhase3(layers, scale, size, &context)
                case .commit: drawCommit(layers, scale, size, &context)
                }
                if !isUnavailable(kind, layers) {
                    let x = scale.x(time)
                    context.fill(Path(CGRect(x: x - 0.5, y: 0, width: 1, height: size.height)),
                                 with: .color(.primary))
                }
            }
            .frame(height: Self.chartHeight)
            .background(Color.gray.opacity(0.15))
            .cornerRadius(4)
        }
    }

    private func isUnavailable(_ kind: SessionLayerKind, _ layers: SessionLayers) -> Bool {
        switch kind {
        case .gate: return layers.gateUnavailable != nil
        case .hold: return layers.holdUnavailable != nil
        case .phase3, .commit: return false
        }
    }

    // MARK: - Readouts

    /// What the layer says at the playhead, in words.
    private func readout(_ kind: SessionLayerKind, _ layers: SessionLayers,
                         _ time: TimeInterval) -> String {
        let recent: TimeInterval = 1.5
        let row = SessionLayers.lastIndex(in: layers.rows, atOrBefore: time) { $0.time }
            .map { layers.rows[$0] }

        switch kind {
        case .gate:
            guard layers.gateUnavailable == nil, let row else { return "no data" }
            var parts = [row.gateOpen ? "open" : "closed",
                         "energy " + (row.rawEnergy.map { String(format: "%.2f", $0) } ?? "no value"),
                         "buffer \(row.bufferCount)"]
            if let end = SessionLayers.lastIndex(in: layers.cycleEnds, atOrBefore: time, by: { $0.time }),
               time - layers.cycleEnds[end].time < recent {
                parts.append("ended: \(layers.cycleEnds[end].text)")
            }
            return parts.joined(separator: ", ")

        case .hold:
            guard layers.holdUnavailable == nil else { return "no data" }
            var parts = ["energy " + (row?.smoothedEnergy.map { String(format: "%.2f", $0) } ?? "no value")]
            if let index = SessionLayers.lastIndex(in: layers.holds, atOrBefore: time, by: { $0.end }) {
                let hold = layers.holds[index]
                parts.append("last: \(poseLabel(hold)) \(hold.action)")
            }
            return parts.joined(separator: ", ")

        case .phase3:
            let prediction = SessionLayers.lastIndex(in: layers.predictions, atOrBefore: time) { $0.time }
                .map { layers.predictions[$0] }
            let skip = SessionLayers.lastIndex(in: layers.cycleSkips, atOrBefore: time) { $0.time }
                .map { layers.cycleSkips[$0] }
            if let skip, skip.time > prediction?.time ?? -.infinity {
                return "skipped: \(skip.text)"
            }
            guard let prediction else { return "none yet" }
            let path = prediction.restricted
                ? "masked [\(prediction.candidates.joined(separator: ", "))]" : "unrestricted"
            let result: String
            if prediction.accepted {
                result = String(format: "%@ %.2f", prediction.gestureId ?? "?", prediction.confidence)
            } else if let best = prediction.gestureId {
                result = String(format: "rejected: %@ %.2f", best, prediction.confidence)
            } else {
                result = "rejected, confidence not recorded"
            }
            return "\(result), \(path)"

        case .commit:
            var parts: [String] = []
            if let index = SessionLayers.lastIndex(in: layers.cooldowns, atOrBefore: time, by: { $0.start }),
               time < layers.cooldowns[index].end {
                parts.append("cooldown (\(layers.cooldowns[index].gestureId))")
            }
            if let index = SessionLayers.lastIndex(in: layers.suppressed, atOrBefore: time, by: { $0.detectedAt }),
               time - layers.suppressed[index].detectedAt < recent {
                let gesture = layers.suppressed[index]
                parts.append(String(format: "suppressed %@ %.2f, replaced by %@",
                                    gesture.gestureId, gesture.confidence, gesture.replacedBy))
            }
            if let index = SessionLayers.lastIndex(in: layers.commits, atOrBefore: time, by: { $0.time }),
               time - layers.commits[index].time < recent {
                parts.append("fired: \(layers.commits[index].text)")
            }
            return parts.isEmpty ? "nothing pending" : parts.joined(separator: ", ")
        }
    }

    private func poseLabel(_ hold: SessionLayers.Hold) -> String {
        hold.poseId < 0 ? "pose failed" : String(format: "pose %d %.2f", hold.poseId, hold.confidence)
    }

    // MARK: - Gate layer

    private func drawGate(_ layers: SessionLayers, _ scale: LayerScale, _ size: CGSize,
                          _ context: inout GraphicsContext) {
        if let reason = layers.gateUnavailable {
            drawUnavailable(reason, size, &context)
            return
        }
        for span in layers.gateSpans where scale.isVisible(span.start, span.end) {
            fill(span.start, span.end, y: 0, height: size.height, LayerColor.gate.opacity(0.28), scale, &context)
        }
        if let config = layers.config {
            drawThreshold(config.tOpen, "T_open", labelBelow: false, size, &context)
            drawThreshold(config.tClose, "T_close", labelBelow: true, size, &context)
        }
        drawEnergy(layers.rows, scale, size, LayerColor.energy, &context) { $0.rawEnergy }
    }

    // MARK: - Hold layer

    private func drawHold(_ layers: SessionLayers, _ scale: LayerScale, _ size: CGSize,
                          _ context: inout GraphicsContext) {
        if let reason = layers.holdUnavailable {
            drawUnavailable(reason, size, &context)
            return
        }
        var labels = LabelRow()
        for hold in layers.holds where scale.isVisible(hold.start, hold.end) {
            let color: Color = hold.accepted ? LayerColor.smoothed : .gray
            fill(hold.start, hold.end, y: 0, height: size.height, color.opacity(0.28), scale, &context)
            labels.draw(poseLabel(hold), x: scale.x(hold.end), y: size.height - 6, .secondary, size, &context)
        }
        if let config = layers.config {
            drawThreshold(config.tHold, "T_hold", labelBelow: false, size, &context)
        }
        drawEnergy(layers.rows, scale, size, LayerColor.smoothed, &context) { $0.smoothedEnergy }
    }

    // MARK: - Phase 3 layer

    private func drawPhase3(_ layers: SessionLayers, _ scale: LayerScale, _ size: CGSize,
                            _ context: inout GraphicsContext) {
        let y = { (confidence: Float) in size.height - 3 - CGFloat(confidence) * (size.height - 14) }

        if let config = layers.config {
            // The masked path applies its own threshold, used only in holds mode.
            let masked = config.holdsEnabled && !config.bypassPhase2
            let same = !masked || config.tauPhase3Confidence == config.confidenceThreshold
            drawConfidenceLine(config.confidenceThreshold, same ? "tau" : "tau U", y(config.confidenceThreshold), size, &context)
            if !same {
                drawConfidenceLine(config.tauPhase3Confidence, "tau M", y(config.tauPhase3Confidence), size, &context)
            }
        }

        for skip in layers.cycleSkips where scale.isVisible(skip.time, skip.time) {
            let x = scale.x(skip.time)
            context.fill(Path(CGRect(x: x - 0.5, y: size.height - 6, width: 1, height: 6)), with: .color(.gray))
        }

        var labels = LabelRow()
        for prediction in layers.predictions where scale.isVisible(prediction.time, prediction.time) {
            let x = scale.x(prediction.time)
            let top = y(prediction.confidence)
            let color: Color = prediction.accepted ? LayerColor.emitted : .gray
            var stem = Path()
            stem.move(to: CGPoint(x: x, y: size.height))
            stem.addLine(to: CGPoint(x: x, y: top))
            context.stroke(stem, with: .color(color), lineWidth: 1.5)
            let dot = Path(ellipseIn: CGRect(x: x - 3, y: top - 3, width: 6, height: 6))
            if prediction.accepted {
                context.fill(dot, with: .color(color))
            } else {
                context.stroke(dot, with: .color(color), lineWidth: 1.5)
            }
            let path = prediction.restricted ? "M" : "U"
            let label = prediction.gestureId.map { String(format: "%@ %.2f %@", $0, prediction.confidence, path) }
                ?? "rejected \(path)"
            labels.draw(label, x: x, y: max(top, 6), prediction.accepted ? .secondary : .gray, size, &context)
        }
    }

    private func drawConfidenceLine(_ value: Float, _ name: String, _ y: CGFloat, _ size: CGSize,
                                    _ context: inout GraphicsContext) {
        drawDashedLine(y: y, size, &context)
        context.draw(smallText("\(name) \(thresholdText(value))", .secondary),
                     at: CGPoint(x: 3, y: y + 6), anchor: .leading)
    }

    // MARK: - Commit layer

    private func drawCommit(_ layers: SessionLayers, _ scale: LayerScale, _ size: CGSize,
                            _ context: inout GraphicsContext) {
        for cooldown in layers.cooldowns where scale.isVisible(cooldown.start, cooldown.end) {
            fill(cooldown.start, cooldown.end, y: 0, height: size.height, LayerColor.cooldown.opacity(0.3), scale, &context)
        }
        for wait in layers.commitWaits where scale.isVisible(wait.start, wait.end) {
            fill(wait.start, wait.end, y: 5, height: 6, LayerColor.energy, scale, &context)
        }
        for wait in layers.minBufferWaits where scale.isVisible(wait.start, wait.end) {
            fill(wait.start, wait.end, y: 14, height: 6, LayerColor.emitted, scale, &context)
        }
        for commit in layers.commits where scale.isVisible(commit.time, commit.time) {
            let x = scale.x(commit.time)
            var diamond = Path()
            diamond.move(to: CGPoint(x: x, y: 23))
            diamond.addLine(to: CGPoint(x: x + 5, y: 28))
            diamond.addLine(to: CGPoint(x: x, y: 33))
            diamond.addLine(to: CGPoint(x: x - 5, y: 28))
            diamond.closeSubpath()
            context.fill(diamond, with: .color(.primary))
        }
        var labels = LabelRow()
        for gesture in layers.suppressed where scale.isVisible(gesture.detectedAt, gesture.detectedAt) {
            let x = scale.x(gesture.detectedAt)
            let y = size.height - 9
            var cross = Path()
            cross.move(to: CGPoint(x: x - 4, y: y - 4))
            cross.addLine(to: CGPoint(x: x + 4, y: y + 4))
            cross.move(to: CGPoint(x: x + 4, y: y - 4))
            cross.addLine(to: CGPoint(x: x - 4, y: y + 4))
            context.stroke(cross, with: .color(LayerColor.critical), lineWidth: 2)
            labels.draw(gesture.gestureId, x: x + 3, y: y, .secondary, size, &context)
        }
    }

    // MARK: - Drawing helpers

    private func fill(_ start: TimeInterval, _ end: TimeInterval, y: CGFloat, height: CGFloat, _ color: Color,
                      _ scale: LayerScale, _ context: inout GraphicsContext) {
        let x = scale.x(start)
        // A wait of zero length is still drawn, as a thin bar.
        let rect = CGRect(x: x, y: y, width: max(scale.x(end) - x, 1.5), height: height)
        context.fill(Path(rect), with: .color(color))
    }

    /// The rows in view as one line. A row with no value breaks the line, and so
    /// does a stretch with no rows, so a gap is never drawn over.
    private func drawEnergy(_ rows: [SessionLayers.Row], _ scale: LayerScale, _ size: CGSize, _ color: Color,
                            _ context: inout GraphicsContext, value: (SessionLayers.Row) -> Float?) {
        let first = SessionLayers.lastIndex(in: rows, atOrBefore: scale.start) { $0.time } ?? 0
        var path = Path()
        var previous: TimeInterval?
        for row in rows[first...] {
            guard let energy = value(row) else { previous = nil; continue }
            let point = CGPoint(x: scale.x(row.time), y: energyY(energy, size))
            if let previous, row.time - previous <= 0.35 {
                path.addLine(to: point)
            } else {
                path.move(to: point)
            }
            previous = row.time
            if row.time > scale.end { break }
        }
        context.stroke(path, with: .color(color), lineWidth: 1.5)
    }

    /// A fixed logarithmic axis from 0.1 to 100. The thresholds sit near 1 and
    /// two-hand energy near 30, so a linear axis cannot show both, and a fixed
    /// axis does not jump during playback.
    private func energyY(_ energy: Float, _ size: CGSize) -> CGFloat {
        let position = (log10(max(energy, 0.1)) + 1) / 3
        return size.height - 2 - CGFloat(min(position, 1)) * (size.height - 4)
    }

    private func drawThreshold(_ value: Float, _ name: String, labelBelow: Bool, _ size: CGSize,
                               _ context: inout GraphicsContext) {
        let y = energyY(value, size)
        drawDashedLine(y: y, size, &context)
        context.draw(smallText("\(name) \(thresholdText(value))", .secondary),
                     at: CGPoint(x: 3, y: labelBelow ? min(y + 6, size.height - 5) : max(y - 6, 5)),
                     anchor: .leading)
    }

    private func drawDashedLine(y: CGFloat, _ size: CGSize, _ context: inout GraphicsContext) {
        var line = Path()
        line.move(to: CGPoint(x: 0, y: y))
        line.addLine(to: CGPoint(x: size.width, y: y))
        context.stroke(line, with: .color(.gray), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
    }

    /// As few digits as the value needs, so a stored 2.0999999 reads 2.1.
    private func thresholdText(_ value: Float) -> String {
        String(format: "%g", (value * 1000).rounded() / 1000)
    }

    private func drawUnavailable(_ reason: String, _ size: CGSize, _ context: inout GraphicsContext) {
        var hatch = Path()
        var x = -size.height
        while x < size.width {
            hatch.move(to: CGPoint(x: x, y: size.height))
            hatch.addLine(to: CGPoint(x: x + size.height, y: 0))
            x += 10
        }
        context.stroke(hatch, with: .color(.gray.opacity(0.35)), lineWidth: 1)
        context.draw(Text("Unavailable: \(reason)").font(.system(size: 11, weight: .semibold)),
                     at: CGPoint(x: size.width / 2, y: size.height / 2))
    }
}

/// Time on the charts: the window around the playhead, across the chart's width.
private struct LayerScale {
    static let window: TimeInterval = 8

    let time: TimeInterval
    let width: CGFloat

    var start: TimeInterval { time - Self.window / 2 }
    var end: TimeInterval { time + Self.window / 2 }

    func x(_ t: TimeInterval) -> CGFloat {
        width / 2 + CGFloat((t - time) / Self.window) * width
    }

    func isVisible(_ a: TimeInterval, _ b: TimeInterval) -> Bool {
        b >= start - 1 && a <= end + 1
    }
}

private func smallText(_ string: String, _ color: Color) -> Text {
    Text(string).font(.system(size: 9)).foregroundColor(color)
}

/// Draws labels left to right and skips one that would overlap the previous,
/// because predictions can come closer together than their labels are wide.
private struct LabelRow {
    private var right: CGFloat = -.infinity

    mutating func draw(_ string: String, x: CGFloat, y: CGFloat, _ color: Color, _ size: CGSize,
                       _ context: inout GraphicsContext) {
        let text = context.resolve(smallText(string, color))
        let width = text.measure(in: size).width
        var left = x + 3
        if left + width > size.width { left = x - width - 3 }
        guard left >= right + 3, left >= 0 else { return }
        context.draw(text, at: CGPoint(x: left, y: y), anchor: .leading)
        right = left + width
    }
}

#endif

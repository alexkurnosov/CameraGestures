import SwiftUI
import HandGestureTypes
import GestureModelModule

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

                    Text("\u{2022}")
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Text("\(example.handfilm.frames.count) frames")
                        .font(.caption2)
                        .foregroundColor(.secondary)

                    Text("\u{2022}")
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

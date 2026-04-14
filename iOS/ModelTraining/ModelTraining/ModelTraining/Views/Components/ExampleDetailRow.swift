import SwiftUI
import HandGestureTypes

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

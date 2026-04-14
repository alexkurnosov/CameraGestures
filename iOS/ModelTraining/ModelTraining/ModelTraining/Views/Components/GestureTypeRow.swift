import SwiftUI
import GestureModelModule
import HandGestureTypes

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
                            Text("\u{2022}")
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

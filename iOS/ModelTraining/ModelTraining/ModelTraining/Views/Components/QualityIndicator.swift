import SwiftUI

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
        if count >= 20 { return "\u{2713}" }
        else if count >= 10 { return "~" }
        else { return "!" }
    }
}

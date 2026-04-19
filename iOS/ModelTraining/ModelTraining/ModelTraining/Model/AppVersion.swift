import Foundation

enum AppVersion {
    static let string: String = {
        guard let url = Bundle.main.url(forResource: "VERSION", withExtension: "txt"),
              let raw = try? String(contentsOf: url, encoding: .utf8) else {
            return "unknown"
        }
        return raw.trimmingCharacters(in: .whitespacesAndNewlines)
    }()
}

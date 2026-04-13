import Foundation
import Combine
import HandGestureRecognizingFramework

@MainActor
class GestureRecognizerWrapper: ObservableObject {
    let recognizer: HandGestureRecognizing

    @Published var isRecognizing: Bool = false
    @Published var currentGesture: String?
    @Published var confidence: Float = 0.0
    @Published var lastError: String?

    init(recognizer: HandGestureRecognizing) {
        self.recognizer = recognizer
    }
}

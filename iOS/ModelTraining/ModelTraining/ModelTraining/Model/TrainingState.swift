import Foundation
import HandGestureTypes

enum TrainingState: Equatable {
    case idle
    case training
    case done(ModelMetrics)
    case failed(String)

    static func == (lhs: TrainingState, rhs: TrainingState) -> Bool {
        switch (lhs, rhs) {
        case (.idle, .idle), (.training, .training): return true
        case (.done, .done): return true
        case (.failed(let a), .failed(let b)): return a == b
        default: return false
        }
    }
}

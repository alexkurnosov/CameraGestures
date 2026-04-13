import SwiftUI
import HandGestureTypes
import HandGestureRecognizingFramework

@main
struct ModelTrainingApp: App {

    // MARK: - State Management

    @StateObject private var gestureRecognizer = GestureRecognizerWrapper(recognizer: HandGestureRecognizing())
    @StateObject private var trainingDataManager = TrainingDataManager()
    @StateObject private var appSettings = AppSettings()
    @StateObject private var gestureRegistry = GestureRegistry()
    @StateObject private var apiClient = GestureModelAPIClient()

    // MARK: - Scene Configuration

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(gestureRecognizer)
                .environmentObject(trainingDataManager)
                .environmentObject(appSettings)
                .environmentObject(gestureRegistry)
                .environmentObject(apiClient)
                .preferredColorScheme(appSettings.colorScheme)
                .onAppear {
                    trainingDataManager.apiClient = apiClient
                }
        }
    }
}

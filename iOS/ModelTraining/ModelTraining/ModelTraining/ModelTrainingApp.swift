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
    @StateObject private var serverManager = ServerTrainingManager()

    // MARK: - Scene Configuration

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(gestureRecognizer)
                .environmentObject(trainingDataManager)
                .environmentObject(appSettings)
                .environmentObject(gestureRegistry)
                .environmentObject(apiClient)
                .environmentObject(serverManager)
                .preferredColorScheme(appSettings.colorScheme)
                .onAppear {
                    trainingDataManager.apiClient = apiClient
                    trainingDataManager.gestureRecognizer = gestureRecognizer
                    trainingDataManager.appSettings = appSettings
                    serverManager.configure(
                        apiClient: apiClient,
                        appSettings: appSettings,
                        gestureRecognizer: gestureRecognizer
                    )
                }
        }
    }
}

import SwiftUI
import Combine
import HandGestureTypes
import HandGestureRecognizingFramework

struct ContentView: View {
    @EnvironmentObject var gestureRecognizer: GestureRecognizerWrapper
    @EnvironmentObject var trainingDataManager: TrainingDataManager
    @EnvironmentObject var appSettings: AppSettings

    @State private var selectedTab = 0
    @State private var showingAlert = false
    @State private var alertMessage = ""
    @State private var cancellables = Set<AnyCancellable>()

    var body: some View {
        TabView(selection: $selectedTab) {
            CameraView()
                .tabItem {
                    Image(systemName: "camera.fill")
                    Text("Camera")
                }
                .tag(0)

            TrainingView()
                .tabItem {
                    Image(systemName: "hand.raised.fill")
                    Text("Training")
                }
                .tag(1)

            GestureListView()
                .tabItem {
                    Image(systemName: "list.bullet")
                    Text("Gestures")
                }
                .tag(2)

            MetricsView()
                .tabItem {
                    Image(systemName: "chart.bar.xaxis")
                    Text("Metrics")
                }
                .tag(3)

            SettingsView()
                .tabItem {
                    Image(systemName: "gear")
                    Text("Settings")
                }
                .tag(4)
        }
        .accentColor(.blue)
        .onAppear {
            subscribeToStatus()
            initializeRecognizer()
        }
        .alert("System Alert", isPresented: $showingAlert) {
            Button("OK") { }
        } message: {
            Text(alertMessage)
        }
    }

    // MARK: - Private

    private func initializeRecognizer() {
        Task {
            do {
                try await gestureRecognizer.initialize(appSettings: appSettings)
            } catch {
                await MainActor.run {
                    showAlert("Initialization failed: \(error.localizedDescription)")
                }
            }
        }
    }

    private func subscribeToStatus() {
        gestureRecognizer.statusChanged
            .receive(on: DispatchQueue.main)
            .sink { [self] status in
                switch status {
                case .error(let error):
                    showAlert("Error: \(error)")
                case .running:
                    print("Gesture recognition started")
                case .idle:
                    print("Gesture recognition stopped")
                default:
                    break
                }
            }
            .store(in: &cancellables)
    }

    private func showAlert(_ message: String) {
        alertMessage = message
        showingAlert = true
    }
}

// MARK: - Preview

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .environmentObject(GestureRecognizerWrapper(recognizer: HandGestureRecognizing()))
            .environmentObject(TrainingDataManager())
            .environmentObject(AppSettings())
            .environmentObject(GestureRegistry())
            .environmentObject(GestureModelAPIClient())
    }
}

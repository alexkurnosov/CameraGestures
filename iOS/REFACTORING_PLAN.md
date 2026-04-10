# Refactoring Plan: Move Business Logic from Views to Model Layer

## Context

The iOS ModelTraining app has ~2700 lines of View code (CameraView, TrainingView, HandFilmsView) containing business logic that should live in a Model layer: async networking, recognizer lifecycle, polling loops, state machines, and callback management. Additionally, `ModelTrainingApp.swift` (683 lines) bundles 5 classes + 6 DTOs that need splitting. There's also a **callback overwrite bug** where 3 views each set `gestureDetectionCallback` on the recognizer -- last one wins.

## New Directory Structure

```
ModelTraining/
  Model/
    DTOs/
      Point3DDTO.swift
      HandShotDTO.swift
      HandFilmDTO.swift
      TrainingExampleDTO.swift
      TrainingDatasetDTO.swift
      FailedHandFilmDTO.swift
    TrainingState.swift
    TrainingDataManager.swift
    AppSettings.swift
    GestureRecognizerWrapper.swift   (enriched with Combine publishers)
    TrainingSeriesCoordinator.swift
    CameraViewModel.swift            (NEW)
    ServerTrainingManager.swift      (NEW)
    FilmPlaybackManager.swift        (NEW - Phase 2)
  Views/  (unchanged structure, but views become thin)
  Networking/  (unchanged)
  ModelTrainingApp.swift  (~33 lines - just @main + @StateObject declarations)
```

## Phase 1A: Extract Files from ModelTrainingApp.swift (no logic changes)

Move each class/enum/struct to its own file under `Model/`:

| Source (ModelTrainingApp.swift) | Destination |
|---|---|
| Lines 38-297: `TrainingDataManager` | `Model/TrainingDataManager.swift` |
| Lines 301-315: `TrainingState` enum | `Model/TrainingState.swift` |
| Lines 319-387: `AppSettings` | `Model/AppSettings.swift` |
| Lines 392-403: `GestureRecognizerWrapper` | `Model/GestureRecognizerWrapper.swift` |
| Lines 414-544: `TrainingSeriesCoordinator` | `Model/TrainingSeriesCoordinator.swift` |
| Lines 549-682: All DTOs | `Model/DTOs/*.swift` (one file each) |

`ModelTrainingApp.swift` becomes just the `@main struct` (~33 lines).

**Verify**: Build succeeds with no logic changes.

## Phase 1B: Fix Callback Overwrite Bug via GestureRecognizerWrapper

**Problem**: `ContentView:61`, `CameraView:594`, `TrainingView:752` all overwrite `gestureDetectionCallback` -- only the last one wins.

**Solution**: Enrich `GestureRecognizerWrapper` with Combine publishers:

```swift
@MainActor
class GestureRecognizerWrapper: ObservableObject {
    let recognizer: HandGestureRecognizing
    
    // Combine publishers -- set callbacks ONCE, publish to many subscribers
    let gestureDetected = PassthroughSubject<DetectedGesture, Never>()
    let handTrackingUpdate = PassthroughSubject<HandShot, Never>()
    let statusChanged = PassthroughSubject<GestureRecognizingStatus, Never>()
    
    @Published var isRecognizing = false
    // ...existing properties...
    
    func setupCallbacks() {
        recognizer.gestureDetectionCallback = { [weak self] gesture in
            DispatchQueue.main.async { self?.gestureDetected.send(gesture) }
        }
        recognizer.handTrackingUpdateCallback = { [weak self] handshot in
            DispatchQueue.main.async { self?.handTrackingUpdate.send(handshot) }
        }
        recognizer.statusChangeCallback = { [weak self] status in
            DispatchQueue.main.async { self?.statusChanged.send(status) }
        }
    }
}
```

Move recognizer initialization from `ContentView.setupGestureRecognizer()` into `GestureRecognizerWrapper.initialize(config:)`.

**Verify**: ContentView becomes a pure TabView host (~30 lines). No view sets callbacks directly anymore.

## Phase 1C: Create CameraViewModel

New `@MainActor class CameraViewModel: ObservableObject` that owns everything currently in CameraView except pure UI layout.

**State moved from CameraView @State to CameraViewModel @Published**:
- `isRecognitionActive`, `currentGesture`, `recentGestures`, `recognitionHandPoints`
- `stats`, `cameraPermissionGranted`, `showModelNotTrainedBanner`
- `captureWindow`, `pauseInterval`

**Computed properties moved**:
- `isModelTrained`, `canStartTraining`, `displayPoints`, `previewIsActive`

**Methods moved**:
- `checkCameraPermission()`, `startRecognition()`, `startTrainingSeries()`
- `stopAll()`, `clearGestures()`
- Stats polling (replace leaked `Timer.scheduledTimer` with cancellable `Task`)
- Subscribe to `gestureRecognizerWrapper.gestureDetected` and `.handTrackingUpdate`

**CameraView keeps only**:
- `@State showingPermissionAlert`, `recPulse` (pure UI state)
- All `@ViewBuilder` layout code
- Button actions delegate to `viewModel.startRecognition()` etc.

**Configuration pattern**: `@StateObject` on CameraView, configured via `.onAppear`:
```swift
viewModel.configure(recognizer:, dataManager:, settings:, registry:)
```

**Result**: CameraView shrinks from ~893 to ~300 lines.

## Phase 1D: Create ServerTrainingManager

New `@MainActor class ServerTrainingManager: ObservableObject` that owns all server training workflow from TrainingView.

**State moved from TrainingView @State**:
- `serverStatus`, `isPollingStatus`, `isDownloadingModel`, `isWipingModel`, `serverActionError`

**Methods moved**:
- `refreshServerStatus()`, `triggerServerTraining()`, `downloadModelFromServer()`
- `wipeServerModel()`, `startPollingStatus()`
- `statusPollingTask` becomes a stored property with cancellation in `deinit`

**Also move to TrainingDataManager** (it already owns `isCollecting`, `currentGestureId`, `trainingState`):
- `startCollection()` / `stopCollection()` logic
- `startTraining()` (local model training)
- `handleTrainingGesture()` (collection progress tracking -- add `@Published currentSamples`, `collectionProgress`, `targetSamples`)

**TrainingView keeps only**:
- `@State` for sheets/alerts: `showingNewDatasetAlert`, `showingAddGestureSheet`, `showingMetricsSheet`, `showingTrainingError`, `showingServerError`, `showingWipeModelAlert`
- All `@ViewBuilder` UI sections
- Button actions delegate to `serverManager.*` or `trainingDataManager.*`

**Result**: TrainingView shrinks from ~1146 to ~550 lines.

## Phase 2: Create FilmPlaybackManager (lower priority)

New `@MainActor class FilmPlaybackManager: ObservableObject` for HandFilmsView.

**Moves**: `currentFrameIndex`, `isPlaying`, `currentIndex`, `filterGestureId`, `filteredExamples`, `currentPoints`, playback Timer, `deleteCurrentExample()`, navigation logic.

**Result**: HandFilmsView shrinks from ~557 to ~300 lines.

## Summary of What Stays in Views

Views keep ONLY:
- `@State` for UI chrome: sheet/alert presentation booleans, `recPulse` animation
- `@ViewBuilder` layout code
- Button actions that call one method on a model object
- `@EnvironmentObject` / `@StateObject` declarations

Views do NOT:
- Set callbacks on the recognizer
- Create `Task {}` blocks with async work
- Manage `Timer` instances
- Call API client methods directly
- Track business state in `@State`

## Verification

After each phase:
1. Build succeeds (`xcodebuild` or Xcode)
2. Run the app -- camera tab works, training tab works, films tab works
3. Test: start prediction, start training series, trigger server training, download model
4. Verify no callback overwrite (both camera prediction and training data collection work without switching tabs)

## Critical Files

- `ModelTrainingApp.swift` -- split into Model/ files
- `Views/ContentView.swift` -- simplify to pure TabView
- `Views/CameraView.swift` -- extract to CameraViewModel
- `Views/TrainingView.swift` -- extract to ServerTrainingManager + enrich TrainingDataManager
- `Views/HandFilmsView.swift` -- Phase 2, extract to FilmPlaybackManager

# Client-Server GestureModel Implementation Plan

## Overview

Replace the planned on-device CoreML training approach with a **client-server architecture**:
- **iOS `ModelTrainingApp`**: collects labelled `HandFilm` examples, uploads them one-by-one to the
  server, runs local inference (`.tflite` backend), and downloads fresh models on demand.
- **Python FastAPI server** (`/server`): receives training examples, stores them, trains models
  (RandomForest first, LSTM later), serves the trained `.tflite` for download.

No on-device training. No `MLActivityClassifier`. No `CreateMLAdapter`. No `CoreML` training APIs.

---

## Step 0 — Revert this agent's changes

The previous agent made changes that must be reverted before implementing this plan.
Run in the repo root:

```bash
# Restore all tracked files to HEAD
git checkout HEAD -- \
  iOS/GestureModel/GestureModel/GestureModel/GestureModel.swift \
  iOS/GestureModel/GestureModel/GestureModel/MockData.swift \
  iOS/GestureModel/GestureModel/GestureModel/Types.swift \
  iOS/GestureModel/GestureModelModule.podspec \
  iOS/ModelTraining/ModelTraining/ModelTraining/ModelTrainingApp.swift \
  iOS/ModelTraining/ModelTraining/ModelTraining/Views/CameraView.swift \
  iOS/ModelTraining/ModelTraining/ModelTraining/Views/TrainingView.swift

# Delete new files created by previous agent (keep TRAINING_BACKEND_COMPARISON.md and GestureRegistry.swift)
rm iOS/GestureModel/GestureModel/GestureModel/CreateMLAdapter.swift
rm iOS/GestureModel/GestureModel/GestureModel/FeaturePreprocessor.swift
```

**Do NOT revert or delete:**
- `iOS/GestureModel/TRAINING_BACKEND_COMPARISON.md`
- `iOS/HandGestureTypes/HandGestureTypes/GestureRegistry.swift`
- Any other files already modified before this agent ran (HandGestureTypes/Types.swift,
  GestureListView.swift, ContentView.swift, SettingsView.swift, Podfile, etc.)

After revert, confirm with `git status` that only the two keeper untracked files remain (`??`).

---

## Architecture

```
┌─────────────────────────────────────┐        HTTP/JSON        ┌────────────────────────────────┐
│         iOS ModelTrainingApp        │ ──────────────────────► │     Python FastAPI Server       │
│                                     │                          │          /server                │
│  GestureRegistry (user gestures)    │  POST /examples          │                                │
│  HandsRecognizing (camera)          │  ──────────────────────► │  SQLite / flat JSON store      │
│  TrainingDataManager                │  (one HandFilm at a time)│  per-gesture example store     │
│                                     │                          │                                │
│  GestureModel (.tflite backend)     │  POST /train             │  scikit-learn RF  (phase 1)    │
│  ┌──────────────────────────────┐   │ ──────────────────────►  │  Keras LSTM       (phase 2)    │
│  │  TFLiteInterpreter           │   │                          │  → exports .tflite             │
│  │  (inference only)            │   │  GET /model/status       │                                │
│  │  model.tflite in Documents   │ ◄─│ ◄───────────────────────  │  training state + metrics      │
│  └──────────────────────────────┘   │                          │                                │
│                                     │  GET /model/download     │                                │
│  "Update Model" button              │ ◄────────────────────────│  serve latest .tflite file     │
└─────────────────────────────────────┘                          └────────────────────────────────┘
```

---

## Repository Structure Changes

```
CameraGestures/
├── iOS/                        (existing — iOS app + pods)
│   └── GestureModel/
│       └── TRAINING_BACKEND_COMPARISON.md   (keep)
└── server/                     (NEW)
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── main.py                 (FastAPI app entry point)
    ├── routers/
    │   ├── examples.py         (POST /examples)
    │   ├── training.py         (POST /train, GET /model/status)
    │   └── model.py            (GET /model/download, GET /model/info)
    ├── ml/
    │   ├── preprocessor.py     (HandFilm → feature vector, mirrors FeaturePreprocessor.swift)
    │   ├── trainer_rf.py       (scikit-learn RandomForest trainer — phase 1)
    │   └── trainer_lstm.py     (Keras LSTM trainer — phase 2, stub initially)
    ├── storage/
    │   ├── example_store.py    (persist TrainingExamples as JSON files on disk)
    │   └── model_store.py      (manage model versions, latest .tflite path)
    └── data/                   (gitignored — uploaded examples + trained models)
        ├── examples/
        └── models/
```

---

## Part 1 — Python FastAPI Server (`/server`)

### 1.1 Data Models (Pydantic)

```python
# Mirrors HandGestureTypes Swift structs exactly
class Point3D(BaseModel):
    x: float; y: float; z: float

class HandShot(BaseModel):
    landmarks: list[Point3D]   # always 21
    timestamp: float
    left_or_right: str          # "left" | "right" | "unknown"

class HandFilm(BaseModel):
    frames: list[HandShot]
    start_time: float

class TrainingExamplePayload(BaseModel):
    hand_film: HandFilm
    gesture_id: str             # slug, e.g. "thumbs_up"
    session_id: str
    user_id: str | None = None
```

### 1.2 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check — no auth required. |
| `POST` | `/auth/register` | Exchange the pre-shared `REGISTRATION_TOKEN` for a per-device JWT. No auth required to call this. |
| `POST` | `/examples` | Upload one `TrainingExamplePayload`. Returns `{ "id": "<uuid>", "total_for_gesture": N }`. |
| `GET` | `/examples/stats` | Returns per-gesture example counts and total. |
| `POST` | `/train` | Trigger training immediately (regardless of threshold). Runs as a background task. Returns `{ "job_id": "...", "status": "started" }`. |
| `GET` | `/model/status` | Returns `{ "status": "idle|training|ready|failed", "accuracy": float|null, "trained_on": N, "gesture_ids": [...], "trained_at": timestamp|null, "error": str|null }`. |
| `GET` | `/model/download` | Returns the latest `.tflite` binary file. 404 if no model trained yet. |
| `GET` | `/model/info` | Returns model metadata (accuracy, F1, confusion matrix, gesture list, training date). |

All endpoints except `/health` and `/auth/register` require `Authorization: Bearer <token>`.

### 1.3 Auto-training threshold

The server maintains a counter. After each `POST /examples`, if the total example count for **every registered gesture** reaches a configurable threshold (default: 10 examples per gesture), a background training job is triggered automatically — unless a job is already running.

Configuration via environment variable: `AUTO_TRAIN_THRESHOLD=10`.

### 1.4 Feature Preprocessor (Python)

`ml/preprocessor.py` mirrors `FeaturePreprocessor.swift` exactly:
- Input: `HandFilm` (list of `HandShot`)
- Normalize all 21 landmark xyz relative to landmark 0 (wrist)
- Compute frame-to-frame velocity (zero for first frame)
- Pad to 60 frames (zero-pad) or trim to last 60 frames
- Output: numpy array of shape `(60, 126)`

### 1.5 Phase 1 trainer — scikit-learn Random Forest

`ml/trainer_rf.py`:
- Input: all stored `TrainingExample` JSON files
- For each example: run preprocessor → compute statistical summary per-film:
  - mean, std of each of 63 normalized coord dims across 60 frames → 126 features
  - mean, std of each of 63 velocity dims across 60 frames → 126 features
  - net displacement (last_frame − first_frame) for wrist xyz → 3 features
  - dominant motion axis magnitude → 1 feature
  - Total: **256 features** per example (flat vector)
- Train `sklearn.ensemble.RandomForestClassifier(n_estimators=100)`
- Export via `coremltools` — **no, wrong format**. Export to `.tflite`:
  - Convert sklearn model to ONNX (`sklearn-onnx`) → convert to TFLite (`onnx-tf` + `tf.lite.TFLiteConverter`)
  - **Or simpler**: implement the forest manually in TFLite via a small Keras model that wraps it — not practical
  - **Correct approach**: export directly to a TFLite-compatible format using a TFLite-friendly model:
    use a Keras model with `Dense` layers trained on the same 256-feature summary vectors, export with `tf.lite.TFLiteConverter`. This is equivalent to a shallow MLP, trains in < 2 seconds, and produces a clean `.tflite`.

> **Note**: Pure sklearn RandomForest cannot be exported to `.tflite` directly. The phase 1 model
> will be a **shallow Keras MLP** (2 dense layers) trained on statistical summary features — same
> accuracy profile as RF for this problem, but with a clean TFLite export path.

### 1.6 Phase 2 trainer — Keras LSTM (stub in phase 1)

`ml/trainer_lstm.py` (initially a stub that raises `NotImplementedError`):
- Input: full 60×126 sequence per example (no summarisation)
- Architecture: `Input(60,126) → LSTM(64) → Dense(32, relu) → Dense(n_classes, softmax)`
- Export: `tf.lite.TFLiteConverter.from_keras_model()` with float16 quantisation → `gesture_model.tflite`
- Training time: ~30s on CPU for 200 examples, 10 classes

Server config variable `TRAINER=rf_mlp|lstm` selects which trainer runs. Default: `rf_mlp`.

### 1.7 Docker

```dockerfile
# server/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`docker-compose.yml` mounts `./data` as a volume so models and examples persist across restarts.

### 1.8 requirements.txt (key packages)

```
fastapi>=0.111
uvicorn[standard]>=0.29
pydantic>=2.0
numpy>=1.26
scikit-learn>=1.4
tensorflow>=2.16          # includes tf.lite
skl2onnx>=1.16            # sklearn → onnx (for phase 1 if needed)
onnx>=1.16
```

---

## Part 2 — iOS Changes

### 2.1 New file: `GestureModel/GestureModel/GestureModel/FeaturePreprocessor.swift`

Same as the file created by the previous agent. **No changes needed** — the feature preprocessing
logic is identical regardless of backend. Re-create this file.

### 2.2 New file: `GestureModel/GestureModel/GestureModel/TFLiteBackend.swift`

Implements the `.tensorFlow` backend path in `GestureModel`:

```swift
// Wraps TensorFlowLiteSwift Interpreter for inference only
class TFLiteBackend {
    private var interpreter: Interpreter?

    func load(from path: String) throws
    func predict(handfilm: HandFilm, k: Int, threshold: Float) throws -> [GesturePrediction]
    // Input: FeaturePreprocessor.featureMatrix(from:) → flat [Float32] (7560 values)
    // Output: parse outputTensor as [Float32] of length n_classes → top-k GesturePredictions
}
```

Note: TFLite `Interpreter` takes `Float32`, not `Double`. `FeaturePreprocessor` outputs `Double`
— the backend must convert.

The backend also needs to know the gesture label list (same `gesture_ids.json` sidecar used by
the CoreML path).

### 2.3 Update `GestureModel/GestureModel/GestureModel/GestureModel.swift`

- `loadTensorFlowModel(from:)` — instantiate `TFLiteBackend`, call `load(from:)`, load
  `gesture_ids.json` sidecar from same directory
- `predictWithTensorFlow(handfilm:k:)` — delegate to `TFLiteBackend.predict`
- Keep `trainWithTensorFlow` as a no-op stub (training happens on server, not on device)
- `defaultModelDirectory()` → `Documents/GestureModel/`
- `sidecarURL(forModelPath:)` → `gesture_ids.json` alongside the `.tflite`

### 2.4 Update `GestureModel/GestureModel/GestureModel/Types.swift`

Add one error case:
```swift
case modelNotDownloaded
// "No model available. Use 'Update Model' in the Training tab to download one from the server."
```

### 2.5 New file: `ModelTraining/…/Networking/GestureModelAPIClient.swift`

```swift
class GestureModelAPIClient: ObservableObject {
    var baseURL: URL      // configurable, stored in UserDefaults

    // Upload one training example (call immediately after collection)
    func uploadExample(_ example: TrainingExample) async throws -> UploadExampleResponse

    // Fetch server-side per-gesture counts
    func fetchExampleStats() async throws -> ExampleStatsResponse

    // Trigger training manually
    func triggerTraining() async throws -> TrainingJobResponse

    // Poll training status
    func fetchModelStatus() async throws -> ModelStatusResponse

    // Download latest .tflite + write to Documents/GestureModel/gesture_model.tflite
    // Also writes gesture_ids.json sidecar from status response
    func downloadModel() async throws -> URL
}
```

All request/response types are `Codable` structs mirroring the server's JSON schema.

The example payload serialises `HandFilm` using the same `HandFilmDTO` / `HandShotDTO` /
`Point3DDTO` structs (Codable wrappers — same as the previous agent introduced, kept in the app
target since `HandGestureTypes` stays non-Codable).

### 2.6 Update `ModelTraining/…/ModelTrainingApp.swift`

- Add `@StateObject private var apiClient = GestureModelAPIClient()`
- Inject via `.environmentObject(apiClient)`
- `TrainingDataManager.addTrainingExample(_:)` — after appending locally, call
  `apiClient.uploadExample(example)` in a detached `Task` (fire-and-forget with error logging)
- Add `@Published var uploadState: UploadState = .idle` to `TrainingDataManager` for UI feedback
- Keep DTO structs (`Point3DDTO` etc.) here — they are needed for the upload serialisation
- Add `TrainingState` enum: `.idle | .training | .done(ModelStatusResponse) | .failed(String)`
- `AppSettings.updateModelConfig()` — check for
  `Documents/GestureModel/gesture_model.tflite`; if found, use `.tensorFlow` backend with that
  path; otherwise `.mock`

### 2.7 Update `ModelTraining/…/Views/TrainingView.swift`

**New sections / changes:**

1. **Upload status indicator** — small inline badge per gesture card showing how many examples the
   server has received (from `apiClient.fetchExampleStats()` polled on appear).

2. **Server training controls** — replaces the old "Train Model" button:
   - `POST /train` button: "Train on Server" — calls `apiClient.triggerTraining()`
   - Training status display: polls `GET /model/status` every 3s while `status == "training"`,
     shows `ProgressView` with last-known accuracy
   - "Update Model" button: calls `apiClient.downloadModel()`, then
     `appSettings.updateModelConfig()`, then reloads `GestureModel`

3. **Server URL configuration field** — small text field (or moved to SettingsView) showing the
   current `baseURL`. Defaults to `http://localhost:8000`.

4. **Auto-train notice** — if `modelStatus.status == "ready"` and the locally loaded model is
   older than the server model, show a banner: "A new model is available. Tap 'Update Model'."

### 2.8 Update `ModelTraining/…/Views/CameraView.swift`

- Same "no model" banner as before, but message updated to:
  *"No model downloaded yet. Go to Training → Update Model."*
- `isModelTrained` checks for `gesture_model.tflite` in Documents (not `.mlmodelc`)

### 2.9 `GestureModelModule.podspec`

- Remove `CreateML` from frameworks (no longer needed)
- Keep `TensorFlowLiteSwift ~> 2.13.0` dependency (already present)
- Keep `CoreML` framework (used for `MLModel` inference only — may not be needed anymore; can be
  removed if CoreML inference path is fully dropped)

---

## Part 3 — Data Flow (end-to-end)

```
User performs gesture in ModelTrainingApp
  ↓
HandsRecognizing captures HandFilm
  ↓
TrainingView: TrainingDataManager.addTrainingExample(example)
  ↓ (local)                        ↓ (background Task)
Stored in trainingExamples[]     apiClient.uploadExample(example)
                                   ↓
                              POST /examples → server stores JSON

                         [auto-trigger when threshold reached]
                         [or user taps "Train on Server"]
                                   ↓
                              POST /train → background job starts
                              (Phase 1: MLP on stat features, < 5s)
                              (Phase 2: LSTM on full sequence, ~30s)
                                   ↓
                         GET /model/status → polling
                                   ↓
                         status: "ready" + accuracy shown in UI

                         User taps "Update Model"
                                   ↓
                         GET /model/download → writes gesture_model.tflite
                                   ↓
                         GET /model/status → writes gesture_ids.json sidecar
                                   ↓
                         appSettings.updateModelConfig() → .tensorFlow backend
                                   ↓
                         GestureModel.loadModel(from: tflitePath)
                                   ↓
                         CameraView: live inference works
```

---

## Part 4 — File Change Summary

### New files to create

| File | Description |
|------|-------------|
| `server/main.py` | FastAPI app, startup, CORS, auto-train background task |
| `server/routers/examples.py` | POST /examples, GET /examples/stats |
| `server/routers/training.py` | POST /train, GET /model/status |
| `server/routers/model.py` | GET /model/download, GET /model/info |
| `server/ml/preprocessor.py` | HandFilm → 60×126 numpy array (mirrors Swift) |
| `server/ml/trainer_rf_mlp.py` | Stat feature extraction + Keras MLP training → .tflite |
| `server/ml/trainer_lstm.py` | Full sequence Keras LSTM → .tflite (stub initially) |
| `server/storage/example_store.py` | Save/load TrainingExample JSON files |
| `server/storage/model_store.py` | Manage model versions, latest path, metadata JSON |
| `server/requirements.txt` | Python dependencies |
| `server/Dockerfile` | Container definition |
| `server/docker-compose.yml` | Local dev compose with data volume |
| `iOS/GestureModel/…/FeaturePreprocessor.swift` | Re-create (same as previous agent) |
| `iOS/GestureModel/…/TFLiteBackend.swift` | New — wraps TFLite Interpreter |
| `iOS/ModelTraining/…/Networking/GestureModelAPIClient.swift` | New — HTTP client |

### Files to modify

| File | Change |
|------|--------|
| `iOS/GestureModel/…/GestureModel.swift` | Implement `loadTensorFlowModel`, `predictWithTensorFlow`; add `defaultModelDirectory`, `sidecarURL` helpers |
| `iOS/GestureModel/…/Types.swift` | Add `modelNotDownloaded` error case |
| `iOS/GestureModel/GestureModelModule.podspec` | Remove CreateML; keep TFLiteSwift |
| `iOS/ModelTraining/…/ModelTrainingApp.swift` | Add `apiClient`, DTO structs, `TrainingState`, `uploadExample` on add, `updateModelConfig` for TFLite |
| `iOS/ModelTraining/…/Views/TrainingView.swift` | Server training controls, upload status, "Update Model" button, status polling |
| `iOS/ModelTraining/…/Views/CameraView.swift` | Update "no model" banner for TFLite path |

### Files to NOT change

- `iOS/HandGestureTypes/HandGestureTypes/Types.swift` — no Codable, stays clean
- `iOS/HandGestureTypes/HandGestureTypes/GestureRegistry.swift` — already in good shape
- `iOS/HandGestureRecognizing/…` — no changes needed
- `iOS/HandsRecognizing/…` — no changes needed

---

## Part 5 — Implementation Order for the Next Agent

1. **Revert** (Step 0 above) — git checkout + delete 2 files
2. **Server skeleton** — `main.py`, routers (empty handlers), Pydantic models, `requirements.txt`
3. **Server storage layer** — `example_store.py`, `model_store.py`
4. **Server ML** — `preprocessor.py`, `trainer_rf_mlp.py` (working), `trainer_lstm.py` (stub)
5. **Server training logic** — auto-threshold trigger, background job, status tracking
6. **Server Docker** — `Dockerfile`, `docker-compose.yml`
7. **iOS `FeaturePreprocessor.swift`** — re-create
8. **iOS `TFLiteBackend.swift`** — implement inference wrapper
9. **iOS `GestureModel.swift`** — wire TFLite backend methods
10. **iOS `GestureModelAPIClient.swift`** — HTTP client + DTO types
11. **iOS `ModelTrainingApp.swift`** — add apiClient, upload-on-add, update config for TFLite
12. **iOS `TrainingView.swift`** — server controls UI
13. **iOS `CameraView.swift`** — update banner

---

## Key Decisions (recorded)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Training location | Server (Python) | `MLActivityClassifier` macOS-only; TFLite on-device training iOS-unsupported |
| Inference on iOS | TFLite (`.tflite`) | `TensorFlowLiteSwift` already linked; no additional pods |
| Phase 1 model | Keras MLP on stat features | sklearn RF can't export to `.tflite` cleanly; MLP equivalent |
| Phase 2 model | Keras LSTM on full sequence | Best accuracy for dynamic gestures; ~30s CPU training |
| Model download | Manual "Update Model" button | User controls when to update; avoids silent model swaps |
| Auto-training | Threshold (default 10/gesture) + manual trigger | Balance between automation and control |
| Upload granularity | Per-example, immediately after collection | Server always has fresh data; no manual sync step |
| Authentication | Per-device JWT (implemented) | iOS device registers once with a pre-shared `REGISTRATION_TOKEN`; receives a long-lived JWT stored in Keychain; all endpoints except `/health` and `/auth/register` require `Authorization: Bearer <token>` |
| iOS min version | iOS 16.0 — no change | TFLite supports iOS 12+; no bump needed |
| Server stack | FastAPI + uvicorn, Dockerised | Python-native ML tools; Docker-friendly for cloud deploy |

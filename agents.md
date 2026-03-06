# CameraGestures - Dynamic Gesture Recognition System

## Project Overview

CameraGestures is a modular dynamic gesture recognition system that captures hand movements through a camera and translates them into recognizable gestures for application control. The system leverages MediaPipe Hands for real-time hand tracking and supports multiple machine learning backends for gesture classification.

The project is designed with modularity and interchangeability in mind, allowing different neural network implementations while maintaining consistent APIs across modules. This architecture enables easy experimentation with different ML approaches and seamless integration into various applications.

## Design Principles

- **Modularity**: Each module has a single, well-defined responsibility with clear interfaces
- **Interoperability**: Consistent APIs enable seamless integration between modules
- **Extensibility**: New gesture types and ML backends can be added without modifying core architecture
- **Performance**: Real-time processing optimized for <500ms latency requirement
- **Testability**: Independent module testing and validation capabilities
- **Flexibility**: Support for different deployment scenarios from development to production

## System Architecture

The system consists of five interconnected modules that work together to provide end-to-end gesture recognition:

```
Camera Input → HandsRecognizing → GestureModel → Application Output
                     ↓                  ↓
                Training Data → ModelTraining
                     ↑
              HandGestureTypes (shared data types, all modules depend on this)
```

## Module Descriptions

### 1. HandGestureTypes Module
**Purpose**: Shared data type definitions used by all other modules.

**Responsibilities**:
- Define core data structures: `Point3D`, `HandShot`, `HandFilm`, `GesturePrediction`
- Define the dynamic gesture definition type via `GestureDefinition` struct
- Manage the runtime gesture list via `GestureRegistry` (JSON-persisted)
- Define training structures: `TrainingExample`, `TrainingDataset`
- Define evaluation structures: `ModelMetrics`
- Serve as the single source of truth for inter-module data contracts

**Implementation Language**: Swift (iOS); C++ equivalent planned for cross-platform
**Platform Support**: iOS (current); Cross-platform (planned)

### 2. HandsRecognizing Module
**Purpose**: Real-time hand detection and coordinate extraction from camera input.

**Responsibilities**:
- Capture video frames from camera
- Detect and track hand landmarks using MediaPipe Hands
- Extract 21-keypoint hand skeleton coordinates
- Generate timestamped sequences of hand positions
- Output structured `HandShot` and `HandFilm` data

**Technology**: MediaPipe Hands (Google) — `MediaPipeTasksVision` on iOS
**Implementation Language**: Swift (iOS); C++ (cross-platform, planned)
**Platform Support**: iOS (current); Cross-platform (planned)

### 3. GestureModel Module
**Purpose**: Neural network abstraction layer for gesture classification.

**Responsibilities**:
- Provide unified API for different ML backends
- Accept `HandFilm` sequences as input
- Output `GesturePrediction` results with confidence scores
- Support model loading/saving operations
- Enable model switching without code changes

**Backend Options**:
- CoreML backend (iOS, planned)
- TensorFlow Lite backend (iOS, in progress — pod linked, inference stubbed)
- Mock backend (iOS, working — heuristic predictions for development/testing)
- TensorFlow/Keras backend (cross-platform, planned)
- Scikit-learn backend (cross-platform, planned)

**Implementation Language**: Swift (iOS); C++ (cross-platform, planned)
**Platform Support**: iOS (current); Cross-platform (planned)

### 4. ModelTraining Module
**Purpose**: Training pipeline for gesture recognition models.

**Responsibilities**:
- Collect training data using HandsRecognizing module
- Store handfilm datasets locally
- Train GestureModel instances on collected data
- Provide testing and validation capabilities
- Enable manual correction of predictions
- Support iterative model improvement

**Implementation Language**: Swift (iOS SwiftUI application)
**Platform Support**: iOS (current); macOS (planned)

### 5. HandGestureRecognizing Module
**Purpose**: Production-ready gesture recognition for external applications.

**Responsibilities**:
- Orchestrate HandsRecognizing and GestureModel into a single lifecycle-managed pipeline
- Process live camera input and emit `DetectedGesture` events
- Provide simplified API for application integration (initialize → start → callbacks)
- Expose performance statistics (latency, FPS, confidence, gesture counts)
- Limit access to training functionality (read-only model usage)
- Ensure consistent performance and reliability

**Implementation Language**: Swift (iOS CocoaPod); C++ (cross-platform, planned)
**Platform Support**: iOS (current); Cross-platform (planned)

---

## Platform Implementations

### iOS Implementation (`/iOS`)

The iOS implementation is a fully functional Swift realization of the architecture above, packaged as a CocoaPods workspace (`ModelTrainingApp.xcworkspace`). It serves as both the primary development environment and the reference implementation for the overall system design.

**Location**: `/iOS`
**Language**: Swift 5
**Minimum Deployment Target**: iOS 15.0
**Package Manager**: CocoaPods 1.16.2
**Linkage**: Static frameworks (`use_frameworks! :linkage => :static`)

#### Pod Structure and Dependency Graph

```
HandGestureTypes          (no dependencies)
       ↑
       ├── HandsRecognizingModule
       │       + MediaPipeTasksVision 0.10.14
       │         → MediaPipeTasksCommon 0.10.14
       │
       ├── GestureModelModule
       │       + TensorFlowLiteSwift 2.13.0
       │         → TensorFlowLiteC 2.13.0
       │
       └── HandGestureRecognizingFramework
               (depends on all three above)
                       ↑
               ModelTrainingApp  ← iOS application target
```

#### Module Implementation Status

| Module | Pod Name | Status |
|---|---|---|
| `HandGestureTypes` | `HandGestureTypes` | Complete — all types defined |
| `HandsRecognizing` | `HandsRecognizingModule` | Working — real MediaPipe integration |
| `GestureModel` | `GestureModelModule` | Partial — mock backend works; CoreML/TFLite stubbed |
| `HandGestureRecognizing` | `HandGestureRecognizingFramework` | Working — orchestration and callbacks functional |
| `ModelTraining` | App target | Partial — UI complete; persistent storage and real training stubbed |

#### Key Implementation Notes

- **MediaPipe integration is real**: `HandsRecognizing` calls `HandLandmarker.detectAsync()` with live `AVCaptureSession` frames and converts results to `HandShot` structs.
- **ML backends are currently stubbed**: `GestureModel` CoreML and TensorFlow Lite code paths exist but return mock data pending real model integration. The mock backend returns empty predictions (no heuristics).
- **Training pipeline is incomplete end-to-end**: the `ModelTraining` UI allows data collection and labeling, but persistent storage and actual model training are not yet implemented.
- **Gesture set is dynamic**: gestures are defined at runtime as `GestureDefinition` values (name + description + slug ID) managed by `GestureRegistry`. The registry persists to `<AppSupport>/gestures.json`. There is no longer a fixed compile-time enum. The ModelTrainingApp provides an "Add Gesture" sheet (accessible from the Training and Gestures tabs) to define new gestures at runtime.

### Python Server (`/server`)

The Python server is the training backend for the client-server architecture. The iOS app uploads labelled `HandFilm` examples to it, it trains a gesture recognition model, and serves the resulting `.tflite` file back to the device.

**Location**: `/server`
**Language**: Python 3.11
**Framework**: FastAPI + uvicorn
**Database**: SQLite via SQLAlchemy Core (async, `aiosqlite`)
**Deployment**: Docker / docker-compose (data volume at `./data`)
**Config**: `pydantic-settings` — all values overridable via `.env` (see `.env.example`)

#### Server File Structure

```
server/
├── main.py               — FastAPI app, CORS, DB init on startup
├── config.py             — Settings (HOST, PORT, DATA_DIR, AUTO_TRAIN_THRESHOLD, TRAINER, …)
├── database.py           — SQLAlchemy engine + table definitions (examples, models)
├── models.py             — Pydantic schemas mirroring HandGestureTypes Swift structs
├── training_state.py     — In-process training state singleton (idle/training/ready/failed)
├── routers/
│   ├── examples.py       — POST /examples, GET /examples/stats, DELETE /examples
│   ├── training.py       — POST /train, GET /model/status + auto-train threshold logic
│   └── model.py          — GET /model/download, GET /model/info, DELETE /model
├── ml/
│   ├── preprocessor.py   — HandFilm → (60, 126) numpy array + 256-feature summary vector
│   ├── trainer_rf_mlp.py — Phase 1: shallow Keras MLP on stat features → .tflite (working)
│   └── trainer_lstm.py   — Phase 2: Keras LSTM on full sequence → .tflite (stub)
├── storage/
│   ├── example_store.py  — SQLite CRUD for training examples
│   └── model_store.py    — SQLite model registry; keeps last N versions on disk
├── test_api.sh           — End-to-end smoke test script (see below)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── .gitignore            — excludes data/ and .env
```

#### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/examples` | Upload one labelled `HandFilm`; triggers auto-train check |
| `GET` | `/examples/stats` | Per-gesture example counts |
| `DELETE` | `/examples` | Wipe all examples (add `?gesture_id=slug` to wipe one gesture) |
| `POST` | `/train` | Trigger training immediately |
| `GET` | `/model/status` | Poll training state + latest accuracy |
| `GET` | `/model/download` | Download `gesture_model.tflite` |
| `GET` | `/model/info` | Model metadata (accuracy, F1, confusion matrix, gesture list) |
| `DELETE` | `/model` | Wipe all model versions and reset training state |

Interactive docs available at `http://localhost:8000/docs` when the server is running.

#### Running locally

```bash
cd server
cp .env.example .env       # adjust if needed
docker compose up --build
```

#### Smoke tests

`server/test_api.sh` is a self-contained shell script that runs a full end-to-end test suite against a running server — from `/health` through uploading examples, triggering training, downloading the model, and wiping everything at the end.

```bash
cd server
./test_api.sh                                    # localhost:8000
./test_api.sh http://192.168.1.5:8000            # custom host
./test_api.sh --verbose                          # show full JSON for every response
./test_api.sh http://my-vps.com:8000 --verbose
```

#### Key Configuration Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `DATA_DIR` | `data` | Root for SQLite DB, examples, and model files |
| `AUTO_TRAIN_THRESHOLD` | `10` | Examples per gesture before auto-training fires |
| `TRAINER` | `rf_mlp` | `rf_mlp` (MLP, working) or `lstm` (stub) |
| `MAX_MODEL_VERSIONS` | `5` | Number of `.tflite` versions to retain on disk |

---

## Glossary

### Core Concepts
- **Handshot**: A data structure containing the 21 3D coordinates of hand landmarks captured at a specific moment in time
- **Handfilm**: A time-ordered sequence of handshots with associated timestamps, representing a complete gesture motion
- **Dynamic Gesture**: A hand movement pattern that unfolds over time, requiring temporal analysis for recognition

### Module Names
- **HandGestureTypes**: The shared types module defining all data structures and contracts used across modules
- **HandsRecognizing**: The computer vision module responsible for hand detection and coordinate extraction
- **GestureModel**: The machine learning abstraction layer that classifies gestures from handfilm data
- **ModelTraining**: The training pipeline module for developing and refining gesture recognition models
- **HandGestureRecognizing**: The production module that provides gesture recognition services to external applications

### Technical Terms
- **Landmark**: Individual coordinate points (x, y, z) representing specific anatomical features of the hand
- **Keypoint**: Synonym for landmark, referring to the 21 tracked points on each hand
- **Temporal Sequence**: Time-ordered data representing how hand positions change over the duration of a gesture
- **Model Backend**: The underlying machine learning framework (CoreML, TensorFlow Lite, or mock on iOS; TensorFlow/Keras or Scikit-learn on cross-platform)
- **Confidence Score**: Numerical value indicating the model's certainty about a gesture prediction
- **Mock Backend**: A substitute for a real ML model used during development and testing; currently returns empty predictions until a real model is trained
- **GestureDefinition**: A runtime struct holding a gesture's slug ID, display name, and description — replaces the former `GestureType` enum
- **GestureRegistry**: An `ObservableObject` that manages the list of `GestureDefinition` values and persists them as JSON on disk

### Data Structures
- **Point3D**: (x, y, z) position data for a single hand landmark
- **Coordinate Triplet**: Synonym for Point3D
- **Timestamp**: Time marker associated with each handshot for temporal analysis
- **Gesture Label**: Classification identifier assigned to recognized gesture patterns (a `GestureDefinition.id` string slug)
- **Training Dataset**: Collection of labeled handfilms used for model development (`TrainingDataset` type)
- **TrainingExample**: A single labeled `HandFilm` paired with a gesture ID string (`gestureId`)
- **ModelMetrics**: Evaluation results including accuracy, precision, recall, F1-score, confusion matrix, and training time
- **DetectedGesture**: A recognized result bundled with its handfilm, handedness, timestamp, and processing latency

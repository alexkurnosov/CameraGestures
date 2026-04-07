# CameraGestures - Dynamic Gesture Recognition System

A modular C++ library for real-time hand gesture recognition using MediaPipe and machine learning.

## Overview

CameraGestures captures hand movements through a camera and translates them into recognizable gestures for application control. The system is designed for cross-platform deployment with primary support for iOS and Android.

## Architecture

- **HandsRecognizing**: Hand detection and landmark extraction using MediaPipe
- **GestureModel**: ML model abstraction supporting TensorFlow and Scikit-learn backends
- **HandGestureRecognizing**: Production-ready library with C API for platform integration
- **ModelTraining**: Swift application for training and testing gesture models

## Building the Library

### Prerequisites

- CMake 3.16 or higher
- C++17 compatible compiler
- Platform-specific tools:
  - **macOS/iOS**: Xcode 12+ 
  - **Android**: Android NDK r21+
  - **Windows**: Visual Studio 2019+

### Build Instructions

#### macOS/iOS

```bash
# Clone the repository
git clone https://github.com/yourusername/CameraGestures.git
cd CameraGestures

# Create build directory
mkdir build && cd build

# Configure for macOS
cmake .. -DCMAKE_BUILD_TYPE=Release

# Or configure for iOS
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_TOOLCHAIN_FILE=../cmake/ios.toolchain.cmake \
         -DPLATFORM=OS64

# Build
cmake --build .

# The library will be in build/lib/
```

#### Android

```bash
# Set up environment
export ANDROID_NDK_HOME=/path/to/android-ndk

# Create build directory
mkdir build-android && cd build-android

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a \
         -DANDROID_PLATFORM=android-21

# Build
cmake --build .
```

### Build Options

- `BUILD_SHARED_LIBS`: Build shared libraries (default: ON)
- `BUILD_TESTS`: Build unit tests (default: ON)
- `BUILD_EXAMPLES`: Build example applications (default: ON)
- `USE_TENSORFLOW`: Enable TensorFlow backend (default: ON)
- `USE_SKLEARN`: Enable Scikit-learn backend (default: OFF)

## Using the Library

### C++ API

```cpp
#include <CameraGestures/HandGestureRecognizing/HandGestureRecognizing.h>

using namespace CameraGestures;

// Configure recognizer
HandGestureRecognizingConfig config;
config.modelPath = "path/to/gesture_model.tflite";
config.cameraIndex = 0;
config.minGestureDurationMs = 200.0;

// Create and initialize
auto recognizer = std::make_unique<HandGestureRecognizing>();
recognizer->initialize(config);

// Set callback
recognizer->setGestureDetectedCallback([](const GesturePrediction& pred) {
    std::cout << "Detected: " << pred.gestureType 
              << " (confidence: " << pred.confidence << ")" << std::endl;
});

// Start recognition
recognizer->start();
```

### C API (for Swift/Java/Kotlin integration)

```c
#include <CameraGestures/HandGestureRecognizing/CameraGestureAPI.h>

// Create recognizer
CGHandGestureRecognizerRef recognizer = cg_create_recognizer();

// Configure
CGConfig config = {
    .cameraIndex = 0,
    .targetFPS = 30,
    .modelPath = "path/to/model.tflite",
    .minGestureDurationMs = 200.0
};

// Initialize
CGErrorCode error = cg_initialize(recognizer, &config);

// Set callback
void gesture_callback(CGGesturePrediction prediction, void* userData) {
    printf("Detected: %s (%.2f)\n", prediction.gestureType, prediction.confidence);
}
cg_set_gesture_callback(recognizer, gesture_callback, NULL);

// Start
cg_start(recognizer);

// Cleanup
cg_destroy_recognizer(recognizer);
```

### Swift Integration

```swift
// Import the C module
import CameraGesturesC

class GestureManager {
    private var recognizer: CGHandGestureRecognizerRef?
    
    init() {
        recognizer = cg_create_recognizer()
        
        var config = CGConfig()
        config.cameraIndex = 0
        config.modelPath = Bundle.main.path(forResource: "model", ofType: "tflite")
        
        let error = cg_initialize(recognizer, &config)
        guard error == CG_SUCCESS else {
            print("Failed to initialize: \(String(cString: cg_get_error_description(error)))")
            return
        }
        
        // Set callback
        cg_set_gesture_callback(recognizer, { prediction, userData in
            let gesture = String(cString: prediction.gestureType)
            print("Detected: \(gesture) (\(prediction.confidence))")
        }, nil)
    }
    
    func start() {
        cg_start(recognizer)
    }
    
    deinit {
        if let recognizer = recognizer {
            cg_destroy_recognizer(recognizer)
        }
    }
}
```

## ModelTraining Swift App

The ModelTraining app is a macOS/iOS application for collecting training data and creating gesture recognition models.

### Building the Swift App

1. Open `ModelTraining/ModelTraining.xcodeproj` in Xcode
2. Build and run the application
3. The app will link against the CameraGestures C++ library

### Features

- Live camera preview with hand tracking visualization
- Training data collection and management
- Model training via remote Python server (server-side Keras MLP / LSTM)
- Real-time testing and validation
- Export trained models for production use
- Per-device JWT authentication for the training server

## Training Server (`/server`)

The Python FastAPI server receives labelled hand gesture examples from the iOS app, trains a `.tflite` model, and serves it back for download.

### Running the server

```bash
cd server
cp .env.example .env
```

Edit `.env` and set the two required auth variables:

```
JWT_SECRET=<output of: openssl rand -hex 32>
REGISTRATION_TOKEN=<any secret string you choose>
```

Then start with Docker:

```bash
docker compose up --build
```

Interactive API docs are available at `http://localhost:8000/docs`.

### Authentication

The server uses per-device JWT authentication. Every iOS device must register once before it can use any endpoint:

1. **Server operator** sets `REGISTRATION_TOKEN` in `.env`
2. **iOS user** enters the same value in Settings → Server → Registration Token
3. On first request, the app calls `POST /auth/register` with the token, receives a JWT, and stores it in the device Keychain
4. All subsequent requests include `Authorization: Bearer <token>`

`/health` is open (no auth) for Docker health probes. Re-registering the same device is idempotent and issues a fresh token.

### API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Liveness check |
| `POST` | `/auth/register` | No | Register device, receive JWT |
| `POST` | `/examples` | Yes | Upload one labelled HandFilm |
| `GET` | `/examples/stats` | Yes | Per-gesture example counts |
| `DELETE` | `/examples` | Yes | Wipe examples |
| `POST` | `/train` | Yes | Trigger training job |
| `GET` | `/model/status` | Yes | Poll training state |
| `GET` | `/model/download` | Yes | Download `gesture_model.tflite` |
| `GET` | `/model/info` | Yes | Model metadata |
| `DELETE` | `/model` | Yes | Wipe all model versions |

## MediaPipe Integration

This project requires MediaPipe for hand tracking. The project uses MediaPipe's C++ API for real-time hand detection and landmark tracking.

### macOS Setup

1. **Automated Setup** (Recommended):
```bash
# Run the setup script to download and build MediaPipe
chmod +x scripts/setup_mediapipe_macos.sh
./scripts/setup_mediapipe_macos.sh
```

2. **Manual Setup**:
```bash
# Install dependencies
brew install bazel opencv@4 protobuf eigen

# Clone MediaPipe
cd third_party
git clone https://github.com/google/mediapipe.git
cd mediapipe

# Build hand tracking libraries
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
    //mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu
```

### iOS Setup

1. Configure for iOS in CMake:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_TOOLCHAIN_FILE=../cmake/ios.toolchain.cmake \
         -DPLATFORM=OS64 \
         -DMEDIAPIPE_ROOT_DIR=/path/to/mediapipe
```

2. MediaPipe models are embedded in the binary for iOS deployment.

### Android Setup

1. Set up Android NDK:
```bash
export ANDROID_NDK_HOME=/path/to/android-ndk
```

2. Build with MediaPipe:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a \
         -DMEDIAPIPE_ROOT_DIR=/path/to/mediapipe
```

### Model Files

MediaPipe requires these model files:
- `hand_landmark_lite.tflite` - Hand landmark detection model
- `palm_detection_lite.tflite` - Palm detection model

These are automatically downloaded by the setup script or can be found in the MediaPipe repository.

## License

[Your License Here]

## Contributing

See CONTRIBUTING.md for guidelines.

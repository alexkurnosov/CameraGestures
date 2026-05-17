# Training App v2

Forked from `iOS/ModelTraining/` in Stage 1. Source is identical to V1 — only the
Podfile is new, with pod paths adjusted for this location.

## Setup

```
cd apps/training-ios
pod install
open ModelTraining/ModelTraining.xcworkspace
```

## Pod swap plan (Stages 2–5)

Each pod below will be replaced by the unified `CameraGestures` pod as the
corresponding C++ module lands in `core/`:

| Pod | Status |
|-----|--------|
| `HandGestureTypes` | V1 (Stage 2 will replace) |
| `HandsRecognizingModule` | V1 (Stage 3 will replace) |
| `GestureModelModule` | V1 (Stage 4 will replace) |
| `HandGestureRecognizingFramework` | V1 (Stage 5 will replace) |

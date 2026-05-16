# CameraGestures — core C++ library

Cross-platform gesture-recognition library. Exposes a C ABI consumed by platform
bindings in `../bindings/`.

## Build (macOS stub — Stage 0)

```sh
./scripts/build-macos.sh
# Output: bindings/macos/CameraGestures.framework
```

## iOS / Android

See `scripts/build-ios.sh` and `scripts/build-android.sh`. Both require the
respective toolchain setup documented in `cmake/ios.toolchain.cmake` and
`cmake/android.toolchain.cmake`.

## macOS status

macOS support is a stretch goal. A go/no-go spike runs in Stage 3 to determine
whether the LiteRT-only `hand_landmarker.task` decode can match the MediaPipe
Tasks output within tolerance. If the spike fails, the macOS target is deferred
to a follow-up plan and this README will be updated.

## Third-party prebuilts

See `third_party/tflite/README.md`.

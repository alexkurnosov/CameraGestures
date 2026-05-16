# TFLite / LiteRT prebuilts — version 2.17.0

Prebuilt native libraries are vendored here to make `git clone → build` a
one-step operation (no build-from-source, no Bazel).

| Platform | Status | Location |
|---|---|---|
| iOS | ✅ vendored | `ios/TensorFlowLiteC.xcframework` |
| Android | ⬜ see README | `android/README.md` |
| macOS | ⬜ see README | `macos/README.md` |

Shared C API headers (platform-agnostic): `include/`

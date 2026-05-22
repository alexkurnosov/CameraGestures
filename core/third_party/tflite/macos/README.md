# TFLite C++ prebuilts — macOS

Version: **2.17.0** (re-platformed from iOS Simulator xcframework)

## Source

`libTensorFlowLiteC.a` is derived from the iOS Simulator slice of
`TensorFlowLiteC.xcframework` (CocoaPod version 2.17.0), which is already
downloaded by the iOS build.  The `vendor-tflite-macos.sh` script extracts
the universal (arm64 + x86_64) slice, uses `vtool` to change the Mach-O
platform tag from iOS-Simulator to macOS 13.0, then packages it as a static
archive with `libtool`.

Re-run the vendor script if the library is missing:

```sh
./core/scripts/vendor-tflite-macos.sh
```

This requires the TensorFlowLiteC 2.17.0 CocoaPod to be present in the
system's CocoaPods cache (i.e. `pod install` must have been run for either
`iOS/ModelTrainingApp` or `apps/training-ios` at least once).

Headers are shared with `../include/`.

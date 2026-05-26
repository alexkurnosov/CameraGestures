#!/usr/bin/env bash
# Builds CameraGestures.framework for macOS (arm64 + x86_64).
#
# Hand landmark detection uses Apple Vision (VNDetectHumanHandPoseRequest) in Swift —
# no TFLite landmarker code. TFLite is still required for the gesture classifier
# (GestureModel / TFLiteBackend).
#
# Prerequisites:
#   - Run core/scripts/vendor-tflite-macos.sh at least once to populate
#     core/third_party/tflite/macos/libTensorFlowLiteC.a  (gesture model backend)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CORE_DIR="$REPO_ROOT/core"
BUILD_DIR="$REPO_ROOT/build-macos"
OUT_DIR="$REPO_ROOT/bindings/macos"
FRAMEWORK="$OUT_DIR/CameraGestures.framework"

# Verify vendored TFLite library exists (needed by TFLiteBackend.cpp / gesture model)
TFLITE_LIB="$CORE_DIR/third_party/tflite/macos/libTensorFlowLiteC.a"
if [[ ! -f "$TFLITE_LIB" ]]; then
    echo "ERROR: $TFLITE_LIB not found."
    echo "Run: ./core/scripts/vendor-tflite-macos.sh"
    exit 1
fi

echo "==> Building CameraGestures (macOS, TFLite gesture model)"

build_arch() {
    local arch="$1"
    cmake -S "$CORE_DIR" -B "$BUILD_DIR/$arch" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES="$arch" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
        -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/$arch/install" \
        -DCG_ENABLE_TFLITE=ON \
        -G "Unix Makefiles"
    cmake --build "$BUILD_DIR/$arch" --config Release
}

build_arch arm64
build_arch x86_64

echo "==> Creating universal binary"
mkdir -p "$FRAMEWORK/Versions/A/Headers" "$FRAMEWORK/Versions/A/Resources"

lipo -create \
    "$BUILD_DIR/arm64/libCameraGestures.a" \
    "$BUILD_DIR/x86_64/libCameraGestures.a" \
    -output "$FRAMEWORK/Versions/A/CameraGestures"

cp -R "$CORE_DIR/include/CameraGestures/." "$FRAMEWORK/Versions/A/Headers/"

cat > "$FRAMEWORK/Versions/A/Resources/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>    <string>com.cameragestures.core</string>
    <key>CFBundleName</key>          <string>CameraGestures</string>
    <key>CFBundleVersion</key>       <string>0.1.0</string>
    <key>CFBundlePackageType</key>   <string>FMWK</string>
</dict>
</plist>
PLIST

ln -sfh A          "$FRAMEWORK/Versions/Current"
ln -sfh Versions/Current/CameraGestures "$FRAMEWORK/CameraGestures"
ln -sfh Versions/Current/Headers        "$FRAMEWORK/Headers"
ln -sfh Versions/Current/Resources      "$FRAMEWORK/Resources"

echo "==> Done: $FRAMEWORK"
file "$FRAMEWORK/Versions/A/CameraGestures"

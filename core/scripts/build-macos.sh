#!/usr/bin/env bash
# Builds a stub CameraGestures.framework for macOS (arm64 + x86_64).
# Stage 0: no real TFLite linkage yet — just proves the CMake scaffold works.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CORE_DIR="$REPO_ROOT/core"
BUILD_DIR="$REPO_ROOT/build-macos"
OUT_DIR="$REPO_ROOT/bindings/macos"
FRAMEWORK="$OUT_DIR/CameraGestures.framework"

echo "==> Building CameraGestures (macOS stub)"

build_arch() {
    local arch="$1"
    cmake -S "$CORE_DIR" -B "$BUILD_DIR/$arch" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_OSX_ARCHITECTURES="$arch" \
        -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
        -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/$arch/install" \
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

# Canonical symlinks expected by the framework bundle layout
ln -sfh A          "$FRAMEWORK/Versions/Current"
ln -sfh Versions/Current/CameraGestures "$FRAMEWORK/CameraGestures"
ln -sfh Versions/Current/Headers        "$FRAMEWORK/Headers"
ln -sfh Versions/Current/Resources      "$FRAMEWORK/Resources"

echo "==> Done: $FRAMEWORK"
file "$FRAMEWORK/Versions/A/CameraGestures"

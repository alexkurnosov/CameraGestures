#!/usr/bin/env bash
# Builds a stub CameraGestures.xcframework for iOS (device + simulator).
# Stage 0: no real TFLite linkage yet.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CORE_DIR="$REPO_ROOT/core"
BUILD_DIR="$REPO_ROOT/build-ios"
OUT_DIR="$REPO_ROOT/bindings/ios/XCFramework"
TOOLCHAIN="$CORE_DIR/cmake/ios.toolchain.cmake"

echo "==> Building CameraGestures (iOS stub)"

build_slice() {
    local name="$1"; shift
    cmake -S "$CORE_DIR" -B "$BUILD_DIR/$name" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/$name/install" \
        -G "Unix Makefiles" "$@"
    cmake --build "$BUILD_DIR/$name" --config Release
}

build_slice device  -DPLATFORM=OS64
build_slice sim     -DPLATFORM=SIMULATOR64

echo "==> Creating XCFramework"
xcodebuild -create-xcframework \
    -library "$BUILD_DIR/device/libCameraGestures.a"    -headers "$CORE_DIR/include" \
    -library "$BUILD_DIR/sim/libCameraGestures.a"       -headers "$CORE_DIR/include" \
    -output "$OUT_DIR/CameraGestures.xcframework"

echo "==> Done: $OUT_DIR/CameraGestures.xcframework"

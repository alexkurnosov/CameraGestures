#!/usr/bin/env bash
# Builds CameraGestures.xcframework for iOS (device + simulator).
# Stage 3: HandGestureTypes + HandsRecognizing C ABI (no MediaPipe in static lib;
# MediaPipe is a CocoaPod dep compiled separately by the podspec).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CORE_DIR="$REPO_ROOT/core"
BUILD_DIR="$REPO_ROOT/build-ios"
OUT_DIR="$REPO_ROOT/bindings/ios/XCFramework"
TOOLCHAIN="$CORE_DIR/cmake/ios.toolchain.cmake"
IOS_CMAKE_DIR="$CORE_DIR/cmake/ios-cmake"

# ---- Ensure ios-cmake toolchain is present --------------------------------
if [[ ! -f "$IOS_CMAKE_DIR/ios.toolchain.cmake" ]]; then
    echo "==> Cloning leetal/ios-cmake toolchain..."
    git clone --depth 1 --branch 4.4.1 \
        https://github.com/leetal/ios-cmake "$IOS_CMAKE_DIR"
fi

echo "==> Building CameraGestures (iOS)"

# Prefer Ninja if available; fall back to make.
if command -v ninja &>/dev/null; then
    GENERATOR="Ninja"
    MAKE_PROG="$(command -v ninja)"
else
    GENERATOR="Unix Makefiles"
    MAKE_PROG="/usr/bin/make"
fi

build_slice() {
    local name="$1"; shift
    cmake -S "$CORE_DIR" -B "$BUILD_DIR/$name" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/$name/install" \
        -DCMAKE_MAKE_PROGRAM="$MAKE_PROG" \
        -G "$GENERATOR" "$@"
    cmake --build "$BUILD_DIR/$name" --config Release
}

build_slice device  -DPLATFORM=OS64          -DDEPLOYMENT_TARGET=16.0
build_slice sim     -DPLATFORM=SIMULATOR64   -DDEPLOYMENT_TARGET=16.0

echo "==> Creating XCFramework"
# Pass include/CameraGestures/ directly so the XCFramework headers directory
# is flat: CameraGestures.h, Types.h, module.modulemap — no extra subdirectory.
HEADERS_DIR="$CORE_DIR/include/CameraGestures"
rm -rf "$OUT_DIR/CameraGestures.xcframework"
mkdir -p "$OUT_DIR"
xcodebuild -create-xcframework \
    -library "$BUILD_DIR/device/libCameraGestures.a"    -headers "$HEADERS_DIR" \
    -library "$BUILD_DIR/sim/libCameraGestures.a"       -headers "$HEADERS_DIR" \
    -output "$OUT_DIR/CameraGestures.xcframework"

echo "==> Done: $OUT_DIR/CameraGestures.xcframework"

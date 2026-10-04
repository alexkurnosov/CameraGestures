#!/usr/bin/env bash
# Builds CameraGestures.xcframework for iOS (device + simulator).
# MediaPipe is a CocoaPod dependency compiled separately by the podspec; it is not linked into the static lib.
#
# Usage: build-ios.sh [--capture]
#   (no argument)  the standard framework, without session capture code,
#                  into bindings/ios/XCFramework
#   --capture      the capture framework, with the session recorder and reader,
#                  into bindings/ios/XCFrameworkCapture
set -euo pipefail

CAPTURE=OFF
case "${1:-}" in
    "")        ;;
    --capture) CAPTURE=ON ;;
    *)         echo "Usage: $0 [--capture]" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CORE_DIR="$REPO_ROOT/core"
if [[ "$CAPTURE" == ON ]]; then
    BUILD_DIR="$REPO_ROOT/build-ios/capture"
    OUT_DIR="$REPO_ROOT/bindings/ios/XCFrameworkCapture"
else
    BUILD_DIR="$REPO_ROOT/build-ios/standard"
    OUT_DIR="$REPO_ROOT/bindings/ios/XCFramework"
fi
TOOLCHAIN="$CORE_DIR/cmake/ios.toolchain.cmake"
IOS_CMAKE_DIR="$CORE_DIR/cmake/ios-cmake"

# ---- Ensure ios-cmake toolchain is present --------------------------------
if [[ ! -f "$IOS_CMAKE_DIR/ios.toolchain.cmake" ]]; then
    echo "==> Cloning leetal/ios-cmake toolchain..."
    git clone --depth 1 --branch 4.4.1 \
        https://github.com/leetal/ios-cmake "$IOS_CMAKE_DIR"
fi

echo "==> Building CameraGestures (iOS, session capture $CAPTURE)"

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
        -DCG_SESSION_CAPTURE="$CAPTURE" \
        -G "$GENERATOR" "$@"
    cmake --build "$BUILD_DIR/$name" --config Release
}

build_slice device  -DPLATFORM=OS64          -DDEPLOYMENT_TARGET=16.0 -DCG_ENABLE_TFLITE=ON
build_slice sim     -DPLATFORM=SIMULATOR64   -DDEPLOYMENT_TARGET=16.0 -DCG_ENABLE_TFLITE=ON

echo "==> Creating XCFramework"
# Pass include/CameraGestures/ directly so the XCFramework headers directory
# is flat: CameraGestures.h, Types.h, module.modulemap — no extra subdirectory.
# The standard framework ships without SessionCapture.h; CameraGestures.h
# includes it only when it is there.
HEADERS_DIR="$BUILD_DIR/headers"
rm -rf "$HEADERS_DIR"
cp -R "$CORE_DIR/include/CameraGestures" "$HEADERS_DIR"
if [[ "$CAPTURE" == OFF ]]; then
    rm "$HEADERS_DIR/SessionCapture.h"
fi
rm -rf "$OUT_DIR/CameraGestures.xcframework"
mkdir -p "$OUT_DIR"
xcodebuild -create-xcframework \
    -library "$BUILD_DIR/device/libCameraGestures.a"    -headers "$HEADERS_DIR" \
    -library "$BUILD_DIR/sim/libCameraGestures.a"       -headers "$HEADERS_DIR" \
    -output "$OUT_DIR/CameraGestures.xcframework"

echo "==> Done: $OUT_DIR/CameraGestures.xcframework"

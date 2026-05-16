#!/usr/bin/env bash
# Builds a stub libcameragestures.so for Android (arm64-v8a, armeabi-v7a, x86_64).
# Stage 0: no real TFLite linkage yet.
# Requires: Android NDK r25+ with cmake support. Set ANDROID_NDK env var.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CORE_DIR="$REPO_ROOT/core"
BUILD_DIR="$REPO_ROOT/build-android"
OUT_DIR="$REPO_ROOT/bindings/android/src/main/cpp/prebuilt"
TOOLCHAIN="${ANDROID_NDK:?Set ANDROID_NDK to your NDK root}/build/cmake/android.toolchain.cmake"

build_abi() {
    local abi="$1"
    cmake -S "$CORE_DIR" -B "$BUILD_DIR/$abi" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DANDROID_ABI="$abi" \
        -DANDROID_PLATFORM=android-24 \
        -DCMAKE_BUILD_TYPE=Release \
        -G "Unix Makefiles"
    cmake --build "$BUILD_DIR/$abi" --config Release
    mkdir -p "$OUT_DIR/$abi"
    cp "$BUILD_DIR/$abi/libCameraGestures.a" "$OUT_DIR/$abi/"
}

for abi in arm64-v8a armeabi-v7a x86_64; do
    echo "==> ABI: $abi"
    build_abi "$abi"
done

echo "==> Done: $OUT_DIR"

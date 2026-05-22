#!/usr/bin/env bash
# Vendor the TFLite C static library for macOS by re-platforming the iOS
# Simulator slice from TensorFlowLiteC 2.17.0 CocoaPod.
#
# Requires:
#   - lipo, vtool, libtool  (all shipped with macOS Xcode CLT)
#   - TensorFlowLiteC 2.17.0 in the CocoaPods cache
#     (run `pod install` for apps/training-ios or iOS/ModelTrainingApp first)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$REPO_ROOT/core/third_party/tflite/macos"
OUT_LIB="$OUT_DIR/libTensorFlowLiteC.a"
TFLITE_VERSION="2.17.0"

# ---- locate the iOS Simulator xcframework in CocoaPods cache ----------------

PODS_CACHE="$HOME/Library/Caches/CocoaPods/Pods/Release/TensorFlowLiteC"
# Find the versioned subdirectory (may have a hash suffix)
SIM_FW=""
for dir in "$PODS_CACHE/$TFLITE_VERSION"* "$PODS_CACHE/$TFLITE_VERSION"; do
    candidate="$dir/Frameworks/TensorFlowLiteC.xcframework/ios-arm64_x86_64-simulator/TensorFlowLiteC.framework/TensorFlowLiteC"
    if [[ -f "$candidate" ]]; then
        SIM_FW="$candidate"
        break
    fi
done

if [[ -z "$SIM_FW" ]]; then
    echo "ERROR: TensorFlowLiteC $TFLITE_VERSION simulator framework not found in:"
    echo "  $PODS_CACHE"
    echo ""
    echo "Run 'pod install' in apps/training-ios or iOS/ModelTrainingApp first,"
    echo "then re-run this script."
    exit 1
fi

echo "==> Found: $SIM_FW"

# ---- extract and re-platform each arch slice --------------------------------

TMPDIR="$(mktemp -d)"
trap "rm -rf $TMPDIR" EXIT

replatform_slice() {
    local arch="$1"
    local tmp_obj="$TMPDIR/tflite_${arch}.o"
    local tmp_mac="$TMPDIR/tflite_macos_${arch}.o"
    lipo "$SIM_FW" -extract "$arch" -output "$tmp_obj"
    vtool -set-build-version macos 13.0 13.0 -replace -output "$tmp_mac" "$tmp_obj"
    echo "$tmp_mac"
}

echo "==> Extracting + re-platforming arm64"
arm64_obj="$(replatform_slice arm64)"
echo "==> Extracting + re-platforming x86_64"
x86_obj="$(replatform_slice x86_64)"

# ---- create universal binary and wrap in static archive --------------------

echo "==> Creating universal macOS object"
universal="$TMPDIR/TensorFlowLiteC_macos_universal.o"
lipo -create "$arm64_obj" "$x86_obj" -output "$universal"

echo "==> Creating static archive"
libtool -static "$universal" -o "$OUT_LIB"

echo "==> Done: $OUT_LIB"
file "$OUT_LIB"
ls -lh "$OUT_LIB"

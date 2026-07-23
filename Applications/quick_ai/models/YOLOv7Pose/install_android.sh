#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Push the YOLOv7ReIDtiny pose app + shared libs (+ optional weights/input) to
# an Android device and optionally run it.
#
#   ./install_android.sh [res_dir]
#
# res_dir (optional) is a local directory containing yolov7_pose.safetensors /
# yolov7_pose_q8_0.safetensors and input_320.bin; when given, its contents are
# pushed alongside the binary.

set -e

DEVICE_DIR="/data/local/tmp/nntrainer/yolov7_pose"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ABI=arm64-v8a
LIBS_DIR="$NNTRAINER_ROOT/build_android/jni/$ABI"
BIN="$SCRIPT_DIR/jni/libs/$ABI/yolov7_pose_infer"

RES_DIR="$1"

command -v adb >/dev/null || { echo "[ERROR] adb not found"; exit 1; }
[ -f "$BIN" ] || { echo "[ERROR] $BIN missing; run build_android.sh first"; exit 1; }

adb shell "mkdir -p $DEVICE_DIR"

echo "[push] shared libraries"
for so in libnntrainer.so libccapi-nntrainer.so; do
  adb push "$LIBS_DIR/$so" "$DEVICE_DIR/" >/dev/null
done
# c++_shared from the NDK
NDK_LIBCXX="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
[ -f "$NDK_LIBCXX" ] && adb push "$NDK_LIBCXX" "$DEVICE_DIR/" >/dev/null

echo "[push] executable"
adb push "$BIN" "$DEVICE_DIR/" >/dev/null
adb shell "chmod +x $DEVICE_DIR/yolov7_pose_infer"

if [ -n "$RES_DIR" ] && [ -d "$RES_DIR" ]; then
  echo "[push] resources from $RES_DIR"
  for f in yolov7_pose.safetensors yolov7_pose_q8_0.safetensors input_320.bin; do
    [ -f "$RES_DIR/$f" ] && adb push "$RES_DIR/$f" "$DEVICE_DIR/" >/dev/null
  done
fi

echo "[OK] installed to $DEVICE_DIR"
echo
echo "Run on device, e.g.:"
echo "  adb shell 'cd $DEVICE_DIR && LD_LIBRARY_PATH=. \\"
echo "    YOLO_TENSOR_TYPE=w8a32 YOLO_WEIGHTS=yolov7_pose_q8_0.safetensors \\"
echo "    ./yolov7_pose_infer . input_320.bin'"

#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the YOLOv7ReIDtiny pose+ReID inference app for Android (arm64-v8a).
#
#   1. builds libnntrainer.so / libccapi-nntrainer.so via tools/package_android.sh
#      (skipped with --cache if already present)
#   2. ndk-builds yolov7_pose_infer (custom rtmcc_gau / rms_norm layers are
#      compiled straight into the executable).
#
# Requires: ANDROID_NDK env var pointing at the NDK.

set -e

USE_CACHE=0
[ "$1" = "--cache" ] && USE_CACHE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export NNTRAINER_ROOT

if [ -z "$ANDROID_NDK" ]; then
  echo "[ERROR] ANDROID_NDK is not set."; exit 1
fi

echo "[1/2] Build nntrainer for Android"
LIB_OUT="$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so"
if [ "$USE_CACHE" -eq 1 ] && [ -f "$LIB_OUT" ]; then
  echo "  cache: reusing $LIB_OUT"
else
  cd "$NNTRAINER_ROOT"
  rm -rf builddir
  # OptimizedV3Planner packs the activation tensor pool far tighter than the
  # default V1 (measured: 39 MB -> near the 16 MB theoretical minimum on this
  # W8A8 pose graph). It only changes tensor memory placement, not values, so
  # the pose output is unchanged; remove this flag if a planner regression is
  # ever suspected (falls back to the validated V1).
  ./tools/package_android.sh -Dmemory-planner=v3
fi
[ -f "$LIB_OUT" ] || { echo "[ERROR] nntrainer android build failed"; exit 1; }

# Stage the prebuilt libs where the Android.mk expects them.
LIBS_DIR="$NNTRAINER_ROOT/build_android/jni/arm64-v8a"
mkdir -p "$LIBS_DIR"
cp -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/"*.so "$LIBS_DIR/"

echo "[2/2] ndk-build yolov7_pose_infer"
cd "$SCRIPT_DIR/jni"
rm -rf libs obj
ndk-build NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj \
  APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk \
  -j "$(nproc)"

echo "[OK] built: $SCRIPT_DIR/jni/libs/arm64-v8a/yolov7_pose_infer"

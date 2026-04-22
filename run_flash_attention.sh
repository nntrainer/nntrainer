#!/bin/bash
set -e

# Script to build and run flash attention OpenCL kernel tests
# Optimized for Qualcomm Adreno GPUs

# Activate build virtual environment
source "$(dirname "$0")/build_venv/bin/activate"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/Android/android-ndk-r28c/
export PATH=$PATH:$HOME/Android/android-ndk-r28c/
export ANDROID_NDK=$HOME/Android/android-ndk-r28c/

echo "Building NNTrainer with OpenCL support..."
./tools/package_android.sh -Denable-opencl=true

echo "Copying googletest library..."
cp -r $ANDROID_NDK/sources/third_party/googletest test/jni

echo "Installing built libraries..."
ninja install -C builddir
mkdir -p libs/arm64-v8a
cp builddir/android_build_result/lib/arm64-v8a/*.so libs/arm64-v8a

echo "Building flash attention unittest..."
ndk-build -C test/jni -j$(nproc) MESON_ENABLE_OPENCL=1 unittest_flash_attention_cl NDK_DEBUG=0

echo "Deploying to device..."
adb shell "mkdir -p /data/local/tmp/nntrainer/test"
adb push test/obj/local/arm64-v8a/unittest_flash_attention_cl /data/local/tmp/nntrainer/test
adb push libs/arm64-v8a/*.so /data/local/tmp/nntrainer/test
adb shell chmod +x /data/local/tmp/nntrainer/test/unittest_flash_attention_cl

echo "Running flash attention tests..."
adb shell "cd /data/local/tmp/nntrainer/test; export LD_LIBRARY_PATH=.; ./unittest_flash_attention_cl $@"

echo "Collecting logs..."
adb pull /data/local/tmp/nntrainer/test/logs/. ./logs/ 2>/dev/null || echo "No logs to pull"
adb shell "rm -f /data/local/tmp/nntrainer/test/logs/*" 2>/dev/null || echo "No log files to clean"

echo "Flash attention tests completed!"
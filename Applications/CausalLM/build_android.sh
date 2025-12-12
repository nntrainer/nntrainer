#!/bin/bash

# Build script for CausalLM Android application
set -e

# Check if NDK path is set
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK is not set. Please set it to your Android NDK path."
    echo "Example: export ANDROID_NDK=/path/to/android-ndk-r21d"
    exit 1
fi

# Set NNTRAINER_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export NNTRAINER_ROOT

echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK: $ANDROID_NDK"

# Step 1: Build nntrainer for Android if not already built
if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    echo "Building nntrainer for Android..."
    cd "$NNTRAINER_ROOT"
    if [ -d "$NNTRAINER_ROOT/builddir" ]; then
        rm -rf builddir
    fi
    ./tools/package_android.sh -Domp-num-threads=4 -Dthread-backend=omp -Dmmap-read=false
else
    echo "nntrainer for Android already built."
fi

# Check if build was successful
if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    echo "Error: nntrainer build failed. Please check the build logs."
    exit 1
fi

# Step 2: Build tokenizer library if not present
echo "Build Tokenizer Library If not Present"
cd "$SCRIPT_DIR"
if [ ! -f "lib/libtokenizers_android_c.a" ]; then
    echo "Warning: libtokenizers_android_c.a not found in lib directory."
    echo "Attempting to build tokenizer library..."
    if [ -f "build_tokenizer_android.sh" ]; then
        ./build_tokenizer_android.sh
    else
        echo "Error: tokenizer library not found and build script is missing."
        echo "Please build or download the tokenizer library for Android arm64-v8a"
        echo "and place it in: $SCRIPT_DIR/lib/libtokenizers_android_c.a"
        exit 1
    fi
fi
echo "Tokenizer Library Built Successfully"

# Step 3: Build CausalLM application
echo "Building CausalLM application..."
cd "$SCRIPT_DIR/jni"

# Clean previous builds
rm -rf libs obj

# Run ndk-build
ndk-build NDK_PROJECT_PATH=./ APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk -j $(nproc)

echo "Build completed successfully!"
echo "Output files are in: $SCRIPT_DIR/jni/libs/arm64-v8a/"
echo ""
echo "Executable: nntrainer_causallm"
echo "Libraries: libnntrainer.so, libccapi-nntrainer.so, libc++_shared.so"

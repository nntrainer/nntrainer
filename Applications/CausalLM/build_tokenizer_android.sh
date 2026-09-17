#!/bin/bash

# Script to build tokenizers-cpp library for Android
set -e

# Default target ABI
TARGET_ABI="${1:-arm64-v8a}"

echo "Building tokenizers-cpp library for Android $TARGET_ABI..."

# Check prerequisites
if [ -z "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK is not set. Please set it to your Android NDK path."
    exit 1
fi

# Check if cmake is installed
if ! command -v cmake &> /dev/null; then
    echo "Error: cmake is not installed. Please install cmake."
    exit 1
fi

# Check if rust is installed
if ! command -v rustc &> /dev/null || ! command -v cargo &> /dev/null; then
    echo "Error: Rust is not installed. Please install Rust from https://rustup.rs/"
    exit 1
fi

# Map Android ABI to Rust target
case "$TARGET_ABI" in
    "arm64-v8a")
        RUST_TARGET="aarch64-linux-android"
        ;;
    "armeabi-v7a")
        RUST_TARGET="armv7-linux-androideabi"
        ;;
    "x86")
        RUST_TARGET="i686-linux-android"
        ;;
    "x86_64")
        RUST_TARGET="x86_64-linux-android"
        ;;
esac

# Install Rust target if not already installed
echo "Checking Rust target: $RUST_TARGET"
if ! rustup target list --installed | grep -q "$RUST_TARGET"; then
    echo "Installing Rust target: $RUST_TARGET"
    rustup target add "$RUST_TARGET"
fi

# Validate target ABI
case "$TARGET_ABI" in
    "arm64-v8a"|"armeabi-v7a"|"x86"|"x86_64")
        echo "Target ABI: $TARGET_ABI"
        ;;
    *)
        echo "Error: Invalid target ABI: $TARGET_ABI"
        echo "Supported ABIs: arm64-v8a, armeabi-v7a, x86, x86_64"
        exit 1
        ;;
esac

# Set build directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/tokenizers-cpp-build"

# Clone tokenizers-cpp repository if not exists
if [ ! -d "$BUILD_DIR/tokenizers-cpp" ]; then
    echo "Cloning tokenizers-cpp repository..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    git clone https://github.com/mlc-ai/tokenizers-cpp.git
fi

cd "$BUILD_DIR/tokenizers-cpp"

# Update submodules
echo "Updating submodules..."
git submodule update --init --recursive

# Create build directory for specific ABI
mkdir -p "build-android-$TARGET_ABI"
cd "build-android-$TARGET_ABI"

# Set up Android toolchain variables
ANDROID_PLATFORM="android-29"
ANDROID_STL="c++_static"

# Detect platform for NDK paths
if [[ "$OSTYPE" == "darwin"* ]]; then
    NDK_HOST="darwin-x86_64"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    NDK_HOST="linux-x86_64"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    NDK_HOST="windows-x86_64"
else
    echo "Warning: Unknown platform $OSTYPE, assuming linux-x86_64"
    NDK_HOST="linux-x86_64"
fi

# Set Rust environment variables for cross-compilation
export CARGO_TARGET_DIR="$BUILD_DIR/tokenizers-cpp/build-android-$TARGET_ABI/rust"

# Additional Rust configuration for Android
export CARGO_BUILD_TARGET="$RUST_TARGET"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/aarch64-linux-android29-clang"
export CARGO_TARGET_ARMV7_LINUX_ANDROIDEABI_LINKER="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/armv7a-linux-androideabi29-clang"
export CARGO_TARGET_I686_LINUX_ANDROID_LINKER="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/i686-linux-android29-clang"
export CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/x86_64-linux-android29-clang"

export CC_aarch64_linux_android="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/aarch64-linux-android29-clang"
export CXX_aarch64_linux_android="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/aarch64-linux-android29-clang++"
export AR_aarch64_linux_android="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/llvm-ar"
export CC_armv7_linux_androideabi="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/armv7a-linux-androideabi29-clang"
export CXX_armv7_linux_androideabi="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/armv7a-linux-androideabi29-clang++"
export AR_armv7_linux_androideabi="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/llvm-ar"
export CC_i686_linux_android="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/i686-linux-android29-clang"
export CXX_i686_linux_android="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/i686-linux-android29-clang++"
export AR_i686_linux_android="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/llvm-ar"
export CC_x86_64_linux_android="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/x86_64-linux-android29-clang"
export CXX_x86_64_linux_android="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/x86_64-linux-android29-clang++"
export AR_x86_64_linux_android="$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/llvm-ar"

# Configure with CMake for Android
echo "Configuring CMake for Android $TARGET_ABI..."
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$TARGET_ABI" \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DANDROID_STL="$ANDROID_STL" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DTOKENIZERS_CPP_BUILD_TESTS=OFF \
    -DTOKENIZERS_CPP_BUILD_EXAMPLES=OFF \
    -DCMAKE_VERBOSE_MAKEFILE=ON

# Build the library
echo "Building tokenizers-cpp..."
cmake --build . -j$(nproc) --verbose

# Show what was actually built
echo "Build complete. Checking build outputs..."
echo "Contents of build directory:"
ls -la
echo ""
echo "Looking for static libraries (.a files):"
find . -name "*.a" -type f -ls
echo ""

# Find and copy the built library
echo "Searching for built libraries..."
mkdir -p "$SCRIPT_DIR/lib/$TARGET_ABI"

# Current directory is build-android-$TARGET_ABI
CURRENT_BUILD_DIR="$BUILD_DIR/tokenizers-cpp/build-android-$TARGET_ABI"

# Find all the generated libraries
echo "Looking for .a files in build directory..."
find "$CURRENT_BUILD_DIR" -name "*.a" -type f | while read -r lib; do
    echo "Found library: $lib"
done

# Collect every static library produced by the build (tokenizers-cpp,
# sentencepiece, protobuf-lite, and all the Abseil archives sentencepiece
# depends on for logging/random/strings/synchronization). Combining only a
# hand-picked subset silently drops transitive deps and produces an archive
# with undefined symbols at final link time (e.g. absl::StrCat, absl::
# SetMinLogLevel, google::protobuf::operator<<).
LIBS_TO_COMBINE=$(find "$CURRENT_BUILD_DIR" -name "*.a" -type f | grep -v "CMakeFiles" | tr '\n' ' ')

# Combine all libraries into one
if [ -n "$LIBS_TO_COMBINE" ]; then
    echo "Libraries to combine: $LIBS_TO_COMBINE"

    # Create a temporary directory for extracting object files
    TEMP_DIR="$BUILD_DIR/temp_objs"
    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"

    # Extract each library's object files into its own subdirectory. Many
    # Abseil archives contain identically-named translation units (e.g.
    # globals.cc.o, internal.cc.o) across different subdirs — extracting them
    # all into one flat directory silently overwrites earlier objects and
    # drops their symbols from the final combined archive.
    LIB_INDEX=0
    for lib in $LIBS_TO_COMBINE; do
        if [ -f "$lib" ]; then
            echo "Extracting from $lib..."
            LIB_INDEX=$((LIB_INDEX + 1))
            LIB_OBJ_DIR="$TEMP_DIR/lib_$LIB_INDEX"
            mkdir -p "$LIB_OBJ_DIR"
            (cd "$LIB_OBJ_DIR" && "$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/llvm-ar" x "$lib")
        else
            echo "Warning: Could not find $lib"
        fi
    done

    # Create the combined library
    echo "Creating combined library..."
    ALL_OBJS=$(find "$TEMP_DIR" -name "*.o" -type f)
    if [ -n "$ALL_OBJS" ]; then
        "$ANDROID_NDK/toolchains/llvm/prebuilt/$NDK_HOST/bin/llvm-ar" rcs "$SCRIPT_DIR/lib/$TARGET_ABI/libtokenizers_android_c.a" $ALL_OBJS
        echo "Combined library created successfully"
    else
        echo "Error: No object files found to combine"
        echo "Checking if any libraries were built..."

        # If no object files, maybe the libraries are header-only or built differently
        # Try to copy the first found library as-is
        first_lib=$(echo $LIBS_TO_COMBINE | awk '{print $1}')
        if [ -f "$first_lib" ]; then
            echo "Copying $first_lib as libtokenizers_android_c.a"
            cp "$first_lib" "$SCRIPT_DIR/lib/$TARGET_ABI/libtokenizers_android_c.a"
        else
            rm -rf "$TEMP_DIR"
            exit 1
        fi
    fi

    # Clean up
    rm -rf "$TEMP_DIR"
else
    echo "Error: No libraries found to combine"
    echo "Build may have failed. Check the build output above."
    exit 1
fi

# For backward compatibility, also copy to lib directory for default ABI
if [ "$TARGET_ABI" = "arm64-v8a" ] && [ -f "$SCRIPT_DIR/lib/$TARGET_ABI/libtokenizers_android_c.a" ]; then
    cp "$SCRIPT_DIR/lib/$TARGET_ABI/libtokenizers_android_c.a" "$SCRIPT_DIR/lib/libtokenizers_android_c.a"
fi

if [ -f "$SCRIPT_DIR/lib/$TARGET_ABI/libtokenizers_android_c.a" ]; then
    echo "Build completed successfully!"
    echo "Library copied to: $SCRIPT_DIR/lib/$TARGET_ABI/libtokenizers_android_c.a"
    if [ "$TARGET_ABI" = "arm64-v8a" ]; then
        echo "Also copied to: $SCRIPT_DIR/lib/libtokenizers_android_c.a (for backward compatibility)"
    fi
else
    echo "Error: Failed to build or find the tokenizers library"
    exit 1
fi

#!/bin/bash
# Run CausalLM model unit tests on a connected Android device.
# Usage: ./run_unittest_android.sh [--cache] [--filter <gtest_filter>]
#   --cache          Reuse existing nntrainer builddir and ndk-build artifacts
#   --filter <pat>   Pass --gtest_filter=<pat> to the test binary (e.g. '*Gemma4*')
#
# Prerequisites:
#   - ANDROID_NDK must be set (e.g. export ANDROID_NDK=/path/to/android-ndk-r26d)
#   - A device must be connected and visible in `adb devices`
#   - Applications/CausalLM/lib/libtokenizers_android_c.a (arm64 build)
#   - Applications/CausalLM/json.hpp
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JNI_DIR="$SCRIPT_DIR/jni"
OBJ_DIR="$JNI_DIR/obj/local/arm64-v8a"
NATIVE_BUILD="$NNTRAINER_ROOT/builddir_app/cpu"
NNTRAINER_LIB_DIR="$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a"
TEST_CACHE_MARKER="$JNI_DIR/obj/.canonical-libcausallm-test"
INSTALL_DIR="/data/local/tmp/nntr_causallm_test"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()    { echo -e "\n${YELLOW}[Step $1]${NC} $2\n${YELLOW}----------------------------------------${NC}"; }
log_header()  { echo -e "\n${CYAN}========================================\n $1\n========================================${NC}"; }

USE_CACHE=0
GTEST_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cache)  USE_CACHE=1; shift ;;
        --filter) GTEST_FILTER="$2"; shift 2 ;;
        *) echo "Usage: $0 [--cache] [--filter <gtest_filter>]"; exit 1 ;;
    esac
done

log_header "CausalLM Android Unit Tests"
log_info "NNTRAINER_ROOT: $NNTRAINER_ROOT"
log_info "INSTALL_DIR:    $INSTALL_DIR"
[[ -n "$GTEST_FILTER" ]] && log_info "Filter:         $GTEST_FILTER"

# ---------------------------------------------------------------------------
log_step "1/6" "Check prerequisites"
# ---------------------------------------------------------------------------

if [[ -z "$ANDROID_NDK" ]]; then
    log_error "ANDROID_NDK is not set."
    log_info  "Example: export ANDROID_NDK=/path/to/android-ndk-r26d"
    exit 1
fi
export PATH="$ANDROID_NDK:$PATH"

if ! command -v ndk-build &>/dev/null; then
    log_error "ndk-build not found in PATH. Check ANDROID_NDK."
    exit 1
fi

if ! adb devices | grep -q "device$"; then
    log_error "No Android device connected."
    exit 1
fi
DEVICE_ID=$(adb devices | grep "device$" | head -1 | cut -f1)
log_success "Device: $DEVICE_ID"

log_success "Prerequisites OK"

# ---------------------------------------------------------------------------
log_step "2/6" "Build canonical CausalLM native artifacts"
# ---------------------------------------------------------------------------

BUILD_ARGS=()
if [[ "$USE_CACHE" -eq 1 ]]; then
    BUILD_ARGS+=(--cache)
fi
"$SCRIPT_DIR/build_android.sh" "${BUILD_ARGS[@]}"

for artifact in \
    libcausallm.so nntr_quantize libquick_dot_ai_api.so nntr_causallm; do
    if [[ ! -f "$NATIVE_BUILD/$artifact" ]]; then
        log_error "Canonical native build is missing: $NATIVE_BUILD/$artifact"
        exit 1
    fi
done
log_success "Canonical native artifacts ready"

# ---------------------------------------------------------------------------
log_step "3/6" "Vendor googletest (one-time)"
# ---------------------------------------------------------------------------

GTEST_DIR="$JNI_DIR/googletest"
if [[ ! -d "$GTEST_DIR" ]]; then
    log_info "Copying googletest from NDK..."
    cp -r "$ANDROID_NDK/sources/third_party/googletest" "$GTEST_DIR"
    log_success "googletest vendored"
else
    log_info "googletest already present"
fi

# ---------------------------------------------------------------------------
log_step "4/6" "ndk-build"
# ---------------------------------------------------------------------------

if [[ "$USE_CACHE" -eq 1 && -f "$OBJ_DIR/unittest_causallm_models" && \
      -f "$TEST_CACHE_MARKER" ]]; then
    log_info "Reusing cached ndk-build artifacts"
else
    log_info "Building unittest_causallm_models against Meson libcausallm.so..."
    ndk-build \
        -C "$JNI_DIR" \
        NDK_PROJECT_PATH="$JNI_DIR" \
        NDK_LIBS_OUT="$JNI_DIR/libs" \
        NDK_OUT="$JNI_DIR/obj" \
        APP_BUILD_SCRIPT="$JNI_DIR/Android.mk" \
        NDK_APPLICATION_MK="$JNI_DIR/Application.mk" \
        CAUSALLM_PREBUILT_LIB="../../../builddir_app/cpu/libcausallm.so" \
        unittest_causallm_models \
        -j"$(nproc)"
    touch "$TEST_CACHE_MARKER"
    log_success "ndk-build done"
fi

[[ -f "$OBJ_DIR/unittest_causallm_models" ]] || {
    log_error "Missing artifact: $OBJ_DIR/unittest_causallm_models"
    exit 1
}

# ---------------------------------------------------------------------------
log_step "5/6" "Push to device"
# ---------------------------------------------------------------------------

FIXTURE_DIR="$NNTRAINER_ROOT/test/unittest/models/causallm_reference"
if [[ ! -d "$FIXTURE_DIR" ]]; then
    log_info "Extracting fixture tarballs..."
    tar xzf "$NNTRAINER_ROOT/packaging/causallm_reference_generation.tar.gz" \
        -C "$NNTRAINER_ROOT/test/unittest/models/"
    tar xzf "$NNTRAINER_ROOT/packaging/causallm_reference_embedding.tar.gz" \
        -C "$NNTRAINER_ROOT/test/unittest/models/"
fi

adb shell "mkdir -p $INSTALL_DIR/causallm_reference $INSTALL_DIR/tmp"
adb shell "rm -f \
    $INSTALL_DIR/libcausallm_core.so \
    $INSTALL_DIR/libcausallm.so \
    $INSTALL_DIR/nntr_quantize \
    $INSTALL_DIR/unittest_causallm_models"

log_info "Pushing binaries and libraries..."
adb push "$OBJ_DIR/unittest_causallm_models"  "$INSTALL_DIR/" 2>&1 | tail -1
adb push "$NATIVE_BUILD/nntr_quantize"        "$INSTALL_DIR/" 2>&1 | tail -1
adb push "$NATIVE_BUILD/libcausallm.so"       "$INSTALL_DIR/" 2>&1 | tail -1
adb push "$NNTRAINER_LIB_DIR/libnntrainer.so" "$INSTALL_DIR/" 2>&1 | tail -1
adb push "$NNTRAINER_LIB_DIR/libccapi-nntrainer.so" "$INSTALL_DIR/" 2>&1 | tail -1
adb push "$NNTRAINER_LIB_DIR/libc++_shared.so" \
         "$INSTALL_DIR/" 2>&1 | tail -1

log_info "Pushing fixtures..."
adb push "$FIXTURE_DIR" "$INSTALL_DIR/" 2>&1 | tail -1

adb shell "chmod 755 $INSTALL_DIR/unittest_causallm_models $INSTALL_DIR/nntr_quantize"
log_success "Push complete"

# ---------------------------------------------------------------------------
log_step "6/6" "Run tests on device"
# ---------------------------------------------------------------------------

ARGS="./unittest_causallm_models"
[[ -n "$GTEST_FILTER" ]] && ARGS="$ARGS --gtest_filter=$GTEST_FILTER"

log_info "Executing on device..."
echo ""
adb shell "cd $INSTALL_DIR && \
    LD_LIBRARY_PATH=$INSTALL_DIR \
    TMPDIR=$INSTALL_DIR/tmp \
    NNTRAINER_CAUSALLM_FIXTURE_DIR=$INSTALL_DIR/causallm_reference \
    NNTR_QUANTIZE_BIN=$INSTALL_DIR/nntr_quantize \
    $ARGS 2>&1"
EXIT=$?

echo ""
if [[ $EXIT -eq 0 ]]; then
    log_success "All tests passed."
else
    log_error "Some tests failed (exit code $EXIT)."
fi
exit $EXIT

#!/usr/bin/env bash
# Build the public CausalLM Android native artifacts and optional QuickDotAI
# app package from one canonical Meson graph.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./build_android.sh [options] [engine Meson options]

Options:
  --app                   Also assemble the QuickDotAI AAR and sample APK
  --install               Push native artifacts; with --app, install the APK
  --qnn                   Build the QNN-enabled native/app variant (default: CPU-only)
  --cache                 Reuse a compatible engine build; build on cache miss
  --clean                 Recreate the selected CausalLM build outputs
  --nntr-threads=N        Set the nntrainer compute thread count (default: 7)
  --help, -h              Show this help

Engine Meson options beginning with -D, and --arm-arch=..., are forwarded to
tools/package_android.sh.

Environment:
  ANDROID_NDK / NDK_ROOT  Android NDK root (required)
  QNN_SDK_ROOT            Qualcomm QNN SDK root (required with --qnn)
  ANDROID_SERIAL          Device selected for --install (optional if exactly
                          one authorized device is connected)
EOF
}

CLEAN=false
USE_BUILD_CACHE=false
ENABLE_QNN=false
INSTALL=false
BUILD_APP=false
NNTR_THREADS="${NNTR_THREADS:-7}"
ENGINE_ARGS=()

deprecated_option() {
    echo "Warning: $1 is deprecated; use $2." >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --app)
            BUILD_APP=true
            ;;
        --install)
            INSTALL=true
            ;;
        --qnn)
            ENABLE_QNN=true
            ;;
        --assemble-aar|--aar)
            deprecated_option "$1" "--app"
            BUILD_APP=true
            ;;
        --cache)
            USE_BUILD_CACHE=true
            ;;
        --clean)
            CLEAN=true
            ;;
        --nntr-threads=*)
            NNTR_THREADS="${1#*=}"
            ;;
        --legacy-ndk|--native-only|--skip-gradle|--skip-install)
            echo "Error: $1 was removed by the single-graph build migration." >&2
            echo "Use no option for native, --app for AAR/APK, and --install for deployment." >&2
            exit 2
            ;;
        -D*|--arm-arch=*)
            if [[ "$1" == -Denable-npu=* ]]; then
                echo "Error: -Denable-npu is controlled by --qnn." >&2
                exit 2
            fi
            ENGINE_ARGS+=("$1")
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

if ! [[ "$NNTR_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --nntr-threads must be a positive integer." >&2
    exit 2
fi
if [[ "$ENABLE_QNN" == true ]]; then
    BUILD_VARIANT="qnn"
else
    BUILD_VARIANT="cpu"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAUSALLM_ROOT="$SCRIPT_DIR"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
XGRAMMAR_ROOT="$CAUSALLM_ROOT/xgrammar"
NATIVE_BUILD="$NNTRAINER_ROOT/builddir_app/$BUILD_VARIANT"
NATIVE_MACHINE_DIR="$NNTRAINER_ROOT/builddir_app_machine/$BUILD_VARIANT"

if command -v nproc >/dev/null 2>&1; then
    BUILD_JOBS="$(nproc)"
else
    BUILD_JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 1)"
fi

ANDROID_NDK="${ANDROID_NDK:-${NDK_ROOT:-}}"
if [[ -z "$ANDROID_NDK" ]]; then
    echo "Error: ANDROID_NDK (or NDK_ROOT) is not set." >&2
    exit 1
fi
if [[ ! -d "$ANDROID_NDK/toolchains/llvm/prebuilt" ]]; then
    echo "Error: invalid Android NDK root: $ANDROID_NDK" >&2
    exit 1
fi
ANDROID_NDK="$(cd "$ANDROID_NDK" && pwd -P)"
export ANDROID_NDK
export PATH="$ANDROID_NDK:$PATH"

case "$(uname -s)" in
    Linux*)
        ANDROID_NDK_HOST="linux-x86_64"
        ANDROID_CLANG_SUFFIX=""
        ANDROID_EXE_SUFFIX=""
        ;;
    Darwin*)
        ANDROID_NDK_HOST="darwin-x86_64"
        ANDROID_CLANG_SUFFIX=""
        ANDROID_EXE_SUFFIX=""
        ;;
    CYGWIN*|MINGW*|MSYS*)
        ANDROID_NDK_HOST="windows-x86_64"
        ANDROID_CLANG_SUFFIX=".cmd"
        ANDROID_EXE_SUFFIX=".exe"
        ;;
    *)
        echo "Error: unsupported NDK host: $(uname -s)" >&2
        exit 1
        ;;
esac
if [[ ! -d "$ANDROID_NDK/toolchains/llvm/prebuilt/$ANDROID_NDK_HOST" ]]; then
    echo "Error: NDK host toolchain not found: $ANDROID_NDK_HOST" >&2
    exit 1
fi

# Meson is a native Windows process under Git Bash, so paths embedded inside
# its cross file must use the C:/... spelling rather than MSYS /c/....
ANDROID_NDK_MESON="$ANDROID_NDK"
if [[ "$ANDROID_NDK_HOST" == "windows-x86_64" ]] && command -v cygpath >/dev/null 2>&1; then
    ANDROID_NDK_MESON="$(cygpath -m "$ANDROID_NDK")"
fi

GRADLE_NDK_ARG="-PnntrainerNdkPath=$ANDROID_NDK_MESON"

QNN_SDK_ROOT_SHELL=""
if [[ "$ENABLE_QNN" == true ]]; then
    if [[ -z "${QNN_SDK_ROOT:-}" ]]; then
        echo "Error: QNN_SDK_ROOT is required with --qnn." >&2
        exit 1
    fi
    if [[ ! -f "$QNN_SDK_ROOT/include/QNN/QnnCommon.h" ]]; then
        echo "Error: invalid QNN SDK root (QnnCommon.h not found): $QNN_SDK_ROOT" >&2
        exit 1
    fi
    QNN_SDK_ROOT_SHELL="$(cd "$QNN_SDK_ROOT" && pwd -P)"
    QNN_SDK_ROOT_MESON="$QNN_SDK_ROOT_SHELL"
    if [[ "$ANDROID_NDK_HOST" == "windows-x86_64" ]] && command -v cygpath >/dev/null 2>&1; then
        QNN_SDK_ROOT_MESON="$(cygpath -m "$QNN_SDK_ROOT_SHELL")"
    fi
    export QNN_SDK_ROOT="$QNN_SDK_ROOT_MESON"
fi

echo "=== nntrainer CausalLM Android build ==="
echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK:    $ANDROID_NDK"
echo "Variant:        $BUILD_VARIANT"
echo "App:            $BUILD_APP"
echo "Install:        $INSTALL"
echo "Engine cache:   $USE_BUILD_CACHE"

# Initialize missing build dependencies.
if [[ ! -f "$XGRAMMAR_ROOT/cpp/compiled_grammar.cc" ]]; then
    echo "[0] Initializing xgrammar..."
    git -C "$NNTRAINER_ROOT" submodule update --init Applications/CausalLM/xgrammar
fi
if [[ ! -d "$XGRAMMAR_ROOT/3rdparty/dlpack/include" ]]; then
    echo "[0] Initializing xgrammar/dlpack..."
    git -C "$XGRAMMAR_ROOT" submodule update --init 3rdparty/dlpack
fi
if [[ ! -f "$NNTRAINER_ROOT/subprojects/iniparser/src/iniparser.h" ]]; then
    echo "[0] Initializing nntrainer nested submodules..."
    git -C "$NNTRAINER_ROOT" submodule update --init --recursive --depth 1
fi

restore_cached_json_header() {
    local candidate
    for candidate in \
        "$NNTRAINER_ROOT/builddir_x86/json.hpp" \
        "$NNTRAINER_ROOT/builddir/json.hpp" \
        "$NNTRAINER_ROOT/builddir/encoder/json.hpp"; do
        if [[ -f "$candidate" ]]; then
            cp "$candidate" "$CAUSALLM_ROOT/json.hpp"
            return 0
        fi
    done
    return 1
}

if [[ ! -f "$CAUSALLM_ROOT/json.hpp" ]]; then
    restore_cached_json_header || true
fi
if [[ ! -f "$CAUSALLM_ROOT/json.hpp" ]]; then
    echo "[1] Preparing json.hpp..."
    # Clear a stale encoder marker if its generated header was removed.
    rm -rf "$NNTRAINER_ROOT/builddir/encoder"
    "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir" "0.2"
    restore_cached_json_header || true
fi
if [[ ! -f "$CAUSALLM_ROOT/json.hpp" ]]; then
    echo "Error: failed to prepare $CAUSALLM_ROOT/json.hpp" >&2
    exit 1
fi

TOKENIZER="$CAUSALLM_ROOT/lib/libtokenizers_android_c.a"
if [[ ! -f "$TOKENIZER" ]]; then
    echo "[2] Building the Android tokenizer library..."
    "$CAUSALLM_ROOT/build_tokenizer_android.sh"
fi
if [[ ! -f "$TOKENIZER" ]]; then
    echo "Error: tokenizer build did not produce $TOKENIZER" >&2
    exit 1
fi

NNTRAINER_ANDROID_RESULT="$NNTRAINER_ROOT/builddir/android_build_result"
NNTRAINER_ANDROID_LIBDIR="$NNTRAINER_ANDROID_RESULT/lib/arm64-v8a"
NNTRAINER_ABI_FILE="$NNTRAINER_ANDROID_RESULT/nntrainer-abi.ini"
NNTRAINER_PREBUILT_MK="$NNTRAINER_ANDROID_RESULT/Android.mk"
NNTRAINER_MODE_FILE="$NNTRAINER_ANDROID_RESULT/causallm-build-mode.ini"

engine_mode_text() {
    # Schema 2 invalidates engines built with the former mmap-read override.
    printf 'schema=2\n'
    printf 'variant=%s\n' "$BUILD_VARIANT"
    printf 'enable_qnn=%s\n' "$ENABLE_QNN"
    printf 'nntr_threads=%s\n' "$NNTR_THREADS"
    printf 'android_ndk=%s\n' "$ANDROID_NDK"
    if [[ "$ENABLE_QNN" == true ]]; then
        printf 'qnn_sdk_root=%s\n' "$QNN_SDK_ROOT"
    fi
    local arg
    for arg in "${ENGINE_ARGS[@]}"; do
        printf 'engine_arg=%s\n' "$arg"
    done
}

engine_mode_matches() {
    [[ -f "$NNTRAINER_MODE_FILE" ]] || return 1
    [[ "$(engine_mode_text)" == "$(<"$NNTRAINER_MODE_FILE")" ]]
}

engine_prebuilt_metadata_valid() {
    [[ -f "$NNTRAINER_PREBUILT_MK" ]] || return 1
    grep -Fq 'LOCAL_SRC_FILES := lib/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so' \
        "$NNTRAINER_PREBUILT_MK" || return 1
    grep -Fq 'LOCAL_SRC_FILES := lib/$(TARGET_ARCH_ABI)/libnntrainer.so' \
        "$NNTRAINER_PREBUILT_MK" || return 1
    ! grep -Fq 'LOCAL_SRC_FILES := $(LOCAL_PATH)/lib/' \
        "$NNTRAINER_PREBUILT_MK"
}

engine_cache_valid() {
    local required=(
        "$NNTRAINER_ANDROID_LIBDIR/libnntrainer.so"
        "$NNTRAINER_ANDROID_LIBDIR/libccapi-nntrainer.so"
        "$NNTRAINER_ABI_FILE"
        "$NNTRAINER_PREBUILT_MK"
    )
    if [[ "$ENABLE_QNN" == true ]]; then
        required+=("$NNTRAINER_ANDROID_LIBDIR/libqnn_context.so")
    fi
    local file
    for file in "${required[@]}"; do
        [[ -f "$file" ]] || return 1
    done
    engine_prebuilt_metadata_valid || return 1
    engine_mode_matches
}

describe_missing_engine_cache() {
    local required=(
        "$NNTRAINER_ANDROID_LIBDIR/libnntrainer.so"
        "$NNTRAINER_ANDROID_LIBDIR/libccapi-nntrainer.so"
        "$NNTRAINER_ABI_FILE"
        "$NNTRAINER_PREBUILT_MK"
    )
    if [[ "$ENABLE_QNN" == true ]]; then
        required+=("$NNTRAINER_ANDROID_LIBDIR/libqnn_context.so")
    fi
    local file
    for file in "${required[@]}"; do
        [[ -f "$file" ]] || echo "  missing: $file" >&2
    done
    if [[ -f "$NNTRAINER_PREBUILT_MK" ]] && ! engine_prebuilt_metadata_valid; then
        echo "  incompatible: $NNTRAINER_PREBUILT_MK has stale prebuilt paths" >&2
    fi
    if [[ ! -f "$NNTRAINER_MODE_FILE" ]]; then
        echo "  missing: $NNTRAINER_MODE_FILE" >&2
    elif ! engine_mode_matches; then
        echo "  incompatible: cached engine mode does not match $BUILD_VARIANT" >&2
    fi
}

if [[ "$USE_BUILD_CACHE" == true && ${#ENGINE_ARGS[@]} -eq 0 ]] && engine_cache_valid; then
    echo "[3] Reusing a compatible engine build (--cache)."
else
    if [[ "$USE_BUILD_CACHE" == true ]]; then
        if [[ ${#ENGINE_ARGS[@]} -ne 0 ]]; then
            echo "[3] Explicit engine options bypass --cache; rebuilding."
        else
            echo "[3] Engine cache miss or incompatible metadata; rebuilding."
            describe_missing_engine_cache
        fi
    else
        echo "[3] Building the Android engine."
    fi
    (
        cd "$NNTRAINER_ROOT"
        ./tools/package_android.sh \
            -Dnntr-num-threads="$NNTR_THREADS" \
            -Denable-npu="$ENABLE_QNN" \
            "${ENGINE_ARGS[@]}"
    )
    engine_mode_text > "$NNTRAINER_MODE_FILE"
    if ! engine_cache_valid; then
        echo "Error: engine build did not produce a compatible artifact set." >&2
        describe_missing_engine_cache
        exit 1
    fi
fi

# Keep machine files outside NATIVE_BUILD so Meson can wipe it safely.
mkdir -p "$NATIVE_MACHINE_DIR"
CROSS_FILE="$NATIVE_MACHINE_DIR/android-aarch64.cross"
ABI_CROSS_FILE="$NATIVE_MACHINE_DIR/nntrainer-abi.ini"
CROSS_FILE_IN="$CAUSALLM_ROOT/app_build/android-aarch64.cross.in"
MACHINE_CHANGED=false

rendered_cross="$(mktemp)"
rendered_abi="$(mktemp)"
cleanup_temp_files() {
    rm -f "$rendered_cross" "$rendered_abi"
}
trap cleanup_temp_files EXIT

sed -e "s|@ANDROID_NDK@|$ANDROID_NDK_MESON|g" \
    -e "s|@ANDROID_NDK_HOST@|$ANDROID_NDK_HOST|g" \
    -e "s|@ANDROID_CLANG_SUFFIX@|$ANDROID_CLANG_SUFFIX|g" \
    -e "s|@ANDROID_EXE_SUFFIX@|$ANDROID_EXE_SUFFIX|g" \
    "$CROSS_FILE_IN" > "$rendered_cross"
cp "$NNTRAINER_ABI_FILE" "$rendered_abi"

if [[ ! -f "$CROSS_FILE" ]] || ! cmp -s "$rendered_cross" "$CROSS_FILE"; then
    MACHINE_CHANGED=true
    cp "$rendered_cross" "$CROSS_FILE"
fi
if [[ ! -f "$ABI_CROSS_FILE" ]] || ! cmp -s "$rendered_abi" "$ABI_CROSS_FILE"; then
    MACHINE_CHANGED=true
    cp "$rendered_abi" "$ABI_CROSS_FILE"
fi

if [[ "$CLEAN" == true || "$MACHINE_CHANGED" == true ]]; then
    if [[ -d "$NATIVE_BUILD" ]]; then
        reason="machine configuration changed"
        [[ "$CLEAN" == true ]] && reason="--clean requested"
        echo "[4] Recreating $BUILD_VARIANT native build directory ($reason)."
        rm -rf "$NATIVE_BUILD"
    fi
fi

MESON_NATIVE_OPTS=(
    -Dplatform=android
    -Denable-qnn="$ENABLE_QNN"
    -Denable-api=true
    -Denable-api-test=true
)
MESON_CROSS_OPTS=(
    --cross-file "$CROSS_FILE"
    --cross-file "$ABI_CROSS_FILE"
)

echo "[4] Configuring the canonical $BUILD_VARIANT native build."
if [[ ! -f "$NATIVE_BUILD/build.ninja" ]]; then
    meson setup "$NATIVE_BUILD" "$CAUSALLM_ROOT/app_build" \
        "${MESON_CROSS_OPTS[@]}" "${MESON_NATIVE_OPTS[@]}"
else
    meson setup "$NATIVE_BUILD" "$CAUSALLM_ROOT/app_build" --reconfigure \
        "${MESON_CROSS_OPTS[@]}" "${MESON_NATIVE_OPTS[@]}"
fi

echo "[5] Building the canonical native libraries and tools."
ninja -C "$NATIVE_BUILD" -j "$BUILD_JOBS"
NATIVE_ARTIFACT_NAMES=(
    libcausallm.so
    libquick_dot_ai_api.so
    nntr_causallm
    quick_dot_ai_test
    nntr_quantize
    nntr_safetensors_info
)
for file in "${NATIVE_ARTIFACT_NAMES[@]}"; do
    if [[ ! -f "$NATIVE_BUILD/$file" ]]; then
        echo "Error: expected native artifact missing: $NATIVE_BUILD/$file" >&2
        exit 1
    fi
done

# Remove only the retired production outputs that older Android.mk builds may
# have left behind. The test-only Android.mk harness and its artifacts remain.
LEGACY_ANDROID_MK_ARTIFACT_NAMES=(
    libcausallm_core.so
    libquick_dot_ai_api.so
    nntrainer_causallm
    quick_dot_ai_test
    nntr_quantize
    nntr_safetensors_info
)
legacy_artifact_removed=false
for legacy_output_dir in \
    "$CAUSALLM_ROOT/jni/libs/arm64-v8a" \
    "$CAUSALLM_ROOT/jni/obj/local/arm64-v8a"; do
    for file in "${LEGACY_ANDROID_MK_ARTIFACT_NAMES[@]}"; do
        if [[ -f "$legacy_output_dir/$file" ]]; then
            rm -f "$legacy_output_dir/$file"
            legacy_artifact_removed=true
        fi
    done
done
if [[ "$legacy_artifact_removed" == true ]]; then
    echo "[5] Removed retired Android.mk production artifacts."
fi

PREBUILT_DIR="$CAUSALLM_ROOT/Android/QuickDotAI/prebuilt_libs"
LIBCXX="$ANDROID_NDK/toolchains/llvm/prebuilt/$ANDROID_NDK_HOST/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
NATIVE_EXECUTABLES=(
    "$NATIVE_BUILD/nntr_causallm"
    "$NATIVE_BUILD/quick_dot_ai_test"
    "$NATIVE_BUILD/nntr_quantize"
    "$NATIVE_BUILD/nntr_safetensors_info"
)
RUNTIME_LIBRARIES=(
    "$NNTRAINER_ANDROID_LIBDIR/libnntrainer.so"
    "$NNTRAINER_ANDROID_LIBDIR/libccapi-nntrainer.so"
    "$NATIVE_BUILD/libcausallm.so"
    "$NATIVE_BUILD/libquick_dot_ai_api.so"
    "$LIBCXX"
)
QNN_REQUIRED_AARCH64_LIB_NAMES=(
    libQnnHtp.so
    libQnnHtpNetRunExtensions.so
    libQnnHtpPrepare.so
    libQnnSystem.so
)
QNN_OPTIONAL_AARCH64_LIB_NAMES=(
    libQnnHtpProfilingReader.so
    libQnnHtpOptraceProfilingReader.so
    libQnnSaver.so
    libQnnHtpV75Stub.so
    libQnnHtpV75CalculatorStub.so
    libQnnHtpV79Stub.so
    libQnnHtpV79CalculatorStub.so
    libQnnHtpV81Stub.so
    libQnnHtpV81CalculatorStub.so
)
QNN_AARCH64_LIB_NAMES=(
    "${QNN_REQUIRED_AARCH64_LIB_NAMES[@]}"
    "${QNN_OPTIONAL_AARCH64_LIB_NAMES[@]}"
)
QNN_SKEL_RELATIVE_PATHS=(
    hexagon-v75/unsigned/libQnnHtpV75Skel.so
    hexagon-v79/unsigned/libQnnHtpV79Skel.so
    hexagon-v81/unsigned/libQnnHtpV81Skel.so
)

if [[ "$ENABLE_QNN" == true ]]; then
    RUNTIME_LIBRARIES+=("$NNTRAINER_ANDROID_LIBDIR/libqnn_context.so")
    QNN_AARCH64_LIB_DIR="$QNN_SDK_ROOT_SHELL/lib/aarch64-android"
    for file in "${QNN_REQUIRED_AARCH64_LIB_NAMES[@]}"; do
        if [[ ! -f "$QNN_AARCH64_LIB_DIR/$file" ]]; then
            echo "Error: required QNN runtime library not found: $file" >&2
            exit 1
        fi
        RUNTIME_LIBRARIES+=("$QNN_AARCH64_LIB_DIR/$file")
    done
    for file in "${QNN_OPTIONAL_AARCH64_LIB_NAMES[@]}"; do
        if [[ -f "$QNN_AARCH64_LIB_DIR/$file" ]]; then
            RUNTIME_LIBRARIES+=("$QNN_AARCH64_LIB_DIR/$file")
        else
            echo "Warning: optional QNN library not found: $file" >&2
        fi
    done
    for relative_path in "${QNN_SKEL_RELATIVE_PATHS[@]}"; do
        if [[ -f "$QNN_SDK_ROOT_SHELL/lib/$relative_path" ]]; then
            RUNTIME_LIBRARIES+=("$QNN_SDK_ROOT_SHELL/lib/$relative_path")
        else
            echo "Warning: optional QNN skel not found: $relative_path" >&2
        fi
    done

    qnn_soc_pair_count=0
    for htp_version in 75 79 81; do
        qnn_stub="$QNN_AARCH64_LIB_DIR/libQnnHtpV${htp_version}Stub.so"
        qnn_skel="$QNN_SDK_ROOT_SHELL/lib/hexagon-v${htp_version}/unsigned/libQnnHtpV${htp_version}Skel.so"
        if [[ -f "$qnn_stub" && -f "$qnn_skel" ]]; then
            qnn_soc_pair_count=$((qnn_soc_pair_count + 1))
        elif [[ -f "$qnn_stub" || -f "$qnn_skel" ]]; then
            echo "Error: incomplete QNN HTP V${htp_version} Stub/Skel pair." >&2
            exit 1
        fi
    done
    if [[ "$qnn_soc_pair_count" -eq 0 ]]; then
        echo "Error: no complete supported QNN HTP Stub/Skel pair was found." >&2
        exit 1
    fi
fi

for source in "${NATIVE_EXECUTABLES[@]}" "${RUNTIME_LIBRARIES[@]}"; do
    if [[ ! -f "$source" ]]; then
        echo "Error: required native artifact missing: $source" >&2
        exit 1
    fi
done

stage_runtime_libraries() {
    echo "[6] Staging public native libraries in $PREBUILT_DIR."
    mkdir -p "$PREBUILT_DIR"
    find "$PREBUILT_DIR" -maxdepth 1 -type f -name '*.so' -delete

    local source
    for source in "${RUNTIME_LIBRARIES[@]}"; do
        cp "$source" "$PREBUILT_DIR/"
    done

    if [[ "$ENABLE_QNN" == false ]]; then
        if find "$PREBUILT_DIR" -maxdepth 1 -type f \
            \( -name 'libqnn_context.so' -o -name 'libQnn*.so' \) | grep -q .; then
            echo "Error: a CPU-only stage contains QNN libraries." >&2
            exit 1
        fi
    fi
    if [[ -f "$PREBUILT_DIR/libquick_dot_ai.so" ]]; then
        echo "Error: proprietary model overlay leaked into the public stage." >&2
        exit 1
    fi
}

AAR=""
APK=""
if [[ "$BUILD_APP" == true ]]; then
    stage_runtime_libraries
    echo "[7] Assembling the QuickDotAI AAR and sample APK."
    (
        cd "$CAUSALLM_ROOT/Android"
        ./gradlew "$GRADLE_NDK_ARG" \
            :QuickDotAI:assembleDebug :SampleTestAPP:assembleDebug
    )
    AAR="$CAUSALLM_ROOT/Android/QuickDotAI/build/outputs/aar/QuickDotAI-debug.aar"
    APK="$CAUSALLM_ROOT/Android/SampleTestAPP/build/outputs/apk/debug/SampleTestAPP-debug.apk"
    for file in "$AAR" "$APK"; do
        if [[ ! -f "$file" ]]; then
            echo "Error: Gradle artifact missing: $file" >&2
            exit 1
        fi
    done
fi

if [[ "$INSTALL" == false ]]; then
    if [[ "$BUILD_APP" == true ]]; then
        echo "=== Done (native + AAR/APK; no device modified) ==="
        echo "AAR: $AAR"
        echo "APK: $APK"
    else
        echo "=== Done (native $BUILD_VARIANT build) ==="
        echo "Artifacts: $NATIVE_BUILD"
    fi
    exit 0
fi

if ! command -v adb >/dev/null 2>&1; then
    echo "Error: adb is required with --install." >&2
    exit 1
fi
if [[ -z "${ANDROID_SERIAL:-}" ]]; then
    connected_devices="$(adb devices | awk '$2 == "device" { print $1 }')"
    device_count="$(printf '%s\n' "$connected_devices" | awk 'NF { count++ } END { print count + 0 }')"
    if [[ "$device_count" -ne 1 ]]; then
        echo "Error: --install needs ANDROID_SERIAL or exactly one authorized device." >&2
        exit 1
    fi
    export ANDROID_SERIAL="$connected_devices"
fi
ADB=(adb -s "$ANDROID_SERIAL")
if [[ "$("${ADB[@]}" get-state 2>/dev/null)" != "device" ]]; then
    echo "Error: Android device is not ready: $ANDROID_SERIAL" >&2
    exit 1
fi

if [[ "$BUILD_APP" == true ]]; then
    echo "[8] Installing the already-built sample APK."
    "${ADB[@]}" install -r "$APK"
fi

DEVICE_DIR="/data/local/tmp/Quick.AI"
"${ADB[@]}" shell "mkdir -p $DEVICE_DIR"

DEVICE_MANAGED_NAMES=(
    "${NATIVE_ARTIFACT_NAMES[@]}"
    libcausallm_core.so
    nntrainer_causallm
    libnntrainer.so
    libccapi-nntrainer.so
    libc++_shared.so
    libqnn_context.so
    run_test.sh
    run_causallm.sh
)
DEVICE_MANAGED_NAMES+=("${QNN_AARCH64_LIB_NAMES[@]}")
for relative_path in "${QNN_SKEL_RELATIVE_PATHS[@]}"; do
    DEVICE_MANAGED_NAMES+=("${relative_path##*/}")
done
DEVICE_MANAGED_PATHS=()
for file in "${DEVICE_MANAGED_NAMES[@]}"; do
    DEVICE_MANAGED_PATHS+=("$DEVICE_DIR/$file")
done
"${ADB[@]}" shell rm -f "${DEVICE_MANAGED_PATHS[@]}"

echo "[8] Pushing canonical native artifacts."
for source in "${NATIVE_EXECUTABLES[@]}" "${RUNTIME_LIBRARIES[@]}"; do
    "${ADB[@]}" push "$source" "$DEVICE_DIR/"
done

"${ADB[@]}" shell "cat > $DEVICE_DIR/run_test.sh << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=$DEVICE_DIR:\$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$DEVICE_DIR:\$ADSP_LIBRARY_PATH
export NNTR_NUM_THREADS=$NNTR_THREADS
cd $DEVICE_DIR
./quick_dot_ai_test \"\$@\"
EOF
chmod 755 $DEVICE_DIR/run_test.sh"

"${ADB[@]}" shell "cat > $DEVICE_DIR/run_causallm.sh << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=$DEVICE_DIR:\$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$DEVICE_DIR:\$ADSP_LIBRARY_PATH
export NNTR_NUM_THREADS=$NNTR_THREADS
cd $DEVICE_DIR
./nntr_causallm \"\$@\"
EOF
chmod 755 $DEVICE_DIR/run_causallm.sh"

"${ADB[@]}" shell chmod 755 \
    "$DEVICE_DIR/nntr_causallm" \
    "$DEVICE_DIR/quick_dot_ai_test" \
    "$DEVICE_DIR/nntr_quantize" \
    "$DEVICE_DIR/nntr_safetensors_info"

if [[ "$BUILD_APP" == true ]]; then
    echo "=== Done (native artifacts pushed and APK installed on $ANDROID_SERIAL) ==="
else
    echo "=== Done (native artifacts pushed to $ANDROID_SERIAL) ==="
fi
echo "Device artifacts: $DEVICE_DIR"

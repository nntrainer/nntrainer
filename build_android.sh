#!/bin/bash
# Android build script for the CausalLM application (this nntrainer repo).
#
# Unlike the root build.sh (which targets the meson/quick_dot_ai superproject
# layout), this script drives THIS repo's real Android flow via ndk-build:
#   [1] nntrainer Android library   -> tools/package_android.sh
#   [2] tokenizer static library    -> Applications/CausalLM/build_tokenizer_android.sh
#   [3] json.hpp encoder header     -> jni/prepare_encoder.sh
#   [4] CausalLM ndk-build targets  -> Applications/CausalLM/jni
#
# Usage:
#   ./build_android.sh                          # build core targets (default)
#   ./build_android.sh --target=core,api,test   # pick targets
#   ./build_android.sh --target=all             # core + api + test
#   ./build_android.sh --clean                  # force-rebuild nntrainer lib too
#   ./build_android.sh --cache                  # reuse nntrainer builddir if present
#
# Targets:
#   core   libcausallm_core.so, nntrainer_causallm, nntr_quantize,
#          nntr_safetensors_info   (default; always built)
#   api    libcausallm_api.so      (requires core)
#   test   test_api                (requires core + api)
#
# Environment:
#   ANDROID_NDK  path to Android NDK. Auto-detected if unset.
set -e

cd /home/jwon/Desktop/workspace/release/Quick.AI/nntrainer

# ── NDK detection ───────────────────────────────────────────────────────
# Honor an existing ANDROID_NDK; otherwise fall back to the workspace default.
if [ -z "$ANDROID_NDK" ]; then
    for candidate in \
        "$HOME/Desktop/workspace/android-ndk-r26d" \
        /opt/android-ndk-r26d; do
        if [ -d "$candidate" ]; then
            export ANDROID_NDK="$candidate"
            break
        fi
    done
fi

if [ -z "$ANDROID_NDK" ] || [ ! -d "$ANDROID_NDK" ]; then
    echo "Error: ANDROID_NDK is not set (or does not exist)."
    echo "Example: export ANDROID_NDK=/path/to/android-ndk-r26d"
    exit 1
fi
export PATH="$PATH:$ANDROID_NDK"

# ── Parse arguments ─────────────────────────────────────────────────────
CLEAN=false
USE_CACHE=false
TARGETS="core"

for arg in "$@"; do
    case "$arg" in
        --target=*) TARGETS="${arg#*=}" ;;
        --clean)    CLEAN=true ;;
        --cache)    USE_CACHE=true ;;
        --help|-h)
            sed -n '2,/^set -e$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $arg (use --help)"
            exit 1 ;;
    esac
done

# Resolve target list -> ndk-build module set.
ENABLE_CORE=true   # always built (other targets link against it)
ENABLE_API=false
ENABLE_TEST=false

if [ "$TARGETS" = "all" ]; then
    ENABLE_API=true
    ENABLE_TEST=true
else
    IFS=',' read -ra T <<< "$TARGETS"
    for t in "${T[@]}"; do
        case "$(echo "$t" | tr -d ' ')" in
            core) ;;                       # always on
            api)  ENABLE_API=true ;;
            test) ENABLE_API=true; ENABLE_TEST=true ;;   # test needs the api lib
            *)    echo "Unknown target: $t (core|api|test|all)"; exit 1 ;;
        esac
    done
fi

# ── Paths & logging ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$SCRIPT_DIR"
CAUSALLM_ROOT="$NNTRAINER_ROOT/Applications/CausalLM"
JNI_DIR="$CAUSALLM_ROOT/jni"
NNTRAINER_ANDROID_LIB="$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log_step()    { echo -e "\n${YELLOW}[$1]${NC} $2"; }
log_ok()      { echo -e "  ${GREEN}[OK]${NC} $1"; }
log_err()     { echo -e "${RED}[ERROR]${NC} $1"; }

echo -e "${CYAN}=== CausalLM Android Build ===${NC}"
echo "NNTRAINER_ROOT: $NNTRAINER_ROOT"
echo "ANDROID_NDK:    $ANDROID_NDK"
echo "TARGETS:        core$([ "$ENABLE_API" = true ] && echo ,api)$([ "$ENABLE_TEST" = true ] && echo ,test)"
echo "CLEAN:          $CLEAN"

# ── Step 1: nntrainer Android library ───────────────────────────────────
log_step "1/4" "Build nntrainer for Android"
if [ "$USE_CACHE" = true ] && [ "$CLEAN" = false ] && [ -f "$NNTRAINER_ANDROID_LIB" ]; then
    log_ok "Reusing existing nntrainer builddir (--cache)"
else
    cd "$NNTRAINER_ROOT"
    if [ "$CLEAN" = true ] && [ -d "$NNTRAINER_ROOT/builddir" ]; then
        echo "  Removing existing builddir (--clean)..."
        rm -rf "$NNTRAINER_ROOT/builddir"
    fi
    ./tools/package_android.sh
    cd "$SCRIPT_DIR"
fi
if [ ! -f "$NNTRAINER_ANDROID_LIB" ]; then
    log_err "nntrainer Android build failed (missing $NNTRAINER_ANDROID_LIB)"
    exit 1
fi
log_ok "nntrainer ready"

# ── Step 2: tokenizer static library ────────────────────────────────────
log_step "2/4" "Tokenizer library"
TOKENIZER="$CAUSALLM_ROOT/lib/libtokenizers_android_c.a"
if [ ! -f "$TOKENIZER" ]; then
    echo "  libtokenizers_android_c.a not found, building..."
    if [ -f "$CAUSALLM_ROOT/build_tokenizer_android.sh" ]; then
        cd "$CAUSALLM_ROOT" && ./build_tokenizer_android.sh && cd "$SCRIPT_DIR"
    else
        log_err "Tokenizer library missing and no build script found."
        log_err "Place it at: $TOKENIZER"
        exit 1
    fi
fi
log_ok "Tokenizer library ready"

# ── Step 3: json.hpp encoder header ─────────────────────────────────────
log_step "3/4" "Prepare json.hpp"
if [ ! -f "$CAUSALLM_ROOT/json.hpp" ]; then
    echo "  json.hpp not found, fetching via prepare_encoder.sh..."
    "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir" "0.2" || true
    if [ ! -f "$CAUSALLM_ROOT/json.hpp" ]; then
        log_err "Failed to prepare json.hpp"
        exit 1
    fi
fi
log_ok "json.hpp ready"

# ── Step 4: ndk-build CausalLM targets ──────────────────────────────────
log_step "4/4" "Build CausalLM (ndk-build)"
export NNTRAINER_ROOT
cd "$JNI_DIR"

# A clean build wipes prior ndk-build outputs so stale objects never leak in.
if [ "$CLEAN" = true ]; then
    rm -rf libs obj
fi

MODULES="causallm_core nntrainer_causallm nntr_quantize nntr_safetensors_info"
[ "$ENABLE_API" = true ]  && MODULES="$MODULES causallm_api"
[ "$ENABLE_TEST" = true ] && MODULES="$MODULES test_api"

echo "  Modules: $MODULES"
ndk-build \
    NDK_PROJECT_PATH=. \
    NDK_LIBS_OUT=./libs \
    NDK_OUT=./obj \
    APP_BUILD_SCRIPT=./Android.mk \
    NDK_APPLICATION_MK=./Application.mk \
    $MODULES -j "$(nproc)"
cd "$SCRIPT_DIR"

# ── Verify & report ─────────────────────────────────────────────────────
LIBS_DIR="$JNI_DIR/libs/arm64-v8a"
OBJ_DIR="$JNI_DIR/obj/local/arm64-v8a"
NNTR_PREBUILT_DIR="$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a"

# libnntrainer.so / libccapi-nntrainer.so are PREBUILT_SHARED_LIBRARY deps in
# Android.mk. On an incremental build where the local module sources are
# unchanged, ndk-build refreshes obj/ but does NOT re-install the prebuilt into
# libs/, so a stale copy lingers there and gets deployed — even though
# package_android.sh just regenerated a fresh one. Always overwrite libs/ with
# the authoritative android_build_result output so install pushes this build.
mkdir -p "$LIBS_DIR"
for plib in libnntrainer.so libccapi-nntrainer.so; do
    if [ -f "$NNTR_PREBUILT_DIR/$plib" ]; then
        cp -p "$NNTR_PREBUILT_DIR/$plib" "$LIBS_DIR/$plib"
        log_ok "staged fresh $plib from android_build_result"
    fi
done

# ndk-build relinks executables/shared-libs under obj/ but does NOT refresh
# libs/. A stale libs/ copy from a previous build would otherwise be deployed by
# install_android.sh, so always restage from the authoritative obj/ output.
ensure_artifact() {
    local name=$1
    if [ -f "$OBJ_DIR/$name" ]; then
        mkdir -p "$LIBS_DIR"
        cp "$OBJ_DIR/$name" "$LIBS_DIR/"
        [ -x "$OBJ_DIR/$name" ] && chmod +x "$LIBS_DIR/$name"
        log_ok "$name ($(du -h "$LIBS_DIR/$name" | cut -f1)) (from obj)"
    elif [ -f "$LIBS_DIR/$name" ]; then
        log_ok "$name ($(du -h "$LIBS_DIR/$name" | cut -f1))"
    else
        log_err "$name not found in libs/ or obj/"
        exit 1
    fi
}

echo -e "\n${CYAN}=== Build artifacts ===${NC}"
ensure_artifact "libcausallm_core.so"
ensure_artifact "nntrainer_causallm"
ensure_artifact "nntr_quantize"
ensure_artifact "nntr_safetensors_info"
[ "$ENABLE_API" = true ]  && ensure_artifact "libcausallm_api.so"
[ "$ENABLE_TEST" = true ] && ensure_artifact "test_api"

echo -e "\n${GREEN}=== Build completed ===${NC}"
echo "Artifacts in: $LIBS_DIR"
echo ""
echo "Install to a connected device:"
echo "  ./install_android.sh"

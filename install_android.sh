#!/bin/bash
# Install the CausalLM Android build to a connected device via adb.
#
# Pushes the ndk-build artifacts from Applications/CausalLM/jni/libs/arm64-v8a
# (built by ./build_android.sh) to the device and writes run wrappers.
#
# Usage:
#   ./install_android.sh                     # push binaries + libs + run scripts
#   ./install_android.sh --model=qwen3       # also push res/qwen3 to the device
#   ./install_android.sh --model=all         # push every model under res/
#
# Environment:
#   ANDROID_NDK  used only as a fallback source for libc++_shared.so.
set -e

INSTALL_DIR="/data/local/tmp/nntrainer/causallm"
MODEL_DIR="$INSTALL_DIR/models"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo ${SCRIPT_DIR}
CAUSALLM_ROOT="$SCRIPT_DIR/Applications/CausalLM"
LIBS_DIR="$CAUSALLM_ROOT/jni/libs/arm64-v8a"
OBJ_DIR="$CAUSALLM_ROOT/jni/obj/local/arm64-v8a"
RES_DIR="$CAUSALLM_ROOT/res"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log_step() { echo -e "\n${YELLOW}[$1]${NC} $2"; }
log_ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $1"; }

# ── Parse arguments ─────────────────────────────────────────────────────
MODEL=""
for arg in "$@"; do
    case "$arg" in
        --model=*) MODEL="${arg#*=}" ;;
        --help|-h)
            sed -n '2,/^set -e$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $arg (use --help)"; exit 1 ;;
    esac
done

echo -e "${CYAN}=== Install CausalLM to Android device ===${NC}"
echo "INSTALL_DIR: $INSTALL_DIR"

# ── Step 1: device check ────────────────────────────────────────────────
log_step "1/4" "Check device connection"
if ! adb devices | grep -q "device$"; then
    log_err "No Android device connected."
    exit 1
fi
DEVICE_ID=$(adb devices | grep "device$" | head -1 | cut -f1)
log_ok "Device connected: $DEVICE_ID"

# ── Step 2: locate artifacts ────────────────────────────────────────────
log_step "2/4" "Check build artifacts"
if [ ! -d "$LIBS_DIR" ]; then
    log_err "Build output not found: $LIBS_DIR"
    log_err "Run ./build_android.sh first."
    exit 1
fi

# Required executables and libraries produced by the core build.
REQUIRED=(
    nntrainer_causallm
    nntr_quantize
    nntr_safetensors_info
    libcausallm_core.so
    libnntrainer.so
    libccapi-nntrainer.so
)
# Optional artifacts (present only if built with --target=api/test).
OPTIONAL=(
    libcausallm_api.so
    test_api
    libc++_shared.so
)

# Resolve a file from libs/, falling back to obj/ (ndk-build sometimes leaves
# executables and the c++ runtime only under obj/local/arm64-v8a).
resolve() {
    local name=$1
    # Prefer obj/ when it is newer than libs/: ndk-build relinks into obj/ but
    # leaves libs/ stale, so a plain "libs/ exists -> use it" would deploy an
    # outdated binary/lib.
    if [ -f "$OBJ_DIR/$name" ] && \
       { [ ! -f "$LIBS_DIR/$name" ] || [ "$OBJ_DIR/$name" -nt "$LIBS_DIR/$name" ]; }; then
        mkdir -p "$LIBS_DIR"
        cp "$OBJ_DIR/$name" "$LIBS_DIR/"
        return 0
    fi
    if [ -f "$LIBS_DIR/$name" ]; then return 0; fi
    if [ -f "$OBJ_DIR/$name" ]; then
        cp "$OBJ_DIR/$name" "$LIBS_DIR/"
        return 0
    fi
    # libc++_shared.so can be pulled straight from the NDK.
    if [ "$name" = "libc++_shared.so" ] && [ -n "$ANDROID_NDK" ]; then
        local found
        found=$(find "$ANDROID_NDK" -name libc++_shared.so 2>/dev/null | grep aarch64 | head -1)
        if [ -n "$found" ]; then cp "$found" "$LIBS_DIR/"; return 0; fi
    fi
    return 1
}

PUSH_LIST=()
for f in "${REQUIRED[@]}"; do
    if resolve "$f"; then
        log_ok "$f ($(du -h "$LIBS_DIR/$f" | cut -f1))"
        PUSH_LIST+=("$f")
    else
        log_err "Missing required artifact: $f  (run ./build_android.sh)"
        exit 1
    fi
done
for f in "${OPTIONAL[@]}"; do
    if resolve "$f"; then
        log_ok "$f ($(du -h "$LIBS_DIR/$f" | cut -f1)) (optional)"
        PUSH_LIST+=("$f")
    fi
done

# ── Step 3: push artifacts + run scripts ────────────────────────────────
log_step "3/4" "Push to device"
adb shell "mkdir -p $INSTALL_DIR $MODEL_DIR"

# Executables (no extension / not a shared lib) get +x on the device.
is_executable() { [[ "$1" != *.so && "$1" != *.a ]]; }

for f in "${PUSH_LIST[@]}"; do
    adb push "$LIBS_DIR/$f" "$INSTALL_DIR/" >/dev/null
    if is_executable "$f"; then
        adb shell "chmod 755 $INSTALL_DIR/$f"
    fi
    log_ok "pushed $f"
done

write_run_script() {
    local name=$1 binary=$2
    adb shell "cat > $INSTALL_DIR/$name << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=$INSTALL_DIR:\$LD_LIBRARY_PATH
export NNTR_NUM_THREADS=4
cd $INSTALL_DIR
./$binary \$@
EOF"
    adb shell "chmod 755 $INSTALL_DIR/$name"
}

write_run_script run_causallm.sh        nntrainer_causallm
write_run_script run_quantize.sh        nntr_quantize
write_run_script run_safetensors_info.sh nntr_safetensors_info
[ -f "$LIBS_DIR/test_api" ] && write_run_script run_test_api.sh test_api
log_ok "run scripts created"

# ── Step 4: optional model push ─────────────────────────────────────────
log_step "4/4" "Model files"
if [ -z "$MODEL" ]; then
    echo "  (skipped) push models with --model=<name> or --model=all"
elif [ "$MODEL" = "all" ]; then
    for d in "$RES_DIR"/*/; do
        m=$(basename "$d")
        adb shell "mkdir -p $MODEL_DIR/$m"
        adb push "$d." "$MODEL_DIR/$m/" >/dev/null
        log_ok "pushed model: $m"
    done
elif [ -d "$RES_DIR/$MODEL" ]; then
    adb shell "mkdir -p $MODEL_DIR/$MODEL"
    adb push "$RES_DIR/$MODEL/." "$MODEL_DIR/$MODEL/" >/dev/null
    log_ok "pushed model: $MODEL"
else
    log_err "Model not found: $RES_DIR/$MODEL"
    echo "  Available: $(ls "$RES_DIR" 2>/dev/null | tr '\n' ' ')"
    exit 1
fi

# ── Summary ─────────────────────────────────────────────────────────────
echo -e "\n${CYAN}=== Installation complete ===${NC}"
echo "Device:      $DEVICE_ID"
echo "Install dir: $INSTALL_DIR"
echo ""
echo "Run on device (after pushing a model, e.g. --model=qwen3):"
echo "  adb shell $INSTALL_DIR/run_causallm.sh $MODEL_DIR/qwen3"
echo ""
echo "Quantize:"
echo "  adb shell $INSTALL_DIR/run_quantize.sh $MODEL_DIR/qwen3 --fc_dtype Q4_0"
if [ -f "$LIBS_DIR/test_api" ]; then
    echo ""
    echo "API test:"
    echo "  adb shell $INSTALL_DIR/run_test_api.sh [ARGS]"
fi

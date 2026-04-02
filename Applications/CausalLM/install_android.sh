#!/bin/bash

# Installation script for CausalLM Android application
set -e

# Configuration
INSTALL_DIR="/data/local/tmp/nntrainer/causallm"
MODEL_DIR="$INSTALL_DIR/models"

# Set SCRIPT_DIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN} $1 ${NC}"
    echo -e "${CYAN}========================================${NC}"
}

log_step() {
    echo -e "\n${YELLOW}[Step $1]${NC} $2"
    echo -e "${YELLOW}----------------------------------------${NC}"
}

log_header "Install CausalLM to Android Device"
log_info "INSTALL_DIR: $INSTALL_DIR"
log_info "SCRIPT_DIR: $SCRIPT_DIR"

# Check if device is connected
log_step "1/3" "Check device connection"
if ! adb devices | grep -q "device$"; then
    log_error "No Android device connected. Please connect a device and try again."
    exit 1
fi

DEVICE_ID=$(adb devices | grep "device$" | head -1 | cut -f1)
log_success "Device connected: $DEVICE_ID"

# Check if all required files exist
log_step "2/3" "Check build artifacts"
REQUIRED_FILES=(
    "$SCRIPT_DIR/jni/libs/arm64-v8a/nntrainer_causallm"
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_core.so"
    "$SCRIPT_DIR/jni/libs/arm64-v8a/nntr_quantize"
)

# Optional dependency files (might not be in libs/arm64-v8a depending on build)
DEP_FILES=(
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libnntrainer.so"
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libccapi-nntrainer.so"
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libc++_shared.so"
)

# Check main build artifacts
ALL_FOUND=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo -e "  ${GREEN}[OK]${NC} $(basename $file) ($size)"
    else
        echo -e "  ${RED}[MISSING]${NC} $file"
        ALL_FOUND=false
    fi
done

if [ "$ALL_FOUND" = false ]; then
    log_error "Some required files are missing"
    log_info "Please run: ./build_android.sh"
    exit 1
fi

# Check dependencies with fallback to obj/local/arm64-v8a
for file in "${DEP_FILES[@]}"; do
    filename=$(basename "$file")

    # Special handling for libc++_shared.so (Try copy from NDK)
    if [[ "$filename" == "libc++_shared.so" ]] && [ ! -f "$file" ]; then
        if [ -n "$ANDROID_NDK" ]; then
            # Attempt to find it in typical NDK locations for aarch64
            NDK_LIBCXX=$(find "$ANDROID_NDK" -name "libc++_shared.so" 2>/dev/null | grep "aarch64" | head -n 1)

            if [ -n "$NDK_LIBCXX" ] && [ -f "$NDK_LIBCXX" ]; then
                log_warning "libc++_shared.so not found in build dir, copying from NDK..."
                cp "$NDK_LIBCXX" "$file"
                # Fall through to standard check to confirm copy success
            fi
        fi
    fi

    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo -e "  ${GREEN}[OK]${NC} $filename ($size)"
    else
        # Try to find in obj directory
        obj_path="$SCRIPT_DIR/jni/obj/local/arm64-v8a/$filename"
        if [ -f "$obj_path" ]; then
            log_warning "$filename found in obj, copying to libs..."
            cp "$obj_path" "$file"
            size=$(du -h "$file" | cut -f1)
            echo -e "  ${GREEN}[OK]${NC} $filename ($size) (Copied)"
        else
            echo -e "  ${RED}[MISSING]${NC} $filename"
            log_error "Required dependency not found"
            exit 1
        fi
    fi
done

log_success "All required build artifacts found"

# Check optional files (API and test app)
OPTIONAL_FILES=(
    "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_api.so"
    "$SCRIPT_DIR/jni/libs/arm64-v8a/test_api"
)

for file in "${OPTIONAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo -e "  ${GREEN}[OK]${NC} $(basename $file) ($size) (Optional)"
    fi
done

# Create directories on device
log_step "3/3" "Push files to device"
log_info "Creating directories on device..."
adb shell "mkdir -p $INSTALL_DIR"
adb shell "mkdir -p $MODEL_DIR"
log_success "Directories created"

# Push executables
log_info "Pushing executables..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/nntrainer_causallm" "$INSTALL_DIR/" 2>&1 | tail -1
adb shell "chmod 755 $INSTALL_DIR/nntrainer_causallm"
log_success "nntrainer_causallm pushed"

# Push optional test_api if exists
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/test_api" ]; then
    log_info "Pushing test_api..."
    adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/test_api" "$INSTALL_DIR/" 2>&1 | tail -1
    adb shell "chmod 755 $INSTALL_DIR/test_api"
    log_success "test_api pushed"
fi


log_info "Pushing nntr_quantize..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/nntr_quantize" "$INSTALL_DIR/" 2>&1 | tail -1
adb shell "chmod 755 $INSTALL_DIR/nntr_quantize"
log_success "nntr_quantize pushed"

# Push shared libraries
log_info "Pushing shared libraries..."
log_info "  [1/5] libcausallm_core.so (CausalLM Core library)..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_core.so" "$INSTALL_DIR/" 2>&1 | tail -1

log_info "  [2/5] libnntrainer.so (nntrainer library)..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libnntrainer.so" "$INSTALL_DIR/" 2>&1 | tail -1

log_info "  [3/5] libccapi-nntrainer.so (nntrainer C/C API)..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libccapi-nntrainer.so" "$INSTALL_DIR/" 2>&1 | tail -1

log_info "  [4/5] libc++_shared.so (C++ runtime)..."
adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libc++_shared.so" "$INSTALL_DIR/" 2>&1 | tail -1

log_info "  [5/5] libcausallm_api.so (CausalLM API library)..."
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_api.so" ]; then
    adb push "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_api.so" "$INSTALL_DIR/" 2>&1 | tail -1
else
    log_warning "libcausallm_api.so not found (Optional, skipping)"
fi

log_success "All libraries pushed"

# Create run script on device
log_info "Creating run script on device..."
adb shell "cat > $INSTALL_DIR/run_causallm.sh << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=$INSTALL_DIR:\$LD_LIBRARY_PATH
export NNTR_NUM_THREADS=4
cd $INSTALL_DIR
./nntrainer_causallm \$@
EOF
"
adb shell "chmod 755 $INSTALL_DIR/run_causallm.sh"

# Create quantize run script on device
adb shell "cat > $INSTALL_DIR/run_quantize.sh << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=$INSTALL_DIR:\$LD_LIBRARY_PATH
cd $INSTALL_DIR
./nntr_quantize \$@
EOF"

adb shell "chmod 755 $INSTALL_DIR/run_quantize.sh"

# Create test script on device if API lib exists
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/test_api" ]; then
    adb shell "cat > $INSTALL_DIR/run_test_api.sh << 'EOF'
#!/system/bin/sh
export LD_LIBRARY_PATH=$INSTALL_DIR:\$LD_LIBRARY_PATH
export NNTR_NUM_THREADS=4
cd $INSTALL_DIR
./test_api \$@
EOF
"
    adb shell "chmod 755 $INSTALL_DIR/run_test_api.sh"
    log_info "Run script for test_api created"
fi

log_success "Run scripts created"

# Summary
log_header "Installation Complete!"
log_info "Device: $DEVICE_ID"
log_info "Install directory: $INSTALL_DIR"
log_info "Installed files:"
log_info "  - nntrainer_causallm (executable)"
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/test_api" ]; then
    log_info "  - test_api (executable)"
fi
log_info "  - libcausallm_core.so (CausalLM Core library)"
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm_api.so" ]; then
    log_info "  - libcausallm_api.so (CausalLM API library)"
fi
log_info "  - libnntrainer.so"
log_info "  - libccapi-nntrainer.so"
log_info "  - libc++_shared.so"
log_header "How to run"
log_info "To run CausalLM on the device:"
log_info "  1. Push your model files to: $MODEL_DIR/"
log_info "      Example: adb push res/qwen3/qwen3-4b $MODEL_DIR/qwen3-4b/"
log_info "2. Run the application:"
log_info "   adb shell $INSTALL_DIR/run_causallm.sh $MODEL_DIR/qwen3-4b"
log_info ""
log_info "(optional) Run quantization:"
log_info "  adb shell $INSTALL_DIR/run_quantize.sh $MODEL_DIR/qwen3-4b --fc_dtype Q4_0"
log_info ""
log_info "For interactive shell:"
log_info "   adb shell"
log_info "   cd $INSTALL_DIR"
log_info "   ./run_causallm.sh $MODEL_DIR/qwen3-4b"
if [ -f "$SCRIPT_DIR/jni/libs/arm64-v8a/test_api" ]; then
    log_info "To run API Test on the device:"
    log_info "  adb shell $INSTALL_DIR/run_test_api.sh [ARGS]"
fi

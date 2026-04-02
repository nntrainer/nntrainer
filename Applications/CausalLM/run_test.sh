#!/bin/bash

# Run test_causal_lm on Android device
# This script pushes the test app to device and executes it
set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    local timestamp=$(date "+%H:%M:%S")
    echo -e "${YELLOW}[$timestamp] [INFO]${NC} $1"
}

log_success() {
    local timestamp=$(date "+%H:%M:%S")
    echo -e "${GREEN}[$timestamp] [SUCCESS]${NC} $1"
}

log_warning() {
    local timestamp=$(date "+%H:%M:%S")
    echo -e "${CYAN}[$timestamp] [WARNING]${NC} $1"
}

log_error() {
    local timestamp=$(date "+%H:%M:%S")
    echo -e "${RED}[$timestamp] [ERROR]${NC} $1"
}

log_step() {
    local step="$1"
    local total="$2"
    local desc="$3"
    echo ""
    echo "[$step/$total] $desc"
    echo "----------------------------------------"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/data/local/tmp/nntrainer/causallm"
MODEL_DIR="${INSTALL_DIR}/models"

# Default arguments
MODEL_NAME="qwen3-0.6b"
PROMPT="Hello, how are you?"
USE_CHAT_TEMPLATE="1"
QUANTIZATION="W4A32"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        --chat-template)
            USE_CHAT_TEMPLATE="$2"
            shift 2
            ;;
        --quantization)
            QUANTIZATION="$2"
            shift 2
            ;;
        *)
            log_warning "Unknown option: $1 (ignoring)"
            shift
            ;;
    esac
done


# Start time measurement
START_TIME=$(date +%s)

log_info "Run Test Causal LM"
log_info "========================================"
log_info "Working directory: $(pwd)"
echo ""
log_info "Test parameters:"
log_info "  Model Name: $MODEL_NAME"
log_info "  Prompt: $PROMPT"
log_info "  Chat Template: $USE_CHAT_TEMPLATE"
log_info "  Quantization: $QUANTIZATION"
echo ""

# Step 1: Check device connection
log_step "1" "6" "Check device connection"
log_info "Checking adb devices..."

if ! command -v adb &> /dev/null; then
    log_error "adb command not found"
    log_error "Please install Android SDK Platform Tools"
    exit 1
fi

DEVICES=$(adb devices | grep -c "device$") || true
if [ "$DEVICES" -eq 0 ]; then
    log_error "No Android device connected"
    log_error "Please connect a device and enable USB debugging"
    exit 1
fi

if [ "$DEVICES" -gt 1 ]; then
    log_warning "Multiple devices detected, using first one"
fi

DEVICE_ID=$(adb devices | grep "device$" | head -1 | cut -f1)
log_success "Device connected: $DEVICE_ID"

# Step 2: Check test executable
log_step "2" "6" "Check test executable"
TEST_EXEC="$SCRIPT_DIR/jni/libs/arm64-v8a/test_causal_lm"
log_info "Checking: $TEST_EXEC"

if [ ! -f "$TEST_EXEC" ]; then
    log_error "test_causal_lm not found"
    log_error "Please run: ./build_test_app.sh"
    exit 1
fi

TEST_SIZE=$(du -h "$TEST_EXEC" | cut -f1)
log_success "test_causal_lm found (size: $TEST_SIZE)"

# Step 3: Check libcausallm.so
log_step "3" "6" "Check libcausallm.so"
LIB_CAUSALLM="$SCRIPT_DIR/jni/libs/arm64-v8a/libcausallm.so"
log_info "Checking: $LIB_CAUSALLM"

if [ ! -f "$LIB_CAUSALLM" ]; then
    log_error "libcausallm.so not found"
    log_error "Please run: ./build_causallm_lib.sh"
    exit 1
fi

LIB_SIZE=$(du -h "$LIB_CAUSALLM" | cut -f1)
log_success "libcausallm.so found (size: $LIB_SIZE)"

# Step 4: Prepare device directories
log_step "4" "6" "Prepare device directories"
log_info "Creating directories on device..."
log_info "  Install dir: $INSTALL_DIR"
log_info "  Model dir: $MODEL_DIR"

adb shell "mkdir -p $INSTALL_DIR" || true
adb shell "mkdir -p $MODEL_DIR" || true
log_success "Directories created"

# Step 5: Push files to device
log_step "5" "6" "Push files to device"
cd "$SCRIPT_DIR/jni/libs/arm64-v8a"

log_info "Pushing files to $INSTALL_DIR/..."

# Push executable
log_info "  [1/6] test_causal_lm..."
adb push test_causal_lm "$INSTALL_DIR/" || true

# Push libraries
log_info "  [2/6] libcausallm.so..."
adb push libcausallm.so "$INSTALL_DIR/" || true

log_info "  [3/6] libnntrainer.so..."
adb push libnntrainer.so "$INSTALL_DIR/" || true

log_info "  [4/6] libccapi-nntrainer.so..."
adb push libccapi-nntrainer.so "$INSTALL_DIR/" || true

log_info "  [5/6] libc++_shared.so..."
adb push libc++_shared.so "$INSTALL_DIR/" || true

# Push json.hpp
log_info "  [6/6] json.hpp..."
if [ -f "$SCRIPT_DIR/json.hpp" ]; then
    adb push "$SCRIPT_DIR/json.hpp" "$INSTALL_DIR/" || true
else
    log_warning "json.hpp not found, skipping"
fi

log_success "All files pushed to device"

# Make executable
log_info "Setting executable permissions..."
adb shell "chmod 755 $INSTALL_DIR/test_causal_lm" || true
log_success "Permissions set"

# Step 6: Run test
log_step "6" "6" "Run test on device"
log_info "Executing test_causal_lm..."
log_info "Command: export LD_LIBRARY_PATH=$INSTALL_DIR:\$LD_LIBRARY_PATH && ./test_causal_lm '$MODEL_NAME' '$PROMPT' $USE_CHAT_TEMPLATE $QUANTIZATION"
echo ""

# Run and capture output
RUN_START=$(date +%s)

echo "========================================"
echo "TEST EXECUTION OUTPUT"
echo "========================================"
adb shell "cd $INSTALL_DIR && export LD_LIBRARY_PATH=$INSTALL_DIR:\$LD_LIBRARY_PATH && ./test_causal_lm '$MODEL_NAME' '$PROMPT' $USE_CHAT_TEMPLATE $QUANTIZATION" 2>&1 || true
TEST_EXIT_CODE=$?
RUN_END=$(date +%s)
RUN_DURATION=$((RUN_END - RUN_START))

echo ""
echo "========================================"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    log_success "Test execution completed successfully"
else
    log_error "Test execution failed with exit code: $TEST_EXIT_CODE"
fi
log_info "Execution time: ${RUN_DURATION}s"

# End time measurement
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

# Summary
echo ""
log_info "========================================"
log_info "Run Complete"
log_info "========================================"
log_info "Total time: ${TOTAL_DURATION}s"
log_info "Device: $DEVICE_ID"
log_info "Install dir: $INSTALL_DIR"
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    log_success "All steps completed successfully!"
else
    log_error "Test failed with exit code: $TEST_EXIT_CODE"
    exit $TEST_EXIT_CODE
fi

log_info ""
log_info "For interactive shell access:"
log_info "  adb shell"
log_info "  cd $INSTALL_DIR"
log_info "  ./test_causal_lm <model_path> <prompt> <chat_template> <quantization>"

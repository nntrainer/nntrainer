#!/usr/bin/env bash
set -e

# This is a script to run NNTrainer unit tests on Android devices
# Note that this script assumes to be run on the nntrainer root path.

# ========== CONFIGURATION ==========
# Device paths
DEVICE_INSTALL_PATH="/data/local/tmp/nntr_android_test"
DEVICE_RES_PATH="${DEVICE_INSTALL_PATH}/res"
DEVICE_OPENCL_PATH="${DEVICE_INSTALL_PATH}/nntrainer_opencl_kernels"

# OpenCL configuration
OPENCL_MESON_FLAG="-Denable-opencl=true"
ENABLE_GPU=0
USE_BUILD_CACHE=0

# Meson base options (shared for consistency)
declare -a MESON_OPTIONS=(
  "-Dopenblas-num-threads=1"
  "-Denable-tflite-interpreter=false"
  "-Denable-tflite-backbone=false"
  "-Denable-fp16=true"
  "-Dnntr-num-threads=1"
  "-Dhgemm-experimental-kernel=false"
)

# ========== LOGGING UTILITIES ==========
log_info()    { echo "[INFO] $1"; }
log_error()   { echo "[ERROR] $1" >&2; }
log_warning() { echo "[WARNING] $1" >&2; }
log_step()    { echo ""; echo "========== $1 =========="; }

exit_error() {
  log_error "$1"
  exit 1
}

check_dir() {
  if [ ! -d "$1" ]; then
    exit_error "Directory not found: $1"
  fi
}

# ========== ARGUMENT PARSING ==========
parse_arguments() {
  declare -a filtered_args=()

  for arg in "$@"; do
    case "$arg" in
      --cache)
        USE_BUILD_CACHE=1
        log_info "Build cache enabled (will skip package_android.sh)"
        ;;
      "$OPENCL_MESON_FLAG")
        ENABLE_GPU=1
        log_info "OpenCL GPU enabled"
        filtered_args+=("$arg")
        ;;
      -D*)
        filtered_args+=("$arg")
        ;;
      *)
        filtered_args+=("$arg")
        ;;
    esac
  done

  printf '%s\n' "${filtered_args[@]}"
}

# ========== ADB UTILITIES ==========
adb_cmd() {
  if [ -v ADB_IP ]; then
    adb -H "${ADB_IP}" "$@"
  else
    adb "$@"
  fi
}

verify_adb_connection() {
  log_step "Verify ADB Connection"

  local connected_devices=$(adb_cmd devices | grep -c "device$")
  if [ "$connected_devices" -lt 1 ]; then
    exit_error "No ADB devices found. Please ensure device is connected"
  fi

  log_info "ADB connection verified (devices: $connected_devices)"
}

setup_device_directories() {
  log_step "Setup Device Directories"

  if ! adb_cmd shell mkdir -p "$DEVICE_INSTALL_PATH"; then
    exit_error "Failed to create device directory: $DEVICE_INSTALL_PATH"
  fi

  if ! adb_cmd shell mkdir -p "$DEVICE_RES_PATH"; then
    exit_error "Failed to create device res directory: $DEVICE_RES_PATH"
  fi

  if [[ $ENABLE_GPU -eq 1 ]]; then
    if ! adb_cmd shell mkdir -p "$DEVICE_OPENCL_PATH"; then
      exit_error "Failed to create device OpenCL directory: $DEVICE_OPENCL_PATH"
    fi
  fi

  log_info "Device directories ready"
}

push_test_binaries() {
  log_step "Push Test Binaries"

  pushd test/libs/arm64-v8a > /dev/null

  if ! adb_cmd push . "$DEVICE_INSTALL_PATH"; then
    exit_error "Failed to push binaries to $DEVICE_INSTALL_PATH"
  fi

  popd > /dev/null
  log_info "Test binaries pushed successfully"
}

# ========== BUILD UTILITIES ==========
build_ndk() {
  log_step "NDK Build"

  check_dir "$ANDROID_NDK"

  local ndk_args="-j$(nproc)"

  if [[ $ENABLE_GPU -eq 1 ]]; then
    ndk_args+=" MESON_ENABLE_OPENCL=1"
  fi

  pushd test/jni > /dev/null

  if ! ndk-build ${ndk_args}; then
    exit_error "NDK build failed"
  fi

  popd > /dev/null
  log_info "NDK build succeeded"
}

prepare_golden_data() {
  log_step "Prepare Golden Data"

  local meson_opts=("${MESON_OPTIONS[@]}")

  if [[ $ENABLE_GPU -eq 1 ]]; then
    meson_opts+=("$OPENCL_MESON_FLAG")
  fi

  if [ ! -d build ]; then
    log_info "Configuring Meson build (fresh)"
    if ! meson setup build "${meson_opts[@]}"; then
      exit_error "Meson build configuration failed"
    fi
  else
    log_warning "Build directory already exists, reconfiguring"

    if ! meson setup build --reconfigure "${meson_opts[@]}"; then
      exit_error "Meson reconfigure failed"
    fi

  fi

  log_info "Golden Data ready"
}

push_golden_data() {
  log_step "Push Golden Data"

  check_dir "build"

  if ! adb_cmd push build/res/ "$DEVICE_INSTALL_PATH"; then
    exit_error "Failed to push golden data to device"
  fi

  log_info "Golden Data pushed successfully"
}

build_libnntrainer() {
  local build_artifacts="builddir/android_build_result/lib/arm64-v8a/libnntrainer.so"

  if [[ $USE_BUILD_CACHE -eq 1 ]] && [ -f "$build_artifacts" ]; then
    log_step "Build libnntrainer.so (skipped: --cache enabled)"
    log_info "Reusing existing build artifacts"
  elif [[ $USE_BUILD_CACHE -eq 1 ]] && [ ! -f "$build_artifacts" ]; then
    log_step "Build libnntrainer.so"
    log_warning "Artifacts not found, building despite --cache flag"
    if ! ./tools/package_android.sh "${filtered_args[@]}"; then
      exit_error "package_android.sh failed"
    fi
  else
    log_step "Build libnntrainer.so"
    if ! ./tools/package_android.sh "${all_args[@]}"; then
      exit_error "package_android.sh failed"
    fi
  fi
}

# ========== MAIN FLOW ==========
main() {
  log_step "Android Unit Tests Setup"

  # Step 1: Parse arguments
  local all_args=("$@")
  local filtered_args=($(parse_arguments "$@"))

  # Step 2: Build libnntrainer.so
  build_libnntrainer

  # Step 3: NDK build
  build_ndk

  # Step 4: Device connection and setup
  verify_adb_connection
  setup_device_directories

  # Step 5: Push test binaries
  push_test_binaries

  # Step 6: Prepare golden data
  # To test unittest_layer, unittest_model, etc., golden data is required for the layer.
  # Meson setup without build will unzip golden data for the unit tests
  prepare_golden_data

  # Step 7: Push golden data
  push_golden_data

  log_info "All steps completed successfully"
}

main "$@"

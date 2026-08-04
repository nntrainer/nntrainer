#!/bin/bash
# setup.sh — Automated device setup for the Neuron backend smoke test.
# Pushes binaries, libraries, and test data to an Android device via adb.
#
# Usage:
#   ./setup.sh [device-serial]
#
# Examples:
#   ./setup.sh              # Uses the first connected device
#   ./setup.sh emulator-5554  # Pushes to a specific device

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE_SERIAL="${1:-}"

# Determine which adb command to use
ADB_CMD="adb"
if [ -n "$DEVICE_SERIAL" ]; then
  ADB_CMD="adb -s $DEVICE_SERIAL"
fi

# Check if device is connected
echo "Checking device connectivity..."
if ! $ADB_CMD devices | grep -q "device$"; then
  echo "ERROR: No device found. Make sure your device is connected and adb is enabled."
  exit 1
fi

echo "Device found. Pushing files to /data/local/tmp..."

# Files to push. libnntrainer.so, libccapi-nntrainer.so and libc++_shared.so
# are in the test binary's DT_NEEDED -- it will not start without them.
FILES=(
  "nntrainer_neuron_smoke"
  "libneuron_context.so"
  "libnntrainer.so"
  "libccapi-nntrainer.so"
  "libc++_shared.so"
  "libneuron_runtime.so"
  "model.dla"
  "golden_output.bin"
)

# golden_output.bin is genuinely optional (the first run produces it via
# --dump); everything else missing means the test cannot run, so fail loudly
# rather than pushing a bundle that dies with a linker error on device.
OPTIONAL=("golden_output.bin")

missing=()
for file in "${FILES[@]}"; do
  if [ ! -f "$SCRIPT_DIR/$file" ]; then
    is_optional=0
    for opt in "${OPTIONAL[@]}"; do
      [ "$file" = "$opt" ] && is_optional=1
    done
    if [ "$is_optional" -eq 1 ]; then
      echo "NOTE: $file not present (optional). Skipping."
    else
      missing+=("$file")
    fi
    continue
  fi
  echo "Pushing $file..."
  $ADB_CMD push "$SCRIPT_DIR/$file" /data/local/tmp/ > /dev/null
done

if [ "${#missing[@]}" -gt 0 ]; then
  echo ""
  echo "ERROR: required file(s) missing from $SCRIPT_DIR:"
  for file in "${missing[@]}"; do
    echo "  - $file"
  done
  echo ""
  echo "Copy them from your build output, e.g.:"
  echo "  cp builddir/jni/arm64-v8a/{nntrainer_neuron_smoke,libneuron_context.so,libnntrainer.so,libccapi-nntrainer.so,libc++_shared.so} $SCRIPT_DIR/"
  echo "plus the SoC-matched libneuron_runtime.so and your model.dla."
  exit 1
fi

echo "Files pushed successfully."

# Make the binary executable
echo "Setting execute permissions..."
$ADB_CMD shell chmod +x /data/local/tmp/nntrainer_neuron_smoke

# Verify files
echo "Verifying files on device..."
$ADB_CMD shell ls -lh /data/local/tmp/ | grep -E "(nntrainer_neuron_smoke|libneuron|model|golden)"

echo ""
echo "Setup complete! To run the test:"
echo "  adb shell"
echo "  cd /data/local/tmp"
echo "  export LD_LIBRARY_PATH=/data/local/tmp:\$LD_LIBRARY_PATH"
echo "  export NNTRAINER_NEURON_SMOKE_VERBOSE=1"
echo ""
echo "  # Pass your .dla's real shapes -- the 1:1:1:1 defaults are a placeholder."
echo "  # First run: dump the output so it can seed a golden."
echo "  ./nntrainer_neuron_smoke model.dla \\"
echo "      --in-shape=1:3:224:224 --out-shape=1:1000:1:1 --dump=golden_output.bin"
echo ""
echo "  # Once the values are independently confirmed, use it as a regression check:"
echo "  ./nntrainer_neuron_smoke model.dla golden_output.bin \\"
echo "      --in-shape=1:3:224:224 --out-shape=1:1000:1:1"

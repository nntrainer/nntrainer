#!/usr/bin/env bash
# Run the V-JEPA-2 ViT-B/16 encoder (24 frames x 256x256, Q4_0-FP16) on an
# Android device and report latency, peak RAM and cosine-vs-torch accuracy.
#
# Prereqg:
#   - build_android.sh has produced jni/libs/arm64-v8a/{libcausallm_core.so,
#     nntrainer_causallm} and builddir/android_build_result/lib/arm64-v8a/
#     {libnntrainer.so,libccapi-nntrainer.so,libc++_shared.so}
#   - the model dir res/vjepa2/vjepa2_24f256_q4arm (config + Q4_0 ARM-repacked
#     weight bin + input_video.bin) exists
#
# Usage: run_device.sh [DEVICE_SERIAL] [THREADS]
set -euo pipefail

DEV="${1:-$(adb devices | awk 'NR==2{print $1}')}"
THREADS="${2:-8}"
DST=/data/local/tmp/nntrainer/causallm
MODEL=vjepa2_24f256_q4arm

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$HERE/../.."                  # Applications/CausalLM
ROOT="$APP/../.."                  # repo root
JLIB="$APP/jni/libs/arm64-v8a"
ABR="$ROOT/builddir/android_build_result/lib/arm64-v8a"
RES="$APP/res/vjepa2/$MODEL"

echo "[run_device] device=$DEV threads=$THREADS"

adb -s "$DEV" shell "mkdir -p $DST/models/$MODEL"

# --- push libs + executable ---
for f in "$ABR/libnntrainer.so" "$ABR/libccapi-nntrainer.so" "$ABR/libc++_shared.so" \
         "$JLIB/libcausallm_core.so" "$JLIB/nntrainer_causallm"; do
  [ -f "$f" ] && adb -s "$DEV" push "$f" "$DST/" >/dev/null
done
adb -s "$DEV" shell "chmod +x $DST/nntrainer_causallm"

# --- push model (config + weights + input) ---
for f in config.json generation_config.json nntr_config.json \
         nntr_vjepa2_vitb_q40_arm.bin input_video.bin; do
  [ -f "$RES/$f" ] && adb -s "$DEV" push "$RES/$f" "$DST/models/$MODEL/" >/dev/null
done

# --- run ---
echo "[run_device] running..."
adb -s "$DEV" shell "cd $DST && LD_LIBRARY_PATH=. NNTR_NUM_THREADS=$THREADS \
  ./nntrainer_causallm models/$MODEL models/$MODEL/input_video.bin 2>&1 | \
  grep -iE 'First 10|nan|e2e|Resident'"

# --- pull output + compare to torch reference (token 0) ---
REF="${VJEPA_REF:-$HOME/vjepa2_ref/out_24f256/ref_output.npy}"
if [ -f "$REF" ]; then
  adb -s "$DEV" pull "$DST/models/$MODEL/input_video.bin.nntr_out.bin" /tmp/vjepa_dev_out.bin >/dev/null
  python3 "$HERE/compare_cosine.py" --ref "$REF" --nntr /tmp/vjepa_dev_out.bin
else
  echo "[run_device] torch ref not found ($REF); skipping cosine. Set VJEPA_REF=..."
fi

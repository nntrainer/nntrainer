#!/usr/bin/env bash
# Run the nntrainer YOLOv11m example on a given input .bin (default the one
# written by run_pytorch.py). Builds the target if needed.
# Usage: ./run_nntrainer.sh [INPUT_BIN]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
RES="$HERE/res"
INPUT="${1:-$RES/input_run.bin}"
TARGET="Applications/CausalLM/models/YOLOv11/jni/yolov11_infer"
BIN="$ROOT/build/$TARGET"

if [ ! -f "$RES/yolov11m.safetensors" ]; then
  echo "ERROR: $RES/yolov11m.safetensors not found."
  echo "  Generate it first:"
  echo "    python $HERE/PyTorch/convert_weights.py --weights <model.pt> --out $RES/yolov11m.safetensors"
  exit 1
fi
if [ ! -f "$INPUT" ]; then
  echo "ERROR: input bin not found: $INPUT (run run_pytorch.py first)"
  exit 1
fi

if [ ! -x "$BIN" ]; then
  echo ">> building $TARGET ..."
  [ -d "$ROOT/build" ] || meson setup "$ROOT/build" -Denable-app=true
  ninja -C "$ROOT/build" "$TARGET"
fi

echo "=== nntrainer (input: $INPUT) ==="
# YOLO_VERIFY=1 also prints per-stage max_abs_diff vs res/ref_*.bin if present.
"$BIN" "$RES" "$INPUT"

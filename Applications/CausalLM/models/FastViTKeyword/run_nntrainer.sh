#!/bin/bash
# Run FastViTKeyword inference on nntrainer.
#
# Prerequisites:
#   1. Build nntrainer with: meson setup build -Denable-transformer=true -Denable-app=true
#      ninja -C build
#   2. Convert weights: python PyTorch/convert_weights.py --weights /path/to/ckpt.pth
#   3. Extract reference: python PyTorch/extract_reference.py --weights /path/to/ckpt.pth
#
# Usage:
#   ./run_nntrainer.sh [input.bin]
#   KEYWORD_VERIFY=1 ./run_nntrainer.sh  # enable verification

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RES_DIR="$SCRIPT_DIR/res"
BIN="$REPO_ROOT/build/fastvit_keyword_infer"

INPUT="${1:-$RES_DIR/input.bin}"

if [ ! -f "$BIN" ]; then
    echo "Error: $BIN not found. Build first with:"
    echo "  meson setup build -Denable-transformer=true -Denable-app=true"
    echo "  ninja -C build"
    exit 1
fi

if [ ! -f "$RES_DIR/fastvit_keyword.safetensors" ]; then
    echo "Error: $RES_DIR/fastvit_keyword.safetensors not found."
    echo "Convert weights first:"
    echo "  DEEP_VISION_MODELS_PATH=/path/to/deep-vision-models \\"
    echo "  python PyTorch/convert_weights.py --weights /path/to/ckpt.pth"
    exit 1
fi

if [ ! -f "$INPUT" ]; then
    echo "Error: Input $INPUT not found."
    echo "Extract reference first:"
    echo "  DEEP_VISION_MODELS_PATH=/path/to/deep-vision-models \\"
    echo "  python PyTorch/extract_reference.py --weights /path/to/ckpt.pth"
    exit 1
fi

export OPENBLAS_NUM_THREADS=1
exec "$BIN" "$RES_DIR" "$INPUT"

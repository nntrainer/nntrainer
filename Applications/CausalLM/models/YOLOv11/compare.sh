#!/usr/bin/env bash
# One-shot PyTorch vs nntrainer comparison on the SAME input.
# run_pytorch.py preprocesses the input to res/input_run.bin (so both sides see
# identical bytes), prints PyTorch detections; then nntrainer runs on that bin.
#
# Usage:
#   ./compare.sh --weights /path/to/v11m_832rect_best.pt                 # seed-42 input
#   ./compare.sh --weights /path/to/v11m_832rect_best.pt --image cat.jpeg
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
INPUT="$HERE/res/input_run.bin"

echo "############## PyTorch ##############"
python3 "$HERE/PyTorch/run_pytorch.py" "$@" --save-input "$INPUT"

echo
echo "############## nntrainer ##############"
"$HERE/run_nntrainer.sh" "$INPUT"

echo
echo "Compare the detection lists above (boxes are xyxy pixels at the 832 scale)."

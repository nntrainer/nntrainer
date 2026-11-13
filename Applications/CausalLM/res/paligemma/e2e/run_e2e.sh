#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compare both engines against an explicitly generated PyTorch reference.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO/build_cuda}"
TEXT_DIR="${TEXT_DIR:-$HOME/paligemma-3b-nntr}"
VISION_DIR="${VISION_DIR:-$TEXT_DIR/vision}"
GOLDEN_DIR="${GOLDEN_DIR:?Set GOLDEN_DIR to the golden_dump.py output directory}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/paligemma-validation}"
export LD_LIBRARY_PATH="${CUDA_LIB_DIR:-$HOME/cuda-toolkit/lib64}:${LD_LIBRARY_PATH:-}"
for engine in cpu cuda; do
  python3 "$REPO/tools/validate_vla_ports.py" --out-dir "$OUTPUT_DIR" \
    "paligemma-$engine" "$engine" "$BUILD_DIR/Applications/CausalLM/nntr_paligemma_e2e" \
    "$TEXT_DIR" "$VISION_DIR" "$GOLDEN_DIR"
done

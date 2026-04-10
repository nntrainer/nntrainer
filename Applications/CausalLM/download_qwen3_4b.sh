#!/bin/bash
# Helper script to download Qwen3-0.6B and convert it to NNTrainer format

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_DIR="${SCRIPT_DIR}/res/qwen3/qwen3-0.6b"

# 1. Download model from HuggingFace
echo "========================================="
echo "Downloading Qwen3-0.6B from HuggingFace..."
echo "========================================="
# Requires: pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-0.6B --local-dir "${RES_DIR}" --local-dir-use-symlinks False

# 2. Copy the nntr_config.json into the model directory
echo "========================================="
echo "Setting up NNTrainer configuration..."
echo "========================================="
cp "${SCRIPT_DIR}/res/qwen3/qwen3-0.6b/nntr_config.json" "${RES_DIR}/nntr_config.json" 2>/dev/null || true

# 3. Convert weights to NNTrainer BIN format
echo "========================================="
echo "Converting weights to NNTrainer format..."
echo "========================================="
cd "${RES_DIR}"

# Requirements: pip install transformers torch
# Uses the weight_converter.py from the qwen3-4b directory (same converter works for 0.6B)
CONVERTER="${SCRIPT_DIR}/res/qwen3/qwen3-4b/weight_converter.py"
if [ ! -f "${CONVERTER}" ]; then
  echo "ERROR: weight_converter.py not found at ${CONVERTER}"
  echo "Please ensure the converter script exists."
  exit 1
fi

python3 "${CONVERTER}" --model_path . --output_name nntr_qwen3_0.6b_fp32.bin --data_type float32

echo "========================================="
echo "Conversion complete!"
echo "Weight file: ${RES_DIR}/nntr_qwen3_0.6b_fp32.bin"
echo ""
echo "To run training:"
echo "  ./causal_lm_train ${RES_DIR} <train_data.txt> --max_samples 10 --epochs 1"
echo ""
echo "To test without pre-trained weights (random init):"
echo "  ./causal_lm_train ${RES_DIR} <train_data.txt> --max_samples 5 --epochs 1 --skip_weights"
echo "========================================="

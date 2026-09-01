#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
#
# Build the Gemma2-2B QS4CX-FP16 model used in the 1K benchmark
# (gemma2_lg_q6k): embedding Q6_K, FC QS4CX, lm_head Q6_K, FP16 activations.
#
# Pipeline (verified to reproduce the tested nntr_config recipe):
#   1. weight_converter.py : HF Gemma2-2B -> FP32-weights / FP16-norms .bin
#   2. nntr_quantize       : --fc_dtype QS4CX --embd_dtype Q6_K --lmhead_dtype Q6_K
#                            => model_tensor_type QS4CX-FP16
#
# Usage:
#   build_qs4cx.sh <hf_gemma2_dir> <out_dir> [nntr_quantize_binary]
#
#   <hf_gemma2_dir>  HF Gemma2-2B checkpoint (model.safetensors, config.json,
#                    tokenizer*.json, special_tokens_map.json)
#   <out_dir>        destination for the quantized model + nntr_config.json
#   nntr_quantize    optional path to the built tool (default: auto-detect under
#                    ../../../../build*/Applications/CausalLM/nntr_quantize)
set -euo pipefail

HF="${1:?usage: build_qs4cx.sh <hf_gemma2_dir> <out_dir> [nntr_quantize]}"
OUT="${2:?usage: build_qs4cx.sh <hf_gemma2_dir> <out_dir> [nntr_quantize]}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"

NNTR_QUANTIZE="${3:-}"
if [ -z "$NNTR_QUANTIZE" ]; then
  NNTR_QUANTIZE="$(ls "$REPO"/build*/Applications/CausalLM/nntr_quantize 2>/dev/null | head -1 || true)"
fi
[ -x "$NNTR_QUANTIZE" ] || { echo "nntr_quantize not found; pass it as arg 3" >&2; exit 1; }

STAGE="$(mktemp -d)/stage"
mkdir -p "$STAGE" "$OUT"

# 1) HF -> FP32 weights + FP16 norms layer-graph .bin (quantize source)
python3 "$HERE/weight_converter.py" \
  --model_path "$HF" \
  --output_name "$STAGE/nntr_gemma2_2b_mixed.bin" \
  --data_type float32 --norm_fp16

# 2) assemble the stage dir nntr_quantize reads (config + tokenizer + source config)
cp "$HF/config.json" "$STAGE/" 2>/dev/null || true
cp "$HF/generation_config.json" "$STAGE/" 2>/dev/null || true
cp "$HF"/tokenizer*.json "$STAGE/" 2>/dev/null || true
cp "$HF/special_tokens_map.json" "$STAGE/" 2>/dev/null || true
cat > "$STAGE/nntr_config.json" <<EOF
{
  "model_type": "CausalLM",
  "model_tensor_type": "FP32-FP16",
  "model_file_name": "nntr_gemma2_2b_mixed.bin",
  "fc_layer_dtype": "FP32",
  "embedding_dtype": "FP32",
  "lora_rank": 0, "lora_alpha": 0, "lora_target": [], "bad_word_ids": [],
  "fsu": false, "fsu_lookahead": 2,
  "num_to_generate": 32, "init_seq_len": 1024, "max_seq_len": 2048, "batch_size": 1,
  "tokenizer_file": "$STAGE/tokenizer.json",
  "sample_input": "The capital of France is"
}
EOF

# 3) quantize to the 1K-benchmark recipe: FC QS4CX + embedding Q6_K + lm_head Q6_K
"$NNTR_QUANTIZE" "$STAGE" \
  --fc_dtype QS4CX --embd_dtype Q6_K --lmhead_dtype Q6_K \
  -o "$OUT"

rm -rf "$(dirname "$STAGE")"
echo "Gemma2-2B QS4CX-FP16 (Q6_K embed/lm_head) written to: $OUT"
echo "Update nntr_config.json:tokenizer_file to the deployed tokenizer.json path before running."

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
#
# Build the Gemma4-E2B QS4CX-FP16 model used in the 1K benchmark
# (gemma4_e2b_qint4fp16_lmint4): embedding Q6_K, FC QS4CX, lm_head QS4CX
# (untied), FP16 activations, skip_prefill.
#
# Pipeline (matches the tested nntr_config recipe):
#   1. weight_converter.py : HF Gemma4-E2B -> FP32 weights / FP16 norms .bin.
#      Gemma4 always re-emits a dedicated lm_head slot (output_of_causallm), so
#      the source is "untied" and the lm_head can be quantized independently.
#   2. nntr_quantize       : --fc_dtype QS4CX --embd_dtype Q6_K --lmhead_dtype QS4CX
#                            => model_tensor_type QS4CX-FP16, untied QS4CX lm_head
#
# Usage:
#   build_qs4cx.sh <hf_gemma4_e2b_dir> <out_dir> [nntr_quantize_binary]
set -euo pipefail

HF="${1:?usage: build_qs4cx.sh <hf_gemma4_e2b_dir> <out_dir> [nntr_quantize]}"
OUT="${2:?usage: build_qs4cx.sh <hf_gemma4_e2b_dir> <out_dir> [nntr_quantize]}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"

NNTR_QUANTIZE="${3:-}"
if [ -z "$NNTR_QUANTIZE" ]; then
  NNTR_QUANTIZE="$(ls "$REPO"/build*/Applications/CausalLM/nntr_quantize 2>/dev/null | head -1 || true)"
fi
[ -x "$NNTR_QUANTIZE" ] || { echo "nntr_quantize not found; pass it as arg 3" >&2; exit 1; }

STAGE="$(mktemp -d)/stage"
mkdir -p "$STAGE" "$OUT"

# 1) HF -> FP32 weights + FP16 norms .bin (quantize source; dedicated lm_head slot)
python3 "$HERE/weight_converter.py" \
  --model_path "$HF" \
  --output_name "$STAGE/nntr_gemma4_fp32fp16.bin" \
  --data_type float32 --norm_dtype float16

# 2) assemble the stage dir nntr_quantize reads
cp "$HF/config.json" "$STAGE/" 2>/dev/null || true
cp "$HF/generation_config.json" "$STAGE/" 2>/dev/null || true
cp "$HF"/tokenizer*.json "$STAGE/" 2>/dev/null || true
cp "$HF/special_tokens_map.json" "$STAGE/" 2>/dev/null || true
cat > "$STAGE/nntr_config.json" <<EOF
{
  "model_type": "CausalLM",
  "model_tensor_type": "FP32-FP16",
  "model_file_name": "nntr_gemma4_fp32fp16.bin",
  "fc_layer_dtype": "FP32",
  "embedding_dtype": "FP32",
  "lmhead_dtype": "FP32",
  "lmhead_untie": true,
  "skip_prefill": true,
  "lora_rank": 0, "lora_alpha": 0, "lora_target": [], "bad_word_ids": [],
  "num_to_generate": 24, "init_seq_len": 512, "max_seq_len": 1024, "batch_size": 1,
  "tokenizer_file": "$STAGE/tokenizer.json",
  "sample_input": "<bos><|turn>user\nWhat is the capital of South Korea?<turn|>\n<|turn>model\n"
}
EOF

# 3) quantize to the 1K-benchmark recipe: FC QS4CX + embedding Q6_K + lm_head QS4CX
"$NNTR_QUANTIZE" "$STAGE" \
  --fc_dtype QS4CX --embd_dtype Q6_K --lmhead_dtype QS4CX \
  -o "$OUT"

rm -rf "$(dirname "$STAGE")"
echo "Gemma4-E2B QS4CX-FP16 (Q6_K embed, untied QS4CX lm_head) written to: $OUT"
echo "Update nntr_config.json:tokenizer_file to the deployed tokenizer.json path before running."

#!/usr/bin/env bash
# Run the V-JEPA-2 ViT-B/16 encoder on the host (x86) build and report the
# cosine vs the torch reference. x86 is used as a correctness sanity gate;
# it runs the reference attention path (the GEMM/flash path is ARM-FP16 only)
# and the FP32 weights, which match torch to ~1.0.
#
# Usage: run_x86.sh [MODEL_DIR] [REF_NPY]
#   default MODEL_DIR is an FP32 24f/256 dir assembled in /tmp.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$HERE/../.."                 # Applications/CausalLM
ROOT="$APP/../.."                 # repo root
BUILD="$ROOT/build"
RESV="$APP/res/vjepa2"

REF="${2:-$HOME/vjepa2_ref/out_24f256/ref_output.npy}"
MODEL="${1:-}"

# Assemble a host-runnable FP32 24f/256 model dir if none was given.
if [ -z "$MODEL" ]; then
  MODEL=/tmp/vjepa_x86_24f256
  mkdir -p "$MODEL"
  cp "$RESV/vjepa2_24f256_q4arm/config.json" "$MODEL/"
  cp "$RESV/vjepa2_24f256_q4arm/generation_config.json" "$MODEL/"
  ln -sf "$RESV/vjepa2_1_vit_base_384/nntr_vjepa2_vitb_fp32.bin" "$MODEL/"
  ln -sf "$(dirname "$REF")/input_video.bin" "$MODEL/"
  cat > "$MODEL/nntr_config.json" <<'JSON'
{ "model_tensor_type":"FP32-FP32","model_file_name":"nntr_vjepa2_vitb_fp32.bin",
  "model_type":"Model","embedding_dtype":"FP32","fc_layer_dtype":"FP32",
  "batch_size":1,"max_seq_len":3072,"init_seq_len":3072,"num_to_generate":0,
  "fsu":false,"skip_tokenizer":true }
JSON
fi

export LD_LIBRARY_PATH="$BUILD/nntrainer:$BUILD/api/ccapi:$BUILD/Applications/CausalLM:$BUILD/Applications/CausalLM/layers"
echo "[run_x86] model=$MODEL"
"$BUILD/Applications/CausalLM/nntr_causallm" "$MODEL" "$MODEL/input_video.bin" 2>&1 | grep -iE 'First 10|nan|e2e'
python3 "$HERE/compare_cosine.py" --ref "$REF" --nntr "$MODEL/input_video.bin.nntr_out.bin"

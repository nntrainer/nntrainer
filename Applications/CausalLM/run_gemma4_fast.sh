#!/usr/bin/env bash
# gemma4-e2b QINT4-FP16 on Jetson Orin (sm_87, integrated) CUDA backend -- FAST recipe.
#   prefill ~3460 TPS @1K (52x the eager floor) + decode ~16.8 TPS @1K ctx (2x).
# How it gets there (all committed on gpu/v8c-on-main):
#   - device-resident forward (the GPU paths below) so the forward is host-op-free,
#   - PREFILL CUDA-graph capture -> kills the cMA=0 per-op cudaStreamSynchronize floor
#     (integrated-gated: discrete GPUs/RTX keep their eager-async prefill, untouched),
#   - cuBLAS int8 IMMA FC + cuBLAS Tensor-Core prefill attention (NNTR_CUDA_GEMM_ATTN),
#   - M2B decode: capture the decode forward once, replay with the device d_pos updated.
# Run from anywhere:  Applications/CausalLM/run_gemma4_fast.sh [MODEL_DIR] [PROMPT]
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # repo root
cd "$ROOT"
MODEL=${1:-$HOME/jijoongmoon/models/gemma4}
BIN=./build/Applications/CausalLM/nntr_causallm
export LD_LIBRARY_PATH=$ROOT/build/nntrainer:$ROOT/build/Applications/CausalLM:$ROOT/build/Applications/CausalLM/layers:$ROOT/build/api/ccapi:$ROOT/subprojects/OpenBLAS/build/lib:/usr/local/cuda/lib64

export NNTR_ENGINE=cuda
# full device residency (forward must be host-op-free so the prefill graph captures)
export NNTR_CUDA_DEV_ACT=1
export NNTR_RMSNORM_CUDA_OFF=all
export NNTR_CUDA_ROPE=1 NNTR_CUDA_ATTN=1 NNTR_CUDA_QKNORM=1 NNTR_CUDA_GEGLU=1
export NNTR_CUDA_ELTWISE=1 NNTR_CUDA_KV_UVM=1 NNTR_CUDA_VCOPY_PREFILL=1
export NNTR_CUDA_FLASH_DECODE=64  # split-KV decode chunk size (64; =1 is degenerate per-key)
# the speed levers:
export NNTR_CUDA_GRAPH=1        # decode + PREFILL graph capture (prefill: integrated-gated)
export NNTR_CUDA_M2B=1          # decode: capture once + d_pos-aware replay -> 2x @1K ctx
export NNTR_FC_CUDA_CUBLAS=1    # int8 IMMA Tensor-Core FC (down-proj via K-chunk)
export NNTR_CUDA_GEMM_ATTN=1    # cuBLAS Tensor-Core prefill attention (d=256/512): 248ms->11ms vs block-Q
# diagnostics (optional): export NNTR_CUDA_GRAPH_DBG=1   # [PREFILL_GRAPH]/[CUDA_GRAPH] timings

echo "==== gemma4-e2b CUDA fast (prefill-graph + gemm-attn + M2B decode) ===="
"$BIN" "$MODEL" ${2:+"$2"}
echo "==== EXIT: $? ===="

#!/bin/bash
# CPU vs NPU prefill benchmark with batching
# Device: R3CX9078DNH (Samsung S24 Ultra, Snapdragon 8 Elite, HTP v79)
# Model: Qwen3-0.6B, Q4_0 FC weights
#
# Modes:
#   CPU      = NNTR_HEXAGON_DISABLE=1 (all CPU, no DSP)
#   NPU      = NNTR_USE_HEXAGON_CDSP=1 NNTR_HEXAGON_FLASH_ATTN=1 NNTR_HEXAGON_FUSED_FFN=1 (batched NPU)
#
# Per BUILD_OBSERVATIONS doc §6: must run from binary's own directory,
# NOT cd into model dir (DSP skel resolves relative to cwd).

set -e

DEVICE=R3CX9078DNH
APP_DIR=/data/local/tmp/nntrainer/causallm
MODEL_DIR=/data/local/tmp/nntrainer/causallm/models/qwen3-0.6b
BIN=nntrainer_causallm
RESULTS_DIR=/home/anirudh/nntrainer/bench_results_2026-08-20
RUNS=2
SEQ_LENS="300 600 900"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  CPU vs NPU Prefill Benchmark (with batching)"
echo "  Date: $(date)"
echo "  Device: $DEVICE"
echo "  Model: Qwen3-0.6B (Q4_0)"
echo "  Runs per cell: $RUNS"
echo "  Seq lens: $SEQ_LENS"
echo "============================================================"
echo ""

# Summary table file
SUMMARY="$RESULTS_DIR/summary.csv"
echo "seq_len,mode,run,prefill_ms,prefill_tps,prefill_tokens,gen_ms,gen_tps,total_ms,rpc_roundtrips,layer_flushes,output_token" > "$SUMMARY"

for seq_len in $SEQ_LENS; do
  for mode in CPU NPU; do
    for run in $(seq 1 $RUNS); do
      TAG="${seq_len}_${mode}_run${run}"
      OUTFILE="$RESULTS_DIR/${TAG}.txt"
      
      echo "--- Running SEQLEN=${seq_len} MODE=${mode} RUN=${run} ---"
      
      # Copy the right config
      adb -s $DEVICE shell "cp ${MODEL_DIR}/nntr_config_bench_${seq_len}.json ${MODEL_DIR}/nntr_config.json"
      
      # Set env vars based on mode
      case "$mode" in
        CPU)
          ENV_VARS="NNTR_HEXAGON_DISABLE=1"
          ;;
        NPU)
          ENV_VARS="NNTR_USE_HEXAGON_CDSP=1 NNTR_HEXAGON_FLASH_ATTN=1 NNTR_HEXAGON_FUSED_FFN=1"
          ;;
      esac
      
      # Run from APP_DIR (not MODEL_DIR) per BUILD_OBSERVATIONS §6
      adb -s $DEVICE shell "cd $APP_DIR && export LD_LIBRARY_PATH=$APP_DIR:\$LD_LIBRARY_PATH && export NNTR_NUM_THREADS=4 && $ENV_VARS timeout 300 ./$BIN $MODEL_DIR" > "$OUTFILE" 2>&1
      exit_code=$?
      
      # Extract results
      prefill_line=$(grep "prefill:" "$OUTFILE" 2>/dev/null | head -1)
      gen_line=$(grep "generation:" "$OUTFILE" 2>/dev/null | head -1)
      total_line=$(grep "total:" "$OUTFILE" 2>/dev/null | head -1)
      rpc_line=$(grep "REAL FastRPC" "$OUTFILE" 2>/dev/null | head -1)
      flush_count=$(grep -c "LAYER_FLUSH" "$OUTFILE" 2>/dev/null || echo 0)
      output_token=$(grep -E '^[&<].*$' "$OUTFILE" 2>/dev/null | head -1)
      
      # Parse numbers
      prefill_ms=$(echo "$prefill_line" | grep -oP '\d+ ms' | head -1 | grep -oP '\d+')
      prefill_tps=$(echo "$prefill_line" | grep -oP '\d+ TPS' | head -1 | grep -oP '\d+')
      prefill_tokens=$(echo "$prefill_line" | grep -oP '\d+ tokens' | head -1 | grep -oP '\d+')
      gen_ms=$(echo "$gen_line" | grep -oP '\d+ ms' | head -1 | grep -oP '\d+')
      gen_tps=$(echo "$gen_line" | grep -oP '\d+ TPS' | head -1 | grep -oP '\d+')
      total_ms=$(echo "$total_line" | grep -oP '\d+ ms' | head -1 | grep -oP '\d+')
      rpc_trips=$(echo "$rpc_line" | grep -oP '\d+' | head -1)
      
      echo "  Exit: $exit_code"
      echo "  $prefill_line"
      echo "  $gen_line"
      echo "  $total_line"
      echo "  RPC: $rpc_line"
      echo "  LAYER_FLUSH count: $flush_count"
      echo "  Output: $output_token"
      echo ""
      
      # Append to CSV
      echo "${seq_len},${mode},${run},${prefill_ms:-0},${prefill_tps:-0},${prefill_tokens:-0},${gen_ms:-0},${gen_tps:-0},${total_ms:-0},${rpc_trips:-0},${flush_count:-0},${output_token}" >> "$SUMMARY"
      
      # Sleep between runs to let DSP clean up
      sleep 5
    done
  done
done

echo ""
echo "============================================================"
echo "  BENCHMARK COMPLETE"
echo "  Results: $RESULTS_DIR/"
echo "  Summary: $SUMMARY"
echo "============================================================"
echo ""
echo "--- Summary CSV ---"
cat "$SUMMARY"

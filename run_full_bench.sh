#!/bin/bash
# Full benchmark: FP32 CPU vs FP32 NPU vs FP16 CPU vs FP16 NPU
# Across 300/600/900 token prefill lengths
# 3 runs each, report averages

DEVICE=R3CX9078DNH
APP_DIR=/data/local/tmp/nntrainer/causallm
MODEL_DIR=/data/local/tmp/nntrainer/causallm/models/qwen3-0.6b
BIN=nntrainer_causallm
RESULTS_DIR=/home/anirudh/nntrainer/bench_results_2026-08-24
RUNS=3
SEQ_LENS="300 600 900"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  FULL BENCHMARK: FP32/FP16 x CPU/NPU x 300/600/900"
echo "  Date: $(date)"
echo "  Device: $DEVICE"
echo "  Model: Qwen3-0.6B (Q4_0 weights, Q6_K embedding/lmhead)"
echo "  Runs per cell: $RUNS"
echo "============================================================"
echo ""

SUMMARY="$RESULTS_DIR/summary.csv"
echo "seq_len,dtype,mode,run,prefill_ms,prefill_tps,prefill_tokens,gen_ms,gen_tps,total_ms,peak_mem_kb" > "$SUMMARY"

for seq_len in $SEQ_LENS; do
  for dtype in FP32 FP16; do
    for mode in CPU NPU; do
      for run in $(seq 1 $RUNS); do
        TAG="${seq_len}_${dtype}_${mode}_run${run}"
        OUTFILE="$RESULTS_DIR/${TAG}.txt"
        
        echo "--- Running SEQLEN=${seq_len} DTYPE=${dtype} MODE=${mode} RUN=${run} ---"
        
        # Copy the right config
        if [ "$dtype" = "FP16" ]; then
          adb -s $DEVICE shell "cp ${MODEL_DIR}/nntr_config_bench_${seq_len}_fp16.json ${MODEL_DIR}/nntr_config.json"
        else
          adb -s $DEVICE shell "cp ${MODEL_DIR}/nntr_config_bench_${seq_len}.json ${MODEL_DIR}/nntr_config.json"
        fi
        
        # Set env vars based on mode
        case "$mode" in
          CPU)
            ENV_VARS="NNTR_HEXAGON_DISABLE=1"
            ;;
          NPU)
            ENV_VARS="NNTR_USE_HEXAGON_CDSP=1 NNTR_HEXAGON_FLASH_ATTN=1 NNTR_HEXAGON_FUSED_FFN=1"
            ;;
        esac
        
        # Run from APP_DIR
        adb -s $DEVICE shell "cd $APP_DIR && export LD_LIBRARY_PATH=$APP_DIR:\$LD_LIBRARY_PATH && export NNTR_NUM_THREADS=4 && $ENV_VARS timeout 300 ./$BIN $MODEL_DIR" > "$OUTFILE" 2>&1
        exit_code=$?
        
        # Extract results
        prefill_line=$(grep "prefill:" "$OUTFILE" 2>/dev/null | head -1)
        gen_line=$(grep "generation:" "$OUTFILE" 2>/dev/null | head -1)
        total_line=$(grep "total:" "$OUTFILE" 2>/dev/null | head -1)
        mem_line=$(grep "peak memory:" "$OUTFILE" 2>/dev/null | head -1)
        
        # Parse numbers
        prefill_ms=$(echo "$prefill_line" | grep -oP '\d+ ms' | head -1 | grep -oP '\d+')
        prefill_tps=$(echo "$prefill_line" | grep -oP '\d+ TPS' | head -1 | grep -oP '\d+')
        prefill_tokens=$(echo "$prefill_line" | grep -oP '\d+ tokens' | head -1 | grep -oP '\d+')
        gen_ms=$(echo "$gen_line" | grep -oP '\d+ ms' | head -1 | grep -oP '\d+')
        gen_tps=$(echo "$gen_line" | grep -oP '\d+ TPS' | head -1 | grep -oP '\d+')
        total_ms=$(echo "$total_line" | grep -oP '\d+ ms' | head -1 | grep -oP '\d+')
        peak_mem=$(echo "$mem_line" | grep -oP '\d+ KB' | head -1 | grep -oP '\d+')
        
        echo "  Exit: $exit_code"
        echo "  $prefill_line"
        echo "  $gen_line"
        echo "  $total_line"
        echo "  $mem_line"
        echo ""
        
        # Append to CSV
        echo "${seq_len},${dtype},${mode},${run},${prefill_ms:-0},${prefill_tps:-0},${prefill_tokens:-0},${gen_ms:-0},${gen_tps:-0},${total_ms:-0},${peak_mem:-0}" >> "$SUMMARY"
        
        # Sleep between runs to let DSP clean up
        sleep 5
      done
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

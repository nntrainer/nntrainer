 aut#!/bin/bash
# Host-side benchmark runner - runs each test individually with sleep between runs
# to avoid DSP session exhaustion
DEVICE=R3CX9078DNH
MODEL_DIR=/data/local/tmp/nntrainer/causallm/models/qwen3-0.6b
APP_DIR=/data/local/tmp/nntrainer/causallm
BIN=nntrainer_causallm
RESULTS_DIR=/home/anirudh/nntrainer/bench_results_2026-08-19
mkdir -p $RESULTS_DIR

run_one() {
  local seq_len=$1
  local mode=$2
  local run=$3
  local tag="${seq_len}_${mode}_run${run}"
  local outfile="$RESULTS_DIR/${tag}.txt"
  
  echo "=== Running SEQLEN=${seq_len} MODE=${mode} RUN=${run} ==="
  
  # Set up config
  adb -s $DEVICE shell "cp ${MODEL_DIR}/nntr_config_bench_${seq_len}.json ${MODEL_DIR}/nntr_config.json"
  
  # Build the env var prefix
  local env_prefix=""
  case "$mode" in
    CPU)
      env_prefix="NNTR_HEXAGON_DISABLE=1"
      ;;
    NPU_BATCH)
      env_prefix="NNTR_USE_HEXAGON_CDSP=1"
      ;;
    NPU_NOBATCH)
      env_prefix="NNTR_USE_HEXAGON_CDSP=1 NNTR_HEXAGON_NO_BATCH=1"
      ;;
    NPU_NO_ELEM)
      env_prefix="NNTR_USE_HEXAGON_CDSP=1 NNTR_HEXAGON_NO_ELEM_OPS=1"
      ;;
  esac
  
  # Run the benchmark
  adb -s $DEVICE shell "cd $APP_DIR && export LD_LIBRARY_PATH=$APP_DIR:\$LD_LIBRARY_PATH && export NNTR_NUM_THREADS=4 && $env_prefix timeout 300 ./nntrainer_causallm $MODEL_DIR" > "$outfile" 2>&1
  local exit_code=$?
  
  # Extract results
  local prefill=$(grep "prefill:" "$outfile" 2>/dev/null | head -1)
  local gen=$(grep "generation:" "$outfile" 2>/dev/null | head -1)
  local total=$(grep "total:" "$outfile" 2>/dev/null | head -1)
  local mem=$(grep "peak memory:" "$outfile" 2>/dev/null | head -1)
  local rpc=$(grep "REAL FastRPC" "$outfile" 2>/dev/null | head -1)
  
  echo "  Exit: $exit_code"
  echo "  $prefill"
  echo "  $gen"
  echo "  $total"
  echo "  $mem"
  echo "  $rpc"
  echo ""
  
  # Sleep between runs to let DSP clean up
  sleep 5
}

echo "Starting comprehensive benchmarks at $(date)"
echo "Device: $DEVICE"
echo ""

for seq_len in 300 600 900 1200; do
  for mode in CPU NPU_BATCH NPU_NOBATCH NPU_NO_ELEM; do
    for run in 1 2 3; do
      run_one $seq_len $mode $run
    done
  done
done

echo "All benchmarks complete at $(date)"
echo "Results in $RESULTS_DIR/"

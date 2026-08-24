#!/system/bin/sh
# Comprehensive benchmark sweep - runs on Android device
# Tests: 4 prompt lengths × 4 modes × 3 runs each
# Modes: CPU, NPU_BATCH, NPU_NOBATCH, NPU_NO_ELEM
MODEL_DIR=/data/local/tmp/nntrainer/causallm/models/qwen3-0.6b
LIB_PATH=/data/local/tmp/nntrainer/causallm
BIN=/data/local/tmp/nntrainer/causallm/nntrainer_causallm
export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH
export NNTR_NUM_THREADS=4

cd $MODEL_DIR

RUNS=3
RESULTS_DIR=/tmp/bench_results
mkdir -p $RESULTS_DIR

echo "=== COMPREHENSIVE BENCHMARK SWEEP START ==="
echo "Date: $(date)"
echo "Device: $(getprop ro.product.model)"
echo "==="

for SEQLEN in 300 600 900 1200; do
  # Copy the right config
  if [ -f nntr_config_bench_${SEQLEN}.json ]; then
    cp nntr_config_bench_${SEQLEN}.json nntr_config.json
  else
    echo "SKIP: nntr_config_bench_${SEQLEN}.json not found"
    continue
  fi

  for MODE in CPU NPU_BATCH NPU_NOBATCH NPU_NO_ELEM; do
    for RUN in $(seq 1 $RUNS); do
      TAG="${SEQLEN}_${MODE}_run${RUN}"
      echo "--- Running SEQLEN=${SEQLEN} MODE=${MODE} RUN=${RUN} ---"

      case "$MODE" in
        CPU)
          NNTR_HEXAGON_DISABLE=1 timeout 300 $BIN $MODEL_DIR > $RESULTS_DIR/${TAG}.txt 2>&1
          ;;
        NPU_BATCH)
          NNTR_USE_HEXAGON_CDSP=1 timeout 300 $BIN $MODEL_DIR > $RESULTS_DIR/${TAG}.txt 2>&1
          ;;
        NPU_NOBATCH)
          NNTR_USE_HEXAGON_CDSP=1 NNTR_HEXAGON_NO_BATCH=1 timeout 300 $BIN $MODEL_DIR > $RESULTS_DIR/${TAG}.txt 2>&1
          ;;
        NPU_NO_ELEM)
          NNTR_USE_HEXAGON_CDSP=1 NNTR_HEXAGON_NO_ELEM_OPS=1 timeout 300 $BIN $MODEL_DIR > $RESULTS_DIR/${TAG}.txt 2>&1
          ;;
      esac
      EXIT_CODE=$?
      echo "  Exit: $EXIT_CODE"

      # Extract results
      PREFILL=$(grep -oP 'prefill: \d+ tokens, \d+ ms' $RESULTS_DIR/${TAG}.txt 2>/dev/null)
      GEN=$(grep -oP 'generation: \d+ tokens, \d+ ms' $RESULTS_DIR/${TAG}.txt 2>/dev/null)
      TOTAL=$(grep -oP 'total: \d+ ms' $RESULTS_DIR/${TAG}.txt 2>/dev/null)
      MEM=$(grep -oP 'peak memory: \d+ KB' $RESULTS_DIR/${TAG}.txt 2>/dev/null)
      TPS=$(grep -oP 'prefill: \d+ tokens, \d+ ms, \d+ TPS' $RESULTS_DIR/${TAG}.txt 2>/dev/null)
      GEN_TPS=$(grep -oP 'generation: \d+ tokens, \d+ ms, \d+ TPS' $RESULTS_DIR/${TAG}.txt 2>/dev/null)
      RPC=$(grep -oP 'REAL FastRPC round-trip\(s\).*' $RESULTS_DIR/${TAG}.txt 2>/dev/null)
      OUTPUT_TOKEN=$(grep -E '^[&<].*$' $RESULTS_DIR/${TAG}.txt 2>/dev/null | head -1)

      echo "  RESULT: $PREFILL | $GEN | $TOTAL | $MEM"
      echo "  TPS: $TPS | $GEN_TPS"
      echo "  RPC: $RPC"
      echo "  Output: $OUTPUT_TOKEN"
      echo ""
    done
  done
done

echo "=== COMPREHENSIVE BENCHMARK SWEEP DONE ==="
echo "Raw results in $RESULTS_DIR/"

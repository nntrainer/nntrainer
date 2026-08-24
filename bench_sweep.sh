#!/system/bin/sh
# Benchmark sweep script - runs on Android device
MODEL_DIR=/data/local/tmp/nntrainer/causallm/models/qwen3-0.6b
LIB_PATH=/data/local/tmp/nntrainer/causallm
BIN=/data/local/tmp/nntrainer/causallm/nntrainer_causallm
export LD_LIBRARY_PATH=$LIB_PATH:$LD_LIBRARY_PATH

cd $MODEL_DIR

echo "=== BENCHMARK SWEEP START ==="
for SEQLEN in 300 600 900 1200; do
  cp nntr_config_bench_${SEQLEN}.json nntr_config.json
  for MODE in CPU NOBATCH BATCH; do
    echo "--- Running SEQLEN=${SEQLEN} MODE=${MODE} ---"
    if [ "$MODE" = "CPU" ]; then
      NNTR_HEXAGON_DISABLE=1 timeout 300 $BIN $MODEL_DIR > /tmp/bench_${SEQLEN}_${MODE}.txt 2>&1
    elif [ "$MODE" = "NOBATCH" ]; then
      NNTR_HEXAGON_NO_BATCH=1 timeout 300 $BIN $MODEL_DIR > /tmp/bench_${SEQLEN}_${MODE}.txt 2>&1
    else
      NNTR_HEXAGON_BATCH=1 timeout 300 $BIN $MODEL_DIR > /tmp/bench_${SEQLEN}_${MODE}.txt 2>&1
    fi
    echo "  Exit: $?"
    grep -E "prefill:|generation:|total:|peak" /tmp/bench_${SEQLEN}_${MODE}.txt 2>/dev/null
  done
done
echo "=== BENCHMARK SWEEP DONE ==="

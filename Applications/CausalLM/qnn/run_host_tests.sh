#!/usr/bin/env bash
# Standalone host checks, independent of Android and Meson configuration.
set -euo pipefail
causallm_dir="$(cd "$(dirname "$0")/.." && pwd)"
nntrainer_dir="$(cd "$causallm_dir/../.." && pwd)"
test_dir="$(mktemp -d /tmp/nntr-qnn-test.XXXXXX)"
trap 'rm -rf "$test_dir"' EXIT
includes=(-I"$causallm_dir" -I"$causallm_dir/api" -I"$causallm_dir/models"
          -I"$causallm_dir/qnn" -I"$causallm_dir/third_party"
          -I"$causallm_dir/third_party/minja/include"
          -I"$causallm_dir/xgrammar/include"
          -I"$causallm_dir/xgrammar/3rdparty/dlpack/include"
          -I"$nntrainer_dir/api" -I"$nntrainer_dir/api/ccapi/include")
while IFS= read -r dir; do includes+=(-I"$dir"); done < <(
  rg --files "$nntrainer_dir/nntrainer" -g '*.h' | xargs -r -n1 dirname | sort -u
)
"${CXX:-g++}" -std=c++17 -O1 -ffunction-sections -fdata-sections \
  "${includes[@]}" "$causallm_dir/api/unittest_qnn_runtime.cpp" \
  "$causallm_dir/qnn/generate_qnn_utils.cpp" \
  "$causallm_dir/qnn/qnn_kv_cache_manager.cpp" \
  "$causallm_dir/qnn/quick_dot_ai_qnn.cpp" \
  "$causallm_dir/qnn/qnn_row_readahead.cpp" \
  "$nntrainer_dir/nntrainer/utils/thread_manager.cpp" \
  "$nntrainer_dir/nntrainer/utils/thread_manager_util.cpp" \
  -Wl,--gc-sections -lgtest -lgtest_main -lpthread -ldl -o "$test_dir/qnn_test"
"$test_dir/qnn_test"
NNTR_HOSTOPT=0 QUICKAI_SAMPLE_SEED=123 "$test_dir/qnn_test"

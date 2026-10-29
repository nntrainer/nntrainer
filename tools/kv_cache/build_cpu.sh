#!/usr/bin/env bash
# Reproducible Linux CPU profile. Run from any directory; no system install.
set -euo pipefail
kv_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$kv_root"
kv_build=${1:-build-kv-cpu}
shift "$(( $# > 0 ? 1 : 0 ))"
export CC=${CC:-gcc}
export CXX=${CXX:-g++}
# The tracked tokenizer archive may require a newer libstdc++ than the host.
# Always build an isolated archive with this host's compiler; never replace it.
kv_build_abs=$(realpath -m "$kv_build")
mkdir -p "$kv_build_abs"
if [[ ! -e "$kv_build_abs/.gitignore" ]]; then
  printf '*\n' > "$kv_build_abs/.gitignore"
fi
kv_tokenizer_args=()
if [[ ${KV_TOKENIZER_OFFLINE:-0} == 1 ]]; then
  kv_tokenizer_args+=(--offline)
fi
Applications/CausalLM/tools/build_tokenizers_c.sh \
  --out-dir="$kv_build_abs/tokenizer" "${kv_tokenizer_args[@]}"
kv_rust_target=$(rustc -vV | sed -n 's/^host: //p')
# prepare_encoder.sh uses this marker but does not create it after extracting.
if [[ -f Applications/CausalLM/json.hpp && \
      -f Applications/LLaMA/jni/encoder.hpp ]]; then
  mkdir -p "$kv_build_abs/encoder"
fi
meson setup "$kv_build" --buildtype=debugoptimized \
  -Dplatform=none -Denable-app=true -Denable-transformer=true \
  -Denable-test=true -Denable-api=false -Denable-fp16=false \
  -Denable-opencl=false -Denable-cuda=false -Denable-npu=false \
  -Denable-htp=false -Denable-tflite-interpreter=false \
  -Denable-tflite-backbone=false -Denable-nnstreamer-backbone=false \
  -Dml-api-support=disabled -Denable-capi=disabled \
  -Denable-nnstreamer-tensor-filter=disabled \
  -Denable-nnstreamer-tensor-trainer=disabled -Dopenblas-num-threads=1 \
  -Dcausallm-tokenizer-lib="$kv_build_abs/tokenizer/$kv_rust_target/libtokenizers_c.a" "$@"
mkdir -p "$kv_build_abs/encoder"
ninja -C "$kv_build" -j"${KV_BUILD_JOBS:-8}"

#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Builds the x86 host-side hexagon tools/tests (no SDK, no device).
# Usage: ./tools/hexagon/build_host_x86.sh [target ...]
#   targets: test_lowering test_w8cx_bin nntr_hexpack hexagon_ref_run
#   (default: all)

set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO/build_x86_hexagon"
HOST="$REPO/nntrainer/tensor/hexagon/host"
HTP="$REPO/nntrainer/tensor/hexagon/htp"
APP="$REPO/Applications/CausalLM/hexagon"
SIM="$REPO/test/hexagon/sim"
CXX="${CXX:-g++}"

mkdir -p "$OUT"
INCS=(-I "$HOST" -I "$HTP" -I "$APP" -I "$SIM")
CXXFLAGS=(-std=c++17 -O2 -Wall -Werror "${INCS[@]}")

LOWER=("$HOST/graph_lowering.cpp" "$APP/qwen3_lowering.cpp")
IMAGE=("$APP/qwen3_w8cx_bin.cpp" "$APP/hex_image.cpp")

build_test_lowering() {
  "$CXX" "${CXXFLAGS[@]}" "$REPO/test/hexagon/test_lowering.cpp" \
      "${LOWER[@]}" "$APP/hex_image.cpp" -o "$OUT/test_lowering"
}
build_test_w8cx_bin() {
  "$CXX" "${CXXFLAGS[@]}" "$REPO/test/hexagon/test_w8cx_bin.cpp" \
      "$APP/qwen3_w8cx_bin.cpp" -o "$OUT/test_w8cx_bin"
}
build_nntr_hexpack() {
  "$CXX" "${CXXFLAGS[@]}" "$APP/hex_pack.cpp" "${LOWER[@]}" "${IMAGE[@]}" \
      -o "$OUT/nntr_hexpack"
}
build_hexagon_ref_run() {
  # ref_ops.c is C shared with the simulator tests; on x86 it is compiled
  # as C++ so that __fp16 can be a conversion struct (see ref_fp16_x86.h).
  "$CXX" "${CXXFLAGS[@]}" -march=native "$REPO/test/hexagon/hexagon_ref_run.cpp" \
      -x c++ "$SIM/ref_ops.c" -x none "${LOWER[@]}" "$APP/hex_image.cpp" \
      -o "$OUT/hexagon_ref_run"
}

if [ $# -eq 0 ]; then
  set -- test_lowering test_w8cx_bin nntr_hexpack hexagon_ref_run
fi
for t in "$@"; do
  "build_$t"
  echo "built: $OUT/$t"
done

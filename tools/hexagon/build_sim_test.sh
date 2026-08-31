#!/bin/bash
# tools/hexagon/build_sim_test.sh
# Cross-builds the hexagon-sim test lib (v75; SDK 6.0.0.2 has QuRT sim
# images up to v75). Prereq: source $HEXAGON_SDK_ROOT/setup_sdk_env.source
set -eu
: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"
HEX_ARCH="${HEX_ARCH:-v75}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HTP_DIR="$REPO/nntrainer/tensor/hexagon/htp"
SIM_DIR="$REPO/test/hexagon/sim"
OUT="$REPO/build_hexagon/sim"
mkdir -p "$OUT"

SRCS=("$SIM_DIR"/*.c)
# htp 소스는 Task 진행에 따라 자동 포함 (ops/, hvx/, worker_pool, htp_graph)
for f in "$HTP_DIR"/worker_pool.c "$HTP_DIR"/htp_graph.c \
         "$HTP_DIR"/ops/*.c "$HTP_DIR"/hvx/*.c "$HTP_DIR"/dma/*.c; do
  [ -e "$f" ] && SRCS+=("$f")
done

"$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang" \
    -m"$HEX_ARCH" -mhvx -mhvx-length=128B -mhvx-ieee-fp -G0 -O2 -g -fPIC -shared \
    -Wall -Werror -Wno-unused-function \
    -I "$HTP_DIR" -I "$HTP_DIR/ops" -I "$HTP_DIR/hvx" -I "$HTP_DIR/hex" -I "$HTP_DIR/dma" -I "$SIM_DIR" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/qurt" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/include/posix" \
    -isystem "$HEXAGON_SDK_ROOT/incs" \
    -isystem "$HEXAGON_SDK_ROOT/incs/stddef" \
    ${HEX_EXTRA_CFLAGS:-} \
    "${SRCS[@]}" \
    -o "$OUT/libnntr_sim_test.so"
echo "built: $OUT/libnntr_sim_test.so ($HEX_ARCH)"

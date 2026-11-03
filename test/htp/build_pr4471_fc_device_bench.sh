#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the original PR kernel, and a benchmark that only uses its FP32 API.
set -euo pipefail
SOURCE=${1:?Usage: build_pr4471_fc_device_bench.sh PR4471_WORKTREE OUTPUT_DIR}
OUT=${2:?Missing output directory}
HERE=$(cd "$(dirname "$0")" && pwd)
SOURCE=$(cd "$SOURCE" && pwd)
: "${HEXAGON_SDK_ROOT:?Set HEXAGON_SDK_ROOT}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?Set DEFAULT_HEXAGON_TOOLS_ROOT}"
: "${HEXKL_ROOT:?Set HEXKL_ROOT}"
export HEX_ARCH=${HEX_ARCH:-v81}
test "$(git -C "$SOURCE" rev-parse HEAD)" = 3881042471377ec3c083afd06e55a04ab6934cbe
git -C "$SOURCE" diff --quiet HEAD -- nntrainer test/htp
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
if [[ ${NNTR_BENCH_SKIP_BUILD:-0} != 1 ]]; then
  bash "$SOURCE/test/htp/build.sh" > "$OUT/build.log" 2>&1
fi
cp "$SOURCE/test/htp/build/libnntr_hvx_skel.so" "$OUT/"
BACKEND="$SOURCE/nntrainer/tensor/htp_backend"
CC="$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang"
"$CC" -m"$HEX_ARCH" -mhvx -mhvx-length=128B -mhmx -G0 -O3 -fPIC -shared \
  -Wall -Werror -I "$SOURCE/test/htp" -I "$SOURCE/test/htp/generated" \
  -I "$BACKEND" -I "$BACKEND/hmx" -I "$BACKEND/hvx" -I "$HEXKL_ROOT/include" \
  -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" \
  -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix" \
  -isystem "$HEXAGON_SDK_ROOT/incs" -isystem "$HEXAGON_SDK_ROOT/incs/stddef" \
  -isystem "$HEXAGON_SDK_ROOT/ipc/fastrpc/incs" \
  "$HERE/pr4471_fc_device_bench.c" "$OUT/libnntr_hvx_skel.so" -o "$OUT/libinteger_bench.so"
RUNNER="$HEXAGON_SDK_ROOT/libs/run_main_on_hexagon/ship"
cp "$RUNNER/android_aarch64/run_main_on_hexagon" "$OUT/"
cp "$RUNNER/hexagon_toolv19_$HEX_ARCH/librun_main_on_hexagon_skel.so" "$OUT/"
git -C "$SOURCE" rev-parse HEAD > "$OUT/source-commit.txt"
"$CC" --version > "$OUT/compiler.txt"

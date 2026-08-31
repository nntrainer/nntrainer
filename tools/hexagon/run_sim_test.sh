#!/bin/bash
# tools/hexagon/run_sim_test.sh <test-name> [args...]
set -eu
: "${HEXAGON_SDK_ROOT:?source setup_sdk_env.source first}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?source setup_sdk_env.source first}"
HEX_ARCH="${HEX_ARCH:-v75}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO/build_hexagon/sim"
# run_main_on_hexagon ships per (tools, arch); pick whichever exists for HEX_ARCH
RUNMAIN="$(ls -d "$HEXAGON_SDK_ROOT"/libs/run_main_on_hexagon/ship/hexagon_toolv*_"${HEX_ARCH}" | tail -1)/run_main_on_hexagon_sim"
ISS="$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/lib/iss"
cd "$OUT"
printf '%s\n' \
  "$ISS/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1" \
  "$ISS/l2vic.so 32 0xFC910000" > q6ss.cfg
echo "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/debugger/lnx64/qurt_model.so" > osam.cfg
"$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-sim" -m"$HEX_ARCH" \
    --simulated_returnval --usefs "$OUT" --nullptr=2 \
    --cosim_file "$OUT/q6ss.cfg" --l2tcm_base 0xd800 --rtos "$OUT/osam.cfg" \
    "$HEXAGON_SDK_ROOT/rtos/qurt/compute${HEX_ARCH}/sdksim_bin/runelf.pbn" -- \
    "$RUNMAIN" -- ./libnntr_sim_test.so "$@"

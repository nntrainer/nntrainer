#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Pushes the skel + test binary to the device, runs the round-trip test,
# and saves stdout + FARF logcat lines under logs/hexagon/.
#
# Usage: ./tools/hexagon/run_device_test.sh [adb-serial]

set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO/build_hexagon"
DEV_DIR=/data/local/tmp/nntr_htp
LOG_DIR="$REPO/logs/hexagon"
STAMP="$(date +%Y%m%d_%H%M%S)"
ADB=(adb)
[ $# -ge 1 ] && ADB=(adb -s "$1")

mkdir -p "$LOG_DIR"

"${ADB[@]}" shell mkdir -p "$DEV_DIR"
"${ADB[@]}" push "$OUT/skel/libnntr_htp_skel.so" "$DEV_DIR/"
"${ADB[@]}" push "$OUT/host/hexagon_rpc_test" "$DEV_DIR/"
# Forward DSP FARF messages to logcat (all levels); without this file the
# executor's FARF lines never reach device_farf_*.log.
"${ADB[@]}" shell "echo 0x1f > $DEV_DIR/hexagon_rpc_test.farf"
"${ADB[@]}" logcat -c || true

# libcdsprpc.so is intentionally NOT pushed: the device's own
# /vendor/lib64/libcdsprpc.so must be used (the SDK ship lib is a link stub).
"${ADB[@]}" shell "cd $DEV_DIR && chmod +x hexagon_rpc_test && \
  LD_LIBRARY_PATH=/vendor/lib64 \
  ADSP_LIBRARY_PATH='$DEV_DIR' DSP_LIBRARY_PATH='$DEV_DIR' \
  ./hexagon_rpc_test" 2>&1 | tee "$LOG_DIR/device_test_$STAMP.log"

"${ADB[@]}" logcat -d | grep -i "nntr_htp\|adsprpc\|fastrpc" \
  > "$LOG_DIR/device_farf_$STAMP.log" || true

echo "logs: $LOG_DIR/device_test_$STAMP.log, $LOG_DIR/device_farf_$STAMP.log"

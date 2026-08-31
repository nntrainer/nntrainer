#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Pushes the skel, the e2e harness, the packed image (<prefix>.hexw/.hexcfg)
# and the token file to the device, runs the harness, saves stdout + FARF
# logcat lines under logs/hexagon/, and pulls back any --dump-out
# result file.
#
# Usage: ./tools/hexagon/run_e2e_test.sh <prefix> [adb-serial] -- <harness args...>
#   e.g. run_e2e_test.sh /tmp/qwen3_full R3CY205ZMND -- --tokens /tmp/t.i32 --eval

set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$REPO/build_hexagon"
DEV_DIR=/data/local/tmp/nntr_htp
LOG_DIR="$REPO/logs/hexagon"
STAMP="$(date +%Y%m%d_%H%M%S)"

PREFIX="${1:?usage: run_e2e_test.sh <prefix> [adb-serial] -- <args...>}"
shift
ADB=(adb)
if [ $# -ge 1 ] && [ "$1" != "--" ]; then
  ADB=(adb -s "$1")
  shift
fi
[ "${1:-}" = "--" ] && shift

mkdir -p "$LOG_DIR"
"${ADB[@]}" shell mkdir -p "$DEV_DIR"

# push only when the device copy is missing or has a different size (the
# full image is ~600MB; re-pushing it every run would dominate the loop)
push_if_changed() {
  local src="$1" dst="$DEV_DIR/$(basename "$1")"
  local lsize dsize
  lsize=$(stat -c %s "$src")
  dsize=$("${ADB[@]}" shell "stat -c %s '$dst' 2>/dev/null || echo -1" | tr -d '\r')
  if [ "$lsize" != "$dsize" ]; then
    "${ADB[@]}" push "$src" "$dst"
  fi
}

push_if_changed "$OUT/skel/libnntr_htp_skel.so"
push_if_changed "$OUT/host/hexagon_e2e_test"
push_if_changed "$PREFIX.hexw"
push_if_changed "$PREFIX.hexcfg"

# Rewrite host paths in the harness args to device paths, remembering which
# result files to pull back.
DEV_ARGS=("$(basename "$PREFIX")")
PULL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --tokens)
      push_if_changed "$2"
      DEV_ARGS+=("$1" "$(basename "$2")"); shift 2 ;;
    --dump-out)
      PULL+=("$2")
      DEV_ARGS+=("$1" "$(basename "$2")"); shift 2 ;;
    *)
      DEV_ARGS+=("$1"); shift ;;
  esac
done

"${ADB[@]}" shell "echo 0x1f > $DEV_DIR/hexagon_e2e_test.farf"
"${ADB[@]}" logcat -c || true

# libcdsprpc.so is intentionally NOT pushed: the device's own
# /vendor/lib64/libcdsprpc.so must be used (the SDK ship lib is a link stub).
set +e
"${ADB[@]}" shell "cd $DEV_DIR && chmod +x hexagon_e2e_test && \
  LD_LIBRARY_PATH=/vendor/lib64 \
  ADSP_LIBRARY_PATH='$DEV_DIR' DSP_LIBRARY_PATH='$DEV_DIR' \
  ./hexagon_e2e_test ${DEV_ARGS[*]}" 2>&1 | tee "$LOG_DIR/e2e_$STAMP.log"
RC=${PIPESTATUS[0]}
set -e

"${ADB[@]}" logcat -d | grep -i "nntr_htp\|adsprpc\|fastrpc" \
  > "$LOG_DIR/device_farf_$STAMP.log" || true

for f in "${PULL[@]}"; do
  "${ADB[@]}" pull "$DEV_DIR/$(basename "$f")" "$f" >/dev/null && echo "pulled: $f"
done

echo "logs: $LOG_DIR/e2e_$STAMP.log, $LOG_DIR/device_farf_$STAMP.log"
exit "$RC"

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Explicit device only; installs into a separate temporary directory.
set -euo pipefail
SERIAL="${1:?usage: $0 SERIAL OUTPUT_DIR}"
OUT="${2:?usage: $0 SERIAL OUTPUT_DIR}"
case "$SERIAL" in
  *[Ll][Hh]) echo "Devices ending in LH are excluded." >&2; exit 2 ;;
esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPS="${HADAMARD_REPS:-20}"
if [[ ! "$REPS" =~ ^[1-9][0-9]*$ ]] || (( REPS > 200 )); then
  echo "HADAMARD_REPS must be 1..200" >&2
  exit 2
fi
for binary in libnntr_hvx_skel.so hadamard_check_device hadamard_hmx_check hadamard_moe_bench; do
  test -f "$HERE/build/$binary"
done
test "$(adb -s "$SERIAL" get-state)" = device
mkdir -p "$OUT"
REMOTE="/data/local/tmp/nntr_hadamard_$(date -u +%Y%m%d_%H%M%S)"
{
  date -u +%FT%TZ
  printf 'serial=%s\nremote=%s\n' "$SERIAL" "$REMOTE"
  git -C "$HERE" rev-parse HEAD
  adb -s "$SERIAL" shell getprop ro.product.model
  adb -s "$SERIAL" shell getprop ro.soc.model
  adb -s "$SERIAL" shell getprop ro.build.fingerprint
  sha256sum "$HERE/build/libnntr_hvx_skel.so" "$HERE/build/hadamard_moe_bench"
} > "$OUT/manifest.txt"
adb -s "$SERIAL" shell mkdir -p "$REMOTE"
adb -s "$SERIAL" push "$HERE/build/libnntr_hvx_skel.so" \
  "$HERE/build/hadamard_check_device" "$HERE/build/hadamard_hmx_check" \
  "$HERE/build/hadamard_moe_bench" "$REMOTE/"
# The SDK's android_aarch64 libcdsprpc.so is a link-time stub. Run with
# the phone's real vendor runtime; never push the SDK stub to a device.
adb -s "$SERIAL" shell "cp /vendor/lib64/libcdsprpc.so $REMOTE/; chmod 755 $REMOTE/hadamard_check_device $REMOTE/hadamard_hmx_check $REMOTE/hadamard_moe_bench; sync"
adb -s "$SERIAL" shell "cd $REMOTE && LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. ./hadamard_check_device" \
  > "$OUT/rotation-check.log" 2>&1
adb -s "$SERIAL" shell "cd $REMOTE && LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. ./hadamard_hmx_check" \
  > "$OUT/rotation-hmx-check.log" 2>&1
for bits in 2 4; do
  for tokens in 1 128 1024; do
    adb -s "$SERIAL" shell "cd $REMOTE && LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. ./hadamard_moe_bench $tokens $REPS $bits" \
      > "$OUT/moe-m${tokens}-w${bits}.csv" 2> "$OUT/moe-m${tokens}-w${bits}.stderr"
  done
done
for bits in 2 4; do
  adb -s "$SERIAL" shell "cd $REMOTE && LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. ./hadamard_moe_bench 1024 $REPS $bits 1" \
    > "$OUT/moe-m1024-w${bits}-norm.csv" 2> "$OUT/moe-m1024-w${bits}-norm.stderr"
done
python3 "$HERE/summarize_hadamard_bench.py" "$OUT"/moe-*.csv > "$OUT/summary.json"
printf 'Results: %s\n' "$OUT"

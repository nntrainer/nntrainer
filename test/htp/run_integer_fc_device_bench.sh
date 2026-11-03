#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Builds scalar/HVX variants and measures matching kernels on an Android DSP.
# Requires the same SDK/HexKL environment as build.sh, adb, and python3.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
BACKEND="$REPO/nntrainer/tensor/htp_backend"
: "${HEXAGON_SDK_ROOT:?Set HEXAGON_SDK_ROOT}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:?Set DEFAULT_HEXAGON_TOOLS_ROOT}"
: "${HEXKL_ROOT:?Set HEXKL_ROOT}"
export HEX_ARCH=${HEX_ARCH:-v81}
SERIAL=${1:?Usage: run_integer_fc_device_bench.sh SERIAL [OUTPUT_DIR]}
OUT=${2:-/tmp/nntr-integer-fc-bench}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
REMOTE=/data/local/tmp/nntr-integer-fc-bench
RUNNER="$HEXAGON_SDK_ROOT/libs/run_main_on_hexagon/ship"
CC="$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang"
BASE_FLAGS=${HEX_EXTRA_CFLAGS:-}
{
  git -C "$REPO" rev-parse HEAD
  "$CC" --version
  adb -s "$SERIAL" shell getprop ro.product.model
  adb -s "$SERIAL" shell getprop ro.soc.model
  adb -s "$SERIAL" shell dumpsys battery | grep -E '^  (USB powered|level|temperature):'
  printf 'arch=%s extra_cflags=%s\n' "$HEX_ARCH" "$BASE_FLAGS"
} > "$OUT/environment.txt"
for variant in scalar hvx; do
  dir="$OUT/$variant"
  mkdir -p "$dir"
  flags=()
  if [[ $variant == scalar ]]; then flags=(-DHEXKL_INTEGER_SCALAR); fi
  if [[ ${NNTR_BENCH_SKIP_BUILD:-0} != 1 ]]; then
    HEX_EXTRA_CFLAGS="$BASE_FLAGS ${flags[*]}" bash "$HERE/build.sh" > "$dir/build.log" 2>&1
    cp "$HERE/build/libnntr_hvx_skel.so" "$dir/"
  fi
  "$CC" -m"$HEX_ARCH" -mhvx -mhvx-length=128B -mhmx -G0 -O3 -fPIC -shared \
    -Wall -Werror "${flags[@]}" \
    -I "$HERE" -I "$HERE/generated" -I "$BACKEND" -I "$BACKEND/hmx" -I "$BACKEND/hvx" \
    -I "$HEXKL_ROOT/include" -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/qurt" \
    -I "$HEXAGON_SDK_ROOT/rtos/qurt/compute$HEX_ARCH/include/posix" \
    -isystem "$HEXAGON_SDK_ROOT/incs" -isystem "$HEXAGON_SDK_ROOT/incs/stddef" \
    -isystem "$HEXAGON_SDK_ROOT/ipc/fastrpc/incs" \
    "$HERE/integer_fc_device_bench.c" "$dir/libnntr_hvx_skel.so" -o "$dir/libinteger_bench.so"
  cp "$RUNNER/android_aarch64/run_main_on_hexagon" "$dir/"
  cp "$RUNNER/hexagon_toolv19_$HEX_ARCH/librun_main_on_hexagon_skel.so" "$dir/"
  sha256sum "$dir/"*.so > "$dir/binaries.sha256"
  adb -s "$SERIAL" push "$dir/." "$REMOTE/$variant/" > "$dir/push.log"
done
for round in 1 2 3; do
  order=(scalar hvx)
  if [[ $round == 2 ]]; then order=(hvx scalar); fi
  for variant in "${order[@]}"; do
    # The SDK runner has a 10-second invocation timeout. Keep each shape
    # in a separate process; the slow scalar reference also fits the limit.
    for shape in 0 1 2 3 4 5; do
      log="$variant-round$round-shape$shape.txt"
      adb -s "$SERIAL" shell "cd $REMOTE/$variant && ADSP_LIBRARY_PATH='$REMOTE/$variant;/vendor/lib/rfsa/adsp;/dsp' LD_LIBRARY_PATH=/vendor/lib64:$REMOTE/$variant ./run_main_on_hexagon 3 libinteger_bench.so $REMOTE/$log $shape" > "$OUT/$log.runner" 2>&1
      adb -s "$SERIAL" pull "$REMOTE/$log" "$OUT/$log" > /dev/null
      # The runner may return success even when the DSP main returns failure.
      if ! grep -q '^BENCH OK$' "$OUT/$log" || grep -q '^FAIL' "$OUT/$log"; then
        cat "$OUT/$log"
        exit 1
      fi
      printf '%s complete\n' "$log"
    done
  done
done
python3 - "$OUT" <<'PY'
import csv, pathlib, re, statistics, sys
root = pathlib.Path(sys.argv[1])
rows = []
hashes = {}
for path in sorted(root.glob('*-round[123]-shape[0-5].txt')):
    variant = path.name.split('-')[0]
    for line in path.read_text().splitlines():
        values = dict(re.findall(r'(\w+)=([\w.]+)', line))
        if line.startswith('TIME '):
            rows.append(dict(variant=variant, run=path.stem, **values))
        elif line.startswith('CHECK '):
            key = tuple(values[k] for k in ('M', 'K', 'N'))
            hashes.setdefault(key, set()).add(values['hash'])
assert hashes and all(len(v) == 1 for v in hashes.values()), hashes
with (root / 'samples.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
metrics = ('fp32_us', 'i32_us', 'chain_us', 'rq_us', 'rq_single_us')
with (root / 'medians.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(('variant', 'M', 'K', 'N', 'samples', *metrics))
    for key in sorted({(r['variant'], r['M'], r['K'], r['N']) for r in rows}):
        group = [r for r in rows if (r['variant'], r['M'], r['K'], r['N']) == key]
        assert len(group) == 18, (key, len(group))
        vals = [round(statistics.median(float(r[m]) for r in group), 3) for m in metrics]
        writer.writerow((*key, len(group), *vals))
        print(*key, *vals)
print('All output hashes match across scalar/HVX and all three processes.')
PY

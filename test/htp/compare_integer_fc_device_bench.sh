#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compare two saved build directories without rebuilding either baseline.
# Each contains the skel, libinteger_bench.so and SDK run_main_on_hexagon pair.
set -euo pipefail
SERIAL=${1:?Usage: compare_integer_fc_device_bench.sh SERIAL BEFORE_DIR AFTER_DIR [OUTPUT_DIR]}
BEFORE=${2:?Missing baseline binary directory}
AFTER=${3:?Missing candidate binary directory}
OUT=${4:-/tmp/nntr-integer-fc-comparison}
BASELINE_KIND=${NNTR_BENCH_BASELINE_KIND:-integer}
[[ $BASELINE_KIND == integer || $BASELINE_KIND == fp32 ]]
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
REMOTE=/data/local/tmp/nntr-integer-fc-comparison
files=(libnntr_hvx_skel.so libinteger_bench.so run_main_on_hexagon librun_main_on_hexagon_skel.so)
{
  git -C "$REPO" rev-parse HEAD
  adb -s "$SERIAL" shell getprop ro.product.model
  adb -s "$SERIAL" shell getprop ro.soc.model
  adb -s "$SERIAL" shell dumpsys battery | grep -E '^  (USB powered|level|temperature):'
  printf 'baseline=%s\ncandidate=%s\n' "$BEFORE" "$AFTER"
  printf 'baseline_kind=%s\n' "$BASELINE_KIND"
  printf 'Two warmup rounds; six samples/process; three calls/sample; three processes/shape.\n'
} > "$OUT/environment.txt"
: > "$OUT/binaries.sha256"
for variant in before after; do
  src=$BEFORE
  if [[ $variant == after ]]; then src=$AFTER; fi
  mkdir -p "$OUT/$variant"
  for file in "${files[@]}"; do
    cp "$src/$file" "$OUT/$variant/$file"
  done
  (cd "$OUT" && sha256sum "$variant/"*) >> "$OUT/binaries.sha256"
  adb -s "$SERIAL" push "$OUT/$variant/." "$REMOTE/$variant/" > "$OUT/$variant-push.log"
done
for round in 1 2 3; do
  order=(before after)
  if [[ $round == 2 ]]; then order=(after before); fi
  for shape in 0 1 2 3 4 5; do
    for variant in "${order[@]}"; do
      # The SDK runner imposes a 10-second timeout: one shape per process.
      log="$variant-round$round-shape$shape.txt"
      adb -s "$SERIAL" shell "cd $REMOTE/$variant && ADSP_LIBRARY_PATH='$REMOTE/$variant;/vendor/lib/rfsa/adsp;/dsp' LD_LIBRARY_PATH=/vendor/lib64:$REMOTE/$variant ./run_main_on_hexagon 3 libinteger_bench.so $REMOTE/$log $shape" > "$OUT/$log.runner" 2>&1
      adb -s "$SERIAL" pull "$REMOTE/$log" "$OUT/$log" > /dev/null
      if ! grep -q '^BENCH OK$' "$OUT/$log" || grep -q '^FAIL' "$OUT/$log"; then
        cat "$OUT/$log"
        exit 1
      fi
      printf '%s complete\n' "$log"
    done
  done
done
python3 - "$OUT" "$BASELINE_KIND" <<'PY'
import csv, pathlib, re, statistics, sys
root = pathlib.Path(sys.argv[1])
fp32_baseline = sys.argv[2] == 'fp32'
rows, profiles, hashes = [], [], {}
with (root / 'runs.txt').open('w') as combined:
    for path in sorted(root.glob('*-round[123]-shape[0-5].txt')):
        variant = path.name.split('-')[0]
        combined.write(f'--- {path.name} ---\n{path.read_text()}\n')
        times = []
        for line in path.read_text().splitlines():
            v = dict(re.findall(r'(\w+)=([\w.]+)', line))
            if line.startswith('TIME '):
                times.append(dict(variant=variant, run=path.stem, **v))
            elif line.startswith('PROFILE '):
                profiles.append(dict(variant=variant, run=path.stem,
                                     **{k: times[0][k] for k in ('M', 'K', 'N')}, **v))
            elif line.startswith('CHECK '):
                key = ((variant,) if fp32_baseline else ()) + tuple(v[k] for k in ('M', 'K', 'N'))
                hashes.setdefault(key, []).append(v['hash'])
        assert len(times) == 6, (path, len(times))
        rows.extend(times)
assert len(hashes) == (12 if fp32_baseline else 6), hashes
assert all(len(v) == (3 if fp32_baseline else 6) and len(set(v)) == 1 for v in hashes.values()), hashes
for name, values in (('samples', rows), ('profiles', profiles)):
    if values:
        with (root / f'{name}.csv').open('w') as f:
            fields = list(dict.fromkeys(k for r in values for k in r))
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(values)
metrics = ('fp32_us', 'i32_us', 'chain_us', 'rq_us', 'rq_single_us')
with (root / 'medians.csv').open('w') as f:
    w = csv.writer(f)
    w.writerow(('variant', 'M', 'K', 'N', 'samples', *metrics))
    for key in sorted({(r['variant'], r['M'], r['K'], r['N']) for r in rows}):
        group = [r for r in rows if (r['variant'], r['M'], r['K'], r['N']) == key]
        assert len(group) == 18, (key, len(group))
        vals = [round(statistics.median(float(r[m]) for r in group), 3)
                if m in group[0] else '' for m in metrics]
        w.writerow((*key, len(group), *vals))
        print(*key, *vals)
print('All 36 processes passed. Output hashes are stable for every shape.'
      if fp32_baseline else 'All 36 processes passed. INT32 hashes match before/after for all shapes.')
PY

# Measurement 25 (H1): cross-op weight prefetch A/B, the W8 weight-stream ceiling, and the 64-row chunk

Branch `hvx/25-decode-prefetch` @ `7d318718` (the last commit that changes DSP bytes; the later `MM16_R`
pin commit adds a static check only — all 11 DSP object md5s are identical — and the rest is tools/docs) — estimated device time: **30 min** (8 runs at 512 ≈ 10 min, four
1024 / 4096 runs ≈ 10 min, the 4096 image push ≈ 3 min if absent, md5 checks; the H2 kernel sweep is a
separate, later handoff).

## Why
Decode is the 596 MB weight stream (#24 closed the host path: the logits return is < 0.1 ms inside a
34 ms step). Today every tiled matmul kicks its own first DMA chunk only after the op starts — with 6
workers the decode W8A8 ops are 1–2 chunks per worker, so the first chunk is fully exposed and the DDR
idles through ROPE / ATTN / ADD / RMSNORM / SILU_MUL. This branch keeps one DMA queue per worker for the
session and lets each matmul kick the *next* matmul's chunk 0 into its idle half-slab (plan
`docs/plans/25-decode-prefetch.md`, ledger ①). The simulator does not charge DDR, so only silicon can
say what that buys; the same run measures the **weight-stream ceiling** (a skel that streams every
weight byte but multiplies nothing) that replaces the provisional `≥ 60 tok/s` goal and gates #51, and
the 64-row chunk of ledger ⑨. Decisions that hang on it: the prefetch default (on / compiled-in-but-off),
the chunk default, the stage-1 W8A8 goal cell, and whether H2 (`MM_TB` / `MM16_TB`; `MM16_R` is pinned
to 4 until the W8A16 epilogue is generalised) runs on B or D.

## Artifacts (built in the container, SDK 6.4.0.2 / hexagon-clang 19.0.04 `toolv19`, `HEX_ARCH=v79`, @ `7d318718`)

| file | md5 (size) | built with (`HEX_EXTRA_CFLAGS=`) | answers |
|---|---|---|---|
| build_hexagon/skel/libnntr_htp_skel.A.so | `e25597fe0dc36adfcb0247c63f384b7c` (51,112 B) | `-DHTP_PROF_FARF -DHTP_MM_NO_PREFETCH` | **A control**: environment check, the A/B baseline, the device non-matmul share |
| build_hexagon/skel/libnntr_htp_skel.B.so | `208b25a23c4b308d5d8eb0eec8a4d103` (55,208 B) | `-DHTP_PROF_FARF` | **B prefetch**: G1 |
| build_hexagon/skel/libnntr_htp_skel.C.so | `853af711a5772852b440ef45a22cfaf3` (51,112 B) | `-DHTP_PROF_FARF -DHTP_MM_STREAM_ONLY` | **C ceiling**: G1' (GB/s, ceiling tok/s at 512 / 1024 / 4096). Outputs are garbage by construction |
| build_hexagon/skel/libnntr_htp_skel.D.so | `2da2a4f9217ae904bf11c9bd99825f9b` (55,208 B) | `-DHTP_PROF_FARF -DHTP_MM_CHUNK_ROWS=64u` | **D chunk**: ledger ⑨ on top of B |
| build_hexagon/skel/libnntr_htp_skel.so (default, not run) | `7dc42f5981cf000393d2d23c3c181314` (55,208 B) — **read-back 2026-09-18: this row was a `-DHTP_PROF_FARF` build** (its size equals B's; a clean default build of the same tree is **50,984 B**, md5 `811182a82a0f39359981b0945b6c7c4c` in the container, and the link is not byte-reproducible so only the size and the object md5s are the proof — see the Simulator record below) | (none) | the shipping build = B without the FARF line; prefetch on, `HTP_PROF_FARF` off |
| build_hexagon/host/hexagon_e2e_test | `88996fee9a57113cf47793d877e4ae19` | `./tools/hexagon/build_host_test.sh` (host sources unchanged since #24) | |
| build_hexagon/host/hexagon_rpc_test | `dd05aaab4cbe916c2efabfb9b6757678` | same | |
| /tmp/qwen3_full.hexw / .hexcfg | `5abf61bef086368559423daa3bea9a99` / `5926be5703f85531c9c46c1978c25287` | packer unchanged since P4 (#23 / #35 tables) | |
| /tmp/qwen3_full4k.hexw / .hexcfg | `e5c92fad696fb35405918941d1062be3` / `b0d8aca9a6c7626b0af8dc81e1de59eb` | `--max-seq 4224` | |
| /tmp/t512_23.i32 | `10bb428f92c792aa7733339f8e6b0da1` | `make_tokens.py --limit 512` on `eval_23.txt` (#35 §1) | accuracy anchor: #35 B1 on this unit **41.4947 / 162** |
| /tmp/t1024.i32, /tmp/t4096.i32 | `86e44c0633ebab556aafe3aa68d80c63` / `29684fbd372495912951cb3cc0d25e20` | #24 table | goal rows |

All four skels carry `-DHTP_PROF_FARF` so its cost (one FARF per call) cancels in every ratio. **If the
Mac binaries did not reach the workstation, rebuild them there** (SDK 6.4.0.1 gives the same `toolv19`;
#35 measured a uniform +192 B and a different md5 per skel from the runtime/header delta — write the
md5 you get, keep the size):

```bash
git fetch && git checkout hvx/25-decode-prefetch      # do not rebase; commit the result on this branch
source <SDK>/setup_sdk_env.source; export ANDROID_NDK=<ndk path>
for v in "A:-DHTP_PROF_FARF -DHTP_MM_NO_PREFETCH" "B:-DHTP_PROF_FARF" "C:-DHTP_PROF_FARF -DHTP_MM_STREAM_ONLY" "D:-DHTP_PROF_FARF -DHTP_MM_CHUNK_ROWS=64u"; do
  HEX_EXTRA_CFLAGS="${v#*:}" ./tools/hexagon/build_skel.sh && cp build_hexagon/skel/libnntr_htp_skel.so build_hexagon/skel/libnntr_htp_skel.${v%%:*}.so
done
./tools/hexagon/build_host_test.sh
md5sum build_hexagon/skel/libnntr_htp_skel.{A,B,C,D}.so build_hexagon/host/hexagon_e2e_test
```

## Steps (workstation, S25 Ultra on USB; ≈ 30 min)

Name the unit: `adb devices` → paste the serial in the Results header. Pass rules are DSP Mcyc ratios
within this session, so the unit only has to be the same for all rows (HEXAGON.md §7 rule 9). Insert
the serial after the image path only if more than one device is attached.

```bash
adb devices
run() { cp build_hexagon/skel/libnntr_htp_skel.$1.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so; shift; ./tools/hexagon/run_e2e_test.sh "$@"; }
# 0. transport sanity on A (RPC_TEST PASS), also warms the DSP
cp build_hexagon/skel/libnntr_htp_skel.A.so build_hexagon/skel/libnntr_htp_skel.so && ./tools/hexagon/run_device_test.sh
# 1. pass 1 at 512: A, B, C, D — speed, then --eval (C has no --eval: garbage outputs by construction)
for v in A B C D; do run $v /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --chunk 128 --steps 64; [ $v != C ] && run $v /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --eval; done
# 2. pass 2, reversed: D, C, B, A — the pass the tables read (rule 9(d): pass 1's first run is cold)
for v in D C B A; do run $v /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --chunk 128 --steps 64; [ $v != C ] && run $v /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --eval; done
# 3. goal rows: B at 1024 / 4096 (speed + --eval), C at 1024 / 4096 (speed)
run B /tmp/qwen3_full   -- --tokens /tmp/t1024.i32 --chunk 128 --steps 64
run B /tmp/qwen3_full   -- --tokens /tmp/t1024.i32 --eval
run B /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64
run B /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --eval
run C /tmp/qwen3_full   -- --tokens /tmp/t1024.i32 --chunk 128 --steps 64
run C /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64
adb shell md5sum /data/local/tmp/nntr_htp/libnntr_htp_skel.so /data/local/tmp/nntr_htp/hexagon_e2e_test   # the C skel, last pushed
# 4. the per-kind DSP split of every run (FARF lines are in the device_farf log of each run)
python3 tools/hexagon/summ_farf_prof.py logs/hexagon/device_farf_*.log
```

Expected lines per run: `E2E init ok weights=598623744 kv=234881024 act=3932160 n_ops=451` (the 4096
image: `weights=599180800`), 4 × `E2E step … n=128 …`, 63 × `E2E step … n=1 pcycles=<c> us=<t> top1=<id>`,
`E2E gen <64 ids>`, `E2E decode steps=63 median_us=<u> median_pcycles=<c> pcycles_per_us=<r>`, `E2E wall_ms`;
`--eval`: `E2E ppl <x> steps 511 top1 <n> wall_ms <w>` then `E2E decode steps=511 …`. In
`logs/hexagon/device_farf_<stamp>.log` (every variant): one `nntr_htp: prof n=<n> pos=<p> ops=451 kcyc=<k>
mm8=<k> mm16=<k> lg=<k> attn=<k> rest=<k>` per call (kilo-pcycles); for **C** additionally
`nntr_htp: stream bytes/step=595984384 (STREAM_ONLY, outputs are garbage)` after `init ok`. C's `top1`
ids and `E2E gen` are meaningless and may repeat one id — that is not a failure. A hang or DSP restart:
note the last `E2E step` line and the FARF tail in the Notes, `adb reboot`, continue with the next command
(risk 3 of the plan: a descriptor pending across the worker barrier is the one new runtime pattern; a stall
shows as a > 2× step in the `pcycles` column of the first prefetched op, i.e. right after step 0 of a
B / D run).

Reading the numbers (HEXAGON.md §8.2, #24): prefill tok/s = 512 ÷ (sum of `us` of the 4 `n=128` steps
/ 1e6); decode median host ms = `median_us` / 1000; decode tok/s = 1000 ÷ that; **DSP Mcyc/step =
`median_pcycles` / 1e6** (the column every rule reads); `pcycles_per_us` = the run's own clock. For C:
GB/s = 595,984,384 ÷ (`median_pcycles` ÷ `pcycles_per_us`) ÷ 1000; ceiling tok/s = 1000 ÷ (`median_us` / 1000).

## Results — unit serial: `R3CY10WM83Y` (S25 Ultra, v79), measured 2026-09-18, pass 2 rows are the record

Reference cells for this unit (`R3CY10WM83Y`, v79 shipping skel, #35 B1 = `hvx_impl` today): 512 →
198.7 prefill tok/s / 32.391 ms / 30.87 tok/s / **61.200 Mcyc** at 1905–1915 `pcycles_per_us`; 1024 →
131.1 / 40.227 / 24.86 / **77.781**; 4096 → 32.71 / 89.103 / 11.22 / **186.753** (cooled re-run; the 4096 cell
compares only within this session, rule 9(e)). On `R3CY205ZMND` the v79 references are 60.1 / 76.5 /
177.6 Mcyc. **Environment check:** A (pass 2) must land within ±5 % of the unit's 512 Mcyc reference
(61.2 or 60.1); outside that band the session is void (stale artifact or wrong unit), not a result.

### Speed, 512 tokens (`--chunk 128 --steps 64`, `t512_23.i32`, `qwen3_full`)

Skels were **rebuilt on the workstation** (SDK 6.4.0.2, `HEXAGON_Tools/19.0.04`, `HEX_ARCH=v79`) because the
container binaries were not on this machine, so the md5s below replace the artifact table's; the sizes are
the doc's + 160 B (the doc expected #35's + 192 B), A/C 51,272 B and B/D 55,368 B:
A `de8050b87810a85bb4341fee9cd940b5`, B `98b07cafa2f273bd7a587159efe759de`,
C `6520a381bcf8f71c702d18a70da1c3ce`, D `8effb8755c69d9c261fa7389d614d0f8`,
default (not run) `f032508a80e0ac8a09dcdf26bf361dbb`, `hexagon_e2e_test` `743ba32c935ada2571b97b9003de3163`,
`hexagon_rpc_test` `60a7098b27a8868ad820b139772a9ea1`. Every model / token artifact md5 matched the table
exactly. Step 0 `run_device_test.sh` on A: `RPC_TEST full-logits pattern ok`, **`RPC_TEST PASS`**.

| variant | pass | skel md5 (from `run()`) | prefill tok/s | decode median host ms | decode tok/s | **DSP Mcyc/step** | `pcycles_per_us` | FARF median Mcyc: mm8 / mm16 / lg / attn / rest | log stamp |
|---|---|---|---|---|---|---|---|---|---|
| A control | 1 | `de8050b8…` | 179.5 | 34.990 | 28.58 | 66.968 | 1913.9 | 26.273 / 8.554 / 9.908 / 18.973 / 3.150 | 132629 |
| B prefetch | 1 | `98b07caf…` | 185.0 | 29.694 | 33.68 | 56.284 | 1895.5 | 16.200 / 8.611 / 8.963 / 18.499 / 3.653 | 132702 |
| C ceiling | 1 | `6520a381…` | 340.8 | 30.739 | 32.53 | 58.214 | 1893.8 | 17.954 / 7.823 / 9.076 / 18.801 / 4.478 | 132732 |
| D chunk 64 | 1 | `8effb875…` | 179.1 | 31.705 | 31.54 | 59.711 | 1883.3 | 20.430 / 7.948 / 9.052 / 18.617 / 3.550 | 132738 |
| **D chunk 64** | 2 | `8effb875…` | 181.4 | 29.329 | 34.10 | **59.993** | 2045.5 | 20.662 / 7.914 / 9.012 / 18.766 / 3.603 | 132820 |
| **C ceiling** | 2 | `6520a381…` | 370.3 | 30.399 | 32.90 | **52.997** | 1743.4 | 16.583 / 7.233 / 8.345 / 16.651 / 4.130 | 132851 |
| **B prefetch** | 2 | `98b07caf…` | 191.1 | 29.715 | 33.65 | **54.575** | 1836.6 | 16.052 / 8.588 / 8.955 / 17.260 / 3.647 | 132857 |
| **A control** | 2 | `de8050b8…` | 188.5 | 32.637 | 30.64 | **61.735** | 1891.6 | 24.347 / 7.874 / 9.066 / 17.312 / 3.080 | 132929 |

The clock moved between runs (1743–2046 `pcycles_per_us`), which is why the host-ms column disagrees with the
Mcyc column on D (lowest host ms of pass 2, highest Mcyc of B/C/D) — rule 9 reads Mcyc only.

### Goal rows (B and C at 1024 / 4096; references are #35 B1 on `R3CY10WM83Y`)

| variant | ctx | image | prefill tok/s (ref) | decode median host ms (ref) | decode tok/s (ref) | DSP Mcyc/step (ref) | `pcycles_per_us` | C only: GB/s over the step, ceiling tok/s | log stamp |
|---|---|---|---|---|---|---|---|---|---|
| B prefetch | 1024 | qwen3_full | 123.7 (131.1) | 38.002 (40.227) | 26.31 (24.86) | **73.744** (77.781) | 1940.5 | — | 133021 |
| B prefetch | 4096 | qwen3_full4k | 30.84 (32.71) | 90.280 (89.103) | 11.08 (11.22) | **184.603** (186.753) | 2044.8 | — | 133125 |
| C ceiling | 512 | qwen3_full | 370.3 | 30.399 | 32.90 | 52.997 | 1743.4 | **19.61 GB/s, 32.90 tok/s** | 132851 |
| C ceiling | 1024 | qwen3_full | 186.6 | 38.156 | 26.21 | 73.809 | 1934.4 | 15.62 GB/s, 26.21 tok/s | 133817 |
| C ceiling | 4096 | qwen3_full4k | 33.60 | 91.791 | 10.89 | 186.884 | 2036.0 | 6.49 GB/s, 10.89 tok/s | 133828 |

**The step-level GB/s of C is not the DDR ceiling**, because the stream-only skel still runs ATTN and the
eltwise ops. The stream phase of C is `mm8`+`mm16`+`lg` — `HTP_MM_STREAM_ONLY` drops `mm_tiles` inside
`mm_worker_vtcm`, which serves `MATMUL_LOGITS` as well as `MATMUL_W8A8`, and the printed
`stream bytes/step=595,984,384` is exactly 28 × 15,728,640 B of layer weights + 151,936 × 1024 B of lm_head, so
`lg` is DMA wait as much as `mm8` is. (An earlier version of this section took `mm8`+`mm16` alone as "the
matmul phase" and derived ≈ 74 tok/s / 43.6–46.0 GB/s; that left out the lm_head's 26 % of the bytes.
Corrected at the read-back, supervisor comment of 2026-09-18.)

| ctx | `mm8` | `mm16` | `lg` | stream Mcyc | `pcycles_per_us` | stream ms | GB/s | **ceiling tok/s** | rest of the step (`attn` + `rest`) |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 16.583 | 7.233 | 8.345 | **32.16** | 1743.4 | 18.45 | **32.3** | **54.2** | 16.65 + 4.13 = 20.8 Mcyc |
| 1024 | 25.9 (`mm8`+`mm16`) | | ≈ 9.6 (derived) | ≈ 35.5 | 1934.4 | ≈ 18.3 | ≈ 32.5 | **≈ 54.5** | 34.2 + ≈ 4.1 |
| 4096 | 26.4 (`mm8`+`mm16`) | | ≈ 9.8 (derived) | ≈ 36.2 | 2036.0 | ≈ 17.8 | ≈ 33.6 | **≈ 56.3** | 146.6 + ≈ 4.1 |

The 512 row is read from the pass-2 FARF median (stamp 132851). The 1024 / 4096 `lg` cells are **derived, not
read**: the `device_farf_133817.log` / `_133828.log` files live on the workstation (`logs/hexagon/` is
git-ignored and the Mac never had them), so `lg` = step Mcyc − `mm8`+`mm16` − `attn` − `rest` with `rest`
taken as C's 512 value (4.13 Mcyc): 73.809 − 25.9 − 34.2 − 4.13 and 186.884 − 26.4 − 146.6 − 4.13. Even with
`rest` = 0 the stream phase is ≤ 39.6 / 40.3 Mcyc, i.e. the ceiling is ≥ 48.8 / 50.5 tok/s at 1024 / 4096;
with the derived `lg` it is 54–56 tok/s, **flat with context**, as expected for a weight stream whose bytes do
not depend on the position. Running `python3 tools/hexagon/summ_farf_prof.py logs/hexagon/device_farf_1338{17,28}.log`
on the workstation replaces the two derived cells with read ones (expected `lg` ≈ 9–10 Mcyc).

Per path at 512 (from C's pass-2 split): the tiled q/k/v/o/gate/up DMA moves 352.3 MB in `mm8` 9.51 ms =
**37.0 GB/s**; the lm_head 155.6 MB in `lg` 4.79 ms = **32.5 GB/s**; `down` 88.1 MB in `mm16` 4.15 ms =
**21.2 GB/s** (the C build DMAs the row-major rows in slab-sized pieces without the up → q handover; the
shipping kernel reads them straight from DDR in 8.59 Mcyc, issue #59). B's own stream phase at 512 is
`mm8`+`mm16`+`lg` = 16.052 + 8.588 + 8.955 = 33.6 Mcyc, within 4 % of C's 32.2 — **the matmul compute is
already fully hidden behind the stream** — and what remains above the ceiling is ATTN + eltwise
(`attn` + `rest` = 20.9 Mcyc at 512, ≈ 38.3 at 1024 and ≈ 150.7 at 4096 with C's `rest`; ATTN alone is 32 / 46 / 79 % of the step,
issue #58).

### Accuracy (`--eval`, both passes; C is `n/a (stream only)`)

| variant | pass | ctx | token file | `--eval` PPL (ref) | top-1 (ref) | `E2E gen` identical to A pass 2? (`diff <(grep -o 'top1=[0-9]*' a.log) <(… b.log)`) |
|---|---|---|---|---|---|---|
| A control | 1 | 512 | t512_23.i32 | **41.4947** (41.4947) | **162** (162) | yes |
| B prefetch | 1 | 512 | t512_23.i32 | **41.4947** (41.4947) | **162** (162) | yes |
| D chunk 64 | 1 | 512 | t512_23.i32 | **41.4947** (41.4947) | **162** (162) | yes |
| D chunk 64 | 2 | 512 | t512_23.i32 | **41.4947** (41.4947) | **162** (162) | yes |
| B prefetch | 2 | 512 | t512_23.i32 | **41.4947** (41.4947) | **162** (162) | yes |
| A control | 2 | 512 | t512_23.i32 | **41.4947** (41.4947) | **162** (162) | — (the reference) |
| B prefetch | — | 1024 | t1024.i32 | **5.8595** (5.8595) | **692** (692) | equal to #35 B1 |
| B prefetch | — | 4096 | t4096.i32 | **1.5623** (1.5623) | **3759** (3759) | equal to #35 B1 |
| C ceiling | — | — | — | n/a (stream only) | n/a | n/a |

All six 512 speed runs (A / B / D, both passes) have a **byte-identical** 63-step `top1=` sequence
(md5 `2fb368f6c82386c4da2481090c37f996`) and `E2E gen` line (md5 `14ea2f315231c20614bb8f74356e4d54`).

`adb shell md5sum` after the last run: `6520a381bcf8f71c702d18a70da1c3ce` (the C skel, as built) /
`743ba32c935ada2571b97b9003de3163` (hexagon_e2e_test).

### Pass rules (plan §1; fill the verdict column)

| rule | reads | verdict |
|---|---|---|
| env | A pass 2 within ±5 % of 61.2 Mcyc (`R3CY10WM83Y`) or 60.1 (`R3CY205ZMND`) at 512 | **PASS** — 61.735 Mcyc = +0.9 % of 61.2 on `R3CY10WM83Y` |
| G1 | `Mcyc(B) ≤ 0.90 × Mcyc(A)` at 512, pass 2 → prefetch **on** by default; 0.90–0.97 partial (adopt, record); > 0.97 no lever (keep the graph-lifetime queue, prefetch compiled in but off) | **PASS** — 54.575 / 61.735 = **0.884** (−11.6 %); `mm8` 24.347 → 16.052 Mcyc (−34 %). Prefetch **on** by default |
| G1' | C: GB/s and ceiling tok/s at 512 / 1024 / 4096 → the stage-1 W8A8 goal cell in HEXAGON_BENCHMARK.md and the #51 input | **MEASURED** — stream phase `mm8`+`mm16`+`lg` = **32.2 Mcyc = 18.5 ms at 1743 MHz = 32.3 GB/s, ceiling 54 tok/s at 512**; ≈ 54.5 / ≈ 56.3 at 1024 / 4096 (derived `lg`, see the stream table), flat with context. Per path: tiled DMA 37.0 GB/s, lm_head 32.5, row-major `down` 21.2. The step-level C figures (19.61 / 15.62 / 6.49 GB/s = 32.90 / 26.21 / 10.89 tok/s) still run ATTN + eltwise. The provisional `≥ 60 tok/s` stage-1 goal is **above the ceiling** and is withdrawn in HEXAGON_BENCHMARK.md (replacement ≥ 40 proposed, needs-user); the earlier "≈ 74 tok/s" reading counted `mm8`+`mm16` only |
| D rule | D ≥ 2 % below B at 512 decode (Mcyc, pass 2) **and** D prefill tok/s ≥ B → `HTP_MM_CHUNK_ROWS 64` becomes the default; else keep max-fit | **FAIL → keep max-fit** — D is **+9.9 %** above B (59.993 vs 54.575 Mcyc) and its prefill is lower (181.4 vs 191.1 tok/s); D's `mm8` 20.662 vs B 16.052 Mcyc |
| G3 | A, B, D: PPL / top-1 equal to the digit across variants and passes, and `E2E gen` byte-identical to A pass 2 | **PASS** — 41.4947 / 162 in all six 512 `--eval` runs; 1024 5.8595 / 692 and 4096 1.5623 / 3759 equal the #35 B1 cells; `top1=` and `E2E gen` md5-identical across A / B / D, both passes |
| no hang / SSR / FARF fatal | all 14 runs | **PASS** — 14/14 completed, no DSP restart (the only `Reset` line is the benign `Reset loading vote for libnntr_htp_skel.so`); the `E` lines are the usual fastrpc `open_shell` / `log_config` permission noise present since #23. No > 2× step after the prefill chunks in any B / D run |

## Notes from the run (2026-09-18, all 14 runs in one session, ≈ 15 min of device time)
- phone / serial / SDK: S25 Ultra `R3CY10WM83Y` (the P3 / P4 / #24 unit), workstation SDK **6.4.0.2**,
  `HEXAGON_Tools/19.0.04`, NDK r26d. Both images were already on the device (md5 match, no 600 MB push).
- **rebuilt on the workstation: yes** — the md5s in the Results header replace the artifact table; sizes are
  + 160 B vs the container build (the doc expected #35's + 192 B), and each variant pair A/C and B/D has the
  expected identical size.
- warm-up: pass 1 A is **+8.5 %** above pass 2 A (66.968 vs 61.735 Mcyc) — larger than the 3–5 % of rule 9(d),
  so the reversed double pass mattered: on pass-1 numbers alone B/A would read 0.84 and D/B 1.06.
- nothing odd: no thermal throttle visible (SoC zone0 29.3 °C at start), no stale push (`run_e2e_test.sh`
  compares md5), no FARF fatal, no SSR. The clock wandered 1743–2046 `pcycles_per_us` across runs, which is
  the reason every verdict above is read in Mcyc.
- `summ_farf_prof.py`, decode median, pass 2: **A** `mm8=24.347 mm16=7.874 lg=9.066 attn=17.312 rest=3.080`
  → non-matmul (attn+rest) 20.392 Mcyc = **33.0 %** of the loop; **C** `mm8=16.583 mm16=7.233 lg=8.345
  attn=16.651 rest=4.130` → its matmul phase is 23.8 of 53.0 Mcyc (**45 %**), the rest is ATTN / LOGITS /
  eltwise that the stream-only skel still executes. **B** lands at `mm8=16.052` — 0.5 Mcyc below C's DMA-only
  matmul wait, i.e. the W8A8 compute is fully covered by the stream.
- context scaling: at 1024 and 4096 the prefetch has nothing left to win (B 73.744 vs C 73.809 Mcyc at 1024;
  184.603 vs 186.884 at 4096) because ATTN is 48.9 % / 79.3 % of the decode loop there. The next decode lever
  above 512 is ATTN, not the weight stream.
- vs the #35 B1 reference on this unit, B is −5.2 % Mcyc at 1024 and −1.2 % at 4096, but its prefill tok/s
  reads 3–6 % below the reference cells (188–191 vs 198.7 at 512) — host time at a lower clock this session;
  in Mcyc B's prefill sum is 1.3 % *below* A's (5463.9 vs 5537.5), so this is a clock artefact, not a
  prefill regression.

Hand back: `git add docs/measurements/25-decode-prefetch.md && git commit -s -m "[docs] Fill the #25 H1 measurement (<unit>, SDK <ver>)" && git push`,
then `gh issue edit 25 -R dlwlzzero/nntrainer --remove-label state:needs-measurement --add-label state:measured`.

## Simulator record (v79, container SDK 6.4.0.2, 2026-09-18, `logs/hexagon/sim_v79_25_full.log`, DSP bytes of `7d318718`)
`profile acc` PASS, `profile_prefill_acc STAT max_abs=0.0659682 max_rel=92.7791` — bit-identical to the #35 v79
record; `workers=6`, `total_pcycles` 6,158,964 (#35 6,174,196, −0.2 %), `MATMUL_W8A8` per_call 215,153 (#35
216,272), `MATMUL_W8A16` 1,261,864 (=), `ATTN` 174,287 (174,095), `MATMUL_LOGITS` 113,504, `barrier_empty_x1000`
8,319,592 (=) — the per-kind numbers are a relative signal only (no DDR model: the prefetch cannot show here).
13/13 PASS with every STAT equal to the #35 v79 record (`quant_generic 0/65536`, `quant16_generic 184/25600`, every
`matmul_w8a8_*` / `matmul_dma_ref_*` / `logits` `0/0`, `graph_prefill 0.0218946/6.88818`, `graph_decode
0.0197323/23.2374`, `graph_prefill_2workers 0.0218946/6.88818`), plus the new `SIM_TEST matmul_dma prefetch hits=6
workers=6`. Rung 1: `LOWER_TEST PASS`, `W8CX_BIN_TEST PASS`, oplist header `PASS`. Rung 4: default skel
`7dc42f5981cf000393d2d23c3c181314` (55,208 B) + both harnesses build; the four variants above.

**Read-back (2026-09-18, container SDK 6.4.0.2, v79, tree = the PR head).** All 11 DSP objects of the PR
head (`executor.c`, `worker_pool.c`, `htp_graph.c`, `dma-queue.c`, the six `ops/*.c`, the generated
`nntr_htp_skel.c`, each compiled with `-c` at the same path) have md5s identical to `7d318718`'s — no DSP byte
changed after the tree the simulator record and the device run used (the `MM16_R` pin is a static check
only). The default build's `hvx-matmul.c.o` differs from a `-DHTP_MM_NO_PREFETCH` build and its
`htp_graph.c.o` from a `-DHTP_PROF_FARF` build, i.e. **the shipping build has the prefetch on and the FARF
line off**. Rebuilt in the container from the PR head: A `51,112 B`, B `55,208 B` (the sizes of the artifact
table), default `50,984 B` (md5 `811182a8…`; the artifact table's 55,208 B "default" was a FARF-on build).
Rung 1 on the PR head: `LOWER_TEST PASS`, `W8CX_BIN_TEST PASS`, oplist header `PASS`; `hexagon_rpc_test` /
`hexagon_e2e_test` rebuilt with the artifact table's md5s (`dd05aaab…` / `88996fee…`, host sources unchanged).

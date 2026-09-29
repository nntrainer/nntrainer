# Measurement 58: decode attention split over the workers with GQA-fused K^T / V streams — A/B/C/D

Branch `hvx/58-decode-attn-gqa` @ `d26df395` (the last commit that changes DSP bytes; the handoff commit on
top is this file) — estimated device time: **≈ 17 min core session** (steps 1–4 below) **+ ≈ 9 min tail**
(step 5: the 4096 `--eval` of B and the 4096 speed run of C). Run the tail if the sitting allows; the
core session alone decides G1 and the `l2fetch` default, the tail decides G2 at 4096 and G5.

## Why
The #25 FARF split put ATTN at 17.3 / 34.2 / 146.6 Mcyc per decode step at 512 / 1024 / 4096 on this unit
(32 / 46 / 79 % of the step): the m=1 kernel streamed K^T and V once per q head and gave two of the six
workers two kv heads each. This branch (plan `docs/plans/58-decode-attn-gqa.md`) cuts each kv head's
context into position blocks so the 8 heads balance over the 6 workers (24 jobs, 4 per worker), serves both
q heads of a GQA pair from one K^T / V stream, and merges the block partials in a second pass; variant C
adds an `l2fetch` of the next K^T / V lane block. The simulator charges no DDR latency, so only silicon
can say (G1) whether `attn` halves at 4096 and drops under 9 Mcyc at 512, (G2) whether the merge's
re-association stays inside the ± 0.3 % PPL band, and (G5 / the C-vs-B read) whether the fused loop is
still latency-bound (C ≪ B → the prefetch is the next lever) or already bandwidth-bound (C ≈ B → fewer KV
bytes is). Decisions that hang on it: the `HTP_ATTN_L2FETCH` default, whether the kernel merges as is,
and the input of the next decode-attention issue (DMA-to-VTCM or KV bytes).

## Artifacts (built in the container, SDK 6.4.0.2 / hexagon-clang 19.0.04 `toolv19`, `HEX_ARCH=v79`)

| file | md5 (size) | built from / with (`HEX_EXTRA_CFLAGS=`) | answers |
|---|---|---|---|
| build_hexagon/skel/libnntr_htp_skel.A.so | `4a19acb5ff2ec2ea8163164ea7d51908` (55,208 B) | `hvx_impl` @ `2c880173` (clean worktree), `-DHTP_PROF_FARF` | **A control**: environment check, the A/B baseline of every ratio |
| build_hexagon/skel/libnntr_htp_skel.B.so | `f1887aa9acc6a309f9454eb57de56cb2` (59,432 B) | this branch @ `d26df395`, `-DHTP_PROF_FARF` | **B**: the split + GQA fusion, `HTP_ATTN_NB` auto, `HTP_ATTN_L2FETCH 0` — G1, G2 |
| build_hexagon/skel/libnntr_htp_skel.C.so | `d41a8c35270258cfb77818861fbcda7d` (59,464 B) | same, `-DHTP_PROF_FARF -DHTP_ATTN_L2FETCH=1` | **C**: B + the K^T / V `l2fetch` — latency share, G5, the `l2fetch` default |
| build_hexagon/skel/libnntr_htp_skel.D.so | `468ab6142ea32955b3d753ece8cfe1d6` (59,432 B) | same, `-DHTP_PROF_FARF -DHTP_ATTN_NB=1` | **D**: GQA fusion without the split — the balance share (informational, 512 only) |
| build_hexagon/skel/libnntr_htp_skel.so (default, not run) | `e16d86acba59668ee68b0504dee751f3` (59,304 B) | same, no flags | the shipping build = B without the FARF line |
| build_hexagon/host/hexagon_e2e_test | `88996fee9a57113cf47793d877e4ae19` | `./tools/hexagon/build_host_test.sh` (host sources unchanged since #24) | |
| build_hexagon/host/hexagon_rpc_test | `dd05aaab4cbe916c2efabfb9b6757678` | same | |
| /tmp/qwen3_full.hexw / .hexcfg | `5abf61bef086368559423daa3bea9a99` / `5926be5703f85531c9c46c1978c25287` | packer unchanged since P4 (#23 / #35 / #25 tables); no ABI change (`NNTR_HTP_ABI_VERSION` 4) | |
| /tmp/qwen3_full4k.hexw / .hexcfg | `e5c92fad696fb35405918941d1062be3` / `b0d8aca9a6c7626b0af8dc81e1de59eb` | `--max-seq 4224` | |
| /tmp/t512_23.i32 | `10bb428f92c792aa7733339f8e6b0da1` | `make_tokens.py --limit 512` on `eval_23.txt` (#35 §1) | accuracy anchor: #35 B1 / #25 on this unit **41.4947 / 162** |
| /tmp/t1024.i32, /tmp/t4096.i32 | `86e44c0633ebab556aafe3aa68d80c63` / `29684fbd372495912951cb3cc0d25e20` | #24 table | goal rows; #25 B `--eval` 5.8595 / 692 and 1.5623 / 3759 |

All four skels carry `-DHTP_PROF_FARF` so its cost (one FARF per call) cancels in every ratio. Every
model / token artifact is already on the device from #25. **If the Mac binaries did not reach the
workstation, rebuild them there** (SDK 6.4.0.1 gives the same `toolv19`; the link is not byte-reproducible,
so write the md5 you get and keep the size — #35 / #25 saw a uniform +160 / +192 B per skel):

```bash
git fetch && git checkout hvx/58-decode-attn-gqa      # do not rebase; commit the result on this branch
source <SDK>/setup_sdk_env.source; export ANDROID_NDK=<ndk path>
# A is the base branch's kernel: build it from a clean worktree of hvx_impl @ 2c880173
git worktree add /tmp/wt_A 2c880173 && (cd /tmp/wt_A && HEX_EXTRA_CFLAGS=-DHTP_PROF_FARF ./tools/hexagon/build_skel.sh) && cp /tmp/wt_A/build_hexagon/skel/libnntr_htp_skel.so build_hexagon/skel/libnntr_htp_skel.A.so
for v in "B:-DHTP_PROF_FARF" "C:-DHTP_PROF_FARF -DHTP_ATTN_L2FETCH=1" "D:-DHTP_PROF_FARF -DHTP_ATTN_NB=1"; do
  HEX_EXTRA_CFLAGS="${v#*:}" ./tools/hexagon/build_skel.sh && cp build_hexagon/skel/libnntr_htp_skel.so build_hexagon/skel/libnntr_htp_skel.${v%%:*}.so
done
./tools/hexagon/build_host_test.sh
md5sum build_hexagon/skel/libnntr_htp_skel.{A,B,C,D}.so build_hexagon/host/hexagon_e2e_test
```

## Steps (workstation, S25 Ultra `R3CY10WM83Y` on USB)

Name the unit: `adb devices` → paste the serial in the Results header. Every pass rule is a DSP Mcyc ratio
inside this session (HEXAGON.md §7 rule 9); the 4096 verdict is read **inside one order** (rule 9(e): A and
B back-to-back after the same idle, never against another sitting's row), and `dumpsys battery` is recorded
before each 4096 order. Insert the serial after the image path only if more than one device is attached.

```bash
adb devices
run() { cp build_hexagon/skel/libnntr_htp_skel.$1.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so; shift; ./tools/hexagon/run_e2e_test.sh "$@"; }
batt() { date +%T; adb shell dumpsys battery | grep -E 'temperature|status'; }
# 0. transport sanity on A (RPC_TEST PASS), also warms the DSP
cp build_hexagon/skel/libnntr_htp_skel.A.so build_hexagon/skel/libnntr_htp_skel.so && ./tools/hexagon/run_device_test.sh
# 1. 512 speed, pass 1 A B C D then pass 2 reversed D C B A (pass 2 is the record; rule 9(d))        ≈ 4 min
for v in A B C D; do run $v /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --chunk 128 --steps 64; done
for v in D C B A; do run $v /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --chunk 128 --steps 64; done
# 2. 4096 speed: order 1 = A then B, order 2 = B then A; battery before each order (rule 9(e))       ≈ 10 min
batt; for v in A B; do run $v /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64; done
batt; for v in B A; do run $v /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64; done
# 3. goal row: B at 1024                                                                              ≈ 1 min
run B /tmp/qwen3_full -- --tokens /tmp/t1024.i32 --chunk 128 --steps 64
# 4. accuracy at 512: B and C                                                                         ≈ 2 min
run B /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --eval
run C /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --eval
# ---- core session ends here (≈ 17 min) ----
# 5. tail: G2 at 4096 (B --eval, 4095 teacher-forced steps ≈ 6 min) and G5 (C at 4096 once, battery first)
run B /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --eval
batt; run C /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64
adb shell md5sum /data/local/tmp/nntr_htp/libnntr_htp_skel.so /data/local/tmp/nntr_htp/hexagon_e2e_test   # the last skel pushed
# 6. the per-kind DSP split of every run (one line per forward call in the device_farf log of each run)
python3 tools/hexagon/summ_farf_prof.py logs/hexagon/device_farf_*.log
```

Expected lines per run: `E2E init ok weights=598623744 kv=234881024 act=3932160 n_ops=451` (the 4096 image:
`weights=599180800`), 4 × `E2E step … n=128 …` (32 × at 4096), 63 × `E2E step … n=1 pcycles=<c> us=<t>
top1=<id>`, `E2E gen <64 ids>`, `E2E decode steps=63 median_us=<u> median_pcycles=<c> pcycles_per_us=<r>`,
`E2E wall_ms`; `--eval`: `E2E ppl <x> steps 511 top1 <n> wall_ms <w>` (4095 at 4096) then `E2E decode
steps=… `. In `logs/hexagon/device_farf_<stamp>.log` (every variant): one `nntr_htp: prof n=<n> pos=<p>
ops=451 kcyc=<k> mm8=<k> mm16=<k> lg=<k> attn=<k> rest=<k>` per call (kilo-pcycles);
`summ_farf_prof.py` prints the decode median of each field in Mcyc — **the `attn` column is what G1 reads**.
A hang, DSP restart or FARF fatal: note the last `E2E step` line and the FARF tail in the Notes, `adb reboot`,
continue with the next command (new on silicon in B / C / D: the block-relative fp16 narrowing of the
probabilities, the qf32 weighted merge of ≤ 3 partials per head, and — C only — the `l2fetch` descriptors;
a wrong descriptor can only slow a run, never change a value).

Reading the numbers (HEXAGON.md §8.2, #24 / #25): prefill tok/s = tokens ÷ (sum of `us` of the `n=128`
steps / 1e6); decode median host ms = `median_us` / 1000; decode tok/s = 1000 ÷ that; **DSP Mcyc/step =
`median_pcycles` / 1e6**; `pcycles_per_us` = the run's own clock; the FARF columns are `summ_farf_prof.py`'s
decode medians for that run's log. G5: KV GB/s of C at 4096 = 470 MB (28 layers × 8 heads × 2 × 4097 × 128
× 2 B, read once per step with the fusion) ÷ (`attn` Mcyc ÷ `pcycles_per_us` ÷ 1000 s).

## Results — unit serial: `__________` (S25 Ultra, v79), measured 2026-__-__

Reference cells for this unit (`R3CY10WM83Y`, #25 B = today's `hvx_impl` = variant A, 2026-09-18):
512 → 191.1 prefill tok/s / 29.715 ms / 33.65 tok/s / **54.575 Mcyc** (`attn` **17.260**, `mm8 / mm16 / lg / rest`
16.052 / 8.588 / 8.955 / 3.647) at 1836.6 `pcycles_per_us`; 1024 → 123.7 / 38.002 / 26.31 / **73.744** (`attn`
≈ 34.2); 4096 → 30.84 / 90.280 / 11.08 / **184.603** (`attn` ≈ 146.6; compares only inside this session's
own orders, rule 9(e)). **Environment check:** A (512, pass 2) must land within ± 5 % of **54.6 Mcyc**;
outside that band the session is void (stale artifact, wrong unit or a throttled sitting), not a result.

### Speed, 512 tokens (`--chunk 128 --steps 64`, `t512_23.i32`, `qwen3_full`) — pass 2 rows are the record

| variant | pass | skel md5 (from `run()`) | prefill tok/s | decode median host ms | decode tok/s | **DSP Mcyc/step** | `pcycles_per_us` | FARF median Mcyc: mm8 / mm16 / lg / **attn** / rest | log stamp |
|---|---|---|---|---|---|---|---|---|---|
| A control | 1 | | | | | | | | |
| B split+fusion | 1 | | | | | | | | |
| C + l2fetch | 1 | | | | | | | | |
| D fusion only | 1 | | | | | | | | |
| **D fusion only** | 2 | | | | | | | | |
| **C + l2fetch** | 2 | | | | | | | | |
| **B split+fusion** | 2 | | | | | | | | |
| **A control** | 2 | | | | | | | | |

### Speed, 4096 tokens (`qwen3_full4k`, `t4096.i32`) — each order is its own A/B (rule 9(e))

| order | `batt()` before the order (°C tenths / status / time) | idle since the previous run (min) | variant | skel md5 | prefill tok/s | decode median host ms | decode tok/s | **DSP Mcyc/step** | `pcycles_per_us` | FARF: mm8 / mm16 / lg / **attn** / rest | log stamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | | A | | | | | | | | |
| 1 | | | B | | | | | | | | |
| 2 | | | B | | | | | | | | |
| 2 | | | A | | | | | | | | |
| tail | | | C | | | | | | | | |

### Goal row (B at 1024; reference #25 B on this unit)

| variant | ctx | prefill tok/s (ref 123.7) | decode median host ms (38.002) | decode tok/s (26.31) | DSP Mcyc/step (73.744) | `pcycles_per_us` | FARF: mm8 / mm16 / lg / **attn** (≈ 34.2) / rest | log stamp |
|---|---|---|---|---|---|---|---|---|
| B | 1024 | | | | | | | |

### Accuracy (`--eval`; references #25 B / #35 B1 on this unit)

| variant | ctx | token file | `--eval` PPL (ref) | top-1 (ref) | within ± 0.3 % / ± 4? | `E2E gen` of the speed run identical to A pass 2? (`diff <(grep -o 'top1=[0-9]*' a.log) <(… b.log)`) |
|---|---|---|---|---|---|---|
| B | 512 | t512_23.i32 | (41.4947) | (162) | | |
| C | 512 | t512_23.i32 | (41.4947) | (162) | | |
| B | 4096 | t4096.i32 | (1.5623) | (3759) | | tail |
| D | 512 | — | not run (fusion only; nb = 1 keeps the pre-#58 numerics bit for bit on the simulator) | | | speed run only |

### Pass rules (plan §1; fill the verdict column)

| rule | reads | verdict |
|---|---|---|
| env | A, 512, pass 2, DSP Mcyc within ± 5 % of 54.6 | |
| G1 @512 | `attn(B) ≤ 9.0 Mcyc` (pass 2; A ≈ 17.3) **and** the DSP Mcyc/step column of B drops by the same amount ± 2 Mcyc (no cost moved into another kind) | |
| G1 @4096 | `attn(B) ≤ 0.5 × attn(A)` **inside each order** (order 1 and order 2 each; A ≈ 146.6 → B ≤ 73) and the step column drops by the same amount ± 2 Mcyc | |
| G2 | B `--eval` 512 within ± 0.3 % / ± 4 of 41.4947 / 162; B 4096 within ± 0.3 % / ± 4 of 1.5623 / 3759 (tail); the same for C at 512 if C becomes the default | |
| C default | `attn(C) ≤ 0.9 × attn(B)` at 4096 (tail, C vs the order-2 B) **and** `attn(C) ≤ attn(B)` at 512 (pass 2) → `HTP_ATTN_L2FETCH 1` ships; else 0 | |
| D (informational) | `attn(D)` vs `attn(B)` at 512, pass 2 = the balance share; `attn(D)` vs `attn(A)` = the fusion share | |
| G5 (recorded, not gated) | C's `attn` at 4096 → KV GB/s (formula above); C ≪ B ⇒ latency was the wall (next: DMA-to-VTCM); C ≈ B ⇒ bandwidth-bound (next: fewer KV bytes) | |

## Notes from the run
<thermal, warm-up, FARF errors, stale-file pushes, hangs — and the `summ_farf_prof.py` output of every run>

Simulator record for the reader (v79, `workers=6`, plan §4 gates; logs `logs/hexagon/sim_v79_58_*.log` on the
Mac): `attn` PASS on the plain, `-DHTP_ATTN_NB=1` and `-DHTP_ATTN_L2FETCH=1` builds; `attn_prefill`
0.000488281/0.208165 and `attn_decode` 0.000244141/0.186483 bit-identical to `hvx_impl` on all three; the
NB=1 build reproduces every one of the eight STATs of the old kernel bit for bit; `attn_decode_p1024`
pcycles 455511 (old kernel) → 329790 (B, final ladder; 329638 on the first run) / 239840 (D) / 331746 (C). The B op total misses the plan's
0.6 × floor (0.72 ×) because the simulator's **sixth worker runs at half speed in every configuration**
(per-worker timers: D gives worker 5 one head in 232 Kcyc where workers 2–4 take 118 Kcyc for one head;
B gives workers 0–4 four jobs each in 152–167 Kcyc and worker 5 the same four jobs in 308 Kcyc; the merge
pass costs 14.4 Kcyc); the balance the gate was written to check is the 152–167 vs 236 Kcyc of a two-head
worker. Whether the sixth worker is also slow on silicon is not known — if the B `attn` column lands near
2/3 of A instead of 1/2 at 4096, that is the first thing to split (per-worker FARF).

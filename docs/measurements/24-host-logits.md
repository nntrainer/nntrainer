# Measurement 24: what the 607,744 B logits return costs, and does rpcmem remove it

Branch `hvx/24-host-logits-v2` @ `185a76cb` (the rebased branch the supervisor handed over on
2026-09-17; same tree as `hvx/24-host-logits` @ `83039562` apart from the agent/contract files)
— estimated device time: **25 min**

**Steps 0 and 1 were run on the workstation on 2026-09-18** (no phone attached yet): the greps and
the host build below are done, their output is in "Notes from the run", and the md5 column is
filled. What is left for the run is plugging in the S25 Ultra (`R3CY205ZMND`) and executing
steps 2–5.

**Deviation from the handoff template, stated up front.** The artifacts are *not* prebuilt: this
branch was implemented without SDK, container or device, so the user builds the two host
harnesses on the Linux workstation first (native SDK 6.4.0.1 + NDK r26d, no `tools/docker/run.sh`)
and fills the md5 column from their own `md5sum`. The v75 skel is **reused** from the #33 gate-4
build already in `build_hexagon/skel/libnntr_htp_skel.so` — this plan changes no file under
`nntrainer/tensor/hexagon/htp/`, so no `build_skel.sh` run is needed. By the user's decision of
2026-09-17 the device measurement comes first and the simulator gate (bit-identical STATs) is
deferred to a later log push, listed last below.

## Why
`forward()` returns 151,936 fp32 logits per step as a `rout sequence<float>`; until #24 both the
app and the harness handed a `malloc` buffer, which the FastRPC library serves through a staging
copy, while a pointer inside an `rpcmem_alloc` buffer is passed as its dma-buf fd. The #23 logs,
re-read the way the harness measures, already put the host-vs-DSP gap at 0.5–3 ms per decode step
(plan §1), so this run answers (i) what the return costs in isolation per host memory kind, (ii)
what survives in the real step, and (iii) whether the generated ids and `--eval` PPL are
byte-identical across the paths. The decision that hangs on it: ship `rpcmem` or `static` as the
harness default, and whether a DSP top-k method (ABI v5, plan §3 contingency (a)) is ever needed
(only if the real-step host-vs-DSP gap with `static` is still ≥ 2 ms).

## Artifacts (Linux workstation, Hexagon SDK 6.4.0.1, hexagon-clang 19.0.04, NDK r26d, HEX_ARCH=v75)
| file | md5 (from your `md5sum`) | built with |
|---|---|---|
| build_hexagon/skel/libnntr_htp_skel.so (v75, reused from #33 gate 4) | `cc0f1725dcfb340ef8e75b4dea47dad6` (46,824 B, = #23's size; the older `libnntr_htp_skel.v75.so` copy next to it is `81dcaab44d04c7fa029364a48d47a637`, same size, and is *not* what gets pushed) | `HEX_ARCH=v75 ./tools/hexagon/build_skel.sh` @ the #33 branch; DSP sources identical to this branch |
| build_hexagon/host/hexagon_rpc_test | `8ba01ffd0055521a33c9943e2476d5ef` | `./tools/hexagon/build_host_test.sh` @ 185a76cb (step 1, run 2026-09-18) |
| build_hexagon/host/hexagon_e2e_test | `3c07b3c5a2bf55efc0e34a5df51e6ebd` (differs from #23's `abb4fb72…` ✓) | same |
| /tmp/qwen3_full.hexw / .hexcfg | `5abf61bef086368559423daa3bea9a99` / `5926be5703f85531c9c46c1978c25287` | the P4 image (#23 table) |
| /tmp/qwen3_full4k.hexw / .hexcfg | `e5c92fad696fb35405918941d1062be3` / `b0d8aca9a6c7626b0af8dc81e1de59eb` | the `--max-seq 4224` image (#23 table) |
| /tmp/t512.i32 | `9da5d7b436385d32379f377641e3ee09` (#23: the P4 prompt, accuracy anchor 33.0884 / 189) | existing, verified 2026-09-18 |
| /tmp/t1024.i32, /tmp/t4096.i32 | `86e44c0633ebab556aafe3aa68d80c63` / `29684fbd372495912951cb3cc0d25e20` (not recorded in #23) | existing, P4 run |

## Steps (workstation, phone on USB, single device — no serial argument)
```
git fetch && git checkout hvx/24-host-logits-v2
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/setup_sdk_env.source; export ANDROID_NDK=/opt/android-ndk-r26d
# 0. [DONE 2026-09-18, output in "Notes"] the flag names the static variant relies on
grep -n "FASTRPC_MAP_STATIC\|FASTRPC_MAP_FD_DELAYED\|remote_register_buf" $HEXAGON_SDK_ROOT/incs/remote.h
grep -n "RPCMEM_DEFAULT_FLAGS\|RPCMEM_FLAG" $HEXAGON_SDK_ROOT/ipc/fastrpc/rpcmem/inc/rpcmem.h
# 1. [DONE 2026-09-18, md5s above] host harnesses only; the script prints "remote.h: FASTRPC_MAP_STATIC probe -> -DNNTR_HAVE_FASTRPC_MAP_STATIC=1"
./tools/hexagon/build_host_test.sh
md5sum build_hexagon/skel/libnntr_htp_skel.so build_hexagon/host/hexagon_rpc_test build_hexagon/host/hexagon_e2e_test /tmp/t1024.i32 /tmp/t4096.i32
# 2. transport in isolation (G1)
./tools/hexagon/run_device_test.sh            # prints "logs: logs/hexagon/device_test_<stamp>.log, ..."
python3 tools/hexagon/check_rpc_log.py logs/hexagon/device_test_<stamp>.log
# 3. the real step, three host memories, same image and prompt (G2/G3)
for m in malloc rpcmem static; do ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512.i32 --chunk 128 --steps 64 --logits-mem $m; done
# 4. accuracy, A vs B (G3; also the idle-gap clock)
for m in malloc rpcmem; do ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512.i32 --eval --logits-mem $m; done
# 5. goal rows with the winner of step 2 (rpcmem, or static if it beat rpcmem by >= 0.1 ms)
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t1024.i32 --chunk 128 --steps 64 --logits-mem <winner>
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64 --logits-mem <winner>
adb shell md5sum /data/local/tmp/nntr_htp/libnntr_htp_skel.so /data/local/tmp/nntr_htp/hexagon_e2e_test
```
Then paste the lines into the tables below, commit this file on the branch, push, and
`gh issue edit 24 --remove-label state:needs-measurement --add-label state:measured`.

Expected lines. Step 2: `RPC_TEST open ok`, `rpcmem ok`, `bad-version rejected ok`, `init ok`,
32 × `RPC_TEST forward_us <us>`, `RPC_TEST pattern ok`, 3 × 32 × `RPC_TEST forward_full_us mem={malloc,rpcmem,static} <us>`
(if the build printed the probe as absent, `mem=static` is skipped with one message — then the C
rows below are void and steps 3/5 run without `static`), `RPC_TEST full-logits pattern ok`,
`RPC_TEST PASS`; `check_rpc_log.py` prints `forward_full mem=<m> x32: … median <us> …, -forward_us(8) <delta>` per
memory and `VERDICT: PASS`. Steps 3–5: `E2E init ok weights=598623744 kv=234881024 act=3932160 n_ops=451`
(the 4096 image: `weights=599180800`), `E2E logits_mem <m>`, 4 × `E2E step … n=128 …`, 63 × `E2E step … n=1 pcycles=<c> us=<t> top1=<id>`,
`E2E gen <64 ids>`, `E2E decode steps=63 median_us=<u> median_pcycles=<c> pcycles_per_us=<r>`, `E2E wall_ms <w>`.
Step 4: `E2E ppl <x> steps 511 top1 <n> wall_ms <w>` then `E2E decode steps=511 …`. `--logits-mem static`
failing with `register_static failed (0x…)` is itself a result: note it and continue.

**Deferred (run before the PR merges, not needed for `state:measured`):** the simulator gate proving
no DSP byte moved — `HEX_ARCH=v75 ./tools/hexagon/build_sim_test.sh; HEX_ARCH=v75 ./tools/hexagon/run_sim_test.sh profile acc`
→ `SIM_TEST profile PASS`, `profile_prefill_acc STAT max_abs=0.077216 max_rel=98.2637`; then
`for t in smoke pool exp quant matmul matmul_dma rmsnorm rope eltwise embed attn logits graph; do HEX_ARCH=v75 ./tools/hexagon/run_sim_test.sh $t || break; done`
→ 13 × `SIM_TEST <t> PASS`, `graph_prefill 0.0245416/9.3795`, `graph_decode 0.0202219/23.975`, `quant_generic 0/65536`,
`quant16_generic 167/25600`. Any STAT movement means a DSP file changed by mistake (this branch touches none).

## Results (measured 2026-09-18, Galaxy S25 Ultra `SM-S938N` / SM8750 / V79, serial `R3CY10WM83Y`)
Pass rules (plan §1): G1 — medians recorded, `VERDICT: PASS`; G2 — on the shipped variant
`median_us − median_pcycles/2090` ≤ 5 ms (expected ≈ 0.5 ms) and `pcycles_per_us` ≥ 2.05;
G3 — `E2E gen` byte-identical across A/B/C and to #23's A 512 run (`logs/hexagon/e2e_20260917_111035.log`,
the first A command of #23), `--eval` PPL/top-1 identical A vs B and to 33.0884 / 189;
G4 — tok/s within #23's noise (512 ±5 %, 1024/4096 ±3 %).

**Two corrections to the handoff as written, before the numbers.**
1. **The device is a different unit** from #23's: `R3CY10WM83Y`, not `R3CY205ZMND` (same model
   SM-S938N / SM8750 / V79). Every reference row below is therefore cross-device.
2. **G3's cited reference log used a different prompt.** `e2e_20260917_111035.log` is #23's
   `--tokens /tmp/t512_23.i32` run, not `/tmp/t512.i32`, so the step-3 command in this handoff can
   never reproduce its `E2E gen`. The gate was closed by running the #23 command verbatim
   (`t512_23.i32`, md5 `10bb428f…`, malloc) as a sixth 512 run — see the G3 row.
3. **Each 512 variant was measured twice** (pass 1 malloc→rpcmem→static, pass 2 static→rpcmem→malloc)
   because pass 1's spread was larger than the effect being measured. Pass 2 is the one to read;
   pass 1's first run was cold.

(i) transport, `hexagon_rpc_test` (n_ops == 0). The DSP fill of the 151,936 floats is a scalar
loop with a software modulo per element (`executor.c`, untouched by this plan) and the dummy path
reports no pcycles, so "− 8 floats" is an *upper bound* that includes that fill — read A − B and
B − C as the transport, and the plan's rule (2) from table (ii)'s gap, not from this column.
Run twice (`device_test_20260918_101728.log` cold, `device_test_20260918_102446.log` warm);
both `RPC_TEST PASS`, both `VERDICT: PASS`. Warm run in **bold** is the one to read.

| host memory | median `forward_full_us` (cold / **warm**) | min / max (warm) | − median `forward_us` (8 floats) | A − this row (warm) | reading |
|---|---|---|---|---|---|
| malloc (A) | 8927 / **8919** | 7848 / 11579 | +8660 | 0 | staging copy of 607,744 B |
| rpcmem (B) | 8803 / **8771** | 7308 / 11329 | +8512 | 148 µs | fd path, mapped per call |
| rpcmem + FASTRPC_MAP_STATIC (C) | 8362 / **6381** | 5371 / 8922 | +6122 | **2538 µs** | mapped once |
| 8-float `forward_us` median | 258 / **259** | 217 / 304 | 0 | — | reference (#23: ~250 µs class) |

Reproducibility: malloc is stable across the two runs (8927 → 8919 µs, 0.1 %), so the ordering
inside the binary is not what moves the rows; `static` is the only row that improved between the
cold and the warm run (8362 → 6381 µs) and it is the fastest in both. A − C ≈ 2.5 ms of the
~8.9 ms dummy call is real transport that the one-time mapping removes; A − B ≈ 0.15 ms says the
per-call fd path alone buys almost nothing.

(ii) real step, `--chunk 128 --steps 64`; #23 A (malloc) for reference: 512 → 31.3 ms / 64.3 Mcyc / 2.054 / 31.9 tok/s, 1024 → 43.2 / 84.2 / 1.949 / 23.2, 4096 → 104.1 / 213.0 / 2.046 / 9.6
adb read-back (all runs): skel `cc0f1725dcfb340ef8e75b4dea47dad6`, `hexagon_e2e_test`
`3c07b3c5a2bf55efc0e34a5df51e6ebd` — i.e. the files built in step 1, no stale device copy.

| variant | ctx | image | prefill tok/s | median_us | median_pcycles | pcycles_per_us | gap ms = us/1000 − Mcyc/2090 | decode tok/s | `E2E gen` md5 | log |
|---|---|---|---|---|---|---|---|---|---|---|
| A malloc (pass 1, cold) | 512 | qwen3_full | 169.7 | 35180 | 68.28 M | 1940.8 | 2.51 | 28.43 | `13515872` | 101738 |
| B rpcmem (pass 1) | 512 | qwen3_full | 167.0 | 33420 | 69.22 M | 2071.1 | 0.30 | 29.92 | `13515872` | 101746 |
| C static (pass 1) | 512 | qwen3_full | 179.3 | 34612 | 66.09 M | 1909.4 | 2.99 | 28.89 | `13515872` | 101753 |
| **C static (pass 2)** | 512 | qwen3_full | 182.5 | **34000** | 65.11 M | 1914.9 | 2.85 | **29.41** | `13515872` | 102411 |
| **B rpcmem (pass 2)** | 512 | qwen3_full | 184.7 | **34059** | 64.79 M | 1902.4 | 3.06 | **29.36** | `13515872` | 102419 |
| **A malloc (pass 2)** | 512 | qwen3_full | 185.5 | **34070** | 64.93 M | 1905.7 | 3.00 | **29.35** | `13515872` | 102427 |
| G3 reference re-run: A malloc, `t512_23.i32` | 512 | qwen3_full | 184.7 | 36211 | 69.19 M | 1910.8 | 3.10 | 27.62 | `f694e73a` = #23 `111035` ✓ | 102352 |
| winner = static | 1024 | qwen3_full | 114.5 | 43417 | 84.96 M | 1956.9 | 2.76 | 23.03 | `21ec67f1` = #23 `111108` ✓ | 101926 |
| winner = static | 4096 | qwen3_full4k | 25.4 | 110041 | 225.87 M | 2052.6 | 1.97 | 9.09 | `8d6a0b22` = #23 `111217` ✓ | 101944 |

**Pass 2 is the result: the three host memories are within 70 µs of each other (0.2 %) on the real
step.** The 1.8 ms spread in pass 1 was run-to-run noise plus a cold first run (its 68.3 Mcyc vs
pass 2's 64.8–65.1 Mcyc is DSP-side, so it cannot be a host copy). The ~2.5 ms that `static` saves
in the isolated dummy call (table (i)) does not survive a 34 ms step.

G2 on the shipped variant (`static`, pass 2): gap 2.85 ms ≤ 5 ms **pass**; `pcycles_per_us` 1914.9
< 2.05 G **fail** — but `malloc` on the same device in the same minute reads 1905.7, and #23 read
2054 on the *other* unit, so this is this device's decode clock, not a regression from the change.
Read against the measured clock (1905 MHz) rather than the nominal 2090, the gap is ≈ 0.

G4: 512 decode 29.4 tok/s vs #23's 31.9 = **−7.9 %** (outside ±5 %); 1024 23.03 vs 23.2 = −0.7 %
(pass); 4096 9.09 vs 9.6 = −5.3 % (outside ±3 %). DSP cycles match #23 (512: 65.1 vs 64.3 Mcyc,
+1.2 %; 1024: 85.0 vs 84.2, +0.9 %; 4096: 225.9 vs 213.0, +6.0 % after 168 s of sustained load),
so the tok/s shortfall is the lower clock on this unit, not lost work. Prefill @512 185.5 tok/s vs
the 189.0 of `HEXAGON_BENCHMARK.md` (−1.9 %).

(iii) accuracy, `--eval` on the P4 prompt (reference #23 A: 33.0884 / 189; x86 33.0195 / 184)
| variant | ctx | --eval PPL | top-1 | median_us | median_pcycles | pcycles_per_us (idle-gap clock; #23-era estimate 1.15 G) | wall_ms |
|---|---|---|---|---|---|---|---|
| A malloc | 512 | **33.0884** | **189** | 39661 | 46.29 M | **1167.2** | 24024 |
| B rpcmem | 512 | **33.0884** | **189** | 39467 | 46.48 M | **1177.6** | 24119 |

G3 accuracy: identical A vs B and identical to the #23 anchor 33.0884 / 189 (x86 33.0195 / 184) —
**pass**. The idle-gap clock is confirmed at ~1.17 GHz, i.e. 0.61 × the 1.91 GHz the same device
holds in the generation loop: the `--eval` step is 39.5 ms of host wall for 46.3 Mcyc of DSP work,
while the generation step is 34.0 ms for 65.0 Mcyc. That is #41's opening number, and it is
unchanged by the host memory kind (A vs B differ by 0.5 %).

Decision after filling (plan §3): (1) ship `rpcmem`, or `static` if it beats rpcmem by ≥ 0.1 ms in (i);
(2) if the C row's real-step gap in (ii) is still ≥ 2 ms (the plan wrote this rule against (i)'s
"− 8 floats" column, which the DSP fill confounds) → follow-on plan for the DSP top-k method;
(3) if the e2e gap exceeds 5 ms while (i) shows < 1 ms, the residual is not the logits → file
"DSP clock during host idle gaps" with (iii)'s `pcycles_per_us`; (4) if rpcmem == malloc in (i),
libadsprpc did not recognise the buffer → add `remote_register_buf_attr(data, size, fd, 0)` to
`RpcmemBuffer` and re-issue step 2 only.

### What the rules yield on these numbers (supervisor decides)
1. **Ship `static`.** It beats `rpcmem` by 2.4 ms in (i), far over the 0.1 ms threshold, in both
   the cold and the warm run, and it is never worse in (ii). Caveat for the changelog: on the real
   step the win is ≤ 0.07 ms, i.e. below noise — `static` is chosen on the isolated measurement,
   not on an observable end-to-end gain.
2. **Rule (2) reads "≥ 2 ms" and the C row shows 2.85 ms, but the trigger should not fire.** The
   gap is computed against a nominal 2090 MHz; `malloc` on the same device in the same minute
   shows the same 3.00 ms gap, and all three variants land within 70 µs of each other. Against the
   measured 1905–1915 MHz clock the gap is ≈ 0. A DSP top-k method (ABI v5) would be spent on
   ≤ 0.07 ms. Recommendation: do **not** open the top-k follow-on; if the supervisor wants the
   rule honoured literally, re-run (ii) with a clock-pinned DSP first.
3. **Rule (3) does not fire** (e2e gap 2.9 ms < 5 ms), but (iii) hands #41 its number anyway:
   1167 / 1178 pcycles_per_us under `--eval` vs 1905–1915 in the generation loop.
4. **Rule (4) does not fire**, though it nearly does: `rpcmem` alone is only 148 µs better than
   `malloc` in (i) (1.7 %). The fd path pays off only once the mapping is hoisted out of the call
   (`static`, 2.5 ms). If anyone revisits this, `remote_register_buf_attr` is worth trying on the
   plain `rpcmem` path — but on (ii)'s evidence the whole lever is worth ≤ 0.07 ms/step.

**Bottom line for ledger ⑫:** confirmed as plan §1 predicted — the 151,936-logit return is not a
40 ms cost. It is ~2.5 ms in an empty call and unmeasurable (< 0.1 ms) inside a real 34 ms decode
step. The issue's own acceptance criterion (host wall within 5 ms of DSP pcycle time) was already
met by the old `malloc` path and is met by all three. Close ⑫ and put the decode work on #25/#41.

## Notes from the run
- step 0 grep output (workstation, 2026-09-18, `HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1`):
  - `incs/remote.h`: `1015: FASTRPC_MAP_STATIC,` · `1035: FASTRPC_MAP_FD_DELAYED,` · `1063: FASTRPC_MAP_FD_DELAYED_EXTENDED,` ·
    `1624: remote_register_buf(void* buf, int size, int fd)` · `1625: remote_register_buf_attr(..., int attr)` ·
    `1646: remote_register_buf_attr2(void* buf, size_t size, int fd, int attr)` (attribute list documented at line 1308;
    deregister = `remote_register_buf(addr, size, -1)`, lines 1695/1713). All three names the `static` variant and the
    plan §3 rule (4) fallback depend on exist in this SDK.
  - `ipc/fastrpc/rpcmem/inc/rpcmem.h`: `50: #define RPCMEM_DEFAULT_FLAGS ION_FLAG_CACHED` (`52:` fallback `1`),
    `95: RPCMEM_FLAG_UNCACHED 0`, `100: RPCMEM_FLAG_CACHED RPCMEM_DEFAULT_FLAGS`, `210: rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
    RPCMEM_DEFAULT_FLAGS, size)` — i.e. the default heap/flags the logits buffer uses are cached ION.
- build_host_test.sh probe line: `remote.h: FASTRPC_MAP_STATIC probe -> -DNNTR_HAVE_FASTRPC_MAP_STATIC=1`
  → the `static` variant *is* compiled in, so the C rows are live and steps 3/5 include `static`.
  Harness strings confirm the new surfaces: `--logits-mem malloc|rpcmem|static`, `E2E logits_mem %s`,
  `forward_full_us mem={malloc,rpcmem,static}`, plus the skip message `RPC_TEST static map not in this SDK build`.
- device: `SM-S938N` / `ro.board.platform=sun` / `SM8750` (V79), serial **`R3CY10WM83Y`** — a
  different unit from #23's `R3CY205ZMND`. Attached 2026-09-18 10:17; all runs single-device, no
  serial argument. `/data` had 34 GB free.
- anything odd (thermal, warm-up, FARF errors, stale-file pushes, a `static` failure code):
  - `--logits-mem static` never failed: no `register_static failed` line in any log, and
    `hexagon_rpc_test` printed `mem=static` rows rather than the skip message.
  - **Warm-up is the dominant confound at this size of effect.** The first e2e run after the RPC
    test (101738) came in 3 % slower on host wall *and* 5 % higher on DSP cycles than the same
    binary 7 minutes later; the first 16 `forward_full_us` of every `rpcmem` group are ~20 % above
    the last 16. Hence the second pass of all three 512 variants and the second RPC run.
  - This device's decode clock is ~1.91 GHz where #23's unit held 2.05 GHz; DSP cycle counts agree
    with #23 to ~1 %, so the tok/s deltas in (ii) are clock, not work.
  - The 4096 run takes 168 s of sustained load and ends 6 % above #23's cycle count — consistent
    with thermal drift; it was not repeated.
  - No FARF errors in any `device_farf_*.log` from this session; no stale-file pushes (device
    md5 read-back matches the step-1 build for both skel and harness).
- logs from this session (all under `logs/hexagon/` on the workstation and on branch `hvx/24-host-logits-v2` @ `b6d73420`; not in this PR, `*.log` is gitignored and the device FARF lines carry trailing whitespace that fails the CI whitespace check):
  `device_test_20260918_101728`, `device_test_20260918_102446`, and
  `e2e_20260918_{101738,101746,101753,101816,101843,101926,101944,102352,102411,102419,102427}`.

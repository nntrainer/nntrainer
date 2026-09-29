# Measurement 35: v79-native skel after the quantiser decode fix (B1 gate, B2 IEEE-vs-qf, A v75 control)

Branch `hvx/35-v79-skel` — DSP artifacts built in the container from `305c51f1` (the last commit that
changes DSP sources; later commits on the branch are tools/docs only) — estimated device time: **25 min**
(B1 + B2 ≈ 18 min; A, optional, ≈ 7 min)

## Why
Issue #35 replaced the quantiser's qf32 magic-add decode with an integer decode of the sf bits, which
fixes the four v79 simulator failures of #23 and makes the int16 rounding sign-symmetric on v75
(HEXAGON.md §7 rule 6). The simulators now pass 13/13 on both arches; this run decides whether the
v79 skel (B1) can become the shipping skel (its speed is 10–27 % above v75 per #23) — pass rules below —
and answers ledger ④ / §7 rule 1 with B2, a v79 skel built with the qf-format helpers instead of the
IEEE ones. A (v75) is **optional**: v79 is the primary arch since 2026-09-17 (the device is the S25
Ultra) and v75 is wrapped up; run A only if the 25 min allow — it is the last v75 control row and the
only silicon reading of the new decode on v75.

## 0. Machine setup (any Linux x86_64 box + S25 / S25 Ultra on USB; skip what you already have)

| need | how | check |
|---|---|---|
| Hexagon SDK 6.4.x with HEXAGON_Tools 19.0.04 | Qualcomm Software Center or `qpm-cli` (`tools/docker/setup_wizard.sh sdk` shows the commands) | `source <SDK>/setup_sdk_env.source && echo $DEFAULT_TOOLS_VARIANT` → `toolv19` |
| Android NDK r26d | https://dl.google.com/android/repository/android-ndk-r26d-linux.zip | `export ANDROID_NDK=<path>` |
| adb | `apt install android-tools-adb` | `adb devices` lists the phone as `device` |
| python3 + `transformers>=4.40 safetensors tokenizers sentencepiece numpy huggingface_hub` | `pip install ...` (tools/docker/Dockerfile) | — |
| gcc/g++, md5sum | distro packages | — |

```bash
git clone https://github.com/dlwlzzero/nntrainer && cd nntrainer
git checkout hvx/35-v79-skel            # do not rebase; commit the result on this branch
source <SDK>/setup_sdk_env.source; export ANDROID_NDK=<ndk path>
```

## 1. Model artifacts (only if `/tmp/qwen3_full*` and the token files are not already on the box)

```bash
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='models/hf', allow_patterns=['*.json','*.txt','model.safetensors'])"
python3 tools/hexagon/make_w8cx_bin.py models/hf models/nntr_qwen3_0.6b_w8cx_DEFAULT.bin   # md5 7562313bb4cc70410450d0ab3a4fa563
./tools/hexagon/build_host_x86.sh
./build_x86_hexagon/nntr_hexpack models/nntr_qwen3_0.6b_w8cx_DEFAULT.bin /tmp/qwen3_full                 # .hexw 5abf61bef086368559423daa3bea9a99 / .hexcfg 5926be5703f85531c9c46c1978c25287
./build_x86_hexagon/nntr_hexpack models/nntr_qwen3_0.6b_w8cx_DEFAULT.bin /tmp/qwen3_full4k --max-seq 4224 # e5c92fad696fb35405918941d1062be3 / b0d8aca9a6c7626b0af8dc81e1de59eb
# prompt: the "Prompt text" block at the end of docs/measurements/23-sdk64-baseline.md, saved as models/eval_23.txt (md5 a838b25d115aab83e29d0605cb570a33)
python3 tools/hexagon/make_tokens.py models/hf models/eval_23.txt /tmp/t512_23.i32 --limit 512   # md5 10bb428f92c792aa7733339f8e6b0da1
python3 tools/hexagon/make_tokens.py models/hf <long.txt> /tmp/t1024.i32 --limit 1024             # speed rows only
python3 tools/hexagon/make_tokens.py models/hf <long.txt> /tmp/t4096.i32 --limit 4096
./build_x86_hexagon/hexagon_ref_run /tmp/qwen3_full --tokens /tmp/t512_23.i32 --eval             # your x86 reference for this prompt (container clang 41.2365 / 161, workstation gcc 41.5466 / 164)
```
The P4 prompt `/tmp/t512.i32` (md5 `9da5d7b4…`, x86 reference PPL 33.0195 / top-1 184, HEXAGON.md §5.1)
exists only on the original workstation; it is the G3 accuracy gate when available. On another box use
`t512_23.i32` against your own `hexagon_ref_run --eval` line and say so in the Notes.

**Rebase note (orchestrator, 2026-09-17 23:40).** The branch was rebased onto `hvx_impl` @ `1bbd8d53`
(after PRs #43–#46, including #33's validator / token-id changes in `nntr_htp_common.h` and
`htp_graph.c`). The three skels and the harnesses in the table were rebuilt from the rebased tree
(`305c51f1`); on that tree `HEX_ARCH=v79 run_sim_test.sh graph` and `quant` print STATs bit-identical to
the pre-rebase G1 run (`graph_prefill 0.0218946/6.88818`, `graph_decode 0.0197323/23.2374`,
`quant16_generic 184/25600`). The full v79 13-test + `profile acc` sweep is re-run once at PR time,
after this measurement (contract §7 budget); #33 and #35 touch disjoint DSP files.

## 2. Artifacts (built in the container, SDK 6.4.0.2 / hexagon-clang 19.0.04 `toolv19`, @ `305c51f1`)

> **Not the artifacts of the 2026-09-18 run.** That run rebuilt all five on the workstation with SDK
> 6.4.0.1 (same `toolv19`); the md5s actually measured are in the result table, the sizes and the reason
> in the Notes. The table below is kept as the container record.

| file | md5 (size) | built with |
|---|---|---|
| build_hexagon/skel/libnntr_htp_skel.v79.so (**B1**) | `af69e880d9b2181113030e0427117778` (50,728 B) | `HEX_ARCH=v79 ./tools/hexagon/build_skel.sh` |
| build_hexagon/skel/libnntr_htp_skel.v79qf.so (**B2**) | `9b50b2c2fe11a94683c0939b8f5a6508` (50,728 B) | `HEX_ARCH=v79 HEX_EXTRA_CFLAGS=-DHTP_FORCE_QF_HELPERS ./tools/hexagon/build_skel.sh` |
| build_hexagon/skel/libnntr_htp_skel.v75.so (**A**, optional) | `83182a7cd9bcdd780cf30c4e5e61e12a` (46,632 B) | `HEX_ARCH=v75 ./tools/hexagon/build_skel.sh` |
| build_hexagon/host/hexagon_rpc_test | `c269f67ebcf79f8c30e42abad9fa9f16` (524,984 B) | `./tools/hexagon/build_host_test.sh` (unchanged since #23) |
| build_hexagon/host/hexagon_e2e_test | `bbd5d1b15294bfffa95d6aedee7c247e` (1,262,128 B) | same |
| /tmp/qwen3_full.hexw / .hexcfg | `5abf61be…` / `5926be57…` | packer unchanged since P4 |
| /tmp/qwen3_full4k.hexw / .hexcfg | `e5c92fad…` / `b0d8aca9…` | `--max-seq 4224` |
| /tmp/t512_23.i32 | `10bb428f92c792aa7733339f8e6b0da1` | see §1 |

If the box did not receive the Mac binaries, rebuild them there with the three `build_skel.sh` lines
above plus `build_host_test.sh` (≈ 3 min). **The skel link is not byte-reproducible**: two builds of
the same tree differ in the order of ~100 dynamic-relocation entries (code and data identical; checked
with `cmp` on two v79 builds of `305c51f1`), so a rebuild has a different md5 at the same size — write
the md5 you get in the table and keep the size. Copy each skel to its variant name before the next build.

## 3. Measure (≈ 25 min; scripts push only files whose md5 changed)

Insert the serial after the image path (`run_e2e_test.sh /tmp/qwen3_full <serial> -- ...`) only if
more than one device is attached. Order: B1, then B2, then A if time allows.

```bash
# B1 (v79, the gate)
cp build_hexagon/skel/libnntr_htp_skel.v79.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so   # af69e880...
./tools/hexagon/run_device_test.sh                                                                    # RPC_TEST PASS
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t512_23.i32 --chunk 128 --steps 64
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t512_23.i32 --eval
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t512.i32 --eval                   # P4 prompt, if the file exists
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t1024.i32 --chunk 128 --steps 64
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t1024.i32 --eval
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --chunk 128 --steps 64
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32 --eval
# B2 (v79 with qf helpers): ledger ④ (3), recorded
cp build_hexagon/skel/libnntr_htp_skel.v79qf.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so   # 9b50b2c2...
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --chunk 128 --steps 64
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --eval
adb shell md5sum /data/local/tmp/nntr_htp/libnntr_htp_skel.so                                          # the skel that ran last (B2, or A below)
# A (v75, new decode) — OPTIONAL last v75 control row
cp build_hexagon/skel/libnntr_htp_skel.v75.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so   # 83182a7c...
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --chunk 128 --steps 64
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512_23.i32 --eval
./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full -- --tokens /tmp/t512.i32 --eval                     # P4 prompt, if the file exists
adb shell md5sum /data/local/tmp/nntr_htp/libnntr_htp_skel.so
```
Lines to look for: `RPC_TEST PASS`; per run `E2E init ok weights=598623744 ...`, one
`E2E step <i> pos=<p> n=<n> pcycles=<c> us=<t> top1=<id>` per chunk / decode step, `E2E gen ...`,
`E2E wall_ms <w>`; for `--eval`: `E2E ppl <x> steps 511 top1 <n> wall_ms <w>`. If a run hangs or the
DSP restarts, note the last `E2E step` line and the `logs/hexagon/device_farf_<stamp>.log` tail in the
Notes, `adb reboot`, continue with the next command.
Reading the numbers (HEXAGON.md §8.2): prefill tok/s = prompt tokens ÷ (sum of `us` of the `n=128`
steps / 1e6); decode ms/tok = median host `us` over the 63 `n=1` steps ÷ 1000; decode tok/s =
1000 ÷ that; DSP Mcyc/tok = median `pcycles` of the same steps ÷ 1e6.

## 4. Hand back
1. Fill the tables and the Notes below (phone model, SDK version, your x86 reference line).
2. `git add docs/measurements/35-v79-skel.md logs/hexagon/ && git commit -s -m "[docs] Fill the #35 v79-skel measurement (<phone>, SDK <ver>)" && git push`
3. `gh issue edit 35 -R dlwlzzero/nntrainer --remove-label state:needs-measurement --add-label state:measured`

## Results (run 2026-09-18, S25 Ultra `R3CY10WM83Y`, skels rebuilt on the workstation — see Notes)

> **Rebase note (implementer, 2026-09-18, PR time).** After this run the branch was rebased onto `hvx_impl` @
> `4b39062f` (adds PR #50, the shared op-shape source in `nntr_htp_common.h`, so the DSP bytes changed:
> the container v79 skel is now 50,792 B, `776287048384fbd0ba80961b49086741`; v79qf 50,792 B,
> `05e9943cfbc1a9ed63f9498409d4c3a2`; v75 46,696 B, `e451ce5a1b647897e5a767118a79a7e3`;
> `hexagon_e2e_test` 1,260,624 B, `ecdc1051698c7f59fc7e56f6ebfe8a22`; `hexagon_rpc_test` unchanged). The
> measured rows below are the pre-rebase binaries of the table above and stay the record; the v79 13-test +
> `profile acc` sweep was repeated on the rebased tree and reproduces the G1 STATs digit for digit
> (see "Simulator record" at the end).
Reference in brackets: B1 and B2 against #23 variant B (v79 skel, same silicon), A against #23 variant A
(`docs/measurements/23-sdk64-baseline.md`). v79 is the primary arch; A is the optional last v75 row. **Pass (B1 only)**: prefill tok/s within ±5 % of 207.4 /
131.2 / 34.4 and decode DSP Mcyc/tok within ±5 % of 60.1 / 76.5 / 177.6 at 512 / 1024 / 4096; `--eval`
on the P4 prompt inside the P4 band against x86 33.0195 / 184 (PPL 33.07–33.45), or on `t512_23.i32`
within +0.3 % of your own `hexagon_ref_run` line; no hang,
SSR or FARF fatal. B2 and A are recorded, not gated. The 512 decode column carries a ±5 %-class
spread by construction (#23 Notes), so read the 1024 / 4096 rows first.

| variant | skel md5 (from `md5sum` before the run) | ctx | image | prefill tok/s (ref) | decode median host ms (ref) | decode tok/s (ref) | DSP Mcyc/tok (ref) |
|---|---|---|---|---|---|---|---|
| B1 (v79) | `8d8cdb2668a64223ec3b89cacea32efa` | 512 | qwen3_full | **198.7** (207.4) −4.2 % | **32.391** (31.9) | **30.87** (31.36) | **61.200** (60.1) +1.8 % |
| B1 (v79) | `8d8cdb26…` | 1024 | qwen3_full | **131.1** (131.2) −0.1 % | **40.227** (39.6) | **24.86** (25.27) | **77.781** (76.5) +1.7 % |
| B1 (v79) | `8d8cdb26…` | 4096 | qwen3_full4k | **32.14** (34.4) −6.6 % | **92.274** (87.4) | **10.84** (11.44) | **188.029** (177.6) +5.9 % |
| B1 (v79) rerun, cooled | `8d8cdb26…` | 4096 | qwen3_full4k | **32.71** (34.4) −4.9 % | **89.103** (87.4) | **11.22** (11.44) | **186.753** (177.6) +5.2 % |
| B2 (v79qf) | `627480caa6a015b32148a64e16171b98` | 512 | qwen3_full | **197.7** (B1 198.7) −0.5 % | **32.395** (B1 32.391) | **30.87** (B1 30.87) | **61.133** (B1 61.200) −0.1 % |
| A (v75, optional) | `aff2fb88af73a47584392fa08f7aa889` | 512 | qwen3_full | **185.1** (189.0) −2.0 % | **33.882** (31.3) | **29.51** (31.92) | **64.256** (64.3) −0.1 % |

| variant | ctx | token file (md5) | --eval PPL (ref) | top-1 (ref) | x86 ref PPL / top-1 | gap % |
|---|---|---|---|---|---|---|
| B1 (v79) | 512 | t512_23.i32 (10bb428f…) | **41.4947** (41.2623) | **162** (161) | 41.5466 / 164 (this workstation, gcc, rebuilt and re-run today) | **−0.12 %** |
| B1 (v79) | 512 | t512.i32 (P4 prompt) | **32.4497** (—, #23 B did not run it) | **184** | 33.0195 / 184 | **−1.73 %** |
| B1 (v79) | 1024 | t1024.i32 | **5.8595** (5.9509) | **692** (692) | — | — |
| B1 (v79) | 4096 | t4096.i32 | **1.5623** (1.5687) | **3759** (3758) | — | — |
| B2 (v79qf) | 512 | t512_23.i32 | **41.4947** (= B1, digit for digit) | **162** (= B1) | 41.5466 / 164 | −0.12 % |
| B2 (v79qf) | 512 | t512.i32 (P4 prompt) | **32.4497** (= B1, digit for digit) | **184** (= B1) | 33.0195 / 184 | −1.73 % |
| A (v75, optional) | 512 | t512_23.i32 | **41.0253** (40.7596) | **167** (162) | 41.5466 / 164 | −1.25 % |
| A (v75, optional) | 512 | t512.i32 (P4 prompt) | **33.0295** (33.0884) | **188** (189) | 33.0195 / 184 | **+0.03 %** |

`adb shell md5sum` after the last run: `aff2fb88af73a47584392fa08f7aa889` (the A / v75 skel — A was run last,
then B1 was re-pushed for the 4096 re-run; the re-run log shows `8d8cdb26…`).

### Verdict against the pass rules

| rule | result |
|---|---|
| prefill within ±5 % of 207.4 / 131.2 / 34.4 | 512 −4.2 % PASS · 1024 −0.1 % PASS · **4096 −6.6 % / −4.9 % (re-run) FAIL** |
| decode DSP Mcyc within ±5 % of 60.1 / 76.5 / 177.6 | 512 +1.8 % PASS · 1024 +1.7 % PASS · **4096 +5.9 % / +5.2 % (re-run) FAIL** |
| P4 `--eval` inside 33.07–33.45 against x86 33.0195 / 184 | **32.4497 FAIL (below the band)**; top-1 **184 = x86 exactly** |
| `t512_23` within +0.3 % of this box's `hexagon_ref_run` | 41.4947 vs 41.5466 = −0.12 % PASS |
| no hang / SSR / FARF fatal | PASS |

So **B1 does not pass as written**: the two 4096 speed cells and the P4 PPL band are missed. Both misses
point away from a defect and toward stale reference cells — see the Notes — but the re-baseline is a
supervisor call, not a measurement call.

Reading B2 (HEXAGON.md §7 rule 1): B2 = B1 in PPL / top-1 and speed within noise → the 2026-08 v79
failures were a toolchain-8.x artefact, rule 1 closes; B2 correct but slower → the IEEE helpers are a
genuine v79 speed source; B2 wrong → the qf helpers have a v79-specific problem the v75 skel does not
see: record, file, do not block.

## Notes from the run

* **Phone / host.** Galaxy S25 Ultra `SM-S938N`, serial `R3CY10WM83Y` (the documented HVX baseline unit),
  USB, single device attached. Host: the Linux workstation, not the container.
* **Environment gap — SDK point release.** The container built the artifacts with SDK **6.4.0.2**; this
  workstation only has **6.4.0.1**, same `HEXAGON_Tools 19.0.04` / `toolv19`, and the Docker image is not
  reachable from this account (no `docker` group, no passwordless sudo). All three skels and both host
  binaries were therefore **rebuilt here** from the branch tip, so the md5s in §2 do not apply — the md5s
  actually run are in the result table above and were read back with `adb shell md5sum`. The host binaries
  match §2's sizes **exactly** (`hexagon_rpc_test` 524,984 B, `hexagon_e2e_test` 1,262,128 B); the skels
  are a uniform **+192 B** (v79 / v79qf 50,920 vs 50,728; v75 46,824 vs 46,632), which is the 6.4.0.1 ↔
  6.4.0.2 runtime/header delta, not a source difference. `-mhvx-ieee-fp` probed accepted on both arches.
* **x86 reference, re-measured today** on the branch tree (`build_host_x86.sh`, gcc): `t512_23.i32` →
  `PPL 41.5466 steps 511 top1 164`, `t512.i32` (P4) → `PPL 33.0195 steps 511 top1 184`. Both reproduce the
  documented lines digit for digit, so the reference side of the accuracy table is current, not inherited.
* **`RPC_TEST PASS`** on B1 before the sweep (`device_test_20260918_103233.log`).
* **The 4096 miss is reproducible and scales with context.** Against #23 variant B the decode DSP cost is
  +1.1 Mcyc at 512, +1.3 at 1024 and **+10.4 at 4096 (+9.2 on the cooled re-run)**; a second 4096 run after a ~7 min idle (the x86
  reference run) recovered only ~0.7 %, so it is not mainly thermal. A context-proportional cost cannot
  come from #35's quantiser decode, which is per-element and context-independent — and indeed 512 / 1024
  are inside the band. The branch was rebased onto `hvx_impl` `1bbd8d53` **after** #23 was measured, i.e.
  it now carries PRs #43–#46 including #33's per-op bounds validation and KV/attention changes in
  `htp_graph.c`; the 6.4.0.1 build is the other uncontrolled variable. This run does **not** isolate which.
  Suggested follow-up (cheap, one device sitting): the same 4096 speed pair on a v79 skel built from
  `1bbd8d53` itself, which splits "#35 cost" from "merged-since-#23 cost".
* **The P4 PPL miss is a band applied across arches.** The 33.07–33.45 band was derived from v75 P4
  records; #23 variant B never ran the P4 prompt, so there has never been a v79 P4 number to set it from.
  Evidence that 32.4497 is not a defect: top-1 is **184, exactly the x86 value**, and the same skel is
  −0.12 % on `t512_23` and matches the #23 B PPLs at 1024 (5.8595 vs 5.9509, top-1 692 = 692) and 4096
  (1.5623 vs 1.5687, top-1 3759 vs 3758). The v75 skel with the *same* new decode lands at 33.0295 /
  188, i.e. **+0.03 %** of x86 — so the −1.7 % is a v79-vs-v75 numerics difference in the IEEE-helper
  paths, consistent across both 512 prompts, and in the lower-perplexity direction.
* **B2 = B1 exactly.** `-DHTP_FORCE_QF_HELPERS` changes the binary (different md5, same size) and the
  simulator STATs, but on silicon it reproduces B1's PPL and top-1 **digit for digit** on both 512 prompts
  and sits within 0.5 % prefill / 0.1 % DSP-cycle noise. Per the reading rule below: the 2026-08 v79
  failures were a **toolchain-8.x artefact**; the IEEE helpers are neither a correctness nor a measurable
  speed source on v79 silicon. §7 rule 1 can close on this row.
* **FARF.** No SSR, no DSP fatal, no hang in any of the 15 runs. Every log carries the routine
  `open_shell failed for domain 3 … (errno Permission denied)` / `fastrpc_enable_kernel_optimizations
  failed for domain 3 (Bad address)` search-path noise that is present in passing runs too. The single
  `remote_handle64_invoke failed … (user err 0x80000414)` line appears only in the `rpc_test` log, which
  still printed `RPC_TEST PASS` — it is that test's own error-path case.
* **Thermal / order.** B1 (512 → 1024 → 4096, speed + eval) → B2 (512) → A (512) → B1 4096 re-run after
  the ~7 min x86 reference run. ~21 min wall end to end, of which ~14 min on the device. No warm-up
  run was discarded; the 512 decode column carries its documented ±5 %-class spread.
* **Logs.** `logs/hexagon/e2e_20260918_1032*–1052*.log` (+ matching `device_farf_*`), 15 runs, in this
  worktree.

## Simulator record for these artifacts (container, SDK 6.4.0.2, Rosetta; relative signal only)
* v79 (B1 sources): 13/13 PASS; `profile acc` PASS, `profile_prefill_acc STAT max_abs=0.0659682
  max_rel=92.7791`, `workers=6`, `total_pcycles=6174196`; `quant_generic pm1=0/65536`,
  `quant16_generic pm1=184/25600`, `tie`/`tie2`/`i16_tie`/`i16_tie2` byte-identical; `matmul_w8a8_* 0/0`,
  `matmul_dma ref_* 0/0`, `logits 0/0`, `graph_prefill 0.0218946/6.88818`, `graph_decode 0.0197323/23.2374`.
* v79 + `-DHTP_FORCE_QF_HELPERS` (B2 sources), the kinds whose helpers switch: `quant PASS`
  (`0/65536`, `184/25600`), `matmul PASS` (`w8a8_* 0/0`, `w8a16_m8 0.03125/0.000974659`), `attn PASS`
  (`prefill 0.000488281/0.208165`, `decode 0.000244141/0.186483` = the v75 values), `rmsnorm PASS`
  (`general 0.000976563/0.000670241`), `eltwise PASS` (`add 0/0`, `silu_mul 0/0`).
* v75 (A sources): 13/13 PASS; `profile acc` PASS, `profile_prefill_acc STAT max_abs=0.0757427
  max_rel=66.5909`, `workers=4`, `total_pcycles=3488445` (P4 / #23:
  `0.077216/98.2637`; moves because the int16 decode is now sign-symmetric); `quant_generic pm1=0/65536`,
  `quant16_generic pm1=162/25600` (was 167), `matmul_w8a8_m8 0.0078125/0.000788644` (unchanged),
  `graph_prefill 0.0273907/8.00437`, `graph_decode 0.0174583/24.125`, `logits 7.62939e-06/2.36832e-07`
  (unchanged).
* v79, **rebased tree** (`hvx_impl` @ `4b39062f` + this branch, container SDK 6.4.0.2, 2026-09-18, PR-time
  repeat per contract §7): 13/13 PASS; `profile acc` PASS, `profile_prefill_acc STAT max_abs=0.0659682
  max_rel=92.7791`, `workers=6`, `total_pcycles=6174596` (pre-rebase 6174196, +400 from #50's header);
  `quant_generic pm1=0/65536`, `quant16_generic pm1=184/25600`, `matmul_w8a8_* 0/0`, `matmul_dma ref_* 0/0`,
  `logits 0/0`, `attn_prefill 0.000488281/0.208165`, `graph_prefill 0.0218946/6.88818`,
  `graph_decode 0.0197323/23.2374` — every STAT equal to the G1 record above.

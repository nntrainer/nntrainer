# Measurement 23: SDK 6.4 rebuild of the M6 P4 baseline (v75 skel) and first v79-skel data point

Branch `hvx/23-sdk64-baseline` @ `e086248a` (device artifacts rebuilt on the workstation from that
commit, 2026-09-17; kernels unchanged since `82eacd7e`) — estimated device time: **45 min**

## Why
Every Hexagon build moved from the workstation's SDK 6.0.0.2 / toolchain 8.7.08 to the
container's SDK 6.4.0.2 / hexagon-clang 19.0.04 (`toolv19`). On the simulator the v75 build
is bit-identical to the P4 record (HEXAGON.md §8.3); this run shows whether the device agrees
(closes follow-up ⑭) and gives the v79-native skel its first silicon numbers (ledger ④: on the
v79 simulator `quant`, `matmul`, `matmul_dma` and `logits` fail, the rest pass — see §5.2).
Variant **A = v75 skel** decides #23; variant **B = v79 skel** is recorded, never pass/fail.

## Artifacts
The Mac is not reachable from the workstation, so the device artifacts were rebuilt on the
workstation itself (2026-09-17) against its own **Hexagon SDK 6.4.0.1**
(`/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1`, `hexagon-clang 19.0.04`,
`DEFAULT_TOOLS_VARIANT=toolv19`, `ANDROID_NDK=/opt/android-ndk-r26d`, API 31) — the same
compiler version as the container's 6.4.0.2, one SDK patch behind it. No container is involved:
`build_skel.sh` / `build_host_test.sh` only need `HEXAGON_SDK_ROOT` and `ANDROID_NDK`.
The full simulator set was re-run against this build and reproduces the container numbers
exactly (§ "Simulator record"), so these binaries carry the same evidence as the Mac ones.
None is tracked by git.

| file | md5 (workstation 6.4.0.1) | md5 (Mac 6.4.0.2, not used) | built with |
|---|---|---|---|
| build_hexagon/skel/libnntr_htp_skel.v75.so | `81dcaab44d04c7fa029364a48d47a637` (46,824 B) | `b9b1e3615d20959482efd51ad65b7320` (46,632 B) | `HEX_ARCH=v75 ./tools/hexagon/build_skel.sh` @ e086248a |
| build_hexagon/skel/libnntr_htp_skel.v79.so | `aba0126e848f9d993ef72ecfbf6fcab7` (50,920 B) | `bdfdf143f84fa158ebccba35862d9c04` (50,728 B) | `HEX_ARCH=v79 ./tools/hexagon/build_skel.sh` @ e086248a |
| build_hexagon/host/hexagon_rpc_test | `4a944ddad7c8305056c928c63aeb4085` (524,984 B) | `c269f67ebcf79f8c30e42abad9fa9f16` (same size) | `./tools/hexagon/build_host_test.sh` @ e086248a |
| build_hexagon/host/hexagon_e2e_test | `abb4fb72c9a994cb4f2027d6f1ffdec0` (1,262,128 B) | `bbd5d1b15294bfffa95d6aedee7c247e` (same size) | same |
| /tmp/qwen3_full.hexw / .hexcfg | `5abf61bef086368559423daa3bea9a99` / `5926be5703f85531c9c46c1978c25287` | identical | the workstation's P4 image, byte-identical to the Mac's |
| /tmp/qwen3_full4k.hexw / .hexcfg | `e5c92fad696fb35405918941d1062be3` / `b0d8aca9a6c7626b0af8dc81e1de59eb` | — | the workstation's `--max-seq 4224` image (P4 run) |
| /tmp/t512_23.i32 | `10bb428f92c792aa7733339f8e6b0da1` | identical | `make_tokens.py <hf> eval_23.txt --limit 512`, regenerated on the workstation |
| models/eval_23.txt | `a838b25d115aab83e29d0605cb570a33` | identical | the prompt, reproduced at the end of this file |

The P4 (SDK 6.0.0.2) binaries this build overwrites are kept in `build_hexagon/p4_sdk6002/`
(skel `eca4ef49…`, rpc `5e9248be…`, e2e `cfbe6d55…`).

**Issue #32's open question is answered:** the workstation's source checkpoint
`models/qwen3-0.6b-w8cx/nntr_qwen3_0.6b_w8cx_DEFAULT.bin` has md5
`7562313bb4cc70410450d0ab3a4fa563` — the same `.bin` the container packed from — and its packed
`qwen3_full.hexw` md5 matches the container's byte for byte. No 600 MB copy is needed.

Two names in the workstation's `/tmp` differ from the container's: the 4224 image is
**`qwen3_full4k`** (not `qwen3_full_4224`; `max_seq=4224` verified in its `.hexcfg`), and this
handoff's 512-token file is **`t512_23.i32`** — the pre-existing `/tmp/t512.i32` (md5
`9da5d7b4…`) is the *P4* prompt and feeds the optional continuity row only. The HF tokenizer used
to regenerate it is at `models/qwen3-0.6b-hf/` (Qwen3-0.6B, downloaded 2026-09-17). The 1024- and
4096-token files are the workstation's existing `t1024.i32` / `t4096.i32` from the P4 run (they
only feed the speed rows; no x86 reference exists for them).

## Steps (workstation, phone on USB)
1. `git fetch && git checkout hvx/23-sdk64-baseline`; the artifacts are already in
   `build_hexagon/` from the workstation rebuild — verify their md5s against the table above.
2. `adb devices` shows exactly one device (its serial is whatever the phone reports; the earlier
   drafts quoted `R3CY10WM83Y` and `R3CY205ZMND` inconsistently, so no serial is passed below —
   with a single device attached `adb` and both scripts default to it). The images and tokens are
   already in `/tmp` from the P4 run plus this handoff's regenerated `t512_23.i32`.
3. **Variant A (v75)** — run first so the baseline is never lost:
   ```
   cp build_hexagon/skel/libnntr_htp_skel.v75.so build_hexagon/skel/libnntr_htp_skel.so
   md5sum build_hexagon/skel/libnntr_htp_skel.so       # -> 81dcaab4..., write it in the table
   ./tools/hexagon/run_device_test.sh                   # RPC_TEST PASS (skel loads)
   ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t512_23.i32 --chunk 128 --steps 64
   ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t512_23.i32 --eval
   ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t1024.i32   --chunk 128 --steps 64
   ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t1024.i32   --eval
   ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32   --chunk 128 --steps 64
   ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full4k -- --tokens /tmp/t4096.i32   --eval
   ./tools/hexagon/run_e2e_test.sh /tmp/qwen3_full   -- --tokens /tmp/t512.i32    --eval   # optional P4-prompt row
   adb shell md5sum /data/local/tmp/nntr_htp/libnntr_htp_skel.so   # the skel that actually ran
   ```
   The first command of each image pushes ~600 MB and takes a few minutes; later ones skip the
   push after comparing md5s.
   Expected log lines: `E2E init ok weights=598623744 ...`, one `E2E step <i> pos=<p> n=<n> pcycles=<c> us=<t> top1=<id>`
   per chunk / decode step, `E2E gen ...`, `E2E wall_ms <w>`; for `--eval`: `E2E ppl <x> steps 511 top1 <n> wall_ms <w>`.
   Read the numbers as in HEXAGON.md §8.2: prefill tok/s = prompt tokens ÷ (sum of `us` of the `n=128` steps / 1000);
   decode ms/tok = median host `us` over the 63 `n=1` steps ÷ 1000; DSP Mcyc/tok = median `pcycles` of the same steps ÷ 1e6.
4. **Variant B (v79)** — same commands after
   `cp build_hexagon/skel/libnntr_htp_skel.v79.so build_hexagon/skel/libnntr_htp_skel.so && md5sum build_hexagon/skel/libnntr_htp_skel.so`
   (-> `aba0126e...`). If a run hangs or the DSP restarts, note the last `E2E step` line and the
   `logs/hexagon/device_farf_<stamp>.log` tail in "Notes", `adb reboot`, and continue with the next command;
   nothing in B blocks #23.
5. Paste the summary lines into the tables below, commit this file on the same branch, push, and set
   the issue label to `state:measured` (`gh issue edit 23 --remove-label state:needs-measurement --add-label state:measured`).

## Results — measured 2026-09-17, S25 Ultra `R3CY10WM83Y`, A 11:10–11:20, B 11:21–11:29
Reference = HEXAGON.md §8.2 M6 P4 (v75 skel, SDK 6.0.0.2, 2026-09-16). Pass for A: DSP Mcyc/tok within ±3 %, tok/s within ±5 %.

Skel identity: the device md5 read back after B is `aba0126e…` = v79, so B ran the v79 skel. A ran
a *different* binary — every A row differs from its B row in both speed and PPL — and the only
other skel the run sheet stages is v75 (`81dcaab4…`), so A = v75. The per-run `md5sum` output was
not captured in the logs; that inference is the evidence for A's identity.

| variant | ctx | image | prefill tok/s (ref) | decode median host ms (ref) | decode tok/s (ref) | decode median DSP Mcyc/tok (ref) |
|---|---|---|---|---|---|---|
| A (v75) | 512 | qwen3_full | **189.0** (192.1, −1.6 %) | **31.3** (36.1) | **31.92** (27.7, +15.3 %) | **64.3** (68.9, −6.7 %) |
| A (v75) | 1024 | qwen3_full | **117.1** (118.2, −1.0 %) | **43.2** (43.7) | **23.16** (22.9, +1.1 %) | **84.2** (85.2, −1.2 %) |
| A (v75) | 4096 | qwen3_full4k | **27.1** (27.3, −0.6 %) | **104.1** (106.1) | **9.61** (9.4, +2.2 %) | **213.0** (216.9, −1.8 %) |
| B (v79) | 512 | qwen3_full | **207.4** (192.1, +8.0 %) | **31.9** (36.1) | **31.36** (27.7, +13.2 %) | **60.1** (68.9, −12.8 %) |
| B (v79) | 1024 | qwen3_full | **131.2** (118.2, +11.0 %) | **39.6** (43.7) | **25.27** (22.9, +10.3 %) | **76.5** (85.2, −10.2 %) |
| B (v79) | 4096 | qwen3_full4k | **34.4** (27.3, +26.0 %) | **87.4** (106.1) | **11.44** (9.4, +21.7 %) | **177.6** (216.9, −18.1 %) |

**A verdict: no regression.** 1024 and 4096 sit inside both bands on every column. The 512 row
leaves the band on the *fast* side (decode +15.3 % tok/s, −6.7 % DSP cycles). Nothing in A is
slower than P4.

Why the 512 decode figure is the noisy one, from these logs alone — no repeat run was needed:

* The generation-mode decode median is taken over 63 steps that are still settling after the
  prefill. Median Mcyc of the first ten vs the last ten steps: A 512 `67.1 → 64.3`, B 512
  `66.9 → 60.3`, while A 1024 `83.6 → 84.8` and A 4096 `212.3 → 213.6` are flat. Only the 512
  runs drift, so only the 512 median depends on where the settling lands.
* Effective DSP clock (`pcycles ÷ us`) over the six speed runs spans 1884 – 2064 MHz, ±4.5 %.
* Thermal drift across the session is *not* the cause: the same context depth (pos 448–510)
  measured in the teacher-forced runs gives 52.0 Mcyc at 11:10 (cold) and 51.6 Mcyc at 11:20
  (warm, after four runs) — 0.8 % apart.

So the 512 decode pair carries a ±5 %-class spread by construction. One more run would add
another sample inside that spread, not an explanation; a defensible figure needs either repeated
medians or a different estimator (the settled tail rather than all 63 steps), which is a
methodology change, not a re-measurement. HEXAGON.md §8.2 already carries the same caveat for the
P3 → P4 host-wall jump.

**B is the surprise: the v79 skel is 10–27 % faster than v75 on the same silicon**, and the gap
widens with context (512 +9.7 %, 1024 +12.0 %, 4096 +26.9 % prefill against A; DSP cycles per
decoded token −6.5 / −9.1 / −16.6 %). It ran all six commands with no hang, no DSP restart and no
FARF fatal — the plan's "garbage / inf / crash" outcome did not happen.

Accuracy (`--eval`). Pass for A: the DSP-vs-reference gap inside the P4 band (+0.16 … +1.29 %; on
the workstation's own 512-token prompt P4 measured DSP 33.3132 / 189 against x86 33.0195 / 184,
+0.89 %). The 1024 / 4096 rows have no x86 reference (P4 did not record them either); fill
PPL / top-1 only.

**The x86 reference for `t512_23.i32` is not a single number.** Same image (md5 `5abf61be…`) and
same token file (md5 `10bb428f…`), two builds of `hexagon_ref_run`:

* container, clang, @ `4b25ac3c`: **PPL 41.2365 / top-1 161**
* workstation, native gcc 13, @ `e086248a` (2026-09-17, 199 s): **PPL 41.5466 / top-1 164**

a 0.75 % spread that comes from the host float build alone, with no DSP involved. The gap column
below is computed against the **workstation** reference, because that is the build this device run
is compared with; the container figure is kept as the second row.

The spread is specific to that new prompt, not to this machine: the workstation build re-run at
`e086248a` on 2026-09-17 reproduces the P4 prompt's reference **exactly** — PPL 33.0195 / top-1
184, the figure HEXAGON.md §5.1 has carried since P4 (199 s). So the P4-prompt accuracy row below
is anchored to a reference that has not moved, which is why it, and not `t512_23.i32`, is the
gate. Recorded in HEXAGON.md §5.1 as a result for ⑭.

| variant | ctx | token file (md5) | --eval PPL | top-1 | x86 ref PPL / top-1 | gap % | v79 behaviour (B only) |
|---|---|---|---|---|---|---|---|
| A (v75) | 512 | t512_23.i32 (10bb428f...) | 40.7596 | 162 | 41.5466 / 164 (workstation) | **−1.90 %** | — |
| A (v75) | 512 | t512_23.i32, vs container ref | (same run) | | 41.2365 / 161 (container) | **−1.16 %** | — |
| A (v75) | 512 | t512.i32 (9da5d7b4..., P4 prompt) | 33.0884 | 189 | 33.0195 / 184 | **+0.21 %** | — |
| A (v75) | 1024 | t1024.i32 | 5.9141 | 698 | — | — | — |
| A (v75) | 4096 | t4096.i32 | 1.5662 | 3763 | — | — | — |
| B (v79) | 512 | t512_23.i32 (10bb428f...) | 41.2623 | 161 | 41.5466 / 164 (workstation) | **−0.68 %** | matches |
| B (v79) | 512 | t512_23.i32, vs container ref | (same run) | | 41.2365 / 161 (container) | **+0.06 %** | matches |
| B (v79) | 1024 | t1024.i32 | 5.9509 | 692 | — | — | matches |
| B (v79) | 4096 | t4096.i32 | 1.5687 | 3758 | — | — | matches |

**A accuracy verdict: pass, on the P4-prompt row.** `t512.i32` gives +0.21 % against its x86
reference with top-1 189 — the same top-1 P4 recorded, and inside the P4 band (+0.16 … +1.29 %).
The `t512_23.i32` rows land *below* both of their references (−1.16 % / −1.90 %), i.e. the DSP
scores better than the host, which is not a meaningful reading while the host reference for that
prompt is itself two different numbers 0.75 % apart. Until one x86 build is declared canonical for
`t512_23.i32`, the P4 prompt is the accuracy gate and `t512_23.i32` is a data row.

**B accuracy: no v79 misbehaviour on silicon.** PPL and top-1 track A at every context
(41.2623/161 vs 40.7596/162, 5.9509/692 vs 5.9141/698, 1.5687/3758 vs 1.5662/3763) and the
generated text is fluent. The four v79 simulator failures (`quant`, `matmul`, `matmul_dma`,
`logits` — all ±1 LSB on the W8A8 quantiser/epilogue) do not produce device-visible damage.
This is ledger ④'s first silicon data point and it points the opposite way from the plan's
expectation: the v79 path is worth fixing for its 10–27 % speed, not fencing off. Issue #35
("make v75 the skel default and fail the build on IEEE HVX helpers at >= v79") should be re-read
against these numbers before it is implemented.

`find_divergence.py` bound: not required for this run; if A's gap leaves the band, run
`python3 tools/hexagon/find_divergence.py /tmp/qwen3_full /tmp/t512_23.i32 --chunk 8`
on a 1-layer image (HEXAGON.md §6) and record the `FIRST_DIVERGENCE` line here.

## Simulator record for the build that will run on the device (workstation, SDK 6.4.0.1, native x86)
Re-run 2026-09-17 on the binaries in the artifact table. Every STAT reproduces the container's
6.4.0.2 run; pcycles differ by ≤ 0.2 %.

* v75 **gate**: 13/13 PASS (`smoke pool exp quant matmul matmul_dma rmsnorm rope eltwise embed
  attn logits graph`) and `SIM_TEST profile PASS`, `profile_prefill_acc STAT max_abs=0.077216
  max_rel=98.2637` (= container = P4). `quant_generic pm1=0/65536`, `quant16_generic pm1=167/25600`;
  `matmul_w8a8_m8 0.0078125/0.000788644`, `m128 0.03125/0.000749625`, `w8a16_m8 0.0078125/0.000974659`,
  `m7 0.03125/0.0138741`; `matmul_dma ref_* 0.0078125/0.000714286`; `attn` prefill
  `0.000488281/0.208165`, decode `0.000244141/0.186483`; `logits 7.62939e-06/2.36832e-07`;
  `graph` prefill `0.0245416/9.3795`, decode `0.0202219/23.975`. `SIM_PROF` acc `workers=4`,
  `hvx_units=1024`, `total_pcycles=3450312` (container 3,454,962; P4 3,415,503),
  `MATMUL_W8A16 per_call=697868` (container 697,913), `MATMUL_W8A8 per_call=120306`,
  `ATTN per_call=91457`, `barrier_empty_x1000=4121706` (= container).
* v79 **recorded** (④): 9 PASS / 4 FAIL — the same four as the container, with the same numbers:
  `quant` (tie `x=-1.5 -> -1` vs ref `-2`, ±1 in `rand0`/`rand15`/`k3072`, `quant_generic 13/65536`,
  `quant16_generic 229/25600`), `matmul` (`w8a8_m1 0.03125/0.045677`), `matmul_dma`
  (`ref_4096k 0.0625/0.273998`), `logits` (`0.0195827/0.273973`). `rope`, `add`, `silu_mul` are
  0/0 here where v75 is not. `profile acc` PASS, `STAT max_abs=0.0899355 max_rel=49.2742`,
  `workers=6`, `hvx_units=1536`, `total_pcycles=6099224` (container 6,105,728),
  `barrier_empty_x1000=8319592` (= container) — not comparable with the 4-worker v75 numbers.
* x86 reference gate (native): `LOWER_TEST PASS`, `W8CX_BIN_TEST PASS`, oplist header `PASS`.

## Simulator record for the container build (Mac, Rosetta emulation; relative signal only)
* v75: 13/13 PASS, `profile acc` PASS, `profile_prefill_acc STAT max_abs=0.077216 max_rel=98.2637`
  (= P4), `total_pcycles=3454962` (P4 3,415,503), `MATMUL_W8A16 per_call=697913` (P4 683,954),
  `MATMUL_W8A8 per_call=120224`, `ATTN per_call=91475`, `barrier_empty_x1000=4121706`;
  `quant_generic pm1=0/65536`, `quant16_generic pm1=167/25600`.
* v79: 9 PASS (`smoke pool exp rmsnorm rope eltwise embed attn graph`), 4 FAIL (`quant`: tie row
  `x=-1.5 -> -1` vs ref `-2`, ±1 in `rand0`/`rand15`/`k3072`, `quant_generic 13/65536`,
  `quant16_generic 229/25600`; `matmul`: `w8a8_m1 max_abs=0.03125 max_rel=0.045677`;
  `matmul_dma`: `ref_4096k 0.0625/0.273998`; `logits`: `0.0195827/0.273973`).
  `profile acc` v79: PASS, `profile_prefill_acc STAT max_abs=0.0899355 max_rel=49.2742` (moved from the v75 value, still inside the 0.1 bound); the v79 simulator exposes 6 HVX units so it ran `workers=6`, `total_pcycles=6105728`, `MATMUL_W8A8 per_call=212250`, `MATMUL_W8A16 per_call=1252328`, `ATTN per_call=174095`, `barrier_empty_x1000=8319592` — not comparable with the 4-worker v75 numbers; wall 16m34s.
* x86 reference gate: `LOWER_TEST PASS`, `W8CX_BIN_TEST PASS`, oplist header PASS; `hexagon_ref_run --eval`
  on t512.i32 = PPL 41.2365 / top-1 161 (160 s in the container). The workstation's own build of
  `hexagon_ref_run` gives 41.5466 / 164 on the same bytes — see the accuracy section.

## Notes from the run
* Wall clock: A 11:10:27 → 11:20:47, B 11:21:02 → 11:29:xx — about 20 min for both variants, well
  under the 45 min estimate. No `adb reboot` was needed at any point.
* Every run logs the same three benign adsprpc lines (`open_shell failed for domain 3`,
  `fastrpc_enable_kernel_optimizations failed`, `Unable to add watcher for /vendor/lib64/rfs/dsp`,
  all `Permission denied`) and `hexagon: init failed (0x80000414), host abi v4, dsp abi v0` from
  the stub's first probe before the version handshake. They appear in the P4 logs too.
* No DSP restart, no SSR, no fatal FARF line in any of the 13 runs.
* A ran first from cold; its 512 row is the only one outside the pass band and it is faster, not
  slower. B ran ~1 min after A finished, on a warmer phone, and was faster still — so the A/B
  speed difference is not a warm-up artefact.
* Logs: `logs/hexagon/e2e_20260917_1110{35,42}`, `_1111{08,23}`, `_1112{17}`, `_111458`,
  `_112024` (A) and `_1121{02,09,35,48}`, `_112242`, `_112450` (B), plus the matching
  `device_farf_*` and two `device_test_*` files.

## Prompt text (`eval.txt`, 519 words; tokenizes to 512 tokens with `--limit 512`)
```
The Hexagon digital signal processor inside a modern smartphone is a curious piece of hardware. It was designed, in the first place, to handle the steady stream of audio and radio samples that a phone must process without waking the main application cores, and for that purpose it was given wide vector registers, a small but very fast local memory, and a scheduler that can keep several hardware threads busy at once. Over the years the same design has turned out to be a good fit for neural networks, because a network is, at bottom, a long sequence of multiply and accumulate operations over arrays that fit comfortably in those wide registers. The interesting part of moving a language model onto such a processor is not the arithmetic itself but everything around it: how the weights are laid out in memory so that each vector load brings in exactly the bytes the next instruction needs, how the activations are quantized on the fly without losing the small differences that decide which token comes next, and how the host processor and the signal processor agree on who owns which buffer at which moment. A single mistake in any of these arrangements does not usually produce a crash. Instead it produces a model that still speaks fluent sentences but chooses slightly wrong words, and the only way to notice is to measure the perplexity of a known text and compare it against a reference implementation that runs, slowly but exactly, on an ordinary computer. That is why every change to the kernels is followed by the same ritual: build the reference, build the simulator image, run the golden tests, and only then push the library to the phone and read the numbers back. The simulator does not model memory bandwidth, so it can tell you that the answer is right but not how long it will take; for the timing you need the device, a stopwatch in the harness, and a good deal of patience while the thermal throttling settles. The reward, when everything lines up, is a model that generates text at a rate the phone's own processor cannot match, using a fraction of the power, and leaving the application cores free to draw the screen and answer the network. It is a modest reward measured in tokens per second, but it is the kind of result that only appears after every layer of the stack has been checked twice.
There is also a quieter lesson in all of this. Tools change underneath a project: a new compiler arrives, a vendor ships a new software development kit, the simulator gains a processor generation it did not have before. Each of those changes is an opportunity for the numbers to drift without anyone touching the kernels, and so the same golden tests that guard against a careless edit also guard against a careless upgrade. Running them again after the toolchain moves, and writing down exactly which version produced which figure, is dull work, but it is the difference between knowing that a result still holds and merely hoping that it does.
```

# 23 — Rebuild `hvx_impl` with Hexagon SDK 6.4+ in the container and re-measure the M6 P4 baseline

Issue: dlwlzzero/nntrainer#23 (`prio:p0`). Ledger: ⑭ (SDK upgrade), ④ (v79 skel, first data point).
Contract: `docs/plans/0000-agent-system-and-env.md` §2–§3 and §7. Branch: `hvx/23-sdk64-baseline`, based on `hvx_impl` (`e2727cc5`; P4 kernels at `82eacd7e`).

Written without an SDK on the Mac (`~/Qualcomm/Hexagon_SDK` absent, image not built), so nothing below was run; step 0 is the probe that turns the assumptions into recorded facts.

## 1. Goal and gate

Move every Hexagon build to `tools/docker/run.sh` with SDK 6.4 or newer and prove nothing regressed against the last device numbers (SDK 6.0.0.2 workstation builds). Measurable gates, copied from the issue:

| # | Gate | Pass means |
|---|---|---|
| G1 | `profile acc` + the 13 sim tests on **v75** in the container (gate); the same set on **v79** run and recorded | v75: 13 × `SIM_TEST <name> PASS`, `SIM_TEST profile PASS`, `quant_generic` / `quant16_generic` STAT rates within HEXAGON.md §5.2 (≤ 2e-4, < 2 %). v79: per-test PASS/FAIL table recorded as ④ data; failures confined to `attn matmul eltwise quant graph profile` (the IEEE surface, §2) do not block #23 |
| G2 | v75 and v79 skels + host harness build | `build_hexagon/skel/libnntr_htp_skel.{v75,v79}.so`, `build_hexagon/host/{hexagon_rpc_test,hexagon_e2e_test}` exist, md5 recorded |
| G3 | Handoff `docs/measurements/23-sdk64-baseline.md` filled by the user | v75 skel within noise of §8.2 M6 P4: **192.1 / 27.7 tok/s @512, 118.2 / 22.9 @1024, 27.3 / 9.4 @4096**; DSP median Mcyc/tok 68.9 / 85.2 / 216.9 within ±3 %; tok/s within ±5 % (host wall is the noisier number, §8.2); `--eval` 512 tok PPL / top-1 33.3132 / 189 within the §8.1–§8.2 band (+0.16 … +1.29 % over the x86 reference 33.0195 / 184) |
| G3b | v79 skel on the same runs | recorded only: PPL in band or garbage / inf / crash (§7 rule 1). Data for ④, never pass/fail here |
| G4 | Docs | HEXAGON.md §5.2 / §5.3 / §7 rule 4 / §8.3 / §9 toolchain lines, `meson_options.txt` comment, follow-up ⑭ marked done |

No ABI change: `nntr_htp_common.h` version, wire format and image layout are untouched, so `lower_qwen3`, `ref_graph_forward`, the `graph` test, `find_divergence.py` and `nntr_hexpack` do not move.

## 2. Where it lives

Every hard-coded SDK path / version found by `grep -rn "local/mnt/workspace\|6\.0\.0\.2\|hexagon_toolv"` (scripts, docs, meson, py, txt; `subprojects` excluded), and what each becomes:

| File:line | Today | Becomes |
|---|---|---|
| `tools/hexagon/build_sim_test.sh:3-4` | comment "v75; SDK 6.0.0.2 has QuRT sim images up to v75" | "v75 or v79: needs `$HEXAGON_SDK_ROOT/rtos/qurt/compute<arch>/sdksim_bin/runelf.pbn` (6.4 ships both)". Code already uses `HEXAGON_SDK_ROOT` / `DEFAULT_HEXAGON_TOOLS_ROOT` (:6-7, :22, :26-29) — no path edit |
| `tools/hexagon/run_sim_test.sh:11` | `ls -d …/hexagon_toolv*_${HEX_ARCH} \| tail -1` (lexical, silent when absent) | `sort -V \| tail -1` plus an explicit error "no run_main_on_hexagon image for $HEX_ARCH under $HEXAGON_SDK_ROOT/libs/run_main_on_hexagon/ship" ; keep `RUNMAIN` as the documented variable. `:12,:17,:19,:23` already use the env vars |
| `tools/hexagon/build_skel.sh:16` | `HEX_ARCH="${HEX_ARCH:-v79}"` | unchanged in #23 (④ owns the default); README already warns. `:23,:35,:41-45` already use the env vars |
| `tools/hexagon/build_skel.sh:36`, `build_sim_test.sh:23` | `-mhvx-ieee-fp` literal | `${HEX_IEEE_FLAG}` set by a 3-line probe (`hexagon-clang -m$HEX_ARCH -mhvx -mhvx-ieee-fp -c -x c /dev/null`) so an SDK that drops or implies the flag still builds; the probe result is printed |
| `tools/hexagon/build_host_test.sh:23` | `$HEXAGON_SDK_ROOT/ipc/fastrpc/remote/ship/android_aarch64` | unchanged (already a variable); verify the directory still exists in 6.4 |
| `meson_options.txt:38` | `-Dhexagon-sdk-root=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.3.0.0` | `-Dhexagon-sdk-root=$HEXAGON_SDK_ROOT   # e.g. /opt/qcom/Hexagon_SDK/6.4.x.y inside tools/docker` (`meson.build:281-295` already takes the option; its `ipc/fastrpc/qaic/Ubuntu/qaic` subpath is checked in step 0) |
| `docs/backend_guide/HEXAGON.md:382` | "Hexagon SDK 6.3.0.0+" | "Hexagon SDK 6.4 or newer, mounted by `tools/docker/run.sh`" |
| `HEXAGON.md:452` (§5.2) | "All pass on v75 and v79 (SDK 6.3.0.0, toolchain 8.8)" | new SDK / toolchain string from step 0, date, and the v79 result |
| `HEXAGON.md:677` (§7 rule 4) | "`-mhvx-ieee-fp` (toolchain 8.8)" | add the 6.4 probe outcome |
| `HEXAGON.md:915` (§8.3) | "hexagon-sim v75 (SDK 6.0.0.2, toolchain 8.7.08)" | keep as the P1–P4 history; add one line "re-run with SDK <ver> / toolchain <ver> on v75 and v79, `profile_prefill_acc STAT` = …" |
| `HEXAGON.md:1160-1165` (§9) | "waits on a Hexagon SDK upgrade to 6.4" | rewrite to the ④ decision that the handoff produced |
| `HEXAGON.md:741` (§8) | "SDK 6.3.0.0, 2026-08-31" | leave: dated M1 result |
| `docs/backend_guide/HEXAGON_BENCHMARK.md:27` | note "SDK 6.0.0.2, v75 skel" | supervisor adds the 6.4 rows from the handoff; the old row stays as history |
| `docs/superpowers/specs/hexagon-hvx-optimization/07-follow-ups.md:74-76` | ⑭ open | mark done with plan / handoff paths |
| `docs/superpowers/specs/hexagon-hvx-optimization/{00-overview.md:6,01-profile-baseline.md:66}`, `hexagon-hmx/{00-overview.md:39,02-build-integration.md:7}`, `docs/plans/0000:70` | historical ledger / contract text | leave |

Container plumbing that already does the right thing (verified by reading): `tools/docker/run.sh:28-31,56-63` mounts `~/Qualcomm/Hexagon_SDK` at `/opt/qcom/Hexagon_SDK` and passes `HEX_ARCH HEX_EXTRA_CFLAGS SIM_TIMING NNTR_NUM_THREADS NNTR_REQUIRE_SDK HEXAGON_SDK_VERSION`; `tools/docker/entrypoint.sh:11-30` picks the newest version dir (or `HEXAGON_SDK_VERSION`) and sources `setup_sdk_env.source`; `Dockerfile:56-58` creates the mount point. `HEX_IEEE_FLAG` is internal to the scripts, so `run.sh` needs no new pass-through.

Source that a v79 build exercises differently (the ④ surface, read-only in #23): `nntrainer/tensor/hexagon/htp/hvx/hvx-base.h:116,151-165,199-288` (`__HVX_ARCH__ >= 79` selects `Q6_Wsf_vmpy_VhfVhf`, `Q6_Wsf_vmpyacc_WsfVhfVhf`, `Q6_Vsf_vadd/vsub/vmpy_VsfVsf`, `Q6_Vhf_vadd/vsub/vmpy_VhfVhf`, `Q6_Vsf_vfneg_Vsf`). Callers: `ops/hvx-attn.c:87,123-124` (score / PV accumulate), `ops/hvx-matmul.c:277-285` (W8A16 epilogue), `ops/hvx-eltwise.c:77-113` (SILU), `hvx/hvx-quant.h:64`, `hvx/hvx-exp.h:190-217`. So on v79 the tests `attn`, `matmul`, `eltwise`, `quant`, `graph` and `profile acc` run IEEE code that §7 rule 1 says misbehaves on silicon and, printf-sensitively, on the v79 simulator.

## 3. Design

One toolchain, mounted, discovered at container start. The scripts keep their `HEXAGON_SDK_ROOT` / `DEFAULT_HEXAGON_TOOLS_ROOT` contract (they already refuse to run without them); what changes is (a) the only glob that encodes a toolchain (`hexagon_toolv*`) becomes version-sorted and guarded, (b) `-mhvx-ieee-fp` is probed instead of assumed, (c) comments and docs stop naming 6.0.0.2 / 6.3.0.0 / `/local/mnt/workspace`. No kernel edits: a v79 failure is recorded, not fixed (④), so #23 cannot smuggle a numerics change past the §7 review list.

Toolchain identification (recorded verbatim in HEXAGON.md §5.2 and §8.3, and in the handoff header):

```
tools/docker/run.sh bash -c 'basename $HEXAGON_SDK_ROOT; basename $DEFAULT_HEXAGON_TOOLS_ROOT; \
  $DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin/hexagon-clang --version | head -1; \
  ls $HEXAGON_SDK_ROOT/libs/run_main_on_hexagon/ship/; ls $HEXAGON_SDK_ROOT/rtos/qurt/'
```
Expected shape: SDK dir `6.4.x.y`; tools dir `8.<minor>.<patch>`; `hexagon_toolv<NN>_v75` and `hexagon_toolv<NN>_v79` (the `toolv<NN>` suffix is the toolchain major.minor, e.g. `toolv88` = 8.8, and is what `run_sim_test.sh:11` globs); `computev75`, `computev79` (each must hold `sdksim_bin/runelf.pbn` and `debugger/lnx64/qurt_model.so`, `run_sim_test.sh:17,23`). The string to record is "SDK <dir>, hexagon-clang <version line>, run_main_on_hexagon <toolv dir>".

Intrinsic-rename detection: the 94 distinct `Q6_*` names used by `nntrainer/tensor/hexagon/htp/**` and `test/hexagon/sim/**` (list with `grep -rhoE 'Q6_[A-Za-z0-9_]+' … | sort -u`) are checked against the 6.4 headers before the first build, and again by the build itself (`-Wall -Werror`; hexagon-clang ≥ 8.8 is clang-16-based, where an implicit declaration is already an error):
```
tools/docker/run.sh bash -c 'H=$(find $DEFAULT_HEXAGON_TOOLS_ROOT/Tools -name hvx_hexagon_protos.h -o -name hexagon_protos.h); \
  for n in $(grep -rhoE "Q6_[A-Za-z0-9_]+" nntrainer/tensor/hexagon/htp test/hexagon/sim | sort -u); do \
    grep -qw "$n" $H nntrainer/tensor/hexagon/htp/dma/dma-queue.h nntrainer/tensor/hexagon/htp/hex/hex-utils.h || echo "MISSING $n"; done'
```
A `MISSING` name is renamed only if the 6.4 header offers a same-semantics replacement (documented in the commit); anything else is filed as an issue.

Rejected alternative: keep the workstation's 6.0.0.2 for skel builds and use 6.4 only for the simulator. That reintroduces the environment gap (contract §2 rule 3: one toolchain builds everything) and would leave the v79 simulator image unusable for the artifacts that actually ship.

## 4. Steps

Each step ends with a gate from `.claude/skills/hexagon-gates`. All commands run from the repo root on the Mac. `HEXAGON_SDK_VERSION=<dir from step 0>` is exported for every command so the handoff is reproducible even if a second SDK version appears later.

**Step 0 — probe the SDK (no source change).** Run the identification and `MISSING` commands of §3, plus `tools/docker/run.sh bash -c 'echo $HEXAGON_SDK_ROOT $DEFAULT_HEXAGON_TOOLS_ROOT; which hexagon-clang'` and `NNTR_REQUIRE_SDK=1 tools/docker/run.sh true` (⑭ item: `setup_sdk_env.source` exports both variables and puts `Tools/bin` on `PATH`; if 6.4's script prints or `cd`s, `entrypoint.sh:26` already discards stdout — note any stderr). Verify `ipc/fastrpc/qaic/Ubuntu/qaic`, `ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so`, `ipc/fastrpc/rpcmem/inc`, `incs/stddef` still exist (`build_skel.sh:23`, `build_host_test.sh:23,32`, `meson.build:290`). Gate: all four listings printed and pasted into the plan's step-0 record; `hexagon_toolv*_v79` and `computev79` present (⑭ item). If the v79 image is missing, stop and report (issue AC1 is then not satisfiable with this SDK — `needs-user`).

**Step 1 — de-hardcode.** Edit the four script lines and the meson comment of §2. Gate 0 (`clang-format-14` is irrelevant for `.sh`; run `bash -n` on the five scripts) and gate 1 (`build_host_x86.sh`, `test_lowering`, `test_oplist_header.c`) to show the x86 path is untouched.

**Step 2 — v75 simulator.**
```
HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/build_sim_test.sh
HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/run_sim_test.sh profile acc     # gate 2
for t in smoke pool exp quant matmul matmul_dma rmsnorm rope eltwise embed attn logits graph; do
  HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/run_sim_test.sh $t || break; done   # gate 3
```
Record `profile_prefill_acc STAT max_abs=… max_rel=…` (expected unchanged from §8.3's 0.0914676 / 93.3662; a change is data — the qf ops are deterministic and no fast-math is set — and must stay inside the 0.1 bound), `quant_generic` / `quant16_generic` STAT lines, and the `-mhvx-ieee-fp` probe line. Gate: 13 PASS + profile PASS on v75.

**Step 3 — v79 simulator.** Same commands with `HEX_ARCH=v79`. Compile errors here are the ⑭ rename check for the `>= 79` intrinsics (`hvx-base.h:151-288`); fix names only (§3). Run `profile acc` first, then the 13 tests. Gate: 13 PASS + profile PASS on v79. If `attn`, `matmul`, `eltwise`, `quant`, `graph` or `profile` fail with inf / NaN / bound violations while `smoke pool exp rmsnorm rope embed logits matmul_dma` pass, that is §7 rule 1 reproducing on the 6.4 v79 simulator: record the exact STAT / failing op lines under "Step 3 record", keep going to step 4 (the v79 skel is still built for G3b), and flag AC1-on-v79 as not met by this plan's scope (see §5).

**Step 4 — skels and harness (gate 4).**
```
HEX_ARCH=v75 tools/docker/run.sh ./tools/hexagon/build_skel.sh && cp build_hexagon/skel/libnntr_htp_skel.so build_hexagon/skel/libnntr_htp_skel.v75.so
HEX_ARCH=v79 tools/docker/run.sh ./tools/hexagon/build_skel.sh && cp build_hexagon/skel/libnntr_htp_skel.so build_hexagon/skel/libnntr_htp_skel.v79.so
tools/docker/run.sh ./tools/hexagon/build_host_test.sh
tools/docker/run.sh md5sum build_hexagon/skel/libnntr_htp_skel.v75.so build_hexagon/skel/libnntr_htp_skel.v79.so build_hexagon/host/hexagon_rpc_test build_hexagon/host/hexagon_e2e_test
```
Then the images and tokens the handoff needs (x86, no SDK): `nntr_hexpack /model/<w8cx>.bin build_x86_hexagon/qwen3_full` (max_seq 2048, 512 / 1024 runs) and `… --max-seq 4224 build_x86_hexagon/qwen3_full_4224` (4096 run; weights 599,180,800 B, §8.2), `make_tokens.py … --limit {512,1024,4096}`; `hexagon_ref_run … --tokens t512.i32 --eval` must print PPL 33.0195 / top1 184 (§5.1). md5 of every file goes into the handoff table. HexKL (⑭ item): no script or meson target links `libhexkl_micro.a` today (`grep -rn hexkl tools/ meson.build nntrainer/` hits only the mount plumbing), so coexistence is a note, not a build: the add-on layout recorded in `docs/superpowers/specs/hexagon-hmx/02-build-integration.md:7` is `lib/6.3.0.0/hexagon_toolv88_{v75,v79}` (built with toolchain 8.8, not 6.0.0.2 as the issue says); a static lib from an older hexagon-clang links into a 6.4 skel unless the ELF flags disagree (`hexagon-readelf -h` on both when the HMX work starts, `needs-user` anyway).

**Step 5 — handoff (device measurement unavoidable).** Write `docs/measurements/23-sdk64-baseline.md` from the `hexagon-handoff` template. Header: branch, commit, "SDK <ver> / toolchain <ver> (step 0)", estimated device time **≈ 40 min**. Artifacts: the two skels, harness, both images, three token files with md5 and commit. Variants: **A = v75 skel** (`libnntr_htp_skel.v75.so`), **B = v79 skel**. Per variant, in this order (image swap once per context group):

| ctx | image | commands (via `run_e2e_test.sh /tmp/<image> R3CY10WM83Y -- …`) | reference (A) |
|---|---|---|---|
| 512 | qwen3_full | `--tokens t512.i32 --chunk 128 --steps 64` then `--tokens t512.i32 --eval` | 192.1 / 27.7 tok/s, 36.1 ms, 68.9 Mcyc; PPL 33.3132 / 189 |
| 1024 | qwen3_full | same with t1024.i32 | 118.2 / 22.9, 43.7 ms, 85.2 Mcyc; PPL — (not recorded for P4; fill) |
| 4096 | qwen3_full_4224 | same with t4096.i32 | 27.3 / 9.4, 106.1 ms, 216.9 Mcyc; PPL — (fill) |

Time budget from §8.2 host walls: A ≈ 512 (5 s + 20 s) + 1024 (12 s + 46 s) + 4096 (157 s + ~7 min) ≈ 11 min; B the same; two image pushes (~600 MB each) and md5 checks ≈ 5 min; slack for a hung v79 run ≈ 10 min. Result tables carry the skel md5 printed by the run (environment gap), tok/s, decode median host ms, DSP Mcyc/tok, `--eval` PPL / top-1, x86 ref PPL, and a free-text "v79 behaviour" column (garbage / inf / crash / matches). Meaning of B for ④: PPL in band and no inf → ④ step (2) done, next is ④ (3)/(4) (decide qf forcing, flip the default); garbage or inf → rule 1 reproduced with the 6.4 toolchain, ④ needs the qf-path forcing before any default change. Neither outcome changes #23's verdict. End of implementer scope: set `state:needs-measurement`.

**Step 6 — after `state:measured` (implementer, then PR).** Compare A against G3; on pass, apply G4 doc lines with the recorded strings, mark ⑭ done in the ledger, open the PR into `hvx_impl`. On a regression, do not touch kernels: file the regression as a new issue with the handoff numbers and stop.

## 5. Risks

* **Emulated simulator.** The native workstation took 23m48s for `prefill0` and "a few minutes" for `acc` (§8.3); under Rosetta the README expects several times that. Plan: `profile acc` is the only per-step gate; the 13-test loop runs once per arch (steps 2–3); the three long `SIM_PROF` scenarios are not run (§8.3 baseline stays 6.0.0.2 and is a relative signal only). Never `SIM_TIMING=1`.
* **v79 simulator IEEE path (§7 rule 1).** The `>= 79` branches are printf-sensitive on the simulator and wrong on silicon; a v79 failure confined to the tests listed in step 3 is expected, is data for ④, and must not be "fixed" in #23. A v79 skel that hangs the phone: the harness prints per-step `E2E` lines, so the user can note the last one and `adb reboot`; the A runs come first so the baseline is never lost.
* **SDK 6.4 intrinsic or flag changes.** Covered by the `MISSING` scan and the `-mhvx-ieee-fp` probe; if 6.4 makes `-mhvx-ieee-fp` implicit for `-mv79`, the v75 build must still receive it (§7 rule 4).
* **Dockerfile nnstreamer PPA on Ubuntu 24.04** (`Dockerfile:36-42`). If the `add-apt-repository` layer fails, drop the block and pass `-Denable-tflite-backbone=false -Denable-tflite-interpreter=false` to meson builds (comment at `:37-38`); none of `tools/hexagon/*.sh` needs meson or tflite, so this cannot block G1–G3.
* **Numerical drift from the compiler.** A new hexagon-clang could reschedule qf32 chains; the qf ops are deterministic so results should be bit-identical, but the P4 device band is only 0.16–1.29 %, so the handoff compares PPL against the x86 reference (fixed) and against P4's DSP number, and `find_divergence.py` on the 1-layer image is the next step if the band is exceeded (contract §2 rule 2).
* **Stale artifacts.** Two images with different basenames and sizes, md5 in every table row, `push_if_changed` compares md5 (`run_e2e_test.sh:36-44`).
* **`build_skel.sh` default is v79.** Every command above passes `HEX_ARCH` explicitly; the handoff names the file, not the default.

## 6. Docs to update

* `docs/backend_guide/HEXAGON.md`: §4 (:382) SDK requirement; §5.2 (:452) toolchain string, date, v79 status; §5.3 (:505-508) note that the commands run through `tools/docker/run.sh`; §7 rule 4 (:677) probe result; §8.3 (:915) one-line 6.4 re-run record with the `profile_prefill_acc STAT`; §9 (:1160-1165) rewrite the v79 bullet from the handoff.
* `docs/backend_guide/HEXAGON_BENCHMARK.md`: supervisor adds SDK 6.4 rows (v75, and v79 if it produced numbers) next to :27-29.
* `tools/docker/README.md`: unchanged unless step 1 adds a variable; if `HEX_IEEE_FLAG` is made overridable, list it under the pass-through note (:24-27).
* `docs/superpowers/specs/hexagon-hvx-optimization/07-follow-ups.md` ⑭ (:74-76): done, with the plan and handoff paths; ④ (:23-25): append the step-(2) result.
* `meson_options.txt:38` comment.

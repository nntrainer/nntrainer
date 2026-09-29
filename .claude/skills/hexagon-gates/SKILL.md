---
name: hexagon-gates
description: The verification ladder for any change to the Hexagon backend (x86 reference → simulator → skel/harness compile → device handoff), with the exact container commands and what "pass" means at each rung. Use before claiming a Hexagon change works.
---

All commands run natively on the Ubuntu workstation from the repo root
after `source tools/hexagon/env.sh` (SDK 6.4.0.1, HexKL, model dir,
simulator compat libs; contract §2). Never
report a rung as passed without its pass line in the output. Rungs are
cumulative: a PR needs 0–4; a single step inside a plan needs the rung the
plan names.

**Simulator budget (contract §7).** Times below were measured under Rosetta
on the Mac; the native workstation is faster but the budget stands: on v79 (the default since #35, 6 HVX units so `workers=6`)
`profile acc` ≈ 17 min and the 13 tests ≈ 6 min; on v75 ≈ 10 / 5 min. So:

* No file under `nntrainer/tensor/hexagon/htp/`, `test/hexagon/sim_*`,
  the packer or the lowering changed → **skip rungs 2 and 3 entirely**.
  Write "no DSP bytes changed" in the PR with the proof: `git diff
  <base> --stat -- nntrainer/tensor/hexagon/htp` empty, or the md5 of each
  DSP **object** (compile every `build_skel.sh` source with `-c`) equal on
  both trees. The linked `libnntr_htp_skel.so` md5 is **not** that proof:
  hexagon-link embeds its command line with random `/tmp/<src>-xxxxxx.o`
  names and orders PLT entries after them, so two builds of one tree
  differ (#24: `64196878…` vs `431695e2…`, 11/11 objects identical).
* One op kind changed → per step run only that kind
  (`run_sim_test.sh <kind>`); run rung 2 and rung 3 **once**, right
  before opening the PR.
* Never run rung 2 or 3 per commit, never with `SIM_TIMING`, never twice
  for the same tree (reuse the log in `logs/hexagon/`).
* Rungs 2–4 run on **v79 only** (the primary arch since #35; `HEX_ARCH`
  unset = v79; the device is the S25 Ultra). v75 is not built or
  simulated per PR: run it (`HEX_ARCH=v75`, same commands) only when a
  change touches an `__HVX_ARCH__` branch in `hvx-base.h` or a kernel,
  or when the user asks, and compare against the "v75 final record" in
  HEXAGON.md §5.2 / §8.3.

## 0. Format (every commit)

```
clang-format-14 -i <changed .c .cpp .h>
```
Pass: `git diff --stat` after formatting shows only intended files.

## 1. x86 reference (no SDK; seconds)

```
./tools/hexagon/build_host_x86.sh
./build_x86_hexagon/test_lowering        # LOWER_TEST PASS
./build_x86_hexagon/test_w8cx_bin $NNTR_MODEL_DIR/w8cx/nntr_qwen3_0.6b_w8cx_DEFAULT.bin   # W8CX_BIN_TEST PASS
gcc -Wall -Werror -o /tmp/t test/hexagon/test_oplist_header.c -lm && /tmp/t
```
When the packer, lowering or `ref_ops.c` changed, also regenerate the image
and check the reference perplexity did not move unless the plan says it
should (record the number in the plan):
```
./build_x86_hexagon/nntr_hexpack $NNTR_MODEL_DIR/w8cx/nntr_qwen3_0.6b_w8cx_DEFAULT.bin build_x86_hexagon/qwen3_full
python3 tools/hexagon/make_tokens.py $NNTR_MODEL_DIR <text.txt> build_x86_hexagon/t.i32 --limit 512
./build_x86_hexagon/hexagon_ref_run build_x86_hexagon/qwen3_full --tokens build_x86_hexagon/t.i32 --eval
```
Reference values live in HEXAGON.md §5.1 (e.g. 512-token local prompt PPL
33.0195 / top1 184 after M6 P4).

## 2. Simulator, per-task gate (SDK; tens of minutes under emulation)

```
HEX_ARCH=v79 ./tools/hexagon/build_sim_test.sh
HEX_ARCH=v79 ./tools/hexagon/run_sim_test.sh profile acc
```
Pass: `SIM_TEST profile PASS` and the 8-token accuracy check within the
0.1 atol/rtol bound, with the STAT bit-identical to HEXAGON.md §8.3's
v79 record unless the plan says it moves; note the `SIM_PROF` per-kind
pcycles in the plan as a *relative* signal only (v79 `workers=6` pcycles
are not comparable with the 4-worker v75 rows).

## 3. Simulator, full (before a PR)

```
for t in smoke pool exp quant matmul matmul_dma rmsnorm rope eltwise embed attn logits graph; do
  HEX_ARCH=v79 ./tools/hexagon/run_sim_test.sh $t || break
done
```
Pass: 13 × `SIM_TEST <name> PASS`; `quant_generic` and `quant16_generic`
STAT lines within HEXAGON.md §5.2 rates (v79 values). Since #68 (#65 S0)
a build with HexKL under `$HEXKL_ADDON_ROOT` (`build_sim_test.sh` prints
`HexKL: … (HTP_HMX=1)`) also runs `run_sim_test.sh hmx` → `SIM_TEST hmx
PASS`; the plain v79 simulator core executes HMX, so its STATs are real
gates (`acc_layout usable=1`, every `hmx_mm_*` STAT `max_abs=0`,
`wh_reloc bytes_same=1`). Without HexKL the test returns 2 with
`SKIP: built without HTP_HMX` and is not counted. A change that touches
only `hmx/` or `test_hmx.c` runs `hmx` per step and the 13 + `hmx` once
before the PR like any other kind. The "no DSP bytes changed" object
proof is `tools/hexagon/dsp_obj_md5.sh <tree> [out]` on both trees
(same `HEX_EXTRA_CFLAGS`; add `-ffile-prefix-map=<other tree>=/work`
when the other tree is a worktree, because FARF embeds `__FILE__`).
A v75 repeat (only
per the budget bullet) needs its own `HEX_ARCH=v75 build_sim_test.sh`
first — the two builds share `build_hexagon/sim/` and `run_sim_test.sh`
refuses to run a library built for another arch or with other
`HEX_EXTRA_CFLAGS` (the stamp carries both).

## 4. Skel and host harness compile (SDK; no device. The host harness needs an Android NDK, which the workstation does not have yet: skip it and say so)

```
./tools/hexagon/build_skel.sh                   # build_hexagon/skel/libnntr_htp_skel.so (v79, shipping)
./tools/hexagon/build_host_test.sh              # build_hexagon/host/{hexagon_rpc_test,hexagon_e2e_test}
```
Pass: both artifacts exist; record `md5sum` of each for the handoff. For
variants use `HEX_EXTRA_CFLAGS=-D...` and copy each skel to
`build_hexagon/skel/libnntr_htp_skel.<variant>.so` before the next build.

## 5. Device (user only)

Never run here. Write a handoff (`hexagon-handoff` skill), set
`state:needs-measurement`, stop. Performance conclusions are drawn only
from filled handoff tables; simulator cycle counts never decide a
performance question (plan §2, gap rule 1).

## Kernel review list (HEXAGON.md §7, apply to any hvx-*.c change)

* qf16/qf32 format ops only; no `Vsf`/`Vhf` IEEE arithmetic paths on the
  device build.
* Vector quantizer: tie row and zero row byte-identical to `ref_quant_row`
  / `ref_quant_row_i16`; ±1 LSB rate within the §5.2 bound.
* Integer paths (W8A8, LOGITS, EMBED, int16 down_proj accumulate) stay
  bit-exact against `ref_ops.c`.
* 128-byte alignment of every DMA/VTCM buffer; worker count from
  `wp_create`, never hard-coded.
* Any ABI/layout change bumps the version in `nntr_htp_common.h`, rejects
  old images in `.hexcfg`, and updates HEXAGON.md §1.4/§2 in the same PR.

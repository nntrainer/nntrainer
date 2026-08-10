# START HERE — MHA-core on Hexagon HTP

**This branch (`htp/attention-handoff`) is a working reference, not a PR.** It is
never merged upstream. It exists so that a fresh session — a person or an agent —
can pick up the attention-on-HTP work without re-deriving a month of measurements.

Read this file first, then §"What to read next" tells you which of the others
apply to your task.

---

## 1. What this work is

`Applications/CausalLM/layers/mha_core.cpp` runs multi-head attention on the ARM
CPU. The goal is to run its two matmuls — `Q·Kᵀ` and `P·V` — plus softmax on the
Hexagon DSP (HMX for the matmuls, HVX for softmax), in **one FastRPC call per
attention layer**, and then to make that call cheaper with flash-style KV
streaming and, later, block sparsity.

Three things are already true and measured on device (Galaxy S25 Ultra, V79,
`R3CY10WM83Y`). Do not re-derive them:

| | |
| :-- | :-- |
| the DSP beats the CPU on both attention matmuls | 2.4–3.4× at decode, **43–65× at prefill** (`ref_14` §3) |
| the win is **data movement**, not HMX | HMX `mm_f16` is 1.7% of a decode call; DMA row size and DDR-vs-VTCM are the whole game (`ref_14` §5) |
| a u8i4 / u8i8 integer matmul path already ships in this tree | weight registry, WH bake, DMA ring with cross-block prefetch, HVX quant/dequant, QuRT worker pool — all device-verified (PR #4243/#4244/#4249) |

## 2. The two decisions that shape everything

**(a) One FastRPC call per layer, so the three stages must be fused.**
FastRPC costs ~404 µs fixed per call. An unfused seam (matmul → CPU softmax →
matmul) pays that three times per layer. This **inverts** `ref_08` §8 and
`ref_14` §1, which concluded fusion was unnecessary — those measurements were
taken inside a standalone DSP `main()` with no FastRPC in the timed region at
all (`ref_14` §7 says so). Softmax therefore runs on the DSP, on HVX.

**(b) The matmuls use the integer path (u8 activation × i4/i8 weight), not fp16.**
This is a later decision than `10_mha_htp_plan.md` was written under; where the
two disagree, **`11_u8_task_split.md` wins**. The reason is (a) above plus
reuse: the fp16 attention path exists only as a bench on another branch, while
the integer path is in this tree and verified. Halving (i8) or quartering (i4)
the KV bytes also cuts the dominant cost directly.

Everything else in `10_mha_htp_plan.md` — the transpose-free formulation, the
VTCM budget, the sparsity level, the FastRPC arithmetic, the verification
strategy — carries over unchanged to the integer path.

## 3. Where the work stands

| piece | state |
| :-- | :-- |
| u8i4 layer endpoint, DMA ring, cross-matmul prefetch | shipped, device-verified — PR #4243 |
| u8i8 mirror | shipped, device-verified — PR #4244 |
| quant/dequant path optimisation (async output DMA, vectorised quant, QuRT worker pool) | shipped, device-verified — PR #4249 |
| HVX f32 softmax + vector `exp` | **PR #4245, another contributor's**, contains #4243's commits plus the softmax. Not on this branch |
| KV block quantizer (task 1 of the split) | drafted on `htp/u8i8-dma-cross` at `f901e6f6d`, **two known defects — see §5** |
| everything else (tasks 2–11) | not started |

**Task 0 of `11_u8_task_split.md` is still open**: rebasing the u8i8 work onto
`pr4245` so the HVX softmax and the u8 matmul share one skel. It conflicts in
`test/htp/build.sh` (both sides add sources to `$SRCS`) — a small, real merge,
but it needs a device run afterwards, so it was not folded into this branch.

## 4. What to read next

| you are doing | read |
| :-- | :-- |
| **the flash-attention task** | `30_flash_attention_task.md` — self-contained, start there |
| implementing any task from the split | `11_u8_task_split.md` §1 (design) then your task in §3 |
| prompting an agent to do one | `12_prompt_kit.md` |
| the architecture, VTCM budget, sparsity, verification strategy | `10_mha_htp_plan.md` |
| why a HexKL micro function is slow, or a DMA descriptor shape | `ref_14` §5 — the measured rules |
| the RM / AH / WH layouts | `ref_08` §3 |
| branch layout, device recipe, environment gotchas | `13_htp_pr_plan.md` |
| how to work here | `01_working_style.md` |

`ref_*.md` are copies of `docs/backend_guide/htp_backend/*` from branch
`claude/hexkl-mha-hmx-optimization-6ycsx0`, brought here so this branch is
self-contained. They are historical records: where they disagree with
`10_`/`11_`, the newer document says so explicitly and wins.

## 5. Known defects in the drafted Task 1 (`f901e6f6d`, other branch)

The quantization arithmetic is correct — including the one thing most likely to
be wrong, `V`'s per-column scale computed over that block's rows only. The host
test is genuine (144 combinations, bound derived rather than hardcoded, `colsum`
independently recomputed, tail poisoning). It passes.

Two defects, both about placement rather than arithmetic:

1. **It is `hexkl_kv_quant.cpp`, and it calls `nntrainer::compute_fp16_to_fp32`
   from `fp16.h`.** That function is C++-namespaced and defined in
   libnntrainer. The consumer of this file is the DSP skel, built by
   `test/htp/build.sh` with `hexagon-clang` as **C**, with no C++ runtime and no
   libnntrainer in the link. As written the file can never enter the skel — it
   compiles only because the gtest is currently its only consumer. Fix: rename
   to `.c`, replace the fp16 decode with a self-contained `static inline` bit
   conversion, `<cmath>`/`<cstring>` → `<math.h>`/`<string.h>`, and add it to
   `test/htp/build.sh`'s `$SRCS`. The `extern "C"` guard in the header is
   already correct.
2. **It was committed onto `htp/u8i8-dma-cross`, which is PR #4244's branch.**
   New work needs its own branch created *before* the first commit.

## 6. The one process rule that has repeatedly paid off here

**Measure the breakdown before acting on a hypothesis.** The FastRPC
investigation's first hypothesis (marshalling dominates) measured 16–32%; the
real cost was a scalar accumulator copy-out at 92% of DSP-internal time, and it
was found by adding per-stage timing rather than by guessing a second time.
Every stage of this plan asks for a per-stage breakdown for that reason.

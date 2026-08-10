# Attention on HMX — Design Record

Status: **analysis superseded by measurement.** This document was written
when the attention path looked entangled with HVX work owned by someone else
(the softmax kernels) and Option B (the micro API) looked theoretical. Both
premises turned out false: [14](14_mha_fp16_micro_verified.md) measures both
`mha_core` matmuls on the micro API, on device, and finds the softmax round
trip costs ~1% of the total — not gating. **Read 14 first.** This document is
kept for the parts of the analysis 14 does not repeat (RM/AH/WH mechanics,
the beta2 migration notes, §7's llama.cpp survey) — every claim 14
contradicts is marked inline below rather than silently removed.

---

## 1. What is being accelerated

`Applications/CausalLM/layers/mha_core.cpp` is the CPU-optimised attention
core. The decode path is `one_batch_incremental_forwarding`
(`mha_core.cpp:693`), three stages:

```
compute_kcaches(...)              // Q · Kᵀ   -> scores
softmax_triangle(...)             // softmax over the causal triangle
compute_fp16vcache_transposed(...)// P · V    -> attention output
```

`out_` is a `(1, 1, packed_triangular_len, num_heads_Q)` tensor: the scores are
**triangular-packed and head-minor**. That layout is what makes the CPU kernels
cheap and is also the single biggest obstacle to handing a stage to a GEMM, which
wants a dense `[M, K] × [K, N]`.

### The FP16 fact that changes everything

`mha_core.cpp:346` is guarded by `#if ENABLE_FP16 && defined(__ANDROID__)`, and
`package_android.sh:76` hardcodes `-Denable-fp16=true`. **On device, Q, K, V and
the scores are all `_FP16` already.** Any plan written against the FP32 reading
of this file is wrong. In particular there is no conversion to save — the FP16
operands are what the HMX fp16 kernel wants.

The KV cache is separately hardcoded to FP16/UINT16 at `mha_core.cpp:233-245`;
it is not driven by `fc_layer_dtype`. So the attention "weight" operand is
always fp16 regardless of how the FC layers are quantised.

---

## 2. Which stage is actually the easy one

The first instinct — that `P · V` is harder than `Q · Kᵀ` — is backwards.

| stage | M | N | K | N % 32 |
| :-- | :-- | :-- | :-- | :-- |
| `Q · Kᵀ` | q_len | **kv_len** | head_dim | varies with the cache position — needs padding |
| `P · V`  | q_len | **head_dim = 128** | kv_len | always aligned |

SDKL requires `N % 32 == 0`. `P · V` satisfies it for free because `N` is
`head_dim`. `Q · Kᵀ` has `N = kv_len`, which changes every token, so it needs
padding on every call. **`P · V` is the stage to do first** — true for the
**macro** API this section is reasoning about.

**Reversed for the micro API — [14](14_mha_fp16_micro_verified.md) §5.7.** Once
you own the tile loop, SDKL's `N % 32` constraint does not apply, and the real
considerations point the other way: written as `Sᵀ = K·Qᵀ`, `Q·Kᵀ` needs no
transpose, its output *is* the existing scores layout, its accumulation depth
(K=128) carries none of `P·V`'s fp16-accumulation risk, and it depends on
nothing from the softmax owner. Do `Q·Kᵀ` first on the micro API.

### The GQA stride problem

The cache is laid out `[kv_position][kv_head][head_dim]`. A single kv_head is
therefore *strided*, not contiguous (see `neon_impl.cpp:2431`). SDKL wants a
contiguous `[N, K]`, and a `sdkl_npu_mm_*` call cannot express the stride, so
under the **macro** API a gather/pack step is unavoidable.

**This is wrong for the micro API — [14](14_mha_fp16_micro_verified.md) §5.4.**
A single 2D DMA descriptor (`row_size=head_dim*2, src_stride=nch*head_dim*2,
nrows=kv_len`) reads the strided cache directly; the "gather" is a DMA
parameter, not a compute step, and doc 14 measured a whole layer's K cache
(all kv_heads) moved in one such descriptor at 53 GB/s.

---

## 3. RM, AH and WH

Three layouts, not a chain. This was the most-repeated point of confusion, so it
is stated plainly:

```
            rm_to_ah                        rm_to_wh
   RM  ─────────────────►  AH        RM ─────────────────►  WH
   (row-major, "flat")   (activation)      (weights, HMX-tiled)
        ▲                    │
        └──── ah_to_rm ──────┘
```

- **RM** — plain row-major. What the rest of nntrainer holds.
- **AH** — *Activation*-HMX. The tiling the HMX unit wants for the **left**
  operand.
- **WH** — *Weights*-HMX. The tiling for the **right** operand. 32×32 tiles.

RM → AH and RM → WH are two **parallel** conversions from the same source.
AH is not an intermediate on the way to WH, and **there is no `ah_to_wh`
function** in any addon revision. A result that comes out of the accumulator as
AH cannot be re-used as a weight without going back through RM.

Whether an AH accumulator output can be fed straight back in as an AH
*activation* is exactly what probe (5) in
`test/unittest/jni_htp/hexagon/hexkl_layout_probe.c` measures — see that
directory's README. If it can, a score matrix could stay in VTCM between
`Q · Kᵀ` and `P · V`; if it cannot, every stage boundary costs a round trip.

---

## 4. HexKL 1.0.0-beta2

Obtained through `qpm-cli` (the public Software Center channel only carries
beta1; beta2 needs a signed Product Kit License Agreement). Relevant changes:

| change | why it matters here |
| :-- | :-- |
| **`sdkl_npu_mm_u8i8` and `sdkl_npu_mm_u8i4`** (AH-native) | **new in beta2** — beta1 shipped only the `_i32` row-major forms. A fused attention chain needs these; under beta1 it was not expressible at all. |
| `sdkl_npu_mm_u8i4_i32` now "accepts arbitrary (unaligned) dimensions and handles X layout conversion and output padding internally" | **New sentence, but not new behaviour.** beta1's own sample for this kernel already passes `n_row` unpadded while rounding `n_col` and `n_inner` up to 32, so beta2 documented what was already true. `hexkl_mm.cpp`'s `Mp = 64` looks copied from the u8i8 wrapper rather than verified. See [09](09_lmhead_u8i4_plan.md) §4. |
| `sdkl_npu_mm_f16f16_f16` | fp16 in, fp16 out. Removes *all* dtype conversion from the attention path, which is fp16 end to end on device. |
| `sdkl_npu_get_hw_info` / `sdkl_npu_hw_info_t` | new. Reports `vtcm_size`, `num_hvx_units`, `hmx_fp16_rate`. `hexkl_pin_probe.c` calls it, so that probe is beta2-only. |
| New AH converters: `sdkl_cpu_u8_rm_to_u8_ah(_inplace)`, `sdkl_cpu_f32_rm_to_f16_ah`, `sdkl_cpu_f16_ah_to_f32_rm`, non-inplace `sdkl_cpu_f16_rm_to_f16_ah` | new. Needed to feed the AH-native kernels above. |
| Header now `#include "remote.h"` itself | the include-order workaround at `hexkl_mm.cpp:17` becomes unnecessary. |
| Layout helpers renamed to `<dtype>_rm_to_<dtype>_<layout>` | beta1 spellings do not compile — see the migration table below. |
| Micro API shipped (`hexkl_micro.h`, `libhexkl_micro.a`) | makes the hand-written-DSP-kernel option real rather than theoretical. |
| `hexkl_micro_hw_init` takes **three** args | `(vtcm_base, vtcm_size, hmx_fp16_rate)`. beta1 probes pass two. |

`sdkl_npu_init_config_t` is **not** new and is not a lever: it is an empty
struct in both revisions, "reserved for future use".

### Migrating this tree from beta1 to beta2

Every SDKL layout call in the tree still uses the beta1 spelling. The
signatures are unchanged, so this is a mechanical rename:

| beta1 (what is in the tree) | beta2 | callers |
| :-- | :-- | :-- |
| `sdkl_cpu_rm_to_wh_f16_inplace` | `sdkl_cpu_f16_rm_to_f16_wh_inplace` | `hexkl_mm.cpp`, `layer_devel.h`, `sdkl_npu_probe.cpp`, `unittest_nntrainer_htp_kernels.cpp` |
| `sdkl_cpu_rm_to_wh_i8_inplace` | `sdkl_cpu_i8_rm_to_i8_wh_inplace` | `quantizer.cpp`, both htp unittests |
| `sdkl_cpu_rm_to_wh_i4` | `sdkl_cpu_i4_rm_to_i4_wh` | `quantizer.cpp` (u8i4 branch) |
| `sdkl_cpu_rm_to_ah_f16_inplace` | `sdkl_cpu_f16_rm_to_f16_ah_inplace` | — |
| `sdkl_cpu_ah_to_rm_f16_inplace` | `sdkl_cpu_f16_ah_to_f16_rm_inplace` | — |
| `sdkl_cpu_ui8i8_ah_to_i32_rm` | `sdkl_cpu_i32_ah_to_i32_rm` (+ `_inplace`) | — |
| `sdkl_cpu_ui8i4_ah_to_i32_rm` | **removed** | — |
| `sdkl_cpu_rm_to_wh_i8` | `sdkl_cpu_u8_rm_to_u8_ah` | — |

Two of those rows are more than renames:

- **`sdkl_cpu_rm_to_wh_i8` was misnamed in beta1.** Its parameters are
  `(n_inner, n_row, X_i8_cpu, Xq)` — an *activation*, not a weight — and beta2
  renames it to `sdkl_cpu_u8_rm_to_u8_ah`, i.e. it always produced AH, not WH.
  Nothing in this tree calls it (only the `_inplace` weight variant), so there
  is no latent bug, but do not reach for it expecting a WH converter.
- **The u8i4-specific AH→RM converter is gone.** beta1 had
  `sdkl_cpu_ui8i4_ah_to_i32_rm`; beta2 keeps only `sdkl_cpu_i32_ah_to_i32_rm`,
  documented as the u8i8 variant. If the AH-native `sdkl_npu_mm_u8i4` is used,
  whether that one reads its output correctly is an open question.

### The naming is about layout, not output dtype

`sdkl_npu_mm_u8i4` and `sdkl_npu_mm_u8i4_i32` both produce int32. The suffix
marks the **layout contract**, and the same split runs through the fp16
kernels:

| kernel | X and A layout | who converts |
| :-- | :-- | :-- |
| `sdkl_npu_mm_u8i4` | **AH** | the caller |
| `sdkl_npu_mm_u8i4_i32` | **row-major** | the kernel, internally |
| `sdkl_npu_mm_f16` | **AH** | the caller |
| `sdkl_npu_mm_f16f16_f16` | **row-major** | the kernel, internally |

The unsuffixed forms are the HMX-native ones — "the ideal kernel … assuming the
caller handles layout and type preparation". The suffixed forms are the
convenience wrappers.

This is the seam that matters for attention: a fused `Q·Kᵀ → softmax → P·V`
that keeps the score matrix on the DSP needs the **AH-native** kernels, since
the suffixed ones round-trip through row-major on every call. It is also why
probe (5) in `hexkl_layout_probe.c` — is the accumulator's AH the same AH that
`rm_to_ah` produces — decides whether that fusion is reachable at all.

`sdkl_npu_mm_u8i4_i32` is also the only kernel taking `size_t` dimensions
rather than `int` — true in beta1 as well, so it has been the odd one out for a
while. What is new in beta2 is the unaligned-dimension guarantee. See
[09](09_lmhead_u8i4_plan.md) §4 for what that changes.

---

## 5. Two implementation options

### Option A — macro API (`sdkl_npu_mm_*`)

Treat HexKL as a black-box GEMM. Host prepares operands, calls the kernel.

- Cheap to build; reuses everything in `hexkl_mm.cpp`.
- Every stage boundary is a host round trip: pack → convert → call → read back.
- The score matrix cannot stay on the DSP between `Q · Kᵀ` and `P · V`.

### Option B — micro API (`hexkl_micro_*`)

Write the attention kernel as a DSP-side program: load tiles into VTCM, issue
HMX ops, keep the intermediate resident.

- The only way to fuse the three stages and keep the scores in VTCM.
- Requires a Hexagon toolchain path nntrainer does not have today, and the
  softmax between the two matmuls is HVX work owned by someone else.

**Decision, superseded — [14](14_mha_fp16_micro_verified.md):** this section's
premise (micro is theoretical, softmax fusion is gating) is measured false.
Option B was built, runs on device, beats the CPU 2.4-65× depending on shape,
and does **not** need the fused/VTCM-resident softmax to win — the unfused
round trip costs ~1% of the total (§6/§7 below still describe a real, optional
follow-up, not a blocker).

---

## 6. The WH KV-cache question

A WH-baked KV cache resident in NPU memory is not possible: at the sequence
lengths of interest the cache is ~224 MB against an NPU weight budget measured
in tens of MB (see [09](09_lmhead_u8i4_plan.md) §3 — the same budget question).

The workable shape is a **WH cache in ordinary host RAM**, memcpy'd into the NPU
buffer per call. The point is that `rm_to_wh` is the expensive part (~2.7 ms for
the shapes measured) and the memcpy is not (~22 µs); keeping the *converted*
bytes around means only the cheap half is paid per token. Only the newly
appended KV entry needs converting each step.

**Confirmed, not just designed — [14](14_mha_fp16_micro_verified.md) §5.5.**
V's full re-bake measured 123.7 µs/head; the append-time incremental bake (the
newly-dirtied k-tile only) measured 3.35 µs/head — a 32× gap, and the entire
reason decode P·V beats the CPU.

---

## 7. What transfers from llama.cpp's `ggml-hexagon/htp`

Reviewed for adoptable infrastructure. The conclusion is narrower than it first
looks: llama.cpp writes its **own** DSP kernels, so its DSP-side machinery
(work-queue, dma-queue, hmx-queue, VTCM layout builders) has nothing to attach
to while nntrainer calls a black-box vendor GEMM.

What does transfer, all host-side:

- **GQA row-folding** — fold the `gqa_size` Q heads that share a kv_head into
  the `M` dimension of one GEMM instead of issuing `gqa_size` separate calls.
  This is the single largest structural win available and needs no DSP code.
- **Profitability gates** — llama.cpp checks shape before dispatching to the
  DSP at all. nntrainer needs the same: at short `kv_len` the pack cost exceeds
  the GEMM gain.
- **Tile planning ahead of dispatch** rather than inside the call.
- **`dspqueue` batching** — amortising FastRPC round trips across calls.
- **Per-stage instrumentation** — llama.cpp's counters are the model for the
  `MmProfile` breakdown already present in `hexkl_mm.cpp`.

---

## 8. The HVX owner — one optional ask, not a blocker

This section originally read as a blocking checklist. **It is not one —
[14](14_mha_fp16_micro_verified.md) §7 measured the unfused round trip to CPU
softmax at ~1% of the per-layer total.** Only one item survives as worth
raising, and it is optional:

1. Whether the softmax entry point can write `P` as `[gqa, kv]` instead of
   the current triangular-packed, head-minor `[kv, gqa]` — saves ~7 µs/head
   of DSP-side transpose if yes, costs a measured ~1-2% of the per-layer total
   either way if no (doc 14 §6).

Items 2-5 of the original checklist (fp16 in/out, streaming softmax,
mask placement, VTCM-resident softmax) are now moot for the path doc 14
verified: it does not fuse the three stages, so none of them gate it. They
remain relevant only if a *fused* design is revisited later, which doc 14 §1
found unnecessary for the measured win.

---

## 9. Implementation order — superseded

This order assumed a fused single op was necessary (step 1) and P·V-first
(step 2). Both are corrected by [14](14_mha_fp16_micro_verified.md): the seam
does **not** need to be a single fused `attn_forward` — three ops (or two,
since softmax stays CPU-side either way) cost only the ~1% round-trip §8
measured, and `Q·Kᵀ` goes first. The real next step is **wiring the kernels
doc 14 verified into a `ComputeOps` seam and solving the persistent-skel /
FastRPC problem doc 14 §7 flags** — that is tracked as separate architecture
work, not as an update to this list.

---

## 10. Probes in this tree

| probe | side | question |
| :-- | :-- | :-- |
| `test/unittest/jni_htp/hexkl_pin_probe.c` | host (ARM) | how much NPU memory can stay resident |
| `test/unittest/jni_htp/hexagon/hexkl_layout_probe.c` | DSP | what RM/AH/WH actually permute to, read off the hardware |

### The fp16 gate is cleared

`hmx_fp16_rate` was the precondition on everything above: zero there means the
device has no HMX fp16 and no `mm_f16`-based plan is worth measuring.
`sdkl_npu_get_hw_info` reports it, and `hexkl_gemv_probe` prints it on the way
past. Galaxy S25 Ultra, `1_0_56_beta.2_HEXAGON_V79`:

```
[HW_INFO] Hexagon Architecture Version: 35961     (0x8C79 -> ISA 0x79 = V79)
[HW_INFO] HMX FP16 Rate (ops/cycle): 8
[HW_INFO] Number of HVX Units: 6
[HW_INFO] VTCM total size (bytes): 18874368       (18 MiB)
```

**8 ops/cycle, so HMX fp16 exists.** The direction is open.

**VTCM is 18 MiB** per `sdkl_npu_get_hw_info` — **but `hexkl_micro_hw_init`
reports 8 MiB, confirmed across every run of both the MHA and the FC micro
benches** ([14](14_mha_fp16_micro_verified.md) §7). The two APIs disagree and
this is unresolved; use 8 MiB for any VTCM budget check until reconciled. It
did not change any conclusion in doc 14 (everything measured fit
comfortably), but it would matter for a larger fused design, and it means the
74 MiB lm_head comparison in §6 above should be read against 8 MiB, not 18.

The DSP-side layout probe is still the one that answers §3 — what RM, AH and WH
actually permute to, and whether the accumulator's AH matches `rm_to_ah`'s.

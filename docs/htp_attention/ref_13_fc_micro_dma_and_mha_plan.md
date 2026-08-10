# 13 — FC micro+DMA (verified) and the MHA-core micro-mm plan

State of the work as of branch `claude/hexkl-mha-hmx-optimization-6ycsx0`.
Read [08](08_attention_hmx_design.md) first — this document **updates its
§5 "Decision: A first" and §9 implementation order** with a measured result,
and sets up the next task: putting `Q·Kᵀ` and `P·V` in `mha_core.cpp` on the
micro API.

**§5 below is superseded — [14](14_mha_fp16_micro_verified.md) is the
measured result of the plan this section proposed.** `Q·Kᵀ` and `P·V` were
both built and measured; the ordering below (P·V first) and §4's "cross-matmul
prefetch does not apply to attention" are both corrected there. Kept for the
FC-verified numbers in §1-§3 and §6-§7, which still stand.

---

## 1. One-paragraph summary

The FC matmul was rebuilt on the **micro API** (`hexkl_micro_hmx_mm_u8i4_i32`)
plus a hand-written Hexagon **user-DMA** stager (raw `dmstart/dmlink/dmpoll`,
lifted from llama.cpp `ggml-hexagon/dma-queue`), with **cross-matmul weight
prefetch**: matmul *i+1*'s weight is DMA'd DDR→VTCM while matmul *i* computes.
On a Galaxy S25 Ultra (V79) this **beats the shipped SDKL kernel 1.7–2.0×** for
u8i4 FC, verified on device, correctness OK vs a scalar integer reference. This
overturns doc 08 §5's "macro (Option A) first" and doc 07's "micro is a losing
bet": **Option B is proven reachable and now has a working tile+DMA harness to
reuse for attention.** The single caveat is that the win needs a *persistent
skel* (sees the matmul sequence) — a per-call/stateless kernel like SDKL
structurally cannot prefetch across calls, which is exactly why it loses.

---

## 2. What was built and verified

Bench + one-shot runner (build on-DSP via beta2 addon example + hexagon-clang
v79, run via `run_main_on_hexagon`, timed with `HAP_perf_get_time_us`):

- `test/unittest/jni_htp/hexagon/hexkl_micro_fc_bench.c`
- `test/unittest/jni_htp/hexagon/run_micro_fc_bench.sh` — sources SDK env,
  builds, pushes, runs, and prints an **`hexkl_fc_compare`-format** report.

Three kernels are measured per shape, all **on-DSP, host quant/dequant excluded
from both sides** (that conversion is the HVX owner's scope and is identical for
ours and SDKL — same u8i4 recipe):

| kernel | what it is |
| :-- | :-- |
| **cross-matmul (weight prefetch)** | KMM=6 pipeline; next weight staged during current compute. **The result.** |
| single micro+DMA | one matmul, weight NOT hidden (no cross-call knowledge). |
| SDKL full row-major | `hexkl_macro` kernel on-DSP — the bar to beat. |

### Verified numbers (on-DSP, µs, quant/dequant excluded both sides)

| shape | M | N | K | **cross-matmul** | single | SDKL on-DSP | speedup |
| :-- | :-- | :-- | :-- | --: | --: | --: | --: |
| medium | 64 | 1024 | 1024 | **28.5** | 60.5 | 48.5 | 1.7× |
| q_proj | 64 | 2048 | 1024 | **56.3** | 105  | 107  | 1.9× |
| ffn_up | 64 | 3072 | 1024 | **84.5** | 150  | 173  | 2.0× |

q_proj cross 3-run stable (~56.3/56.5/56.8). All shapes correctness=OK.

### fc_compare-format comparison (q_proj), with host + QNN reference

The NPU-µs column is the apples-to-apples axis. Host NEON (76.7 µs) is a
**reference** from the SDKL `hexkl_fc_compare` run (identical for ours & SDKL);
SDKL-production 270/350.5 and QNN 69.9 are references (QNN from doc 06).

| method | relErr | NPU µs | host NEON µs | per-call µs | engine |
| :-- | :-- | --: | --: | --: | :-- |
| **OURS cross-matmul (micro+DMA)** | 0.07207 | **56.3** | 76.7 | **133.0** | micro+DMA skel |
| SDKL production (`sdkl_npu_mm`) | 0.07207 | 270.0 | 76.7 | 350.5 | NPU/HMX |
| SDKL on-DSP kernel (no per-op RPC) | 0.07207 | 107.8 | 76.7 | 184.5 | NPU/HMX |
| QNN (AI Engine Direct, doc 06) | ~ | 69.9 | (uint8 i/o) | — | QNN |

**NPU kernel only: ours 56.3 < QNN 69.9 < SDKL on-DSP 107.8.** SDKL
production's 270 includes per-op FastRPC (~97 µs); ours assumes a persistent
skel with `dspqueue` batching, so no per-op RPC. Do **not** compare the QNN host
column — QNN is uint8 i/o, no F32↔ conversion. Override the reference constants
via env: `HOST_NEON_US= SDKL_PROD_NPU_US= SDKL_PROD_TOT_US= QNN_FC_US=`.

---

## 3. Why it wins, and the DMA facts that were paid for in bugs

- **Cross-matmul prefetch is the whole win.** SDKL (`sdkl_npu_mm`) is
  per-call/stateless — it re-stages the weight each call and cannot see, let
  alone prefetch, the next matmul's weight. A persistent skel prefetches
  matmul *i+1*'s weight (DDR→VTCM, one async DMA) during matmul *i*'s ~57 µs
  compute, so the ~29 µs weight staging hides behind compute. Single-matmul
  (no cross-call knowledge) is only **parity** with SDKL (108 vs 107).
- **The optimised half is the read-*out*, not the read-*in*.** The output DMA
  (HMX accumulator → VTCM → DDR) is pipelined against the next tile's compute.
  Activation/weight read-in is the part hidden by cross-matmul prefetch.
- **HW landmine:** a single-row DMA with `row_size > ~512 KB` **silently
  corrupts** on this device. Use `nrows`-based descriptors (`row_size ≤ 256 KB`).
  This masked earlier bugs (a "fast 28 µs" weight DMA that wasn't transferring).
- **Two bugs found+fixed** (Opus-plan / Sonnet-execute):
  1. `plan_vtcm(M,N,K)` arg swap in `run_pipelined_layer` (every other caller
     uses `(M,K,N)`) → K over-accumulated by N/K. Invisible when N==K.
  2. "146 µs slowdown" was a **measurement artifact** — the correctness
     `diff_index_i32(M*N)` sat inside the timed window on the last iter. Real
     cost ~57 µs.

Tile constants (u8i4 HMX): `N_ROW=64, N_COL=32, N_INNER=32`; weight tile 512 B,
accumulator tile 8192 B (64×32 int32). Sequence: DMA weight→VTCM (prefetched) →
`hexkl_micro_hmx_mm_u8i4` per tile → `acc_read_int32` → output DMA VTCM→DDR
(pipelined).

---

## 3a. u8i8 measured against u8i4 (2026-08-04, same bench, now dtype-parametrised)

`hexkl_micro_fc_bench.c` now threads a `dtype_ops` descriptor through one code
path instead of forking a second file — u8i4 and u8i8 share the exact tile
geometry (64×32 activation × 32×32 weight → 64×32 int32 accumulator); the only
delta is weight-tile bytes (512 vs 1024) and three symbol names
(`hexkl_micro_hmx_mm_u8i4`/`_u8i8`, `rm_to_wh_i4`/`_i8`,
`hexkl_macro_mm_u8i4`/`_u8i8`). u8i4's numbers were re-verified against this
document's table (±10%, correctness OK) before trusting the u8i8 comparison.

| shape | u8i4 cross-matmul | u8i8 cross-matmul | weight-DMA GB/s (both) |
| :-- | --: | --: | --: |
| medium | 29.0-29.5 µs | 32.3-33.9 µs | ~39 |
| q_proj | 57.8-60.5 µs | 59.5-64.0 µs | ~40 |
| ffn_up | 90.3-90.5 µs | 91.5-96.0 µs | ~41 |

u8i8 still beats SDKL 1.7-2.0×, same order as u8i4. The gap between the two
dtypes is small in the **cross-matmul** (prefetched) column specifically
because weight DMA is already mostly hidden behind compute for both — the
mm op count and compute time are identical (HMX consumes one 32×32 tile per
call regardless of its storage width); only the **`single`** (no-prefetch)
column shows the real 2× weight-byte difference cleanly: u8i4 106.5,
u8i8 131.3 µs at q_proj (~19% gap, growing with shape size). Do not read the
small cross-matmul gap as "packing doesn't matter" — it means prefetch is
doing its job on both dtypes; the packing effect is real and is what the
`single` column shows.

**A pre-existing, unrelated bug surfaced while doing this: the SDKL-macro
correctness check in this bench is unreliable** (§7 restates this; see doc 14
§7 for the fix needed — `qurt_mem_cache_clean()` around the macro calls). It
predates this session's changes and does not affect the timing numbers above.

**VTCM confirmed 8 MiB** (`hexkl_micro_hw_init`, both dtypes, reproducible) —
see doc 14 §7 for the disagreement with `sdkl_npu_get_hw_info`'s 18 MiB.

---

## 4. What this changes for attention (updates doc 08)

doc 08 §5 chose **Option A (macro) first, Option B (micro) only if stage-
boundary cost justifies it**, because the micro path was "theoretical" and
needed a toolchain nntrainer lacked. **Both premises are now false:** the micro
path builds and runs on device (this bench), beats SDKL, and there is reusable
tile+DMA infrastructure. So for attention the recommendation flips: **go
straight to Option B (micro), reusing this harness**, because the entire point
of attention fusion (keep the score matrix in VTCM across `Q·Kᵀ → softmax →
P·V`) is *only* expressible in the micro API — the macro/SDKL path round-trips
through row-major host memory at every stage boundary (doc 08 §5, §3).

**But note the difference in where the win comes from:**

| | FC (done) | Attention (next) |
| :-- | :-- | :-- |
| right operand | **static weight** — same every token | **activation** (K, then V cache) — changes every token |
| the reusable trick | cross-matmul **weight prefetch** | retargeted, not absent — [14](14_mha_fp16_micro_verified.md) §5.6: prefetch the *next kv_head's* KV slab within one call, instead of the next matmul's weight across calls |
| the actual win | hide weight staging behind compute | measured to be **DMA-descriptor-geometry redesign**, not fusion — doc 14 §1/§7: the unfused round trip to CPU softmax cost only ~1% of the total, so keeping scores VTCM-resident across all 3 stages was not the win after all |
| dtype | u8i4 (`_u8i4_i32`) | **fp16** (`hexkl_micro_hmx_mm_f16` / `hexkl_micro_matmul_f16f16_f32`) |

So the FC harness contributes its **tile loop + user-DMA stager + acc_read +
pipelined output DMA**, but the attention kernel is a *fused* program, not a
prefetch pipeline. The fp16 micro ops exist and are confirmed present in beta2
(`hexkl_micro_hmx_mm_f16`, `..._acc_read_f16`, `..._rm_to_ah_f16`,
`..._rm_to_wh_f16`, `hexkl_micro_matmul_f16f16_f32`). HMX fp16 rate = 8 on this
device (doc 08 §10), so the fp16 path is real.

---

## 5. The MHA-core plan (next session)

Target file: `Applications/CausalLM/layers/mha_core.cpp`. Decode path is
`one_batch_incremental_forwarding` (`mha_core.cpp:693`), three stages:

```
compute_kcaches(...)               // Q · Kᵀ   -> scores   (mha_core.cpp:577)
softmax_triangle(...)              // softmax   (HVX owner's — leave as CPU)
compute_fp16vcache_transposed(...) // P · V     -> output   (mha_core.cpp:1381)
```

FP32 reference kernels to match bit-for-bit:
`compute_kcaches_fp32_reference` (`mha_core.cpp:42`) and
`compute_vcache_fp32_transposed_reference` (`mha_core.cpp:73`).

### Order — corrected, do `Q·Kᵀ` first

This subsection originally said "do P·V first — doc 08 §2", reasoning from
SDKL's macro-API `N % 32 == 0` constraint. **That constraint does not apply
once you own the tile loop (the micro API), and doing `Q·Kᵀ` first turned out
right for different reasons** — [14](14_mha_fp16_micro_verified.md) §5.7: as
`Sᵀ = K·Qᵀ` it needs no transpose, its output is the existing scores layout,
its K=128 accumulation carries none of `P·V`'s fp16-accumulation risk, and it
needs nothing from the softmax owner. The shape table below still stands as a
reference for both stages' dimensions:

| stage | M | N | K | note |
| :-- | :-- | :-- | :-- | :-- |
| `Q·Kᵀ` | q_len | kv_len | head_dim=128 | N padded to 32/token |
| `P·V`  | q_len | head_dim=128 | kv_len | N aligned for free — **start here** |

### Concrete steps

1. **Bench first, in the harness (no nntrainer build).** Add an fp16 attention
   shape to `hexkl_micro_fc_bench.c` using `hexkl_micro_hmx_mm_f16`: a single
   `P·V` (M=q_len, N=128, K=kv_len for a realistic decode kv_len, e.g. 512/1024)
   vs the SDKL `sdkl_npu_mm_f16` / on-DSP f16 kernel. Confirm correctness vs a
   scalar fp16 reference and get the µs. This proves the fp16 micro op end-to-end
   before touching mha_core.
2. **GQA row-fold** (doc 08 §7): the `gqa_size` Q heads sharing one kv_head fold
   into `M` of one GEMM instead of `gqa_size` separate calls. Largest structural
   win, host-side, no DSP code. Mind the **GQA stride** (cache is
   `[kv_pos][kv_head][head_dim]` — a single kv_head is strided; a gather/pack
   step is unavoidable, count its cost — doc 08 §2).
3. **Add a fused seam in mha_core** (doc 08 §9): a *single* `attn_forward`, not
   three ops, so the score matrix stays VTCM-resident across `Q·Kᵀ → softmax →
   P·V`. CPU path stays, selected by a **shape gate** (profitability: at short
   kv_len the pack cost exceeds the GEMM gain — doc 08 §7).
4. **P·V on HMX** behind the seam, then **Q·Kᵀ** with N padded to 32.
5. Only then measure stage-boundary cost.

### The two hard unknowns (verify, don't assume)

- **Scores layout.** `out_` is `(1,1,packed_triangular_len, num_heads_Q)` —
  **triangular-packed, head-minor** (doc 08 §1). That is what makes the CPU
  softmax cheap and is the biggest obstacle to feeding a GEMM, which wants dense
  `[M,K]×[K,N]`. Decide: pack for the GEMM, or keep the intermediate in a dense
  VTCM tile and only pack when handing to softmax.
- **Softmax is the HVX owner's** (doc 08 §8). For the fused Option-B path the
  softmax must run on a **VTCM-resident** buffer between the two matmuls. Until
  that lands, either (a) prototype with the score matrix round-tripping to the
  CPU softmax (measures P·V and Q·Kᵀ in isolation, not the fusion), or (b)
  coordinate the VTCM-resident softmax entry point. **This is the gating
  dependency for the *fused* win; the individual matmuls can be measured
  without it.**

### AH/WH caveat (doc 08 §3)

RM→AH and RM→WH are parallel, not a chain; there is **no `ah_to_wh`**. Whether
an HMX accumulator's AH output can be fed straight back as an AH activation
(needed to keep scores in VTCM across stages) is exactly what probe (5) in
`hexkl_layout_probe.c` measures. Run it before assuming the fused path is
reachable.

---

## 6. Files touched / to reuse

| file | role |
| :-- | :-- |
| `test/unittest/jni_htp/hexagon/hexkl_micro_fc_bench.c` | FC micro+DMA bench; **reuse its tile loop + user-DMA stager for the fp16 attention bench** |
| `test/unittest/jni_htp/hexagon/run_micro_fc_bench.sh` | one-shot build/run/report (fc_compare format) |
| `Applications/CausalLM/layers/mha_core.cpp` | target; `compute_kcaches` (577), `compute_fp16vcache_transposed` (1381), fp32 refs (42, 73) |
| `test/unittest/jni_htp/hexagon/hexkl_layout_probe.c` | probe (5): does accumulator AH == rm_to_ah AH (decides fusion) |
| `test/unittest/jni_htp/hexagon/hexkl_micro_mha_bench.c` | **the fp16 attention bench this section called for — built, verified, see [14](14_mha_fp16_micro_verified.md)** |
| `test/unittest/jni_htp/hexagon/mha_cpu_bench.cpp` | the ARM CPU comparison bar for doc 14's numbers |

Verified-result memory: `htp-micro-dma-fc-beats-sdkl`,
`htp-mha-qk-decode-beats-cpu`.

---

## 7. Do-not-re-litigate (measured)

- Micro+DMA cross-matmul **beats SDKL 1.7–2× for u8i4 FC** — on device, correct.
- The win **requires a persistent skel** (cross-call weight prefetch); single
  matmul is only parity. A stateless per-call kernel cannot win.
- **fp16 micro HMX ops exist** and HMX fp16 rate = 8 on V79 — attention fp16
  path is real.
- host quant/dequant (~77 µs) is the **HVX owner's scope**, identical both
  sides, excluded from the kernel comparison.
- **Corrected by [14](14_mha_fp16_micro_verified.md), do not re-litigate the old
  version:** it is `Q·Kᵀ` **before** `P·V` (doc 14 §5.7, not an N-alignment
  argument). Fusion turned out unnecessary — softmax-on-VTCM is **not** gating;
  the unfused round trip to CPU softmax costs ~1% of the per-layer total (doc
  14 §7). The win is DMA-descriptor-geometry redesign (row size, VTCM-vs-DDR
  source, incremental bake at KV-append time — doc 14 §5), not fusion and not
  HMX compute (1.7-24% of the per-call time depending on decode vs prefill).
- Both `Q·Kᵀ` and `P·V` decode and prefill are now measured, on device, beating
  CPU 2.4-65× — doc 14 §3. Not yet wired into `mha_core.cpp`'s dispatch or a
  persistent FastRPC skel; doc 14 §7 is explicit that none of these numbers
  include per-op RPC cost.

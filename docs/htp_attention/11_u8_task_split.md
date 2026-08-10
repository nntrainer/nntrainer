# MHA-core on HTP with micro u8i4 **and** u8i8 — task split and prompts

Companion to `MHA_HTP_PLAN.md`. That document's architecture still holds (one
FastRPC call per layer, fusion mandatory, prefill is the target, sparsity at
`(kv_head, kv_block)` granularity, FC migration not a prerequisite). **What
changes here is the arithmetic: the two attention matmuls run on the HexKL
*micro* integer path — `hexkl_micro_hmx_mm_u8i4` and `hexkl_micro_hmx_mm_u8i8` —
instead of fp16.** Both widths are first-class from Task 1, selectable at
runtime, and selectable *per operand* (§0.2).

§1 is the design delta, §2 the prompting rules, §3 the tasks.

---

## 0. Dtype: both widths, one code path, per-operand choice

### 0.1 Why not "i4 first, mirror to i8 later"

Because the tree already contains both halves, verified:

| | u8i4 | u8i8 |
| :-- | :-- | :-- |
| registry + layer path | `hmx/hexkl_mm_u8i4_dma.{c,h}` | `hmx/hexkl_mm_u8i8_dma.{c,h}` |
| register / release / run | `hexkl_weight_u8i4_register` / `_release` / `hexkl_mm_u8i4_layer_run` | `hexkl_weight_u8i8_register` / `_release` / `hexkl_mm_u8i8_layer_run` |
| WH bake | `hexkl_micro_hmx_rm_to_wh_i4` | `hexkl_micro_hmx_rm_to_wh_i8` |
| HMX tile op | `hexkl_micro_hmx_mm_u8i4` | `hexkl_micro_hmx_mm_u8i8` |
| WH bytes per 32×32 tile | 512 | 1024 |
| tile geometry | **identical**: 64×32 activation × 32×32 weight → 64×32 int32 acc | ← same |
| activation quant | `hvx_quant_rows_u8_params` + `hvx_quant_pack_u8_ah` | ← **same code**, u8 either way |
| int32 dequant | `hvx_dequant_i32_to_f32` | ← **same code** |
| device-verified | PR①, 9/9 tests | PR②, 5/5 tests |

The signatures are identical apart from the type of the weight table. So the
whole difference is **three symbol names and one constant**, which is exactly
what the FC bench already collapsed into a single `dtype_ops` descriptor
("one code path, `dtype_ops` descriptor" — memory `htp-micro-dma-fc-beats-sdkl`).

**This reverses PR②'s duplication decision, deliberately, and the reason
matters.** PR② mirrored rather than parameterised because *the u8i4 path was
already device-verified and folding both behind one abstraction would have meant
re-verifying it, for a saving of ~150 lines*. That argument is about protecting
verified code. It does not apply to `hexkl_attn_*`, where **neither width is
verified yet** — there is nothing to protect, and duplicating ~350 lines of
brand-new orchestration means every later bug gets fixed twice and every gate
runs twice. So: **one dtype-parametric attention module, dispatching through a
3-entry vtable** (Task 4).

The IDL follows the same reversal: **one `width` parameter, not separate ops per
dtype.** PR② chose separate ops (`weight_register_u8i8` beside
`weight_register_u8i4`) because there were two of them. Here the combinations are
four (§0.2), and four copies of every attention entry point is not a surface
worth carrying.

### 0.2 K and V do not need the same width

This is the payoff of a parametric design, and it is nearly free.

`MHA_HTP_PLAN.md` §1.4 establishes that the KV cache is registered **per
`(head, kv_block)`** — each block is its own weight handle with its own scales.
Nothing forces `Kᵀ` blocks and `V` blocks to share a width; the width is a
property of the handle, resolved when `layer_run` dispatches. So all four
combinations are reachable with no structural change:

| K | V | rationale to test it |
| :-- | :-- | :-- |
| i8 | i8 | the safe baseline — QNN's shipping Gauss deployment quantizes its KV cache to **int8** on this same silicon (verified by reading `~/workspace/Quick.AI`'s `gauss*_qnn.cpp` runtime dtype assertions) |
| i4 | i4 | the smallest footprint; what was asked for |
| **i8** | **i4** | scores feed an exponential, so K error amplifies while V error only averages. If one operand has to be i4, this is the informed guess for which |
| i4 | i8 | the control that tests whether that guess is right instead of assuming it |

**Keep the knob even if the numbers say one combination wins.** It is a
hardware/calibration parameter, it costs one enum per handle, and the answer will
differ per model.

### 0.3 Why the integer path is a better plan than fp16 anyway

Worth putting in the PR body, because "we quantized attention" invites the wrong
question:

1. **The design is data-movement-bound, not compute-bound** (doc14 §5: HMX
   `mm_f16` is 1.7% of a decode call; DMA row size and DDR-vs-VTCM are the whole
   game). Resident KV bytes **per kv position, all heads**, at
   `nch=8, head_dim=128`:

   | | bytes / kv position | K+V slab at kv=1024 | VTCM residency ceiling (8 MiB) |
   | :-- | --: | --: | --: |
   | fp16 | 4 KiB | 4 MiB | ~1 024 |
   | **u8i8** | 2 KiB | 2 MiB | ~2 048 |
   | **u8i4** | 1 KiB | 1 MiB | ~4 096 |

   That is a direct cut to the dominant cost, and it moves
   `MHA_HTP_PLAN.md` §4.2's residency wall out by 2–4×.
   *Caveat for the implementer:* the **input** to `hexkl_weight_u8i4_register` is
   "int4 values in int8 containers", 1 byte per value — the halving happens in the
   **baked WH bytes** (512 vs 1024 per tile), which is what stays resident and
   what the DMA moves. Do not conclude i4 saves nothing from looking at the
   quantizer's output buffer.
2. **Reuse.** The fp16 attention path exists only as a bench on another branch.
   Both integer paths are in this tree, on this branch, device-verified: weight
   registry, WH bake, DMA ring with cross-block prefetch, HVX activation quant,
   HVX int32 dequant, the async accumulator copy-out, the QuRT worker pool.
   §1.3 counts what is actually new.
3. The int32 accumulator removes fp16 accumulation as a question entirely
   (§1.6 does the overflow arithmetic; it does not come close).

---

## 1. Design delta: the chain on the micro integer path

### 1.1 The chain, per (kv_head `h`, q-band `b`)

```
Q band (f32, [M][head_dim], M = n_query * gqa)
  │  hvx_quant_rows_u8_params + hvx_quant_pack_u8_ah    ← in-tree, width-agnostic
  ▼
u8 AH activation
  │  hexkl_micro_hmx_mm_u8i4  OR  _mm_u8i8   vs Kt blocks (WH)   ← in-tree
  ▼
int32 acc ─ async DMA copy-out ─► hvx_dequant_i32_to_f32   ← in-tree, width-agnostic
  ▼
S, f32, BLOCK-MAJOR: [n_blocks][M][T]
  │  blocked masked softmax → UNNORMALIZED exp + l[M]            ← NEW (Task 3/5)
  ▼
P blocks, f32, non-negative
  │  per block: quant → mm_u8i4/u8i8 vs V block → dequant        ← in-tree
  ▼
O partials f32 [M][head_dim] per block
  │  f32 accumulate over blocks, then × 1/l per row              ← NEW (thin)
  ▼
attention output [n_query][nHq * head_dim]
```

Only the two `mm_*` steps and the two `register` calls are width-dependent.
Activation quant and int32 dequant are shared verbatim — u8 activations and
int32 accumulators are the same at both widths.

### 1.2 `hexkl_mm_u8iX_layer_run` *is* the `S = Q·Kᵀ` call

Both widths expose the identical signature (`hexkl_mm_u8i4_dma.h:63`,
`hexkl_mm_u8i8_dma.h:63`):

```c
int hexkl_mm_u8iX_layer_run(tbl, vtcm_base, vtcm_size, config_off,
                            M, K, handles, n_handles, act_f32, out_cat);
```

- `act_f32` = the Q band; it quantizes internally
  (`hexkl_mm_u8i4_dma.c:205-206`) — one activation shared across every handle,
  which is exactly "the `gqa` query heads of kv_head *h* share one K".
- `handles` = **kv_head h's `Kᵀ` blocks**, each registered `K = head_dim`,
  `N = T`. One handle per KV block.
- `out_cat` = the S band, `M × Σ N` f32, dequantized, **one contiguous `M × N`
  block per handle in call order**.
- Cross-block weight prefetch (doc13 §3a's 1.7–2×) comes for free — it already
  prefetches handle *i+1*'s weight while *i* computes
  (`hexkl_mm_u8i4_dma.c:230`, `hexkl_mm_u8i8_dma.c:229`).

So the first matmul is **an existing call with different arguments, at either
width**. `P·V` likewise needs no new matmul: call `layer_run` once per block with
`n_handles = 1`, `act_f32 = P_block` (`[M][T]`), the V block's handle
(`K = T, N = head_dim`), and accumulate the f32 outputs.

### 1.3 What is actually new

| new | where | size |
| :-- | :-- | --: |
| KV block quantizer + WH-source builder, **parametric over i4/i8** | `hmx/hexkl_kv_quant.{c,h}` | ~250 |
| dtype vtable over the two registries | `hmx/hexkl_attn_dtype.{h,c}` | ~90 |
| blocked masked softmax on HVX (causal / window / sink, unnormalized-exp out) | `hvx/hvx_softmax_blocked_f32.{c,h}` | ~220 |
| attention orchestration: KV registry, band loop, block loop, O accumulate | `hmx/hexkl_attn_u8.{c,h}` | ~350 |
| IDL entries + skel glue | `test/htp/nntr_hvx.idl`, `test/htp/nntr_hvx_attn.c` | ~180 |
| host model (the oracle) + gtests | `test/unittest/…` | ~600 |

Everything else is called, not written. **One** orchestration file, not two.

### 1.4 The dtype vtable

Three function pointers and a constant. Resolve it **once per `layer_run` call**,
never per tile — a function-pointer call inside the tile loop would put an
indirect branch in the hottest loop on the DSP for no reason.

```c
typedef enum { HEXKL_W_I4 = 4, HEXKL_W_I8 = 8 } hexkl_w_width;

typedef struct {
  hexkl_w_width width;
  uint32_t wh_tile_bytes;            /* 512 (i4) or 1024 (i8) */
  void    *table;                    /* hexkl_weight_u8i4_table* or _u8i8_table* */
  int (*reg)(void *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
             uint32_t K, uint32_t N, const int8_t *w_rm, const float *w_scale,
             const int32_t *colsum_w, const float *bias, uint32_t *out_handle);
  int (*rel)(void *tbl, uint32_t handle);
  int (*run)(void *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
             uint32_t config_off, uint32_t M, uint32_t K,
             const uint32_t *handles, uint32_t n_handles,
             const float *act_f32, float *out_cat);
} hexkl_attn_ops;
```

The two implementations are thin casts over the existing functions. `Kᵀ` and `V`
each hold their own `hexkl_attn_ops`, which is what makes §0.2's per-operand
width free.

### 1.5 Quantization scheme — and why the scale axes work out

The dequant contract (`hvx_dequant_i32.h`) is fixed and identical at both widths:

```
out[m][n] = (acc[m][n] − act_zp[m]·colsum_w[n]) · act_scale[m] · w_scale[n] + bias[n]
```

i.e. **per-row activation scale + zero point, per-column (per-N) weight scale.**
Mapped onto attention:

| operand | role | quant axis needed | incremental? |
| :-- | :-- | :-- | :-- |
| `Q` | activation | per row (query row × head) | dynamic per call ✓ |
| `Kᵀ` | weight `[head_dim][T]`, N = kv position | **per kv position** | **= per token ✓**, computed once at append, never revisited |
| `P` | activation | per row, per block | dynamic per call ✓ |
| `V` | weight `[T][head_dim]`, N = head_dim column | per column, **over this block's T rows only** | ✓ **because the block is the registration unit** |

The `V` row is the trap: a per-`head_dim` scale over the *whole* cache changes as
tokens are appended, invalidating every already-quantized value. **Blocking the KV
cache fixes it structurally** — a block is registered once, with its own
`[head_dim]` scale vector and `colsum_w`, and never touched again. Block-wise
scales are also strictly more accurate than a global one. This is why the KV cache
must be **a sequence of registered weight blocks**, not one big weight, and it is
what makes §0.2 possible.

`colsum_w` is incremental either way: for `Kᵀ` it is `Σ_d` over the appended
token's own row; for `V` it is a running column sum within the block.

Only the **tail block** (partially filled) is re-registered as tokens arrive.
Zero-fill its unused columns/rows at registration; a garbage tail is a silent
wrong answer and the mask alone will not save you.

Symmetric ranges: **i4 → `[-8, 7]`, i8 → `[-127, 127]`** (not −128, so the
symmetric scale stays exact). Round to nearest, ties away from zero.

### 1.6 int32 overflow — checked, not assumed

Worst-case product: `255 × 7 = 1 785` (i4) or `255 × 127 = 32 385` (i8).

| matmul | K | worst-case accumulator | int32 headroom |
| :-- | --: | --: | --: |
| `S = Q·Kᵀ` | `head_dim` = 128 | 2.3e5 (i4) / 4.1e6 (i8) | ×9 000 / ×500 |
| `O = P·V` | `T` ≤ 1024 | 1.8e6 (i4) / 3.3e7 (i8) | ×1 100 / ×65 |

No chunking, no saturation concern, at either width. fp16 accumulation's accuracy
question (doc14 §4) simply does not exist on this path.

### 1.7 Fold the softmax normalization into the dequant, not a third pass

PR #4245's softmax is 3-pass (max → exp+sum → normalize). On the integer path the
third pass is free to delete:

- After pass 2 the row maximum is **exactly `exp(0) = 1`** and every value is in
  `(0, 1]`.
- Feed those unnormalized values to `P·V`. Per-row dynamic quantization then sees
  a known, well-conditioned range — no wasted codes, `zp = 0` naturally. This
  matters more at i4 (16 levels) than at i8, so it is not an optional nicety.
- Multiply `O` by `1/l` per row **once**, after accumulating every block's f32
  partial.

`l` must be returned per row from the softmax; that is the only signature change
this implies.

### 1.8 Constraints the implementer will trip over, from reading the code

- **`m_pad = ROUND_UP(M, 64)`** (`HEXKL_HMX_INT8_BLOCK_N_ROW`) — the integer
  accumulator is 64×32, not the fp16 path's 32×32, **at both widths**.
  **Consequence: decode is a bad fit.** At decode `M = gqa = 8` → `m_pad = 64`,
  12.5% row utilisation; combined with `MHA_HTP_PLAN.md` §2's ~404 µs transport,
  **do not expect a decode win on this path.** Prefill (`M = n_query·gqa`, e.g.
  1024) is where it pays. Put that in the PR body rather than letting a reviewer
  discover it.
- **`K % 32 == 0` and `N % 32 == 0`** are enforced by `hexkl_mm_u8i4_plan`
  (`AEE_EBADPARM`). So **`T` must be a multiple of 32**; `head_dim` 64 and 128
  both qualify.
- **`T ≥ 64` for DMA reasons** (doc14 §5.1: a `(head, block)` slab wants to be
  ≥16 KiB to stay out of the 8.4 GB/s regime). Recommend `T = 256`. Note the slab
  is *smaller* at i4 for the same `T`, so i4 pushes toward a **larger** `T` than
  i8 to keep the same descriptor row size — measure, do not assume one `T` is
  right for both widths.
- **`out_cat` is block-major, not a contiguous `[M][kv]` matrix.** Row `m` of
  block `j` is at `out_cat + off_j + m*T`. Do **not** repack it — write the
  softmax to walk segments (Task 5). The block-major layout *is* the KV tiling.
- **`layer_run` mallocs `act_scale`, `act_zp`, `acc_scratch` on every call.**
  Calling it per block per head per band is a malloc storm. Leave it alone until
  Task 9 (perf); then hoist. When you do: **`HVX_UVector*`, never
  `HVX_Vector*`, for anything `malloc` returned** — `malloc` carries no 128-byte
  guarantee and an aligned vector store on it crashed the whole unsigned-PD DSP
  process the last time this was forgotten.
- **Fold `1/sqrt(head_dim)` into the softmax's `scale` parameter**, which PR
  #4245's kernel already takes. Not into `w_scale` — that hides a model constant
  inside quantization metadata where nobody will look for it.

---

## 2. How to prompt the implementing model

The failure mode to design against is not bad code; it is **declaring success on
evidence that does not support it** — loosening a tolerance, editing the
reference, timing a region that contains the correctness check, reporting one
lucky run. Six rules, all about making that hard:

1. **One shared context file, referenced by every prompt.** Each opencode session
   starts blank. Open every prompt with *"Read `MHA_HTP_PLAN.md` and
   `MHA_HTP_U8_TASKS.md` in full before doing anything, then restate §1.8's
   constraint list in your own words before writing code."* The restatement is a
   cheap check that it actually read them.
2. **One concern per task, with a line budget.** *"If your change exceeds N
   lines, stop and report why instead of continuing."* A model allowed to grow the
   diff will fold three tasks into one and the per-task gate disappears.
3. **Never combine "implement" and "optimize".** Correctness tasks forbid touching
   performance; the perf task forbids changing behaviour and must prove it
   (bit-exact against the pre-change output).
4. **The acceptance criterion is a command plus its expected output**, and every
   prompt ends with *"paste the complete output, not a summary"*. Summaries are
   where "14/14 passed" comes from a run that never happened.
5. **Front-load the host-only tasks.** Tasks 1–3 need no phone, no NDK, no skel
   build, no `ADSP_LIBRARY_PATH`. A model iterating on the host cannot mistake an
   environment problem for a logic problem, and that confusion burns whole
   sessions.
6. **Hand over the failure-signature table so it self-diagnoses.** Include this in
   every prompt that checks numerics:

   > At i8, maxRelErr ≈ 1e-3 or below is healthy. At i4, ≈1e-2 is healthy.
   > An error that is **10× the other width's on the same shape** means the width
   > is doing what it should. An error that is the **same at both widths** means
   > the bug is not quantization — look at layout or masking. ≥1e-1, or a
   > blocky/structured error pattern, is a layout bug. Report which case you have;
   > do not adjust the tolerance.

   (Comparing the two widths against each other is the sharpest diagnostic this
   plan has, and it exists only because both widths run in the same test.)

**Rule zero, verbatim in every prompt:**

> 1. Do not modify any reference implementation:
>    `compute_kcaches_fp32_reference` (`Applications/CausalLM/layers/mha_core.cpp:42`),
>    `compute_vcache_fp32_transposed_reference` (`:73`),
>    `nntrainer::softmax_row` / `softmax_row_inplace`, or the scalar reference in
>    `test/unittest/unittest_hvx_mm_u8i4.cpp`. If a test fails, the new code is wrong.
> 2. Do not change a tolerance or a shape from the ones written in the task.
> 3. Do not invent a HexKL, HVX or QuRT API. Every call must exist in
>    `~/workspace/hxkl-beta2/hexkl_addon/include/hexkl_micro.h` or in a header
>    already included by this tree. If you need something that does not exist,
>    stop and report it.
> 4. Do not modify `hexkl_mm_u8i4_dma.{c,h}`, `hexkl_mm_u8i8_dma.{c,h}`,
>    `hexkl_mm_u8i4.{c,h}`, `hexkl_dma_ring.{c,h}`, `hvx_quant_u8.c` or
>    `hvx_dequant_i32.c`. All are device-verified. If you believe one must change,
>    stop and report which line and why.
> 5. If a test fails, do not "fix" the test. Stop and report the failure with its
>    full output.

**Two human checkpoints before much code exists:** after Task 1's restatement +
plan, and after Task 3's host model. Those are the two places a wrong decision
propagates into every later task.

---

## 3. The tasks

Dependency order; each is a separate opencode session. Do not start a task until
a human has read the previous one's acceptance output.

**Task 0 is yours, not the model's:** rebase onto `pr4245` (which already carries
PR①'s u8i4 commits plus the HVX softmax — `MHA_HTP_PLAN.md` §7.1), create the
branch **before** the first commit, and confirm `git log --oneline HEAD..pr4245`
is empty afterwards.

---

### Task 1 — KV block quantizer, parametric over i4 and i8 (host only)

```
Read MHA_HTP_PLAN.md and MHA_HTP_U8_TASKS.md in full. Before writing code,
restate §1.8's constraint list in your own words, and state which of §1.5's four
scale axes applies to each of Q, Kt, P and V.

[paste Rule zero]

GOAL
New nntrainer/tensor/htp_backend/hmx/hexkl_kv_quant.{c,h}: turn appended fp16 K
and V rows into exactly the inputs hexkl_weight_u8i4_register /
hexkl_weight_u8i8_register want, one KV block at a time. Pure arithmetic: no HVX
intrinsics, no HexKL calls, no DSP-only headers -- this file must compile and run
on the host.

  typedef enum { HEXKL_W_I4 = 4, HEXKL_W_I8 = 8 } hexkl_w_width;

  /* Kt block: logical [head_dim][T], N = kv position within the block.
     Symmetric per-N (per kv position) quantization. */
  void hexkl_kvq_pack_kt_block(const uint16_t *k_rows_f16, uint32_t n_rows_valid,
                               uint32_t T, uint32_t head_dim, uint32_t nch,
                               uint32_t head, hexkl_w_width w,
                               int8_t *out_rm, float *out_scale,
                               int32_t *out_colsum);

  /* V block: logical [T][head_dim], N = head_dim column.
     Symmetric per-N (per column) quantization over THIS BLOCK's rows only. */
  void hexkl_kvq_pack_v_block(const uint16_t *v_rows_f16, uint32_t n_rows_valid,
                              uint32_t T, uint32_t head_dim, uint32_t nch,
                              uint32_t head, hexkl_w_width w,
                              int8_t *out_rm, float *out_scale,
                              int32_t *out_colsum);

Both read the CPU cache layout [kv_position][nch][head_dim] (fp16): row r of head
h is at (r*nch + h)*head_dim -- see mha_core.cpp:61 and :95.

The width is a runtime parameter, ONE code path, not two functions or a macro
expansion. The only differences are the clamp range and the rounding target:
  i4 -> [-8, 7]      i8 -> [-127, 127]   (not -128; symmetric scale stays exact)
Both widths write out_rm as one int8 container per value -- that is what
rm_to_wh_i4 and rm_to_wh_i8 both take; the packing to 4 bits happens inside the
WH bake, not here. Round to nearest, ties away from zero.

n_rows_valid < T means a partial tail block: zero-fill every unused element of
out_rm, and set the corresponding scale to 1.0f and colsum to 0.

MUST NOT TOUCH any existing file. This task adds two files and one test file.
BUDGET ~250 lines across the two new files. Exceed it -> stop and report.

ACCEPTANCE
New host gtest test/unittest/unittest_hexkl_kv_quant.cpp proving, over
{T=32,64,256} x {head_dim=64,128} x {nch=1,8} x {head=0,nch-1} x
{n_rows_valid=1,T-1,T} x {I4,I8}:
  a) round trip -- dequantizing out_rm with out_scale reproduces the input within
     the theoretical bound for that width. COMPUTE the bound from the width and
     the row's dynamic range; do not hardcode a number.
  b) out_colsum[n] equals a plain independent sum of column n of out_rm, computed
     by a separate loop in the test.
  c) every element past n_rows_valid in a partial tail block is exactly 0, and its
     scale is exactly 1.0f.
  d) no value falls outside the width's range.
  e) i4 and i8 agree in SIGN and ORDER on every element (i8 is a refinement of
     i4, not a different mapping) -- if this fails, one of the two ranges is wrong.
Register the test in test/unittest/meson.build the way the neighbouring tests are.

Run: cd builddir && meson test unittest_hexkl_kv_quant --print-errorlogs
Paste the complete output. Report how many parameter combinations ran; the count
must equal the cross product, and if it does not, say so.
```

---

### Task 2 — blocked masked softmax: host reference and math contract

```
Read MHA_HTP_PLAN.md and MHA_HTP_U8_TASKS.md in full. Before writing code, explain
§1.7's "unnormalized exp + l" contract in your own words, including why the row
maximum after pass 2 is exactly 1.0 and why that matters more at i4 than at i8.

[paste Rule zero]
[paste the failure-signature table from §2 rule 6]

GOAL
A host-side scalar reference in test/unittest/mha_htp_host_model.{cpp,h} (new).
A reference, not a kernel: plain C++, clarity over speed.

  struct MhaSoftmaxOut { std::vector<float> p; std::vector<float> l; };

  /* Block-major scores: segment j is s[off_j + m*T + t], t in [0, T).
     Softmax runs along the kv axis (ACROSS segments) for each row m.
     p holds UNNORMALIZED exp values (row max exactly 1.0);
     l[m] is the row denominator, including the sink term when present. */
  MhaSoftmaxOut mha_softmax_blocked_ref(
      const std::vector<const float *> &seg, uint32_t n_seg, uint32_t T,
      uint32_t M, float scale,
      const std::vector<uint32_t> &begin, const std::vector<uint32_t> &end,
      const float *sink /* nullptr, or one logit per row */);

Masked positions must be excluded from the MAXIMUM pass, not only from the sum.
State in a comment why (hint: hvx_exp_f32.h documents its domain, and it is
UNDEFINED above +88.7 -- a masked lane that skips the max pass but reaches exp is
undefined behaviour, not a small numerical error).

MUST NOT TOUCH anything under nntrainer/. Test-side only.
BUDGET ~200 lines.

ACCEPTANCE
test/unittest/unittest_mha_htp_host_model.cpp proving:
  a) with begin=0, end=kv for all rows, no sink, n_seg=1: p[m][*]/l[m] equals
     nntrainer::softmax_row_inplace on the same data within 1e-6;
  b) sum(p[m][*]) == l[m] within 1e-5 in EVERY configuration -- and assert this
     inside the reference itself, not only in the test;
  c) max(p[m][*]) == 1.0f exactly whenever row m has at least one valid position;
  d) window and sink behaviour matches nntrainer::softmax_row's sink overload;
  e) the segment split is invisible: the same logical scores split into n_seg =
     1, 2 and 4 give BITWISE IDENTICAL p and l.
Cover kv in {1,31,32,33,255,256,257,1023,1024}, T in {32,256}, M in {1,7,64},
window in {0,1,T-1,T,T+1,kv-1,kv}, sink present/absent.

Run: cd builddir && meson test unittest_mha_htp_host_model --print-errorlogs
Paste the complete output. (e) is the most important case -- report it explicitly.
```

---

### Task 3 — the host model of the whole kernel, both widths (the oracle)

```
Read MHA_HTP_PLAN.md §9.4 and MHA_HTP_U8_TASKS.md §1 in full.

[paste Rule zero]
[paste the failure-signature table]

GOAL
Extend test/unittest/mha_htp_host_model.{cpp,h} with a scalar model of the WHOLE
attention kernel -- the same band loop, block loop, quantization points and mask
arithmetic the DSP kernel will use, with plain arithmetic instead of HMX/HVX/DMA.
It CALLS Task 1's hexkl_kvq_* and Task 2's mha_softmax_blocked_ref; it does not
reimplement either.

  void mha_htp_host_forward(
      uint32_t n_query, uint32_t kv_from, uint32_t nch, uint32_t gqa,
      uint32_t head_dim, uint32_t T, uint32_t M_band,
      bool is_causal, uint32_t window, const float *sink,
      hexkl_w_width w_k, hexkl_w_width w_v,     /* per-operand, see §0.2 */
      const float *q, const uint16_t *k_cache, const uint16_t *v_cache,
      float *out);

This is a SPECIFICATION of the loop structure. Two properties matter above all:
(1) the ORDER of operations must be the order the DSP kernel will use, so a later
bit-exactness comparison is meaningful; (2) every quantization step must be
present, so the model's error equals the kernel's expected error rather than an
fp32 upper bound.

MUST NOT TOUCH anything under nntrainer/ or Applications/.
BUDGET ~300 lines.

ACCEPTANCE
Compare mha_htp_host_forward against the fp32 ground truth
(compute_kcaches_fp32_reference -> nntrainer::softmax_row_inplace ->
compute_vcache_fp32_transposed_reference) across this matrix. Every value of every
axis must appear at least twice, and every (kv_len % 32 != 0) x
(n_query % 32 != 0) combination at least once:

  kv_len    1, 31, 32, 33, 64, 255, 256, 257, 1023, 1024
  n_query   1, 7, 32, 33, 128
  gqa       1, 4, 8
  nch       1, 8
  head_dim  64, 128
  T         64, 256
  M_band    64, 256
  is_causal true, false
  window    0, 1, T-1, T, T+1, kv_len-1, kv_len
  sink      absent, present
  (w_k, w_v) (I8,I8), (I4,I4), (I8,I4), (I4,I8)      <-- all four

Tolerance on out, max relative error, FIXED HERE -- do not change:
  (I8,I8) <= 5e-3    (I8,I4) <= 2e-2    (I4,I8) <= 5e-2    (I4,I4) <= 5e-2
A configuration outside these is a finding to report with the configuration and
the observed value; state which failure signature it matches.

DELIVERABLE: a printed table of observed max relative error per (shape, w_k, w_v).
That table, not a pass/fail, is the output of this task. It is also the first real
evidence about §0.2 -- if (I8,I4) is close to (I8,I8) while (I4,I8) is close to
(I4,I4), then K's width dominates, which is the hypothesis §0.2 states.

Run: cd builddir && meson test unittest_mha_htp_host_model --print-errorlogs
Paste the complete output including the table.
```

---

### Task 4 — the dtype vtable (small, host-compilable, no behaviour)

```
Read MHA_HTP_U8_TASKS.md §0.1, §0.2 and §1.4. Read both
nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_dma.h and hexkl_mm_u8i8_dma.h and
confirm for yourself that the two triples of functions differ only in the table
type. State that confirmation before writing code.

[paste Rule zero]

GOAL
New nntrainer/tensor/htp_backend/hmx/hexkl_attn_dtype.{h,c}: the hexkl_attn_ops
vtable of §1.4, plus two constructors:

  int hexkl_attn_ops_init(hexkl_attn_ops *out, hexkl_w_width w);
  void hexkl_attn_ops_fini(hexkl_attn_ops *ops);

Thin casting adapters over the existing u8i4 / u8i8 functions. No logic, no
allocation beyond the weight table itself, no new behaviour.

The vtable is resolved ONCE PER layer_run CALL and never per tile. Write that as a
comment where the struct is defined, so the next person does not put an indirect
call in the tile loop.

MUST NOT TOUCH either hexkl_mm_u8i*_dma file. If an adapter cannot be written
without changing one of them, stop and report which line.
BUDGET ~120 lines.

ACCEPTANCE
Device gtest (this needs the DSP because the register/run functions are DSP-side):
for each width, register a small known K x N weight through the vtable, run one
matmul through it, and check the result is BITWISE IDENTICAL to calling
hexkl_weight_u8iX_register / hexkl_mm_u8iX_layer_run directly with the same
arguments. Bitwise, not within a tolerance: the vtable adds no arithmetic, so any
difference is a plumbing bug.

Build the skel:
  export HEXAGON_SDK_ROOT=~/workspace/Hexagon_SDK/6.4.0.2
  export DEFAULT_HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04
  export HEXKL_ROOT=~/workspace/hxkl-beta2/hexkl_addon HEXKL_SDK_VER=6.4.0.2
  cd test/htp && bash build.sh
Use the beta2 addon. The one under Hexagon_SDK/6.4.0.2/addons/ is beta1 and its
hexkl_micro_hw_init takes 2 args, which fails at DSP compile with "too many
arguments to function call" -- that is a library version mismatch, not a code bug.
Do not try to fix setup_sdk_env.source; it fails here and the two exports above
are what it would have set.
Run via test/htp/run_u8i4_layer_on_device.sh (extend it; do not write a second
runner). Device is R3CY10WM83Y.

Paste the complete output. If the DSP call hangs or fails with no
application-level detail, run `adb logcat -d | grep adsprpc` FIRST and paste that
-- the kernel driver reports PD crashes with the crashed function name
independently of FARF.
```

---

### Task 5 — `hvx_softmax_blocked_f32` on the DSP

```
Read MHA_HTP_PLAN.md §7 and MHA_HTP_U8_TASKS.md §1.7. Read
nntrainer/tensor/htp_backend/hvx/hvx_softmax_f32.c and hvx_exp_f32.h COMPLETELY
first -- you are extending that design, not replacing it, and its comments explain
choices you must preserve (why the scale is folded before the max, why the tail is
staged through an aligned buffer, why the sum accumulates in qf32).

[paste Rule zero]

GOAL
New nntrainer/tensor/htp_backend/hvx/hvx_softmax_blocked_f32.{c,h} implementing
Task 2's contract on HVX:

  void hvx_softmax_blocked_f32(float *const *seg, uint32_t n_seg, uint32_t T,
                               uint32_t m_first, uint32_t m_last, uint32_t M,
                               float scale, const uint32_t *begin,
                               const uint32_t *end, const float *sink,
                               float *l_out);

In place: seg[j] is overwritten with unnormalized exp values. The [m_first,
m_last) row range exists so this drops onto hvx_worker_pool_run unchanged -- same
convention as hvx_softmax_rows_f32 and hvx_dequant_i32_to_f32.

Reuse hvx_exp_sf and the reduce/tail helpers from hvx_softmax_f32.c. If a helper
there is static and you need it, move it to a header in a SEPARATE FIRST COMMIT
that changes nothing else, so the diff shows the move as a move.

Masked lanes must be excluded from the maximum pass -- see Task 2's comment
requirement and hvx_exp_f32.h's documented domain.

MUST NOT TOUCH hvx_softmax_f32.c/.h beyond that mechanical helper move; do not
change its behaviour. Do not touch any hmx/ file.
BUDGET ~220 lines in the new files.

ACCEPTANCE
1. FastRPC entry in test/htp/nntr_hvx.idl mirroring the existing softmax_f32
   entry, plus a gtest comparing it against Task 2's mha_softmax_blocked_ref --
   max ABSOLUTE error <= 1e-6 -- over Task 2's full shape matrix.
2. Build and run on device exactly as Task 4 describes.

Paste the complete output.
```

---

### Task 6 — `S = Q·Kᵀ` through the existing layer_run, both widths

```
Read MHA_HTP_U8_TASKS.md §1.1-§1.6 and §1.8, and read hexkl_mm_u8i4_dma.{c,h} end
to end. Before writing code, restate what M, K, N, handles and out_cat mean for
the S matmul, and where out_cat's block-major layout puts row m of block j.

[paste Rule zero]
[paste the failure-signature table]

GOAL
New nntrainer/tensor/htp_backend/hmx/hexkl_attn_u8.{c,h} -- ONE file pair covering
both widths via Task 4's vtable -- with the KV registry and the S half:

  /* ctx holds two hexkl_attn_ops (one for Kt, one for V), the per-head block
     handle arrays, and the shapes. */
  int hexkl_attn_u8_ctx_init(hexkl_attn_u8_ctx *ctx, uint8_t *vtcm_base,
                             uint32_t vtcm_size, uint32_t config_off,
                             uint32_t nch, uint32_t gqa, uint32_t head_dim,
                             uint32_t max_kv, uint32_t T,
                             hexkl_w_width w_k, hexkl_w_width w_v);
  void hexkl_attn_u8_ctx_fini(hexkl_attn_u8_ctx *ctx);

  /* Registers/updates one KV block pair per head. Calls Task 1's
     hexkl_kvq_pack_* then ops->reg. Re-registering a tail block must
     ops->rel the previous handle first. */
  int hexkl_attn_u8_kv_append(hexkl_attn_u8_ctx *ctx, uint32_t kv_from,
                              uint32_t n_rows, const uint16_t *k_rows_f16,
                              const uint16_t *v_rows_f16);

  /* S band for one kv_head: ONE ops->run call, handles = that head's Kt blocks.
     out_s is block-major, [n_blocks][M][T]. */
  int hexkl_attn_u8_scores(hexkl_attn_u8_ctx *ctx, uint32_t head, uint32_t M,
                           const float *q_band, float *out_s);

DO NOT write a matmul. ops->run IS the matmul. If you find yourself calling
hexkl_micro_hmx_mm_u8i4 or _u8i8 directly you have taken a wrong turn -- stop and
report why the existing call did not fit.

Fold 1/sqrt(head_dim) nowhere; it belongs to the softmax's scale parameter.

MUST NOT TOUCH: Rule zero item 4's list.
BUDGET ~350 lines in the new files.

ACCEPTANCE
1. IDL: attn_register (taking w_k and w_v as parameters -- ONE entry point, not
   one per width; see §0.1) / attn_release / attn_kv_append, plus a DEBUG-ONLY
   attn_scores_debug returning the S band. Mirror how mm_u8i4_from_f32 returns
   intermediates as an accuracy harness.
2. gtest comparing attn_scores_debug against Task 3's host model's S over
   kv_len {32,33,256,257,1024} x n_query {1,33,128} x gqa {1,8} x
   head_dim {64,128} x T {64,256} x (w_k,w_v) all four. Tolerances: Task 3's.
3. Weight lifecycle: ctx_init -> kv_append across a block boundary -> scores ->
   ctx_fini -> ctx_init again gives identical results, at both widths.
4. Three consecutive device runs.

Paste all three runs. Report max relative error per (shape, w_k) as a table --
w_v does not affect S, and if your table shows it does, that is a bug in the
plumbing, not noise. Say so if you see it.
```

---

### Task 7 — `P·V` and the fused single-call forward

```
Read MHA_HTP_PLAN.md §2 (why exactly one FastRPC round trip per layer) and
MHA_HTP_U8_TASKS.md §1.1, §1.7, §1.8.

[paste Rule zero]
[paste the failure-signature table]

GOAL
Complete hexkl_attn_u8.{c,h}:

  int hexkl_attn_u8_forward(hexkl_attn_u8_ctx *ctx, uint32_t kv_from,
                            uint32_t n_query, float scale, int is_causal,
                            uint32_t window, const float *sink,
                            const float *q, float *out);

One call does everything: kv_append for the step rows, then per kv_head per
q-band -- scores (Task 6), hvx_softmax_blocked_f32 (Task 5), then P.V as
ops_v->run once per block with n_handles=1 and act_f32 = that block's P rows,
accumulating the f32 outputs, then one multiply by 1/l per row.

There must be exactly ONE FastRPC entry point involved in a forward pass. If your
implementation needs two round trips, stop and report why.

Per-block dynamic quantization of P is intended, not a compromise -- §1.7 says
why, and it matters more at i4. Do not add a fixed scale.

MUST NOT TOUCH: Rule zero item 4's list.
BUDGET ~250 lines added.

ACCEPTANCE
1. IDL: attn_forward per MHA_HTP_PLAN.md §2.1, plus attn_forward_debug returning
   S, P and O separately (§9.5 there explains why: without per-stage outputs a
   wrong O gives no signal about which of five stages produced it).
2. The debug path asserts, ON THE DSP SIDE, |sum_kv P[q][h] * (1/l) - 1| <= 1e-5
   for every (q, head). Report any violation as a failure. This single check
   catches nearly every masking, window, sink and range bug.
3. gtest against Task 3's host model AND against the fp32 CPU path, over Task 3's
   full matrix including all four (w_k, w_v) combinations, at Task 3's tolerances.
4. Three consecutive device runs.

Paste everything. Produce the per-shape, per-(w_k,w_v) max-relative-error table.
State plainly whether any shape failed and which failure signature it matched.
```

---

### Task 8 — masking and block skipping (window / sink), exactly

```
Read MHA_HTP_PLAN.md §5 in full and MHA_HTP_U8_TASKS.md §1.8.

[paste Rule zero]

GOAL
Skip whole KV blocks that fall entirely outside a row band's allowed range,
instead of computing and masking them. The allowed range is already Task 2's
begin/end plus the sink positions; this task only makes the block loop honour it.

Constraint from the DMA design: the skip list must be computed BEFORE the block
loop starts, because ops->run prefetches handle i+1's weight while i computes. A
skip decided inside the loop cannot be prefetched. So build the selected-handle
array first, then pass it to ops->run.

Choose M_band <= T so a band's rows never span more than one extra block
(MHA_HTP_PLAN.md §5.4 explains why). Assert it.

MUST NOT TOUCH the softmax kernel, the quantizer, the vtable, or any hexkl_mm_*
file.
BUDGET ~120 lines.

ACCEPTANCE
The gate is BIT-EXACTNESS, not a tolerance: for every (kv_len, window, w_k, w_v)
where Task 7 computed-and-masked, this version that skips must produce a BITWISE
IDENTICAL output. The surviving arithmetic is the same arithmetic, so any
difference is a bug. Cover window in {1, T-1, T, T+1, kv_len-1, kv_len}; the
{T-1, T, T+1} triple is the off-by-one trap and is mandatory.

Also report the measured SKIP RATE per configuration. A version that silently
skips nothing passes bit-exactness perfectly while delivering no speedup, and the
skip rate is the only thing that distinguishes the two.

Three consecutive device runs. Paste everything.
```

---

### Task 9 — the width/perf report: four combinations, one table

```
Read MHA_HTP_U8_TASKS.md §0.2 and §0.3. This task adds NO functionality. It
produces the numbers that decide what ships.

[paste Rule zero]

GOAL
Extend the gtest to sweep all four (w_k, w_v) combinations against the fp32 CPU
path, and to report both accuracy and cost per combination. No new kernels, no
optimization, no behaviour change.

ACCEPTANCE
One table, emitted as field=... value=... marker lines with units in the field
name (max_rel_err, us_per_layer, resident_kv_kib -- a report script keyed by
column position silently printed a stale number for every shape past the first,
twice, in this project):

  shape | w_k | w_v | max_rel_err_vs_fp32 | us_per_layer | resident_kv_kib
over kv_len {256, 1024} x n_query {1, 128} x all four combinations, three runs
each.

Then answer, in one short paragraph each, using only the numbers:
  1. Does K's width dominate the error, as §0.2 hypothesizes? Compare (I8,I4)
     against (I8,I8) and (I4,I8) against (I4,I4).
  2. Is (I8,I4) within the (I8,I8) tolerance? If yes, that is the interesting
     configuration -- state its resident_kv_kib saving.
  3. How does us_per_layer move with width, and is the direction consistent with
     §0.3's claim that this path is data-movement-bound?
Do NOT recommend a configuration. Report the numbers.
```

---

### Task 10 — performance: breakdown first, then one change at a time

```
Read MHA_HTP_PLAN.md §9.7 in full. This task must not change any output value.

[paste Rule zero]

GOAL, in this order, as separate commits:
1. Per-stage timing behind a build flag: kv_append / quant / qk / softmax / pv /
   dequant / out. Emit field=... value=... marker lines. Do this FIRST and report
   the breakdown before changing anything. The last time this project acted on a
   performance hypothesis before measuring the breakdown, the hypothesis was wrong
   (FastRPC marshalling, guessed dominant, measured 16-32%) and the real cost was
   elsewhere at 92%.
2. THEN, guided only by that breakdown, at most three changes, each its own
   commit, each device-verified independently before the next. Likely candidates
   in the order §1.8 predicts:
     - hoist ops->run's per-call mallocs (act_scale, act_zp, acc_scratch) out of
       the per-block loop. If you allocate a buffer you then index as a vector
       type, it MUST be HVX_UVector*, never HVX_Vector* -- malloc gives no
       128-byte guarantee and this crashed the whole DSP process the last time it
       was forgotten.
     - run hvx_softmax_blocked_f32 on hvx_worker_pool_run (see
       nntrainer/tensor/htp_backend/hvx/hvx_worker_pool.h). Before splitting,
       check the row-range split does not fight a vectorization axis inside the
       kernel -- that exact mistake was caught once already, where a row split
       would have scattered a vectorized 4-row store group.
     - a larger T if the breakdown shows DMA-bound behaviour, AND note that the
       right T may differ between i4 and i8 because the slab size does (§1.8).
       Measured curve to aim at: 64 B rows 8.4 GB/s, 2 KiB 42, 64 KiB 55,
       256 KiB 67.

RULES
- Correctness checks OUTSIDE the timed region. A diff check left inside the timed
  window on the last iteration once produced a phantom "146 us slowdown" here.
- Every commit: three consecutive device runs, all three printed, and BITWISE
  IDENTICAL output to the commit before it, at all four (w_k, w_v) combinations.
  A perf change that alters a value is a behaviour change and belongs elsewhere.
- Never assert a speed threshold in a test. Print it.

ACCEPTANCE
The stage breakdown before and after, three runs each, plus the bitwise-identical
proof per commit. Paste everything. If a change made things slower, keep the
measurement and revert the change -- do not explain the number away.
```

---

### Task 11 — the PR: strip the scaffolding, rebuild the branch clean

```
Read MHA_HTP_PLAN.md §9.5's last paragraph and HTP_PR_PLAN.md §6 (commit and PR
conventions) before starting.

GOAL
A clean branch carrying only shipped functionality: no debug endpoints, no
per-stage timing IDL parameters, no *_debug entries.

Do NOT revert or rebase the debug commits out. That entanglement was untangled
once already and the faster, safer route was measured to be:
  1. branch fresh from the last clean pushed tip,
  2. re-apply each change's actual code by hand against that clean baseline
     (git show <work-branch>:<file> where scaffolding lived in a different file,
     git checkout <work-branch> -- <file> where the file was already clean),
  3. device-verify each resulting commit INDEPENDENTLY before the next.

Commit format per HTP_PR_PLAN.md §6, bodies 2-4 lines with numbers.
The PR body must carry: Task 9's four-combination table, Task 10's breakdown, the
statement that decode is not a target on this path and why (§1.8's m_pad=64
utilisation plus MHA_HTP_PLAN.md §2's transport arithmetic), the §0.1 note that
the attention module is dtype-parametric while hexkl_mm_u8i*_dma stayed
duplicated and why that is not inconsistent, and Depends on #4245.

ACCEPTANCE
git log --oneline of the clean commits; a grep proving no *_debug entry or timing
parameter survives in the IDL; one final full device run, three times, all four
width combinations. Paste all of it.
```

---

## 4. What to watch for as the reviewer

The five places a weak-verification implementer will most plausibly produce
something that looks finished and is not:

1. **Task 1's V scale axis.** If it computes a per-column scale over anything
   other than *that block's* rows, everything downstream is subtly wrong and
   Tasks 3/7 may still pass on friendly data. Read the code, not the test result.
2. **Task 1's i4 range.** `[-8, 7]` is asymmetric; a symmetric scale with an
   asymmetric range wastes a code and biases the result. Check the clamp and the
   scale denominator agree.
3. **Task 3's host model diverging from the kernel's operation order.** If it
   does, Tasks 7 and 8's bit-exactness gates become meaningless while still
   reporting green. Read the two loops side by side.
4. **The partial tail block.** Zero-fill *and* mask, both. A matrix that only uses
   `kv_len % T == 0` will never catch it — verify the odd sizes actually ran, by
   count.
5. **Task 8's skip rate.** Bit-exactness is easy to pass by skipping nothing. The
   skip rate is the real deliverable of that task.

And one specific to the two-width design: **the vtable must not appear in the
tile loop.** An indirect call per 32×32 tile would be a real regression that no
correctness test can see. Grep the inner loop for `ops->`.

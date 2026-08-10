# Flash attention on HTP — the task, self-contained

Read `00_START_HERE.md` and `01_working_style.md` first. This file is the task
itself. It is written so that a session with **only this repo and no Hexagon
device** can do it and can *verify* it.

---

## 0. What "flash attention" means here, and what it does not

Do not implement textbook flash attention. Two hardware facts make the classic
structure either impossible or unnecessary on this part:

**Online softmax with `O` living in the accumulator is not expressible.** There
is exactly **one** HMX accumulator (32×32 fp16, or 64×32 int32 on the integer
path), and both `Q·Kᵀ` and `P·V` use it — see
`~/workspace/hxkl-beta2/hexkl_addon/include/hexkl_micro.h`, and
`10_mha_htp_plan.md` §4.1. There is no accumulator-scaling primitive either.
Keeping `O` in the accumulator across KV tiles and rescaling it by
`exp(m_old − m_new)` is not slow here; it cannot be written.

**Online softmax is not needed at our context lengths anyway.** The whole
`[M_band][kv_len]` score band fits in VTCM — 128 KiB at `M_band=64, kv_len=1024`
against an 8 MiB arena. So a two-pass exact softmax is simpler, exact, reuses an
already-device-verified kernel, and — the real prize — lets `P·V` accumulate
over the **entire** `kv_len` because `P`'s whole row is available before `P·V`
starts. The crossover where online softmax becomes necessary is worked out in
`10_mha_htp_plan.md` §4.5 and sits around `kv_len ≈ 16 K`.

**So flash attention here = fusion + KV streaming + band tiling:**

- the three stages run inside **one** FastRPC call, so `S` and `P` never cross
  the host boundary and their layout is ours to choose;
- `K` and `V` are streamed **one block at a time** into VTCM instead of being
  resident, which is what lifts the `kv_len ≈ 1024` VTCM wall;
- queries are processed in **bands** of `M_band` rows so the score band is
  bounded;
- the block structure is what later block-sparsity skips (`10_mha_htp_plan.md`
  §5), so getting the block boundaries right now is what makes sparsity a
  hundred-line change later.

State this in your final report in your own words. If your design has a running
max being rescaled per tile, you have built the wrong thing.

## 1. The loop, exactly

Per attention layer, per kv_head `n`, per q-band `b` (`M = n_query · gqa` rows
folded, banded by `M_band`):

```
PHASE A — scores, streaming the K blocks
  for each kv block j in the allowed set:
      S_j = Q_band(u8, per-row scale/zp) · Ktᵀ_j(i4|i8, per-kv-position scale)
            -> int32 -> dequant -> f32
  result: S band, BLOCK-MAJOR: [n_blocks][M_band][T]

PHASE B — one masked softmax pass over the whole band
      P = exp(S*scale - rowmax)          <- UNNORMALIZED; row max is exactly 1.0
      l[m] = sum over the row's valid range (+ sink term when present)
  masking is per row: causal, sliding window, attention sink

PHASE C — output, streaming the V blocks
  for each kv block j in the allowed set:
      O += P_j(u8, per-row-per-block scale) · V_j(i4|i8, per-column scale)
           -> int32 -> dequant -> f32 accumulate
  O[m][:] *= 1/l[m]
```

Five properties of this loop are load-bearing. A later device kernel will be
checked **bit-exactly** against your host model, so these must be right:

1. **Block-major intermediates, no repack.** `hexkl_mm_u8i4_layer_run` returns
   one contiguous `M × N` block per weight handle in call order — row `m` of
   block `j` lives at `off_j + m*T`. Do not flatten it into `[M][kv]`. The
   block-major layout *is* the KV tiling.
2. **`P` is quantized per block.** Each block's `P` rows get their own scale;
   that is correct and desirable, not a compromise — after Phase B the row
   maximum is exactly 1.0, so the u8 codes are well used, and it matters more at
   i4 than i8.
3. **`O` accumulates in f32 across blocks**, because each block's dequant
   constants differ.
4. **The `1/l` normalization happens once, at the very end**, not as a third
   softmax pass. This is why Phase B emits unnormalized values plus `l`.
5. **`M_band ≤ T`**, so a band's rows never span more than one extra block once
   masking restricts them (`10_mha_htp_plan.md` §5.4).

`S = Q·Kᵀ` with `Q` as the activation and `Kᵀ` as the weight — **not**
`Sᵀ = K·Qᵀ` — is deliberate, and `10_mha_htp_plan.md` §3 explains it: same HMX
tile-multiply count, but zero transposes, and `S` lands in the orientation both
the softmax and the `P·V` activation slot want. A transpose per tile costs
~3.7 ms/layer at prefill. Do not reintroduce one.

## 2. Your deliverable, and the honest constraint on it

**You cannot verify a DSP kernel.** This task needs the Hexagon SDK,
`hexagon-clang`, and a physical Galaxy S25 Ultra. If you write DSP code you
cannot run it, and "verified by inspection" is not verification — the last agent
to work on this branch produced arithmetically correct code that could never
enter the DSP skel, and it looked green because the only place it was compiled
was a host gtest.

So the deliverable is the part that is both the highest-leverage and fully
verifiable where you are: **the flash loop's executable specification, and its
proof against the CPU implementation.** Concretely, two things.

### Deliverable 1 — the blocked masked softmax reference

`test/unittest/mha_htp_host_model.{cpp,h}` (new):

```cpp
struct MhaSoftmaxOut { std::vector<float> p; std::vector<float> l; };

/* Block-major scores: segment j is s[off_j + m*T + t], t in [0, T).
   Softmax runs along the kv axis, ACROSS segments, for each row m.
   p holds UNNORMALIZED exp values (row max exactly 1.0);
   l[m] is the row denominator, including the sink term when present. */
MhaSoftmaxOut mha_softmax_blocked_ref(
    const std::vector<const float *> &seg, uint32_t n_seg, uint32_t T,
    uint32_t M, float scale,
    const std::vector<uint32_t> &begin, const std::vector<uint32_t> &end,
    const float *sink /* nullptr, or one logit per row */);
```

Masked positions must be excluded from the **maximum** pass, not only from the
sum. The HVX `exp` this mirrors (`hvx_exp_f32.h` on PR #4245) is **undefined
above +87.3**; a masked lane that skips the max pass but reaches `exp` is
undefined behaviour, not a small numerical error. Put that reasoning in a
comment.

### Deliverable 2 — the whole-kernel host model

Same files, extended:

```cpp
void mha_htp_host_forward(
    uint32_t n_query, uint32_t kv_from, uint32_t nch, uint32_t gqa,
    uint32_t head_dim, uint32_t T, uint32_t M_band,
    bool is_causal, uint32_t window, const float *sink,
    hexkl_w_width w_k, hexkl_w_width w_v,
    const float *q, const uint16_t *k_cache, const uint16_t *v_cache,
    float *out);
```

A scalar model of §1's loop: same band order, same block order, same
quantization points, same mask arithmetic — plain arithmetic instead of
HMX/HVX/DMA. It **calls** `hexkl_kvq_pack_kt_block` /
`hexkl_kvq_pack_v_block` (already in this tree at
`nntrainer/tensor/htp_backend/hmx/hexkl_kv_quant.h`) and your
`mha_softmax_blocked_ref`. It does not reimplement either.

Two properties matter more than anything else:

- **the order of operations must be the order the DSP kernel will use**, so the
  later bit-exactness comparison is meaningful;
- **every quantization step must be present**, so the model's error equals the
  kernel's expected error rather than an fp32 upper bound.

This is a *specification*, not a second implementation. Clarity over speed.

## 3. What exists already — use it, do not rewrite it

| you need | it is at |
| :-- | :-- |
| KV block quantizer (fp16 K/V rows → int8 containers + per-N scale + colsum) | `nntrainer/tensor/htp_backend/hmx/hexkl_kv_quant.h` |
| the CPU ground truth for scores | `compute_kcaches_fp32_reference`, `Applications/CausalLM/layers/mha_core.cpp:42` |
| the CPU ground truth for `P·V` | `compute_vcache_fp32_transposed_reference`, same file `:73` |
| the CPU softmax, including the sink overload | `nntrainer::softmax_row` / `softmax_row_inplace` |
| how the CPU packs scores, and the window/sink semantics | `mha_core.cpp` `softmax_triangle` (`:1176`) and `one_batch_incremental_forwarding` (`:693`) |
| the KV cache layout | `[kv_position][nch][head_dim]` fp16; row `r` of head `h` at `(r*nch + h)*head_dim` — `mha_core.cpp:61`, `:95` |
| the dequant contract the quantizer feeds | `nntrainer/tensor/htp_backend/hvx/hvx_dequant_i32.h` |
| the matmul call the device kernel will use | `hexkl_mm_u8i4_layer_run`, `hmx/hexkl_mm_u8i4_dma.h` — read its contract; your block loop must match its `out_cat` layout |

**Known, and not your problem:** `hexkl_kv_quant` is currently a `.cpp` and
depends on `nntrainer::compute_fp16_to_fp32`, which means it cannot enter the
DSP skel as written. Its *arithmetic* is correct and tested, which is all this
task needs. The fix is tracked separately (`00_START_HERE.md` §5). Do not fix it
here and do not build around it.

## 4. Acceptance

`test/unittest/unittest_mha_htp_host_model.cpp` (new), registered in
`test/unittest/meson.build` the way its neighbours are.

**Softmax reference:**

- a) `begin=0, end=kv`, no sink, `n_seg=1` → `p[m][*]/l[m]` equals
  `nntrainer::softmax_row_inplace` on the same data within `1e-6`
- b) `sum(p[m][*]) == l[m]` within `1e-5` in **every** configuration — and
  assert it inside the reference itself, not only in the test
- c) `max(p[m][*]) == 1.0f` exactly whenever row `m` has a valid position
- d) window and sink match `nntrainer::softmax_row`'s sink overload
- e) **the segment split is invisible**: the same logical scores split into
  `n_seg = 1, 2, 4` give **bitwise identical** `p` and `l`

(e) is the most important case in this task. It is what proves the KV tiling
did not change the mathematics, and it is the property the whole flash structure
rests on.

**Whole-kernel model**, against the fp32 ground truth
(`compute_kcaches_fp32_reference` → `softmax_row_inplace` →
`compute_vcache_fp32_transposed_reference`):

```
kv_len    1, 31, 32, 33, 64, 255, 256, 257, 1023, 1024
n_query   1, 7, 32, 33, 128
gqa       1, 4, 8
nch       1, 8
head_dim  64, 128
T         64, 256
M_band    64, 256          (assert M_band <= T)
is_causal true, false
window    0, 1, T-1, T, T+1, kv_len-1, kv_len
sink      absent, present
(w_k,w_v) (I8,I8), (I4,I4), (I8,I4), (I4,I8)
```

Every value of every axis must appear at least twice, and every
`(kv_len % 32 != 0) × (n_query % 32 != 0)` combination at least once. The
`window ∈ {T-1, T, T+1}` triple is the off-by-one trap in the block arithmetic
and is mandatory.

Tolerances on `out`, max relative error — **fixed here, do not change them:**

| `(w_k, w_v)` | bound |
| :-- | --: |
| (I8, I8) | 5e-3 |
| (I8, I4) | 2e-2 |
| (I4, I8) | 5e-2 |
| (I4, I4) | 5e-2 |

A configuration outside its bound is a **finding to report**, not a bound to
raise. Diagnose it with this table rather than adjusting anything:

> Error ~10× larger at i4 than i8 on the same shape → the width is doing what it
> should; not a bug. Error **the same at both widths** → not a quantization
> problem; look at layout, masking, or an index calculation. ≥1e-1, or a
> blocky/structured rather than scattered error pattern → a layout bug.

**Deliverable beyond pass/fail:** a printed table of observed max relative error
per `(shape, w_k, w_v)`. That table is the first real evidence on whether `K`'s
width dominates the error — compare `(I8,I4)` against `(I8,I8)`, and `(I4,I8)`
against `(I4,I4)`.

## 5. Build and run

```bash
git submodule sync && git submodule update --init --depth 1   # first time only
meson build -Denable-transformer=true
ninja -C build
cd build && meson test unittest_mha_htp_host_model --print-errorlogs
```

Use `build`, not `builddir` — the checked-in `builddir` on the original machine
is configured for an Android cross build and cannot run host tests. Paste the
complete output, including the command line. Not a summary.

## 6. Rules

1. Do not modify any reference implementation:
   `compute_kcaches_fp32_reference`, `compute_vcache_fp32_transposed_reference`,
   `nntrainer::softmax_row` / `softmax_row_inplace`. If a test fails, the new
   code is wrong.
2. Do not change a tolerance, a shape, or the test matrix. They were fixed
   before any code existed, so that they cannot be adjusted to fit a result.
3. Do not write DSP code, HVX intrinsics, or FastRPC/IDL entries in this task.
   You cannot verify them. If you believe the task cannot be completed without
   them, stop and say so.
4. Do not modify anything under `nntrainer/tensor/htp_backend/` — all of it is
   device-verified and this task does not require changing any of it. If you
   believe otherwise, stop and report which line and why.
5. Do not "fix" a failing test by loosening it, skipping a case, or narrowing
   the matrix. Stop and report with the complete output.
6. Follow `AGENTS.md`: `git commit -s`, `Co-authored-by:` trailer,
   `[<component>] <description>` subject, `clang-format-14` on changed lines.
7. Read `01_working_style.md` and work that way. In particular: the two things
   above that say "already in this tree — call it, do not reimplement it" are
   rung 2 of that ladder and are the most likely thing to be got wrong.

## 7. If you have capacity left

Do **not** start the device kernel. Instead, the highest-value follow-on that is
still verifiable where you are: a `ponytail:`-marked note in the host model
naming, for each of §1's five load-bearing properties, the single assertion a
later device test should use to check the kernel reproduces it. That turns your
specification into the device task's test plan, which is the artifact the next
session actually needs.

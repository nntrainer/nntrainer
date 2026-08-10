<!-- SPDX-License-Identifier: Apache-2.0 -->

# 36 -- What to build, read off the QNN optrace

Source: a QAIRT 2.47 HTP optrace of ONE Qwen3-0.6B attention block, seq 1024,
w8 (`Qwen3-0.6B_attn_quant_w8_1024_chromeTrace_opTrace.json`), analysed in a
separate session. **That analysis is kept verbatim in
`ref_16_qnn_optrace_analysis.md`** -- every section number cited below (§1.5,
§2.1, §7.3, ...) refers to it. This doc does not repeat it: it maps it onto
OUR codebase and orders the work by what it is worth to US.

Note the two QNN numbers are not the same measurement: this trace says
**4.72 ms** for the block (5,666,199 cycles @ 1.2 GHz), the org comparison
table says 3.21 ms at seq 1024. Different config or device. Do not mix them.

## 1. The trace confirms our own diagnosis

| | QNN (trace) | ours (doc 35 §2) |
|---|---|---|
| bound by | **HVX** -- HMX only 9.5% busy | **HVX** -- 66% of block time |
| what fills the vector lane | elementwise, softmax, transpose, layout | quant, dequant, softmax, gather |
| idle | 0.09% | not measured (single-lane, so ~0 by construction) |
| parallel compression | 4.1x | 1.0x (no HMX/HVX overlap at all) |

**Both stacks agree the matrix engine is not the problem.** Any plan that
starts from "make the matmul faster" is aimed at 9.5% of their time and 34%
of ours. This is the strongest cross-validation we have of doc 34 §5 and doc
35 §3, and it should end that argument.

The one number that has no counterpart on our side is **4.1x parallel
compression**. We run one lane at a time. That is the whole difference in a
single figure.

## 2. Three things we already do better -- do not rebuild them

The trace's §9 lists what QNN got wrong. We structurally do not do any of it:

| QNN cost | unit-cycles | share | why we do not pay it |
|---|---|---|---|
| `mul_op` -- GQA `repeat_kv` as a broadcast multiply | 8,011,003 | 34.7% | we fold gqa into the band's M (`M = n_query * gqa`, `hexkl_attn_u8.h:152`); the gqa query heads of a kv head share one K. **KV is never replicated.** |
| `Transpose_115` -- transposing K | 1,479,786 | 6.4% | Kt is baked to WH layout at `kv_append`; QK^T is a plain matmul against a registered weight |
| `@Spill` / `@Fill` -- the 16 MB QK^T matrix evicting from VTCM | 509,321 | 2.2% | band tiling: our S band is 256 KB, never materialised whole |
| **total** | **10,000,110** | **43.3%** | |

So **QNN's 4.72 ms contains ~2.0 ms of work we do not do at all.** Their
irreducible core is roughly **2.68 ms**, and that -- not 4.72 -- is the bar.
Against our derived 12.9 ms that is a **4.8x** gap.

Also already done, don't redo: weight pre-shuffle to the hardware layout
(their §2.3 = our WH bake at register), async DMA prefetch with double
buffering (§5.2 = our DMA ring), and dynamic matmul reusing the static-weight
kernel (§3.1 = our `hexkl_mm_u8iX` for QK^T and P·V both). The trace
validates three design choices we already made.

## 3. What to build, in order

### T1 -- Integer requantize: get float out of the data path

**Their evidence (§2.1, §2.2):** the entire scale-preparation chain --
`combine_scales`, `invscale_to_qi32`, `bias_update_and_fused_shuffle` -- runs
**once, outside the tile loop**. The zero-point correction
`Σ(x-zx)(w-zw) = Σxw - zx·Σw - zw·Σx + N·zx·zw` has its constant terms folded
into an int32 bias at compile time. Result: **there is not one float dequant
kernel in their data path.** HMX output becomes uint8 in one integer step:
`(int32_acc + folded_bias) * multiplier >> shift`.

**Ours:** we compute `(acc - zp·colsum) · act_scale · w_scale + bias` in
**float**, per tile, on HVX, and then write f32 to DDR so the next op can
re-quantize it. Measured: quant + dequant is **53% of FC dsp** and **~6.6 ms
of the 12.9 ms block**.

Three sub-items, in dependency order:

1. **Fold `act_zp·colsum_w` into an int32 bias.** We already precompute
   `colsum_w`; the correction is still applied in float per element. Folding
   is arithmetic we can do once per call in the quant step.
2. **Replace the float dequant with `(acc + bias) * multiplier >> shift`.**
   Integer multiplier and shift derived once per call from
   `act_scale · w_scale / out_scale`, exactly their `scale_convert`.
3. **u8-in / u8-out endpoints** so op boundaries stop round-tripping through
   f32 DDR. `hexkl_mm_opts` already accepts caller-supplied `act_scale` /
   `act_zp`, so half the plumbing exists. Also cuts the FastRPC payload 4x.

**Worth ~4.4 ms of 12.9.** Biggest single item, and a prerequisite for T2
being worth tuning.

### T2 -- Tile-level pipelining so HMX and HVX both run

**Their evidence (§7.3):** QK^T (HMX, 256 tiles), mask (HVX), softmax (HVX),
P·V (HMX, 64 tiles) have overlapping spans and their tiles stay interleaved
through the whole window -- 21 QK^T tiles, 12 mask, 12 softmax, 4 PV in one
50k-cycle bucket. HVX does softmax on block j while HMX is still producing
block j+1. **This requires tile-granular dependencies** (§8: token/fence
sync, 0.66% overhead), not node barriers.

**Ours:** the fused attention loop is strictly `qk(j) -> softmax(j) -> pv(j)`.
One lane at a time, always.

Design and staging are doc 35 §4-§7 unchanged, with one correction it now
gets from this trace: **do it in attention first, not FC.** Attention has a
natural 3-stage pipeline (QK^T / softmax / P·V) where FC has 2, and after T1
the attention lanes are the ones that balance.

Keep doc 35 §4(a): **HMX stays on the caller thread**, dequant/softmax go
async to the existing pool. ggml locks HMX inside its own thread, which means
the lock is thread-affine and ours is taken on the open thread -- inverting
the topology avoids that question entirely.

**Worth ~1.85x on whatever T1 leaves** -- block-scope. Confined to SDPA it is
only 1.59x, because attention's own HMX lane is 37% of its post-T1 total and
overlap cannot hide more than the smaller lane holds. That gap is why T3 comes
first. **Full design, including why only ONE of the two candidate mechanisms
should be built: `37_t2_pipelining_design.md`.**

### T3 -- One DSP call for the whole attention block

**Their evidence (§7.1, §7.2):** node spans overlap 28-deep; V projection's
HMX runs at ts 131,963 while Q's runs at 1,204,562 -- **V is scheduled a
million cycles early because its dependency chain is the longest**
(V→transpose→expand→PV, with no RMSNorm and no RoPE in front of it).

That is only possible when one scheduler sees the whole block. **We issue
three separate FastRPC calls** (q/k/v, SDPA, o_proj). No scheduler can
overlap o_proj's weight DMA with SDPA's tail across a FastRPC return.

Fusing the block into one DSP entry point:
- removes 2 of 3 transports (FC measured 963 us/call at M=1024),
- is the precondition for T4,
- lets the shared activation be quantized once instead of three times.

### T4 -- Critical-path-first ordering inside the fused block

Only actionable after T3, and cheap once there: order the sub-ops by remaining
path cost rather than by source order. Our longest chain is the same as
theirs -- V projection feeds P·V, with nothing in front of it -- so V should
issue first. Their §7.2 is worth "수십만 cycle" on their graph; ours is worth
measuring, not guessing.

### T5 -- Adopt their per-kernel efficiency metric

The trace's §3.3 gives **cycles per element** per kernel, and that is how they
found their own `mul_op` at 1.52 cy/elem against a 0.07 norm -- a 20x outlier
hiding inside a graph that looked fine in aggregate. Their reference points:

| | cy/elem |
|---|---|
| HMX conv | 0.01 - 0.06 |
| healthy HVX elementwise | 0.07 - 0.10 |
| softmax (incl. exp) | 0.24 |
| pathological | 1.52 |

Also `Cycles per Packet`: 2.0-2.4 is healthy dual-issue, >6 means stalls.

**We have no equivalent.** Our probes are wall-microseconds per stage, which
cannot tell "this kernel is slow" from "this kernel has a lot of work". Add
cy/elem to the FC and attention reports -- it is arithmetic over numbers we
already emit plus the shapes, and it is the cheapest item on this list.

### T6 -- Audit tile counts against thread count

Their §4.2 rule: **tiles ≥ threads × 4**, or the tail thread stalls.
`node_Transpose_115` got 1.1x from 8 threads because it had exactly 8 tiles.
Our `hvx_quant_pack_u8_ah` splits over `n_ktiles` = K/32 = **32 units on 6
threads = 5.3x** -- above the rule but not by much, and it drops below it for
any K < 768. Worth one pass over every `hvx_worker_pool_run` call site.

## 4. What NOT to build

The trace's Tier-1 list is mostly **compiler** features -- view-based tensor
IR, pattern fusion, constant folding, automatic spill/fill, layout
propagation. We are not writing a compiler; we hand-write kernels for a fixed
graph, so:

| their item | our position |
|---|---|
| view-based tensor IR (§1.5) | N/A -- no graph to rewrite. Applies only if we ever take ONNX directly. |
| pattern fusion, RMSNorm 7→1 (§1.1) | our attention is already one fused call; RMSNorm/RoPE are not on the DSP at all yet. If they arrive, arrive fused. |
| constant folding of weight transposes (§1.2) | already done at `weight_register` |
| automatic spill/fill (§5.3) | avoided by band tiling; adding it would be building a solution to a problem we designed out |
| layout propagation / Crouton (§6) | our AH/WH layouts + in-place dequant are this. The remaining conversion is the f32 boundary, which is T1. |

## 5. Order of operations

    T5 (metrics)  ->  T1 (integer path)  ->  T3 (fuse block)  ->  T2 (pipeline)  ->  T4  ->  T6

T5 first because it is hours of work and tells us whether we have a `mul_op`
of our own -- a single 20x outlier would reorder everything below it. T1
before T2 because overlap tuned around work T1 deletes is tuning thrown away
(doc 35 §3). T3 before T2's cross-op half, since a FastRPC return is a barrier
no scheduler can cross.

Targets to hold ourselves to, from their §10:

| metric | QNN | ours today | target |
|---|---|---|---|
| parallel compression | 4.1x | 1.0x | ≥ 2x after T2 |
| idle | 0.09% | -- | < 5% |
| sync overhead | 0.66% | -- | < 2% |
| elementwise cy/elem | 0.07 | unknown | ≤ 0.15 |

The 4.1x is against 8 HVX threads plus HMX plus DMA. Our pool is 6 contexts,
so parity in compression is not the goal -- parity in **wall time against
their 2.68 ms core** is.

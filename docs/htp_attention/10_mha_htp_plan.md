# MHA-core on HTP — fused single-RPC attention, then flash, then sparse

Working plan, not repo documentation. Written 2026-08-06.
Reader: whoever implements this (GLM 5.2 via opencode) **and** the reviewer.
Read §0–§3 before writing a line of code. §9 is not optional; it is the part
that decides whether the result can be trusted.

Companion documents (all already exist, do not re-derive them):

| doc | what it settles |
| :-- | :-- |
| `HTP_PR_PLAN.md` (this repo, untracked) | branch layout, device recipe, PR①/②/③ state, environment gotchas |
| `docs/backend_guide/htp_backend/08_attention_hmx_design.md` (on `claude/hexkl-mha-hmx-optimization-6ycsx0`) | RM/AH/WH layouts, WH-cache-in-host-RAM decision, probe inventory |
| `…/13_fc_micro_dma_and_mha_plan.md` (same branch) | FC micro+DMA numbers, cross-matmul prefetch |
| `…/14_mha_fp16_micro_verified.md` (same branch) | **the measured MHA baseline** — decode *and prefill* numbers, DMA rules, fp16 accuracy |
| `…/15_handoff_seam_and_fastrpc.md` (same branch) | FastRPC per-call cost, macro/micro one-way door |

Extract them once with
`git show claude/hexkl-mha-hmx-optimization-6ycsx0:docs/backend_guide/htp_backend/14_mha_fp16_micro_verified.md`.

---

## 0. Decisions this document makes, up front

1. **Prefill is the target. Decode is a correctness target, not a perf target,
   until the transport work lands.** With one FastRPC call per attention layer,
   transport is ~28% of a prefill call and ~49% of a decode call (§2). Prefill
   keeps ~30× over CPU; decode's measured 2.4–3.4× collapses to roughly 1.4×.
2. **One formulation for decode and prefill, and it is *not* doc 14's.** Compute
   `S = Q·Kᵀ` (act = Q, weight = **Kᵀ in WH**), not doc 14's `Sᵀ = K·Qᵀ`. Same
   number of HMX tile-multiplies, but it removes every transpose from the chain
   and puts `S` in exactly the orientation both the HVX softmax and the P·V
   activation slot want (§3). Doc 14's orientation was right for a
   CPU-softmax-in-the-middle design; it is the wrong one for a fused design.
3. **"Flash attention" here is K/V streaming + fusion, not online softmax.**
   At our shapes the whole `[M_band][kv_len]` score band fits in VTCM
   (128 KiB at `M_band=64, kv=1024`), so a two-pass exact softmax is
   simpler, more accurate, reuses PR #4245's kernel as-is, and lets `P·V`
   accumulate over the *entire* `kv_len` in one HMX accumulator lifetime — no
   per-tile rescale tax at all. Online softmax becomes necessary only past
   `kv_len ≈ 16K` (§4.5 has the crossover). Build it then, not now.
4. **Sparsity skips at `(kv_head, kv_block)` granularity, block ≥ 64 positions,
   and the skip list must be known before the DMA ring is primed** (§5). The
   first increment is free and exact: honour the `local_window_size` / sink
   semantics `mha_core` *already* declares, as a tile-range restriction.
5. **FC migration is not a prerequisite.** On this branch lineage there is no
   SDKL-macro FC left to migrate; the only macro-API call remaining is a version
   print in `HtpBackend`'s constructor. Delete it (≈20 lines) and the one-way
   door stops being a hazard permanently (§6).
6. **The likely dominant cost at prefill is softmax, not the matmuls.**
   Estimated 0.8–1.3 ms/layer of HVX `exp` against 924 µs of measured HMX
   matmul. Four independent mitigations exist; the point is that this must be
   measured in Stage 0, not discovered in Stage 2 (§7.3).

---

## 1. Established facts — evidence, not inference

Cited so the implementer does not re-measure them and does not contradict them.

| # | fact | evidence |
| --: | :-- | :-- |
| F1 | **Both MHA matmuls already beat the CPU at prefill shape**: 128-token chunk, kv=1024 — Q·Kᵀ 553–558 µs vs 23.9–36.1 ms CPU; P·V 371–380 µs vs 16.3–20.9 ms CPU (43–65×) | doc14 §3 |
| F2 | Decode: Q·Kᵀ 239 µs vs 553–590 µs CPU (2.4×); P·V 180 µs vs 620 µs (3.4×) | doc14 §3 |
| F3 | **fp16 HMX accumulation over K=1024 is safe**: maxRelErr 3e-4 vs an fp32-accumulating reference. Per-term-fp16-rounded reference gives 9e-3, i.e. the accumulator carries more than fp16 of internal precision. No K-chunking needed | doc14 §4 |
| F4 | **DMA row size is the whole game**: 64 B rows → 8.4 GB/s; 2 KiB → 42; 64 KiB → 55; 256 KiB → 67 GB/s. Descriptor *count* barely matters | doc14 §5.1 |
| F5 | VTCM→VTCM is ~4× DDR→VTCM at identical geometry (31.6 vs 8.4 GB/s for a 64 B-row repack) | doc14 §5.2 |
| F6 | **Extraction-style micro functions (`row_tile/col_tile/input_rows/input_cols`: `rm_to_wh_f16`, `copy_submatrix_to_f16`) must read DDR, never VTCM** — ~26 µs/tile from VTCM vs ~1–3 µs from DDR; one case 3376 µs vs 8.3 µs. Pure-format ops (`rm_to_ah_f16`, `ah_to_rm_f16`, fixed 32×32) are fast from either | doc14 §5.3 |
| F7 | A single kv_head's stride is one 2D DMA descriptor parameter, not a gather step | doc14 §5.4 |
| F8 | **Incremental WH bake at KV-append time is what makes P·V win**: full V re-bake 123.7 µs/head vs 3.35 µs/head for the newly-dirtied k-tile (32×) | doc14 §5.5 |
| F9 | Cross-call weight prefetch retargets to attention: prefetch kv_head *h+1*'s slab while *h* computes — same dmlink ring as FC | doc14 §5.6, doc13 §3a |
| F10 | **VTCM is 8 MiB** on this device (`hexkl_micro_hw_init`, every run). `sdkl_npu_get_hw_info`'s 18 MiB disagrees and is unresolved — budget against 8 MiB | doc14 §7 |
| F11 | **There is exactly one 32×32 fp16 HMX accumulator**, no banks, shared with the int32 path. `mm_f16` accumulates into it; `acc_clear_f16` / `acc_read_f16` are the only controls | `hexkl_micro.h:344–515` (beta2) |
| F12 | **There is no `ah_to_wh`.** An accumulator result (AH) cannot become a weight without going through RM | doc08 §3, header inventory |
| F13 | FastRPC costs **~404 µs/call fixed + 0.019 µs/KB** (re-measured after the P0 scalar-copy fix removed the contamination in the earlier 670 µs + 0.59 µs/KB estimate) | `/tmp/htp_plan/P1_QUANT_DEQUANT_MARSHALLING.md`, memory `htp-fastrpc-seam-next-steps` |
| F14 | The macro/micro door is **one-way and irreversible**: macro→micro works (556 µs, every time); anything calling `hexkl_macro_initialize()` after a successful `hexkl_micro_hw_init()` fails permanently with `0x80000401`, no backoff recovers | memory `htp-macro-micro-oneway-door` |
| F15 | An HVX worker pool exists and is designed for reuse: `hvx_worker_pool_run(pool, func, ctx, n_units)` with `func(n_threads, i, ctx)`. Device has **6 HVX contexts, 8 hw threads** | `origin/htp/quant-dequant-hvx-opt:nntrainer/tensor/htp_backend/hvx/hvx_worker_pool.h` |
| F16 | **PR #4245 contains PR①'s u8i4 commits *plus* an HVX f32 softmax and a vector `exp`**, and its last two commits adapt softmax to take the session handle PR① introduced. Its `hvx_softmax_rows_f32(x, y, m_first, m_last, k, scale)` is 3-pass exact, in-place-safe, row-range-parameterised (worker-pool ready), dense rows of stride `k` | `gh pr view 4245`; `pr4245:nntrainer/tensor/htp_backend/hvx/hvx_softmax_f32.c` |
| F17 | `hvx_exp_sf` is a 7th-order qf32 polynomial, rel err ≤1e-6 for x ≥ −87.3, returns 0 below −87.7, **undefined above 88.7 (no overflow clamp)** and undefined on NaN/Inf | `pr4245:…/hvx/hvx_exp_f32.h` |
| F18 | `mha_core.cpp` already implements sliding-window and attention-sink attention (`local_window_size`, `calc_windowed_attn_index`, the `sink_step` overloads) | `mha_core.cpp:1176–1379`, `:693–863` |
| F19 | The score tensor `out_` is `[1,1,rows_total,num_heads_Q]` — **triangular-packed, head-minor**; per query row *i* the block is `[L_i][num_heads_Q]` and softmax runs *down a column* with stride `num_heads_Q` | `mha_core.cpp:766–777`, `:42–71` (index arithmetic) |
| F20 | K and V caches are `[kv_position][num_cache_head][head_dim]`, fp16 on device | `mha_core.cpp:61`, `:95` |

### Two corrections to earlier conclusions — read these, they invert decisions

**C1 — doc 08 §8 / doc 14 §1's "fusion is unnecessary" inverts under the
one-RPC constraint.** Doc 14 measured the round trip to CPU softmax at ~1% of
the per-layer total — but every DSP number in doc 14 §3 was taken inside a
standalone DSP `main()` with **no FastRPC in the timed region at all** (doc14
§7 says so explicitly). Once transport is counted (F13), an unfused
three-op seam costs **three** ~404 µs round trips per layer instead of one.
Fusion is not an optimisation here; it is the requirement.

**C2 — doc 14 §6's transpose cost does not survive the jump to prefill.**
Doc 14 measured a scalar DSP-side transpose of `P` at 7.3 µs/head for
`[kv=1024][gqa=8]`. At prefill the same transpose covers
`[kv=1024][M_band·gqa]` — 64× more data at `M_band=64`, i.e. ~470 µs/head,
~3.7 ms/layer. A design that transposes per tile at prefill is dead. §3
removes the transpose instead of vectorising it.

---

## 2. The constraint that shapes everything: one FastRPC call per layer

The requirement is one call in, one call out per `mha_core` invocation. F13 makes
the arithmetic unavoidable, so here it is explicitly, for
`nch=8, gqa=8 (⇒ nHq=64), head_dim=128, kv_len=1024`:

| | payload | marshalling | + fixed | compute (F1/F2 + §7.3 est.) | transport share |
| :-- | --: | --: | --: | --: | --: |
| decode (`n_query=1`) | 36 KiB | 0.7 µs | **~405 µs** | ~420 µs | **49%** |
| prefill (`n_query=128`) | 4.6 MiB | 88 µs | **~492 µs** | ~1.25 ms | **28%** |

Consequences, in order of importance:

1. **Decode's win shrinks to ~1.4×** (CPU bar 1.17 ms/layer vs ~825 µs). 28
   layers × 825 µs ≈ 23 ms/token of attention alone. This is a real result, not
   pessimism: *one RPC per layer is not enough for decode.* The fix is
   `dspqueue` (llama.cpp's `htp_iface.idl` carries no tensor data — `rpcmem_alloc2`
   + `mmap(fd)` once, then small request structs over a queue), already
   identified as the real answer to the ~404 µs fixed cost. It is separate
   architecture work; do not let it block Stage 1–3.
2. **Prefill keeps ~30×.** That is where this work pays, and it is why Stage 2
   (prefill) is the substantial stage, not Stage 1.
3. **Softmax must be on the DSP.** Not for speed — because a CPU softmax means a
   second and third round trip. PR #4245 (F16) exists precisely in time.
4. **The KV cache cannot cross the boundary per call.** 4 MiB/layer/call of
   marshalling is not payable. The DSP must own a registered KV shadow, appended
   from the small per-step rows — the same lifecycle
   `weight_register_u8i4`/`weight_release_u8i4` already implements.
5. **Every intermediate layout is ours to choose.** `S` and `P` never cross the
   RPC boundary, so doc 08's "triangular-packed head-minor scores" (F19) stops
   being a layout obstacle. We use a plain `[M_band][kv]` rectangle plus a mask
   and waste ~3% on the causal triangle at `M_band=64`.

### 2.1 Endpoint shape

Mirrors the existing registry/layer-endpoint pattern exactly (`nntr_hvx.idl`),
so the IDL, session struct and lifecycle tests are all already-proven shapes.

```
// One per attention layer, at model load. Allocates the DSP-side KV shadows.
AEEResult attn_register(in uint32 nch, in uint32 gqa, in uint32 head_dim,
                        in uint32 max_kv, rout uint32 h);
AEEResult attn_release(in uint32 h);

// One call per mha_core invocation. Appends the new KV rows, then runs
// Q·Kᵀ → softmax → P·V entirely on the DSP.
AEEResult attn_forward(in uint32 h,
                       in uint32 kv_from,      // = cache_index
                       in uint32 n_query,      // 1 at decode, chunk at prefill
                       in float  scale,        // 1/sqrt(head_dim), and softcapping fold
                       in uint32 window,       // local_window_size, 0 = none
                       in uint32 is_causal,
                       in sequence<uint16> q_f16,      // [n_query][nHq*head_dim], post-RoPE
                       in sequence<uint16> k_step,     // [n_query][nch*head_dim], post-RoPE
                       in sequence<uint16> v_step,     // [n_query][nch*head_dim]
                       in sequence<uint16> sink,       // empty, or [nHq]
                       rout sequence<uint16> out_f16); // [n_query][nHq*head_dim]
```

RoPE stays on ARM (it is cheap and already correct). `attn_forward` takes
post-RoPE Q and K. `attn_logit_softcapping` — if a target model uses it — is a
`tanh` per score element and does **not** fold into `scale`; leave that model on
the CPU path in Stage 1–3 and say so in `supports_*()`.

### 2.2 The DSP-side KV shadow, and what it costs

The shadow is **not** a copy of the ARM cache layout. It holds what the kernel
actually consumes (§3), built incrementally per call:

| shadow | layout | why |
| :-- | :-- | :-- |
| `Kt_wh[nch]` | WH-baked `Kᵀ`, logically `[nch][head_dim][max_kv]` | weight operand of `S = Q·Kᵀ`; head-major so a per-head kv block is contiguous (F4, §5.3) |
| `V_wh[nch]` | WH-baked `V`, logically `[nch][max_kv][head_dim]` | weight operand of `P·V`; F8's incremental bake |

Both live in **DDR/host RAM**, not VTCM (doc08 §6; F6 requires the WH bake to
read a DDR source anyway). Append cost per token per head is F8's measured
3.35 µs — dirtying `ceil(head_dim/32)=4` WH tiles for `Kᵀ` (a new column) and
`ceil(head_dim/32)=4` for `V` (a new row).

Byte count: `2 × nch × head_dim × max_kv × 2 B`. At `max_kv=4096` that is
16 MiB/layer, 448 MiB over 28 layers — **the same 224 MB × 2 doc 08 §6 flagged.**
Two ways out, pick one before Stage 1:

- **(a) Shadow replaces the ARM cache** when the HTP path is enabled. Same byte
  count as today (a permutation, not a duplication), zero extra memory. Requires
  that nothing else reads `cache_key`/`cache_value` — check `use_external_cache`
  and the `sink_step` overload before committing.
- **(b) rpcmem/ION-map the shadows** so the ARM side can write the step rows
  directly and the DSP reads them with no marshalling. Note that P1's "ION
  zero-copy saves only ~25 µs, not worth building" conclusion was about a
  512 KiB–1.25 MiB *activation* payload; **it does not transfer to a
  4 MiB/layer/call KV payload.** Do not cite it here.

(a) is the smaller change and is recommended; (b) is the better endgame.

---

## 3. The formulation: transpose-free `S = Q·Kᵀ` then `P·V`

This is the core design decision and the one most likely to be got wrong.

### 3.1 Why doc 14's orientation is the wrong one now

Doc 14 computed `Sᵀ = K·Qᵀ`: activation = K (natural `[kv][d]` orientation),
weight = `Qᵀ` (a 1 KB transpose at decode). Its output is `[kv][gqa]`, which
*is* `mha_core`'s head-minor scores layout (F19) — perfect when the CPU does the
softmax next.

Fused, that orientation costs two things:

- HVX softmax must reduce **down a column** of `[kv][q]`. That is actually fine
  (see §4.4) — not the problem.
- `P·V` needs its activation as `[q][kv]`, but the accumulator produced
  `[kv][q]`. F12 says there is no `ah_to_wh`, so the only routes are a transpose
  (C2: ~3.7 ms/layer at prefill — dead) or a DDR round trip through
  `rm_to_wh_f16` from DDR (F6): ~8 MiB of DDR traffic per layer, ~160 µs, plus
  it serialises the chain.

### 3.2 The formulation to build

```
per (kv_head n, q-band b):
  A  := Q[band b, heads of n]   →  rm_to_ah_f16   (pure-format, F6-safe)
  W  := Kt_wh[n]                (DDR-resident, incrementally baked)
  S  := A · W                    →  acc is [M_b][kv] in AH
        ah_to_rm_f16             (pure-format)      → S rm [M_b][kv], VTCM
  P  := hvx_softmax(S)           rows = query rows, softmax axis = kv  ✔ dense rows, PR#4245's exact signature
        rm_to_ah_f16             (pure-format)      → P AH
  W2 := V_wh[n]                 (DDR-resident, incrementally baked)
  O  := P · W2                   →  acc is [M_b][head_dim] in AH
        ah_to_rm_f16 → f16 out
```

Properties, each one a reason to prefer it:

1. **Zero transposes.** Every layout change is either a pure-format 32×32 op
   (F6-safe from VTCM) or a DDR-sourced incremental WH bake (F6-safe, F8-cheap).
2. **`S` lands in the orientation the HVX softmax wants.** `hvx_softmax_rows_f32`
   needs dense rows of stride `k` with the softmax axis along the row (F16).
   `[M_b][kv]` is exactly that. Zero adaptation.
3. **`P` lands in the orientation the `P·V` activation slot wants** — doc 14 §6's
   ask of the softmax owner disappears, it is satisfied structurally.
4. **`O` lands as `[q][head_dim]`**, which is the output tensor's own layout.
5. **Identical HMX tile-multiply count to doc 14's form**, decode included.
   Decode, per kv_head: this form is `M=gqa=8` (GQA row-fold, doc08 §7),
   `N=kv=1024` (32 n-tiles), `K=d=128` (4 k-tiles) = 128 `mm_f16` calls at 8/32
   row utilisation. Doc 14's form: `M=kv=1024` (32 m-tiles), `N=gqa=8` (1
   n-tile), `K=128` (4 k-tiles) = 128 calls at 8/32 column utilisation. **Same
   128 calls, same waste, mirrored.** So there is no decode regression to trade
   against, and one code path covers both regimes.
6. **`K` becomes a weight instead of an activation**, which means it gets F8's
   incremental-bake treatment — the same 32× that made P·V win, now applied to
   Q·Kᵀ as well. Doc 14 paid a per-tile `rm_to_ah` repack for K instead.

The cost is one new shadow (`Kt_wh`) and the append-time column bake. That is
§2.2's accounting, already done.

### 3.3 Traps in this formulation

- **`kv_len % 32 != 0`.** The last n-tile of `S` is partial. The padding columns
  contain whatever was in the accumulator/weight tile and softmax **will
  include them**. Zero the WH tail at bake time *and* clamp the softmax range
  per row. Both, not either — belt and braces here is a few lines and the
  failure is silent.
- **`n_query·gqa % 32 != 0`** likewise leaves partial m-tiles; those produce
  garbage output rows that must not be written out.
- **`acc_clear_f16()` between the two matmuls.** F11: one accumulator, shared.
  Forgetting the clear makes `O` include `S`. The symptom is a large but
  *structured* error, easy to mistake for a scale bug.
- **Two HMX config regions, not one.** `setup_acc_read_int32` (FC path, already
  in `nntr_hvx_session`) and `setup_acc_read_f16` (this path) write different
  config data. The API takes `hmx_config_offset` per call, so allocate **two**
  regions in the VTCM arena and pass the right one. Re-calling `setup_*` per
  matmul instead would be a per-call cost for nothing.
- **`hvx_exp_sf` is undefined above +88.7 and on NaN/Inf** (F17). Softmax feeds
  `x − max ≤ 0` so the domain is safe *by construction* — but only if the max
  pass runs over exactly the same lanes the exp pass does. A masked lane that
  escapes the max pass and reaches the exp pass is an undefined-value bug, not a
  small numerical error.

---

## 4. Q1 — flash attention: online softmax + KV tiling on HMX micro + the DMA ring

### 4.1 The structural fact that decides the shape: one accumulator, and it is shared

F11. There is a single 32×32 fp16 HMX accumulator, and both `Q·Kᵀ` and `P·V` use
it. Therefore:

- **The textbook flash-attention structure — keep `O` in the accumulator across
  KV tiles, rescaling it by `exp(m_old − m_new)` — is not expressible on this
  hardware.** Not slow: not expressible. There is no accumulator-scaling
  primitive, and the accumulator is needed by `Q·Kᵀ` of the next tile anyway.
- So in an online-softmax design, `O` must live in **VTCM as f32**, and **HVX**
  performs `O *= alpha; O += O_tile`. "Accumulator reuse" in this design means
  reusing a VTCM buffer, never the HMX accumulator.
- Cost of that: the `[M_b][head_dim]` f32 rescale is paid **once per KV tile**.
  At `M_b=64, d=128` that is 32 KiB read-modify-write per (band, head, kv-tile);
  at `kv=1024, T=256, nch=8` ≈ 1 MiB/layer ≈ 64 µs at F5's 31.6 GB/s. Real but
  small, and it **shrinks as T grows** — the opposite of the GPU intuition where
  T is chosen to fit SRAM. Here T should be as large as the VTCM budget allows.

### 4.2 VTCM budget (against F10's 8 MiB)

**Full residency, no KV tiling** (the Stage 1 shape). Per kv position the two
shadows cost `nch·d·2·2 = 4 KiB`, so the hard ceiling is `8 MiB / 4 KiB = 2048`
positions with room for literally nothing else; realistically **kv_len ≈ 1024 is
the wall**. This — not speed — is the honest reason KV tiling has to exist.

**Streamed** (Stage 2), per (kv_head, q-band), `M_b=64, d=128, T=256, kv=1024`:

| slot | bytes |
| :-- | --: |
| `S` band f16 `[M_b][kv]` | 128 KiB |
| `S`/`P` f32 working copy for the HVX softmax | 256 KiB |
| `Kt_wh` tile buffer `[d][T]`, double-buffered | 2 × 64 KiB |
| `V_wh` tile buffer `[T][d]`, double-buffered | 2 × 64 KiB |
| `Q` AH `[M_b][d]` | 16 KiB |
| `P` AH staging | 128 KiB |
| 2 × HMX config regions | 512 B |
| **total** | **≈ 800 KiB** |

So 8 MiB holds ~8 such contexts. That is a lot of headroom, and it is what pays
for §7.4's software pipelining (overlap band *i*'s softmax with band *i−1*'s
`P·V`) and for a much larger `T`.

### 4.3 What the existing DMA ring gives us, unchanged

`hexkl_dma_ring.{c,h}` (in-tree, PR①) plus the cross-matmul prefetch pattern
(`run_pipelined_layer`) transfer directly. Only the *boundary* changes (F9):

| FC | attention |
| :-- | :-- |
| prefetch matmul *i+1*'s weight while *i* computes | prefetch KV tile *j+1* (and kv_head *h+1*) while *j* computes |
| weight tiles from a DDR WH registry | `Kt_wh` / `V_wh` tiles from the DDR shadow registry |
| one descriptor per weight tile | one descriptor per **(head, kv-tile)** slab |

Two rules from prior sessions carry over verbatim and are the two most likely
performance bugs:

- **`row_size > ~512 KiB` in a single-row descriptor silently corrupts on this
  hardware.** Use nrows-based descriptors with `row_size ≤ 256 KiB`.
- **`ring_init`'s `memset` of the 256-entry ring, if called per row-band, was
  35 µs of an "82 µs" result.** `ring_push2d` writes every field; the memset is
  dead weight. Initialise the ring once per call.

With the head-major shadows (§2.2), a `(head, kv-tile)` slab is contiguous:
`T·d·2 = 64 KiB` at `T=256` → F4's 55 GB/s regime. Good. At `T=64` it is 16 KiB
→ ~42 GB/s, still fine. **Do not go below `T=32`** (8 KiB rows) and never slice
per-head out of a `[kv][nch][d]` layout (256 B rows → ~8–15 GB/s, F4).

### 4.4 Where online softmax *would* map, if we needed it

For completeness, because Stage 2b may need it and because getting the lane
assignment right is the whole trick:

- In `[M_b][kv]` (§3.2's orientation), the softmax axis is along the row, so
  `m` and `l` are **scalars per row** and the reductions are *in-vector*
  (`Q6_V_vror_VR` chains — exactly what PR #4245's `reduce_max_sf`/`reduce_sum_sf`
  already do).
- In doc 14's `[kv][q]` orientation they would be **vectors indexed by q**, and
  the reduction is a loop of `vmax`/`vadd` down the tile with *no* in-vector
  reduction at all — cheaper per element. That is the one argument in doc 14's
  orientation's favour, and it is worth less than the transposes it costs (§3.1).
- Either way the loop body is: `m_new = max(m, rowmax(S_tile·scale))`;
  `alpha = exp(m − m_new)`; `P_tile = exp(S_tile·scale − m_new)`;
  `l = l·alpha + rowsum(P_tile)`; `O = O·alpha + P_tile·V_tile`. The `exp(m−m_new)`
  and the `O` rescale are the added work versus the two-pass form.

### 4.5 …and why we should not build it yet

The two-pass exact softmax needs the whole `[M_b][kv_len]` band resident. It
is `M_b · kv_len · 2 B` (f16) or `4 B` (f32 working copy):

| `kv_len` | band f16 @ `M_b=64` | f32 working copy |
| --: | --: | --: |
| 1 024 | 128 KiB | 256 KiB |
| 4 096 | 512 KiB | 1 MiB |
| 16 384 | 2 MiB | 4 MiB |
| 32 768 | 4 MiB | 8 MiB ✗ |

**Crossover: online softmax becomes necessary at roughly `kv_len ≥ 16 K`** (or
proportionally sooner at larger `M_b`). Below that, the two-pass form is:

- exact (no `alpha` chain, no accumulated rounding across tiles),
- a verbatim reuse of PR #4245's already-device-verified kernel,
- and — the real prize — it lets `P·V` accumulate over the **entire** `kv_len` in
  **one** HMX accumulator lifetime per `(m-tile, n-tile)`, because `P`'s whole row
  is available before `P·V` starts. That removes §4.1's per-tile `O` rescale
  *entirely*, and F3 already certifies fp16 accumulation at K=1024.

So Stage 2's loop is: stream `Kt_wh` tiles → build the full `S` band in VTCM →
one HVX softmax pass over the band → stream `V_wh` tiles → `P·V` accumulating
across all of them. K and V are each streamed once, nothing is resident for the
whole `kv_len`, and there is no rescale tax. **This is what "flash attention"
buys on this hardware: fusion and streaming, not online softmax.** Say that in
the PR body; a reviewer who expects the textbook structure will otherwise think
something is missing.

---

## 5. Q2 — sparse attention: what level to skip at, and what it does to the DMA

### 5.1 The level: `(kv_head, kv_block)`, block ≥ 64 positions

Because that is the granularity the loop and the DMA descriptors already have
(§4.3). Skipping means: do not enqueue that slab's descriptor, and do not run
its `mm_f16` calls. Nothing else changes.

**Head-level skipping is a *policy*, not a kernel decision.** DuoAttention-style
"streaming heads" (some heads get only sinks + a local window) is a legitimate
and effective form of head-level sparsity, but it must arrive as declared
per-head configuration, not as something the kernel infers. In the kernel it
degenerates into "this head's allowed block set is {sinks} ∪ {last window}",
i.e. the same block-set mechanism. So: **implement one mechanism — a per-head
allowed-block set — and let both window/sink semantics and any head policy
express themselves through it.**

### 5.2 Rung 1 is free and exact: the sparsity `mha_core` already declares

F18: `local_window_size` and `sink_step` are already in the layer, already used
by shipping configs, and today are implemented on the CPU by the packed
triangular scores layout (F19). On the DSP they become a **tile-range
restriction**:

```
row i (absolute position p = kv_from + i):
  allowed = [max(0, p − window + 1), p]   ∪  [0, n_sink)
  blocks  = the kv blocks intersecting `allowed`
```

Whole blocks outside the union are skipped: no DMA, no matmul, no exp. For a
4096-position context with a 1024 window this is a **4× reduction in every
stage**, at exactly zero accuracy cost, and it is a strictly smaller diff than
what `mha_core` does today. It also directly cuts §7.3's softmax cost, which is
the thing most likely to dominate.

**Do this before any dynamic sparsity.** It is the whole ladder's first rung and
it needs no accuracy story.

### 5.3 Rung 2, opt-in: dynamic block top-k

If a workload needs it. Design constraints, all of them forced by the DMA:

1. **The skip list must exist before the ring is primed.** The ring prefetches
   tile `j+1` while `j` computes; a selection discovered mid-loop cannot be
   prefetched. So dynamic sparsity is necessarily **two-phase**: (A) score all
   blocks with a cheap proxy, (B) run the streaming loop over the selected set.
2. **The proxy must be precomputable at KV-append time**, exactly like the WH
   bake (F8). Per-block mean-`K` (`[nch][n_blocks][d]`, `1/T` of the K bytes) is
   the standard choice; `q · mean_k_block` gives a per-block score for
   `n_blocks` values instead of `kv_len`. Appending a token updates one block's
   mean — a running sum, a few hundred bytes.
3. **Selection shared across the GQA group is free; per-q-head selection is
   not.** All `gqa` query heads of one kv_head read the same K/V rows, so
   letting them share one block set costs nothing structurally. Per-q-head sets
   would multiply the descriptor count by `gqa` and, worse, fragment each slab.
4. **Selection shared across kv_heads is *not* required** thanks to the
   head-major shadow layout (§2.2) — a `(head, block)` slab is contiguous there.
   This is the payoff of that layout choice, and it is the reason to make it now
   even though Stage 1 does not need it. In the CPU's `[kv][nch][d]` layout a
   per-head block would be 256 B rows → F4's ~8–15 GB/s regime, i.e. 3–4× worse
   DMA, and sparsity would have to be shared across all heads to stay fast.
5. **Block size trades DMA efficiency against selection precision.** `T=64`
   → 16 KiB rows (~42 GB/s), `T=256` → 64 KiB (~55 GB/s). Below `T=32` the
   descriptor geometry collapses. Recommend `T=128` or `256`, and make it a
   parameter with the measured bandwidth table (F4) in a comment.

Verification for rung 2 is *not* a tolerance check — that is §9.6.

### 5.4 What sparsity does to the VTCM plan

Nothing structural: the block set only shortens the loop. Two second-order
effects worth writing down:

- The `S` band becomes `[M_b][n_selected·T]` — smaller, so §4.5's crossover
  moves *out*, not in. Sparsity makes the two-pass form viable at longer
  contexts, not shorter.
- With a per-row allowed range (rung 1), different rows in a band select
  different blocks. Either take the band's union (simple, wastes work on rows
  that did not need a block, then masks it) or choose `M_b` so a band's rows
  share a range (`M_b ≤ T` makes the union at most one extra block). **Choose
  `M_b ≤ T`** and the problem disappears.

---

## 6. Q3 — does the macro/micro one-way door make FC migration a prerequisite?

**No. And the reason is better than "we can work around it": there is no
SDKL-macro FC left on this lineage to migrate.**

Verified by reading the tree, not inferred:

- `git grep sdkl_npu_mm HEAD -- nntrainer/` finds **nothing**. `sdkl_npu_mm`
  exists only on the abandoned `claude/…-6ycsx0-v2` branch's
  `hmx_ops/hexkl_mm.cpp`, which `HTP_PR_PLAN.md` §0 already decided never to
  upstream.
- The FC path on this branch **is** the micro API: `mm_u8i4_layer` /
  `mm_u8i8_layer` → `hexkl_mm_u8i4_dma.c` / `hexkl_mm_u8i8_dma.c` →
  `hexkl_micro_hmx_mm_u8i4/u8i8`. The migration doc 15 called the "next
  architecture step" is, in effect, already shipped as PR①/PR②.
- `htp_compute_ops.cpp` currently returns `get_cpu_ops()` — no macro dispatch
  behind it.

**The one hazard that does remain**, and it is a real one:
`HtpBackend::HtpBackend()` (`nntrainer/tensor/htp_backend/htp_backend.cpp:39`)
calls `sdkl_npu_initialize(domain_, nullptr, nullptr)`, then
`sdkl_npu_get_version`, and `~HtpBackend` calls `sdkl_npu_finalize`. Per F14, a
macro-API session opened **after** the skel's `hexkl_micro_hw_init()` has
succeeded fails permanently. Today the order happens to be safe (backend
construction at load, `nntr_hvx_open` later), but nothing enforces it and a
lazily-constructed singleton makes it an ordering accident.

Recommended resolution, in ladder order:

1. **Delete the three `sdkl_npu_*` calls.** They initialise a session nobody
   uses and print a version string. That removes the hazard permanently, drops
   `sdkl_compat.h`/`libsdkl.so` from the link, and is ~20 lines of deletion.
   `npuAlive()`/`enabled_` can key off the FastRPC session opening instead.
2. If they must stay (someone reads that version), add an assertion that no
   `nntr_hvx_open` has happened yet, and a comment citing F14.

Either way: **attention work proceeds in parallel, in the same skel and the same
session, sharing one `hexkl_micro_hw_init`.** No prerequisite, no separate DSP
process, no cross-process RPC per layer boundary.

**The upside of this being already true**: because FC and attention are both
micro-API in one skel and one session, a future `block_forward` endpoint — QKV
projection, attention, and the output/FFN projections in **one** FastRPC call —
is architecturally available. That is the answer to §2's decode problem, and it
is only available because the FC path is already micro. Worth stating in the PR
body; it is the strategic payoff of PR①/②.

---

## 7. Softmax: what PR #4245 gives us, and the one thing it does not

### 7.1 Use it, do not rewrite it

F16. `hvx_softmax_rows_f32(x, y, m_first, m_last, k, scale)`:

- 3-pass exact (max → exp+sum → normalize), which is precisely the form §4.5
  argues for.
- `[m_first, m_last)` row range → drops straight onto the F15 worker pool.
- In-place safe (`y == x`), tail-handled, `HVX_UVector` throughout (correct for
  FastRPC and heap buffers alike).
- Correct for negative `scale` (it scales *before* the max, deliberately).
- Dense rows of stride `k`, softmax along the row — which §3.2's `S` orientation
  provides for free.

**PR #4245 also contains PR①'s u8i4 commits and adapts the softmax entry points
to PR①'s session handle** (`be0ffed74`, `dfffa72ab`). So the merge base for this
work is `pr4245`, and our u8i8 commits rebase on top. Sort that out before
writing code, not after: `git log --oneline HEAD..pr4245` shows the delta.

### 7.2 The one extension needed: a masked / ranged variant

`mha_core`'s semantics need three things the current kernel does not have:

1. **A per-row valid range.** Causal + sliding window means row `i` softmaxes
   over `[begin_i, end_i)`, not the whole row. The current signature uses one
   `k` for both stride and length, so this needs either a `stride` parameter or
   a `(begin, len)` pair per row.
2. **A sink term.** `softmax_row(…, sink)` includes an extra logit in the
   denominator only. One extra `exp` and one add per row.
3. **Masked lanes excluded from the max pass, not just the sum.** F17: masked
   lanes that reach `hvx_exp_sf` with a large positive value are *undefined*,
   not merely wrong.

This is a ~40-line addition to an upstream contributor's file. **Contribute it
to PR #4245 rather than forking it** — the ranged form is independently useful
(it is what a windowed CPU softmax wants too), and a fork of a file that is
mid-review is a merge conflict with someone else's name on it. Coordinate first;
if that stalls, put the variant in a *new* file that includes theirs.

### 7.3 The number that should worry us: softmax may dominate at prefill

Doc 14's "softmax is ~1% of the total" was measured at **decode**, where `S` is
`kv·nHq = 64 K` elements per layer. At prefill it is `n_query · nHq · kv_len` =
`128 · 64 · 1024` = **8.4 M elements per layer** — 128× more.

Estimate (arithmetic shown so it can be checked and then replaced by a
measurement): 8.4 M f32 / 32 lanes = 262 K vector-`exp`s; `hvx_exp_sf` is a
7th-order qf32 Horner plus range reduction plus exponent insert ≈ 25–30 vector
ops; plus the max and normalize passes ≈ 3 more. ≈ 8 M cycles ≈ **5–8 ms on one
HVX context, ~0.8–1.3 ms across F15's 6 contexts** — against F1's **924 µs** for
both matmuls combined.

**Treat this as the single largest open risk in the plan.** It is an estimate,
and estimates in this project have been wrong in both directions (P0's first
hypothesis was wrong; the worker pool beat its own estimate). So: **Stage 0
measures it before Stage 2 designs around it.** Four mitigations, in the order
they should be tried:

| # | mitigation | expected | cost |
| --: | :-- | :-- | :-- |
| 1 | per-row valid range (§7.2) — causal prefill masks ~half the rectangle | ~2× | needed for correctness anyway |
| 2 | run it on the F15 worker pool | up to 6× | pool exists; interface designed for reuse |
| 3 | fp16 softmax (64 lanes/vector) | ~2× | needs an `hvx_exp_hf`; accuracy is fine — output is normalized and feeds an fp16 matmul |
| 4 | sparsity (§5.2) | proportional | Stage 3 |

1 + 2 alone bring the estimate to ~110–220 µs/layer, i.e. below the matmuls.

### 7.4 Overlap HVX and HMX

HVX and HMX are separate units; the HMX owner thread holds the compute-resource
lock while pool workers run HVX (this is exactly llama.cpp's split of a
dedicated `hmx_queue` from an N-worker `work_queue`). So the band loop should be
software-pipelined: **HVX softmaxes band `i` while HMX runs `P·V` for band
`i−1`.** §4.2's budget already has room for the double-buffered `S`/`P` bands
this requires. Do not build this in Stage 1; design the buffers so Stage 2 can.

---

## 8. Staging and gates

Each stage ends with a device run on `R3CY10WM83Y` and does not start the next
until its gate is green. Branch per stage, created **before** the first commit
(the mistake caught mid-session in PR②: do not commit-then-branch).

### Stage 0 — measure the three unknowns (no kernel code)

Small probes, ~1 day, and they can change Stage 1's design:

| probe | question | why it must come first |
| :-- | :-- | :-- |
| P0.1 | HVX softmax throughput for `[64][1024]` f32 on device, single-thread and on the F15 pool | §7.3 may make softmax the dominant cost; if the estimate holds, mitigations 1+2 move into Stage 1 instead of Stage 2 |
| P0.2 | `attn_forward`-shaped FastRPC round trip: fixed cost + per-KB at 36 KiB and 4.6 MiB payloads | §2's whole argument rests on F13's ~404 µs, measured for a different payload shape |
| P0.3 | `Kᵀ` WH incremental column bake cost per head per token | §2.2/§3.2 assume F8's 3.35 µs (measured for a V *row*) transfers to a `Kᵀ` *column*. It should; verify, it is cheap to check |

Do **not** run doc 08's layout probe (5) (`is the accumulator's AH the same AH
that rm_to_ah produces`). It is moot for this design: HVX cannot read AH, so the
chain goes through RM regardless, and both directions are pure-format/fast (F6).
That probe only mattered for a design where softmax could operate in AH.

### Stage 1 — the seam: fused single-RPC attention, decode shape

Scope: `attn_register`/`attn_release`/`attn_forward`; KV shadows with incremental
WH bakes; the §3.2 chain; PR #4245's softmax unmodified; `n_query = 1`;
`is_causal`, no window, no sink; `kv_len ≤ 1024` fully resident (no KV tiling).

Deliberately excluded: KV tiling, q-banding, masking beyond causal, sparsity,
worker pool, pipelining, `attn_logit_softcapping`.

**Gate:** every §9.3 shape matches the CPU reference within §9.2's tolerance,
the weight/session lifecycle tests pass, and the end-to-end per-layer µs
(including FastRPC) is *printed* alongside the CPU bar. Do not assert a speed
threshold — thermal state makes it flaky, and §2 already predicts decode only
reaches ~1.4×.

### Stage 2 — prefill: q-banding, KV streaming, masking. **The substantial stage.**

Scope: q-bands (`M_b`, with `M_b ≤ T` per §5.4); KV tiling with the
double-buffered ring (§4.3); the VTCM plan of §4.2 as an explicit planner
function, not scattered offsets; the masked/ranged softmax of §7.2 (causal +
window + sink); the worker pool for softmax; `n_query` up to 256.

Online softmax (§4.4) is **out of scope** unless P0.1 plus the §4.5 table say
the target `kv_len` needs it. If it does, it is Stage 2b with its own gate.

**Gate:** §9.3's prefill shapes within tolerance; **bit-exact agreement with
Stage 1 for the shapes both can run** (`n_query=1`, no window) — same operation
order, so this is achievable and it is the strongest available check that the
tiling did not change the math; per-stage µs breakdown printed
(`dma / qk / softmax / pv / out`), because P0's lesson is that the breakdown is
what finds the real bottleneck and guessing twice wastes a session.

### Stage 3 — sparsity

3a: window + sink as block-range skipping (§5.2). **Gate: bit-exact against
Stage 2** with the same window parameters — Stage 2 computes and masks, Stage 3a
skips; the surviving arithmetic is identical, so any difference is a bug.

3b (opt-in, flagged off by default): dynamic block top-k (§5.3), per-block
mean-`K` maintained at append time. **Gate: §9.6, which is not a tolerance
check.**

### Stage 4 — direction, not scope

`block_forward`: QKV projection + attention + output projection in one FastRPC
call (available because §6 — FC is already micro-API in the same session). This
is the answer to §2's decode problem. Alternative/complement: `dspqueue`.
Neither belongs in Stages 1–3.

---

## 9. Verification — written for an implementer whose verification is weak

The implementation is going to GLM 5.2 via opencode. The failure mode to design
against is not "writes bad code"; it is **"declares success on evidence that
does not support it"** — loosening a tolerance, editing the reference, timing a
region that includes the correctness check, reporting one lucky run. Every item
below exists to make that hard.

### 9.1 Rule zero: three things the implementer may not touch

Put this verbatim in the task prompt.

1. **Do not modify any reference implementation.** `compute_kcaches_fp32_reference`
   (`mha_core.cpp:42`), `compute_vcache_fp32_transposed_reference` (`:73`),
   `nntrainer::softmax_row` / `softmax_row_inplace`, and the existing scalar
   reference in `test/unittest/unittest_hvx_mm_u8i4.cpp` are the ground truth.
   If a test fails, the kernel is wrong.
2. **Do not change a tolerance.** §9.2's table is fixed here, before any code
   exists, precisely so it cannot be adjusted to fit a result. A number outside
   it is a finding to report, not a threshold to raise.
3. **Do not delete or skip a shape** from §9.3's matrix. A dropped shape has bit
   this project twice already (a report script that silently dropped every shape
   past its first hardcoded name, and a bench whose SDKL correctness column was
   never actually read).

### 9.2 Tolerance table — fixed now

Against the **fp32-accumulating** CPU reference, max relative error:

| quantity | pass | investigate | fail |
| :-- | --: | :-- | :-- |
| `S` (post-`Q·Kᵀ`, pre-softmax) | ≤ 1e-3 | — | > 1e-3 |
| `P` (post-softmax) | ≤ 1e-3 | — | > 1e-3 |
| `O` (post-`P·V`), `kv ≤ 1024` | ≤ 1e-3 | 1e-3 … 5e-3 | > 5e-3 |
| row sum of `P` (should be exactly 1) | ≤ 1e-5 | — | > 1e-5 |
| Stage 2 vs Stage 1, overlapping shapes | **bit-exact** | — | any difference |
| Stage 3a vs Stage 2, same window | **bit-exact** | — | any difference |

Calibration from F3, so these are not arbitrary: fp16 HMX accumulation at
K=1024 measures **3e-4**; a reference that rounds to fp16 after every term
measures **9e-3**. Therefore:

- **~3e-4 is the expected, healthy number.**
- **~9e-3 means the accumulation is losing precision it should not** — a chunking
  or read-back bug, not "fp16 is just like that".
- **≥1e-1, or a structured/blocky error pattern, means a layout or transpose
  bug**, which is exactly what §3.3 and doc 14 §4 warn is the class that hides
  silently because the shapes still line up.

Test data must be **softmax-like** (non-negative, rows summing to ~1) for `P·V`,
not a synthetic ramp — doc 14 §4 measured against realistic data deliberately,
and a ramp both flatters and misleads.

### 9.3 Shape matrix — every one runs, every one prints

The non-multiples are where a tiling bug hides. `kv_len` values chosen to
straddle the 32 (HMX tile) and `T` (KV block) boundaries.

| axis | values |
| :-- | :-- |
| `kv_len` | 1, 31, 32, 33, 64, 255, 256, 257, 1023, 1024 |
| `n_query` | 1, 7, 32, 33, 128 (Stage 2+) |
| `gqa` | 1, 4, 8 |
| `nch` | 1, 8 |
| `head_dim` | 64, 128 |
| `is_causal` | true, false |
| `window` (Stage 2+) | 0 (off), 1, T−1, T, T+1, kv_len−1, kv_len |
| sink | absent, present |

Full cross product is too many; require **every value of every axis to appear at
least twice, and every `(kv_len % 32 != 0) × (n_query % 32 != 0)` combination to
appear at least once**. The `window ∈ {T−1, T, T+1}` triple is the off-by-one
trap in the block-skipping arithmetic and is mandatory.

### 9.4 The highest-value artifact: a host-side model of the DSP kernel

**Build this first, before the DSP kernel.** A plain-C++ function in the normal
`test/unittest` build that implements the *same* tiling, the *same* band/block
loop order, the *same* mask arithmetic and the *same* two-pass softmax — with
scalar arithmetic instead of HMX/HVX/DMA — and a gtest comparing it to the CPU
reference across §9.3's whole matrix.

Why this is worth more than any device test to this implementer:

- It separates "is the algorithm right" from "is the hardware plumbing right".
  Those two failures look identical in a device run and are debugged completely
  differently.
- It runs on the host in seconds, in CI, with no phone, no NDK, no skel build,
  no `ADSP_LIBRARY_PATH`. The implementer can iterate on the loop structure at
  full speed and cannot mistake a build/environment problem for a logic problem.
- It is where the off-by-one shape bugs (§9.3) get caught — cheaply, and with a
  debugger.
- The DSP kernel then has a *second* oracle: it must match the host model, not
  just the CPU reference. A disagreement localises immediately to plumbing.

Do not let it become a second implementation that drifts. It is a *specification*
of the loop structure; if it disagrees with the kernel, one of them is a bug and
the review decides which.

### 9.5 The debug endpoint — and stripping it before the PR

Add `attn_forward_debug` returning `S`, `P`, and `O` as separate `rout`
sequences. This is the same pattern `mm_u8i4_from_f32` already uses as the
accuracy harness (it returns `act_u8_ah`, `act_scale`, `act_zp`, `acc_i32`,
`out_f32` for exactly this reason) — follow it rather than inventing a new one.

Without per-stage outputs, a wrong `O` gives no signal at all about which of the
five stages produced it. With them, §9.2's per-quantity tolerances localise the
bug in one run.

Two self-checks belong inside the debug endpoint (they cost nothing and catch the
bugs a tolerance check misses):

- **`|Σ_kv P[q][kv] − 1| ≤ 1e-5` for every `(q, head)`.** This single assertion
  catches nearly every masking, sink, range and (later) online-rescale bug.
  A masked lane that leaked in, a window off by one, a sink added to the
  numerator instead of the denominator — all show up here and only here.
- **Every element of `S` that should be masked is exactly the mask sentinel**,
  checked on the DSP side, not inferred from `O`.

**Strip the debug endpoint and any per-stage timing IDL parameters before the
PR**, and reconstruct the clean branch the way PR③ did: branch fresh from the
last clean tip and re-apply each optimisation by hand against it, rather than
trying to untangle instrumentation from optimisation inside shared commits.
That was measured to be the faster and safer route.

### 9.6 Verifying sparsity — a tolerance check is the wrong instrument

Stage 3a is exact, so it gets §9.2's bit-exact gate against Stage 2. Stage 3b
(dynamic top-k) changes the *result* on purpose, so "within 1e-3 of full
attention" is neither achievable nor meaningful. Require instead:

1. **Bit-exact against Stage 2 with the selector forced to select everything.**
   This separates "the skip machinery is correct" from "the selection policy is
   good", and it is a hard, unfakeable check.
2. **A quality metric on a real task**, not on tensors: perplexity on a held-out
   sample, or exact-match on a fixed prompt set, at several `k` values, against
   the dense baseline. Report the curve, not one number.
3. **Report the actual skip rate** and the resulting per-stage µs. A selector
   that silently selects everything will pass (1) and (2) perfectly while
   delivering no speedup.
4. **Never enable it by default.** Flagged off; the flag's default is part of the
   review.

### 9.7 Performance measurement — the rules that were paid for

Every one of these is a bug this project has already shipped and found:

- **Correctness checks must be outside the timed region.** A `diff_index_i32`
  inside the timed window on the last iteration produced a phantom "146 µs
  slowdown" that cost real debugging time.
- **≥3 consecutive runs before committing a number**, and print all three.
  Thermal state moves these numbers.
- **`field=… value=…` marker lines, keyed by name, with units in the field
  name** — `us_per_layer`, not `time`. A report script keyed by column position
  silently printed a stale number for every shape past the first.
- **Print the per-stage breakdown from day one** (`dma / qk / softmax / pv / out`),
  behind a build flag. The P0 session's first hypothesis (FastRPC marshalling
  dominates) was wrong at 16–32%; the breakdown found the real culprit at 92%.
  Guessing a second time would likely have been wrong too.
- **Never assert a speed threshold in a test.** Print it; the reviewer compares
  against F1/F2 and §2's predictions.
- **Pre-registered expectations** (so a regression cannot be rationalised after
  the fact): Stage 2 prefill should land near F1's `553 + 371 = 924 µs` of matmul
  plus P0.1's measured softmax plus §2's ~492 µs transport. If the fused total
  exceeds the sum of its measured parts, something regressed — find it, do not
  explain it.

### 9.8 Environment and hardware traps to hand over verbatim

These are not hypothetical; each cost a session or a device crash. Give the
implementer this list, not a summary of it.

- **`HVX_UVector*`, never `HVX_Vector*`, for any heap- or FastRPC-allocated
  buffer.** `malloc` gives no 128-byte alignment; an aligned vector store on an
  unaligned base crashed the entire unsigned-PD DSP process. This applies to
  DSP-heap `malloc` too, not just FastRPC parameters — that is exactly how it was
  missed last time.
- **Extraction-style micro functions must read DDR, never VTCM** (F6). ~26 µs
  vs ~1–3 µs per tile; one case 3376 µs vs 8.3 µs.
- **Single-row DMA descriptors with `row_size > ~512 KiB` silently corrupt.**
  Use nrows-based descriptors, `row_size ≤ 256 KiB`.
- **`ring_init`'s `memset` is dead weight** (`ring_push2d` writes every field) —
  35 µs of an 82 µs result when called per row-band.
- **`qurt_futex_wake(addr, n)` wakes `n` *arbitrary* sleepers**, not `n` chosen
  ones. Wake **all** pool workers every job and let non-participants
  (`id >= n_threads`) self-select out. The first version of this hung the very
  first call into the pool, indefinitely.
- **Check whether a parallelisation axis fights a vectorisation axis** before
  splitting work across pool threads. `hvx_quant_pack_u8_ah` had to split by
  k-tile, not row, or a worker's row range straddled the vectorised 4-row store
  groups.
- **Use the beta2 HexKL addon**: `HEXKL_ROOT=~/workspace/hxkl-beta2/hexkl_addon`,
  `HEXKL_SDK_VER=6.4.0.2`. The one under `Hexagon_SDK/6.4.0.2/addons/` is beta1
  (2-arg `hexkl_micro_hw_init`); mixing them fails with "too many arguments to
  function call" at DSP compile, which looks like a code bug and is not.
- **This shell's profile pins `LD_LIBRARY_PATH`** to an unrelated build's
  `libnntrainer.so` (a fresh test binary silently links the stale lib and
  segfaults like heap corruption) **and exports
  `NNTRAINER_ROOT=/home/leeseunghui/nntrainer`**, which silently beats
  `Android.mk`'s `ifndef` default. Override both explicitly, every invocation.
- **`test/htp/build.sh`'s `setup_sdk_env.source` fails here** ("missed
  components"); export `HEXAGON_SDK_ROOT` and `DEFAULT_HEXAGON_TOOLS_ROOT`
  (`.../tools/HEXAGON_Tools/19.0.04`) by hand instead of fixing it.
- **`qaic` here has no plain `out` for a scalar** — a single scalar out-param must
  be `rout uint32 x`. Error is `unexpected "o" / expecting "in", "rout" or "inrout"`.
- **When a DSP call hangs or errors with no application-level detail, run
  `adb logcat -d | grep adsprpc` first.** The kernel driver reports PD crashes
  with the crashed function name and a call trace, independently of FARF (which
  needs a `.farf` file next to the binary to reach logcat at all).
- Reuse `test/htp/run_u8i4_layer_on_device.sh` as the runner; it already handles
  the `subprojects/googletest` submodule, the `test/jni/googletest` symlink, and
  the `iniparser` wrap-git placeholder. Do not build a second harness.

### 9.9 Review checkpoints — where a human looks

1. **After Stage 0, before Stage 1 code**: the three measured numbers, and
   whether they change the plan.
2. **After §9.4's host model, before DSP code**: the loop structure and the mask
   arithmetic, reviewed as a specification.
3. **At each stage gate**: the shape matrix output in full (not a summary), three
   timed runs, the per-stage breakdown.
4. **Before each PR**: that the debug endpoint and timing parameters are gone,
   and that the branch was built fresh rather than untangled.

---

## 10. What would change this plan

Honest list of the load-bearing assumptions, so the implementer escalates instead
of improvising:

| assumption | if it is false |
| :-- | :-- |
| P0.1's softmax cost is within ~2× of §7.3's estimate | if it is 10× worse, softmax needs its own design pass (fp16 exp, or a LUT) before Stage 2 |
| F13's ~404 µs fixed FastRPC cost holds at attention payload sizes | if much worse, Stage 1 is not worth landing without `dspqueue`, and Stage 4 moves ahead of Stage 3 |
| `Kᵀ` column bake costs ~F8's 3.35 µs/head/token | if much worse, reconsider doc 14's `Sᵀ = K·Qᵀ` for decode only, and pay §3.1's DDR round trip |
| target `kv_len` stays ≤ ~8 K | past ~16 K, Stage 2b (online softmax) becomes mandatory, per §4.5 |
| the ARM KV cache can be replaced rather than duplicated (§2.2a) | otherwise +448 MiB at `max_kv=4096`, and (b) rpcmem-mapping moves into Stage 1 |
| PR #4245 is the merge base and its author accepts the ranged-softmax extension | otherwise the variant goes in a new file that includes theirs; do not fork |

One process note, because it is the pattern that kept working across every prior
session in this thread and the one that kept failing: **measure the breakdown
before acting on a hypothesis.** The first guess was wrong in the FastRPC
investigation, and a second guess would probably have been wrong too; the
per-stage instrumentation is what found the real 92% cost. Build the breakdown
into Stage 1, not into Stage 2 when something looks slow.

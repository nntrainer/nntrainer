# 14 — MHA-core fp16 micro+DMA: verified results, and what doc 08/13 got wrong

State as of branch `claude/hexkl-mha-hmx-optimization-6ycsx0`, measured
2026-08-04 on a Galaxy S25 Ultra (V79). This document supersedes doc 08 §2/§5/§6/§8
and doc 13 §5's ordering and prediction for the specific claims listed in §5
below — read this first if you are deciding whether to build the `mha_core`
ComputeOps seam; it is not analysis, it is a measured result.

---

## 1. One-paragraph summary

Both `mha_core.cpp` decode matmuls (`compute_kcaches` / Q·Kᵀ and
`compute_fp16vcache_transposed` / P·V) and their prefill counterparts were
rebuilt on the fp16 **micro API** (`hexkl_micro_hmx_mm_f16` + a hand-written
Hexagon user-DMA stager, the same infrastructure doc 13 verified for u8i4 FC)
and measured end to end against the current CPU (NEON) implementation, on
device, with correctness checked against the fp32-accumulating reference at
every step. **The DSP wins in every measured shape** — 2.4-3.4× at decode,
43-65× at prefill — and the win is a **data-movement redesign, not an HMX
compute optimisation**: the HMX multiply is 1.7% of a decode call and ~24% of
a prefill call; the rest is DMA descriptor geometry and avoiding two
specific micro-API functions that are ~10-400× slower than expected when
misused. Softmax stays on CPU throughout and the round trip to it costs ~1%
of the total — the fused/VTCM-resident-softmax path doc 08 §8 called a
*gating* dependency is not gating.

---

## 2. What was built and measured

Four new files, none touching `mha_core.cpp`, `hexkl_micro_fc_bench.c`, or
anything softmax-related:

| file | role |
| :-- | :-- |
| `test/unittest/jni_htp/hexagon/hexkl_micro_mha_bench.c` | fp16 micro+DMA kernel for both matmuls, all 5 shapes, dtype-agnostic-style reusable tile loop |
| `test/unittest/jni_htp/hexagon/run_micro_mha_bench.sh` | one-shot build/run/report; every shape runs on a plain invocation, no env overrides needed |
| `test/unittest/jni_htp/hexagon/mha_cpu_bench.cpp` | the ARM-side comparison bar: calls `nntrainer::compute_kcaches` / `compute_fp16vcache_transposed` directly, cross-compiled against the on-device prebuilt `libnntrainer.so` |
| `test/unittest/jni_htp/hexagon/run_mha_cpu_bench.sh` | builds and runs the CPU bench with the NDK, no meson rebuild required |

Diagnostic-only sweeps (DMA bandwidth ceiling, output narrow-vs-full) are
gated behind `MHA_RUN_DIAGNOSTICS=1` and skipped by default; the runner's
footer states plainly whether they ran.

---

## 3. Verified numbers

All DSP numbers are on-DSP via `HAP_perf_get_time_us`, **excluding per-op
FastRPC** (§7 explains why that exclusion is not yet safe to remove). All CPU
numbers are on-device via `mha_cpu_bench.cpp`, pooled-thread methodology (a
thread pool spawned once, reused across timed iterations — a naive
per-iteration `std::thread` spawn/join measures spawn overhead, not the
kernel). Per layer = ×8 kv_heads (this model's `num_cache_head`).

| stage | shape | DSP (µs/layer) | CPU (µs/layer or ms/layer) | speedup |
| :-- | :-- | --: | --: | --: |
| decode Q·Kᵀ | kv=1024 | 239.1 | 553-590 µs | **2.4×** |
| decode P·V | kv=1024 | 180.4-181.3 | 619.7 µs (sequential*) | **3.4×** |
| decode P·V | kv=512 | 95.6-97.0 | 243.5 µs (sequential*) | **2.5×** |
| prefill Q·Kᵀ (128-tok chunk) | kv=1024 | 553.0-558.5 (steady + incremental bake) | 23.9-36.1 ms | **43-65×** |
| prefill P·V (128-tok chunk) | kv=1024 | 370.6-380.4 (steady + incremental bake) | 16.3-20.9 ms | **43-55×** |

\* Decode P·V's CPU bar uses **sequential**, not pooled-8-thread: 8-way
parallelism measured *no better, sometimes worse* than sequential at this
shape (556-594 µs vs 553-570 µs single-thread) — decode attention is
DDR-bandwidth-bound, so more cores just contend for the same channel. Do not
assume this scales with core count.

**Prefill's CPU parallel scaling is measured but not fully explained.**
8-thread pooled scaling over single-thread was only 1.6-3.0× (not the ~8× a
compute-bound kernel should show), and one measurement was *worse* after
removing spawn overhead than before — the leading hypothesis is
memory-bandwidth contention or thermal throttling under sustained multi-core
load, not a measurement bug, but this was not pinned down with perf counters.
Treat the prefill CPU bar as a measured fact, not a well-understood one.

**Prefill's DSP total is steady-state-only plus a measured, but likely
still-underestimated, incremental bake.** A 128-token chunk's KV entries are
all new, so — unlike decode, which can assume an already-populated,
pre-baked cache from prior turns — the per-tile weight-bake cost for the
newly-appended tiles is mandatory, not amortisable, and is included above.
The incremental-bake measurement itself excludes a small VTCM→DDR
sync-extraction step that the full-bake number does include (see the bench's
own comments), so the true prefill total may run somewhat higher than shown;
it was not re-measured with that step included this session.

---

## 4. Accuracy — the two risks this session existed to retire, both clear

**fp16 accumulation at K=1024.** `P·V`'s HMX accumulator sums up to 1024
terms in fp16 — a real risk relative to the fp32-accumulating CPU reference
it replaces (`compute_vcache_fp32_transposed_reference`, `mha_core.cpp:73`).
Measured against realistic softmax-like data (non-negative, rows summing to
~1, not a synthetic ramp): **max rel err 3e-4** at both kv=512 and kv=1024.
Against a scalar reference that rounds to fp16 after every term (isolating
"fp16 accumulation is the limit" from "HMX is wrong"): 9-11e-3 — i.e. the
**HMX accumulator carries more effective precision than naive per-term fp16
rounding would suggest.** No K-chunking mitigation is needed.

**Transposed-P indexing.** Feeding P into the P·V activation slot pre
transposed (`[gqa, kv]` instead of the native `[kv, gqa]`, see §6) is exactly
where an off-by-transpose bug would hide silently — the shapes still line up,
just wrong. Re-verified against the fp32 reference at every step; maxRelErr
held at 3e-4, unchanged from the untransposed measurement.

---

## 5. Design rules established here — read before touching any HTP kernel that streams from DDR

1. **Row size, not descriptor count, is what determines Hexagon DMA
   bandwidth.** Measured on this engine, same 256 KB, varying only geometry:
   64 B rows → 8.4 GB/s; 2 KiB → 42; 64 KiB → 55; 256 KiB → 67 GB/s.
   Descriptor count barely matters (1 vs 64 descriptors at a similar row size
   stays within ~48-67 GB/s). **The ~11.5 GB/s this session first measured for
   activation DMA was a descriptor-shape artifact of 64 B-row tile packing,
   not a hardware ceiling** — do not treat any early single-digit-GB/s number
   as the engine's limit without checking the row size first.
2. **VTCM source is worth another ~4× over DDR source, at identical
   geometry.** The 64 B-row tile repack `rm_to_ah_f16` needs costs 8.4 GB/s
   from DDR but 31.6 GB/s VTCM→VTCM. So: one big contiguous DDR→VTCM grab,
   then repack on-chip, not per-kv_head strided DDR pulls.
3. **A hard rule, confirmed independently twice (K-cache repack, V-cache
   repack): any micro-API "extraction-style" function — one taking
   `row_tile/col_tile/input_rows/input_cols` addressing, e.g.
   `copy_submatrix_to_f16`, `rm_to_wh_f16` — must read from DDR, never VTCM.**
   Reading VTCM directly through these costs ~10-400× more than reading DDR
   through the same call (measured: ~26 µs/tile from VTCM vs ~1-3 µs/tile from
   DDR; one case measured 3376 µs vs 8.3 µs for the same 256 KB). The
   "pure-format" functions — fixed 32×32, no larger-matrix addressing,
   `rm_to_ah_f16`/`ah_to_rm_f16` — are fast regardless of source. This is a
   property of the *function*, not the data, and it is not documented in the
   HexKL header.
4. **A single kv_head's stride is not an obstacle — one 2D DMA descriptor
   carries it.** The K/V cache is `[kv_position][kv_head][head_dim]`; a
   single head's slice is strided by `nch*head_dim*2` bytes, but
   `row_size=head_dim*2, src_stride=nch*head_dim*2, nrows=kv_len` expresses
   that in one descriptor. Doc 08 §2's "a gather/pack step is unavoidable" is
   wrong for this reason — the pack is a DMA parameter, not a compute step.
5. **Bake incrementally at KV-append time, keep the WH bytes in host RAM —
   now measured, not just designed.** Doc 08 §6 proposed this; this session
   measured it: V's full 128-tile re-bake costs 123.7 µs/head, the
   append-time incremental bake (4 tiles, the newly-dirtied k-tile) costs
   3.35 µs/head — a 32× difference that is the entire reason decode P·V wins.
6. **Cross-call weight prefetch maps to attention too, just retargeted.**
   Doc 13 §4 called this "does not apply" for attention because K/V are
   activations with nothing static to prefetch across *tokens*. It still
   applies **across kv_heads within one call**: while head *h* computes,
   prefetch head *h+1*'s KV slab. This is the same dmlink-ring mechanism doc
   13 verified for FC's cross-*matmul* prefetch, aimed at a different
   boundary.
7. **`Q·Kᵀ` should be attempted before `P·V`, but not for doc 08 §2's
   reason.** Doc 08/13 chose P·V first because SDKL's macro API requires
   `N % 32 == 0` and `Q·Kᵀ`'s `N = kv_len` varies per token — a constraint
   that does not exist once you own the tile loop (the micro API). The real
   reasons to do `Q·Kᵀ` first: written as `Sᵀ = K·Qᵀ`, it needs no transpose
   (K is the natural-orientation activation, `Qᵀ` is a cheap 1 KB transpose),
   its output **is** the existing head-minor scores layout, its accumulation
   depth (K=128) carries none of §4's accuracy risk, and it depends on
   nothing from the softmax owner. `P·V` carries the accuracy question and a
   real (if small) layout dependency (§6).

---

## 6. The one thing worth raising with the softmax/HVX owner — not blocking

`P`'s native layout (`[kv, gqa]`, head-minor, written by the CPU softmax) is
the wrong orientation for the P·V activation slot, which wants `[gqa, kv]`.
Feeding it pre-transposed (i.e. asking the softmax owner to write it
`[gqa, kv]` instead) drops P-staging from 90.7 µs to 4.4 µs/head at kv=1024.
**This is worth asking for, but is not a blocker either way** — a rough,
unoptimised DSP-side scalar transpose of the untransposed layout measured
7.3 µs/head at kv=1024 (3.7 µs at kv=512), which is a few percent of the
per-layer total either way. Everything in §3 already assumes the
pre-transposed layout is available; if it turns out not to be, add ~58 µs to
the decode P·V per-layer total and the 3.4× shrinks to roughly 2.9×, still a
clear win.

---

## 7. What is *not* in the numbers above — read before trusting them for a ship decision

**Per-op FastRPC round-trip cost is entirely excluded.** Every DSP number in
§3 was measured by a standalone Hexagon SDK test harness
(`run_main_on_hexagon`) running as a single, self-contained DSP `main()` —
there is no host↔DSP call boundary inside the timed region at all. Doc 13
recorded that SDKL's own *production* macro-API path pays ~97 µs of per-op
FastRPC overhead on top of its NPU time. Nothing in this branch's actual
production dispatch path (`hexkl_mm.cpp`, called from `HtpComputeOps`) has a
persistent, batched-call skel that could avoid paying that cost per
`mha_core.cpp` call — building one is a separate, substantial piece of
infrastructure, not a detail of wiring `mha_core.cpp` into `ComputeOps`. This
is tracked as the architecture-level open item; do not read §3's numbers as
"this is what a real forward pass would see" until that skel exists.

**VTCM is 8 MiB on this device, confirmed by `hexkl_micro_hw_init` across
every run in both this session's bench and the FC u8i4/u8i8 bench** — not the
18 MiB doc 08 §10 reports from `sdkl_npu_get_hw_info`. The two APIs disagree
and this is unresolved; use 8 MiB for any VTCM budget check until reconciled.
It does not change any conclusion in this document (everything measured fit
comfortably), but it would matter for a larger fused design.

**The FC bench's SDKL-macro correctness check is unreliable — a pre-existing
bug, not introduced this session.** While parametrising `hexkl_micro_fc_bench.c`
over u8i4/u8i8 (a separate but related piece of this session's work), the
`sdkl_correct` check was found to flip OK/WRONG for the same shape across
reruns — the signature of a missing `qurt_mem_cache_clean()` around the
`hexkl_macro_mm_*` calls (the header requires FLUSH on inputs,
FLUSH_INVALIDATE on output; this bench has never done that). The **timing**
numbers are unaffected (HMX pays the same tile-multiply cost regardless of
whether the operand it read was stale), but do not cite that bench's SDKL
correctness column as a correctness proof until the cache-clean calls are
added.

---

## 8. Reproducing this

```bash
./test/unittest/jni_htp/hexagon/run_micro_mha_bench.sh   # DSP numbers, §3
./test/unittest/jni_htp/hexagon/run_mha_cpu_bench.sh     # CPU numbers, §3
```

Both run on a plain invocation with no env overrides and print every shape in
§3's table via self-describing `MHA_FIELD shape=... field=..._per_head|_8heads
value=...` marker lines — units are embedded in the field name specifically
because an earlier draft of the MHA bench's report script silently dropped
every shape past its first hardcoded name and printed a stale number instead;
keying by field name, not column position, is what stops that recurring.

---

## 9. What this changes for the next step

The next piece of work is wiring this into `mha_core.cpp` for real — a
`ComputeOps` seam (§1's "the win is verified, not yet shippable" applies
directly) plus the persistent-skel/FastRPC question §7 flags as unresolved.
That is architecture work, not a benchmark, and is tracked separately from
this document.

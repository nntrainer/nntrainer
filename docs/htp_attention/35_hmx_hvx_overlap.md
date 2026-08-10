<!-- SPDX-License-Identifier: Apache-2.0 -->

# 35 -- Keeping HMX and HVX both busy: ceiling, feasibility, staging

Proposal: schedule work so HMX never idles while HVX runs, the way llama.cpp
ggml-hexagon does (`htp/hmx-queue.c`, `matmul-ops.c`), because QNN's attention
is >2x faster than ours and reportedly does exactly this.

This is the analysis before any code. Doc 34 and doc 32 §5 are where every
measured number comes from; anything derived is marked.

## 1. The comparison is not like-for-like, and that comes first

**QNN's "attention layer" is the whole block: q/k/v projection, RoPE, the
scaled-dot-product attention, and o_proj. Ours is the SDPA core only.** RoPE
we do not implement at all -- `hexkl_attn_u8.h` documents Q and K as arriving
post-RoPE, so it is caller-side work outside every number we publish.

| seq | QNN (ms), S26U | HexKL (ms), S25U | what each covers |
|---|---|---|---|
| 512 | 1.45 | 4.068 | QNN: qkv + RoPE + SDPA + o. Ours: SDPA |
| 1024 | 3.21 | 6.093 | same mismatch |

So the published 1.9x understates the gap: **QNN does strictly more work in
that time.** Adding our own projection costs from doc 34 (derivation in §2)
puts the real block-level figure near **4x**, not 1.9x. Before anything is
built, that table needs a footnote or a matching measurement -- optimising
against a number that flatters us by 2x is how the wrong lever gets pulled.

Two provenance questions to close while doing it: whether QNN's 1.45/3.21 are
NetRun (wall) or device-timeline, and which of ours 4.068/6.093 are. Doc 34
§2 showed those two differ by ~2x on FC, so mixing them silently is a real
risk.

One thing the earlier QNN data does settle: their **FC** profile's stages sum
to 343.6 us against a 347 us total, so *within that one op* QNN is serialized
too. The scheduling they get credit for is therefore **across ops in a graph**,
not inside one -- which matches the attention claim exactly, and tells us
where to aim.

## 2. Where the attention block's time actually is

Measured: SDPA at kv=1024 prefill (doc 32 §5). **Derived**: q/k/v and o_proj
scaled from doc 34's measured q_proj -- quant by M·K, dequant by M·N, mm and
acc_read by tiles, and quant counted ONCE for a shared-activation group call.
HMX lane = mm + acc_read; HVX lane = quant + dequant + softmax + gather.

| part | HVX us | HMX us | total |
|---|---|---|---|
| q/k/v (one x3 call) | 1,757 | 1,973 | 3,731 |
| o_proj | 1,545 | 809 | 2,354 |
| SDPA | 5,206 | 1,614 | 6,820 |
| **block** | **8,508** | **4,396** | **12,905** |

**The block is 66% HVX-bound, and 12.9 ms against QNN's 3.21.** That is the
honest starting point. Note what the HVX lane is made of: quant and dequant
are ~6.6 ms of the 8.5, and they exist because **every op boundary of ours
round-trips through f32 in DDR** -- q_proj emits f32, SDPA re-quantises it to
u8, SDPA emits f32, o_proj re-quantises it again. QNN's graph stays uint8
between ops and pays one i32->u8 requantize instead.

## 3. Scheduling alone does not close a 4x gap

| | block us | win | vs QNN 3,210 |
|---|---|---|---|
| today | 12,905 | -- | 4.0x |
| + perfect HMX/HVX overlap | 8,508 | 1.52x | **still 2.7x** |
| u8 op boundaries instead | 8,127 | 1.59x | 2.5x |
| **u8 boundaries, THEN overlap** | **4,396** | **2.94x** | **1.4x** |

Perfect overlap is a ceiling nobody reaches, and it caps at 1.52x here for a
structural reason: **you cannot overlap your way out of being 66% one-lane.**
The most a scheduler can do is hide the smaller lane, and ours is HMX.

The order is not arbitrary:

- **u8 boundaries first.** They *delete* ~4.4 ms of HVX work rather than
  hiding it, and doing overlap first means tuning a pipeline around work that
  is about to be removed -- chunk sizes, buffer counts and lane balance would
  all be re-derived afterwards anyway.
- **Then overlap, and it pays MORE than it does today.** After the u8 change
  the lanes are HVX 3,731 against HMX 4,396 -- nearly balanced, which is
  precisely the regime where overlap extracts its maximum. Overlap on top of
  u8 is worth 1.85x; overlap alone is worth 1.52x.

Together they are ~2.9x and land at 1.4x of QNN while still not doing RoPE.
Neither alone gets under 2.5x.

**A third lever, unmeasured:** the block is **three separate FastRPC calls**
today (q/k/v, SDPA, o_proj). QNN runs it as one on-device graph. Three calls
means three transports and three hard barriers -- no scheduler can overlap
o_proj's weight DMA with SDPA's tail across a FastRPC return. Fusing the
block into one DSP call is what makes cross-op scheduling *possible* at all,
and on FC the measured transport at M=1024 was 963 us per call. This should
be measured before it is sized, but it is plausibly the cheapest of the three.

## 4. Feasibility: three structural facts

**(a) The HMX lock is thread-affine, and this is the one real risk.**
ggml's HMX thread calls `HAP_compute_res_hmx_lock()` *inside its own loop*
(`hmx-queue.c:46-49`) rather than inheriting a lock -- only necessary if the
lock binds to a thread. Our `hexkl_micro_hmx_lock()` is taken once in
`nntr_hvx_open` (`hvx_add_f32.c:78`) and used by layer calls arriving on later
FastRPC invocations, so either the lock is not thread-affine or FastRPC pins a
session to one thread. **We do not know which**, and a dedicated HMX thread
turns that unknown into a hard dependency.

**So do not copy ggml's topology.** Invert it:

| | who drives HMX | who does HVX | HMX lock |
|---|---|---|---|
| ggml | a dedicated queue thread | caller + pool | must move to that thread |
| **ours (proposed)** | **the caller thread, as today** | **pool workers, async** | **untouched** |

Same overlap, no new thread, no lock question, and the HMX side stays the code
already gated bitwise. If a real HMX thread is wanted later, settle the lock
with a 20-line experiment (lock on thread A, issue one micro-mm on thread B)
rather than discovering it mid-implementation.

**(b) One accumulator, so the pipeline is 2-stage, not deeper.** HMX cannot
start tile i+1 until `acc_read(i)` has drained the accumulator to VTCM. The
structure is fixed: `[clear, mm x k_tiles, acc_read -> buf A]` for tile i while
HVX dequants `buf B` from tile i-1. Two result buffers, alternating.

**(c) The worker pool is fork-join only.** `hvx_worker_pool_run()` blocks and
runs unit 0 on the caller. This needs `submit()` + `wait()` with every unit on
a worker, since the caller is busy issuing HMX. An addition to
`hvx_worker_pool.{c,h}`, not a rewrite -- the parked-thread machinery and the
`(n_threads, i, ctx)` contract already exist.

## 5. Granularity and VTCM budget

**Per tile is far too fine.** At M=1024 dequant is 558 us over 1,024 tiles =
**0.55 us per tile**, against a fork/join measured in the multi-us range (doc
32: "the pool pays at the >= 100 us jobs it now runs"). That measurement is
what killed pooling the tile dequant the first time; a job per tile repeats
the mistake.

Pipeline a **chunk of N-tiles** instead, 16 of them:

| | per 16-tile chunk, M=1024 |
|---|---|
| HMX work (mm + acc_read) | 15.6 us |
| HVX work (dequant) | 8.7 us |
| VTCM staging, double-buffered | 16 x 8,192 x 2 = **256 KB** |

Both sides clear fork/join, and 256 KB fits: the arena peaks at 5.25 MB of
~8.3 MB usable at M=1024 i8, 3.15 MB at i4. Chunk size is one constant, to be
swept on device rather than argued about.

`hexkl_acc_layout_get()` ramp-probes the accumulator permutation at ONE
`result_off`; with two buffers it must run at both or assert they agree.

## 6. What discounts the ceiling

**VTCM bank contention, already measured on this device.** Doc 32 §5: moving a
pool-parallel working set into VTCM cost prefill softmax **+30%** while the
single-threaded decode path was unchanged to the microsecond -- six HVX threads
contending for VTCM banks lost to six threads hitting L2. This puts HMX writing
a VTCM tile beside workers reading one: same class. **Plan for 1.4-1.6x of the
overlap term, not its ceiling.**

Second, the probes stop being additive once stages overlap: `quant_us +
dequant_us + ...` will exceed `dsp_total` and the report's
"remainder = micro-mm" arithmetic breaks silently. Per-lane totals must land
first.

## 7. Staging, with a gate on each step

| step | work | gate before continuing |
|---|---|---|
| 0 | measure the real q/k/v/o shapes (add 3 entries to `kShapes`) and footnote the org table | block total is measured, not scaled from q_proj as §2 is |
| 1 | u8-in / u8-out layer + attention endpoints (doc 34 §5) | existing bitwise and integer-reference gates PASS; HVX lane drops |
| 2 | per-lane probe totals + a `pipelined` flag in FC_STAGE | breakdown still sums with overlap off |
| 3 | `hvx_worker_pool_submit/wait`; second result buffer; layout probe asserts both offsets | bitwise timed==production PASSes at all 24 rows |
| 4 | chunked dequant pipeline (HMX on caller, HVX async) | bitwise gate PASSes; `dsp_total` drops; sweep chunk size |
| 5 | fuse q/k/v + SDPA + o_proj into one DSP call | transport drops ~2 calls' worth; cross-op overlap becomes possible |
| 6 | only if HMX is still critical after 1-5: the real HMX thread, after the lock experiment | -- |

Step 0 is the cheapest and highest-value: it is a handful of lines, and it
decides whether §2's derived 12.9 ms is right before anything is built on it.
Stop at any step that does not move `dsp_total`.

## 8. Where this does NOT help

- **Decode.** The FC lane split at M=1 gives a 1.18x ceiling on a 113 us
  kernel, against 326 us of FastRPC in the same call. Decode's problem is
  transport, and step 5 is the one that touches it.
- **Anything before step 1.** Overlap tuned around quant/dequant that step 1
  deletes is tuning that gets thrown away.

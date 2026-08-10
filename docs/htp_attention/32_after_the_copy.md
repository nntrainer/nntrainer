<!-- SPDX-License-Identifier: Apache-2.0 -->

# 32 -- After the copy: what remains, and what ggml-hexagon does about it

State as of the in-place tile dequant (80db05e), measured on device, all
correctness gates unchanged (S band bit-identical, fused forward 3.21e-05):

| shape | dsp before | dsp now | wall now | vs CPU bar |
|---|---|---|---|---|
| kv=512 decode | 10,475 | **326** | 427 | **3.6x faster** |
| kv=1024 decode | 20,869 | **578** | 714 | **2.0x faster** |
| kv=512 prefill | 51,980 | **11,644** | 14,916 | -- |
| kv=1024 prefill | 100,507 | **19,456** | 23,423 | -- |

Decode is now UNDER the plan's §2 prediction. 28 layers of decode attention:
666 ms/token at the start of this effort, 20 ms/token now.

## 1. Where prefill kv=1024's 19,456 us actually is

The stage table hides quant inside Q.Kt and P.V (it runs inside layer_run).
Attributed by scan+pack volume -- P is 128 calls of 64x256 against Q's 32 of
64x128, so 89% of the 10,082 us quant probe is P:

| item | us | share | note |
|---|---|---|---|
| activation quant | 10,082 | 52% | ~9,000 of it re-quantizing P per block |
| f32 accum + gather/1-l | 4,607 | 24% | scalar loops in hexkl_attn_u8.c |
| dequant (in-place tiles) | 2,250 | 12% | already vectorized |
| mm + readout + dma + drains | ~2,500 | 13% | acc_read floor is 610 us of this |

Decode kv=1024 (578 us) is per-call overhead + the acc_read floor (145 us);
its next lever is not the kernel but FastRPC (transport 136-150 us/call and
28 calls/token).

## 2. What llama.cpp ggml-hexagon/htp does differently (read at master, 2026-08)

Their flash attention (`flash-attn-ops.c`, `hmx-fa-kernels.h`) is the same
problem on the same accelerator with different choices:

1. **fp16 HMX, no quantization anywhere in the FA path.** Q/K/V stay fp16;
   S and P are fp16 tiles. The entire 52% quant bucket above does not exist
   for them, and neither does dequant: the fp16 accumulator drains through
   `mxmem(out, scale):after.hf`, a hardware store that applies a column scale
   on the way out (2 KB fp16, not 8 KB int32).
2. **Online softmax IS expressible on HMX** -- as a diagonal-matrix multiply.
   `hmx_fa_o_update_tile` computes `O_new = D.O_old + P.V` in one
   accumulation group, where D is a 32x32 diagonal tile holding
   exp(m_old - m_new) per row (built by `fa_build_d_diag_inv_l`, scattered
   onto the diagonal with precomputed vscatter offsets). Our plan's "no
   accumulator-scaling primitive" claim is true of the INT path through
   HexKL, but false of fp16 HMX driven directly. That is what unlocks a
   streaming loop with no S band in DDR at all.
3. **A dedicated HMX thread** (`hmx-queue.c`): HMX work is pushed as jobs to
   one parked thread; the calling thread and the HVX worker pool do
   interleave/softmax for block j+1 while HMX runs block j. HMX and HVX
   genuinely overlap instead of taking turns.
4. **Everything lives in VTCM, double-buffered** when pipelining: K, V, S,
   P tiles all have [2] buffers; DMA (their `dma_queue`, same lineage as our
   ring) prefetches block j+1's K/V/mask while j computes.
5. Small tools worth stealing eventually: fastdiv for hot-loop div/mod,
   vscatter-based transposes that fuse layout change into the store.

## 3. The list, ranked by measured ceiling

| # | item | attacks | ceiling (pre kv1024) | shape of work |
|---|---|---|---|---|
| 1 | merge htp/quant-dequant-hvx-opt's quant pool + vectorized pack | 10.1 ms quant | -> ~2-3 ms | already written on that branch; resolve the copy-out commit (75e2193) in favor of our in-place dequant |
| 2 | accumulate flag on the tile dequant + 1/l on the last block | 2.7 ms accum + part of 1.9 ms gather | -> ~0 | we own hvx_dequant_acc_tile_to_f32 now; one extra vector load+add per row, deletes o_part and both scalar loops |
| 3 | softmax exports per-(row,block) maxima -> P quant params without the scan | the scan half of #1's residual | bit-identical by construction, unlike the constant-params idea 30_/attn header records | kernel + host ref + gates, one task |
| 4 | HMX job thread (ggml's hmx-queue pattern) | serialization of quant/softmax vs HMX | overlap min(HVX, HMX) ~ up to 30% | new concurrency; after 1-3, not before |
| 5 | FastRPC: layer batching + QoS vote | decode wall (28 x ~430 us) | decode e2e 2-3x | host-side seam change |
| 6 | fp16 HMX attention path a la ggml (D-diag online softmax) | quant+dequant+accum wholesale | the endgame; ref_14 already verified fp16 micro on this device | new kernel family; PR-sized |

## 5. Falsified on device, do not retry without new evidence

**s_band in VTCM (2fa199c, reverted).** The premise was that s_band carries
~32 MB of DDR traffic per prefill layer for a 256 KB buffer, so parking it
in VTCM would delete that traffic. Measured A/B with everything else fixed
(poll QoS and ION on in both), kv=1024 prefill:

| stage | s_band in DDR | in VTCM | |
|---|---|---|---|
| dequant (writes it) | 2,357 | 2,353 | **no change** |
| quant (reads it) | 1,166 | 1,155 | **no change** |
| softmax (pool, 6 threads) | 806 | 1,050 | **+30%** |
| gather+1/l | 877 | 1,002 | +14% |
| dsp_total | 6,820 | **7,239** | **+6%** |

Two things this settles. The producer and consumer of s_band did not care
where it lived, so that traffic was never reaching DDR -- 256 KB reused
within a band iteration stays in L2, and the "32 MB" was 32 MB of L2 hits.
And the one stage that got worse is the one running on the worker pool:
prefill softmax +30% at kv=1024 and +50% at kv=512, while DECODE softmax,
which takes the inline single-threaded path, was 15 and 30 us in both runs
-- unchanged to the microsecond. Six HVX threads contending for VTCM banks
lose to six threads hitting L2.

Anything that moves a pool-parallel working set into VTCM inherits this.
A single-threaded stage might still win there; nothing measured says so yet.

Not worth it, measured: pooling the tile dequant (1.5 us per tile against a
multi-us fork/join -- the pool pays at the >= 100 us jobs it now runs),
V-width narrowing for speed (still within noise), decode-side kernel work
(the kernel is 326 us; the phone call to it costs more).

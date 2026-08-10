<!-- SPDX-License-Identifier: Apache-2.0 -->

# 31 -- The fused attention dataflow, as built

Where every buffer actually lives and which engine touches it, drawn from
`hexkl_attn_u8.c` and `hexkl_mm_u8i4_dma.c` rather than from the design intent.
Figures are pure ASCII on purpose: they get pasted into other documents, and
box-drawing characters mixed with CJK text do not survive that.

Read `10_mha_htp_plan.md` for why the loop has this shape and
`11_u8_task_split.md` for the quantization contract. This file only says what
the code does today.

## 1. kv_append -- once per step, outside the forward pass

```
+- DDR ---------------------+   +- HVX ------------------+   +- DDR (registry) ------+
|                           |   |                        |   |                       |
| K rows (fp16) --- memcpy ---> | hexkl_kvq_pack_kt_block|   |                       |
| V rows (fp16) --- memcpy ---> |   fp16 decode          |   |                       |
|                           |   |   per-axis scale       |   |                       |
|   k_shadow, v_shadow      |   |   WH tile layout ----------> Kt_wh (i4 | i8)       |
|   (fp16, kv_len rows)     |   |                        |   | V_wh  (i4 | i8)       |
|                           |   |                        |   | + w_scale, colsum_w   |
|                           |   |                        |   |   = registered handle |
+---------------------------+   +------------------------+   +-----------------------+

  Only blocks touched by the new rows are re-registered, so append cost does
  not grow with kv_len. Quantization happens HERE, once. The forward pass
  never re-quantizes K or V -- it only DMAs the WH tiles into VTCM.
```

## 2. forward -- one (head, band) iteration

```
+- DDR --------------------------+  +- VTCM (8 MB) -------+  +- ENGINE -----------+
|                                |  |                     |  |                    |
| q[n_query][nHq][head_dim]      |  |                     |  |                    |
|   | (1) gather: memcpy, GQA    |  |                     |  |                    |
|   v                            |  |                     |  |                    |
| q_gather[mb][head_dim] --------+->| quant + pack -------+->| HVX  f32 -> u8 AH  |
|                                |<-+-- act AH tiles <----+--|  once per band     |
| Kt_wh blocks 0..n-1 ---- DMA --+->| wbuf[0] / wbuf[1]   |  |  shared by all blk |
|   * cross-prefetch: ON         |  |   double buffer     |  |                    |
|                                |  |        |            |  |                    |
|                                |  |        v (2) Q.Kt   +->| HMX  32x micro-mm  |
|                                |  |                     |  |        |           |
|                                |  | result tile <-------+--| acc_i32 (64x32)    |
|                                |  |   64x32 i32, 8 KB   |  | acc_read_int32     |
| ### acc_scratch (i32) <--------+--| copy_32b_to_submat  |  |                    |
| ###   <== BOTTLENECK           |  |                     |  |                    |
|   |                            |  |                     |  |                    |
|   v dequant -------------------+--+---------------------+->| HVX  i32 -> f32    |
| s_band[blk][mb][T] <-----------+--+---------------------+--|  zp, colsum, scale |
|   |                            |  |                     |  |                    |
|   v (3) softmax, in place -----+--+---------------------+->| HVX  one pass over |
| s_band becomes P               |  |                     |  |  the whole band    |
| l_row[mb] <--------------------+--+---------------------+--|  (NOT online)      |
|   |                            |  |                     |  |                    |
|   | (4) for each block j:      |  |                     |  |                    |
|   +-- seg[j] ------------------+->| quant + pack -------+->| HVX  P f32 -> u8   |
|   |                            |  |                     |  |  requant per block |
| V_wh block j ---------- DMA ---+->| wbuf[0]             |  |                    |
|   * prefetch: NONE, blocking   |  |        | (5) P.V    +->| HMX  32x micro-mm  |
|   |                            |  |        v            |  |        |           |
| ### acc_scratch (i32) <--------+--| result tile <-------+--| acc_i32            |
|   |                            |  |                     |  |                    |
|   v dequant -------------------+--+---------------------+->| HVX  i32 -> f32    |
| o_part[mb][head_dim] <---------+--+---------------------+--|                    |
|   |                            |  |                     |  |                    |
|   v (6) o_band += o_part       |  |                     |  | scalar f32 accum   |
|   +-- end of block loop        |  |                     |  |                    |
|   v (7) out = o_band * (1/l)   |  |                     |  | exactly once       |
| out[n_query][nHq][head_dim]    |  |                     |  |                    |
+--------------------------------+  +---------------------+  +--------------------+
```

What lives in VTCM: the activation AH tiles, the weight double buffer, and one
64x32 result tile. That is all. `s_band`, `o_band`, `o_part`, `q_gather`,
`l_row` and `acc_scratch` are `malloc` -- DDR.

`seg[j]` points into `s_band`, so the softmax overwrites S with P in place;
there is no separate P buffer, and P is quantized inside the P.V `layer_run`
rather than as a step of its own.

## 3. The hot path, inside layer_run

```
  for rblock in 0 .. m_pad/64 - 1        <-- m_pad = ROUND_UP(M, 64).
    for n_tile in 0 .. N/32 - 1              decode has M=2, so 62 of the 64
      acc_clear                              accumulator rows are padding.
      for k_tile in 0 .. K/32 - 1
          HMX micro-mm ......................  ~0.1 us   64x32x32 (probed)
      acc_read_int32      [HMX acc -> VTCM]     0.3 us   vendor drain: free
      copy_32b_to_submat  [VTCM -> DDR]        52.8 us   8 KB, ~155 MB/s <== HERE
  drain DMA
  hvx_dequant_i32_to_f32  [DDR -> DDR]

  one block, model vs measured
    Q.Kt   32 micro-mm (12 us) + 8 readouts (416 us) = 428     measured 429 us
    P.V    32 micro-mm (12 us) + 4 readouts (208 us) = 220     measured 221 us
                                ^^^^^^^^^^^^^^^^^^^
                                reading the accumulator costs 34x the
                                multiplies that filled it
```

A two-parameter model -- ~0.1 us per micro-mm, 52.8 us per accumulator
readout -- predicts all four attention shapes and the standalone `layer_x4`
micro-mm benchmark within 7%. The Tier 0 probes then split the readout: the
vendor drain into VTCM costs 0.3-0.4 us, and `copy_32b_to_submatrix` to the
DDR scratch carries the remaining 52.8 us, identical at every shape and both
widths. The bottleneck is one strided 8 KB store loop into uncached DDR --
96% of decode qk+pv, 84% of prefill -- and it is entirely deletable: the
tile is already in VTCM when the copy starts, so dequant can read it there
and write f32 straight to the output, no scratch at all. After that, prefill's
next item is activation quant (5.7-10.1 ms, ~10%) -- softmax was second until
PHASE B moved onto the worker pool and dropped to 0.5-0.7 ms.

## 4. Where this differs from the intended design

| # | Intended | As built |
|---|----------|----------|
| 1 | Online softmax, running `l` correction | Blocked: every block's S is computed first, then one masked pass over the whole band; `1/l` applied once at the end. There is one HMX accumulator and no primitive that scales it, so online is not expressible. |
| 2 | S, P, O held in VTCM | All in DDR. VTCM holds only act tiles, the weight double buffer, and one result tile. |
| 3 | P quantization is its own stage with its own buffer | S becomes P in place; quantization happens inside the P.V `layer_run`. |
| 4 | Accumulator feeds HVX directly | `acc_read_int32` to VTCM, `copy_32b_to_submatrix` to a DDR scratch, then dequant reads it back. This round trip is the bottleneck. |
| 5 | Asynchronous V-cache prefetch | Not running. P.V is called per block with `n_handles = 1`, so there is no next handle to prefetch behind. Q.Kt does prefetch, because it passes every block's handle in one call. |
| 6 | Q DMAed into VTCM as f32 | f32 Q never enters VTCM. `hvx_quant_pack_u8_ah` reads DDR and writes u8 AH tiles straight into VTCM. |
| 7 | dmlink weight prefetch | User-DMA ring (`hexkl_dma_ring_push2d`). |

Resident Kt/V in DDR, and cross-matmul prefetch on Q.Kt, are as intended.

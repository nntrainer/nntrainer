# A8W8 / A8W4 flash attention on HMX over a quantized KV cache

Status: 2026-09-29. All phases through Q6 are delivered. Q1, Q2, Q2b
(A8W8 and A8W4 prefill kernel, at throughput) and Q4 (the vrmpy decode
kernel) run on device (SM8850 / v81); Q3's kind switch came for free with
Q2. Q5 / Q5b (the `ComputeOps` / `MHACoreLayer` seam, the
`attention_kv_dtype` config key, one FastRPC call per layer per step) run
end to end: Qwen3-0.6B (fp32 weights, 28 layers, 16/8 heads, hd 128) on
the device produces the identical 24 greedy tokens with the CPU, with
`attention_engine: htp` over the fp16 cache, and with `attention_kv_dtype:
q8`, every layer's attention on the DSP; `q4` collapses to "a a a a ..."
-- int4 K with one scale per token is what the model analysis said it is,
so int4 stays an experiment until a K-int8 / V-int4 kind exists. Q6's
long-context table (below) says where the DSP pays: prefill, not
per-token decode. Branch:
`htp/quant-dequant-hvx-opt`. Builds on `20_hmx_flash_attention_plan.md`
(fp16 attention, delivered through Phase 4d).

- Q1: `unittest_hexkl_kv_q` 5/5 host, `unittest_hvx_attn_q` registry tests
  4/4 device -- the DSP quantizer matches the same C on the ARM side bit for
  bit for both kinds, HMX over the baked int8 and int4 tiles equals a plain
  int matmul over the masters exactly, and the int32 accumulator readout
  probe found `usable=1 base=0 row_stride=32`: a plain row-major 64x32 int32
  tile, 64 HVX vectors, no permutation table.
- Q2: `hexkl_attn_q_prefill` reproduces a CPU model of its own arithmetic
  (`test/unittest/hvx_attn_q_model.h`) to 45-59 dB on every shape (GQA,
  straddling q blocks, window, softcap, sink, hd 32..256, both kinds), and
  its SNR against the exact reference equals the model's to within 0.05 dB.
  The device gate is therefore agreement with the model (>= 45 dB) plus a
  sanity floor; the absolute numbers below are the *scheme's*, measured on
  the test's uniform random data (Q at +-3, K/V at +-1) with the model:

  | term alone | dB | | full scheme | dB |
  |---|---|---|---|---|
  | Q u8 per row | 48.2 | | A8W8, bc=32 / 64 / 128 / 256 / 1024 | 41.8 / 41.1 / 40.3 / 39.3 / 37.1 |
  | K i8 per token | 48.0 | | A8W8 with P exact (bc=128) | 43.4 |
  | V i8 per token+group | 48.2 | | A8W4 | 20.0 |
  | P u8 per row+block (bc=128) | 43.2 | | K i8 + V i4 | 23.2 |
  | K i4 or V i4, any grouping | 23.0 / 23.2 | | K i4 + V i8 | 22.9 |

  Every int8 term sits at the theoretical floor of 8-bit uniform
  quantization on uniform data (step/sqrt(12) against amax/sqrt(3) is
  48 dB), and any int4 term at its 23 dB; finer scale groups do not move
  uniform data, and neither does the u8 zero point by more than ~1 dB. So
  the plan's "A8W4 >= 30 dB" was not reachable on this data by any int4
  scheme, and the K-vs-V asymmetry the prior branch saw was its per-channel
  V scale, not int4 itself. What int4 is worth has to be judged on real
  K/V (Phase Q5's model run), where token-correlated V and outlier-free
  groups behave very differently from uniform noise. Smaller blocks make u8
  P *more* precise (the row max is local), which also rules out a
  global-max two-pass P for accuracy reasons.
- Q2b (throughput): 128x1024 (16/4 heads, hd 128) went from 16 ms to
  3.95 ms on-DSP against the fp16 resident path's 2.58 ms; 32x4096 from
  12.0 to 2.69 ms. The DSP runs at 2.1 GHz throughout (measured from the
  processor cycle counter the stats now carry). The cause was not the
  arithmetic: **the scalar unit's access to VTCM is slow** (tens of cycles
  per load or store, not pipelined). The first kernel kept its per-row
  metadata -- Q scales and zero points, P' scales, rescale factors -- in
  VTCM and read them with scalar loads to feed `vsplat`; gathered Q rows
  and block V scales into VTCM with `memcpy` and scalar stores; and
  extracted row maxima to scalars (`vextract`) for a scalar divide. Each
  cost 60-140 cycles per tile row against ~15 of vector work. Now the
  metadata lives in cached heap memory, Q rows are gathered with vector
  loads and stores, the registry stores V scales group-major so P-quant
  reads a block's scales from DDR as one unaligned vector per column tile
  (no staging), and the P' scale path is all vectors (row max broadcast by
  the reduction, Newton reciprocal, scale stored as the splat the O update
  loads). The rule for this backend: **VTCM is for vector instructions and
  DMA only; anything the scalar unit touches goes in heap memory.**

- Q4 (decode): `hvx_attn_decode_q` matches the CPU model (bc = 32, f32
  scores) to 69-137 dB on every shape (GQA, ragged rows, hd 32..128,
  window, softcap, sink, both kinds) -- effectively bit-exact, the f32
  polynomial exp2 and the model's exp2 agreeing to the last bit or two --
  and its SNR against the exact reference equals the model's: 42 dB at
  int8 (32-row P' blocks are more precise than the prefill's), 20 dB at
  int4. On-DSP time, 16/4 heads, hd 128: 1x1024 194 us (fp16 decode
  963 us, 5.0x), 1x4096 922 us (fp16 3734 us, 4.05x); int4 957 us at 4096
  -- the same bytes as int8 until the masters are packed. The trick that
  made it cheap: the masters are stored offset-binary (value + 128) so
  both matmuls are `vrmpy(Vub, Rub)` with Q or P' as 4 uint8 in a scalar
  register -- one vector load and one vrmpy per 32 rows x 4 dims (scores)
  and per 4 rows x 32 dims (values), no splats -- and the +128 comes out as
  one integer correction per block from the sum of the scalar side's
  bytes, which the P' quantization computes anyway. Every running softmax
  quantity is a vector with all lanes equal; nothing is extracted to a
  scalar and nothing touches VTCM.

- Q5 (end to end): three things kept the DSP path from ever engaging in
  the app until this phase, none of them in the kernels. (1) Every model
  class (Qwen3, Gemma, GPT-OSS, LFM2, Qwen2, BERT, ViT) builds its own
  `mha_core` and never saw the `engine` property the base
  `Transformer::createAttention` added; all of them now go through
  `Transformer::createAttentionCore`, which appends `engine` and
  `kv_cache_quant` from the config. (2) On Android with fp16 the layer
  hands its per-batch helpers fp16 Q/output steps, and the accelerated
  paths demanded f32; they now stage fp16 through f32. (3) `forwarding()`
  (the external-cache path) never captured the context's `ComputeOps`.
  The fp16 HTP path of Phase 4c had therefore never run in the app either.
  With those fixed, Qwen3-0.6B at 22 prompt + 24 generated tokens
  (512-row cache, NNTR_NUM_THREADS=4): CPU 287 ms prefill / 1081 ms
  generation; HTP fp16 310 / 1174; HTP int8 708 / 3181; HTP int4 1397 /
  3824 (and wrong). At this cache size the attention itself is a few
  microseconds per layer and the time is FastRPC: the fp16 path makes one
  call per layer per step, the quantized path two (append, then attend),
  each ~1 ms round trip including the DSP-side scalar quantizer of the
  appended rows, times 28 layers. The fix is mechanical and is the next
  item: one `attn_q_step` entry that appends the step's rows and attends
  in the same call, and an HVX quantizer for the appended rows. The DSP
  path pays for itself only where the attention is large -- long
  contexts, where the fp16 numbers above (2.6 ms vs the CPU's tens of ms
  per layer at 4096 rows) apply.

  Phase times, 128x1024, us: qprep 359, dma 259, qk 335, dequant 470,
  softmax 387 (exposed), pquant 1316, pv 304, oupd 308, store 119. What is
  left is P-quant's arithmetic on the pool (two passes over P per V
  group); folding its row-max pass into the shared softmax's pass 2 would
  take ~0.4 ms more, later. Also from this phase: `nntr_hvx_open` now
  votes the DSP to its top core / bus / HMX clocks (llama.cpp's sequence);
  it trimmed the HMX phases by ~30% and is a precondition for comparable
  numbers.

- Q5b (one call per layer per step): `attn_q_step` appends the step's
  rows and attends in the same FastRPC call, so the quantized path makes
  as many calls as the fp16 one. The append itself was 1.45 ms per row at
  int8 (scalar quantizer 252 us, staging 587 us, rebaking the 32-row tile
  607 us). Two changes took it to 30 us: the int8 WH tiles are written
  directly from a probed layout (two ramp bakes through `rm_to_wh_i8` at
  open give the byte position of every (row, column) of a 32x32 tile, and
  the appended row's 32 bytes per head-dim tile go straight to those
  positions -- no staging, no rebake), and the quantizer is HVX
  (`hvx_kv_quant.c`: widen, absmax, multiply, round-to-nearest-even,
  clamp and the column sum as vectors; the scale from the vector maximum
  on the scalar unit, so the scales are bit-identical to the C reference
  and the values differ by at most one step on rare f32 ties, which the
  dump test tolerates). Int4 still restages and rebakes (~1.3 ms per row)
  because its WH byte order is not the int8 one; a second probe is the
  obvious follow-up if int4 ever earns it. `StepTiming` on device
  (16/8 heads, hd 128, us): int8 append 30 / attend 109 at 512 rows, 41 /
  1137 at 4096; int4 1222 / 109 and 1371 / 1141. Model level, the same
  22 + 24 tokens as Q5: HTP int8 408 ms prefill / 1136 ms generation
  (was 708 / 3181), HTP fp16 310 / 1164, CPU 300 / 1077 -- the quantized
  path is now at parity with the fp16 one, and both sit on the FastRPC
  floor at this cache size.

- Q6 (model-level timing, long context): Qwen3-0.6B, a 3083-token prompt
  plus 32 greedy tokens, `init_seq_len` 4096, NNTR_NUM_THREADS=4, the same
  device. Per-call numbers come from `NNTR_HTP_ATTN_TRACE=1`, which logs
  every attention call's host wall time and the skel's stats to logcat
  (`HtpComputeOps`).

  | path | prefill (3083 tok) | generation (32 tok) | attention per layer: prefill / decode (host wall) |
  |---|---|---|---|
  | CPU | 34.6 s | 2.37 s (74 ms/tok) | -- |
  | HTP fp16 | 125.1 s | 8.80 s (275 ms/tok) | 3684 ms / 4.04 ms |
  | HTP int8 | 31.8 s | 4.31 s (135 ms/tok) | 210 ms / 1.54 ms |

  Both DSP paths reproduce the CPU's text; int8 agrees for 27 of the 32
  tokens and then continues with a different, plausible clause, the same
  in every run (the scheme's ~40 dB at 3k rows, deterministic). What the
  trace says:
  - **fp16 prefill is tile-conversion-bound.** Of a layer's 3.68 s, 3.59 s
    is the kernel's tile phase: the raw path converts every K/V block to
    the HMX layout again for each of the 97 query blocks (5096 block
    visits per layer). The quantized path bakes tiles once at append and
    attends the same shape in 110 ms; its whole 210 ms is 90 ms of
    quantizing 3083 rows (30 us per row, one thread) plus the 110 ms
    kernel. So the int8 prefill beats the CPU by 3 s, and the fp16 raw
    path is not usable for long prompts until the app uses the Phase 4d
    resident registry (baked tiles) the way the quantized path does.
  - **Decode is DDR-bound on the DSP and loses to four CPU cores.** The
    int8 decode call is 1.54 ms wall per layer: ~0.65 ms FastRPC round
    trip plus 0.85 ms kernel, which reads the 6.3 MB int8 K/V of 3083
    rows twice (the kernel's unit is one (query row, q head), so each KV
    head is streamed G=2 times), ~15 GB/s. fp16 decode is 4.04 ms
    (3.1 ms kernel over twice the bytes). The CPU's attention at this
    length is a few hundred microseconds per layer. 28 layers x 1.54 ms is
    43 ms of the int8 path's 135 ms per token; the remaining ~25 ms per
    token over the CPU run is not attention and is unexplained (CPU
    clocks dropping while the cores wait on the DSP is the likely cause;
    not measured).
  - Where the DSP pays: prefill of long prompts (int8: 3 s faster than
    the CPU here, ~40% of it the single-threaded row quantization) and
    memory (the int8 masters are half the fp16 cache). Where it does not:
    per-token decode at any length measured, because of the round trip
    and the DDR stream.

  Follow-ups in cost order: (1) quantize the appended rows on the worker
  pool (90 -> ~25 ms per prefill layer, ~1.8 s off the run); (2) walk the
  G q heads of a KV head together in the decode kernel (halves its DDR
  traffic); (3) route the fp16 path through the resident registry; (4) a
  batched decode that runs all layers' attention in one call is what it
  would take to remove the round trip, and needs the model loop's
  cooperation.

## 1. Where things stand

### 1.1 Reusable, on this branch
- fp16 attention (`hexkl_attn_f16.c`): FA-2 pipeline (softmax on the pool
  overlapping the next block's DMA + Q.K^T), causal geometry in
  `hexkl_attn_f16_plan.h`, fp16 HVX softmax (`hvx_attn_softmax_f16.h`),
  the pure-HVX decode kernel, the resident tile registry pattern
  (`hexkl_kv_tiles_f16`: register / append / attention by handle), and the
  device gtest with SNR gates. Device numbers: 62-78 dB, 27/27.
- A8W4 / A8W8 matmul: `hexkl_mm_u8i4{,_dma}.c`, `hexkl_mm_u8i8_dma.c` --
  WH bake once + resident weights, `hvx_quant_u8` (per-row asymmetric u8,
  writes AH tiles directly: a u8 activation tile is flat row-major 64x32),
  `hvx_dequant_i32` (`(acc - zp*colsum) * s_act * s_w + bias`).
- Session: int32 accumulator config is already the session's permanent
  one (`config_off`); the fp16 kernel writes its own f16 config per call.
- rpcmem-shared host cache (Phase 4d) -- not needed here: the quantized
  cache lives on the DSP, only the new rows cross FastRPC per step.

### 1.2 Prior u8 SDPA (`seunghui/htp/rope-u8-qnn-parity`, fa80ac82a; not here)
What it settled, and this plan takes as given:
- **Never call `hexkl_micro_hmx_copy_32b_to_submatrix` on the hot path**:
  52.8 us per 8 KiB tile, 80% of a prefill layer and 97% of a decode one.
  `hexkl_acc_tile.{c,h}` derives the int32 readout permutation at runtime
  by pushing a ramp through the vendor copy once; on every part measured
  the tile is row-major with a constant row stride, so a 64x32 int32 tile
  is 64 HVX vectors and the dequant reads it in place. This file is
  ported as is.
- **Quantization axes that work on HMX.** Dequant constants can only live
  on the non-reduced axes: for S = Q.K^T that is per Q row (activation)
  and per cache row (weight column); for O = P.V it is per Q row and per
  head dim. A per-token V scale therefore cannot be a "weight scale" --
  it sits on the reduced axis.
- **Accuracy**: (K i8, V i8) 5e-3 max rel err; (K i4, V i8) 2e-2;
  V at i4 with per-channel-over-block scales 5e-2 and red on rows that
  attend few positions. "V to i4 costs ~15x, K to i4 costs ~3x." Its V
  scale axis (per head dim over a 32-64 row block) is the reason: a
  symmetric 15-level grid over a block's column range.
- FastRPC fixed cost ~404 us per call; the transport spreads 90 -> 3900 us
  without a HAP power vote. Fork/join costs more than a decode-band softmax.
- It quantized on the host and kept fp16 shadows; no `ComputeOps` /
  `MHACoreLayer` seam existed. It used a blocked (two-pass) f32 softmax
  with S round-tripping DDR.

### 1.3 Contract (unchanged from the fp16 kernel)
Q f32 post-RoPE, out f32, causal by geometry, sliding window, softcap,
sinks, GQA rows packed as `q*G + g`. Reference is `MHACoreLayer`'s f32 math;
the gate is SNR against it. The int paths are expected to land lower than
fp16's 62-78 dB; the gates below are set from the prior branch's numbers
and tightened once measured.

## 2. Decisions

1. **Cache scales are per token, computed on the DSP at append.** K: one
   symmetric scale per (row, kv head) over head_dim, plus `colsum` (the
   int sum over head_dim) for the u8 zero-point correction. V: one
   symmetric scale per (row, kv head, 32-dim group). Every appended row is
   quantized once and never revisited, so the cache grows token by token
   without re-quantization and without calibration data. No host-side
   quantizer, no fp16 shadow.
2. **The per-token V scale is folded into P, per 32-dim group.** P.V on
   HMX produces output tile (row tile, d) from activation P and weight
   tile column d. Since the V scale of group d is per cache row k -- the
   reduction axis -- it is multiplied into P before P is quantized:
   `P'_d[q][k] = P[q][k] * s_v[k][d]`, then `P'_d` gets its own per-row u8
   scale. There are head_dim/32 activation buffers for P instead of one;
   HMX runs exactly the same number of passes (output tile (i, d) always
   used activation i and weight column d), and V gets llama.cpp's Q8_0 /
   Q4_0 granularity (32-value groups) for free. This is the answer to the
   prior branch's V-at-i4 result; it is verified on device before A8W4
   is called done.
3. **Q is per-row asymmetric u8** (`hvx_quant_rows_u8_params` +
   `hvx_quant_pack_u8_ah`, unchanged; the pool parallelism they already
   have). The score dequant is exactly `hvx_dequant_i32`'s formula with
   `act = Q rows`, `w_scale = s_k[k]`, `colsum_w = colsum_k[k]`, `bias = 0`,
   with `log2e/sqrt(hd)` folded into the per-row activation scale. P is
   u8 with zero point 0 (P' >= 0), per-row scale = row max / 255 -- the
   prior branch's finding that a fixed 1/255 is wrong for blocks that do
   not hold the row max.
4. **S and O leave the accumulator through HVX, in VTCM, never DDR.**
   acc(int32, 64x32) -> dequant per row -> f16 S tiles in the 2-row
   interleaved layout the fp16 softmax consumes, so `softmax_row_pair` runs
   unmodified. O is a f32 HVX-resident block `[g_br][hd]`: per cache
   block `O = a * O + s_p[q][d] * acc` (the online rescale is an HVX FMA,
   not the fp16 kernel's diagonal HMX matmul -- with int accumulators O
   cannot stay on HMX across blocks). Final `out = O / l`.
5. **Row tile is 64.** The u8 activation tile is 64x32, so `g_br` aligns
   to 64 (the fp16 kernel's 32). The geometry helpers in
   `hexkl_attn_f16_plan.h` are parametric in `g_br` already; only the
   layout differs, in a new `hexkl_attn_q_plan.h`.
6. **A8W4 is A8W8 with the weight kind swapped**: `mm_u8i4` for `mm_u8i8`,
   512 B tiles for 1024 B, `rm_to_wh_i4` for `_i8`, values in [-8, 7] for
   [-127, 127]. One kernel, a `kind` struct with tile bytes, matmul and
   bake function pointers and the int range. Masters (row-major copies the
   decode path reads) stay int8 containers for both kinds in this phase;
   packing the i4 master is listed as follow-up.
7. **Decode (n_q < 5, hd <= 128) is pure HVX on `vrmpy`**: 4 MACs per
   lane per instruction on 8-bit data, against fp16's 1. The registry
   keeps the masters in the layout that makes this one instruction per
   32 cache rows x 4 dims (K^T `[hd/4][rows][4]`) and one per 4 rows x 32
   dims (V `[rows/4][hd][4]`), and bakes the HMX WH tiles from a 32-row
   row-major staging of the column tile being filled. Q is symmetric i8
   for this path (`vrmpy(Vb, Rb)`); softmax and the f32 O accumulator are
   the fp16 decode kernel's.
8. **Seam**: the cache is a DSP handle. `ComputeOps` gains
   `kv_cache_register / append / release` and `sdpa_q_kvcache(handle, ...)`,
   `HtpComputeOps` forwards over FastRPC, `MHACoreLayer` appends this
   step's fp16 rows right where it writes them into the host cache and
   calls attention by handle. The host fp16 cache stays the source of
   truth in this phase (save/load, rollback, CPU fallback all keep
   working), so host memory does not shrink yet; dropping the host copy
   is a `KVCacheManager` follow-up. Selection: `"attention_kv_dtype":
   "q8" | "q4"` in `nntr_config.json` next to `"attention_engine"`.
9. **Validation as before**: every pure-arithmetic part host-unit-tested
   (plan/layout, quantizer references, dequant formulas); kernels gated on
   device by SNR against the f32 reference (A8W8 >= 40 dB, A8W4 >= 30 dB
   to start), resident-vs-raw bit-identity where a raw path exists;
   `-Wall -Werror` on hexagon-clang.

## 3. Design

### 3.1 Registry `hexkl_kv_q` (per attention layer, per kind)
```
masters (DSP heap, int8 containers, zero until written):
  kT4 [n_head_kv][hd/4][max_rows][4]         decode K^T, 4-dim interleaved
  v4  [n_head_kv][max_rows/4][hd][4]          decode V, 4-row interleaved
  s_k [n_head_kv][max_rows] f32, colsum_k [n_head_kv][max_rows] i32
  s_v [n_head_kv][max_rows][hd/32] f32
tiles (DSP heap, WH layout, tile = 1024 B (i8) | 512 B (i4)):
  kt  [n_head_kv][max_rows/32][hd/32]         K^T weight tiles (rows=dims, cols=cache rows)
  v   [n_head_kv][max_rows/32][hd/32]         V weight tiles (rows=cache rows, cols=dims)
staging: [n_head_kv] x (32 rows x hd) int8 for K^T (as [hd][32]) and V (as [32][hd])
```
`append(handle, row0, n_rows, k_f16_rows, v_f16_rows)`: per row, per head:
HVX absmax over hd -> s_k, round-to-nearest to the kind's range, colsum;
per 32-dim group of V -> s_v; scatter into the masters and the staging;
for every column tile touched, re-bake its hd/32 K^T tiles and hd/32 V
tiles with `rm_to_wh_*` from the staging (source in DDR: the fast regime
for HexKL's extraction helpers) through a VTCM scratch tile into the heap
tile array. Cost per token: 2*(hd/32) tile bakes per head, a few us.

### 3.2 Prefill kernel `hexkl_attn_q_prefill`
VTCM regions (`hexkl_attn_q_plan.h`), rt = g_br/64, ct = bc/32, dt = hd/32:
```
q_ah    [rt][dt]         u8 AH tiles (2 KiB), zp/scale per row in q_meta
kt_wh[2][ct][dt]         K^T tiles (kind bytes)      <- DMA from registry
v_wh [2][ct][dt]         V tiles                     <- DMA from registry
acc     [2]              8 KiB int32 readout tiles
s_hf [2][2*rt][ct]       f16 32x32 tiles for the softmax (as fp16 kernel)
p_ah [dt][rt][ct]        u8 AH tiles of P'_d
o_f32   [g_br][hd]       f32
meta    q_scale/q_zp/m/l/a/p_scale[dt] per row
```
Per (kv head, q block): qprep = `hvx_quant_rows_u8_params` +
`hvx_quant_pack_u8_ah` over the gathered f32 Q rows (rows `q*G+g`, scale
fold `log2e/sqrt(hd)` into `q_scale`). Per cache block, the fp16 kernel's
pipeline with three inserted HVX passes:
- **acc -> S**: per output tile (i64, col): dt x `mm_u8i8|i4`, one
  `acc_read_int32` into `acc[..]`, then 64 rows x
  `(acc - zp*colsum_k) * q_scale * s_k` -> f32 -> f16, written as two
  interleaved 32x32 tiles into `s_hf`. Runs on the calling thread right
  after the read (the pool is in the softmax of the previous block).
- **softmax**: unchanged (`softmax_row_pair` over `s_hf`), yields P (f16)
  in place, `a` and `l` per row.
- **P -> P'_d**: per d, per row: `P * s_v[k][d]` (s_v splat per lane from
  the block's cache rows), row max -> `p_scale[r][d]`, u8 RNE, flat 64x32
  store. Pool workers, one (row tile, d) per unit -- it replaces the fp16
  kernel's D-tile build.
- **P.V and O update**: per output tile (i64, d): ct x `mm_*(p_ah[d][i][col],
  v_wh[col][d])`, `acc_read_int32`, then 64 rows x
  `O[r][32d..] = a[r]*O[r][32d..] + p_scale[r][d] * acc[r]` (f32 FMA).
Store: `out = O / l` per real row.

Phase counters as in `hexkl_attn_f16_stats` plus `us_dequant` and
`us_pquant`, so the device test reports where int time goes.

### 3.3 Decode kernel `hvx_attn_decode_q`
Per (query row, head): Q -> i8 symmetric (one scale per unit). Scores in
blocks of 128 cache rows: `acc[k] = sum_g vrmpy(kT4[g][k..k+127], q4[g])`
(hd/4 instructions per 128 rows), dequant `s_q*s_k[k]*acc[k]` to f32 (two
64-lane halves), then the fp16 decode kernel's online softmax per 64 rows
(f16) and f32 O accumulate with `P'` = `P*s_v[k][d]` folded per 32-dim group:
`o[32d..] += vrmpy(v4[k/4][32d..][..], P'4)` over 4 rows at a time.
Gate: A8W8 >= 40 dB vs the f32 reference on 1x1024 and 1x4096.

### 3.4 IDL / skel
```
kv_register_q(kind, max_rows, n_head_kv, head_dim, rout handle)
kv_release_q(handle)
kv_append_q(handle, row0, in seq<uint16> k_rows, in seq<uint16> v_rows)
attn_q_prefill(handle, n_q, cache_from, cache_to, n_head_q, window, br, bc,
               softcap, q_f32, sinks, rout out_f32, rout stats_us)
attn_q_decode (same, no br/bc)
attn_q_step   (handle, row0, in seq<uint16> k_rows, v_rows, n_q, cache_from,
               cache_to, n_head_q, window, softcap, q_f32, sinks,
               rout out_f32, rout stats_us)          -- Q5b: append + attend
probe_acc_i32_layout(rout base, rout stride)          -- device test only
```
kind: 0 = A8W8, 1 = A8W4. `attn_q_step` is what the host uses per layer
per step; it picks the decode kernel for n_q < 5 (hd <= 128) and the
prefill kernel otherwise, and its stats carry append / attend / quant /
stage / bake microseconds.

### 3.5 Host seam
`ComputeOps`: `supports_sdpa_q_kvcache()`, `kv_cache_register(kind, ...)`,
`kv_cache_append(handle, row0, n, k_f16, v_f16)`, `kv_cache_release`,
`sdpa_q_kvcache(handle, append_row0, append_rows, k_rows, v_rows, q, ...)`
(Q5b: the rows to append travel with the attention call). `HtpComputeOps`
forwards. `MHACoreLayer`: on the first step with `attention_kv_dtype` set,
registers one handle per layer (`max_timestep`); `try_quantized_attention`
hands the fp16 cache rows from the first unsynced one up to `cache_to` to
that single call and marks them synced on success; any failure ->
CPU path over the fp16 cache, exactly as today. Batch > 1: one handle per
(layer, batch).

## 4. Phases

| # | Deliverable | Gate |
|---|---|---|
| Q1 | `hexkl_acc_tile` port; `hexkl_kv_q` registry + on-DSP quantizer; IDL register/append/release + a `kv_q_dump` debug entry; host references for the quantizer | host: quantizer/dequant round trip; device: dump == host reference bit-exact, acc layout probe usable |
| Q2 | `hexkl_attn_q_plan.h` + host test; A8W8 / A8W4 prefill kernel; `attn_q_prefill`; CPU model of the arithmetic | done: DSP vs model >= 45 dB on the fp16 suite's shapes (softcap, sink, window, GQA, both kinds); scheme SNR in the status header |
| Q2b | HVX passes at throughput: no scalar VTCM access (metadata in heap, vector gathers, group-major V scales read from DDR, all-vector P' scales), Q quantized once per head chunk, DSP power vote | done: 128x1024 at 1.53x of the fp16 resident path, 3.95 vs 2.58 ms |
| Q3 | (folded into Q2: the kind switch is one struct) mixed K/V kinds if real data asks for K i8 + V i4 | -- |
| Q4 | `hvx_attn_decode_q`: both kinds from the same offset-binary masters, Q and P' as 4 uint8 in a scalar register against `vrmpy(Vub, Rub)`, f32 softmax with all-lanes-equal running state, nothing touches VTCM | done: DSP vs model 69-137 dB; 5.0x / 4.05x faster than the fp16 decode at 1x1024 / 1x4096 |
| Q5 | `ComputeOps::kv_cache_q_{register,append,release}` + `sdpa_q_kvcache`, `HtpComputeOps` forwarding, `MHACoreLayer` `kv_cache_quant` property with a per-batch mirror that re-appends from the first row that may differ (rewind, cache load), `attention_kv_dtype` in nntr_config.json, `Transformer::createAttentionCore` | done: Qwen3-0.6B on device, identical greedy tokens for CPU / HTP fp16 / HTP int8 |
| Q5b | One FastRPC call per layer per step for the quantized path (append + attend), direct WH-tile writes for int8 from a probed layout, HVX quantizer for appended rows | done: int8 append 1.45 ms -> 30 us per row; model-level generation 3181 -> 1136 ms, at parity with the fp16 path (1164) |
| Q6 | Device timing table (prefill 128x1024, 32x4096; decode 1x1024, 1x4096) for f16 / q8 / q4 -- kernel numbers are in the Q2b / Q4 notes; model-level table at 3083 + 32 tokens in the Q6 note, with `NNTR_HTP_ATTN_TRACE` per-call tracing | done: int8 prefill 31.8 s vs CPU 34.6 s vs fp16 125 s; decode 135 / 74 / 275 ms per token |

## 5. Risks
- **Accumulator layout**: the probe may find a non-affine layout on v81;
  then `acc -> S` needs the permutation table (still in VTCM, still no
  vendor copy). The probe result is printed by the device test first.
- **P at 8 bits.** Probabilities below 1/510 of the row max vanish. With
  the per-(row, block) scale this is the prior branch's measured regime
  (5e-3 for A8W8). If the gate fails on long rows, the fallback is P in
  two u8 planes (hi/lo) -- doubling P.V HMX passes -- before giving up on
  the int path for P.V.
- **HVX work per block grows**: two dequant passes and a P quant pass that
  the fp16 kernel does not have. Int8 HMX runs at twice the fp16 MAC rate,
  so prefill should land near fp16; the memory and decode wins are the
  point. Numbers decide, in Q6.
- **DSP heap**: master + tiles is 1.5 B/value (A8W8) or 1.5 B (A8W4 with
  int8 masters) against the host's fp16 2 B/value, with the host copy
  still present. Packing the i4 master (0.5 + 0.5 B) and dropping the host
  copy are the follow-ups that turn this into a memory win.

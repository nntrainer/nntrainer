# W8A8 Optimization Playbook

Techniques developed for the YOLOv7ReIDtiny pose port, written up here because
every one of them is model-agnostic: they apply to any int8 CNN (and several to
any nntrainer model). Final result on device (Android arm64, 320x320 input):
81/87 keypoints (== ONNX Runtime), 12.1 ms (ORT 17.9 ms), 114 MB peak
(ORT 120 MB).

Deployment configuration:

```
NNTR_W8A8=1 NNTR_W8A8_PERCH=1 NNTR_W8A8_PERCH_INPLACE=1 NNTR_MEMORY_PLANNER=v3
```

## 1. Quantization scheme (accuracy)

### 1.1 Per-output-channel int8 weights, FP32 scale, one int32 accumulation
Each conv output channel gets its own scale (`amax/127` over that channel's
weights); the kernel accumulates the WHOLE K reduction in int32 and applies
`a_scale * w_scale[ch]` once at the end. No per-32-block float folding (the
per-block Q8_0 scheme), so there is no intermediate rounding in the reduction.
- Generalizes to: any conv/FC in any model. Per-channel is the standard
  accuracy-preserving weight scheme; the single-final-scale kernel is what
  makes it fast.
- Code: `conv2d_layer.cpp` (`getPerChConvWeight`), kernels
  `nntr_gemm_q8ch_4x4_f32` / `nntr_gemm_q8ch_plainA_f32`.

### 1.2 Shared-offset affine ("asym") int8 activations — the decisive fix
Activations that live in a known asymmetric range (here SiLU outputs, range
`[-0.2785, +inf)`) waste half the int8 levels under symmetric quantization.
Encode `x = (q + 128) * s - C` with a single GLOBAL constant `C` (= |min| of
the activation function) and `s = (amax + C) / 255`, i.e. all 256 levels cover
the range that actually occurs (~2x resolution). Because `C` is a shared
constant:
- no zero-point tensor is needed (the scale slot suffices);
- concat/pool/upsample rescale exactly as before modulo a +-128 shift;
- the GEMM needs NO kernel change: the offset folds into the bias as
  `s_w[ch] * (128*s_a - C) * colsum_w[ch]` (colsum = per-channel sum of the
  int8 weight quants, cached once at conversion).
This took the model from 79/87 to 81/87 (S0 sim margin +0.008 -> +0.054).
- Generalizes to: any network whose int8 edges carry a bounded-below
  activation (SiLU, GELU ~ -0.17, ReLU C=0, ReLU6...). Pick C per activation
  function, not per tensor.
- Code: `conv2d_layer.cpp` (kActOff, ebias fold, pad_q), `concat_layer.cpp`.

### 1.3 Quantize weights ONCE (avoid double quantization)
Per-block Q8_0 file -> dequant -> per-channel requant loses a keypoint versus
quantizing once from FP32 with a per-channel FP32 scale. If the file format
forces fp16 scales, a per-channel-constant-scale Q8_0 file ("pch") lets the
runtime recover the per-channel scheme losslessly enough once the asym margin
exists.
- Generalizes to: any quantized-weight pipeline. Also: keep the ACTIVATION
  scale FP32 end-to-end — fp16-rounding ~50 layer scales accumulated visible
  error here.

### 1.4 FP32 islands: don't quantize what doesn't pay
The stem (in_ch=3: quantizing the image costs accuracy, and a 27-wide im2col
row makes int8 GEMM pointless) and the pose head (most accuracy-sensitive)
stay FP32. Quantize the 90% in the middle; leave the ends alone until the
simulator proves the margin.
- Generalizes to: first conv + task heads of most detection/pose models.

## 2. Compute (speed)

### 2.1 Taps-last weight permutation -> pack-free gathered conv
Filter-order K ([in_ch][kh][kw]) forces an NHWC gather into a byte-wise
transposing scatter plus a q8_0x4 interleave ("gemm.pack"). Permuting each
weight's K to [kh][kw][in_ch] ONCE at load makes every tap a contiguous
`in_ch`-byte copy (a whole kw-run is one memcpy at unit dilation), and the
plain-row activation feeds the plain-A kernel directly — the interleave pass
disappears. Integer accumulation is order-exact, so results are bit-identical
(validated byte-for-byte under qemu).
- Effect here: head-final conv 2.85 -> 0.98 ms; ~3.5 ms/iter of gemm.pack
  became ~0.4 ms of gemm.gather across the net.
- Generalizes to: every NHWC int8 conv with k > 1x1 in any framework that
  im2col-gathers. The key insight — int8 GEMMs may permute K freely — has no
  FP analogue (FP addition is order-sensitive); exploit it.
- Code: `conv_indirect.h` (`gather_conv_act_rows_tapslast`),
  `ggml_interface.cpp` (`taps_last` path), `conv2d_layer.cpp` (perm build).

### 2.2 Pack-free 1x1 convs (identity gather)
A 1x1 stride-1 NHWC conv's im2col is the identity: the input already IS the
[M, K] int8 matrix. Detect it and call the plain-A kernel on the input
directly — no gather, no pack, no staging buffer.
- Code: `ggml_interface.cpp` (`conv_gather_is_identity`).

### 2.3 Fused epilogue: bias + SiLU + amax in one pass
The conv epilogue does bias-add, activation, and (for an int8 output) the
next scale's amax in a single NEON pass over each output row instead of three
separate passes. Vectorized SiLU uses a Cephes expf (see 2.4).
- Code: `conv2d_layer.cpp` (`convBiasActRow`).

### 2.4 NEON exact SiLU via Cephes expf
`x / (1 + e^-x)` with a 6th-order polynomial expf (~1e-6 rel error) + two
Newton reciprocal steps. Not a LUT approximation — accuracy-neutral by
construction (validated 1.9e-6 max abs against double-exp). Reused for the
pose head's 100k-element GAU SiLU (head 2.0 -> 1.9 ms; the win compounds on
models with bigger MLPs).
- Generalizes to: any FP32 sigmoid/SiLU/GELU epilogue on ARM.
- Code: `conv2d_layer.cpp` (`nntr_vexpq_f32`), `rtmcc_head.cpp`.

### 2.5 NEON int8 data movement ops
- Concat rescale `q' = round((q+128)*mult) - 128`: widen s8->s32, FRINTA
  (ties-away == std::round), saturating narrows; 16 ch/iteration. Bit-exact
  vs scalar over all 256 codes.
- Max-pool on int8 NHWC: 16 channels per `vmaxq_s8` register, bounds hoisted.
- Quantize FP32->affine int8: `vcvtaq_s32_f32` (FCVTAS == std::round) +
  saturating narrows, chunk-parallel with a two-pass amax.
- Generalizes to: every int8-resident elementwise/movement op. Rule of thumb:
  if an op touches every int8 byte and has no NEON path, it will show up in
  the profile.
- Code: `concat_layer.cpp`, `pooling2d_layer.cpp`, `conv2d_layer.cpp`
  (`convQuantAffine`, `convAbsMaxF32`).

### 2.6 Persistent scratch, no zeroing
Per-forward buffers (GEMM output, quantized input, gather rows) are
`static thread_local`, grown monotonically, and never cleared when every byte
is overwritten anyway. Removes malloc/page-fault churn that showed up as
unattributed per-conv time.

## 3. Memory

### 3.1 OptimizedV3Planner (runtime-selectable)
The single biggest win: the default V1 activation planner left the pool at
2.4x the theoretical minimum (39 MB vs 16.2 MB here). V3 packs near-minimum
(offline: overlap-free over 14k random request sets, ~33% of V1's pool size,
never worse). `NNTR_MEMORY_PLANNER=v3|v2|v1|basic` selects it at runtime —
planning only relocates tensors, values are unchanged.
- Effect here: peak RSS 134 -> 114 MB.
- Generalizes to: EVERY nntrainer inference deployment. Check
  `tensor_pool.size()` against `minMemoryRequirement()` first — if they're
  far apart, this is free memory.
- Code: `manager.cpp` (`finalizeTensorPool`).

### 3.2 Zero-copy in-place weight requant (`NNTR_W8A8_PERCH_INPLACE=1`)
When the runtime repack has the same layout/size as the source weight
(out_ch%4==0, CRS%32==0 q8_0x4), requantize in place into the source buffer
instead of keeping a second ~25 MB copy. Safe because each 4-row group is
fully consumed before its blocks are overwritten.
- Code: `conv2d_layer.cpp` (`qs_ext` / `qs_data()`).

### 3.3 Streamed weight conversion
Convert per 4-output-channel group with a 4xCRS scratch instead of
materializing whole-tensor FP32 + int8 intermediates (~5x the weight's int8
size, ~16 MB transient on the largest conv). Bit-identical output.

## 4. Methodology (what actually made this work)

1. **Simulate before device.** A python fake-quant simulator (S0) that mirrors
   the exact runtime scheme, reporting per-keypoint MARGINS, not just counts.
   The symmetric->affine decision was made on margin (+0.008 vs +0.054), not
   on a device run.
2. **Bit-exact kernel validation under qemu.** Every kernel/layout change
   (taps-last, in-place, streaming, concat NEON) shipped only after a
   byte-identical comparison against the reference path, on x86 or
   qemu-aarch64, including tail/edge shapes (M%4, N=87, CRS%32!=0).
3. **Measure, don't guess.** Two memory hypotheses were wrong (FP32W
   residency, conversion spike) before a one-off breakdown
   (pool sizes + scratch high-water) found the planner gap in one run.
   The env-gated layer profiler (`NNTR_LAYER_PROFILE=1`) drove every speed
   decision.
4. **Escape hatches.** Each optimization keeps a revert switch
   (`NNTR_W8A8_PACKA=1`, `NNTR_W8A8_SYM=1`, `NNTR_MEMORY_PLANNER=v1`) so any
   device regression bisects in minutes without a rebuild.

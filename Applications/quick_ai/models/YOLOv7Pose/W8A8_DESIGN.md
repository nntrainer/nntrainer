# W8A8 Design — int8-resident activations for YOLOv7Pose

Status: **design (pre-implementation)**. Target branch:
`claude/yolov7reldtiny-pose-support-k1q9wi`.

## 0. Goal and the accuracy gate

Run the pose model with **Q8_0 weights + int8 activations between layers**
(W8A8), at W8A32-level accuracy:

- **Accuracy gate: 81/87 visible keypoints (thr=0.5) on `input_320.bin`** —
  the W8A32 result, which equals the ONNX-Runtime int8 model. Every stage
  below re-measures this gate; a stage that loses it does not proceed.
- Speed reference points (same device, comparable thermal state):
  W8A32 ≈ 76 ms, ORT int8 ≈ 42 ms. W8A8 removes the consumer-side
  re-quantization and shrinks activation traffic 4x vs FP32, targeting the
  ORT band.

Why not W8A16: FP16 *storage* between ~50 conv layers accumulates rounding
error and collapses keypoint confidence (38/87, positions right / scores
low). The failure is the intermediate format itself, so the fix is
ORT's structure — int8 storage with real (FP32) scales, FP32 arithmetic
everywhere else. W8A8 is the accuracy-safe replacement for W8A16, not an
optimization of it.

## 1. Numeric scheme

| Item | Choice | Rationale |
|---|---|---|
| Weights | Q8_0 per-32-block (unchanged, same file) | proven 81/87 |
| Activations | **QINT8, per-tensor symmetric dynamic scale (FP32)** | see below |
| Conv accumulate | int8×int8 → int32 (SMMLA) — exact | existing kernel |
| Conv epilogue | int32 → FP32 (`act_scale × w_block_scale`), + bias, SiLU in FP32, then quantize to the output's int8 | FP32 math at every nonlinearity |
| Scale handoff | producer writes 1 FP32 scale into the tensor's inline scale slot; consumer reads it | CharTensor layout |

Per-tensor (not per-block) activation scale, because:

- **maxpool** compares raw int8 — exact iff every element shares one scale.
  Per-block scales (Q8_0-style) would force dequant per comparison.
- **concat** is memcpy — needs a common scale across joined inputs
  (§3 handles reconciliation).
- ORT reaches 81/87 with per-tensor activation scales (calibrated static);
  ours are **dynamic per-forward**, which is at least as tight.
- A per-tensor-scale int8 row is exactly a `block_q8_0` stream whose blocks
  all carry `d = scale` → the existing SMMLA GEMM
  (`nntr_gemm_q8_0_q8_0_4x4_f32`) consumes it **unchanged**; the conv
  gather becomes a byte shuffle + constant-`d` fill (no amax scan, no
  rounding math on the consumer side).

Symmetric (no zero-point) keeps the SMMLA kernel free of zero-point
correction terms. Stage 0 validates this choice before any code.

## 2. Graph partitioning

From the layer survey (concat/maxpool-max/upsample-nearest are pure
copy/compare ops; `rtmcc_head` is a hard FP32 boundary; conv handles both
directions of dtype change naturally):

- **int8-resident region**: every edge whose producer is a Q8_0 conv and
  whose consumers are all int8-capable (Q8_0 conv / concat / max-pool /
  nearest-upsample). With today's eligibility this covers blocks.2 →
  ends.0 (backbone tail, SPP, most of the neck).
- **FP32 island (front)**: input → stem (`blocks.0.0`, 3-ch) → blocks.1
  (five FP32 convs with 48-ch shapes). Also `feature_up.1.elan`'s FP32
  bottleneck convs fragment the neck.
- **FP32 island (tail)**: `head.final_layer` (out_ch=87, FP32) +
  `rtmcc_head` (FP32-only by assertion).
- **No standalone Q/DQ layers needed**: a Q8_0 conv entering the region
  takes FP32 in / int8 out (today's path + quantize epilogue); a conv
  leaving the region takes int8 in / FP32 out. Conv is the universal
  boundary op. Non-conv layers never change dtype.
- **fp-conv padding (out_ch→%32 zero-pad + CRS→%32 zero-pad via the
  gather `dst_stride`)** is the enabler that dissolves the FP32 islands
  in blocks.1 / feature_up.1 and later the head conv — it turns "int8
  archipelago" into one contiguous region. It is scheduled as its own
  stage (S4), after residency works.

Edge dtypes are decided **statically in the app graph builder**
(`yolov7_pose_graph.h` already knows per-conv quantization eligibility);
the framework only provides mechanics. This keeps generic-framework risk
near zero.

## 3. Framework changes (from the infra survey)

1. **TensorPool QScheme**: `TensorPool::request` hardcodes
   `QScheme::PER_CHANNEL_AFFINE` (weight convention, `scale_size()==width()`).
   Activation tensors need `PER_TENSOR_AFFINE` (1 scale). Plumb the scheme
   through the request (or default per-tensor for non-weight groups).
2. **conv2d QINT8 output**: fused epilogue — GEMM writes FP32 to scratch;
   one pass applies bias+SiLU while reducing `amax`; one pass quantizes to
   the output CharTensor and writes `scale = amax/127` to its inline slot.
3. **conv2d QINT8 input**: read `getScale()[0]`, gather int8 NHWC rows
   (channel-innermost memcpy per kernel tap) into `block_q8_0x4` layout
   with constant `d` → existing SMMLA GEMM. No float quantize math.
4. **concat / pooling2d(max) / upsample2d(nearest) QINT8 branches**:
   1-byte memcpy / compare / replicate. Concat reconciles scales:
   `out_scale = max(in_scales)`, each input rescaled int8→int8 by
   `s_i/s_out` (fixed-point multiply) during its copy. Max-pool and
   upsample preserve the input scale exactly.
5. **Preset**: `YOLO_TENSOR_TYPE=w8a8` in the app; per-edge dtype set by
   the graph builder. Batch is 1 throughout (CharTensor inline-scale slice
   semantics are only exercised with batch=1).

## 4. Staged plan — each stage ends at the accuracy gate

- **S0 — numeric validation by simulation (no framework code).**
  In the PyTorch reference (`make_reference.py` infra), fake-quantize
  every would-be-int8 edge to per-tensor symmetric int8 — **including the
  concat max-scale rescale** — and measure keypoints.
  Gate: 81/87 (±1). If it fails, revisit the scheme (asymmetric, per-edge
  exceptions) *before* writing any C++.
- **S1 — kernel prototype (x86).** int8-activation gather + constant-`d`
  x4 packing feeding `nntr_gemm_q8_0_q8_0_4x4_f32`; unit test against the
  FP32-gather path (error ≤ per-tensor-vs-per-block quantization delta
  predicted by S0).
- **S2 — one int8 edge end-to-end (x86).** Env-gated: a single conv→conv
  edge carries QINT8 through the pool (scale handoff live). Parity vs
  W8A32 within S0-predicted delta.
- **S3 — region rollout + device.** QINT8 branches in concat/mp/upsample,
  blocks.2→ends.0 resident, `w8a8` preset. Device: **81/87 gate** + speed.
- **S4 — fp-conv padding integration.** out_ch/CRS zero-pad path (gather
  `dst_stride` groundwork already in tree) makes blocks.1 / feature_up.1 /
  stem / head-conv int8 → near-full residency. Gate re-run.
- **S5 — cleanup/tuning.** Thread sweep, retire the broken w8a16 preset.

## 5. Risks

- **R1 — per-tensor dynamic scale accuracy.** The one real numeric risk;
  S0 settles it for ~zero cost. (W8A32 uses per-block activation scales;
  per-tensor is coarser. ORT's 81/87 with per-tensor static is strong
  evidence, not proof.)
- **R2 — concat rescale resolution loss** when branch scales differ
  widely. Modeled explicitly in S0.
- **R3 — pool/planner behavior for QINT8 activations** (sizes honored via
  `getMemoryBytes`; the PER_CHANNEL hardcode is the known fix; batch=1
  sidesteps slice-scale semantics).
- **R4 — stem/head stay FP32**, so the accuracy-critical extremes carry
  no new quantization risk until S4 (which re-runs the gate).

## 6. Expected effect (order-of-magnitude)

- Consumer-side re-quantization (amax+round per gathered row, per conv)
  disappears; activation traffic 4x below FP32; concat/pool/upsample move
  1-byte data.
- Estimate: ~76 ms → 55–60 ms at S3, → ~45 ms at S4, converging on the
  ORT band (~42 ms measured same-device). Estimates are secondary to the
  accuracy gate at every step.

## 7. Memory optimizations (opt-in)

- `NNTR_W8A8_PERCH_INPLACE=1` — by default the per-channel path
  (`getPerChConvWeight`) keeps a second, ~equal-sized copy of every conv's
  int8 weight in the kernel's `block_q8_0x4` layout, coexisting with the
  source Q8_0 weight in the pool (~25 MB duplicate for this model). When the
  source is already `block_q8_0x4` with `out_ch % 4 == 0` and `CRS % 32 == 0`,
  the repacked stream is byte-for-byte the same layout and size, so this flag
  requantizes **in place** into the source buffer and shares it zero-copy
  instead of allocating the duplicate. The dequantize pass has fully consumed
  the source before the in-place write, and the packed int8/scale/colsum are
  **numerically identical** to the owned-buffer path (validated byte-exact on
  x86 for both per-channel and per-block Q8_0). No weight-file change; the
  default path is untouched. Convs that don't meet the layout constraint
  (e.g. the `out_ch = 87` head-final) fall back to the owned buffer.

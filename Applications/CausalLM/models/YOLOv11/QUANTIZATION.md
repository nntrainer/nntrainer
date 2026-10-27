# YOLOv11m Q4_0 quantization & device optimization (baseline v5)

This documents the framework-native Q4_0 conv quantization and the kernel
work that landed in `yolov11-device-opt-baselinev5`. The earlier one-off numpy
quantizer (`PyTorch/quantize_q4_0_conv.py`) is **retired** — quantization now
goes through nntrainer's own save path, the same machinery LLMs use.

## Result (S23, cat input, OMP=4, same thermal state)

| | latency | RSS | detections |
|---|---|---|---|
| FP32 (BN-fused) | 4355 ms | 353 MB | 0.9387 / 0.9247 |
| **all-conv Q4_0 + depthwise backend op** | **3540 ms (−18.7%)** | 350 MB | 0.9275 / 0.8906 |

(Speedup ranges −18 to −23% depending on device thermal state. Detections
preserved; memory ~flat — see "Known limits".)

## Two-step workflow (general, no model-specific script)

1. **Convert** PyTorch → FP32 nntrainer safetensors (BN folded into conv at
   convert time):
   ```
   python PyTorch/convert_weights.py --weights v11m.pt --out res/yolov11m_fused.safetensors
   ```
2. **Quantize** with the framework quantizer (the YOLO inference binary in
   quantize mode). Produces an ISA-specific Q4_0 file; x86 and arm are
   different repack layouts (q4_0x8 vs q4_0x4) and are managed as separate
   files:
   ```
   # arm (device):
   YOLO_QUANTIZE_OUT=res/yolov11m_fw_q40_arm.safetensors YOLO_QUANTIZE_ISA=arm \
     YOLO_WEIGHTS=yolov11m_fused.safetensors yolov11_infer res res/input_cat.bin
   # x86 (host verify):
   YOLO_QUANTIZE_OUT=res/yolov11m_fw_q40_x86.safetensors YOLO_QUANTIZE_ISA=x86 \
     YOLO_WEIGHTS=yolov11m_fused.safetensors yolov11_infer res res/input_cat.bin
   ```
3. **Run** quantized (runtime builds eligible convs as Q4_0 and loads the file):
   ```
   YOLO_CONV_Q40=1 YOLO_WEIGHTS=yolov11m_fw_q40_arm.safetensors \
     yolov11_infer res res/input_cat.bin
   ```

## Where the quantization lives (framework, not YOLO-specific)

- `nntrainer/layers/conv2d_layer.cpp` `Conv2DLayer::save()`: quantizes a conv
  FP32 filter `[out_ch, in_ch, kh, kw]` (already row-major `[out_ch, CRS]`,
  CRS = in_ch·kh·kw) straight through `quantize_q4_0` + `repack_q4_0(ISA)` with
  **no transpose** (the FC base path transposes because its weight is `[K,N]`).
  Bias / 32-misaligned filters stay FP32. Stored matmul shape `[1,1,CRS,out_ch]`.
- `nntrainer/models/neuralnet.cpp`: the safetensors save accounting +
  `nntr_shape` were FC-shaped. `quantRowsCols()` now derives `(N,K)` per weight
  (batch>1 → conv filter: N=out_ch, K=CRS; else FC: N=width, K=height) so a 1×1
  conv (height==1, not a bias) quantizes and the header shape is `[1,1,K,N]`.
- Eligibility (must match on both sides): `out_ch>1 && out_ch%32==0 &&
  (in_ch·kh·kw)%32==0`. The graph builders own this gate
  (`yolov11_graph.h`); `quantConvSink()` records eligible conv names during the
  build to form the save dtype map, so the quantize-time set always equals the
  runtime Q4_0 set.

## Kernel: depthwise in the backend (not the layer)

`depthwise_conv2d_fp32` is a CPU-backend compute op
(`ComputeOps`→`CpuComputeOps`→fallback/arm/x86, link-time dispatch), parallel
over batch×channels via `ThreadManager::parallel_for`. `Conv2DLayer::forwarding`
calls `getComputeOps()->depthwise_conv2d_fp32(...)` for true depthwise
(groups==channels) instead of the old in-layer loop. Saves ~350-400 ms vs the
generic grouped path (which dispatched 256-512 tiny im2col+GEMM calls/layer).
`NNTR_NO_DW_FASTPATH=1` falls back to the generic path. arm/x86 currently
delegate to the scalar fallback (TODO: NEON/AVX). Depthwise is **not** Q4_0:
per-channel K = kh·kw = 9 < the 32-block, and the weights are ~83 KB total.

## 1×1 / 3×3 conv Q4_0 (matmul path)

`conv2d_layer.cpp` forwarding: a quantized groups==1 conv runs as a single GEMM.
1×1 stride-1 skips im2col (identity); k>1 uses im2col → `[OH*OW, CRS]` (already
the activation layout — **no transpose**, the bug that caused the deep-3×3
heap-overflow SIGSEGV) → `dotQnK`. See baselinev3/v4 commits.

## Known limits / next session

- **Memory ~flat** (353→350 MB): weights drop ~65 MB but the im2col `col`
  buffer + GEMM temporaries offset it, and whether Q4_0 weights stay packed in
  RAM is unconfirmed. Investigate for a real memory win.
- **Quantizer entry point**: v5 quantizes via the YOLO binary's `--quantize`
  mode (reusing the framework save). Full unification — registering YOLOv11 as
  a factory/config model so `nntr_quantize <dir>` handles it like an LLM
  (option B) — is deferred. The framework quantization core is already done, so
  B is mostly factory/config plumbing.
- **ONNX comparison**: not yet measured on device; cannot claim "beats ONNX".
- arm/x86 depthwise NEON/AVX specialization.

## Branches

- `yolov11-device-opt-baselinev5` — this state (framework quant + depthwise
  backend op), origin.
- v4 = +depthwise inline (−22.7%); v3 = 3×3 Q4_0 crash fix; v2 = 1×1 Q4_0.

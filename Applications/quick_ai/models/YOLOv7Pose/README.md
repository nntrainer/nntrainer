# YOLOv7ReIDtiny — Pose + ReID on nntrainer

Port of the `YOLOv7ReIDtiny` model (RTMPose-style SimCC pose head + ReID
embedding head on a YOLOv7-tiny CSP backbone with a dual FPN neck) onto
nntrainer's functional graph API.

- **Backbone**: YOLOv7-tiny CSP, `widen_factor = 1.5`. block 1 downsamples with
  a strided conv (`base`); blocks 2-4 downsample with a parameter-free maxpool
  and the ELAN doubles the channels (nodes 48 / 96 / 192 / 384 / 768).
- **Neck** (`backbone.features`): SPPCSPCTiny(768→384) + 2× upsample + 2×
  downsample (each a `base` sample block + ELAN, bottleneck = out/2), single
  stride-32 end (768 ch @ 10×10).
- **Pose head** (`head`): 7×7 conv → flatten → ScaleNorm+Linear MLP → GAU
  (`rtmcc_gau` custom layer) → `cls_x` / `cls_y` SimCC classifiers → output
  `[1, 2·87, 640]` (`cls_x` rows stacked over `cls_y` rows).
- **ReID head** (optional, `head_feat`): a second neck `features_feat` +
  global-avg-pool → Linear → `[1, 128]`. Enable with `YOLO_WITH_REID=1`
  (a merged pose+ReID checkpoint); `pose_base_v311.pt` is pose-only.

Input `1×3×320×320` (NCHW FP32). Pose decode is RTMPose SimCC: per keypoint,
`argmax` over 640 bins for x and y, `score = min(max_x, max_y)`, coordinates
divided by 2 (simcc split ratio).

## Files

| Path | Purpose |
|------|---------|
| `jni/yolov7_pose_graph.h` | Inline graph builders (backbone / neck / heads) |
| `jni/main.cpp` | `yolov7_pose_infer` — build, load, run, decode |
| `yolov7_pose.h` | `quick_ai::Model` wrapper for `nntr_quantize` |
| `../../layers/rtmcc_gau.{h,cpp}` | GAU (RTMCCBlock) custom layer |
| `../../res/yolov7_pose/` | reference model, weight converter, verify scripts |

## Build stages

Three precision stages, verified in order:

| Stage | Preset (`YOLO_TENSOR_TYPE`) | Weights | Activations | Layout |
|-------|-----|---------|-------------|--------|
| 1 | `w32a32` | FP32 | FP32 | NCHW |
| 2 | `w8a32` | Q8_0 | FP32 | NHWC |
| 3 (future) | `w8a16` | Q8_0 | FP16 | NHWC |

Stages 1 and 2 are implemented here. Channel-last (NHWC) is used from stage 2
on so every layer runs channel-last (no per-conv transposes).

## End-to-end (x86)

```bash
# 0. build (transformer feature gates the whole quick_ai tree)
meson setup build -Denable-transformer=true -Denable-app=true
ninja -C build Applications/quick_ai/models/YOLOv7Pose/jni/yolov7_pose_infer \
               Applications/quick_ai/nntr_quantize

cd Applications/quick_ai/res/yolov7_pose

# 1. FP32 weights: PyTorch .pt -> nntrainer safetensors
#    Accepts either a state_dict or a full torch.save(model) object; if the
#    checkpoint pickles the original training repo's classes (e.g. a missing
#    `models` package), the converter shims them to recover the state_dict.
#    Use --inspect first to print the checkpoint's raw keys/shapes and confirm
#    they match the reconstructed model.
python3 weight_converter.py --weights pose_base_v311.pt --inspect | head
python3 weight_converter.py --weights pose_base_v311.pt \
        --output /path/res/yolov7_pose.safetensors

# 1b. an input tensor (NCHW FP32, 1x3x320x320) from an image:
python3 make_input.py --image sample.jpg --out /path/res
#     (make_reference.py also writes input_320.bin alongside a PyTorch ref)

# 1c. stage 1 (W32A32) inference  (pass an ABSOLUTE input path)
YOLO_TENSOR_TYPE=w32a32 YOLO_WEIGHTS=/path/res/yolov7_pose.safetensors \
  ../../../../build/Applications/quick_ai/models/YOLOv7Pose/jni/yolov7_pose_infer \
  /path/res /path/res/input_320.bin

# 2. Q8_0 weights via nntr_quantize (uses nntrainer's own packing)
../../../../build/Applications/quick_ai/nntr_quantize /path/res \
  --conv_dtype Q8_0 --output_format safetensors \
  --output_bin yolov7_pose_q8_0.safetensors -o /path/res

# 2b. stage 2 (W8A32) inference  -- ARM/Android only, see note below
YOLO_TENSOR_TYPE=w8a32 YOLO_WEIGHTS=/path/res/yolov7_pose_q8_0.safetensors \
  ./yolov7_pose_infer /path/res /path/res/input_320.bin

# 2c. (x86) check Q8_0-weight ACCURACY without the ARM kernel: store
#     dequantized-Q8_0 conv weights as FP32 and run w32a32. These numerics are
#     exactly W8A32's (weights quantized, activations FP32).
python3 weight_converter.py --weights pose_base_v311.pt --sim-q8-conv \
        --output /path/res/yolov7_pose_w8sim.safetensors
YOLO_TENSOR_TYPE=w32a32 YOLO_WEIGHTS=/path/res/yolov7_pose_w8sim.safetensors \
  ./yolov7_pose_infer /path/res /path/res/input_320.bin
```

Every run prints a timing/memory summary:

```
================[ YOLOv7 Pose with NNTrainer ]================
compile:   9.3 ms
load:      49.8 ms
inference: 12.7 ms (avg over 10 iters)
keypoints: 87/87 visible
peak memory: 125196 KB
=============================================================
[e2e time]: 6579 ms
Max Resident Set Size: 125196 KB
```

(`YOLO_BENCH_ITERS=N` averages the inference time over N runs.)

> **Q8_0 runtime is ARM-only.** The quantized conv uses the NEON indirect-conv
> kernel (dotprod / i8mm); x86 has no NHWC quantized-conv fallback, so the real
> `w8a32` inference runs on the Android target. On x86, `--sim-q8-conv` (step
> 2c) reproduces the **W8A32 accuracy** (identical numerics), and `nntr_quantize`
> still produces/inspects the on-device Q8_0 safetensors; runtime speed is
> measured on device.

### Parity check

`make_reference.py` dumps the PyTorch reference (`ref_pose.bin`, `ref_reid.bin`)
and the input. Dump the nntrainer raw outputs and compare:

```bash
POSE_DUMP=/path/res/nn YOLO_TENSOR_TYPE=w32a32 \
  YOLO_WEIGHTS=/path/res/yolov7_pose.safetensors \
  yolov7_pose_infer /path/res input_320.bin
python3 verify_parity.py \
  --ref-pose /path/res/ref_pose.bin --ref-reid /path/res/ref_reid.bin \
  --out-pose /path/res/nn_pose.bin  --out-reid /path/res/nn_reid.bin
```

## Android (arm64-v8a)

```bash
export ANDROID_NDK=/path/to/android-ndk
cd Applications/quick_ai/models/YOLOv7Pose
./build_android.sh                 # libnntrainer + yolov7_pose_infer
./install_android.sh /path/res     # push binary + libs + weights + input
# then run on device (see script output)
```

## Notes

- **Q8_0 conv kernel (W8A16)**: the Q8_0-weight FP16 conv defaults to the
  interleaved `q8_0x4` SMMLA kernel (`nntr_gemm_q8_0_q8_0_4x4_fp16`) — verified
  correct on-device and ~18% faster end-to-end (identical keypoint output) than
  the plain kernel. The plain reference/debug path (`nntr_gemm_q8_0_q8_0_fp16`,
  which de-interleaves the saved `q8_0x4` weight on the fly) is opt-in via
  `NNTR_Q8_CONV_PLAIN=1`. Both consume the identical weight file, so switching
  is a runtime env toggle only.
- **Checkpoint formats**: `weight_converter.py` maps keys model-free and
  accepts a state_dict *or* a full `torch.save(model)` object (fused or
  unfused). If unpickling needs the training repo's `models` package it is
  shimmed automatically. Use `--inspect` to print the checkpoint's raw
  keys+shapes.
- **Reference model**: `models_pose/pose_ref.py` reproduces the pose-only
  checkpoint's exact module hierarchy (verified key+shape match) and drives
  `make_reference.py` for parity. Random-weight parity vs PyTorch: pose logits
  agree to ~0.1 % of range, SimCC arg-max keypoints match 173–174/174 (the odd
  near-tie flip is a random-weight artifact; trained weights are unimodal).

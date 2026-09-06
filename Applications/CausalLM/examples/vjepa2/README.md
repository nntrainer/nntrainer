# V-JEPA-2 ViT-B/16 encoder on nntrainer (Android / x86)

Runnable example for the V-JEPA-2.1 **ViT-B/16** video encoder ported into the
CausalLM app. It encodes a **24-frame × 256×256** clip and emits the per-patch
hidden states, matching the PyTorch reference to ~0.99 cosine on-device.

## The model

`vjepa2_1_vit_base_384` — embed_dim 768, depth 12, heads 12, head_dim 64,
patch 16, tubelet 2, GELU MLP, LayerNorm eps 1e-6, **3D axial RoPE** (applied in
a custom `vjepa_rope` layer before attention; `mha_core` runs with rope
disabled), modality embedding, no CLS / no registers.

For the **24 frame × 256×256** input the token grid is

```
grid_t = 24 / tubelet(2)        = 12
grid_h = 256 / patch(16)        = 16
grid_w = 256 / patch(16)        = 16
NUM_PATCHES = 12 × 16 × 16      = 3072 tokens   (each 768-d)
```

Input tensor: raw FP32 `[24, 3, 256, 256]` = 18,874,368 bytes, host-patchified
to `[1, 1, 3072, 1536]` then `patch_embed` (FC) → 12 transformer blocks → final
LayerNorm.

## Files

| File | Purpose |
| --- | --- |
| `run_device.sh` | push libs + model to an Android device, run, report latency/RAM/cosine |
| `run_x86.sh` | run the host build (FP32) as a correctness sanity gate |
| `compare_cosine.py` | cosine / max-abs-diff of the token-0 dump vs `ref_output.npy` |

Model assets live in `../../res/vjepa2/vjepa2_24f256_q4arm/`:
`config.json`, `nntr_config.json`, `nntr_vjepa2_vitb_q40_arm.bin` (ARM-repacked
Q4_0 weights, 47 MB), `input_video.bin`. The torch reference
(`ref_output.npy`, `input_video.bin`) is produced by the reference script under
`vjepa2_ref/` (point `VJEPA_REF` at it).

## Build

```bash
# host (x86) — sanity gate
meson setup build -Denable-app=true -Denable-test=false \
  -Denable-tflite-backbone=false -Denable-tflite-interpreter=false \
  -Denable-transformer=true
ninja -C build

# Android arm64 (device) — builds libnntrainer.so (ARM) + the CausalLM app
cd Applications/CausalLM
export ANDROID_NDK=/path/to/android-ndk
./build_android.sh            # add --cache to skip the nntrainer core rebuild
```

The device path is built `ENABLE_FP16=1` (`armv8.2-a+fp16+dotprod+i8mm`); the
Q4_0 weights must be **ARM-repacked** (`nntr_quantize --fc_dtype Q4_0 --isa ARM`)
for the ggml 4×8 NEON kernel.

## Run

```bash
# device (auto-detects first adb device; pass a serial + thread count to override)
examples/vjepa2/run_device.sh <DEVICE_SERIAL> 8

# host
examples/vjepa2/run_x86.sh
```

`nntrainer_causallm <model_dir> <input_video.bin>` writes the token-0 hidden
state (768 FP32) to `<input_video.bin>.nntr_out.bin` and prints the first 10
values + `[e2e time]` + peak RSS.

## Measured results

**Device: Galaxy S26 Ultra (SM-S948U, `m3q`), 24f × 256², 3072 tokens.**

| Activation | threads | e2e (cool) | peak RAM | cosine vs torch |
| --- | --- | --- | --- | --- |
| Q4_0-FP16 | 8 | **~2.3 s** | 369 MB | **0.9898** |
| Q4_0-FP16 | 4 | ~3.8 s | 365 MB | 0.9898 |
| Q4_0-FP16 | 1 | ~9.4 s | 359 MB | 0.9898 |
| Q4_0-FP32 | 8 | ~2.6 s | 431 MB | 0.9906 |
| Q4_0-FP32 | 4 | ~4.0 s | 426 MB | 0.9906 |

(Re-verified 2026-06-10 on the current build: token-0 output bit-identical to the
cosine-reference run — `[0]=0.06958` FP16 / `[0]=0.07917` FP32, NaN-free.)

FP16 activation is a bit faster and ~60 MB lighter; FP32 activation is a hair
more accurate. Sustained back-to-back runs thermally throttle (timings drift
+20–40 %); numbers above are best-of cool-start. The optimized path uses 2D
tiled flash attention (online softmax), FMLAL-widening QK (block-0 logits reach
~457 k, past FP16's 65 504 ceiling), and parallel custom GELU/LayerNorm.

**Host (x86), correctness gate:** FP32 weights cosine **1.0000** (max abs diff
0.011), Q4_0 weights cosine **0.9914**. x86 uses the reference attention path
(the GEMM/flash path is ARM-FP16 only).

## Note on RoPE

V-JEPA applies its 3D axial RoPE in the custom `vjepa_rope` layer **before**
`mha_core`, so `mha_core`'s own rotary embedding is disabled via
`use_rope=false`. (Upstream gates the internal RoPE on `use_rope` rather than
the older `rope_theta > 0` check — without `use_rope=false` the internal RoPE
runs with theta=0 and produces NaN.)

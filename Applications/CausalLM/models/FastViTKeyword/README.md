# FastViTKeyword — nntrainer Inference Port

Port of the **FastViTKeyword** model from
[deep-vision-models](https://github.sec.samsung.net/RS8-ARGlass-SW/deep-vision-models)
(branch `dev/keyword/train-inference`) to nntrainer.

## Model Architecture

**FastViTKeyword** is a 507-class multi-label keyword classifier:

- **Backbone**: FastViT-S12 (timm `fastvit_sa12.apple_dist_in1k`)
  - 3-layer stem (Conv+GELU)
  - Stage 0: 2× RepMixerBlock (64ch, 80×80)
  - Stage 1: downsample + 2× RepMixerBlock (128ch, 40×40)
  - Stage 2: downsample + 6× RepMixerBlock (256ch, 20×20)
  - Stage 3: downsample + pos_emb + 2× AttentionBlock (512ch, 10×10, 16 heads, head_dim=32)
  - Final conv: dw 3×3 (512→1024) + SE(1024, rd=64) + GELU
- **Head**: GlobalAvgPool → Linear(1024,1024) + LayerNorm + GELU → Linear(1024,1019)
- **Output**: split → logits[507], features[512]

All RepConv/BN pairs are **fused** (reparameterized) at inference time, so the
nntrainer graph uses single biased convolutions.

## Files

```
FastViTKeyword/
├── jni/
│   ├── fastvit_keyword_graph.h     # Graph block builders (stem, stages, head)
│   ├── fastvit_attention_layer.h   # Custom multi-head attention layer (header)
│   ├── fastvit_attention_layer.cpp # Custom multi-head attention layer (impl)
│   ├── main.cpp                    # Model build + inference + verification
│   └── meson.build                 # Build definition
├── PyTorch/
│   ├── extract_reference.py        # Extract PyTorch reference outputs (.bin)
│   └── convert_weights.py          # Convert .pth → nntrainer safetensors
├── res/                            # Weights + reference bins (generated)
├── run_nntrainer.sh                # Run script
└── README.md
```

## Build

```bash
# From nntrainer repo root
meson setup build -Denable-transformer=true -Denable-app=true
ninja -C build
```

This produces `build/fastvit_keyword_infer`.

## Weight Conversion & Reference Extraction

```bash
# Set path to deep-vision-models repo
export DEEP_VISION_MODELS_PATH=/home/seungbaek/projects/deep-vision-models

# Convert weights (.pth → safetensors)
python PyTorch/convert_weights.py --weights /path/to/ckpt.pth

# Extract reference outputs (for verification)
python PyTorch/extract_reference.py --weights /path/to/ckpt.pth
```

## Run

```bash
# Basic inference
./run_nntrainer.sh

# With verification against PyTorch references
KEYWORD_VERIFY=1 ./run_nntrainer.sh

# Custom input
./run_nntrainer.sh /path/to/input.bin
```

## Verification

When `KEYWORD_VERIFY=1` is set, the nntrainer output is compared against
PyTorch reference `.bin` files:

- `ref_logits.bin` — 507-dim logits
- `ref_features.bin` — 512-dim features
- `ref_sigmoid.bin` — sigmoid(logits)

The `max_abs_diff` is printed for each. For FP32 inference, the difference
should be < 1e-4 (floating-point accumulation order differences).

## Architecture Details

### RepMixerBlock (fused)

```
x = dw_conv3x3(x)                          # token mixer (no act)
mlp_out = dw_conv7x7(x) + BN               # conv path
mlp_out = conv1x1(C, 4C) + GELU            # fc1
mlp_out = conv1x1(4C, C)                   # fc2
x = x + layer_scale(mlp_out)               # residual + scale
```

### AttentionBlock (fused, stage 3)

```
normed = BatchNorm2d(x)
qkv = conv1x1(512, 1536, normed)           # qkv projection
attn = fastvit_attention(qkv)              # 16-head attention (custom layer)
proj = conv1x1(512, 512, attn)             # output projection
x = x + layer_scale_1(proj)                # residual + scale 1
mlp_out = dw_conv7x7(x) + BN
mlp_out = conv1x1(512, 2048) + GELU
mlp_out = conv1x1(2048, 512)
x = x + layer_scale_2(mlp_out)             # residual + scale 2
```

### SE Module (final conv)

```
se = global_avg_pool(x)                    # [B, C, 1, 1]
se = conv1x1(C, C/16) + ReLU               # fc1
se = conv1x1(C/16, C) + Sigmoid            # fc2
out = x * se                               # channel reweighting
```

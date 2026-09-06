# V-JEPA 2.1 ViT-B Video Encoder

A faithful nntrainer port of the **V-JEPA 2.1 `vit_base` (ViT-B/16)** video
encoder, runnable through the CausalLM application on x86 and on-device (ARM,
Android). Both an FP32 and a Q4_0-quantized variant are supported.

## Model

`architectures: "VJEPA2ViT"` (HF `model_type` mapped from
`vjepa2_1_vit_base_384`).

| | |
|---|---|
| embed_dim / depth / heads | 768 / 12 / 12 (head_dim 64) |
| patch / tubelet | 16 (spatial) / 2 (temporal) |
| MLP | GELU, intermediate 3072 |
| norm | LayerNorm, eps 1e-6 |
| positional | **3D axial RoPE** (no learned pos-embed, no CLS token) |
| attention | **bidirectional** (non-causal), qkv_bias = true |
| params | ~86 M |

Key implementation choices:

- **Patch embedding** – the non-overlapping `Conv3d` tubelet projection is
  mathematically a linear map, so it is implemented as a host-side `patchify`
  (re-order `[C,T,H,W]` into `[num_tokens, in_chans*tubelet*patch*patch]`)
  followed by a single `fully_connected`. The video modality embedding is
  folded into the patch-embed bias by the converter.
- **3D axial RoPE** – `head_dim` (64) is split into depth/height/width slices
  (20/20/20, last 4 dims unrotated) and rotated by the frame/row/col index.
  Implemented by the custom `vjepa_rope` layer applied to Q and K; `mha_core`
  runs with its own RoPE disabled (`rope_theta=0`).
- Token ordering is `n = t·(Gh·Gw) + h·Gw + w` (matches `flatten(2)` of the
  reference `PatchEmbed3D`).

### Custom layers

| layer | purpose |
|---|---|
| `vjepa_rope` | 3D axial RoPE on Q/K |
| `vjepa_gelu` | NEON GELU parallelized over tokens (the core `activation` runs single-threaded) |
| `vjepa_layernorm` | per-token LayerNorm parallelized over tokens, pure FP32 |
| `mha_core` (`use_gemm_attention=true`) | optional non-causal flash attention |

The **flash attention** path (enabled for this encoder) tiles the score matrix
into `[Bq × Bk]` blocks with an online (running max/sum) softmax, so the
`O(N²)` score buffer is never materialized. Work is distributed over
`(head × query-block)` units across the thread pool so all cores stay busy.
QK uses `shgemm` (FP32 Q × FP16 K → FP32) because block-0 logits reach ~457k
and would overflow an FP16 product; AV uses the FP16 path. Tile sizes default
to `Bq=256, Bk=512` and can be overridden with `VJEPA_BQ` / `VJEPA_BK`.

## 1. Convert the released checkpoint

`weight_converter.py` (in `res/vjepa2/`) reads the released V-JEPA 2.1
`ema_encoder` checkpoint and writes the nntrainer weight blob in the exact
order the graph requests it.

```bash
python res/vjepa2/weight_converter.py \
    --input  vjepa2_1_vitb_dist_vitG_384.pt \
    --output nntr_vjepa2_vitb_fp32.bin
```

This produces the **FP32** weights. Place the `.bin` next to a model
directory containing `config.json`, `generation_config.json` and
`nntr_config.json` (see "Running" below).

## 2. Quantization (Q4_0)

Quantize the FP32 model with `nntr_quantize`. The FC layers (patch-embed,
q/k/v, attention-out, ffn-up/down) become `Q4_0`; activations stay FP32
(`Q4_0-FP32`). LayerNorm γ/β and FC biases remain FP32.

> **ARM / on-device: you must pass `--isa ARM`.**
> The ARM `q4_0` GEMM kernel (`__ggml_q4_0_4x8_q8_0_GEMM`) expects the weights
> repacked into the ggml 4×8 interleaved layout. The default (x86) layout
> produces all-zero output on ARM. The repack is layout-only, so it can be run
> on an x86 host; the resulting `.bin` is ARM-specific.

```bash
# x86 inference build:
nntr_quantize <model_dir> --fc_dtype Q4_0 --embd_dtype FP32 \
              --output_bin nntr_vjepa2_vitb_q40.bin

# Android / ARM device:
nntr_quantize <model_dir> --fc_dtype Q4_0 --embd_dtype FP32 --isa ARM \
              --output_bin nntr_vjepa2_vitb_q40_arm.bin
```

`nntr_config.json` for the quantized model:

```json
{
  "model_tensor_type": "Q4_0-FP32",
  "model_file_name": "nntr_vjepa2_vitb_q40_arm.bin",
  "model_type": "Model",
  "embedding_dtype": "FP32",
  "fc_layer_dtype": "Q4_0",
  "batch_size": 1,
  "max_seq_len": 4608,
  "init_seq_len": 4608,
  "num_to_generate": 0,
  "fsu": false,
  "skip_tokenizer": true
}
```

`max_seq_len`/`init_seq_len` = number of tokens = `(T/2)·(384/16)²`
(e.g. 16 frames → 8·24·24 = 4608, 32 frames → 9216). `config.json` carries the
HF geometry (`num_frames`, `img_size`, `patch_size`, `tubelet_size`, …).

## 3. Running

The input is a raw `float32` `[C, T, H, W]` tensor (already resized and
normalized to the model's expectations):

```bash
nntr_causallm <model_dir> <input_video.bin>
```

The last-token hidden state (`[hidden_size]`) is printed and dumped to
`<input_video.bin>.nntr_out.bin` for offline comparison against a torch
reference.

On Android, set the thread count via the environment (the CausalLM core is
built with a small default); 8 (= core count on recent Galaxy SoCs) is optimal:

```bash
NNTR_NUM_THREADS=8 LD_LIBRARY_PATH=. ./nntrainer_causallm <model_dir> <input.bin>
```

## Accuracy & performance

Cosine similarity of the last-token output vs. the official torch reference:

| variant | cos |
|---|---|
| x86 FP32 | ~1.000 |
| x86 Q4_0 | ~0.991 |
| device Q4_0 (ARM) | ~0.99 |

End-to-end latency (Q4_0, Galaxy S25 Ultra, 8 threads), after the attention
and activation optimizations (flash attention + load balancing + parallel
GELU/LayerNorm):

| frames (tokens) | e2e | peak RAM |
|---|---|---|
| 16 (4608) | ~3.8 s | ~0.6 GB |
| 32 (9216) | ~13 s | ~1.0 GB |

## Notes

- The encoder is built with `ENABLE_FP16=0` (FP32 attention scores). A fully
  FP16 attention path overflows on the large block-0 logits (~457k > 65504).
- `vjepa_gelu` / `vjepa_layernorm` reproduce the core ops exactly (the LN even
  improves device accuracy by keeping FP32 throughout) while parallelizing the
  per-token work that the stock single-threaded layers leave on one core.

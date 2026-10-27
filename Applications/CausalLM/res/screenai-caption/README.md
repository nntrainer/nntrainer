<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com> -->

# ScreenAI Caption Model

Image-to-text captioning model ported to nntrainer's `Applications/CausalLM` app.

## Architecture

The model consists of three modules:

| Module | Class | File |
| :--- | :--- | :--- |
| Vision encoder | `Siglip2VisionEncoder` | `models/siglip2/siglip2_vision_encoder.cpp` |
| Cross-attention decoder | `BertDecoder` | `models/bert_decoder/bert_decoder.cpp` |
| Orchestrator | `ScreenAICaption` | `models/screenai_caption/screenai_caption.cpp` |

**Encoder** — SigLIP2 ViT-B/16: patch embedding (16 × 16, stride 16) → 12 transformer layers with self-attention → post-LayerNorm → linear projection to decoder hidden size. The pooling head is intentionally skipped (captioning uses all 196 patch tokens).

**Decoder** — Mini-BERT (4 layers, hidden 256, 4 heads, vocab 30522): word / position / token-type embeddings → causal self-attention → cross-attention over encoder output → FFN → BERT LM head with tied word-embedding projection and per-token bias. Greedy decoding stops at `[SEP]` (token 102).

**Orchestrator** — `ScreenAICaption` is registered under the factory key `"ScreenAICaption"`. `config.json` `architectures=["VisionEncoderDecoderModel"]` dispatches to it from `main.cpp`.

Entry point: `nntr_causallm <model_dir> <image>`.

> Note: conv `im2col` pre-pack and YOLOv11m detection are a **separate track** and are not part of this model.

---

## Resource layout

```
res/screenai-caption/caption-s02/
├── config.json                         # HuggingFace model config (architecture dispatch)
├── nntr_config.json                    # nntrainer runtime config
├── tokenizer.json                      # BERT WordPiece tokenizer
├── sample.png                          # Sample input image
├── golden.json                         # Reference token IDs + caption (parity gate)
├── weight_converter.py                 # Converts model.safetensors → nntrainer weights
├── verify_parity.py                    # PyTorch reference + nntrainer token comparison
├── nntr_siglip2_encoder_fp32.bin       # Encoder weights (gitignored, regenerate below)
└── nntr_caption_decoder_fp32.bin       # Decoder weights (gitignored, regenerate below)
```

The `.bin` (and `.safetensors`) weight files are listed in `.gitignore` and must be regenerated from the original `caption_s02/model.safetensors` checkpoint.

---

## Weight conversion

### Prerequisites

```bash
pip install numpy safetensors torch
```

### Convert (default binary format)

```bash
cd Applications/CausalLM/res/screenai-caption/caption-s02
python weight_converter.py \
    --model_path /path/to/caption_s02/model.safetensors \
    --encoder_output nntr_siglip2_encoder_fp32.bin \
    --decoder_output nntr_caption_decoder_fp32.bin
```

### Convert to safetensors format

```bash
python weight_converter.py \
    --model_path /path/to/caption_s02/model.safetensors \
    --safetensors
```

This writes `nntr_siglip2_encoder_fp32.safetensors` and `nntr_caption_decoder_fp32.safetensors` (in safetensors format; no `.bin` files are produced in this invocation).

### Output files

| File | Tensors | Contents |
| :--- | :---: | :--- |
| `nntr_siglip2_encoder_fp32.bin` | 199 | Patch embed, pos embed, 12× encoder-layer weights, post-LayerNorm, enc-to-dec projection |
| `nntr_caption_decoder_fp32.bin` | 114 | Word/pos/type embeddings, emb-LayerNorm, 4× decoder-layer weights (self+cross attn, FFN), LM head |

The script asserts the exact tensor counts and aborts if they do not match.

---

## nntr_config.json key fields

| Key | Value | Meaning |
| :--- | :--- | :--- |
| `model_type` | `"ScreenAICaption"` | Factory key used in `main.cpp` dispatch |
| `encoder_model_file_name` | `"nntr_siglip2_encoder_fp32.bin"` | Encoder weight file (relative to model dir) |
| `decoder_model_file_name` | `"nntr_caption_decoder_fp32.bin"` | Decoder weight file (relative to model dir) |
| `model_tensor_type` | `"FP32-FP32"` | Weight and compute dtype |
| `patch_size` / `img_size` | `16` / `224` | SigLIP2 ViT-B/16 image config |
| `num_patches` | `196` | 14 × 14 patches after patchify |
| `encoder_hidden_size` | `768` | SigLIP2 output dimension |
| `decoder_hidden_size` | `256` | BERT decoder hidden dimension |
| `max_seq_len` | `64` | KV-cache capacity for decoder |
| `num_to_generate` | `32` | Maximum new tokens to generate |
| `tokenizer_file` | `"tokenizer.json"` | BERT WordPiece tokenizer (relative to model dir) |
| `fsu` | `false` | FSU (on-the-fly expert loading) not used for this model |

---

## Desktop build and run

### Build

```bash
# From the repo root
meson setup builddir-desktop \
    -Denable-transformer=true

ninja -C builddir-desktop Applications/CausalLM/nntr_causallm
```

### Run

**LD_LIBRARY_PATH gotcha**: the environment default may point to a stale nntrainer install. Prepend the freshly built libraries before running:

```bash
export LD_LIBRARY_PATH="\
$PWD/builddir-desktop/nntrainer:\
$PWD/builddir-desktop/Applications/CausalLM:\
$PWD/builddir-desktop/Applications/CausalLM/layers:\
$(find $PWD/builddir-desktop/Applications/CausalLM/models \
      -name '*.so' -exec dirname {} \; | sort -u | tr '\n' ':')$LD_LIBRARY_PATH"

./builddir-desktop/Applications/CausalLM/nntr_causallm \
    Applications/CausalLM/res/screenai-caption/caption-s02 \
    Applications/CausalLM/res/screenai-caption/caption-s02/sample.png
```

---

## Android build and run

### Prerequisites

- `ANDROID_NDK` env var set to your NDK path (r21d or later recommended)
- `ndk-build` on `PATH`
- `subprojects/iniparser` must be populated (run `meson subprojects download iniparser` or equivalent before the Android build if not already present)

### Build

```bash
cd Applications/CausalLM
./build_android.sh
# Optionally reuse existing nntrainer Android build:
#   ./build_android.sh --cache
```

This runs `ndk-build` via `jni/Android.mk` and produces arm64-v8a artifacts under `jni/libs/arm64-v8a/`.

### Install to device

```bash
./install_android.sh
```

Pushes `nntrainer_causallm`, `libcausallm_core.so`, `libnntrainer.so`, `libccapi-nntrainer.so`, and `libc++_shared.so` to `/data/local/tmp/nntrainer/causallm/` and creates `run_causallm.sh` on the device.

### Push model weights

The weight files must be regenerated (see Weight conversion above) then pushed:

```bash
adb push Applications/CausalLM/res/screenai-caption/caption-s02 \
    /data/local/tmp/nntrainer/causallm/models/caption-s02
```

This transfers the entire `caption-s02/` directory: `nntr_config.json`, `config.json`, `tokenizer.json`, `sample.png`, `golden.json`, the regenerated weight files (`.bin` or `.safetensors`), and any `.npy` debug artifacts present.

### Run on device

```bash
adb shell /data/local/tmp/nntrainer/causallm/run_causallm.sh \
    /data/local/tmp/nntrainer/causallm/models/caption-s02 \
    /data/local/tmp/nntrainer/causallm/models/caption-s02/sample.png
```

---

## Parity verification

### Verification command

```bash
python Applications/CausalLM/res/screenai-caption/caption-s02/verify_parity.py \
    --ckpt /path/to/caption_s02 \
    --image Applications/CausalLM/res/screenai-caption/caption-s02/sample.png \
    --nntr-tokens nntr_tokens.json
```

Prints `TOKEN MATCH: True` and exits 0 on success, 1 on mismatch.

### Desktop (x86, fp32) — PASS

The nntrainer greedy-decode token sequence is byte-identical to the PyTorch golden on x86:

```
[ScreenAICaption] token_ids: 101 1037 12117 12326 1997 1037 12117 12326 4760 1037 8370 1997 3793 1998 1037 3793 1012 102
[ScreenAICaption] caption: a screenshot of a screenshot showing a grid of text and a text.
[e2e time]: 3156 ms
Max Resident Set Size: 779172 KB
```

Golden reference (from `golden.json`):
- Token IDs: `[101,1037,12117,12326,1997,1037,12117,12326,4760,1037,8370,1997,3793,1998,1037,3793,1012,102]` (18 tokens)
- Caption: `"a screenshot of a screenshot showing a grid of text and a text."`

**TOKEN MATCH: True** — this is the acceptance gate and it passes on **both desktop (x86 fp32) and Android (arm64 enable-fp16)**, producing the identical token sequence. PyTorch fp16 inference also produces the identical tokens, confirming the model is fp16-robust.

### Android (arm64, enable-fp16) — PASS (exact token match)

Android builds, runs, and is deterministic across runs. After the core fixes below, the
on-device greedy token sequence **matches the x86/PyTorch golden exactly** — same 18 tokens,
same caption `"a screenshot of a screenshot showing a grid of text and a text."`. The
on-device decoder layer-0 cross-attention output is `max|x|≈1.65`, matching desktop (i.e. the
encoder signal reaches the decoder correctly). Encoder output cosine-similarity vs the x86
reference is ≈0.99999, and that small fp32 cross-architecture difference does not flip any
greedy token for this model.

Android acceptance criteria (all met): deterministic on-device output **and** exact token-ID
match to the golden (same gate as desktop).

### Core nntrainer / decoder fixes required for Android

These fixes are all guarded on the Android / ARM fp16 path (or are platform-symmetric) —
desktop and other models remain byte-identical:

1. **LayerNorm fp16 RMS-norm intrinsic (ARM)**: the deprecated `rms_norm_wrt_width_fp16_intrinsic(float*)` overload throws on ARM; LayerNorm now uses the generic fp32 path for FP32 width-axis normalization, so FP32 LayerNorm works on ARM fp16 builds.

2. **MHA-core attention accumulation (ARM)**: `mha_core` no longer down-casts FP32 Q/K/V to fp16 on Android; it fp32-accumulates attention over the fp16-stored KV cache (the same path desktop uses), avoiding precision loss across the 12 encoder layers.

3. **Cross-attention cross-cache wiring + encoding (ARM)**: the decoder's cross-cache was (a) created as bare tensors instead of `input` layers on ARM, so the prefilled encoder K/V was never fed to the graph (cross-attention read zeros → image-agnostic captions), and (b) stored via an FP32→UINT16 path that is a real fp16 conversion on x86 but an integer cast on the ARM fallback (collapsing |x|<1 to 0/1). Both fixed by making the ARM cross-cache an `input` layer with FP16 dtype end-to-end, so the prefilled encoder K/V reaches the decoder and is encoded correctly. This was the root cause of the earlier on-device caption divergence.

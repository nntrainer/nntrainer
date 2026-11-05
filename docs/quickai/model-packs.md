# Model packs

A model pack is a directory that holds a model's weights in the layout the
HTP backend reads, plus the JSON files that describe it. The app, the C API
and the command-line tool all open the same pack. This page gives the pack
contract for the two supported models, explains where packs come from, and
documents the converters in this repository.

## Where a pack lives

Every pack has a fixed directory name and sits under a *model base path*.
The default base path, used by the app, the library and the C API when none
is given, is `/sdcard/Download/aistudio-mobile/models/`.

| Model | Catalog id | Directory under the base path |
|---|---|---|
| Gemma4 26B-A4B | `gemma4-26b-a4b-htp` | `gemma4-26b-a4b-htp-w4a32/` |
| Gemma4 26B-A4B, 30k context | `gemma4-26b-a4b-htp-long` | `gemma4-26b-a4b-htp-long-w4a32/` |
| Gemma4 26B-A4B, rotated 2/4-bit | `gemma4-26b-a4b-htp-2bit-4bit` | `gemma4-26b-a4b-2bit-4bit/` |
| LFM2 24B-A2B | `lfm2-24ba2b-htp` | `lfm2-24ba2b-htp-w4a32/` |

To stage a pack:

```bash
adb shell mkdir -p /sdcard/Download/aistudio-mobile/models
adb push gemma4-26b-a4b-htp-w4a32 /sdcard/Download/aistudio-mobile/models/
```

The app shows a "found" mark in its model sheet when
`<base>/<directory>/nntr_config.json` exists. Reading from `/sdcard`
needs the all-files access the app asks for on first launch.
`/data/local/tmp` is not readable by an app on a production phone; use it
only for the command-line tool.

## Files in a pack

| File | Role | Required |
|---|---|---|
| `nntr_config.json` | Selects the model, the weight formats, the engines and the runtime policy (`htp` / `moe` objects, see [Configuration](configuration.md)) | yes |
| `config.json` | The model's Hugging Face configuration, **flat** (the text config at top level, not wrapped in `text_config`) | yes |
| the weight file named by `model_file_name` | Positional binary in nntrainer's save order | yes |
| `tokenizer.json` | Hugging Face tokenizer. The loader writes a `tokenizer.json.ntkc` snapshot beside it on first use (or under `$XDG_CACHE_HOME/nntrainer/tokenizer/` when the directory is read-only) | yes |
| `chat_template.jinja` | The chat template; without it the template in `tokenizer_config.json` is used, and without either a built-in fallback with a logcat warning | recommended |
| `tokenizer_config.json`, `special_tokens_map.json` | Template fallback and BOS/EOS ids | optional |
| `generation_config.json` | `do_sample`, `top_k`, `top_p`, `temperature`, `eos_token_id`, `bos_token_id` | optional |
| `conversion_report.json` | LFM2 only: the converter's record of tensors and rotations, validated at load | LFM2: yes |
| sidecars named in `nntr_config.json` | `moe_hadamard_signs` (rotated Gemma4), `static_activation_file`, `fc_wh_file_name`, `moe.tl2_prepared_file`; resolved relative to the pack | optional |
| `<weights>.lmhead_wh` | Written by the loader on first load of a Gemma4 pack with an untied `QS4CX` head; a cache of the head's WH slices. Delete it after requantizing the head. | written |

## Where packs come from

The packs are not built from a public Hugging Face checkpoint in one step.
Each starts from a checkpoint prepared outside this repository and is
repacked and verified by the converters below:

- **Gemma4 26B-A4B**, native TL2 pack: from the pre-rotated, fake-quantized
  BF16 checkpoint (exact ternary experts, FC and head on an int4 grid, a
  `rotation_meta.json`), in one run of
  `Applications/CausalLM/res/gemma4-26b-a4b/weight_converter.py`.
- **Gemma4 26B-A4B**, 2-bit palette pack (`QS2CX_WH` experts, `QS4CX`
  dense, `Q4_0` tied head; 7,540,724,856 bytes): this is a delivered
  artifact. `Applications/CausalLM/res/gemma4/htp_weight_converter.py`
  writes the FP32 `htp_fused_v1` layout from the Hugging Face checkpoint,
  but the quantizer that produced the shipped 2-bit file is not in this
  tree.
- **LFM2 24B-A2B**: from the SR15 checkpoint (a single `model.safetensors`,
  the two rotation matrices `R_in_f32.npy` / `R_down_f32.npy`, and its
  activation encodings) with `tools/convert_lfm2_sr15.py`.

If you received a pack, you only need to stage it. If you have one of the
source checkpoints, follow the converter sections below. Changing a dtype
name in `nntr_config.json` never converts the weights; the loaders verify
that the serialized records match the declared formats and refuse a pack
that does not.

## Gemma4 26B-A4B

Two weight layouts are accepted. Both declare
`gemma4_execution_profile: "htp_fused"` and
`gemma4_weight_format: "htp_fused_v1"`; they differ in the dtype keys.

| Tensor | 2-bit palette pack | Native TL2 pack |
|---|---|---|
| Routed experts (`moe_layer_dtype`) | `QS2CX_WH` | `TL2CX_WH` (some layers may be `QS4CX_WH_AFFINE` through `moe_layer_dtype_per_layer`) |
| Attention and dense FFN projections (`fc_layer_dtype`) | `QS4CX` | `QS4CX_WH_AFFINE` |
| Embedding (`embedding_dtype`) | `Q4_0` | `Q4_0` |
| Language-model head (`lmhead_dtype`, `lmhead_engine`) | tied to the embedding, `Q4_0`, `lmhead_engine: cpu` | untied, `QS4CX_WH_AFFINE`, `lmhead_engine: htp` |
| Router, norms, block scalars | FP32 | FP32 |
| Attention KV cache (`attention_kv_dtype`) | `q8` | `q8` |
| Weight file (`model_file_name`) | `nntr_gemma4_26b_a4b_qs2cx_fcqs4cx.bin` | `nntr_gemma4_26b_a4b_tl2_rotated.bin` (one-run converter) or `nntr_gemma4_ternary_tl2_qs4cx_wh_affine.bin` (two-stage chain) |

The remaining `nntr_config.json` keys of a Gemma4 pack:

| Key | Value | Meaning |
|---|---|---|
| `model_type`, `batch_size` | `CausalLM`, `1` | Fixed. |
| `model_tensor_type` | `Q4_0-FP32` | The part after the last `-` is the activation type and must be `FP32`. |
| `moe_engine`, `attention_engine`, `attn_proj_engine`, `dense_ffn_engine` | `htp` | Which blocks run on the DSP. The C API requires all four. |
| `moe_htp_layers`, `attn_proj_htp_layers`, `dense_ffn_htp_layers` | absent | Comma-separated layer ids that restrict an engine to some layers. Must be absent for WH expert formats. |
| `init_seq_len`, `max_seq_len` | `2048`, `2304` in the chat preset | Prompt capacity and total context. A prompt may use `min(init_seq_len, max_seq_len - num_to_generate)` tokens, 2,048 in the chat preset. `max_seq_len` is clamped to the checkpoint's `max_position_embeddings`. A prompt whose rendered length exceeds the capacity is **truncated silently** (nntrainer issue #4474), which is why the app sends only the newest message to this model. Raise both for longer prompts; memory grows with them. |
| `num_to_generate` | `127` in the preset | Generation budget per call. |
| `bad_word_ids` | `[]` | Token ids never generated. A fixed-length benchmark puts the EOS ids here. |
| `tokenizer_file` | `tokenizer.json` | The C API rewrites it to the pack's absolute path. |
| `use_chat_template` | `true` | Apply the chat template to plain-text prompts (API). |
| `skip_prefill` | `false` | |
| `sample_input` | the default prompt | Used by the command line when no prompt is given. |
| `moe_cache_experts`, `moe_hadamard`, `moe_hadamard_scope` | legacy flat keys | Still accepted; prefer the `moe` object. See [Configuration](configuration.md#legacy-flat-keys). |
| `use_kvcache`, `save_kvcache`, `kv_window_ring`, `fsu` | absent or `false` | Not supported for HTP packs through the API. |

`config.json` must be the flat `Gemma4ForCausalLM` text configuration with
`hidden_size` 2816, `num_hidden_layers` 30 and `num_experts` 128 (the C API
checks these four before loading), the `layer_types` list that marks
`sliding_attention` and `full_attention` layers, and the global attention
geometry `global_head_dim` and `num_global_key_value_heads`.
`tie_word_embeddings` decides whether the head is the embedding table (the
palette pack) or its own record (the native pack). `hidden_size_per_layer_input`
must be 0 and `attn_logit_softcapping` absent for the resident decode graph.

### The chat preset

`Applications/CausalLM/res/gemma4/htp_chat/` holds the three files that make
the palette pack usable for conversation. Copy them into the pack directory:

| File | What it sets |
|---|---|
| `nntr_config.json` | The keys above with the 2-bit palette dtypes, 2,304 positions, prompts up to 2,048 tokens, 256 tokens per answer, EOS stopping enabled. |
| `generation_config.json` | Greedy decoding (`do_sample` false), EOS ids `[1, 106, 50]`, BOS 2. |
| `chat_template.jinja` | Google's canonical Gemma 4 template, which closes the thought channel when thinking is off. Take it from here, not from another pack: a variant that splits one string literal over several lines is rejected by the built-in template engine, the API then falls back to a bare `<|turn>user … <|turn>model` format, and the model starts its answer with `<|channel>thought`. The only symptom is a `Chat template apply failed` line in logcat. |

A benchmark pack that disables EOS (so every run generates a fixed number of
tokens) is not suitable for interactive use; keep the preset for anything a
person reads.

### Rotated checkpoints

A pre-rotated checkpoint needs the runtime to apply the same rotation to the
expert inputs. Its pack carries:

```json
{
  "moe_hadamard": "both_paley",
  "moe_hadamard_scope": "all",
  "moe_hadamard_signs": "rotation_signs.json"
}
```

`moe_hadamard` names the basis (`both` or `both_paley` for the delivered
checkpoints; the full list is in [Configuration](configuration.md#moe-keys)),
`all` rotates every routed expert layer and never the dense FFN, and
`rotation_signs.json` (format `nntr_hadamard_signs_v1`, an `in` array of
2,816 and a `down` array of 704 values ±1) holds the checkpoint's
per-coordinate signs. The one-run converter regenerates the signs from the
seed in `rotation_meta.json`. Mode, scope and signs are part of the weight
contract: a mismatch produces fluent but wrong text, and rotation stays off
for an unrotated checkpoint.

A checkpoint that ships static input encodings adds
`"static_activation_file": "static_linear.json"` (either the
`static_linear.json` or the `encoding_provenance.json` format is accepted).
The linear inputs then use the supplied unsigned 8-bit grids in prefill and
decode; the attention cache keeps the backend's own int8 scales.

### Converting a Gemma4 checkpoint

**One run, from the rotated BF16 checkpoint** (needs Python 3 with NumPy,
`gcc-13` and `g++` on the host):

```bash
python3 Applications/CausalLM/res/gemma4-26b-a4b/weight_converter.py \
  --source /path/to/gemma4-26b-a4b-rotated-bf16 \
  --output /path/to/gemma4-26b-a4b-htp-w4a32 \
  [--affine-layers 28,29]
```

It writes the 2-bit stage file, builds `tools/convert_tl2_model` and turns
the experts into native TL2, recovers the FC and head int4 grids from the
BF16 values as affine WH records (no requantization), writes
`rotation_signs.json`, the rotation keys and the global attention keys, and
copies `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`generation_config.json` and `chat_template.jinja`. The experts of
`--affine-layers` (default `28,29`, the recipe's 4-bit expert layers) are
written as `QS4CX_WH_AFFINE`; a layer whose values are not ternary is written
affine regardless. `--affine-layers ''` makes every layer TL2 when the
checkpoint allows it, which uses less arena memory. The output directory
must not exist.

**Two stages, from a verified 2-bit pack** (a pack with `QS2CX_WH` experts,
`Q4_0` dense and head, and its `conversion_report.json`) plus the BF16
checkpoint for the affine grids:

```bash
mkdir -p /tmp/nntr-tl2-tools
cc -O3 -c nntrainer/tensor/htp_backend/htp_tl2_codec.c -I nntrainer/tensor/htp_backend \
  -o /tmp/nntr-tl2-tools/htp_tl2_codec.o
c++ -O3 -std=c++17 -pthread tools/convert_tl2_model.cpp /tmp/nntr-tl2-tools/htp_tl2_codec.o \
  -I nntrainer/tensor -I nntrainer/tensor/htp_backend -o /tmp/nntr-tl2-tools/convert_tl2_model
python3 Applications/CausalLM/res/gemma4/convert_gemma4_tl2.py /path/to/verified-qs2-q40-pack \
  --output /path/to/gemma4-tl2-affine \
  --converter /tmp/nntr-tl2-tools/convert_tl2_model \
  --hf-source /path/to/bf16-checkpoint
```

Python 3.11 or newer and NumPy; the affine stage reads the BF16 rows itself
(no PyTorch). Every converted record is decoded and compared with its
source; the output is deleted on any failure.

**FP32 layout only**: `Applications/CausalLM/res/gemma4/htp_weight_converter.py
--model_path <HF dir> --output_name <file.bin>` writes the FP32
`htp_fused_v1` positional file and a `.manifest.json` beside it (needs
`torch`, `transformers` with `Gemma4Config`, `numpy`, `safetensors`). It is
the reference serializer that the host test
`generators/check_gemma4_htp_converter.py` pins; it does not produce a
runnable 2-bit pack.

Host checks for the converters: `test/htp/host/tl2_model_conversion_check.py
<path to convert_tl2_model>` and `test/htp/host/wh_affine_conversion_check.py`.

## LFM2 24B-A2B

The LFM2 pack has one accepted format, `lfm2_weight_format: "sr15_v2"`, with
`lfm2_execution_profile: "htp_fused"`:

| Tensor | Format |
|---|---|
| Routed experts (`moe_layer_dtype`) | `TL2CX_WH` |
| Attention, convolution and dense projections (`fc_layer_dtype`), router gate (`router_dtype`), untied head (`lmhead_dtype`) | `QS4CX_WH_AFFINE` |
| Embedding (`embedding_dtype`) | `Q4_0` |
| Short-convolution taps (`short_conv_dtype`) | `DW3_INT4` |
| Norms, expert bias | FP32 |
| Attention KV cache | per-row int8 (`q8_row`, forced by the loader) |
| Rotations | two records (`R_in`, `R_down`) described in `conversion_report.json` and validated at load; `moe_hadamard: both`, `moe_hadamard_scope: all` |

All five engine keys are `htp`, `model_tensor_type` is `FP32-FP32`,
`activation_quantization` is `dynamic` (the shipped `activation_scales.bin`
is not consumed), `fsu` and `kv_window_ring` are false. The converter writes
`init_seq_len` 256, `max_seq_len` 2048 and `num_to_generate` 32; raise them
in the pack for longer prompts and answers (the measured 2,048-token
workload used `init_seq_len` 2048 and `max_seq_len` 4096). The legacy
unrotated LFM2 layout is rejected with a named error.

`config.json` must say `model_type` `lfm2_moe` and `architectures`
`["Lfm2MoeForCausalLM"]`, with `layer_types` naming each layer `conv` or
`full_attention`, `head_dim` 64, `conv_L_cache` 3, `conv_bias` false,
`norm_topk_prob` true, `routed_scaling_factor` 1.0, `use_expert_bias` true,
`tie_word_embeddings` false, and `num_experts` a multiple of 32.

### Converting the SR15 checkpoint

```bash
python3 tools/convert_lfm2_sr15.py /path/to/sr15-checkpoint \
  --encodings /path/to/encodings \
  --output /path/to/lfm2-24ba2b-htp-w4a32 \
  --tokenizer-source /path/to/tokenizer \
  --rotation both
python3 test/htp/host/lfm2_conversion_check.py
```

Inputs: `config.json`, a single `model.safetensors`, `R_in_f32.npy` and
`R_down_f32.npy` in the checkpoint directory; the two `.encodings` files
in the encodings directory; `tokenizer.json` and `tokenizer_config.json` in
the tokenizer directory. Needs NumPy and a host C compiler (the script
compiles the TL2 codec and loads it through `ctypes`). The output directory
must not exist.

Outputs: `LFM2-24B-A2B-Ternary-INT4.bin`, `conversion_report.json`,
`activation_scales.bin`, `nntr_config.json`, and the copied `config.json`,
`generation_config.json`, `chat_template.jinja`, `tokenizer.json` and
`tokenizer_config.json`. The converter re-reads and decodes every TL2
expert against the source, verifies the affine tiles, checks the embedding
requantization error, the rotation round trip and that every source tensor
was consumed exactly once. Keep `conversion_report.json` with the weights:
the loader validates the rotation records against it and refuses to load
without it. `--rotation both` is required for correct SR15 inference.

## Formats

| Format | Definition | Where it is used |
|---|---|---|
| `QS4CX` | Per-output-channel int4: row-major nibbles plus one FP32 scale per channel | Gemma4 palette pack projections |
| `QS4CX_WH` | The same values arranged in the HMX 32×32 tile order plus per-channel column sums | Accepted for Gemma4 projections, used by neither shipped pack |
| `QS2CX_WH` | WH tiles at two bits: each code indexes one of four int4 levels in a per-tensor palette; `K*N/4` bytes plus the palette, scales and sums | Gemma4 palette pack experts |
| `TL2CX_WH` | Native ternary: 640 bytes per 24-row × 128-column block, plus FP32 scales and column sums; levels −1, 0, 1 | Gemma4 native pack experts, LFM2 experts |
| `QS4CX_WH_AFFINE` | WH int4 plus FP32 scales, FP32 code sums and INT32 channel offsets; `w = scale × (q + offset)`; `K*N/2 + 12*N` bytes | Gemma4 native pack projections and head; LFM2 projections, router and head; optional 4-bit expert layers |
| `Q4_0` | GGML Q4_0: 32 values in 18 bytes | Embeddings; the Gemma4 palette pack's tied head |
| `DW3_INT4` | Three int4 taps per channel with FP32 scale and INT32 offset, decoded to FP32 at load | LFM2 short convolutions |

`Q6_K` and the other CPU quantization types are not accepted by the HTP
loaders. The exact record layouts are in
[HTP backend architecture](../backend_guide/HTP_ARCHITECTURE.md#2-weight-formats).

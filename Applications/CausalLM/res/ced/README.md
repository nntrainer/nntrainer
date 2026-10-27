# CED — nntrainer Inference Port

Port of [CED](https://huggingface.co/mispeech/ced-tiny) (Consistent Ensemble
Distillation), an AudioSet tagging model, driven entirely from the upstream
HuggingFace `config.json`.

## Architecture

CED is a ViT over mel-spectrogram patches, so the encoder trunk is the one
`TimmViTTransformer` already builds. `CedTransformer` only supplies what
differs:

| | timm ViT (image) | CED (audio) |
|---|---|---|
| input | 3 × 224 × 224 | **1 × n_mels × target_length** (1 × 64 × 1012) |
| input norm | — | **`init_bn`**: BatchNorm over the frequency axis |
| pos embed | one `[1, N, D]` table | **separable** `time_pos_embed` + `freq_pos_embed` |
| head | none (feature output) | **mean-pool → LayerNorm → Linear → sigmoid** |
| trunk | 12 pre-norm blocks | identical (config only: D=192, heads=3, mlp=768) |

Two details are easy to get wrong:

- **LayerNorm epsilons differ.** The encoder is built with
  `partial(nn.LayerNorm, eps=1e-6)`, but the head's `outputlayer.0` is a plain
  `nn.LayerNorm`, so it uses PyTorch's `1e-5` default.
- **The positional embeddings are pre-summed offline.** `modeling_ced.py` adds
  `time_pos_embed [1,D,1,T]` and `freq_pos_embed [1,D,F,1]` in NCHW *before*
  flattening. Both are constants, so `weight_converter.py` folds their
  broadcast sum into the single `[1,1,F*T,D]` table the graph adds after
  flattening. `flatten(2,3)` is frequency-major, which is also how a
  `[D,F,T]` NCHW tensor reshapes to `[D,F*T]`, so the orderings agree.

## Input boundary

The model input is `input_values`, the mel-dB spectrogram — the same tensor
HuggingFace's `CedModel.forward()` takes. The STFT / mel / dB front-end lives
outside the model in both implementations, so the runner reads a raw FP32
`[n_mels, frames]` blob.

## Weight conversion

```bash
python weight_converter.py \
  --input  /path/to/ced-tiny \
  --output /path/to/ced-tiny/nntr_ced_tiny_fp32.bin
```

The blob order matches the graph's layer creation order; see the header comment
in `weight_converter.py`.

## Reference extraction and verification

```bash
# PyTorch reference: input, intermediates, and final 527-class logits
python extract_reference.py --out-dir /path/to/ced-tiny/nntr_ref

# optionally from a real 16 kHz wav instead of a deterministic signal
python extract_reference.py --out-dir <dir> --wav sample.wav
```

Then run with `CED_REF_BIN` pointing at the reference logits:

```bash
CED_REF_BIN=/path/to/ced-tiny/nntr_ref/ref_logits.bin \
  ./builddir/Applications/CausalLM/nntr_causallm /path/to/ced-tiny
```

`CED_OUT_BIN=<path>` additionally dumps the raw 527-float output.

## Reproduce / cross-verify in one command

```bash
# from the repo root, after: meson setup builddir -Denable-transformer=true \
#                                 -Denable-app=true && ninja -C builddir
Applications/CausalLM/res/ced/verify_e2e.sh
```

It extracts the PyTorch reference from the HuggingFace checkpoint, converts the
weights, then checks FP32 and Q8_0 against it on two unrelated inputs, and
reports Q4_0 alongside for contrast. Exits non-zero if a gated check misses its
budget. `--skip-reference` reuses the existing `nntr_ref/`, `--skip-quant`
stops after FP32, and `TOL_FP32` / `TOL_Q8` override the budgets.

### Provenance of the verified numbers

| | |
|---|---|
| checkpoint | `mispeech/ced-tiny` |
| `model.safetensors` | `0e086f0cd62814c6def89001f3f25193f75955696f6975ef6800af31d00d6dd7` |
| `config.json` | `3a060c57c28b8138cf66bbab2c0dcab79f07c871b600f52a06f2652bf3cad2bd` |
| reference path | `AutoModelForAudioClassification.from_pretrained(..., trust_remote_code=True)` |
| torch / torchaudio / transformers | 2.11.0+cu130 / 2.11.0+cu130 / 5.12.1 |
| host | x86-64, gcc 11.4, FP32, `NNTR_NUM_THREADS=4` |

Inputs are generated deterministically inside the scripts (seed 1234 noise, and
a 200->3500 Hz chirp with click train), so no audio files need to travel with
the fixtures.

## Verified results (x86, FP32)

`ced-tiny`, all 527 classes compared against HuggingFace
`CedForAudioClassification`:

| input | max_abs_diff | rms_diff | NaN | top-1 |
|---|---|---|---|---|
| white noise (seed 1234) | 2.01e-05 | 9.86e-07 | 0 | `White noise` 0.4694 (ref 0.4694) |
| chirp 200→3500 Hz + clicks | 1.76e-05 | 1.01e-06 | 0 | `Sine wave` 0.5435 (ref 0.5435) |

Top-5 labels and their order match the reference exactly for both inputs.

## Quantization

FC layers carry their own `weight_dtype`, taken from `fc_layer_dtype`, so the
encoder can be quantized while the patch conv, the LayerNorms, the positional
table and `init_bn` stay FP32 -- none of those has a block-quantized form and
their shapes are not 32-divisible in general. The classifier head follows
`lmhead_dtype` (or `head_dtype`) separately.

```bash
nntr_quantize /path/to/ced-tiny --fc_dtype Q8_0 --isa X86 \
  -o /path/to/ced-tiny-w8a8 --output_bin nntr_ced_tiny_q8_0.bin

CED_REF_TOL=0.05 CED_REF_BIN=/path/to/ced-tiny/nntr_ref/ref_logits.bin \
  nntr_causallm /path/to/ced-tiny-w8a8 \
  /path/to/ced-tiny/nntr_ref/input_values.bin
```

`Q8_0` is W8A8: the weights are int8 blocks with fp16 per-32 scales, and
`gemm_q8_0` quantizes every FP32 activation row to `block_q8_0` before the
int8 x int8 GEMM, so both operands are 8-bit at compute time. `Q4_0` is W4A8 --
4-bit weights against the same int8-quantized activations.

Measured against the same HuggingFace FP32 reference logits (x86, all 527
classes, two inputs). "order" is pairwise ranking agreement among the
reference's top-20 classes:

| weights | size | input | max_abs_diff | rms_diff | top-1 | top-5 | order |
|---|---|---|---|---|---|---|---|
| FP32 | 21.13 MB | noise | 2.01e-05 | 9.9e-07 | same | 5/5 | 190/190 |
| FP32 | 21.13 MB | chirp | 1.76e-05 | 1.0e-06 | same | 5/5 | 190/190 |
| Q8_0 (W8A8) | 6.26 MB | noise | 1.13e-02 | 5.8e-04 | same | 5/5 | 188/190 |
| Q8_0 (W8A8) | 6.26 MB | chirp | 6.13e-03 | 3.1e-04 | same | 5/5 | 190/190 |
| Q4_0 (W4A8) | 3.73 MB | noise | 2.34e-01 | 1.1e-02 | **differs** | 4/5 | 180/190 |
| Q4_0 (W4A8) | 3.73 MB | chirp | 1.02e-01 | 6.6e-03 | same | 4/5 | 176/190 |

W8A8 keeps the top-5 set and the top-1 label on both inputs at 3.4x
compression. W4A8 reorders the top-1 on the noise input (`Static` above
`White noise`, two classes whose reference scores differ by only 0.079), so on
a 5.5M-parameter model 4-bit weights are visibly lossy where 8-bit are not.

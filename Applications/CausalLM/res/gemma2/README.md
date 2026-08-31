# Gemma2-2B → nntrainer QS4CX-FP16

Conversion pipeline for the Gemma2-2B model used in the 1K GPU benchmark
(`gemma2_lg_q6k`). The on-disk recipe is **embedding Q6_K, FC QS4CX, lm_head
Q6_K, FP16 activations** (`model_tensor_type: QS4CX-FP16`).

> **int4 = QS4CX.** QS4CX (per-channel int4: plain nibbles + fp32 scales) is the
> canonical int4 weight format. The legacy `QINT4` on-disk record is still
> readable — transcoded losslessly to QS4CX at load time (compat shim) — so
> existing `QINT4-FP16` `.bin` files keep working, but new models should be
> built as QS4CX.

## Files

- `weight_converter.py` — numpy-only HF Gemma2-2B → nntrainer layer-graph `.bin`.
  Step 1 of the pipeline: produces an FP32-weight / FP16-norm source for
  `nntr_quantize`. No torch/transformers needed (reads `model.safetensors`
  directly, bf16→fp32 exact widen).
- `build_qs4cx.sh` — end-to-end HF → QS4CX-FP16 model (runs the converter, then
  `nntr_quantize` with the benchmark recipe).

## Quick start

```bash
# builds nntr_quantize if needed: ninja -C build Applications/CausalLM/nntr_quantize
./build_qs4cx.sh /path/to/hf/gemma2-2b /path/to/out_gemma2_qs4cx
```

## Manual (2 steps)

```bash
# 1) HF -> FP32 weights + FP16 norms layer-graph .bin (quantize source)
python3 weight_converter.py \
  --model_path /path/to/hf/gemma2-2b \
  --output_name stage/nntr_gemma2_2b_mixed.bin \
  --data_type float32 --norm_fp16
#    (place config.json + tokenizer*.json + a source nntr_config.json with
#     model_tensor_type "FP32-FP16", fc/embedding FP32, in the stage/ dir —
#     see build_qs4cx.sh for the exact stage nntr_config.json)

# 2) quantize to the benchmark recipe (QS4CX FC, Q6_K embedding, Q6_K lm_head)
nntr_quantize stage \
  --fc_dtype QS4CX --embd_dtype Q6_K --lmhead_dtype Q6_K \
  -o out_gemma2_qs4cx
```

`nntr_quantize` writes the output `.bin` plus an `nntr_config.json` with
`model_tensor_type: QS4CX-FP16`, `fc_layer_dtype: QS4CX`, `embedding_dtype:
Q6_K`, `lmhead_dtype: Q6_K` — matching the tested `gemma2_lg_q6k` model. Set its
`tokenizer_file` to the deployed `tokenizer.json` path before running.

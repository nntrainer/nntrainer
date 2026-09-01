# Gemma4-E2B → nntrainer QS4CX-FP16

Conversion pipeline for the Gemma4-E2B model used in the 1K GPU benchmark
(`gemma4_e2b_qint4fp16_lmint4`). The recipe is **embedding Q6_K, FC QS4CX,
lm_head QS4CX (untied), FP16 activations, skip_prefill**
(`model_tensor_type: QS4CX-FP16`).

> **int4 = QS4CX.** QS4CX (per-channel int4: plain nibbles + fp32 scales) is the
> canonical int4 weight format. The legacy `QINT4` on-disk record is still
> readable — it is transcoded losslessly to QS4CX at load time (compat shim) —
> so existing `QINT4-FP16` `.bin` files keep working unchanged, but new models
> should be built as QS4CX.

## Files

- `weight_converter.py` — HF Gemma4 (text model) → nntrainer `.bin` / safetensors.
  With `--norm_dtype float16 --data_type float32` it writes an FP32-weight /
  FP16-norm source for `nntr_quantize`. Gemma4 always re-emits a dedicated
  `output_of_causallm` (lm_head) slot — even when `tie_word_embeddings` is true
  it writes a copy of the embedding — so the lm_head can be quantized to a
  different dtype than the embedding (untied QS4CX lm_head).
- `build_qs4cx.sh` — end-to-end HF → QS4CX-FP16 model.
- `nntr_config.json` — sample runtime config.

## Quick start

```bash
# builds nntr_quantize if needed: ninja -C build Applications/CausalLM/nntr_quantize
./build_qs4cx.sh /path/to/hf/gemma4-e2b /path/to/out_gemma4_qs4cx
```

## Manual (2 steps)

```bash
# 1) HF -> FP32 weights + FP16 norms .bin (quantize source)
python3 weight_converter.py \
  --model_path /path/to/hf/gemma4-e2b \
  --output_name stage/nntr_gemma4_fp32fp16.bin \
  --data_type float32 --norm_dtype float16
#    (place config.json + tokenizer*.json + a source nntr_config.json with
#     model_tensor_type "FP32-FP16", fc/embedding/lmhead FP32, lmhead_untie true,
#     skip_prefill true, in the stage/ dir — see build_qs4cx.sh)

# 2) quantize: FC QS4CX, embedding Q6_K, lm_head QS4CX (untied)
nntr_quantize stage \
  --fc_dtype QS4CX --embd_dtype Q6_K --lmhead_dtype QS4CX \
  -o out_gemma4_qs4cx
```

`nntr_quantize` writes the output `.bin` plus an `nntr_config.json` with
`model_tensor_type: QS4CX-FP16`, `fc_layer_dtype: QS4CX`, `embedding_dtype:
Q6_K`, `lmhead_dtype: QS4CX` — matching the tested `gemma4_e2b_qint4fp16_lmint4`
model. Set `tokenizer_file` to the deployed `tokenizer.json` path. `init_seq_len`
/ `max_seq_len` are deployment knobs (raise `init_seq_len` to 1024 for an
uncapped 1K prefill).

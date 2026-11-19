# PaliGemma differential validation

The `nntr_paligemma_e2e` driver compares image preprocessing, 256 projected
image embeddings, every last-prompt-row logit, and the complete greedy token
sequence with Hugging Face/PyTorch. It checks both the bidirectional prefix
and the causal suffix described by `token_type_ids.json`.

Generate a deterministic FP32 reference from the downloaded checkpoint:

```sh
python Applications/CausalLM/res/paligemma/e2e/golden_dump.py \
  --model_dir ~/Downloads/paligemma --out_dir /tmp/paligemma-reference
```

The image backend is explicitly `pil`. C++ antialiased bicubic preprocessing
matches that backend; Transformers' torchvision backend can differ by one
8-bit pixel level and should not be silently mixed into this comparison.

For an FP32 vision + FP16 decoder reference on PyTorch CUDA, add
`--device cuda --text_dtype float16`. The NNTrainer decoder package must
use the same dtype: `model_tensor_type=FP16-FP16`, `fc_layer_dtype=FP16`,
`lmhead_dtype=FP16`, with the standalone embedding table kept FP32.
`weight_converter.py --dtype fp16` creates the FP16 bypass layout;
`--connected` selects the token-ID layout instead. The default converter
layout matches `use_embedding=true`.

Build and validate both engines:

```sh
ninja -C build_cuda Applications/CausalLM/nntr_paligemma_e2e
export LD_LIBRARY_PATH="$HOME/cuda-toolkit/lib64:${LD_LIBRARY_PATH:-}"
python tools/validate_vla_ports.py --out-dir /tmp/paligemma-results \
  paligemma-cuda cuda build_cuda/Applications/CausalLM/nntr_paligemma_e2e \
  /path/to/text-model /path/to/vision-model /tmp/paligemma-reference
```

Use `cpu` instead of `cuda` for the CPU run. The reference and model package
must agree on precision. Full-precision comparisons use atol 0.01 and rtol
0.001 for logits; FP16 comparisons use atol 0.1 and the same rtol. Raw errors,
RMSE, token counts, inference timings and process/device memory are recorded.
A mismatch exits nonzero; failures are not hidden by shell pipelines.

`--suffix_tokens 3` generates a mixed prefix/causal-suffix case. Change
`--mm_prompt`, `--image_seed`, and `--n_gen` to cover other inputs. Passing
`--preprocess-only` as the driver's fourth argument checks the image loader
without loading weights. Preprocessed FP32 CHW buffers remain supported.

The optimized dense checkpoint profile uses `--weight-transposed` in both
language and vision converters and `weight_transposed=true` in both configs.
This stores dense matrices in PyTorch `[output,input]` order without changing
their values. The language config can enable `fuse_mlp=true`; CPU FP32 also
fuses Q/K/V by default with that setting (`fuse_qkv` overrides it). CUDA
retains separate Q/K/V projections. Quantized and existing checkpoint layouts
keep their original paths.

For CPU FP32 PaliGemma with a tied head and the embedding bypass, the language
converter also accepts `--cpu-external-head`: set `cpu_external_head=true`
to use the embedding table directly for vocabulary projection. This package
omits the duplicate final head record and is specific to CPU FP32 inference.

`NNTR_BENCH_REPEATS` repeats complete requests on one loaded model. Set
`NNTR_VALIDATE_ALTERNATE_REFERENCE` to another reference directory to change
the image and prompt (and pi0.5 noise) on odd requests. Every request is checked;
this is useful for finding stale cache and graph replay errors. CUDA uses
pedantic FP32 GEMMs and places managed dense weights during model loading.
See the [optimization report](../../../../../docs/validation/paligemma-pi05-optimization-20261113.md)
for measured profiles, PyTorch comparisons and exact reproduction commands.

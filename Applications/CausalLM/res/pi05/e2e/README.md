# pi0.5 differential validation

Generate the reference with the upstream openpi PyTorch implementation and
`lerobot/pi05_base` weights:

```sh
python Applications/CausalLM/res/pi05/e2e/golden_dump.py \
  --openpi_dir /path/to/openpi \
  --checkpoint /path/to/pi05-base-hf/model.safetensors \
  --tokenizer /path/to/paligemma/tokenizer.json \
  --out_dir /tmp/pi05-reference
```

Use openpi's supported Transformers replacement modules. The reference creates
`Pi0Config(dtype="float32", pytorch_compile_mode=None)` before constructing
`PI0Pytorch`. Constructing the default BF16 model and then calling `.float()`
leaves the RoPE frequency buffer rounded to BF16 and changes the reference.
Missing weights are rejected; only tied weights with demonstrably shared
storage are restored before strict loading.

The driver checks all projected vision values, all 16,000 per-step velocity
values, and all 1,600 final action values for a 50x32, ten-step observation:

```sh
ninja -C build_cuda Applications/CausalLM/nntr_pi05_e2e
export LD_LIBRARY_PATH="$HOME/cuda-toolkit/lib64:${LD_LIBRARY_PATH:-}"
python tools/validate_vla_ports.py --out-dir /tmp/pi05-results \
  pi05-cuda cuda build_cuda/Applications/CausalLM/nntr_pi05_e2e \
  /path/to/pi05-model /path/to/pi05-vision /tmp/pi05-reference
```

Use `cpu` to compare the CPU BLAS path. CUDA uses resident FP32 expert weights,
cuBLAS projections and device normalization/attention/residual kernels.
The backbone may use FP16 or FP32 weights/activations. Vision attention and
exported prefix K/V retain FP32 precision; no unused vocabulary head runs in
the backbone. `weight_converter.py --backbone_dtype fp16` exports the compact
backbone without a vocabulary head. Embeddings, vision and expert weights
remain FP32. Existing packages with the old unused head tail can still load.

The checks retain absolute tolerances 0.005 for velocities and 0.01 for actions.
Use `--prompt` and `--seed` to generate other observations. The synthetic
observation uses one camera and fixed initial noise, so this validates model
math and execution rather than a robot's closed-loop behavior. The low-level
`sampleActions` API accepts preprocessed image embeddings and tokenized prompts.

`NNTR_PI05_DIAGNOSTICS=/existing/directory` records FP32 prefix K/V, first-step
conditioning, first expert-layer states and velocities for detailed comparison.

The optimized dense checkpoint profile uses `--weight-transposed` in both
language and vision converters and `weight_transposed=true` in both configs.
This stores dense matrices in PyTorch `[output,input]` order without changing
their values. The language config can enable `fuse_mlp=true`; CPU FP32 also
fuses Q/K/V by default with that setting (`fuse_qkv` overrides it). CUDA
retains separate Q/K/V projections. Quantized and existing checkpoint layouts
keep their original paths.

`NNTR_BENCH_REPEATS` repeats complete requests on one loaded model. Set
`NNTR_VALIDATE_ALTERNATE_REFERENCE` to another reference directory to change
the image and prompt (and pi0.5 noise) on odd requests. Every request is checked;
this is useful for finding stale cache and graph replay errors. CUDA uses
pedantic FP32 GEMMs and places managed dense weights during model loading.
See the [optimization report](../../../../../docs/validation/paligemma-pi05-optimization-20261113.md)
for measured profiles, PyTorch comparisons and exact reproduction commands.

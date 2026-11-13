# PaliGemma / π0.5 Ubuntu CUDA validation — 2026-11-13

Both ports pass the tested CPU and CUDA numerical checks after corrections.
NNTrainer CUDA runs faster than its CPU path, but PyTorch inference remains
faster in every measured framework/device pair. This report covers one active
camera, batch size one and two synthetic observations per model; it does not
establish correctness for every input or a robot's closed-loop behavior.

## Follow-up implementation audit

The [implementation audit](paligemma-pi05-implementation-audit-20261113.md)
finds that model execution and common backend integration are not yet optimal.
The original first-request/peak-loading measurements below remain historical
results. In particular, the π0.5 peak RAM difference is not evidence of the
same steady-state memory advantage; the follow-up reports warm host RSS and
controlled existing-kernel/model-mapping experiments separately.

## Hardware and measurement

Ubuntu 22.04.5 LTS, Intel Core i9-14900, NVIDIA GeForce RTX 3050 OEM 8 GiB,
driver 595.99.02, CUDA 13.2. NNTrainer: GCC release build, optimization 3,
transformer enabled; CUDA build also enables FP16.

The table uses the median of three separate processes, one request per process
after model loading. The measured request includes vision and complete text
or action inference, with CUDA synchronization. Model loading, input-file I/O,
reference comparison and destruction are outside this interval. First-use
runtime costs remain included. All three requests validate their outputs.

CPU uses FP32 and eight threads in both frameworks. GPU uses FP32 vision,
FP16 language weights/activations and four host threads; π0.5's expert and
flow heads remain FP32. Language/vision/expert compute precision is matched
within each CPU/GPU framework pair. PaliGemma NNTrainer retains a separate
FP32 host lookup table; Hugging Face shares its FP16 CUDA embedding/head
weights. Comparing CPU with GPU also changes language precision.
PyTorch runs in eval/inference mode without torch.compile: PaliGemma uses
Transformers 5.14.1 SDPA; π0.5 uses openpi's Transformers 4.53.2 replacements,
SDPA vision, and upstream eager backbone/expert. Both use CUDA PyTorch
2.13.0+cu132. Float32 matmul precision is highest.

| Model | Device | Framework | Inference seconds | Peak RAM GiB | Peak VRAM increment GiB |
|---|---|---|---:|---:|---:|
| PaliGemma 3B | CPU | NNTrainer | 7.243 | 13.25 | — |
| PaliGemma 3B | CPU | PyTorch | 3.243 | 11.83 | — |
| PaliGemma 3B | CUDA | NNTrainer | 3.771 | 8.61 | 6.54 |
| PaliGemma 3B | CUDA | PyTorch | 0.952 | 12.29 | 6.39 |
| π0.5 3B | CPU | NNTrainer | 9.756 | 12.95 | — |
| π0.5 3B | CPU | PyTorch | 2.815 | 27.74 | — |
| π0.5 3B | CUDA | NNTrainer | 3.774 | 9.30 | 7.03 |
| π0.5 3B | CUDA | PyTorch | 0.581 | 27.74 | 7.08 |

RAM is peak process RSS, including weight loading and file-backed mappings.
In particular, PyTorch π0.5's ordinary model construction plus strict checkpoint
loading creates a transient host-memory peak; this is not its steady inference
RAM. VRAM is the maximum sampled whole-device usage minus that process's idle
baseline, including managed-memory residency. Sampling is approximately 100 ms
plus query overhead, so short peaks may be missed. Desktop baseline varied
between roughly 0.45 and 0.62 GiB. Per-process NVML memory omits much of
NNTrainer's managed allocation and is not used as its VRAM figure. RAM and
VRAM can overlap and must not be added.

PaliGemma processes 272 prompt rows and produces exactly 11 greedy tokens,
including EOS. π0.5 processes a 261-row valid prefix and performs ten Euler
steps for a 50×32 action chunk. Masked trailing language padding is removed
in both π0.5 benchmarks. Its FP32 lookup table stays on CPU in both GPU paths;
PyTorch converts the FP16 prefix cache to FP32 for its FP32 expert. Unused
vocabulary heads are removed. PyTorch CUDA uses
`PYTORCH_ALLOC_CONF=expandable_segments:True`; its default allocator ran out
of memory during π0.5 model transfer on this card.

## Numerical verification

- PaliGemma FP32 CPU and CUDA: both base and mixed prefix/causal-suffix
  observations pass all prefill logits and complete greedy sequences (11/11
  and 8/8 tokens). FP32 logits use atol 0.01 + rtol 0.001 × |reference|.
- PaliGemma FP16 CUDA: both observations pass; maximum prefill logit errors
  are 0.101562 and 0.046875. The FP16 check is atol 0.1 + rtol 0.001 ×
  |reference|, not a pure 0.1 absolute threshold. Generation matches complete
  PyTorch CUDA FP16 token sequences. CPU/GPU PyTorch SDPA benchmarks also
  produce the exact expected 11-token sequence.
- Image preprocessing: all 150,528 float values match the explicit HF PIL
  backend exactly for six source sizes: 336×336, 41×33, 1×1, 224×224,
  897×531 and 129×783. Torchvision and PIL resizing are not silently mixed.
- π0.5 FP32 CUDA versus upstream FP32 PyTorch: maximum velocity error
  5.75185e-6 and action error 1.72108e-6 in the first observation; the second
  also passes. Every one of 16,000 velocities and 1,600 actions is checked.
- π0.5 optimized CUDA versus FP32 PyTorch: the three benchmark runs have
  maximum velocity error at most 0.00346035 and action error at most
  0.000653088. The second observation also passes. Absolute tolerances remain
  0.005 for velocity and 0.01 for actions.
- Direct CUDA-to-CUDA π0.5 comparison: 16,000 velocities have maximum error
  0.00192830, and 1,600 final actions have maximum error 0.000426657; no
  nonfinite values or tolerance violations. Full arrays are retained.

Full FP32 CUDA remains numerically valid, but the models exceed this card's
comfortable resident capacity. Diagnostic first-observation inference times
were 30.111 s for PaliGemma and 8.958 s for π0.5. These single diagnostic runs
are not the three-repeat FP16 deployment results in the table. PaliGemma's
FP32 GPU profile pages managed weights and is slower than its CPU profile.

## Corrections and remaining performance limits

1. Route new model compute nodes through the selected backend; enable existing
   FP32 CUDA LayerNorm and GELU paths. Keep host-required graph boundaries
   readable, and explicitly configure the CUDA model profile.
2. Use the existing named incremental-inference wrapper for cache binding.
   Remove the positional cache-order workaround and the added common public
   cache API. π0.5 exports FP32 post-RoPE prefix K/V directly.
3. Project only the final PaliGemma row to vocabulary logits. Correct its CUDA
   argmax row, FP16 embedding input, generation budget/EOS behavior and resumed
   cache-capacity guard. Fuse the existing GeGLU graph operation.
4. Honor noncausal CUDA attention. Use materialized FP16 probabilities for
   noncausal single-query decode, matching PyTorch eager rounding; an online
   FP32-probability path changed the third token in the extra observation.
   The CUDA tiny regression now checks this rounding and full-cache visibility.
5. Use FP32 cuBLAS attention for vision/prefix math, including a direct
   interleaved device path when its tensors permit it. The π0.5 expert keeps
   FP32 weights resident on CUDA and uses device projection, normalization,
   attention, rotary, residual and MLP operations. CPU projections use BLAS.
6. Fix the π0.5 reference itself: construct `Pi0Config(dtype="float32",
   pytorch_compile_mode=None)` before the model. Constructing BF16 and calling
   `.float()` leaves rounded RoPE frequencies. Strict checkpoint loading and
   tied-storage alias verification replace permissive loading. The previous
   roughly 0.00497 action-error report used that flawed reference.
7. Strengthen differential drivers, converters, preprocessing, failure exit
   codes, fixture packaging and reproducible timing/memory tools.

CUDA execution still crosses host graph boundaries, including patch embedding,
shape handling and exported prefix data; some generic operations use coherent
host fallbacks. Host/device traffic and graph overhead remain performance
limits. The table does not claim that the entire request stays on GPU or that
NNTrainer matches PyTorch performance.

## Reproduce on this machine

Validated model profiles are in
`.validation/paligemma-pi05-20261113/models/{paligemma,pi05}-{fp32,fp16}`;
vision profiles are linked alongside them. Original downloaded checkpoints
and existing FP32 model binaries were preserved. The new FP16 decoder files
are stored in the workspace rather than /tmp.

```sh
export LD_LIBRARY_PATH="$HOME/cuda-toolkit/lib64:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
VLA_ROOT="$PWD/.validation/paligemma-pi05-20261113"
python3 tools/validate_vla_ports.py --out-dir /tmp/vla-repeat --threads 4 \
  paligemma-cuda cuda build_cuda/Applications/CausalLM/nntr_paligemma_e2e \
  "$VLA_ROOT/models/paligemma-fp16" "$VLA_ROOT/models/paligemma-vision" \
  "$VLA_ROOT/results/references/paligemma-pytorch-cuda-reference"
python3 tools/validate_vla_ports.py --out-dir /tmp/vla-repeat --threads 4 \
  pi05-pytorch-cuda cuda "$VLA_ROOT/pytorch-cuda-env/bin/python" -- \
  tools/benchmark_vla_pytorch.py pi05 cuda --threads 4 --attention sdpa \
  --weights "$HOME/pi05-base-hf/model.safetensors" \
  --reference "$VLA_ROOT/results/references/pi05-reference-fp32"
```

For CPU use `cpu`, eight threads and the FP32 model/reference profile. The
complete per-run commands, all 24 benchmark logs/JSON files, complete reference
outputs, preprocessing results, build settings, source/binary SHA-256 hashes
and direct GPU arrays live under the local results directory. The π0.5 CUDA
Python environment was created inside the ignored validation directory; the
existing Python environments were not modified. Source provenance records
checkpoint sizes and safetensors-header hashes, not full checkpoint hashes.

## Regression status and scope

- CPU model suite: 103 tests, 102 pass, one existing Gemma4 QINT4 skip: its
  packing requires ARM NEON and has no x86 CPU dequant path. No failures.
- CUDA VLA suite: 3/3 pass, including repeated π0.5 inference and FP16
  noncausal single-query rounding. CUDA GELU and LayerNorm: 4/4 pass.
- clang-format 14, Python compilation and git diff whitespace checks pass.
  The fixture archive matches all 112 PaliGemma/π0.5 fixture files.
- Separate Gemma4 CUDA FP32 differential smoke test fails a greedy-prefix
  threshold (2 matched tokens versus 4 required). The same failure reproduces
  with all three changed common CUDA/MHA translation units rebuilt from HEAD
  into isolated baseline libraries. It is recorded as an existing failure,
  not counted as a passing full CUDA regression suite.
- No Android/Windows/Tizen/Yocto build matrix or closed-loop robot test was
  run. The action API consumes preprocessed vision embeddings/tokenized
  prompts; policy-level state tokenization and action unnormalization are
  outside this validation. Changes remain uncommitted.

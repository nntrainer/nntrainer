# LoRA Training for Qwen3 CausalLM

This document describes the LoRA (Low-Rank Adaptation) fine-tuning support
added to the CausalLM application, and the commit structure used to land it.

## Background

An earlier fork of nntrainer had backprop and LoRA training wired up for
Qwen3, but the mainline CausalLM layer stack (`mha_core`, `rms_norm`,
`swiglu`, `tie_word_embedding`, `lm_head`) had since been substantially
rewritten (in particular, attention moved to an externally-owned
`KVCacheManager` with absolute cache-index addressing). The old training
code could not be reapplied as a patch; the backward math had to be
re-derived against the current forward implementations.

Scope for this change: **plain FP32 LoRA training on Qwen3-0.6B (dense,
causal, no sliding window/sink/softcapping)**. Quantization-aware LoRA
training (Q4_0/Q6_K/W4A8) is a deliberately separate, later effort.

## What was added

- **Backward math** (`calcDerivative`, and `calcGradient` where the layer has
  weights) for `rms_norm`, `reshaped_rms_norm`, `swiglu`, `mha_core`,
  `lm_head`, and `tie_word_embedding` — none of these supported training
  before this change.
- **A fix to a pre-existing crash** in `nntrainer/layers/fc_layer.cpp`:
  `incremental_forwarding` read LoRA scratch tensors that are never allocated
  in an inference-only (no-backward) compiled graph, independent of anything
  else here.
- **LoRA initialization order swap** (also in `fc_layer.cpp`): mainline
  initialized `loraA=zeros, loraB=random`; standard LoRA practice (and what
  training stability requires) is the reverse — see the commit for the
  derivation of why the original order is badly conditioned.
- **LoRA config threading**: `lora_rank` / `lora_alpha` / `lora_target` (and
  the new `lora_clip_grad_by_norm`, `lora_train_norms`) read from
  `nntr_config.json`, applied to the seven FC projections per decoder block,
  with every non-adapted layer explicitly frozen.
- **A training entrypoint** on `Transformer`: `initializeForTraining()`,
  `train()`, `save_weight_lora()` / `load_weight_lora()` (loads a pretrained
  non-LoRA checkpoint into a graph that has LoRA weights, by diffing weight
  names against a throwaway reference graph — see the commit for why this
  has to match the *inference* topology exactly).
- **A training data pipeline and CLI** (`nntr_lora_train`): tokenizes a text
  file (plain or chat-format), right-pads to a fixed sequence length, and
  tracks the true last-token position per sample so the LM head reads the
  right row instead of a pad slot.
- **Inference-side adapter loading** in `main.cpp` (`lora_file_name` in
  `nntr_config.json`).

## Verification performed

- 13 unit tests (`unittest_lora_backward_gradcheck`): finite-difference
  gradient checks for every `calcDerivative`/`calcGradient` implementation
  above, plus a dedicated regression test for a tensor-aliasing hazard (see
  below) and a check that the training-mode attention forward agrees with
  the trusted inference prefill path.
- 3 end-to-end tests (`unittest_qwen3_lora_training_smoke`): a tiny
  synthetic Qwen3 graph compiles for training with LoRA enabled, the
  trainable/frozen split matches the freeze policy, and a few training steps
  followed by `save_weight_lora`/`load_weight_lora` round-trips correctly.
- All pre-existing tests continue to pass (67 CausalLM model tests, 17
  KV-cache tests, 10 embedding-sidecar tests, etc. — no regressions).
- Real-model validation on Qwen3-0.6B: trained on identity/SST-2/LaMP-3
  data, loss decreases correctly, saved adapters scanned byte-for-byte for
  NaN/Inf (clean), and a full train → save → load → generate round trip
  produces coherent, adapter-influenced output (verified across an LR
  sweep: gentle LR ≈ unchanged from base, `1e-4` picks up trained
  vocabulary while staying fluent, `1e-3` overfits a 20-sample set into
  repetition — the expected LoRA failure mode, not a bug).

### A note on the tensor-aliasing bug found during validation

nntrainer recycles an input's storage to hold its own gradient (i.e.
`calcDerivative`'s output tensor for input *i* is the *same buffer* as input
*i*). The first version of `SwiGLULayer::calcDerivative` and
`MHACoreLayer::calcDerivative` wrote to one output before finishing reading
the input it aliased, silently corrupting the value and producing NaN
gradients after the first training step. This was caught by validating an
actual training run rather than trusting the gradient-check tests, which
used independently-allocated buffers and so never exercised the aliasing.
Both layers were fixed (snapshot-before-write) and a dedicated regression
test (`SwiGLUCalcDerivativeIsSafeWhenGradAliasesInput`) was added that
reconstructs the aliasing directly.

## Commit structure

Nothing was committed while this work was in progress; the sequence below
is how the final state is organized into reviewable, individually-buildable
commits. Commits are ordered so each one leaves the tree in a working state
(compiles, and any test it introduces passes at that point in history).

1. **`[fc_layer] fix incremental_forwarding crash for inference-only LoRA`**
   `nntrainer/layers/fc_layer.cpp` (the crash-fix hunk only)
2. **`[fc_layer] use standard LoRA init ordering (A=random, B=zero)`**
   `nntrainer/layers/fc_layer.cpp` (the initializer-swap hunk)
3. **`[CausalLM] add backprop (derivative + gradient) for rms_norm and reshaped_rms_norm`**
   `Applications/CausalLM/layers/{rms_norm,reshaped_rms_norm}.{cpp,h}` +
   the corresponding gradient-check tests + meson wiring
4. **`[CausalLM] add backprop for swiglu`**
   `Applications/CausalLM/layers/swiglu.{cpp,h}` + its tests (including the
   aliasing regression test) + meson wiring
5. **`[CausalLM] add backprop for lm_head and tie_word_embedding`**
   `Applications/CausalLM/layers/{lm_head,tie_word_embedding}.{cpp,h}` +
   their tests + meson wiring
6. **`[CausalLM] add LoRA training forward/backward for mha_core`**
   `Applications/CausalLM/layers/mha_core.{cpp,h}` + its tests (including
   the training-forward-vs-inference-prefill check) + meson wiring
7. **`[CausalLM] thread LoRA config and add a training entrypoint to Transformer`**
   `Applications/CausalLM/models/{transformer.{cpp,h},qwen3/qwen3_causallm.cpp,causal_lm.cpp}`
   + `unittest_qwen3_lora_training_smoke.cpp` + meson wiring
8. **`[CausalLM] add a LoRA training data pipeline and CLI driver`**
   `Applications/CausalLM/{lora_train.{cpp,h},train_qwen3_lora.cpp}` +
   meson wiring for `nntr_lora_train`
9. **`[CausalLM] load LoRA adapters at inference time`**
   `Applications/CausalLM/main.cpp`

## Known limitations / follow-ups

- `mha_core`'s training path is plain scalar C++ (chosen for verifiability
  over speed while the math was unproven); it is single-threaded and not
  SIMD, unlike the inference path. Training throughput on a dataset the
  size of SST-2 (67k samples) would take days, not hours, until this is
  optimized.
- `calcGradient` for the norm layers is unit-tested but was only exercised
  in a real run once (`--train_norms` / `lora_train_norms`); under the
  default LoRA-only recipe it is never called (those layers are frozen).
- Quantization-aware LoRA (Q4_0/Q6_K/W4A8) is out of scope for this change.
- `mha_core`'s training forward explicitly rejects sliding-window,
  attention-sink, and logit-softcapping configurations (throws rather than
  silently computing them wrong) — correct for Qwen3-dense, but any of the
  other CausalLM architectures using those features would need the training
  path extended first.

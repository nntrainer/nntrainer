# LoRA Training for Qwen3 CausalLM

This document is a full account of the LoRA (Low-Rank Adaptation) fine-tuning
support added to the CausalLM application: why it was needed, what was built,
every bug found along the way (especially the NaN investigation), how each
was diagnosed and fixed, how it was all verified, and how to use it. It is
meant to be exhaustive enough that someone who wasn't in the room can
understand not just *what* changed but *why*, and can reproduce the
verification independently.

## Table of contents

- [Background](#background)
- [Scope](#scope)
- [Architecture: what was added, and why it works this way](#architecture-what-was-added-and-why-it-works-this-way)
- [The NaN investigation](#the-nan-investigation)
- [The LoRA initialization order bug](#the-lora-initialization-order-bug)
- [Was it gradient clipping?](#was-it-gradient-clipping)
- [Two memory bugs](#two-memory-bugs)
- [A silent-wrong-answer bug in validation](#a-silent-wrong-answer-bug-in-validation)
- [Verification performed](#verification-performed)
- [How to train](#how-to-train)
- [Commit structure](#commit-structure)
- [Known limitations / follow-ups](#known-limitations--follow-ups)

## Background

An earlier fork of nntrainer had backprop and LoRA training wired up for
Qwen3. By the time this work started, mainline had moved 344 commits past the
fork point, and the exact files the fork touched had been substantially
rewritten — most importantly, attention (`mha_core`) had moved from an
internal, self-managed KV cache to an externally-owned `KVCacheManager` with
absolute cache-index addressing. The fork's training code assumed the old
internal-cache-only design and could not be reapplied as a patch; the
backward math had to be re-derived from scratch against the *current* forward
implementations, not ported.

Before this change, the CausalLM application was inference-only end to end:
`calcDerivative`/`calcGradient` were either absent or empty stubs on every
layer in the Qwen3 stack, and `NeuralNetwork::initialize()` as called from
`Transformer` hardcoded `ExecutionMode::INFERENCE`.

## Scope

Plain FP32 LoRA training on Qwen3-0.6B: dense, causal attention, no sliding
window, no attention sink, no logit softcapping. Quantization-aware LoRA
(Q4_0 / Q6_K / W4A8) is a deliberately separate, later effort — the training
math here assumes full-precision weights throughout.

## Architecture: what was added, and why it works this way

### Backward math for six layers

None of these supported training before this change:

**`rms_norm.cpp` / `reshaped_rms_norm.cpp`** — standard RMSNorm backward:

```
dx      = inv_rms * (gamma*dy - x * mean(gamma*dy*x) * inv_rms^2)
dgamma  = sum_over_rows(dy * x * inv_rms)
```

`inv_rms` is cached in an `ITERATION_LIFESPAN` tensor during the forward pass
so `calcDerivative` doesn't recompute the norm from scratch. `gamma` is
frozen under the default LoRA-only recipe (it's never a LoRA target), so
`calcGradient` here only fires when `lora_train_norms` opts into training the
norms alongside the adapters. `reshaped_rms_norm` applies the same math
per `feature_size` chunk (used for Qwen3's per-head `q_norm`/`k_norm`) and is
a no-op when `use_gamma` is false.

**`swiglu.cpp`** — SwiGLU backward:

```
Swish(g) = g * sigmoid(g)
d_up     = Swish(gate) * dy
d_gate   = Swish'(gate) * up * dy
```

No weights, so no `calcGradient`. This is the layer where the [aliasing bug](#the-nan-investigation)
was first found.

**`lm_head.cpp` / `tie_word_embedding.cpp`** — both need to handle
right-padded training batches, where a sample's last *real* token isn't
necessarily the last row of the sequence (only that row's logits actually
feed the loss; every other row would just add noise gradient). A module-level
global (`g_lm_head_read_row` / `g_tie_embedding_lm_head_read_row`, sentinel
`UINT_MAX` meaning "use `height - 1`") tells `forwarding()`/`calcDerivative()`
which row to project to vocab logits. `tie_word_embedding` derives this value
itself, inside its own `forwarding()` embedding-mode branch, by scanning
token ids for the last non-pad position — see
[the thread_local bug](#bug-1-the-thread_local-thread-visibility-bug) for why
that design replaced an earlier one.

`calcGradient`:
  - `lm_head`: `dL/dW` and `dL/dbias` accumulate only from the read row.
  - `tie_word_embedding`: embedding mode scatter-adds `dy * scale` into the
    row named by each token id (repeated ids accumulate, matching the
    embedding lookup's read semantics); `lm_head` mode accumulates the same
    outer-product gradient as `lm_head`. Both are gated on
    `context.isGradientFirstAccess(...)` because the weight is *shared*
    between the two mode instances (tied embeddings) — without that gate,
    whichever mode's `calcGradient` runs second would silently double-count
    or overwrite the other's contribution.

**`mha_core.cpp`** — training forward reuses the *existing* internal-cache
machinery (`incremental_forwarding` over the full sequence, `from=0` to
`to=seq_len`) rather than reimplementing QKᵀ/softmax/·V from scratch, so
training numerics are provably the same code path as the trusted inference
prefill path. This is checked directly by a dedicated test
(`MhaCoreTrainForwardMatchesInferencePrefill`), not just assumed.
`calcDerivative` caches RoPE'd Q/K, raw V, and the softmax attention weights
during the forward pass (new tensors sized off `query_dim.height()`, i.e. the
actual sequence length — see [the memory bugs](#two-memory-bugs) for why this
matters), then computes: masked-causal `d_attn = dy·Vᵀ`, GQA-grouped `dV`,
softmax backward, `dQ`/`dK` through the cached attention weights, and finally
an inverse RoPE rotation on the `dQ`/`dK` gradients using the same cached
rotation angles the forward pass used. `mha_core` has no learnable weights of
its own (Q/K/V/O are separate FC layers), so `calcGradient` stays a no-op.

This layer's training path is plain scalar C++, not SIMD or multi-threaded
like the inference path — see [Known limitations](#known-limitations--follow-ups).

### A pre-existing crash fix (unrelated to training)

`nntrainer/layers/fc_layer.cpp`'s `incremental_forwarding()` unconditionally
read the LoRA scratch tensors `loraTmp`/`loraOut` via `context.getSharedDataTensor(...)`.
Those tensors use `FORWARD_GRAD_LIFESPAN`/`FORWARD_FUNC_LIFESPAN` and are
never allocated in a pure-inference (no-backward) compiled graph — so calling
existing mainline LoRA inference support (`lora_rank` was already a
recognized property before this work) with incremental decoding would crash.
Fixed by computing the LoRA scratch math into freshly-allocated local
`Tensor` objects instead of pulling from possibly-null context storage. This
is a standalone fix, independent of everything else in this document.

### LoRA config threading

`lora_rank` / `lora_alpha` / `lora_target` were already being read out of
`nntr_config.json` into unused fields; two more were added:
`lora_clip_grad_by_norm` and `lora_train_norms`. These are threaded through
every FC projection construction call in `Transformer::createAttention`/
`createMlp` and `Qwen3Transformer::createAttention`: a targeted layer gets
`lora_rank`/`lora_alpha` properties (and `clip_grad_by_norm` if set);
everything else gets `trainable=false` whenever LoRA is active. That
includes `lm_head` (never a LoRA target — see `causal_lm.cpp`) and, unless
`lora_train_norms` is set, every RMSNorm/`reshaped_rms_norm` instance
(attention norms, FFN norms, output norm, and Qwen3's per-head
`q_norm`/`k_norm`).

### Training entrypoint

`Transformer::initializeForTraining(lr, epochs)` builds the same model graph
as inference but in `ExecutionMode::TRAIN`, appends a loss layer, and sets up
optimizer/batch/epoch properties; `train()`/`getTrainingStats()`/
`getValidStats()`/`setDataset()` are thin wrappers around the underlying
`NeuralNetwork` calls. `save_weight_lora()`/`load_weight_lora()` handle the
adapter file format — `load_weight_lora()` in particular loads a pretrained
*non-LoRA* checkpoint into a graph that *has* LoRA weights by diffing weight
names against a throwaway reference graph built with the exact same layer
construction calls as the real graph (so the two topologies can never
desync). Base-model weights are collected as **borrowed pointers** into that
reference graph, not clones — see [the memory bugs](#two-memory-bugs).

### Training data pipeline and CLI

`lora_train.cpp`/`.h` (`TrainingDataGenerator`) tokenizes a text file
(plain-text or chat-format, auto-detected), right-pads every sample to a
fixed sequence length, shuffles epochs with a seeded `mt19937`, and tracks
each sample's true last-token position for the read-row mechanism above.
`train_qwen3_lora.cpp` is the `nntr_lora_train` CLI driver. Full flag list:

```
nntr_lora_train <model_dir> <train_data.txt> [options]
  --lr <float>          learning rate (default 1e-4)
  --epochs <int>        number of epochs (default 1)
  --output <path>       LoRA adapter output path (default <model_dir>/lora_adapter.bin)
  --lora_path <path>    resume from an existing LoRA adapter
  --lora_rank <int>     LoRA rank; overrides nntr_config.json
  --lora_alpha <int>    LoRA alpha; overrides nntr_config.json
  --max_samples <int>   cap the number of training samples
  --seq_len <int>       training sequence length; overrides nntr_config.json's
                        init_seq_len. Attention memory is quadratic in this,
                        so prefer the smallest length that fits your samples.
  --clip_grad <float>   clip LoRA gradients to this global norm (0 = off, default)
  --train_norms         also train the RMSNorm gammas alongside the LoRA
                        adapters (default: norms frozen)
  --seed <int>          RNG seed for epoch shuffling (default 42)
```

The adapter is saved whenever validation loss improves, so an interrupted
run still leaves a usable adapter on disk.

### Inference-side adapter loading

`main.cpp` reads an optional `lora_file_name` from `nntr_config.json`; when
present, `load_weight_lora(weight_file, lora_file)` is used instead of
`load_weight(weight_file)`, applying the trained adapter on top of the base
checkpoint before inference.

## The NaN investigation

This is the part worth reading carefully if you hit something similar in the
future: the loss went to NaN during real training, and the eventual root
cause was *not* what the first two fixes addressed. Both of those fixes were
real bugs and had to be fixed regardless, but neither was the reason for the
NaN. The actual cause was found only by bisection.

### Symptom

Training on Qwen3-0.6B produced `NaN` loss, generally within the first few
steps. The first, most decisive diagnostic used was: train with `lr=0`. At
`lr=0` no weight update happens, so a correct implementation must produce
*exactly* the same train loss and validation loss on the same data (both are
just forward passes through an unchanging model). If they don't match, the
bug is in the forward path, not backward or the optimizer. If they match but
loss is still garbage (e.g. ~15.38, far from a sane cross-entropy value),
there's a correctness bug somewhere in the read path. If loss is sane at
`lr=0` but goes NaN at `lr>0`, the bug is specifically in backward.

### Bug 1: the `thread_local` thread-visibility bug

**Symptom found via the `lr=0` test**: loss was ~15.38 instead of a sane
value — not NaN yet, just wrong.

nntrainer's `GENERATOR`-based dataset callback runs on a **separate producer
thread** from the training thread that actually runs the forward/backward
passes. The original design had `g_lm_head_read_row`/
`g_tie_embedding_lm_head_read_row` declared `thread_local` and set by the
data generator before yielding each sample, on the assumption that the value
would be visible to the layer code that reads it. It is not — `thread_local`
storage is per-thread, so the value the generator set was invisible on the
training thread, which always saw the sentinel (`UINT_MAX`, meaning
"use `height - 1`"). For right-padded samples where the last real token
isn't the last row, this meant the LM head was scoring a pad-token position
instead of the intended one: wrong logits, wrong loss, but not NaN.

**Fix**: made the globals plain (non-`thread_local`) statics, and moved the
read-row *derivation* into `TieWordEmbedding::forwarding()`'s embedding-mode
branch — scanning the input token ids for the last non-pad id, on the same
thread that will later read the value in `calcDerivative`/`calcGradient`.
This removed the need for the training driver to communicate the value
across threads at all.

This fixed the *incorrect loss* problem. It did **not** fix the NaN.

### Bug 2 (root cause): the tensor-aliasing bug

**How it was found**: rather than guessing, each layer's `calcDerivative`
was temporarily gated behind an env-var switch (`NNTR_SKIP_CD_RMSNORM`,
`NNTR_SKIP_CD_SWIGLU`, etc. — a scratch/diagnostic mechanism, not part of the
final code) so that individual layers' backward passes could be disabled one
at a time and the run re-tried. Disabling `SwiGLULayer::calcDerivative`
alone made the NaN disappear. That isolated the bug to one layer, out of six
newly-implemented ones.

**Root cause**: nntrainer aliases each layer's *outgoing derivative* (the
gradient it produces for a given input) onto the **same memory buffer** as
that input's own storage — a memory-reuse optimization in the tensor pool.
The original SwiGLU backward was written intuitively, computing `d_up`
first and writing it out, then computing `d_gate` from `up`:

```cpp
// WRONG — writes d_up (which aliases the up input's own buffer) before
// d_gate has finished reading up.
for (i) {
  du[i] = swish(gate[i]) * dy[i];       // du aliases up: this OVERWRITES up[i]
  dg[i] = swish_prime(gate[i]) * up[i] * dy[i]; // reads the now-corrupted up[i]
}
```

Because `d_up`'s output tensor is literally the same memory as `up`'s input
tensor, the first line destroys `up[i]` before the second line can read it.
The result: `d_gate` is computed from garbage, and depending on what
happened to be in that memory, this manifests as wildly wrong values that
compound into NaN within a step or two of training. This was confirmed by
printing the two tensors' data pointers and observing they were identical.

**Fix**: snapshot every value the computation still needs to read *before*
writing any output:

```cpp
// RIGHT — read gate[i]/up[i]/dy[i] into locals first, then it doesn't
// matter that writing du[i]/dg[i] aliases up's storage.
for (i) {
  float gi = gate[i], ui = up[i], dyi = dy[i];
  du[i] = swish(gi) * dyi;
  dg[i] = swish_prime(gi) * ui * dyi;
}
```

The same class of bug was proactively audited for in every other new
`calcDerivative`. It was found again in `mha_core.cpp`: `calcDerivative` was
calling `dV.setZero()` before finishing reading `V` (`dV` aliases the `V`
input), which would have caused the identical failure mode the first time
`mha_core`'s backward was exercised with real data. Fixed by `.clone()`-ing
`V` into a local snapshot before zeroing `dV`. `rms_norm`/`reshaped_rms_norm`
and `lm_head`/`tie_word_embedding` were also audited and found *not* to have
this problem (they either accumulate-then-write without an intervening read,
or never read their input in `calcDerivative` at all).

A dedicated regression test, `SwiGLUCalcDerivativeIsSafeWhenGradAliasesInput`,
was added specifically because the other gradient-check tests build each
`Var_Grad` with independently-allocated variable/gradient storage and
**never exercise the aliasing** — they would pass even with the buggy code
above. The regression test explicitly constructs both an aliased and a
non-aliased version of the same computation and asserts they agree,
reproducing the exact hazard directly rather than relying on it showing up
by chance in a broader test.

### Verification that the fix actually worked

Rather than just asserting "should be fixed now," the same `lr=0` decisive
test was rerun after both fixes: train loss and validation loss matched
*exactly* — `0.420231` both — confirming the forward path is correct and
deterministic. Training was then run at `lr>0` and the loss decreased
monotonically with no NaN, across multiple datasets and a learning-rate
sweep (see [Verification performed](#verification-performed)).

## The LoRA initialization order bug

Separately from the NaN, mainline's `fc_layer.cpp` initialized LoRA weights
as `loraA = zeros`, `loraB = random`. This does give `B·A = 0` at
initialization (so the adapter starts as a no-op, correctly reproducing the
pretrained model), which is *necessary* but not *sufficient* for a good
initialization.

**Why the original order is badly conditioned**: with `A = 0`, the only
nonzero gradient at step 0 is `dL/dA = xᵀ(dy·Bᵀ) * scaling` — note this
depends on `B`, which is *random*. So the very first update to `A` is
steered entirely by whatever random values `B` happens to have, i.e. an
essentially arbitrary direction with no relationship to the actual gradient
signal. This produces an uncontrolled large first step. (An earlier version
of this analysis mistakenly claimed `B` would never receive gradient because
`A = 0` — that's wrong: `dL/dB = (dyᵀ · xA)ᵀ`-style terms are zero only on
the very first step; once `A` moves away from zero on step 2 onward, `B`
does receive real gradient. The actual defect is the conditioning of `A`'s
first update, not "B never trains.") Empirically, this ordering diverged
above `lr ≈ 1e-6` on Qwen3-0.6B.

**Standard LoRA practice (Hu et al. 2021)** is the reverse: `A` drawn from a
zero-mean distribution (`LECUN_NORMAL` here), `B = zeros`. This still gives
`B·A = 0` at step 0, but now the first gradient that matters,
`dL/dA = xᵀ(dy·Bᵀ)`, is zero (since `B = 0`), while `dL/dB = (something
involving the real, non-random A)` is well-defined and driven by actual
signal. `A`'s random values only ever get *scaled* by gradients that are
themselves shaped by real data, never used as the sole source of direction
for an update. This ordering was stable at `lr = 1e-4` on the same model.

Both the crash fix and this init-order swap live in `nntrainer/layers/fc_layer.cpp`,
each as their own commit.

## Was it gradient clipping?

A natural question once NaN loss shows up: is this actually just an
exploding (very large but finite) gradient, and would gradient clipping have
masked or fixed it? The answer here is **no**, and the reasoning matters:

Gradient clipping (`clip_grad_by_norm`, computed globally across all flagged
weights in `NetworkGraph`/`layer_node.cpp`) rescales a gradient whose norm is
large but *finite*. It cannot help with a gradient that is already `NaN`,
because `‖g‖` of a `NaN` vector is itself `NaN` — there's no finite norm to
clip against. The decisive piece of evidence: the NaN was reproduced even at
`lr = 0`. At `lr = 0`, no weight update happens at all, so an exploding
(but finite) gradient literally cannot be the cause — the forward pass
itself was already producing corrupted numbers (from the aliasing bug),
independent of what the optimizer does with the resulting gradient.

Gradient clipping was still added (`--clip_grad` / `lora_clip_grad_by_norm`,
threaded into the LoRA-targeted layers' `clip_grad_by_norm` property) as a
genuinely useful, orthogonal feature for training stability at higher
learning rates or noisier data — just not as the fix for this specific bug.

## Two memory bugs

Neither of these caused incorrect results; both were found while checking
resource usage on a memory-constrained target and would have been fatal
there (though not necessarily on a workstation with abundant RAM).

**`mha_core` training cache oversized** (~7 GB-scale at the sequence lengths
tested): the training-mode cache tensors (`train_q_roped`, `train_k_roped`,
`train_attn_wt`) were originally sized off `max_timestep` — the maximum
*decode* horizon used for incremental inference — rather than the actual
training sequence length (`query_dim.height()`). Training never does
incremental decoding, so this was allocating buffers sized for a much longer
sequence than any training batch actually needs. Fixed by sizing these
tensors from `query_dim.height()`. Additionally, the KV cache itself
(`cache_key`/`cache_value`) is now only allocated when `!for_training` (a new
`internal_cache_requested` member gates this in `setBatch()` and
`updateTensorsByInputDimensions()`), since training never uses incremental
decode caching at all.

**`load_weight_lora` cloning the entire base model** (~2 GB-scale): the
original implementation collected every weight of a throwaway reference
`base_model` graph into an `unordered_map<string, Tensor>` by **value**
(i.e. cloning each tensor), so that peak memory during a checkpoint load was
roughly `base_model + the clone map + this` — nearly triple the model size.
Fixed by storing `const Tensor *` (borrowed pointers into `base_model`)
instead; this is safe because `base_model` is guaranteed to outlive the copy
loop within the same function scope. Verified the save/load round-trip test
still passes identically after the change.

## A silent-wrong-answer bug in validation

`mha_core::forwarding()` gated its real attention computation on the
`training` boolean flag rather than on whether the training-mode tensors had
actually been allocated (`train_tensors_requested`). Validation is called
with `training=false` (it's not a training step), so this caused validation
to **skip attention entirely** and score whatever was left in a stale output
buffer — producing a loss (~13.5) that happened to look plausible (close to
a random-guess baseline for the vocab size) but was not actually measuring
the model. This is the more dangerous kind of bug: it doesn't crash and
doesn't produce an obviously-wrong number, it just silently measures the
wrong thing. Caught by the same `lr=0` train-vs-validation-loss-must-match
test: at `lr=0` the two forward passes should be identical, and they weren't,
which is what motivated looking at what differs between a "training-mode"
and "validation-mode" call in the first place. Fixed by gating on
`train_tensors_requested` instead of `training`.

## Verification performed

- **13 unit tests** (`unittest_lora_backward_gradcheck`): finite-difference
  gradient checks (central difference, `L(x) := dot(forward(x), dy)`, so the
  exact gradient of `L` w.r.t. `x` is what a correct backward pass should
  produce) for every `calcDerivative`/`calcGradient` added, plus the
  dedicated aliasing regression test and the mha_core
  training-forward-vs-inference-prefill consistency check.
- **3 end-to-end tests** (`unittest_qwen3_lora_training_smoke`): a tiny
  synthetic Qwen3 graph compiles for training with LoRA enabled, the
  resulting trainable/frozen flags match the freeze policy exactly, and a
  few training steps followed by `save_weight_lora`/`load_weight_lora`
  round-trip correctly.
- **No regressions**: the full pre-existing suite (67 CausalLM model tests,
  17 KV-cache tests, 10 embedding-sidecar tests, and everything else in
  nntrainer's unit test suite — 38 test suites in total) continues to pass.
- **Real-model validation on Qwen3-0.6B**:
  - The `lr=0` train-loss-equals-validation-loss test (`0.420231` both,
    exact match) as the decisive correctness check described above.
  - Loss decreases monotonically at `lr > 0` on identity, SST-2, and LaMP-3
    style data, with no NaN.
  - Saved adapters scanned byte-for-byte for NaN/Inf across every parameter
    — clean.
  - A full train → save → load → generate round trip produces coherent,
    adapter-influenced output.
  - A learning-rate sweep confirms the expected qualitative behavior, not
    just "doesn't crash": a very small LR leaves generation ≈ unchanged from
    the base model; `1e-4` visibly picks up trained vocabulary/style while
    staying fluent; `1e-3` overfits a small (~20-sample) dataset into
    repetitive output — the textbook LoRA overfitting failure mode at too
    high an LR, not a bug.
  - Peak RSS was measured during a training run (`~8 GB`) to sanity-check
    the memory-bug fixes above actually brought usage into a reasonable
    range.

## How to train

```
nntr_lora_train <model_dir> <train_data.txt> \
  --lr 1e-4 --epochs 3 --lora_rank 8 --lora_alpha 16 \
  --output <model_dir>/lora_adapter.bin
```

`<model_dir>` must contain the same `nntr_config.json`/checkpoint layout used
for inference; `lora_rank`/`lora_alpha`/`lora_target` can be set there or
overridden on the command line. To also train the RMSNorm gammas alongside
the adapters, pass `--train_norms` (or set `lora_train_norms: true` in the
config). To use the resulting adapter at inference time, set
`lora_file_name` to the saved adapter's filename in `nntr_config.json` and
run `nntr_causallm` as usual — `main.cpp` will call `load_weight_lora`
instead of `load_weight` automatically.

## Commit structure

The work above is organized into ten reviewable, individually-buildable
commits. Each commit leaves the tree in a working state (compiles, and any
test it introduces passes at that point in history):

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
10. **`[CausalLM] document LoRA training support and its commit structure`**
    this file

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
- The tensor-aliasing hazard documented above is specific to nntrainer's
  memory-reuse design and is not something the compiler or existing test
  harness catches automatically; any *future* layer that implements
  `calcDerivative` and reads an input after writing a different input's
  outgoing derivative should be checked for the same class of bug.

# Implementation Plan: Port Remaining Prefill CPU Ops to NPU + Enqueue Support

**Date:** 2026-08-17
**Audience:** an implementing agent/engineer, not the architecture-discussion audience of
the companion docs (`NPU_WHOLE_GRAPH_DELEGATION_STUDY_2026-08-17.md`).
**Scope:** Qwen3-0.6B **prefill only**. Decode is explicitly out of scope (deliberately
CPU by design — profiling showed a net loss below ~150-200 tokens, see
`mha_core.cpp:118-121`). Training is not touched in this repo directly, but everything
here is scoped so it is directly reusable when this work moves to `nntrainer-lora`,
since QLoRA training's forward pass is architecturally "continuous prefill."
**Non-goals for this pass:** the "sync"/lazy-flush-on-CPU-read design
(`LayerNode`-level auto-sync guard) discussed in the companion doc is explicitly
deferred. This plan only covers (1) moving currently-CPU ops onto the DSP, and
(2) adding an enqueue/flush split so multiple DSP ops can share one flush — not the
smart automatic sync-point insertion.

---

## 0. Current state (do not re-derive, verified by direct code reading)

**Already on NPU today (working, no changes needed):**
- Q/K/V/O projections, gate/up, down, LM-head: `fully_connected`/`gate_up_layer` with
  `engine=cdsp` → `HexagonComputeOps::gemm_q4_0_accel_fp32`/`gemm_q4_0_batch_fp32`
  (`nntrainer/hexagon/hexagon_compute_ops.cpp:228-282`).
- Attention (prefill only): `nntr_htp_bridge_flash_attn`, wired into
  `MHACoreLayer::one_batch_incremental_forwarding`
  (`Applications/CausalLM/layers/mha_core.cpp:829-948`), gated by
  `NNTR_HEXAGON_FLASH_ATTN=1` + `step_size>=160` + `head_dim%64==0`
  (`mha_core.cpp:98-156`). **Already correct and working — do not modify the gating
  logic or the bridge call itself in this plan**, only what feeds it (RoPE, see Part B).
- FFN (gate+up+SwiGLU+down), as an alternative to the separate-layer path: `fused_ffn`
  layer using `nntr_htp_bridge_ffn_swiglu`
  (`Applications/CausalLM/layers/fused_ffn_layer.cpp:244-278`), selected in
  `Transformer::createMlp` when `NNTR_HEXAGON_FUSED_FFN=1`
  (`Applications/CausalLM/models/transformer.cpp:545-562`). Already correct and working.

**Still CPU today (the actual gap this plan closes):**
- **RMSNorm** — runs **4 times per decoder block**, not 2: `attn_norm` and `ffn_norm`
  (via `causallm::RMSNormLayer`, full hidden width 1024), plus `q_norm` and `k_norm`
  (via `causallm::ReshapedRMSNormLayer`, per-head width 128, applied to reshaped
  Q/K) — plus one final `output_norm` after all 28 layers. **Zero DSP path exists.**
  `nntrainer::ComputeOps` (the interface `HexagonComputeOps` implements) has **no
  virtual method for normalization at all** — RMSNorm calls
  `nntrainer::rms_norm_wrt_width_fp32_intrinsic` as a free function
  (`Applications/CausalLM/layers/rms_norm.cpp:82-104`), completely outside the
  ComputeOps dispatch table. Porting this cannot be "override a ComputeOps method" —
  it must follow the same **direct dlopen/dlsym bypass** pattern already used by
  `mha_core`/`fused_ffn` (see Part A).
- **RoPE** — applied on CPU to Q (`mha_core.cpp:881-885`) and to K at KV-cache-append
  time (`mha_core.cpp:861-862`), both via `apply_rotary_emb_tensor_v2`
  (`mha_core.cpp:1240-1339`), strictly **before** the `flash_attn` bridge call. The
  bridge's C signature (`mha_core.cpp:48-51`) has **no RoPE-related parameter at all**
  — it expects pre-rotated Q/K. Porting RoPE means adding it as its own new bridge
  op(s), issued before `flash_attn`.
- **Embedding lookup** — `causallm::TieWordEmbedding` (Qwen3-0.6B has
  `tie_word_embeddings: true`), FP32 table (~622MB, `vocab=151936 × hidden=1024`),
  zero-copy per-token slice copy (`tie_word_embedding.cpp:214-298`). See Part D for why
  this is explicitly deprioritized.
- **Residual add** — plain `nntrainer::AdditionLayer`, elementwise. Not ported in this
  pass (see Part D); it's the cheapest op in the block and moving it only matters once
  block-level batching (Part D) is attempted.

---

## Part A: RMSNorm DSP bridge

### A.1 Target DSP op

Use **`HTP_OP_RMS_NORM_MUL`** (in `ggml-hexagon/ggml/src/ggml-hexagon/htp/unary-ops.c`,
kernel `hvx_fast_rms_norm_mul_f32`, lines 166-229), **not** the plain `HTP_OP_RMS_NORM`
+ a separate `HTP_OP_MUL`. The fused variant takes the gamma weight as `src[1]` and
does normalize+scale in one DSP kernel/DMA pass — this is exactly what
`ggml_backend_hexagon_graph_optimize` already does automatically for ordinary ggml
graphs (`ggml-hexagon.cpp:3375-3381`, fusing `RMS_NORM`→`MUL` into one node), so we're
matching an established, proven pattern, not inventing one.

**Confirmed exact math match** (no semantic gap — verified against both sides):
```
mean  = sum(x_i^2) / W                    // htp/unary-ops.c, and
scale = rsqrt(mean + epsilon)             // nntrainer's rms_norm.cpp both compute
out_i = (x_i * scale) * gamma_i           // this identically
```
nntrainer reference: `rms_norm.cpp:82-104` (kernel dispatch) →
`nntrainer/tensor/cpu_backend/fallback/fallback_internal.cpp:632-652`
(`__fallback_rms_norm_wrt_width_fp32_intrinsic`, the readable reference implementation:
`sum_sq/W`, `1/sqrt(mean+epsilon)`, multiply). Gamma multiply is currently a *separate*
CPU step in nntrainer (`rms_norm.cpp:110-117`, `out_step.multiply_i(gamma)`) — folding
it into the single DSP op is a net simplification, not just a port.

**Constraint:** DSP kernel is F32-only for both input and gamma
(`ggml_hexagon_supported_unary`, `ggml-hexagon.cpp:2885-2905`; confirmed independently
in the op's own dtype checks). This is **not a blocker for Qwen3-0.6B** — both shipped
configs (`nntr_config.json`, `nntr_config_quantized.json`) use FP32 activations
throughout (only FC *weights* are Q4_0 in the quantized config; RMSNorm/embedding
activations are always FP32). Flag explicitly in code comments that this bridge call
must not be used if a future config ships FP16 activations, since there is no F16 path
on the DSP side for this op.

### A.2 New bridge function (in `ggml-hexagon/ggml/src/ggml-hexagon/nntr-htp-bridge.cpp`)

Follow the exact structural template already used by every existing bridge function
(study `nntr_htp_bridge_gemm_q4_0`, lines 797-913, as the simplest reference):

1. Lock `state.mtx`.
2. Validate dims, throw on violation (matches `nntr_htp_bridge_check_dims` pattern,
   lines 747-761).
3. `nntr_htp_bridge_ensure_session(state)` (idempotent, lines 719-740).
4. For each of the 3 tensors (input activation, gamma, output): check
   `nntr_htp_bridge_find_ext_pool` first (zero-copy if the pointer falls inside a
   registered rpcmem pool — see A.4), else stage via
   `nntr_htp_bridge_get_staging`+`memcpy` (same branching as
   `gemm_q4_0`, lines 838-872).
5. Build 3 `ggml_tensor` structs via the existing `nntr_htp_bridge_set_tensor` helper
   (lines 410-434) — `t_in` (shape `[W, M]`), `t_gamma` (shape `[W, 1]`, broadcast — the
   DSP kernel already supports `src1` having 1 row, `unary-ops.c:948`), `t_out` (shape
   `[W, M]`).
6. Set `t_out.op_params[0]` = epsilon via `memcpy(&t_out.op_params[0], &epsilon,
   sizeof(float))` — exactly the pattern flash_attn uses for its own scale param
   (`nntr-htp-bridge.cpp:1275-1280`).
7. `t_out.op = GGML_OP_RMS_NORM` won't matter here since we bypass ggml's op-type field
   entirely and set the HTP opcode directly on the `htp_opnode`:
   `htp_opnode node{}; node.node = &t_out; node.opcode = HTP_OP_RMS_NORM_MUL;`
   `t_out.src[0] = &t_in; t_out.src[1] = &t_gamma;`
8. `sess->enqueue_op(node); sess->flush(true);` (see Part C for the enqueue-only
   variant).
9. Copy result out of staging if not zero-copy.
10. Unlock, return 0/-1.

**Proposed signature:**
```c
int nntr_htp_bridge_rms_norm(const float *in, const float *gamma, float *out,
                             unsigned int M, unsigned int W, float epsilon);
```
`M` = number of rows to normalize (independent normalization units), `W` = row width
(the dimension being normalized over). This single signature covers **all four
RMSNorm call sites** by construction — the reshape work nntrainer already does on the
caller side is preserved unchanged:

| Call site | M (rows) | W (width) | Caller reshape already done by |
|---|---|---|---|
| `attn_norm` | `seq_len` | 1024 | none needed, natural layout |
| `ffn_norm` | `seq_len` | 1024 | none needed |
| `output_norm` | `seq_len` (last block only) | 1024 | none needed |
| `q_norm` | `seq_len * 16` | 128 | `ReshapedRMSNormLayer`'s existing reshape (`reshaped_rms_norm.cpp:78-82`) |
| `k_norm` | `seq_len * 8` | 128 | same, `feature_size=128` |

Since the reshape is a contiguous, stride-preserving view in both existing CPU call
sites, passing the reshaped buffer's raw pointer + the resulting `M`/`W` to the new
bridge function requires no new reshape logic — reuse exactly what
`reshaped_rms_norm.cpp` already produces before it currently calls the CPU kernel.

### A.3 nntrainer-side wiring

Add to **both** `Applications/CausalLM/layers/rms_norm.cpp` and
`Applications/CausalLM/layers/reshaped_rms_norm.cpp`:

- A `get_rms_norm_bridge()` helper, structurally identical to `mha_core.cpp`'s
  `get_flash_attn_bridge()` (`mha_core.cpp:57-88`): `dlopen("libggml-hexagon.so",
  RTLD_NOW|RTLD_GLOBAL)`, `dlsym(handle, "nntr_htp_bridge_rms_norm")`, cache the
  function pointer for process lifetime.
- A `should_use_rms_norm_dsp(unsigned int M, bool is_prefill)` gate, matching the style
  of `should_use_flash_attn`/`should_use_fused_ffn`: new env var
  `NNTR_HEXAGON_RMS_NORM=1` (off by default), `is_prefill` check (this plan is
  prefill-only — do not enable for decode/`step_size<=1`), F32-activation-dtype check
  (throw/fallback to CPU if not FP32), bridge-availability check.
- On `rc != 0` from the bridge call, fall back to the existing CPU path unchanged —
  same safety discipline as every other bridge integration in this codebase.
- This is a **new call path**, not a `HexagonComputeOps` override — do not attempt to
  add an `rms_norm` method to the `ComputeOps` interface; that interface has 60+
  existing pass-through methods and no normalization slot, and every other custom-op
  integration (flash_attn, fused_ffn) already establishes direct-dlsym as the accepted
  pattern for ops the `ComputeOps` abstraction doesn't cover.

### A.4 Zero-copy consideration

If nntrainer's RMSNorm activation buffers are allocated through the same
`HexagonRpcAllocator`/rpcmem-backed pool already registered via
`nntr_htp_bridge_register_activation_pool` (`nntr-htp-bridge.cpp:673-698`), the new
call gets zero-copy input/output automatically via `nntr_htp_bridge_find_ext_pool` —
no extra work needed, this falls out of the existing pool-registration mechanism. If
not (e.g., transient RMSNorm output tensors allocated from a plain CPU `MemoryPool`),
the call transparently falls back to staging+memcpy, exactly like every other bridge
function's non-pool path. **Verify which is actually true for RMSNorm's tensors** as
part of implementation — this affects performance, not correctness.

---

## Part B: RoPE DSP bridge

### B.1 Confirmed op parameters for Qwen3-0.6B (resolved, not TBD)

- **Target op**: `HTP_OP_ROPE` (`htp/rope-ops.c`).
- **Mode: `HTP_ROPE_TYPE_NEOX` (value 2)** — confirmed by direct inspection of the
  actual rotation math in `nntrainer/tensor/cpu_backend/x86/avx2_impl.cpp:1757-1809`
  (`compute_rotary_emb_value`): it pairs element `k` with element `k+half_` (`i0=w+k,
  i1=w+k+half_`) and computes
  `out[i0]=a*cos-b*sin, out[i1]=a*sin+b*cos` — this is the **split-half** convention,
  matching `hvx_rope_neox_f32_aa` (`rope-ops.c:272-318`), **not** the interleaved
  `hvx_rope_f32_aa` (pairs `x[2i],x[2i+1]`). Do not re-derive this from scratch — it is
  settled; using the wrong mode produces plausible-looking but numerically wrong
  attention with no crash to signal the bug.
- **No YaRN**: `Applications/CausalLM/res/qwen3/qwen3-0.6b/config.json:21` has
  `"rope_scaling": null`. Set `ext_factor=0`, `attn_factor=1`; `beta_fast`/`beta_slow`
  are irrelevant when `ext_factor=0` (they only affect the YaRN ramp).
- `freq_base` (theta) = `1000000` (`config.json`'s `rope_theta`, read at
  `transformer.cpp:185`).
- `freq_scale` = `1.0` (no linear scaling reported).
- `n_dims` = `head_dim` = **128** (Qwen3-0.6B's shipped `config.json:9` explicitly sets
  `head_dim=128`, overriding the `hidden_size/num_heads=1024/16=64` default computed at
  `transformer.cpp:170-172` — a stale code comment at `mha_core.cpp:95-96` claims
  head_dim=64 for this model; **the actual shipped config uses 128**, verify this
  doesn't silently regress if `head_dim` is ever read from the wrong source). Full
  rotation, no partial-rotary remainder.
- `n_ctx_orig` = `max_position_embeddings` = 40960 (`config.json:14`) — only matters
  for YaRN, so inert here, but pass correctly anyway.

### B.2 New bridge function

```c
int nntr_htp_bridge_rope(float *inout, const int32_t *positions,
                         unsigned int n_tokens, unsigned int n_heads,
                         unsigned int head_dim, float freq_base, int mode);
```

Structural template: same skeleton as A.2, but modeled on the **multi-input** pattern
of `nntr_htp_bridge_flash_attn` (lines 1079-1316) rather than the single-weight
`gemm_q4_0` pattern, since ROPE needs two tensor inputs (`src[0]`=Q-or-K activation,
`src[1]`=positions):

1. Lock/validate/ensure-session as usual.
2. Build `t_in` with `ne = [head_dim, n_heads, n_tokens]` (matches the layout
   `op_rope()` expects — it loops internally over batch/position/head,
   `rope-ops.c:471-569` — a single call handles the **entire** Q or K tensor for this
   forward pass, not per-head; do not issue one call per head).
3. Build a small **I32** `positions` tensor of shape `[n_tokens]` — nntrainer currently
   tracks position as a scalar `cache_from` offset, not a tensor, so this must be
   materialized: a monotonic range `[cache_from, cache_from+1, ..., cache_from+n_tokens-1]`.
   This is cheap (a few hundred bytes, computed once per call) — do not treat this as a
   performance concern, but do NOT skip it: `HTP_OP_ROPE` has no scalar-offset
   parameter, positions must come from `src[1]` (confirmed: `rope-ops.c:460,524`).
4. Set `t_out.op_params` per §B.1: `[1]=n_dims(head_dim), [2]=mode(2 for NEOX), [4]=n_ctx_orig,
   [5]=freq_base, [6]=freq_scale(1.0), [7]=ext_factor(0.0), [8]=attn_factor(1.0),
   [9]=beta_fast(default, e.g. 32.0 — ggml's own default, irrelevant since ext_factor=0),
   [10]=beta_slow(default, e.g. 1.0)` — pack via `memcpy` exactly as flash_attn does for
   its scale param.
5. `t_out.src[0] = &t_in; t_out.src[1] = &t_positions;` (`src[2]` freq_factors left
   null — not used, no NTK-by-parts scaling for this model).
6. `htp_opnode node{}; node.node=&t_out; node.opcode=HTP_OP_ROPE;`
7. `sess->enqueue_op(node);` — **do not flush yet** if this call is part of the
   RoPE(Q)+RoPE(K)+flash_attn sequence (see B.3/Part C). Provide both an
   `nntr_htp_bridge_rope` (enqueue+flush, for standalone/testing use) and
   `nntr_htp_bridge_rope_enqueue` (enqueue only) per the Part C split.

### B.3 nntrainer-side wiring — KV-cache interaction

This is the trickiest correctness point in this plan. Current CPU behavior
(`mha_core.cpp:861-862`):
```cpp
apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim, cache_index, !use_rope);
```
— K is rotated **as it is written into the KV cache**, i.e. the cache stores
already-RoPE'd K (this matters because the cache persists across future
decode/continuation steps, unlike a scratch buffer).

For the DSP port, two options — **recommend option (a)**:

- **(a) Copy raw K into the cache slot first (unchanged, cheap CPU copy — this already
  happens as part of cache management), then RoPE the cache slice in place via the new
  bridge**, operating directly on the cache's memory (if the cache is rpcmem/pool-backed,
  this is zero-copy; if not, stage+copy-back exactly once). This preserves the existing
  invariant that the cache always holds post-RoPE K, with the DSP doing the rotation
  instead of the CPU.
- (b) RoPE a fresh K on a scratch buffer via the bridge, then copy that into the cache.
  Simpler to reason about in isolation, but adds an extra copy that (a) avoids. Only
  fall back to (b) if the cache's memory isn't safely bridgeable in place (e.g., if it's
  interleaved with other cache metadata in a way that makes an in-place DSP write risky).

Q is simpler — RoPE is applied to `query_step` in place (`mha_core.cpp:881-885`), a
scratch/activation tensor, not the cache; port this call directly to the new bridge
with no cache-interaction subtlety.

**Only wire this into the flash_attn-eligible path** (same gate as
`should_use_flash_attn`: prefill, `step_size>=160`, `head_dim%64==0`). Do not touch the
CPU RoPE call used by the decode/non-flash-attn path — leave that entirely unchanged.

### B.4 Sharing one flush with flash_attn

Since `nntr_htp_bridge_flash_attn` already requires pre-RoPE'd Q/K as input, and the
existing bridge already proves that **multiple dependent enqueued ops execute correctly
within one flush in FIFO order** (see `nntr_htp_bridge_ffn_swiglu`, which enqueues 5
sequentially-dependent ops before a single flush, `nntr-htp-bridge.cpp:1531-1540`), the
target sequence for the attention sub-block is:

```
enqueue(RoPE, Q)
enqueue(RoPE, K-into-cache-slot)
enqueue(FLASH_ATTN_EXT, using the now-rotated Q and the cache's now-rotated K/V)
flush()   // single flush for all 3 ops
```

This requires the enqueue/flush split from Part C — without it, this lands as 2
flushes (RoPE pair, then flash_attn) instead of 1, which is still a strict improvement
over today (RoPE ops didn't exist as DSP calls at all) and can ship as an intermediate
step if Part C isn't ready yet.

---

## Part C: Enqueue/flush split (the "adding enqueue support" deliverable)

### C.1 Recommended design: a generic enqueue primitive, not per-op boilerplate duplication

The existing 9 bridge functions each hand-duplicate the same skeleton (lock → validate
→ ensure-session → per-operand zero-copy-or-stage decision → build `ggml_tensor`s →
build `htp_opnode` → `enqueue_op`+`flush` → copy-out). Adding RMSNorm and RoPE as two
more hand-duplicated copies is acceptable but not ideal, **and this plan is explicitly
about adding infrastructure for enqueue support** — so build it once, correctly, now:

```c
// Enqueue exactly one HTP op without flushing. Caller must already hold the
// batch scope (see begin/end below). Returns 0 on success.
int nntr_htp_bridge_enqueue_op(int opcode,
                               const nntr_htp_tensor_desc *srcs, int n_srcs,
                               const nntr_htp_tensor_desc *dst,
                               const int32_t op_params[16]);

// Where nntr_htp_tensor_desc bundles {ptr, ne[4], is_output_of_prior_enqueue}
// and reuses the existing find_ext_pool/staging logic per-tensor exactly as
// today, just factored out of the 9 per-op functions into one shared helper.

// Explicit batch scope — required because state.mtx currently gets
// acquired-and-released WITHIN each existing bridge call. Splitting
// enqueue/flush means the lock must span the whole batch, not one call.
void nntr_htp_bridge_begin_batch();   // acquires state.mtx, marks "deferred" mode
void nntr_htp_bridge_flush();        // the shared sync point: sess->flush(true)
void nntr_htp_bridge_end_batch();     // flush() if not already flushed, releases mtx
```

Each of RMSNorm and RoPE (and, if desired, the existing gemm_q4_0/flash_attn/ffn_swiglu
implementations, as a follow-on refactor — **not required for this pass**) becomes a
thin wrapper: build the tensor descriptors, call `nntr_htp_bridge_enqueue_op`, and
either flush immediately (for standalone/back-compat blocking callers) or leave it
pending (when called inside a `begin_batch`/`end_batch` scope).

### C.2 Why the batch scope must be explicit, not implicit

`state.mtx` (the single global mutex serializing all bridge calls today) is currently
acquired and released **within** each existing function
(`nntr_htp_bridge_gemm_q4_0`, etc.). If `enqueue` and `flush` become independently
callable, naively removing the lock from inside each function and leaving callers to
lock nothing would allow two threads' enqueue sequences to interleave into the same
batch — silent corruption, not a crash. The `begin_batch()`/`end_batch()` pair makes
lock ownership explicit and scoped to exactly the set of ops that should share one
flush; `end_batch()` must defensively call `flush()` if the caller forgot to, so a
missing explicit flush degrades to "one extra round trip" rather than "ops silently
never execute."

### C.3 nntrainer-side call sites to update

Two independent call paths hit the same underlying process-wide
`ggml_hexagon_session` singleton today (confirmed:
`nntr_htp_bridge_ensure_session` lazily creates exactly one session shared by both) —
both need updating to participate in shared batches:

1. **Direct-dlsym path** (`mha_core.cpp`, `fused_ffn_layer.cpp`) — straightforward,
   these already call bridge functions directly; wrap the RoPE(Q)+RoPE(K)+flash_attn
   sequence in B.4 with `begin_batch()`/`end_batch()`.
2. **`HexagonComputeOps` path** (FC-family layers' GEMMs) — for a *future* pass that
   extends batching to include GEMMs (Part D), `HexagonComputeOps::gemm_q4_0_accel_fp32`
   etc. would need to check "is a batch already open?" and use the enqueue-only variant
   if so. **Not required for this pass** — Parts A/B only need the direct-dlsym path
   updated, since RMSNorm and RoPE are both new direct-dlsym-style call sites, not
   ComputeOps overrides (per Part A.3). Flag this as necessary scope for Part D, not now.

---

## Part D: Explicitly out of scope for this pass (recorded for later, do not implement now)

- **Embedding lookup (`HTP_OP_GET_ROWS`)**: technically portable — the op exists, and
  Qwen3-0.6B's FP32 tied-embedding table matches the op's F32-only constraint
  (`get-rows-ops.c:151-160`; `ggml-hexagon.cpp:3043-3061`). **Do not implement in this
  pass**: it runs once per forward pass, gathering only `seq_len` rows out of a large
  but simple table — a pure memory-bound gather nntrainer already does as a zero-copy
  slice view (`tie_word_embedding.cpp:244-245`). The DSP round-trip overhead is very
  likely to exceed the CPU cost this would save. Additionally, token IDs are stored as
  **FP32** in nntrainer (`tie_word_embedding.cpp:231-232`) but `GET_ROWS` requires I32/I64
  indices (`get-rows-ops.c:162-164`), so this would need an extra cast step. Revisit
  only if profiling *after* Parts A-C shows embedding lookup as a non-trivial fraction
  of prefill wall-clock (expected to be negligible).
- **Residual add (`HTP_OP_ADD`)**: trivial op, exists on the DSP, but only worth
  porting once full block-level batching (below) is attempted — moving it alone doesn't
  reduce flush count today since it already sits directly between two already-CPU or
  already-flushed boundaries.
- **Full block-level batching**: with Parts A-C landed, plus `fused_ffn` enabled, plus a
  future residual-add bridge, an entire Qwen3 decoder block's prefill forward could in
  principle collapse to **one flush**: `enqueue(attn_norm) → enqueue(QKV batch) →
  enqueue(RoPE×2) → enqueue(flash_attn) → enqueue(O-proj) → enqueue(residual-add) →
  enqueue(ffn_norm) → enqueue(fused_ffn) → enqueue(residual-add) → flush()`. This is
  the literal "whole transformer block on NPU, zero mid-block flushes" outcome. It
  requires widening Part C's batch scope to also cover the `HexagonComputeOps`/GEMM
  path (C.3 item 2). **Recorded as the natural next phase, not part of this plan's
  deliverable.**

---

## Part E: Correctness verification checklist (must close before trusting numerical output)

1. **RoPE convention** — resolved above (NEOX, split-half). No further investigation
   needed, but the implementer should still write a standalone numerical parity test
   (random input, same shape as Qwen3-0.6B's Q/K, compare CPU vs bridge output
   bit-for-bit-ish) before wiring into `mha_core.cpp`.
2. **RoPE scaling** — resolved above (`rope_scaling: null`, no YaRN). If this plan is
   ever reused for a different Qwen3 variant with non-null `rope_scaling`, the YaRN
   params (`ext_factor`, `beta_fast`, `beta_slow`) must be derived from that config,
   not hardcoded to the Qwen3-0.6B defaults above.
3. **ARM NEON RoPE reference gap (new finding, unresolved)**: the FP32
   `compute_rotary_emb_value` function that `arm_compute_backend.cpp:586` calls (the
   actual on-device ARM path, as opposed to the x86 AVX2 path used to derive the
   convention above) **has no definition found in `neon_impl.cpp`** — only
   `compute_rotary_emb_value_uint16` is defined there. Before treating "CPU behavior"
   as ground truth on the actual ARM target, the implementer must verify: does this
   symbol actually link on the Android aarch64 build (maybe defined in a file not
   checked, or behind a preprocessor guard), or is this dead/broken code that never
   actually executes on-device for the FP32 activation path? If it's genuinely
   unlinked, the AVX2 math (already confirmed to match the NEOX kernel) should be
   treated as the authoritative reference instead, and the ARM gap reported/fixed
   separately as a pre-existing bug independent of this plan.
4. **RMSNorm formula** — resolved above, exact match confirmed (mean+eps under sqrt,
   gamma multiply after normalize). Still write a parity test before wiring in.
5. **q_norm/k_norm row/width bookkeeping** — verify the per-head reshape (`M = seq_len *
   num_heads`, `W = head_dim = 128`) is passed to the new bridge call with the exact
   same row-major layout `reshaped_rms_norm.cpp` already produces for the CPU kernel —
   a transposed or mis-strided reshape would silently normalize over the wrong axis.
6. **F32-only enforcement** — add an explicit assertion/throw (not just a silent
   fallback) if any of RMSNorm/RoPE/GET_ROWS-in-the-future are ever invoked with FP16
   activations, since none of these DSP kernels support F16 input today. This should
   never trigger for the current shipped Qwen3-0.6B configs, but should fail loudly
   rather than silently corrupt if a future config changes that.
7. **head_dim source of truth** — confirmed 128 for Qwen3-0.6B from `config.json:9`,
   overriding the `hidden_size/num_heads` default. Ensure the RoPE bridge wiring reads
   `head_dim` from the same place `mha_core.cpp` already does (its member variable),
   not re-derived from `hidden_size/num_heads`, to avoid silently reintroducing the
   stale-comment confusion already found in `mha_core.cpp:95-96`.

---

## Part F: Rollout discipline

Match the existing project convention exactly (every existing bridge feature follows
this pattern — don't deviate):
- Every new bridge call gated behind its own off-by-default env var
  (`NNTR_HEXAGON_RMS_NORM`, `NNTR_HEXAGON_ROPE`), checked once and cached (`static`
  pattern in `should_use_flash_attn`, `mha_core.cpp:101-102`).
- Automatic CPU fallback on any non-zero bridge return code, with a `ml_logw` warning —
  never hard-fail a forward pass because a DSP call failed.
- Recommended implementation order (lowest risk first):
  1. **A (RMSNorm)** — simplest formula, single well-matched fused DSP op, no
     KV-cache interaction subtlety, no convention ambiguity.
  2. **B (RoPE)**, wired standalone first (own flush) before attempting B.4's shared
     flush with flash_attn — validate RoPE alone against the parity test in Part E
     before composing it with anything else.
  3. **C (enqueue/flush split)** — once A and B both work as standalone (each with
     their own flush), build the generic enqueue primitive and batch-scope API, then
     switch B.4's sequence to share one flush.
  4. **Part D** — explicitly deferred, do not start until A-C are verified on-device.
- On-device verification is out of scope for whoever writes the code in this pass to
  perform themselves per the requester's instruction — but every change must be built
  such that it is verifiable (parity tests, explicit fallback paths, loud failure on
  unsupported dtypes) by whoever does have device access.

---

## File-level task list

| File | Repo | Change |
|---|---|---|
| `nntr-htp-bridge.cpp` | ggml-hexagon | Add `nntr_htp_bridge_rms_norm[_enqueue]`, `nntr_htp_bridge_rope[_enqueue]`, `nntr_htp_bridge_enqueue_op`, `nntr_htp_bridge_begin_batch`/`flush`/`end_batch` |
| `nntr-htp-bridge.h` (or equivalent public header) | ggml-hexagon | Declare the above |
| `Applications/CausalLM/layers/rms_norm.cpp` | nntrainer | Add `get_rms_norm_bridge()`, `should_use_rms_norm_dsp()`, wire into `incremental_forwarding()` prefill path, CPU fallback |
| `Applications/CausalLM/layers/reshaped_rms_norm.cpp` | nntrainer | Same, for q_norm/k_norm |
| `Applications/CausalLM/layers/mha_core.cpp` | nntrainer | Add `get_rope_bridge()`, wire RoPE(Q)/RoPE(K-into-cache) into the flash_attn-eligible prefill path, wrap RoPE×2+flash_attn in `begin_batch`/`end_batch` once Part C lands |
| new: parity test harness (location per implementer's judgement, e.g. `test/`) | nntrainer | CPU-vs-bridge numerical comparison for RMSNorm and RoPE before wiring into the model |

Signed-off-by: Anirudh <anirudh1023@gmail.com>

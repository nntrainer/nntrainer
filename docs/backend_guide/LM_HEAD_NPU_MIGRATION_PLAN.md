# LM_Head NPU Migration Plan

**Date:** 2026-08-21
**Author:** Cline SR
**Status:** Draft — awaiting approval

**Current model config:** `lmhead_dtype: "Q6_K"`, `embedding_dtype: "Q6_K"`, `fc_layer_dtype: "Q4_0"`
**→ The lm_head weights are Q6_K, NOT Q4_0.**

---

## 1. Current State

### 1.1 What runs on NPU today

The following ops are already dispatched to the Hexagon cDSP via `libggml-hexagon.so`:

| Op | Bridge function | Used by |
|---|---|---|
| Q4_0 GEMM (FP32 act) | `nntr_htp_bridge_gemm_q4_0` | FC layers (prefill) |
| Q4_0 GEMM (FP16 act) | `nntr_htp_bridge_gemm_q4_0_fp16` | FC layers (prefill, FP16 build) |
| Q4_0 batched GEMM | `nntr_htp_bridge_gemm_q4_0_batch` | QKV/gate-up fusion |
| FP32 SGEMM | `nntr_htp_bridge_sgemm_fp32` | Training forward/backward |
| Flash attention | `nntr_htp_bridge_flash_attn` | MHA (prefill) |
| Fused FFN+SwiGLU | `nntr_htp_bridge_ffn_swiglu` | FFN (prefill) |
| Weight upload | `nntr_htp_bridge_upload_weight_q4x4x2` | All Q4_0 weights |

### 1.2 What stays on CPU today

| Layer | Op | Why it's on CPU |
|---|---|---|
| **lm_head** (decode) | GEMV: `logits[v] = dot(input, dequant(weight_row_v))` | No bridge function for Q4_0 GEMV with tied-weight layout |
| **lm_head** (prefill, skip_prefill=true) | Skipped entirely | `skip_prefill` flag set in config |
| **embedding** | Row dequantize + memcpy | Pure gather, no matmul — CPU is bandwidth-bound, DSP adds round-trip latency |
| **RMSNorm** | Element-wise | Already on CPU; not a matmul bottleneck |
| **RoPE** | Element-wise | Already on CPU; not a matmul bottleneck |
| **Addition (residual)** | Element-wise | Already on CPU; not a matmul bottleneck |

### 1.3 lm_head decode path (the target)

The lm_head layer runs **only during decode** (prefill is skipped via `skip_prefill=true`).
For each generated token, it computes `logits[1..vocab_size]` from the last hidden state.

**Current implementation** (`tie_word_embedding.cpp::incremental_forwarding_lmhead`):

- **Q6_K weights**: calls `input_fp32.dot(weight, hidden_step, false, true)` — this goes
  through `FloatTensor::dotQnK`, which is a CPU-only GEMV. No NPU dispatch.
- **Q4_0 weights**: manually loops over vocab rows, dequantizes each row, and calls
  `sdot()`. Pure CPU, single-threaded per row (parallelized across rows by ThreadManager).
- **FP32 weights**: calls `input_step.dot(weight, hidden_step, false, true)` — standard
  CPU SGEMV.

**Cost**: For Qwen3-0.6B (hidden=1024, vocab=151,936):
- Q4_0: 151,936 rows × (dequant + sdot of 1024 floats) ≈ ~5-8ms per token on 4 threads
- This is a significant fraction of the ~17ms decode step (62ms generation / 10 tokens
  includes lm_head + sampling)

---

## 2. Migration Target: lm_head Q4_0 GEMV to NPU

### 2.1 Why lm_head (not embedding)

- **lm_head is a GEMM/GEMV** — the DSP's HMX array accelerates matmul. The existing
  `nntr_htp_bridge_gemm_q4_0` already does exactly this: `C[M,N] = B[M,K] × A[N,K]`.
  For decode, M=1, so it's a GEMV — the same bridge function handles it.
- **Embedding is a gather** — not a matmul. The DSP has no advantage for random-access
  row copies. The CPU's cache + memcpy is already optimal. NPU would add a FastRPC
  round-trip for zero compute benefit.
- **lm_head runs every decode token** — it's on the critical path. Embedding runs once
  at prefill.

### 2.2 The problem: tied weight layout

The lm_head weight is the **transposed** embedding weight. The embedding weight is
stored as `[vocab, hidden]` (row-major, one row per token). The FC layers store weights
as `[unit, hidden]` (also row-major). The existing `nntr_htp_bridge_gemm_q4_0` expects:

```
C[M,N] = B[M,K] × A_q4_0[N,K]
```

Where `A` is the weight in `[N, K]` layout (N=output dim, K=input dim).

For lm_head with tied weights:
- `B` = input hidden state, shape `[1, hidden]` (M=1, K=hidden)
- `A` = tied weight, shape `[vocab, hidden]` (N=vocab, K=hidden)
- `C` = logits, shape `[1, vocab]` (M=1, N=vocab)

This is **exactly** the layout the existing bridge function expects! The weight is
already `[vocab, hidden]` = `[N, K]`, and the input is `[1, hidden]` = `[M, K]`.

**The only issue**: the weight is stored in ARM q4_0x4 layout, and the bridge needs
q4x4x2 layout. But `ensure_uploaded()` in `hexagon_compute_ops.cpp` already handles
this conversion and caches it — and the embedding weight was **already uploaded** during
prefill (if the embedding layer uses the same weight tensor).

Wait — actually, the embedding layer and lm_head are separate layer instances. They
share the weight *data* (tied), but the weight is requested separately in each layer's
`finalize()`. The `ensure_uploaded()` cache is keyed on the weight pointer address. If
the tied weight is the same `Tensor` object (shared), the upload is already done. If not,
it will upload once on first lm_head call and cache it.

### 2.3 The plan

**The lm_head Q4_0 path can be migrated to NPU with zero new bridge functions.** The
existing `nntr_htp_bridge_gemm_q4_0` (FP32 activation) or `nntr_htp_bridge_gemm_q4_0_fp16`
(FP16 activation) already handles M=1 GEMV.

The migration is purely in `tie_word_embedding.cpp` (or `lm_head.cpp`):

#### Step 1: Add NPU dispatch to `incremental_forwarding_lmhead`

In `TieWordEmbedding::incremental_forwarding_lmhead`, replace the Q4_0 manual loop:

**Before** (current CPU path):
```cpp
} else if (weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0) {
    // ... manual dequant + sdot loop over all vocab rows ...
    const unsigned int num_blocks_per_row = (hidden_size + 32 - 1) / 32;
    const size_t row_stride = (sizeof(uint16_t) + 16) * num_blocks_per_row;
    // ... parallel_for over vocab rows ...
}
```

**After** (NPU dispatch):
```cpp
} else if (weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0) {
    // Dispatch to NPU via the existing gemm_q4_0_accel_fp32 path.
    // The weight is [vocab, hidden] = [N, K], input is [1, hidden] = [M, K],
    // output is [1, vocab] = [M, N]. This is exactly what the bridge expects.
    input_step.dot(weight, hidden_step, false, false);
}
```

Wait — `Tensor::dot` already dispatches to `gemm_q4_0_accel_fp32` when the context's
ComputeOps is `HexagonComputeOps`. The issue is that the current code **bypasses**
`Tensor::dot` and does a manual loop. So the fix is simply: **remove the manual Q4_0
branch and let `Tensor::dot` handle it**, same as the FP32 and Q6_K branches already do.

#### Step 2: Verify the weight layout matches what the bridge expects

The bridge's `nntr_htp_bridge_gemm_q4_0` computes:
```
C[M,N] = B[M,K] × A_q4_0[N,K]
```

The tied weight is stored as `[vocab, hidden]` = `[N, K]` where N=vocab, K=hidden.
The input is `[1, hidden]` = `[M, K]` where M=1, K=hidden.
The output is `[1, vocab]` = `[M, N]` where M=1, N=vocab.

**This matches.** The `Tensor::dot(weight, hidden, false, false)` call in the FP32
branch already does `input.dot(weight, hidden, false, false)` which computes
`hidden = input × weight^T`... wait, no. Let me re-check.

Actually, looking at the existing code more carefully:

```cpp
// FP32 branch:
input_step.dot(weight, hidden_step, false, false);
```

`Tensor::dot(a, b, c, transA, transB)` computes `c = a × b` (with optional transposes).
So this is `hidden = input × weight` where input is `[1, hidden]` and weight is
`[vocab, hidden]`. For this to produce `[1, vocab]`, weight must be treated as
`[hidden, vocab]` — i.e., it's **transposed**.

But the weight is stored as `[vocab, hidden]`. So `dot(false, false)` would compute
`[1, hidden] × [vocab, hidden]` which is dimensionally invalid.

Looking at the Q6_K branch:
```cpp
input_fp32.dot(weight, hidden_step, false, true);  // transB=true
```

This computes `hidden = input × weight^T` = `[1, hidden] × [hidden, vocab]` = `[1, vocab]`.
**This is correct.**

So the FP32 branch with `dot(false, false)` must be relying on the weight being stored
transposed. Let me check the weight dim in `finalize_lmhead`:

```cpp
ml::train::TensorDim weight_dim(
    1, is_nchw ? 1 : in_dim.channel(), is_nchw ? unit : 1,
    is_nchw ? in_dim.width() : unit, ...);
```

For NCHW: `weight_dim = (1, 1, unit, in_dim.width())` = `(1, 1, vocab, hidden)`.
So the weight is `[vocab, hidden]` and `dot(false, false)` would be
`[1, hidden] × [vocab, hidden]` — invalid.

But the Q4_0 manual loop does:
```cpp
logits[row] = sdot(hidden_size, input_data, 1, dequant_row.data(), 1);
```
which is `logits[vocab] = input[hidden] · weight[vocab, hidden]` — i.e., `input × weight^T`.

And the Q6_K branch uses `dot(false, true)` — `input × weight^T`.

So the FP32 branch with `dot(false, false)` seems wrong, unless the weight is actually
stored as `[hidden, vocab]` for FP32. This is a pre-existing issue, not our concern.

**For the Q4_0 NPU migration, we need `dot(false, true)`** — same as Q6_K.

But `Tensor::dot` with Q4_0 weights and `transB=true` — does the existing
`gemm_q4_0_accel_fp32` path support transposed weights?

Looking at `hexagon_compute_ops.cpp`:
```cpp
void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata, float *matCdata,
                           unsigned int M, unsigned int N, unsigned int K)
```

This computes `C[M,N] = B[M,K] × A_q4_0[N,K]`. The weight `A` is `[N,K]` = `[vocab, hidden]`.
So `C[1,vocab] = B[1,hidden] × A[vocab,hidden]` — this is `input × weight^T` already!

The bridge function **already** does the transposed multiply. So calling
`input_step.dot(weight, hidden_step, false, false)` with Q4_0 weights should dispatch
to `gemm_q4_0_accel_fp32` which computes `input × weight^T` — exactly what we want.

But wait — does `Tensor::dot` with `transB=false` and Q4_0 weight actually call
`gemm_q4_0_accel_fp32`? Let me check `float_tensor.cpp`:

The `Tensor::dot` implementation for Q4_0 weights calls `gemm_q4_0_accel_fp32` when
`supports_gemm_q4_0_accel_fp32()` returns true. The M/N/K mapping depends on the
transpose flags. With `transB=false`, the weight is used as-is `[N,K]`, and the GEMM
is `C[M,N] = B[M,K] × A[N,K]` — which is `input × weight^T` when weight is `[vocab, hidden]`.

Actually, `Tensor::dot(a, b, c, transA=false, transB=false)` computes `c = a × b`.
If `a` is `[1, hidden]` and `b` (weight) is `[vocab, hidden]`, this is invalid.
But if the weight is stored as `[hidden, vocab]` (transposed), then `a × b` = `[1, hidden] × [hidden, vocab]` = `[1, vocab]` — valid.

The weight_dim in `finalize_lmhead` for NCHW is `(1, 1, unit, in_dim.width())` =
`(1, 1, vocab, hidden)`. But the `dot` call with `transB=false` treats it as
`[hidden, vocab]`... unless the Tensor internally stores it differently.

Actually, looking at the Q4_0 manual loop, the weight is accessed as:
```cpp
const void *wrow = weight_data + row_stride * row;  // row 0..vocab-1
```
So weight is `[vocab, hidden]` row-major. And `sdot(hidden, input, dequant(wrow))`
computes `input · weight[row]` = `(input × weight^T)[row]`.

For the NPU bridge, `gemm_q4_0_accel_fp32(A=weight, B=input, C=output, M=1, N=vocab, K=hidden)`
computes `C[1,vocab] = B[1,hidden] × A[vocab,hidden]` = `input × weight^T`. **Correct.**

So the question is: does `Tensor::dot(input, weight, output, false, false)` with Q4_0
weight dispatch to `gemm_q4_0_accel_fp32(weight_ptr, input_ptr, output_ptr, M=1, N=vocab, K=hidden)`?

Looking at `float_tensor.cpp`'s dot implementation for Q4_0 weights — it should, because
that's exactly what the FC layer does. The FC layer calls `input.dot(weight, hidden, false, false)`
and it dispatches to NPU. The FC weight is `[unit, hidden]` = `[N, K]`, and the GEMM is
`C[M,N] = B[M,K] × A[N,K]`.

For lm_head, the weight is `[vocab, hidden]` = `[N, K]` — same layout. So
`input.dot(weight, hidden, false, false)` should dispatch to the same NPU path.

**The fix is literally: remove the Q4_0 manual loop and use `input_step.dot(weight, hidden_step, false, false)`.**

But wait — the Q6_K branch uses `dot(false, true)`. Why? Because Q6_K's `dotQnK` path
handles the transpose internally. The Q4_0 NPU bridge does not — it expects the weight
in `[N, K]` layout and computes `B × A` directly.

Actually, I think the confusion is that `Tensor::dot(a, b, transA, transB)` with
`transB=false` computes `c = a × b`, and with Q4_0 weights, the internal implementation
in `float_tensor.cpp` maps this to `gemm_q4_0_accel_fp32` which computes
`C[M,N] = B[M,K] × A[N,K]`. So if `a` is `[1, K]` and `b` (weight) is `[N, K]`,
then `a × b` would need `b` to be `[K, N]` for the multiply to work. But the bridge
computes `B[M,K] × A[N,K]` which is `a × b^T`.

So `Tensor::dot(input, weight, false, false)` with Q4_0 weight actually computes
`input × weight^T` — which is what we want! The "false, false" is the nntrainer API
convention, and the Q4_0 implementation internally handles the layout.

**Conclusion: the fix is to replace the Q4_0 manual loop with `input_step.dot(weight, hidden_step, false, false)`.**

---

## 3. Implementation Steps

### Step 1: Modify `TieWordEmbedding::incremental_forwarding_lmhead`

**File:** `Applications/CausalLM/layers/tie_word_embedding.cpp`

Replace the Q4_0 manual loop (lines ~351-391) with a single `dot` call:

```cpp
} else if (weight.getDataType() == nntrainer::TensorDim::DataType::Q4_0) {
    // Dispatch to NPU via Tensor::dot, same path as FC layers.
    // The weight is [vocab, hidden] = [N, K], input is [1, hidden] = [M, K],
    // output is [1, vocab] = [M, N]. The bridge computes C = B × A^T.
    nntrainer::Tensor input_fp32 =
        (input_step.getDataType() == nntrainer::TensorDim::DataType::FP32)
            ? input_step
            : input_step.clone(nntrainer::TensorDim::DataType::FP32);
    input_fp32.dot(weight, hidden_step, false, false);
}
```

### Step 2: Do the same for `LmHeadLayer::incremental_forwarding`

**File:** `Applications/CausalLM/layers/lm_head.cpp`

The `LmHeadLayer` already uses `input_step.dot(weight, hidden_step, false, false)` for
all weight types. But it doesn't handle Q4_0 specially — it falls through to the default
`dot` call. So **no change needed** in `lm_head.cpp` — it already dispatches to NPU
if the context is cdsp.

Wait — actually, looking at `lm_head.cpp` line 154:
```cpp
input_step.dot(weight, hidden_step, false, false);
```
This is the **only** path — no Q4_0 special case. So `LmHeadLayer` already dispatches
to NPU. The issue is only in `TieWordEmbedding`, which has the manual Q4_0 loop.

### Step 3: Verify weight upload caching

The tied embedding weight is shared between embedding and lm_head. If they're the same
`Tensor` object (tied), the `ensure_uploaded()` cache in `HexagonComputeOps` will hit
on the second call (lm_head) because the pointer is the same.

If they're **not** the same object (separate weight requests), the weight will be
uploaded twice — once for embedding (which doesn't use NPU) and once for lm_head. This
is wasteful but correct. To verify, check if `tie_word_embedding` uses
`context.getWeight(weight_idx[weight])` with the same index for both modes.

Looking at the code: `TieWordEmbedding` requests the weight once in `finalize_embedding`
or `finalize_lmhead` (mutually exclusive). The tying happens at the model graph level
(shared weight tensor), not at the layer level. So the weight pointer should be the same.

### Step 4: Handle the M=1 fallback

The existing `gemm_q4_0_accel_min_rows()` returns 1, meaning M=1 (decode) is offloaded.
But the comment in `hexagon_compute_ops.cpp` says decode on DSP is ~3.5x slower than CPU.
This is because decode is GEMV (bandwidth-bound), and the DSP has no bandwidth advantage.

**However**, the current CPU Q4_0 path is also slow (manual dequant + sdot loop). The
NPU path may actually be faster because:
1. The weight is already in rpcmem (zero-copy, no dequant per token)
2. The DSP's HMX array can do the GEMV in one shot
3. No per-row dequantization on CPU

**This needs to be benchmarked.** If the NPU is slower for M=1, we can set
`NNTR_HEXAGON_MIN_ROWS=2` to keep decode on CPU and only offload prefill.

### Step 5: Test

1. Build with `meson build -Denable-transformer=true -Denable-hexagon-cdsp=true`
2. Deploy to device
3. Run with `NNTR_USE_HEXAGON_CDSP=1` and verify output tokens match CPU run
4. Benchmark decode TPS (tokens/second) with and without NPU lm_head

---

## 4. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| NPU GEMV slower than CPU for M=1 | High | Medium | Benchmark; fall back to `NNTR_HEXAGON_MIN_ROWS=2` if needed |
| Weight not uploaded (separate tensor) | Low | Low | `ensure_uploaded` handles it; just uploads twice |
| Output mismatch (Q4_0 quantization) | Low | High | Compare logits with CPU reference |
| Batch mode flush interaction | Low | Low | lm_head runs outside batch scope (decode only) |

---

## 5. Alternative: Embedding NPU Migration (Not Recommended)

Embedding is a **gather** operation: for each token ID, copy a row from the weight
matrix. This is purely memory-bandwidth-bound, not compute-bound. The DSP has no
advantage over the CPU for random-access memory copies, and adds a FastRPC round-trip.

**Do not migrate embedding to NPU.**

---

## 6. Summary

The lm_head Q4_0 decode path can be migrated to NPU by **removing the manual CPU
dequant+sdot loop** in `TieWordEmbedding::incremental_forwarding_lmhead` and letting
`Tensor::dot` dispatch to the existing `nntr_htp_bridge_gemm_q4_0` bridge function.
This is a ~20 line code change in one file. The main risk is that M=1 GEMV on the DSP
may be slower than CPU (as documented in `hexagon_compute_ops.cpp`), so this must be
benchmarked.

# Whole-Graph NPU Delegation for QLoRA Training — Research Findings

**Date:** 2026-08-17
**Scope:** Can nntrainer's Hexagon/cDSP integration be restructured to delegate
an entire training graph to the NPU in (approximately) one FastRPC round trip,
the way `ggml-hexagon` does for inference — and specifically, what would it
take to run the `/home/anirudh/nntrainer-lora` QLoRA pipeline (Q4 frozen base
+ FP32 LoRA adapters) that way?

**Bottom line up front:** ggml-hexagon's "one call" is not literally one
FastRPC call — it's one session-establishment call followed by a persistent
shared-memory op queue that batches many ops per invocation with zero
per-op host round-trips, and it has **no backward/gradient/optimizer support
at all**. nntrainer cannot adopt that model outright because its execution
model is a layer-level interpreted graph, not an op-level IR — there is no
`ggml_cgraph` equivalent to hand to the DSP. Getting most of the practical
benefit (collapsing ~600-1176 GEMM-level FastRPC flushes per training step
down to ~60-120) is achievable without new DSP hardware kernels. Getting the
literal ask — the *entire* QLoRA forward+backward+optimizer step resident on
the NPU — requires new DSP-side backward kernels and a graph-capture
mechanism that don't exist anywhere today; it's a multi-week backend project,
not a port.

---

## 1. How ggml-hexagon actually achieves single-flush whole-graph delegation

The assumption "ggml-hexagon delegates the graph once via a single FastRPC
call" is subtly wrong and the actual mechanism matters for design purposes.

### 1.1 The FastRPC IDL boundary is minimal and control-only

`ggml-hexagon/htp/htp_iface.idl:13-20` — the entire interface:

```
interface htp_iface : remote_handle64 {
    AEEResult start(in uint32 sess_id, in uint64 dsp_queue_id, in uint32 n_hvx, in uint32 use_hmx, in uint64 max_vmem);
    AEEResult stop();
    AEEResult mmap(in uint32 fd, in uint32 size);
    AEEResult munmap(in uint32 fd);
    AEEResult profiler(in uint32 mode, in htp_iface_pmu_conf pmu);
    AEEResult etm(in uint32 enable);
}
```

There is **no per-op or per-graph "compute" RPC method**. `start()`
(`ggml-hexagon.cpp:2480-2500`) hands the DSP a `dspqueue_id` — a lock-free
shared-memory ring buffer (`dspqueue_create`/`dspqueue_export`) — exactly
once, at session init.

### 1.2 Op dispatch happens over the shared-memory queue, not FastRPC `invoke`

- `ggml_backend_hexagon_graph_compute()` (`ggml-hexagon.cpp:3352`) flattens
  `ggml_cgraph::nodes[]` into a flat op list and calls `enqueue_op()` /
  `flush()` (`ggml-hexagon.cpp:3390-3398`), which write/read the
  already-established `dspqueue` — a shared-memory write, not a kernel-level
  FastRPC round trip.
- One graph-compute call packs the whole batch (up to `opt_opbatch=1024` ops,
  `ggml-hexagon.cpp:69`) into one `htp_buf_desc[]` + `htp_tensor[]` +
  `htp_op_desc[]` blob per `dspqueue_write` (`ggml_hexagon_opqueue::push()`,
  `ggml-hexagon.cpp:2097-2166`).
- DSP-side, `htp_packet_callback()` (`htp/main.c:811-947`) drains one batch
  and executes **all ops in a tight loop with zero host round-trips between
  them** (`htp/main.c:897-920`), dispatching each via a plain
  `switch (octx->op)` in `execute_op()` (`htp/main.c:549-649`) into kernels
  like `op_matmul`, `op_softmax`, `op_rope`, `op_flash_attn_ext` (all in
  `htp/*.c`, using HVX/HMX + a worker pool for parallelism).
- Confirmed empirically: `docs/backend/snapdragon/README.md` perf logs show
  "graphs reused = 473/475" — essentially **one queue submission per
  generated token**, not one call for the whole session, and not one call
  per op either.

### 1.3 Weights are permanently resident in DSP-mapped memory; only activations cross per call

- `ggml_hexagon_shared_buffer` (`ggml-hexagon.cpp:215-330`) wraps
  `rpcmem_alloc2` + `fastrpc_mmap()` — pinned for the lifetime of the
  `ggml_backend_buffer` (i.e. lifetime of the loaded model / KV cache), not
  reallocated per call.
- Weight tensors use a distinct `repack_buffer_type`
  (`ggml-hexagon.cpp:1787-1811`): standard `block_q4_0` bytes are repacked
  once at load time into HTP's `q4x4x2` tile format
  (`ggml_backend_hexagon_buffer_set_tensor`, `:1655`).
- DSP-side, a 16-slot mmap cache (`htp_context::mmap[HTP_MAX_MMAPS]`,
  `htp/htp-ctx.h:17,64`) keyed by `fd` means a weight buffer mapped once
  stays mapped and is reused across every subsequent opbatch — the
  nntrainer bridge's own comments (`nntr-htp-bridge.cpp:49-53`) describe
  exactly this cache being thrashed by an earlier, naive per-call
  `rpcmem_alloc2` design.

### 1.4 There's a real two-phase split: compile-once vs execute-every-call

- **`ggml_backend_hexagon_graph_optimize(backend, gf)`**
  (`ggml-hexagon.cpp:3459-3533`) — a topology-rewrite pass: fuses runs of
  ADD/MUL/NORM/RMS_NORM (up to 16 ops), reorders nodes for better
  buffer/queue packing. Runs once per distinct graph *shape*, not every
  token.
- **`ggml_backend_hexagon_graph_compute(...)`** (`:3352-3401`) — the
  per-invocation step: a second, narrower fusion check
  (RMS_NORM+MUL → one op), remaps `ggml_op` → `htp_op_code`, enqueues, and
  flushes. Runs every `llama_decode`.

### 1.5 There is zero backward/gradient/optimizer support in ggml-hexagon itself

Exhaustive grep across `ggml-hexagon/` turns up no `GGML_OP_*_BACK`, no
gradient buffers, no optimizer-step code anywhere in `ggml-hexagon.cpp` or
`htp/*.c`. The training-related code that exists — `nntr-htp-bridge.cpp`
(compiled into the same `.so`, sharing `ggml_hexagon_session` /
`ggml_hexagon_shared_buffer` internals) — is a **separate, parallel bridge**
exposing raw SGEMM primitives (`nntr_htp_bridge_sgemm_fp32`,
`nntr_htp_bridge_sgemm_batch_fp32`) that nntrainer's CPU-side
`FullyConnectedLayer` calls for forward/`calcDerivative`/`calcGradient`. All
of the actual gradient math (which GEMM shape to compute, transpose flags,
accumulation, the optimizer step) is **entirely nntrainer's CPU-side
responsibility**; the DSP only ever executes a forward `MUL_MAT` op through
the exact same queue mechanism used for inference. There is no `HTP_OP_*`
opcode for a gradient or backward kernel anywhere in `htp/htp-ops.h` /
`htp/htp-ctx.h`, and `execute_op()`'s switch has no backward cases.

### 1.6 Host-side API surface (for reference)

- **Init session:** `ggml_hexagon_session::allocate()` (`:2367`) →
  `htp_iface_open()` (FastRPC, once) → `dspqueue_create`/`export` →
  `htp_iface_start(handle, sess_id, dsp_queue_id, n_hvx, use_hmx, max_vmem)`
  (FastRPC, once).
- **Build/register a graph once:** `ggml_backend_hexagon_buffer_type_alloc_buffer`
  / `_repack_buffer_type_alloc_buffer` (`:1787,1800`) to place
  weights/activations in DSP-visible memory; `ggml_backend_hexagon_graph_optimize`
  (`:3459`) for the one-time-per-shape fuse/reorder pass.
- **Execute repeatedly:** `ggml_backend_hexagon_graph_compute` (`:3352`);
  `ggml_hexagon_session::enqueue_op()` / `flush()` (`:2325,2333`) as the raw
  primitives underneath (which is exactly what the nntrainer bridge calls
  directly, bypassing `ggml_cgraph`).

---

## 2. Why nntrainer can't adopt that model outright

### 2.1 The execution model is layer-level interpreted, not an op-level IR

`NetworkGraph` (`nntrainer/graph/network_graph.h:37`) is a topologically
sorted list of `LayerNode` objects (`GraphCore graph`) — not an op-level IR
like `ggml_cgraph`. `NetworkGraph::compile()`
(`nntrainer/graph/network_graph.cpp:55-100`) does real scheduling work
(`topologicalSort()` at `:87`, `setExecutionOrder()` at `:89` for tensor
lifetime, `inPlaceOptimize()` at `:92`) but never lowers a `LayerNode` into
primitive tensor ops — the unit of the compiled structure remains a whole
layer, whose internals are an opaque C++ black box until it actually runs.

Execution is a plain interpreted loop re-walked on every call:
`NetworkGraph::forwarding()` (`:393-416`) and `backwarding()` (`:440-538`)
both iterate the sorted `LayerNode` list and invoke
`node->forwarding(training)` / `calcGradient()` / `calcDerivative()`
(`layer_node.cpp:786-816`), which call straight into
`Layer::forwarding`/`calcDerivative`/`calcGradient` — **pure virtual
methods** (`layer_devel.h:209,237,246`) that each concrete layer (FC,
RMSNorm, RoPE-inside-mha_core, softmax, etc.) implements with however many
loops, branches, and `Tensor::dot()` calls it wants. There is no "compile
once, execute many times" plan: every step re-dispatches every virtual call.

The `compiler/` directory (TFLite/ONNX/flatbuffer interpreters/exporters)
exists only for model import/export, not for producing an executable
op-plan for an accelerator.

### 2.2 NPU dispatch granularity is one flush per GEMM call-site

`HexagonContext::initialize()` (`hexagon_context.cpp:83-106`) attaches a
`ComputeOps` vtable (`HexagonComputeOps`) to every tensor created under
`engine=cdsp`. That vtable overrides `gemm_q4_0_accel_fp32`,
`gemm_q4_0_batch_fp32`, and `sgemm_fp32` (`hexagon_compute_ops.cpp:228-317`)
— every other method forwards straight to the CPU. So the granularity of
NPU dispatch is exactly **one call site per matmul**, inside whichever
layer's `forwarding()`/`calcGradient()`/`calcDerivative()` happens to invoke
`Tensor::dot()` at that moment, e.g.
`input_.dot(weight_, hidden_, ...)` in `fc_layer.cpp:362`. Each such call
becomes one `nntr_htp_bridge_*` call, ending in one blocking `flush()`
(FastRPC-class round trip) — the bridge's own comment
(`nntr-htp-bridge.cpp:36-46`) states this plainly for inference: "28 blocks
x 7 FC = 196 GEMMs/token," ~3.3-3.7 ms/GEMM overhead dominated by host-side
bookkeeping, not DSP compute.

For training, the existing docs (`HYBRID_NPU_TRAINING_REPORT_2026-08-13.md`,
`NPU_BACKWARD_SCOPE_2026-08-13.md`) put concrete numbers on this: **~9-11
forward GEMMs/block + ~19 backward GEMMs/block × 28 blocks ≈ 600+ GEMM
dispatches per Qwen3 training step**, batchable at best to ~336 flushes
with today's hand-built batching. The QLoRA variant is worse:
`QWEN3_NPU_TRAINING_ARCHITECTURE.md` counts **~1176 NPU GEMMs per training
step** once LoRA's extra forward/backward GEMMs are included.

**Current regression:** `sgemm_fp32` — the FP32 GEMM path that
backward/training needs — is presently **hard-bypassed to CPU** in
`hexagon_compute_ops.cpp` (confirmed in the live diff of this repo) due to
an unresolved transpose-logic bug (`transB=1`/`transA=1`) in the bridge. So
today, real NPU utilization in this repo is **Q4_0-forward-only**; all
training compute — even the GEMMs — runs on CPU.

### 2.3 Existing batching is real but narrow, hand-built per layer, not automatic

- `gemm_q4_0_batch_fp32` (`hexagon_compute_ops.cpp:253-282`) — fuses Q/K/V
  or gate/up (weights sharing one activation) into one flush. Used by
  `QKVLayer`/`GateUpLayer`. **This is the only batching mechanism actively
  exercised from nntrainer today.**
- `nntr_htp_bridge_sgemm_batch_fp32`, `nntr_htp_bridge_flash_attn`,
  `nntr_htp_bridge_ffn_swiglu`, `nntr_htp_bridge_fused_fc_forward` all exist
  in the bridge library but have **zero live call sites** from nntrainer's
  layers (`mha_core.cpp` does not call the flash-attention bridge; Qwen3
  uses separate `gate_up_layer`+`swiglu`+FC, not the fused-FFN path; the
  fused-FC-forward path is documented as buggy).
- There is no graph-level scheduler that could fuse across layer
  boundaries automatically — every existing batching mechanism had to be
  hand-built into a specific layer class, precisely because `NetworkGraph`
  gives no op-level view for a generic fuser to work with.

### 2.4 Two independent reasons control returns to the CPU between dispatches

1. **Genuine data-flow dependency:** RoPE, softmax, RMSNorm, SwiGLU,
   residual-add, embedding lookup, and the Adam optimizer step are real
   CPU-only ops sitting *between* GEMMs in the actual computation (e.g.
   `gate = X·W_gate` → SwiGLU → `out = act·W_down`) — the DSP genuinely
   cannot proceed without handing the intermediate tensor back.
2. **An abstraction artifact, independent of (1):** even two DSP GEMMs with
   nothing but a data dependency between them (e.g. `dX = dY·Wᵀ` then
   `dW = Xᵀ·dY` inside one FC layer's backward) are still two independent
   `ComputeOps` virtual calls made from different points in the interpreted
   per-layer loop, with no "session" concept to defer the flush. Batching
   only happens where a specific layer author explicitly collected several
   weight pointers and called one batched method by hand.

### 2.5 Could `NetworkGraph::compile()` be extended into an AOT device plan?

`compile()` already produces the scheduling half of what a device plan
needs (fixed topological order, execution-order/tensor-lifetime metadata).
What's missing — and what makes this hard, not just tedious — is that
`compile()` never lowers `LayerNode`s into primitive ops: nodes remain
`Layer` objects with an opaque virtual `forwarding()` body. ggml's
`cgraph` works because *every op* is already a node in an op-level IR
before any backend sees it; nntrainer's IR stops one level higher, at the
layer, and each layer's internal op sequence is invisible until it actually
executes. Fixing this doesn't require abandoning nntrainer's architecture,
but it does require introducing a genuine op-level IR beneath
`LayerNode` — either (a) a first-run trace-and-replay of a given layer's op
sequence (valid as long as shapes are fixed for a training run — comparable
to CUDA-graph capture), or (b) rewriting layers to emit explicit op nodes
into a builder instead of doing math directly. Neither exists anywhere in
`nntrainer/graph/` or `nntrainer/compiler/` today.

---

## 3. QLoRA repo (`nntrainer-lora`) — current state and a correction

### 3.1 What's implemented

- LoRA on `fc_layer.cpp`: `LoraRank`/`LoraAlpha`/`LoraQAT`/`LoraWeightQ4`
  properties; `loraA` (in_dim×rank, LeCun-normal init) / `loraB`
  (rank×unit, zero init) — deliberately *not* the reverse Hu et al. order
  (documented divergence above lr≈1e-6 with the reverse order).
- Forward: base weight (Q4_0/Q6_K) is **dequantized to the activation
  dtype (W4A16)**, not run through the fused W4A8 kernel — explicitly to
  avoid the Q8_0 activation-quantization noise that kernel introduces —
  then `hidden = base(X) + (X·loraA·loraB)*scaling`.
- Backward: base weight frozen (no gradient computed for it at all when
  `lora_rank>0`); only `dA`/`dB` computed via 3 GEMMs through
  `dot_deriv_wrt_1/2`.
- Optional Q4 QAT (`--lora_qat`, off by default): `fakeQuantizeQ4_0()` —
  per-32-block EMA-tracked scale + straight-through estimator — lets
  trained adapters be saved as real Q4_0 for inference.
- All CausalLM layers have real backward implemented except
  `embedding_layer` (`supportBackwarding()=false`) and `qkv_layer`
  (present but stubbed and unused).
- Training loop (`train_qwen3_lora.cpp`): Adam-only optimizer, W4A16
  dequantized-and-cached base weights, FP32 LoRA A/B always. Attention
  forward/backward (`mha_core.cpp`) is pure scalar CPU loops — no BLAS, no
  SIMD, no threading at all, confirmed by this repo's own
  `docs/training_optimization_notes.md`.

### 3.2 No hexagon backend exists here at all

`engine.cpp` registers only `cpu`, optional `gpu` (OpenCL), and an optional
dynamically-loaded QNN plugin — there is no `cdsp` context, no
`nntrainer/hexagon/` directory, anywhere in this repo.

### 3.3 Correction to the "native Q4→FP16×FP16 kernel" characterization

The kernel found (`nntr_gemm_q4_0_4x8_q8_0_fp16`, NEON,
`nntr_ggml_impl_neon.cpp`) is **not** a plain FP16×FP16 (unquantized
activation) kernel. It's **W4A8→FP16-output**: the activation is quantized
on-the-fly to Q8_0 before the integer dot product, and only the final
accumulator is cast to FP16 on the way out. The LoRA repo's own code
comments explicitly avoid this kernel for the base-weight matmul in
training, precisely because the on-the-fly Q8_0 activation quantization
adds noise that hurts convergence — it uses dequantize-once-then-dense-GEMM
(true W4A16 behavior, but via unpacking, not a fused low-precision kernel)
instead. **A genuine fused W4A16 kernel — 4-bit weight, unquantized FP16
activation, no activation-quantization noise — does not exist on CPU or
DSP today.** This matters directly for the NPU goal: porting `engine=cdsp`
onto this repo's FC layer as-is would dispatch the *dequantized* weight
through `sgemm_fp32` (currently disabled anyway), not through the fast
native Q4_0 HMX path — so it would not get the ~10-15× Q4_0-native speedup
the earlier `LORA_NPU_FEASIBILITY_STUDY.md` implicitly assumes, only the
~5× `sgemm_fp32` number, and only once that bridge bug is fixed.

---

## 4. What genuine whole-graph NPU delegation for QLoRA training would require

Roughly three separable tiers of increasing cost, not evaluated further here
per current scope (this document is findings-only), but recorded for future
reference:

- **Tier 0 (days):** fix the `sgemm_fp32` transpose bug so FP32
  backward GEMMs can dispatch to NPU at all.
- **Tier 1 (weeks, no new DSP kernels):** give the bridge a
  `begin_batch()/end_batch()` scope so layers enqueue into the same
  `dspqueue`-style mechanism without flushing until a real CPU-dependency
  boundary is hit — collapsing per-GEMM flushes into ~1-2 per block per
  direction (order-of-magnitude reduction, matching ggml-hexagon's own
  batching granularity, no new hardware kernels needed).
- **Tier 2 (weeks):** reuse ggml-hexagon's *existing* forward HTP kernels
  for RMSNorm/RoPE/softmax/SiLU (`htp/rope-ops.c`, `htp/softmax-ops.c`,
  `htp/act-ops.c` already exist) instead of nntrainer's CPU
  reimplementations, shrinking the CPU boundary further for inference.
- **Tier 3 (multi-week R&D, the literal ask):** author new DSP-side
  backward kernels (softmax-backward, RMSNorm-backward, RoPE-inverse,
  SwiGLU-backward, attention-backward, cross-entropy+grad), a DSP-resident
  Adam optimizer-step kernel with persistent gradient/optimizer-state
  buffers, a genuine W4A16 (no-activation-quant) HMX matmul kernel for the
  frozen QLoRA base weight, and a host-side graph-capture mechanism
  (trace-and-replay or an explicit op-emitting layer rewrite) since
  nntrainer has no op-level IR to hand to the DSP today. This is
  effectively building a new autograd-capable Hexagon backend, not a port
  of existing pieces.

---

## 5. Key files referenced

| Area | File |
|---|---|
| ggml-hexagon FastRPC IDL | `ggml-hexagon/htp/htp_iface.idl` |
| ggml-hexagon graph compute/optimize | `ggml-hexagon/ggml-hexagon.cpp:3352,3459` |
| ggml-hexagon DSP-side op loop | `ggml-hexagon/htp/main.c:811-947` |
| ggml-hexagon shared buffer / weight residency | `ggml-hexagon/ggml-hexagon.cpp:215-330,1787-1811` |
| nntrainer↔bridge training GEMMs | `ggml-hexagon/nntr-htp-bridge.cpp` |
| nntrainer graph/execution model | `nntrainer/graph/network_graph.cpp`, `nntrainer/layers/layer_node.cpp`, `nntrainer/layers/layer_devel.h` |
| nntrainer Hexagon dispatch | `nntrainer/hexagon/hexagon_context.cpp`, `nntrainer/hexagon/hexagon_compute_ops.cpp` |
| QLoRA implementation | `nntrainer-lora/nntrainer/nntrainer/layers/fc_layer.cpp` |
| QLoRA training loop | `nntrainer-lora/nntrainer/Applications/CausalLM/train_qwen3_lora.cpp`, `lora_train.cpp` |
| Q4×Q8→FP16 CPU kernel | `nntrainer-lora/nntrainer/nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_neon.cpp:697` |

Signed-off-by: Anirudh <anirudh1023@gmail.com>

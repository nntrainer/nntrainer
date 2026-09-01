# GPU Inference for CausalLM — Architecture, Features, Build & Run

This guide covers running the CausalLM stack (`nntr_causallm`) on the GPU. It
applies to **Gemma4-E2B**, **Gemma2-2B**, and **Qwen3-0.6B** (and any future
CausalLM model that opts into the GPU layers). The same model definition and the
same kernel sources run on **four device classes** from one tree:

| Class | Device (verified) | Compute path | FC / KV path |
|-------|-------------------|--------------|--------------|
| **Adreno** (OpenCL) | Galaxy S26 Ultra, Adreno 840 | `image2d` `read_imageui` (texture-L1 cache) | image2d weights + image2d KV mirror |
| **Intel-XMX** (OpenCL) | Xe2 / Xe3 iGPU (e.g. Panther Lake `8086:b0a0`) | systolic **DPAS** (`int8` matrix-MAD) for prefill, dp4a for decode | `cl_mem` buffer + SVM flash attention |
| **Intel-noXMX** (OpenCL) | Meteor Lake / older NEO | `dp4a` (`cl_khr_integer_dot_product`) | `cl_mem` buffer + SVM flash attention |
| **CUDA-discrete** | RTX 4070 / 5060 (Ada/Blackwell) | cuBLAS **int8 IMMA** Tensor Cores + dp4a | UVM weights, device-mirror KV |
| **CUDA-integrated** | Jetson **Orin** (Tegra sm_87) | cuBLAS int8 (K-chunked) + GEMM attention | UVM weights, pass-through KV |

The activation/compute precision is **FP16** and weights are **4-bit**
(`model_tensor_type: "QINT4-FP16"`). The quantized matmuls are **w4a8**: 4-bit
weights × FP16 activations that are quantized to **int8 on the GPU** at each FC
input. FP16 activations are the precondition for full GPU residency (attention /
RoPE / KV cache all on device).

> The engine is chosen by `causallm_engine()` (`llm_util.hpp`): it defaults to
> **gpu** (OpenCL), drops to host CPU when `NNTR_ENGINE=cpu`, and selects CUDA
> when `NNTR_ENGINE=cuda`. There is **no** `engine` key in `nntr_config.json` —
> the single neutral graph resolves to the right backend at finalize, then the
> GPU path is gated by the env vars below.

> Conventions: **prefill** = the M-row prompt GEMM (`M=1024` in the perf tables);
> **decode** = the `M=1` per-token step. The two are structurally different
> (prefill is compute-bound, decode is bandwidth/latency-bound), which is why so
> many features below have a separate prefill and decode path.

---

# Part I — Build & Run

## 1. Quick start

### Intel Xe (x86, `build_cl`)

```bash
# Build (meson + ninja). The ninja target is the *path*, not the bare name.
ninja -C build_cl Applications/CausalLM/nntr_causallm
# (first-time configure, if build_cl is absent:)
#   meson setup build_cl . -Denable-opencl=true -Denable-fp16=true \
#       -Dwerror=false --buildtype=release

# Run (canonical Intel env). NNTR_V8C_BUF is MANDATORY (NEO can't read_imageui);
# NNTR_GPU_CLMEM_POOL is MANDATORY for coherence; NNTR_XE3_SYNC is MANDATORY on
# Xe3 (Panther Lake). XMX is now auto-selected from device caps — no NNTR_FC_XMX.
NNTR_GPU_SVM_POOL=1 NNTR_V8C_BUF=1 NNTR_MHA_GPU=1 NNTR_FC_INT8_GPU=1 \
NNTR_GPU_CLMEM_POOL=1 NNTR_XE3_SYNC=1 \
  ./build_cl/Applications/CausalLM/nntr_causallm <MODEL_DIR> ["prompt"]
```

x86 links the host's Intel NEO ICD (`/lib/x86_64-linux-gnu/libOpenCL.so.1`) —
nothing to push. Wrapper scripts live in `.claude/scripts/run_gemma*_x86.sh`.

### CUDA (x86 / Jetson, `build_cuda`)

```bash
ninja -C build_cuda Applications/CausalLM/nntr_causallm
BD=build_cuda
export LD_LIBRARY_PATH="$BD/Applications/CausalLM:$BD/Applications/CausalLM/layers:$BD/nntrainer:$BD/api/ccapi:/usr/local/cuda/lib64"

# Discrete RTX (canonical set). block-Q attention incl. the head_dim=128 kernel
# means qwen3 needs NO NNTR_CUDA_GEMM_ATTN. GEMM_ATTN now auto-enables on Orin.
NNTR_ENGINE=cuda NNTR_CUDA_DEV_ACT=1 NNTR_RMSNORM_CUDA_OFF=all NNTR_CUDA_ROPE=1 \
NNTR_CUDA_ATTN=1 NNTR_CUDA_QKNORM=1 NNTR_CUDA_GEGLU=1 NNTR_CUDA_ELTWISE=1 \
NNTR_CUDA_KV_UVM=1 NNTR_CUDA_VCOPY_PREFILL=1 NNTR_CUDA_FLASH_DECODE=64 \
NNTR_CUDA_BLOCKQ=1 NNTR_FC_CUDA_CUBLAS=1 NNTR_CUDA_PREWARM=1 \
  "$BD/Applications/CausalLM/nntr_causallm" <MODEL_DIR> ["prompt"]
```

NVRTC compiles kernels at runtime for the live device arch (sm_89 Ada, sm_120
Blackwell, sm_87 Orin …) and caches the PTX on disk — no fatbin to ship. On Orin
use `run_gemma4_fast.sh` (the host-coherent safe-set, see §7.3.3).

### Adreno (Android, ndk-build)

```bash
export ANDROID_NDK=/path/to/android-ndk        # e.g. ~/Android/Sdk/ndk/27.2.12479018

# (a) Build libnntrainer + libccapi (meson). package_android.sh leaves OpenCL OFF
#     by default, so force it on the first time, then ninja install:
./tools/package_android.sh
meson configure builddir -Denable-opencl=true -Dwerror=false
ninja -C builddir install
#     ⚠️ ndk-build does NOT run meson's .cl->.cpp codegen. If you edited a kernel,
#     regenerate first (.claude/regen_cl.py / build_lib.sh) or you get silent
#     stale-kernel garbage. See §11.

# (b) Build the app (ndk-build):
cd Applications/CausalLM/jni
ndk-build NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj \
  APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk \
  causallm_core nntrainer_causallm -j$(nproc)

# (c) Deploy ALL SIX artifacts to /data/local/tmp/nntrainer/causallm (see §9):
#     nntrainer_causallm, libcausallm_core.so, libccapi-nntrainer.so,
#     libnntrainer.so, libOpenCL.so, libc++_shared.so

# (d) Run (canonical Adreno env). NNTR_KV_IMG_ATTN selects image2d KV;
#     NNTR_GPU_CLMEM_POOL is MANDATORY for coherence.
adb -s <SERIAL> shell 'cd /data/local/tmp/nntrainer/causallm && \
  LD_LIBRARY_PATH=$PWD NNTR_FC_INT8_GPU=1 NNTR_MHA_GPU=1 NNTR_GPU_SVM_POOL=1 \
  NNTR_KV_IMG_ATTN=1 NNTR_GPU_CLMEM_POOL=1 \
  ./nntrainer_causallm models/<MODEL_DIR> ["prompt"]'
```

## 2. Canonical environment sets

The minimal, verified-coherent env sets. Everything else (§8) is tuning /
diagnostics. **Both pools are mandatory together** — `NNTR_GPU_SVM_POOL` orders
the producer before the consumer (in-order queue) and `NNTR_GPU_CLMEM_POOL`
gives them a shared device buffer; drop either and output collapses.

| | Adreno | Intel (Xe / Meteor) | CUDA (RTX / Orin) |
|---|---|---|---|
| `NNTR_ENGINE` | — (gpu default) | — (gpu default) | `cuda` |
| `NNTR_FC_INT8_GPU` | `1` | `1` | — (FC is `cuda_fc`) |
| `NNTR_MHA_GPU` | `1` | `1` | `NNTR_CUDA_ATTN=1` |
| `NNTR_GPU_SVM_POOL` | `1` | `1` | — (UVM) |
| `NNTR_GPU_CLMEM_POOL` | `1` (coherence) | `1` (coherence) | `NNTR_CUDA_DEV_ACT=1` |
| `NNTR_V8C_BUF` | — (image2d) | `1` (buffer/dp4a — mandatory on NEO) | — |
| `NNTR_KV_IMG_ATTN` | `1` (image2d KV) | — | — |
| `NNTR_XE3_SYNC` | — | `1` **mandatory on Xe3** | — |
| GEMM family | dp4a (image) | **XMX auto** (caps) → dp4a | cuBLAS IMMA + block-Q |

**Long context.** The sliding-window KV ring and chunked prefill are opt-in and
off by default; add `NNTR_KV_WINDOW_RING=1` (which also turns chunked prefill on
at 4096 rows) on top of a canonical set, and `NNTR_CUDA_SPLITKV_PREFILL=1` on
CUDA. Turning the ring on changes what `init_seq_len` means: with chunking off
it is the prompt ceiling and a longer prompt is truncated to it, while with
chunking on it is only the height of one chunk's activation plane — the prompt
is then bounded by the KV budget (`max_timestep`), not by `init_seq_len`, and
the truncation warning reports the KV budget instead.

`NNTR_FC_GPU` appears in some wrapper scripts but is a **legacy no-op alias** —
the real FC gate is `NNTR_FC_INT8_GPU`. `NNTR_FC_XMX` and `NNTR_CUDA_GEMM_ATTN`
are now **caps-derived defaults** (auto-on where the hardware supports them) and
are only needed as explicit overrides (§12).

## 3. Measured performance

Prefill at **M=1024** (`prompt_1p2k.txt`); decode at the corresponding ~1K
context. Best-of-3, all coherent (the generated 1K passage continues
sensibly). Models: Gemma4-E2B / Gemma2-2B / Qwen3-0.6B, all QINT4-FP16
(`gemma4_lmint4` / `gemma2_lg_q6k` / `qwen3_lg_q6k`).

| Model | Adreno 840 | Intel Xe3 | CUDA RTX 5060 |
|-------|-----------:|----------:|--------------:|
|  | prefill / decode | prefill / decode | prefill / decode |
| Gemma4-E2B | **2454** / 18.2 | **2964** / 18.2 | **5400** / 35.3 |
| Gemma2-2B | **827** / 14.5 | **1756** / 13.8 | **3151** / 50.7 |
| Qwen3-0.6B | **2151** / 30.0 | **2301** / 37.6 | **4511** / 84.2 |

Notes:
- **Gemma4 prefill** is fast despite ~2B params because ~57% of its layers share
  KV and skip prefill (`skip_prefill`, Gemma4-only — see §10).
- **Intel XMX** lifts Xe3 prefill ~1.7–1.9× over dp4a (now auto-selected).
- **CUDA** block-Q attention beats cuBLAS for every head_dim on RTX; the `d128`
  kernel takes Qwen3 prefill 916 (fall-through) → 4511 with no `GEMM_ATTN`. The
  cuBLAS int8 K-chunk (sm_87 large-K workaround) is gated to integrated only, so
  discrete RTX runs the full-K FC.
- Decode is the structurally hard case on every backend (single `M=1` query,
  bandwidth-bound, per-op dispatch floor); §6.2 / §6.4 cover the levers.

---

# Part II — How GPU support is implemented

This is the engineering overview: *what we built to run a transformer on the GPU*
and why it is structured this way. The guiding principle is **additive** —
nothing in the CPU/training path changed. A GPU backend is a new `Context` +
allocator + op-table + a handful of GPU layers + a kernel library, all behind
`#if ENABLE_OPENCL` (and `#if ENABLE_CUDA`), so an `enable-opencl=false` /
`enable-cuda=false` build is byte-identical to before.

## 4. The additive backend architecture

`engine.cpp` registers `"cpu"→AppContext`, `"gpu"→ClContext`,
`"cuda"→CudaContext` (each under its own `#if`), and `dlopen`s
`libqnn_context.so` for `"npu"`. A backend is exactly four things:

1. **Context** — `ClContext` / `CudaContext` (`Singleton<Context>`): the
   per-engine layer-factory map + kernel/PTX cache. Registered at link time next
   to the others; a new device is "add a `Context` under one `#if`" with zero
   edits to models or other backends.
2. **MemAllocator** — decides what *kind* of memory a pooled tensor gets.
   `ClSVMAllocator` routes `MemoryPool` alloc/free through `clSVMAlloc` (one
   host+device pointer = **device-resident, no copy step**); `CudaMemAllocator`
   uses `cudaMallocManaged` (UVM) plus a `device_only` `cudaMalloc` variant for
   the activation pool. The CPU base is host `aligned_alloc`. The calloc/SVM
   macros were removed from `MemoryPool` itself.
3. **ComputeOps op-table** — routes tensor ops to the right kernels (§4.1).
4. **GPU layer factories** — the `cl_layers` (`FullyConnectedLayerCl`,
   `RMSNormLayerCl`, `SwiGLULayerCl`, the now-neutral `GeGLULayer`,
   `AdditionLayerCL`, `Concat`/`Reshape`/`TransposeLayerCl`) and `cuda_layers`
   (`CudaFcLayer`, …), each registered only if its kernels compile.

### 4.1 The op-table (`ComputeOps`) and per-context dispatch

`ComputeOps` (`tensor/cpu_backend/compute_ops.h`) is an abstract vtable: base
bodies throw "not implemented", and every accelerator-only op pairs with a
`supports_*()` predicate that defaults `false`. `CpuComputeOps` forwards each op
to the arch-dispatched `nntrainer::sgemm` etc.; `ClComputeOps`
(`cl_operations/cl_compute_ops.cpp`) overrides only the int4/Q4_0 GEMM/GEMV
virtuals (flipping `supports_*()→true`); `CudaComputeOps` derives from
`CpuComputeOps` (UVM is host-coherent, so un-accelerated ops just run on the
managed pointer) and overrides what it accelerates.

`Tensor::getOps()` returns the **per-context** table — a tensor on a CL context
dispatches to GPU kernels, one on a CPU context computes on the host, with no
`#ifdef` at the call site. The `ContextData` carrying that table is attached to
every tensor a layer owns (`LayerNode::configureRunContext`), so even activation
tensors reach the right backend. This is the mechanism the layer-fork collapse
(§6.6, §12) builds on.

### 4.2 Implementation matrix — what each device class uses

The four pieces above, instantiated per device class. The key structural fact:
the three **OpenCL** classes (Adreno / Intel / Intel-XMX) share one `ClContext` /
`ClSVMAllocator` / `ClComputeOps` — the device difference is **not** a separate
class but the *kernel path the op-table picks*, derived from `DeviceCaps`. **CUDA**
has its own trio. The **GPU layers are backend-neutral** (the same class on every
backend; the divergence is inside `ComputeOps`).

| Device class | Context | MemAllocator | ComputeOps | GPU layer |
|---|---|---|---|---|
| **Adreno** (Qualcomm) | `ClContext` | `ClSVMAllocator` (coarse SVM) | `ClComputeOps` | neutral (shared) |
| **Intel** non-XMX (Meteor/NEO) | `ClContext` | `ClSVMAllocator` | `ClComputeOps` | neutral (shared) |
| **Intel-XMX** (Xe2/Xe3) | `ClContext` | `ClSVMAllocator` | `ClComputeOps` | neutral (shared) |
| **CUDA** (RTX/Orin) | `CudaContext` | `CudaMemAllocator` (UVM + device-only act pool) | `CudaComputeOps : CpuComputeOps` | neutral (shared) |

Per-piece highlights (full detail in §6–§7, §12):

- **Context** — link-time self-registration (`"gpu"` / `"cuda"`); kernel compile +
  cache (OpenCL `clBuildProgram` + disk KERNEL_CACHE; CUDA **NVRTC** + PTX disk
  cache + CUmodule cache); a once-probed `DeviceCaps`; the `ExecPlan` resolver
  (`DP4A` / **`XMX`** if `caps.subgroups` / `CUBLAS`) and the `ModelFeatures ×
  DeviceCaps` matcher; in-order SVM queue (+ `NNTR_XE3_SYNC` on Xe3) vs CUDA-graph
  capture/replay.
- **MemAllocator** — OpenCL `clSVMAlloc` (one host+device pointer) vs CUDA
  `cudaMallocManaged` (UVM) + a `device_only cudaMalloc` activation pool; the
  capability predicates (`isHostAddressable`/`isDeviceVisible`/`isSVM`/
  `supportsDevicePool`); `makePool()` chooses SVM pool ↔ `ClBufferPool`
  (`GPU_CLMEM`).
- **ComputeOps** — the accelerator quantized GEMM/GEMV (`gemm_q4_0_*`,
  `gemv_int4_*`, `sgemm_int4_*`) plus the whole-op table (`fc`,
  `fc_prebuild_weight`, `geglu`, `swiglu`, `residual_op`) and the host copies.
- **GPU layer** — one neutral class per op (`FullyConnectedLayerCl`, `GeGLULayer`,
  `SwiGLULayer`, the core `AdditionLayer`, …), registered on both the gpu and cuda
  contexts; the not-yet-collapsed forks (`RMSNormLayerCl`/`CudaRMSNormLayer`,
  `Concat`/`Reshape`/`TransposeLayerCl`); the app-side `MHACoreLayer` /
  `reshaped_rms_norm` / lm_head / `per_layer_slice`.

**Where the device difference actually lives** — the kernel path each op-table op
dispatches to per device class:

| op-table op | Adreno | Intel (non-XMX) | Intel-XMX | CUDA |
|---|---|---|---|---|
| `fc` prefill GEMM | image2d v8c (`read_imageui`) | buffer v8c (dp4a) | `gemm_xmx_i4` systolic DPAS | `cuda_fc_qint4` dp4a / cuBLAS int8 IMMA |
| `fc` decode GEMV | 64-wide coop split-K | 64-wide coop split-K | dp4a (XMX is prefill-only) | dp4a GEMV |
| attention (`mha_core`) | image2d KV mirror (`two_conv_attention`) | SVM flash + split-KV flash-decode | SVM flash + split-KV | block-Q (d128/256/512) + flash-decode + cuBLAS GEMM-attn |
| `geglu` / `swiglu` | `geglu_cl_op` / `swiglu_cl_op` (SVM/cl_mem) | same | same | device fp16 kernel / host-on-UVM |
| `residual_op` (add) | `clmem_residual` / `add_i_cl` / `gpu_copy_f16` | same | same | host add on UVM (+ fused `cuda_add_fp16`) |
| RoPE | `rotary_emb` + LUT-cap | same | same | `cuda_rope` (device-pos) |
| q/k/v-norm | `reshaped_rms_norm` GPU-resident | same | same | `cuda_rmsnorm` + QKNORM |
| lm_head | Q6_K GEMV + on-GPU 2-pass argmax | same | same | cuda Q6_K GEMV + argmax |

Only **FC** and **attention** truly diverge per OpenCL device (image vs buffer vs
DPAS); the activations / residual / RoPE / norms run the *same* OpenCL kernel on
Adreno and both Intel variants, and only split host-vs-device on CUDA.

## 5. Engine selection

`causallm_engine()` returns `"gpu"` by default, `"cpu"` under `NNTR_ENGINE=cpu`,
`"cuda"` under `NNTR_ENGINE=cuda`. Every built layer carries
`engine=causallm_engine()`; there is **no** engine key in the config. One model
definition runs on host or any GPU backend purely by env, with no per-model
config edit and no graph fork.

## 6. The OpenCL compute pipeline (device-agnostic core)

These features are shared by **all** OpenCL devices (Adreno + Intel). The
per-device specializations are §7.

### 6.1 The w4a8 `v8c` quantized FC GEMM — the core compute

FC layers dominate cost, so this is where the speedup lives
(`blas_kernels.cpp` + `cl_kernels/int8_int4_gemm_v8c.cl`). The scheme is a
paper-aligned "8/4/4" **w4a8**:

- **Weights** are int4 nibbles **offset-encoded** (`stored = real + 8`, range
  `[0..15]`) with a per-channel scale and a precomputed per-channel row-sum. The
  on-disk KAI Section-A nibbles are permuted *once* at load into the v8c
  row-major layout (no dequant/requant), cached per weight pointer.
- **Activations** are quantized per row to int8 on the GPU with an **asymmetric**
  zero-point: `scale = (rmax − rmin)/255`, a nudged zero-point, plus a per-row
  `row_sum_act`. Asymmetric was *necessary* — symmetric `amax/127` rounded
  skewed post-SwiGLU outliers to 0 and flipped token logits over a deep stack.
  A parallel-reduction quantizer (64 work-items/row, LDS min/max+sum tree)
  replaced the 1-WI-per-row scalar pass that dominated ~71% of Adreno prefill.
- **Accumulation** is integer: each K-block mask-unpacks 8 nibble lanes and
  accumulates int32 via `dot_4x8packed_su_int` (signed act × unsigned encoded
  weight), the portable `cl_khr_integer_dot_product` builtin (`dp4a`).
- **Epilogue** is one shared FP16 formula:
  `corrected = acc − 8·row_sum_act − zp_act·row_sum_w; v = (half)(corrected ·
  scale_act · scale_wgt)`. The `−8` term removes the weight offset; the `zp` term
  removes the activation asymmetry. Because the int32 accumulation is
  order-independent and every variant shares this epilogue, **image / buffer /
  dp4a / XMX outputs are bit-identical** — which is what makes greedy
  token-ID equality the verification gate.

### 6.2 Shape-driven dispatch: prefill GEMM / decode GEMV / cooperative GEMV

One host entry (`dotCl_v8c`) selects the kernel from the shape:

- **`M>4` prefill GEMM** — a tiled GEMM (`TM=4`, `TN=8`: 4 activation rows × 8
  output channels per work-item), ~87% of Adreno dp4a peak. M is padded up to a
  multiple of 64 so the tuned 4×16 LWS fits every FC shape.
- **`M≤4` decode GEMV (`m1`)** — collapses the tile to row 0 only, avoiding the 3
  padded-row reads/writes the `TM=4` kernel would burn (~4× cut in decode GEMM
  cost).
- **64-wide K-split cooperative GEMV** — the real decode lever. A plain `m1`
  GEMV exposes only `N/8` work-items (latency-bound at 12–22 GB/s while decode FC
  streams ~1 MB/token). `v8c_gemv_int8_int4_coop` runs a 64-work-item group per
  8-column tile with an 8-way K-split (LDS tree reduce), staging the activation
  row once in LDS — parallelism scales to `N×8` so the weight loads stream at
  memory rate. Bit-identical to `m1`. It reads plain buffers and serves *both*
  the Intel buffer path and the Adreno image path (which extracts the image's
  backing `cl_mem` via `clGetImageInfo`).

### 6.3 lm_head GEMV + on-GPU argmax

The decode lm_head is a **Q6_K GEMV** (`q6_k_sgemv.cl`) — the vocab projection
keeps Q6_K precision for argmax fidelity while staying on-GPU. (An untied int4
vocab larger than the device image-height cap, e.g. 262144, falls back to an
imageless int4 buffer GEMV, and layer planning forces its output height to 1 to
avoid ~1 GB of resident-but-dead vocab-wide activation.) Greedy sampling then
runs an on-GPU **2-pass argmax** that reduces the device logits to one token id
and reads back **4 bytes**, not the full vocab — the precondition for a
single-submission decode step. Ties break to the lower index to match
`std::max_element`, and the kernel is plain OpenCL (no subgroup builtins) so it
is byte-identical on Adreno and Intel.

### 6.4 Attention — GPU MHA, flash-decode, softcap, sliding window

The per-token attention runs through one `MHACoreLayer`
(`Applications/CausalLM/layers/mha_core.cpp`) that picks an attention backend at
dispatch time and shares kernels in `attention_kernels.cpp`:

- **GPU MHA** (`NNTR_MHA_GPU`): Q·Kᵀ / softmax / ·V on-device, FP16-Q / FP16-out,
  GQA-aware. Adreno uses per-layer `image2d` KV mirrors (texture cache); Intel
  uses an SVM-buffer **flash** path (register-tiled, no materialized scores
  tensor). `head_dim > 128` (e.g. Gemma2 `d=256`) uses the d-tiling `blockq`/`vec`
  variants.
- **Flash-decoding (split-KV)** for the `M=1` decode query: split the KV axis
  into chunks (`chunk_kv=64`), launch `num_heads × n_chunks` partial-softmax
  groups + a reduce pass — recovering parallelism for the lone query (Gemma4
  long-context decode **+68%**).
- **Logit softcap & sliding window**: `attn_logit_softcapping` (Gemma2/Gemma4)
  and `SlidingWindow→local_window_size` are threaded into the prefill kernels;
  required for correct Gemma2 (QK soft-cap) and Gemma4 (sliding-window layers)
  numerics on the GPU.

### 6.5 RoPE (LUT-cap fix) and q/k/v-norm residency

- **GPU RoPE + LUT-cap fix.** RoPE is applied in-place on device from a flat
  cos/sin FP16 LUT uploaded once. The fix: cap the LUT to the model's actual
  max timestep (live KV length, e.g. 1024) instead of `max_position_embeddings`
  (131072) — `θ_j` is position-independent so a shorter LUT is exact. This cut
  the per-layer re-upload from tens of MB to hundreds of KB and is what made
  `M≥32` prefill GPU-RoPE coherent (≈ +500 TPS @ M=1024).
- **q/k/v-norm GPU residency** is *structural, not a flag*. The per-head
  `reshaped_rms_norm` layers (qwen3 q/k-norm, Gemma4 gamma-free v_norm/PLE-norm)
  are created with `engine=gpu`, so their output stays GPU-resident instead of
  bouncing to host (qwen3 was slow purely because q/k-norm was on the host). The
  gamma-free case uses an FP32 sum-of-squares (overflow-safe).
- **Decode-step gates** (`GpuDecodeRope` / `GpuDecodeAttn`, default-on for
  gemma2/gemma4) move RoPE and attention onto the GPU at `M=1` too, so the
  blocking `lower_q`/`lower_kv` host drains (~65 ms/token over 35 layers) become
  no-ops. Env `NNTR_MHA_GPU_DECODE` is the global override.

### 6.6 GeGLU — the first backend-neutral collapsed layer

GeGLU (`gelu_tanh(gate)·up`) is the template for the add-only refactor (§12). The
former `GeGLULayerCl` (OpenCL) and `CudaGeGLULayer` (CUDA) forks are collapsed
into **one** backend-neutral `GeGLULayer` (`nntrainer/layers/geglu_layer.h`) that
owns structure/orchestration and dispatches the kernel via
`in1.getOps()->geglu(...)` — a whole-op `ComputeOps` virtual. It lands on
`ClComputeOps::geglu` (`geglu_cl_op`, cl_mem/SVM residency), `CudaComputeOps::geglu`
(device FP16 kernel under `NNTR_CUDA_GEGLU`, else host-on-UVM), or
`CpuComputeOps::geglu` (host loop). Both `ClContext` and `CudaContext` register
this same neutral layer; the forks are deleted. Verified token-identical on
Adreno + Intel + CUDA.

### 6.7 Residency & coherence — the two pools

Residency is layered: the `MemAllocator` decides the memory *kind*, and a static
`ResidencyClass` (`HOST` / `SVM` / `GPU_CLMEM`), derived once at
`TensorPool::allocate()` from the producer engine + all consumers + dtype,
decides how layers *bind* it. Two opt-in pools turn the host round-trips off:

- **`NNTR_GPU_SVM_POOL`** switches the OpenCL command queue to **in-order** with
  SVM-resident buffers. This in-order property is what orders a producer kernel's
  write *before* the consuming FC's device-direct read (out-of-order would race
  to garbage). It also lets per-op SVM map/unmap (and the `clFinish` drain they
  imply) be skipped — consecutive GPU kernels are already device-coherent.
- **`NNTR_GPU_CLMEM_POOL`** allocates the planned activation plane as plain device
  `cl_mem` (`ClBufferPool`) — one handle per distinct planner offset (binding the
  *same* handle for tensors reused at one offset, required for Adreno per-handle
  cache coherence). It stamps `GPU_CLMEM` residency so an FC consumes its
  producer's exact device buffer with no SVM map. Oversized host-resident
  quantized planes (e.g. a Gemma4 Q6_K embedding > the device alloc cap) stay on
  SVM; the pool **hard-fails** rather than silently degrading to a corrupting
  hybrid.

Both are mandatory together for coherence. Carve-outs keep correctness while
maximizing residency: the KV cache stays SVM; input-boundary RAISE and
output-boundary LOWER let a host-produced/consumed tensor still be `GPU_CLMEM`
because the producing/consuming layer explicitly uploads/reads it.

## 7. Per-device implementations

The one-source split: device differences are runtime knobs / caps, not forks.

### 7.1 Adreno (Qualcomm) — the image2d texture path

Both quantized weights and int8 activations are wrapped as **zero-copy
`image2d`-from-buffer** views (`CL_RGBA` / `CL_UNSIGNED_INT32`, 16-byte texels)
and read with integer-coordinate `read_imageui`, hitting the Adreno **texture-L1
cache**. This is the Adreno default (no `-D` option); the same bytes are
byte-identical to indexing the `cl_mem` as `uint4[]`, which is what gives Intel
its buffer path.

- **image2d KV attention** (`NNTR_KV_IMG_ATTN`): because the layer-graph KV cache
  is SVM (an image can't wrap an SVM pointer), per-layer `cl_mem` KV **mirrors**
  are filled by scatter kernels (K as OHWI `[hKV,S,d]`, V as reversed-OHWI
  `[hKV,d,S]`) and read through `read_imageui`. The **tight-stride V image** sizes
  its row pitch to the *live* sequence (not `S_max`) to avoid wasting texture
  cache on padding (sv_matmul 63 ms → 41 ms at M=843). This is Adreno-only —
  `read_imageui` won't compile on Intel NEO.
- **Structural prefill ceiling ≈ 2430–2454 TPS.** In-kernel
  `cl_khr_kernel_clock` instrumentation (`V8C_KCLOCK`) decomposes the GEMM K-loop
  and shows it is **fetch/unpack-bound** on the texture path (compute hidden
  under fetch), at ~87% of peak. LDS staging / weight prefetch / M-fast dispatch
  were all measured **negative** on Adreno — documented dead ends.
- **ARM build delicacy.** `ndk-build` does not run meson's `.cl→.cpp` codegen, so
  a stale embedded kernel string silently produces garbage; `build_lib.sh`
  regenerates any `.cl` newer than its `.cpp` first. Six artifacts must be
  co-located on device (§9), `libccapi-nntrainer.so` being the most-forgotten.

### 7.2 Intel (Xe / Meteor Lake) — DPAS, NEO buffer path, Xe3 sync

Intel selects one of two GEMM families by capability and always uses the buffer
(not image) path.

- **7.2.1 XMX/DPAS prefill (Xe2/Xe3).** `gemm_xmx_i4`
  (`int8_int8_gemm_xmx.cl`) is a drop-in for the `M>4` GEMM: it SWAR-unpacks the
  v8c int4 nibbles and feeds the **systolic `int8` matrix-MAD**
  (`intel_sub_group_i8_u8_matrix_mad_k32`) via subgroup 2D block reads, with the
  *same* v8c packing and byte-identical epilogue (~30 TOP/s, ~1.7–1.9× over dp4a).
  Prefill-only — decode (`M=1`) is a bandwidth-bound GEMV that a
  compute-throughput engine can't speed up, so the `M>4` gate keeps it out.
  **Now caps-derived**: enabled by default when the device advertises
  `cl_intel_subgroups` (`caps().subgroups`), retiring `NNTR_FC_XMX` as an opt-in;
  a device without the matrix-MAD fails kernel registration and falls through to
  dp4a (so it is safe on non-XMX Intel too).
- **7.2.2 non-XMX (Meteor Lake / older NEO) buffer path.** Intel NEO's SPIR-V
  backend **cannot compile** integer-coordinate `read_imageui` (it fails the
  whole program build), so the image kernel bodies are `#ifndef V8C_BUFFER_ONLY`
  and Intel compiles `-DV8C_BUFFER_ONLY -cl-std=CL3.0` (CL3.0 is needed to expose
  the `dot_4x8packed_*` builtins NEO doesn't declare under CL1.2). The
  buffer-load sibling kernels read the identical v8c bytes as `uint4[]`. This is
  the `NNTR_V8C_BUF` switch — **mandatory on every Intel device**, XMX or not (the
  XMX path's `buf_kernel` prerequisite rides the same flag). Decode uses the same
  64-wide cooperative GEMV (§6.2).
- **7.2.3 Xe3 (Panther Lake) coherence.** Xe3 (NEO 26.22) does **not** honor
  in-order kernel→kernel memory consistency for the coarse-grained-SVM hand-offs
  the residency model relies on (Meteor Lake's fine-grained SVM did — this is a
  new-ISA regression). **`NNTR_XE3_SYNC` is mandatory** on Xe3: it inserts a
  `clFinish` at the producer→consumer boundary. Without it the consumer reads
  stale output and the model emits garbage. It deliberately has no caps probe
  (wrong ⇒ garbage), so it stays an explicit override.

### 7.3 NVIDIA CUDA — RTX (discrete) and Orin (integrated)

`CudaContext` is a full peer of `ClContext`: registered `"cuda"`, NVRTC runtime
compile + PTX disk cache, `CudaMemAllocator` = `cudaMallocManaged` (UVM) + a
`device_only` `cudaMalloc` activation pool. Un-accelerated ops run the
`CpuComputeOps` body on the host-coherent managed pointer; the heavy layers bind
NVRTC kernels.

**Shared CUDA techniques:**
- **QINT4 fused dequant-GEMM FC** (`cuda_fc_qint4`): decodes KAI Section-A int4
  on-device and runs w4a8 `__dp4a` with three shapes — `dp4a_gemv` (M=1 decode),
  `dp4a_gemm_reg` (M≥8, 64×64 register-blocked), `dp4a_gemm` (small M). Default
  ON (the host QINT4 dot is ARM/KAI-only).
- **cuBLAS int8 IMMA prefill FC** (`NNTR_FC_CUDA_CUBLAS`): for M≥32, route w4a8 to
  cuBLAS int8×int8→int32 on the **Tensor Cores** (~10× dp4a). The int4 weight is
  unpacked to int8 once and cached; the int32 result is bit-identical so the same
  dequant epilogue applies.
- **Block-Q warp-shuffle prefill attention** (`NNTR_CUDA_BLOCKQ`): one warp owns a
  row tile, the d-dot is a single `__shfl_xor` warp reduce (no shared mem). Three
  instantiations cover head_dim 256 / 512 / **128** — the d128 instance is what
  lets qwen3/llama run fast on RTX without GEMM-attention.
- **Flash-decode (split-KV)** (`NNTR_CUDA_FLASH_DECODE=64`) for `M=1`, mirroring
  the OpenCL design.
- **CUDA-graph capture/replay** for decode: a whole decode forward (~350
  dispatches / ~1000 kernels) is captured into one `cudaGraph` and replayed to
  collapse CPU launch overhead. **M2-B** captures *once*, then per token only
  refreshes the embeddings on host and updates a **device position buffer**
  (`cuda_set_pos`) that the kernels read — so one frozen graph stays correct
  across tokens (RoPE position, KV slot, `N_kv` all read device-side).
- **Repack-at-load + prewarm** (`NNTR_CUDA_PREWARM`): ThreadManager-parallel CPU
  code mirrors the device repack (Section-A → int4 / int8 + rowsum) and pre-grows
  all decode/prefill scratch to max capacity, keeping one-time repacks off the
  timed path and avoiding capture-invalidating `cudaMalloc`s.
- **Full-GPU-residency decode** + on-GPU argmax: RoPE, geglu/swiglu/add/scalar,
  rmsnorm, q/k/v-norm, and a Q6_K lm_head GEMV are all NVRTC kernels, so the
  decode chain drains once per token.

**Discrete vs integrated — one truth source.**
`cuda::ContextManager::isIntegrated()` (1 on Tegra/Orin, 0 on RTX) gates *every*
discrete-VRAM assumption so one binary stays correct and fast on both:

| | RTX (discrete) | Orin (integrated, sm_87) |
|---|---|---|
| activation pool | `device_only` `cudaMalloc` | forced managed (UVM, shared physical pool) |
| cuBLAS int8 | single full-K call | **K-chunked at 2048** (sm_87 large-K algo bug) |
| KV cache | device-mirror | pass-through (shared pool; avoids stale-KV snapshot) |
| attention default | block-Q | **GEMM attention** (block-Q is ~0.2 TFLOP/s on sm_87) |
| sync | honors `NNTR_CUDA_ASYNC` | forced sync (no UVM page-fault ordering) |
| prefill graph | off (not sync-bound) | **on** (the per-op sync floor dominates) |

`NNTR_CUDA_GEMM_ATTN` is now caps-derived (auto-on when `isIntegrated()`), and on
Orin the host-coherent **safe-set** (`run_gemma4_fast.sh`) keeps decode ops on
GPU without the discrete-VRAM tricks. A host-resident FC input/weight is staged
into device buffers rather than dereferenced on Orin (no `i8mm` → would `SIGILL`).

---

# Part III — Reference

## 8. Environment variable reference (selected)

The canonical run sets are in §2; this adds the common tuning / diagnostic knobs.
The full list is in the source (`grep -rn std::getenv`).

| Var | Effect | Platform |
|-----|--------|----------|
| `NNTR_ENGINE` | `=cpu` host layers, `=cuda` CUDA; unset ⇒ OpenCL GPU. | all |
| `NNTR_FC_INT8_GPU` | Master gate for the v8c int4/int8 quantized FC GEMM. | OpenCL |
| `NNTR_V8C_BUF` | Buffer-path v8c (cl_mem uint4) vs image2d — the Adreno⇄Intel switch; mandatory on Intel NEO. | Intel |
| `NNTR_KV_IMG_ATTN` | image2d KV mirrors + image KV attention (texture cache). | Adreno |
| `NNTR_MHA_GPU` / `NNTR_MHA_GPU_DECODE` | GPU attention; the `_DECODE` form extends it (and GPU-RoPE + flash-decode) to the `M=1` step. | OpenCL |
| `NNTR_GPU_SVM_POOL` | In-order SVM-resident queue; skips per-layer `clFinish`. | OpenCL |
| `NNTR_GPU_CLMEM_POOL` | Device `cl_mem` activation pool (FC consumes producer's output directly). Mandatory for coherence. | OpenCL |
| `NNTR_XE3_SYNC` | `clFinish` at producer→consumer; **mandatory on Xe3** (coherence regression). | Intel-Xe3 |
| `NNTR_FC_XMX` | Override the caps-derived XMX default (force on/off). | Intel-XMX |
| `NNTR_GEMV_COOP` | 64-wide K-split cooperative decode GEMV (default on). | OpenCL |
| `NNTR_ROPE_LUT_CAP` / `NNTR_NO_GPU_ROPE` | Override / disable the GPU-RoPE LUT cap. | OpenCL |
| `NNTR_VNORM_HOST` | Kill-switch: q/k/v-norm back to the host. | OpenCL |
| `NNTR_KV_INT8` | int8 (quantized) KV cache instead of FP16. | OpenCL |
| `NNTR_CUDA_DEV_ACT` | Device-only activation pool (discrete); ignored on integrated. | CUDA |
| `NNTR_CUDA_ATTN` / `NNTR_CUDA_ROPE` / `NNTR_CUDA_QKNORM` / `NNTR_CUDA_GEGLU` / `NNTR_CUDA_ELTWISE` | Move the matching decode op onto the GPU. | CUDA |
| `NNTR_CUDA_BLOCKQ` / `NNTR_CUDA_FLASH_DECODE` / `NNTR_FC_CUDA_CUBLAS` | block-Q attention / split-KV decode / cuBLAS IMMA prefill FC. | CUDA |
| `NNTR_CUDA_GEMM_ATTN` | Override the caps-derived GEMM-attention default (auto-on for Orin). | CUDA |
| `NNTR_CUDA_GRAPH` / `NNTR_CUDA_M2B` | CUDA-graph decode capture / single-capture replay. | CUDA |
| `NNTR_CUDA_PREWARM` / `NNTR_CUDA_KV_UVM` / `NNTR_CUDA_VCOPY_PREFILL` | Load-time repack+scratch prewarm / KV residency / V-copy into the live KV slot. | CUDA |
| `NNTR_OPENCL_PROFILING` / `NNTR_LAYER_PROFILE` / `NNTR_V8C_KCLOCK` | clprof / per-layer latency / in-kernel clock profiling. | diag |
| `NNTR_KV_WINDOW_RING` | **Long context, opt-in (default off).** `=1` stores a sliding-window layer's KV cache as a ring of `Wcap` physical rows instead of the full context window, and turns chunked prefill on. The request is granted only where a ring-aware attention arm resolves (`NNTR_KV_OHWI=1` + `NNTR_MHA_GPU=1` on OpenCL, `NNTR_CUDA_ATTN=1` on `NNTR_ENGINE=cuda`, and neither `NNTR_KV_IMG_ATTN` nor `NNTR_MHA_GPU_IMG`); otherwise the linear full-height cache is kept and the reason is printed once. | OpenCL, CUDA |
| `NNTR_PREFILL_CHUNK` | Query rows per prefill forward (`0`/unset ⇒ one block, unless the ring is on, which requests 4096). Clamped to `init_seq_len`, the activation-plane height. A non-positive or unparseable value is rejected with a message. | all |
| `NNTR_CUDA_SPLITKV_PREFILL` | **Opt-in (default off).** Inside `NNTR_CUDA_BLOCKQ`, splits the key axis of a full-attention prefill (`=1` ⇒ 4096-key split, `=N>1` ⇒ custom, `=0` ⇒ off). Engages only above the split length, so shorter contexts are bit-unchanged. | CUDA |
| `NNTR_CUDA_SPLITKV_PREFILL_MB` | Scratch budget in MiB for the split-KV partial buffers. | CUDA |

## 9. Build artifacts & deploy (Adreno)

A working device run needs **all six** co-located in
`/data/local/tmp/nntrainer/causallm` (with `LD_LIBRARY_PATH` pointing there):

1. `nntrainer_causallm` — the executable (`chmod 755` on device)
2. `libcausallm_core.so` — CausalLM model/layer core
3. **`libccapi-nntrainer.so`** — the nntrainer C++/CC-API; **holds the Tensor-API
   graph-compile (KV-placeholder dtype) logic**. ⚠️ **The #1 forgotten artifact**
   — a stale copy aborts with `cache placeholder dtype mismatch` at layer 0.
4. `libnntrainer.so` — nntrainer core + OpenCL GPU symbols (needs `enable-opencl=true`)
5. `libOpenCL.so` — the Adreno OpenCL ICD (from `builddir/opencl/lib/arm64-v8a`)
6. `libc++_shared.so` — NDK C++ runtime

ndk-build links into `obj/local/arm64-v8a/`; the `libs/arm64-v8a/` copy can lag —
prefer pushing from `obj/local/arm64-v8a/` and check timestamps. When the
`ComputeOps` vtable is unchanged in existing slots (e.g. virtuals appended at the
tail), a new `libnntrainer.so` is ABI-compatible with an older app/ccapi, so only
that one `.so` needs pushing.

## 10. `nntr_config.json` keys

| Key | Meaning |
|-----|---------|
| `model_type` | **Must be `"CausalLM"`** or the runtime throws a model_type mismatch and aborts. |
| `model_tensor_type` | `WEIGHT-ACTIVATION` pair, e.g. `"QINT4-FP16"`. The activation half sets compute precision; **FP16 is required for full GPU residency**. |
| `fc_layer_dtype` | Weight dtype for Q/K/V/O + FFN FC (e.g. `QINT4`, `Q4_0`). The v8c/CUDA GPU GEMM expects QINT4/Q4_0. |
| `embedding_dtype` | Token-embedding weight dtype (e.g. `Q6_K`). Default for `lmhead_dtype`. |
| `lmhead_dtype` | LM-head weight dtype (e.g. `Q6_K` — the GPU GEMV decode path). Falls back to `embedding_dtype`. |
| `lmhead_untie` | Untied LM head vs sharing the embedding matrix. Default false. |
| `skip_prefill` | **Gemma4-only** KV-shared fast path. NOT transferable to Qwen3/Gemma2 (garbage if enabled). Default false. |
| `tokenizer_file` | Absolute path to `tokenizer.json`. ⚠️ A device-pulled config has a `/data/local/tmp/...` path — edit it for x86 runs. |
| `init_seq_len` / `max_seq_len` | Prefill window M / KV-cache time dimension. |
| `num_to_generate` | Decode tokens to generate after prefill. |
| `model_file_name` | The nntrainer weight `.bin` inside the model dir; must match the configured quantization. |

## 11. Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `allocateAndBindKVCache: cache placeholder dtype mismatch` (abort at layer 0) | **Stale `libccapi-nntrainer.so` on the device.** The KV-placeholder dtype is decided by the Tensor-API graph compile in libccapi; an old copy gives `kp=FP32 ≠ kc=FP16`. Push the fresh libccapi (verify with `md5sum`). |
| Output collapses to a single repeated token | Missing `NNTR_GPU_CLMEM_POOL=1` (mandatory for coherence on both OpenCL backends). |
| Garbage on **Xe3** specifically | Missing `NNTR_XE3_SYNC=1` (Panther Lake SVM coherence regression). |
| `model_type mismatch` crash at load | `nntr_config.json` lacks `"model_type":"CausalLM"`. |
| `Failed to open file` (tokenizer) | `tokenizer_file` points at a device path; set the local absolute path. |
| Silent garbage after editing a `.cl` kernel (Android) | ndk-build does not re-run meson's `.cl`→`.cpp` codegen. Regenerate (`.claude/regen_cl.py` / `build_lib.sh`) before rebuilding. |
| `dlopen`/undefined-symbol for `clSVM*` on Android | `libnntrainer.so` was built without OpenCL. Reconfigure `builddir` with `-Denable-opencl=true` and `ninja install`. |
| Orin: `SIGILL` / host-pointer fault in an FC | A host-resident input/weight reached a device kernel; ensure the safe-set (`run_gemma4_fast.sh`) so inputs are staged to device buffers. |

## 12. The multi-HW refactor (add-only architecture)

The knobs above and the residual `#if` leakage are being folded into a
principled **add-only** model (`nntrainer/docs/ARCHITECTURE_REFACTOR.md`):
express backend differences solely as op-table virtuals + `Context`
capability/sync + `MemAllocator` capability predicates, so a new device becomes
"register a `Context`, report its caps, provide an op-table subset" with **zero**
edits to models or core. Status:

- **Phase 0 — landed.**
  - *T1 DeviceCaps probe.* A read-only `DeviceCaps` (`Context::caps()`) snapshot,
    probed once per backend from real device queries (vendor, arch, integrated,
    unified_memory, `subgroups`=XMX, compute_units, max_alloc) — describes
    attributes, never identity.
  - *T2 MemAllocator predicates.* `isHostAddressable` / `isDeviceVisible` /
    `isSVM` / `needsRegister` / `supportsDevicePool` (+ `makePool`) replace the
    `getName()=="gpu-svm"` string hacks; byte-identical.
  - *T3 registry open.* `parseComputeEngine` validates against the live
    registered-context set (not a closed enum), and a `registerLayerFactory`
    facade lets a vendor self-register a context (e.g. `"npu"`) without
    downcasting to a concrete `*Context`.
- **Phase 1 — ExecPlan resolver, shadow → authoritative.** `resolveExecPlan(caps)`
  is a pure function (`cuda→CUBLAS`, `gpu→subgroups?XMX:DP4A`, else CPU). Landed as
  a logged shadow (T4), then **two cells flipped authoritative (T8)**: the v8c FC
  **XMX gemm_path now defaults to `caps().subgroups`** (retiring `NNTR_FC_XMX` as
  an opt-in) and **CUDA GEMM-attention now defaults to `isIntegrated()`**
  (retiring `NNTR_CUDA_GEMM_ATTN`). Both keep the env var as an explicit override.
  This is the first concrete payoff of the probe→shadow→authoritative arc.
- **Phase 2 — collapsing the layer forks.**
  - *T6.* `CudaComputeOps : public CpuComputeOps` (extracted to a header so it is
    inheritable); `CudaContext` binds `get_cuda_ops()`.
  - *T7.* The **first layer-fork collapse**: GeGLU is now one backend-neutral
    `GeGLULayer` dispatching `in1.getOps()->geglu(...)` (§6.6) — the
    `GeGLULayerCl` + `CudaGeGLULayer` forks are deleted, token-identical on all 3
    HW. This is the template for the remaining `*_cl` / `*_cuda` layers (FC,
    attention, rmsnorm, …).

**Env-knob status.** *Retired to caps-default* (env now an override): `NNTR_FC_XMX`,
`NNTR_CUDA_GEMM_ATTN`. *Never an env flag* (structural): q/k/v-norm residency
(`engine=` property). *No-op alias*: `NNTR_FC_GPU` (real gate `NNTR_FC_INT8_GPU`).
*Still required* (not a pure function of caps; wrong ⇒ garbage, deliberately not
resolved): `NNTR_V8C_BUF` & `NNTR_KV_IMG_ATTN` (the NEO `read_imageui`-compile
quirk, both devices advertise image2d), `NNTR_XE3_SYNC` (new-ISA coherence), plus
`NNTR_FC_INT8_GPU` / `NNTR_MHA_GPU` / `NNTR_GPU_SVM_POOL` / `NNTR_GPU_CLMEM_POOL`
as the canonical run set.

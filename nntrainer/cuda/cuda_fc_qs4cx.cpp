// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_fc_qs4cx.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Fused QS4CX dequant-GEMM implementation (NVRTC kernel).
 */

#include "cuda_fc_qs4cx.h"

#include <cuda_blas_manager.h>
#include <cuda_common.h> // cuda_vec4_rows_ok
#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#if defined(_WIN32)
#include <windows.h> // DiscardVirtualMemory
#else
#include <sys/mman.h> // madvise
#endif
#include <cstdlib>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fp16.h>
#include <thread_manager.h>

namespace nntrainer::cuda {

// The QS4CX plain payload (row-major [N][(K+1)/2] nibbles,
// stored uint4 = int4+8, even k = low nibble) is consumed by the CUDA FC
// paths DIRECTLY, the way the OpenCL v8c kernel consumes it: the derived
// device-only caches (dp4a packed-int4 / cuBLAS int8) are built straight from
// it and keyed by its pointer, so no host/UVM Section-A copy of the nibble
// payload exists anymore (it used to double every FC weight's host RSS —
// CUDA 2x vs OpenCL 1x). The only per-weight side allocation left is this
// N-entry fp16 scale buffer: the dequant kernels read the per-channel scale
// on device every call, while the tensor stores fp32 scales. UVM (host+device
// readable, host-readable matters for the _resident staging path), built once
// at load, cached by the fp32-scale pointer with no erase (weights live for
// the process lifetime), never under a graph capture.
bool cuda_fc_qs4cx_scales_to_uvm_fp16(const float *fp32_scales, unsigned int N,
                                      const unsigned short **out_sc) {
  static std::map<const void *, unsigned short *> cache;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = cache.find(fp32_scales);
  if (it == cache.end()) {
    // cudaMallocManaged inside a CUDA-graph capture invalidates the capture;
    // the load-time prewarm builds this before any capture, so a miss here
    // under capture only happens on an un-prewarmed weight -- bail so the
    // caller falls back instead of corrupting the graph.
    if (StreamManager::Global().isCapturing())
      return false;
    unsigned short *usc = nullptr;
    // [WDDM coherence] This buffer is host-WRITTEN once and device-READ every
    // FC call -- exactly the pattern that is incoherent on cMA==0 managed
    // memory (see cuda_mem_allocator use_host_mapped). Use pinned host-mapped
    // (zero-copy, UVA same-pointer) there; managed elsewhere.
    static const bool host_mapped = []() {
      const char *e = std::getenv("NNTR_CUDA_HOST_MAPPED");
      if (e != nullptr)
        return e[0] == '1';
      return !ContextManager::Global().concurrentManagedAccess();
    }();
    if (host_mapped) {
      if (cudaHostAlloc(&usc, sizeof(unsigned short) * (size_t)N,
                        cudaHostAllocMapped) != cudaSuccess)
        return false;
    } else if (cudaMallocManaged(&usc, sizeof(unsigned short) * (size_t)N) !=
               cudaSuccess)
      return false;
    for (unsigned int n = 0; n < N; ++n)
      usc[n] = compute_fp32_to_fp16(fp32_scales[n]);
    it = cache.emplace(fp32_scales, usc).first;
  }
  *out_sc = it->second;
  return true;
}

// [lm_head fp-act] The same side buffer in fp32. The dp4a family reads the
// fp16 copy above -- a per-channel scale rounded to 11 significant bits, which
// is far below its own int8-activation noise and therefore free there. The
// fp-activation lm_head GEMV exists precisely to take the last argmax noise
// out of the logits, so it keeps the tensor's fp32 scale instead (the OpenCL
// lm_head GEMV reads fp32 for the same reason) -- one extra MiB for the one
// weight that asks for it.
//
// @p source_readable is the caller's promise that @p fp32_scales may still be
// dereferenced. It may NOT be once the caller has released the payload: the
// fp32 scale tail lives inside the same allocation as the nibbles, so
// releasing one releases the other and it reads back as zeros -- which would
// silently produce all-zero logits rather than fail. On a
// cache HIT nothing is read and the flag is irrelevant; on a miss with the
// flag false the build is refused and the caller falls back.
bool cuda_fc_qs4cx_scales_to_uvm_fp32(const float *fp32_scales, unsigned int N,
                                      const float **out_sc,
                                      bool source_readable) {
  static std::map<const void *, float *> cache;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = cache.find(fp32_scales);
  if (it == cache.end()) {
    if (!source_readable || fp32_scales == nullptr || N == 0)
      return false;
    if (StreamManager::Global().isCapturing())
      return false;
    float *usc = nullptr;
    // Same host-written-once / device-read-every-call pattern as the fp16
    // buffer, so the same WDDM coherence choice applies.
    static const bool host_mapped = []() {
      const char *e = std::getenv("NNTR_CUDA_HOST_MAPPED");
      if (e != nullptr)
        return e[0] == '1';
      return !ContextManager::Global().concurrentManagedAccess();
    }();
    if (host_mapped) {
      if (cudaHostAlloc(&usc, sizeof(float) * (size_t)N, cudaHostAllocMapped) !=
          cudaSuccess)
        return false;
    } else if (cudaMallocManaged(&usc, sizeof(float) * (size_t)N) !=
               cudaSuccess)
      return false;
    std::memcpy(usc, fp32_scales, sizeof(float) * (size_t)N);
    it = cache.emplace(fp32_scales, usc).first;
  }
  *out_sc = it->second;
  return true;
}

// Per-op cudaStreamSynchronize is ~90% of inference wall time (nsys): each GPU
// op drains the stream, fully serializing CPU and GPU. This drain is a sync
// point hook for the future selective-sync work (sync only before a HOST
// consumer reads a UVM output, not after every FC).
//
// NNTR_CUDA_ASYNC=1 drops the drains -- EXPERIMENTAL/UNSAFE: it makes decode
// ~40% faster but produces GARBAGE, because the host ops between FCs (RoPE,
// attention, geglu) then read UVM the GPU is still writing -- the
// concurrentManagedAccess page-fault does NOT order a host read against an
// in-flight kernel write. The coherent path to that speedup is to move those
// decode host ops onto the GPU too (GPU RoPE/geglu, the GPU attention exists)
// so the whole decode step is one ordered GPU chain drained once per token.
// Default (sync) is coherent.
static inline void maybe_finish(const void *out = nullptr) {
  // Device-only (cudaMalloc) destination: host code CANNOT read it directly
  // (it would AV) -- every legal host access goes through a stream-ordered
  // staging copy (copy_any / EnqueueReadBuffer / explicit finish), so the
  // per-op drain is provably unnecessary. Skipping it removes the WDDM
  // submit+wait round-trip (measured at 0.2-0.6 ms per op in a 1K prefill)
  // while ops with host-visible outputs keep their sync-mode drain.
  static const bool skip_dev_drain = []() {
    const char *e = std::getenv("NNTR_CUDA_DRAINSKIP_FC");
    return e != nullptr && e[0] == '1';
  }();
  if (skip_dev_drain && out != nullptr && dev_only(out))
    return;
  static const bool async = []() {
    const char *e = std::getenv("NNTR_CUDA_ASYNC");
    if (e == nullptr || e[0] != '1')
      return false;
    // Integrated GPU (Tegra/Orin): force sync -- async is non-coherent on the
    // shared-memory iGPU (see cuda_stream_manager cuda_async_mode()).
    return !ContextManager::Global().isIntegrated();
  }();
  if (!async)
    StreamManager::Global().finish();
}

static const char *FC_QS4CX_PLAIN_SRC = R"CU(
extern "C" {

__device__ __forceinline__ float plain_h2f(unsigned short h) {
  unsigned int sign = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int exp = (h >> 10) & 0x1Fu;
  unsigned int mant = h & 0x3FFu;
  unsigned int out;
  if (exp == 0u) {
    if (mant == 0u) {
      out = sign;
    } else {
      int e = -1;
      do { mant <<= 1; e++; } while ((mant & 0x400u) == 0u);
      mant &= 0x3FFu;
      out = sign | ((unsigned int)(127 - 15 - e) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    out = sign | 0x7F800000u | (mant << 13);
  } else {
    out = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
  }
  return __int_as_float((int)out);
}

__global__ void fc_qs4cx_plain_gemm(const float *X, const unsigned char *W,
                                    const unsigned short *sc, float *Y, int M,
                                    int N, int K, int Kh) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  const unsigned char *wrow = W + (long)n * Kh;
  const float *xr = X + (long)m * K;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    unsigned char b = wrow[k >> 1];
    int nib = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
    acc += xr[k] * (float)(nib - 8);
  }
  Y[(long)m * N + n] = acc * plain_h2f(sc[n]);
}

}
)CU";

bool cuda_fc_qs4cx_gemm_fp32(const float *X, const unsigned char *plain_w,
                             const unsigned short *scales_fp16, float *Y,
                             unsigned int M, unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;

  auto kernel = CudaContext::Global().registerCudaKernel(FC_QS4CX_PLAIN_SRC,
                                                         "fc_qs4cx_plain_gemm");
  if (!kernel) {
    ml_loge("[CUDA] fc_qs4cx_plain: kernel registration failed");
    return false;
  }

  int m = (int)M, n = (int)N, k = (int)K;
  int kh = (int)((K + 1u) / 2u);
  kernel->SetKernelArguments(0, &X, sizeof(X));
  kernel->SetKernelArguments(1, &plain_w, sizeof(plain_w));
  kernel->SetKernelArguments(2, &scales_fp16, sizeof(scales_fp16));
  kernel->SetKernelArguments(3, &Y, sizeof(Y));
  kernel->SetKernelArguments(4, &m, sizeof(m));
  kernel->SetKernelArguments(5, &n, sizeof(n));
  kernel->SetKernelArguments(6, &k, sizeof(k));
  kernel->SetKernelArguments(7, &kh, sizeof(kh));

  const int block[3] = {16, 16, 1};
  const int grid[3] = {((int)N + 15) / 16, ((int)M + 15) / 16, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  maybe_finish(Y);
  return true;
}

namespace {
// Device mirror of a host-resident QS4CX weight + reusable activation/output
// staging buffers. Weights are constant for the model lifetime, so the plain
// nibble payload + fp16 scales are uploaded once and cached by host pointer.
struct DevWeight {
  unsigned char *d_w = nullptr;
  unsigned short *d_sc = nullptr;
};
std::unordered_map<const void *, DevWeight> g_qs4cx_weight_cache;
float *g_stage_x = nullptr;
size_t g_stage_x_cap = 0;
float *g_stage_y = nullptr;
size_t g_stage_y_cap = 0;
std::mutex g_qs4cx_mtx;

bool ensure_stage(float **buf, size_t *cap, size_t bytes) {
  if (bytes <= *cap)
    return true;
  // cudaMalloc/cudaFree inside a CUDA-graph stream capture invalidates the
  // capture. The fp32-resident staging buffers are pre-grown at load by
  // cuda_fc_qs4cx_dp4a_prewarm() so this branch must not run under capture; if
  // it ever would (an under-sized prewarm), bail so the caller falls back
  // rather than corrupting the graph.
  if (StreamManager::Global().isCapturing())
    return false;
  if (*buf)
    cudaFree(*buf);
  if (cudaMalloc(buf, bytes) != cudaSuccess) {
    *buf = nullptr;
    *cap = 0;
    return false;
  }
  *cap = bytes;
  return true;
}
} // namespace

bool cuda_fc_qs4cx_gemm_fp32_resident(const float *host_X,
                                      const unsigned char *host_plain,
                                      const unsigned short *host_scales,
                                      float *host_Y, unsigned int M,
                                      unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  std::lock_guard<std::mutex> lk(g_qs4cx_mtx);

  // 1) device weight (upload once, cache by host pointer).
  auto it = g_qs4cx_weight_cache.find(host_plain);
  if (it == g_qs4cx_weight_cache.end()) {
    const size_t w_bytes = (size_t)N * ((K + 1u) / 2u);
    DevWeight dw;
    if (cudaMalloc(&dw.d_w, w_bytes) != cudaSuccess)
      return false;
    if (cudaMalloc(&dw.d_sc, sizeof(unsigned short) * (size_t)N) !=
        cudaSuccess) {
      cudaFree(dw.d_w);
      return false;
    }
    cudaMemcpy(dw.d_w, host_plain, w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dw.d_sc, host_scales, sizeof(unsigned short) * (size_t)N,
               cudaMemcpyHostToDevice);
    it = g_qs4cx_weight_cache.emplace(host_plain, dw).first;
  }

  // 2) stage activation in, output buffer out (grown as needed).
  const size_t xb = sizeof(float) * (size_t)M * K;
  const size_t yb = sizeof(float) * (size_t)M * N;
  if (!ensure_stage(&g_stage_x, &g_stage_x_cap, xb) ||
      !ensure_stage(&g_stage_y, &g_stage_y_cap, yb))
    return false;
  cudaMemcpy(g_stage_x, host_X, xb, cudaMemcpyHostToDevice);

  // 3) device GEMM (synchronizes the backend stream internally).
  if (!cuda_fc_qs4cx_gemm_fp32(g_stage_x, it->second.d_w, it->second.d_sc,
                               g_stage_y, M, N, K))
    return false;

  // 4) output back to the host tensor.
  StreamManager::Global().finishIfAsync();
  cudaMemcpy(host_Y, g_stage_y, yb, cudaMemcpyDeviceToHost);
  return true;
}

// ===========================================================================
// w4a8 dp4a fast path
// ===========================================================================
// Three NVRTC kernels (one module): per-row int8 activation quant, a one-time
// QS4CX-plain -> signed packed int4 repack (a byte-wise XOR — same indexing,
// only the nibble encoding differs), and a __dp4a int8xint4 GEMM. Compiled for
// the device arch (compute_89 on Ada), so __dp4a lowers to the dp4a PTX
// instruction.
static const char *FC_QS4CX_DP4A_SRC =
  R"CU(
extern "C" {

__device__ __forceinline__ float dp4a_h2f(unsigned short h) {
  unsigned int sign = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int exp = (h >> 10) & 0x1Fu;
  unsigned int mant = h & 0x3FFu;
  unsigned int out;
  if (exp == 0u) {
    if (mant == 0u) {
      out = sign;
    } else {
      int e = -1;
      do { mant <<= 1; e++; } while ((mant & 0x400u) == 0u);
      mant &= 0x3FFu;
      out = sign | ((unsigned int)(127 - 15 - e) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    out = sign | 0x7F800000u | (mant << 13);
  } else {
    out = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
  }
  return __int_as_float((int)out);
}

// float -> fp16 (IEEE half), round to nearest even.
__device__ __forceinline__ unsigned short dp4a_f2h(float f) {
  unsigned int x = (unsigned int)__float_as_int(f);
  unsigned int sign = (x >> 16) & 0x8000u;
  int e = (int)((x >> 23) & 0xFFu);
  unsigned int mant = x & 0x7FFFFFu;
  if (e == 0xFF)
    return (unsigned short)(sign | 0x7C00u | (mant ? 0x200u : 0u)); // inf/nan
  int exp = e - 127 + 15;
  if (exp >= 0x1F)
    return (unsigned short)(sign | 0x7C00u); // overflow -> inf
  if (exp <= 0) {
    if (exp < -10)
      return (unsigned short)sign; // underflow -> 0
    mant |= 0x800000u;
    int shift = 14 - exp;
    unsigned int h = mant >> shift;
    unsigned int rem = mant & ((1u << shift) - 1u);
    unsigned int half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1u)))
      h++;
    return (unsigned short)(sign | h);
  }
  unsigned int h = ((unsigned int)exp << 10) | (mant >> 13);
  unsigned int rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u)))
    h++;
  return (unsigned short)(sign | h);
}

// asymmetric int8 quant params for a row's [min,max] (range forced to include
// 0, nudged zero-point) -- mirrors the OpenCL v8c act-quant. Returns recip
// (dequant scale) and zp; sets scale_q (quant multiplier) by reference.
__device__ __forceinline__ void asym_qparams(float fmn, float fmx,
                                             float &scale_q, float &recip,
                                             int &zp) {
  float rmin = fminf(0.f, fmn), rmax = fmaxf(0.f, fmx);
  float range = rmax - rmin;
  scale_q = range > 0.f ? 255.f / range : 1.f;
  recip = range > 0.f ? range / 255.f : 1.f;
  float dmin = rmin * scale_q, dmax = rmax * scale_q;
  float zp_lo = -128.f - dmin, zp_hi = 127.f - dmax;
  float zp_f = ((-128.f + dmin) + (127.f + dmax) > 0.f) ? zp_lo : zp_hi;
  zp_f = fmaxf(-128.f, fminf(127.f, zp_f));
  zp = (int)rintf(zp_f);
}

// per-row asymmetric int8 quant of an fp16 activation (one block per row).
// stores recip in ascale[m], zero-point in azp[m].
__global__ void act_quant_i8_h(const unsigned short *Xh, signed char *q8,
                               float *ascale, int *azp, int M, int K) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  __shared__ float smn[256];
  __shared__ float smx[256];
  const unsigned short *xr = Xh + (long)m * K;
  float lmn = 0.f, lmx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = dp4a_h2f(xr[k]);
    lmn = fminf(lmn, v);
    lmx = fmaxf(lmx, v);
  }
  smn[threadIdx.x] = lmn;
  smx[threadIdx.x] = lmx;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x + s]);
      smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x + s]);
    }
    __syncthreads();
  }
  float scale_q, recip;
  int zp;
  asym_qparams(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) {
    ascale[m] = recip;
    azp[m] = zp;
  }
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = (int)rintf(dp4a_h2f(xr[k]) * scale_q) + zp;
    q = max(-128, min(127, q));
    q8[(long)m * K + k] = (signed char)q;
  }
}

// Hardware half<->float conversion, reachable from NVRTC without cuda_fp16.h.
//
// The scalar software routines above are ~20 integer ops each; the hardware
// instruction is one. On the DECODE row shapes (one block, a few thousand
// elements) that difference is the whole kernel: measured on RTX 5060, the
// row-at-a-time norm/quant kernels spend far more time converting than moving
// their few KB. Verified bit-identical to dp4a_h2f / dp4a_f2h over all 65536
// half patterns and 4M random floats, so the vectorized kernels below can use
// them without changing any value.
__device__ __forceinline__ float vq_h2f(unsigned short h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short vq_f2h(float f) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
// gamma is a weight at an arbitrary 2-byte offset in the model blob, so it is
// routinely NOT vector-aligned even when the activation rows are. Reading it
// with four scalar loads keeps the activation traffic vectorized instead of
// dropping the whole row to the scalar kernel. (has_gamma: 0 none, 1 vector,
// 2 scalar.)
__device__ __forceinline__ float4 vq_gather4(const unsigned short *g) {
  return make_float4(vq_h2f(g[0]), vq_h2f(g[1]), vq_h2f(g[2]), vq_h2f(g[3]));
}
__device__ __forceinline__ float4 vq_load4(uint2 r) {
  return make_float4(vq_h2f((unsigned short)(r.x & 0xFFFFu)),
                     vq_h2f((unsigned short)(r.x >> 16)),
                     vq_h2f((unsigned short)(r.y & 0xFFFFu)),
                     vq_h2f((unsigned short)(r.y >> 16)));
}
// Warp-shuffle reduce + one shared round over the warp results. blockDim.x
// must be a multiple of 32 and at most 1024. IDENT pads the lanes past the
// warp count in the final round, so it must be OP's identity.
#define VQ_REDUCE(scratch, val, OP, IDENT)                                     \
  do {                                                                         \
    for (int _o = 16; _o > 0; _o >>= 1)                                        \
      val = OP(val, __shfl_down_sync(0xffffffffu, val, _o));                   \
    if ((threadIdx.x & 31) == 0)                                               \
      scratch[threadIdx.x >> 5] = val;                                         \
    __syncthreads();                                                           \
    if (threadIdx.x < 32) {                                                    \
      float _a =                                                               \
        (threadIdx.x < (blockDim.x >> 5)) ? scratch[threadIdx.x] : (IDENT);    \
      for (int _o = 16; _o > 0; _o >>= 1)                                      \
        _a = OP(_a, __shfl_down_sync(0xffffffffu, _a, _o));                    \
      if (threadIdx.x == 0)                                                    \
        scratch[0] = _a;                                                       \
    }                                                                          \
    __syncthreads();                                                           \
  } while (0)
__device__ __forceinline__ float vq_add(float a, float b) { return a + b; }
#define VQ_POSINF __int_as_float(0x7F800000)
#define VQ_NEGINF __int_as_float((int)0xFF800000)

// Per-thread carry of the decoded row: with 4 halves per slot and 512 threads
// this covers K up to 16384 without a second global read; wider rows fall back
// to re-reading (still correct, just one more pass over an L1-hot row).
#define VQ_NCARRY 8

// Vectorized per-row asymmetric int8 activation quant. BIT-IDENTICAL to
// act_quant_i8_h: min/max are order-independent, the conversions are the same
// values, and the rint/clamp is unchanged.
__global__ void act_quant_i8_h_v4(const unsigned short *Xh, signed char *q8,
                                  float *ascale, int *azp, int M, int K) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  const uint2 *xv = (const uint2 *)(Xh + (long)m * K);
  int *q32 = (int *)(q8 + (long)m * K);
  const int nv = K >> 2;
  __shared__ float smn[32];
  __shared__ float smx[32];
  float lmn = 0.f, lmx = 0.f;
  float4 carry[VQ_NCARRY];
  int nc = 0;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 f = vq_load4(xv[i]);
    if (nc < VQ_NCARRY)
      carry[nc++] = f;
    lmn = fminf(lmn, fminf(fminf(f.x, f.y), fminf(f.z, f.w)));
    lmx = fmaxf(lmx, fmaxf(fmaxf(f.x, f.y), fmaxf(f.z, f.w)));
  }
  VQ_REDUCE(smn, lmn, fminf, VQ_POSINF);
  VQ_REDUCE(smx, lmx, fmaxf, VQ_NEGINF);
  float scale_q, recip;
  int zp;
  asym_qparams(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) {
    ascale[m] = recip;
    azp[m] = zp;
  }
  nc = 0;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 f = (nc < VQ_NCARRY) ? carry[nc++] : vq_load4(xv[i]);
    int q0 = max(-128, min(127, (int)rintf(f.x * scale_q) + zp));
    int q1 = max(-128, min(127, (int)rintf(f.y * scale_q) + zp));
    int q2 = max(-128, min(127, (int)rintf(f.z * scale_q) + zp));
    int q3 = max(-128, min(127, (int)rintf(f.w * scale_q) + zp));
    q32[i] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) |
             ((q3 & 0xFF) << 24);
  }
}

// Vectorized RMSNorm + int8 quant of the normed row (see rmsnorm_quant_i8_h
// below for the fusion rationale). The sum of squares is reduced in a
// different ORDER than the scalar kernels (vector-of-4 per thread, then warp
// shuffles), so `inv` can differ by an ulp -- the one place this lever is not
// bit-identical. Everything downstream of `inv` is.
__global__ void rmsnorm_quant_i8_h_v4(const unsigned short *x,
                                      const unsigned short *gamma,
                                      unsigned short *y, signed char *q8,
                                      float *ascale, int *azp, int M, int K,
                                      float eps, int has_gamma) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  const uint2 *xv = (const uint2 *)(x + (long)m * K);
  const uint2 *gv = (const uint2 *)gamma;
  uint2 *yv = (uint2 *)(y + (long)m * K);
  int *q32 = (int *)(q8 + (long)m * K);
  const int nv = K >> 2;
  __shared__ float ssq[32];
  __shared__ float smn[32];
  __shared__ float smx[32];
  float4 carry[VQ_NCARRY];
  int nc = 0;
  float p = 0.f;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 f = vq_load4(xv[i]);
    if (nc < VQ_NCARRY)
      carry[nc++] = f;
    p += f.x * f.x + f.y * f.y + f.z * f.z + f.w * f.w;
  }
  VQ_REDUCE(ssq, p, vq_add, 0.f);
  const float inv = rsqrtf(ssq[0] / (float)K + eps);

  float lmn = 0.f, lmx = 0.f;
  nc = 0;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    const int slot = (nc < VQ_NCARRY) ? nc++ : -1;
    float4 f = (slot >= 0) ? carry[slot] : vq_load4(xv[i]);
    float4 g = make_float4(1.f, 1.f, 1.f, 1.f);
    if (has_gamma == 1)
      g = vq_load4(gv[i]);
    else if (has_gamma == 2)
      g = vq_gather4(gamma + 4 * i);
    unsigned short h0 = vq_f2h(f.x * inv * g.x), h1 = vq_f2h(f.y * inv * g.y);
    unsigned short h2 = vq_f2h(f.z * inv * g.z), h3 = vq_f2h(f.w * inv * g.w);
    uint2 o;
    o.x = (unsigned int)h0 | ((unsigned int)h1 << 16);
    o.y = (unsigned int)h2 | ((unsigned int)h3 << 16);
    yv[i] = o;
    // Quantize the ROUNDED output, exactly what a following act_quant would
    // read back. The carry slot is recycled here: its input value is already
    // consumed for this element.
    float4 r = make_float4(vq_h2f(h0), vq_h2f(h1), vq_h2f(h2), vq_h2f(h3));
    if (slot >= 0)
      carry[slot] = r;
    lmn = fminf(lmn, fminf(fminf(r.x, r.y), fminf(r.z, r.w)));
    lmx = fmaxf(lmx, fmaxf(fmaxf(r.x, r.y), fmaxf(r.z, r.w)));
  }
  VQ_REDUCE(smn, lmn, fminf, VQ_POSINF);
  VQ_REDUCE(smx, lmx, fmaxf, VQ_NEGINF);
  float scale_q, recip;
  int zp;
  asym_qparams(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) {
    ascale[m] = recip;
    azp[m] = zp;
  }
  nc = 0;
  for (int i = threadIdx.x; i < nv; i += blockDim.x) {
    float4 r = (nc < VQ_NCARRY) ? carry[nc++] : vq_load4(yv[i]);
    int q0 = max(-128, min(127, (int)rintf(r.x * scale_q) + zp));
    int q1 = max(-128, min(127, (int)rintf(r.y * scale_q) + zp));
    int q2 = max(-128, min(127, (int)rintf(r.z * scale_q) + zp));
    int q3 = max(-128, min(127, (int)rintf(r.w * scale_q) + zp));
    q32[i] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) |
             ((q3 & 0xFF) << 24);
  }
}

// RMSNorm fused with the int8 activation quant its consumer FC needs.
//
// The decode norm and the quant that follows it are two single-block kernels
// over the same 1..8K-element row: at decode M=1 each is far below the launch
// granularity of the GPU, so the pair costs about twice its own arithmetic.
// Folding them removes one node per (norm -> FC-group) pair from the decode
// graph.
//
// Deliberately BIT-IDENTICAL to rmsnorm_fp16 followed by act_quant_i8_h:
//   - phase 1 reduces the sum of squares with the SAME per-thread stride and
//     the SAME shared-memory pairing tree, so the fp32 accumulation order (and
//     therefore `inv`) is unchanged;
//   - phase 2 writes exactly rmsnorm_fp16's y, and tracks min/max of the
//     ROUNDED fp16 it just stored -- the very values act_quant_i8_h would read
//     back -- so the quant params come out of asym_qparams unchanged;
//   - phase 3 re-reads those stores (each thread reads only its own) and
//     applies the identical rint/clamp.
// The equality is what lets the fused path be the default with a plain
// killswitch: no golden movement to argue about.
__global__ void rmsnorm_quant_i8_h(const unsigned short *x,
                                   const unsigned short *gamma,
                                   unsigned short *y, signed char *q8,
                                   float *ascale, int *azp, int M, int K,
                                   float eps, int has_gamma) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  const unsigned short *xr = x + (long)m * K;
  unsigned short *yr = y + (long)m * K;
  __shared__ float sdata[256];
  __shared__ float smx[256];
  float partial = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = dp4a_h2f(xr[k]);
    partial += v * v;
  }
  sdata[threadIdx.x] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(sdata[0] / (float)K + eps);
  __syncthreads(); // sdata[0] consumed; the arrays are reused below

  float lmn = 0.f, lmx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float g = has_gamma ? dp4a_h2f(gamma[k]) : 1.0f;
    unsigned short h = dp4a_f2h(dp4a_h2f(xr[k]) * inv * g);
    yr[k] = h;
    float v = dp4a_h2f(h);
    lmn = fminf(lmn, v);
    lmx = fmaxf(lmx, v);
  }
  sdata[threadIdx.x] = lmn;
  smx[threadIdx.x] = lmx;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] = fminf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
      smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x + s]);
    }
    __syncthreads();
  }
  float scale_q, recip;
  int zp;
  asym_qparams(sdata[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) {
    ascale[m] = recip;
    azp[m] = zp;
  }
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = (int)rintf(dp4a_h2f(yr[k]) * scale_q) + zp;
    q = max(-128, min(127, q));
    q8[(long)m * K + k] = (signed char)q;
  }
}

// per-output-channel weight row-sum (sum of signed int4) for the activation
// zero-point correction: Y -= recip[m]*scale_w[n]*zp[m]*rowsum_w[n].
__global__ void weight_rowsum(const signed char *plain, int *rowsum, int N,
                              int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N)
    return;
  int Kh = (K + 1) >> 1;
  const signed char *wrow = plain + (long)n * Kh;
  int s = 0;
  for (int kb = 0; kb < Kh; ++kb) {
    int b = (unsigned char)wrow[kb];
    int k0 = 2 * kb, k1 = 2 * kb + 1;
    if (k0 < K)
      s += ((int)(signed char)(b << 4)) >> 4;
    if (k1 < K)
      s += ((int)(signed char)b) >> 4;
  }
  rowsum[n] = s;
}

// float buffer -> fp16 buffer.
__global__ void cvt_f2h(const float *src, unsigned short *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = dp4a_f2h(src[i]);
}

// fp16 buffer -> float buffer.
__global__ void cvt_h2f(const unsigned short *src, float *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = dp4a_h2f(src[i]);
}

// signed int4 weight for (output n, input k) from the QS4CX plain payload
// (row-major [N][Kh] bytes, even k = low nibble, stored uint4 = int4+8).
__device__ __forceinline__ int plain_decode(const unsigned char *qw, int n,
                                            int k, int Kh) {
  unsigned char b = qw[(long)n * Kh + (k >> 1)];
  int nib = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
  return nib - 8;
}

// per-row asymmetric int8 quant of the activation (one block per row).
__global__ void act_quant_i8(const float *X, signed char *q8, float *ascale,
                             int *azp, int M, int K) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  __shared__ float smn[256];
  __shared__ float smx[256];
  const float *xr = X + (long)m * K;
  float lmn = 0.f, lmx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = xr[k];
    lmn = fminf(lmn, v);
    lmx = fmaxf(lmx, v);
  }
  smn[threadIdx.x] = lmn;
  smx[threadIdx.x] = lmx;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x + s]);
      smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x + s]);
    }
    __syncthreads();
  }
  float scale_q, recip;
  int zp;
  asym_qparams(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) {
    ascale[m] = recip;
    azp[m] = zp;
  }
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = (int)rintf(xr[k] * scale_q) + zp;
    q = max(-128, min(127, q));
    q8[(long)m * K + k] = (signed char)q;
  }
}

// QS4CX plain -> signed packed int4 [N, ceil(K/2)]: byte[n][kb] low nibble =
// int4(n, 2kb), high nibble = int4(n, 2kb+1), each stored two's-complement.
// The source has the SAME [N][Kh] byte indexing with uint4 = int4+8 nibbles,
// and (x-8)&0xF == x^8 on a 4-bit lane, so the whole repack is one byte-wise
// XOR with 0x88 (an odd-K pad nibble 8 becomes signed 0, as before).
__global__ void repack_plain_i4(const unsigned char *qw, signed char *packed,
                                int N, int Kh) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (long long)N * Kh)
    packed[i] = (signed char)(qw[i] ^ 0x88);
}

)CU"
  // NOTE: split here into two adjacent raw-string literals — MSVC caps a single
  // string literal at 16380 bytes (C2026); the two concatenate
  // byte-identically.
  R"CU(
// Y[m,n] = recip[m]*w_scale[n]*(sum_k q8[m,k]*int4(n,k) - zp[m]*rowsum_w[n]),
// the asymmetric-activation dequant (zp from act_quant, rowsum_w from the
// weight). via __dp4a.
__global__ void dp4a_gemm(const signed char *q8, const signed char *plain,
                          const float *ascale, const int *azp,
                          const int *wrowsum, const unsigned short *wscale,
                          float *Y, int M, int N, int K, int out_fp16) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  int Kh = (K + 1) >> 1;
  const signed char *qrow = q8 + (long)m * K;
  const signed char *wrow = plain + (long)n * Kh;
  int acc = 0, k = 0;
  for (; k + 4 <= K; k += 4) {
    int a = *(const int *)(qrow + k); // lanes = act k,k+1,k+2,k+3
    int kb = k >> 1;
    int b0 = (unsigned char)wrow[kb];     // k(low), k+1(high)
    int b1 = (unsigned char)wrow[kb + 1]; // k+2(low), k+3(high)
    int w0 = ((int)(signed char)(b0 << 4)) >> 4;
    int w1 = ((int)(signed char)b0) >> 4;
    int w2 = ((int)(signed char)(b1 << 4)) >> 4;
    int w3 = ((int)(signed char)b1) >> 4;
    int w = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) |
            ((w3 & 0xFF) << 24);
    acc = __dp4a(a, w, acc);
  }
  for (; k < K; ++k) { // tail (none when K%32==0)
    int kb = k >> 1;
    int b = (unsigned char)wrow[kb];
    int wv = (k & 1) ? (((int)(signed char)b) >> 4)
                     : (((int)(signed char)(b << 4)) >> 4);
    acc += (int)qrow[k] * wv;
  }
  float r = (float)(acc - azp[m] * wrowsum[n]) * ascale[m] * dp4a_h2f(wscale[n]);
  if (out_fp16)
    ((unsigned short *)Y)[(long)m * N + n] = dp4a_f2h(r);
  else
    Y[(long)m * N + n] = r;
}

// Dedicated M=1 decode GEMV: one block per output n, threads split K and
// block-reduce. The tiled dp4a_gemm wastes 15/16 rows of its 16x16 block at M=1
// (94% idle) and reads weight rows with a stride; here every thread is active
// and reads a contiguous K-slice of one weight row (coalesced). Activation row
// is row 0 (q8). out_fp16 folds the fp16 conversion in.
__global__ void dp4a_gemv(const signed char *q8, const signed char *plain,
                          const float *ascale, const int *azp,
                          const int *wrowsum, const unsigned short *wscale,
                          float *Y, int N, int K, int out_fp16) {
  // One WARP per output n (warps_per_block outputs per block) -> N/warps_per_block
  // blocks instead of N, amortizing the per-block launch/epilogue overhead that
  // dominated the old one-block-per-tiny-output design. No shared memory, no
  // __syncthreads: each lane reads a coalesced K-slice of the weight row and the
  // warp-shuffle reduces. dp4a int32 accumulate is integer-associative so the
  // result is BIT-IDENTICAL to the block-reduce version. (llama.cpp MMVQ shape.)
  const int warps_per_block = blockDim.x >> 5;
  int n = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  if (n >= N)
    return;
  const int lane = threadIdx.x & 31;
  int Kh = (K + 1) >> 1;
  const signed char *wrow = plain + (long)n * Kh;
  int acc = 0;
  // K4 = the input channels covered by whole groups of 4; the dp4a loop can
  // only consume those. k is a multiple of 4 so kb is even, but the 2-byte
  // weight load also needs the ROW base aligned, and wrow = plain + n*Kh is
  // 2-byte aligned only when Kh is even. Odd Kh (K % 4 == 1 or 2) leaves every
  // odd n on an odd address, which aborts the launch with "misaligned address"
  // rather than just computing wrong -- so those shapes read the byte pair
  // directly. The predicate is launch-uniform and loop-invariant: no warp
  // divergence, and K % 4 == 0 (every LLM projection width) keeps the wide load.
  const int K4 = K & ~3;
  const bool wide_w = ((Kh & 1) == 0);
  for (int k = lane * 4; k < K4; k += 32 * 4) {
    int a = *(const int *)(q8 + k);
    int kb = k >> 1;
    int b0, b1;
    if (wide_w) {
      unsigned int w16 = *(const unsigned short *)(wrow + kb);
      b0 = w16 & 0xFF;
      b1 = (w16 >> 8) & 0xFF;
    } else {
      b0 = (unsigned char)wrow[kb];
      b1 = (unsigned char)wrow[kb + 1];
    }
    int w0 = ((int)(signed char)(b0 << 4)) >> 4;
    int w1 = ((int)(signed char)b0) >> 4;
    int w2 = ((int)(signed char)(b1 << 4)) >> 4;
    int w3 = ((int)(signed char)b1) >> 4;
    int w = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) |
            ((w3 & 0xFF) << 24);
    acc = __dp4a(a, w, acc);
  }
  // Scalar tail for K % 4 != 0: the loop above consumes whole groups of 4, so
  // without this the last 1..3 input channels are dropped from every output.
  // One channel per lane over lanes 0..(K-K4-1) -- each is added exactly once
  // by the warp reduction below. Same nibble decode as dp4a_gemm's own tail.
  {
    int k = K4 + lane;
    if (k < K) {
      int b = (unsigned char)wrow[k >> 1];
      int wv = (k & 1) ? (((int)(signed char)b) >> 4)
                       : (((int)(signed char)(b << 4)) >> 4);
      acc += (int)q8[k] * wv;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) {
    float r = (float)(acc - azp[0] * wrowsum[n]) * ascale[0] *
              dp4a_h2f(wscale[n]);
    if (out_fp16)
      ((unsigned short *)Y)[n] = dp4a_f2h(r);
    else
      Y[n] = r;
  }
}

// Register-blocked dp4a GEMM: a 64x64 output tile per block; each of the 256
// threads accumulates a 4x4 micro-tile in registers, so a K-chunk of 32 staged
// once into shared memory feeds 16 dp4a per thread before the next load -- much
// higher arithmetic intensity than the 1-output-per-thread tiled kernel.
#define RB_BM 64
#define RB_BN 64
#define RB_BK 32
#define RB_TM 4
#define RB_TN 4
__global__ void dp4a_gemm_reg(const signed char *q8, const signed char *plain,
                              const float *ascale, const int *azp,
                              const int *wrowsum, const unsigned short *wscale,
                              float *Y, int M, int N, int K, int out_fp16) {
  __shared__ signed char As[RB_BM][RB_BK];
  __shared__ signed char Ws[RB_BN][RB_BK];
  int tx = threadIdx.x, ty = threadIdx.y; // 0..15 each
  int tid = ty * 16 + tx;
  int blockM = blockIdx.y * RB_BM, blockN = blockIdx.x * RB_BN;
  int Kh = (K + 1) >> 1;
  int acc[RB_TM][RB_TN];
#pragma unroll
  for (int i = 0; i < RB_TM; i++)
#pragma unroll
    for (int j = 0; j < RB_TN; j++)
      acc[i][j] = 0;
  for (int k0 = 0; k0 < K; k0 += RB_BK) {
    for (int e = tid; e < RB_BM * RB_BK; e += 256) {
      int i = e / RB_BK, j = e % RB_BK;
      int mm = blockM + i, kk = k0 + j;
      As[i][j] = (mm < M && kk < K) ? q8[(long)mm * K + kk] : (signed char)0;
    }
    for (int e = tid; e < RB_BN * RB_BK; e += 256) {
      int i = e / RB_BK, j = e % RB_BK;
      int nn = blockN + i, kk = k0 + j;
      signed char wv = 0;
      if (nn < N && kk < K) {
        unsigned char b = (unsigned char)plain[(long)nn * Kh + (kk >> 1)];
        wv = (kk & 1) ? (((signed char)b) >> 4)
                      : (((signed char)(b << 4)) >> 4);
      }
      Ws[i][j] = wv;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < RB_BK; kk += 4) {
      int af[RB_TM], wf[RB_TN];
#pragma unroll
      for (int i = 0; i < RB_TM; i++)
        af[i] = *(const int *)&As[ty * RB_TM + i][kk];
#pragma unroll
      for (int j = 0; j < RB_TN; j++)
        wf[j] = *(const int *)&Ws[tx * RB_TN + j][kk];
#pragma unroll
      for (int i = 0; i < RB_TM; i++)
#pragma unroll
        for (int j = 0; j < RB_TN; j++)
          acc[i][j] = __dp4a(af[i], wf[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < RB_TM; i++) {
    int row = blockM + ty * RB_TM + i;
    if (row >= M)
      continue;
    float as = ascale[row];
    int zp = azp[row];
#pragma unroll
    for (int j = 0; j < RB_TN; j++) {
      int col = blockN + tx * RB_TN + j;
      if (col < N) {
        float r =
          (float)(acc[i][j] - zp * wrowsum[col]) * as * dp4a_h2f(wscale[col]);
        if (out_fp16)
          ((unsigned short *)Y)[(long)row * N + col] = dp4a_f2h(r);
        else
          Y[(long)row * N + col] = r;
      }
    }
  }
}

// === cuBLAS INT8 IMMA (Tensor Core) prefill FC support ===
// The __dp4a kernels run on the int ALU (ceiling ~21 TOPS on Ada). cuBLAS int8
// IMMA runs on the Tensor Cores (~30 TOPS measured, ~10x our dp4a GEMM). These
// three kernels feed it: unpack the int4 weight -> int8 ONCE (cached), and the
// int32 GEMM result is bit-identical to the __dp4a acc, so the SAME dequant
// applies in the epilogue.

// int4 plain weight -> int8 [K,N] (w8[k*N+n] = int4(n,k)). Unpacked once and
// cached (weights are static), so cuBLAS reads contiguous int8 -- doing this per
// call would add a memory pass that erases the Tensor-Core win.
__global__ void repack_plain_i8_kn(const unsigned char *qw, signed char *w8,
                                   int N, int K, int Kh) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (n >= N || k >= K)
    return;
  w8[(long)k * N + n] = (signed char)plain_decode(qw, n, k, Kh);
}

// per-output-channel sum of the int8 weight column (k-strided), for the
// activation zero-point correction. one thread per output channel n.
__global__ void weight_rowsum_kn(const signed char *w8, int *rowsum, int N,
                                 int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N)
    return;
  long s = 0;
  for (int k = 0; k < K; ++k)
    s += (int)w8[(long)k * N + n];
  rowsum[n] = (int)s;
}

// dequant epilogue for the int8 IMMA GEMM: C is the int32 dot-product (== the
// __dp4a acc, bit-identical). Y[m,n]=(C - zp[m]*rowsum[n])*recip[m]*wscale[n].
__global__ void dequant_i32_fp16(const int *C, const float *ascale,
                                 const int *azp, const int *wrowsum,
                                 const unsigned short *wscale, unsigned short *Y,
                                 int M, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  float r = (float)(C[(long)m * N + n] - azp[m] * wrowsum[n]) * ascale[m] *
            dp4a_h2f(wscale[n]);
  Y[(long)m * N + n] = dp4a_f2h(r);
}

)CU"
  // Third adjacent literal (same MSVC C2026 cap as the split above).
  R"CU(
// --- fp-ACTIVATION int4 GEMV (the huge-N untied lm_head decode) -----------
//
// Same weight bytes as the dp4a path -- the cached signed packed int4 [N][Kh]
// -- but the ACTIVATION is read as fp16 and multiplied in float: no per-row
// int8 activation quant. That quant maps the row onto a 255-level grid over
// its [min,max], and on this lm_head the resulting per-logit error measured
// sigma 0.18-0.37 against a top1-top2 argmax margin of ~0.117, i.e. it flips
// roughly one argmax in twelve, and the flips compound over a long decode.
// Measured against an fp64 dequant-dot at this shape: this kernel's residual
// error is the fp16 logit rounding alone (rms 4.4e-4), the dp4a route's is
// 1.6e-2 -- 37x more. The OpenCL lane routes this exact weight the same way
// and for the same reason -- lmhead_int4_v8c_gemv in blas_kernels.cpp, "best
// argmax fidelity; no int8 act quant".
//
// The extra cost is only on the activation side (K halves, L2-hot after the
// first block); the weight is ~201 MB per token at vocab 262144 either way, so
// this is bandwidth-bound and reads exactly as many weight bytes as dp4a does.
//
// One WARP per output row (the dp4a_gemv mapping): 8 input channels per lane
// per step = one 4-byte weight load (8 nibbles) + one 16-byte activation load,
// so a warp step consumes 256 channels and reads 128 CONTIGUOUS weight bytes.
// @p wide is the caller's alignment verdict; the scalar arm keeps odd K and
// unaligned activation rows correct.
__global__ void fpact_gemv_i4_h(const unsigned short *Xh,
                                const signed char *plain, const float *wscale,
                                unsigned short *Y, int N, int K, int wide) {
  const int warps_per_block = blockDim.x >> 5;
  const int n = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  if (n >= N)
    return;
  const int lane = threadIdx.x & 31;
  const int Kh = (K + 1) >> 1;
  const signed char *wrow = plain + (long)n * Kh;
  float acc = 0.f;
  if (wide) {
    for (int k = lane * 8; k < K; k += 32 * 8) {
      const unsigned int wb = *(const unsigned int *)(wrow + (k >> 1));
      const uint4 av = *(const uint4 *)(Xh + k);
      const unsigned int aw[4] = {av.x, av.y, av.z, av.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        // byte j holds channels k+2j (low nibble) and k+2j+1 (high nibble),
        // which is exactly the half PAIR packed in aw[j].
        const int b = (int)((wb >> (j * 8)) & 0xFFu);
        acc += vq_h2f((unsigned short)(aw[j] & 0xFFFFu)) *
               (float)(((int)(signed char)(b << 4)) >> 4);
        acc += vq_h2f((unsigned short)(aw[j] >> 16)) *
               (float)(((int)(signed char)b) >> 4);
      }
    }
  } else {
    for (int k = lane; k < K; k += 32) {
      const int b = (unsigned char)wrow[k >> 1];
      const int wv = (k & 1) ? (((int)(signed char)b) >> 4)
                             : (((int)(signed char)(b << 4)) >> 4);
      acc += vq_h2f(Xh[k]) * (float)wv;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0)
    Y[n] = vq_f2h(acc * wscale[n]);
}

}
)CU";

namespace {
// cached signed-packed-int4 repack of each QS4CX weight (keyed by the plain
// host/UVM payload pointer = weight.getData()).
struct DevWeightQ {
  signed char *plain = nullptr; // signed packed int4 [N, ceil(K/2)]
  int *rowsum = nullptr;        // per-channel sum of signed int4 [N]
};
std::unordered_map<const void *, DevWeightQ> g_dp4a_plain_cache;
// int8-unpacked weight [K,N] + per-channel rowsum, for the cuBLAS int8 path
// (keyed by the QS4CX plain payload pointer; unpacked once, weights are
// static).
struct DevWeightI8 {
  signed char *w8 = nullptr; // int8 weight [K,N] (w8[k*N+n] = int4(n,k))
  int *rowsum = nullptr;     // per-channel sum of int8 weight [N]
};
std::unordered_map<const void *, DevWeightI8> g_i8_weight_cache;
// Weights whose FC can never reach the M>=32 cuBLAS gate
// (skip_prefill layers never see prefill M>1; the untied lm_head decodes at
// M=1): their [K,N] int8 cache is 2x the int4 payload of pure dead VRAM
// (~1.5GB total, lm_head alone 673MiB). The app marks them before the
// prewarm walk (load time, single-threaded -- no lock needed); the EAGER
// build below skips them, while the lazy ensure_i8_cache_locked() runtime
// build stays as the self-healing fallback if the premise is ever wrong.
std::unordered_set<const void *> g_i8_exempt;
int *g_i8_c = nullptr; // int32 GEMM output scratch [M,N]
size_t g_i8_c_cap = 0;
signed char *g_dp4a_q8 = nullptr;
size_t g_dp4a_q8_cap = 0;
float *g_dp4a_ascale = nullptr; // per-row recip (dequant scale)
size_t g_dp4a_ascale_cap = 0;
int *g_dp4a_azp = nullptr; // per-row activation zero-point
size_t g_dp4a_azp_cap = 0;

// act_quant dedup (cuBLAS prefill path): sibling FCs that share an input
// activation (q/k/v <- attention_norm; gate/up <- ffn_norm) re-quantize the
// IDENTICAL fp16 rows into the shared g_dp4a_q8/ascale/azp. Since those buffers
// persist across the sibling's GEMM+dequant (neither writes them), the 2nd/3rd
// sibling can reuse the 1st's quantization. Model-graph tensors have stable
// distinct addresses, so keying on (Xh ptr, K) is safe within AND across
// forwards: a different FC has a different input tensor -> different ptr ->
// re-quantizes; only the immediate same-ptr siblings skip. Removes ~244 of 413
// act_quant launches. Decision is made at graph-record time, so the captured
// graph simply omits the redundant nodes (capture-safe). Disable:
// NNTR_QUANT_DEDUP=0.
//
// act-quant handoff: whoever last filled g_dp4a_q8 records WHAT it quantized
// (activation pointer + K) and the stream dispatch count at that moment. A
// consumer FC may reuse the staged quant only if both still match -- the
// pointer alone is forgeable by the activation pool (a recycled buffer reuses
// the address), the sequence number is not: any kernel dispatched in between
// bumps it and the FC re-quantizes. Written by the fused norm+quant producer,
// by the dp4a decode path, and (under NNTR_QUANT_DEDUP) by the cuBLAS prefill
// path.
const void *g_last_quant_xh = nullptr;
int g_last_quant_k = 0;
unsigned long long g_last_quant_seq = 0;
bool g_last_quant_valid = false;

// NNTR_CUDA_FUSED_NORMQ: fold the decode RMSNorm and the int8 activation quant
// of the FC group it feeds into one kernel, and let the sibling FCs of that
// group consume the staged quant instead of recomputing it. Bit-identical to
// the split path (see rmsnorm_quant_i8_h), so it is the DEFAULT; =0 restores
// the separate rmsnorm_fp16 + act_quant_i8_h launches.
bool fused_normq_on() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_CUDA_FUSED_NORMQ");
    return !(e != nullptr && e[0] == '0');
  }();
  return v;
}

// Publish the staged quant as reusable by the very next FC on @p xh.
void mark_quant_staged(const void *xh, int k) {
  g_last_quant_xh = xh;
  g_last_quant_k = k;
  g_last_quant_seq = StreamManager::Global().dispatchSeq();
  g_last_quant_valid = true;
}

// True when g_dp4a_q8 already holds the int8 quant of (xh, k).
bool quant_staged_for(const void *xh, int k) {
  return g_last_quant_valid && xh == g_last_quant_xh && k == g_last_quant_k &&
         StreamManager::Global().dispatchSeq() == g_last_quant_seq;
}
float *g_dp4a_yf = nullptr; // float Y staging for the fp16-output path
size_t g_dp4a_yf_cap = 0;
float *g_dp4a_xf = nullptr; // float X staging for the naive fp16 path
size_t g_dp4a_xf_cap = 0;
// fp16 X staging for a HOST-resident input on the device GPU qs4cx path: when
// the FC input pointer is host memory (e.g. a captured decode graph feeds the
// token via pinned host memory), the fp16 dp4a/cublas kernels still need a
// device X. The M*K fp16 input is copied H2D into this buffer and the device
// pointer is used instead of falling to the i8mm host dot (which SIGILLs on
// Orin).
unsigned short *g_stage_xh = nullptr;
size_t g_stage_xh_cap = 0;
std::mutex g_dp4a_mtx;

// Build (once) the dp4a signed-packed-int4 + rowsum device cache for plain_w
// by dispatching the repack kernels on the backend stream -- the GPU reads
// the plain payload directly, so it must be device-accessible. Caller holds
// g_dp4a_mtx. Returns the cache entry, or nullptr on failure.
DevWeightQ *ensure_dp4a_cache_locked(const unsigned char *plain_w,
                                     unsigned int N, unsigned int K) {
  auto it = g_dp4a_plain_cache.find(plain_w);
  if (it != g_dp4a_plain_cache.end())
    return &it->second;
  const int n = (int)N, k = (int)K;
  const size_t Kh = (K + 1u) / 2u;
  auto kr = CudaContext::Global().registerCudaKernel(FC_QS4CX_DP4A_SRC,
                                                     "repack_plain_i4");
  auto krs = CudaContext::Global().registerCudaKernel(FC_QS4CX_DP4A_SRC,
                                                      "weight_rowsum");
  if (!kr || !krs)
    return nullptr;
  DevWeightQ dw;
  if (cudaMalloc(&dw.plain, (size_t)N * Kh) != cudaSuccess)
    return nullptr;
  if (cudaMalloc(&dw.rowsum, sizeof(int) * (size_t)N) != cudaSuccess) {
    cudaFree(dw.plain);
    return nullptr;
  }
  const int khi = (int)Kh;
  kr->SetKernelArguments(0, &plain_w, sizeof(plain_w));
  kr->SetKernelArguments(1, &dw.plain, sizeof(dw.plain));
  kr->SetKernelArguments(2, &n, sizeof(n));
  kr->SetKernelArguments(3, &khi, sizeof(khi));
  const int rb[3] = {256, 1, 1};
  const int rg[3] = {(int)(((size_t)N * Kh + 255) / 256), 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kr, rg, rb)) {
    cudaFree(dw.plain);
    cudaFree(dw.rowsum);
    return nullptr;
  }
  // per-channel weight row-sum (for the activation zero-point correction).
  krs->SetKernelArguments(0, &dw.plain, sizeof(dw.plain));
  krs->SetKernelArguments(1, &dw.rowsum, sizeof(dw.rowsum));
  krs->SetKernelArguments(2, &n, sizeof(n));
  krs->SetKernelArguments(3, &k, sizeof(k));
  const int sb[3] = {128, 1, 1};
  const int sg[3] = {((int)N + 127) / 128, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*krs, sg, sb)) {
    cudaFree(dw.plain);
    cudaFree(dw.rowsum);
    return nullptr;
  }
  it = g_dp4a_plain_cache.emplace(plain_w, dw).first;
  return &it->second;
}

// repack (cached) + GEMM into a device float Y, using the already-staged
// q8/ascale scratch. Caller holds g_dp4a_mtx and has run act-quant.
bool dp4a_repack_and_gemm(const unsigned char *plain_w,
                          const unsigned short *scales_fp16, float *Yf,
                          unsigned int M, unsigned int N, unsigned int K,
                          int out_fp16 = 0) {
  const int n = (int)N, k = (int)K;
  const bool gemv = (M == 1);
  const bool tiled = (M >= 8);
  auto kg = CudaContext::Global().registerCudaKernel(
    FC_QS4CX_DP4A_SRC,
    gemv ? "dp4a_gemv" : (tiled ? "dp4a_gemm_reg" : "dp4a_gemm"));
  if (!kg)
    return false;

  DevWeightQ *dwp = ensure_dp4a_cache_locked(plain_w, N, K);
  if (!dwp)
    return false;
  signed char *plain = dwp->plain;
  int *wrowsum = dwp->rowsum;

  const int mm = (int)M;
  kg->SetKernelArguments(0, &g_dp4a_q8, sizeof(g_dp4a_q8));
  kg->SetKernelArguments(1, &plain, sizeof(plain));
  kg->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kg->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
  kg->SetKernelArguments(4, &wrowsum, sizeof(wrowsum));
  kg->SetKernelArguments(5, &scales_fp16, sizeof(scales_fp16));
  kg->SetKernelArguments(6, &Yf, sizeof(Yf));
  if (gemv) {
    // dp4a_gemv: one WARP per output, 4 warps (128 threads) per block ->
    // ceil(N/4) blocks instead of N (4x fewer per-block launch/epilogue
    // overheads).
    kg->SetKernelArguments(7, &n, sizeof(n));
    kg->SetKernelArguments(8, &k, sizeof(k));
    kg->SetKernelArguments(9, &out_fp16, sizeof(out_fp16));
    const int gvb[3] = {128, 1, 1};
    const int gvg[3] = {((int)N + 3) / 4, 1, 1};
    return StreamManager::Global().DispatchCommand(*kg, gvg, gvb);
  }
  kg->SetKernelArguments(7, &mm, sizeof(mm));
  kg->SetKernelArguments(8, &n, sizeof(n));
  kg->SetKernelArguments(9, &k, sizeof(k));
  kg->SetKernelArguments(10, &out_fp16, sizeof(out_fp16));
  const int gb[3] = {16, 16, 1};
  const int tile = tiled ? 64 : 16;
  const int gg[3] = {((int)N + tile - 1) / tile, ((int)M + tile - 1) / tile, 1};
  return StreamManager::Global().DispatchCommand(*kg, gg, gb);
}

bool ensure_buf(void **buf, size_t *cap, size_t bytes) {
  if (bytes <= *cap)
    return true;
  // cudaMalloc/cudaFree inside a CUDA-graph stream capture invalidates the
  // capture. The dp4a decode scratch is pre-grown at load by
  // cuda_fc_qs4cx_dp4a_prewarm() so this branch must not run under capture; if
  // it ever would (an under-sized prewarm), bail so the caller falls back
  // rather than corrupting the graph.
  if (StreamManager::Global().isCapturing())
    return false;
  if (*buf)
    cudaFree(*buf);
  if (cudaMalloc(buf, bytes) != cudaSuccess) {
    *buf = nullptr;
    *cap = 0;
    return false;
  }
  *cap = bytes;
  return true;
}
} // namespace

// stage q8 + ascale + azp scratch (caller holds the mutex). False on OOM.
// +256B tail pad on the int8 activation: the cuBLAS int8 IMMA GEMM reads A with
// wide vectorized (>=16B) Tensor-Core loads that can run past the last real
// element; an exactly-sized buffer (esp. large K=6144 down-proj) then faults
// with cudaErrorIllegalAddress. The pad keeps those reads in mapped memory.
static constexpr size_t FC_I8_TAIL_PAD = 256;

// Build (once) the cuBLAS int8 [K,N] + rowsum device cache for plain_w by
// dispatching the unpack kernels on the backend stream (the GPU reads the
// plain payload directly). Caller holds g_dp4a_mtx. Returns the cache entry,
// or nullptr on failure.
static DevWeightI8 *ensure_i8_cache_locked(const unsigned char *plain_w,
                                           unsigned int N, unsigned int K) {
  auto it = g_i8_weight_cache.find(plain_w);
  if (it != g_i8_weight_cache.end())
    return &it->second;
  const int n = (int)N, k = (int)K, kh = (int)((K + 1u) / 2u);
  auto krp = CudaContext::Global().registerCudaKernel(FC_QS4CX_DP4A_SRC,
                                                      "repack_plain_i8_kn");
  auto krs = CudaContext::Global().registerCudaKernel(FC_QS4CX_DP4A_SRC,
                                                      "weight_rowsum_kn");
  if (!krp || !krs)
    return nullptr;
  DevWeightI8 dw;
  if (cudaMalloc(&dw.w8, (size_t)N * K + FC_I8_TAIL_PAD) != cudaSuccess)
    return nullptr;
  if (cudaMalloc(&dw.rowsum, sizeof(int) * (size_t)N) != cudaSuccess) {
    cudaFree(dw.w8);
    return nullptr;
  }
  krp->SetKernelArguments(0, &plain_w, sizeof(plain_w));
  krp->SetKernelArguments(1, &dw.w8, sizeof(dw.w8));
  krp->SetKernelArguments(2, &n, sizeof(n));
  krp->SetKernelArguments(3, &k, sizeof(k));
  krp->SetKernelArguments(4, &kh, sizeof(kh));
  const int pb[3] = {16, 16, 1};
  const int pg[3] = {((int)N + 15) / 16, ((int)K + 15) / 16, 1};
  if (!StreamManager::Global().DispatchCommand(*krp, pg, pb)) {
    cudaFree(dw.w8);
    cudaFree(dw.rowsum);
    return nullptr;
  }
  krs->SetKernelArguments(0, &dw.w8, sizeof(dw.w8));
  krs->SetKernelArguments(1, &dw.rowsum, sizeof(dw.rowsum));
  krs->SetKernelArguments(2, &n, sizeof(n));
  krs->SetKernelArguments(3, &k, sizeof(k));
  const int sb[3] = {128, 1, 1};
  const int sg[3] = {((int)N + 127) / 128, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*krs, sg, sb)) {
    cudaFree(dw.w8);
    cudaFree(dw.rowsum);
    return nullptr;
  }
  it = g_i8_weight_cache.emplace(plain_w, dw).first;
  return &it->second;
}

static bool dp4a_stage_scratch(unsigned int M, unsigned int K) {
  return ensure_buf((void **)&g_dp4a_q8, &g_dp4a_q8_cap,
                    (size_t)M * K + FC_I8_TAIL_PAD) &&
         ensure_buf((void **)&g_dp4a_ascale, &g_dp4a_ascale_cap,
                    sizeof(float) * (size_t)M) &&
         ensure_buf((void **)&g_dp4a_azp, &g_dp4a_azp_cap,
                    sizeof(int) * (size_t)M);
}

// Pre-grow ALL the static dp4a decode scratch buffers to the model's max decode
// capacity at load. The M=1 dp4a decode FC path is reached under graph capture
// once NNTR_CUDA_GRAPH is on; a cudaMalloc/Free inside
// cudaStreamBeginCapture..EndCapture invalidates the capture and surfaces as
// "NvMapMemAllocInternalTagged failed: error 12". Warming here (before any
// capture) makes every captured ensure_buf a pure cap-hit, so the dp4a path
// stays usable under the graph. ensure_buf's isCapturing() guard is the safety
// net if a model exceeds these bounds. Idempotent (cap check). False on OOM.
//
// maxM   max decode token rows (1 for decode; larger is a harmless over-grow)
// maxK   max FC input dim  (hidden DIM; covers every decode FC's K)
// maxN   max FC output dim (max(vocab, intermediate); covers lm_head + FFN)
bool cuda_fc_qs4cx_dp4a_prewarm(unsigned int maxM, unsigned int maxK,
                                unsigned int maxN) {
  if (maxM == 0 || maxK == 0 || maxN == 0)
    return true;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // q8/ascale/azp staging: exact sizes dp4a_stage_scratch() computes.
  if (!dp4a_stage_scratch(maxM, maxK))
    return false;
  // float X/Y staging: exact sizes the fp16-naive / fp32-resident paths use
  // (g_dp4a_xf = M*K floats, g_dp4a_yf = M*N floats). yf is grown to maxM*maxN
  // so the largest decode FC (lm_head N = vocab) is covered.
  // g_stage_xh (M*K fp16) covers the host-resident-input staging on the fp16
  // GPU path, so that copy is also a pure cap-hit under capture.
  return ensure_buf((void **)&g_dp4a_xf, &g_dp4a_xf_cap,
                    sizeof(float) * (size_t)maxM * maxK) &&
         ensure_buf((void **)&g_dp4a_yf, &g_dp4a_yf_cap,
                    sizeof(float) * (size_t)maxM * maxN) &&
         ensure_buf((void **)&g_stage_xh, &g_stage_xh_cap,
                    sizeof(unsigned short) * (size_t)maxM * maxK);
}

// Stage a HOST-resident M*K fp16 activation into a device buffer for the fp16
// GPU qs4cx path. Copies host_Xh H2D (async, on the backend stream so it is
// ordered before the kernels that read it) into the reusable g_stage_xh buffer
// and returns the device pointer. Returns nullptr if the buffer can't be grown
// (OOM, or a capture before prewarm sized it) so the caller falls back to the
// host path. The copy is enqueued on the backend stream; the subsequent kernel
// launch on the same stream sees it complete.
const unsigned short *
cuda_fc_qs4cx_stage_host_x_fp16(const unsigned short *host_Xh, unsigned int M,
                                unsigned int K) {
  if (host_Xh == nullptr || M == 0 || K == 0)
    return nullptr;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  const size_t bytes = sizeof(unsigned short) * (size_t)M * K;
  if (!ensure_buf((void **)&g_stage_xh, &g_stage_xh_cap, bytes))
    return nullptr; // OOM, or capturing before the buffer was prewarmed
  if (cudaMemcpyAsync(g_stage_xh, host_Xh, bytes, cudaMemcpyHostToDevice,
                      StreamManager::Global().GetStream()) != cudaSuccess) {
    cudaGetLastError();
    return nullptr;
  }
  return g_stage_xh;
}

// Stage a HOST-resident QS4CX plain weight + fp16 scales into cached device
// buffers and return the device pointers. A model-load timing race can leave a
// weight in unregistered host memory (cudaPointerGetAttributes Unregistered)
// instead of the managed pool; the dp4a repack kernel reads the plain payload
// on the GPU, so a host pointer makes the cudaFcGemm gate fall to the i8mm
// host dot, which SIGILLs on Orin (no i8mm). Reuses g_qs4cx_weight_cache
// (weights are constant, uploaded once, keyed by the host plain pointer).
// Uploads happen on the first/prefill forward, NOT under graph capture; bails
// if asked to allocate under capture so the caller can fall back. Returns
// false on failure.
bool cuda_fc_qs4cx_stage_host_weight(const unsigned char *host_plain,
                                     const unsigned short *host_scales,
                                     unsigned int N, unsigned int K,
                                     const unsigned char **dev_w,
                                     const unsigned short **dev_scales) {
  if (host_plain == nullptr || host_scales == nullptr || N == 0 || K == 0)
    return false;
  std::lock_guard<std::mutex> lk(g_qs4cx_mtx);
  auto it = g_qs4cx_weight_cache.find(host_plain);
  if (it == g_qs4cx_weight_cache.end()) {
    if (StreamManager::Global().isCapturing())
      return false;
    const size_t w_bytes = (size_t)N * ((K + 1u) / 2u);
    DevWeight dw;
    if (cudaMalloc(&dw.d_w, w_bytes) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    if (cudaMalloc(&dw.d_sc, sizeof(unsigned short) * (size_t)N) !=
        cudaSuccess) {
      cudaFree(dw.d_w);
      cudaGetLastError();
      return false;
    }
    cudaMemcpy(dw.d_w, host_plain, w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dw.d_sc, host_scales, sizeof(unsigned short) * (size_t)N,
               cudaMemcpyHostToDevice);
    it = g_qs4cx_weight_cache.emplace(host_plain, dw).first;
  }
  *dev_w = it->second.d_w;
  *dev_scales = it->second.d_sc;
  return true;
}

// Mark a QS4CX plain payload as exempt from the eager cuBLAS-i8
// [K,N] build (see g_i8_exempt). Called at load time before the prewarm walk.
void cuda_fc_qs4cx_prewarm_exempt_i8(const void *plain_w) {
  g_i8_exempt.insert(plain_w);
}

// NNTR_CUDA_I8_JIT=1: no persistent i8 cache exists at all -- the
// prefill GEMM unpacks the RESIDENT dp4a signed-packed int4 copy (VRAM
// source; unpacking from the pinned-host plain would re-pay ~700MB of PCIe
// per prefill) into a reusable scratch right before the IMMA GEMM, shares
// the dp4a rowsum (same per-channel sums), and leaves nothing resident.
// Removes the whole i8 term from the prefill VRAM peak at ~4-5ms per 1K
// prefill (tiled transpose, coalesced both sides).
static inline bool i8_jit_on() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_CUDA_I8_JIT");
    return e != nullptr && e[0] == '1';
  }();
  return v;
}

// Tiled transpose-unpack: dp4a packed [N, Kh] (byte = plain^0x88, nibbles =
// two's-complement signed 4-bit) -> int8 [K, N]. Reads coalesced along Kh,
// writes coalesced along N via the shared tile.
static const char *I8_JIT_SRC = R"CU(
extern "C" __global__ void i8_jit_unpack(const signed char *q4,
                                         signed char *w8, int N, int K,
                                         int Kh) {
  __shared__ signed char t[32][65];
  int nn0 = blockIdx.y * 32, kh0 = blockIdx.x * 32;
  int nn = nn0 + threadIdx.y, kh = kh0 + threadIdx.x;
  if (nn < N && kh < Kh) {
    unsigned char b = (unsigned char)q4[(long long)nn * Kh + kh];
    t[threadIdx.y][2 * threadIdx.x] =
      (signed char)((((b & 0xF) ^ 8) & 0xF) - 8);
    t[threadIdx.y][2 * threadIdx.x + 1] =
      (signed char)(((((b >> 4) & 0xF) ^ 8) & 0xF) - 8);
  }
  __syncthreads();
  int k0 = kh0 * 2, wn = nn0 + threadIdx.x;
  for (int kk = threadIdx.y; kk < 64; kk += 32) {
    int k = k0 + kk;
    if (k < K && wn < N)
      w8[(long long)k * N + wn] = t[threadIdx.x][kk];
  }
}

// Vectorized variant (K%8==0 && N%4==0, which covers every FC we ship):
// 64n x 64k tile, 256 threads; uint (4-byte) global loads along Kh and int
// (4-byte) coalesced global stores along N -- runs the ~1.8GB/prefill unpack
// traffic at near-memcpy bandwidth instead of byte-granular transactions.
extern "C" __global__ void i8_jit_unpack_v4(const unsigned char *q4,
                                            signed char *w8, int N, int K,
                                            int Kh) {
  __shared__ signed char t[64][68]; // [k_local][n_local], row stride 68 (4B)
  const int nn0 = blockIdx.y * 64;
  const int kh0 = blockIdx.x * 32; // bytes of Kh covered by this tile
  const int tid = threadIdx.x;     // 256 threads
  for (int rep = 0; rep < 2; ++rep) {
    int idx = tid + rep * 256;
    int nn = idx >> 3;   // 0..63
    int kb4 = idx & 7;   // which 4-byte group in the 32-byte span
    int n = nn0 + nn;
    int khb = kh0 + kb4 * 4;
    if (n < N && khb + 3 < Kh) {
      unsigned int v = *reinterpret_cast<const unsigned int *>(
        q4 + (long long)n * Kh + khb);
      int kl = kb4 * 8;
      for (int j = 0; j < 4; ++j) {
        unsigned int b = (v >> (8 * j)) & 0xFFu;
        t[kl + 2 * j][nn] = (signed char)((((b & 0xF) ^ 8) & 0xF) - 8);
        t[kl + 2 * j + 1][nn] =
          (signed char)(((((b >> 4) & 0xF) ^ 8) & 0xF) - 8);
      }
    } else if (n < N) { // Kh tail (unused when K%8==0, kept for safety)
      for (int j = 0; j < 4; ++j) {
        int kb = khb + j;
        if (kb < Kh) {
          unsigned char b = q4[(long long)n * Kh + kb];
          int kl = kb4 * 8 + 2 * j;
          t[kl][nn] = (signed char)((((b & 0xF) ^ 8) & 0xF) - 8);
          t[kl + 1][nn] = (signed char)(((((b >> 4) & 0xF) ^ 8) & 0xF) - 8);
        }
      }
    }
  }
  __syncthreads();
  const int k0 = kh0 * 2;
  for (int rep = 0; rep < 4; ++rep) {
    int idx = tid + rep * 256;
    int kl = idx >> 4; // 0..63
    int ni = idx & 15; // 16 ints cover 64 n
    int k = k0 + kl;
    int n = nn0 + ni * 4;
    if (k < K && n + 3 < N) {
      int val = *reinterpret_cast<const int *>(&t[kl][ni * 4]);
      *reinterpret_cast<int *>(w8 + (long long)k * N + n) = val;
    } else if (k < K) {
      for (int j = 0; j < 4; ++j)
        if (n + j < N)
          w8[(long long)k * N + n + j] = t[kl][ni * 4 + j];
    }
  }
}
)CU";

// Prewarm the dp4a packed-int4 weight cache on the CPU at LOAD (nntrainer
// ThreadManager-parallel), so the first inference does not pay the one-time
// plain -> signed packed int4 repack (nsys: ~38% of the cold-run GPU time when
// it was the Section-A repack) and the GPU is free of it. Mirrors
// repack_plain_i4 + weight_rowsum bit-exactly (the repack is a byte-wise
// XOR 0x88, see the kernel comment), then uploads the packed int4 +

// [i8-ephemeral] Free every cuBLAS-i8 weight cache. Decode (M=1) never reads
// them, so dropping them at the prefill->decode boundary removes their VRAM
// residency for the whole decode phase; a LATER prefill (multi-turn) lazily
// rebuilds per FC via ensure_i8_cache_locked (CPU unpack -- slower TTFT on
// that turn; the GPU repack upgrade is the follow-up). The dp4a int4 cache
// and the pinned-host plain source are untouched.
void cuda_fc_qs4cx_free_i8_caches() {
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  size_t freed = 0;
  for (auto &kv : g_i8_weight_cache) {
    if (kv.second.w8) {
      cudaFree(kv.second.w8);
      ++freed;
    }
    if (kv.second.rowsum)
      cudaFree(kv.second.rowsum);
  }
  g_i8_weight_cache.clear();
  if (freed)
    std::fprintf(stderr, "[i8-ephemeral] freed %zu cuBLAS-i8 weight caches\n",
                 freed);
}

// per-channel rowsum to the device cache (keyed by the plain payload pointer,
// same key the dp4a path looks up at forward). Idempotent.
bool cuda_fc_qs4cx_prewarm(const unsigned char *plain_w, unsigned int N,
                           unsigned int K) {
  if (plain_w == nullptr || N == 0 || K == 0)
    return true;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  if (g_dp4a_plain_cache.count(plain_w))
    return true; // already cached
  const size_t Kh = (K + 1u) / 2u;
  auto &tm = nntrainer::ThreadManager::Global();

  // Build + upload in bounded chunks: a full host mirror of the untied
  // lm_head (N=262144) is ~350MB packed + ~700MB int8 and those transients
  // WERE the process peak RSS once the Section-A copy was gone (RSS timeline:
  // a +1GB step right at the peak, late in load). ~64MB chunks keep the
  // prewarm off the peak entirely; results are byte-identical (same values,
  // same device offsets).
  static constexpr size_t PREWARM_CHUNK_BYTES = 64u << 20;

  DevWeightQ dw;
  if (cudaMalloc(&dw.plain, (size_t)N * Kh) != cudaSuccess)
    return false;
  if (cudaMalloc(&dw.rowsum, sizeof(int) * (size_t)N) != cudaSuccess) {
    cudaFree(dw.plain);
    return false;
  }
  {
    // packed int4 [N][Kh] in row chunks (rows are contiguous on both sides).
    const size_t chunk_rows =
      std::max<size_t>(1, std::min<size_t>(N, PREWARM_CHUNK_BYTES / Kh));
    std::vector<signed char> packed(chunk_rows * Kh);
    std::vector<int> rowsum(N, 0);
    for (size_t n0 = 0; n0 < N; n0 += chunk_rows) {
      const size_t rows = std::min(chunk_rows, (size_t)N - n0);
      tm.parallel_for(0, rows, [&](size_t r) {
        const unsigned char *src = plain_w + (n0 + r) * Kh;
        signed char *prow = packed.data() + r * Kh;
        long acc = 0;
        for (size_t kb = 0; kb < Kh; ++kb) {
          const unsigned char b = src[kb];
          prow[kb] = (signed char)(b ^ 0x88);
          // odd-K pad nibble is stored 8 (= int4 0), so it adds 0 here --
          // same rowsum the old k1<K guard produced.
          acc += ((int)(b & 0xF) - 8) + ((int)((b >> 4) & 0xF) - 8);
        }
        rowsum[n0 + r] = (int)acc;
      });
      cudaMemcpy(dw.plain + n0 * Kh, packed.data(), rows * Kh,
                 cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dw.rowsum, rowsum.data(), sizeof(int) * (size_t)N,
               cudaMemcpyHostToDevice);
  }
  g_dp4a_plain_cache.emplace(plain_w, dw);

  // Also prewarm the cuBLAS int8 [K,N] weight cache when the cuBLAS prefill FC
  // path is on: otherwise its one-time GPU repack (repack_plain_i8_kn, ~32% of
  // cold prefill GPU time) runs on the first prefill instead of at load.
  // Mirrors repack_plain_i8_kn (w8[k*N+n]=int4(n,k)) + weight_rowsum_kn
  // bit-exactly. Chunked along K ([k0,k1) rows of the [K,N] buffer are
  // contiguous on both sides); the per-channel rowsum accumulates across
  // chunks.
  static const char *_cb = std::getenv("NNTR_FC_CUDA_CUBLAS");
  if (_cb && _cb[0] != '0' && !i8_jit_on() &&
      !g_i8_weight_cache.count(plain_w) && !g_i8_exempt.count(plain_w)) {
    const size_t chunk_k =
      std::max<size_t>(1, std::min<size_t>(K, PREWARM_CHUNK_BYTES / N));
    std::vector<signed char> w8(chunk_k * (size_t)N);
    std::vector<long> rs8(N, 0);
    DevWeightI8 dw8;
    if (cudaMalloc(&dw8.w8, (size_t)K * N) == cudaSuccess &&
        cudaMalloc(&dw8.rowsum, sizeof(int) * (size_t)N) == cudaSuccess) {
      for (size_t k0 = 0; k0 < K; k0 += chunk_k) {
        const size_t ks = std::min(chunk_k, (size_t)K - k0);
        tm.parallel_for(0, (size_t)N, [&](size_t n) {
          const unsigned char *src = plain_w + n * Kh;
          long acc = 0;
          for (size_t kk = k0; kk < k0 + ks; ++kk) {
            const unsigned char b = src[kk >> 1];
            const int v = (int)((kk & 1) ? ((b >> 4) & 0xF) : (b & 0xF)) - 8;
            w8[(kk - k0) * N + n] = (signed char)v;
            acc += v;
          }
          rs8[n] += acc;
        });
        cudaMemcpy(dw8.w8 + k0 * N, w8.data(), ks * (size_t)N,
                   cudaMemcpyHostToDevice);
      }
      std::vector<int> rs8i(N);
      for (size_t n = 0; n < N; ++n)
        rs8i[n] = (int)rs8[n];
      cudaMemcpy(dw8.rowsum, rs8i.data(), sizeof(int) * (size_t)N,
                 cudaMemcpyHostToDevice);
      g_i8_weight_cache.emplace(plain_w, dw8);
    } else if (dw8.w8) {
      cudaFree(dw8.w8);
    }
  }
  return true;
}

// True when the dp4a derived cache for this plain pointer
// already exists -- the dispatch then only needs the pointer VALUE as a key,
// so a host-heap (non-device-accessible) payload is fine and the host->device
// weight staging can be skipped entirely.
bool cuda_fc_qs4cx_has_cache(const unsigned char *plain_w) {
  if (plain_w == nullptr)
    return false;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  return g_dp4a_plain_cache.count(plain_w) != 0;
}

bool cuda_fc_qs4cx_fused_normq_enabled() { return fused_normq_on(); }

bool cuda_fc_qs4cx_rmsnorm_prequant_fp16(const unsigned short *x,
                                         const unsigned short *gamma,
                                         unsigned short *y, float eps,
                                         unsigned int rows,
                                         unsigned int width) {
  if (!fused_normq_on())
    return false;
  if (rows == 0 || width == 0)
    return false;
  const bool vec4 =
    cuda_vec4_rows_small(rows) && cuda_vec4_rows_ok(width, x, y);
  auto k = CudaContext::Global().registerCudaKernel(
    FC_QS4CX_DP4A_SRC, vec4 ? "rmsnorm_quant_i8_h_v4" : "rmsnorm_quant_i8_h");
  if (!k)
    return false;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // Sizing the quant scratch is a cudaMalloc, which is illegal mid-capture --
  // ensure_buf refuses there and we hand the row back to the plain norm. In
  // practice prefill has already grown the scratch past a single decode row by
  // the time the decode graph is captured, so this is a cold-start guard, not
  // the steady state.
  if (!dp4a_stage_scratch(rows, width))
    return false;
  int m = (int)rows, kk = (int)width;
  int has_gamma = (gamma == nullptr)                       ? 0
                  : (!vec4 || cuda_vec4_rows_ok(4, gamma)) ? 1
                                                           : 2;
  k->SetKernelArguments(0, &x, sizeof(x));
  k->SetKernelArguments(1, &gamma, sizeof(gamma));
  k->SetKernelArguments(2, &y, sizeof(y));
  k->SetKernelArguments(3, &g_dp4a_q8, sizeof(g_dp4a_q8));
  k->SetKernelArguments(4, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  k->SetKernelArguments(5, &g_dp4a_azp, sizeof(g_dp4a_azp));
  k->SetKernelArguments(6, &m, sizeof(m));
  k->SetKernelArguments(7, &kk, sizeof(kk));
  k->SetKernelArguments(8, &eps, sizeof(eps));
  k->SetKernelArguments(9, &has_gamma, sizeof(has_gamma));
  const int block[3] = {vec4 ? 512 : 256, 1, 1};
  const int grid[3] = {(int)rows, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*k, grid, block))

    return false;
  mark_quant_staged(y, kk);
  maybe_finish(y);
  return true;
}

bool cuda_fc_qs4cx_dp4a_gemm_fp32(const float *X, const unsigned char *plain_w,
                                  const unsigned short *scales_fp16, float *Y,
                                  unsigned int M, unsigned int N,
                                  unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  auto kq =
    CudaContext::Global().registerCudaKernel(FC_QS4CX_DP4A_SRC, "act_quant_i8");
  if (!kq) {
    ml_loge("[CUDA] fc_qs4cx dp4a: kernel registration failed");
    return false;
  }
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  if (!dp4a_stage_scratch(M, K))
    return false;
  int m = (int)M, k = (int)K;
  kq->SetKernelArguments(0, &X, sizeof(X));
  kq->SetKernelArguments(1, &g_dp4a_q8, sizeof(g_dp4a_q8));
  kq->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kq->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
  kq->SetKernelArguments(4, &m, sizeof(m));
  kq->SetKernelArguments(5, &k, sizeof(k));
  const int qb[3] = {256, 1, 1};
  const int qg[3] = {(int)M, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kq, qg, qb))
    return false;
  if (!dp4a_repack_and_gemm(plain_w, scales_fp16, Y, M, N, K))
    return false;
  maybe_finish(Y);
  return true;
}

bool cuda_fc_qs4cx_dp4a_gemm_fp16(const unsigned short *Xh,
                                  const unsigned char *plain_w,
                                  const unsigned short *scales_fp16,
                                  unsigned short *Yh, unsigned int M,
                                  unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  const bool q_vec4 =
    fused_normq_on() && cuda_vec4_rows_small(M) && cuda_vec4_rows_ok(K, Xh);
  auto kqh = CudaContext::Global().registerCudaKernel(
    FC_QS4CX_DP4A_SRC, q_vec4 ? "act_quant_i8_h_v4" : "act_quant_i8_h");
  auto kc =
    CudaContext::Global().registerCudaKernel(FC_QS4CX_DP4A_SRC, "cvt_f2h");
  if (!kqh || !kc) {
    ml_loge("[CUDA] fc_qs4cx dp4a fp16: kernel registration failed");
    return false;
  }
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // No float Y staging here: the GEMM writes fp16 directly (out_fp16=1 below),
  // so g_dp4a_yf is unused on this path. Allocating it lazily would cudaMalloc
  // inside a CUDA-graph capture (NNTR_CUDA_GRAPH) on the first captured decode
  // token and invalidate the graph -- so it is deliberately NOT sized here.
  if (!dp4a_stage_scratch(M, K))
    return false;
  int m = (int)M, k = (int)K;
  // 1) int8 activation quant from the fp16 input -- unless g_dp4a_q8 already
  // holds exactly this activation. That happens for every FC group fed by a
  // norm (q/k/v off attention_norm, gate/up off ffn_norm): the fused
  // norm+quant staged it, and the sibling FCs after the first one would
  // otherwise recompute an identical buffer. The guard is pointer + K + "no
  // kernel dispatched since", so a recycled pool address cannot impersonate
  // the staged row.
  if (!quant_staged_for(Xh, k)) {
    kqh->SetKernelArguments(0, &Xh, sizeof(Xh));
    kqh->SetKernelArguments(1, &g_dp4a_q8, sizeof(g_dp4a_q8));
    kqh->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
    kqh->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
    kqh->SetKernelArguments(4, &m, sizeof(m));
    kqh->SetKernelArguments(5, &k, sizeof(k));
    const int qb[3] = {q_vec4 ? 512 : 256, 1, 1};
    const int qg[3] = {(int)M, 1, 1};
    if (!StreamManager::Global().DispatchCommand(*kqh, qg, qb))
      return false;
  }
  // 2) repack + GEMM writing fp16 directly: the float->fp16 conversion is
  // folded into the GEMM epilogue (out_fp16=1), removing the separate cvt_f2h
  // kernel + the FP32 staging buffer (one fewer kernel per FC -- a decode
  // launch-overhead win). (void)kc keeps the registration check above harmless.
  (void)kc;
  if (!dp4a_repack_and_gemm(plain_w, scales_fp16, reinterpret_cast<float *>(Yh),
                            M, N, K,
                            /*out_fp16=*/1))
    return false;
  // Re-stamp the handoff past this FC's own dispatches so the NEXT sibling on
  // the same activation still sees a valid staging (the GEMM bumped the
  // sequence). With the lever off nothing is ever published, so no FC can
  // skip its quant.
  if (fused_normq_on())
    mark_quant_staged(Xh, k);
  maybe_finish(Yh);
  return true;
}

// NNTR_CUDA_LMHEAD_FPACT: the fp-activation int4 GEMV for the huge-N decode
// lm_head. Default ON -- see fpact_gemv_i4_h for why w4a8's activation quant
// costs argmax fidelity exactly where a 262144-wide output cannot afford it --
// with an explicit =0 restoring the dp4a route. VALUE-checked, not
// presence-checked, and read here next to the path it governs so the
// dispatcher's gate can stay a pure SHAPE test.
static inline bool lmhead_fpact_on() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_CUDA_LMHEAD_FPACT");
    return !(e != nullptr && e[0] == '0');
  }();
  return v;
}

bool cuda_fc_qs4cx_fpact_gemv_fp16(const unsigned short *Xh,
                                   const unsigned char *plain_w,
                                   const float *scales_fp32, unsigned short *Yh,
                                   unsigned int N, unsigned int K) {
  if (!lmhead_fpact_on())
    return false;
  if (N == 0 || K == 0 || Xh == nullptr || plain_w == nullptr ||
      scales_fp32 == nullptr || Yh == nullptr)
    return false;
  // Normally built at load time by the prewarm; this is the lazy fallback for
  // a run that did not prewarm.
  const float *sc = nullptr;
  if (!cuda_fc_qs4cx_scales_to_uvm_fp32(scales_fp32, N, &sc, true))
    return false;
  auto kg = CudaContext::Global().registerCudaKernel(FC_QS4CX_DP4A_SRC,
                                                     "fpact_gemv_i4_h");
  if (!kg) {
    ml_loge("[CUDA] fc_qs4cx fp-act lm_head GEMV: kernel registration failed");
    return false;
  }
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // The SAME derived weight cache the dp4a path uses (signed packed int4 +
  // rowsum; the rowsum is dead here, there is no activation zero-point to
  // correct). Reusing it is what keeps this route free of extra VRAM -- and
  // what lets it run at all under NNTR_QS4CX_HEAP_BYPASS + the plain drop,
  // where the plain payload is neither device-accessible nor still resident.
  DevWeightQ *dwp = ensure_dp4a_cache_locked(plain_w, N, K);
  if (!dwp)
    return false;
  signed char *plain = dwp->plain;
  int n = (int)N, k = (int)K;
  // Wide arm: 8 channels per lane per step needs K a multiple of 8 -- which
  // also makes Kh a multiple of 4, so every weight ROW base is 4-byte aligned
  // -- and a 16-byte aligned activation row. Anything else takes the scalar
  // arm rather than risking a misaligned access (which aborts the launch).
  const int wide =
    ((K & 7u) == 0u && (reinterpret_cast<uintptr_t>(Xh) & 15u) == 0u) ? 1 : 0;
  kg->SetKernelArguments(0, &Xh, sizeof(Xh));
  kg->SetKernelArguments(1, &plain, sizeof(plain));
  kg->SetKernelArguments(2, &sc, sizeof(sc));
  kg->SetKernelArguments(3, &Yh, sizeof(Yh));
  kg->SetKernelArguments(4, &n, sizeof(n));
  kg->SetKernelArguments(5, &k, sizeof(k));
  kg->SetKernelArguments(6, &wide, sizeof(wide));
  // One warp per output row, 4 warps per block -- the dp4a_gemv launch shape.
  const int blk[3] = {128, 1, 1};
  const int grd[3] = {((int)N + 3) / 4, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kg, grd, blk))
    return false;
  // Which route a decode's logits came from is otherwise invisible: every
  // refusal above silently falls through to dp4a, and the difference shows up
  // only as different sampled tokens hundreds of steps later. Say it once.
  static bool announced = false;
  if (!announced) {
    announced = true;
    ml_logi("[CUDA] lm_head decode on the fp-activation int4 GEMV "
            "(N=%u K=%u, %s arm); NNTR_CUDA_LMHEAD_FPACT=0 restores w4a8 dp4a",
            N, K, wide ? "wide" : "scalar");
  }
  maybe_finish(Yh);
  return true;
}

// w4a8 on the INT8 Tensor Cores via cuBLAS (prefill FC). Same quant scheme as
// the dp4a path -- per-row asym int8 activation + symmetric int4 weight -- but
// the int8xint8->int32 GEMM runs on IMMA Tensor Cores instead of __dp4a on the
// int ALU (~10x the GEMM throughput at prefill M). The int32 accumulate is
// exact so the result is bit-identical to dp4a; the int4->int8 weight unpack is
// cached (one-time) to keep it off the per-call critical path.
bool cuda_fc_qs4cx_cublas_i8_gemm_fp16(const unsigned short *Xh,
                                       const unsigned char *plain_w,
                                       const unsigned short *scales_fp16,
                                       unsigned short *Yh, unsigned int M,
                                       unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  const bool q_vec4 =
    fused_normq_on() && cuda_vec4_rows_small(M) && cuda_vec4_rows_ok(K, Xh);
  auto kqh = CudaContext::Global().registerCudaKernel(
    FC_QS4CX_DP4A_SRC, q_vec4 ? "act_quant_i8_h_v4" : "act_quant_i8_h");
  auto kde = CudaContext::Global().registerCudaKernel(FC_QS4CX_DP4A_SRC,
                                                      "dequant_i32_fp16");
  if (!kqh || !kde) {
    ml_loge("[CUDA] fc_qs4cx cublas-i8: kernel registration failed");
    return false;
  }
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // cuBLAS int8 IMMA requires the GEMM dims to be multiples of 32 (measured:
  // M=260/272 -> CUBLAS_STATUS_NOT_SUPPORTED, 256/320/512 OK). The prefill
  // token count M is arbitrary (e.g. 511), so pad the activation row count up
  // to a multiple of 32 for the GEMM only -- the extra rows are computed from
  // (harmless int8) scratch and ignored by the epilogue, which writes just the
  // real M rows. N and K are multiples of 32 by the load invariant.
  const unsigned Mpad = ((M + 31u) / 32u) * 32u;
  if (!dp4a_stage_scratch(Mpad, K))
    return false;
  const int m = (int)M, n = (int)N, k = (int)K, mpad = (int)Mpad;

  // 1) int8 activation quant from the fp16 input (reuse the dp4a quantizer).
  // Skip when this exact (Xh,K) was just quantized into g_dp4a_q8 by a sibling
  // FC (q/k/v share attention_norm; gate/up share ffn_norm) -- the buffer still
  // holds it. See g_last_quant_xh above.
  // Opt-in: measured gain is within the thermal noise floor on Orin (act_quant
  // is not on the critical path -- the GEMM is), so default OFF; correct +
  // ready if a less-throttled host or a power budget makes the redundant
  // launches matter.
  static const bool quant_dedup = []() {
    const char *e = std::getenv("NNTR_QUANT_DEDUP");
    return e != nullptr && e[0] == '1';
  }();
  const bool reuse_quant = quant_dedup && quant_staged_for(Xh, k);
  if (!reuse_quant) {
    kqh->SetKernelArguments(0, &Xh, sizeof(Xh));
    kqh->SetKernelArguments(1, &g_dp4a_q8, sizeof(g_dp4a_q8));
    kqh->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
    kqh->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
    kqh->SetKernelArguments(4, &m, sizeof(m));
    kqh->SetKernelArguments(5, &k, sizeof(k));
    const int qb[3] = {q_vec4 ? 512 : 256, 1, 1};
    const int qg[3] = {(int)M, 1, 1};
    if (!StreamManager::Global().DispatchCommand(*kqh, qg, qb))
      return false;
  }

  // 2) int8 weight [K,N] + per-channel rowsum. JIT mode transpose-
  // unpacks the RESIDENT dp4a packed copy into a reusable scratch (nothing
  // stays resident; rowsum shared with the dp4a cache -- same values); else
  // the persistent per-weight cache (one-time unpack).
  signed char *w8src = nullptr;
  int *rowsum = nullptr;
  if (i8_jit_on()) {
    DevWeightQ *dw4 = ensure_dp4a_cache_locked(plain_w, N, K);
    if (!dw4)
      return false;
    static signed char *jit_w8 = nullptr;
    static size_t jit_cap = 0;
    if (!ensure_buf((void **)&jit_w8, &jit_cap, (size_t)K * N))
      return false;
    // Vectorized transpose for 8|K && 4|N (every FC we ship); byte-granular
    // fallback otherwise.
    const bool vec_ok = ((K & 7u) == 0u) && ((N & 3u) == 0u);
    auto ku = CudaContext::Global().registerCudaKernel(
      I8_JIT_SRC, vec_ok ? "i8_jit_unpack_v4" : "i8_jit_unpack");
    if (!ku)
      return false;
    const int khi = (int)((K + 1u) / 2u);
    ku->SetKernelArguments(0, &dw4->plain, sizeof(dw4->plain));
    ku->SetKernelArguments(1, &jit_w8, sizeof(jit_w8));
    ku->SetKernelArguments(2, &n, sizeof(n));
    ku->SetKernelArguments(3, &k, sizeof(k));
    ku->SetKernelArguments(4, &khi, sizeof(khi));
    const int ub[3] = {vec_ok ? 256 : 32, vec_ok ? 1 : 32, 1};
    const int ug[3] = {(khi + 31) / 32,
                       vec_ok ? ((int)N + 63) / 64 : ((int)N + 31) / 32, 1};
    if (!StreamManager::Global().DispatchCommand(*ku, ug, ub))
      return false;
    w8src = jit_w8;
    rowsum = dw4->rowsum;
  } else {
    DevWeightI8 *dw8 = ensure_i8_cache_locked(plain_w, N, K);
    if (!dw8)
      return false;
    w8src = dw8->w8;
    rowsum = dw8->rowsum;
  }

  // 3) int32 GEMM output scratch [Mpad,N] (+tail pad: IMMA can write/read C in
  // wide vectorized tiles past the last element on large shapes).
  if (!ensure_buf((void **)&g_i8_c, &g_i8_c_cap,
                  sizeof(int) * (size_t)Mpad * N + FC_I8_TAIL_PAD))
    return false;

  // 4) INT8 IMMA GEMM on the Tensor Cores (Mpad rows; same backend stream as
  // the kernels). C is [Mpad,N] row-major; the real M rows are at the same
  // offsets so the epilogue reads C[m*N+n] for m<M directly.
  if (!BlasManager::Global().igemmRowMajor(mpad, n, k, g_dp4a_q8, w8src,
                                           g_i8_c))
    return false;

  // 5) dequant epilogue (bit-identical math to the dp4a kernel) -> fp16 Y.
  kde->SetKernelArguments(0, &g_i8_c, sizeof(g_i8_c));
  kde->SetKernelArguments(1, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kde->SetKernelArguments(2, &g_dp4a_azp, sizeof(g_dp4a_azp));
  kde->SetKernelArguments(3, &rowsum, sizeof(rowsum));
  kde->SetKernelArguments(4, &scales_fp16, sizeof(scales_fp16));
  kde->SetKernelArguments(5, &Yh, sizeof(Yh));
  kde->SetKernelArguments(6, &m, sizeof(m));
  kde->SetKernelArguments(7, &n, sizeof(n));
  const int db[3] = {16, 16, 1};
  const int dg[3] = {((int)N + 15) / 16, ((int)M + 15) / 16, 1};
  if (!StreamManager::Global().DispatchCommand(*kde, dg, db))
    return false;
  // Re-stamp past this FC's own dispatches (the epilogue bumped the sequence)
  // so a sibling prefill FC on the same activation can still reuse the quant.
  if (quant_dedup)
    mark_quant_staged(Xh, k);
  maybe_finish(Yh);
  // Catch an ASYNC failure in the cuBLAS IMMA GEMM / epilogue (the sync cuBLAS
  // status was already checked). On Orin a large-M IMMA can fault at runtime
  // and leave a STICKY cuda error -- which then makes the NEXT layer's
  // cudaPointerGetAttributes (rms_norm dev_ok gate) fail, dropping rms_norm to
  // its host path that reads device/managed activations under cMA=0 -> SIGSEGV.
  // Clearing + returning false makes the caller fall back to the (correct) dp4a
  // GEMM cleanly instead of corrupting the rest of the forward.
  {
    cudaError_t _e = cudaGetLastError();
    if (_e != cudaSuccess) {
      if (std::getenv("NNTR_IGEMM_DBG"))
        std::fprintf(
          stderr,
          "[IGEMM] async error after GEMM M=%d N=%d K=%d: %s -> dp4a "
          "fallback\n",
          m, n, k, cudaGetErrorString(_e));
      return false;
    }
  }
  return true;
}

// Diagnostic / high-accuracy fp16 path: FP32-precision activation (no int8
// quant). fp16 -> fp32, naive plain-decode FP32-act GEMM, fp32 -> fp16. Used
// when NNTR_FC_CUDA_DP4A=0 with an fp16 activation.
bool cuda_fc_qs4cx_gemm_fp16_naive(const unsigned short *Xh,
                                   const unsigned char *plain_w,
                                   const unsigned short *scales_fp16,
                                   unsigned short *Yh, unsigned int M,
                                   unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  auto kh2f =
    CudaContext::Global().registerCudaKernel(FC_QS4CX_DP4A_SRC, "cvt_h2f");
  auto kf2h =
    CudaContext::Global().registerCudaKernel(FC_QS4CX_DP4A_SRC, "cvt_f2h");
  if (!kh2f || !kf2h)
    return false;
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  const size_t xn = (size_t)M * K, yn = (size_t)M * N;
  if (!ensure_buf((void **)&g_dp4a_xf, &g_dp4a_xf_cap, sizeof(float) * xn) ||
      !ensure_buf((void **)&g_dp4a_yf, &g_dp4a_yf_cap, sizeof(float) * yn))
    return false;
  int xni = (int)xn, yni = (int)yn;
  const int cb[3] = {256, 1, 1};
  kh2f->SetKernelArguments(0, &Xh, sizeof(Xh));
  kh2f->SetKernelArguments(1, &g_dp4a_xf, sizeof(g_dp4a_xf));
  kh2f->SetKernelArguments(2, &xni, sizeof(xni));
  const int xg[3] = {((int)xn + 255) / 256, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kh2f, xg, cb))
    return false;
  // naive plain-decode FP32-act GEMM (mutex-free; its own dispatch + finish).
  if (!cuda_fc_qs4cx_gemm_fp32(g_dp4a_xf, plain_w, scales_fp16, g_dp4a_yf, M, N,
                               K))
    return false;
  kf2h->SetKernelArguments(0, &g_dp4a_yf, sizeof(g_dp4a_yf));
  kf2h->SetKernelArguments(1, &Yh, sizeof(Yh));
  kf2h->SetKernelArguments(2, &yni, sizeof(yni));
  const int yg[3] = {((int)yn + 255) / 256, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kf2h, yg, cb))
    return false;
  maybe_finish(Yh);
  return true;
}

// ---------------------------------------------------------------------------
// Q6_K lm_head GEMV (port of the OpenCL kernel_mul_mv_q6_K_f32 = llama.cpp's
// mul_mv_q6_K). One CUDA block = N_SIMDGROUP(2) output rows x N_SIMDWIDTH(16)
// lanes; the 16 lanes split each 256-element super-block, accumulate over all
// blocks of the row, then reduce. Reads the FP16 hidden + (managed) Q6_K weight
// directly on the device and writes FP16 logits to the device output -- no host
// bounce, so it works under a device-only activation pool (NNTR_CUDA_DEV_ACT)
// where the host Q6_K GEMV would fault. gemma2/qwen3 keep the Q6_K lm_head;
// this is the GPU path gemma4 gets from its QS4CX untied lm_head.
static const char *Q6K_GEMV_SRC = R"CU(
#define QK_K 256
typedef unsigned char  u8;
typedef signed char    s8;
typedef unsigned short u16;

typedef struct { u8 ql[128]; u8 qh[64]; s8 scales[16]; u16 d; } block_q6_K;

__device__ __forceinline__ float h2f(u16 h) {
  unsigned int s = (unsigned int)(h & 0x8000) << 16;
  unsigned int e = (h >> 10) & 0x1F;
  unsigned int m = h & 0x3FF;
  unsigned int out;
  if (e == 0) {
    if (m == 0) { out = s; }
    else { e = 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF;
           out = s | ((e + 112) << 23) | (m << 13); }
  } else if (e == 31) { out = s | 0x7F800000u | (m << 13); }
  else { out = s | ((e + 112) << 23) | (m << 13); }
  return __int_as_float((int)out);
}

__device__ __forceinline__ u16 f2h(float f) {
  unsigned int x = (unsigned int)__float_as_int(f);
  unsigned int sign = (x >> 16) & 0x8000u;
  int exp = (int)((x >> 23) & 0xFF) - 127 + 15;
  unsigned int man = x & 0x7FFFFFu;
  if (exp <= 0) {
    if (exp < -10) return (u16)sign;
    man |= 0x800000u;
    unsigned int shift = (unsigned int)(14 - exp);
    unsigned int half = man >> shift;
    if ((man >> (shift - 1)) & 1u) half += 1; // round to nearest
    return (u16)(sign | half);
  } else if (exp >= 31) {
    return (u16)(sign | 0x7C00u);
  }
  unsigned int half = (unsigned int)(exp << 10) | (man >> 13);
  if ((man >> 12) & 1u) half += 1; // round to nearest
  return (u16)(sign | half);
}

extern "C" __global__ void q6k_gemv(const void *src0, const u16 *src1, u16 *dst,
                                    int ne00, int ne01) {
  const int N_SIMDWIDTH = 16;
  __shared__ float red[2][16];
  int nb = ne00 / QK_K;
  int row_group = threadIdx.x / N_SIMDWIDTH;
  int lane = threadIdx.x % N_SIMDWIDTH;
  int row = blockIdx.x * 2 + row_group;
  const block_q6_K *x = (const block_q6_K *)src0 + (long)row * nb;
  const u16 *yy = src1;
  u8 kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
  int tid = lane;
  int ip = tid / 8, il = tid % 8, l0 = 4 * il;
  int is = 8 * ip + l0 / 16;
  int y_offset = 128 * ip + l0;
  int q_offset_l = 64 * ip + l0;
  int q_offset_h = 32 * ip + l0;
  float sumf = 0.0f;
  if (row < ne01) {
    for (int i = 0; i < nb; i++) {
      const u8 *q1 = x[i].ql + q_offset_l;
      const u8 *q2 = q1 + QK_K / 8;
      const u8 *qh = x[i].qh + q_offset_h;
      const s8 *sc = x[i].scales + is;
      const u16 *y = yy + i * QK_K + y_offset;
      float dall = h2f(x[i].d);
      float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (int j = 0; j < 4; j++) {
        s0 += h2f(y[j + 0])  * ((float)((q1[j] & 0xF) | ((qh[j] & kmask1) << 4)) - 32.f);
        s1 += h2f(y[j + 32]) * ((float)((q2[j] & 0xF) | ((qh[j] & kmask2) << 2)) - 32.f);
        s2 += h2f(y[j + 64]) * ((float)((q1[j] >> 4)  | ((qh[j] & kmask3) >> 0)) - 32.f);
        s3 += h2f(y[j + 96]) * ((float)((q2[j] >> 4)  | ((qh[j] & kmask4) >> 2)) - 32.f);
      }
      sumf += dall * (s0 * sc[0] + s1 * sc[2] + s2 * sc[4] + s3 * sc[6]);
    }
  }
  red[row_group][lane] = sumf;
  __syncthreads();
  for (int off = N_SIMDWIDTH / 2; off > 0; off >>= 1) {
    if (lane < off) red[row_group][lane] += red[row_group][lane + off];
    __syncthreads();
  }
  if (lane == 0 && row < ne01)
    dst[row] = f2h(red[row_group][0]);
}
)CU";

bool lmhead_gemv_q6_k_cuda(const void *w_q6k_dev,
                           const unsigned short *hidden_fp16_dev,
                           unsigned short *logits_fp16_dev, int vocab,
                           int hidden) {
  if (vocab <= 0 || hidden <= 0 || (hidden % 256) != 0)
    return false;
  auto kernel =
    CudaContext::Global().registerCudaKernel(Q6K_GEMV_SRC, "q6k_gemv");
  if (!kernel)
    return false;
  kernel->SetKernelArguments(0, &w_q6k_dev, sizeof(w_q6k_dev));
  kernel->SetKernelArguments(1, &hidden_fp16_dev, sizeof(hidden_fp16_dev));
  kernel->SetKernelArguments(2, &logits_fp16_dev, sizeof(logits_fp16_dev));
  kernel->SetKernelArguments(3, &hidden, sizeof(hidden));
  kernel->SetKernelArguments(4, &vocab, sizeof(vocab));
  const int block[3] = {32, 1, 1};
  const int grid[3] = {(vocab + 1) / 2, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  maybe_finish();
  return true;
}

} // namespace nntrainer::cuda

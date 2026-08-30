// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_elementwise.cpp
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Device element-wise ops (NVRTC kernels) -- geglu/add/scalar/slice.
 */

#include "cuda_elementwise.h"

#include <cuda_context.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

#include <cuda_runtime.h>

namespace nntrainer::cuda {

static const char *ELTWISE_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float ew_h2f(unsigned short h) {
  unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, o;
  if (e == 0u) {
    if (m == 0u) o = s;
    else { int x=-1; do{m<<=1;x++;}while((m&0x400u)==0u); m&=0x3FFu;
           o = s | ((unsigned int)(127-15-x)<<23) | (m<<13); }
  } else if (e == 0x1Fu) o = s | 0x7F800000u | (m<<13);
  else o = s | ((e + (127u-15u))<<23) | (m<<13);
  return __int_as_float((int)o);
}
__device__ __forceinline__ unsigned short ew_f2h(float f) {
  unsigned int x=(unsigned int)__float_as_int(f), s=(x>>16)&0x8000u, mant=x&0x7FFFFFu;
  int e=(int)((x>>23)&0xFFu);
  if (e==0xFF) return (unsigned short)(s|0x7C00u|(mant?0x200u:0u));
  int exp=e-127+15;
  if (exp>=0x1F) return (unsigned short)(s|0x7C00u);
  if (exp<=0){ if(exp<-10) return (unsigned short)s; mant|=0x800000u; int sh=14-exp;
    unsigned int hh=mant>>sh, rem=mant&((1u<<sh)-1u), half=1u<<(sh-1);
    if(rem>half||(rem==half&&(hh&1u))) hh++; return (unsigned short)(s|hh); }
  unsigned int hh=((unsigned int)exp<<10)|(mant>>13), rem=mant&0x1FFFu;
  if(rem>0x1000u||(rem==0x1000u&&(hh&1u))) hh++;
  return (unsigned short)(s|hh);
}
__global__ void geglu_fp16(const unsigned short *gate, const unsigned short *up,
                           unsigned short *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = ew_h2f(gate[i]);
  const float k = 0.7978845608028654f;
  float g = 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
  out[i] = ew_f2h(g * ew_h2f(up[i]));
}
// SwiGLU: out[i] = silu(gate[i]) * up[i], silu(x) = x / (1 + exp(-x)) (qwen3/
// llama FFN). Same shape as geglu_fp16, SiLU gate instead of gelu_tanh.
__global__ void swiglu_fp16(const unsigned short *gate, const unsigned short *up,
                            unsigned short *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = ew_h2f(gate[i]);
  float s = x / (1.0f + expf(-x));
  out[i] = ew_f2h(s * ew_h2f(up[i]));
}
__global__ void add_fp16(const unsigned short *a, const unsigned short *b,
                         unsigned short *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = ew_f2h(ew_h2f(a[i]) + ew_h2f(b[i]));
}
__global__ void scalar_mul_fp16(const unsigned short *in, unsigned short *out,
                                int n, float scalar) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = ew_f2h(ew_h2f(in[i]) * scalar);
}
// Device-slot V-copy: write into the KV cache at the live slot d_pos[0] computed
// on-device (out_base is the cache BASE, width = per-row element count), so a
// captured graph writes V to the correct (new-token) slot on every replay.
// [kv-window-ring] ring_cap > 0 maps each ABSOLUTE row (d_pos[0] + i/width) to
// its physical ring row (% ring_cap): a sliding layer's cache only has ring_cap
// physical rows, so writing at the absolute row lands outside it (or, with a
// full-size buffer, at a row nothing reads). Mapped PER ROW, not just at the
// base, so a multi-row prefill-chunk write stays correct without relying on the
// Wcap seam-alignment invariant.
__global__ void scalar_mul_fp16_slot(const unsigned short *in,
                                     unsigned short *out_base, int n, float scalar,
                                     const int *d_pos, int width, int ring_cap) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long row_abs = (long)d_pos[0] + i / width;
  long row = (ring_cap > 0) ? (row_abs % ring_cap) : row_abs;
  out_base[row * width + (i % width)] = ew_f2h(ew_h2f(in[i]) * scalar);
}
__global__ void slice_copy_fp16(const unsigned short *in, unsigned short *out,
                                int rows, int in_width, int layer_off, int fs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * fs) return;
  int r = idx / fs, f = idx % fs;
  out[(size_t)r * fs + f] = in[(size_t)r * in_width + layer_off + f];
}
__global__ void softcap_fp16(const unsigned short *in, unsigned short *out,
                             int n, float cap) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = ew_f2h(cap * tanhf(ew_h2f(in[i]) / cap));
}
}
)CU";

// Two-pass on-GPU greedy argmax over the vocab logits. Pass 1: each of GRIDDIM
// blocks reduces a grid-strided slice of [N] to one (max, idx) pair, written to
// the per-block scratch (pmax[b], pidx[b]). Pass 2: a single block reduces the
// GRIDDIM partials to the final (max, idx) and writes the 4-byte index to
// oidx[0]. Ties resolve to the LOWEST index (matches std::max_element, which
// keeps the first of equal maxima). fp32 and fp16 variants (fp16 decoded inline
// with the same half->float as the other elementwise kernels).
static const char *ARGMAX_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float am_h2f(unsigned short h) {
  unsigned int s = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, o;
  if (e == 0u) {
    if (m == 0u) o = s;
    else { int x=-1; do{m<<=1;x++;}while((m&0x400u)==0u); m&=0x3FFu;
           o = s | ((unsigned int)(127-15-x)<<23) | (m<<13); }
  } else if (e == 0x1Fu) o = s | 0x7F800000u | (m<<13);
  else o = s | ((e + (127u-15u))<<23) | (m<<13);
  return __int_as_float((int)o);
}
// Block-reduce shared (val,idx), tie -> lowest idx. blockDim.x must be 256.
__device__ __forceinline__ void am_block_reduce(float *sv, int *si) {
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      int j = threadIdx.x + s;
      if (sv[j] > sv[threadIdx.x] ||
          (sv[j] == sv[threadIdx.x] && si[j] < si[threadIdx.x])) {
        sv[threadIdx.x] = sv[j];
        si[threadIdx.x] = si[j];
      }
    }
    __syncthreads();
  }
}
__global__ void argmax_p1_f32(const float *logits, int n, float *pmax,
                              int *pidx) {
  __shared__ float sv[256];
  __shared__ int si[256];
  float bv = -3.402823466e+38f; // -FLT_MAX
  int bi = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    float v = logits[i];
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  am_block_reduce(sv, si);
  if (threadIdx.x == 0) { pmax[blockIdx.x] = sv[0]; pidx[blockIdx.x] = si[0]; }
}
__global__ void argmax_p1_f16(const unsigned short *logits, int n, float *pmax,
                              int *pidx) {
  __shared__ float sv[256];
  __shared__ int si[256];
  float bv = -3.402823466e+38f;
  int bi = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    float v = am_h2f(logits[i]);
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  am_block_reduce(sv, si);
  if (threadIdx.x == 0) { pmax[blockIdx.x] = sv[0]; pidx[blockIdx.x] = si[0]; }
}
__global__ void argmax_p2(const float *pmax, const int *pidx, int g,
                          unsigned int *oidx) {
  __shared__ float sv[256];
  __shared__ int si[256];
  float bv = -3.402823466e+38f;
  int bi = 0;
  for (int i = threadIdx.x; i < g; i += blockDim.x) {
    float v = pmax[i];
    int idx = pidx[i];
    if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
  }
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  am_block_reduce(sv, si);
  if (threadIdx.x == 0) oidx[0] = (unsigned int)si[0];
}
}
)CU";

namespace {
constexpr int ARGMAX_GRID = 256;   // pass-1 blocks (== pass-2 reduction width)
float *g_am_pmax = nullptr;        // [ARGMAX_GRID] per-block partial max
int *g_am_pidx = nullptr;          // [ARGMAX_GRID] per-block partial idx
unsigned int *g_am_oidx = nullptr; // [1] device final index
unsigned int *g_am_oidx_host =
  nullptr; // pinned host staging for the 4-byte D2H

// One-time allocation of the small fixed-size argmax scratch (partials + the
// 1-int device/host result). Capture-safe: a cudaMalloc inside stream capture
// invalidates the graph, so bail under capture (the buffers are tiny and are
// allocated on the first non-captured call -- the gating env makes this
// opt-in).
bool ensure_argmax_scratch() {
  if (g_am_pmax && g_am_pidx && g_am_oidx && g_am_oidx_host)
    return true;
  if (StreamManager::Global().isCapturing())
    return false;
  if (!g_am_pmax &&
      cudaMalloc(&g_am_pmax, sizeof(float) * ARGMAX_GRID) != cudaSuccess)
    return false;
  if (!g_am_pidx &&
      cudaMalloc(&g_am_pidx, sizeof(int) * ARGMAX_GRID) != cudaSuccess)
    return false;
  if (!g_am_oidx && cudaMalloc(&g_am_oidx, sizeof(unsigned int)) != cudaSuccess)
    return false;
  if (!g_am_oidx_host && cudaHostAlloc(&g_am_oidx_host, sizeof(unsigned int),
                                       cudaHostAllocDefault) != cudaSuccess)
    return false;
  return true;
}

// Run the two-pass reduction over a device-resident logits pointer (fp32 or
// fp16) and copy the 4-byte token id back to the host. Shared by both dtypes.
bool argmax_dispatch(const void *logits_dev, bool is_fp16, unsigned int vocab,
                     unsigned int *token_out_host) {
  if (logits_dev == nullptr || vocab == 0 || token_out_host == nullptr)
    return false;
  // Capture-safe scratch (no cudaMalloc under graph capture).
  if (!ensure_argmax_scratch())
    return false;

  auto kp1 = CudaContext::Global().registerCudaKernel(
    ARGMAX_SRC, is_fp16 ? "argmax_p1_f16" : "argmax_p1_f32");
  auto kp2 = CudaContext::Global().registerCudaKernel(ARGMAX_SRC, "argmax_p2");
  if (!kp1 || !kp2) {
    ml_loge("[CUDA] argmax: kernel registration failed");
    return false;
  }

  int n = (int)vocab, g = ARGMAX_GRID;
  kp1->SetKernelArguments(0, &logits_dev, sizeof(logits_dev));
  kp1->SetKernelArguments(1, &n, sizeof(n));
  kp1->SetKernelArguments(2, &g_am_pmax, sizeof(g_am_pmax));
  kp1->SetKernelArguments(3, &g_am_pidx, sizeof(g_am_pidx));
  const int b1[3] = {256, 1, 1};
  const int g1[3] = {ARGMAX_GRID, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kp1, g1, b1))
    return false;

  kp2->SetKernelArguments(0, &g_am_pmax, sizeof(g_am_pmax));
  kp2->SetKernelArguments(1, &g_am_pidx, sizeof(g_am_pidx));
  kp2->SetKernelArguments(2, &g, sizeof(g));
  kp2->SetKernelArguments(3, &g_am_oidx, sizeof(g_am_oidx));
  const int b2[3] = {256, 1, 1};
  const int g2[3] = {1, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kp2, g2, b2))
    return false;

  // Drain so the 4-byte D2H sees the final write, then copy the token id.
  StreamManager::Global().finish();
  if (cudaMemcpy(g_am_oidx_host, g_am_oidx, sizeof(unsigned int),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return false;
  *token_out_host = *g_am_oidx_host;
  return true;
}
} // namespace

bool cuda_argmax_fp32(const float *logits_dev, unsigned int vocab,
                      unsigned int *token_out_host) {
  return argmax_dispatch(logits_dev, /*is_fp16=*/false, vocab, token_out_host);
}

bool cuda_argmax_fp16(const unsigned short *logits_dev, unsigned int vocab,
                      unsigned int *token_out_host) {
  return argmax_dispatch(logits_dev, /*is_fp16=*/true, vocab, token_out_host);
}

template <typename K> static bool dispatch1d(K &kernel, unsigned int n) {
  const int block[3] = {256, 1, 1};
  const int grid[3] = {(int)((n + 255) / 256), 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

bool cuda_geglu_fp16(const unsigned short *gate, const unsigned short *up,
                     unsigned short *out, unsigned int n) {
  if (n == 0)
    return true;
  auto k = CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "geglu_fp16");
  if (!k) {
    ml_loge("[CUDA] geglu_fp16: registration failed");
    return false;
  }
  int ni = (int)n;
  k->SetKernelArguments(0, &gate, sizeof(gate));
  k->SetKernelArguments(1, &up, sizeof(up));
  k->SetKernelArguments(2, &out, sizeof(out));
  k->SetKernelArguments(3, &ni, sizeof(ni));
  return dispatch1d(k, n);
}

bool cuda_swiglu_fp16(const unsigned short *gate, const unsigned short *up,
                      unsigned short *out, unsigned int n) {
  if (n == 0)
    return true;
  auto k = CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "swiglu_fp16");
  if (!k) {
    ml_loge("[CUDA] swiglu_fp16: registration failed");
    return false;
  }
  int ni = (int)n;
  k->SetKernelArguments(0, &gate, sizeof(gate));
  k->SetKernelArguments(1, &up, sizeof(up));
  k->SetKernelArguments(2, &out, sizeof(out));
  k->SetKernelArguments(3, &ni, sizeof(ni));
  return dispatch1d(k, n);
}

bool cuda_add_fp16(const unsigned short *a, const unsigned short *b,
                   unsigned short *out, unsigned int n) {
  if (n == 0)
    return true;
  auto k = CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "add_fp16");
  if (!k) {
    ml_loge("[CUDA] add_fp16: registration failed");
    return false;
  }
  int ni = (int)n;
  k->SetKernelArguments(0, &a, sizeof(a));
  k->SetKernelArguments(1, &b, sizeof(b));
  k->SetKernelArguments(2, &out, sizeof(out));
  k->SetKernelArguments(3, &ni, sizeof(ni));
  return dispatch1d(k, n);
}

bool cuda_scalar_mul_fp16(const unsigned short *in, unsigned short *out,
                          unsigned int n, float scalar) {
  if (n == 0)
    return true;
  auto k =
    CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "scalar_mul_fp16");
  if (!k) {
    ml_loge("[CUDA] scalar_mul_fp16: registration failed");
    return false;
  }
  int ni = (int)n;
  k->SetKernelArguments(0, &in, sizeof(in));
  k->SetKernelArguments(1, &out, sizeof(out));
  k->SetKernelArguments(2, &ni, sizeof(ni));
  k->SetKernelArguments(3, &scalar, sizeof(scalar));
  return dispatch1d(k, n);
}

bool cuda_scalar_mul_fp16_slot(const unsigned short *in,
                               unsigned short *out_base, unsigned int n,
                               float scalar, int width, int ring_cap) {
  if (n == 0)
    return true;
  auto k = CudaContext::Global().registerCudaKernel(ELTWISE_SRC,
                                                    "scalar_mul_fp16_slot");
  if (!k) {
    ml_loge("[CUDA] scalar_mul_fp16_slot: registration failed");
    return false;
  }
  int ni = (int)n;
  const int *d_pos = cuda_pos_buffer();
  k->SetKernelArguments(0, &in, sizeof(in));
  k->SetKernelArguments(1, &out_base, sizeof(out_base));
  k->SetKernelArguments(2, &ni, sizeof(ni));
  k->SetKernelArguments(3, &scalar, sizeof(scalar));
  k->SetKernelArguments(4, &d_pos, sizeof(d_pos));
  k->SetKernelArguments(5, &width, sizeof(width));
  k->SetKernelArguments(6, &ring_cap, sizeof(ring_cap)); // [kv-window-ring]
  return dispatch1d(k, n);
}

bool cuda_softcap_fp16(const unsigned short *in, unsigned short *out,
                       unsigned int n, float cap) {
  if (n == 0)
    return true;
  auto k =
    CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "softcap_fp16");
  if (!k) {
    ml_loge("[CUDA] softcap_fp16: registration failed");
    return false;
  }
  int ni = (int)n;
  k->SetKernelArguments(0, &in, sizeof(in));
  k->SetKernelArguments(1, &out, sizeof(out));
  k->SetKernelArguments(2, &ni, sizeof(ni));
  k->SetKernelArguments(3, &cap, sizeof(cap));
  return dispatch1d(k, n);
}

bool cuda_slice_copy_fp16(const unsigned short *in, unsigned short *out,
                          unsigned int rows, unsigned int in_width,
                          unsigned int layer_off, unsigned int fs) {
  if (rows == 0 || fs == 0)
    return true;
  auto k =
    CudaContext::Global().registerCudaKernel(ELTWISE_SRC, "slice_copy_fp16");
  if (!k) {
    ml_loge("[CUDA] slice_copy_fp16: registration failed");
    return false;
  }
  int ri = (int)rows, iw = (int)in_width, lo = (int)layer_off, fsi = (int)fs;
  k->SetKernelArguments(0, &in, sizeof(in));
  k->SetKernelArguments(1, &out, sizeof(out));
  k->SetKernelArguments(2, &ri, sizeof(ri));
  k->SetKernelArguments(3, &iw, sizeof(iw));
  k->SetKernelArguments(4, &lo, sizeof(lo));
  k->SetKernelArguments(5, &fsi, sizeof(fsi));
  return dispatch1d(k, rows * fs);
}

} // namespace nntrainer::cuda

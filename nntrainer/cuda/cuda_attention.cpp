// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_attention.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA flash-style attention core (NVRTC).
 */

#include "cuda_attention.h"

#include <cuda_blas_manager.h>
#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_stream_manager.h>
#include <env_compat.h>

#include <cublas_v2.h>

#include <nntrainer_log.h>

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <mutex>
#include <unordered_map>

namespace nntrainer::cuda {

// One block per (query head h, query row i). Online (flash) softmax in FP32.
// Each thread owns the head dims d = tid, tid+B, ... (<=4 for head_dim<=512,
// B>=128). Shared = Q row [head_dim] + reduction scratch [B].
static const char *ATTN_CORE_SRC = R"CU(
extern "C" __global__ void attn_core(const float *Q, const float *K,
                                     const float *V, float *O, int num_heads,
                                     int num_kv_heads, int q_rows, int kv_len,
                                     int q_pos0, int head_dim, int window,
                                     float softcap) {
  int i = blockIdx.x;
  int h = blockIdx.y;
  if (i >= q_rows || h >= num_heads)
    return;
  int gqa = num_heads / num_kv_heads;
  int hkv = h / gqa;
  const float *Qhi = Q + ((long)h * q_rows + i) * head_dim;
  const float *Kh = K + (long)hkv * kv_len * head_dim;
  const float *Vh = V + (long)hkv * kv_len * head_dim;
  float *Ohi = O + ((long)h * q_rows + i) * head_dim;

  int tid = threadIdx.x;
  int B = blockDim.x;
  extern __shared__ float sh[];
  float *Qsh = sh;        // [head_dim]
  float *red = sh + head_dim; // [B]
  for (int d = tid; d < head_dim; d += B)
    Qsh[d] = Qhi[d];
  __syncthreads();

  const float scale = rsqrtf((float)head_dim);
  float acc[4];
#pragma unroll
  for (int r = 0; r < 4; r++)
    acc[r] = 0.f;

  int i_abs = q_pos0 + i;
  int j_lo = i_abs - window + 1;
  if (j_lo < 0)
    j_lo = 0;
  int j_hi = i_abs;
  if (j_hi >= kv_len)
    j_hi = kv_len - 1;

  float m = -1e30f, l = 0.f;
  for (int j = j_lo; j <= j_hi; ++j) {
    const float *Kj = Kh + (long)j * head_dim;
    float pd = 0.f;
    for (int d = tid; d < head_dim; d += B)
      pd += Qsh[d] * Kj[d];
    red[tid] = pd;
    __syncthreads();
    for (int s = B >> 1; s > 0; s >>= 1) {
      if (tid < s)
        red[tid] += red[tid + s];
      __syncthreads();
    }
    float score = red[0] * scale;
    __syncthreads();
    if (softcap > 0.f)
      score = softcap * tanhf(score / softcap);
    float m_new = fmaxf(m, score);
    float corr = __expf(m - m_new);
    float p = __expf(score - m_new);
    l = l * corr + p;
    m = m_new;
    const float *Vj = Vh + (long)j * head_dim;
    int r = 0;
    for (int d = tid; d < head_dim; d += B, ++r)
      acc[r] = acc[r] * corr + p * Vj[d];
  }

  float inv = l > 0.f ? 1.f / l : 0.f;
  int r = 0;
  for (int d = tid; d < head_dim; d += B, ++r)
    Ohi[d] = acc[r] * inv;
}
)CU";

// Interleaved fp16 variant: reads the head-interleaved fp16 query step + fp16
// KV cache directly (de-interleave + half->float inline), flash core in FP32,
// writes interleaved fp16 output. Drop-in for the host gemm_attention.
static const char *ATTN_IL_FP16_SRC = R"CU(
extern "C" {

__device__ __forceinline__ float a_h2f(unsigned short h) {
  // native fp16->fp32 (exact); replaces the software bit-twiddle -- this runs
  // per Q/K/V element in the hot loop
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short a_f2h(float f) {
  // native fp32->fp16 round-to-nearest-even (same rounding as the software
  // path -> bit-identical outputs)
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}

__global__ void attn_core_il_fp16(const unsigned short *q, const unsigned short *k,
                                  const unsigned short *v, unsigned short *o,
                                  int HQ, int HKV, int N_q, int N_kv,
                                  int cache_from, int d, int window, float softcap,
                                  const int *d_pos, int ring_cap) {
  int i = blockIdx.x, h = blockIdx.y;
  // M2-B: when d_pos is bound, read the live position/key-count from the device
  // buffer so a captured graph reads the new token's state on replay (else use
  // the baked int args = original non-graph behaviour).
  int cf = d_pos ? d_pos[0] : cache_from;
  int nkv = d_pos ? d_pos[1] : N_kv;
  if (i >= N_q || h >= HQ) return;
  int gqa = HQ / HKV, hkv = h / gqa;
  int HD_Q = HQ * d, HD_KV = HKV * d;
  const unsigned short *qrow = q + (long)i * HD_Q + (long)h * d;
  unsigned short *orow = o + (long)i * HD_Q + (long)h * d;
  int tid = threadIdx.x, B = blockDim.x;
  extern __shared__ float sh[];
  float *Qsh = sh; float *red = sh + d;
  for (int dd = tid; dd < d; dd += B) Qsh[dd] = a_h2f(qrow[dd]);
  __syncthreads();
  float scale = rsqrtf((float)d);
  float acc[4];
#pragma unroll
  for (int r=0;r<4;r++) acc[r]=0.f;
  int i_abs = cf + i;
  int j_lo = i_abs - window + 1; if (j_lo<0) j_lo=0;
  int j_hi = i_abs; if (j_hi>=nkv) j_hi=nkv-1;
  float mmax=-1e30f, l=0.f;
  for (int j=j_lo;j<=j_hi;++j) {
    // [kv-window-ring] physical cache row = j % ring_cap (ring_cap<=0: linear).
    long pj = (ring_cap > 0) ? (long)(j % ring_cap) : (long)j;
    const unsigned short *kr = k + pj*HD_KV + (long)hkv*d;
    float pd=0.f;
    for (int dd=tid;dd<d;dd+=B) pd += Qsh[dd]*a_h2f(kr[dd]);
    red[tid]=pd; __syncthreads();
    for (int s=B>>1;s>0;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
    float score=red[0]*scale; __syncthreads();
    if (softcap>0.f) score=softcap*tanhf(score/softcap);
    float mn=fmaxf(mmax,score), corr=__expf(mmax-mn), p=__expf(score-mn);
    l=l*corr+p; mmax=mn;
    const unsigned short *vr = v + pj*HD_KV + (long)hkv*d;
    int r=0; for (int dd=tid;dd<d;dd+=B,++r) acc[r]=acc[r]*corr + p*a_h2f(vr[dd]);
  }
  float inv = l>0.f?1.f/l:0.f;
  int r=0; for (int dd=tid;dd<d;dd+=B,++r) orow[dd]=a_f2h(acc[r]*inv);
}

}
)CU";

namespace {
// device mirror of a host-resident KV cache (keyed by host pointer). The cache
// (cache_key/cache_value) is a MAX_LIFESPAN tensor that is NOT UVM-resident on
// engine=cuda, so a device kernel can't read it directly; mirror it (small:
// num_kv_heads=1). Re-copied each call (the cache grows) -- correct, and cheap
// for the per-layer cache size.
struct DevKV {
  unsigned short *buf = nullptr;
  size_t cap = 0;
};
std::unordered_map<const void *, DevKV> g_kv_mirror;
std::mutex g_kv_mtx;

const unsigned short *mirror_kv(const unsigned short *host, size_t elems) {
  // Integrated GPU (Tegra/Jetson Orin): the iGPU reads ordinary host memory
  // directly (one shared physical pool), so NO device mirror is needed. Passing
  // through also avoids a latent stale-KV bug -- the mirror snapshots the host
  // cache, and a host-side V-copy that updates the slot AFTER the snapshot
  // would be served stale keys/values. Discrete GPUs keep the mirror.
  static const bool integrated = ContextManager::Global().isIntegrated();
  if (integrated)
    return host;
  const bool dev = nntrainer::cuda::dev_accessible(host);
  if (dev)
    return host; // already device-accessible
  std::lock_guard<std::mutex> lk(g_kv_mtx);
  auto &e = g_kv_mirror[host];
  size_t bytes = elems * sizeof(unsigned short);
  if (bytes > e.cap) {
    if (e.buf)
      cudaFree(e.buf);
    if (cudaMalloc(&e.buf, bytes) != cudaSuccess) {
      e.buf = nullptr;
      e.cap = 0;
      return nullptr;
    }
    e.cap = bytes;
  }
  cudaMemcpy(e.buf, host, bytes, cudaMemcpyHostToDevice);
  return e.buf;
}
} // namespace

// Flash-decoding (split-KV) for M=1 decode: the single-pass kernel launches
// only num_heads blocks (8 for gemma4) -- it underutilizes the SMs and
// serializes the long KV loop. Split the KV axis into chunks so
// num_heads*n_chunks blocks run a partial online-softmax in parallel, then a
// small reduce combines the chunks.
static const char *ATTN_SPLITKV_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float s_h2f(unsigned short h) {
  // native fp16->fp32 (exact); replaces the software bit-twiddle -- this runs
  // per Q/K/V element in the hot loop
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short s_f2h(float f) {
  // native fp32->fp16 round-to-nearest-even (same rounding as the software
  // path -> bit-identical outputs)
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
// One block per (head h, chunk c); query row is 0 (decode M=1). Online softmax
// over the chunk's keys; writes (m, l, acc[d]) to scratch[h*n_chunks + c].
__global__ void attn_partial(const unsigned short *q, const unsigned short *k,
                             const unsigned short *v, float *pm, float *pl,
                             float *pacc, int HQ, int HKV, int N_kv,
                             int cache_from, int d, int window, float softcap,
                             int chunk_kv, int n_chunks,
                             const int *d_pos, int max_n_chunks, int ring_cap) {
  int h = blockIdx.x, c = blockIdx.y;
  // M2-B: read the live query position / key-count from the device d_pos buffer
  // (mirrors attn_core_il_fp16) so ONE captured graph is valid for every token.
  // The grid is captured at gridDim.y=max_n_chunks; blocks past the live chunk
  // count early-return without writing, and the partial->reduce stride uses the
  // FIXED max_n_chunks so writer and reader always agree.
  int cf = d_pos ? d_pos[0] : cache_from;
  int nkv = d_pos ? d_pos[1] : N_kv;
  int live_nchunks = (nkv + chunk_kv - 1) / chunk_kv;
  if (c >= live_nchunks) return;
  int gqa = HQ / HKV, hkv = h / gqa;
  int HD_KV = HKV * d;
  const unsigned short *qrow = q + (long)h * d; // i=0
  int tid = threadIdx.x, B = blockDim.x;
  extern __shared__ float sh[];
  float *Qsh = sh; float *red = sh + d;
  for (int dd = tid; dd < d; dd += B) Qsh[dd] = s_h2f(qrow[dd]);
  __syncthreads();
  float scale = rsqrtf((float)d);
  int i_abs = cf; // i=0
  int j_lo_g = i_abs - window + 1; if (j_lo_g < 0) j_lo_g = 0;
  int j_hi_g = i_abs; if (j_hi_g >= nkv) j_hi_g = nkv - 1;
  int j_lo = c * chunk_kv; if (j_lo < j_lo_g) j_lo = j_lo_g;
  int j_hi = (c + 1) * chunk_kv - 1; if (j_hi > j_hi_g) j_hi = j_hi_g;
  float acc[4];
#pragma unroll
  for (int r = 0; r < 4; r++) acc[r] = 0.f;
  float mmax = -1e30f, l = 0.f;
  for (int j = j_lo; j <= j_hi; ++j) {
    // [kv-window-ring] physical cache row = j % ring_cap (ring_cap<=0: linear).
    long pj = (ring_cap > 0) ? (long)(j % ring_cap) : (long)j;
    const unsigned short *kr = k + pj * HD_KV + (long)hkv * d;
    float pd = 0.f;
    for (int dd = tid; dd < d; dd += B) pd += Qsh[dd] * s_h2f(kr[dd]);
    red[tid] = pd; __syncthreads();
    for (int s = B >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
    float score = red[0] * scale; __syncthreads();
    if (softcap > 0.f) score = softcap * tanhf(score / softcap);
    float mn = fmaxf(mmax, score), corr = __expf(mmax - mn), p = __expf(score - mn);
    l = l * corr + p; mmax = mn;
    const unsigned short *vr = v + pj * HD_KV + (long)hkv * d;
    int r = 0; for (int dd = tid; dd < d; dd += B, ++r) acc[r] = acc[r]*corr + p*s_h2f(vr[dd]);
  }
  if (j_lo > j_hi) { mmax = -1e30f; l = 0.f; }
  int base = h * max_n_chunks + c;
  if (tid == 0) { pm[base] = mmax; pl[base] = l; }
  int r = 0; for (int dd = tid; dd < d; dd += B, ++r) pacc[(long)base * d + dd] = acc[r];
}
// One block per head; combine the n_chunks partials into the fp16 output row.
__global__ void attn_reduce(const float *pm, const float *pl, const float *pacc,
                            unsigned short *o, int HQ, int d, int n_chunks,
                            const int *d_pos, int chunk_kv, int max_n_chunks) {
  int h = blockIdx.x;
  int tid = threadIdx.x, B = blockDim.x;
  // M2-B: live chunk count from d_pos; FIXED max_n_chunks stride to match the
  // partial writer (must be the SAME constant the scratch was sized with).
  int nkv = d_pos ? d_pos[1] : (n_chunks * chunk_kv);
  int live_nchunks = d_pos ? (nkv + chunk_kv - 1) / chunk_kv : n_chunks;
  int base = h * max_n_chunks;
  __shared__ float M, L;
  if (tid == 0) {
    float mx = -1e30f;
    for (int c = 0; c < live_nchunks; ++c) mx = fmaxf(mx, pm[base + c]);
    float l = 0.f;
    for (int c = 0; c < live_nchunks; ++c) l += pl[base + c] * __expf(pm[base + c] - mx);
    M = mx; L = l;
  }
  __syncthreads();
  float inv = L > 0.f ? 1.f / L : 0.f;
  unsigned short *orow = o + (long)h * d; // i=0
  for (int dd = tid; dd < d; dd += B) {
    float a = 0.f;
    for (int c = 0; c < live_nchunks; ++c)
      a += pacc[((long)(base + c)) * d + dd] * __expf(pm[base + c] - M);
    orow[dd] = s_f2h(a * inv);
  }
}
}
)CU";

// Block-Q multi-row prefill attention (CUDA mirror of the Intel OpenCL
// flash_attention_prefill_f16_blockq + FBQ_SG kernel). One WARP (32 lanes) owns
// a tile of TM query rows of one head; lane owns VPL = head_dim/32 CONTIGUOUS
// head dims so the K/V/Q loads are coalesced. Per key: the full d-dot is a
// single warp butterfly all-reduce (__shfl_xor, NO __syncthreads / shared mem),
// and K[n]/V[n] are loaded ONCE and reused across all TM rows (register
// online-softmax). Replaces attn_core_il_fp16's per-key 128-way LDS tree-reduce
// (7 __syncthreads x #keys, only 1 key in flight). Measured 3-4x faster on
// gemma4 shapes, fp16-identical output. GQA via hkv = head_q / (HQ/HKV).
static const char *ATTN_BLOCKQ_SRC = R"CU(
__device__ __forceinline__ float bq_h2f(unsigned short h) {
  // native fp16->fp32 (exact); replaces the software bit-twiddle -- this runs
  // per Q/K/V element in the hot loop
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short bq_f2h(float f) {
  // native fp32->fp16 round-to-nearest-even (same rounding as the software
  // path -> bit-identical outputs)
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
__device__ __forceinline__ float bq_wreduce(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}
template <int TM, int VPL>
__device__ __forceinline__ void
blockq_body(const unsigned short *q, const unsigned short *k,
            const unsigned short *v, unsigned short *o, int HQ, int HKV, int N_q,
            int N_kv, int cache_from, int d, int window, float softcap,
            int ring_cap) { // [kv-window-ring] >0: K/V physical row = n%ring_cap
  const int lane = threadIdx.x;             // 0..31
  const int grp = blockIdx.x;
  const int n_row_tiles = (N_q + TM - 1) / TM;
  const int head_q = grp / n_row_tiles;
  const int tile = grp % n_row_tiles;
  const int m0 = tile * TM;
  if (head_q >= HQ || m0 >= N_q) return;
  const int gqa = HQ / HKV, hkv = head_q / gqa;
  const int HD_Q = HQ * d, HD_KV = HKV * d;
  const float scale = rsqrtf((float)d);
  const int lane0 = lane * VPL;
  float q_reg[TM][VPL], acc_reg[TM][VPL], m_i[TM], l_i[TM];
  int valid[TM];
#pragma unroll
  for (int r = 0; r < TM; r++) {
    int m = m0 + r; valid[r] = (m < N_q) ? 1 : 0; m_i[r] = -1e30f; l_i[r] = 0.f;
    long q_base = (long)(valid[r] ? m : 0) * HD_Q + (long)head_q * d;
#pragma unroll
    for (int vv = 0; vv < VPL; vv++) {
      q_reg[r][vv] = bq_h2f(q[q_base + lane0 + vv]); acc_reg[r][vv] = 0.f;
    }
  }
  const int q_pos_off = cache_from;          // absolute query pos = m0+r+cache_from
  int last_row = ((m0 + TM - 1 < N_q) ? (m0 + TM - 1) : (N_q - 1)) + q_pos_off;
  int n_last = (N_kv - 1 < last_row) ? (N_kv - 1) : last_row;   // causal
  // Sliding-window tile-union lower bound (mirror of the CL blockq n_lo fix):
  // keys below row 0's window floor (n + window <= m0 + q_pos_off) are masked
  // for EVERY row in the tile (the floor only grows with r), so start the
  // walk there instead of paying K-load + dot + warp-reduce + V-load per
  // skipped key. Bit-identical by construction; without this the 16 sliding
  // prefill layers do O(M*N) key-visits instead of O(M*W).
  int n_lo = (window > 0) ? (m0 + q_pos_off - window + 1) : 0;
  if (n_lo < 0) n_lo = 0;
  for (int n = n_lo; n <= n_last; ++n) {
    // [kv-window-ring] physical cache row = n % ring_cap (ring_cap<=0: linear).
    long pn = (ring_cap > 0) ? (long)(n % ring_cap) : (long)n;
    long k_base = pn * HD_KV + (long)hkv * d;
    float k_reg[VPL];
#pragma unroll
    for (int vv = 0; vv < VPL; vv++) k_reg[vv] = bq_h2f(k[k_base + lane0 + vv]);
    float sdot[TM];
#pragma unroll
    for (int r = 0; r < TM; r++) {
      float p = 0.f;
#pragma unroll
      for (int vv = 0; vv < VPL; vv++) p += q_reg[r][vv] * k_reg[vv];
      sdot[r] = bq_wreduce(p);
    }
    long v_base = pn * HD_KV + (long)hkv * d;
    float v_reg[VPL];
#pragma unroll
    for (int vv = 0; vv < VPL; vv++) v_reg[vv] = bq_h2f(v[v_base + lane0 + vv]);
#pragma unroll
    for (int r = 0; r < TM; r++) {
      int m = m0 + r + q_pos_off;
      if (!valid[r] || n > m || (window > 0 && n + window <= m)) continue;
      float s = scale * sdot[r];
      if (softcap > 0.f) s = softcap * tanhf(s / softcap);
      float m_new = fmaxf(m_i[r], s), alpha = __expf(m_i[r] - m_new),
            pp = __expf(s - m_new);
#pragma unroll
      for (int vv = 0; vv < VPL; vv++)
        acc_reg[r][vv] = alpha * acc_reg[r][vv] + pp * v_reg[vv];
      l_i[r] = alpha * l_i[r] + pp; m_i[r] = m_new;
    }
  }
#pragma unroll
  for (int r = 0; r < TM; r++) {
    if (!valid[r]) continue;
    float inv = l_i[r] > 0.f ? 1.f / l_i[r] : 0.f;
    long o_base = (long)(m0 + r) * HD_Q + (long)head_q * d;
#pragma unroll
    for (int vv = 0; vv < VPL; vv++)
      o[o_base + lane0 + vv] = bq_f2h(acc_reg[r][vv] * inv);
  }
}
extern "C" __global__ void
attn_blockq_d256(const unsigned short *q, const unsigned short *k,
                 const unsigned short *v, unsigned short *o, int HQ, int HKV,
                 int N_q, int N_kv, int cache_from, int d, int window,
                 float softcap, int ring_cap) {
  blockq_body<4, 8>(q, k, v, o, HQ, HKV, N_q, N_kv, cache_from, d, window,
                    softcap, ring_cap);
}
extern "C" __global__ void
attn_blockq_d512(const unsigned short *q, const unsigned short *k,
                 const unsigned short *v, unsigned short *o, int HQ, int HKV,
                 int N_q, int N_kv, int cache_from, int d, int window,
                 float softcap, int ring_cap) {
  blockq_body<4, 16>(q, k, v, o, HQ, HKV, N_q, N_kv, cache_from, d, window,
                     softcap, ring_cap);
}
extern "C" __global__ void
attn_blockq_d128(const unsigned short *q, const unsigned short *k,
                 const unsigned short *v, unsigned short *o, int HQ, int HKV,
                 int N_q, int N_kv, int cache_from, int d, int window,
                 float softcap, int ring_cap) {
  blockq_body<4, 4>(q, k, v, o, HQ, HKV, N_q, N_kv, cache_from, d, window,
                    softcap, ring_cap);
}
)CU";

// Row-wise causal+window softmax over a per-head scores matrix [N_q, N_kv]
// (row-major: scores[i*N_kv + j] = dot(Q_i, K_j) already scaled). Masks
// j>i_abs (causal) and j<i_abs-window+1 (sliding) to 0, softmax in FP32 over
// the valid range, writes fp16 probabilities in place. One block per query row.
static const char *ATTN_SOFTMAX_SRC = R"CU(
extern "C" {
__device__ __forceinline__ float sm_h2f(unsigned short h) {
  unsigned int s=((unsigned int)(h&0x8000u))<<16, e=(h>>10)&0x1Fu, m=h&0x3FFu, o;
  if(e==0u){ if(m==0u)o=s; else{int x=-1;do{m<<=1;x++;}while((m&0x400u)==0u);m&=0x3FFu;
    o=s|((unsigned int)(127-15-x)<<23)|(m<<13);} }
  else if(e==0x1Fu)o=s|0x7F800000u|(m<<13);
  else o=s|((e+(127u-15u))<<23)|(m<<13);
  return __int_as_float((int)o);
}
__device__ __forceinline__ unsigned short sm_f2h(float f) {
  unsigned int x=(unsigned int)__float_as_int(f), s=(x>>16)&0x8000u, mant=x&0x7FFFFFu;
  int e=(int)((x>>23)&0xFFu);
  if(e==0xFF)return (unsigned short)(s|0x7C00u|(mant?0x200u:0u));
  int exp=e-127+15;
  if(exp>=0x1F)return (unsigned short)(s|0x7C00u);
  if(exp<=0){ if(exp<-10)return (unsigned short)s; mant|=0x800000u; int sh=14-exp;
    unsigned int hh=mant>>sh, rem=mant&((1u<<sh)-1u), half=1u<<(sh-1);
    if(rem>half||(rem==half&&(hh&1u)))hh++; return (unsigned short)(s|hh); }
  unsigned int hh=((unsigned int)exp<<10)|(mant>>13), rem=mant&0x1FFFu;
  if(rem>0x1000u||(rem==0x1000u&&(hh&1u)))hh++;
  return (unsigned short)(s|hh);
}
__global__ void attn_softmax_fp16(unsigned short *scores, int N_q, int N_kv,
                                  int cache_from, int window, float softcap) {
  int i = blockIdx.x;
  if (i >= N_q) return;
  unsigned short *row = scores + (long)i * N_kv;
  int i_abs = cache_from + i;
  int j_hi = i_abs < N_kv - 1 ? i_abs : N_kv - 1;
  int j_lo = (window > 0) ? i_abs - window + 1 : 0;
  if (j_lo < 0) j_lo = 0;
  int tid = threadIdx.x, B = blockDim.x;
  extern __shared__ float sh[];
  float lm = -1e30f;
  for (int j = j_lo + tid; j <= j_hi; j += B) {
    float v = sm_h2f(row[j]);
    if (softcap > 0.f) v = softcap * tanhf(v / softcap);
    lm = fmaxf(lm, v);
  }
  sh[tid] = lm; __syncthreads();
  for (int s = B >> 1; s > 0; s >>= 1) { if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]); __syncthreads(); }
  float mx = sh[0]; __syncthreads();
  float ls = 0.f;
  for (int j = j_lo + tid; j <= j_hi; j += B) {
    float v = sm_h2f(row[j]);
    if (softcap > 0.f) v = softcap * tanhf(v / softcap);
    ls += __expf(v - mx);
  }
  sh[tid] = ls; __syncthreads();
  for (int s = B >> 1; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
  float inv = sh[0] > 0.f ? 1.f / sh[0] : 0.f; __syncthreads();
  for (int j = tid; j < N_kv; j += B) {
    float p = 0.f;
    if (j >= j_lo && j <= j_hi) {
      float v = sm_h2f(row[j]);
      if (softcap > 0.f) v = softcap * tanhf(v / softcap);
      p = __expf(v - mx) * inv;
    }
    row[j] = sm_f2h(p);
  }
}
}
)CU";

namespace {
float *g_pm = nullptr, *g_pl = nullptr, *g_pacc = nullptr;
size_t g_pm_cap = 0, g_pacc_cap = 0;
std::mutex g_sk_mtx;
// M2-B: the FIXED chunk-count stride that the split-KV scratch
// (g_pm/g_pl/g_pacc) is pre-sized to at prewarm. The captured graph launches
// gridDim.y and strides partial<->reduce by THIS value (not the per-token live
// n_chunks) so one capture is valid for every token. 0 until prewarm runs (then
// M2-B is unavailable -> M1).
int g_sk_max_nchunks = 0;
bool ensure_sk(size_t mn, size_t acc) {
  if (mn > g_pm_cap) {
    // cudaMalloc/cudaFree inside a CUDA-graph stream capture invalidates the
    // capture. The decode split-KV scratch is pre-grown at load by
    // cuda_attention_splitkv_prewarm() so this branch must not run under
    // capture; if it ever would (an under-sized prewarm), bail so the caller
    // falls back rather than corrupting the graph.
    if (StreamManager::Global().isCapturing())
      return false;
    if (g_pm)
      cudaFree(g_pm);
    if (g_pl)
      cudaFree(g_pl);
    if (cudaMalloc(&g_pm, mn * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&g_pl, mn * sizeof(float)) != cudaSuccess)
      return false;
    g_pm_cap = mn;
  }
  if (acc > g_pacc_cap) {
    if (StreamManager::Global().isCapturing())
      return false;
    if (g_pacc)
      cudaFree(g_pacc);
    if (cudaMalloc(&g_pacc, acc * sizeof(float)) != cudaSuccess)
      return false;
    g_pacc_cap = acc;
  }
  return true;
}

bool attention_splitkv_decode(const unsigned short *q, const unsigned short *k,
                              const unsigned short *v, unsigned short *o,
                              int HQ, int HKV, int N_kv, int cache_from, int d,
                              int window, float softcap, int chunk_kv,
                              int ring_cap) {
  const int n_chunks = (N_kv + chunk_kv - 1) / chunk_kv;
  // M2-B: when the device pos buffer is bound, the captured graph uses the
  // FIXED max-chunk stride/grid (g_sk_max_nchunks, published at prewarm) so one
  // capture serves every token. M1/non-graph (dpos=nullptr) uses the live
  // n_chunks -> bit-identical to the original. Mirror the dense path's m2b gate
  // (line ~839).
  static const bool m2b = nntr_env_on("NNTR_CUDA_M2B");
  const int *dpos = (m2b && g_sk_max_nchunks > 0) ? cuda_pos_buffer() : nullptr;
  const int max_nc = dpos ? g_sk_max_nchunks : n_chunks;
  std::lock_guard<std::mutex> lk(g_sk_mtx);
  if (!ensure_sk((size_t)HQ * max_nc, (size_t)HQ * max_nc * d))
    return false;
  auto kp =
    CudaContext::Global().registerCudaKernel(ATTN_SPLITKV_SRC, "attn_partial");
  auto kr =
    CudaContext::Global().registerCudaKernel(ATTN_SPLITKV_SRC, "attn_reduce");
  if (!kp || !kr)
    return false;
  const int B = 128;
  kp->SetKernelArguments(0, &q, sizeof(q));
  kp->SetKernelArguments(1, &k, sizeof(k));
  kp->SetKernelArguments(2, &v, sizeof(v));
  kp->SetKernelArguments(3, &g_pm, sizeof(g_pm));
  kp->SetKernelArguments(4, &g_pl, sizeof(g_pl));
  kp->SetKernelArguments(5, &g_pacc, sizeof(g_pacc));
  kp->SetKernelArguments(6, &HQ, sizeof(HQ));
  kp->SetKernelArguments(7, &HKV, sizeof(HKV));
  kp->SetKernelArguments(8, &N_kv, sizeof(N_kv));
  kp->SetKernelArguments(9, &cache_from, sizeof(cache_from));
  kp->SetKernelArguments(10, &d, sizeof(d));
  kp->SetKernelArguments(11, &window, sizeof(window));
  kp->SetKernelArguments(12, &softcap, sizeof(softcap));
  kp->SetKernelArguments(13, &chunk_kv, sizeof(chunk_kv));
  kp->SetKernelArguments(14, &n_chunks, sizeof(n_chunks));
  kp->SetKernelArguments(15, &dpos, sizeof(dpos));
  kp->SetKernelArguments(16, &max_nc, sizeof(max_nc));
  kp->SetKernelArguments(17, &ring_cap, sizeof(ring_cap));
  const int pg[3] = {HQ, max_nc, 1};
  const int pb[3] = {B, 1, 1};
  const unsigned int shmem = (unsigned int)(sizeof(float) * ((size_t)d + B));
  if (!StreamManager::Global().DispatchCommand(*kp, pg, pb, shmem))
    return false;
  kr->SetKernelArguments(0, &g_pm, sizeof(g_pm));
  kr->SetKernelArguments(1, &g_pl, sizeof(g_pl));
  kr->SetKernelArguments(2, &g_pacc, sizeof(g_pacc));
  kr->SetKernelArguments(3, &o, sizeof(o));
  kr->SetKernelArguments(4, &HQ, sizeof(HQ));
  kr->SetKernelArguments(5, &d, sizeof(d));
  kr->SetKernelArguments(6, &n_chunks, sizeof(n_chunks));
  kr->SetKernelArguments(7, &dpos, sizeof(dpos));
  kr->SetKernelArguments(8, &chunk_kv, sizeof(chunk_kv));
  kr->SetKernelArguments(9, &max_nc, sizeof(max_nc));
  const int rg[3] = {HQ, 1, 1};
  const int rb[3] = {B, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kr, rg, rb))
    return false;
  return true;
}

// GEMM-based multi-row prefill attention. The per-key flash kernel
// (attn_core_il) is fetch/sync-bound (~0.4% of peak on d=128) because it
// serial- reduces every key; for prefill (N_q>1) materialising scores via
// cuBLAS fp16 GEMMs (QK^T -> softmax -> PV) is far faster and
// head_dim-agnostic, so it also helps the head_dim=128 (qwen3/llama) case that
// block-Q does not. Layout (interleaved fp16, column-major cuBLAS): per
// query-head h (kv-head hkv=h/gqa) scores_cm[N_kv,N_q] = K_h^T@Q_h reads back
// as row-major scores[N_q,N_kv]; then O_cm[d,N_q] = V_h@scores_cm.
float *g_scores = nullptr;
size_t g_scores_cap = 0;
std::mutex g_ga_mtx;
bool attention_gemm_prefill_fp16(const unsigned short *q,
                                 const unsigned short *k,
                                 const unsigned short *v, unsigned short *o,
                                 int HQ, int HKV, int N_q, int N_kv,
                                 int cache_from, int d, int window,
                                 float softcap) {
  std::lock_guard<std::mutex> lk(g_ga_mtx);
  cublasHandle_t bh = BlasManager::Global().handle();
  if (!bh)
    return false;
  auto sm = CudaContext::Global().registerCudaKernel(ATTN_SOFTMAX_SRC,
                                                     "attn_softmax_fp16");
  if (!sm)
    return false;
  // scratch scores [N_q, N_kv] fp16 reused across heads (heads run serially).
  const size_t need = (size_t)N_q * N_kv;
  if (need > g_scores_cap) {
    if (g_scores)
      cudaFree(g_scores);
    if (cudaMalloc(&g_scores, need * sizeof(unsigned short)) != cudaSuccess) {
      g_scores = nullptr;
      g_scores_cap = 0;
      return false;
    }
    g_scores_cap = need;
  }
  auto *scores = reinterpret_cast<unsigned short *>(g_scores);
  const int gqa = HQ / HKV;
  const int HD_Q = HQ * d, HD_KV = HKV * d;
  const float scale = 1.0f / sqrtf((float)d);
  const float one = 1.0f, zero = 0.0f;
  const int win = (window <= 0 || window >= N_kv) ? 0 : window;
  const int B = 256;
  const size_t shmem = (size_t)B * sizeof(float);
  for (int h = 0; h < HQ; ++h) {
    const int hkv = h / gqa;
    const unsigned short *Qh = q + (long)h * d;   // [N_q,d] ld=HD_Q
    const unsigned short *Kh = k + (long)hkv * d; // [N_kv,d] ld=HD_KV
    const unsigned short *Vh = v + (long)hkv * d; // [N_kv,d] ld=HD_KV
    unsigned short *Oh = o + (long)h * d;         // [N_q,d] ld=HD_Q
    // scores_cm[N_kv,N_q] = (K_h^T)[N_kv,d] @ Q_h[d,N_q] * scale -> row-major
    // scores[i*N_kv+j] = scale*dot(Q_i,K_j).
    cublasStatus_t s1 =
      cublasGemmEx(bh, CUBLAS_OP_T, CUBLAS_OP_N, N_kv, N_q, d, &scale, Kh,
                   CUDA_R_16F, HD_KV, Qh, CUDA_R_16F, HD_Q, &zero, scores,
                   CUDA_R_16F, N_kv, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s1 != CUBLAS_STATUS_SUCCESS)
      return false;
    sm->SetKernelArguments(0, &scores, sizeof(scores));
    sm->SetKernelArguments(1, &N_q, sizeof(N_q));
    sm->SetKernelArguments(2, &N_kv, sizeof(N_kv));
    sm->SetKernelArguments(3, &cache_from, sizeof(cache_from));
    sm->SetKernelArguments(4, &win, sizeof(win));
    sm->SetKernelArguments(5, &softcap, sizeof(softcap));
    const int sg[3] = {N_q, 1, 1};
    const int sb[3] = {B, 1, 1};
    if (!StreamManager::Global().DispatchCommand(*sm, sg, sb, shmem))
      return false;
    // O_cm[d,N_q] = V_h[d,N_kv] @ scores_cm[N_kv,N_q] -> row-major O[i,e].
    cublasStatus_t s2 =
      cublasGemmEx(bh, CUBLAS_OP_N, CUBLAS_OP_N, d, N_q, N_kv, &one, Vh,
                   CUDA_R_16F, HD_KV, scores, CUDA_R_16F, N_kv, &zero, Oh,
                   CUDA_R_16F, HD_Q, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (s2 != CUBLAS_STATUS_SUCCESS)
      return false;
  }
  return true;
}
} // namespace

// Pre-grow the split-KV decode scratch (g_pm/g_pl/g_pacc) to the model's max
// decode capacity at load. The M=1 split-KV path is only reached under graph
// capture once NNTR_CUDA_GRAPH is on; a cudaMalloc/Free inside
// cudaStreamBeginCapture..EndCapture invalidates the capture. Warming here
// (before any capture) makes every captured ensure_sk a pure cap-hit, so the
// fast flash-decode path stays usable under the graph. Idempotent (cap check).
bool cuda_attention_splitkv_prewarm(int max_seq_len, int max_hq,
                                    int max_head_dim) {
  const char *e = std::getenv("NNTR_CUDA_FLASH_DECODE");
  if (!e)
    return true; // split-KV off -> no scratch needed (mirror of interleaved)
  int chunk = atoi(e);
  if (chunk <= 0)
    chunk = 64;
  if (max_seq_len <= 0 || max_hq <= 0 || max_head_dim <= 0)
    return true;
  const int max_nchunks = (max_seq_len + chunk - 1) / chunk;
  const size_t mn = (size_t)max_hq * (size_t)max_nchunks;
  std::lock_guard<std::mutex> lk(g_sk_mtx);
  if (!ensure_sk(mn, mn * (size_t)max_head_dim))
    return false;
  // Publish the fixed stride so attention_splitkv_decode + the kernels all use
  // the SAME value the scratch was just sized with (M2-B capture correctness).
  g_sk_max_nchunks = max_nchunks;
  return true;
}

// =============================================================================
// [cuda-kvi8] int8 KV-cache path (NNTR_KV_INT8 on engine=cuda).
// Mirrors the OpenCL kvi8 design: cache format is the CPU-verified concat
// [row][HKV*d] signed int8 (K roped BEFORE quantization, symmetric amax/127)
// with per-(row,head_kv) fp16 scales at scale[row*HKV + h]; rows are
// ring-indexed when ring_cap > 0 (scales in lockstep). Both scales fold OUT
// of the inner loops: the K scale multiplies the finished dot, the V scale
// pre-multiplies p (decode) / v_reg (prefill).
// =============================================================================
static const char *ATTN_I8_SRC = R"CU(
__device__ __forceinline__ float i8_h2f(unsigned short h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ unsigned short i8_f2h(float f) {
  unsigned short h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}
__device__ __forceinline__ float i8_wreduce(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}
__device__ __forceinline__ float i8_wreduce_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}
// dp4a: 4-way int8 dot with int32 accumulate. Inline PTX like the cvt
// helpers above -- NVRTC needs no header for it. Below sm_61 (no dp4a
// instruction; cubin compile validates the asm against the real arch) fall
// back to exact byte math so the shared ATTN_I8_SRC module still compiles
// and the _dp4a kernels stay correct everywhere.
__device__ __forceinline__ int i8_dp4a(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
  int r;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
#else
  int r = c;
#pragma unroll
  for (int bb = 0; bb < 4; bb++)
    r += (int)((signed char)(a >> (8 * bb))) * (int)((signed char)(b >> (8 * bb)));
  return r;
#endif
}
// Quantize one fp16 step row (token s, head_kv h) into the int8 cache +
// fp16 amax/127 scale, at the ring-aware physical row. Matches the host
// quantize_kv_fp16_to_int8_per_row exactly: symmetric, round-half-away-
// from-zero (roundf), clamp [-127,127], scale = amax/127 (0 when amax==0).
extern "C" __global__ void kv_quant_i8(const unsigned short *src, signed char *dst,
                            unsigned short *scale, int step_n, int HKV, int d,
                            int cache_from, int ring_cap,
                            const int *d_pos) {
  int s = blockIdx.x, h = blockIdx.y;
  if (s >= step_n || h >= HKV) return;
  // [d_pos] M2-B graph replay: read the live write position from the device
  // pos buffer so ONE captured decode graph serves every token.
  int cf = d_pos ? d_pos[0] : cache_from;
  const unsigned short *row = src + ((long)s * HKV + h) * d;
  long abs_row = (long)cf + s;
  long prow = (ring_cap > 0) ? (abs_row % ring_cap) : abs_row;
  signed char *drow = dst + (prow * HKV + h) * (long)d;
  int tid = threadIdx.x, B = blockDim.x;
  __shared__ float red[256];
  float am = 0.f;
  for (int x = tid; x < d; x += B) {
    float v = fabsf(i8_h2f(row[x]));
    am = fmaxf(am, v);
  }
  red[tid] = am;
  __syncthreads();
  for (int st = B >> 1; st > 0; st >>= 1) {
    if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
    __syncthreads();
  }
  float amax = red[0];
  float inv = amax > 0.f ? 127.f / amax : 0.f;
  for (int x = tid; x < d; x += B) {
    int qv = (int)roundf(i8_h2f(row[x]) * inv);
    qv = max(-127, min(127, qv));
    drow[x] = (signed char)qv;
  }
  if (tid == 0)
    scale[prow * HKV + h] = i8_f2h(amax / 127.f);
}
// int8 block-Q prefill: identical structure to blockq_body (warp per
// (head, TM=4 row tile), warp-shuffle d-dot, ring modulo, n_lo window skip)
// with int8 K/V loads and the scales folded outside the dot loops.
// DP4A=1 additionally quantizes Q per (row, head) with the same symmetric
// amax/127 convention and runs the QK dot as packed int8 dp4a words; the
// int32 lane partials stay exact through the float shuffle-reduce
// (|dot| <= 127*127*512 < 2^24). The Q scale folds onto the finished score
// next to the K scale, so softmax/PV are untouched.
template <int TM, int VPL, int DP4A>
__device__ __forceinline__ void
blockq_body_i8(const unsigned short *q, const signed char *k,
               const signed char *v, const unsigned short *kscale,
               const unsigned short *vscale, unsigned short *o, int HQ,
               int HKV, int N_q, int N_kv, int cache_from, int d, int window,
               float softcap, int ring_cap) {
  const int lane = threadIdx.x;
  const int grp = blockIdx.x;
  const int n_row_tiles = (N_q + TM - 1) / TM;
  const int head_q = grp / n_row_tiles;
  const int tile = grp % n_row_tiles;
  const int m0 = tile * TM;
  if (head_q >= HQ || m0 >= N_q) return;
  const int gqa = HQ / HKV, hkv = head_q / gqa;
  const int HD_Q = HQ * d, HD_KV = HKV * d;
  const float scale = rsqrtf((float)d);
  const int lane0 = lane * VPL;
  float q_reg[TM][VPL], acc_reg[TM][VPL], m_i[TM], l_i[TM];
  int valid[TM];
#pragma unroll
  for (int r = 0; r < TM; r++) {
    int m = m0 + r; valid[r] = (m < N_q) ? 1 : 0; m_i[r] = -1e30f; l_i[r] = 0.f;
    long q_base = (long)(valid[r] ? m : 0) * HD_Q + (long)head_q * d;
#pragma unroll
    for (int vv = 0; vv < VPL; vv++) {
      q_reg[r][vv] = i8_h2f(q[q_base + lane0 + vv]); acc_reg[r][vv] = 0.f;
    }
  }
  // [dp4a] per-(row,head) symmetric Q quantization: amax over the full d via
  // warp max-reduce, then each lane packs its VPL int8 into VPL/4 dp4a words.
  float q_scl[TM];
  int q_pk[TM][VPL / 4];
  if (DP4A) {
#pragma unroll
    for (int r = 0; r < TM; r++) {
      float am = 0.f;
#pragma unroll
      for (int vv = 0; vv < VPL; vv++) am = fmaxf(am, fabsf(q_reg[r][vv]));
      am = i8_wreduce_max(am);
      float inv = am > 0.f ? 127.f / am : 0.f;
      q_scl[r] = am / 127.f;
#pragma unroll
      for (int p = 0; p < VPL / 4; p++) {
        unsigned int w = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) {
          int qv = (int)roundf(q_reg[r][p * 4 + b] * inv);
          qv = max(-127, min(127, qv));
          w |= ((unsigned int)(qv & 0xff)) << (8 * b);
        }
        q_pk[r][p] = (int)w;
      }
    }
  }
  const int q_pos_off = cache_from;
  int last_row = ((m0 + TM - 1 < N_q) ? (m0 + TM - 1) : (N_q - 1)) + q_pos_off;
  int n_last = (N_kv - 1 < last_row) ? (N_kv - 1) : last_row;
  int n_lo = (window > 0) ? (m0 + q_pos_off - window + 1) : 0;
  if (n_lo < 0) n_lo = 0;
  for (int n = n_lo; n <= n_last; ++n) {
    long pn = (ring_cap > 0) ? (long)(n % ring_cap) : (long)n;
    long kv_base = pn * HD_KV + (long)hkv * d;
    // per-key scales (ring-lockstep row index)
    const float ks = scale * i8_h2f(kscale[pn * HKV + hkv]);
    const float vs = i8_h2f(vscale[pn * HKV + hkv]);
    float sdot[TM];
    if (DP4A) {
      // K lane slice as packed dp4a words. kv_base/lane0 are multiples of 4;
      // the K/V BASE pointers are verified 4-aligned by the host dispatch
      // (the tensor pool packs by byte size with no alignment guarantee).
      const int *kp = (const int *)(k + kv_base + lane0);
      int k_pk[VPL / 4];
#pragma unroll
      for (int p = 0; p < VPL / 4; p++) k_pk[p] = kp[p];
#pragma unroll
      for (int r = 0; r < TM; r++) {
        int di = 0;
#pragma unroll
        for (int p = 0; p < VPL / 4; p++) di = i8_dp4a(q_pk[r][p], k_pk[p], di);
        sdot[r] = i8_wreduce((float)di);
      }
    } else {
      // byte loads: ld.global.s8 has no alignment requirement, and the pool
      // gives the cache base none -- keep the always-on path unconditional.
      float k_reg[VPL];
#pragma unroll
      for (int vv = 0; vv < VPL; vv++)
        k_reg[vv] = (float)k[kv_base + lane0 + vv];
#pragma unroll
      for (int r = 0; r < TM; r++) {
        float p = 0.f;
#pragma unroll
        for (int vv = 0; vv < VPL; vv++) p += q_reg[r][vv] * k_reg[vv];
        sdot[r] = i8_wreduce(p);
      }
    }
    float v_reg[VPL];
    if (DP4A) {
      // packed words: base alignment verified by the host dispatch.
      const int *vp = (const int *)(v + kv_base + lane0);
#pragma unroll
      for (int p2 = 0; p2 < VPL / 4; p2++) {
        int w = vp[p2];
#pragma unroll
        for (int b = 0; b < 4; b++)
          v_reg[p2 * 4 + b] = vs * (float)((signed char)(w >> (8 * b)));
      }
    } else {
#pragma unroll
      for (int vv = 0; vv < VPL; vv++)
        v_reg[vv] = vs * (float)v[kv_base + lane0 + vv];
    }
#pragma unroll
    for (int r = 0; r < TM; r++) {
      int m = m0 + r + q_pos_off;
      if (!valid[r] || n > m || (window > 0 && n + window <= m)) continue;
      float s = ks * sdot[r];
      if (DP4A) s *= q_scl[r];
      if (softcap > 0.f) s = softcap * tanhf(s / softcap);
      float m_new = fmaxf(m_i[r], s), alpha = __expf(m_i[r] - m_new),
            pp = __expf(s - m_new);
#pragma unroll
      for (int vv = 0; vv < VPL; vv++)
        acc_reg[r][vv] = alpha * acc_reg[r][vv] + pp * v_reg[vv];
      l_i[r] = alpha * l_i[r] + pp; m_i[r] = m_new;
    }
  }
#pragma unroll
  for (int r = 0; r < TM; r++) {
    if (!valid[r]) continue;
    float inv = l_i[r] > 0.f ? 1.f / l_i[r] : 0.f;
    long o_base = (long)(m0 + r) * HD_Q + (long)head_q * d;
#pragma unroll
    for (int vv = 0; vv < VPL; vv++)
      o[o_base + lane0 + vv] = i8_f2h(acc_reg[r][vv] * inv);
  }
}
extern "C" __global__ void attn_blockq_i8_d128(const unsigned short *q,
    const signed char *k, const signed char *v, const unsigned short *kscale,
    const unsigned short *vscale, unsigned short *o, int HQ, int HKV, int N_q,
    int N_kv, int cache_from, int d, int window, float softcap, int ring_cap) {
  blockq_body_i8<4, 4, 0>(q, k, v, kscale, vscale, o, HQ, HKV, N_q, N_kv,
                          cache_from, d, window, softcap, ring_cap);
}
extern "C" __global__ void attn_blockq_i8_d256(const unsigned short *q,
    const signed char *k, const signed char *v, const unsigned short *kscale,
    const unsigned short *vscale, unsigned short *o, int HQ, int HKV, int N_q,
    int N_kv, int cache_from, int d, int window, float softcap, int ring_cap) {
  blockq_body_i8<4, 8, 0>(q, k, v, kscale, vscale, o, HQ, HKV, N_q, N_kv,
                          cache_from, d, window, softcap, ring_cap);
}
extern "C" __global__ void attn_blockq_i8_d512(const unsigned short *q,
    const signed char *k, const signed char *v, const unsigned short *kscale,
    const unsigned short *vscale, unsigned short *o, int HQ, int HKV, int N_q,
    int N_kv, int cache_from, int d, int window, float softcap, int ring_cap) {
  blockq_body_i8<4, 16, 0>(q, k, v, kscale, vscale, o, HQ, HKV, N_q, N_kv,
                           cache_from, d, window, softcap, ring_cap);
}
// dp4a variants (NNTR_CUDA_KVI8_DP4A=1): int8 arithmetic QK^T, same layout.
extern "C" __global__ void attn_blockq_i8_d128_dp4a(const unsigned short *q,
    const signed char *k, const signed char *v, const unsigned short *kscale,
    const unsigned short *vscale, unsigned short *o, int HQ, int HKV, int N_q,
    int N_kv, int cache_from, int d, int window, float softcap, int ring_cap) {
  blockq_body_i8<4, 4, 1>(q, k, v, kscale, vscale, o, HQ, HKV, N_q, N_kv,
                          cache_from, d, window, softcap, ring_cap);
}
extern "C" __global__ void attn_blockq_i8_d256_dp4a(const unsigned short *q,
    const signed char *k, const signed char *v, const unsigned short *kscale,
    const unsigned short *vscale, unsigned short *o, int HQ, int HKV, int N_q,
    int N_kv, int cache_from, int d, int window, float softcap, int ring_cap) {
  blockq_body_i8<4, 8, 1>(q, k, v, kscale, vscale, o, HQ, HKV, N_q, N_kv,
                          cache_from, d, window, softcap, ring_cap);
}
extern "C" __global__ void attn_blockq_i8_d512_dp4a(const unsigned short *q,
    const signed char *k, const signed char *v, const unsigned short *kscale,
    const unsigned short *vscale, unsigned short *o, int HQ, int HKV, int N_q,
    int N_kv, int cache_from, int d, int window, float softcap, int ring_cap) {
  blockq_body_i8<4, 16, 1>(q, k, v, kscale, vscale, o, HQ, HKV, N_q, N_kv,
                           cache_from, d, window, softcap, ring_cap);
}
// int8 split-KV decode partial: identical to attn_partial with int8 K/V and
// the scales folded (K onto the score, V onto p). Reduce stage is the shared
// fp32 attn_reduce from ATTN_SPLITKV_SRC. d_pos/M2-B is supported since the
// d_pos indirection (57e8a7b82); see the [d_pos] note inside attn_partial_i8.
extern "C" __global__ void attn_partial_i8(const unsigned short *q, const signed char *k,
                                const signed char *v, float *pm, float *pl,
                                float *pacc, int HQ, int HKV, int N_kv,
                                int cache_from, int d, int window,
                                float softcap, int chunk_kv, int n_chunks,
                                int ring_cap, const unsigned short *kscale,
                                const unsigned short *vscale,
                                const int *d_pos, int max_n_chunks) {
  int h = blockIdx.x, c = blockIdx.y;
  // [d_pos] M2-B: live position/key-count from the device buffer; the grid is
  // captured at gridDim.y = max_n_chunks, blocks past the live chunk count
  // early-return, and the partial stride uses the FIXED max_n_chunks so the
  // writer and attn_reduce always agree (mirrors the fp16 attn_partial).
  int cf = d_pos ? d_pos[0] : cache_from;
  int nkv = d_pos ? d_pos[1] : N_kv;
  int live_nchunks = (nkv + chunk_kv - 1) / chunk_kv;
  if (c >= live_nchunks) return;
  int gqa = HQ / HKV, hkv = h / gqa;
  int HD_KV = HKV * d;
  const unsigned short *qrow = q + (long)h * d;
  int tid = threadIdx.x, B = blockDim.x;
  extern __shared__ float sh[];
  float *Qsh = sh; float *red = sh + d;
  for (int dd = tid; dd < d; dd += B) Qsh[dd] = i8_h2f(qrow[dd]);
  __syncthreads();
  float scale = rsqrtf((float)d);
  int i_abs = cf;
  int j_lo_g = i_abs - window + 1; if (j_lo_g < 0) j_lo_g = 0;
  int j_hi_g = i_abs; if (j_hi_g >= nkv) j_hi_g = nkv - 1;
  int j_lo = c * chunk_kv; if (j_lo < j_lo_g) j_lo = j_lo_g;
  int j_hi = (c + 1) * chunk_kv - 1; if (j_hi > j_hi_g) j_hi = j_hi_g;
  float acc[4];
#pragma unroll
  for (int r = 0; r < 4; r++) acc[r] = 0.f;
  float mmax = -1e30f, l = 0.f;
  for (int j = j_lo; j <= j_hi; ++j) {
    long pj = (ring_cap > 0) ? (long)(j % ring_cap) : (long)j;
    const signed char *kr = k + pj * HD_KV + (long)hkv * d;
    const float ksj = scale * i8_h2f(kscale[pj * HKV + hkv]);
    const float vsj = i8_h2f(vscale[pj * HKV + hkv]);
    float pd = 0.f;
    for (int dd = tid; dd < d; dd += B) pd += Qsh[dd] * (float)kr[dd];
    red[tid] = pd; __syncthreads();
    for (int s = B >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid+s]; __syncthreads(); }
    float score = red[0] * ksj; __syncthreads();
    if (softcap > 0.f) score = softcap * tanhf(score / softcap);
    float mn = fmaxf(mmax, score), corr = __expf(mmax - mn), p = __expf(score - mn);
    l = l * corr + p; mmax = mn;
    const signed char *vr = v + pj * HD_KV + (long)hkv * d;
    float pv = p * vsj;
    int r = 0; for (int dd = tid; dd < d; dd += B, ++r) acc[r] = acc[r]*corr + pv*(float)vr[dd];
  }
  if (j_lo > j_hi) { mmax = -1e30f; l = 0.f; }
  int base = h * max_n_chunks + c;
  if (tid == 0) { pm[base] = mmax; pl[base] = l; }
  int r = 0; for (int dd = tid; dd < d; dd += B, ++r) pacc[(long)base * d + dd] = acc[r];
}
)CU";

bool cuda_kv_int8_quantize(const unsigned short *src_fp16, signed char *dst_i8,
                           unsigned short *scale_fp16, int step_n, int HKV,
                           int d, int cache_from, int ring_cap) {
  if (step_n <= 0 || HKV <= 0 || d <= 0 || d > 4096)
    return false;
  auto kq =
    CudaContext::Global().registerCudaKernel(ATTN_I8_SRC, "kv_quant_i8");
  if (!kq)
    return false;
  const int B = 128;
  kq->SetKernelArguments(0, &src_fp16, sizeof(src_fp16));
  kq->SetKernelArguments(1, &dst_i8, sizeof(dst_i8));
  kq->SetKernelArguments(2, &scale_fp16, sizeof(scale_fp16));
  kq->SetKernelArguments(3, &step_n, sizeof(step_n));
  kq->SetKernelArguments(4, &HKV, sizeof(HKV));
  kq->SetKernelArguments(5, &d, sizeof(d));
  kq->SetKernelArguments(6, &cache_from, sizeof(cache_from));
  kq->SetKernelArguments(7, &ring_cap, sizeof(ring_cap));
  // decode + M2-B: live write position via the device pos buffer (prefill
  // stays baked -- prefill is never graph-captured).
  static const bool _m2b_q = nntr_env_on("NNTR_CUDA_M2B");
  const int *dpos_q = (_m2b_q && step_n == 1) ? cuda_pos_buffer() : nullptr;
  kq->SetKernelArguments(8, &dpos_q, sizeof(dpos_q));
  const int grid[3] = {step_n, HKV, 1};
  const int block[3] = {B, 1, 1};
  return StreamManager::Global().DispatchCommand(*kq, grid, block, 0);
}

static bool attention_splitkv_decode_i8(
  const unsigned short *q, const signed char *k, const signed char *v,
  const unsigned short *kscale, const unsigned short *vscale, unsigned short *o,
  int HQ, int HKV, int N_kv, int cache_from, int d, int window, float softcap,
  int chunk_kv, int ring_cap) {
  const int n_chunks = (N_kv + chunk_kv - 1) / chunk_kv;
  // M2-B: fixed max-chunk grid/stride from the prewarm-published
  // g_sk_max_nchunks so one captured graph serves every token (fp16 idiom).
  static const bool _m2b_i8 = nntr_env_on("NNTR_CUDA_M2B");
  const int *dpos =
    (_m2b_i8 && g_sk_max_nchunks > 0) ? cuda_pos_buffer() : nullptr;
  const int max_nc = dpos ? g_sk_max_nchunks : n_chunks;
  std::lock_guard<std::mutex> lk(g_sk_mtx);
  if (!ensure_sk((size_t)HQ * max_nc, (size_t)HQ * max_nc * d))
    return false;
  auto kp =
    CudaContext::Global().registerCudaKernel(ATTN_I8_SRC, "attn_partial_i8");
  auto kr =
    CudaContext::Global().registerCudaKernel(ATTN_SPLITKV_SRC, "attn_reduce");
  if (!kp || !kr)
    return false;
  const int B = 128;
  kp->SetKernelArguments(0, &q, sizeof(q));
  kp->SetKernelArguments(1, &k, sizeof(k));
  kp->SetKernelArguments(2, &v, sizeof(v));
  kp->SetKernelArguments(3, &g_pm, sizeof(g_pm));
  kp->SetKernelArguments(4, &g_pl, sizeof(g_pl));
  kp->SetKernelArguments(5, &g_pacc, sizeof(g_pacc));
  kp->SetKernelArguments(6, &HQ, sizeof(HQ));
  kp->SetKernelArguments(7, &HKV, sizeof(HKV));
  kp->SetKernelArguments(8, &N_kv, sizeof(N_kv));
  kp->SetKernelArguments(9, &cache_from, sizeof(cache_from));
  kp->SetKernelArguments(10, &d, sizeof(d));
  kp->SetKernelArguments(11, &window, sizeof(window));
  kp->SetKernelArguments(12, &softcap, sizeof(softcap));
  kp->SetKernelArguments(13, &chunk_kv, sizeof(chunk_kv));
  kp->SetKernelArguments(14, &n_chunks, sizeof(n_chunks));
  kp->SetKernelArguments(15, &ring_cap, sizeof(ring_cap));
  kp->SetKernelArguments(16, &kscale, sizeof(kscale));
  kp->SetKernelArguments(17, &vscale, sizeof(vscale));
  kp->SetKernelArguments(18, &dpos, sizeof(dpos));
  kp->SetKernelArguments(19, &max_nc, sizeof(max_nc));
  const int pg[3] = {HQ, max_nc, 1};
  const int pb[3] = {B, 1, 1};
  const unsigned int shmem = (unsigned int)(sizeof(float) * ((size_t)d + B));
  if (!StreamManager::Global().DispatchCommand(*kp, pg, pb, shmem))
    return false;
  kr->SetKernelArguments(0, &g_pm, sizeof(g_pm));
  kr->SetKernelArguments(1, &g_pl, sizeof(g_pl));
  kr->SetKernelArguments(2, &g_pacc, sizeof(g_pacc));
  kr->SetKernelArguments(3, &o, sizeof(o));
  kr->SetKernelArguments(4, &HQ, sizeof(HQ));
  kr->SetKernelArguments(5, &d, sizeof(d));
  kr->SetKernelArguments(6, &n_chunks, sizeof(n_chunks));
  kr->SetKernelArguments(7, &dpos, sizeof(dpos));
  kr->SetKernelArguments(8, &chunk_kv, sizeof(chunk_kv));
  kr->SetKernelArguments(9, &max_nc, sizeof(max_nc));
  const int rg[3] = {HQ, 1, 1};
  const int rb[3] = {B, 1, 1};
  return StreamManager::Global().DispatchCommand(*kr, rg, rb);
}

bool cuda_attention_interleaved_kvi8(
  const unsigned short *q_fp16, const signed char *k_i8,
  const signed char *v_i8, const unsigned short *k_scale,
  const unsigned short *v_scale, unsigned short *o_fp16, int num_heads_Q,
  int num_heads_KV, int N_q, int N_kv, int cache_from, int head_dim, int window,
  float softcap, int ring_cap) {
  if (num_heads_Q == 0 || N_q == 0 || N_kv == 0 || head_dim == 0)
    return true;
  if (!dev_accessible(q_fp16) || !dev_accessible(k_i8) ||
      !dev_accessible(v_i8) || !dev_accessible(k_scale) ||
      !dev_accessible(v_scale) || !dev_accessible(o_fp16))
    return false;
  // decode: split-KV int8 partials + shared fp32 reduce
  static const int sk_chunk = []() {
    const char *e = std::getenv("NNTR_CUDA_FLASH_DECODE");
    if (!e)
      return 64; // kvi8 decode has no dense fallback -- default the chunk on
    int c = atoi(e);
    return c > 0 ? c : 64;
  }();
  if (N_q == 1) {
    const int ck = (N_kv > sk_chunk) ? sk_chunk : N_kv;
    return attention_splitkv_decode_i8(q_fp16, k_i8, v_i8, k_scale, v_scale,
                                       o_fp16, num_heads_Q, num_heads_KV, N_kv,
                                       cache_from, head_dim, window, softcap,
                                       ck, ring_cap) &&
           (StreamManager::Global().maybeFinish(), true);
  }
  // prefill: int8 block-Q. NNTR_CUDA_KVI8_DP4A=1 routes to the dp4a variant
  // (int8-arithmetic QK^T with per-row Q quantization) -- experimental,
  // default off: it adds Q quantization noise on top of the K/V int8 loss.
  if (head_dim != 128 && head_dim != 256 && head_dim != 512)
    return false;
  static const bool _dp4a_on = nntr_env_on("NNTR_CUDA_KVI8_DP4A");
  // The dp4a kernel reads K/V as packed int words. Offsets inside the cache
  // are provably 4-aligned (d and VPL are multiples of 4) but the memory
  // planners pack pool tensors by cumulative byte size with NO alignment
  // padding, so the cache BASE itself can land misaligned (e.g. after an
  // odd-element fp16 scale tensor). Verify it here; fall back to the scalar
  // int8 kernel otherwise.
  const bool dp4a_ok =
    _dp4a_on && (((size_t)k_i8 | (size_t)v_i8) & (size_t)3) == 0;
  if (_dp4a_on && !dp4a_ok) {
    static bool warned = false;
    if (!warned) {
      warned = true;
      fprintf(stderr,
              "[cuda-kvi8] WARNING: KVI8_DP4A requested but the int8 K/V "
              "cache base is not 4-byte aligned; using the scalar kernel\n");
    }
  }
  const char *fn = dp4a_ok ? ((head_dim == 256)   ? "attn_blockq_i8_d256_dp4a"
                              : (head_dim == 512) ? "attn_blockq_i8_d512_dp4a"
                                                  : "attn_blockq_i8_d128_dp4a")
                           : ((head_dim == 256)   ? "attn_blockq_i8_d256"
                              : (head_dim == 512) ? "attn_blockq_i8_d512"
                                                  : "attn_blockq_i8_d128");
  auto kb = CudaContext::Global().registerCudaKernel(ATTN_I8_SRC, fn);
  if (!kb)
    return false;
  int win_bq = (window <= 0 || window >= N_kv) ? 0 : window;
  const int TM = 4;
  const int n_row_tiles = (N_q + TM - 1) / TM;
  kb->SetKernelArguments(0, &q_fp16, sizeof(q_fp16));
  kb->SetKernelArguments(1, &k_i8, sizeof(k_i8));
  kb->SetKernelArguments(2, &v_i8, sizeof(v_i8));
  kb->SetKernelArguments(3, &k_scale, sizeof(k_scale));
  kb->SetKernelArguments(4, &v_scale, sizeof(v_scale));
  kb->SetKernelArguments(5, &o_fp16, sizeof(o_fp16));
  kb->SetKernelArguments(6, &num_heads_Q, sizeof(num_heads_Q));
  kb->SetKernelArguments(7, &num_heads_KV, sizeof(num_heads_KV));
  kb->SetKernelArguments(8, &N_q, sizeof(N_q));
  kb->SetKernelArguments(9, &N_kv, sizeof(N_kv));
  kb->SetKernelArguments(10, &cache_from, sizeof(cache_from));
  kb->SetKernelArguments(11, &head_dim, sizeof(head_dim));
  kb->SetKernelArguments(12, &win_bq, sizeof(win_bq));
  kb->SetKernelArguments(13, &softcap, sizeof(softcap));
  kb->SetKernelArguments(14, &ring_cap, sizeof(ring_cap));
  const int grid[3] = {num_heads_Q * n_row_tiles, 1, 1};
  const int block[3] = {32, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kb, grid, block, 0))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

bool cuda_attention_interleaved_fp16(const unsigned short *q_fp16,
                                     const unsigned short *k_fp16,
                                     const unsigned short *v_fp16,
                                     unsigned short *o_fp16, int num_heads_Q,
                                     int num_heads_KV, int N_q, int N_kv,
                                     int cache_from, int head_dim, int window,
                                     float softcap, int ring_cap) {
  if (num_heads_Q == 0 || N_q == 0 || N_kv == 0 || head_dim == 0)
    return true;

  // mirror the KV cache to the device if it is host-resident (engine=cuda KV
  // cache is not UVM). K/V are [N_kv, num_heads_KV*head_dim] interleaved.
  const size_t kv_elems = (size_t)N_kv * num_heads_KV * head_dim;
  k_fp16 = mirror_kv(k_fp16, kv_elems);
  v_fp16 = mirror_kv(v_fp16, kv_elems);
  if (!k_fp16 || !v_fp16)
    return false;

  // Flash-decoding (split-KV) for M=1 decode with enough keys to fill the SMs.
  static const int sk_chunk = []() {
    const char *e = std::getenv("NNTR_CUDA_FLASH_DECODE");
    if (!e)
      return 0; // off
    int c = atoi(e);
    return c > 0 ? c : 64; // =1 -> default chunk 64; or an explicit chunk size
  }();
  if (sk_chunk > 0 && N_q == 1 && N_kv > sk_chunk) {
    if (attention_splitkv_decode(q_fp16, k_fp16, v_fp16, o_fp16, num_heads_Q,
                                 num_heads_KV, N_kv, cache_from, head_dim,
                                 window, softcap, sk_chunk, ring_cap)) {
      StreamManager::Global().maybeFinish();
      return true;
    }
  }

  // GEMM prefill attention (cuBLAS fp16 QK^T -> softmax -> PV): materialises
  // scores instead of the per-key flash reduce, so it is far faster for prefill
  // and head_dim-agnostic -- the lever for head_dim=128 (qwen3/llama) where
  // block-Q underperforms. Opt-in (NNTR_CUDA_GEMM_ATTN); falls through on any
  // cuBLAS/registration failure.
  // head_dim 256/512 are faster on block-Q (warp-shuffle, K/V reuse); GEMM wins
  // for the smaller head dims (128 = qwen3/llama) where block-Q underutilises.
  static const bool gemm_attn_on = []() {
    if (std::getenv("NNTR_CUDA_GEMM_ATTN") != nullptr)
      return true; // explicit opt-in (presence), preserves prior semantics
    // Caps-derived default (resolver attention cell): integrated GPUs (Orin
    // sm_87) need the cuBLAS GEMM attention because block-Q runs ~0.2 TFLOP/s
    // there; discrete (RTX) keeps block-Q. Safe — any cuBLAS/registration
    // failure falls through to block-Q, never wrong output.
    return ContextManager::Global().isIntegrated();
  }();
  // head_dim 256/512 (gemma4 sliding/global) were historically excluded because
  // block-Q beat the cuBLAS path on RTX/Adreno. On Orin (sm_87) block-Q runs at
  // only ~0.2 TFLOP/s, so the cuBLAS int8/fp16 Tensor-Core QK/PV is worth
  // trying here -- opt-in via NNTR_CUDA_GEMM_ATTN, so other arches keep
  // block-Q.
  if (gemm_attn_on && N_q > 1) {
    if (attention_gemm_prefill_fp16(q_fp16, k_fp16, v_fp16, o_fp16, num_heads_Q,
                                    num_heads_KV, N_q, N_kv, cache_from,
                                    head_dim, window, softcap)) {
      StreamManager::Global().maybeFinish();
      return true;
    }
  }

  // Block-Q multi-row prefill: one warp per (head, TM=4 row tile), warp-shuffle
  // d-dot, K/V reused across rows. 3-4x faster than the per-key LDS-reduce
  // attn_core_il_fp16 below, fp16-identical. Opt-in (NNTR_CUDA_BLOCKQ) until
  // folded; only the multi-row (prefill) path with head_dim in {256, 512}
  // (gemma4 sliding/global) -- decode (N_q==1) keeps split-KV above.
  static const bool blockq_on = nntr_env_on("NNTR_CUDA_BLOCKQ");
  if (blockq_on && N_q > 1 &&
      (head_dim == 128 || head_dim == 256 || head_dim == 512)) {
    const char *fn = (head_dim == 256)   ? "attn_blockq_d256"
                     : (head_dim == 512) ? "attn_blockq_d512"
                                         : "attn_blockq_d128";
    auto kb = CudaContext::Global().registerCudaKernel(ATTN_BLOCKQ_SRC, fn);
    if (kb) {
      // window<=0 or window>=N_kv -> disable the sliding mask (full causal);
      // avoids n+window overflow when mha passes INT_MAX for global layers.
      int win_bq = (window <= 0 || window >= N_kv) ? 0 : window;
      const int TM = 4;
      const int n_row_tiles = (N_q + TM - 1) / TM;
      kb->SetKernelArguments(0, &q_fp16, sizeof(q_fp16));
      kb->SetKernelArguments(1, &k_fp16, sizeof(k_fp16));
      kb->SetKernelArguments(2, &v_fp16, sizeof(v_fp16));
      kb->SetKernelArguments(3, &o_fp16, sizeof(o_fp16));
      kb->SetKernelArguments(4, &num_heads_Q, sizeof(num_heads_Q));
      kb->SetKernelArguments(5, &num_heads_KV, sizeof(num_heads_KV));
      kb->SetKernelArguments(6, &N_q, sizeof(N_q));
      kb->SetKernelArguments(7, &N_kv, sizeof(N_kv));
      kb->SetKernelArguments(8, &cache_from, sizeof(cache_from));
      kb->SetKernelArguments(9, &head_dim, sizeof(head_dim));
      kb->SetKernelArguments(10, &win_bq, sizeof(win_bq));
      kb->SetKernelArguments(11, &softcap, sizeof(softcap));
      kb->SetKernelArguments(12, &ring_cap, sizeof(ring_cap));
      const int grid[3] = {num_heads_Q * n_row_tiles, 1, 1};
      const int block[3] = {32, 1, 1};
      if (StreamManager::Global().DispatchCommand(*kb, grid, block, 0)) {
        StreamManager::Global().maybeFinish();
        return true;
      }
    }
  }

  auto kernel = CudaContext::Global().registerCudaKernel(ATTN_IL_FP16_SRC,
                                                         "attn_core_il_fp16");
  if (!kernel) {
    ml_loge("[CUDA] attn_core_il_fp16: registration failed");
    return false;
  }
  const int B = 128;
  kernel->SetKernelArguments(0, &q_fp16, sizeof(q_fp16));
  kernel->SetKernelArguments(1, &k_fp16, sizeof(k_fp16));
  kernel->SetKernelArguments(2, &v_fp16, sizeof(v_fp16));
  kernel->SetKernelArguments(3, &o_fp16, sizeof(o_fp16));
  kernel->SetKernelArguments(4, &num_heads_Q, sizeof(num_heads_Q));
  kernel->SetKernelArguments(5, &num_heads_KV, sizeof(num_heads_KV));
  kernel->SetKernelArguments(6, &N_q, sizeof(N_q));
  kernel->SetKernelArguments(7, &N_kv, sizeof(N_kv));
  kernel->SetKernelArguments(8, &cache_from, sizeof(cache_from));
  kernel->SetKernelArguments(9, &head_dim, sizeof(head_dim));
  kernel->SetKernelArguments(10, &window, sizeof(window));
  kernel->SetKernelArguments(11, &softcap, sizeof(softcap));
  // M2-B: bind the device position buffer so the captured graph reads the live
  // cache_from/N_kv on replay; nullptr keeps the baked-arg (non-graph) path.
  static const bool m2b_attn = nntr_env_on("NNTR_CUDA_M2B");
  const int *attn_dpos = m2b_attn ? cuda_pos_buffer() : nullptr;
  kernel->SetKernelArguments(12, &attn_dpos, sizeof(attn_dpos));
  kernel->SetKernelArguments(13, &ring_cap, sizeof(ring_cap));
  const int grid[3] = {N_q, num_heads_Q, 1};
  const int block[3] = {B, 1, 1};
  const unsigned int shmem =
    (unsigned int)(sizeof(float) * ((size_t)head_dim + B));
  static const bool dbg = std::getenv("NNTR_CUDA_ATTN_DBG") != nullptr;
  if (dbg)
    fprintf(
      stderr,
      "[ATTNDBG] HQ=%d HKV=%d N_q=%d N_kv=%d from=%d d=%d win=%d cap=%.0f "
      "shmem=%u\n",
      num_heads_Q, num_heads_KV, N_q, N_kv, cache_from, head_dim, window,
      softcap, shmem);
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block, shmem))
    return false;
  StreamManager::Global().maybeFinish();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    ml_loge("[CUDA] attn_core_il_fp16 runtime error: %s",
            cudaGetErrorString(e));
    return false;
  }
  return true;
}

bool cuda_attention_core_fp32(const float *Q, const float *K, const float *V,
                              float *O, int num_heads, int num_kv_heads,
                              int q_rows, int kv_len, int q_pos0, int head_dim,
                              int window, float softcap) {
  if (num_heads == 0 || q_rows == 0 || kv_len == 0 || head_dim == 0)
    return true;

  auto kernel =
    CudaContext::Global().registerCudaKernel(ATTN_CORE_SRC, "attn_core");
  if (!kernel) {
    ml_loge("[CUDA] attn_core: kernel registration failed");
    return false;
  }

  const int B = 128; // head_dim/B <= 4 for head_dim<=512
  kernel->SetKernelArguments(0, &Q, sizeof(Q));
  kernel->SetKernelArguments(1, &K, sizeof(K));
  kernel->SetKernelArguments(2, &V, sizeof(V));
  kernel->SetKernelArguments(3, &O, sizeof(O));
  kernel->SetKernelArguments(4, &num_heads, sizeof(num_heads));
  kernel->SetKernelArguments(5, &num_kv_heads, sizeof(num_kv_heads));
  kernel->SetKernelArguments(6, &q_rows, sizeof(q_rows));
  kernel->SetKernelArguments(7, &kv_len, sizeof(kv_len));
  kernel->SetKernelArguments(8, &q_pos0, sizeof(q_pos0));
  kernel->SetKernelArguments(9, &head_dim, sizeof(head_dim));
  kernel->SetKernelArguments(10, &window, sizeof(window));
  kernel->SetKernelArguments(11, &softcap, sizeof(softcap));

  const int grid[3] = {q_rows, num_heads, 1};
  const int block[3] = {B, 1, 1};
  const unsigned int shmem =
    (unsigned int)(sizeof(float) * ((size_t)head_dim + B));
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block, shmem))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

} // namespace nntrainer::cuda

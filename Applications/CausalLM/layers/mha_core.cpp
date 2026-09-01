// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   mha_core.cpp
 * @date   11 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This code is based on custom_multi_head_attention_layer.cpp.
 *         This code is a part of the break down version of the mha layer.
 */
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <env_compat.h>
#include <kv_ring.h> // causallm::kvRingCap (shared with models/transformer.h)
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_attention.h>
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_rope.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

static std::mutex rope_init_mtx;

// Minimum prefill step_size for routing attention/RoPE onto the GPU paths.
// Below the threshold the prefill falls to the host path. This is HEAD-DIM
// AWARE because the GPU flash/RoPE prefill kernels are only verified correct
// for head_dim 256/512 (e.g. gemma): those models keep q/k-norm GPU-resident,
// so the host fallback would read a stale SVM shadow -> they MUST take the GPU
// path, which is also a big win even at tiny step_size (~5.7x on Intel Arc,
// 16-token prefill). head_dim 128 (e.g. qwen3) keeps q/k-norm on the HOST, so
// the host prefill path is numerically correct there, whereas the Intel GPU
// flash kernel (flash_attention_prefill_f16_cl) produces GARBAGE for head_dim
// 128 at small/medium step_size (only very long prefill happens to survive) --
// so route its prefill to the host path. Env override: NNTR_MIN_PREFILL.
//   Regression context: commit 143543f71 lowered the x86 gate to 1 for ALL
//   head_dims (verified only on gemma / head_dim 256); that silently broke
//   qwen3 short-prompt coherence on Intel while the 1k-prompt benchmarks (long
//   prefill) kept passing.
static unsigned int min_prefill_thr(unsigned int head_dim) {
  static const int env = []() {
    const char *e = std::getenv("NNTR_MIN_PREFILL");
    return e ? std::atoi(e) : -1;
  }();
  if (env >= 0)
    return (unsigned int)env;

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
  // Intel/CUDA: no host NEON, so the GPU wins even at tiny step_size, for every
  // head_dim. The head_dim=128 (qwen3) degeneration under sampling was NOT the
  // flash kernel -- it was the GPU RoPE doing the cos/sin rotation in fp16
  // (rope_inplace_f16), which distorted the softmax distribution; fixed to fp32
  // rotation. So GPU flash + GPU RoPE is correct for all head_dims now.
  (void)head_dim;
  return 1u;
#else
  // ARM (Adreno): the image attention path is coherent (incl. qwen3); the
  // host-NEON crossover / GPU-request behaviour is unchanged from before.
  (void)head_dim;
  // value-checked (=0 disables): the CL bundle auto-injects NNTR_MHA_GPU=1,
  // so a presence check could never be turned off (env_compat.h trap).
  if (nntr_env_on("NNTR_MHA_GPU"))
    return 1u;
  return 32u;
#endif
}

#if defined(ENABLE_OPENCL)
// OpenCL attention/blas kernel interfaces (GPU attention + clmem residency).
// Guarded so the no-OpenCL CPU build compiles the host attention path.
#include <attention_kernels.h>
#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#endif
#include <fp16.h>
#include <layer_context.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <thread_manager.h>
#include <util_func.h>

#include <cstdint>

#if (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||            \
     defined(_M_IX86)) &&                                                      \
  defined(ENABLE_FP16)
#include <climits>
// libnntrainer builds the all-FP16 CPU attention kernels (compute_kcaches /
// compute_fp16vcache_transposed / compute_rotary_emb_value with _FP16 I/O, and
// the softmax_row[_inplace] _FP16 instantiations) only for ARM/NEON
// (neon_impl_fp16.cpp); the x86 backend has only the FP32/templated variants.
// This layer's FP16 CPU attention path references the all-FP16 versions, so on
// x86 they are undefined.
//
// nntrainer activations can be FP16 independently of the weight type (the
// model dtype is Weight-Activation, e.g. QINT4-FP16), so the FP16 CPU path IS
// reachable on x86 — notably RoPE, which the mha layer applies on the host even
// when attention runs on the GPU. Provide real x86 implementations (scalar,
// FP32 accumulation — matching the GPU's FP32-accumulate numerics) so x86+FP16
// works end to end. This block is x86-only; ARM keeps its NEON kernels.
namespace nntrainer {

// O[row] scores = scale * (Q · K), per (kv-head n, gqa group g, key row).
// Mirrors neon_impl_fp16.cpp compute_kcaches (tiling dropped — it is a pure
// cache-blocking optimization with identical results).
void compute_kcaches(const _FP16 *in, const _FP16 *kcache, _FP16 *output,
                     int num_rows, int num_cache_head, int head_dim,
                     int gqa_size, int tile_size,
                     size_t local_window_size = UINT_MAX, int head_start = 0,
                     int head_end = -1) {
  (void)tile_size;
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  const int start_row = ((size_t)num_rows < local_window_size)
                          ? 0
                          : num_rows - (int)local_window_size;
  const int row_cnt =
    ((size_t)num_rows < local_window_size) ? num_rows : (int)local_window_size;
  const float inv = 1.0f / std::sqrt((float)head_dim);
  for (int n = head_start; n < actual_head_end; ++n) {
    for (int r = 0; r < row_cnt; ++r) {
      const int row = start_row + r;
      const _FP16 *k_row =
        kcache + (size_t)(row * num_cache_head + n) * head_dim;
      for (int g = 0; g < gqa_size; ++g) {
        const _FP16 *in_ptr = in + (size_t)(n * gqa_size + g) * head_dim;
        float sum = 0.0f;
        for (int i = 0; i < head_dim; ++i)
          sum += (float)in_ptr[i] * (float)k_row[i];
        output[(size_t)r * num_cache_head * gqa_size + n * gqa_size + g] =
          (_FP16)(sum * inv);
      }
    }
  }
}

// O[head][d] = sum_j scores[j] * V[j, head, d], over the (windowed) key rows up
// to row_num. Mirrors neon_impl_fp16.cpp compute_fp16vcache_transposed.
void compute_fp16vcache_transposed(int row_num, const _FP16 *in,
                                   const _FP16 *vcache, _FP16 *output,
                                   int num_cache_head, int gqa_size,
                                   int head_dim,
                                   size_t local_window_size = UINT_MAX,
                                   int head_start = 0, int head_end = -1) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  const int j_start = ((size_t)row_num < local_window_size)
                        ? 0
                        : row_num + 1 - (int)local_window_size;
  std::vector<float> acc((size_t)gqa_size * head_dim);
  for (int n = head_start; n < actual_head_end; ++n) {
    std::fill(acc.begin(), acc.end(), 0.0f);
    for (int j = j_start; j <= row_num; ++j) {
      const _FP16 *vptr = vcache + (size_t)(j * num_cache_head + n) * head_dim;
      const int score_row =
        ((size_t)row_num < local_window_size) ? j : j - j_start;
      for (int h = 0; h < gqa_size; ++h) {
        const float a = (float)
          in[(size_t)score_row * gqa_size * num_cache_head + n * gqa_size + h];
        float *acc_h = acc.data() + (size_t)h * head_dim;
        for (int d = 0; d < head_dim; ++d)
          acc_h[d] += a * (float)vptr[d];
      }
    }
    for (int h = 0; h < gqa_size; ++h) {
      _FP16 *out_h = output + (size_t)(n * gqa_size + h) * head_dim;
      const float *acc_h = acc.data() + (size_t)h * head_dim;
      for (int d = 0; d < head_dim; ++d)
        out_h[d] = (_FP16)acc_h[d];
    }
  }
}

// RoPE: rotate (a,b) pairs across the half_ boundary by (cos,sin). Mirrors the
// all-FP16 neon_impl_fp16.cpp compute_rotary_emb_value.
void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, _FP16 *inout, _FP16 *output,
                              const _FP16 *cos_, const _FP16 *sin_) {
  for (unsigned int w = 0; w < width; w += dim) {
    for (unsigned int k = 0; k < half_; ++k) {
      const unsigned int i0 = w + k, i1 = w + k + half_;
      const float a = (float)inout[i0], b = (float)inout[i1];
      const float c = (float)cos_[k], s = (float)sin_[k];
      const float o0 = a * c - b * s, o1 = a * s + b * c;
      if (output != nullptr) {
        output[i0] = (_FP16)o0;
        output[i1] = (_FP16)o1;
      } else {
        inout[i0] = (_FP16)o0;
        inout[i1] = (_FP16)o1;
      }
    }
  }
}

// Column-wise softmax: softmax over rows [start_row,end_row) for each of the
// num_heads columns. Optional attention sink contributes exp(sink-max) to the
// denominator only. Mirrors neon_impl_fp16.cpp softmax_row_inplace.
template <>
void softmax_row_inplace<_FP16>(_FP16 *qk_out, size_t start_row, size_t end_row,
                                size_t num_heads, _FP16 *sink) {
  for (size_t c = 0; c < num_heads; ++c) {
    float mx = sink ? (float)sink[c] : -INFINITY;
    for (size_t r = start_row; r < end_row; ++r)
      mx = std::max(mx, (float)qk_out[r * num_heads + c]);
    float sum = sink ? std::exp((float)sink[c] - mx) : 0.0f;
    for (size_t r = start_row; r < end_row; ++r) {
      const float e = std::exp((float)qk_out[r * num_heads + c] - mx);
      qk_out[r * num_heads + c] = (_FP16)e;
      sum += e;
    }
    const float invsum = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    for (size_t r = start_row; r < end_row; ++r)
      qk_out[r * num_heads + c] =
        (_FP16)((float)qk_out[r * num_heads + c] * invsum);
  }
}
template <>
void softmax_row<_FP16>(_FP16 *qk_out, size_t start_row, size_t end_row,
                        size_t num_heads, _FP16 *sink) {
  softmax_row_inplace<_FP16>(qk_out, start_row, end_row, num_heads, sink);
}
} // namespace nntrainer
#endif

inline float convert_scalar(uint16_t h) {
  return nntrainer::compute_fp16_to_fp32(h);
}

// =============================================================================
// KV int8 helpers (paper section 3.7 path).
// =============================================================================
// Quantize a [step_size, num_heads, head_dim] fp16 source (laid out as
// row-major with row stride num_heads * head_dim) into an int8 destination
// plus a per-(token, head) fp16 scale tensor. Scale = amax / 127; with
// dequant via `int8_value * scale`. Symmetric (no zero point) so the
// matmul stays a simple int8 dot product plus a fp16 scale-out multiply.
static inline void quantize_kv_fp16_to_int8_per_row(
  const uint16_t *src_fp16, // [step_size, num_heads * head_dim]
  int8_t *dst_int8,         // [step_size, num_heads * head_dim]
  uint16_t *dst_scale_fp16, // [step_size, num_heads]
  unsigned int step_size, unsigned int num_heads, unsigned int head_dim) {
  for (unsigned int s = 0; s < step_size; ++s) {
    for (unsigned int h = 0; h < num_heads; ++h) {
      const uint16_t *row =
        src_fp16 + (size_t)s * num_heads * head_dim + (size_t)h * head_dim;
      float amax = 0.0f;
      for (unsigned int d = 0; d < head_dim; ++d) {
        float v = std::fabs(nntrainer::compute_fp16_to_fp32(row[d]));
        if (v > amax)
          amax = v;
      }
      const float scale = amax / 127.0f;
      const float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
      dst_scale_fp16[s * num_heads + h] =
        nntrainer::compute_fp32_to_fp16(scale);
      int8_t *out =
        dst_int8 + (size_t)s * num_heads * head_dim + (size_t)h * head_dim;
      for (unsigned int d = 0; d < head_dim; ++d) {
        int q = (int)std::round(nntrainer::compute_fp16_to_fp32(row[d]) * inv);
        if (q < -127)
          q = -127;
        if (q > 127)
          q = 127;
        out[d] = (int8_t)q;
      }
    }
  }
}

// =============================================================================
// §3.8 KV OHWI Phase 1 — env-gated K-cache layout switch.
// =============================================================================
// Default K cache layout: [B, max_seq_len, num_heads_KV, head_dim] row-major
// (concat over [t, h, d]). Paper §3.8 OHWI layout reinterprets the SAME buffer
// as [B, num_heads_KV, max_seq_len, head_dim] (per-head contiguous), which is
// the convolution-weight form that attention's QK matmul wants for 1×1-conv
// kernel access. Phase 1 only flips the WRITE side. The READ side allocates
// a fresh concat-layout scratch tensor and gathers, so all existing downstream
// attention paths (CPU gemm_attention / compute_kcaches / GPU
// two_conv_attention) keep working without modification.
static inline bool is_kv_ohwi_enabled() {
  static const bool on = std::getenv("NNTR_KV_OHWI") != nullptr;
  return on;
}

// Scatter a step's K (concat layout [step_size, num_heads_kv * head_dim])
// into the OHWI cache buffer at positions [cache_pos, cache_pos + step_size).
// cache_base points to the start of element [0,0,0,0] of the cache (i.e. the
// raw Tensor data pointer). The cache's total per-batch element count is
// max_seq_len * num_heads_kv * head_dim.
static inline void scatter_k_concat_to_ohwi_fp16(
  const uint16_t *src,  // [step_size, num_heads_kv * head_dim]
  uint16_t *cache_base, // raw cache buffer
  unsigned int batch, unsigned int cache_pos, unsigned int step_size,
  unsigned int num_heads_kv, unsigned int head_dim, unsigned int max_seq_len) {
  const size_t HD = (size_t)num_heads_kv * head_dim;
  const size_t per_head = (size_t)max_seq_len * head_dim;
  const size_t batch_off = (size_t)batch * num_heads_kv * per_head;
  for (unsigned int h = 0; h < num_heads_kv; ++h) {
    uint16_t *dst_head = cache_base + batch_off + (size_t)h * per_head;
    const uint16_t *src_head = src + (size_t)h * head_dim;
    for (unsigned int t = 0; t < step_size; ++t) {
      std::memcpy(dst_head + (size_t)(cache_pos + t) * head_dim,
                  src_head + (size_t)t * HD,
                  (size_t)head_dim * sizeof(uint16_t));
    }
  }
}

// Gather an OHWI K cache slice [0, cache_to) into concat layout
// [cache_to, num_heads_kv * head_dim]. dst is a freshly-allocated buffer of
// size cache_to * num_heads_kv * head_dim elements.
static inline void gather_k_ohwi_to_concat_fp16(
  const uint16_t *cache_base, // raw cache buffer
  uint16_t *dst,              // [cache_to, num_heads_kv * head_dim]
  unsigned int batch, unsigned int cache_to, unsigned int num_heads_kv,
  unsigned int head_dim, unsigned int max_seq_len) {
  const size_t HD = (size_t)num_heads_kv * head_dim;
  const size_t per_head = (size_t)max_seq_len * head_dim;
  const size_t batch_off = (size_t)batch * num_heads_kv * per_head;
  for (unsigned int h = 0; h < num_heads_kv; ++h) {
    const uint16_t *src_head = cache_base + batch_off + (size_t)h * per_head;
    uint16_t *dst_head = dst + (size_t)h * head_dim;
    for (unsigned int t = 0; t < cache_to; ++t) {
      std::memcpy(dst_head + (size_t)t * HD, src_head + (size_t)t * head_dim,
                  (size_t)head_dim * sizeof(uint16_t));
    }
  }
}

#ifdef ENABLE_FP16
// Decode/per-row K-cache dot product with int8-quantized K + per-(row, head)
// fp16 scale. Mirrors the layout of nntrainer::compute_kcaches<__fp16> but
// reads the cache as int8 bytes and scales the dot product by the matching
// scale value before the 1/sqrt(d) inverse. Output layout matches the fp16
// helper: out[(row-start)*num_cache_head*gqa + n*gqa + g].
static inline void compute_kcaches_int8_fp16(
  const _FP16 *in,         // [num_rows? actually only the latest row is used]
  const int8_t *kcache_i8, // [num_rows, num_cache_head, head_dim]
  const uint16_t *kscale,  // [num_rows, num_cache_head] fp16 bits
  _FP16 *output,           // [row_cnt, num_cache_head * gqa_size]
  int num_rows, int num_cache_head, int head_dim, int gqa_size,
  size_t local_window_size, int head_start, int head_end) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  // Note: local_window_size is size_t. Comparing num_rows (int) directly
  // promotes to size_t so UINT_MAX-as-no-window works correctly. Casting
  // local_window_size to int first would wrap UINT_MAX to -1 and skip the
  // entire row loop.
  const int start_row = (size_t)num_rows < local_window_size
                          ? 0
                          : num_rows - (int)local_window_size;
  const int row_cnt =
    (size_t)num_rows < local_window_size ? num_rows : (int)local_window_size;
  const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      const _FP16 *in_ptr = in + (n * gqa_size + g) * head_dim;
      for (int t = 0; t < row_cnt; ++t) {
        const int row = start_row + t;
        const int8_t *k_row =
          kcache_i8 + ((size_t)row * num_cache_head + n) * head_dim;
        const float s =
          nntrainer::compute_fp16_to_fp32(kscale[row * num_cache_head + n]);
        // Dot in fp32 keeps precision; int8 * fp16-as-fp32 -> fp32 acc.
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          sum += nntrainer::compute_fp16_to_fp32(
                   *reinterpret_cast<const uint16_t *>(&in_ptr[d])) *
                 (float)k_row[d];
        }
        const float v = sum * s * inv_sqrt_d;
        const size_t out_idx =
          (size_t)(row - start_row) * num_cache_head * gqa_size +
          (size_t)n * gqa_size + g;
        output[out_idx] = static_cast<_FP16>(v);
      }
    }
  }
}

// V cache dequant + transposed accumulation mirroring
// nntrainer::compute_fp16vcache_transposed but with int8 V + per-(row, head)
// fp16 scale. Output is fp16 [(num_cache_head*gqa_size), head_dim].
static inline void compute_fp16vcache_transposed_int8(
  int row_num, const _FP16 *in, const int8_t *vcache_i8, const uint16_t *vscale,
  _FP16 *output, int num_cache_head, int gqa_size, int head_dim,
  size_t local_window_size, int head_start, int head_end) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  // Promote to size_t to avoid wrapping UINT_MAX (no window) to -1.
  const int start_j = (size_t)row_num < local_window_size
                        ? 0
                        : row_num + 1 - (int)local_window_size;
  const int in_row_off = (size_t)row_num < local_window_size
                           ? 0
                           : row_num + 1 - (int)local_window_size;

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int h = 0; h < gqa_size; ++h) {
      float acc[256]; // head_dim <= 256 for our target models (Qwen3/Gemma3)
      NNTR_THROW_IF(head_dim > 256, std::invalid_argument)
        << "head_dim (" << head_dim << ") exceeds the int8 V-read scratch size";
      for (int d = 0; d < head_dim; ++d)
        acc[d] = 0.0f;

      for (int j = start_j; j <= row_num; ++j) {
        const int8_t *vptr =
          vcache_i8 + ((size_t)j * num_cache_head + n) * head_dim;
        const float vs =
          nntrainer::compute_fp16_to_fp32(vscale[j * num_cache_head + n]);
        const int attn_idx =
          (j - in_row_off) * gqa_size * num_cache_head + n * gqa_size + h;
        const float a_val = nntrainer::compute_fp16_to_fp32(
          *reinterpret_cast<const uint16_t *>(&in[attn_idx]));
        const float a_scaled = a_val * vs;
        for (int d = 0; d < head_dim; ++d) {
          acc[d] += a_scaled * (float)vptr[d];
        }
      }

      _FP16 *out_row = output + (n * gqa_size + h) * head_dim;
      for (int d = 0; d < head_dim; ++d)
        out_row[d] = static_cast<_FP16>(acc[d]);
    }
  }
}
#endif

namespace causallm {

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
// Upload a vector<vector<_FP16>> RoPE LUT to a flat device buffer ONCE (cached
// by table identity), [num_positions * half] row-major. The per-call host LUT
// row would otherwise force a blocking host->device cudaMemcpy in cuda_rope --
// nsys showed that at 74% of API time under NNTR_CUDA_ASYNC. There are only a
// couple of distinct tables (one per head_dim), so this is a few MB, uploaded
// once. Returns a device pointer; cuda_rope's dev check then skips the mirror.
static const unsigned short *
rope_lut_device(std::vector<std::vector<_FP16>> *table, int half) {
  static std::unordered_map<const void *, unsigned short *> cache;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = cache.find(table);
  if (it != cache.end())
    return it->second;
  const size_t npos = table->size();
  std::vector<unsigned short> flat(npos * (size_t)half);
  for (size_t p = 0; p < npos; ++p)
    std::memcpy(&flat[p * half], (*table)[p].data(),
                (size_t)half * sizeof(unsigned short));
  unsigned short *dev = nullptr;
  if (cudaMalloc(&dev, flat.size() * sizeof(unsigned short)) != cudaSuccess)
    dev = nullptr;
  else
    cudaMemcpy(dev, flat.data(), flat.size() * sizeof(unsigned short),
               cudaMemcpyHostToDevice);
  cache[table] = dev;
  return dev;
}
#endif

#define tile_size 4

static void compute_kcaches_fp32_reference(
  const float *in, const float *kcache, float *output, int num_rows,
  int num_cache_head, int head_dim, int gqa_size, size_t local_window_size,
  int head_start = 0, int head_end = -1) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  const int window = static_cast<int>(
    std::min(static_cast<size_t>(num_rows), local_window_size));
  const int start_row = num_rows - window;
  const float inv_sqrt_head_dim =
    1.0f / std::sqrt(static_cast<float>(head_dim));

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      const float *query = in + (n * gqa_size + g) * head_dim;
      for (int row = start_row; row < num_rows; ++row) {
        const float *key = kcache + (row * num_cache_head + n) * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          sum += query[d] * key[d];
        }
        output[(row - start_row) * num_cache_head * gqa_size + n * gqa_size +
               g] = sum * inv_sqrt_head_dim;
      }
    }
  }
}

static void compute_vcache_fp32_transposed_reference(
  int row_num, const float *in, const float *vcache, float *output,
  int num_cache_head, int gqa_size, int head_dim, size_t local_window_size,
  int head_start = 0, int head_end = -1) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  const int window = static_cast<int>(
    std::min(static_cast<size_t>(row_num + 1), local_window_size));
  const int start_row = row_num + 1 - window;

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int h = 0; h < gqa_size; ++h) {
      float *out = output + (n * gqa_size + h) * head_dim;
      std::fill(out, out + head_dim, 0.0f);

      for (int row = start_row; row <= row_num; ++row) {
        const int attn_row = row - start_row;
        const float a_val =
          in[attn_row * (num_cache_head * gqa_size) + n * gqa_size + h];
        const float *value = vcache + (row * num_cache_head + n) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          out[d] += a_val * value[d];
        }
      }
    }
  }
}

/************************************************************** */

/**
 * @brief constructor of MHACoreLayer
 */
MHACoreLayer::MHACoreLayer() :
  mha_core_props(
    nntrainer::props::NumHeads(), props::NumHeads_KV(),
    nntrainer::props::ProjectedKeyDim(), nntrainer::props::ProjectedValueDim(),
    nntrainer::props::OutputShape(), nntrainer::props::DropOutRate(),
    nntrainer::props::ReturnAttentionWeight(),
    nntrainer::props::AverageAttentionWeight(), nntrainer::props::MaxTimestep(),
    props::SlidingWindow(), props::InitSeqLen(), props::MaxNewTokens(),
    props::RopeTheta(), props::UseRope(), props::MaxPositionEmbeddings(),
    props::UseSink(), props::RopeScalingType(), props::RopeScalingFactor(),
    props::RopePartialRotaryFactor(), props::RopeScalingMaxPositionEmbeddings(),
    props::AttnLogitSoftcapping(), props::IsCausal(), props::UseGemmAttention(),
    props::GpuDecodeAttn(), props::GpuDecodeRope(), props::GpuOhwiRope()),
  sm(nntrainer::ActivationType::ACT_SOFTMAX),
  epsilon(1e-3),
  cache_index(0),
  num_heads_Q(0),
  num_heads_KV(0),
  head_dim(0),
  cache_shift(false) {
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

MHACoreLayer::~MHACoreLayer() {
  // Release the Adreno image-attention OHWI mirrors / image views, if any.
  // Routed through libnntrainer (release_cl_mem) so this layer needn't link
  // OpenCL directly.
#if defined(ENABLE_OPENCL)
  // release_cl_mem is OpenCL-only; the mirror handles stay null without it.
  nntrainer::release_cl_mem(k_image_ohwi);
  nntrainer::release_cl_mem(v_image_ohwi);
  nntrainer::release_cl_mem(v_image_tight);
  nntrainer::release_cl_mem(k_buf_ohwi);
  nntrainer::release_cl_mem(v_buf_ohwi);
#endif
}

/************************************************************** */

// [kv-window-ring] Guard for an attention arm that indexes the KV cache
// LINEARLY from the logical key count. When the ring is on the cache holds only
// kv_ring_cap physical rows, so such an arm would read past the buffer (the
// GPU two_conv / image kernels) or read physical rows as if they were absolute
// (the host compute path). Only three arms modulo-map the row:
// flash_attention_prefill_f16_cl, flash_decode_f16_cl and
// cuda_attention_interleaved_fp16.
//
// causallm::kvRingArmAvailable() already refuses to turn the ring on unless one
// of those is selectable, so this is the second line: it catches the case where
// the selected arm fails at RUNTIME and the cascade would otherwise walk down
// into a linear arm. Returns true when the arm must be skipped.
// (maybe_unused: every caller sits inside an ENABLE_OPENCL block.)
[[maybe_unused]] static bool mha_ring_refuses_arm(unsigned int ring_cap,
                                                  const char *arm) {
  if (ring_cap == 0)
    return false; // ring off: every arm is correct, nothing to refuse
  static bool logged = false;
  if (!logged) {
    logged = true;
    ml_loge("[kv-window-ring] skipping the '%s' attention arm: it indexes the "
            "KV cache linearly, but the cache holds only %u physical rows. "
            "Set NNTR_KV_WINDOW_RING=0 to keep the linear full-height cache.",
            arm, ring_cap);
  }
  return true;
}

// [kv-window-ring] Last resort: the cascade exhausted every ring-aware arm and
// is about to hand a Wcap-high cache to the host attention path, which walks
// absolute rows. Silently wrong attention is worse than a stop, so stop.
static void mha_ring_assert_host_path_ok(unsigned int ring_cap,
                                         const char *where) {
  NNTR_THROW_IF(ring_cap != 0, std::runtime_error)
    << "[kv-window-ring] " << where
    << ": no ring-aware attention arm resolved, but the KV cache holds only "
    << ring_cap
    << " physical rows. The host attention path reads absolute rows and would "
       "produce wrong attention. Enable a ring-aware arm (NNTR_KV_OHWI=1 + "
       "NNTR_MHA_GPU=1 on OpenCL, adding NNTR_MHA_GPU_DECODE=1 for decode; "
       "NNTR_CUDA_ATTN=1 on NNTR_ENGINE=cuda) or set NNTR_KV_WINDOW_RING=0.";
}

void MHACoreLayer::finalize(nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 5,
                std::invalid_argument)
    << "Multi head Attention layer needs 3, 4, or 5 inputs. "
       "(query, key, value; mask is optional; external cache_key + cache_value "
       "for external cache mode)";

  use_external_cache = (context.getNumInputs() >= 5);
  ml::train::TensorDim::TensorType activation_type = {
    context.getFormat(), context.getActivationDataType()};
  ml::train::TensorDim empty_dim(activation_type);

  const std::vector<ml::train::TensorDim> &input_dims =
    context.getInputDimensions();
  const ml::train::TensorDim &query_dim = input_dims[INOUT_INDEX::QUERY];
  const ml::train::TensorDim &key_dim = input_dims[INOUT_INDEX::KEY];

  /** max time step of this model */
  const unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  /** max position embeddings */
  max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mha_core_props).get();

  /** local window size */
  local_window_size = std::get<props::SlidingWindow>(mha_core_props).get();

  /** [kv-window-ring] physical ring capacity for this (sliding) layer. The ring
   * is fp16 external-cache only for now; the int8 internal cache (allocated at
   * full max_seq below) is NOT ring-sized, so keep it linear. */
  const unsigned int init_seq_len =
    std::get<props::InitSeqLen>(mha_core_props).get()
      ? std::get<props::InitSeqLen>(mha_core_props).get()
      : query_dim.height(); // unset -> the plane we were handed
  kv_ring_cap =
    causallm::kvRingLayerEligible(
      std::get<props::UseSink>(mha_core_props).get(), use_external_cache)
      ? causallm::kvRingCap((unsigned int)local_window_size, max_timestep,
                            causallm::effectivePrefillChunk(init_seq_len))
      : 0u;

  /** attention scaling computation */
  rope_scaling_type = std::get<props::RopeScalingType>(mha_core_props).get();
  scale = std::get<props::RopeScalingFactor>(mha_core_props).get();
  rope_partial_rotary_factor =
    std::get<props::RopePartialRotaryFactor>(mha_core_props).get();
  if (rope_scaling_type == "yarn")
    original_max_position_embeddings =
      std::get<props::RopeScalingMaxPositionEmbeddings>(mha_core_props).get();

  /** query_dim = (B, 1, seq_len, H_Q * Head_Dim ) */
  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_width = query_dim.width();
  /** key_dim = (B, 1, max_seq_len, H_KV * Head_Dim ) */
  const unsigned int key_width = key_dim.width();

  /**
   *  @note If NumHeads_KV is set, then use the value. Otherwise,
   *        we initialize num_heads_KV with num_heads_Q.
   */
  num_heads_Q = static_cast<size_t>(
    std::get<nntrainer::props::NumHeads>(mha_core_props).get());
  num_heads_KV =
    std::get<props::NumHeads_KV>(mha_core_props).empty()
      ? num_heads_Q
      : static_cast<size_t>(std::get<props::NumHeads_KV>(mha_core_props).get());

  // head_dim
  head_dim = static_cast<size_t>(query_width) / num_heads_Q;
  NNTR_THROW_IF(head_dim != key_width / num_heads_KV, std::invalid_argument)
    << "num_heads_Q and num_heads_KV are not properly given. Please check the "
       "num_heads_* are set correctly so that the `head_dim`s are all same for "
       "query / key / value";

  /** Weight for Sink */
  use_sink = std::get<props::UseSink>(mha_core_props).get();
  if (use_sink) {
#if ENABLE_FP16 && defined(__ANDROID__)
    nntrainer::TensorDim sink_dim(
      1, 1, 1, num_heads_Q,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       ml::train::TensorDim::DataType::FP16));
#else
    nntrainer::TensorDim sink_dim(
      1, 1, 1, num_heads_Q,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getActivationDataType()));
#endif
    sink_idx = context.requestWeight(sink_dim, nntrainer::Initializer::ZEROS,
                                     nntrainer::WeightRegularizer::NONE, 0.0f,
                                     0.0f, "sink");
  }

  attn_logit_softcapping =
    std::get<props::AttnLogitSoftcapping>(mha_core_props).get();

  /** Is Causal */
  is_causal = std::get<props::IsCausal>(mha_core_props).get();
  use_gemm_attention = std::get<props::UseGemmAttention>(mha_core_props).get();

  // Honor the LayerImpl `skip_prefill` property (parsed into layer_impl_props
  // by LayerImpl::setProperty). When set, mha_core writes its KV cache during
  // the prefill big-step but skips the (unused) prefill attention compute --
  // see the skip_prefill member doc in mha_core.h.
  if (!std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).get();

  // Paper section 3.7 int8 KV cache path. Reduces KV cache memory + read
  // bandwidth ~2x. The byte buffer is stored as UINT8; we treat the
  // bytes as signed int8 in the read/write code paths. Per-(token,
  // head) FP16 scale captures the row's amax. Env-gated so the FP16
  // baseline stays default.
  kv_int8 = std::getenv("NNTR_KV_INT8") != nullptr;

  /** Tensor for KV-Cache (only allocate internally when not using external
   * cache) */
  if (!use_external_cache) {
    if (kv_int8) {
      ml::train::TensorDim cache_key_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::UINT8});
      ml::train::TensorDim cache_value_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::UINT8});
      ml::train::TensorDim cache_key_scale_dim(
        {batch_size, 1, max_timestep, num_heads_KV},
        {context.getFormat(), ml::train::TensorDim::DataType::FP16});
      ml::train::TensorDim cache_value_scale_dim(
        {batch_size, 1, max_timestep, num_heads_KV},
        {context.getFormat(), ml::train::TensorDim::DataType::FP16});
      tensor_idx[AttentionParams::cache_key] = context.requestTensor(
        cache_key_dim, "cache_key", nntrainer::Initializer::NONE, false,
        nntrainer::TensorLifespan::MAX_LIFESPAN);
      tensor_idx[AttentionParams::cache_value] = context.requestTensor(
        cache_value_dim, "cache_value", nntrainer::Initializer::NONE, false,
        nntrainer::TensorLifespan::MAX_LIFESPAN);
      tensor_idx[AttentionParams::cache_key_scale] = context.requestTensor(
        cache_key_scale_dim, "cache_key_scale", nntrainer::Initializer::NONE,
        false, nntrainer::TensorLifespan::MAX_LIFESPAN);
      tensor_idx[AttentionParams::cache_value_scale] =
        context.requestTensor(cache_value_scale_dim, "cache_value_scale",
                              nntrainer::Initializer::NONE, false,
                              nntrainer::TensorLifespan::MAX_LIFESPAN);
    } else {
#ifdef ENABLE_FP16
      ml::train::TensorDim cache_key_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::FP16});
      ml::train::TensorDim cache_value_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::FP16});
#else
      ml::train::TensorDim cache_key_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::UINT16});
      ml::train::TensorDim cache_value_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::UINT16});
#endif

      tensor_idx[AttentionParams::cache_key] = context.requestTensor(
        cache_key_dim, "cache_key", nntrainer::Initializer::NONE, false,
        nntrainer::TensorLifespan::MAX_LIFESPAN);
      tensor_idx[AttentionParams::cache_value] = context.requestTensor(
        cache_value_dim, "cache_value", nntrainer::Initializer::NONE, false,
        nntrainer::TensorLifespan::MAX_LIFESPAN);
    }
  }

  theta = (float)std::get<props::RopeTheta>(mha_core_props).get();

  /** set Output dimension! - one output */
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = input_dims[0];
  output_dims[0].width(head_dim * num_heads_Q);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);

#if ENABLE_FP16 && defined(__ANDROID__)
  // Pre-build the one-time lazy state the first timed prefill otherwise pays
  // for (measured with NNTR_KV_STAGE_TPROF / clprof):
  //  - the process-wide flat RoPE LUT (~40ms, the KVST "lutcheck" segment),
  //  - the per-layer OHWI K/V mirrors + image views + the pitch-aligned
  //    tight V view (~0.5ms/layer of clCreateBuffer/clCreateImage).
  // Both are idempotent; refinalize re-entry is a no-op. Host/load-time
  // only -- no forward-path command ordering changes.
  if (theta > 0.0f && head_dim > 0 && head_dim % 2 == 0)
    ensure_rope_flat_lut();
#if defined(ENABLE_OPENCL)
  // OHWI K/V mirror + image-view prebuild (create_ohwi_kv_mirror etc.) is
  // OpenCL-only.
  // NOTE: image attn is ALL-OR-NOTHING per process. use_image_attn is the
  // switch the whole pipeline keys on (concat-RoPE drain mode, Q staging,
  // OHWI decode RoPE, engage); mixing image and flash layers (or flipping
  // per call) desyncs those stages — empirically garbage even at short
  // context. Sliding windows are handled IN the OHWI kernels (local_window
  // arg); the per-MODEL safety decision for geometry the kernels cannot
  // serve (d > 256 → force NNTR_KV_IMG_ATTN=0) is made in the model class
  // before layers finalize; here we only honor the env uniformly
  // (value-checked).
  if ([] {
        const char *e = std::getenv("NNTR_KV_IMG_ATTN");
        return e != nullptr && std::atoi(e) != 0; // value-checked: =0 disables
      }() &&
      !kv_int8 && head_dim % 8 == 0 && !kv_mirror_init) {
    static const unsigned int mirror_cap = []() {
      const char *e = std::getenv("NNTR_KV_MIRROR_CAP");
      return e ? (unsigned int)std::atoi(e) : 0u;
    }();
    unsigned int S_max = (max_timestep + 7u) & ~7u;
    if (mirror_cap >= 8 && mirror_cap < S_max)
      S_max = (mirror_cap + 7u) & ~7u;
    kv_mirror_S_max = S_max;
    bool m_ok = nntrainer::create_ohwi_kv_mirror(
                  /*is_v=*/false, num_heads_KV, head_dim, S_max,
                  reinterpret_cast<cl_mem *>(&k_buf_ohwi),
                  reinterpret_cast<cl_mem *>(&k_image_ohwi)) &&
                nntrainer::create_ohwi_kv_mirror(
                  /*is_v=*/true, num_heads_KV, head_dim, S_max,
                  reinterpret_cast<cl_mem *>(&v_buf_ohwi),
                  reinterpret_cast<cl_mem *>(&v_image_ohwi));
    kv_mirror_init = m_ok;
    if (m_ok) {
      // The env probe normally happens lazily at the first engage; with the
      // mirrors prebuilt the earlier GPU-RoPE block needs the answer too
      // (the q_stage gate runs before the engage block on the same call).
      use_image_attn = 1;
      // Tight V view at the typical prefill capacity (the engage path grows
      // it on demand if the live sequence exceeds the guess).
      unsigned int s_tight = S_max < 1024u ? S_max : 1024u;
      void *nimg = nullptr;
      if (nntrainer::create_ohwi_v_image_view(v_buf_ohwi, num_heads_KV,
                                              head_dim, &s_tight, &nimg) &&
          s_tight < S_max) {
        v_image_tight = nimg;
        kv_v_img_S = s_tight;
      } else if (nimg) {
        nntrainer::release_cl_mem(nimg);
      }
    } else {
      use_image_attn = 0; // permanent disable; flash takes over (same as
                          // the lazy-init failure path)
    }
  }
#endif // ENABLE_OPENCL (OHWI mirror prebuild)
#endif
}

/************************************************************** */

#ifdef ENABLE_FP16
// The RoPE LUT only needs to cover positions [0, max sequence). Models expose
// max_position_embeddings (the theoretical RoPE max, e.g. Gemma4 = 131072) but
// the live KV cache is only max_seq_len (e.g. 1024); sizing/uploading the LUT
// at 131072 is a 128x waste (the rope angle theta_j is position-independent, so
// a shorter LUT is exact). Cap to MaxTimestep (the model's max sequence).
// NNTR_ROPE_LUT_CAP overrides for experiments. (Capping the
// per-layer-transition re-upload from ~33-67MB to ~256-512KB was worth ~+500
// TPS at M=1024.)
unsigned int MHACoreLayer::rope_lut_positions() const {
  unsigned int cap =
    (unsigned int)std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();
  if (const char *e = std::getenv("NNTR_ROPE_LUT_CAP"))
    cap = (unsigned int)std::atoi(e);
  if (cap == 0 || cap > max_position_embeddings)
    cap = max_position_embeddings;
  return cap;
}

void MHACoreLayer::ensure_rope_flat_lut() {
  const unsigned int half_ = head_dim / 2;
  const unsigned int mp = rope_lut_positions();
  const RopeFlatKey key{(int)head_dim, theta, mp};

  // The cache is process-wide static and shared by every layer instance; guard
  // both the lookup and the build under rope_init_mtx (the flatten was already
  // run single-threaded at finalize, but the GPU-RoPE fallback can re-enter
  // from the forward path, and a per-slot insert must not race a concurrent
  // slot's insert -- std::map node stability protects EXISTING entries' data()
  // pointers across inserts, but the insert itself must be serialized).
  std::lock_guard<std::mutex> lock(rope_init_mtx);

  auto it = rope_flat_cache.find(key);
  if (it != rope_flat_cache.end()) {
    // Existing slot: just repoint the current-slot pointers. Stable across
    // calls (std::map nodes never move) so the GPU upload stays cached.
    rope_cos_flat_cur = it->second.first.data();
    rope_sin_flat_cur = it->second.second.data();
    return;
  }

  // New slot: build the flat table (math identical to the previous path).
  precompute_freqs((int)head_dim, mp, theta, true);
  RopeFlatVal val;
  val.first.assign((size_t)mp * half_, 0);
  val.second.assign((size_t)mp * half_, 0);
  // Row-wise bulk copy (rows are contiguous _FP16 = 2 bytes, same as the
  // uint16 bit view): the old per-element memcpy pair was ~12ms for
  // [2048 x 128].
  for (unsigned int p = 0; p < mp; ++p) {
    std::memcpy(&val.first[(size_t)p * half_], (*freqs_cos_fp16)[p].data(),
                (size_t)half_ * sizeof(uint16_t));
    std::memcpy(&val.second[(size_t)p * half_], (*freqs_sin_fp16)[p].data(),
                (size_t)half_ * sizeof(uint16_t));
  }
  auto ins = rope_flat_cache.emplace(key, std::move(val));
  rope_cos_flat_cur = ins.first->second.first.data();
  rope_sin_flat_cur = ins.first->second.second.data();
}
#endif

/**
 * @note In external KV cache mode (use_external_cache == true), this
 *       implements the inference forward pass using cache tensors supplied
 *       as input[3] (cache_key) and input[4] (cache_value). The host (e.g.
 *       KVCacheManager via setExternalTensors) is responsible for owning
 *       these buffers and for calling setCacheIndex() before each step to
 *       set the write position. After this call cache_index is advanced by
 *       input.height().
 *
 *       In legacy 3/4-input mode (use_external_cache == false) training is
 *       NYI and incremental_forwarding() is the inference path.
 *
 *       Input layout for external cache mode:
 *         input[0] = Q   (B, 1, step_size, num_heads_Q  * head_dim)
 *         input[1] = K   (B, 1, step_size, num_heads_KV * head_dim)
 *         input[2] = V   (B, 1, step_size, num_heads_KV * head_dim)
 *         input[3] = cache_key   (B, 1, max_seq_len, num_heads_KV * head_dim)
 *         input[4] = cache_value (B, 1, max_seq_len, num_heads_KV * head_dim)
 */
void MHACoreLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
  if (!use_external_cache) {
    // 3/4-input (internal KV cache) mode: full-sequence self-attention.
    //
    // incremental_forwarding() is being phased out as the inference entry
    // point, so attention must run through forwarding(). The internal-cache
    // attention worker already lives in incremental_forwarding(): it sources
    // the pool-backed cache_key/cache_value via context.getTensor() — which,
    // under the GPU SVM pool, are GPU-resident — and dispatches the same
    // GPU/CPU one_batch_incremental_forwarding paths. Drive it here over the
    // whole current sequence [0, seq). This is a stateless prefill: reset the
    // write base to 0 so RoPE positions and the cache write start at the
    // sequence origin, and do not carry cache_index across forwarding() calls.
    // (Decode / cache reuse remains incremental_forwarding's job — Step 8.)
    //
    // Training for the internal-cache mode is NYI (calcGradient/calcDerivative
    // are no-ops); preserve the previous training no-op and run attention only
    // for inference, so this change adds inference support without silently
    // changing training behaviour.
    if (training)
      return;
    // Precondition: the internal cache is sized to max_timestep, and
    // incremental_forwarding rejects to > max_timestep, so the model must set
    // max_timestep > prompt length (it already does: MAX_SEQ_LEN + new tokens).
    cache_index = 0;
    const unsigned int seq =
      (unsigned int)context.getInput(INOUT_INDEX::QUERY).height();
    incremental_forwarding(context, 0, seq, training);
    return;
  }
  if (kv_int8) {
    // The internal cache_key_scale/cache_value_scale tensors are
    // allocated only when use_external_cache is false. In the Qwen3
    // setup the cache_k/cache_v inputs come from
    // Transformer::createKVCachePlaceholders, which still emits FP16
    // placeholders. Until the placeholder helper is taught to emit
    // int8 + scale (or the mha layer is rewired with 7 inputs), the
    // external-cache + kv_int8 combination is unsupported.
    throw std::runtime_error(
      "NNTR_KV_INT8 requires internal cache mode (3-input mha). The "
      "current model uses external cache placeholders (5-input mha) "
      "that are still FP16. See project_kv_int8_plan memory for the "
      "wiring work needed in transformer.cpp.");
  }

  nntrainer::Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY);
  nntrainer::Tensor &value = context.getInput(INOUT_INDEX::VALUE);
  nntrainer::Tensor &output = context.getOutput(INOUT_INDEX::OUTPUT);

  nntrainer::Tensor &cache_key = context.getInput(3);
  nntrainer::Tensor &cache_value = context.getInput(4);

  nntrainer::Tensor sink;
  if (use_sink) {
    sink = context.getWeight(sink_idx);
  }

  unsigned int step_size = (incremental_step_size > 0)
                             ? incremental_step_size
                             : (unsigned int)query.height();
  unsigned int from = cache_index;
  unsigned int to = cache_index + step_size;

  auto get_step_dim = [step_size](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(step_size);
    return step_dim;
  };

  ml::train::TensorDim query_dim = query.getDim();
  ml::train::TensorDim key_dim = key.getDim();
  ml::train::TensorDim value_dim = value.getDim();
  ml::train::TensorDim output_dim = output.getDim();
  ml::train::TensorDim cache_key_dim = cache_key.getDim();
  ml::train::TensorDim cache_value_dim = cache_value.getDim();

  ml::train::TensorDim query_step_dim = get_step_dim(query_dim);
  ml::train::TensorDim key_step_dim = get_step_dim(key_dim);
  ml::train::TensorDim value_step_dim = get_step_dim(value_dim);
  ml::train::TensorDim output_step_dim = get_step_dim(output_dim);
  ml::train::TensorDim cache_key_step_dim = get_step_dim(cache_key_dim);
  ml::train::TensorDim cache_value_step_dim = get_step_dim(cache_value_dim);

  unsigned int batch_size = query_dim.batch();
  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    nntrainer::Tensor query_step = query.getSharedDataTensor(
      query_step_dim, batch * query_dim.getFeatureLen(), true);
    nntrainer::Tensor key_step = key.getSharedDataTensor(
      key_step_dim, batch * key_dim.getFeatureLen(), true);
    nntrainer::Tensor value_step = value.getSharedDataTensor(
      value_step_dim, batch * value_dim.getFeatureLen(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, batch * output_dim.getFeatureLen(), true);

    if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
#if ENABLE_FP16 && defined(__ANDROID__)
      nntrainer::TensorDim Q_step_dim = query_step_dim;
      nntrainer::TensorDim K_step_dim = key_step_dim;
      nntrainer::TensorDim V_step_dim = value_step_dim;
      nntrainer::TensorDim O_step_dim = output_step_dim;
      Q_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      K_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      V_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      O_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);

      nntrainer::Tensor Q_step = nntrainer::Tensor(Q_step_dim, true);
      nntrainer::Tensor K_step = nntrainer::Tensor(K_step_dim, true);
      nntrainer::Tensor V_step = nntrainer::Tensor(V_step_dim, true);
      nntrainer::Tensor O_step = nntrainer::Tensor(O_step_dim, true);

      Q_step.copyData(query_step);
      K_step.copyData(key_step);
      V_step.copyData(value_step);

      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, from, from, to, Q_step, K_step, V_step, O_step, cache_key,
          cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
          cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(batch, from, from, to, Q_step, K_step,
                                         V_step, O_step, cache_key, cache_value,
                                         cache_key_dim, cache_key_step_dim,
                                         cache_value_dim, cache_value_step_dim);
      }
      output_step.copyData(O_step);
#else
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(
          batch, from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim);
      }
#endif
    } else {
      one_batch_incremental_forwarding(
        batch, from, from, to, query_step, key_step, value_step, output_step,
        cache_key, cache_value, cache_key_dim, cache_key_step_dim,
        cache_value_dim, cache_value_step_dim);
    }
  }

  cache_index += step_size;
}

/**
 * @note This incremental_forwarding method is invoked for inference mode.
 *       Please note that Transformer Decoder's MHA takes only one sequence at a
 * step. Incremental forwarding function is used for this.
 */
// NYI guard fires inside incremental_forwarding() if kv_int8 was set.
// The cache + scale tensors are allocated (Phase 1), but the
// write/read paths are not yet implemented (Phase 2/3).
void MHACoreLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int _from, unsigned int _to,
                                          bool training) {
  // External KV cache path: from/to are interpreted as the absolute write
  // position; route through forwarding() which reads cache_key/cache_value
  // from input slots 3/4. forwarding() advances cache_index internally.
  if (use_external_cache) {
    cache_index = _from;
    incremental_step_size = _to - _from;
    forwarding(context, training);
    incremental_step_size = 0;
    return;
  }
  if (kv_int8) {
    // Internal cache mode: scale tensors were allocated in finalize().
    // Plumb them through to one_batch_incremental_forwarding via member
    // pointers (no signature churn on the per-batch entry point).
    kv_int8_key_scale =
      &context.getTensor(tensor_idx[AttentionParams::cache_key_scale]);
    kv_int8_value_scale =
      &context.getTensor(tensor_idx[AttentionParams::cache_value_scale]);
  }

  /// @todo replace step_size into input height
  unsigned int step_size = _to - _from;

  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  unsigned int from = _from;
  unsigned int to = _to;

  // `to` is an exclusive end index, so to == max_timestep exactly fills the
  // cache and is legal; only to > max_timestep overflows it.
  if (to > max_timestep) {
    // initial forwarding
    if (!_from) {
      throw std::invalid_argument(
        "to shouldn't greater than max_timestep for initial forwarding");
    } else {
      throw std::runtime_error("NYI: cache shift is not available");
      // exceeds the kv_cache size
      // KV_cache is shifted!
      cache_shift = true;
      from = max_timestep - 1;
      to = max_timestep;
    }
  }

  // util fn to compute tensor dimension for one step.
  auto get_step_dim = [step_size](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(step_size);
    return step_dim;
  };

  /** incremental forwarding for each batch */
  nntrainer::Tensor &query =
    context.getInput(INOUT_INDEX::QUERY); // projected query
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY); // projected key
  nntrainer::Tensor &value =
    context.getInput(INOUT_INDEX::VALUE); // projected value
  nntrainer::Tensor &output =
    context.getOutput(INOUT_INDEX::OUTPUT); // output to be projected

  nntrainer::Tensor &cache_key =
    context.getTensor(tensor_idx[AttentionParams::cache_key]);
  nntrainer::Tensor &cache_value =
    context.getTensor(tensor_idx[AttentionParams::cache_value]);

  nntrainer::Tensor sink;
  if (use_sink) {
    sink = context.getWeight(sink_idx);
  }

  ml::train::TensorDim query_dim =
    query.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim key_dim =
    key.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim value_dim =
    value.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim output_dim =
    output.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_dim =
    cache_key.getDim(); // (B, 1, max_timestep, n_heads_KV * head_dim)
  ml::train::TensorDim cache_value_dim =
    cache_value.getDim(); // (B, 1, max_timestep, n_heads_KV * head_dim)

  ml::train::TensorDim query_step_dim =
    get_step_dim(query_dim); // (1, 1, step_size, n_heads_Q * head_dim)
  ml::train::TensorDim key_step_dim = get_step_dim(key_dim);
  ml::train::TensorDim value_step_dim = get_step_dim(value_dim);
  ml::train::TensorDim output_step_dim =
    get_step_dim(output_dim); // (1, 1, step_size, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_step_dim =
    get_step_dim(cache_key_dim); // (1, 1, step_size, n_heads_KV * head_dim)

  ml::train::TensorDim cache_value_step_dim =
    get_step_dim(cache_value_dim); // (1, 1, step_size, n_heads_KV * head_dim)

  unsigned int batch_size = query_dim.batch();
  // do the incremental forwarding
  for (unsigned int batch = 0; batch < batch_size; ++batch) {

    // preparing step tensors
    nntrainer::Tensor query_step = query.getSharedDataTensor(
      query_step_dim, batch * query_dim.getFeatureLen(), true);
    nntrainer::Tensor key_step = key.getSharedDataTensor(
      key_step_dim, batch * key_dim.getFeatureLen(), true);
    nntrainer::Tensor value_step = value.getSharedDataTensor(
      value_step_dim, batch * value_dim.getFeatureLen(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, batch * output_dim.getFeatureLen(), true);

    if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
#if ENABLE_FP16 && defined(__ANDROID__)
      nntrainer::TensorDim Q_step_dim = query_step_dim;
      nntrainer::TensorDim K_step_dim = key_step_dim;
      nntrainer::TensorDim V_step_dim = value_step_dim;
      nntrainer::TensorDim O_step_dim = output_step_dim;
      Q_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      K_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      V_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      O_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);

      nntrainer::Tensor Q_step = nntrainer::Tensor(Q_step_dim, true);
      nntrainer::Tensor K_step = nntrainer::Tensor(K_step_dim, true);
      nntrainer::Tensor V_step = nntrainer::Tensor(V_step_dim, true);
      nntrainer::Tensor O_step = nntrainer::Tensor(O_step_dim, true);

      Q_step.copyData(query_step);
      K_step.copyData(key_step);
      V_step.copyData(value_step);
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, _from, from, to, Q_step, K_step, V_step, O_step, cache_key,
          cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
          cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(batch, _from, from, to, Q_step, K_step,
                                         V_step, O_step, cache_key, cache_value,
                                         cache_key_dim, cache_key_step_dim,
                                         cache_value_dim, cache_value_step_dim);
      }
      output_step.copyData(O_step);
#else
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, _from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(
          batch, _from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim);
      }
#endif
    } else {
      one_batch_incremental_forwarding(
        batch, _from, from, to, query_step, key_step, value_step, output_step,
        cache_key, cache_value, cache_key_dim, cache_key_step_dim,
        cache_value_dim, cache_value_step_dim);
    }
  }

  // increase cache size
  cache_index += step_size;
}

/**
 * @brief Function to compute Attention Scores using Tensor inputs. Wrapper
 * around nntrainer::compute_kcaches with multi-threading support
 *
 * Expected Input Shapes:
 * @param in (Query): [Batch, 1, sequence_len, Num_Heads_Q * Head_Dim]
 * @param cache (Key Cache): [Batch, 1, Max_Timestep, Num_Heads_KV * Head_Dim]
 * @param out (Attention Score): [Batch, 1, 1, Num_Heads_Q * Context_Len]
 *            where Context_Len is usually the current timestep 'to'.
 *
 */
void MHACoreLayer::compute_kcaches(nntrainer::Tensor &in,
                                   nntrainer::Tensor &cache,
                                   nntrainer::Tensor &out, unsigned int from,
                                   size_t sequence_len, unsigned int num_head,
                                   unsigned int group_size,
                                   unsigned int head_dim) {

  // Dispatch based on data type (FP32 or FP16)
  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (sequence_len == 1) {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_to_compute = is_causal ? from + 1 : from + sequence_len;
      unsigned int num_cache_head = num_head / group_size;

      // Use ThreadManager for lower overhead parallelization during decoding
      const float *in_data = in.getData<float>();
      float *out_data = out.getData<float>();

      auto &tm = nntrainer::ThreadManager::Global();
      if (cache.getDataType() == ml::train::TensorDim::DataType::FP32) {
        const float *cache_data = cache.getData<float>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            compute_kcaches_fp32_reference(
              in_data, cache_data, out_data, row_to_compute, num_cache_head,
              head_dim, group_size, local_window_size, head_kv, head_kv + 1);
          });
      } else {
        const uint16_t *cache_data = cache.getData<uint16_t>();
        tm.parallel_for(0, static_cast<size_t>(num_cache_head),
                        [=](size_t head_kv) {
                          nntrainer::compute_kcaches<uint16_t>(
                            in_data, cache_data, out_data, row_to_compute,
                            num_cache_head, head_dim, group_size, tile_size,
                            local_window_size, head_kv, head_kv + 1);
                        });
      }

    } else {
      // Sequence processing (prefill or chunked)
      // Iterate over ALL query rows so that no row is skipped even when
      // sequence_len > local_window_size.
      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(sequence_len), [=](size_t i) {
        float *input_addr = in.getData<float>() + num_head * head_dim * i;
        int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
        // Windowed cumulative offset so that each row's scores are placed
        // contiguously after the previous row's scores (respecting the window).
        size_t out_start_row = is_causal ? calc_windowed_attn_index(from + i) -
                                             calc_windowed_attn_index(from)
                                         : i * (from + sequence_len);
        float *output_addr = out.getData<float>() + out_start_row * num_head;

        if (cache.getDataType() == ml::train::TensorDim::DataType::FP32) {
          float *cache_addr = cache.getData<float>();
          compute_kcaches_fp32_reference(
            input_addr, cache_addr, output_addr, row_to_compute,
            num_head / group_size, head_dim, group_size, local_window_size);
        } else {
          uint16_t *cache_addr = cache.getData<uint16_t>();
          nntrainer::compute_kcaches<uint16_t>(
            input_addr, cache_addr, output_addr, row_to_compute,
            num_head / group_size, head_dim, group_size, tile_size,
            local_window_size);
        }
      });
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (sequence_len == 1) {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int num_rows = is_causal ? from + 1 : from + sequence_len;
      unsigned int num_cache_head = num_head / group_size;

      // Use ThreadManager for lower overhead parallelization during decoding
      const _FP16 *in_data = in.getData<_FP16>();
      _FP16 *out_data = out.getData<_FP16>();

      auto &tm = nntrainer::ThreadManager::Global();
      if (kv_int8) {
        // int8 K cache + per-(row, head) fp16 scale.
        const int8_t *cache_i8 =
          reinterpret_cast<const int8_t *>(cache.getData<uint8_t>());
        const uint16_t *kscale = cur_kv_int8_key_scale_batch;
        NNTR_THROW_IF(kscale == nullptr, std::invalid_argument)
          << "kv_int8 read path missing per-batch K-scale pointer";
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            compute_kcaches_int8_fp16(in_data, cache_i8, kscale, out_data,
                                      num_rows, num_cache_head, head_dim,
                                      group_size, local_window_size,
                                      (int)head_kv, (int)head_kv + 1);
          });
      } else {
        const _FP16 *cache_data = cache.getData<_FP16>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            nntrainer::compute_kcaches(
              in_data, cache_data, out_data, num_rows, num_cache_head, head_dim,
              group_size, tile_size, local_window_size, head_kv, head_kv + 1);
          });
      }
    } else {
      // Iterate over ALL query rows so that no row is skipped even when
      // sequence_len > local_window_size.
      auto &tm = nntrainer::ThreadManager::Global();
      if (kv_int8) {
        const int8_t *cache_i8 =
          reinterpret_cast<const int8_t *>(cache.getData<uint8_t>());
        const uint16_t *kscale = cur_kv_int8_key_scale_batch;
        const unsigned int num_cache_head = num_head / group_size;
        NNTR_THROW_IF(kscale == nullptr, std::invalid_argument)
          << "kv_int8 read path (prefill) missing per-batch K-scale pointer";
        // Iterate over ALL query rows + windowed cumulative offset (PR #3989).
        tm.parallel_for(0, static_cast<size_t>(sequence_len), [=](size_t i) {
          _FP16 *input_addr = in.getData<_FP16>() + num_head * head_dim * i;
          int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
          size_t out_start_row = is_causal
                                   ? calc_windowed_attn_index(from + i) -
                                       calc_windowed_attn_index(from)
                                   : i * (from + sequence_len);
          _FP16 *output_addr = out.getData<_FP16>() + out_start_row * num_head;
          compute_kcaches_int8_fp16(input_addr, cache_i8, kscale, output_addr,
                                    row_to_compute, num_cache_head, head_dim,
                                    group_size, local_window_size, 0, -1);
        });
        return;
      }
      tm.parallel_for(0, static_cast<size_t>(sequence_len), [=](size_t i) {
        _FP16 *input_addr = in.getData<_FP16>() + num_head * head_dim * i;
        _FP16 *cache_addr = cache.getData<_FP16>();
        int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
        // Windowed cumulative offset so that each row's scores are placed
        // contiguously after the previous row's scores (respecting the window).
        size_t out_start_row = is_causal ? calc_windowed_attn_index(from + i) -
                                             calc_windowed_attn_index(from)
                                         : i * (from + sequence_len);

        _FP16 *output_addr = out.getData<_FP16>() + out_start_row * num_head;

        nntrainer::compute_kcaches(input_addr, cache_addr, output_addr,
                                   row_to_compute, num_head / group_size,
                                   head_dim, group_size, tile_size,
                                   local_window_size);
      });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

// NNTR_KV_STAGE_TPROF=1: host-side wall timing of the V-copy -> k_scatter ->
// attention-return window (attributes the 39ms/layer GPU idle the CL-event
// profiler pins between scatter_copy_f16 and k_scatter_ohwi). Prints
// accumulated per-segment totals at each multiple of 26 windows (= 1 prefill
// forward of Gemma2-2B).
static bool _kvst_on() {
  static const bool on = std::getenv("NNTR_KV_STAGE_TPROF") != nullptr;
  return on;
}
static double _kvst_now() {
#ifdef _WIN32
  // No clock_gettime on MSVC; steady_clock is the same monotonic source.
  return std::chrono::duration<double, std::milli>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
#endif
}
// _kvst_mark_scatter and its accumulator state are only fed from the
// ENABLE_FP16 && ENABLE_OPENCL GPU scatter paths below (the only call site
// is inside that guard); mirror the same guard here or the state variables
// go unused (and the function undefined-but-unused) in other configs.
#if defined(ENABLE_FP16) && defined(ENABLE_OPENCL)
static double _kvst_t0 = 0;
static double _kvst_acc01 = 0, _kvst_acc_k = 0, _kvst_acc_v = 0,
              _kvst_acc_a = 0;
static int _kvst_n = 0;
static void _kvst_mark_scatter(double t1, double tk, double tv, double t2) {
  if (_kvst_t0 == 0)
    return;
  _kvst_acc01 += t1 - _kvst_t0;
  _kvst_acc_k += tk - t1;
  _kvst_acc_v += tv - tk;
  _kvst_acc_a += t2 - tv;
  _kvst_t0 = 0;
  if (++_kvst_n % 26 == 0) {
    std::fprintf(stderr,
                 "[KVST] n=%d vcopy->scatter %.2fms k_scatter %.2fms "
                 "v_scatter %.2fms attn-call %.2fms\n",
                 _kvst_n, _kvst_acc01, _kvst_acc_k, _kvst_acc_v, _kvst_acc_a);
    std::fflush(stderr);
    _kvst_acc01 = _kvst_acc_k = _kvst_acc_v = _kvst_acc_a = 0;
  }
}
#endif

void MHACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query_step,
  nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
  nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
  ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim) {

  /**
   *
   *  cache_key
   *  +------------------------------------------+
   *  |<--cache_index-->|<--b_cache_value_step-->|
   *  +------------------------------------------+
   *                    |<-------key_step------->|
   *  |<-------------b_cached_key--------------->|
   */

  // [kv-window-ring] Fail loud if a step write would straddle the ring seam.
  // Wcap is a multiple of the prefill chunk C and cache_index is C-aligned
  // (chunked prefill) or step==1 (decode), so cacheRow(cache_index)+step <=
  // Wcap by construction. A violation means a misconfigured chunk -- throw
  // rather than silently corrupt the neighbouring pool region (SVM writes do
  // not bounds-check). Covers every cache-row write below (same cache_index).
  if (kv_ring_cap &&
      cacheRow(cache_index) + (size_t)cache_key_step_dim.height() >
        (size_t)kv_ring_cap) {
    throw std::runtime_error(
      "mha_core kv-window-ring: step write straddles the ring seam "
      "(NNTR_PREFILL_CHUNK must divide the window ring capacity; keep the "
      "chunk a power-of-two <= the sliding window)");
  }

  // Load Input Tensors of this batch : b_ denotes a Tensor for this batch
  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() +
      cacheRow(cache_index) * cache_key_dim.width(),
    true);
  nntrainer::Tensor b_cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim,
    batch * cache_value_dim.getFeatureLen() +
      cacheRow(cache_index) * cache_value_dim.width(),
    true);

  // Static GPU_CLMEM residency: the wq/wk/wv FC outputs and the attention
  // output may live in planner cl_mem sub-buffers (class GPU_CLMEM). The GPU
  // stages bind these handles directly (RoPE / V-copy / image attention); any
  // path that touches them on the HOST (host RoPE, kv_int8 quant, CPU/NEON
  // attention, SVM-pointer kernels) must LOWER first (one blocking readback
  // into the SVM shadow) -- after a lower the handle is nulled so every later
  // consumer uses the now-fresh SVM plane. The output is RAISED back into its
  // sub-buffer whenever a non-cl_mem path wrote it (the wo FC reads cl_mem).
  // Offset-0 views only (batch 0; b_size==1 on the live path).
  // NNTR_CLMEM_MHA_OFF=1 (bisect): ignore residency handles entirely -- the
  // mha consumes the legacy SVM plane (valid only with NNTR_CLMEM_DUALOUT).
  // NNTR_KV_STAGE_TPROF: host time from mha entry to the rope-Q enqueue
  // (decomposes the copy_h2h->rope GPU-idle gap: executor/plumbing vs rope
  // wrapper).
  const double _mha_entry_t = _kvst_on() ? _kvst_now() : 0;

  static const bool mha_handles_off =
    std::getenv("NNTR_CLMEM_MHA_OFF") != nullptr;
  void *q_cl =
    (!mha_handles_off && query_step.getOffset() == 0 && query_step.isClMem())
      ? query_step.getClMem()
      : nullptr;
  void *k_cl =
    (!mha_handles_off && key_step.getOffset() == 0 && key_step.isClMem())
      ? key_step.getClMem()
      : nullptr;
  void *v_cl =
    (!mha_handles_off && value_step.getOffset() == 0 && value_step.isClMem())
      ? value_step.getClMem()
      : nullptr;
  void *o_cl =
    (attention_output_step.getOffset() == 0 && attention_output_step.isClMem())
      ? attention_output_step.getClMem()
      : nullptr;
  bool o_written_clmem = false;
  // Rotated-Q handle for the image attention's qk kernel. Defaults to the
  // tensor's own cl_mem (q_cl); the staged GPU-RoPE path below redirects it
  // to a dedicated q_stage temp -- writing the rotation IN-PLACE into q_cl
  // and reading the SAME handle from the next kernel without a drain flips
  // tokens on this driver (K and V survive precisely because their rope/
  // scatter chains write a DIFFERENT handle than they read; bisected
  // 2026-06-12: Q-only CLMEM flips, Q-only + a qk-side drain is clean).
  void *q_attn_clmem = q_cl;
  void *q_rope_staged = nullptr;
  auto lower_q = [&]() {
#if defined(ENABLE_OPENCL)
    if (q_cl) {
      nntrainer::clmem_lower_cl(query_step, 0);
      q_cl = nullptr;
    }
#endif // clmem_lower_cl is OpenCL-only; q_cl is null without it
  };
  auto lower_kv = [&]() {
#if defined(ENABLE_OPENCL)
    if (k_cl) {
      nntrainer::clmem_lower_cl(key_step, 0);
      k_cl = nullptr;
    }
    if (v_cl) {
      nntrainer::clmem_lower_cl(value_step, 0);
      v_cl = nullptr;
    }
#endif //
  };

  // NNTR_CLMEM_MHA_LOWER=1 (bisect): lower Q/K/V at entry unconditionally --
  // the mha then consumes the legacy SVM plane (read side = baseline) while
  // the FC write side stays cl_mem. Divergence persisting under this flag
  // indicts the FC out-copy; divergence vanishing indicts the mha bindings.
  static const bool mha_lower_all =
    std::getenv("NNTR_CLMEM_MHA_LOWER") != nullptr;
  if (mha_lower_all) {
    // After a lower the bytes live in the HOST view; the GPU stages below
    // read the SVM device-side, so hand the buffers back (unmap_force) --
    // the same protocol the embedding raise path uses.
    bool any = q_cl || k_cl || v_cl;
    lower_q();
    lower_kv();
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
    // cl_svm_unmap_force lives in the FP16 OpenCL kernel block.
    if (any) {
      nntrainer::cl_svm_unmap_force(query_step.getData<uint8_t>());
      nntrainer::cl_svm_unmap_force(key_step.getData<uint8_t>());
      nntrainer::cl_svm_unmap_force(value_step.getData<uint8_t>());
    }
#else
    (void)any;
#endif
  }

  // Kernel-chain K/V staging (drain removal). When the Adreno image attention
  // consumes this step's K/V (NNTR_KV_IMG_ATTN), the GPU RoPE/copy below route
  // through a persistent cl_mem staging temp instead of draining per op:
  //   rope-K -> k_stage (cl_mem, no clFinish) -> k_scatter(src_clmem=k_stage)
  //                                        \-> side-fill k_stage -> SVM cache
  //   V copy -> SVM cache (no clFinish); v_scatter reads value_step directly.
  // The per-op clFinish drains here measured 1289ms of GPU idle per 1K
  // prefill (V-copy->k_scatter 1010ms + gemm->rope 279ms). The SVM cache
  // writes are consumed only by host decode after the lm_head lower (a full
  // queue drain), so no per-op drain is needed; the scatters and the image
  // attention read pure cl_mem chains the in-order queue serializes.
  // NNTR_MHA_CLMEM: the OHWI mirrors are the ONLY store during the prefill
  // window -- the K side-fill and the V SVM cache write (the last undrained
  // SVM writes in the mha window) are skipped, and any host/SVM slab reader
  // first gathers the missing rows back from the mirrors (sync_kv_slab,
  // one drained boundary sync at decode entry). Pair with NNTR_CLMEM_QKV=1
  // for the full island-free window. v1 limit: prefill must fit the mirror
  // capacity (engage-gate misses beyond S_max leave the slab stale) and the
  // save_kvcache path is not synced.
  static const bool mha_clmem_mode = []() {
    // 2026-06-12 re-baseline: DEFAULT ON (inert without NNTR_KV_IMG_ATTN
    // staging; NNTR_MHA_CLMEM=0 restores the SVM side-fills).
    const char *e = std::getenv("NNTR_MHA_CLMEM");
    return !(e && e[0] == '0');
  }();
  // The cl_mem staging temps (q_stage/k_stage, tca_ensure device buffers) are
  // NOT coherent with the in-order SVM queue on Adreno: scatter/qk read garbage
  // from them at M>=32 (bisected 2026-06-18; the non-staged path, which routes
  // the GPU RoPE through the planner cl_mem q_cl in-place + the SVM cache, IS
  // coherent -- e.g. gemma4 "what is the capital of South Korea" -> "Seoul").
  // So staging is DISABLED BY DEFAULT. The non-staged path is still
  // GPU-resident (no host RoPE bounce) and, after the RoPE-LUT cap fix
  // (rope_lut_positions), is within ~2% of the broken-staged TPS at M=1024
  // (1520 vs 1556). Opt back into staging with NNTR_KV_STAGE=1 (to test an
  // SVM-backed staging fix); NNTR_NO_KV_STAGE still force-disables.
  static const bool kv_stage_on =
    [] {
      const char *e = std::getenv("NNTR_KV_IMG_ATTN");
      return e != nullptr && std::atoi(e) != 0; // value-checked: =0 disables
    }() &&
    std::getenv("NNTR_KV_STAGE") != nullptr &&
    std::getenv("NNTR_NO_KV_STAGE") == nullptr;
  void *k_stage = nullptr;               // rope-K wrote the staging temp
  const uint16_t *v_stage_svm = nullptr; // v_scatter source (value_step)
  void *v_stage_clmem = nullptr;         // v_scatter cl_mem source (v_cl)

  const double _mha_t_prelude = _kvst_on() ? _kvst_now() : 0;

  bool use_rope = theta > 0.0f;
  if (kv_int8) {
    // Host RoPE + host int8 quant read Q/K/V on the host: lower first.
    lower_q();
    lower_kv();
  }
  if (kv_int8) {
    // KV int8 path (paper section 3.7): Q RoPE in fp16 as usual, K RoPE
    // in-place on the fp16 key_step buffer, then quantize key_step and
    // value_step to int8 cache + per-(token, head) fp16 scale. V has no
    // RoPE so it's a direct quantize.
    if (use_rope) {
      apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, cache_index,
                                 true);
      apply_rotary_emb_tensor_v2(key_step, key_step, head_dim, cache_index,
                                 true);
    }
#ifdef ENABLE_FP16
    // Slice the int8 cache + fp16 scale for this batch/step window.
    const size_t scale_feature_len =
      kv_int8_key_scale->getDim().getFeatureLen();
    const size_t scale_step_width =
      static_cast<size_t>(num_heads_KV); // per-token scale row width
    const size_t scale_batch_offset = batch * scale_feature_len;
    // [kv-window-ring] scale row ring-indexed in lockstep with the int8 cache.
    const size_t scale_step_offset =
      scale_batch_offset + cacheRow(cache_index) * scale_step_width;

    const uint16_t *key_src =
      reinterpret_cast<const uint16_t *>(key_step.getData<_FP16>());
    const uint16_t *val_src =
      reinterpret_cast<const uint16_t *>(value_step.getData<_FP16>());

    int8_t *key_dst =
      reinterpret_cast<int8_t *>(b_cache_key_step.getData<uint8_t>());
    int8_t *val_dst =
      reinterpret_cast<int8_t *>(b_cache_value_step.getData<uint8_t>());
    uint16_t *key_scale_dst =
      reinterpret_cast<uint16_t *>(kv_int8_key_scale->getData<_FP16>()) +
      scale_step_offset;
    uint16_t *val_scale_dst =
      reinterpret_cast<uint16_t *>(kv_int8_value_scale->getData<_FP16>()) +
      scale_step_offset;

    const unsigned int step_n = to - from;
    quantize_kv_fp16_to_int8_per_row(key_src, key_dst, key_scale_dst, step_n,
                                     num_heads_KV, head_dim);
    quantize_kv_fp16_to_int8_per_row(val_src, val_dst, val_scale_dst, step_n,
                                     num_heads_KV, head_dim);

    // Stash per-batch scale base pointers (row 0 of this batch) for the
    // read path. compute_kcaches / compute_fp16vcache_transposed /
    // gemm_attention pick these up through layer members.
    cur_kv_int8_key_scale_batch =
      reinterpret_cast<const uint16_t *>(kv_int8_key_scale->getData<_FP16>()) +
      scale_batch_offset;
    cur_kv_int8_value_scale_batch = reinterpret_cast<const uint16_t *>(
                                      kv_int8_value_scale->getData<_FP16>()) +
                                    scale_batch_offset;

#else
    NNTR_THROW_IF(true, std::invalid_argument)
      << "NNTR_KV_INT8 requires ENABLE_FP16";
#endif
  } else if (use_rope) {
    bool gpu_rope_done = false;
#ifdef ENABLE_FP16
#if defined(ENABLE_OPENCL) // GPU concat-RoPE (rope_inplace_f16_cl) path
    // GPU-resident RoPE (residency): rotate Q in place and K into its cache
    // slice on the device, so the activation never bounces to the host. Active
    // only when the GPU attention path will follow (same NNTR_MHA_GPU +
    // use_gemm_attention + prefill-length + FP16 gate) on the standard concat
    // FP16 path. int8/OHWI keep their host RoPE (separate scatter/layout).
    // Kill-switch: NNTR_NO_GPU_ROPE. cos/sin LUT is uploaded as a small
    // constant; Q/K stay SVM-direct when the pool is SVM.
    static const bool _gpu_rope_off =
      std::getenv("NNTR_NO_GPU_ROPE") != nullptr;
    static const bool _mha_gpu_on = nntr_env_on("NNTR_MHA_GPU");
    const bool kv_ohwi_now =
      is_kv_ohwi_enabled() && !kv_int8 &&
      key_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      cache_key.getDataType() == ml::train::TensorDim::DataType::FP16;
    const unsigned int ROPE_MIN_PREFILL =
      min_prefill_thr((unsigned int)head_dim); // env NNTR_MIN_PREFILL
    // S3 decode: NNTR_MHA_GPU_DECODE moves the M=1 decode RoPE onto the GPU so
    // Q stays SVM-resident (q_cl handled in-place) and lower_q/lower_kv become
    // no-ops -- those are clEnqueueReadBuffer(CL_TRUE) blocking drains that
    // cost ~65 ms/token (35 layers x 2) on the host-RoPE decode path. (A)
    // GPU-RoPE-decode gate. Enabled by the NNTR_MHA_GPU_DECODE env flag (global
    // testing override) OR per-LAYER via the gpu_decode_rope property
    // (DEFAULT-ON only for models where decode GPU-RoPE is token-identical,
    // e.g. gemma4). The env read stays process-wide static; the property is
    // read PER-CALL (per-layer state) and OR'd in. The NNTR_NO_GPU_ROPE
    // kill-switch (_gpu_rope_off, checked at the if() below) still wins.
    static const bool _gpu_rope_decode_env =
      std::getenv("NNTR_MHA_GPU_DECODE") != nullptr;
    // The gpu_decode_rope property default is validated ONLY on the Intel
    // buffer/flash decode path. On the Adreno image-attention path
    // (NNTR_KV_IMG_ATTN) GPU-RoPE-at-decode produces garbage (the image KV/attn
    // consumes host-RoPE'd Q/K in a different layout), so suppress the property
    // there. The explicit NNTR_MHA_GPU_DECODE env override is unaffected.
    static const bool _kv_img_attn_env = [] {
      const char *e = std::getenv("NNTR_KV_IMG_ATTN");
      return e != nullptr && std::atoi(e) != 0; // value-checked: =0 disables
    }();
    const bool _gpu_rope_decode =
      _gpu_rope_decode_env ||
      (std::get<props::GpuDecodeRope>(mha_core_props).get() &&
       !_kv_img_attn_env);
    // Image path (Adreno): a GPU RoPE kernel dispatch per layer for the M=1
    // decode step is SLOWER than the trivial host rotation of a single row
    // (measured ~14.7 vs ~18 TPS); keep the GPU path for prefill (the big win)
    // and let decode fall to the host RoPE. Non-image (Intel/CUDA) unchanged.
    // NNTR_MHA_GPU_DECODE still forces the GPU decode path (_gpu_rope_decode).
    const bool _rope_len_ok = ((to - from) >= ROPE_MIN_PREFILL &&
                               !(_kv_img_attn_env && (to - from) == 1)) ||
                              (_gpu_rope_decode && (to - from) == 1);
    if (!_gpu_rope_off && _mha_gpu_on && use_gemm_attention && !kv_int8 &&
        !kv_ohwi_now && _rope_len_ok &&
        query_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      // Flat [max_pos, half_d] cos/sin LUT (fp16 bits) from the cached trig
      // table. Built once (cached by theta/head_dim/max_pos) so the GPU RoPE
      // path has no per-call flatten; rope_inplace_f16_cl uploads it once.
      const unsigned int mp = rope_lut_positions();
      if (mp >= cache_index + (to - from)) {
        ensure_rope_flat_lut();
        const double _mha_t_lut = _kvst_on() ? _kvst_now() : 0;
        // Current-slot stable pointers (per-(head_dim, theta, max_pos) cache):
        // for Gemma4 the two attention slots each keep their own resident
        // table, so the sliding<->full transition no longer rebuilds +
        // re-uploads the shared LUT every layer. The pointers are
        // std::map-node-stable, which keeps the rope_inplace_f16_cl device-LUT
        // upload cached per slot.
        const uint16_t *cos_lut = rope_cos_flat_cur;
        const uint16_t *sin_lut = rope_sin_flat_cur;
        uint16_t *q_p =
          reinterpret_cast<uint16_t *>(query_step.getData<_FP16>());
        const uint16_t *k_p =
          reinterpret_cast<const uint16_t *>(key_step.getData<_FP16>());
        uint16_t *kc_p =
          reinterpret_cast<uint16_t *>(b_cache_key_step.getData<_FP16>());
        const bool q_svm =
          query_step.getMemoryData() && query_step.getMemoryData()->isSVM();
        const bool kc_svm = b_cache_key_step.getMemoryData() &&
                            b_cache_key_step.getMemoryData()->isSVM();
        // Rotated-K staging temp: rope-K writes a cl_mem temp (no trailing
        // drain) which feeds k_scatter + the SVM-cache side-fill as a pure
        // kernel chain (see kv_stage_on decl). Grow-only, shared across
        // layers (in-order queue serializes reuse).
        static void *k_stage_buf = nullptr;
        static size_t k_stage_cap = 0;
        const size_t k_stage_bytes =
          (size_t)(to - from) * num_heads_KV * head_dim * sizeof(uint16_t);
        void *k_out_stage = nullptr;
        if (kv_stage_on && kc_svm &&
            nntrainer::ensure_cl_stage_buf(&k_stage_buf, &k_stage_cap,
                                           k_stage_bytes))
          k_out_stage = k_stage_buf;
        // Rotated-Q staging temp (CLMEM Q only): rope-Q writes q_stage and
        // the qk kernel reads q_stage -- write-handle != read-handle, the
        // same shape that keeps the K and V chains coherent without drains
        // (see q_attn_clmem). Only when the image attention is certain to
        // engage (mirrors prebuilt, capacity check) -- otherwise rope-Q
        // keeps its in-place + SVM-out (drained) form for the fallbacks.
        static void *q_stage_buf = nullptr;
        static size_t q_stage_cap = 0;
        void *q_out_stage = nullptr;
        if (kv_stage_on && q_cl != nullptr && use_image_attn == 1 &&
            kv_mirror_init && cache_index + (to - from) <= kv_mirror_S_max) {
          const size_t q_stage_bytes =
            (size_t)(to - from) * num_heads_Q * head_dim * sizeof(uint16_t);
          if (nntrainer::ensure_cl_stage_buf(&q_stage_buf, &q_stage_cap,
                                             q_stage_bytes))
            q_out_stage = q_stage_buf;
        }
        if (_kvst_on() && _mha_entry_t > 0) {
          static double acc_pre = 0, acc_mid = 0, acc_lut = 0;
          static int n_e2r = 0;
          const double tnow = _kvst_now();
          acc_pre += _mha_t_prelude - _mha_entry_t;
          acc_mid += _mha_t_lut - _mha_t_prelude;
          acc_lut += tnow - _mha_t_lut;
          if (++n_e2r % 26 == 0) {
            std::fprintf(stderr,
                         "[KVST] entry->ropeQ n=%d prelude=%.2f "
                         "lutcheck=%.2f tail=%.2f ms\n",
                         n_e2r, acc_pre, acc_mid, acc_lut);
            std::fflush(stderr);
            acc_pre = acc_mid = acc_lut = 0;
          }
        }
        // rope-Q's trailing SVM-out drain is LOAD-BEARING when the rotation
        // lands in the same plane the consumer reads (drain_svm_out=false
        // flipped " itſelf" -> " tubes", 2026-06-12). The staged form writes
        // a DIFFERENT cl_mem handle (q_stage) than it reads, which is
        // coherent without a drain (the K/V chain shape); the qk kernel then
        // binds q_stage. Non-staged keeps the drained in-place form.
        // On the Adreno image-attention path the qk kernel reads Q from the SVM
        // plane (Q_p, q_clmem=null; see the OHWI dispatch below). Rotating INTO
        // the FC's planner cl_mem (q_cl) leaves that SVM plane stale on this
        // driver (a cl_mem write is not visible through the aliased SVM
        // pointer), so the qk reads the UN-rotated Q -> garbage. Rotate
        // straight into the SVM plane (out_clmem=null -> writes Q_p) instead:
        // read the FC output from q_cl, write the rotation to the SVM Q_p the
        // qk actually consumes. GPU-resident, no cl_mem->SVM copy.
        // (Non-image/flash keeps q_cl.)
        void *q_rope_out = q_out_stage != nullptr
                             ? q_out_stage
                             : (_kv_img_attn_env ? nullptr : q_cl);
        bool ok =
          nntrainer::rope_inplace_f16_cl(
            q_p, q_p, cos_lut, sin_lut, to - from, num_heads_Q, head_dim,
            cache_index, mp, q_svm, /*in_clmem=*/q_cl,
            /*out_clmem=*/q_rope_out,
            // Image path: rotation lands in the SVM plane the same-queue
            // qk kernel reads next; the in-order queue serializes it, so
            // only a submission flush is needed (no full clFinish drain).
            // This is the pre-regression cost (a drain here cost ~5 TPS
            // of decode). Non-image keeps the default drain.
            /*drain_svm_out=*/!_kv_img_attn_env) &&
          nntrainer::rope_inplace_f16_cl(k_p, kc_p, cos_lut, sin_lut, to - from,
                                         num_heads_KV, head_dim, cache_index,
                                         mp, kc_svm, /*in_clmem=*/k_cl,
                                         /*out_clmem=*/k_out_stage);
        if (ok && q_out_stage != nullptr) {
          q_attn_clmem = q_out_stage;
          q_rope_staged = q_out_stage;
        }
        if (ok && k_out_stage != nullptr) {
          // Side-fill the SVM cache slice for host decode (consumed only
          // after the lm_head lower drains the queue): no per-op drain.
          const unsigned int kc_n =
            (to - from) * (unsigned int)num_heads_KV * (unsigned int)head_dim;
          if (mha_clmem_mode) {
            // MHA_CLMEM: mirror-only store; the boundary gather syncs the
            // slab for host decode/save. No SVM write here.
            k_stage = k_stage_buf;
          } else if (nntrainer::gpu_copy_f16_cl(
                       k_p, kc_p, kc_n, /*svm_inputs=*/true,
                       /*in_clmem=*/k_stage_buf, /*out_clmem=*/nullptr,
                       /*drain=*/false)) {
            k_stage = k_stage_buf;
            static int _kv_stage_logged = 0;
            if (!_kv_stage_logged) {
              _kv_stage_logged = 1;
              ml_logd("[KV-STAGE] engaged M=%u kbytes=%zu", to - from,
                      k_stage_bytes);
            }
          } else {
            // Side-fill kernel unavailable: redo rope-K straight into the
            // SVM cache slice (drained), abandoning the staging chain.
            ok = nntrainer::rope_inplace_f16_cl(
              k_p, kc_p, cos_lut, sin_lut, to - from, num_heads_KV, head_dim,
              cache_index, mp, kc_svm, /*in_clmem=*/k_cl,
              /*out_clmem=*/nullptr);
          }
        }
        if (ok) {
          // V: no RoPE — scatter the V slice into its cache window on the GPU
          // (residency: keep V on the device, no host copy). Falls back to the
          // host copy if the GPU copy is unsupported.
          const bool v_svm = value_step.getMemoryData() &&
                             value_step.getMemoryData()->isSVM() &&
                             b_cache_value_step.getMemoryData() &&
                             b_cache_value_step.getMemoryData()->isSVM();
          const uint16_t *v_in =
            reinterpret_cast<const uint16_t *>(value_step.getData<_FP16>());
          uint16_t *v_out =
            reinterpret_cast<uint16_t *>(b_cache_value_step.getData<_FP16>());
          const unsigned int v_n =
            (to - from) * (unsigned int)num_heads_KV * (unsigned int)head_dim;
          // Staged chain: the cache write is a decode side-fill (drain
          // skipped); v_scatter reads this step's V straight from value_step
          // (same bytes, no dependency on the cache write).
          const bool v_stage = kv_stage_on && v_svm;
          if (mha_clmem_mode && v_stage) {
            // MHA_CLMEM: mirror-only store; v_scatter reads value_step.
            v_stage_svm = v_in;
            v_stage_clmem = v_cl;
            if (_kvst_on())
              _kvst_t0 = _kvst_now();
          } else if (nntrainer::gpu_copy_f16_cl(v_in, v_out, v_n, v_svm,
                                                /*in_clmem=*/v_cl,
                                                /*out_clmem=*/nullptr,
                                                /*drain=*/!v_stage)) {
            if (v_stage) {
              v_stage_svm = v_in;
              v_stage_clmem = v_cl;
            }
            if (_kvst_on())
              _kvst_t0 = _kvst_now();
          } else {
            // Host fallback reads V on the host: lower first.
            if (v_cl) {
              nntrainer::clmem_lower_cl(value_step, 0);
              v_cl = nullptr;
            }
            b_cache_value_step.copyData(value_step); // host fallback
          }
          gpu_rope_done = true;
        }
      }
    }
#endif // ENABLE_OPENCL (GPU concat-RoPE)
#endif
#ifdef ENABLE_FP16
#if defined(ENABLE_OPENCL) // OHWI image GPU-RoPE (rope_inplace_f16_cl)
    // ---- (C) Adreno OHWI image-attention DECODE GPU-RoPE
    // --------------------- The concat GPU-RoPE branch above is gated
    // `!kv_ohwi_now`, but the live Adreno gemma4 decode path is the
    // IMAGE-attention path (NNTR_KV_IMG_ATTN), which is a SEPARATE env from
    // NNTR_KV_OHWI: is_kv_ohwi_enabled()/kv_ohwi_now are FALSE there, so the
    // concat branch is also skipped and gpu_rope_done stays false -> the host
    // fallback (1810) runs lower_q()+lower_kv() (two BLOCKING
    // clEnqueueReadBuffer drains, ~16-35 ms/token over 35 layers) and host
    // apply_rotary_emb. Rotate Q/K/V ON THE GPU instead, landing the bytes in
    // exactly the SVM buffers the existing GPU OHWI scatter + image attention
    // already consume:
    //   * Q  -> rotate IN-PLACE into query_step's SVM plane (Q_p). The OHWI
    //          decode qk kernel reads Q_p with q_clmem=nullptr (see 2340), so
    //          the rotation must land in the SVM shadow. When the FC parked Q
    //          in a planner cl_mem (q_cl != null) we read cl_mem -> SVM
    //          (drained once, then null q_cl) so Q_p is fresh.
    //   * K  -> rotate from key_step into b_cache_key_step's SVM concat cache
    //          slice (kc_p). k_scatter_ohwi_cl reads b_cache_key_step (2317).
    //          This is the SAME destination the host `else` branch writes
    //          (1881) and the SAME destination the concat GPU-RoPE writes
    //          (kc_p, 1619).
    //   * V  -> flat copy value_step -> b_cache_value_step's SVM slice (no
    //   RoPE).
    //          v_scatter_ohwi_t_cl reads b_cache_value_step (2326) when
    //          v_stage_svm is null (the unstaged decode path here). Matches the
    //          host V copy (1892) byte-for-byte.
    // Token-identical: rope_inplace_f16_cl == apply_rotary_emb_tensor_v2 for
    // the rotated bytes (the concat GPU-RoPE already depends on this
    // equivalence) and the destinations are identical to the host fallback's.
    // drain_svm_out=false / drain=false keep the writes as in-order enqueues
    // consumed only by the same-queue GPU OHWI scatter + image attention (the
    // SVM cache is read on the host only after the lm_head lower drains the
    // queue) -- the residency win. Gate: NNTR_OHWI_GPU_ROPE (default OFF ->
    // host path unchanged, A/B-able). Falls back (leaves gpu_rope_done=false ->
    // host RoPE) if: not the IMG-ATTN env, not a decode step (M!=1), not FP16,
    // kv_int8, the rope LUT does not cover the position, the cache slices are
    // not SVM, or any GPU op fails. Enabled by the NNTR_OHWI_GPU_ROPE env
    // (global override) OR per-LAYER via the gpu_ohwi_rope property (default-on
    // where token-identical: gemma4 +32%, gemma2 +8%; qwen3 stays false --
    // head_dim=128/q-k-norm diverges).
    static const bool _ohwi_gpu_rope_env =
      std::getenv("NNTR_OHWI_GPU_ROPE") != nullptr;
    const bool _ohwi_gpu_rope =
      _ohwi_gpu_rope_env || std::get<props::GpuOhwiRope>(mha_core_props).get();
    // Invariant: OHWI GPU-RoPE only feeds the image-attention layout — run
    // it only when this layer's attention actually takes the image path
    // (use_image_attn == 1; uniform per process, see the prebuild note).
    if (!gpu_rope_done && _ohwi_gpu_rope && !_gpu_rope_off && _mha_gpu_on &&
        _kv_img_attn_env && use_image_attn == 1 && use_gemm_attention &&
        !kv_int8 && (to - from) == 1 &&
        query_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        key_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        value_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        cache_key.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        cache_value.getDataType() == ml::train::TensorDim::DataType::FP16) {
      const unsigned int mp = rope_lut_positions();
      // SVM cache slices are the scatter sources; require them SVM-backed (they
      // are on the live NNTR_GPU_SVM_POOL path). Q may be cl_mem (FC output) or
      // SVM; rope_inplace handles cl_mem-in -> SVM-out.
      const bool kc_svm = b_cache_key_step.getMemoryData() &&
                          b_cache_key_step.getMemoryData()->isSVM();
      const bool vc_svm = b_cache_value_step.getMemoryData() &&
                          b_cache_value_step.getMemoryData()->isSVM();
      const bool q_svm =
        query_step.getMemoryData() && query_step.getMemoryData()->isSVM();
      const bool v_svm =
        value_step.getMemoryData() && value_step.getMemoryData()->isSVM();
      if (mp >= cache_index + (to - from) && kc_svm && vc_svm) {
        ensure_rope_flat_lut();
        const uint16_t *cos_lut = rope_cos_flat_cur;
        const uint16_t *sin_lut = rope_sin_flat_cur;
        uint16_t *q_p =
          reinterpret_cast<uint16_t *>(query_step.getData<_FP16>());
        const uint16_t *k_p =
          reinterpret_cast<const uint16_t *>(key_step.getData<_FP16>());
        uint16_t *kc_p =
          reinterpret_cast<uint16_t *>(b_cache_key_step.getData<_FP16>());
        const uint16_t *v_in =
          reinterpret_cast<const uint16_t *>(value_step.getData<_FP16>());
        uint16_t *v_out =
          reinterpret_cast<uint16_t *>(b_cache_value_step.getData<_FP16>());
        const unsigned int kv_n =
          (to - from) * (unsigned int)num_heads_KV * (unsigned int)head_dim;
        // ORDER MATTERS for safe partial-failure fallback: rotate K and V into
        // the cache slices FIRST (these read the UNMODIFIED key_step/value_step
        // and write b_cache_key/value_step), and rotate Q IN-PLACE LAST. If any
        // op fails, gpu_rope_done stays false and the host fallback re-reads
        // the still-untouched key_step/value_step/query_step (the partial GPU
        // writes into the cache slices get overwritten by the host RoPE; Q is
        // untouched unless every op already succeeded). So Q is never
        // double-rotated. K: rotate key_step -> b_cache_key_step SVM slice (the
        // OHWI scatter source). cl_mem-in (k_cl) -> SVM-out. key_step left
        // unmodified.
        bool ok = nntrainer::rope_inplace_f16_cl(
          k_p, kc_p, cos_lut, sin_lut, to - from, num_heads_KV, head_dim,
          cache_index, mp, kc_svm, /*in_clmem=*/k_cl,
          /*out_clmem=*/nullptr, /*drain_svm_out=*/false);
        // V: flat copy value_step -> b_cache_value_step SVM slice (no RoPE, the
        // OHWI v-scatter source). value_step left unmodified.
        if (ok)
          ok = nntrainer::gpu_copy_f16_cl(v_in, v_out, kv_n, v_svm,
                                          /*in_clmem=*/v_cl,
                                          /*out_clmem=*/nullptr,
                                          /*drain=*/false);
        // Q LAST: in-place into the SVM shadow (the OHWI decode qk reads Q_p
        // with q_clmem=null). cl_mem-in -> SVM-out when the FC parked Q in
        // cl_mem. Only run once K+V are committed so a Q failure cannot leave a
        // half-rotated, then re-rotated, Q.
        if (ok)
          ok = nntrainer::rope_inplace_f16_cl(
            q_p, q_p, cos_lut, sin_lut, to - from, num_heads_Q, head_dim,
            cache_index, mp, q_svm, /*in_clmem=*/q_cl, /*out_clmem=*/nullptr,
            /*drain_svm_out=*/false);
        if (ok) {
          // All committed. Q/K/V no longer read on the host: null their cl_mem
          // handles (the GPU rotation already read them, SVM shadows are fresh)
          // so no later lower_q/lower_kv fires.
          q_cl = nullptr;
          k_cl = nullptr;
          v_cl = nullptr;
          gpu_rope_done = true;
          static int _ohwi_rope_logged = 0;
          if (!_ohwi_rope_logged) {
            _ohwi_rope_logged = 1;
            ml_logd("[OHWI-GPU-ROPE] engaged M=%u cache_index=%u mp=%u",
                    to - from, cache_index, mp);
          }
        }
        // ok==false -> gpu_rope_done stays false: host fallback runs unchanged
        // (the partial GPU writes above are harmless -- the host RoPE rewrites
        // Q/K/V over them after lower_q/lower_kv).
      }
    }
#endif // ENABLE_OPENCL (OHWI image GPU-RoPE)
#endif
// kv_ohwi_now / gpu_rope_done are declared inside the ENABLE_FP16 +
// ENABLE_OPENCL concat-RoPE block above, so this diagnostic must carry the
// same pair of guards (an enable-fp16=false OpenCL build otherwise fails to
// compile on the undeclared kv_ohwi_now).
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
    if (std::getenv("NNTR_ROPE_TPROF")) {
      static int _dbg_pf = 0, _dbg_dec = 0;
      const bool _dec = (to - from) == 1;
      if ((_dec && _dbg_dec < 1) || (!_dec && _dbg_pf < 1)) {
        if (_dec)
          ++_dbg_dec;
        else
          ++_dbg_pf;
        std::fprintf(
          stderr,
          "[ROPE-PATH] %s M=%u gpu_rope_done=%d kv_ohwi=%d -> %s RoPE\n",
          _dec ? "DECODE" : "PREFILL", to - from, (int)gpu_rope_done,
          (int)kv_ohwi_now, gpu_rope_done ? "GPU" : "HOST(lower)");
        std::fflush(stderr);
      }
    }
#endif
    if (!gpu_rope_done) {
      // Host RoPE reads/writes Q/K/V on the host: lower first (decode and any
      // GPU-RoPE-ineligible path).
      // NNTR_ROPE_TPROF: per-token wall time of the host-RoPE lower_q+lower_kv
      // drains (decode only). Accumulated across layers; printed each decode
      // token (reset). Measures whether GPU-RoPE-for-OHWI is worth the work.
      static const bool _rope_tprof = std::getenv("NNTR_ROPE_TPROF") != nullptr;
      static double _rope_acc = 0.0;
      static int _rope_layers = 0;
      static unsigned int _rope_M = 0;
      std::chrono::steady_clock::time_point _rt0;
      if (_rope_tprof)
        _rt0 = std::chrono::steady_clock::now();
      lower_q();
      lower_kv();
      if (_rope_tprof) {
        const double _dt = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - _rt0)
                             .count();
        if ((to - from) > 1) {
          // Prefill: accumulate across its (skip_prefill) producing layers and
          // print a running cumulative — the last line before decode = the
          // total prefill host-RoPE lower cost.
          static double _pf_acc = 0.0;
          static int _pf_n = 0;
          _pf_acc += _dt;
          ++_pf_n;
          std::fprintf(
            stderr,
            "[ROPE-TPROF-PREFILL] layer lower=%.2fms (M=%u) cum=%.2fms "
            "(%d layers)\n",
            _dt, to - from, _pf_acc, _pf_n);
          std::fflush(stderr);
        } else {
          _rope_acc += _dt;
          if (++_rope_layers >= 35) {
            std::fprintf(stderr,
                         "[ROPE-TPROF-DECODE] host lower_q+lower_kv = %.2f "
                         "ms/token (%d layers)\n",
                         _rope_acc, _rope_layers);
            std::fflush(stderr);
            _rope_acc = 0.0;
            _rope_layers = 0;
          }
        }
      }
      // apply rotary embedding for query.
      //
      // [skip-prefill] ...except on a KV-shared layer's prefill step. Such a
      // layer returns below as soon as K/V are in its cache, because its
      // attention output is unused, and the rotated query is read by NOTHING
      // ELSE -- only that attention. So the rotation is pure waste there.
      // Property-gated (skip_prefill) and prefill-only: no other layer, step
      // or engine changes behaviour. K/V still rotate and scatter -- the cache
      // is the whole point of the layer.
      const bool skip_q_rope = skip_prefill && (to - from) > 1;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
      // GPU RoPE (decode, device-resident query): split-half rotation on the
      // device matching apply_rotary_emb_tensor_v2, keeping the query off the
      // host. Opt-in (NNTR_CUDA_ROPE) until the whole decode chain is on-GPU.
      bool q_rope_gpu = false;
      {
        static const bool cuda_rope = nntr_env_on("NNTR_CUDA_ROPE");
        if (cuda_rope &&
            query_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
          if (cached_freqs_cos_fp16 == nullptr ||
              cached_freqs_sin_fp16 == nullptr) {
            const std::lock_guard<std::mutex> lock(rope_init_mtx);
            // Cap the RoPE trig table to the live max sequence length
            // (rope_lut_positions() = MaxTimestep ~= max_seq_len) instead of
            // the model's max_position_embeddings (131072 for gemma4). The
            // table is already shared across layers via precompute_freqs'
            // rope_freq_cache (only the 2 distinct (head_dim, theta) configs
            // are built), but each build was the full 128K positions => ~376ms
            // of prefill host time for the 2 trig builds + ~113ms flatten/H2D,
            // when prefill only touches <=max_seq_len positions. The OpenCL
            // GPU-RoPE path already caps via rope_lut_positions(); this mirrors
            // it for the CUDA path.
            precompute_freqs(head_dim, rope_lut_positions(), theta, true);
            cached_freqs_cos_fp16 = freqs_cos_fp16;
            cached_freqs_sin_fp16 = freqs_sin_fp16;
          }
          const unsigned int nrows = query_step.height();
          if ((size_t)cache_index + nrows <= (*cached_freqs_cos_fp16).size()) {
            unsigned short *q =
              reinterpret_cast<unsigned short *>(query_step.getData<_FP16>());
            const bool dev = nntrainer::cuda::dev_accessible(q);
            if (dev) {
              const int half = head_dim / 2;
              const unsigned short *cosd =
                rope_lut_device(cached_freqs_cos_fp16, half);
              const unsigned short *sind =
                rope_lut_device(cached_freqs_sin_fp16, half);
              if (cosd && sind && skip_q_rope) {
                // [skip-prefill] everything above is warm-up that must keep
                // happening on the prefill step -- the trig table build and
                // its one-time device upload. Only the rotation itself is
                // dropped, and only for this layer's dead prefill query.
                q_rope_gpu = true;
              } else if (cosd && sind) {
                // M2-B: read the RoPE position from the device d_pos buffer so
                // a captured decode graph stays valid across tokens. Set d_pos
                // here only when NOT capturing (non-graph decode); under
                // capture the neuralnet scaffold sets it once per token before
                // the replay.
                static const bool m2b = nntr_env_on("NNTR_CUDA_M2B");
                if (m2b) {
                  if (!nntrainer::cuda::StreamManager::Global().isCapturing())
                    nntrainer::cuda::cuda_set_pos((int)cache_index,
                                                  (int)cache_index + 1);
                  q_rope_gpu = nntrainer::cuda::cuda_rope_fp16_dpos(
                    q, q, cosd, sind, query_step.width() / head_dim, head_dim,
                    (int)nrows, /*out_slot_dpos=*/0);
                } else {
                  q_rope_gpu = nntrainer::cuda::cuda_rope_fp16(
                    q, q, cosd, sind, query_step.width() / head_dim, head_dim,
                    (int)nrows, (int)cache_index);
                }
              }
            }
          }
        }
      }
      if (!q_rope_gpu) {
        // host RoPE fallback (prefill height>1): sync first so the host read of
        // GPU-produced q is coherent under NNTR_CUDA_ASYNC.
        nntrainer::cuda::drain_if_async();
#endif
        apply_rotary_emb_tensor_v2(query_step, query_step, head_dim,
                                   cache_index, true);
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
      }
#endif

      // append kcache with rotary embedding. §3.8 OHWI write path: when
      // enabled and on the FP16 cache path, rotate K in-place on key_step then
      // scatter per-head into the OHWI-laid-out cache buffer. Otherwise stick
      // to the original concat write.
      const bool kv_ohwi_active =
        is_kv_ohwi_enabled() && !kv_int8 &&
        key_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        cache_key.getDataType() == ml::train::TensorDim::DataType::FP16;
      if (kv_ohwi_active) {
#ifdef ENABLE_FP16
        apply_rotary_emb_tensor_v2(key_step, key_step, head_dim, cache_index,
                                   true);
        scatter_k_concat_to_ohwi_fp16(
          reinterpret_cast<const uint16_t *>(key_step.getData<_FP16>()),
          reinterpret_cast<uint16_t *>(cache_key.getData<_FP16>()), batch,
          cache_index, /*step_size*/ to - from, num_heads_KV, head_dim,
          cache_key_dim.height());
#else
        NNTR_THROW_IF(true, std::invalid_argument)
          << "NNTR_KV_OHWI requires ENABLE_FP16";
#endif
      } else {
        // GPU RoPE for K straight into the (UVM) cache slice + GPU V copy:
        // keeps the whole KV-cache write off the host. Requires NNTR_CUDA_ROPE
        // and a device-resident cache (NNTR_CUDA_KV_UVM); the dev checks gate
        // it.
        bool k_rope_gpu = false;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
        {
          static const bool cuda_rope = nntr_env_on("NNTR_CUDA_ROPE");
          auto dev = [](const void *p) {
            return nntrainer::cuda::dev_accessible(p);
          };
          const unsigned int knrows = key_step.height();
          if (cuda_rope &&
              key_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
              cached_freqs_cos_fp16 != nullptr &&
              (size_t)cache_index + knrows <= (*cached_freqs_cos_fp16).size()) {
            auto *kin =
              reinterpret_cast<unsigned short *>(key_step.getData<_FP16>());
            auto *kout = reinterpret_cast<unsigned short *>(
              b_cache_key_step.getData<_FP16>());
            if (dev(kin) && dev(kout)) {
              const int half = head_dim / 2;
              const unsigned short *cosd =
                rope_lut_device(cached_freqs_cos_fp16, half);
              const unsigned short *sind =
                rope_lut_device(cached_freqs_sin_fp16, half);
              static const bool m2b_k = nntr_env_on("NNTR_CUDA_M2B");
              if (cosd && sind && m2b_k) {
                // M2-B: write RoPE'd K into the cache at the live slot computed
                // on-device from d_pos[0] (kbase = cache BASE for this batch,
                // not the host pre-offset b_cache_key_step) -> correct slot on
                // replay.
                unsigned short *kbase =
                  reinterpret_cast<unsigned short *>(
                    cache_key.getData<_FP16>()) +
                  (size_t)batch * cache_key_dim.getFeatureLen();
                k_rope_gpu = nntrainer::cuda::cuda_rope_fp16_dpos(
                  kin, kbase, cosd, sind, key_step.width() / head_dim, head_dim,
                  (int)knrows, /*out_slot_dpos=*/1,
                  /*ring_cap=*/(int)kv_ring_cap);
              } else if (cosd && sind) {
                k_rope_gpu = nntrainer::cuda::cuda_rope_fp16(
                  kin, kout, cosd, sind, key_step.width() / head_dim, head_dim,
                  (int)knrows, (int)cache_index);
              }
            }
          }
        }
#endif
        if (!k_rope_gpu) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
          nntrainer::cuda::drain_if_async();
#if defined(ENABLE_FP16)
          NNTR_THROW_IF(
            b_cache_key_step.getMemoryData() &&
              !b_cache_key_step.getMemoryData()->isHostAddressable(),
            std::runtime_error)
            << "device-only KV (NNTR_CUDA_KV_DEV) requires GPU RoPE for K; the "
               "host fallback would fault (NNTR_CUDA_ROPE off or RoPE LUT "
               "range miss at cache_index="
            << cache_index << ")";
#endif
#endif
          apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim,
                                     cache_index, true);
        }
      }

      // append vcache without rotary embedding
      if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
        apply_rotary_emb_tensor_v2(value_step, b_cache_value_step, head_dim,
                                   cache_index, false);
      } else if (query_step.getDataType() ==
                 ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
        bool v_copy_gpu = false;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
        {
          static const bool cuda_elt = nntr_env_on("NNTR_CUDA_ROPE");
          // V-copy historically stayed host for the prefill big-step (height>1)
          // because a host attention path could read the V cache unsynced. With
          // GPU attention (NNTR_CUDA_ATTN) + UVM KV cache (NNTR_CUDA_KV_UVM),
          // prefill attention reads the cache on the GPU (same stream), so a
          // GPU V copy is a GPU->GPU handoff -- no host read, no drain. Gated
          // by NNTR_CUDA_VCOPY_PREFILL while validating; removes the per-layer
          // finishIfAsync bubble that made async-on prefill slow.
          static const bool vcopy_prefill =
            nntr_env_on("NNTR_CUDA_VCOPY_PREFILL");
          auto *vin =
            reinterpret_cast<unsigned short *>(value_step.getData<_FP16>());
          auto *vout = reinterpret_cast<unsigned short *>(
            b_cache_value_step.getData<_FP16>());
          const bool dev = nntrainer::cuda::dev_accessible(vout);
          // Device-only KV (NNTR_CUDA_KV_DEV): the host copyData fallback
          // below would dereference a cudaMalloc pointer -- always take the
          // GPU copy for a device-only destination, independent of the
          // VCOPY_PREFILL opt-in. Asked via the MemoryData residency stamp
          // (pool-bind time), not a per-call driver query -- layering rule.
          const auto v_md = b_cache_value_step.getMemoryData();
          const bool v_dev_only = v_md && !v_md->isHostAddressable();
          static const bool m2b_v = nntr_env_on("NNTR_CUDA_M2B");
          if (cuda_elt && dev &&
              (value_step.height() == 1 || vcopy_prefill || v_dev_only)) {
            if (m2b_v) {
              // M2-B: write V into the cache at the live slot d_pos[0] computed
              // on-device (vbase = cache BASE for this batch) -> correct on
              // replay.
              unsigned short *vbase =
                reinterpret_cast<unsigned short *>(
                  cache_value.getData<_FP16>()) +
                (size_t)batch * cache_value_dim.getFeatureLen();
              if (nntrainer::cuda::cuda_scalar_mul_fp16_slot(
                    vin, vbase, (unsigned int)value_step.size(), 1.0f,
                    (int)cache_value_dim.width(),
                    /*ring_cap=*/(int)kv_ring_cap))
                v_copy_gpu = true;
            } else if (nntrainer::cuda::cuda_scalar_mul_fp16(
                         vin, vout, (unsigned int)value_step.size(), 1.0f)) {
              v_copy_gpu = true;
            }
          }
        }
#endif
        if (!v_copy_gpu) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
          nntrainer::cuda::drain_if_async();
          NNTR_THROW_IF(
            b_cache_value_step.getMemoryData() &&
              !b_cache_value_step.getMemoryData()->isHostAddressable(),
            std::runtime_error)
            << "device-only KV (NNTR_CUDA_KV_DEV) requires the GPU V-copy; the "
               "host copyData fallback would fault -- check NNTR_CUDA_ELTWISE";
#endif
          b_cache_value_step.copyData(value_step);
        }
#else
        NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
      }
    }
  } else {
    // No-RoPE path: host scatters/copies read K/V on the host -- lower first.
    lower_kv();
    const bool kv_ohwi_active =
      is_kv_ohwi_enabled() && !kv_int8 &&
      key_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      cache_key.getDataType() == ml::train::TensorDim::DataType::FP16;
    if (kv_ohwi_active) {
#ifdef ENABLE_FP16
      scatter_k_concat_to_ohwi_fp16(
        reinterpret_cast<const uint16_t *>(key_step.getData<_FP16>()),
        reinterpret_cast<uint16_t *>(cache_key.getData<_FP16>()), batch,
        cache_index, /*step_size*/ to - from, num_heads_KV, head_dim,
        cache_key_dim.height());
#else
      NNTR_THROW_IF(true, std::invalid_argument)
        << "NNTR_KV_OHWI requires ENABLE_FP16";
#endif
    } else {
      b_cache_key_step.copyData(key_step);
    }
    b_cache_value_step.copyData(value_step);
  }

  /// @todo replace step_size into input height
  unsigned int step_size = to - from;
  unsigned int cache_from = cache_index;
  unsigned int cache_to = cache_from + step_size;

  // skip_prefill (Gemma4 KV-shared layers): K/V are now written + scattered
  // into this layer's own cache slab above, which is all decode needs (decode
  // re-derives cache_index from the absolute `from` each step, and attends to
  // these cached positions). The prefill attention OUTPUT is unused -- the O
  // FC and every downstream per-layer op for a KV-shared layer also skip the
  // prefill big-step -- so skip the heavy attention compute + output write.
  // Decode (step_size == 1) always runs the full path. The CLMEM output raise
  // is skipped intentionally: the downstream wo FC does not read O in prefill.
  if (skip_prefill && step_size > 1)
    return;

  // NNTR_MHA_CLMEM boundary sync (see mha_clmem_mode): gather the mirror
  // rows the slab is missing back into the concat SVM slab, capped by the
  // mirror's valid range, then drain once so the host may read immediately.
  auto sync_kv_slab = [&](unsigned int upto) {
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
    // OHWI mirror gather (k_gather_ohwi_cl / v_gather_ohwi_t_cl, cl_mem) is
    // OpenCL-only; no mirror exists without it.
    // ENABLE_FP16 is required too: the KV mirror is an FP16-only layout and the
    // gather reinterprets cache_key/cache_value as _FP16, a type that only
    // exists under ENABLE_FP16.
    if (!mha_clmem_mode || !kv_mirror_init)
      return;
    if (upto > kv_k_valid_to)
      upto = kv_k_valid_to;
    if (upto > kv_v_valid_to)
      upto = kv_v_valid_to;
    if (kv_slab_synced_to >= upto)
      return;
    const size_t hd = (size_t)num_heads_KV * head_dim;
    const unsigned int n = upto - kv_slab_synced_to;
    uint16_t *k_dst = reinterpret_cast<uint16_t *>(
      cache_key.getData<_FP16>() +
      (size_t)batch * cache_key_dim.getFeatureLen() +
      (size_t)kv_slab_synced_to * hd);
    uint16_t *v_dst = reinterpret_cast<uint16_t *>(
      cache_value.getData<_FP16>() +
      (size_t)batch * cache_value_dim.getFeatureLen() +
      (size_t)kv_slab_synced_to * hd);
    nntrainer::k_gather_ohwi_cl(reinterpret_cast<cl_mem>(k_buf_ohwi), k_dst, n,
                                num_heads_KV, head_dim, kv_mirror_S_max,
                                kv_slab_synced_to, /*drain=*/false);
    nntrainer::v_gather_ohwi_t_cl(
      reinterpret_cast<cl_mem>(v_buf_ohwi), v_dst, n, num_heads_KV, head_dim,
      kv_v_cur_stride != 0 ? kv_v_cur_stride : kv_mirror_S_max,
      kv_slab_synced_to, /*drain=*/true);
    kv_slab_synced_to = upto;
#else
    (void)upto;
#endif // ENABLE_OPENCL (sync_kv_slab)
  };

  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  // [kv-window-ring] The cache buffer is only kv_ring_cap rows, so the read
  // VIEW must fit it. The GPU flash kernel walks the LOGICAL range (N_kv =
  // cache_to, passed separately) and modulo-maps to the physical ring, so a
  // Wcap-high view is sufficient. (Ring off: full cache_to, unchanged.)
  const unsigned int read_rows =
    kv_ring_cap ? std::min<unsigned int>(cache_to, kv_ring_cap) : cache_to;
  cached_key_dim.height(read_rows);
  cached_value_dim.height(read_rows);

  // §3.8 Phase 2: OHWI-direct GPU prefill. When both NNTR_KV_OHWI=1 and
  // NNTR_MHA_GPU=1 are set, the K cache is already laid out as
  // [H_kv, S_max, d] (Phase 1 write) and we can dispatch the
  // qk_matmul_f16_ohwi kernel against the raw SVM cache_key buffer
  // directly — skipping the Phase 1 gather entirely. V is still concat
  // (Phase 3 will move V); sv_matmul_f16 is reused. Returns early on
  // success; on failure falls through to the Phase 1 gather + concat
  // path below. Opt-in by both env vars together; no broken-gate.
  {
    const unsigned int FLASH_MIN_PREFILL =
      min_prefill_thr((unsigned int)head_dim); // env NNTR_MIN_PREFILL
    static const bool _ohwi_gpu_on =
      is_kv_ohwi_enabled() && !kv_int8 && nntr_env_on("NNTR_MHA_GPU");
    // NNTR_MHA_GPU_DECODE=1 also routes DECODE (step_size==1) through this
    // OHWI image-attention path. The decode CPU NEON path (compute_kcaches) is
    // the long-context bottleneck; the OHWI qk/sv image kernels + KV scatter
    // already handle M=1. Falls through to CPU on not-ok.
    // (B) flash/OHWI decode-attention gate. Enabled by the NNTR_MHA_GPU_DECODE
    // env flag (global testing override) OR per-LAYER via the gpu_decode_attn
    // property (DEFAULT-ON for models where decode flash attention is
    // token-identical, e.g. gemma4, gemma2). Env read stays static; property
    // read PER-CALL and OR'd in.
    static const bool _ohwi_decode_env =
      std::getenv("NNTR_MHA_GPU_DECODE") != nullptr;
    const bool _ohwi_decode_on =
      _ohwi_decode_env || std::get<props::GpuDecodeAttn>(mha_core_props).get();
    const unsigned int step_size_p2 = to - from;
    const unsigned int cache_to_p2 = cache_index + step_size_p2;
    if (_ohwi_gpu_on && use_gemm_attention &&
        (step_size_p2 >= FLASH_MIN_PREFILL || _ohwi_decode_on) &&
        query_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        cache_key.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        attention_output_step.getDataType() ==
          ml::train::TensorDim::DataType::FP16 &&
        head_dim > 0 && num_heads_KV > 0 && num_heads_Q % num_heads_KV == 0) {
#ifdef ENABLE_FP16
#if defined(ENABLE_OPENCL) // OHWI-direct GPU prefill attention
      const uint16_t *Q_p =
        reinterpret_cast<const uint16_t *>(query_step.getData<_FP16>());
      uint16_t *O_p =
        reinterpret_cast<uint16_t *>(attention_output_step.getData<_FP16>());
      const size_t kv_per_batch =
        (size_t)num_heads_KV * cache_key_dim.height() * head_dim;
      const uint16_t *K_ohwi =
        reinterpret_cast<const uint16_t *>(cache_key.getData<_FP16>()) +
        (size_t)batch * kv_per_batch;
      const uint16_t *V_concat =
        reinterpret_cast<const uint16_t *>(cache_value.getData<_FP16>()) +
        (size_t)batch * cache_value_dim.getFeatureLen();
      const bool svm_ok =
        query_step.getMemoryData() && cache_key.getMemoryData() &&
        cache_value.getMemoryData() && attention_output_step.getMemoryData() &&
        query_step.getMemoryData()->isSVM() &&
        cache_key.getMemoryData()->isSVM() &&
        cache_value.getMemoryData()->isSVM() &&
        attention_output_step.getMemoryData()->isSVM();
      // SVM-gated by default. With the current KVCacheManager (plain
      // host-tensor cache), svm_ok is false and the non-SVM wrapper
      // path would upload the full H_kv*S_max*d slab per layer per
      // step (4MB at S_max=2048, ~7x more than the live N_kv slice
      // would need). Phase 2.5 will allocate the KV cache through the
      // SVM allocator; until then, opting into NNTR_KV_OHWI_GPU_FORCE=1
      // exercises the non-SVM path (slow + drift-prone, for kernel
      // validation only).
      static const bool _ohwi_force =
        std::getenv("NNTR_KV_OHWI_GPU_FORCE") != nullptr;
      static int _ohwi_logged = 0;
      static int _ohwi_dec_logged = 0;
      const bool _is_dec = (step_size_p2 == 1);
      if (!_ohwi_logged || (_is_dec && _ohwi_dec_logged < 3)) {
        _ohwi_logged = 1;
        if (_is_dec)
          ++_ohwi_dec_logged;
        ml_logd("[OHWI-P2] %s M=%u N_kv=%u S_max=%zu H_q=%zu H_kv=%zu "
                "d=%zu svm=%d force=%d qSVM=%d kSVM=%d vSVM=%d oSVM=%d",
                _is_dec ? "DECODE" : "PREFILL", step_size_p2, cache_to_p2,
                cache_key_dim.height(), num_heads_Q, num_heads_KV, head_dim,
                (int)svm_ok, (int)_ohwi_force,
                (int)(query_step.getMemoryData() &&
                      query_step.getMemoryData()->isSVM()),
                (int)(cache_key.getMemoryData() &&
                      cache_key.getMemoryData()->isSVM()),
                (int)(cache_value.getMemoryData() &&
                      cache_value.getMemoryData()->isSVM()),
                (int)(attention_output_step.getMemoryData() &&
                      attention_output_step.getMemoryData()->isSVM()));
      }
      if (svm_ok || _ohwi_force) {
        const unsigned int win_p2 = (local_window_size >= (size_t)cache_to_p2)
                                      ? 0u
                                      : (unsigned int)local_window_size;
        bool ok = nntrainer::two_conv_attention_prefill_f16_ohwi_cl(
          Q_p, K_ohwi, V_concat, O_p, step_size_p2, cache_to_p2, num_heads_Q,
          num_heads_KV, head_dim, cache_key_dim.height(), is_causal,
          /*svm_inputs=*/svm_ok, win_p2);
        if (ok)
          return;
      }
#endif // ENABLE_OPENCL (OHWI-direct GPU prefill)
#endif
    }
  }

  // §3.8 OHWI Phase 1 read path: when OHWI is on, the cache_key buffer is
  // stored in [B, H_kv, S, D] order. Existing downstream readers (CPU
  // gemm_attention / compute_kcaches / GPU two_conv_attention) all expect
  // the concat layout [B, 1, S, H_kv*D]. Gather into a fresh scratch tensor
  // so no downstream code needs to change. The fresh tensor is non-SVM so
  // the GPU prefill SVM check below will naturally fall through to CPU —
  // intended in Phase 1; Phase 2's OHWI-direct path above handles the
  // success case for the OHWI+MHA_GPU combo.
  const bool kv_ohwi_read_active =
    is_kv_ohwi_enabled() && !kv_int8 &&
    cache_key.getDataType() == ml::train::TensorDim::DataType::FP16;
  nntrainer::Tensor b_cached_key;
  if (kv_ohwi_read_active) {
#ifdef ENABLE_FP16
    ml::train::TensorDim per_batch_key_dim = cached_key_dim;
    per_batch_key_dim.batch(1);
    b_cached_key = nntrainer::Tensor(per_batch_key_dim, true);
    gather_k_ohwi_to_concat_fp16(
      reinterpret_cast<const uint16_t *>(cache_key.getData<_FP16>()),
      reinterpret_cast<uint16_t *>(b_cached_key.getData<_FP16>()), batch,
      cache_to, num_heads_KV, head_dim, cache_key_dim.height());
#else
    NNTR_THROW_IF(true, std::invalid_argument)
      << "NNTR_KV_OHWI requires ENABLE_FP16";
#endif
  } else {
    b_cached_key = cache_key.getSharedDataTensor(
      cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  }
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  // Optional flash GEMM attention path. Handles both non-causal (encoder)
  // and causal-prefill paths, supports GQA and sliding window. Gated on a
  // minimum prefill length: for decode (step_size == 1) the per-row dot
  // path is preferred (no benefit from blocking + softmax bookkeeping).
  const unsigned int FLASH_MIN_PREFILL =
    min_prefill_thr((unsigned int)head_dim); // env NNTR_MIN_PREFILL
  // S3 (decode GPU attention): NNTR_MHA_GPU_DECODE lets step_size==1 (decode)
  // enter the GPU attention block too. The KV-image (kvimg_view) path scatters
  // the single new token into the mirrors and reads the full N_kv context, so
  // M=1 is handled; otherwise decode falls to the host compute_kcaches NEON
  // path (a per-layer GPU->host queue drain). Pays off only with the host
  // RoPE/v_norm drains also removed (land S1+S2+S3 as a SET).
  // (B) flash decode-attention gate. Enabled by NNTR_MHA_GPU_DECODE env
  // (global testing override) OR per-LAYER via the gpu_decode_attn property
  // (DEFAULT-ON for gemma4 / gemma2). Env read static; property read PER-CALL.
  static const bool _mha_gpu_decode_env =
    std::getenv("NNTR_MHA_GPU_DECODE") != nullptr;
  const bool _mha_gpu_decode =
    _mha_gpu_decode_env || std::get<props::GpuDecodeAttn>(mha_core_props).get();
  // CUDA GPU attention (NNTR_CUDA_ATTN) handles any step_size via
  // gemm_attention
  // -> cuda_attention. Route ALL step_size to it so the short-prefill window
  // (2..FLASH_MIN_PREFILL-1) does not fall to the host compute_kcaches path,
  // which faults on a device-only activation pool (NNTR_CUDA_DEV_ACT) and was
  // the short-prompt crash. OpenCL is unaffected (NNTR_CUDA_ATTN is cuda-only).
  static const bool _cuda_attn_on = nntr_env_on("NNTR_CUDA_ATTN");
  if (use_gemm_attention &&
      (step_size >= FLASH_MIN_PREFILL || (_mha_gpu_decode && step_size == 1) ||
       _cuda_attn_on)) {
    // GPU two-1x1-conv attention path (paper section 3.7). Env-gated via
    // NNTR_MHA_GPU=1. FP16-Q + FP16-out only; K/V is either FP16 or, when
    // kv_int8 is set, int8 + per-(token, head) FP16 scale. Falls back to
    // the CPU path on any shape mismatch.
    static const bool _tca_on = nntr_env_on("NNTR_MHA_GPU");
    if (_tca_on &&
        query_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        attention_output_step.getDataType() ==
          ml::train::TensorDim::DataType::FP16 &&
        head_dim > 0 && num_heads_KV > 0 && num_heads_Q % num_heads_KV == 0) {
#ifdef ENABLE_FP16
#if defined(ENABLE_OPENCL) // GPU two-conv/image/flash attention path
      const uint16_t *Q_p =
        reinterpret_cast<const uint16_t *>(query_step.getData<_FP16>());
      uint16_t *O_p =
        reinterpret_cast<uint16_t *>(attention_output_step.getData<_FP16>());
      const bool svm_ok = query_step.getMemoryData() &&
                          b_cached_key.getMemoryData() &&
                          b_cached_value.getMemoryData() &&
                          attention_output_step.getMemoryData() &&
                          query_step.getMemoryData()->isSVM() &&
                          b_cached_key.getMemoryData()->isSVM() &&
                          b_cached_value.getMemoryData()->isSVM() &&
                          attention_output_step.getMemoryData()->isSVM();
      bool ok = false;
      // int8 GPU prefill is wired but currently produces degraded output
      // (model generates early-EOS); kept behind a second env var
      // (NNTR_KV_INT8_GPU=1) until the numerical issue is isolated. Default
      // for kv_int8 + NNTR_MHA_GPU is to fall through to the CPU NEON path
      // which is verified working.
      static const bool _tca_int8_on =
        std::getenv("NNTR_KV_INT8_GPU") != nullptr;
      if (kv_int8 && _tca_int8_on) {
        const int8_t *K_i8 =
          reinterpret_cast<const int8_t *>(b_cached_key.getData<uint8_t>());
        const int8_t *V_i8 =
          reinterpret_cast<const int8_t *>(b_cached_value.getData<uint8_t>());
        NNTR_THROW_IF(cur_kv_int8_key_scale_batch == nullptr ||
                        cur_kv_int8_value_scale_batch == nullptr,
                      std::invalid_argument)
          << "kv_int8 GPU prefill missing per-batch scale pointers";
        ok = nntrainer::two_conv_attention_prefill_f16_kvi8_cl(
          Q_p, K_i8, V_i8, cur_kv_int8_key_scale_batch,
          cur_kv_int8_value_scale_batch, O_p, step_size, cache_to, num_heads_Q,
          num_heads_KV, head_dim, is_causal, svm_ok);
      } else if (kv_int8) {
        // kv_int8 + GPU prefill not enabled: fall through to CPU path
        // (caller's `gemm_attention` below dequantizes on the fly).
        ok = false;
      } else {
        const uint16_t *K_p =
          reinterpret_cast<const uint16_t *>(b_cached_key.getData<_FP16>());
        const uint16_t *V_p =
          reinterpret_cast<const uint16_t *>(b_cached_value.getData<_FP16>());

        // Adreno image attention (gpu_native's ~9x prefill path, §3.7/§3.8).
        // Read K/V via image2d_from_buffer (read_imageui texture cache) instead
        // of the SVM buffer flash kernel. The SVM KV cache can't back an image
        // (clCreateImage needs a cl_mem handle), so keep per-layer cl_mem OHWI
        // mirrors (lazy-init), scatter this step's rotated K / raw V from the
        // SVM cache slice into them, then attention reads the images. Gated by
        // NNTR_KV_IMG_ATTN (Adreno only — read_imageui won't build on Intel
        // NEO, which keeps the flash path below). Preempts flash on success.
        if (use_image_attn < 0) {
          // Value-checked so NNTR_KV_IMG_ATTN=0 really disables the image
          // path (the Adreno auto-default in cl_context uses overwrite=0 and
          // cannot override a user-provided 0).
          const char *e = std::getenv("NNTR_KV_IMG_ATTN");
          use_image_attn = (e != nullptr && std::atoi(e) != 0) ? 1 : 0;
        }
        // Sliding-window layers past their window must NOT take the image
        // path: qk_matmul_f16_ohwi_img has only the causal upper-bound mask
        // (n > q_off + m) and no window lower bound (n + W <= m), so once
        // cache_to exceeds the window it silently computes full causal
        // attention over evicted keys (gemma4 W=512: 999-tok Adreno prefill
        // degenerates into word salad, severity ~ (cache_to - W)). Route
        // those calls to the flash kernels below, which take local_window.
        // The OHWI image kernels take a window but not a ring capacity, so a
        // ringed cache must not reach them (see mha_ring_refuses_arm).
        if (use_image_attn == 1 && svm_ok && !kv_int8 && head_dim % 8 == 0 &&
            !mha_ring_refuses_arm(kv_ring_cap, "ohwi-image")) {
          // NNTR_KV_MIRROR_CAP (experiment): clamp the OHWI mirror S_max.
          // gpu_native runs S_max=1024 and its qk_matmul_f16_ohwi_img is
          // 4.7x faster than ours at S_max=2048 (59.8 vs ~281ms at M=1024,
          // same kernel/wrapper) -- isolates whether the K-image height
          // (hKV*S_max rows) is the texture-cache culprit. NOT safe for
          // sequences beyond the cap (attention reads garbage rows).
          static const unsigned int mirror_cap = []() {
            const char *e = std::getenv("NNTR_KV_MIRROR_CAP");
            return e ? (unsigned int)std::atoi(e) : 0u;
          }();
          unsigned int S_max = (cache_key_dim.height() + 7u) & ~7u;
          if (mirror_cap >= 8 && mirror_cap < S_max)
            S_max = (mirror_cap + 7u) & ~7u;
          if (!kv_mirror_init) {
            kv_mirror_S_max = S_max;
            bool m_ok = nntrainer::create_ohwi_kv_mirror(
                          /*is_v=*/false, num_heads_KV, head_dim, S_max,
                          reinterpret_cast<cl_mem *>(&k_buf_ohwi),
                          reinterpret_cast<cl_mem *>(&k_image_ohwi)) &&
                        nntrainer::create_ohwi_kv_mirror(
                          /*is_v=*/true, num_heads_KV, head_dim, S_max,
                          reinterpret_cast<cl_mem *>(&v_buf_ohwi),
                          reinterpret_cast<cl_mem *>(&v_image_ohwi));
            kv_mirror_init = m_ok;
            if (!m_ok)
              use_image_attn = 0; // permanent disable; flash takes over
          }
          if (kv_mirror_init && S_max == kv_mirror_S_max &&
              cache_to <= kv_mirror_S_max) {
            // --- Tight-stride V image (texture-cache cliff). The sv kernel
            // walks V texels along the sequence axis; a pitch sized to the
            // 2048 allocation cap instead of the live sequence wastes the
            // texture cache on padding (sv_matmul 63 -> 41ms at M=843 tight).
            // The view shares v_buf_ohwi; only the scatter stride must match
            // (the sv kernels never address V via their S_max argument).
            // NNTR_KV_VTIGHT=0 restores the full-stride image.
            static const bool v_tight_on = []() {
              const char *e = std::getenv("NNTR_KV_VTIGHT");
              return !(e && e[0] == '0');
            }();
            void *v_img_use = v_image_ohwi;
            unsigned int v_stride = kv_mirror_S_max;
            if (v_tight_on) {
              unsigned int need = (cache_to + 7u) & ~7u;
              if (need > kv_v_img_S) {
                void *nimg = nullptr;
                // The helper rounds `need` up to the device image pitch
                // alignment (Adreno: multiples of 256 halves) and reports the
                // stride actually used; only adopt it while it still fits
                // the full-capacity buffer (else the full image is as good).
                if (nntrainer::create_ohwi_v_image_view(
                      v_buf_ohwi, num_heads_KV, head_dim, &need, &nimg)) {
                  if (need < kv_mirror_S_max) {
                    if (v_image_tight)
                      nntrainer::release_cl_mem(v_image_tight);
                    v_image_tight = nimg;
                    kv_v_img_S = need;
                  } else {
                    nntrainer::release_cl_mem(nimg);
                  }
                }
              }
              if (v_image_tight != nullptr && cache_to <= kv_v_img_S) {
                v_img_use = v_image_tight;
                v_stride = kv_v_img_S;
              }
            }
            // --- Mirror content repair (stride change / decode gap). The
            // SVM K/V caches are complete (RoPE side-fill + decode host
            // writes), so re-scatter whatever the mirrors are missing:
            // a V stride change invalidates ALL prior rows; decode tokens
            // land only in the SVM cache, leaving [valid_to, cache_from)
            // missing from both mirrors on a follow-up prefill.
            if (v_stride != kv_v_cur_stride) {
              if (cache_from > 0) {
                // MHA_CLMEM: the re-scatter source is the concat slab --
                // gather the mirror-only rows back first (no-op otherwise).
                sync_kv_slab(cache_from);
                nntrainer::v_scatter_ohwi_t_cl(
                  reinterpret_cast<const uint16_t *>(
                    cache_value.getData<_FP16>() +
                    (size_t)batch * cache_value_dim.getFeatureLen()),
                  reinterpret_cast<cl_mem>(v_buf_ohwi), cache_from,
                  num_heads_KV, head_dim, v_stride, 0);
              }
              kv_v_cur_stride = v_stride;
              kv_v_valid_to = cache_from;
            } else if (cache_from > kv_v_valid_to) {
              nntrainer::v_scatter_ohwi_t_cl(
                reinterpret_cast<const uint16_t *>(
                  cache_value.getData<_FP16>() +
                  (size_t)batch * cache_value_dim.getFeatureLen() +
                  (size_t)kv_v_valid_to * num_heads_KV * head_dim),
                reinterpret_cast<cl_mem>(v_buf_ohwi),
                cache_from - kv_v_valid_to, num_heads_KV, head_dim, v_stride,
                kv_v_valid_to);
              kv_v_valid_to = cache_from;
            }
            if (cache_from > kv_k_valid_to) {
              nntrainer::k_scatter_ohwi_cl(
                reinterpret_cast<const uint16_t *>(
                  cache_key.getData<_FP16>() +
                  (size_t)batch * cache_key_dim.getFeatureLen() +
                  (size_t)kv_k_valid_to * num_heads_KV * head_dim),
                reinterpret_cast<cl_mem>(k_buf_ohwi),
                cache_from - kv_k_valid_to, num_heads_KV, head_dim,
                kv_mirror_S_max, kv_k_valid_to);
              kv_k_valid_to = cache_from;
            }
            // Scatter this step's rotated K into the OHWI K mirror at row
            // cache_from, and V into the reversed-OHWI V mirror at column
            // cache_from. Staged chain (kv_stage_on): K reads the cl_mem
            // staging temp the GPU RoPE wrote (src_clmem) and V reads
            // value_step directly -- pure kernel chains, no host drain.
            // Unstaged: both read the SVM cache slices the (drained)
            // RoPE/copy wrote. In-order queue (NNTR_GPU_SVM_POOL) keeps
            // RoPE -> scatter -> attention ordered with no explicit sync.
            const double _kvst_t1 = _kvst_on() ? _kvst_now() : 0;
            nntrainer::k_scatter_ohwi_cl(
              reinterpret_cast<const uint16_t *>(
                b_cache_key_step.getData<_FP16>()),
              reinterpret_cast<cl_mem>(k_buf_ohwi), step_size, num_heads_KV,
              head_dim, kv_mirror_S_max, cache_from, /*src_clmem=*/k_stage,
              /*src_off=*/0u);
            kv_k_valid_to = cache_to;
            const double _kvst_tk = _kvst_on() ? _kvst_now() : 0;
            nntrainer::v_scatter_ohwi_t_cl(
              v_stage_svm != nullptr ? v_stage_svm
                                     : reinterpret_cast<const uint16_t *>(
                                         b_cache_value_step.getData<_FP16>()),
              reinterpret_cast<cl_mem>(v_buf_ohwi), step_size, num_heads_KV,
              head_dim, v_stride, cache_from,
              /*src_clmem=*/v_stage_clmem,
              /*src_off=*/0u);
            kv_v_valid_to = cache_to;
            const double _kvst_tv = _kvst_on() ? _kvst_now() : 0;
            // S3 decode: OHWI rotates Q on the HOST (query_step SVM, in-place);
            // q_attn_clmem (= q_cl, the FC output cl_mem) is NOT rotated
            // because the GPU-RoPE staging path is gated off for OHWI. Binding
            // it would feed the qk kernel an UNROTATED Q -> degenerate decode
            // attention. For decode (step_size==1) pass null so the kernel
            // reads Q_p (the SVM query). Prefill: the concat GPU-RoPE now
            // rotates straight into the SVM plane Q_p on the image path
            // (out_clmem=null above), so the qk must ALSO read Q_p (null) --
            // binding q_cl would feed the UN-rotated FC output. Only the
            // (opt-in) staged path writes a separate cl_mem handle the qk
            // should bind (q_rope_staged).
            void *q_clmem_use = (step_size == 1 || q_rope_staged == nullptr)
                                  ? nullptr
                                  : q_attn_clmem;
            // Sliding-window layers: pass the effective window so the OHWI
            // kernels mask keys older than the window (n + W <= q_pos) —
            // same convention as the flash call below (0 = no window).
            const unsigned int win_img = (local_window_size >= (size_t)cache_to)
                                           ? 0u
                                           : (unsigned int)local_window_size;
            ok = nntrainer::two_conv_attention_prefill_f16_ohwi_kvimg_view_cl(
              Q_p, reinterpret_cast<cl_mem>(k_image_ohwi),
              reinterpret_cast<cl_mem>(v_img_use), O_p, step_size, cache_to,
              num_heads_Q, num_heads_KV, head_dim, kv_mirror_S_max, is_causal,
              attn_logit_softcapping, /*q_clmem=*/q_clmem_use,
              /*o_clmem=*/o_cl, win_img);
            if (_kvst_on())
              _kvst_mark_scatter(_kvst_t1, _kvst_tk, _kvst_tv, _kvst_now());
            if (ok && o_cl != nullptr)
              o_written_clmem = true;
            static int _img_attn_logged = 0;
            if (!_img_attn_logged) {
              _img_attn_logged = 1;
              ml_logd("[IMG-ATTN] engaged ok=%d M=%u N_kv=%u S_max=%u "
                      "hQ=%zu hKV=%zu d=%zu softcap=%.1f",
                      (int)ok, step_size, cache_to, kv_mirror_S_max,
                      num_heads_Q, num_heads_KV, head_dim,
                      attn_logit_softcapping);
            }
          }
        }

        // gpu_native's proven d=256 attention: register-tiled, barrier-free,
        // scores-free flash (FBQ_SG Block-Q when NNTR_V8C_BUF). This is the
        // SAME public kernel gpu_native uses for the ~875-TPS Intel path
        // (qwen3_forward.cpp). The layer-graph previously only had the scalar
        // two_conv path (materializes a [hQ,M,N_kv] fp16 scores tensor in DRAM,
        // ~820ms at M=1024 d=256) and otherwise fell to host gemm_attention
        // (~8.5s, M=1024). Concat K (non-OHWI b_cached_key) => max_seq_len=0.
        // SVM-only; falls through to two_conv/host when svm_ok is false.
        // The SVM-pointer fallbacks below read Q via its SVM shadow: lower
        // first when Q is GPU_CLMEM-resident and the image path did not run.
        if (!ok && q_rope_staged != nullptr) {
          // The rotated Q lives only in the q_stage temp; the SVM fallbacks
          // below read Q's SVM plane. Land it there (drained) and drop the
          // (stale, un-rotated) cl_mem handle so lower_q stays a no-op.
          nntrainer::gpu_copy_f16_cl(
            reinterpret_cast<const uint16_t *>(query_step.getData<_FP16>()),
            reinterpret_cast<uint16_t *>(query_step.getData<_FP16>()),
            step_size * (unsigned int)num_heads_Q * (unsigned int)head_dim,
            /*svm_inputs=*/true, /*in_clmem=*/q_rope_staged,
            /*out_clmem=*/nullptr, /*drain=*/true);
          q_cl = nullptr;
        }
        if (!ok &&
            (kv_stage_on || k_stage != nullptr || v_stage_svm != nullptr)) {
          // The staged K/V cache side-fills (and, in staged mode, the
          // undrained rope-Q SVM output) were never drained; the
          // flash/two_conv SVM readers below depend on them. One drain.
          nntrainer::cl_queue_finish();
        }
        // MHA_CLMEM: the SVM fallbacks below read the concat slab, which
        // the mirror-only prefill window left stale -- gather it back.
        if (!ok)
          sync_kv_slab(cache_to);
        if (!ok)
          lower_q();
        if (!ok && svm_ok) {
          // Gemma4 sliding-window: pass local_window_size so the flash kernel
          // masks keys older than the window (n + W <= m). UINT_MAX (full
          // attention) and any window >= cache length are treated as "no
          // window" inside the kernel. Prefill big-step writes from
          // cache_from==0 so the kernel's row index m == absolute query pos.
          const unsigned int win = (local_window_size >= (size_t)cache_to)
                                     ? 0u
                                     : (unsigned int)local_window_size;
          // Decode (step_size==1): single query starves blockq/coop_vec
          // (num_heads_Q groups). flash_decode splits the KV axis into chunks
          // (num_heads_Q * n_chunks groups) for parallelism. Falls back to the
          // prefill flash kernel on shape mismatch / softcap.
          if (step_size == 1)
            ok = nntrainer::flash_decode_f16_cl(
              Q_p, K_p, V_p, O_p, cache_to, num_heads_Q, num_heads_KV, head_dim,
              /*max_seq_len=*/0u, /*svm_inputs=*/true, attn_logit_softcapping,
              /*local_window=*/win, /*ring_cap=*/kv_ring_cap);
          if (!ok)
            ok = nntrainer::flash_attention_prefill_f16_cl(
              Q_p, K_p, V_p, O_p, step_size, cache_to, num_heads_Q,
              num_heads_KV, head_dim, /*max_seq_len=*/0u, is_causal,
              /*svm_inputs=*/true, attn_logit_softcapping, /*local_window=*/win,
              /*ring_cap=*/kv_ring_cap);
        }
        // image2d_from_buffer variant gated by NNTR_MHA_GPU_IMG=1. Uses 8-half
        // texel loads (RGBA UINT32) for the d-axis reduction, ~8x fewer
        // memory transactions vs the scalar fp16 path. Falls back to scalar
        // on any failure. Non-SVM only (image2d_from_buffer needs cl_mem).
        static const bool _tca_img_on =
          std::getenv("NNTR_MHA_GPU_IMG") != nullptr;
        if (!ok && _tca_img_on && head_dim % 8 == 0 &&
            (num_heads_Q * head_dim) % 8 == 0 &&
            (num_heads_KV * head_dim) % 8 == 0 &&
            !mha_ring_refuses_arm(kv_ring_cap, "two_conv-image")) {
          ok = nntrainer::two_conv_attention_prefill_f16_img_cl(
            Q_p, K_p, V_p, O_p, step_size, cache_to, num_heads_Q, num_heads_KV,
            head_dim, is_causal);
        }
        if (!ok && !mha_ring_refuses_arm(kv_ring_cap, "two_conv")) {
          ok = nntrainer::two_conv_attention_prefill_f16_cl(
            Q_p, K_p, V_p, O_p, step_size, cache_to, num_heads_Q, num_heads_KV,
            head_dim, is_causal, svm_ok);
        }
      }
      if (ok) {
        // The wo FC consumes O through its static plane: raise it into the
        // planner sub-buffer when a non-cl_mem path (flash/two_conv/kvi8)
        // wrote the SVM shadow instead.
        if (o_cl != nullptr && !o_written_clmem) {
          // The flash/two_conv prefill kernel wrote O into the SVM shadow and,
          // on the in-order SVM-pool queue, deliberately SKIPS its post-kernel
          // drain (attention_kernels.cpp) on the assumption that the next GPU
          // op orders behind it. But clmem_raise_cl below is NOT a GPU->GPU
          // dependency: it reads the SVM shadow as a plain HOST pointer
          // (the clEnqueueWriteBuffer source). The host view of a kernel-
          // written coarse-grained SVM region is not coherent until the
          // kernel completes, so at large prefill M (sliding-window prefill at
          // ctx > window is the first config that runs a long flash kernel on
          // this path) the raise can snapshot a partially-written O => corrupt
          // last-position hidden state => non-deterministic immediate EOS /
          // garbage. Drain once here so the host source is coherent before the
          // raise. Measured no prefill regression (M=1024 ~737 TPS, identical
          // to the undrained path within noise) because the flash kernel has
          // effectively completed by the time this layer's O is consumed; the
          // host gemm path below wrote O on the host and needs no such drain.
          // GPU-side O raise (o_svm -> o_cl, GPU->GPU on the in-order queue):
          // the host clmem_raise_cl reads the O SVM shadow as a HOST pointer
          // (clEnqueueWriteBuffer source), which is NOT coherent on
          // coarse-grain SVM (Xe3 / Panther Lake) even after cl_queue_finish --
          // a finish drains the queue but does NOT map the SVM for host access,
          // so the raise snapshots a stale/partially-written O. A short prefill
          // (M~4) exposes this: the flash completes just before the raise ->
          // corrupt last-position hidden -> garbage (large-M survived because
          // the flash finished long before the raise). The GPU copy needs no
          // host coherence and is perf-neutral (no drain, no host bounce).
          // DEFAULT-ON; override with NNTR_NO_GPU_OBRIDGE for the legacy
          // host-raise path.
          static const bool _gpu_obridge =
            std::getenv("NNTR_NO_GPU_OBRIDGE") == nullptr;
          if (_gpu_obridge) {
            uint16_t *o_svm = reinterpret_cast<uint16_t *>(
              attention_output_step.getData<_FP16>());
            nntrainer::gpu_copy_f16_cl(
              o_svm, o_svm, (unsigned int)attention_output_step.size(),
              /*svm_inputs=*/true, /*in_clmem=*/nullptr, /*out_clmem=*/o_cl,
              /*drain=*/false);
          } else {
            nntrainer::cl_queue_finish();
            nntrainer::clmem_raise_cl(attention_output_step, 0);
          }
        }
        return;
      }
#endif // ENABLE_OPENCL (GPU two-conv/image/flash attention)
#endif
    }
    // Host/NEON attention reads Q and writes O on the host.
    mha_ring_assert_host_path_ok(kv_ring_cap, "host prefill attention");
    lower_q();
    sync_kv_slab(cache_from);
    if (mha_clmem_mode && cache_to > kv_slab_synced_to)
      kv_slab_synced_to = cache_to; // this step's rows were host-written
    gemm_attention(query_step, b_cached_key, b_cached_value,
                   attention_output_step, cache_to, step_size, cache_from);
    // raise host-written O into its planner cl_mem (OpenCL-only)
#if defined(ENABLE_OPENCL)
    if (o_cl != nullptr)
      nntrainer::clmem_raise_cl(attention_output_step, 0);
#endif
    return;
  }

  // Host (decode) attention reads Q and writes O on the host.
  mha_ring_assert_host_path_ok(kv_ring_cap, "host decode attention");
  lower_q();
  // MHA_CLMEM: decode reads the whole prefix from the concat slab; gather
  // the prefill rows (mirror-only during the prefill window) back once.
  // This decode step's own K/V were host-written into the slab above.
  sync_kv_slab(cache_from);
  if (mha_clmem_mode && cache_to > kv_slab_synced_to)
    kv_slab_synced_to = cache_to;

  // out_ stores the output of Q * K
  nntrainer::Tensor out_(1, 1,
                         is_causal ? (calc_windowed_attn_index(cache_to) -
                                      calc_windowed_attn_index(cache_from))
                                   : (step_size * cache_to),
                         num_heads_Q, query_step.getTensorType());

  // Host decode attention reads GPU-produced Q/K on the host: sync first so the
  // read is coherent under NNTR_CUDA_ASYNC (no-op in sync mode).
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  nntrainer::cuda::drain_if_async();
#endif
  compute_kcaches(query_step, b_cached_key, out_, cache_from,
                  cache_to - cache_from, num_heads_Q, gqa_size, head_dim);

  softmax_triangle(out_, step_size, num_heads_Q, cache_from);

  compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                cache_from, num_heads_KV, gqa_size, head_dim,
                                cache_to);
#if defined(ENABLE_OPENCL)
  if (o_cl != nullptr)
    nntrainer::clmem_raise_cl(attention_output_step, 0);
#endif
}

#if defined(__ARM_NEON)
#include <arm_neon.h>
// Cephes exp() for 4 floats at once (matches neon_mathfun.hxx exp_ps).
static inline float32x4_t vjepa_expq_f32(float32x4_t x) {
  const float32x4_t one = vdupq_n_f32(1.0f);
  x = vminq_f32(x, vdupq_n_f32(88.3762626647949f));
  x = vmaxq_f32(x, vdupq_n_f32(-88.3762626647949f));
  float32x4_t fx =
    vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.44269504088896341f));
  float32x4_t tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));
  uint32x4_t mask = vandq_u32(vcgtq_f32(tmp, fx), vreinterpretq_u32_f32(one));
  fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));
  x = vsubq_f32(x, vmulq_f32(fx, vdupq_n_f32(0.693359375f)));
  x = vsubq_f32(x, vmulq_f32(fx, vdupq_n_f32(-2.12194440e-4f)));
  float32x4_t z = vmulq_f32(x, x);
  float32x4_t y = vdupq_n_f32(1.9875691500E-4f);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(1.3981999507E-3f));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(8.3334519073E-3f));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(4.1665795894E-2f));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(1.6666665459E-1f));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(5.0000001201E-1f));
  y = vmulq_f32(y, z);
  y = vaddq_f32(y, x);
  y = vaddq_f32(y, one);
  int32x4_t mm =
    vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(0x7f)), 23);
  return vmulq_f32(y, vreinterpretq_f32_s32(mm));
}
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
#include <immintrin.h>
#endif

// Bulk convert N FP16-bits (uint16_t) values to FP32. Uses AVX2+F16C on x86
// (_mm256_cvtph_ps, available on Ivy Bridge+) and NEON fp16<->fp32 instructions
// on ARMv8.2+. Falls back to scalar nntrainer::compute_fp16_to_fp32. Treats the
// uint16 input as raw IEEE 754 half-precision bits — this is how the KV cache
// is stored regardless of ENABLE_FP16 build flag.
static inline void
mha_convert_fp16bits_to_fp32(unsigned int N, const uint16_t *src, float *dst) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
  unsigned int i = 0;
  for (; i + 16 <= N; i += 16) {
    __m256 a = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + i)));
    __m256 b = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + i + 8)));
    _mm256_storeu_ps(dst + i, a);
    _mm256_storeu_ps(dst + i + 8, b);
  }
  for (; i + 8 <= N; i += 8) {
    _mm256_storeu_ps(
      dst + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + i))));
  }
  for (; i < N; ++i)
    dst[i] = nntrainer::compute_fp16_to_fp32(src[i]);
#elif defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
  unsigned int i = 0;
  for (; i + 8 <= N; i += 8) {
    float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(src + i));
    vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(h)));
    vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(h)));
  }
  for (; i < N; ++i)
    dst[i] = nntrainer::compute_fp16_to_fp32(src[i]);
#else
  for (unsigned int i = 0; i < N; ++i)
    dst[i] = nntrainer::compute_fp16_to_fp32(src[i]);
#endif
}

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)

// Fused FP32 x FP16-bits -> FP32 GEMM for x86 (AVX2 + F16C). Equivalent of ARM
// shgemm but reads FP16-bits (uint16_t) directly without materializing an FP32
// copy of B — saves the temporary buffer and halves memory traffic compared to
// {convert+sgemm}. Row-major only, alpha applied, beta hard-coded to 0 to keep
// the kernel small (this is all the flash path needs).
//
// Two operand layouts:
//   TransB=true  (QK): C[m, n] = alpha * sum_k A[m,k] * fp16(B[n,k])
//                       B is N rows x K cols, row-major, ldb columns
//   TransB=false (AV): C[m, n] = alpha * sum_k A[m,k] * fp16(B[k,n])
//                       B is K rows x N cols, row-major, ldb columns
static inline void mha_hsgemm_avx2(unsigned int M, unsigned int N,
                                   unsigned int K, float alpha, const float *A,
                                   unsigned int lda, const uint16_t *B,
                                   unsigned int ldb, bool TransB, float *C,
                                   unsigned int ldc) {
  const __m256 valpha = _mm256_set1_ps(alpha);
  if (TransB) {
    // QK path. Block 4 m-rows so we amortize the B (K-row) conversion across 4
    // accumulators per inner k-step.
    unsigned int m = 0;
    for (; m + 4 <= M; m += 4) {
      const float *a0 = A + (size_t)(m + 0) * lda;
      const float *a1 = A + (size_t)(m + 1) * lda;
      const float *a2 = A + (size_t)(m + 2) * lda;
      const float *a3 = A + (size_t)(m + 3) * lda;
      for (unsigned int n = 0; n < N; ++n) {
        const uint16_t *b_row = B + (size_t)n * ldb;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        unsigned int k = 0;
        for (; k + 8 <= K; k += 8) {
          __m256 b =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b_row + k)));
          acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + k), b, acc0);
          acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + k), b, acc1);
          acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + k), b, acc2);
          acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + k), b, acc3);
        }
        // Horizontal-reduce 4 accumulators in parallel via two hadd-pairs.
        // acc0 = [s00 s01 s02 s03 | s04 s05 s06 s07] -> partial sums
        __m256 h01 = _mm256_hadd_ps(acc0, acc1);
        __m256 h23 = _mm256_hadd_ps(acc2, acc3);
        __m256 h = _mm256_hadd_ps(h01, h23);
        // h lanes: [s0_lo s1_lo s2_lo s3_lo | s0_hi s1_hi s2_hi s3_hi]
        __m128 lo = _mm256_castps256_ps128(h);
        __m128 hi = _mm256_extractf128_ps(h, 1);
        __m128 sums = _mm_add_ps(lo, hi); // [s0 s1 s2 s3]
        float s[4];
        _mm_storeu_ps(s, sums);
        // tail k
        for (; k < K; ++k) {
          const float bv = nntrainer::compute_fp16_to_fp32(b_row[k]);
          s[0] += a0[k] * bv;
          s[1] += a1[k] * bv;
          s[2] += a2[k] * bv;
          s[3] += a3[k] * bv;
        }
        C[(size_t)(m + 0) * ldc + n] = alpha * s[0];
        C[(size_t)(m + 1) * ldc + n] = alpha * s[1];
        C[(size_t)(m + 2) * ldc + n] = alpha * s[2];
        C[(size_t)(m + 3) * ldc + n] = alpha * s[3];
      }
    }
    // m tail (unblocked)
    for (; m < M; ++m) {
      const float *a_row = A + (size_t)m * lda;
      for (unsigned int n = 0; n < N; ++n) {
        const uint16_t *b_row = B + (size_t)n * ldb;
        __m256 acc = _mm256_setzero_ps();
        unsigned int k = 0;
        for (; k + 8 <= K; k += 8) {
          __m256 a = _mm256_loadu_ps(a_row + k);
          __m256 b =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b_row + k)));
          acc = _mm256_fmadd_ps(a, b, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float sum = _mm_cvtss_f32(s);
        for (; k < K; ++k)
          sum += a_row[k] * nntrainer::compute_fp16_to_fp32(b_row[k]);
        C[(size_t)m * ldc + n] = alpha * sum;
      }
    }
  } else {
    // AV path. Block n in 8-wide vector lanes; broadcast A[m,k] inside loop.
    for (unsigned int m = 0; m < M; ++m) {
      const float *a_row = A + (size_t)m * lda;
      float *c_row = C + (size_t)m * ldc;
      unsigned int n = 0;
      for (; n + 8 <= N; n += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (unsigned int k = 0; k < K; ++k) {
          __m256 a_b = _mm256_set1_ps(a_row[k]);
          __m256 b = _mm256_cvtph_ps(
            _mm_loadu_si128((const __m128i *)(B + (size_t)k * ldb + n)));
          acc = _mm256_fmadd_ps(a_b, b, acc);
        }
        _mm256_storeu_ps(c_row + n, _mm256_mul_ps(valpha, acc));
      }
      // n tail
      for (; n < N; ++n) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; ++k)
          sum +=
            a_row[k] * nntrainer::compute_fp16_to_fp32(B[(size_t)k * ldb + n]);
        c_row[n] = alpha * sum;
      }
    }
  }
}

#endif // __x86_64__ || __i386__

#if !defined(__x86_64__) && !defined(__i386__) && defined(__ARM_NEON)
} // namespace causallm

// libnntrainer.so is built with ENABLE_FP16=1 and exports these symbols. The
// CausalLM app may be built with ENABLE_FP16=0, in which case cpu_backend.h
// hides them behind #ifdef. Re-declare here at global / ::nntrainer scope.
// - shgemm:         FP32 A × FP16 B -> FP32 C   (FP32 partial accumulation)
// - hgemm_classify: FP16 A × FP16 B -> FP32 C   (FP32 partial accumulation)
// - custom_hgemm:   FP16 A × FP16 B -> FP16 C   (FP32 partial accumulation,
//                                                FP16-stored result).
namespace nntrainer {
void shgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
            const unsigned int M, const unsigned int N, const unsigned int K,
            const float alpha, const float *A, const unsigned int lda,
            const __fp16 *B, const unsigned int ldb, const float beta, float *C,
            const unsigned int ldc);
namespace neon {
void custom_hgemm(const __fp16 *A, const __fp16 *B, __fp16 *C, uint32_t M,
                  uint32_t N, uint32_t K, float alpha, float beta, bool TransA,
                  bool TransB);
} // namespace neon
} // namespace nntrainer
void hgemm_classify(const __fp16 *A, const __fp16 *B, float *C32,
                    unsigned int M, unsigned int N, unsigned int K, float alpha,
                    float beta, bool TransA, bool TransB);

namespace causallm {
#endif

void MHACoreLayer::gemm_attention(nntrainer::Tensor &query_step,
                                  nntrainer::Tensor &b_cached_key,
                                  nntrainer::Tensor &b_cached_value,
                                  nntrainer::Tensor &attention_output_step,
                                  unsigned int N_kv, unsigned int N_q,
                                  unsigned int cache_from) {
  const unsigned int d = head_dim;
  const unsigned int HD_Q = num_heads_Q * d;
  const unsigned int HD_KV = num_heads_KV * d;
  const unsigned int gqa =
    (num_heads_KV > 0) ? static_cast<unsigned int>(num_heads_Q / num_heads_KV)
                       : 1u;
  const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(d));
  const unsigned int order =
    static_cast<unsigned int>(query_step.getDim().getStorageOrder());
  const bool causal = is_causal;
  // Treat any local_window_size >= cache length as "no window".
  const bool windowed = (local_window_size < N_kv);
  const size_t W = static_cast<size_t>(local_window_size);

  // Runtime dtype dispatch: forwarding() may convert Q/V/output to FP16 when
  // ENABLE_FP16 && __ANDROID__ build. K/V are always FP16 storage.
  const bool q_fp16 =
    (query_step.getDataType() == ml::train::TensorDim::DataType::FP16);
  const bool o_fp16 = (attention_output_step.getDataType() ==
                       ml::train::TensorDim::DataType::FP16);

  const float *Q = nullptr;
  const uint16_t *Q_fp16_src = nullptr;
  float *O = nullptr;
  uint16_t *O_fp16 = nullptr;
  if (q_fp16) {
#ifdef ENABLE_FP16
    Q_fp16_src =
      reinterpret_cast<const uint16_t *>(query_step.getData<_FP16>());
#endif
  } else {
    Q = query_step.getData<float>();
  }
  if (o_fp16) {
#ifdef ENABLE_FP16
    O_fp16 =
      reinterpret_cast<uint16_t *>(attention_output_step.getData<_FP16>());
#endif
  } else {
    O = attention_output_step.getData<float>();
  }

  // tile sizes (cache-resident S); overridable via env for tuning
  unsigned int Bq = 256, Bk = 512;
  if (const char *e = std::getenv("VJEPA_BQ"))
    Bq = static_cast<unsigned int>(std::stoul(e));
  if (const char *e = std::getenv("VJEPA_BK"))
    Bk = static_cast<unsigned int>(std::stoul(e));

  const unsigned int num_qb = (N_q + Bq - 1) / Bq;
  auto &tm = nntrainer::ThreadManager::Global();

  // Cache always stores half-precision (FP16-bit) values; read as raw uint16_t
  // bits so we don't depend on ENABLE_FP16 / _FP16 / _Float16 being defined.
  // When kv_int8 is active the cache holds int8 bytes; the de-interleave
  // loop below dequantizes on the fly using cur_kv_int8_*_scale_batch.
  const uint16_t *Kbase = nullptr;
  const uint16_t *Vbase = nullptr;
#ifdef ENABLE_FP16
  const int8_t *Kbase_i8 = nullptr;
  const int8_t *Vbase_i8 = nullptr;
#endif
  if (kv_int8) {
#ifdef ENABLE_FP16
    Kbase_i8 =
      reinterpret_cast<const int8_t *>(b_cached_key.getData<uint8_t>());
    Vbase_i8 =
      reinterpret_cast<const int8_t *>(b_cached_value.getData<uint8_t>());
#endif
    NNTR_THROW_IF(cur_kv_int8_key_scale_batch == nullptr ||
                    cur_kv_int8_value_scale_batch == nullptr,
                  std::invalid_argument)
      << "kv_int8 gemm_attention path missing per-batch scale pointers";
  } else {
#ifdef ENABLE_FP16
    Kbase = reinterpret_cast<const uint16_t *>(b_cached_key.getData<_FP16>());
    Vbase = reinterpret_cast<const uint16_t *>(b_cached_value.getData<_FP16>());
#else
    Kbase = b_cached_key.getData<uint16_t>();
    Vbase = b_cached_value.getData<uint16_t>();
#endif
  }

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Opt-in GPU attention (engine=cuda, UVM): the interleaved fp16 query +
  // fp16 KV cache feed a flash core on the device, replacing the host O(M^2)
  // loop below. Matches gemm_attention exactly (scale 1/sqrt(d), causal +
  // sliding mask, GQA, NO softcap). Default OFF (NNTR_CUDA_ATTN) until
  // verified; falls through to the host path when off / not device-resident.
  {
    static const bool cuda_attn = nntr_env_on("NNTR_CUDA_ATTN");
    if (cuda_attn && q_fp16 && o_fp16 && !kv_int8 && Q_fp16_src && O_fp16) {
      auto dev_ok = [](const void *p) {
        return nntrainer::cuda::dev_accessible(p);
      };
      // Query + output must be device-resident (the kernel uses them directly);
      // the KV cache may be host-heap (it is, on engine=cuda) -- the launcher
      // mirrors it to the device. A device kernel touching host memory faults
      // and corrupts the context, so this guard is required.
      bool dev = dev_ok(Q_fp16_src) && dev_ok(O_fp16);
      if (dev) {
        const int win = windowed ? (int)local_window_size : INT_MAX;
        if (nntrainer::cuda::cuda_attention_interleaved_fp16(
              Q_fp16_src, Kbase, Vbase, O_fp16, (int)num_heads_Q,
              (int)num_heads_KV, (int)N_q, (int)N_kv, (int)cache_from, (int)d,
              win, /*softcap=*/0.0f, /*ring_cap=*/(int)kv_ring_cap))
          return;
      }
    }
  }
#endif

  // Phase 1: de-interleave heads once into shared contiguous buffers.
  // K/V always kept as raw FP16 bits (uint16). Q either FP32 (V-JEPA
  // path) or FP16 (when forwarding() pre-converts to FP16; ENABLE_FP16+
  // Android). The FP16 Q path keeps the entire attention in FP16
  // (custom_hgemm for QK and AV, FP16 softmax) without ever materializing
  // an FP32 score buffer.
  //
  // x86 guardrail (Gemma4): the ALL-FP16 NEON QK/AV branch is compiled out on
  // x86, and the FP32 fallback dereferences Qp_fp32. So when q_fp16 is set on
  // x86 we must materialize an FP32 de-interleaved Q (Qa_fp32) and take the
  // FP32 path; otherwise Qp_fp32==nullptr -> SIGSEGV (the crash the diagnosis
  // hit when use_gemm_attention was flipped and a d=512 layer fell to here).
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
  const bool q_fp16_to_fp32 = q_fp16; // x86 has no NEON FP16-Q kernel
#else
  const bool q_fp16_to_fp32 = false;
#endif
  std::vector<float> Qa_fp32;
  std::vector<uint16_t> Qa_fp16;
  if (q_fp16 && !q_fp16_to_fp32)
    Qa_fp16.resize((size_t)num_heads_Q * N_q * d);
  else
    Qa_fp32.resize((size_t)num_heads_Q * N_q * d);
  std::vector<uint16_t> Ka((size_t)num_heads_KV * N_kv * d);
  std::vector<uint16_t> Va((size_t)num_heads_KV * N_kv * d);
  {
    if (q_fp16_to_fp32) {
#ifdef ENABLE_FP16
      // FP16 bits -> FP32, de-interleaved per head (x86 FP32 attention path).
      tm.parallel_for(0, static_cast<size_t>(num_heads_Q), [&](size_t h) {
        float *qa = Qa_fp32.data() + (size_t)h * N_q * d;
        const uint16_t *qh = Q_fp16_src + h * d;
        for (unsigned int n = 0; n < N_q; ++n) {
          const uint16_t *qsrc = qh + (size_t)n * HD_Q;
          float *qdst = qa + (size_t)n * d;
          for (unsigned int x = 0; x < d; ++x)
            qdst[x] = nntrainer::compute_fp16_to_fp32(qsrc[x]);
        }
      });
#endif
    } else if (q_fp16) {
      tm.parallel_for(0, static_cast<size_t>(num_heads_Q), [&](size_t h) {
        uint16_t *qa = Qa_fp16.data() + (size_t)h * N_q * d;
        const uint16_t *qh = Q_fp16_src + h * d;
        for (unsigned int n = 0; n < N_q; ++n)
          std::memcpy(qa + (size_t)n * d, qh + (size_t)n * HD_Q,
                      d * sizeof(uint16_t));
      });
    } else {
      tm.parallel_for(0, static_cast<size_t>(num_heads_Q), [&](size_t h) {
        float *qa = Qa_fp32.data() + (size_t)h * N_q * d;
        const float *qh = Q + h * d;
        for (unsigned int n = 0; n < N_q; ++n)
          std::memcpy(qa + (size_t)n * d, qh + (size_t)n * HD_Q,
                      d * sizeof(float));
      });
    }
    if (kv_int8) {
#ifdef ENABLE_FP16
      // Dequant on the fly: K/V int8[n, hkv, :] * scale[n, hkv] -> fp16
      // bits stored in the de-interleaved Ka/Va buffers, so downstream
      // Phase 2 sees the same fp16 layout as the regular path.
      const uint16_t *Kscale = cur_kv_int8_key_scale_batch;
      const uint16_t *Vscale = cur_kv_int8_value_scale_batch;
      tm.parallel_for(0, static_cast<size_t>(num_heads_KV), [&](size_t hkv) {
        uint16_t *ka = Ka.data() + (size_t)hkv * N_kv * d;
        uint16_t *va = Va.data() + (size_t)hkv * N_kv * d;
        const int8_t *kh = Kbase_i8 + hkv * d;
        const int8_t *vh = Vbase_i8 + hkv * d;
        for (unsigned int n = 0; n < N_kv; ++n) {
          const float ks = nntrainer::compute_fp16_to_fp32(
            Kscale[(size_t)n * num_heads_KV + hkv]);
          const float vs = nntrainer::compute_fp16_to_fp32(
            Vscale[(size_t)n * num_heads_KV + hkv]);
          const int8_t *k_row = kh + (size_t)n * HD_KV;
          const int8_t *v_row = vh + (size_t)n * HD_KV;
          uint16_t *ka_row = ka + (size_t)n * d;
          uint16_t *va_row = va + (size_t)n * d;
          for (unsigned int x = 0; x < d; ++x) {
            ka_row[x] = nntrainer::compute_fp32_to_fp16((float)k_row[x] * ks);
            va_row[x] = nntrainer::compute_fp32_to_fp16((float)v_row[x] * vs);
          }
        }
      });
#else
      NNTR_THROW_IF(true, std::invalid_argument)
        << "kv_int8 gemm_attention requires ENABLE_FP16";
#endif
    } else {
      tm.parallel_for(0, static_cast<size_t>(num_heads_KV), [&](size_t hkv) {
        uint16_t *ka = Ka.data() + (size_t)hkv * N_kv * d;
        uint16_t *va = Va.data() + (size_t)hkv * N_kv * d;
        const uint16_t *kh = Kbase + hkv * d;
        const uint16_t *vh = Vbase + hkv * d;
        for (unsigned int n = 0; n < N_kv; ++n) {
          std::memcpy(ka + (size_t)n * d, kh + (size_t)n * HD_KV,
                      d * sizeof(uint16_t));
          std::memcpy(va + (size_t)n * d, vh + (size_t)n * HD_KV,
                      d * sizeof(uint16_t));
        }
      });
    }
  }

  // Phase 2: flash attention over balanced (h_q, query-block) work units.
  tm.parallel_for(0, static_cast<size_t>(num_heads_Q) * num_qb, [&](size_t u) {
    const unsigned int h_q = static_cast<unsigned int>(u / num_qb);
    const unsigned int h_kv = h_q / gqa;
    const unsigned int qb = static_cast<unsigned int>(u % num_qb) * Bq;
    const unsigned int bq = std::min(Bq, N_q - qb);
    const float *Qp_fp32 = (q_fp16 && !q_fp16_to_fp32)
                             ? nullptr
                             : (Qa_fp32.data() + (size_t)h_q * N_q * d);
    const uint16_t *Qp_fp16 = (q_fp16 && !q_fp16_to_fp32)
                                ? (Qa_fp16.data() + (size_t)h_q * N_q * d)
                                : nullptr;
    const uint16_t *Kp = Ka.data() + (size_t)h_kv * N_kv * d;
    const uint16_t *Vp = Va.data() + (size_t)h_kv * N_kv * d;
    float *Oh = o_fp16 ? nullptr : (O + h_q * d);
    uint16_t *Oh_fp16 = o_fp16 ? (O_fp16 + h_q * d) : nullptr;

    thread_local std::vector<float> S, Pacc, Ol, mrow, lrow;
    thread_local std::vector<uint16_t> Sp16, Pacc16;
    S.resize((size_t)Bq * Bk);
    Pacc.resize((size_t)Bq * d);
    Ol.resize((size_t)Bq * d);
    mrow.resize(Bq);
    lrow.resize(Bq);
#if !defined(__x86_64__) && !defined(__i386__) && defined(__ARM_NEON)
    Sp16.resize((size_t)Bq * Bk);
    Pacc16.resize((size_t)Bq * d);
#endif
    // FP16-throughout path uses Sp16 for both QK output (custom_hgemm,
    // FP16-stored) and AV input (softmax in-place updates the same
    // buffer). The FP32 S buffer is unused in that path.

    std::fill(Ol.begin(), Ol.begin() + (size_t)bq * d, 0.0f);
    for (unsigned int i = 0; i < bq; ++i) {
      mrow[i] = -3.0e38f;
      lrow[i] = 0.0f;
    }

    // The absolute query positions in this work unit are
    // [cache_from + qb, cache_from + qb + bq).
    const size_t q_abs_lo = (size_t)cache_from + qb;
    const size_t q_abs_hi = q_abs_lo + bq - 1; // inclusive

    for (unsigned int kb = 0; kb < N_kv; kb += Bk) {
      const unsigned int bk = std::min(Bk, N_kv - kb);

      // Causal upper-bound block-skip: smallest k_abs in block > largest
      // q_abs -> this and all later key blocks contribute nothing.
      if (causal && (size_t)kb > q_abs_hi)
        break;

      // Sliding-window lower-bound block-skip: largest k_abs in block <
      // smallest visible threshold (q_abs_lo - W + 1, i.e., k_abs must
      // satisfy k_abs > q_abs - W).
      if (windowed && (size_t)kb + bk + W <= q_abs_lo + 1)
        continue;

      // Does this block straddle the causal diagonal for any row?
      const bool causal_boundary = causal && ((size_t)kb + bk > q_abs_lo + 1);
      // Does this block straddle the sliding-window lower bound for any row?
      const bool window_boundary = windowed && ((size_t)kb + W < q_abs_hi + 1);

#if !defined(__x86_64__) && !defined(__i386__) && defined(__ARM_NEON)
      if (q_fp16) {
        // ALL-FP16 PATH (matches pre-V-JEPA mha_core precision: FP16
        // storage with FP32 partial accumulation inside NEON kernels,
        // no upgrade to FP32 in the score buffer).
        // QK: FP16 × FP16 -> FP16 (custom_hgemm with FP32 partial acc).
        nntrainer::neon::custom_hgemm(
          reinterpret_cast<const __fp16 *>(Qp_fp16 + (size_t)qb * d),
          reinterpret_cast<const __fp16 *>(Kp + (size_t)kb * d),
          reinterpret_cast<__fp16 *>(Sp16.data()), bq, bk, d, inv_sqrt, 0.0f,
          /*TransA=*/false, /*TransB=*/true);

        // Boundary masking in FP16: write -INFINITY as bit pattern 0xFC00.
        for (unsigned int i = 0; i < bq; ++i) {
          uint16_t *sp16 = Sp16.data() + (size_t)i * bk;
          const long long q_abs = (long long)cache_from + qb + i;
          if (causal_boundary) {
            long long valid_count_ll = q_abs + 1 - (long long)kb;
            unsigned int valid_count = (valid_count_ll <= 0)
                                         ? 0u
                                         : (valid_count_ll >= (long long)bk
                                              ? bk
                                              : (unsigned int)valid_count_ll);
            for (unsigned int k = valid_count; k < bk; ++k)
              sp16[k] = 0xFC00; // FP16 -infinity
          }
          if (window_boundary) {
            long long first_valid_ll = q_abs - (long long)W - (long long)kb + 1;
            unsigned int first_valid = (first_valid_ll <= 0)
                                         ? 0u
                                         : (first_valid_ll >= (long long)bk
                                              ? bk
                                              : (unsigned int)first_valid_ll);
            for (unsigned int k = 0; k < first_valid; ++k)
              sp16[k] = 0xFC00;
          }

          // Block max (read FP16, compute FP32 register for stability).
          float bm = -3.0e38f;
          {
            float32x4_t vmx = vdupq_n_f32(-3.0e38f);
            unsigned int k = 0;
            for (; k + 4 <= bk; k += 4) {
              float16x4_t h = vreinterpret_f16_u16(vld1_u16(sp16 + k));
              vmx = vmaxq_f32(vmx, vcvt_f32_f16(h));
            }
            bm = vmaxvq_f32(vmx);
            for (; k < bk; ++k)
              bm = std::max(bm, nntrainer::compute_fp16_to_fp32(sp16[k]));
          }
          const float nm = std::max(mrow[i], bm);
          const float c = std::exp(mrow[i] - nm);
          float bs = 0.0f;
          {
            // Softmax: read FP16 -> FP32 register, exp, store FP16.
            float32x4_t vsum = vdupq_n_f32(0.0f), vnm = vdupq_n_f32(nm);
            unsigned int k = 0;
            for (; k + 4 <= bk; k += 4) {
              float16x4_t h = vreinterpret_f16_u16(vld1_u16(sp16 + k));
              float32x4_t v = vcvt_f32_f16(h);
              float32x4_t e = vjepa_expq_f32(vsubq_f32(v, vnm));
              float16x4_t e_h = vcvt_f16_f32(e);
              vst1_u16(sp16 + k, vreinterpret_u16_f16(e_h));
              vsum = vaddq_f32(vsum, e);
            }
            bs = vaddvq_f32(vsum);
            for (; k < bk; ++k) {
              float v = nntrainer::compute_fp16_to_fp32(sp16[k]);
              float e = std::exp(v - nm);
              sp16[k] = nntrainer::compute_fp32_to_fp16(e);
              bs += e;
            }
          }
          lrow[i] = lrow[i] * c + bs;
          mrow[i] = nm;
          float *ol = Ol.data() + (size_t)i * d;
          for (unsigned int x = 0; x < d; ++x)
            ol[x] *= c;
        }

        // AV: FP16 × FP16 -> FP16 (custom_hgemm).
        nntrainer::neon::custom_hgemm(
          reinterpret_cast<const __fp16 *>(Sp16.data()),
          reinterpret_cast<const __fp16 *>(Vp + (size_t)kb * d),
          reinterpret_cast<__fp16 *>(Pacc16.data()), bq, d, bk, 1.0f, 0.0f,
          /*TransA=*/false, /*TransB=*/false);
        // Accumulate Pacc16 -> Ol FP32 (FP32 accumulator across kb).
        for (unsigned int i = 0; i < bq; ++i) {
          float *ol = Ol.data() + (size_t)i * d;
          const uint16_t *pa = Pacc16.data() + (size_t)i * d;
          unsigned int x = 0;
          for (; x + 8 <= d; x += 8) {
            float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(pa + x));
            float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
            float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
            vst1q_f32(ol + x, vaddq_f32(vld1q_f32(ol + x), lo));
            vst1q_f32(ol + x + 4, vaddq_f32(vld1q_f32(ol + x + 4), hi));
          }
          for (; x < d; ++x)
            ol[x] += nntrainer::compute_fp16_to_fp32(pa[x]);
        }
      } else
#endif // ARM NEON q_fp16 branch
      {
        // FP32 Q path: QK -> FP32 S, fused FP16 softmax store to Sp16, AV.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
        mha_hsgemm_avx2(bq, bk, d, inv_sqrt, Qp_fp32 + (size_t)qb * d, d,
                        Kp + (size_t)kb * d, d, /*TransB=*/true, S.data(), bk);
#elif defined(__ARM_NEON)
          nntrainer::shgemm(
            order, false, true, bq, bk, d, inv_sqrt,
            Qp_fp32 + (size_t)qb * d, d,
            reinterpret_cast<const __fp16 *>(Kp + (size_t)kb * d), d, 0.0f,
            S.data(), bk);
#else
          nntrainer::sgemm(order, false, true, bq, bk, d, inv_sqrt,
                           Qp_fp32 + (size_t)qb * d, d,
                           Kp + (size_t)kb * d, d, 0.0f, S.data(), bk);
#endif

        for (unsigned int i = 0; i < bq; ++i) {
          float *s = S.data() + (size_t)i * bk;
          const long long q_abs = (long long)cache_from + qb + i;
          if (causal_boundary) {
            long long valid_count_ll = q_abs + 1 - (long long)kb;
            unsigned int valid_count = (valid_count_ll <= 0)
                                         ? 0u
                                         : (valid_count_ll >= (long long)bk
                                              ? bk
                                              : (unsigned int)valid_count_ll);
            for (unsigned int k = valid_count; k < bk; ++k)
              s[k] = -INFINITY;
          }
          if (window_boundary) {
            long long first_valid_ll = q_abs - (long long)W - (long long)kb + 1;
            unsigned int first_valid = (first_valid_ll <= 0)
                                         ? 0u
                                         : (first_valid_ll >= (long long)bk
                                              ? bk
                                              : (unsigned int)first_valid_ll);
            for (unsigned int k = 0; k < first_valid; ++k)
              s[k] = -INFINITY;
          }

          float bm = -3.0e38f;
#if defined(__ARM_NEON)
          {
            float32x4_t vmx = vdupq_n_f32(-3.0e38f);
            unsigned int k = 0;
            for (; k + 4 <= bk; k += 4)
              vmx = vmaxq_f32(vmx, vld1q_f32(s + k));
            bm = vmaxvq_f32(vmx);
            for (; k < bk; ++k)
              bm = std::max(bm, s[k]);
          }
#else
            for (unsigned int k = 0; k < bk; ++k)
              bm = std::max(bm, s[k]);
#endif
          const float nm = std::max(mrow[i], bm);
          const float c = std::exp(mrow[i] - nm);
          float bs = 0.0f;
#if !defined(__x86_64__) && !defined(__i386__) && defined(__ARM_NEON)
          {
            uint16_t *sp16 = Sp16.data() + (size_t)i * bk;
            float32x4_t vsum = vdupq_n_f32(0.0f), vnm = vdupq_n_f32(nm);
            unsigned int k = 0;
            for (; k + 4 <= bk; k += 4) {
              float32x4_t e = vjepa_expq_f32(vsubq_f32(vld1q_f32(s + k), vnm));
              float16x4_t e_h = vcvt_f16_f32(e);
              vst1_u16(sp16 + k, vreinterpret_u16_f16(e_h));
              vsum = vaddq_f32(vsum, e);
            }
            bs = vaddvq_f32(vsum);
            for (; k < bk; ++k) {
              float e = std::exp(s[k] - nm);
              sp16[k] = nntrainer::compute_fp32_to_fp16(e);
              bs += e;
            }
          }
#elif defined(__ARM_NEON)
            {
              float32x4_t vsum = vdupq_n_f32(0.0f), vnm = vdupq_n_f32(nm);
              unsigned int k = 0;
              for (; k + 4 <= bk; k += 4) {
                float32x4_t e =
                  vjepa_expq_f32(vsubq_f32(vld1q_f32(s + k), vnm));
                vst1q_f32(s + k, e);
                vsum = vaddq_f32(vsum, e);
              }
              bs = vaddvq_f32(vsum);
              for (; k < bk; ++k) {
                float e = std::exp(s[k] - nm);
                s[k] = e;
                bs += e;
              }
            }
#else
            for (unsigned int k = 0; k < bk; ++k) {
              float e = std::exp(s[k] - nm);
              s[k] = e;
              bs += e;
            }
#endif
          lrow[i] = lrow[i] * c + bs;
          mrow[i] = nm;
          float *ol = Ol.data() + (size_t)i * d;
          for (unsigned int x = 0; x < d; ++x)
            ol[x] *= c;
        }

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
        mha_hsgemm_avx2(bq, d, bk, 1.0f, S.data(), bk, Vp + (size_t)kb * d, d,
                        /*TransB=*/false, Pacc.data(), d);
        for (unsigned int i = 0; i < bq; ++i) {
          float *ol = Ol.data() + (size_t)i * d;
          const float *pa = Pacc.data() + (size_t)i * d;
          for (unsigned int x = 0; x < d; ++x)
            ol[x] += pa[x];
        }
#elif defined(__ARM_NEON)
          nntrainer::neon::custom_hgemm(
            reinterpret_cast<const __fp16 *>(Sp16.data()),
            reinterpret_cast<const __fp16 *>(Vp + (size_t)kb * d),
            reinterpret_cast<__fp16 *>(Pacc16.data()), bq, d, bk, 1.0f, 0.0f,
            /*TransA=*/false, /*TransB=*/false);
          for (unsigned int i = 0; i < bq; ++i) {
            float *ol = Ol.data() + (size_t)i * d;
            const uint16_t *pa = Pacc16.data() + (size_t)i * d;
            unsigned int x = 0;
            for (; x + 8 <= d; x += 8) {
              float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(pa + x));
              float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
              float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
              vst1q_f32(ol + x, vaddq_f32(vld1q_f32(ol + x), lo));
              vst1q_f32(ol + x + 4, vaddq_f32(vld1q_f32(ol + x + 4), hi));
            }
            for (; x < d; ++x)
              ol[x] += nntrainer::compute_fp16_to_fp32(pa[x]);
          }
#else
          nntrainer::sgemm(order, false, false, bq, d, bk, 1.0f, S.data(), bk,
                           Vp + (size_t)kb * d, d, 0.0f, Pacc.data(), d);
          for (unsigned int i = 0; i < bq; ++i) {
            float *ol = Ol.data() + (size_t)i * d;
            const float *pa = Pacc.data() + (size_t)i * d;
            for (unsigned int x = 0; x < d; ++x)
              ol[x] += pa[x];
          }
#endif
      } // FP32 Q path
    }
    for (unsigned int i = 0; i < bq; ++i) {
      const float inv = (lrow[i] > 0.0f) ? (1.0f / lrow[i]) : 0.0f;
      const float *ol = Ol.data() + (size_t)i * d;
      if (o_fp16) {
        uint16_t *oh = Oh_fp16 + (size_t)(qb + i) * HD_Q;
        for (unsigned int x = 0; x < d; ++x)
          oh[x] = nntrainer::compute_fp32_to_fp16(ol[x] * inv);
      } else {
        float *oh = Oh + (size_t)(qb + i) * HD_Q;
        for (unsigned int x = 0; x < d; ++x)
          oh[x] = ol[x] * inv;
      }
    }
  });
}

void MHACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query_step,
  nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
  nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
  ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim, nntrainer::Tensor &sink_step) {
  /// @todo replace from, to into cache_index, input height
  /// @note currently, only gpt-oss uses this method

  // Static GPU_CLMEM residency, defensive bridge: this (host-heavy) sink
  // variant is not cl_mem-converted -- lower any resident I/O up front and
  // raise the output at exit so a GPU_CLMEM classification cannot leave it
  // reading/writing a stale plane. No-ops when the classes are SVM.
#if defined(ENABLE_OPENCL)
  // GPU_CLMEM lower/raise bridge is OpenCL-only (I/O is plain host memory
  // without it).
  if (query_step.isClMem())
    nntrainer::clmem_lower_cl(query_step, 0);
  if (key_step.isClMem())
    nntrainer::clmem_lower_cl(key_step, 0);
  if (value_step.isClMem())
    nntrainer::clmem_lower_cl(value_step, 0);
  struct ORaise {
    nntrainer::Tensor &t;
    ~ORaise() {
      if (t.isClMem())
        nntrainer::clmem_raise_cl(t, 0);
    }
  } _oraise{attention_output_step};
#endif

  /**
   *  cache_key
   *  +--------+                        ->
   *  |        |                        ->
   *  |        |                        ->
   *  |........| from                   ->
   *  |........| to -> b_cache_key_step -> b_cached_key
   *  |        |
   *  +--------+
   *
   */

  // NNTR_KV_INT8 is wired through the Qwen3 entry point only. The
  // gpt-oss/sink_step variant has no int8 write/read plumbing yet.
  NNTR_THROW_IF(kv_int8, std::invalid_argument)
    << "NNTR_KV_INT8 is not supported for the sink_step (gpt-oss) variant of "
       "one_batch_incremental_forwarding";

  /** 1. Load Input Tensors of this batch : b_ denotes a Tensor for this batch
   * **/
  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(), true);
  nntrainer::Tensor b_cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim,
    batch * cache_value_dim.getFeatureLen() + from * cache_value_dim.width(),
    true);

  apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, _from, true);

  apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim, _from, true);

  if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    apply_rotary_emb_tensor_v2(value_step, b_cache_value_step, head_dim, _from,
                               false);
  } else if (query_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    b_cache_value_step.copyData(value_step);
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }

  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  // [kv-window-ring] No clamp here: this is the attention-sink overload, and
  // causallm::kvRingLayerEligible() returns false for a use_sink layer, so
  // kv_ring_cap is 0 by construction and the cache is the full max_seq height.
  // A clamp would read as though the ring could reach this path.
  cached_key_dim.height(to);
  cached_value_dim.height(to);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  // skip_prefill (see the non-sink overload above): K/V are written into the
  // cache; skip the unused prefill attention compute + output write. The
  // _oraise guard above raises the (untouched) output on return -- a no-op for
  // the SVM/host case and harmless for cl_mem since the downstream wo FC also
  // skips the prefill big-step.
  if (skip_prefill && (to - from) > 1)
    return;

  nntrainer::Tensor out_(1, 1,
                         is_causal ? (((to - from) == 1)
                                        ? to
                                        : calc_windowed_attn_index(to) -
                                            calc_windowed_attn_index(from))
                                   : ((to - from) * to),
                         num_heads_Q, query_step.getTensorType());

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  compute_kcaches(query_step, b_cached_key, out_, _from, to - from, num_heads_Q,
                  gqa_size, head_dim);

  softmax_triangle(out_, to - from, num_heads_Q, from, sink_step);

  compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                from, num_heads_KV, gqa_size, head_dim, to);
}

/************************************************************** */

/**
 * @brief rotary embedding-related member function
 * @note seq_len -> max_position_embeddings
 */
void MHACoreLayer::precompute_freqs(int head_dim, unsigned int seq_len,
                                    float theta, bool is_fp16) {
  const std::string cache_key =
    std::string(is_fp16 ? "fp16:" : "fp32:") + std::to_string(head_dim) + ":" +
    std::to_string(seq_len) + ":" + std::to_string(theta) + ":" +
    rope_scaling_type + ":" + std::to_string(scale) + ":" +
    std::to_string(original_max_position_embeddings);

  auto cache_iter = rope_freq_cache.find(cache_key);
  if (cache_iter != rope_freq_cache.end()) {
    if (is_fp16) {
#ifdef ENABLE_FP16
      freqs_cos_fp16 = &cache_iter->second.cos_fp16;
      freqs_sin_fp16 = &cache_iter->second.sin_fp16;
#endif
    } else {
      freqs_cos = &cache_iter->second.cos;
      freqs_sin = &cache_iter->second.sin;
    }
    return;
  }

  thetas.clear();
  if (rope_scaling_type == "yarn")
    _compute_yarn_parameters(head_dim, theta);
  else if (rope_scaling_type == "proportional" ||
           rope_partial_rotary_factor != 1.0f)
    // Proportional rope (Gemma3n/Gemma4 E2B). Also routes here when a
    // partial_rotary_factor < 1 is configured even under the "default" type,
    // since _compute_proportional_parameters is the one that zeroes the
    // non-rotary tail of the frequency table.
    _compute_proportional_parameters(head_dim, theta);
  else {
    // "default" plus any model-specific type we don't special-case (e.g.
    // Gemma4/Gemma3n "linear"/local rope) -> standard RoPE frequencies. Warn
    // once so the unhandled type is visible without aborting the run.
    if (rope_scaling_type != "default") {
      static bool warned = false;
      if (!warned) {
        warned = true;
        ml_logw("[mha_core] rope_scaling_type='%s' not special-cased; "
                "using default RoPE.",
                rope_scaling_type.c_str());
      }
    }
    _compute_default_parameters(head_dim, theta);
  }

  unsigned int half_ = head_dim / 2;
  auto &cache = rope_freq_cache[cache_key];

  if (!is_fp16) {
    // cos / sin
    cache.cos.assign(seq_len, std::vector<float>(head_dim, 0));
    cache.sin.assign(seq_len, std::vector<float>(head_dim, 0));

    // update cos / sin frequency
    for (unsigned int i = 0; i < seq_len; ++i) {

      // cpu_backend provides calc_trigonometric_vals_dup on every arch (NEON
      // on ARM, scalar fallback elsewhere), so the app-side USE_NEON fork is
      // gone — one code path, one rounding behaviour across backends.
      nntrainer::calc_trigonometric_vals_dup(
        half_, thetas.data(), cache.cos[i].data(), cache.sin[i].data(), i,
        attention_scaling);
    }
    freqs_cos = &cache.cos;
    freqs_sin = &cache.sin;
  }

#ifdef ENABLE_FP16
  if (is_fp16) {
    // cos / sin for FP16
    cache.cos_fp16.assign(seq_len, std::vector<_FP16>(head_dim, 0));
    cache.sin_fp16.assign(seq_len, std::vector<_FP16>(head_dim, 0));

    std::vector<float> cos_tmp(head_dim);
    std::vector<float> sin_tmp(head_dim);

    for (unsigned int i = 0; i < seq_len; ++i) {
      // Same as the FP32 branch: one arch-neutral cpu_backend call, no
      // app-side USE_NEON fork.
      nntrainer::calc_trigonometric_vals_dup(half_, thetas.data(),
                                             cos_tmp.data(), sin_tmp.data(), i,
                                             attention_scaling);
      for (unsigned int j = 0; j < head_dim; ++j) {
        cache.cos_fp16[i][j] = (_FP16)cos_tmp[j];
        cache.sin_fp16[i][j] = (_FP16)sin_tmp[j];
      }
    }
    freqs_cos_fp16 = &cache.cos_fp16;
    freqs_sin_fp16 = &cache.sin_fp16;
  }
#endif
};

void MHACoreLayer::_compute_default_parameters(int head_dim, float theta) {

  // no attention scaling
  attention_scaling = 1.0f;

  // theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... , dim/2]
  // head_dim should be divisible by 2
  unsigned int half_ = head_dim / 2;
  for (unsigned int i = 0; i < half_; ++i) {
    thetas.push_back(1.0 /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }
}

void MHACoreLayer::_compute_proportional_parameters(int head_dim, float theta) {

  // no attention scaling for proportional rope
  attention_scaling = 1.0f;

  // Partial rotary: only the first rope_angles frequencies receive rotary
  // embedding; the rest of the head_dim/2 entries are zeroed so cos=1/sin=0,
  // i.e. those channels pass through unrotated. With
  // rope_partial_rotary_factor == 1.0 this reduces to default RoPE scaled by
  // 1/scale.
  const int half_dim = static_cast<int>(head_dim / 2);
  const int rope_angles =
    static_cast<int>((rope_partial_rotary_factor * head_dim) / 2.0f);
  thetas.reserve(half_dim);
  for (int i = 0; i < rope_angles; ++i)
    thetas.push_back(1.0f /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  for (int i = rope_angles; i < half_dim; ++i)
    thetas.push_back(0.0f);
  for (auto &val : thetas)
    val /= scale;
}

void MHACoreLayer::_compute_yarn_parameters(int head_dim, float theta) {

  // Config parameters
  ///@todo partial_rotary_factor should be generalized to fully support
  /// transformers's implementation
  // const float partial_rotary_factor = has_partial_rotary_factor ?
  // config_partial_rotary_factor : 1.0f;
  const float partial_rotary_factor = 1.0f;
  const int dim = static_cast<int>(head_dim * partial_rotary_factor);
  const float base = theta;

  // Handle max position embeddings

  // Attention scaling calculation (simplified from Python version)
  auto get_mscale = [](float scale, float mscale = 1.0f) {
    return (scale <= 1.0f) ? 1.0f : (0.1f * mscale * std::log(scale) + 1.0f);
  };

  ///@todo attention_scaling should be generalized to fully support
  /// transformers's implementation
  // if (has_mscale && has_mscale_all_dim) {
  // attention_scaling = get_mscale(factor, mscale) / get_mscale(factor,
  // mscale_all_dim);
  // } else {
  // attention_scaling = get_mscale(factor);
  // }
  attention_scaling = get_mscale(scale);

  ///@todo attention_scaling should be generalized to fully support
  /// transformers's implementation
  // const float beta_fast = has_beta_fast ? config_beta_fast : 32.0f;
  // const float beta_slow = has_beta_slow ? config_beta_slow : 1.0f;
  // const bool truncate = has_truncate ? config_truncate : true;
  // Beta parameters
  const float beta_fast = 32.0f;
  const float beta_slow = 1.0f;
  const bool truncate = false;

  // Helper functions
  auto find_correction_dim = [&](float num_rotations) {
    return (dim * std::log(original_max_position_embeddings /
                           (num_rotations * 2 * M_PI))) /
           (2 * std::log(base));
  };

  auto [low, high] = [&]() {
    float low_val = find_correction_dim(beta_fast);
    float high_val = find_correction_dim(beta_slow);
    if (truncate) {
      low_val = std::floor(low_val);
      high_val = std::ceil(high_val);
    }
    return std::make_pair(low_val, high_val);
  }();

  // Compute position frequencies
  thetas.resize(dim / 2);

  // Compute interpolation and extrapolation frequencies
  std::vector<float> inv_freq_interpolation;
  std::vector<float> inv_freq_extrapolation;
  for (size_t i = 0; i < dim / 2; ++i) {
    inv_freq_extrapolation.push_back(
      1.0 / (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
    inv_freq_interpolation.push_back(
      1.0 / (scale * std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }

  auto linear_ramp_factor = [](float min, float max, int size) {
    if (min == max) {
      max += 0.001f; // Prevent singularity
    }
    std::vector<float> ramp(size);
    for (int i = 0; i < size; ++i) {
      float val = (i - min) / (max - min);
      ramp[i] = std::clamp(val, 0.0f, 1.0f);
    }
    return ramp;
  };

  std::vector<float> inv_freq_extrapolation_factor =
    linear_ramp_factor(low, high, dim / 2);
  for (auto &val : inv_freq_extrapolation_factor) {
    val = 1.0f - val;
  }

  // Combine frequencies
  for (size_t i = 0; i < thetas.size(); ++i) {
    thetas[i] =
      inv_freq_extrapolation[i] * inv_freq_extrapolation_factor[i] +
      inv_freq_interpolation[i] * (1.0f - inv_freq_extrapolation_factor[i]);
  }
}

void MHACoreLayer::apply_rotary_emb_tensor_v2(nntrainer::Tensor &in,
                                              nntrainer::Tensor &out,
                                              unsigned int dim,
                                              unsigned int from,
                                              bool apply_rope) {
  unsigned int half_ = dim / 2;
  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (cached_freqs_cos == nullptr || cached_freqs_sin == nullptr) {
      const std::lock_guard<std::mutex> lock(rope_init_mtx);
      precompute_freqs(head_dim, max_position_embeddings, theta, false);
      cached_freqs_cos = freqs_cos;
      cached_freqs_sin = freqs_sin;
    }
    std::vector<std::vector<float>> *freqs_cos_local = cached_freqs_cos;
    std::vector<std::vector<float>> *freqs_sin_local = cached_freqs_sin;
    std::vector<float> *cos_ = nullptr;
    std::vector<float> *sin_ = nullptr;

    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &(*freqs_cos_local)[from + h];
            sin_ = &(*freqs_sin_local)[from + h];
          }
          float *in_ptr = in.getData<float>() +
                          b * in.channel() * in.height() * in.width() +
                          c * in.height() * in.width() + h * in.width();

          if (out.getDataType() == ml::train::TensorDim::DataType::FP32) {
            float *out_ptr = out.getData<float>() +
                             b * out.channel() * out.height() * out.width() +
                             c * out.height() * out.width() + h * out.width();

            if (out_ptr != in_ptr) {
              std::memcpy(out_ptr, in_ptr, sizeof(float) * in.width());
            }
            if (apply_rope) {
              nntrainer::compute_rotary_emb_value(
                in.width(), dim, half_, out_ptr, nullptr, cos_->data(),
                sin_->data(), false);
            }
          } else if (out.getDataType() ==
                       ml::train::TensorDim::DataType::UINT16 ||
                     out.getDataType() ==
                       ml::train::TensorDim::DataType::FP16) {
            uint16_t *out_ptr = out.getData<uint16_t>() +
                                b * out.channel() * out.height() * out.width() +
                                c * out.height() * out.width() +
                                h * out.width();

            nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                                out_ptr, cos_->data(),
                                                sin_->data(), !apply_rope);
          }
        }
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (cached_freqs_cos_fp16 == nullptr || cached_freqs_sin_fp16 == nullptr) {
      const std::lock_guard<std::mutex> lock(rope_init_mtx);
      // Cap to the live max sequence length (= the `from < max_timestep` bound
      // used below), not the model's max_position_embeddings (131072 for
      // gemma4). The host-RoPE table is indexed [from + h] with from+height <=
      // max_timestep, so rope_lut_positions() positions suffice. Mirrors the
      // CUDA/OpenCL GPU-RoPE paths which already cap; avoids a 128K-position
      // trig build on any platform that falls back to host RoPE.
      precompute_freqs(head_dim, rope_lut_positions(), theta, true);
      cached_freqs_cos_fp16 = freqs_cos_fp16;
      cached_freqs_sin_fp16 = freqs_sin_fp16;
    }
    std::vector<std::vector<_FP16>> *freqs_cos_fp16_local =
      cached_freqs_cos_fp16;
    std::vector<std::vector<_FP16>> *freqs_sin_fp16_local =
      cached_freqs_sin_fp16;
    std::vector<_FP16> *cos_ = nullptr;
    std::vector<_FP16> *sin_ = nullptr;

    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &(*freqs_cos_fp16_local)[from + h];
            sin_ = &(*freqs_sin_fp16_local)[from + h];
          }
          _FP16 *in_ptr = in.getData<_FP16>() +
                          b * in.channel() * in.height() * in.width() +
                          c * in.height() * in.width() + h * in.width();
          _FP16 *out_ptr = out.getData<_FP16>() +
                           b * out.channel() * out.height() * out.width() +
                           c * out.height() * out.width() + h * out.width();

          nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                              out_ptr, cos_->data(),
                                              sin_->data());
        }
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
                                    size_t num_head, unsigned int from) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] =
          std::tanh(qk_out_[i] * inv_softcapping) * attn_logit_softcapping;
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      // Iterate over ALL rows (not just min(row, window)) so that every query
      // row in a long prefill gets softmaxed over the correct windowed range.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head);
      });
    }
  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] = (_FP16)(std::tanh((float)qk_out_[i] * inv_softcapping) *
                             attn_logit_softcapping);
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      // Iterate over ALL rows (not just min(row, window)) so that every query
      // row in a long prefill gets softmaxed over the correct windowed range.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
      });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
                                    size_t num_head, unsigned int from,
                                    nntrainer::Tensor &sink_step) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] =
          std::tanh(qk_out_[i] * inv_softcapping) * attn_logit_softcapping;
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        unsigned int to = from + row;
        end_row = to;
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head,
                                     sink_step.getData());
    } else {
      // Iterate over ALL rows (not just min(row, window)) for correct windowed
      // prefill when sequence_len > local_window_size.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(i + from) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head,
                               sink_step.getData());
      });
    }
  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();
    _FP16 *sink_step_ = sink_step.getData<_FP16>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] = (_FP16)(std::tanh((float)qk_out_[i] * inv_softcapping) *
                             attn_logit_softcapping);
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head,
                                     sink_step_);
    } else {
      // Iterate over ALL rows (not just min(row, window)) for correct windowed
      // prefill when sequence_len > local_window_size.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(i + from) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head,
                               sink_step_);
      });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::compute_fp16vcache_transposed(
  nntrainer::Tensor &in, nntrainer::Tensor &vcache, nntrainer::Tensor &output,
  int from, int num_cache_head, int gqa_size, int head_dim, int to) {

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if ((to - from) != 1) {
      // Iterate over ALL output rows so every query row gets an output even
      // when (to - from) > local_window_size.
      int total = to - from;
      if (!is_causal)
        total = to - from;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(total), [=](size_t i) {
        size_t start_idx;
        if (is_causal) {
          start_idx =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
        } else {
          start_idx = i * to; // linear index
        }
        const float *input =
          in.getData<float>() + start_idx * num_cache_head * gqa_size;
        float *out =
          output.getData<float>() + i * (num_cache_head * gqa_size * head_dim);

        int row_num = is_causal ? (from + (int)i) : to - 1;
        if (vcache.getDataType() == ml::train::TensorDim::DataType::FP32) {
          compute_vcache_fp32_transposed_reference(
            row_num, input, vcache.getData<float>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        } else {
          nntrainer::compute_fp16vcache_fp32_transposed(
            row_num, input, vcache.getData<uint16_t>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        }
      });
    } else {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_num = to - 1;

      // Use OpenMP for lower overhead parallelization during decoding
      const float *in_data = in.getData<float>();
      float *output_data = output.getData<float>();

      auto &tm = nntrainer::ThreadManager::Global();
      if (vcache.getDataType() == ml::train::TensorDim::DataType::FP32) {
        const float *vcache_data = vcache.getData<float>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            compute_vcache_fp32_transposed_reference(
              row_num, in_data, vcache_data, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
          });
      } else {
        const uint16_t *vcache_data = vcache.getData<uint16_t>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            nntrainer::compute_fp16vcache_fp32_transposed(
              row_num, in_data, vcache_data, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
          });
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if ((to - from) != 1) {
      // Iterate over ALL output rows so every query row gets an output even
      // when (to - from) > local_window_size.
      int total = to - from;
      if (!is_causal)
        total = to - from;

      auto &tm = nntrainer::ThreadManager::Global();
      if (kv_int8) {
        const int8_t *vcache_i8 =
          reinterpret_cast<const int8_t *>(vcache.getData<uint8_t>());
        const uint16_t *vscale = cur_kv_int8_value_scale_batch;
        NNTR_THROW_IF(vscale == nullptr, std::invalid_argument)
          << "kv_int8 V read path missing per-batch V-scale pointer";
        // Iterate over ALL output rows + windowed cumulative offset (PR #3989).
        tm.parallel_for(0, static_cast<size_t>(total), [=](size_t i) {
          size_t start_idx;
          if (is_causal) {
            start_idx = calc_windowed_attn_index(from + i) -
                        calc_windowed_attn_index(from);
          } else {
            start_idx = i * to;
          }
          const _FP16 *input =
            in.getData<_FP16>() + start_idx * num_cache_head * gqa_size;
          _FP16 *out_p = output.getData<_FP16>() +
                         i * (num_cache_head * gqa_size * head_dim);
          int row_num = is_causal ? (from + (int)i) : to - 1;
          compute_fp16vcache_transposed_int8(
            row_num, input, vcache_i8, vscale, out_p, num_cache_head, gqa_size,
            head_dim, local_window_size, 0, -1);
        });
      } else {
        tm.parallel_for(0, static_cast<size_t>(total), [=](size_t i) {
          size_t start_idx;
          if (is_causal) {
            start_idx = calc_windowed_attn_index(from + i) -
                        calc_windowed_attn_index(from);
          } else {
            start_idx = i * to;
          }
          const _FP16 *input =
            in.getData<_FP16>() + start_idx * num_cache_head * gqa_size;
          _FP16 *out = output.getData<_FP16>() +
                       i * (num_cache_head * gqa_size * head_dim);
          int row_num = is_causal ? (from + (int)i) : to - 1;
          nntrainer::compute_fp16vcache_transposed(
            row_num, input, vcache.getData<_FP16>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        });
      }
    } else {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_num = to - 1;

      // Use OpenMP for lower overhead parallelization during decoding
      const _FP16 *in_data = in.getData<_FP16>();
      _FP16 *output_data = output.getData<_FP16>();

      auto &tm_fp16 = nntrainer::ThreadManager::Global();
      if (kv_int8) {
        const int8_t *vcache_i8 =
          reinterpret_cast<const int8_t *>(vcache.getData<uint8_t>());
        const uint16_t *vscale = cur_kv_int8_value_scale_batch;
        NNTR_THROW_IF(vscale == nullptr, std::invalid_argument)
          << "kv_int8 V read path (decode) missing per-batch V-scale pointer";
        tm_fp16.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            compute_fp16vcache_transposed_int8(
              row_num, in_data, vcache_i8, vscale, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, (int)head_kv,
              (int)head_kv + 1);
          });
      } else {
        const _FP16 *vcache_data = vcache.getData<_FP16>();
        tm_fp16.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            nntrainer::compute_fp16vcache_transposed(
              row_num, in_data, vcache_data, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
          });
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::setBatch(nntrainer::RunLayerContext &context,
                            unsigned int batch) {

  const float dropout_rate =
    std::get<nntrainer::props::DropOutRate>(mha_core_props).get();
  context.updateTensor(tensor_idx[AttentionParams::cache_key], batch);
  context.updateTensor(tensor_idx[AttentionParams::cache_value], batch);
  // context.updateTensor(tensor_idx[AttentionParams::attention_weight], batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(tensor_idx[AttentionParams::dropout_mask], batch);
  }
}

void MHACoreLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  unsigned int height = input_dimensions[0].height();
  unsigned int &max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();
  unsigned int &max_new_tokens =
    std::get<props::MaxNewTokens>(mha_core_props).get();
  max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mha_core_props).get();
  max_timestep = height + max_new_tokens;

  ml::train::TensorDim kv_dim = input_dimensions[0];
  kv_dim.width(kv_dim.width() / (num_heads_Q / num_heads_KV));

  ml::train::TensorDim kv_cache_dim = kv_dim;
#ifdef ENABLE_FP16
  kv_cache_dim.setDataType(ml::train::TensorDim::DataType::FP16);
#else
  kv_cache_dim.setDataType(ml::train::TensorDim::DataType::UINT16);
#endif
  kv_cache_dim.height(max_timestep);

  context.updateInput(INOUT_INDEX::QUERY, input_dimensions[0]);
  context.updateInput(INOUT_INDEX::KEY, kv_dim);
  context.updateInput(INOUT_INDEX::VALUE, kv_dim);
  context.updateOutput(0, input_dimensions[0]);

  context.updateTensor(tensor_idx[AttentionParams::cache_key], kv_cache_dim);
  context.updateTensor(tensor_idx[AttentionParams::cache_value], kv_cache_dim);
}

void MHACoreLayer::calcDerivative(nntrainer::RunLayerContext &context) {}

void MHACoreLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void MHACoreLayer::exportTo(nntrainer::Exporter &exporter,
                            const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(mha_core_props, method, this);
}

void MHACoreLayer::setProperty(const std::vector<std::string> &values) {
  std::vector<std::string> props;
  props.reserve(values.size());
  for (const auto &value : values) {
    std::string key;
    std::string parsed_value;
    if (nntrainer::getKeyValue(value, key, parsed_value) == ML_ERROR_NONE &&
        key == "cache_index") {
      setCacheIndex(static_cast<unsigned int>(std::stoul(parsed_value)));
    } else {
      props.push_back(value);
    }
  }

  auto remain_props = loadProperties(props, mha_core_props);
  LayerImpl::setProperty(remain_props);
  cached_freqs_cos = nullptr;
  cached_freqs_sin = nullptr;
#ifdef ENABLE_FP16
  cached_freqs_cos_fp16 = nullptr;
  cached_freqs_sin_fp16 = nullptr;
#endif
}

size_t MHACoreLayer::calc_attn_index(size_t i) { return (i * (i + 1)) / 2; };

size_t MHACoreLayer::calc_windowed_attn_index(size_t i) {
  // S(i) = sum_{k=0}^{i-1} min(k+1, W)
  // For i <= W:  S(i) = i*(i+1)/2   (same as full-attention triangular index)
  // For i >  W:  S(i) = W*(W+1)/2 + (i - W)*W
  // When W == UINT_MAX, i <= W is always true, so we never evaluate
  // W*(W+1)/2 and there is no overflow.
  if (i <= local_window_size) {
    return (i * (i + 1)) / 2;
  } else {
    return (local_window_size * (local_window_size + 1)) / 2 +
           (i - local_window_size) * local_window_size;
  }
};

#ifdef PLUGGABLE

nntrainer::Layer *create_mha_core_layer() {
  auto layer = new MHACoreLayer();
  return layer;
}

void destroy_mha_core_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_mha_core_layer,
                                                   destroy_mha_core_layer};
}

#endif

} // namespace causallm

// SPDX-License-Identifier: Apache-2.0
/**
 * HexKL SDKL backend for nntrainer.
 *
 * Offloads prefill-phase FP32 SGEMM (M > 1) to the Snapdragon HMX unit via
 * the HexKL CPU Macro API (libsdkl.so / sdkl.h).
 *
 * Design:
 *   g_wh_cache  — per-weight WH-layout FP16 buffer (regular malloc, keyed on
 *                 (B_ptr, N, K, TransB)).  Built on first call then reused.
 *                 Avoids exhausting sdkl_npu_alloc pool (no per-weight alloc).
 *   g_W_buf     — single sdkl_npu_alloc buffer, sized for the largest weight.
 *                 Per dispatch: WH-layout is memcpy'd here from g_wh_cache
 *                 (~0.24ms) instead of full FP32→FP16+WH conversion (~3.5ms).
 *   g_X_stage   — malloc, largest padded activation [M_pad × K].
 *   g_C_stage   — malloc, largest padded output    [M_pad × N].
 *
 * HMX constraint: n_row (M) must be a multiple of 32; rows are zero-padded.
 *
 * HMX lock: acquired only around the DSP dispatch (not held idle) so CDSP
 * power-state does not throttle CPU during decode between prefill steps.
 */

#ifdef USE_HMX

// remote.h constants needed by sdkl.h enum body — define them inline to
// avoid depending on the full Hexagon SDK headers at build time.
#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif
#ifndef CDSP1_DOMAIN_ID
#define CDSP1_DOMAIN_ID 4
#endif

#include "hexkl_backend.h"

// sdkl.h uses C99 'restrict' which is not a C++ keyword. Map it.
#ifndef restrict
#define restrict __restrict__
#endif
#include <sdkl.h>
#undef restrict

#include <android/log.h>
#include <arm_neon.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <time.h>
#include <unordered_map>

#define HEXKL_LOG_TAG "nntr_hexkl"
#define HEXKL_LOGI(...) \
  __android_log_print(ANDROID_LOG_INFO,  HEXKL_LOG_TAG, __VA_ARGS__)
#define HEXKL_LOGE(...) \
  __android_log_print(ANDROID_LOG_ERROR, HEXKL_LOG_TAG, __VA_ARGS__)

namespace nntrainer {
namespace hexkl {

// ---------------------------------------------------------------------------
// Per-weight WH-layout cache (regular malloc — avoids sdkl_npu_alloc pool)
// ---------------------------------------------------------------------------

struct WeightKey {
  uintptr_t    ptr;
  unsigned int n_col;   // N
  unsigned int n_inner; // K
  bool         transB;
  bool operator==(const WeightKey &o) const {
    return ptr == o.ptr && n_col == o.n_col &&
           n_inner == o.n_inner && transB == o.transB;
  }
};

struct WeightKeyHash {
  size_t operator()(const WeightKey &k) const noexcept {
    size_t h = std::hash<uintptr_t>{}(k.ptr);
    h ^= std::hash<unsigned>{}(k.n_col)   + 0x9e3779b9u + (h << 6) + (h >> 2);
    h ^= std::hash<unsigned>{}(k.n_inner) + 0x9e3779b9u + (h << 6) + (h >> 2);
    h ^= std::hash<bool>{}(k.transB)      + 0x9e3779b9u + (h << 6) + (h >> 2);
    return h;
  }
};

// FP16 WH cache (used by sgemm_hmx / shgemm_hmx paths)
static std::unordered_map<WeightKey, _Float16 *, WeightKeyHash> g_wh_cache;

// INT8 WH cache (used by sgemm_hmx_i8 path)
struct WeightI8Entry {
  int8_t* wh_buf;    // malloc'd INT8 WH layout [N×K]
  float*  bias128;   // malloc'd float[N]: 128.0f * sum_k(W_i8[n,k])
  float   w_scale;
};
static std::unordered_map<WeightKey, WeightI8Entry, WeightKeyHash> g_wh_i8_cache;

static std::mutex g_cache_mutex;

// ---------------------------------------------------------------------------
// Persistent shared buffers — grown as needed, freed at finalize()
// ---------------------------------------------------------------------------

static std::mutex g_hmx_mutex;   // serializes HMX dispatch + g_W_buf/stage access

// FP16 path — single sdkl_npu_alloc W buffer; X/C are plain malloc.
static _Float16 *g_W_buf         = nullptr;
static size_t    g_W_buf_bytes   = 0;
static float    *g_X_stage       = nullptr;
static size_t    g_X_stage_bytes = 0;
static float    *g_C_stage       = nullptr;
static size_t    g_C_stage_bytes = 0;

// INT8 path — three sdkl_npu_alloc buffers (W, X, C all NPU-visible).
static int8_t   *g_W_i8_buf      = nullptr;
static size_t    g_W_i8_buf_bytes = 0;
static uint8_t  *g_X_u8_buf      = nullptr;
static size_t    g_X_u8_buf_bytes = 0;
static int32_t  *g_C_i32_buf     = nullptr;
static size_t    g_C_i32_buf_bytes = 0;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

static bool           g_initialized = false;
static int            g_domain      = CDSP_DOMAIN_ID;
static std::once_flag g_init_flag;

static inline int64_t now_us() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000LL;
}

// ---------------------------------------------------------------------------
// NEON-vectorized FP32→FP16 conversion with optional transpose.
//
// For TransB=true  (B is [N×K] row-major, stride ldb):
//   dst[n*K + k] = (f16) B[n*ldb + k]   — sequential reads, no reorder.
//
// For TransB=false (B is [K×N] row-major, stride ldb):
//   dst[n*K + k] = (f16) B[k*ldb + n]   — transpose required.
//   Uses blocked traversal (BK×BN tiles) so B reads stay sequential
//   within each tile; NEON converts 4 FP32 → 4 FP16 per cycle.
// ---------------------------------------------------------------------------
static void f32_to_f16_convert(
    _Float16 *__restrict__ dst,
    const float *__restrict__ B,
    unsigned N, unsigned K, unsigned ldb, bool TransB)
{
  if (TransB) {
    // B is [N×K] — straight copy row by row, NEON 8-wide
    for (unsigned n = 0; n < N; ++n) {
      const float *brow = B + (size_t)n * ldb;
      _Float16    *drow = dst + (size_t)n * K;
      unsigned k = 0;
      for (; k + 8 <= K; k += 8) {
        float32x4_t v0 = vld1q_f32(brow + k);
        float32x4_t v1 = vld1q_f32(brow + k + 4);
        vst1q_f16((__fp16 *)(drow + k),
                  vcombine_f16(vcvt_f16_f32(v0), vcvt_f16_f32(v1)));
      }
      for (; k < K; ++k) drow[k] = (_Float16)brow[k];
    }
  } else {
    // B is [K×N] — blocked transpose + NEON 4-wide conversion.
    // Outer tile over k keeps B reads sequential; inner tile over n
    // keeps the dst scatter writes L1-local.
    const unsigned BK = 4, BN = 32;
    for (unsigned k0 = 0; k0 < K; k0 += BK) {
      unsigned k_end = k0 + BK < K ? k0 + BK : K;
      for (unsigned n0 = 0; n0 < N; n0 += BN) {
        unsigned n_end = n0 + BN < N ? n0 + BN : N;
        for (unsigned k = k0; k < k_end; ++k) {
          const float *brow = B + (size_t)k * ldb + n0;
          unsigned n = n0;
          for (; n + 4 <= n_end; n += 4) {
            float32x4_t v = vld1q_f32(brow + (n - n0));
            __fp16 tmp[4]; vst1_f16(tmp, vcvt_f16_f32(v));
            dst[(size_t)n     * K + k] = (_Float16)tmp[0];
            dst[(size_t)(n+1) * K + k] = (_Float16)tmp[1];
            dst[(size_t)(n+2) * K + k] = (_Float16)tmp[2];
            dst[(size_t)(n+3) * K + k] = (_Float16)tmp[3];
          }
          for (; n < n_end; ++n)
            dst[(size_t)n * K + k] = (_Float16)brow[n - n0];
        }
      }
    }
  }
}

// Accumulated timing (reset each forward pass by tracking call count).
static int64_t g_t_cache_us = 0, g_t_memcpy_us = 0,
               g_t_lock_us = 0, g_t_dsp_us = 0, g_t_unlock_us = 0;
static int     g_call_count = 0;

void initialize() {
  std::call_once(g_init_flag, []() {
    int ret = sdkl_npu_initialize(g_domain, nullptr, nullptr);
    if (ret != 0) {
      HEXKL_LOGE("sdkl_npu_initialize failed: %d", ret);
      return;
    }
    // sdkl_npu_initialize locks HMX by default.  Release the lock
    // immediately; sgemm_hmx will acquire it per-dispatch so that
    // the CDSP does not stay at high-power between prefill matmuls
    // (which would throttle the CPU during decode).
    sdkl_npu_unlock_hmx(g_domain);
    char ver[SDKL_VERSION_STR_LEN] = {};
    sdkl_npu_get_version(g_domain, ver);
    HEXKL_LOGI("HexKL ready: %s", ver);
    g_initialized = true;
  });
}

void finalize() {
  if (!g_initialized) return;
  g_initialized = false;

  {
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    for (auto &kv : g_wh_cache) free(kv.second);
    g_wh_cache.clear();
    for (auto &kv : g_wh_i8_cache) {
      free(kv.second.wh_buf);
      free(kv.second.bias128);
    }
    g_wh_i8_cache.clear();
  }

  std::lock_guard<std::mutex> lock(g_hmx_mutex);
  if (g_W_buf)     { sdkl_npu_free(g_W_buf);     g_W_buf     = nullptr; g_W_buf_bytes     = 0; }
  if (g_X_stage)   { free(g_X_stage);             g_X_stage   = nullptr; g_X_stage_bytes   = 0; }
  if (g_C_stage)   { free(g_C_stage);             g_C_stage   = nullptr; g_C_stage_bytes   = 0; }
  if (g_W_i8_buf)  { sdkl_npu_free(g_W_i8_buf);  g_W_i8_buf  = nullptr; g_W_i8_buf_bytes  = 0; }
  if (g_X_u8_buf)  { sdkl_npu_free(g_X_u8_buf);  g_X_u8_buf  = nullptr; g_X_u8_buf_bytes  = 0; }
  if (g_C_i32_buf) { sdkl_npu_free(g_C_i32_buf); g_C_i32_buf = nullptr; g_C_i32_buf_bytes = 0; }

  sdkl_npu_finalize(g_domain);
  HEXKL_LOGI("HexKL finalized — total dispatch calls: %d", g_call_count);
}

// ---------------------------------------------------------------------------
// preload_weight_f32 / is_weight_cached — warm the WH cache without dispatch.
// Call this for each FC weight during model loading or decode steps so that
// the first real prefill (M > 1) hits the cache and skips the build cost.
// ---------------------------------------------------------------------------

// is_weight_cached / preload_weight_f32 operate on the INT8 cache since
// sgemm_hmx_i8 is the primary FP32-weight dispatch path.
bool is_weight_cached(bool TransB, unsigned N, unsigned K, const float *B) {
  if (!g_initialized) return false;
  WeightKey key{reinterpret_cast<uintptr_t>(B), N, K, TransB};
  std::lock_guard<std::mutex> cl(g_cache_mutex);
  return g_wh_i8_cache.count(key) != 0;
}

void preload_weight_f32(bool TransB, unsigned N, unsigned K,
                        const float *B, unsigned ldb) {
  if (!g_initialized) return;
  WeightKey key{reinterpret_cast<uintptr_t>(B), N, K, TransB};
  {
    std::lock_guard<std::mutex> cl(g_cache_mutex);
    if (g_wh_i8_cache.count(key)) return;
  }
  const size_t w_bytes = (size_t)N * K * sizeof(int8_t);
  int8_t* w_i8   = static_cast<int8_t*>(malloc(w_bytes));
  float*  bias128 = static_cast<float*>(malloc(N * sizeof(float)));
  if (!w_i8 || !bias128) { free(w_i8); free(bias128); return; }

  // Find max |B| for scale
  float max_abs = 1e-8f;
  if (TransB) {
    for (unsigned n = 0; n < N; ++n)
      for (unsigned k = 0; k < K; ++k) {
        float v = B[n*ldb+k]; if (v < 0) v = -v;
        if (v > max_abs) max_abs = v;
      }
  } else {
    for (unsigned k = 0; k < K; ++k)
      for (unsigned n = 0; n < N; ++n) {
        float v = B[k*ldb+n]; if (v < 0) v = -v;
        if (v > max_abs) max_abs = v;
      }
  }
  float w_scale = max_abs / 127.0f;
  float inv_scale = 1.0f / w_scale;
  if (TransB) {
    for (unsigned n = 0; n < N; ++n) {
      float bsum = 0.0f;
      for (unsigned k = 0; k < K; ++k) {
        int q = (int)rintf(B[n*ldb+k] * inv_scale);
        if (q > 127) q = 127; if (q < -127) q = -127;
        w_i8[n*K+k] = (int8_t)q; bsum += (float)q;
      }
      bias128[n] = 128.0f * bsum;
    }
  } else {
    for (unsigned n = 0; n < N; ++n) bias128[n] = 0.0f;
    for (unsigned k = 0; k < K; ++k)
      for (unsigned n = 0; n < N; ++n) {
        int q = (int)rintf(B[k*ldb+n] * inv_scale);
        if (q > 127) q = 127; if (q < -127) q = -127;
        w_i8[n*K+k] = (int8_t)q; bias128[n] += (float)q;
      }
    for (unsigned n = 0; n < N; ++n) bias128[n] *= 128.0f;
  }
  if (sdkl_cpu_rm_to_wh_i8_inplace(N, K, w_i8) != 0) {
    free(w_i8); free(bias128); return;
  }
  WeightI8Entry entry{w_i8, bias128, w_scale};
  std::lock_guard<std::mutex> cl(g_cache_mutex);
  auto [it, inserted] = g_wh_i8_cache.emplace(key, entry);
  if (!inserted) { free(w_i8); free(bias128); }
}

// ---------------------------------------------------------------------------
// INT8 helpers
// ---------------------------------------------------------------------------

// Quantize FP32 weight B[N×K] (or B[K×N] if !TransB) → INT8 WH layout.
// Returns w_scale; fills out_bias128[n] = 128 * sum_k(W_i8[n,k]).
static float quantize_weight_fp32_to_i8(
    int8_t* __restrict__ out_i8,
    float*  __restrict__ out_bias128,
    const float* __restrict__ B,
    unsigned N, unsigned K, unsigned ldb, bool TransB)
{
  float max_abs = 1e-8f;
  if (TransB) {
    for (unsigned n = 0; n < N; ++n)
      for (unsigned k = 0; k < K; ++k) {
        float v = B[n*ldb+k]; if (v < 0) v = -v;
        if (v > max_abs) max_abs = v;
      }
  } else {
    for (unsigned k = 0; k < K; ++k)
      for (unsigned n = 0; n < N; ++n) {
        float v = B[k*ldb+n]; if (v < 0) v = -v;
        if (v > max_abs) max_abs = v;
      }
  }
  const float w_scale   = max_abs / 127.0f;
  const float inv_scale = 1.0f   / w_scale;

  if (TransB) {
    for (unsigned n = 0; n < N; ++n) {
      const float* brow = B + (size_t)n * ldb;
      int8_t*      drow = out_i8 + (size_t)n * K;
      float bsum = 0.0f;
      unsigned k = 0;
      for (; k + 4 <= K; k += 4) {
        float32x4_t v = vmulq_n_f32(vld1q_f32(brow + k), inv_scale);
        int32x4_t   q = vcvtaq_s32_f32(v);
        q = vmaxq_s32(vminq_s32(q, vdupq_n_s32(127)), vdupq_n_s32(-127));
        int32_t tmp[4]; vst1q_s32(tmp, q);
        drow[k]   = (int8_t)tmp[0]; bsum += (float)tmp[0];
        drow[k+1] = (int8_t)tmp[1]; bsum += (float)tmp[1];
        drow[k+2] = (int8_t)tmp[2]; bsum += (float)tmp[2];
        drow[k+3] = (int8_t)tmp[3]; bsum += (float)tmp[3];
      }
      for (; k < K; ++k) {
        int q = (int)rintf(brow[k] * inv_scale);
        if (q > 127) q = 127; if (q < -127) q = -127;
        drow[k] = (int8_t)q; bsum += (float)q;
      }
      out_bias128[n] = 128.0f * bsum;
    }
  } else {
    // B is [K×N] — produce out_i8 as [N×K]
    for (unsigned n = 0; n < N; ++n) out_bias128[n] = 0.0f;
    for (unsigned k = 0; k < K; ++k) {
      const float* brow = B + (size_t)k * ldb;
      for (unsigned n = 0; n < N; ++n) {
        int q = (int)rintf(brow[n] * inv_scale);
        if (q > 127) q = 127; if (q < -127) q = -127;
        out_i8[(size_t)n * K + k] = (int8_t)q;
        out_bias128[n] += (float)q;
      }
    }
    for (unsigned n = 0; n < N; ++n) out_bias128[n] *= 128.0f;
  }
  return w_scale;
}

// Quantize FP32 activations A[M×K] (contiguous) → UINT8 x_u8[M_pad×K].
// Zero-point = 128 (symmetric: 0.0f maps to 128).
// Returns x_scale.
static float quant_fp32_to_u8(uint8_t* __restrict__ x_u8,
                               const float* __restrict__ A,
                               unsigned M, unsigned K, unsigned M_pad)
{
  const unsigned total = M * K;
  // NEON find max |A|
  float32x4_t vmax4 = vdupq_n_f32(0.0f);
  unsigned i = 0;
  for (; i + 4 <= total; i += 4)
    vmax4 = vmaxq_f32(vmax4, vabsq_f32(vld1q_f32(A + i)));
  float max_abs = vmaxvq_f32(vmax4);
  for (; i < total; ++i) {
    float v = A[i]; if (v < 0) v = -v;
    if (v > max_abs) max_abs = v;
  }
  if (max_abs < 1e-8f) max_abs = 1e-8f;
  const float x_scale   = max_abs / 127.0f;
  const float inv_scale = 1.0f   / x_scale;

  float32x4_t vscale = vdupq_n_f32(inv_scale);
  float32x4_t voff   = vdupq_n_f32(128.0f);
  i = 0;
  for (; i + 8 <= total; i += 8) {
    float32x4_t v0 = vmlaq_f32(voff, vld1q_f32(A + i),     vscale);
    float32x4_t v1 = vmlaq_f32(voff, vld1q_f32(A + i + 4), vscale);
    int32x4_t   i0 = vcvtaq_s32_f32(v0);
    int32x4_t   i1 = vcvtaq_s32_f32(v1);
    // saturating narrow s32→u16→u8: clamps to [0, 255]
    uint8x8_t u8 = vqmovn_u16(
                     vcombine_u16(vqmovun_s32(i0), vqmovun_s32(i1)));
    vst1_u8(x_u8 + i, u8);
  }
  for (; i < total; ++i) {
    int q = (int)rintf(A[i] * inv_scale) + 128;
    x_u8[i] = (uint8_t)(q < 0 ? 0 : q > 255 ? 255 : q);
  }
  // Pad remaining rows with 128 (encodes 0.0f)
  if (M_pad > M)
    memset(x_u8 + total, 128, (size_t)(M_pad - M) * K);
  return x_scale;
}

// Dequantize INT32 output → FP32 with bias correction:
// C[m,n] = combined * (C_i32[m,n] - bias128[n])
// where combined = x_scale * w_scale, bias128[n] = 128 * sum_k(W_i8[n,k]).
static void dequant_i32_to_f32(float* __restrict__ C, unsigned ldc,
                                const int32_t* __restrict__ C_i32,
                                unsigned N,
                                const float* __restrict__ bias128,
                                float combined, unsigned M)
{
  float32x4_t vcomb = vdupq_n_f32(combined);
  for (unsigned m = 0; m < M; ++m) {
    const int32_t* row_i = C_i32 + (size_t)m * N;
    float*         row_o = C     + (size_t)m * ldc;
    unsigned n = 0;
    for (; n + 4 <= N; n += 4) {
      float32x4_t vi = vcvtq_f32_s32(vld1q_s32(row_i + n));
      float32x4_t vb = vld1q_f32(bias128 + n);
      vst1q_f32(row_o + n, vmulq_f32(vcomb, vsubq_f32(vi, vb)));
    }
    for (; n < N; ++n)
      row_o[n] = combined * ((float)row_i[n] - bias128[n]);
  }
}

// INT8 timing accumulators (separate from FP16 g_t_* counters)
static int64_t g_i8_cache_us = 0, g_i8_quant_us = 0,
               g_i8_lock_us  = 0, g_i8_dsp_us   = 0,
               g_i8_unlock_us = 0, g_i8_deq_us   = 0;
static int     g_i8_call_count = 0;

// ---------------------------------------------------------------------------
// sgemm_hmx_i8: FP32×FP32 → INT8 quantize → HMX u8i8_i32 → FP32 dequant
// ---------------------------------------------------------------------------

bool sgemm_hmx_i8(bool TransB,
                  unsigned int M, unsigned int N, unsigned int K,
                  const float *A, unsigned int lda,
                  const float *B, unsigned int ldb,
                  float *C, unsigned int ldc) {
  if (!g_initialized) return false;

  // INT8 HMX: M must be a multiple of 64.
  const unsigned M_pad    = ((M + 63u) / 64u) * 64u;
  const size_t   w_bytes  = (size_t)N * K * sizeof(int8_t);
  const size_t   xu_bytes = (size_t)M_pad * K * sizeof(uint8_t);
  const size_t   ci_bytes = (size_t)M_pad * N * sizeof(int32_t);

  int64_t t0, t1, t2, t3, t4, t5, t6;
  t0 = now_us();

  // --- 1. Weight cache lookup or build ---
  WeightKey key{reinterpret_cast<uintptr_t>(B), N, K, TransB};
  WeightI8Entry entry{};
  {
    std::lock_guard<std::mutex> cl(g_cache_mutex);
    auto it = g_wh_i8_cache.find(key);
    if (it != g_wh_i8_cache.end()) entry = it->second;
  }
  if (!entry.wh_buf) {
    int8_t* w_i8   = static_cast<int8_t*>(malloc(w_bytes));
    float*  bias128 = static_cast<float*>(malloc(N * sizeof(float)));
    if (!w_i8 || !bias128) {
      free(w_i8); free(bias128);
      HEXKL_LOGE("malloc failed w_bytes=%zu N=%u", w_bytes, N);
      return false;
    }
    int64_t tq0 = now_us();
    float w_scale = quantize_weight_fp32_to_i8(w_i8, bias128, B, N, K, ldb, TransB);
    int64_t tq1 = now_us();
    if (sdkl_cpu_rm_to_wh_i8_inplace(N, K, w_i8) != 0) {
      HEXKL_LOGE("sdkl_cpu_rm_to_wh_i8_inplace failed N=%u K=%u", N, K);
      free(w_i8); free(bias128); return false;
    }
    int64_t tq2 = now_us();
    HEXKL_LOGI("I8 WH build N=%u K=%u: quant=%lldus wh=%lldus",
               N, K, (long long)(tq1-tq0), (long long)(tq2-tq1));

    WeightI8Entry new_entry{w_i8, bias128, w_scale};
    {
      std::lock_guard<std::mutex> cl(g_cache_mutex);
      auto [it, inserted] = g_wh_i8_cache.emplace(key, new_entry);
      if (!inserted) { free(w_i8); free(bias128); entry = it->second; }
      else           { entry = new_entry; }
    }
  }
  t1 = now_us();

  std::lock_guard<std::mutex> lock(g_hmx_mutex);

  // --- 2. Grow W_i8 NPU buffer if needed ---
  if (w_bytes > g_W_i8_buf_bytes) {
    if (g_W_i8_buf) sdkl_npu_free(g_W_i8_buf);
    g_W_i8_buf = nullptr; g_W_i8_buf_bytes = 0;
    if (sdkl_npu_alloc(w_bytes, reinterpret_cast<void**>(&g_W_i8_buf)) != 0 || !g_W_i8_buf) {
      HEXKL_LOGE("sdkl_npu_alloc W_i8(%zu) failed", w_bytes); return false;
    }
    g_W_i8_buf_bytes = w_bytes;
    HEXKL_LOGI("W_i8_buf grown to %zu bytes", w_bytes);
  }
  memcpy(g_W_i8_buf, entry.wh_buf, w_bytes);

  // --- 3. Grow X_u8 NPU buffer if needed ---
  if (xu_bytes > g_X_u8_buf_bytes) {
    if (g_X_u8_buf) sdkl_npu_free(g_X_u8_buf);
    g_X_u8_buf = nullptr; g_X_u8_buf_bytes = 0;
    if (sdkl_npu_alloc(xu_bytes, reinterpret_cast<void**>(&g_X_u8_buf)) != 0 || !g_X_u8_buf) {
      HEXKL_LOGE("sdkl_npu_alloc X_u8(%zu) failed", xu_bytes); return false;
    }
    g_X_u8_buf_bytes = xu_bytes;
  }

  // --- 4. Grow C_i32 NPU buffer if needed ---
  if (ci_bytes > g_C_i32_buf_bytes) {
    if (g_C_i32_buf) sdkl_npu_free(g_C_i32_buf);
    g_C_i32_buf = nullptr; g_C_i32_buf_bytes = 0;
    if (sdkl_npu_alloc(ci_bytes, reinterpret_cast<void**>(&g_C_i32_buf)) != 0 || !g_C_i32_buf) {
      HEXKL_LOGE("sdkl_npu_alloc C_i32(%zu) failed", ci_bytes); return false;
    }
    g_C_i32_buf_bytes = ci_bytes;
  }

  // --- 5. Quantize activations FP32 → UINT8 (NEON, per-dispatch) ---
  float x_scale = quant_fp32_to_u8(g_X_u8_buf, A, M, K, M_pad);
  t2 = now_us();

  // --- 6. HMX dispatch ---
  if (sdkl_npu_lock_hmx(g_domain) != 0) {
    HEXKL_LOGE("sdkl_npu_lock_hmx failed"); return false;
  }
  t3 = now_us();

  int ret = sdkl_npu_mm_u8i8_i32(g_domain, (int)M_pad, (int)N, (int)K,
                                  g_C_i32_buf, g_X_u8_buf, g_W_i8_buf);
  t4 = now_us();
  sdkl_npu_unlock_hmx(g_domain);
  t5 = now_us();

  // --- 7. Dequantize INT32 → FP32 with bias correction ---
  if (ret == 0) {
    float combined = x_scale * entry.w_scale;
    dequant_i32_to_f32(C, ldc, g_C_i32_buf, N, entry.bias128, combined, M);
  } else {
    HEXKL_LOGE("sdkl_npu_mm_u8i8_i32 failed: %d (M=%u M_pad=%u N=%u K=%u)",
               ret, M, M_pad, N, K);
  }
  t6 = now_us();

  g_i8_cache_us  += t1 - t0;
  g_i8_quant_us  += t2 - t1;
  g_i8_lock_us   += t3 - t2;
  g_i8_dsp_us    += t4 - t3;
  g_i8_unlock_us += t5 - t4;
  g_i8_deq_us    += t6 - t5;
  ++g_i8_call_count;

  if (g_i8_call_count == 1 || g_i8_call_count % 196 == 0) {
    HEXKL_LOGI("I8_TIMING call=%d M=%u N=%u K=%u: "
               "cache=%lldus quant=%lldus lock=%lldus dsp=%lldus unlock=%lldus deq=%lldus",
               g_i8_call_count, M, N, K,
               (long long)g_i8_cache_us,  (long long)g_i8_quant_us,
               (long long)g_i8_lock_us,   (long long)g_i8_dsp_us,
               (long long)g_i8_unlock_us, (long long)g_i8_deq_us);
  }
  if (g_i8_call_count % 196 == 0) {
    int64_t total = g_i8_cache_us + g_i8_quant_us + g_i8_lock_us +
                    g_i8_dsp_us + g_i8_unlock_us + g_i8_deq_us;
    HEXKL_LOGI("I8_PASS call=%d total=%lldms", g_i8_call_count, (long long)(total/1000));
    g_i8_cache_us = g_i8_quant_us = g_i8_lock_us =
    g_i8_dsp_us   = g_i8_unlock_us = g_i8_deq_us = 0;
  }

  return ret == 0;
}

// ---------------------------------------------------------------------------
// sgemm_hmx
// ---------------------------------------------------------------------------

bool sgemm_hmx(bool TransB,
               unsigned int M, unsigned int N, unsigned int K,
               const float *A, unsigned int lda,
               const float *B, unsigned int ldb,
               float *C, unsigned int ldc) {
  if (!g_initialized) return false;

  // HMX requires n_row to be a multiple of 32.
  const unsigned int M_pad    = ((M + 31u) / 32u) * 32u;
  const size_t       w_bytes  = (size_t)N * K * sizeof(_Float16);
  const size_t       x_bytes  = (size_t)M_pad * K * sizeof(float);
  const size_t       c_bytes  = (size_t)M_pad * N * sizeof(float);

  int64_t t0, t1, t2, t3, t4, t5;
  t0 = now_us();

  // Look up (or build) the WH-layout FP16 weight in regular malloc memory.
  WeightKey key{reinterpret_cast<uintptr_t>(B), N, K, TransB};
  _Float16 *wh_buf = nullptr;
  {
    std::lock_guard<std::mutex> cache_lock(g_cache_mutex);
    auto it = g_wh_cache.find(key);
    if (it != g_wh_cache.end()) wh_buf = it->second;
  }
  if (!wh_buf) {
    wh_buf = static_cast<_Float16 *>(malloc(w_bytes));
    if (!wh_buf) { HEXKL_LOGE("malloc wh_buf(%zu) failed", w_bytes); return false; }

    // NEON-vectorized FP32→FP16 (+ transpose for TransB=false).
    f32_to_f16_convert(wh_buf, B, N, K, ldb, TransB);
    int64_t t_cvt = now_us();

    if (sdkl_cpu_rm_to_wh_f16_inplace(N, K, wh_buf) != 0) {
      HEXKL_LOGE("sdkl_cpu_rm_to_wh_f16_inplace failed N=%u K=%u", N, K);
      free(wh_buf);
      return false;
    }
    int64_t t_wh = now_us();
    // Split log on first build: shows convert vs WH-layout cost separately.
    HEXKL_LOGI("WH build N=%u K=%u: cvt=%lldus wh=%lldus",
               N, K, (long long)(t_cvt - t0), (long long)(t_wh - t_cvt));

    {
      std::lock_guard<std::mutex> cache_lock(g_cache_mutex);
      auto [it, inserted] = g_wh_cache.emplace(key, wh_buf);
      if (!inserted) { free(wh_buf); wh_buf = it->second; }
    }
  }

  t1 = now_us();

  std::lock_guard<std::mutex> lock(g_hmx_mutex);

  // Grow W buffer (sdkl_npu_alloc — single buffer for DSP access) if needed.
  if (w_bytes > g_W_buf_bytes) {
    if (g_W_buf) sdkl_npu_free(g_W_buf);
    g_W_buf = nullptr; g_W_buf_bytes = 0;
    if (sdkl_npu_alloc(w_bytes, reinterpret_cast<void **>(&g_W_buf)) != 0 || !g_W_buf) {
      HEXKL_LOGE("sdkl_npu_alloc W(%zu) failed N=%u K=%u", w_bytes, N, K);
      return false;
    }
    g_W_buf_bytes = w_bytes;
    HEXKL_LOGI("W_buf grown to %zu bytes", w_bytes);
  }
  // Copy WH-layout from cached malloc to the DSP-accessible sdkl_npu_alloc buffer.
  memcpy(g_W_buf, wh_buf, w_bytes);

  // Grow X staging buffer (malloc) if needed.
  if (x_bytes > g_X_stage_bytes) {
    free(g_X_stage);
    g_X_stage = static_cast<float *>(malloc(x_bytes));
    if (!g_X_stage) { g_X_stage_bytes = 0; HEXKL_LOGE("malloc X_stage(%zu) failed", x_bytes); return false; }
    g_X_stage_bytes = x_bytes;
  }

  // Grow C staging buffer (malloc) if needed.
  if (c_bytes > g_C_stage_bytes) {
    free(g_C_stage);
    g_C_stage = static_cast<float *>(malloc(c_bytes));
    if (!g_C_stage) { g_C_stage_bytes = 0; HEXKL_LOGE("malloc C_stage(%zu) failed", c_bytes); return false; }
    g_C_stage_bytes = c_bytes;
  }

  // Copy A into g_X_stage; zero-pad extra rows for M alignment.
  memcpy(g_X_stage, A, (size_t)M * K * sizeof(float));
  if (M_pad > M)
    memset(g_X_stage + (size_t)M * K, 0, (size_t)(M_pad - M) * K * sizeof(float));

  t2 = now_us();

  // Lock HMX only for the duration of the DSP dispatch.
  if (sdkl_npu_lock_hmx(g_domain) != 0) {
    HEXKL_LOGE("sdkl_npu_lock_hmx failed");
    return false;
  }
  t3 = now_us();

  int ret = sdkl_npu_mm_f32f16_f32(g_domain, (int)M_pad, (int)N, (int)K,
                                    g_C_stage, g_X_stage, g_W_buf);
  t4 = now_us();
  sdkl_npu_unlock_hmx(g_domain);
  t5 = now_us();

  g_t_cache_us  += t1 - t0;
  g_t_memcpy_us += t2 - t1;
  g_t_lock_us   += t3 - t2;
  g_t_dsp_us    += t4 - t3;
  g_t_unlock_us += t5 - t4;
  ++g_call_count;

  // Log first call and then every 196 calls.
  if (g_call_count == 1 || g_call_count % 196 == 0) {
    HEXKL_LOGI("TIMING call=%d M=%u N=%u K=%u: cache=%lldus memcpy=%lldus lock=%lldus dsp=%lldus unlock=%lldus",
               g_call_count, M, N, K,
               (long long)(g_t_cache_us), (long long)(g_t_memcpy_us),
               (long long)(g_t_lock_us),  (long long)(g_t_dsp_us),
               (long long)(g_t_unlock_us));
  }
  if (g_call_count % 196 == 0) {
    HEXKL_LOGI("TIMING PASS call=%d total=%lldms",
               g_call_count,
               (long long)((g_t_cache_us+g_t_memcpy_us+g_t_lock_us+g_t_dsp_us+g_t_unlock_us)/1000));
    g_t_cache_us = g_t_memcpy_us = g_t_lock_us = g_t_dsp_us = g_t_unlock_us = 0;
  }

  if (ret == 0) {
    // Copy real M rows back; ldc==N is guaranteed by the dispatch guard.
    memcpy(C, g_C_stage, (size_t)M * N * sizeof(float));
  } else {
    HEXKL_LOGE("sdkl_npu_mm_f32f16_f32 failed: %d (M=%u M_pad=%u N=%u K=%u)",
               ret, M, M_pad, N, K);
  }

  return ret == 0;
}

// ---------------------------------------------------------------------------
// shgemm_hmx: FP16 weights already available — just WH-layout + dispatch
// ---------------------------------------------------------------------------

bool shgemm_hmx(bool TransB,
                unsigned int M, unsigned int N, unsigned int K,
                const float *A, unsigned int lda,
                const __fp16 *B_f16, unsigned int ldb,
                float *C, unsigned int ldc) {
  if (!g_initialized) return false;

  const unsigned int M_pad   = ((M + 31u) / 32u) * 32u;
  const size_t       w_bytes = (size_t)N * K * sizeof(_Float16);
  const size_t       x_bytes = (size_t)M_pad * K * sizeof(float);
  const size_t       c_bytes = (size_t)M_pad * N * sizeof(float);

  int64_t t0 = now_us();

  // Look up (or build) WH-layout cache — B is already FP16, so just transpose
  // (if needed) + apply WH layout.  No FP32→FP16 conversion.
  WeightKey key{reinterpret_cast<uintptr_t>(B_f16), N, K, TransB};
  _Float16 *wh_buf = nullptr;
  {
    std::lock_guard<std::mutex> cl(g_cache_mutex);
    auto it = g_wh_cache.find(key);
    if (it != g_wh_cache.end()) wh_buf = it->second;
  }
  if (!wh_buf) {
    wh_buf = static_cast<_Float16 *>(malloc(w_bytes));
    if (!wh_buf) { HEXKL_LOGE("malloc wh_buf(%zu) failed", w_bytes); return false; }

    // B is already FP16 — only transpose if needed (no FP32→FP16 conversion).
    if (TransB) {
      for (unsigned n = 0; n < N; ++n) {
        const __fp16 *brow = B_f16 + (size_t)n * ldb;
        _Float16     *drow = wh_buf + (size_t)n * K;
        unsigned k = 0;
        for (; k + 8 <= K; k += 8)
          vst1q_f16((__fp16 *)(drow + k), vld1q_f16(brow + k));
        for (; k < K; ++k) drow[k] = (_Float16)brow[k];
      }
    } else {
      const unsigned BK = 4, BN = 32;
      for (unsigned k0 = 0; k0 < K; k0 += BK) {
        unsigned k_end = k0 + BK < K ? k0 + BK : K;
        for (unsigned n0 = 0; n0 < N; n0 += BN) {
          unsigned n_end = n0 + BN < N ? n0 + BN : N;
          for (unsigned k = k0; k < k_end; ++k) {
            const __fp16 *brow = B_f16 + (size_t)k * ldb + n0;
            unsigned n = n0;
            for (; n + 8 <= n_end; n += 8) {
              float16x8_t v = vld1q_f16(brow + (n - n0));
              __fp16 tmp[8]; vst1q_f16(tmp, v);
              for (int i = 0; i < 8; ++i)
                wh_buf[(size_t)(n+i) * K + k] = (_Float16)tmp[i];
            }
            for (; n < n_end; ++n)
              wh_buf[(size_t)n * K + k] = (_Float16)brow[n - n0];
          }
        }
      }
    }

    if (sdkl_cpu_rm_to_wh_f16_inplace(N, K, wh_buf) != 0) {
      HEXKL_LOGE("sdkl_cpu_rm_to_wh_f16_inplace failed N=%u K=%u", N, K);
      free(wh_buf); return false;
    }
    {
      std::lock_guard<std::mutex> cl(g_cache_mutex);
      auto [it, inserted] = g_wh_cache.emplace(key, wh_buf);
      if (!inserted) { free(wh_buf); wh_buf = it->second; }
    }
  }

  int64_t t1 = now_us();

  std::lock_guard<std::mutex> lock(g_hmx_mutex);

  if (w_bytes > g_W_buf_bytes) {
    if (g_W_buf) sdkl_npu_free(g_W_buf);
    g_W_buf = nullptr; g_W_buf_bytes = 0;
    if (sdkl_npu_alloc(w_bytes, reinterpret_cast<void **>(&g_W_buf)) != 0 || !g_W_buf) {
      HEXKL_LOGE("sdkl_npu_alloc W(%zu) failed", w_bytes); return false;
    }
    g_W_buf_bytes = w_bytes;
    HEXKL_LOGI("W_buf grown to %zu bytes (shgemm path)", w_bytes);
  }
  memcpy(g_W_buf, wh_buf, w_bytes);

  if (x_bytes > g_X_stage_bytes) {
    free(g_X_stage);
    g_X_stage = static_cast<float *>(malloc(x_bytes));
    if (!g_X_stage) { g_X_stage_bytes = 0; return false; }
    g_X_stage_bytes = x_bytes;
  }
  if (c_bytes > g_C_stage_bytes) {
    free(g_C_stage);
    g_C_stage = static_cast<float *>(malloc(c_bytes));
    if (!g_C_stage) { g_C_stage_bytes = 0; return false; }
    g_C_stage_bytes = c_bytes;
  }

  memcpy(g_X_stage, A, (size_t)M * K * sizeof(float));
  if (M_pad > M)
    memset(g_X_stage + (size_t)M * K, 0, (size_t)(M_pad - M) * K * sizeof(float));

  int64_t t2 = now_us();

  if (sdkl_npu_lock_hmx(g_domain) != 0) {
    HEXKL_LOGE("sdkl_npu_lock_hmx failed"); return false;
  }
  int64_t t3 = now_us();

  int ret = sdkl_npu_mm_f32f16_f32(g_domain, (int)M_pad, (int)N, (int)K,
                                    g_C_stage, g_X_stage, g_W_buf);
  int64_t t4 = now_us();
  sdkl_npu_unlock_hmx(g_domain);
  int64_t t5 = now_us();

  g_t_cache_us  += t1 - t0;
  g_t_memcpy_us += t2 - t1;
  g_t_lock_us   += t3 - t2;
  g_t_dsp_us    += t4 - t3;
  g_t_unlock_us += t5 - t4;
  ++g_call_count;

  if (g_call_count == 1 || g_call_count % 196 == 0) {
    HEXKL_LOGI("SH_TIMING call=%d M=%u N=%u K=%u: cache=%lldus memcpy=%lldus lock=%lldus dsp=%lldus unlock=%lldus",
               g_call_count, M, N, K,
               (long long)(g_t_cache_us), (long long)(g_t_memcpy_us),
               (long long)(g_t_lock_us),  (long long)(g_t_dsp_us),
               (long long)(g_t_unlock_us));
  }
  if (g_call_count % 196 == 0) {
    HEXKL_LOGI("SH_PASS call=%d total=%lldms",
               g_call_count,
               (long long)((g_t_cache_us+g_t_memcpy_us+g_t_lock_us+g_t_dsp_us+g_t_unlock_us)/1000));
    g_t_cache_us = g_t_memcpy_us = g_t_lock_us = g_t_dsp_us = g_t_unlock_us = 0;
  }

  if (ret == 0) {
    memcpy(C, g_C_stage, (size_t)M * N * sizeof(float));
  } else {
    HEXKL_LOGE("sdkl_npu_mm_f32f16_f32 failed: %d (M=%u M_pad=%u N=%u K=%u)",
               ret, M, M_pad, N, K);
  }
  return ret == 0;
}

} // namespace hexkl
} // namespace nntrainer

#endif // USE_HMX
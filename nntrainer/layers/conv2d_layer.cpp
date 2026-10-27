// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv2d_layer.h
 * @date   02 June 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Convolution Layer Class for Neural Network
 *
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include <act_simd.h> // shared NEON expf/SiLU + affine int8 quantize helpers
#include <conv2d_layer.h>
#include <conv_indirect.h> // repack_q8_0 (q8_0x4 weight interleave for the indirect conv kernel)
#include <cpu_backend.h>
#include <ggml_interface.h> // __ggml_q8_0_q8_0_indirect_GEMM_i8a (W8A8)
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <profiler.h>
#include <q8_0_tensor.h>
#include <tensor_dim.h>
#include <thread>
#include <thread_manager.h>
#include <util_func.h>

namespace nntrainer {

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace {

// ---- Stem direct-conv weight cache ----------------------------------------
// The stem (in_ch == 3) runs a dedicated direct FP32 conv (no im2col, no
// GEMM): with only 3 input channels the im2col row is 27 elements, so the
// generic col-buffer + sgemm machinery costs more than the arithmetic. The
// kernel wants the filter tap-major -- [kh][kw][in][out] so each (tap,
// channel) scalar broadcasts against a contiguous out-channel vector -- which
// is transposed once here from the [out][in][kh][kw] tensor and cached.
static const std::vector<float> &
getStemTapWeight(const float *w, unsigned int out_ch, unsigned int in_ch,
                 unsigned int kh, unsigned int kw) {
  static std::mutex mtx;
  static std::unordered_map<const void *, std::vector<float>> cache;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = cache.find((const void *)w);
  if (it != cache.end())
    return it->second;
  std::vector<float> t((size_t)kh * kw * in_ch * out_ch);
  for (unsigned int o = 0; o < out_ch; ++o)
    for (unsigned int c = 0; c < in_ch; ++c)
      for (unsigned int y = 0; y < kh; ++y)
        for (unsigned int x = 0; x < kw; ++x)
          t[(((size_t)y * kw + x) * in_ch + c) * out_ch + o] =
            w[(((size_t)o * in_ch + c) * kh + y) * kw + x];
  return cache.emplace((const void *)w, std::move(t)).first->second;
}

// ---- Depthwise int8 (channel-wise symmetric) weight cache -----------------
// W8A8 depthwise: the weight file never stores depthwise filters quantized
// (grouped convs are incompatible with the Q8_0/matmul GEMM strategy -- see
// quant_matmul_filter's groups==1 requirement below), so to get int8 compute
// here the FP32 depthwise filter is quantized once per layer (cached by
// weight pointer, same pattern as getStemTapWeight above): per-channel
// (= per depthwise group, ocg==icg==1) symmetric scale (amax/127), repacked
// tap-major/channel-last ([kh][kw][C]) so a fixed tap's C channels are
// contiguous for a vectorized int8 load (the natural [C][kh][kw] filter
// layout has channels C*fh*fw apart per tap, not usable directly).
// Guarded on __ARM_NEON: the sole consumer is the NEON int8 depthwise path
// below, so on non-NEON targets this would be an unused static function
// (-Werror=unused-function).
#ifdef __ARM_NEON
struct DWQ8Weight {
  std::vector<int8_t> qs;   /**< [fh*fw][C], tap-major, channel-last */
  std::vector<float> scale; /**< [C], per-channel symmetric weight scale */
};

[[maybe_unused]] static const DWQ8Weight &getDWQ8Weight(const float *filt,
                                                        unsigned int C,
                                                        unsigned int fh,
                                                        unsigned int fw) {
  static std::mutex mtx;
  static std::unordered_map<const void *, DWQ8Weight> cache;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = cache.find((const void *)filt);
  if (it != cache.end())
    return it->second;
  const unsigned int fhfw = fh * fw;
  DWQ8Weight w;
  w.scale.resize(C);
  w.qs.resize((size_t)fhfw * C);
  for (unsigned int c = 0; c < C; ++c) {
    const float *row = filt + (size_t)c * fhfw;
    float amax = 0.f;
    for (unsigned int t = 0; t < fhfw; ++t)
      amax = std::max(amax, std::fabs(row[t]));
    const float sc = amax > 0.f ? amax / 127.f : 1.f;
    w.scale[c] = sc;
    const float inv = 1.f / sc;
    for (unsigned int t = 0; t < fhfw; ++t) {
      const float q = std::round(row[t] * inv);
      w.qs[(size_t)t * C + c] = (int8_t)std::max(-127.f, std::min(127.f, q));
    }
  }
  return cache.emplace((const void *)filt, std::move(w)).first->second;
}
#endif // __ARM_NEON

#ifdef ENABLE_FP16
static const _FP16 *get_silu_lut_fp16() {
  static std::vector<_FP16> lut;
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    lut.resize(65536);
    for (uint32_t i = 0; i < 65536; ++i) {
      uint16_t u = static_cast<uint16_t>(i);
      _FP16 x;
      std::memcpy(&x, &u, sizeof(x));
      float x_f = static_cast<float>(x);
      if (std::isnan(x_f)) {
        lut[i] = x;
      } else if (std::isinf(x_f)) {
        if (x_f > 0.0f)
          lut[i] = x;
        else
          lut[i] = (_FP16)0.0f;
      } else {
        float silu = x_f / (1.0f + std::exp(-x_f));
        lut[i] = static_cast<_FP16>(silu);
      }
    }
  });
  return lut.data();
}
#endif

/**
 * @brief Round an FP32 value to the nearest fp16-representable value
 * (round-to-nearest-even), returned as FP32. Used by the W8A8 quantize
 * epilogue: block_q8_0 stores scales as fp16, so quantizing with the
 * fp16-rounded scale makes the consumer's (q, d) pair reproduce the
 * produced values exactly. Portable bit manipulation (no FP16 types).
 */
static inline float convRoundScaleFp16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exp = (int32_t)((bits >> 23) & 0xff) - 127 + 15;
  uint32_t man = bits & 0x7fffff;
  uint16_t h;
  if (exp <= 0)
    h = (uint16_t)sign; // underflow -> 0 (scales are tiny-positive or 0)
  else if (exp >= 31)
    h = (uint16_t)(sign | 0x7c00);
  else {
    uint32_t m = man >> 13;
    uint32_t rem = man & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (m & 1)))
      ++m;
    if (m == 0x400) {
      m = 0;
      if (++exp >= 31)
        return f; // saturate: keep original
    }
    h = (uint16_t)(sign | ((uint32_t)exp << 10) | m);
  }
  // decode back
  uint32_t hexp = (h >> 10) & 0x1f, hman = h & 0x3ff;
  uint32_t obits;
  if (hexp == 0)
    obits = (uint32_t)(h >> 15) << 31; // zero (denormal scales -> 0)
  else
    obits =
      ((uint32_t)(h >> 15) << 31) | ((hexp - 15 + 127) << 23) | (hman << 13);
  float out;
  std::memcpy(&out, &obits, 4);
  return out;
}

/**
 * @brief In-place SiLU / swish (x * sigmoid(x)) over a contiguous buffer.
 */
template <typename T>
static inline void convApplySwishInplace(T *data, size_t n) {
  auto &tm = ThreadManager::Global();
  const size_t nthreads = std::max<size_t>(1, tm.getComputeThreadCount());
  const size_t chunk = (n + nthreads - 1) / nthreads;

#ifdef ENABLE_FP16
  if constexpr (std::is_same_v<T, _FP16>) {
    const _FP16 *lut = get_silu_lut_fp16();
    tm.parallel_for(0, nthreads, [&](size_t t) {
      const size_t start = t * chunk;
      if (start >= n)
        return;
      const size_t end = std::min(start + chunk, n);
      size_t i = start;
      for (; i + 7 < end; i += 8) {
        uint16_t u0 = *reinterpret_cast<const uint16_t *>(&data[i + 0]);
        uint16_t u1 = *reinterpret_cast<const uint16_t *>(&data[i + 1]);
        uint16_t u2 = *reinterpret_cast<const uint16_t *>(&data[i + 2]);
        uint16_t u3 = *reinterpret_cast<const uint16_t *>(&data[i + 3]);
        uint16_t u4 = *reinterpret_cast<const uint16_t *>(&data[i + 4]);
        uint16_t u5 = *reinterpret_cast<const uint16_t *>(&data[i + 5]);
        uint16_t u6 = *reinterpret_cast<const uint16_t *>(&data[i + 6]);
        uint16_t u7 = *reinterpret_cast<const uint16_t *>(&data[i + 7]);
        data[i + 0] = lut[u0];
        data[i + 1] = lut[u1];
        data[i + 2] = lut[u2];
        data[i + 3] = lut[u3];
        data[i + 4] = lut[u4];
        data[i + 5] = lut[u5];
        data[i + 6] = lut[u6];
        data[i + 7] = lut[u7];
      }
      for (; i < end; ++i) {
        uint16_t u = *reinterpret_cast<const uint16_t *>(&data[i]);
        data[i] = lut[u];
      }
    });
    return;
  }
#endif

  static const bool use_approx = []() {
    if (const char *env_p = std::getenv("NNTR_APPROX_SILU")) {
      return std::string(env_p) == "1";
    }
    return false;
  }();

  tm.parallel_for(0, nthreads, [&](size_t t) {
    const size_t start = t * chunk;
    if (start >= n)
      return;
    const size_t end = std::min(start + chunk, n);
    size_t i = start;

    if (use_approx) {
#if defined(__ARM_NEON) && defined(ENABLE_FP16)
      if constexpr (std::is_same_v<T, _FP16>) {
        const float32x4_t v_three = vdupq_n_f32(3.0f);
        const float32x4_t v_six_inv = vdupq_n_f32(1.0f / 6.0f);
        const float32x4_t v_zero = vdupq_n_f32(0.0f);
        const float32x4_t v_six = vdupq_n_f32(6.0f);

        for (; i + 7 < end; i += 8) {
          float16x8_t vx =
            vld1q_f16(reinterpret_cast<const __fp16 *>(&data[i]));

          float32x4_t vx_lo = vcvt_f32_f16(vget_low_f16(vx));
          float32x4_t v_add_lo = vaddq_f32(vx_lo, v_three);
          float32x4_t v_relu6_lo =
            vminq_f32(vmaxq_f32(v_add_lo, v_zero), v_six);
          float32x4_t vres_lo =
            vmulq_f32(vmulq_f32(vx_lo, v_relu6_lo), v_six_inv);

          float32x4_t vx_hi = vcvt_f32_f16(vget_high_f16(vx));
          float32x4_t v_add_hi = vaddq_f32(vx_hi, v_three);
          float32x4_t v_relu6_hi =
            vminq_f32(vmaxq_f32(v_add_hi, v_zero), v_six);
          float32x4_t vres_hi =
            vmulq_f32(vmulq_f32(vx_hi, v_relu6_hi), v_six_inv);

          vst1q_f16(reinterpret_cast<__fp16 *>(&data[i]),
                    vcombine_f16(vcvt_f16_f32(vres_lo), vcvt_f16_f32(vres_hi)));
        }
      }
#endif
      for (; i < end; ++i) {
        const float x = static_cast<float>(data[i]);
        float relu6 = std::max(0.0f, std::min(x + 3.0f, 6.0f));
        data[i] = static_cast<T>(x * relu6 / 6.0f);
      }
    } else {
      // EXACT SiLU x/(1+exp(-x)). FP32 gets a NEON-vectorized path (4-wide
      // expf + reciprocal, ~1e-6 error) so the epilogue is no longer a scalar
      // std::exp per element -- the FP16 build already uses a LUT; only the
      // FP32 (W8A32 / W32A32) path was scalar.
#if defined(__ARM_NEON)
      if constexpr (std::is_same_v<T, float>) {
        const float32x4_t one = vdupq_n_f32(1.0f);
        for (; i + 3 < end; i += 4) {
          float32x4_t vx = vld1q_f32(&data[i]);
          float32x4_t e = nntr_vexpq_f32(vnegq_f32(vx));
          float32x4_t denom = vaddq_f32(one, e);
          float32x4_t r = vrecpeq_f32(denom);
          r = vmulq_f32(vrecpsq_f32(denom, r), r);
          r = vmulq_f32(vrecpsq_f32(denom, r), r);
          vst1q_f32(&data[i], vmulq_f32(vx, r));
        }
      }
#endif
      for (; i < end; ++i) {
        const float x = static_cast<float>(data[i]);
        data[i] = static_cast<T>(x / (1.0f + std::exp(-x)));
      }
    }
  });
}

// Fused bias + activation over one NHWC output row [C] (channels contiguous,
// bias[c] per channel), in place; returns max(|result|) so the caller can build
// a per-tensor quantization scale in the same streaming pass -- no separate
// bias / SiLU / amax passes over the output (the per-channel W8A8 epilogue was
// three single-threaded scalar passes; this collapses them into one, and the
// caller runs it row-parallel). act: 0 none, 1 exact SiLU x*sigmoid(x), 2 the
// hard/approx SiLU x*relu6(x+3)/6. The NEON path is bit-identical to
// convApplySwishInplace's vectorized SiLU (same vexpq + two Newton reciprocal
// refinements), so this reorders work without changing numerics.
static inline float convBiasActRow(float *row, const float *bias,
                                   unsigned int C, int act) {
  float am = 0.f;
  unsigned int cc = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
  float32x4_t vam = vdupq_n_f32(0.f);
  const float32x4_t one = vdupq_n_f32(1.f);
  const float32x4_t three = vdupq_n_f32(3.f);
  const float32x4_t six = vdupq_n_f32(6.f);
  const float32x4_t zero = vdupq_n_f32(0.f);
  const float32x4_t sixth = vdupq_n_f32(1.f / 6.f);
  for (; cc + 3 < C; cc += 4) {
    float32x4_t v = vld1q_f32(row + cc);
    if (bias)
      v = vaddq_f32(v, vld1q_f32(bias + cc));
    if (act == 1) {
      float32x4_t e = nntr_vexpq_f32(vnegq_f32(v));
      float32x4_t denom = vaddq_f32(one, e);
      float32x4_t r = vrecpeq_f32(denom);
      r = vmulq_f32(vrecpsq_f32(denom, r), r);
      r = vmulq_f32(vrecpsq_f32(denom, r), r);
      v = vmulq_f32(v, r);
    } else if (act == 2) {
      float32x4_t relu6 = vminq_f32(vmaxq_f32(vaddq_f32(v, three), zero), six);
      v = vmulq_f32(vmulq_f32(v, relu6), sixth);
    }
    vst1q_f32(row + cc, v);
    vam = vmaxq_f32(vam, vabsq_f32(v));
  }
  am = vmaxvq_f32(vam);
#endif
  for (; cc < C; ++cc) {
    float v = row[cc];
    if (bias)
      v += bias[cc];
    if (act == 1)
      v = v / (1.f + std::exp(-v));
    else if (act == 2) {
      float r6 = std::max(0.f, std::min(v + 3.f, 6.f));
      v = v * r6 / 6.f;
    }
    row[cc] = v;
    am = std::max(am, std::fabs(v));
  }
  return am;
}

static TensorDim calcCol2ImOutputDim(const TensorDim &out,
                                     const TensorDim &kdim) {

  return TensorDim({kdim.getFeatureLen(), out.width() * out.height()},
                   out.getTensorType());
}

#ifdef ENABLE_FP16
/**
 * @brief Quantize FP16 NHWC [n_spatial, in_ch] -> plain block_q8_0
 * [n_spatial][in_ch/32].
 *
 * NHWC input is already row-major (channel innermost): src[r * in_ch + c].
 * No transpose needed — each row r has in_ch contiguous FP16 channels.
 * Q8_0 requires in_ch % 32 == 0 (caller must check).
 * dst must hold n_spatial * (in_ch/32) * sizeof(block_q8_0) bytes.
 */
static inline void quantize_nhwc_q8_0_rows(const _FP16 *src, int n_spatial,
                                           int in_ch,
                                           ::nntrainer::block_q8_0 *dst) {
  const int nb = in_ch / 32;
  auto &tm = ThreadManager::Global();
  const unsigned int chunk = 512;
  const size_t loops = ((size_t)n_spatial + chunk - 1) / chunk;
  tm.parallel_for(0, loops, [=](size_t idx) {
    unsigned r0 = (unsigned)idx * chunk;
    unsigned r1 = std::min(r0 + chunk, (unsigned)n_spatial);
    for (unsigned r = r0; r < r1; ++r) {
      const _FP16 *row = src + (size_t)r * in_ch;
      for (int b = 0; b < nb; ++b) {
        const _FP16 *blk = row + b * 32;
        float amax = 0.f;
        for (int j = 0; j < 32; ++j) {
          float v = std::abs(static_cast<float>(blk[j]));
          if (v > amax)
            amax = v;
        }
        const float d = amax / 127.f;
        const float id = d ? 1.f / d : 0.f;
        _FP16 d_h = static_cast<_FP16>(d);
        uint16_t d_u16;
        std::memcpy(&d_u16, &d_h, 2);
        ::nntrainer::block_q8_0 &out_blk = dst[(size_t)r * nb + b];
        out_blk.d = d_u16;
        for (int j = 0; j < 32; ++j)
          out_blk.qs[j] = (int8_t)std::roundf(static_cast<float>(blk[j]) * id);
      }
    }
  });
}

/**
 * @brief Quantize FP16 NHWC [owoh, in_ch] directly into the block_q8_0x4
 *        (4-row interleaved) layout the SMMLA GEMM consumes — single pass.
 *
 * NHWC source is row-major (channel innermost): element (r, c) at
 * src[r*in_ch+c]. This is the NHWC-read counterpart of
 * transpose_quantize_q8_0x4_act (which reads NCHW channel-major). It fuses the
 * two passes the prior 1x1 W4A8 path performed (quantize_nhwc_q8_0_rows ->
 * plain block_q8_0, then Q8_0_Tensor::dot repacks to x4) into one, and lets the
 * caller invoke Q8_0_Tensor::dot_prepacked_x4 (no per-conv repack, no per-conv
 * QA malloc). Output bytes are identical to what the two-pass path produced.
 * dst layout: M4=owoh/4 groups of block_q8_0x4 (136 B/blk) followed by (owoh %
 * 4) remainder rows as plain block_q8_0 (34 B/blk) — exactly what
 * dot_prepacked_x4 expects. dst must hold the same total as the block_q8_0
 * buffer (136 B per 4 rows == 4 * 34 B). Q8_0 requires in_ch % 32 == 0.
 */
static inline void quantize_nhwc_q8_0x4_rows(const _FP16 *src, int in_ch,
                                             int owoh, void *dst) {
  struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
  };
  struct block_q8_0x4 {
    uint16_t d[4];
    int8_t qs[128];
  };
  const int qk = 32;
  const int nb = in_ch / qk;
  const int M4 = owoh / 4;
  const int rem = owoh % 4;
  block_q8_0x4 *y4 = static_cast<block_q8_0x4 *>(dst);
  const size_t qa_4_rows_size = sizeof(block_q8_0x4) * nb;

  auto &tm = ThreadManager::Global();
  const unsigned int chunk = 256; // groups of 4 rows per task
  const size_t loops = (M4 + chunk - 1) / chunk;
  tm.parallel_for(0, loops, [=](size_t idx) {
    unsigned int g0 = idx * chunk;
    unsigned int g1 = std::min(g0 + chunk, (unsigned int)M4);
    for (unsigned int g = g0; g < g1; ++g) {
      unsigned int r0 = g * 4;
      for (int b = 0; b < nb; ++b) {
        block_q8_0x4 &dst_b = y4[g * nb + b];
        for (unsigned int row = 0; row < 4; ++row) {
          const _FP16 *blk = src + (size_t)(r0 + row) * in_ch + b * qk;
          float amax = 0.0f;
          for (int j = 0; j < qk; ++j) {
            float val = std::abs(static_cast<float>(blk[j]));
            if (val > amax)
              amax = val;
          }
          const float d = amax / ((1 << 7) - 1);
          const float id = d ? 1.0f / d : 0.0f;
          _FP16 d_half = static_cast<_FP16>(d);
          uint16_t d_u16;
          std::memcpy(&d_u16, &d_half, 2);
          dst_b.d[row] = d_u16;
          for (int j = 0; j < qk; ++j) {
            // qs[32*(j/8) + 8*row + (j%8)] — matches the SMMLA x4 layout.
            dst_b.qs[32 * (j / 8) + 8 * row + (j % 8)] =
              static_cast<int8_t>(std::roundf(static_cast<float>(blk[j]) * id));
          }
        }
      }
    }
  });

  // Remainder rows (owoh % 4): plain block_q8_0 for the GEMV tail.
  if (rem > 0) {
    block_q8_0 *yrem = reinterpret_cast<block_q8_0 *>(
      reinterpret_cast<char *>(dst) + (size_t)M4 * qa_4_rows_size);
    for (int i = 0; i < rem; ++i) {
      unsigned int r = M4 * 4 + i;
      for (int b = 0; b < nb; ++b) {
        const _FP16 *blk = src + (size_t)r * in_ch + b * qk;
        float amax = 0.0f;
        for (int j = 0; j < qk; ++j) {
          float val = std::abs(static_cast<float>(blk[j]));
          if (val > amax)
            amax = val;
        }
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        _FP16 d_half = static_cast<_FP16>(d);
        uint16_t d_u16;
        std::memcpy(&d_u16, &d_half, 2);
        yrem[i * nb + b].d = d_u16;
        for (int j = 0; j < qk; ++j)
          yrem[i * nb + b].qs[j] =
            static_cast<int8_t>(std::roundf(static_cast<float>(blk[j]) * id));
      }
    }
  }
}

/**
 * @brief Transpose-and-quantize FP16 NCHW [in_ch, owoh] -> Q8_0 [owoh, in_ch]
 *        in a single fused pass (no intermediate transpose copy).
 *
 * Each output row r (a spatial position) is quantized per 32-channel block:
 * block (r, b) covers channels [b*32, b*32+32). The FP16 source is NCHW
 * (channel-major), so channel c at position r lives at src[c*owoh + r]. This
 * gathers a 32-wide channel run with a strided read and writes a packed
 * block_q8_0 (fp16 scale + 32 int8). Parallelized over spatial positions.
 *
 * dst must hold (owoh * in_ch/32) block_q8_0 = owoh*in_ch/32*34 bytes, laid out
 * row-major as [owoh][in_ch/32] blocks — exactly the [M,K] block_q8_0 layout
 * Q8_0_Tensor::dot / the indirect GEMM consumes (M=owoh, K=in_ch).
 */
static inline void transpose_quantize_q8_0_act(const _FP16 *src, int in_ch,
                                               int owoh, void *dst) {
  struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
  };
  block_q8_0 *y = static_cast<block_q8_0 *>(dst);
  const int qk = 32;
  const int nb = in_ch / qk;

  auto &tm = ThreadManager::Global();
  const unsigned int chunk = 1024;
  const size_t loops = (owoh + chunk - 1) / chunk;
  tm.parallel_for(0, loops, [=](size_t idx) {
    unsigned int r0 = idx * chunk;
    unsigned int r1 = std::min(r0 + chunk, (unsigned int)owoh);
    for (unsigned int r = r0; r < r1; ++r) {
      for (int b = 0; b < nb; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < qk; ++j) {
          int c = b * qk + j;
          float val = std::abs(static_cast<float>(src[c * owoh + r]));
          if (val > amax)
            amax = val;
        }
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        _FP16 d_half = static_cast<_FP16>(d);
        uint16_t d_u16;
        std::memcpy(&d_u16, &d_half, 2);
        y[r * nb + b].d = d_u16;
        for (int j = 0; j < qk; ++j) {
          int c = b * qk + j;
          float x0 = static_cast<float>(src[c * owoh + r]) * id;
          y[r * nb + b].qs[j] = std::roundf(x0);
        }
      }
    }
  });
}

/**
 * @brief Fused transpose + quantize FP16 NCHW [in_ch, owoh] directly into the
 *        block_q8_0x4 (4-row interleaved) layout the SMMLA GEMM consumes, with
 *        NO intermediate plain-block pass and NO separate interleave copy.
 *
 * Outputs (M4 = owoh/4) groups of 4 rows; each group packs nb=in_ch/32
 * block_q8_0x4. block_q8_0x4.qs[128] layout = qs[32*j + 8*row + lane],
 * j=8-element chunk (0..3), row=0..3 (matches __ggml_quantize_mat_q8_0_4x8).
 * Remainder (owoh % 4) rows packed as plain block_q8_0 afterward for the GEMV
 * tail. dst must hold the block_q8_0x4 region followed by the remainder
 * block_q8_0 region (same total as Q8_0_Tensor::dot's QA buffer).
 */
static inline void transpose_quantize_q8_0x4_act(const _FP16 *src, int in_ch,
                                                 int owoh, void *dst) {
  struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
  };
  struct block_q8_0x4 {
    uint16_t d[4];
    int8_t qs[128];
  };
  const int qk = 32;
  const int nb = in_ch / qk;
  const int M4 = owoh / 4;
  const int rem = owoh % 4;
  block_q8_0x4 *y4 = static_cast<block_q8_0x4 *>(dst);
  const size_t qa_4_rows_size = sizeof(block_q8_0x4) * nb;

  auto &tm = ThreadManager::Global();
  const unsigned int chunk = 256; // groups of 4 rows per task
  const size_t loops = (M4 + chunk - 1) / chunk;
  tm.parallel_for(0, loops, [=](size_t idx) {
    unsigned int g0 = idx * chunk;
    unsigned int g1 = std::min(g0 + chunk, (unsigned int)M4);
    for (unsigned int g = g0; g < g1; ++g) {
      unsigned int r0 = g * 4;
      for (int b = 0; b < nb; ++b) {
        block_q8_0x4 &dst_b = y4[g * nb + b];
        for (unsigned int row = 0; row < 4; ++row) {
          unsigned int r = r0 + row;
          float amax = 0.0f;
          for (int j = 0; j < qk; ++j) {
            int c = b * qk + j;
            float val = std::abs(static_cast<float>(src[c * owoh + r]));
            if (val > amax)
              amax = val;
          }
          const float d = amax / ((1 << 7) - 1);
          const float id = d ? 1.0f / d : 0.0f;
          _FP16 d_half = static_cast<_FP16>(d);
          uint16_t d_u16;
          std::memcpy(&d_u16, &d_half, 2);
          dst_b.d[row] = d_u16;
          for (int j = 0; j < qk; ++j) {
            int c = b * qk + j;
            float x0 = static_cast<float>(src[c * owoh + r]) * id;
            // qs[32*chunk + 8*row + lane], chunk = j/8, lane = j%8
            dst_b.qs[32 * (j / 8) + 8 * row + (j % 8)] =
              static_cast<int8_t>(std::roundf(x0));
          }
        }
      }
    }
  });

  // Remainder rows (owoh % 4): plain block_q8_0 for the GEMV tail.
  if (rem > 0) {
    block_q8_0 *yrem = reinterpret_cast<block_q8_0 *>(
      reinterpret_cast<char *>(dst) + (size_t)M4 * qa_4_rows_size);
    const unsigned int rchunk = 1024;
    const size_t rloops = (rem + rchunk - 1) / rchunk;
    tm.parallel_for(0, rloops, [=](size_t idx) {
      unsigned int i0 = idx * rchunk;
      unsigned int i1 = std::min(i0 + rchunk, (unsigned int)rem);
      for (unsigned int i = i0; i < i1; ++i) {
        unsigned int r = M4 * 4 + i;
        for (int b = 0; b < nb; ++b) {
          float amax = 0.0f;
          for (int j = 0; j < qk; ++j) {
            int c = b * qk + j;
            float val = std::abs(static_cast<float>(src[c * owoh + r]));
            if (val > amax)
              amax = val;
          }
          const float d = amax / ((1 << 7) - 1);
          const float id = d ? 1.0f / d : 0.0f;
          _FP16 d_half = static_cast<_FP16>(d);
          uint16_t d_u16;
          std::memcpy(&d_u16, &d_half, 2);
          yrem[i * nb + b].d = d_u16;
          for (int j = 0; j < qk; ++j) {
            int c = b * qk + j;
            float x0 = static_cast<float>(src[c * owoh + r]) * id;
            yrem[i * nb + b].qs[j] = static_cast<int8_t>(std::roundf(x0));
          }
        }
      }
    });
  }
}
#endif

/**
 * @brief     reconstruct image data from 2d column matrix
 *
 * @param[in] kdim kernel dimesion for define number of row
 * @param[in] padding padding information
 * @param[in] mstride stride value : x, y direction
 * @param[in] dilation kernel dilation factor : x, y each
 * @param[out] image image tensor to put
 */
static void col2im(const Tensor &col_matrix, const TensorDim &kdim,
                   const std::array<unsigned, 4> &padding,
                   const std::array<props::Stride, CONV2D_DIM> &mstride,
                   const std::array<props::Dilation, CONV2D_DIM> &dilation,
                   Tensor &image) {

  auto pt = padding[0];
  auto pb = padding[1];
  auto pl = padding[2];
  auto pr = padding[3];

  unsigned k_height = kdim.height();
  unsigned k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned im_channel = image.channel();
  int im_height = image.height();
  int im_width = image.width();

  unsigned hstride = mstride[0];
  unsigned wstride = mstride[1];

  unsigned hdilation = dilation[0];
  unsigned wdilation = dilation[1];

  /// image considering padding
  unsigned im_eff_height = im_height + pt + pb;
  unsigned im_eff_width = im_width + pl + pr;
  image.setZero();

  int h_stride_end = im_eff_height - eff_k_height - pt;
  int w_stride_end = im_eff_width - eff_k_width - pl;

  /** @todo We need to implement way to use this kind of function to work inside
   * of Tensor. Then we could remove to access the getData or getValue which has
   * dependecy of data type.
   */
  auto apply_data = [&](auto *val) {
    using T = std::decay_t<decltype(*val)>;
    unsigned col_w = 0;
    for (int hs = -(int)pt; hs <= h_stride_end; hs += hstride) {
      for (int ws = -(int)pl; ws <= w_stride_end; ws += wstride) {
        unsigned col_h = 0;
        int patch_height_end = hs + eff_k_height;
        int patch_width_end = ws + eff_k_width;
        for (unsigned c = 0; c < im_channel; c++) {
          for (int h = hs; h < patch_height_end; h += hdilation) {
            if (h < 0 || im_height <= h) {
              col_h += k_width;
              continue;
            }
            for (int w = ws; w < patch_width_end; w += wdilation) {
              if (w < 0 || im_width <= w) {
                col_h++;
                continue;
              }

              val = image.getAddress<T>(0, c, h, w);
              *val += col_matrix.getValue<T>(0, 0, col_h, col_w);
              col_h++;
            }
          }
        }
        col_w++;
      }
    }
  };

  if (image.getDataType() == nntrainer::Tdatatype::FP32) {
    float val;
    apply_data(&val);
  }
#ifdef ENABLE_FP16
  else if (image.getDataType() == nntrainer::Tdatatype::FP16) {
    _FP16 val;
    apply_data(&val);
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }
}

/**
 * @brief     reform the data to 2d matrix
 * a region is sampled considering @a padding, @a mstride of unit @a kdim
 * Each region is mapped to one column,
 * if channel mode, kernel channel is considered part of kernel feature
 * if not, kernel channel is consider part of output dimension
 *
 * @param[in] in input data
 * @param[in] kdim kernel dimesion for define number of row
 * @param[in] padding padding information
 * @param[in] mstride stride value : x, y direction
 * @param[in] dilation kernel dilation factor : x, y each
 * @param[out] out out tensor, padding set each time for now
 * @note if out is initialized tensor, setting padding is skipped.
 */
static void im2col(const Tensor &in, const TensorDim &kdim,
                   const std::array<unsigned int, 4> &padding,
                   const std::array<props::Stride, CONV2D_DIM> &mstride,
                   const std::array<props::Dilation, CONV2D_DIM> &dilation,
                   Tensor &out) {
  /// for channel last mode, this is deprecated for now, leaving here on
  /// purpose.
  /** @code
  //   ================ initialize part ====================
  //   out_height -= 2;
  //   out =
  //     Tensor(k_height * k_width, in.channel() * (out_height) *
  //     (out_width));
  //   unsigned int im_w = 0;
  //   ================ loop part ====================
  //   if (eff_k_height > height || eff_k_width > width)
  //     throw std::runtime_error("Kernel shape bigger than input shape");

  //   for (unsigned int c = 0; c < channel; ++c) {
  //     for (unsigned int hs = 0; hs <= height - eff_k_height; hs +=
  //     mstride[0]) {
  //       for (unsigned int ws = 0; ws <= width - eff_k_width; ws +=
  //       mstride[1]) {
  //         unsigned int im_h = 0;
  //         unsigned int patch_height_end = eff_k_height + hs;
  //         unsigned int patch_width_end = eff_k_width + ws;

  //         for (unsigned int h = hs; h < patch_height_end; h += dilation[0]) {
  //           if (h < ph || in_height + ph <= h) {
  //             im_h += k_width;
  //             continue;
  //           }

  //           for (unsigned int w = ws; w < patch_width_end; w += dilation[1])
  //           {
  //             if (w < pw || in_width + pw <= w) {
  //               im_h++;
  //               continue;
  //             }

  //             float val = in.getValue(0, c, h - ph, w - pw);
  //             out.setValue(0, 0, im_h, im_w, val);
  //             im_h++;
  //           }
  //         }
  //         im_w++;
  //       }
  //     }
  //   }
  */

  auto pt = padding[0];
  auto pb = padding[1];
  auto pl = padding[2];
  auto pr = padding[3];

  unsigned int channel = in.channel();
  int in_height = in.height();
  int in_width = in.width();
  unsigned int height = in_height + pt + pb;
  unsigned int width = in_width + pl + pr;
  unsigned int k_height = kdim.height();
  unsigned int k_width = kdim.width();

  /// effective kernel height considering dilation
  unsigned int eff_k_height = (k_height - 1) * dilation[0] + 1;
  /// effective kernel width considering dilation
  unsigned int eff_k_width = (k_width - 1) * dilation[1] + 1;

  unsigned int out_height = (height - eff_k_height) / mstride[0] + 1;
  unsigned int out_width = (width - eff_k_width) / mstride[1] + 1;

  out.reshape(
    TensorDim({out_height * out_width, in.channel() * k_height * k_width},
              in.getTensorType()));
  // float *out_data = out.getData();

  auto apply_data = [&]<typename T>(T *out_data) {
    int h_stride_end = height - eff_k_height - pt;
    int w_stride_end = width - eff_k_width - pl;

    /// get a patch, size of kernel
    /// hs is height_strided, ws is width_strided
    unsigned int owidth = out.width();
    const int hstride = mstride[0];

    /// Raw contiguous-NCHW input base + inner strides. `in` is a (batch-sliced)
    /// contiguous NCHW tensor, so element (0,c,h,w) lives at
    /// in_base + c*inHW + h*inW + w. Hoisting these out of the inner loop turns
    /// the old per-element in.getValue() -- which re-fetched the data pointer
    /// (getData<T>()) and recomputed a 4-D linear offset (getIndex(): a format
    /// branch + 4 muls + 3 adds) plus a per-element padding branch -- into one
    /// contiguous run copy. im2col is pure data movement; the previous form ran
    /// far below memory bandwidth because that overhead dominated the 4-byte
    /// move. Padding columns are left untouched (the caller zeroes the buffer
    /// once before im2col), so the fast path only writes the valid span.
    const T *in_base = in.getData<T>();
    const size_t inW = (size_t)in_width;
    const size_t inHW = (size_t)in_height * (size_t)in_width;
    const bool unit_dil =
      ((unsigned int)dilation[0] == 1 && (unsigned int)dilation[1] == 1);
    const bool is_nhwc = (in.getFormat() == ml::train::TensorDim::Format::NHWC);

    /// Each output row (oh) writes a disjoint band of `out_width` columns
    /// (rows [oh*out_width, (oh+1)*out_width) of the [OH*OW, CRS] matrix), so
    /// the per-row work is independent and bit-identical when parallelized.
    /// hs and base_im_w are derived from oh directly (no sequential carry).
    auto fill_row = [&](size_t oh) {
      int hs = -(int)pt + (int)oh * hstride;
      unsigned int base_im_w = (unsigned int)oh * out_width;
      unsigned int base_im_h = 0;
      int patch_height_end = eff_k_height + hs;
      /// map the patch to a single line looping through channel
      for (unsigned int c = 0; c < channel; ++c) {
        for (int h = hs; h < patch_height_end; h += dilation[0]) {
          if (h < 0 || in_height <= h) {
            base_im_h += k_width;
            continue;
          }

          if (unit_dil) {
            /// Fast path (dilation == 1): for each output column position the
            /// kernel-width window maps a contiguous source run
            /// in_row[w_lo,w_hi) to a contiguous dest run; copy it in one
            /// memcpy.
            const T *in_row = in_base + (size_t)c * inHW + (size_t)h * inW;
            unsigned int im_w = base_im_w;
            for (int ws = -(int)pl; ws <= w_stride_end; ws += mstride[1]) {
              int w_lo = ws < 0 ? 0 : ws;
              int w_hi = ws + (int)k_width;
              if (w_hi > in_width)
                w_hi = in_width;
              if (w_hi > w_lo) {
                T *dst =
                  out_data + (size_t)im_w * owidth + base_im_h + (w_lo - ws);
                if (!is_nhwc) {
                  const T *in_row =
                    in_base + (size_t)c * inHW + (size_t)h * inW;
                  std::memcpy(dst, in_row + w_lo,
                              (size_t)(w_hi - w_lo) * sizeof(T));
                } else {
                  /// NHWC: channel is innermost, so the w-run is NOT
                  /// contiguous in source. Gather per (w, channel) element.
                  for (int w = w_lo; w < w_hi; ++w) {
                    dst[w - w_lo] =
                      in_base[((size_t)h * inW + (size_t)w) * channel + c];
                  }
                }
              }
              im_w++;
            }
          } else {
            /// General (dilated) path: original scalar gather, but via the
            /// hoisted base pointer (no per-element getData()/getIndex()).
            unsigned int im_w = base_im_w;
            for (int ws = -(int)pl; ws <= w_stride_end; ws += mstride[1]) {
              unsigned int im_h = base_im_h;
              int patch_width_end = eff_k_width + ws;

              for (int w = ws; w < patch_width_end; w += dilation[1]) {
                if (w < 0 || in_width <= w) {
                  im_h++;
                  continue;
                }
                if (!is_nhwc) {
                  out_data[(size_t)im_w * owidth + im_h] =
                    in_base[(size_t)c * inHW + (size_t)h * inW + w];
                } else {
                  out_data[(size_t)im_w * owidth + im_h] =
                    in_base[((size_t)h * inW + (size_t)w) * channel + c];
                }
                im_h++;
              }
              im_w++;
            }
          }
          base_im_h += k_width;
        }
      }
    };

    ThreadManager::Global().parallel_for(0, out_height, fill_row);
  };

  if (out.getDataType() == nntrainer::Tdatatype::FP32) {
    float *out_data = out.getData<float>();
    apply_data(out_data);
  }
#ifdef ENABLE_FP16
  else if (out.getDataType() == nntrainer::Tdatatype::FP16) {
    _FP16 *out_data = out.getData<_FP16>();
    apply_data(out_data);
  }
#endif
  else {
    throw std::runtime_error("Not supported datatype");
  }
}

} // namespace

enum ConvParams { weight, bias, im2col_scratch, qgemm_scratch, q8act_scratch };

Conv2DLayer::Conv2DLayer(
  const std::array<unsigned int, CONV2D_DIM * 2> &padding_) :
  LayerImpl(),
  padding(padding_),
  conv_props(props::FilterSize(), std::array<props::KernelSize, CONV2D_DIM>(),
             std::array<props::Stride, CONV2D_DIM>(), props::Padding2D(),
             std::array<props::Dilation, CONV2D_DIM>(), props::ConvGroups(),
             props::FusedActivation()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void Conv2DLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Convolution layer takes only one input";

  const TensorDim &in_dim = context.getInputDimensions()[0];

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &kernel_size =
    std::get<std::array<props::KernelSize, CONV2D_DIM>>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  auto &groups_prop = std::get<props::ConvGroups>(conv_props);
  unsigned int groups = groups_prop.empty() ? 1 : groups_prop.get();
  NNTR_THROW_IF(in_dim.channel() % groups != 0 || filter_size % groups != 0,
                std::invalid_argument)
    << "[Conv2D] input channels (" << in_dim.channel() << ") and filters ("
    << filter_size << ") must both be divisible by groups (" << groups << ")";

  auto in_t_type = in_dim.getTensorType();
  in_t_type.data_type = context.getWeightDataType();

  // A quantized (Q4_0/QINT4) 1x1 conv is computed as a matmul, so its filter is
  // stored as a [in_ch, out_ch] (K, N) weight that the quantized GEMM consumes
  // directly (no im2col-style [out_ch, CRS] squeeze). Non-quantized or larger
  // kernels keep the standard [out_ch, in_ch/groups, kh, kw] layout.
  const bool quant_matmul_filter =
    (in_t_type.data_type == nntrainer::Tdatatype::Q4_0 ||
     in_t_type.data_type == nntrainer::Tdatatype::QINT4 ||
     in_t_type.data_type == nntrainer::Tdatatype::Q8_0) &&
    groups == 1;

  // Real conv kernel geometry — used for padding/output-size computation even
  // when the quantized weight is stored flattened as [CRS, out_ch].
  TensorDim real_kernel_dim(filter_size, in_dim.channel() / groups,
                            kernel_size[0], kernel_size[1], in_t_type);
  // A quantized (groups==1) conv stores its filter as a [CRS, out_ch] matmul
  // weight (CRS = in_ch*kh*kw), consumed by the quantized GEMM after im2col.
  TensorDim kernel_dim = quant_matmul_filter
                           ? TensorDim(1, 1,
                                       in_dim.channel() * kernel_size[0].get() *
                                         kernel_size[1].get(),
                                       filter_size, in_t_type)
                           : real_kernel_dim;

  // Bias is never quantized (no dequantizer for add); follow activation dtype
  // like other compute layers so a Q4_0/QINT4 weight does not force a Q4_0
  // bias.
  auto bias_t_type = in_dim.getTensorType();
  bias_t_type.data_type = context.getActivationDataType();
  TensorDim bias_dim = TensorDim(1, filter_size, 1, 1, bias_t_type);

  padding = std::get<props::Padding2D>(conv_props)
              .compute(in_dim, real_kernel_dim, {stride[0], stride[1]},
                       {dilation[0], dilation[1]});

  wt_idx[ConvParams::weight] = context.requestWeight(
    kernel_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "filter", true, 0);

  if (disable_bias.empty() || disable_bias.get() == false) {
    wt_idx[ConvParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true, 0);
  }

  // this output_dim must be the same with dimension of hidden
  unsigned int eff_in_height = in_dim.height() + padding[0] + padding[1];
  unsigned int eff_in_width = in_dim.width() + padding[2] + padding[3];

  unsigned int eff_k_height = (kernel_size[0] - 1) * dilation[0] + 1;
  unsigned int eff_k_width = (kernel_size[1] - 1) * dilation[1] + 1;

  TensorDim out_dim;
  out_dim.batch(in_dim.batch());
  out_dim.channel(filter_size);
  out_dim.height((eff_in_height - eff_k_height) / stride[0] + 1);
  out_dim.width((eff_in_width - eff_k_width) / stride[1] + 1);

  // NOTE the output dtype follows the INPUT, not the model's activation dtype.
  // Making it follow the activation dtype (so a quantized FP16-activation model
  // could use this conv as its FP32-image -> FP16-graph boundary) is not enough
  // on its own: the quantized NCHW branch materializes its im2col/GEMM scratch
  // in the input dtype and writes the result out through transpose(), which
  // cannot convert FP32 -> FP16. Both would have to change together.
  out_dim.setTensorType(in_dim.getTensorType());

  // W8A8 (env NNTR_W8A8, see W8A8_DESIGN.md): a Q8_0-weight NHWC conv emits a
  // per-tensor-scale QINT8 activation instead of FP32. Every consumer handles
  // a QINT8 input (Q8_0 convs via the i8a GEMM, FP32 convs by dequantizing
  // into a scratch, concat/max-pool/nearest-upsample with int8 branches), so
  // the emission is unconditional for quantized convs -- no per-edge graph
  // analysis. Env-gated: all other modes are untouched.
  const bool w8a8_mode = std::getenv("NNTR_W8A8") != nullptr;
  // Per-channel W8A8 (NNTR_W8A8_PERCH): EVERY NHWC conv runs int8 through the
  // int32-accumulate kernel and emits a QINT8 activation, except the head-final
  // conv (out_ch % 4 != 0, uniquely out_ch=87) whose consumer rtmcc_head is a
  // hard FP32 boundary -> it stays FP32 out. Independent of the per-block
  // eligibility (quant_matmul_filter); the weight is converted in-memory.
  // The stem (in_ch == 3) is excluded: its input is the network's FP32 image,
  // and quantizing the raw image to int8 injects error at the very first layer
  // that the whole net then amplifies. The validated S0 simulation keeps the
  // stem input FP32 ("the image tensor stays FP32") and holds 81/87; the stem
  // runs the ordinary FP32 conv (already exercised as the per-block mode's
  // FP32 island) and its per-channel consumer quantizes its FP32 output.
  // NOTE (2026-08-11): an earlier version of this comment claimed the
  // non-identity gather path was broken based on a bisection against
  // `FACE_DET_TAP_STAGES`'s backbone-tap dump -- that dump mechanism turned
  // out to be corrupted by memory-pool aliasing (cosine 0.005 against an
  // NNTR_DUMP_LAYER dump of the *same* layer under the *same* W8A32 run).
  // Re-verified against the reliable NNTR_DUMP_LAYER baseline: conv1 (3x3,
  // taps-last, K padded 144->160) matches at cosine 0.987. perch_mode is
  // restored to applying unconditionally (matching the upstream PR), not
  // restricted to 1x1/identity convs.
  const bool perch_mode =
    w8a8_mode && std::getenv("NNTR_W8A8_PERCH") != nullptr && groups == 1 &&
    in_dim.getFormat() == ml::train::TensorDim::Format::NHWC &&
    in_dim.channel() != 3;
  // The asymmetric int8 encoding x=(q+128)*s-kActOff is tuned specifically
  // for SiLU's output range (bounded below by -0.27846, per the epilogue
  // below) -- applying it to a conv with NO activation (raw linear output,
  // e.g. the DFL box-regression head cv2_2, or any other bias-only conv)
  // silently clips every value below -kActOff to that floor, since the
  // encoding has no representation for anything more negative. Confirmed
  // via on-device bisection 2026-08-11: this is what corrupted the
  // detector's box branch even after every other W8A8 bug was fixed (cls,
  // which IS SiLU-activated through cv3_0_pw/cv3_1_pw, recovered fine; the
  // final cv3_2/cv2_2 1x1 convs have no activation and were still wrong).
  // Only quantize a conv's output to int8 when it actually applies SiLU;
  // a non-activated conv's raw output stays FP32 (dequantized on entry by
  // any int8 consumer, same as every other FP32-output conv).
  const bool perch_has_silu = [&] {
    if (auto &actp = std::get<props::FusedActivation>(conv_props);
        !actp.empty() && actp.get() == ActivationType::ACT_SWISH)
      return true;
    return false;
  }();
  const bool perch_q8out =
    perch_mode && (filter_size % 4 == 0) && perch_has_silu;
  const bool w8a8_q8out =
    !perch_mode && w8a8_mode && quant_matmul_filter &&
    in_t_type.data_type == nntrainer::Tdatatype::Q8_0 &&
    in_dim.getFormat() == ml::train::TensorDim::Format::NHWC;
  if (w8a8_q8out || perch_q8out) {
    out_dim.setDataType(nntrainer::Tdatatype::QINT8);
  } else if (out_dim.getDataType() == nntrainer::Tdatatype::QINT8) {
    // A conv NEVER passes int8 through: a Q8_0 conv emits int8 via the
    // quantize epilogue (handled above), any other conv dequantizes its
    // input and produces FP32. Without this, an FP32-weight conv fed a QINT8
    // activation would inherit QINT8 from its input dtype and write FP32
    // bytes into an int8-typed tensor -- the next layer then reads a garbage
    // per-tensor scale.
    out_dim.setDataType(nntrainer::Tdatatype::FP32);
  }

  context.setOutputDimensions({out_dim});

  NNTR_THROW_IF(eff_in_height < kernel_size[0] || eff_in_width < kernel_size[1],
                std::invalid_argument)
    << "Failed to initialize: in size + padding is smaller than effective "
       "kernel";

  unsigned int IM = std::numeric_limits<int>::max();

  NNTR_THROW_IF(eff_in_height - padding[0] - kernel_size[0] > IM ||
                  eff_in_width - padding[2] - kernel_size[1] > IM,
                std::invalid_argument)
    << "Failed to initialize: Calculated patch end is over int max";

  // inline fused activation (set by the FusionRealizer): build the same
  // ActiFunc a standalone ActivationLayer would, for a value-identical
  // epilogue.
  auto &fused_act = std::get<props::FusedActivation>(conv_props);
  rejectFusedActivationOnTrainingGraph(fused_act, context.getExecutionMode(),
                                       "Conv2D");
  if (!fused_act.empty() && fused_act.get() != ActivationType::ACT_NONE) {
    if (context.getActivationDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      acti_func.setActiFunc<_FP16>(fused_act.get());
#else
      throw std::invalid_argument("[Conv2D] fused fp16 act needs enable-fp16");
#endif
    } else {
      acti_func.setActiFunc<float>(fused_act.get());
    }
  }

  // Forward scratch (groups==1 path only): the im2col column buffer and the
  // quantized-GEMM output are otherwise heap-allocated on every forwarding()
  // call. Request them once here (planned into the shared activation arena,
  // FORWARD_FUNC_LIFESPAN) and reuse — no per-forward malloc/free churn. The
  // grouped path keeps its local buffer. NOTE: the im2col buffer must still be
  // re-zeroed each forward (im2col skips padding positions and the arena is
  // reused across layers), so this saves the allocation, not the zeroing.
  wt_idx[ConvParams::im2col_scratch] = std::numeric_limits<unsigned int>::max();
  wt_idx[ConvParams::qgemm_scratch] = std::numeric_limits<unsigned int>::max();
  wt_idx[ConvParams::q8act_scratch] = std::numeric_limits<unsigned int>::max();
  if (groups == 1) {
    auto scratch_type = in_dim.getTensorType();
    // W8A8: scratch buffers hold FP32 intermediates (im2col columns, the
    // pre-quantize GEMM output) even when the graph edge dtype is QINT8.
    if (scratch_type.data_type == nntrainer::Tdatatype::QINT8)
      scratch_type.data_type = nntrainer::Tdatatype::FP32;
    const unsigned int owoh = out_dim.width() * out_dim.height();
    const bool is_1x1_s1 = kernel_size[0].get() == 1 &&
                           kernel_size[1].get() == 1 && stride[0].get() == 1 &&
                           stride[1].get() == 1;
    // im2col column buffer [batch, 1, CRS, OH*OW]. Unused by the quant paths
    // that never materialize a col buffer: the 1x1 path (im2col is an identity
    // handled by an input transpose) and, where the fused backend op exists,
    // the non-1x1 path (gather is fused into the q8_0 activation packing).
    // (Q8_0 weights always use the portable indirect GEMM -- also on x86 --
    // so they never need the col buffer regardless of the ISA-specific
    // NNTR_HAS_Q4_0_INDIRECT_CONV flag.)
    if (!(quant_matmul_filter &&
          (is_1x1_s1 || NNTR_HAS_Q4_0_INDIRECT_CONV ||
           in_t_type.data_type == nntrainer::Tdatatype::Q8_0))) {
      // FP path or quant fallback: materialize the im2col column buffer
      // [batch, 1, CRS, OH*OW] once (planned into the activation arena). The
      // quant 1x1 path (identity input transpose) and the quant indirect path
      // (gather fused into the GEMM's q8_0 packing) never materialize a col
      // buffer, so they request no im2col_scratch here.
      TensorDim col_dim(in_dim.batch(), 1, real_kernel_dim.getFeatureLen(),
                        owoh, scratch_type);
      wt_idx[ConvParams::im2col_scratch] =
        context.requestTensor(col_dim, "im2col", Initializer::NONE, false,
                              TensorLifespan::FORWARD_FUNC_LIFESPAN);
    }
    // quantized GEMM output [batch, 1, OH*OW, out_ch] (quant path only).
    if (quant_matmul_filter) {
      TensorDim tmp_dim(in_dim.batch(), 1, owoh, filter_size, scratch_type);
      wt_idx[ConvParams::qgemm_scratch] =
        context.requestTensor(tmp_dim, "qgemm_out", Initializer::NONE, false,
                              TensorLifespan::FORWARD_FUNC_LIFESPAN);
    }
    // (W8A8 uses forward-local heap buffers for its FP32 GEMM output and input
    // dequantization -- pool INFER scratch is not materialized for inference
    // and a FUNC scratch aliases live activation memory; see forwarding().)
    // Q8_0 activation scratch for NHWC W4A8 path: pre-allocated once so
    // forwarding never calls malloc. Size = max(owoh, in_h*in_w) * nb blocks,
    // stored as a plain float buffer and reinterpret-cast to block_q8_0*.
    // Sized per batch so concurrent batch slices (ParallelBatch with batch>1)
    // each get an isolated region — a single shared buffer would let one slice
    // overwrite another's q8_0 bytes mid-GEMM.
    // FORWARD_INFER_LIFESPAN (LongTerm) is used instead of
    // FORWARD_FUNC_LIFESPAN (ShortTerm) because ShortTerm scratch shares memory
    // with activation tensors in the pool. Writing Q8_0 bytes there corrupts
    // skip-connection activations that are still live when this layer's forward
    // runs. LongTerm gives this scratch its own allocation that never overlaps
    // with activation memory.
    const bool nhwc_layout =
      (in_dim.getFormat() == ml::train::TensorDim::Format::NHWC);
    if (quant_matmul_filter && nhwc_layout && NNTR_HAS_Q4_0_INDIRECT_CONV) {
      const int in_ch_i = (int)in_dim.channel();
      if (in_ch_i % 32 == 0) {
        const unsigned int max_sp =
          std::max(owoh, (unsigned int)(in_dim.height() * in_dim.width()));
        const unsigned int nb = (unsigned int)in_ch_i / 32;
        // block_q8_0 = 34 bytes; use scratch_type (FP16=2 bytes) for compat.
        const unsigned int n_elems =
          (max_sp * nb * 34 + 1) / 2; // round up to FP16 elements
        // One independent scratch region per batch slot.
        TensorDim q8dim(in_dim.batch(), 1, 1, n_elems, scratch_type);
        wt_idx[ConvParams::q8act_scratch] =
          context.requestTensor(q8dim, "q8act", Initializer::NONE, false,
                                TensorLifespan::FORWARD_INFER_LIFESPAN);
      }
    }
  }
}

void Conv2DLayer::forwarding(RunLayerContext &context, bool training) {
  int status = ML_ERROR_NONE;
  // Optional per-conv timing (env NNTR_CONV_TIME). RAII: prints on scope exit,
  // covering all return paths. Skipped unless the env var is set (zero overhead
  // otherwise — the chrono calls are behind the getenv).
  struct ConvTimer {
    std::chrono::steady_clock::time_point t0;
    std::string name;
    bool on;
    ConvTimer(const std::string &n) :
      t0(std::chrono::steady_clock::now()),
      name(n),
      on(std::getenv("NNTR_CONV_TIME") != nullptr) {}
    ~ConvTimer() {
      if (on) {
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cerr << "[CONVTIME] " << name << " " << ms << " ms\n"
                  << std::flush;
      }
    }
  };
  ConvTimer __ct(context.getName());

  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);
  auto &kernel_size =
    std::get<std::array<props::KernelSize, CONV2D_DIM>>(conv_props);

  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);

  if (const char *dl = std::getenv("NNTR_WDUMP_LAYER");
      dl && context.getName() == dl) {
    const char *dp = std::getenv("NNTR_WDUMP_PATH");
    if (dp && filter_kernel.getDataType() == nntrainer::Tdatatype::FP32) {
      std::ofstream of(dp, std::ios::binary);
      of.write(reinterpret_cast<const char *>(filter_kernel.getData<float>()),
               filter_kernel.size() * sizeof(float));
    }
  }

  if (const char *dl = std::getenv("NNTR_IDUMP_LAYER");
      dl && context.getName() == dl) {
    const char *dp = std::getenv("NNTR_IDUMP_PATH");
    if (dp) {
      std::ofstream of(dp, std::ios::binary);
      if (input_.getDataType() == nntrainer::Tdatatype::QINT8) {
        int32_t marker = -1; // dtype marker: -1=QINT8
        float sc = input_.getScale<float>()[0];
        int32_t n = (int32_t)input_.size();
        of.write(reinterpret_cast<const char *>(&marker), sizeof(int32_t));
        of.write(reinterpret_cast<const char *>(&sc), sizeof(float));
        of.write(reinterpret_cast<const char *>(&n), sizeof(int32_t));
        of.write(reinterpret_cast<const char *>(input_.getData<int8_t>()),
                 input_.size());
      } else if (input_.getDataType() == nntrainer::Tdatatype::FP32) {
        int32_t marker = 1; // dtype marker: 1=FP32
        of.write(reinterpret_cast<const char *>(&marker), sizeof(int32_t));
        of.write(reinterpret_cast<const char *>(input_.getData<float>()),
                 input_.size() * sizeof(float));
      }
    }
  }

#if defined(__ARM_NEON) && defined(ENABLE_FP16)
  if (context.getName() == "conv0" &&
      hidden_.getDataType() == nntrainer::Tdatatype::FP16 &&
      input_.getDataType() == nntrainer::Tdatatype::FP16 &&
      filter_kernel.getDataType() == nntrainer::Tdatatype::FP16 &&
      input_.channel() == 3 && hidden_.channel() == 64 &&
      input_.height() == 832 && input_.width() == 832 &&
      kernel_size[0].get() == 3 && kernel_size[1].get() == 3 &&
      stride[0].get() == 2 && stride[1].get() == 2 && padding[0] == 1 &&
      padding[1] == 1 && padding[2] == 1 && padding[3] == 1) {

    // Repack weights on first run into [3, 3, 3, 64]
    if (repacked_conv0_weights_fp16.empty()) {
      repacked_conv0_weights_fp16.resize(3 * 3 * 3 * 64);
      const _FP16 *filt = filter_kernel.getData<_FP16>();
      for (unsigned int kh = 0; kh < 3; ++kh) {
        for (unsigned int kw = 0; kw < 3; ++kw) {
          for (unsigned int ic = 0; ic < 3; ++ic) {
            for (unsigned int oc = 0; oc < 64; ++oc) {
              repacked_conv0_weights_fp16[kh * 192 + kw * 64 + ic * 64 + oc] =
                filt[oc * 27 + ic * 9 + kh * 3 + kw];
            }
          }
        }
      }
    }

    const _FP16 *in = input_.getData<_FP16>();
    _FP16 *out = hidden_.getData<_FP16>();
    Tensor &bias_kernel = context.getWeight(wt_idx[ConvParams::bias]);
    const _FP16 *bias = bias_kernel.getData<_FP16>();
    const _FP16 *r_weight = repacked_conv0_weights_fp16.data();

    ThreadManager::Global().parallel_for(0, 416, [&](size_t oh) {
      int hs = -(int)1 + (int)oh * 2;
      _FP16 *out_row = out + (size_t)oh * 416 * 64;

      for (int ow = 0; ow < 416; ++ow) {
        int ws = -(int)1 + (int)ow * 2;
        _FP16 *out_pixel = out_row + (size_t)ow * 64;

        float16x8_t vacc0 =
          vld1q_f16(reinterpret_cast<const __fp16 *>(bias + 0));
        float16x8_t vacc1 =
          vld1q_f16(reinterpret_cast<const __fp16 *>(bias + 8));
        float16x8_t vacc2 =
          vld1q_f16(reinterpret_cast<const __fp16 *>(bias + 16));
        float16x8_t vacc3 =
          vld1q_f16(reinterpret_cast<const __fp16 *>(bias + 24));
        float16x8_t vacc4 =
          vld1q_f16(reinterpret_cast<const __fp16 *>(bias + 32));
        float16x8_t vacc5 =
          vld1q_f16(reinterpret_cast<const __fp16 *>(bias + 40));
        float16x8_t vacc6 =
          vld1q_f16(reinterpret_cast<const __fp16 *>(bias + 48));
        float16x8_t vacc7 =
          vld1q_f16(reinterpret_cast<const __fp16 *>(bias + 56));

        for (int kh = 0; kh < 3; ++kh) {
          int ih = hs + kh;
          if (ih < 0 || ih >= 832)
            continue;

          for (int kw = 0; kw < 3; ++kw) {
            int iw = ws + kw;
            if (iw < 0 || iw >= 832)
              continue;

            const _FP16 *in_pixel = in + ((size_t)ih * 832 + iw) * 3;
            float16x8_t vin0 = vdupq_n_f16(in_pixel[0]);
            float16x8_t vin1 = vdupq_n_f16(in_pixel[1]);
            float16x8_t vin2 = vdupq_n_f16(in_pixel[2]);

            const _FP16 *w_base = r_weight + kh * 192 + kw * 64;

            vacc0 = vfmaq_f16(
              vacc0, vin0,
              vld1q_f16(reinterpret_cast<const __fp16 *>(w_base + 0 * 64 + 0)));
            vacc1 = vfmaq_f16(
              vacc1, vin0,
              vld1q_f16(reinterpret_cast<const __fp16 *>(w_base + 0 * 64 + 8)));
            vacc2 = vfmaq_f16(vacc2, vin0,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 0 * 64 + 16)));
            vacc3 = vfmaq_f16(vacc3, vin0,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 0 * 64 + 24)));
            vacc4 = vfmaq_f16(vacc4, vin0,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 0 * 64 + 32)));
            vacc5 = vfmaq_f16(vacc5, vin0,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 0 * 64 + 40)));
            vacc6 = vfmaq_f16(vacc6, vin0,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 0 * 64 + 48)));
            vacc7 = vfmaq_f16(vacc7, vin0,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 0 * 64 + 56)));

            vacc0 = vfmaq_f16(
              vacc0, vin1,
              vld1q_f16(reinterpret_cast<const __fp16 *>(w_base + 1 * 64 + 0)));
            vacc1 = vfmaq_f16(
              vacc1, vin1,
              vld1q_f16(reinterpret_cast<const __fp16 *>(w_base + 1 * 64 + 8)));
            vacc2 = vfmaq_f16(vacc2, vin1,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 1 * 64 + 16)));
            vacc3 = vfmaq_f16(vacc3, vin1,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 1 * 64 + 24)));
            vacc4 = vfmaq_f16(vacc4, vin1,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 1 * 64 + 32)));
            vacc5 = vfmaq_f16(vacc5, vin1,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 1 * 64 + 40)));
            vacc6 = vfmaq_f16(vacc6, vin1,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 1 * 64 + 48)));
            vacc7 = vfmaq_f16(vacc7, vin1,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 1 * 64 + 56)));

            vacc0 = vfmaq_f16(
              vacc0, vin2,
              vld1q_f16(reinterpret_cast<const __fp16 *>(w_base + 2 * 64 + 0)));
            vacc1 = vfmaq_f16(
              vacc1, vin2,
              vld1q_f16(reinterpret_cast<const __fp16 *>(w_base + 2 * 64 + 8)));
            vacc2 = vfmaq_f16(vacc2, vin2,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 2 * 64 + 16)));
            vacc3 = vfmaq_f16(vacc3, vin2,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 2 * 64 + 24)));
            vacc4 = vfmaq_f16(vacc4, vin2,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 2 * 64 + 32)));
            vacc5 = vfmaq_f16(vacc5, vin2,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 2 * 64 + 40)));
            vacc6 = vfmaq_f16(vacc6, vin2,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 2 * 64 + 48)));
            vacc7 = vfmaq_f16(vacc7, vin2,
                              vld1q_f16(reinterpret_cast<const __fp16 *>(
                                w_base + 2 * 64 + 56)));
          }
        }

        vst1q_f16(reinterpret_cast<__fp16 *>(out_pixel + 0), vacc0);
        vst1q_f16(reinterpret_cast<__fp16 *>(out_pixel + 8), vacc1);
        vst1q_f16(reinterpret_cast<__fp16 *>(out_pixel + 16), vacc2);
        vst1q_f16(reinterpret_cast<__fp16 *>(out_pixel + 24), vacc3);
        vst1q_f16(reinterpret_cast<__fp16 *>(out_pixel + 32), vacc4);
        vst1q_f16(reinterpret_cast<__fp16 *>(out_pixel + 40), vacc5);
        vst1q_f16(reinterpret_cast<__fp16 *>(out_pixel + 48), vacc6);
        vst1q_f16(reinterpret_cast<__fp16 *>(out_pixel + 56), vacc7);
      }
    });

    // Fused activation epilogue for conv0 SiLU (Swish) using LUT!
    if (auto &act = std::get<props::FusedActivation>(conv_props);
        !act.empty() && act.get() == ActivationType::ACT_SWISH) {
      const size_t n = hidden_.size();
      _FP16 *data = out;
      const _FP16 *lut = get_silu_lut_fp16();
      auto &tm = ThreadManager::Global();
      const size_t nthreads = std::max<size_t>(1, tm.getComputeThreadCount());
      const size_t chunk = (n + nthreads - 1) / nthreads;

      tm.parallel_for(0, nthreads, [&](size_t t) {
        const size_t start = t * chunk;
        if (start >= n)
          return;
        const size_t end = std::min(start + chunk, n);
        size_t i = start;
        for (; i + 7 < end; i += 8) {
          uint16_t u0 = *reinterpret_cast<const uint16_t *>(&data[i + 0]);
          uint16_t u1 = *reinterpret_cast<const uint16_t *>(&data[i + 1]);
          uint16_t u2 = *reinterpret_cast<const uint16_t *>(&data[i + 2]);
          uint16_t u3 = *reinterpret_cast<const uint16_t *>(&data[i + 3]);
          uint16_t u4 = *reinterpret_cast<const uint16_t *>(&data[i + 4]);
          uint16_t u5 = *reinterpret_cast<const uint16_t *>(&data[i + 5]);
          uint16_t u6 = *reinterpret_cast<const uint16_t *>(&data[i + 6]);
          uint16_t u7 = *reinterpret_cast<const uint16_t *>(&data[i + 7]);
          data[i + 0] = lut[u0];
          data[i + 1] = lut[u1];
          data[i + 2] = lut[u2];
          data[i + 3] = lut[u3];
          data[i + 4] = lut[u4];
          data[i + 5] = lut[u5];
          data[i + 6] = lut[u6];
          data[i + 7] = lut[u7];
        }
        for (; i < end; ++i) {
          uint16_t u = *reinterpret_cast<const uint16_t *>(&data[i]);
          data[i] = lut[u];
        }
      });
    }
    return;
  }
#endif

  /** Calculate Convolution 2D
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : filter_size
   *   . Width  : Input Channel * Kernel_size[0] * Kernel_size[1]
   *
   *                              imKernel
   *                        +------|------|------+
   *                        |------|------|------|
   * [filter_size (height)] |------|------|------|
   *                        |------|------|------|
   *                        +------|------|------+
   *                     [Input Channel * Kernel_size[0]
   *                       * Kernel_size[1] (width)]
   *
   *
   * After im2Col with channel_mode true (in : input)
   *
   * This is the 2D Matrix Shape [ height ] x [ width ]
   *   . Height : Input Channel * Kernel_size[0] * Kernel_size[1]
   *   . Width  : output_dim.height * output_dim.width
   *
   *                      +-|-|-|-|      |-|-|-|-+
   *   [Input Channel     | | | | |      | | | | |
   *   * Kernel_size[0]   |_|_|_|_|      |_|_|_|_|
   *  * Kenel_size[1]     | | | | | .... | | | | |
   *    (height)]         |_|_|_|_|      |_|_|_|_|
   *                      | | | | |      | | | | |
   *                      +_|_|_|_|      |_|_|_|_+
   *                     [ output_dim.height
   *                      * output_dim.width (width) ]
   *
   * Output Dimention
   *   -> [Channel ( = filter_size = output_dim.channel )]
   *       x [output_dim.height x output_dim.width]
   */
  const TensorDim &in_dim = input_.getDim();
  const TensorDim &out_dim = hidden_.getDim();
  const TensorDim &filter_dim = filter_kernel.getDim();
  auto &groups_prop = std::get<props::ConvGroups>(conv_props);
  unsigned int groups = groups_prop.empty() ? 1 : groups_prop.get();

  if (groups == 1) {
    // A quantized 1x1 conv stores its filter as a [in_ch, out_ch] matmul weight
    // (K, N). The quantized GEMM (dotQnK for Q4_0) takes the weight as the dot
    // *input* (activation is the receiver), so we keep that layout as-is and do
    // NOT squeeze it to [out_ch, CRS] like the FP32 path.
    const auto weight_dtype = filter_kernel.getDataType();
    if (std::getenv("NNTR_CONV_DEBUG")) {
      std::cerr << "[CONVDBG] lay=" << context.getName()
                << " in_dt=" << (int)input_.getDataType()
                << " hid_dt=" << (int)hidden_.getDataType()
                << " w_dt=" << (int)weight_dtype << " filt=" << filter_size
                << " k=" << kernel_size[0].get()
                << " stride=" << stride[0].get() << " in_fmt="
                << (in_dim.getFormat() == ml::train::TensorDim::Format::NHWC
                      ? "NHWC"
                      : "NCHW")
                << " in_c=" << in_dim.channel() << " in_h=" << in_dim.height()
                << " in_w=" << in_dim.width() << " NNTR_STEM_FP32="
                << (std::getenv("NNTR_STEM_FP32") ? "set" : "null") << "\n"
                << std::flush;
    }
    const bool weight_is_q8 = (weight_dtype == nntrainer::Tdatatype::Q8_0);
    const bool weight_is_quant =
      (weight_dtype == nntrainer::Tdatatype::Q4_0 ||
       weight_dtype == nntrainer::Tdatatype::QINT4 || weight_is_q8);
    const unsigned int owoh = out_dim.width() * out_dim.height();
    // Mirrors finalize's perch_mode, including the stem exclusion (in_ch == 3):
    // the stem consumes the FP32 image directly through the ordinary FP32 conv
    // so the raw input is never quantized (matches the 81/87 S0 simulation).
    const bool perch_mode =
      std::getenv("NNTR_W8A8") != nullptr &&
      std::getenv("NNTR_W8A8_PERCH") != nullptr &&
      in_dim.getFormat() == ml::train::TensorDim::Format::NHWC &&
      in_dim.channel() != 3;

    TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                  filter_kernel.getDim().getFeatureLen()};
    filter_dim_squeezed.setTensorType(filter_kernel.getTensorType());
    if (!weight_is_quant) {
      filter_kernel.reshape(filter_dim_squeezed);
    }

    /**
     * Below sets the pad area values to zero
     * it is faster to do this way than seting selective area to zero
     */
    const bool is_1x1_s1 = kernel_size[0].get() == 1 &&
                           kernel_size[1].get() == 1 && stride[0].get() == 1 &&
                           stride[1].get() == 1;
    // Pre-allocated forward scratch (requested once in finalize). The im2col
    // column buffer is used by the FP32 path and the quant non-1x1 *fallback*;
    // the quant 1x1 path (identity input transpose) and the quant non-1x1 fused
    // path (gather folded into the GEMM) need no col buffer — finalize skips
    // im2col_scratch is materialized by finalize only for the FP/fallback path.
    // The quant 1x1 path (identity transpose) and the quant indirect path
    // (gather fused into the GEMM) requested no col buffer in finalize, so the
    // pointer stays null for them.
    const bool q8_nchw_fallback =
      weight_is_q8 &&
      in_dim.getFormat() == ml::train::TensorDim::Format::NCHW && !is_1x1_s1;
    const bool use_im2col_scratch =
      !(weight_is_quant &&
        (is_1x1_s1 || NNTR_HAS_Q4_0_INDIRECT_CONV || weight_is_q8));
    Tensor *col_scratch =
      use_im2col_scratch
        ? &context.getTensor(wt_idx[ConvParams::im2col_scratch])
        : nullptr;
    Tensor *qgemm_scratch =
      weight_is_quant ? &context.getTensor(wt_idx[ConvParams::qgemm_scratch])
                      : nullptr;
    // Q8_0 activation scratch is sized per batch (see finalize) and shared by
    // reference here; each batch slice b takes its own region via
    // getBatchSlice(b, 1) so concurrent ParallelBatch slices never alias.
    const bool has_q8act_scratch =
#ifdef ENABLE_FP16
      weight_is_quant && wt_idx[ConvParams::q8act_scratch] !=
                           std::numeric_limits<unsigned int>::max();
#else
      false;
#endif
    Tensor *q8act_scratch =
      has_q8act_scratch ? &context.getTensor(wt_idx[ConvParams::q8act_scratch])
                        : nullptr;
    if (col_scratch != nullptr) {
      col_scratch->setZero();
    }

    auto forwarding_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                              void *user_data) {
      for (unsigned int b = s; b < e; ++b) {
        Tensor out = hidden_.getBatchSlice(b, 1);
        Tensor in_sub = input_.getBatchSlice(b, 1);

        if (std::getenv("NNTR_NAN_TRACE_IN")) {
          // Check the input to THIS conv at entry, to separate tsuzuk input-nan
          // from GEMM-output-nan.
          const unsigned int n = in_sub.size();
          unsigned int n_nan = 0, n_fin = 0;
          float max_abs = 0.0f;
          if (in_sub.getDataType() == nntrainer::Tdatatype::FP16) {
#ifdef ENABLE_FP16
            const _FP16 *d = in_sub.getData<_FP16>();
            for (unsigned int i = 0; i < n; ++i) {
              float v = (float)d[i];
              if (std::isnan(v))
                n_nan++;
              else if (std::isfinite(v)) {
                n_fin++;
                float a = std::fabs(v);
                if (a > max_abs)
                  max_abs = a;
              }
            }
#endif
          } else {
            const float *d = in_sub.getData<float>();
            for (unsigned int i = 0; i < n; ++i) {
              float v = d[i];
              if (std::isnan(v))
                n_nan++;
              else if (std::isfinite(v)) {
                n_fin++;
                float a = std::fabs(v);
                if (a > max_abs)
                  max_abs = a;
              }
            }
          }
          std::cerr << "[NANIN] " << context.getName() << " dt="
                    << (in_sub.getDataType() == nntrainer::Tdatatype::FP16
                          ? "FP16"
                          : "FP32")
                    << " fin=" << n_fin << " nan=" << n_nan
                    << " maxAbs=" << max_abs << "\n"
                    << std::flush;
        }
        // ---- Per-channel W8A8 path (NNTR_W8A8_PERCH) --------------------
        // Every NHWC conv runs int8 through the int32-accumulate kernel: the
        // weight is converted once to per-channel int8 (cached); the input is
        // int8 (from the previous conv) or quantized here (the stem); the FP32
        // GEMM output gets bias + SiLU, then is quantized to int8 (or written
        // FP32 for the head-final conv, whose consumer is FP32).
        if (perch_mode) {
          // Attribute the intra-layer op scopes below (and the GEMM's internal
          // pack/matmul split) to this conv, robust to batch>1 workers.
          const unsigned int in_ch = in_dim.channel();
          const unsigned int CRS =
            in_ch * kernel_size[0].get() * kernel_size[1].get();
          bool is_q4 =
            filter_kernel.getDataType() == nntrainer::Tdatatype::Q4_0;
          const PerChConvWeight &W = __ggml_q8ch_prepare_conv_weight(
            filter_kernel.getData(),
            weight_is_q8 ? filter_kernel.getData() : nullptr,
            (!weight_is_q8 && !is_q4) ? filter_kernel.getData<float>()
                                      : nullptr,
            filter_size, CRS, kernel_size[0].get() * kernel_size[1].get(),
            in_ch, is_q4 ? filter_kernel.getData() : nullptr);

          if (const char *dl = std::getenv("NNTR_PERCH_DEBUG");
              dl && context.getName() == dl) {
            if (const char *qp = std::getenv("NNTR_PERCH_QSDUMP_PATH")) {
              const unsigned int nb_dbg = W.Kpad / QK8_0;
              const unsigned int NB4_dbg = W.Npad / 4;
              const size_t nblocks = (size_t)NB4_dbg * nb_dbg;
              std::ofstream of(qp, std::ios::binary);
              int32_t hdr[4] = {(int32_t)W.Npad, (int32_t)W.Kpad,
                                (int32_t)W.Nreal,
                                (int32_t)(W.taps_last ? 1 : 0)};
              of.write(reinterpret_cast<const char *>(hdr), sizeof(hdr));
              const size_t block_q8_0x4_bytes =
                8 + 128; // d[4] (uint16) + qs[128]
              of.write(reinterpret_cast<const char *>(W.qs_data()),
                       nblocks * block_q8_0x4_bytes);
              of.write(reinterpret_cast<const char *>(W.scale.data()),
                       W.scale.size() * sizeof(float));
            }
            float smin = 1e30f, smax = -1e30f;
            for (float s : W.scale) {
              if (s < smin)
                smin = s;
              if (s > smax)
                smax = s;
            }
            int32_t cmin = 2147483647, cmax = -2147483647;
            for (int32_t c : W.colsum) {
              if (c < cmin)
                cmin = c;
              if (c > cmax)
                cmax = c;
            }
            float wmin = 1e30f, wmax = -1e30f;
            if (!weight_is_q8) {
              const float *fp = filter_kernel.getData<float>();
              size_t n = (size_t)filter_size * CRS;
              for (size_t i = 0; i < n; ++i) {
                float v = fp[i];
                if (v < wmin)
                  wmin = v;
                if (v > wmax)
                  wmax = v;
              }
              if (const char *wp = std::getenv("NNTR_PERCH_WDUMP_PATH")) {
                std::ofstream of(wp, std::ios::binary);
                of.write(reinterpret_cast<const char *>(fp), n * sizeof(float));
              }
            }
            std::cerr << "[PERCHDBG] " << context.getName()
                      << " out_ch=" << filter_size << " CRS=" << CRS
                      << " khkw=" << kernel_size[0].get() * kernel_size[1].get()
                      << " in_ch=" << in_ch << " Npad=" << W.Npad
                      << " Kpad=" << W.Kpad << " weight_is_q8=" << weight_is_q8
                      << " raw_weight_range=[" << wmin << "," << wmax << "]"
                      << " scale_range=[" << smin << "," << smax << "]"
                      << " colsum_range=[" << cmin << "," << cmax << "]\n"
                      << std::flush;
          }

          // Asymmetric (affine) int8 activations with a SHARED fixed offset
          // (default; NNTR_W8A8_SYM=1 reverts to symmetric): every int8 edge
          // in this graph carries SiLU-domain values, whose global minimum is
          // -0.27846 (min of x*sigmoid(x)), so the representation
          //     x = (q + 128) * s - kActOff
          // spans [-kActOff, 255*s - kActOff] and uses all 256 levels on the
          // range that actually occurs -- ~2x the resolution of symmetric
          // [-amax, amax], which wastes half its levels on negatives that
          // never happen. This is what keeps ORT's borderline keypoints from
          // flipping (S0 sim: min margin +0.008 symmetric -> +0.054 affine).
          // Because the offset is a shared CONSTANT, no zero point needs to be
          // stored (the tensor scale slot keeps s) and concat/pool/upsample
          // rescale exactly as before modulo the +128 shift. The GEMM needs no
          // kernel change: sum_k w*x = s_a*s_w*acc + s_w*(128*s_a -
          // kActOff)*colsum_w, and the second term folds into the bias.
          static const bool perch_sym_env =
            std::getenv("NNTR_W8A8_SYM") != nullptr;
          // NNTR_W8A8_PERCH_LINEAR_SYM (opt-in, default off): when THIS conv
          // has NO fused activation (e.g. YOLOv11 C2PSA's proj/ffn1) and it
          // is about to quantize its OWN fresh FP32 input (the `else` branch
          // below), do it symmetrically instead of via the network-wide
          // asymmetric SiLU-floor encoding. That encoding assumes the input
          // is bounded below by -kActOff (true for a genuine SiLU output,
          // per the comment below), but a linear/no-activation conv's input
          // (a plain residual sum here) has no such bound, so applying it
          // anyway silently clips values below -kActOff before the GEMM even
          // runs. Confirmed via on-device bisection 2026-08-12: proj/ffn1
          // cosine vs W8A32 golden 0.77/0.58 with the network-wide asymmetric
          // encoding, ~0.98+ (matching every SiLU-activated conv) with this
          // fix. Deliberately does NOT change how an ALREADY-quantized QINT8
          // input (the `if` branch below) is interpreted: that data's actual
          // encoding was chosen by its PRODUCER (always has_silu=true, since
          // perch_q8out below requires it), not by this consumer, so
          // decoding it must still follow perch_sym_env alone -- see
          // a_is_asym vs the FP32-quantize-only perch_asym below. Default
          // off so no other perch_mode caller (video_pipeline's detector/
          // pose always pair NNTR_W8A8 with NNTR_W8A8_PERCH) is affected.
          static const bool perch_linear_sym_env =
            std::getenv("NNTR_W8A8_PERCH_LINEAR_SYM") != nullptr;
          const bool this_has_silu = [&] {
            if (auto &actp = std::get<props::FusedActivation>(conv_props);
                !actp.empty() && actp.get() == ActivationType::ACT_SWISH)
              return true;
            return false;
          }();
          const bool perch_asym =
            !perch_sym_env && !(perch_linear_sym_env && !this_has_silu);
          constexpr float kActOff = 0.27846455f;

          // input int8 + per-tensor scale
          // (thread_local: reused across layers/forwards on this thread --
          // ParallelBatch workers each get their own -- so the per-forward
          // malloc/free churn disappears; every byte is overwritten below.)
          const int8_t *a_i8 = nullptr;
          float a_scale = 1.f;
          // Whether a_i8/a_scale (whichever branch below fills them) actually
          // carries the asymmetric encoding -- used below (pad_q, bias fold)
          // instead of perch_asym directly, since the two branches decide it
          // differently (see perch_linear_sym_env comment above).
          bool a_is_asym = !perch_sym_env;
          static thread_local std::vector<int8_t> a_buf;
          if (in_sub.getDataType() == nntrainer::Tdatatype::QINT8) {
            a_i8 = in_sub.getData<int8_t>();
            a_scale = in_sub.getScale<float>()[0];
            if (const char *dl2 = std::getenv("NNTR_PERCH_AT_USE_DUMP_LAYER");
                dl2 && context.getName() == dl2) {
              const char *dp2 = std::getenv("NNTR_PERCH_AT_USE_DUMP_PATH");
              if (dp2) {
                std::ofstream of(dp2, std::ios::binary);
                float sc = a_scale;
                int32_t n = (int32_t)in_sub.size();
                of.write(reinterpret_cast<const char *>(&sc), sizeof(float));
                of.write(reinterpret_cast<const char *>(&n), sizeof(int32_t));
                of.write(reinterpret_cast<const char *>(a_i8), in_sub.size());
              }
            }
          } else {
            const float *fin = in_sub.getData<float>();
            const size_t n_in = in_sub.size();
            // Chunk-parallel NEON amax + quantize: this input is the stem's
            // full-resolution FP32 output (~800K elements) and the former
            // single-thread scalar loops were the largest single conv cost.
            auto &tmq = ThreadManager::Global();
            const size_t qch = 1 << 15;
            const size_t nqc = (n_in + qch - 1) / qch;
            static thread_local std::vector<float> chunk_amax;
            if (chunk_amax.size() < nqc)
              chunk_amax.resize(nqc);
            float *cam = chunk_amax.data();
            tmq.parallel_for(0, nqc, [=](size_t ci) {
              const size_t i0 = ci * qch, i1 = std::min(i0 + qch, n_in);
              cam[ci] = nntr_absmax_f32(fin + i0, i1 - i0);
            });
            float amax = 0.f;
            for (size_t ci = 0; ci < nqc; ++ci)
              amax = std::max(amax, cam[ci]);
            // FP32 activation scale (NOT fp16-rounded): the per-channel kernel
            // consumes a_scale as an FP32 multiplier and ignores block d, so
            // the convRoundScaleFp16 used by the per-block path is a needless
            // per-layer precision loss here. Matching the FP32-scale S0
            // simulation (which holds 81/87) is what the fp16 rounding broke on
            // device (per-channel 81 -> 80); ~50 layers of ~5e-4 scale error
            // accumulate across the int8-resident pipeline.
            // (This input is the stem's SiLU output, so the affine range
            // [-kActOff, amax] is valid; amax >= max(x) covers the top end.)
            a_buf.resize(n_in);
            int8_t *ab = a_buf.data();
            a_is_asym = perch_asym; // this conv's own quantize choice
            if (perch_asym) {
              a_scale = amax > 0.f ? (amax + kActOff) / 255.f : 1.f;
              const float inv = 1.f / a_scale;
              tmq.parallel_for(0, nqc, [=](size_t ci) {
                const size_t i0 = ci * qch, i1 = std::min(i0 + qch, n_in);
                nntr_quantize_affine_i8(fin + i0, ab + i0, i1 - i0, inv,
                                        kActOff);
              });
            } else {
              a_scale = amax > 0.f ? amax / 127.f : 1.f;
              const float inv = 1.f / a_scale;
              tmq.parallel_for(0, nqc, [=](size_t ci) {
                const size_t i0 = ci * qch, i1 = std::min(i0 + qch, n_in);
                for (size_t i = i0; i < i1; ++i)
                  ab[i] = (int8_t)std::max(
                    -128.f, std::min(127.f, std::round(fin[i] * inv)));
              });
            }
            a_i8 = a_buf.data();
          }

          ConvGatherParams geom;
          geom.in_ch = (int)in_ch;
          geom.in_h = in_dim.height();
          geom.in_w = in_dim.width();
          geom.k_h = kernel_size[0].get();
          geom.k_w = kernel_size[1].get();
          geom.pad_t = padding[0];
          geom.pad_l = padding[2];
          geom.stride_h = stride[0].get();
          geom.stride_w = stride[1].get();
          geom.dil_h = dilation[0].get();
          geom.dil_w = dilation[1].get();
          geom.out_w = out_dim.width();
          geom.is_nhwc = true;

          const bool out_qint8 =
            out.getDataType() == nntrainer::Tdatatype::QINT8;
          // Reused thread_local GEMM output buffer, NOT zeroed: the GEMM tiles
          // cover every [M x N] element (bulk rows via the tiled kernel, the
          // M%4 tail via its scratch memcpy), so the former per-forward
          // assign(.., 0.f) was a full-tensor memset thrown away immediately.
          static thread_local std::vector<float> cbuf;
          float *cptr;
          if (out_qint8) {
            const size_t need = (size_t)owoh * filter_size;
            if (cbuf.size() < need)
              cbuf.resize(need);
            cptr = cbuf.data();
          } else {
            cptr = out.getData<float>(); // NHWC [owoh, out_ch] == GEMM layout
          }

          // Padded gather positions must represent x = 0, whose affine code is
          // round(kActOff / s) - 128 (the byte 0 would mean x = 128*s -
          // kActOff).
          int8_t pad_q = 0;
          if (a_is_asym)
            pad_q = (int8_t)std::max(
              -128L, std::min(127L, std::lround(kActOff / a_scale) - 128L));

          if (const char *dl = std::getenv("NNTR_PERCH_ADUMP_LAYER");
              dl && context.getName() == dl) {
            const char *dp = std::getenv("NNTR_PERCH_ADUMP_PATH");
            if (dp) {
              std::ofstream of(dp, std::ios::binary);
              of.write(reinterpret_cast<const char *>(&a_scale), sizeof(float));
              int32_t nel = (int32_t)in_sub.size();
              of.write(reinterpret_cast<const char *>(&nel), sizeof(int32_t));
              of.write(reinterpret_cast<const char *>(a_i8), in_sub.size());
            }
          }

          __ggml_q8ch_indirect_GEMM(owoh, filter_size, W.Kpad, a_i8, a_scale,
                                    geom, W.qs_data(), W.scale.data(), cptr,
                                    pad_q, W.taps_last);

          if (const char *dl = std::getenv("NNTR_PERCH_CDUMP_LAYER");
              dl && context.getName() == dl) {
            const char *dp = std::getenv("NNTR_PERCH_CDUMP_PATH");
            if (dp) {
              std::ofstream of(dp, std::ios::binary);
              of.write(reinterpret_cast<const char *>(cptr),
                       (size_t)owoh * filter_size * sizeof(float));
            }
          }

          if (const char *dl = std::getenv("NNTR_PERCH_DEBUG");
              (dl && context.getName() == dl) ||
              std::getenv("NNTR_PERCH_DEBUG_ALL") != nullptr) {
            float cmin = 1e30f, cmax = -1e30f;
            size_t n = (size_t)owoh * filter_size;
            for (size_t i = 0; i < n; ++i) {
              float v = cptr[i];
              if (v < cmin)
                cmin = v;
              if (v > cmax)
                cmax = v;
            }
            int8_t amin = 127, amax_i8 = -128;
            for (size_t i = 0; i < in_sub.size(); ++i) {
              int8_t v = a_i8[i];
              if (v < amin)
                amin = v;
              if (v > amax_i8)
                amax_i8 = v;
            }
            std::cerr << "[PERCHDBG] " << context.getName()
                      << " post-GEMM cptr_range=[" << cmin << "," << cmax << "]"
                      << " a_scale=" << a_scale << " a_i8_range=[" << (int)amin
                      << "," << (int)amax_i8 << "]"
                      << " pad_q=" << (int)pad_q << " owoh=" << owoh
                      << " Kpad=" << W.Kpad << " taps_last=" << W.taps_last
                      << "\n"
                      << std::flush;
          }

          // bias + SiLU on FP32, then (for an int8 output) requantize.
          {
            auto &tm = ThreadManager::Global();
            const unsigned int C = filter_size;
            const size_t nout = (size_t)owoh * filter_size;
            const float *bptr = nullptr;
            if (auto &db = std::get<props::DisableBias>(*layer_impl_props);
                db.empty() || db.get() == false)
              bptr =
                context.getWeight(wt_idx[ConvParams::bias]).getData<float>();
            // Affine activations: the GEMM accumulated sum_k q_w * q_a with the
            // offset still inside q_a; the exact linear term
            //   s_w[j] * (128*s_a - kActOff) * colsum_w[j]
            // completes sum_k w*x, and being per-output-channel constant it
            // folds into the bias (no kernel change, no per-element cost).
            static thread_local std::vector<float> ebias;
            if (a_is_asym) {
              const float koff = 128.f * a_scale - kActOff;
              if (ebias.size() < C)
                ebias.resize(C);
              for (unsigned int j = 0; j < C; ++j)
                ebias[j] = (bptr ? bptr[j] : 0.f) +
                           W.scale[j] * koff * (float)W.colsum[j];
              bptr = ebias.data();
            }
            static const bool approx_silu =
              std::getenv("NNTR_APPROX_SILU") != nullptr &&
              std::string(std::getenv("NNTR_APPROX_SILU")) == "1";
            int act = 0;
            if (auto &actp = std::get<props::FusedActivation>(conv_props);
                !actp.empty() && actp.get() == ActivationType::ACT_SWISH)
              act = approx_silu ? 2 : 1;

            if (out_qint8) {
              // One fused row-parallel pass: bias + SiLU + per-row amax; then a
              // parallel quantize pass. Replaces the former four passes (three
              // single-threaded) so the epilogue is no longer the conv's cost.
              static thread_local std::vector<float> ramax;
              if (ramax.size() < owoh)
                ramax.resize(owoh);
              float *rp = ramax.data();
              tm.parallel_for(0, owoh, [=](size_t p) {
                rp[p] = convBiasActRow(cptr + (size_t)p * C, bptr, C, act);
              });
              float amax = 0.f;
              for (unsigned int p = 0; p < owoh; ++p)
                amax = std::max(amax, rp[p]);

              if (const char *dl = std::getenv("NNTR_PERCH_DUMP_LAYER");
                  dl && context.getName() == dl) {
                const char *dp = std::getenv("NNTR_PERCH_DUMP_PATH");
                if (dp) {
                  std::ofstream of(dp, std::ios::binary);
                  of.write(reinterpret_cast<const char *>(cptr),
                           (size_t)owoh * filter_size * sizeof(float));
                }
              }
              // FP32 output scale (see the input-quant note): the consumer is a
              // per-channel conv (FP32 act_scale) or an int8
              // concat/pool/upsample that forwards this scale, never a
              // block_q8_0 fp16 d, so keep the scale in FP32 to match the S0
              // simulation's 81/87. Affine: the output is a SiLU tensor (>=
              // -kActOff); span
              // [-kActOff, amax] over 256 levels (amax = max|x| >= max(x)).
              float sc;
              if (perch_asym)
                sc = amax > 0.f ? (amax + kActOff) / 255.f : 1.f;
              else
                sc = amax > 0.f ? amax / 127.f : 1.f;
              const float inv = 1.f / sc;
              const bool asym_q = perch_asym;
              int8_t *qo = out.getData<int8_t>();
              const size_t chunk = 4096;
              const size_t nchunk = (nout + chunk - 1) / chunk;
              tm.parallel_for(0, nchunk, [=](size_t ci) {
                const size_t i0 = ci * chunk;
                const size_t i1 = std::min(i0 + chunk, nout);
                if (asym_q) {
                  // NEON affine quantize (bit-identical to the scalar form).
                  nntr_quantize_affine_i8(cptr + i0, qo + i0, i1 - i0, inv,
                                          kActOff);
                } else {
                  for (size_t i = i0; i < i1; ++i)
                    qo[i] = (int8_t)std::max(
                      -128.f, std::min(127.f, std::round(cptr[i] * inv)));
                }
              });
              hidden_.getScale<float>()[0] = sc;
            } else if (bptr || act) {
              // FP32 output (head-final conv, out_ch % 4 != 0): fused bias
              // (incl. the affine correction) + SiLU, row-parallel, no requant.
              tm.parallel_for(0, owoh, [=](size_t p) {
                convBiasActRow(cptr + (size_t)p * C, bptr, C, act);
              });
            }
          }
          continue;
        }
        // ----------------------------------------------------------------

        // ---- Stem direct FP32 conv (per-channel W8A8 mode) --------------
        // The stem is excluded from the int8 path (its input is the FP32
        // image), but the generic im2col+sgemm route costs ~3 ms for a 22
        // MMAC conv. With in_ch == 3 a direct tap loop is far cheaper: 16
        // output channels stay in four NEON accumulators per pixel, each of
        // the <=27 (tap, channel) scalars broadcast-FMAs a contiguous
        // out-channel weight vector (tap-major cache above). Bias + SiLU
        // reuse the fused convBiasActRow row epilogue.
        // Originally W8A8-gated (kept the stem FP32 in an int8 pipeline); the
        // same tap-direct path is a large win for W8A32 too — a tiny 3x3 s2
        // 3->16 stem at 640x640 hits ~15ms on the generic im2col+sgemm route
        // (the dominant detector cost) because the GEMV-shaped conv is ill-
        // served by a full im2col slab. Enable for W8A32 via
        // NNTR_STEM_DIRECT_FP32.
        const bool stem_direct_w8a8 = std::getenv("NNTR_W8A8") != nullptr &&
                                      std::getenv("NNTR_W8A8_PERCH") != nullptr;
        if ((stem_direct_w8a8 ||
             (std::getenv("NNTR_STEM_DIRECT_FP32") != nullptr &&
              !weight_is_quant)) &&
            !weight_is_quant &&
            (std::get<props::ConvGroups>(conv_props).empty() ||
             std::get<props::ConvGroups>(conv_props).get() == 1) &&
            in_dim.getFormat() == ml::train::TensorDim::Format::NHWC &&
            [&] {
              // W8A8 path keeps the legacy in_ch==3 stem restriction; the
              // W8A32 tap-direct opt-in (NNTR_STEM_DIRECT_FP32) extends to the
              // small-in_ch early backbone convs (conv1: in_ch=16) via
              // NNTR_STEM_DIRECT_MAXCIN (default 3, raise to cover conv1).
              // IMPORTANT: gate on the layer name containing "/" so the
              // detector's "conv0/conv"/"conv1/conv" (slash-suffixed) match but
              // PFLD's bare "conv1"/"conv2_dw" (no slash) do NOT — PFLD is
              // bit-exact-verified and a tap-direct path change breaks its
              // parity (observed: landmark first-5 shifted 0.80→0.27).
              if (context.getName().find('/') == std::string::npos)
                return false;
              if (stem_direct_w8a8)
                return in_dim.channel() == 3;
              unsigned maxc = 3;
              if (const char *m = std::getenv("NNTR_STEM_DIRECT_MAXCIN"))
                maxc = (unsigned)std::strtoul(m, nullptr, 10);
              return in_dim.channel() <= maxc;
            }() &&
            in_sub.getDataType() == nntrainer::Tdatatype::FP32 &&
            out.getDataType() == nntrainer::Tdatatype::FP32 &&
            dilation[0].get() == 1 && dilation[1].get() == 1 &&
            filter_size % 4 == 0) {
          const unsigned int Cin = in_dim.channel();
          const unsigned int Cout = filter_size;
          const unsigned int kh_ = kernel_size[0].get();
          const unsigned int kw_ = kernel_size[1].get();
          const std::vector<float> &TW = getStemTapWeight(
            filter_kernel.getData<float>(), Cout, Cin, kh_, kw_);
          const float *twp = TW.data();
          const float *bptr = nullptr;
          if (auto &db = std::get<props::DisableBias>(*layer_impl_props);
              db.empty() || db.get() == false)
            bptr = context.getWeight(wt_idx[ConvParams::bias]).getData<float>();
          static const bool approx_silu_s =
            std::getenv("NNTR_APPROX_SILU") != nullptr &&
            std::string(std::getenv("NNTR_APPROX_SILU")) == "1";
          int act = 0;
          if (auto &actp = std::get<props::FusedActivation>(conv_props);
              !actp.empty() && actp.get() == ActivationType::ACT_SWISH)
            act = approx_silu_s ? 2 : 1;

          const int Hi = in_dim.height(), Wi = in_dim.width();
          const int Ho = out_dim.height(), Wo = out_dim.width();
          const int sh = stride[0].get(), sw = stride[1].get();
          const int ptop = padding[0], pleft = padding[2];
          const float *inp = in_sub.getData<float>();
          float *outp = out.getData<float>();
          auto &tms = ThreadManager::Global();
          tms.parallel_for(0, (size_t)Ho, [=](size_t ohs) {
            const int oh = (int)ohs;
            const int h0 = oh * sh - ptop;
            for (int ow = 0; ow < Wo; ++ow) {
              float *op = outp + ((size_t)oh * Wo + ow) * Cout;
              const int w0 = ow * sw - pleft;
              unsigned int cb = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
              for (; cb + 15 < Cout; cb += 16) {
                float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
                float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
                for (unsigned int y = 0; y < kh_; ++y) {
                  const int ih = h0 + (int)y;
                  if (ih < 0 || ih >= Hi)
                    continue;
                  for (unsigned int x = 0; x < kw_; ++x) {
                    const int iw = w0 + (int)x;
                    if (iw < 0 || iw >= Wi)
                      continue;
                    const float *px = inp + ((size_t)ih * Wi + iw) * Cin;
                    const float *wt =
                      twp + (((size_t)y * kw_ + x) * Cin) * Cout + cb;
                    for (unsigned int c = 0; c < Cin; ++c) {
                      const float xv = px[c];
                      const float *wv = wt + (size_t)c * Cout;
                      a0 = vfmaq_n_f32(a0, vld1q_f32(wv + 0), xv);
                      a1 = vfmaq_n_f32(a1, vld1q_f32(wv + 4), xv);
                      a2 = vfmaq_n_f32(a2, vld1q_f32(wv + 8), xv);
                      a3 = vfmaq_n_f32(a3, vld1q_f32(wv + 12), xv);
                    }
                  }
                }
                vst1q_f32(op + cb + 0, a0);
                vst1q_f32(op + cb + 4, a1);
                vst1q_f32(op + cb + 8, a2);
                vst1q_f32(op + cb + 12, a3);
              }
#endif
              for (; cb < Cout; ++cb) {
                float acc = 0.f;
                for (unsigned int y = 0; y < kh_; ++y) {
                  const int ih = h0 + (int)y;
                  if (ih < 0 || ih >= Hi)
                    continue;
                  for (unsigned int x = 0; x < kw_; ++x) {
                    const int iw = w0 + (int)x;
                    if (iw < 0 || iw >= Wi)
                      continue;
                    const float *px = inp + ((size_t)ih * Wi + iw) * Cin;
                    const float *wt =
                      twp + (((size_t)y * kw_ + x) * Cin) * Cout + cb;
                    for (unsigned int c = 0; c < Cin; ++c)
                      acc += px[c] * wt[(size_t)c * Cout];
                  }
                }
                op[cb] = acc;
              }
              // fused bias + SiLU on the freshly built out-channel row
              convBiasActRow(op, bptr, Cout, act);
            }
          });
          continue;
        }
        // ----------------------------------------------------------------

        // ---- Stem direct FP16 conv (W8A16 keyword mode) -------------------
        // Same tap-major direct-conv idea as the W8A8 stem block above, but
        // for the FastViT keyword stem0 (FP16 in/out, FP32 weight, in_ch==3,
        // NHWC, groups==1, 3x3 stride-2 @ 320x320). convQuantEligible excludes
        // it from Q8_0 (27 = 3*9 not %32), so it falls to the generic
        // im2col+sgemm path at ~4.4ms -- the #1 keyword cost after the
        // grouped-conv levers. With only 3 input channels and 64 outputs the
        // tap-direct broadcast (each (tap,channel) scalar FMA'd against a 16-
        // wide out-channel weight vector) is far cheaper than materializing
        // the 27xOH*OW col slab + sgemm. No fused activation (keyword stem
        // GELU is a separate activation layer), so act==0; the bias add is
        // folded into the FP32 accumulators before the FP16 narrow.
        if (std::getenv("NNTR_W8A8") == nullptr && !weight_is_quant &&
            (std::get<props::ConvGroups>(conv_props).empty() ||
             std::get<props::ConvGroups>(conv_props).get() == 1) &&
            in_dim.getFormat() == ml::train::TensorDim::Format::NHWC &&
            in_dim.channel() == 3 && filter_size % 16 == 0 &&
            dilation[0].get() == 1 && dilation[1].get() == 1 &&
#ifdef ENABLE_FP16
            in_sub.getDataType() == nntrainer::Tdatatype::FP16 &&
            out.getDataType() == nntrainer::Tdatatype::FP16 &&
#else
            false &&
#endif
            std::getenv("NNTR_STEM_FP32") == nullptr) {
#ifdef ENABLE_FP16
          const unsigned int Cin = in_dim.channel();
          const unsigned int Cout = filter_size;
          const unsigned int kh_ = kernel_size[0].get();
          const unsigned int kw_ = kernel_size[1].get();
          const std::vector<float> &TW = getStemTapWeight(
            filter_kernel.getData<float>(), Cout, Cin, kh_, kw_);
          const float *twp = TW.data();
          // The keyword stem0/conv carries a BN-folded weight with NO separate
          // bias: the generic path relies on disable_bias=true here and never
          // adds one (it is bit-exact at 0.00157 without a bias add). Reading
          // the bias weight slot anyway injects stale/garbage values and breaks
          // accuracy (0.029 backbone drift), so the FP16-stem branch skips bias
          // by default. NNTR_STEM_BIAS=1 forces it back on for A/B / debugging.
          const float *bptr = nullptr;
          if (std::getenv("NNTR_STEM_BIAS") != nullptr) {
            auto &db = std::get<props::DisableBias>(*layer_impl_props);
            if (db.empty() || db.get() == false)
              bptr =
                context.getWeight(wt_idx[ConvParams::bias]).getData<float>();
          }
          const int Hi = in_dim.height(), Wi = in_dim.width();
          const int Ho = out_dim.height(), Wo = out_dim.width();
          const int sh = stride[0].get(), sw = stride[1].get();
          const int ptop = padding[0], pleft = padding[2];
          const _FP16 *inp = in_sub.getData<_FP16>();
          _FP16 *outp = out.getData<_FP16>();
          auto &tms = ThreadManager::Global();
          tms.parallel_for(0, (size_t)Ho, [=](size_t ohs) {
            const int oh = (int)ohs;
            const int h0 = oh * sh - ptop;
            for (int ow = 0; ow < Wo; ++ow) {
              _FP16 *op = outp + ((size_t)oh * Wo + ow) * Cout;
              const int w0 = ow * sw - pleft;
              unsigned int cb = 0;
#if defined(__ARM_NEON)
              for (; cb + 15 < Cout; cb += 16) {
                float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
                float32x4_t a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
                for (unsigned int y = 0; y < kh_; ++y) {
                  const int ih = h0 + (int)y;
                  if (ih < 0 || ih >= Hi)
                    continue;
                  for (unsigned int x = 0; x < kw_; ++x) {
                    const int iw = w0 + (int)x;
                    if (iw < 0 || iw >= Wi)
                      continue;
                    const _FP16 *px = inp + ((size_t)ih * Wi + iw) * Cin;
                    const float *wt =
                      twp + (((size_t)y * kw_ + x) * Cin) * Cout + cb;
                    for (unsigned int c = 0; c < Cin; ++c) {
                      const float xv = (float)px[c];
                      const float *wv = wt + (size_t)c * Cout;
                      a0 = vfmaq_n_f32(a0, vld1q_f32(wv + 0), xv);
                      a1 = vfmaq_n_f32(a1, vld1q_f32(wv + 4), xv);
                      a2 = vfmaq_n_f32(a2, vld1q_f32(wv + 8), xv);
                      a3 = vfmaq_n_f32(a3, vld1q_f32(wv + 12), xv);
                    }
                  }
                }
                if (bptr) {
                  a0 = vaddq_f32(a0, vld1q_f32(bptr + cb + 0));
                  a1 = vaddq_f32(a1, vld1q_f32(bptr + cb + 4));
                  a2 = vaddq_f32(a2, vld1q_f32(bptr + cb + 8));
                  a3 = vaddq_f32(a3, vld1q_f32(bptr + cb + 12));
                }
                __fp16 tmp[4];
                vst1_f16(tmp + 0, vcvt_f16_f32(a0));
                for (int v = 0; v < 4; ++v)
                  op[cb + v] = (_FP16)tmp[v];
                vst1_f16(tmp + 0, vcvt_f16_f32(a1));
                for (int v = 0; v < 4; ++v)
                  op[cb + 4 + v] = (_FP16)tmp[v];
                vst1_f16(tmp + 0, vcvt_f16_f32(a2));
                for (int v = 0; v < 4; ++v)
                  op[cb + 8 + v] = (_FP16)tmp[v];
                vst1_f16(tmp + 0, vcvt_f16_f32(a3));
                for (int v = 0; v < 4; ++v)
                  op[cb + 12 + v] = (_FP16)tmp[v];
              }
#endif
              for (; cb < Cout; ++cb) {
                float acc = bptr ? bptr[cb] : 0.f;
                for (unsigned int y = 0; y < kh_; ++y) {
                  const int ih = h0 + (int)y;
                  if (ih < 0 || ih >= Hi)
                    continue;
                  for (unsigned int x = 0; x < kw_; ++x) {
                    const int iw = w0 + (int)x;
                    if (iw < 0 || iw >= Wi)
                      continue;
                    const _FP16 *px = inp + ((size_t)ih * Wi + iw) * Cin;
                    const float *wt =
                      twp + (((size_t)y * kw_ + x) * Cin) * Cout + cb;
                    for (unsigned int c = 0; c < Cin; ++c)
                      acc += (float)px[c] * wt[(size_t)c * Cout];
                  }
                }
                op[cb] = (_FP16)acc;
              }
            }
          });
          continue;
#endif
        }
        // ----------------------------------------------------------------

        // W8A8: an FP32-weight conv fed by a QINT8 activation dequantizes it
        // once into a heap-local FP32 buffer and runs the standard path
        // unchanged (conv is the universal dtype boundary).
        std::vector<float> deq_in_buf;
        if (!weight_is_quant &&
            in_sub.getDataType() == nntrainer::Tdatatype::QINT8) {
          const int8_t *q = in_sub.getData<int8_t>();
          const float sc = in_sub.getScale<float>()[0];
          const size_t n_in = in_sub.size();
          deq_in_buf.resize(n_in);
          float *fp = deq_in_buf.data();
          auto &tmd = ThreadManager::Global();
          const size_t chunk = 65536;
          // perch_mode's default asymmetric encoding x=(q+128)*s-kActOff (see
          // the epilogue above) must be inverted with the SAME formula here,
          // not read as plain q*s -- see float_tensor.cpp's QINT8 copyData
          // for the identical fix and rationale.
          static const bool perch_asym_env =
            std::getenv("NNTR_W8A8_PERCH") != nullptr &&
            std::getenv("NNTR_W8A8_SYM") == nullptr;
          if (perch_asym_env) {
            constexpr float kActOff = 0.27846455f;
            tmd.parallel_for(0, (n_in + chunk - 1) / chunk, [=](size_t ci) {
              const size_t i0 = ci * chunk, i1 = std::min(i0 + chunk, n_in);
              for (size_t i = i0; i < i1; ++i)
                fp[i] = ((float)q[i] + 128.f) * sc - kActOff;
            });
          } else {
            tmd.parallel_for(0, (n_in + chunk - 1) / chunk, [=](size_t ci) {
              const size_t i0 = ci * chunk, i1 = std::min(i0 + chunk, n_in);
              for (size_t i = i0; i < i1; ++i)
                fp[i] = sc * (float)q[i];
            });
          }
          TensorDim din = in_sub.getDim();
          din.setDataType(nntrainer::Tdatatype::FP32);
          in_sub = Tensor::Map<float>(deq_in_buf.data(),
                                      deq_in_buf.size() * sizeof(float), din);
        }

        if (weight_is_quant) {
          if (in_sub.getFormat() == ml::train::TensorDim::Format::NHWC) {
            // NHWC channel-last quantized convolution:
            // Since physical layout is [owoh, filter_size], we reshape `out` to
            // flat NCHW [1, 1, owoh, filter_size] and write directly,
            // completely bypassing qgemm_scratch and transposes!
            // W8A8 QINT8 output: the int8 GEMM still produces FP32, so route
            // it into the (FP32) qgemm scratch instead; the fused epilogue
            // below adds bias, applies SiLU, and quantizes into `out`.
            const bool out_qint8 =
              out.getDataType() == nntrainer::Tdatatype::QINT8;
            Tensor out_flat = out;
            std::vector<float> w8a8_out_buf;
            if (out_qint8) {
              // Dedicated FP32 GEMM-output buffer (heap-local: pool INFER
              // scratch is not materialized for inference, and a FUNC scratch
              // aliases live activation memory). The epilogue reads it back to
              // quantize into `out`.
              w8a8_out_buf.assign((size_t)owoh * filter_size, 0.f);
              out_flat = Tensor::Map<float>(
                w8a8_out_buf.data(), w8a8_out_buf.size() * sizeof(float),
                TensorDim(1, 1, owoh, filter_size,
                          {ml::train::TensorDim::Format::NCHW,
                           nntrainer::Tdatatype::FP32}));
            } else {
              out_flat.reshape(TensorDim(
                1, 1, owoh, filter_size,
                {ml::train::TensorDim::Format::NCHW, out.getDataType()}));
            }

            const int in_ch_i = (int)in_dim.channel();
            // Q8_0-activation path is an env opt-in only. Q8_0 weights use the
            // proven FP16-activation indirect path (W8A16), dispatched by
            // weight dtype inside convQ4_0Indirect, so they must NOT force q8
            // act here.
            const bool can_q8act =
              (in_ch_i % 32 == 0) &&
              (std::getenv("NNTR_CONV_Q8ACT") != nullptr) &&
              (in_sub.getDataType() == TensorDim::DataType::FP16);
            // Pre-allocated Q8_0 scratch (no per-forward malloc).
            // Take this batch's own slice (q8act_scratch is batch-sized) so
            // concurrent ParallelBatch slices don't race on the q8_0 bytes.
            // Only consumed under ENABLE_FP16 (Q8_0 activation path); guard the
            // declaration too so non-FP16 builds don't trip
            // -Werror=unused-but-set-variable.
#ifdef ENABLE_FP16
            ::nntrainer::block_q8_0 *q8_buf = nullptr;
            if (can_q8act && q8act_scratch != nullptr) {
              Tensor q8act_b = q8act_scratch->getBatchSlice(b, 1);
              q8_buf =
                reinterpret_cast<::nntrainer::block_q8_0 *>(q8act_b.getData());
            }
#endif
            if (is_1x1_s1 && !weight_is_q8) {
#ifdef ENABLE_FP16
              if (can_q8act && q8_buf) {
                // Fused: quantize NHWC activation directly into the
                // block_q8_0x4 SMMLA layout and run the prepacked GEMM. This
                // replaces the two passes (quantize -> block_q8_0, then dot()
                // repacks to x4 with a per-call malloc) with one; output is
                // bit-identical.
                quantize_nhwc_q8_0x4_rows(in_sub.getData<_FP16>(), in_ch_i,
                                          owoh, q8_buf);
                Q8_0_Tensor::dot_prepacked_x4(
                  (unsigned)owoh, (unsigned)in_ch_i, filter_size, q8_buf,
                  filter_kernel.getData(), out_flat.getData<_FP16>(),
                  filter_size);
              } else {
#endif
                Tensor act = in_sub;
                act.reshape(TensorDim(
                  1, 1, owoh, in_dim.channel(),
                  {ml::train::TensorDim::Format::NCHW, in_sub.getDataType()}));
                act.dot(filter_kernel, out_flat, false, false);
#ifdef ENABLE_FP16
              }
#endif
            } else if (NNTR_HAS_Q4_0_INDIRECT_CONV || weight_is_q8) {
              // (Q8_0 dispatches to the portable __ggml Q8_0xQ8_0 indirect
              // GEMMs, which have scalar fallbacks on every ISA -- so Q8_0
              // convs run on x86 too, unlike the ISA-gated Q4_0 kernels.)
              ConvGatherParams geom;
              geom.in_ch = in_ch_i;
              geom.in_h = in_dim.height();
              geom.in_w = in_dim.width();
              geom.k_h = kernel_size[0].get();
              geom.k_w = kernel_size[1].get();
              geom.pad_t = padding[0];
              geom.pad_l = padding[2];
              geom.stride_h = stride[0].get();
              geom.stride_w = stride[1].get();
              geom.dil_h = dilation[0].get();
              geom.dil_w = dilation[1].get();
              geom.out_w = out_dim.width();
#ifdef ENABLE_FP16
              if (can_q8act && q8_buf) {
                const int n_sp = geom.in_h * geom.in_w;
                quantize_nhwc_q8_0_rows(in_sub.getData<_FP16>(), n_sp, in_ch_i,
                                        q8_buf);
                TensorDim q8dim({1, 1, (unsigned)n_sp, (unsigned)in_ch_i},
                                {ml::train::TensorDim::Format::NCHW,
                                 nntrainer::Tdatatype::Q8_0});
                Q8_0_Tensor q8act(q8dim, q8_buf);
                q8act.convQ4_0Indirect(filter_kernel, out_flat, geom);
              } else {
#endif
                geom.is_nhwc = true;
                if (in_sub.getDataType() == nntrainer::Tdatatype::QINT8) {
                  // W8A8: pre-quantized per-tensor int8 activation -- byte
                  // gather + constant-d packing straight into the SMMLA GEMM.
                  __ggml_q8_0_q8_0_indirect_GEMM_i8a(
                    owoh, filter_size,
                    (unsigned int)in_ch_i * kernel_size[0].get() *
                      kernel_size[1].get(),
                    in_sub.getData<int8_t>(), in_sub.getScale<float>()[0], geom,
                    filter_kernel.getData(), filter_size,
                    out_flat.getData<float>(), filter_size);
                } else {
                  in_sub.convQ4_0Indirect(filter_kernel, out_flat, geom);
                }
#ifdef ENABLE_FP16
              }
#endif
            } else {
              throw std::runtime_error(
                "Fallback quantized NHWC conv is not supported (requires "
                "indirect conv on ARM).");
            }
            // W8A8 fused quantize epilogue: bias + SiLU on the FP32 scratch,
            // then per-tensor symmetric int8 quantize into `out` with the
            // fp16-rounded dynamic scale written to the tensor's scale slot.
            // The generic bias/activation sections below are dtype-gated and
            // never touch a QINT8 output.
            if (out_qint8) {
              float *f = out_flat.getData<float>();
              const size_t n_out = (size_t)owoh * filter_size;
              const float *bptr = nullptr;
              if (auto &db = std::get<props::DisableBias>(*layer_impl_props);
                  db.empty() || db.get() == false)
                bptr =
                  context.getWeight(wt_idx[ConvParams::bias]).getData<float>();
              if (bptr) {
                auto &tmq = ThreadManager::Global();
                const unsigned int C = filter_size;
                tmq.parallel_for(0, owoh, [=](size_t p) {
                  float *row = f + p * C;
                  for (unsigned int c = 0; c < C; ++c)
                    row[c] += bptr[c];
                });
              }
              if (auto &actp = std::get<props::FusedActivation>(conv_props);
                  !actp.empty() && actp.get() == ActivationType::ACT_SWISH)
                convApplySwishInplace(f, n_out);
              // amax reduce + quantize
              auto &tmq = ThreadManager::Global();
              const size_t chunk = 4096;
              const size_t nchunk = (n_out + chunk - 1) / chunk;
              std::vector<float> amaxes(nchunk, 0.f);
              float *amax_arr = amaxes.data();
              tmq.parallel_for(0, nchunk, [=](size_t ci) {
                const size_t i0 = ci * chunk;
                const size_t i1 = std::min(i0 + chunk, n_out);
                float am = 0.f;
                for (size_t i = i0; i < i1; ++i) {
                  float a = std::fabs(f[i]);
                  if (a > am)
                    am = a;
                }
                amax_arr[ci] = am;
              });
              float amax = 0.f;
              for (float a : amaxes)
                amax = std::max(amax, a);
              float scale = amax > 0.f ? convRoundScaleFp16(amax / 127.f) : 1.f;
              const float inv = 1.f / scale;
              int8_t *qout = out.getData<int8_t>();
              tmq.parallel_for(0, nchunk, [=](size_t ci) {
                const size_t i0 = ci * chunk;
                const size_t i1 = std::min(i0 + chunk, n_out);
                for (size_t i = i0; i < i1; ++i) {
                  float q = std::round(f[i] * inv);
                  qout[i] = (int8_t)std::max(-128.f, std::min(127.f, q));
                }
              });
              hidden_.getScale<float>()[0] = scale;
            }
          } else {
            // Quantized conv as matmul: act [OH*OW, CRS] . weight [CRS, out_ch]
            // -> [OH*OW, out_ch] -> out [out_ch, OH*OW]. CRS = in_ch*kh*kw.
            // NOTE: col must outlive `act` (act aliases col's storage); here
            // col is a view into the context-owned scratch, so its storage
            // outlives the loop iteration regardless.
            Tensor tmp = qgemm_scratch->getBatchSlice(b, 1);
            tmp.reshape(
              TensorDim(1, 1, owoh, filter_size, in_sub.getTensorType()));
            if (is_1x1_s1) {
              // 1x1 stride-1: im2col is an identity. The raw input is laid out
              // as [in_ch, OH*OW] (NCHW), so transpose to the act layout
              // [OH*OW, CRS] (CRS == in_ch here).
              in_sub.reshape({in_dim.channel(), owoh});
              Tensor act = in_sub.transpose("0:2:1");
              act.dot(filter_kernel, tmp, false, false);
            } else if (NNTR_HAS_Q4_0_INDIRECT_CONV || weight_is_q8) {
              // Quantized 3x3+ indirect: fold im2col gather into the q8_0
              // activation quantization so the activation matrix is never
              // materialized (the FP16 input is gathered on the fly and
              // quantized per tile inside the indirect GEMM). Output tmp is
              // FP16 [OH*OW, out_ch].
              // Q8_0 weights dispatch to the portable __ggml Q8_0xQ8_0 indirect
              // GEMMs (FloatTensor::convQ4_0Indirect ->
              // __ggml_q8_0_q8_0_indirect_GEMM_fp32), which have scalar
              // fallbacks on every ISA -- so Q8_0 convs run on x86 too, unlike
              // the ISA-gated Q4_0 kernels.
              ConvGatherParams geom;
              geom.in_ch = in_dim.channel();
              geom.in_h = in_dim.height();
              geom.in_w = in_dim.width();
              geom.k_h = kernel_size[0].get();
              geom.k_w = kernel_size[1].get();
              geom.pad_t = padding[0];
              geom.pad_l = padding[2];
              geom.stride_h = stride[0].get();
              geom.stride_w = stride[1].get();
              geom.dil_h = dilation[0].get();
              geom.dil_w = dilation[1].get();
              geom.out_w = out_dim.width();
              geom.is_nhwc = false;
              in_sub.convQ4_0Indirect(filter_kernel, tmp, geom);
            } else {
              // Fallback (no fused backend op): materialize im2col into the col
              // scratch, then the standard quant GEMM.
              // build the real kernel geometry (filter is stored as
              // [CRS,out_ch])
              TensorDim kdim(filter_size, in_dim.channel(),
                             kernel_size[0].get(), kernel_size[1].get(),
                             in_sub.getTensorType());
              Tensor col = col_scratch->getBatchSlice(b, 1);
              // im2col reshapes col in place to [OH*OW, CRS] (spatial-major),
              // which is ALREADY the act layout — no transpose (unlike the
              // raw-input 1x1 branch above). Transposing here gives [CRS,
              // OH*OW] and makes the GEMM emit CRS rows into the owoh-row
              // `tmp`, overflowing it whenever CRS > owoh (deep convs) -> heap
              // corruption.
              im2col(in_sub, kdim, padding, stride, dilation, col);
              col.dot(filter_kernel, tmp, false, false);
            }
            // [OH*OW, out_ch] -> [out_ch, OH*OW] written straight into the
            // (memory-planned) output. `tmp` is a separate scratch buffer and
            // `out` is a separate output view, so there is no aliasing.
            out.reshape({filter_size, owoh});
            tmp.transpose("0:2:1", out);
          }
        } else {
          Tensor result = col_scratch->getBatchSlice(b, 1);
          out.reshape({filter_size, owoh});
          im2col(in_sub, filter_dim, padding, stride, dilation, result);
          // filter kernel is (K, CRS), result is (CRS, OH*OW)
          if (out.getFormat() == ml::train::TensorDim::Format::NCHW) {
            filter_kernel.dot(result, out, false, true);
          } else {
            // NHWC: out's physical layout is [OH,OW,C] (channel innermost), so
            // a dot writing [out_ch, OH*OW] NCHW-order would land in the wrong
            // physical cells.
            if (filter_size == 1) {
              // If out_ch == 1, NCHW and NHWC layouts are identical [OH*OW, 1]
              // == [1, OH*OW]. We can compute directly into out (mapped as
              // NCHW-format view) without temporary allocation.
              auto onchw_type = out.getTensorType();
              onchw_type.format = ml::train::TensorDim::Format::NCHW;
              TensorDim odim_nchw({1, 1, 1, owoh}, onchw_type);

              auto fnchw_type = filter_kernel.getTensorType();
              fnchw_type.format = ml::train::TensorDim::Format::NCHW;
              auto cnchw_type = result.getTensorType();
              cnchw_type.format = ml::train::TensorDim::Format::NCHW;
              TensorDim fdim_nchw(
                filter_kernel.batch(), filter_kernel.channel(),
                filter_kernel.height(), filter_kernel.width(), fnchw_type);
              TensorDim cdim_nchw(result.batch(), result.channel(),
                                  result.height(), result.width(), cnchw_type);

              Tensor filt_nchw;
              if (filter_kernel.getDataType() == nntrainer::Tdatatype::FP32) {
                filt_nchw =
                  Tensor::Map<float>(filter_kernel.getData<float>(),
                                     filter_kernel.bytes(), fdim_nchw);
              }
#ifdef ENABLE_FP16
              else {
                filt_nchw =
                  Tensor::Map<_FP16>(filter_kernel.getData<_FP16>(),
                                     filter_kernel.bytes(), fdim_nchw);
              }
#endif

              if (out.getDataType() == nntrainer::Tdatatype::FP32) {
                Tensor col_nchw = Tensor::Map<float>(result.getData<float>(),
                                                     result.bytes(), cdim_nchw);
                Tensor out_mapped = Tensor::Map<float>(out.getData<float>(),
                                                       out.bytes(), odim_nchw);
                filt_nchw.dot(col_nchw, out_mapped, false, true);
              }
#ifdef ENABLE_FP16
              else {
                Tensor col_nchw = Tensor::Map<_FP16>(result.getData<_FP16>(),
                                                     result.bytes(), cdim_nchw);
                Tensor out_mapped = Tensor::Map<_FP16>(out.getData<_FP16>(),
                                                       out.bytes(), odim_nchw);
                filt_nchw.dot(col_nchw, out_mapped, false, true);
              }
#endif
            } else {
              // Standard path for filter_size > 1: Compute into NCHW temporary
              // buffer and scatter.
              auto nchw_type = out.getTensorType();
              nchw_type.format = ml::train::TensorDim::Format::NCHW;
              // nchw_out holds the GEMM result [out_ch, OH*OW] in channel-major
              // (NCHW) order. Shape it [1, 1, out_ch, OH*OW] so width()==OH*OW
              // is the row stride the GEMM writes with (ldc); a
              // [1,out_ch,OH*OW,1] shape would give width()==1 and stride the
              // output wrong.
              TensorDim nchw_dim({1, 1, filter_size, owoh}, nchw_type);
              Tensor nchw_out(nchw_dim, true);
              // filter_kernel (weight [out_ch, CRS]) and result (im2col columns
              // [OH*OW, CRS]) are plain 2D matmul matrices whose image format
              // is irrelevant to this GEMM. dot() derives the contraction axis
              // from the tensor format: their inherited NHWC tag makes it
              // contract over channel() (==1) instead of width() (==CRS) ->
              // zero output. Re-map the same bytes as NCHW-format views (no
              // copy) so the GEMM contracts over CRS==width().
              // getSharedDataTensor can't relabel format (it enforces a match),
              // so use Tensor::Map.
              auto fnchw_type = filter_kernel.getTensorType();
              fnchw_type.format = ml::train::TensorDim::Format::NCHW;
              auto cnchw_type = result.getTensorType();
              cnchw_type.format = ml::train::TensorDim::Format::NCHW;
              TensorDim fdim_nchw(
                filter_kernel.batch(), filter_kernel.channel(),
                filter_kernel.height(), filter_kernel.width(), fnchw_type);
              TensorDim cdim_nchw(result.batch(), result.channel(),
                                  result.height(), result.width(), cnchw_type);

              Tensor filt_nchw;
              if (filter_kernel.getDataType() == nntrainer::Tdatatype::FP32) {
                filt_nchw =
                  Tensor::Map<float>(filter_kernel.getData<float>(),
                                     filter_kernel.bytes(), fdim_nchw);
              }
#ifdef ENABLE_FP16
              else {
                filt_nchw =
                  Tensor::Map<_FP16>(filter_kernel.getData<_FP16>(),
                                     filter_kernel.bytes(), fdim_nchw);
              }
#endif

              if (out.getDataType() == nntrainer::Tdatatype::FP32) {
                Tensor col_nchw = Tensor::Map<float>(result.getData<float>(),
                                                     result.bytes(), cdim_nchw);
                filt_nchw.dot(col_nchw, nchw_out, false, true);
              }
#ifdef ENABLE_FP16
              else {
                Tensor col_nchw = Tensor::Map<_FP16>(result.getData<_FP16>(),
                                                     result.bytes(), cdim_nchw);
                filt_nchw.dot(col_nchw, nchw_out, false, true);
              }
#endif
              if (out.getDataType() == nntrainer::Tdatatype::FP32) {
                const float *s = nchw_out.getData<float>();
                float *d = out.getData<float>();
                for (unsigned int oc = 0; oc < filter_size; ++oc)
                  for (unsigned int r = 0; r < owoh; ++r)
                    d[r * filter_size + oc] = s[oc * owoh + r];
              }
#ifdef ENABLE_FP16
              else if (out.getDataType() == nntrainer::Tdatatype::FP16) {
                const _FP16 *s = nchw_out.getData<_FP16>();
                _FP16 *d = out.getData<_FP16>();
                for (unsigned int oc = 0; oc < filter_size; ++oc)
                  for (unsigned int r = 0; r < owoh; ++r)
                    d[r * filter_size + oc] = s[oc * owoh + r];
              }
#endif
            }
          }
        }
      }
    };

    auto workers = ParallelBatch(forwarding_job, in_dim.batch(), nullptr);

    if (workers.getNumWorkers() > 1) {
      workers.run();
    } else {
      forwarding_job(0, in_dim.batch(), 0, nullptr);
    }

    if (std::getenv("NNTR_NAN_TRACE")) {
      // Clean per-layer nan/magnitude check inside the engine (no OOB risk from
      // env tap-override). Counts finite/nan/inf and max-abs over the REAL
      // output buffer, at FP32 precision (cast FP16 up so values are reliable).
      const unsigned int n = hidden_.size();
      unsigned int n_nan = 0, n_inf = 0, n_fin = 0, n_over = 0;
      float max_abs = 0.0f;
      if (hidden_.getDataType() == nntrainer::Tdatatype::FP16) {
#ifdef ENABLE_FP16
        const _FP16 *d = hidden_.getData<_FP16>();
        for (unsigned int i = 0; i < n; ++i) {
          float v = (float)d[i];
          if (std::isnan(v))
            n_nan++;
          else if (std::isinf(v))
            n_inf++;
          else {
            n_fin++;
            float a = std::fabs(v);
            if (a > max_abs)
              max_abs = a;
            if (a > 65504.0f)
              n_over++;
          }
        }
#endif
      } else {
        const float *d = hidden_.getData<float>();
        for (unsigned int i = 0; i < n; ++i) {
          float v = d[i];
          if (std::isnan(v))
            n_nan++;
          else if (std::isinf(v))
            n_inf++;
          else {
            n_fin++;
            float a = std::fabs(v);
            if (a > max_abs)
              max_abs = a;
            if (a > 65504.0f)
              n_over++;
          }
        }
      }
      std::cerr << "[NANTRACE] " << context.getName() << " n=" << n << " dt="
                << (hidden_.getDataType() == nntrainer::Tdatatype::FP16
                      ? "FP16"
                      : "FP32")
                << " fin=" << n_fin << " nan=" << n_nan << " inf=" << n_inf
                << " >65504=" << n_over << " maxAbs=" << max_abs << "\n"
                << std::flush;
    }

    // Dump one named layer's real output (from inside the untruncated
    // production forward pass, so no separate-model-output/memory-pool
    // aliasing risk) as raw FP32 floats, for external bisection.
    if (const char *dl = std::getenv("NNTR_DUMP_LAYER");
        dl && context.getName() == dl) {
      const char *dp = std::getenv("NNTR_DUMP_PATH");
      if (dp) {
        std::vector<float> buf(hidden_.size());
        if (hidden_.getDataType() == nntrainer::Tdatatype::FP16) {
#ifdef ENABLE_FP16
          const _FP16 *d = hidden_.getData<_FP16>();
          for (unsigned int i = 0; i < hidden_.size(); ++i)
            buf[i] = (float)d[i];
#endif
        } else if (hidden_.getDataType() == nntrainer::Tdatatype::QINT8) {
          const int8_t *q = hidden_.getData<int8_t>();
          const float sc = hidden_.getScale<float>()[0];
          static const bool perch_asym_env =
            std::getenv("NNTR_W8A8_PERCH") != nullptr &&
            std::getenv("NNTR_W8A8_SYM") == nullptr;
          constexpr float kActOff = 0.27846455f;
          for (unsigned int i = 0; i < hidden_.size(); ++i)
            buf[i] = perch_asym_env ? ((float)q[i] + 128.f) * sc - kActOff
                                    : sc * (float)q[i];
        } else {
          const float *d = hidden_.getData<float>();
          std::copy(d, d + hidden_.size(), buf.begin());
        }
        std::ofstream of(dp, std::ios::binary);
        of.write(reinterpret_cast<const char *>(buf.data()),
                 buf.size() * sizeof(float));
      }
    }

    if (!weight_is_quant) {
      filter_kernel.reshape(filter_dim);
    }
  } else {
    // Grouped convolution: split channels into `groups` independent groups.
    const unsigned int ocg = filter_size / groups;      // out ch per group
    const unsigned int icg = in_dim.channel() / groups; // in ch per group
    const unsigned int fh = filter_dim.height(), fw = filter_dim.width();
    const unsigned int owoh = out_dim.width() * out_dim.height();
    const unsigned int ihw = in_dim.height() * in_dim.width();
    TensorDim fdim_g(ocg, icg, fh, fw, filter_dim.getTensorType());

    // W8A8 grouped/depthwise dequantization. finalize() (see the comment at the
    // QINT8 branch above) promises "any other conv dequantizes its input and
    // produces FP32" — but historically only the groups==1 path honored that
    // (see the deq_in_buf block ~2100). A grouped conv (groups>1, including the
    // YOLOv11 depthwise-separable detect head cv3_*_dw) fed a QINT8 activation
    // had no dequant path here, so it silently skipped every FP32/FP16 dispatch
    // and left hidden_ unwritten -> NaN. This is the real root cause that made
    // face YOLOv11 W8A8 fail while video YOLOv7 (pure 1x1 pointwise detect
    // head, always groups==1) succeeded. Dequantize the QINT8 input once into a
    // heap-local FP32 buffer and route every path below through it; finalize's
    // output-dtype rule above already forces the grouped output to FP32, so the
    // graph edge stays FP32 and the next layer sees a clean FP32 boundary.
    Tensor input_use = input_;
    std::vector<float> grouped_deq_buf;
    if (input_.getDataType() == nntrainer::Tdatatype::QINT8) {
      const int8_t *q = input_.getData<int8_t>();
      const float sc = input_.getScale<float>()[0];
      const size_t n_in = input_.size();
      // Padded well past any plausible SIMD width (diagnostic 2026-08-11:
      // testing whether the depthwise NEON paths over-read past a tightly
      // sized external buffer when fed a Tensor::Map-wrapped input, unlike
      // a normal pool-allocated tensor).
      grouped_deq_buf.assign(n_in + 4096, 0.f);
      float *fp = grouped_deq_buf.data();
      auto &tmd = ThreadManager::Global();
      const size_t chunk = 65536;
      // Same asymmetric-encoding fix as the groups==1 dequant above (and
      // float_tensor.cpp's QINT8 copyData) -- perch_mode's producer encodes
      // x=(q+128)*s-kActOff by default, not q*s.
      static const bool perch_asym_env =
        std::getenv("NNTR_W8A8_PERCH") != nullptr &&
        std::getenv("NNTR_W8A8_SYM") == nullptr;
      if (perch_asym_env) {
        constexpr float kActOff = 0.27846455f;
        tmd.parallel_for(0, (n_in + chunk - 1) / chunk, [=](size_t ci) {
          const size_t i0 = ci * chunk, i1 = std::min(i0 + chunk, n_in);
          for (size_t i = i0; i < i1; ++i)
            fp[i] = ((float)q[i] + 128.f) * sc - kActOff;
        });
      } else {
        tmd.parallel_for(0, (n_in + chunk - 1) / chunk, [=](size_t ci) {
          const size_t i0 = ci * chunk, i1 = std::min(i0 + chunk, n_in);
          for (size_t i = i0; i < i1; ++i)
            fp[i] = sc * (float)q[i];
        });
      }
      TensorDim din = in_dim;
      din.setDataType(nntrainer::Tdatatype::FP32);
      input_use =
        Tensor::Map<float>(grouped_deq_buf.data(), n_in * sizeof(float), din);

      if (const char *dl = std::getenv("NNTR_GDW_DEBUG");
          dl && context.getName() == dl) {
        float dmin = 1e30f, dmax = -1e30f;
        for (float v : grouped_deq_buf) {
          if (v < dmin)
            dmin = v;
          if (v > dmax)
            dmax = v;
        }
        std::cerr << "[GDWDBG] " << context.getName() << " deq_range=[" << dmin
                  << "," << dmax << "]"
                  << " a_scale=" << sc << " groups=" << groups << " icg=" << icg
                  << " ocg=" << ocg
                  << " filter_dt=" << (int)filter_kernel.getDataType() << "\n"
                  << std::flush;
      }
    }

    const bool is_true_depthwise = ocg == 1 && icg == 1;
    const bool nhwc_depthwise =
      is_true_depthwise &&
      in_dim.getFormat() == ml::train::TensorDim::Format::NHWC;

    if (nhwc_depthwise) {
      // NHWC depthwise convolution: input/output are channel-last, but the
      // backend depthwise ops (depthwise_conv2d_fp32/fp16) and the generic
      // grouped fallback below are all NCHW-planar (channel-major). Handle the
      // channel-last layout inline here. Filter keeps the standard
      // [out_ch, 1, fh, fw] layout, so channel c tap (kh,kw) is at
      // filt[c*fh*fw + kh*fw + kw]. Accumulate in float for parity with the
      // NCHW paths regardless of activation precision.
      const unsigned int B = in_dim.batch();
      const unsigned int C = filter_size; // channels (== groups)
      const int IH = static_cast<int>(in_dim.height());
      const int IW = static_cast<int>(in_dim.width());
      const int OH = static_cast<int>(out_dim.height());
      const int OW = static_cast<int>(out_dim.width());
      const int sh = static_cast<int>(stride[0].get());
      const int sw = static_cast<int>(stride[1].get());
      const int ph = static_cast<int>(padding[0]);
      const int pw = static_cast<int>(padding[2]);
      const int dh = static_cast<int>(dilation[0].get());
      const int dw_ = static_cast<int>(dilation[1].get());
      const unsigned int fhfw = fh * fw;
      // The filter dtype follows the requested weight dtype
      // (context.getWeightDataType(), see finalize()), which may be FP16 for
      // FP16-FP16 / FP16-FP32 modes while the activation is FP16. Read the
      // filter through the matching type so we do not reinterpret half bytes
      // as float.
      const auto filt_dt = filter_kernel.getDataType();

      // Env-gated INT8 depthwise (NNTR_W8A8_DW=1, opt-in): channel-vectorized
      // (groups of 8) int8 widen-multiply-accumulate over the tap window,
      // int32 accumulation, one dequantize+scale pass at the end. Symmetric
      // activation quantization only (no zero-point correction term needed
      // since padding contributes exactly 0 either way) -- covers the
      // landmark(PFLD)/feature(ResNet+Mona) W8A8 paths, which already force
      // NNTR_W8A8_SYM=1 (see face_pipeline.cpp); falls through to the
      // existing FP32 kernels below otherwise (detector's asymmetric C2PSA
      // "pe" depthwise, or a channel count not a multiple of 8).
#ifdef __ARM_NEON
      // NOTE: this is a fresh check, not the QINT8-input dequant block's
      // same-named local above -- that one is scoped to its own `if`, and
      // most depthwise convs here receive an already-FP32 input (this
      // graph's convs don't chain QINT8 across layer boundaries).
      static const bool dw_perch_asym_env =
        std::getenv("NNTR_W8A8_PERCH") != nullptr &&
        std::getenv("NNTR_W8A8_SYM") == nullptr;
      const bool dw_int8 =
        std::getenv("NNTR_W8A8") != nullptr &&
        std::getenv("NNTR_W8A8_DW") != nullptr && !dw_perch_asym_env &&
        (C % 8) == 0 && input_use.getDataType() == nntrainer::Tdatatype::FP32 &&
        filt_dt == nntrainer::Tdatatype::FP32;
      if (dw_int8) {
        const DWQ8Weight &Wq =
          getDWQ8Weight(filter_kernel.getData<float>(), C, fh, fw);

        // Quantize the FP32 activation to int8 once, per-tensor symmetric
        // (amax-based) -- same convention as the groups==1 perch_mode path's
        // auto-quantize-FP32-input branch, just without the asymmetric
        // kActOff offset.
        const float *fin = input_use.getData<float>();
        const size_t n_in = input_use.size();
        auto &tmq = ThreadManager::Global();
        const float amax = nntr_absmax_f32(fin, n_in);
        const float act_scale = amax > 0.f ? amax / 127.f : 1.f;
        const float act_inv = 1.f / act_scale;
        static thread_local std::vector<int8_t> dw_in_i8;
        dw_in_i8.resize(n_in);
        int8_t *in_i8 = dw_in_i8.data();
        const size_t qchunk = 1 << 15;
        const size_t nqc = (n_in + qchunk - 1) / qchunk;
        tmq.parallel_for(0, nqc, [=](size_t ci) {
          const size_t i0 = ci * qchunk, i1 = std::min(i0 + qchunk, n_in);
          for (size_t i = i0; i < i1; ++i) {
            const float q = std::round(fin[i] * act_inv);
            in_i8[i] = (int8_t)std::max(-127.f, std::min(127.f, q));
          }
        });

        float *outb0 = hidden_.getData<float>();
        const int8_t *w_qs = Wq.qs.data();
        const float *w_scale = Wq.scale.data();
        const unsigned int Cg8 = C / 8;
        tmq.parallel_for(0, (size_t)B * Cg8, [&](size_t bcg) {
          const unsigned int b = (unsigned int)(bcg / Cg8);
          const unsigned int cg = (unsigned int)(bcg % Cg8);
          const unsigned int c0 = cg * 8;
          const int8_t *inb = in_i8 + (size_t)b * IH * IW * C;
          float *outb = outb0 + (size_t)b * OH * OW * C;
          for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
              int32x4_t acc_lo = vdupq_n_s32(0);
              int32x4_t acc_hi = vdupq_n_s32(0);
              for (unsigned int kh = 0; kh < fh; ++kh) {
                const int ih = oh * sh - ph + (int)kh * dh;
                if (ih < 0 || ih >= IH)
                  continue;
                const int8_t *row = inb + (size_t)ih * IW * C;
                for (unsigned int kw = 0; kw < fw; ++kw) {
                  const int iw = ow * sw - pw + (int)kw * dw_;
                  if (iw < 0 || iw >= IW)
                    continue;
                  const unsigned int tap = kh * fw + kw;
                  int8x8_t vin = vld1_s8(row + (size_t)iw * C + c0);
                  int8x8_t vw = vld1_s8(w_qs + (size_t)tap * C + c0);
                  int16x8_t prod = vmull_s8(vin, vw);
                  acc_lo = vaddw_s16(acc_lo, vget_low_s16(prod));
                  acc_hi = vaddw_s16(acc_hi, vget_high_s16(prod));
                }
              }
              float32x4_t f_lo = vmulq_n_f32(vcvtq_f32_s32(acc_lo), act_scale);
              float32x4_t f_hi = vmulq_n_f32(vcvtq_f32_s32(acc_hi), act_scale);
              f_lo = vmulq_f32(f_lo, vld1q_f32(w_scale + c0));
              f_hi = vmulq_f32(f_hi, vld1q_f32(w_scale + c0 + 4));
              const size_t on = (size_t)(oh * OW + ow) * C + c0;
              vst1q_f32(outb + on, f_lo);
              vst1q_f32(outb + on + 4, f_hi);
            }
          }
        });
        goto depthwise_done;
      }
#endif

      // Env-gated vectorized NHWC depthwise (default ON, NNTR_DW_NHWC_SCALAR=1
      // restores the per-output-element scalar `run` below for A/B / rollback).
      // Parallelize across (batch*channels) and vectorize across 4 output
      // columns (ow) with FP32 accumulators, widening FP16 input + FP32 kernel
      // per MAC (same accumulate-in-float semantics as `run`, so output is
      // bit-identical up to the final FP16 narrow). Beats the scalar
      // per-(oh,ow) loop for the FastViT 7x7 stride-1 depthwise `mlp_conv/dw`
      // blocks, which the profiler showed at ~4.5ms each (the scalar path
      // vectorizes channels but iterates 49 taps per output point; this
      // vectorizes the output points so each tap's 4-ow FMA is one
      // instruction). Env-gated CHANNEL-vectorized NHWC depthwise
      // (NNTR_DW_NHWC_CHAN=1): groups of 4 consecutive channels at once. NHWC
      // stores channels contiguously, so the 4 channels' input taps at a given
      // spatial pos are a single contiguous FP16x4 load (vld1_f16) -> vcvt to
      // FP32x4, instead of the ow-vectorized `dw_vec` below which gathers 4
      // C-strided ow taps. For the FastViT s0b1 mlp_conv/dw 7x7 @ 80x80 (6400
      // spatials * 64 ch) this is the #1 keyword cost (3.3ms) and is
      // memory-bound: channel-vector turns 49 strided gathers into 49
      // contiguous loads, ~4x less traffic. Accumulate-in-float per channel (4
      // separate FP32 accumulators, FMA'd with that channel's FP32 weight
      // lane), so bit-identical to scalar up to the final FP16 narrow. Default
      // ON (opt-out NNTR_DW_NHWC_CHAN=0); restricted to C%4==0 + stride1 +
      // dilation1 + FP32 weights. Falls through to `dw_vec` (ow-vectorized)
      // otherwise.
#if defined(__ARM_NEON) && defined(__aarch64__)
      const bool dw_chan = (C % 4) == 0 && stride[0].get() == 1 &&
                           stride[1].get() == 1 && dilation[0].get() == 1 &&
                           dilation[1].get() == 1 &&
                           std::getenv("NNTR_DW_NHWC_CHAN") == nullptr &&
                           filt_dt == nntrainer::Tdatatype::FP32;
      if (dw_chan) {
        auto run_chan = [&]<typename T>(const T *in, T *out,
                                        const float *filt) {
          const unsigned int Cg = C / 4; // groups of 4 channels
          auto &tm = ThreadManager::Global();
          tm.parallel_for(0, (size_t)B * Cg, [&](size_t bcg) {
            const unsigned int b = (unsigned int)(bcg / Cg);
            const unsigned int cg = (unsigned int)(bcg % Cg);
            const unsigned int c0 = cg * 4;
            const T *inb = in + (size_t)b * IH * IW * C;
            T *outb = out + (size_t)b * OH * OW * C;
            for (int oh = 0; oh < OH; ++oh) {
              for (int ow = 0; ow < OW; ++ow) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (unsigned int kh = 0; kh < fh; ++kh) {
                  const int ih = oh * sh - ph + (int)kh * dh;
                  if (ih < 0 || ih >= IH)
                    continue;
                  const T *row = inb + (size_t)ih * IW * C;
                  for (unsigned int kw = 0; kw < fw; ++kw) {
                    const int iw = ow * sw - pw + (int)kw * dw_;
                    if (iw < 0 || iw >= IW)
                      continue;
                    // 4 contiguous channels at (ih,iw): NHWC -> one contiguous
                    // load (FP16x4 widened to FP32x4, or FP32x4 directly).
                    float32x4_t vin;
                    if constexpr (std::is_same_v<T, _FP16>) {
                      __fp16 h4[4];
                      vst1_f16(h4, vld1_f16(reinterpret_cast<const __fp16 *>(
                                     row + (size_t)iw * C + c0)));
                      float vals[4] = {(float)h4[0], (float)h4[1], (float)h4[2],
                                       (float)h4[3]};
                      vin = vld1q_f32(vals);
                    } else {
                      vin = vld1q_f32(row + (size_t)iw * C + c0);
                    }
                    // Gather the 4 channels' filter for this tap
                    // (strided: filt[c*fh*fw + tap], c=c0..c0+3).
                    const unsigned int tap = kh * fw + kw;
                    float vw_arr[4] = {
                      (float)filt[(size_t)c0 * fhfw + tap],
                      (float)filt[(size_t)(c0 + 1) * fhfw + tap],
                      (float)filt[(size_t)(c0 + 2) * fhfw + tap],
                      (float)filt[(size_t)(c0 + 3) * fhfw + tap]};
                    float32x4_t vw = vld1q_f32(vw_arr);
                    acc = vfmaq_f32(acc, vin, vw);
                  }
                }
                // 4 contiguous output channels at (oh,ow): NHWC scatter is a
                // contiguous store.
                const size_t on = (size_t)(oh * OW + ow) * C + c0;
                if constexpr (std::is_same_v<T, _FP16>) {
                  __fp16 tmp[4];
                  vst1_f16(tmp, vcvt_f16_f32(acc));
                  for (int v = 0; v < 4; ++v)
                    outb[on + v] = (T)tmp[v];
                } else {
                  vst1q_f32(outb + on, acc);
                }
              }
            }
          });
        };
#ifdef ENABLE_FP16
        if (input_use.getDataType() == nntrainer::Tdatatype::FP16)
          run_chan(input_use.getData<_FP16>(), hidden_.getData<_FP16>(),
                   filter_kernel.getData<float>());
        else
#endif
          run_chan(input_use.getData<float>(), hidden_.getData<float>(),
                   filter_kernel.getData<float>());
        goto depthwise_done;
      }

      const bool dw_vec = stride[1].get() == 1 && dilation[1].get() == 1 &&
                          OW >= 4 &&
                          std::getenv("NNTR_DW_NHWC_SCALAR") == nullptr &&
                          filt_dt == nntrainer::Tdatatype::FP32;
      if (dw_vec) {
        auto run_vec = [&]<typename T>(const T *in, T *out, const float *filt) {
          auto &tm = ThreadManager::Global();
          tm.parallel_for(0, (size_t)B * C, [&](size_t bc) {
            const unsigned int b = (unsigned int)(bc / C);
            const unsigned int c = (unsigned int)(bc % C);
            const T *inc = in + (size_t)b * IH * IW * C + (size_t)c;
            T *outc = out + (size_t)b * OH * OW * C + (size_t)c;
            const float *fkc = filt + (size_t)c * fhfw;
            for (int oh = 0; oh < OH; ++oh) {
              int ow = 0;
              for (; ow + 3 < OW; ow += 4) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (unsigned int kh = 0; kh < fh; ++kh) {
                  const int ih = oh * sh - ph + (int)kh * dh;
                  if (ih < 0 || ih >= IH)
                    continue;
                  const T *row = inc + (size_t)ih * IW * C;
                  for (unsigned int kw = 0; kw < fw; ++kw) {
                    const int iw0 = ow - pw + (int)kw * dw_;
                    const float kf = fkc[kh * fw + kw];
                    if (iw0 >= 0 && iw0 + 3 < IW) {
                      // 4 taps at channel c, spatial taps iw0..iw0+3 are
                      // C-strided apart (NHWC: channel is innermost). Gather
                      // them into an FP32x4 (no contiguous FP16 load -- the 4
                      // values live C elements apart), then FMA the FP32
                      // kernel.
                      float vals[4] = {(float)row[(size_t)iw0 * C],
                                       (float)row[(size_t)(iw0 + 1) * C],
                                       (float)row[(size_t)(iw0 + 2) * C],
                                       (float)row[(size_t)(iw0 + 3) * C]};
                      float32x4_t vid = vld1q_f32(vals);
                      acc = vmlaq_n_f32(acc, vid, kf);
                    } else {
                      float arr[4] = {0, 0, 0, 0};
                      for (int v = 0; v < 4; ++v) {
                        int idx = iw0 + v;
                        if (idx >= 0 && idx < IW)
                          arr[v] = (float)row[(size_t)idx * C];
                      }
                      acc = vmlaq_n_f32(acc, vld1q_f32(arr), kf);
                    }
                  }
                }
                if constexpr (std::is_same_v<T, _FP16>) {
                  // 4 outputs are C-strided apart (NHWC); scatter-store.
                  __fp16 tmp[4];
                  vst1_f16(tmp, vcvt_f16_f32(acc));
                  for (int v = 0; v < 4; ++v)
                    outc[(size_t)(oh * OW + ow + v) * C] = (T)tmp[v];
                } else {
                  float tmp[4];
                  vst1q_f32(tmp, acc);
                  for (int v = 0; v < 4; ++v)
                    outc[(size_t)(oh * OW + ow + v) * C] = (T)tmp[v];
                }
              }
              for (; ow < OW; ++ow) {
                float acc = 0.0f;
                for (unsigned int kh = 0; kh < fh; ++kh) {
                  const int ih = oh * sh - ph + (int)kh * dh;
                  if (ih < 0 || ih >= IH)
                    continue;
                  for (unsigned int kw = 0; kw < fw; ++kw) {
                    const int iw = ow - pw + (int)kw * dw_;
                    if (iw < 0 || iw >= IW)
                      continue;
                    acc += (float)inc[((size_t)ih * IW + iw) * C] *
                           fkc[kh * fw + kw];
                  }
                }
                outc[(size_t)(oh * OW + ow) * C] = (T)acc;
              }
            }
          });
        };
#ifdef ENABLE_FP16
        if (input_use.getDataType() == nntrainer::Tdatatype::FP16)
          run_vec(input_use.getData<_FP16>(), hidden_.getData<_FP16>(),
                  filter_kernel.getData<float>());
        else
#endif
          run_vec(input_use.getData<float>(), hidden_.getData<float>(),
                  filter_kernel.getData<float>());
        goto depthwise_done;
      }
#endif // __ARM_NEON

      auto run = [&]<typename T, typename F>(const T *in, T *out,
                                             const F *filt) {
        std::vector<float> acc(C);
        for (unsigned int b = 0; b < B; ++b) {
          const T *inb = in + static_cast<size_t>(b) * IH * IW * C;
          T *outb = out + static_cast<size_t>(b) * OH * OW * C;
          for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
              std::fill(acc.begin(), acc.end(), 0.0f);
              for (unsigned int kh = 0; kh < fh; ++kh) {
                const int ih = oh * sh - ph + static_cast<int>(kh) * dh;
                if (ih < 0 || ih >= IH)
                  continue;
                for (unsigned int kw = 0; kw < fw; ++kw) {
                  const int iw = ow * sw - pw + static_cast<int>(kw) * dw_;
                  if (iw < 0 || iw >= IW)
                    continue;
                  const T *id = inb + (static_cast<size_t>(ih) * IW + iw) * C;
                  const F *fk = filt + kh * fw + kw;
                  unsigned int c = 0;
#if defined(__ARM_NEON) && defined(ENABLE_FP16)
                  const unsigned int C_aligned = (C / 4) * 4;
                  for (; c < C_aligned; c += 4) {
                    float32x4_t v_id;
                    if constexpr (std::is_same_v<T, _FP16>) {
                      float16x4_t v_id_h =
                        vld1_f16(reinterpret_cast<const __fp16 *>(id + c));
                      v_id = vcvt_f32_f16(v_id_h);
                    } else {
                      v_id = vld1q_f32(reinterpret_cast<const float *>(id + c));
                    }

                    float32x4_t v_fk = {static_cast<float>(fk[c * fhfw]),
                                        static_cast<float>(fk[(c + 1) * fhfw]),
                                        static_cast<float>(fk[(c + 2) * fhfw]),
                                        static_cast<float>(fk[(c + 3) * fhfw])};

                    float32x4_t v_acc = vld1q_f32(&acc[c]);
                    v_acc = vmlaq_f32(v_acc, v_id, v_fk);
                    vst1q_f32(&acc[c], v_acc);
                  }
#elif defined(__ARM_NEON)
                  const unsigned int C_aligned_f32 = (C / 4) * 4;
                  for (; c < C_aligned_f32; c += 4) {
                    float32x4_t v_id =
                      vld1q_f32(reinterpret_cast<const float *>(id + c));
                    float32x4_t v_fk = {static_cast<float>(fk[c * fhfw]),
                                        static_cast<float>(fk[(c + 1) * fhfw]),
                                        static_cast<float>(fk[(c + 2) * fhfw]),
                                        static_cast<float>(fk[(c + 3) * fhfw])};
                    float32x4_t v_acc = vld1q_f32(&acc[c]);
                    v_acc = vmlaq_f32(v_acc, v_id, v_fk);
                    vst1q_f32(&acc[c], v_acc);
                  }
#endif
                  for (; c < C; ++c)
                    acc[c] +=
                      static_cast<float>(id[c]) *
                      static_cast<float>(fk[static_cast<size_t>(c) * fhfw]);
                }
              }
              T *od = outb + (static_cast<size_t>(oh) * OW + ow) * C;
              unsigned int c = 0;
#if defined(__ARM_NEON) && defined(ENABLE_FP16)
              if constexpr (std::is_same_v<T, _FP16>) {
                const unsigned int C_aligned = (C / 8) * 8;
                for (; c < C_aligned; c += 8) {
                  float32x4_t v_acc0 = vld1q_f32(&acc[c]);
                  float32x4_t v_acc1 = vld1q_f32(&acc[c + 4]);
                  float16x4_t v_out0 = vcvt_f16_f32(v_acc0);
                  float16x4_t v_out1 = vcvt_f16_f32(v_acc1);
                  float16x8_t v_res = vcombine_f16(v_out0, v_out1);
                  vst1q_f16(reinterpret_cast<__fp16 *>(od + c), v_res);
                }
              } else
#endif
#if defined(__ARM_NEON)
              {
                const unsigned int C_aligned_f32 = (C / 4) * 4;
                for (; c < C_aligned_f32; c += 4) {
                  float32x4_t v_acc = vld1q_f32(&acc[c]);
                  vst1q_f32(reinterpret_cast<float *>(od + c), v_acc);
                }
              }
#endif
              for (; c < C; ++c)
                od[c] = static_cast<T>(acc[c]);
            }
          }
        }
      };

      if (input_use.getDataType() == nntrainer::Tdatatype::FP32) {
        if (filt_dt == nntrainer::Tdatatype::FP32)
          run(input_use.getData<float>(), hidden_.getData<float>(),
              filter_kernel.getData<float>());
#ifdef ENABLE_FP16
        else
          run(input_use.getData<float>(), hidden_.getData<float>(),
              filter_kernel.getData<_FP16>());
#endif
      }
#ifdef ENABLE_FP16
      else {
        if (filt_dt == nntrainer::Tdatatype::FP32)
          run(input_use.getData<_FP16>(), hidden_.getData<_FP16>(),
              filter_kernel.getData<float>());
        else
          run(input_use.getData<_FP16>(), hidden_.getData<_FP16>(),
              filter_kernel.getData<_FP16>());
      }
#endif
    } else if (is_true_depthwise &&
               input_use.getDataType() == nntrainer::Tdatatype::FP32) {
      // True depthwise (groups == channels): delegate to the CPU backend op so
      // the optimised kernel lives in the backend, not in the layer.
      nntrainer::getComputeOps()->depthwise_conv2d_fp32(
        input_use.getData<float>(), filter_kernel.getData<float>(),
        hidden_.getData<float>(), in_dim.batch(), filter_size, in_dim.height(),
        in_dim.width(), out_dim.height(), out_dim.width(), fh, fw,
        stride[0].get(), stride[1].get(), padding[0], padding[2],
        dilation[0].get(), dilation[1].get());
#ifdef ENABLE_FP16
    } else if (is_true_depthwise &&
               input_use.getDataType() == nntrainer::Tdatatype::FP16 &&
               hidden_.getDataType() == nntrainer::Tdatatype::FP16 &&
               filter_kernel.getDataType() == nntrainer::Tdatatype::FP32 &&
               fh == 3 && fw == 3 && dilation[0].get() == 1 &&
               stride[1].get() == 1) {
      // FP16-activation depthwise: weights are never Q4_0 for groups>1 and stay
      // FP32 (BN-folded), so this is FP16 input/output x FP32 kernel. Keep it
      // on the tight channel-parallel direct-loop kernel instead of falling
      // into the generic grouped else-branch (per-channel im2col + FP16 GEMV),
      // which is ~2x slower for many-channel depthwise convs.
      // Restricted to 3x3 stride1 dilation1 -- the only case the ARM
      // depthwise_conv2d_fp16 NEON kernel optimizes (see
      // arm_compute_backend.cpp). For other kernels (e.g. FastViT 7x7
      // mlp_conv/dw) the backend is scalar, so fall through to the NHWC
      // grouped-conv GEMM path below (ocg=1), which is faster than the scalar
      // depthwise.
      nntrainer::getComputeOps()->depthwise_conv2d_fp16(
        input_use.getData<_FP16>(), filter_kernel.getData<float>(),
        hidden_.getData<_FP16>(), in_dim.batch(), filter_size, in_dim.height(),
        in_dim.width(), out_dim.height(), out_dim.width(), fh, fw,
        stride[0].get(), stride[1].get(), padding[0], padding[2],
        dilation[0].get(), dilation[1].get());
#endif
    } else if (in_dim.getFormat() == ml::train::TensorDim::Format::NHWC) {
      // NHWC grouped conv (groups>1, including out_ch>in_ch cases like the
      // FastViT downsample "large+small" fused conv, ocg>1). The generic
      // else-branch below slices channels with NCHW-stride offsets
      // (g*icg*ihw) which is WRONG for NHWC (channel is innermost, stride 1)
      // and reads garbage. Implement the grouped conv directly with correct
      // NHWC indexing: in[b][ih][iw][c], out[b][oh][ow][oc], filter
      // [oc, icg, fh, fw] where oc = g*ocg + o.
      const unsigned int B = in_dim.batch();
      const unsigned int IH = in_dim.height();
      const unsigned int IW = in_dim.width();
      const unsigned int OH = out_dim.height();
      const unsigned int OW = out_dim.width();
      const unsigned int sh = stride[0].get();
      const unsigned int sw = stride[1].get();
      const int ph = padding[0];
      const int pw = padding[2];
      const unsigned int dh = dilation[0].get();
      const unsigned int dw_ = dilation[1].get();
      const auto filt_dt = filter_kernel.getDataType();
      const auto in_dt = input_use.getDataType();
      const auto out_dt = hidden_.getDataType();
      auto run_g = [&]<typename T, typename F>(const T *in, T *out,
                                               const F *filt) {
        // Env-gated NHWC grouped-conv GEMM (NNTR_CONV_GEMM_GROUPS=1): replace
        // the per-output-element scalar MAC loop with, per group, an im2col
        // gather of that group's icg channels into an FP32 column matrix
        // [icg*fh*fw (=K), OH*OW (=N)] and a single sgemm
        // filt[ocg, K] x col[K, N] -> out[ocg, N], scattered into NHWC output.
        // The gather dequantizes FP16 input to FP32 and the result is narrowed
        // to the output dtype, so the math matches the scalar path
        // (accumulate-in-float). This beats the scalar loop for the FastViT
        // 7x7 stride-2 downsample convs (ocg=2, icg=1, M up to 1600), which
        // the profiler showed dominating keyword backbone latency.
        // Default ON (env opt-out: NNTR_CONV_GEMM_GROUPS=0 restores the scalar
        // loop for A/B / rollback). Gated to FP32 weights (F==float) + K>1 +
        // the gemoetry where im2col+sgemm pays off (icg*fh*fw > 1). Env-gated
        // fused direct grouped conv (NNTR_CONV_FUSED_GROUPS=1): for icg==1 (the
        // FastViT downsample down0 7x7 s=2 convs: groups=in_ch,
        // ocg=out_ch/in_ch e.g. 2), skip the im2col col-buffer materialization
        // the GEMM path does (a 49xOH*OW FP32 slab per group = ~20MB for
        // s1_down) and instead fuse gather+FMA: vectorize 4 output columns
        // (ow), loop the ocg output rows internally sharing the same 49 input
        // taps. Each FP16 tap is read once and FMA'd into ocg FP32 accumulators
        // (one per output row), narrowed at the end -- same accumulate-in-float
        // semantics as the scalar `run_g` loop, so bit-identical up to the
        // final narrow. The GEMM path's win was N-vectorization via sgemm, but
        // it paid for it with the col gather+write; this keeps the
        // N-vectorization and drops the col. Default ON (env opt-out:
        // NNTR_CONV_FUSED_GROUPS=0 -> GEMM/scalar). Restricted to FP32 weights
        // (F==float) + icg==1 + OW>=4.
        const char *__fused_env = std::getenv("NNTR_CONV_FUSED_GROUPS");
#if defined(__ARM_NEON) && defined(__aarch64__)
        const bool use_fused =
          (__fused_env == nullptr || std::string(__fused_env) != "0") &&
          icg == 1 && ocg >= 1 && fh * fw >= 1 && OW >= 4 &&
          std::is_same_v<F, float>;
        if (use_fused) {
          const unsigned int K = icg * fh * fw; // == fh*fw (icg==1)
          const unsigned int N = OH * OW;
          const unsigned int Cin = in_dim.channel(); // == groups (icg==1)
          auto &tmg = ThreadManager::Global();
          tmg.parallel_for(0, (size_t)B * groups, [&](size_t bg) {
            const unsigned int b = (unsigned int)(bg / groups);
            const unsigned int g = (unsigned int)(bg % groups);
            const T *inb = in + (size_t)b * IH * IW * Cin;
            T *outb = out + (size_t)b * OH * OW * filter_size;
            const F *filt_g = filt + (size_t)g * ocg * K; // [ocg, K]
            const unsigned int oc_base = g * ocg;
            for (unsigned int oh = 0; oh < OH; ++oh) {
              unsigned int ow = 0;
              for (; ow + 3 < OW; ow += 4) {
                // ocg FP32x4 accumulators (one per output row, 4 ow lanes).
                std::array<float32x4_t, 4> acc;
                for (unsigned int o = 0; o < ocg; ++o)
                  acc[o] = vdupq_n_f32(0.0f);
                for (unsigned int kh = 0; kh < fh; ++kh) {
                  const int ih = (int)(oh * sh) - ph + (int)kh * dh;
                  if (ih < 0 || ih >= (int)IH)
                    continue;
                  const T *row = inb + (size_t)ih * IW * Cin;
                  for (unsigned int kw = 0; kw < fw; ++kw) {
                    // 4 ow taps: spatial x = (ow+v)*sw - pw + kw*dw_, C-strided
                    // apart (NHWC, icg==1 -> channel offset = g). Gather into
                    // FP32x4 (no contiguous load -- sw may be 2 and taps are
                    // C-strided regardless).
                    float vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int v = 0; v < 4; ++v) {
                      const int iw = (int)((ow + v) * sw) - pw + (int)kw * dw_;
                      if (iw >= 0 && iw < (int)IW)
                        vals[v] = (float)row[(size_t)iw * Cin + g];
                    }
                    const float32x4_t vid = vld1q_f32(vals);
                    const unsigned int kbase = (kh * fw + kw) * icg;
                    for (unsigned int o = 0; o < ocg; ++o) {
                      const float kf = (float)filt_g[o * K + kbase];
                      acc[o] = vmlaq_n_f32(acc[o], vid, kf);
                    }
                  }
                }
                // Narrow + scatter: ocg rows x 4 ow, NHWC (C-strided store).
                if constexpr (std::is_same_v<T, _FP16>) {
                  for (unsigned int o = 0; o < ocg; ++o) {
                    __fp16 tmp[4];
                    vst1_f16(tmp, vcvt_f16_f32(acc[o]));
                    const unsigned int oc = oc_base + o;
                    for (int v = 0; v < 4; ++v)
                      outb[(size_t)(oh * OW + ow + v) * filter_size + oc] =
                        (T)tmp[v];
                  }
                } else {
                  for (unsigned int o = 0; o < ocg; ++o) {
                    float tmp[4];
                    vst1q_f32(tmp, acc[o]);
                    const unsigned int oc = oc_base + o;
                    for (int v = 0; v < 4; ++v)
                      outb[(size_t)(oh * OW + ow + v) * filter_size + oc] =
                        (T)tmp[v];
                  }
                }
              }
              // Scalar tail (ow remainder) -- same accumulate-in-float order.
              for (; ow < OW; ++ow) {
                float accs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (unsigned int kh = 0; kh < fh; ++kh) {
                  const int ih = (int)(oh * sh) - ph + (int)kh * dh;
                  if (ih < 0 || ih >= (int)IH)
                    continue;
                  const T *row = inb + (size_t)ih * IW * Cin;
                  for (unsigned int kw = 0; kw < fw; ++kw) {
                    const int iw = (int)(ow * sw) - pw + (int)kw * dw_;
                    if (iw < 0 || iw >= (int)IW)
                      continue;
                    const float idv = (float)row[(size_t)iw * Cin + g];
                    const unsigned int kbase = (kh * fw + kw) * icg;
                    for (unsigned int o = 0; o < ocg; ++o)
                      accs[o] += idv * (float)filt_g[o * K + kbase];
                  }
                }
                for (unsigned int o = 0; o < ocg; ++o)
                  outb[(size_t)(oh * OW + ow) * filter_size + oc_base + o] =
                    (T)accs[o];
              }
            }
          });
          return;
        }
#endif
        const char *__gemm_env = std::getenv("NNTR_CONV_GEMM_GROUPS");
        const bool use_gemm =
          (__gemm_env == nullptr || std::string(__gemm_env) != "0") &&
          icg * fh * fw > 1 && ocg >= 1 && std::is_same_v<F, float>;
        if (use_gemm) {
          const unsigned int K = icg * fh * fw;
          const unsigned int N = OH * OW;
          auto &tmg = ThreadManager::Global();
          // Per-thread col tile + FP32 row scratch. col is [K, N] row-major.
          tmg.parallel_for(0, (size_t)B * groups, [&](size_t bg) {
            const unsigned int b = (unsigned int)(bg / groups);
            const unsigned int g = (unsigned int)(bg % groups);
            const T *inb = in + (size_t)b * IH * IW * in_dim.channel();
            T *outb = out + (size_t)b * OH * OW * filter_size;
            const F *filt_g = filt + (size_t)g * ocg * K; // [ocg, K] row-major
            std::vector<float> col((size_t)K * N, 0.0f);
            // im2col gather: col[(kh*fw+kw)*icg + c][oh*OW+ow]
            for (unsigned int oh = 0; oh < OH; ++oh) {
              for (unsigned int ow = 0; ow < OW; ++ow) {
                const size_t on = (size_t)oh * OW + ow;
                for (unsigned int kh = 0; kh < fh; ++kh) {
                  int ih = (int)(oh * sh) - ph + (int)(kh * dh);
                  if (ih < 0 || ih >= (int)IH)
                    continue;
                  for (unsigned int kw = 0; kw < fw; ++kw) {
                    int iw = (int)(ow * sw) - pw + (int)(kw * dw_);
                    if (iw < 0 || iw >= (int)IW)
                      continue;
                    const T *id = inb +
                                  ((size_t)ih * IW + iw) * in_dim.channel() +
                                  (size_t)g * icg;
                    const unsigned int kbase = (kh * fw + kw) * icg;
                    for (unsigned int c = 0; c < icg; ++c)
                      col[(size_t)(kbase + c) * N + on] =
                        static_cast<float>(id[c]);
                    (void)id;
                  }
                }
              }
            }
            // sgemm: out_g[ocg, N] = filt_g[ocg, K] x col[K, N]
            // Row-major: M=ocg, N=N, K=K, lda=K, ldb=N, ldc=N. ROW_MAJOR=0
            // (matches ARM/x86 storage-order constants).
            std::vector<float> out_g((size_t)ocg * N, 0.0f);
            nntrainer::getComputeOps()->sgemm_fp32(
              0 /*ROW_MAJOR*/, false /*transA*/, false /*transB*/, ocg, N, K,
              1.0f, reinterpret_cast<const float *>(filt_g), K, col.data(), N,
              0.0f, out_g.data(), N);
            // Narrow + scatter to NHWC out: outb[(oh*OW+ow)*filter_size + g*ocg
            // + o]
            for (unsigned int o = 0; o < ocg; ++o) {
              const unsigned int oc = g * ocg + o;
              for (unsigned int on = 0; on < N; ++on)
                outb[(size_t)on * filter_size + oc] =
                  static_cast<T>(out_g[(size_t)o * N + on]);
            }
          });
          return;
        }
        for (unsigned int b = 0; b < B; ++b) {
          const T *inb = in + (size_t)b * IH * IW * in_dim.channel();
          T *outb = out + (size_t)b * OH * OW * filter_size;
          for (unsigned int g = 0; g < groups; ++g) {
            for (unsigned int o = 0; o < ocg; ++o) {
              const unsigned int oc = g * ocg + o;
              const F *filt_oc = filt + (size_t)oc * icg * fh * fw;
              for (unsigned int oh = 0; oh < OH; ++oh) {
                for (unsigned int ow = 0; ow < OW; ++ow) {
                  float acc = 0.0f;
                  for (unsigned int kh = 0; kh < fh; ++kh) {
                    int ih = (int)(oh * sh) - ph + (int)(kh * dh);
                    if (ih < 0 || ih >= (int)IH)
                      continue;
                    for (unsigned int kw = 0; kw < fw; ++kw) {
                      int iw = (int)(ow * sw) - pw + (int)(kw * dw_);
                      if (iw < 0 || iw >= (int)IW)
                        continue;
                      const T *id = inb +
                                    ((size_t)ih * IW + iw) * in_dim.channel() +
                                    (size_t)g * icg;
                      const F *fk = filt_oc + (kh * fw + kw) * icg;
                      for (unsigned int c = 0; c < icg; ++c)
                        acc +=
                          static_cast<float>(id[c]) * static_cast<float>(fk[c]);
                    }
                  }
                  outb[((size_t)oh * OW + ow) * filter_size + oc] =
                    static_cast<T>(acc);
                }
              }
            }
          }
        }
      };
      if (in_dt == nntrainer::Tdatatype::FP32) {
        if (filt_dt == nntrainer::Tdatatype::FP32)
          run_g(input_use.getData<float>(), hidden_.getData<float>(),
                filter_kernel.getData<float>());
#ifdef ENABLE_FP16
        else
          run_g(input_use.getData<float>(), hidden_.getData<float>(),
                filter_kernel.getData<_FP16>());
#endif
      }
#ifdef ENABLE_FP16
      else {
        if (filt_dt == nntrainer::Tdatatype::FP32)
          run_g(input_use.getData<_FP16>(), hidden_.getData<_FP16>(),
                filter_kernel.getData<float>());
        else
          run_g(input_use.getData<_FP16>(), hidden_.getData<_FP16>(),
                filter_kernel.getData<_FP16>());
      }
#endif
    } else {
      // getSharedDataTensor()/reshape() adopt the *passed* TensorDim's dtype
      // (TensorBase::getSharedDataTensor: ret->dim = dim_). A bare {..} dim
      // literal defaults to FP32, so on an FP16 activation graph every sub-view
      // below would silently relabel FP16-backed storage as FP32 -- im2col
      // would gather at the wrong precision and the dot would write 4-byte
      // floats into 2-byte half storage (overflow / garbage). Carry each
      // parent's real dtype onto its view. (For an all-FP32 graph these are
      // no-ops, so the historical path is unchanged.)
      const auto in_dt = input_use.getDataType();
      const auto filt_dt = filter_kernel.getDataType();
      const auto out_dt = hidden_.getDataType();
      for (unsigned int b = 0; b < in_dim.batch(); ++b) {
        Tensor out = hidden_.getBatchSlice(b, 1);
        TensorDim out_rdim({filter_size, owoh});
        out_rdim.setDataType(out_dt);
        out.reshape(out_rdim);
        Tensor in_sub = input_use.getBatchSlice(b, 1);
        TensorDim col_dim = calcCol2ImOutputDim(out_dim, fdim_g);
        col_dim.setDataType(in_dt);
        Tensor result = Tensor(col_dim);
        for (unsigned int g = 0; g < groups; ++g) {
          TensorDim ing_dim({1, icg, in_dim.height(), in_dim.width()},
                            in_sub.getTensorType());
          Tensor in_g =
            in_sub.getSharedDataTensor(ing_dim, (size_t)g * icg * ihw);
          TensorDim filtg_dim({ocg, (size_t)icg * fh * fw},
                              filter_kernel.getTensorType());
          Tensor filt_g = filter_kernel.getSharedDataTensor(
            filtg_dim, (size_t)g * ocg * icg * fh * fw);
          TensorDim outg_dim({ocg, owoh}, out.getTensorType());
          Tensor out_g =
            out.getSharedDataTensor(outg_dim, (size_t)g * ocg * owoh);
          result.setZero();
          im2col(in_g, fdim_g, padding, stride, dilation, result);
          filt_g.dot(result, out_g, false, true);
        }
        result.deallocate();
      }
    }
#ifdef __ARM_NEON
  depthwise_done:; // jump target for the env-gated vectorized NHWC depthwise
#endif
  }

  // Per-channel W8A8 convs applied bias + SiLU + quantize inline per batch, so
  // skip the generic epilogue entirely (it would double the bias). MUST mirror
  // the forward branches exactly: the per-channel branch covers every NHWC
  // conv EXCEPT the stem (in_ch == 3), and the stem is covered by the direct
  // FP32 fast path below IF its conditions held (mirrored in
  // stem_direct_done); a stem that fell back to the standard path still needs
  // the generic bias here -- skipping it once left the stem bias-less
  // (catastrophic accuracy loss).
  const bool w8a8_perch_env = std::getenv("NNTR_W8A8") != nullptr &&
                              std::getenv("NNTR_W8A8_PERCH") != nullptr;
  const bool conv_groups_one =
    (std::get<props::ConvGroups>(conv_props).empty() ||
     std::get<props::ConvGroups>(conv_props).get() == 1);
  const bool stem_direct_done =
    w8a8_perch_env && conv_groups_one &&
    filter_kernel.getDataType() == nntrainer::Tdatatype::FP32 &&
    input_.getDim().getFormat() == ml::train::TensorDim::Format::NHWC &&
    input_.getDim().channel() == 3 &&
    input_.getDataType() == nntrainer::Tdatatype::FP32 &&
    hidden_.getDataType() == nntrainer::Tdatatype::FP32 &&
    dilation[0].get() == 1 && dilation[1].get() == 1 &&
    (unsigned int)std::get<props::FilterSize>(conv_props) % 4 == 0;
  const bool perch_done =
    (w8a8_perch_env && conv_groups_one &&
     input_.getDim().getFormat() == ml::train::TensorDim::Format::NHWC &&
     input_.getDim().channel() != 3) ||
    stem_direct_done;

  if (const char *dl = std::getenv("NNTR_PREBIAS_DEBUG");
      dl && context.getName() == dl &&
      hidden_.getDataType() == nntrainer::Tdatatype::FP32) {
    const float *hd = hidden_.getData<float>();
    float hmin = 1e30f, hmax = -1e30f;
    size_t nnan = 0;
    for (size_t i = 0; i < hidden_.size(); ++i) {
      float v = hd[i];
      if (std::isnan(v)) {
        nnan++;
        continue;
      }
      if (v < hmin)
        hmin = v;
      if (v > hmax)
        hmax = v;
    }
    std::cerr << "[PREBIAS] " << context.getName()
              << " pre-bias hidden_ range=[" << hmin << "," << hmax
              << "] nan=" << nnan << " perch_done=" << perch_done << "\n"
              << std::flush;
  }

  if (!perch_done)
    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &bias_kernel = context.getWeight(wt_idx[ConvParams::bias]);
      if (hidden_.getFormat() == ml::train::TensorDim::Format::NCHW) {
        status = hidden_.add_i(bias_kernel);
        if (status != ML_ERROR_NONE) {
          throw std::invalid_argument("[Conv2D] adding bias failed");
        }
      } else {
        // NHWC: channel is innermost. bias [out_ch] must be added per
        // (n,h,w,c). add_i assumes NCHW channel-major broadcast, so do it
        // inline.
        const unsigned int C = out_dim.channel();
        const unsigned int HW = out_dim.height() * out_dim.width();
        const unsigned int B = out_dim.batch();
        // bias_kernel's requested dtype can differ from hidden_'s (e.g. a
        // non-Q8_0-eligible conv like the stem stays FP32-activation while the
        // bias weight itself follows the model's FP16 default) -- the raw
        // getData<T>() casts below have no dtype check, so convert by value
        // first instead of silently reinterpreting the wrong-width bytes.
        Tensor bias_conv_owner;
        const Tensor *bias_ptr = &bias_kernel;
        if (bias_kernel.getDataType() != hidden_.getDataType()) {
          bias_conv_owner = bias_kernel.clone(hidden_.getDataType());
          bias_ptr = &bias_conv_owner;
        }
        const Tensor &bias_kernel_c = *bias_ptr;
        if (hidden_.getDataType() == nntrainer::Tdatatype::FP32) {
          float *d = hidden_.getData<float>();
          const float *bias = bias_kernel_c.getData<float>();
#if defined(__ARM_NEON)
          const unsigned int C_aligned = (C / 4) * 4;
          for (unsigned int b = 0; b < B; ++b) {
            for (unsigned int p = 0; p < HW; ++p) {
              float *curr_d = d + ((size_t)b * HW + p) * C;
              unsigned int c = 0;
              for (; c < C_aligned; c += 4) {
                float32x4_t v_d = vld1q_f32(curr_d + c);
                float32x4_t v_b = vld1q_f32(bias + c);
                float32x4_t v_res = vaddq_f32(v_d, v_b);
                vst1q_f32(curr_d + c, v_res);
              }
              for (; c < C; ++c) {
                curr_d[c] += bias[c];
              }
            }
          }
#else
          for (unsigned int b = 0; b < B; ++b)
            for (unsigned int p = 0; p < HW; ++p)
              for (unsigned int c = 0; c < C; ++c)
                d[((size_t)b * HW + p) * C + c] += bias[c];
#endif
        }
#ifdef ENABLE_FP16
        else if (hidden_.getDataType() == nntrainer::Tdatatype::FP16) {
          _FP16 *d = hidden_.getData<_FP16>();
          const _FP16 *bias = bias_kernel_c.getData<_FP16>();
#if defined(__ARM_NEON)
          const unsigned int C_aligned = (C / 8) * 8;
          for (unsigned int b = 0; b < B; ++b) {
            for (unsigned int p = 0; p < HW; ++p) {
              _FP16 *curr_d = d + ((size_t)b * HW + p) * C;
              unsigned int c = 0;
              for (; c < C_aligned; c += 8) {
                float16x8_t v_d =
                  vld1q_f16(reinterpret_cast<const __fp16 *>(curr_d + c));
                float16x8_t v_b =
                  vld1q_f16(reinterpret_cast<const __fp16 *>(bias + c));
                float16x8_t v_res = vaddq_f16(v_d, v_b);
                vst1q_f16(reinterpret_cast<__fp16 *>(curr_d + c), v_res);
              }
              for (; c < C; ++c) {
                curr_d[c] += bias[c];
              }
            }
          }
#else
          for (unsigned int b = 0; b < B; ++b)
            for (unsigned int p = 0; p < HW; ++p)
              for (unsigned int c = 0; c < C; ++c)
                d[((size_t)b * HW + p) * C + c] += bias[c];
#endif
        }
#endif
      }
    }

  // fused activation epilogue (value-identical to a standalone
  // ActivationLayer). SiLU -- the YOLO conv activation -- takes the
  // vectorized in-place kernel; every other activation runs the generic
  // ActiFunc, in-place for relu/sigmoid/tanh, else via a temp copy.
  // A QINT8 output already applied SiLU inside the W8A8 quantize epilogue and
  // the stem direct fast path fuses bias + SiLU per row itself, so both skip
  // this epilogue.
  auto &fused_act = std::get<props::FusedActivation>(conv_props);
  if (!fused_act.empty() && fused_act.get() != ActivationType::ACT_NONE &&
      hidden_.getDataType() != nntrainer::Tdatatype::QINT8 &&
      !stem_direct_done) {
    if (fused_act.get() == ActivationType::ACT_SWISH) {
      const size_t n = hidden_.size();
#ifdef ENABLE_FP16
      if (hidden_.getDataType() == nntrainer::Tdatatype::FP16) {
        convApplySwishInplace(hidden_.getData<_FP16>(), n);
      } else
#endif
      {
        convApplySwishInplace(hidden_.getData<float>(), n);
      }
    } else if (acti_func.supportInPlace()) {
      acti_func.run_fn(hidden_, hidden_);
    } else {
      Tensor in_copy = hidden_.clone();
      acti_func.run_fn(in_copy, hidden_);
    }
  }

  // Dump one named layer's TRUE final output -- after the fused-activation
  // epilogue above, so (unlike the mid-function NNTR_DUMP_LAYER site earlier
  // in this function, which fires before that epilogue and forces external
  // bisection scripts to re-approximate SiLU themselves, mismatching the
  // engine's fast vectorized approximation by ~1-2%) this needs no
  // reconstruction and is safe to diff directly against any other dump.
  if (const char *dl = std::getenv("NNTR_DUMP_LAYER_FINAL");
      dl && context.getName() == dl) {
    const char *dp = std::getenv("NNTR_DUMP_PATH");
    if (dp) {
      std::vector<float> buf(hidden_.size());
      if (hidden_.getDataType() == nntrainer::Tdatatype::FP16) {
#ifdef ENABLE_FP16
        const _FP16 *d = hidden_.getData<_FP16>();
        for (unsigned int i = 0; i < hidden_.size(); ++i)
          buf[i] = (float)d[i];
#endif
      } else if (hidden_.getDataType() == nntrainer::Tdatatype::QINT8) {
        const int8_t *q = hidden_.getData<int8_t>();
        const float sc = hidden_.getScale<float>()[0];
        static const bool perch_asym_env =
          std::getenv("NNTR_W8A8_PERCH") != nullptr &&
          std::getenv("NNTR_W8A8_SYM") == nullptr;
        constexpr float kActOff = 0.27846455f;
        for (unsigned int i = 0; i < hidden_.size(); ++i)
          buf[i] = perch_asym_env ? ((float)q[i] + 128.f) * sc - kActOff
                                  : sc * (float)q[i];
      } else {
        const float *d = hidden_.getData<float>();
        std::copy(d, d + hidden_.size(), buf.begin());
      }
      std::ofstream of(dp, std::ios::binary);
      of.write(reinterpret_cast<const char *>(buf.data()),
               buf.size() * sizeof(float));
    }
  }
}

void Conv2DLayer::calcDerivative(RunLayerContext &context) {
  NNTR_THROW_IF(!std::get<props::ConvGroups>(conv_props).empty() &&
                  std::get<props::ConvGroups>(conv_props).get() != 1,
                std::invalid_argument)
    << "[Conv2D] backward for grouped convolution (groups>1) is not yet "
       "implemented; only forward/inference is supported.";
  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  Tensor &filter_kernel = context.getWeight(wt_idx[ConvParams::weight]);

  TensorDim filter_dim = filter_kernel.getDim();
  TensorDim filter_dim_squeezed{filter_kernel.batch(),
                                filter_kernel.getDim().getFeatureLen()};

  filter_kernel.reshape(filter_dim_squeezed);

  /// for each batch
  /// filter_kernel^T X derivaitive  -> column matrix
  /// col2im(column matrix) to reconstruct the original image

  auto compute_derivative = [&](unsigned int s, unsigned int e,
                                unsigned int pid, void *user_data) {
    Tensor result =
      Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));

    for (unsigned int b = s; b < e; ++b) {
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      Tensor in_deriv_sub = input_derivative.getBatchSlice(b, 1);
      deriv_sub.reshape(
        {filter_size, derivative.width() * derivative.height()});
      // filter_kernel is (K, CRS), deriv_sub is (K, OH*OW), result is (CRS,
      // OH*OW)
      filter_kernel.dot(deriv_sub, result, true, false);
      col2im(result, filter_dim, padding, stride, dilation, in_deriv_sub);
      // in_derv_sub is (C,H,W)
    }
    result.deallocate();
  };

  auto workers = ParallelBatch(compute_derivative, derivative.batch(), nullptr);

  if (workers.getNumWorkers() > 1) {
    workers.run();
  } else {
    compute_derivative(0, derivative.batch(), 0, nullptr);
  }

  filter_kernel.reshape(filter_dim);
}

void Conv2DLayer::calcGradient(RunLayerContext &context) {
  NNTR_THROW_IF(!std::get<props::ConvGroups>(conv_props).empty() &&
                  std::get<props::ConvGroups>(conv_props).get() != 1,
                std::invalid_argument)
    << "[Conv2D] backward for grouped convolution (groups>1) is not yet "
       "implemented; only forward/inference is supported.";
  unsigned int filter_size = std::get<props::FilterSize>(conv_props);
  auto &stride = std::get<std::array<props::Stride, CONV2D_DIM>>(conv_props);
  auto &dilation =
    std::get<std::array<props::Dilation, CONV2D_DIM>>(conv_props);

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  Tensor &delK = context.getWeightGrad(wt_idx[ConvParams::weight]);
  delK.setZero();

  TensorDim filter_dim = delK.getDim();
  TensorDim filter_dim_squeezed{filter_dim.batch(), filter_dim.getFeatureLen()};

  delK.reshape(filter_dim_squeezed);

  /**
   * no need to set zero for im2col_result, as its lifespan is ITERATION,
   * so its zero padded values will still be zero
   */

  TensorDim out_dim_squeezed{filter_size,
                             derivative.width() * derivative.height()};
  auto workers = ParallelBatch(input_.batch());
  /// input -(im2col)-> column_matrix -> filter x (column_matrix) = output
  /// so delK = dy x column_matrix ^ T;
  if (workers.getNumWorkers() > 1) {

    TensorDim delK_ext = filter_dim_squeezed;
    delK_ext.batch(input_.batch());

    Tensor delK_par = Tensor(delK_ext);
    delK_par.setZero();

    auto calc_grad_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                             void *user_data) {
      Tensor result =
        Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));
      result.setZero();
      for (unsigned int b = s; b < e; ++b) {
        Tensor deriv_sub = derivative.getBatchSlice(b, 1);
        Tensor delK_sub = delK_par.getBatchSlice(b, 1);
        deriv_sub.reshape(out_dim_squeezed);

        Tensor in_sub = input_.getBatchSlice(b, 1);

        /**
         * @todo this result can be cached from the forward iteration at the
         * expense of memory. In this case, memory of im2col_result must be
         * saved for the whole batch. try this while benchmarking.
         */
        // deriv_sub is (K, OH*OW) and result is (CRS, OH*OW)
        im2col(in_sub, filter_dim, padding, stride, dilation, result);
        deriv_sub.dot(result, delK_sub, false, false);
      }
      result.deallocate();
    };

    workers.setCallback(calc_grad_job, nullptr);

    workers.run();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      Tensor delK_sub = delK_par.getBatchSlice(b, 1);
      delK.add_i(delK_sub);
    }

  } else {
    Tensor result =
      Tensor(calcCol2ImOutputDim(derivative.getDim(), filter_dim));
    result.setZero();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      deriv_sub.reshape(out_dim_squeezed);

      Tensor in_sub = input_.getBatchSlice(b, 1);

      /**
       * @todo this result can be cached from the forward iteration at the
       * expense of memory. In this case, memory of im2col_result must be saved
       * for the whole batch. try this while benchmarking.
       */
      im2col(in_sub, filter_dim, padding, stride, dilation, result);
      deriv_sub.dot(result, delK, false, false, b == 0 ? 0.0f : 1.0f);
    }
    result.deallocate();
  }
  delK.reshape(filter_dim);
  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &delBias = context.getWeightGrad(wt_idx[ConvParams::bias]);
    delBias.setZero();
    derivative.sum({0, 2, 3}, delBias);
  }
}

void Conv2DLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  // The forward scratch buffers (im2col column buffer and quantized-GEMM
  // output) are requested in finalize() sized to the batch present at init.
  // When the runtime batch changes the framework resizes inputs/outputs but
  // not these layer-private scratch tensors, so rebatch them here — otherwise
  // forwarding()'s getBatchSlice(b, 1) for b >= the init batch reads past the
  // planned storage and aborts ("shared tensor bigger than tensor memory").
  if (wt_idx[ConvParams::im2col_scratch] !=
      std::numeric_limits<unsigned int>::max())
    context.updateTensor(wt_idx[ConvParams::im2col_scratch], batch);
  if (wt_idx[ConvParams::qgemm_scratch] !=
      std::numeric_limits<unsigned int>::max())
    context.updateTensor(wt_idx[ConvParams::qgemm_scratch], batch);
  if (wt_idx[ConvParams::q8act_scratch] !=
      std::numeric_limits<unsigned int>::max())
    context.updateTensor(wt_idx[ConvParams::q8act_scratch], batch);
}

void Conv2DLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void Conv2DLayer::save(std::ofstream &file, RunLayerContext &run_context,
                       bool opt_var, ml::train::ExecutionMode mode,
                       bool trainable, ml::train::TensorDim::DataType dtype,
                       ml::train::ISA target_isa) const {
  // Optimizer-variable save (training only) has no conv-specific layout, so
  // defer to the base implementation.
  if (opt_var) {
    Layer::save(file, run_context, opt_var, mode, trainable, dtype, target_isa);
    return;
  }

  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    if (!run_context.isGradientFirstAccess(i))
      continue;

    auto &weight = run_context.getWeight(i);

    // No conversion requested, or already the target dtype: save as-is.
    if (dtype == TensorDim::DataType::NONE || weight.getDataType() == dtype) {
      weight.save(file);
      continue;
    }

    NNTR_THROW_IF(dtype != TensorDim::DataType::Q4_0 &&
                    dtype != TensorDim::DataType::Q8_0,
                  std::runtime_error)
      << "[Conv2D] save: unsupported quantization dtype (expected Q4_0 or "
         "Q8_0).";
    NNTR_THROW_IF(weight.getDataType() != TensorDim::DataType::FP32,
                  std::runtime_error)
      << "[Conv2D] quantized save only supports an FP32 source weight.";

    // A conv FP32 filter is [out_ch, in_ch, kh, kw] in NCHW, i.e. already
    // row-major [out_ch, CRS] (CRS = in_ch*kh*kw) = [N rows, K cols]. This is
    // exactly the layout quantize_q4_0/quantize_q8_0 consume (N rows of K), so
    // no transpose is needed (the FC path in the base class transposes because
    // its weight is stored [K, N]). The bias and any non-matmul weight (CRS ==
    // 1 or out_ch == 1) are kept FP32.
    const TensorDim dim = weight.getDim();
    const unsigned int out_ch = dim.batch();
    const unsigned int CRS = dim.channel() * dim.height() * dim.width();

    // Q4_0 and Q8_0 both block on QK=32, so the same block-alignment guard
    // applies. Non-eligible weights (bias, block-misaligned) stay FP32 so the
    // saved tensor still matches what the runtime layer allocates for it.
    if (out_ch <= 1 || CRS <= 1 || out_ch % 32 != 0 || CRS % 32 != 0) {
      weight.save(file);
      continue;
    }

    // [1, 1, K=CRS, N=out_ch] is the matmul-weight shape the quantized conv
    // consumes at load (Conv2DLayer::finalize builds the same shape).
    Tensor quant_weight(1, 1, CRS, out_ch, {Tformat::NCHW, dtype});
    std::vector<char> tmp(quant_weight.size());

    if (dtype == TensorDim::DataType::Q4_0) {
      // Q4_0 weights are ISA-repacked at save time (the runtime Q4_0 kernels
      // expect the interleaved layout: x4 on ARM, x8 on x86).
      quantize_q4_0(weight.getData<float>(), tmp.data(), out_ch, CRS, nullptr);
      repack_q4_0(quant_weight.getData<uint8_t>(), tmp.data(),
                  quant_weight.size(), out_ch, CRS, target_isa);
    } else {
      // Q8_0: quantize to plain block_q8_0 (2B fp16 d + 32 int8 qs per block),
      // d = max(|x|)/127, qs = round(x/d) clipped to [-128,127], matching
      // ggml's quantize_row_q8_0_ref. The plain stream is then repacked to the
      // block_q8_0x4 layout the FP16 q8_0×q8_0 indirect-conv kernel consumes
      // (4 weight columns interleaved per super-block). Without this repack the
      // kernel reads plain-block bytes as q8_0x4 and produces garbage/NaN.
      // The q8_0x4 layout is a logical byte interleave (ISA-independent); the
      // kernel decodes it, so the repacked file is portable across ISAs (unlike
      // Q4_0, whose kernel ISA-specific interleaving is dispatched at save).
      //
      // When quantizing FOR the W8A8 per-channel runtime (env NNTR_W8A8_PERCH,
      // the same flag that selects the per-channel kernel at run time), use one
      // scale per output-channel row instead of one per 32-block. This makes
      // the runtime's dequant->per-channel-requant the identity (see
      // __ggml_quantize_q8_0_per_channel), removing the per-block->per-channel
      // double quantization that costs one keypoint (81 -> 80). The repack and
      // byte layout are unchanged, so a per-channel file is still a valid Q8_0
      // tensor for every other reader.
      static const bool perch_quant = std::getenv("NNTR_W8A8_PERCH") != nullptr;
      if (perch_quant)
        __ggml_quantize_q8_0_per_channel(weight.getData<float>(), tmp.data(),
                                         out_ch, CRS);
      else
        quantize_q8_0(weight.getData<float>(), tmp.data(),
                      static_cast<int64_t>(out_ch), static_cast<int64_t>(CRS),
                      nullptr);
      repack_q8_0(quant_weight.getData<uint8_t>(), tmp.data(), out_ch, CRS);
    }
    quant_weight.save(file);
  }
}

void Conv2DLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

} /* namespace nntrainer */

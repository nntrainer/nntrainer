// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   int8_gemm.h
 * @date   24 August 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  W8A8 GEMM with channel-wise weight scales and a tensor-wise
 * activation scale.
 *
 * The weight is a Q8_0 byte stream whose blocks all carry the same scale within
 * an output channel (written by nntr_quantize_q8_0_per_channel), so there is
 * exactly one weight scale per channel. The activation is quantized with a
 * single scale for the whole tensor. Together those two properties make the
 * inner loop a pure int8 dot product accumulating into int32, with one scalar
 * multiply per output element at the end -- no per-block scale arithmetic,
 * which is what the block-scaled Q8_0 x Q8_0 kernel has to carry.
 *
 * The per-channel scales are read once with prepareWeight() and kept by the
 * caller next to the weight, so they never outlive or mismatch it.
 */

#ifndef __NNTR_INT8_GEMM_H__
#define __NNTR_INT8_GEMM_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <fp16.h>
#include <thread_manager.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace nntrainer {
namespace int8_gemm {

/** Q8_0 block geometry: fp16 scale followed by 32 int8 quants. */
constexpr unsigned int QK = 32;
constexpr unsigned int BLOCK_BYTES = 2 + QK;

/**
 * @brief Per-channel scales of a weight, prepared once for the kernel.
 */
struct PerChannelWeight {
  bool usable = false;      /**< false when the stream is not channel-wise */
  std::vector<float> scale; /**< [N] one scale per output channel */
};

/**
 * @brief Read the per-channel scales of a Q8_0 weight [N, K].
 *
 * The int8 values are read in place from the Q8_0 stream later, so only the
 * N scales are extracted. The caller keeps the result with the weight and
 * prepares it again whenever the weight changes.
 *
 * `usable` is false when any row's blocks disagree on their scale, i.e. the
 * stream is a genuine per-block Q8_0 weight. Callers must then use the
 * block-scaled kernel: reading it as channel-wise would silently drop every
 * scale but the first of each row.
 */
inline PerChannelWeight prepareWeight(const void *B, unsigned int N,
                                      unsigned int K) {
  PerChannelWeight w;
  const unsigned int nb = K / QK;
  if (nb == 0 || K % QK != 0)
    return w;

  const char *base = static_cast<const char *>(B);
  w.scale.resize(N);
  for (unsigned int n = 0; n < N; ++n) {
    const char *src = base + (size_t)n * nb * BLOCK_BYTES;
    uint16_t d0;
    std::memcpy(&d0, src, 2);
    for (unsigned int b = 1; b < nb; ++b) {
      uint16_t d;
      std::memcpy(&d, src + (size_t)b * BLOCK_BYTES, 2);
      if (d != d0)
        return PerChannelWeight();
    }
    w.scale[n] = compute_fp16_to_fp32(d0);
  }
  w.usable = true;
  return w;
}

/**
 * @brief int8 dot product of `len` elements accumulating in int32.
 */
inline int32_t dotI8(const int8_t *a, const int8_t *b, unsigned int len) {
#if defined(__ARM_FEATURE_DOTPROD)
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  unsigned int i = 0;
  for (; i + 32 <= len; i += 32) {
    acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
    acc1 = vdotq_s32(acc1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
  }
  for (; i + 16 <= len; i += 16)
    acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
  int32_t sum = vaddvq_s32(vaddq_s32(acc0, acc1));
  for (; i < len; ++i)
    sum += (int32_t)a[i] * (int32_t)b[i];
  return sum;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  int32x4_t acc = vdupq_n_s32(0);
  unsigned int i = 0;
  for (; i + 16 <= len; i += 16) {
    const int8x16_t va = vld1q_s8(a + i);
    const int8x16_t vb = vld1q_s8(b + i);
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
  }
#if defined(__aarch64__)
  int32_t sum = vaddvq_s32(acc);
#else
  // armv7 has no across-vector add (vaddvq); reduce pairwise instead.
  const int32x2_t half = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
  int32_t sum = vget_lane_s32(vpadd_s32(half, half), 0);
#endif
  for (; i < len; ++i)
    sum += (int32_t)a[i] * (int32_t)b[i];
  return sum;
#else
  int32_t sum = 0;
  for (unsigned int i = 0; i < len; ++i)
    sum += (int32_t)a[i] * (int32_t)b[i];
  return sum;
#endif
}

/**
 * @brief Quantize a whole activation block with one shared scale.
 *
 * @return the scale that maps the int8 values back to FP32
 */
inline float quantizeActivation(const float *A, size_t count,
                                std::vector<int8_t> &out) {
  float amax = 0.0f;
  for (size_t i = 0; i < count; ++i)
    amax = std::max(amax, std::fabs(A[i]));
  const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;

  out.resize(count);
  int8_t *q = out.data();
  for (size_t i = 0; i < count; ++i) {
    const float r = std::round(A[i] * inv);
    q[i] = (int8_t)std::max(-127.0f, std::min(127.0f, r));
  }
  return scale;
}

/**
 * @brief W8A8 GEMM: FP32 activation [M, K] x channel-wise Q8_0 weight [N, K]
 * -> FP32 output [M, N].
 *
 * @param M activation rows
 * @param N output channels
 * @param K reduction length, must be a multiple of 32
 * @param A activation, row-major [M, K]
 * @param B weight, Q8_0 blocks laid out per output channel
 * @param w scales of @a B from prepareWeight()
 * @param C output, row-major [M, N]
 * @retval false the weight is not channel-wise; nothing was written and the
 *         caller must use the block-scaled kernel
 */
inline bool gemmPerChannelA8(unsigned int M, unsigned int N, unsigned int K,
                             const float *A, const void *B,
                             const PerChannelWeight &w, float *C) {
  if (!w.usable || w.scale.size() != N)
    return false;

  // One scratch buffer per thread, reused across calls: the quantized
  // activation is the only per-call allocation this kernel would otherwise do.
  static thread_local std::vector<int8_t> scratch;
  const float a_scale = quantizeActivation(A, (size_t)M * K, scratch);
  const int8_t *Aq = scratch.data();

  const unsigned int nb = K / QK;
  const char *wbase = static_cast<const char *>(B);
  auto &tm = ThreadManager::Global();
  tm.parallel_for(0, N, [&](size_t n) {
    const char *wrow = wbase + (size_t)n * nb * BLOCK_BYTES;
    const float s = a_scale * w.scale[n];
    for (unsigned int m = 0; m < M; ++m) {
      const int8_t *a = Aq + (size_t)m * K;
      int32_t acc = 0;
      for (unsigned int b = 0; b < nb; ++b) {
        const int8_t *wq =
          reinterpret_cast<const int8_t *>(wrow + (size_t)b * BLOCK_BYTES + 2);
        acc += dotI8(a + (size_t)b * QK, wq, QK);
      }
      C[(size_t)m * N + n] = (float)acc * s;
    }
  });
  return true;
}

/**
 * @brief W8A8 GEMM with a pre-quantized int8 activation.
 *
 * Same kernel as gemmPerChannelA8 but the activation arrives already quantized
 * to int8 with a known per-tensor scale (the int8-resident path: the previous
 * layer emitted a QINT8 tensor whose inline scale is a_scale). Skipping the
 * amax+quantize here is what makes activations stay int8 between layers.
 *
 * @param Aq activation int8 data, row-major [M, K]
 * @param a_scale per-tensor FP32 activation scale
 * @param w scales of @a B from prepareWeight()
 * @retval false the weight is not channel-wise; nothing was written
 */
inline bool gemmPerChannelA8_q8in(unsigned int M, unsigned int N,
                                  unsigned int K, const int8_t *Aq,
                                  float a_scale, const void *B,
                                  const PerChannelWeight &w, float *C) {
  if (!w.usable || w.scale.size() != N)
    return false;

  const unsigned int nb = K / QK;
  const char *wbase = static_cast<const char *>(B);
  auto &tm = ThreadManager::Global();
  tm.parallel_for(0, N, [&](size_t n) {
    const char *wrow = wbase + (size_t)n * nb * BLOCK_BYTES;
    const float s = a_scale * w.scale[n];
    for (unsigned int m = 0; m < M; ++m) {
      const int8_t *a = Aq + (size_t)m * K;
      int32_t acc = 0;
      for (unsigned int b = 0; b < nb; ++b) {
        const int8_t *wq =
          reinterpret_cast<const int8_t *>(wrow + (size_t)b * BLOCK_BYTES + 2);
        acc += dotI8(a + (size_t)b * QK, wq, QK);
      }
      C[(size_t)m * N + n] = (float)acc * s;
    }
  });
  return true;
}

} // namespace int8_gemm
} // namespace nntrainer

#endif /* __NNTR_INT8_GEMM_H__ */

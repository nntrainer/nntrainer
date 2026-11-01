// SPDX-License-Identifier: Apache-2.0
/** @file paged_attention.cpp
 * @brief Direct CPU attention over checked KV segments using online softmax.
 * @author Codex (OpenAI)
 */
#include <fp16.h>
#include <kv_cache/kv_cache_manager.h>
#include <thread_manager.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace nntrainer::kv {
namespace {

inline double compute_dot_f32(const float *a, const float *b, size_t n) {
#if defined(__AVX2__)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 =
      _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                           _mm256_loadu_ps(b + i + 8), acc1);
  }
  for (; i + 8 <= n; i += 8) {
    acc0 =
      _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
  }
  __m256 acc = _mm256_add_ps(acc0, acc1);
  float buf[8];
  _mm256_storeu_ps(buf, acc);
  double sum =
    buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
  for (; i < n; ++i)
    sum += static_cast<double>(a[i]) * b[i];
  return sum;
#elif defined(__aarch64__) || defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  for (; i + 4 <= n; i += 4) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
  }
  acc0 = vaddq_f32(acc0, acc1);
  double sum = vgetq_lane_f32(acc0, 0) + vgetq_lane_f32(acc0, 1) +
               vgetq_lane_f32(acc0, 2) + vgetq_lane_f32(acc0, 3);
  for (; i < n; ++i)
    sum += static_cast<double>(a[i]) * b[i];
  return sum;
#else
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i)
    sum += static_cast<double>(a[i]) * b[i];
  return sum;
#endif
}

inline void accum_f32(double *accum, const float *v, double old_w, double w,
                      size_t n) {
#if defined(__AVX2__)
  __m256d vow = _mm256_set1_pd(old_w);
  __m256d vw = _mm256_set1_pd(w);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    __m128 v_f32 = _mm_loadu_ps(v + i);
    __m256d v_f64 = _mm256_cvtps_pd(v_f32);
    __m256d acc = _mm256_loadu_pd(accum + i);
    acc = _mm256_add_pd(_mm256_mul_pd(acc, vow), _mm256_mul_pd(v_f64, vw));
    _mm256_storeu_pd(accum + i, acc);
  }
  for (; i < n; ++i)
    accum[i] = accum[i] * old_w + w * v[i];
#elif defined(__aarch64__) || defined(__ARM_NEON)
  float64x2_t vow = vdupq_n_f64(old_w);
  float64x2_t vw = vdupq_n_f64(w);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t vf = vld1q_f32(v + i);
    float64x2_t vf_low = vcvt_f64_f32(vget_low_f32(vf));
    float64x2_t vf_high = vcvt_f64_f32(vget_high_f32(vf));
    float64x2_t acc0 = vld1q_f64(accum + i);
    float64x2_t acc1 = vld1q_f64(accum + i + 2);
    acc0 = vmlaq_f64(vmulq_f64(acc0, vow), vf_low, vw);
    acc1 = vmlaq_f64(vmulq_f64(acc1, vow), vf_high, vw);
    vst1q_f64(accum + i, acc0);
    vst1q_f64(accum + i + 2, acc1);
  }
  for (; i < n; ++i)
    accum[i] = accum[i] * old_w + w * v[i];
#else
  for (size_t i = 0; i < n; ++i)
    accum[i] = accum[i] * old_w + w * v[i];
#endif
}

inline double compute_dot_f16(const float *a, const uint16_t *b, size_t n) {
#if defined(__AVX2__) && defined(__F16C__)
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m128i b_raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i b_raw1 =
      _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i + 8));
    __m256 b_f32_0 = _mm256_cvtph_ps(b_raw0);
    __m256 b_f32_1 = _mm256_cvtph_ps(b_raw1);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), b_f32_0, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), b_f32_1, acc1);
  }
  for (; i + 8 <= n; i += 8) {
    __m128i b_raw0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i));
    __m256 b_f32_0 = _mm256_cvtph_ps(b_raw0);
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), b_f32_0, acc0);
  }
  __m256 acc = _mm256_add_ps(acc0, acc1);
  float buf[8];
  _mm256_storeu_ps(buf, acc);
  double sum =
    buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
  for (; i < n; ++i)
    sum += static_cast<double>(a[i]) * nntrainer::compute_fp16_to_fp32(b[i]);
  return sum;
#elif defined(__aarch64__) || defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    float16x8_t b_f16 = vld1q_f16(reinterpret_cast<const __fp16 *>(b + i));
    float32x4_t b_low = vcvt_f32_f16(vget_low_f16(b_f16));
    float32x4_t b_high = vcvt_f32_f16(vget_high_f16(b_f16));
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), b_low);
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), b_high);
  }
  for (; i + 4 <= n; i += 4) {
    float16x4_t b_f16 = vld1_f16(reinterpret_cast<const __fp16 *>(b + i));
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vcvt_f32_f16(b_f16));
  }
  acc0 = vaddq_f32(acc0, acc1);
  double sum = vgetq_lane_f32(acc0, 0) + vgetq_lane_f32(acc0, 1) +
               vgetq_lane_f32(acc0, 2) + vgetq_lane_f32(acc0, 3);
  for (; i < n; ++i)
    sum += static_cast<double>(a[i]) * nntrainer::compute_fp16_to_fp32(b[i]);
  return sum;
#else
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i)
    sum += static_cast<double>(a[i]) * nntrainer::compute_fp16_to_fp32(b[i]);
  return sum;
#endif
}

inline void accum_f16(double *accum, const uint16_t *v, double old_w, double w,
                      size_t n) {
#if defined(__AVX2__) && defined(__F16C__)
  __m256d vow = _mm256_set1_pd(old_w);
  __m256d vw = _mm256_set1_pd(w);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m128i v_raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(v + i));
    __m256 v_f32 = _mm256_cvtph_ps(v_raw);
    __m256d v_f64_0 = _mm256_cvtps_pd(_mm256_castps256_ps128(v_f32));
    __m256d v_f64_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v_f32, 1));
    __m256d acc0 = _mm256_loadu_pd(accum + i);
    __m256d acc1 = _mm256_loadu_pd(accum + i + 4);
    acc0 = _mm256_add_pd(_mm256_mul_pd(acc0, vow), _mm256_mul_pd(v_f64_0, vw));
    acc1 = _mm256_add_pd(_mm256_mul_pd(acc1, vow), _mm256_mul_pd(v_f64_1, vw));
    _mm256_storeu_pd(accum + i, acc0);
    _mm256_storeu_pd(accum + i + 4, acc1);
  }
  for (; i < n; ++i)
    accum[i] = accum[i] * old_w + w * nntrainer::compute_fp16_to_fp32(v[i]);
#elif defined(__aarch64__) || defined(__ARM_NEON)
  float64x2_t vow = vdupq_n_f64(old_w);
  float64x2_t vw = vdupq_n_f64(w);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    float16x8_t vf = vld1q_f16(reinterpret_cast<const __fp16 *>(v + i));
    float32x4_t vf_low = vcvt_f32_f16(vget_low_f16(vf));
    float32x4_t vf_high = vcvt_f32_f16(vget_high_f16(vf));
    float64x2_t acc0 = vld1q_f64(accum + i);
    float64x2_t acc1 = vld1q_f64(accum + i + 2);
    float64x2_t acc2 = vld1q_f64(accum + i + 4);
    float64x2_t acc3 = vld1q_f64(accum + i + 6);
    acc0 =
      vmlaq_f64(vmulq_f64(acc0, vow), vcvt_f64_f32(vget_low_f32(vf_low)), vw);
    acc1 =
      vmlaq_f64(vmulq_f64(acc1, vow), vcvt_f64_f32(vget_high_f32(vf_low)), vw);
    acc2 =
      vmlaq_f64(vmulq_f64(acc2, vow), vcvt_f64_f32(vget_low_f32(vf_high)), vw);
    acc3 =
      vmlaq_f64(vmulq_f64(acc3, vow), vcvt_f64_f32(vget_high_f32(vf_high)), vw);
    vst1q_f64(accum + i, acc0);
    vst1q_f64(accum + i + 2, acc1);
    vst1q_f64(accum + i + 4, acc2);
    vst1q_f64(accum + i + 6, acc3);
  }
  for (; i < n; ++i)
    accum[i] = accum[i] * old_w + w * nntrainer::compute_fp16_to_fp32(v[i]);
#else
  for (size_t i = 0; i < n; ++i)
    accum[i] = accum[i] * old_w + w * nntrainer::compute_fp16_to_fp32(v[i]);
#endif
}

} // namespace

std::vector<float> KvCacheManager::attention(const StepLease &step,
                                             size_t layer,
                                             const std::vector<float> &queries,
                                             const AttentionOptions &options,
                                             size_t batch) {
  checkStep(step);
  const auto &s = layerSpec(layer);
  if (s.capacity != s.max_tokens)
    throw std::invalid_argument(
      "Segmented attention requires non-ring storage");
  if (!options.query_heads || options.query_heads % s.num_kv_heads ||
      !std::isfinite(options.scale) || options.scale < 0 ||
      !std::isfinite(options.logit_softcap) || options.logit_softcap < 0)
    throw std::invalid_argument("Invalid segmented attention options");
  if (!options.sinks.empty() && options.sinks.size() != options.query_heads)
    throw std::invalid_argument(
      "Attention sink shape differs from query heads");
  for (float sink : options.sinks)
    if (!std::isfinite(sink))
      throw std::invalid_argument("Attention sink must be finite");
  const size_t width = checkedMultiply(options.query_heads, s.head_dim);
  const size_t q_tokens = step.to() - step.from();
  const size_t elements = checkedMultiply(q_tokens, width);
  if (queries.size() != elements)
    throw std::invalid_argument("Attention query shape differs from KV step");
  for (float value : queries)
    if (!std::isfinite(value))
      throw std::invalid_argument("Attention query must be finite");
  const auto keys = readKey(step, layer, batch);
  const auto values = readValue(step, layer, batch);
  std::vector<float> output(elements);
  const double scale = options.scale
                         ? options.scale
                         : 1.0 / std::sqrt(static_cast<double>(s.head_dim));
  const size_t group = options.query_heads / s.num_kv_heads;
  const size_t kv_width = s.num_kv_heads * s.head_dim;
  const bool is_f32 = (s.encoding == Encoding::Float32);

  const size_t total_tasks = q_tokens * options.query_heads;
  auto compute_task = [&](size_t task_idx) {
    const size_t q = task_idx / options.query_heads;
    const size_t h = task_idx % options.query_heads;

    std::vector<double> accumulator(s.head_dim, 0.0);
    double maximum = -std::numeric_limits<double>::infinity();
    double denominator = 0.0;
    if (!options.sinks.empty()) {
      maximum = options.sinks[h];
      denominator = 1.0;
    }

    const size_t end = options.causal ? step.from() + q + 1 : step.to();
    const size_t start =
      options.window && end > options.window ? end - options.window : 0;
    const float *q_head = queries.data() + q * width + h * s.head_dim;

    size_t token = 0;
    for (size_t p = 0; p < keys.size() && token < end; ++p) {
      const auto *k_raw = static_cast<const char *>(keys[p].data());
      const auto *v_raw = static_cast<const char *>(values[p].data());
      const size_t rows = keys[p].sizeBytes() / s.rowBytes();

      for (size_t row = 0; row < rows && token < end; ++row, ++token) {
        if (token < start)
          continue;
        const size_t offset = row * kv_width + (h / group) * s.head_dim;

        double score = 0.0;
        if (is_f32) {
          const auto *k_ptr = reinterpret_cast<const float *>(k_raw) + offset;
          score = compute_dot_f32(q_head, k_ptr, s.head_dim);
        } else {
          const auto *k_ptr =
            reinterpret_cast<const uint16_t *>(k_raw) + offset;
          score = compute_dot_f16(q_head, k_ptr, s.head_dim);
        }

        score *= scale;
        if (options.logit_softcap)
          score =
            options.logit_softcap * std::tanh(score / options.logit_softcap);

        if (!std::isfinite(score))
          throw std::invalid_argument(
            "Attention key produces a nonfinite score");

        const double next_max = std::max(maximum, score);
        const double old_weight = std::exp(maximum - next_max);
        const double weight = std::exp(score - next_max);
        denominator = denominator * old_weight + weight;

        if (is_f32) {
          const auto *v_ptr = reinterpret_cast<const float *>(v_raw) + offset;
          accum_f32(accumulator.data(), v_ptr, old_weight, weight, s.head_dim);
        } else {
          const auto *v_ptr =
            reinterpret_cast<const uint16_t *>(v_raw) + offset;
          accum_f16(accumulator.data(), v_ptr, old_weight, weight, s.head_dim);
        }

        maximum = next_max;
      }
    }

    float *out_ptr = output.data() + q * width + h * s.head_dim;
    for (size_t d = 0; d < s.head_dim; ++d) {
      out_ptr[d] = static_cast<float>(accumulator[d] / denominator);
    }
  };

  if (total_tasks <= 16 || q_tokens == 1) {
    for (size_t task_idx = 0; task_idx < total_tasks; ++task_idx)
      compute_task(task_idx);
  } else {
    auto &tm = nntrainer::ThreadManager::Global();
    tm.parallel_for(0, total_tasks, compute_task);
  }

  return output;
}

} // namespace nntrainer::kv

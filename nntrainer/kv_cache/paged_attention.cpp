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
#include <exception>
#include <limits>
#include <mutex>

// FMA and F16C are separate x86 extensions from AVX2: GCC/Clang reject their
// intrinsics unless enabled. MSVC exposes them with /arch:AVX2 without
// defining the feature macros. The NEON kernels need AArch64 (vfmaq_f32,
// vaddvq_f32, half conversion); ARMv7 NEON and other targets use scalar code.
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
#include <immintrin.h>
#define KV_ATTN_AVX2 1
#if defined(__F16C__) || defined(_MSC_VER)
#define KV_ATTN_F16C 1
#endif
#elif defined(__aarch64__)
#include <arm_neon.h>
#define KV_ATTN_NEON 1
#endif

namespace nntrainer::kv {
namespace {

/** Keys scored per online-softmax update; bounds per-thread scratch. */
constexpr size_t kChunk = 64;
/** Approximate multiply-adds below which threading costs more than it saves. */
constexpr double kParallelWork = 1 << 16;

#if defined(KV_ATTN_AVX2)
inline float hsum(__m256 v) {
  __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  x = _mm_add_ps(x, _mm_movehl_ps(x, x));
  x = _mm_add_ss(x, _mm_shuffle_ps(x, x, 0x55));
  return _mm_cvtss_f32(x);
}
#endif

#if defined(KV_ATTN_F16C)
/** Load eight binary16 values as FP32. */
inline __m256 load_ph8(const uint16_t *p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)));
}
#endif

inline float dot_f32(const float *a, const float *b, size_t n) {
  size_t i = 0;
  float sum = 0.0f;
#if defined(KV_ATTN_AVX2)
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  for (; i + 16 <= n; i += 16) {
    acc0 =
      _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                           _mm256_loadu_ps(b + i + 8), acc1);
  }
  for (; i + 8 <= n; i += 8)
    acc0 =
      _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
  sum = hsum(_mm256_add_ps(acc0, acc1));
#elif defined(KV_ATTN_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
  for (; i + 8 <= n; i += 8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  for (; i + 4 <= n; i += 4)
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
  sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#endif
  for (; i < n; ++i)
    sum += a[i] * b[i];
  return sum;
}

inline float dot_f16(const float *a, const uint16_t *b, size_t n) {
  size_t i = 0;
  float sum = 0.0f;
#if defined(KV_ATTN_F16C)
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  for (; i + 16 <= n; i += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), load_ph8(b + i), acc0);
    acc1 =
      _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), load_ph8(b + i + 8), acc1);
  }
  for (; i + 8 <= n; i += 8)
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), load_ph8(b + i), acc0);
  sum = hsum(_mm256_add_ps(acc0, acc1));
#elif defined(KV_ATTN_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
  for (; i + 8 <= n; i += 8) {
    const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(b + i));
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vcvt_f32_f16(vget_low_f16(h)));
    acc1 =
      vfmaq_f32(acc1, vld1q_f32(a + i + 4), vcvt_f32_f16(vget_high_f16(h)));
  }
  for (; i + 4 <= n; i += 4) {
    const float16x4_t h = vreinterpret_f16_u16(vld1_u16(b + i));
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vcvt_f32_f16(h));
  }
  sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#endif
  for (; i < n; ++i)
    sum += a[i] * nntrainer::compute_fp16_to_fp32(b[i]);
  return sum;
}

/** acc += w * v */
inline void axpy_f32(float *acc, float w, const float *v, size_t n) {
  size_t i = 0;
#if defined(KV_ATTN_AVX2)
  const __m256 vw = _mm256_set1_ps(w);
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(vw, _mm256_loadu_ps(v + i),
                                              _mm256_loadu_ps(acc + i)));
#elif defined(KV_ATTN_NEON)
  const float32x4_t vw = vdupq_n_f32(w);
  for (; i + 4 <= n; i += 4)
    vst1q_f32(acc + i, vfmaq_f32(vld1q_f32(acc + i), vld1q_f32(v + i), vw));
#endif
  for (; i < n; ++i)
    acc[i] += w * v[i];
}

inline void axpy_f16(float *acc, float w, const uint16_t *v, size_t n) {
  size_t i = 0;
#if defined(KV_ATTN_F16C)
  const __m256 vw = _mm256_set1_ps(w);
  for (; i + 8 <= n; i += 8)
    _mm256_storeu_ps(
      acc + i, _mm256_fmadd_ps(vw, load_ph8(v + i), _mm256_loadu_ps(acc + i)));
#elif defined(KV_ATTN_NEON)
  const float32x4_t vw = vdupq_n_f32(w);
  for (; i + 8 <= n; i += 8) {
    const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(v + i));
    vst1q_f32(acc + i,
              vfmaq_f32(vld1q_f32(acc + i), vcvt_f32_f16(vget_low_f16(h)), vw));
    vst1q_f32(acc + i + 4, vfmaq_f32(vld1q_f32(acc + i + 4),
                                     vcvt_f32_f16(vget_high_f16(h)), vw));
  }
#endif
  for (; i < n; ++i)
    acc[i] += w * nntrainer::compute_fp16_to_fp32(v[i]);
}

inline void scale_f32(float *acc, float c, size_t n) {
  for (size_t i = 0; i < n; ++i)
    acc[i] *= c;
}

/** One checked view resolved to raw pointers on the calling thread. */
struct Segment {
  const char *key;
  const char *value;
  size_t begin; /**< First logical token of this segment. */
  size_t rows;
};

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
  const size_t head_dim = s.head_dim;
  const size_t width = checkedMultiply(options.query_heads, head_dim);
  const size_t q_tokens = step.to() - step.from();
  const size_t elements = checkedMultiply(q_tokens, width);
  if (queries.size() != elements)
    throw std::invalid_argument("Attention query shape differs from KV step");
  for (float value : queries)
    if (!std::isfinite(value))
      throw std::invalid_argument("Attention query must be finite");

  // Resolve every checked view once here. Workers only touch raw pointers,
  // whose allocations stay pinned by these views until this call returns.
  const auto keys = readKey(step, layer, batch);
  const auto values = readValue(step, layer, batch);
  const size_t row_bytes = s.rowBytes();
  std::vector<Segment> segments;
  segments.reserve(keys.size());
  size_t covered = 0;
  for (size_t p = 0; p < keys.size(); ++p) {
    if (keys[p].sizeBytes() != values[p].sizeBytes())
      throw std::logic_error("KV key/value segmentation differs");
    const size_t rows = keys[p].sizeBytes() / row_bytes;
    segments.push_back({static_cast<const char *>(keys[p].data()),
                        static_cast<const char *>(values[p].data()), covered,
                        rows});
    covered += rows;
  }
  if (covered != step.to())
    throw std::logic_error("KV views do not cover the attended history");

  std::vector<float> output(elements);
  const float scale =
    options.scale
      ? options.scale
      : static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
  const float softcap = options.logit_softcap;
  const bool has_sinks = !options.sinks.empty();
  const size_t group = options.query_heads / s.num_kv_heads;
  const size_t kv_head_bytes = head_dim * s.elementBytes();
  const bool is_f32 = (s.encoding == Encoding::Float32);

  // Query heads sharing one KV head run in the same task so every K/V row is
  // loaded once per group. The group is split only as far as needed to give
  // every thread work (decode has a single query token).
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t threads = std::max<size_t>(1, tm.getComputeThreadCount());
  const size_t base_tasks = q_tokens * s.num_kv_heads;
  size_t splits = 1;
  while (base_tasks * splits < threads && splits < group) {
    size_t next = splits + 1;
    while (group % next)
      ++next;
    splits = next;
  }
  const size_t heads = group / splits;
  const size_t per_token = s.num_kv_heads * splits;
  const size_t tasks = q_tokens * per_token;

  std::mutex failure_lock;
  std::exception_ptr failure;

  auto run = [&](size_t task) {
    try {
      const size_t q = task / per_token;
      const size_t kv_head = (task % per_token) / splits;
      const size_t h0 = kv_head * group + (task % splits) * heads;
      const size_t end = options.causal ? step.from() + q + 1 : step.to();
      const size_t start =
        options.window && end > options.window ? end - options.window : 0;
      const float *q_base = queries.data() + q * width + h0 * head_dim;
      const size_t kv_offset = kv_head * kv_head_bytes;

      thread_local std::vector<float> scratch;
      scratch.resize(heads * (head_dim + kChunk + 2));
      float *acc = scratch.data();              // [heads, head_dim]
      float *scores = acc + heads * head_dim;   // [heads, kChunk]
      float *row_max = scores + heads * kChunk; // [heads]
      float *row_sum = row_max + heads;         // [heads]
      std::fill(acc, acc + heads * head_dim, 0.0f);
      for (size_t j = 0; j < heads; ++j) {
        // A sink is one extra denominator-only logit with no value row.
        row_max[j] = has_sinks ? options.sinks[h0 + j]
                               : -std::numeric_limits<float>::infinity();
        row_sum[j] = has_sinks ? 1.0f : 0.0f;
      }

      // First segment that contains `start`; windows skip whole pages.
      size_t seg = static_cast<size_t>(
        std::upper_bound(
          segments.begin(), segments.end(), start,
          [](size_t pos, const Segment &item) { return pos < item.begin; }) -
        segments.begin() - 1);
      for (size_t pos = start; pos < end;) {
        const Segment &segment = segments[seg];
        const size_t row0 = pos - segment.begin;
        const size_t n = std::min({end - pos, segment.rows - row0, kChunk});
        const char *k_rows = segment.key + row0 * row_bytes + kv_offset;
        const char *v_rows = segment.value + row0 * row_bytes + kv_offset;

        for (size_t r = 0; r < n; ++r) {
          const char *k_row = k_rows + r * row_bytes;
          for (size_t j = 0; j < heads; ++j) {
            const float *q_head = q_base + j * head_dim;
            float score =
              is_f32
                ? dot_f32(q_head, reinterpret_cast<const float *>(k_row),
                          head_dim)
                : dot_f16(q_head, reinterpret_cast<const uint16_t *>(k_row),
                          head_dim);
            score *= scale;
            if (softcap)
              score = softcap * std::tanh(score / softcap);
            scores[j * kChunk + r] = score;
          }
        }

        for (size_t j = 0; j < heads; ++j) {
          float *chunk = scores + j * kChunk;
          float next_max = row_max[j];
          for (size_t r = 0; r < n; ++r) {
            if (!std::isfinite(chunk[r]))
              throw std::invalid_argument(
                "Attention key produces a nonfinite score");
            next_max = std::max(next_max, chunk[r]);
          }
          float sum = 0.0f;
          for (size_t r = 0; r < n; ++r) {
            chunk[r] = std::exp(chunk[r] - next_max);
            sum += chunk[r];
          }
          // exp(-inf) == 0 before the first key; acc is still zero then.
          const float rescale = std::exp(row_max[j] - next_max);
          if (rescale != 1.0f)
            scale_f32(acc + j * head_dim, rescale, head_dim);
          row_sum[j] = row_sum[j] * rescale + sum;
          row_max[j] = next_max;
        }

        for (size_t r = 0; r < n; ++r) {
          const char *v_row = v_rows + r * row_bytes;
          for (size_t j = 0; j < heads; ++j) {
            const float w = scores[j * kChunk + r];
            if (is_f32)
              axpy_f32(acc + j * head_dim, w,
                       reinterpret_cast<const float *>(v_row), head_dim);
            else
              axpy_f16(acc + j * head_dim, w,
                       reinterpret_cast<const uint16_t *>(v_row), head_dim);
          }
        }

        pos += n;
        if (row0 + n == segment.rows)
          ++seg;
      }

      for (size_t j = 0; j < heads; ++j) {
        float *out = output.data() + q * width + (h0 + j) * head_dim;
        const float *a = acc + j * head_dim;
        for (size_t d = 0; d < head_dim; ++d)
          out[d] = a[d] / row_sum[j];
      }
    } catch (...) {
      // ThreadManager workers must not throw; report on the calling thread.
      std::lock_guard<std::mutex> lock(failure_lock);
      if (!failure)
        failure = std::current_exception();
    }
  };

  // Causal prefill makes later queries more expensive; a cyclic assignment
  // balances them where ThreadManager's contiguous ranges would not.
  const double work = static_cast<double>(q_tokens) * options.query_heads *
                      static_cast<double>(step.to()) * head_dim;
  const size_t workers = std::min(threads, tasks);
  if (workers <= 1 || work < kParallelWork) {
    for (size_t task = 0; task < tasks; ++task)
      run(task);
  } else {
    tm.parallel_for(0, workers, [&](size_t worker) {
      for (size_t task = worker; task < tasks; task += workers)
        run(task);
    });
  }
  if (failure)
    std::rethrow_exception(failure);
  return output;
}

} // namespace nntrainer::kv

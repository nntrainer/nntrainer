// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   cpu_kv_q_attention.cpp
 * @date   29 Sep 2026
 * @brief  A8W8 attention over the int8 KV cache registry, on the CPU
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Per (query row, head), over cache rows [lo, hi):
 *
 *   acc[r] = sum_d Qu[d] * (K[r][d] + 128)              uint8 x uint8
 *   S[r]   = (acc[r] - 128*sum(Qu) - zp*colsum_k[r]) * s_q * log2e/sqrt(hd)
 *            * s_k[r]
 *   P      = exp2(S - max S) (two passes; the denominator is exact)
 *   per 32-row block b, per 32-dim group g:
 *   P'_g   = uint8 of P * s_v[r][g], scale = block max / 255
 *   O_g   += scale_g * (sum_r P'_g[r] * (V[r][32g..] + 128) - 128*sum(P'_g))
 *   out    = O / sum P
 *
 * The masters' 4-interleaved layouts (hexkl_kv_q.h) are the udot layouts:
 * K as [dim/4][row][4] gives a 16-byte load of 4 rows x 4 dims, V as
 * [row/4][dim][4] gives 4 dims x 4 rows, and udot by lane broadcasts the 4
 * Q dims or the 4 P' rows. Rows the mask excludes contribute P' = 0; rows
 * past the cache read as 128 (zero) with scale 1, so a 4-row group may run
 * past cache_to.
 */

#include "cpu_kv_q_attention.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

#include <hexkl_kv_q.h>
#include <thread_manager.h>

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define CPU_KV_Q_UDOT 1
#endif

namespace nntrainer {

namespace {

constexpr float kLog2e = 1.4426950408889634f;
constexpr unsigned int kBlock = 32;
/** @brief Query rows per task; a task is (chunk, head). */
constexpr unsigned int kQChunk = 16;
/** @brief Rows per append task. */
constexpr unsigned int kAppendChunk = 64;

hexkl_kv_q_table &table() {
  static hexkl_kv_q_table t; // zero-initialized: no slot in use
  return t;
}

std::mutex &table_mutex() {
  static std::mutex m;
  return m;
}

/**
 * @brief Per-row asymmetric uint8 (hvx_quant_u8's rules): range
 *        [min(x, 0), max(x, 0)] over 255 steps, zero point and values
 *        rounded to nearest even.
 */
void quant_q_row_u8(const float *x, unsigned int n, uint8_t *u, float &scale,
                    int32_t &zp, int32_t &sum) {
  float mn = 0.0f, mx = 0.0f;
  for (unsigned int i = 0; i < n; ++i) {
    mn = std::min(mn, x[i]);
    mx = std::max(mx, x[i]);
  }
  scale = 1.0f;
  zp = 0;
  sum = 0;
  if (mn == mx) {
    std::memset(u, 0, n);
    return;
  }
  scale = (mx - mn) / 255.0f;
  zp = std::min(255,
                std::max(0, static_cast<int32_t>(std::nearbyint(-mn / scale))));
  const float inv = 1.0f / scale;
  for (unsigned int i = 0; i < n; ++i) {
    const int32_t v = static_cast<int32_t>(std::nearbyint(x[i] * inv)) + zp;
    u[i] = static_cast<uint8_t>(std::min(255, std::max(0, v)));
    sum += u[i];
  }
}

/** @brief exp2 of n values in place; x is clamped below at -126. */
void exp2_inplace(float *x, unsigned int n) {
  unsigned int i = 0;
#ifdef CPU_KV_Q_UDOT
  const float32x4_t c1 = vdupq_n_f32(0.69314718f);
  const float32x4_t c2 = vdupq_n_f32(0.24022652f);
  const float32x4_t c3 = vdupq_n_f32(0.05550357f);
  const float32x4_t c4 = vdupq_n_f32(0.00961803f);
  const float32x4_t c5 = vdupq_n_f32(0.00133336f);
  const float32x4_t one = vdupq_n_f32(1.0f);
  const float32x4_t lo = vdupq_n_f32(-126.0f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vmaxq_f32(vld1q_f32(x + i), lo);
    const float32x4_t fl = vrndmq_f32(v);
    const float32x4_t f = vsubq_f32(v, fl);
    float32x4_t p = vfmaq_f32(c4, c5, f);
    p = vfmaq_f32(c3, p, f);
    p = vfmaq_f32(c2, p, f);
    p = vfmaq_f32(c1, p, f);
    p = vfmaq_f32(one, p, f);
    const int32x4_t e = vshlq_n_s32(vcvtq_s32_f32(fl), 23);
    vst1q_f32(x + i,
              vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), e)));
  }
#endif
  for (; i < n; ++i) {
    x[i] = std::exp2(std::max(x[i], -126.0f));
  }
}

struct Task {
  const hexkl_kv_q *kv;
  const float *q;
  unsigned int q_stride;
  unsigned int n_q, cache_from, cache_to, n_head_q, window;
  float softcap;
  const float *sinks;
  float *out;
  unsigned int out_stride;
};

/**
 * @brief Scores of rows [r0, r0 + 4) of kv head n against the uint8 Q row:
 *        the raw uint8 x uint8 sums, before the corrections.
 */
inline void scores4(const hexkl_kv_q *kv, unsigned int n, unsigned int r0,
                    const uint8_t *qu, int32_t acc_out[4]) {
  const unsigned int d4 = kv->head_dim / 4u;
  const size_t mr = kv->max_rows;
  const uint8_t *kt = kv->kt4 + (size_t)n * d4 * mr * 4u + (size_t)r0 * 4u;
#ifdef CPU_KV_Q_UDOT
  uint32x4_t acc = vdupq_n_u32(0);
  for (unsigned int j = 0; j < d4; j += 4) {
    const uint8x16_t qv = vld1q_u8(qu + 4u * j);
    acc = vdotq_laneq_u32(acc, vld1q_u8(kt + (size_t)(j + 0) * mr * 4u), qv, 0);
    acc = vdotq_laneq_u32(acc, vld1q_u8(kt + (size_t)(j + 1) * mr * 4u), qv, 1);
    acc = vdotq_laneq_u32(acc, vld1q_u8(kt + (size_t)(j + 2) * mr * 4u), qv, 2);
    acc = vdotq_laneq_u32(acc, vld1q_u8(kt + (size_t)(j + 3) * mr * 4u), qv, 3);
  }
  vst1q_s32(acc_out, vreinterpretq_s32_u32(acc));
#else
  for (unsigned int rr = 0; rr < 4; ++rr) {
    int32_t a = 0;
    for (unsigned int j = 0; j < d4; ++j) {
      const uint8_t *k4 = kt + (size_t)j * mr * 4u + rr * 4u;
      a += qu[4 * j] * k4[0] + qu[4 * j + 1] * k4[1] + qu[4 * j + 2] * k4[2] +
           qu[4 * j + 3] * k4[3];
    }
    acc_out[rr] = a;
  }
#endif
}

/**
 * @brief O[32 dims of group g] += scale * (sum_r P'[r] * V[r] - 128 sum P')
 *        over the 32 rows of block b (P' has 32 entries, zeros where
 *        masked).
 */
inline void pv_block(const hexkl_kv_q *kv, unsigned int n, unsigned int b,
                     unsigned int g, const uint8_t *pu, int32_t sum_pu,
                     float scale, float *o) {
  const unsigned int hd = kv->head_dim;
  const uint8_t *vb =
    kv->v4 + ((size_t)n * (kv->max_rows / 4u) + (size_t)b * 8u) * hd * 4u +
    (size_t)(32u * g) * 4u;
  int32_t acc[32];
#ifdef CPU_KV_Q_UDOT
  uint32x4_t a[8];
  for (int k = 0; k < 8; ++k) {
    a[k] = vdupq_n_u32(0);
  }
  for (unsigned int rg = 0; rg < 8; rg += 4) {
    const uint8x16_t pv = vld1q_u8(pu + 4u * rg);
    for (unsigned int rr = 0; rr < 4; ++rr) {
      const uint8_t *base = vb + (size_t)(rg + rr) * hd * 4u;
      switch (rr) {
      case 0:
        for (int k = 0; k < 8; ++k)
          a[k] = vdotq_laneq_u32(a[k], vld1q_u8(base + 16 * k), pv, 0);
        break;
      case 1:
        for (int k = 0; k < 8; ++k)
          a[k] = vdotq_laneq_u32(a[k], vld1q_u8(base + 16 * k), pv, 1);
        break;
      case 2:
        for (int k = 0; k < 8; ++k)
          a[k] = vdotq_laneq_u32(a[k], vld1q_u8(base + 16 * k), pv, 2);
        break;
      default:
        for (int k = 0; k < 8; ++k)
          a[k] = vdotq_laneq_u32(a[k], vld1q_u8(base + 16 * k), pv, 3);
        break;
      }
    }
  }
  for (int k = 0; k < 8; ++k) {
    vst1q_s32(acc + 4 * k, vreinterpretq_s32_u32(a[k]));
  }
#else
  std::memset(acc, 0, sizeof(acc));
  for (unsigned int r = 0; r < 32; ++r) {
    const uint8_t p = pu[r];
    if (!p) {
      continue;
    }
    const uint8_t *base = vb + (size_t)(r / 4u) * hd * 4u + (r & 3u);
    for (unsigned int d = 0; d < 32; ++d) {
      acc[d] += p * base[d * 4u];
    }
  }
#endif
  const int32_t corr = 128 * sum_pu;
  for (unsigned int d = 0; d < 32; ++d) {
    o[d] += scale * static_cast<float>(acc[d] - corr);
  }
}

/** @brief One (query chunk, head) task. */
void run_task(const Task &t, unsigned int q0, unsigned int q1, unsigned int h) {
  const hexkl_kv_q *kv = t.kv;
  const unsigned int hd = kv->head_dim;
  const unsigned int dt = hd / 32u;
  const unsigned int G = t.n_head_q / kv->n_head_kv;
  const unsigned int n = h / G;
  const float scale0 = kLog2e / std::sqrt(static_cast<float>(hd));
  const float *s_k = kv->s_k + (size_t)n * kv->max_rows;
  const int32_t *cs = kv->colsum_k + (size_t)n * kv->max_rows;

  static thread_local std::vector<float> S, o;
  static thread_local std::vector<uint8_t> qu;
  const unsigned int rows4 = (t.cache_to + 3u) & ~3u;
  if (S.size() < rows4) {
    S.resize(rows4);
  }
  o.resize(hd);
  qu.resize(hd);
  alignas(16) uint8_t pu[kBlock];

  for (unsigned int qi = q0; qi < q1; ++qi) {
    const unsigned int pos = t.cache_from + qi;
    const unsigned int hi = std::min(pos + 1u, t.cache_to);
    const unsigned int lo =
      (t.window != 0 && hi > t.window) ? hi - t.window : 0u;
    const float *qrow = t.q + (size_t)qi * t.q_stride + (size_t)h * hd;

    float q_scale;
    int32_t q_zp, q_sum;
    quant_q_row_u8(qrow, hd, qu.data(), q_scale, q_zp, q_sum);
    const float sf = q_scale * scale0;

    // Scores.
    const unsigned int lo4 = lo & ~3u, hi4 = (hi + 3u) & ~3u;
    float mx = -INFINITY;
    for (unsigned int r0 = lo4; r0 < hi4; r0 += 4) {
      int32_t acc[4];
      scores4(kv, n, r0, qu.data(), acc);
      for (unsigned int rr = 0; rr < 4; ++rr) {
        const unsigned int r = r0 + rr;
        const int32_t a = acc[rr] - 128 * q_sum - q_zp * cs[r];
        S[r] = static_cast<float>(a) * sf * s_k[r];
      }
    }
    if (t.softcap > 0.0f) {
      const float cap = t.softcap * kLog2e;
      for (unsigned int r = lo; r < hi; ++r) {
        S[r] = std::tanh(S[r] / cap) * cap;
      }
    }
    for (unsigned int r = lo; r < hi; ++r) {
      mx = std::max(mx, S[r]);
    }
    double den = 0.0;
    if (t.sinks) {
      const float sk = t.sinks[h] * kLog2e;
      mx = std::max(mx, sk);
      den = std::exp2(sk - mx);
    }
    for (unsigned int r = lo; r < hi; ++r) {
      S[r] -= mx;
    }
    exp2_inplace(S.data() + lo, hi - lo);
    for (unsigned int r = lo; r < hi; ++r) {
      den += S[r];
    }

    // P.V per block and group.
    std::fill(o.begin(), o.end(), 0.0f);
    for (unsigned int b = lo / kBlock; b * kBlock < hi; ++b) {
      const unsigned int blo = std::max(b * kBlock, lo);
      const unsigned int bhi = std::min(b * kBlock + kBlock, hi);
      for (unsigned int g = 0; g < dt; ++g) {
        const float *s_v = kv->s_v + ((size_t)n * dt + g) * kv->max_rows;
        float pm = 0.0f;
        for (unsigned int r = blo; r < bhi; ++r) {
          pm = std::max(pm, S[r] * s_v[r]);
        }
        const float ps = pm > 0.0f ? pm / 255.0f : 1.0f;
        const float inv = 1.0f / ps;
        std::memset(pu, 0, sizeof(pu));
        int32_t sum_pu = 0;
        for (unsigned int r = blo; r < bhi; ++r) {
          const float v = std::nearbyint(S[r] * s_v[r] * inv);
          const uint8_t u =
            static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, v)));
          pu[r - b * kBlock] = u;
          sum_pu += u;
        }
        if (sum_pu) {
          pv_block(kv, n, b, g, pu, sum_pu, ps, o.data() + 32u * g);
        }
      }
    }
    const float inv_l = static_cast<float>(1.0 / den);
    float *orow = t.out + (size_t)qi * t.out_stride + (size_t)h * hd;
    for (unsigned int d = 0; d < hd; ++d) {
      orow[d] = o[d] * inv_l;
    }
  }
}

} // namespace

int cpu_kv_q_register(unsigned int kind, unsigned int max_rows,
                      unsigned int n_head_kv, unsigned int head_dim) {
  if (kind != 0 || max_rows == 0 || n_head_kv == 0 || head_dim == 0 ||
      (head_dim % 32u) != 0 || head_dim > 256u) {
    return -1;
  }
  std::lock_guard<std::mutex> lock(table_mutex());
  uint32_t h = 0;
  const int rc = hexkl_kv_q_register(&table(), HEXKL_KV_Q8, max_rows, n_head_kv,
                                     head_dim, &h);
  return rc == 0 ? static_cast<int>(h) : -1;
}

bool cpu_kv_q_append(int handle, unsigned int row0, unsigned int n_rows,
                     unsigned int kv_stride, const uint16_t *k_rows,
                     const uint16_t *v_rows) {
  if (handle < 0 || n_rows == 0 || !k_rows || !v_rows) {
    return false;
  }
  const hexkl_kv_q *kv =
    hexkl_kv_q_get(&table(), static_cast<uint32_t>(handle));
  if (!kv || kv_stride != kv->n_head_kv * kv->head_dim ||
      row0 + n_rows > kv->max_rows) {
    return false;
  }
  // Disjoint row ranges write disjoint registry entries; no bake on the
  // host (vtcm_base NULL), so chunks are independent.
  const unsigned int n_chunks = (n_rows + kAppendChunk - 1u) / kAppendChunk;
  std::vector<int> rcs(n_chunks, 0);
  hexkl_kv_q_table *tbl = &table();
  ThreadManager::Global().parallel_for(0, n_chunks, [&](size_t c) {
    const unsigned int r0 = static_cast<unsigned int>(c) * kAppendChunk;
    const unsigned int nr = std::min(kAppendChunk, n_rows - r0);
    rcs[c] =
      hexkl_kv_q_append(tbl, static_cast<uint32_t>(handle), row0 + r0, nr,
                        k_rows + (size_t)r0 * kv_stride,
                        v_rows + (size_t)r0 * kv_stride, nullptr, nullptr);
  });
  for (int rc : rcs) {
    if (rc != 0) {
      return false;
    }
  }
  return true;
}

void cpu_kv_q_release(int handle) {
  if (handle < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(table_mutex());
  hexkl_kv_q_release(&table(), static_cast<uint32_t>(handle));
}

bool cpu_sdpa_q_kvcache(int handle, unsigned int append_row0,
                        unsigned int append_rows, unsigned int kv_stride,
                        const uint16_t *k_rows, const uint16_t *v_rows,
                        const float *q, unsigned int q_stride, unsigned int n_q,
                        unsigned int cache_from, unsigned int cache_to,
                        unsigned int n_head_q, unsigned int n_head_kv,
                        unsigned int head_dim, unsigned int window,
                        float softcap, const float *sinks, float *out,
                        unsigned int out_stride) {
  if (handle < 0 || !q || !out || n_q == 0 || n_head_kv == 0 ||
      (n_head_q % n_head_kv) != 0 || cache_to < cache_from + n_q) {
    return false;
  }
  const hexkl_kv_q *kv =
    hexkl_kv_q_get(&table(), static_cast<uint32_t>(handle));
  if (!kv || kv->n_head_kv != n_head_kv || kv->head_dim != head_dim ||
      cache_to > kv->max_rows || (kv->head_dim % 32u) != 0) {
    return false;
  }
  if (append_rows != 0 && !cpu_kv_q_append(handle, append_row0, append_rows,
                                           kv_stride, k_rows, v_rows)) {
    return false;
  }
  Task t;
  t.kv = kv;
  t.q = q;
  t.q_stride = q_stride;
  t.n_q = n_q;
  t.cache_from = cache_from;
  t.cache_to = cache_to;
  t.n_head_q = n_head_q;
  t.window = window;
  t.softcap = softcap;
  t.sinks = sinks;
  t.out = out;
  t.out_stride = out_stride;
  const unsigned int n_chunks = (n_q + kQChunk - 1u) / kQChunk;
  ThreadManager::Global().parallel_for(
    0, static_cast<size_t>(n_chunks) * n_head_q, [&](size_t i) {
      const unsigned int c = static_cast<unsigned int>(i / n_head_q);
      const unsigned int h = static_cast<unsigned int>(i % n_head_q);
      const unsigned int q0 = c * kQChunk;
      run_task(t, q0, std::min(q0 + kQChunk, n_q), h);
    });
  return true;
}

} // namespace nntrainer

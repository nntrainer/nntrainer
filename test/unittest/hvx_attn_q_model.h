// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_attn_q_model.h
 * @date   28 Sep 2026
 * @brief  CPU model of the quantized attention kernel's arithmetic
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The int pipeline on the CPU, term by term switchable, so the quantization
 * noise the kernel must carry is known separately from anything the kernel
 * gets wrong: the cache quantized by the registry's own code, Q as uint8
 * per row (hvx_quant_u8's rules), scores through fp16 (the softmax's
 * format), exact base-2 softmax over the visible positions, P scaled by the
 * per-token V scales and quantized to uint8 per (row, 32-dim group) with the
 * row max over each block of bc rows, and the quantized V accumulated
 * exactly. Header-only: the device test and host sweeps share it.
 */

#ifndef __NNTRAINER_HVX_ATTN_Q_MODEL_H__
#define __NNTRAINER_HVX_ATTN_Q_MODEL_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "hexkl_kv_q.h"
#include "hvx_attn_test_util.h"

namespace hvx_test {

/** @brief Which terms are quantized. 0 for a width means exact (f32). */
struct QModelKnobs {
  int32_t k_qmax = 127; /**< 127 int8, 7 int4, 0 exact */
  int32_t v_qmax = 127; /**< 127 int8, 7 int4, 0 exact */
  bool q_u8 = true;     /**< Q as per-row uint8, else exact */
  bool p_u8 = true;     /**< P' as per-row uint8, else exact */
  bool s_hf = true;     /**< scores rounded through fp16 */
};

/**
 * @brief Per-row asymmetric uint8, hvx_quant_u8's rules: range [min(x,0),
 *        max(x,0)] over 255 steps, zero point nearest-even, values RNE.
 */
inline void quant_q_row_u8(const float *x, uint32_t n, std::vector<uint8_t> &u,
                           float &scale, int32_t &zp) {
  float mn = 0.0f, mx = 0.0f;
  for (uint32_t i = 0; i < n; ++i) {
    mn = std::min(mn, x[i]);
    mx = std::max(mx, x[i]);
  }
  u.assign(n, 0);
  scale = 1.0f;
  zp = 0;
  if (mn == mx) {
    return;
  }
  scale = (mx - mn) / 255.0f;
  zp = std::min(255,
                std::max(0, static_cast<int32_t>(std::nearbyint(-mn / scale))));
  const float inv = 1.0f / scale;
  for (uint32_t i = 0; i < n; ++i) {
    const int32_t v = static_cast<int32_t>(std::nearbyint(x[i] * inv)) + zp;
    u[i] = static_cast<uint8_t>(std::min(255, std::max(0, v)));
  }
}

inline void model_attention_q(const QModelKnobs &kn, const AttnShape &s,
                              const std::vector<float> &q,
                              const std::vector<uint16_t> &k,
                              const std::vector<uint16_t> &v,
                              const std::vector<float> &sinks, uint32_t bc,
                              std::vector<float> &out) {
  const float LOG2E = 1.4426950408889634f;
  const uint32_t G = s.n_head_q / s.n_head_kv;
  const uint32_t qs = s.n_head_q * s.head_dim;
  const uint32_t ks = s.n_head_kv * s.head_dim;
  const uint32_t hd = s.head_dim, dt = hd / 32;
  const float scale = LOG2E / std::sqrt(static_cast<float>(hd));

  // Cache: quantized by the registry's own code, or exact (scale 1, the
  // f32 values kept in kf / vf).
  std::vector<float> kf(k.size()), vf(v.size());
  for (size_t i = 0; i < k.size(); ++i) {
    kf[i] = hf_to_f32(k[i]);
    vf[i] = hf_to_f32(v[i]);
  }
  std::vector<int8_t> kq(k.size()), vq(v.size());
  std::vector<float> sk(static_cast<size_t>(s.cache_to) * s.n_head_kv, 1.0f),
    sv(static_cast<size_t>(s.cache_to) * s.n_head_kv, 1.0f);
  std::vector<int32_t> cs(sk.size(), 0);
  for (uint32_t r = 0; r < s.cache_to; ++r) {
    for (uint32_t n = 0; n < s.n_head_kv; ++n) {
      const size_t base = static_cast<size_t>(r) * ks + n * hd;
      if (kn.k_qmax) {
        hexkl_kv_q_quant_k_row(&kf[base], hd, kn.k_qmax, &kq[base],
                               &sk[r * s.n_head_kv + n],
                               &cs[r * s.n_head_kv + n]);
      }
      if (kn.v_qmax) {
        hexkl_kv_q_quant_v_row(&vf[base], hd, kn.v_qmax, &vq[base],
                               &sv[r * s.n_head_kv + n]);
      }
    }
  }

  out.assign(static_cast<size_t>(s.n_q) * qs, 0.0f);
  std::vector<uint8_t> qu;
  std::vector<float> sc, p;
  for (uint32_t qi = 0; qi < s.n_q; ++qi) {
    const uint32_t pos = s.cache_from + qi;
    const uint32_t hi = std::min(pos + 1, s.cache_to);
    const uint32_t lo = (s.window != 0 && hi > s.window) ? hi - s.window : 0;
    for (uint32_t h = 0; h < s.n_head_q; ++h) {
      const uint32_t n = h / G;
      const float *qrow = &q[static_cast<size_t>(qi) * qs + h * hd];
      float q_scale = 1.0f;
      int32_t q_zp = 0;
      if (kn.q_u8) {
        quant_q_row_u8(qrow, hd, qu, q_scale, q_zp);
      }
      sc.assign(hi - lo, 0.0f);
      float mx = -std::numeric_limits<float>::infinity();
      for (uint32_t kk = lo; kk < hi; ++kk) {
        const size_t kb = static_cast<size_t>(kk) * ks + n * hd;
        float sf;
        if (kn.q_u8 && kn.k_qmax) {
          int32_t acc = 0;
          for (uint32_t d = 0; d < hd; ++d) {
            acc += static_cast<int32_t>(qu[d]) * kq[kb + d];
          }
          const float corr = static_cast<float>(acc) -
                             static_cast<float>(q_zp) *
                               static_cast<float>(cs[kk * s.n_head_kv + n]);
          sf = corr * (q_scale * scale) * sk[kk * s.n_head_kv + n];
        } else {
          double acc = 0.0;
          for (uint32_t d = 0; d < hd; ++d) {
            const float qd =
              kn.q_u8 ? q_scale * (static_cast<float>(qu[d]) - q_zp) : qrow[d];
            const float kd =
              kn.k_qmax ? sk[kk * s.n_head_kv + n] * kq[kb + d] : kf[kb + d];
            acc += static_cast<double>(qd) * kd;
          }
          sf = static_cast<float>(acc) * scale;
        }
        if (kn.s_hf) {
          sf = hf_to_f32(f32_to_hf(sf));
        }
        if (s.softcap > 0.0f) {
          const float cap = s.softcap * LOG2E;
          sf = std::tanh(sf / cap) * cap;
        }
        sc[kk - lo] = sf;
        mx = std::max(mx, sf);
      }
      if (s.use_sink) {
        mx = std::max(mx, sinks[h] * LOG2E);
      }
      double den = s.use_sink ? std::exp2(sinks[h] * LOG2E - mx) : 0.0;
      p.assign(hi - lo, 0.0f);
      for (uint32_t i = 0; i < hi - lo; ++i) {
        p[i] = static_cast<float>(std::exp2(sc[i] - mx));
        den += p[i];
      }
      float *orow = &out[static_cast<size_t>(qi) * qs + h * hd];
      // One P quantization per block: V's scale is per row, not per
      // group, so the same P' serves every group.
      for (uint32_t b0 = lo - lo % bc; b0 < hi; b0 += bc) {
        const uint32_t blo = std::max(b0, lo), bhi = std::min(b0 + bc, hi);
        float pm = 0.0f;
        for (uint32_t kk = blo; kk < bhi; ++kk) {
          pm = std::max(pm, p[kk - lo] * sv[kk * s.n_head_kv + n]);
        }
        const float ps = pm > 0.0f ? pm / 255.0f : 1.0f;
        for (uint32_t g = 0; g < dt; ++g) {
          double acc[32] = {0.0};
          for (uint32_t kk = blo; kk < bhi; ++kk) {
            const float pp = p[kk - lo] * sv[kk * s.n_head_kv + n];
            const float pq =
              kn.p_u8
                ? ps * std::min(255.0f, std::max(0.0f, std::nearbyint(pp / ps)))
                : pp;
            const size_t vb = static_cast<size_t>(kk) * ks + n * hd + 32 * g;
            for (uint32_t d = 0; d < 32; ++d) {
              const float vd =
                kn.v_qmax ? static_cast<float>(vq[vb + d]) : vf[vb + d];
              acc[d] += static_cast<double>(pq) * vd;
            }
          }
          for (uint32_t d = 0; d < 32; ++d) {
            orow[32 * g + d] += static_cast<float>(acc[d]);
          }
        }
      }
      const float inv_l = static_cast<float>(1.0 / den);
      for (uint32_t d = 0; d < hd; ++d) {
        orow[d] *= inv_l;
      }
    }
  }
}

/** @brief The kernel's arithmetic for a registry kind. */
inline QModelKnobs knobs_for_kind(uint32_t kind) {
  QModelKnobs kn;
  kn.k_qmax = hexkl_kv_q_qmax(static_cast<hexkl_kv_q_kind>(kind));
  kn.v_qmax = kn.k_qmax;
  return kn;
}

} // namespace hvx_test

#endif /* __NNTRAINER_HVX_ATTN_Q_MODEL_H__ */

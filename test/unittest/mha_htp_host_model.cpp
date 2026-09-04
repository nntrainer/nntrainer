// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file mha_htp_host_model.cpp
 * @date 07 Aug 2026
 * @brief Host-side executable specification of the fused HTP attention loop
 * @see https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug No known bugs.
 *
 * mha_htp_host_model.h for the contract and for the five load-bearing
 * properties this file exists to pin down. */

#include "mha_htp_host_model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

void rope_u8_ref(const uint8_t *x, uint8_t *y, uint32_t n_rows, uint32_t width,
                 uint32_t dim, const float *s_in, const int32_t *zp_in,
                 const uint8_t *cos_u8, const uint8_t *sin_u8, float cos_scale,
                 int32_t cos_zp, float sin_scale, int32_t sin_zp, float s_out,
                 int32_t zp_out, uint32_t *n_saturated) {
  const uint32_t half = dim / 2u;
  *n_saturated = 0u;
  /** memcpy(p, p, n) with x == y is technically UB (the standard's "shall
  not overlap" is written without a same-pointer exception), even though
  every real implementation treats it as a no-op. y may alias x per the
  kernel's contract, so this has to be guarded rather than relied on. */
  if (y != x) {
    std::memcpy(y, x, (size_t)n_rows * width);
  }

  for (uint32_t m = 0; m < n_rows; ++m) {
    const uint8_t *row = x + (size_t)m * width;
    uint8_t *out = y + (size_t)m * width;
    const double inv = (double)s_in[m] / (double)s_out;
    const int32_t z = zp_in[m];
    const uint8_t *cq = cos_u8 + (size_t)m * half;
    const uint8_t *sq = sin_u8 + (size_t)m * half;

    for (uint32_t w = 0; w + dim <= width; w += dim) {
      for (uint32_t k = 0; k < half; ++k) {
        const double a = (double)((int32_t)row[w + k] - z);
        const double b = (double)((int32_t)row[w + k + half] - z);
        const double c = ((double)cq[k] - (double)cos_zp) * (double)cos_scale;
        const double s = ((double)sq[k] - (double)sin_zp) * (double)sin_scale;
        const double t0 = a * c - b * s;
        const double t1 = a * s + b * c;
        const double q0 = std::nearbyint(t0 * inv) + (double)zp_out;
        const double q1 = std::nearbyint(t1 * inv) + (double)zp_out;

        if (q0 < 0.0 || q0 > 255.0) {
          ++*n_saturated;
        }
        if (q1 < 0.0 || q1 > 255.0) {
          ++*n_saturated;
        }
        out[w + k] = (uint8_t)std::max(0.0, std::min(255.0, q0));
        out[w + k + half] = (uint8_t)std::max(0.0, std::min(255.0, q1));
      }
    }
  }
}

namespace {

/**
 * @brief Per-row asymmetric uint8 dynamic activation quantization.
 *
 * Mirrors hvx_quant_rows_u8_params + hvx_quant_pack_u8_ah exactly, including
 * their rounding: nearbyintf is round-to-nearest-EVEN under the default FP
 * environment, which is what hvx_sf_to_w_rne does. roundf (ties away) would
 * disagree on exact.5 codes and quietly widen the error against the device.
 *
 * ponytail: m_pad = ROUND_UP(M, 64) is not modelled. The padding rows carry
 * synthetic scale 1 / zp 0 and hvx_dequant_i32_to_f32 drops them at m_valid,
 * so they cannot reach the output they cost HMX row utilisation, not
 * accuracy. Model them if a device mismatch ever points at the pad rows. */
void quant_rows_u8(const float *x, uint32_t m_valid, uint32_t k, float *scale,
                   int32_t *zp, uint8_t *u) {
  for (uint32_t m = 0; m < m_valid; ++m) {
    const float *row = x + (size_t)m * k;
    float mn = row[0];
    float mx = row[0];
    for (uint32_t i = 1; i < k; ++i) {
      mn = std::min(mn, row[i]);
      mx = std::max(mx, row[i]);
    }
    const float rmin = mn < 0.0f ? mn : 0.0f;
    const float rmax = mx > 0.0f ? mx : 0.0f;

    float s = 1.0f;
    int32_t z = 0;
    if (rmin != rmax) {
      s = (rmax - rmin) / 255.0f;
      z = (int32_t)std::nearbyintf(-rmin / s);
      z = std::min(255, std::max(0, z));
    }
    scale[m] = s;
    zp[m] = z;

    const float inv_s = 1.0f / s;
    for (uint32_t i = 0; i < k; ++i) {
      int32_t v = (int32_t)std::nearbyintf(row[i] * inv_s) + z;
      u[(size_t)m * k + i] = (uint8_t)std::min(255, std::max(0, v));
    }
  }
}

/**
 * @brief One u8 x iX matmul plus its dequant, mirroring the contract of
 * hexkl_mm_u8iX_layer_run followed by hvx_dequant_i32_to_f32:
 *
 * out[m][n] = (acc[m][n] - act_zp[m]*colsum_w[n]) * act_scale[m] * w_scale[n]
 *
 * with bias omitted attention registers no bias. @a w is row-major [K][N].
 * @a accumulate adds into @a out instead of overwriting it, which is what
 * PHASE C needs across blocks (property 3). */
void mm_u8_iX_dequant(const uint8_t *act_u, const float *act_scale,
                      const int32_t *act_zp, const int8_t *w,
                      const float *w_scale, const int32_t *colsum_w, uint32_t M,
                      uint32_t K, uint32_t N, bool accumulate, float *out) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (uint32_t k = 0; k < K; ++k) {
        acc +=
          (int32_t)act_u[(size_t)m * K + k] * (int32_t)w[(size_t)k * N + n];
      }
      const float v =
        (float)(acc - act_zp[m] * colsum_w[n]) * act_scale[m] * w_scale[n];
      if (accumulate) {
        out[(size_t)m * N + n] += v;
      } else {
        out[(size_t)m * N + n] = v;
      }
    }
  }
}

} // namespace

void mha_htp_host_scores(uint32_t kv_len, uint32_t nch, uint32_t head_dim,
                         uint32_t T, uint32_t M, uint32_t head,
                         hexkl_w_width w_k, const float *q_band,
                         const uint16_t *k_cache, float *out_s) {
  const uint32_t n_blocks = (kv_len + T - 1) / T;

  std::vector<float> q_scale(M, 1.0f);
  std::vector<int32_t> q_zp(M, 0);
  std::vector<uint8_t> q_u((size_t)M * head_dim, 0);
  quant_rows_u8(q_band, M, head_dim, q_scale.data, q_zp.data, q_u.data);

  std::vector<int8_t> kt_rm((size_t)head_dim * T);
  std::vector<float> kt_scale(T);
  std::vector<int32_t> kt_cs(T);

  for (uint32_t j = 0; j < n_blocks; ++j) {
    const uint32_t valid = std::min(T, kv_len - j * T);
    hexkl_kvq_pack_kt_block(k_cache + (size_t)j * T * nch * head_dim, valid, T,
                            head_dim, nch, head, w_k, kt_rm.data, kt_scale.data,
                            kt_cs.data);
    /** One ops->run call on device: handles = this head's Kt blocks, and
     * out_cat comes back BLOCK-MAJOR, [n_blocks][M][T]. Do not flatten it * the
     * block-major layout is the KV tiling (property 1). */
    mm_u8_iX_dequant(q_u.data, q_scale.data, q_zp.data, kt_rm.data,
                     kt_scale.data, kt_cs.data, M, head_dim, T, false,
                     out_s + (size_t)j * M * T);
  }
}

MhaSoftmaxOut mha_softmax_blocked_ref(const std::vector<const float *> &seg,
                                      uint32_t n_seg, uint32_t T, uint32_t M,
                                      float scale,
                                      const std::vector<uint32_t> &begin,
                                      const std::vector<uint32_t> &end,
                                      const float *sink) {
  MhaSoftmaxOut o;
  o.p.assign((size_t)n_seg * M * T, 0.0f);
  o.l.assign(M, 0.0f);

  const uint32_t kv_total = n_seg * T;

  for (uint32_t m = 0; m < M; ++m) {
    const uint32_t lo = std::min(begin[m], kv_total);
    const uint32_t hi = std::min(end[m], kv_total);

    /** PASS 1 maximum over VALID positions only.
     *
     * Masked positions are excluded from the MAXIMUM, not merely from the
     * sum, and that is not a tidiness point. The HVX exp this mirrors
     * (hvx_exp_f32.h) is UNDEFINED above +87.3 and has no overflow clamp. A
     * masked lane holds whatever the padded weight tile left in the
     * accumulator; if it skips the max pass but still reaches exp, the result
     * is undefined not a small numerical error that a tolerance would
     * catch. Keeping the two passes over the SAME lane set is what makes
     * x - max <= 0 true by construction.
     *
     * The sink is deliberately NOT in this maximum: leaving it out is what
     * makes max(p[m][*]) exactly 1.0f, which is the whole reason PHASE C's
     * per-block u8 quantization of P uses its codes well and lands zp = 0
     * naturally. The sink is a single scalar exp
     * per row, not a vector lane, so it is not subject to the domain hazard
     * above. */
    float mx = 0.0f;
    bool any = false;
    for (uint32_t j = 0; j < n_seg; ++j) {
      for (uint32_t t = 0; t < T; ++t) {
        const uint32_t pos = j * T + t;
        if (pos < lo || pos >= hi) {
          continue;
        }
        const float v = seg[j][(size_t)m * T + t] * scale;
        if (!any || v > mx) {
          mx = v;
          any = true;
        }
      }
    }
    if (!any) {
      mx = (sink != nullptr) ? sink[m] : 0.0f;
    }

    /** PASS 2 unnormalized exp over the same lane set, in kv order.
     *
     * The j-then-t walk visits kv positions strictly ascending regardless of
     * how the band was split, which is what makes the segmentation bitwise
     * invisible: identical addends in identical order. */
    float sum = 0.0f;
    for (uint32_t j = 0; j < n_seg; ++j) {
      for (uint32_t t = 0; t < T; ++t) {
        const uint32_t pos = j * T + t;
        if (pos < lo || pos >= hi) {
          continue; /* p stays 0: masked lanes contribute nothing to P.V */
        }
        const float e = std::exp(seg[j][(size_t)m * T + t] * scale - mx);
        o.p[(size_t)j * M * T + (size_t)m * T + t] = e;
        sum += e;
      }
    }

    const float sink_term = (sink != nullptr) ? std::exp(sink[m] - mx) : 0.0f;
    o.l[m] = sum + sink_term;

    /** Self-check: re-walk the EMITTED p and confirm it adds back up to the
     * denominator minus the sink term. Independent of the accumulator above,
     * so a wrong block-major offset is caught here rather than surfacing as a
     * plausible-looking wrong O three phases later. */
    float check = 0.0f;
    for (uint32_t j = 0; j < n_seg; ++j) {
      for (uint32_t t = 0; t < T; ++t) {
        check += o.p[(size_t)j * M * T + (size_t)m * T + t];
      }
    }
    assert(std::fabs(check - (o.l[m] - sink_term)) <=
           1e-5f * std::max(1.0f, sum));
    (void)check;
  }

  return o;
}

void mha_htp_host_forward(uint32_t n_query, uint32_t kv_from, uint32_t nch,
                          uint32_t gqa, uint32_t head_dim, uint32_t T,
                          uint32_t M_band, bool is_causal, uint32_t window,
                          const float *sink, hexkl_w_width w_k,
                          hexkl_w_width w_v, const float *q,
                          const uint16_t *k_cache, const uint16_t *v_cache,
                          float *out) {
  assert(M_band <= T); /* property 5 */
  assert(T % 32 == 0); /* hexkl_mm_u8iX_plan enforces N % 32 == 0 */

  const uint32_t kv_len = kv_from + n_query;
  const uint32_t n_blocks = (kv_len + T - 1) / T;
  const uint32_t nHq = nch * gqa;
  const uint32_t M = n_query * gqa; /* GQA row fold, per kv_head */
  const float scale = 1.0f / std::sqrt((float)head_dim);

  /** Per-block KV weights for one head. On device this is append-time work
   * done once per token (the incremental WH bake), not per forward; here it
   * is hoisted out of the band loop for the same reason the band loop must
   * see the same constants every band. */
  std::vector<std::vector<int8_t>> kt_rm(n_blocks), v_rm(n_blocks);
  std::vector<std::vector<float>> kt_scale(n_blocks), v_scale(n_blocks);
  std::vector<std::vector<int32_t>> kt_cs(n_blocks), v_cs(n_blocks);

  std::vector<float> qb, s_band, o_band, q_scale, p_scale;
  std::vector<int32_t> q_zp, p_zp;
  std::vector<uint8_t> q_u, p_u;
  std::vector<uint32_t> begin, end;
  std::vector<float> sink_row;
  std::vector<const float *> seg(n_blocks);

  for (uint32_t n = 0; n < nch; ++n) {
    for (uint32_t j = 0; j < n_blocks; ++j) {
      const uint32_t valid = std::min(T, kv_len - j * T);
      const size_t off = (size_t)j * T * nch * head_dim;

      kt_rm[j].assign((size_t)head_dim * T, 0);
      kt_scale[j].assign(T, 1.0f);
      kt_cs[j].assign(T, 0);
      hexkl_kvq_pack_kt_block(k_cache + off, valid, T, head_dim, nch, n, w_k,
                              kt_rm[j].data, kt_scale[j].data, kt_cs[j].data);

      v_rm[j].assign((size_t)T * head_dim, 0);
      v_scale[j].assign(head_dim, 1.0f);
      v_cs[j].assign(head_dim, 0);
      hexkl_kvq_pack_v_block(v_cache + off, valid, T, head_dim, nch, n, w_v,
                             v_rm[j].data, v_scale[j].data, v_cs[j].data);
    }

    for (uint32_t b0 = 0; b0 < M; b0 += M_band) {
      const uint32_t mb = std::min(M_band, M - b0);

      /* ---- PHASE A: scores, streaming the K blocks ------------------- */
      qb.assign((size_t)mb * head_dim, 0.0f);
      for (uint32_t m = 0; m < mb; ++m) {
        const uint32_t i = (b0 + m) / gqa;
        const uint32_t g = (b0 + m) % gqa;
        const float *src = q + ((size_t)i * nHq + n * gqa + g) * head_dim;
        std::copy(src, src + head_dim, qb.begin + (size_t)m * head_dim);
      }
      q_scale.assign(mb, 1.0f);
      q_zp.assign(mb, 0);
      q_u.assign((size_t)mb * head_dim, 0);
      quant_rows_u8(qb.data, mb, head_dim, q_scale.data, q_zp.data, q_u.data);

      /** One ops->run call on device: handles = this head's Kt blocks, and
       * out_cat comes back BLOCK-MAJOR, [n_blocks][mb][T]. Do not flatten
       * it the block-major layout is the KV tiling (property 1). */
      s_band.assign((size_t)n_blocks * mb * T, 0.0f);
      mha_htp_host_scores(kv_len, nch, head_dim, T, mb, n, w_k, qb.data,
                          k_cache, s_band.data);

      /* ---- PHASE B: one masked softmax pass over the whole band ------- */
      begin.assign(mb, 0);
      end.assign(mb, 0);
      sink_row.assign(mb, 0.0f);
      for (uint32_t m = 0; m < mb; ++m) {
        const uint32_t i = (b0 + m) / gqa;
        const uint32_t g = (b0 + m) % gqa;
        const uint32_t p = kv_from + i;
        const uint32_t e = is_causal ? (p + 1) : kv_len;
        end[m] = e;
        begin[m] = (window != 0 && window < e) ? (e - window) : 0;
        if (sink != nullptr) {
          sink_row[m] = sink[n * gqa + g];
        }
      }
      for (uint32_t j = 0; j < n_blocks; ++j) {
        seg[j] = s_band.data + (size_t)j * mb * T;
      }
      const MhaSoftmaxOut sm =
        mha_softmax_blocked_ref(seg, n_blocks, T, mb, scale, begin, end,
                                sink != nullptr ? sink_row.data : nullptr);

      /* ---- PHASE C: output, streaming the V blocks -------------------- */
      o_band.assign((size_t)mb * head_dim, 0.0f);
      p_scale.assign(mb, 1.0f);
      p_zp.assign(mb, 0);
      p_u.assign((size_t)mb * T, 0);
      for (uint32_t j = 0; j < n_blocks; ++j) {
        /** P is quantized PER BLOCK (property 2): after PHASE B the row max is
         * exactly 1.0 and every value is in (0, 1], so each block's own scale
         * uses the u8 codes well and zp lands at 0 naturally. */
        quant_rows_u8(sm.p.data + (size_t)j * mb * T, mb, T, p_scale.data,
                      p_zp.data, p_u.data);
        /** Accumulate in f32 across blocks (property 3): each block carries
         * its own dequant constants, so an integer accumulation would apply
         * block 0's scale to every block. */
        mm_u8_iX_dequant(p_u.data, p_scale.data, p_zp.data, v_rm[j].data,
                         v_scale[j].data, v_cs[j].data, mb, T, head_dim, true,
                         o_band.data);
      }

      /** The 1/l normalization happens ONCE, here, after every block has been
       * accumulated (property 4) not as a third softmax pass. */
      for (uint32_t m = 0; m < mb; ++m) {
        const uint32_t i = (b0 + m) / gqa;
        const uint32_t g = (b0 + m) % gqa;
        const float inv_l = (sm.l[m] > 0.0f) ? (1.0f / sm.l[m]) : 0.0f;
        float *dst = out + ((size_t)i * nHq + n * gqa + g) * head_dim;
        for (uint32_t d = 0; d < head_dim; ++d) {
          dst[d] = o_band[(size_t)m * head_dim + d] * inv_l;
        }
      }
    }
  }
}

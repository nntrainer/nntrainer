// SPDX-License-Identifier: Apache-2.0
/** @file htp_decode_kv_convert.h
 * @brief Convert immutable tiled decode INT8 KV to native prefill FP16 rows.
 */
#ifndef NNTR_HTP_DECODE_KV_CONVERT_H
#define NNTR_HTP_DECODE_KV_CONVERT_H
#include <cstddef>
#include <cstdint>
#include <fp16.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace nntrainer {
/** Scales are per (KV head, dimension); K tiles are transposed, V tiles are
 * row-major. Padded rows are excluded from the destination. */
inline void decodeKVToPrefill(const int8_t *key, const int8_t *value,
                              const float *sk, const float *sv, unsigned rows,
                              unsigned heads, unsigned dim, uint16_t *out_k,
                              uint16_t *out_v, unsigned first_row = 0) {
  const size_t padded = (size_t(rows) + 63u) / 64u * 64u;
  const size_t width = size_t(heads) * dim;
  for (unsigned r = first_row; r < rows; ++r)
    for (unsigned h = 0; h < heads; ++h) {
      const auto *kt =
        key + size_t(h) * padded * dim + size_t(r / 64u) * dim * 64u + r % 64u;
      const auto *v = value + (size_t(h) * padded + r) * dim;
      const size_t col = size_t(h) * dim;
      auto *dk = out_k + size_t(r - first_row) * width + col;
      auto *dv = out_v + size_t(r - first_row) * width + col;
      unsigned d = 0;
#if defined(__aarch64__)
      auto widen = [](int8x8_t q, const float *scale) {
        const int16x8_t s = vmovl_s8(q);
        const auto lo = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(s))),
                                  vld1q_f32(scale));
        const auto hi = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(s))),
                                  vld1q_f32(scale + 4));
        return vreinterpretq_u16_f16(
          vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
      };
      for (; d + 8u <= dim; d += 8u) {
        int8_t k[8];
        for (unsigned i = 0; i < 8u; ++i)
          k[i] = kt[size_t(d + i) * 64u];
        vst1q_u16(dk + d, widen(vld1_s8(k), sk + col + d));
        vst1q_u16(dv + d, widen(vld1_s8(v + d), sv + col + d));
      }
#endif
      for (; d < dim; ++d) {
        dk[d] = compute_fp32_to_fp16(float(kt[size_t(d) * 64u]) * sk[col + d]);
        dv[d] = compute_fp32_to_fp16(float(v[d]) * sv[col + d]);
      }
    }
}
} // namespace nntrainer
#endif

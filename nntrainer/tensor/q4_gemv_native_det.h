// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   q4_gemv_native_det.h
 * @date   30 Sep 2026
 * @brief  [#194 L1] The native M=1 Q4_0 FC as a scalar spec: a vector
 *         Q8_0 quantizer with no scalar divide and a per-block
 *         multiply-then-add chain with no emulated FMA (htp_moe_ppl only)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * WHY THIS EXISTS
 *
 * On htp_moe_ppl the FC need not be the Android CPU's bits (plan 194: the
 * gate is decode PPL); q4_gemv_cpu_det.h's order costs ~85 packets a
 * 32-column step on HVX (the emulated FMA), this one ~12. This header is
 * what hvx_q4m1_prep_vec and hvx_q4m1_gemv_groups_native compute, bit for
 * bit (q4_gemv_host_check.c holds them to it on hvx_emu/, the device gtest
 * HvxFcQ4.NativeMatchesSpec on silicon), and the host check measures how
 * far it sits from the CPU order (SNR).
 *
 * THE QUANTIZER, per block of 32 (every op one IEEE f32 op, RN):
 *   amax = max |x|                          (bits, exact)
 *   amax < 2^-60 or not finite: d = 0, id = 0 (q = 0)
 *   d  = RN(amax * RN(1/127))               (not the CPU's true divide)
 *   id = 1/d by three Newton steps from the integer seed 0x7EF311C3 - d:
 *        r = RN(r * RN(2 - RN(d * r)))      (within ~1 ulp of 1/d)
 *   q  = fcvtns(RN(x * id))                 (the CPU's own rounding)
 *   the block stores f16(d), RN, subnormals kept (as the CPU stores it)
 *
 * THE GEMV, per output column, acc = +0, for b = 0 .. K/32 - 1 in order:
 *   isum = sum (w - 8) q                    (exact int32, vrmpy)
 *   s    = f16->f32(d_a) * f16->f32(d_w)    (exact: 11 x 11 bits)
 *   acc  = RN(acc + RN(isum * s))           (two roundings; the CPU's fma
 *                                            has one)
 *
 * DOMAIN. Finite inputs with block amax below ~8.3e6 (d must be a finite
 * f16). A block under 2^-60 contributes 0 here (the CPU keeps it); no
 * decode activation block is that small.
 */

#ifndef __NNTRAINER_Q4_GEMV_NATIVE_DET_H__
#define __NNTRAINER_Q4_GEMV_NATIVE_DET_H__

#include "q4_gemv_cpu_det.h"

/** @brief amax bits below this quantize to 0 (2^-60). */
#define Q4N_TINY_BITS 0x21800000u
/** @brief The reciprocal's integer seed. */
#define Q4N_RECIP_SEED 0x7EF311C3u
/** @brief RN(1 / 127) as f32 bits. */
#define Q4N_INV127_BITS 0x3C010204u

/** @brief d and id of one block from its amax bits (the quantizer's
 *         scalar steps, shared with the kernel's vector pass). */
static inline void q4n_block_scale(uint32_t amax_bits, float *d, float *id) {
  if (amax_bits < Q4N_TINY_BITS || amax_bits >= 0x7f800000u) {
    *d = 0.0f;
    *id = 0.0f;
    return;
  }
  const float dd =
    cpu_det_mul(cpu_det_float(amax_bits), cpu_det_float(Q4N_INV127_BITS));
  float r = cpu_det_float(Q4N_RECIP_SEED - cpu_det_bits(dd));
  for (int i = 0; i < 3; ++i) {
    r = cpu_det_mul(r, cpu_det_sub(2.0f, cpu_det_mul(dd, r)));
  }
  *d = dd;
  *id = r;
}

/** @brief The native Q8_0 quantizer: K floats (K % 32 == 0) into K int8
 *         and K / 32 f16 scales. */
static inline void q8_0_quant_native_det(const float *x, uint32_t K, int8_t *q,
                                         uint16_t *d) {
  for (uint32_t b = 0; b < K / Q4_CPU_QK; ++b) {
    const float *xb = x + (size_t)b * Q4_CPU_QK;
    uint32_t amax = 0u;
    for (uint32_t j = 0; j < Q4_CPU_QK; ++j) {
      const uint32_t v = cpu_det_bits(xb[j]) & 0x7fffffffu;
      amax = v > amax ? v : amax;
    }
    float db, id;
    q4n_block_scale(amax, &db, &id);
    for (uint32_t j = 0; j < Q4_CPU_QK; ++j) {
      q[(size_t)b * Q4_CPU_QK + j] =
        (int8_t)(uint8_t)(cpu_det_fcvtns(cpu_det_mul(xb[j], id)) & 0xff);
    }
    d[b] = cpu_det_f32_to_f16(db);
  }
}

/** @brief The native chain per column over canonical block_q4_0 rows
 *         (q4_gemv_cpu_det's arguments). */
static inline void q4_gemv_native_det(const uint8_t *w, const int8_t *q,
                                      const uint16_t *da, uint32_t K,
                                      uint32_t N, float *y) {
  const uint32_t nb = K / Q4_CPU_QK;
  for (uint32_t n = 0; n < N; ++n) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; ++b) {
      const uint8_t *blk = w + ((size_t)n * nb + b) * Q4_CPU_BLOCK_BYTES;
      const int32_t isum =
        q4_cpu_block_isum(blk + 2, q + (size_t)b * Q4_CPU_QK);
      const float s = cpu_det_mul(cpu_det_f16_to_f32(da[b]),
                                  cpu_det_f16_to_f32(q4_cpu_block_d(blk)));
      acc = cpu_det_add(acc, cpu_det_mul((float)isum, s));
    }
    y[n] = acc;
  }
}

#endif /* __NNTRAINER_Q4_GEMV_NATIVE_DET_H__ */

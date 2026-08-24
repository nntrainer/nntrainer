// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Portions of this file are derived from llama.cpp
 * (https://github.com/ggml-org/llama.cpp), licensed under the MIT License.
 * Copyright (c) Contributors to llama.cpp
 *
 * Modified by Sungsik Kong, 2025: Adapted for CPU backend integration
 *
 * @file   nntr_ggml_impl_fallback.cpp
 * @date   9 December 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Fallback (scalar C) implementations of GGML functions
 *
 * This file provides portable C implementations that work on any platform.
 * For optimized SIMD implementations, use architecture-specific files:
 * - nntr_ggml_impl_neon.cpp for ARM NEON (armv8.2-a)
 * - nntr_ggml_impl_sve.cpp for ARM SVE (armv9.2-a)
 * - nntr_ggml_impl_avx.cpp for x86_64 AVX/AVX2
 */

#include <algorithm>
#include <assert.h>
#include <cstring>
#include <math.h>
#include <stddef.h>
#include <stdexcept>
#include <stdint.h>

#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_common.h>
#include <nntr_ggml_impl_utils.h>

//============================================================================
// Helper functions for block packing
//============================================================================

static block_q4_0x4 nntr_make_block_q4_0x4(block_q4_0 *in,
                                           unsigned int blck_size_interleave) {
  block_q4_0x4 out;

  for (int i = 0; i < 4; i++) {
    out.d[i] = in[i].d;
  }

  const int end = Q4_0 * 2 / blck_size_interleave;

  if (blck_size_interleave == 8) {
    const uint64_t xor_mask = 0x8888888888888888ULL;
    for (int i = 0; i < end; ++i) {
      int src_id = i % 4;
      int src_offset = (i / 4) * blck_size_interleave;
      int dst_offset = i * blck_size_interleave;

      uint64_t elems;
      // Using memcpy to avoid unaligned memory accesses
      memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
      elems ^= xor_mask;
      memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }
  } else if (blck_size_interleave == 4) {
    const uint32_t xor_mask = 0x88888888;
    for (int i = 0; i < end; ++i) {
      int src_id = i % 4;
      int src_offset = (i / 4) * blck_size_interleave;
      int dst_offset = i * blck_size_interleave;

      uint32_t elems;
      memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint32_t));
      elems ^= xor_mask;
      memcpy(&out.qs[dst_offset], &elems, sizeof(uint32_t));
    }
  } else {
    assert(false);
  }

  return out;
}

static block_q4_0x8 nntr_make_block_q4_0x8(block_q4_0 *in,
                                           unsigned int blck_size_interleave) {
  block_q4_0x8 out;

  for (int i = 0; i < 8; i++) {
    out.d[i] = in[i].d;
  }

  const int end = QK_0<4>() * 4 / blck_size_interleave;
  const uint64_t xor_mask = 0x8888888888888888ULL;

  for (int i = 0; i < end; ++i) {
    int src_id = i % 8;
    int src_offset = (i / 8) * blck_size_interleave;
    int dst_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
    elems ^= xor_mask;
    memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
  }

  return out;
}

static block_q8_0x4 nntr_make_block_q8_0x4(block_q8_0 *in,
                                           unsigned int blck_size_interleave) {
  block_q8_0x4 out;

  for (int i = 0; i < 4; i++) {
    out.d[i] = in[i].d;
  }

  const int end = QK8_0 * 4 / blck_size_interleave;
  for (int i = 0; i < end; ++i) {
    int src_id = i % 4;
    int src_offset = (i / 4) * blck_size_interleave;
    int dst_offset = i * blck_size_interleave;
    memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset],
           blck_size_interleave);
  }
  return out;
}

static block_q4_Kx8 make_block_q4_Kx8(block_q4_K *in,
                                      unsigned int blck_size_interleave) {
  block_q4_Kx8 out;
  // Delta(scale) and dmin values of the eight Q4_K structures are copied onto
  // the output interleaved structure
  for (int i = 0; i < 8; i++) {
    out.d[i] = in[i].data.data.d;
  }

  for (int i = 0; i < 8; i++) {
    out.dmin[i] = in[i].data.data.dmin;
  }

  const int end = QK_K * 4 / blck_size_interleave;

  // Interleave Q4_K quants by taking 8 bytes at a time
  for (int i = 0; i < end; ++i) {
    int src_id = i % 8;
    int src_offset = (i / 8) * blck_size_interleave;
    int dst_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
    memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
  }

  // The below logic is designed so as to unpack and rearrange scales and mins
  // values in Q4_K Currently the Q4_K structure has 8 scales and 8 mins packed
  // in 12 bytes ( 6 bits for each value) The output Q4_Kx8 structure has 96
  // bytes Every 12 byte is packed such that it contains scales and mins for
  // corresponding sub blocks from Q4_K structure For eg - First 12 bytes
  // contains 8 scales and 8 mins - each of first sub block from different Q4_K
  // structures
  uint8_t s[8], m[8];

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 8; j++) {
      s[j] = in[j].scales[i] & 63;
      m[j] = in[j].scales[i + 4] & 63;
    }

    out.scales[i * 12] = (s[0] & 63) + ((s[4] & 48) << 2);
    out.scales[i * 12 + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
    out.scales[i * 12 + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
    out.scales[i * 12 + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
    out.scales[i * 12 + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
    out.scales[i * 12 + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
    out.scales[i * 12 + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
    out.scales[i * 12 + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
    out.scales[i * 12 + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
    out.scales[i * 12 + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
    out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
    out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 8; j++) {
      s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
      m[j] =
        ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
    }

    out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
    out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
    out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
    out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
    out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
    out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
    out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
    out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
    out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
    out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
    out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
    out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
  }

  return out;
}

//============================================================================
// GEMV (General Matrix-Vector Multiplication) - Q4_0 4x8
//============================================================================

void nntr_gemv_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  /**
   * A32 dot-product path. aarch64 gets this kernel from
   * nntr_ggml_impl_neon.cpp; armv7l lands here instead, and Advanced SIMD in
   * ARMv8.2-A exposes the same VSDOT, so the schedule carries over unchanged.
   * Needs -march=armv8.2-a+dotprod (armv8-a rejects +dotprod on A32) and
   * -mfp16-format=ieee for the scale conversions.
   *
   * Both nibbles are lifted into the high half of a byte (<<4 and &0xf0), so
   * every product carries a factor of 16 that vcvtq_n_f32_s32(.., 4) removes
   * when the accumulator is converted. That matches the scalar `>> 4` exactly:
   * the summed pair is always a multiple of 16, so nothing is rounded away.
   */
  (void)blocklen;
  const block_q4_0x4 *vb_ptr = (const block_q4_0x4 *)vx;
  for (int c = 0; c < nc; c += ncols_interleaved) {
    const block_q8_0 *va_ptr = (const block_q8_0 *)vy;
    float32x4_t acc = vdupq_n_f32(0);

    for (int b = 0; b < nb; b++) {
      const int8_t *bq = (const int8_t *)vb_ptr->qs;
      int8x16_t b0 = vld1q_s8(bq);      // k = 0, cols 0-1
      int8x16_t b1 = vld1q_s8(bq + 16); // k = 0, cols 2-3
      int8x16_t b2 = vld1q_s8(bq + 32); // k = 1, cols 0-1
      int8x16_t b3 = vld1q_s8(bq + 48); // k = 1, cols 2-3
      float16x4_t bd = vld1_f16((const __fp16 *)vb_ptr->d);

      // Load as bytes, not as int64: block_q8_0 puts qs at offset 2, so an
      // int64_t* would let gcc emit `vld1.64 [rN:64]`, whose :64 qualifier
      // faults on a 2-byte-aligned address. aarch64 has no such constraint on
      // LD1R, which is why the kernel there can dup straight from memory.
      const int8_t *aq = (const int8_t *)va_ptr->qs;
      int8x8_t t0 = vld1_s8(aq);      // k = 0, low nibble operand
      int8x8_t t1 = vld1_s8(aq + 8);  // k = 1, low
      int8x8_t t2 = vld1_s8(aq + 16); // k = 0, high
      int8x8_t t3 = vld1_s8(aq + 24); // k = 1, high
      int8x16_t a0 = vcombine_s8(t0, t0);
      int8x16_t a1 = vcombine_s8(t1, t1);
      int8x16_t a2 = vcombine_s8(t2, t2);
      int8x16_t a3 = vcombine_s8(t3, t3);
      float16x4_t ad = vld1_dup_f16((const __fp16 *)&va_ptr->d);

      int32x4_t ret0 = vdupq_n_s32(0);
      int32x4_t ret1 = vdupq_n_s32(0);

      ret0 = vdotq_s32(ret0, b0 << 4, a0);
      ret1 = vdotq_s32(ret1, b1 << 4, a0);
      ret0 = vdotq_s32(ret0, b2 << 4, a1);
      ret1 = vdotq_s32(ret1, b3 << 4, a1);

      ret0 = vdotq_s32(ret0, b0 & 0xf0U, a2);
      ret1 = vdotq_s32(ret1, b1 & 0xf0U, a2);
      ret0 = vdotq_s32(ret0, b2 & 0xf0U, a3);
      ret1 = vdotq_s32(ret1, b3 & 0xf0U, a3);

      // { col0, col1, col2, col3 } once each column's two lanes are folded
      int32x4_t ret = vpaddq_s32(ret0, ret1);

      acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4),
                      vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
      va_ptr++;
      vb_ptr++;
    }
    vst1q_f32(s, acc);
    s += ncols_interleaved;
  }
#else

  float sumf[4];
  int sumi;

  const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);

    for (int j = 0; j < ncols_interleaved; j++)
      sumf[j] = 0.0;
    for (int l = 0; l < nb; l++) {
      for (int k = 0; k < (qk / (2 * blocklen)); k++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumi = 0;
          for (int i = 0; i < blocklen; ++i) {
            const int v0 =
              (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                   j * blocklen + i]
                       << 4);
            const int v1 =
              (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                   j * blocklen + i] &
                       0xF0);
            sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) +
                     (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >>
                    4;
          }
          sumf[j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                     nntr_compute_fp16_to_fp32(a_ptr[l].d);
        }
      }
    }
    for (int j = 0; j < ncols_interleaved; j++)
      s[x * ncols_interleaved + j] = sumf[j];
  }
#endif
}

//============================================================================
// GEMM (General Matrix-Matrix Multiplication) - Q4_0 4x8
//============================================================================

void nntr_gemm_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  /**
   * A32 dot-product path; the aarch64 kernel in nntr_ggml_impl_neon.cpp uses
   * SMMLA, which needs FEAT_I8MM (armv8.6-a) and has no A32 form here, so the
   * four rows of the tile are driven through VSDOT instead. B is loaded once
   * per block and reused across all four rows.
   *
   * Scalar reference folds `>> 4` into each i-step and scales per k; both are
   * hoisted here. The shift is exact because every product is a multiple of 16
   * (both nibbles sit in the high half of the byte), and the two k halves share
   * one scale pair, so folding them before the FMA is algebraically identical.
   */
  (void)blocklen;
  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *va_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_0x4 *vb_ptr = (const block_q4_0x4 *)vx + (x * nb);

      // Row-outer: only one accumulator is live across the block loop, which
      // is what keeps this inside the sixteen Q registers A32 has. Re-reading
      // B once per row costs four L1 hits per block; carrying four
      // accumulators plus four rows of A instead costs far more in spills.
      for (int m = 0; m < 4; m++) {
        float32x4_t acc = vdupq_n_f32(0);
        // Walk the blocks with plain increments; indexing vb_ptr[l] makes gcc
        // redo a 72/136-byte stride multiply every iteration.
        const block_q4_0x4 *bp = vb_ptr;
        const block_q8_0x4 *ap = va_ptr;

        for (int l = 0; l < nb; l++, bp++, ap++) {
          const int8_t *bq = (const int8_t *)bp->qs;
          const int8_t *aq = (const int8_t *)ap->qs + m * 8;

          int8x16_t b0 = vld1q_s8(bq);      // k = 0, cols 0-1
          int8x16_t b1 = vld1q_s8(bq + 16); // k = 0, cols 2-3
          int8x16_t b2 = vld1q_s8(bq + 32); // k = 1, cols 0-1
          int8x16_t b3 = vld1q_s8(bq + 48); // k = 1, cols 2-3

          int8x8_t t0 = vld1_s8(aq);       // k = 0, low nibble operand
          int8x8_t t1 = vld1_s8(aq + 32);  // k = 1, low
          int8x8_t t2 = vld1_s8(aq + 64);  // k = 0, high
          int8x8_t t3 = vld1_s8(aq + 96);  // k = 1, high
          int8x16_t a0 = vcombine_s8(t0, t0);
          int8x16_t a1 = vcombine_s8(t1, t1);
          int8x16_t a2 = vcombine_s8(t2, t2);
          int8x16_t a3 = vcombine_s8(t3, t3);

          int32x4_t r0 = vdupq_n_s32(0);
          int32x4_t r1 = vdupq_n_s32(0);

          r0 = vdotq_s32(r0, b0 << 4, a0);
          r1 = vdotq_s32(r1, b1 << 4, a0);
          r0 = vdotq_s32(r0, b2 << 4, a1);
          r1 = vdotq_s32(r1, b3 << 4, a1);

          r0 = vdotq_s32(r0, b0 & 0xf0U, a2);
          r1 = vdotq_s32(r1, b1 & 0xf0U, a2);
          r0 = vdotq_s32(r0, b2 & 0xf0U, a3);
          r1 = vdotq_s32(r1, b3 & 0xf0U, a3);

          float32x4_t bd = vcvt_f32_f16(vld1_f16((const __fp16 *)bp->d));
          // Inline __fp16 conversion, not nntr_compute_fp16_to_fp32: an
          // out-of-line call here spills every live vector register.
          __fp16 adh;
          memcpy(&adh, &ap->d[m], sizeof(adh));

          acc = vfmaq_f32(acc, vcvtq_n_f32_s32(vpaddq_s32(r0, r1), 4),
                          vmulq_n_f32(bd, (float)adh));
        }
        vst1q_f32(&s[(y * 4 + m) * bs + x * ncols_interleaved], acc);
      }
    }
  }
#else

  float sumf[4][4];
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          sumf[m][j] = 0.0;
      }
      for (int l = 0; l < nb; l++) {
        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
          for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
              sumi = 0;
              for (int i = 0; i < blocklen; ++i) {
                const int v0 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i]
                           << 4);
                const int v1 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i] &
                           0xF0);
                sumi +=
                  ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                   (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i +
                                     qk / 2 * 4])) >>
                  4;
              }
              sumf[m][j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                            nntr_compute_fp16_to_fp32(a_ptr[l].d[m]);
            }
          }
        }
      }
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
      }
    }
  }
#endif
}

//============================================================================
// GEMM/GEMV - Q4_0 8x8 (NYI in fallback - requires SIMD for performance)
//============================================================================

void nntr_gemm_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  // NYI: Fallback implementation for 8x8 GEMM
  // For armv7l, the 4x8 kernels should be used instead
  throw std::runtime_error("NYI: nntr_gemm_q4_0_8x8_q8_0 fallback - use 4x8 "
                           "kernels for armv7l");
}

void nntr_gemv_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  // NYI: Fallback implementation for 8x8 GEMV
  throw std::runtime_error("NYI: nntr_gemv_q4_0_8x8_q8_0 fallback - use 4x8 "
                           "kernels for armv7l");
}

//============================================================================
// GEMM/GEMV - Q8_0 4x4
//============================================================================

void nntr_gemm_q8_0_4x4_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 4;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  /**
   * A32 dot-product path, ported from nntr_ggml_impl_neon.cpp. vmulq_laneq_f32
   * is aarch64-only and comes from the shim in nntr_ggml_impl_utils.h.
   */
  (void)blocklen;
  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *va_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q8_0x4 *vb_ptr = (const block_q8_0x4 *)vx + (x * nb);

      float32x4_t sf0 = vdupq_n_f32(0);
      float32x4_t sf1 = vdupq_n_f32(0);
      float32x4_t sf2 = vdupq_n_f32(0);
      float32x4_t sf3 = vdupq_n_f32(0);

      for (int l = 0; l < nb; l++) {
        float32x4_t a_d = vcvt_f32_f16(vld1_f16((const __fp16 *)va_ptr[l].d));
        float32x4_t b_d = vcvt_f32_f16(vld1_f16((const __fp16 *)vb_ptr[l].d));

        int32x4_t si0 = vdupq_n_s32(0);
        int32x4_t si1 = vdupq_n_s32(0);
        int32x4_t si2 = vdupq_n_s32(0);
        int32x4_t si3 = vdupq_n_s32(0);

        const int8_t *aq = (const int8_t *)va_ptr[l].qs;
        const int8_t *bq = (const int8_t *)vb_ptr[l].qs;

        for (int kk = 0; kk < 8; kk++) {
          int8x16_t av = vld1q_s8(aq + 16 * kk);
          int8x16_t bv = vld1q_s8(bq + 16 * kk);
          si0 = vdotq_laneq_s32(si0, bv, av, 0);
          si1 = vdotq_laneq_s32(si1, bv, av, 1);
          si2 = vdotq_laneq_s32(si2, bv, av, 2);
          si3 = vdotq_laneq_s32(si3, bv, av, 3);
        }

        sf0 = vmlaq_f32(sf0, vmulq_laneq_f32(b_d, a_d, 0), vcvtq_f32_s32(si0));
        sf1 = vmlaq_f32(sf1, vmulq_laneq_f32(b_d, a_d, 1), vcvtq_f32_s32(si1));
        sf2 = vmlaq_f32(sf2, vmulq_laneq_f32(b_d, a_d, 2), vcvtq_f32_s32(si2));
        sf3 = vmlaq_f32(sf3, vmulq_laneq_f32(b_d, a_d, 3), vcvtq_f32_s32(si3));
      }

      float *out = &s[(y * 4) * bs + x * ncols_interleaved];
      vst1q_f32(out, sf0);
      vst1q_f32(out + bs, sf1);
      vst1q_f32(out + 2 * bs, sf2);
      vst1q_f32(out + 3 * bs, sf3);
    }
  }
#else

  float sumf[4][4];
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q8_0x4 *b_ptr = (const block_q8_0x4 *)vx + (x * nb);
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumf[m][j] = 0.0;
        }
      }
      for (int l = 0; l < nb; l++) {
        for (int k = 0; k < (qk / blocklen); k++) {
          for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
              sumi = 0;
              for (int i = 0; i < blocklen; ++i) {
                const int v0 =
                  b_ptr[l]
                    .qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
              }
              sumf[m][j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                            nntr_compute_fp16_to_fp32(a_ptr[l].d[m]);
            }
          }
        }
      }
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
        }
      }
    }
  }
#endif
}

void nntr_gemv_q8_0_4x4_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 4;

  assert(nr == 1);
  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  /**
   * A32 dot-product path, ported from the aarch64 kernel in
   * nntr_ggml_impl_neon.cpp. Every intrinsic it uses exists in A32 Advanced
   * SIMD once -march=armv8.2-a+dotprod is on, vdotq_laneq_s32 included.
   */
  (void)blocklen;
  const block_q8_0x4 *vb_ptr = (const block_q8_0x4 *)vx;

  for (int c = 0; c < nc; c += ncols_interleaved) {
    const block_q8_0 *va_ptr = (const block_q8_0 *)vy;
    float32x4_t acc = vdupq_n_f32(0);

    for (int b = 0; b < nb; b++) {
      // Separate vld1q_s8 rather than vld1q_s8_x4: on A32 gcc declares the _xN
      // forms as taking const uint8_t *, and four plain loads schedule the same.
      const int8_t *bq = (const int8_t *)vb_ptr->qs;
      const int8_t *aq = (const int8_t *)va_ptr->qs;
      float16x4_t bd = vld1_f16((const __fp16 *)vb_ptr->d);

      int8x16_t a_lo = vld1q_s8(aq);
      int8x16_t a_hi = vld1q_s8(aq + 16);
      float16x4_t ad = vld1_dup_f16((const __fp16 *)&va_ptr->d);

      int32x4_t ret = vdupq_n_s32(0);

      ret = vdotq_laneq_s32(ret, vld1q_s8(bq), a_lo, 0);
      ret = vdotq_laneq_s32(ret, vld1q_s8(bq + 16), a_lo, 1);
      ret = vdotq_laneq_s32(ret, vld1q_s8(bq + 32), a_lo, 2);
      ret = vdotq_laneq_s32(ret, vld1q_s8(bq + 48), a_lo, 3);

      ret = vdotq_laneq_s32(ret, vld1q_s8(bq + 64), a_hi, 0);
      ret = vdotq_laneq_s32(ret, vld1q_s8(bq + 80), a_hi, 1);
      ret = vdotq_laneq_s32(ret, vld1q_s8(bq + 96), a_hi, 2);
      ret = vdotq_laneq_s32(ret, vld1q_s8(bq + 112), a_hi, 3);

      acc = vfmaq_f32(acc, vcvtq_f32_s32(ret),
                      vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
      va_ptr++;
      vb_ptr++;
    }
    vst1q_f32(s, acc);
    s += ncols_interleaved;
  }
#else

  float sumf[4];
  int sumi;

  const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_q8_0x4 *b_ptr = (const block_q8_0x4 *)vx + (x * nb);

    for (int j = 0; j < ncols_interleaved; j++) {
      sumf[j] = 0.0;
    }
    for (int l = 0; l < nb; l++) {
      for (int k = 0; k < (qk / blocklen); k++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumi = 0;
          for (int i = 0; i < blocklen; ++i) {
            const int v0 =
              b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
            sumi += v0 * a_ptr[l].qs[k * blocklen + i];
          }
          sumf[j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                     nntr_compute_fp16_to_fp32(a_ptr[l].d);
        }
      }
    }
    for (int j = 0; j < ncols_interleaved; j++) {
      s[x * ncols_interleaved + j] = sumf[j];
    }
  }
#endif
}

//============================================================================
// GEMM/GEMV - Q8_0 4x8
//============================================================================

void nntr_gemm_q8_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  /**
   * A32 dot-product path. The aarch64 kernel drives this shape with SMMLA
   * (vmmlaq_s32), which needs FEAT_I8MM: absent on Cortex-A76 and with no A32
   * form, so this is built on VSDOT instead, in the same 4x4 tile shape as the
   * q4_0 4x8 kernel above. B is loaded once per block and reused per row.
   */
  (void)blocklen;
  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *va_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q8_0x4 *vb_ptr = (const block_q8_0x4 *)vx + (x * nb);

      float32x4_t acc0 = vdupq_n_f32(0);
      float32x4_t acc1 = vdupq_n_f32(0);
      float32x4_t acc2 = vdupq_n_f32(0);
      float32x4_t acc3 = vdupq_n_f32(0);

      for (int l = 0; l < nb; l++) {
        const int8_t *bq = (const int8_t *)vb_ptr[l].qs;
        const int8_t *aq = (const int8_t *)va_ptr[l].qs;
        float32x4_t bd = vcvt_f32_f16(vld1_f16((const __fp16 *)vb_ptr[l].d));

        // qs is int8_t[128] laid out as {k, row/col, i}: 4 groups of 32 bytes,
        // each group holding four 8-byte lanes.
        // See the q4_0 kernel above: hoist the scale conversion so the innermost
        // loop stays free of out-of-line calls.
        float32x4_t ad = vcvt_f32_f16(vld1_f16((const __fp16 *)va_ptr[l].d));

        auto row = [&](const int8_t *ap, float32x4_t scale, float32x4_t acc) {
          int32x4_t r0 = vdupq_n_s32(0);
          int32x4_t r1 = vdupq_n_s32(0);
          for (int k = 0; k < 4; k++) {
            int8x8_t t = vld1_s8(ap + k * 32);
            int8x16_t a = vcombine_s8(t, t);
            r0 = vdotq_s32(r0, vld1q_s8(bq + k * 32), a);      // cols 0-1
            r1 = vdotq_s32(r1, vld1q_s8(bq + k * 32 + 16), a); // cols 2-3
          }
          return vfmaq_f32(acc, vcvtq_f32_s32(vpaddq_s32(r0, r1)), scale);
        };

        acc0 = row(aq, vmulq_laneq_f32(bd, ad, 0), acc0);
        acc1 = row(aq + 8, vmulq_laneq_f32(bd, ad, 1), acc1);
        acc2 = row(aq + 16, vmulq_laneq_f32(bd, ad, 2), acc2);
        acc3 = row(aq + 24, vmulq_laneq_f32(bd, ad, 3), acc3);
      }

      float *out = &s[(y * 4) * bs + x * ncols_interleaved];
      vst1q_f32(out, acc0);
      vst1q_f32(out + bs, acc1);
      vst1q_f32(out + 2 * bs, acc2);
      vst1q_f32(out + 3 * bs, acc3);
    }
  }
#else

  float sumf[4][4];
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q8_0x4 *b_ptr = (const block_q8_0x4 *)vx + (x * nb);
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumf[m][j] = 0.0;
        }
      }
      for (int l = 0; l < nb; l++) {
        for (int k = 0; k < (qk / blocklen); k++) {
          for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
              sumi = 0;
              for (int i = 0; i < blocklen; ++i) {
                const int v0 =
                  b_ptr[l]
                    .qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
              }
              sumf[m][j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                            nntr_compute_fp16_to_fp32(a_ptr[l].d[m]);
            }
          }
        }
      }
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
        }
      }
    }
  }
#endif
}

void nntr_gemv_q8_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(nr == 1);
  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
  /**
   * A32 dot-product path, ported from nntr_ggml_impl_neon.cpp. vpaddq_s32 is
   * the only aarch64-ism and it already has a shim in nntr_ggml_impl_utils.h.
   */
  (void)blocklen;
  const block_q8_0x4 *vb_ptr = (const block_q8_0x4 *)vx;

  for (int c = 0; c < nc; c += ncols_interleaved) {
    const block_q8_0 *va_ptr = (const block_q8_0 *)vy;
    float32x4_t acc = vdupq_n_f32(0);

    for (int b = 0; b < nb; b++) {
      // Separate loads rather than vld1q_s8_x4 / vld1_s8_x4: on A32 gcc declares
      // the _xN forms as taking const uint8_t *.
      const int8_t *bq = (const int8_t *)vb_ptr->qs;
      const int8_t *aq = (const int8_t *)va_ptr->qs;
      float16x4_t bd = vld1_f16((const __fp16 *)vb_ptr->d);

      int8x8_t t0 = vld1_s8(aq);
      int8x8_t t1 = vld1_s8(aq + 8);
      int8x8_t t2 = vld1_s8(aq + 16);
      int8x8_t t3 = vld1_s8(aq + 24);
      int8x16_t a0 = vcombine_s8(t0, t0);
      int8x16_t a1 = vcombine_s8(t1, t1);
      int8x16_t a2 = vcombine_s8(t2, t2);
      int8x16_t a3 = vcombine_s8(t3, t3);
      float16x4_t ad = vld1_dup_f16((const __fp16 *)&va_ptr->d);

      int32x4_t ret0 = vdupq_n_s32(0);
      int32x4_t ret1 = vdupq_n_s32(0);

      ret0 = vdotq_s32(ret0, vld1q_s8(bq), a0); // k = 0
      ret1 = vdotq_s32(ret1, vld1q_s8(bq + 16), a0);
      ret0 = vdotq_s32(ret0, vld1q_s8(bq + 32), a1); // k = 1
      ret1 = vdotq_s32(ret1, vld1q_s8(bq + 48), a1);
      ret0 = vdotq_s32(ret0, vld1q_s8(bq + 64), a2); // k = 2
      ret1 = vdotq_s32(ret1, vld1q_s8(bq + 80), a2);
      ret0 = vdotq_s32(ret0, vld1q_s8(bq + 96), a3); // k = 3
      ret1 = vdotq_s32(ret1, vld1q_s8(bq + 112), a3);

      int32x4_t ret = vpaddq_s32(ret0, ret1);

      acc = vfmaq_f32(acc, vcvtq_f32_s32(ret),
                      vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
      va_ptr++;
      vb_ptr++;
    }
    vst1q_f32(s, acc);
    s += ncols_interleaved;
  }
#else

  float sumf[4];
  int sumi;

  const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_q8_0x4 *b_ptr = (const block_q8_0x4 *)vx + (x * nb);

    for (int j = 0; j < ncols_interleaved; j++) {
      sumf[j] = 0.0;
    }
    for (int l = 0; l < nb; l++) {
      for (int k = 0; k < (qk / blocklen); k++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumi = 0;
          for (int i = 0; i < blocklen; ++i) {
            const int v0 =
              b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
            sumi += v0 * a_ptr[l].qs[k * blocklen + i];
          }
          sumf[j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                     nntr_compute_fp16_to_fp32(a_ptr[l].d);
        }
      }
    }
    for (int j = 0; j < ncols_interleaved; j++) {
      s[x * ncols_interleaved + j] = sumf[j];
    }
  }
#endif
}

//============================================================================
// GEMM/GEMV - Q4_K 8x8 (NYI in fallback)
//============================================================================

void nntr_gemm_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  // NYI: Fallback implementation for q4_K GEMM
  throw std::runtime_error("NYI: nntr_gemm_q4_K_8x8_q8_K fallback");
}

void nntr_gemv_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  // NYI: Fallback implementation for q4_K GEMV
  throw std::runtime_error("NYI: nntr_gemv_q4_K_8x8_q8_K fallback");
}

//============================================================================
// Quantization helper functions (matrix packing)
//============================================================================

void nntr_quantize_mat_q8_0_4x4(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  assert(QK8_0 == 32);
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  block_q8_0x4 *__restrict y = (block_q8_0x4 *)vy;

  // scalar
  const int blck_size_interleave = 4;
  float srcv[4][QK8_0];
  float id[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      float amax = 0.0f; // absolute max

      for (int j = 0; j < QK8_0; j++) {
        srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
        amax = MAX(amax, fabsf(srcv[row_iter][j]));
      }

      const float d = amax / ((1 << 7) - 1);
      id[row_iter] = d ? 1.0f / d : 0.0f;

      y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);
    }

    for (int j = 0; j < QK8_0 * 4; j++) {
      int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
      int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
      src_offset += (j % blck_size_interleave);

      float x0 = srcv[src_id][src_offset] * id[src_id];
      y[i].qs[j] = roundf(x0);
    }
  }
}

void nntr_quantize_mat_q8_0_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  assert(QK8_0 == 32);
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  block_q8_0x4 *__restrict y = (block_q8_0x4 *)vy;

  // Same shape as nntr_quantize_mat_q8_0_4x4, with an 8-wide interleave.
  const int blck_size_interleave = 8;
  float srcv[4][QK8_0];
  float id[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      float amax = 0.0f; // absolute max

      for (int j = 0; j < QK8_0; j++) {
        srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
        amax = MAX(amax, fabsf(srcv[row_iter][j]));
      }

      const float d = amax / ((1 << 7) - 1);
      id[row_iter] = d ? 1.0f / d : 0.0f;

      y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);
    }

    for (int j = 0; j < QK8_0 * 4; j++) {
      int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
      int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
      src_offset += (j % blck_size_interleave);

      float x0 = srcv[src_id][src_offset] * id[src_id];
      y[i].qs[j] = roundf(x0);
    }
  }
}

void nntr_quantize_mat_q8_K_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  // NYI: Fallback quantization
  throw std::runtime_error("NYI: nntr_quantize_mat_q8_K_4x8 fallback");
}

//============================================================================
// Vector dot product - Q6_K
//============================================================================

void nntr_vec_dot_q6_K_q8_K(int n, float *__restrict s, size_t bs,
                            const void *__restrict vx, size_t bx,
                            const void *__restrict vy, size_t by, int nrc) {
  assert(n % QK_K == 0);
  assert(nrc == 1);

  const block_q6_K *__restrict x = (const block_q6_K *)vx;
  const block_q8_K *__restrict y = (const block_q8_K *)vy;

  const int nb = n / QK_K;

  float sumf = 0.0f;

  for (int i = 0; i < nb; ++i) {
    const float d = y[i].d * nntr_compute_fp16_to_fp32(x[i].d);
    const uint8_t *__restrict q4 = x[i].ql;
    const uint8_t *__restrict qh = x[i].qh;
    const int8_t *__restrict q8 = y[i].qs;
    const int8_t *__restrict scales = x[i].scales;

    int32_t sum = 0;

    for (int j = 0; j < QK_K / 128; ++j) {
      int32_t sum_block = 0;
      for (int l = 0; l < 32; ++l) {
        // Decode 6-bit quantized values
        const int q4_0 = q4[l] & 0xF;
        const int q4_1 = q4[l] >> 4;
        const int q4_2 = q4[l + 32] & 0xF;
        const int q4_3 = q4[l + 32] >> 4;

        const int qh_0 = (qh[l] & 0x03) << 4;
        const int qh_1 = (qh[l] & 0x0C) << 2;
        const int qh_2 = (qh[l] & 0x30);
        const int qh_3 = (qh[l] & 0xC0) >> 2;

        const int q6_0 = (q4_0 | qh_0) - 32;
        const int q6_1 = (q4_1 | qh_1) - 32;
        const int q6_2 = (q4_2 | qh_2) - 32;
        const int q6_3 = (q4_3 | qh_3) - 32;

        sum_block += q6_0 * q8[l] * scales[0] + q6_1 * q8[l + 32] * scales[1] +
                     q6_2 * q8[l + 64] * scales[2] +
                     q6_3 * q8[l + 96] * scales[3];
      }
      sum += sum_block;
      q4 += 64;
      qh += 32;
      q8 += 128;
      scales += 4;
    }
    sumf += d * sum;
  }
  *s = sumf;
}

//============================================================================
// Repack functions - Q4_0 to interleaved formats
//============================================================================

int nntr_repack_q4_0_to_q4_0_4_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 4 || interleave_block == 8);
  constexpr int nrows_interleaved = 4;

  block_q4_0x4 *dst_ = (block_q4_0x4 *)dst;
  const block_q4_0 *src = (const block_q4_0 *)data;
  block_q4_0 dst_tmp[4];
  int nblocks = k / Q4_0;

  assert(data_size == nrow * nblocks * sizeof(block_q4_0));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = nntr_make_block_q4_0x4(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}

int nntr_repack_q4_0_to_q4_0_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 8);
  constexpr size_t nrows_interleaved = 8;

  block_q4_0x8 *dst_ = (block_q4_0x8 *)dst;
  const block_q4_0 *src = (const block_q4_0 *)data;
  block_q4_0 dst_tmp[8];
  int nblocks = k / QK_0<4>();

  assert(data_size == nrow * nblocks * sizeof(block_q4_0));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = nntr_make_block_q4_0x8(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}

int nntr_repack_q8_0_to_q8_0_4_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 4 || interleave_block == 8);
  constexpr int nrows_interleaved = 4;

  block_q8_0x4 *dst_ = (block_q8_0x4 *)dst;
  const block_q8_0 *src = (const block_q8_0 *)data;
  block_q8_0 dst_tmp[4];
  int nblocks = k / QK8_0;

  assert(data_size == nrow * nblocks * sizeof(block_q8_0));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (int b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (int i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = nntr_make_block_q8_0x4(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}

int nntr_repack_q4_K_to_q4_K_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 8);
  constexpr size_t nrows_interleaved = 8;

  block_q4_Kx8 *dst_ = (block_q4_Kx8 *)dst;
  const block_q4_K *src = (const block_q4_K *)data;
  block_q4_K dst_tmp[8];
  int nblocks = k / QK_K;

  assert(data_size == nrow * nblocks * sizeof(block_q4_K));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = make_block_q4_Kx8(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}

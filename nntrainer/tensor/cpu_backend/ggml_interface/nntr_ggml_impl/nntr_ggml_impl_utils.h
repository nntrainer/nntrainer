// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Portions of this file are derived from llama.cpp
 * (https://github.com/ggml-org/llama.cpp), licensed under the MIT License.
 * Copyright (c) Contributors to llama.cpp
 *
 * Modified by Pawel Debski, 2025: Adapted for CPU backend integration
 *
 * @file   nntr_ggml_impl_utils.h
 * @date   20 August 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author  Pawel Debski <p.debski2@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Platform-specific logic and helper functions for ggml implementation
 */

#ifndef __NNTR_GGML_IMPL_INTERNAL__
#define __NNTR_GGML_IMPL_INTERNAL__

#include <cstring>
#include <stdint.h>

#include <nntr_ggml_impl_common.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#if defined(__ARM_NEON) && (!ARMV7)
#include <arm_neon.h>
#elif defined(__ARM_ARCH_7A__) || defined(__arm__) || ARMV7
#include <armv7_neon.h>
#elif defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif // !defined(__aarch64__)

// some compilers don't provide _mm256_set_m128i, e.g. gcc 7
#define MM256_SET_M128I(a, b)                                                  \
  _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) ||           \
  defined(__SSSE3__)
// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
  // Get absolute values of x vectors
  const __m128i ax = _mm_sign_epi8(x, x);
  // Sign the values of the y vectors
  const __m128i sy = _mm_sign_epi8(y, x);
  // Perform multiplication and create 16-bit values
  const __m128i dot = _mm_maddubs_epi16(ax, sy);
  const __m128i ones = _mm_set1_epi16(1);
  return _mm_madd_epi16(ones, dot);
}

#if __AVX__ || __AVX2__ || __AVX512F__
// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
  __m128 res = _mm256_extractf128_ps(x, 1);
  res = _mm_add_ps(res, _mm256_castps256_ps128(x));
  res = _mm_add_ps(res, _mm_movehl_ps(res, res));
  res = _mm_add_ss(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
}

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
  const __m128i sum128 =
    _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
  const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
  const __m128i sum64 = _mm_add_epi32(hi64, sum128);
  const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
  return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// horizontally add 4 int32_t
static inline int hsum_i32_4(const __m128i a) {
  const __m128i hi64 = _mm_unpackhi_epi64(a, a);
  const __m128i sum64 = _mm_add_epi32(hi64, a);
  const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
  return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

#if defined(__AVX2__) || defined(__AVX512F__)
// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t *x) {
  uint32_t x32;
  memcpy(&x32, x, sizeof(uint32_t));
  const __m256i shuf_mask =
    _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202,
                      0x0101010101010101, 0x0000000000000000);
  __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
  const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
  bytes = _mm256_or_si256(bytes, bit_mask);
  return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t *rsi) {
  const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
  const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
  const __m256i lowMask = _mm256_set1_epi8(0xF);
  return _mm256_and_si256(lowMask, bytes);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
  return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax,
                                             const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
  const __m256i zero = _mm256_setzero_si256();
  const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
  return _mm256_cvtepi32_ps(summed_pairs);
#elif defined(__AVXVNNI__)
  const __m256i zero = _mm256_setzero_si256();
  const __m256i summed_pairs = _mm256_dpbusd_avx_epi32(zero, ax, sy);
  return _mm256_cvtepi32_ps(summed_pairs);
#else
  // Perform multiplication and create 16-bit values
  const __m256i dot = _mm256_maddubs_epi16(ax, sy);
  return sum_i16_pairs_float(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
  const __m256i zero = _mm256_setzero_si256();
  const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
  return _mm256_cvtepi32_ps(summed_pairs);
#else
  // Get absolute values of x vectors
  const __m256i ax = _mm256_sign_epi8(x, x);
  // Sign the values of the y vectors
  const __m256i sy = _mm256_sign_epi8(y, x);
  return mul_sum_us8_pairs_float(ax, sy);
#endif
}

static inline __m128i packNibbles(__m256i bytes) {
  // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into
  // 0000_0000_abcd_efgh
#if __AVX512F__
  const __m256i bytes_srli_4 =
    _mm256_srli_epi16(bytes, 4);                // 0000_0000_abcd_0000
  bytes = _mm256_or_si256(bytes, bytes_srli_4); // 0000_abcd_abcd_efgh
  return _mm256_cvtepi16_epi8(bytes);           // abcd_efgh
#else
  const __m256i lowByte = _mm256_set1_epi16(0xFF);
  __m256i high = _mm256_andnot_si256(lowByte, bytes);
  __m256i low = _mm256_and_si256(lowByte, bytes);
  high = _mm256_srli_epi16(high, 4);
  bytes = _mm256_or_si256(low, high);

  // Compress uint16_t lanes into bytes
  __m128i r0 = _mm256_castsi256_si128(bytes);
  __m128i r1 = _mm256_extracti128_si256(bytes, 1);
  return _mm_packus_epi16(r0, r1);
#endif
}
#elif defined(__AVX__)
static inline __m128i packNibbles(__m128i bytes1, __m128i bytes2) {
  // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into
  // 0000_0000_abcd_efgh
  const __m128i lowByte = _mm_set1_epi16(0xFF);
  __m128i high = _mm_andnot_si128(lowByte, bytes1);
  __m128i low = _mm_and_si128(lowByte, bytes1);
  high = _mm_srli_epi16(high, 4);
  bytes1 = _mm_or_si128(low, high);
  high = _mm_andnot_si128(lowByte, bytes2);
  low = _mm_and_si128(lowByte, bytes2);
  high = _mm_srli_epi16(high, 4);
  bytes2 = _mm_or_si128(low, high);

  return _mm_packus_epi16(bytes1, bytes2);
}

static inline __m128i mul_add_epi8_sse(const __m128i x, const __m128i y) {
  const __m128i ax = _mm_sign_epi8(x, x);
  const __m128i sy = _mm_sign_epi8(y, x);
  return _mm_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t *x) {
  uint32_t x32;
  memcpy(&x32, x, sizeof(uint32_t));
  const __m128i shuf_maskl =
    _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
  const __m128i shuf_maskh =
    _mm_set_epi64x(0x0303030303030303, 0x0202020202020202);
  __m128i bytesl = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
  __m128i bytesh = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);
  const __m128i bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfe);
  bytesl = _mm_or_si128(bytesl, bit_mask);
  bytesh = _mm_or_si128(bytesh, bit_mask);
  bytesl = _mm_cmpeq_epi8(bytesl, _mm_set1_epi64x(-1));
  bytesh = _mm_cmpeq_epi8(bytesh, _mm_set1_epi64x(-1));
  return MM256_SET_M128I(bytesh, bytesl);
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t *rsi) {
  // Load 16 bytes from memory
  __m128i tmpl = _mm_loadu_si128((const __m128i *)rsi);
  __m128i tmph = _mm_srli_epi16(tmpl, 4);
  const __m128i lowMask = _mm_set1_epi8(0xF);
  tmpl = _mm_and_si128(lowMask, tmpl);
  tmph = _mm_and_si128(lowMask, tmph);
  return MM256_SET_M128I(tmph, tmpl);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m128i xh, const __m128i xl) {
  const __m128i ones = _mm_set1_epi16(1);
  const __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
  const __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
  const __m256i summed_pairs = MM256_SET_M128I(summed_pairsh, summed_pairsl);
  return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax,
                                             const __m256i sy) {
  const __m128i axl = _mm256_castsi256_si128(ax);
  const __m128i axh = _mm256_extractf128_si256(ax, 1);
  const __m128i syl = _mm256_castsi256_si128(sy);
  const __m128i syh = _mm256_extractf128_si256(sy, 1);
  // Perform multiplication and create 16-bit values
  const __m128i dotl = _mm_maddubs_epi16(axl, syl);
  const __m128i doth = _mm_maddubs_epi16(axh, syh);
  return sum_i16_pairs_float(doth, dotl);
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
  const __m128i xl = _mm256_castsi256_si128(x);
  const __m128i xh = _mm256_extractf128_si256(x, 1);
  const __m128i yl = _mm256_castsi256_si128(y);
  const __m128i yh = _mm256_extractf128_si256(y, 1);
  // Get absolute values of x vectors
  const __m128i axl = _mm_sign_epi8(xl, xl);
  const __m128i axh = _mm_sign_epi8(xh, xh);
  // Sign the values of the y vectors
  const __m128i syl = _mm_sign_epi8(yl, xl);
  const __m128i syh = _mm_sign_epi8(yh, xh);
  // Perform multiplication and create 16-bit values
  const __m128i dotl = _mm_maddubs_epi16(axl, syl);
  const __m128i doth = _mm_maddubs_epi16(axh, syh);
  return sum_i16_pairs_float(doth, dotl);
}

// larger version of mul_sum_i8_pairs_float where x and y are each represented
// by four 128-bit vectors
static inline __m256
mul_sum_i8_quad_float(const __m128i x_1_0, const __m128i x_1_1,
                      const __m128i x_2_0, const __m128i x_2_1,
                      const __m128i y_1_0, const __m128i y_1_1,
                      const __m128i y_2_0, const __m128i y_2_1) {
  const __m128i mone = _mm_set1_epi16(1);

  const __m128i p16_1_0 = mul_add_epi8_sse(x_1_0, y_1_0);
  const __m128i p16_1_1 = mul_add_epi8_sse(x_1_1, y_1_1);
  const __m128i p16_2_0 = mul_add_epi8_sse(x_2_0, y_2_0);
  const __m128i p16_2_1 = mul_add_epi8_sse(x_2_1, y_2_1);
  const __m128i p_1_0 = _mm_madd_epi16(p16_1_0, mone);
  const __m128i p_1_1 = _mm_madd_epi16(p16_1_1, mone);
  const __m128i p_2_0 = _mm_madd_epi16(p16_2_0, mone);
  const __m128i p_2_1 = _mm_madd_epi16(p16_2_1, mone);
  const __m128i p_1 = _mm_add_epi32(p_1_0, p_1_1);
  const __m128i p_2 = _mm_add_epi32(p_2_0, p_2_1);
  return _mm256_cvtepi32_ps(MM256_SET_M128I(p_2, p_1));
}

// quad fp16 delta calculation
static inline __m256 quad_fp16_delta_float(const float x0, const float y0,
                                           const float x1, const float y1) {
  // nntr_compute_fp16_to_fp32 is faster than Intel F16C
  return _mm256_set_m128(
    _mm_set1_ps(nntr_compute_fp16_to_fp32(x1) * nntr_compute_fp16_to_fp32(y1)),
    _mm_set1_ps(nntr_compute_fp16_to_fp32(x0) * nntr_compute_fp16_to_fp32(y0)));
}
#endif
#elif defined(__SSSE3__)
// horizontally add 4x4 floats
static inline float hsum_float_4x4(const __m128 a, const __m128 b,
                                   const __m128 c, const __m128 d) {
  __m128 res_0 = _mm_hadd_ps(a, b);
  __m128 res_1 = _mm_hadd_ps(c, d);
  __m128 res = _mm_hadd_ps(res_0, res_1);
  res = _mm_hadd_ps(res, res);
  res = _mm_hadd_ps(res, res);

  return _mm_cvtss_f32(res);
}
#endif // __AVX__ || __AVX2__ || __AVX512F__
#endif // defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) ||
       // defined(__SSSE3__)

#if defined(__loongarch_asx)

#ifdef __clang__
#define VREGS_PREFIX "$vr"
#define XREGS_PREFIX "$xr"
#else // GCC
#define VREGS_PREFIX "$f"
#define XREGS_PREFIX "$f"
#endif
#define __ALL_REGS                                                             \
  "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27," \
  "28,29,30,31"
// Convert __m128i to __m256i
static inline __m256i ____m256i(__m128i in) {
  __m256i out = __lasx_xvldi(0);
  __asm__ volatile(".irp i," __ALL_REGS "\n\t"
                   " .ifc %[out], " XREGS_PREFIX "\\i    \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[in], " VREGS_PREFIX "\\j  \n\t"
                   "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   : [out] "+f"(out)
                   : [in] "f"(in));
  return out;
}
// Convert two __m128i to __m256i
static inline __m256i lasx_set_q(__m128i inhi, __m128i inlo) {
  __m256i out;
  __asm__ volatile(".irp i," __ALL_REGS "\n\t"
                   " .ifc %[hi], " VREGS_PREFIX "\\i    \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[lo], " VREGS_PREFIX "\\j  \n\t"
                   "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   ".ifnc %[out], %[hi]                 \n\t"
                   ".irp i," __ALL_REGS "\n\t"
                   " .ifc %[out], " XREGS_PREFIX "\\i   \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[hi], " VREGS_PREFIX "\\j  \n\t"
                   "    xvori.b $xr\\i, $xr\\j, 0       \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   ".endif                              \n\t"
                   : [out] "=f"(out), [hi] "+f"(inhi)
                   : [lo] "f"(inlo));
  return out;
}
// Convert __m256i low part to __m128i
static inline __m128i lasx_extracti128_lo(__m256i in) {
  __m128i out;
  __asm__ volatile(".ifnc %[out], %[in]                 \n\t"
                   ".irp i," __ALL_REGS "\n\t"
                   " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
                   "    vori.b $vr\\i, $vr\\j, 0        \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   ".endif                              \n\t"
                   : [out] "=f"(out)
                   : [in] "f"(in));
  return out;
}
// Convert __m256i high part to __m128i
static inline __m128i lasx_extracti128_hi(__m256i in) {
  __m128i out;
  __asm__ volatile(".irp i," __ALL_REGS "\n\t"
                   " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
                   "  .irp j," __ALL_REGS "\n\t"
                   "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
                   "    xvpermi.q $xr\\i, $xr\\j, 0x11  \n\t"
                   "   .endif                           \n\t"
                   "  .endr                             \n\t"
                   " .endif                             \n\t"
                   ".endr                               \n\t"
                   : [out] "=f"(out)
                   : [in] "f"(in));
  return out;
}

static __m256i lasx_set_w(int e7, int e6, int e5, int e4, int e3, int e2,
                          int e1, int e0) {
  v8i32 __ret = {e0, e1, e2, e3, e4, e5, e6, e7};
  return (__m256i)__ret;
}

static __m256i lasx_set_d(int64_t a, int64_t b, int64_t c, int64_t d) {
  v4i64 __ret = {d, c, b, a};
  return (__m256i)__ret;
}

static __m256i lasx_insertf128(__m128i x, __m128i y) {
  return lasx_set_q(x, y);
}

static __m256i lasx_shuffle_b(__m256i a, __m256i b) {
  __m256i mask_f, zero, tmp0, tmp2, mask;
  int f = 0x8f;
  mask_f = __lasx_xvreplgr2vr_b(f);
  zero = __lasx_xvldi(0);
  tmp0 = __lasx_xvand_v(b, mask_f); // get mask with low 4 bit and sign bits
  tmp0 = __lasx_xvori_b(
    tmp0, 0x10); // make each mask or  with 0x10 prepare for positive
  mask = __lasx_xvsle_b(zero, tmp0); // if mask >= 0, set mask
  tmp2 = __lasx_xvand_v(tmp0, mask); // maskout the in2 < ones
  return __lasx_xvshuf_b(a, zero, tmp2);
}

static __m256i lasx_extu8_16(__m128i a) {
  return __lasx_vext2xv_hu_bu(____m256i(a));
}

static __m256i lasx_ext8_16(__m128i a) {
  return __lasx_vext2xv_h_b(____m256i(a));
}

static __m256i lasx_ext16_32(__m128i a) {
  return __lasx_vext2xv_w_h(____m256i(a));
}

static __m128i lasx_extracti128(__m256i a, int pos) {
  __m128i ret;
  if (pos == 0) {
    ret = lasx_extracti128_lo(a);
  } else {
    ret = lasx_extracti128_hi(a);
  }
  return ret;
}

static __m128 lasx_extractf128(__m256 a, int pos) {
  __m128 ret;
  if (pos == 0) {
    ret = (__m128)lasx_extracti128_lo((__m256i)a);
  } else {
    ret = (__m128)lasx_extracti128_hi((__m256i)a);
  }
  return ret;
}

static __m256i lasx_maddubs_h(__m256i a, __m256i b) {
  __m256i tmp1, tmp2;
  tmp1 = __lasx_xvmulwev_h_b(a, b);
  tmp2 = __lasx_xvmulwod_h_b(a, b);
  return __lasx_xvsadd_h(tmp1, tmp2);
}

static __m256i lasx_madd_h(__m256i a, __m256i b) {
  __m256i tmp1, tmp2;
  tmp1 = __lasx_xvmulwev_w_h(a, b);
  tmp2 = __lasx_xvmulwod_w_h(a, b);
  return __lasx_xvadd_w(tmp1, tmp2);
}

static __m256i lasx_packs_w(__m256i a, __m256i b) {
  __m256i tmp, tmp1;
  tmp = __lasx_xvsat_w(a, 15);
  tmp1 = __lasx_xvsat_w(b, 15);
  return __lasx_xvpickev_h(tmp1, tmp);
}

static __m256i lasx_packs_h(__m256i a, __m256i b) {
  __m256i tmp, tmp1;
  tmp = __lasx_xvsat_h(a, 7);
  tmp1 = __lasx_xvsat_h(b, 7);
  return __lasx_xvpickev_b(tmp1, tmp);
}

static inline __m256i lasx_madd_h_b(__m256i a, __m256i b) {
  __m256i tmp1, tmp2;
  tmp1 = __lasx_xvmulwev_h_b(a, b);
  tmp2 = __lasx_xvmulwod_h_b(a, b);
  return __lasx_xvadd_h(tmp1, tmp2);
}

static inline __m256i lasx_xvrepl128vei_h(__m256i a, const unsigned int b) {
  switch (b) {
  case 0:
    return __lasx_xvrepl128vei_h(a, 0);
  case 1:
    return __lasx_xvrepl128vei_h(a, 1);
  case 2:
    return __lasx_xvrepl128vei_h(a, 2);
  case 3:
    return __lasx_xvrepl128vei_h(a, 3);
  case 4:
    return __lasx_xvrepl128vei_h(a, 4);
  case 5:
    return __lasx_xvrepl128vei_h(a, 5);
  case 6:
    return __lasx_xvrepl128vei_h(a, 6);
  case 7:
    return __lasx_xvrepl128vei_h(a, 7);
  default:
    __builtin_unreachable();
  }
}

static inline __m256i lasx_xvandi_b_bit(__m256i a, const unsigned int b) {
  switch (b) {
  case 0:
    return __lasx_xvandi_b(a, 1 << 0);
  case 1:
    return __lasx_xvandi_b(a, 1 << 1);
  case 2:
    return __lasx_xvandi_b(a, 1 << 2);
  case 3:
    return __lasx_xvandi_b(a, 1 << 3);
  case 4:
    return __lasx_xvandi_b(a, 1 << 4);
  case 5:
    return __lasx_xvandi_b(a, 1 << 5);
  case 6:
    return __lasx_xvandi_b(a, 1 << 6);
  case 7:
    return __lasx_xvandi_b(a, 1 << 7);
  default:
    __builtin_unreachable();
  }
}

// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
  // Get absolute values of x vectors
  const __m128i ax = __lsx_vsigncov_b(x, x);
  // Sign the values of the y vectors
  const __m128i sy = __lsx_vsigncov_b(x, y);
  // Perform multiplication and create 16-bit values
  const __m128i dot = lsx_maddubs_h(ax, sy);
  const __m128i ones = __lsx_vreplgr2vr_h(1);
  return lsx_madd_h(ones, dot);
}

// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
  __m128 res = lasx_extractf128(x, 1);
  res = __lsx_vfadd_s(res, lasx_extractf128(x, 0));
  res = __lsx_vfadd_s(res, (__m128)__lsx_vpickod_d((__m128i)res, (__m128i)res));
  res = __lsx_vfadd_s(res, (__m128)__lsx_vinsgr2vr_w(
                             __lsx_vldi(0), __lsx_vpickve2gr_w(res, 1), 0));
  return ((v4f32)res)[0];
}

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {

  __m256i tmp1 = __lasx_xvpermi_q(a, a, 0x11);
  __m256i tmp2 = __lasx_xvpermi_q(a, a, 0x00);

  __m128i tmp1_128 = lasx_extracti128_lo(tmp1);
  __m128i tmp2_128 = lasx_extracti128_lo(tmp2);

  __m128i sum128 = __lsx_vadd_w(tmp1_128, tmp2_128);

  __m128i ev = __lsx_vpickev_w(sum128, sum128);
  __m128i od = __lsx_vpickod_w(sum128, sum128);
  __m128i sum64 = __lsx_vadd_w(ev, od);

  int sum64_1, sum64_2;
  sum64_1 = __lsx_vpickve2gr_w(sum64, 0);
  sum64_2 = __lsx_vpickve2gr_w(sum64, 1);

  return sum64_1 + sum64_2;
}

// horizontally add 4 int32_t
static inline int hsum_i32_4(const __m128i a) {
  __m128i ev = __lsx_vpickev_w(a, a);
  __m128i od = __lsx_vpickod_w(a, a);
  __m128i sum64 = __lsx_vadd_w(ev, od);

  int sum64_1, sum64_2;
  sum64_1 = __lsx_vpickve2gr_w(sum64, 0);
  sum64_2 = __lsx_vpickve2gr_w(sum64, 1);

  return sum64_1 + sum64_2;
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t *x) {

  uint32_t x32;
  memcpy(&x32, x, sizeof(uint32_t));
  const __m256i shuf_mask = lasx_set_d(0x0303030303030303, 0x0202020202020202,
                                       0x0101010101010101, 0x0000000000000000);

  __m256i bytes = lasx_shuffle_b(__lasx_xvreplgr2vr_w(x32), shuf_mask);
  const __m256i bit_mask = __lasx_xvreplgr2vr_d(0x7fbfdfeff7fbfdfe);
  bytes = __lasx_xvor_v(bytes, bit_mask);
  return __lasx_xvseq_b(bytes, __lasx_xvreplgr2vr_d(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t *rsi) {
  const __m128i lo = __lsx_vld((const __m128i *)rsi, 0);
  __m128i hi = __lsx_vsrli_h(lo, 4);
  return __lasx_xvandi_b(lasx_insertf128(hi, lo), 0xf);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
  __m256i v = __lasx_xvpackod_h(x, x);
  __m256i summed_pairs = __lasx_xvaddwev_w_h(x, v);
  return __lasx_xvffint_s_w(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax,
                                             const __m256i sy) {
  // Perform multiplication and create 16-bit values
  const __m256i dot = lasx_maddubs_h(ax, sy);
  return sum_i16_pairs_float(dot);
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
  const __m256i dot = lasx_madd_h_b(x, y);
  return sum_i16_pairs_float(dot);
}

static inline __m128i packNibbles(__m256i bytes) {
  // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into
  // 0000_0000_abcd_efgh
  const __m256i lowByte = __lasx_xvreplgr2vr_h(0xFF);
  __m256i high = __lasx_xvandn_v(lowByte, bytes);
  __m256i low = __lasx_xvand_v(lowByte, bytes);
  high = __lasx_xvsrli_h(high, 4);
  bytes = __lasx_xvor_v(low, high);
  // Compress uint16_t lanes into bytes
  __m128i *r0 = (__m128i *)&bytes;
  __m256i tmp_h128 = __lasx_xvpermi_q(bytes, bytes, 0x11);
  __m128i *r1 = (__m128i *)&tmp_h128;

  __m128i zero = __lsx_vldi(0);
  __m128i tmp, tmp2, tmp3;

  tmp = __lsx_vmax_h(zero, *r0);
  tmp2 = __lsx_vsat_hu(tmp, 7);

  tmp = __lsx_vmax_h(zero, *r1);
  tmp3 = __lsx_vsat_hu(tmp, 7);
  return __lsx_vpickev_b(tmp3, tmp2);
}
#endif //__loongarch_asx

#endif

#if defined(__AVX2__) || defined(__AVX512F__)
#if defined(__AVX512F__)
// add int16_t pairwise and return as 512 bit int vector, then add the
// accumulator
static inline __m512i sum_i16_pairs_acc_int32x16(const __m512i acc,
                                                 const __m512i x) {
  const __m512i ones = _mm512_set1_epi16(1);
  return _mm512_add_epi32(acc, _mm512_madd_epi16(ones, x));
}

static inline __m512i mul_sum_us8_pairs_acc_int32x16(const __m512i acc,
                                                     const __m512i ax,
                                                     const __m512i sy) {
#if defined(__AVX512VNNI__)
  return _mm512_dpbusd_epi32(acc, ax, sy);
#else
  // Perform multiplication and create 16-bit values
  const __m512i dot = _mm512_maddubs_epi16(ax, sy);
  return sum_i16_pairs_acc_int32x16(acc, dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as 512 bit int
// vector，then add the accumulator
static inline __m512i mul_sum_i8_pairs_acc_int32x16(const __m512i acc,
                                                    const __m512i x,
                                                    const __m512i y) {
  const __m512i zero = _mm512_setzero_si512();
  // Get absolute values of x vectors
  const __m512i ax = _mm512_abs_epi8(x);
  // Sign the values of the y vectors
  __mmask64 blt0 = _mm512_movepi8_mask(x);
  const __m512i sy = _mm512_mask_sub_epi8(y, blt0, zero, y);
  return mul_sum_us8_pairs_acc_int32x16(acc, ax, sy);
}
#endif

// add int16_t pairwise and return as 256 bit int vector, then add the
// accumulator
static inline __m256i sum_i16_pairs_acc_int32x8(const __m256i acc,
                                                const __m256i x) {
  const __m256i ones = _mm256_set1_epi16(1);
  return _mm256_add_epi32(acc, _mm256_madd_epi16(ones, x));
}

static inline __m256i mul_sum_us8_pairs_acc_int32x8(const __m256i acc,
                                                    const __m256i ax,
                                                    const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
  return _mm256_dpbusd_epi32(acc, ax, sy);
#elif defined(__AVXVNNI__)
  return _mm256_dpbusd_avx_epi32(acc, ax, sy);
#else
  // Perform multiplication and create 16-bit values
  const __m256i dot = _mm256_maddubs_epi16(ax, sy);
  return sum_i16_pairs_acc_int32x8(acc, dot);
#endif
}

// Integer variant of the function defined in ggml-quants.c
// multiply int8_t, add results pairwise twice and return as 256 bit int vector,
// then add the accumulator
static inline __m256i mul_sum_i8_pairs_acc_int32x8(const __m256i acc,
                                                   const __m256i x,
                                                   const __m256i y) {
#if defined(__AVXVNNIINT8__)
  return _mm256_dpbssd_epi32(acc, x, y);
#else
  // Get absolute values of x vectors
  const __m256i ax = _mm256_sign_epi8(x, x);
  // Sign the values of the y vectors
  const __m256i sy = _mm256_sign_epi8(y, x);
  return mul_sum_us8_pairs_acc_int32x8(acc, ax, sy);
#endif
}
#endif

#if defined(__AVX__)
#if defined(__F16C__)
#if defined(__AVX512F__)
#define GGML_F32Cx8x2_LOAD(x, y)                                               \
  _mm512_cvtph_ps(_mm256_set_m128i(_mm_loadu_si128((const __m128i *)(y)),      \
                                   _mm_loadu_si128((const __m128i *)(x))))
#define GGML_F32Cx16_REPEAT_LOAD(x) _mm512_cvtph_ps(_mm256_set_m128i(x, x))
#endif
// the  _mm256_cvt intrinsics require F16C
#define GGML_F32Cx8_LOAD(x)                                                    \
  _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask)                                   \
  _mm256_cvtph_ps(                                                             \
    _mm_shuffle_epi32(_mm_maskload_epi32((int const *)(x), loadMask), 68))
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask)                             \
  _mm256_cvtph_ps(                                                             \
    _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)x), arrangeMask))
#else
#if defined(__AVX512F__)
static inline __m512 __avx512_f32cx8x2_load(nntr_fp16_t *x, nntr_fp16_t *y) {
  float tmp[16];

  for (int i = 0; i < 8; i++) {
    tmp[i] = nntr_fp16_to_fp32(x[i]);
  }

  for (int i = 0; i < 8; i++) {
    tmp[i + 8] = nntr_fp16_to_fp32(y[i]);
  }

  return _mm512_loadu_ps(tmp);
}
static inline __m512 __avx512_repeat_f32cx16_load(__m128i x) {
  float tmp[16];
  uint16_t tmphalf[8];
  _mm_storeu_si128((__m128i *)tmphalf, x);

  for (int i = 0; i < 4; i++) {
    tmp[i] = nntr_fp16_to_fp32(tmphalf[i]);
    tmp[i + 4] = nntr_fp16_to_fp32(tmphalf[i]);
    tmp[i + 8] = nntr_fp16_to_fp32(tmphalf[i]);
    tmp[i + 12] = nntr_fp16_to_fp32(tmphalf[i]);
  }

  return _mm512_loadu_ps(tmp);
}
#endif
static inline __m256 __avx_f32cx8_load(nntr_fp16_t *x) {
  float tmp[8];

  for (int i = 0; i < 8; i++) {
    tmp[i] = nntr_fp16_to_fp32(x[i]);
  }

  return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_repeat_f32cx8_load(nntr_fp16_t *x) {
  float tmp[8];

  for (int i = 0; i < 4; i++) {
    tmp[i] = nntr_fp16_to_fp32(x[i]);
    tmp[i + 4] = nntr_fp16_to_fp32(x[i]);
  }

  return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_rearranged_f32cx8_load(nntr_fp16_t *x,
                                                  __m128i arrangeMask) {
  uint16_t tmphalf[8];
  float tmp[8];

  _mm_storeu_si128(
    (__m128i *)tmphalf,
    _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)x), arrangeMask));
  for (int i = 0; i < 8; i++) {
    tmp[i] = nntr_fp16_to_fp32(tmphalf[i]);
  }

  return _mm256_loadu_ps(tmp);
}

#define GGML_F32Cx8_LOAD(x) __avx_f32cx8_load((nntr_fp16_t *)(x))
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask)                                   \
  __avx_repeat_f32cx8_load((nntr_fp16_t *)(x))
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask)                             \
  __avx_rearranged_f32cx8_load((nntr_fp16_t *)(x), arrangeMask)
#if defined(__AVX512F__)
#define GGML_F32Cx8x2_LOAD(x, y)                                               \
  __avx512_f32cx8x2_load((nntr_fp16_t *)(x), (nntr_fp16_t *)(y))
#define GGML_F32Cx16_REPEAT_LOAD(x) __avx512_repeat_f32cx16_load(x)
#endif
#endif
#endif

#if defined(__ARM_NEON)

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#ifdef _MSC_VER
#define ggml_vld1q_u32(w, x, y, z)                                             \
  { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }
#else
#define ggml_vld1q_u32(w, x, y, z)                                             \
  { (w), (x), (y), (z) }
#endif // _MSC_VER

#if !defined(__aarch64__)

// 32-bit ARM compatibility

// vaddlvq_s16
// vpaddq_s16
// vpaddq_s32
// vaddvq_s32
// vaddvq_f32
// vmaxvq_f32
// vcvtnq_s32_f32
// vzip1_u8
// vzip2_u8

inline static int32_t vaddlvq_s16(int16x8_t v) {
  int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
  return vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
}

inline static int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
  int16x4_t a0 = vpadd_s16(vget_low_s16(a), vget_high_s16(a));
  int16x4_t b0 = vpadd_s16(vget_low_s16(b), vget_high_s16(b));
  return vcombine_s16(a0, b0);
}

inline static int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
  int32x2_t a0 = vpadd_s32(vget_low_s32(a), vget_high_s32(a));
  int32x2_t b0 = vpadd_s32(vget_low_s32(b), vget_high_s32(b));
  return vcombine_s32(a0, b0);
}

inline static int32_t vaddvq_s32(int32x4_t v) {
  return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) +
         vgetq_lane_s32(v, 3);
}

#if not(defined(ARMV7)) || !(ARMV7)

inline static float vaddvq_f32(float32x4_t v) {
  return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) +
         vgetq_lane_f32(v, 3);
}

inline static float vmaxvq_f32(float32x4_t v) {
  return MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
             MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

inline static int32x4_t vcvtnq_s32_f32(float32x4_t v) {
  int32x4_t res;

  res[0] = roundf(vgetq_lane_f32(v, 0));
  res[1] = roundf(vgetq_lane_f32(v, 1));
  res[2] = roundf(vgetq_lane_f32(v, 2));
  res[3] = roundf(vgetq_lane_f32(v, 3));

  return res;
}

#endif

inline static uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
  uint8x8_t res;

  res[0] = a[0];
  res[1] = b[0];
  res[2] = a[1];
  res[3] = b[1];
  res[4] = a[2];
  res[5] = b[2];
  res[6] = a[3];
  res[7] = b[3];

  return res;
}

inline static uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
  uint8x8_t res;

  res[0] = a[4];
  res[1] = b[4];
  res[2] = a[5];
  res[3] = b[5];
  res[4] = a[6];
  res[5] = b[6];
  res[6] = a[7];
  res[7] = b[7];

  return res;
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

typedef struct ggml_int16x8x2_t {
  int16x8_t val[2];
} ggml_int16x8x2_t;

inline static ggml_int16x8x2_t ggml_vld1q_s16_x2(const int16_t *ptr) {
  ggml_int16x8x2_t res;

  res.val[0] = vld1q_s16(ptr + 0);
  res.val[1] = vld1q_s16(ptr + 8);

  return res;
}

typedef struct ggml_uint8x16x2_t {
  uint8x16_t val[2];
} ggml_uint8x16x2_t;

inline static ggml_uint8x16x2_t ggml_vld1q_u8_x2(const uint8_t *ptr) {
  ggml_uint8x16x2_t res;

  res.val[0] = vld1q_u8(ptr + 0);
  res.val[1] = vld1q_u8(ptr + 16);

  return res;
}

typedef struct ggml_uint8x16x4_t {
  uint8x16_t val[4];
} ggml_uint8x16x4_t;

inline static ggml_uint8x16x4_t ggml_vld1q_u8_x4(const uint8_t *ptr) {
  ggml_uint8x16x4_t res;

  res.val[0] = vld1q_u8(ptr + 0);
  res.val[1] = vld1q_u8(ptr + 16);
  res.val[2] = vld1q_u8(ptr + 32);
  res.val[3] = vld1q_u8(ptr + 48);

  return res;
}

typedef struct ggml_int8x16x2_t {
  int8x16_t val[2];
} ggml_int8x16x2_t;

inline static ggml_int8x16x2_t ggml_vld1q_s8_x2(const int8_t *ptr) {
  ggml_int8x16x2_t res;

  res.val[0] = vld1q_s8(ptr + 0);
  res.val[1] = vld1q_s8(ptr + 16);

  return res;
}

typedef struct ggml_int8x16x4_t {
  int8x16_t val[4];
} ggml_int8x16x4_t;

inline static ggml_int8x16x4_t ggml_vld1q_s8_x4(const int8_t *ptr) {
  ggml_int8x16x4_t res;

  res.val[0] = vld1q_s8(ptr + 0);
  res.val[1] = vld1q_s8(ptr + 16);
  res.val[2] = vld1q_s8(ptr + 32);
  res.val[3] = vld1q_s8(ptr + 48);

  return res;
}

// NOTE: not tested
inline static int8x16_t ggml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
  int8x16_t res;

  res[0] = a[b[0]];
  res[1] = a[b[1]];
  res[2] = a[b[2]];
  res[3] = a[b[3]];
  res[4] = a[b[4]];
  res[5] = a[b[5]];
  res[6] = a[b[6]];
  res[7] = a[b[7]];
  res[8] = a[b[8]];
  res[9] = a[b[9]];
  res[10] = a[b[10]];
  res[11] = a[b[11]];
  res[12] = a[b[12]];
  res[13] = a[b[13]];
  res[14] = a[b[14]];
  res[15] = a[b[15]];

  return res;
}

// NOTE: not tested
inline static uint8x16_t ggml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
  uint8x16_t res;

  res[0] = a[b[0]];
  res[1] = a[b[1]];
  res[2] = a[b[2]];
  res[3] = a[b[3]];
  res[4] = a[b[4]];
  res[5] = a[b[5]];
  res[6] = a[b[6]];
  res[7] = a[b[7]];
  res[8] = a[b[8]];
  res[9] = a[b[9]];
  res[10] = a[b[10]];
  res[11] = a[b[11]];
  res[12] = a[b[12]];
  res[13] = a[b[13]];
  res[14] = a[b[14]];
  res[15] = a[b[15]];

  return res;
}

#else

#define ggml_int16x8x2_t int16x8x2_t
#define ggml_uint8x16x2_t uint8x16x2_t
#define ggml_uint8x16x4_t uint8x16x4_t
#define ggml_int8x16x2_t int8x16x2_t
#define ggml_int8x16x4_t int8x16x4_t

#define ggml_vld1q_s16_x2 vld1q_s16_x2
#define ggml_vld1q_u8_x2 vld1q_u8_x2
#define ggml_vld1q_u8_x4 vld1q_u8_x4
#define ggml_vld1q_s8_x2 vld1q_s8_x2
#define ggml_vld1q_s8_x4 vld1q_s8_x4
#define ggml_vqtbl1q_s8 vqtbl1q_s8
#define ggml_vqtbl1q_u8 vqtbl1q_u8

#endif // !defined(__aarch64__)

#if !defined(__ARM_FEATURE_DOTPROD)

inline static int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a,
                                       int8x16_t b) {
  const int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
  const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));

  return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#else

#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#endif // !defined(__ARM_FEATURE_DOTPROD)

#endif // defined(__ARM_NEON)

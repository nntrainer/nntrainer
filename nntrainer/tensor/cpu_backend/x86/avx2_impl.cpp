// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 *
 * @file   avx2_impl.cpp
 * @date   20 Feb 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a source for AVX implementation
 *
 */

#include "avx2_impl.h"
#include "avx2_internal.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fallback_internal.h>
#include <fp16.h>
#include <immintrin.h>
#include <limits>
#include <nntrainer_error.h>
#include <thread_manager.h>
#include <type_traits>
#include <util_func.h>
#include <vector>

#include "nntr_ggml_impl_common.h"

#if !defined(_MSC_VER) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#if defined(__clang__) || defined(__GNUC__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

namespace nntrainer::avx2 {

// Shared SIMD primitives defined in avx2_internal.h that this file depends on.
using nntrainer::avx2::internal::avx2_approx_swiglu;
using nntrainer::avx2::internal::avx2_approx_swiglu_alpha;
using nntrainer::avx2::internal::exp256_ps;
using nntrainer::avx2::internal::hsum_avx;
using nntrainer::avx2::internal::poly_gelu_erf_avx2;
using nntrainer::avx2::internal::poly_gelu_tanh_avx2;
using nntrainer::avx2::internal::rcp_ps;

/**
 * @brief struct of q4_0x8 block
 */
struct block_q4_0x8 {
  uint16_t d[8];   // 16B
  uint8_t qs[128]; // 16 x u64
};

#define USE_NONTEMPORAL_STORES 1

static inline void store256_u16(void *dst, __m256i v) {
#if defined(USE_NONTEMPORAL_STORES)
  // use NT only if 32B-aligned; otherwise fall back (correctness first)
  if (((uintptr_t)dst & 31u) == 0) {
    _mm256_stream_si256((__m256i *)dst, v);
    return;
  }
#endif
  _mm256_storeu_si256((__m256i *)dst, v);
}

void unpack_q4_0x8_transpose16(const void *src, unsigned short *__restrict dT,
                               unsigned short *__restrict qsT, int N, int K,
                               int CT) // column tile (in units of 32-cols)
{
  assert((K % 256) == 0);
  assert((N % 8) == 0);

  const auto *__restrict x = static_cast<const block_q4_0x8 *>(src);

  const int groups_N8 = N / 8;    // number of 8-row groups
  const int cols_scales = K / 32; // K subblocks

  // AVX2 constants
  const __m128i v88 = _mm_set1_epi8((char)0x88);
  const __m128i v0f = _mm_set1_epi8((char)0x0F);
  const __m128i vF0 = _mm_set1_epi8((char)0xF0);

  const __m128i idx_even =
    _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, (char)0xFF, (char)0xFF, (char)0xFF,
                  (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF);
  const __m128i idx_odd =
    _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15, (char)0xFF, (char)0xFF, (char)0xFF,
                  (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF);
  const __m128i idx_0246 =
    _mm_setr_epi8(0, 2, 4, 6, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF,
                  (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF,
                  (char)0xFF, (char)0xFF, (char)0xFF);
  const __m128i idx_1357 =
    _mm_setr_epi8(1, 3, 5, 7, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF,
                  (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF, (char)0xFF,
                  (char)0xFF, (char)0xFF, (char)0xFF);

  auto pack_row8 = [&](const unsigned char *qs0, const unsigned char *qs1,
                       int off) -> __m128i {
    __m128i lo8 = _mm_loadl_epi64((const __m128i *)(qs0 + 8 * off));
    __m128i hi8 = _mm_loadl_epi64((const __m128i *)(qs1 + 8 * off));
    __m128i v = _mm_unpacklo_epi64(lo8, hi8);
    v = _mm_xor_si128(v, v88);
    __m128i lo = _mm_and_si128(v, v0f);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), v0f);
    __m128i lo_e = _mm_shuffle_epi8(lo, idx_even);
    __m128i lo_o = _mm_shuffle_epi8(lo, idx_odd);
    __m128i hi_e = _mm_shuffle_epi8(hi, idx_even);
    __m128i hi_o = _mm_shuffle_epi8(hi, idx_odd);
    __m128i low_lane =
      _mm_or_si128(lo_e, _mm_and_si128(_mm_slli_epi16(lo_o, 4), vF0));
    __m128i high_lane =
      _mm_or_si128(hi_e, _mm_and_si128(_mm_slli_epi16(hi_o, 4), vF0));
    __m128i low_e2 = _mm_shuffle_epi8(low_lane, idx_0246);
    __m128i low_o2 = _mm_shuffle_epi8(low_lane, idx_1357);
    __m128i high_e2 = _mm_shuffle_epi8(high_lane, idx_0246);
    __m128i high_o2 = _mm_shuffle_epi8(high_lane, idx_1357);
    __m128i pack_lo = _mm_unpacklo_epi8(low_e2, low_o2);   // 4×u16 (w0..w3)
    __m128i pack_hi = _mm_unpacklo_epi8(high_e2, high_o2); // 4×u16 (w4..w7)
    return _mm_unpacklo_epi64(pack_lo, pack_hi);           // 8×u16 (w0..w7)
  };

  auto transpose8x8_epi16 =
    [](__m128i r0, __m128i r1, __m128i r2, __m128i r3, __m128i r4, __m128i r5,
       __m128i r6, __m128i r7, __m128i &c0, __m128i &c1, __m128i &c2,
       __m128i &c3, __m128i &c4, __m128i &c5, __m128i &c6, __m128i &c7) {
      __m128i t0 = _mm_unpacklo_epi16(r0, r1);
      __m128i t1 = _mm_unpackhi_epi16(r0, r1);
      __m128i t2 = _mm_unpacklo_epi16(r2, r3);
      __m128i t3 = _mm_unpackhi_epi16(r2, r3);
      __m128i t4 = _mm_unpacklo_epi16(r4, r5);
      __m128i t5 = _mm_unpackhi_epi16(r4, r5);
      __m128i t6 = _mm_unpacklo_epi16(r6, r7);
      __m128i t7 = _mm_unpackhi_epi16(r6, r7);

      __m128i u0 = _mm_unpacklo_epi32(t0, t2);
      __m128i u1 = _mm_unpackhi_epi32(t0, t2);
      __m128i u2 = _mm_unpacklo_epi32(t1, t3);
      __m128i u3 = _mm_unpackhi_epi32(t1, t3);
      __m128i u4 = _mm_unpacklo_epi32(t4, t6);
      __m128i u5 = _mm_unpackhi_epi32(t4, t6);
      __m128i u6 = _mm_unpacklo_epi32(t5, t7);
      __m128i u7 = _mm_unpackhi_epi32(t5, t7);

      c0 = _mm_unpacklo_epi64(u0, u4);
      c1 = _mm_unpackhi_epi64(u0, u4);
      c2 = _mm_unpacklo_epi64(u1, u5);
      c3 = _mm_unpackhi_epi64(u1, u5);
      c4 = _mm_unpacklo_epi64(u2, u6);
      c5 = _mm_unpackhi_epi64(u2, u6);
      c6 = _mm_unpacklo_epi64(u3, u7);
      c7 = _mm_unpackhi_epi64(u3, u7);
    };

  // -------- pair-processing path: handle two 8-row groups (16 rows) per pass
  // --------
  const int groups_pairs = groups_N8 / 2;

  {
    const int cols_chunks = (cols_scales + CT - 1) / CT;
    auto &tm = nntrainer::ThreadManager::Global();
    tm.parallel_for(
      0, static_cast<size_t>(cols_chunks * groups_pairs), [&](size_t idx) {
        int c0 = (static_cast<int>(idx) / groups_pairs) * CT;
        int bp = static_cast<int>(idx) % groups_pairs;
        const int b0 = 2 * bp;
        const int b1 = b0 + 1;
        const int r0 = b0 * 8; // 16 rows: r0..r0+15
        const int c1 = std::min(c0 + CT, cols_scales);

        for (int c = c0; c < c1; ++c) {
          const block_q4_0x8 &A = x[b0 * cols_scales + c];
          const block_q4_0x8 &B = x[b1 * cols_scales + c];

          unsigned short *__restrict dT_c = dT + c * N;
          unsigned short *__restrict qsT_c0 = qsT + (c * 8) * N;

          // scales: pack two 8×u16 vectors → one 256b store to dT[c, r0..r0+15]
          __m128i sd0 = _mm_loadu_si128((const __m128i *)A.d);
          __m128i sd1 = _mm_loadu_si128((const __m128i *)B.d);
          __m256i sdp = _mm256_set_m128i(sd1, sd0);
          store256_u16(dT_c + r0, sdp);

          // pre-split stripes
          const unsigned char *__restrict A0 = A.qs;      // + 8*off
          const unsigned char *__restrict A1 = A.qs + 64; // + 8*off
          const unsigned char *__restrict B0 = B.qs;
          const unsigned char *__restrict B1 = B.qs + 64;

          // build 8 rows for A and 8 rows for B
          __m128i Ra[8], Rb[8];
          for (int off = 0; off < 8; ++off) {
            Ra[off] = pack_row8(A0, A1, off);
            Rb[off] = pack_row8(B0, B1, off);
          }

          // 8×8 transpose → columns (each 8×u16) for A and B
          __m128i Ca0, Ca1, Ca2, Ca3, Ca4, Ca5, Ca6, Ca7;
          __m128i Cb0, Cb1, Cb2, Cb3, Cb4, Cb5, Cb6, Cb7;
          transpose8x8_epi16(Ra[0], Ra[1], Ra[2], Ra[3], Ra[4], Ra[5], Ra[6],
                             Ra[7], Ca0, Ca1, Ca2, Ca3, Ca4, Ca5, Ca6, Ca7);
          transpose8x8_epi16(Rb[0], Rb[1], Rb[2], Rb[3], Rb[4], Rb[5], Rb[6],
                             Rb[7], Cb0, Cb1, Cb2, Cb3, Cb4, Cb5, Cb6, Cb7);

          // pair and store 32B per column t: rows r0..r0+15 are contiguous
          unsigned short *__restrict base = qsT_c0 + r0;
          const int S = N;
          store256_u16(base + 0 * S, _mm256_set_m128i(Cb0, Ca0));
          store256_u16(base + 1 * S, _mm256_set_m128i(Cb1, Ca1));
          store256_u16(base + 2 * S, _mm256_set_m128i(Cb2, Ca2));
          store256_u16(base + 3 * S, _mm256_set_m128i(Cb3, Ca3));
          store256_u16(base + 4 * S, _mm256_set_m128i(Cb4, Ca4));
          store256_u16(base + 5 * S, _mm256_set_m128i(Cb5, Ca5));
          store256_u16(base + 6 * S, _mm256_set_m128i(Cb6, Ca6));
          store256_u16(base + 7 * S, _mm256_set_m128i(Cb7, Ca7));
        }
      });
  }

  // -------- tail: if odd number of 8-row groups, process the last one (8 rows)
  // --------
  if (groups_N8 & 1) {
    const int b = groups_N8 - 1;
    const int r0 = b * 8;

    const int cols_chunks = (cols_scales + CT - 1) / CT;
    auto &tm = nntrainer::ThreadManager::Global();
    tm.parallel_for(0, static_cast<size_t>(cols_chunks), [&](size_t chunk_idx) {
      int c0 = static_cast<int>(chunk_idx) * CT;
      const int c1 = std::min(c0 + CT, cols_scales);
      for (int c = c0; c < c1; ++c) {
        const block_q4_0x8 &A = x[b * cols_scales + c];
        unsigned short *__restrict dT_c = dT + c * N;
        unsigned short *__restrict qsT_c0 = qsT + (c * 8) * N;

        // scales (8×u16)
        __m128i sd0 = _mm_loadu_si128((const __m128i *)A.d);
        _mm_storeu_si128((__m128i *)(dT_c + r0), sd0);

        const unsigned char *__restrict A0 = A.qs;
        const unsigned char *__restrict A1 = A.qs + 64;

        __m128i R[8];
        for (int off = 0; off < 8; ++off)
          R[off] = pack_row8(A0, A1, off);

        __m128i C0, C1, C2, C3, C4, C5, C6, C7;
        transpose8x8_epi16(R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], C0,
                           C1, C2, C3, C4, C5, C6, C7);

        unsigned short *__restrict base = qsT_c0 + r0;
        const int S = N;
        _mm_storeu_si128((__m128i *)(base + 0 * S), C0);
        _mm_storeu_si128((__m128i *)(base + 1 * S), C1);
        _mm_storeu_si128((__m128i *)(base + 2 * S), C2);
        _mm_storeu_si128((__m128i *)(base + 3 * S), C3);
        _mm_storeu_si128((__m128i *)(base + 4 * S), C4);
        _mm_storeu_si128((__m128i *)(base + 5 * S), C5);
        _mm_storeu_si128((__m128i *)(base + 6 * S), C6);
        _mm_storeu_si128((__m128i *)(base + 7 * S), C7);
      }
    });
  }

#if defined(USE_NONTEMPORAL_STORES)
  _mm_sfence(); // ensure NT stores are globally visible before returning
#endif
}

static inline __m256i butterfly32(__m256i a) {
  const __m256i SHUF_EVEN = _mm256_setr_epi8(
    0, 2, 4, 6, 8, 10, 12, 14, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
    (char)0x80, (char)0x80, (char)0x80, (char)0x80, 0, 2, 4, 6, 8, 10, 12, 14,
    (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
    (char)0x80, (char)0x80);
  const __m256i SHUF_ODD = _mm256_setr_epi8(
    1, 3, 5, 7, 9, 11, 13, 15, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
    (char)0x80, (char)0x80, (char)0x80, (char)0x80, 1, 3, 5, 7, 9, 11, 13, 15,
    (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80, (char)0x80,
    (char)0x80, (char)0x80);
  const __m256i even = _mm256_shuffle_epi8(a, SHUF_EVEN);
  const __m256i odd = _mm256_shuffle_epi8(a, SHUF_ODD);
  const __m256i LO = _mm256_set1_epi8(0x0F);
  const __m256i HI = _mm256_set1_epi8((char)0xF0);
  __m256i low =
    _mm256_or_si256(_mm256_and_si256(even, LO),
                    _mm256_slli_epi16(_mm256_and_si256(odd, LO), 4));
  __m256i high =
    _mm256_or_si256(_mm256_srli_epi16(_mm256_and_si256(even, HI), 4),
                    _mm256_and_si256(odd, HI));
  high = _mm256_slli_si256(high, 8);
  return _mm256_or_si256(low, high);
}

// Build 16B packet [d0|d1] from two 8B chunks using vector loads (no GPR
// moves).
static inline __m128i make_pkt128(const uint8_t *base_qs, int d0, int d1) {
  __m128i lo = _mm_loadl_epi64((const __m128i *)(base_qs + ((size_t)d0 << 3)));
  __m128i hi = _mm_loadl_epi64((const __m128i *)(base_qs + ((size_t)d1 << 3)));
  return _mm_unpacklo_epi64(lo, hi);
}

// ================== core template with QS unrolled by 8 blocks
// ==================
template <int UNIT, int GROUPS>
static inline void convert_q4_0x8_noshuffle(const void *src,
                                            uint16_t *RESTRICT d_out,
                                            uint8_t *RESTRICT qs_out) {
  static_assert(UNIT % 16 == 0, "UNIT must be multiple of 16");
  constexpr int BLOCKS_PER_GROUP = UNIT / 8;  // d entries per offset per group
  constexpr int PAIRS_PER_OFFSET = UNIT / 16; // 16B packets per half per offset
  static_assert((PAIRS_PER_OFFSET % 4) == 0,
                "need multiple of 4 packets (8 blocks) per iter");

  constexpr size_t D_ELEMS_PER_GROUP = 8 * BLOCKS_PER_GROUP;
  constexpr size_t QS_BYTES_PER_GROUP = (size_t)16 * UNIT;
  constexpr size_t QS_BYTES_PER_OFFSET = (size_t)2 * UNIT;

  const block_q4_0x8 *x = (const block_q4_0x8 *)src;
  const __m256i bias256 = _mm256_set1_epi8((char)0x88);

  {
    auto &tm = nntrainer::ThreadManager::Global();
    tm.parallel_for(0, static_cast<size_t>(GROUPS * 8), [&](size_t idx) {
      int b = static_cast<int>(idx) / 8;
      int offset = static_cast<int>(idx) % 8;

      // ---- D slice ----
      {
        uint16_t *d_ptr = d_out + (size_t)b * D_ELEMS_PER_GROUP +
                          (size_t)offset * BLOCKS_PER_GROUP;
        const block_q4_0x8 *xb = x + (size_t)b * BLOCKS_PER_GROUP;
        for (int i = 0; i < BLOCKS_PER_GROUP; ++i) {
          d_ptr[i] = xb[i].d[offset];
        }
      }

      // ---- QS slice (unroll 8 blocks / 128B per iter) ----
      {
        uint8_t *qs_ptr = qs_out + (size_t)b * QS_BYTES_PER_GROUP +
                          (size_t)offset * QS_BYTES_PER_OFFSET;
        const int base_q = (b * UNIT * 2) + offset;
        const int d0 = (base_q & 15), d1 = d0 ^ 8;

        auto do_half = [&](int blk_base) {
          // Each iter handles 8 consecutive blocks: j..j+7
          for (int j = 0; j < PAIRS_PER_OFFSET; j += 8) {
            const uint8_t *q0 = x[blk_base + j + 0].qs;
            const uint8_t *q1 = x[blk_base + j + 1].qs;
            const uint8_t *q2 = x[blk_base + j + 2].qs;
            const uint8_t *q3 = x[blk_base + j + 3].qs;
            const uint8_t *q4 = x[blk_base + j + 4].qs;
            const uint8_t *q5 = x[blk_base + j + 5].qs;
            const uint8_t *q6 = x[blk_base + j + 6].qs;
            const uint8_t *q7 = x[blk_base + j + 7].qs;

#if Q4X8_PREFETCH_DIST > 0
            _mm_prefetch(
              (const char *)(x[blk_base + j + Q4X8_PREFETCH_DIST].qs),
              _MM_HINT_NTA);
#endif
            // Build 8 packets in XMM regs
            __m128i pkt0 = make_pkt128(q0, d0, d1);
            __m128i pkt1 = make_pkt128(q1, d0, d1);
            __m128i pkt2 = make_pkt128(q2, d0, d1);
            __m128i pkt3 = make_pkt128(q3, d0, d1);
            __m128i pkt4 = make_pkt128(q4, d0, d1);
            __m128i pkt5 = make_pkt128(q5, d0, d1);
            __m128i pkt6 = make_pkt128(q6, d0, d1);
            __m128i pkt7 = make_pkt128(q7, d0, d1);

            // Four 32B batches: [0|1], [2|3], [4|5], [6|7]
            __m256i v01 = _mm256_set_m128i(pkt1, pkt0);
            __m256i v23 = _mm256_set_m128i(pkt3, pkt2);
            __m256i v45 = _mm256_set_m128i(pkt5, pkt4);
            __m256i v67 = _mm256_set_m128i(pkt7, pkt6);

            v01 = _mm256_xor_si256(v01, bias256);
            v23 = _mm256_xor_si256(v23, bias256);
            v45 = _mm256_xor_si256(v45, bias256);
            v67 = _mm256_xor_si256(v67, bias256);

            __m256i o01 = butterfly32(v01);
            __m256i o23 = butterfly32(v23);
            __m256i o45 = butterfly32(v45);
            __m256i o67 = butterfly32(v67);

#if Q4X8_USE_STREAMING_STORES
            _mm256_stream_si256((__m256i *)(qs_ptr + 0), o01);
            _mm256_stream_si256((__m256i *)(qs_ptr + 32), o23);
            _mm256_stream_si256((__m256i *)(qs_ptr + 64), o45);
            _mm256_stream_si256((__m256i *)(qs_ptr + 96), o67);
#else
            _mm256_storeu_si256((__m256i *)(qs_ptr + 0), o01);
            _mm256_storeu_si256((__m256i *)(qs_ptr + 32), o23);
            _mm256_storeu_si256((__m256i *)(qs_ptr + 64), o45);
            _mm256_storeu_si256((__m256i *)(qs_ptr + 96), o67);
#endif
            qs_ptr += 128;
          }
        };

        // first half
        do_half(base_q >> 4);
        // second half (same d0/d1 pattern)
        do_half((base_q + UNIT) >> 4);
      }
    });
  }

#if Q4X8_USE_STREAMING_STORES
  _mm_sfence();
#endif
}

// ================== wrappers for your K,N combinations ==================
// K = 3072 (UNIT = 768)
/**
 * @brief convert_q4_0x8_shuffle for K=3072 N=98304
 */
void convert_q4_0x8_shuffle_K3072_N98304(const void *src, uint16_t *d_out,
                                         uint8_t *qs_out) {
  // groups = (N*8)/UNIT = 1024
  convert_q4_0x8_noshuffle<768, 1024>(src, d_out, qs_out);
}

/**
 * @brief convert_q4_0x8_shuffle for K=3072 N=36864
 */
void convert_q4_0x8_shuffle_K3072_N36864(const void *src, uint16_t *d_out,
                                         uint8_t *qs_out) {
  // groups = 384
  convert_q4_0x8_noshuffle<768, 384>(src, d_out, qs_out);
}

/**
 * @brief convert_q4_0x8_shuffle for K=3072 N=3072
 */
void convert_q4_0x8_shuffle_K3072_N3072(const void *src, uint16_t *d_out,
                                        uint8_t *qs_out) {
  // groups = 32
  convert_q4_0x8_noshuffle<768, 32>(src, d_out, qs_out);
}

// K = 8192 (UNIT = 2048)
/**
 * @brief convert_q4_0x8_shuffle for K=8192 N=98304
 */
void convert_q4_0x8_shuffle_K8192_N98304(const void *src, uint16_t *d_out,
                                         uint8_t *qs_out) {
  // groups = 384
  convert_q4_0x8_noshuffle<2048, 384>(src, d_out, qs_out);
}

/**
 * @brief convert_q4_0x8_shuffle for K=8192 N=36864
 */
void convert_q4_0x8_shuffle_K8192_N36864(const void *src, uint16_t *d_out,
                                         uint8_t *qs_out) {
  // groups = 144
  convert_q4_0x8_noshuffle<2048, 144>(src, d_out, qs_out);
}

/**
 * @brief convert_q4_0x8_shuffle for K=8192 N=3072
 */
void convert_q4_0x8_shuffle_K8192_N3072(const void *src, uint16_t *d_out,
                                        uint8_t *qs_out) {
  // groups = 12
  convert_q4_0x8_noshuffle<2048, 12>(src, d_out, qs_out);
}

// Optional tiny dispatcher if you want one entry point:
void convert_q4_0x8_shuffle_dispatch_avx(const void *src, uint16_t *d_out,
                                         uint8_t *qs_out, int N, int K) {
  if (K == 3072) {
    if (N == 98304)
      return convert_q4_0x8_shuffle_K3072_N98304(src, d_out, qs_out);
    if (N == 36864)
      return convert_q4_0x8_shuffle_K3072_N36864(src, d_out, qs_out);
    if (N == 3072)
      return convert_q4_0x8_shuffle_K3072_N3072(src, d_out, qs_out);
  } else { // K == 8192
    if (N == 98304)
      return convert_q4_0x8_shuffle_K8192_N98304(src, d_out, qs_out);
    if (N == 36864)
      return convert_q4_0x8_shuffle_K8192_N36864(src, d_out, qs_out);
    if (N == 3072)
      return convert_q4_0x8_shuffle_K8192_N3072(src, d_out, qs_out);
  }
  // If a new combo appears, fall back to a generic version (not shown here).
  assert(!"Unsupported (K,N) combination");
}

bool is_valid(const unsigned int N, const float *input) {
  assert(N != 0);
  assert(input != NULL);

  int temp = 0;
  unsigned int idx = 0;

  const __m256 SIGN_MASK = _mm256_set1_ps(-0.0);
  const __m256 INF = _mm256_set1_ps(std::numeric_limits<float>::infinity());

  // 16 single-precision check : ( X != X )
  for (; N - idx >= 16; idx += 16) {
    __m256 vec0 = _mm256_loadu_ps(input);
    __m256 vec1 = _mm256_loadu_ps(input + 8);
    input += 16;
    __m256 res = _mm256_cmp_ps(vec0, vec0, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);

    if (temp)
      return false;

    // check infinity in vec0
    vec0 = _mm256_andnot_ps(SIGN_MASK, vec0);
    vec0 = _mm256_cmp_ps(vec0, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec0);
    if (temp)
      return false;

    __m256 res1 = _mm256_cmp_ps(vec1, vec1, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res1);

    if (temp)
      return false;

    // check infinity in vec1
    vec1 = _mm256_andnot_ps(SIGN_MASK, vec1);
    vec1 = _mm256_cmp_ps(vec1, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec1);

    if (temp)
      return false;
  }

  // 8 single-precision check : ( X != X )
  for (; N - idx >= 8; idx += 8) {
    __m256 vec = _mm256_loadu_ps(input);
    input += 8;
    __m256 res = _mm256_cmp_ps(vec, vec, _CMP_NEQ_UQ);
    temp = temp | _mm256_movemask_ps(res);

    if (temp)
      return false;

    // check infinity in vec
    vec = _mm256_andnot_ps(SIGN_MASK, vec);
    vec = _mm256_cmp_ps(vec, INF, _CMP_EQ_OQ);

    temp = temp | _mm256_movemask_ps(vec);

    if (temp)
      return false;
  }

  while (idx < N) {
    if (!isFloatValid(*input)) {
      return false;
    }
    ++input;
    ++idx;
  }

  return true;
}

void custom_scopy(const unsigned int N, const float *X, const int incX,
                  float *Y, const int incY) {
  unsigned int N8 = (N >> 3) << 3;
  for (unsigned int i = 0; i < N8; i += 8) {
#if defined(_WIN32)
    __m256 temp = _mm256_loadu_ps(&X[i]);
    _mm256_storeu_ps(&Y[i], temp);
#else
    __asm__ __volatile__("vmovups (%1), %%ymm0\n\t"
                         "vmovups %%ymm0, (%0)\n\t"
                         :
                         : "r"(&Y[i]), "r"(&X[i])
                         : "ymm0", "memory");
#endif
  }
  for (unsigned int i = N8; i < N; ++i) {
    Y[i] = X[i];
  }
}

void transpose_matrix(const unsigned int M, const unsigned int N,
                      const float *src, unsigned int ld_src, float *dst,
                      unsigned int ld_dst) {
  unsigned int vindexm[8] = {0,          ld_src,     ld_src * 2, ld_src * 3,
                             ld_src * 4, ld_src * 5, ld_src * 6, ld_src * 7};
  __m256i vindex = _mm256_loadu_si256((__m256i *)&vindexm[0]);
  __m256 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8;

  unsigned int M8 = (M & ~(7));
  unsigned int N8 = (N & ~(7));
  for (unsigned int i = 0; i < M8; i += 8) {
    for (unsigned int j = 0; j < N8; j += 8) {
      // loading from columns
      vec1 = _mm256_i32gather_ps(&src[ld_src * i + j + 0], vindex, 4);
      vec2 = _mm256_i32gather_ps(&src[ld_src * i + j + 1], vindex, 4);
      vec3 = _mm256_i32gather_ps(&src[ld_src * i + j + 2], vindex, 4);
      vec4 = _mm256_i32gather_ps(&src[ld_src * i + j + 3], vindex, 4);
      vec5 = _mm256_i32gather_ps(&src[ld_src * i + j + 4], vindex, 4);
      vec6 = _mm256_i32gather_ps(&src[ld_src * i + j + 5], vindex, 4);
      vec7 = _mm256_i32gather_ps(&src[ld_src * i + j + 6], vindex, 4);
      vec8 = _mm256_i32gather_ps(&src[ld_src * i + j + 7], vindex, 4);

      // storing to the rows
      _mm256_storeu_ps(&dst[(j + 0) * ld_dst + i], vec1);
      _mm256_storeu_ps(&dst[(j + 1) * ld_dst + i], vec2);
      _mm256_storeu_ps(&dst[(j + 2) * ld_dst + i], vec3);
      _mm256_storeu_ps(&dst[(j + 3) * ld_dst + i], vec4);
      _mm256_storeu_ps(&dst[(j + 4) * ld_dst + i], vec5);
      _mm256_storeu_ps(&dst[(j + 5) * ld_dst + i], vec6);
      _mm256_storeu_ps(&dst[(j + 6) * ld_dst + i], vec7);
      _mm256_storeu_ps(&dst[(j + 7) * ld_dst + i], vec8);
    }
  }

  // tailing right
  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = N8; j < N; j++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  }

  // tailing bottom
  for (unsigned int i = M8; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      dst[i + j * ld_dst] = src[i * ld_src + j];
    }
  }
}

void swiglu(const unsigned int N, float *X, const float *Y, const float *Z) {
  size_t i = 0;

  const auto oldcsr = _mm_getcsr();
  _mm_setcsr(oldcsr | 0x8040); // DAZ | FTZ

  // 16-wide blocks
  for (; i + 16 <= N; i += 16) {
    const __m256 y0 = _mm256_loadu_ps(Y + i);
    const __m256 y1 = _mm256_loadu_ps(Y + i + 8);
    const __m256 z0 = _mm256_loadu_ps(Z + i);
    const __m256 z1 = _mm256_loadu_ps(Z + i + 8);

    _mm256_storeu_ps(X + i, avx2_approx_swiglu(y0, z0));
    _mm256_storeu_ps(X + i + 8, avx2_approx_swiglu(y1, z1));
  }

  // One 8-wide block if available
  if (i + 8 <= N) {
    const __m256 y0 = _mm256_loadu_ps(Y + i);
    const __m256 z0 = _mm256_loadu_ps(Z + i);
    _mm256_storeu_ps(X + i, avx2_approx_swiglu(y0, z0));
    i += 8;
  }

  // Remaining 1..7 elements via maskload/maskstore
  if (i < N) {
    const int remain = static_cast<int>(N - i); // 1..7

    alignas(64) const int mtab[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
                                      0,  0,  0,  0,  0,  0,  0,  0};
    // Start so that we take 'remain' ones then zeros.
    const int off = 8 - remain; // in [1..7], or 0 if remain==8
    const __m256i vmask = _mm256_loadu_si256((const __m256i *)(mtab + off));

    const __m256 y = _mm256_maskload_ps(Y + i, vmask);
    const __m256 z = _mm256_maskload_ps(Z + i, vmask);
    const __m256 r = avx2_approx_swiglu(y, z);
    _mm256_maskstore_ps(X + i, vmask, r);
  }

  _mm_setcsr(oldcsr);
}

void swiglu(const unsigned int N, float *X, const float *Y, const float *Z,
            float alpha) {
  size_t i = 0;

  const auto oldcsr = _mm_getcsr();
  _mm_setcsr(oldcsr | 0x8040); // DAZ | FTZ

  const __m256 alpha_vec = _mm256_set1_ps(alpha);

  // 16-wide blocks
  for (; i + 16 <= N; i += 16) {
    const __m256 y0 = _mm256_loadu_ps(Y + i);
    const __m256 y1 = _mm256_loadu_ps(Y + i + 8);
    const __m256 z0 = _mm256_loadu_ps(Z + i);
    const __m256 z1 = _mm256_loadu_ps(Z + i + 8);

    _mm256_storeu_ps(X + i, avx2_approx_swiglu_alpha(y0, z0, alpha_vec));
    _mm256_storeu_ps(X + i + 8, avx2_approx_swiglu_alpha(y1, z1, alpha_vec));
  }

  // One 8-wide block if present
  if (i + 8 <= N) {
    const __m256 y0 = _mm256_loadu_ps(Y + i);
    const __m256 z0 = _mm256_loadu_ps(Z + i);
    _mm256_storeu_ps(X + i, avx2_approx_swiglu_alpha(y0, z0, alpha_vec));
    i += 8;
  }

  // Remaining 1..7 elements via masked AVX (no stray stores)
  if (i < N) {
    const int remain = static_cast<int>(N - i); // 1..7

    alignas(64) const int mtab[16] = {
      -1, -1, -1, -1, -1, -1, -1, -1, // ones
      0,  0,  0,  0,  0,  0,  0,  0   // zeros
    };
    const int off = 8 - remain; // choose first `remain` lanes active
    const __m256i vmask = _mm256_loadu_si256((const __m256i *)(mtab + off));

    const __m256 y = _mm256_maskload_ps(Y + i, vmask);
    const __m256 z = _mm256_maskload_ps(Z + i, vmask);
    const __m256 r = avx2_approx_swiglu_alpha(y, z, alpha_vec);
    _mm256_maskstore_ps(X + i, vmask, r);
  }

  _mm_setcsr(oldcsr);
}

void tanh_gelu_v2(const unsigned int N, const float *X, float *Y) {
  unsigned int i = 0;

  for (; i + 8 <= N; i += 8) {
    __m256 x = _mm256_loadu_ps(&X[i]);
    __m256 y = poly_gelu_tanh_avx2(x);
    _mm256_storeu_ps(&Y[i], y);
  }

  for (; i < N; ++i) {
    const float x = X[i];
    Y[i] = 0.5f * x *
           (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
  }
}

void gelu_v2(const unsigned int N, const float *X, float *Y) {
  unsigned int i = 0;

  for (; i + 8 <= N; i += 8) {
    __m256 x = _mm256_loadu_ps(&X[i]);
    __m256 y = poly_gelu_erf_avx2(x);
    _mm256_storeu_ps(&Y[i], y);
  }

  for (; i < N; ++i) {
    const float x = X[i];
    Y[i] = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
  }
}

void ele_mul(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (N == 0)
    return; // the i_stride == 0 broadcast paths read Y[0] unconditionally
  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int N8 = (N & ~(7));
    if (i_stride == 0) {
      float vy8[8] = {Y[0], Y[0], Y[0], Y[0], Y[0], Y[0], Y[0], Y[0]};
      auto y = _mm256_loadu_ps(&vy8[0]);
      for (unsigned int i = 0; i < N8; i += 8) {
        auto x = _mm256_loadu_ps(X);
        auto z = _mm256_mul_ps(x, y);
        _mm256_storeu_ps(Z, z);
        X += 8;
        Y += i_stride * 8;
        Z += 8;
      }
    } else {
      for (unsigned int i = 0; i < N8; i += 8) {
        auto x = _mm256_loadu_ps(X);
        auto y = _mm256_loadu_ps(Y);
        auto z = _mm256_mul_ps(x, y);
        _mm256_storeu_ps(Z, z);
        X += 8;
        Y += i_stride * 8;
        Z += 8;
      }
    }
    for (unsigned int i = N8; i < N; ++i) {
      *Z = *X * *Y;
      X++;
      Y += i_stride;
      Z++;
    }
  } else {
    if (o_stride == 1 && (i_stride == 0 || i_stride == 1)) {
      unsigned int N8 = (N & ~(7));
      auto alpha_v = _mm256_set1_ps(alpha);
      auto beta_v = _mm256_set1_ps(beta);

      if (i_stride == 0) {
        auto y = _mm256_set1_ps(Y[0]);
        for (unsigned int i = 0; i < N8; i += 8) {
          auto x = _mm256_loadu_ps(X);
          auto z = _mm256_mul_ps(_mm256_mul_ps(x, alpha_v), y);
          if (beta != 0.0f) {
            auto z_old = _mm256_loadu_ps(Z);
            z = _mm256_fmadd_ps(beta_v, z_old, z);
          }
          _mm256_storeu_ps(Z, z);
          X += 8;
          Z += 8;
        }
      } else {
        for (unsigned int i = 0; i < N8; i += 8) {
          auto x = _mm256_loadu_ps(X);
          auto y = _mm256_loadu_ps(Y);
          auto z = _mm256_mul_ps(_mm256_mul_ps(x, alpha_v), y);
          if (beta != 0.0f) {
            auto z_old = _mm256_loadu_ps(Z);
            z = _mm256_fmadd_ps(beta_v, z_old, z);
          }
          _mm256_storeu_ps(Z, z);
          X += 8;
          Y += 8;
          Z += 8;
        }
      }

      for (unsigned int i = N8; i < N; ++i) {
        *Z = *X * alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
        X++;
        Y += i_stride;
        Z++;
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        *Z = *X * alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
        X += o_stride;
        Y += i_stride;
        Z += o_stride;
      }
    }
  }
}

void ele_add(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (N == 0)
    return; // the i_stride == 0 broadcast paths read Y[0] unconditionally
  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int N8 = (N & ~(7));
    if (i_stride == 0) {
      float vy8[8] = {Y[0], Y[0], Y[0], Y[0], Y[0], Y[0], Y[0], Y[0]};
      auto y = _mm256_loadu_ps(&vy8[0]);
      for (unsigned int i = 0; i < N8; i += 8) {
        auto x = _mm256_loadu_ps(X);
        auto z = _mm256_add_ps(x, y);
        _mm256_storeu_ps(Z, z);
        X += 8;
        Y += i_stride * 8;
        Z += 8;
      }
    } else {
      for (unsigned int i = 0; i < N8; i += 8) {
        auto x = _mm256_loadu_ps(X);
        auto y = _mm256_loadu_ps(Y);
        auto z = _mm256_add_ps(x, y);
        _mm256_storeu_ps(Z, z);
        X += 8;
        Y += i_stride * 8;
        Z += 8;
      }
    }
    for (unsigned int i = N8; i < N; ++i) {
      *Z = *X + *Y;
      X++;
      Y += i_stride;
      Z++;
    }
  } else {
    if (o_stride == 1 && (i_stride == 0 || i_stride == 1)) {
      unsigned int N8 = (N & ~(7));
      auto alpha_v = _mm256_set1_ps(alpha);
      auto beta_v = _mm256_set1_ps(beta);

      if (i_stride == 0) {
        auto y = _mm256_set1_ps(Y[0]);
        for (unsigned int i = 0; i < N8; i += 8) {
          auto x = _mm256_loadu_ps(X);
          auto z = _mm256_fmadd_ps(alpha_v, y, x);
          if (beta != 0.0f) {
            auto z_old = _mm256_loadu_ps(Z);
            z = _mm256_fmadd_ps(beta_v, z_old, z);
          }
          _mm256_storeu_ps(Z, z);
          X += 8;
          Z += 8;
        }
      } else {
        for (unsigned int i = 0; i < N8; i += 8) {
          auto x = _mm256_loadu_ps(X);
          auto y = _mm256_loadu_ps(Y);
          auto z = _mm256_fmadd_ps(alpha_v, y, x);
          if (beta != 0.0f) {
            auto z_old = _mm256_loadu_ps(Z);
            z = _mm256_fmadd_ps(beta_v, z_old, z);
          }
          _mm256_storeu_ps(Z, z);
          X += 8;
          Y += 8;
          Z += 8;
        }
      }

      for (unsigned int i = N8; i < N; ++i) {
        *Z = *X + alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
        X++;
        Y += i_stride;
        Z++;
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        *Z = *X + alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
        X += o_stride;
        Y += i_stride;
        Z += o_stride;
      }
    }
  }
}

void ele_sub(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (N == 0)
    return; // the i_stride == 0 broadcast paths read Y[0] unconditionally
  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int N8 = (N & ~(7));
    if (i_stride == 0) {
      auto y = _mm256_set1_ps(Y[0]);
      for (unsigned int i = 0; i < N8; i += 8) {
        auto x = _mm256_loadu_ps(X);
        auto z = _mm256_sub_ps(x, y);
        _mm256_storeu_ps(Z, z);
        X += 8;
        Z += 8;
      }
      for (unsigned int i = N8; i < N; ++i) {
        *Z = *X - Y[0];
        X++;
        Z++;
      }
    } else if (i_stride == 1) {
      for (unsigned int i = 0; i < N8; i += 8) {
        auto x = _mm256_loadu_ps(X);
        auto y = _mm256_loadu_ps(Y);
        auto z = _mm256_sub_ps(x, y);
        _mm256_storeu_ps(Z, z);
        X += 8;
        Y += 8;
        Z += 8;
      }
      for (unsigned int i = N8; i < N; ++i) {
        *Z = *X - *Y;
        X++;
        Y++;
        Z++;
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        *Z = *X - *Y;
        X++;
        Y += i_stride;
        Z++;
      }
    }
  } else {
    if (o_stride == 1 && (i_stride == 0 || i_stride == 1)) {
      unsigned int N8 = (N & ~(7));
      auto alpha_v = _mm256_set1_ps(alpha);
      auto beta_v = _mm256_set1_ps(beta);

      if (i_stride == 0) {
        auto y = _mm256_set1_ps(Y[0]);
        for (unsigned int i = 0; i < N8; i += 8) {
          auto x = _mm256_loadu_ps(X);
          auto z = _mm256_fnmadd_ps(alpha_v, y, x);
          if (beta != 0.0f) {
            auto z_old = _mm256_loadu_ps(Z);
            z = _mm256_fmadd_ps(beta_v, z_old, z);
          }
          _mm256_storeu_ps(Z, z);
          X += 8;
          Z += 8;
        }
      } else {
        for (unsigned int i = 0; i < N8; i += 8) {
          auto x = _mm256_loadu_ps(X);
          auto y = _mm256_loadu_ps(Y);
          auto z = _mm256_fnmadd_ps(alpha_v, y, x);
          if (beta != 0.0f) {
            auto z_old = _mm256_loadu_ps(Z);
            z = _mm256_fmadd_ps(beta_v, z_old, z);
          }
          _mm256_storeu_ps(Z, z);
          X += 8;
          Y += 8;
          Z += 8;
        }
      }

      for (unsigned int i = N8; i < N; ++i) {
        *Z = *X - alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
        X++;
        Y += i_stride;
        Z++;
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        *Z = *X - alpha * *Y + ((0.0f == beta) ? 0.0f : beta * *Z);
        X += o_stride;
        Y += i_stride;
        Z += o_stride;
      }
    }
  }
}

void ele_div(const unsigned int N, const float *X, const float *Y, float *Z,
             float alpha, float beta, unsigned int i_stride,
             unsigned int o_stride) {
  if (N == 0)
    return; // the i_stride == 0 broadcast paths read Y[0] unconditionally
  if (alpha == 1.0f && beta == 0.0f && o_stride == 1) {
    unsigned int N8 = (N & ~(7));
    if (i_stride == 0) {
      auto y = _mm256_set1_ps(Y[0]);
      for (unsigned int i = 0; i < N8; i += 8) {
        auto x = _mm256_loadu_ps(X);
        auto z = _mm256_div_ps(x, y);
        _mm256_storeu_ps(Z, z);
        X += 8;
        Z += 8;
      }
      for (unsigned int i = N8; i < N; ++i) {
        *Z = *X / Y[0];
        X++;
        Z++;
      }
    } else if (i_stride == 1) {
      for (unsigned int i = 0; i < N8; i += 8) {
        auto x = _mm256_loadu_ps(X);
        auto y = _mm256_loadu_ps(Y);
        auto z = _mm256_div_ps(x, y);
        _mm256_storeu_ps(Z, z);
        X += 8;
        Y += 8;
        Z += 8;
      }
      for (unsigned int i = N8; i < N; ++i) {
        *Z = *X / *Y;
        X++;
        Y++;
        Z++;
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        *Z = *X / *Y;
        X++;
        Y += i_stride;
        Z++;
      }
    }
  } else {
    if (o_stride == 1 && (i_stride == 0 || i_stride == 1)) {
      unsigned int N8 = (N & ~(7));
      auto alpha_v = _mm256_set1_ps(alpha);
      auto beta_v = _mm256_set1_ps(beta);

      if (i_stride == 0) {
        auto y = _mm256_set1_ps(Y[0]);
        auto denom = _mm256_mul_ps(alpha_v, y);
        for (unsigned int i = 0; i < N8; i += 8) {
          auto x = _mm256_loadu_ps(X);
          auto z = _mm256_div_ps(x, denom);
          if (beta != 0.0f) {
            auto z_old = _mm256_loadu_ps(Z);
            z = _mm256_fmadd_ps(beta_v, z_old, z);
          }
          _mm256_storeu_ps(Z, z);
          X += 8;
          Z += 8;
        }
      } else {
        for (unsigned int i = 0; i < N8; i += 8) {
          auto x = _mm256_loadu_ps(X);
          auto y = _mm256_loadu_ps(Y);
          auto denom = _mm256_mul_ps(alpha_v, y);
          auto z = _mm256_div_ps(x, denom);
          if (beta != 0.0f) {
            auto z_old = _mm256_loadu_ps(Z);
            z = _mm256_fmadd_ps(beta_v, z_old, z);
          }
          _mm256_storeu_ps(Z, z);
          X += 8;
          Y += 8;
          Z += 8;
        }
      }

      for (unsigned int i = N8; i < N; ++i) {
        *Z = *X / (alpha * *Y) + ((0.0f == beta) ? 0.0f : beta * *Z);
        X++;
        Y += i_stride;
        Z++;
      }
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        *Z = *X / (alpha * *Y) + ((0.0f == beta) ? 0.0f : beta * *Z);
        X += o_stride;
        Y += i_stride;
        Z += o_stride;
      }
    }
  }
}

// exp256_ps and rcp_ps are now in avx2_internal.h

static void softmax_row_inplace(float *qk_out, size_t start_row, size_t end_row,
                                size_t num_heads) {
  const size_t vec_end = num_heads & ~((size_t)7); // floor(num_heads / 8) * 8

  // 1. find max for each head (reusable thread-local scratch: 0 per-call alloc)
  // TODO: these thread_local buffers stay alive per worker thread for the
  // process lifetime (thread pool) and only grow to a high-water mark of
  // 2 * max(num_heads) floats per worker (sub-KB for typical models); they are
  // never released. Free or shrink them here if the attention memory footprint
  // needs to be reduced.
  static thread_local std::vector<float> max_vals_buf;
  static thread_local std::vector<float> sum_vals_buf;
  max_vals_buf.resize(num_heads);
  sum_vals_buf.resize(num_heads);
  float *max_vals = max_vals_buf.data();

  // initialize max_vals with first row of qk_out
  std::memcpy(max_vals, qk_out + start_row * num_heads,
              num_heads * sizeof(float));

  // update max_vals for each row
  for (size_t r = start_row + 1; r < end_row; ++r) {
    float *row = qk_out + (num_heads * r);
    for (size_t c = 0; c < vec_end; c += 8) {
      __m256 v = _mm256_loadu_ps(row + c);
      __m256 m = _mm256_loadu_ps(max_vals + c);
      m = _mm256_max_ps(v, m);
      _mm256_storeu_ps(max_vals + c, m);
    }
    for (size_t c = vec_end; c < num_heads; ++c) {
      max_vals[c] = std::max(max_vals[c], row[c]);
    }
  }

  // 2. calc exp(x - max) and sum
  float *sum_vals = sum_vals_buf.data();
  std::memset(sum_vals, 0, num_heads * sizeof(float));

  for (size_t r = start_row; r < end_row; ++r) {
    float *row = qk_out + (num_heads * r);
    for (size_t c = 0; c < vec_end; c += 8) {
      __m256 s = _mm256_loadu_ps(sum_vals + c);
      __m256 v = _mm256_loadu_ps(row + c);
      __m256 m = _mm256_loadu_ps(max_vals + c);
      __m256 d = _mm256_sub_ps(v, m);    // x - max
      __m256 e = exp256_ps(d);           // exp(x - max)
      _mm256_storeu_ps(row + c, e);      // overwrite qk_out
      s = _mm256_add_ps(s, e);           // sum += exp(x - max)
      _mm256_storeu_ps(sum_vals + c, s); // update sum_vals
    }
    for (size_t c = vec_end; c < num_heads; ++c) {
      float e = std::exp(row[c] - max_vals[c]);
      row[c] = e;
      sum_vals[c] += e;
    }
  }

  // 3. calc 1/sum
  // _mm256_div_ps is slow
  // precalculate (1/sum) and then multiply is much faster
  for (size_t c = 0; c < vec_end; c += 8) {
    __m256 s = _mm256_loadu_ps(sum_vals + c);
    s = rcp_ps(s); // sum = 1/sum
    _mm256_storeu_ps(sum_vals + c, s);
  }
  for (size_t c = vec_end; c < num_heads; ++c) {
    sum_vals[c] = 1 / sum_vals[c];
  }

  // 4. calc exp(x - max) * (1/sum)
  for (size_t r = start_row; r < end_row; ++r) {
    float *row = qk_out + (num_heads * r);
    for (size_t c = 0; c < vec_end; c += 8) {
      __m256 s = _mm256_loadu_ps(sum_vals + c); // 1/sum
      __m256 v = _mm256_loadu_ps(row + c);      // exp(x - max)
      __m256 o = _mm256_mul_ps(v, s);           // exp(x - max) * (1/sum)
      _mm256_storeu_ps(row + c, o);             // overwrite qk_out
    }
    for (size_t c = vec_end; c < num_heads; ++c) {
      row[c] *= sum_vals[c];
    }
  }
}

static void softmax_row_with_sink_inplace(float *qk_out, size_t start_row,
                                          size_t end_row, size_t num_heads,
                                          float *sink) {
  const size_t vec_end = num_heads & ~((size_t)7); // floor(num_heads / 8) * 8

  // 1. find max for each head (reusable thread-local scratch: 0 per-call alloc)
  // TODO: these thread_local buffers stay alive per worker thread for the
  // process lifetime (thread pool) and only grow to a high-water mark of
  // 2 * max(num_heads) floats per worker (sub-KB for typical models); they are
  // never released. Free or shrink them here if the attention memory footprint
  // needs to be reduced.
  static thread_local std::vector<float> max_vals_buf;
  static thread_local std::vector<float> sum_vals_buf;
  max_vals_buf.resize(num_heads);
  sum_vals_buf.resize(num_heads);
  float *max_vals = max_vals_buf.data();

  // initialize max_vals with sink
  std::memcpy(max_vals, sink, num_heads * sizeof(float));

  // update max_vals for each row
  for (size_t r = start_row; r < end_row; ++r) {
    float *row = qk_out + (num_heads * r);
    for (size_t c = 0; c < vec_end; c += 8) {
      __m256 v = _mm256_loadu_ps(row + c);
      __m256 m = _mm256_loadu_ps(max_vals + c);
      m = _mm256_max_ps(v, m);
      _mm256_storeu_ps(max_vals + c, m);
    }
    for (size_t c = vec_end; c < num_heads; ++c) {
      max_vals[c] = std::max(max_vals[c], row[c]);
    }
  }

  // 2. calc exp(x - max) and sum
  float *sum_vals = sum_vals_buf.data();
  // init sum_vals with exp(sink - max)
  {
    for (size_t c = 0; c < vec_end; c += 8) {
      __m256 v = _mm256_loadu_ps(sink + c);
      __m256 m = _mm256_loadu_ps(max_vals + c);
      __m256 d = _mm256_sub_ps(v, m); // sink - max
      __m256 e = exp256_ps(d);        // exp(sink - max)
      _mm256_storeu_ps(sum_vals + c, e);
    }
    for (size_t c = vec_end; c < num_heads; ++c) {
      float e = std::exp(sink[c] - max_vals[c]);
      sum_vals[c] = e;
    }
  }

  for (size_t r = start_row; r < end_row; ++r) {
    float *row = qk_out + (num_heads * r);
    for (size_t c = 0; c < vec_end; c += 8) {
      __m256 s = _mm256_loadu_ps(sum_vals + c);
      __m256 v = _mm256_loadu_ps(row + c);
      __m256 m = _mm256_loadu_ps(max_vals + c);
      __m256 d = _mm256_sub_ps(v, m);    // x - max
      __m256 e = exp256_ps(d);           // exp(x - max)
      _mm256_storeu_ps(row + c, e);      // overwrite qk_out
      s = _mm256_add_ps(s, e);           // sum += exp(x - max)
      _mm256_storeu_ps(sum_vals + c, s); // update sum_vals
    }
    for (size_t c = vec_end; c < num_heads; ++c) {
      float e = std::exp(row[c] - max_vals[c]);
      row[c] = e;
      sum_vals[c] += e;
    }
  }

  // 3. calc 1/sum
  // _mm256_div_ps is slow
  // precalculate (1/sum) and then multiply is much faster
  for (size_t c = 0; c < vec_end; c += 8) {
    __m256 s = _mm256_loadu_ps(sum_vals + c);
    s = rcp_ps(s); // sum = 1/sum
    _mm256_storeu_ps(sum_vals + c, s);
  }
  for (size_t c = vec_end; c < num_heads; ++c) {
    sum_vals[c] = 1 / sum_vals[c];
  }

  // 4. calc exp(x - max) * (1/sum)
  for (size_t r = start_row; r < end_row; ++r) {
    float *row = qk_out + (num_heads * r);
    for (size_t c = 0; c < vec_end; c += 8) {
      __m256 s = _mm256_loadu_ps(sum_vals + c); // 1/sum
      __m256 v = _mm256_loadu_ps(row + c);      // exp(x - max)
      __m256 o = _mm256_mul_ps(v, s);           // exp(x - max) * (1/sum)
      _mm256_storeu_ps(row + c, o);             // overwrite qk_out
    }
    for (size_t c = vec_end; c < num_heads; ++c) {
      row[c] *= sum_vals[c];
    }
  }
}

template <>
void softmax_row_inplace(float *qk_out, size_t start_row, size_t end_row,
                         size_t num_heads, float *sink) {
  if (sink == nullptr) {
    return softmax_row_inplace(qk_out, start_row, end_row, num_heads);
  } else {
    return softmax_row_with_sink_inplace(qk_out, start_row, end_row, num_heads,
                                         sink);
  }
}

static void softmax_row(float *qk_out, size_t start_row, size_t end_row,
                        size_t num_heads) {
  softmax_row_inplace(qk_out, start_row, end_row, num_heads);
}

static void softmax_row_with_sink(float *qk_out, size_t start_row,
                                  size_t end_row, size_t num_heads,
                                  float *sink) {
  softmax_row_with_sink_inplace(qk_out, start_row, end_row, num_heads, sink);
}

template <>
void softmax_row(float *qk_out, size_t start_row, size_t end_row,
                 size_t num_heads, float *sink) {
  if (sink == nullptr) {
    return softmax_row(qk_out, start_row, end_row, num_heads);
  } else {
    return softmax_row_with_sink(qk_out, start_row, end_row, num_heads, sink);
  }
}
#ifdef _WIN32
#define COMPUTE_FP16_TO_FP32(x)                                                \
  _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define COMPUTE_FP32_TO_FP16(x)                                                \
  _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)
#elif defined(__TIZEN__) && !defined(__F16C__)
#define COMPUTE_FP16_TO_FP32(x) nntrainer::compute_fp16_to_fp32(x)
#define COMPUTE_FP32_TO_FP16(x) nntrainer::compute_fp32_to_fp16(x)
#else
#define COMPUTE_FP16_TO_FP32(x) _cvtsh_ss(x)
#define COMPUTE_FP32_TO_FP16(x) _cvtss_sh(x, 0)
#endif

static inline __m256 convert_vector_f16_to_f32(__m128i x) {
#if defined(__TIZEN__) && !defined(__F16C__)
  alignas(32) uint16_t u16_array[8]; // 32-byte aligned storage
  alignas(32) float f32_array[8];    // 32-byte aligned storage

  // Safely store __m128i to array (avoids aliasing)
  _mm_storeu_si128(reinterpret_cast<__m128i *>(u16_array), x);

  // Convert each FP16 value to FP32
  for (int i = 0; i < 8; i++) {
    f32_array[i] = COMPUTE_FP16_TO_FP32(u16_array[i]);
  }

  // Load aligned array into __m256
  return _mm256_load_ps(f32_array);
#else
  return _mm256_cvtph_ps(x);
#endif
}

static inline __m128i convert_vector_f32_to_f16(__m256 x) {
#if defined(__TIZEN__) && !defined(__F16C__)
  __m128i vec_f16;
  float *f32_ptr = reinterpret_cast<float *>(&x);
  uint16_t *u16_ptr = reinterpret_cast<uint16_t *>(&vec_f16);
  for (int i = 0; i < 8; i++) {
    u16_ptr[i] = COMPUTE_FP32_TO_FP16(f32_ptr[i]);
  }
  return vec_f16;
#else
  return _mm256_cvtps_ph(x, 0);
#endif
}

static inline __m128i convert_vector_f32_to_f16(__m128 x) {
#if defined(__TIZEN__) && !defined(__F16C__)
  __m128i vec_f16;
  float *f32_ptr = reinterpret_cast<float *>(&x);
  uint16_t *u16_ptr = reinterpret_cast<uint16_t *>(&vec_f16);

  for (int i = 0; i < 4; i++) {
    u16_ptr[i] = COMPUTE_FP32_TO_FP16(f32_ptr[i]);
  }
  return vec_f16;
#else
  return _mm_cvtps_ph(x, 0);
#endif
}

static inline void load_fp16_8_to_chunk(const uint16_t *src, float *dst,
                                        int chunk_size) {
  int i = 0;
  for (; i + 8 <= chunk_size; i += 8) {
    __m128i half = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
    __m256 f32 = convert_vector_f16_to_f32(half);
    _mm256_storeu_ps(&dst[i], f32);
  }
  for (; i < chunk_size; ++i) {
    dst[i] = nntrainer::compute_fp16_to_fp32(src[i]);
  }
}

void compute_fp16vcache_fp32_transposed(int row_num, const float *in,
                                        const uint16_t *vcache, float *output,
                                        int num_cache_head, int gqa_size,
                                        int head_dim, size_t local_window_size,
                                        int head_start, int head_end) {

  // If head_end is -1, process all heads from head_start to num_cache_head.
  // No other negative values are accepted for head_end.
  int actual_head_end = (head_end < 0) ? num_cache_head : head_end;

  // Validate head range: head_start must be less than actual_head_end
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  const int num_blocks = head_dim / 8;
  const int rem = head_dim % 8;

  // Reusable thread-local scratch so the kernel performs zero per-call
  // allocations after warm-up (ThreadManager::parallel_for gives each worker
  // its own thread_local copy). sumVec holds num_blocks*gqa_size accumulator
  // vectors as a flat float buffer (8 floats each) to avoid the
  // -Wignored-attributes warning that a std::vector<__m256> would raise.
  // Resolve each thread_local data pointer once into a local so the hot loops
  // do plain pointer arithmetic (no per-access TLS/vector indirection).
  // TODO: these thread_local buffers stay alive per worker thread for the
  // process lifetime (thread pool) and only grow to a high-water mark of about
  // head_dim * (1 + gqa_size) floats per worker (a few KB for typical LLM
  // shapes); they are never released. Free or shrink them here if the
  // attention memory footprint needs to be reduced.
  static thread_local std::vector<float> tmp_fp32;
  static thread_local std::vector<float> sumVec;
  static thread_local std::vector<float> sumRem;
  tmp_fp32.resize(head_dim);
  sumVec.resize((size_t)std::max(1, num_blocks * gqa_size) * 8);
  sumRem.resize((size_t)gqa_size * rem);
  float *tmp = tmp_fp32.data();
  float *sv = sumVec.data();
  float *rem_buf = sumRem.data();

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int i = 0; i < num_blocks * gqa_size; i++) {
      _mm256_storeu_ps(&sv[(size_t)i * 8], _mm256_setzero_ps());
    }
    std::fill(sumRem.begin(), sumRem.end(), 0.0f);

    for (int j = row_num < local_window_size ? 0
                                             : row_num + 1 - local_window_size;
         j <= row_num; ++j) {
      const uint16_t *vptr = vcache + (j * num_cache_head + n) * head_dim;
      load_fp16_8_to_chunk(vptr, tmp, head_dim);

      for (int h = 0; h < gqa_size; ++h) {
        float a_val =
          in[(row_num < local_window_size
                ? j
                : (unsigned long)(j - (row_num + 1 - local_window_size))) *
               (unsigned long)(gqa_size * num_cache_head) +
             (unsigned long)(n * gqa_size) + h];

        __m256 inVec = _mm256_set1_ps(a_val);

        for (int b = 0; b < num_blocks; ++b) {
          __m256 bVec = _mm256_loadu_ps(&tmp[b * 8]);
          float *accPtr = &sv[(size_t)(h * num_blocks + b) * 8];
          _mm256_storeu_ps(
            accPtr, _mm256_fmadd_ps(inVec, bVec, _mm256_loadu_ps(accPtr)));
        }

        if (rem > 0) {
          float *remPtr = &rem_buf[(size_t)h * rem];
          int base = num_blocks * 8;
          for (int r = 0; r < rem; ++r) {
            remPtr[r] += a_val * tmp[base + r];
          }
        }
      }
    }

    for (int h = 0; h < gqa_size; ++h) {
      for (int b = 0; b < num_blocks; ++b) {
        int out_base = (n * gqa_size + h) * head_dim + b * 8;
        _mm256_storeu_ps(
          &output[out_base],
          _mm256_loadu_ps(&sv[(size_t)(h * num_blocks + b) * 8]));
      }

      if (rem > 0) {
        float *remPtr = &rem_buf[(size_t)h * rem];
        int base = num_blocks * 8;
        for (int r = 0; r < rem; ++r) {
          int out_idx = (n * gqa_size + h) * head_dim + base + r;
          output[out_idx] = remPtr[r];
        }
      }
    }
  }
}

template <>
void compute_kcaches(const float *in, const uint16_t *kcache, float *output,
                     int num_rows, int num_cache_head, int head_dim,
                     int gqa_size, int tile_size, size_t local_window_size,
                     int head_start, int head_end) {
  // If head_end is -1, process all heads from head_start to num_cache_head.
  // No other negative values are accepted for head_end.
  int actual_head_end = (head_end < 0) ? num_cache_head : head_end;

  // Validate head range: head_start must be less than actual_head_end
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  int start_row =
    num_rows < local_window_size ? 0 : num_rows - local_window_size;
  int row_cnt = num_rows < local_window_size ? num_rows : local_window_size;
  const int tile_count = (row_cnt + tile_size - 1) / tile_size;
  const float inv_sqrt_head_dim =
    1.0f / std::sqrt(static_cast<float>(head_dim));

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int t = 0; t < tile_count; ++t) {
      int row_tile_start = t * tile_size;
      int tile_rows = std::min(tile_size, row_cnt - row_tile_start);

      for (int g = 0; g < gqa_size; ++g) {
        const float *in_ptr = in + n * gqa_size * head_dim + g * head_dim;
        for (int t_row = 0; t_row < tile_rows; ++t_row) {
          int row = start_row + row_tile_start + t_row;
          if (row + 1 < num_rows) {
            const uint16_t *next_kptr =
              kcache + ((row + 1) * num_cache_head + n) * head_dim;
            _mm_prefetch(reinterpret_cast<const char *>(next_kptr),
                         _MM_HINT_T0);
          }
          const uint16_t *kptr = kcache + (row * num_cache_head + n) * head_dim;

          // Convert the FP16 key row to FP32 on the fly inside the dot product
          // (8 lanes via F16C) instead of staging it in a temporary buffer.
          float sum = 0.0f;
          int i = 0;
          __m256 acc = _mm256_setzero_ps();
          for (; i + 8 <= head_dim; i += 8) {
            __m256 va = _mm256_loadu_ps(in_ptr + i);
            __m256 vb = convert_vector_f16_to_f32(
              _mm_loadu_si128(reinterpret_cast<const __m128i *>(kptr + i)));
            acc = _mm256_fmadd_ps(va, vb, acc);
          }

          __m128 low = _mm256_castps256_ps128(acc);
          __m128 high = _mm256_extractf128_ps(acc, 1);
          __m128 sum128 = _mm_add_ps(low, high);
          sum128 = _mm_hadd_ps(sum128, sum128);
          sum128 = _mm_hadd_ps(sum128, sum128);
          sum += _mm_cvtss_f32(sum128);

          for (; i < head_dim; ++i)
            sum += in_ptr[i] * nntrainer::compute_fp16_to_fp32(kptr[i]);

          output[(row - start_row) * num_cache_head * gqa_size + n * gqa_size +
                 g] = sum * inv_sqrt_head_dim;
        }
      }
    }
  }
}

void compute_rotary_emb_value(unsigned int width, unsigned int dim,
                              unsigned int half_, float *inout, void *output,
                              const float *cos_, const float *sin_,
                              bool only_convert_to_fp16) {
  enum class OutputType { FP16, FP32 };

  OutputType out_type = OutputType::FP32;
  if (output != nullptr)
    out_type = OutputType::FP16;

  for (unsigned int w = 0; w < width; w += dim) {
    unsigned int k = 0;
    for (; k + 7 < half_; k += 8) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;

      __m256 a = _mm256_loadu_ps(&inout[i0]);
      __m256 b = _mm256_loadu_ps(&inout[i1]);

      if (only_convert_to_fp16) {
        if (out_type == OutputType::FP16) {
          __m128i a_fp16 = convert_vector_f32_to_f16(a);
          __m128i b_fp16 = convert_vector_f32_to_f16(b);

          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i0),
            a_fp16);
          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i1),
            b_fp16);
        }

      } else {
        __m256 cos_v = _mm256_loadu_ps(&cos_[k]);
        __m256 sin_v = _mm256_loadu_ps(&sin_[k]);

        __m256 out0 =
          _mm256_sub_ps(_mm256_mul_ps(a, cos_v), _mm256_mul_ps(b, sin_v));
        __m256 out1 =
          _mm256_add_ps(_mm256_mul_ps(a, sin_v), _mm256_mul_ps(b, cos_v));

        if (out_type == OutputType::FP16) {
          __m128i out0_fp16 = convert_vector_f32_to_f16(out0);
          __m128i out1_fp16 = convert_vector_f32_to_f16(out1);

          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i0),
            out0_fp16);
          _mm_storeu_si128(
            reinterpret_cast<__m128i *>(static_cast<uint16_t *>(output) + i1),
            out1_fp16);

        } else if (out_type == OutputType::FP32) {
          _mm256_storeu_ps(&inout[i0], out0);
          _mm256_storeu_ps(&inout[i1], out1);
        }
      }
    }

    for (; k < half_; ++k) {
      unsigned int i0 = w + k;
      unsigned int i1 = w + k + half_;
      // assert(i1 < width && "Scalar i1 overflow!");
      float a = inout[i0];
      float b = inout[i1];

      if (only_convert_to_fp16) {
        static_cast<uint16_t *>(output)[i0] = COMPUTE_FP32_TO_FP16(a);
        static_cast<uint16_t *>(output)[i1] = COMPUTE_FP32_TO_FP16(b);
      } else {
        float c = cos_[k];
        float s = sin_[k];

        float out0 = a * c - b * s;
        float out1 = a * s + b * c;

        if (out_type == OutputType::FP16) {
          static_cast<uint16_t *>(output)[i0] = COMPUTE_FP32_TO_FP16(out0);
          static_cast<uint16_t *>(output)[i1] = COMPUTE_FP32_TO_FP16(out1);
        } else if (out_type == OutputType::FP32) {
          inout[i0] = out0;
          inout[i1] = out1;
        }
      }
    }
  }
}

// hsum_avx is now in avx2_internal.h

void rms_norm_wrt_width_fp32_intrinsic(const float *__restrict X,
                                       float *__restrict Y, size_t H, size_t W,
                                       float epsilon) {
  for (std::size_t h = 0; h < H; ++h) {
    const float *rowX = X + h * W;
    float *rowY = Y + h * W;

    std::size_t i = 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (; i + 32 <= W; i += 32) {
      __m256 x0 = _mm256_loadu_ps(rowX + i);
      __m256 x1 = _mm256_loadu_ps(rowX + i + 8);
      __m256 x2 = _mm256_loadu_ps(rowX + i + 16);
      __m256 x3 = _mm256_loadu_ps(rowX + i + 24);
      acc0 = _mm256_fmadd_ps(x0, x0, acc0);
      acc1 = _mm256_fmadd_ps(x1, x1, acc1);
      acc2 = _mm256_fmadd_ps(x2, x2, acc2);
      acc3 = _mm256_fmadd_ps(x3, x3, acc3);
    }
    for (; i + 8 <= W; i += 8) {
      __m256 x = _mm256_loadu_ps(rowX + i);
      acc0 = _mm256_fmadd_ps(x, x, acc0);
    }
    float sumsq =
      hsum_avx(acc0) + hsum_avx(acc1) + hsum_avx(acc2) + hsum_avx(acc3);
    for (; i < W; ++i) {
      float v = rowX[i];
      sumsq += v * v;
    }

    float mean = sumsq / static_cast<float>(W);
    float scale = 1.0f / std::sqrt(mean + epsilon);
    __m256 vscale = _mm256_set1_ps(scale);

    i = 0;
    for (; i + 32 <= W; i += 32) {
      __m256 x0 = _mm256_loadu_ps(rowX + i);
      __m256 x1 = _mm256_loadu_ps(rowX + i + 8);
      __m256 x2 = _mm256_loadu_ps(rowX + i + 16);
      __m256 x3 = _mm256_loadu_ps(rowX + i + 24);
      _mm256_storeu_ps(rowY + i, _mm256_mul_ps(x0, vscale));
      _mm256_storeu_ps(rowY + i + 8, _mm256_mul_ps(x1, vscale));
      _mm256_storeu_ps(rowY + i + 16, _mm256_mul_ps(x2, vscale));
      _mm256_storeu_ps(rowY + i + 24, _mm256_mul_ps(x3, vscale));
    }
    for (; i + 8 <= W; i += 8) {
      __m256 x = _mm256_loadu_ps(rowX + i);
      _mm256_storeu_ps(rowY + i, _mm256_mul_ps(x, vscale));
    }
    for (; i < W; ++i) {
      rowY[i] = rowX[i] * scale;
    }
  }
}

template <>
void clamp(const float *input, float *output, size_t length, float lower_bound,
           float upper_bound) {
  const size_t step = 8;
  const __m256 vLo = _mm256_set1_ps(lower_bound);
  const __m256 vHi = _mm256_set1_ps(upper_bound);

  size_t i = 0;
  for (; i + step <= length; i += step) {
    __m256 v = _mm256_loadu_ps(input + i);
    v = _mm256_max_ps(v, vLo);
    v = _mm256_min_ps(v, vHi);
    _mm256_storeu_ps(output + i, v);
  }
  if (i < length) {
    for (size_t k = i; k < length; ++k) {
      float v = input[k];
      // If v is NaN, the comparisons below will yield false; we keep NaN.
      // This matches most framework "pass-through NaN" behavior.
      output[k] =
        (v < lower_bound) ? lower_bound : ((v > upper_bound) ? upper_bound : v);
    }
  }
}

void copy_f16_f32(unsigned int N, const uint16_t *input, float *output) {
  unsigned int idx = 0;
  const uint16_t *data = (const uint16_t *)input;

  // 16 half-precision floating point values to single-precision values
  for (; N - idx >= 16; idx += 16) {
    const __m256 vec0 =
      convert_vector_f16_to_f32(_mm_loadu_si128((const __m128i *)data));
    const __m256 vec1 =
      convert_vector_f16_to_f32(_mm_loadu_si128((const __m128i *)(data + 8)));
    data += 16;

    _mm256_storeu_ps(output, vec0);
    _mm256_storeu_ps(output + 8, vec1);
    output += 16;
  }
  // 8 half-precision floating point values to single-precision values
  for (; N - idx >= 8; idx += 8) {
    const __m256 vec =
      convert_vector_f16_to_f32(_mm_loadu_si128((const __m128i *)data));
    data += 8;

    _mm256_storeu_ps(output, vec);
    output += 8;
  }
  // remaining half-precision floating point values to single-precision values
  while (idx < N) {
    *output = compute_fp16_to_fp32(*data);
    ++output;
    ++data;
    ++idx;
  }
}

void copy_f32_f16(unsigned int N, const float *input, uint16_t *output) {
  unsigned int idx = 0;
  uint16_t *out_data = (uint16_t *)output;

  // 16 single-precision floating point values to half-precision values
  for (; N - idx >= 16; idx += 16) {
    const __m256 vec0 = _mm256_loadu_ps(input);
    const __m256 vec1 = _mm256_loadu_ps(input + 8);
    input += 16;

    _mm_storeu_si128((__m128i *)out_data, convert_vector_f32_to_f16(vec0));
    _mm_storeu_si128((__m128i *)(out_data + 8),
                     convert_vector_f32_to_f16(vec1));
    out_data += 16;
  }
  // 8 single-precision floating point values to half-precision values
  for (; N - idx >= 8; idx += 8) {
    const __m256 vec = _mm256_loadu_ps(input);
    input += 8;

    _mm_storeu_si128((__m128i *)out_data, convert_vector_f32_to_f16(vec));
    out_data += 8;
  }
  // 4 single-precision floating point values to half-precision values
  for (; N - idx >= 4; idx += 4) {
    const __m128 vec = _mm_loadu_ps(input);
    input += 4;

    _mm_storeu_si64((__m128i *)out_data, convert_vector_f32_to_f16(vec));
    out_data += 4;
  }
  // remaining single-precision floating point values to half-precision values
  while (idx < N) {
    *out_data = compute_fp32_to_fp16(*input);
    ++out_data;
    ++input;
    ++idx;
  }
}

void create_q4_0_weights(const uint8_t *int4_weight, uint8_t *q4_0_weight) {
  // Load 16 bytes of input data
  __m128i input = _mm_loadu_si128((const __m128i *)int4_weight);

  // Create masks for extracting low and high nibbles
  const __m128i low_nibble_mask = _mm_set1_epi8(0x0F);
  const __m128i high_nibble_mask = _mm_set1_epi8(static_cast<char>(0xF0));

  // Extract low nibbles from first 8 bytes
  __m128i A = _mm_and_si128(input, low_nibble_mask);

  // Extract high nibbles from first 8 bytes and shift right
  __m128i B = _mm_and_si128(input, high_nibble_mask);
  B = _mm_srli_epi16(B, 4);

  // Extract low nibbles from second 8 bytes
  __m128i input_shifted = _mm_bsrli_si128(input, 8);
  __m128i C = _mm_and_si128(input_shifted, low_nibble_mask);

  // Extract high nibbles from second 8 bytes and shift right
  __m128i D = _mm_and_si128(input_shifted, high_nibble_mask);
  D = _mm_srli_epi16(D, 4);

  // Interleave low nibbles: v0 from first8, v2 from second8
  __m128i AC = _mm_or_si128(A, _mm_slli_epi16(C, 4));

  // Interleave high nibbles: v1 from first8, v3 from second8
  __m128i BD = _mm_or_si128(B, _mm_slli_epi16(D, 4));

  // Pack the results: interleave low and high bytes
  __m128i result = _mm_unpacklo_epi8(AC, BD);

  // Store the 16 bytes result
  _mm_storeu_si128((__m128i *)q4_0_weight, result);
}

static inline void transpose_matrix_16x16(const uint8_t *input,
                                          int input_stride, uint8_t *output,
                                          int output_stride) {
  const uint8_t *src = input;
  uint8_t *dst = output;

  __m256i rows[8];
  for (int i = 0; i < 8; ++i) {
    rows[i] =
      _mm256_loadu2_m128i((const __m128i *)(src + (8 + i) * input_stride),
                          (const __m128i *)(src + i * input_stride));
  }

  // Step 1: Transpose within 2x2 sub-blocks
  __m256i temp0 = _mm256_unpacklo_epi8(rows[0], rows[1]);
  __m256i temp1 = _mm256_unpackhi_epi8(rows[0], rows[1]);
  __m256i temp2 = _mm256_unpacklo_epi8(rows[2], rows[3]);
  __m256i temp3 = _mm256_unpackhi_epi8(rows[2], rows[3]);
  __m256i temp4 = _mm256_unpacklo_epi8(rows[4], rows[5]);
  __m256i temp5 = _mm256_unpackhi_epi8(rows[4], rows[5]);
  __m256i temp6 = _mm256_unpacklo_epi8(rows[6], rows[7]);
  __m256i temp7 = _mm256_unpackhi_epi8(rows[6], rows[7]);

  // Step 2: Transpose within 4x4 sub-blocks
  __m256i interleave0 = _mm256_unpacklo_epi16(temp0, temp2);
  __m256i interleave1 = _mm256_unpackhi_epi16(temp0, temp2);
  __m256i interleave2 = _mm256_unpacklo_epi16(temp1, temp3);
  __m256i interleave3 = _mm256_unpackhi_epi16(temp1, temp3);
  __m256i interleave4 = _mm256_unpacklo_epi16(temp4, temp6);
  __m256i interleave5 = _mm256_unpackhi_epi16(temp4, temp6);
  __m256i interleave6 = _mm256_unpacklo_epi16(temp5, temp7);
  __m256i interleave7 = _mm256_unpackhi_epi16(temp5, temp7);

  // Step 3: Transpose within 8x8 block
  __m256i final0 = _mm256_unpacklo_epi32(interleave0, interleave4);
  __m256i final1 = _mm256_unpackhi_epi32(interleave0, interleave4);
  __m256i final2 = _mm256_unpacklo_epi32(interleave1, interleave5);
  __m256i final3 = _mm256_unpackhi_epi32(interleave1, interleave5);
  __m256i final4 = _mm256_unpacklo_epi32(interleave2, interleave6);
  __m256i final5 = _mm256_unpackhi_epi32(interleave2, interleave6);
  __m256i final6 = _mm256_unpacklo_epi32(interleave3, interleave7);
  __m256i final7 = _mm256_unpackhi_epi32(interleave3, interleave7);

  // Step 4: Transpose within 16x16 block
  __m256i res[8];
  res[0] = _mm256_unpacklo_epi64(final0, final4);
  res[1] = _mm256_unpackhi_epi64(final0, final4);
  res[2] = _mm256_unpacklo_epi64(final1, final5);
  res[3] = _mm256_unpackhi_epi64(final1, final5);
  res[4] = _mm256_unpacklo_epi64(final2, final6);
  res[5] = _mm256_unpackhi_epi64(final2, final6);
  res[6] = _mm256_unpacklo_epi64(final3, final7);
  res[7] = _mm256_unpackhi_epi64(final3, final7);

  const int perm_0213 = 0xd8; // 0, 2, 1, 3
  const int perm_02 = 0x20;   // 0, 2
  const int perm_13 = 0x31;   // 1, 3
  for (int i = 0; i < 4; i++) {
    __m256i a128x2 = _mm256_permute4x64_epi64(res[2 * i], perm_0213);
    __m256i b128x2 = _mm256_permute4x64_epi64(res[2 * i + 1], perm_0213);
    _mm256_storeu_si256((__m256i *)&dst[2 * i * output_stride],
                        _mm256_permute2x128_si256(a128x2, b128x2, perm_02));
    _mm256_storeu_si256((__m256i *)&dst[(8 + 2 * i) * output_stride],
                        _mm256_permute2x128_si256(a128x2, b128x2, perm_13));
  }
}

static inline void create_q4_0_weights_x8(const uint8_t *int4_weight,
                                          __m256i *q4_blocks) {
  constexpr const size_t ROW_BLOCK_BYTE_SIZE = 16;

  // Create masks for extracting low and high nibbles
  const __m256i low_nibble_mask = _mm256_set1_epi8(0x0F);
  const __m256i high_nibble_mask = _mm256_set1_epi8(0xF0);

  // Create two blocks in one iteration
  for (int i = 0; i < 4; ++i) {
    // Load 16 bytes of input data
    __m256i input = _mm256_loadu_si256(
      (const __m256i *)(int4_weight + 2 * ROW_BLOCK_BYTE_SIZE * i));

    // A = input & low_nibble_mask
    __m256i A = _mm256_and_si256(input, low_nibble_mask);

    // B = (input & high_nibble_mask) >> 4
    __m256i B = _mm256_srli_epi16(_mm256_and_si256(input, high_nibble_mask), 4);

    // input_shifted = input >> 8 bytes
    __m256i input_shifted = _mm256_bsrli_epi128(input, 8);
    // C = input_shifted & low_nibble_mask
    __m256i C = _mm256_and_si256(input_shifted, low_nibble_mask);

    // D = (input_shifted & high_nibble_mask) >> 4
    __m256i D =
      _mm256_srli_epi16(_mm256_and_si256(input_shifted, high_nibble_mask), 4);

    // AC = A | (C << 4)
    __m256i AC = _mm256_or_si256(A, _mm256_slli_epi16(C, 4));

    // BD = B | (D << 4)
    __m256i BD = _mm256_or_si256(B, _mm256_slli_epi16(D, 4));

    // Interleave AC and BD
    __m256i result = _mm256_unpacklo_epi8(AC, BD);

    _mm256_store_si256(&q4_blocks[i], result);
  }
}

inline static void nntr_make_block_q4_0x8(const __m256i *in, block_q4_0x8 *out,
                                          const uint16_t *scales) {
  constexpr size_t IN_CNT = 8;
  memcpy(out->d, scales, IN_CNT * sizeof(uint16_t));

  const int perm_0213 = 0xd8; // 0, 2, 1, 3
  const int perm_02 = 0x20;   // 0, 2
  const int perm_13 = 0x31;   // 1, 3
  __m256i a128x2 = _mm256_permute4x64_epi64(*(__m256i *)&in[0], perm_0213);
  __m256i b128x2 = _mm256_permute4x64_epi64(*(__m256i *)&in[1], perm_0213);
  __m256i c128x2 = _mm256_permute4x64_epi64(*(__m256i *)&in[2], perm_0213);
  __m256i d128x2 = _mm256_permute4x64_epi64(*(__m256i *)&in[3], perm_0213);
  _mm256_storeu_si256((__m256i *)&out->qs[0],
                      _mm256_permute2x128_si256(a128x2, b128x2, perm_02));
  _mm256_storeu_si256((__m256i *)&out->qs[32],
                      _mm256_permute2x128_si256(c128x2, d128x2, perm_02));
  _mm256_storeu_si256((__m256i *)&out->qs[64],
                      _mm256_permute2x128_si256(a128x2, b128x2, perm_13));
  _mm256_storeu_si256((__m256i *)&out->qs[96],
                      _mm256_permute2x128_si256(c128x2, d128x2, perm_13));
}

void transform_int4_osv32_isv2_to_q4_0x8(size_t N, size_t K,
                                         const uint8_t *osv32_weights,
                                         const uint16_t *osv32_scales,
                                         size_t scale_group_size,
                                         void *dst_q4_0x) {

  NNTR_THROW_IF((!(scale_group_size == 32 || scale_group_size == 64 ||
                   scale_group_size == 128)),
                std::invalid_argument)
    << "Scale group size must be 32/64/128";
  NNTR_THROW_IF(K % QK4_0 != 0, std::invalid_argument)
    << "K size must be divisable by QK4_0 (32)";
  NNTR_THROW_IF(N % 8 != 0, std::invalid_argument)
    << "N size must be divisable by 8";

  static constexpr const size_t NUM_Q4_0_BLOCKS = 8;
  static constexpr const size_t ROW_BLOCK_SIZE = 32;
  static constexpr const size_t COLUMN_BLOCK_SIZE = 2;
  static constexpr const size_t ROW_BLOCK_BYTE_SIZE = 16;

  static constexpr const size_t dst_tmp_size =
    (8 * ROW_BLOCK_BYTE_SIZE) / sizeof(__m256i);
  uint8_t *dst_ = reinterpret_cast<uint8_t *>(dst_q4_0x);

  // --- Layout ---
  const size_t rows_count_pad = align(N, ROW_BLOCK_SIZE);
  const size_t columns_count_pad = align(K, ROW_BLOCK_SIZE);
  const size_t column_blocks_count =
    columns_count_pad / COLUMN_BLOCK_SIZE; // COLUMN_BLOCK_SIZE == 2
  const size_t bytes_per_row_block_span = column_blocks_count * ROW_BLOCK_SIZE;
  const int column_blocks_cnt = K / QK4_0;

  const size_t row_iters = (N + 15) / 16;
  auto &tm = nntrainer::ThreadManager::Global();
  tm.parallel_for(0, row_iters, [&](size_t iter) {
    alignas(32) __m256i dst_tmp_local[dst_tmp_size];
    alignas(32) uint8_t mx16x16_local[16 * 16];
    int row_id = static_cast<int>(iter) * 16;
    const size_t row_in_block_id = row_id / ROW_BLOCK_SIZE;
    size_t i_in_block = row_id % ROW_BLOCK_SIZE;
    for (int column_out_block_id = 0; column_out_block_id < column_blocks_cnt;
         column_out_block_id++) {
      int column_idx = column_out_block_id * QK4_0;
      int scale_offset = (column_idx / scale_group_size) * rows_count_pad;
      const size_t row_block_base =
        row_in_block_id * bytes_per_row_block_span + i_in_block;
      int src_offset =
        row_block_base + column_out_block_id * 16 * ROW_BLOCK_SIZE;
      transpose_matrix_16x16(&osv32_weights[src_offset], ROW_BLOCK_SIZE,
                             mx16x16_local, 16);
      int max_r = std::min((size_t)16, N - row_id);
      size_t row_out_block_id = row_id / NUM_Q4_0_BLOCKS;
      int dst_offset =
        (NUM_Q4_0_BLOCKS * sizeof(block_q4_0)) *
        (column_out_block_id + row_out_block_id * column_blocks_cnt);
      for (int r = 0; r < max_r; r += NUM_Q4_0_BLOCKS) {
        create_q4_0_weights_x8(&mx16x16_local[16 * r], dst_tmp_local);

        nntr_make_block_q4_0x8(dst_tmp_local,
                               (block_q4_0x8 *)(dst_ + dst_offset),
                               &osv32_scales[scale_offset + row_id + r]);
        row_out_block_id++;
        dst_offset +=
          (NUM_Q4_0_BLOCKS * sizeof(block_q4_0)) * column_blocks_cnt;
      }
    }
  });
}

// ---------------------------------------------------------------------------
// causal_depthwise_conv1d_k3 - fp32 prefill.
// Computes y_t = w0*x_t + w1*x_{t-1} + w2*x_{t-2} (+ bias) over H.
// TILE=32 (4 x AVX2 vectors) unrolled inner loop.
// ---------------------------------------------------------------------------
void causal_depthwise_conv1d_k3(const float *input, const float *packed_weight,
                                const float *bias, float *output,
                                unsigned int B, unsigned int H,
                                unsigned int W) {
  const float *w0 = packed_weight;
  const float *w1 = packed_weight + W;
  const float *w2 = packed_weight + 2 * W;

  constexpr unsigned int VEC = 8;
  constexpr unsigned int TILE = 32; // 4 x VEC

  for (unsigned int b = 0; b < B; ++b) {
    const float *x_base = input + static_cast<size_t>(b) * H * W;
    float *y_base = output + static_cast<size_t>(b) * H * W;

    unsigned int c = 0;

    // ---- 4-wide unrolled tile (TILE = 32) --------------------------------
    for (; c + TILE <= W; c += TILE) {
      const __m256 vw0_0 = _mm256_loadu_ps(w0 + c + 0);
      const __m256 vw0_1 = _mm256_loadu_ps(w0 + c + 8);
      const __m256 vw0_2 = _mm256_loadu_ps(w0 + c + 16);
      const __m256 vw0_3 = _mm256_loadu_ps(w0 + c + 24);

      const __m256 vw1_0 = _mm256_loadu_ps(w1 + c + 0);
      const __m256 vw1_1 = _mm256_loadu_ps(w1 + c + 8);
      const __m256 vw1_2 = _mm256_loadu_ps(w1 + c + 16);
      const __m256 vw1_3 = _mm256_loadu_ps(w1 + c + 24);

      const __m256 vw2_0 = _mm256_loadu_ps(w2 + c + 0);
      const __m256 vw2_1 = _mm256_loadu_ps(w2 + c + 8);
      const __m256 vw2_2 = _mm256_loadu_ps(w2 + c + 16);
      const __m256 vw2_3 = _mm256_loadu_ps(w2 + c + 24);

      __m256 prev1_0 = _mm256_setzero_ps(), prev1_1 = _mm256_setzero_ps();
      __m256 prev1_2 = _mm256_setzero_ps(), prev1_3 = _mm256_setzero_ps();
      __m256 prev2_0 = _mm256_setzero_ps(), prev2_1 = _mm256_setzero_ps();
      __m256 prev2_2 = _mm256_setzero_ps(), prev2_3 = _mm256_setzero_ps();

      for (unsigned int t = 0; t < H; ++t) {
        const float *x_ptr = x_base + static_cast<size_t>(t) * W + c;
        float *y_ptr = y_base + static_cast<size_t>(t) * W + c;

        const __m256 cur0 = _mm256_loadu_ps(x_ptr + 0);
        const __m256 cur1 = _mm256_loadu_ps(x_ptr + 8);
        const __m256 cur2 = _mm256_loadu_ps(x_ptr + 16);
        const __m256 cur3 = _mm256_loadu_ps(x_ptr + 24);

        __m256 acc0 = _mm256_mul_ps(cur0, vw0_0);
        __m256 acc1 = _mm256_mul_ps(cur1, vw0_1);
        __m256 acc2 = _mm256_mul_ps(cur2, vw0_2);
        __m256 acc3 = _mm256_mul_ps(cur3, vw0_3);

#if defined(__FMA__)
        acc0 = _mm256_fmadd_ps(prev1_0, vw1_0, acc0);
        acc1 = _mm256_fmadd_ps(prev1_1, vw1_1, acc1);
        acc2 = _mm256_fmadd_ps(prev1_2, vw1_2, acc2);
        acc3 = _mm256_fmadd_ps(prev1_3, vw1_3, acc3);

        acc0 = _mm256_fmadd_ps(prev2_0, vw2_0, acc0);
        acc1 = _mm256_fmadd_ps(prev2_1, vw2_1, acc1);
        acc2 = _mm256_fmadd_ps(prev2_2, vw2_2, acc2);
        acc3 = _mm256_fmadd_ps(prev2_3, vw2_3, acc3);
#else
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(prev1_0, vw1_0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(prev1_1, vw1_1));
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(prev1_2, vw1_2));
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(prev1_3, vw1_3));

        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(prev2_0, vw2_0));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(prev2_1, vw2_1));
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(prev2_2, vw2_2));
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(prev2_3, vw2_3));
#endif
        if (bias) {
          acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(bias + c + 0));
          acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(bias + c + 8));
          acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(bias + c + 16));
          acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(bias + c + 24));
        }

        _mm256_storeu_ps(y_ptr + 0, acc0);
        _mm256_storeu_ps(y_ptr + 8, acc1);
        _mm256_storeu_ps(y_ptr + 16, acc2);
        _mm256_storeu_ps(y_ptr + 24, acc3);

        prev2_0 = prev1_0;
        prev1_0 = cur0;
        prev2_1 = prev1_1;
        prev1_1 = cur1;
        prev2_2 = prev1_2;
        prev1_2 = cur2;
        prev2_3 = prev1_3;
        prev1_3 = cur3;
      }
    }

    // ---- VEC=8 tail -------------------------------------------------------
    for (; c + VEC <= W; c += VEC) {
      const __m256 vw0v = _mm256_loadu_ps(w0 + c);
      const __m256 vw1v = _mm256_loadu_ps(w1 + c);
      const __m256 vw2v = _mm256_loadu_ps(w2 + c);
      __m256 prev1v = _mm256_setzero_ps();
      __m256 prev2v = _mm256_setzero_ps();

      for (unsigned int t = 0; t < H; ++t) {
        const float *x_ptr = x_base + static_cast<size_t>(t) * W + c;
        float *y_ptr = y_base + static_cast<size_t>(t) * W + c;

        const __m256 cur = _mm256_loadu_ps(x_ptr);
        __m256 acc = _mm256_mul_ps(cur, vw0v);
#if defined(__FMA__)
        acc = _mm256_fmadd_ps(prev1v, vw1v, acc);
        acc = _mm256_fmadd_ps(prev2v, vw2v, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(prev1v, vw1v));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(prev2v, vw2v));
#endif
        if (bias)
          acc = _mm256_add_ps(acc, _mm256_loadu_ps(bias + c));
        _mm256_storeu_ps(y_ptr, acc);
        prev2v = prev1v;
        prev1v = cur;
      }
    }

    // ---- scalar tail -------------------------------------------------------
    for (; c < W; ++c) {
      float prev2 = 0.0f, prev1 = 0.0f;
      for (unsigned int t = 0; t < H; ++t) {
        float cur = x_base[static_cast<size_t>(t) * W + c];
        float acc = w0[c] * cur + w1[c] * prev1 + w2[c] * prev2;
        if (bias)
          acc += bias[c];
        y_base[static_cast<size_t>(t) * W + c] = acc;
        prev2 = prev1;
        prev1 = cur;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// causal_depthwise_conv1d_k3_decode - fp32 single-token decode.
// Reads persistent state [s0=x_{t-2} | s1=x_{t-1}] (2*W floats),
// computes y = w0*x + w1*s1 + w2*s0, then updates state in-place.
// TILE=32 (4 x AVX2 vectors) unrolled, matching the prefill kernel style.
// ---------------------------------------------------------------------------
void causal_depthwise_conv1d_k3_decode(const float *x_cur,
                                       const float *packed_weight, float *state,
                                       float *y_cur, unsigned int W) {
  const float *w0 = packed_weight;
  const float *w1 = packed_weight + W;
  const float *w2 = packed_weight + 2 * W;
  const float *s0 = state;     // x_{t-2}
  const float *s1 = state + W; // x_{t-1}

  constexpr unsigned int VEC = 8;
  constexpr unsigned int TILE = 32;

  unsigned int c = 0;

  // ---- 4-wide unrolled tile (TILE = 32) ------------------------------------
  for (; c + TILE <= W; c += TILE) {
    const __m256 vw0_0 = _mm256_loadu_ps(w0 + c + 0);
    const __m256 vw0_1 = _mm256_loadu_ps(w0 + c + 8);
    const __m256 vw0_2 = _mm256_loadu_ps(w0 + c + 16);
    const __m256 vw0_3 = _mm256_loadu_ps(w0 + c + 24);

    const __m256 vw1_0 = _mm256_loadu_ps(w1 + c + 0);
    const __m256 vw1_1 = _mm256_loadu_ps(w1 + c + 8);
    const __m256 vw1_2 = _mm256_loadu_ps(w1 + c + 16);
    const __m256 vw1_3 = _mm256_loadu_ps(w1 + c + 24);

    const __m256 vw2_0 = _mm256_loadu_ps(w2 + c + 0);
    const __m256 vw2_1 = _mm256_loadu_ps(w2 + c + 8);
    const __m256 vw2_2 = _mm256_loadu_ps(w2 + c + 16);
    const __m256 vw2_3 = _mm256_loadu_ps(w2 + c + 24);

    const __m256 vx0 = _mm256_loadu_ps(x_cur + c + 0);
    const __m256 vx1 = _mm256_loadu_ps(x_cur + c + 8);
    const __m256 vx2 = _mm256_loadu_ps(x_cur + c + 16);
    const __m256 vx3 = _mm256_loadu_ps(x_cur + c + 24);

    const __m256 vs1_0 = _mm256_loadu_ps(s1 + c + 0);
    const __m256 vs1_1 = _mm256_loadu_ps(s1 + c + 8);
    const __m256 vs1_2 = _mm256_loadu_ps(s1 + c + 16);
    const __m256 vs1_3 = _mm256_loadu_ps(s1 + c + 24);

    const __m256 vs0_0 = _mm256_loadu_ps(s0 + c + 0);
    const __m256 vs0_1 = _mm256_loadu_ps(s0 + c + 8);
    const __m256 vs0_2 = _mm256_loadu_ps(s0 + c + 16);
    const __m256 vs0_3 = _mm256_loadu_ps(s0 + c + 24);

    __m256 acc0 = _mm256_mul_ps(vw0_0, vx0);
    __m256 acc1 = _mm256_mul_ps(vw0_1, vx1);
    __m256 acc2 = _mm256_mul_ps(vw0_2, vx2);
    __m256 acc3 = _mm256_mul_ps(vw0_3, vx3);

#if defined(__FMA__)
    acc0 = _mm256_fmadd_ps(vw1_0, vs1_0, acc0);
    acc1 = _mm256_fmadd_ps(vw1_1, vs1_1, acc1);
    acc2 = _mm256_fmadd_ps(vw1_2, vs1_2, acc2);
    acc3 = _mm256_fmadd_ps(vw1_3, vs1_3, acc3);

    acc0 = _mm256_fmadd_ps(vw2_0, vs0_0, acc0);
    acc1 = _mm256_fmadd_ps(vw2_1, vs0_1, acc1);
    acc2 = _mm256_fmadd_ps(vw2_2, vs0_2, acc2);
    acc3 = _mm256_fmadd_ps(vw2_3, vs0_3, acc3);
#else
    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(vw1_0, vs1_0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(vw1_1, vs1_1));
    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(vw1_2, vs1_2));
    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(vw1_3, vs1_3));

    acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(vw2_0, vs0_0));
    acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(vw2_1, vs0_1));
    acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(vw2_2, vs0_2));
    acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(vw2_3, vs0_3));
#endif
    _mm256_storeu_ps(y_cur + c + 0, acc0);
    _mm256_storeu_ps(y_cur + c + 8, acc1);
    _mm256_storeu_ps(y_cur + c + 16, acc2);
    _mm256_storeu_ps(y_cur + c + 24, acc3);

    // state update: s0 <- s1, s1 <- x_cur
    _mm256_storeu_ps(state + c + 0, vs1_0);
    _mm256_storeu_ps(state + c + 8, vs1_1);
    _mm256_storeu_ps(state + c + 16, vs1_2);
    _mm256_storeu_ps(state + c + 24, vs1_3);

    _mm256_storeu_ps(state + W + c + 0, vx0);
    _mm256_storeu_ps(state + W + c + 8, vx1);
    _mm256_storeu_ps(state + W + c + 16, vx2);
    _mm256_storeu_ps(state + W + c + 24, vx3);
  }

  // ---- VEC=8 tail ----------------------------------------------------------
  for (; c + VEC <= W; c += VEC) {
    const __m256 vw0v = _mm256_loadu_ps(w0 + c);
    const __m256 vw1v = _mm256_loadu_ps(w1 + c);
    const __m256 vw2v = _mm256_loadu_ps(w2 + c);
    const __m256 vxv = _mm256_loadu_ps(x_cur + c);
    const __m256 vs1v = _mm256_loadu_ps(s1 + c);
    const __m256 vs0v = _mm256_loadu_ps(s0 + c);

    __m256 acc = _mm256_mul_ps(vw0v, vxv);
#if defined(__FMA__)
    acc = _mm256_fmadd_ps(vw1v, vs1v, acc);
    acc = _mm256_fmadd_ps(vw2v, vs0v, acc);
#else
    acc = _mm256_add_ps(acc, _mm256_mul_ps(vw1v, vs1v));
    acc = _mm256_add_ps(acc, _mm256_mul_ps(vw2v, vs0v));
#endif
    _mm256_storeu_ps(y_cur + c, acc);

    // state update
    _mm256_storeu_ps(state + c, vs1v);
    _mm256_storeu_ps(state + W + c, vxv);
  }

  // ---- scalar tail ---------------------------------------------------------
  for (; c < W; ++c) {
    y_cur[c] = w0[c] * x_cur[c] + w1[c] * s1[c] + w2[c] * s0[c];
    state[c] = s1[c];        // s0 <- s1
    state[W + c] = x_cur[c]; // s1 <- x_cur
  }
}

} // namespace nntrainer::avx2

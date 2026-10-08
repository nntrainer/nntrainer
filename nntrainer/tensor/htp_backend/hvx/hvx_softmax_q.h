// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_softmax_q.h
 * @date   06 Oct 2026
 * @brief  Integer softmax over int16 log2-domain scores -> uint8 P'
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Stage 2 of the row-blocked attention kernel (23_attention_v2_plan.md).
 * One 64-row block of query rows, every column tile it can see, in the
 * HMX activation tile layout: element (r, j) of tile c is at
 * tiles[c * 2048 + r * 32 + j] -- the same positions the convert unit
 * wrote the scores to and the same positions P.V will read P' from, so
 * the whole routine is elementwise except the row max and the row sum.
 *
 * Numbers. The score tile holds s = floor(acc * 2^k / 512) as int16, the
 * two convert planes zipped. With rho = alpha * 2^F * 512 / 2^k in
 * (0.5, 1], where alpha = s_q * s_k * log2(e) turns the accumulator into a
 * log2-domain logit, the logit in Q(15-F).F is u = s * rho. Then, with
 * every multiply the HVX vmpy(Vh,Vh):<<1:rnd:sat, i.e. Q15 x Q15 -> Q15
 * rounded and saturated,
 *
 *   t  = s - max_row(s)                     <= 0, saturating
 *   u  = t * rho_q15                        one vmpy
 *   e  = u >> F (floor), f = u & (2^F - 1)   integer and fraction of log2
 *   x  = f << (15 - F)                      the fraction as Q15
 *   m  = x * (A + x * (B + x * C))          2^x - 1 as Q15, cubic, error
 *                                           <= 5 Q15 units (1.5e-4)
 *   T  = 16384 + (m >> 1)                   2^x as Q14, in [16384, 32767]
 *   q  = T >> min(5 - e, 15)                per-lane arithmetic shift
 *   P' = sat_u8((q + 1) >> 1)               vasr(Vh,Vh,#1):rnd:sat
 *
 * so P' = round(2^(u / 2^F) * 256): T >> (5 - e) is 2^(u/2^F) * 512, the
 * final rounding shift halves it, the saturation turns the 256 at the row
 * maximum into 255, and the clamp of the shift at 15 makes everything
 * below 2^-9 (a shift of 17 or more) zero, which is what rounding gives.
 * Truncating instead of rounding measured 8.5 dB worse on P, because so
 * many entries sit in the lowest steps of the grid. On its 8-bit grid this
 * is as accurate as a float a8 P with the same fixed 1/256 scale: the host
 * test measures both and they agree within 0.1 dB. The row sum of P'
 * (int32) carries the normalization to the output stage.
 *
 * The zero-point term of Q.K^T, zq * colsum_k[column], is a per-column
 * constant in the same int16 units; corr[c * 32 + j] is subtracted from
 * the raw score before anything else. Because the convert truncates to 16
 * bits, raw score and correction are both modulo 2^16 and the difference
 * is exact as long as the true score fits int16.
 *
 * Masks are positions, not data: a column is masked for a row when it is
 * past the row (causal), more than window - 1 behind it, past n_cols, or
 * the row itself is padding (r >= n_rows). Masked entries get P' = 0 and
 * take no part in the max or the sum.
 *
 * hvx_softmax_q_ref is the scalar definition, compiled everywhere;
 * hvx_softmax_q (Hexagon only) matches it bit for bit.
 */

#ifndef __NNTRAINER_HVX_SOFTMAX_Q_H__
#define __NNTRAINER_HVX_SOFTMAX_Q_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Rows in a block and elements in a tile. */
#define HVX_SOFTMAX_Q_ROWS 64u
#define HVX_SOFTMAX_Q_TILE 2048u

/** @brief Q15 coefficients of 2^x - 1 ~ x (A + x (B + x C)) on [0, 1). */
#define HVX_SOFTMAX_Q_EXP2_A 22784
#define HVX_SOFTMAX_Q_EXP2_B 7440
#define HVX_SOFTMAX_Q_EXP2_C 2539

/**
 * @brief int32 words of the row-sum output: 16 vectors of 32 lanes, row r
 *        in word (r / 4) * 32 + (r % 4) * 8. That is where a vrmpy over a
 *        128-byte P' vector (4 rows x 32 columns) leaves each row's sum
 *        after an in-vector reduction, and the output stage reads it as a
 *        vector anyway. The other lanes are unspecified.
 */
#define HVX_SOFTMAX_Q_ROWSUM_WORDS (16u * 32u)

static inline uint32_t hvx_softmax_q_rowsum_index(uint32_t r) {
  return (r >> 2) * 32u + (r & 3u) * 8u;
}

/** @brief Bytes of VTCM scratch hvx_softmax_q needs: the 32 row-max
 *         vectors, the 16 row-sum accumulators and up to 128 staged
 *         per-tile correction vectors. */
#define HVX_SOFTMAX_Q_MAX_TILES 128u
#define HVX_SOFTMAX_Q_SCRATCH_BYTES ((48u + HVX_SOFTMAX_Q_MAX_TILES) * 128u)

/** @brief Geometry and scale of one block. */
typedef struct {
  uint32_t n_col_tiles; /**< tiles in the score / P' arrays */
  uint32_t col0;        /**< absolute cache column of tile 0, element 0 */
  uint32_t n_cols;      /**< valid columns from col0; the rest are masked */
  uint32_t row0;        /**< absolute position of row 0 */
  uint32_t n_rows;      /**< valid rows (<= 64); rows past it are padding */
  uint32_t window;      /**< sliding window, 0 = unlimited */
  uint16_t rho_q15;     /**< rho in (0.5, 1] times 32768, at most 32767 */
  uint8_t frac_bits;    /**< F: fraction bits of the log2 logit, 1..8 */
} hvx_softmax_q_block;

/**
 * @brief Chooses the convert scale 2^k and the residual rho for a layer
 *        whose accumulator-to-log2 factor is @a alpha = s_q*s_k*log2(e).
 *
 * @param[out] k        convert scale exponent: scale = 2^k (fp16 bits via
 *                      hexkl_cvt_f32_to_hf(ldexpf(1, k)))
 * @param[out] rho_q15  residual in (0.5, 1] as Q15, clamped to 32767
 * @return 0, or -1 when 2^k is outside fp16's normal range
 */
int hvx_softmax_q_scale(float alpha, uint32_t frac_bits, int *k,
                        uint16_t *rho_q15);

/** @brief 2^(f / 2^F) as Q14 by the fixed-point cubic, f in [0, 2^F). */
uint16_t hvx_softmax_q_exp2_q14(uint32_t f, uint32_t frac_bits);

/**
 * @brief Scalar definition of the block softmax.
 *
 * @param s_tiles  [n_col_tiles][64][32] int16 scores as the convert
 *                 planes zipped (raw, before the zero-point correction)
 * @param corr     [n_col_tiles][32] per-column correction, or NULL
 * @param p_tiles  [n_col_tiles][64][32] uint8 P' out
 * @param rowsum   HVX_SOFTMAX_Q_ROWSUM_WORDS int32 out, see
 *                 hvx_softmax_q_rowsum_index
 */
void hvx_softmax_q_ref(const hvx_softmax_q_block *b, const int16_t *s_tiles,
                       const int16_t *corr, uint8_t *p_tiles, int32_t *rowsum);

#if defined(__hexagon__)
/** @brief Micro-benchmark of the HVX primitives the softmax uses: cycles
 *         per vadd, per Q15 vmpy, per exp2 chain, per VTCM vector load,
 *         over @a n iterations. @a vtcm is 1 KiB of VTCM. */
void hvx_softmax_q_rate(uint32_t n, void *vtcm, void *vtcm_big,
                        uint32_t out[12]);

/**
 * @brief The HVX implementation. All buffers in VTCM, 128-byte aligned;
 *        @a scratch is HVX_SOFTMAX_Q_SCRATCH_BYTES. Bit-exact with
 *        hvx_softmax_q_ref.
 */
void hvx_softmax_q(const hvx_softmax_q_block *b, const int16_t *s_tiles,
                   const int16_t *corr, uint8_t *p_tiles, int32_t *rowsum,
                   void *scratch);
#endif

#ifdef __cplusplus
}
#endif

#endif /* __NNTRAINER_HVX_SOFTMAX_Q_H__ */

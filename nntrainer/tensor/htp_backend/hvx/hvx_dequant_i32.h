// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_dequant_i32.h
 * @date   03 Aug 2026
 * @brief  int32 accumulator to f32 dequantization for the A8W4 path
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_DEQUANT_I32_H__
#define __NNTRAINER_HVX_DEQUANT_I32_H__

#include <stdint.h>

#include "hvx_worker_pool.h"

/**
 * @brief Turns HMX int32 accumulators back into f32 (K3).
 *
 * out[m][n] = (acc[m][n] - act_zp[m]*colsum_w[n]) * act_scale[m]
 *             * w_scale[n] + bias[n]
 *
 * The zp*colsum term corrects for HMX taking unsigned activations: with
 * x = s*(u - zp), the dot product expands to s*d*(sum u*q_w - zp*sum q_w),
 * and sum q_w over the reduction depends only on the weights, so it is a
 * precomputed table rather than runtime work.
 *
 * HexKL micro exposes no bias, scale or zero-point registers, which is why
 * this pass exists at all.
 *
 * @param[in]  acc      m_pad by n int32, row-major
 * @param[in]  m_valid  rows to emit; padded rows are skipped because their
 *                      quantization parameters are synthetic
 * @param[in]  colsum_w per-channel sum of the int4 weights, n entries
 * @param[in]  w_scale  per-channel dequantization multiplier, n entries
 * @param[out] out      m_valid by n f32, row-major
 */
void hvx_dequant_i32_to_f32(const int32_t *acc, uint32_t m_valid,
                            uint32_t m_pad, uint32_t n, const float *act_scale,
                            const int32_t *act_zp, const int32_t *colsum_w,
                            const float *w_scale, const float *bias, float *out,
                            int accumulate);

/**
 * @brief Same dequantization, applied to ONE 64x32 accumulator tile still
 *        sitting in VTCM.
 *
 * The formula, the intrinsics and their order are hvx_dequant_i32_to_f32's,
 * element for element -- this is where the tile comes from and where the
 * result goes that differ, never the arithmetic. Results are therefore
 * bitwise identical to dequantizing the same tile after a round trip through
 * a DDR scratch, which is what makes the existing bit-exactness gates a valid
 * check on this path (hexkl_acc_tile.h explains why the round trip existed).
 *
 * A tile column count of 32 is exactly one HVX f32 vector, so each row is a
 * single load, a single store and no tail loop.
 *
 * @param[in]  tile       first element of the tile's row 0, i.e. the VTCM
 *                        result tile advanced by hexkl_acc_layout::base
 * @param[in]  row_stride int32 elements between tile rows (from the probe)
 * @param[in]  m_count    tile rows carrying real data; padded rows are
 *                        skipped, same rule as m_valid above
 * @param[in]  act_scale  m_count entries, already offset to this row block
 * @param[in]  act_zp     m_count entries, already offset to this row block
 * @param[in]  colsum_w   32 entries, already offset to this column tile
 * @param[in]  w_scale    32 entries, already offset to this column tile
 * @param[in]  bias       32 entries, already offset to this column tile
 * @param[out] out        first output element of this tile
 * @param[in]  out_stride f32 elements between output rows (the full N)
 * @param[in]  accumulate nonzero adds into @a out instead of storing; one
 *                        add per element per call either way, so this is
 *                        bitwise the staged add it replaces
 */
void hvx_dequant_acc_tile_to_f32(const int32_t *tile, uint32_t row_stride,
                                 uint32_t m_count, const float *act_scale,
                                 const int32_t *act_zp, const int32_t *colsum_w,
                                 const float *w_scale, const float *bias,
                                 float *out, uint32_t out_stride,
                                 int accumulate);

/**
 * @brief Dequantizes a RUN of accumulator tiles in one pooled pass.
 *
 * Same arithmetic as hvx_dequant_acc_tile_to_f32, tile for tile -- this
 * calls it -- so the bytes are identical to dequantizing the same tiles one
 * at a time. What changes is who runs it and how often: the per-tile form
 * is invoked once per n-tile from the matmul loop, on the calling thread
 * only, 176 times per 64-row block. That leaves the pool's other workers
 * idle through the whole epilogue and pays a call's worth of setup for
 * 2048 elements of work. Here the caller stages several tiles in VTCM
 * first and hands them over together, so the split is across tiles and the
 * pool forks twice a block instead of 176 times.
 *
 * Tiles are consecutive n-tiles starting at @a nt0, each
 * HEXKL_ACC_TILE_COLS wide. A tile whose first column is below @a split
 * lands in @a dst_a at that column; at or above it, in @a dst_b at
 * (column - split). That is the gate/up halves of a gate_up result; pass
 * @a split >= the total width and @a dst_b NULL when there is only one
 * destination, as the down matmul has.
 *
 * @param tiles_base  first staged tile, already advanced by
 *                    hexkl_acc_layout::base
 * @param tile_stride bytes between consecutive staged tiles
 * @param n_tiles     how many are staged
 * @param nt0         n-tile index of the first staged tile
 * @param dst_stride  f32 elements between consecutive rows of dst_a/dst_b
 */
void hvx_dequant_acc_tiles_to_f32(const uint8_t *tiles_base,
                                  uint32_t tile_stride, uint32_t n_tiles,
                                  uint32_t nt0, uint32_t row_stride,
                                  uint32_t m_count, const float *act_scale,
                                  const int32_t *act_zp,
                                  const int32_t *colsum_w,
                                  const float *w_scale, const float *bias,
                                  float *dst_a, float *dst_b, uint32_t split,
                                  uint32_t dst_stride, hvx_worker_pool *pool);

/**
 * @brief The batch dequant as a job the caller owns, so it can run on the
 *        pool's workers (hvx_worker_pool_submit) while the caller issues
 *        the next batch's HMX tiles. Fields are
 *        hvx_dequant_acc_tiles_to_f32's arguments; that function is this
 *        job run synchronously. Units for the submit: n_tiles.
 */
typedef struct {
  const uint8_t *tiles_base;
  uint32_t tile_stride;
  uint32_t nt0;
  uint32_t row_stride;
  uint32_t m_count;
  const float *act_scale;
  const int32_t *act_zp;
  const int32_t *colsum_w;
  const float *w_scale;
  const float *bias;
  float *dst_a;
  float *dst_b;
  uint32_t split;
  uint32_t dst_stride;
  uint32_t n_tiles;
} hvx_dq_tiles_job;

/** @brief hvx_worker_pool_func over an hvx_dq_tiles_job. */
void hvx_dq_tiles_worker(uint32_t n_threads, uint32_t i, void *job);

/** @brief Same, for the fused gate/up dequant + SwiGLU below. Units for
 *         the submit: n_pairs. */
typedef struct {
  const uint8_t *tiles_base;
  uint32_t tile_stride;
  uint32_t n_pairs;
  uint32_t g0;
  uint32_t row_stride;
  uint32_t m_count;
  const float *act_scale;
  const int32_t *act_zp;
  const int32_t *colsum_w;
  const float *w_scale;
  const float *bias;
  uint32_t inter;
  float *dst;
  uint32_t dst_stride;
} hvx_dq_swiglu_job;

/** @brief hvx_worker_pool_func over an hvx_dq_swiglu_job. */
void hvx_dq_swiglu_worker(uint32_t n_threads, uint32_t i, void *job);

/**
 * @brief Dequantizes gate/up tile PAIRS and applies SwiGLU before anything
 *        is stored: dst = silu(gate) * up, one vector at a time.
 *
 * The staged run holds @a n_pairs gate tiles (columns g0.. of the gate
 * half) followed by the @a n_pairs up tiles opposite them (the same
 * columns, @a inter further along), which is how the matmul loop issues
 * them. Each row's gate and up vectors are dequantized with exactly
 * hvx_dequant_acc_tile_to_f32's operations and then handed to
 * hvx_swiglu_det_sf -- the same function hvx_swiglu_inplace_f32 applies to
 * the same two vectors after they have been through VTCM -- so the bytes
 * are identical to the dequant-store-swiglu sequence this replaces, and
 * the ARM parity gate on swiglu_det.h still holds. What changes is that
 * [64 x inter] f32 of gate and of up are never written or read back:
 * measured, that separate SwiGLU pass was 2.05 ms of a 22.6 ms prefill
 * call (doc 47).
 *
 * @param g0          first gate n-tile in the run (its column is g0 * 32)
 * @param colsum_w, w_scale, bias  the FULL gate_up tables (2 * inter);
 *                    the up half is found at inter + column
 * @param dst         the [rows x inter] SwiGLU output; column g0 * 32 of it
 *                    is where this run's first pair lands
 */
/**
 * @brief Fused dequant + elementwise product over tile PAIRS from two
 *        weights that share the activation: staged slot j is the first
 *        weight's tile at column c0 + 32 j, slot n_pairs + j the second's
 *        at the same column, and dst gets dq(first) * dq(second).
 *
 * The conv block's pre-conv gate (doc 51 section 2): in_proj's a and c
 * column slices are two registered weights, and a * c is the only thing
 * the block needs of them, so neither is ever stored. Same shape of job
 * as hvx_dq_swiglu_job, with the second weight's constants separate
 * because they come from a different handle.
 */
typedef struct {
  const uint8_t *tiles_base;
  uint32_t tile_stride;
  uint32_t n_pairs;
  uint32_t c0; /**< first COLUMN of the batch (a multiple of 32) */
  uint32_t row_stride;
  uint32_t m_count;
  const float *act_scale;
  const int32_t *act_zp;
  const int32_t *colsum_a; /**< the first weight's, at column 0 */
  const float *w_scale_a;
  const float *bias_a;
  const int32_t *colsum_b; /**< the second weight's, at column 0 */
  const float *w_scale_b;
  const float *bias_b;
  float *dst;
  uint32_t dst_stride;
} hvx_dq_mul_job;

/** @brief hvx_worker_pool_func over an hvx_dq_mul_job; units are pairs. */
void hvx_dq_mul_worker(uint32_t n_threads, uint32_t i, void *job);

void hvx_dequant_swiglu_acc_tiles_to_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_pairs,
  uint32_t g0, uint32_t row_stride, uint32_t m_count, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, uint32_t inter, float *dst, uint32_t dst_stride,
  hvx_worker_pool *pool);

#endif /* __NNTRAINER_HVX_DEQUANT_I32_H__ */

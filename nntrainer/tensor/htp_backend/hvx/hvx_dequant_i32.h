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
 * @brief hvx_dequant_acc_tile_to_f32 for an a16 activation: the tile's
 *        accumulator is 256 @a hi + @a lo (the high- and low-byte sweeps),
 *        combined in f32 -- hvx_dq_tiles_rows_worker's lo_off arithmetic,
 *        so a row's floats are the HMX epilogue's bit for bit.
 */
void hvx_dequant_acc16_tile_to_f32(
  const int32_t *hi, const int32_t *lo, uint32_t row_stride, uint32_t m_count,
  const float *act_scale, const int32_t *act_zp, const int32_t *colsum_w,
  const float *w_scale, const float *bias, float *out, uint32_t out_stride,
  int accumulate);

/**
 * @brief Rows [m0, m1) of a run of @a n_tiles adjacent staged tiles,
 *        dequantized (hvx_dequant_acc_tile_to_f32's operations) into
 *        registers and added into out row @a rows[m] times @a weights[m]
 *        (hvx_scale_add_rows_f32's operations): the MoE down epilogue's
 *        dequant and scatter without the f32 round trip between them, so
 *        the same floats land. Row-major -- an output row's n_tiles vectors
 *        are one contiguous run behind an l2fetch -- because tile-major
 *        every (row, tile) add was a DDR miss once the output outgrew the
 *        L2: 128 of a 2048-row call's 226 ms of worker time on the S26
 *        Ultra. colsum_w, w_scale, bias and out are offset to the first
 *        tile's column. @a rows must not repeat within [m0, m1) nor across
 *        concurrent calls (one expert's tokens).
 */
void hvx_dequant_acc_tiles_scatter_rows_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_tiles,
  uint32_t row_stride, uint32_t m0, uint32_t m1, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, float *out, uint32_t out_stride, const uint32_t *rows,
  const float *weights);

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
void hvx_dequant_acc_tiles_to_f32(
  const uint8_t *tiles_base, uint32_t tile_stride, uint32_t n_tiles,
  uint32_t nt0, uint32_t row_stride, uint32_t m_count, const float *act_scale,
  const int32_t *act_zp, const int32_t *colsum_w, const float *w_scale,
  const float *bias, float *dst_a, float *dst_b, uint32_t split,
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
  /** 0, or (hvx_dq_tiles_rows_worker only) the bytes from a tile to its
   *  activation's low-byte sweep: the tile holds the high-byte sweep and
   *  the accumulator is 256 hi + lo, combined in f32 (an a16 activation,
   *  hexkl_mm_opts.act16) */
  uint32_t lo_off;
  /** 0, or (hvx_dq_tiles_rows_worker only) every tile is the accumulator
   *  as three byte planes of HVX_DQ_PLANE_BYTES -- bytes 0, 1, 2 of its
   *  int24 value, element p at row p / 32, column p % 32, the convert
   *  unit's layout (hexkl_cvt.h) -- instead of int32 at row_stride */
  uint32_t planes24;
  /** 0, or (hvx_dq_tiles_rows_worker only) add into dst_a instead of
   *  storing, one add per element (hvx_dequant_acc_tile_to_f32's rule) */
  uint32_t accumulate;
} hvx_dq_tiles_job;

/** @brief Bytes of one accumulator byte plane (hvx_dq_tiles_job.planes24). */
#define HVX_DQ_PLANE_BYTES 2048u

/** @brief hvx_worker_pool_func over an hvx_dq_tiles_job. */
void hvx_dq_tiles_worker(uint32_t n_threads, uint32_t i, void *job);

/** @brief An hvx_dq_tiles_job taken row-major (dst_a only): a lane takes a
 *         run of rows and writes each row's n_tiles vectors as one
 *         contiguous run, then hands the row to @a row_done while it is
 *         still in the cache (a norm, a rotation: the FC call's per-head
 *         passes). The floats are hvx_dq_tiles_worker's. Units for the
 *         submit: m_count. */
typedef struct {
  hvx_dq_tiles_job t;
  /** NULL, or called once per row with the row's first written element
   *  and its index in [0, m_count) */
  void (*row_done)(void *ctx, float *seg, uint32_t row);
  void *ctx;
} hvx_dq_rows_job;

void hvx_dq_tiles_rows_worker(uint32_t n_threads, uint32_t i, void *job);

/**
 * @brief The gate_up epilogue that stores no f32 (hexkl_mm_opts.glu_lo):
 *        byte-plane tiles (t.planes24) only, t.n_tiles gate tiles from n-tile
 *        t.nt0 and then the t.n_tiles up tiles opposite them. Each pair is
 *        dequantized as hvx_dq_tiles_rows_worker does, gated by
 *        hvx_geglu_det_sf / hvx_swiglu_det_sf and quantized at one
 *        encoding into the two byte planes of AH tiles, four rows a store:
 *        the bytes of that worker, hvx_glu_row_f32 and
 *        hvx_quant_params_pack_u16_ah in a row. Units for the submit:
 *        (m_count + 3) / 4 * n_tiles.
 */
typedef struct {
  hvx_dq_tiles_job t; /**< dst_a, dst_b, split, dst_stride unused */
  uint32_t inter;     /**< the up tiles' column offset in t's tables */
  uint32_t act;       /**< HVX_GLU_SILU or HVX_GLU_GELU_TANH */
  float scale;        /**< the codes' encoding */
  int32_t zp;
  uint8_t *lo, *hi; /**< the planes' row block of t's row 0 */
  uint32_t col; /**< the planes' column of gate tile t.nt0, a multiple of 32 */
  /** NULL, or (act HVX_GLU_GELU_TANH) hvx_gelu_tab_build's tables in VTCM:
      gelu(g) is then read off them instead of computed, through land's
      HVX_GELU_TAB_LANE_BYTES a lane (VTCM too, for the gathers) */
  const float *tab;
  uint8_t *land;
} hvx_dq_glu16_job;

void hvx_dq_glu16_worker(uint32_t n_threads, uint32_t i, void *job);

/**
 * GeGLU's gelu by linear interpolation in a table of nodes g = k / 256 - 6,
 * k < HVX_GELU_TAB_NODES, and g itself past 6 (where gelu(g) is g to 1e-8,
 * as it is 0 below -6). On Gemma-4 E2B's real gate / up it moves 0.012% of
 * the down's input codes, by one, against an exact GELU -- a tenth of the
 * tanh / erf GELU gap -- where the 40 ops of hvx_geglu_det_sf were 10% of
 * the prefill. The values, then the slopes T[k + 1] - T[k], each
 * HVX_GELU_TAB_STRIDE floats.
 */
#define HVX_GELU_TAB_NODES 3073u
#define HVX_GELU_TAB_STRIDE 3104u
#define HVX_GELU_TAB_LANE_BYTES 1024u
#define HVX_GELU_TAB_BYTES(lanes)                                              \
  (2u * 4u * HVX_GELU_TAB_STRIDE + (lanes) * HVX_GELU_TAB_LANE_BYTES)

/** @brief Fills @a tab (VTCM, 128-byte aligned, 2 * HVX_GELU_TAB_STRIDE
 *         floats) with the tables: the nodes through hvx_geglu_det_sf. */
void hvx_gelu_tab_build(float *tab);

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
  uint32_t act; /**< HVX_GLU_SILU or HVX_GLU_GELU_TANH */
} hvx_dq_swiglu_job;

/** @brief The gated activation the gate/up epilogue applies: silu(g)*u
 *         (LFM2) or gelu_tanh(g)*u (Gemma-4). The same code is the one
 *         value the layer call carries (nntr_hvx.idl mm_u8i4_moe_layer act). */
#define HVX_GLU_SILU 0u
#define HVX_GLU_GELU_TANH 1u

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
 * @param geglu       1: hvx_geglu_det_sf (gelu_tanh(gate) * up) instead
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
  const float *bias, uint32_t inter, uint32_t act, float *dst,
  uint32_t dst_stride, hvx_worker_pool *pool);

#endif /* __NNTRAINER_HVX_DEQUANT_I32_H__ */

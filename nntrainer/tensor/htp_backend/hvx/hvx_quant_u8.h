// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_quant_u8.h
 * @date   03 Aug 2026
 * @brief  Per-row asymmetric uint8 dynamic activation quantization
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_QUANT_U8_H__
#define __NNTRAINER_HVX_QUANT_U8_H__

#include <stdint.h>

#include "hvx_worker_pool.h"

/**
 * @brief Computes the per-row scale and zero point (K1).
 *
 * x[m][k] is recovered as scale[m] * (u[m][k] - zp[m]).
 *
 * Rows at or past @a m_valid are padding: they get scale 1 and zp 0 so a
 * host reference can reproduce them without special cases.
 *
 * @param[in]  x        activation, m_valid rows by k columns, row-major f32
 * @param[in]  m_valid  rows carrying real data
 * @param[in]  m_pad    rows after padding up to a multiple of 64
 * @param[out] scale    m_pad entries
 * @param[out] zp       m_pad entries, each in [0, 255]
 * @param[in]  pool     rows are independent, so this splits by row range.
 *                      NULL runs single-threaded.
 */
void hvx_quant_rows_u8_params(const float *x, uint32_t m_valid, uint32_t m_pad,
                              uint32_t k, float *scale, int32_t *zp,
                              hvx_worker_pool *pool);

/**
 * @brief Quantizes to uint8 and writes AH tiles (K2).
 *
 * Writes directly in the layout the HMX activation port expects: 64x32
 * tiles, flat row-major inside a tile, tiles in (row_block, inner_tile)
 * order at a 2048-byte stride. No separate layout pass is needed for
 * 8-bit activations.
 *
 * Rounding is round-to-nearest-even, matching hvx_sf_to_w_rne.
 *
 * @param[in]  out_ah  destination, m_pad * k bytes. Usually VTCM, and then
 *                     it must be 2048-byte aligned.
 * @param[in]  pool    splits the vectorized part by k-tile, not by row: a
 *                     k-tile's destination bytes are its own, disjoint from
 *                     every other k-tile's, so this preserves the 4-row
 *                     vectorized store groups a row split would break.
 *                     NULL runs single-threaded.
 * @return AEE_SUCCESS, or AEE_ENOMEMORY if the small per-row vector cache
 *         this needs to parallelize safely fails to allocate.
 */
int hvx_quant_pack_u8_ah(const float *x, uint32_t m_valid, uint32_t m_pad,
                         uint32_t k, const float *scale, const int32_t *zp,
                         uint8_t *out_ah, hvx_worker_pool *pool);

/**
 * @brief Same pack, but destination row d takes its input from source row
 *        @a row_map[d].
 *
 * hvx_quant_pack_u8_ah is this with @a row_map NULL. @a scale and @a zp are
 * indexed by DESTINATION row, so a caller that repeats a source row (a MoE
 * token routed to several experts does) must repeat its quantization
 * parameters to match -- they belong to the source row and do not change
 * with where it lands.
 *
 * Exists so a MoE layer can pack straight into expert order, 64-row block
 * by 64-row block, and skip gathering rows out of a row-ordered buffer
 * afterwards. That gather read 32 bytes at a time from addresses scattered
 * across the whole activation and cost 3.2 ms a layer; tidying its
 * destination did nothing, because the scattered reads were the cost
 * (doc 46 section 26.3).
 */
int hvx_quant_pack_u8_ah_mapped(const float *x, const uint32_t *row_map,
                                uint32_t m_valid, uint32_t m_pad, uint32_t k,
                                const float *scale, const int32_t *zp,
                                uint8_t *out_ah, hvx_worker_pool *pool);

/**
 * @brief The mapped pack for destination rows [m0, m1) only, on the
 *        calling thread. Same bytes as hvx_quant_pack_u8_ah_mapped would
 *        write for those rows. m0 and m1 are multiples of 4 (a row group).
 *
 * A row group's bytes in a tile are its own (r0*32 within every k-tile),
 * so disjoint row ranges can be packed by different threads in any order
 * -- this is the unit the MoE kernel hands to the worker pool's background
 * lane, 16 rows at a time, so the pack runs under the first experts' HMX
 * issue instead of before it and a unit never holds a worker for more
 * than a few microseconds. Every row in the range is packed; a caller
 * with fewer valid rows pads @a row_map, as the MoE kernel's slot order
 * already does.
 */
void hvx_quant_pack_u8_ah_rows(const float *x, const uint32_t *row_map,
                               uint32_t m0, uint32_t m1, uint32_t k,
                               const float *scale, const int32_t *zp,
                               uint8_t *out_ah);

#endif /* __NNTRAINER_HVX_QUANT_U8_H__ */

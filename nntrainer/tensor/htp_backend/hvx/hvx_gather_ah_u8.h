// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_gather_ah_u8.h
 * @date   10 Sep 2026
 * @brief  Pick rows out of an AH-tiled uint8 activation into a fresh block
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_GATHER_AH_U8_H__
#define __NNTRAINER_HVX_GATHER_AH_U8_H__

#include <stdint.h>

#include "hvx_worker_pool.h"

/**
 * @brief Gathers rows of an AH-tiled uint8 activation into one 64-row block.
 *
 * The MoE layer quantizes its whole activation once and then hands each
 * expert the rows routing gave it. Doing that as "gather the f32 rows, then
 * quantize the block" reads four times the bytes at top-4 routing and scans
 * 1776 rows where 444 would do; doing it here means the quantizer runs once
 * and this moves uint8.
 *
 * It also means the bytes are not requantized, which is the point: they are
 * whatever hvx_quant_pack_u8_ah wrote, so the result is identical to
 * quantizing each expert's block separately -- row quantization is
 * independent, so a row's scale and zp do not depend on which rows it was
 * grouped with. Reimplementing the quantizer's arithmetic here would put
 * that identity at risk for no gain.
 *
 * AH layout, which both sides must agree on: 64x32 tiles, flat row-major
 * inside a tile, tiles in (row_block, inner_tile) order at a 2048-byte
 * stride. So source row t's bytes for inner tile kt start at
 * (t/64)*k_tiles*2048 + kt*2048 + (t%64)*32.
 *
 * @param[out] dst_ah   one block: k/32 tiles of 2048 bytes. Zeroed first,
 *                      so the rows past n_rows are deterministic rather
 *                      than whatever the previous expert left there.
 * @param[in]  src_ah   the whole activation, m_pad rows, AH-tiled
 * @param[in]  rows     n_rows source row indices, each < m_pad
 * @param[in]  n_rows   at most 64
 * @param[in]  k        must be a multiple of 32
 * @param[in]  pool     splits by row; NULL runs on the caller
 */
void hvx_gather_ah_u8(uint8_t *dst_ah, const uint8_t *src_ah,
                      const uint32_t *rows, uint32_t n_rows, uint32_t k,
                      hvx_worker_pool *pool);

#endif /* __NNTRAINER_HVX_GATHER_AH_U8_H__ */

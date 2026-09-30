// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_acc_tile.h
 * @date   08 Aug 2026
 * @brief  Derives the int32 accumulator tile's VTCM layout at runtime
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * hexkl_micro_hmx_acc_read_int32 lands a 64x32 int32 tile in VTCM in a layout
 * HexKL does not document, which is the whole reason
 * hexkl_micro_hmx_copy_32b_to_submatrix exists: it knows the shuffle and
 * un-shuffles into a plain row-major matrix. Measured, that scalar copy is
 * 52.8 us per 8 KB tile -- 80% of a prefill attention layer and 97% of a
 * decode one (docs/htp_attention/31_dataflow_as_built.md).
 *
 * Reading the tile ourselves needs the layout. Rather than guess it, ask the
 * vendor function: write the ramp 0, 1, ... 2047 into the tile, run the copy,
 * and the destination now holds, at every position, the SOURCE INDEX that
 * lands there. That is the permutation, exactly, with the vendor's own code
 * as the specification -- no reverse engineering, and a future SDK that
 * reshuffles is followed automatically instead of silently miscomputing.
 *
 * Only one shape of answer is useful: destination row r, column c coming from
 * source base + r*row_stride + c. Then each destination row is 32 contiguous
 * int32 -- exactly one HVX vector -- and dequant can read the tile in place.
 * Anything else leaves @c usable at 0 and the caller keeps the vendor copy.
 */

#ifndef __NNTRAINER_HEXKL_ACC_TILE_H__
#define __NNTRAINER_HEXKL_ACC_TILE_H__

#include <stdint.h>

/** @brief Rows in one int32 accumulator tile. */
#define HEXKL_ACC_TILE_ROWS 64u
/** @brief Columns in one tile. Also exactly one HVX f32 vector, which is what
 *         makes the in-place dequant a single load per row. */
#define HEXKL_ACC_TILE_COLS 32u

/** @brief What the probe found. */
typedef struct {
  int probed;          /**< the probe has run; the rest is meaningful */
  int usable;          /**< rows are contiguous and evenly strided */
  uint32_t base;       /**< source index of destination element (0, 0) */
  uint32_t row_stride; /**< int32 elements between consecutive tile rows */
} hexkl_acc_layout;

/**
 * @brief Probes once, then returns the cached result.
 *
 * CLOBBERS the tile at @a result_off, so call it before the tile holds
 * anything -- layer_run does, once, before its matmul loop starts.
 *
 * Not thread safe, on the same single-owner assumption as the HMX lock: the
 * FastRPC thread that owns the session is the only caller.
 *
 * @param result_off  offset of the 8 KB accumulator tile within @a vtcm_base,
 *                     already bounds-checked by the caller
 */
const hexkl_acc_layout *hexkl_acc_layout_get(uint8_t *vtcm_base,
                                             uint32_t result_off);

#endif /* __NNTRAINER_HEXKL_ACC_TILE_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_hmx_geom.h
 * @date   11 Aug 2026
 * @brief  HMX tile geometry derived from HexKL's block macros
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Every file that lays bytes out for the HMX ports needs the same handful of
 * tile sizes. They were previously spelled out as literals in three places --
 * hexkl_mm_u8i4.c, hexkl_mm_u8i4_dma.c and hvx_quant_u8.c -- which agreed
 * only by inspection. The quantizer writes activation tiles at one stride
 * and the matmul reads them at another, so a divergence there is silent
 * wrong data rather than a build failure.
 *
 * Deriving them from HEXKL_HMX_* instead means HexKL's own header is the
 * single source, and the static assertions below fail the build if a future
 * SDK changes a block size out from under an assumption in this tree.
 */

#ifndef __NNTRAINER_HEXKL_HMX_GEOM_H__
#define __NNTRAINER_HEXKL_HMX_GEOM_H__

#include "hexkl_micro.h"

/** @brief Rounds @a v up to a multiple of @a a. Division form, not a
 *         power-of-two bitmask, so it is correct for any alignment. */
#define HEXKL_ROUND_UP_U32(v, a) ((((v) + ((a)-1)) / (a)) * (a))

/** @brief Rows in one HMX tile (activation and accumulator). */
#define HEXKL_TILE_ROW ((uint32_t)HEXKL_HMX_INT8_BLOCK_N_ROW)
/** @brief Reduction depth of one HMX tile. */
#define HEXKL_TILE_INNER ((uint32_t)HEXKL_HMX_INT8_BLOCK_N_INNER)
/** @brief Columns in one HMX tile. */
#define HEXKL_TILE_COL ((uint32_t)HEXKL_HMX_INT8_BLOCK_N_COL)

/** @brief Bytes in one uint8 activation tile: one byte per value. */
#define HEXKL_ACT_TILE_BYTES (HEXKL_TILE_ROW * HEXKL_TILE_INNER)
/** @brief Bytes in one packed int4 weight tile: two values per byte. */
#define HEXKL_WEIGHT_TILE_BYTES_I4 (HEXKL_TILE_INNER * HEXKL_TILE_COL / 2u)
/** @brief Bytes in one int32 accumulator tile. */
#define HEXKL_ACC_TILE_BYTES (HEXKL_TILE_ROW * HEXKL_TILE_COL * 4u)

/** The activation port reads tiles at the alignment HexKL publishes, so a
    tile that is not exactly that many bytes would make the quantizer's
    stride and the matmul's disagree. */
_Static_assert(HEXKL_ACT_TILE_BYTES == HEXKL_HMX_ACTIVATION_ALIGNMENT,
               "activation tile size must equal the activation alignment");
_Static_assert(HEXKL_ACT_TILE_BYTES == 2048u, "HMX activation tile is 64x32");
_Static_assert(HEXKL_WEIGHT_TILE_BYTES_I4 == 512u, "i4 weight tile is 32x32/2");
_Static_assert(HEXKL_ACC_TILE_BYTES == 8192u,
               "int32 accumulator tile is 64x32");

#endif /* __NNTRAINER_HEXKL_HMX_GEOM_H__ */

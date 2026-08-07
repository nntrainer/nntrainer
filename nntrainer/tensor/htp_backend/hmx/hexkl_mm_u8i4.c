// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_mm_u8i4.c
 * @date   03 Aug 2026
 * @brief  VTCM layout and HMX orchestration for the A8W4 matmul
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <stddef.h>

#include <AEEStdErr.h>

#include "hexkl_micro.h"

#include "hexkl_mm_u8i4.h"
#include "hvx_dequant_i32.h"

#define ROUND_UP_U32(v, a) ((((v) + ((a)-1)) / (a)) * (a))

/** @brief Bytes in one packed i4 weight tile (32x32 values). */
#define WEIGHT_TILE_BYTES 512u
/** @brief Bytes in one int32 accumulator tile (64x32 values). */
#define ACC_TILE_BYTES 8192u

int hexkl_mm_u8i4_plan(uint8_t *vtcm_base, uint32_t vtcm_size, uint32_t m_pad,
                       uint32_t k, uint32_t n, hexkl_mm_u8i4_layout *out) {
  if (!vtcm_base || !out || k == 0 || n == 0 || m_pad == 0) {
    return AEE_EBADPARM;
  }
  if ((k % HEXKL_HMX_INT8_BLOCK_N_INNER) != 0 ||
      (n % HEXKL_HMX_INT8_BLOCK_N_COL) != 0 ||
      (m_pad % HEXKL_HMX_INT8_BLOCK_N_ROW) != 0) {
    return AEE_EBADPARM;
  }

  const uint32_t n_ktiles = k / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t n_ntiles = n / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t n_rblocks = m_pad / HEXKL_HMX_INT8_BLOCK_N_ROW;

  out->vtcm_base = vtcm_base;
  out->vtcm_size = vtcm_size;

  out->w_base = 0u;
  out->w_size = n_ktiles * n_ntiles * WEIGHT_TILE_BYTES;

  // Activation tiles must sit on HEXKL_HMX_ACTIVATION_ALIGNMENT, and each
  // tile is exactly that many bytes, so aligning the base is enough.
  out->act_base = ROUND_UP_U32(out->w_size, HEXKL_HMX_ACTIVATION_ALIGNMENT);
  out->act_size = n_rblocks * n_ktiles * HEXKL_HMX_ACTIVATION_ALIGNMENT;

  out->result_off = out->act_base + out->act_size;

  // The unshuffled tile lands next to the shuffled one. result_off is
  // 2048-aligned and the tile is 8192 bytes, so this inherits the
  // alignment; HVX only needs 128 of it.
  out->unshuf_off = out->result_off + ACC_TILE_BYTES;

  // Put the config region at the top of the arena, rounded down to its
  // alignment so the rounding can never eat into the result region.
  if (vtcm_size < hexkl_micro_hmx_config_size()) {
    return AEE_ENOMEMORY;
  }
  out->config_off = (vtcm_size - hexkl_micro_hmx_config_size()) &
                    ~(HEXKL_HMX_CONFIG_ALIGNMENT - 1u);

  if (out->unshuf_off + ACC_TILE_BYTES > out->config_off) {
    return AEE_ENOMEMORY;
  }
  return AEE_SUCCESS;
}

int hexkl_mm_u8i4_bake_weights(const hexkl_mm_u8i4_layout *L,
                               const int8_t *w_i4_rm, uint32_t k, uint32_t n) {
  if (!L || !w_i4_rm) {
    return AEE_EBADPARM;
  }
  const uint32_t n_ktiles = k / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t n_ntiles = n / HEXKL_HMX_INT8_BLOCK_N_COL;

  for (uint32_t kt = 0; kt < n_ktiles; ++kt) {
    for (uint32_t nt = 0; nt < n_ntiles; ++nt) {
      const uint32_t off = L->w_base + (kt * n_ntiles + nt) * WEIGHT_TILE_BYTES;
      int res =
        hexkl_micro_hmx_rm_to_wh_i4(L->vtcm_base, off, w_i4_rm, kt, nt, n);
      if (res != AEE_SUCCESS) {
        return res;
      }
    }
  }
  return AEE_SUCCESS;
}

/**
 * @brief Accumulates one output tile over the whole K reduction and reads
 *        it back into L->result_off.
 *
 * Shared by both entry points so the two can never drift on the tile index
 * arithmetic, which is the part a reader cannot check by inspection.
 *
 * @return AEE_SUCCESS or the first failing HexKL micro status.
 */
static inline int accumulate_tile(const hexkl_mm_u8i4_layout *L,
                                  uint32_t n_ktiles, uint32_t n_ntiles,
                                  uint32_t rb, uint32_t nt) {
  hexkl_micro_hmx_acc_clear_int32();

  for (uint32_t kt = 0; kt < n_ktiles; ++kt) {
    const uint32_t act_off =
      L->act_base + (rb * n_ktiles + kt) * HEXKL_HMX_ACTIVATION_ALIGNMENT;
    const uint32_t w_off = L->w_base + (kt * n_ntiles + nt) * WEIGHT_TILE_BYTES;
    int res = hexkl_micro_hmx_mm_u8i4(L->vtcm_base, act_off, w_off);
    if (res != AEE_SUCCESS) {
      return res;
    }
  }
  return hexkl_micro_hmx_acc_read_int32(L->vtcm_base, L->config_off,
                                        L->result_off);
}

int hexkl_mm_u8i4_run(const hexkl_mm_u8i4_layout *L, uint32_t m_pad, uint32_t k,
                      uint32_t n, int32_t *acc_i32) {
  if (!L || !acc_i32) {
    return AEE_EBADPARM;
  }
  const uint32_t n_ktiles = k / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t n_ntiles = n / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t n_rblocks = m_pad / HEXKL_HMX_INT8_BLOCK_N_ROW;

  for (uint32_t rb = 0; rb < n_rblocks; ++rb) {
    for (uint32_t nt = 0; nt < n_ntiles; ++nt) {
      int res = accumulate_tile(L, n_ktiles, n_ntiles, rb, nt);
      if (res != AEE_SUCCESS) {
        return res;
      }
      // Going through copy_32b_to_submatrix keeps us off the accumulator's
      // shuffled layout, which HexKL does not document.
      res = hexkl_micro_hmx_copy_32b_to_submatrix(L->vtcm_base, L->result_off,
                                                  acc_i32, rb, nt, m_pad, n);
      if (res != AEE_SUCCESS) {
        return res;
      }
    }
  }
  return AEE_SUCCESS;
}

int hexkl_mm_u8i4_run_dequant(const hexkl_mm_u8i4_layout *L, uint32_t m_valid,
                              uint32_t m_pad, uint32_t k, uint32_t n,
                              const hexkl_mm_u8i4_dequant *D) {
  if (!L || !D || !D->act_scale || !D->act_zp || !D->colsum_w || !D->w_scale ||
      !D->bias || !D->out) {
    return AEE_EBADPARM;
  }
  const uint32_t n_ktiles = k / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const uint32_t n_ntiles = n / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t n_rblocks = m_pad / HEXKL_HMX_INT8_BLOCK_N_ROW;

  // tile_row/col of 0 with the tile's own dimensions as the output extent
  // makes copy_32b_to_submatrix write a dense 64x32 block, so element
  // (r, c) lands at unshuf_off + (r*32 + c)*4 and every row starts on a
  // 128-byte boundary.
  int32_t *const unshuf = (int32_t *)(L->vtcm_base + L->unshuf_off);

  for (uint32_t rb = 0; rb < n_rblocks; ++rb) {
    const uint32_t row0 = rb * HEXKL_HMX_INT8_BLOCK_N_ROW;
    if (row0 >= m_valid) {
      // Nothing in this row block reaches the output, so skip the multiply
      // as well as the epilogue.
      continue;
    }
    const uint32_t rows_left = m_valid - row0;
    const uint32_t n_rows = rows_left < HEXKL_HMX_INT8_BLOCK_N_ROW
                              ? rows_left
                              : HEXKL_HMX_INT8_BLOCK_N_ROW;

    for (uint32_t nt = 0; nt < n_ntiles; ++nt) {
      const uint32_t col0 = nt * HEXKL_HMX_INT8_BLOCK_N_COL;

      int res = accumulate_tile(L, n_ktiles, n_ntiles, rb, nt);
      if (res != AEE_SUCCESS) {
        return res;
      }

      res = hexkl_micro_hmx_copy_32b_to_submatrix(
        L->vtcm_base, L->result_off, unshuf, 0, 0, HEXKL_HMX_INT8_BLOCK_N_ROW,
        HEXKL_HMX_INT8_BLOCK_N_COL);
      if (res != AEE_SUCCESS) {
        return res;
      }

      hvx_dequant_tile_i32_to_f32(unshuf, n_rows, D->act_scale + row0,
                                  D->act_zp + row0, D->colsum_w + col0,
                                  D->w_scale + col0, D->bias + col0,
                                  D->out + (size_t)row0 * n + col0, n);
    }
  }
  return AEE_SUCCESS;
}

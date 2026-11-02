// SPDX-License-Identifier: Apache-2.0
/**
 * @file hvx_gemv_u8tl2.h
 * @brief Exact HVX LUT GEMV and native TL2 expansion.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __NNTRAINER_HVX_GEMV_U8TL2_H__
#define __NNTRAINER_HVX_GEMV_U8TL2_H__

#include "../htp_tl2_codec.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t hvx_tl2_table_bytes(uint32_t K);
/** Choose 1/2/4 K partitions to balance 128-column panels across workers.
 * Partial int32 results must be summed before scale/zero-point correction. */
uint32_t hvx_tl2_k_splits(uint32_t K, uint32_t N, uint32_t workers);

/** Restore at most 96x128 weights (four native blocks) directly to HMX WH.
 * src contains ceil(rows/24)*640 bytes, dst has row_stride bytes per 32 K
 * rows. Source and destination must not overlap. No scales are changed. */
void hvx_tl2_expand_wh96(const uint8_t *src, uint32_t rows, uint32_t columns,
                         uint32_t row_stride, int magnitude, uint8_t *dst);

/** Same bytes as expand_wh96 with its non-overlap contract made visible to
 * the compiler. The LFM prefill caller uses only its separate native source
 * slabs; in-place weight staging retains its snapshot and original entry. */
void hvx_tl2_expand_wh96_disjoint(const uint8_t *src, uint32_t rows,
                                  uint32_t columns, uint32_t row_stride,
                                  int magnitude, uint8_t *dst);

/** Restore consecutive complete 96x128 groups from disjoint native input.
 * Source groups occupy 2560 bytes each; output groups occupy three WH rows.
 * Aligned callers amortize setup and use fixed dimensions. Partial groups
 * must use expand_wh96_disjoint with their actual row count. */
void hvx_tl2_expand_wh96_disjoint_batch(const uint8_t *src, uint32_t groups,
                                        uint32_t row_stride, int magnitude,
                                        uint8_t *dst);

/** Exact int16 LUTs, 512 bytes per 24 input rows. The u8 activation is
 * contiguous and has K bytes; the last block is read with bounds checks.
 * Workers partition [block_begin,block_end) of ceil(K/24). Tables depend only
 * on activations, and may be shared across all experts with the same input.
 * Preparation may run concurrently with weight DMA into a disjoint region. */
void hvx_tl2_prepare(const uint8_t *act, uint32_t K, uint32_t block_begin,
                     uint32_t block_end, int16_t *tables);

/** One 128-column panel, producing 128 contiguous int32 dot products.
 * src, tables, out are 128-byte aligned. No lookahead or over-read.
 * Padded columns are included; callers copy only their logical N columns.
 * Accumulation remains integer-exact before the separate scale epilogue. */
/** For affine-u8 input: y[n] = a_scale * w_scale[n] *
 * (out[n] - a_zero_point * sum_k(ternary_weight[k,n])). */
void hvx_gemv_u8tl2_panel(const uint8_t *src, uint32_t K, const int16_t *tables,
                          int32_t *out);

#ifdef __cplusplus
}
#endif
#endif

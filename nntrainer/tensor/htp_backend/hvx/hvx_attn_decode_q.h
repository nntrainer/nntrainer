// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_attn_decode_q.h
 * @date   28 Sep 2026
 * @brief  Pure-HVX decode attention over the int8 / int4 KV cache
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_ATTN_DECODE_Q_H__
#define __NNTRAINER_HVX_ATTN_DECODE_Q_H__

#include <stdint.h>

#include "hexkl_attn_q.h"
#include "hvx_worker_pool.h"

/**
 * @brief Attention for few query rows over a registered quantized cache,
 *        one (row, head) per work unit, no HMX.
 *
 * The decode counterpart of hexkl_attn_q_prefill, with the same
 * quantization of everything (Q per-row uint8, P' per row and 32-row
 * block per V group as uint8, the registry's K and V) so the CPU model
 * with bc = 32 describes it. Scores and values come from the registry's
 * masters by vrmpy: one instruction per 32 cache rows x 4 dims for Q.K^T
 * and per 4 rows x 32 dims for P'.V, Q and P' as 4 bytes in a scalar
 * register against the offset-binary masters. Softmax in f32, 32 rows per
 * vector, running quantities as all-lanes-equal vectors. K/V are read
 * from DDR straight from the registry; nothing touches VTCM or the DMA
 * engine. Requires head_dim in {32, 64, 96, 128}.
 *
 * @param pool  may be NULL
 * @param us    may be NULL; total on-DSP microseconds
 * @return AEE_SUCCESS or AEE_EBADPARM
 */
int hvx_attn_decode_q(const hexkl_attn_f16_shape *s, const hexkl_attn_q_io *io,
                      hvx_worker_pool *pool, uint64_t *us);

#endif /* __NNTRAINER_HVX_ATTN_DECODE_Q_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_attn_decode_f16.h
 * @date   14 Sep 2026
 * @brief  Pure-HVX flash attention for a handful of query rows (decode)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HVX_ATTN_DECODE_F16_H__
#define __NNTRAINER_HVX_ATTN_DECODE_F16_H__

#include <stdint.h>

#include "hexkl_attn_f16.h"
#include "hvx_worker_pool.h"

/**
 * @brief Attention for few query rows, one (row, head) per work unit, no HMX.
 *
 * llama.cpp sends decode (fewer than 5 tokens, head_dim <= 128) to a
 * pure-HVX kernel rather than HMX: with one or two real rows per 32-row
 * tile the matrix unit is almost entirely padding, and the accumulator
 * round trips dominate. This is that path for nntrainer's cache layout.
 * Same math and options as hexkl_attn_f16_prefill (causal/window ranges,
 * softcap, sinks), but scores are computed as fp16 dot products on HVX,
 * probabilities in fp16, and the output accumulates in f32 -- so it is
 * also the more precise of the two.
 *
 * K and V are read straight from DDR with vector loads; nothing touches
 * VTCM or the DMA engine, which is what lets every pool worker run its
 * own (row, head) units with no shared state. Requires head_dim in
 * {32, 64, 96, 128}.
 *
 * @param pool  may be NULL
 * @param us    may be NULL; total on-DSP microseconds
 * @return AEE_SUCCESS or AEE_EBADPARM
 */
int hvx_attn_decode_f16(const hexkl_attn_f16_shape *s,
                        const hexkl_attn_f16_io *io, hvx_worker_pool *pool,
                        uint64_t *us);

#endif /* __NNTRAINER_HVX_ATTN_DECODE_F16_H__ */

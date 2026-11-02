// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   htp_fixed_opts.h
 * @date   14 October 2026
 * @brief  Built-in HTP configuration values. Model properties and
 *         environment overrides are handled at their individual call sites.
 * @see    https://github.com/nntrainer/nntrainer/pull/4428
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HTP_FIXED_OPTS_H__
#define __NNTRAINER_HTP_FIXED_OPTS_H__

namespace nntrainer {
namespace htp_fixed {

/** Run the decode token as one resident DSP graph, returning the next id. */
constexpr bool E2E = true;

/** Default resident experts per MoE layer; moe_cache_experts overrides it. */
constexpr unsigned int MOE_CACHE_EXPERTS = 16u;

/** Release prefill's quantized KV caches once decode has its own copy. */
constexpr bool E2E_FREE_KVQ = true;

/** Drop an attention layer's CPU KV mirror once the DSP holds its rows. */
constexpr bool KV_DROP_CPU = true;

/** Drop host projection weights after registration in the DSP arena. */
constexpr bool DROP_HOST_FC = true;

/** Decode KV mode: 0 = fp16, 1 = int8 for sliding layers, 2 = int8 for all. */
constexpr int ATTN_M1_Q8 = 2;

/** Store dense projections in the DSP arena's 256 MiB chunks. */
constexpr bool FC_IN_ARENA = true;

/** Default DMA queue count for the M = 1 MoE weight feed. */
constexpr unsigned int MOE_DMA_QUEUES = 4u;

/** Predict the next MoE layer's experts on ARM when every router uses
 *  Gemma's softmax. Prediction never changes the DSP's routing. */
constexpr unsigned int PREDICT = 2u;

/** Reader count for prefetching predicted experts into spare DMA slots. */
constexpr unsigned int PREFETCH_READERS = 4u;
/** Decode readers for native TL2 experts with page cache bypass.
 *  Prefill lookahead and other file/cache formats use their own defaults. */
constexpr unsigned int TL2_DIRECT_PREFETCH_READERS = 2u;
constexpr unsigned int PREFETCH_SLOTS = 32u;
constexpr float PREFETCH_RESIDUAL = 0.5f;

/** Discard CPU prefill scratch after the first token. */
constexpr bool RELEASE_PREFILL_SCRATCH = true;

/** After the first token, grow towards this many experts per layer.
 *  Growth is bounded by PD_MAP_CEILING_MIB minus existing expert/FC arenas,
 *  the DSP heap and PD_MAP_MARGIN_MIB. */
constexpr unsigned int MOE_POOL_TARGET = 34u;
constexpr unsigned int PD_MAP_CEILING_MIB = 3840u;
constexpr unsigned int PD_MAP_MARGIN_MIB = 256u;

} // namespace htp_fixed
} // namespace nntrainer

#endif // __NNTRAINER_HTP_FIXED_OPTS_H__

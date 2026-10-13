// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   htp_fixed_opts.h
 * @date   14 October 2026
 * @brief  The HTP path's runtime configuration, built in. These were
 *         environment options while the configuration was being found; the
 *         values below are the ones the Gemma4 26B A4B runs were validated
 *         with (the fastest measured), so every HTP run takes them and
 *         nothing has to set them. The only knob left to the environment is
 *         the weight page-cache policy (NNTR_HTP_PAGE_CACHE,
 *         weight_file_reader.h).
 * @see    https://github.com/nntrainer/nntrainer/pull/4428
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NNTRAINER_HTP_FIXED_OPTS_H__
#define __NNTRAINER_HTP_FIXED_OPTS_H__

namespace nntrainer {
namespace htp_fixed {

/** The whole decode token runs as one resident DSP graph and the token
 *  driver returns only the next id (formerly NNTR_HTP_E2E=1). */
constexpr bool E2E = true;

/** Experts kept resident per MoE layer in the DSP arena (formerly
 *  NNTR_MOE_CACHE_EXPERTS=16); a model's moe_cache_experts, when set, still
 *  names its own count. */
constexpr unsigned int MOE_CACHE_EXPERTS = 16u;

/** The prefill's quantized KV caches are released once the decode graph
 *  holds its own (formerly NNTR_HTP_E2E_FREE_KVQ=1). */
constexpr bool E2E_FREE_KVQ = true;

/** The CPU mirror of an attention layer's KV cache is dropped once the DSP
 *  holds the rows (formerly NNTR_HTP_KV_DROP_CPU=1). */
constexpr bool KV_DROP_CPU = true;

/** The host copies of the dense projections are dropped once they are
 *  registered in the DSP arena (formerly NNTR_HTP_DROP_HOST_FC=1). */
constexpr bool DROP_HOST_FC = true;

/** Decode attention over the int8 KV cache for every attention layer
 *  (formerly NNTR_HTP_ATTN_M1_Q8=2; 1 was the sliding layers only, 0 fp16). */
constexpr int ATTN_M1_Q8 = 2;

/** The dense projections live in the DSP arena, in 256 MiB chunks, not on
 *  the DSP heap (formerly NNTR_HTP_FC_IN_ARENA=1). */
constexpr bool FC_IN_ARENA = true;

} // namespace htp_fixed
} // namespace nntrainer

#endif // __NNTRAINER_HTP_FIXED_OPTS_H__

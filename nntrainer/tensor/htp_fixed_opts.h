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

/** The decode-fast environment of #4416's run_gemma4_memory.sh, measured on
 * the S26 Ultra (2026-10-14, page cache off, C16 p1024 g128): with it the
 * decode goes 9.8 -> 10.1 tok/s and the misses' exposed wait 46 -> 40 ms a
 * token, at the same arena. The pool growth it also set (NNTR_MOE_CACHE_GROW)
 * stays an option: it is what takes the decode to 13.9 tok/s, for 1 GB of
 * arena. */

/** The M = 1 MoE feed's DMA queues (formerly NNTR_MOE_DMA_QUEUES=4; one
 *  queue reads 28-34 GB/s on the v81 engine, four 62 GB/s). */
constexpr unsigned int MOE_DMA_QUEUES = 4u;

/** The next MoE layer's experts predicted on the ARM from the router's
 *  input row (formerly NNTR_HTP_PREDICT=2), when every router is Gemma's
 *  softmax router; the DSP's routing is never changed by it. */
constexpr unsigned int PREDICT = 2u;

/** The predicted experts read into spare DMA slots ahead of their layer
 *  (formerly NNTR_MOE_PREFETCH_DIRECT=1 with NNTR_MOE_PREFETCH=4,
 *  NNTR_MOE_PREFETCH_READERS=4, NNTR_MOE_PREFETCH_SLOTS=32,
 *  NNTR_MOE_PREFETCH_RESIDUAL=0.5). */
constexpr unsigned int PREFETCH_READERS = 4u;
/** Already-native TL2 experts with page cache bypass: two decode readers
 *  reduced exposed miss wait on S26 Ultra at C16, p2048. Prefill lookahead
 *  and other file/cache formats retain their existing reader defaults. */
constexpr unsigned int TL2_DIRECT_PREFETCH_READERS = 2u;
constexpr unsigned int PREFETCH_SLOTS = 32u;
constexpr float PREFETCH_RESIDUAL = 0.5f;

/** The CPU prefill scratch is discarded once the first token is out
 *  (formerly NNTR_HTP_RELEASE_PREFILL_SCRATCH=1). */
constexpr bool RELEASE_PREFILL_SCRATCH = true;

/** After the first token the expert pool grows towards this many experts a
 *  layer (formerly NNTR_MOE_CACHE_GROW=22 over C=12), as far as the PD's
 *  address space allows: the growth is the part of PD_MAP_CEILING_MIB left
 *  above what is mapped (expert and FC arenas, the DSP heap) and
 *  PD_MAP_MARGIN_MIB. S26 Ultra, page cache off, C16 p1024: 10.1 -> 13.9
 *  tok/s for about 1 GB of arena; C32 + 22 did not map (fastrpc_mmap). */
constexpr unsigned int MOE_POOL_TARGET = 34u;
constexpr unsigned int PD_MAP_CEILING_MIB = 3840u;
constexpr unsigned int PD_MAP_MARGIN_MIB = 256u;

} // namespace htp_fixed
} // namespace nntrainer

#endif // __NNTRAINER_HTP_FIXED_OPTS_H__

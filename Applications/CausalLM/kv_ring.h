// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   kv_ring.h
 * @date   01 September 2026
 * @brief  Single source of truth for the sliding-window KV ring and the
 *         chunked-prefill size.
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details The ring rule is consumed by two independent translation units that
 * must not disagree:
 *
 *   - the MODEL side (models/transformer.h) sizes the KV placeholder and the
 *     KVCacheManager allocation -- how many PHYSICAL rows exist;
 *   - the LAYER side (layers/mha_core.cpp) modulo-maps the write position and
 *     the attention read view into those rows.
 *
 * A disagreement in the direction "model allocates Wcap, layer writes absolute"
 * is an out-of-bounds write, not a wrong answer, so the rule lives here and
 * both sides call it. Nothing in this header keeps state; every entry point is
 * a pure function of its arguments plus the process environment.
 */

#ifndef __CAUSALLM_KV_RING_H__
#define __CAUSALLM_KV_RING_H__

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <env_compat.h> // nntr_env_on (an auto-injected flag needs =0 to work)

namespace causallm {

/**
 * @brief Whether the engine this process resolved can host the ring at all.
 * @details The ring is only correct where the attention kernels modulo-map the
 * cache row, which today means the GPU attention paths. The host CPU attention
 * fallback walks absolute rows, so a CPU run must keep the linear cache.
 */
inline bool kvRingEngineEligible() {
  const char *e = std::getenv("NNTR_ENGINE");
  if (e != nullptr && std::string(e) == "cpu")
    return false;
  if (e != nullptr && std::string(e) == "cuda")
    return true;
#if defined(ENABLE_OPENCL)
  return true; // no explicit engine + an OpenCL build == the gpu engine
#else
  return false;
#endif
}

/**
 * @brief Whether a ring-AWARE attention arm is reachable in this configuration.
 * @details mha_core resolves attention through a cascade, and only three of its
 * arms take the ring capacity and read row (n % cap):
 * flash_attention_prefill_f16_cl, flash_decode_f16_cl and
 * cuda_attention_interleaved_fp16. The remaining arms (the two_conv family, the
 * OHWI image path, and the host compute_kcaches / gemm_attention fallback)
 * index the cache linearly from the LOGICAL key count, so pointing them at a
 * Wcap-high buffer reads past its end.
 *
 * The arms that do map the row are behind env gates that are readable here, so
 * the ring refuses to turn on unless one of them is actually selectable. This
 * is deliberately evaluated BEFORE any allocation happens: the answer feeds
 * kvRingCap(), which both sides use, so a refusal leaves the ordinary
 * full-height linear cache in place rather than leaving a ringed allocation
 * with a linear reader.
 *
 * Runtime failure of the selected arm still drops into a non-ring arm, so
 * mha_core additionally refuses those at dispatch time; this function only
 * removes the statically-knowable mismatches.
 */
inline bool kvRingArmAvailable() {
  const char *e = std::getenv("NNTR_ENGINE");
  if (e != nullptr && std::string(e) == "cuda")
    return nntr_env_on("NNTR_CUDA_ATTN"); // cuda_attention_interleaved_fp16
#if defined(ENABLE_OPENCL)
  // The flash arm sits inside the OHWI-direct GPU attention block, which needs
  // both NNTR_KV_OHWI and NNTR_MHA_GPU. The image arm, which preempts flash on
  // Adreno, is not ring-aware -- so its opt-ins disqualify the ring.
  if (!nntr_env_on("NNTR_KV_OHWI") || !nntr_env_on("NNTR_MHA_GPU"))
    return false;
  if (nntr_env_on("NNTR_KV_IMG_ATTN") || nntr_env_on("NNTR_MHA_GPU_IMG"))
    return false;
  return true;
#else
  return false;
#endif
}

/**
 * @brief Whether the KV ring is enabled for this process.
 * @details NNTR_KV_WINDOW_RING is the opt-in: unset or '0' keeps the linear
 * cache (the pre-ring behaviour, bit-identical), anything else REQUESTS the
 * ring. A request is granted only where it is also correct -- the engine can
 * host it (kvRingEngineEligible) and a ring-aware attention arm is reachable
 * (kvRingArmAvailable). A refused request is reported once so the reason is
 * visible instead of showing up as a silent performance or memory difference.
 */
inline bool kvRingEnabled() {
  if (!nntr_env_on("NNTR_KV_WINDOW_RING"))
    return false; // default OFF: opt-in for this cycle
  const bool ok = kvRingEngineEligible() && kvRingArmAvailable();
  if (!ok) {
    static bool reported = false;
    if (!reported) {
      reported = true;
      std::fprintf(stderr,
                   "[kv-window-ring] NNTR_KV_WINDOW_RING is set but no "
                   "ring-aware attention arm resolves in this configuration "
                   "(engine_eligible=%d arm_available=%d); keeping the linear "
                   "full-height KV cache. The ring needs NNTR_KV_OHWI=1 and "
                   "NNTR_MHA_GPU=1 on OpenCL, or NNTR_CUDA_ATTN=1 on "
                   "NNTR_ENGINE=cuda.\n",
                   (int)kvRingEngineEligible(), (int)kvRingArmAvailable());
    }
  }
  return ok;
}

/**
 * @brief Per-layer structural preconditions for the ring, evaluated the same
 *        way on the model side and on the layer side.
 * @param attention_sink the model uses the attention-sink attention variant,
 *        which reads the cache through the host compute path (no modulo map).
 * @param external_cache mha_core's external (5-input) KV cache mode; the
 *        layer-internal cache is allocated at full max_seq and is not ringed.
 * @details The int8 KV cache is likewise allocated at full max_seq and written
 * with absolute rows, so NNTR_KV_INT8 disqualifies the ring on both sides. That
 * condition used to exist only on the layer side, which left the model free to
 * ring the ALLOCATION while the layer wrote absolute rows into it.
 */
inline bool kvRingLayerEligible(bool attention_sink, bool external_cache) {
  if (attention_sink || !external_cache)
    return false;
  if (std::getenv("NNTR_KV_INT8") != nullptr)
    return false;
  return true;
}

/**
 * @brief Requested prefill chunk size (0 = no chunking / single-block prefill).
 * @details An explicit NNTR_PREFILL_CHUNK always wins (user override, per-GPU
 * tuning); a non-positive or unparseable value is REJECTED (treated as unset)
 * rather than wrapped into a ~4e9 unsigned, which the (W/C + 2) * C ring
 * arithmetic would have consumed. Otherwise, chunking follows the ring:
 * chunking is what bounds a launch's live key span, so the ring picks the
 * chunk, 4096 for every backend. The equal-thermal ring-on sweep is monotone in
 * the chunk but with a poor marginal ratio past 4096 (the next step up buys
 * under a percent of prefill for another GB of working set), and the CUDA
 * tensor-core GEMMs want a large chunk anyway -- so one constant, no backend
 * branch.
 *
 * This is the REQUEST, not what the prefill actually runs: a chunk cannot
 * exceed the activation-plane height it has to fit in. Use
 * effectivePrefillChunk() (or Transformer::prefillChunk(), which calls it)
 * anywhere the answer feeds sizing or control flow.
 */
inline unsigned int requestedPrefillChunk() {
  const char *pc = std::getenv("NNTR_PREFILL_CHUNK");
  if (pc != nullptr && pc[0] != '\0') {
    char *end = nullptr;
    const long v = std::strtol(pc, &end, 10);
    if (end != pc && *end == '\0' && v > 0)
      return static_cast<unsigned int>(v); // explicit override wins
    static bool reported = false;
    if (!reported) {
      reported = true;
      std::fprintf(stderr,
                   "[prefill-chunk] NNTR_PREFILL_CHUNK='%s' is not a positive "
                   "integer; ignoring it.\n",
                   pc);
    }
  }
  if (!kvRingEnabled())
    return 0u; // chunking is auto-enabled only by the ring
  return 4096u;
}

/**
 * @brief The prefill chunk that actually runs, given the activation-plane
 *        height it is fed through (0 = no chunking).
 * @param plane_height INIT_SEQ_LEN -- the height of the plane one chunk is fed
 *        at row 0 of. 0 means "unknown", which leaves the request unclamped.
 * @details Every consumer must read the SAME clamped number: the prompt budget,
 * the prefill drive loop, and the ring capacity (Wcap is a multiple of the
 * chunk). They used to disagree, and sizing the ring off the unclamped request
 * (a 4096 request against a 1024-row plane) leaves Wcap up to 4x too large.
 */
inline unsigned int effectivePrefillChunk(unsigned int plane_height) {
  const unsigned int c = requestedPrefillChunk();
  if (c == 0u || plane_height == 0u)
    return c;
  return std::min(c, plane_height);
}

/**
 * @brief Sliding-window KV ring capacity.
 * @details A sliding-window attention layer with local window W only ever
 * attends to the last W keys, so with chunked prefill -- which bounds one
 * launch's live key span to W+C -- its KV storage can be a ring of Wcap rows
 * instead of the full max_seq.
 *
 * Returns Wcap (the physical row capacity to allocate and modulo-index) for a
 * sliding layer, or 0 meaning "no ring, keep full max_seq" (full-attention
 * layer, ring disabled, no chunking, or no benefit). Every site -- placeholder
 * shape, KV allocation, cache write, attention kernel dispatch -- computes Wcap
 * from THIS one function so they stay consistent.
 *
 * Wcap is a multiple of C and >= W + C: a multiple of C means a C-aligned chunk
 * write never straddles the wrap seam (it stays one contiguous slice), and
 * >= W + C means the live window [pos-W+1, pos+C) never self-collides mod Wcap.
 * Returning 0 keeps the exact pre-ring behaviour, so ring-off is bit-identical.
 *
 * @param local_window W, the layer's sliding window (0 = full attention).
 * @param max_seq the full context window this layer would otherwise allocate.
 * @param chunk C, the chunk the prefill ACTUALLY runs --
 *        effectivePrefillChunk(), not requestedPrefillChunk(). It is a
 *        parameter rather than a call so that the caller's chunk and this cap
 *        cannot drift apart.
 */
inline unsigned int kvRingCap(unsigned int local_window, unsigned int max_seq,
                              unsigned int chunk) {
  if (!kvRingEnabled())
    return 0; // ring off -> full max_seq (bit-identical legacy)
  if (local_window == 0 || local_window >= max_seq)
    return 0; // full-attention layer -> no ring
  const unsigned int C = chunk;
  if (C == 0)
    return 0; // the ring requires chunked prefill to bound the live span
  // multiple of C, >= W + C (headroom so the window never wraps onto itself).
  const unsigned int cap = (local_window / C + 2u) * C;
  return (cap < max_seq) ? cap : 0u; // no benefit if it would not shrink
}

/**
 * @brief Physical cache row for an absolute position under a ring of `cap`
 *        rows (cap == 0 => linear, the identity).
 * @details The host-side twin of the kernels' `n % ring_cap`.
 */
inline unsigned long kvCacheRow(unsigned long abs_pos, unsigned int cap) {
  return cap ? (abs_pos % static_cast<unsigned long>(cap)) : abs_pos;
}

} // namespace causallm

#endif // __CAUSALLM_KV_RING_H__

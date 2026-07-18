// SPDX-License-Identifier: Apache-2.0
/**
 * @file   residency_policy.h
 * @date   04 July 2026
 * @brief  App-declared residency-boundary policy
 *
 * @details Decouples the core static-residency planner (tensor_pool /
 * manager) from application-specific layer & tensor NAMES. The residency
 * heuristic needs three application facts that used to be hardcoded in core as
 * string literals (a core->app coupling):
 *
 *   - RAISE: a tensor that a HOST producer nevertheless uploads to the cl_mem
 *     plane (so it may be GPU_CLMEM despite its CPU producer engine). Formerly
 *     the core default `clmem_raise = "embedding0:out0"`.
 *   - LOWER: a GPU-produced tensor whose lone HOST consumer explicitly reads it
 *     back once (so it may stay GPU_CLMEM despite that CPU consumer). Formerly
 *     the core default `clmem_lower = "output_norm:out0"`.
 *   - ENGINE-NEUTRAL layer types: a CPU-engine layer that actually
 * binds/consumes its inputs on the GPU plane (so it must NOT downgrade its
 * producer's output out of GPU_CLMEM). Formerly `node.getType() == "mha_core"`
 * in manager.cpp.
 *
 * The application declares these once at model build (see CausalLM), and core
 * reads the policy. Declaring the same names is byte-identical to the former
 * hardcoded defaults. The NNTR_CLMEM_RAISE / NNTR_CLMEM_LOWER env vars still
 * override the raise/lower patterns for debugging.
 *
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __RESIDENCY_POLICY_H__
#define __RESIDENCY_POLICY_H__

#include <string>
#include <vector>

namespace nntrainer {

/**
 * @brief Process-wide, application-populated residency boundary policy.
 */
struct ResidencyPolicy {
  /** comma-separated any-substring patterns for input-boundary RAISE tensors */
  std::string raise_patterns;
  /** comma-separated any-substring patterns for output-boundary LOWER tensors
   */
  std::string lower_patterns;
  /** layer getType() values that are engine-neutral GPU-plane consumers */
  std::vector<std::string> engine_neutral_types;

  /**
   * @brief the process-wide policy instance (app-populated before allocation).
   * @note  Defined out-of-line in tensor_pool.cpp so there is exactly ONE
   *        instance in libnntrainer.so, shared across the .so boundary by the
   *        application that populates it (avoids a per-.so inline-static copy).
   */
  static ResidencyPolicy &global();

  /** @brief is @a layer_type an app-declared engine-neutral GPU-plane consumer
   */
  bool isEngineNeutral(const std::string &layer_type) const {
    for (const auto &t : engine_neutral_types)
      if (t == layer_type)
        return true;
    return false;
  }
};

} // namespace nntrainer

#endif // __RESIDENCY_POLICY_H__

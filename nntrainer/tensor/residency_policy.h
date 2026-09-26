// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   residency_policy.h
 * @date   24 August 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Application-declared residency boundaries for the static residency
 *         planner.
 *
 * @details The residency planner decides where a tensor lives from facts core
 * owns: the producing layer's compute engine, whether every consumer is on the
 * same engine, and the data type. Three facts it cannot derive belong to the
 * model rather than to core, and an application declares them here once before
 * the graph allocates:
 *
 *   - RAISE: a tensor a host producer nevertheless uploads to the device plane
 *     itself, so it may be device-resident despite its CPU-engine producer.
 *   - LOWER: a device-produced tensor whose only host consumer reads it back
 *     once explicitly, so it may stay device-resident despite that consumer.
 *   - EXCLUDE: a tensor that must stay on the shared plane whatever the
 *     heuristic says (a sequence-persistent cache the host also touches).
 *   - engine-neutral layer types: a layer registered on the CPU engine that
 *     nevertheless binds its inputs on the device plane, so it must not
 *     downgrade its producer's output.
 *
 * Keeping them here is what lets core carry no model-specific tensor or layer
 * names. An application that declares nothing gets the pure heuristic.
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
  /** comma-separated substring patterns for input-boundary RAISE tensors */
  std::string raise_patterns;
  /** comma-separated substring patterns for output-boundary LOWER tensors */
  std::string lower_patterns;
  /** comma-separated substring patterns for tensors kept off the device plane
   */
  std::string exclude_patterns;
  /** layer types that are engine-neutral consumers of the device plane */
  std::vector<std::string> engine_neutral_types;

  /**
   * @brief the process-wide policy instance, populated before allocation.
   * @note  Defined out-of-line in tensor_pool.cpp so there is exactly one
   *        instance in the library, shared across the shared-object boundary
   *        with the application that populates it.
   */
  static ResidencyPolicy &global();

  /**
   * @brief is @a layer_type an application-declared engine-neutral consumer
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

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
 *
 * @details WRITE WINDOW, and the limits that follow from it. The policy is a
 * process-wide singleton the application fills in ONCE, before it builds any
 * model, and does not touch again. Two separate reads depend on that:
 * Manager::requestInputs() consults isEngineNeutral() while the graph is being
 * built, and TensorPool::allocate() reads the pattern lists at allocation. A
 * write between those two points produces a graph planned against one policy
 * and allocated against another, and nothing diagnoses it -- the placements
 * simply disagree with the consumer votes that were recorded earlier.
 *
 * Three consequences are stated rather than enforced, because enforcing them
 * means moving the policy onto the model (or onto the TensorPool), which is
 * where it belongs once a second caller needs a second policy:
 *
 *   - TWO MODELS IN ONE PROCESS SHARE ONE POLICY. A second model with
 *     different residency boundaries is NOT supported: the last write wins,
 *     for every model. This is the same process-wide latching NNTR_ENGINE
 *     has, and it is deliberate only in the sense that no caller needs
 *     otherwise yet.
 *   - NOT SYNCHRONISED. There is no lock. Populating it from one thread while
 *     another builds or allocates a graph is a data race.
 *   - NOT RE-READ. Changing a pattern after allocate() has run does not move
 *     a tensor; placement is a planner decision taken once (see
 *     residency_planner.h), so a late write affects only pools allocated
 *     after it.
 *
 * The patterns themselves are comma-separated substrings, not globs and not
 * regular expressions -- see ResidencyPlanner::nameMatchesAny() for exactly
 * what matches.
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
   * @note  Write it before building a model and leave it alone afterwards;
   *        the class comment states what a later write does and does not do.
   * @return the one policy instance in this process
   */
  static ResidencyPolicy &global();

  /**
   * @brief is @a layer_type an application-declared engine-neutral consumer
   * @param layer_type the consuming layer's registered type name
   * @return true if the application declared that type engine-neutral
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

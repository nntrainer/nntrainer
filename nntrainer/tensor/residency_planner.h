// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   residency_planner.h
 * @date   24 August 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Static residency-class planner.
 *
 * @details Assigns each planned tensor its ResidencyClass as a pure function of
 * what the graph already knows before it runs: the producing layer's compute
 * engine, whether every consumer is on the same engine, the data type, the
 * tensor name against the application-declared boundaries (ResidencyPolicy),
 * and what the pool's allocator can actually back.
 *
 * It is a planner decision rather than a runtime one on purpose. Flipping a
 * tensor between the shared and the device plane per edge, at execution time,
 * leaves a producer writing one plane while a consumer reads the other; the
 * planner already knows every tensor's producer, consumers and lifetime, so
 * the placement can be decided once and applied to all of them uniformly.
 * This is also the seam a partitioning strategy (whole-graph versus per-op
 * offload) plugs into later.
 */

#ifndef __RESIDENCY_PLANNER_H__
#define __RESIDENCY_PLANNER_H__

#include <string>

#include <common.h>      // ml::train::LayerComputeEngine
#include <memory_data.h> // ResidencyClass

namespace nntrainer {

/**
 * @brief Per-pool residency configuration and the classify() pass.
 */
struct ResidencyPlanner {
  /** the pool's allocator produces device-visible memory at all */
  bool device_backed = false;
  /** the allocator can additionally back a device-resident pool */
  bool device_pool = false;
  /** input-boundary raise patterns (application policy) */
  const char *raise = nullptr;
  /** output-boundary lower patterns (application policy) */
  const char *lower = nullptr;
  /** patterns kept off the device plane (application policy) */
  const char *exclude = nullptr;

  /**
   * @brief Classify one tensor's static residency class.
   * @param engine compute engine of the producing / requesting layer
   * @param all_consumers_device every view consumer is on the same engine
   * @param is_fp16 tensor data type is FP16
   * @param name tensor name, matched against the declared boundaries
   * @return the static ResidencyClass
   */
  ResidencyClass classify(ml::train::LayerComputeEngine engine,
                          bool all_consumers_device, bool is_fp16,
                          const std::string &name) const {
    /** A host-only allocator has one plane and nothing to decide. */
    if (!device_backed)
      return ResidencyClass::HOST;

    /** The heuristic: a tensor is device-resident when the layer that writes
     *  it and every layer that reads it run on the device, and it is in the
     *  data type those kernels compute in. Anything else keeps the shared
     *  plane, which both sides can address. */
    ResidencyClass cls = (engine == ml::train::LayerComputeEngine::GPU &&
                          all_consumers_device && is_fp16)
                           ? ResidencyClass::GPU_CLMEM
                           : ResidencyClass::SVM;

    /** Declared boundaries. A raise is a tensor a host producer uploads to the
     *  device plane itself; a lower is one whose single host consumer reads it
     *  back explicitly. Both are places where the copy IS the coherence point,
     *  so the engine test above may be bypassed -- but only in the direction
     *  the application declared. */
    if (cls == ResidencyClass::SVM && is_fp16) {
      const bool boundary_raise =
        all_consumers_device && nameMatchesAny(name, raise);
      const bool boundary_lower =
        engine == ml::train::LayerComputeEngine::GPU &&
        nameMatchesAny(name, lower);
      if (boundary_raise || boundary_lower)
        cls = ResidencyClass::GPU_CLMEM;
    }

    /** Downgrades: the pool has no device plane to place it in, or the
     *  application excluded the tensor by name. */
    if (cls == ResidencyClass::GPU_CLMEM &&
        (!device_pool || nameMatchesAny(name, exclude)))
      cls = ResidencyClass::SVM;

    return cls;
  }

private:
  /** comma-separated substring match against a declared pattern list. */
  static bool nameMatchesAny(const std::string &name, const char *list) {
    if (list == nullptr)
      return false;
    const std::string s(list);
    size_t pos = 0;
    while (pos <= s.size()) {
      const size_t comma = s.find(',', pos);
      const std::string tok = s.substr(
        pos, comma == std::string::npos ? std::string::npos : comma - pos);
      if (!tok.empty() && name.find(tok) != std::string::npos)
        return true;
      if (comma == std::string::npos)
        break;
      pos = comma + 1;
    }
    return false;
  }
};

} // namespace nntrainer

#endif // __RESIDENCY_PLANNER_H__

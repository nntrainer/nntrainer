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
 * whether the tensor declares an Initializer that writes host bytes, and what
 * the pool's allocator can actually back.
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
  /** a fan-out tensor stays eligible for the device plane (NNTR_CLMEM_FANOUT=0
   *  clears it). True by default, so the classification is unchanged; clearing
   *  it is the A/B lever for the case where a tensor read through more than
   *  one view chain corrupts on the device plane while every single-consumer
   *  tensor stays bit-identical. */
  bool allow_fanout = true;

  /**
   * @brief Classify one tensor's static residency class.
   * @param engine compute engine of the producing / requesting layer
   * @param all_consumers_device every view consumer is on the same engine
   * @param is_fp16 tensor data type is FP16
   * @param needs_host_init the tensor declares an Initializer, so its bytes
   *        are written on the host plane before the first kernel runs
   * @param name tensor name, matched against the declared boundaries
   * @param view_count number of views registered on this tensor; read only
   *        when allow_fanout is cleared. Defaults to 0 so a caller that does
   *        not track fan-out gets the unrestricted classification.
   * @return the static ResidencyClass
   */
  ResidencyClass classify(ml::train::LayerComputeEngine engine,
                          bool all_consumers_device, bool is_fp16,
                          bool needs_host_init, const std::string &name,
                          unsigned int view_count = 0) const {
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

    /** A declared Initializer and the device plane are incompatible here, so
     *  the combination is refused rather than half honoured. The initializer
     *  writes the host side of the allocation; nothing in this plane uploads
     *  those bytes -- core deliberately owns no upload path, because a raise
     *  boundary makes the copy the application's, at the point where the two
     *  planes are declared to agree. Placing such a tensor in device memory
     *  would leave the kernels reading a buffer that never saw the
     *  initialisation. The shared plane honours both: the bytes the
     *  initializer wrote ARE the bytes the device reads. */
    if (cls == ResidencyClass::GPU_CLMEM && needs_host_init)
      cls = ResidencyClass::SVM;

    /** Fan-out demotion (NNTR_CLMEM_FANOUT=0). A tensor read through more than
     *  one view chain -- the shape an auto-inserted multiout produces -- has
     *  been seen to corrupt on the device plane while every single-consumer
     *  tensor stayed bit-identical. This restricts the device placement to the
     *  single-consumer partition. The demotion is OFF by default, so the
     *  classification is unchanged unless the lever is set; it exists so a
     *  device run that disagrees with its own reference can bisect this
     *  without a rebuild. A declared boundary is exempt either way: there the
     *  copy IS the coherence point, so the hazard does not arise. */
    const bool at_boundary =
      nameMatchesAny(name, raise) || nameMatchesAny(name, lower);
    if (cls == ResidencyClass::GPU_CLMEM && !allow_fanout && !at_boundary &&
        view_count > 1)
      cls = ResidencyClass::SVM;

    return cls;
  }

private:
  /**
   * @brief comma-separated substring match against a declared pattern list.
   *
   * @details Substring, not glob and not regex: a pattern "fc" matches every
   * tensor whose name contains those two characters, "fc1" and "fc12" among
   * them. A comma always separates, so a pattern cannot contain one. That is
   * deliberately the least machinery that expresses a boundary, and it is the
   * matcher an application has to write its patterns against -- pick a prefix
   * the model's naming makes unambiguous ("cache_" rather than "c"). Empty
   * tokens are ignored, so a trailing or doubled comma is harmless.
   *
   * @param name tensor name to test
   * @param list comma-separated pattern list, or nullptr for "no patterns"
   * @return true if any non-empty pattern occurs in @a name
   */
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

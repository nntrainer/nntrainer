// SPDX-License-Identifier: Apache-2.0
/**
 * @file   residency_planner.h
 * @date   04 July 2026
 * @brief  Static residency-class planner
 *
 * @details Single-responsibility component that assigns a tensor its static
 * ResidencyClass. Previously this lived as an inline block inside
 * TensorPool::allocate(); lifting it into a declarative `classify()` pass makes
 * the residency policy a named, testable unit that is a pure function of the
 * tensor's (engine, consumers, dtype, role, name, fan-out) and the planner's
 * per-pool configuration -- with no hidden file-scope dependencies. It absorbs
 * the former deriveResidency() heuristic, the app-declared input/output
 * boundary raise/lower (ResidencyPolicy), and the device-measured cl_mem
 * exclusions (fan-out / KV / QKV / filter). Byte-identical to the former inline
 * logic. This is the seam a future partition-strategy planner (whole-graph vs
 * op-level offload) plugs into.
 *
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __RESIDENCY_PLANNER_H__
#define __RESIDENCY_PLANNER_H__

#include <string>

#include <common.h>            // ml::train::LayerComputeEngine
#include <memory_data.h>       // ResidencyClass
#include <tensor_wrap_specs.h> // TensorRole

namespace nntrainer {

/**
 * @brief Per-pool residency classification config + the classify() pass.
 */
struct ResidencyPlanner {
  bool gpu_svm_backend = false; /**< allocator can back a device cl_mem pool */
  bool clmem_eligible = false; /**< gpu_svm_backend AND the cl_mem pool is on */
  const char *clmem_filter =
    nullptr; /**< NNTR_CLMEM_CLASS_FILTER (keep-list) */
  const char *clmem_exclude =
    nullptr; /**< NNTR_CLMEM_CLASS_EXCLUDE (drop-list) */
  const char *clmem_raise =
    nullptr; /**< input-boundary raise patterns (policy) */
  const char *clmem_lower =
    nullptr; /**< output-boundary lower patterns (policy) */
  bool allow_fanout =
    true; /**< NNTR_CLMEM_FANOUT != 0 (keep fan-out on cl_mem) */
  bool qkv_off =
    false; /**< NNTR_CLMEM_QKV == 0 (force q/k/v projections SVM) */

  /**
   * @brief Classify one tensor's static residency class.
   * @param engine producing/requesting layer's compute engine
   * @param all_consumers_gpu every view consumer is on the GPU plane
   * @param is_fp16 tensor dtype is FP16 (FP32 paths stay host/NEON)
   * @param role semantic role hint (currently falls through)
   * @param name tensor name (for the app-declared / debug substring gates)
   * @param view_count number of view consumers (fan-out > 1)
   * @return the static ResidencyClass
   */
  ResidencyClass classify(ml::train::LayerComputeEngine engine,
                          bool all_consumers_gpu, bool is_fp16, TensorRole role,
                          const std::string &name,
                          unsigned int view_count) const {
    // Non-GPU backends (CPU build / CPU allocator) keep everything on the host.
    if (!gpu_svm_backend)
      return ResidencyClass::HOST;

    // engine/consumer/dtype heuristic. Every role currently falls through to it
    // (byte-identical); the switch marks where per-role residency crossovers
    // slot in once role-bearing layers tag their tensors (Mem M4).
    ResidencyClass cls = (engine == ml::train::LayerComputeEngine::GPU &&
                          all_consumers_gpu && is_fp16)
                           ? ResidencyClass::GPU_CLMEM
                           : ResidencyClass::SVM;
    switch (role) {
    case TensorRole::KV:   // -> IMAGE2D on the Adreno image path (M6 reserved)
    case TensorRole::QKV:  // q/k/v projection: device-resident for GPU attn
    case TensorRole::ACT:  // layer I/O: the heuristic GPU_CLMEM/SVM split
    case TensorRole::EMB:  // token embedding table
    case TensorRole::NORM: // norm gamma/output
    case TensorRole::WEIGHT: // model weight
    case TensorRole::GENERIC:
    default:
      break;
    }

    // Input-boundary raise: a host-produced tensor its producer uploads to the
    // cl_mem plane; the fan-out restriction below does NOT apply (the upload IS
    // the coherence point). Output-boundary lower: a GPU producer whose lone
    // host consumer reads it back once -- bypass the consumer-AND. Both use the
    // app-declared patterns (Mem M4).
    const bool boundary_raise = cls == ResidencyClass::SVM && gpu_svm_backend &&
                                is_fp16 && all_consumers_gpu &&
                                nameMatchesAny(name, clmem_raise);
    const bool boundary_lower = cls == ResidencyClass::SVM && gpu_svm_backend &&
                                is_fp16 &&
                                engine == ml::train::LayerComputeEngine::GPU &&
                                nameMatchesAny(name, clmem_lower);
    if (boundary_raise || boundary_lower)
      cls = ResidencyClass::GPU_CLMEM;

    // Device-measured downgrades GPU_CLMEM -> SVM:
    //  - the cl_mem pool is not eligible;
    //  - fan-out (view_count > 1) corrupts on cl_mem unless lifted / a
    //  boundary;
    //  - a debug filter/exclude substring gate;
    //  - the KV cache stays SVM by design (sequence-persistent, mha-consumed);
    //  - q/k/v projections stay SVM when qkv_off (a schedule-dependent driver
    //    divergence; the attention OUTPUT conversion is baseline-identical).
    if (cls == ResidencyClass::GPU_CLMEM &&
        (!clmem_eligible ||
         (!allow_fanout && !boundary_raise && !boundary_lower &&
          view_count > 1) ||
         (clmem_filter != nullptr && !nameMatchesAny(name, clmem_filter)) ||
         nameMatchesAny(name, clmem_exclude) ||
         nameMatchesAny(name, "cache_") ||
         (qkv_off && (nameMatchesAny(name, "_wq:out0,_wk:out0") ||
                      nameMatchesAny(name, "_wv:out0")))))
      cls = ResidencyClass::SVM;

    return cls;
  }

private:
  /** comma-separated any-substring match (bisect / boundary filters). */
  static bool nameMatchesAny(const std::string &name, const char *list) {
    if (list == nullptr)
      return false;
    const std::string s(list);
    size_t pos = 0;
    while (pos <= s.size()) {
      size_t comma = s.find(',', pos);
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

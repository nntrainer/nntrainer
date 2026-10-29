// SPDX-License-Identifier: Apache-2.0
/**
 * @file runtime_config.h
 * @date 29 October 2026
 * @brief CausalLM runtime selection and CPU attention admission policy
 * @author Codex (OpenAI)
 * @bug Runtime snapshots must be refreshed when a model is re-finalized.
 */
#ifndef __CAUSALLM_RUNTIME_CONFIG_H__
#define __CAUSALLM_RUNTIME_CONFIG_H__

#include <engine.h>
#include <env_compat.h>
#include <string>

namespace causallm {

/** Resolve the live backend using the same registered-context fallback as
 * Engine::parseComputeEngine. Callers pass a lowercased requested name. */
inline std::string resolveRuntimeEngine(const std::string &requested,
                                        bool requested_registered,
                                        bool gpu_registered,
                                        bool cuda_registered) {
  if (!requested.empty())
    return requested_registered ? requested : "cpu";
  if (gpu_registered)
    return "gpu";
  if (cuda_registered)
    return "cuda";
  return "cpu";
}

/** Resolve at model construction/allocation/finalize, never latch test env
 * changes process-wide. Keep the returned snapshot for hot execution paths. */
inline std::string runtimeEngine() {
  const auto requested = nntr_engine_env();
  const auto &engine = nntrainer::Engine::Global();
  return resolveRuntimeEngine(
    requested, !requested.empty() && engine.isContextRegistered(requested),
    engine.isContextRegistered("gpu"), engine.isContextRegistered("cuda"));
}

struct CpuAttentionProfile {
  bool cpu = true;
  bool inference = true;
  unsigned int batch = 1;
  bool external_cache = true;
  bool sink = false;
  bool int8_cache = false;
  bool ring = false;
  bool ohwi = false;
  bool rope = true;
  bool fp32_io = true;
  bool half_cache = true;
};

/** Platform-independent policy, shared by Android finalize and native CI. */
inline bool supportsCpuAttentionFp32(const CpuAttentionProfile &p) {
  return p.cpu && p.inference && p.batch == 1 && p.external_cache && !p.sink &&
         !p.int8_cache && !p.ring && !p.ohwi && p.rope && p.fp32_io &&
         p.half_cache;
}

struct RuntimeConfig {
  std::string engine = "cpu";
  bool cpu = true;
  bool cpu_trace = false;
  bool cpu_attention_fp32 = false;
  bool cpu_kv_tiled = false;
  bool cpu_kv_legacy = false;
  bool device_attention = false;

  /** Capture per owning model/layer. Mutating the environment later affects
   * new objects, while existing objects keep their execution configuration. */
  static RuntimeConfig capture() { return captureForEngine(runtimeEngine()); }

  /** Capture flags for an already resolved engine. This pure backend-policy
   * boundary lets native CI verify options without initializing device/plugin
   * registries; production capture() supplies the live resolution above. */
  static RuntimeConfig captureForEngine(const std::string &selected_engine) {
    RuntimeConfig result;
    result.engine = selected_engine;
    result.cpu = result.engine == "cpu";
    result.cpu_trace = result.cpu && nntr_env_on("NNTR_KV_CPU_TRACE");
    result.cpu_attention_fp32 = nntr_env_on("NNTR_CPU_ATTENTION_FP32");
    result.cpu_kv_tiled = result.cpu && nntr_env_on("NNTR_CPU_KV_TILED");
    result.cpu_kv_legacy = result.cpu && nntr_env_on("NNTR_CPU_KV_LEGACY");
    result.device_attention =
      result.engine == "cuda"
        ? nntr_env_on("NNTR_CUDA_ATTN")
        : (result.engine == "gpu" && nntr_env_on("NNTR_MHA_GPU"));
    return result;
  }
};
} // namespace causallm
#endif // __CAUSALLM_RUNTIME_CONFIG_H__

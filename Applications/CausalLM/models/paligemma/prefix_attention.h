// SPDX-License-Identifier: Apache-2.0
/** @file prefix_attention.h @brief Noncausal attention without cache
 * quantization. */
#ifndef NNTR_PREFIX_ATTENTION_H
#define NNTR_PREFIX_ATTENTION_H
#include <cstdlib>
#include <layer.h>
#include <memory>
#include <string>
#include <vector>
namespace causallm {
/** Keep graph host boundaries coherent while retaining device compute. */
inline void configurePaliGemmaCuda() {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  const auto setDefault = [](const char *name, const char *value) {
    if (std::getenv(name))
      return;
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
  };
  setDefault("NNTR_CUDA_DEV_ACT", "0");
  setDefault("NNTR_CUDA_ASYNC", "0");
  setDefault("NNTR_CUDA_KV_UVM", "1");
  setDefault("NNTR_CUDA_ELTWISE", "1");
  setDefault("NNTR_CUDA_ROPE", "1");
  setDefault("NNTR_CUDA_PREFETCH_WEIGHTS", "1");
  // Match highest-precision FP32 reference GEMMs, including conditioning.
  setDefault("NNTR_CUBLAS_PEDANTIC", "1");
  setDefault("NNTR_CUDA_VCOPY_PREFILL", "1");
  // Only certified FP16 embedding-input graphs expose replay feed nodes.
  setDefault("NNTR_CUDA_DECODE_GRAPH", "1");
#endif
}

/** Post-RoPE prefix K/V in row-major [rows, kv_heads * head_dim]. */
struct PrefixKV {
  std::vector<float> key, value;
};
/** Construct noncausal attention and optionally retain its FP32 prefix K/V. */
std::shared_ptr<ml::train::Layer>
createPrefixAttention(const std::string &name, unsigned int heads,
                      unsigned int kv_heads, unsigned int head_dim,
                      float rope_theta, const std::shared_ptr<PrefixKV> &cache);
} // namespace causallm
#endif

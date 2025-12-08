// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   nntr_qwen3_moe_causallm.h
 * @date   15 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __NNTR_QWEN_MOE_CAUSAL_LM_H__
#define __NNTR_QWEN_MOE_CAUSAL_LM_H__

#include <causal_lm.h>
#include <nntr_qwen3_causallm.h>

namespace causallm {

/**
 * @brief Qwen3MoECausalLM class
 * @note  This class inherits Qwewn3CaUSALlm
 */
class NNTRQwen3MoECausalLM : public NNTRQwen3CausalLM {

public:
  static constexpr const char *architectures = "NNTRQwen3MoeCausalLM";

  NNTRQwen3MoECausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    NNTRQwen3CausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~NNTRQwen3MoECausalLM() = default;

  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void registerCustomLayers() override;

private:
  unsigned int NUM_EXPERTS;
  unsigned int NUM_EXPERTS_PER_TOK;
};
}; // namespace causallm

#endif /* __NNTR_QWEN_MOE_CAUSAL_LM_H__ */

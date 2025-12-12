// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   qwen3_moe_causallm.h
 * @date   15 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __QWEN_MOE_CAUSAL_LM_H__
#define __QWEN_MOE_CAUSAL_LM_H__

#include <causal_lm.h>
#include <qwen3_causallm.h>

namespace causallm {

/**
 * @brief Qwen3MoECausalLM class
 * @note  This class inherits Qwewn3CaUSALlm
 */
class Qwen3MoECausalLM : public Qwen3CausalLM {

public:
  static constexpr const char *architectures = "Qwen3MoeForCausalLM";

  Qwen3MoECausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Qwen3CausalLM(cfg, generation_cfg, nntr_cfg) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Qwen3MoECausalLM() = default;

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

#endif /* __QWEN_MOE_CAUSAL_LM_H__ */

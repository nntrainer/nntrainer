// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   qwen3_causallm.h
 * @date   10 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Please refer to the following code :
 *  https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/qwen3/modeling_qwen3.py
 */

#ifndef __QWEN_CAUSAL_LM_H__
#define __QWEN_CAUSAL_LM_H__ __QWEN_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Qwen3Transformer class
 */
class Qwen3Transformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "Qwen3Transformer";

  Qwen3Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen3Transformer() = default;

  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  void registerCustomLayers() override;
};

/**
 * @brief Qwen3CausalLM class
 */
class Qwen3CausalLM : public CausalLM, public Qwen3Transformer {

public:
  static constexpr const char *architectures = "Qwen3ForCausalLM";

  Qwen3CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    Qwen3Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen3CausalLM() = default;

  void registerCustomLayers() override;

private:
};
} // namespace causallm

#endif /* __QWEN3_CAUSAL_LM_H__ */

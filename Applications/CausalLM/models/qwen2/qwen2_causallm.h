// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   qwen2_causallm.h
 * @date   6 January 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Please refer to the following code :
 *  https://github.com/huggingface/transformers/blob/v4.37.0/src/transformers/models/qwen2/modeling_qwen2.py
 */

#ifndef __QWEN2_CAUSAL_LM_H__
#define __QWEN2_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Qwen2Transformer class
 */
class Qwen2Transformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "Qwen2Transformer";

  Qwen2Transformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen2Transformer() = default;

  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;
};

/**
 * @brief Qwen2CausalLM class
 */
class Qwen2CausalLM : public CausalLM, public Qwen2Transformer {

public:
  static constexpr const char *architectures = "Qwen2CausalLM";

  Qwen2CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    Qwen2Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen2CausalLM() = default;
};
} // namespace causallm

#endif /* __QWEN2_CAUSAL_LM_H__*/

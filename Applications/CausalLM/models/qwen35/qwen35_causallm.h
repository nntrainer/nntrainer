// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen35_causallm.h
 * @brief  Qwen3.5 text-only causal language model.
 */

#ifndef __QWEN35_CAUSAL_LM_H__
#define __QWEN35_CAUSAL_LM_H__

#include <causal_lm.h>

namespace causallm {

class WIN_EXPORT Qwen35CausalLM : public CausalLM {
public:
  static constexpr const char *architectures = "Qwen3_5ForConditionalGeneration";

  Qwen35CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);
  virtual ~Qwen35CausalLM() = default;

  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id,
                                std::string input_name) override;

  void registerCustomLayers() override;

private:
  void setupQwen35Parameters(json &cfg);
  std::vector<LayerHandle> createQwen35FullAttention(const int layer_id,
                                                     std::string input_name);
  std::vector<LayerHandle> createQwen35LinearAttention(const int layer_id,
                                                       std::string input_name);

  std::vector<std::string> layer_types_;
  unsigned int linear_num_key_heads = 16;
  unsigned int linear_num_value_heads = 16;
  unsigned int linear_key_head_dim = 128;
  unsigned int linear_value_head_dim = 128;
  unsigned int linear_conv_kernel_dim = 4;
  float partial_rotary_factor = 0.25f;
};

} // namespace causallm

#endif /* __QWEN35_CAUSAL_LM_H__ */

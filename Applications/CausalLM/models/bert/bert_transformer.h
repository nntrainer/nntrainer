// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   bert_transformer.h
 * @date   29 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   Please refer to the following code :
 *  https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/bert/modeling_bert.py
 */

#ifndef __BERT_TRANSFORMER_H__
#define __BERT_TRANSFORMER_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief BertTransformer class
 * @note  Base class for BERT-style encoder-only models.
 *        The structure is :
 *
 *          [Input] [PositionIds] [TokenTypeIds]
 *             |         |              |
 *        [WordEmb]  [PosEmb]     [TokenTypeEmb]
 *             \_________|_____________/
 *                       |
 *                  [LayerNorm]
 *                       |
 *             [Encoder Block] (repeated N times)
 *                       |
 *                   [Output]
 *
 *        Each encoder block uses post-norm :
 *          x = LayerNorm(x + SelfAttention(x))
 *          x = LayerNorm(x + FFN(x))
 */
class BertTransformer : virtual public Transformer {

public:
  static constexpr const char *architectures = "BertTransformer";

  BertTransformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::EMBEDDING) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~BertTransformer() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void initialize() override;

  std::pair<Tensor, Tensor> constructModel() override;

  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                   Tensor input) override;

  void registerCustomLayers() override;

protected:
  /**
   * @brief Sanitize config to fill defaults that are not present in
   * typical BERT-style config.json files (e.g. rope_theta, rms_norm_eps).
   */
  static json &sanitizeConfig(json &cfg);

  /**
   * @brief Construct the BERT graph with three symbolic inputs.
   */
  std::pair<std::vector<Tensor>, Tensor> constructBertGraph();

  /**
   * @brief Type-vocab size for token_type_ids (BERT default: 2)
   */
  unsigned int TYPE_VOCAB_SIZE = 2;

public:
  /**
   * @brief Encode the prompt and return the embedding output
   */
  virtual std::vector<float *> encode(const WSTR prompt,
                                      const WSTR system_prompt = "",
                                      const WSTR tail_prompt = "") = 0;

  /**
   * @brief run the BertTransformer model
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prmopt = "",
           bool log_output = true) override;
};

} // namespace causallm

#endif /* __BERT_TRANSFORMER_H__ */

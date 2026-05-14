// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   multilingual_tinybert_16mb.h
 * @date   21 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This multilingual_tinybert_16mb.h constructs a class for
 *         a BERT-based encoder-only embedding model
 *         (multilingual-TinyBERT-16MB) built on top of the causallm
 *         Transformer base class.
 * @note   Please refer to the following code :
 *  https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/bert/modeling_bert.py
 */

#ifndef __MULTILINGUAL_TINYBERT_16MB_H__
#define __MULTILINGUAL_TINYBERT_16MB_H__

#include <bert_transformer.h>
#include <sentence_transformer.h>

namespace causallm {

/**
 * @brief MultilingualTinyBert class
 * @note  Concrete runnable model for multilingual-TinyBERT-16MB.
 *        It inherits BertTransformer and provides the encode / run
 *        methods that feed three inputs (input_ids, position_ids,
 *        token_type_ids) into the underlying nntrainer model.
 */
class MultilingualTinyBert : public BertTransformer {

public:
  static constexpr const char *architectures = "MultilingualTinyBert";

  MultilingualTinyBert(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::EMBEDDING),
    BertTransformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~MultilingualTinyBert() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override {
    BertTransformer::setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  std::pair<Tensor, Tensor> constructModel() override {
    return BertTransformer::constructModel();
  }

  void registerCustomLayers() override {
    BertTransformer::registerCustomLayers();
  }

  /**
   * @brief Encode the prompt and return the embedding output
   */
  std::vector<float *> encode(const WSTR prompt, const WSTR system_prompt = "",
                              const WSTR tail_prompt = "");
};

} // namespace causallm

#endif /* __MULTILINGUAL_TINYBERT_16MB_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file    sentence_transformer.h
 * @date    02 Jan 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 * @note    This embedding.h constructs a class for SentenceTransformer model
 * which can be a parent of models with embedding (encoder) structure.
 */

#ifndef __SENTENCE_TRANSFORMER_H__
#define __SENTENCE_TRANSFORMER_H__

#pragma once

#include <map>
#include <transformer.h>

namespace causallm {

/**
 * @brief SentenceTransformer Class
 */
WIN_EXPORT class SentenceTransformer : virtual public Transformer {

public:
  /**
   * @brief Construct a new SentenceTransformer object
   * @param cfg Configuration for the model (config.json)
   * @param generation_cfg Configuration for the generation (generation.json)
   * @param nntr_cfg Configuration for nntrainer (nntr_config.json)
   */
  SentenceTransformer(json &cfg, json &generation_cfg, json &nntr_cfg);

  /**
   * @brief Destroy the SentenceTransformer object
   */
  virtual ~SentenceTransformer() {}

  /**
   * @brief run the SentenceTransformer model
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prmopt = "",
           bool log_output = true) override;

  /**
   * @brief Encode the prompt and return the embedding
   * @param prompt User prompt
   * @param system_prompt System prompt
   * @param tail_prompt Tail prompt
   * @return SentenceTransformer output from the model
   */
  std::vector<float *> encode(const WSTR prompt, const WSTR system_prompt = "",
                              const WSTR tail_prompt = "");

protected:
  /**
   * @brief Setup the parameters for the SentenceTransformer model
   */
  void setupParameters(json &cfg, json &generation_dfg,
                       json &nntr_cfg) override;

  /**
   * @brief Construct Model
   */
  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief Map of module type suffix to layer type name
   * @note This map is used to dynamically resolve the nntrainer layer type from
   * the module configuration type suffix.
   * Key: Suffix of the module type (e.g., "Pooling")
   * Value: Registered layer name in nntrainer (e.g., "embedding_pooling")
   * @note All layers in this map correspond to operations defined in
   * sentence_transformers/models/ and are prefixed with "embedding_" in
   * nntrainer to distinguish with the general layers.
   */
  static std::map<std::string, std::string> layer_map;

  /**
   * @brief Add Module Layer onto the symbolic graph
   * @param type module type string (e.g.,
   *             "sentence_transformers.models.Pooling")
   * @param idx  module index used to look up its config
   * @param input tensor that feeds the new module
   * @return the module's output tensor (or @p input unchanged if the type
   *         cannot be mapped, so the chain continues)
   */
  Tensor addModule(const std::string &type, int idx, Tensor input);

  /**
   * @brief register CustomLayers
   */
  void registerCustomLayers() override;

private:
  /**
   * @brief Module metadata list (from modules.json)
   */
  std::vector<json> modules;

  /**
   * @brief Module property configurations (from Module_name/config.json)
   */
  std::map<int, json> module_configs;

  /**
   * @brief Get the last component of the module type string
   * @param type Full type string (e.g., "sentence_transformers.models.Pooling")
   * @return Last component (e.g., "Pooling")
   */
  std::string getLastComponent(const std::string &type);
};

} // namespace causallm

#endif

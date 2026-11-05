// SPDX-License-Identifier: Apache-2.0
/** @file htp_cache_inputs.h
 * @brief Symbolic external KV inputs used by HTP attention models.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef CAUSALLM_HTP_CACHE_INPUTS_H
#define CAUSALLM_HTP_CACHE_INPUTS_H
#include <llm_util.hpp>
#include <transformer.h>
namespace causallm {
inline LayerHandle htpAttentionCore(const std::vector<std::string> &properties,
                                    const std::string &engine,
                                    const std::string &kv_dtype) {
  auto props = properties;
  if (!engine.empty())
    props.push_back(withKey("engine", engine));
  if (!kv_dtype.empty())
    props.push_back(withKey("kv_cache_quant", kv_dtype));
  return LayerHandle(ml::train::createLayer("htp_mha_core", props));
}
inline std::pair<Tensor, Tensor> htpCacheInputs(int layer_id, unsigned batch,
                                                unsigned rows, unsigned width) {
#ifdef ENABLE_FP16
  ml::train::TensorDim cache_dim(
    {batch, 1, rows, width},
    {ml::train::TensorDim::Format::NCHW, ml::train::TensorDim::DataType::FP16});

  Tensor cache_k(cache_dim, "cache_k_l" + std::to_string(layer_id));
  Tensor cache_v(cache_dim, "cache_v_l" + std::to_string(layer_id));
  return {cache_k, cache_v};
#else
  const std::string cache_shape = std::to_string(batch) +
                                  ":1:" + std::to_string(rows) + ":" +
                                  std::to_string(width);

  LayerHandle cache_k_input(ml::train::createLayer(
    "input",
    {withKey("name", "cache_k_l" + std::to_string(layer_id)),
     withKey("input_shape", cache_shape), withKey("input_dtype", "UINT16")}));
  LayerHandle cache_v_input(ml::train::createLayer(
    "input",
    {withKey("name", "cache_v_l" + std::to_string(layer_id)),
     withKey("input_shape", cache_shape), withKey("input_dtype", "UINT16")}));

  return {cache_k_input(Tensor()), cache_v_input(Tensor())};
#endif
}
} // namespace causallm
#endif

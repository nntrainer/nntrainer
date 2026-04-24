// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@@samsung.com>
 *
 * @file   tensor_api_impl.h
 * @date   11 December 2023
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Private implementation types shared across tensor_api_*.cpp
 *
 * @note This header is NOT installed. It is consumed only by the translation
 *       units that split the Tensor API implementation.
 */

#ifndef __ML_TRAIN_TENSOR_API_IMPL_H__
#define __ML_TRAIN_TENSOR_API_IMPL_H__

#include <tensor_api.h>

#include <tensor.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ml {
namespace train {

/**
 * @brief Lightweight graph node for API-level symbolic graph construction.
 *
 * Unlike the internal GraphNode (used for execution with LayerNode),
 * this only holds connection info needed to build the graph before
 * model.compile(). Stored as shared_ptr so Tensor copies share graph
 * structure without recursive deep copies (O(N!) memory explosion).
 */
struct SymbolicGraphNode {
  std::shared_ptr<Layer> producing_layer;
  std::vector<std::shared_ptr<SymbolicGraphNode>> inputs;
  TensorDim dim;
  std::string name;
  int output_index = -1; ///< >=0 means indexed output, e.g. split(0)
};

/**
 * @brief Internal implementation of Tensor
 */
struct Tensor::Impl {
  TensorDim dim;
  std::string name;
  bool valid = false;
  bool external = false;

  std::shared_ptr<nntrainer::Tensor> eager_data;
  void *external_ptr = nullptr;

  std::shared_ptr<Layer> src_layer;

  // Graph edge (shared_ptr to avoid O(N!) deep-copy on Tensor copy)
  std::shared_ptr<SymbolicGraphNode> graph_edge;

  // Bound internal tensor (set after model compile+initialize)
  nntrainer::Tensor *bound_tensor = nullptr;

  // Lazy operation chain
  std::vector<std::function<void(nntrainer::Tensor &)>> call_chain;

  Impl() = default;

  Impl(const TensorDim &d, const std::string &n) :
    dim(d), name(n), valid(true) {}
};

/**
 * @brief Cast a void pointer from Tensor::getInternalPtr() to the
 *        internal nntrainer::Tensor pointer.
 */
inline nntrainer::Tensor *asInternal(void *ptr) {
  return static_cast<nntrainer::Tensor *>(ptr);
}

} // namespace train
} // namespace ml

#endif // __ML_TRAIN_TENSOR_API_IMPL_H__
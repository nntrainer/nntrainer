// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Yeonjae Kim <duswo1120@snu.ac.kr>
 * Copyright (C) 2025 Hoyeon Jo <jhy213@snu.ac.kr>
 *
 * @file gradient_checkpointing_test_util.h
 * @date 16 December 2025
 * @brief NNTrainer gradient checkpointing related common functions
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Yeonjae Kim <duswo1120@snu.ac.kr>
 * @author Hoyeon Jo <jhy213@snu.ac.kr>
 * @bug No known bugs except for NYI items
 */
#ifndef __GRADIENT_CHECKPOINTING_TEST_UTIL_H__
#define __GRADIENT_CHECKPOINTING_TEST_UTIL_H__

#include <map>
#include <string>
#include <vector>

#include <tensor.h>

/**
 * @class GradientCheckpointingVerifier
 * @brief Helper to snapshot forward tensors and verify recomputed tensors
 *        during gradient checkpointing tests.
 */
class GradientCheckpointingVerifier {
public:
  /**
   * @brief     Constructor of GradientCheckpointingVerifier Class
   */
  GradientCheckpointingVerifier() = default;

  /**
   * @brief Save inputs in forwarding
   */
  void saveForwardInputs(const std::shared_ptr<nntrainer::LayerNode> &lnode,
                         const std::vector<nntrainer::Tensor> &inputs);

  /**
   * @brief Save outputs in forwarding
   */
  void saveForwardOutputs(const std::shared_ptr<nntrainer::LayerNode> &lnode,
                          const std::vector<nntrainer::Tensor> &outputs);

  /**
   * @brief Save weights in forwarding
   */
  void saveForwardWeights(const std::shared_ptr<nntrainer::LayerNode> &lnode,
                          const std::vector<nntrainer::Tensor> &weights);

  /**
   * @brief Save tensors in forwarding
   */
  void saveForwardTensors(const std::shared_ptr<nntrainer::LayerNode> &lnode,
                          const std::vector<nntrainer::Tensor> &tensors);

  /**
   * @brief Compare inputs in forwarding and recomputation
   */
  void
  verifyRecomputeInputs(const std::shared_ptr<nntrainer::LayerNode> &lnode,
                        const std::vector<nntrainer::Tensor> &recompute_inputs);

  /**
   * @brief Compare outputs in forwarding and recomputation
   */
  void verifyRecomputeOutputs(
    const std::shared_ptr<nntrainer::LayerNode> &lnode,
    const std::vector<nntrainer::Tensor> &recompute_outputs);

  /**
   * @brief Compare weights in forwarding and recomputation
   */
  void verifyRecomputeWeights(
    const std::shared_ptr<nntrainer::LayerNode> &lnode,
    const std::vector<nntrainer::Tensor> &recompute_weights);

  /**
   * @brief Compare tensors in forwarding and recomputation
   */
  void verifyRecomputeTensors(
    const std::shared_ptr<nntrainer::LayerNode> &lnode,
    const std::vector<nntrainer::Tensor> &recompute_tensors);

private:
  std::map<std::string, std::vector<nntrainer::Tensor>>
    saved_forward_inputs; /** saved input tensors in forwarding */
  std::map<std::string, std::vector<nntrainer::Tensor>>
    saved_forward_outputs; /** saved output tensors in forwarding */
  std::map<std::string, std::vector<nntrainer::Tensor>>
    saved_forward_weights; /** saved weight tensors in forwarding */
  std::map<std::string, std::vector<nntrainer::Tensor>>
    saved_forward_tensors; /** saved tensors in forwarding */

  /**
   * @brief prototypical version of checking tensor is equal
   * @param lhs forward tensor to be compared
   * @param rhs recomute tensor to be compared
   * @retval true tensor is equal
   * @retval false tensor is not equal
   */
  void tensorEqual(const nntrainer::Tensor &lhs, const nntrainer::Tensor &rhs);
};

#endif // __GRADIENT_CHECKPOINTING_TEST_UTIL_H__

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

class GradientCheckpointingVerifier {
public:
  GradientCheckpointingVerifier() = default;

  void saveForwardInputs(const std::shared_ptr<nntrainer::LayerNode> &lnode,
                         const std::vector<nntrainer::Tensor> &inputs);

  void saveForwardOutputs(const std::shared_ptr<nntrainer::LayerNode> &lnode,
                          const std::vector<nntrainer::Tensor> &outputs);

  void saveForwardWeights(const std::shared_ptr<nntrainer::LayerNode> &lnode,
                          const std::vector<nntrainer::Tensor> &weights);

  void saveForwardTensors(const std::shared_ptr<nntrainer::LayerNode> &lnode,
                          const std::vector<nntrainer::Tensor> &tensors);

  void
  verifyRecomputeInputs(const std::shared_ptr<nntrainer::LayerNode> &lnode,
                        const std::vector<nntrainer::Tensor> &recompute_inputs);

  void verifyRecomputeOutputs(
    const std::shared_ptr<nntrainer::LayerNode> &lnode,
    const std::vector<nntrainer::Tensor> &recompute_outputs);

  void verifyRecomputeWeights(
    const std::shared_ptr<nntrainer::LayerNode> &lnode,
    const std::vector<nntrainer::Tensor> &recompute_weights);

  void verifyRecomputeTensors(
    const std::shared_ptr<nntrainer::LayerNode> &lnode,
    const std::vector<nntrainer::Tensor> &recompute_tensors);

private:
  std::map<std::string, std::vector<nntrainer::Tensor>> saved_forward_inputs;
  std::map<std::string, std::vector<nntrainer::Tensor>> saved_forward_outputs;
  std::map<std::string, std::vector<nntrainer::Tensor>> saved_forward_weights;
  std::map<std::string, std::vector<nntrainer::Tensor>> saved_forward_tensors;

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

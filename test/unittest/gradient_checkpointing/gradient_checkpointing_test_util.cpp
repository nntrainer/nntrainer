// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Yeonjae Kim <duswo1120@snu.ac.kr>
 * Copyright (C) 2025 Hoyeon Jo <jhy213@snu.ac.kr>
 *
 * @file gradient_checkpointing_test_util.cpp
 * @date 16 December 2025
 * @brief NNTrainer gradient checkpointing related common functions
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Yeonjae Kim <duswo1120@snu.ac.kr>
 * @author Hoyeon Jo <jhy213@snu.ac.kr>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <layer_node.h>
#include <layer_normalization_layer.h>

#include <gradient_checkpointing_test_util.h>

void GradientCheckpointingVerifier::saveForwardInputs(
  const std::shared_ptr<nntrainer::LayerNode> &lnode,
  const std::vector<nntrainer::Tensor> &inputs) {
  std::vector<nntrainer::Tensor> saved_inputs;
  for (const auto &input : inputs)
    saved_inputs.push_back(input.clone());
  saved_forward_inputs[lnode->getName()] = saved_inputs;
}

void GradientCheckpointingVerifier::saveForwardOutputs(
  const std::shared_ptr<nntrainer::LayerNode> &lnode,
  const std::vector<nntrainer::Tensor> &outputs) {
  std::vector<nntrainer::Tensor> saved_outputs;
  for (const auto &output : outputs)
    saved_outputs.push_back(output.clone());
  saved_forward_outputs[lnode->getName()] = saved_outputs;
}

void GradientCheckpointingVerifier::saveForwardWeights(
  const std::shared_ptr<nntrainer::LayerNode> &lnode,
  const std::vector<nntrainer::Tensor> &weights) {
  std::vector<nntrainer::Tensor> saved_weights;
  for (const auto &weight : weights)
    saved_weights.push_back(weight.clone());
  saved_forward_weights[lnode->getName()] = saved_weights;
}

void GradientCheckpointingVerifier::saveForwardTensors(
  const std::shared_ptr<nntrainer::LayerNode> &lnode,
  const std::vector<nntrainer::Tensor> &tensors) {
  std::vector<nntrainer::Tensor> saved_tensors;
  for (const auto &tensor : tensors)
    saved_tensors.push_back(tensor.clone());
  saved_forward_tensors[lnode->getName()] = saved_tensors;
}

void GradientCheckpointingVerifier::verifyRecomputeInputs(
  const std::shared_ptr<nntrainer::LayerNode> &lnode,
  const std::vector<nntrainer::Tensor> &recompute_inputs) {
  EXPECT_NE(saved_forward_inputs.find(lnode->getName()),
            saved_forward_inputs.end());
  std::vector<nntrainer::Tensor> &forward_inputs =
    saved_forward_inputs.at(lnode->getName());
  EXPECT_EQ(forward_inputs.size(), recompute_inputs.size());
  for (int i = 0; i < forward_inputs.size(); i++)
    tensorEqual(forward_inputs[i], recompute_inputs[i]);
}

void GradientCheckpointingVerifier::verifyRecomputeOutputs(
  const std::shared_ptr<nntrainer::LayerNode> &lnode,
  const std::vector<nntrainer::Tensor> &recompute_outputs) {
  EXPECT_NE(saved_forward_outputs.find(lnode->getName()),
            saved_forward_outputs.end());
  std::vector<nntrainer::Tensor> &forward_outputs =
    saved_forward_outputs.at(lnode->getName());
  EXPECT_EQ(forward_outputs.size(), recompute_outputs.size());
  for (int i = 0; i < forward_outputs.size(); i++)
    tensorEqual(forward_outputs[i], recompute_outputs[i]);
}

void GradientCheckpointingVerifier::verifyRecomputeWeights(
  const std::shared_ptr<nntrainer::LayerNode> &lnode,
  const std::vector<nntrainer::Tensor> &recompute_weights) {
  EXPECT_NE(saved_forward_weights.find(lnode->getName()),
            saved_forward_weights.end());
  std::vector<nntrainer::Tensor> &forward_weights =
    saved_forward_weights.at(lnode->getName());
  EXPECT_EQ(forward_weights.size(), recompute_weights.size());
  for (int i = 0; i < forward_weights.size(); i++)
    tensorEqual(forward_weights[i], recompute_weights[i]);
}

void GradientCheckpointingVerifier::verifyRecomputeTensors(
  const std::shared_ptr<nntrainer::LayerNode> &lnode,
  const std::vector<nntrainer::Tensor> &recompute_tensors) {
  EXPECT_NE(saved_forward_tensors.find(lnode->getName()),
            saved_forward_tensors.end());
  std::vector<nntrainer::Tensor> &forward_tensors =
    saved_forward_tensors.at(lnode->getName());
  EXPECT_EQ(forward_tensors.size(), recompute_tensors.size());
  for (int i = 0; i < forward_tensors.size(); i++) {
    // Skip stateful tensors like KV cache that accumulate across iterations
    if (recompute_tensors[i].getName().find("cache_key") != std::string::npos ||
        recompute_tensors[i].getName().find("cache_value") !=
          std::string::npos ||
        (lnode->getType() == nntrainer::LayerNormalizationLayer::type &&
         i > 2)) {
      continue;
    }
    tensorEqual(forward_tensors[i], recompute_tensors[i]);
  }
}

void GradientCheckpointingVerifier::tensorEqual(const nntrainer::Tensor &lhs,
                                                const nntrainer::Tensor &rhs) {
  // Check dimensions
  EXPECT_EQ(lhs.getDim(), rhs.getDim());

  // Compute element-wise difference
  const float rel_tolerance = 1e-5f;
  const float abs_tolerance = 1e-6f;
  const float *lhs_data = lhs.getData();
  const float *rhs_data = rhs.getData();

  bool match = true;
  for (size_t i = 0; i < lhs.size(); i++) {
    auto diff = std::abs(lhs.getValue(i) - rhs.getValue(i));
    float threshold = abs_tolerance + rel_tolerance * std::abs(lhs.getValue(i));
    if (diff > threshold) {
      printf("diff : %lf, thresh : %lf\n", diff, threshold);
      match = false;
      break;
    }
  }
  if (!match) {
    printf("diff %s\n", rhs.getName().c_str());
  }
  EXPECT_TRUE(match);
}
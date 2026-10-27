// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   operation_layer.h
 * @date   4 Oct 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is common class for operation layers
 *
 */
#ifndef __LAYER_OPERATION_H__
#define __LAYER_OPERATION_H__
#ifdef __cplusplus

#include <fstream>

#include <layer_context.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @brief Base class for Unary Tensor Operation Layer
 *
 */
class UnaryOperationLayer : public Layer {
public:
  /**
   * @brief forwarding operation for unary input
   *
   */
  virtual void forwarding_operation(const Tensor &input, Tensor &hidden) = 0;

  /**
   * @brief copydoc Layer::forwarding(RunLayerContext &context, bool training)
   *
   */
  void forwarding(RunLayerContext &context, bool training) override {
    Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

    const Tensor input = context.getInput(0);
    forwarding_operation(input, hidden_);

    // Dump one named layer's real output (from inside the untruncated
    // production forward pass), for external bisection. Mirrors the
    // NNTR_DUMP_LAYER hooks in conv2d_layer.cpp/addition_layer.cpp/
    // concat_layer.cpp/etc; needed here (rather than in forwarding_operation)
    // since only this base-class forwarding() has the layer's name.
    if (const char *dl = std::getenv("NNTR_DUMP_LAYER");
        dl && context.getName() == dl) {
      const char *dp = std::getenv("NNTR_DUMP_PATH");
      if (dp) {
        std::vector<float> buf(hidden_.size());
        if (hidden_.getDataType() == ml::train::TensorDim::DataType::QINT8) {
          const int8_t *q = hidden_.getData<int8_t>();
          const float sc = hidden_.getScale<float>()[0];
          static const bool perch_asym_env =
            std::getenv("NNTR_W8A8_PERCH") != nullptr &&
            std::getenv("NNTR_W8A8_SYM") == nullptr;
          constexpr float kActOff = 0.27846455f;
          for (size_t i = 0; i < hidden_.size(); ++i)
            buf[i] = perch_asym_env ? ((float)q[i] + 128.f) * sc - kActOff
                                    : sc * (float)q[i];
        } else if (hidden_.getDataType() ==
                   ml::train::TensorDim::DataType::FP32) {
          const float *d = hidden_.getData<float>();
          std::copy(d, d + hidden_.size(), buf.begin());
        }
        std::ofstream of(dp, std::ios::binary);
        of.write(reinterpret_cast<const char *>(buf.data()),
                 buf.size() * sizeof(float));
      }
    }
  }

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   *
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override {
    if (from) {
      // Normalize to 0-based while preserving step size for multi-token prefill
      to = to - from;
      from = 0;
    }

    Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
    TensorDim hidden_dim = hidden_.getDim();
    TensorDim hidden_step_dim = hidden_dim;

    hidden_step_dim.batch(1);
    hidden_step_dim.height(to - from);

    const Tensor &input = context.getInput(0);
    TensorDim input_dim = input.getDim();
    TensorDim input_step_dim = input_dim;
    input_step_dim.batch(1);
    input_step_dim.height(to - from);

    for (unsigned int b = 0; b < hidden_.batch(); ++b) {
      Tensor hidden_step = hidden_.getSharedDataTensor(
        hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

      Tensor input_step = input.getSharedDataTensor(
        input_step_dim, b * input_dim.getFeatureLen(), true);

      forwarding_operation(input_step, hidden_step);
    }
  }

  static constexpr size_t SINGLE_INOUT_IDX = 0;
};

/**
 * @brief Base class for Binary Tensor Operation Layer
 *
 */
class BinaryOperationLayer : public Layer {
public:
  /**
   * @brief forwarding operation for binary inputs
   *
   */
  virtual void forwarding_operation(const Tensor &input0, const Tensor &input1,
                                    Tensor &hidden) = 0;

  /**
   * @brief copydoc Layer::forwarding(RunLayerContext &context, bool training)
   *
   */
  void forwarding(RunLayerContext &context, bool training) override {
    Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

    const Tensor &input0 = context.getInput(0);
    const Tensor &input1 = context.getInput(1);
    forwarding_operation(input0, input1, hidden_);
  }

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   *
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override {
    if (from) {
      // Normalize to 0-based while preserving step size for multi-token prefill
      to = to - from;
      from = 0;
    }

    Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
    TensorDim hidden_dim = hidden_.getDim();
    TensorDim hidden_step_dim = hidden_dim;

    hidden_step_dim.batch(1);
    hidden_step_dim.height(to - from);

    const Tensor &input0 = context.getInput(0);
    const Tensor &input1 = context.getInput(1);

    TensorDim input0_dim = input0.getDim();
    TensorDim input1_dim = input1.getDim();
    if (input0_dim != input1_dim) {
      throw std::invalid_argument(
        "If the two input dimensions are different, the incremental "
        "forwarding implementation must be overridden.");
    }

    TensorDim input_step_dim = input0_dim;
    input_step_dim.batch(1);
    input_step_dim.height(to - from);

    for (unsigned int b = 0; b < hidden_.batch(); ++b) {
      Tensor hidden_step = hidden_.getSharedDataTensor(
        hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

      Tensor input0_step = input0.getSharedDataTensor(
        input_step_dim, b * input0_dim.getFeatureLen(), true);

      Tensor input1_step = input1.getSharedDataTensor(
        input_step_dim, b * input1_dim.getFeatureLen(), true);

      forwarding_operation(input0_step, input1_step, hidden_step);
    }
  }

  static constexpr size_t SINGLE_INOUT_IDX = 0;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYER_OPERATION_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   custom_rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <iostream>

#include "reshaped_rms_norm.h"

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ReshapedRMSNormLayer::finalize(InitLayerContext &context) {
  std::vector<TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);

  unsigned int feature_size = std::get<props::FeatureSize>(rms_props).get();

  NNTR_THROW_IF(dim[0].width() % feature_size != 0, std::invalid_argument)
    << "[reshaped_rms_norm] feature_size must be a divisor of width. "
    << "width=" << dim[0].width() << ", feature_size=" << feature_size;

  TensorDim gamma_dim(
    1, 1, 1, feature_size,
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()));

  wt_idx[static_cast<size_t>(ReshapedRMSParams::gamma)] =
    context.requestWeight(gamma_dim, Initializer::ONES, WeightRegularizer::NONE,
                          1.0f, 0.0f, "gamma", true);
}

void ReshapedRMSNormLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  unsigned int height = in.getDim().height();
  incremental_forwarding(context, 0, height, training);
}

void ReshapedRMSNormLayer::incremental_forwarding(RunLayerContext &context,
                                                  unsigned int from,
                                                  unsigned int to,
                                                  bool training) {
  auto &epsilon = std::get<props::Epsilon>(rms_props).get();
  unsigned int feature_size = std::get<props::FeatureSize>(rms_props).get();

  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &gamma =
    context.getWeight(wt_idx[static_cast<size_t>(ReshapedRMSParams::gamma)]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  // set reshaped dim to (1, 1, -1, feature_size)
  ml::train::TensorDim step_reshaped_dim = in_step_dim;
  step_reshaped_dim.width(feature_size);
  step_reshaped_dim.height(in_step_dim.height() *
                           (in_dim.width() / feature_size));

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    in_step.reshape(step_reshaped_dim);
    out_step.reshape(step_reshaped_dim);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      // Compute RMS norm: out = x * rsqrt(mean(x^2) + epsilon) * gamma
      auto t = in_step.multiply(in_step).average(3).add(epsilon);
      t.inv_sqrt_i();
      in_step.multiply(t, out_step);
    } else {
      throw std::invalid_argument(
        "[reshaped_rms_norm] Error: not yet implemented for this data type");
    }
    out_step.multiply_i(gamma);

    out_step.reshape(out_step_dim);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }
}

void ReshapedRMSNormLayer::updateTensorsByInputDimensions(
  RunLayerContext &context, std::vector<TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void ReshapedRMSNormLayer::calcDerivative(RunLayerContext &context) {
  auto &epsilon = std::get<props::Epsilon>(rms_props).get();
  unsigned int feature_size = std::get<props::FeatureSize>(rms_props).get();

  const Tensor &incoming_deriv =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &outgoing_deriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &gamma =
    context.getWeight(wt_idx[static_cast<size_t>(ReshapedRMSParams::gamma)]);

  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    unsigned int batch = input.getDim().batch();
    unsigned int channel = input.getDim().channel();
    unsigned int height = input.getDim().height();
    unsigned int width = input.getDim().width();

    // Reshaped dimensions
    unsigned int reshaped_height = height * (width / feature_size);

    // Create inv_rms tensor on-the-fly with shape (batch, channel,
    // reshaped_height, 1)
    TensorDim inv_rms_dim(
      batch, channel, reshaped_height, 1,
      TensorDim::TensorType(input.getDim().getFormat(), input.getDataType()));
    Tensor inv_rms(inv_rms_dim, true);

    // Get raw pointers for efficient computation
    const float *in_data = input.getData<float>();
    const float *dy_data = incoming_deriv.getData<float>();
    float *dx_data = outgoing_deriv.getData<float>();
    const float *gamma_data = gamma.getData<float>();
    float *inv_rms_data = inv_rms.getData<float>();

    // First pass: compute inv_rms for each row
    // Each row in reshaped space has feature_size elements
    unsigned int total_reshaped_rows = batch * channel * reshaped_height;

    for (unsigned int row = 0; row < total_reshaped_rows; ++row) {
      // Map reshaped row to original tensor position
      unsigned int orig_batch = row / (channel * reshaped_height);
      unsigned int remainder = row % (channel * reshaped_height);
      unsigned int orig_channel = remainder / reshaped_height;
      unsigned int reshaped_h = remainder % reshaped_height;

      // Original position: compute which part of width this corresponds to
      unsigned int orig_height = reshaped_h / (width / feature_size);
      unsigned int width_group = reshaped_h % (width / feature_size);
      unsigned int orig_offset =
        orig_batch * channel * height * width + orig_channel * height * width +
        orig_height * width + width_group * feature_size;

      // Compute mean(x^2) for this row
      float sum_sq = 0.0f;
      for (unsigned int w = 0; w < feature_size; ++w) {
        float val = in_data[orig_offset + w];
        sum_sq += val * val;
      }
      float mean_sq = sum_sq / static_cast<float>(feature_size);
      inv_rms_data[row] = 1.0f / std::sqrt(mean_sq + epsilon);
    }

    // Second pass: compute derivative
    // dx = inv_rms * (gamma*dy - x * mean(gamma*dy*x) * inv_rms^2)
    for (unsigned int row = 0; row < total_reshaped_rows; ++row) {
      // Map reshaped row to original tensor position
      unsigned int orig_batch = row / (channel * reshaped_height);
      unsigned int remainder = row % (channel * reshaped_height);
      unsigned int orig_channel = remainder / reshaped_height;
      unsigned int reshaped_h = remainder % reshaped_height;

      unsigned int orig_height = reshaped_h / (width / feature_size);
      unsigned int width_group = reshaped_h % (width / feature_size);
      unsigned int orig_offset =
        orig_batch * channel * height * width + orig_channel * height * width +
        orig_height * width + width_group * feature_size;

      float inv_rms_val = inv_rms_data[row];
      float inv_rms_sq = inv_rms_val * inv_rms_val;

      // Compute c = mean(gamma * dy * x) over feature_size
      float c = 0.0f;
      for (unsigned int w = 0; w < feature_size; ++w) {
        c +=
          gamma_data[w] * dy_data[orig_offset + w] * in_data[orig_offset + w];
      }
      c /= static_cast<float>(feature_size);

      // Compute dx[w] = inv_rms * (gamma[w]*dy[w] - x[w] * c * inv_rms^2)
      for (unsigned int w = 0; w < feature_size; ++w) {
        dx_data[orig_offset + w] =
          inv_rms_val * (gamma_data[w] * dy_data[orig_offset + w] -
                         in_data[orig_offset + w] * c * inv_rms_sq);
      }
    }
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
    throw std::invalid_argument(
      "[reshaped_rms_norm] calcDerivative: FP16 is not implemented yet");
  }
}

void ReshapedRMSNormLayer::calcGradient(RunLayerContext &context) {
  auto &epsilon = std::get<props::Epsilon>(rms_props).get();
  unsigned int feature_size = std::get<props::FeatureSize>(rms_props).get();

  const Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &dy = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &dgamma = context.getWeightGrad(
    wt_idx[static_cast<size_t>(ReshapedRMSParams::gamma)]);

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    unsigned int batch = in.getDim().batch();
    unsigned int channel = in.getDim().channel();
    unsigned int height = in.getDim().height();
    unsigned int width = in.getDim().width();

    // Reshaped dimensions
    unsigned int reshaped_height = height * (width / feature_size);

    // Create inv_rms tensor on-the-fly
    TensorDim inv_rms_dim(
      batch, channel, reshaped_height, 1,
      TensorDim::TensorType(in.getDim().getFormat(), in.getDataType()));
    Tensor inv_rms(inv_rms_dim, true);

    const float *in_data = in.getData<float>();
    const float *dy_data = dy.getData<float>();
    float *dgamma_data = dgamma.getData<float>();
    float *inv_rms_data = inv_rms.getData<float>();

    // Initialize dgamma to zero
    dgamma.setZero();

    unsigned int total_reshaped_rows = batch * channel * reshaped_height;

    // First pass: compute inv_rms for each row
    for (unsigned int row = 0; row < total_reshaped_rows; ++row) {
      // Map reshaped row to original tensor position
      unsigned int orig_batch = row / (channel * reshaped_height);
      unsigned int remainder = row % (channel * reshaped_height);
      unsigned int orig_channel = remainder / reshaped_height;
      unsigned int reshaped_h = remainder % reshaped_height;

      unsigned int orig_height = reshaped_h / (width / feature_size);
      unsigned int width_group = reshaped_h % (width / feature_size);
      unsigned int orig_offset =
        orig_batch * channel * height * width + orig_channel * height * width +
        orig_height * width + width_group * feature_size;

      // Compute mean(x^2) for this row
      float sum_sq = 0.0f;
      for (unsigned int w = 0; w < feature_size; ++w) {
        float val = in_data[orig_offset + w];
        sum_sq += val * val;
      }
      float mean_sq = sum_sq / static_cast<float>(feature_size);
      inv_rms_data[row] = 1.0f / std::sqrt(mean_sq + epsilon);
    }

    // Second pass: accumulate gradient for gamma
    // dgamma[w] += sum over all rows: dy[w] * x[w] * inv_rms
    for (unsigned int row = 0; row < total_reshaped_rows; ++row) {
      // Map reshaped row to original tensor position
      unsigned int orig_batch = row / (channel * reshaped_height);
      unsigned int remainder = row % (channel * reshaped_height);
      unsigned int orig_channel = remainder / reshaped_height;
      unsigned int reshaped_h = remainder % reshaped_height;

      unsigned int orig_height = reshaped_h / (width / feature_size);
      unsigned int width_group = reshaped_h % (width / feature_size);
      unsigned int orig_offset =
        orig_batch * channel * height * width + orig_channel * height * width +
        orig_height * width + width_group * feature_size;

      float inv_rms_val = inv_rms_data[row];

      for (unsigned int w = 0; w < feature_size; ++w) {
        dgamma_data[w] +=
          dy_data[orig_offset + w] * in_data[orig_offset + w] * inv_rms_val;
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
    throw std::invalid_argument(
      "[reshaped_rms_norm] calcGradient: FP16 is not implemented yet");
  }
}

} // namespace nntrainer

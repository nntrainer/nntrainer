// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <iostream>

#include "rms_norm.h"

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void RMSNormLayer::finalize(InitLayerContext &context) {
  std::vector<TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()));
  wt_idx[static_cast<size_t>(RMSParams::gamma)] =
    context.requestWeight(gamma_dim, Initializer::ONES, WeightRegularizer::NONE,
                          1.0f, 0.0f, "gamma", true);
}

void RMSNormLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  unsigned int height = in.getDim().height();
  incremental_forwarding(context, 0, height, training);
}

void RMSNormLayer::incremental_forwarding(RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  auto &epsilon = std::get<props::Epsilon>(rms_props).get();

  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &gamma =
    context.getWeight(wt_idx[static_cast<size_t>(RMSParams::gamma)]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      auto t = in_step.multiply(in_step).average(3).add(epsilon);
      t.inv_sqrt_i();
      in_step.multiply(t, out_step);
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }
    out_step.multiply_i(gamma);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }
}

void RMSNormLayer::updateTensorsByInputDimensions(
  RunLayerContext &context, std::vector<TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void RMSNormLayer::calcDerivative(RunLayerContext &context) {
  auto &epsilon = std::get<props::Epsilon>(rms_props).get();

  const Tensor &incoming_deriv =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &outgoing_deriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &gamma =
    context.getWeight(wt_idx[static_cast<size_t>(RMSParams::gamma)]);

  if (input.getDataType() == ml::train::TensorDim::DataType::FP32) {
    unsigned int batch = input.getDim().batch();
    unsigned int channel = input.getDim().channel();
    unsigned int height = input.getDim().height();
    unsigned int width = input.getDim().width();

    // inv_rms: one scalar per (batch, channel, height) row.
    // shape: (batch, channel, height, 1)
    TensorDim inv_rms_dim(
      batch, channel, height, 1,
      TensorDim::TensorType(input.getDim().getFormat(), input.getDataType()));
    Tensor inv_rms(inv_rms_dim, true);

    // Compute inv_rms: inv_rms = 1 / sqrt(mean(x^2) + epsilon)
    // inv_rms[row] = 1 / sqrt(mean(x[row]^2) + epsilon)
    input.multiply(input,
                   outgoing_deriv); // outgoing_deriv = x^2 (temporary buffer)
    outgoing_deriv.average(3, inv_rms); // inv_rms = mean(x^2) along width
    inv_rms.add_i(epsilon);             // inv_rms = mean(x^2) + epsilon
    inv_rms.inv_sqrt_i(); // inv_rms = 1 / sqrt(mean(x^2) + epsilon)

    // Now compute the derivative
    // dx = inv_rms * (gamma*dy - x * mean(gamma*dy*x) * inv_rms²)

    // Flatten batch/channel/height into a single row index
    unsigned int total_rows = batch * channel * height;

    const float *in_data = input.getData<float>();
    const float *dy_data = incoming_deriv.getData<float>();
    float *dx_data = outgoing_deriv.getData<float>();
    const float *gamma_data = gamma.getData<float>();
    const float *inv_rms_data = inv_rms.getData<float>();

    for (unsigned int row = 0; row < total_rows; ++row) {
      unsigned int offset = row * width;
      float inv_rms_val = inv_rms_data[row];
      float inv_rms_sq = inv_rms_val * inv_rms_val;

      // c = mean(gamma * dy * x) over width
      float c = 0.0f;
      for (unsigned int w = 0; w < width; ++w) {
        c += gamma_data[w] * dy_data[offset + w] * in_data[offset + w];
      }
      c /= static_cast<float>(width);

      // dx[w] = inv_rms * (gamma[w]*dy[w] - x[w] * c * inv_rms²)
      for (unsigned int w = 0; w < width; ++w) {
        dx_data[offset + w] =
          inv_rms_val * (gamma_data[w] * dy_data[offset + w] -
                         in_data[offset + w] * c * inv_rms_sq);
      }
    }
  } else if (input.getDataType() == ml::train::TensorDim::DataType::FP16) {
    throw std::invalid_argument(
      "RMSNorm calcDerivative: FP16 is not implemented yet");
  }
}

void RMSNormLayer::calcGradient(RunLayerContext &context) {
  auto &epsilon = std::get<props::Epsilon>(rms_props).get();

  const Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  const Tensor &dy = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &dgamma =
    context.getWeightGrad(wt_idx[static_cast<size_t>(RMSParams::gamma)]);

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    unsigned int batch = in.getDim().batch();
    unsigned int channel = in.getDim().channel();
    unsigned int height = in.getDim().height();
    unsigned int width = in.getDim().width();

    TensorDim inv_rms_dim(
      batch, channel, height, 1,
      TensorDim::TensorType(in.getDim().getFormat(), in.getDataType()));
    Tensor inv_rms(inv_rms_dim, true);

    // Compute inv_rms: inv_rms = 1 / sqrt(mean(x^2) + epsilon)
    // Use a temporary tensor for computation
    Tensor temp(in.getDim(), true);
    in.multiply(in, temp);    // temp = x^2
    temp.average(3, inv_rms); // inv_rms = mean(x^2) along width
    inv_rms.add_i(epsilon);   // inv_rms = mean(x^2) + epsilon
    inv_rms.inv_sqrt_i();     // inv_rms = 1 / sqrt(mean(x^2) + epsilon)

    dgamma.setZero();

    const float *in_data = in.getData<float>();
    const float *dy_data = dy.getData<float>();
    float *dgamma_data = dgamma.getData<float>();
    const float *inv_rms_data = inv_rms.getData<float>();

    unsigned int total_rows = batch * channel * height;

    for (unsigned int row = 0; row < total_rows; ++row) {
      unsigned int offset = row * width;
      float inv_rms_val = inv_rms_data[row];

      for (unsigned int w = 0; w < width; ++w) {
        dgamma_data[w] +=
          dy_data[offset + w] * in_data[offset + w] * inv_rms_val;
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
    throw std::invalid_argument(
      "RMSNorm calcGradient: FP16 is not implemented yet");
  }
}

} // namespace nntrainer

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of RMS normalisation.
 *
 *         Forward:  out = x * inv_rms * gamma
 *         where     inv_rms = 1 / sqrt(mean(x²) + epsilon)
 *         Normalisation is over the width (last) dimension.
 *
 *         This is the special case of ReshapedRMSNorm where
 *         feature_size == width — one chunk per row, no reshape needed.
 *
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "rms_norm.h"
#include <cmath>
#include <iostream>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// ---------------------------------------------------------------------------
// finalize
// ---------------------------------------------------------------------------

void RMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);

  // gamma: one scale per width element.
  // ONES initialiser — identity pass-through at init.
  // trainable=true so a gradient buffer is allocated for calcGradient.
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::Initializer::ONES, nntrainer::WeightRegularizer::NONE,
    1.0f, 0.0f, "gamma", true);

  // inv_rms cache: one scalar per (batch, channel, height) row.
  // shape: (batch, channel, height, 1)
  // ITERATION_LIFESPAN: lives from forwarding() through calcDerivative /
  // calcGradient, then freed. Not needed during inference.
  nntrainer::TensorDim inv_rms_dim(
    dim[0].batch(), dim[0].channel(), dim[0].height(), 1,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::inv_rms] =
    context.requestTensor(inv_rms_dim, "inv_rms", nntrainer::Initializer::NONE,
                          false, nntrainer::TensorLifespan::ITERATION_LIFESPAN);
}

// ---------------------------------------------------------------------------
// forwarding  (training: full sequence, caches inv_rms)
// ---------------------------------------------------------------------------

/**
 * @brief Full-sequence forward pass used during training.
 *
 * Computes RMSNorm over the width dimension for every row and writes each
 * row's inv_rms into the cache tensor so that calcDerivative and
 * calcGradient can reuse it without recomputing sqrt.
 *
 * This implementation is unified with incremental_forwarding() by processing
 * the entire sequence in a single call (from=0, to=height). This ensures
 * consistency between training and inference modes while maintaining code
 * simplicity and enabling the use_inc_forward test mode.
 */
void RMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  unsigned int height = in.getDim().height();

  // Process the full sequence by calling incremental_forwarding once with
  // from=0 and to=height. This unifies the implementation and ensures
  // training/inference consistency.
  incremental_forwarding(context, 0, height, training);
}

// ---------------------------------------------------------------------------
// incremental_forwarding  (inference: token-by-token)
// ---------------------------------------------------------------------------

/**
 * @brief Token-by-token forward pass used during inference.
 *
 * Processes the [from, to) slice of the height dimension per batch.
 * Writes each step's inv_rms into the correct slice of the cache tensor
 * using the correct from offset so multi-step generation stays consistent.
 *
 * Only FP32 is supported; FP16 throws with a clear message.
 */
void RMSNormLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {

  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  nntrainer::Tensor &inv_rms = context.getTensor(wt_idx[RMSParams::inv_rms]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();
  ml::train::TensorDim inv_rms_dim = inv_rms.getDim();

  unsigned int step_height = to - from;
  unsigned int b_size = in_dim.batch();

  // Step-sized slice dimensions
  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;
  ml::train::TensorDim inv_rms_step_dim = inv_rms_dim;

  in_step_dim.batch(1);
  in_step_dim.height(step_height);
  out_step_dim.batch(1);
  out_step_dim.height(step_height);
  inv_rms_step_dim.batch(1);
  inv_rms_step_dim.height(step_height);

  for (unsigned int b = 0; b < b_size; ++b) {
    // Correct from offset applied so multi-step generation reads/writes
    // the right position rather than always starting from 0.
    nntrainer::Tensor in_step = in.getSharedDataTensor(
      in_step_dim, b * in_dim.getFeatureLen() + from * in_dim.width(), true);
    nntrainer::Tensor out_step = out.getSharedDataTensor(
      out_step_dim, b * out_dim.getFeatureLen() + from * out_dim.width(), true);
    nntrainer::Tensor inv_rms_step = inv_rms.getSharedDataTensor(
      inv_rms_step_dim, b * inv_rms_dim.getFeatureLen() + from, true);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      // Compute inv_rms and write into cache slice
      in_step.multiply(in_step, out_step); // out_step = x²
      out_step.average(3, inv_rms_step);   // inv_rms_step = mean(x²)
      inv_rms_step.add_i(epsilon);         // + ε
      inv_rms_step.inv_sqrt_i();           // 1/sqrt(...)

      in_step.multiply(inv_rms_step, out_step); // out_step = x * inv_rms
    } else {
      throw std::invalid_argument(
        "RMSNorm incremental_forwarding: only FP32 is currently supported");
    }
    out_step.multiply_i(gamma); // out_step = x * inv_rms * γ

#ifdef DEBUG
    std::cout << context.getName() << "\n input:" << in_step
              << "\n output:" << out_step << "\n gamma:" << gamma << std::endl;
#endif
  }
}

// ---------------------------------------------------------------------------
// updateTensorsByInputDimensions
// ---------------------------------------------------------------------------

/**
 * @brief Update input, output, and inv_rms cache shapes when input dims change.
 *
 * Without updating inv_rms here, its height would go stale if the input
 * height changes dynamically (e.g. different sequence lengths across runs).
 */
void RMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);

  // Keep inv_rms shape consistent with (batch, channel, height, 1)
  nntrainer::TensorDim new_inv_rms_dim(
    input_dimensions[0].batch(), input_dimensions[0].channel(),
    input_dimensions[0].height(), 1, input_dimensions[0].getTensorType());
  context.updateTensor(wt_idx[RMSParams::inv_rms], new_inv_rms_dim);
}

// ---------------------------------------------------------------------------
// calcDerivative  (backward through the normalisation)
// ---------------------------------------------------------------------------

/**
 * @brief Backward pass for the normalisation step.
 *
 * Forward per row:  out = x * inv_rms * gamma
 * Backward per row:
 *   dx = inv_rms * (gamma*dy - x * mean(gamma*dy*x) * inv_rms²)
 *
 * inv_rms is read from the cache filled during forwarding() — no sqrt
 * recomputation needed.
 *
 * Works entirely with raw float pointers per row — no heap tensor
 * allocations, no const_cast, no in-place reshape of input/dy tensors.
 *
 * mean_val and inv_rms_sq are stack scalars per row — zero heap allocations.
 */
void RMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {

  const nntrainer::Tensor &incoming_deriv =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &outgoing_deriv =
    context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  nntrainer::Tensor &inv_rms_tensor =
    context.getTensor(wt_idx[RMSParams::inv_rms]);

  const float *in_data = input.getData<float>();
  const float *dy_data = incoming_deriv.getData<float>();
  float *dx_data = outgoing_deriv.getData<float>();
  const float *gamma_data = gamma.getData<float>();
  const float *inv_rms_data = inv_rms_tensor.getData<float>();

  unsigned int batch = input.getDim().batch();
  unsigned int channel = input.getDim().channel();
  unsigned int height = input.getDim().height();
  unsigned int width = input.getDim().width();

  // Flatten batch/channel/height into a single row index.
  // inv_rms_data is indexed by row (one value per row) and hoisted outside
  // the width loop — one memory read per row not one per element.
  unsigned int total_rows = batch * channel * height;

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
}

// ---------------------------------------------------------------------------
// calcGradient  (backward through gamma)
// ---------------------------------------------------------------------------

/**
 * @brief Compute gradient for gamma.
 *
 *   dgamma[w] += sum over all rows: dy[row,w] * x[row,w] * inv_rms[row]
 *
 * inv_rms is read from the cache — same value used in calcDerivative.
 * No sqrt recomputation.
 *
 * inv_rms_val is hoisted outside the width loop — one memory read per row
 * not one per element. Batch/channel/height are flattened into a single
 * row index to avoid redundant index arithmetic in the inner loop.
 */
void RMSNormLayer::calcGradient(nntrainer::RunLayerContext &context) {

  const nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &dy = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dgamma = context.getWeightGrad(wt_idx[RMSParams::gamma]);
  const nntrainer::Tensor &inv_rms_tensor =
    context.getTensor(wt_idx[RMSParams::inv_rms]);

  if (in.getDataType() != ml::train::TensorDim::DataType::FP32) {
    throw std::invalid_argument(
      "RMSNorm calcGradient: only FP32 is currently supported");
  }

  dgamma.setZero();

  const float *in_data = in.getData<float>();
  const float *dy_data = dy.getData<float>();
  float *dgamma_data = dgamma.getData<float>();
  const float *inv_rms_data = inv_rms_tensor.getData<float>();

  unsigned int batch = in.getDim().batch();
  unsigned int channel = in.getDim().channel();
  unsigned int height = in.getDim().height();
  unsigned int width = in.getDim().width();

  unsigned int total_rows = batch * channel * height;

  for (unsigned int row = 0; row < total_rows; ++row) {
    unsigned int offset = row * width;
    float inv_rms_val = inv_rms_data[row]; // hoisted: one read per row

    for (unsigned int w = 0; w < width; ++w) {
      dgamma_data[w] += dy_data[offset + w] * in_data[offset + w] * inv_rms_val;
    }
  }
}

// ---------------------------------------------------------------------------
// Plugin registration
// ---------------------------------------------------------------------------

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new RMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
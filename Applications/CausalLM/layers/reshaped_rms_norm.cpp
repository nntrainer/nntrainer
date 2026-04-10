// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   reshaped_rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of ReshapedRMSNorm — RMS normalisation over
 *         feature_size-wide chunks of the width dimension.
 *
 *         This allows per-head or per-group normalisation without changing
 *         the tensor layout.  With feature_size == width it degenerates to
 *         standard RMSNorm.
 *
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "reshaped_rms_norm.h"
#include <cmath>
#include <cpu_backend.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// ---------------------------------------------------------------------------
// finalize
// ---------------------------------------------------------------------------

void ReshapedRMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);

  feature_size = std::get<props::FeatureSize>(rms_props);

  NNTR_THROW_IF(dim[0].width() % feature_size != 0, std::invalid_argument)
    << "feature_size (" << feature_size
    << ") must be a divisor of input width (" << dim[0].width() << ")";

  // gamma: one scale per feature element, shared across all chunks.
  // ONES initialiser so the initial pass-through is an identity.
  // Marked trainable=true so a gradient buffer is allocated for calcGradient.
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, feature_size,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::Initializer::ONES, nntrainer::WeightRegularizer::NONE,
    1.0f, 0.0f, "gamma", true);

  // inv_rms cache: one scalar per chunk per (batch, channel, height) row.
  //   num_chunks = height * (width / feature_size)
  //   shape: (batch, channel, num_chunks, 1)
  // ITERATION_LIFESPAN: lives from forwarding() through calcDerivative/
  // calcGradient, then freed. Not needed during inference.
  unsigned int num_chunks = dim[0].height() * (dim[0].width() / feature_size);
  nntrainer::TensorDim inv_rms_dim(
    dim[0].batch(), dim[0].channel(), num_chunks, 1,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::inv_rms] =
    context.requestTensor(inv_rms_dim, "inv_rms", nntrainer::Initializer::NONE,
                          false, nntrainer::TensorLifespan::ITERATION_LIFESPAN);
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/**
 * @brief Normalise one feature_size-wide chunk in-place and return inv_rms.
 *
 *   out[j] = in[j] * inv_rms * gamma[j]
 *   inv_rms = 1 / sqrt(mean(in²) + epsilon)
 *
 * Keeping this as a static helper deduplicates the identical scalar kernel
 * used by forwarding(), incremental_forwarding() (training path), and
 * ensures calcDerivative/calcGradient read the same value that was written
 * to the cache rather than recomputing it.
 */
static inline float rms_norm_chunk(const float *in, float *out,
                                   const float *gamma, unsigned int fs,
                                   float epsilon) {
  float sum = 0.0f;
  for (unsigned int j = 0; j < fs; ++j)
    sum += in[j] * in[j];
  float inv_rms_val = 1.0f / std::sqrt(sum / fs + epsilon);
  for (unsigned int j = 0; j < fs; ++j)
    out[j] = in[j] * inv_rms_val * gamma[j];
  return inv_rms_val;
}

// ---------------------------------------------------------------------------
// forwarding  (training: full sequence, caches inv_rms)
// ---------------------------------------------------------------------------

/**
 * @brief Full-sequence forward pass used during training.
 *
 * For training mode: uses the scalar rms_norm_chunk kernel and writes
 * each chunk's inv_rms into the cache tensor so that calcDerivative and
 * calcGradient can reuse it without recomputing sqrt.
 *
 * For inference mode: delegates to incremental_forwarding() to use SIMD
 * intrinsics, ensuring consistency between forwarding and
 * incremental_forwarding.
 */
void ReshapedRMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {
  if (training) {
    // Training path: scalar kernel + cache inv_rms for backward
    auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

    nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
    nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
    nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

    const float *in_data = in.getData<float>();
    float *out_data = out.getData<float>();
    const float *gamma_data = gamma.getData<float>();

    unsigned int batch = in.getDim().batch();
    unsigned int channel = in.getDim().channel();
    unsigned int height = in.getDim().height();
    unsigned int width = in.getDim().width();
    unsigned int num_features = width / feature_size;

    nntrainer::Tensor &inv_rms_tensor =
      context.getTensor(wt_idx[RMSParams::inv_rms]);
    float *inv_rms_data = inv_rms_tensor.getData<float>();

    for (unsigned int b = 0; b < batch; ++b) {
      for (unsigned int c = 0; c < channel; ++c) {
        for (unsigned int h = 0; h < height; ++h) {
          for (unsigned int f = 0; f < num_features; ++f) {
            // Flat index into inv_rms: (b*channel + c) * num_chunks + h*nf + f
            unsigned int row =
              (b * channel + c) * height * num_features + h * num_features + f;
            unsigned int offset =
              (b * channel + c) * height * width + h * width + f * feature_size;

            inv_rms_data[row] =
              rms_norm_chunk(in_data + offset, out_data + offset, gamma_data,
                             feature_size, epsilon);
          }
        }
      }
    }
  } else {
    // Inference path: delegate to incremental_forwarding for consistency
    unsigned int height = context.getInput(SINGLE_INOUT_IDX).getDim().height();
    incremental_forwarding(context, 0, height, false);
  }
}

// ---------------------------------------------------------------------------
// incremental_forwarding  (inference: token-by-token, no cache)
// ---------------------------------------------------------------------------

/**
 * @brief Token-by-token forward pass used during inference.
 *
 * The training path caches inv_rms using the correct [from, to) offset into
 * the full inv_rms tensor so that if backward were ever called the values
 * would be valid.
 *
 * The inference path uses SIMD intrinsics for speed. The from offset is
 * correctly applied so multi-step generation processes the right tokens.
 */
void ReshapedRMSNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  unsigned int step_height = to - from;
  unsigned int width = in_dim.width();
  unsigned int num_features = width / feature_size;
  unsigned int full_height = in_dim.height();
  unsigned int b_size = in_dim.batch();
  unsigned int channel = in_dim.channel();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;
  in_step_dim.batch(1);
  in_step_dim.height(step_height);
  out_step_dim.batch(1);
  out_step_dim.height(step_height);

  // Reshaped view: (1, 1, step_height * num_features, feature_size)
  ml::train::TensorDim step_reshaped_dim = in_step_dim;
  step_reshaped_dim.width(feature_size);
  step_reshaped_dim.height(step_height * num_features);

  if (training) {
    // Training path: scalar kernel + cache inv_rms
    // Use raw pointers like RMS norm - process ALL channels at once per batch
    nntrainer::Tensor &inv_rms_tensor =
      context.getTensor(wt_idx[RMSParams::inv_rms]);
    const float *gamma_data = gamma.getData<float>();
    float *inv_rms_data = inv_rms_tensor.getData<float>();
    const float *in_data = in.getData<float>();
    float *out_data = out.getData<float>();

    for (unsigned int b = 0; b < b_size; ++b) {
      for (unsigned int c = 0; c < channel; ++c) {
        for (unsigned int h = from; h < to; ++h) {
          for (unsigned int f = 0; f < num_features; ++f) {
            unsigned int row = (b * channel + c) * full_height * num_features +
                               h * num_features + f;
            unsigned int offset = (b * channel + c) * full_height * width +
                                  h * width + f * feature_size;

            inv_rms_data[row] =
              rms_norm_chunk(in_data + offset, out_data + offset, gamma_data,
                             feature_size, epsilon);
          }
        }
      }
    }
  } else {
    // Inference path: SIMD intrinsics
    for (unsigned int b = 0; b < b_size; ++b) {
      // Create tensor views starting from each batch.
      // The in_step_dim with height(to-from) automatically views the correct
      // range.
      nntrainer::Tensor in_step =
        in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
      nntrainer::Tensor out_step = out.getSharedDataTensor(
        out_step_dim, b * out_dim.getFeatureLen(), true);

      in_step.reshape(step_reshaped_dim);
      out_step.reshape(step_reshaped_dim);

      if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
#ifdef ENABLE_FP16
        nntrainer::rms_norm_wrt_width_fp16_intrinsic(
          in_step.getData<float>(), out_step.getData<float>(),
          in_step.getDim().height(), in_step.getDim().width(), epsilon);
#else
        nntrainer::rms_norm_wrt_width_fp32_intrinsic(
          in_step.getData<float>(), out_step.getData<float>(),
          in_step.getDim().height(), in_step.getDim().width(), epsilon);
#endif
      } else {
        throw std::invalid_argument(
          "ReshapedRMSNorm: incremental_forwarding not yet implemented "
          "for this data type");
      }
      out_step.multiply_i(gamma);
      out_step.reshape(out_step_dim);

#ifdef DEBUG
      std::cout << context.getName() << "\n input:" << in_step
                << "\n output:" << out_step << "\n gamma:" << gamma
                << std::endl;
#endif
    }
  }
}

// ---------------------------------------------------------------------------
// updateTensorsByInputDimensions
// ---------------------------------------------------------------------------

void ReshapedRMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);

  // Update inv_rms cache shape to match the new input dimensions.
  // Without this, inv_rms height goes stale if input height changes.
  unsigned int new_height = input_dimensions[0].height();
  unsigned int new_batch = input_dimensions[0].batch();
  unsigned int new_channel = input_dimensions[0].channel();
  unsigned int new_num_chunks =
    new_height * (input_dimensions[0].width() / feature_size);

  nntrainer::TensorDim new_inv_rms_dim(new_batch, new_channel, new_num_chunks,
                                       1, input_dimensions[0].getTensorType());
  context.updateTensor(wt_idx[RMSParams::inv_rms], new_inv_rms_dim);
}

// ---------------------------------------------------------------------------
// calcDerivative  (backward through the normalisation)
// ---------------------------------------------------------------------------

/**
 * @brief Backward pass for the normalisation step.
 *
 * Forward per chunk:  out = x * inv_rms * gamma
 * Backward per chunk:
 *   dx = inv_rms * (gamma*dy - x * mean(gamma*dy*x) * inv_rms²)
 *
 * inv_rms is read from the cache filled during forwarding() — no recomputation
 * of sqrt needed.
 *
 * Works entirely with raw float pointers per chunk — no heap tensor
 * allocations, no const_cast, no in-place reshape of input/dy tensors.
 */
void ReshapedRMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {

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
  unsigned int num_features = width / feature_size;

  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int c = 0; c < channel; ++c) {
      for (unsigned int h = 0; h < height; ++h) {
        for (unsigned int f = 0; f < num_features; ++f) {
          unsigned int row =
            (b * channel + c) * height * num_features + h * num_features + f;
          unsigned int offset =
            (b * channel + c) * height * width + h * width + f * feature_size;

          // Read cached inv_rms — computed once during forwarding()
          float inv_rms_val = inv_rms_data[row];
          float inv_rms_sq = inv_rms_val * inv_rms_val;

          // c = mean(gamma * dy * x) over this chunk
          float c_val = 0.0f;
          for (unsigned int j = 0; j < feature_size; ++j) {
            c_val += gamma_data[j] * dy_data[offset + j] * in_data[offset + j];
          }
          c_val /= static_cast<float>(feature_size);

          // dx[j] = inv_rms * (gamma[j]*dy[j] - x[j] * c * inv_rms²)
          for (unsigned int j = 0; j < feature_size; ++j) {
            dx_data[offset + j] =
              inv_rms_val * (gamma_data[j] * dy_data[offset + j] -
                             in_data[offset + j] * c_val * inv_rms_sq);
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// calcGradient  (backward through gamma)
// ---------------------------------------------------------------------------

/**
 * @brief Compute gradient for gamma.
 *
 *   dgamma[j] += sum over all chunks: dy[chunk,j] * x[chunk,j] * inv_rms[chunk]
 *
 * inv_rms is read from the cache — same value used in calcDerivative.
 * No recomputation of sqrt.
 */
void ReshapedRMSNormLayer::calcGradient(nntrainer::RunLayerContext &context) {

  const nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &dy = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dgamma = context.getWeightGrad(wt_idx[RMSParams::gamma]);
  const nntrainer::Tensor &inv_rms_tensor =
    context.getTensor(wt_idx[RMSParams::inv_rms]);

  if (in.getDataType() != ml::train::TensorDim::DataType::FP32) {
    throw std::invalid_argument(
      "ReshapedRMSNorm calcGradient: only FP32 is supported");
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
  unsigned int num_features = width / feature_size;

  // Flatten batch/channel/height into a single chunk index to avoid
  // redundant index arithmetic in the inner loop.
  // inv_rms_data[row] is hoisted outside the j-loop to avoid a repeated
  // memory read per element.
  for (unsigned int b = 0; b < batch; ++b) {
    for (unsigned int c = 0; c < channel; ++c) {
      for (unsigned int h = 0; h < height; ++h) {
        for (unsigned int f = 0; f < num_features; ++f) {
          unsigned int row =
            (b * channel + c) * height * num_features + h * num_features + f;
          unsigned int offset =
            (b * channel + c) * height * width + h * width + f * feature_size;
          float inv_rms_val = inv_rms_data[row]; // hoisted: one read per chunk

          for (unsigned int j = 0; j < feature_size; ++j) {
            dgamma_data[j] +=
              dy_data[offset + j] * in_data[offset + j] * inv_rms_val;
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Plugin registration
// ---------------------------------------------------------------------------

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new ReshapedRMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
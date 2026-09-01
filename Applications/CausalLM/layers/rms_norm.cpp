// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cpu_backend.h>
#include <iostream>

#include "rms_norm.h"

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void RMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  // gamma is an unquantized weight read straight out of the model .bin, and
  // the .bin carries no per-tensor dtype: NeuralNetwork::load() derives every
  // weight's file offset by accumulating getMemoryBytes() over the graph. The
  // request must therefore reproduce the dtype the exporting graph used, which
  // for a packed=false norm is the activation dtype -- exactly what
  // getWeightDataType() yields here, and what the quantizer records in
  // model_tensor_type. Hard-coding FP32 misreads an FP16-stored gamma *and*
  // shifts every following weight's offset, and would disagree with the sibling
  // implementations of this same on-disk weight (RMSNormLayerGPU,
  // nntrainer::CudaRMSNormLayer, RMSNormLayerCl) which all use
  // getWeightDataType(). A package that really stores FP32 gamma declares an
  // FP32 activation dtype and lands here as FP32; the forward path still casts
  // gamma at the multiply site for any mismatched case.
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);
}

void RMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {}

void RMSNormLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  // A multi-token step is a prefill even when it does not start at 0 (a
  // resumed or chunked prefill), so (to - from) > 1 counts as prefill too.
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const auto &dim = in_step.getDim();
#ifdef ENABLE_FP16
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(), dim.height(),
        dim.width(), epsilon);

      // DO NOT USE rms_norm_wrt_width_fp16_intrinsic. It causes overflow!

      // nntrainer::rms_norm_wrt_width_fp16_intrinsic(
      //   in_step.getData<float>(), out_step.getData<float>(), dim.height(),
      //   dim.width(), epsilon);
#else

      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(), dim.height(),
        dim.width(), epsilon);
#endif
#ifdef ENABLE_FP16
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      // FP16 path: the sum of squares MUST be accumulated in FP32. Accumulating
      // it in FP16 loses precision and overflows on a large residual (|x| up to
      // ~1700 squares past the FP16 maximum of 65504 -> +Inf -> wrong scale ->
      // an exploded norm). On x86 the FP16 reduction happens to accumulate in
      // FP32, but the aarch64 NEON FP16 path accumulates in FP16 and produces
      // garbage there, while reshaped_rms_norm and CudaRMSNormLayer -- which
      // already reduce in FP32 -- stay correct. Reducing explicitly in FP32
      // here makes the host norm correct on every architecture. gamma is
      // applied below.
      const unsigned int rows = in_step_dim.channel() * in_step_dim.height();
      const unsigned int W = in_step_dim.width();
      const _FP16 *xi = in_step.getData<_FP16>();
      _FP16 *yi = out_step.getData<_FP16>();
      for (unsigned int r = 0; r < rows; ++r) {
        const _FP16 *xr = xi + (size_t)r * W;
        _FP16 *yr = yi + (size_t)r * W;
        float ss = 0.f;
        for (unsigned int k = 0; k < W; ++k) {
          const float v = static_cast<float>(xr[k]);
          ss += v * v;
        }
        const float inv =
          1.0f / std::sqrt(ss / static_cast<float>(W) + epsilon);
        for (unsigned int k = 0; k < W; ++k)
          yr[k] = static_cast<_FP16>(static_cast<float>(xr[k]) * inv);
      }
#endif
    } else {
      throw std::invalid_argument(
        "rms_norm NYI dtype=" +
        std::to_string(static_cast<int>(in_step.getDataType())) +
        " layer=" + context.getName());
    }
    // gamma normally matches the activation dtype (see finalize); cast it when
    // a package pins it to a different one before the elementwise multiply.
    if (gamma.getDataType() != out_step.getDataType()) {
      nntrainer::Tensor gamma_cast = gamma.clone(out_step.getDataType());
      out_step.multiply_i(gamma_cast);
    } else {
      out_step.multiply_i(gamma);
    }
#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }
}

void RMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

void RMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

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

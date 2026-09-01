// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   per_layer_slice.cpp
 * @date   07 Apr 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Selects per-layer input chunk from packed per-layer embedding tensor.
 */

#include <cstdlib>
#include <cstring>
#include <env_compat.h>
#include <per_layer_slice.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void PerLayerSliceLayer::finalize(nntrainer::InitLayerContext &context) {
  auto dims = context.getInputDimensions();
  auto in_dim = dims[0];
  if (!std::get<nntrainer::props::SkipPrefill>(slice_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(slice_props).get();

  unsigned int feature_size = std::get<props::FeatureSize>(slice_props).get();
  NNTR_THROW_IF(feature_size == 0, std::invalid_argument)
    << "feature_size must be > 0";
  NNTR_THROW_IF(in_dim.width() % feature_size != 0, std::invalid_argument)
    << "input width must be divisible by feature_size";

  auto out_dim = in_dim;
  out_dim.width(feature_size);
  context.setOutputDimensions({out_dim});
}

void PerLayerSliceLayer::forwarding(nntrainer::RunLayerContext &context,
                                    bool training) {}

void PerLayerSliceLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  // A multi-token step is a prefill even when it does not start at 0 (resumed
  // / chunked prefill), so recognize (to - from) > 1 as prefill too.
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  auto &in = context.getInput(SINGLE_INOUT_IDX);
  auto &out = context.getOutput(SINGLE_INOUT_IDX);

  unsigned int feature_size = std::get<props::FeatureSize>(slice_props).get();
  unsigned int layer_index = std::get<props::LayerIndex>(slice_props).get();

  auto in_dim = in.getDim();
  unsigned int num_layers = in_dim.width() / feature_size;
  NNTR_THROW_IF(layer_index >= num_layers, std::invalid_argument)
    << "layer_index out of range";

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out.getDim();
  in_step_dim.batch(1);
  out_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.height(to - from);

  unsigned int b_size = in_dim.batch();
  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step = out.getSharedDataTensor(
      out_step_dim, b * out.getDim().getFeatureLen(), true);

    unsigned int tokens = in_step_dim.height();
    const unsigned int W = in_dim.width();
    // dtype-aware slice: getData<float> on an FP16 tensor strides by 4 bytes
    // over 2-byte cells (wrong offset) and memcpy(sizeof(float)*fs) copies
    // twice the bytes -> a corrupt PLE slice. gemma4 QINT4-FP16 activations are
    // FP16.
#ifdef ENABLE_FP16
    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      _FP16 *in_data = in_step.getData<_FP16>();
      _FP16 *out_data = out_step.getData<_FP16>();
      bool done = false;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      static const bool gpu = nntr_env_on("NNTR_CUDA_ELTWISE");
      if (gpu) {
        const bool dev = nntrainer::cuda::dev_accessible(in_data);
        if (dev && nntrainer::cuda::cuda_slice_copy_fp16(
                     reinterpret_cast<const unsigned short *>(in_data),
                     reinterpret_cast<unsigned short *>(out_data), tokens, W,
                     layer_index * feature_size, feature_size))
          done = true;
      }
#endif
      if (!done) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
        // Host memcpy slicing reads the GPU-produced UVM input on the CPU; sync
        // first in async mode (no-op in default sync mode).
        nntrainer::cuda::drain_if_async();
#endif
        for (unsigned int t = 0; t < tokens; ++t)
          std::memcpy(out_data + t * feature_size,
                      in_data + t * W + layer_index * feature_size,
                      sizeof(_FP16) * feature_size);
      }
    } else
#endif
      if (in_step.getDataType() != ml::train::TensorDim::DataType::FP32) {
      throw std::invalid_argument(
        "[PerLayerSlice] unsupported activation data type");
    } else {
      float *in_data = in_step.getData<float>();
      float *out_data = out_step.getData<float>();
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // Host memcpy slicing reads the GPU-produced UVM input on the CPU; sync
      // first in async mode (no-op in default sync mode).
      nntrainer::cuda::drain_if_async();
#endif
      for (unsigned int t = 0; t < tokens; ++t)
        std::memcpy(out_data + t * feature_size,
                    in_data + t * W + layer_index * feature_size,
                    sizeof(float) * feature_size);
    }
  }
}

void PerLayerSliceLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  auto out_dim = input_dimensions[0];
  out_dim.width(std::get<props::FeatureSize>(slice_props).get());
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, out_dim);
}

void PerLayerSliceLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE
nntrainer::Layer *create_per_layer_slice_layer() {
  return new PerLayerSliceLayer();
}
void destroy_per_layer_slice_layer(nntrainer::Layer *layer) { delete layer; }
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_per_layer_slice_layer, destroy_per_layer_slice_layer};
}
#endif

} // namespace causallm

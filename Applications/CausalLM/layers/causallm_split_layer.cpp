// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   split_layer.cpp
 * @date   09 Dec 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is split layer class (operation layer)
 */

#include "common_properties.h"
#include "tensor_base.h"
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <causallm_split_layer.h>
#include <stdexcept>
#include <util_func.h>

#include <layer_context.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void SplitLayer::finalize(nntrainer::InitLayerContext &context) {
  axis = std::get<causallm::props::Axis>(split_props).get();
  start = std::get<causallm::props::StartIndex>(split_props).get() - 1;
  unsigned int end = std::get<causallm::props::EndIndex>(split_props).get() - 1;

  const nntrainer::TensorDim &in_dim = context.getInputDimensions()[0];
  nntrainer::TensorDim outputDim = context.getInputDimensions()[0];

  for (unsigned int i = 0; i < 4; ++i) {
    if (i == axis) {
      // outputDim[i] = end - start;
      outputDim.setTensorDim(i, end - start);
    } else {
      // outputDim[i] = in_dim[i];
      outputDim.setTensorDim(i, in_dim[i]);
    }
  }



  context.setOutputDimensions({outputDim});
}

void SplitLayer::forwarding(nntrainer::RunLayerContext &context, bool training) {
}

void SplitLayer::calcDerivative(nntrainer::RunLayerContext &context) {
}

void SplitLayer::setProperty(const std::vector<std::string> &values) {
  // std::cout << "DEBUG: SplitLayer::setProperty new version" << std::endl;
  auto remain_props = nntrainer::loadProperties(values, split_props);
  // Do not call LayerImpl::setProperty or check remain_props to avoid errors on 'name', 'input_layers' etc.
}

void SplitLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                        unsigned int from, unsigned int to,
                                        bool training) {
  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::TensorDim hidden_dim = hidden_.getDim();
  nntrainer::TensorDim hidden_step_dim = hidden_dim;

  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  const nntrainer::Tensor &input = context.getInput(0);
  nntrainer::TensorDim input_dim = input.getDim();
  nntrainer::TensorDim input_step_dim = input_dim;
  input_step_dim.batch(1);
  input_step_dim.height(to - from);

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    // Crucial fix: reset_stride = true for hidden (newly created), but false for input (which might have strides)
    // Actually, hidden_step is a new view on contiguous memory usually, but let's stick to the pattern.
    // The fix was specifically for input tensor access.
    nntrainer::Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    // FIXED: reset_stride = false to preserve strides of the input tensor
    nntrainer::Tensor input_step = input.getSharedDataTensor(
      input_step_dim, b * input_dim.getFeatureLen(), false);

    // Re-implement the loop logic since we don't have forwarding_operation anymore
    // Wait, reusing valid forwarding logic for steps might be cleaner if we extract it.
    // But for now, duplicating the loop for the step tensors is fine.
    
    nntrainer::TensorDim stepOutputDim = hidden_step.getDim();
    for (unsigned int cb = 0; cb < hidden_step.batch(); ++cb) { // batch is 1 here
      for (unsigned int c = 0; c < hidden_step.channel(); ++c) {
        for (unsigned int h = 0; h < hidden_step.height(); ++h) {
          for (unsigned int w = 0; w < hidden_step.width(); ++w) {
             unsigned int c_idx = (axis == 1) ? c + start : c;
             unsigned int h_idx = (axis == 2) ? h + start : h;
             unsigned int w_idx = (axis == 3) ? w + start : w;
             hidden_step.setValue(cb, c, h, w, input_step.getValue(cb, c_idx, h_idx, w_idx));
           }
         }
       }
    }
  }
}


#ifdef PLUGGABLE

nntrainer::Layer *create_split_layer() {
  auto layer = new SplitLayer();
  return layer;
}

void destroy_split_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_split_layer,
                                                   destroy_split_layer};
}

#endif

} /* namespace causallm */

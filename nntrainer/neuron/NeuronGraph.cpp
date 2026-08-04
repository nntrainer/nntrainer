// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   NeuronGraph.cpp
 * @date   30 Jul 2026
 * @brief  Layer that wraps execution of one precompiled MediaTek NeuroPilot
 *         (.dla) network via the Neuron Runtime.
 * @see    https://github.com/nnstreamer/nntrainer
 */
#include "NeuronGraph.h"

#include <common_properties.h>
#include <engine.h>
#include <layer_context.h>
#include <neuron_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

namespace {

std::shared_ptr<NeuronVar> getNeuronVar(RunLayerContext &context) {
  return std::static_pointer_cast<NeuronBackendVar>(context.getContextData())
    ->getVar();
}

} // namespace

NeuronGraph::NeuronGraph() :
  LayerImpl(), graph_props({}, {}, {}, props::FilePath(), {}, {}) {}

NeuronGraph::~NeuronGraph() {
  if (dla_path.empty()) {
    return;
  }

  // Mirror QNNGraph's teardown: reach the shared backend state through the
  // registered Engine context and free this DLA's runtime. Each .dla holds
  // exactly one network, so unlike QNN there is no risk of one layer's
  // destructor tearing down a runtime that a sibling layer still needs.
  auto *ctx = Engine::Global().getRegisteredContext("neuron");
  if (ctx == nullptr) {
    return;
  }
  auto *neuron_ctx = static_cast<NeuronContext *>(ctx);
  auto neuron_data = neuron_ctx->getNeuronData();
  if (neuron_data && neuron_data->findRuntime(dla_path).has_value()) {
    neuron_data->freeRuntime(dla_path);
  }
}

void NeuronGraph::finalize(InitLayerContext &context) {
  dla_path = std::get<props::FilePath>(graph_props).get();

  auto &dims = std::get<std::vector<props::TensorDimension>>(graph_props);
  auto t_dtype = std::get<std::vector<props::TensorDataType>>(graph_props);
  auto t_type = std::get<std::vector<props::TensorType>>(graph_props);

  NNTR_THROW_IF(dims.size() != t_dtype.size(), std::invalid_argument)
    << "Size of Dimension, DataTypes must be same!";
  NNTR_THROW_IF(dims.size() != t_type.size(), std::invalid_argument)
    << "Size of Dimension, Types must be same!";

  std::vector<TensorDim> out_dim;
  std::vector<TensorDim> t_dims(dims.begin(), dims.end());

  for (unsigned int i = 0; i < t_dims.size(); ++i) {
    t_dims[i].setFormat(context.getFormat());
    t_dims[i].setDataType(t_dtype[i]);

    switch (t_type[i]) {
    case nntrainer::TensorType_::OUT_TENSOR:
      out_dim.push_back(t_dims[i]);
      break;
    case nntrainer::TensorType_::IN_TENSOR: {
      std::string name = "w_" + std::to_string(i);
      context.requestTensor(t_dims[i], name, Initializer::NONE, true,
                            TensorLifespan::FORWARD_FUNC_LIFESPAN);
    } break;
    default:
      break;
    }
  }

  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  context.setOutputDimensions(out_dim);
}

void NeuronGraph::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, graph_props);
  LayerImpl::setProperty(remain_props);
}

void NeuronGraph::read(std::ifstream &file, RunLayerContext &run_context,
                       bool opt_var, ml::train::ExecutionMode mode,
                       bool trainable, TensorDim::DataType defineWeightDataType,
                       bool fsu, size_t start_offset, bool read_from_offset,
                       int file_fd) {}

void NeuronGraph::forwarding(RunLayerContext &context, bool training) {
  auto neuron_var = getNeuronVar(context);

  if (!neuron_var->findRuntime(dla_path)) {
    ml_logw("NeuronGraph: runtime not created yet for %s; creating now",
            dla_path.c_str());
    neuron_var->makeRuntime(dla_path);
  }

  auto entry_opt = neuron_var->findRuntime(dla_path);
  NNTR_THROW_IF(!entry_opt, std::invalid_argument)
    << "cannot create/retrieve Neuron runtime for " << dla_path;
  NeuronRuntimeEntry &entry = *entry_opt;

  NNTR_THROW_IF(context.getNumInputs() != entry.numInputs,
                std::invalid_argument)
    << "Number of nntrainer's inputs " << context.getNumInputs()
    << " does not match number of Neuron's input tensors " << entry.numInputs
    << "!";
  NNTR_THROW_IF(context.getNumOutputs() != entry.numOutputs,
                std::invalid_argument)
    << "Number of nntrainer's outputs " << context.getNumOutputs()
    << " does not match number of Neuron's output tensors " << entry.numOutputs
    << "!";

  // suppressInputConversion/suppressOutputConversion are left false (see
  // NeuronContext::init()), so the Runtime expects the buffer length to be
  // at least the EXACT (non-padded) size reported by getInputSize/
  // getOutputSize. The padded size only becomes the relevant one when those
  // options are flipped on for the raw-device-format zero-copy path (a
  // later optimization, not needed for correctness here).
  for (size_t i = 0; i < context.getNumInputs(); ++i) {
    Tensor &t = context.getInput(i);
    const size_t required = entry.inputExactSizes[i];
    NNTR_THROW_IF(t.bytes() < required, std::invalid_argument)
      << "neuron_graph input " << i << ": nntrainer tensor buffer is "
      << t.bytes() << " bytes but the Neuron network requires at least "
      << required
      << " bytes; adjust this layer's dim/tensor_dtype properties to match";

    void *ptr = t.getData<void>();
    BufferAttribute attr = neuron_var->DmaAlloc
                             ? neuron_var->DmaAlloc->attributeFor(ptr)
                             : BufferAttribute{-1};
    int ret = neuron_var->api.setInput(entry.runtime, i, ptr, required, attr);
    NNTR_THROW_IF(ret != NEURONRUNTIME_NO_ERROR, std::runtime_error)
      << "NeuronRuntime_setInput failed for input " << i << " with error code "
      << ret;
  }

  for (size_t i = 0; i < context.getNumOutputs(); ++i) {
    Tensor &t = context.getOutput(i);
    const size_t required = entry.outputExactSizes[i];
    NNTR_THROW_IF(t.bytes() < required, std::invalid_argument)
      << "neuron_graph output " << i << ": nntrainer tensor buffer is "
      << t.bytes() << " bytes but the Neuron network requires at least "
      << required
      << " bytes; adjust this layer's dim/tensor_dtype properties to match";

    void *ptr = t.getData<void>();
    BufferAttribute attr = neuron_var->DmaAlloc
                             ? neuron_var->DmaAlloc->attributeFor(ptr)
                             : BufferAttribute{-1};
    int ret = neuron_var->api.setOutput(entry.runtime, i, ptr, required, attr);
    NNTR_THROW_IF(ret != NEURONRUNTIME_NO_ERROR, std::runtime_error)
      << "NeuronRuntime_setOutput failed for output " << i
      << " with error code " << ret;
  }

  int ret = neuron_var->api.inference(entry.runtime);
  NNTR_THROW_IF(ret != NEURONRUNTIME_NO_ERROR, std::runtime_error)
    << "NeuronRuntime_inference failed with error code " << ret;
}

} // namespace nntrainer

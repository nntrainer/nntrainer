// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   QNNGraph.cpp
 * @date   10 Jan 2025
 * @brief  This is QNN Graph Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "QNNGraph.h"
#include "QnnTypes.h"
#include <cstdint>
#include <fcntl.h>
#include <inttypes.h>
#include <memory>
#include <sys/mman.h>
#include <unistd.h>

#include <sys/resource.h>
#include <thread>

#include "QnnSampleAppUtils.hpp"
#include "Utils/DataUtil.hpp"
#include <common_properties.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

std::chrono::duration<double> exec_seconds;

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

std::shared_ptr<QNNVar> getQNNVar(RunLayerContext &context) {
  std::shared_ptr<QNNVar> qc_var =
    (std::static_pointer_cast<QNNBackendVar>(context.getContextData()))
      ->getVar();
  return qc_var;
}

QNNGraph::QNNGraph() :
  LayerImpl(), graph_props({}, {}, {}, {}, props::FilePath(), {}, {}) {
  m_isContextCreated = false;
  m_inputDataType = iotensor::InputDataType::NATIVE;
}

QNNGraph::~QNNGraph() {
  if (m_context) {
    if (QNN_CONTEXT_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.contextFree(m_context, nullptr)) {
      ml_loge("Faile to free Context");
    }
  }
}

void QNNGraph::finalize(InitLayerContext &context) {
  bin_path = std::get<props::FilePath>(graph_props).get();
  bin_size = std::get<props::FilePath>(graph_props).file_size();

  auto &dims = std::get<std::vector<props::TensorDimension>>(graph_props);
  t_dims.assign(dims.begin(), dims.end());

  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);

  t_dtype = std::get<std::vector<props::TensorDataType>>(graph_props);

  t_name = std::get<std::vector<props::WeightName>>(graph_props);

  t_type = std::get<std::vector<props::TensorType>>(graph_props);

  n_weight = t_dims.size();

  if (!t_dtype.size()) {
    t_dtype.resize(t_dims.size());
    for (auto t : t_dims) {
      t_dtype.push_back(context.getWeightDataType());
    }
  }

  if (!t_type.size()) {
    t_type.resize(t_dims.size());
    int i = 0;
    for (auto t : t_dims) {
      from_string("WEIGHT", t_type[i]);
      i++;
    }
  }

  auto engine = context.getComputeEngineType();

  NNTR_THROW_IF(t_dims.size() != t_dtype.size(), std::invalid_argument)
    << "Size of Dimension, DataTypes must be same!";
  NNTR_THROW_IF(t_dims.size() != t_type.size(), std::invalid_argument)
    << "Size of Dimension, Types must be same!";

  weight_idx.reserve(t_dims.size());
  tensor_idx.reserve(t_dims.size());

  std::vector<TensorDim> out_dim;

  for (unsigned int i = 0; i < t_dims.size(); ++i) {
    t_dims[i].setFormat(context.getFormat());
    t_dims[i].setDataType(t_dtype[i]);

    std::string name = "w_" + std::to_string(i);

    if (!t_name.empty())
      name = t_name[i];

    switch (t_type[i]) {
    case nntrainer::TensorType_::WEIGHT:
      weight_idx.push_back(context.requestWeight(
        t_dims[i], Initializer::NONE, weight_regularizer,
        weight_regularizer_constant, weight_decay, name, true));
      break;
    case nntrainer::TensorType_::OUT_TENSOR:
      out_dim.push_back(t_dims[i]);
      break;
    case nntrainer::TensorType_::IN_TENSOR:
      tensor_idx.push_back(
        context.requestTensor(t_dims[i], name, Initializer::NONE, true,
                              TensorLifespan::FORWARD_FUNC_LIFESPAN));
      break;
    case nntrainer::TensorType_::MAX_IN_TENSOR:
    case nntrainer::TensorType_::MAX_OUT_TENSOR:
      out_dim.push_back(t_dims[i]);
      tensor_idx.push_back(context.requestTensor(t_dims[i], name,
                                                 Initializer::NONE, true,
                                                 TensorLifespan::MAX_LIFESPAN));
      break;
    default:
      break;
    }
  }

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  /** set output dimensions */
  /// @todo read number of outputs and out dimensions from graph info
  if (context.getName() == "my_model_29") {
    out_dim[0].setTensorDim("1:1:1:96000", out_dim[0].getTensorType());
  }
  context.setOutputDimensions(out_dim);
}

void QNNGraph::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, graph_props);

  LayerImpl::setProperty(remain_props);
}

StatusCode QNNGraph::freeContext(RunLayerContext &context) {
  std::shared_ptr<QNNVar> qc_var = getQNNVar(context);

  if (m_context) {
    if (QNN_CONTEXT_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.contextFree(m_context, nullptr)) {
      ml_loge("Faile to free Context");
      return StatusCode::FAILURE;
    }
    m_isContextCreated = false;
  }
  m_context = nullptr;
  return StatusCode::SUCCESS;
}

StatusCode QNNGraph::makeContext(RunLayerContext &context) {

  std::shared_ptr<QNNVar> qc_var = getQNNVar(context);

  return qc_var->makeContext(bin_path);
}

void QNNGraph::read(std::ifstream &file, RunLayerContext &run_context,
                    bool opt_var, ml::train::ExecutionMode mode, bool trainable,
                    TensorDim::DataType defineWeightDataType, bool fsu,
                    size_t start_offset, bool read_from_offset, int file_fd) {}

void QNNGraph::forwarding(RunLayerContext &context, bool training) {
  auto returnStatus = StatusCode::SUCCESS;

  auto qc_var = getQNNVar(context);
  unsigned int graphIdx = 0;

  if (!qc_var->findContext(bin_path)) {
    ml_logw("Context is not created. Create Now");
    auto start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);

    qc_var->makeContext(bin_path);

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  }

  auto graphInfo = qc_var->graphRetrieve(bin_path, context.getName());

  std::optional<std::reference_wrapper<Qnn_Context_Graph_t>> op =
    qc_var->findContext(bin_path);

  Qnn_Context_Graph_t &context_i = *op;

  NNTR_THROW_IF(!graphInfo, std::invalid_argument)
    << "cannot retrieve graph " << context.getName() << " from " << bin_path;

  NNTR_THROW_IF(context.getNumInputs() + context.getNumWeights() !=
                  graphInfo->numInputTensors,
                std::invalid_argument)
    << "Number of NNtrainer's inputs " << context.getNumInputs() << " + "
    << context.getNumWeights()
    << " does not match with number of QNN's input tensors "
    << graphInfo->numInputTensors << "!";

  NNTR_THROW_IF(context.getNumOutputs() != graphInfo->numOutputTensors,
                std::invalid_argument)
    << "Number of NNtrainer's outputs " << context.getNumOutputs()
    << " does not match with number of QNN's output tensors "
    << graphInfo->numOutputTensors << "!";

  for (size_t i = 0; i < context.getNumInputs(); ++i) {
    updateBufferType(currentInputBuffers, context.getInput(i));
  }

  for (size_t i = 0; i < context.getNumWeights(); ++i) {
    updateBufferType(currentWeightBuffers, context.getWeight(i));
  }

  for (size_t i = 0; i < context.getNumTensors(); ++i) {
    updateBufferType(currentTensorBuffers, context.getTensor(i));
  }

  for (size_t i = 0; i < graphInfo->numOutputTensors; ++i) {
    updateBufferType(currentOutputBuffers, context.getOutput(i));
  }

  Qnn_Tensor_t *inputs = nullptr;
  Qnn_Tensor_t *outputs = nullptr;

  qc_var->m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs, *graphInfo);

  size_t in_idx = 0;
  size_t out_idx = 0;
  size_t weight_idx = 0;
  size_t tensor_idx = 0;

  auto input_quant_params =
    std::get<std::vector<props::InputQuantParam>>(graph_props);
  std::map<std::string, std::pair<float, int>> input_quant_param_map;
  for (auto &param : input_quant_params) {
    auto p = param.get();
    input_quant_param_map[p.first] = p.second;
  }
  auto output_quant_params =
    std::get<std::vector<props::OutputQuantParam>>(graph_props);
  std::map<std::string, std::pair<float, int>> output_quant_param_map;
  for (auto &param : output_quant_params) {
    auto p = param.get();
    output_quant_param_map[p.first] = p.second;
  }

  for (size_t i = 0; i < context.getNumInputs(); ++i) {
    auto key = inputs[in_idx].v1.name;
    NNTR_THROW_IF(input_quant_param_map.find(key) ==
                    input_quant_param_map.end(),
                  std::invalid_argument)
      << key;
    auto value = input_quant_param_map[key];
    inputs[in_idx].v1.quantizeParams.scaleOffsetEncoding.scale = value.first;
    inputs[in_idx].v1.quantizeParams.scaleOffsetEncoding.offset = value.second;
    populateTensor(qc_var, context_i, currentInputBuffers[i],
                   &(inputs[in_idx]));
    ++in_idx;
  }

  for (size_t i = 0; i < context.getNumOutputs(); ++i) {
    auto key = outputs[out_idx].v1.name;
    NNTR_THROW_IF(output_quant_param_map.find(key) ==
                    output_quant_param_map.end(),
                  std::invalid_argument)
      << key;
    auto value = output_quant_param_map[key];
    outputs[out_idx].v1.quantizeParams.scaleOffsetEncoding.scale = value.first;
    outputs[out_idx].v1.quantizeParams.scaleOffsetEncoding.offset =
      value.second;
    populateTensor(qc_var, context_i, currentOutputBuffers[i],
                   &(outputs[out_idx]));
    ++out_idx;
  }

  auto start = std::chrono::system_clock::now();
  std::time_t start_time = std::chrono::system_clock::to_time_t(start);

  Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
  QnnGraph_Config_t **customGraphConfigs{nullptr};
  uint32_t configCount{0};
  auto backend_extensions = qc_var->m_backendExtensions;
  if (nullptr != backend_extensions && backend_extensions->interface()) {
    if (!backend_extensions->interface()->beforeExecute(
          graphInfo->graphName, &customGraphConfigs, &configCount)) {
      QNN_ERROR("Extensions Failure in beforeExecute()");
    }
    if (customGraphConfigs) {
      std::vector<const QnnGraph_Config_t *> graphConfigsPointers(
        configCount + 1, nullptr);
      for (size_t idx = 0u; idx < configCount; idx++) {
        graphConfigsPointers[idx] = customGraphConfigs[idx];
      }
      if (QNN_SUCCESS !=
          qc_var->m_qnnFunctionPointers.qnnInterface.graphSetConfig(
            graphInfo->graph, graphConfigsPointers.data())) {
        QNN_ERROR("Failure in setGraphConfigsBeforeExecute()");
      }
    }
  }
  executeStatus = qc_var->m_qnnFunctionPointers.qnnInterface.graphExecute(
    graphInfo->graph, inputs, graphInfo->numInputTensors, outputs,
    graphInfo->numOutputTensors, qc_var->m_profileBackendHandle, nullptr);

  if (nullptr != backend_extensions && backend_extensions->interface()) {
    if (!backend_extensions->interface()->afterExecute()) {
      QNN_ERROR("Extensions Failure in afterExecute()");
    }
  }

  // std::cout << "executed QNNGraph, name: " << graphInfo->graphName <<
  // std::endl;
  if (QNN_GRAPH_NO_ERROR != executeStatus) {
    returnStatus = StatusCode::FAILURE;
  }

  // std::cout << context.getOutput(0) << std::endl;

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  exec_seconds += elapsed_seconds;

  // qc_var->RpcMem->deRegisterQnnTensor();
  counter++;

  // std::cout << "graph exec_time : " << exec_seconds.count() << " " << counter
  //           << std::endl;

  if (StatusCode::SUCCESS != returnStatus) {
    ml_loge("Execution of Graph: %d failed!", graphIdx);
    std::cout << "Execution of Graph : " << graphIdx << " failed!" << std::endl;
  }
}

void QNNGraph::updateBufferType(std::vector<BufferTypePtr> &buffers,
                                Tensor &T) {
  Tdatatype type = T.getDataType();
  switch (type) {
  case Tdatatype::UINT4:
  case Tdatatype::UINT8:
    buffers.push_back(T.getData<uint8_t>());
    break;
  case Tdatatype::UINT16:
    buffers.push_back(T.getData<uint16_t>());
    break;
  case Tdatatype::FP32:
    buffers.push_back(T.getData<float>());
    break;
  default:
    break;
  }
}

void QNNGraph::populateTensor(std::shared_ptr<QNNVar> qc_var,
                              Qnn_Context_Graph_t &context_i,
                              BufferTypePtr buffers, Qnn_Tensor_t *T) {
  switch (buffers.index()) {
  case 1: // uint8_t *
  {
    qc_var->m_ioTensor.populateInputTensor(std::get<uint8_t *>(buffers), T,
                                           m_inputDataType);
    qc_var->RpcMem->registerQnnTensor(std::get<uint8_t *>(buffers), *T,
                                      context_i.m_context);
  } break;
  case 2: // uint16_t*
  {
    qc_var->m_ioTensor.populateInputTensor(std::get<uint16_t *>(buffers), T,
                                           m_inputDataType);
    qc_var->RpcMem->registerQnnTensor(std::get<uint16_t *>(buffers), *T,
                                      context_i.m_context);
  } break;
  case 3: {
    qc_var->m_ioTensor.populateInputTensor(std::get<float *>(buffers), T,
                                           m_inputDataType);
    qc_var->RpcMem->registerQnnTensor(std::get<float *>(buffers), *T,
                                      context_i.m_context);
  } break;
  default:
    std::cout << "Unknown type: " << buffers.index() << std::endl;
    break;
  }
}

} // namespace nntrainer
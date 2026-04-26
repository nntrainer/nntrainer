// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   QNNGraph.h
 * @date   10 Jan 2025
 * @brief  This is QNN Graph Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __NNTR_QNNGRAPH_H__
#define __NNTR_QNNGRAPH_H__

#include <iostream>
#include <layer_impl.h>
#include <qnn_context_var.h>
#include <qnn_properties.h>
#include <qnn_rpc_manager.h>

namespace nntrainer {

/**
 * @class   QNNGraph
 * @brief   Layer that wraps a precompiled QNN graph so it can be inserted
 *          into an nntrainer model. Owns the graph handle and forwards
 *          input/output tensors through the QNN runtime.
 */
class QNNGraph : public LayerImpl {
public:
  using BufferTypePtr =
    std::variant<std::monostate, uint8_t *, uint16_t *, float *>;

  QNNGraph();
  ~QNNGraph();

  inline static const std::string type = "qnn_graph";

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return QNNGraph::type; };

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override{};

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  StatusCode makeContext(RunLayerContext &context);

  StatusCode freeContext(RunLayerContext &context);

  void read(std::ifstream &file, RunLayerContext &run_context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType defineWeightDataType, bool fsu = false,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1) override;

  void updateBufferType(std::vector<BufferTypePtr> &buffers, Tensor &T);

  void populateTensor(std::shared_ptr<QNNVar> qc_var,
                      Qnn_Context_Graph_t &context_i, BufferTypePtr buffer,
                      Qnn_Tensor_t *T);

private:
  std::tuple<std::vector<props::TensorDimension>,
             std::vector<props::TensorDataType>, std::vector<props::WeightName>,
             std::vector<props::TensorType>, props::FilePath,
             std::vector<props::InputQuantParam>,
             std::vector<props::OutputQuantParam>>
    graph_props;

  unsigned int n_weight;
  std::vector<unsigned int> weight_idx;

  unsigned int n_tensor;
  std::vector<unsigned int> tensor_idx;

  int in_features_;
  int out_features_;
  bool state_load;
  Qnn_ContextHandle_t m_context;
  qnn_wrapper_api::GraphInfo_t **m_graphsInfo;
  uint32_t m_graphsCount;
  size_t bin_size;
  std::string bin_path;
  QnnContext_Config_t **m_contextConfig = nullptr;
  bool m_isContextCreated;

  iotensor::InputDataType m_inputDataType;

  std::vector<props::TensorDataType> t_dtype;
  std::vector<TensorDim> t_dims;
  std::vector<props::WeightName> t_name;
  std::vector<props::TensorType> t_type;

  std::vector<BufferTypePtr> currentInputBuffers;
  std::vector<BufferTypePtr> currentWeightBuffers;
  std::vector<BufferTypePtr> currentTensorBuffers;
  std::vector<BufferTypePtr> currentOutputBuffers;

  sample_app::QnnFunctionPointers m_qnnFunctionPointers;

  StatusCode initializeIOTensors(std::shared_ptr<nntrainer::QNNVar> qc_var);
  StatusCode
  extractBackendProfilingInfo(std::shared_ptr<nntrainer::QNNVar> qc_var);
  StatusCode
  extractProfilingSubEvents(std::shared_ptr<nntrainer::QNNVar> qc_var,
                            QnnProfile_EventId_t profileEventId);
  StatusCode extractProfilingEvent(std::shared_ptr<nntrainer::QNNVar> qc_var,
                                   QnnProfile_EventId_t profileEventId);
  bool mmapBinaryFile(std::string filePath, void **buffer, size_t bufferSize);

  int counter;
};

} // namespace nntrainer

#endif

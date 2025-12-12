// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   layer_node.h
 * @date   1 April 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer node for network graph
 *
 * @todo   Add printPreset support
 *
 * @details LayerNode provides a node wrapper around the Layer class to form a
 * GraphNode. Each layer is wrapped with LayerNode in order to add it to a
 * graph. Each LayerNode contains only 1 layer inside. LayerNode also intercepts
 * certain properties of the layer which are either related to graph related
 * connections (input_connections, output_connections, activation, flatten,
 * distribute, name) or essential for the description of the layer (trainable,
 * input_dims) itself. These properties, if needed by the layer object, are
 * provided access to via LayerContext.
 */

#ifndef __LAYER_NODE_H__
#define __LAYER_NODE_H__

#include <memory>
#include <tuple>
#include <vector>

#include <graph_node.h>
#include <layer.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <weight.h>

namespace nntrainer {

class Layer;
class Connection;
class Exporter;
class ContextData;

namespace props {
class Name;
class Distribute;
class Flatten;
class Loss;
class InputShape;
class Activation;
class SharedFrom;
class InputConnection;
class ClipGradByGlobalNorm;
class Packed;
class LossScaleForMixed;
class ComputeEngine;
} // namespace props

/**
 * @class   LayerNode class
 * @brief   layer node class for the graph
 */
class LayerNode final : public ml::train::Layer, public GraphNode {
public:
  /**
   * @brief Constructor of LayerNode class for v2
   * @param l layer to wrap with, the ownership is transferred to layer node
   *
   */
  LayerNode(std::unique_ptr<nntrainer::Layer> &&l);

  /**
   * @brief     Destructor of LayerNode Class
   *
   */
  ~LayerNode();

  /**
   * Support all the interface requirements by ml::train::Layer
   */

  /**
   * @brief Get the layer type
   *
   * @return const std::string type representation
   */
  const std::string getType() const override;

  /**
   * @brief     set Property of layer
   * @param[in] properties values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name=property_val, ...}
   *
   */
  void setProperty(const std::vector<std::string> &properties) override;

  std::string getProperty(const std::string &key) override;

  /**
   * @brief     Get name of the layer
   *
   * @retval    name of the layer
   * @note      This name is unique to this layer in a model
   * @note      This name might be changed once this layer is added to the model
   * to keep the name unique to the model
   */
  const std::string getName() const override;

  /**
   * Support all the interface requirements by nntrainer::GraphNode
   */

  /**
   * @brief     set name of layer
   *
   * @param[in] name Name of the layer
   */
  void setName(const std::string &name) override {
    setProperty({"name=" + name});
  }

  /**
   * @brief     set weight and activation data type of layer
   *
   * @param[in] weight data type, activation data type
   */
  void setDataType(const TensorDim::DataType w_type,
                   const TensorDim::DataType a_type) {
    data_type = {w_type, a_type};
  }

  /**
   * @brief Get the Weight Data Type
   *
   * @return TensorDim::DataType weight data type
   */
  const TensorDim::DataType getWeightDataType() const { return data_type[0]; }

  /**
   * @brief Get the Activation Data Type
   *
   * @return TensorDim::DataType activation data type
   */
  const TensorDim::DataType getActivationDataType() const {
    return data_type[1];
  }

  /**
   * @brief Get the Input Connection Index object
   *
   * @param nth nth input
   * @throws if nth is out of range of getNumInputConnection()
   * @return const unsigned index
   */
  const unsigned getInputConnectionIndex(unsigned nth) const;

  /**
   * @brief Get the Input Connection Name object
   *
   * @param nth nth input
   * @throws if nth is out of range of getNumInputConnection()
   * @return const std::string& name
   */
  const std::string &getInputConnectionName(unsigned nth) const;

  /**
   * @brief Set the Input Connection Index object
   *
   * @param nth nth input
   * @param index index to set
   * @throws if nth is out of range of getNumInputConnection()
   */
  void setInputConnectionIndex(unsigned nth, unsigned index);

  /**
   * @brief Get the Input Connection Name object
   *
   * @param nth input
   * @param index index to set
   * @throws if nth is out of range of getNumInputConnection()
   * @throws if new identifier is invalid
   */
  void setInputConnectionName(unsigned nth, const std::string &name);

  /**
   * @brief Get the output connection object
   *
   * @param nth nth input
   * @throws if nth is out of range of getNumOutputConnection()
   * @return Connection * view of a connection, null means this does not exist
   */
  const Connection *getOutputConnection(unsigned nth) const;

  /**
   * @brief Set the Output Connection
   * @note Each output must be identified only ONCE.
   * @note when nth comes, getNumOutput() expends to nth + 1 as resize occurs.
   * Please also notice none identified intermediate output (or mismatch between
   * actual number of out tensors and output) is allowed but will produce
   * warning, this implies that the output is not used else where.
   * @throw std::invalid_argument when trying to identify output
   * more then once
   *
   * @param nth nth output
   * @param name name of the output bound connection
   * @param index index of the output bound connection
   */
  void setOutputConnection(unsigned nth, const std::string &name,
                           unsigned index);

  /**
   * @brief set the compute engine for this node
   * @param compute engine (CPU/GPU)
   */
  void setComputeEngine(const ml::train::LayerComputeEngine &compute_engine =
                          ml::train::LayerComputeEngine::CPU);

  /**
   * @brief     Get the input connections for this node
   *
   * @return list of name of the nodes which form input connections
   */
  const std::vector<std::string> getInputConnections() const override {
    return getInputLayers();
  }

  /**
   * @brief     Get the output connections for this node
   *
   * @return list of name of the nodes which form output connections
   */
  const std::vector<std::string> getOutputConnections() const override {
    return getOutputLayers();
  }

  /**
   * @brief     get the execution order/location of this node
   *
   * @retval    the execution order/location of this node
   */
  ExecutionOrder getExecutionOrder() const override { return exec_order; }

  /**
   * @brief     set the execution order/location of this node
   *
   * @param     exec_order the execution order/location of this node
   */
  void setExecutionOrder(ExecutionOrder exec_order_) override {
    exec_order = exec_order_;
  }

  /**
   * Support all the interface requirements by nntrainer::Layer
   */

  /**
   * @brief     Finalize creating the layer node
   *
   * @param   input_dims input dimension provided to be used to set output
   * dimensions. if empty function This function must set output dimensions in
   * the given context. Further, context can be used to request weights for the
   * layer, and any extra tensor required for the operation of the layer.
   * @note      After calling this it is not allowed to
   * change properties.
   * @note      No memory allocation must be performed in the initialization
   * step. Any tensor memory required must be requested to the context which
   * will be made available during execution of the layer with the context.
   * @note configureRunContext() is expected to called right after this.
   */
  InitLayerContext
  finalize(const std::vector<TensorDim> &input_dims = {},
           std::array<std::string, 3> tensor_type = {"NCHW", "FP32", "FP32"},
           ml::train::ExecutionMode mode = ml::train::ExecutionMode::TRAIN);

  /**
   * @brief     Refinalize creating the layer node
   *
   * @param   input_dims input dimension provided to be used to set output
   * dimensions. if empty function This function must set output dimensions in
   * the given context. Further, context can be used to request weights for the
   * layer, and any extra tensor required for the operation of the layer.
   * @note      After calling this it is not allowed to
   * change properties.
   * @note      No memory allocation must be performed in the reinitialization
   * step. Any tensor memory required must be requested to the context which
   * will be made available during execution of the layer with the context.
   * @note configureRunContext() is expected to called right after this.
   */
  InitLayerContext refinalize(const std::vector<TensorDim> &input_dims = {});

  void initialize() override { layer->initialize(*run_context); }

  /**
   * @brief     Forward Propagation of a layer
   * @param     training true if training, false if inference
   *
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   */
  void forwarding(bool training = true);

  /**
   * @brief     Incremental forward Propagation of a layer
   * @param     from start step
   * @param     to end step
   * @param     training true if training, false if inference
   *
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   */
  void incremental_forwarding(unsigned int from, unsigned int to,
                              bool training = true);

  /**
   * @brief     calc the derivative to be passed to the previous layer
   *
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   */
  void calcDerivative();

  /**
   * @brief     Calculate the derivative of a layer
   * @details   context provides access to the weights (if any), inputs,
   * outputs, and tensors (if any) for the layer. Input and output dimensions
   * can be access from the inputs/outputs tensors themselves.
   */
  void calcGradient();

  /**
   * @brief this function helps exporting the layer in a predefined format,
   * while workarounding issue caused by templated function type eraser
   *
   * @param     exporter exporter that contains exporting logic
   * @param     method enum value to identify how it should be exported to
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const;

  /**
   * @brief Set the batch for the layer
   * @param     batch Batch value to be set
   * @details Update the run context based on the updated batch size if required
   */
  void setBatch(unsigned int batch);

  /**
   * @brief Update the tensors dimensions of the layer by input dimensions
   * @param     input_dimensions input dimensions of the layer
   * @details Update the dimensions of inputs, outputs, weights, tensors based
   * on the input dimensions
   */
  void updateTensorsByInputDimensions(std::vector<TensorDim> input_dimensions);

  /**
   * @brief   If the current layer can support in-place
   * @return  true if inplace, else false
   */
  bool supportInPlace() const override;

  /**
   * @brief Initialize the in-place settings of the layer
   * @details If it is a layer that supports in-place, the default in-place type
   * is NONE_RESTRICTING, but if there is a RESTRICTING type among the input
   * layers, it is set to NONE in the network_graph.cpp.
   * Layers with exceptional behavior such as No-Operation layers should
   * override this function.
   * @return InPlaceType
   */
  InPlaceType initializeInPlace();
  /**
   * @brief   Notify that this layer will execute in-place
   *
   * @param val in place state for the layer
   */
  void setInPlaceType(InPlaceType val) {
    if (val != InPlaceType::NONE && !supportInPlace())
      throw std::runtime_error("Error setting layer to work in-place");

    inplace_type = val;
  }

  /**
   * @brief   Get if the layer is going to execute in-place
   *
   * @return Inplace type for the layer
   */
  InPlaceType getInPlaceType() const { return inplace_type; }

  /**
   * @brief Get the inplace direction for the layer
   *
   * @return InPlaceDirection
   */
  InPlaceDirection getInPlaceDirection() const;

  /**
   * @brief  check if this layer requires label to be passed
   * @return true if requires a label when training, else false
   * @note   if requireLabel() == true means, for now, that it is endpoint of
   * a graph(numOutlayers == 0). label will be fed to the gradient of hidden
   * if requireLabel is true
   */
  bool requireLabel() const;

  /**
   * Add rest of the helper interfaces required by other internal classes
   */

  /**
   * @brief     Get the trainable property of the underlying object
   *
   * @return boolean true if trainable, else false
   */
  bool supportBackwarding() const { return getLayer()->supportBackwarding(); }

  /**
   * Support interfaces for the properties intercepted from layer
   */

  /**
   * @brief     Get the trainable property of the underlying object
   *
   * @return boolean true if trainable, else false
   */
  bool getTrainable() const override;

  /**
   * @brief     get if the output of this layer must be flatten
   * @retval    flatten value
   */
  bool getFlatten() const;

  /**
   * @brief Get the Shared From property of the layer node
   *
   * @return std::string node name where the weights are borrowed
   */
  std::string getSharedFrom() const;

  /**
   * @brief     get distribute for this layer
   * @retval dist to enable/disable distribute
   */
  bool getDistribute() const;

  /**
   * @brief     get activation for this layer
   * @retval dist to enable/disable distribute
   */
  ActivationType getActivationToBeRealized() const;

  /**
   * @brief     Activation Type Getter
   * @retval    Activation Type.
   */
  ActivationType getActivationType() const;

  /**
   * @brief     Get number of input connections
   * @retval    number of inputs
   */
  unsigned int getNumInputConnections() const;

  /**
   * @brief     Get number of output connections
   * @retval    number of outputs
   */
  unsigned int getNumOutputConnections() const;

  /**
   * @brief     Get number of inputs
   * @retval    number of inputs
   */
  unsigned int getNumInputs() const {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getNumInputs();
  }

  /**
   * @brief     Get number of outputs
   * @retval    number of outputs
   */
  unsigned int getNumOutputs() const {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getNumOutputs();
  }

  /**
   * @brief Get the number of weights
   *
   * @return unsigned int number of weights
   */
  unsigned int getNumWeights() const {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getNumWeights();
  }

  /**
   * @brief Set the Output Layers object
   *
   * @param layers Name of the layers
   */
  void setOutputLayers(const std::vector<std::string> &layers);

  /**
   * @brief check if input shape property is set
   *
   * @return bool true if input shape property has set
   */
  bool hasInputShapeProperty() const;

  /**
   * @brief Get the input dimension
   * @return TensorDim dimension of the input
   */
  const std::vector<TensorDim> getInputDimensions() const;

  /**
   * @brief Get the output dimension
   * @return TensorDim dimension of the output
   */
  const std::vector<TensorDim> getOutputDimensions() const;
  /**
   * @brief Get the Weight object
   * currently, only unittest uses this func.
   *
   * @param idx Identifier of the weight
   * @return Weight& Reference to the weight
   */
  Weight getWeightWrapper(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    if (run_context->weightHasGradient(idx)) {
      return Weight(
        run_context->getWeight(idx), run_context->getWeightGrad(idx),
        run_context->getWeightFP32(idx), run_context->getWeightName(idx));
    } else {
      return Weight(run_context->getWeight(idx), Tensor(), Tensor(),
                    run_context->getWeightName(idx));
    }
  }

  /**
   * @brief Get the Weight object
   *
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight tensor
   */
  Weight &getWeightObject(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getWeightObject(idx);
  }

  /**
   * @brief Get the Weight tensor object
   *
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight tensor
   */
  Tensor &getWeight(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getWeight(idx);
  }

  /**
   * @brief Get the Weight Gradient tensor object
   *
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight grad tensor
   */
  Tensor &getWeightGrad(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getWeightGrad(idx);
  }

  /**
   * @brief Get the Weight object name
   *
   * @param idx Identifier of the weight
   * @return const std::string &Name of the weight
   */
  const std::string &getWeightName(unsigned int idx) override {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getWeightName(idx);
  }

  /**
   * @brief     Get weight data of the layer
   * @retval    weight data of the layer
   * @note      nntrainer assign the vector and if there is no weights, the size
   * of vector is zero
   * @note      layer needs to be finalized before called.
   */
  const std::vector<float *> getWeights() override {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";

    std::vector<float *> weights;
    for (unsigned int idx = 0; idx < getNumWeights(); ++idx) {

      if (getWeight(idx).getDataType() ==
          ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
        _FP16 *data = getWeight(idx).getData<_FP16>();
        float *d = new float[getWeight(idx).size()]();
        weights.emplace_back(d);
        for (unsigned int i = 0; i < getWeight(idx).size(); ++i) {
          weights[idx][i] = static_cast<float>(data[i]);
        }
#else
        throw std::runtime_error("enable-fp16 is not set");
#endif
      } else {
        weights.emplace_back(getWeight(idx).getData());
      }
    }
    return weights;
  }

  /**
   * @brief     Get weight data of the layer
   * @param[out]    weights : float * arrary to store weight data
   * @param[out]    weights_dim : TensorDim for each weights
   * @note      nntrainer assign the vector and if there is no weights, the size
   * of vector is zero
   * @note      layer needs to be finalized before called.
   */
  void getWeights(std::vector<float *> &weights,
                  std::vector<TensorDim> &weight_dim) override {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";

    std::vector<int *> weights_dim;
    for (unsigned int idx = 0; idx < getNumWeights(); ++idx) {
      TensorDim d = getWeight(idx).getDim();
      weights.emplace_back(getWeight(idx).getData());
      weight_dim.emplace_back(d);
    }
    return;
  }
#ifdef ENABLE_FP16
  /**
   * @brief     Get weight data of the layer
   * @retval    weight data of the layer
   * @note      nntrainer assign the vector and if there is no weights, the size
   * of vector is zero
   * @note      layer needs to be finalized before called.
   */
  const std::vector<_FP16 *> getFP16Weights() override {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";

    std::vector<_FP16 *> weights;
    for (unsigned int idx = 0; idx < getNumWeights(); ++idx) {
      weights.emplace_back(getWeight(idx).getData<_FP16>());
    }
    return weights;
  }

  /**
   * @brief     Get weight data of the layer
   * @param[out]    weights : float * arrary to store weight data
   * @param[out]    weights_dim : TensorDim for each weights
   * @note      nntrainer assign the vector and if there is no weights, the size
   * of vector is zero
   * @note      layer needs to be finalized before called.
   */
  void getFP16Weights(std::vector<_FP16 *> &weights,
                      std::vector<TensorDim> &weight_dim) override {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";

    std::vector<int *> weights_dim;
    for (unsigned int idx = 0; idx < getNumWeights(); ++idx) {
      TensorDim d = getWeight(idx).getDim();
      weights.emplace_back(getWeight(idx).getData<_FP16>());
      weight_dim.emplace_back(d);
    }
    return;
  }
#endif

  /**
   * @brief     Set weight data of the layer
   * @note      Size of vector must be the same with number of weights.
   * @note      layer needs to be finalized before called.
   */
  void setWeights(const std::vector<float *> weights) override;

  /**
   * @brief Get the Input tensor object
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the input grad tensor
   */
  Tensor &getInput(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getInput(idx);
  }

  /**
   * @brief Get the Input Grad tensor object
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the input grad tensor
   */
  Tensor &getInputGrad(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getInputGrad(idx);
  }

  /**
   * @brief Get the Output tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output tensor
   */
  Tensor &getOutput(unsigned int idx) {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getOutput(idx);
  }

  /**
   * @brief Get the Output Grad tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output grad tensor
   */
  const Tensor getOutputGrad(unsigned int idx) const {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return run_context->getOutputGrad(idx);
  }

  /**
   * @brief Get the Output Grad unsafe
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output grad tensor
   */
  const Tensor &getOutputGradUnsafe(unsigned int idx) const {
    return run_context->getOutputGradUnsafe(idx);
  }

  /**
   * @brief     read layer Weight & Bias data from file
   * @param file input file stream
   * @param fsu fsu type
   * @param mode Execution mode
   * @param opt_var read optimizer variables
   */
  void read(std::ifstream &file, bool opt_var = false,
            ml::train::ExecutionMode mode = ml::train::ExecutionMode::TRAIN,
            bool fsu = false, size_t start_offset = 0,
            bool read_from_offset = false, int file_fd = -1);

  /**
   * @brief read layer Weight & Bias data from file
   * @param src input file/mmaped stream
   * @param fsu fsu type
   * @param mode Execution mode
   * @param opt_var read optimizer variables
   */
  void read(ReadSource src, bool opt_var = false,
            ml::train::ExecutionMode mode = ml::train::ExecutionMode::TRAIN,
            bool fsu = false, size_t start_offset = 0,
            bool read_from_offset = false);

  /**
   * @brief     save layer Weight & Bias data from file
   * @param file output file stream
   * @param bool save optimizer variables
   */
  void
  save(std::ofstream &file, bool opt_var = false,
       ml::train::ExecutionMode mode = ml::train::ExecutionMode::TRAIN) const;

  /**
   * @brief clear optimizer variable to initial state
   *
   */
  void clearOptVar();

  /**
   * @brief     get loss for the layer
   * @return    loss of the layer
   */
  float getLoss() const;

#ifdef PROFILE
  int forward_event_key;
  int calc_deriv_event_key;
  int calc_grad_event_key;
#endif

  /**
   * @brief   Overriding output stream for layers and it's derived class
   */
  friend std::ostream &operator<<(std::ostream &out, const LayerNode &l);

  /**
   * @brief   Get run layer context
   *
   * @retval  run layer context
   */
  RunLayerContext &getRunContext() {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be configured first!";
    return *run_context;
  }

  /**
   * @brief   Get run layer context
   *
   * @retval  run layer context
   */
  const RunLayerContext &getRunContext() const {
    NNTR_THROW_IF(!run_context, std::runtime_error)
      << __func__ << " layer needs to be configured first!";
    return *run_context;
  }

#ifdef ENABLE_TEST
  /**
   * @brief   Get init layer context
   *
   * @retval  init layer context
   */
  InitLayerContext &getInitContext() {
    NNTR_THROW_IF(!init_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return *init_context;
  }

  /**
   * @brief   Get init layer context
   *
   * @retval  init layer context
   */
  const InitLayerContext &getInitContext() const {
    NNTR_THROW_IF(!init_context, std::runtime_error)
      << __func__ << " layer needs to be finalized first!";
    return *init_context;
  }
#endif // ENABLE_TEST

  /**
   * @brief   check if layer is finalized
   *
   * @retval  bool true if the layer is finalized else false
   */
  bool isFinalized() const {
    if (!run_context)
      return false;

    return true;
  }

  /**
   * @brief Set the Run Context object with given tensor packs
   *
   * @param weights weights
   * @param inputs inputs
   * @param outputs outputs
   * @param tensors tensors
   */
  void configureRunContext(const std::vector<Weight *> &weights,
                           const std::vector<Var_Grad *> &inputs,
                           const std::vector<Var_Grad *> &outputs,
                           const std::vector<Var_Grad *> &tensors,
                           float loss_scale,
                           std::shared_ptr<ContextData> ct_data = nullptr);

  /**
   * @brief Preset modes for printing summary for the layer
   */
  enum class PrintPreset {
    PRINT_NONE = 0,     /**< Print nothing */
    PRINT_SUMMARY,      /**< Print preset including summary information */
    PRINT_SUMMARY_META, /**< Print summary preset that includes meta information
                         */
    PRINT_ALL           /**< Print everything possible */
  };

  /**
   * @brief print using PrintPreset
   *
   * @param out oustream
   * @param preset preset to be used
   */
  void printPreset(std::ostream &out,
                   PrintPreset preset = PrintPreset::PRINT_SUMMARY);

  /**
   * @brief remap identifier inside layer node
   *
   * @param remap_fn function to remap
   */
  void remapIdentifiers(std::function<void(std::string &)> remap_fn);

  /**
   * @brief remap connections(input, output layers ) inside layer node
   *
   * @param remap_fn function to remap
   */
  void
  remapConnections(std::function<void(std::string &, unsigned &)> remap_fn);

  /**
   * @brief create the same node with same properties and types
   *
   * @note this must be done before finalize() as finalize has some potential to
   * change some properties
   * @return LayerNode newly created node
   */
  std::unique_ptr<LayerNode> cloneConfiguration();

  /**
   * @brief Set if the layer needs to do derivative calculation
   *
   * @param nb true if the layer needs to do backwarding, else false
   */
  void needsCalcDerivative(bool nb) {
    NNTR_THROW_IF(nb && !supportBackwarding(), std::invalid_argument)
      << "[Layer] " << getName()
      << " does not support backwarding but is needed";
    needs_calc_derivative = nb;
  }

  /**
   * @brief Set if the layer output needs reinitialization @mixed precsion
   *
   * @param nb true if the layer needs to do reinitialization, eles false
   */
  void reStoreData(bool nb) {
    needs_restore_data = nb;
    run_context->reStoreData(nb);
  }

  /**
   * @brief Set if the layer needs to do calculation of gradients
   *
   * @param nb true if the layer needs to do backwarding, else false
   */
  void needsCalcGradient(bool nb) { needs_calc_gradient = nb; }

  /**
   * @brief Get the layer needs to do calculation of derivatives
   *
   * @return true if the layer needs to do backwarding, else false
   */
  bool needsCalcDerivative() { return needs_calc_derivative; }

  /**
   * @brief Set if the layer needs to do calculation of gradient
   *
   * @param nb true if the layer needs to do backwarding, else false
   */
  bool needsCalcGradient() { return needs_calc_gradient; }

  /**
   * @brief Set if the layer needs to reinitialization @mixed precsion
   *
   * @param nb true if the layer needs reinitialization, eles false
   */
  bool reStoreData() { return needs_restore_data; }

  std::string getComputeEngineType() {
    auto size = props::ComputeEngineTypeInfo::EnumList.size();
    auto data = std::data(props::ComputeEngineTypeInfo::EnumList);
    for (unsigned i = 0; i < size; ++i) {
      if (data[i] == compute_engine) {
        return props::ComputeEngineTypeInfo::EnumStr[i];
      }
    }
    return "cpu";
  }

private:
  /**
   * @brief     Get the Input Layers object
   *
   * @return const std::vector<std::string>
   */
  const std::vector<std::string> getInputLayers() const;

  /**
   * @brief Get the Output Layers object
   *
   * @return const std::vector<std::string>
   */
  const std::vector<std::string> getOutputLayers() const;

  std::unique_ptr<nntrainer::Layer>
    layer; /**< The actual object in the graph node */

  InPlaceType inplace_type; /**< store if the current layer is going to operate
                               in-place */
  bool needs_calc_derivative; /**< cache if this layer needs to do
                                 calcDerivative */
  bool needs_calc_gradient; /**< cache if this layer needs to do calcGradient */

  std::vector<std::unique_ptr<Connection>>
    output_connections; /**< output layer names */

  /**
   * @brief compute_engine Information about the compute backend being used
   *
   */
  ml::train::LayerComputeEngine compute_engine =
    ml::train::LayerComputeEngine::CPU;

#ifdef ENABLE_TEST
  /**
   * @brief   Init context which is stored for debugging issue
   *
   * @note init context is stored only for testing purpose
   */
  std::unique_ptr<InitLayerContext> init_context;
#endif // ENABLE_TEST

  std::unique_ptr<RunLayerContext>
    run_context; /**< context required for running/execution of the layer. This
will also contain the properties of the layer. The properties will be copied
upon final creation. Editing properties of the layer after init will not the
properties in the context/graph unless intended. */

  using PropsType =
    std::tuple<props::Name, props::Distribute, props::Trainable,
               std::vector<props::InputConnection>,
               std::vector<props::InputShape>, props::SharedFrom,
               props::ClipGradByGlobalNorm, props::Packed, props::WeightDtype,
               props::LossScaleForMixed, props::ComputeEngine>;

  using RealizationPropsType = std::tuple<props::Flatten, props::Activation>;
  /** these realization properties results in addition of new layers, hence
   * skipped in generation of model architecture as the correspondingly layer
   * itself is added. Distribute is also a property which is realized, but as it
   * doesn't add new layer, it is saved. */

  /**
   * These properties are set for the layer by the user but are intercepted
   * and used in the node which forms the basic element of the graph.
   */
  std::unique_ptr<PropsType> layer_node_props; /**< properties for the node */
  std::unique_ptr<RealizationPropsType>
    layer_node_props_realization;    /**< properties for the node */
  std::unique_ptr<props::Loss> loss; /**< loss */
  ExecutionOrder exec_order; /**< order/location of execution for this node
                                   in forward and backwarding operations */

  bool needs_restore_data; /**< cache if this layer needs reinitialization
                                 output  */

  std::array<TensorDim::DataType, 2> data_type;

  /**
   * @brief   Get the effective layer managed by this layer node
   *
   * @details this is layer inside the distribution layer if this layer node
   * is distributed.
   */
  const nntrainer::Layer *getLayer() const;

  /**
   * @brief   Get the effective layer managed by this layer node
   *
   * @details this is layer inside the distribution layer if this layer node
   * is distributed.
   */
  nntrainer::Layer *getLayer();

  /**
   * @brief anchor point to override if PRINT_SHAPE_INFO is enabled for
   * Layer::print()
   */
  void printShapeInfo(std::ostream &out);

  /**
   * @brief anchor point to override if PRINT_METRIC is enabled for
   * Layer::print()
   */
  void printMetric(std::ostream &out);

  /**
   * @brief     Print layer related information. Do not override without clear
   * reason. It is recommended to override printShapeInfo, printPropertiesMeta,
   * printProperties, printMetric instead
   * @param[in] out outstream
   * @param[in] flags combination of LayerPrintOption
   */
  void print(std::ostream &out, unsigned int flags = 0);
};

/**
 * @brief LayerNode creator with constructor
 *
 * @params[in] type Type of the layer to be constructed
 * @params[in] properties Properties of the layer
 * @params[in] compute engine for the layer to run on
 */
std::unique_ptr<LayerNode>
createLayerNode(const ml::train::LayerType &type,
                const std::vector<std::string> &properties = {});

/**
 * @brief LayerNode creator with constructor
 *
 * @params[in] type Type of the layer to be constructed
 * @params[in] properties Properties of the layer
 */
std::unique_ptr<LayerNode>
createLayerNode(const std::string &type,
                const std::vector<std::string> &properties = {});

/**
 * @brief LayerNode creator with constructor
 *
 * @params[in] layer Already constructed layer
 * @params[in] properties Properties of the layer
 * @params[in] compute engine for the layer to run on
 */
std::unique_ptr<LayerNode>
createLayerNode(std::unique_ptr<nntrainer::Layer> &&layer,
                const std::vector<std::string> &properties);

} // namespace nntrainer
#endif // __LAYER_NODE_H__

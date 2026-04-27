// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   layer_context.h
 * @date   10 June 2021
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer context for each layer
 */

#ifndef __LAYER_CONTEXT_H__
#define __LAYER_CONTEXT_H__

#include <memory>
#include <vector>

#include <common_properties.h>
#include <context_data.h>
#include <layer.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <tensor_wrap_specs.h>
#include <weight.h>

namespace nntrainer {

class Var_Grad;

/**
 * @class   Layer Context class for all layers
 * @brief   Class for Layer context
 *
 * @details This provides for the layer initialization. This context will not
 * contain any structures which allow allocation of memory or support to
 * allocate any new memory, but rather only support storing specifications based
 * on which memory will be allocated later.
 */
class InitLayerContext {
public:
  /**
   * @brief Construct a new Init Layer Context object
   *
   * @param dim Input dimensions for the layer
   * @param req_out_connected bool vector to tell if requested output is
   * trainable or not
   * @param is_inplace_ true if the context is inplacable
   * @param name name
   * @param prefix_ prefix
   * @param max_norm max norm
   * @param tensor_type array including tensor format and weight, activation
   * type.
   * @param loss_scale loss scale value for mixed precision training
   * @param mode execution mode.
   */
  InitLayerContext(
    const std::vector<TensorDim> &dim,
    const std::vector<bool> &req_out_connected, bool is_inplace_,
    const std::string &n = "", const std::string &prefix_ = "",
    const float max_norm = 0.0,
    std::array<std::string, 3> tensor_type_ = {"NCHW", "FP32", "FP32"},
    const float loss_scale = 1.0,
    ml::train::ExecutionMode mode = ml::train::ExecutionMode::TRAIN,
    ml::train::LayerComputeEngine engine = ml::train::LayerComputeEngine::CPU);
  /**
   * @brief   get Tensor Format of Layer
   *
   * @return Tensor Format of the layer
   */
  TensorDim::Format getFormat() {
    return str_converter<enum_class_prop_tag, nntrainer::TensorFormatInfo>::
      from_string(tensor_type[0]);
  };

  /**
   * @brief   get Tensor DataType of the Weight
   *
   * @return Tensor DataType of the the Weight
   */
  TensorDim::DataType getWeightDataType() {
    return str_converter<enum_class_prop_tag, nntrainer::TensorDataTypeInfo>::
      from_string(tensor_type[1]);
  };

  /**
   * @brief   get Tensor DataType of the Activation
   *
   * @return Tensor DataType of the the Activation
   */
  TensorDim::DataType getActivationDataType() {
    return str_converter<enum_class_prop_tag, nntrainer::TensorDataTypeInfo>::
      from_string(tensor_type[2]);
  };

  /**
   * @brief   get Layer Compute Engine Type
   *
   * @return Engine Engine Type
   */
  ml::train::LayerComputeEngine getComputeEngineType() { return engine; };

  /**
   * @brief   get name by the layer
   *
   * @return name of the layer
   */
  const std::string &getName() const { return name; }

  /**
   * @brief   get Execution Mode
   *
   * @return Mode Execution Mode : ml::train::ExecutionMode::INFERNECE |
   * ml::train::ExecutionMode::TRAIN
   */
  const ml::train::ExecutionMode &getExecutionMode() const { return mode; }

  /**
   * @brief Get the number of inputs for the layer
   *
   * @return unsigned int number of inputs
   */
  unsigned int getNumInputs() const { return input_dim.size(); }

  /**
   * @brief Get the number of requested outputs for the layer
   *
   * @return unsigned int number of requested outputs
   */
  unsigned int getNumRequestedOutputs() const;

  /**
   * @brief Get the Input Dimensions object
   *
   * @return const std::vector<TensorDim>& Input dimensions
   */
  const std::vector<TensorDim> &getInputDimensions() const { return input_dim; }

  /**
   * @brief Retrieves the data type of input tensor at the given index
   *
   * @return The data type of the input tensor
   */
  const TensorDim::DataType getInputDataType(int idx) const {
    return input_dim[idx].getDataType();
  }

  /**
   * @brief Get the Mutable Input Dimensions object
   *
   * @return std::vector<TensorDim>& Input dimensions
   */
  std::vector<TensorDim> &getMutableInputDimensions() { return input_dim; }

  /**
   * @brief Set Data Type for Input Dimensions
   *
   * @param ty data type to set
   */
  void setInputDataType(TensorDim::DataType ty) {
    for (auto &d : input_dim)
      d.setDataType(ty);
  }

  /**
   * @brief Set the Dim Flag to retrieve effective dimension
   *
   * @param dim_flag_ dimension bit to calculate, rightmost is width
   */
  void
  setEffDimFlagInputDimension(unsigned int idx,
                              const std::bitset<TensorDim::MAXDIM> &dim_flag_) {
    input_dim[idx].setEffDimFlag(dim_flag_);
  }

  /**
   * @brief Set the dynamic Dim Flag to retrieve dynamic dimension (that can
   * change during running)
   *
   * @param dim_flag_ dimension bit to calculate, rightmost is width
   */
  void
  setDynDimFlagInputDimension(unsigned int idx,
                              const std::bitset<TensorDim::MAXDIM> &dim_flag_) {
    input_dim[idx].setDynDimFlag(dim_flag_);
  }

  /**
   * @brief Set the Output Dimensions object
   *
   * @param out_dim the output dimension to set to
   */
  void setOutputDimensions(const std::vector<TensorDim> &out_dim);

  /**
   * @brief Request a new weight for the layer
   *
   * @param dim dimension of Variable of the weight
   * @param init initializer for the weight
   * @param reg regularizer for the weight
   * @param reg_const regularization constant for the weight
   * @param name name of the weight
   * @param trainable if the weight is trainable (require gradient or not)
   * @param is_virtual if the weight is virtual (not allocate)
   * @return unsigned int index of the weight for its getter
   *
   * @todo Consider providing a guarantee that the returned indices will always
   * start from 0 and will always be incremental.
   */
  unsigned int requestWeight(const TensorDim &dim, const Initializer init,
                             const WeightRegularizer reg, const float reg_const,
                             const float decay, const std::string &name,
                             bool trainable = true, bool is_virtual = false,
                             unsigned int out_axis = 3) {

    /** @note : We assumes the gradient type is same with Activation data
     * type.*/
    TensorDim dim_g(dim);

    dim_g.setDataType(getActivationDataType());

    weights_spec.emplace_back(
      dim, dim_g, init, reg, reg_const, decay, clip_by_global_norm, trainable,
      prefix + ":" + name, out_axis, loss_scale,
      (getWeightDataType() != ml::train::TensorDim::DataType::FP32),
      is_virtual);
    return weights_spec.size() - 1;
  }

  /**
   * @brief Request a new weight for the layer
   *
   * @param dim dimension of Variable of the weight
   * @param dim_g dimension of Gradient of the weight
   * @param init initializer for the weight
   * @param reg regularizer for the weight
   * @param reg_const regularization constant for the weight
   * @param name name of the weight
   * @param trainable if the weight is trainable (require gradient or not)
   * @return unsigned int index of the weight for its getter
   *
   * @todo Consider providing a guarantee that the returned indices will always
   * start from 0 and will always be incremental.
   */
  unsigned int requestWeight(const TensorDim &dim, const TensorDim &dim_g,
                             const Initializer init,
                             const WeightRegularizer reg, const float reg_const,
                             const float decay, const std::string &name,
                             bool trainable = true, unsigned int out_axis = 3,
                             bool is_virtual = false) {

    /** @note : We assumes the gradient type is same with Activation data
     * type.*/
    weights_spec.emplace_back(
      dim, dim_g, init, reg, reg_const, decay, clip_by_global_norm, trainable,
      prefix + ":" + name, out_axis, loss_scale,
      (getWeightDataType() != ml::train::TensorDim::DataType::FP32),
      is_virtual);
    return weights_spec.size() - 1;
  }

  /**
   * @brief Request a new weight for the layer
   *
   * @param spec tensor spec
   * @return unsigned int index of the weight for its getter
   *
   * @todo Consider providing a guarantee that the returned indices will always
   * start from 0 and will always be incremental.
   */
  unsigned int requestWeight(const WeightSpec &spec) {
    weights_spec.emplace_back(spec);
    return weights_spec.size() - 1;
  }

  /**
   * @brief Request a new tensor for the layer
   *
   * @param dim dimension of the tensor
   * @param trainable if the tensor is trainable (require gradient or not)
   * @param name name of the tensor
   * @param lifespan lifespan of the tensor
   * @param private_ if custom tensor should not be shared, and only for soleuse
   * @return unsigned int index of the tensor for its getter
   *
   * @todo Consider providing a guarantee that the returned indices will always
   * start from 0 and will always be incremental.
   */
  unsigned int requestTensor(
    const TensorDim &dim, const std::string &name,
    const Initializer init = Initializer::NONE, bool trainable = false,
    TensorLifespan lifespan = TensorLifespan::ITERATION_LIFESPAN,
    bool private_ = true,
    ml::train::LayerComputeEngine engine = ml::train::LayerComputeEngine::CPU) {
    const auto &prefix_ = private_ ? this->name : this->prefix;
    tensors_spec.emplace_back(dim, init, trainable, prefix_ + ":" + name,
                              lifespan, engine);
    return tensors_spec.size() - 1;
  }

  /**
   * @brief Specification of the tensors
   *
   */
  typedef VarGradSpec TensorSpec;

  /**
   * @brief Request a new tensor for the layer
   *
   * @param spec tensor spec
   * @return unsigned int index of the tensor for its getter
   *
   * @todo Consider providing a guarantee that the returned indices will always
   * start from 0 and will always be incremental.
   */
  unsigned int requestTensor(const TensorSpec &spec) {
    tensors_spec.emplace_back(spec);
    return tensors_spec.size() - 1;
  }

  /**
   * @brief Get the current weights spec
   *
   * @return The current weights spec
   */
  const std::vector<WeightSpec> &getWeightsSpec() const { return weights_spec; }

  /**
   * @brief Get the number of requested weights
   *
   * @return The current number of requested weights
   */
  unsigned int getNumWeights() const { return weights_spec.size(); }

  /**
   * @brief Get the current tensors spec
   *
   * @return The current tensors spec
   */
  const std::vector<TensorSpec> &getTensorsSpec() const { return tensors_spec; }

  /**
   * @brief Get the number of requested tensors objects
   *
   * @return unsigned int number of requested tensors
   */
  unsigned int getNumTensors() const { return tensors_spec.size(); }

  /**
   * @brief create var grad specification with output default
   *
   * @param dim dimension
   * @param name name
   * @param ls variable lifespan
   * @param grad_ls gradient lifespan
   * @return VarGradSpecV2 var grad specification
   */
  static VarGradSpecV2
  outSpec(const TensorDim &dim, const std::string &name = "out",
          TensorLifespan ls = TensorLifespan::FORWARD_FUNC_LIFESPAN,
          TensorLifespan grad_ls = TensorLifespan::CALC_GRAD_DERIV_LIFESPAN);

  /**
   * @brief request outputs
   *
   * @param out_specs pack of out specification, name will be automatically
   * indexed to prevent name clash
   */
  void requestOutputs(std::vector<VarGradSpecV2> &&out_specs);

  /**
   * @brief Get the Out Specs object
   *
   * @return std::vector<VarGradSpecV2> out specification
   */
  const std::vector<VarGradSpecV2> &getOutSpecs() const;

  /**
   * @brief Validate the context
   *
   * @return true if validated, else false
   * @note this must be called before passing a context to a layer for finalize
   */
  bool validate() {
    if (input_dim.empty()) {
      return false;
    }

    for (auto const &dim : input_dim) {
      if (dim.getDataLen() == 0) {
        return false;
      }
    }

    if (name.empty()) {
      return false;
    }

    return true;
  }

  /**
   * @brief   check if the layer is expected to run in-place
   *
   * @return true if in-place, else false
   */
  bool getInPlace() const { return is_inplace; }

  /**
   * @brief   get Initial value of Loss_Scale. This is set to RunLayerContext
   * and updated
   *
   * @return loss_scale
   */
  float getLossScale() const { return loss_scale; }

  /**
   * @brief   get Mixed Precision Training. If the weight is not the FP32, then
   * it is mixed training.
   *
   * @return true if it is mixed training
   */
  bool isMixedTraining() { return istrequal(tensor_type[1], "FP32"); }

private:
  std::vector<TensorDim> input_dim; /**< Input dimensions for the layer */
  bool is_inplace;           /**< if the layer is expected to run in-place */
  float clip_by_global_norm; /**< max norm value for clip by norm */

  std::vector<VarGradSpecV2> output_specs; /**< Specification for the output */
  std::vector<WeightSpec> weights_spec;    /**< Specification for the weights */
  std::vector<TensorSpec>
    tensors_spec; /**< Specification for the var_grad (trainable/non-trainable
                     variables) */

  std::vector<bool> req_out_is_connected;
  /**< a bool vector to tell if requested out is actually connected to others */
  std::string name;   /**< name of the layer */
  std::string prefix; /**< prefix of the layer */
  std::array<std::string, 3> tensor_type;
  float loss_scale; /**< loss_scale value */
  ml::train::ExecutionMode mode;
  ml::train::LayerComputeEngine engine;
};

/**
 * @class   Layer Context class for all layers
 * @brief   Class for Layer context
 *
 * @details This provides for the layer executing. This context will contain
 * structures with memory allocated or support to allocate any new memory, but
 * rather only support storing specifications based on which memory will be
 * allocated later.
 *
 * @todo Check the caller of the getTensor() and set restrictions on the tensors
 * to be accessed based on which function is requesting it.
 */
class RunLayerContext {
public:
  /**
   * @brief Construct a new Run Layer Context object
   *
   */
  RunLayerContext() :
    loss(0.0), is_inplace(false), loss_scale(1.0), restoreData(false) {}

  /**
   * @brief Construct a new Run Layer Context object
   *
   */
  RunLayerContext(const std::string &name, bool is_inplace_) :
    RunLayerContext() {
    is_inplace = is_inplace_;
    std::get<props::Name>(props).set(name);
  }

  /**
   * @brief Construct a new Run Layer Context object
   *
   */
  RunLayerContext(const std::string &name, bool is_inplace_,
                  float loss_scale_) :
    RunLayerContext() {
    is_inplace = is_inplace_;
    std::get<props::Name>(props).set(name);
    loss_scale = loss_scale_;
  }

  /**
   * @brief Construct a new Run Layer Context object
   *
   * @param name name of the layer
   * @param trainable if the layer is trainable
   * @param l loss of the layer
   * @param is_inplace_ execution in-place of the layer
   * @param loss_scale loss_scale of the layer
   * @param w weights of the layer
   * @param in inputs of the layer
   * @param out outputs of the layer
   * @param t extra tensors of the layer
   */
  RunLayerContext(const std::string &name, bool trainable, float l,
                  bool is_inplace_, float loss_scale_,
                  std::shared_ptr<ContextData> ct_data, bool restoreData_,
                  const std::vector<Weight *> &w,
                  const std::vector<Var_Grad *> &in,
                  const std::vector<Var_Grad *> &out,
                  const std::vector<Var_Grad *> &t);

  /**
   * @brief Get the Weight tensor object
   *
   * @param w out tensor
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight tensor
   */
  void getWeight(Tensor &w, unsigned int idx) {
    Tensor &t_w = weights[idx]->getVariableRef();

    if (t_w.getDataType() == Tdatatype::FP32 ||
        t_w.getDataType() == Tdatatype::FP16 ||
        t_w.getDataType() == Tdatatype::BCQ ||
        t_w.getDataType() == Tdatatype::Q4_K) {
      w = t_w;
      return;
    }

    unsigned int base_idx = 0;
    Tdatatype o_t = getOutput(base_idx).getDataType();

    if (w.empty()) {
      TensorDim d = t_w.getDim();
      d.setDataType(o_t);
      w = Tensor(d, true);
    }

    return;
  }

  /**
   * @brief Get the Weight tensor object
   *
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight tensor
   */
  Tensor &getWeight(unsigned int idx) const;

  /**
   * @brief Get the Weight Gradient tensor object
   *
   * @note this method returns the fresh gradient to be filled
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight grad tensor
   */
  Tensor &getWeightGrad(unsigned int idx) const;

  /**
   * @brief Get the Weight Gradient tensor object
   *
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight grad tensor
   */
  Tensor &getWeightFP32(unsigned int idx) const;

  /**

   * @brief Get the Weight Optimizer Variable tensor object
   *
   * @param idx Identifier of the weight
   * @param jdx Identifier of the weight optimizer variable
   * @return Tensor& Reference to the weight grad tensor
   */
  Tensor &getWeightOptVar(unsigned int idx, unsigned int jdx) const;

  /**
   * @brief Get the Weight name
   *
   * @param idx Identifier of the weight
   * @return name of the weight
   */
  const std::string &getWeightName(unsigned int idx) const;

  /**
   * @brief check if the weight has gradient
   *
   * @param idx Identifier of the weight
   * @return true if weight has gradient, else false
   */
  bool weightHasGradient(unsigned int idx) const;

  /**
   * @brief Get the Output tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output tensor
   */
  Tensor &getOutput(unsigned int idx);

  /**
   * @brief Get the Output tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output tensor
   */
  const Tensor &getOutput(unsigned int idx) const;

  /**
   * @brief Get the Output Grad tensor object
   *
   * @param idx Identifier of the output
   * @return Read-only output grad tensor, if derivative does not have
   * gradient, return a temporary, initialized to zero
   */
  const Tensor getOutputGrad(unsigned int idx) const;

  /**
   * @brief Get the Output Grad tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output grad tensor, this is valid only if
   * the given output is trainable
   *
   * @note recommended to NOT use this function as a layer developer but rather
   * use getOutputGrad().
   */
  Tensor &getOutputGradUnsafe(unsigned int idx);

  /**
   * @brief check if the weight has gradient
   *
   * @param idx Identifier of the weight
   * @return true if weight has gradient, else false
   */
  bool outputHasGradient(unsigned int idx) const;

  /**
   * @brief Get the incoming Derivative tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor output derivative tensor, if derivative does not have
   * gradient, return a temporary, initialized to zero
   */
  const Tensor getIncomingDerivative(unsigned int idx) const;

  /**
   * @brief Get the Input tensor object
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the input grad tensor
   */
  Tensor &getInput(unsigned int idx);

  /**
   * @brief Get the Input tensor object
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the input grad tensor
   */
  const Tensor &getInput(unsigned int idx) const;

  /**
   * @brief Get the Input Grad tensor object
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the input grad tensor
   */
  Tensor &getInputGrad(unsigned int idx);

  /**
   * @brief check if the weight has gradient
   *
   * @param idx Identifier of the weight
   * @return true if weight has gradient, else false
   */
  bool inputHasGradient(unsigned int idx) const;

  /**
   * @brief Get the outgoing Derivative tensor object
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the input derivative tensor
   */
  Tensor &getOutgoingDerivative(unsigned int idx);

  /**
   * @brief Get the Tensor object
   *
   * @param idx Identifier of the tensor
   * @return Tensor& Reference to the tensor
   */
  Tensor &getTensor(unsigned int idx);

  /**
   * @brief Get the Tensor object
   *
   * @param idx Identifier of the tensor
   * @return Tensor& Reference to the tensor
   */
  const Tensor &getTensor(unsigned int idx) const;

  /**
   * @brief Get the Tensor Grad object
   *
   * @param idx Identifier of the tensor
   * @return Tensor& Reference to the tensor grad tensor
   */
  Tensor &getTensorGrad(unsigned int idx);

  /**
   * @brief Get the Tensor Grad object
   *
   * @param idx Identifier of the tensor
   * @return Tensor& Reference to the tensor grad tensor
   */
  const Tensor &getTensorGrad(unsigned int idx) const;

  /**
   * @brief check if the tensor has gradient
   *
   * @param idx Identifier of the tensor
   * @return true if tensor has gradient, else false
   */
  bool tensorHasGradient(unsigned int idx) const;

  /**
   * @brief check if the weight is burrowed from others so it is dependent
   *
   * @param idx index
   * @return bool true if weight is burrowed from outside
   */
  bool isWeightDependent(unsigned int idx) const;

  /**
   * @brief check current gradient is first access
   * @note for now, it equivalent to weight last access, so this value is
   * accessible for non-trainable weights as well. This is in terms of execution
   * order.
   *
   * @param idx index
   * @return bool true if first access
   */
  bool isGradientFirstAccess(unsigned int idx) const;

  /**
   * @brief check current gradient is last access
   * @note for now, it equivalent to weight last access, so this value is
   * accessible for non-trainable weights as well. This is in terms of execution
   * order.
   *
   * @param idx index
   * @return bool true if last access
   */
  bool isGradientLastAccess(unsigned int idx) const;

  /**
   * @brief check if the gradient is to be clipped by global norm
   *
   * @param idx index
   * @return bool true if it is to be clipped else false
   */
  bool isGradientClipByGlobalNorm(unsigned int idx) const;

  /**
   * @brief check if the weight is mixed precsion
   *
   * @param idx index
   * @return bool true if it is mixed precision
   */
  bool isMixedPrecision(unsigned int idx) const;

  /**
   * @brief Get the tensor name
   *
   * @param idx Identifier of the tensor
   * @return name of the tensor
   */
  const std::string &getTensorName(unsigned int idx) const;

  /**
   * @brief Get the number of Outputs tensor objects
   *
   * @return unsigned int number of output tensors
   */
  unsigned int getNumOutputs() const;

  /**
   * @brief Get the number of inputs tensor objects
   *
   * @return unsigned int number of input tensors
   */
  unsigned int getNumInputs() const;

  /**
   * @brief Get the number of weights tensor objects
   *
   * @return unsigned int number of weight tensors
   */
  unsigned int getNumWeights() const;

  /**
   * @brief Get the Number of Weight Optimizer Variable tensor object
   *
   * @param idx Identifier of the weight
   * @return unsigned int Number of the weight optimizer variable
   */
  unsigned int getNumWeightOptVar(unsigned int idx) const;

  /**
   * @brief Get the number of requested tensors objects
   *
   * @return unsigned int number of requested tensors
   */
  unsigned int getNumTensors() const;
  /**
   * @brief Set the batch for the run context
   *
   * @param batch Update batch size
   */
  void setBatch(unsigned int batch);

  /**
   * @brief Update the dimensions for a requested tensor
   *
   * @param idx index of the tensor (identifier)
   * @param batch Updated batch size
   */
  void updateTensor(unsigned int idx, unsigned int batch);

  /**
   * @brief Update the dimensions for a requested input
   *
   * @param idx index of the input (identifier)
   * @param dim dimension to be updated
   */
  void updateInput(unsigned int idx, TensorDim dim);

  /**
   * @brief Update the dimensions for a requested output
   *
   * @param idx index of the output (identifier)
   * @param dim dimension to be updated
   */
  void updateOutput(unsigned int idx, TensorDim dim);

  /**
   * @brief Update the dimensions for a requested tensor
   *
   * @param idx index of the tensor (identifier)
   * @param dim dimension to be updated
   */
  void updateTensor(unsigned int idx, TensorDim dim);

  /**
   * @brief   Get weight object for the weights
   *
   * @param idx index of the weight (identifier)
   * @return weight object
   */
  Weight &getWeightObject(unsigned int idx);

  /**
   * @brief   check if the label is available
   *
   * @param idx Identifier of the input
   * @return true if label is available else false
   */
  bool isLabelAvailable(unsigned int idx) const;

  /**
   * @brief   Get label tensor
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the label tensor
   */
  Tensor &getLabel(unsigned int idx);

  /**
   * @brief   update loss by the layer
   *
   * @param val updated loss value
   * @note loss value is only used for loss layers. For non-loss layers, setting
   * this value will have no change on the behavior of the model.
   */
  void setLoss(float val) { loss = val; }

  /**
   * @brief   update loss by the layer
   *
   * @return loss of the layer
   * @note does not includes the regularization loss.
   */
  float getLoss() const { return loss; }

  /**
   * @brief   get regularization loss of the layer
   *
   * @return regularization loss of the layer
   */
  float getRegularizationLoss() const {
    float loss_ = 0;
    for (unsigned int idx = 0; idx < getNumWeights(); idx++) {
      loss_ += getWeightRegularizationLoss(idx);
    }
    return loss_;
  }

  std::shared_ptr<ContextData> getContextData() { return ct_data; }

  /**
   * @brief Get the ComputeOps table for this layer's backend (may be null).
   *
   * Shortcut for `getContextData()->getComputeOps()`. Returns nullptr if
   * no ContextData is attached, in which case callers fall back to the
   * global table via nntrainer::getComputeOps().
   */
  ComputeOps *getComputeOps() const {
    return ct_data ? ct_data->getComputeOps() : nullptr;
  }

  /**
   * @brief   get name by the layer
   *
   * @return name of the layer
   */
  const std::string &getName() const { return std::get<props::Name>(props); }

  /**
   * @brief   get trainable by the layer
   *
   * @return trainable of the layer
   */
  bool getTrainable() const { return std::get<props::Trainable>(props); }

  /**
   * @brief   check if run context is set and is ready to use
   *
   * @return true if ready, else false
   */
  bool readyToUse() const;

  /**
   * @brief   validates the run context after run
   *
   * @param skip_input  skip verifying the input
   * @param skip_label  skip verifying the label
   *
   * @return true if ready, else false
   */
  bool validate(bool skip_input = false, bool skip_label = false);

  /**
   * @brief   check if the layer is expected to run in-place
   *
   * @return true if in-place, else false
   */
  bool getInPlace() const { return is_inplace; }

  /**
   * @brief   get layer weights
   *
   * @return weights
   */
  std::vector<Weight *> getWeights() { return weights; }

  /**
   * @brief get loss scale
   * @return loss scale
   */
  float getLossScale() { return loss_scale; }

  /**
   * @brief   set Loss_Scale.
   *
   * @return loss_scale
   */
  void setLossScale(float scale) {
    loss_scale = scale;
    for (auto w : weights) {
      w->setLossScale(scale);
    }
  }

  /**
   * @brief   set Output Zero Flag.
   *
   */
  void reStoreData(bool nb) { restoreData = nb; }

  /**
   * @brief   get Output Zero Flag.
   *
   */
  bool reStoreData() { return restoreData; }

private:
  std::tuple<props::Name, props::Trainable> props; /**< props of the layer */
  std::shared_ptr<ContextData> ct_data;
  float loss;       /**< loss of the layer */
  bool is_inplace;  /**< if the layer is expected to run in-place */
  float loss_scale; /**< loss_scale of the layer */
  bool restoreData; /**< reset output for mixed precsion */

  std::vector<Weight *> weights;   /**< weights of the layer */
  std::vector<Var_Grad *> inputs;  /**< inputs of the layer */
  std::vector<Var_Grad *> outputs; /**< outputs of the layer */
  std::vector<Var_Grad *> tensors; /**< tensors of the layer */

#ifdef DEBUG
  std::map<std::string, const void *>
    tensor_map; /**< map of tensor name to tensor address */
#endif

  /**
   * @brief Get regularization loss for the weight
   *
   * @param idx Identifier of the weight
   * @return float Value of the loss
   */
  float getWeightRegularizationLoss(unsigned int idx) const;
};

} // namespace nntrainer
#endif // __LAYER_CONTEXT_H__

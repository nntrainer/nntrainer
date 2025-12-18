// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   onnx_interpreter.h
 * @date   12 February 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is onnx converter interface for c++ API
 */

#ifndef __ONNX_INTERPRETER_H__
#define __ONNX_INTERPRETER_H__
#include "compiler_fwd.h"
#ifdef ENABLE_ONNX_INTERPRETER

#include <app_context.h>
#include <interpreter.h>
#include <iostream>
#include <layer.h>
#include <layer_node.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <onnx.pb.h>
#include <string>
#include <util_func.h>

namespace nntrainer {

using NodeHandler =
  std::function<void(const onnx::NodeProto &, GraphRepresentation &)>;

/**
 * @brief ONNX Interpreter class for converting onnx model to nntrainer model.
 *
 */
class ONNXInterpreter : public GraphInterpreter {
public:
  /**
   * @brief Construct a new ONNXInterpreter object
   *
   */
  ONNXInterpreter() { registerNodeHandlers(); };

  /**
   * @brief Destroy the ONNXInterpreter object
   *
   */
  ~ONNXInterpreter(){};

  void handleUnaryOp(const onnx::NodeProto &node,
                     GraphRepresentation &representation,
                     const std::string &op_type,
                     std::vector<std::string> &props);

  void handleBinaryOp(const onnx::NodeProto &node,
                      GraphRepresentation &representation,
                      const std::string &op_type,
                      std::vector<std::string> &props);

  void registerBasicUnaryOp(const std::string &op_type);

  void registerBasicBinaryOp(const std::string &op_type);

  void registerNodeHandlers();

  std::vector<std::string> createOutputRemap(const onnx::NodeProto &node);

  /**
   * @brief Get data type from onnx type enum.
   *
   * @param onnx_type
   * @return std::string tensor data type
   */
  std::string getDataTypeFromONNX(int onnx_type);

  /**
   * @brief Load the onnx model from file
   *
   */
  void loadONNXModel(const std::string &file_path);

  /**
   * @brief Load inputs and weights from onnx model to nntrainer model.
   *
   */
  void loadInputsAndWeights(GraphRepresentation &representation);

  /**
   * @brief Load operations from onnx model to nntrainer model.
   *
   */
  void loadOperations(GraphRepresentation &representation);

  /**
   * @copydoc GraphInterpreter::serialize(const GraphRepresentation
   * representation, const std::string &out)
   */
  void serialize(const GraphRepresentation &representation,
                 const std::string &out) override;

  /**
   * @copydoc GraphInterpreter::deserialize(const std::string &in)
   */
  GraphRepresentation deserialize(const std::string &in) override;

  /**
   * @brief Clean the name of the layer to be used in nntrainer model
   *
   * @param name name of the layer
   */
  std::string cleanName(std::string name);

  /**
   * @brief Extracts an attribute value from an ONNX node and converts it to a
   * string for NNTrainer's property format.
   */
  std::string extractAttribute(const onnx::NodeProto &node,
                               const std::string &attr_name,
                               const std::string &prefix = "",
                               const std::string &separator = ",",
                               const int start_offset = 0,
                               const bool need_default_value = false,
                               const std::string &default_value = "");

  /**
   * @brief Extracts an attribute value from an Constant node and converts it to
   * a string for NNTrainer's property format.
   */
  std::string extractTensorAttribute(const onnx::NodeProto &node,
                                     const std::string &attr_name,
                                     const std::string &prefix = "",
                                     const std::string &separator = ",",
                                     const int start_offset = 0,
                                     const int alpha = 0);

  /**
   * @brief Transform dimension string to nntrainer's format.
   *
   * @param shape ONNX TensorShapeProto
   */
  std::string transformDimString(onnx::TensorShapeProto shape);

  // /**
  //  * @brief Transform dimension string to nntrainer's format.
  //  *
  //  * @param initializer ONNX TensorProto
  //  */
  std::string transformDimString(onnx::TensorProto initializer);

private:
  std::unordered_map<std::string, NodeHandler>
    NodeHandlers;              // node handlers function map
  onnx::ModelProto onnx_model; // parsed onnx model
  std::unique_ptr<ml::train::Model>
    nntrainer_model; // converted nntrainer model
  std::unordered_map<std::string, onnx::TensorProto>
    initializers; // initializers are used to identify weights
  std::unordered_map<std::string, onnx::NodeProto>
    constantTensors; // Constant tensor list
  std::unordered_map<std::string, onnx::TensorShapeProto>
    inputDims; // input dims
  std::unordered_map<std::string, std::string>
    layerOutputMap; // key: name of output, value: name of layer

  std::unordered_map<std::string, std::string> layerKeyMap = {
    {"Add", "add"},
    {"Sub", "subtract"},
    {"Mul", "multiply"},
    {"Div", "divide"},
    {"MatMul", "matmul"},
    {"ReduceMean", "reduce_mean"},
    {"Reshape", "reshape"},
    {"Transpose", "permute"},
    {"Cast", "cast"},
    {"Pow", "pow"},
    {"Sqrt", "sqrt"},
    {"Softmax", "activation"},
    {"Sigmoid", "activation"},
    {"Relu", "activation"},
    {"Identity", "identity"},
    {"Unsqueeze", "reshape"},
    {"Gather", "gather"},
    {"Cosine", "cosine"},
    {"Sine", "sine"},
    {"Tangent", "tangent"},
    {"Slice", "slice"},
    {"Neg", "negative"},
    {"Concat", "concat"}};

  std::unordered_map<std::string, std::string> activationKeyMap = {
    {"Softmax", "softmax"}, {"Relu", "relu"}, {"Sigmoid", "sigmoid"}};
};

} // namespace nntrainer

#endif // ENABLE_ONNX_INTERPRETER
#endif // __ONNX_INTERPRETER_H__

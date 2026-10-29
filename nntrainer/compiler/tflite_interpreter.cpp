// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file tflite_interpreter.cpp
 * @date 12 April 2021
 * @brief NNTrainer *.tflite Interpreter
 * @see	https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tflite_interpreter.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <bn_realizer.h>
#include <fc_layer.h>
#include <layer_node.h>
#include <tflite_export_realizer.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tf_schema_generated.h>
#include <tflite_opnode.h>

static constexpr const char *FUNC_TAG = "[TFLITE INTERPRETER] ";

// This Variables need for create tflite nodes
nntrainer::TfOpNode::Variables new_variable;
nntrainer::Tensor new_weight_add[50];
unsigned int new_alloc_tensors_idx = 0;

namespace nntrainer {

namespace {
/**
 * @brief after finishing building, call this to safe to a file
 *
 * @param builder flatbuffer builder
 * @param out out
 */
void builder2file(const flatbuffers::FlatBufferBuilder &builder,
                  const std::string &out) {
  uint8_t *buf = builder.GetBufferPointer();
  size_t size = builder.GetSize();
  flatbuffers::Verifier v(buf, size);
  NNTR_THROW_IF(!tflite::VerifyModelBuffer(v), std::invalid_argument)
    << FUNC_TAG << "Verifying serialized model failed";
  std::ofstream os(out, std::ios_base::binary);

  const size_t error_buflen = 100;
  char error_buf[error_buflen];
  NNTR_THROW_IF(!os.good(), std::invalid_argument)
    << FUNC_TAG << "failed to open, reason: "
    << SAFE_STRERROR(errno, error_buf, error_buflen);

  std::streamsize sz = static_cast<std::streamsize>(builder.GetSize());
  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << FUNC_TAG << "builder size: " << builder.GetSize()
    << " is too big. It cannot be represented by std::streamsize";

  os.write((char *)builder.GetBufferPointer(), sz);
  os.close();
}

/**
 * @brief get predecessor nodes
 *
 * @param node the node from which to get predecessor nodes
 * @note virtual nodes are ignored
 */
std::vector<const TfOpNode *> getPredNodes(const TfOpNode &node) {
  std::vector<const TfOpNode *> predNodes;

  for (auto input : node.getInputNodes()) {
    const TfOpNode *pred = input;
    while (pred->isVirtualNode()) {
      /// Assume that virtual nodes have single input
      assert(pred->arity() == 1);
      pred = pred->arg(0);
    }
    predNodes.push_back(pred);
  }
  return predNodes;
}

using TfOpNodes = std::vector<std::unique_ptr<TfOpNode>>;

/**
 * @brief Bidirectional Index map
 *
 * @tparam Key type of a underlying hashable value, please note that T will be
 * copied, so please use this for pointers and primitive values that is okay to
 * copy
 * @tparam Data data type to be stored inside the vector, if not given, same as
 * KeyType
 */
template <typename KeyType, typename DataType = KeyType>
class BidirectionalIndexMap {
public:
  /**
   * @brief addDatapoint to the map
   *
   * @param key key to be added to search for the data
   * @param data data to be added if there is no occurrence, data will be
   * copied.
   */
  void addDataWhenNotFound(KeyType key, DataType data) {
    auto search = key2index.find(key);

    if (search == key2index.end()) {
      key2index[key] = index2data.size();
      index2data.push_back(data);
    }
  }

  /**
   * @brief addDatapoint to the map when key and datatype is same
   *
   * @param key key/data to add
   */
  void addDataWhenNotFound(KeyType key) {
    static_assert(std::is_same<KeyType, DataType>::value == true,
                  "key type and data type are different!");
    addDataWhenNotFound(key, key);
  }

  /**
   * @brief Get the Index of the data
   *
   * @param key data that will be the key
   * @return unsigned int index
   */
  unsigned int getIndex(const KeyType &key) const {
    auto search = key2index.find(key);

    NNTR_THROW_IF(search == key2index.end(), std::invalid_argument)
      << FUNC_TAG << "Cannot find index for key: " << key;

    return search->second;
  }

  /**
   * @brief Get the Data object
   *
   * @param idx index to be searched
   * @return T datapoint T
   */
  DataType getData(unsigned int index) const {
    NNTR_THROW_IF(index >= index2data.size(), std::invalid_argument)
      << FUNC_TAG << "Cannot find data for index: " << index;

    return index2data[index];
  }

  /**
   * @brief Get the Data object
   *
   * @return const std::vector<T>& underlying data
   */
  const std::vector<DataType> &getData() const { return index2data; }

private:
  std::unordered_map<KeyType, unsigned int> key2index; /**< key -> index map */
  std::vector<DataType> index2data;                    /**< index -> data map */
};

/**
 * @brief tensorflow operation index map, this class manages operation index
 * mapping
 *
 */
class TfOpIdxMap {
public:
  using Buffer = std::pair<size_t, const float *>;

  TfOpIdxMap(const TfOpNodes &nodes) {
    auto &opcode_map = getIndexMap<tflite::BuiltinOperator>();
    auto update_opcode = [&opcode_map](tflite::BuiltinOperator opcode) {
      opcode_map.addDataWhenNotFound(opcode);
    };

    auto &buffer_map = getIndexMap<const float *, Buffer>();
    buffer_map.addDataWhenNotFound(
      nullptr, {0, empty_buffer}); // this represents undefined buffer
    buffer_map.addDataWhenNotFound(
      empty_buffer, {0, empty_buffer}); // this represents empty buffer

    auto update_buffer_map = [&buffer_map](const TfOpNode::Variables &variables,
                                           bool dynamic) {
      for (auto &variable : variables) {
        const float *buf = variable->getData();
        assert(buf != nullptr);
        auto byte_size = dynamic ? 0 : variable->bytes();
        buffer_map.addDataWhenNotFound(buf, {byte_size, buf});
      }
    };

    auto register_tensors =
      [&tensors = this->tensors](const TfOpNode::Variables &variables) {
        for (auto &variable : variables) {
          auto tensor_it = std::find(tensors.begin(), tensors.end(), variable);
          if (tensor_it == tensors.end()) {
            tensors.push_back(variable);
          }
        }
      };

    for (auto &op_node : nodes) {
      if (op_node->isVirtualNode())
        continue;
      update_opcode(op_node->getOpType());

      if (op_node->isInputNode()) {
        /**
         * Q) Why only register graph input tensor?
         *
         * A) the tflite needs only one tensor between nodes. Therefore,
         *basically, no inputs are considered except graph input that doesn't
         *have FROM node.
         **/
        register_tensors(op_node->getInputs());
        /**
         * Q) Why only update second input of the input node?
         *
         * A) 1. graph input nodes should be Transpose operator to change data
         *format from NCHW to NHWC.
         *    2. Transpose operator has two inputs - input to be
         *transposed(input[0]), 1d permute vector(input[1])
         *    3. input[0] has nullptr data pointer, which can't be added to
         *buffer_map. But, input[0] should have its own buffer and it will be
         *considered when the tflite buffers are built.
         **/
        assert(op_node->getInputs()[0]->getData() == nullptr);
        update_buffer_map({op_node->getInputs()[1]}, false);
      }
      register_tensors(op_node->getWeights());
      update_buffer_map(op_node->getWeights(), false);

      register_tensors(op_node->getOutputs());
      update_buffer_map(op_node->getOutputs(), true);
    }

    auto update_model_io_to = [this](const TfOpNode::Variables &variables,
                                     std::vector<int> &v) {
      for (auto &variable : variables) {
        if (variable->getName().find("nntrainer_internal_perm") !=
            std::string::npos)
          continue;
        v.push_back(this->getTensorIndex(variable));
      }
    };

    for (auto &op_node : nodes) {
      if (op_node->isVirtualNode())
        continue;
      if (op_node->isInputNode()) {
        update_model_io_to(op_node->getInputs(), inputs);
      }
      if (op_node->isOutputNode()) {
        update_model_io_to(op_node->getOutputs(), outputs);
      }
    }
  }

  template <typename KeyType, typename DataType = KeyType>
  BidirectionalIndexMap<KeyType, DataType> &getIndexMap() {
    return std::get<BidirectionalIndexMap<KeyType, DataType>>(maps);
  }

  template <typename KeyType, typename DataType = KeyType>
  const BidirectionalIndexMap<KeyType, DataType> &getIndexMap() const {
    return std::get<BidirectionalIndexMap<KeyType, DataType>>(maps);
  }

  const float *get_empty_buffer() const { return empty_buffer; }

  const std::vector<int> &getInputs() const { return inputs; }

  const std::vector<int> &getOutputs() const { return outputs; }

  const std::vector<const Tensor *> &getTensors() const { return tensors; }

  std::ptrdiff_t getTensorIndex(const Tensor *tensor) const {
    auto tensor_it = std::find(tensors.begin(), tensors.end(), tensor);
    NNTR_THROW_IF(tensor_it == tensors.cend(), std::invalid_argument)
      << FUNC_TAG << "Cannot find index for tensor: " << tensor->getName();
    return std::distance(tensors.begin(), tensor_it);
  }

private:
  float empty_buffer[0]; /**< reserved uninitialized tensor points to this
                            buffer */

  std::tuple<BidirectionalIndexMap<const float *, Buffer>,   /**< buffer map
                                                              */
             BidirectionalIndexMap<tflite::BuiltinOperator>> /**< opcode map
                                                              */
    maps;

  std::vector<int> inputs;
  std::vector<int> outputs;
  /// since it is used as a tensor index, the order is important
  std::vector<const Tensor *> tensors;
};

TfOpNodes buildOpNodes(const GraphRepresentation &representation,
                       flatbuffers::FlatBufferBuilder &fbb) {
  TfOpNodes nodes;
  /// @todo TfOpNode needs to have LayerNode pointer
  std::map<TfOpNode *, const LayerNode *> tf_to_layer;
  std::map<const LayerNode *, TfOpNode *> layer_to_tf;

  /// @todo, look ahead of layers to get nodes that can be fused
  /// we will need to have a dedicated builder
  for (auto iter = representation.cbegin(); iter != representation.cend();
       iter++) {
    const auto &ln = *iter;

    Exporter e(&fbb);
    ln->exportTo(e, ml::train::ExportMethods::METHOD_TFLITE);
    auto export_output = e.getResult<ml::train::ExportMethods::METHOD_TFLITE>();

    if (export_output.get()->getWeights().size() == 0) {
      export_output.get()->setTrainable(false);
    }

    nodes.emplace_back(std::move(export_output));
    tf_to_layer.insert({nodes.back().get(), ln.get()});
    layer_to_tf.insert({ln.get(), nodes.back().get()});
  }

  int node_count = 0;
  bool is_local_first = true;
  /** is_local_first : first FC Layer after Channel related layer
   * For example
   * : Input -> Conv -> Conv -> Flatten -> [FC]:local_first
   * : Input -> Conv -> Flatten -> [FC]:local_first -> Conv -> Flatten ->
   * [FC]:local_first
   */

  // set reorder weight flag for FullyConnected layer
  for (auto &n : nodes) {
    auto tf_node = n.get();

    if (tf_node->getOptionType() ==
          tflite::BuiltinOptions::BuiltinOptions_FullyConnectedOptions &&
        node_count != 0 && is_local_first) {
      tf_node->setNeedReorderWeight();
      is_local_first = false;
    }

    if (is_local_first == false &&
        tf_node->getOptionType() !=
          tflite::BuiltinOptions::BuiltinOptions_FullyConnectedOptions) {
      is_local_first = true;
    }

    node_count++;
  }

  /// set arity of TfOpNodes
  for (auto &n : nodes) {
    auto tf_node = n.get();
    auto searched_layer = tf_to_layer.find(tf_node);
    if (searched_layer == tf_to_layer.end())
      throw std::runtime_error("Cannot find layer for TfOpNode");
    auto layer_node = searched_layer->second;
    auto layer_node_inputs = layer_node->getInputConnections();

    /// assume that the TfOpNode and the LayerNode have a one-to-one
    /// relationship
    tf_node->arity(layer_node_inputs.size());
    for (size_t index = 0; index < layer_node_inputs.size(); index++) {
      auto input_layer_name = layer_node_inputs[index];
      auto input_layer_node_iterator = std::find_if(
        representation.begin(), representation.end(),
        [&input_layer_name](std::shared_ptr<nntrainer::LayerNode> node) {
          return istrequal(node.get()->getName(), input_layer_name);
        });

      if (input_layer_node_iterator != representation.end()) {
        auto input_layer_node = input_layer_node_iterator->get();
        if (layer_to_tf.find(input_layer_node) != layer_to_tf.end()) {
          tf_node->setArg(index, layer_to_tf.find(input_layer_node)->second);
        }
      }
    }
  }

  node_count = 0;
  for (auto &n : nodes) {
    auto tf_node = n.get();
    if (tf_node->getOptionType() ==
        tflite::BuiltinOptions::BuiltinOptions_FullyConnectedOptions) {
      tf_node->weightReorder(node_count);
    }

    if (tf_node->getOpType() ==
          tflite::BuiltinOperator::BuiltinOperator_CONV_2D &&
        nodes.at(node_count + 1).get()->getOpType() ==
          tflite::BuiltinOperator::BuiltinOperator_MUL &&
        nodes.at(node_count + 2).get()->getOpType() ==
          tflite::BuiltinOperator::BuiltinOperator_RELU) {
      // Fuse Conv2D + Mul(Batch Norm) + ReLU to Conv2D

      auto props = tf_node->getProps();
      auto tf_padding = tflite::Padding_SAME;

      if (props[0] == 1) {
        tf_padding = tflite::Padding_VALID;
      }
      auto new_options =
        tflite::CreateConv2DOptions(fbb, tf_padding, props[1], props[2],
                                    tflite::ActivationFunctionType_RELU)
          .Union();
      tf_node->setBuiltinOptions(tflite::BuiltinOptions_Conv2DOptions,
                                 new_options);
      // After Fusing Mark ReLU Node to be removed
      nodes.at(node_count + 2).get()->setToBeRemoved(true);
    }

    if (node_count < 1) {
      node_count++;
      continue;
    } else {
      if (nodes.at(node_count - 1).get()->isTrainable() == true &&
          tf_node->getOpType() == tflite::BuiltinOperator_MUL) {

        // Fused weight(conv)
        // = weight(conv) * (weight(bn) / sqrt(var(bn) + eps))

        auto conv_weights = nodes.at(node_count - 1).get()->getWeights();
        Tensor conv_weight(conv_weights.at(0)->getDim());
        Tensor conv_bias(conv_weights.at(1)->getDim());
        conv_weight.copyData(conv_weights.at(0)->clone());
        conv_bias.copyData(conv_weights.at(1)->clone());

        auto mul_weights = tf_node->getWeights();
        auto mul_mean = mul_weights.at(0)->clone().transpose("1:2:0");
        auto mul_var = mul_weights.at(1)->clone().transpose("1:2:0");
        auto mul_weight = mul_weights.at(2)->clone().transpose("1:2:0");
        auto mul_bias = mul_weights.at(3)->clone().transpose("1:2:0");
        auto mul_epsilon = tf_node->getAdditionalProps().at(0);

        // run sqrt(var(bn) + eps)
        mul_var.add_i(mul_epsilon);
        mul_var.pow_i(-0.5f);
        mul_weight.multiply_i(mul_var);

        Tensor reshape_mul_weight(mul_weight.getDim());
        reshape_mul_weight.copy(mul_weight);
        reshape_mul_weight.reshape(
          TensorDim{mul_weight.getDim().width(), 1, 1, 1});
        conv_weight.multiply_i(reshape_mul_weight);

        conv_bias.subtract_i(mul_mean);
        conv_bias.multiply_i(mul_weight);
        conv_bias.add_i(mul_bias);

        TfOpNode::Variables conv_new_weights;
        conv_new_weights.push_back(&conv_weight);
        conv_new_weights.push_back(&conv_bias);
        nodes.at(node_count - 1).get()->setWeights(conv_new_weights);
        // set mul node to be removed (mul mean batch normalization)
        n->setToBeRemoved(true);
      }
    }
    node_count++;
  }

  return nodes;
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>>
buildBuffers(const TfOpIdxMap &map, flatbuffers::FlatBufferBuilder &fbb) {
  const auto &buffers =
    map.getIndexMap<const float *, TfOpIdxMap::Buffer>().getData();

  std::vector<flatbuffers::Offset<tflite::Buffer>> fb_buffers;
  fb_buffers.reserve(buffers.size());

  auto create_buffer_offset = [&fbb](const TfOpIdxMap::Buffer &buffer) {
    if (buffer.first == 0) {
      return tflite::CreateBuffer(fbb);
    }

    auto data = fbb.CreateVector(
      reinterpret_cast<const uint8_t *>(buffer.second), buffer.first);

    return tflite::CreateBuffer(fbb, data);
  };

  std::transform(buffers.begin(), buffers.end(), std::back_inserter(fb_buffers),
                 create_buffer_offset);

  // add input buffer
  for (unsigned index = 0; index < map.getInputs().size(); index++) {
    fb_buffers.push_back(create_buffer_offset({0, nullptr}));
  }
  return fbb.CreateVector(fb_buffers);
}

flatbuffers::Offset<
  flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>>
buildOperatorCodes(const TfOpIdxMap &map, flatbuffers::FlatBufferBuilder &fbb) {
  const auto &op_codes = map.getIndexMap<tflite::BuiltinOperator>().getData();

  std::vector<flatbuffers::Offset<tflite::OperatorCode>> fb_op_codes;
  fb_op_codes.reserve(op_codes.size());

  auto create_op_offset = [&fbb](const tflite::BuiltinOperator &op,
                                 int32_t version = 1) {
    tflite::OperatorCodeBuilder builder(fbb);
    builder.add_deprecated_builtin_code(static_cast<int8_t>(op));
    /// @todo find reason why version field is not shown
    /// on json when version is 1 (other versions are fine)
    builder.add_version(version);
    builder.add_builtin_code(op);
    return builder.Finish();
  };

  std::transform(op_codes.begin(), op_codes.end(),
                 std::back_inserter(fb_op_codes), create_op_offset);

  return fbb.CreateVector(fb_op_codes);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Tensor>>>
buildTensors(const TfOpIdxMap &map, flatbuffers::FlatBufferBuilder &fbb) {
  /// @todo: the actual (squeezed) tensor dimension must be known before
  /// coming here. For now, it is directly guessed for the fc layer
  const auto &variables = map.getTensors();
  const auto &buffer_map = map.getIndexMap<const float *, TfOpIdxMap::Buffer>();
  auto graph_input_offset = map.getInputs().size() - 1;

  std::vector<flatbuffers::Offset<tflite::Tensor>> fb_tensors;
  fb_tensors.reserve(variables.size());

  auto create_tensor = [&fbb, &buffer_map,
                        &graph_input_offset](const Tensor *var) {
    auto dim = var->getDim();
    bool need_shape_signature = dim.is_dynamic();
    std::vector<int32_t> eff_dim = dim.getEffectiveDimension();
    auto shape = fbb.CreateVector(eff_dim);

    decltype(shape) shape_sig;
    if (need_shape_signature) {
      std::vector<int32_t> dyn_dim = dim.getEffectiveDimension(true);
      shape_sig = fbb.CreateVector(dyn_dim);
    }

    /// change this var->getName when tensor have it's own name
    auto name = fbb.CreateString("nntrainer_converted" + var->getName());

    /// only graph inputs have nullptr data pointer.
    unsigned int buffer_idx =
      var->getData() == nullptr
        ? buffer_map.getData().size() - graph_input_offset--
        : buffer_map.getIndex(var->getData());

    tflite::TensorBuilder builder(fbb);
    builder.add_name(name);
    builder.add_buffer(buffer_idx);
    /// @todo support more data types
    /// @note this is workaround because nntrainer tensor allows only float
    /// dtype
    if (var->getName().find("nntrainer_internal_perm") != std::string::npos) {
      builder.add_type(tflite::TensorType_INT32);
    } else
      builder.add_type(tflite::TensorType_FLOAT32);
    builder.add_shape(shape);
    if (need_shape_signature) {
      builder.add_shape_signature(shape_sig);
    }
    return builder.Finish();
  };

  std::transform(variables.begin(), variables.end(),
                 std::back_inserter(fb_tensors), create_tensor);

  return fbb.CreateVector(fb_tensors);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Operator>>>
buildOperators(const TfOpNodes &nodes, const TfOpIdxMap &map,
               flatbuffers::FlatBufferBuilder &fbb) {

  /// this lambda maps variables to list of indexes in the map
  auto variables_to_idx_vector = [&map](const TfOpNode::Variables &v) {
    std::vector<int> idx_vector;
    idx_vector.reserve(v.size());

    std::transform(
      v.begin(), v.end(), std::back_inserter(idx_vector),
      [&map](const Tensor *variable) { return map.getTensorIndex(variable); });
    return idx_vector;
  };

  auto create_operator = [&fbb, &map,
                          &variables_to_idx_vector](const TfOpNode &node) {
    auto &index_map = map.getIndexMap<tflite::BuiltinOperator>();

    auto op_code = index_map.getIndex(node.getOpType());
    std::vector<int> inputs;
    if (node.isInputNode()) {
      inputs = variables_to_idx_vector(node.getInputs());
    } else {
      /**
       *  Q) Why find a tensor that shares a buffer with input tensor?
       *
       *  A) the tflite needs only one tensor between nodes. Therefore,
       *basically, output tensors are used for tflite tensor that shares its
       *buffer with input's
       **/
      TfOpNode::Variables input_tensors;
      for (auto parent_node : getPredNodes(node)) {
        for (auto parent_out : parent_node->getOutputs()) {
          for (auto in : node.getInputs()) {
            /// second condition is a workaround
            /// Transpose op output tensor originally had nullptr data pointer
            /// but it has been allocated (parent_out->getData()). But, the
            /// buffer that shared its buffer hasn't so it has still nullptr
            /// (in->getData()).
            /// @todo remove this workaround
            if (parent_out->getData() == in->getData() ||
                (in->getData() == nullptr && parent_out->getData())) {
              if (std::find(input_tensors.begin(), input_tensors.end(),
                            parent_out) != input_tensors.end())
                continue;
              input_tensors.push_back(parent_out);
            }
          }
        }
      }
      inputs = variables_to_idx_vector(input_tensors);
    }
    auto weights = variables_to_idx_vector(node.getWeights());

    /// weights are part of input in tflite
    inputs.insert(inputs.end(), weights.begin(), weights.end());

    auto outputs = variables_to_idx_vector(node.getOutputs());

    auto fb_inputs = fbb.CreateVector(inputs);
    auto fb_outputs = fbb.CreateVector(outputs);
    auto fb_options = node.getBuiltinOps();

    tflite::OperatorBuilder builder(fbb);
    builder.add_opcode_index(op_code);
    builder.add_builtin_options_type(node.getOptionType());
    builder.add_builtin_options(fb_options);
    builder.add_inputs(fb_inputs);
    builder.add_outputs(fb_outputs);
    return builder.Finish();
  };

  std::vector<flatbuffers::Offset<tflite::Operator>> v;
  v.reserve(nodes.size());

  for (auto &node : nodes) {
    if (node->isVirtualNode())
      continue;
    auto op = create_operator(*node);
    v.push_back(op);
  }

  return fbb.CreateVector(v);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::SubGraph>>>
buildSubGraphs(const TfOpNodes &nodes, const TfOpIdxMap &map,
               flatbuffers::FlatBufferBuilder &fbb) {

  auto tensors = buildTensors(map, fbb);
  auto ops = buildOperators(nodes, map, fbb);

  /// @todo extract this to buildSubgraph if there is one or more subgraph
  auto name = fbb.CreateString("main");
  auto inputs = fbb.CreateVector(map.getInputs());
  auto outputs = fbb.CreateVector(map.getOutputs());

  auto builder = tflite::SubGraphBuilder(fbb);
  builder.add_tensors(tensors);
  builder.add_inputs(inputs);
  builder.add_outputs(outputs);
  builder.add_name(name);
  builder.add_operators(ops);
  auto subgraph = builder.Finish();

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  subgraphs.reserve(1);
  subgraphs.push_back(subgraph);

  return fbb.CreateVector(subgraphs);
}

} // namespace

TfOpNodes buildRealizedOpNodes(TfOpNodes &nodes,
                               flatbuffers::FlatBufferBuilder &fbb) {
  TfOpNodes realized_nodes;

  bool set_input = false;
  unsigned int node_count = 0;

  for (auto &node : nodes) {
    if (set_input) { // if front node is new added node set input output
      node->setArg(0, realized_nodes.back().get());
      realized_nodes.back()->setOutputs(node->getInputs());
      set_input = false;
    }

    if (node->isToBeRemoved() == true) {
      // Remove node, Assume that Input Node is not removed
      realized_nodes.back().get()->setOutputs(
        nodes.at(node_count)->getOutputs());
      nodes.at(node_count + 1)->setArg(0, realized_nodes.back().get());
      nodes.at(node_count + 1)->setInputs(nodes.at(node_count)->getInputs());
    } else {
      realized_nodes.push_back(std::move(node));

      if (realized_nodes.back().get()->getOpType() ==
          tflite::BuiltinOperator_MUL) { // Fused MUL ADD (Non Trainable)
        /**
          y = x * (gamma / sqrt(variance + epsilon)) +
          (beta - mean * gamma / sqrt(variance + epsilon))
        */
        const TfOpNode *next_node =
          node_count + 2 < nodes.size() ? nodes[node_count + 1].get() : nullptr;
        NNTR_THROW_IF(!next_node ||
                        next_node->getOpType() !=
                          tflite::BuiltinOperator_RELU ||
                        next_node->arity() != 1 ||
                        next_node->arg(0) != realized_nodes.back().get(),
                      std::invalid_argument)
          << FUNC_TAG
          << "batch normalization after a layer without weights must be "
             "followed by a relu activation that is not the last layer";

        auto removed_weights = realized_nodes.back().get()->getWeights();
        auto mul_mean = removed_weights.at(0)->clone();
        auto mul_variance = removed_weights.at(1)->clone();
        auto mul_gamma = removed_weights.at(2)->clone();
        auto mul_beta = removed_weights.at(3)->clone();
        auto mul_epsilon =
          realized_nodes.back().get()->getAdditionalProps().at(0);

        std::unique_ptr<Tensor> new_mul_weight =
          std::make_unique<Tensor>(mul_gamma.getDim());
        new_mul_weight->allocate();
        new_mul_weight->copy(mul_gamma);

        // new_mul_weight = (gamma / sqrt(variance + epsilon))
        mul_variance.add_i(mul_epsilon);
        mul_variance.pow_i(-0.5f);
        new_mul_weight->multiply_i(mul_variance);

        // beta =  (beta - mean * gamma / sqrt(variance + epsilon))
        Tensor sub_result(new_mul_weight->getDim());
        sub_result.allocate();
        sub_result.copyData(*new_mul_weight);

        mul_mean.multiply_i(sub_result);
        mul_beta.subtract_i(mul_mean);
        new_mul_weight->setName("MUL");
        // setWeights() keeps a transposed copy in node_owned_variable
        TfOpNode::Variables fused_weights = {new_mul_weight.get()};

        realized_nodes.back().get()->replaceWeights(fused_weights);
        realized_nodes.back().get()->setWeights(fused_weights, true);

        // Insert Add layer into Graph
        std::unique_ptr<TfOpNode> tf_node = std::make_unique<TfOpNode>();
        tf_node->setInputs(realized_nodes.back()->getOutputs());
        tf_node->setOpType(tflite::BuiltinOperator_ADD);
        auto options =
          tflite::CreateAddOptions(fbb, tflite::ActivationFunctionType_RELU)
            .Union();

        new_weight_add[new_alloc_tensors_idx].allocate();
        new_weight_add[new_alloc_tensors_idx].copy(mul_beta);
        std::string name = "ADD_tensor";
        new_weight_add[new_alloc_tensors_idx].setName(name);

        new_variable.clear();
        new_variable.emplace_back(&new_weight_add[new_alloc_tensors_idx]);
        new_alloc_tensors_idx++;

        tf_node->replaceWeights(new_variable);
        tf_node->setWeights(new_variable, true);
        tf_node->setBuiltinOptions(tflite::BuiltinOptions_AddOptions, options);

        nodes.at(node_count + 1)
          .get()
          ->setToBeRemoved(true); // remove ReLU Layer and Fuse with Add

        auto mul_node = realized_nodes.back().get();
        tf_node->arity(1);
        tf_node->setArg(0, mul_node);

        realized_nodes.push_back(std::move(tf_node));
        set_input = true;
      }
    }
    node_count++;
  }

  return realized_nodes;
}

void TfliteInterpreter::serialize(const GraphRepresentation &representation,
                                  const std::string &out) {

  /// 1. remove loss layer in GraphRepresentation
  TfliteExportRealizer tflite_realizer({});
  GraphRepresentation graph_loss = tflite_realizer.realize(representation);
  GraphRepresentation graph = tflite_realizer.realize_dropout(graph_loss);

  /// 2. The graph must have weights, input dims, output dims set
  flatbuffers::FlatBufferBuilder fbb;

  auto opNodes = buildOpNodes(graph, fbb);
  auto converted_opNodes = buildRealizedOpNodes(opNodes, fbb);

  TfOpIdxMap map(converted_opNodes); /// build TfOpIdxMap from opNodes
  auto opcodes = buildOperatorCodes(map, fbb);
  auto subgraphs = buildSubGraphs(converted_opNodes, map, fbb);
  auto buffers = buildBuffers(map, fbb);
  auto desc = fbb.CreateString("This file is generated from NNTrainer");
  tflite::ModelBuilder model_builder(fbb);

  model_builder.add_operator_codes(opcodes);
  model_builder.add_subgraphs(subgraphs);
  model_builder.add_buffers(buffers);
  model_builder.add_version(3);
  model_builder.add_description(desc);
  auto model = model_builder.Finish();

  fbb.Finish(model, tflite::ModelIdentifier());
  builder2file(fbb, out);
}

GraphRepresentation TfliteInterpreter::deserialize(const std::string &in) {
  /// ======== list of things to consider ========
  /// we need to reconstruct some properties from the shape
  /// eg) units are not saved as a property

  /** NYI! */
  return {};
}

} // namespace nntrainer

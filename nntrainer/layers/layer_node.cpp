// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   layer_node.cpp
 * @date   1 April 2021
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer node for network graph
 */

#include "layer_context.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <utility>

#include <activation_layer.h>
#include <base_properties.h>
#include <bn_layer.h>
#include <common_properties.h>
#include <connection.h>
#include <context.h>
#include <engine.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <profiler.h>
#include <time_dist.h>
#include <tracer.h>
#include <util_func.h>

#ifdef ENABLE_OPENCL
#include <cl_context.h>
#endif

namespace nntrainer {

#ifdef PROFILE
static constexpr const char *FORWARD_SUFFIX = ":forward";
static constexpr const char *CALC_DERIV_SUFFIX = ":calcDeriv";
static constexpr const char *CALC_GRAD_SUFFIX = ":calcGrad";
#endif

namespace props {

/**
 * @brief Flatten property, true if needs flatten layer afterwards
 */
class Flatten : public Property<bool> {
public:
  Flatten() : Property<bool>() {}               /**< has default value of 0 */
  static constexpr const char *key = "flatten"; /**< unique key to access */
  using prop_tag = bool_prop_tag;               /**< property type */
};

/**
 * @brief Distribute property, true if it distribute across layer
 *
 */
class Distribute : public Property<bool> {
public:
  Distribute() : Property<bool>() {}
  static constexpr const char *key = "distribute";
  using prop_tag = bool_prop_tag;
};

/**
 * @brief Loss property, this defines loss specification of layer
 *
 */
class Loss : public Property<float> {

public:
  /**
   * @brief Construct a new loss object with a default value 0.0
   *
   */
  Loss(float value = 0.0) : nntrainer::Property<float>(value) {}
  static constexpr const char *key = "loss"; /**< unique key to access */
  using prop_tag = float_prop_tag;           /**< property type */

  /**
   * @brief LossSpec validator
   * @todo  detect when loss becomes Nan is useful. But it will need dedicated
   * throw
   * @param v float to validate
   * @retval true if is valid number
   * @retval false if it is nan
   */
  bool isValid(const float &v) const override {
    if (std::isnan(v)) {
      ml_logw("loss value is NAN");
    }

    return true;
  }
};

/**
 * @brief Input shape property which saves a single tensor shape
 * (practically, std::array<InputShape> is used)
 *
 */
class InputShape : public GenericShape {

public:
  static constexpr const char *key = "input_shape"; /**< unique key to access */
  using prop_tag = dimension_prop_tag;              /**< property type */
};

/**
 * @brief Weight Dimension property which saves a single tensor dim
 *
 */
class WeightDimension : public GenericShape {

public:
  static constexpr const char *key = "weight_dim"; /**< unique key to access */
  using prop_tag = dimension_prop_tag;             /**< property type */
};

/**
 * @brief properties for shared from
 *
 */
class SharedFrom : public Name {
public:
  static constexpr const char *key = "shared_from"; /**< unique key to access */
  using prop_tag = str_prop_tag;                    /**< property type */
};

} // namespace props

/**
 * @brief Destroy the Layer Node object
 *
 */
LayerNode::~LayerNode() = default;

/**
 * @brief get the compute engine property from property string vector
 *  : default is CPU
 * @return LayerComputeEngine Enum : CPU, GPU, QNN
 *
 */
ml::train::LayerComputeEngine
getComputeEngine(const std::vector<std::string> &props) {
  for (auto &prop : props) {
    std::string key, value;
    int status = nntrainer::getKeyValue(prop, key, value);
    if (nntrainer::istrequal(key, "engine")) {
      constexpr const auto data =
        std::data(props::ComputeEngineTypeInfo::EnumList);
      for (unsigned int i = 0;
           i < props::ComputeEngineTypeInfo::EnumList.size(); ++i) {
        if (nntrainer::istrequal(value.c_str(),
                                 props::ComputeEngineTypeInfo::EnumStr[i])) {
          return data[i];
        }
      }
    }
  }

  return ml::train::LayerComputeEngine::CPU;
}

/**
 * @brief Layer factory creator with constructor
 */
std::unique_ptr<LayerNode>
createLayerNode(const ml::train::LayerType &type,
                const std::vector<std::string> &properties) {
  auto &eg = nntrainer::Engine::Global();
  return createLayerNode(eg.createLayerObject(type, properties), properties);
}

/**
 * @brief Layer factory creator with constructor
 */
std::unique_ptr<LayerNode>
createLayerNode(const std::string &type,
                const std::vector<std::string> &properties) {
  auto &eg = nntrainer::Engine::Global();
  return createLayerNode(eg.createLayerObject(type, properties), properties);
}

/**
 * @brief Layer factory creator with constructor
 */
std::unique_ptr<LayerNode>
createLayerNode(std::unique_ptr<nntrainer::Layer> &&layer,
                const std::vector<std::string> &properties) {
  auto lnode = std::make_unique<LayerNode>(std::move(layer));

  lnode->setProperty(properties);

  return lnode;
}

LayerNode::LayerNode(std::unique_ptr<nntrainer::Layer> &&l) :
  layer(std::move(l)),
  inplace_type(InPlaceType::NONE),
  needs_calc_derivative(false),
  needs_calc_gradient(false),

  output_connections(),
  run_context(nullptr),
  layer_node_props(new PropsType(
    props::Name(), props::Distribute(), props::Trainable(), {}, {},
    props::WeightDimension(), props::SharedFrom(),
    props::ClipGradByGlobalNorm(), props::Packed(), props::WeightDtype(),
    props::LossScaleForMixed(), props::ComputeEngine())),
  layer_node_props_realization(
    new RealizationPropsType(props::Flatten(), props::Activation())),
  loss(new props::Loss()),
  exec_order({0, 0, 0, 0}),
  needs_restore_data(false),
  data_type({TensorDim::DataType::FP32, TensorDim::DataType::FP32}) {
  if (layer && layer->getType() == TimeDistLayer::type) {
    std::get<props::Distribute>(*layer_node_props).set(true);
  }
}

void LayerNode::setProperty(const std::vector<std::string> &properties) {
  auto left_properties = loadProperties(properties, *layer_node_props);
  left_properties =
    loadProperties(left_properties, *layer_node_props_realization);
  layer->setProperty(left_properties);

  if (getType() == ActivationLayer::type) {
    auto &act_prop = std::get<props::Activation>(*layer_node_props_realization);
    if (!act_prop.empty()) {
      layer->setProperty({"activation=" + to_string(act_prop)});
    }
  }
}

std::string LayerNode::getProperty(const std::string &key) {
  if (layer_node_props) {
    std::string result = find_in_tuple(*layer_node_props, key);
    if (!result.empty()) {
      return result;
    }
  }

  if (layer_node_props_realization) {
    std::string result = find_in_tuple(*layer_node_props_realization, key);
    if (!result.empty()) {
      return result;
    }
  }

  return layer->getProperty(key);
}

void LayerNode::setWeights(const std::vector<float *> weights) {
  NNTR_THROW_IF(!run_context, std::runtime_error)
    << __func__ << " layer needs to be finalized first!";

  NNTR_THROW_IF(getNumWeights() != weights.size(), std::runtime_error)
    << __func__ << " Number of Weights dismatch!";

  // Needs Deep copy
  for (unsigned int idx = 0; idx < getNumWeights(); ++idx) {
    Tensor &w = getWeight(idx);
    std::copy(weights[idx], weights[idx] + w.size(), w.getData());
  }
}

const unsigned LayerNode::getInputConnectionIndex(unsigned nth) const {
  auto &input_conns =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  return input_conns.at(nth).get().getIndex();
}

const std::string &LayerNode::getInputConnectionName(unsigned nth) const {
  auto &input_conns =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  return input_conns.at(nth).get().getName();
}

void LayerNode::setInputConnectionIndex(unsigned nth, unsigned index) {
  auto &input_conns =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  input_conns.at(nth).get().getIndex() = index;
}

void LayerNode::setInputConnectionName(unsigned nth, const std::string &name) {
  auto &input_conns =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  input_conns.at(nth).get().getName() = name;
}

const Connection *LayerNode::getOutputConnection(unsigned nth) const {
  return output_connections.at(nth).get();
}

void LayerNode::setOutputConnection(unsigned nth, const std::string &name,
                                    unsigned index) {
  if (nth >= output_connections.size()) {
    output_connections.resize(nth + 1);
  }

  auto &con = output_connections[nth];
  // Should be override connection for the batch normalization realizer
  // NNTR_THROW_IF(con, std::invalid_argument)
  //   << "cannot override connection, this slot is reserved for "
  //   << con->toString();

  con = std::make_unique<Connection>(name, index);
}

void LayerNode::setComputeEngine(
  const ml::train::LayerComputeEngine &compute_engine) {
  // setting compute_engine of LayerNode
  // can be reused later to propagate this info
  this->compute_engine = compute_engine;
}

const std::string LayerNode::getName() const {
  auto &name = std::get<props::Name>(*layer_node_props);
  return name.empty() ? "" : name.get();
}

std::ostream &operator<<(std::ostream &out, const LayerNode &l) {

  auto &input_connections =
    std::get<std::vector<props::InputConnection>>(*l.layer_node_props);

  out << "[" << l.getName() << '/' << l.getType() << "]\n";
  auto print_vector = [&out](const auto &cons, const std::string &title) {
    out << title << "[" << cons.size() << "] ";
    for (auto &con : cons) {
      out << con.toString() << ' ';
    }
    out << '\n';
  };

  auto print_vector_2 = [&out](const auto &cons, const std::string &title) {
    out << title << "[" << cons.size() << "] ";
    for (auto &con : cons) {
      out << con->toString() << ' ';
    }
    out << '\n';
  };

  print_vector(
    std::vector<Connection>(input_connections.begin(), input_connections.end()),
    " input_connections");
  print_vector_2(l.output_connections, "output_connections");
  return out;
}

ActivationType LayerNode::getActivationType() const {
  auto &act_prop = std::get<props::Activation>(*layer_node_props_realization);
  if (act_prop.empty()) {
    return ActivationType::ACT_NONE;
  }

  return act_prop;
}

unsigned int LayerNode::getNumInputConnections() const {
  auto &input_conns =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  return input_conns.size();
}

unsigned int LayerNode::getNumOutputConnections() const {
  return output_connections.size();
}

const std::vector<std::string> LayerNode::getInputLayers() const {
  auto &input_connections =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  std::vector<std::string> names;
  names.reserve(input_connections.size());
  std::transform(
    input_connections.begin(), input_connections.end(),
    std::back_inserter(names),
    [](const Connection &con) -> const auto & { return con.getName(); });
  return names;
}

const std::vector<std::string> LayerNode::getOutputLayers() const {
  std::vector<std::string> names;
  names.reserve(output_connections.size());

  for (auto &conn : output_connections) {
    if (conn == nullptr) {
      ml_logw("intermediate output is empty for layer: %s", getName().c_str());
      continue;
    }
    names.push_back(conn->getName());
  }
  return names;
}

ActivationType LayerNode::getActivationToBeRealized() const {
  if (getType() == ActivationLayer::type)
    return ActivationType::ACT_NONE;
  else
    return getActivationType();
}

const std::string LayerNode::getType() const { return getLayer()->getType(); }

bool LayerNode::getTrainable() const {
  if (run_context)
    /**
     * if a layer does not contain any weights, it will be treated as a
     * non-trainable layer.
     */
    return std::get<props::Trainable>(*layer_node_props) &&
           (run_context->getNumWeights() > 0);
  else
    return std::get<props::Trainable>(*layer_node_props);
}

bool LayerNode::getFlatten() const {
  auto &flatten = std::get<props::Flatten>(*layer_node_props_realization);
  if (flatten.empty()) {
    return false;
  }
  return flatten.get();
}

std::string LayerNode::getSharedFrom() const {
  auto &shared_from = std::get<props::SharedFrom>(*layer_node_props);
  return shared_from.empty() ? "" : shared_from.get();
}

bool LayerNode::getDistribute() const {
  auto &distribute = std::get<props::Distribute>(*layer_node_props);
  if (distribute.empty()) {
    return false;
  }
  return distribute.get();
}

const nntrainer::Layer *LayerNode::getLayer() const {
  if (run_context && getDistribute())
    return static_cast<TimeDistLayer *>(layer.get())->getDistLayer();
  else
    return layer.get();
}

nntrainer::Layer *LayerNode::getLayer() {
  if (run_context && getDistribute())
    return static_cast<TimeDistLayer *>(layer.get())->getDistLayer();
  else
    return layer.get();
}

void LayerNode::setOutputLayers(const std::vector<std::string> &layers) {
  output_connections.clear();
  output_connections.reserve(layers.size());
  std::transform(
    layers.begin(), layers.end(), std::back_inserter(output_connections),
    [](const std::string &id) { return std::make_unique<Connection>(id); });
}

bool LayerNode::hasInputShapeProperty() const {
  auto &input_shapes =
    std::get<std::vector<props::InputShape>>(*layer_node_props);

  return !input_shapes.empty() &&
         std::all_of(input_shapes.begin(), input_shapes.end(),
                     [](const auto &input) { return !input.empty(); });
}

const std::vector<TensorDim> LayerNode::getInputDimensions() const {
  NNTR_THROW_IF(!run_context, std::runtime_error)
    << __func__ << " layer needs to be finalized first!";
  auto sz = run_context->getNumInputs();
  std::vector<TensorDim> dims;
  dims.reserve(sz);

  for (auto i = 0u; i < sz; ++i) {
    dims.push_back(run_context->getInput(i).getDim());
  }

  return dims;
}

const std::vector<TensorDim> LayerNode::getOutputDimensions() const {
  NNTR_THROW_IF(!run_context, std::runtime_error)
    << __func__ << " layer needs to be finalized first!";
  auto sz = run_context->getNumOutputs();
  std::vector<TensorDim> dims;
  dims.reserve(sz);

  for (auto i = 0u; i < sz; ++i) {
    dims.push_back(run_context->getOutput(i).getDim());
  }

  return dims;
}

bool LayerNode::isWeightNode() const { return layer->getType() == "weight"; }

void LayerNode::exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const {
  exporter.saveResult(*layer_node_props, method, this);
  layer->exportTo(exporter, method);
}

void LayerNode::read(std::ifstream &file, bool opt_var,
                     ml::train::ExecutionMode mode, bool fsu,
                     size_t start_offset, bool read_from_offset, int file_fd) {
  NNTR_THROW_IF(!run_context, std::runtime_error)
    << __func__ << " layer needs to be finalized first!";
  getLayer()->read(file, *run_context, opt_var, mode,
                   (getTrainable() && mode == ml::train::ExecutionMode::TRAIN),
                   getWeightDataType(), fsu, start_offset, read_from_offset,
                   file_fd);
}

void LayerNode::read(ReadSource src, bool opt_var,
                     ml::train::ExecutionMode mode, bool fsu,
                     size_t start_offset, bool read_from_offset) {
  NNTR_THROW_IF(!run_context, std::runtime_error)
    << __func__ << " layer needs to be finalized first!";
  getLayer()->read(src, *run_context, opt_var, mode,
                   (getTrainable() && mode == ml::train::ExecutionMode::TRAIN),
                   getWeightDataType(), fsu, start_offset, read_from_offset);
}

void LayerNode::save(std::ofstream &file, bool opt_var,
                     ml::train::ExecutionMode mode) const {
  NNTR_THROW_IF(!run_context, std::runtime_error)
    << __func__ << " layer needs to be finalized first!";
  getLayer()->save(file, *run_context, opt_var, mode,
                   (getTrainable() && mode == ml::train::ExecutionMode::TRAIN),
                   getWeightDataType());
}

void LayerNode::clearOptVar() {
  NNTR_THROW_IF(!run_context, std::runtime_error)
    << __func__ << " layer needs to be finalized first!";
  for (unsigned int i = 0; i < run_context->getNumWeights(); ++i) {
    if (run_context->isGradientLastAccess(i) && getTrainable()) {
      /// @note read optimizer variables
      for (unsigned int j = 0; j < run_context->getNumWeightOptVar(i); ++j) {
        run_context->getWeightOptVar(i, j).initialize();
      }
    }
  }
}

/**
 * @brief     Finalize creating the layer node
 */
InitLayerContext LayerNode::finalize(const std::vector<TensorDim> &input_dims,
                                     std::array<std::string, 3> tensor_type,
                                     ml::train::ExecutionMode mode) {
  // auto get_tensor_datatype = [](const std::string ty) -> TensorDim::DataType
  // { 			       return from_string(ty);
  // };

  if (run_context)
    throw std::runtime_error(
      "Trying to finalizing a layer which is already finalized in layer: " +
      getName());

  std::vector<TensorDim> actual_input_dims;
  auto &prop_dims = std::get<std::vector<props::InputShape>>(*layer_node_props);
  auto &prop_in_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);

  /** prepare input dimensions */
  if (!input_dims.empty()) {
    actual_input_dims = input_dims;
    if (hasInputShapeProperty()) {
      std::vector<TensorDim> actual_prop_dims(prop_dims.begin(),
                                              prop_dims.end());
      /// if prop_dims exist, check if it's same with given input_dims
      NNTR_THROW_IF(input_dims != actual_prop_dims, std::invalid_argument)
        << "calculated input dimension is different from given input_shape "
           "property";
      for (auto &d : actual_prop_dims) {
        d.setDataType(
          str_converter<enum_class_prop_tag, nntrainer::TensorDataTypeInfo>::
            from_string(tensor_type[2]));
        d.setFormat(
          str_converter<enum_class_prop_tag, nntrainer::TensorFormatInfo>::
            from_string(tensor_type[0]));
      }
    }
  } else if (isWeightNode()) {
    actual_input_dims = std::vector<TensorDim>(
      {std::get<props::WeightDimension>(*layer_node_props)});
    for (auto &d : actual_input_dims) {
      d.setDataType(
        str_converter<enum_class_prop_tag, nntrainer::TensorDataTypeInfo>::
          from_string(tensor_type[2]));
      d.setFormat(
        str_converter<enum_class_prop_tag, nntrainer::TensorFormatInfo>::
          from_string(tensor_type[0]));
    }
  } else {
    NNTR_THROW_IF(!hasInputShapeProperty(), std::invalid_argument)
      << "if input dims not given, input shapes must be given by the user as "
         "property";
    /// arguably, below check can go away
    NNTR_THROW_IF((prop_dims.size() != prop_in_layers.size()) &&
                    (prop_dims.size() != 1 || !prop_in_layers.empty()),
                  std::invalid_argument)
      << "input shapes must be one if connection is not given but given "
         "dimesions size of: "
      << prop_dims.size();
    actual_input_dims =
      std::vector<TensorDim>(prop_dims.begin(), prop_dims.end());
    for (auto &d : actual_input_dims) {
      /// Input Tensor type of input layer needs to be float.
      d.setDataType(
        str_converter<enum_class_prop_tag,
                      nntrainer::TensorDataTypeInfo>::from_string("FP32"));
      d.setFormat(
        str_converter<enum_class_prop_tag, nntrainer::TensorFormatInfo>::
          from_string(tensor_type[0]));
    }
  }

  NNTR_THROW_IF(actual_input_dims.size() < getNumInputConnections(),
                std::invalid_argument)
    << "number of input dimensions must be equal or larger "
    << "than number of input connections, node name: " << getName()
    << " num input dims: " << input_dims.size()
    << " num connections: " << getNumInputConnections();

  /** manipulate layers if required */
  if (getType() != TimeDistLayer::type && getDistribute()) {
    std::unique_ptr<TimeDistLayer> dlayer(new TimeDistLayer());
    NNTR_THROW_IF(!dlayer, std::invalid_argument)
      << "Error creating time distribution layer";
    dlayer->setDistLayer(std::move(layer));
    layer = std::move(dlayer);
  }

  const auto &scope = getSharedFrom().empty() ? getName() : getSharedFrom();
  float max_norm = 0.0;
  float loss_scale = 1.0;
  if (!std::get<props::ClipGradByGlobalNorm>(*layer_node_props).empty())
    max_norm = std::get<props::ClipGradByGlobalNorm>(*layer_node_props).get();

  if (!std::get<props::LossScaleForMixed>(*layer_node_props).empty())
    loss_scale = std::get<props::LossScaleForMixed>(*layer_node_props).get();

  if (!std::get<props::ComputeEngine>(*layer_node_props).empty()) {
    compute_engine = std::get<props::ComputeEngine>(*layer_node_props).get();
  }

  if (!std::get<props::Packed>(*layer_node_props).empty()) {
    bool isPacked = std::get<props::Packed>(*layer_node_props);
    if (!isPacked) {
      // set weight type = activation type
      tensor_type[1] = tensor_type[2];
    }
  }

  if (!std::get<props::WeightDtype>(*layer_node_props).empty()) {
    // set weight type = tensor_data_type
    tensor_type[1] = to_string(std::get<props::WeightDtype>(*layer_node_props));
  }

  std::vector<bool> out_info;
  out_info.reserve(output_connections.size());
  std::transform(output_connections.begin(), output_connections.end(),
                 std::back_inserter(out_info), [](auto &con) { return !!con; });

  if (requireLabel() && out_info.empty()) {
    /// as we are using output Grad to save label, add fake out info if it's
    /// label. This should be substituted to the proper label management
    out_info.push_back(true);
  }

  auto context = InitLayerContext(
    actual_input_dims, out_info, getInPlaceType() != InPlaceType::NONE,
    getName(), scope, max_norm, tensor_type, loss_scale, mode, compute_engine);

  layer->finalize(context);

#ifdef ENABLE_TEST
  init_context = std::make_unique<InitLayerContext>(context);
#endif // ENABLE_TEST

#ifdef PROFILE
  auto profile_name = [this](const char *suffix) {
    return getName() + suffix + "(" + getType() + ")";
  };
#endif

  PROFILE_TIME_REGISTER_EVENT(forward_event_key, profile_name(FORWARD_SUFFIX));
  PROFILE_TIME_REGISTER_EVENT(calc_deriv_event_key,
                              profile_name(CALC_DERIV_SUFFIX));
  PROFILE_TIME_REGISTER_EVENT(calc_grad_event_key,
                              profile_name(CALC_GRAD_SUFFIX));

  return context;
}

/**
 * @brief     Refinalize creating the layer node
 */
InitLayerContext
LayerNode::refinalize(const std::vector<TensorDim> &input_dims) {
  std::vector<TensorDim> actual_input_dims;
  auto &prop_dims = std::get<std::vector<props::InputShape>>(*layer_node_props);
  auto &prop_in_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);

  /** prepare input dimensions */
  if (!input_dims.empty()) {
    actual_input_dims = input_dims;
    if (hasInputShapeProperty()) {
      std::vector<TensorDim> actual_prop_dims(prop_dims.begin(),
                                              prop_dims.end());
      /// if prop_dims exist, check if it's same with given input_dims
      NNTR_THROW_IF(input_dims != actual_prop_dims, std::invalid_argument)
        << "calculated input dimension is different from given input_shape "
           "property";
    }
  } else if (isWeightNode()) {
    actual_input_dims = std::vector<TensorDim>(
      {std::get<props::WeightDimension>(*layer_node_props)});
  } else {
    NNTR_THROW_IF(!hasInputShapeProperty(), std::invalid_argument)
      << "if input dims not given, input shapes must be given by the user as "
         "property";
    /// arguably, below check can go away
    NNTR_THROW_IF((prop_dims.size() != prop_in_layers.size()) &&
                    (prop_dims.size() != 1 || !prop_in_layers.empty()),
                  std::invalid_argument)
      << "input shapes must be one if connection is not given but given "
         "dimesions size of: "
      << prop_dims.size();
    actual_input_dims =
      std::vector<TensorDim>(prop_dims.begin(), prop_dims.end());
  }

  NNTR_THROW_IF(actual_input_dims.size() < getNumInputConnections(),
                std::invalid_argument)
    << "number of input dimensions must be equal or larger "
    << "than number of input connections, node name: " << getName()
    << " num input dims: " << input_dims.size()
    << " num connections: " << getNumInputConnections();

  /** manipulate layers if required */
  if (getType() != TimeDistLayer::type && getDistribute()) {
    std::unique_ptr<TimeDistLayer> dlayer(new TimeDistLayer());
    NNTR_THROW_IF(!dlayer, std::invalid_argument)
      << "Error creating time distribution layer";
    dlayer->setDistLayer(std::move(layer));
    layer = std::move(dlayer);
  }

  const auto &scope = getSharedFrom().empty() ? getName() : getSharedFrom();
  float max_norm = 0.0;
  if (!std::get<props::ClipGradByGlobalNorm>(*layer_node_props).empty())
    max_norm = std::get<props::ClipGradByGlobalNorm>(*layer_node_props).get();

  std::vector<bool> out_info;
  out_info.reserve(output_connections.size());
  std::transform(output_connections.begin(), output_connections.end(),
                 std::back_inserter(out_info), [](auto &con) { return !!con; });

  if (requireLabel() && out_info.empty()) {
    /// as we are using output Grad to save label, add fake out info if it's
    /// label. This should be substituted to the proper label management
    out_info.push_back(true);
  }

  auto context = InitLayerContext(actual_input_dims, out_info,
                                  getInPlaceType() != InPlaceType::NONE,
                                  getName(), scope, max_norm);

  layer->finalize(context);

#ifdef ENABLE_TEST
  init_context = std::make_unique<InitLayerContext>(context);
#endif // ENABLE_TEST

#ifdef PROFILE
  auto profile_name = [this](const char *suffix) {
    return getName() + suffix + "(" + getType() + ")";
  };
#endif

  PROFILE_TIME_REGISTER_EVENT(forward_event_key, profile_name(FORWARD_SUFFIX));
  PROFILE_TIME_REGISTER_EVENT(calc_deriv_event_key,
                              profile_name(CALC_DERIV_SUFFIX));
  PROFILE_TIME_REGISTER_EVENT(calc_grad_event_key,
                              profile_name(CALC_GRAD_SUFFIX));

  return context;
}

/**
 * @brief     Forward Propagation of a layer
 */
void LayerNode::forwarding(bool training) {
  loss->set(run_context->getRegularizationLoss());

  PROFILE_TIME_START(forward_event_key);
  if (reStoreData()) {
    if (getInPlaceType() == InPlaceType::NONE) {
      for (unsigned int i = 0; i < run_context->getNumOutputs(); ++i) {
        run_context->getOutput(i).setValue(0);
        if (!run_context->getOutputGradUnsafe(i).isValid())
          run_context->getOutputGradUnsafe(i).setValue(0);
      }
      for (unsigned int i = 0; i < run_context->getNumWeights(); ++i) {
        if (run_context->weightHasGradient(i)) {
          run_context->getWeightGrad(i).setValue(0);
        }
      }
    }
  }

  layer->forwarding(*run_context, training);
  reStoreData(false);
  PROFILE_TIME_END(forward_event_key);
  TRACE_MEMORY() << getName() + ": F";
  TRACE_TIME() << getName() + ": F";

#ifdef DEBUG
  if (!run_context->validate(getNumInputConnections() == 0, !requireLabel()))
    throw std::runtime_error("Running forwarding() layer " + getName() +
                             " invalidated the context.");
#endif

  /** add loss only for loss layers */
  if (requireLabel())
    loss->set(*loss + run_context->getLoss());
}

/**
 * @brief     Incremental forward Propagation of a layer
 */
void LayerNode::incremental_forwarding(unsigned int from, unsigned int to,
                                       bool training) {
  loss->set(run_context->getRegularizationLoss());
  PROFILE_TIME_START(forward_event_key);
  // std::cerr << getType() << "\n";
  layer->incremental_forwarding(*run_context, from, to, training);
  PROFILE_TIME_END(forward_event_key);
  TRACE_MEMORY() << getName() + ": F";
  TRACE_TIME() << getName() + ": F";

#ifdef DEBUG
  if (!run_context->validate(getNumInputConnections() == 0, !requireLabel()))
    throw std::runtime_error("Running forwarding() layer " + getName() +
                             " invalidated the context.");
#endif

  /** add loss only for loss layers */
  if (requireLabel())
    loss->set(*loss + run_context->getLoss());
}

/**
 * @brief     calc the derivative to be passed to the previous layer
 */
void LayerNode::calcDerivative() {
  PROFILE_TIME_START(calc_deriv_event_key);
  PROFILE_MEM_ANNOTATE("CalcDerivative: " + getName());
  layer->calcDerivative(*run_context);
  PROFILE_TIME_END(calc_deriv_event_key);
  TRACE_MEMORY() << getName() + ": CD";
  TRACE_TIME() << getName() + ": CD";

#ifdef DEBUG
  if (!run_context->validate(getNumInputConnections() == 0, !requireLabel()))
    throw std::runtime_error("Running calcDerivative() layer " + getName() +
                             " invalidated the context.");
#endif
}

/**
 * @brief     Calculate the derivative of a layer
 */
void LayerNode::calcGradient() {
  PROFILE_TIME_START(calc_grad_event_key);
  if (needs_calc_gradient) {
    PROFILE_MEM_ANNOTATE("CalcGradient: " + getName());
    layer->calcGradient(*run_context);
    TRACE_MEMORY() << getName() + ": CG";
    TRACE_TIME() << getName() + ": CG";
  }
  PROFILE_TIME_END(calc_grad_event_key);

#ifdef DEBUG
  if (!run_context->validate(getNumInputConnections() == 0, !requireLabel()))
    throw std::runtime_error("Running calcGradient() layer " + getName() +
                             " invalidated the context.");
#endif
}

/**
 * @brief Set the batch for the layer
 */
void LayerNode::setBatch(unsigned int batch) {
  NNTR_THROW_IF(!run_context, std::invalid_argument)
    << " setting batch not supported before initialization";

  getLayer()->setBatch(*run_context, batch);
}

void LayerNode::updateTensorsByInputDimensions(
  std::vector<TensorDim> input_dimensions) {
  NNTR_THROW_IF(!run_context, std::invalid_argument)
    << " update tensors not supported before initialization";

  getLayer()->updateTensorsByInputDimensions(*run_context, input_dimensions);
}

/**
 * @brief   If the current layer can support in-place
 */
bool LayerNode::supportInPlace() const {
  ///@note below is a quick fix, we need to have a guard that this shouldn't
  /// be
  /// query until realizeProps has been finalized ( which means we will need
  /// another end point to fixate this property )
  if (getDistribute()) {
    return false;
  }
  return layer->supportInPlace();
}

/**
 * @brief Get the inplace direction for the layer
 */
InPlaceDirection LayerNode::getInPlaceDirection() const {
  return layer->getInPlaceDirection();
};

/**
 * @brief Initialize the in-place settings of the layer
 * @return InPlaceType
 */
InPlaceType LayerNode::initializeInPlace() {
  inplace_type = layer->initializeInPlace();
  return inplace_type;
}

/**
 * @brief  check if this layer requires label to be passed
 */
bool LayerNode::requireLabel() const { return getLayer()->requireLabel(); }

/**
 * @brief     get loss for the layer
 * @return    loss of the layer
 */
float LayerNode::getLoss() const { return *loss; }

void LayerNode::configureRunContext(const std::vector<Weight *> &weights,
                                    const std::vector<Var_Grad *> &inputs,
                                    const std::vector<Var_Grad *> &outputs,
                                    const std::vector<Var_Grad *> &tensors,
                                    float loss_scale,
                                    std::shared_ptr<ContextData> ct_data) {
  run_context = std::make_unique<RunLayerContext>(
    getName(), getTrainable(), 0.0f, getInPlaceType() != InPlaceType::NONE,
    loss_scale, ct_data, false, weights, inputs, outputs, tensors);
}

/**
 * @brief   Print Options when printing layer info
 */
typedef enum {
  // clang-format off
  PRINT_INST_INFO  = (1 << 0), /**< Option to print type & instance address info */
  PRINT_SHAPE_INFO = (1 << 1), /**< Option to print shape information, invalid before initiation*/
  PRINT_PROP       = (1 << 2), /**< Option to print properties */
  PRINT_PROP_META  = (1 << 3), /**< Option to print properties that describe meta info
                                 e.g) layer activation type for non-activation layer. */
  PRINT_WEIGHTS    = (1 << 4), /**< Option to print weights */
  PRINT_METRIC     = (1 << 5)  /**< Option to print metrics (currently loss only) */
  // clang-format on
} PrintOption;

void LayerNode::printPreset(std::ostream &out, PrintPreset preset) {
  unsigned int flags = 0;

  switch (preset) {
  case PrintPreset::PRINT_ALL:
    flags = PRINT_WEIGHTS | PRINT_METRIC;
    /// fall through intended
  case PrintPreset::PRINT_SUMMARY_META:
    flags |= PRINT_PROP_META;
    /// fall through intended
  case PrintPreset::PRINT_SUMMARY:
    flags |= PRINT_INST_INFO | PRINT_SHAPE_INFO | PRINT_PROP | PRINT_PROP_META;
    break;
  case PrintPreset::PRINT_NONE:
    return;
  default:
    throw ::std::invalid_argument("undefined preset given");
  }
  print(out, flags);
}

void LayerNode::remapIdentifiers(std::function<void(std::string &)> remap_fn) {
  NNTR_THROW_IF(isFinalized(), std::invalid_argument)
    << "cannot remap identifiers after finalized";
  auto &name = std::get<props::Name>(*layer_node_props);
  if (!name.empty()) {
    remap_fn(name.get());
  }

  auto &shared_from = std::get<props::SharedFrom>(*layer_node_props);
  if (!shared_from.empty()) {
    remap_fn(shared_from.get());
  }

  /** remap connections without touching index */
  remapConnections(
    [&remap_fn](std::string &name, unsigned &_) { remap_fn(name); });
}

void LayerNode::remapConnections(
  std::function<void(std::string &, unsigned &)> remap_fn) {
  NNTR_THROW_IF(isFinalized(), std::invalid_argument)
    << "cannot remap identifiers after finalized";

  auto &input_conns =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);

  for (auto &input_layer : input_conns) {
    auto &name = input_layer.get().getName();
    auto &idx = input_layer.get().getIndex();
    remap_fn(name, idx);
  }

  for (auto &output_layer : output_connections) {
    if (output_layer == nullptr) {
      continue;
    }

    auto &name = output_layer->getName();
    auto &idx = output_layer->getIndex();
    remap_fn(name, idx);
  }
}

std::unique_ptr<LayerNode> LayerNode::cloneConfiguration() {
  NNTR_THROW_IF(isFinalized(), std::invalid_argument)
    << "It is prohibited to clone configuration";
  Exporter e;
  exportTo(e, ml::train::ExportMethods::METHOD_STRINGVECTOR);
  e.saveResult(*layer_node_props_realization,
               ml::train::ExportMethods::METHOD_STRINGVECTOR, this);
  auto props = e.getResult<ml::train::ExportMethods::METHOD_STRINGVECTOR>();

  std::vector<std::string> key_val_props;
  key_val_props.reserve(props->size());
  for (auto &entry : *props) {
    key_val_props.push_back(entry.first + "=" + entry.second);
  }

  return createLayerNode(getType(), key_val_props);
}

void LayerNode::printShapeInfo(std::ostream &out) {
  for (unsigned int idx = 0; idx < getNumInputs(); ++idx) {
    out << "input " << run_context->getInput(idx).getDim();
  }
  for (unsigned int idx = 0; idx < getNumWeights(); idx++) {
    out << "weight " << run_context->getWeight(idx).getDim();
  }
  for (unsigned int idx = 0; idx < getNumOutputs(); ++idx) {
    out << "output " << run_context->getOutput(idx).getDim();
  }
}

void LayerNode::printMetric(std::ostream &out) {
  out << "Layer loss value: " << getLoss() << "\n";
}

void LayerNode::print(std::ostream &out, unsigned int flags) {
  /** @todo properly move print to LayerNode */
  if (flags & PRINT_INST_INFO) {
    out << "===================";
    if (getName().empty())
      printInstance(out, this);
    else
      out << "<" << getName() << ">" << std::endl;

    out << "Layer Type: " << getType() << std::endl;
  }

  if (flags & PRINT_SHAPE_INFO) {
    if (run_context) {
      out << "======shape information: " << std::endl;
      printShapeInfo(out);
    }
  }

  if (flags & PRINT_PROP_META) {
    out << "======meta properties: " << std::endl;
    /** @todo print local and layer properties with node_exporter */
    nntrainer::Exporter e;
    getLayer()->exportTo(e, ml::train::ExportMethods::METHOD_STRINGVECTOR);
    auto prop_meta =
      e.getResult<ml::train::ExportMethods::METHOD_STRINGVECTOR>();
    if (prop_meta != nullptr) {
      for (unsigned int i = 0; i < prop_meta->size(); ++i) {
        out << (*prop_meta)[i].first << ": " << (*prop_meta)[i].second << "\n";
      }
    }
  }

  if (flags & PRINT_PROP) {
    out << "======properties: " << std::endl;
    /** @todo print local and layer properties with node_exporter */
  }

  if (flags & PRINT_WEIGHTS) {
    if (run_context) {
      out << "======weights: " << std::endl;
      for (unsigned int idx = 0; idx < getNumWeights(); idx++) {
        out << run_context->getWeight(idx);
      }
    }
  }

  if (flags & PRINT_METRIC) {
    out << "======metrics: " << std::endl;
    printMetric(out);
  }
};
}; // namespace nntrainer

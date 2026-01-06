// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file node_exporter.h
 * @date 09 April 2021
 * @brief NNTrainer Node exporter
 * @see	https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NODE_EXPORTER_H__
#define __NODE_EXPORTER_H__

#include <memory>
#include <regex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <base_properties.h>
#include <common.h>
#include <common_properties.h>
#include <layer.h>
#include <nntrainer_error.h>
#include <util_func.h>

#ifdef ENABLE_TFLITE_INTERPRETER
#include <flatbuffers/flatbuffers.h>
#endif

namespace nntrainer {

/**
 * @brief Forward declaration of TfOpNode
 *
 */
class TfOpNode;

namespace {

/**
 * @brief meta function that return return_type when a method is being called
 *
 * @tparam method returned when certain method is being called
 */
template <ml::train::ExportMethods method> struct return_type {
  using type = void;
};

/**
 * @brief meta function to check return type when the method is string vector
 *
 * @tparam specialized so not given
 */
template <> struct return_type<ml::train::ExportMethods::METHOD_STRINGVECTOR> {
  using type = std::vector<std::pair<std::string, std::string>>;
};

/**
 * @brief meta function to check return type when the method is string vector
 *
 * @tparam specialized so not given
 */
template <> struct return_type<ml::train::ExportMethods::METHOD_TFLITE> {
  using type = TfOpNode;
};

/**
 * @brief Create a empty ptr if null
 *
 * @tparam T type to create
 * @param[in/out] ptr ptr to create
 */
template <typename T> void createIfNull(std::unique_ptr<T> &ptr) {
  if (ptr == nullptr) {
    ptr = std::make_unique<T>();
  }
}

} // namespace

/**
 * @brief Exporter class helps to exports the node information in a predefined
 * way. because each method will require complete different methods, this class
 * exploits visitor pattern to make a custom defined saving method
 *
 */
class Exporter {
public:
  /**
   * @brief Construct a new Exporter object
   *
   */
  Exporter();

#ifdef ENABLE_TFLITE_INTERPRETER
  /**
   * @brief Construct a new Exporter object with flatbuffer builder
   *
   */
  Exporter(flatbuffers::FlatBufferBuilder *fbb);

  flatbuffers::FlatBufferBuilder *getFlatbufferBuilder() { return fbb; }
#endif

  /**
   * @brief Destroy the Exporter object
   *
   */
  ~Exporter();

  /**
   * @brief this function iterates over the property and process the property in
   * a designated way.
   *
   * @tparam Ts type of elements
   * @tparam NodeType NodeType element
   * @param props tuple that contains properties
   * @param method method to export
   * @param self this pointer to the layer which is being exported
   */
  template <typename... Ts, typename NodeType = void>
  void saveResult(const std::tuple<Ts...> &props,
                  ml::train::ExportMethods method,
                  const NodeType *self = nullptr) {
    switch (method) {
    case ml::train::ExportMethods::METHOD_STRINGVECTOR: {
      createIfNull(stored_result);

      /**
       * @brief function to pass to the iterate_prop, this saves the property
       * to stored_result
       *
       * @param prop property property to pass
       * @param index index of the current property
       */
      auto callable = [this](auto &&prop, size_t index) {
        if (!prop.empty()) {
          std::string key = getPropKey(prop);
          stored_result->emplace_back(std::move(key), to_string(prop));
        }
      };
      iterate_prop(callable, props);
    } break;
    case ml::train::ExportMethods::METHOD_TFLITE:
      saveTflResult(props, self);
      break;
    case ml::train::ExportMethods::METHOD_UNDEFINED:
    /// fall through intended
    default:
      throw exception::not_supported("given method is not supported yet");
    }

    is_exported = true;
  }

  /**
   * @brief Get the result object
   * @note @a unique_ptr here will be a member of @a this. The ownership is
   * passed to the caller, thus after getResult, the target member is cleared.
   * This is intended design choice to mimic single function call, between @a
   * saveResult and @a getResult
   *
   * @tparam methods method to get
   * @tparam T appropriate return type regarding the export method
   * @return std::unique_ptr<T> predefined return type according to the method.
   * @retval nullptr not exported
   */
  template <ml::train::ExportMethods methods,
            typename T = typename return_type<methods>::type>
  std::unique_ptr<T> getResult();

private:
  /**
   * @brief ProtoType to enable saving result to tfnode
   *
   * @tparam PropsType propsType PropsType to receive with self
   * @tparam NodeType NodeType of current layer.
   * @param props properties to get with @a self
   * @param self @a this of the current layer
   */
  template <typename PropsType, typename NodeType>
  void saveTflResult(const PropsType &props, const NodeType *self);

#ifdef ENABLE_TFLITE_INTERPRETER
  std::unique_ptr<TfOpNode> tf_node;   /**< created node from the export */
  flatbuffers::FlatBufferBuilder *fbb; /**< flatbuffer builder */
#endif

  std::unique_ptr<std::vector<std::pair<std::string, std::string>>>
    stored_result; /**< stored result */

  /// consider changing this to a promise / future if there is a async function
  /// involved to `saveResult`
  bool is_exported; /**< boolean to check if exported */
};

/**
 * @brief ProtoType to enable saving result to tfnode
 *
 * @tparam PropsType propsType PropsType to receive with self
 * @tparam NodeType NodeType of current layer.
 * @param props properties to get with @a self
 * @param self @a this of the current layer
 */
template <typename PropsType, typename NodeType>
void Exporter::saveTflResult(const PropsType &props, const NodeType *self) {
  NNTR_THROW_IF(true, nntrainer::exception::not_supported)
    << "given node cannot be converted to tfnode, type: "
    << typeid(self).name();
}

#ifdef ENABLE_TFLITE_INTERPRETER
namespace props {
class Name;
class Unit;
class Flatten;
class Distribute;
class Trainable;
class WeightDimension;
class InputShape;
class WeightRegularizer;
class WeightRegularizerConstant;
class WeightInitializer;
class WeightDecay;
class BiasDecay;
class BiasInitializer;
class SharedFrom;
class InputConnection;
class ClipGradByGlobalNorm;
class DisableBias;
class Activation;
class BatchNormalization;
class Packed;
class LossScaleForMixed;
class InPlaceProp;
class InPlaceDirectionProp;
class Exponent;
class StartIndex;
class EndIndex;
} // namespace props

class LayerNode;
/**
 * @copydoc template <typename PropsType, typename NodeType> void
 * Exporter::saveTflResult(const PropsType &props, const NodeType *self);
 */
template <>
void Exporter::saveTflResult(
  const std::tuple<props::Name, props::Distribute, props::Trainable,
                   std::vector<props::InputConnection>,
                   std::vector<props::InputShape>, props::WeightDimension,
                   props::SharedFrom, props::ClipGradByGlobalNorm,
                   props::Packed, props::WeightDtype, props::LossScaleForMixed,
                   props::ComputeEngine> &props,
  const LayerNode *self);

class BatchNormalizationLayer;
/**
 * @copydoc template <typename PropsType, typename NodeType> void
 * Exporter::saveTflResult(const PropsType &props, const NodeType *self);
 */
template <>
void Exporter::saveTflResult(
  const std::tuple<props::Epsilon, props::MuInitializer, props::VarInitializer,
                   props::BetaInitializer, props::GammaInitializer,
                   props::Momentum, props::Axis, props::WeightDecay,
                   props::BiasDecay> &props,
  const BatchNormalizationLayer *self);

class LayerImpl;

/**
 * @copydoc template <typename PropsType, typename NodeType> void
 * Exporter::saveTflResult(const PropsType &props, const NodeType *self);
 */
template <>
void Exporter::saveTflResult(
  const std::tuple<props::WeightRegularizer, props::WeightRegularizerConstant,
                   props::WeightInitializer, props::WeightDecay,
                   props::BiasDecay, props::BiasInitializer, props::DisableBias,
                   props::Print> &props,
  const LayerImpl *self);

class FullyConnectedLayer;
/**
 * @copydoc template <typename PropsType, typename NodeType> void
 * Exporter::saveTflResult(const PropsType &props, const NodeType *self);
 */
template <>
void Exporter::saveTflResult(
  const std::tuple<props::Unit, props::LoraRank, props::LoraAlpha> &props,
  const FullyConnectedLayer *self);

class ActivationLayer;
/**
 * @copydoc template <typename PropsType, typename NodeType> void
 * Exporter::saveTflResult(const PropsType &props, const NodeType *self);
 */
template <>
void Exporter::saveTflResult(const std::tuple<props::Activation> &props,
                             const ActivationLayer *self);

class Conv2DLayer;
/**
 * @copydoc template <typename PropsType, typename NodeType> void
 * Exporter::saveTflResult(const PropsType &props, const NodeType *self);
 */
template <>
void Exporter::saveTflResult(
  const std::tuple<props::FilterSize, std::array<props::KernelSize, 2>,
                   std::array<props::Stride, 2>, props::Padding2D,
                   std::array<props::Dilation, 2>> &props,
  const Conv2DLayer *self);

class InputLayer;
/**
 * @copydoc template <typename PropsType, typename NodeType> void
 * Exporter::saveTflResult(const PropsType &props, const NodeType *self);
 */
template <>
void Exporter::saveTflResult(
  const std::tuple<props::Normalization, props::Standardization> &props,
  const InputLayer *self);

class Pooling2DLayer;
/**
 * @copydoc template <typename PropsType, typename NodeType> void
 * Exporter::saveTflResult(const PropsType &props, const NodeType *self);
 */
template <>
void Exporter::saveTflResult(
  const std::tuple<props::PoolingType, std::vector<props::PoolSize>,
                   std::array<props::Stride, 2>, props::Padding2D> &props,
  const Pooling2DLayer *self);

class ReshapeLayer;
/**
 * @copydoc template <typename PropsType, typename NodeType> void
 * Exporter::saveTflResult(const PropsType &props, const NodeType *self);
 */
template <>
void Exporter::saveTflResult(const std::tuple<props::TargetShape> &props,
                             const ReshapeLayer *self);

class FlattenLayer;
/**
 * @copydoc template <typename PropsType, typename NodeType> void
 * Exporter::saveTflResult(const PropsType &props, const NodeType *self);
 */
template <>
void Exporter::saveTflResult(const std::tuple<props::TargetShape> &props,
                             const FlattenLayer *self);

class AdditionLayer;
/**
 * @copydoc template <typename PropsType, typename NodeType> void
 * Exporter::saveTflResult(const PropsType &props, const NodeType *self);
 */
template <>
void Exporter::saveTflResult(const std::tuple<> &props,
                             const AdditionLayer *self);
#endif

/**
 * @brief base case of iterate_prop, iterate_prop iterates the given tuple
 *
 * @tparam I size of tuple(automated)
 * @tparam Callable generic lambda to be called during iteration
 * @tparam Ts types from tuple
 * @param c callable generic lambda
 * @param tup tuple to be iterated
 * @return void
 */
template <size_t I = 0, typename Callable, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type
iterate_prop(Callable &&c, const std::tuple<Ts...> &tup) {
  // end of recursion;
}

/**
 * @brief base case of iterate_prop, iterate_prop iterates the given tuple
 *
 * @tparam I size of tuple(automated)
 * @tparam Callable generic lambda to be called during iteration
 * @tparam Ts types from tuple
 * @param c callable generic lambda
 * @param tup tuple to be iterated
 * @return not used
 */
template <size_t I = 0, typename Callable, typename... Ts>
typename std::enable_if<(I < sizeof...(Ts)), void>::type
iterate_prop(Callable &&c, const std::tuple<Ts...> &tup) {
  c(std::get<I>(tup), I);

  iterate_prop<I + 1>(c, tup);
}

/**
 * @copydoc  template <size_t I = 0, typename Callable, typename... Ts>
typename std::enable_if<(I < sizeof...(Ts)), void>::type iterate_prop(Callable
&&c, const std::tuple<Ts...> &tup)
 */
template <size_t I = 0, typename Callable, typename... Ts>
typename std::enable_if<(I < sizeof...(Ts)), void>::type
iterate_prop(Callable &&c, std::tuple<Ts...> &tup) {
  c(std::get<I>(tup), I);

  iterate_prop<I + 1>(c, tup);
}

/**
 * @brief load property from the api formatted string ({"key=value",
 * "key1=value1"})
 *
 * @tparam Tuple tuple type
 * @param string_vector api formatted string;
 * @param[out] props props to be iterated
 * @return std::vector<std::string> vector of string that is not used while
 * setting the property
 */
template <typename Tuple>
std::vector<std::string>
loadProperties(const std::vector<std::string> &string_vector, Tuple &&props) {
  std::vector<std::string> string_vector_splited;
  string_vector_splited.reserve(string_vector.size());
  for (auto &item : string_vector) {
    auto splited = split(item, std::regex("\\|"));
    string_vector_splited.insert(string_vector_splited.end(), splited.begin(),
                                 splited.end());
  }

  std::vector<std::pair<std::string, std::string>> left;
  left.reserve(string_vector_splited.size());
  std::transform(string_vector_splited.begin(), string_vector_splited.end(),
                 std::back_inserter(left), [](const std::string &property) {
                   std::string key, value;
                   int status = getKeyValue(property, key, value);
                   NNTR_THROW_IF(status != ML_ERROR_NONE, std::invalid_argument)
                     << "parsing property failed, original format: \""
                     << property << "\"";
                   return std::make_pair(key, value);
                 });

  auto callable = [&left](auto &&prop, size_t index) {
    std::string prop_key = getPropKey(prop);

    for (auto iter = left.begin(); iter < left.end();) {
      if (istrequal(prop_key, iter->first) == true) {
        from_string(iter->second, prop);
        iter = left.erase(iter);
      } else {
        iter++;
      }
    }
  };

  iterate_prop(callable, props);

  std::vector<std::string> remainder;
  remainder.reserve(left.size());

  std::transform(left.begin(), left.end(), std::back_inserter(remainder),
                 [](const decltype(left)::value_type &v) {
                   return v.first + "=" + v.second;
                 });

  return remainder;
}

} // namespace nntrainer
#endif // __NODE_EXPORTER_H__

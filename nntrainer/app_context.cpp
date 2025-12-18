// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   app_context.cpp
 * @date   10 November 2020
 * @brief  This file contains app context related functions and classes that
 * manages the global configuration of the current environment
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <iniparser.h>

#include <app_context.h>
#include <layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer.h>
#include <util_func.h>

#include <adam.h>
#include <adamw.h>
#include <sgd.h>

#include <activation_layer.h>
#include <add_layer.h>
#include <addition_layer.h>
#include <attention_layer.h>
#include <bn_layer.h>
#include <cast_layer.h>
#include <centroid_knn.h>
#include <channel_shuffle.h>
#include <concat_layer.h>
#include <constant_derivative_loss_layer.h>
#include <conv1d_layer.h>
#include <conv2d_layer.h>
#include <conv2d_transpose_layer.h>
#include <cosine_layer.h>
#include <cross_entropy_sigmoid_loss_layer.h>
#include <cross_entropy_softmax_loss_layer.h>
#include <divide_layer.h>
#include <dropout.h>
#include <dynamic_library_loader.h>
#include <embedding.h>
#include <fc_layer.h>
#include <flatten_layer.h>
#include <gather_layer.h>
#include <gru.h>
#include <grucell.h>
#include <identity_layer.h>
#include <input_layer.h>
#include <layer_normalization_layer.h>
#include <lr_scheduler_constant.h>
#include <lr_scheduler_cosine.h>
#include <lr_scheduler_exponential.h>
#include <lr_scheduler_linear.h>
#include <lr_scheduler_step.h>
#include <lstm.h>
#include <lstmcell.h>
#include <matmul_layer.h>
#include <mol_attention_layer.h>
#include <mse_loss_layer.h>
#include <multi_head_attention_layer.h>
#include <multiout_layer.h>
#include <multiply_layer.h>
#include <negative_layer.h>
#include <nntrainer_error.h>
#include <permute_layer.h>
#include <plugged_layer.h>
#include <plugged_optimizer.h>
#include <pooling2d_layer.h>
#include <positional_encoding_layer.h>
#include <pow_layer.h>
#include <preprocess_flip_layer.h>
#include <preprocess_l2norm_layer.h>
#include <preprocess_translate_layer.h>
#include <reduce_mean_layer.h>
#include <reduce_sum_layer.h>
#include <rnn.h>
#include <rnncell.h>
#include <sine_layer.h>
#include <slice_layer.h>
#include <split_layer.h>
#include <sqrt_layer.h>
#include <subtract_layer.h>
#include <tangent_layer.h>
#include <tensor_layer.h>
#include <time_dist.h>
#include <upsample2d_layer.h>
#include <weight_layer.h>
#include <zoneout_lstmcell.h>

#ifdef ENABLE_TFLITE_BACKBONE
#include <tflite_layer.h>
#endif

#ifdef ENABLE_NNSTREAMER_BACKBONE
#include <nnstreamer_layer.h>
#endif

/// add #ifdef across platform
static std::string solib_suffix = ".so";
static std::string layerlib_suffix = "layer.so";
static std::string optimizerlib_suffix = "optimizer.so";
static const std::string func_tag = "[AppContext] ";

#ifdef NNTRAINER_CONF_PATH
constexpr const char *DEFAULT_CONF_PATH = NNTRAINER_CONF_PATH;
#else
constexpr const char *DEFAULT_CONF_PATH = "/etc/nntrainer.ini";
#endif

constexpr const char *getConfPath() { return DEFAULT_CONF_PATH; }

namespace nntrainer {

namespace {

/**
 * @brief Get the plugin path from conf ini
 *
 * @return std::string plugin path
 */
std::string getConfig(const std::string &key) {
  std::string conf_path{getConfPath()};

  ml_logd("%s conf path: %s", func_tag.c_str(), conf_path.c_str());
  if (!isFileExist(conf_path)) {
    ml_logw(
      "%s conf path does not exist, skip getting plugin path from the conf",
      func_tag.c_str());
    return std::string();
  }

  dictionary *ini = iniparser_load(conf_path.c_str());
  NNTR_THROW_IF(ini == nullptr, std::runtime_error)
    << func_tag << "loading ini failed";

  std::string value;
  int nsec = iniparser_getnsec(ini);
  for (int i = 0; i < nsec; i++) {
    std::string query(iniparser_getsecname(ini, i));
    query += ":";
    query += key;

    value = std::string(iniparser_getstring(ini, query.c_str(), ""));
    if (!value.empty())
      break;
  }

  if (value.empty())
    ml_logd("key %s is not found in config(%s)", key.c_str(),
            conf_path.c_str());

  iniparser_freedict(ini);

  return value;
}

/**
 * @brief Get the plugin paths
 *
 * @return std::vector<std::string> list of paths to search for
 */
std::vector<std::string> getPluginPaths() {
  std::vector<std::string> ret;

  /*** @note NNTRAINER_PATH is an environment variable stating a @a directory
   * where you would like to look for the layers, while NNTRAINER_CONF_PATH is a
   * (buildtime hardcoded @a file path) to locate configuration file *.ini file
   */
  /*** @note for now, NNTRAINER_PATH is a SINGLE PATH rather than series of path
   * like PATH environment variable. this could be improved but for now, it is
   * enough
   */
  const char *env_path = std::getenv("NNTRAINER_PATH");
  if (env_path != nullptr) {
    if (isFileExist(env_path)) {
      ml_logd("NNTRAINER_PATH is defined and valid. path: %s", env_path);
      ret.emplace_back(env_path);
    } else {
      ml_logw("NNTRAINER_PATH is given but it is not valid. path: %s",
              env_path);
    }
  }

  std::string plugin_path = getConfig("layer");
  if (!plugin_path.empty()) {
    ret.emplace_back(plugin_path);
    ml_logd("DEFAULT CONF PATH, path: %s", plugin_path.c_str());
  }

  return ret;
}

/**
 * @brief Get the Full Path from given string
 * @details path is resolved in the following order
 * 1) if @a path is absolute, return path
 * ----------------------------------------
 * 2) if @a base == "" && @a path == "", return "."
 * 3) if @a base == "" && @a path != "", return @a path
 * 4) if @a base != "" && @a path == "", return @a base
 * 5) if @a base != "" && @a path != "", return @a base + "/" + path
 *
 * @param path path to calculate from base
 * @param base base path
 * @return const std::string
 */
const std::string getFullPath(const std::string &path,
                              const std::string &base) {
  /// if path is absolute, return path
  if (std::filesystem::path(path).is_absolute()) {
    return path;
  }

  if (base == std::string()) {
    return path == std::string() ? "." : path;
  }

  return path == std::string()
           ? base
           : std::filesystem::path(base).append(path).string();
}

} // namespace

std::mutex factory_mutex;

std::once_flag global_app_context_init_flag;

void AppContext::initialize() noexcept {
  try {
    setMemAllocator(std::make_shared<MemAllocator>());

    add_default_object();
    add_extension_object();
  } catch (std::exception &e) {
    ml_loge("registering layers failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("registering layer failed due to unknown reason");
  }
};

void AppContext::add_default_object() {
  /// @note all layers should be added to the app_context to guarantee that
  /// createLayer/createOptimizer class is created
  using OptType = ml::train::OptimizerType;
  registerFactory(nntrainer::createOptimizer<SGD>, SGD::type, OptType::SGD);
  registerFactory(nntrainer::createOptimizer<Adam>, Adam::type, OptType::ADAM);
  registerFactory(nntrainer::createOptimizer<AdamW>, AdamW::type,
                  OptType::ADAMW);
  registerFactory(AppContext::unknownFactory<nntrainer::Optimizer>, "unknown",
                  OptType::UNKNOWN);

  using LRType = LearningRateSchedulerType;
  registerFactory(
    ml::train::createLearningRateScheduler<ConstantLearningRateScheduler>,
    ConstantLearningRateScheduler::type, LRType::CONSTANT);
  registerFactory(
    ml::train::createLearningRateScheduler<ExponentialLearningRateScheduler>,
    ExponentialLearningRateScheduler::type, LRType::EXPONENTIAL);
  registerFactory(
    ml::train::createLearningRateScheduler<StepLearningRateScheduler>,
    StepLearningRateScheduler::type, LRType::STEP);
  registerFactory(ml::train::createLearningRateScheduler<
                    CosineAnnealingLearningRateScheduler>,
                  CosineAnnealingLearningRateScheduler::type, LRType::COSINE);
  registerFactory(
    ml::train::createLearningRateScheduler<LinearLearningRateScheduler>,
    LinearLearningRateScheduler::type, LRType::LINEAR);

  using LayerType = ml::train::LayerType;
  registerFactory(nntrainer::createLayer<InputLayer>, InputLayer::type,
                  LayerType::LAYER_IN);
  registerFactory(nntrainer::createLayer<WeightLayer>, WeightLayer::type,
                  LayerType::LAYER_WEIGHT);
  registerFactory(nntrainer::createLayer<AddLayer>, AddLayer::type,
                  LayerType::LAYER_ADD);
  registerFactory(nntrainer::createLayer<SubtractLayer>, SubtractLayer::type,
                  LayerType::LAYER_SUBTRACT);
  registerFactory(nntrainer::createLayer<MultiplyLayer>, MultiplyLayer::type,
                  LayerType::LAYER_MULTIPLY);
  registerFactory(nntrainer::createLayer<DivideLayer>, DivideLayer::type,
                  LayerType::LAYER_DIVIDE);
  registerFactory(nntrainer::createLayer<PowLayer>, PowLayer::type,
                  LayerType::LAYER_POW);
  registerFactory(nntrainer::createLayer<SQRTLayer>, SQRTLayer::type,
                  LayerType::LAYER_SQRT);
  registerFactory(nntrainer::createLayer<SineLayer>, SineLayer::type,
                  LayerType::LAYER_SINE);
  registerFactory(nntrainer::createLayer<CosineLayer>, CosineLayer::type,
                  LayerType::LAYER_COSINE);
  registerFactory(nntrainer::createLayer<TangentLayer>, TangentLayer::type,
                  LayerType::LAYER_TANGENT);
  registerFactory(nntrainer::createLayer<MatMulLayer>, MatMulLayer::type,
                  LayerType::LAYER_MATMUL);
  registerFactory(nntrainer::createLayer<CastLayer>, CastLayer::type,
                  LayerType::LAYER_CAST);
  registerFactory(nntrainer::createLayer<GatherLayer>, GatherLayer::type,
                  LayerType::LAYER_GATHER);
  registerFactory(nntrainer::createLayer<SliceLayer>, SliceLayer::type,
                  LayerType::LAYER_SLICE);
  registerFactory(nntrainer::createLayer<NegativeLayer>, NegativeLayer::type,
                  LayerType::LAYER_NEG);
  registerFactory(nntrainer::createLayer<FullyConnectedLayer>,
                  FullyConnectedLayer::type, LayerType::LAYER_FC);
  registerFactory(nntrainer::createLayer<BatchNormalizationLayer>,
                  BatchNormalizationLayer::type, LayerType::LAYER_BN);
  registerFactory(nntrainer::createLayer<LayerNormalizationLayer>,
                  LayerNormalizationLayer::type,
                  LayerType::LAYER_LAYER_NORMALIZATION);
  registerFactory(nntrainer::createLayer<Conv2DLayer>, Conv2DLayer::type,
                  LayerType::LAYER_CONV2D);
  registerFactory(nntrainer::createLayer<Conv2DTransposeLayer>,
                  Conv2DTransposeLayer::type,
                  LayerType::LAYER_CONV2D_TRANSPOSE);
  registerFactory(nntrainer::createLayer<Conv1DLayer>, Conv1DLayer::type,
                  LayerType::LAYER_CONV1D);
  registerFactory(nntrainer::createLayer<Pooling2DLayer>, Pooling2DLayer::type,
                  LayerType::LAYER_POOLING2D);
  registerFactory(nntrainer::createLayer<FlattenLayer>, FlattenLayer::type,
                  LayerType::LAYER_FLATTEN);
  registerFactory(nntrainer::createLayer<ReshapeLayer>, ReshapeLayer::type,
                  LayerType::LAYER_RESHAPE);
  registerFactory(nntrainer::createLayer<ActivationLayer>,
                  ActivationLayer::type, LayerType::LAYER_ACTIVATION);
  registerFactory(nntrainer::createLayer<AdditionLayer>, AdditionLayer::type,
                  LayerType::LAYER_ADDITION);
  registerFactory(nntrainer::createLayer<ConcatLayer>, ConcatLayer::type,
                  LayerType::LAYER_CONCAT);
  registerFactory(nntrainer::createLayer<MultiOutLayer>, MultiOutLayer::type,
                  LayerType::LAYER_MULTIOUT);
  registerFactory(nntrainer::createLayer<EmbeddingLayer>, EmbeddingLayer::type,
                  LayerType::LAYER_EMBEDDING);
  registerFactory(nntrainer::createLayer<RNNLayer>, RNNLayer::type,
                  LayerType::LAYER_RNN);
  registerFactory(nntrainer::createLayer<RNNCellLayer>, RNNCellLayer::type,
                  LayerType::LAYER_RNNCELL);
  registerFactory(nntrainer::createLayer<LSTMLayer>, LSTMLayer::type,
                  LayerType::LAYER_LSTM);
  registerFactory(nntrainer::createLayer<LSTMCellLayer>, LSTMCellLayer::type,
                  LayerType::LAYER_LSTMCELL);
  registerFactory(nntrainer::createLayer<ZoneoutLSTMCellLayer>,
                  ZoneoutLSTMCellLayer::type,
                  LayerType::LAYER_ZONEOUT_LSTMCELL);
  registerFactory(nntrainer::createLayer<SplitLayer>, SplitLayer::type,
                  LayerType::LAYER_SPLIT);
  registerFactory(nntrainer::createLayer<GRULayer>, GRULayer::type,
                  LayerType::LAYER_GRU);
  registerFactory(nntrainer::createLayer<GRUCellLayer>, GRUCellLayer::type,
                  LayerType::LAYER_GRUCELL);
  registerFactory(nntrainer::createLayer<PermuteLayer>, PermuteLayer::type,
                  LayerType::LAYER_PERMUTE);
  registerFactory(nntrainer::createLayer<DropOutLayer>, DropOutLayer::type,
                  LayerType::LAYER_DROPOUT);
  registerFactory(nntrainer::createLayer<AttentionLayer>, AttentionLayer::type,
                  LayerType::LAYER_ATTENTION);
  registerFactory(nntrainer::createLayer<MoLAttentionLayer>,
                  MoLAttentionLayer::type, LayerType::LAYER_MOL_ATTENTION);
  registerFactory(nntrainer::createLayer<MultiHeadAttentionLayer>,
                  MultiHeadAttentionLayer::type,
                  LayerType::LAYER_MULTI_HEAD_ATTENTION);
  registerFactory(nntrainer::createLayer<ReduceMeanLayer>,
                  ReduceMeanLayer::type, LayerType::LAYER_REDUCE_MEAN);
  registerFactory(nntrainer::createLayer<ReduceSumLayer>, ReduceSumLayer::type,
                  LayerType::LAYER_REDUCE_SUM);
  registerFactory(nntrainer::createLayer<PositionalEncodingLayer>,
                  PositionalEncodingLayer::type,
                  LayerType::LAYER_POSITIONAL_ENCODING);
  registerFactory(nntrainer::createLayer<IdentityLayer>, IdentityLayer::type,
                  LayerType::LAYER_IDENTITY);
  registerFactory(nntrainer::createLayer<Upsample2dLayer>,
                  Upsample2dLayer::type, LayerType::LAYER_UPSAMPLE2D);

  registerFactory(nntrainer::createLayer<ChannelShuffle>, ChannelShuffle::type,
                  LayerType::LAYER_CHANNEL_SHUFFLE);

#ifdef ENABLE_NNSTREAMER_BACKBONE
  registerFactory(nntrainer::createLayer<NNStreamerLayer>,
                  NNStreamerLayer::type, LayerType::LAYER_BACKBONE_NNSTREAMER);
#endif
#ifdef ENABLE_TFLITE_BACKBONE
  registerFactory(nntrainer::createLayer<TfLiteLayer>, TfLiteLayer::type,
                  LayerType::LAYER_BACKBONE_TFLITE);
#endif
  registerFactory(nntrainer::createLayer<CentroidKNN>, CentroidKNN::type,
                  LayerType::LAYER_CENTROID_KNN);

  /** preprocess layers */
  registerFactory(nntrainer::createLayer<PreprocessFlipLayer>,
                  PreprocessFlipLayer::type, LayerType::LAYER_PREPROCESS_FLIP);
  registerFactory(nntrainer::createLayer<PreprocessTranslateLayer>,
                  PreprocessTranslateLayer::type,
                  LayerType::LAYER_PREPROCESS_TRANSLATE);
  registerFactory(nntrainer::createLayer<PreprocessL2NormLayer>,
                  PreprocessL2NormLayer::type,
                  LayerType::LAYER_PREPROCESS_L2NORM);

  /** register losses */
  registerFactory(nntrainer::createLayer<MSELossLayer>, MSELossLayer::type,
                  LayerType::LAYER_LOSS_MSE);
  registerFactory(nntrainer::createLayer<CrossEntropySigmoidLossLayer>,
                  CrossEntropySigmoidLossLayer::type,
                  LayerType::LAYER_LOSS_CROSS_ENTROPY_SIGMOID);
  registerFactory(nntrainer::createLayer<CrossEntropySoftmaxLossLayer>,
                  CrossEntropySoftmaxLossLayer::type,
                  LayerType::LAYER_LOSS_CROSS_ENTROPY_SOFTMAX);
  registerFactory(nntrainer::createLayer<ConstantDerivativeLossLayer>,
                  ConstantDerivativeLossLayer::type,
                  LayerType::LAYER_LOSS_CONSTANT_DERIVATIVE);

  registerFactory(nntrainer::createLayer<TimeDistLayer>, TimeDistLayer::type,
                  LayerType::LAYER_TIME_DIST);

  registerFactory(AppContext::unknownFactory<nntrainer::Layer>, "unknown",
                  LayerType::LAYER_UNKNOWN);
}

void AppContext::add_extension_object() {
  auto dir_list = getPluginPaths();

  for (auto &path : dir_list) {
    try {
      registerPluggableFromDirectory(path);
    } catch (std::exception &e) {
      ml_logw("tried to register extension from %s but failed, reason: %s",
              path.c_str(), e.what());
    }
  }
}

void AppContext::setWorkingDirectory(const std::string &base) {
  if (!std::filesystem::is_directory(base)) {
    std::stringstream ss;
    ss << func_tag << "path is not directory or has no permission: " << base;
    throw std::invalid_argument(ss.str().c_str());
  }

  char *ret = getRealpath(base.c_str(), nullptr);

  if (ret == nullptr) {
    std::stringstream ss;
    ss << func_tag << "failed to get canonical path for the path: ";
    throw std::invalid_argument(ss.str().c_str());
  }

  working_path_base = std::string(ret);
  ml_logd("working path base has set: %s", working_path_base.c_str());
  free(ret);
}

const std::string AppContext::getWorkingPath(const std::string &path) {
  return getFullPath(path, working_path_base);
}

/**
 * @brief base case of iterate_prop, iterate_prop iterates the given tuple
 *
 * @tparam I size of tuple(automated)
 * @tparam V container type of properties
 * @tparam Ts types from tuple
 * @param prop property container to be added to
 * @param tup tuple to be iterated
 * @return void
 */
template <size_t I = 0, typename V, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type inline parse_properties(
  V &props, std::tuple<Ts...> &tup) {
  // end of recursion.
}

/**
 * @brief base case of iterate_prop, iterate_prop iterates the given tuple
 *
 * @tparam I size of tuple(automated)
 * @tparam V container type of properties
 * @tparam Ts types from tuple
 * @param prop property container to be added to
 * @param tup tuple to be iterated
 * @return void
 */
template <size_t I = 0, typename V, typename... Ts>
  typename std::enable_if <
  I<sizeof...(Ts), void>::type inline parse_properties(V &props,
                                                       std::tuple<Ts...> &tup) {
  std::string name = std::get<I>(tup);
  std::string prop = getConfig(name);
  if (!prop.empty())
    props.push_back(name + "=" + prop);

  parse_properties<I + 1>(props, tup);
}

const std::vector<std::string> AppContext::getProperties(void) {
  std::vector<std::string> properties;

  auto props = std::tuple("fsu", "fsu_path");
  parse_properties(properties, props);

  return properties;
}

int AppContext::registerLayer(const std::string &library_path,
                              const std::string &base_path) {
  const std::string full_path = getFullPath(library_path, base_path);

  void *handle = DynamicLibraryLoader::loadLibrary(full_path.c_str(),
                                                   RTLD_LAZY | RTLD_LOCAL);
  const char *error_msg = DynamicLibraryLoader::getLastError();

  NNTR_THROW_IF(handle == nullptr, std::invalid_argument)
    << func_tag << "open plugin failed, reason: " << error_msg;

  nntrainer::LayerPluggable *pluggable =
    reinterpret_cast<nntrainer::LayerPluggable *>(
      DynamicLibraryLoader::loadSymbol(handle, "ml_train_layer_pluggable"));

  error_msg = DynamicLibraryLoader::getLastError();
  auto close_dl = [handle] { DynamicLibraryLoader::freeLibrary(handle); };
  NNTR_THROW_IF_CLEANUP(error_msg != nullptr || pluggable == nullptr,
                        std::invalid_argument, close_dl)
    << func_tag << "loading symbol failed, reason: " << error_msg;

  auto layer = pluggable->createfunc();
  NNTR_THROW_IF_CLEANUP(layer == nullptr, std::invalid_argument, close_dl)
    << func_tag << "created pluggable layer is null";
  auto type = layer->getType();
  NNTR_THROW_IF_CLEANUP(type == "", std::invalid_argument, close_dl)
    << func_tag << "custom layer must specify type name, but it is empty";
  pluggable->destroyfunc(layer);

  FactoryType<nntrainer::Layer> factory_func =
    [pluggable](const PropsType &prop) {
      std::unique_ptr<nntrainer::Layer> layer =
        std::make_unique<internal::PluggedLayer>(pluggable);

      return layer;
    };

  return registerFactory<nntrainer::Layer>(std::move(factory_func), type);
}

int AppContext::registerOptimizer(const std::string &library_path,
                                  const std::string &base_path) {
  const std::string full_path = getFullPath(library_path, base_path);

  void *handle = DynamicLibraryLoader::loadLibrary(full_path.c_str(),
                                                   RTLD_LAZY | RTLD_LOCAL);
  const char *error_msg = DynamicLibraryLoader::getLastError();

  NNTR_THROW_IF(handle == nullptr, std::invalid_argument)
    << func_tag << "open plugin failed, reason: " << error_msg;

  nntrainer::OptimizerPluggable *pluggable =
    reinterpret_cast<nntrainer::OptimizerPluggable *>(
      DynamicLibraryLoader::loadSymbol(handle, "ml_train_optimizer_pluggable"));

  error_msg = DynamicLibraryLoader::getLastError();
  auto close_dl = [handle] { DynamicLibraryLoader::freeLibrary(handle); };
  NNTR_THROW_IF_CLEANUP(error_msg != nullptr || pluggable == nullptr,
                        std::invalid_argument, close_dl)
    << func_tag << "loading symbol failed, reason: " << error_msg;

  auto optimizer = pluggable->createfunc();
  NNTR_THROW_IF_CLEANUP(optimizer == nullptr, std::invalid_argument, close_dl)
    << func_tag << "created pluggable optimizer is null";
  auto type = optimizer->getType();
  NNTR_THROW_IF_CLEANUP(type == "", std::invalid_argument, close_dl)
    << func_tag << "custom optimizer must specify type name, but it is empty";
  pluggable->destroyfunc(optimizer);

  FactoryType<nntrainer::Optimizer> factory_func =
    [pluggable](const PropsType &prop) {
      std::unique_ptr<nntrainer::Optimizer> optimizer =
        std::make_unique<internal::PluggedOptimizer>(pluggable);

      return optimizer;
    };

  return registerFactory<nntrainer::Optimizer>(std::move(factory_func), type);
}

std::vector<int>
AppContext::registerPluggableFromDirectory(const std::string &base_path) {
  const auto directory_exist = std::filesystem::is_directory(base_path);

  NNTR_THROW_IF(!directory_exist, std::invalid_argument)
    << func_tag << "failed to open the directory: " << base_path;

  std::vector<int> keys = {};

  for (const auto &entry : std::filesystem::directory_iterator(base_path)) {
    const auto &entry_name = entry.path().string();

    if (endswith(entry_name, solib_suffix)) {
      if (endswith(entry_name, layerlib_suffix)) {
        try {
          int key = registerLayer(entry_name, base_path);
          keys.emplace_back(key);
        } catch (std::exception &e) {
          throw;
        }
      } else if (endswith(entry_name, optimizerlib_suffix)) {
        try {
          int key = registerOptimizer(entry_name, base_path);
          keys.emplace_back(key);
        } catch (std::exception &e) {
          throw;
        }
      }
    }
  }

  return keys;
}

template <typename T>
const int AppContext::registerFactory(const FactoryType<T> factory,
                                      const std::string &key,
                                      const int int_key) {
  static_assert(isSupported<T>::value,
                "given type is not supported for current app context");

  auto &index = std::get<IndexType<T>>(factory_map);
  auto &str_map = std::get<StrIndexType<T>>(index);
  auto &int_map = std::get<IntIndexType>(index);

  std::string assigned_key = key == "" ? factory({})->getType() : key;

  std::transform(assigned_key.begin(), assigned_key.end(), assigned_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::lock_guard<std::mutex> lock(factory_mutex);
  if (str_map.find(assigned_key) != str_map.end()) {
    std::stringstream ss;
    ss << "cannot register factory with already taken key: " << key;
    throw std::invalid_argument(ss.str().c_str());
  }

  if (int_key != -1 && int_map.find(int_key) != int_map.end()) {
    std::stringstream ss;
    ss << "cannot register factory with already taken int key: " << int_key;
    throw std::invalid_argument(ss.str().c_str());
  }

  int assigned_int_key = int_key == -1 ? str_map.size() + 1 : int_key;

  str_map[assigned_key] = factory;
  int_map[assigned_int_key] = assigned_key;

  ml_logd("factory has registered with key: %s, int_key: %d",
          assigned_key.c_str(), assigned_int_key);

  return assigned_int_key;
}

/**
 * @copydoc const int AppContext::registerFactory
 */
template const int AppContext::registerFactory<nntrainer::Optimizer>(
  const FactoryType<nntrainer::Optimizer> factory, const std::string &key,
  const int int_key);

/**
 * @copydoc const int AppContext::registerFactory
 */
template const int AppContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

/**
 * @copydoc const int AppContext::registerFactory
 */
template const int
AppContext::registerFactory<ml::train::LearningRateScheduler>(
  const FactoryType<ml::train::LearningRateScheduler> factory,
  const std::string &key, const int int_key);

} // namespace nntrainer

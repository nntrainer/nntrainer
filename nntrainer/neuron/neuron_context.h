// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    neuron_context.h
 * @date    30 Jul 2026
 * @see     https://github.com/nnstreamer/nntrainer
 * @brief   Context that manages the global configuration of the current
 *          MediaTek NeuroPilot (Neuron Runtime) environment. Structurally
 *          mirrors nntrainer/qnn/qnn_context.h, adapted for a backend that
 *          has no provider/version negotiation and no name-keyed graph
 *          lookup (see neuron_context_var.h for why).
 */
#ifndef __NEURON_CONTEXT_H__
#define __NEURON_CONTEXT_H__

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <context.h>
#include <layer.h>
#include <layer_devel.h>

#include <neuron_context_var.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

#include "singleton.h"

namespace nntrainer {

extern std::mutex neuron_factory_mutex;

/**
 * @class NeuronContext contains user-dependent configuration for MediaTek
 * NeuroPilot (Neuron Runtime) support
 * @brief Neuron support for app context
 */
class NeuronContext : public Context, public Singleton<NeuronContext> {

public:
  /**
   * @brief   Default constructor
   */
  NeuronContext() : Context(std::make_shared<NeuronBackendVar>()) {}

  ~NeuronContext() {
    auto neuron_data = getNeuronData();
    // Release every loaded DLA runtime before the backend goes away.
    if (neuron_data) {
      neuron_data->freeAllRuntimes();
    }
  }

  int init() override;

  /**
   * @copydoc QNNContext::registerFactory
   */
  template <typename T>
  const int registerFactory(const PtrFactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1) {
    FactoryType<T> f = factory;
    return registerFactory(f, key, int_key);
  }

  /**
   * @copydoc QNNContext::registerFactory
   */
  template <typename T>
  const int registerFactory(const FactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1);

  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &properties) override {
    return createObject<nntrainer::Layer>(type, properties);
  }

  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(int_key, properties);
  }

  /**
   * @brief Create an Object from the integer key
   */
  template <typename T>
  PtrType<T> createObject(const int int_key,
                          const PropsType &props = {}) const {
    static_assert(isSupported<T>::value,
                  "given type is not supported for current neuron context");
    auto &index = std::get<IndexType<T>>(factory_map);
    auto &int_map = std::get<IntIndexType>(index);

    const auto &entry = int_map.find(int_key);

    if (entry == int_map.end()) {
      std::stringstream ss;
      ss << "Int Key is not found for the object. Key: " << int_key;
      throw exception::not_supported(ss.str().c_str());
    }

    return createObject<T>(entry->second, props);
  }

  /**
   * @brief Create an Object from the string key
   */
  template <typename T>
  PtrType<T> createObject(const std::string &key,
                          const PropsType &props = {}) const {
    auto &index = std::get<IndexType<T>>(factory_map);
    auto &str_map = std::get<StrIndexType<T>>(index);

    std::string lower_key;
    lower_key.resize(key.size());
    std::transform(key.begin(), key.end(), lower_key.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    const auto &entry = str_map.find(lower_key);

    if (entry == str_map.end()) {
      std::stringstream ss;
      ss << "Key is not found for the object. Key: " << lower_key;
      throw exception::not_supported(ss.str().c_str());
    }

    return entry->second(props);
  }

  std::string getName() override { return "neuron"; }

  void setMemAllocator(std::shared_ptr<NeuronDmaAllocator> mem) {
    getContextData()->setMemAllocator(mem);
  }

  std::shared_ptr<NeuronVar> getNeuronData() {
    std::shared_ptr<NeuronBackendVar> d =
      std::static_pointer_cast<NeuronBackendVar>(this->getContextData());
    return d->getVar();
  }

  /** @copydoc Context::load. Loads/creates a NeuronRuntime for the given
   * .dla path; idempotent, see NeuronVar::makeRuntime. */
  int load(const std::string &file_path) override {
    NeuronStatusCode ret = getNeuronData()->makeRuntime(file_path);
    return (int)ret;
  }

private:
  void initialize() noexcept override;

  FactoryMap<nntrainer::Layer> factory_map;

  template <typename Args, typename T> struct isSupportedHelper;

  template <typename T, typename... Args>
  struct isSupportedHelper<T, NeuronContext::FactoryMap<Args...>> {
    static constexpr bool value =
      (std::is_same_v<std::decay_t<T>, std::decay_t<Args>> || ...);
  };

  template <typename T>
  struct isSupported : isSupportedHelper<T, decltype(factory_map)> {};
};

/**
 * @copydoc const int NeuronContext::registerFactory
 */
extern template const int NeuronContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

} // namespace nntrainer

#endif /* __NEURON_CONTEXT_H__ */

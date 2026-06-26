// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   app_context.h
 * @date   10 November 2020
 * @brief  This file contains app context related functions and classes that
 * manages the global configuration of the current environment
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __APP_CONTEXT_H__
#define __APP_CONTEXT_H__

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

#include <layer.h>
#include <layer_devel.h>
#include <optimizer.h>
#include <optimizer_devel.h>

#include <context.h>
#include <mem_allocator.h>
#include <nntrainer_error.h>

#include "singleton.h"

namespace nntrainer {

extern std::mutex factory_mutex;
namespace {} // namespace

/**
 * @class AppContext contains user-dependent configuration
 * @brief App
 */
class AppContext : public Context, public Singleton<AppContext> {
public:
  /**
   * @brief   Default constructor
   */
  AppContext() : Context(std::make_shared<ContextData>()) {}

  /**
   * @brief   Default destructor
   */
  ~AppContext() override = default;

  /**
   * @brief Set Working Directory for a relative path. working directory is set
   * canonically
   * @param[in] base base directory
   * @throw std::invalid_argument if path is not valid for current system
   */
  void setWorkingDirectory(const std::string &base);

  /**
   * @brief unset working directory
   *
   */
  void unsetWorkingDirectory() { working_path_base = ""; }

  /**
   * @brief query if the appcontext has working directory set
   *
   * @retval true working path base is set
   * @retval false working path base is not set
   */
  bool hasWorkingDirectory() { return !working_path_base.empty(); }

  /**
   * @brief register a layer factory from a shared library
   * plugin must have **extern "C" LayerPluggable *ml_train_layer_pluggable**
   * defined else error
   *
   * @param library_path a file name of the library
   * @param base_path    base path to make a full path (optional)
   * @return int integer key to create the layer
   * @throws std::invalid_parameter if library_path is invalid or library is
   * invalid
   */
  int registerLayer(const std::string &library_path,
                    const std::string &base_path = "");

  /**
   * @brief register a optimizer factory from a shared library
   * plugin must have **extern "C" OptimizerPluggable
   * *ml_train_optimizer_pluggable** defined else error
   *
   * @param library_path a file name of the library
   * @param base_path    base path to make a full path (optional)
   * @return int integer key to create the optimizer
   * @throws std::invalid_parameter if library_path is invalid or library is
   * invalid
   */
  int registerOptimizer(const std::string &library_path,
                        const std::string &base_path = "");

  /**
   * @brief register pluggables from a directory.
   * @note if you have a clashing type with already registered pluggable, it
   * will be a no-op (existing int_key is returned)
   *
   * @param base_path a directory path to search pluggables's
   * @return std::vector<int> list of integer key to create a pluggable
   */
  std::vector<int> registerPluggableFromDirectory(const std::string &base_path);

  /**
   * @brief Get Working Path from a relative or representation of a path
   * starting from @a working_path_base.
   * @param[in] path to make full path
   * @return If absolute path is given, returns @a path
   * If relative path is given and working_path_base is not set, return
   * relative path.
   * If relative path is given and working_path_base has set, return absolute
   * path from current working directory
   */
  const std::string getWorkingPath(const std::string &path = "");

  /**
   * @brief Get memory fsu file path from configuration file
   * @return memory fsu path.
   * If memory fsu path is not presented in configuration file, it returns
   * empty string
   */
  const std::vector<std::string> getProperties(void);

  /**
   * @brief Factory register function, use this function to register custom
   * object
   *
   * @tparam T object to create. Currently Optimizer, Layer is supported
   * @param factory factory function that creates std::unique_ptr<T>
   * @param key key to access the factory, if key is empty, try to find key by
   * calling factory({})->getType();
   * @param int_key key to access the factory by integer, if it is -1(default),
   * the function automatically unsigned the key and return
   * @return const int unique integer value to access the current factory
   * @note If key and/or int_key is already taken, the existing int_key is
   * returned (no-op re-registration). This allows multi-model handles to
   * safely re-register shared custom layers.
   */
  template <typename T>
  const int registerFactory(const PtrFactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1) {
    FactoryType<T> f = factory;
    return registerFactory(f, key, int_key);
  }

  /**
   * @brief Factory register function, use this function to register custom
   * object
   *
   * @tparam T object to create. Currently Optimizer, Layer is supported
   * @param factory factory function that creates std::unique_ptr<T>
   * @param key key to access the factory, if key is empty, try to find key by
   * calling factory({})->getType();
   * @param int_key key to access the factory by integer, if it is -1(default),
   * the function automatically unsigned the key and return
   * @return const int unique integer value to access the current factory
   * @note If key and/or int_key is already taken, the existing int_key is
   * returned (no-op re-registration). This allows multi-model handles to
   * safely re-register shared custom layers.
   */
  template <typename T>
  const int registerFactory(const FactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1);

  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(type, properties);
  }

  std::unique_ptr<nntrainer::Optimizer> createOptimizerObject(
    const std::string &type,
    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Optimizer>(type, properties);
  }

  std::unique_ptr<ml::train::LearningRateScheduler>
  createLearningRateSchedulerObject(
    const std::string &type,
    const std::vector<std::string> &properties = {}) override {
    return createObject<ml::train::LearningRateScheduler>(type, properties);
  }

  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(int_key, properties);
  }

  std::unique_ptr<nntrainer::Optimizer> createOptimizerObject(
    const int int_key,
    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Optimizer>(int_key, properties);
  }

  std::unique_ptr<ml::train::LearningRateScheduler>
  createLearningRateSchedulerObject(
    const int int_key,
    const std::vector<std::string> &properties = {}) override {
    return createObject<ml::train::LearningRateScheduler>(int_key, properties);
  }

  /**
   * @brief Create an Object from the integer key
   *
   * @tparam T Type of Object, currently, Only optimizer is supported
   * @param int_key integer key
   * @param props property
   * @return PtrType<T> unique pointer to the object
   */
  template <typename T>
  PtrType<T> createObject(const int int_key,
                          const PropsType &props = {}) const {
    static_assert(isSupported<T>::value,
                  "given type is not supported for current app context");
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
   *
   * @tparam T Type of object, currently, only optimizer is supported
   * @param key integer key
   * @param props property
   * @return PtrType<T> unique pointer to the object
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

  /**
   * @brief special factory that throws for unknown
   *
   * @tparam T object to create
   * @param props props to pass, not used
   * @throw always throw runtime_error
   */
  template <typename T>
  static PtrType<T> unknownFactory(const PropsType &props) {
    throw std::invalid_argument("cannot create unknown object");
  }

  std::string getName() override { return "cpu"; }

  void setMemAllocator(std::shared_ptr<MemAllocator> mem) {
    getContextData()->setMemAllocator(mem);
  }

private:
  /**
   * @brief   Overriden initialization function
   */
  void initialize() noexcept override;

  void add_default_object();

  void add_extension_object();

  FactoryMap<nntrainer::Optimizer, nntrainer::Layer,
             ml::train::LearningRateScheduler>
    factory_map;
  std::string working_path_base;

  template <typename Args, typename T> struct isSupportedHelper;

  /**
   * @brief supportHelper to check if given type is supported within appcontext
   */
  template <typename T, typename... Args>
  struct isSupportedHelper<T, AppContext::FactoryMap<Args...>> {
    static constexpr bool value =
      (std::is_same_v<std::decay_t<T>, std::decay_t<Args>> || ...);
  };

  /**
   * @brief supportHelper to check if given type is supported within appcontext
   */
  template <typename T>
  struct isSupported : isSupportedHelper<T, decltype(factory_map)> {};
};

/**
 * @copydoc const int AppContext::registerFactory
 */
extern template const int AppContext::registerFactory<nntrainer::Optimizer>(
  const FactoryType<nntrainer::Optimizer> factory, const std::string &key,
  const int int_key);

/**
 * @copydoc const int AppContext::registerFactory
 */
extern template const int AppContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

/**
 * @copydoc const int AppContext::registerFactory
 */
extern template const int
AppContext::registerFactory<ml::train::LearningRateScheduler>(
  const FactoryType<ml::train::LearningRateScheduler> factory,
  const std::string &key, const int int_key);

namespace plugin {}

} // namespace nntrainer

#endif /* __APP_CONTEXT_H__ */

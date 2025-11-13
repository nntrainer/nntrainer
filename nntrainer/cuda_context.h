// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    cuda_context.h
 * @date    13 Nov 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Samsung Electronics Co., Ltd.
 * @bug     No known bugs except for NYI items
 * @brief   This file contains app context related functions and classes that
 * manages the global configuration of the current CUDA environment. It also
 * creates the CUDA stream and context.
 */

#ifndef __CUDA_CONTEXT_H__
#define __CUDA_CONTEXT_H__

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <context.h>
#include <layer.h>
#include <layer_devel.h>
#include <mem_allocator.h>

#include "singleton.h"

namespace nntrainer {

extern std::mutex cuda_factory_mutex;

/**
 * @class CudaContext contains user-dependent configuration for CUDA support
 * @brief CUDA support for app context
 */
class CudaContext : public Context, public Singleton<CudaContext> {
public:
  /**
   * @brief   Default constructor
   */
  CudaContext() : Context(std::make_shared<ContextData>()) {}

  /**
   * @brief destructor to release cuda context
   */
  ~CudaContext() override {
    if (cuda_initialized) {
      // Release CUDA resources
      if (stream_) {
        cudaStreamDestroy(stream_);
      }
    }
  };

  /**
   * @brief Factory register function, use this function to register custom
   * object
   *
   * @tparam T object to create. Currently Layer is supported
   * @param factory factory function that creates std::unique_ptr<T>
   * @param key key to access the factory, if key is empty, try to find key by
   * calling factory({})->getType();
   * @param int_key key to access the factory by integer, if it is -1(default),
   * the function automatically unsigned the key and return
   * @return const int unique integer value to access the current factory
   * @throw invalid argument when key and/or int_key is already taken
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
   * @tparam T object to create. Currently Layer is supported
   * @param factory factory function that creates std::unique_ptr<T>
   * @param key key to access the factory, if key is empty, try to find key by
   * calling factory({})->getType();
   * @param int_key key to access the factory by integer, if it is -1(default),
   * the function automatically unsigned the key and return
   * @return const int unique integer value to access the current factory
   * @throw invalid argument when key and/or int_key is already taken
   */
  template <typename T>
  const int registerFactory(const FactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1);

  /**
   * @brief Create an Object from the integer key
   *
   * @tparam T Type of Object, currently, Only Layer is supported
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
      ml_loge("Int Key is not found for the object. Key: %d", int_key);
      throw exception::not_supported(std::to_string(int_key));
    }

    // entry is an object of int_map which is an unordered_map<int, std::string>
    return createObject<T>(entry->second, props);
  }

  /**
   * @brief Create an Object from the string key
   *
   * @tparam T Type of object, currently, only Layer is supported
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
      ml_loge("Key is not found for the object. Key: %s", lower_key.c_str());
      throw exception::not_supported(lower_key);
    }

    // entry -> object of str_map -> unordered_map<std::string, FactoryType<T>>
    return entry->second(props);
  }

  /**
   * @brief Create a Layer object from the string key
   *
   * @param type string key
   * @param properties property
   * @return std::unique_ptr<nntrainer::Layer> unique pointer to the object
   */
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(type, properties);
  }

  /**
   * @brief Create a Layer object from the integer key
   *
   * @param type integer key
   * @param properties property
   * @return std::unique_ptr<nntrainer::Layer> unique pointer to the object
   */
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(int_key, properties);
  }

  /**
   * @brief Get the name of the context
   */
  std::string getName() override { return "cuda"; }

  /**
   * @brief Set the Mem Allocator object
   *
   * @param mem Memory allocator object
   */
  void setMemAllocator(std::shared_ptr<MemAllocator> mem) {
    getContextData()->setMemAllocator(mem);
  }

  /**
   * @brief Get CUDA stream
   * @return cudaStream_t
   */
  cudaStream_t getStream() const { return stream_; }

private:
  /**
   * @brief   Overriden init function
   */
  void initialize() noexcept override;

  void add_default_object();

  // flag to check cuda initialization
  bool cuda_initialized = false;

  // CUDA stream for asynchronous operations
  cudaStream_t stream_ = nullptr;

  FactoryMap<nntrainer::Layer> factory_map;

  template <typename Args, typename T> struct isSupportedHelper;

  /**
   * @brief supportHelper to check if given type is supported within cuda
   * context
   */
  template <typename T, typename... Args>
  struct isSupportedHelper<T, CudaContext::FactoryMap<Args...>> {
    static constexpr bool value =
      (std::is_same_v<std::decay_t<T>, std::decay_t<Args>> || ...);
  };

  /**
   * @brief supportHelper to check if given type is supported within cuda
   * context
   */
  template <typename T>
  struct isSupported : isSupportedHelper<T, decltype(factory_map)> {};

  /**
   * @brief Initialize cuda context and stream
   * @return true if CUDA context and stream creation is successful,
   * false otherwise
   */
  bool cudaInit();
};

/**
 * @copydoc const int CudaContext::registerFactory
 */
extern template const int CudaContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

} // namespace nntrainer

#endif /* __CUDA_CONTEXT_H__ */

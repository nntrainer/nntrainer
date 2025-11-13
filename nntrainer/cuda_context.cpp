// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    cuda_context.cpp
 * @date    13 Nov 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Samsung Electronics Co., Ltd.
 * @bug     No known bugs except for NYI items
 * @brief   This file contains app context related functions and classes that
 * manages the global configuration of the current CUDA environment. It also
 * creates the CUDA stream and context.
 */

#include "cuda_context.h"

#include <addition_layer.h>
#include <fc_layer.h>
#include <nntrainer_log.h>
#include <reshape_layer.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace nntrainer {
std::mutex cuda_factory_mutex;

void CudaContext::initialize() noexcept {
  try {
    if (!cudaInit()) {
      ml_loge("Error: CudaContext::initialize() failed");
      return;
    }

    add_default_object();
    setMemAllocator(std::make_shared<MemAllocator>());

  } catch (std::exception &e) {
    ml_loge("cuda_context: registering layers failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("cuda_context: registering layer failed due to unknown reason");
  }
};

void CudaContext::add_default_object() {
  // Register default layers that support CUDA
  registerFactory(nntrainer::createLayer<FullyConnectedLayer>,
                  FullyConnectedLayer::type, ml::train::LayerType::LAYER_FC);

  registerFactory(nntrainer::createLayer<AdditionLayer>, AdditionLayer::type,
                  ml::train::LayerType::LAYER_ADDITION);

  registerFactory(nntrainer::createLayer<ReshapeLayer>, ReshapeLayer::type,
                  ml::train::LayerType::LAYER_RESHAPE);
}

template <typename T>
const int CudaContext::registerFactory(const FactoryType<T> factory,
                                       const std::string &key,
                                       const int int_key) {
  static_assert(
    isSupported<T>::value,
    "cuda_context: given type is not supported for current context");

  auto &index = std::get<IndexType<T>>(factory_map);
  auto &str_map = std::get<StrIndexType<T>>(index);
  auto &int_map = std::get<IntIndexType>(index);

  std::string assigned_key = key == "" ? factory({})->getType() : key;

  std::transform(assigned_key.begin(), assigned_key.end(), assigned_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::lock_guard<std::mutex> lock(cuda_factory_mutex);
  if (str_map.find(assigned_key) != str_map.end()) {
    ml_loge("cuda_context: cannot register factory with already taken key: %s",
            key.c_str());
    throw std::invalid_argument(key);
  }

  if (int_key != -1 && int_map.find(int_key) != int_map.end()) {
    ml_loge(
      "cuda_context: cannot register factory with already taken int key: %d",
      int_key);
    throw std::invalid_argument(std::to_string(int_key));
  }

  int assigned_int_key = int_key == -1 ? str_map.size() + 1 : int_key;

  str_map[assigned_key] = factory;
  int_map[assigned_int_key] = assigned_key;

  ml_logd("cuda_context: factory has registered with key: %s, int_key: %d",
          assigned_key.c_str(), assigned_int_key);

  return assigned_int_key;
}

bool CudaContext::cudaInit() {
  // if already initialized
  if (cuda_initialized)
    return true;

  // Initialize CUDA context
  cudaError_t err = cudaSetDevice(0);
  if (err != cudaSuccess) {
    ml_loge("Failed to set CUDA device: %s", cudaGetErrorString(err));
    return false;
  }

  // Create CUDA stream for asynchronous operations
  err = cudaStreamCreate(&stream_);
  if (err != cudaSuccess) {
    ml_loge("Failed to create CUDA stream: %s", cudaGetErrorString(err));
    return false;
  }

  cuda_initialized = true;
  return cuda_initialized;
}

/**
 * @copydoc const int CudaContext::registerFactory
 */
template const int CudaContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

} // namespace nntrainer

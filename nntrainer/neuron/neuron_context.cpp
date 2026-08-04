// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    neuron_context.cpp
 * @date    30 Jul 2026
 * @see     https://github.com/nnstreamer/nntrainer
 * @brief   Context that manages the global configuration of the current
 *          MediaTek NeuroPilot (Neuron Runtime) environment.
 */
#include "neuron_context.h"

#include <NeuronGraph.h>
#include <cstdlib>

namespace nntrainer {

std::mutex neuron_factory_mutex;

namespace {

/**
 * @brief Which shared library to dlopen for the Neuron Runtime API.
 *
 * QUICK_DOT_AI_NEURON_LIB overrides the default "libneuron_runtime.so" so
 * host-side testing can point directly at the SDK's dummy/lib copy
 * (kEnvOptNullDevice + the dummy runtime lets the whole plugin/registration
 * path be exercised without a device attached).
 */
std::string resolve_neuron_library_name() {
  const char *override_lib = std::getenv("QUICK_DOT_AI_NEURON_LIB");
  if (override_lib != nullptr && override_lib[0] != '\0') {
    return std::string(override_lib);
  }
  return "libneuron_runtime.so";
}

/**
 * @brief Whether to force EnvOptions.deviceKind = kEnvOptNullDevice.
 *
 * Set QUICK_DOT_AI_NEURON_NULL_DEVICE=1 to validate the plugin/registration
 * path without real NPU hardware (paired with QUICK_DOT_AI_NEURON_LIB
 * pointing at the SDK's dummy runtime).
 */
bool resolve_use_null_device() {
  const char *override_null = std::getenv("QUICK_DOT_AI_NEURON_NULL_DEVICE");
  return override_null != nullptr && override_null[0] == '1';
}

} // namespace

void NeuronContext::initialize() noexcept {
  try {
    int status = init();
    if (status != 0) {
      ml_loge("NeuronContext::initialize: init() failed with status %d",
              status);
      return;
    }
    ml_logi("neuron init done");

    auto neuron_data = getNeuronData();
    auto allocator = std::make_shared<NeuronDmaAllocator>();
    setMemAllocator(allocator);
    neuron_data->DmaAlloc = allocator;

    registerFactory(nntrainer::createLayer<NeuronGraph>, NeuronGraph::type, -1);
    ml_logi("neuron registerFactory done");
  } catch (std::exception &e) {
    ml_loge("NeuronContext::initialize: registering neuron layers failed, "
            "reason: %s",
            e.what());
  } catch (...) {
    ml_loge("NeuronContext::initialize: registering neuron layer failed due to "
            "unknown reason");
  }
}

int NeuronContext::init() {
  auto neuron_data = getNeuronData();

  const std::string library_name = resolve_neuron_library_name();
  if (!neuron_data->api.load(library_name)) {
    ml_loge("NeuronContext::init: failed to load Neuron Runtime API from %s",
            library_name.c_str());
    return -1;
  }

  EnvOptions &opts = neuron_data->envOptions;
  opts.deviceKind =
    resolve_use_null_device() ? kEnvOptNullDevice : kEnvOptHardware;
  opts.MDLACoreOption = MDLACoreMode::Auto;
  opts.CPUThreadNum = 0;
  // Leave conversion enabled by default: suppressing it requires every
  // caller to already produce hardware-padded/aligned buffers. Revisit once
  // Milestone 1's padded-vs-exact-size measurement (see neuron_context_var.h
  // makeRuntime()) is in.
  opts.suppressInputConversion = false;
  opts.suppressOutputConversion = false;

  return 0;
}

template <typename T>
const int NeuronContext::registerFactory(const FactoryType<T> factory,
                                         const std::string &key,
                                         const int int_key) {
  static_assert(
    isSupported<T>::value,
    "neuron_context: given type is not supported for current context");

  auto &index = std::get<IndexType<T>>(factory_map);
  auto &str_map = std::get<StrIndexType<T>>(index);
  auto &int_map = std::get<IntIndexType>(index);

  std::string assigned_key = key == "" ? factory({})->getType() : key;

  std::transform(assigned_key.begin(), assigned_key.end(), assigned_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::lock_guard<std::mutex> lock(neuron_factory_mutex);
  if (str_map.find(assigned_key) != str_map.end()) {
    for (const auto &[ik, sk] : int_map) {
      if (sk == assigned_key)
        return ik;
    }
    return -1;
  }
  if (int_key != -1 && int_map.find(int_key) != int_map.end()) {
    return int_key;
  }

  int assigned_int_key = int_key == -1 ? str_map.size() + 1 : int_key;

  str_map[assigned_key] = factory;
  int_map[assigned_int_key] = assigned_key;

  ml_logd("neuron_context: factory has registered with key: %s, int_key: %d",
          assigned_key.c_str(), assigned_int_key);

  return assigned_int_key;
}

/**
 * @copydoc const int NeuronContext::registerFactory
 */
template const int NeuronContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

#ifdef PLUGGABLE
nntrainer::Context *create_neuron_context() {
  nntrainer::NeuronContext *neuron_context = new nntrainer::NeuronContext();
  neuron_context->initializeOnce();
  return neuron_context;
}

void destroy_neuron_context(nntrainer::Context *ct) { delete ct; }

extern "C" {
nntrainer::ContextPluggable ml_train_context_pluggable{create_neuron_context,
                                                       destroy_neuron_context};
}
#endif

} // namespace nntrainer

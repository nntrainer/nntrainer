// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    neuron_api.cpp
 * @date    30 Jul 2026
 * @see     https://github.com/nnstreamer/nntrainer
 * @brief   Hand-resolved dlsym table over libneuron_runtime.so.
 */
#include "neuron_api.h"

#include <dynamic_library_loader.h>
#include <nntrainer_log.h>

namespace nntrainer {

namespace {

template <typename T>
bool resolve(void *handle, const char *symbol, T &out_fn) {
  out_fn =
    reinterpret_cast<T>(DynamicLibraryLoader::loadSymbol(handle, symbol));
  if (out_fn == nullptr) {
    ml_loge("NeuronApi: failed to resolve symbol %s", symbol);
    return false;
  }
  return true;
}

} // namespace

bool NeuronApi::load(const std::string &library_name) {
  if (handle_ != nullptr) {
    return true;
  }

  handle_ = DynamicLibraryLoader::loadLibrary(library_name.c_str(),
                                              RTLD_NOW | RTLD_LOCAL);
  if (handle_ == nullptr) {
    ml_loge("NeuronApi: failed to dlopen %s: %s", library_name.c_str(),
            DynamicLibraryLoader::getLastError());
    return false;
  }

  bool ok = true;
  ok &= resolve(handle_, "NeuronRuntime_create", create);
  ok &=
    resolve(handle_, "NeuronRuntime_create_with_options", create_with_options);
  ok &=
    resolve(handle_, "NeuronRuntime_loadNetworkFromFile", loadNetworkFromFile);
  ok &= resolve(handle_, "NeuronRuntime_loadNetworkFromBuffer",
                loadNetworkFromBuffer);
  ok &= resolve(handle_, "NeuronRuntime_setInput", setInput);
  ok &= resolve(handle_, "NeuronRuntime_setOutput", setOutput);
  ok &= resolve(handle_, "NeuronRuntime_setQoSOption", setQoSOption);
  ok &= resolve(handle_, "NeuronRuntime_getInputNumber", getInputNumber);
  ok &= resolve(handle_, "NeuronRuntime_getOutputNumber", getOutputNumber);
  ok &= resolve(handle_, "NeuronRuntime_getInputSize", getInputSize);
  ok &= resolve(handle_, "NeuronRuntime_getOutputSize", getOutputSize);
  ok &=
    resolve(handle_, "NeuronRuntime_getInputPaddedSize", getInputPaddedSize);
  ok &=
    resolve(handle_, "NeuronRuntime_getOutputPaddedSize", getOutputPaddedSize);
  ok &= resolve(handle_, "NeuronRuntime_getInputPaddedDimensions",
                getInputPaddedDimensions);
  ok &= resolve(handle_, "NeuronRuntime_getOutputPaddedDimensions",
                getOutputPaddedDimensions);
  ok &= resolve(handle_, "NeuronRuntime_inference", inference);
  ok &= resolve(handle_, "NeuronRuntime_release", release);
  ok &= resolve(handle_, "NeuronRuntime_getVersion", getVersion);
  ok &= resolve(handle_, "NeuronRuntime_getMetadataInfo", getMetadataInfo);
  ok &= resolve(handle_, "NeuronRuntime_getMetadata", getMetadata);

  if (!ok) {
    ml_loge("NeuronApi: one or more required symbols could not be resolved "
            "from %s; refusing to report the API as usable",
            library_name.c_str());
    DynamicLibraryLoader::freeLibrary(handle_);
    handle_ = nullptr;
    return false;
  }

  NeuronVersion version{};
  if (getVersion(&version) == NEURONRUNTIME_NO_ERROR) {
    ml_logi("NeuronApi: loaded %s, Neuron Runtime version %u.%u.%u",
            library_name.c_str(), version.major, version.minor, version.patch);
  }

  return true;
}

NeuronApi::~NeuronApi() {
  if (handle_ != nullptr) {
    DynamicLibraryLoader::freeLibrary(handle_);
    handle_ = nullptr;
  }
}

} // namespace nntrainer

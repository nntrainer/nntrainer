// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    neuron_api.h
 * @date    30 Jul 2026
 * @see     https://github.com/nnstreamer/nntrainer
 * @brief   Hand-resolved dlsym table over libneuron_runtime.so (MediaTek
 *          NeuroPilot Neuron Runtime). Unlike Qualcomm QNN, Neuron exports
 *          plain, stable C symbols with no provider/version-negotiation
 *          dance, so this table is the entire vendor-glue layer: there is
 *          no equivalent of QNN's DynamicLoadUtil/SampleApp machinery.
 */
#ifndef __NEURON_API_H__
#define __NEURON_API_H__

#include "neuron/api/RuntimeAPI.h"
#include "neuron/api/Types.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace nntrainer {

/**
 * @brief Function-pointer table resolved from libneuron_runtime.so.
 *
 * Only the subset of NeuronRuntime_* needed for Milestone 1 (single-graph
 * load + index-based I/O + padded-size validation) is resolved. Extend as
 * later phases need more (setInputShape for dynamic shapes, clone for
 * prefill/decode weight sharing helpers, getInputRank/getInputPitch for
 * richer validation, etc).
 */
struct NeuronApi {
  int (*create)(const EnvOptions *, void **) = nullptr;
  // NeuronRuntime_create's own header names its parameter
  // "optionsToDeprecate" -- on-device testing (mt6991, Neuron Runtime 9.3.x)
  // confirmed the reference neuronrt CLI creates its runtime through this
  // newer, string-options entry point instead, and a .dla that neuronrt
  // loads and runs successfully under `-m hw` failed with the generic
  // "Cannot load network" via the plain create() path above. makeRuntime()
  // therefore uses this one exclusively; `create` above is kept resolved
  // only so its absence doesn't itself break load().
  int (*create_with_options)(const char *, const EnvOptions *,
                             void **) = nullptr;
  int (*loadNetworkFromFile)(void *, const char *) = nullptr;
  int (*loadNetworkFromBuffer)(void *, const void *, size_t) = nullptr;
  int (*setInput)(void *, uint64_t, const void *, size_t,
                  BufferAttribute) = nullptr;
  int (*setOutput)(void *, uint64_t, void *, size_t, BufferAttribute) = nullptr;
  int (*setQoSOption)(void *, const QoSOptions *) = nullptr;
  int (*getInputNumber)(void *, size_t *) = nullptr;
  int (*getOutputNumber)(void *, size_t *) = nullptr;
  int (*getInputSize)(void *, uint64_t, size_t *) = nullptr;
  int (*getOutputSize)(void *, uint64_t, size_t *) = nullptr;
  int (*getInputPaddedSize)(void *, uint64_t, size_t *) = nullptr;
  int (*getOutputPaddedSize)(void *, uint64_t, size_t *) = nullptr;
  int (*getInputPaddedDimensions)(void *, uint64_t,
                                  RuntimeAPIDimensions *) = nullptr;
  int (*getOutputPaddedDimensions)(void *, uint64_t,
                                   RuntimeAPIDimensions *) = nullptr;
  int (*inference)(void *) = nullptr;
  void (*release)(void *) = nullptr;
  int (*getVersion)(NeuronVersion *) = nullptr;
  int (*getMetadataInfo)(void *, const char *, size_t *) = nullptr;
  int (*getMetadata)(void *, const char *, char *, size_t) = nullptr;

  /**
   * @brief dlopen libneuron_runtime.so and resolve every symbol above.
   * @param library_name shared library to dlopen; overridable for tests
   *        that want to point at the SDK's dummy/lib copy.
   * @return true if the library and every required symbol resolved.
   *         Logs which symbol failed to resolve on false.
   */
  bool load(const std::string &library_name = "libneuron_runtime.so");

  /** @brief whether load() has succeeded (handle_ non-null) */
  bool valid() const { return handle_ != nullptr; }

  NeuronApi() = default;
  ~NeuronApi();
  NeuronApi(const NeuronApi &) = delete;
  NeuronApi &operator=(const NeuronApi &) = delete;

private:
  void *handle_ = nullptr;
};

} // namespace nntrainer

#endif /* __NEURON_API_H__ */

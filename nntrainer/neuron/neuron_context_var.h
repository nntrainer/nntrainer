// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    neuron_context_var.h
 * @date    30 Jul 2026
 * @see     https://github.com/nnstreamer/nntrainer
 * @brief   Backend-global and per-DLA state for the MediaTek Neuron context.
 *
 * @details Mirrors nntrainer/qnn/jni/qnn_context_var.h's QNNVar /
 * QNNBackendVar split, but is considerably simpler: a MediaTek `.dla` file
 * holds exactly one compiled network (no QNN-style multiple named graphs
 * per binary), and Neuron addresses I/O tensors by a plain `uint64_t` index
 * rather than by name, so there is no graph_map / case-insensitive-name-
 * fallback machinery to port.
 */
#ifndef __NEURON_CONTEXT_VAR_H__
#define __NEURON_CONTEXT_VAR_H__

#include "neuron_api.h"
#include "neuron_dma_allocator.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <context_data.h>

#include <nntrainer_log.h>

namespace nntrainer {

enum class NeuronStatusCode {
  SUCCESS,
  FAILURE,
};

/**
 * @brief Per-DLA runtime state: the opaque runtime handle plus the I/O
 * geometry queried from it right after load (there is no offline metadata
 * query analogous to QNN's systemContextGetBinaryInfo — Neuron only
 * exposes this after NeuronRuntime_loadNetworkFromFile/Buffer).
 */
struct NeuronRuntimeEntry {
  void *runtime = nullptr;
  size_t numInputs = 0;
  size_t numOutputs = 0;

  /** @brief exact (unpadded) per-tensor buffer sizes, indexed like the DLA's
   * own I/O handles */
  std::vector<size_t> inputExactSizes;
  std::vector<size_t> outputExactSizes;

  /** @brief hardware-aligned buffer sizes; NeuronRuntime_setInput/setOutput
   * must be called with a buffer at least this large */
  std::vector<size_t> inputPaddedSizes;
  std::vector<size_t> outputPaddedSizes;

  bool initialized = false;
};

/** @brief Backend-global state: the resolved API table, env options, and
 * the map of currently-loaded DLA runtimes keyed by file path. */
struct NeuronVar {
  NeuronApi api;
  EnvOptions envOptions{};
  std::shared_ptr<NeuronDmaAllocator> DmaAlloc;

  std::map<std::string, NeuronRuntimeEntry> rt_map;

  std::optional<std::reference_wrapper<NeuronRuntimeEntry>>
  findRuntime(const std::string &dla_path) {
    auto it = rt_map.find(dla_path);
    if (it != rt_map.end()) {
      return it->second;
    }
    return std::nullopt;
  }

  NeuronStatusCode freeRuntime(const std::string &dla_path) {
    auto it = rt_map.find(dla_path);
    if (it == rt_map.end()) {
      ml_logw("NeuronVar: runtime not found for: %s", dla_path.c_str());
      return NeuronStatusCode::FAILURE;
    }
    if (it->second.runtime != nullptr && api.valid()) {
      api.release(it->second.runtime);
    }
    rt_map.erase(it);
    ml_logi("NeuronVar: freed runtime for: %s", dla_path.c_str());
    return NeuronStatusCode::SUCCESS;
  }

  NeuronStatusCode freeAllRuntimes() {
    // Copy keys first: freeRuntime() erases from rt_map, so iterating the
    // map directly while erasing would invalidate the iterator.
    std::vector<std::string> keys;
    keys.reserve(rt_map.size());
    for (auto &[k, _] : rt_map) {
      keys.push_back(k);
    }
    for (auto &k : keys) {
      freeRuntime(k);
    }
    return NeuronStatusCode::SUCCESS;
  }

  /**
   * @brief Create a NeuronRuntime, load the DLA at dla_path, and cache its
   * I/O geometry. Idempotent: if a runtime already exists for this path,
   * returns SUCCESS without creating a second one (mirrors QNNVar::
   * makeContext, which the app layer relies on for lazy-load-on-first-
   * forwarding semantics).
   */
  NeuronStatusCode makeRuntime(const std::string &dla_path) {
    if (findRuntime(dla_path)) {
      ml_logw("NeuronVar: runtime already exists for: %s", dla_path.c_str());
      return NeuronStatusCode::SUCCESS;
    }

    if (!api.valid()) {
      ml_loge("NeuronVar: NeuronApi is not loaded; cannot create runtime");
      return NeuronStatusCode::FAILURE;
    }

    NeuronRuntimeEntry entry;
    void *runtime = nullptr;
    // NeuronRuntime_create's EnvOptions* parameter is documented as
    // "optionsToDeprecate", and on-device testing on mt6991 (Neuron Runtime
    // 9.3.x) confirmed it: a real hardware-mode .dla that MediaTek's own
    // neuronrt CLI loads and runs successfully (via `-m hw`, which neuronrt
    // implements through this same _with_options entry point) failed with a
    // generic "Cannot load network" when the runtime was created through
    // the plain NeuronRuntime_create() above instead. Pass an empty options
    // string -- envOptions.deviceKind carries the real device selection,
    // identical to what neuronrt sets for `-m hw`.
    if (api.create_with_options("", &envOptions, &runtime) !=
          NEURONRUNTIME_NO_ERROR ||
        runtime == nullptr) {
      ml_loge("NeuronVar: NeuronRuntime_create_with_options failed for: %s",
              dla_path.c_str());
      return NeuronStatusCode::FAILURE;
    }
    entry.runtime = runtime;

    if (api.loadNetworkFromFile(runtime, dla_path.c_str()) !=
        NEURONRUNTIME_NO_ERROR) {
      ml_loge("NeuronVar: NeuronRuntime_loadNetworkFromFile failed for: %s",
              dla_path.c_str());
      api.release(runtime);
      return NeuronStatusCode::FAILURE;
    }

    size_t num_in = 0, num_out = 0;
    api.getInputNumber(runtime, &num_in);
    api.getOutputNumber(runtime, &num_out);
    entry.numInputs = num_in;
    entry.numOutputs = num_out;

    entry.inputExactSizes.resize(num_in, 0);
    entry.inputPaddedSizes.resize(num_in, 0);
    for (size_t i = 0; i < num_in; ++i) {
      api.getInputSize(runtime, i, &entry.inputExactSizes[i]);
      api.getInputPaddedSize(runtime, i, &entry.inputPaddedSizes[i]);
      if (entry.inputPaddedSizes[i] != entry.inputExactSizes[i]) {
        ml_logw("NeuronVar: %s input[%zu] padded size %zu != exact size %zu; "
                "buffers must be sized to the padded value and any tensor-name/"
                "quant-param bookkeeping that assumes dense packing (KV-cache "
                "memcpy layout, etc.) needs to account for this before Phase 3",
                dla_path.c_str(), i, entry.inputPaddedSizes[i],
                entry.inputExactSizes[i]);
      }
    }

    entry.outputExactSizes.resize(num_out, 0);
    entry.outputPaddedSizes.resize(num_out, 0);
    for (size_t i = 0; i < num_out; ++i) {
      api.getOutputSize(runtime, i, &entry.outputExactSizes[i]);
      api.getOutputPaddedSize(runtime, i, &entry.outputPaddedSizes[i]);
      if (entry.outputPaddedSizes[i] != entry.outputExactSizes[i]) {
        ml_logw("NeuronVar: %s output[%zu] padded size %zu != exact size %zu; "
                "buffers must be sized to the padded value",
                dla_path.c_str(), i, entry.outputPaddedSizes[i],
                entry.outputExactSizes[i]);
      }
    }

    entry.initialized = true;
    rt_map.insert({dla_path, std::move(entry)});
    return NeuronStatusCode::SUCCESS;
  }
};

/** @brief ContextData adapter, analogous to QNNBackendVar. Unlike
 * QNNBackendVar, overrides getType() — QNNBackendVar's failure to do so
 * was a known gap in the QNN backend; ContextData::as<T>() only works
 * correctly when getType() actually identifies the subclass. */
class NeuronBackendVar : public ContextData {
public:
  NeuronBackendVar() : data(std::make_shared<NeuronVar>()) {}

  const char *getType() const override { return "neuron"; }

  std::shared_ptr<NeuronVar> &getVar() { return data; }

private:
  std::shared_ptr<NeuronVar> data;
};

} // namespace nntrainer

#endif /* __NEURON_CONTEXT_VAR_H__ */

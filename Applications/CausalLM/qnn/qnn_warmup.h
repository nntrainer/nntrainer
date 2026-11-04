// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qnn_warmup.h
 * @brief  Load-time graph warm-up shared by the QNN model lanes.
 *
 * The first `QnnGraph_execute` of a process carries one-time work that has
 * nothing to do with the prompt: `QNNGraph::execute` lazily registers an
 * RPC-mem handle for every input and output buffer it has not seen before
 * (including all KV tensors), the DSP faults the graph in, and the HTP clocks
 * ramp. Without a warm-up that first execute *is* prefill chunk 0, so the
 * whole constant lands inside the reported prefill window and a short prompt
 * "prefills" in several hundred ms of pure setup.
 *
 * Running every graph once on defined inputs at load time moves the cost into
 * model load, where it can be overlapped with UI, and leaves `run()` warm.
 * The outputs are discarded; the caller restores the KV cache afterwards.
 *
 * @note  Placement matters. This has to be called from a `load_weight()`
 *        override, *not* at the end of `initialize()`: the loader calls
 *        `initialize()` and only afterwards `load_weight()`, whose base
 *        implementation is what calls `model_handle->allocate(INFERENCE)`.
 *        Executing a graph before that allocation binds the QNN context to
 *        buffers `allocate()` then replaces, after which *every* execute
 *        fails with `QnnDsp err 6004` - silently, with no exception and no
 *        non-zero exit, while the metrics box happily reports an impossibly
 *        fast prefill.
 */

#ifndef __QNN_WARMUP_H__
#define __QNN_WARMUP_H__

#include "generate_qnn_utils.h"
#include "graph_parser.h"
#include "quick_dot_ai_qnn.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <variant>
#include <vector>

/**
 * @brief Process-wide total ns spent inside QnnGraph_execute().
 *
 * Provided by libnntrainer (nntrainer/utils/util_func.h). Declared weak so
 * this still builds and links against a libnntrainer without the counter, in
 * which case the accelerator fraction is simply reported as 0.
 */
#if defined(__GNUC__) && !defined(_WIN32)
extern "C" __attribute__((weak)) unsigned long long nntr_qnn_exec_ns(void);
#endif

namespace causallm {
namespace qnnwarm {

using clk = std::chrono::steady_clock;

/** @brief Milliseconds elapsed since @p t. */
inline double ms_since(const clk::time_point &t) {
  return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

/** @brief Accelerator-time counter, 0 when libnntrainer does not export it. */
inline unsigned long long exec_ns() {
#if defined(__GNUC__) && !defined(_WIN32)
  return nntr_qnn_exec_ns ? nntr_qnn_exec_ns() : 0ULL;
#else
  return 0ULL;
#endif
}

/** @brief True when @p name is set to exactly "0". */
inline bool env_off(const char *name) {
  const char *e = std::getenv(name);
  return e && e[0] == '0' && e[1] == '\0';
}

/** @brief True when @p name is set to a non-empty value other than "0". */
inline bool env_on(const char *name) {
  const char *e = std::getenv(name);
  return e && e[0] != '\0' && !(e[0] == '0' && e[1] == '\0');
}

/** @brief Raw pointer behind an IO_TensorType, whatever it holds. */
inline void *io_raw_ptr(const causallm::IO_TensorType &b) {
  return std::visit([](auto *p) { return static_cast<void *>(p); }, b);
}

/**
 * @brief Allocated size of a bound QNN input buffer, in bytes.
 *
 * NOT simply `GraphParser::get_tensor_size(info)`. When the model owns the
 * embedding, `input_embeds` is not the graph's quantised activation buffer at
 * all: `Quick_Dot_AI_QNN::constructModel` allocates it as
 * `sizeof(float) * <product of dims except the last>` token-id slots and lets
 * the embedding layer expand them into the real tensor. For a prefill graph
 * this can be a few hundred token IDs against a multi-megabyte embedding tensor
 * - writing the declared size into it would smash the heap.
 *
 * The variant alternative is the reliable discriminator: only the embedding
 * branch yields `float *`; every other input is allocated at exactly its
 * declared tensor size.
 */
inline size_t input_buffer_bytes(const causallm::IO_TensorType &b,
                                 const TensorInfo &info) {
  if (std::holds_alternative<float *>(b)) {
    size_t n = 1;
    for (size_t k = 0; k + 1 < info.dimensions.size(); ++k)
      n *= (size_t)std::max(0, info.dimensions[k]);
    return n * sizeof(float);
  }
  const int bytes = GraphParser::get_tensor_size(info);
  return bytes > 0 ? (size_t)bytes : 0;
}

/** @brief Outcome of a warm-up attempt, for the caller's log line. */
struct Stats {
  bool ran = false;     /**< false when NNTR_QNN_WARMUP=0 */
  size_t graphs = 0;    /**< graphs actually executed */
  double host_ms = 0.0; /**< wall time of the whole warm-up */
  double dsp_ms = 0.0;  /**< of which inside QnnGraph_execute */
};

/**
 * @brief Execute each of @p graph_names once on zeroed inputs, discarding the
 *        outputs, so `run()` never pays the first-execute cost.
 *
 * Enabled by default; `NNTR_QNN_WARMUP=0` skips it (A/B and bisect use).
 * Failures are logged and swallowed - a warm-up is an optimisation, never a
 * precondition: on any failure the model is still perfectly usable, it just
 * pays the cost in `run()` as before.
 *
 * Inputs are zeroed because their contents are irrelevant (the outputs are
 * discarded) but must be *defined*: the token-id slot behind `input_embeds`
 * is still uninitialised at this point and a NaN there would be a real
 * execute hazard rather than a harmless one. LoRA inputs are the one
 * exception and are deliberately left alone - `initialize()` fills them with
 * the neutral quantised value (or with the LoRA file) and `run()` never
 * refills them, so zeroing those would be a silent weight corruption.
 *
 * @param models       the model map (graph name -> handle/inputs/graph info)
 * @param graph_names  graphs to warm, in execution order
 * @note  The caller must restore its KV cache afterwards; this touches the
 *        `past_*_in` buffers and does not know how to reset them.
 */
inline Stats warmup_graphs(std::map<std::string, QNNModelInfo> &models,
                           const std::vector<std::string> &graph_names) {
  Stats s;
  if (env_off("NNTR_QNN_WARMUP")) {
    std::cout << "[WARMUP] disabled by NNTR_QNN_WARMUP=0" << std::endl;
    return s;
  }
  s.ran = true;

  const auto t0 = clk::now();
  const unsigned long long dsp0 = exec_ns();

  try {
    for (const std::string &g : graph_names) {
      auto it = models.find(g);
      if (it == models.end()) {
        std::cout << "[WARMUP] no such graph: " << g << std::endl;
        continue;
      }
      auto &inputs = it->second.model_inputs;
      auto &info = it->second.graph_info;

      const size_t n = std::min(inputs.size(), info.raw_inputs.size());
      for (size_t i = 0; i < n; ++i) {
        if (info.raw_inputs[i].name.find("_lora_") != std::string::npos)
          continue;
        void *ptr = io_raw_ptr(inputs[i]);
        const size_t bytes = input_buffer_bytes(inputs[i], info.raw_inputs[i]);
        if (ptr && bytes)
          std::memset(ptr, 0, bytes);
      }

      (void)run_qnn_inference(it->second.model_handle, 1, inputs, info);
      ++s.graphs;
    }
  } catch (const std::exception &e) {
    std::cout << "[WARMUP] skipped: " << e.what() << std::endl;
  }

  s.host_ms = ms_since(t0);
  s.dsp_ms = (double)(exec_ns() - dsp0) / 1.0e6;
  return s;
}

/** @brief The one log line every lane prints after warming. */
inline void log(const Stats &s) {
  if (!s.ran)
    return;
  std::cout << "[WARMUP] " << s.graphs << " graph(s) warmed in " << s.host_ms
            << " ms (dsp " << s.dsp_ms << " ms)" << std::endl;
}

} // namespace qnnwarm
} // namespace causallm

#endif /* __QNN_WARMUP_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * @file   quick_dot_ai_qnn_base.h
 * @brief  QNN model for Quick.AI template
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @note   This file implements a layer that executes QNN binary file within
 * transformer.h architecture.
 *
 */

#ifndef __QUICK_DOT_AI_QNN_H__
#define __QUICK_DOT_AI_QNN_H__

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "QuickAI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(fmt, ...) fprintf(stdout, fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif

#include "graph_parser.h"
#include "qnn_compat_types.h"
#include "streamer.h"
#include <atomic>
#include <iostream>
#include <set>
#include <string>
#include <transformer.h>

// Forward declaration for XGrammar
namespace causallm {
class XGrammar;
}

namespace causallm {
/**
 * @brief QNN Model info
 * @note  This struct contains QNN model information for execution.
 */
struct QNNModelInfo {
  GraphInfo graph_info;
  ModelHandle model_handle;
  std::vector<causallm::IO_TensorType> model_inputs;
};

/**
 * @brief Quick_Dot_AI_QNN_Base class
 * @note  This is the base class for QNN.
 */
class Quick_Dot_AI_QNN : public Transformer {

public:
  Quick_Dot_AI_QNN(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    LOGD("--------------------------------- Quick_Dot_AI_QNN");
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  ~Quick_Dot_AI_QNN() override;

  void initialize() override;
  void initialize(const std::string &native_lib_dir);

  void load_weight(const std::string &weight_path) override;

  // No-op: QNN runs precompiled graphs (the `models` map), so the base
  // Transformer symbolic `model` is never built (stays null) and there are no
  // QS4CX symbolic-graph weights to repack. The base repack_weight() would
  // dereference the null `model` via model->forEachLayer() and segfault, so QNN
  // models must override it away.
  void repack_weight() override;

  void save_weight(const std::string &weight_path) override;

  virtual bool supportsKvCachePersistence() const { return false; }
  virtual int getKvLen() const { return 0; }
  std::string getOutput(int batch_idx = 0) const {
    (void)batch_idx;
    return last_output_;
  }
  virtual void resetKvCache() {
    throw std::runtime_error("QNN KV cache is not supported by this model");
  }
  virtual void saveKvCache(const std::string &cache_path) const {
    (void)cache_path;
    throw std::runtime_error("QNN KV cache is not supported by this model");
  }
  virtual void loadKvCache(const std::string &cache_path) {
    (void)cache_path;
    throw std::runtime_error("QNN KV cache is not supported by this model");
  }

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  std::pair<Tensor, Tensor> constructModel() override;

  /**
   * @brief Sample token with XGrammar
   */
  int sample(uint16_t *pointer, int length, int *tokens, int number_of_tokens,
             float logit_scale, int logit_offset, float repetition_penalty,
             float temperature, float top_p, int top_k);

  /**
   * @brief Attach (or detach) a BaseStreamer to intercept per-token output.
   *        Passing nullptr detaches any currently-attached streamer.
   */
  void setStreamer(::BaseStreamer *streamer) { streamer_ = streamer; }

  /**
   * @brief Attach an XGrammar instance for grammar-constrained generation.
   */
  void setXGrammar(XGrammar *grammar) { xgrammar_ = grammar; }

  /**
   * @brief Reset the XGrammar matcher state after generation.
   */
  void resetXGrammar();

  /**
   * @brief Request cancellation of the current run().
   *
   * Thread-safe: sets the stop flag atomically, causing the token
   * generation loop to exit at the next token boundary. Safe to call
   * from any thread (e.g., from a UI cancel button handler).
   */
  void requestStop() {
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                        "requestStop: setting stop_requested_ to true");
#else
    std::cout << "[DEBUG] requestStop: setting stop_requested_ to true"
              << std::endl;
#endif
    stop_requested_.store(true, std::memory_order_release);
  }

  /**
   * @brief Check if stop has been requested.
   * Thread-safe: can be called from any thread.
   */
  bool isStopRequested() const {
    return stop_requested_.load(std::memory_order_acquire);
  }

  /**
   * @brief Clear the stop request flag.
   * Thread-safe: can be called from any thread.
   */
  void clearStopRequest() {
    stop_requested_.store(false, std::memory_order_release);
  }

  Tensor createTransformerDecoderBlock(const int layer_id,
                                       Tensor input) override;

  Tensor createAttention(const int layer_id, int seq_len, int n_heads,
                         int head_dim, Tensor query, Tensor key,
                         Tensor value) override;

  Tensor createMlp(const int layer_id, int dim, int hidden_dim,
                   Tensor input) override;

  void registerCustomLayers() override;

  static void quantize_uint16_memcpy(float *src, uint16_t *dest, int count,
                                     float scale, int offset);

protected:
  // nntr_config
  std::string model_file_name;
  std::string embedding_path;
  std::string binary_config_path;
  std::string native_lib_dir_;
  std::vector<std::string> graphs_to_use;
  std::string last_output_;

  // config
  int vocab_size;

  // generation_config
  int padding_token;
  int eos_token;
  int top_k;
  float top_p;
  float temperature;
  float repetition_penalty;
  float logit_scale;
  int logit_offset;

  // LoRA path (optional)
  std::string lora_path;

  // Model map, key: graph name, value: QNN model info
  std::map<std::string, QNNModelInfo> models;

  bool uses_embedding = true;

  // Optional external embedding file path (absolute after fix_paths).
  // Only used when uses_embedding=false — derived classes mmap this
  // and provide per-token lookup during generation.
  std::string embedding_file_name;

  // Streaming support
  ::BaseStreamer *streamer_ = nullptr;

  // XGrammar instance for grammar-constrained generation (non-owning)
  XGrammar *xgrammar_ = nullptr;

  /**
   * @brief Cooperative cancellation flag set by the attached streamer's
   *        put() returning non-zero, or by requestStop() from any thread.
   *        The token generation loop in run() checks this once per iteration
   *        and breaks out at the next safe boundary.
   *
   * Uses std::atomic for thread-safe access from any thread (e.g.,
   * cancel button handler in UI thread).
   */
  std::atomic<bool> stop_requested_{false};

  /// Raw execution timing for console output (replaces global variable).
  std::chrono::duration<double> raw_exec_seconds;

  // Tracked resource management: all allocate()'d pointers are recorded
  // here so that ~Quick_Dot_AI_QNN can free them in one pass without
  // risking double-free or omission.
  std::set<void *> allocated_ptrs_;

  /// Allocate size bytes via allocate() and record the pointer for
  /// automatic cleanup in the destructor.
  void *tracked_allocate(size_t size);

  /// Deallocate every tracked pointer and clear the set.
  void deallocate_all();

  /**
   * @brief Convert the run() prompt argument into UTF-8 text for tokenization.
   *
   * This helper only normalizes the platform string representation required by
   * Transformer::run(WSTR): on Windows it converts wide strings to UTF-8, while
   * on Android/Linux WSTR is already UTF-8-compatible and is returned as-is.
   * Chat-template formatting and incremental conversation handling must remain
   * in the API layer before the prompt reaches the model.
   */
  static std::string promptToUtf8(const WSTR &prompt);
};

} // namespace causallm

#endif /* __QUICK_DOT_AI_QNN_BASE_H__ */

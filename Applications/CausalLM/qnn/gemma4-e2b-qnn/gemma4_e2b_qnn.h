// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gemma4_e2b_qnn.h
 * @brief  QNN model extension for Gemma 4 E2B (with PLE)
 */

#ifndef __GEMMA4_E2B_QNN_H__
#define __GEMMA4_E2B_QNN_H__

#include "generate_qnn_utils.h"
#include "quick_dot_ai_qnn.h"

#include <cstdint>
#include <string>
#include <vector>

namespace causallm {

class Gemma4_E2B_QNN : public Quick_Dot_AI_QNN {

public:
  static constexpr const char *architectures = "Gemma4_E2B_QNN";

  Gemma4_E2B_QNN(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Quick_Dot_AI_QNN(cfg, generation_cfg, nntr_cfg) {
    LOGD("Gemma4 E2B parameters set up ");
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  ~Gemma4_E2B_QNN() override;

  void initialize();

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

  // Gemma4 E2B implements the full QNN KV-cache machinery
  // (initialize_kv_cache / fresh_kvs). run() resets the cache at the start of
  // every generation via resetKvCache(); route that base hook to the real
  // implementation instead of inheriting the stub that throws
  // "QNN KV cache is not supported by this model".
  void resetKvCache() override { initialize_kv_cache(); }
  int getKvLen() const override { return kv_len; }

private:
  // -------------------------------------------------------------------
  // PLE: tri-mode (auto-detected from `ple_file_name` + manifest
  // datatype).
  //   *.json + datatype="ufixed8" → tensorwise 4-bit (legacy):
  //       single (ple_scale_, ple_offset_) for the whole table.
  //   *.json + datatype="sfixed4" → per-row-per-layer signed 4-bit:
  //       ple_row_layer_scales_[token_id*num_layers + layer] is the
  //       float scale, no offset, sign-extend the 4-bit nibble.
  //   any other extension → raw UINT16 binary (already in each
  //       layer's consumer quant space; per-layer fill is a memcpy).
  // -------------------------------------------------------------------
  bool ple_is_4bit_ = false;
  bool ple_is_signed4_ = false; // sfixed4 (per-row-per-layer)
  int ple_fd_ = -1;
  const uint8_t *ple_mmap_ = nullptr;      // 4-bit byte view
  const uint16_t *ple_u16_mmap_ = nullptr; // raw uint16 view
  size_t ple_file_size_ = 0;
  float ple_scale_ = 1.0f; // ufixed8 only
  int ple_offset_ = 0;     // ufixed8 only
  // sfixed4: per-token per-layer scales. Layout is row-major
  // [vocab][num_layers]; index with token_id * ple_layers_ + layer.
  std::vector<float> ple_row_layer_scales_;
  size_t ple_row_elems_ = 0;
  size_t ple_row_bytes_ = 0;
  size_t ple_layers_ = 0;
  size_t ple_per_layer_ = 0;

  std::vector<uint16_t *> prefill_per_layer_dst_;    // 14
  std::vector<uint16_t *> generation_per_layer_dst_; // 35
  std::vector<float> prefill_per_layer_scale_;
  std::vector<int> prefill_per_layer_offset_;
  std::vector<float> generation_per_layer_scale_;
  std::vector<int> generation_per_layer_offset_;
  // Model layer index that each prefill/generation `per_layer_inputs_N` slot
  // corresponds to. Built from the integer `N` parsed from the tensor name
  // (sorted ascending). The PLE binary file is laid out per model-layer 0..L-1
  // per row, so source rows must be indexed by these model indices, NOT by
  // the dense slot index.
  std::vector<int> prefill_per_layer_model_index_;
  std::vector<int> generation_per_layer_model_index_;
  std::vector<uint8_t> prefill_kv_zero_byte_;

  void open_ple_file_();
  void close_ple_file_();
  void fill_prefill_ple_chunk_(const std::vector<int> &tokens, int chunk_idx,
                               int chunk_len);
  void fill_generation_ple_(int token_id);

  // -------------------------------------------------------------------
  // Input / output tensors
  // -------------------------------------------------------------------
  uint16_t *attention_mask;
  uint16_t *sliding_attention_mask;
  uint16_t *generation_attention_mask;
  uint16_t *generation_sliding_attention_mask;

  uint16_t *position_ids_cos;
  uint16_t *position_ids_sin;
  uint16_t *swa_position_ids_cos;
  uint16_t *swa_position_ids_sin;
  uint16_t *prefill_position_ids_cos;
  uint16_t *prefill_position_ids_sin;
  uint16_t *prefill_swa_position_ids_cos;
  uint16_t *prefill_swa_position_ids_sin;
  uint16_t *generation_position_ids_cos;
  uint16_t *generation_position_ids_sin;
  uint16_t *generation_swa_position_ids_cos;
  uint16_t *generation_swa_position_ids_sin;

  float *input_sample;
  float *generation_sample;

  // -------------------------------------------------------------------
  // KV cache (Gauss 3.6 pattern adapted for Gemma 4: 2 KV per layer)
  // -------------------------------------------------------------------
  int kv_len = 0;
  bool conversation_started_ = false;

  std::vector<uint16_t *> kvs;       // generation KV (canonical)
  std::vector<uint16_t *> fresh_kvs; // initial state copy
  std::vector<int> kv_sizes;
  std::vector<int> kv_row_lengths; // per layer
  std::vector<int> kv_columns;     // per layer

  std::vector<uint8_t *> prefill_kvs;
  std::vector<int> prefill_kv_sizes;
  std::vector<int> prefill_kv_row_lengths;
  std::vector<int> prefill_to_generation_kv_indices;
  std::vector<int> prefill_kv_is_key;

  std::vector<QnnKvOutputBinding> prefill_output_kv_bindings;
  std::vector<QnnKvOutputBinding> generation_output_kv_bindings;

  void initialize_kv_cache();
  void reset_prefill_kv_cache_inputs();
  void sync_generation_kv_cache_to_prefill();

  // -------------------------------------------------------------------
  // Mask / RoPE element counts
  // -------------------------------------------------------------------
  int prefill_attention_mask_elements = 0;
  int prefill_attention_mask_columns = 0;
  int prefill_sliding_attention_mask_elements = 0;
  int prefill_sliding_attention_mask_columns = 0;
  int generation_attention_mask_elements = 0;
  int generation_sliding_attention_mask_elements = 0;
  int generation_logits_output_index = -1;
  int generation_full_kv_past_length = 0;
  int generation_sliding_kv_past_length = 0;
  int rope_cache_seq_len = 0;

  // -------------------------------------------------------------------
  // Model config
  // -------------------------------------------------------------------
  int num_hidden_layers;
  int max_window_layers;
  int hidden_size;
  int sequence_length;
  int vocab_size;
  int max_seq_len;
  int sliding_window;
  float local_rope_theta;
  float rope_theta_sliding = 10000.0f;
  float rope_scaling_factor_sliding = 1.0f;
  std::string rope_type_sliding = "default";
  float rope_theta_full = 1000000.0;
  float rope_partial_factor = 1.0f;
  float rope_scaling_factor_full = 1.0f;
  std::string rope_type_full = "default";

  int context_size;
  int pos_dim;
  int swa_pos_dim;
  int g_head_dim;
  int l_head_dim;
  int head_dim;

  int padding_token;
  std::vector<int> eos_tokens;
  int top_k;
  float top_p;
  float temperature;
  float repetition_penalty;
  float logit_scale;
  int logit_offset;
  // Gemma soft-cap on final logits. config.json key:
  // "final_logit_softcapping": 30.0. 0 disables the cap.
  float final_logit_softcapping = 0.0f;

  std::string lora_path;
  std::string ple_file_name;
};

} // namespace causallm

#endif /* __GEMMA4_E2B_QNN_H__ */

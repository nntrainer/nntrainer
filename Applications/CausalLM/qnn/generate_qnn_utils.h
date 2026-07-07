// SPDX-License-Identifier: Apache-2.0
/**
 * @file   generate_qnn_utils.h
 * @brief  Helper utilities for QNN generation: KV cache bookkeeping,
 *         attention mask/rope buffer prep, and sampling.
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __GENERATE_QNN_UTILS_HPP__
#define __GENERATE_QNN_UTILS_HPP__

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "android_memory_allocator.h"
#include "graph_parser.h"
#include "qnn_compat_types.h"
#include <model.h>
#include <tokenizers_cpp.h>

using ModelHandle = std::unique_ptr<ml::train::Model>;
using IO_TensorType = causallm::IO_TensorType;

extern std::mt19937 rng;

/**
 * @brief Run a QNN graph through the main nntrainer Model::inference API.
 *
 * The main nntrainer inference API is type-erased: it takes/returns
 * std::vector<float*> as opaque buffer carriers and recovers the real per-slot
 * dtype from the model's configured tensor dims (see NeuralNetwork::
 * mapExternalTensor, which reinterpret_casts the float* to the graph's input
 * dtype). The OLD nntrainer instead used a dtype-tagged variant (IO_TensorType)
 * at the API boundary.
 *
 * To bridge the two without changing any runtime behavior (the bound pointer
 * addresses are identical either way), this helper:
 *   1. strips each input IO_TensorType down to its raw pointer (as float*),
 *   2. calls Model::inference,
 *   3. re-tags each returned float* with the graph's declared output dtype so
 *      callers can keep using std::get<uint16_t*>/std::get<uint8_t*> as before.
 *
 * @param model     Model handle for the graph
 * @param batch     Batch size
 * @param inputs    Input buffers (dtype-tagged); only the pointer is used
 * @param graph_info Parsed graph info whose raw_outputs supply output dtypes
 * @return Output buffers re-tagged with their declared dtype
 */
std::vector<IO_TensorType>
run_qnn_inference(ModelHandle &model, unsigned int batch,
                  const std::vector<IO_TensorType> &inputs,
                  const GraphInfo &graph_info);

/**
 * @brief Maps a QNN graph output slot to its destination KV cache slot.
 */
struct QnnKvOutputBinding {
  int output_index;
  int kv_index;
  int layer_index;
  bool is_key;
};

std::tuple<uint16_t *, uint16_t *>
get_cos_sin(int context_size, int pos_dim, const double theta,
            const std::string &rope_type = "default",
            double partial_rotary_factor = 1.0,
            double rope_scaling_factor = 1.0, int rope_head_dim = 0);

bool qnn_starts_with(const std::string &value, const std::string &prefix);

int find_tensor_index_or_minus_one(const TensorInfoList &tensor_infos,
                                   const std::string &tensor_name);

std::string kv_output_to_input_name(const std::string &output_name);

int get_kv_row_length(const TensorInfo &tensor_info, bool is_key,
                      const std::string &tensor_name);

void copy_kv_cache_window(uint8_t *dest, int dest_row_length,
                          const uint8_t *src, int src_row_length,
                          int history_length, bool is_key,
                          int num_columns = 128);

std::vector<QnnKvOutputBinding> build_kv_output_bindings(
  const TensorInfoList &outputs,
  const std::unordered_map<std::string, int> &generation_kv_index_by_name,
  const std::string &graph_name, int kv_per_layer = 4);

void append_outputs_to_kv_cache(const std::vector<IO_TensorType> &step_outputs,
                                const std::vector<QnnKvOutputBinding> &bindings,
                                const std::vector<uint8_t *> &kvs,
                                const std::vector<int> &kv_row_lengths,
                                int target_position, int rows,
                                int src_row_length,
                                const std::string &graph_name,
                                const std::vector<int> *kv_columns = nullptr);

void process_key(uint8_t *pointer, int row, int column, uint8_t *dest, int idx,
                 int dest_row_length, int src_row_length);

void process_value(uint8_t *pointer, int row, int column, uint8_t *dest,
                   int idx);

void fill_attention_mask_with_length(int rows, int columns, int length,
                                     uint16_t *attention_mask);

void fill_attention_mask_with_prev_length(int rows, int columns, int length,
                                          uint16_t *attention_mask);

uint16_t *get_zero_memory(int size, int zero_point);

void fill_generation_inputs(
  float *generation_sample, int current_token,
  uint16_t *generation_attention_mask, int generation_attention_mask_elements,
  uint16_t *generation_sliding_attention_mask,
  int generation_sliding_attention_mask_elements,
  int generation_full_kv_past_length, int generation_sliding_kv_past_length,
  uint16_t *generation_position_ids_cos, uint16_t *generation_position_ids_sin,
  const uint16_t *position_ids_cos, const uint16_t *position_ids_sin,
  int pos_dim, uint16_t *generation_swa_position_ids_cos,
  uint16_t *generation_swa_position_ids_sin,
  const uint16_t *swa_position_ids_cos, const uint16_t *swa_position_ids_sin,
  int swa_pos_dim, int position, int rope_cache_seq_len);

void fill_generation_inputs_u16(
  uint16_t *generation_attention_mask, int generation_attention_mask_elements,
  uint16_t *generation_sliding_attention_mask,
  int generation_sliding_attention_mask_elements,
  int generation_full_kv_past_length, int generation_sliding_kv_past_length,
  uint16_t *generation_position_ids_cos, uint16_t *generation_position_ids_sin,
  const uint16_t *position_ids_cos, const uint16_t *position_ids_sin,
  int pos_dim, uint16_t *generation_swa_position_ids_cos,
  uint16_t *generation_swa_position_ids_sin,
  const uint16_t *swa_position_ids_cos, const uint16_t *swa_position_ids_sin,
  int swa_pos_dim, int position, int rope_cache_seq_len);

int sample(uint16_t *pointer, int length, int *tokens, int number_of_tokens,
           float logit_scale, int logit_offset, float repetition_penalty,
           float temperature, float top_p, int top_k,
           float final_logit_softcapping = 0.0f);

#endif // __GENERATE_QNN_UTILS_HPP__

// SPDX-License-Identifier: Apache-2.0
/**
 * @file   generate_qnn_utils.cpp
 * @brief  Helper utilities for QNN generation: KV cache bookkeeping,
 *         attention mask/rope buffer prep, and sampling.
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include "generate_qnn_utils.h"

#include "android_memory_allocator.h"
#include <thread_manager.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <model.h>
#include <tokenizers_cpp.h>

#include <cstdlib>

// Sampling RNG. One fixed seed everywhere (causallm::kDefaultSamplingSeed,
// shared with the CausalLM sampler), re-seeded at the start of every run, so
// the same input gives the same tokens on every run, in every process and on
// every C++ runtime of the same hardware. QUICKAI_SAMPLE_SEED overrides the
// seed. std::random_device is deliberately not used: a request must be
// reproducible.
static uint64_t sampling_seed() {
  if (const char *s = std::getenv("QUICKAI_SAMPLE_SEED"))
    return static_cast<uint64_t>(std::strtoull(s, nullptr, 10));
  return causallm::kDefaultSamplingSeed;
}
causallm::SamplingRng rng{sampling_seed()};

void reseed_sampling_rng() { rng.seed(sampling_seed()); }

std::vector<IO_TensorType>
run_qnn_inference(ModelHandle &model, unsigned int batch,
                  const std::vector<IO_TensorType> &inputs,
                  const GraphInfo &graph_info) {
  // 1. Strip dtype tags: the main nntrainer inference API is type-erased and
  //    treats float* purely as an opaque buffer carrier (the real dtype is
  //    recovered from the graph's input dims inside mapExternalTensor). The
  //    bound pointer address is preserved across the reinterpret_cast, so no
  //    runtime behavior changes vs. the OLD variant-typed API.
  std::vector<float *> float_inputs;
  float_inputs.reserve(inputs.size());
  for (const auto &in : inputs) {
    float *ptr =
      std::visit([](auto *p) { return reinterpret_cast<float *>(p); }, in);
    float_inputs.push_back(ptr);
  }

  std::vector<float *> float_outputs = model->inference(batch, float_inputs);

  // 2. Re-tag each output buffer with the graph's declared output dtype so
  //    callers can keep using std::get<uint16_t*>/std::get<uint8_t*>.
  std::vector<IO_TensorType> outputs;
  outputs.reserve(float_outputs.size());
  for (size_t i = 0; i < float_outputs.size(); ++i) {
    std::string dtype = (i < graph_info.raw_outputs.size())
                          ? graph_info.raw_outputs[i].data_type
                          : std::string();
    if (dtype == "QNN_DATATYPE_UFIXED_POINT_8") {
      outputs.emplace_back(reinterpret_cast<uint8_t *>(float_outputs[i]));
    } else if (dtype == "QNN_DATATYPE_UFIXED_POINT_16" ||
               dtype == "QNN_DATATYPE_FLOAT_16") {
      outputs.emplace_back(reinterpret_cast<uint16_t *>(float_outputs[i]));
    } else {
      // Unknown / FP32 / unspecified: keep as float* (matches OLD default).
      outputs.emplace_back(float_outputs[i]);
    }
  }
  return outputs;
}

namespace {

constexpr int kQnnKvNumColumns = 128;

constexpr float kRopeQuantScale = 3.051804378628731e-05f;
constexpr int kRopeQuantOffset = -32768;

uint16_t quantize_rope_value(double value, double attention_factor) {
  const double q =
    (value * attention_factor) / kRopeQuantScale - kRopeQuantOffset;
  if (q <= 0.0)
    return 0;
  if (q >= 65535.0)
    return 65535;
  return static_cast<uint16_t>(std::lrint(q));
}

} // namespace

std::tuple<uint16_t *, uint16_t *>
get_cos_sin(int context_size, int pos_dim, const double theta,
            const std::string &rope_type, double partial_rotary_factor,
            double rope_scaling_factor, int rope_head_dim) {
  double attention_factor = 1.0;
  const double scaling_factor =
    rope_scaling_factor > 0.0 ? rope_scaling_factor : 1.0;
  const int frequency_dim = rope_head_dim > 0 ? rope_head_dim : pos_dim * 2;
  // inv_freq indexes angle pairs, so the exponent advances by 2/head_dim.
  const double exponent = 2.0 / static_cast<double>(frequency_dim);
  std::vector<double> inv_freq(pos_dim, 0.0);

  if (rope_type == "default") {
    for (int j = 0; j < pos_dim; j++)
      inv_freq[j] = 1.0 / std::pow(theta, j * exponent);
  } else if (rope_type == "linear" || rope_type == "proportional") {
    int rotary_freq_count = pos_dim;

    if (rope_type == "proportional") {
      double proportion = partial_rotary_factor;
      if (proportion < 0.0)
        proportion = 0.0;
      if (proportion > 1.0)
        proportion = 1.0;
      rotary_freq_count = static_cast<int>(
        std::floor(proportion * static_cast<double>(frequency_dim) / 2.0));
      rotary_freq_count = std::max(0, std::min(pos_dim, rotary_freq_count));
    }

    for (int j = 0; j < rotary_freq_count; j++)
      inv_freq[j] = (1.0 / std::pow(theta, j * exponent)) / scaling_factor;
  } else {
    for (int j = 0; j < pos_dim; j++)
      inv_freq[j] = 1.0 / std::pow(theta, j * exponent);
  }

  // Partial RoPE (Gemma 4 full attention uses partial_rotary_factor=0.25):
  // only the first `effective_dim` lanes carry an actual rotation; the
  // remaining lanes are identity (cos=1, sin=0) so multiplication is
  // pass-through. Without this the non-rotary lanes get a small bogus
  // rotation that compounds across the 35 transformer layers and the
  // model collapses into repetition after a few dozen tokens.
  int effective_dim = pos_dim;
  if (partial_rotary_factor > 0.0 && partial_rotary_factor < 1.0) {
    effective_dim =
      static_cast<int>(std::floor(pos_dim * partial_rotary_factor));
    if (effective_dim < 0)
      effective_dim = 0;
    if (effective_dim > pos_dim)
      effective_dim = pos_dim;
  }

  uint16_t *cos_val =
    (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);
  uint16_t *sin_val =
    (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);

  // Quantized identity values for non-rotary lanes.
  const uint16_t cos_one = quantize_rope_value(1.0, attention_factor);
  const uint16_t sin_zero = quantize_rope_value(0.0, attention_factor);

  for (int i = 0; i < context_size; i++) {
    for (int j = 0; j < effective_dim; j++) {
      const double freq = i * inv_freq[j];
      cos_val[i * pos_dim + j] =
        quantize_rope_value(std::cos(freq), attention_factor);
      sin_val[i * pos_dim + j] =
        quantize_rope_value(std::sin(freq), attention_factor);
    }
    for (int j = effective_dim; j < pos_dim; j++) {
      cos_val[i * pos_dim + j] = cos_one;
      sin_val[i * pos_dim + j] = sin_zero;
    }
  }
  return std::make_tuple(cos_val, sin_val);
}

bool qnn_starts_with(const std::string &value, const std::string &prefix) {
  return value.compare(0, prefix.size(), prefix) == 0;
}

int find_tensor_index_or_minus_one(const TensorInfoList &tensor_infos,
                                   const std::string &tensor_name) {
  for (size_t idx = 0; idx < tensor_infos.size(); idx++) {
    if (tensor_infos[idx].name == tensor_name) {
      return static_cast<int>(idx);
    }
  }
  return -1;
}

std::string kv_output_to_input_name(const std::string &output_name) {
  if (output_name.size() >= 4 &&
      output_name.compare(output_name.size() - 4, 4, "_out") == 0) {
    return output_name.substr(0, output_name.size() - 4) + "_in";
  }
  return output_name;
}

int get_kv_row_length(const TensorInfo &tensor_info, bool is_key,
                      const std::string &tensor_name) {
  if (tensor_info.dimensions.size() < 2) {
    throw std::runtime_error("Unexpected KV dims for " + tensor_name);
  }

  if (is_key) {
    return tensor_info.dimensions.back();
  }

  return tensor_info.dimensions[tensor_info.dimensions.size() - 2];
}

void copy_kv_cache_window(uint8_t *dest, int dest_row_length,
                          const uint8_t *src, int src_row_length,
                          int history_length, bool is_key, int num_columns) {
  if (dest == nullptr || src == nullptr || history_length <= 0 ||
      dest_row_length <= 0 || src_row_length <= 0) {
    return;
  }

  const int available_history = std::min(history_length, src_row_length);
  const int copy_length = std::min(available_history, dest_row_length);
  const int src_start = available_history - copy_length;
  const bool align_to_tail =
    history_length >= src_row_length && dest_row_length > copy_length;
  const int dest_start = align_to_tail ? dest_row_length - copy_length : 0;

  if (is_key) {
    for (int col = 0; col < num_columns; ++col) {
      std::memcpy(dest + col * dest_row_length + dest_start,
                  src + col * src_row_length + src_start, copy_length);
    }
  } else {
    std::memcpy(dest + dest_start * num_columns, src + src_start * num_columns,
                copy_length * num_columns);
  }
}

std::vector<QnnKvOutputBinding> build_kv_output_bindings(
  const TensorInfoList &outputs,
  const std::unordered_map<std::string, int> &generation_kv_index_by_name,
  const std::string &graph_name, int kv_per_layer) {
  std::vector<QnnKvOutputBinding> bindings;
  for (size_t idx = 0; idx < outputs.size(); idx++) {
    const auto &name = outputs[idx].name;
    if (!qnn_starts_with(name, "past_")) {
      continue;
    }

    const auto input_name = kv_output_to_input_name(name);
    auto it = generation_kv_index_by_name.find(input_name);
    if (it == generation_kv_index_by_name.end()) {
      throw std::runtime_error(graph_name +
                               " KV output has no generation input: " + name);
    }

    const int kv_index = it->second;
    bindings.push_back({static_cast<int>(idx), kv_index,
                        kv_index / kv_per_layer,
                        qnn_starts_with(name, "past_key_")});
  }
  return bindings;
}

void append_outputs_to_kv_cache(const std::vector<IO_TensorType> &step_outputs,
                                const std::vector<QnnKvOutputBinding> &bindings,
                                const std::vector<uint8_t *> &kvs,
                                const std::vector<int> &kv_row_lengths,
                                int target_position, int rows,
                                int src_row_length,
                                const std::string &graph_name,
                                const std::vector<int> *kv_columns) {
  for (const auto &binding : bindings) {
    if (binding.output_index < 0 ||
        binding.output_index >= static_cast<int>(step_outputs.size()) ||
        binding.kv_index < 0 ||
        binding.kv_index >= static_cast<int>(kvs.size()) ||
        binding.layer_index < 0 ||
        binding.layer_index >= static_cast<int>(kv_row_lengths.size()) ||
        (kv_columns != nullptr &&
         binding.layer_index >= static_cast<int>(kv_columns->size()))) {
      throw std::runtime_error(graph_name +
                               " output KV binding is out of range");
    }
  }

  nntrainer::ThreadManager::Global().parallel_for(
    0, bindings.size(), [&](size_t binding_idx) {
      const auto &binding = bindings[binding_idx];
      const int dest_row_length = kv_row_lengths[binding.layer_index];
      const int num_columns =
        kv_columns ? (*kv_columns)[binding.layer_index] : kQnnKvNumColumns;
      auto output = std::get<uint8_t *>(step_outputs[binding.output_index]);
      auto dest = kvs[binding.kv_index];

      int target_idx = target_position;
      const int valid_before = std::min(target_position, dest_row_length);
      const int shift = valid_before + rows - dest_row_length;
      if (shift > 0) {
        target_idx = valid_before - shift;
        if (binding.is_key) {
          for (int col = 0; col < num_columns; ++col) {
            uint8_t *col_base = dest + col * dest_row_length;
            std::memmove(col_base, col_base + shift, dest_row_length - shift);
          }
        } else {
          std::memmove(dest, dest + shift * num_columns,
                       (dest_row_length - shift) * num_columns);
        }
      }

      if (binding.is_key) {
        process_key(output, rows, num_columns, dest, target_idx,
                    dest_row_length, src_row_length);
      } else {
        process_value(output, rows, num_columns, dest, target_idx);
      }
    });
}

// NNTR_HOSTOPT=0 restores the pre-optimisation host paths exactly, so an A/B
// runs one binary and cannot be confounded by codegen or layout differences.
static bool host_fastpath_enabled() {
  static const bool on = []() {
    const char *e = std::getenv("NNTR_HOSTOPT");
    return !(e && e[0] == '0');
  }();
  return on;
}

void process_key(uint8_t *pointer, int row, int column, uint8_t *dest, int idx,
                 int dest_row_length, int src_row_length) {
  // format: 1:1:col:row
  // Decode appends exactly one position, and `row` is a runtime value, so the
  // generic form below emits a real memcpy call per column - 4 608 libc calls
  // per generated token across the 15 KV layers, each moving one byte. Store
  // it directly instead; identical bytes, no call.
  if (row == 1 && host_fastpath_enabled()) {
    for (int i = 0; i < column; i++)
      dest[(size_t)i * dest_row_length + idx] =
        pointer[(size_t)i * src_row_length];
    return;
  }
  for (int i = 0; i < column; i++) {
    std::memcpy(dest + i * dest_row_length + idx, pointer + i * src_row_length,
                row);
  }
}

void process_value(uint8_t *pointer, int row, int column, uint8_t *dest,
                   int idx) {
  // format: 1:1:row:col
  for (int i = 0; i < row; i++) {
    std::memcpy(dest + (idx + i) * column, pointer + i * column, column);
  }
}

void fill_attention_mask_with_length(int rows, int columns, int length,
                                     uint16_t *attention_mask) {

  int index = 0;
  // TODO use std::fill_n?
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (i >= length) {
        attention_mask[index] = std::numeric_limits<uint16_t>::min();
      } else if (j >= columns - rows && j <= columns - rows + i) {
        attention_mask[index] = std::numeric_limits<uint16_t>::max();
      } else {
        attention_mask[index] = std::numeric_limits<uint16_t>::min();
      }
      index++;
    }
  }
}

void fill_attention_mask_with_prev_length(int rows, int columns, int length,
                                          uint16_t *attention_mask) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < length; j++) {
      attention_mask[i * columns + j] = std::numeric_limits<uint16_t>::max();
    }
  }
}

void fill_sliding_mask_with_prev_length(int rows, int columns, int past_len,
                                        int window, uint16_t *attention_mask) {
  if (window <= 0) {
    fill_attention_mask_with_prev_length(rows, columns, past_len,
                                         attention_mask);
    return;
  }
  for (int i = 0; i < rows; i++) {
    int j0 = i + past_len - window;
    if (j0 < 0)
      j0 = 0;
    for (int j = j0; j < past_len; j++) {
      attention_mask[i * columns + j] = std::numeric_limits<uint16_t>::max();
    }
  }
}

uint16_t *get_zero_memory(int size, int zero_point) {
  uint16_t *memory = (uint16_t *)allocate(size * sizeof(uint16_t));
  for (int i = 0; i < size; i++)
    memory[i] = zero_point;
  return memory;
}

void fill_generation_inputs_common(
  uint16_t *generation_attention_mask, int generation_attention_mask_elements,
  uint16_t *generation_sliding_attention_mask,
  int generation_sliding_attention_mask_elements,
  int generation_full_kv_past_length, int generation_sliding_kv_past_length,
  uint16_t *generation_position_ids_cos, uint16_t *generation_position_ids_sin,
  const uint16_t *position_ids_cos, const uint16_t *position_ids_sin,
  int pos_dim, uint16_t *generation_swa_position_ids_cos,
  uint16_t *generation_swa_position_ids_sin,
  const uint16_t *swa_position_ids_cos, const uint16_t *swa_position_ids_sin,
  int swa_pos_dim, int position, int rope_cache_seq_len) {
  if (position < 0 || position >= rope_cache_seq_len) {
    throw std::runtime_error("Generation position is out of rope cache");
  }

  std::fill_n(generation_attention_mask, generation_attention_mask_elements, 0);
  std::fill_n(generation_sliding_attention_mask,
              generation_sliding_attention_mask_elements, 0);

  // Self-attention slot: the current token's own KV position.
  // Using full_kv_past_length (= per-row elements - 1) rather than
  // attention_mask_elements - 1 ensures the slot lands in row 0 when the
  // generation graph batches multiple sequence positions
  // (generation_seq_len>1).
  generation_attention_mask[generation_full_kv_past_length] =
    std::numeric_limits<uint16_t>::max();
  generation_sliding_attention_mask[generation_sliding_kv_past_length] =
    std::numeric_limits<uint16_t>::max();

  for (int i = 0; i < position && i < generation_full_kv_past_length; i++) {
    generation_attention_mask[i] = std::numeric_limits<uint16_t>::max();
  }
  for (int i = 0; i < position && i < generation_sliding_kv_past_length; i++) {
    generation_sliding_attention_mask[i] = std::numeric_limits<uint16_t>::max();
  }

  std::memcpy(generation_position_ids_cos,
              position_ids_cos + position * pos_dim,
              pos_dim * sizeof(uint16_t));
  std::memcpy(generation_position_ids_sin,
              position_ids_sin + position * pos_dim,
              pos_dim * sizeof(uint16_t));
  std::memcpy(generation_swa_position_ids_cos,
              swa_position_ids_cos + position * swa_pos_dim,
              swa_pos_dim * sizeof(uint16_t));
  std::memcpy(generation_swa_position_ids_sin,
              swa_position_ids_sin + position * swa_pos_dim,
              swa_pos_dim * sizeof(uint16_t));
}

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
  int swa_pos_dim, int position, int rope_cache_seq_len) {
  generation_sample[0] = current_token;
  fill_generation_inputs_common(
    generation_attention_mask, generation_attention_mask_elements,
    generation_sliding_attention_mask,
    generation_sliding_attention_mask_elements, generation_full_kv_past_length,
    generation_sliding_kv_past_length, generation_position_ids_cos,
    generation_position_ids_sin, position_ids_cos, position_ids_sin, pos_dim,
    generation_swa_position_ids_cos, generation_swa_position_ids_sin,
    swa_position_ids_cos, swa_position_ids_sin, swa_pos_dim, position,
    rope_cache_seq_len);
}

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
  int swa_pos_dim, int position, int rope_cache_seq_len) {
  fill_generation_inputs_common(
    generation_attention_mask, generation_attention_mask_elements,
    generation_sliding_attention_mask,
    generation_sliding_attention_mask_elements, generation_full_kv_past_length,
    generation_sliding_kv_past_length, generation_position_ids_cos,
    generation_position_ids_sin, position_ids_cos, position_ids_sin, pos_dim,
    generation_swa_position_ids_cos, generation_swa_position_ids_sin,
    swa_position_ids_cos, swa_position_ids_sin, swa_pos_dim, position,
    rope_cache_seq_len);
}

int sample(uint16_t *pointer, int length, int *tokens, int number_of_tokens,
           float logit_scale, int logit_offset, float repetition_penalty,
           float temperature, float top_p, int top_k,
           float final_logit_softcapping) {
  // top_k == 1 (greedy) short circuit. With a single candidate the rest of
  // this function cannot change the answer: dequant, soft-cap, repetition
  // penalty, temperature and top-p are all monotone or point-mass operations
  // on a one-element set, and the draw over {1.0} always yields index 0. So the
  // result is exactly the argmax, and the general path's tie-break (a strict
  // `<` against the running maximum, i.e. the *first* maximal logit wins) is
  // reproduced here exactly.
  //
  // Worth short-circuiting because the general path walks 262 144 uint16
  // logits through a std::priority_queue - a dependent branch per element
  // that runs at ~1.8 GB/s. Reducing over blocks lets the compiler emit a
  // vector max and only rescan the block that actually raised the maximum.
  if (top_k == 1 && length > 0 && host_fastpath_enabled()) {
    constexpr int kBlock = 512;
    uint16_t best = 0;
    int best_block = -1;
    for (int base = 0; base < length; base += kBlock) {
      const int end = std::min(base + kBlock, length);
      // Eight independent accumulators. A single running max is a
      // loop-carried dependency that measures *slower* than the priority
      // queue it replaces (129 us vs 77 us per 262 144 logits on x86); split
      // eight ways it vectorises to a umax chain and lands at 28 us.
      uint16_t m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0, m5 = 0, m6 = 0, m7 = 0;
      int i = base;
      for (; i + 8 <= end; i += 8) {
        m0 = std::max(m0, pointer[i + 0]);
        m1 = std::max(m1, pointer[i + 1]);
        m2 = std::max(m2, pointer[i + 2]);
        m3 = std::max(m3, pointer[i + 3]);
        m4 = std::max(m4, pointer[i + 4]);
        m5 = std::max(m5, pointer[i + 5]);
        m6 = std::max(m6, pointer[i + 6]);
        m7 = std::max(m7, pointer[i + 7]);
      }
      uint16_t block_max =
        std::max(std::max(std::max(m0, m1), std::max(m2, m3)),
                 std::max(std::max(m4, m5), std::max(m6, m7)));
      for (; i < end; ++i)
        block_max = std::max(block_max, pointer[i]);
      // Strictly-greater keeps the FIRST block that attains the maximum,
      // matching the priority queue's `<` replacement test.
      if (best_block < 0 || block_max > best) {
        best = block_max;
        best_block = base;
      }
    }
    (void)rng(); // Keep the generic sampler's one RNG step, even for top_k=1.
    const int end = std::min(best_block + kBlock, length);
    for (int i = best_block; i < end; ++i)
      if (pointer[i] == best)
        return i;
    return best_block; // unreachable: `best` was taken from this block
  }

  // Priority queue!
  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>,
                      std::greater<std::pair<int, int>>>
    top_k_elements;
  for (int i = 0; i < top_k && i < length; i++) {
    top_k_elements.push(std::make_pair(pointer[i], i));
  }
  for (int i = top_k; i < length; i++) {
    if (top_k_elements.top().first < pointer[i]) {
      top_k_elements.pop();
      top_k_elements.push(std::make_pair(pointer[i], i));
    }
  }
  length = top_k_elements.size();

  // Convert to float, dequant, then apply Gemma final-logit soft-cap.
  // Soft-cap: l = soft_cap * tanh(l / soft_cap). Without it, a few raw
  // logits dominate softmax and the model collapses into repetition.
  std::vector<int> indices(length);
  std::vector<float> logits(length);
  const bool use_softcap = final_logit_softcapping > 0.0f;
  const float inv_softcap = use_softcap ? 1.0f / final_logit_softcapping : 0.0f;
  for (int i = 0; i < length; i++) {
    auto element = top_k_elements.top();
    float l = (1.0f * element.first + logit_offset) * logit_scale;
    if (use_softcap) {
      l = final_logit_softcapping * std::tanh(l * inv_softcap);
    }
    logits[i] = l;
    indices[i] = element.second;
    top_k_elements.pop();
  }

  for (unsigned int i = 0; i < number_of_tokens; ++i) {
    const int t = tokens[i];
    for (int j = 0; j < length; ++j) {
      if (indices[j] == tokens[i]) {
        if (logits[j] > 0.0f)
          logits[j] /= repetition_penalty;
        else
          logits[j] *= repetition_penalty;
        break;
      }
    }
  }

  std::vector<std::pair<int, float>> top_indices_and_logits(length);
  for (int i = 0; i < length; ++i) {
    if (temperature > 1e-5)
      logits[i] = logits[i] / temperature;
    top_indices_and_logits[i] = {i, logits[i]};
  }
  // Higher logit first, lower token id on an exact tie: the order (and so the
  // top-p cut and the draw) is fully determined.
  sort(top_indices_and_logits.begin(), top_indices_and_logits.end(),
       [&indices](auto &a, auto &b) {
         if (a.second != b.second)
           return a.second > b.second;
         return indices[a.first] < indices[b.first];
       });

  const float max_logit = top_indices_and_logits[0].second;
  std::vector<float> probs(length);
  float sum_exp = 0.0f;
  for (int i = 0; i < length; ++i) {
    probs[i] = std::exp(top_indices_and_logits[i].second - max_logit);
    sum_exp += probs[i];
  }
  if (sum_exp <= 0.0f)
    sum_exp = 1.0f;
  for (int i = 0; i < length; ++i) {
    probs[i] /= sum_exp;
  }

  float cum_prob = 0.0f;
  unsigned int top_index = 0;
  while (top_index < (unsigned)length && cum_prob < top_p) {
    cum_prob += probs[top_index];
    ++top_index;
  }
  if (top_index == 0)
    top_index = 1;

  // Apply Top-P: nuke beyond top_index
  for (int i = 0; i < length; ++i)
    logits[i] = -INFINITY;
  for (unsigned int i = 0; i < top_index; ++i) {
    logits[top_indices_and_logits[i].first] = top_indices_and_logits[i].second;
  }

  // Final softmax for sampling
  const float final_max = top_indices_and_logits[0].second;
  float final_sum_exp = 0.0f;
  for (int i = 0; i < length; ++i) {
    float ex = std::exp(logits[i] - final_max);
    final_sum_exp += ex;
    logits[i] = ex;
  }
  if (final_sum_exp <= 0.0f)
    final_sum_exp = 1.0f;
  for (int i = 0; i < length; ++i)
    logits[i] /= final_sum_exp;

  // Sample: inverse CDF in double over the kept weights (token_sampler.h),
  // identical on every C++ runtime -- std::discrete_distribution is not.
  std::vector<double> weights(logits.begin(), logits.begin() + length);
  return indices[causallm::sampleIndex(weights.data(), weights.size(), rng)];
}

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   cpu_attention.cpp
 * @date   30 October 2026
 * @brief  CpuComputeOps::attention, the host per-row attention arm
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * @details The per-row arm of causallm::MHACoreLayer, moved here so the
 * layer reaches it through ComputeOps (#4451): scores row by row
 * (compute_kcaches), a softmax over each row's visible keys
 * (softmax_triangle, with the softcap and the optional sink), and the
 * weighted sum of the values (compute_fp16vcache_transposed). The bodies are
 * the layer's, unchanged; the layer's members they read are the fields of
 * RowwiseAttention below.
 */

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <cpu_backend.h>
#include <cpu_ops_table.h>
#include <fp16.h>
#include <nntrainer_error.h>
#include <tensor.h>
#include <thread_manager.h>

namespace nntrainer {

namespace {

#ifdef ENABLE_FP16
// Decode/per-row K-cache dot product with int8-quantized K + per-(row, head)
// fp16 scale. Mirrors the layout of nntrainer::compute_kcaches<__fp16> but
// reads the cache as int8 bytes and scales the dot product by the matching
// scale value before the 1/sqrt(d) inverse. Output layout matches the fp16
// helper: out[(row-start)*num_cache_head*gqa + n*gqa + g].
static inline void compute_kcaches_int8_fp16(
  const _FP16 *in,         // [num_rows? actually only the latest row is used]
  const int8_t *kcache_i8, // [num_rows, num_cache_head, head_dim]
  const uint16_t *kscale,  // [num_rows, num_cache_head] fp16 bits
  _FP16 *output,           // [row_cnt, num_cache_head * gqa_size]
  int num_rows, int num_cache_head, int head_dim, int gqa_size,
  size_t local_window_size, int head_start, int head_end) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  // Note: local_window_size is size_t. Comparing num_rows (int) directly
  // promotes to size_t so UINT_MAX-as-no-window works correctly. Casting
  // local_window_size to int first would wrap UINT_MAX to -1 and skip the
  // entire row loop.
  const int start_row = (size_t)num_rows < local_window_size
                          ? 0
                          : num_rows - (int)local_window_size;
  const int row_cnt =
    (size_t)num_rows < local_window_size ? num_rows : (int)local_window_size;
  const float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      const _FP16 *in_ptr = in + (n * gqa_size + g) * head_dim;
      for (int t = 0; t < row_cnt; ++t) {
        const int row = start_row + t;
        const int8_t *k_row =
          kcache_i8 + ((size_t)row * num_cache_head + n) * head_dim;
        const float s =
          nntrainer::compute_fp16_to_fp32(kscale[row * num_cache_head + n]);
        // Dot in fp32 keeps precision; int8 * fp16-as-fp32 -> fp32 acc.
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          sum += nntrainer::compute_fp16_to_fp32(
                   *reinterpret_cast<const uint16_t *>(&in_ptr[d])) *
                 (float)k_row[d];
        }
        const float v = sum * s * inv_sqrt_d;
        const size_t out_idx =
          (size_t)(row - start_row) * num_cache_head * gqa_size +
          (size_t)n * gqa_size + g;
        output[out_idx] = static_cast<_FP16>(v);
      }
    }
  }
}

// V cache dequant + transposed accumulation mirroring
// nntrainer::compute_fp16vcache_transposed but with int8 V + per-(row, head)
// fp16 scale. Output is fp16 [(num_cache_head*gqa_size), head_dim].
static inline void compute_fp16vcache_transposed_int8(
  int row_num, const _FP16 *in, const int8_t *vcache_i8, const uint16_t *vscale,
  _FP16 *output, int num_cache_head, int gqa_size, int head_dim,
  size_t local_window_size, int head_start, int head_end) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  // Promote to size_t to avoid wrapping UINT_MAX (no window) to -1.
  const int start_j = (size_t)row_num < local_window_size
                        ? 0
                        : row_num + 1 - (int)local_window_size;
  const int in_row_off = (size_t)row_num < local_window_size
                           ? 0
                           : row_num + 1 - (int)local_window_size;

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int h = 0; h < gqa_size; ++h) {
      float acc[256]; // head_dim <= 256 for our target models (Qwen3/Gemma3)
      NNTR_THROW_IF(head_dim > 256, std::invalid_argument)
        << "head_dim (" << head_dim << ") exceeds the int8 V-read scratch size";
      for (int d = 0; d < head_dim; ++d)
        acc[d] = 0.0f;

      for (int j = start_j; j <= row_num; ++j) {
        const int8_t *vptr =
          vcache_i8 + ((size_t)j * num_cache_head + n) * head_dim;
        const float vs =
          nntrainer::compute_fp16_to_fp32(vscale[j * num_cache_head + n]);
        const int attn_idx =
          (j - in_row_off) * gqa_size * num_cache_head + n * gqa_size + h;
        const float a_val = nntrainer::compute_fp16_to_fp32(
          *reinterpret_cast<const uint16_t *>(&in[attn_idx]));
        const float a_scaled = a_val * vs;
        for (int d = 0; d < head_dim; ++d) {
          acc[d] += a_scaled * (float)vptr[d];
        }
      }

      _FP16 *out_row = output + (n * gqa_size + h) * head_dim;
      for (int d = 0; d < head_dim; ++d)
        out_row[d] = static_cast<_FP16>(acc[d]);
    }
  }
}
#endif

static void compute_kcaches_fp32_reference(
  const float *in, const float *kcache, float *output, int num_rows,
  int num_cache_head, int head_dim, int gqa_size, size_t local_window_size,
  int head_start = 0, int head_end = -1) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  const int window = static_cast<int>(
    std::min(static_cast<size_t>(num_rows), local_window_size));
  const int start_row = num_rows - window;
  const float inv_sqrt_head_dim =
    1.0f / std::sqrt(static_cast<float>(head_dim));

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      const float *query = in + (n * gqa_size + g) * head_dim;
      for (int row = start_row; row < num_rows; ++row) {
        const float *key = kcache + (row * num_cache_head + n) * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          sum += query[d] * key[d];
        }
        output[(row - start_row) * num_cache_head * gqa_size + n * gqa_size +
               g] = sum * inv_sqrt_head_dim;
      }
    }
  }
}

static void compute_vcache_fp32_transposed_reference(
  int row_num, const float *in, const float *vcache, float *output,
  int num_cache_head, int gqa_size, int head_dim, size_t local_window_size,
  int head_start = 0, int head_end = -1) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  const int window = static_cast<int>(
    std::min(static_cast<size_t>(row_num + 1), local_window_size));
  const int start_row = row_num + 1 - window;

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int h = 0; h < gqa_size; ++h) {
      float *out = output + (n * gqa_size + h) * head_dim;
      std::fill(out, out + head_dim, 0.0f);

      for (int row = start_row; row <= row_num; ++row) {
        const int attn_row = row - start_row;
        const float a_val =
          in[attn_row * (num_cache_head * gqa_size) + n * gqa_size + h];
        const float *value = vcache + (row * num_cache_head + n) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          out[d] += a_val * value[d];
        }
      }
    }
  }
}

/**
 * @brief The layer state the per-row arm reads, as one call's parameters.
 */
struct RowwiseAttention {
  bool is_causal;
  size_t local_window_size;
  float attn_logit_softcapping;
  bool
    kv_int8; /**< UINT8 planes: int8 K / V with an fp16 scale per row, head */
  const uint16_t *cur_kv_int8_key_scale_batch;
  const uint16_t *cur_kv_int8_value_scale_batch;
  static constexpr int tile_size = 4;

  /**
   * @brief Number of scores before absolute query row i: sum over the rows
   *        before it of min(row + 1, local_window_size).
   */
  size_t calc_windowed_attn_index(size_t i) const {
    if (i <= local_window_size) {
      return (i * (i + 1)) / 2;
    } else {
      return (local_window_size * (local_window_size + 1)) / 2 +
             (i - local_window_size) * local_window_size;
    }
  }

  void compute_kcaches(const nntrainer::Tensor &in,
                       const nntrainer::Tensor &cache, nntrainer::Tensor &out,
                       unsigned int from, size_t sequence_len,
                       unsigned int num_head, unsigned int group_size,
                       unsigned int head_dim);
  void softmax_triangle(nntrainer::Tensor &qk_out, size_t row, size_t num_head,
                        unsigned int from);
  void softmax_triangle(nntrainer::Tensor &qk_out, size_t row, size_t num_head,
                        unsigned int from, nntrainer::Tensor &sink_step);
  void compute_fp16vcache_transposed(const nntrainer::Tensor &in,
                                     const nntrainer::Tensor &vcache,
                                     nntrainer::Tensor &output, int from,
                                     int num_cache_head, int gqa_size,
                                     int head_dim, int to);
};

/**
 * @brief Function to compute Attention Scores using Tensor inputs. Wrapper
 * around nntrainer::compute_kcaches with multi-threading support
 *
 * Expected Input Shapes:
 * @param in (Query): [Batch, 1, sequence_len, Num_Heads_Q * Head_Dim]
 * @param cache (Key Cache): [Batch, 1, Max_Timestep, Num_Heads_KV * Head_Dim]
 * @param out (Attention Score): [Batch, 1, 1, Num_Heads_Q * Context_Len]
 *            where Context_Len is usually the current timestep 'to'.
 *
 */
void RowwiseAttention::compute_kcaches(
  const nntrainer::Tensor &in, const nntrainer::Tensor &cache,
  nntrainer::Tensor &out, unsigned int from, size_t sequence_len,
  unsigned int num_head, unsigned int group_size, unsigned int head_dim) {

  // Dispatch based on data type (FP32 or FP16)
  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (sequence_len == 1) {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_to_compute = is_causal ? from + 1 : from + sequence_len;
      unsigned int num_cache_head = num_head / group_size;

      // Use ThreadManager for lower overhead parallelization during decoding
      const float *in_data = in.getData<float>();
      float *out_data = out.getData<float>();

      auto &tm = nntrainer::ThreadManager::Global();
      if (cache.getDataType() == ml::train::TensorDim::DataType::FP32) {
        const float *cache_data = cache.getData<float>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            compute_kcaches_fp32_reference(
              in_data, cache_data, out_data, row_to_compute, num_cache_head,
              head_dim, group_size, local_window_size, head_kv, head_kv + 1);
          });
      } else {
        const uint16_t *cache_data = cache.getData<uint16_t>();
        tm.parallel_for(0, static_cast<size_t>(num_cache_head),
                        [=](size_t head_kv) {
                          nntrainer::compute_kcaches<uint16_t>(
                            in_data, cache_data, out_data, row_to_compute,
                            num_cache_head, head_dim, group_size, tile_size,
                            local_window_size, head_kv, head_kv + 1);
                        });
      }

    } else {
      // Sequence processing (prefill or chunked)
      // Iterate over ALL query rows so that no row is skipped even when
      // sequence_len > local_window_size.
      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(sequence_len), [=](size_t i) {
        float *input_addr =
          in.getData<float>() + (size_t)num_head * head_dim * i;
        int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
        // Windowed cumulative offset so that each row's scores are placed
        // contiguously after the previous row's scores (respecting the window).
        size_t out_start_row = is_causal ? calc_windowed_attn_index(from + i) -
                                             calc_windowed_attn_index(from)
                                         : i * (from + sequence_len);
        float *output_addr = out.getData<float>() + out_start_row * num_head;

        if (cache.getDataType() == ml::train::TensorDim::DataType::FP32) {
          float *cache_addr = cache.getData<float>();
          compute_kcaches_fp32_reference(
            input_addr, cache_addr, output_addr, row_to_compute,
            num_head / group_size, head_dim, group_size, local_window_size);
        } else {
          uint16_t *cache_addr = cache.getData<uint16_t>();
          nntrainer::compute_kcaches<uint16_t>(
            input_addr, cache_addr, output_addr, row_to_compute,
            num_head / group_size, head_dim, group_size, tile_size,
            local_window_size);
        }
      });
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (sequence_len == 1) {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int num_rows = is_causal ? from + 1 : from + sequence_len;
      unsigned int num_cache_head = num_head / group_size;

      // Use ThreadManager for lower overhead parallelization during decoding
      const _FP16 *in_data = in.getData<_FP16>();
      _FP16 *out_data = out.getData<_FP16>();

      auto &tm = nntrainer::ThreadManager::Global();
      if (kv_int8) {
        // int8 K cache + per-(row, head) fp16 scale.
        const int8_t *cache_i8 =
          reinterpret_cast<const int8_t *>(cache.getData<uint8_t>());
        const uint16_t *kscale = cur_kv_int8_key_scale_batch;
        NNTR_THROW_IF(kscale == nullptr, std::invalid_argument)
          << "kv_int8 read path missing per-batch K-scale pointer";
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            compute_kcaches_int8_fp16(in_data, cache_i8, kscale, out_data,
                                      num_rows, num_cache_head, head_dim,
                                      group_size, local_window_size,
                                      (int)head_kv, (int)head_kv + 1);
          });
      } else {
        const _FP16 *cache_data = cache.getData<_FP16>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            nntrainer::compute_kcaches(
              in_data, cache_data, out_data, num_rows, num_cache_head, head_dim,
              group_size, tile_size, local_window_size, head_kv, head_kv + 1);
          });
      }
    } else {
      // Iterate over ALL query rows so that no row is skipped even when
      // sequence_len > local_window_size.
      auto &tm = nntrainer::ThreadManager::Global();
      if (kv_int8) {
        const int8_t *cache_i8 =
          reinterpret_cast<const int8_t *>(cache.getData<uint8_t>());
        const uint16_t *kscale = cur_kv_int8_key_scale_batch;
        const unsigned int num_cache_head = num_head / group_size;
        NNTR_THROW_IF(kscale == nullptr, std::invalid_argument)
          << "kv_int8 read path (prefill) missing per-batch K-scale pointer";
        // Iterate over ALL query rows + windowed cumulative offset (PR #3989).
        tm.parallel_for(0, static_cast<size_t>(sequence_len), [=](size_t i) {
          _FP16 *input_addr =
            in.getData<_FP16>() + (size_t)num_head * head_dim * i;
          int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
          size_t out_start_row = is_causal
                                   ? calc_windowed_attn_index(from + i) -
                                       calc_windowed_attn_index(from)
                                   : i * (from + sequence_len);
          _FP16 *output_addr = out.getData<_FP16>() + out_start_row * num_head;
          compute_kcaches_int8_fp16(input_addr, cache_i8, kscale, output_addr,
                                    row_to_compute, num_cache_head, head_dim,
                                    group_size, local_window_size, 0, -1);
        });
        return;
      }
      tm.parallel_for(0, static_cast<size_t>(sequence_len), [=](size_t i) {
        _FP16 *input_addr =
          in.getData<_FP16>() + (size_t)num_head * head_dim * i;
        _FP16 *cache_addr = cache.getData<_FP16>();
        int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
        // Windowed cumulative offset so that each row's scores are placed
        // contiguously after the previous row's scores (respecting the window).
        size_t out_start_row = is_causal ? calc_windowed_attn_index(from + i) -
                                             calc_windowed_attn_index(from)
                                         : i * (from + sequence_len);

        _FP16 *output_addr = out.getData<_FP16>() + out_start_row * num_head;

        nntrainer::compute_kcaches(input_addr, cache_addr, output_addr,
                                   row_to_compute, num_head / group_size,
                                   head_dim, group_size, tile_size,
                                   local_window_size);
      });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void RowwiseAttention::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
                                        size_t num_head, unsigned int from) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] =
          std::tanh(qk_out_[i] * inv_softcapping) * attn_logit_softcapping;
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      // Iterate over ALL rows (not just min(row, window)) so that every query
      // row in a long prefill gets softmaxed over the correct windowed range.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head);
      });
    }
  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] = (_FP16)(std::tanh((float)qk_out_[i] * inv_softcapping) *
                             attn_logit_softcapping);
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      // Iterate over ALL rows (not just min(row, window)) so that every query
      // row in a long prefill gets softmaxed over the correct windowed range.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
      });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void RowwiseAttention::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
                                        size_t num_head, unsigned int from,
                                        nntrainer::Tensor &sink_step) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] =
          std::tanh(qk_out_[i] * inv_softcapping) * attn_logit_softcapping;
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        unsigned int to = from + row;
        end_row = to;
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head,
                                     sink_step.getData());
    } else {
      // Iterate over ALL rows (not just min(row, window)) for correct windowed
      // prefill when sequence_len > local_window_size.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(i + from) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head,
                               sink_step.getData());
      });
    }
  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();
    _FP16 *sink_step_ = sink_step.getData<_FP16>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] = (_FP16)(std::tanh((float)qk_out_[i] * inv_softcapping) *
                             attn_logit_softcapping);
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head,
                                     sink_step_);
    } else {
      // Iterate over ALL rows (not just min(row, window)) for correct windowed
      // prefill when sequence_len > local_window_size.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(i + from) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head,
                               sink_step_);
      });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void RowwiseAttention::compute_fp16vcache_transposed(
  const nntrainer::Tensor &in, const nntrainer::Tensor &vcache,
  nntrainer::Tensor &output, int from, int num_cache_head, int gqa_size,
  int head_dim, int to) {

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if ((to - from) != 1) {
      // Iterate over ALL output rows so every query row gets an output even
      // when (to - from) > local_window_size.
      int total = to - from;
      if (!is_causal)
        total = to - from;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(total), [=](size_t i) {
        size_t start_idx;
        if (is_causal) {
          start_idx =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
        } else {
          start_idx = i * to; // linear index
        }
        const float *input =
          in.getData<float>() + start_idx * num_cache_head * gqa_size;
        float *out =
          output.getData<float>() + i * (num_cache_head * gqa_size * head_dim);

        int row_num = is_causal ? (from + (int)i) : to - 1;
        if (vcache.getDataType() == ml::train::TensorDim::DataType::FP32) {
          compute_vcache_fp32_transposed_reference(
            row_num, input, vcache.getData<float>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        } else {
          nntrainer::compute_fp16vcache_fp32_transposed(
            row_num, input, vcache.getData<uint16_t>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        }
      });
    } else {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_num = to - 1;

      // Use OpenMP for lower overhead parallelization during decoding
      const float *in_data = in.getData<float>();
      float *output_data = output.getData<float>();

      auto &tm = nntrainer::ThreadManager::Global();
      if (vcache.getDataType() == ml::train::TensorDim::DataType::FP32) {
        const float *vcache_data = vcache.getData<float>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            compute_vcache_fp32_transposed_reference(
              row_num, in_data, vcache_data, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
          });
      } else {
        const uint16_t *vcache_data = vcache.getData<uint16_t>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            nntrainer::compute_fp16vcache_fp32_transposed(
              row_num, in_data, vcache_data, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
          });
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if ((to - from) != 1) {
      // Iterate over ALL output rows so every query row gets an output even
      // when (to - from) > local_window_size.
      int total = to - from;
      if (!is_causal)
        total = to - from;

      auto &tm = nntrainer::ThreadManager::Global();
      if (kv_int8) {
        const int8_t *vcache_i8 =
          reinterpret_cast<const int8_t *>(vcache.getData<uint8_t>());
        const uint16_t *vscale = cur_kv_int8_value_scale_batch;
        NNTR_THROW_IF(vscale == nullptr, std::invalid_argument)
          << "kv_int8 V read path missing per-batch V-scale pointer";
        // Iterate over ALL output rows + windowed cumulative offset (PR #3989).
        tm.parallel_for(0, static_cast<size_t>(total), [=](size_t i) {
          size_t start_idx;
          if (is_causal) {
            start_idx = calc_windowed_attn_index(from + i) -
                        calc_windowed_attn_index(from);
          } else {
            start_idx = i * to;
          }
          const _FP16 *input =
            in.getData<_FP16>() + start_idx * num_cache_head * gqa_size;
          _FP16 *out_p = output.getData<_FP16>() +
                         i * (num_cache_head * gqa_size * head_dim);
          int row_num = is_causal ? (from + (int)i) : to - 1;
          compute_fp16vcache_transposed_int8(
            row_num, input, vcache_i8, vscale, out_p, num_cache_head, gqa_size,
            head_dim, local_window_size, 0, -1);
        });
      } else {
        tm.parallel_for(0, static_cast<size_t>(total), [=](size_t i) {
          size_t start_idx;
          if (is_causal) {
            start_idx = calc_windowed_attn_index(from + i) -
                        calc_windowed_attn_index(from);
          } else {
            start_idx = i * to;
          }
          const _FP16 *input =
            in.getData<_FP16>() + start_idx * num_cache_head * gqa_size;
          _FP16 *out = output.getData<_FP16>() +
                       i * (num_cache_head * gqa_size * head_dim);
          int row_num = is_causal ? (from + (int)i) : to - 1;
          nntrainer::compute_fp16vcache_transposed(
            row_num, input, vcache.getData<_FP16>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        });
      }
    } else {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_num = to - 1;

      // Use OpenMP for lower overhead parallelization during decoding
      const _FP16 *in_data = in.getData<_FP16>();
      _FP16 *output_data = output.getData<_FP16>();

      auto &tm_fp16 = nntrainer::ThreadManager::Global();
      if (kv_int8) {
        const int8_t *vcache_i8 =
          reinterpret_cast<const int8_t *>(vcache.getData<uint8_t>());
        const uint16_t *vscale = cur_kv_int8_value_scale_batch;
        NNTR_THROW_IF(vscale == nullptr, std::invalid_argument)
          << "kv_int8 V read path (decode) missing per-batch V-scale pointer";
        tm_fp16.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            compute_fp16vcache_transposed_int8(
              row_num, in_data, vcache_i8, vscale, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, (int)head_kv,
              (int)head_kv + 1);
          });
      } else {
        const _FP16 *vcache_data = vcache.getData<_FP16>();
        tm_fp16.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            nntrainer::compute_fp16vcache_transposed(
              row_num, in_data, vcache_data, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
          });
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

} // namespace

void CpuComputeOps::attention(const Tensor &q, const KVCacheView &kv,
                              Tensor &out, const AttentionOpParams &p) {
  if (kv.key == nullptr || kv.value == nullptr)
    throw std::invalid_argument(
      "CpuComputeOps::attention: the cache needs both planes");
  if (kv.ring_cap != 0)
    throw std::invalid_argument(
      "CpuComputeOps::attention: the per-row arm cannot read a ringed cache");
  const unsigned int head_dim = kv.head_dim;
  const unsigned int rows = q.height();
  if (head_dim == 0 || p.num_heads_q == 0 ||
      q.width() != (size_t)p.num_heads_q * head_dim ||
      kv.key->width() % head_dim != 0 || kv.value->width() != kv.key->width())
    throw std::invalid_argument("CpuComputeOps::attention: q and the cache "
                                "must hold heads of head_dim");
  const unsigned int num_heads_kv = kv.key->width() / head_dim;
  if (num_heads_kv == 0 || p.num_heads_q % num_heads_kv != 0)
    throw std::invalid_argument("CpuComputeOps::attention: num_heads_q must "
                                "be a multiple of the cache's heads");
  if (rows == 0 || p.kv_len != p.q_pos + rows || out.size() != q.size() ||
      p.kv_len > kv.key->height() || p.kv_len > kv.value->height())
    throw std::invalid_argument(
      "CpuComputeOps::attention: kv_len must be q_pos + the query rows and "
      "fit the cache, and out must have q's size");
  // Both arms read an fp16-bits cache (FP16 or UINT16); the FP32 arm also an
  // FP32 one, the FP16 arm also an int8 (UINT8) one. out takes q's dtype.
  using DataType = ml::train::TensorDim::DataType;
  const DataType q_type = q.getDataType();
  const DataType kv_type = kv.key->getDataType();
  const bool fp16_bits =
    kv_type == DataType::FP16 || kv_type == DataType::UINT16;
  const bool kv_ok =
    (q_type == DataType::FP32 && (fp16_bits || kv_type == DataType::FP32)) ||
    (q_type == DataType::FP16 && (fp16_bits || kv_type == DataType::UINT8));
  if (!kv_ok || kv.value->getDataType() != kv_type ||
      out.getDataType() != q_type)
    throw std::invalid_argument(
      "CpuComputeOps::attention: an FP32 q reads an fp16-bits or FP32 cache, "
      "an FP16 q an fp16-bits or int8 one, and out takes q's dtype");
  // The sink overload reads num_heads_q values of q's dtype.
  if (p.sink != nullptr &&
      (p.sink->size() != p.num_heads_q || p.sink->getDataType() != q_type))
    throw std::invalid_argument("CpuComputeOps::attention: the sink must hold "
                                "num_heads_q values of q's dtype");

  RowwiseAttention r{p.causal,     p.window,
                     p.softcap,    kv_type == DataType::UINT8,
                     kv.key_scale, kv.value_scale};
  const unsigned int gqa_size = p.num_heads_q / num_heads_kv;
  const unsigned int from = p.q_pos;
  const unsigned int to = p.kv_len;
  Tensor scores(1, 1,
                p.causal ? r.calc_windowed_attn_index(to) -
                             r.calc_windowed_attn_index(from)
                         : (size_t)rows * to,
                p.num_heads_q, q.getTensorType());

  r.compute_kcaches(q, *kv.key, scores, from, rows, p.num_heads_q, gqa_size,
                    head_dim);
  if (p.sink != nullptr)
    r.softmax_triangle(scores, rows, p.num_heads_q, from, *p.sink);
  else
    r.softmax_triangle(scores, rows, p.num_heads_q, from);
  r.compute_fp16vcache_transposed(scores, *kv.value, out, from, num_heads_kv,
                                  gqa_size, head_dim, to);
}

} // namespace nntrainer

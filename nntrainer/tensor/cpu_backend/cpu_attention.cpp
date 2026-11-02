// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   cpu_attention.cpp
 * @date   30 October 2026
 * @brief  CpuComputeOps::attention: the host per-row and flash attention
 *         arms, and the choice between them
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
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cpu_backend.h>
#include <cpu_ops_table.h>
#include <env_compat.h>
#include <fp16.h>
#include <nntrainer_error.h>
#include <tensor.h>
#include <thread_manager.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#include <neon_impl.h> // custom_hgemm, exp_ps (neon_mathfun.h)
#endif
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
#include <immintrin.h>
#endif

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

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)

// Fused FP32 x FP16-bits -> FP32 GEMM for x86 (AVX2 + F16C). Equivalent of ARM
// shgemm but reads FP16-bits (uint16_t) directly without materializing an FP32
// copy of B — saves the temporary buffer and halves memory traffic compared to
// {convert+sgemm}. Row-major only, alpha applied, beta hard-coded to 0 to keep
// the kernel small (this is all the flash path needs).
//
// Two operand layouts:
//   TransB=true  (QK): C[m, n] = alpha * sum_k A[m,k] * fp16(B[n,k])
//                       B is N rows x K cols, row-major, ldb columns
//   TransB=false (AV): C[m, n] = alpha * sum_k A[m,k] * fp16(B[k,n])
//                       B is K rows x N cols, row-major, ldb columns
static inline void mha_hsgemm_avx2(unsigned int M, unsigned int N,
                                   unsigned int K, float alpha, const float *A,
                                   unsigned int lda, const uint16_t *B,
                                   unsigned int ldb, bool TransB, float *C,
                                   unsigned int ldc) {
  const __m256 valpha = _mm256_set1_ps(alpha);
  if (TransB) {
    // QK path. Block 4 m-rows so we amortize the B (K-row) conversion across 4
    // accumulators per inner k-step.
    unsigned int m = 0;
    for (; m + 4 <= M; m += 4) {
      const float *a0 = A + (size_t)(m + 0) * lda;
      const float *a1 = A + (size_t)(m + 1) * lda;
      const float *a2 = A + (size_t)(m + 2) * lda;
      const float *a3 = A + (size_t)(m + 3) * lda;
      for (unsigned int n = 0; n < N; ++n) {
        const uint16_t *b_row = B + (size_t)n * ldb;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        unsigned int k = 0;
        for (; k + 8 <= K; k += 8) {
          __m256 b =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b_row + k)));
          acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + k), b, acc0);
          acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + k), b, acc1);
          acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + k), b, acc2);
          acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + k), b, acc3);
        }
        // Horizontal-reduce 4 accumulators in parallel via two hadd-pairs.
        // acc0 = [s00 s01 s02 s03 | s04 s05 s06 s07] -> partial sums
        __m256 h01 = _mm256_hadd_ps(acc0, acc1);
        __m256 h23 = _mm256_hadd_ps(acc2, acc3);
        __m256 h = _mm256_hadd_ps(h01, h23);
        // h lanes: [s0_lo s1_lo s2_lo s3_lo | s0_hi s1_hi s2_hi s3_hi]
        __m128 lo = _mm256_castps256_ps128(h);
        __m128 hi = _mm256_extractf128_ps(h, 1);
        __m128 sums = _mm_add_ps(lo, hi); // [s0 s1 s2 s3]
        float s[4];
        _mm_storeu_ps(s, sums);
        // tail k
        for (; k < K; ++k) {
          const float bv = nntrainer::compute_fp16_to_fp32(b_row[k]);
          s[0] += a0[k] * bv;
          s[1] += a1[k] * bv;
          s[2] += a2[k] * bv;
          s[3] += a3[k] * bv;
        }
        C[(size_t)(m + 0) * ldc + n] = alpha * s[0];
        C[(size_t)(m + 1) * ldc + n] = alpha * s[1];
        C[(size_t)(m + 2) * ldc + n] = alpha * s[2];
        C[(size_t)(m + 3) * ldc + n] = alpha * s[3];
      }
    }
    // m tail (unblocked)
    for (; m < M; ++m) {
      const float *a_row = A + (size_t)m * lda;
      for (unsigned int n = 0; n < N; ++n) {
        const uint16_t *b_row = B + (size_t)n * ldb;
        __m256 acc = _mm256_setzero_ps();
        unsigned int k = 0;
        for (; k + 8 <= K; k += 8) {
          __m256 a = _mm256_loadu_ps(a_row + k);
          __m256 b =
            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b_row + k)));
          acc = _mm256_fmadd_ps(a, b, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float sum = _mm_cvtss_f32(s);
        for (; k < K; ++k)
          sum += a_row[k] * nntrainer::compute_fp16_to_fp32(b_row[k]);
        C[(size_t)m * ldc + n] = alpha * sum;
      }
    }
  } else {
    // AV path. Block n in 8-wide vector lanes; broadcast A[m,k] inside loop.
    for (unsigned int m = 0; m < M; ++m) {
      const float *a_row = A + (size_t)m * lda;
      float *c_row = C + (size_t)m * ldc;
      unsigned int n = 0;
      for (; n + 8 <= N; n += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (unsigned int k = 0; k < K; ++k) {
          __m256 a_b = _mm256_set1_ps(a_row[k]);
          __m256 b = _mm256_cvtph_ps(
            _mm_loadu_si128((const __m128i *)(B + (size_t)k * ldb + n)));
          acc = _mm256_fmadd_ps(a_b, b, acc);
        }
        _mm256_storeu_ps(c_row + n, _mm256_mul_ps(valpha, acc));
      }
      // n tail
      for (; n < N; ++n) {
        float sum = 0.0f;
        for (unsigned int k = 0; k < K; ++k)
          sum +=
            a_row[k] * nntrainer::compute_fp16_to_fp32(B[(size_t)k * ldb + n]);
        c_row[n] = alpha * sum;
      }
    }
  }
}

#endif // __x86_64__ || __i386__

/**
 * @brief The layer state the flash arm reads, as one call's parameters. The
 *        fields keep the names they had in MHACoreLayer, where the arm
 *        (gemm_attention) was moved from.
 */
struct FlashAttention {
  size_t num_heads_Q;
  size_t num_heads_KV;
  size_t head_dim;
  bool is_causal;
  size_t local_window_size;
  float attn_logit_softcapping;
  bool
    kv_int8; /**< UINT8 planes: int8 K / V with an fp16 scale per row, head */
  const uint16_t *cur_kv_int8_key_scale_batch;
  const uint16_t *cur_kv_int8_value_scale_batch;

  /**
   * @brief GEMM-based flash attention for one batch (covers both encoder
   *        non-causal and causal-LLM prefill paths).
   *        2-phase: (1) de-interleave Q (num_heads_Q heads) and K/V
   *        (num_heads_KV heads) into shared contiguous [H,N,d] buffers; (2)
   *        balanced parallel_for over (h_q, query_block) units with online
   *        softmax over key-blocks (shgemm QK -> NEON exp -> shgemm AV).
   *        GQA: h_kv = h_q / gqa_size. Causal: key-block upper-bound break
   *        + in-block boundary mask. Sliding window: key-block lower-bound
   *        skip + in-block lower mask.
   * @param[in] N_kv      total cache length (= cache_to, keys [0, N_kv))
   * @param[in] N_q       step length (= step_size, rows of the output)
   * @param[in] cache_from absolute starting position of queries in the cache
   *                      (so q_abs(i) = cache_from + i, k_abs(k) = k)
   */
  void run(const Tensor &query_step, const Tensor &b_cached_key,
           const Tensor &b_cached_value, Tensor &attention_output_step,
           unsigned int N_kv, unsigned int N_q, unsigned int cache_from);
};

void FlashAttention::run(const Tensor &query_step, const Tensor &b_cached_key,
                         const Tensor &b_cached_value,
                         Tensor &attention_output_step, unsigned int N_kv,
                         unsigned int N_q, unsigned int cache_from) {
  const unsigned int d = head_dim;
  const unsigned int HD_Q = num_heads_Q * d;
  const unsigned int HD_KV = num_heads_KV * d;
  const unsigned int gqa =
    (num_heads_KV > 0) ? static_cast<unsigned int>(num_heads_Q / num_heads_KV)
                       : 1u;
  const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(d));
  // Gemma2-style logit softcap, score = cap * tanh(score / cap), as
  // softmax_triangle applies it on the per-row arm. 0 = off.
  const float softcap = attn_logit_softcapping;
  const float inv_softcap = softcap > 0.0f ? 1.0f / softcap : 0.0f;
  const unsigned int order =
    static_cast<unsigned int>(query_step.getDim().getStorageOrder());
  const bool causal = is_causal;
  // Treat any local_window_size >= cache length as "no window".
  const bool windowed = (local_window_size < N_kv);
  const size_t W = static_cast<size_t>(local_window_size);

  // Runtime dtype dispatch: MHACoreLayer::forwarding() may convert Q/V/output
  // to FP16 (ENABLE_FP16 && __ANDROID__ build). K/V are always FP16 storage.
  const bool q_fp16 =
    (query_step.getDataType() == ml::train::TensorDim::DataType::FP16);
  const bool o_fp16 = (attention_output_step.getDataType() ==
                       ml::train::TensorDim::DataType::FP16);

  const float *Q = nullptr;
  const uint16_t *Q_fp16_src = nullptr;
  float *O = nullptr;
  uint16_t *O_fp16 = nullptr;
  if (q_fp16) {
#ifdef ENABLE_FP16
    Q_fp16_src =
      reinterpret_cast<const uint16_t *>(query_step.getData<_FP16>());
#endif
  } else {
    Q = query_step.getData<float>();
  }
  if (o_fp16) {
#ifdef ENABLE_FP16
    O_fp16 =
      reinterpret_cast<uint16_t *>(attention_output_step.getData<_FP16>());
#endif
  } else {
    O = attention_output_step.getData<float>();
  }

  // tile sizes (cache-resident S); overridable via env for tuning.
  // ARM: a 128-key block keeps the per-thread score and accumulator tiles
  // inside the core's L1/L2, and a 64-query block gives the causal prefill
  // more, smaller work units to balance across the threads. With the former
  // 256 x 512 tiles the host flash prefill was slower than the per-row path on
  // an aarch64 phone (1K prompt, 7 threads, Qwen3-0.6B QS4CX-FP32: 658 vs
  // 700 tok/s); with 64 x 128 it runs at 822 tok/s, and a dense 1.5B FP16
  // model goes from 263 to 293 tok/s.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
  unsigned int Bq = 256, Bk = 512;
#else
  unsigned int Bq = 64, Bk = 128;
#endif
  if (const char *e = std::getenv("VJEPA_BQ"))
    Bq = static_cast<unsigned int>(std::stoul(e));
  if (const char *e = std::getenv("VJEPA_BK"))
    Bk = static_cast<unsigned int>(std::stoul(e));

  const unsigned int num_qb = (N_q + Bq - 1) / Bq;
  auto &tm = nntrainer::ThreadManager::Global();

  // Cache always stores half-precision (FP16-bit) values; read as raw uint16_t
  // bits so we don't depend on ENABLE_FP16 / _FP16 / _Float16 being defined.
  // When kv_int8 is active the cache holds int8 bytes; the de-interleave
  // loop below dequantizes on the fly using cur_kv_int8_*_scale_batch.
  const uint16_t *Kbase = nullptr;
  const uint16_t *Vbase = nullptr;
#ifdef ENABLE_FP16
  const int8_t *Kbase_i8 = nullptr;
  const int8_t *Vbase_i8 = nullptr;
#endif
  if (kv_int8) {
#ifdef ENABLE_FP16
    Kbase_i8 =
      reinterpret_cast<const int8_t *>(b_cached_key.getData<uint8_t>());
    Vbase_i8 =
      reinterpret_cast<const int8_t *>(b_cached_value.getData<uint8_t>());
#endif
    NNTR_THROW_IF(cur_kv_int8_key_scale_batch == nullptr ||
                    cur_kv_int8_value_scale_batch == nullptr,
                  std::invalid_argument)
      << "kv_int8 flash attention missing per-batch scale pointers";
  } else {
#ifdef ENABLE_FP16
    Kbase = reinterpret_cast<const uint16_t *>(b_cached_key.getData<_FP16>());
    Vbase = reinterpret_cast<const uint16_t *>(b_cached_value.getData<_FP16>());
#else
    Kbase = b_cached_key.getData<uint16_t>();
    Vbase = b_cached_value.getData<uint16_t>();
#endif
  }

  // Phase 1: de-interleave heads once into shared contiguous buffers.
  // K/V always kept as raw FP16 bits (uint16). Q either FP32 (V-JEPA
  // path) or FP16 (when forwarding() pre-converts to FP16; ENABLE_FP16+
  // Android). The FP16 Q path keeps the entire attention in FP16
  // (custom_hgemm for QK and AV, FP16 softmax) without ever materializing
  // an FP32 score buffer.
  //
  // x86 guardrail (Gemma4): the ALL-FP16 NEON QK/AV branch is compiled out on
  // x86, and the FP32 fallback dereferences Qp_fp32. So when q_fp16 is set on
  // x86 we must materialize an FP32 de-interleaved Q (Qa_fp32) and take the
  // FP32 path; otherwise Qp_fp32==nullptr -> SIGSEGV (the crash the diagnosis
  // hit when use_gemm_attention was flipped and a d=512 layer fell to here).
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
  const bool q_fp16_to_fp32 = q_fp16; // x86 has no NEON FP16-Q kernel
#else
  const bool q_fp16_to_fp32 = false;
#endif
  std::vector<float> Qa_fp32;
  std::vector<uint16_t> Qa_fp16;
  if (q_fp16 && !q_fp16_to_fp32)
    Qa_fp16.resize((size_t)num_heads_Q * N_q * d);
  else
    Qa_fp32.resize((size_t)num_heads_Q * N_q * d);
  std::vector<uint16_t> Ka((size_t)num_heads_KV * N_kv * d);
  std::vector<uint16_t> Va((size_t)num_heads_KV * N_kv * d);
  {
    if (q_fp16_to_fp32) {
#ifdef ENABLE_FP16
      // FP16 bits -> FP32, de-interleaved per head (x86 FP32 attention path).
      tm.parallel_for(0, static_cast<size_t>(num_heads_Q), [&](size_t h) {
        float *qa = Qa_fp32.data() + (size_t)h * N_q * d;
        const uint16_t *qh = Q_fp16_src + h * d;
        for (unsigned int n = 0; n < N_q; ++n) {
          const uint16_t *qsrc = qh + (size_t)n * HD_Q;
          float *qdst = qa + (size_t)n * d;
          for (unsigned int x = 0; x < d; ++x)
            qdst[x] = nntrainer::compute_fp16_to_fp32(qsrc[x]);
        }
      });
#endif
    } else if (q_fp16) {
      tm.parallel_for(0, static_cast<size_t>(num_heads_Q), [&](size_t h) {
        uint16_t *qa = Qa_fp16.data() + (size_t)h * N_q * d;
        const uint16_t *qh = Q_fp16_src + h * d;
        for (unsigned int n = 0; n < N_q; ++n)
          std::memcpy(qa + (size_t)n * d, qh + (size_t)n * HD_Q,
                      d * sizeof(uint16_t));
      });
    } else {
      tm.parallel_for(0, static_cast<size_t>(num_heads_Q), [&](size_t h) {
        float *qa = Qa_fp32.data() + (size_t)h * N_q * d;
        const float *qh = Q + h * d;
        for (unsigned int n = 0; n < N_q; ++n)
          std::memcpy(qa + (size_t)n * d, qh + (size_t)n * HD_Q,
                      d * sizeof(float));
      });
    }
    if (kv_int8) {
#ifdef ENABLE_FP16
      // Dequant on the fly: K/V int8[n, hkv, :] * scale[n, hkv] -> fp16
      // bits stored in the de-interleaved Ka/Va buffers, so downstream
      // Phase 2 sees the same fp16 layout as the regular path.
      const uint16_t *Kscale = cur_kv_int8_key_scale_batch;
      const uint16_t *Vscale = cur_kv_int8_value_scale_batch;
      tm.parallel_for(0, static_cast<size_t>(num_heads_KV), [&](size_t hkv) {
        uint16_t *ka = Ka.data() + (size_t)hkv * N_kv * d;
        uint16_t *va = Va.data() + (size_t)hkv * N_kv * d;
        const int8_t *kh = Kbase_i8 + hkv * d;
        const int8_t *vh = Vbase_i8 + hkv * d;
        for (unsigned int n = 0; n < N_kv; ++n) {
          const float ks = nntrainer::compute_fp16_to_fp32(
            Kscale[(size_t)n * num_heads_KV + hkv]);
          const float vs = nntrainer::compute_fp16_to_fp32(
            Vscale[(size_t)n * num_heads_KV + hkv]);
          const int8_t *k_row = kh + (size_t)n * HD_KV;
          const int8_t *v_row = vh + (size_t)n * HD_KV;
          uint16_t *ka_row = ka + (size_t)n * d;
          uint16_t *va_row = va + (size_t)n * d;
          for (unsigned int x = 0; x < d; ++x) {
            ka_row[x] = nntrainer::compute_fp32_to_fp16((float)k_row[x] * ks);
            va_row[x] = nntrainer::compute_fp32_to_fp16((float)v_row[x] * vs);
          }
        }
      });
#else
      NNTR_THROW_IF(true, std::invalid_argument)
        << "kv_int8 flash attention requires ENABLE_FP16";
#endif
    } else {
      tm.parallel_for(0, static_cast<size_t>(num_heads_KV), [&](size_t hkv) {
        uint16_t *ka = Ka.data() + (size_t)hkv * N_kv * d;
        uint16_t *va = Va.data() + (size_t)hkv * N_kv * d;
        const uint16_t *kh = Kbase + hkv * d;
        const uint16_t *vh = Vbase + hkv * d;
        for (unsigned int n = 0; n < N_kv; ++n) {
          std::memcpy(ka + (size_t)n * d, kh + (size_t)n * HD_KV,
                      d * sizeof(uint16_t));
          std::memcpy(va + (size_t)n * d, vh + (size_t)n * HD_KV,
                      d * sizeof(uint16_t));
        }
      });
    }
  }

  // Phase 2: flash attention over balanced (h_q, query-block) work units.
  tm.parallel_for(0, static_cast<size_t>(num_heads_Q) * num_qb, [&](size_t u) {
    const unsigned int h_q = static_cast<unsigned int>(u / num_qb);
    const unsigned int h_kv = h_q / gqa;
    const unsigned int qb = static_cast<unsigned int>(u % num_qb) * Bq;
    const unsigned int bq = std::min(Bq, N_q - qb);
    const float *Qp_fp32 = (q_fp16 && !q_fp16_to_fp32)
                             ? nullptr
                             : (Qa_fp32.data() + (size_t)h_q * N_q * d);
    // Read by the all-FP16 NEON path only.
    [[maybe_unused]] const uint16_t *Qp_fp16 =
      (q_fp16 && !q_fp16_to_fp32) ? (Qa_fp16.data() + (size_t)h_q * N_q * d)
                                  : nullptr;
    const uint16_t *Kp = Ka.data() + (size_t)h_kv * N_kv * d;
    const uint16_t *Vp = Va.data() + (size_t)h_kv * N_kv * d;
    float *Oh = o_fp16 ? nullptr : (O + h_q * d);
    uint16_t *Oh_fp16 = o_fp16 ? (O_fp16 + h_q * d) : nullptr;

    thread_local std::vector<float> S, Pacc, Ol, mrow, lrow;
    thread_local std::vector<uint16_t> Sp16, Pacc16;
    S.resize((size_t)Bq * Bk);
    Pacc.resize((size_t)Bq * d);
    Ol.resize((size_t)Bq * d);
    mrow.resize(Bq);
    lrow.resize(Bq);
#if !defined(__x86_64__) && !defined(__i386__) && defined(__ARM_NEON) &&       \
  defined(ENABLE_FP16)
    Sp16.resize((size_t)Bq * Bk);
    Pacc16.resize((size_t)Bq * d);
#endif
    // FP16-throughout path uses Sp16 for both QK output (custom_hgemm,
    // FP16-stored) and AV input (softmax in-place updates the same
    // buffer). The FP32 S buffer is unused in that path.

    std::fill(Ol.begin(), Ol.begin() + (size_t)bq * d, 0.0f);
    for (unsigned int i = 0; i < bq; ++i) {
      mrow[i] = -3.0e38f;
      lrow[i] = 0.0f;
    }

    // The absolute query positions in this work unit are
    // [cache_from + qb, cache_from + qb + bq).
    const size_t q_abs_lo = (size_t)cache_from + qb;
    const size_t q_abs_hi = q_abs_lo + bq - 1; // inclusive

    for (unsigned int kb = 0; kb < N_kv; kb += Bk) {
      const unsigned int bk = std::min(Bk, N_kv - kb);

      // Causal upper-bound block-skip: smallest k_abs in block > largest
      // q_abs -> this and all later key blocks contribute nothing.
      if (causal && (size_t)kb > q_abs_hi)
        break;

      // Sliding-window lower-bound block-skip: largest k_abs in block <
      // smallest visible threshold (q_abs_lo - W + 1, i.e., k_abs must
      // satisfy k_abs > q_abs - W).
      if (windowed && (size_t)kb + bk + W <= q_abs_lo + 1)
        continue;

      // Does this block straddle the causal diagonal for any row?
      const bool causal_boundary = causal && ((size_t)kb + bk > q_abs_lo + 1);
      // Does this block straddle the sliding-window lower bound for any row?
      const bool window_boundary = windowed && ((size_t)kb + W < q_abs_hi + 1);

#if !defined(__x86_64__) && !defined(__i386__) && defined(__ARM_NEON) &&       \
  defined(ENABLE_FP16)
      if (q_fp16) {
        // ALL-FP16 PATH (matches pre-V-JEPA mha_core precision: FP16
        // storage with FP32 partial accumulation inside NEON kernels,
        // no upgrade to FP32 in the score buffer).
        // QK: FP16 × FP16 -> FP16 (custom_hgemm with FP32 partial acc).
        nntrainer::neon::custom_hgemm(
          reinterpret_cast<const __fp16 *>(Qp_fp16 + (size_t)qb * d),
          reinterpret_cast<const __fp16 *>(Kp + (size_t)kb * d),
          reinterpret_cast<__fp16 *>(Sp16.data()), bq, bk, d, inv_sqrt, 0.0f,
          /*TransA=*/false, /*TransB=*/true);

        // Boundary masking in FP16: write -INFINITY as bit pattern 0xFC00.
        for (unsigned int i = 0; i < bq; ++i) {
          uint16_t *sp16 = Sp16.data() + (size_t)i * bk;
          const long long q_abs = (long long)cache_from + qb + i;
          // Softcap before the mask: tanh would turn its -inf into -cap.
          if (softcap > 0.0f) {
            for (unsigned int k = 0; k < bk; ++k)
              sp16[k] = nntrainer::compute_fp32_to_fp16(
                std::tanh(nntrainer::compute_fp16_to_fp32(sp16[k]) *
                          inv_softcap) *
                softcap);
          }
          if (causal_boundary) {
            long long valid_count_ll = q_abs + 1 - (long long)kb;
            unsigned int valid_count = (valid_count_ll <= 0)
                                         ? 0u
                                         : (valid_count_ll >= (long long)bk
                                              ? bk
                                              : (unsigned int)valid_count_ll);
            for (unsigned int k = valid_count; k < bk; ++k)
              sp16[k] = 0xFC00; // FP16 -infinity
          }
          if (window_boundary) {
            long long first_valid_ll = q_abs - (long long)W - (long long)kb + 1;
            unsigned int first_valid = (first_valid_ll <= 0)
                                         ? 0u
                                         : (first_valid_ll >= (long long)bk
                                              ? bk
                                              : (unsigned int)first_valid_ll);
            for (unsigned int k = 0; k < first_valid; ++k)
              sp16[k] = 0xFC00;
          }

          // Block max (read FP16, compute FP32 register for stability).
          float bm = -3.0e38f;
          {
            float32x4_t vmx = vdupq_n_f32(-3.0e38f);
            unsigned int k = 0;
            for (; k + 4 <= bk; k += 4) {
              float16x4_t h = vreinterpret_f16_u16(vld1_u16(sp16 + k));
              vmx = vmaxq_f32(vmx, vcvt_f32_f16(h));
            }
            bm = vmaxvq_f32(vmx);
            for (; k < bk; ++k)
              bm = std::max(bm, nntrainer::compute_fp16_to_fp32(sp16[k]));
          }
          const float nm = std::max(mrow[i], bm);
          const float c = std::exp(mrow[i] - nm);
          float bs = 0.0f;
          {
            // Softmax: read FP16 -> FP32 register, exp, store FP16.
            float32x4_t vsum = vdupq_n_f32(0.0f), vnm = vdupq_n_f32(nm);
            unsigned int k = 0;
            for (; k + 4 <= bk; k += 4) {
              float16x4_t h = vreinterpret_f16_u16(vld1_u16(sp16 + k));
              float32x4_t v = vcvt_f32_f16(h);
              float32x4_t e = exp_ps(vsubq_f32(v, vnm));
              float16x4_t e_h = vcvt_f16_f32(e);
              vst1_u16(sp16 + k, vreinterpret_u16_f16(e_h));
              vsum = vaddq_f32(vsum, e);
            }
            bs = vaddvq_f32(vsum);
            for (; k < bk; ++k) {
              float v = nntrainer::compute_fp16_to_fp32(sp16[k]);
              float e = std::exp(v - nm);
              sp16[k] = nntrainer::compute_fp32_to_fp16(e);
              bs += e;
            }
          }
          lrow[i] = lrow[i] * c + bs;
          mrow[i] = nm;
          float *ol = Ol.data() + (size_t)i * d;
          for (unsigned int x = 0; x < d; ++x)
            ol[x] *= c;
        }

        // AV: FP16 × FP16 -> FP16 (custom_hgemm).
        nntrainer::neon::custom_hgemm(
          reinterpret_cast<const __fp16 *>(Sp16.data()),
          reinterpret_cast<const __fp16 *>(Vp + (size_t)kb * d),
          reinterpret_cast<__fp16 *>(Pacc16.data()), bq, d, bk, 1.0f, 0.0f,
          /*TransA=*/false, /*TransB=*/false);
        // Accumulate Pacc16 -> Ol FP32 (FP32 accumulator across kb).
        for (unsigned int i = 0; i < bq; ++i) {
          float *ol = Ol.data() + (size_t)i * d;
          const uint16_t *pa = Pacc16.data() + (size_t)i * d;
          unsigned int x = 0;
          for (; x + 8 <= d; x += 8) {
            float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(pa + x));
            float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
            float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
            vst1q_f32(ol + x, vaddq_f32(vld1q_f32(ol + x), lo));
            vst1q_f32(ol + x + 4, vaddq_f32(vld1q_f32(ol + x + 4), hi));
          }
          for (; x < d; ++x)
            ol[x] += nntrainer::compute_fp16_to_fp32(pa[x]);
        }
      } else
#endif // ARM NEON q_fp16 branch
      {
        // FP32 Q path: QK -> FP32 S, fused FP16 softmax store to Sp16, AV.
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
        mha_hsgemm_avx2(bq, bk, d, inv_sqrt, Qp_fp32 + (size_t)qb * d, d,
                        Kp + (size_t)kb * d, d, /*TransB=*/true, S.data(), bk);
#elif defined(__ARM_NEON) && defined(ENABLE_FP16)
          nntrainer::shgemm(
            order, false, true, bq, bk, d, inv_sqrt,
            Qp_fp32 + (size_t)qb * d, d,
            reinterpret_cast<const __fp16 *>(Kp + (size_t)kb * d), d, 0.0f,
            S.data(), bk);
#else
        // No fused fp16-bits GEMM here: widen the key block once per tile,
        // then an fp32 GEMM. Live on ARM without enable-fp16 and elsewhere.
        thread_local std::vector<float> Kf;
        Kf.resize((size_t)bk * d);
        for (size_t i = 0; i < (size_t)bk * d; ++i)
          Kf[i] = nntrainer::compute_fp16_to_fp32(Kp[(size_t)kb * d + i]);
        nntrainer::sgemm(order, false, true, bq, bk, d, inv_sqrt,
                         Qp_fp32 + (size_t)qb * d, d, Kf.data(), d, 0.0f,
                         S.data(), bk);
#endif

        for (unsigned int i = 0; i < bq; ++i) {
          float *s = S.data() + (size_t)i * bk;
          const long long q_abs = (long long)cache_from + qb + i;
          // Softcap before the mask: tanh would turn its -inf into -cap.
          if (softcap > 0.0f) {
            for (unsigned int k = 0; k < bk; ++k)
              s[k] = std::tanh(s[k] * inv_softcap) * softcap;
          }
          if (causal_boundary) {
            long long valid_count_ll = q_abs + 1 - (long long)kb;
            unsigned int valid_count = (valid_count_ll <= 0)
                                         ? 0u
                                         : (valid_count_ll >= (long long)bk
                                              ? bk
                                              : (unsigned int)valid_count_ll);
            for (unsigned int k = valid_count; k < bk; ++k)
              s[k] = -INFINITY;
          }
          if (window_boundary) {
            long long first_valid_ll = q_abs - (long long)W - (long long)kb + 1;
            unsigned int first_valid = (first_valid_ll <= 0)
                                         ? 0u
                                         : (first_valid_ll >= (long long)bk
                                              ? bk
                                              : (unsigned int)first_valid_ll);
            for (unsigned int k = 0; k < first_valid; ++k)
              s[k] = -INFINITY;
          }

          float bm = -3.0e38f;
#if defined(__ARM_NEON)
          {
            float32x4_t vmx = vdupq_n_f32(-3.0e38f);
            unsigned int k = 0;
            for (; k + 4 <= bk; k += 4)
              vmx = vmaxq_f32(vmx, vld1q_f32(s + k));
            bm = vmaxvq_f32(vmx);
            for (; k < bk; ++k)
              bm = std::max(bm, s[k]);
          }
#else
            for (unsigned int k = 0; k < bk; ++k)
              bm = std::max(bm, s[k]);
#endif
          const float nm = std::max(mrow[i], bm);
          const float c = std::exp(mrow[i] - nm);
          float bs = 0.0f;
#if !defined(__x86_64__) && !defined(__i386__) && defined(__ARM_NEON) &&       \
  defined(ENABLE_FP16)
          {
            uint16_t *sp16 = Sp16.data() + (size_t)i * bk;
            float32x4_t vsum = vdupq_n_f32(0.0f), vnm = vdupq_n_f32(nm);
            unsigned int k = 0;
            for (; k + 4 <= bk; k += 4) {
              float32x4_t e = exp_ps(vsubq_f32(vld1q_f32(s + k), vnm));
              float16x4_t e_h = vcvt_f16_f32(e);
              vst1_u16(sp16 + k, vreinterpret_u16_f16(e_h));
              vsum = vaddq_f32(vsum, e);
            }
            bs = vaddvq_f32(vsum);
            for (; k < bk; ++k) {
              float e = std::exp(s[k] - nm);
              sp16[k] = nntrainer::compute_fp32_to_fp16(e);
              bs += e;
            }
          }
#elif defined(__ARM_NEON)
            {
              float32x4_t vsum = vdupq_n_f32(0.0f), vnm = vdupq_n_f32(nm);
              unsigned int k = 0;
              for (; k + 4 <= bk; k += 4) {
                float32x4_t e =
                  exp_ps(vsubq_f32(vld1q_f32(s + k), vnm));
                vst1q_f32(s + k, e);
                vsum = vaddq_f32(vsum, e);
              }
              bs = vaddvq_f32(vsum);
              for (; k < bk; ++k) {
                float e = std::exp(s[k] - nm);
                s[k] = e;
                bs += e;
              }
            }
#else
            for (unsigned int k = 0; k < bk; ++k) {
              float e = std::exp(s[k] - nm);
              s[k] = e;
              bs += e;
            }
#endif
          lrow[i] = lrow[i] * c + bs;
          mrow[i] = nm;
          float *ol = Ol.data() + (size_t)i * d;
          for (unsigned int x = 0; x < d; ++x)
            ol[x] *= c;
        }

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
        mha_hsgemm_avx2(bq, d, bk, 1.0f, S.data(), bk, Vp + (size_t)kb * d, d,
                        /*TransB=*/false, Pacc.data(), d);
        for (unsigned int i = 0; i < bq; ++i) {
          float *ol = Ol.data() + (size_t)i * d;
          const float *pa = Pacc.data() + (size_t)i * d;
          for (unsigned int x = 0; x < d; ++x)
            ol[x] += pa[x];
        }
#elif defined(__ARM_NEON) && defined(ENABLE_FP16)
          nntrainer::neon::custom_hgemm(
            reinterpret_cast<const __fp16 *>(Sp16.data()),
            reinterpret_cast<const __fp16 *>(Vp + (size_t)kb * d),
            reinterpret_cast<__fp16 *>(Pacc16.data()), bq, d, bk, 1.0f, 0.0f,
            /*TransA=*/false, /*TransB=*/false);
          for (unsigned int i = 0; i < bq; ++i) {
            float *ol = Ol.data() + (size_t)i * d;
            const uint16_t *pa = Pacc16.data() + (size_t)i * d;
            unsigned int x = 0;
            for (; x + 8 <= d; x += 8) {
              float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(pa + x));
              float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
              float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
              vst1q_f32(ol + x, vaddq_f32(vld1q_f32(ol + x), lo));
              vst1q_f32(ol + x + 4, vaddq_f32(vld1q_f32(ol + x + 4), hi));
            }
            for (; x < d; ++x)
              ol[x] += nntrainer::compute_fp16_to_fp32(pa[x]);
          }
#else
        thread_local std::vector<float> Vf;
        Vf.resize((size_t)bk * d);
        for (size_t i = 0; i < (size_t)bk * d; ++i)
          Vf[i] = nntrainer::compute_fp16_to_fp32(Vp[(size_t)kb * d + i]);
        nntrainer::sgemm(order, false, false, bq, d, bk, 1.0f, S.data(), bk,
                         Vf.data(), d, 0.0f, Pacc.data(), d);
        for (unsigned int i = 0; i < bq; ++i) {
          float *ol = Ol.data() + (size_t)i * d;
          const float *pa = Pacc.data() + (size_t)i * d;
          for (unsigned int x = 0; x < d; ++x)
            ol[x] += pa[x];
        }
#endif
      } // FP32 Q path
    }
    for (unsigned int i = 0; i < bq; ++i) {
      const float inv = (lrow[i] > 0.0f) ? (1.0f / lrow[i]) : 0.0f;
      const float *ol = Ol.data() + (size_t)i * d;
      if (o_fp16) {
        uint16_t *oh = Oh_fp16 + (size_t)(qb + i) * HD_Q;
        for (unsigned int x = 0; x < d; ++x)
          oh[x] = nntrainer::compute_fp32_to_fp16(ol[x] * inv);
      } else {
        float *oh = Oh + (size_t)(qb + i) * HD_Q;
        for (unsigned int x = 0; x < d; ++x)
          oh[x] = ol[x] * inv;
      }
    }
  });
}

/**
 * @brief Rows a step needs to take the flash arm. MHACoreLayer's
 *        min_prefill_thr(), which the layer keeps for its device arms, read
 *        here for the host arm that follows them. NNTR_MIN_PREFILL overrides.
 *        x86 takes the flash arm for every step. ARM needs 32 rows, or any
 *        step with NNTR_MHA_GPU on, where the device arms are offered every
 *        prefill and the host flash arm is what follows one that declined.
 */
static unsigned int flash_min_rows() {
  static const int env = []() {
    const char *e = std::getenv("NNTR_MIN_PREFILL");
    return e ? std::atoi(e) : -1;
  }();
  if (env >= 0)
    return (unsigned int)env;
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
  return 1u;
#else
  static const bool mha_gpu = nntr_env_on("NNTR_MHA_GPU");
  return mha_gpu ? 1u : 32u;
#endif
}

/**
 * @brief Which host arm a decode step (one row) takes when the flash arm is
 *        allowed. ARM: the per-row arm, whose NEON compute_kcaches /
 *        compute_fp16vcache_transposed kernels measured 2.5-5x faster than the
 *        flash arm's M=1 blocked fp16 GEMM plus its thread hand-off (KleidiAI
 *        CPU decode, 1K summary prompt: 62.5 vs 12.5 TPS on a dense 1.5B
 *        model, 18.6 vs 7.5 TPS on a 2B-class model). x86: the flash arm,
 *        whose FP32 GEMM measured 4-6 % faster than the scalar per-row
 *        fallbacks there (4.90 vs 4.62 and 2.04 vs 1.96 TPS on the same two).
 */
static constexpr bool decode_prefers_per_row() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
  return false;
#else
  return true;
#endif
}

} // namespace

void CpuComputeOps::attention(const Tensor &q, const KVCacheView &kv,
                              Tensor &out, const AttentionOpParams &p) {
  if (kv.key == nullptr || kv.value == nullptr)
    throw std::invalid_argument(
      "CpuComputeOps::attention: the cache needs both planes");
  if (kv.ring_cap != 0)
    throw std::invalid_argument(
      "CpuComputeOps::attention: the host arms read the cache by absolute row "
      "and cannot read a ringed one; no ring-aware device arm took this step "
      "(set NNTR_KV_WINDOW_RING=0 for a linear cache)");
  // A tensor in a device-only pool (NNTR_CUDA_DEV_ACT activations,
  // NNTR_CUDA_KV_DEV cache) is not host-addressable, and the host arms would
  // fault reading it in a worker thread. This holds whichever ops table the
  // call came through.
  auto device_only = [](const Tensor *t) {
    const auto md = t->getMemoryData();
    return md && !md->isHostAddressable();
  };
  if (device_only(&q) || device_only(kv.key) || device_only(kv.value) ||
      device_only(&out))
    throw std::runtime_error(
      "CpuComputeOps::attention: the query, KV cache or output is in "
      "device-only memory, which the host arms cannot read");
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

  const unsigned int from = p.q_pos;
  const unsigned int to = p.kv_len;

  // The flash arm when the caller allows it (p.flash) and it applies: it
  // reads an fp16-bits or int8 cache and has no sink. It pays off from
  // flash_min_rows() rows; a decode step takes the platform's faster arm.
  // Everything else is the per-row arm.
  const bool flash = p.flash && p.sink == nullptr &&
                     kv_type != DataType::FP32 && rows >= flash_min_rows() &&
                     !(rows == 1 && decode_prefers_per_row());
  if (flash) {
    FlashAttention f{p.num_heads_q,
                     num_heads_kv,
                     head_dim,
                     p.causal,
                     p.window,
                     p.softcap,
                     kv_type == DataType::UINT8,
                     kv.key_scale,
                     kv.value_scale};
    f.run(q, *kv.key, *kv.value, out, to, rows, from);
    return;
  }

  RowwiseAttention r{p.causal,     p.window,
                     p.softcap,    kv_type == DataType::UINT8,
                     kv.key_scale, kv.value_scale};
  const unsigned int gqa_size = p.num_heads_q / num_heads_kv;
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

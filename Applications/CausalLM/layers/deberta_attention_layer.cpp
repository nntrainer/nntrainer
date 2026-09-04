// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   deberta_attention_layer.cpp
 * @date   16 March 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  DeBERTa Attention Layer based on mha_core optimized path.
 *
 *         Main execution flow:
 *           1) compute_kcaches()               -> content-to-content score
 *           2) rescale content score           -> match DeBERTa scaling
 *           3) add_relative_attn_score()       -> add c2p / p2c score
 *           4) softmax_triangle()              -> softmax on score
 *           5) compute_fp16vcache_transposed() -> score * value
 */

#include <deberta_attention_layer.h>

#include <algorithm> // std::min, std::max
#include <cmath>     // std::log, std::ceil, std::sqrt, std::tanh, std::abs
#include <limits>    // std::numeric_limits
#include <mutex>     // std::lock_guard, std::mutex
#include <vector>    // std::vector

#include <fp16.h>
#include <nntrainer_error.h>

#include <node_exporter.h>
#include <thread_manager.h>

namespace causallm {

#define tile_size 4

namespace {

/**
 * @brief clamp helper
 */
template <typename T> static inline T clampv(T v, T lo, T hi) {
  return std::min(std::max(v, lo), hi);
}

/**
 * @brief exact bucket implementation used in the original working version
 */
static int compute_bucket_pos_impl(int relative_pos, int bucket_size,
                                   int max_position) {
  const int mid = bucket_size / 2;
  const int sign = (relative_pos >= 0) ? 1 : -1;
  const int abs_rp = std::abs(relative_pos);
  const int abs_pos = (abs_rp < mid) ? (mid - 1) : abs_rp;

  const double num = std::log((double)abs_pos / (double)mid);
  const double den = std::log((double)(max_position - 1) / (double)mid);
  int log_pos = (int)std::ceil(num / den * (mid - 1)) + mid;

  return (abs_pos <= mid) ? relative_pos : (log_pos * sign);
}

/**
 * @brief cache key for shared c2p/p2c relative index tables
 *
 * These indices depend only on relative geometry, not on layer weights.
 * So all attention layers can reuse them.
 */
struct RelativeIndexKey {
  unsigned int from;
  unsigned int to;
  unsigned int att_span;
  unsigned int rel_len_q;
  unsigned int rel_len_k;
  bool c2p;
  bool p2c;

  bool operator==(const RelativeIndexKey &rhs) const {
    return from == rhs.from && to == rhs.to && att_span == rhs.att_span &&
           rel_len_q == rhs.rel_len_q && rel_len_k == rhs.rel_len_k &&
           c2p == rhs.c2p && p2c == rhs.p2c;
  }
};

/**
 * @brief shared c2p/p2c relative index value
 */
struct RelativeIndexValue {
  std::vector<int> c2p_idx;
  std::vector<int> p2c_idx;
};

/**
 * Android/mobile path prefers avoiding global lock + deep copy on every call.
 * Reuse per-thread relative index cache instead.
 */
static thread_local bool tl_rel_index_ready = false;
static thread_local RelativeIndexKey tl_rel_index_key{};
static thread_local RelativeIndexValue tl_rel_index_value{};

/**
 * Scratch buffer for unpacking key cache to FP32 in FP32 relative-attention
 * path. Reused per-thread to avoid repeated allocation/resizing overhead.
 */
static thread_local std::vector<float> tl_key_cache_fp32_buf;

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
        const float *score =
          in + row * num_cache_head * gqa_size + n * gqa_size + h;
        const float *value = vcache + (row * num_cache_head + n) * head_dim;
        const float weight = *score;

        for (int d = 0; d < head_dim; ++d) {
          out[d] += weight * value[d];
        }
      }
    }
  }
}

} // namespace

/**
 * @brief constructor
 */
DebertaAttentionLayer::DebertaAttentionLayer() :
  LayerImpl(),
  deberta_props(nntrainer::props::NumHeads(), props::MaxPositionEmbeddings(),
                props::MaxRelativePositions(), props::C2P(), props::P2C(),
                props::ShareAttKey(), props::RelativeAttention(),
                props::PositionBuckets(), props::InputLen(),
                nntrainer::props::DisableBias()),
  epsilon(1e-3f),
  num_heads_Q(0),
  num_heads_KV(0),
  head_dim(0),
  max_position_embeddings(0),
  max_relative_positions(0),
  position_buckets(0),
  local_window_size(0),
  attn_logit_softcapping(0.0f),
  bucket_table_max_seq_len(0),
  bucket_table_ready(false) {
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

DebertaAttentionLayer::~DebertaAttentionLayer() {}

/**
 * @brief finalize
 */
void DebertaAttentionLayer::finalize(nntrainer::InitLayerContext &context) {
  const bool share_att_key = std::get<props::ShareAttKey>(deberta_props).get();
  const bool relative_attention =
    std::get<props::RelativeAttention>(deberta_props).get();
  const bool c2p = std::get<props::C2P>(deberta_props).get();
  const bool p2c = std::get<props::P2C>(deberta_props).get();

  NNTR_THROW_IF(!share_att_key, nntrainer::exception::not_supported)
    << "DebertaAttentionLayer: share_att_key=false is not supported yet.";

  unsigned int expected_inputs = 3;
  if (relative_attention) {
    if (p2c)
      expected_inputs++;
    if (c2p)
      expected_inputs++;
  }

  NNTR_THROW_IF(context.getNumInputs() != expected_inputs,
                std::invalid_argument)
    << "DebertaAttentionLayer expects " << expected_inputs
    << " inputs, but got " << context.getNumInputs();

  [[maybe_unused]] auto [output_dims, weight_dims, tensor_dims] =
    getLayerDimensions(context);

  const auto &input_dims = context.getInputDimensions();
  const auto &query_dim = input_dims[INPUT_IDX_Q];
  const auto &key_dim = input_dims[INPUT_IDX_K];
  const auto &value_dim = input_dims[INPUT_IDX_V];

  NNTR_THROW_IF(query_dim.width() != key_dim.width(), std::invalid_argument)
    << "query/key hidden width mismatch";

  NNTR_THROW_IF(key_dim.width() != value_dim.width(), std::invalid_argument)
    << "key/value hidden width mismatch";

  num_heads_Q = static_cast<size_t>(
    std::get<nntrainer::props::NumHeads>(deberta_props).get());
  num_heads_KV = num_heads_Q;

  NNTR_THROW_IF(num_heads_Q == 0, std::invalid_argument)
    << "num_heads must be > 0";

  head_dim = static_cast<size_t>(query_dim.width()) / num_heads_Q;

  NNTR_THROW_IF(head_dim * num_heads_Q != query_dim.width(),
                std::invalid_argument)
    << "query width must be divisible by num_heads";

  max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(deberta_props).get();
  max_relative_positions =
    std::get<props::MaxRelativePositions>(deberta_props).get();
  position_buckets = std::get<props::PositionBuckets>(deberta_props).get();

  if (max_relative_positions < 1)
    max_relative_positions = max_position_embeddings;

  local_window_size = std::max<unsigned int>(
    query_dim.height(),
    max_position_embeddings > 0 ? max_position_embeddings : query_dim.height());

  attn_logit_softcapping = 0.0f;

  tensor_idx[AttentionParams::cache_key] = context.requestTensor(
    tensor_dims[0], "cache_key", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  tensor_idx[AttentionParams::cache_value] = context.requestTensor(
    tensor_dims[1], "cache_value", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  context.setOutputDimensions(output_dims);

  prepare_bucket_table(
    std::max<unsigned int>(query_dim.height(), max_position_embeddings));
}

void DebertaAttentionLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, deberta_props);
  LayerImpl::setProperty(remain_props);
}

void DebertaAttentionLayer::prepare_bucket_table(unsigned int max_seq_len) {
  const bool relative_attention =
    std::get<props::RelativeAttention>(deberta_props).get();
  if (!relative_attention || position_buckets <= 0)
    return;

  std::lock_guard<std::mutex> lock(bucket_mtx);

  if (bucket_table_ready && bucket_table_max_seq_len >= max_seq_len)
    return;

  bucket_table.resize(max_seq_len * 2 + 1);
  const int offset = static_cast<int>(max_seq_len);

  for (int diff = -static_cast<int>(max_seq_len);
       diff <= static_cast<int>(max_seq_len); ++diff) {
    bucket_table[diff + offset] =
      compute_bucket_pos_impl(diff, static_cast<int>(position_buckets),
                              static_cast<int>(max_relative_positions));
  }

  bucket_table_max_seq_len = max_seq_len;
  bucket_table_ready = true;
}

int DebertaAttentionLayer::lookup_bucket(int relative_pos) const {
  if (!bucket_table_ready || position_buckets <= 0)
    return relative_pos;

  const int offset = static_cast<int>(bucket_table_max_seq_len);
  const int idx =
    clampv(relative_pos + offset, 0, static_cast<int>(bucket_table.size()) - 1);
  return bucket_table[idx];
}

void DebertaAttentionLayer::forwarding(nntrainer::RunLayerContext &context,
                                       bool training) {
  throw nntrainer::exception::not_supported(
    "DebertaAttentionLayer::forwarding is not supported yet");
}

/**
 * @brief incremental forwarding
 */
void DebertaAttentionLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int _from, unsigned int _to,
  bool training) {

  auto get_step_dim = [_from, _to](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(_to - _from);
    return step_dim;
  };

  nntrainer::Tensor &query = context.getInput(INPUT_IDX_Q);
  nntrainer::Tensor &key = context.getInput(INPUT_IDX_K);
  nntrainer::Tensor &value = context.getInput(INPUT_IDX_V);
  nntrainer::Tensor &output = context.getOutput(OUTPUT_IDX);

  nntrainer::Tensor &cache_key =
    context.getTensor(tensor_idx[AttentionParams::cache_key]);
  nntrainer::Tensor &cache_value =
    context.getTensor(tensor_idx[AttentionParams::cache_value]);

  ml::train::TensorDim query_dim = query.getDim();
  ml::train::TensorDim key_dim = key.getDim();
  ml::train::TensorDim value_dim = value.getDim();
  ml::train::TensorDim output_dim = output.getDim();
  ml::train::TensorDim cache_key_dim = cache_key.getDim();
  ml::train::TensorDim cache_value_dim = cache_value.getDim();

  ml::train::TensorDim query_step_dim = get_step_dim(query_dim);
  ml::train::TensorDim key_step_dim = get_step_dim(key_dim);
  ml::train::TensorDim value_step_dim = get_step_dim(value_dim);
  ml::train::TensorDim output_step_dim = get_step_dim(output_dim);
  ml::train::TensorDim cache_key_step_dim = get_step_dim(cache_key_dim);
  ml::train::TensorDim cache_value_step_dim = get_step_dim(cache_value_dim);

  const unsigned int batch_size = query_dim.batch();

  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    nntrainer::Tensor query_step = query.getSharedDataTensor(
      query_step_dim,
      batch * query_dim.getFeatureLen() + _from * query_dim.width(), true);
    nntrainer::Tensor key_step = key.getSharedDataTensor(
      key_step_dim, batch * key_dim.getFeatureLen() + _from * key_dim.width(),
      true);
    nntrainer::Tensor value_step = value.getSharedDataTensor(
      value_step_dim,
      batch * value_dim.getFeatureLen() + _from * value_dim.width(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim,
      batch * output_dim.getFeatureLen() + _from * output_dim.width(), true);

    one_batch_incremental_forwarding(
      context, batch, _from, _from, _to, query_step, key_step, value_step,
      output_step, cache_key, cache_value, cache_key_dim, cache_key_step_dim,
      cache_value_dim, cache_value_step_dim);
  }
}

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
void DebertaAttentionLayer::compute_kcaches(
  nntrainer::Tensor &in, nntrainer::Tensor &cache, nntrainer::Tensor &out,
  unsigned int from, size_t sequence_len, unsigned int num_head,
  unsigned int group_size, unsigned int head_dim) {

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (sequence_len == 1) {
      const int row_to_compute = from + sequence_len;
      const unsigned int num_cache_head = num_head / group_size;

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
      const unsigned int seq = static_cast<unsigned int>(sequence_len);

      for (unsigned int i = 0; i < seq; ++i) {
        float *input_addr = in.getData<float>() + num_head * head_dim * i;
        const int row_to_compute = from + sequence_len;
        const size_t out_start_row = i * (from + sequence_len);
        float *output_addr = out.getData<float>() + out_start_row * num_head;

        if (cache.getDataType() == ml::train::TensorDim::DataType::FP32) {
          compute_kcaches_fp32_reference(
            input_addr, cache.getData<float>(), output_addr, row_to_compute,
            num_head / group_size, head_dim, group_size, local_window_size);
        } else {
          uint16_t *cache_addr = cache.getData<uint16_t>();
          nntrainer::compute_kcaches<uint16_t>(
            input_addr, cache_addr, output_addr, row_to_compute,
            num_head / group_size, head_dim, group_size, tile_size,
            local_window_size);
        }
      }
    }

  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (sequence_len == 1) {
      const int num_rows = from + sequence_len;
      const unsigned int num_cache_head = num_head / group_size;

      const _FP16 *in_data = in.getData<_FP16>();
      const _FP16 *cache_data = cache.getData<_FP16>();
      _FP16 *out_data = out.getData<_FP16>();

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(
        0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
          nntrainer::compute_kcaches(
            in_data, cache_data, out_data, num_rows, num_cache_head, head_dim,
            group_size, tile_size, local_window_size, head_kv, head_kv + 1);
        });
    } else {
      const unsigned int seq = static_cast<unsigned int>(sequence_len);

      for (unsigned int i = 0; i < seq; ++i) {
        _FP16 *input_addr = in.getData<_FP16>() + num_head * head_dim * i;
        _FP16 *cache_addr = cache.getData<_FP16>();
        const int row_to_compute = from + sequence_len;
        const size_t out_start_row = i * (from + sequence_len);
        _FP16 *output_addr = out.getData<_FP16>() + out_start_row * num_head;

        nntrainer::compute_kcaches(input_addr, cache_addr, output_addr,
                                   row_to_compute, num_head / group_size,
                                   head_dim, group_size, tile_size,
                                   local_window_size);
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

/**
 * @brief softmax for linear layout [S_q, to, H]
 */
void DebertaAttentionLayer::softmax_triangle(nntrainer::Tensor &qk_out,
                                             size_t row, size_t num_head,
                                             unsigned int from) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();

    if (attn_logit_softcapping > 0.0f) {
      const size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      const float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] =
          std::tanh(qk_out_[i] * inv_softcapping) * attn_logit_softcapping;
      }
    }

    if (row == 1) {
      const size_t start_row = 0;
      const size_t end_row = from + row;
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      for (unsigned int i = 0; i < row; ++i) {
        const unsigned int to = from + row;
        const size_t start_row = i * to;
        const size_t end_row = (i + 1) * to;
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head);
      }
    }

  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();

    if (attn_logit_softcapping > 0.0f) {
      const size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      const float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] = (_FP16)(std::tanh((float)qk_out_[i] * inv_softcapping) *
                             attn_logit_softcapping);
      }
    }

    if (row == 1) {
      const size_t start_row = 0;
      const size_t end_row = from + row;
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      for (unsigned int i = 0; i < row; ++i) {
        const unsigned int to = from + row;
        const size_t start_row = i * to;
        const size_t end_row = (i + 1) * to;
        nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

/**
 * @brief score * value using mha_core optimized path
 */
void DebertaAttentionLayer::compute_fp16vcache_transposed(
  nntrainer::Tensor &in, nntrainer::Tensor &vcache, nntrainer::Tensor &output,
  int from, int num_cache_head, int gqa_size, int head_dim, int to) {

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if ((to - from) != 1) {
      const int seq = to - from;

      for (int i = 0; i < seq; ++i) {
        const size_t start_idx = i * to;
        const float *input =
          in.getData<float>() + start_idx * num_cache_head * gqa_size;
        float *out =
          output.getData<float>() + i * (num_cache_head * gqa_size * head_dim);
        const int row_num = to - 1;

        if (vcache.getDataType() == ml::train::TensorDim::DataType::FP32) {
          compute_vcache_fp32_transposed_reference(
            row_num, input, vcache.getData<float>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        } else {
          nntrainer::compute_fp16vcache_fp32_transposed(
            row_num, input, vcache.getData<uint16_t>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        }
      }

    } else {
      const int row_num = to - 1;

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
      const int seq = to - from;

      for (int i = 0; i < seq; ++i) {
        const size_t start_idx = i * to;
        const _FP16 *input =
          in.getData<_FP16>() + start_idx * num_cache_head * gqa_size;
        _FP16 *out =
          output.getData<_FP16>() + i * (num_cache_head * gqa_size * head_dim);
        const int row_num = to - 1;

        nntrainer::compute_fp16vcache_transposed(
          row_num, input, vcache.getData<_FP16>(), out, num_cache_head,
          gqa_size, head_dim, local_window_size);
      }

    } else {
      const int row_num = to - 1;

      const _FP16 *in_data = in.getData<_FP16>();
      const _FP16 *vcache_data = vcache.getData<_FP16>();
      _FP16 *output_data = output.getData<_FP16>();

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(
        0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
          nntrainer::compute_fp16vcache_transposed(
            row_num, in_data, vcache_data, output_data, num_cache_head,
            gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
        });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

/**
 * @brief add DeBERTa relative score (c2p / p2c)
 */
void DebertaAttentionLayer::add_relative_attn_score(
  nntrainer::RunLayerContext &context, nntrainer::Tensor &score,
  nntrainer::Tensor &query_step, nntrainer::Tensor &key_cache,
  unsigned int from, unsigned int to) {

  const bool relative_attention =
    std::get<props::RelativeAttention>(deberta_props).get();
  const bool c2p = std::get<props::C2P>(deberta_props).get();
  const bool p2c = std::get<props::P2C>(deberta_props).get();

  if (!relative_attention || (!c2p && !p2c))
    return;

  size_t next_input_idx = 3;
  nntrainer::Tensor rel_query;
  nntrainer::Tensor rel_key;

  if (p2c) {
    rel_query = context.getInput(next_input_idx++);
  }
  if (c2p) {
    rel_key = context.getInput(next_input_idx++);
  }

  const unsigned int S_q = to - from;
  const unsigned int S_k = to;
  const unsigned int hidden = static_cast<unsigned int>(num_heads_Q * head_dim);

  const unsigned int att_span =
    position_buckets > 0 ? position_buckets : max_relative_positions;

  const int scale_factor_int = 1 + (c2p ? 1 : 0) + (p2c ? 1 : 0);
  const float scale =
    1.0f / std::sqrt(static_cast<float>(head_dim * scale_factor_int));

  /**
   * Shared relative index cache across attention layers
   */
  const unsigned int rel_len_q = p2c ? rel_query.height() : 0u;
  const unsigned int rel_len_k = c2p ? rel_key.height() : 0u;

  const RelativeIndexKey cache_key{from,      to,  att_span, rel_len_q,
                                   rel_len_k, c2p, p2c};

  RelativeIndexValue &rel_idx_local = tl_rel_index_value;

  if (!tl_rel_index_ready || !(tl_rel_index_key == cache_key)) {
    rel_idx_local.c2p_idx.clear();
    rel_idx_local.p2c_idx.clear();

    if (c2p) {
      rel_idx_local.c2p_idx.resize(static_cast<size_t>(S_q) * S_k);
    }
    if (p2c) {
      rel_idx_local.p2c_idx.resize(static_cast<size_t>(S_q) * S_k);
    }

    auto &tm_idx = nntrainer::ThreadManager::Global();
    tm_idx.parallel_for(0, static_cast<size_t>(S_q), [&](size_t q) {
      for (unsigned int k = 0; k < S_k; ++k) {
        if (c2p) {
          const int rel = static_cast<int>(q + from) - static_cast<int>(k);
          const int bucketed = lookup_bucket(rel);
          rel_idx_local.c2p_idx[static_cast<size_t>(q) * S_k + k] =
            clampv(bucketed + static_cast<int>(att_span), 0,
                   static_cast<int>(rel_len_k) - 1);
        }

        if (p2c) {
          const int rel = static_cast<int>(q + from) - static_cast<int>(k);
          const int bucketed = lookup_bucket(rel);
          // HF gathers key·pos_query with clamp(-r_pos+span) and then
          // transposes (q<->k); the net effective index for score[q,k] is
          // clamp(bucketed(q-k)+span), matching the c2p index.
          rel_idx_local.p2c_idx[static_cast<size_t>(q) * S_k + k] =
            clampv(bucketed + static_cast<int>(att_span), 0,
                   static_cast<int>(rel_len_q) - 1);
        }
      }
    });

    tl_rel_index_key = cache_key;
    tl_rel_index_ready = true;
  }

  if (score.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *score_ptr = score.getData<float>();
    const float *q_ptr = query_step.getData<float>();
    const float *rel_query_ptr = p2c ? rel_query.getData<float>() : nullptr;
    const float *rel_key_ptr = c2p ? rel_key.getData<float>() : nullptr;

    const uint16_t *key_u16_ptr =
      key_cache.getDataType() == ml::train::TensorDim::DataType::UINT16
        ? key_cache.getData<uint16_t>()
        : nullptr;

#ifdef ENABLE_FP16
    const _FP16 *key_fp16_ptr =
      key_cache.getDataType() == ml::train::TensorDim::DataType::FP16
        ? key_cache.getData<_FP16>()
        : nullptr;
#else
    const void *key_fp16_ptr = nullptr;
#endif

    std::vector<float> &key_cache_fp32_buf = tl_key_cache_fp32_buf;
    const float *key_unpacked_ptr =
      key_cache.getDataType() == ml::train::TensorDim::DataType::FP32
        ? key_cache.getData<float>()
        : nullptr;

    /**
     * p2c path needs key-cache values in FP32 for dot(q_rel, k_cache).
     * If cache is packed/FP16, unpack once outside the hot loop.
     */
    if (p2c && (key_u16_ptr
#ifdef ENABLE_FP16
                || key_fp16_ptr
#endif
                )) {
      const size_t unpack_len = static_cast<size_t>(S_k) * hidden;
      if (key_cache_fp32_buf.size() < unpack_len) {
        key_cache_fp32_buf.resize(unpack_len);
      }

      for (unsigned int k = 0; k < S_k; ++k) {
        float *dst =
          key_cache_fp32_buf.data() + static_cast<size_t>(k) * hidden;

        if (key_u16_ptr) {
          const uint16_t *src = key_u16_ptr + static_cast<size_t>(k) * hidden;
          for (unsigned int i = 0; i < hidden; ++i) {
            dst[i] = nntrainer::compute_fp16_to_fp32(src[i]);
          }
        }
#ifdef ENABLE_FP16
        else if (key_fp16_ptr) {
          const _FP16 *src = key_fp16_ptr + static_cast<size_t>(k) * hidden;
          for (unsigned int i = 0; i < hidden; ++i) {
            dst[i] = static_cast<float>(src[i]);
          }
        }
#endif
      }

      key_unpacked_ptr = key_cache_fp32_buf.data();
    }

    NNTR_THROW_IF(p2c && key_unpacked_ptr == nullptr, std::invalid_argument)
      << "FP32 p2c path expected FP32, UINT16, or FP16 key cache";

    auto &tm_fp32 = nntrainer::ThreadManager::Global();
    tm_fp32.parallel_for(0, static_cast<size_t>(S_q), [&](size_t q_idx) {
      const size_t qk_row = static_cast<size_t>(q_idx) * S_k;

      for (unsigned int h = 0; h < num_heads_Q; ++h) {
        const unsigned int h_base = h * head_dim;
        const float *q_head = q_ptr + q_idx * hidden + h_base;

        for (unsigned int k_idx = 0; k_idx < S_k; ++k_idx) {
          float rel_score = 0.0f;

          if (c2p) {
            const int rel_index = rel_idx_local.c2p_idx[qk_row + k_idx];
            const float *rk_head =
              rel_key_ptr + static_cast<size_t>(rel_index) * hidden + h_base;

            float c2p_dot = 0.0f;
            for (unsigned int d = 0; d < head_dim; ++d) {
              c2p_dot += q_head[d] * rk_head[d];
            }
            rel_score += c2p_dot * scale;
          }

          if (p2c) {
            const int rel_index = rel_idx_local.p2c_idx[qk_row + k_idx];
            const float *rq_head =
              rel_query_ptr + static_cast<size_t>(rel_index) * hidden + h_base;

            float p2c_dot = 0.0f;
            const float *k_head =
              key_unpacked_ptr + static_cast<size_t>(k_idx) * hidden + h_base;
            for (unsigned int d = 0; d < head_dim; ++d) {
              p2c_dot += k_head[d] * rq_head[d];
            }
            rel_score += p2c_dot * scale;
          }

          const size_t linear_idx =
            (static_cast<size_t>(q_idx) * S_k + k_idx) * num_heads_Q + h;
          score_ptr[linear_idx] += rel_score;
        }
      }
    });
  } else if (score.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    NNTR_THROW_IF(key_cache.getDataType() !=
                    ml::train::TensorDim::DataType::FP16,
                  std::invalid_argument)
      << "FP16 relative attention path requires FP16 key cache";

    _FP16 *score_ptr = score.getData<_FP16>();
    const _FP16 *q_ptr = query_step.getData<_FP16>();
    const _FP16 *rel_query_ptr = p2c ? rel_query.getData<_FP16>() : nullptr;
    const _FP16 *rel_key_ptr = c2p ? rel_key.getData<_FP16>() : nullptr;
    const _FP16 *key_fp16_ptr = key_cache.getData<_FP16>();

    auto &tm_fp16 = nntrainer::ThreadManager::Global();
    tm_fp16.parallel_for(0, static_cast<size_t>(S_q), [&](size_t q_idx) {
      const size_t qk_row_base = static_cast<size_t>(q_idx) * S_k;

      for (unsigned int h = 0; h < num_heads_Q; ++h) {
        const unsigned int h_base = h * head_dim;
        const _FP16 *q_head = q_ptr + q_idx * hidden + h_base;

        for (unsigned int k_idx = 0; k_idx < S_k; ++k_idx) {
          float rel_score = 0.0f;

          if (c2p) {
            const int rel_index = rel_idx_local.c2p_idx[qk_row_base + k_idx];
            const _FP16 *rk_head =
              rel_key_ptr + static_cast<size_t>(rel_index) * hidden + h_base;

            float c2p_dot = 0.0f;
            for (unsigned int d = 0; d < head_dim; ++d) {
              c2p_dot +=
                static_cast<float>(q_head[d]) * static_cast<float>(rk_head[d]);
            }
            rel_score += c2p_dot * scale;
          }

          if (p2c) {
            const int rel_index = rel_idx_local.p2c_idx[qk_row_base + k_idx];
            const _FP16 *rq_head =
              rel_query_ptr + static_cast<size_t>(rel_index) * hidden + h_base;
            const _FP16 *k_head =
              key_fp16_ptr + static_cast<size_t>(k_idx) * hidden + h_base;

            float p2c_dot = 0.0f;
            for (unsigned int d = 0; d < head_dim; ++d) {
              p2c_dot +=
                static_cast<float>(k_head[d]) * static_cast<float>(rq_head[d]);
            }
            rel_score += p2c_dot * scale;
          }

          const size_t linear_idx =
            (static_cast<size_t>(q_idx) * S_k + k_idx) * num_heads_Q + h;
          score_ptr[linear_idx] =
            (_FP16)(static_cast<float>(score_ptr[linear_idx]) + rel_score);
        }
      }
    });
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

/**
 * @brief one-batch path
 */
void DebertaAttentionLayer::one_batch_incremental_forwarding(
  nntrainer::RunLayerContext &context, const unsigned int batch,
  const unsigned int _from, const unsigned int from, const unsigned int to,
  nntrainer::Tensor &query_step, nntrainer::Tensor &key_step,
  nntrainer::Tensor &value_step, nntrainer::Tensor &attention_output_step,
  nntrainer::Tensor &cache_key, nntrainer::Tensor &cache_value,
  ml::train::TensorDim &cache_key_dim, ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim) {

  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(), true);
  nntrainer::Tensor b_cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim,
    batch * cache_value_dim.getFeatureLen() + from * cache_value_dim.width(),
    true);

  b_cache_key_step.copyData(key_step);
  b_cache_value_step.copyData(value_step);

  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  cached_key_dim.height(to);
  cached_value_dim.height(to);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  nntrainer::Tensor out_(1, 1, (to - from) * to, num_heads_Q,
                         query_step.getTensorType());

  const unsigned int gqa_size = num_heads_Q / num_heads_KV;

  compute_kcaches(query_step, b_cached_key, out_, _from, to - from, num_heads_Q,
                  gqa_size, head_dim);

  const bool relative_attention =
    std::get<props::RelativeAttention>(deberta_props).get();
  if (relative_attention) {
    const bool c2p = std::get<props::C2P>(deberta_props).get();
    const bool p2c = std::get<props::P2C>(deberta_props).get();
    const int scale_factor_int = 1 + (c2p ? 1 : 0) + (p2c ? 1 : 0);
    const float content_rescale =
      1.0f / std::sqrt(static_cast<float>(scale_factor_int));

    if (out_.getDataType() == ml::train::TensorDim::DataType::FP32) {
      float *ptr = out_.getData<float>();
      const size_t len =
        out_.batch() * out_.channel() * out_.height() * out_.width();
      for (size_t i = 0; i < len; ++i) {
        ptr[i] *= content_rescale;
      }
    }
#ifdef ENABLE_FP16
    else if (out_.getDataType() == ml::train::TensorDim::DataType::FP16) {
      _FP16 *ptr = out_.getData<_FP16>();
      const size_t len =
        out_.batch() * out_.channel() * out_.height() * out_.width();
      for (size_t i = 0; i < len; ++i) {
        ptr[i] = (_FP16)((float)ptr[i] * content_rescale);
      }
    }
#endif
    add_relative_attn_score(context, out_, query_step, b_cached_key, from, to);
  }
  softmax_triangle(out_, to - from, num_heads_Q, from);

  compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                from, num_heads_KV, gqa_size, head_dim, to);
}

void DebertaAttentionLayer::setBatch(nntrainer::RunLayerContext &context,
                                     unsigned int batch) {
  context.updateTensor(tensor_idx[AttentionParams::cache_key], batch);
  context.updateTensor(tensor_idx[AttentionParams::cache_value], batch);
}

std::array<std::vector<nntrainer::TensorDim>, 3>
DebertaAttentionLayer::getLayerDimensions(
  nntrainer::InitLayerContext &context) {
  const auto &input_dims = context.getInputDimensions();
  const auto &query_dim = input_dims[INPUT_IDX_Q];
  const auto &key_dim = input_dims[INPUT_IDX_K];
  const auto &value_dim = input_dims[INPUT_IDX_V];

#ifdef ENABLE_FP16
  ml::train::TensorDim cache_key_dim(
    {query_dim.batch(), 1, query_dim.height(), key_dim.width()},
    {context.getFormat(), ml::train::TensorDim::DataType::FP16});
  ml::train::TensorDim cache_value_dim(
    {query_dim.batch(), 1, query_dim.height(), value_dim.width()},
    {context.getFormat(), ml::train::TensorDim::DataType::FP16});
#else
  ml::train::TensorDim cache_key_dim(
    {query_dim.batch(), 1, query_dim.height(), key_dim.width()},
    {context.getFormat(), key_dim.getDataType()});
  ml::train::TensorDim cache_value_dim(
    {query_dim.batch(), 1, query_dim.height(), value_dim.width()},
    {context.getFormat(), value_dim.getDataType()});
#endif

  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = query_dim;

  std::vector<nntrainer::TensorDim> tensor_dims(2);
  tensor_dims[0] = cache_key_dim;   // cache_key is index 0
  tensor_dims[1] = cache_value_dim; // cache_value is index 1

  return std::array<std::vector<nntrainer::TensorDim>, 3>{
    output_dims, {}, tensor_dims};
}

std::vector<nntrainer::TensorDim>
DebertaAttentionLayer::updateTensorsByInputDimensions(
  nntrainer::InitLayerContext &init_context,
  nntrainer::RunLayerContext &run_context) {
  [[maybe_unused]] auto [output_dims, weight_dims, tensor_dims] =
    getLayerDimensions(init_context);

  auto input_dimensions = init_context.getInputDimensions();

  run_context.updateInput(INPUT_IDX_Q, input_dimensions[INPUT_IDX_Q]);
  run_context.updateInput(INPUT_IDX_K, input_dimensions[INPUT_IDX_K]);
  run_context.updateInput(INPUT_IDX_V, input_dimensions[INPUT_IDX_V]);
  run_context.updateOutput(OUTPUT_IDX, output_dims[OUTPUT_IDX]);

  run_context.updateTensor(tensor_idx[AttentionParams::cache_key],
                           tensor_dims[0]);
  run_context.updateTensor(tensor_idx[AttentionParams::cache_value],
                           tensor_dims[1]);

  prepare_bucket_table(std::max<unsigned int>(
    input_dimensions[INPUT_IDX_Q].height(), max_position_embeddings));

  return output_dims;
}

void DebertaAttentionLayer::calcDerivative(
  nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "DebertaAttentionLayer::calcDerivative not supported");
}

void DebertaAttentionLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "DebertaAttentionLayer::calcGradient not supported");
}

void DebertaAttentionLayer::exportTo(
  nntrainer::Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(deberta_props, method, this);
}

#ifdef PLUGGABLE

nntrainer::Layer *create_deberta_attention_layer() {
  auto layer = new DebertaAttentionLayer();
  return layer;
}

void destroy_deberta_attention_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_deberta_attention_layer, destroy_deberta_attention_layer};
}

#endif

} // namespace causallm

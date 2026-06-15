// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   mha_core.cpp
 * @date   11 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This code is based on custom_multi_head_attention_layer.cpp.
 *         This code is a part of the break down version of the mha layer.
 */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

static std::mutex rope_init_mtx;

#include <fp16.h>
#include <layer_context.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <thread_manager.h>
#include <util_func.h>

#include <cstdint>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

inline float convert_scalar(uint16_t h) {
  return nntrainer::compute_fp16_to_fp32(h);
}

namespace causallm {

#define tile_size 4

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

/************************************************************** */

/**
 * @brief constructor of MHACoreLayer
 */
MHACoreLayer::MHACoreLayer() :
  mha_core_props(
    nntrainer::props::NumHeads(), props::NumHeads_KV(),
    nntrainer::props::ProjectedKeyDim(), nntrainer::props::ProjectedValueDim(),
    nntrainer::props::OutputShape(), nntrainer::props::DropOutRate(),
    nntrainer::props::ReturnAttentionWeight(),
    nntrainer::props::AverageAttentionWeight(), nntrainer::props::MaxTimestep(),
    props::SlidingWindow(), props::MaxNewTokens(), props::RopeTheta(),
    props::UseRope(), props::MaxPositionEmbeddings(), props::UseSink(),
    props::RopeScalingType(), props::RopeScalingFactor(),
    props::RopePartialRotaryFactor(), props::RopeScalingMaxPositionEmbeddings(),
    props::AttnLogitSoftcapping(), props::IsCausal(),
    props::UseGemmAttention()),
  sm(nntrainer::ActivationType::ACT_SOFTMAX),
  epsilon(1e-3),
  cache_index(0),
  num_heads_Q(0),
  num_heads_KV(0),
  head_dim(0),
  cache_shift(false) {
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

MHACoreLayer::~MHACoreLayer() {}

/************************************************************** */

void MHACoreLayer::finalize(nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 5,
                std::invalid_argument)
    << "Multi head Attention layer needs 3, 4, or 5 inputs. "
       "(query, key, value; mask is optional; external cache_key + cache_value "
       "for external cache mode)";

  use_external_cache = (context.getNumInputs() >= 5);
  ml::train::TensorDim::TensorType activation_type = {
    context.getFormat(), context.getActivationDataType()};
  ml::train::TensorDim empty_dim(activation_type);

  const std::vector<ml::train::TensorDim> &input_dims =
    context.getInputDimensions();
  const ml::train::TensorDim &query_dim = input_dims[INOUT_INDEX::QUERY];
  const ml::train::TensorDim &key_dim = input_dims[INOUT_INDEX::KEY];

  /** max time step of this model */
  const unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  /** max position embeddings */
  max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mha_core_props).get();

  /** local window size */
  local_window_size = std::get<props::SlidingWindow>(mha_core_props).get();

  /** use rope */
  use_rope = std::get<props::UseRope>(mha_core_props).get();

  /** attention scaling computation */
  rope_scaling_type = std::get<props::RopeScalingType>(mha_core_props).get();
  scale = std::get<props::RopeScalingFactor>(mha_core_props).get();
  rope_partial_rotary_factor =
    std::get<props::RopePartialRotaryFactor>(mha_core_props).get();
  if (rope_scaling_type == "yarn")
    original_max_position_embeddings =
      std::get<props::RopeScalingMaxPositionEmbeddings>(mha_core_props).get();

  /** query_dim = (B, 1, seq_len, H_Q * Head_Dim ) */
  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_width = query_dim.width();
  /** key_dim = (B, 1, max_seq_len, H_KV * Head_Dim ) */
  const unsigned int key_width = key_dim.width();

  /**
   *  @note If NumHeads_KV is set, then use the value. Otherwise,
   *        we initialize num_heads_KV with num_heads_Q.
   */
  num_heads_Q = static_cast<size_t>(
    std::get<nntrainer::props::NumHeads>(mha_core_props).get());
  num_heads_KV =
    std::get<props::NumHeads_KV>(mha_core_props).empty()
      ? num_heads_Q
      : static_cast<size_t>(std::get<props::NumHeads_KV>(mha_core_props).get());

  // head_dim
  head_dim = static_cast<size_t>(query_width) / num_heads_Q;
  NNTR_THROW_IF(head_dim != key_width / num_heads_KV, std::invalid_argument)
    << "num_heads_Q and num_heads_KV are not properly given. Please check the "
       "num_heads_* are set correctly so that the `head_dim`s are all same for "
       "query / key / value";

  /** Weight for Sink */
  use_sink = std::get<props::UseSink>(mha_core_props).get();
  if (use_sink) {
#if ENABLE_FP16 && defined(__ANDROID__)
    nntrainer::TensorDim sink_dim(
      1, 1, 1, num_heads_Q,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       ml::train::TensorDim::DataType::FP16));
#else
    nntrainer::TensorDim sink_dim(
      1, 1, 1, num_heads_Q,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getActivationDataType()));
#endif
    sink_idx = context.requestWeight(sink_dim, nntrainer::Initializer::ZEROS,
                                     nntrainer::WeightRegularizer::NONE, 0.0f,
                                     0.0f, "sink");
  }

  attn_logit_softcapping =
    std::get<props::AttnLogitSoftcapping>(mha_core_props).get();

  /** Is Causal */
  is_causal = std::get<props::IsCausal>(mha_core_props).get();
  use_gemm_attention = std::get<props::UseGemmAttention>(mha_core_props).get();

  if (!std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).get();

  /** Tensor for KV-Cache (only allocate internally when not using external
   * cache) */
  if (!use_external_cache) {
#ifdef ENABLE_FP16
    ml::train::TensorDim cache_key_dim(
      {batch_size, 1, max_timestep, num_heads_KV * head_dim},
      {context.getFormat(), ml::train::TensorDim::DataType::FP16});
    ml::train::TensorDim cache_value_dim(
      {batch_size, 1, max_timestep, num_heads_KV * head_dim},
      {context.getFormat(), ml::train::TensorDim::DataType::FP16});
#else
    ml::train::TensorDim cache_key_dim(
      {batch_size, 1, max_timestep, num_heads_KV * head_dim},
      {context.getFormat(), ml::train::TensorDim::DataType::UINT16});
    ml::train::TensorDim cache_value_dim(
      {batch_size, 1, max_timestep, num_heads_KV * head_dim},
      {context.getFormat(), ml::train::TensorDim::DataType::UINT16});
#endif

    tensor_idx[AttentionParams::cache_key] = context.requestTensor(
      cache_key_dim, "cache_key", nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::MAX_LIFESPAN);
    tensor_idx[AttentionParams::cache_value] = context.requestTensor(
      cache_value_dim, "cache_value", nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::MAX_LIFESPAN);
  }

  theta = (float)std::get<props::RopeTheta>(mha_core_props).get();

  /** set Output dimension! - one output */
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = input_dims[0];
  output_dims[0].width(head_dim * num_heads_Q);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);
}

/************************************************************** */

/**
 * @note In external KV cache mode (use_external_cache == true), this
 *       implements the inference forward pass using cache tensors supplied
 *       as input[3] (cache_key) and input[4] (cache_value). The host (e.g.
 *       KVCacheManager via setExternalTensors) is responsible for owning
 *       these buffers and for calling setCacheIndex() before each step to
 *       set the write position. After this call cache_index is advanced by
 *       input.height().
 *
 *       In legacy 3/4-input mode (use_external_cache == false) training is
 *       NYI and incremental_forwarding() is the inference path.
 *
 *       Input layout for external cache mode:
 *         input[0] = Q   (B, 1, step_size, num_heads_Q  * head_dim)
 *         input[1] = K   (B, 1, step_size, num_heads_KV * head_dim)
 *         input[2] = V   (B, 1, step_size, num_heads_KV * head_dim)
 *         input[3] = cache_key   (B, 1, max_seq_len, num_heads_KV * head_dim)
 *         input[4] = cache_value (B, 1, max_seq_len, num_heads_KV * head_dim)
 */
void MHACoreLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
  if (!use_external_cache) {
    return;
  }

  nntrainer::Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY);
  nntrainer::Tensor &value = context.getInput(INOUT_INDEX::VALUE);
  nntrainer::Tensor &output = context.getOutput(INOUT_INDEX::OUTPUT);

  nntrainer::Tensor &cache_key = context.getInput(3);
  nntrainer::Tensor &cache_value = context.getInput(4);

  nntrainer::Tensor sink;
  if (use_sink) {
    sink = context.getWeight(sink_idx);
  }

  unsigned int step_size = (incremental_step_size > 0)
                             ? incremental_step_size
                             : (unsigned int)query.height();
  unsigned int from = cache_index;
  unsigned int to = cache_index + step_size;

  auto get_step_dim = [step_size](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(step_size);
    return step_dim;
  };

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

  unsigned int batch_size = query_dim.batch();
  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    nntrainer::Tensor query_step = query.getSharedDataTensor(
      query_step_dim, batch * query_dim.getFeatureLen(), true);
    nntrainer::Tensor key_step = key.getSharedDataTensor(
      key_step_dim, batch * key_dim.getFeatureLen(), true);
    nntrainer::Tensor value_step = value.getSharedDataTensor(
      value_step_dim, batch * value_dim.getFeatureLen(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, batch * output_dim.getFeatureLen(), true);

    if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
#if ENABLE_FP16 && defined(__ANDROID__)
      // The Android decode path runs attention in FP16, but only when the
      // external KV cache is itself FP16 (the autoregressive LM case). A
      // bidirectional encoder (e.g. the SigLIP2 vision tower) binds an FP32
      // KV cache; converting Q/K/V to FP16 against an FP32 cache yields a
      // dtype mismatch and garbage output, so route FP32 caches through the
      // FP32 path used on host.
      const bool fp16_cache =
        cache_key.getDataType() == ml::train::TensorDim::DataType::FP16;
      if (fp16_cache) {
        nntrainer::TensorDim Q_step_dim = query_step_dim;
        nntrainer::TensorDim K_step_dim = key_step_dim;
        nntrainer::TensorDim V_step_dim = value_step_dim;
        nntrainer::TensorDim O_step_dim = output_step_dim;
        Q_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
        K_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
        V_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
        O_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);

        nntrainer::Tensor Q_step = nntrainer::Tensor(Q_step_dim, true);
        nntrainer::Tensor K_step = nntrainer::Tensor(K_step_dim, true);
        nntrainer::Tensor V_step = nntrainer::Tensor(V_step_dim, true);
        nntrainer::Tensor O_step = nntrainer::Tensor(O_step_dim, true);

        Q_step.copyData(query_step);
        K_step.copyData(key_step);
        V_step.copyData(value_step);

        if (use_sink) {
          one_batch_incremental_forwarding(
            batch, from, from, to, Q_step, K_step, V_step, O_step, cache_key,
            cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
            cache_value_step_dim, sink);
        } else {
          one_batch_incremental_forwarding(
            batch, from, from, to, Q_step, K_step, V_step, O_step, cache_key,
            cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
            cache_value_step_dim);
        }
        output_step.copyData(O_step);
      } else if (use_sink) {
        one_batch_incremental_forwarding(
          batch, from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(
          batch, from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim);
      }
#else
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(
          batch, from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim);
      }
#endif
    } else {
      one_batch_incremental_forwarding(
        batch, from, from, to, query_step, key_step, value_step, output_step,
        cache_key, cache_value, cache_key_dim, cache_key_step_dim,
        cache_value_dim, cache_value_step_dim);
    }
  }

  cache_index += step_size;
}

/**
 * @note This incremental_forwarding method is invoked for inference mode.
 *       Please note that Transformer Decoder's MHA takes only one sequence at a
 * step. Incremental forwarding function is used for this.
 */
void MHACoreLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int _from, unsigned int _to,
                                          bool training) {
  // External KV cache path: from/to are interpreted as the absolute write
  // position; route through forwarding() which reads cache_key/cache_value
  // from input slots 3/4. forwarding() advances cache_index internally.
  if (use_external_cache) {
    cache_index = _from;
    incremental_step_size = _to - _from;
    forwarding(context, training);
    incremental_step_size = 0;
    return;
  }

  /// @todo replace step_size into input height
  unsigned int step_size = _to - _from;

  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  unsigned int from = _from;
  unsigned int to = _to;

  if (to >= max_timestep) {
    // initial forwarding
    if (!_from) {
      throw std::invalid_argument(
        "to shouldn't greater than max_timestep for initial forwarding");
    } else {
      throw std::runtime_error("NYI: cache shift is not available");
      // exceeds the kv_cache size
      // KV_cache is shifted!
      cache_shift = true;
      from = max_timestep - 1;
      to = max_timestep;
    }
  }

  // util fn to compute tensor dimension for one step.
  auto get_step_dim = [step_size](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(step_size);
    return step_dim;
  };

  /** incremental forwarding for each batch */
  nntrainer::Tensor &query =
    context.getInput(INOUT_INDEX::QUERY); // projected query
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY); // projected key
  nntrainer::Tensor &value =
    context.getInput(INOUT_INDEX::VALUE); // projected value
  nntrainer::Tensor &output =
    context.getOutput(INOUT_INDEX::OUTPUT); // output to be projected

  nntrainer::Tensor &cache_key =
    context.getTensor(tensor_idx[AttentionParams::cache_key]);
  nntrainer::Tensor &cache_value =
    context.getTensor(tensor_idx[AttentionParams::cache_value]);

  nntrainer::Tensor sink;
  if (use_sink) {
    sink = context.getWeight(sink_idx);
  }

  ml::train::TensorDim query_dim =
    query.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim key_dim =
    key.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim value_dim =
    value.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim output_dim =
    output.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_dim =
    cache_key.getDim(); // (B, 1, max_timestep, n_heads_KV * head_dim)
  ml::train::TensorDim cache_value_dim =
    cache_value.getDim(); // (B, 1, max_timestep, n_heads_KV * head_dim)

  ml::train::TensorDim query_step_dim =
    get_step_dim(query_dim); // (1, 1, step_size, n_heads_Q * head_dim)
  ml::train::TensorDim key_step_dim = get_step_dim(key_dim);
  ml::train::TensorDim value_step_dim = get_step_dim(value_dim);
  ml::train::TensorDim output_step_dim =
    get_step_dim(output_dim); // (1, 1, step_size, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_step_dim =
    get_step_dim(cache_key_dim); // (1, 1, step_size, n_heads_KV * head_dim)

  ml::train::TensorDim cache_value_step_dim =
    get_step_dim(cache_value_dim); // (1, 1, step_size, n_heads_KV * head_dim)

  unsigned int batch_size = query_dim.batch();
  // do the incremental forwarding
  for (unsigned int batch = 0; batch < batch_size; ++batch) {

    // preparing step tensors
    nntrainer::Tensor query_step = query.getSharedDataTensor(
      query_step_dim, batch * query_dim.getFeatureLen(), true);
    nntrainer::Tensor key_step = key.getSharedDataTensor(
      key_step_dim, batch * key_dim.getFeatureLen(), true);
    nntrainer::Tensor value_step = value.getSharedDataTensor(
      value_step_dim, batch * value_dim.getFeatureLen(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, batch * output_dim.getFeatureLen(), true);

    if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
#if ENABLE_FP16 && defined(__ANDROID__)
      nntrainer::TensorDim Q_step_dim = query_step_dim;
      nntrainer::TensorDim K_step_dim = key_step_dim;
      nntrainer::TensorDim V_step_dim = value_step_dim;
      nntrainer::TensorDim O_step_dim = output_step_dim;
      Q_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      K_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      V_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      O_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);

      nntrainer::Tensor Q_step = nntrainer::Tensor(Q_step_dim, true);
      nntrainer::Tensor K_step = nntrainer::Tensor(K_step_dim, true);
      nntrainer::Tensor V_step = nntrainer::Tensor(V_step_dim, true);
      nntrainer::Tensor O_step = nntrainer::Tensor(O_step_dim, true);

      Q_step.copyData(query_step);
      K_step.copyData(key_step);
      V_step.copyData(value_step);
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, _from, from, to, Q_step, K_step, V_step, O_step, cache_key,
          cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
          cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(batch, _from, from, to, Q_step, K_step,
                                         V_step, O_step, cache_key, cache_value,
                                         cache_key_dim, cache_key_step_dim,
                                         cache_value_dim, cache_value_step_dim);
      }
      output_step.copyData(O_step);
#else
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, _from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(
          batch, _from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim);
      }
#endif
    } else {
      one_batch_incremental_forwarding(
        batch, _from, from, to, query_step, key_step, value_step, output_step,
        cache_key, cache_value, cache_key_dim, cache_key_step_dim,
        cache_value_dim, cache_value_step_dim);
    }
  }

  // increase cache size
  cache_index += step_size;
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
void MHACoreLayer::compute_kcaches(nntrainer::Tensor &in,
                                   nntrainer::Tensor &cache,
                                   nntrainer::Tensor &out, unsigned int from,
                                   size_t sequence_len, unsigned int num_head,
                                   unsigned int group_size,
                                   unsigned int head_dim) {

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
          0, static_cast<size_t>(num_cache_head), [=, this](size_t head_kv) {
            compute_kcaches_fp32_reference(
              in_data, cache_data, out_data, row_to_compute, num_cache_head,
              head_dim, group_size, local_window_size, head_kv, head_kv + 1);
          });
      } else {
        const uint16_t *cache_data = cache.getData<uint16_t>();
        tm.parallel_for(0, static_cast<size_t>(num_cache_head),
                        [=, this](size_t head_kv) {
                          nntrainer::compute_kcaches<uint16_t>(
                            in_data, cache_data, out_data, row_to_compute,
                            num_cache_head, head_dim, group_size, tile_size,
                            local_window_size, head_kv, head_kv + 1);
                        });
      }

    } else {
      // Sequence processing (prefill or chunked)
      // Parallelize over the sequence length
      int seq =
        sequence_len < local_window_size ? sequence_len : local_window_size;
      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(seq), [=, this](size_t i) {
        float *input_addr = in.getData<float>() + num_head * head_dim * i;
        int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
        // Calculate dynamic offset for the output (triangle optimization)
        size_t out_start_row =
          is_causal ? calc_attn_index(from + i) - calc_attn_index(from)
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
#if defined(ENABLE_FP16) &&                                                    \
  (defined(__aarch64__) || defined(__ARM_ARCH_7A__) || defined(__ANDROID__) || \
   defined(__arm__) || defined(_M_ARM) || defined(_M_ARM64))
    if (sequence_len == 1) {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int num_rows = is_causal ? from + 1 : from + sequence_len;
      unsigned int num_cache_head = num_head / group_size;

      // Use ThreadManager for lower overhead parallelization during decoding
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
      unsigned int seq_start =
        sequence_len < local_window_size ? 0 : sequence_len - local_window_size;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(
        static_cast<size_t>(seq_start), static_cast<size_t>(sequence_len),
        [=](size_t i) {
          _FP16 *input_addr = in.getData<_FP16>() + num_head * head_dim * i;
          _FP16 *cache_addr = cache.getData<_FP16>();
          int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
          size_t out_start_row =
            is_causal ? calc_attn_index(from + i) - calc_attn_index(from)
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

void MHACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query_step,
  nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
  nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
  ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim) {

  /**
   *
   *  cache_key
   *  +------------------------------------------+
   *  |<--cache_index-->|<--b_cache_value_step-->|
   *  +------------------------------------------+
   *                    |<-------key_step------->|
   *  |<-------------b_cached_key--------------->|
   */

  // Load Input Tensors of this batch : b_ denotes a Tensor for this batch
  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + cache_index * cache_key_dim.width(),
    true);
  nntrainer::Tensor b_cache_value_step =
    cache_value.getSharedDataTensor(cache_value_step_dim,
                                    batch * cache_value_dim.getFeatureLen() +
                                      cache_index * cache_value_dim.width(),
                                    true);

  // append kcache with or without rotary embedding
  apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim, cache_index,
                             !use_rope);

  // append vcache without rotary embedding
  if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    apply_rotary_emb_tensor_v2(value_step, b_cache_value_step, head_dim,
                               cache_index, true);
  } else if (query_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    b_cache_value_step.copyData(value_step);
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }

  unsigned int step_size = to - from;
  bool is_prefill = !from || step_size > 1;
  if (skip_prefill && is_prefill)
    return;

  // apply rotary embedding for query
  if (use_rope) {
    apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, cache_index,
                               false);
  }

  /// @todo replace step_size into input height
  unsigned int cache_from = cache_index;
  unsigned int cache_to = cache_from + step_size;

  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  cached_key_dim.height(cache_to);
  cached_value_dim.height(cache_to);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  // Optional flash GEMM attention path. Handles both non-causal (encoder)
  // and causal-prefill paths, supports GQA and sliding window. Gated on a
  // minimum prefill length: for decode (step_size == 1) the per-row dot
  // path is preferred (no benefit from blocking + softmax bookkeeping).
  constexpr unsigned int FLASH_MIN_PREFILL = 32;
  if (use_gemm_attention && step_size >= FLASH_MIN_PREFILL) {
    gemm_attention(query_step, b_cached_key, b_cached_value,
                   attention_output_step, cache_to, step_size, cache_from);
    return;
  }

  // out_ stores the output of Q * K
  nntrainer::Tensor out_(
    1, 1,
    is_causal ? (calc_attn_index(cache_to) - calc_attn_index(cache_from))
              : (step_size * cache_to),
    num_heads_Q, query_step.getTensorType());

  compute_kcaches(query_step, b_cached_key, out_, cache_from,
                  cache_to - cache_from, num_heads_Q, gqa_size, head_dim);

  softmax_triangle(out_, step_size, num_heads_Q, cache_from);

  compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                cache_from, num_heads_KV, gqa_size, head_dim,
                                cache_to);
}

#if defined(__ARM_NEON)
#include <arm_neon.h>
// Cephes exp() for 4 floats at once (matches neon_mathfun.hxx exp_ps).
static inline float32x4_t vjepa_expq_f32(float32x4_t x) {
  const float32x4_t one = vdupq_n_f32(1.0f);
  x = vminq_f32(x, vdupq_n_f32(88.3762626647949f));
  x = vmaxq_f32(x, vdupq_n_f32(-88.3762626647949f));
  float32x4_t fx =
    vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(1.44269504088896341f));
  float32x4_t tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));
  uint32x4_t mask = vandq_u32(vcgtq_f32(tmp, fx), vreinterpretq_u32_f32(one));
  fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));
  x = vsubq_f32(x, vmulq_f32(fx, vdupq_n_f32(0.693359375f)));
  x = vsubq_f32(x, vmulq_f32(fx, vdupq_n_f32(-2.12194440e-4f)));
  float32x4_t z = vmulq_f32(x, x);
  float32x4_t y = vdupq_n_f32(1.9875691500E-4f);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(1.3981999507E-3f));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(8.3334519073E-3f));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(4.1665795894E-2f));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(1.6666665459E-1f));
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, vdupq_n_f32(5.0000001201E-1f));
  y = vmulq_f32(y, z);
  y = vaddq_f32(y, x);
  y = vaddq_f32(y, one);
  int32x4_t mm =
    vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(0x7f)), 23);
  return vmulq_f32(y, vreinterpretq_f32_s32(mm));
}
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

// Bulk convert N FP16-bits (uint16_t) values to FP32. Uses AVX2+F16C on x86
// (_mm256_cvtph_ps, available on Ivy Bridge+) and NEON fp16<->fp32 instructions
// on ARMv8.2+. Falls back to scalar nntrainer::compute_fp16_to_fp32. Treats the
// uint16 input as raw IEEE 754 half-precision bits — this is how the KV cache
// is stored regardless of ENABLE_FP16 build flag.
static inline void
mha_convert_fp16bits_to_fp32(unsigned int N, const uint16_t *src, float *dst) {
#if defined(__x86_64__) || defined(__i386__)
  unsigned int i = 0;
  for (; i + 16 <= N; i += 16) {
    __m256 a = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + i)));
    __m256 b = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + i + 8)));
    _mm256_storeu_ps(dst + i, a);
    _mm256_storeu_ps(dst + i + 8, b);
  }
  for (; i + 8 <= N; i += 8) {
    _mm256_storeu_ps(
      dst + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + i))));
  }
  for (; i < N; ++i)
    dst[i] = nntrainer::compute_fp16_to_fp32(src[i]);
#elif defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
  unsigned int i = 0;
  for (; i + 8 <= N; i += 8) {
    float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(src + i));
    vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(h)));
    vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(h)));
  }
  for (; i < N; ++i)
    dst[i] = nntrainer::compute_fp16_to_fp32(src[i]);
#else
  for (unsigned int i = 0; i < N; ++i)
    dst[i] = nntrainer::compute_fp16_to_fp32(src[i]);
#endif
}

#if defined(__x86_64__) || defined(__i386__)

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
static void mha_hsgemm_avx2(unsigned int M, unsigned int N,
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

#if !defined(__x86_64__) && !defined(__i386__) && defined(__ARM_NEON)
} // namespace causallm

// libnntrainer.so is built with ENABLE_FP16=1 and exports these symbols. The
// CausalLM app may be built with ENABLE_FP16=0, in which case cpu_backend.h
// hides them behind #ifdef. Re-declare here at global / ::nntrainer scope.
// - shgemm:         FP32 A × FP16 B -> FP32 C   (FP32 partial accumulation)
// - hgemm_classify: FP16 A × FP16 B -> FP32 C   (FP32 partial accumulation)
// - custom_hgemm:   FP16 A × FP16 B -> FP16 C   (FP32 partial accumulation,
//                                                FP16-stored result).
namespace nntrainer {
void shgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
            const unsigned int M, const unsigned int N, const unsigned int K,
            const float alpha, const float *A, const unsigned int lda,
            const __fp16 *B, const unsigned int ldb, const float beta, float *C,
            const unsigned int ldc);
namespace neon {
void custom_hgemm(const __fp16 *A, const __fp16 *B, __fp16 *C, uint32_t M,
                  uint32_t N, uint32_t K, float alpha, float beta, bool TransA,
                  bool TransB);
} // namespace neon
} // namespace nntrainer
void hgemm_classify(const __fp16 *A, const __fp16 *B, float *C32,
                    unsigned int M, unsigned int N, unsigned int K, float alpha,
                    float beta, bool TransA, bool TransB);

namespace causallm {
#endif

#if !defined(__x86_64__) && !defined(__i386__) && defined(__ARM_NEON)
// QK micro-kernel for the FP16-query + FP32-score attention path.
// Computes S[m, n] = alpha * sum_k A[m, k] * B[n, k]   (TransB-style dot)
// using ARMv8.2-A FMLAL (vfmlalq_low/high_f16): each pair of intrinsics
// widens 8 FP16 products into two FP32 accumulators, so the per-element
// product is computed in FP32 from the start. This avoids the FP16-product
// overflow that custom_hgemm hits when packing wide encoder logits
// (V-JEPA-2 block 0 Q,K magnitudes ~400 -> products ~160k > FP16 max
// 65504), and unlike a cast-up-Q+shgemm path it never materialises an
// FP32 copy of Q.
//
// Layout: A row-major (M rows, lda cols), B row-major (N rows, ldb cols),
// C row-major (M rows, ldc cols). Inner length K is the dot length and
// must match both A's and B's stride argument (i.e. lda == ldb == K when
// the rows are contiguous, which is how Phase 1 of gemm_attention packs).
//
// Requires FEAT_FHM (asimdfhm in /proc/cpuinfo). Confirmed on Cortex-A78/X4
// derivatives used by S25/S26 Ultra; the target attribute pulls the
// fp16fml ISA extension in only for this function so the rest of the TU
// can stay on the build-wide -march flags.
__attribute__((target(
  "arch=armv8.2-a+fp16+fp16fml+dotprod+i8mm"))) static inline void
mha_qk_fmlal_f16xf16_to_f32(
  const __fp16 *A, const __fp16 *B, float *C, unsigned int M, unsigned int N,
  unsigned int K, float alpha, unsigned int lda, unsigned int ldb,
  unsigned int ldc) {
  for (unsigned int m = 0; m < M; ++m) {
    const __fp16 *a_row = A + (size_t)m * lda;
    float *c_row = C + (size_t)m * ldc;
    for (unsigned int n = 0; n < N; ++n) {
      const __fp16 *b_row = B + (size_t)n * ldb;
      float32x4_t acc0 = vdupq_n_f32(0.0f);
      float32x4_t acc1 = vdupq_n_f32(0.0f);
      unsigned int k = 0;
      // Process 16 lanes per iter (two 8-lane fmlal pairs) to keep the
      // FP32 pipelines busy on Cortex-A76 (FMLAL latency ~4, two FP units).
      for (; k + 16 <= K; k += 16) {
        float16x8_t a0 = vld1q_f16(a_row + k);
        float16x8_t b0 = vld1q_f16(b_row + k);
        float16x8_t a1 = vld1q_f16(a_row + k + 8);
        float16x8_t b1 = vld1q_f16(b_row + k + 8);
        acc0 = vfmlalq_low_f16(acc0, a0, b0);
        acc1 = vfmlalq_high_f16(acc1, a0, b0);
        acc0 = vfmlalq_low_f16(acc0, a1, b1);
        acc1 = vfmlalq_high_f16(acc1, a1, b1);
      }
      // 8-lane tail.
      if (k + 8 <= K) {
        float16x8_t a0 = vld1q_f16(a_row + k);
        float16x8_t b0 = vld1q_f16(b_row + k);
        acc0 = vfmlalq_low_f16(acc0, a0, b0);
        acc1 = vfmlalq_high_f16(acc1, a0, b0);
        k += 8;
      }
      float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
      // <8 tail (head_dim is usually a multiple of 8, but be safe).
      for (; k < K; ++k)
        sum += (float)a_row[k] * (float)b_row[k];
      c_row[n] = alpha * sum;
    }
  }
}
#endif

void MHACoreLayer::gemm_attention(nntrainer::Tensor &query_step,
                                  nntrainer::Tensor &b_cached_key,
                                  nntrainer::Tensor &b_cached_value,
                                  nntrainer::Tensor &attention_output_step,
                                  unsigned int N_kv, unsigned int N_q,
                                  unsigned int cache_from) {
  const unsigned int d = head_dim;
  const unsigned int HD_Q = num_heads_Q * d;
  const unsigned int HD_KV = num_heads_KV * d;
  const unsigned int gqa =
    (num_heads_KV > 0) ? static_cast<unsigned int>(num_heads_Q / num_heads_KV)
                       : 1u;
  const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(d));
  const unsigned int order =
    static_cast<unsigned int>(query_step.getDim().getStorageOrder());
  const bool causal = is_causal;
  // Treat any local_window_size >= cache length as "no window".
  const bool windowed = (local_window_size < N_kv);
  const size_t W = static_cast<size_t>(local_window_size);

  // Runtime dtype dispatch: forwarding() may convert Q/V/output to FP16 when
  // ENABLE_FP16 && __ANDROID__ build. K/V are always FP16 storage.
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

  // tile sizes (cache-resident S); overridable via env for tuning
  unsigned int Bq = 256, Bk = 512;
  if (const char *e = std::getenv("VJEPA_BQ"))
    Bq = static_cast<unsigned int>(std::stoul(e));
  if (const char *e = std::getenv("VJEPA_BK"))
    Bk = static_cast<unsigned int>(std::stoul(e));

  const unsigned int num_qb = (N_q + Bq - 1) / Bq;
  auto &tm = nntrainer::ThreadManager::Global();

  // Cache always stores half-precision (FP16-bit) values; read as raw uint16_t
  // bits so we don't depend on ENABLE_FP16 / _FP16 / _Float16 being defined.
  const uint16_t *Kbase;
  const uint16_t *Vbase;
#ifdef ENABLE_FP16
  Kbase = reinterpret_cast<const uint16_t *>(b_cached_key.getData<_FP16>());
  Vbase = reinterpret_cast<const uint16_t *>(b_cached_value.getData<_FP16>());
#else
  Kbase = b_cached_key.getData<uint16_t>();
  Vbase = b_cached_value.getData<uint16_t>();
#endif

  // Phase 1: de-interleave heads once into shared contiguous buffers.
  // K/V always kept as raw FP16 bits (uint16). Q either FP32 (V-JEPA
  // path) or FP16 (when forwarding() pre-converts to FP16; ENABLE_FP16+
  // Android). The FP16 Q path keeps the entire attention in FP16
  // (custom_hgemm for QK and AV, FP16 softmax) without ever materializing
  // an FP32 score buffer.
  std::vector<float> Qa_fp32;
  std::vector<uint16_t> Qa_fp16;
  if (q_fp16)
    Qa_fp16.resize((size_t)num_heads_Q * N_q * d);
  else
    Qa_fp32.resize((size_t)num_heads_Q * N_q * d);
  std::vector<uint16_t> Ka((size_t)num_heads_KV * N_kv * d);
  std::vector<uint16_t> Va((size_t)num_heads_KV * N_kv * d);
  {
    if (q_fp16) {
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

  // Phase 2: flash attention over balanced (h_q, query-block) work units.
  tm.parallel_for(0, static_cast<size_t>(num_heads_Q) * num_qb, [&](size_t u) {
    const unsigned int h_q = static_cast<unsigned int>(u / num_qb);
    const unsigned int h_kv = h_q / gqa;
    const unsigned int qb = static_cast<unsigned int>(u % num_qb) * Bq;
    const unsigned int bq = std::min(Bq, N_q - qb);
    const float *Qp_fp32 =
      q_fp16 ? nullptr : (Qa_fp32.data() + (size_t)h_q * N_q * d);
    const uint16_t *Qp_fp16 =
      q_fp16 ? (Qa_fp16.data() + (size_t)h_q * N_q * d) : nullptr;
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
#if !defined(__x86_64__) && !defined(__i386__) && defined(__ARM_NEON)
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

      {
        // QK -> FP32 score buffer S. One buffer, two query sources:
        //   - q_fp16:  FP16 Q × FP16 K via FMLAL-widening — every product is
        //     accumulated in FP32 (vfmlalq_low/high_f16), so V-JEPA-2 block-0's
        //     ~160k per-element products and ~457k logits never overflow FP16.
        //     No FP32 copy of Q is materialized.
        //   - !q_fp16: FP32 Q × FP16 K via shgemm / avx2 / sgemm.
        // The softmax below reads S (FP32) and stores normalized FP16 probs to
        // Sp16 for the AV custom_hgemm.
#if defined(__x86_64__) || defined(__i386__)
        mha_hsgemm_avx2(bq, bk, d, inv_sqrt, Qp_fp32 + (size_t)qb * d, d,
                        Kp + (size_t)kb * d, d, /*TransB=*/true, S.data(), bk);
#elif defined(__ARM_NEON)
        if (q_fp16) {
          mha_qk_fmlal_f16xf16_to_f32(
            reinterpret_cast<const __fp16 *>(Qp_fp16 + (size_t)qb * d),
            reinterpret_cast<const __fp16 *>(Kp + (size_t)kb * d), S.data(), bq,
            bk, d, inv_sqrt, d, d, bk);
        } else {
          nntrainer::shgemm(
            order, false, true, bq, bk, d, inv_sqrt, Qp_fp32 + (size_t)qb * d,
            d, reinterpret_cast<const __fp16 *>(Kp + (size_t)kb * d), d, 0.0f,
            S.data(), bk);
        }
#else
        nntrainer::sgemm(order, false, true, bq, bk, d, inv_sqrt,
                         Qp_fp32 + (size_t)qb * d, d, Kp + (size_t)kb * d, d,
                         0.0f, S.data(), bk);
#endif

        for (unsigned int i = 0; i < bq; ++i) {
          float *s = S.data() + (size_t)i * bk;
          const long long q_abs = (long long)cache_from + qb + i;
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
#if !defined(__x86_64__) && !defined(__i386__) && defined(__ARM_NEON)
          {
            uint16_t *sp16 = Sp16.data() + (size_t)i * bk;
            float32x4_t vsum = vdupq_n_f32(0.0f), vnm = vdupq_n_f32(nm);
            unsigned int k = 0;
            for (; k + 4 <= bk; k += 4) {
              float32x4_t e = vjepa_expq_f32(vsubq_f32(vld1q_f32(s + k), vnm));
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
                  vjepa_expq_f32(vsubq_f32(vld1q_f32(s + k), vnm));
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

#if defined(__x86_64__) || defined(__i386__)
        mha_hsgemm_avx2(bq, d, bk, 1.0f, S.data(), bk, Vp + (size_t)kb * d, d,
                        /*TransB=*/false, Pacc.data(), d);
        for (unsigned int i = 0; i < bq; ++i) {
          float *ol = Ol.data() + (size_t)i * d;
          const float *pa = Pacc.data() + (size_t)i * d;
          for (unsigned int x = 0; x < d; ++x)
            ol[x] += pa[x];
        }
#elif defined(__ARM_NEON)
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
          nntrainer::sgemm(order, false, false, bq, d, bk, 1.0f, S.data(), bk,
                           Vp + (size_t)kb * d, d, 0.0f, Pacc.data(), d);
          for (unsigned int i = 0; i < bq; ++i) {
            float *ol = Ol.data() + (size_t)i * d;
            const float *pa = Pacc.data() + (size_t)i * d;
            for (unsigned int x = 0; x < d; ++x)
              ol[x] += pa[x];
          }
#endif
      } // unified QK (fmlal/shgemm) -> FP32 S -> softmax -> AV
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

void MHACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query_step,
  nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
  nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
  ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim, nntrainer::Tensor &sink_step) {
  /// @todo replace from, to into cache_index, input height
  /// @note currently, only gpt-oss uses this method

  /**
   *  cache_key
   *  +--------+                        ->
   *  |        |                        ->
   *  |        |                        ->
   *  |........| from                   ->
   *  |........| to -> b_cache_key_step -> b_cached_key
   *  |        |
   *  +--------+
   *
   */

  /** 1. Load Input Tensors of this batch : b_ denotes a Tensor for this batch
   * **/
  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(), true);
  nntrainer::Tensor b_cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim,
    batch * cache_value_dim.getFeatureLen() + from * cache_value_dim.width(),
    true);

  if (use_rope) {
    apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, _from, false);
  }

  apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim, _from,
                             !use_rope);

  if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    apply_rotary_emb_tensor_v2(value_step, b_cache_value_step, head_dim, _from,
                               true);
  } else if (query_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    b_cache_value_step.copyData(value_step);
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }

  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  cached_key_dim.height(to);
  cached_value_dim.height(to);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  nntrainer::Tensor out_(
    1, 1,
    is_causal
      ? (((to - from) == 1) ? to : calc_attn_index(to) - calc_attn_index(from))
      : ((to - from) * to),
    num_heads_Q, query_step.getTensorType());

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  compute_kcaches(query_step, b_cached_key, out_, _from, to - from, num_heads_Q,
                  gqa_size, head_dim);

  softmax_triangle(out_, to - from, num_heads_Q, from, sink_step);

  compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                from, num_heads_KV, gqa_size, head_dim, to);
}

/************************************************************** */

/**
 * @brief rotary embedding-related member function
 * @note seq_len -> max_position_embeddings
 */
void MHACoreLayer::precompute_freqs(int head_dim, unsigned int seq_len,
                                    float theta, bool is_fp16) {
  const std::string rope_cache_key = getRopeCacheKey(head_dim, seq_len, theta);
  thetas.clear();
  if (rope_scaling_type == "default")
    _compute_default_parameters(head_dim, theta);
  else if (rope_scaling_type == "yarn")
    _compute_yarn_parameters(head_dim, theta);
  else if (rope_scaling_type == "proportional")
    _compute_proportional_parameters(head_dim, theta);
  else
    NNTR_THROW_IF(true, std::invalid_argument) << "Unsupported rope type!";

  unsigned int half_ = head_dim / 2;

  if (!is_fp16) {
    auto it = rope_cache_fp32.find(rope_cache_key);
    if (it != rope_cache_fp32.end()) {
      freqs_fp32 = it->second;
      return;
    }

    auto cached = std::make_shared<RopeCacheFP32>();
    cached->cos.assign(seq_len, std::vector<float>(head_dim, 0));
    cached->sin.assign(seq_len, std::vector<float>(head_dim, 0));

    for (unsigned int i = 0; i < seq_len; ++i) {
#ifdef USE_NEON
      nntrainer::calc_trigonometric_vals_dup(
        half_, thetas.data(), cached->cos[i].data(), cached->sin[i].data(), i,
        attention_scaling);
#else
      for (unsigned int j = 0; j < half_; ++j) {
        float angle = i * thetas[j];
        cached->cos[i][j] = std::cos(angle) * attention_scaling;
        cached->cos[i][j + half_] = std::cos(angle) * attention_scaling;

        cached->sin[i][j] = std::sin(angle) * attention_scaling;
        cached->sin[i][j + half_] = std::sin(angle) * attention_scaling;
      }
#endif
    }
    rope_cache_fp32[rope_cache_key] = cached;
    freqs_fp32 = cached;
  }

#ifdef ENABLE_FP16
  if (is_fp16) {
    auto it = rope_cache_fp16.find(rope_cache_key);
    if (it != rope_cache_fp16.end()) {
      freqs_fp16 = it->second;
      return;
    }

    auto cached = std::make_shared<RopeCacheFP16>();
    cached->cos.assign(seq_len, std::vector<_FP16>(head_dim, 0));
    cached->sin.assign(seq_len, std::vector<_FP16>(head_dim, 0));

    std::vector<float> cos_tmp(head_dim);
    std::vector<float> sin_tmp(head_dim);

    for (unsigned int i = 0; i < seq_len; ++i) {
#ifdef USE_NEON
      nntrainer::calc_trigonometric_vals_dup(half_, thetas.data(),
                                             cos_tmp.data(), sin_tmp.data(), i,
                                             attention_scaling);
#else
      for (unsigned int j = 0; j < half_; ++j) {
        float angle = i * thetas[j];
        cos_tmp[j] = std::cos(angle) * attention_scaling;
        cos_tmp[j + half_] = std::cos(angle) * attention_scaling;

        sin_tmp[j] = std::sin(angle) * attention_scaling;
        sin_tmp[j + half_] = std::sin(angle) * attention_scaling;
      }
#endif
      for (unsigned int j = 0; j < head_dim; ++j) {
        cached->cos[i][j] = (_FP16)cos_tmp[j];
        cached->sin[i][j] = (_FP16)sin_tmp[j];
      }
    }
    rope_cache_fp16[rope_cache_key] = cached;
    freqs_fp16 = cached;
  }
#endif
}

std::string MHACoreLayer::getRopeCacheKey(int head_dim, unsigned int seq_len,
                                          float theta) const {
  std::ostringstream ss;
  ss << rope_scaling_type << "|" << head_dim << "|" << seq_len << "|" << theta
     << "|" << scale << "|" << rope_partial_rotary_factor << "|"
     << original_max_position_embeddings;
  return ss.str();
}

void MHACoreLayer::_compute_default_parameters(int head_dim, float theta) {

  // no attention scaling
  attention_scaling = 1.0f;

  // theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... , dim/2]
  // head_dim should be divisible by 2
  unsigned int half_ = head_dim / 2;
  for (unsigned int i = 0; i < half_; ++i) {
    thetas.push_back(1.0 /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }
}

void MHACoreLayer::_compute_proportional_parameters(int head_dim, float theta) {
  attention_scaling = 1.0f;
  const int half_dim = static_cast<int>(head_dim / 2);
  const int rope_angles =
    static_cast<int>((rope_partial_rotary_factor * head_dim) / 2.0f);

  thetas.reserve(half_dim);
  for (int i = 0; i < rope_angles; ++i) {
    thetas.push_back(1.0f /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }

  for (int i = rope_angles; i < half_dim; ++i) {
    thetas.push_back(0.0f);
  }

  for (auto &val : thetas) {
    val /= scale;
  }
}

void MHACoreLayer::_compute_yarn_parameters(int head_dim, float theta) {

  // Config parameters
  ///@todo partial_rotary_factor should be generalized to fully support
  /// transformers's implementation
  // const float partial_rotary_factor = has_partial_rotary_factor ?
  // config_partial_rotary_factor : 1.0f;
  const float partial_rotary_factor = 1.0f;
  const int dim = static_cast<int>(head_dim * partial_rotary_factor);
  const float base = theta;

  // Handle max position embeddings

  // Attention scaling calculation (simplified from Python version)
  auto get_mscale = [](float scale, float mscale = 1.0f) {
    return (scale <= 1.0f) ? 1.0f : (0.1f * mscale * std::log(scale) + 1.0f);
  };

  ///@todo attention_scaling should be generalized to fully support
  /// transformers's implementation
  // if (has_mscale && has_mscale_all_dim) {
  // attention_scaling = get_mscale(factor, mscale) / get_mscale(factor,
  // mscale_all_dim);
  // } else {
  // attention_scaling = get_mscale(factor);
  // }
  attention_scaling = get_mscale(scale);

  ///@todo attention_scaling should be generalized to fully support
  /// transformers's implementation
  // const float beta_fast = has_beta_fast ? config_beta_fast : 32.0f;
  // const float beta_slow = has_beta_slow ? config_beta_slow : 1.0f;
  // const bool truncate = has_truncate ? config_truncate : true;
  // Beta parameters
  const float beta_fast = 32.0f;
  const float beta_slow = 1.0f;
  const bool truncate = false;

  // Helper functions
  auto find_correction_dim = [&](float num_rotations) {
    return (dim * std::log(original_max_position_embeddings /
                           (num_rotations * 2 * M_PI))) /
           (2 * std::log(base));
  };

  auto [low, high] = [&]() {
    float low_val = find_correction_dim(beta_fast);
    float high_val = find_correction_dim(beta_slow);
    if (truncate) {
      low_val = std::floor(low_val);
      high_val = std::ceil(high_val);
    }
    return std::make_pair(low_val, high_val);
  }();

  // Compute position frequencies
  thetas.resize(dim / 2);

  // Compute interpolation and extrapolation frequencies
  std::vector<float> inv_freq_interpolation;
  std::vector<float> inv_freq_extrapolation;
  for (size_t i = 0; i < dim / 2; ++i) {
    inv_freq_extrapolation.push_back(
      1.0 / (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
    inv_freq_interpolation.push_back(
      1.0 / (scale * std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }

  auto linear_ramp_factor = [](float min, float max, int size) {
    if (min == max) {
      max += 0.001f; // Prevent singularity
    }
    std::vector<float> ramp(size);
    for (int i = 0; i < size; ++i) {
      float val = (i - min) / (max - min);
      ramp[i] = std::clamp(val, 0.0f, 1.0f);
    }
    return ramp;
  };

  std::vector<float> inv_freq_extrapolation_factor =
    linear_ramp_factor(low, high, dim / 2);
  for (auto &val : inv_freq_extrapolation_factor) {
    val = 1.0f - val;
  }

  // Combine frequencies
  for (size_t i = 0; i < thetas.size(); ++i) {
    thetas[i] =
      inv_freq_extrapolation[i] * inv_freq_extrapolation_factor[i] +
      inv_freq_interpolation[i] * (1.0f - inv_freq_extrapolation_factor[i]);
  }
}

void MHACoreLayer::apply_rotary_emb_tensor_v2(nntrainer::Tensor &in,
                                              nntrainer::Tensor &out,
                                              unsigned int dim,
                                              unsigned int from,
                                              bool convert_only) {
  if (!use_rope) {
    if (&in != &out) {
      if (in.getDataType() == ml::train::TensorDim::DataType::FP32 &&
          out.getDataType() == ml::train::TensorDim::DataType::UINT16) {
        NNTR_THROW_IF(in.size() != out.size(), std::invalid_argument)
          << "Size of tensor to copy must match";
        const float *in_ptr = in.getData<float>();
        uint16_t *out_ptr = out.getData<uint16_t>();
        for (size_t i = 0; i < in.size(); ++i)
          out_ptr[i] = nntrainer::compute_fp32_to_fp16(in_ptr[i]);
      } else {
        out.copyData(in);
      }
    }
    return;
  }
  unsigned int half_ = dim / 2;
  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (freqs_fp32 == nullptr) {
      const std::lock_guard<std::mutex> lock(rope_init_mtx);
      if (freqs_fp32 == nullptr) {
        precompute_freqs(head_dim, max_position_embeddings, theta, false);
      }
    }
    std::vector<float> *cos_ = nullptr;
    std::vector<float> *sin_ = nullptr;

    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &freqs_fp32->cos[from + h];
            sin_ = &freqs_fp32->sin[from + h];
          }
          float *in_ptr = in.getData<float>() +
                          b * in.channel() * in.height() * in.width() +
                          c * in.height() * in.width() + h * in.width();

          if (out.getDataType() == ml::train::TensorDim::DataType::FP32) {
            float *out_ptr = out.getData<float>() +
                             b * out.channel() * out.height() * out.width() +
                             c * out.height() * out.width() + h * out.width();

            if (out_ptr != in_ptr) {
              std::memcpy(out_ptr, in_ptr, sizeof(float) * in.width());
            }
            if (!convert_only) {
              nntrainer::compute_rotary_emb_value(
                in.width(), dim, half_, out_ptr, nullptr, cos_->data(),
                sin_->data(), false);
            }
          } else if (out.getDataType() ==
                       ml::train::TensorDim::DataType::UINT16 ||
                     out.getDataType() ==
                       ml::train::TensorDim::DataType::FP16) {
            uint16_t *out_ptr = out.getData<uint16_t>() +
                                b * out.channel() * out.height() * out.width() +
                                c * out.height() * out.width() +
                                h * out.width();

            nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                                out_ptr, cos_->data(),
                                                sin_->data(), convert_only);
          }
        }
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (freqs_fp16 == nullptr) {
      const std::lock_guard<std::mutex> lock(rope_init_mtx);
      if (freqs_fp16 == nullptr) {
        precompute_freqs(head_dim, max_position_embeddings, theta, true);
      }
    }
    std::vector<_FP16> *cos_ = nullptr;
    std::vector<_FP16> *sin_ = nullptr;

    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &freqs_fp16->cos[from + h];
            sin_ = &freqs_fp16->sin[from + h];
          }
          _FP16 *in_ptr = in.getData<_FP16>() +
                          b * in.channel() * in.height() * in.width() +
                          c * in.height() * in.width() + h * in.width();
          _FP16 *out_ptr = out.getData<_FP16>() +
                           b * out.channel() * out.height() * out.width() +
                           c * out.height() * out.width() + h * out.width();

          nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                              out_ptr, cos_->data(),
                                              sin_->data());
        }
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
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
      int seq = row < local_window_size ? row : local_window_size;
      if (!is_causal)
        seq = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(seq), [=, this](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row = calc_attn_index(from + i) - calc_attn_index(from);
          end_row = calc_attn_index(from + i + 1) - calc_attn_index(from);
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
      int seq = row < local_window_size ? row : local_window_size;
      if (!is_causal)
        seq = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(seq), [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row = calc_attn_index(from + i) - calc_attn_index(from);
          end_row = calc_attn_index(from + i + 1) - calc_attn_index(from);
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

void MHACoreLayer::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
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
      int seq = row < local_window_size ? row : local_window_size;
      if (!is_causal)
        seq = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(seq), [=, this](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row = calc_attn_index(i + from) - calc_attn_index(from);
          end_row = calc_attn_index(from + i + 1) - calc_attn_index(from);
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
      int seq = row < local_window_size ? row : local_window_size;
      if (!is_causal)
        seq = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(seq), [=](size_t i) {
        size_t start_row = calc_attn_index(i + from) - calc_attn_index(from);
        size_t end_row = calc_attn_index(from + i + 1) - calc_attn_index(from);
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head,
                               sink_step_);
      });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::compute_fp16vcache_transposed(
  nntrainer::Tensor &in, nntrainer::Tensor &vcache, nntrainer::Tensor &output,
  int from, int num_cache_head, int gqa_size, int head_dim, int to) {

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if ((to - from) != 1) {
      int seq = (to - from) < local_window_size ? to - from : local_window_size;
      // if non-causal, seq is practically to - from.
      if (!is_causal)
        seq = to - from;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(seq), [=, this](size_t i) {
        size_t start_idx;
        if (is_causal) {
          start_idx = calc_attn_index(to - seq + i) - calc_attn_index(to - seq);
        } else {
          start_idx = i * to; // linear index
        }
        const float *input =
          in.getData<float>() + start_idx * num_cache_head * gqa_size;
        float *out =
          output.getData<float>() + i * (num_cache_head * gqa_size * head_dim);

        int row_num = is_causal ? (to - seq + i) : to - 1;
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
          0, static_cast<size_t>(num_cache_head), [=, this](size_t head_kv) {
            compute_vcache_fp32_transposed_reference(
              row_num, in_data, vcache_data, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
          });
      } else {
        const uint16_t *vcache_data = vcache.getData<uint16_t>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=, this](size_t head_kv) {
            nntrainer::compute_fp16vcache_fp32_transposed(
              row_num, in_data, vcache_data, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
          });
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#if defined(ENABLE_FP16) &&                                                    \
  (defined(__aarch64__) || defined(__ARM_ARCH_7A__) || defined(__ANDROID__) || \
   defined(__arm__) || defined(_M_ARM) || defined(_M_ARM64))
    if ((to - from) != 1) {
      int seq = (to - from) < local_window_size ? to - from : local_window_size;
      if (!is_causal)
        seq = to - from;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(seq), [=](size_t i) {
        size_t start_idx;
        if (is_causal) {
          start_idx = calc_attn_index(to - seq + i) - calc_attn_index(to - seq);
        } else {
          start_idx = i * to;
        }
        const _FP16 *input =
          in.getData<_FP16>() + start_idx * num_cache_head * gqa_size;
        _FP16 *out =
          output.getData<_FP16>() + i * (num_cache_head * gqa_size * head_dim);
        int row_num = is_causal ? (to - seq + i) : to - 1;
        nntrainer::compute_fp16vcache_transposed(
          row_num, input, vcache.getData<_FP16>(), out, num_cache_head,
          gqa_size, head_dim, local_window_size);
      });
    } else {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_num = to - 1;

      // Use OpenMP for lower overhead parallelization during decoding
      const _FP16 *in_data = in.getData<_FP16>();
      const _FP16 *vcache_data = vcache.getData<_FP16>();
      _FP16 *output_data = output.getData<_FP16>();

      auto &tm_fp16 = nntrainer::ThreadManager::Global();
      tm_fp16.parallel_for(
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

void MHACoreLayer::setBatch(nntrainer::RunLayerContext &context,
                            unsigned int batch) {

  const float dropout_rate =
    std::get<nntrainer::props::DropOutRate>(mha_core_props).get();
  context.updateTensor(tensor_idx[AttentionParams::cache_key], batch);
  context.updateTensor(tensor_idx[AttentionParams::cache_value], batch);
  // context.updateTensor(tensor_idx[AttentionParams::attention_weight], batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(tensor_idx[AttentionParams::dropout_mask], batch);
  }
}

void MHACoreLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  unsigned int height = input_dimensions[0].height();
  unsigned int &max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();
  unsigned int &max_new_tokens =
    std::get<props::MaxNewTokens>(mha_core_props).get();
  max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mha_core_props).get();
  max_timestep = height + max_new_tokens;

  ml::train::TensorDim kv_dim = input_dimensions[0];
  kv_dim.width(kv_dim.width() / (num_heads_Q / num_heads_KV));

  ml::train::TensorDim kv_cache_dim = kv_dim;
#ifdef ENABLE_FP16
  kv_cache_dim.setDataType(ml::train::TensorDim::DataType::FP16);
#else
  kv_cache_dim.setDataType(ml::train::TensorDim::DataType::UINT16);
#endif
  kv_cache_dim.height(max_timestep);

  context.updateInput(INOUT_INDEX::QUERY, input_dimensions[0]);
  context.updateInput(INOUT_INDEX::KEY, kv_dim);
  context.updateInput(INOUT_INDEX::VALUE, kv_dim);
  context.updateOutput(0, input_dimensions[0]);

  context.updateTensor(tensor_idx[AttentionParams::cache_key], kv_cache_dim);
  context.updateTensor(tensor_idx[AttentionParams::cache_value], kv_cache_dim);
}

void MHACoreLayer::calcDerivative(nntrainer::RunLayerContext &context) {}

void MHACoreLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void MHACoreLayer::exportTo(nntrainer::Exporter &exporter,
                            const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(mha_core_props, method, this);
}

void MHACoreLayer::setProperty(const std::vector<std::string> &values) {
  std::vector<std::string> props;
  props.reserve(values.size());
  for (const auto &value : values) {
    std::string key;
    std::string parsed_value;
    if (nntrainer::getKeyValue(value, key, parsed_value) == ML_ERROR_NONE &&
        key == "cache_index") {
      setCacheIndex(static_cast<unsigned int>(std::stoul(parsed_value)));
    } else {
      props.push_back(value);
    }
  }

  auto remain_props = loadProperties(props, mha_core_props);
  LayerImpl::setProperty(remain_props);
}

size_t MHACoreLayer::calc_attn_index(size_t i) { return (i * (i + 1)) / 2; };

#ifdef PLUGGABLE

nntrainer::Layer *create_mha_core_layer() {
  auto layer = new MHACoreLayer();
  return layer;
}

void destroy_mha_core_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_mha_core_layer,
                                                   destroy_mha_core_layer};
}

#endif

} // namespace causallm

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   mha_core.cpp
 * @date   11 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This code is based on custom_multi_head_attention_layer.cpp.
 *         This code is a part of the break down version of the mha layer.
 */
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include <engine.h>
#include <fp16.h>
#include <layer_context.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <node_exporter.h>

#include <cstdint>

inline float convert_scalar(uint16_t h) {
  return nntrainer::compute_fp16_to_fp32(h);
}

namespace causallm {

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
    props::MaxPositionEmbeddings(), props::UseSink(), props::RopeScalingType(),
    props::RopeScalingFactor(), props::RopeScalingMaxPositionEmbeddings()),
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

  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 4,
                std::invalid_argument)
    << "Multi head Attention layer needs 3 or 4 inputs. (query, key, value and "
       "mask is optional)";
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
  unsigned int max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mha_core_props).get();

  /** local window size */
  local_window_size = std::get<props::SlidingWindow>(mha_core_props).get();

  /** attention scaling computation */
  rope_scaling_type = std::get<props::RopeScalingType>(mha_core_props).get();
  scale = std::get<props::RopeScalingFactor>(mha_core_props).get();
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

  /** Tensor for KV-Cache */
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

  theta = (float)std::get<props::RopeTheta>(mha_core_props).get();

  /** precompute_freqs will be invoked only once */
  if (freqs_cos == nullptr)
    precompute_freqs(head_dim, max_position_embeddings, theta);

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
 * @note This forwarding function is used for training mode.
 *       This will be implemented ASAP.
 * @date 2024-09-02
 */
void MHACoreLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {}

/**
 * @note This incremental_forwarding method is invoked for inference mode.
 *       Please note that Transformer Decoder's MHA takes only one sequence at a
 * step. Incremental forwarding function is used for this.
 */
void MHACoreLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int _from, unsigned int _to,
                                          bool training) {

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
      // exceeds the kv_cache size
      // KV_cache is shifted!
      cache_shift = true;
      from = max_timestep - 1;
      to = max_timestep;
    }
  }

  // util fn to compute tensor dimension for one step.
  auto get_step_dim = [to, from](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(to - from); // One is expected.
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

  const unsigned int num_heads_Q =
    std::get<nntrainer::props::NumHeads>(mha_core_props).get();

  ml::train::TensorDim query_dim =
    query.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim key_dim =
    key.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim value_dim =
    value.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim output_dim =
    output.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_dim =
    cache_key.getDim(); // (B, 1, max_seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim cache_value_dim =
    cache_value.getDim(); // (B, 1, max_seq_len, n_heads_KV * head_dim)

  ml::train::TensorDim query_step_dim =
    get_step_dim(query_dim); // (B, 1, from-to, n_heads_Q * head_dim)
  ml::train::TensorDim key_step_dim = get_step_dim(key_dim);
  ml::train::TensorDim value_step_dim = get_step_dim(value_dim);
  ml::train::TensorDim output_step_dim =
    get_step_dim(output_dim); // (B, 1, from-to, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_step_dim =
    get_step_dim(cache_key_dim); // (B, 1, from-to, n_heads_KV * head_dim)

  ml::train::TensorDim cache_value_step_dim =
    get_step_dim(cache_value_dim); // (B, 1, from-to, n_heads_KV * head_dim)

  unsigned int batch_size = (_from) ? 1 : query_dim.batch();
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

  if (!_from) {
    batch_size = query_dim.batch();
    nntrainer::Tensor cache_key_0_step =
      cache_key.getSharedDataTensor(cache_key_step_dim, 0, true);
    nntrainer::Tensor cache_value_0_step =
      cache_value.getSharedDataTensor(cache_value_step_dim, 0, true);

    for (unsigned int batch = 1; batch < batch_size; ++batch) {
      nntrainer::Tensor cache_key_nth_step = cache_key.getSharedDataTensor(
        cache_key_step_dim,
        batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(),
        true);
      nntrainer::Tensor cache_value_nth_step =
        cache_key.getSharedDataTensor(cache_value_step_dim,
                                      batch * cache_value_dim.getFeatureLen() +
                                        from * cache_value_dim.width(),
                                      true);

      cache_key_nth_step.copyData(cache_key_0_step);
      cache_key_nth_step.copyData(cache_value_0_step);
    }
  }
}

void MHACoreLayer::compute_kcaches(
  nntrainer::Tensor &in, nntrainer::Tensor &cache, nntrainer::Tensor &out,
  unsigned int from, size_t sequence_len, unsigned int num_head,
  unsigned int group_size, unsigned int head_dim, BS::thread_pool<> &pool) {

  int tile_size = 8;

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (sequence_len == 1) {
      nntrainer::compute_kcaches<uint16_t>(
        in.getData<float>(), cache.getData<uint16_t>(), out.getData<float>(),
        from + 1, num_head / group_size, head_dim, group_size, tile_size,
        local_window_size);
    } else {
      std::vector<std::future<void>> futures;
      int seq =
        sequence_len < local_window_size ? sequence_len : local_window_size;

      for (int i = 0; i < seq; ++i) {
        float *input_addr = in.getData<float>() + num_head * head_dim * i;
        uint16_t *cache_addr = cache.getData<uint16_t>();
        int row_to_compute = from + i + 1;
        size_t out_start_row =
          calc_attn_index(from + i) - calc_attn_index(from);
        float *output_addr = out.getData<float>() + out_start_row * num_head;

        futures.emplace_back(pool.submit_task([=]() {
          nntrainer::compute_kcaches<uint16_t>(
            input_addr, cache_addr, output_addr, row_to_compute,
            num_head / group_size, head_dim, group_size, tile_size,
            local_window_size);
        }));
      }
      for (auto &fut : futures)
        fut.get();
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    int num_cache_head = num_head / group_size;
    if (sequence_len == 1) {
      tile_size = 300; // the best value on the base test on gauss B1 & B3
      std::vector<std::future<void>> futures;
      int num_rows = from + 1;
      int row_cnt = num_rows < local_window_size ? num_rows : local_window_size;
      const int tile_count = (row_cnt + tile_size - 1) / tile_size;
      for (int n = 0; n < num_cache_head; ++n) {
        const __fp16 *in_ptr = in.getData<_FP16>() + n * group_size * head_dim;
        const __fp16 *cache_ptr = cache.getData<_FP16>() + n * head_dim;
        __fp16 *out_ptr = out.getData<_FP16>() + n * group_size;
        for (int tile_off = 0; tile_off < tile_count; ++tile_off) {
          futures.emplace_back(pool.submit_task([=]() {
            nntrainer::compute_kcaches(in_ptr, cache_ptr, out_ptr, num_rows,
                                       num_cache_head, head_dim, group_size,
                                       tile_off, tile_size, local_window_size);
          }));
        }
      }
      for (auto &fut : futures)
        fut.get();
    } else {
      std::vector<std::future<void>> futures;
      unsigned int seq_start =
        sequence_len < local_window_size ? 0 : sequence_len - local_window_size;
      for (unsigned int i = seq_start; i < sequence_len; ++i) {
        _FP16 *input_addr = in.getData<_FP16>() + num_head * head_dim * i;
        _FP16 *cache_addr = cache.getData<_FP16>();
        int row_to_compute = from + i + 1;
        size_t out_start_row =
          calc_attn_index(from + i) - calc_attn_index(from);

        _FP16 *output_addr = out.getData<_FP16>() + out_start_row * num_head;

        futures.emplace_back(pool.submit_task([=]() {
          int num_rows = row_to_compute;
          int row_cnt =
            num_rows < local_window_size ? num_rows : local_window_size;
          const int tile_count = (row_cnt + tile_size - 1) / tile_size;
          for (int n = 0; n < num_cache_head; ++n) {
            const __fp16 *in_ptr = input_addr + n * group_size * head_dim;
            const __fp16 *cache_ptr = cache_addr + n * head_dim;
            __fp16 *out_ptr = output_addr + n * group_size;
            for (int tile_off = 0; tile_off < tile_count; ++tile_off) {
              nntrainer::compute_kcaches(
                in_ptr, cache_ptr, out_ptr, num_rows, num_cache_head, head_dim,
                group_size, tile_off, tile_size, local_window_size);
            }
          }
        }));
      }
      for (auto &fut : futures)
        fut.get();
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
  auto &pool =
    nntrainer::Engine::Global().getThreadPoolManager()->getThreadPool();

  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(), true);
  nntrainer::Tensor b_cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim,
    batch * cache_value_dim.getFeatureLen() + from * cache_value_dim.width(),
    true);

  apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, _from, false);

  apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim, _from,
                             false);

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
    1, 1, ((to - from) == 1) ? to : calc_attn_index(to) - calc_attn_index(from),
    num_heads_Q, query_step.getTensorType());

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  compute_kcaches(query_step, b_cached_key, out_, _from, to - from, num_heads_Q,
                  gqa_size, head_dim, pool);

  softmax_triangle(out_, to - from, num_heads_Q, from, pool);

  compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                from, num_heads_KV, gqa_size, head_dim, to,
                                pool);
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
  auto &pool =
    nntrainer::Engine::Global().getThreadPoolManager()->getThreadPool();

  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(), true);
  nntrainer::Tensor b_cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim,
    batch * cache_value_dim.getFeatureLen() + from * cache_value_dim.width(),
    true);

  apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, _from, false);

  apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim, _from,
                             false);

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
    1, 1, ((to - from) == 1) ? to : calc_attn_index(to) - calc_attn_index(from),
    num_heads_Q, query_step.getTensorType());

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  compute_kcaches(query_step, b_cached_key, out_, _from, to - from, num_heads_Q,
                  gqa_size, head_dim, pool);

  softmax_triangle(out_, to - from, num_heads_Q, from, pool, sink_step);

  compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                from, num_heads_KV, gqa_size, head_dim, to,
                                pool);
}

/************************************************************** */

/**
 * @brief rotary embedding-related member function
 * @note seq_len -> max_position_embeddings
 */
void MHACoreLayer::precompute_freqs(int head_dim, unsigned int seq_len,
                                    float theta) {
  // compute the freqs only when it is the first time to call this function
  if (freqs_cos != nullptr && freqs_cos->size() == seq_len)
    return;

  if (rope_scaling_type == "default")
    _compute_default_parameters(head_dim, theta);
  else if (rope_scaling_type == "yarn")
    _compute_yarn_parameters(head_dim, theta);
  else
    NNTR_THROW_IF(true, std::invalid_argument) << "Unsupported rope type!";

  // cos / sin
  unsigned int half_ = head_dim / 2;
  auto cos = new std::vector<std::vector<float>>();
  cos->assign(seq_len, std::vector<float>(head_dim, 0));
  auto sin = new std::vector<std::vector<float>>();
  sin->assign(seq_len, std::vector<float>(head_dim, 0));

  // update cos / sin frequency
  for (unsigned int i = 0; i < seq_len; ++i) {

#ifdef USE_NEON
    nntrainer::calc_trigonometric_vals_dup(half_, thetas.data(),
                                           (*cos)[i].data(), (*sin)[i].data(),
                                           i, attention_scaling);
#else
    for (unsigned int j = 0; j < half_; ++j) {
      float angle = i * thetas[j];
      (*cos)[i][j] = std::cos(angle) * attention_scaling;
      (*cos)[i][j + half_] =
        std::cos(angle) * attention_scaling; // repeated 2 times

      (*sin)[i][j] = std::sin(angle) * attention_scaling;
      (*sin)[i][j + half_] =
        std::sin(angle) * attention_scaling; // repeated 2 times
    }
#endif
  }
  freqs_cos = cos;
  freqs_sin = sin;

#ifdef ENABLE_FP16
  // cos / sin for FP16
  auto cos_fp16 = new std::vector<std::vector<_FP16>>();
  cos_fp16->assign(seq_len, std::vector<_FP16>(head_dim, 0));
  auto sin_fp16 = new std::vector<std::vector<_FP16>>();
  sin_fp16->assign(seq_len, std::vector<_FP16>(head_dim, 0));
  for (unsigned int i = 0; i < seq_len; ++i) {
    for (unsigned int j = 0; j < head_dim; ++j) {
      (*cos_fp16)[i][j] = (_FP16)(*cos)[i][j];
      (*sin_fp16)[i][j] = (_FP16)(*sin)[i][j];
    }
  }
  freqs_cos_fp16 = cos_fp16;
  freqs_sin_fp16 = sin_fp16;
#endif
};
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
  unsigned int half_ = dim / 2;
  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    std::vector<float> *cos_ = nullptr;
    std::vector<float> *sin_ = nullptr;

    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &(*freqs_cos)[from + h];
            sin_ = &(*freqs_sin)[from + h];
          }
          float *in_ptr = in.getData<float>() +
                          b * in.channel() * in.height() * in.width() +
                          c * in.height() * in.width() + h * in.width();

          if (out.getDataType() == ml::train::TensorDim::DataType::FP32) {

            nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                                nullptr, cos_->data(),
                                                sin_->data(), convert_only);
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
    std::vector<_FP16> *cos_ = nullptr;
    std::vector<_FP16> *sin_ = nullptr;

    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &(*freqs_cos_fp16)[from + h];
            sin_ = &(*freqs_sin_fp16)[from + h];
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
                                    size_t num_head, unsigned int from,
                                    BS::thread_pool<> &pool) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = from < local_window_size ? from + 1 : local_window_size;
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      std::vector<std::future<void>> futures;
      int seq = row < local_window_size ? row : local_window_size;

      for (int i = 0; i < seq; ++i) {
        size_t start_row = calc_attn_index(from + i) - calc_attn_index(from);
        size_t end_row = calc_attn_index(from + i + 1) - calc_attn_index(from);
        futures.push_back(pool.submit_task([=]() {
          nntrainer::softmax_row(qk_out_, start_row, end_row, num_head);
        }));
      }
      for (auto &fut : futures) {
        fut.get();
      }
    }
  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = from < local_window_size ? from + 1 : local_window_size;
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      std::vector<std::future<void>> futures;
      int seq = row < local_window_size ? row : local_window_size;
      for (int i = 0; i < seq; ++i) {
        size_t start_row = calc_attn_index(from + i) - calc_attn_index(from);
        size_t end_row = calc_attn_index(from + i + 1) - calc_attn_index(from);
        futures.push_back(pool.submit_task([=]() {
          nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
        }));
      }
      for (auto &fut : futures) {
        fut.get();
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
                                    size_t num_head, unsigned int from,
                                    BS::thread_pool<> &pool,
                                    nntrainer::Tensor &sink_step) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = from < local_window_size ? from + 1 : local_window_size;
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head,
                                     sink_step.getData());
    } else {
      std::vector<std::future<void>> futures;
      int min_row = row < local_window_size ? 0 : row - local_window_size;
      // for (int i = row - 1; i >= min_row; --i) {
      for (int i = 0; i < row; ++i) {
        size_t start_row = calc_attn_index(i + from) - calc_attn_index(from);
        size_t end_row = calc_attn_index(from + i + 1) - calc_attn_index(from);
        futures.push_back(pool.submit_task([=]() {
          nntrainer::softmax_row(qk_out_, start_row, end_row, num_head,
                                 sink_step.getData());
        }));
      }
      for (auto &fut : futures) {
        fut.get();
      }
    }
  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();
    _FP16 *sink_step_ = sink_step.getData<_FP16>();

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = from < local_window_size ? from + 1 : local_window_size;
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head,
                                     sink_step_);
    } else {
      std::vector<std::future<void>> futures;
      int min_row = row < local_window_size ? 0 : row - local_window_size;
      // for (int i = row - 1; i >= min_row; --i) {
      for (int i = 0; i < row; ++i) {
        size_t start_row = calc_attn_index(i + from) - calc_attn_index(from);
        size_t end_row = calc_attn_index(from + i + 1) - calc_attn_index(from);
        futures.push_back(pool.submit_task([=]() {
          nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head,
                                         sink_step_);
        }));
      }
      for (auto &fut : futures) {
        fut.get();
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::compute_fp16vcache_transposed(
  nntrainer::Tensor &in, nntrainer::Tensor &vcache, nntrainer::Tensor &output,
  int from, int num_cache_head, int gqa_size, int head_dim, int to,
  BS::thread_pool<> &pool) {

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if ((to - from) != 1) {
      std::vector<std::future<void>> futures;

      int seq = (to - from) < local_window_size ? to - from : local_window_size;
      futures.reserve(seq);

      for (int i = 0; i < seq; ++i) {
        futures.push_back(pool.submit_task([=]() {
          size_t start_idx =
            calc_attn_index(to - seq + i) - calc_attn_index(to - seq);
          const float *input =
            in.getData<float>() + start_idx * num_cache_head * gqa_size;
          float *out = output.getData<float>() +
                       i * (num_cache_head * gqa_size * head_dim);
          nntrainer::compute_fp16vcache_fp32_transposed(
            to - seq + i, input, vcache.getData<uint16_t>(), out,
            num_cache_head, gqa_size, head_dim, local_window_size);
        }));
      }
      for (auto &fut : futures)
        fut.get();
    } else {
      nntrainer::compute_fp16vcache_fp32_transposed(
        to - 1, in.getData<float>(), vcache.getData<uint16_t>(),
        output.getData<float>(), num_cache_head, gqa_size, head_dim,
        local_window_size);
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if ((to - from) != 1) {
      std::vector<std::future<void>> futures;
      int seq = (to - from) < local_window_size ? to - from : local_window_size;
      futures.reserve(seq);

      for (int i = 0; i < seq; ++i) {
        futures.push_back(pool.submit_task([=]() {
          size_t start_idx =
            calc_attn_index(to - seq + i) - calc_attn_index(to - seq);
          const _FP16 *input =
            in.getData<_FP16>() + start_idx * num_cache_head * gqa_size;
          _FP16 *out = output.getData<_FP16>() +
                       i * (num_cache_head * gqa_size * head_dim);
          for (int n = 0; n < num_cache_head; ++n) {
            int chunk_size = head_dim;
            const _FP16 *in_ptr = input + n * gqa_size;
            const _FP16 *vcache_ptr = vcache.getData<_FP16>() + n * head_dim;
            _FP16 *out_ptr = out + n * gqa_size * head_dim;
            nntrainer::compute_fp16vcache_transposed(
              to - seq + i, in_ptr, vcache_ptr, out_ptr, num_cache_head,
              gqa_size, head_dim, chunk_size, local_window_size);
          }
        }));
      }
      for (auto &fut : futures)
        fut.get();
    } else {
      std::vector<std::future<void>> futures;
      for (int n = 0; n < num_cache_head; ++n) {
        const int CHUNK_SIZE = 32;
        for (int chunk_off = 0; chunk_off < head_dim; chunk_off += CHUNK_SIZE) {
          const _FP16 *in_ptr = in.getData<_FP16>() + n * gqa_size;
          const _FP16 *vcache_ptr =
            vcache.getData<_FP16>() + n * head_dim + chunk_off;
          _FP16 *out_ptr =
            output.getData<_FP16>() + n * gqa_size * head_dim + chunk_off;
          futures.emplace_back(pool.submit_task([=]() {
            int chunk_size = std::min(CHUNK_SIZE, head_dim - chunk_off);
            nntrainer::compute_fp16vcache_transposed(
              to - 1, in_ptr, vcache_ptr, out_ptr, num_cache_head, gqa_size,
              head_dim, chunk_size, local_window_size);
          }));
        }
      }
      for (auto &fut : futures)
        fut.get();
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
  unsigned int &max_position_embeddings =
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

  precompute_freqs(head_dim, max_position_embeddings, theta);

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
  auto remain_props = loadProperties(values, mha_core_props);
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

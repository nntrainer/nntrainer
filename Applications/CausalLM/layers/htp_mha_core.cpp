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
#include <cstring>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

static std::mutex rope_init_mtx;

#include <chrono>
#include <cstdlib>
#include <fp16.h>
#include <layer_context.h>

#include <abs_max.h>
#include <mha_core.h>

#include "htp_decode_hook.h"
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <thread_manager.h>
#include <util_func.h>

#include <cstdint>

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
    props::AttnLogitSoftcapping(), props::IsCausal(), props::KvCacheQuant()),
  sm(nntrainer::ActivationType::ACT_SOFTMAX),
  epsilon(1e-3),
  cache_index(0),
  num_heads_Q(0),
  num_heads_KV(0),
  head_dim(0),
  cache_shift(false) {
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

std::vector<uint16_t> MHACoreLayer::q2_q_u16, MHACoreLayer::q2_out_u16;

MHACoreLayer::~MHACoreLayer() { release_quantized_cache(); }

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

  kv_cache_quant_kind = -1;
  if (!std::get<props::KvCacheQuant>(mha_core_props).empty()) {
    const std::string kind =
      std::get<props::KvCacheQuant>(mha_core_props).get();
    if (kind == "q8") {
      kv_cache_quant_kind = 0;
    } else if (kind == "q4") {
      kv_cache_quant_kind = 1;
    } else if (!kind.empty()) {
      throw std::invalid_argument("mha_core: kv_cache_quant must be \"\", "
                                  "\"q8\" or \"q4\", got \"" +
                                  kind + "\"");
    }
  }

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
  // The external-cache path is driven through forwarding() directly as
  // well as via incremental_forwarding(); either way the per-batch helpers
  // ask this for an accelerated attention.
  compute_ops_ = context.getComputeOps();

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

    // The accelerated paths take the f32 query and write f32 output, and
    // the cache write converts the f32 rows itself, so the fp16 staging
    // below (four step-sized tensors and five element-wise conversions a
    // layer) only costs: ~36 ms of a 1023-row step beside a 41 ms call
    // (doc 57 section 9.16). If the call still fails, the CPU computes
    // from the f32 query as the non-Android build always does.
    const bool accelerated = compute_ops_ && is_causal &&
                             (compute_ops_->supports_sdpa_fp16_kvcache() ||
                              (kv_cache_quant_kind >= 0 && !q_cache_failed &&
                               compute_ops_->supports_kv_cache_q()));
    if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32 &&
        !accelerated) {
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
          batch, from, from, to, Q_step, K_step, V_step, O_step, cache_key,
          cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
          cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(batch, from, from, to, Q_step, K_step,
                                         V_step, O_step, cache_key, cache_value,
                                         cache_key_dim, cache_key_step_dim,
                                         cache_value_dim, cache_value_step_dim);
      }
      output_step.copyData(O_step);
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
  // Whatever ComputeOps this context resolves to (CPU, or HTP when the
  // model runs with engine="htp" and the DSP session opened) decides per
  // batch below whether attention leaves the CPU.
  compute_ops_ = context.getComputeOps();

  // [#130] one decode row through the HTP when ATTN_M1 is resident, on
  // either cache path: _from is the absolute position on both.
  if (_to - _from == 1 && htpDecodeAttention(context, _from)) {
    cache_index = _to;
    htp_owns_cache_ = true;
    return;
  }
  // ponytail: once the DSP owns the cache this layer's rows [prompt, ..)
  // are stale, so a later multi-row prefill (a second prompt appended to
  // the same context) would attend to garbage and then re-seed the DSP
  // from it. Refused, not silently wrong; the upgrade is writing the
  // decoded k / v back from the DSP, or running that prefill on the DSP.
  NNTR_THROW_IF(htp_owns_cache_ && _to - _from > 1 && _from > 0,
                std::runtime_error)
    << "mha_core: a multi-row forwarding at " << _from
    << " after the HTP decode hook took over the KV cache "
       "(NNTR_HTP_FORWARD=1 supports one prompt per process)";
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

  if (to > max_timestep) {
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

bool MHACoreLayer::htpDecodeAttention(nntrainer::RunLayerContext &context,
                                      unsigned int pos) {
  nntrainer::Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY);
  nntrainer::Tensor &value = context.getInput(INOUT_INDEX::VALUE);
  nntrainer::Tensor &output = context.getOutput(INOUT_INDEX::OUTPUT);
  if (query.getDataType() != ml::train::TensorDim::DataType::FP32 ||
      query.batch() != 1)
    return false;
  // The row is q | k | v of the step (row 0 of batch 0 on every path).
  const unsigned int wq = query.width(), wk = key.width(), wv = value.width();
  static thread_local std::vector<float> row;
  row.resize(wq + wk + wv);
  std::memcpy(row.data(), query.getData<float>(), wq * sizeof(float));
  std::memcpy(row.data() + wq, key.getData<float>(), wk * sizeof(float));
  std::memcpy(row.data() + wq + wk, value.getData<float>(), wv * sizeof(float));
  // The RoPE table, once per layer instance, in the ROPE op's shape:
  // cos[i][0..hd/2) | sin[i][0..hd/2) per position, the CPU RoPE's own
  // source (plan 82 section 3.4). Head_dim 64 builds it up front (LFM2);
  // [plan 201 S4] any other head_dim when the hook asks for it (3), so a
  // CPU run of such a model keeps no table.
  auto build_rope_table = [&]() {
    const unsigned int rows =
      std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();
    if (freqs_fp32 == nullptr) {
      const std::lock_guard<std::mutex> lock(rope_init_mtx);
      if (freqs_fp32 == nullptr)
        precompute_freqs(head_dim, std::min(max_position_embeddings, rows),
                         theta, false);
    }
    const unsigned int half = head_dim / 2;
    // ponytail: htp_rope_table_ is built once, at the max_timestep of the
    // first decode row, like the DSP graph's KV cache; a later
    // updateTensorsByInputDimensions that grows max_timestep rebuilds the
    // CPU tables but not this one. Upgrade path: clear it there and
    // re-bind the graph at the new length.
    NNTR_THROW_IF(freqs_fp32->cos.size() < rows, std::runtime_error)
      << "mha_core: the RoPE cache holds " << freqs_fp32->cos.size()
      << " positions, max_timestep is " << rows;
    htp_rope_table_.resize(static_cast<size_t>(rows) * head_dim);
    for (unsigned int i = 0; i < rows; ++i) {
      float *t = htp_rope_table_.data() + static_cast<size_t>(i) * head_dim;
      std::memcpy(t, freqs_fp32->cos[i].data(), half * sizeof(float));
      std::memcpy(t + half, freqs_fp32->sin[i].data(), half * sizeof(float));
    }
  };
  if (htp_rope_table_.empty() && use_rope && head_dim == 64)
    build_rope_table();
  int r = htpDecodeAttn(pos, row.data(), wq + wk + wv, output.getData<float>(),
                        wq, htp_rope_table_.data(),
                        static_cast<unsigned>(htp_rope_table_.size()));
  if (r == 3) {
    // [#260] use_rope may be off here when the token runs the row: its
    // q / k are the DSP's own, unrotated (qkv_layer rotates on the CPU
    // path, #4415, with the table precompute_freqs also builds). A row the
    // CPU projected arrives rotated already.
    NNTR_THROW_IF(!htp_rope_table_.empty() ||
                    (!use_rope && !htpDecodeRowResident(pos)),
                  std::runtime_error)
      << "mha_core: the HTP attention hook asked for a RoPE table it has";
    build_rope_table();
    r = htpDecodeAttn(pos, row.data(), wq + wk + wv, output.getData<float>(),
                      wq, htp_rope_table_.data(),
                      static_cast<unsigned>(htp_rope_table_.size()));
  }
  if (r == 2) {
    // Seed the DSP cache from this layer's rows [0, pos): fp16 bits on
    // every platform's cache (FP16 or UINT16 storage), f32 when it is.
    nntrainer::Tensor &ck =
      use_external_cache
        ? context.getInput(3)
        : context.getTensor(tensor_idx[AttentionParams::cache_key]);
    nntrainer::Tensor &cv =
      use_external_cache
        ? context.getInput(4)
        : context.getTensor(tensor_idx[AttentionParams::cache_value]);
    const auto t_seed = std::chrono::steady_clock::now(); // [#282 F]
    const size_t n = static_cast<size_t>(pos) * wk;
    std::vector<float> k_rows(n), v_rows(n);
    auto rows_f32 = [n](const nntrainer::Tensor &c, float *dst) {
      if (c.getDataType() == ml::train::TensorDim::DataType::FP32) {
        std::memcpy(dst, c.getData<float>(), n * sizeof(float));
        return;
      }
      const uint16_t *src = c.getData<uint16_t>();
      for (size_t i = 0; i < n; ++i)
        dst[i] = nntrainer::compute_fp16_to_fp32(src[i]);
    };
    rows_f32(ck, k_rows.data());
    rows_f32(cv, v_rows.data());
    const auto t_conv = std::chrono::steady_clock::now();
    htpDecodeKvSeed(pos, k_rows.data(), v_rows.data());
    {
      static double conv_ms = 0.0, all_ms = 0.0; // [#282 F] summed, logged
      using ms = std::chrono::duration<double, std::milli>;
      conv_ms += ms(t_conv - t_seed).count();
      all_ms += ms(std::chrono::steady_clock::now() - t_seed).count();
      ml_logi("mha_core: t1 seed pos=%u rows_convert_ms_sum=%.1f "
              "seed_ms_sum=%.1f",
              pos, conv_ms, all_ms);
    }
    r = htpDecodeAttn(pos, row.data(), wq + wk + wv, output.getData<float>(),
                      wq, htp_rope_table_.data(),
                      static_cast<unsigned>(htp_rope_table_.size()));
    NNTR_THROW_IF(r != 1, std::runtime_error)
      << "mha_core: the HTP attention hook asked for a seed twice at pos "
      << pos;
  }
  return r == 1;
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
        float *input_addr = in.getData<float>() + num_head * head_dim * i;
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
      // Iterate over ALL query rows so that no row is skipped even when
      // sequence_len > local_window_size.
      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(sequence_len), [=](size_t i) {
        _FP16 *input_addr = in.getData<_FP16>() + num_head * head_dim * i;
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

namespace {

/**
 * @brief f32 views of a query step and an output step for the accelerator
 *        kernels, which take f32 on both sides. On Android with fp16 the
 *        layer hands its per-batch helpers fp16 step tensors, so those are
 *        staged through f32 scratch here (a few hundred KB per step) and
 *        the result is narrowed back on commit.
 */
struct AccelF32Io {
  const float *q = nullptr;
  float *out = nullptr;
  const float *sinks = nullptr; /**< n_head_q f32, or nullptr */

  /**
   * @param sink  the per-head sink weight, or nullptr when the layer has
   *              none; fp16 sinks are widened here (the weight is fp16 on
   *              Android with fp16 enabled, whatever the activation type)
   */
  bool prepare(nntrainer::Tensor &query_step,
               nntrainer::Tensor &attention_output_step,
               const nntrainer::Tensor *sink) {
    const auto qt = query_step.getDataType();
    const auto ot = attention_output_step.getDataType();
    if (sink && !sink->empty()) {
      const auto st = sink->getDataType();
      if (st == ml::train::TensorDim::DataType::FP32) {
        sinks = sink->getData<float>();
      }
#ifdef ENABLE_FP16
      else if (st == ml::train::TensorDim::DataType::FP16) {
        const _FP16 *src = sink->getData<_FP16>();
        sink_scratch.resize(sink->size());
        for (size_t i = 0; i < sink_scratch.size(); ++i) {
          sink_scratch[i] = static_cast<float>(src[i]);
        }
        sinks = sink_scratch.data();
      }
#endif
      else {
        return false;
      }
    }
    if (qt == ml::train::TensorDim::DataType::FP32) {
      q = query_step.getData<float>();
    }
#ifdef ENABLE_FP16
    else if (qt == ml::train::TensorDim::DataType::FP16) {
      const _FP16 *src = query_step.getData<_FP16>();
      q_scratch.resize(query_step.size());
      for (size_t i = 0; i < q_scratch.size(); ++i) {
        q_scratch[i] = static_cast<float>(src[i]);
      }
      q = q_scratch.data();
    }
#endif
    else {
      return false;
    }
    if (ot == ml::train::TensorDim::DataType::FP32) {
      out = attention_output_step.getData<float>();
    }
#ifdef ENABLE_FP16
    else if (ot == ml::train::TensorDim::DataType::FP16) {
      out_scratch.resize(attention_output_step.size());
      out = out_scratch.data();
    }
#endif
    else {
      return false;
    }
    return true;
  }

  void commit(nntrainer::Tensor &attention_output_step) {
#ifdef ENABLE_FP16
    if (!out_scratch.empty()) {
      _FP16 *dst = attention_output_step.getData<_FP16>();
      for (size_t i = 0; i < out_scratch.size(); ++i) {
        dst[i] = static_cast<_FP16>(out_scratch[i]);
      }
    }
#else
    (void)attention_output_step;
#endif
  }

private:
  std::vector<float> q_scratch, out_scratch, sink_scratch;
};

} // namespace

bool MHACoreLayer::try_accelerated_attention(
  nntrainer::Tensor &query_step, nntrainer::Tensor &cached_key,
  nntrainer::Tensor &cached_value, nntrainer::Tensor &attention_output_step,
  unsigned int cache_from, unsigned int cache_to,
  const nntrainer::Tensor *sink) {
  if (!compute_ops_ || !compute_ops_->supports_sdpa_fp16_kvcache()) {
    return false;
  }
  // The accelerated kernel implements the causal, windowed attention this
  // layer computes over the fp16 cache -- the shape every CausalLM model
  // here runs -- with f32 query and output; fp16 steps (Android) are
  // staged through f32. Anything else stays on the CPU.
  if (!is_causal) {
    return false;
  }
  AccelF32Io io;
  if (!io.prepare(query_step, attention_output_step, sink)) {
    return false;
  }
  const uint16_t *k_bits = nullptr;
  const uint16_t *v_bits = nullptr;
  switch (cached_key.getDataType()) {
  case ml::train::TensorDim::DataType::UINT16:
    k_bits = cached_key.getData<uint16_t>();
    v_bits = cached_value.getData<uint16_t>();
    break;
#ifdef ENABLE_FP16
  case ml::train::TensorDim::DataType::FP16:
    k_bits = reinterpret_cast<const uint16_t *>(cached_key.getData<_FP16>());
    v_bits = reinterpret_cast<const uint16_t *>(cached_value.getData<_FP16>());
    break;
#endif
  default:
    return false;
  }
  const unsigned int n_q = cache_to - cache_from;
  const unsigned int q_stride = num_heads_Q * head_dim;
  const unsigned int kv_stride = num_heads_KV * head_dim;
  // local_window_size defaults to UINT_MAX meaning "no window"; the kernel
  // spells that 0.
  const unsigned int window =
    (local_window_size == 0 || local_window_size >= cache_to)
      ? 0u
      : static_cast<unsigned int>(local_window_size);
  if (!compute_ops_->sdpa_fp16_kvcache(
        io.q, q_stride, k_bits, v_bits, kv_stride, n_q, cache_from, cache_to,
        num_heads_Q, num_heads_KV, head_dim, window, attn_logit_softcapping,
        io.sinks, io.out, q_stride)) {
    return false;
  }
  io.commit(attention_output_step);
  if (!accel_logged_) {
    accel_logged_ = true;
    ml_logi("mha_core: attention over the fp16 KV cache on the accelerator");
  }
  return true;
}

void MHACoreLayer::registerQuantizedCache(nntrainer::RunLayerContext &context) {
  compute_ops_ = context.getComputeOps();
  // [#282 D] with NNTR_HTP_E2E_FREE_KVQ the caches go at the first decode
  // token anyway; registering them at the prefill's first call (the lazy
  // path, try_quantized_attention) keeps them out of the address space the
  // load's FC set and expert pool map into (p4096: 915 MiB at load)
  static const bool lazy = std::getenv("NNTR_HTP_E2E_FREE_KVQ") != nullptr;
  if (lazy)
    return;
  if (kv_cache_quant_kind < 0 || q_cache_failed || !compute_ops_ ||
      !compute_ops_->supports_kv_cache_q() || !is_causal ||
      !q_cache_handles.empty())
    return;
  // try_quantized_attention's own gate for a layer the int8 kernels take
  const bool q2 = kv_cache_quant_kind == 0 &&
                  compute_ops_->supports_kv_cache_q2() &&
                  attn_logit_softcapping <= 0.0f;
  if (!q2 && head_dim > 256)
    return;
  const nntrainer::Tensor &ck =
    use_external_cache
      ? context.getInput(3)
      : context.getTensor(tensor_idx[AttentionParams::cache_key]);
  const int h = compute_ops_->kv_cache_q_register(
    static_cast<unsigned int>(kv_cache_quant_kind), ck.getDim().height(),
    num_heads_KV, head_dim);
  if (h < 0)
    return;
  q_cache_handles.assign(1, h);
  q_cache_synced.assign(1, 0);
  q2_scales_set.assign(1, 0);
}

void MHACoreLayer::release_quantized_cache() {
  if (compute_ops_) {
    for (int h : q_cache_handles) {
      if (h >= 0) {
        compute_ops_->kv_cache_q_release(h);
      }
    }
  }
  q_cache_handles.clear();
  q_cache_synced.clear();
  q2_scales_set.clear();
}

/**
 * @brief Scales for the fixed-scale int8 path from the rows at hand: K per
 *        KV head and V per (KV head, dim) over fp16 cache rows [0, n_rows),
 *        Q per query head over this step's f32 rows. max / 127, 1 for an
 *        all-zero tensor.
 */
void MHACoreLayer::calibrate_q2_scales(const uint16_t *k_rows,
                                       const uint16_t *v_rows,
                                       unsigned int n_rows,
                                       unsigned int kv_stride, const float *q,
                                       unsigned int q_stride,
                                       unsigned int n_q) {
  q2_scale_k.assign(num_heads_KV, 0.0f);
  q2_scale_v.assign(static_cast<size_t>(num_heads_KV) * head_dim, 0.0f);
  q2_q_enc.assign(2 * static_cast<size_t>(num_heads_Q), 0.0f);
  q2_out_enc.assign(2 * static_cast<size_t>(num_heads_Q), 0.0f);
  for (unsigned int r = 0; r < n_rows; ++r) {
    const uint16_t *kr = k_rows + static_cast<size_t>(r) * kv_stride;
    const uint16_t *vr = v_rows + static_cast<size_t>(r) * kv_stride;
    for (unsigned int n = 0; n < num_heads_KV; ++n) {
      q2_scale_k[n] =
        causallm::abs_max_f16(kr + n * head_dim, head_dim, q2_scale_k[n]);
    }
    causallm::abs_max_f16_lanes(vr, num_heads_KV * head_dim, q2_scale_v.data());
  }
  // Q's per-head range, NEON (min / max are order-free): scalar it was
  // 11-22 ms a 1024-row layer (doc 59 section 2.2).
  std::vector<float> q_lo(num_heads_Q, 1e30f), q_hi(num_heads_Q, -1e30f);
  for (unsigned int r = 0; r < n_q; ++r) {
    const float *qr = q + static_cast<size_t>(r) * q_stride;
    for (unsigned int h = 0; h < num_heads_Q; ++h)
      causallm::min_max_f32(qr + h * head_dim, head_dim, q_lo[h], q_hi[h]);
  }
  const unsigned int G = num_heads_Q / num_heads_KV;
  for (unsigned int h = 0; h < num_heads_Q; ++h) {
    const float range = std::max(q_hi[h] - q_lo[h], 1e-6f);
    q2_q_enc[2 * h] = range / 65535.0f;
    q2_q_enc[2 * h + 1] = std::round(-q_lo[h] / q2_q_enc[2 * h]);
    float vmax = 0.0f;
    for (unsigned int d = 0; d < head_dim; ++d) {
      vmax = std::max(vmax, q2_scale_v[(h / G) * head_dim + d]);
    }
    q2_out_enc[2 * h] = std::max(2.0f * vmax, 1e-6f) / 65535.0f;
    q2_out_enc[2 * h + 1] = 32768.0f;
  }
  auto finish = [](float &x) { x = x > 0.0f ? x / 127.0f : 1.0f; };
  for (auto &x : q2_scale_k) {
    finish(x);
  }
  for (auto &x : q2_scale_v) {
    finish(x);
  }
  q2_calibrated = true;
}

bool MHACoreLayer::try_quantized_attention(
  unsigned int batch, nntrainer::Tensor &query_step,
  nntrainer::Tensor &cache_key, nntrainer::Tensor &cache_value,
  const ml::train::TensorDim &cache_key_dim,
  nntrainer::Tensor &attention_output_step, unsigned int cache_from,
  unsigned int cache_to, const nntrainer::Tensor *sink) {
  if (kv_cache_quant_kind < 0 || q_cache_failed || !compute_ops_ ||
      !compute_ops_->supports_kv_cache_q() || !is_causal) {
    return false;
  }
  using clk = std::chrono::steady_clock;
  const auto ts0 = clk::now();
  AccelF32Io io;
  if (!io.prepare(query_step, attention_output_step, sink)) {
    return false;
  }
  const auto ts1 = clk::now();
  clk::time_point ts2 = ts1, ts3 = ts1, ts4 = ts1, ts5 = ts1;
  const uint16_t *k_base = nullptr;
  const uint16_t *v_base = nullptr;
  switch (cache_key.getDataType()) {
  case ml::train::TensorDim::DataType::UINT16:
    k_base = cache_key.getData<uint16_t>();
    v_base = cache_value.getData<uint16_t>();
    break;
#ifdef ENABLE_FP16
  case ml::train::TensorDim::DataType::FP16:
    k_base = reinterpret_cast<const uint16_t *>(cache_key.getData<_FP16>());
    v_base = reinterpret_cast<const uint16_t *>(cache_value.getData<_FP16>());
    break;
#endif
  default:
    return false;
  }
  const unsigned int max_rows = cache_key_dim.height();
  const unsigned int width = cache_key_dim.width();
  if (width != num_heads_KV * head_dim || cache_to > max_rows) {
    return false;
  }

  auto fail = [this](const char *what) {
    ml_logw("mha_core: quantized KV cache %s failed; this layer stays on the "
            "fp16 path",
            what);
    release_quantized_cache();
    q_cache_failed = true;
    return false;
  };

  if (q_cache_handles.size() <= batch) {
    q_cache_handles.resize(batch + 1, -1);
    q_cache_synced.resize(batch + 1, 0);
    q2_scales_set.resize(batch + 1, 0);
  }
  int &handle = q_cache_handles[batch];
  unsigned int &synced = q_cache_synced[batch];
  // int8 with the row-blocked kernel when the backend has it: fixed scales,
  // no softcap, no sinks, head_dim up to 512.
  const bool use_q2 = kv_cache_quant_kind == 0 &&
                      compute_ops_->supports_kv_cache_q2() &&
                      attn_logit_softcapping <= 0.0f && io.sinks == nullptr;
  if (!use_q2 && head_dim > 256) {
    return false;
  }
  if (handle < 0) {
    ts2 = clk::now();
    handle = compute_ops_->kv_cache_q_register(
      static_cast<unsigned int>(kv_cache_quant_kind), max_rows, num_heads_KV,
      head_dim);
    if (handle < 0) {
      return fail("register");
    }
    synced = 0;
    q2_scales_set[batch] = 0;
    ts3 = clk::now();
  }
  // Rows past cache_from may have been rewritten (a rewound session, a
  // loaded cache); re-append from the first row that could differ.
  if (synced > cache_from) {
    synced = cache_from;
  }
  // The missing rows -- normally just this step's -- ride along with the
  // attention call: one round trip per layer per step.
  const unsigned int append_row0 = synced;
  const unsigned int append_rows = cache_to - synced;
  const size_t off = batch * cache_key_dim.getFeatureLen() +
                     static_cast<size_t>(append_row0) * width;

  const unsigned int n_q = cache_to - cache_from;
  const unsigned int q_stride = num_heads_Q * head_dim;
  const unsigned int window =
    (local_window_size == 0 || local_window_size >= cache_to)
      ? 0u
      : static_cast<unsigned int>(local_window_size);
  if (use_q2) {
    if (!q2_calibrated) {
      // The rows about to be appended are the first this layer sees.
      const auto t0 = std::chrono::steady_clock::now();
      calibrate_q2_scales(k_base + off, v_base + off, append_rows, width, io.q,
                          q_stride, n_q);
      if (std::getenv("NNTR_HTP_ATTN_TRACE")) {
        ml_logi("mha_core trace: q2 calibration rows=%u us=%lld", append_rows,
                static_cast<long long>(
                  std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0)
                    .count()));
      }
    }
    const auto ts_sc = clk::now();
    if (!q2_scales_set[batch]) {
      if (!compute_ops_->kv_cache_q_set_fixed_scales(
            handle, num_heads_KV, head_dim, q2_scale_k.data(),
            q2_scale_v.data())) {
        return fail("fixed scales");
      }
      q2_scales_set[batch] = 1;
    }
    // Q as the model's u16 tensor and the output back from u16: in a
    // quantized graph both are the neighbouring layers' formats already;
    // here the f32 layer converts at its edges.
    const size_t q_elems = static_cast<size_t>(n_q) * q_stride;
    ts4 = clk::now();
    if (q2_q_u16.size() < q_elems) {
      q2_q_u16.resize(q_elems);
      q2_out_u16.resize(q_elems);
    }
    ts5 = clk::now();
    // Both conversions NEON on this thread: scalar they were 8-13 ms a
    // 1024-row layer (doc 59 section 2.2); NEON 1.0 / 1.1 ms, and split over
    // 4 or 8 compute threads slower (doc 57 section 9.23).
    static const bool conv_trace = std::getenv("NNTR_HTP_ATTN_TRACE");
    const auto tq0 = std::chrono::steady_clock::now();
    for (unsigned int r = 0; r < n_q; ++r) {
      const float *qr = io.q + static_cast<size_t>(r) * q_stride;
      uint16_t *dst = q2_q_u16.data() + static_cast<size_t>(r) * q_stride;
      for (unsigned int h = 0; h < num_heads_Q; ++h)
        causallm::quant_u16_f32(qr + h * head_dim, head_dim,
                                1.0f / q2_q_enc[2 * h], q2_q_enc[2 * h + 1],
                                dst + h * head_dim);
    }
    const auto tq1 = std::chrono::steady_clock::now();
    if (!compute_ops_->sdpa_q2_kvcache(
          handle, append_row0, append_rows, width, k_base + off, v_base + off,
          q2_q_u16.data(), q2_q_enc.data(), q_stride, n_q, cache_from, cache_to,
          num_heads_Q, num_heads_KV, head_dim, window, q2_out_u16.data(),
          q2_out_enc.data(), q_stride)) {
      return fail("attention");
    }
    const auto tq2 = std::chrono::steady_clock::now();
    for (unsigned int r = 0; r < n_q; ++r) {
      const uint16_t *src =
        q2_out_u16.data() + static_cast<size_t>(r) * q_stride;
      float *dst = io.out + static_cast<size_t>(r) * q_stride;
      for (unsigned int h = 0; h < num_heads_Q; ++h)
        causallm::dequant_u16_f32(src + h * head_dim, head_dim,
                                  q2_out_enc[2 * h], q2_out_enc[2 * h + 1],
                                  dst + h * head_dim);
    }
    if (conv_trace && n_q > 1) {
      const auto tq3 = std::chrono::steady_clock::now();
      auto us = [](auto d) {
        return static_cast<long long>(
          std::chrono::duration_cast<std::chrono::microseconds>(d).count());
      };
      ml_logi("mha_core trace: q2 setup rows=%u prepare_us=%lld "
              "register_us=%lld fixed_scales_us=%lld resize_us=%lld",
              n_q, us(ts1 - ts0), us(ts3 - ts2), us(ts4 - ts_sc),
              us(ts5 - ts4));
      ml_logi("mha_core trace: q2 convert rows=%u q_us=%lld call_us=%lld "
              "out_us=%lld",
              n_q, us(tq1 - tq0), us(tq2 - tq1), us(tq3 - tq2));
    }
  } else if (!compute_ops_->sdpa_q_kvcache(
               handle, append_row0, append_rows, width, k_base + off,
               v_base + off, io.q, q_stride, n_q, cache_from, cache_to,
               num_heads_Q, num_heads_KV, head_dim, window,
               attn_logit_softcapping, io.sinks, io.out, q_stride)) {
    return fail("attention");
  }
  synced = cache_to;
  io.commit(attention_output_step);
  if (!accel_logged_) {
    accel_logged_ = true;
    ml_logi("mha_core: attention over the %s quantized KV cache on the "
            "accelerator%s",
            kv_cache_quant_kind == 0 ? "int8" : "int4",
            use_q2 ? " (row-blocked, fixed scales)" : "");
  }
  return true;
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

  // NNTR_HTP_ATTN_TRACE: the host's own share of this node beside the
  // accelerator call's (htp_compute_ops' attn trace), per step, to logcat.
  // The profile build counted ~40 ms a sliding layer outside a 40 ms call
  // at 1023 rows (doc 57 section 9.13).
  static const bool host_trace = [] {
    const char *e = std::getenv("NNTR_HTP_ATTN_TRACE");
    return e && *e && *e != '0';
  }();
  auto now_us = [] {
    return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
  };
  const long long t_kv0 = host_trace ? now_us() : 0;

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
  const long long t_kv1 = host_trace ? now_us() : 0;

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

  if (try_quantized_attention(batch, query_step, cache_key, cache_value,
                              cache_key_dim, attention_output_step, cache_from,
                              cache_to, nullptr) ||
      try_accelerated_attention(query_step, b_cached_key, b_cached_value,
                                attention_output_step, cache_from, cache_to,
                                nullptr)) {
    if (host_trace && step_size > 1) {
      ml_logi("mha_core trace: rows=%u kv_write_us=%lld accel_call_us=%lld",
              step_size, t_kv1 - t_kv0, now_us() - t_kv1);
    }
    return;
  }

  // out_ stores the output of Q * K
  nntrainer::Tensor out_(1, 1,
                         is_causal ? (calc_windowed_attn_index(cache_to) -
                                      calc_windowed_attn_index(cache_from))
                                   : (step_size * cache_to),
                         num_heads_Q, query_step.getTensorType());

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  compute_kcaches(query_step, b_cached_key, out_, cache_from,
                  cache_to - cache_from, num_heads_Q, gqa_size, head_dim);

  softmax_triangle(out_, step_size, num_heads_Q, cache_from);

  compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                cache_from, num_heads_KV, gqa_size, head_dim,
                                cache_to);
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

  // The sink weight goes as a tensor: it is fp16 on Android with fp16
  // enabled, and the accelerated paths take f32 (staged inside).
  if (try_quantized_attention(batch, query_step, cache_key, cache_value,
                              cache_key_dim, attention_output_step, from, to,
                              &sink_step) ||
      try_accelerated_attention(query_step, b_cached_key, b_cached_value,
                                attention_output_step, from, to, &sink_step)) {
    return;
  }

  nntrainer::Tensor out_(1, 1,
                         is_causal ? (((to - from) == 1)
                                        ? to
                                        : calc_windowed_attn_index(to) -
                                            calc_windowed_attn_index(from))
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
      nntrainer::calc_trigonometric_vals_dup(
        half_, thetas.data(), cached->cos[i].data(), cached->sin[i].data(), i,
        attention_scaling);
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
      nntrainer::calc_trigonometric_vals_dup(half_, thetas.data(),
                                             cos_tmp.data(), sin_tmp.data(), i,
                                             attention_scaling);
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
      out.copyData(in);
    }
    return;
  }
  unsigned int half_ = dim / 2;
  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();
  // The table only needs the positions the cache can hold: a model's
  // max_position_embeddings (262144 on one here) is far past max_timestep,
  // and cos/sin rows past it are never indexed. Measured on device: the
  // first layer of each RoPE shape spent about 0.5 s building the full
  // table (doc 57 section 3).
  const unsigned int rope_rows =
    std::min(max_position_embeddings, max_timestep);

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (freqs_fp32 == nullptr) {
      const std::lock_guard<std::mutex> lock(rope_init_mtx);
      if (freqs_fp32 == nullptr) {
        precompute_freqs(head_dim, rope_rows, theta, false);
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
        precompute_freqs(head_dim, rope_rows, theta, true);
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

void MHACoreLayer::compute_fp16vcache_transposed(
  nntrainer::Tensor &in, nntrainer::Tensor &vcache, nntrainer::Tensor &output,
  int from, int num_cache_head, int gqa_size, int head_dim, int to) {

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
      tm.parallel_for(0, static_cast<size_t>(total), [=](size_t i) {
        size_t start_idx;
        if (is_causal) {
          start_idx =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
        } else {
          start_idx = i * to;
        }
        const _FP16 *input =
          in.getData<_FP16>() + start_idx * num_cache_head * gqa_size;
        _FP16 *out =
          output.getData<_FP16>() + i * (num_cache_head * gqa_size * head_dim);
        int row_num = is_causal ? (from + (int)i) : to - 1;
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

size_t MHACoreLayer::calc_windowed_attn_index(size_t i) {
  // S(i) = sum_{k=0}^{i-1} min(k+1, W)
  // For i <= W:  S(i) = i*(i+1)/2   (same as full-attention triangular index)
  // For i >  W:  S(i) = W*(W+1)/2 + (i - W)*W
  // When W == UINT_MAX, i <= W is always true, so we never evaluate
  // W*(W+1)/2 and there is no overflow.
  if (i <= local_window_size) {
    return (i * (i + 1)) / 2;
  } else {
    return (local_window_size * (local_window_size + 1)) / 2 +
           (i - local_window_size) * local_window_size;
  }
};

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

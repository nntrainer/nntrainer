// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   fp16_attention.cpp
 * @date   11 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Attention arithmetic with an FP16 KV cache.
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

// FP16 CPU attention kernels are implemented by the ARM backend only.
#if defined(ENABLE_FP16) &&                                                    \
  (defined(__aarch64__) || defined(__ARM_ARCH_7A__) || defined(__ANDROID__) || \
   defined(__arm__) || defined(_M_ARM) || defined(_M_ARM64))
#define NNTR_ARM_FP16_ATTENTION
#endif

static std::mutex rope_init_mtx;

#include <chrono>
#include <cstdlib>
#include <fp16.h>
#include <layer_context.h>

#include <fp16_attention.h>
#include <htp_backend/htp_attention_state.h>
#include <htp_backend/htp_runtime_config.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <thread_manager.h>
#include <util_func.h>

#include <cstdint>

inline float convert_scalar(uint16_t h) {
  return nntrainer::compute_fp16_to_fp32(h);
}

namespace nntrainer {

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
 * @brief constructor of Fp16Attention
 */
Fp16Attention::Fp16Attention() :
  mha_core_props(
    nntrainer::props::NumHeads(), attention_props::NumHeads_KV(),
    nntrainer::props::ProjectedKeyDim(), nntrainer::props::ProjectedValueDim(),
    nntrainer::props::OutputShape(), nntrainer::props::DropOutRate(),
    nntrainer::props::ReturnAttentionWeight(),
    nntrainer::props::AverageAttentionWeight(), nntrainer::props::MaxTimestep(),
    attention_props::SlidingWindow(), attention_props::MaxNewTokens(),
    attention_props::RopeTheta(), attention_props::UseRope(),
    attention_props::MaxPositionEmbeddings(), attention_props::UseSink(),
    attention_props::RopeScalingType(), attention_props::RopeScalingFactor(),
    attention_props::RopePartialRotaryFactor(),
    attention_props::RopeScalingMaxPositionEmbeddings(),
    attention_props::AttnLogitSoftcapping(), attention_props::IsCausal(),
    attention_props::KvCacheQuant(), attention_props::AttnOutU16()),
  sm(nntrainer::ActivationType::ACT_SOFTMAX),
  epsilon(1e-3),
  cache_shift(false) {
  state_ =
    std::make_unique<HtpAttentionState>(static_cast<AttentionConfig &>(*this));
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

Fp16Attention::~Fp16Attention() = default;

/************************************************************** */

void Fp16Attention::finalize(nntrainer::InitLayerContext &context, bool skip) {

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
    std::get<attention_props::MaxPositionEmbeddings>(mha_core_props).get();

  /** local window size */
  local_window_size =
    std::get<attention_props::SlidingWindow>(mha_core_props).get();

  /** use rope */
  use_rope = std::get<attention_props::UseRope>(mha_core_props).get();

  /** attention scaling computation */
  rope_scaling_type =
    std::get<attention_props::RopeScalingType>(mha_core_props).get();
  scale = std::get<attention_props::RopeScalingFactor>(mha_core_props).get();
  rope_partial_rotary_factor =
    std::get<attention_props::RopePartialRotaryFactor>(mha_core_props).get();
  if (rope_scaling_type == "yarn")
    original_max_position_embeddings =
      std::get<attention_props::RopeScalingMaxPositionEmbeddings>(
        mha_core_props)
        .get();

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
    std::get<attention_props::NumHeads_KV>(mha_core_props).empty()
      ? num_heads_Q
      : static_cast<size_t>(
          std::get<attention_props::NumHeads_KV>(mha_core_props).get());

  // head_dim
  head_dim = static_cast<size_t>(query_width) / num_heads_Q;
  NNTR_THROW_IF(head_dim != key_width / num_heads_KV, std::invalid_argument)
    << "num_heads_Q and num_heads_KV are not properly given. Please check the "
       "num_heads_* are set correctly so that the `head_dim`s are all same for "
       "query / key / value";

  /** Weight for Sink */
  use_sink = std::get<attention_props::UseSink>(mha_core_props).get();
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
    std::get<attention_props::AttnLogitSoftcapping>(mha_core_props).get();

  /** Is Causal */
  is_causal = std::get<attention_props::IsCausal>(mha_core_props).get();

  attn_out_u16 = std::get<attention_props::AttnOutU16>(mha_core_props).get();
  kv_cache_quant_kind = -1;
  kv_cache_fixed_scales = true;
  if (!std::get<attention_props::KvCacheQuant>(mha_core_props).empty()) {
    const std::string kind =
      std::get<attention_props::KvCacheQuant>(mha_core_props).get();
    if (kind == "q8" || kind == "q8_row") {
      kv_cache_quant_kind = 0;
      kv_cache_fixed_scales = kind != "q8_row";
    } else if (kind == "q4") {
      kv_cache_quant_kind = 1;
    } else if (!kind.empty()) {
      throw std::invalid_argument("mha_core: kv_cache_quant must be \"\", "
                                  "\"q8\", \"q8_row\" or \"q4\", got \"" +
                                  kind + "\"");
    }
  }

  skip_prefill = skip;

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

  theta = (float)std::get<attention_props::RopeTheta>(mha_core_props).get();

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
void Fp16Attention::forwarding(nntrainer::RunLayerContext &context,
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

    // Accelerated attention accepts f32 query/output and converts cache
    // rows itself, so skip fp16 staging. On failure the CPU uses f32 too.
    const bool accelerated = compute_ops_ && is_causal &&
                             (compute_ops_->supports_sdpa_fp16_kvcache() ||
                              (kv_cache_quant_kind >= 0 && !state_->failed() &&
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
void Fp16Attention::incremental_forwarding(nntrainer::RunLayerContext &context,
                                           unsigned int _from, unsigned int _to,
                                           bool training) {
  // Whatever ComputeOps this context resolves to (CPU, or HTP when the
  // model runs with engine="htp" and the DSP session opened) decides per
  // batch below whether attention leaves the CPU.
  compute_ops_ = context.getComputeOps();
  state_->name(context.getName());

  // One decode row through the HTP when ATTN_M1 is resident, on
  // either cache path: _from is the absolute position on both.
  if (_to - _from == 1 && htpDecodeAttention(context, _from)) {
    cache_index = _to;
    state_->markResident();
    return;
  }
  // Direct layer calls cannot consume stale host KV. Supported appended
  // chunks enter through HtpContext, which preserves the resident DSP state.
  NNTR_THROW_IF(state_->ownsCache() && _to - _from > 1 && _from > 0,
                std::runtime_error)
    << "mha_core: a multi-row forwarding at " << _from
    << " after the HTP decode hook took over the KV cache "
       "(use the HtpContext continuation entry for appended tokens)";
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
void Fp16Attention::compute_kcaches(nntrainer::Tensor &in,
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
#ifdef NNTR_ARM_FP16_ATTENTION
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
#elif defined(ENABLE_FP16)
    NNTR_THROW_IF(true, std::invalid_argument)
      << "FP16 input attention is unsupported by this CPU backend";
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

/**
 * @brief Scales for the fixed-scale int8 path from the rows at hand: K per
 *        KV head and V per (KV head, dim) over fp16 cache rows [0, n_rows),
 *        Q per query head over this step's f32 rows. max / 127, 1 for an
 *        all-zero tensor.
 */

void Fp16Attention::one_batch_incremental_forwarding(
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

  // The step's first row in the cache (setCacheBase): cache_index itself
  // unless the cache holds a window of the context.
  NNTR_THROW_IF(cache_index < cache_base ||
                  cache_index - cache_base + (to - from) >
                    cache_key_dim.height(),
                std::runtime_error)
    << "mha_core: rows [" << cache_index << ", " << cache_index + (to - from)
    << ") do not fit a KV cache of " << cache_key_dim.height()
    << " rows based at " << cache_base;
  const unsigned int cache_row = cache_index - cache_base;

  // Load Input Tensors of this batch : b_ denotes a Tensor for this batch
  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + cache_row * cache_key_dim.width(),
    true);
  nntrainer::Tensor b_cache_value_step =
    cache_value.getSharedDataTensor(cache_value_step_dim,
                                    batch * cache_value_dim.getFeatureLen() +
                                      cache_row * cache_value_dim.width(),
                                    true);

  const bool host_trace = htpRuntimeConfig().htp.attn_trace;
  auto now_us = [] {
    return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
  };
  const long long t_kv0 = host_trace ? now_us() : 0;

  // The projection call may have kept its K / V rows (qkv_layer's
  // out_to_host false): then they are read where the engine has them.
  auto viewed = [this](nntrainer::Tensor &t) {
    if (!compute_ops_ ||
        t.getDataType() != ml::train::TensorDim::DataType::FP32)
      return t;
    const size_t bytes = t.size() * sizeof(float);
    const void *v = compute_ops_->host_view(t.getData<float>(), bytes);
    return v == t.getData<float>()
             ? t
             : nntrainer::Tensor::Map(
                 static_cast<float *>(const_cast<void *>(v)),
                 static_cast<unsigned int>(bytes), t.getDim());
  };
  nntrainer::Tensor key_src = viewed(key_step);
  nntrainer::Tensor value_src = viewed(value_step);

  // append kcache with or without rotary embedding
  apply_rotary_emb_tensor_v2(key_src, b_cache_key_step, head_dim, cache_index,
                             !use_rope);

  // append vcache without rotary embedding
  if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    apply_rotary_emb_tensor_v2(value_src, b_cache_value_step, head_dim,
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
  unsigned int cache_from = cache_row;
  unsigned int cache_to = cache_from + step_size;

  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  cached_key_dim.height(cache_to);
  cached_value_dim.height(cache_to);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  bool done = state_->try_quantized_attention(
    batch, query_step, cache_key, cache_value, cache_key_dim,
    attention_output_step, cache_from, cache_to, nullptr);
  if (!done) {
    // every path below reads Q in the host tensor
    if (compute_ops_ &&
        query_step.getDataType() == ml::train::TensorDim::DataType::FP32)
      compute_ops_->sync_to_host(query_step.getData<float>(),
                                 query_step.size() * sizeof(float));
    done = state_->try_accelerated_attention(
      query_step, b_cached_key, b_cached_value, attention_output_step,
      cache_from, cache_to, nullptr);
  }
  // Compare the accelerator and CPU from the same Q/K/V rows.
  const bool f16_check = htpRuntimeConfig().htp.attn_f16_check;
  std::vector<float> accel_out;
  if (done && f16_check && step_size > 1 &&
      attention_output_step.getDataType() ==
        ml::train::TensorDim::DataType::FP32) {
    accel_out.assign(attention_output_step.getData<float>(),
                     attention_output_step.getData<float>() +
                       attention_output_step.size());
    // the CPU reads Q in the host tensor (the quantized path left it on the
    // accelerator when it succeeded)
    if (compute_ops_ &&
        query_step.getDataType() == ml::train::TensorDim::DataType::FP32)
      compute_ops_->sync_to_host(query_step.getData<float>(),
                                 query_step.size() * sizeof(float));
    done = false; // the CPU path below recomputes into the same tensor
  }
  if (done) {
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
  if (!accel_out.empty()) {
    const float *cpu = attention_output_step.getData<float>();
    const unsigned int w = num_heads_Q * head_dim;
    double sd = 0.0, ss = 0.0;
    unsigned int worst_row = 0;
    double worst = 0.0;
    for (unsigned int r = 0; r < step_size; ++r) {
      double rd = 0.0, rs = 0.0;
      for (unsigned int i = 0; i < w; ++i) {
        const double d = accel_out[static_cast<size_t>(r) * w + i] -
                         cpu[static_cast<size_t>(r) * w + i];
        rd += d * d;
        rs += static_cast<double>(cpu[static_cast<size_t>(r) * w + i]) *
              cpu[static_cast<size_t>(r) * w + i];
      }
      sd += rd;
      ss += rs;
      if (rs > 0.0 && std::sqrt(rd / rs) > worst) {
        worst = std::sqrt(rd / rs);
        worst_row = r;
      }
    }
    ml_logi("mha_core f16 check: %s rows=%u from=%u rel_rms=%.5f worst_row=%u "
            "(%.4f)",
            state_->name().c_str(), step_size, cache_from,
            ss > 0.0 ? std::sqrt(sd / ss) : 0.0, worst_row, worst);
  }
}

void Fp16Attention::one_batch_incremental_forwarding(
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
  if (state_->try_quantized_attention(batch, query_step, cache_key, cache_value,
                                      cache_key_dim, attention_output_step,
                                      from, to, &sink_step) ||
      state_->try_accelerated_attention(query_step, b_cached_key,
                                        b_cached_value, attention_output_step,
                                        from, to, &sink_step)) {
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
void Fp16Attention::precompute_freqs(int head_dim, unsigned int seq_len,
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

std::string Fp16Attention::getRopeCacheKey(int head_dim, unsigned int seq_len,
                                           float theta) const {
  std::ostringstream ss;
  ss << rope_scaling_type << "|" << head_dim << "|" << seq_len << "|" << theta
     << "|" << scale << "|" << rope_partial_rotary_factor << "|"
     << original_max_position_embeddings;
  return ss.str();
}

void Fp16Attention::_compute_default_parameters(int head_dim, float theta) {

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

void Fp16Attention::_compute_proportional_parameters(int head_dim,
                                                     float theta) {
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

void Fp16Attention::_compute_yarn_parameters(int head_dim, float theta) {

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

void Fp16Attention::apply_rotary_emb_tensor_v2(nntrainer::Tensor &in,
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
  // Build RoPE only for positions the cache can hold.
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
#ifdef NNTR_ARM_FP16_ATTENTION
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
#elif defined(ENABLE_FP16)
    NNTR_THROW_IF(true, std::invalid_argument)
      << "FP16 input attention is unsupported by this CPU backend";
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void Fp16Attention::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
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
#ifdef NNTR_ARM_FP16_ATTENTION
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
#elif defined(ENABLE_FP16)
    NNTR_THROW_IF(true, std::invalid_argument)
      << "FP16 input attention is unsupported by this CPU backend";
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void Fp16Attention::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
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
#ifdef NNTR_ARM_FP16_ATTENTION
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
#elif defined(ENABLE_FP16)
    NNTR_THROW_IF(true, std::invalid_argument)
      << "FP16 input attention is unsupported by this CPU backend";
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void Fp16Attention::compute_fp16vcache_transposed(
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
#ifdef NNTR_ARM_FP16_ATTENTION
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
#elif defined(ENABLE_FP16)
    NNTR_THROW_IF(true, std::invalid_argument)
      << "FP16 input attention is unsupported by this CPU backend";
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void Fp16Attention::setBatch(nntrainer::RunLayerContext &context,
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

void Fp16Attention::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  unsigned int height = input_dimensions[0].height();
  unsigned int &max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();
  unsigned int &max_new_tokens =
    std::get<attention_props::MaxNewTokens>(mha_core_props).get();
  max_position_embeddings =
    std::get<attention_props::MaxPositionEmbeddings>(mha_core_props).get();
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

void Fp16Attention::exportTo(nntrainer::Exporter &exporter,
                             const ml::train::ExportMethods &method,
                             const Layer *owner) const {
  exporter.saveResult(mha_core_props, method, owner);
}

std::vector<std::string>
Fp16Attention::setProperty(const std::vector<std::string> &values) {
  std::vector<std::string> props;
  props.reserve(values.size());
  for (const auto &value : values) {
    std::string key;
    std::string parsed_value;
    if (nntrainer::getKeyValue(value, key, parsed_value) == ML_ERROR_NONE &&
        key == "cache_index") {
      setCacheIndex(static_cast<unsigned int>(std::stoul(parsed_value)));
    } else if (key == "cache_base") {
      setCacheBase(static_cast<unsigned int>(std::stoul(parsed_value)));
    } else {
      props.push_back(value);
    }
  }

  auto remain_props = loadProperties(props, mha_core_props);
  return remain_props;
}

size_t Fp16Attention::calc_attn_index(size_t i) { return (i * (i + 1)) / 2; };

size_t Fp16Attention::calc_windowed_attn_index(size_t i) {
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

void Fp16Attention::registerQuantizedCache(RunLayerContext &context) {
  compute_ops_ = context.getComputeOps();
  const auto &key =
    use_external_cache
      ? context.getInput(3)
      : context.getTensor(tensor_idx[AttentionParams::cache_key]);
  state_->registerQuantizedCache(key.height());
}
void Fp16Attention::beginHtpPrompt() { state_->beginHtpPrompt(); }
void Fp16Attention::setCacheIndex(unsigned int index) {
  cache_index = index;
  state_->invalidate();
}
void Fp16Attention::setCacheBase(unsigned int base) {
  if (base != cache_base) {
    cache_base = base;
    state_->invalidate();
  }
}
void Fp16Attention::foldDroppedAmax(const std::vector<float> &k,
                                    const std::vector<float> &v) {
  state_->foldDroppedAmax(k, v);
}
void Fp16Attention::saveQ2Calibration(std::ostream &file) {
  HtpAttentionState::saveQ2Calibration(file);
}
void Fp16Attention::loadQ2Calibration(std::istream &file) {
  HtpAttentionState::loadQ2Calibration(file);
}
bool Fp16Attention::htpDecodeAttention(RunLayerContext &context,
                                       unsigned int pos) {
  auto make_rope = [&](std::vector<float> &table) {
    const unsigned int rows =
      std::get<props::MaxTimestep>(mha_core_props).get();
    if (!freqs_fp32) {
      const std::lock_guard<std::mutex> lock(rope_init_mtx);
      if (!freqs_fp32)
        precompute_freqs(head_dim, std::min(max_position_embeddings, rows),
                         theta, false);
    }
    NNTR_THROW_IF(freqs_fp32->cos.size() < rows, std::runtime_error)
      << "mha_core: RoPE cache is shorter than max_timestep";
    const unsigned half = head_dim / 2;
    table.resize(static_cast<size_t>(rows) * head_dim);
    for (unsigned i = 0; i < rows; ++i) {
      auto *t = table.data() + static_cast<size_t>(i) * head_dim;
      std::memcpy(t, freqs_fp32->cos[i].data(), half * sizeof(float));
      std::memcpy(t + half, freqs_fp32->sin[i].data(), half * sizeof(float));
    }
  };
  auto &ck = use_external_cache
               ? context.getInput(3)
               : context.getTensor(tensor_idx[AttentionParams::cache_key]);
  auto &cv = use_external_cache
               ? context.getInput(4)
               : context.getTensor(tensor_idx[AttentionParams::cache_value]);
  return state_->htpDecodeAttention(
    context.getInput(QUERY), context.getInput(KEY), context.getInput(VALUE),
    context.getOutput(OUTPUT), ck, cv, pos, make_rope);
}

} // namespace nntrainer

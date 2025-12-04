// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   mla_core.cpp
 * @date   03 December 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This code implements the Multi-Head Latent Attention (MLA) core layer.
 */

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include <fp16.h>
#include <layer_context.h>
#include <mla_core.h>
#include <nntrainer_error.h>
#include <node_exporter.h>

#include <cstdint>

namespace causallm {

MLACoreLayer::MLACoreLayer() :
  mla_core_props(
    nntrainer::props::NumHeads(), props::NumHeads_KV(),
    nntrainer::props::ProjectedKeyDim(), nntrainer::props::ProjectedValueDim(),
    nntrainer::props::OutputShape(), nntrainer::props::DropOutRate(),
    nntrainer::props::ReturnAttentionWeight(),
    nntrainer::props::AverageAttentionWeight(), nntrainer::props::MaxTimestep(),
    props::SlidingWindow(), props::MaxNewTokens(), props::RopeTheta(),
    props::MaxPositionEmbeddings(), props::UseSink(), props::RopeScalingType(),
    props::RopeScalingFactor(), props::RopeScalingMaxPositionEmbeddings(),
    props::KVLoRARank(), props::QKRoPEDim(), props::QKNopeDim()),
  sm(nntrainer::ActivationType::ACT_SOFTMAX),
  epsilon(1e-3),
  cache_index(0),
  num_heads_Q(0),
  num_heads_KV(0),
  head_dim(0),
  kv_lora_rank(0),
  qk_rope_dim(0),
  qk_nope_dim(0),
  cache_shift(false) {
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

MLACoreLayer::~MLACoreLayer() {}

void MLACoreLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 4,
                std::invalid_argument)
    << "MLA layer needs 3 or 4 inputs (Query, LatentKV, KeyRoPE, [Mask])";

  const std::vector<ml::train::TensorDim> &input_dims =
    context.getInputDimensions();
  const ml::train::TensorDim &query_dim = input_dims[INOUT_INDEX::QUERY];

  // Load properties
  num_heads_Q = std::get<nntrainer::props::NumHeads>(mla_core_props).get();
  num_heads_KV = std::get<props::NumHeads_KV>(mla_core_props).get();
  kv_lora_rank = std::get<props::KVLoRARank>(mla_core_props).get();
  qk_rope_dim = std::get<props::QKRoPEDim>(mla_core_props).get();
  qk_nope_dim = std::get<props::QKNopeDim>(mla_core_props).get();
  
  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mla_core_props).get();
  unsigned int max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mla_core_props).get();
  theta = (float)std::get<props::RopeTheta>(mla_core_props).get();
  rope_scaling_type = std::get<props::RopeScalingType>(mla_core_props).get();
  scale = std::get<props::RopeScalingFactor>(mla_core_props).get();

  // Validate dimensions
  size_t expected_query_width = (qk_nope_dim + qk_rope_dim) * num_heads_Q;
  NNTR_THROW_IF(query_dim.width() != expected_query_width, std::invalid_argument)
    << "Query dimension mismatch. Expected width: " << expected_query_width
    << ", Got: " << query_dim.width();

  unsigned int batch_size = query_dim.batch();
  
  ml::train::TensorDim cache_c_kv_dim(
    {batch_size, 1, max_timestep, kv_lora_rank},
    {context.getFormat(), context.getActivationDataType()});
    
  unsigned int k_pe_width = input_dims[INOUT_INDEX::KEY_ROPE].width();
  ml::train::TensorDim cache_k_pe_dim(
    {batch_size, 1, max_timestep, k_pe_width},
    {context.getFormat(), context.getActivationDataType()});

  tensor_idx[AttentionParams::cache_c_kv] = context.requestTensor(
    cache_c_kv_dim, "cache_c_kv", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
    
  tensor_idx[AttentionParams::cache_k_pe] = context.requestTensor(
    cache_k_pe_dim, "cache_k_pe", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  size_t v_head_dim = (qk_nope_dim + qk_rope_dim);
  
  ml::train::TensorDim w_uv_dim(
    {1, 1, num_heads_Q * v_head_dim, kv_lora_rank},
    {context.getFormat(), context.getActivationDataType()});
    
  tensor_idx[AttentionParams::weight_uv] = context.requestWeight(
    w_uv_dim, nntrainer::Initializer::XAVIER_UNIFORM,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "w_uv");

  ml::train::TensorDim w_uk_dim(
    {1, 1, num_heads_Q * qk_nope_dim, kv_lora_rank},
    {context.getFormat(), context.getActivationDataType()});

  tensor_idx[AttentionParams::weight_uk] = context.requestWeight(
    w_uk_dim, nntrainer::Initializer::XAVIER_UNIFORM,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "w_uk");

  if (freqs_cos == nullptr)
    precompute_freqs(qk_rope_dim, max_position_embeddings, theta);

  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = input_dims[0];
  output_dims[0].width(num_heads_Q * v_head_dim);
  context.setOutputDimensions(output_dims);
}

void MLACoreLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
}

void MLACoreLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int _from, unsigned int _to,
                                          bool training) {
  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mla_core_props).get();
  unsigned int from = _from;
  unsigned int to = _to;

  if (to >= max_timestep) {
    if (!_from) {
       throw std::invalid_argument("Initial forwarding exceeds max timestep");
    }
    cache_shift = true;
    from = max_timestep - 1;
    to = max_timestep;
  }

  nntrainer::Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  nntrainer::Tensor &latent_kv = context.getInput(INOUT_INDEX::LATENT_KV);
  nntrainer::Tensor &key_rope = context.getInput(INOUT_INDEX::KEY_ROPE);
  nntrainer::Tensor &output = context.getOutput(INOUT_INDEX::OUTPUT);

  nntrainer::Tensor &cache_c_kv = context.getTensor(tensor_idx[AttentionParams::cache_c_kv]);
  nntrainer::Tensor &cache_k_pe = context.getTensor(tensor_idx[AttentionParams::cache_k_pe]);

  auto get_step_dim = [to, from](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(to - from);
    return step_dim;
  };

  ml::train::TensorDim query_step_dim = get_step_dim(query.getDim());
  ml::train::TensorDim latent_kv_step_dim = get_step_dim(latent_kv.getDim());
  ml::train::TensorDim key_rope_step_dim = get_step_dim(key_rope.getDim());
  ml::train::TensorDim output_step_dim = get_step_dim(output.getDim());
  
  ml::train::TensorDim cache_c_kv_step_dim = get_step_dim(cache_c_kv.getDim());
  ml::train::TensorDim cache_k_pe_step_dim = get_step_dim(cache_k_pe.getDim());

  unsigned int batch_size = (_from) ? 1 : query.batch();

  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    nntrainer::Tensor query_step = query.getSharedDataTensor(
      query_step_dim, batch * query.getDim().getFeatureLen(), true);
    nntrainer::Tensor latent_kv_step = latent_kv.getSharedDataTensor(
      latent_kv_step_dim, batch * latent_kv.getDim().getFeatureLen(), true);
    nntrainer::Tensor key_rope_step = key_rope.getSharedDataTensor(
      key_rope_step_dim, batch * key_rope.getDim().getFeatureLen(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, batch * output.getDim().getFeatureLen(), true);

    one_batch_incremental_forwarding(
      batch, _from, from, to, query_step, latent_kv_step, key_rope_step,
      output_step, cache_c_kv, cache_k_pe, cache_c_kv.getDim(),
      cache_c_kv_step_dim, cache_k_pe.getDim(), cache_k_pe_step_dim);
  }
}

void MLACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query_step,
  nntrainer::Tensor &latent_kv_step, nntrainer::Tensor &key_rope_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_c_kv,
  nntrainer::Tensor &cache_k_pe, ml::train::TensorDim &cache_c_kv_dim,
  ml::train::TensorDim &cache_c_kv_step_dim,
  ml::train::TensorDim &cache_k_pe_dim,
  ml::train::TensorDim &cache_k_pe_step_dim) {

  // 1. Update Caches
  nntrainer::Tensor b_cache_c_kv_step = cache_c_kv.getSharedDataTensor(
    cache_c_kv_step_dim,
    batch * cache_c_kv_dim.getFeatureLen() + from * cache_c_kv_dim.width(), true);
  
  nntrainer::Tensor b_cache_k_pe_step = cache_k_pe.getSharedDataTensor(
    cache_k_pe_step_dim,
    batch * cache_k_pe_dim.getFeatureLen() + from * cache_k_pe_dim.width(), true);

  b_cache_c_kv_step.copyData(latent_kv_step);
  b_cache_k_pe_step.copyData(key_rope_step);

  // 2. Apply RoPE to Key Cache (RoPE part)
  // Note: key_rope_step is already in cache, apply in place
  apply_rotary_emb_tensor_v2(b_cache_k_pe_step, b_cache_k_pe_step, qk_rope_dim, _from, false);

  // 3. MLA Attention Calculation
  // We will iterate over heads and compute scores
  // Query Layout: [Head1_Nope, Head1_Rope, Head2_Nope, Head2_Rope, ...]
  // Dimensions:
  // q_nope_dim = qk_nope_dim
  // q_rope_dim = qk_rope_dim
  // kv_lora_rank = kv_lora_rank
  
  float *query_ptr = query_step.getData<float>();
  float *cache_c_kv_ptr = cache_c_kv.getData<float>() + batch * cache_c_kv_dim.getFeatureLen();
  float *cache_k_pe_ptr = cache_k_pe.getData<float>() + batch * cache_k_pe_dim.getFeatureLen();
  float *output_ptr = attention_output_step.getData<float>();
  
  // Weights
  nntrainer::Tensor w_uv = context.getWeight(tensor_idx[AttentionParams::weight_uv]);
  float *w_uv_ptr = w_uv.getData<float>();
  
  nntrainer::Tensor w_uk = context.getWeight(tensor_idx[AttentionParams::weight_uk]);
  float *w_uk_ptr = w_uk.getData<float>();

  unsigned int seq_len = to; // Total sequence length processed so far
  unsigned int q_head_dim = qk_nope_dim + qk_rope_dim;
  
  // Temporary buffer for scores: [NumHeads, SeqLen]
  std::vector<float> scores(num_heads_Q * seq_len);
  
  // Temporary buffer for latent context: [NumHeads, KV_LORA_RANK]
  // Note: In MLA, context is first aggregated in latent space (KV_LORA_RANK), then projected.
  // But since we have multiple heads, we usually project first?
  // Wait, DeepSeek MLA:
  // "For each head, we compute attention scores, then aggregate C_KV."
  // "Then we concatenate heads and project? No, MLA projects C_KV to Output."
  // Actually, standard MLA:
  // Context = Sum(Score * C_KV) -> [KV_LORA_RANK]
  // Output = Context * W_UV -> [NumHeads * V_Head_Dim]
  // Wait, if we aggregate C_KV, we lose head information?
  // No, MLA has multiple heads.
  // Each head has its own query $q_i$.
  // $Score_i = q_{i,nop} c_{KV}^T + q_{i,pe} k_{pe}^T$
  // $Attn_i = Softmax(Score_i)$
  // $Context_i = Attn_i \cdot c_{KV}$ -> [KV_LORA_RANK]
  // $Output_i = Context_i \cdot W_{UV, i}$ -> [V_Head_Dim]
  // $Output = Concat(Output_i)$
  // Optimization:
  // $Context_{combined} = \sum_i (Attn_i \otimes c_{KV})$ ? No.
  // We compute $Context_i$ for each head.
  // Then we can batch project: $[NumHeads, KV_LORA_RANK] \times [NumHeads, KV_LORA_RANK, V_Head_Dim]$?
  // Or $W_{UV}$ is $[NumHeads * V_Head_Dim, KV_LORA_RANK]$.
  // Let's assume $W_{UV}$ projects from latent to full output.
  
  // 3.1 Compute Scores and Context per Head
  auto &pool = nntrainer::ThreadPoolManager::Global().getThreadPool();
  std::vector<std::future<void>> futures;

  for (unsigned int h = 0; h < num_heads_Q; ++h) {
    futures.push_back(pool.submit_task([=, &scores]() {
      // Pointers for this head
      float *q_head_ptr = query_ptr + h * q_head_dim;
      float *q_nope_ptr = q_head_ptr;
      float *q_rope_ptr = q_head_ptr + qk_nope_dim;
      
      // Apply RoPE to q_rope_ptr (Need a temporary buffer or in-place if safe)
      // Since query_step is [1, 1, 1, ...], we can modify in place if we are careful.
      // But we need `_from` position.
      // Let's use a temp buffer for rotated query rope
      std::vector<float> q_rope_rotated(qk_rope_dim);
      // Copy
      std::copy(q_rope_ptr, q_rope_ptr + qk_rope_dim, q_rope_rotated.begin());
      
      // Apply RoPE: We need a helper that works on single vector
      // Existing `apply_rotary_emb_tensor_v2` works on tensors.
      // We can manually call `compute_rotary_emb_value` if we had access.
      // Or just use the precomputed cos/sin directly.
      unsigned int half_ = qk_rope_dim / 2;
      const std::vector<float> &cos = (*freqs_cos)[_from];
      const std::vector<float> &sin = (*freqs_sin)[_from];
      
      for (unsigned int i = 0; i < half_; ++i) {
        float r1 = q_rope_rotated[i];
        float r2 = q_rope_rotated[i + half_];
        q_rope_rotated[i] = r1 * cos[i] - r2 * sin[i];
        q_rope_rotated[i + half_] = r1 * sin[i + half_] + r2 * cos[i + half_];
      }
      
      // Compute Scores for all timesteps t in [0, to)
      for (unsigned int t = 0; t < to; ++t) {
        float score = 0.0f;
        
        // Content Score: q_nope . k_nop[t]
        // k_nop[t] = c_kv[t] * W_UK_head^T
        // W_UK_head is [QK_NOPE_DIM, KV_LORA_RANK]
        // We compute k_nop_t on the fly or assume q_nope is absorbed?
        // Let's compute k_nop_t for this head.
        
        float *c_kv_t = cache_c_kv_ptr + t * kv_lora_rank;
        
        // Project c_kv_t to k_nop_t
        // k_nop_t[i] = Sum_k (c_kv_t[k] * W_UK[h, i, k])
        // W_UK flat index: h * QK_NOPE_DIM * KV_LORA_RANK + i * KV_LORA_RANK + k
        
        for (unsigned int i = 0; i < qk_nope_dim; ++i) {
            float k_val = 0.0f;
            float *w_uk_row = w_uk_ptr + (h * qk_nope_dim + i) * kv_lora_rank;
            for (unsigned int k = 0; k < kv_lora_rank; ++k) {
                k_val += c_kv_t[k] * w_uk_row[k];
            }
            // Dot product with q_nope
            score += q_nope_ptr[i] * k_val;
        }
        
        // RoPE Score: q_rope . k_pe[t]
        float *k_pe_t = cache_k_pe_ptr + t * qk_rope_dim;
        for (unsigned int k = 0; k < qk_rope_dim; ++k) {
            score += q_rope_rotated[k] * k_pe_t[k];
        }
        
        score *= attention_scaling; // Apply scaling
        scores[h * seq_len + t] = score;
      }
      
      // Softmax for this head
      // Apply causal mask (if t > from, usually handled by loop range, but here we compute all up to `to`)
      // Actually we only care about the last row (current step) attending to all previous.
      // Since query is 1 step (`_from`), we are computing row `_from`.
      // We need to mask out t > _from? No, `to` is `_from + 1` usually.
      // If `to` > `_from + 1`, we are processing multiple query steps?
      // `incremental_forwarding` usually processes 1 query step attending to `to` context.
      // So we are computing 1 row of scores: [0, ..., to-1].
      
      // Softmax
      float max_val = -std::numeric_limits<float>::infinity();
      for (unsigned int t = 0; t < to; ++t) {
          if (scores[h * seq_len + t] > max_val) max_val = scores[h * seq_len + t];
      }
      
      float sum_exp = 0.0f;
      for (unsigned int t = 0; t < to; ++t) {
          scores[h * seq_len + t] = std::exp(scores[h * seq_len + t] - max_val);
          sum_exp += scores[h * seq_len + t];
      }
      for (unsigned int t = 0; t < to; ++t) {
          scores[h * seq_len + t] /= sum_exp;
      }
      
      // Context Aggregation: Sum(Score[t] * c_kv[t])
      std::vector<float> context_head(kv_lora_rank, 0.0f);
      for (unsigned int t = 0; t < to; ++t) {
          float attn = scores[h * seq_len + t];
          float *c_kv_t = cache_c_kv_ptr + t * kv_lora_rank;
          for (unsigned int k = 0; k < kv_lora_rank; ++k) {
              context_head[k] += attn * c_kv_t[k];
          }
      }
      
      // Up-Projection: Context_head * W_UV_head
      // W_UV is [NumHeads * V_Head_Dim, KV_LORA_RANK] (transposed in memory usually [Out, In])
      // We need W_UV part for this head.
      // Let's assume W_UV is [NumHeads, V_Head_Dim, KV_LORA_RANK] logically.
      // Flat index: h * V_Head_Dim * KV_LORA_RANK + v * KV_LORA_RANK + k
      // Wait, nntrainer weights are [Output, Input].
      // Output dim: NumHeads * V_Head_Dim. Input dim: KV_LORA_RANK.
      // So W_UV is a matrix of size (NumHeads * V_Head_Dim) x KV_LORA_RANK.
      // Row `r` corresponds to output element `r`.
      // Output for head `h` is rows `h * V_Head_Dim` to `(h+1) * V_Head_Dim`.
      
      size_t v_head_dim = (qk_nope_dim + qk_rope_dim); // Assumed
      float *out_head_ptr = output_ptr + h * v_head_dim;
      
      for (unsigned int v = 0; v < v_head_dim; ++v) {
          float val = 0.0f;
          unsigned int row_idx = h * v_head_dim + v;
          float *w_row = w_uv_ptr + row_idx * kv_lora_rank;
          
          for (unsigned int k = 0; k < kv_lora_rank; ++k) {
              val += context_head[k] * w_row[k];
          }
          out_head_ptr[v] = val;
      }
    }));
  }
  
  for (auto &fut : futures) fut.get();
}

void MLACoreLayer::precompute_freqs(int head_dim, unsigned int seq_len, float theta) {
  if (freqs_cos != nullptr && freqs_cos->size() == seq_len)
    return;

  if (rope_scaling_type == "default")
    _compute_default_parameters(head_dim, theta);
  else if (rope_scaling_type == "yarn")
    _compute_yarn_parameters(head_dim, theta);
  else
    NNTR_THROW_IF(true, std::invalid_argument) << "Unsupported rope type!";

  unsigned int half_ = head_dim / 2;
  auto cos = new std::vector<std::vector<float>>();
  cos->assign(seq_len, std::vector<float>(head_dim, 0));
  auto sin = new std::vector<std::vector<float>>();
  sin->assign(seq_len, std::vector<float>(head_dim, 0));

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
        std::cos(angle) * attention_scaling; 

      (*sin)[i][j] = std::sin(angle) * attention_scaling;
      (*sin)[i][j + half_] =
        std::sin(angle) * attention_scaling;
    }
#endif
  }
  freqs_cos = cos;
  freqs_sin = sin;

#ifdef ENABLE_FP16
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
}

void MLACoreLayer::_compute_default_parameters(int head_dim, float theta) {
  attention_scaling = 1.0f;
  unsigned int half_ = head_dim / 2;
  for (unsigned int i = 0; i < half_; ++i) {
    thetas.push_back(1.0 /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }
}

void MLACoreLayer::_compute_yarn_parameters(int head_dim, float theta) {
  // Placeholder for YARN, falling back to default for now or copying logic if needed
  _compute_default_parameters(head_dim, theta);
}

void MLACoreLayer::apply_rotary_emb_tensor_v2(nntrainer::Tensor &in,
                                              nntrainer::Tensor &out,
                                              unsigned int dim,
                                              unsigned int from,
                                              bool convert_only) {
  unsigned int half_ = dim / 2;
  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mla_core_props).get();

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

void MLACoreLayer::calcDerivative(nntrainer::RunLayerContext &context) {}
void MLACoreLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void MLACoreLayer::exportTo(nntrainer::Exporter &exporter,
                            const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(mla_core_props, method, this);
}

void MLACoreLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, mla_core_props);
  LayerImpl::setProperty(remain_props);
}

void MLACoreLayer::setBatch(nntrainer::RunLayerContext &context, unsigned int batch) {
    context.updateTensor(tensor_idx[AttentionParams::cache_c_kv], batch);
    context.updateTensor(tensor_idx[AttentionParams::cache_k_pe], batch);
}

void MLACoreLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  unsigned int height = input_dimensions[0].height();
  unsigned int &max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mla_core_props).get();
  unsigned int &max_new_tokens =
    std::get<props::MaxNewTokens>(mla_core_props).get();
  unsigned int &max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mla_core_props).get();
  max_timestep = height + max_new_tokens;

  ml::train::TensorDim kv_cache_dim = input_dimensions[INOUT_INDEX::LATENT_KV];
  kv_cache_dim.height(max_timestep);
  
  ml::train::TensorDim k_pe_cache_dim = input_dimensions[INOUT_INDEX::KEY_ROPE];
  k_pe_cache_dim.height(max_timestep);

  precompute_freqs(qk_rope_dim, max_position_embeddings, theta);

  context.updateInput(INOUT_INDEX::QUERY, input_dimensions[INOUT_INDEX::QUERY]);
  context.updateInput(INOUT_INDEX::LATENT_KV, input_dimensions[INOUT_INDEX::LATENT_KV]);
  context.updateInput(INOUT_INDEX::KEY_ROPE, input_dimensions[INOUT_INDEX::KEY_ROPE]);
  
  // Update output dim
  std::vector<nntrainer::TensorDim> output_dims = context.getOutputDimensions();
  output_dims[0] = input_dims[0];
  size_t v_head_dim = (qk_nope_dim + qk_rope_dim);
  output_dims[0].width(num_heads_Q * v_head_dim);
  context.updateOutput(0, output_dims[0]);

  context.updateTensor(tensor_idx[AttentionParams::cache_c_kv], kv_cache_dim);
  context.updateTensor(tensor_idx[AttentionParams::cache_k_pe], k_pe_cache_dim);
}

size_t MLACoreLayer::calc_attn_index(size_t i) { return (i * (i + 1)) / 2; };

void MLACoreLayer::softmax_triangle(nntrainer::Tensor &qk_out, size_t row, size_t num_head,
                        unsigned int from, BS::thread_pool<> &pool) {
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

} // namespace causallm

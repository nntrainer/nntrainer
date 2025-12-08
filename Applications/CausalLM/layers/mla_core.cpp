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
    props::RopeScalingBetaFast(), props::RopeScalingBetaSlow(),
    props::RopeScalingMscale(), props::RopeScalingMscaleAllDim(),
    props::KVLoRARank(), props::QKRoPEDim(), props::QKNopeDim(),
    props::VHeadDim()),
  sm(nntrainer::ActivationType::ACT_SOFTMAX),
  epsilon(1e-3),
  cache_index(0),
  num_heads_Q(0),
  num_heads_KV(0),
  head_dim(0),
  kv_lora_rank(0),
  qk_rope_dim(0),
  qk_nope_dim(0),
  v_head_dim(0),
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
  v_head_dim = std::get<props::VHeadDim>(mla_core_props).get();
  
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
  
  // Cache K (Full K: Nope + RoPE)
  // Dim: [Batch, 1, MaxTimestep, NumHeads * (QK_NOPE + QK_ROPE)]
  unsigned int k_head_dim = qk_nope_dim + qk_rope_dim;
  ml::train::TensorDim cache_k_dim(
    batch_size, 1, max_timestep, num_heads_Q * k_head_dim,
    context.getFormat(), context.getActivationDataType());

  // Cache V
  // Dim: [Batch, 1, MaxTimestep, NumHeads * V_HEAD]
  ml::train::TensorDim cache_v_dim(
    batch_size, 1, max_timestep, num_heads_Q * v_head_dim,
    context.getFormat(), context.getActivationDataType());

  tensor_idx[AttentionParams::cache_c_kv] = context.requestTensor(
    cache_k_dim, "cache_k", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  tensor_idx[AttentionParams::cache_k_pe] = context.requestTensor(
    cache_v_dim, "cache_v", nntrainer::Initializer::NONE, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);

  if (freqs_cos == nullptr)
    precompute_freqs(qk_rope_dim, max_position_embeddings, theta);

  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = input_dims[0]; // Copy dims from Query (Batch, 1, Seq)
  output_dims[0].width(num_heads_Q * v_head_dim); // Set width
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
  nntrainer::Tensor &kv_b_output = context.getInput(INOUT_INDEX::LATENT_KV); // Reused index for KV_B
  nntrainer::Tensor &key_rope = context.getInput(INOUT_INDEX::KEY_ROPE);
  nntrainer::Tensor &output = context.getOutput(INOUT_INDEX::OUTPUT);

  nntrainer::Tensor &cache_k = context.getTensor(tensor_idx[AttentionParams::cache_c_kv]);
  nntrainer::Tensor &cache_v = context.getTensor(tensor_idx[AttentionParams::cache_k_pe]);

  auto get_step_dim = [to, from](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(to - from);
    return step_dim;
  };

  ml::train::TensorDim query_step_dim = get_step_dim(query.getDim());
  ml::train::TensorDim kv_b_step_dim = get_step_dim(kv_b_output.getDim());
  ml::train::TensorDim key_rope_step_dim = get_step_dim(key_rope.getDim());
  ml::train::TensorDim output_step_dim = get_step_dim(output.getDim());
  
  ml::train::TensorDim cache_k_step_dim = get_step_dim(cache_k.getDim());
  ml::train::TensorDim cache_v_step_dim = get_step_dim(cache_v.getDim());

  unsigned int batch_size = (_from) ? 1 : query.batch();

  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    nntrainer::Tensor query_step = query.getSharedDataTensor(
      query_step_dim, batch * query.getDim().getFeatureLen(), true);
    nntrainer::Tensor kv_b_step = kv_b_output.getSharedDataTensor(
      kv_b_step_dim, batch * kv_b_output.getDim().getFeatureLen(), true);
    nntrainer::Tensor key_rope_step = key_rope.getSharedDataTensor(
      key_rope_step_dim, batch * key_rope.getDim().getFeatureLen(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, batch * output.getDim().getFeatureLen(), true);

    one_batch_incremental_forwarding(
      batch, _from, from, to, query_step, kv_b_step, key_rope_step,
      output_step, cache_k, cache_v, cache_k.getDim(),
      cache_k_step_dim, cache_v.getDim(), cache_v_step_dim);
  }
  // std::cout <<"\n " << context.getName() <<"'s Outputs " << std::endl;
  // output.print(std::cout);
  // std::cout << "============================================" << std::endl;
}

void MLACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query_step,
  nntrainer::Tensor &kv_b_step, nntrainer::Tensor &key_rope_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_k,
  nntrainer::Tensor &cache_v, const ml::train::TensorDim &cache_k_dim,
  const ml::train::TensorDim &cache_k_step_dim,
  const ml::train::TensorDim &cache_v_dim,
  const ml::train::TensorDim &cache_v_step_dim) {

  unsigned int step_len = to - from;
  float *kv_b_ptr = kv_b_step.getData<float>();
  float *key_rope_ptr = key_rope_step.getData<float>();
  
  float *cache_k_ptr = cache_k.getData<float>() + batch * cache_k_dim.getFeatureLen() + from * cache_k_dim.width();
  float *cache_v_ptr = cache_v.getData<float>() + batch * cache_v_dim.getFeatureLen() + from * cache_v_dim.width();

  unsigned int kv_b_head_dim = qk_nope_dim + v_head_dim;
  unsigned int k_full_dim = qk_nope_dim + qk_rope_dim;

  // Update Caches (Unchanged)
  for (unsigned int t = 0; t < step_len; ++t) {
    for (unsigned int h = 0; h < num_heads_Q; ++h) {
      float *src_kv_b = kv_b_ptr + t * num_heads_Q * kv_b_head_dim + h * kv_b_head_dim;
      float *src_k_rope = key_rope_ptr + t * qk_rope_dim; // Broadcasted RoPE key
      
      float *dst_k = cache_k_ptr + t * num_heads_Q * k_full_dim + h * k_full_dim;
      float *dst_v = cache_v_ptr + t * num_heads_Q * v_head_dim + h * v_head_dim;
      
      std::copy(src_kv_b, src_kv_b + qk_nope_dim, dst_k);
      std::copy(src_k_rope, src_k_rope + qk_rope_dim, dst_k + qk_nope_dim);
      std::copy(src_kv_b + qk_nope_dim, src_kv_b + kv_b_head_dim, dst_v);
    }
  }

  // Apple RoPE to Cache K (Unchanged)
  for (unsigned int t = 0; t < step_len; ++t) {
    unsigned int pos = from + t;
    if (pos >= (*freqs_cos).size()) continue;
    
    const std::vector<float> &cos_t = (*freqs_cos)[pos];
    const std::vector<float> &sin_t = (*freqs_sin)[pos];
    unsigned int half_ = qk_rope_dim / 2;
    
    for (unsigned int h = 0; h < num_heads_Q; ++h) {
      float *k_head_ptr = cache_k_ptr + t * num_heads_Q * k_full_dim + h * k_full_dim;
      float *k_rope_ptr = k_head_ptr + qk_nope_dim;
      
      for (unsigned int i = 0; i < half_; ++i) {
        float r1 = k_rope_ptr[2 * i];
        float r2 = k_rope_ptr[2 * i + 1];
        k_rope_ptr[2 * i] = r1 * cos_t[i] - r2 * sin_t[i];
        k_rope_ptr[2 * i + 1] = r1 * sin_t[i] + r2 * cos_t[i];
      }
    }
  }

  // 3. Attention Calculation
  float *query_ptr = query_step.getData<float>();
  float *output_ptr = attention_output_step.getData<float>();
  
  float *full_cache_k_ptr = cache_k.getData<float>() + batch * cache_k_dim.getFeatureLen();
  float *full_cache_v_ptr = cache_v.getData<float>() + batch * cache_v_dim.getFeatureLen();
  
  unsigned int q_head_dim = qk_nope_dim + qk_rope_dim;
  unsigned int seq_len = to; 
  
  auto &pool = nntrainer::ThreadPoolManager::Global().getThreadPool();
  std::vector<std::future<void>> futures;

  for (unsigned int h = 0; h < num_heads_Q; ++h) {
    futures.push_back(pool.submit_task([=]() {
      float score_scale = 1.0f / std::sqrt(static_cast<float>(qk_nope_dim + qk_rope_dim));

      for (unsigned int t_q = 0; t_q < step_len; ++t_q) {
        unsigned int q_idx = from + t_q; 

        // Query pointers for this step
        float *q_head_ptr = query_ptr + t_q * num_heads_Q * q_head_dim + h * q_head_dim;
        float *q_nope_ptr = q_head_ptr;
        float *q_rope_ptr = q_head_ptr + qk_nope_dim;

        // Apply RoPE to Query (Interleaved)
        std::vector<float> q_rope_rotated(qk_rope_dim);
        std::copy(q_rope_ptr, q_rope_ptr + qk_rope_dim, q_rope_rotated.begin());

        const std::vector<float> &cos = (*freqs_cos)[_from + t_q]; // Use correct position for query
        const std::vector<float> &sin = (*freqs_sin)[_from + t_q];
        unsigned int half_ = qk_rope_dim / 2;

        for (unsigned int i = 0; i < half_; ++i) {
          float r1 = q_rope_rotated[2 * i];
          float r2 = q_rope_rotated[2 * i + 1];
          q_rope_rotated[2 * i] = r1 * cos[i] - r2 * sin[i];
          q_rope_rotated[2 * i + 1] = r1 * sin[i] + r2 * cos[i];
        }

        // Compute Scores
        std::vector<float> scores(seq_len);
        float max_val = -std::numeric_limits<float>::infinity();

        for (unsigned int t_k = 0; t_k < seq_len; ++t_k) {
          // Causal Masking
          if (t_k > q_idx) {
            scores[t_k] = -std::numeric_limits<float>::infinity();
            continue;
          }

          float score = 0.0f;
          float *k_head_ptr = full_cache_k_ptr + t_k * num_heads_Q * k_full_dim + h * k_full_dim;

          // Nope part
          for (unsigned int i = 0; i < qk_nope_dim; ++i) {
            score += q_nope_ptr[i] * k_head_ptr[i];
          }

          // RoPE part
          float *k_rope_ptr = k_head_ptr + qk_nope_dim;
          for (unsigned int i = 0; i < qk_rope_dim; ++i) {
            score += q_rope_rotated[i] * k_rope_ptr[i];
          }

          score *= score_scale; // Correct scaling
          scores[t_k] = score;
          if (score > max_val) max_val = score;
        }

        // Softmax
        float sum_exp = 0.0f;
        for (unsigned int t_k = 0; t_k < seq_len; ++t_k) {
          if (scores[t_k] == -std::numeric_limits<float>::infinity()) {
            scores[t_k] = 0.0f;
          } else {
            scores[t_k] = std::exp(scores[t_k] - max_val);
            sum_exp += scores[t_k];
          }
        }
        for (unsigned int t_k = 0; t_k < seq_len; ++t_k) {
          scores[t_k] /= sum_exp;
        }

        // Weighted Sum of V
        float *out_head_ptr = output_ptr + t_q * num_heads_Q * v_head_dim + h * v_head_dim;
        std::fill(out_head_ptr, out_head_ptr + v_head_dim, 0.0f);

        for (unsigned int t_k = 0; t_k < seq_len; ++t_k) {
          float attn = scores[t_k];
          if (attn == 0.0f) continue;
          float *v_head_ptr = full_cache_v_ptr + t_k * num_heads_Q * v_head_dim + h * v_head_dim;
          for (unsigned int i = 0; i < v_head_dim; ++i) {
            out_head_ptr[i] += attn * v_head_ptr[i];
          }
        }
      } // End query loop
    }));
  }
  
  for (auto &f : futures) {
    f.wait();
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
  // Yarn implementation simplified:
  // For short sequences, the frequency correction is minimal.
  // The most important part is the attention scaling (mscale).
  
  float mscale = std::get<props::RopeScalingMscale>(mla_core_props).get();
  float mscale_all_dim = std::get<props::RopeScalingMscaleAllDim>(mla_core_props).get();
  
  attention_scaling = mscale * mscale_all_dim;
  
  // Use default frequency calculation for now
  // TODO: Implement full Yarn frequency correction if needed for long context
  unsigned int half_ = head_dim / 2;
  for (unsigned int i = 0; i < half_; ++i) {
    thetas.push_back(1.0 /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }
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
  // Update output dim
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = input_dimensions[0];
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

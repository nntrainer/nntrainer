// SPDX-License-Identifier: Apache-2.0
/** @file htp_attention_state.cpp
 * @brief Attention transport and quantized cache lifecycle.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#include "htp_attention_state.h"
#include "attention_quant_utils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <compute_ops.h>
#include <cpu_backend.h>
#include <cstdlib>
#include <cstring>
#include <htp_backend/htp_runtime_config.h>
#include <istream>
#include <memory>
#include <nntr_htp_decode_hook.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <ostream>
#include <thread_manager.h>
namespace nntrainer {
std::vector<uint16_t> HtpAttentionState::q2_q_u16,
  HtpAttentionState::q2_out_u16;
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

HtpPrefillPlan &htpPrefillPlan() {
  static HtpPrefillPlan plan;
  return plan;
}

bool htpKvAdopt() { return htpRuntimeConfig().htp.kv_adopt; }

void HtpAttentionState::beginHtpPrompt() {
  release_quantized_cache();
  config_.cache_index = 0;
  htp_owns_cache_ = false;
  kv_detached_ = false;
  htp_seed_ord_ = -1;
  q_cache_failed = false;
  q2_calibrated = false;
  kv_floor_k_.clear();
  kv_floor_v_.clear();
  htp_rope_table_.reset();
}

bool HtpAttentionState::htpDecodeAttention(
  Tensor &query, Tensor &key, Tensor &value, Tensor &output, Tensor &ck,
  Tensor &cv, unsigned int pos,
  const std::function<std::shared_ptr<const std::vector<float>>()> &make_rope) {
  if (query.getDataType() != ml::train::TensorDim::DataType::FP32 ||
      query.batch() != 1)
    return false;
  // The row is q | k | v of the step (row 0 of batch 0 on every path).
  const unsigned int wq = query.width(), wk = key.width(), wv = value.width();
  static thread_local std::vector<float> row;
  // A whole resident row uses the DSP's projections. This hook still binds
  // RoPE and seeds KV when needed, but its stale CPU q/k/v are never read.
  const bool resident = htpDecodeRowResident(pos);
  if (!resident) {
    row.resize(wq + wk + wv);
    std::memcpy(row.data(), query.getData<float>(), wq * sizeof(float));
    std::memcpy(row.data() + wq, key.getData<float>(), wk * sizeof(float));
    std::memcpy(row.data() + wq + wk, value.getData<float>(),
                wv * sizeof(float));
  }
  const float *qkv = resident ? nullptr : row.data();
  // Reuse CPU RoPE values as [cos hd/2 | sin hd/2] per position. Build
  // config_.head_dim 64 up front; other dimensions only when the hook requests
  // them.
  auto build_rope_table = [&] { htp_rope_table_ = make_rope(); };
  auto rope_data = [&]() -> const float * {
    return htp_rope_table_ ? htp_rope_table_->data() : nullptr;
  };
  auto rope_len = [&]() -> unsigned {
    return htp_rope_table_ ? static_cast<unsigned>(htp_rope_table_->size())
                           : 0u;
  };
  if (!htp_rope_table_ && config_.use_rope && config_.head_dim == 64)
    build_rope_table();
  int r = htpDecodeAttn(pos, qkv, wq + wk + wv, output.getData<float>(), wq,
                        rope_data(), rope_len());
  if (r == 3) {
    // DSP-resident q/k still need RoPE even when config_.use_rope is off here;
    // CPU-projected rows arrive rotated by qkv_layer.
    NNTR_THROW_IF(htp_rope_table_ != nullptr ||
                    (!config_.use_rope && !htpDecodeRowResident(pos)),
                  std::runtime_error)
      << "mha_core: the HTP attention hook asked for a RoPE table it has";
    build_rope_table();
    r = htpDecodeAttn(pos, qkv, wq + wk + wv, output.getData<float>(), wq,
                      rope_data(), rope_len());
    // The DSP owns a copy and requests it again only after graph init.
    if (r != 3)
      htp_rope_table_.reset();
  }
  if (r == 2) {
    // Seed the DSP cache from this layer's rows [0, pos): fp16 bits on
    // every platform's cache (FP16 or UINT16 storage), f32 when it is.
    // Release shared CPU cache blocks after seeding the DSP. Decode no
    // longer reads them, and htp_owns_cache_ rejects appended prefill.
    // drop_shared leaves non-shared blocks alone.
    auto drop_cpu_cache = [&]() {
      if (config_.compute_ops_)
        for (nntrainer::Tensor *t : {&ck, &cv})
          config_.compute_ops_->drop_shared(
            t->getDataType() == ml::train::TensorDim::DataType::FP32
              ? static_cast<void *>(t->getData<float>())
              : static_cast<void *>(t->getData<uint16_t>()));
    };
    // Reuse DSP int8 V when the configured seed path permits it.
    // K comes from the CPU cache because prefill's per-head scale is too
    // coarse for decode. Widen fp16 rows in parallel when needed.
    auto rows_f32_of = [wk](const nntrainer::Tensor &c, float *dst, size_t m,
                            size_t) {
      if (c.getDataType() == ml::train::TensorDim::DataType::FP32) {
        std::memcpy(dst, c.getData<float>(), m * sizeof(float));
        return;
      }
      const uint16_t *src = c.getData<uint16_t>();
      nntrainer::ThreadManager::Global().parallel_for(0, m / wk, [=](size_t r) {
        for (size_t i = r * wk; i < (r + 1) * wk; ++i)
          dst[i] = nntrainer::compute_fp16_to_fp32(src[i]);
      });
    };
    const bool seed_q = htpRuntimeConfig().htp.kv_seed_q;
    std::vector<float> k_only;
    if (seed_q && config_.cache_base == 0 && !q_cache_handles.empty() &&
        q_cache_handles[0] >= 0 && q_cache_synced[0] == pos && q2_scales_set[0])
      k_only.resize(static_cast<size_t>(pos) * wk);
    if (!k_only.empty())
      rows_f32_of(ck, k_only.data(), k_only.size(), wk);
    if (!k_only.empty() &&
        htpDecodeKvSeedKvq(pos, q_cache_handles[0], k_only.data())) {
      q_cache_handles[0] = -1; // released on the DSP
      drop_cpu_cache();
      r = htpDecodeAttn(pos, qkv, wq + wk + wv, output.getData<float>(), wq,
                        rope_data(), rope_len());
      NNTR_THROW_IF(r != 1, std::runtime_error)
        << "mha_core: the HTP attention hook asked for a seed twice at pos "
        << pos;
      return true;
    }
    const auto t_seed = std::chrono::steady_clock::now();
    // the cache's rows [0, pos - config_.cache_base): a windowed cache seeds
    // its window, the DSP's attention subtracts the same base
    NNTR_THROW_IF(pos < config_.cache_base, std::runtime_error)
      << "mha_core: decode at " << pos << " before the KV cache base "
      << config_.cache_base;
    const unsigned int seed_rows = pos - config_.cache_base;
    const size_t n = static_cast<size_t>(seed_rows) * wk;
    auto t_conv = t_seed;
    // The rows the host cache dropped are not in the seed:
    // their |max| goes ahead of it, so the scales cover the whole context
    if (!kv_floor_k_.empty())
      htpDecodeKvAmaxFloor(kv_floor_k_.data(), kv_floor_v_.data(),
                           static_cast<unsigned>(kv_floor_k_.size()));
    // An fp16 / UINT16 cache goes as its bits: the DSP widens
    // them exactly (hvx_hf_bits_to_sf), nothing converted or copied here
    const bool bits16 =
      ck.getDataType() != ml::train::TensorDim::DataType::FP32 &&
      cv.getDataType() != ml::train::TensorDim::DataType::FP32;
    if (!bits16 || !htpDecodeKvSeedF16(seed_rows, ck.getData<uint16_t>(),
                                       cv.getData<uint16_t>())) {
      std::vector<float> k_rows(n), v_rows(n);
      auto rows_f32 = [&](const nntrainer::Tensor &c, float *dst) {
        rows_f32_of(c, dst, n, wk);
      };
      rows_f32(ck, k_rows.data());
      rows_f32(cv, v_rows.data());
      t_conv = std::chrono::steady_clock::now();
      htpDecodeKvSeed(seed_rows, k_rows.data(), v_rows.data());
    }
    drop_cpu_cache();
    {
      static double conv_ms = 0.0, all_ms = 0.0;
      using ms = std::chrono::duration<double, std::milli>;
      conv_ms += ms(t_conv - t_seed).count();
      all_ms += ms(std::chrono::steady_clock::now() - t_seed).count();
      ml_logi("mha_core: t1 seed pos=%u rows_convert_ms_sum=%.1f "
              "seed_ms_sum=%.1f",
              pos, conv_ms, all_ms);
    }
    r = htpDecodeAttn(pos, qkv, wq + wk + wv, output.getData<float>(), wq,
                      rope_data(), rope_len());
    NNTR_THROW_IF(r != 1, std::runtime_error)
      << "mha_core: the HTP attention hook asked for a seed twice at pos "
      << pos;
  }
  return r == 1;
}

bool HtpAttentionState::try_accelerated_attention(
  nntrainer::Tensor &query_step, nntrainer::Tensor &cached_key,
  nntrainer::Tensor &cached_value, nntrainer::Tensor &attention_output_step,
  unsigned int cache_from, unsigned int cache_to,
  const nntrainer::Tensor *sink) {
  if (!config_.compute_ops_ ||
      !config_.compute_ops_->supports_sdpa_fp16_kvcache()) {
    return false;
  }
  // The accelerated kernel implements the causal, windowed attention this
  // layer computes over the fp16 cache -- the shape every CausalLM model
  // here runs -- with f32 query and output; fp16 steps (Android) are
  // staged through f32. Anything else stays on the CPU.
  if (!config_.is_causal) {
    return false;
  }
  if (!htpRuntimeConfig().htp.attn_f16) {
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
  const unsigned int q_stride = config_.num_heads_Q * config_.head_dim;
  const unsigned int kv_stride = config_.num_heads_KV * config_.head_dim;
  // config_.local_window_size defaults to UINT_MAX meaning "no window"; the
  // kernel spells that 0.
  const unsigned int window =
    (config_.local_window_size == 0 || config_.local_window_size >= cache_to)
      ? 0u
      : static_cast<unsigned int>(config_.local_window_size);
  if (!config_.compute_ops_->sdpa_fp16_kvcache(
        io.q, q_stride, k_bits, v_bits, kv_stride, n_q, cache_from, cache_to,
        config_.num_heads_Q, config_.num_heads_KV, config_.head_dim, window,
        config_.attn_logit_softcapping, io.sinks, io.out, q_stride)) {
    return false;
  }
  io.commit(attention_output_step);
  if (!accel_logged_) {
    accel_logged_ = true;
    ml_logi("mha_core: attention over the fp16 KV cache on the accelerator");
  }
  return true;
}

void HtpAttentionState::registerQuantizedCache(unsigned int cache_rows) {
  if (nntrainer::weightPageCachePolicy() >= 0)
    return; // The memory profile allocates only the current layer's layout.
  // Arena-backed FC weights leave room to register quantized caches at
  // load, avoiding first-prefill registration. Decode releases these caches.
  if (config_.kv_cache_quant_kind < 0 || q_cache_failed ||
      !config_.compute_ops_ || !config_.compute_ops_->supports_kv_cache_q() ||
      !config_.is_causal || !q_cache_handles.empty())
    return;
  // try_quantized_attention's own gate for a layer the int8 kernels take
  const bool q2 = config_.kv_cache_quant_kind == 0 &&
                  config_.kv_cache_fixed_scales &&
                  config_.compute_ops_->supports_kv_cache_q2() &&
                  config_.attn_logit_softcapping <= 0.0f;
  if (!q2 && config_.head_dim > 256)
    return;
  const int h = config_.compute_ops_->kv_cache_q_register(
    static_cast<unsigned int>(config_.kv_cache_quant_kind), cache_rows,
    config_.num_heads_KV, config_.head_dim);
  if (h < 0)
    return;
  q_cache_handles.assign(1, h);
  q_cache_synced.assign(1, 0);
  q2_scales_set.assign(1, 0);
}

void HtpAttentionState::release_quantized_cache() {
  if (config_.compute_ops_) {
    for (int h : q_cache_handles) {
      if (h >= 0) {
        config_.compute_ops_->kv_cache_q_release(h);
      }
    }
  }
  q_cache_handles.clear();
  q_cache_synced.clear();
  q2_scales_set.clear();
}

void HtpAttentionState::calibrate_q2_scales(
  const uint16_t *k_rows, const uint16_t *v_rows, unsigned int n_rows,
  unsigned int kv_stride, const float *q, unsigned int q_stride,
  unsigned int n_q) {
  q2_kmax.assign(q2_k_per_dim() ? config_.num_heads_KV * config_.head_dim
                                : config_.num_heads_KV,
                 0.0f);
  q2_vmax.assign(static_cast<size_t>(config_.num_heads_KV) * config_.head_dim,
                 0.0f);
  q2_grow(k_rows, v_rows, n_rows, kv_stride);
  q2_derive_q(q, q_stride, n_q);
  q2_calibrated = true;
}

bool HtpAttentionState::q2_k_per_dim() {
  // the KV adoption seeds decode K from the masters: per dim, or nll +9.4 %
  return htpRuntimeConfig().htp.kv_adopt;
}

bool HtpAttentionState::q2_grow(const uint16_t *k_rows, const uint16_t *v_rows,
                                unsigned int n_rows, unsigned int kv_stride) {
  const bool kd = q2_kmax.size() != config_.num_heads_KV;
  std::vector<float> kmax(q2_kmax.size(), 0.0f),
    vmax(static_cast<size_t>(config_.num_heads_KV) * config_.head_dim, 0.0f);
  for (unsigned int r = 0; r < n_rows; ++r) {
    const uint16_t *kr = k_rows + static_cast<size_t>(r) * kv_stride;
    const uint16_t *vr = v_rows + static_cast<size_t>(r) * kv_stride;
    if (kd)
      attention_quant::abs_max_f16_lanes(
        kr, config_.num_heads_KV * config_.head_dim, kmax.data());
    else
      for (unsigned int n = 0; n < config_.num_heads_KV; ++n)
        kmax[n] = attention_quant::abs_max_f16(kr + n * config_.head_dim,
                                               config_.head_dim, kmax[n]);
    attention_quant::abs_max_f16_lanes(
      vr, config_.num_heads_KV * config_.head_dim, vmax.data());
  }
  bool grew = false;
  for (size_t i = 0; i < kmax.size(); ++i)
    if (kmax[i] > q2_kmax[i]) {
      q2_kmax[i] = kmax[i];
      grew = true;
    }
  for (size_t i = 0; i < vmax.size(); ++i)
    if (vmax[i] > q2_vmax[i]) {
      q2_vmax[i] = vmax[i];
      grew = true;
    }
  if (grew)
    q2_derive_kv();
  return grew;
}

void HtpAttentionState::q2_derive_kv() {
  const float margin = htpRuntimeConfig().htp.q2_margin;
  q2_scale_k.resize(q2_kmax.size());
  q2_scale_v.resize(q2_vmax.size());
  for (size_t i = 0; i < q2_kmax.size(); ++i)
    q2_scale_k[i] = q2_kmax[i] > 0.0f ? q2_kmax[i] * margin / 127.0f : 1.0f;
  for (size_t i = 0; i < q2_vmax.size(); ++i)
    q2_scale_v[i] = q2_vmax[i] > 0.0f ? q2_vmax[i] * margin / 127.0f : 1.0f;
  // the context is a convex combination of V rows: its range is V's
  const unsigned int G = config_.num_heads_Q / config_.num_heads_KV;
  q2_out_enc.assign(2 * static_cast<size_t>(config_.num_heads_Q), 0.0f);
  for (unsigned int h = 0; h < config_.num_heads_Q; ++h) {
    float vmax = 0.0f;
    for (unsigned int d = 0; d < config_.head_dim; ++d)
      vmax = std::max(vmax, q2_vmax[(h / G) * config_.head_dim + d]);
    q2_out_enc[2 * h] = std::max(2.0f * vmax * margin, 1e-6f) / 65535.0f;
    q2_out_enc[2 * h + 1] = 32768.0f;
  }
  q2_keep();
}

void HtpAttentionState::q2_derive_q(const float *q, unsigned int q_stride,
                                    unsigned int n_q) {
  // Q's per-head range uses order-independent NEON min/max reductions.
  std::vector<float> q_lo(config_.num_heads_Q, 1e30f),
    q_hi(config_.num_heads_Q, -1e30f);
  // The range so far (a loaded or accelerator calibration, earlier steps):
  // like K / V the encoding only widens. A step with a narrow Q would
  // otherwise shrink the softmax's alpha and move its fraction bits off
  // the 9 every head of the checkpoint runs at.
  if (q2_q_enc.size() == 2 * static_cast<size_t>(config_.num_heads_Q))
    for (unsigned int h = 0; h < config_.num_heads_Q; ++h)
      if (q2_q_enc[2 * h] > 0.0f) {
        q_lo[h] = -q2_q_enc[2 * h + 1] * q2_q_enc[2 * h];
        q_hi[h] = q_lo[h] + 65535.0f * q2_q_enc[2 * h];
      }
  if (q2_scale_k.size() != config_.num_heads_KV) {
    // K per dim: the range of the Q the kernel quantizes, q * fs_kd / fs_k
    // (hexkl_attn_q2 stage_qprep); min (q r) = r min (q) for r > 0, so the
    // per-dim extremes first, as the accelerator's calibration takes them
    std::vector<float> lo(q_stride, 1e30f), hi(q_stride, -1e30f);
    for (unsigned int r = 0; r < n_q; ++r) {
      const float *qr = q + static_cast<size_t>(r) * q_stride;
      for (unsigned int i = 0; i < q_stride; ++i) {
        lo[i] = std::min(lo[i], qr[i]);
        hi[i] = std::max(hi[i], qr[i]);
      }
    }
    const unsigned int G = config_.num_heads_Q / config_.num_heads_KV;
    const unsigned int hd = config_.head_dim;
    for (unsigned int h = 0; h < config_.num_heads_Q; ++h) {
      const float *sk = q2_scale_k.data() + (h / G) * hd;
      const float top = *std::max_element(sk, sk + hd);
      for (unsigned int d = 0; d < hd; ++d) {
        const float r = sk[d] / top;
        q_lo[h] = std::min(q_lo[h], lo[h * hd + d] * r);
        q_hi[h] = std::max(q_hi[h], hi[h * hd + d] * r);
      }
    }
  } else {
    for (unsigned int r = 0; r < n_q; ++r) {
      const float *qr = q + static_cast<size_t>(r) * q_stride;
      for (unsigned int h = 0; h < config_.num_heads_Q; ++h)
        attention_quant::min_max_f32(qr + h * config_.head_dim,
                                     config_.head_dim, q_lo[h], q_hi[h]);
    }
  }
  q2_q_enc.assign(2 * static_cast<size_t>(config_.num_heads_Q), 0.0f);
  for (unsigned int h = 0; h < config_.num_heads_Q; ++h) {
    const float range = std::max(q_hi[h] - q_lo[h], 1e-6f);
    q2_q_enc[2 * h] = range / 65535.0f;
    q2_q_enc[2 * h + 1] = std::round(-q_lo[h] / q2_q_enc[2 * h]);
  }
  q2_keep();
}

void HtpAttentionState::q2_keep() {
  if (q2_scale_k.empty() || q2_q_enc.empty() || q2_out_enc.empty())
    return; // not calibrated yet
  std::vector<float> &all = q2_store_[q2_name_];
  all.clear();
  for (const auto *v : {&q2_scale_k, &q2_scale_v, &q2_q_enc, &q2_out_enc})
    all.insert(all.end(), v->begin(), v->end());
  // Append dropped rows' |max| when present.
  for (const auto *v : {&kv_floor_k_, &kv_floor_v_})
    all.insert(all.end(), v->begin(), v->end());
}

void HtpAttentionState::foldDroppedAmax(const std::vector<float> &k_amax,
                                        const std::vector<float> &v_amax) {
  NNTR_THROW_IF(k_amax.size() != v_amax.size() ||
                  (!kv_floor_k_.empty() && kv_floor_k_.size() != k_amax.size()),
                std::invalid_argument)
    << "mha_core: a dropped-rows |max| of " << k_amax.size() << " / "
    << v_amax.size() << " columns against a floor of " << kv_floor_k_.size();
  if (kv_floor_k_.empty()) {
    kv_floor_k_.assign(k_amax.size(), 0.0f);
    kv_floor_v_.assign(v_amax.size(), 0.0f);
  }
  for (size_t i = 0; i < k_amax.size(); ++i) {
    kv_floor_k_[i] = std::max(kv_floor_k_[i], k_amax[i]);
    kv_floor_v_[i] = std::max(kv_floor_v_[i], v_amax[i]);
  }
  q2_keep();
}

void HtpAttentionState::q2_maxima_from_scales() {
  q2_kmax.resize(q2_scale_k.size());
  q2_vmax.resize(q2_scale_v.size());
  for (size_t i = 0; i < q2_scale_k.size(); ++i)
    q2_kmax[i] = q2_scale_k[i] * 127.0f;
  for (size_t i = 0; i < q2_scale_v.size(); ++i)
    q2_vmax[i] = q2_scale_v[i] * 127.0f;
}

bool HtpAttentionState::take_q2_calibration() {
  const auto it = q2_store_.find(q2_name_);
  const size_t nv =
                 static_cast<size_t>(config_.num_heads_KV) * config_.head_dim,
               nk = q2_k_per_dim() ? nv : config_.num_heads_KV,
               nq = 2 * config_.num_heads_Q;
  // A saved dropped-row floor adds 2 * nv values.
  const size_t n_cal = nk + nv + 2 * nq;
  if (!q2_from_file_ || it == q2_store_.end() ||
      (it->second.size() != n_cal && it->second.size() != n_cal + 2 * nv))
    return false;
  const std::vector<float> all = it->second; // q2_keep rewrites the entry
  const float *p = all.data();
  q2_scale_k.assign(p, p + nk);
  q2_scale_v.assign(p + nk, p + nk + nv);
  q2_q_enc.assign(p + nk + nv, p + nk + nv + nq);
  q2_out_enc.assign(p + nk + nv + nq, p + nk + nv + 2 * nq);
  q2_maxima_from_scales();
  if (all.size() == n_cal + 2 * nv)
    foldDroppedAmax(std::vector<float>(p + n_cal, p + n_cal + nv),
                    std::vector<float>(p + n_cal + nv, p + n_cal + 2 * nv));
  return q2_calibrated = true;
}

void HtpAttentionState::saveQ2Calibration(std::ostream &file) {
  auto put = [&](uint32_t v) {
    file.write(reinterpret_cast<const char *>(&v), sizeof(v));
  };
  put(static_cast<uint32_t>(q2_store_.size()));
  for (const auto &[name, v] : q2_store_) {
    put(static_cast<uint32_t>(name.size()));
    file.write(name.data(), name.size());
    put(static_cast<uint32_t>(v.size()));
    file.write(reinterpret_cast<const char *>(v.data()),
               v.size() * sizeof(float));
  }
}

void HtpAttentionState::loadQ2Calibration(std::istream &file) {
  auto get = [&]() {
    uint32_t v = 0;
    file.read(reinterpret_cast<char *>(&v), sizeof(v));
    // a name or a layer's floats: nothing real comes near 1 MiB
    if (!file || v > (1u << 20))
      throw std::runtime_error("loadQ2Calibration: malformed file");
    return v;
  };
  std::map<std::string, std::vector<float>> store;
  for (uint32_t n = get(); n > 0; --n) {
    std::string name(get(), '\0');
    file.read(name.data(), name.size());
    std::vector<float> v(get());
    file.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(float));
    if (!file)
      throw std::runtime_error("loadQ2Calibration: truncated file");
    store[name] = std::move(v);
  }
  q2_store_ = std::move(store);
  q2_from_file_ = true;
}

bool HtpAttentionState::try_quantized_attention(
  unsigned int batch, nntrainer::Tensor &query_step,
  nntrainer::Tensor &cache_key, nntrainer::Tensor &cache_value,
  const ml::train::TensorDim &cache_key_dim,
  nntrainer::Tensor &attention_output_step, unsigned int cache_from,
  unsigned int cache_to, const nntrainer::Tensor *sink) {
  if (config_.kv_cache_quant_kind < 0 || q_cache_failed ||
      !config_.compute_ops_ || !config_.compute_ops_->supports_kv_cache_q() ||
      !config_.is_causal) {
    return false;
  }
  // The fp16 cache is the authoritative history until decode takes ownership.
  // Keep only this layer's temporary quantized layout during prefill. Reusing
  // calibration and resetting the synced row count also supports chunked
  // prefill: the next call rebuilds the layout from that history.
  const bool release_prefill = nntrainer::weightPageCachePolicy() >= 0 &&
                               cache_to - cache_from > 1 &&
                               !htpRuntimeConfig().htp.kv_seed_q;
  auto release = [release_prefill](HtpAttentionState *layer) {
    if (release_prefill)
      layer->release_quantized_cache();
  };
  std::unique_ptr<HtpAttentionState, decltype(release)> cache_lifetime(this,
                                                                       release);
  using clk = std::chrono::steady_clock;
  const auto ts0 = clk::now();
  AccelF32Io io;
  if (!io.prepare(query_step, attention_output_step, sink)) {
    return false;
  }
  // Q for this thread's own reads (the calibration, the ARM conversion):
  // where the engine has it when the projection call kept its rows
  // (qkv_layer's out_to_host false). The engine's calls take io.q.
  const float *q_view =
    static_cast<const float *>(config_.compute_ops_->host_view(
      io.q, static_cast<size_t>(cache_to - cache_from) * config_.num_heads_Q *
              config_.head_dim * sizeof(float)));
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
  if (width != config_.num_heads_KV * config_.head_dim || cache_to > max_rows) {
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

  // htpKvStaging: the CPU cache holds this step's rows only and the decode
  // cache every earlier one, so the int8 cache spans the absolute
  // positions [0, cache_base + cache_to): its rows are the CPU rows shifted
  // by cache_base. Each call registers it afresh, fills the earlier rows
  // from the decode cache, appends its own and seeds the decode cache back.
  const unsigned int cache_base = config_.cache_base;
  const bool staging = htpKvStaging() && batch == 0;
  const unsigned int kv_shift = staging ? cache_base : 0u;
  // htpKvAdopt: with the decode caches registered before the prefill
  // (setHtpPrefillSeed), a layer's call that ends at the prompt's end seeds
  // its decode cache from its int8 cache, which goes with the seed, and
  // frees its CPU rows (KVCacheManager compaction then moves this layer's
  // base only, kvDetached). Under staging every call seeds. Not taken, the
  // first token seeds from the CPU rows as before.
  auto handoff = [&](unsigned int &synced_rows, int h, bool q2) {
    if (staging) {
      // the decode cache takes the rows [0, cache_base + cache_to) the
      // int8 cache now holds (a later call's seed takes over an earlier
      // one's)
      NNTR_THROW_IF(
        !q2 || h < 0 || htp_seed_ord_ < 0 || synced_rows != cache_to ||
          !htpDecodeKvSeedKvqLayer(static_cast<unsigned>(htp_seed_ord_),
                                   cache_base + cache_to, h, 0u,
                                   cache_base > 0),
        std::runtime_error)
        << "mha_core: " << q2_name_ << " could not seed its decode cache "
        << "(int8-only KV: the CPU keeps no rows)";
      q_cache_handles[batch] = -1; // released on the DSP with the seed
      kv_detached_ = true;
      // not htp_owns_cache_: this prompt's next chunk takes its earlier
      // rows back from the decode cache (the fill), so it is not refused;
      // only the decode hook's first row hands the cache over for good
      return;
    }
    const unsigned int end = htpPrefillEnd();
    if (!htpKvAdopt() || !q2 || batch != 0 || h < 0 || !q2_k_per_dim() ||
        q2_scales_set.empty() || !q2_scales_set[0] || synced_rows != cache_to ||
        cache_to - cache_from < 2 || end == 0 || cache_base + cache_to != end ||
        htp_seed_ord_ < 0 || htp_seed_base_ < cache_base ||
        end <= htp_seed_base_ ||
        !htpDecodeKvSeedKvqLayer(static_cast<unsigned>(htp_seed_ord_),
                                 end - htp_seed_base_, h,
                                 htp_seed_base_ - cache_base))
      return;
    q_cache_handles[batch] = -1; // released on the DSP with the seed
    kv_detached_ = true;
    htp_owns_cache_ = true;
    for (const uint16_t *p : {k_base, v_base})
      config_.compute_ops_->drop_shared(const_cast<uint16_t *>(p));
  };

  if (q_cache_handles.size() <= batch) {
    q_cache_handles.resize(batch + 1, -1);
    q_cache_synced.resize(batch + 1, 0);
    q2_scales_set.resize(batch + 1, 0);
  }
  int &handle = q_cache_handles[batch];
  unsigned int &synced = q_cache_synced[batch];
  std::vector<float> fill_kr, fill_vr; // the old scale over the new, or none
  // int8 with the row-blocked kernel when the backend has it: fixed scales,
  // no softcap, no sinks, config_.head_dim up to 512.
  const bool use_q2 =
    config_.kv_cache_quant_kind == 0 && config_.kv_cache_fixed_scales &&
    htpRuntimeConfig().htp.attn_q2 &&
    config_.compute_ops_->supports_kv_cache_q2() &&
    config_.attn_logit_softcapping <= 0.0f && io.sinks == nullptr;
  if (!use_q2 && config_.head_dim > 256) {
    return false;
  }
  if (handle < 0) {
    ts2 = clk::now();
    handle = config_.compute_ops_->kv_cache_q_register(
      static_cast<unsigned int>(config_.kv_cache_quant_kind),
      staging ? kv_shift + cache_to : max_rows, config_.num_heads_KV,
      config_.head_dim);
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
  unsigned int append_row0 = synced;
  unsigned int append_rows = cache_to - synced;
  size_t off = batch * cache_key_dim.getFeatureLen() +
               static_cast<size_t>(append_row0) * width;

  const unsigned int n_q = cache_to - cache_from;
  const unsigned int q_stride = config_.num_heads_Q * config_.head_dim;
  const unsigned int window =
    (config_.local_window_size == 0 ||
     config_.local_window_size >= kv_shift + cache_to)
      ? 0u
      : static_cast<unsigned int>(config_.local_window_size);
  if (use_q2) {
    // A loaded KV cache brought the calibration of the run that saved it
    // (loadQ2Calibration): it wins over calibrating on this step's rows.
    if (!q2_calibrated) {
      take_q2_calibration();
    }
    // [P2] The first rows a layer sees: the accelerator scans them for the
    // scales (exact maxima, the same derivation) inside the step itself,
    // so no pass over K / V / Q runs here. The checking paths and the
    // NEON path keep the layer's own calibration.
    const auto &runtime = htpRuntimeConfig().htp;
    const bool dsp_calib = runtime.attn_dsp_calib;
    if (!q2_calibrated && config_.attn_out_u16 && dsp_calib &&
        !runtime.attn_f32_check.has_value() && runtime.attn_f32) {
      q2_scale_k.assign(q2_k_per_dim() ? config_.num_heads_KV * config_.head_dim
                                       : config_.num_heads_KV,
                        0.0f);
      q2_scale_v.assign(
        static_cast<size_t>(config_.num_heads_KV) * config_.head_dim, 0.0f);
      q2_q_enc.assign(2 * static_cast<size_t>(config_.num_heads_Q), 0.0f);
      q2_out_enc.assign(2 * static_cast<size_t>(config_.num_heads_Q), 0.0f);
      const auto t0 = std::chrono::steady_clock::now();
      if (config_.compute_ops_->sdpa_q2_kvcache_f32_calib(
            handle, append_row0 + kv_shift, append_rows, width, k_base + off,
            v_base + off, io.q, q_stride, n_q, cache_from + kv_shift,
            cache_to + kv_shift, config_.num_heads_Q, config_.num_heads_KV,
            config_.head_dim, window, io.out, q2_scale_k.data(),
            q2_scale_v.data(), q2_q_enc.data(), q2_out_enc.data(), q_stride,
            q2_k_per_dim())) {
        q2_calibrated = true;
        q2_scales_set[batch] = 1;
        synced = cache_to;
        // the maxima later steps grow from (q2_grow), and the scales kept
        // for saveQ2Calibration as calibrate_q2_scales keeps its own
        q2_maxima_from_scales();
        q2_keep();
        if (runtime.attn_trace) {
          ml_logi("mha_core trace: q2 calibration on the accelerator rows=%u "
                  "us=%lld",
                  append_rows,
                  static_cast<long long>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count()));
        }
        if (!accel_logged_) {
          accel_logged_ = true;
          ml_logi("mha_core: attention over the int8 quantized KV cache on "
                  "the accelerator (row-blocked, fixed scales calibrated "
                  "there, u16 context handed to the o-proj)");
        }
        handoff(synced, handle, use_q2);
        return true;
      }
    }
    if (!q2_calibrated) {
      // The rows about to be appended are the first this layer sees.
      const auto t0 = std::chrono::steady_clock::now();
      calibrate_q2_scales(k_base + off, v_base + off, append_rows, width,
                          q_view, q_stride, n_q);
      if (runtime.attn_trace) {
        ml_logi("mha_core trace: q2 calibration rows=%u us=%lld", append_rows,
                static_cast<long long>(
                  std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - t0)
                    .count()));
      }
    } else {
      // A later step's rows (a prefill chunk after the first, a prompt
      // after a loaded cache) may exceed the maxima the scales came from
      // and would clip at 127 (seen: V 2x at a third chunk). The scales
      // widen to them and every row is re-quantized: the mirror takes rows
      // [0, cache_to) again under the new scales.
      std::vector<float> old_k, old_v; // staging: the scales before growing
      if (staging) {
        old_k = q2_scale_k;
        old_v = q2_scale_v;
      }
      if (append_rows != 0 &&
          q2_grow(k_base + off, v_base + off, append_rows, width)) {
        if (staging) {
          // the earlier rows come back at the grown scales
          fill_kr.resize(old_k.size());
          fill_vr.resize(old_v.size());
          for (size_t x = 0; x < old_k.size(); ++x)
            fill_kr[x] = old_k[x] / q2_scale_k[x];
          for (size_t x = 0; x < old_v.size(); ++x)
            fill_vr[x] = old_v[x] / q2_scale_v[x];
        }
        if (runtime.attn_trace)
          ml_logi("mha_core trace: q2 scales grew at rows=%u from=%u; "
                  "re-quantizing %u rows",
                  append_rows, append_row0, cache_to);
        q2_scales_set[batch] = 0;
        synced = 0;
        append_row0 = 0;
        append_rows = cache_to;
        off = batch * cache_key_dim.getFeatureLen();
      }
      // Q is not cached: its encoding follows this step's rows
      q2_derive_q(q_view, q_stride, n_q);
    }
    // Trace rows that exceed the calibrated int8 range.
    if (runtime.attn_trace && append_rows != 0) {
      float worst_k = 0.0f, worst_v = 0.0f;
      size_t clip_k = 0, clip_v = 0;
      const unsigned int k_dims = q2_scale_k.size() / config_.num_heads_KV;
      for (unsigned int r = 0; r < append_rows; ++r) {
        const uint16_t *kr = k_base + off + static_cast<size_t>(r) * width;
        const uint16_t *vr = v_base + off + static_cast<size_t>(r) * width;
        for (unsigned int n = 0; n < config_.num_heads_KV; ++n)
          for (unsigned int d = 0; d < config_.head_dim; ++d) {
            const float kq = std::fabs(nntrainer::compute_fp16_to_fp32(
                               kr[n * config_.head_dim + d])) /
                             q2_scale_k[n * k_dims + d % k_dims];
            const float vq = std::fabs(nntrainer::compute_fp16_to_fp32(
                               vr[n * config_.head_dim + d])) /
                             q2_scale_v[n * config_.head_dim + d];
            worst_k = std::max(worst_k, kq);
            worst_v = std::max(worst_v, vq);
            clip_k += kq > 127.5f;
            clip_v += vq > 127.5f;
          }
      }
      // and this step's Q against its fixed u16 encoding (range, zero point
      // from the calibration rows)
      size_t clip_q = 0;
      float worst_q = 0.0f;
      // K per dim: the encoding is the folded Q's, q * fs_kd / fs_k
      std::vector<float> fold;
      const unsigned int hd = config_.head_dim;
      if (q2_scale_k.size() != config_.num_heads_KV) {
        fold.resize(q2_scale_k.size());
        for (unsigned int n = 0; n < config_.num_heads_KV; ++n) {
          const float *sk = q2_scale_k.data() + n * hd;
          const float top = *std::max_element(sk, sk + hd);
          for (unsigned int d = 0; d < hd; ++d)
            fold[n * hd + d] = sk[d] / top;
        }
      }
      const unsigned int G = config_.num_heads_Q / config_.num_heads_KV;
      for (unsigned int r = 0; r < n_q; ++r)
        for (unsigned int h = 0; h < config_.num_heads_Q; ++h) {
          const float step = q2_q_enc[2 * h], zp = q2_q_enc[2 * h + 1];
          const float *qr =
            q_view + static_cast<size_t>(r) * q_stride + h * config_.head_dim;
          for (unsigned int d = 0; d < config_.head_dim; ++d) {
            const float qd =
              fold.empty() ? qr[d] : qr[d] * fold[(h / G) * hd + d];
            const float code = qd / step + zp; // 0 .. 65535 inside
            const float over = code < 0.0f       ? -code
                               : code > 65535.0f ? code - 65535.0f
                                                 : 0.0f;
            clip_q += over > 0.5f;
            worst_q = std::max(worst_q, over / 65535.0f);
          }
        }
      ml_logi("mha_core trace: q2 clip %s rows=%u from=%u k_max/127=%.2f "
              "v_max/127=%.2f clipped k=%zu v=%zu of %zu | q over range "
              "%.2f clipped q=%zu of %zu",
              q2_name_.c_str(), append_rows, append_row0, worst_k / 127.0f,
              worst_v / 127.0f, clip_k, clip_v,
              static_cast<size_t>(append_rows) * width, worst_q, clip_q,
              static_cast<size_t>(n_q) * q_stride);
    }
    const auto ts_sc = clk::now();
    if (!q2_scales_set[batch]) {
      if (!config_.compute_ops_->kv_cache_q_set_fixed_scales(
            handle, config_.num_heads_KV, config_.head_dim, q2_scale_k.data(),
            q2_scale_v.data(), q2_scale_k.size() != config_.num_heads_KV)) {
        return fail("fixed scales");
      }
      q2_scales_set[batch] = 1;
    }
    if (staging && kv_shift > 0) {
      NNTR_THROW_IF(htp_seed_ord_ < 0 || fill_kr.size() != fill_vr.size() ||
                      !htpDecodeKvFillKvq(
                        static_cast<unsigned>(htp_seed_ord_), handle, kv_shift,
                        fill_kr.empty() ? nullptr : fill_kr.data(),
                        fill_vr.empty() ? nullptr : fill_vr.data(),
                        static_cast<unsigned>(fill_kr.size())),
                    std::runtime_error)
        << "mha_core: " << q2_name_ << " could not take its rows [0, "
        << kv_shift << ") back from the decode cache";
    }
    // Q as the model's u16 tensor and the output back from u16: in a
    // quantized graph both are the neighbouring layers' formats already;
    // here the f32 layer converts at its edges.
    const size_t q_elems = static_cast<size_t>(n_q) * q_stride;
    ts4 = clk::now();
    ts5 = clk::now();
    // No conversions on this thread when the o-proj is on the accelerator
    // too (config_.attn_out_u16): Q goes as the qkv call left it, the u16
    // context stays there for the o-proj. Else NEON on this thread below.
    // Check mode 1 compares with NEON; mode 2 reruns the accelerator.
    const bool f32_check = runtime.attn_f32_check.has_value();
    const int f32_check_mode = runtime.attn_f32_check.value_or(0);
    const bool f32_off = !runtime.attn_f32;
    std::vector<float> q_inv(config_.num_heads_Q);
    for (unsigned int h = 0; h < config_.num_heads_Q; ++h)
      q_inv[h] = 1.0f / q2_q_enc[2 * h];
    std::vector<float> chk;
    std::vector<uint16_t> chk16;
    if (f32_check)
      chk16.resize(q_elems);
    const bool f32_ran =
      config_.attn_out_u16 && !f32_off &&
      config_.compute_ops_->sdpa_q2_kvcache_f32(
        handle, append_row0 + kv_shift, append_rows, width, k_base + off,
        v_base + off, io.q, q_inv.data(), q2_q_enc.data(), q_stride, n_q,
        cache_from + kv_shift, cache_to + kv_shift, config_.num_heads_Q,
        config_.num_heads_KV, config_.head_dim, window, io.out,
        f32_check ? chk16.data() : nullptr, q2_out_enc.data(), q_stride);
    bool again = false;
    if (f32_ran && f32_check && runtime.attn_trace) {
      // the u16 context's saturation against the fixed output encoding
      size_t sat = 0;
      for (size_t i = 0; i < q_elems; ++i)
        sat += chk16[i] == 0u || chk16[i] == 65535u;
      ml_logi("mha_core trace: u16 ctx saturated %s rows=%u from=%u %zu of %zu",
              q2_name_.c_str(), n_q, cache_from, sat, q_elems);
    }
    if (f32_ran && f32_check) {
      // the first run's context in chk; mode 2 reruns into chk16 and the
      // compare below is run vs run instead of accelerator vs NEON
      chk.resize(q_elems);
      for (unsigned int r = 0; r < n_q; ++r)
        for (unsigned int h = 0; h < config_.num_heads_Q; ++h)
          attention_quant::dequant_u16_f32(
            chk16.data() + static_cast<size_t>(r) * q_stride +
              h * config_.head_dim,
            config_.head_dim, q2_out_enc[2 * h], q2_out_enc[2 * h + 1],
            chk.data() + static_cast<size_t>(r) * q_stride +
              h * config_.head_dim);
      again = f32_check_mode == 2 &&
              config_.compute_ops_->sdpa_q2_kvcache_f32(
                handle, append_row0 + kv_shift, append_rows, width,
                k_base + off, v_base + off, io.q, q_inv.data(), q2_q_enc.data(),
                q_stride, n_q, cache_from + kv_shift, cache_to + kv_shift,
                config_.num_heads_Q, config_.num_heads_KV, config_.head_dim,
                window, io.out, chk16.data(), q2_out_enc.data(), q_stride);
      if (again)
        for (unsigned int r = 0; r < n_q; ++r)
          for (unsigned int h = 0; h < config_.num_heads_Q; ++h)
            attention_quant::dequant_u16_f32(
              chk16.data() + static_cast<size_t>(r) * q_stride +
                h * config_.head_dim,
              config_.head_dim, q2_out_enc[2 * h], q2_out_enc[2 * h + 1],
              io.out + static_cast<size_t>(r) * q_stride +
                h * config_.head_dim);
    }
    if (f32_ran && !f32_check) {
      // io.out is not written: the o-proj reads the context from the
      // accelerator (the builder set config_.attn_out_u16 only for that
      // engine).
      synced = cache_to;
      if (!accel_logged_) {
        accel_logged_ = true;
        ml_logi("mha_core: attention over the int8 quantized KV cache on the "
                "accelerator (row-blocked, fixed scales, u16 context handed "
                "to the o-proj)");
      }
      // a later chunk's call ends here, as the first call ends at the
      // calibration: it seeds the decode cache the same way
      handoff(synced, handle, use_q2);
      return true;
    }
    // Only the host conversion path needs these buffers.
    ts4 = clk::now();
    if (!again && q2_q_u16.size() < q_elems) {
      q2_q_u16.resize(q_elems);
      q2_out_u16.resize(q_elems);
    }
    ts5 = clk::now();
    // Keep both NEON conversions on this thread to avoid scheduling overhead.
    const bool conv_trace = runtime.attn_trace;
    const auto tq0 = std::chrono::steady_clock::now();
    // K per dim: Q's inverse step times fs_kd / fs_k per dim, formed as the
    // accelerator's f32 path forms it (hexkl_attn_q2 stage_qprep)
    std::vector<float> q_inv_d;
    if (q2_scale_k.size() != config_.num_heads_KV) {
      const unsigned int G = config_.num_heads_Q / config_.num_heads_KV;
      const unsigned int hd = config_.head_dim;
      q_inv_d.resize(q_stride);
      for (unsigned int h = 0; h < config_.num_heads_Q; ++h) {
        const float *sk = q2_scale_k.data() + (h / G) * hd;
        const float top = *std::max_element(sk, sk + hd);
        for (unsigned int d = 0; d < hd; ++d)
          q_inv_d[h * hd + d] = q_inv[h] * (sk[d] / top);
      }
    }
    for (unsigned int r = 0; r < n_q && !again; ++r) {
      const float *qr = q_view + static_cast<size_t>(r) * q_stride;
      uint16_t *dst = q2_q_u16.data() + static_cast<size_t>(r) * q_stride;
      for (unsigned int h = 0; h < config_.num_heads_Q; ++h)
        if (q_inv_d.empty())
          attention_quant::quant_u16_f32(
            qr + h * config_.head_dim, config_.head_dim, 1.0f / q2_q_enc[2 * h],
            q2_q_enc[2 * h + 1], dst + h * config_.head_dim);
        else
          attention_quant::quant_u16_f32(
            qr + h * config_.head_dim, config_.head_dim,
            q_inv_d.data() + h * config_.head_dim, q2_q_enc[2 * h + 1],
            dst + h * config_.head_dim);
    }
    const auto tq1 = std::chrono::steady_clock::now();
    if (!again &&
        !config_.compute_ops_->sdpa_q2_kvcache(
          handle, append_row0 + kv_shift, append_rows, width, k_base + off,
          v_base + off, q2_q_u16.data(), q2_q_enc.data(), q_stride, n_q,
          cache_from + kv_shift, cache_to + kv_shift, config_.num_heads_Q,
          config_.num_heads_KV, config_.head_dim, window, q2_out_u16.data(),
          q2_out_enc.data(), q_stride)) {
      return fail("attention");
    }
    const auto tq2 = std::chrono::steady_clock::now();
    for (unsigned int r = 0; r < n_q && !again; ++r) {
      const uint16_t *src =
        q2_out_u16.data() + static_cast<size_t>(r) * q_stride;
      float *dst = io.out + static_cast<size_t>(r) * q_stride;
      for (unsigned int h = 0; h < config_.num_heads_Q; ++h)
        attention_quant::dequant_u16_f32(
          src + h * config_.head_dim, config_.head_dim, q2_out_enc[2 * h],
          q2_out_enc[2 * h + 1], dst + h * config_.head_dim);
    }
    if (f32_check && !chk.empty()) {
      size_t bad = 0, steps1 = 0;
      double worst = 0.0;
      for (size_t i = 0; i < q_elems; ++i) {
        if (std::memcmp(&chk[i], &io.out[i], sizeof(float)) == 0)
          continue;
        ++bad;
        const unsigned int h = (i % q_stride) / config_.head_dim;
        const double d =
          std::fabs((double)chk[i] - io.out[i]) / q2_out_enc[2 * h];
        worst = std::max(worst, d);
        steps1 += d > 0.5;
      }
      ml_logi("mha_core f32 edge check: rows=%u from=%u mismatches=%zu of %zu "
              "(>= one u16 step: %zu, worst %.4g steps)",
              n_q, cache_from, bad, q_elems, steps1, worst);
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
  } else if (!config_.compute_ops_->sdpa_q_kvcache(
               handle, append_row0 + kv_shift, append_rows, width, k_base + off,
               v_base + off, q_view, q_stride, n_q, cache_from + kv_shift,
               cache_to + kv_shift, config_.num_heads_Q, config_.num_heads_KV,
               config_.head_dim, window, config_.attn_logit_softcapping,
               io.sinks, io.out, q_stride)) {
    return fail("attention");
  }
  synced = cache_to;
  io.commit(attention_output_step);
  if (!accel_logged_) {
    accel_logged_ = true;
    ml_logi("mha_core: attention over the %s quantized KV cache on the "
            "accelerator%s",
            config_.kv_cache_quant_kind == 0 ? "int8" : "int4",
            use_q2 ? " (row-blocked, fixed scales)" : "");
  }
  handoff(synced, handle, use_q2);
  return true;
}
} // namespace nntrainer

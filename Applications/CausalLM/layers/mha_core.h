// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   mha_core.h
 * @date   02 September 2024
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is custom_mha_core layer supports
 *         the work of multi_head_attention.
 * @note   Unlike custom_multi_head_attention_layer,
 *         which works all of the attention operations
 *         in a layer, this layer is attached after Q / K / V
 *         fully connected layer to post-process them
 *         including KV-Cache.
 *         For inference, forwarding processes the current input rows
 *         at the KV-cache write position, and incremental_forwarding
 *         takes the step rows [`from`, `to`) (one or more rows).
 *         Training is not supported (forwarding is a no-op when
 *         training is set in internal-cache mode).
 */

#ifndef __MHA_CORE_H__
#define __MHA_CORE_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <complex>

#include <acti_func.h>
#include <common_properties.h>
#include <cpu_backend.h>
#include <layer_impl.h>
#include <limits.h>
#include <util_simd.h>

#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace causallm {

namespace props {

/**
 * @brief NumHeads property, NumHeads is number of head in multi head attention
 * of Q
 */
class NumHeads_KV : public nntrainer::PositiveIntegerProperty {
public:
  /**
   * @brief Construct a new NumHeads object with default value 1
   */
  NumHeads_KV(unsigned int value = 1) { set(value); };
  static constexpr const char *key =
    "num_heads_KV";                          /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief SlidingWindow
 */
class SlidingWindow : public nntrainer::Property<unsigned int> {
public:
  SlidingWindow(unsigned int value = UINT_MAX) { set(value); };
  static constexpr const char *key =
    "sliding_window";                        /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief InitSeqLen -- the model's activation-plane height in query rows.
 * @details The prefill feeds at most this many query rows per forward pass,
 * so it is the ceiling the prefill chunk is clamped to. The layer needs the
 * clamped chunk (not the raw NNTR_PREFILL_CHUNK request) to size its KV window
 * ring exactly as the model side does -- the two must agree to the row or the
 * modulo indexing writes out of bounds. 0 means "not told", and the layer then
 * falls back to its query input height, which is the same plane.
 */
class InitSeqLen : public nntrainer::Property<unsigned int> {
public:
  InitSeqLen(unsigned int value = 0) { set(value); };
  static constexpr const char *key =
    "init_seq_len";                          /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief MaxNewTokens
 */
class MaxNewTokens : public nntrainer::Property<unsigned int> {
public:
  MaxNewTokens(unsigned int value = 1) { set(value); };
  static constexpr const char *key =
    "max_new_tokens";                        /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief MaxNewTokens
 */
class MaxPositionEmbeddings : public nntrainer::Property<unsigned int> {
public:
  MaxPositionEmbeddings(unsigned int value = 40960) { set(value); };
  static constexpr const char *key =
    "max_position_embeddings";               /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief RopeTheta
 */
class RopeTheta : public nntrainer::Property<unsigned int> {
public:
  RopeTheta(unsigned int value = 500000) { set(value); };
  static constexpr const char *key = "rope_theta"; /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag;       /**< property type */
};

/**
 * @brief UseRope property (Gemma4/Gemma3n). Accepted but our mha_core derives
 *        rope-enable from theta>0, so it is informational here.
 */
class UseRope : public nntrainer::Property<bool> {
public:
  UseRope(bool value = true) { set(value); };
  static constexpr const char *key = "use_rope"; /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag;     /**< property type */
};

/**
 * @brief UseSink property
 */
class UseSink : public nntrainer::Property<bool> {
public:
  UseSink(bool value = false) { set(value); };
  static constexpr const char *key = "use_sink"; /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag;     /**< property type */
};

/**
 * @brief AttnLogitSoftcapping
 */
class AttnLogitSoftcapping : public nntrainer::Property<float> {
public:
  AttnLogitSoftcapping(float value = 0.0f) { set(value); };
  static constexpr const char *key =
    "attn_logit_softcapping";                 /**< unique key to access */
  using prop_tag = nntrainer::float_prop_tag; /**< property type */
};

/**
 * @brief IsCausal property
 */
class IsCausal : public nntrainer::Property<bool> {
public:
  IsCausal(bool value = true) { set(value); };
  static constexpr const char *key = "is_causal"; /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag;      /**< property type */
};

/**
 * @brief UseGemmAttention property
 * @note  Optional GEMM-based attention path for the non-causal (encoder) case.
 *        QK and AV are computed with (s)gemm per head instead of the per-row
 *        dot kernels, improving cache reuse for large sequence lengths.
 */
class UseGemmAttention : public nntrainer::Property<bool> {
public:
  UseGemmAttention(bool value = false) { set(value); };
  static constexpr const char *key = "use_gemm_attention";
  using prop_tag = nntrainer::bool_prop_tag;
};

/**
 * @brief GpuDecodeAttn property (per-model decode-GPU gate).
 * @note  When true, the M=1 (decode) attention runs the validated GPU flash /
 *        OHWI image-attention path instead of bouncing K/V to the host NEON
 *        path. Mirrors the NNTR_MHA_GPU_DECODE env flag's flash-attention half
 *        ((B)), but per-LAYER so it can be DEFAULT-ON only for the models where
 *        decode flash attention is token-identical (gemma4, gemma2). The env
 *        flag, when set, forces it on globally (testing override). Default
 *        false keeps the host decode attention.
 */
class GpuDecodeAttn : public nntrainer::Property<bool> {
public:
  GpuDecodeAttn(bool value = false) { set(value); };
  static constexpr const char *key = "gpu_decode_attn";
  using prop_tag = nntrainer::bool_prop_tag;
};

/**
 * @brief KvWindowRing property (the model's default for the KV ring).
 * @note  The value causallm::kvRingEnabled() falls back to when
 *        NNTR_KV_WINDOW_RING is unset. It must equal the owning model's
 *        Transformer::kvRingByDefault(): the model sizes the cache plane with
 *        that value and this layer indexes the plane with this one. Default
 *        false keeps the ring an explicit opt-in.
 */
class KvWindowRing : public nntrainer::Property<bool> {
public:
  KvWindowRing(bool value = false) { set(value); };
  static constexpr const char *key = "kv_window_ring";
  using prop_tag = nntrainer::bool_prop_tag;
};

/**
 * @brief GpuDecodeRope property (per-model decode-GPU gate).
 * @note  When true, the M=1 (decode) RoPE runs on the GPU (rope_inplace_f16_cl)
 *        so Q/K stay SVM-resident and lower_q/lower_kv drains are skipped.
 *        Mirrors the NNTR_MHA_GPU_DECODE env flag's GPU-RoPE half ((A)), but
 *        per-LAYER so it is DEFAULT-ON only where decode GPU-RoPE is
 *        token-identical (gemma4). gemma2 diverges on GPU-RoPE-decode, so it
 *        keeps this false (flash attention + host RoPE). Still gated off by the
 *        NNTR_NO_GPU_ROPE kill-switch. The env flag, when set, forces it on
 *        globally (testing override). Default false keeps the host decode RoPE.
 */
class GpuDecodeRope : public nntrainer::Property<bool> {
public:
  GpuDecodeRope(bool value = false) { set(value); };
  static constexpr const char *key = "gpu_decode_rope";
  using prop_tag = nntrainer::bool_prop_tag;
};

/**
 * @brief GpuOhwiRope property (per-model Adreno OHWI decode-RoPE gate).
 * @note  When true, the M=1 (decode) RoPE on the OHWI image-attention path
 *        (NNTR_KV_IMG_ATTN) runs on the GPU (rope_inplace_f16_cl) so Q/K stay
 *        SVM-resident and the per-layer lower_q/lower_kv drain (16-35ms/token)
 *        is skipped. DEFAULT-ON where it is token-identical (gemma4 +32%,
 *        gemma2 +8%); qwen3 diverges (head_dim=128 / q-k-norm), so it stays
 *        false. The NNTR_OHWI_GPU_ROPE env forces it on globally (override);
 *        NNTR_NO_GPU_ROPE kills it. Default false keeps the host decode RoPE.
 */
class GpuOhwiRope : public nntrainer::Property<bool> {
public:
  GpuOhwiRope(bool value = false) { set(value); };
  static constexpr const char *key = "gpu_ohwi_rope";
  using prop_tag = nntrainer::bool_prop_tag;
};

/**
 * @brief RopeScalingType
 * - default
 * - yarn
 */
class RopeScalingType : public nntrainer::Property<std::string> {
public:
  RopeScalingType(std::string value = "default") { set(value); };
  static constexpr const char *key =
    "rope_scaling_type";                    /**< unique key to access */
  using prop_tag = nntrainer::str_prop_tag; /**< property type */
};
/**
 * @brief RopeScalingFactor
 */
class RopeScalingFactor : public nntrainer::Property<float> {
public:
  RopeScalingFactor(float value = 1.0) { set(value); };
  static constexpr const char *key =
    "rope_scaling_factor";                    /**< unique key to access */
  using prop_tag = nntrainer::float_prop_tag; /**< property type */
};

/**
 * @brief RopePartialRotaryFactor (Gemma4/Gemma3n). Outside yarn, a factor
 *        != 1 routes the RoPE table through _compute_proportional_parameters,
 *        which gives rotary frequencies to only factor * head_dim / 2 of the
 *        head_dim / 2 pairs; the rest pass through unrotated.
 */
class RopePartialRotaryFactor : public nntrainer::Property<float> {
public:
  RopePartialRotaryFactor(float value = 1.0f) { set(value); };
  static constexpr const char *key =
    "rope_partial_rotary_factor";             /**< unique key to access */
  using prop_tag = nntrainer::float_prop_tag; /**< property type */
};

/**
 * @brief RopeScalingMaxPositionEmbeddings
 */
class RopeScalingMaxPositionEmbeddings
  : public nntrainer::Property<unsigned int> {
public:
  RopeScalingMaxPositionEmbeddings(unsigned int value = 4096) { set(value); };
  static constexpr const char *key =
    "rope_scaling_max_position_embeddings";  /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

}; // namespace props

/**
 * @brief Declare that this pack wants the prefill K rotation drained, i.e.
 *        whether nntr_config.json sets `"prefill_kv_drain": true`.
 *
 * The attention core uses it for exactly one decision: whether the prefill K
 * rotation drains before `k_scatter_ohwi_cl` reads the SVM cache slice it
 * wrote (see mha_core.cpp, `_kv_nodrain_env`). Default false, i.e. the
 * per-write no-drain, which is what every pack that has not been observed to
 * need the drain keeps.
 *
 * Read the gate honestly. This is a DECLARATION, recording where the defect
 * has been OBSERVED to decide a token -- it is not a decision the tree makes
 * about a pack, and it is emphatically not a model name. The RISK condition is
 * a consumer reading an SVM write issued in an earlier submission, which every
 * model on this chain does; deciding THAT in code, where the chain is built
 * rather than from a config flag, is the per-dispatch hazard check
 * (Kernel::noteUndrainedSvmPlane, opt-in via NNTR_SVM_HAZARD_DRAIN=1), which
 * is meant to replace this declaration with the thing it stands for. Until it
 * does, a pack that does not set it is unobserved, not proven safe.
 */
void setPrefillKvDrain(bool on);

/** @copydoc setPrefillKvDrain(bool) */
bool prefillKvDrain();

/**
 * @brief [prefill-long] Announce the total key span the next prefill reaches.
 * @details Set by the prefill driver before the chunk loop and cleared after,
 * so the tight-stride V image is laid out once for the whole prompt instead of
 * being re-laid (and every scattered row re-written) at each chunk boundary.
 * 0 means "unknown", which restores the per-chunk growth exactly.
 */
void setPrefillSpanHint(unsigned int span);

/** @copydoc setPrefillSpanHint(unsigned int) */
unsigned int prefillSpanHint();

/**
 * @class MHA Core Layer
 * @brief Part of Multi-Head-Attention Layer.
 *        It should be attached after Q / K / V fc layers and before O fc layer.
 *        custom_mha_core_layer computes attention, while updating KV-cache for
 *        inference mode.
 *
 *    [ Q ]    [ K ]    [ V ]
 *      |        |        |
 *     [      mha_core      ]
 *               |
 *             [ O ]
 *
 */
WIN_EXPORT class MHACoreLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Constructor of MhaCore Layer
   */
  WIN_EXPORT MHACoreLayer();

  /**
   * @brief Destructor of MhaCore Layer
   */
  WIN_EXPORT ~MHACoreLayer();

  /**
   *  @brief  Move constructor of MHACoreLayer.
   *  @param[in] MHACoreLayer &&
   */
  WIN_EXPORT
  MHACoreLayer(MHACoreLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs MHACoreLayer to be moved.
   */
  WIN_EXPORT MHACoreLayer &operator=(MHACoreLayer &&rhs) = default;

  /**
   * @brief Finalize funciton of MhaCore Layer
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @brief forwarding function of MhaCore Layer
   *        This is the inference entry point: it runs attention over the
   *        current input rows and updates the KV cache.
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  void one_batch_incremental_forwarding(
    const unsigned int batch, const unsigned int _from, const unsigned int from,
    const unsigned int to, nntrainer::Tensor &query_step,
    nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
    nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
    nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
    ml::train::TensorDim &cache_key_step_dim,
    ml::train::TensorDim &cache_value_dim,
    ml::train::TensorDim &cache_value_step_dim);

  void one_batch_incremental_forwarding(
    const unsigned int batch, const unsigned int _from, const unsigned int from,
    const unsigned int to, nntrainer::Tensor &query_step,
    nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
    nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
    nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
    ml::train::TensorDim &cache_key_step_dim,
    ml::train::TensorDim &cache_value_dim,
    ml::train::TensorDim &cache_value_step_dim, nntrainer::Tensor &sink_step);

  /**
   * @brief Run one batch of an FP32-activation layer on the device attention.
   * @details The device arms (RoPE, image / flash attention) take FP16
   * operands in shared virtual memory. This narrows Q/K/V into FP16 planes on
   * the device, runs one_batch_incremental_forwarding on them as for an
   * FP16-activation model, and widens O back into @a attention_output_step on
   * the device, so no operand is read on the host. The arguments are those of
   * one_batch_incremental_forwarding, with FP32 step operands.
   * @return false, with nothing enqueued that a host arm would read, when this
   *         call cannot take it: no OpenCL device attention in this run, an
   *         int8 KV cache or an attention sink, or operands that are not
   *         plain shared virtual memory. The caller then runs its host path.
   */
  bool attention_fp32_on_device(
    const unsigned int batch, const unsigned int _from, const unsigned int from,
    const unsigned int to, nntrainer::Tensor &query_step,
    nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
    nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
    nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
    ml::train::TensorDim &cache_key_step_dim,
    ml::train::TensorDim &cache_value_dim,
    ml::train::TensorDim &cache_value_step_dim);

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   * @note Returns true, but backwarding is not implemented yet:
   * calcDerivative() and calcGradient() are no-ops.
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return MHACoreLayer::type;
  };

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  WIN_EXPORT void setBatch(nntrainer::RunLayerContext &context,
                           unsigned int batch) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  /**
   * @brief Set the cache index for external cache mode.
   *        Must be called before forwarding() when use_external_cache is true.
   * @param[in] idx current write position in the KV cache
   */
  WIN_EXPORT void setCacheIndex(unsigned int idx) { cache_index = idx; }

  /**
   * @brief Get the current cache index
   */
  WIN_EXPORT unsigned int getCacheIndex() const { return cache_index; }

  inline static const std::string type = "mha_core";

private:
  std::tuple<
    nntrainer::props::NumHeads, props::NumHeads_KV,
    nntrainer::props::ProjectedKeyDim, nntrainer::props::ProjectedValueDim,
    nntrainer::props::OutputShape, nntrainer::props::DropOutRate,
    nntrainer::props::ReturnAttentionWeight,
    nntrainer::props::AverageAttentionWeight, nntrainer::props::MaxTimestep,
    props::SlidingWindow, props::InitSeqLen, props::MaxNewTokens,
    props::RopeTheta, props::UseRope, props::MaxPositionEmbeddings,
    props::UseSink, props::RopeScalingType, props::RopeScalingFactor,
    props::RopePartialRotaryFactor, props::RopeScalingMaxPositionEmbeddings,
    props::AttnLogitSoftcapping, props::IsCausal, props::UseGemmAttention,
    props::GpuDecodeAttn, props::GpuDecodeRope, props::GpuOhwiRope,
    props::KvWindowRing>
    mha_core_props; /**< mha_core layer properties */

  /** softmax activation operation */
  nntrainer::ActiFunc sm;

  float epsilon;            /** to avoid overflow */
  unsigned int cache_index; /** idx of kv cache */

  /**
   * @brief Honor the LayerImpl `skip_prefill` property. Gemma4 KV-shared
   *        layers (the last num_kv_shared_layers) reuse an earlier layer's
   *        K/V projection as their attention K/V *inputs*, and since
   *        [kv-share] they no longer own a dedicated KV-cache slab either:
   *        their cache_k_l{id}/cache_v_l{id} placeholders are bound by
   *        CausalLM::allocateAndBindKVCache to the SOURCE layer's plane
   *        (same MemoryData, same offset), because the values they used to
   *        write into a private slab were byte-identical to the source's.
   *        Nothing in this layer changes: it still sees one cache tensor per
   *        input and still writes through it. The per-layer ops around mha
   *        (Q/O FC, q_norm, scalar, the post-attention add) all honor
   *        skip_prefill and skip the prefill big-step; mha_core previously
   *        ignored it and STILL ran the (heavy) prefill attention for these
   *        shared layers. When set AND this is the prefill call
   *        (to - from > 1), mha_core still WRITES + scatters K/V into its
   *        cache slab (decode attends to those positions, so they MUST be
   *        populated; under aliasing that write lands on the source's plane
   *        and stores the value already there), then early-returns BEFORE the
   *        attention compute and the (unused, downstream-also-skipped)
   *        attention output write.
   *        cache_index is re-set from the absolute `from` argument on every
   *        forward (CausalLM::setKVCachePosition / incremental_forwarding),
   *        so the skipped tail's internal advance is irrelevant to decode.
   *        Parsed from layer_impl_props::SkipPrefill in finalize().
   */
  bool skip_prefill = false;

  /**
   * @brief Whether to use externally provided cache tensors
   *        (true when num_inputs >= 5, i.e., Q, K, V + cache_key + cache_value)
   *        In external mode mha_core does not allocate its own cache tensors,
   *        and reads cache slots from input[3] (cache_key) and input[4]
   *        (cache_value) which are bound by the host (e.g.
   *        CausalLM::allocateAndBindKVCache).
   */
  bool use_external_cache = false;

  /** intermal info */
  size_t num_heads_Q;
  size_t num_heads_KV;
  size_t head_dim;
  bool cache_shift;
  float theta;
  size_t local_window_size;
  /** [kv-window-ring] Physical ring rows for this layer (0 = full max_seq, no
   * ring). Set in finalize() from causallm::kvRingCap(local_window_size,
   * max_timestep, chunk) -- the same function models/transformer.h sizes the
   * allocation with. When non-zero every cache-row index -- write offset and
   * kernel read alike -- is taken modulo this. */
  unsigned int kv_ring_cap = 0;
  /** Map an absolute cache position to its physical ring row (identity when the
   * ring is off). Used ONLY for cache-storage row offsets: RoPE and the
   * causal/window masks keep the absolute position. */
  inline size_t cacheRow(size_t abs_pos) const {
    return kv_ring_cap ? (abs_pos % (size_t)kv_ring_cap) : abs_pos;
  }
  bool use_sink = false;
  float attn_logit_softcapping = 0.0f;
  bool is_causal;
  bool use_gemm_attention = false;

  /**
   * @brief Adreno image-attention path (paper §3.7/§3.8). gpu_native runs
   *        prefill attention by reading K/V through image2d_from_buffer
   *        (read_imageui texture cache) — ~9x faster than the SVM buffer
   *        flash kernel on Adreno. The layer-graph KV cache is SVM, which
   *        cannot back an image (clCreateImage needs a cl_mem handle), so we
   *        keep per-layer cl_mem OHWI mirrors of K (layout [H_kv, S_max, d])
   *        and reversed-V ([H_kv, d, S_max]) plus their image2d views. Each
   *        prefill step scatters this step's rotated K / raw V from the SVM
   *        cache slice into the mirrors (k_scatter_ohwi_cl /
   *        v_scatter_ohwi_t_cl); attention then reads the images. Lazy-init on
   *        the first prefill step. Stored as void* so the layer header stays
   *        free of the OpenCL headers (cast to cl_mem in the .cpp).
   *        Gated by NNTR_KV_IMG_ATTN — Adreno only, since read_imageui fails
   *        to build on Intel NEO (use_image_attn: -1 unprobed, 0 off, 1 on).
   */
  void *k_buf_ohwi = nullptr;
  void *v_buf_ohwi = nullptr;
  void *k_image_ohwi = nullptr;
  void *v_image_ohwi = nullptr;
  void *v_image_ohwi_pp = nullptr; /**< NNTR_KV_VIMG_PP second
                                       full view over the same buffer */
  unsigned int v_img_pp_n = 0;     /**< calls so far (alternates the view) */
  bool kv_mirror_init = false;
  unsigned int kv_mirror_S_max = 0;
  /** [kimg-pack] shift folded into the K image view; see kimg_gsh_for(). Set
   *  once at finalize from max_timestep so every layer's K mirror and the
   *  kernels (KIMG_GSH) agree. 0 = today's one-row-per-(head,seq) view. */
  unsigned int kv_kimg_gsh = 0;
  /** [kv-window-ring] Absolute KV position held in mirror row 0. Always 0 for
   *  a linear (non-ringed) layer. A ringed layer's mirror is only ring-cap
   *  rows high and slides: it holds the absolute rows [kv_mirror_base,
   *  kv_{k,v}_valid_to), and the image kernels are handed mirror-local
   *  positions (absolute - base). Attention depends only on position
   *  DIFFERENCES once K is rotated, so the result is the linear one. */
  unsigned int kv_mirror_base = 0;
  /** [kvring-decode] Rows the mirror is BUILT with for a ringed layer, from
   *  causallm::kvMirrorRows(kv_ring_cap, W, chunk) -- what ONE launch reads
   *  back ((W-1) + C + 63), not the whole ring. 0 = derive as before (the KV
   *  cache height, i.e. max_timestep for a linear layer). Both mirror-creation
   *  sites (the finalize prebuild and the lazy engage) read this one field, so
   *  they cannot disagree -- a disagreement makes mirror_fits false, and a
   *  ringed layer then throws rather than reading stale rows. */
  unsigned int kv_mirror_rows_want = 0;
  int use_image_attn = -1;
  /** [determinism] Whether THIS layer's attention arm is the image path, from
   *  causallm::imageAttnLayer(local_window_size, max_timestep) at finalize.
   *  The fast default puts every layer on the image path; the reproducible
   *  arm (NNTR_DETERMINISTIC=1) routes a layer to the buffer/flash kernels by
   *  a per-layer rule -- so the arm is a per-LAYER fact, not a per-process
   *  one. Everything that used to read the process-wide
   *  env answer (the decode-RoPE suppression, the Q-staging target, the
   *  no-drain K/V chain, the OHWI decode RoPE) reads this instead, so the
   *  stages of one layer cannot disagree about which arm it is on. Distinct
   *  from use_image_attn, which is the RUNTIME state (-1 unprobed, and it
   *  latches to 0 when mirror creation is refused). */
  bool img_arm_want = false;

  /**
   * @brief Tight-stride V image view + mirror content tracking (texture-cache
   *        cliff fix, NNTR_KV_VTIGHT=0 disables). A V image pitch sized to the
   *        allocation cap (S_max, e.g. 2048) instead of the live sequence
   *        wastes texture cache on padding (sv_matmul 63 -> 41ms at M=843 when
   *        tight). v_image_tight is a view over the SAME v_buf_ohwi with
   *        stride kv_v_img_S; kv_v_cur_stride is the stride the buffer's
   *        CURRENT contents were scattered at (a stride change invalidates
   *        them); kv_{k,v}_valid_to track which rows [0, n) the mirrors hold
   *        (decode writes the SVM cache only, so a follow-up prefill
   *        back-fills the gap from the SVM cache before engaging the image
   *        attention).
   */
  void *v_image_tight = nullptr;
  unsigned int kv_v_img_S = 0;
  unsigned int kv_v_cur_stride = 0;
  unsigned int kv_v_valid_to = 0;
  unsigned int kv_k_valid_to = 0;

  /**
   * @brief NNTR_MHA_CLMEM slab-sync watermark: concat SVM slab rows [0, n)
   *        hold valid K/V. In that mode the prefill window writes ONLY the
   *        OHWI mirrors (no SVM side-fill); host readers (decode NEON,
   *        gemm_attention, flash fallback) gather the missing rows back
   *        from the mirrors first (k/v_gather_ohwi, one drained sync).
   */
  unsigned int kv_slab_synced_to = 0;

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
  void gemm_attention(nntrainer::Tensor &query_step,
                      nntrainer::Tensor &b_cached_key,
                      nntrainer::Tensor &b_cached_value,
                      nntrainer::Tensor &attention_output_step,
                      unsigned int N_kv, unsigned int N_q,
                      unsigned int cache_from);

  enum INOUT_INDEX {
    /** input index */
    QUERY = 0,
    KEY = 1,
    VALUE = 2,
    MASK = 3,

    /** output index */
    OUTPUT = 0,
    RETURN_ATTENTION_WEIGHT = 1,
  };

  /**< indices of the weights and tensors */
  enum AttentionParams {
    cache_key,
    cache_value,
    projected_key,
    projected_value,
    /** intended comment for later use of attention_mask */
    // attention_mask,
    attention_weight,
    dropout_mask,
    attention_output,
    /** Per-(token, head) FP16 scales for int8 KV cache (paper section
     * 3.7 int8 KV path; only requested when kv_int8 mode is active). */
    cache_key_scale,
    cache_value_scale,
  };
  std::array<unsigned int, 9> tensor_idx;

  /** True when KV cache is stored as int8 (raw bytes treated as
   * int8) + per-(token, head) FP16 scales. Enabled by setting the
   * NNTR_KV_INT8 env var; default false keeps the FP16 cache layout. */
  bool kv_int8 = false;

  /** Scale tensors for the int8 KV cache; populated by the forwarding
   * entry points when kv_int8 is active. Lifespan is the layer's, so
   * the pointers are valid for as long as the layer node exists. */
  nntrainer::Tensor *kv_int8_key_scale = nullptr;
  nntrainer::Tensor *kv_int8_value_scale = nullptr;
  /** Per-batch raw pointers into the scale tensors above, set by
   * one_batch_incremental_forwarding before the read-path call to
   * compute_kcaches / compute_fp16vcache_transposed / gemm_attention.
   * Single-batch dispatch is sequential, so racing the helpers (which
   * parallelize over heads) against these is safe. */
  const uint16_t *cur_kv_int8_key_scale_batch = nullptr;
  const uint16_t *cur_kv_int8_value_scale_batch = nullptr;
  unsigned int sink_idx;

  /** attention parameters */
  unsigned int max_position_embeddings;

  /** rope_scaling parameters */
  std::string rope_scaling_type;
  float attention_scaling = 1.0f;
  float mscale = 1.0f;
  float scale = 1.0f;
  /** fraction of head_dim that receives rotary embedding (1.0 = full rope).
   * Frequencies past partial_rotary_factor*head_dim/2 are set to 0 (identity
   * passthrough) by _compute_proportional_parameters. */
  float rope_partial_rotary_factor = 1.0f;
  unsigned int original_max_position_embeddings = 4096;

  /** set by incremental_forwarding, used by forwarding */
  unsigned int incremental_step_size = 0;

  /****************** ROTARY EMBEDDING *****************/
  /** static variable - they are all expected to be initialized once */
  /**
   * @brief Rotary frequency cache for FP32 and optional FP16 lookup tables
   */
  struct RopeFreqCache {
    /** fp32 [positions, head_dim / 2], the nntrainer::RotaryTable layout */
    std::vector<float> cos;
    std::vector<float> sin;
#ifdef ENABLE_FP16
    std::vector<std::vector<_FP16>> cos_fp16;
    std::vector<std::vector<_FP16>> sin_fp16;
    /**
     * The same fp16 angles as [positions, head_dim / 2]: the
     * nntrainer::RotaryTable layout the host RoPE reads and the LUT the
     * OpenCL RoPE uploads. The row form above stays for the CUDA arm.
     */
    std::vector<_FP16> cos_fp16_flat;
    std::vector<_FP16> sin_fp16_flat;
#endif
  };
  inline static std::unordered_map<std::string, RopeFreqCache> rope_freq_cache;
  inline static std::vector<float> *freqs_cos = {};
  inline static std::vector<float> *freqs_sin = {};
  inline static std::vector<float> thetas;
  std::vector<float> *cached_freqs_cos = {};
  std::vector<float> *cached_freqs_sin = {};
#ifdef ENABLE_FP16
  inline static std::vector<std::vector<_FP16>> *freqs_cos_fp16 = {};
  inline static std::vector<std::vector<_FP16>> *freqs_sin_fp16 = {};
  std::vector<std::vector<_FP16>> *cached_freqs_cos_fp16 = {};
  std::vector<std::vector<_FP16>> *cached_freqs_sin_fp16 = {};
  inline static std::vector<_FP16> *freqs_cos_fp16_flat = {};
  inline static std::vector<_FP16> *freqs_sin_fp16_flat = {};
  // This instance's [max_pos, head_dim / 2] fp16 cos/sin table (fp16 bits),
  // set by ensure_rope_flat_lut. It points into rope_freq_cache, whose key
  // holds every RoPE parameter, so Gemma4's sliding and full slots each keep
  // their own table. The vectors never move once built (unordered_map nodes
  // are stable), so rope_inplace_f16_cl's device-LUT cache, keyed by this host
  // pointer, uploads each table once.
  const uint16_t *rope_cos_flat_cur = nullptr;
  const uint16_t *rope_sin_flat_cur = nullptr;
  unsigned int rope_flat_positions = 0; /**< rows of that table */

  /**
   * @brief Point rope_cos/sin_flat_cur at this instance's flat fp16 RoPE
   *        table, building it through precompute_freqs on first use. Once set
   *        the pointers are returned as they are, until setProperty() clears
   *        them. Called from finalize() so the build lands at model load
   *        instead of inside the first timed prefill (KVST "lutcheck"
   *        segment), and from the GPU-RoPE and host-RoPE paths as the
   *        fallback.
   */
  void ensure_rope_flat_lut();
  /** @brief positions the RoPE LUT must cover = min(max_position_embeddings,
   *  MaxTimestep). Avoids sizing/uploading the LUT at the theoretical RoPE max
   *  (e.g. 131072) when the live cache is only max_seq_len. */
  unsigned int rope_lut_positions() const;
#endif

  /**
   * @brief pre_compute frequencies for Rotary Embedding.
   * @note it is expected to be called only once at the finalize.
   * @param[in] head_dim dimension of head
   * @param[in] seq_len sequence length
   * @param[in] theta base of theta (default = 10000)
   */
  void precompute_freqs(int head_dim, unsigned int seq_len,
                        float theta = 10000.0, bool is_fp16 = false);

  /**
   * @brief _compute frequency parameters for default ROPE
   */
  void _compute_default_parameters(int head_dim, float theta);

  /**
   * @brief _compute frequency parameters for YaRN ROPE
   */
  void _compute_yarn_parameters(int head_dim, float theta);

  /**
   * @brief _compute frequency parameters for proportional ROPE (also handles
   *        partial rotary via rope_partial_rotary_factor: frequencies beyond
   *        partial_rotary_factor*head_dim/2 are zeroed = identity passthrough)
   */
  void _compute_proportional_parameters(int head_dim, float theta);

  /**
   * @brief     apply rotary embedding
   * @param[in] in input tensor
   * @param[out] out output tensor
   * @param[in] dim hidden dim size
   * @param[in] from sequence order
   * @param[in] apply_rope true to apply rotary embedding, false to only store
   *                       the tensor into the cache dtype
   */
  void apply_rotary_emb_tensor_v2(nntrainer::Tensor &in, nntrainer::Tensor &out,
                                  unsigned int dim, unsigned int from,
                                  bool apply_rope = true);

  /**
   * @brief This layer's RoPE table for an input of dtype @p dt (the fp32 or
   *        the fp16 angles), built on first use.
   */
  nntrainer::RotaryTable rotary_table(ml::train::TensorDim::DataType dt,
                                      unsigned int dim);

  /**
   * @brief     Write one step's K / V rows into batch @p batch of the KV
   *            cache through ComputeOps::kv_cache_write.
   * @param[in] pos absolute position of the first row
   * @param[in] write_key false when K was written already (a device arm or
   *            the OHWI scatter)
   * @param[in] write_value false when V was written already
   * @param[in] rotate_key true to apply RoPE to K
   */
  void write_kv_cache(unsigned int batch, nntrainer::Tensor &key_step,
                      nntrainer::Tensor &value_step,
                      nntrainer::Tensor &cache_key,
                      nntrainer::Tensor &cache_value, unsigned int pos,
                      bool write_key, bool write_value, bool rotate_key);

  template <typename BType>
  void compute(const float *A, const BType *B, float *output, int num_rows,
               int N, int chunk_size, int group_size, int tile_size,
               bool process_all);

  void compute_kcaches(nntrainer::Tensor &in, nntrainer::Tensor &cache,
                       nntrainer::Tensor &out, unsigned int from,
                       size_t sequence_len, unsigned int num_heads,
                       unsigned int group_size, unsigned int head_dim);

  void softmax_triangle(nntrainer::Tensor &qk_out, size_t row, size_t num_heads,
                        unsigned int from);

  void softmax_triangle(nntrainer::Tensor &qk_out, size_t row, size_t num_heads,
                        unsigned int from, nntrainer::Tensor &sink_step);

  void compute_vcaches(nntrainer::Tensor &in, nntrainer::Tensor &vcache,
                       nntrainer::Tensor &out, unsigned int from,
                       size_t sequence_len, unsigned int num_heads,
                       unsigned int group_size, unsigned int head_dim);

  void compute_fp16vcache_transposed(nntrainer::Tensor &in,
                                     nntrainer::Tensor &vcache,
                                     nntrainer::Tensor &output, int from,
                                     int num_cache_head, int gqa_size,
                                     int head_dim, int to);

  /************** END OF  ROTARY EMBEDDING *************/

  /**
   * @brief calculate common derivative
   * @param context Context of the layer
   */
  void calcCommonDerivative(nntrainer::RunLayerContext &context);

  size_t calc_attn_index(size_t i);

  /**
   * @brief Windowed cumulative attention score index.
   *
   * Returns the cumulative number of attention scores before absolute query
   * row i, respecting a sliding window of size local_window_size (W):
   *
   *   S(i) = sum_{k=0}^{i-1} min(k+1, W)
   *        = (i <= W) ? i*(i+1)/2 : W*(W+1)/2 + (i - W)*W
   *
   * When W == UINT_MAX (full attention), this reduces exactly to
   * i*(i+1)/2 == calc_attn_index(i), preserving byte-identical behaviour.
   */
  size_t calc_windowed_attn_index(size_t i);

}; // end of class MHACoreLayer
} // namespace causallm

#endif

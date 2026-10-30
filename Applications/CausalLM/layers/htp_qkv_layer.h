// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moone <jijoong.moon@samsung.com>
 *
 * @file   htp_qkv_layer.h
 * @date   14 May 2020
 * @brief  This is Fully Connected Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __HTP_QKV_LAYER_H__
#define __HTP_QKV_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>
#include <memory>

namespace causallm {

namespace props {

class QUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "q_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

class KUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "k_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

class VUnit : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "v_unit";
  using prop_tag = nntrainer::uint_prop_tag;
};

/** v_norm: a gamma-less per-head RMS norm on v (feature_size wide, the
 *  same kernel as q and k without the multiply) */
class VNorm : public nntrainer::Property<bool> {
public:
  VNorm(bool val = false) : nntrainer::Property<bool>(val) {}
  using prop_tag = nntrainer::bool_prop_tag;
  static constexpr const char *key = "v_norm";
};

/** v_from_k: no v weight; v is the raw k projection, before k_norm (a
 *  model whose attention has K == V). Needs feature_size. */
class VFromK : public nntrainer::Property<bool> {
public:
  VFromK(bool val = false) : nntrainer::Property<bool>(val) {}
  using prop_tag = nntrainer::bool_prop_tag;
  static constexpr const char *key = "v_from_k";
};

/** q_scale: multiplied into q after its norm (folded into the gamma
 *  multiply), e.g. sqrt(head_dim) for a model whose attention scaling is
 *  1.0 against a core that divides by sqrt(head_dim) */
class QScale : public nntrainer::Property<float> {
public:
  QScale(float val = 1.0f) : nntrainer::Property<float>(val) {}
  using prop_tag = nntrainer::float_prop_tag;
  static constexpr const char *key = "q_scale";
};

/**
 * @brief rope: RoPE on q and k after their norms, with rope_theta,
 *        rope_scaling_type (default or proportional),
 * rope_partial_rotary_factor and max_timestep (the table's rows) as the
 * attention core takes them -- whose use_rope is then off. The head dim is
 * feature_size. On the accelerator it rides the projection call; on the CPU it
 * is the attention core's own kernel over the same table.
 */
class Rope : public nntrainer::Property<bool> {
public:
  Rope(bool val = false) : nntrainer::Property<bool>(val) {}
  using prop_tag = nntrainer::bool_prop_tag;
  static constexpr const char *key = "rope";
};

} // namespace props

/**
 * @class   HtpQKVLayer
 * @brief   The q, k and v projections of one attention block as ONE layer,
 *          with the per-head RMS norm of q and k folded in when
 *          feature_size is set.
 *
 * One layer so the three matmuls share one activation: an accelerator
 * takes them as one call (Tensor::dot's vector overload, doc 51 section
 * 2.21 -- three calls each quantized and shipped the same 3.6 MB
 * activation), and on the CPU it is the same three GEMMs. With
 * feature_size (the head dim) the layer holds, in the model file's
 * order, q, q_norm's gamma, k, k_norm's gamma, v, and applies the same
 * rms_norm_wrt_width + gamma multiply ReshapedRMSNormLayer would; the
 * outputs are q_normed, k_normed, v. Without it: q, k, v as they are.
 * v_norm adds the gamma-less norm on v, q_scale a factor on q after its
 * norm, and v_from_k drops the v weight and takes v from the raw k
 * projection -- the three things one model's attention does around the
 * projections, folded in so one call still covers the block. in_norm
 * folds the block's input RMSNorm in too (its gamma leads the weights).
 * On an accelerator with gemm_q4_0_batch_norm_fp32 every norm here rides
 * the projection call; on the CPU they are the same kernels in sequence.
 */
WIN_EXPORT class HtpQKVLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  WIN_EXPORT HtpQKVLayer();

  /**
   * @brief     Destructor of Fully Connected Layer
   */
  WIN_EXPORT ~HtpQKVLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] FullyConnected &&
   */
  WIN_EXPORT HtpQKVLayer(HtpQKVLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs HtpQKVLayer to be moved.
   */
  WIN_EXPORT HtpQKVLayer &operator=(HtpQKVLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   * @note
   * [note for LoRA] implicit calcDerivative is implicitly applied.
   * The weight is already updated with the LoRA's (W = W + W_lora)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return HtpQKVLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  inline static const std::string type = "htp_qkv_layer";

private:
  /** feature_size: the head dim q and k are normed over; unset = no norm */
  std::tuple<props::QUnit, props::KUnit, props::VUnit, props::FeatureSize,
             nntrainer::props::Epsilon, props::VNorm, props::VFromK,
             props::QScale, props::InNorm, props::Rope, props::RopeTheta,
             props::RopeScalingType, props::RopePartialRotaryFactor,
             nntrainer::props::MaxTimestep, props::NormInCall, props::OutToHost>
    qkv_props;
  std::array<unsigned int, 6>
    weight_idx; /**< [in_gamma,] q, [q_gamma,] k, [k_gamma,] v */
  std::array<unsigned int, 4>
    tensor_idx; /**< q, k (and v) before the norm; the normed input */
  bool in_norm = false;
  bool lazy_scratch = false; /**< Allocate HTP fallback scratch only on use */
  bool norm_in_call = true;  /**< props::NormInCall */
  bool out_to_host = true;   /**< props::OutToHost */
  unsigned int feature_size = 0;
  bool v_norm = false;
  bool v_from_k = false;
  float q_scale = 1.0f;
  bool rope = false;
  unsigned int rope_rows = 0; /**< positions the table holds */
  /** [pos][cos row (feature_size, halves duplicated) | sin row], shared by
   *  every layer of the same shape */
  std::shared_ptr<const std::vector<float>> rope_table;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __HTP_QKV_LAYER_H__ */

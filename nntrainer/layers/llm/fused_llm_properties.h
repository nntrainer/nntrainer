// SPDX-License-Identifier: Apache-2.0
/**
 * @file fused_llm_properties.h
 * @brief Properties shared by fused language-model layers.
 * @author NNTrainer contributors
 * @bug See the project issue tracker.
 */
#ifndef NNTR_FUSED_LLM_PROPERTIES_H
#define NNTR_FUSED_LLM_PROPERTIES_H
#include <base_properties.h>
namespace nntrainer {
namespace props {
class FeatureSize : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "feature_size";                          /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};
/**
 * @brief in_norm: the layer owns the RMSNorm gamma of its input (hidden
 *        wide, FP32, the first weight in the file) and applies that norm
 *        itself -- on an accelerator inside the same call as its matmuls.
 */
class InNorm : public nntrainer::Property<bool> {
public:
  InNorm(bool val = false) : nntrainer::Property<bool>(val) {}
  using prop_tag = nntrainer::bool_prop_tag;
  static constexpr const char *key = "in_norm";
};

/**
 * @brief norm_in_call: [#260] qkv_layer's per-head q / k norms
 *        (feature_size) go into the accelerator's projection call with
 *        no in_norm too (#4415's default). LFM2 sets it false: its prefill
 *        keeps the norms on the CPU, as before #4415.
 */
class NormInCall : public nntrainer::Property<bool> {
public:
  NormInCall(bool val = true) : nntrainer::Property<bool>(val) {}
  using prop_tag = nntrainer::bool_prop_tag;
  static constexpr const char *key = "norm_in_call";
};

/**
 * @brief use_weight: the layer's scalar comes from the weight file (one
 *        float), not from a property. scalar_multiply's, shared with
 *        residual_add, which folds the block's scalar into its add.
 */
class UseWeight : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "use_weight"; /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag;       /**< property type */
  UseWeight(bool value = false) { set(value); }
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
 * @brief RopePartialRotaryFactor
 */
class RopePartialRotaryFactor : public nntrainer::Property<float> {
public:
  RopePartialRotaryFactor(float value = 1.0f) { set(value); };
  static constexpr const char *key =
    "rope_partial_rotary_factor";             /**< unique key to access */
  using prop_tag = nntrainer::float_prop_tag; /**< property type */
};

/**
 * @brief softcap: y = softcap * tanh(y / softcap) on a layer's output (the
 *        final logit softcap, folded into the tied lm_head); 0 for none.
 */
class Softcap : public nntrainer::Property<float> {
public:
  Softcap(float val = 0.0f) : nntrainer::Property<float>(val) {}
  using prop_tag = nntrainer::float_prop_tag;
  static constexpr const char *key = "softcap";
};

/**
 * @brief out_norm: likewise the RMSNorm of the layer's output (its gamma
 *        is the last weight in the file).
 */
class OutNorm : public nntrainer::Property<bool> {
public:
  OutNorm(bool val = false) : nntrainer::Property<bool>(val) {}
  using prop_tag = nntrainer::bool_prop_tag;
  static constexpr const char *key = "out_norm";
};

/**
 * @brief router_norm: an MoE layer's router reads its own RMSNorm of the
 *        layer's input (gamma after in_norm's in the file).
 */
class RouterNorm : public nntrainer::Property<bool> {
public:
  RouterNorm(bool val = false) : nntrainer::Property<bool>(val) {}
  using prop_tag = nntrainer::bool_prop_tag;
  static constexpr const char *key = "router_norm";
};

/**
 * @brief proj: a residual_add also holds the projection before it (the
 *        attention's o-proj): input 1 is the projection's input and the
 *        weight is requested first, as that layer's was (doc 57 section
 *        9.32).
 */
class Proj : public nntrainer::Property<bool> {
public:
  Proj(bool val = false) : nntrainer::Property<bool>(val) {}
  using prop_tag = nntrainer::bool_prop_tag;
  static constexpr const char *key = "proj";
};

/** out_to_host: false when every consumer of this layer's prefill output
 *  runs on the same accelerator, so the layer's call may keep the output
 *  there instead of copying it to the host tensor (ComputeOps's
 *  out_to_host; residual_add with proj, dense_ffn, lfm2_moe). The model
 *  builder decides; a decode row is always written. */
class OutToHost : public nntrainer::Property<bool> {
public:
  OutToHost(bool val = true) : nntrainer::Property<bool>(val) {}
  using prop_tag = nntrainer::bool_prop_tag;
  static constexpr const char *key = "out_to_host";
};

/**
 * @brief out_add: an MoE layer also takes the decoder block's post-FFN
 *        epilogue: out = scalar * (input + rmsnorm(moe + input 1) * gamma),
 *        input 1 being the dense branch's output (gamma and the scalar the
 *        last two weights in the file, as the residual_add they replace).
 */
class OutAdd : public nntrainer::Property<bool> {
public:
  OutAdd(bool val = false) : nntrainer::Property<bool>(val) {}
  using prop_tag = nntrainer::bool_prop_tag;
  static constexpr const char *key = "out_add";
};

} // namespace props
} // namespace nntrainer
#endif

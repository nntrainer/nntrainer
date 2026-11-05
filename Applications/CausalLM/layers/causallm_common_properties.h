/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	causallm_common_properties.h
 * @date	23 July 2025
 * @brief	This defines a qwen3 causal language model.
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __CAUSALLM_COMMON_PROPERTIES_H__
#define __CAUSALLM_COMMON_PROPERTIES_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <base_properties.h>
#include <common_properties.h>
#include <tensor.h>
#include <utility>

namespace causallm {

namespace props {

/**
 * @brief MoE activation type
 */
class MoEActivation final
  : public nntrainer::EnumProperty<nntrainer::props::ActivationTypeInfo> {
public:
  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "moe_activation";
};
/**
 * @brief MoE router type: how logits become the top-k set and its weights.
 *  "sigmoid_bias" (default) is LFM2's: sigmoid scores, a per-expert bias
 *  added for selection only, weights normalised over the top-k.
 *  "softmax_scale" is Gemma-4's: softmax scores, top-k, weights normalised
 *  to sum 1, then multiplied by a per-expert scale. The layer's one
 *  [num_experts] weight is the bias for the first and the scale for the
 *  second.
 */
class RouterType : public nntrainer::Property<std::string> {
public:
  RouterType(const std::string &value = "sigmoid_bias") :
    nntrainer::Property<std::string>(value) {}
  static constexpr const char *key = "router_type";
  using prop_tag = nntrainer::str_prop_tag;
};

/**
 * @brief [plan 201 S4] The MoE router: "sigmoid" (LFM2: sigmoid scores plus
 *        an expert bias for the selection) or "softmax" (Gemma 4: an
 *        un-normed router input RMS-normed and scaled, softmax, top-k,
 *        renormalised, times a per-expert scale)
 */
class MoERouter : public nntrainer::Property<std::string> {
public:
  MoERouter(std::string value = "sigmoid") { set(value); };
  static constexpr const char *key = "moe_router";
  using prop_tag = nntrainer::str_prop_tag;
};

/**
 * @brief Per-layer expert cache size C for the streamed (virtual) expert
 *  path, doc 52/55. 0 keeps the experts resident. NNTR_MOE_CACHE_EXPERTS in
 *  the environment overrides it, so one device can sweep C without a
 *  config edit.
 */
class CacheExperts : public nntrainer::Property<unsigned int> {
public:
  CacheExperts(unsigned int value = 0) :
    nntrainer::Property<unsigned int>(value) {}
  static constexpr const char *key = "cache_experts";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief NumExperts,  Number of experts property
 */
class NumExperts : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "num_experts"; /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag;        /**< property type */
};

/**
 * @brief NumExpertsPerToken,  Number of experts per token property
 */
class NumExpertsPerToken : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "num_experts_per_token";                 /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief unit property, unit is used to measure how many weights are there
 *
 */
class FeatureSize : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "feature_size";                          /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief UseGamma property for RMSNorm scale usage.
 */
class UseGamma : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "use_gamma";
  using prop_tag = nntrainer::bool_prop_tag;
  UseGamma(bool value = true) { set(value); }
};

/**
 * @brief RMS_NORM_GAMMA_INIT Initialization Enumeration Information
 *
 */
WIN_EXPORT class RMS_NORM_GAMMA_INIT final
  : public nntrainer::EnumProperty<nntrainer::props::InitializerInfo> {
public:
  /**
   * @brief Construct a CUSTOM_RMS_NORM_GAMMA_INIT object
   */
  WIN_EXPORT RMS_NORM_GAMMA_INIT(
    nntrainer::Initializer value = nntrainer::Initializer::ONES) {
    set(value);
  };

  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "gamma_initializer";
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
 *        dense_ffn's form has no input 1 and no scalar: out = input +
 *        out_norm(ffn), out_norm's gamma being that residual_add's.
 */
class OutAdd : public nntrainer::Property<bool> {
public:
  OutAdd(bool val = false) : nntrainer::Property<bool>(val) {}
  using prop_tag = nntrainer::bool_prop_tag;
  static constexpr const char *key = "out_add";
};

}; // namespace props

WIN_EXPORT enum RMSParams { gamma };

} // namespace causallm

#endif

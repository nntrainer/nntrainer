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
 * @brief	This defines layer properties shared by the CausalLM layers.
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
#include <fused_llm_properties.h>
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
using nntrainer::props::FeatureSize;

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
   * @brief Construct a RMS_NORM_GAMMA_INIT object
   */
  WIN_EXPORT RMS_NORM_GAMMA_INIT(
    nntrainer::Initializer value = nntrainer::Initializer::ONES) {
    set(value);
  };

  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "gamma_initializer";
};
using nntrainer::props::InNorm;
using nntrainer::props::NormInCall;
using nntrainer::props::OutAdd;
using nntrainer::props::OutNorm;
using nntrainer::props::OutToHost;
using nntrainer::props::Proj;
using nntrainer::props::RopePartialRotaryFactor;
using nntrainer::props::RopeScalingType;
using nntrainer::props::RopeTheta;
using nntrainer::props::RouterNorm;
using nntrainer::props::Softcap;
using nntrainer::props::UseWeight;
}; // namespace props

WIN_EXPORT enum RMSParams { gamma };

} // namespace causallm

#endif

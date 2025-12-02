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
 * @brief NumSharedExpers,  Number of shared experts property
 */
class NumSharedExperts : public nntrainer::Property<unsigned int> {
public:
  static constexpr const char *key = "num_shared_experts";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief MoENormMin, Minimum value for MoE normalization
 */
class MoENormMin : public nntrainer::Property<float> {
public:
  static constexpr const char *key = "moe_norm_min";
  using prop_tag = nntrainer::float_prop_tag;
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
}; // namespace props

WIN_EXPORT enum RMSParams { gamma };

} // namespace causallm

#endif

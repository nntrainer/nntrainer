// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   common_properties.h
 * @date   09 April 2021
 * @brief  This file contains list of common properties widely used across
 * layers
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef __QNN_PROPERTIES_H__
#define __QNN_PROPERTIES_H__

#include <array>
#include <fstream>
#include <string>

#include <common_properties.h>

namespace nntrainer {

namespace props {

/**
 * @brief property is treated as quant param, eg 0.001:-12345
 *
 */
struct quant_param_prop_tag {};

/**
 * @class   QuantParam
 * @brief   (name, (scale, zero_point)) tuple describing a tensor's quantization
 *          parameters; passed into a QNN layer via property string.
 */
class QuantParam
  : public nntrainer::Property<std::pair<std::string, std::pair<float, int>>> {
public:
  static constexpr const char *key = "quant_param"; /**< unique key to access */
  using prop_tag = quant_param_prop_tag;            /**< property type */
};

/**
 * @class   InputQuantParam
 * @brief   QuantParam variant pinned to a layer's input tensor.
 */
class InputQuantParam
  : public nntrainer::Property<std::pair<std::string, std::pair<float, int>>> {
public:
  static constexpr const char *key =
    "input_quant_param";                 /**< unique key to access */
  using prop_tag = quant_param_prop_tag; /**< property type */
};

/**
 * @class   OutputQuantParam
 * @brief   QuantParam variant pinned to a layer's output tensor.
 */
class OutputQuantParam
  : public nntrainer::Property<std::pair<std::string, std::pair<float, int>>> {
public:
  static constexpr const char *key =
    "output_quant_param";                /**< unique key to access */
  using prop_tag = quant_param_prop_tag; /**< property type */
};

} // namespace props

template <>
std::string str_converter<props::quant_param_prop_tag,
                          std::pair<std::string, std::pair<float, int>>>::
  to_string(const std::pair<std::string, std::pair<float, int>> &quant_param);

template <>
std::pair<std::string, std::pair<float, int>>
str_converter<props::quant_param_prop_tag,
              std::pair<std::string, std::pair<float, int>>>::
  from_string(const std::string &value);

} // namespace nntrainer

#endif // __QNN_PROPERTIES_H__

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   neuron_properties.h
 * @date   30 Jul 2026
 * @brief  Properties for the neuron_graph layer
 * @see    https://github.com/nnstreamer/nntrainer
 *
 * @details Structurally mirrors nntrainer/qnn/jni/qnn_properties.h's
 * InputQuantParam/OutputQuantParam ("name:scale:offset"), but on the Neuron
 * backend these are descriptive metadata only: unlike QNN's Qnn_Tensor_t,
 * there is no runtime API to stamp quantization onto a Neuron tensor — it
 * is baked into the .dla at compile time. They exist so app-layer code
 * (Phase 2) has a place to carry scale/offset alongside the layer, for its
 * own de/requantization math on the host side, exactly as QNN's app layer
 * already does with these same values.
 *
 * Deliberately not defined here (own module vs. reusing qnn_properties.h's
 * classes): the neuron and qnn contexts are independent plugin .so's built
 * under different feature flags, so this header intentionally does not
 * depend on anything under nntrainer/qnn/.
 */
#ifndef __NEURON_PROPERTIES_H__
#define __NEURON_PROPERTIES_H__

#include <array>
#include <fstream>
#include <string>

#include <common_properties.h>

namespace nntrainer {

namespace props {

/**
 * @brief property is treated as quant param, eg 0.001:-12345
 */
struct neuron_quant_param_prop_tag {};

/** @brief Descriptive quantization scale/zero-point for a neuron_graph
 * input tensor. Not applied at runtime; see file-level note. */
class NeuronInputQuantParam
  : public nntrainer::Property<std::pair<std::string, std::pair<float, int>>> {
public:
  static constexpr const char *key = "input_quant_param";
  using prop_tag = neuron_quant_param_prop_tag;
};

/** @brief Descriptive quantization scale/zero-point for a neuron_graph
 * output tensor. Not applied at runtime; see file-level note. */
class NeuronOutputQuantParam
  : public nntrainer::Property<std::pair<std::string, std::pair<float, int>>> {
public:
  static constexpr const char *key = "output_quant_param";
  using prop_tag = neuron_quant_param_prop_tag;
};

} // namespace props

template <>
std::string str_converter<props::neuron_quant_param_prop_tag,
                          std::pair<std::string, std::pair<float, int>>>::
  to_string(const std::pair<std::string, std::pair<float, int>> &quant_param);

template <>
std::pair<std::string, std::pair<float, int>>
str_converter<props::neuron_quant_param_prop_tag,
              std::pair<std::string, std::pair<float, int>>>::
  from_string(const std::string &value);

} // namespace nntrainer

#endif // __NEURON_PROPERTIES_H__

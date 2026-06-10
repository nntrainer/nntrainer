// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   layer_impl.h
 * @date   21 June 2021
 * @brief  This is base layer implementation class
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <layer_impl.h>

#include <string>
#include <vector>

#include <common_properties.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>
namespace nntrainer {

LayerImpl::LayerImpl() :
  layer_impl_props(
    std::make_unique<
      std::tuple<props::WeightRegularizer, props::WeightRegularizerConstant,
                 props::WeightInitializer, props::WeightDecay, props::BiasDecay,
                 props::BiasInitializer, props::DisableBias, props::Print,
                 props::SkipPrefill>>()) {}

void LayerImpl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, *layer_impl_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[LayerImpl] Unknown Layer Properties count " +
         std::to_string(values.size());
}

std::string LayerImpl::getProperty(const std::string &key) {
  if (!layer_impl_props)
    return "";

  std::string result = find_in_tuple(*layer_impl_props, key);
  return !result.empty() ? result : "";
}

void LayerImpl::exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const {
  exporter.saveResult(*layer_impl_props, method, this);
}

} // namespace nntrainer

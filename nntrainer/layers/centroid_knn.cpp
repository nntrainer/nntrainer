// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   centroid_knn.cpp
 * @date   09 Jan 2021
 * @brief  This file contains the simple nearest neighbor layer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details This layer takes centroid and calculate l2 distance
 */

#include <iostream>
#include <limits>
#include <regex>
#include <sstream>

#include <centroid_knn.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor.h>
#include <weight.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum KNNParams { map, num_samples };

CentroidKNN::CentroidKNN() : Layer(), centroid_knn_props(props::NumClass()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

CentroidKNN::~CentroidKNN() {}

void CentroidKNN::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, centroid_knn_props);
  NNTR_THROW_IF(!left.empty(), std::invalid_argument)
    << "[Centroid KNN] there are unparsed properties " << left.front();
}

void CentroidKNN::finalize(nntrainer::InitLayerContext &context) {
  auto const &input_dim = context.getInputDimensions()[0];
  if (input_dim.channel() != 1 || input_dim.height() != 1) {
    ml_logw("centroid nearest layer is designed for flattend feature for now, "
            "please check");
  }

  auto num_class = std::get<props::NumClass>(centroid_knn_props);

  auto output_dim = nntrainer::TensorDim({num_class});
  context.setOutputDimensions({output_dim});

  /// weight is a distance map that contains centroid of features of each class
  auto map_dim = nntrainer::TensorDim({num_class, input_dim.getFeatureLen()});

  /// samples seen for the current run to calculate the centroid
  auto samples_seen = nntrainer::TensorDim({num_class});

  weight_idx[KNNParams::map] = context.requestWeight(
    map_dim, nntrainer::Initializer::ZEROS, nntrainer::WeightRegularizer::NONE,
    1.0f, 0.0f, "map", false);

  weight_idx[KNNParams::num_samples] = context.requestWeight(
    samples_seen, nntrainer::Initializer::ZEROS,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "num_samples", false);
}

void CentroidKNN::forwarding(nntrainer::RunLayerContext &context,
                             bool training) {
  auto &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  auto &input_ = context.getInput(SINGLE_INOUT_IDX);
  const auto &input_dim = input_.getDim();

  auto &map = context.getWeight(weight_idx[KNNParams::map]);
  auto &num_samples = context.getWeight(weight_idx[KNNParams::num_samples]);
  auto feature_len = input_dim.getFeatureLen();

  auto get_distance = [](const nntrainer::Tensor &a,
                         const nntrainer::Tensor &b) {
    return -a.subtract(b).l2norm();
  };

  if (training) {
    auto &label = context.getLabel(SINGLE_INOUT_IDX);
    auto ans = label.argmax();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      auto saved_feature =
        map.getSharedDataTensor({feature_len}, ans[b] * feature_len);

      //  nntrainer::Tensor::Map(map.getData(), {feature_len},
      // ans[b] * feature_len);
      float num_sample = num_samples.getValue<float>(0, 0, 0, ans[b]);
      auto current_feature = input_.getBatchSlice(b, 1);
      saved_feature.multiply_i(num_sample);
      saved_feature.add_i(current_feature);
      saved_feature.divide_i(num_sample + 1);
      num_samples.setValue(0, 0, 0, ans[b], num_sample + 1);
    }
  }

  for (unsigned int i = 0; i < std::get<props::NumClass>(centroid_knn_props);
       ++i) {
    auto saved_feature =
      map.getSharedDataTensor({feature_len}, i * feature_len);
    // nntrainer::Tensor::Map(map.getData(), {feature_len}, i * feature_len);

    auto num_sample = num_samples.getValue(0, 0, 0, i);

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      auto current_feature = input_.getBatchSlice(b, 1);

      if (num_sample == 0) {
        // Unobserved classes must rank below the negative distance scores.
        hidden_.setValue(b, 0, 0, i, std::numeric_limits<float>::lowest());
      } else {
        hidden_.setValue(b, 0, 0, i,
                         get_distance(current_feature, saved_feature));
      }
    }
  }
}

void CentroidKNN::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::invalid_argument("[CentroidKNN::calcDerivative] This Layer "
                              "does not support backward propagation");
}

void CentroidKNN::exportTo(nntrainer::Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  exporter.saveResult(centroid_knn_props, method, this);
}
} // namespace nntrainer

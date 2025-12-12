// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   multi_loader.h
 * @date   5 July 2023
 * @brief  multi data loader
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include "multi_loader.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <nntrainer_error.h>
#include <numeric>
#include <random>

namespace nntrainer::util {

namespace {
/**
 * @brief fill last to the given memory
 * @note this function increases iteration value, if last is set to true,
 * iteration resets to 0
 *
 * @param[in/out] iteration current iteration
 * @param data_size Data size
 * @return bool true if iteration has finished
 */
bool updateIteration(unsigned int &iteration, unsigned int data_size) {
  if (++iteration == data_size) {
    iteration = 0;
    return true;
  }
  return false;
};

} // namespace

MultiInoutDataLoader::MultiInoutDataLoader(
  const std::vector<TensorDim> &input_shapes,
  const std::vector<TensorDim> &output_shapes, int data_size_) :
  iteration(0),
  data_size(data_size_),
  count(0),
  input_shapes(input_shapes),
  output_shapes(output_shapes),
  input_dist(-1.0f, 1.0f),
  label_dist(0, output_shapes.front().width() - 1) {
  NNTR_THROW_IF(output_shapes.empty(), std::invalid_argument)
    << "output_shape size empty not supported";

  indicies = std::vector<unsigned int>(data_size_);
  std::iota(indicies.begin(), indicies.end(), 0);
  std::shuffle(indicies.begin(), indicies.end(), rng);
}

void MultiInoutDataLoader::next(float **input, float **label, bool *last) {

  // Fill input tensors with random values
  float **cur_input_tensor = input;
  for (unsigned int i = 0; i < input_shapes.size(); ++i) {
    // For each input tensor, fill with random values
    unsigned int tensor_len = input_shapes.at(i).getFeatureLen();
    for (unsigned int j = 0; j < tensor_len; ++j) {
      (*cur_input_tensor)[j] = input_dist(rng);
    }
    ++cur_input_tensor;
  }

  // Get the first value of input0 for target calculation
  float first_input_value = input[1][0];

  // Calculate target outputs: double and square of first input value
  // float double_value = 2.0f * first_input_value;
  // float square_value = first_input_value * first_input_value;

  float double_value = 2.0f * first_input_value;
  float square_value = first_input_value * first_input_value;

  // Fill label tensors with calculated target values
  float **cur_label_tensor = label;
  for (unsigned int i = 0; i < output_shapes.size(); ++i) {
    unsigned int tensor_len = output_shapes.at(i).getFeatureLen();
    float target_value = (i == 0) ? double_value : square_value;

    for (unsigned int j = 0; j < tensor_len; ++j) {
      (*cur_label_tensor)[j] = target_value;
    }
    cur_label_tensor++;
  }

  if (updateIteration(iteration, data_size)) {
    std::shuffle(indicies.begin(), indicies.end(), rng);
    *last = true;
    count = 0;
  } else {
    *last = false;
    count++;
  }
}

} // namespace nntrainer::util

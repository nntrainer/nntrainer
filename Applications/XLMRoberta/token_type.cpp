// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   token_type.cpp
 * @date   7 January 2026
 * @brief  Implementation of token type layer for XLMRoberta
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "token_type.h"

namespace xlmroberta {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void TokenType::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();

  for (unsigned int i = 0; i < dim.size(); ++i) {
    if (dim[i].getDataLen() == 0) {
      throw std::invalid_argument("Input dimension is not set");
    } else {
      dim[i].channel(dim[i].channel());
      dim[i].height(dim[i].height());
      dim[i].width(dim[i].width());
    }
  }

  context.setOutputDimensions(dim);
}

void TokenType::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  // Get input tensor (input_ids)
  nntrainer::Tensor &input_ids = context.getInput(SINGLE_INOUT_IDX);

  // Get output tensor (token_type_ids)
  nntrainer::Tensor &token_type_ids = context.getOutput(SINGLE_INOUT_IDX);

  // Get tensor dimensions
  auto input_dim = input_ids.getDim();
  unsigned int batch_size = input_dim.batch();
  unsigned int seq_length = input_dim.width();

  // First create position_ids from input_ids
  nntrainer::Tensor position_ids(input_dim);

  // Assuming padding_idx is 1 (common default)
  int padding_idx = 1;

  // Create position_ids from input_ids
  for (unsigned int b = 0; b < batch_size; ++b) {
    for (unsigned int i = 0; i < seq_length; ++i) {
      int token_id = static_cast<int>(input_ids.getValue(b, 0, 0, i));
      if (token_id != padding_idx) {
        position_ids.setValue(b, 0, 0, i, static_cast<float>(padding_idx + 1 + i));
      } else {
        position_ids.setValue(b, 0, 0, i, static_cast<float>(0));
      }
    }
  }

  // Create token_type_ids (all zeros as in the Python code when token_type_ids is None)
  for (unsigned int b = 0; b < batch_size; ++b) {
    for (unsigned int i = 0; i < seq_length; ++i) {
      token_type_ids.setValue(b, 0, 0, i, static_cast<float>(0));
    }
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_token_type_layer() {
  auto layer = new TokenType();
  std::cout << "token type layer created\n";
  return layer;
}

void destroy_token_type_layer(nntrainer::Layer *layer) {
  std::cout << "token type layer deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_token_type_layer,
                                                   destroy_token_type_layer};
}

#endif

} // namespace xlmroberta

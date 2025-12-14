// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Yeonjae Kim <duswo1120@snu.ac.kr>
 * Copyright (C) 2025 Hoyeon Jo <jhy213@snu.ac.kr>
 *
 * @file   checkpoint_block.cpp
 * @date   23 Oct 2025
 * @brief  Checkpoint block implementation
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Yeonjae Kim <duswo1120@snu.ac.kr>
 * @author Hoyeon Jo <jhy213@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 */

#include <checkpoint_block.h>
#include <stdexcept>

namespace nntrainer {

CheckpointBlock::CheckpointBlock(const std::vector<std::string> &layers,
                                 const std::string &id) :
  layer_names(layers),
  block_id(id),
  enabled(true) {

  if (layers.empty()) {
    throw std::invalid_argument("CheckpointBlock: layer list cannot be empty");
  }

  start_layer = layers.front();
  end_layer = layers.back();

  // Generate block ID if not provided
  if (block_id.empty()) {
    block_id = "checkpoint_block_" + start_layer + "_to_" + end_layer;
  }
}

} // namespace nntrainer

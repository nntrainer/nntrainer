// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Yeonjae Kim <duswo1120@snu.ac.kr>
 * Copyright (C) 2025 Hoyeon Jo <jhy213@snu.ac.kr>
 *
 * @file   checkpoint_block.h
 * @date   23 Oct 2025
 * @brief  Checkpoint block for gradient checkpointing
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Yeonjae Kim <duswo1120@snu.ac.kr>
 * @author Hoyeon Jo <jhy213@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 */

#ifndef __CHECKPOINT_BLOCK_H__
#define __CHECKPOINT_BLOCK_H__

#include <string>
#include <vector>

namespace nntrainer {

/**
 * @class CheckpointBlock
 * @brief Represents a block of layers for gradient checkpointing
 *
 * @details Gradient checkpointing saves memory by not storing intermediate
 * activations during forward pass. Instead, they are recomputed during
 * backward pass when needed.
 */
class CheckpointBlock {
private:
  std::vector<std::string> layer_names; /**< Names of layers in this block */
  std::string start_layer;              /**< First layer in the block */
  std::string end_layer;                /**< Last layer in the block */
  std::string block_id;                 /**< Unique identifier for this block */
  bool enabled;                         /**< Whether checkpointing is enabled */

public:
  /**
   * @brief Construct a new Checkpoint Block object
   *
   * @param layers Vector of layer names to include in this block
   * @param id Unique identifier for this block
   */
  CheckpointBlock(const std::vector<std::string> &layers,
                  const std::string &id = "");

  /**
   * @brief Get the layer names in this block
   *
   * @return const std::vector<std::string>& Layer names
   */
  const std::vector<std::string> &getLayerNames() const { return layer_names; }

  /**
   * @brief Get the start layer name
   *
   * @return std::string Name of the first layer
   */
  std::string getStartLayer() const { return start_layer; }

  /**
   * @brief Get the end layer name
   *
   * @return std::string Name of the last layer
   */
  std::string getEndLayer() const { return end_layer; }

  /**
   * @brief Get the block ID
   *
   * @return std::string Block identifier
   */
  std::string getBlockId() const { return block_id; }

  /**
   * @brief Check if checkpointing is enabled
   *
   * @return true if enabled, false otherwise
   */
  bool isEnabled() const { return enabled; }

  /**
   * @brief Enable or disable checkpointing for this block
   *
   * @param enable true to enable, false to disable
   */
  void setEnabled(bool enable) { enabled = enable; }

  /**
   * @brief Get the number of layers in this block
   *
   * @return size_t Number of layers
   */
  size_t size() const { return layer_names.size(); }
};

} // namespace nntrainer

#endif // __CHECKPOINT_BLOCK_H__

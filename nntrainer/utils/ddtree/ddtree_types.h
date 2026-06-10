// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @brief  Plain-old-data structures for the DDTree speculative-decoding core
 * @file   ddtree_types.h
 * @date   05 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef __NNTRAINER_DDTREE_TYPES_H__
#define __NNTRAINER_DDTREE_TYPES_H__

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nntrainer {
namespace ddtree {

/**
 * @brief Runtime configuration. budget = max tree nodes minus 1 ("32 tree" ==
 * budget 31).
 */
struct DDTreeConfig {
  int budget = 31;    /**< heap expansion cap; "32 tree" == budget 31 */
  int depthLimit = 0; /**< draft horizon = block_size - 1 */
  float maskFillValue =
    0; /**< additive-mask -inf == finfo(dtype).min (fp32 or fp16 min) */
};

/**
 * @brief Output of buildTree (== build_ddtree_tree). currentLength =
 * 1 + nodeCount.
 */
struct DDTreeStructure {
  std::vector<int32_t> nodeTokenIds; /**< [nodeCount] */
  std::vector<int32_t> nodeDepths;   /**< [nodeCount], 1-based depth */
  std::vector<int32_t> parents;      /**< [currentLength], parents[0] = -1 */
  std::vector<std::unordered_map<int32_t, int32_t>>
    childMaps; /**< [currentLength] */
  std::vector<uint8_t>
    visibility; /**< [currentLength*currentLength] row-major bitmap */
  int nodeCount = 0;
  int currentLength = 1;
};

/**
 * @brief Slice lengths returned by compile (== compile_ddtree_tree).
 */
struct CompiledTree {
  int pastLength = 0;
  int currentLength = 0;
};

/**
 * @brief Result of makeSlidingMasks. When hasSliding is false, use `full` for
 * all layers.
 */
struct SlidingMasks {
  float *full = nullptr; /**< full-attention layers */
  float *sliding =
    nullptr; /**< sliding-attention layers (== full when window<=0) */
  bool hasSliding = false; /**< true => select full vs sliding per layer type */
};

/**
 * @brief Result of followVerified (== follow_verified_tree).
 */
struct Accepted {
  std::vector<int32_t>
    indices;             /**< accepted node indices, indices[0]==0 (root) */
  int32_t nextToken = 0; /**< bonus token after the accepted path */
};

} // namespace ddtree
} // namespace nntrainer

#endif // __NNTRAINER_DDTREE_TYPES_H__

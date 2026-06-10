// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @brief  DDTree core: candidate-tree build, verify-buffer compile,
 * accepted-path follow
 * @file   ddtree.h
 * @date   05 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef __NNTRAINER_DDTREE_H__
#define __NNTRAINER_DDTREE_H__

#include <cstdint>

#include <ddtree_types.h>

namespace nntrainer {
namespace ddtree {

/**
 * @brief Build the best-first candidate token tree (== build_ddtree_tree).
 * @param draftLogits row-major [depthLimit, vocab] draft logits (fp32)
 * @param depthLimit  draft horizon (block_size - 1)
 * @param vocab       vocabulary size (row width of draftLogits)
 * @param cfg         config; uses cfg.budget
 */
DDTreeStructure buildTree(const float *draftLogits, int depthLimit, int vocab,
                          const DDTreeConfig &cfg);

/**
 * @brief Flatten a tree into verify ids / position ids / additive mask
 *        (== compile_ddtree_tree). Writes into caller buffers.
 * @param rootTokenId      token at tree root (block_output_ids[:, 0])
 * @param start            absolute position of the root
 * @param pastLength       length of the already-cached prefix
 * @param tree             output of buildTree
 * @param cfg              config; uses cfg.maskFillValue
 * @param verifyInputIds   [currentLength] out
 * @param verifyPositionIds[currentLength] out
 * @param attentionMask    [currentLength, pastLength+currentLength] out
 * @param attnMaskRowStride elements between consecutive mask rows
 *                          (>= pastLength+currentLength)
 */
CompiledTree compile(int32_t rootTokenId, int start, int pastLength,
                     const DDTreeStructure &tree, const DDTreeConfig &cfg,
                     int32_t *verifyInputIds, int32_t *verifyPositionIds,
                     float *attentionMask, int attnMaskRowStride);

/**
 * @brief Walk the longest accepted root->leaf path (== follow_verified_tree).
 * @param childMaps tree.childMaps
 * @param posterior [currentLength] per-node sampled token id
 */
Accepted followVerified(
  const std::vector<std::unordered_map<int32_t, int32_t>> &childMaps,
  const int32_t *posterior);

} // namespace ddtree
} // namespace nntrainer

#endif // __NNTRAINER_DDTREE_H__

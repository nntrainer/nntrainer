// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @brief  Sliding-window attention-mask stage for the DDTree verify pass
 * @file   ddtree_sliding.h
 * @date   05 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef __NNTRAINER_DDTREE_SLIDING_H__
#define __NNTRAINER_DDTREE_SLIDING_H__

#include <cstdint>

#include <ddtree_types.h>

namespace nntrainer {
namespace ddtree {

/**
 * @brief Build full/sliding additive masks for a tree verify pass.
 *        Operates on contiguous row-major [currentLength, kvLength] buffers.
 * @param attentionMask     full mask, [currentLength, kvLength] (from compile)
 * @param verifyPositionIds [currentLength] absolute query positions
 * @param currentLength     number of tree nodes (rows)
 * @param kvLength          pastLength + currentLength (mask columns)
 * @param slidingWindow     model sliding_window (<=0 => full==sliding)
 * @param hasSlidingLayers  true iff model layer_types contains
 * "sliding_attention"
 * @param cfg               uses cfg.maskFillValue
 * @param slidingBuffer     [currentLength, kvLength] scratch for sliding
 * variant
 */
SlidingMasks makeSlidingMasks(float *attentionMask,
                              const int32_t *verifyPositionIds,
                              int currentLength, int kvLength,
                              int slidingWindow, bool hasSlidingLayers,
                              const DDTreeConfig &cfg, float *slidingBuffer);

/**
 * @brief Sliding-window visibility as a plain 0/1 bitmap (format-neutral).
 *
 * Equivalent to thresholding makeSlidingMasks()'s sliding output, but emitted
 * directly as 0/1 and independent of cfg.maskFillValue / the fp32 additive
 * layout. Intended for consumers (e.g. a QNN runtime) that build their own
 * integer/gating masks and want the full and sliding masks in the same 0/1 form
 * as @c DDTreeStructure::visibility, without an fp32 additive round-trip.
 *
 * out[i][j] = treeVisible(i,j) AND windowVisible(i,j), where for column j
 * (with past = kvLength - currentLength):
 *   - prefix column (j < past): treeVisible = 1, key position k = j;
 *   - tree column   (j >= past): treeVisible = treeVisibility[i][j-past],
 *     key position k = verifyPositionIds[j-past];
 *   windowVisible = (slidingWindow <= 0) ? 1 : (k <= q && k > q -
 * slidingWindow), with query position q = verifyPositionIds[i].
 *
 * @param treeVisibility    [currentLength*currentLength] row-major 0/1
 * (buildTree)
 * @param verifyPositionIds [currentLength] absolute query positions (compile)
 * @param currentLength     number of tree nodes (rows)
 * @param kvLength          pastLength + currentLength (mask columns)
 * @param slidingWindow     model sliding_window (<=0 => window unrestricted)
 * @param outVisible        [currentLength*kvLength] out, row-major 0/1
 */
void makeSlidingVisibility(const uint8_t *treeVisibility,
                           const int32_t *verifyPositionIds, int currentLength,
                           int kvLength, int slidingWindow,
                           uint8_t *outVisible);

} // namespace ddtree
} // namespace nntrainer

#endif // __NNTRAINER_DDTREE_SLIDING_H__

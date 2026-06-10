// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @brief  Reusable, model-agnostic KV tail reorder for DDTree compaction
 * @file   ddtree_compact.h
 * @date   05 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef __NNTRAINER_DDTREE_COMPACT_H__
#define __NNTRAINER_DDTREE_COMPACT_H__

#include <cstdint>

namespace nntrainer {
namespace ddtree {

/**
 * @brief Reorder cache rows [pastLen, pastLen+tailLen) by keepIndices into
 *        [pastLen, pastLen+keepCount). Gathers into a temporary first, so any
 *        permutation is safe. No-op when keepCount == 0.
 * @param cacheBase      base pointer of the cache buffer (row 0)
 * @param elemSizeBytes  element size (2 for fp16, 4 for fp32)
 * @param seqStrideElems elements between consecutive rows (sequence positions)
 * @param rowElems       valid elements per row to copy
 * @param pastLen        index of the first tail row
 * @param tailLen        number of tail rows (must equal the verified window)
 * @param keepIndices    [keepCount] tail-relative indices to keep, in order
 * @param keepCount      number of kept rows (<= tailLen)
 */
void compactTail(void *cacheBase, int elemSizeBytes, int seqStrideElems,
                 int rowElems, int pastLen, int tailLen,
                 const int32_t *keepIndices, int keepCount);

} // namespace ddtree
} // namespace nntrainer

#endif // __NNTRAINER_DDTREE_COMPACT_H__

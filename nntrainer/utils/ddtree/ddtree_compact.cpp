// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @brief  KV tail reorder implementation for DDTree compaction
 * @file   ddtree_compact.cpp
 * @date   05 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <ddtree_compact.h>

#include <cstring>
#include <vector>

namespace nntrainer {
namespace ddtree {

void compactTail(void *cacheBase, int elemSizeBytes, int seqStrideElems,
                 int rowElems, int pastLen, int tailLen,
                 const int32_t *keepIndices, int keepCount) {
  if (keepCount <= 0 || tailLen <= 0)
    return;

  auto *base = static_cast<uint8_t *>(cacheBase);
  const size_t rowBytes = static_cast<size_t>(rowElems) * elemSizeBytes;
  const size_t strideBytes =
    static_cast<size_t>(seqStrideElems) * elemSizeBytes;

  // Gather kept rows into a packed temporary (index_select semantics).
  std::vector<uint8_t> tmp(static_cast<size_t>(keepCount) * rowBytes);
  for (int d = 0; d < keepCount; ++d) {
    const int src = keepIndices[d];
    const uint8_t *srcRow =
      base + (static_cast<size_t>(pastLen) + src) * strideBytes;
    std::memcpy(tmp.data() + static_cast<size_t>(d) * rowBytes, srcRow,
                rowBytes);
  }
  // Write back into [pastLen, pastLen+keepCount).
  for (int d = 0; d < keepCount; ++d) {
    uint8_t *dstRow = base + (static_cast<size_t>(pastLen) + d) * strideBytes;
    std::memcpy(dstRow, tmp.data() + static_cast<size_t>(d) * rowBytes,
                rowBytes);
  }
}

} // namespace ddtree
} // namespace nntrainer

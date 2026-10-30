// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_arena_size.h
 * @brief Overflow-safe arena tail sizing for bounded HTP mappings.
 * @author NNTrainer contributors
 * @bug No known bugs
 */
#ifndef NNTR_HTP_ARENA_SIZE_H
#define NNTR_HTP_ARENA_SIZE_H

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace nntrainer {

/** Keep the established 64 MiB default; 16 and 4 MiB are opt-in tail
 * granularities. Exact parsing avoids accidental oversized chunks. */
inline size_t htpArenaGrainBytes(const char *value) {
  if (!value || std::strcmp(value, "64") == 0)
    return size_t(64) << 20;
  if (std::strcmp(value, "16") == 0)
    return size_t(16) << 20;
  if (std::strcmp(value, "4") == 0)
    return size_t(4) << 20;
  throw std::invalid_argument("NNTR_HTP_ARENA_GRAIN_MB must be 4, 16 or 64");
}

/** Round a new chunk without changing the existing mapping ceiling.
 * Zero means the request cannot hold this weight or would overflow. */
inline size_t htpArenaChunkBytes(size_t bytes, size_t want, size_t cap,
                                 size_t grain) {
  if (!grain || (grain & (grain - 1u)) || bytes > cap)
    return 0;
  const size_t needed = std::max(want, bytes);
  if (needed > std::numeric_limits<size_t>::max() - (grain - 1u))
    return 0;
  const size_t rounded = (needed + grain - 1u) & ~(grain - 1u);
  const size_t size = std::min(std::max(rounded, grain), cap);
  return size >= bytes ? size : 0;
}

} // namespace nntrainer
#endif // NNTR_HTP_ARENA_SIZE_H

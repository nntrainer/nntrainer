// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qnn_row_readahead.h
 * @date   06 September 2026
 * @brief  Batched page-cache warm-up for token-indexed sidecar tables
 * @bug    No known bugs except for NYI items
 *
 * Issues only the requested token rows, in prompt order, under a byte cap.
 */

#ifndef __QNN_ROW_READAHEAD_H__
#define __QNN_ROW_READAHEAD_H__

#include <cstddef>

namespace causallm {

/**
 * @brief What one batch actually issued, for reporting.
 */
struct RowReadaheadStats {
  size_t rows = 0;  /**< distinct rows issued */
  size_t bytes = 0; /**< sum of row_bytes over those rows */
  size_t pages = 0; /**< 4 KiB pages spanned; rows are unaligned, so this is an
                         upper bound (adjacent rows may share a page) */
  bool capped = false; /**< the batch was skipped: over the byte cap */
};

/**
 * @brief Is sidecar readahead enabled? `NNTR_PLE_PREFILL_RA=0` opts out.
 *
 * Default ON, read once per process. The two issue functions below do NOT
 * consult it: the caller does, so that a model which already gates part of its
 * prefetching on an older switch can keep that arm's behaviour bit-identical
 * while putting only the new batch behind this one.
 */
bool row_readahead_enabled();

/**
 * @brief Byte cap for a single batch; `NNTR_PLE_PREFILL_RA_CAP_MIB` overrides.
 *
 * Default 32 MiB. A batch whose rows would exceed it is skipped.
 */
size_t row_readahead_cap_bytes();

/**
 * @brief readahead(2) the rows named by `tokens` on an open sidecar fd.
 *
 * @param fd          sidecar file descriptor (kept open next to the mapping)
 * @param tokens      token ids, in the order the caller will read them
 * @param count       number of token ids
 * @param row_bytes   bytes per row
 * @param row_count   rows in the table; ids outside [0, row_count) are dropped
 * @param base_offset byte offset of row 0 within the file
 * @param tag         short label for the `[RA]` trace line
 *
 * Issued in prompt order and deduplicated through a bitmap rather than by
 * sorting: the first chunk's rows are the ones the dequant loop wants first,
 * and sorting would queue them behind rows no chunk needs for another 40 ms of
 * graph execute. Safe on resident rows; safe to call from a worker thread.
 */
RowReadaheadStats row_readahead_file(int fd, const int *tokens, size_t count,
                                     size_t row_bytes, size_t row_count,
                                     size_t base_offset, const char *tag);

/**
 * @brief madvise(MADV_WILLNEED) the same rows of an mmap'd table.
 *
 * For tables held only as a mapping (no fd retained). Weaker than
 * readahead(2) on FUSE-backed volumes and equivalent on ext4, so prefer
 * row_readahead_file() whenever an fd is available. Ranges are page-aligned
 * outward, which is why the reported page count is an upper bound.
 */
RowReadaheadStats row_readahead_map(const void *base, size_t map_size,
                                    const int *tokens, size_t count,
                                    size_t row_bytes, size_t row_count,
                                    size_t base_offset, const char *tag);

} // namespace causallm

#endif /* __QNN_ROW_READAHEAD_H__ */

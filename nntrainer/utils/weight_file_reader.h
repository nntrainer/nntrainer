// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 * @file weight_file_reader.h
 * @brief Bounded file reads for accelerator weights, including unaligned direct
 * IO.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NNTRAINER_WEIGHT_FILE_READER_H__
#define __NNTRAINER_WEIGHT_FILE_READER_H__

#include <cstddef>
#include <cstdint>

namespace nntrainer {
/** -1 preserves the existing loader; 0 bypasses the page cache; 1 bounds it. */
int weightPageCachePolicy();

/** A file descriptor source never owns the descriptor or changes its offset. */
struct FileReadSource {
  int fd;
};

/** Open weights with the selected cache policy, throwing on unsupported IO. */
int openWeightFile(const char *path);

/** Read exactly a range without changing the fd offset. Returns an errno value.
 * Direct IO handles unaligned ranges with at most 2 MiB of thread-local
 * scratch; it never silently falls back to buffered IO. */
int weightPread(int fd, void *dst, size_t bytes, uint64_t offset);

/** Direct IO scratch bounds, also used to select one-request expert reads. */
constexpr size_t WEIGHT_DIRECT_ALIGNMENT = 4096;
constexpr size_t WEIGHT_DIRECT_WINDOW_BYTES = size_t(2) << 20;

/** One destination for an adjacent part of a file range. */
struct WeightReadSlice {
  void *data;
  size_t bytes;
};

/** Optional per-thread direct-read timing. No payload is retained. A caller
 * owns this object and restores the previous pointer before its lifetime ends.
 * Nanosecond counters separate storage waits from bounce-buffer copies. */
struct WeightReadTiming {
  uint64_t io_ns = 0, copy_ns = 0;
  uint64_t calls = 0, requested_bytes = 0;
};
WeightReadTiming *weightSetReadTiming(WeightReadTiming *timing);

/** Scatter one contiguous file range into one to four destinations. Empty
 * slices are allowed. The same bounded scratch is reused across direct reads;
 * metadata can go to CPU memory without reading back an uncached DMA mapping.
 * Returns errno, including EOVERFLOW, EINVAL or ENODATA for invalid ranges. */
int weightPreadScatter(int fd, const WeightReadSlice *slices, size_t count,
                       uint64_t offset);

/** Read adjacent file ranges into separate code and metadata destinations.
 * Combines their IO without retaining another copy of the weight. */
int weightPreadSplit(int fd, void *dst, size_t bytes, void *tail,
                     size_t tail_bytes, uint64_t offset);
} // namespace nntrainer
#endif

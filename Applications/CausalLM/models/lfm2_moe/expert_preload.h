// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 * @file expert_preload.h
 * @brief Independent load-time budget for the shared expert cache.
 * @author Codex <noreply@openai.com>
 * @bug No known bugs
 */
#ifndef __CAUSALLM_EXPERT_PRELOAD_H__
#define __CAUSALLM_EXPERT_PRELOAD_H__

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace causallm {
/**
 * @brief Total experts to read at startup, across all layers.
 *
 * Unset defaults to 128, capped at the cache capacity. Zero reserves the
 * arena but leaves all expert weight reads to prefill. An explicit value
 * at least as large as the capacity restores a full-cache preload.
 */
inline size_t expertPreloadLimit(size_t capacity, const char *value) {
  if (!value)
    return std::min(capacity, size_t{128});
  if (*value < '0' || *value > '9')
    throw std::invalid_argument("NNTR_MOE_PRELOAD_EXPERTS must be nonnegative");
  char *end = nullptr;
  errno = 0;
  const auto count = std::strtoull(value, &end, 10);
  if (errno || *end || count > std::numeric_limits<size_t>::max())
    throw std::invalid_argument("Invalid NNTR_MOE_PRELOAD_EXPERTS");
  return std::min(capacity, static_cast<size_t>(count));
}
} // namespace causallm
#endif

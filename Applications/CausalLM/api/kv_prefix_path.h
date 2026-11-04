// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    kv_prefix_path.h
 * @brief   Where a JSON request's "kv_prefix" keeps its KV dump, and how a dump
 *          file's token count is read back. Pure functions, no engine types.
 * @author  Joonseok Oh <jrock.oh@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * The dump is `<model_dir>/kv_prefix/<16 hex digits>.kv`, one file per static
 * prefix text. A caller outside this library (the AI Studio app) derives the
 * same path from the same bytes to check or delete a dump, so the name is a
 * contract: the hash below must not change, and it must match that caller byte
 * for byte.
 */
#ifndef __QUICK_DOT_AI_KV_PREFIX_PATH_H__
#define __QUICK_DOT_AI_KV_PREFIX_PATH_H__

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace quick_dot_ai {

/**
 * @brief FNV-1a, 64 bit, of the prefix text's bytes.
 * @note The offset basis is 1469598103934665603, which is the standard FNV
 *       basis (14695981039346656037) with its last digit dropped. That is what
 *       every existing dump and the app's own path computation use, so it is
 *       kept as is; "fixing" it would orphan every saved prefix.
 */
inline uint64_t kv_prefix_hash(const std::string &prefix) {
  uint64_t fnv = 1469598103934665603ULL;
  for (unsigned char c : prefix)
    fnv = (fnv ^ c) * 1099511628211ULL;
  return fnv;
}

/** @brief "<hash as %016x>.kv" */
inline std::string kv_prefix_file_name(const std::string &prefix) {
  char name[24];
  std::snprintf(name, sizeof(name), "%016llx.kv",
                static_cast<unsigned long long>(kv_prefix_hash(prefix)));
  return name;
}

/** @brief `<model_dir>/kv_prefix/<hash>.kv` */
inline std::string kv_prefix_path(const std::string &model_dir,
                                  const std::string &prefix) {
  return (std::filesystem::path(model_dir) / "kv_prefix" /
          kv_prefix_file_name(prefix))
    .string();
}

/**
 * @brief The token count out of KVCacheManager::describeFile()'s one-line
 *        description (" ... tokens=N layers=..."), or -1 when it has none (a
 *        file that cannot be opened or has no header).
 */
inline long long kv_describe_tokens(const std::string &description) {
  static const std::string key = " tokens=";
  const size_t at = description.find(key);
  if (at == std::string::npos)
    return -1;
  const char *begin = description.c_str() + at + key.size();
  if (*begin < '0' || *begin > '9')
    return -1;
  char *end = nullptr;
  const unsigned long long v = std::strtoull(begin, &end, 10);
  if (end == begin || v > 0x7fffffffULL)
    return -1;
  return static_cast<long long>(v);
}

} // namespace quick_dot_ai

#endif // __QUICK_DOT_AI_KV_PREFIX_PATH_H__

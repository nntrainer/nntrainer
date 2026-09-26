// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    env_compat.h
 * @date    09 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   POSIX setenv() shim for MSVC. The GPU contexts apply their
 *          HW-optimal env defaults with setenv(name, value, overwrite=0)
 *          ("explicit env always wins"); MSVC only has _putenv_s, which
 *          unconditionally overwrites, so the overwrite=0 semantics are
 *          reproduced with a getenv() check. Include this in any TU that
 *          calls setenv() and must also build on Windows. No-op elsewhere.
 */

#ifndef __NNTR_ENV_COMPAT_H__
#define __NNTR_ENV_COMPAT_H__

#include <cstdlib>

/**
 * @brief VALUE-checked env truthiness: set AND not starting with '0'.
 *        The GPU contexts auto-inject their tuned defaults with
 *        setenv(..., overwrite=0), so a consumer that only checks presence
 *        can never be turned off with =0 -- three separate debugging
 *        sessions hit that trap (DEV_ACT, KV_UVM, ELTWISE). Every consumer
 *        of an auto-injected flag must use this instead of a raw
 *        getenv()!=nullptr.
 */
static inline bool nntr_env_on(const char *name) {
  const char *e = std::getenv(name);
  return e != nullptr && e[0] != '0';
}

#if defined(_WIN32)
#include <cstdlib>

static inline int setenv(const char *name, const char *value, int overwrite) {
  if (!overwrite && std::getenv(name) != nullptr)
    return 0;
  return _putenv_s(name, value);
}
#endif

#endif // __NNTR_ENV_COMPAT_H__

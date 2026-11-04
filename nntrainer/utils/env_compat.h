// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    env_compat.h
 * @date    09 Jul 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Env helpers shared by every platform -- nntr_engine_env()
 *          (NNTR_ENGINE lowercased) and nntr_env_on() (value-checked
 *          on/off) -- plus a POSIX setenv() shim for MSVC. The GPU contexts
 *          apply their HW-optimal env defaults with setenv(name, value,
 *          overwrite=0) ("explicit env always wins"); MSVC only has
 *          _putenv_s, which unconditionally overwrites, so the overwrite=0
 *          semantics are reproduced with a getenv() check. Include this in
 *          any TU that calls setenv() and must also build on Windows; the
 *          shim is compiled only on _WIN32.
 */

#ifndef __NNTR_ENV_COMPAT_H__
#define __NNTR_ENV_COMPAT_H__

#include <cctype>
#include <cstdlib>
#include <string>

namespace nntrainer {

/**
 * @brief Outcome of requestEngine().
 */
enum class EngineRequestResult {
  APPLIED,   /**< recorded; every later engine read sees it */
  UNCHANGED, /**< the engine was already read and is the one asked for */
  TOO_LATE   /**< the engine was already read and is a different one */
};

/**
 * @brief Ask for a compute engine ("cpu", "gpu", "cuda") without touching the
 *        process environment.
 * @details The library API must not mutate the process environment, but a
 *          caller still has to be able to say which engine a load runs on. A
 *          request made here is what nntr_engine_env() answers, ahead of
 *          NNTR_ENGINE, so the Engine's context bring-up and every model's
 *          engine property follow it.
 *
 *          The engine is a process-wide, first-reader-wins choice: the Engine
 *          and causallm_engine() latch their answer the first time they read
 *          it. A request is therefore only honoured before that first read;
 *          after it the call reports whether the engine already is the
 *          requested one. A caller that wants an exported NNTR_ENGINE to keep
 *          priority must check the environment itself before calling.
 * @param engine "cpu" or "gpu", case-insensitive. ("cuda" is not supported
 *        here: CudaContext::initialize() gates on the environment itself.)
 * @param effective Optional: the engine in force after the call.
 * @note Defined in utils/engine_request.cpp (libnntrainer).
 */
EngineRequestResult requestEngine(const std::string &engine,
                                  std::string *effective = nullptr);

/**
 * @brief The engine request, "" when none was made. Marks the engine as read,
 *        which closes the window for requestEngine(). Reached only through
 *        nntr_engine_env().
 */
std::string engineRequestForRead();

} // namespace nntrainer

/**
 * @brief NNTR_ENGINE as every reader has to see it: lowercased, "" if unset.
 *        An engine requested through nntrainer::requestEngine() takes the
 *        place of the variable.
 *        The Engine registers contexts under lowercased names, so a reader
 *        that compares the raw value (NNTR_ENGINE=CUDA against "cuda") ends
 *        up disagreeing with the context that was brought up: the context
 *        exists, but the reader's own gate declines every path it guards.
 *        Every NNTR_ENGINE comparison goes through this.
 */
static inline std::string nntr_engine_env() {
  std::string s = nntrainer::engineRequestForRead();
  if (s.empty()) {
    const char *e = std::getenv("NNTR_ENGINE");
    s = e != nullptr ? std::string(e) : std::string();
  }
  for (auto &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

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

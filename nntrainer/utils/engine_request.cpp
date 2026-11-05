// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   engine_request.cpp
 * @date   29 October 2026
 * @brief  Process-wide compute-engine request that does not use the
 *         environment. See env_compat.h.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Joonseok Oh <jrock.oh@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "env_compat.h"

#include <cctype>
#include <cstdlib>
#include <mutex>
#include <string>

namespace nntrainer {

namespace {

struct EngineRequestState {
  std::mutex mtx;
  /** Lowercased request, "" when none was made. */
  std::string request;
  /** Set by the first nntr_engine_env(): from then on the engine is latched. */
  bool latched = false;
};

EngineRequestState &state() {
  static EngineRequestState s;
  return s;
}

std::string lower(std::string s) {
  for (auto &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

/** What nntr_engine_env() resolves to with no request in place: NNTR_ENGINE,
 *  else the build default that causallm_engine() and Engine pick. */
std::string environmentOrDefaultEngine() {
  const char *e = std::getenv("NNTR_ENGINE");
  std::string s = lower(e != nullptr ? std::string(e) : std::string());
  if (!s.empty())
    return s;
#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  return "gpu";
#else
  return "cpu";
#endif
}

} // namespace

std::string engineRequestForRead() {
  auto &s = state();
  std::lock_guard<std::mutex> lk(s.mtx);
  s.latched = true;
  return s.request;
}

EngineRequestResult requestEngine(const std::string &engine,
                                  std::string *effective) {
  auto &s = state();
  const std::string want = lower(engine);
  std::lock_guard<std::mutex> lk(s.mtx);
  if (!s.latched) {
    s.request = want;
    if (effective != nullptr)
      *effective = want;
    return EngineRequestResult::APPLIED;
  }
  const std::string now =
    s.request.empty() ? environmentOrDefaultEngine() : s.request;
  if (effective != nullptr)
    *effective = now;
  return now == want ? EngineRequestResult::UNCHANGED
                     : EngineRequestResult::TOO_LATE;
}

} // namespace nntrainer

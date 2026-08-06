// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   factory.cpp
 * @date   13 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Out-of-line definition of causallm::Factory::Instance().
 *
 * @note   This single definition lives in libcausallm.so so that every
 *         consumer .so (libquick_dot_ai_api, the optional libqai_ext_model
 *         model plugin, ...) shares ONE Factory instance. See the rationale on
 *         Factory::Instance() in factory.h; mirrors Engine::Global() in
 *         nntrainer's engine.cpp.
 */

#include "factory.h"

namespace causallm {

Factory &Factory::Instance() {
  // Single definition in libcausallm.so → one Factory (one creators map)
  // shared by every consumer .so (see declaration note in factory.h).
  static Factory factory;
  return factory;
}

} // namespace causallm

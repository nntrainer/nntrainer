// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file singleton_probe_a.cpp
 * @brief First shared-library singleton regression probe
 * @author jayden0701 <jrock.oh@samsung.com>
 * @bug No known bugs
 */

#include "singleton_probe_api.h"

#include <profiler.h>
#include <thread_manager.h>

/** @brief Return probe A's Profiler singleton address. */
const void *nntr_singleton_probe_a_profiler() noexcept {
  return &nntrainer::profile::Profiler::Global();
}

/** @brief Return probe A's ThreadManager singleton address. */
const void *nntr_singleton_probe_a_thread_manager() noexcept {
  return &nntrainer::ThreadManager::Global();
}

/** @brief Return probe A's base ThreadManager singleton address. */
const void *nntr_singleton_probe_a_thread_manager_base() noexcept {
  return &nntrainer::Singleton<nntrainer::ThreadManager>::Global();
}

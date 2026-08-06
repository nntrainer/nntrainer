// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file unittest_singleton_dso.cpp
 * @brief Verify that owner-bound singletons
 * retain identity across DSOs
 * @author jayden0701 <jrock.oh@samsung.com>
 *
 * @bug No known bugs
 */

#include "singleton_probe_api.h"

#include <array>
#include <iostream>
#include <thread>

#include <profiler.h>
#include <thread_manager.h>

/** @brief Run the cross-DSO singleton identity test. */
int main() {
  std::array<const void *, 6> addresses{};
  std::array<std::thread, 6> callers{
    std::thread([&]() { addresses[0] = nntr_singleton_probe_a_profiler(); }),
    std::thread([&]() { addresses[1] = nntr_singleton_probe_b_profiler(); }),
    std::thread(
      [&]() { addresses[2] = nntr_singleton_probe_a_thread_manager(); }),
    std::thread(
      [&]() { addresses[3] = nntr_singleton_probe_b_thread_manager(); }),
    std::thread(
      [&]() { addresses[4] = nntr_singleton_probe_a_thread_manager_base(); }),
    std::thread(
      [&]() { addresses[5] = nntr_singleton_probe_b_thread_manager_base(); }),
  };

  for (auto &caller : callers) {
    caller.join();
  }

  const auto *main_profiler = &nntrainer::profile::Profiler::Global();
  const auto *main_thread_manager = &nntrainer::ThreadManager::Global();
  const auto *main_thread_manager_base =
    &nntrainer::Singleton<nntrainer::ThreadManager>::Global();

  const bool profiler_is_shared =
    addresses[0] == addresses[1] && addresses[0] == main_profiler;
  const bool thread_manager_is_shared =
    addresses[2] == addresses[3] && addresses[2] == addresses[4] &&
    addresses[2] == addresses[5] && addresses[2] == main_thread_manager &&
    addresses[2] == main_thread_manager_base;

  if (!profiler_is_shared || !thread_manager_is_shared) {
    std::cerr << "singleton identity differs across shared libraries\n";
    return 1;
  }

  return 0;
}

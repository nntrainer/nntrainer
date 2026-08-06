// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file singleton_probe_api.h
 * @brief Cross-DSO singleton regression probe API
 * @author jayden0701 <jrock.oh@samsung.com>
 * @bug No known bugs
 */

#ifndef __NNTRAINER_SINGLETON_PROBE_API_H__
#define __NNTRAINER_SINGLETON_PROBE_API_H__

#if defined(_WIN32)
#if defined(NNTR_SINGLETON_PROBE_A_EXPORTS)
#define NNTR_SINGLETON_PROBE_A_API __declspec(dllexport)
#else
#define NNTR_SINGLETON_PROBE_A_API __declspec(dllimport)
#endif
#if defined(NNTR_SINGLETON_PROBE_B_EXPORTS)
#define NNTR_SINGLETON_PROBE_B_API __declspec(dllexport)
#else
#define NNTR_SINGLETON_PROBE_B_API __declspec(dllimport)
#endif
#else
#define NNTR_SINGLETON_PROBE_A_API __attribute__((visibility("default")))
#define NNTR_SINGLETON_PROBE_B_API __attribute__((visibility("default")))
#endif

extern "C" {

/** @brief Return probe A's Profiler singleton address. */
NNTR_SINGLETON_PROBE_A_API const void *
nntr_singleton_probe_a_profiler() noexcept;
/** @brief Return probe A's ThreadManager singleton address. */
NNTR_SINGLETON_PROBE_A_API const void *
nntr_singleton_probe_a_thread_manager() noexcept;
/** @brief Return probe A's base ThreadManager singleton address. */
NNTR_SINGLETON_PROBE_A_API const void *
nntr_singleton_probe_a_thread_manager_base() noexcept;

/** @brief Return probe B's Profiler singleton address. */
NNTR_SINGLETON_PROBE_B_API const void *
nntr_singleton_probe_b_profiler() noexcept;
/** @brief Return probe B's ThreadManager singleton address. */
NNTR_SINGLETON_PROBE_B_API const void *
nntr_singleton_probe_b_thread_manager() noexcept;
/** @brief Return probe B's base ThreadManager singleton address. */
NNTR_SINGLETON_PROBE_B_API const void *
nntr_singleton_probe_b_thread_manager_base() noexcept;
}

#endif // __NNTRAINER_SINGLETON_PROBE_API_H__

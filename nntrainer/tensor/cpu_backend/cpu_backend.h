// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   cpu_backend.h
 * @date   05 Feb 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Computational backend for CPU considering architecture dependency
 *
 */

#ifndef __CPU_BACKEND_H__
#define __CPU_BACKEND_H__
#ifdef __cplusplus
#if defined(__aarch64__) || defined(__ARM_ARCH_7A__) ||                        \
  defined(__ANDROID__) || defined(__arm__) || defined(_M_ARM) ||               \
  defined(_M_ARM64)
#include <arm_compute_backend.h>
#elif defined(__x86_64__) || defined(__i586__) || defined(_M_X64) ||           \
  defined(_M_IX86)
#include <x86_compute_backend.h>
#else
#include <fallback.h>
#endif

// Expose the ComputeOps dispatch table (and init_backend declaration) to any
// consumer that already includes cpu_backend.h.
#include <compute_ops.h>

#include <common.h>
#include <cstdint>
#include <tensor_dim.h>

/**
 * NOTE TO MAINTAINERS:
 * Do NOT declare CPU op functions in this file. This header only selects and
 * includes the correct arch-specific backend header above.
 *
 * Declare every CPU op prototype inside `namespace nntrainer` in ALL THREE
 * backend headers so each platform still builds:
 *   - arm/arm_compute_backend.h
 *   - x86/x86_compute_backend.h
 *   - fallback/fallback.h   (add an NYI-throw stub in fallback.cpp if the op is
 *                            not implemented for the fallback backend)
 *
 * A prototype added here would land in the global namespace (::foo), not
 * nntrainer::foo, so consumers calling nntrainer::foo() would never bind to it
 * -- a dead declaration.
 */

#endif /* __cplusplus */
#endif /* __CPU_BACKEND_H__ */

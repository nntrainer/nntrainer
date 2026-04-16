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

#include <cstdint>
#include <tensor_dim.h>

// All backend function declarations are provided by the architecture-specific
// headers included above (arm_compute_backend.h, x86_compute_backend.h, or
// fallback.h). One of these three is always included via the #if/#elif/#else
// chain, so no additional declarations are needed here.

#endif /* __cplusplus */
#endif /* __CPU_BACKEND_H__ */

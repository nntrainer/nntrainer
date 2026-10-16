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
 *          (NNTR_ENGINE lowercased), nntr_env_on() / nntr_env_on_or()
 *          (value-checked on/off) and nntr_deterministic() (the one
 *          NNTR_DETERMINISTIC reader) -- plus a POSIX setenv() shim for
 *          MSVC. The GPU contexts apply their HW-optimal env defaults with
 *          setenv(name, value, overwrite=0) ("explicit env always wins");
 *          MSVC only has _putenv_s, which unconditionally overwrites, so the
 *          overwrite=0 semantics are reproduced with a getenv() check.
 *          Include this in any TU that calls setenv() and must also build on
 *          Windows; the shim is compiled only on _WIN32.
 */

#ifndef __NNTR_ENV_COMPAT_H__
#define __NNTR_ENV_COMPAT_H__

#include <cctype>
#include <cstdlib>
#include <string>

/**
 * @brief NNTR_ENGINE as every reader has to see it: lowercased, "" if unset.
 *        The Engine registers contexts under lowercased names, so a reader
 *        that compares the raw value (NNTR_ENGINE=CUDA against "cuda") ends
 *        up disagreeing with the context that was brought up: the context
 *        exists, but the reader's own gate declines every path it guards.
 *        Every NNTR_ENGINE comparison goes through this.
 */
static inline std::string nntr_engine_env() {
  const char *e = std::getenv("NNTR_ENGINE");
  std::string s = e != nullptr ? std::string(e) : std::string();
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

/**
 * @brief nntr_env_on() with a default for an unset (or empty) variable.
 *        Set, the variable is value-checked exactly as nntr_env_on() does, so
 *        an explicit =0 is always the opt-out of a default that is on.
 */
static inline bool nntr_env_on_or(const char *name, bool dflt) {
  const char *e = std::getenv(name);
  if (e == nullptr || e[0] == '\0')
    return dflt;
  return e[0] != '0';
}

/**
 * @brief The NNTR_DETERMINISTIC default when the variable is unset: the one
 *        place it is defined.
 * @details On an Android OpenCL build running the gpu engine (NNTR_ENGINE is
 * neither cpu nor cuda) the default is ON: the mirror-free reproducible arm
 * with the Adreno flash decode and the GPU prefill attention
 * (NNTR_ADRENO_FLASH_DECODE / NNTR_ADRENO_PREFILL_ATTN, whose Android
 * defaults follow this answer) is the measured Adreno default -- on the
 * Gemma4-26B MoE 1K cell it is faster in prefill and decode than the image
 * arm and keeps the prefill attention off the CPU. Every other build and
 * engine (x86 / Intel, CUDA, the cpu engine) keeps OFF, so their defaults
 * and goldens do not move. NNTR_DETERMINISTIC=0 is the opt-out on Android
 * (the image arm), =1 the opt-in elsewhere.
 */
static inline bool nntr_deterministic_default_on() {
#if defined(__ANDROID__) && defined(ENABLE_OPENCL)
  const std::string eng = nntr_engine_env();
  return eng != "cpu" && eng != "cuda";
#else
  return false;
#endif
}

/**
 * @brief Whether the reproducible (deterministic) arm is selected:
 *        NNTR_DETERMINISTIC value-checked, nntr_deterministic_default_on()
 *        when it is unset. Every OpenCL / CausalLM reader of the variable
 *        goes through this, so the arm the layers pick, the reduction order
 *        the kernels use and the FC flush policy cannot disagree.
 */
static inline bool nntr_deterministic() {
  return nntr_env_on_or("NNTR_DETERMINISTIC", nntr_deterministic_default_on());
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

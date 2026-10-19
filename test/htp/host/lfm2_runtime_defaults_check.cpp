// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_runtime_defaults_check.cpp
 * @brief Check CLI preset selection, opt-out and explicit override isolation.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#include "lfm2_runtime_defaults.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

/** @brief Set an explicit CLI override in this isolated test process. */
static void setEnvironment(const char *name, const char *value) {
#if defined(_WIN32)
  assert(_putenv_s(name, value) == 0);
#else
  assert(setenv(name, value, 1) == 0);
#endif
}

/** @brief Remove a preset variable inherited from the test launcher. */
static void unsetEnvironment(const char *name) {
#if defined(_WIN32)
  assert(_putenv_s(name, "") == 0);
#else
  assert(unsetenv(name) == 0);
#endif
}

/** @brief Exercise negative model selection before activating the preset. */
int main() {
  for (const char *name : {"NNTR_LFM2_DEFAULT_PRESET", "NNTR_MOE_PREFETCH_TOP",
                           "NNTR_LFM2_CONV_RETAIN_OUT"})
    unsetEnvironment(name);
  setEnvironment("NNTR_NUM_THREADS", "3");
  setEnvironment("NNTR_MOE_CACHE_GROW", "0");
  assert(!causallm::configureLfm2HtpDefaults("gemma4", "sr15_v2", "htp_fused"));
  assert(!causallm::configureLfm2HtpDefaults("lfm2_moe", "htp_fused_v1",
                                             "htp_fused"));
  assert(!causallm::configureLfm2HtpDefaults("lfm2_moe", "sr15_v2", "cpu"));
  assert(std::getenv("NNTR_MOE_PREFETCH_TOP") == nullptr);
  assert(std::getenv("NNTR_LFM2_CONV_RETAIN_OUT") == nullptr);
  setEnvironment("NNTR_LFM2_DEFAULT_PRESET", "0");
  assert(
    !causallm::configureLfm2HtpDefaults("lfm2_moe", "sr15_v2", "htp_fused"));
  assert(std::getenv("NNTR_MOE_PREFETCH_TOP") == nullptr);
  unsetEnvironment("NNTR_LFM2_DEFAULT_PRESET");
  assert(
    causallm::configureLfm2HtpDefaults("lfm2_moe", "sr15_v2", "htp_fused"));
  assert(std::string(std::getenv("NNTR_NUM_THREADS")) == "3");
  assert(std::string(std::getenv("NNTR_MOE_CACHE_GROW")) == "0");
  assert(std::string(std::getenv("NNTR_MOE_PREFETCH_TOP")) == "3");
  assert(std::string(std::getenv("NNTR_LFM2_CONV_RETAIN_OUT")) == "1");
  setEnvironment("NNTR_MOE_PREFETCH_TOP", "2");
  assert(
    causallm::configureLfm2HtpDefaults("lfm2_moe", "sr15_v2", "htp_fused"));
  assert(std::string(std::getenv("NNTR_MOE_PREFETCH_TOP")) == "2");
  std::puts("LFM2 CLI PRESET SELECTION / OPT-OUT / EXPLICIT OVERRIDES OK");
}

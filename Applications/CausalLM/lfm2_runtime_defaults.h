// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_runtime_defaults.h
 * @brief Process-local defaults for the standalone SR15 HTP executable.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef __LFM2_RUNTIME_DEFAULTS_H__
#define __LFM2_RUNTIME_DEFAULTS_H__

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace causallm {

/**
 * @brief Apply the measured SR15 preset before creating the CLI model.
 * @param model_type Architecture from config.json.
 * @param weight_format LFM2 weight layout from nntr_config.json.
 * @param execution_profile Selected LFM2 execution path.
 * @return Whether this invocation selected the SR15 preset.
 * @note Call only in the standalone executable, before model/thread creation.
 * Explicit environment values, including zero, always override defaults.
 * The library and multi-model API must not mutate process environment here.
 */
inline bool configureLfm2HtpDefaults(const std::string &model_type,
                                     const std::string &weight_format,
                                     const std::string &execution_profile) {
  if (model_type != "lfm2_moe" || weight_format != "sr15_v2" ||
      execution_profile != "htp_fused")
    return false;
  const char *preset = std::getenv("NNTR_LFM2_DEFAULT_PRESET");
  if (preset && std::string(preset) == "0")
    return false;
  const struct {
    const char *name;
    const char *value;
  } defaults[] = {
    {"NNTR_NUM_THREADS", "8"},
    {"NNTR_HTP_PAGE_CACHE", "0"},
    {"NNTR_LOAD_PREFETCH", "0"},
    {"NNTR_MOE_CACHE_GROW", "1"},
    {"NNTR_MOE_PRELOAD_EXPERTS", "128"},
    {"NNTR_HTP_TL2_STABLE_SLOTS", "1"},
    {"NNTR_LFM2_PREFILL_EARLY", "1"},
    {"NNTR_LFM2_DECODE_PAIR_READ", "1"},
    {"NNTR_LFM2_DECODE_DEMAND_HANDOFF", "1"},
    {"NNTR_HTP_PREDICT", "2"},
    {"NNTR_MOE_PREFETCH_RESIDUAL", "0.5"},
    {"NNTR_LFM2_PREFILL_PRUNE", "0"},
    {"NNTR_HTP_FC_WHOLE_PREFILL", "1"},
    {"NNTR_LFM2_PARALLEL_NORM", "1"},
    {"NNTR_LFM2_PARALLEL_ADD", "1"},
    {"NNTR_LFM2_COMPACT_RECENCY", "1"},
    {"NNTR_LFM2_CONV_WHOLE_PREFILL", "1"},
    {"NNTR_LFM2_CONV_CHANNEL_TILE", "1"},
    {"NNTR_LFM2_PREFILL_VIEWS", "1"},
    {"NNTR_LFM2_CONV_STATE_ONLY", "1"},
    {"NNTR_LFM2_PARALLEL_STAGE", "1"},
    {"NNTR_LFM2_CONV_RETAIN_OUT", "1"},
    {"NNTR_LFM2_CONV_STRIPE_ROWS", "8"},
    {"NNTR_LFM2_ATTN_ION_HANDOFF", "1"},
    {"NNTR_MOE_PREFETCH_SLOTS", "6"},
    {"NNTR_HTP_ARENA_GRAIN_MB", "4"},
    {"NNTR_HTP_PURGE_PREFILL", "1"},
    {"NNTR_HTP_RELEASE_PREFILL_READERS", "1"},
    {"NNTR_LFM2_PREFETCH_GAMMA", "1"},
    {"NNTR_MOE_PREFETCH_TOP", "3"},
  };
  for (const auto &setting : defaults) {
    if (std::getenv(setting.name) != nullptr)
      continue;
#if defined(_WIN32)
    const int status = _putenv_s(setting.name, setting.value);
#else
    const int status = setenv(setting.name, setting.value, 0);
#endif
    if (status != 0)
      throw std::runtime_error(std::string("Cannot set LFM2 CLI default: ") +
                               setting.name);
  }
  return true;
}

} // namespace causallm
#endif // __LFM2_RUNTIME_DEFAULTS_H__

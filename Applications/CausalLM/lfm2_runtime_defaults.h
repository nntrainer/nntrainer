// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_runtime_defaults.h
 * @brief Configuration defaults for the standalone SR15 HTP executable.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef __LFM2_RUNTIME_DEFAULTS_H__
#define __LFM2_RUNTIME_DEFAULTS_H__

#include "models/htp_model_config.h"

namespace causallm {

/** Merge the measured SR15 policy into CLI configuration, without changing
 * the environment or the defaults of library/API model handles. */
inline bool configureLfm2HtpDefaults(const std::string &model_type,
                                     const std::string &weight_format,
                                     const std::string &execution_profile,
                                     json &nntr) {
  if (model_type != "lfm2_moe" || weight_format != "sr15_v2" ||
      execution_profile != "htp_fused")
    return false;
  for (const char *section : {"htp", "moe"}) {
    if (nntr.contains(section) && !nntr.at(section).is_object())
      throw std::invalid_argument(std::string(section) + " must be an object");
    if (!nntr.contains(section))
      nntr[section] = json::object();
  }
  bool enabled = true;
  if (nntr["htp"].contains("lfm2_default_preset"))
    detail::readHtpConfigValue(nntr["htp"]["lfm2_default_preset"], enabled,
                               "htp.lfm2_default_preset");
  if (!enabled)
    return false;
  const json defaults = {{"htp",
                          {{"page_cache", 0},
                           {"load_prefetch", false},
                           {"fc_whole_prefill", true},
                           {"lfm2_parallel_norm", true},
                           {"lfm2_parallel_add", true},
                           {"lfm2_conv_whole_prefill", true},
                           {"lfm2_conv_channel_tile", true},
                           {"lfm2_prefill_views", true},
                           {"lfm2_conv_state_only", true},
                           {"lfm2_parallel_stage", true},
                           {"lfm2_conv_retain_out", true},
                           {"lfm2_conv_stripe_rows", 8},
                           {"lfm2_attn_ion_handoff", true},
                           {"arena_grain_mib", 4},
                           {"purge_prefill", true},
                           {"release_prefill_readers", true}}},
                         {"moe",
                          {{"cache_grow", 1},
                           {"preload_experts", 128},
                           {"tl2_stable_slots", true},
                           {"lfm2_prefill_early", true},
                           {"lfm2_decode_pair_read", true},
                           {"lfm2_decode_demand_handoff", true},
                           {"prediction", "arm-direct"},
                           {"prefetch_residual", 0.5},
                           {"lfm2_prefill_prune", false},
                           {"lfm2_compact_recency", true},
                           {"prefetch_slots", 6},
                           {"lfm2_prefetch_gamma", true},
                           {"prefetch_top", 3}}}};
  for (const auto &[section, values] : defaults.items())
    for (const auto &[key, value] : values.items())
      if (!nntr[section].contains(key))
        nntr[section][key] = value;
  if (!nntr.contains("num_threads"))
    nntr["num_threads"] = 8;
  return true;
}

} // namespace causallm
#endif // __LFM2_RUNTIME_DEFAULTS_H__

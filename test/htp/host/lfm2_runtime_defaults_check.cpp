// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_runtime_defaults_check.cpp
 * @brief HTP/MoE JSON defaults, validation and runtime configuration lifetime.
 * @author NNTrainer contributors
 * @bug No known bugs
 */
#include "htp_model_config.h"
#include "lfm2_runtime_defaults.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

using causallm::json;
using causallm::parseHtpRuntimeConfig;

static void setEnvironment(const char *name, const char *value) {
#if defined(_WIN32)
  assert(_putenv_s(name, value) == 0);
#else
  assert(setenv(name, value, 1) == 0);
#endif
}

static void invalid(const json &value) {
  bool rejected = false;
  try {
    (void)parseHtpRuntimeConfig(value);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}

static bool preset(json &value, const std::string &model = "lfm2_moe",
                   const std::string &format = "sr15_v2",
                   const std::string &profile = "htp_fused") {
  return causallm::configureLfm2HtpDefaults(model, format, profile, value);
}

int main() {
  // Old process settings cannot override an independently loaded model.
  setEnvironment("NNTR_HTP_PAGE_CACHE", "invalid");
  setEnvironment("NNTR_MOE_CACHE_GROW", "99");
  setEnvironment("NNTR_MOE_PREFETCH_TOP", "15");
  setEnvironment("NNTR_LFM2_DEFAULT_PRESET", "0");
  setEnvironment("NNTR_NUM_THREADS", "3");
  const auto defaults = parseHtpRuntimeConfig(json::object());
  assert(!defaults.htp.kv_session && !defaults.htp.kv_request_reserve);
  const auto session = parseHtpRuntimeConfig(
    {{"htp", {{"kv_session", true}, {"kv_request_reserve", true}}}});
  assert(session.htp.kv_session && session.htp.kv_request_reserve &&
         !(session == defaults));
  assert(defaults.htp.attn_m1_q_bits == 16);
  const auto q8 = parseHtpRuntimeConfig({{"htp",
                                          {{"attn_m1_fast", true},
                                           {"attn_m1_fast_scope", "global"},
                                           {"attn_m1_q_bits", 8}}}});
  assert(q8.htp.attn_m1_q_bits == 8 && !(q8 == defaults));
  invalid({{"htp", {{"attn_m1_q_bits", 8}}}});
  invalid({{"htp", {{"attn_m1_fast", true}, {"attn_m1_q_bits", 8}}}});
  invalid({{"htp", {{"attn_m1_q_bits", 4}}}});
  assert(defaults.htp.page_cache == -1 && defaults.htp.profile == 0);
  assert(defaults.htp.arena_grain_mib == 64 &&
         defaults.htp.prefill_rows == 512);
  assert(defaults.htp.poll_us == 5000 &&
         defaults.htp.dsp_queue_spin_us == 1000);
  assert(!defaults.htp.fc_lanes_small && !defaults.htp.fc_lanes_large);
  assert(defaults.moe.cache_experts == 16 &&
         defaults.moe.preload_experts == 128);
  assert(!defaults.moe.cache_grow && defaults.moe.lrfu_half_life == 32);
  assert(defaults.moe.prediction == "auto" &&
         defaults.moe.prefill_readers == 6);
  assert(!defaults.moe.decode_prefetch_readers &&
         defaults.moe.prefetch_slots == 32);
  assert(!defaults.moe.prefetch_top && defaults.moe.prefetch_residual == 0.5f);
  assert(defaults.moe.miss_readers == 4 && defaults.moe.demand_pool);
  assert(defaults.moe.dma_queues == 4 && defaults.moe.gemv_lead_kb == 192);
  assert(defaults.moe.m1_gemv && defaults.moe.gemv_feed &&
         defaults.moe.gemv_rows1);
  assert(defaults.moe.hadamard == "off" &&
         defaults.moe.hadamard_scope == "2bit");

  for (const auto &selection : {std::make_pair("gemma4", "sr15_v2"),
                                std::make_pair("lfm2_moe", "htp_fused_v1")}) {
    json other = json::object();
    assert(!preset(other, selection.first, selection.second));
    assert(other.empty());
  }
  json other = json::object();
  assert(!preset(other, "lfm2_moe", "sr15_v2", "cpu") && other.empty());
  json disabled = {{"htp", {{"lfm2_default_preset", false}}}};
  const auto disabled_before = parseHtpRuntimeConfig(disabled);
  assert(!preset(disabled) &&
         parseHtpRuntimeConfig(disabled) == disabled_before);

  json cli = json::object();
  assert(preset(cli));
  const auto lfm = parseHtpRuntimeConfig(cli);
  assert(lfm.htp.page_cache == 0 && lfm.htp.load_prefetch == false);
  assert(lfm.htp.arena_grain_mib == 4 && lfm.htp.lfm2_conv_stripe_rows == 8);
  assert(lfm.htp.fc_whole_prefill && lfm.htp.lfm2_conv_whole_prefill);
  assert(lfm.htp.lfm2_parallel_norm && lfm.htp.lfm2_parallel_add);
  assert(lfm.htp.lfm2_parallel_stage && lfm.htp.lfm2_prefill_views);
  assert(lfm.htp.lfm2_conv_retain_out && lfm.htp.lfm2_conv_state_only);
  assert(lfm.htp.lfm2_conv_channel_tile && lfm.htp.lfm2_attn_ion_handoff);
  assert(lfm.htp.purge_prefill && lfm.htp.release_prefill_readers);
  assert(lfm.moe.cache_grow == 1 && lfm.moe.preload_experts == 128);
  assert(lfm.moe.prediction == "arm-direct" && lfm.moe.prefetch_slots == 6);
  assert(lfm.moe.prefetch_top == 3 && lfm.moe.tl2_stable_slots);
  assert(lfm.moe.lfm2_prefill_early && !lfm.moe.lfm2_prefill_prune);
  assert(lfm.moe.lfm2_decode_pair_read && lfm.moe.lfm2_decode_demand_handoff);
  assert(lfm.moe.lfm2_compact_recency && lfm.moe.lfm2_prefetch_gamma);
  const auto cli_once = cli;
  assert(preset(cli) && cli == cli_once);
  assert(std::string(std::getenv("NNTR_MOE_CACHE_GROW")) == "99");
  assert(std::string(std::getenv("NNTR_NUM_THREADS")) == "3");
  assert(parseHtpRuntimeConfig(json::object()) == defaults); // API isolation.

  json overrides = {{"htp", {{"page_cache", 1}, {"lfm2_prefill_views", false}}},
                    {"moe",
                     {{"cache_grow", 0},
                      {"prefill_lookahead", 0},
                      {"prediction", "off"},
                      {"prefetch_slots", 2}}}};
  assert(preset(overrides));
  const auto selected = parseHtpRuntimeConfig(overrides);
  assert(selected.htp.page_cache == 1 && !selected.htp.lfm2_prefill_views);
  assert(selected.moe.cache_grow == 0 && selected.moe.prefill_lookahead == 0);
  assert(selected.moe.prediction == "off" && selected.moe.prefetch_slots == 2);
  auto automatic =
    parseHtpRuntimeConfig({{"moe",
                            {{"cache_grow", nullptr},
                             {"decode_prefetch_readers", nullptr},
                             {"prefetch_top", nullptr},
                             {"hadamard", nullptr}}}});
  assert(!automatic.moe.cache_grow && !automatic.moe.decode_prefetch_readers);
  assert(!automatic.moe.prefetch_top && automatic.moe.hadamard == "off");

  json legacy = {{"moe_cache_experts", 0},
                 {"moe_hadamard", "both"},
                 {"moe_hadamard_scope", "all"}};
  auto normalized = causallm::normalizeHtpRuntimeConfig(legacy);
  assert(normalized.moe.cache_experts == 16 &&
         normalized.moe.hadamard == "both");
  assert(normalized.moe.hadamard_scope == "all");
  assert(parseHtpRuntimeConfig(legacy) == normalized);
  assert(causallm::normalizeHtpRuntimeConfig(legacy) == normalized);
  invalid({{"moe_cache_experts", 16}, {"moe", {{"cache_experts", 24}}}});
  invalid({{"moe_hadamard", "both"}, {"moe", {{"hadamard", "off"}}}});

  for (const auto &value :
       {json{{"htp", false}}, json{{"moe", "default"}},
        json{{"htp", {{"cache_experts", 16}}}},
        json{{"moe", {{"page_cache", 0}}}}, json{{"htp", {{"unknown", true}}}},
        json{{"htp", {{"dsp_queue", 1}}}},
        json{{"moe", {{"cache_experts", -1}}}},
        json{{"moe", {{"preload_experts", "128"}}}},
        json{{"moe", {{"dma_queues", 0}}}}, json{{"moe", {{"dma_queues", 5}}}},
        json{{"moe", {{"decode_prefetch_readers", 0}}}},
        json{{"moe", {{"decode_prefetch_readers", 9}}}},
        json{{"moe", {{"worker_cpus", {0, 32}}}}},
        json{{"moe", {{"worker_cpus", {0, 1.5}}}}},
        json{{"moe", {{"prediction", "arm"}}}},
        json{{"moe", {{"prefetch_residual", 1.01}}}},
        json{{"htp", {{"page_cache", 0}}},
             {"moe", {{"prediction", "dsp-page"}}}},
        json{{"htp", {{"page_cache", 1}}},
             {"moe", {{"tl2_prepared_file", "prepared.bin"}}}}})
    invalid(value);

  for (float value : {std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    invalid({{"htp", {{"q2_margin", value}}}});
    invalid({{"moe", {{"prefetch_residual", value}}}});
    invalid({{"moe", {{"prefetch_confidence", value}}}});
    invalid({{"moe", {{"lrfu_half_life", double(value)}}}});
    auto bad = defaults;
    bad.htp.q2_margin = value;
    bool rejected = false;
    try {
      nntrainer::configureHtpRuntime(bad);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    assert(rejected);
  }

  nntrainer::resetHtpRuntimeConfig();
  nntrainer::configureHtpRuntime(defaults);
  nntrainer::lockHtpRuntimeConfig();
  nntrainer::configureHtpRuntime(defaults); // Same configuration is idempotent.
  bool locked = false;
  try {
    nntrainer::configureHtpRuntime(lfm);
  } catch (const std::logic_error &) {
    locked = true;
  }
  assert(locked && nntrainer::htpRuntimeConfig() == defaults);
  nntrainer::resetHtpRuntimeConfig(); // Model teardown releases the setting
                                      // lock.
  nntrainer::configureHtpRuntime(lfm);
  auto bad = lfm;
  bad.moe.dma_queues = 0;
  bool rejected = false;
  try {
    nntrainer::configureHtpRuntime(bad);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected && nntrainer::htpRuntimeConfig() == lfm);
  nntrainer::resetHtpRuntimeConfig();
  nntrainer::configureHtpRuntime(defaults);
  assert(nntrainer::htpRuntimeConfig() == defaults);
  nntrainer::resetHtpRuntimeConfig();
  std::puts(
    "HTP/MOE CONFIG DEFAULTS / JSON VALIDATION / CLI ISOLATION / RELOAD OK");
}

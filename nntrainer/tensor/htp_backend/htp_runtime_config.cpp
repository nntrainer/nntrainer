// SPDX-License-Identifier: Apache-2.0
/** @file htp_runtime_config.cpp
 * @brief Validation and lifetime of the active HTP configuration.
 * @author NNTrainer contributors
 * @bug No known bugs
 */
#include "htp_runtime_config.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace nntrainer {
namespace {
struct RuntimeConfigState {
  HtpRuntimeConfig config;
  std::mutex mutex;
  bool locked = false;
};

RuntimeConfigState &state() {
  static RuntimeConfigState instance;
  return instance;
}

void require(bool valid, const char *message) {
  if (!valid)
    throw std::invalid_argument(message);
}

bool finiteValue(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

bool finiteValue(double value) {
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

bool oneOf(const std::string &value,
           std::initializer_list<const char *> allowed) {
  return std::any_of(allowed.begin(), allowed.end(),
                     [&](const char *entry) { return value == entry; });
}
} // namespace

#define NNTR_RUNTIME_CONFIG_EQUAL(type, name, value)                           \
  if (name != other.name)                                                      \
    return false;
bool HtpConfig::operator==(const HtpConfig &other) const {
  NNTR_HTP_CONFIG_FIELDS(NNTR_RUNTIME_CONFIG_EQUAL)
  return true;
}
bool MoeConfig::operator==(const MoeConfig &other) const {
  NNTR_MOE_CONFIG_FIELDS(NNTR_RUNTIME_CONFIG_EQUAL)
  return true;
}
#undef NNTR_RUNTIME_CONFIG_EQUAL

const HtpRuntimeConfig &htpRuntimeConfig() { return state().config; }

void validateHtpRuntimeConfig(const HtpRuntimeConfig &config) {
  const auto &h = config.htp;
  const auto &m = config.moe;
  require(h.profile <= 3, "htp.profile must be 0..3");
  require(h.page_cache >= -1 && h.page_cache <= 1,
          "htp.page_cache must be -1, 0 or 1");
  require(h.arena_grain_mib == 4 || h.arena_grain_mib == 16 ||
            h.arena_grain_mib == 64,
          "htp.arena_grain_mib must be 4, 16 or 64");
  require(oneOf(h.fc_feed, {"vtcm", "l2"}), "htp.fc_feed must be vtcm or l2");
  require(
    (!h.fc_lanes_small || (*h.fc_lanes_small >= 1 && *h.fc_lanes_small <= 8)) &&
      (!h.fc_lanes_large || (*h.fc_lanes_large >= 1 && *h.fc_lanes_large <= 8)),
    "htp FC lane counts must be 1..8");
  require(finiteValue(h.q2_margin) && h.q2_margin > 0,
          "htp.q2_margin must be finite and positive");
  require(!h.attn_f32_check || *h.attn_f32_check >= 0,
          "htp.attn_f32_check must be nonnegative");
  require(h.lfm2_conv_stripe_rows == 8 || h.lfm2_conv_stripe_rows == 16 ||
            h.lfm2_conv_stripe_rows == 32 || h.lfm2_conv_stripe_rows == 64 ||
            h.lfm2_conv_stripe_rows == 128 || h.lfm2_conv_stripe_rows == 256,
          "htp.lfm2_conv_stripe_rows must be 8, 16, 32, 64, 128 or 256");
  require(!h.dump_all || !h.dump_dir.empty(),
          "htp.dump_all requires htp.dump_dir");
  require(m.cache_experts >= 1, "moe.cache_experts must be positive");
  require(finiteValue(m.lrfu_half_life) && m.lrfu_half_life >= 0,
          "moe.lrfu_half_life must be finite and nonnegative");
  require(m.prefill_lookahead <= INT_MAX,
          "moe.prefill_lookahead exceeds INT_MAX");
  require(h.dma_trace_calls <= INT_MAX, "htp.dma_trace_calls exceeds INT_MAX");
  require(m.prefill_readers >= 1, "moe.prefill_readers must be positive");
  for (int cpu : m.worker_cpus)
    require(cpu >= 0 && cpu < 32, "moe.worker_cpus entries must be 0..31");
  require(
    oneOf(m.prediction,
          {"auto", "off", "dsp-page", "dsp-direct", "arm-direct"}),
    "moe.prediction must be auto, off, dsp-page, dsp-direct or arm-direct");
  require(!m.decode_prefetch_readers || (*m.decode_prefetch_readers >= 1 &&
                                         *m.decode_prefetch_readers <= 8),
          "moe.decode_prefetch_readers must be 1..8 or null");
  require(m.prefetch_slots >= 1 && m.prefetch_slots <= 32,
          "moe.prefetch_slots must be 1..32");
  require(!m.prefetch_top || *m.prefetch_top <= 16,
          "moe.prefetch_top must be 0..16 or null");
  require(finiteValue(m.prefetch_residual) && m.prefetch_residual >= 0 &&
            m.prefetch_residual <= 1,
          "moe.prefetch_residual must be in [0, 1]");
  require(finiteValue(m.prefetch_confidence) && m.prefetch_confidence >= 0 &&
            m.prefetch_confidence <= 1,
          "moe.prefetch_confidence must be in [0, 1]");
  require(m.miss_readers >= 1, "moe.miss_readers must be positive");
  require(m.fadvise <= 2 && m.pin <= 2, "moe.fadvise and moe.pin must be 0..2");
  require(m.gemv_lead_kb <= 8128, "moe.gemv_lead_kb must be 0..8128");
  require(m.dma_queues >= 1 && m.dma_queues <= 4,
          "moe.dma_queues must be 1..4");
  require(
    !m.hadamard ||
      oneOf(*m.hadamard, {"off", "up", "down", "both", "both_scalar", "up_hmx",
                          "down_hmx", "both_hmx", "both_paley"}),
    "Invalid moe.hadamard mode");
  require(!m.hadamard_scope || oneOf(*m.hadamard_scope, {"2bit", "all"}),
          "moe.hadamard_scope must be 2bit or all");
  require(m.host_mask <= 7, "moe.host_mask must be 0..7");
  require(m.tl2_prepared_file.empty() || h.page_cache == -1,
          "moe.tl2_prepared_file requires htp.page_cache=-1");
  require(m.prediction != "dsp-page" || h.page_cache == -1,
          "moe.prediction=dsp-page requires htp.page_cache=-1");
}

void configureHtpRuntime(const HtpRuntimeConfig &config) {
  validateHtpRuntimeConfig(config);
  auto &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.config == config)
    return;
  if (s.locked)
    throw std::logic_error(
      "HTP configuration cannot change while a session is open");
  s.config = config;
}

void lockHtpRuntimeConfig() {
  auto &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.locked = true;
}

void resetHtpRuntimeConfig() {
  auto &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.config = HtpRuntimeConfig{};
  s.locked = false;
}
} // namespace nntrainer

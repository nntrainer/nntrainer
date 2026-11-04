// SPDX-License-Identifier: Apache-2.0
/** @file attention_calibration_check.cpp
 * @brief Independent prompt calibration must not inherit preceding Q ranges.
 * Link against the same nntrainer engine as the application; no DSP required.
 */
#include <array>
#include <cstdint>
#include <htp_attention_state.h>
#include <htp_runtime_config.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

int main() {
  nntrainer::HtpRuntimeConfig runtime;
  runtime.htp.kv_prefix_cache = true;
  try {
    nntrainer::validateHtpRuntimeConfig(runtime);
    std::cerr << "FAIL: radix was enabled without paged sessions\n";
    return 1;
  } catch (const std::invalid_argument &) {
  }
  runtime.htp.kv_session = runtime.htp.kv_paged = true;
  nntrainer::validateHtpRuntimeConfig(runtime);
  nntrainer::AttentionConfig config;
  config.num_heads_Q = 2;
  config.num_heads_KV = 1;
  config.head_dim = 64;
  nntrainer::HtpAttentionState reused(config), fresh(config);
  reused.name("reused");
  fresh.name("fresh");
  std::array<uint16_t, 64> kv;
  kv.fill(0x3c00); // FP16 one
  std::array<float, 128> wide, narrow;
  for (size_t i = 0; i < wide.size(); ++i) {
    wide[i] = i & 1 ? 100.0f : -100.0f;
    narrow[i] = i & 1 ? 1.0f : -1.0f;
  }
  reused.calibrate_q2_scales(kv.data(), kv.data(), 1, 64, wide.data(), 128, 1);
  reused.beginHtpPrompt();
  reused.calibrate_q2_scales(kv.data(), kv.data(), 1, 64, narrow.data(), 128,
                             1);
  fresh.calibrate_q2_scales(kv.data(), kv.data(), 1, 64, narrow.data(), 128, 1);
  const auto checkpoint = fresh.capturePrefixCalibration();
  nntrainer::HtpAttentionState resumed(config);
  resumed.name("resumed");
  resumed.calibrate_q2_scales(kv.data(), kv.data(), 1, 64, wide.data(), 128, 1);
  resumed.restorePrefixCalibration(*checkpoint);
  const auto restored = resumed.capturePrefixCalibration();
  if (restored->scale_k != checkpoint->scale_k ||
      restored->scale_v != checkpoint->scale_v ||
      restored->kmax != checkpoint->kmax ||
      restored->vmax != checkpoint->vmax ||
      restored->q_enc != checkpoint->q_enc ||
      restored->out_enc != checkpoint->out_enc ||
      restored->calibrated != checkpoint->calibrated) {
    std::cerr << "FAIL: prefix calibration was not restored exactly\n";
    return 1;
  }
  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  nntrainer::HtpAttentionState::saveQ2Calibration(stream);
  auto get = [&] {
    uint32_t value = 0;
    stream.read(reinterpret_cast<char *>(&value), sizeof(value));
    return value;
  };
  std::map<std::string, std::vector<float>> entries;
  const auto count = get();
  for (uint32_t i = 0; i < count; ++i) {
    std::string name(get(), '\0');
    stream.read(name.data(), name.size());
    std::vector<float> values(get());
    stream.read(reinterpret_cast<char *>(values.data()),
                values.size() * sizeof(float));
    entries.emplace(std::move(name), std::move(values));
  }
  if (!stream || entries.at("fresh") != entries.at("reused")) {
    std::cerr << "FAIL: fresh Q calibration inherited an earlier request\n";
    return 1;
  }
  std::cout << "Independent request K/V/Q/output calibration equality: PASS\n";
}

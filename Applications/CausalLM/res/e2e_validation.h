// SPDX-License-Identifier: Apache-2.0
/** @file e2e_validation.h @brief Numeric checks for standalone model
 * validation. */
#ifndef NNTR_E2E_VALIDATION_H
#define NNTR_E2E_VALIDATION_H
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace e2e {
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
inline std::vector<float> readFloats(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + path);
  const auto bytes = f.tellg();
  if (bytes <= 0 || bytes % sizeof(float))
    throw std::runtime_error("invalid float file " + path);
  std::vector<float> v(static_cast<size_t>(bytes) / sizeof(float));
  f.seekg(0);
  f.read(reinterpret_cast<char *>(v.data()), bytes);
  if (!f)
    throw std::runtime_error("short read " + path);
  return v;
}
inline bool compare(const std::string &name, const std::vector<float> &actual,
                    const std::vector<float> &reference, float atol,
                    float rtol = 0) {
  if (actual.empty() || actual.size() != reference.size())
    throw std::runtime_error(name + ": element count mismatch");
  double sq = 0;
  float worst = 0;
  size_t bad = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    float error = std::abs(actual[i] - reference[i]);
    if (!std::isfinite(actual[i]) || !std::isfinite(reference[i]) ||
        error > atol + rtol * std::abs(reference[i]))
      ++bad;
    worst = std::max(worst, error);
    sq += static_cast<double>(error) * error;
  }
  std::cout << "[numeric] " << name << " count=" << actual.size()
            << " max_abs=" << worst << " rmse=" << std::sqrt(sq / actual.size())
            << " atol=" << atol << " rtol=" << rtol << " failed=" << bad
            << '\n';
  return bad == 0;
}
} // namespace e2e
#endif

// SPDX-License-Identifier: Apache-2.0
/**
 * @file   qnn_stage_dump.h
 * @brief  Env-gated per-stage dump hooks for the multimodal (vision+LLM) path.
 *
 * Set NNTR_QNN_DUMP_DIR=<dir> to have the vision encoder, the multimodal
 * composer and the LLM generation loop write their intermediates there. When
 * the variable is unset every helper is a no-op (one getenv per process), so
 * the hooks are safe to leave in a release build.
 *
 */

#ifndef __QNN_STAGE_DUMP_H__
#define __QNN_STAGE_DUMP_H__

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace causallm {
namespace stage_dump {

/// Dump root, or nullptr when NNTR_QNN_DUMP_DIR is unset. Read once.
inline const char *dir() {
  static const char *d = std::getenv("NNTR_QNN_DUMP_DIR");
  static const bool empty = (d == nullptr || d[0] == '\0');
  return empty ? nullptr : d;
}

inline bool enabled() { return dir() != nullptr; }

inline std::string path(const std::string &name) {
  return std::string(dir()) + "/" + name;
}

inline void write_bin(const std::string &name, const void *p, size_t bytes) {
  if (!enabled() || p == nullptr)
    return;
  FILE *f = std::fopen(path(name).c_str(), "wb");
  if (f == nullptr)
    return;
  std::fwrite(p, 1, bytes, f);
  std::fclose(f);
}

inline void write_text(const std::string &name, const std::string &s,
                       const char *mode = "wb") {
  if (!enabled())
    return;
  FILE *f = std::fopen(path(name).c_str(), mode);
  if (f == nullptr)
    return;
  std::fwrite(s.data(), 1, s.size(), f);
  std::fclose(f);
}

inline void append_text(const std::string &name, const std::string &s) {
  write_text(name, s, "ab");
}

template <typename T>
inline void write_ints(const std::string &name, const std::vector<T> &v) {
  if (!enabled())
    return;
  std::string s;
  s.reserve(v.size() * 8);
  for (size_t i = 0; i < v.size(); ++i) {
    s += std::to_string(static_cast<long long>(v[i]));
    s += (i + 1 == v.size()) ? '\n' : ' ';
  }
  if (v.empty())
    s = "\n";
  write_text(name, s);
}

/// Top-k (id, dequantized logit) over one UFIXED16 logits row, for
/// 11_steps.jsonl.
inline std::string topk_json(const uint16_t *row, int n, int k, float scale,
                             int offset) {
  std::vector<int> idx(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i)
    idx[static_cast<size_t>(i)] = i;
  const int kk = std::min(k, n);
  std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                    [row](int a, int b) { return row[a] > row[b]; });
  std::string s = "[";
  char buf[96];
  for (int i = 0; i < kk; ++i) {
    const double v = static_cast<double>(scale) * (row[idx[i]] + offset);
    std::snprintf(buf, sizeof(buf), "%s[%d,%.6f]", i ? "," : "", idx[i], v);
    s += buf;
  }
  s += "]";
  return s;
}

} // namespace stage_dump
} // namespace causallm

#endif /* __QNN_STAGE_DUMP_H__ */

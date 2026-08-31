// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hex_image.cpp
 * @date	31 August 2026
 * @brief	.hexcfg (HexModelConfig as key=value text) and raw file helpers
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "hex_image.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>

namespace nntrainer::hexagon {

void write_hexcfg(const std::string &path, const HexModelConfig &c) {
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error("hexcfg: cannot write " + path);
  f << "n_layers=" << c.n_layers << "\nn_heads=" << c.n_heads
    << "\nn_kv_heads=" << c.n_kv_heads << "\nhead_dim=" << c.head_dim
    << "\nhidden=" << c.hidden << "\nffn=" << c.ffn << "\nvocab=" << c.vocab
    << "\nmax_seq=" << c.max_seq << "\nmax_chunk=" << c.max_chunk << "\n";
  char buf[64];
  // %.9g round-trips any float through strtof exactly.
  snprintf(buf, sizeof(buf), "rms_eps=%.9g\nrope_theta=%.9g\n", c.rms_eps,
           c.rope_theta);
  f << buf;
  if (!f)
    throw std::runtime_error("hexcfg: write failed " + path);
}

HexModelConfig read_hexcfg(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("hexcfg: cannot read " + path);
  std::map<std::string, std::string> kv;
  std::string line;
  while (std::getline(f, line)) {
    auto eq = line.find('=');
    if (eq == std::string::npos)
      continue;
    kv[line.substr(0, eq)] = line.substr(eq + 1);
  }
  auto get = [&](const char *key) -> const std::string & {
    auto it = kv.find(key);
    if (it == kv.end())
      throw std::runtime_error("hexcfg: missing key " + std::string(key) +
                               " in " + path);
    return it->second;
  };
  auto u32 = [&](const char *key) {
    return static_cast<uint32_t>(strtoul(get(key).c_str(), nullptr, 10));
  };
  HexModelConfig c{};
  c.n_layers = u32("n_layers");
  c.n_heads = u32("n_heads");
  c.n_kv_heads = u32("n_kv_heads");
  c.head_dim = u32("head_dim");
  c.hidden = u32("hidden");
  c.ffn = u32("ffn");
  c.vocab = u32("vocab");
  c.max_seq = u32("max_seq");
  c.max_chunk = u32("max_chunk");
  c.rms_eps = strtof(get("rms_eps").c_str(), nullptr);
  c.rope_theta = strtof(get("rope_theta").c_str(), nullptr);
  return c;
}

void write_file(const std::string &path, const void *data, uint64_t size) {
  std::FILE *f = std::fopen(path.c_str(), "wb");
  if (!f)
    throw std::runtime_error("cannot write " + path);
  bool ok = std::fwrite(data, 1, size, f) == size;
  ok = (std::fclose(f) == 0) && ok;
  if (!ok)
    throw std::runtime_error("short write " + path);
}

void read_file_into(const std::string &path, void *dst, uint64_t size) {
  std::FILE *f = std::fopen(path.c_str(), "rb");
  if (!f)
    throw std::runtime_error("cannot read " + path);
  bool ok = std::fread(dst, 1, size, f) == size;
  ok = ok && std::fgetc(f) == EOF; // exact size, no trailing bytes
  std::fclose(f);
  if (!ok)
    throw std::runtime_error("size mismatch reading " + path);
}

std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot read " + path);
  std::vector<uint8_t> v(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(reinterpret_cast<char *>(v.data()), v.size());
  if (!f)
    throw std::runtime_error("short read " + path);
  return v;
}

} // namespace nntrainer::hexagon

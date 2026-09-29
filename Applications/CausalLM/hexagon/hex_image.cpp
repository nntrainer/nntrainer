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
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

namespace nntrainer::hexagon {

namespace {
const char *const kTensorNames[8] = {"embed", "q",    "k",  "v",
                                     "o",     "gate", "up", "down"};
} // namespace

std::string hex_i8_names(uint32_t mask) {
  std::string s;
  for (uint32_t i = 0; i < 8u; ++i)
    if (mask & (1u << i))
      s += (s.empty() ? "" : ",") + std::string(kTensorNames[i]);
  return s;
}

uint32_t hex_i8_mask(const std::string &names) {
  uint32_t mask = 0;
  size_t pos = 0;
  while (pos < names.size()) {
    size_t end = names.find(',', pos);
    if (end == std::string::npos)
      end = names.size();
    const std::string name = names.substr(pos, end - pos);
    uint32_t i = 0;
    while (i < 8u && name != kTensorNames[i])
      ++i;
    if (i == 8u)
      throw std::runtime_error("hexcfg: unknown tensor name '" + name + "'");
    mask |= 1u << i;
    pos = end + 1;
  }
  return mask;
}

const char *hex_layout_name(uint32_t layout) {
  switch (layout) {
  case NNTR_HTP_WEIGHT_LAYOUT_TILED32:
    return "tiled32";
  case NNTR_HTP_WEIGHT_LAYOUT_W4CX_DOWN8:
    return "w4cx_down8";
  case NNTR_HTP_WEIGHT_LAYOUT_W4CX:
    return "w4cx";
  default:
    throw std::runtime_error("hexcfg: unknown weight_layout id " +
                             std::to_string(layout));
  }
}

void write_hexcfg(const std::string &path, const HexModelConfig &c) {
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error("hexcfg: cannot write " + path);
  f << "n_layers=" << c.n_layers << "\nn_heads=" << c.n_heads
    << "\nn_kv_heads=" << c.n_kv_heads << "\nhead_dim=" << c.head_dim
    << "\nhidden=" << c.hidden << "\nffn=" << c.ffn << "\nvocab=" << c.vocab
    << "\nmax_seq=" << c.max_seq << "\nmax_chunk=" << c.max_chunk
    << "\nweight_layout=" << hex_layout_name(c.weight_layout) << "\n";
  if (c.weight_layout != NNTR_HTP_WEIGHT_LAYOUT_TILED32)
    f << "i8_tensors=" << hex_i8_names(c.i8_mask) << "\n";
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
  // v4 images record the WEIGHTS byte order. A .hexcfg without the key was
  // packed row-major by a pre-v4 nntr_hexpack; its .hexw would be read
  // through the tiled index and decode as garbage, so refuse it here.
  auto wl = kv.find("weight_layout");
  if (wl == kv.end())
    throw std::runtime_error("hexcfg: legacy image (no weight_layout) " + path +
                             ", regenerate with nntr_hexpack");
  // Only the layouts the v5 kernels read: w4cx (4-bit embed / down, #65
  // S3) is a reserved id and stays refused until those kernels land.
  HexModelConfig c{};
  if (wl->second == "tiled32") {
    c.weight_layout = NNTR_HTP_WEIGHT_LAYOUT_TILED32;
    c.i8_mask = kHexAllI8;
  } else if (wl->second == "w4cx_down8") {
    c.weight_layout = NNTR_HTP_WEIGHT_LAYOUT_W4CX_DOWN8;
    c.i8_mask = hex_i8_mask(get("i8_tensors"));
    if ((c.i8_mask & kHexW4cxDown8I8) != kHexW4cxDown8I8)
      throw std::runtime_error(
        "hexcfg: w4cx_down8 needs embed and down in i8_tensors, in " + path);
  } else {
    throw std::runtime_error("hexcfg: unsupported weight_layout '" +
                             wl->second + "' in " + path);
  }
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

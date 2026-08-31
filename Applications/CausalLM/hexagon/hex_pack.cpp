// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hex_pack.cpp
 * @date	31 August 2026
 * @brief	nntr_hexpack: build the packed DSP WEIGHTS image (.hexw) and its
 *		shape file (.hexcfg) from a W8_CX qwen3-0.6b checkpoint
 *
 * Usage: nntr_hexpack <w8cx.bin> <out-prefix> [--layers N] [--max-seq N]
 *                     [--max-chunk N]
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "hex_image.h"
#include "nntr_htp_common.h"
#include "qwen3_lowering.h"
#include "qwen3_w8cx_bin.h"

using namespace nntrainer::hexagon;

namespace {

int usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s <w8cx.bin> <out-prefix> [--layers N] [--max-seq N] "
               "[--max-chunk N]\n",
               argv0);
  return 2;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3)
    return usage(argv[0]);
  HexModelConfig full = kQwen3_0_6b;
  uint32_t opt_layers = 0;
  for (int i = 3; i < argc; ++i) {
    if (i + 1 >= argc)
      return usage(argv[0]);
    uint32_t v = static_cast<uint32_t>(strtoul(argv[i + 1], nullptr, 10));
    if (!std::strcmp(argv[i], "--layers"))
      opt_layers = v;
    else if (!std::strcmp(argv[i], "--max-seq"))
      full.max_seq = v;
    else if (!std::strcmp(argv[i], "--max-chunk"))
      full.max_chunk = v;
    else
      return usage(argv[0]);
    ++i;
  }
  if (opt_layers > full.n_layers) {
    std::fprintf(stderr, "--layers exceeds checkpoint layers\n");
    return 2;
  }

  try {
    Qwen3W8cxBin bin(argv[1], full); // reads all 28 layers
    HexModelConfig cfg = full;
    cfg.n_layers = opt_layers ? opt_layers : full.n_layers;

    HexModelWeights w = bin.weights();
    w.layers.resize(cfg.n_layers); // 1-layer bring-up

    HexLoweredGraph g = lower_qwen3(cfg);
    std::vector<uint8_t> image(g.weights_size);
    pack_weights(g, cfg, w, image.data());

    const std::string prefix = argv[2];
    write_file(prefix + ".hexw", image.data(), image.size());
    write_hexcfg(prefix + ".hexcfg", cfg);

    nntr_htp_oplist_header h;
    std::memcpy(&h, g.oplist.data(), sizeof(h));
    std::printf("HEXPACK weights=%llu kv=%llu act=%llu n_ops=%u\n",
                (unsigned long long)g.weights_size,
                (unsigned long long)g.kv_size, (unsigned long long)g.act_size,
                h.n_ops);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "nntr_hexpack: %s\n", e.what());
    return 1;
  }
  return 0;
}

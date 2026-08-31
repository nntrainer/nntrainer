// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hexagon_backend.cpp
 * @date	31 August 2026
 * @brief	Whole-graph DSP session for one qwen3 W8_CX checkpoint
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "hexagon_backend.h"

#ifdef ENABLE_HEXAGON
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "hexagon_runner.h"
#include "qwen3_lowering.h"
#include "qwen3_w8cx_bin.h"
#include "rpcmem_allocator.h"

namespace nntrainer::hexagon {

std::unique_ptr<HexagonBackend>
HexagonBackend::create(const std::string &w8cx_bin, const HexModelConfig &cfg) {
  std::unique_ptr<HexagonBackend> b(new HexagonBackend());
  b->cfg_ = cfg;
  try {
    Qwen3W8cxBin bin(w8cx_bin, cfg); // mmap; released at scope exit
    HexLoweredGraph g = lower_qwen3(cfg);
    b->weights_ = std::make_shared<RpcmemBuffer>(g.weights_size);
    b->kv_ = std::make_shared<RpcmemBuffer>(g.kv_size);
    b->act_ = std::make_shared<RpcmemBuffer>(g.act_size);
    if (!b->weights_->valid() || !b->kv_->valid() || !b->act_->valid())
      throw std::runtime_error("rpcmem allocation failed");
    pack_weights(g, cfg, bin.weights(), (uint8_t *)b->weights_->data());
    std::memset(b->kv_->data(), 0, g.kv_size);
    std::memset(b->act_->data(), 0, g.act_size);
    b->runner_ = HexagonRunner::create(); // logs its own reason
    if (!b->runner_ ||
        b->runner_->init(g.oplist.data(), (uint32_t)g.oplist.size(),
                         *b->weights_, *b->kv_, *b->act_) != 0)
      return nullptr;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "hexagon: %s, CPU fallback\n", e.what());
    return nullptr;
  }
  return b;
}

int HexagonBackend::forward(const int32_t *tokens, uint32_t n_tokens,
                            uint32_t pos, float *logits) {
  while (n_tokens) {
    const uint32_t n = n_tokens < cfg_.max_chunk ? n_tokens : cfg_.max_chunk;
    int err = runner_->forward(tokens, n, pos, logits, cfg_.vocab);
    if (err)
      return err;
    tokens += n;
    pos += n;
    n_tokens -= n;
  }
  return 0;
}

} // namespace nntrainer::hexagon
#endif /* ENABLE_HEXAGON */

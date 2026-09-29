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
#include "nntr_htp_common.h"
#include "qwen3_lowering.h"
#include "qwen3_w8cx_bin.h"
#include "rpcmem_allocator.h"

namespace nntrainer::hexagon {

std::unique_ptr<HexagonBackend>
HexagonBackend::create(const std::string &w8cx_bin,
                       const HexModelConfig &cfg_in) {
  std::unique_ptr<HexagonBackend> b(new HexagonBackend());
  b->cfg_ = cfg_in;
  try {
    Qwen3W8cxBin bin(w8cx_bin, cfg_in); // mmap; released at scope exit
    /* The checkpoint decides the WEIGHTS layout (tiled32 or w4cx_down8). */
    HexModelConfig cfg = cfg_in;
    bin.apply_layout(cfg);
    b->cfg_ = cfg;
    HexLoweredGraph g = lower_qwen3(cfg);
    b->weights_ = std::make_shared<RpcmemBuffer>(g.weights_size);
    b->kv_ = std::make_shared<RpcmemBuffer>(g.kv_size);
    b->act_ = std::make_shared<RpcmemBuffer>(g.act_size);
    b->logits_ = std::make_shared<RpcmemBuffer>((size_t)cfg.vocab * 4);
    if (!b->weights_->valid() || !b->kv_->valid() || !b->act_->valid() ||
        !b->logits_->valid())
      throw std::runtime_error("rpcmem allocation failed");
    pack_weights(g, cfg, bin.weights(), (uint8_t *)b->weights_->data());
    std::memset(b->kv_->data(), 0, g.kv_size);
    std::memset(b->act_->data(), 0, g.act_size);
    b->runner_ = HexagonRunner::create(); // logs its own reason
    if (!b->runner_ ||
        b->runner_->init(g.oplist.data(), (uint32_t)g.oplist.size(),
                         *b->weights_, *b->kv_, *b->act_) != 0)
      return nullptr;
    /** #24: map the logits buffer once (FASTRPC_MAP_STATIC) so forward()
     * reuses the remote mapping instead of mapping the fd per call — 2.5 ms
     * less per isolated call, < 0.1 ms inside a real decode step. Not
     * fatal: an SDK without the flag, or a failed mmap, leaves the plain
     * rpcmem (fd per call) path, which returns the same bytes. */
    if (b->runner_->register_static(*b->logits_) != 0)
      std::fprintf(stderr, "hexagon: logits buffer stays on the per-call "
                           "rpcmem path\n");
  } catch (const std::exception &e) {
    std::fprintf(stderr, "hexagon: %s, CPU fallback\n", e.what());
    return nullptr;
  }
  return b;
}

int HexagonBackend::forward(const int32_t *tokens, uint32_t n_tokens,
                            uint32_t pos, float *logits) {
  /** Same predicate as the DSP, applied to the whole request before the
   * first RPC: the DSP rejects per chunk, and the chunks before a bad id
   * would already sit in KV. */
  if (!nntr_htp_token_ids_ok(tokens, n_tokens, cfg_.vocab))
    return kHexagonBadParm;
  float *out = static_cast<float *>(logits_->data());
  while (n_tokens) {
    const uint32_t n = n_tokens < cfg_.max_chunk ? n_tokens : cfg_.max_chunk;
    int err = runner_->forward(tokens, n, pos, out, cfg_.vocab);
    if (err)
      return err;
    tokens += n;
    pos += n;
    n_tokens -= n;
  }
  /** Only the last chunk's logits are the result; ~0.1 ms on the host for
   * 151,936 floats, against the staging copy the driver no longer makes. */
  std::memcpy(logits, out, (size_t)cfg_.vocab * sizeof(float));
  return 0;
}

} // namespace nntrainer::hexagon
#endif /* ENABLE_HEXAGON */

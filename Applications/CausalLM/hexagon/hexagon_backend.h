// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hexagon_backend.h
 * @date	31 August 2026
 * @brief	Whole-graph DSP session for one qwen3 W8_CX checkpoint: packs the
 *		weight image into rpcmem, owns the KV/ACT buffers and the
 *		FastRPC runner. create() returning nullptr means "run on CPU".
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef NNTR_HEXAGON_BACKEND_H
#define NNTR_HEXAGON_BACKEND_H

#include <cstdint>
#include <memory>
#include <string>

#include "graph_lowering.h"

namespace nntrainer::hexagon {

class RpcmemBuffer;
class HexagonRunner;

class HexagonBackend {
public:
  /** @return nullptr (after a message on stderr) when the DSP path is not
   *          usable: bad checkpoint, rpcmem, skel or ABI failure. */
  static std::unique_ptr<HexagonBackend> create(const std::string &w8cx_bin,
                                                const HexModelConfig &cfg);

  /** Runs n_tokens at sequence position pos in max_chunk pieces.
   *  @param logits cfg.vocab floats of the last token
   *  @return 0 on success, the FastRPC/DSP error otherwise */
  int forward(const int32_t *tokens, uint32_t n_tokens, uint32_t pos,
              float *logits);

private:
  HexagonBackend() = default;
  HexModelConfig cfg_{};
  /* shared_ptr: deleters bind at construction, so this header stays free of
   * the SDK-dependent definitions (builds without ENABLE_HEXAGON too). */
  std::shared_ptr<RpcmemBuffer> weights_, kv_, act_;
  std::shared_ptr<HexagonRunner> runner_;
};

} // namespace nntrainer::hexagon

#endif /* NNTR_HEXAGON_BACKEND_H */

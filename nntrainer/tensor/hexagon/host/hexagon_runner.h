// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hexagon_runner.h
 * @date	15 August 2026
 * @brief	Host-side cDSP session: open / init (one-time buffer handoff
 *		+ version handshake) / forward / close.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef __HEXAGON_RUNNER_H__
#define __HEXAGON_RUNNER_H__

#include <cstdint>
#include <memory>

#include "rpcmem_allocator.h"

namespace nntrainer::hexagon {

/**
 * @class HexagonRunner
 * @brief One cDSP session. create() returning nullptr means "no usable DSP"
 *        and the caller must take the CPU fallback path.
 */
class HexagonRunner {
public:
  static std::unique_ptr<HexagonRunner> create();
  ~HexagonRunner();

  /**
   * @brief One-time buffer handoff. weights content must be final before the
   *        call (the RPC itself performs the cache flush).
   * @return 0 on success, AEE error code otherwise (version mismatch,
   *         validation failure, mapping failure).
   */
  int init(const void *oplist, uint32_t oplist_size,
           const RpcmemBuffer &weights, const RpcmemBuffer &kv,
           const RpcmemBuffer &act);

  /** @return 0 on success. Exactly one RPC per call. */
  /** @param dsp_pcycles optional out: DSP cycles spent in the op loop */
  int forward(const int32_t *token_ids, uint32_t n_tokens, uint32_t pos,
              float *logits, uint32_t n_logits,
              uint64_t *dsp_pcycles = nullptr);

  /**
   * @brief Run ops [0, n_ops_limit) and copy dump_bytes from
   *        bufs[dump_buf] + dump_offset into dump (see nntr_htp.idl).
   */
  int forward_debug(const int32_t *token_ids, uint32_t n_tokens, uint32_t pos,
                    uint32_t n_ops_limit, uint32_t dump_buf,
                    uint32_t dump_offset, uint8_t *dump, uint32_t dump_bytes,
                    uint64_t *dsp_pcycles = nullptr);

  HexagonRunner(const HexagonRunner &) = delete;
  HexagonRunner &operator=(const HexagonRunner &) = delete;

private:
  HexagonRunner() = default;
  uint64_t handle_ = 0;
};

} // namespace nntrainer::hexagon
#endif // __HEXAGON_RUNNER_H__

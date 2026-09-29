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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "rpcmem_allocator.h"

namespace nntrainer::hexagon {

/**
 * @brief The AEE_EBADPARM value the DSP returns for a rejected argument
 *        (bad op-list, token count, position or token id). Spelled out so
 *        SDK-free callers can produce the same code as a DSP rejection:
 *        AEE_EBADPARM (14) offset by FastRPC's 0x80000400 DSP error base;
 *        hexagon_runner.cpp static_asserts it against AEEStdErr.h.
 */
constexpr int kHexagonBadParm = static_cast<int>(0x8000040Eu);

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

  /**
   * @brief Map an rpcmem buffer on the DSP once, with FASTRPC_MAP_STATIC, so
   *        every later forward() that passes a pointer inside it as the
   *        logits argument reuses the same remote mapping (the driver still
   *        does the cache maintenance around each call). Optional: a plain
   *        rpcmem logits buffer already takes the fd (zero-copy) path; this
   *        only removes the per-call map/unmap. The runner remembers what it
   *        mapped and unmaps it in the destructor, before closing the
   *        session, so the buffer must outlive the runner (declare it
   *        first, as the harnesses do with WEIGHTS/KV/ACT).
   * @return 0 on success (also for a buffer this runner already mapped),
   *         AEE_EUNSUPPORTED when the build has no FASTRPC_MAP_STATIC
   *         (build_host_test.sh and meson probe remote.h and define
   *         NNTR_HAVE_FASTRPC_MAP_STATIC), the fastrpc_mmap error otherwise
   *         — including AEE_EALREADY for an fd mapped by someone else.
   */
  int register_static(const RpcmemBuffer &buf);

  /** @return true when register_static() can succeed on this build */
  static bool static_map_supported();

  /**
   * @brief Exactly one RPC per call. logits may point anywhere the caller
   *        likes; inside an RpcmemBuffer the FastRPC library passes the
   *        buffer's fd instead of staging a copy of the n_logits floats.
   * @param dsp_pcycles optional out: DSP cycles spent in the op loop
   * @return 0 on success
   */
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
  /** what register_static() mapped; unmapped in the destructor */
  struct StaticMap {
    int fd;
    void *data;
    size_t size;
  };
  std::vector<StaticMap> static_maps_;
};

} // namespace nntrainer::hexagon
#endif // __HEXAGON_RUNNER_H__

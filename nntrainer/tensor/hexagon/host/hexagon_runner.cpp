// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hexagon_runner.cpp
 * @date	15 August 2026
 * @brief	Host-side cDSP session runner.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "hexagon_runner.h"

#include <cstdio>

#include <AEEStdErr.h>
#include <remote.h>

#include "nntr_htp.h"
#include "nntr_htp_common.h"

namespace nntrainer::hexagon {

std::unique_ptr<HexagonRunner> HexagonRunner::create() {
  // Best-effort: remote_session_control() may fail on devices/firmwares that
  // do not support unsigned PD requests, but that alone does not mean the DSP
  // is unusable - the open() call below is the real success/failure gate, so
  // the return value here is intentionally ignored.
  remote_rpc_control_unsigned_module um;
  um.domain = CDSP_DOMAIN_ID;
  um.enable = 1;
  remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, (void *)&um,
                         sizeof(um));

  remote_handle64 handle = 0;
  int err = nntr_htp_open(nntr_htp_URI CDSP_DOMAIN, &handle);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "hexagon: open failed (0x%x), CPU fallback\n", err);
    return nullptr;
  }
  auto runner = std::unique_ptr<HexagonRunner>(new HexagonRunner());
  runner->handle_ = handle;
  return runner;
}

HexagonRunner::~HexagonRunner() {
  if (handle_ != 0)
    nntr_htp_close(handle_);
}

// A raw fd number is meaningless to the DSP until the host registers it with
// the FastRPC driver for the domain; FASTRPC_MAP_FD_DELAYED defers the actual
// mapping to the DSP-side HAP_mmap call in nntr_htp_init.
static int map_on_dsp(const RpcmemBuffer &buf) {
  int err = fastrpc_mmap(CDSP_DOMAIN_ID, buf.fd(), buf.data(), 0, buf.size(),
                         FASTRPC_MAP_FD_DELAYED);
  return err == AEE_EALREADY ? 0 : err; /* re-init passes the same fds */
}

int HexagonRunner::init(const void *oplist, uint32_t oplist_size,
                        const RpcmemBuffer &weights, const RpcmemBuffer &kv,
                        const RpcmemBuffer &act) {
  uint32_t dsp_abi_version = 0;
  for (const RpcmemBuffer *buf : {&weights, &kv, &act}) {
    int err = map_on_dsp(*buf);
    if (err != AEE_SUCCESS) {
      fprintf(stderr, "hexagon: fastrpc_mmap failed (0x%x, fd=%d)\n", err,
              buf->fd());
      return err;
    }
  }
  int err = nntr_htp_init(handle_, (const uint8_t *)oplist, (int)oplist_size,
                          (const uint8_t *)weights.data(), (int)weights.size(),
                          weights.fd(), kv.fd(), (uint32_t)kv.size(), act.fd(),
                          (uint32_t)act.size(), &dsp_abi_version);
  if (err != AEE_SUCCESS) {
    fprintf(stderr, "hexagon: init failed (0x%x), host abi v%u, dsp abi v%u\n",
            err, NNTR_HTP_ABI_VERSION, dsp_abi_version);
    return err;
  }
  // Any nonzero return means: destroy the runner (the DSP may already be
  // initialized; ~HexagonRunner closes the session).
  if (dsp_abi_version != NNTR_HTP_ABI_VERSION) {
    fprintf(stderr, "hexagon: abi mismatch host v%u vs dsp v%u\n",
            NNTR_HTP_ABI_VERSION, dsp_abi_version);
    return AEE_EUNSUPPORTED;
  }
  return 0;
}

int HexagonRunner::forward(const int32_t *token_ids, uint32_t n_tokens,
                           uint32_t pos, float *logits, uint32_t n_logits,
                           uint64_t *dsp_pcycles) {
  uint64 pc = 0; /* QAIC type; may differ from uint64_t on aarch64 */
  int err = nntr_htp_forward(handle_, token_ids, (int)n_tokens, pos, logits,
                             (int)n_logits, &pc);
  if (dsp_pcycles)
    *dsp_pcycles = pc;
  return err;
}

int HexagonRunner::forward_debug(const int32_t *token_ids, uint32_t n_tokens,
                                 uint32_t pos, uint32_t n_ops_limit,
                                 uint32_t dump_buf, uint32_t dump_offset,
                                 uint8_t *dump, uint32_t dump_bytes,
                                 uint64_t *dsp_pcycles) {
  uint64 pc = 0; /* QAIC type; may differ from uint64_t on aarch64 */
  int err =
    nntr_htp_forward_debug(handle_, token_ids, (int)n_tokens, pos, n_ops_limit,
                           dump_buf, dump_offset, dump, (int)dump_bytes, &pc);
  if (dsp_pcycles)
    *dsp_pcycles = pc;
  return err;
}

} // namespace nntrainer::hexagon

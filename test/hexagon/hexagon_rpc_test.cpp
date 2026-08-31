// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hexagon_rpc_test.cpp
 * @date	16 August 2026
 * @brief	On-device dummy round-trip test for the Hexagon RPC skeleton
 *		(M1). Every line it prints starts with "RPC_TEST" so
 *		tools/hexagon/check_rpc_log.py can parse device logs.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "hexagon_runner.h"
#include "nntr_htp_common.h"

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("RPC_TEST FAIL: %s (line %d)\n", #cond, __LINE__);                \
      return 1;                                                                \
    }                                                                          \
  } while (0)

using nntrainer::hexagon::HexagonRunner;
using nntrainer::hexagon::RpcmemBuffer;

int main() {
  auto runner = HexagonRunner::create();
  CHECK(runner != nullptr);
  printf("RPC_TEST open ok\n");

  RpcmemBuffer weights(4096), kv(4096), act(4096);
  CHECK(weights.valid() && kv.valid() && act.valid());
  printf("RPC_TEST rpcmem ok (weights fd=%d)\n", weights.fd());

  // Host-written word read back by the DSP through the persistent mapping.
  static_cast<int32_t *>(weights.data())[0] = 1000;

  nntr_htp_oplist_header hdr = {NNTR_HTP_OPLIST_MAGIC, NNTR_HTP_ABI_VERSION, 0,
                                0};

  // Version handshake: a wrong version must be rejected before execution.
  hdr.version = 999u;
  CHECK(runner->init(&hdr, sizeof(hdr), weights, kv, act) != 0);
  printf("RPC_TEST bad-version rejected ok\n");

  hdr.version = NNTR_HTP_ABI_VERSION;
  CHECK(runner->init(&hdr, sizeof(hdr), weights, kv, act) == 0);
  printf("RPC_TEST init ok\n");

  const int32_t token_ids[3] = {5, 7, 11};
  const uint32_t pos = 100;
  float logits[8];

  const int n_iter = 32;
  for (int it = 0; it < n_iter; ++it) {
    std::memset(logits, 0, sizeof(logits));
    auto t0 = std::chrono::steady_clock::now();
    CHECK(runner->forward(token_ids, 3, pos, logits, 8) == 0);
    auto t1 = std::chrono::steady_clock::now();
    long long us =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    printf("RPC_TEST forward_us %lld\n", us);
  }

  // Must match the dummy pattern in nntrainer/tensor/hexagon/htp/executor.c:
  // logits[i] = token_ids[i % 3] + pos + i + weights[0]
  for (int i = 0; i < 8; ++i) {
    float expected = (float)(token_ids[i % 3] + (int32_t)pos + i + 1000);
    CHECK(logits[i] == expected);
  }
  printf("RPC_TEST pattern ok\n");
  printf("RPC_TEST PASS\n");
  return 0;
}

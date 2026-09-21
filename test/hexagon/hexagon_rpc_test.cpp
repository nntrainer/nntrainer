// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hexagon_rpc_test.cpp
 * @date	16 August 2026
 * @brief	On-device dummy round-trip test for the Hexagon RPC skeleton
 *		(M1), plus the full-vocab logits transport rows of #24. Every
 *		line it prints starts with "RPC_TEST" so
 *		tools/hexagon/check_rpc_log.py can parse device logs.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

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
  // Buffers before the runner: they are destroyed after it, i.e. after the
  // session is closed and the static mapping below is undone. `shared` is
  // the #24 full-vocab logits buffer (qwen3-0.6b: 151,936 floats =
  // 607,744 B) used by the transport rows at the end.
  const int n_full = 151936;
  RpcmemBuffer weights(4096), kv(4096), act(4096);
  RpcmemBuffer shared((size_t)n_full * sizeof(float));

  auto runner = HexagonRunner::create();
  CHECK(runner != nullptr);
  printf("RPC_TEST open ok\n");

  CHECK(weights.valid() && kv.valid() && act.valid() && shared.valid());
  printf("RPC_TEST rpcmem ok (weights fd=%d)\n", weights.fd());

  // Host-written word read back by the DSP through the persistent mapping.
  static_cast<int32_t *>(weights.data())[0] = 1000;

  nntr_htp_oplist_header hdr = {NNTR_HTP_OPLIST_MAGIC, NNTR_HTP_ABI_VERSION, 0,
                                0};
  hdr.weight_layout = NNTR_HTP_WEIGHT_LAYOUT_TILED32; // v4 header check

  // Version handshake: a wrong version must be rejected before execution.
  hdr.version = 999u;
  CHECK(runner->init(&hdr, sizeof(hdr), weights, kv, act) != 0);
  printf("RPC_TEST bad-version rejected ok\n");

  hdr.version = NNTR_HTP_ABI_VERSION;
  CHECK(runner->init(&hdr, sizeof(hdr), weights, kv, act) == 0);
  printf("RPC_TEST init ok\n");

  const int32_t token_ids[3] = {5, 7, 11};
  const uint32_t pos = 100;
  const int n_iter = 32;

  // One timed forward() loop for the 8-float row and the transport rows,
  // so the two stay comparable (same clock, same cast, same print shape).
  // The dummy pattern is executor.c's: logits[i] = token_ids[i % 3] + pos +
  // i + weights[0]; `expected` computes it for element i.
  auto expected = [&](int i) {
    return (float)(token_ids[i % 3] + (int32_t)pos + i + 1000);
  };
  auto timed_forwards = [&](const char *label, float *out, int n) -> bool {
    for (int it = 0; it < n_iter; ++it) {
      out[0] = out[n - 1] = 0.0f;
      auto t0 = std::chrono::steady_clock::now();
      if (runner->forward(token_ids, 3, pos, out, (uint32_t)n) != 0)
        return false;
      auto t1 = std::chrono::steady_clock::now();
      long long us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      printf("RPC_TEST %s %lld\n", label, us);
    }
    return true;
  };

  float logits[8];
  CHECK(timed_forwards("forward_us", logits, 8));
  for (int i = 0; i < 8; ++i)
    CHECK(logits[i] == expected(i));
  printf("RPC_TEST pattern ok\n");

  // #24: the same dummy path returning a full-vocab logits vector into
  // three kinds of host memory: malloc = the FastRPC staging copy, rpcmem =
  // the buffer's fd passed instead (mapped per call), static = the same fd
  // mapped once with FASTRPC_MAP_STATIC (skipped when the SDK lacks it).
  // The DSP-side scalar fill of 151,936 elements (a software modulo each,
  // executor.c) is identical across the three rows but is NOT in the
  // 8-float row and the dummy path reports no pcycles, so read the rows
  // against each other (malloc - rpcmem - static) for the transport, and
  // "- forward_us(8)" as an upper bound that includes that fill.
  std::vector<float> heap((size_t)n_full);
  struct Variant {
    const char *name;
    const char *label;
    float *logits;
  };
  const Variant variants[] = {
    {"malloc", "forward_full_us mem=malloc", heap.data()},
    {"rpcmem", "forward_full_us mem=rpcmem",
     static_cast<float *>(shared.data())},
    {"static", "forward_full_us mem=static",
     static_cast<float *>(shared.data())}};
  for (const Variant &v : variants) {
    if (std::strcmp(v.name, "static") == 0) {
      if (!HexagonRunner::static_map_supported()) {
        printf("RPC_TEST static map not in this SDK build, mem=static "
               "skipped\n");
        continue;
      }
      // A target that rejects the map at runtime (AEE_EUNSUPPORTED and the
      // like) skips the row too: the required markers above are the
      // pass/fail contract, the transport rows are informational.
      int err = runner->register_static(shared);
      if (err != 0) {
        printf("RPC_TEST static map failed (0x%x), mem=static skipped\n",
               (unsigned)err);
        continue;
      }
    }
    CHECK(timed_forwards(v.label, v.logits, n_full));
    // First and last element of the same pattern (both exact in fp32).
    CHECK(v.logits[0] == expected(0));
    CHECK(v.logits[n_full - 1] == expected(n_full - 1));
  }
  printf("RPC_TEST full-logits pattern ok\n");
  printf("RPC_TEST PASS\n");
  return 0;
}

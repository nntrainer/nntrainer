// SPDX-License-Identifier: Apache-2.0
/**
 * @file tl2_device_check.cpp
 * @brief Compare native TL2 GEMV modes and worker counts on the DSP.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug See docs/gemma4-tl2-affine-handoff.md for validation limits.
 */
#include "hvx_gemv_u8tl2.h"
#include <AEEStdErr.h>
#include <cstdio>
#include <cstdlib>
#include <nntr_hvx.h>
#include <remote.h>
#include <string>
#include <vector>
static void check(int rc, const char *op) {
  if (rc) {
    std::fprintf(stderr, "%s: 0x%x\n", op, rc);
    std::exit(1);
  }
}
int main() {
  remote_rpc_thread_params thread = {CDSP_DOMAIN_ID, -1, 64 * 1024};
  check(remote_session_control(FASTRPC_THREAD_PARAMS, &thread, sizeof(thread)),
        "thread");
  remote_rpc_control_unsigned_module pd = {CDSP_DOMAIN_ID, 1};
  check(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &pd, sizeof(pd)),
        "unsigned");
  remote_handle64 session = 0;
  const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
  check(nntr_hvx_open(uri.c_str(), &session), "open");
  for (auto dims :
       {std::pair<unsigned, unsigned>{96, 128}, {2816, 1408}, {704, 2816}}) {
    const unsigned K = dims.first, N = dims.second;
    std::vector<int8_t> matrix(K * N);
    std::vector<uint8_t> act(K), packed(hvx_tl2_weight_bytes(K, N));
    std::vector<int32_t> expected(N, 0), actual(N);
    for (unsigned k = 0; k < K; ++k) {
      act[k] = uint8_t((k * 37 + 19) % 256);
      for (unsigned n = 0; n < N; ++n) {
        const int q = int((k * 13 + n * 7 + (k * n) % 17) % 3) - 1;
        matrix[k * N + n] = int8_t(q);
        expected[n] += int(act[k]) * q;
      }
    }
    check(hvx_tl2_pack(matrix.data(), K, N, packed.data()), "pack");
    for (unsigned mode : {0u, 3u, 512u})
      for (unsigned workers : {1u, 8u}) {
        uint32_t stats[6] = {};
        check(nntr_hvx_tl2_gemv_probe(
                session, K, N, mode, workers, 1, act.data(), K, packed.data(),
                int(packed.size()), actual.data(), N, stats, 6),
              "TL2 probe");
        size_t bad = 0;
        for (unsigned n = 0; n < N; ++n)
          bad += actual[n] != expected[n];
        std::printf(
          "TL2_DSP K=%u N=%u mode=%u workers=%u mismatches=%zu total_us=%u\n",
          K, N, mode, workers, bad, stats[4]);
        std::fflush(stdout);
        check(bad != 0, "TL2 integer reference");
      }
  }
  check(nntr_hvx_close(session), "close");
  std::puts("TL2_DSP_PASS");
}

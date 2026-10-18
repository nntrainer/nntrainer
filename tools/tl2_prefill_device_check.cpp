// SPDX-License-Identifier: Apache-2.0
/** @file tl2_prefill_device_check.cpp
 * @brief Native TL2 prefill against independent int4 experts on the DSP.
 * Uses only M>1 calls; does not enter the model's token driver.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug See docs/gemma4-tl2-affine-handoff.md for validation limits.
 */
#include "hvx_gemv_u8tl2.h"
#include <htp_rpcmem.h>
#include <nntr_hvx.h>
#include <remote.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void check(int rc, const char *operation) {
  if (rc) {
    std::fprintf(stderr, "%s: 0x%x\n", operation, unsigned(rc));
    std::exit(1);
  }
}

static unsigned round4k(unsigned bytes) { return (bytes + 4095u) & ~4095u; }

static void check_shape(remote_handle64 session, unsigned K, unsigned I,
                        unsigned N, int magnitude, unsigned experts) {
  const unsigned activation = K == 2048 && I == 1536 && N == 2048 ? 0u : 1u;
  const unsigned gu_bytes = round4k(K * 2 * I / 4 + 8 * 2 * I);
  const unsigned dn_bytes = round4k(I * N / 4 + 8 * N);
  nntrainer::HtpRpcBuffer arena_buffer(experts * (gu_bytes + dn_bytes),
                                       nntrainer::HTP_RPC_FLAGS_UNCACHED);
  check(!arena_buffer.isIon(), "ION allocation");
  auto &api = nntrainer::HtpRpcMemApi::get();
  const int fd = arena_buffer.fd();
  check(api.mmap(CDSP_DOMAIN_ID, fd, arena_buffer.data(), 0,
                 arena_buffer.size(), FASTRPC_MAP_FD),
        "map");
  uint32_t arena;
  check(nntr_hvx_arena_attach(session, fd, arena_buffer.size(), &arena),
        "attach");
  // Reference GU[], DN[], then native GU[], DN[].
  std::vector<uint32_t> handles(4 * experts);
  for (unsigned h = 0; h < 2 * experts; ++h) {
    const unsigned expert = h % experts;
    const bool down = h >= experts;
    const unsigned wk = down ? I : K, wn = down ? N : 2 * I;
    std::vector<int8_t> ternary(size_t(wk) * wn), dense(ternary.size());
    std::vector<float> scale(wn), bias(wn, 0);
    std::vector<int32_t> sums(wn, 0);
    for (unsigned n = 0; n < wn; ++n) {
      scale[n] = .001f * (1 + n % 7) / magnitude;
      for (unsigned k = 0; k < wk; ++k) {
        const int q =
          int((k * 13 + n * 7 + (k * n) % 17 + expert * 11) % 3) - 1;
        ternary[k * wn + n] = int8_t(q);
        dense[k * wn + n] = int8_t(q * magnitude);
        sums[n] += q * magnitude;
      }
    }
    check(nntr_hvx_weight_register_u8i4(
            session, wk, wn, dense.data(), dense.size(), scale.data(), wn,
            sums.data(), wn, bias.data(), wn, handles.data() + h),
          "int4 register");
    const unsigned offset =
      expert * (gu_bytes + dn_bytes) + (down ? gu_bytes : 0);
    std::vector<uint8_t> packed(hvx_tl2_weight_bytes(wk, wn));
    check(hvx_tl2_pack(ternary.data(), wk, wn, packed.data()), "TL2 pack");
    std::memcpy(arena_buffer.data() + offset, packed.data(), packed.size());
    const int8_t palette[4] = {int8_t(-magnitude), 0, int8_t(magnitude),
                               int8_t(magnitude)};
    check(nntr_hvx_weight_register_u2i4_arena(
            session, wk, wn, arena, offset, palette, 4, scale.data(), wn,
            sums.data(), wn, bias.data(), wn, handles.data() + h + 2 * experts),
          "TL2 register");
  }
  check(
    nntr_hvx_weights_tl2(session, handles.data() + 2 * experts, 2 * experts, 1),
    "TL2 enable");
  for (unsigned M : {2u, 23u, 65u, 129u}) {
    std::vector<float> input(M * K), expected(M * N), actual(M * N);
    std::vector<uint32_t> rows, counts(experts);
    std::vector<float> routing;
    for (unsigned e = 0; e < experts; ++e) {
      counts[e] = M - e * (M / (experts + 1));
      for (unsigned r = 0; r < counts[e]; ++r) {
        rows.push_back((M - 1 - r + e) % M);
        routing.push_back(.25f * (1 + (r + e) % 4));
      }
    }
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = float(int((i * 17) % 127) - 63) / 31.f;
    for (unsigned pass = 0; pass < 3; ++pass) {
      for (unsigned mode = 0; mode < 2; ++mode) {
        float *out = mode ? actual.data() : expected.data();
        uint32_t *h = handles.data() + 2 * experts * mode;
        const auto start = std::chrono::steady_clock::now();
        check(nntr_hvx_mm_u8i4_moe_layer(
                session, M, K, I, N, activation, h, experts, h + experts,
                experts, rows.data(), rows.size(), counts.data(), experts,
                routing.data(), routing.size(), input.data(), input.size(), out,
                actual.size()),
              "prefill MoE");
        const double us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        std::printf("TL2_PREFILL K=%u I=%u N=%u M=%u E=%u magnitude=%d "
                    "pass=%u native=%u host_us=%.1f\n",
                    K, I, N, M, experts, magnitude, pass, mode, us);
        std::fflush(stdout);
      }
      for (float f : actual)
        check(!std::isfinite(f), "nonfinite output");
      check(std::memcmp(expected.data(), actual.data(), actual.size() * 4),
            "native/int4 output mismatch");
    }
    std::printf("TL2_PREFILL_EXACT K=%u I=%u N=%u M=%u E=%u magnitude=%d\n", K,
                I, N, M, experts, magnitude);
  }
  for (auto h : handles)
    check(nntr_hvx_weight_release_u8i4(session, h), "release");
  check(nntr_hvx_arena_detach(session, arena), "detach");
  check(
    api.munmap(CDSP_DOMAIN_ID, fd, arena_buffer.data(), arena_buffer.size()),
    "unmap");
}

int main() {
  remote_rpc_thread_params thread = {CDSP_DOMAIN_ID, -1, 64 * 1024};
  check(remote_session_control(FASTRPC_THREAD_PARAMS, &thread, sizeof(thread)),
        "thread");
  remote_rpc_control_unsigned_module pd = {CDSP_DOMAIN_ID, 1};
  check(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &pd, sizeof(pd)),
        "PD");
  remote_handle64 session;
  const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
  check(nntr_hvx_open(uri.c_str(), &session), "open");
  uint32_t applied;
  check(nntr_hvx_moe_set_opts(session, 0xc703e1u, &applied), "MoE options");
  std::printf("MOE_FLAGS 0x%x\n", applied);
  for (int magnitude : {1, 7}) {
    for (unsigned experts : {1u, 3u}) {
      check_shape(session, 96, 192, 128, magnitude, experts);
      check_shape(session, 2816, 704, 2816, magnitude, experts);
      check_shape(session, 2048, 1536, 2048, magnitude, experts);
    }
  }
  check(nntr_hvx_close(session), "close");
  std::puts("TL2_PREFILL_DSP_PASS");
}

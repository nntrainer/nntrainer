// SPDX-License-Identifier: Apache-2.0
/** @file tl2_stable_slots_device_check.cpp
 * @brief Reuse native LFM arena handles after replacing codes and metadata.
 * @author NNTrainer contributors
 * @bug Device validation is required for shared-memory cache maintenance.
 */
#include "hvx_gemv_u8tl2.h"
#include <htp_rpcmem.h>
#include <nntr_hvx.h>
#include <remote.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void check(int rc, const char *what) {
  if (rc) {
    std::fprintf(stderr, "%s failed: 0x%x\n", what, unsigned(rc));
    std::exit(1);
  }
}

static void check_shape(remote_handle64 session, const char *name, unsigned K,
                        unsigned I, unsigned N, unsigned activation) {
  constexpr unsigned E = 2;
  std::printf("STABLE_TL2_DSP_START shape=%s K=%u I=%u N=%u activation=%u\n",
              name, K, I, N, activation);
  std::fflush(stdout);
  const unsigned gb = hvx_tl2_weight_bytes(K, 2u * I);
  const unsigned db = hvx_tl2_weight_bytes(I, N);
  const unsigned ge = gb + 16u * I;
  const unsigned gs = (ge + 4095u) & ~4095u;
  const unsigned ds = (db + 8u * N + 4095u) & ~4095u;
  nntrainer::HtpRpcBuffer buffer(E * (gs + ds),
                                 nntrainer::HTP_RPC_FLAGS_UNCACHED);
  check(!buffer.isIon(), "ION");
  std::memset(buffer.data(), 0, buffer.size());
  auto &api = nntrainer::HtpRpcMemApi::get();
  const int fd = buffer.fd();
  check(api.mmap(CDSP_DOMAIN_ID, fd, buffer.data(), 0, buffer.size(),
                 FASTRPC_MAP_FD),
        "map");
  uint32_t arena;
  check(nntr_hvx_arena_attach(session, fd, buffer.size(), &arena), "attach");
  uint32_t old[E] = {UINT32_MAX, UINT32_MAX}, arenas[E] = {arena, arena};
  uint32_t og[E] = {0u, gs + ds}, od[E] = {gs, 2u * gs + ds};
  uint32_t native[2 * E], done = 0;
  int32_t err = 0;
  check(nntr_hvx_weight_swap_batch_tl2_stable_arena(
          session, K, I, N, old, E, old, E, arenas, E, og, E, od, E, native, E,
          native + E, E, &done, &err),
        "stable arena capability / registration");
  check(done != E || err, "stable registration count");
  for (unsigned generation = 1; generation <= 3; ++generation) {
    uint32_t reference[2 * E];
    for (unsigned h = 0; h < 2 * E; ++h) {
      const unsigned e = h % E;
      const bool down = h >= E;
      const unsigned wk = down ? I : K, wn = down ? N : 2u * I;
      std::vector<int8_t> codes(size_t(wk) * wn);
      std::vector<float> scale(wn), bias(wn, 0);
      std::vector<int32_t> sums(wn, 0);
      for (unsigned n = 0; n < wn; ++n) {
        scale[n] = .001f * (1u + (n + generation + e) % 7u);
        for (unsigned k = 0; k < wk; ++k) {
          const int q =
            int((k * 13u + n * 7u + k * n % 17u + e * 11u + generation * 19u) %
                3u) -
            1;
          codes[size_t(k) * wn + n] = int8_t(q);
          sums[n] += q;
        }
      }
      check(nntr_hvx_weight_register_u8i4(
              session, wk, wn, codes.data(), codes.size(), scale.data(), wn,
              sums.data(), wn, bias.data(), wn, reference + h),
            "reference register");
      std::vector<uint8_t> packed(hvx_tl2_weight_bytes(wk, wn));
      check(hvx_tl2_pack(codes.data(), wk, wn, packed.data()), "pack");
      uint8_t *dst = buffer.data() + (down ? od[e] : og[e]);
      std::memcpy(dst, packed.data(), packed.size());
      std::memcpy(dst + packed.size(), scale.data(), 4u * wn);
      std::memcpy(dst + packed.size() + 4u * wn, sums.data(), 4u * wn);
    }
    // No rebind or registration RPC follows these arena writes.
    std::atomic_thread_fence(std::memory_order_release);
    for (unsigned e = 0; e < E; ++e)
      *reinterpret_cast<volatile uint32_t *>(buffer.data() + og[e] + ge) =
        generation;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // Every generation begins on a different entry path. In particular M1
    // must refresh changed metadata, not only consume a warm prefill cache.
    const unsigned row_cases[3][5] = {{2u, 65u, 129u, 1u, 2u},
                                      {1u, 2u, 65u, 129u, 2u},
                                      {129u, 1u, 2u, 65u, 2u}};
    for (unsigned M : row_cases[generation - 1u]) {
      std::vector<float> input(M * K), actual(M * N), expected(M * N);
      std::vector<uint32_t> rows;
      std::vector<float> routing;
      uint32_t counts[E] = {M, M};
      for (unsigned e = 0; e < E; ++e)
        for (unsigned r = 0; r < M; ++r) {
          rows.push_back((M - 1u - r + e) % M);
          routing.push_back(.25f * (1u + (r + e) % 4u));
        }
      for (size_t i = 0; i < input.size(); ++i)
        input[i] = float(int((i * 17u) % 127u) - 63) / 31.f;
      for (unsigned mode = 0; mode < 2; ++mode) {
        const uint32_t *h = mode ? native : reference;
        check(nntr_hvx_mm_u8i4_moe_layer(
                session, M, K, I, N, activation, h, E, h + E, E, rows.data(),
                rows.size(), counts, E, routing.data(), routing.size(),
                input.data(), input.size(),
                mode ? actual.data() : expected.data(), actual.size()),
              "MoE");
      }
      for (float f : actual)
        check(!std::isfinite(f), "nonfinite");
      check(std::memcmp(actual.data(), expected.data(), actual.size() * 4u),
            "stable native / independent int4");
      std::printf("STABLE_TL2_DSP shape=%s activation=%u generation=%u M=%u "
                  "same_handles=1 exact=1\n",
                  name, activation, generation, M);
      std::fflush(stdout);
    }
    for (auto h : reference)
      check(nntr_hvx_weight_release_u8i4(session, h), "reference release");
  }
  for (auto h : native)
    check(nntr_hvx_weight_release_u8i4(session, h), "native release");
  check(nntr_hvx_arena_detach(session, arena), "detach");
  check(api.munmap(CDSP_DOMAIN_ID, fd, buffer.data(), buffer.size()), "unmap");
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
  check(nntr_hvx_moe_set_opts(session, 0xc703e1u, &applied), "options");
  check(applied != 0xc703e1u, "options echo");
  check_shape(session, "LFM", 2048u, 1536u, 2048u, 0u);
  check(nntr_hvx_close(session), "close");
  std::puts("STABLE_TL2_DSP_PASS cases=15");
}

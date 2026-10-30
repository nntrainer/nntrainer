// SPDX-License-Identifier: Apache-2.0
/** @file tl2_swap_device_check.cpp
 * @brief Diagnostic native TL2 expert registration, M=1 and arena swap.
 * No arguments: small shape; "real": Gemma shape; "real fast": also
 * apply the model's 0xcf03e1 MoE options. Device validation is recorded in
 * docs/gemma4-tl2-affine-handoff.md.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug See docs/gemma4-tl2-affine-handoff.md for validation limits.
 */
#include "hvx_gemv_u8tl2.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <htp_rpcmem.h>
#include <nntr_hvx.h>
#include <remote.h>
#include <string>
#include <vector>
static void check(int r, const char *s) {
  if (r) {
    std::fprintf(stderr, "%s: 0x%x\n", s, r);
    std::exit(1);
  }
}
static uint32_t round4k(uint32_t x) { return (x + 4095u) & ~4095u; }
static void record(uint8_t *p, unsigned K, unsigned N, std::vector<float> &sc,
                   std::vector<int32_t> &sum) {
  std::vector<int8_t> q(size_t(K) * N);
  sc.assign(N, .003f);
  sum.assign(N, 0);
  for (unsigned k = 0; k < K; ++k)
    for (unsigned n = 0; n < N; ++n) {
      int v = int((k * 13 + n * 7 + (k * n) % 17) % 3) - 1;
      q[k * N + n] = v;
      sum[n] += v;
    }
  std::vector<uint8_t> packed(hvx_tl2_weight_bytes(K, N));
  check(hvx_tl2_pack(q.data(), K, N, packed.data()), "pack");
  std::memcpy(p, packed.data(), packed.size());
  std::memcpy(p + size_t(K) * N / 4, sc.data(), 4 * N);
  std::memcpy(p + size_t(K) * N / 4 + 4 * N, sum.data(), 4 * N);
}
int main(int argc, char **argv) {
  (void)argv;
  const unsigned K = argc > 1 ? 2816 : 96, I = argc > 1 ? 704 : 64,
                 N = argc > 1 ? 2816 : 128;
  remote_rpc_thread_params tp = {CDSP_DOMAIN_ID, -1, 64 * 1024};
  check(remote_session_control(FASTRPC_THREAD_PARAMS, &tp, sizeof(tp)),
        "thread");
  remote_rpc_control_unsigned_module pd = {CDSP_DOMAIN_ID, 1};
  check(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &pd, sizeof(pd)),
        "PD");
  remote_handle64 session = 0;
  std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
  check(nntr_hvx_open(uri.c_str(), &session), "open");
  uint32_t applied = 0;
  if (argc > 2)
    check(nntr_hvx_moe_set_opts(session, 0xcf03e1u, &applied), "moe opts");
  std::printf("MOE_FLAGS 0x%x\n", applied);
  const unsigned gs = round4k(K * 2 * I / 4 + 8 * 2 * I),
                 ds = round4k(I * N / 4 + 8 * N), pair = gs + ds;
  nntrainer::HtpRpcBuffer buf(2 * pair, nntrainer::HTP_RPC_FLAGS_UNCACHED);
  if (!buf.isIon())
    check(1, "no ION");
  auto &api = nntrainer::HtpRpcMemApi::get();
  int fd = buf.fd();
  check(api.mmap(CDSP_DOMAIN_ID, fd, buf.data(), 0, buf.size(), FASTRPC_MAP_FD),
        "map");
  uint32_t arena = 0;
  check(nntr_hvx_arena_attach(session, fd, buf.size(), &arena), "attach");
  std::vector<float> sg, sd;
  std::vector<int32_t> cg, cd;
  record(buf.data(), K, 2 * I, sg, cg);
  record(buf.data() + gs, I, N, sd, cd);
  std::memcpy(buf.data() + pair, buf.data(), pair);
  const int8_t pal[4] = {-1, 0, 1, 1};
  std::vector<float> bg(2 * I, 0), bd(N, 0);
  uint32_t h[2];
  check(nntr_hvx_weight_register_u2i4_arena(session, K, 2 * I, arena, 0, pal, 4,
                                            sg.data(), sg.size(), cg.data(),
                                            cg.size(), bg.data(), bg.size(), h),
        "register GU");
  check(nntr_hvx_weight_register_u2i4_arena(
          session, I, N, arena, gs, pal, 4, sd.data(), sd.size(), cd.data(),
          cd.size(), bd.data(), bd.size(), h + 1),
        "register DN");
  check(nntr_hvx_weights_tl2(session, h, 2, 1), "TL2 on");
  std::vector<float> input(K), out(N), first(N);
  for (unsigned k = 0; k < K; ++k)
    input[k] = float(int(k % 127) - 63) / 31;
  uint32_t ri = 0, rc = 1;
  float rw = 1;
  for (unsigned it = 0; it < 5; ++it) {
    if (it) {
      uint32_t og = (it % 2) * pair, od = og + gs, done = 0;
      int32_t err = 0;
      uint32_t nh[2];
      std::printf("SWAP_BEGIN K=%u I=%u N=%u iteration=%u\n", K, I, N, it);
      std::fflush(stdout);
      check(nntr_hvx_weight_swap_batch_u2i4_arena(
              session, K, I, N, h, 1, h + 1, 1, &arena, 1, &og, 1, &od, 1, pal,
              4, pal, 4, nh, 1, nh + 1, 1, &done, &err),
            "swap");
      check(err, "swap error");
      check(done != 1, "swap count");
      h[0] = nh[0];
      h[1] = nh[1];
      check(nntr_hvx_weights_tl2(session, h, 2, 1), "retag");
    }
    std::printf("MOE_BEGIN K=%u I=%u N=%u iteration=%u\n", K, I, N, it);
    std::fflush(stdout);
    auto t = std::chrono::steady_clock::now();
    check(nntr_hvx_mm_u8i4_moe_layer(session, 1, K, I, N, 1, h, 1, h + 1, 1,
                                     &ri, 1, &rc, 1, &rw, 1, input.data(), K,
                                     out.data(), N),
          "M1 MoE");
    double us = std::chrono::duration<double, std::micro>(
                  std::chrono::steady_clock::now() - t)
                  .count();
    if (!it)
      first = out;
    check(std::memcmp(first.data(), out.data(), N * 4) != 0, "output changed");
    for (float f : out)
      check(!std::isfinite(f), "nonfinite");
    std::printf(
      "TL2_SWAP_DSP K=%u I=%u N=%u iteration=%u host_us=%.1f bit_identical=1\n",
      K, I, N, it, us);
    std::fflush(stdout);
  }
  for (auto x : h)
    check(nntr_hvx_weight_release_u8i4(session, x), "release");
  check(nntr_hvx_arena_detach(session, arena), "detach");
  check(api.munmap(CDSP_DOMAIN_ID, fd, buf.data(), buf.size()), "unmap");
  check(nntr_hvx_close(session), "close");
  std::puts("TL2_SWAP_DSP_PASS");
}

// SPDX-License-Identifier: Apache-2.0
/**
 * @file tl2_decode_compare.cpp
 * @brief Compare exact matched-weight MoE and affine LM-head decode paths.
 * @author NNTrainer contributors
 * @bug Synthetic weights do not measure whole-model accuracy or storage I/O.
 *
 * Usage: tl2_decode_compare moe [repeats=7]
 *        tl2_decode_compare head [repeats=7] [maximum_parts=64]
 *
 * MoE uses one uncached ION arena for equivalent int4, QS2 and native TL2
 * weights. Each measured pass rotates their execution order and compares the
 * complete float output bitwise. Magnitudes 1/7 and 1/4/8 experts cover the
 * native path's fused and separate gate epilogues at the Gemma4 dimensions.
 *
 * Head compares equivalent int4 and affine WH weights through one resident
 * LM_HEAD graph operation, with 1/8/64 slices of 4096 columns. Full logits
 * must be bit-identical. The maximum run allocates about 710 MiB for both
 * banks; a maximum of 8 parts is a smaller smoke test. No model is loaded.
 *
 * Packing, registration and three warmups are excluded from all timings.
 * MoE DSP timings include quantization, VTCM feed and epilogues. Head op
 * cycles exclude RPC output copying; DSP-call and host times include it.
 * These are per-operation measurements, not whole-token TPS. Head runs
 * without optional RMSNorm/softcap, identically for both weight formats.
 */
#include "htp_graph_desc.h"
#include "htp_wh_palette.h"
#include "hvx_gemv_u8tl2.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <htp_rpcmem.h>
#include <memory>
#include <nntr_hvx.h>
#include <remote.h>
#include <string>
#include <vector>

static void check(int rc, const char *op) {
  if (rc) {
    std::fprintf(stderr, "%s: 0x%x\n", op, unsigned(rc));
    std::exit(1);
  }
}
static unsigned round4k(unsigned n) { return (n + 4095u) & ~4095u; }
static double now_us() {
  return std::chrono::duration<double, std::micro>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}
static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
static uint64_t checksum(const std::vector<float> &a) {
  uint64_t h = 1469598103934665603ull;
  const auto *p = reinterpret_cast<const unsigned char *>(a.data());
  for (size_t i = 0; i < a.size() * 4; ++i)
    h = (h ^ p[i]) * 1099511628211ull;
  return h;
}
struct Arena {
  remote_handle64 s;
  nntrainer::HtpRpcBuffer buffer;
  uint32_t id = 0;
  Arena(remote_handle64 session, unsigned bytes) :
    s(session), buffer(bytes, nntrainer::HTP_RPC_FLAGS_UNCACHED) {
    check(!buffer.isIon(), "ION");
    auto &api = nntrainer::HtpRpcMemApi::get();
    check(api.mmap(CDSP_DOMAIN_ID, buffer.fd(), buffer.data(), 0, buffer.size(),
                   FASTRPC_MAP_FD),
          "map");
    check(nntr_hvx_arena_attach(s, buffer.fd(), buffer.size(), &id), "attach");
    std::memset(buffer.data(), 0, buffer.size());
  }
  ~Arena() {
    check(nntr_hvx_arena_detach(s, id), "detach");
    check(nntrainer::HtpRpcMemApi::get().munmap(CDSP_DOMAIN_ID, buffer.fd(),
                                                buffer.data(), buffer.size()),
          "unmap");
  }
};
static void set_flags(remote_handle64 s, uint32_t flags) {
  uint32_t applied = 0;
  check(nntr_hvx_moe_set_opts(s, flags, &applied), "options");
  check(applied != flags, "option mismatch");
}
static void check_moe(remote_handle64 s, unsigned K, unsigned I, unsigned N,
                      unsigned E, int mag, unsigned repeats) {
  const unsigned gu4 = round4k(K * 2 * I / 2 + 8 * 2 * I),
                 dn4 = round4k(I * N / 2 + 8 * N);
  const unsigned gu2 = round4k(K * 2 * I / 4 + 8 * 2 * I),
                 dn2 = round4k(I * N / 4 + 8 * N);
  const unsigned offsets[3] = {0, E * (gu4 + dn4),
                               E * (gu4 + dn4) + E * (gu2 + dn2)};
  Arena arena(s, E * (gu4 + dn4) + 2 * E * (gu2 + dn2));
  std::vector<uint32_t> handles[3];
  for (auto &h : handles)
    h.resize(2 * E);
  size_t bytes[3] = {0, 0, 0};
  for (unsigned h = 0; h < 2 * E; ++h) {
    const unsigned e = h % E, wk = h >= E ? I : K, wn = h >= E ? N : 2 * I;
    std::vector<int8_t> trit(size_t(wk) * wn), q(trit.size()),
      back(trit.size());
    std::vector<float> scales(wn), bias(wn, 0);
    std::vector<int32_t> sums(wn, 0);
    for (unsigned k = 0; k < wk; ++k)
      for (unsigned n = 0; n < wn; ++n) {
        int v = int((k * 13u + n * 7u + (k * n) % 17u + e * 11u) % 3u) - 1;
        trit[size_t(k) * wn + n] = v;
        q[size_t(k) * wn + n] = v * mag;
        sums[n] += v * mag;
      }
    for (unsigned n = 0; n < wn; ++n)
      scales[n] = .001f * (1 + n % 7) / mag;
    const int8_t pal[4] = {int8_t(-mag), 0, int8_t(mag), int8_t(mag)};
    for (unsigned mode = 0; mode < 3; ++mode) {
      const unsigned gu = mode ? gu2 : gu4, dn = mode ? dn2 : dn4;
      const unsigned off = offsets[mode] + e * (gu + dn) + (h >= E ? gu : 0);
      const size_t raw = mode == 0   ? nntrainer::whBytes(wk, wn)
                         : mode == 1 ? nntrainer::whBytes2(wk, wn)
                                     : hvx_tl2_weight_bytes(wk, wn);
      std::vector<uint8_t> packed(raw);
      if (mode == 0) {
        nntrainer::whPack(q.data(), wk, wn, packed.data());
        nntrainer::whUnpack(packed.data(), wk, wn, back.data());
      } else if (mode == 1) {
        nntrainer::whPack2(q.data(), wk, wn, pal, packed.data());
        nntrainer::whUnpack2(packed.data(), wk, wn, pal, back.data());
      } else {
        check(hvx_tl2_pack(trit.data(), wk, wn, packed.data()), "TL2 pack");
        check(hvx_tl2_unpack(packed.data(), wk, wn, back.data()), "TL2 unpack");
        for (auto &v : back)
          v *= mag;
      }
      check(std::memcmp(q.data(), back.data(), q.size()), "pack roundtrip");
      std::memcpy(arena.buffer.data() + off, packed.data(), raw);
      bytes[mode] += raw;
      if (mode == 0)
        check(nntr_hvx_weight_register_u8i4_arena(
                s, wk, wn, arena.id, off, scales.data(), wn, sums.data(), wn,
                bias.data(), wn, &handles[mode][h]),
              "int4 registration");
      else
        check(nntr_hvx_weight_register_u2i4_arena(
                s, wk, wn, arena.id, off, pal, 4, scales.data(), wn,
                sums.data(), wn, bias.data(), wn, &handles[mode][h]),
              "2bit registration");
    }
  }
  check(nntr_hvx_weights_tl2(s, handles[2].data(), 2 * E, 1), "native marker");
  set_flags(s, 0xcf03e1u);
  std::vector<float> x(K), out[3], routing(E);
  for (auto &o : out)
    o.resize(N);
  std::vector<uint32_t> rows(E, 0), counts(E, 1);
  for (unsigned i = 0; i < K; ++i)
    x[i] = float(int((i * 17) % 127) - 63) / 31.f;
  for (unsigned e = 0; e < E; ++e)
    routing[e] = .25f * (1 + e % 4);
  std::vector<double> host[3], dsp[3], quant[3], mm[3], rq[3], first[3];
  unsigned feed[3] = {}, dma[3] = {};
  const char *names[] = {"wh_int4", "qs2", "native_tl2"};
  for (unsigned pass = 0; pass < repeats + 3; ++pass) {
    for (unsigned j = 0; j < 3; ++j) {
      const unsigned mode = (pass + j) % 3;
      uint32_t st[31] = {};
      const double t0 = now_us();
      check(nntr_hvx_mm_u8i4_moe_layer_timed(
              s, 1, K, I, N, 1, handles[mode].data(), E,
              handles[mode].data() + E, E, rows.data(), E, counts.data(), E,
              routing.data(), E, x.data(), K, out[mode].data(), N, st, 31),
            "decode MoE");
      const double us = now_us() - t0;
      check(st[29] != 1, "expected GEMV path");
      check(st[30] == 0, "expected VTCM feed");
      if (pass >= 3) {
        host[mode].push_back(us);
        dsp[mode].push_back(st[0]);
        quant[mode].push_back(st[1]);
        mm[mode].push_back(st[10]);
        rq[mode].push_back(st[8]);
        first[mode].push_back(st[12]);
      }
      feed[mode] = st[30];
      dma[mode] = st[11];
    }
    check(std::memcmp(out[0].data(), out[1].data(), N * 4), "WH/QS2 mismatch");
    check(std::memcmp(out[0].data(), out[2].data(), N * 4), "WH/TL2 mismatch");
    for (float v : out[0])
      check(!std::isfinite(v), "nonfinite");
  }
  for (unsigned m = 0; m < 3; ++m)
    std::printf(
      "DECODE_COMPARE format=%s K=%u I=%u N=%u E=%u magnitude=%d repeats=%u "
      "exact=1 checksum=%016llx logical_bytes=%zu dsp_median_us=%.1f "
      "host_median_us=%.1f quant_median_us=%.1f mm_median_us=%.1f "
      "requant_median_us=%.1f first_wait_median_us=%.1f feed=%u dma_kb=%u\n",
      names[m], K, I, N, E, mag, repeats, (unsigned long long)checksum(out[m]),
      bytes[m], median(dsp[m]), median(host[m]), median(quant[m]),
      median(mm[m]), median(rq[m]), median(first[m]), feed[m], dma[m]);
  std::fflush(stdout);
  for (auto &hs : handles)
    for (auto h : hs)
      check(nntr_hvx_weight_release_u8i4(s, h), "release");
}

static void head_graph(remote_handle64 s, unsigned K, unsigned partN,
                       const std::vector<uint32_t> &hs, unsigned parts) {
  std::vector<uint32_t> w(htp_graph_words_for(1, 1), 0);
  w[0] = HTP_GRAPH_MAGIC;
  w[1] = HTP_GRAPH_VERSION;
  w[2] = 1;
  w[3] = 1;
  w[4] = K;
  w[5] = partN * parts;
  w[6] = 1;
  uint32_t opn = 0;
  auto *op =
    htp_graph_lfm2_emit(w.data(), &opn, HTP_OP_LM_HEAD, 1, K, partN * parts, 0,
                        1, HTP_GRAPH_KIND_BIT(HTP_OP_LM_HEAD));
  op->n_experts = parts;
  op->feed = HTP_GRAPH_FEED_WH;
  std::copy_n(hs.data(), parts, op->h_gu);
  uint32_t n = 0;
  check(htp_graph_validate(w.data(), w.size(), HTP_GRAPH_KINDS_ALL, &n),
        "local graph validation");
  check(nntr_hvx_graph_init(s, w.data(), w.size(), &n), "head graph init");
}
static void check_head(remote_handle64 s, unsigned maxparts, unsigned repeats) {
  const unsigned K = 2816, N = 4096, group = 8,
                 stride = round4k(K * N / 2 + 12 * N);
  std::vector<std::unique_ptr<Arena>> arenas;
  std::vector<uint32_t> hs[2];
  for (auto &h : hs)
    h.resize(maxparts);
  for (unsigned m = 0; m < 2; ++m)
    for (unsigned first = 0; first < maxparts; first += group) {
      const unsigned count = std::min(group, maxparts - first);
      arenas.emplace_back(new Arena(s, count * stride));
      auto &a = *arenas.back();
      for (unsigned j = 0; j < count; ++j) {
        const unsigned part = first + j;
        std::vector<int8_t> q(size_t(K) * N);
        std::vector<float> sc(N), bias(m ? 2 * N : N, 0);
        std::vector<int32_t> sums(N, 0);
        for (unsigned n = 0; n < N; ++n) {
          const int offset = int((n + part * 3) % 11) - 5;
          sc[n] = .001f * (1 + n % 7);
          if (m)
            bias[N + n] = float(offset);
          for (unsigned k = 0; k < K; ++k) {
            const int raw =
              int((k * 13 + n * 7 + (k * n) % 17 + part * 11) % 5) - 2;
            const int value = m ? raw : raw + offset;
            q[size_t(k) * N + n] = value;
            sums[n] += value;
          }
        }
        std::vector<uint8_t> packed(nntrainer::whBytes(K, N));
        nntrainer::whPack(q.data(), K, N, packed.data());
        std::memcpy(a.buffer.data() + j * stride, packed.data(), packed.size());
        check(nntr_hvx_weight_register_u8i4_arena(
                s, K, N, a.id, j * stride, sc.data(), N, sums.data(), N,
                bias.data(), bias.size(), &hs[m][part]),
              "head register");
      }
    }
  set_flags(s, 0xcf03e1u);
  std::vector<float> x(K);
  for (unsigned i = 0; i < K; ++i)
    x[i] = float(int((i * 17) % 127) - 63) / 31.f;
  for (unsigned parts : {1u, 8u, 64u})
    if (parts <= maxparts) {
      std::vector<float> out[2];
      for (auto &o : out)
        o.resize(parts * N);
      for (unsigned m = 0; m < 2; ++m) {
        head_graph(s, K, N, hs[m], parts);
        std::vector<double> host, dsp, pc;
        for (unsigned pass = 0; pass < repeats + 3; ++pass) {
          uint32_t st[31] = {}, cycles = 0, resume = 0;
          const double t0 = now_us();
          check(nntr_hvx_forward_debug(
                  s, 0, 1, 0, nullptr, 0, nullptr, 0, nullptr, 0, x.data(), K,
                  out[m].data(), out[m].size(), &resume, &cycles, 1, st, 31),
                "head forward");
          check(resume != 1, "head resume");
          if (pass >= 3) {
            host.push_back(now_us() - t0);
            dsp.push_back(st[0]);
            pc.push_back(cycles);
          }
        }
        check(nntr_hvx_graph_release(s), "head graph release");
        std::printf(
          "HEAD_COMPARE format=%s K=%u N=%u parts=%u repeats=%u "
          "weight_bytes=%zu dsp_op_median_pcycles=%.0f "
          "dsp_call_median_us=%.1f host_median_us=%.1f checksum=%016llx\n",
          m ? "wh_affine" : "equivalent_wh_int4", K, parts * N, parts, repeats,
          size_t(parts) * K * N / 2, median(pc), median(dsp), median(host),
          (unsigned long long)checksum(out[m]));
        std::fflush(stdout);
      }
      check(std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4),
            "affine head/reference mismatch");
      for (float v : out[0])
        check(!std::isfinite(v), "head nonfinite");
      std::printf("HEAD_EXACT parts=%u logits=%u exact=1\n", parts, parts * N);
      std::fflush(stdout);
    }
  for (auto &h : hs)
    for (auto v : h)
      check(nntr_hvx_weight_release_u8i4(s, v), "head release");
}
int main(int argc, char **argv) {
  const bool head = argc > 1 && std::string(argv[1]) == "head";
  const unsigned repeats = argc > 2 ? std::max(1, atoi(argv[2])) : 7;
  remote_rpc_thread_params thread = {CDSP_DOMAIN_ID, -1, 64 * 1024};
  check(remote_session_control(FASTRPC_THREAD_PARAMS, &thread, sizeof(thread)),
        "thread");
  remote_rpc_control_unsigned_module pd = {CDSP_DOMAIN_ID, 1};
  check(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &pd, sizeof(pd)),
        "PD");
  remote_handle64 s;
  const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
  check(nntr_hvx_open(uri.c_str(), &s), "open");
  if (head)
    check_head(s, argc > 3 ? std::min(64, std::max(1, atoi(argv[3]))) : 64,
               repeats);
  else
    for (int mag : {1, 7})
      for (unsigned E : {1u, 4u, 8u})
        check_moe(s, 2816, 704, 2816, E, mag, repeats);
  check(nntr_hvx_close(s), "close");
  std::puts("DECODE_COMPARE_PASS");
}

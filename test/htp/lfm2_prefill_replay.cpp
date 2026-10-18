// SPDX-License-Identifier: Apache-2.0
/**
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
/** @file lfm2_prefill_replay.cpp
 * @brief Replay deterministic attention, KV and FP32 router fixtures across
 * baseline/candidate DSP libraries. The output files must match byte for byte.
 */
#include <htp_backend.h>
#include <nntr_hvx.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void check(int result) {
  if (result != 0)
    throw std::runtime_error("FastRPC fixture failed: " +
                             std::to_string(result));
}

template <typename T> static void save(FILE *file, const std::vector<T> &v) {
  if (std::fwrite(v.data(), sizeof(T), v.size(), file) != v.size())
    throw std::runtime_error("Cannot write fixture output");
}

static float sample(unsigned &seed) {
  seed = seed * 1664525u + 1013904223u;
  return (float(seed >> 8) / 16777216.f - 0.5f) * 2.f;
}

static void attention(remote_handle64 handle, FILE *file, unsigned rows,
                      unsigned hd, unsigned window, unsigned kind,
                      float softcap = 0.f, bool use_sinks = false) {
  const unsigned hq = 32, hkv = 8, qw = hq * hd, kw = hkv * hd;
  std::vector<float> q(size_t(rows) * qw), out(q.size());
  std::vector<uint16_t> k(size_t(rows) * kw), v(k.size());
  unsigned seed = 71u + rows + hd + kind;
  for (size_t i = 0; i < q.size(); ++i)
    q[i] = (i / hd) % 17 ? sample(seed) : 0.f;
  for (size_t i = 0; i < k.size(); ++i) {
    __fp16 kh = (i / hd) % 19 ? sample(seed) : 0.f;
    __fp16 vh = (i / hd) % 23 ? sample(seed) : 0.f;
    std::memcpy(k.data() + i, &kh, sizeof(kh));
    std::memcpy(v.data() + i, &vh, sizeof(vh));
  }
  uint32_t cache, stats[8];
  std::vector<float> sinks(use_sinks ? hq : 0, -2.f);
  check(nntr_hvx_kv_register_q(handle, kind, rows + 32, hkv, hd, &cache));
  check(nntr_hvx_attn_q_step(handle, cache, 0, k.data(), k.size(), v.data(),
                             v.size(), rows, 0, rows, hq, window, softcap,
                             q.data(), q.size(), sinks.data(), sinks.size(),
                             out.data(), out.size(), stats, 8));
  for (float value : out)
    if (!std::isfinite(value))
      throw std::runtime_error("Nonfinite attention output");
  save(file, out);
  std::vector<float> decode(qw);
  check(nntr_hvx_attn_q_step(handle, cache, rows, k.data(), kw, v.data(), kw, 1,
                             0, rows + 1, hq, window, 0.f, q.data(), qw,
                             nullptr, 0, decode.data(), qw, stats, 8));
  save(file, decode);
  std::vector<int8_t> kq(size_t(rows + 1) * kw), vq(kq.size());
  std::vector<float> sk(size_t(rows + 1) * hkv), sv(sk.size());
  std::vector<int32_t> cs(sk.size());
  check(nntr_hvx_kv_dump_q(handle, cache, 0, rows + 1, kq.data(), kq.size(),
                           vq.data(), vq.size(), sk.data(), sk.size(),
                           cs.data(), cs.size(), sv.data(), sv.size()));
  save(file, kq);
  save(file, vq);
  save(file, sk);
  save(file, cs);
  save(file, sv);
  check(nntr_hvx_kv_release_q(handle, cache));
  std::printf("ATTENTION rows=%u hd=%u window=%u kind=%u cap=%g sinks=%d OK\n",
              rows, hd, window, kind, softcap, use_sinks);
}

static void router(remote_handle64 handle, FILE *file, unsigned rows,
                   unsigned width, unsigned experts) {
  std::vector<float> x(size_t(rows) * width), w(size_t(width) * experts);
  std::vector<float> out(size_t(rows) * experts);
  unsigned seed = 917 + rows;
  for (float &value : x)
    value = sample(seed);
  for (float &value : w)
    value = sample(seed) / 32.f;
  check(nntr_hvx_router_logits_f32(handle, rows, width, experts, 1.e-5f,
                                   nullptr, 0, w.data(), w.size(), x.data(),
                                   x.size(), 0, 0, nullptr, 0, out.data(),
                                   out.size(), nullptr, 0, nullptr, 0));
  save(file, out);
  std::printf("ROUTER rows=%u width=%u experts=%u OK\n", rows, width, experts);
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  FILE *file = std::fopen(argv[1], "wb");
  if (!file)
    return 2;
  try {
    auto &backend = nntrainer::HtpBackend::global();
    if (!backend.enabled())
      throw std::runtime_error("HTP unavailable");
    const auto handle = static_cast<remote_handle64>(backend.handle());
    for (unsigned rows :
         {1u, 16u, 64u, 65u, 128u, 256u, 257u, 1024u, 2048u, 4096u})
      attention(handle, file, rows, 64, 0, 0);
    attention(handle, file, 256, 64, 128, 0);
    attention(handle, file, 128, 256, 128, 0);
    attention(handle, file, 128, 64, 0, 1);
    attention(handle, file, 128, 64, 0, 0, 5.f);
    attention(handle, file, 128, 64, 0, 0, 0.f, true);
    for (unsigned rows : {1u, 64u, 128u, 1024u, 2048u})
      router(handle, file, rows, 2048, 64);
    router(handle, file, 128, 2880, 128);
    std::fclose(file);
    std::puts("21 fixtures completed; compare replay files across builds");
    return 0;
  } catch (const std::exception &e) {
    std::fclose(file);
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}

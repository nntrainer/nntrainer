// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hexagon_e2e_test.cpp
 * @date	31 August 2026
 * @brief	On-device e2e harness: loads a packed qwen3 image (.hexw/.hexcfg)
 *		into rpcmem, lowers the same op-list the packer used, and runs
 *		it on the DSP through HexagonRunner. Mirrors hexagon_ref_run's
 *		modes so the two outputs compare 1:1. Every stdout line starts
 *		with "E2E ".
 *
 * Usage: hexagon_e2e_test <prefix> --tokens <f> [--chunk N] [--eval]
 *          [--steps N] [--dump-op i --dump-buf b --dump-off o --dump-bytes n
 *           --dump-out <f>]
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "hex_image.h"
#include "hexagon_runner.h"
#include "nntr_htp_common.h"
#include "qwen3_lowering.h"
#include "rpcmem_allocator.h"

using namespace nntrainer::hexagon;

namespace {

struct Opts {
  std::string prefix, tokens, dump_out;
  uint32_t chunk = 0, steps = 0;
  uint32_t dump_op = 0, dump_buf = 0, dump_off = 0, dump_bytes = 0;
  bool eval = false, dump = false;
};

int usage(const char *a0) {
  std::fprintf(stderr,
               "usage: %s <prefix> --tokens <f> [--chunk N] [--eval] "
               "[--steps N] [--dump-op i --dump-buf b --dump-off o "
               "--dump-bytes n --dump-out <f>]\n",
               a0);
  return 2;
}

double log_softmax_at(const float *logits, uint32_t n, uint32_t idx) {
  float mx = logits[0];
  for (uint32_t i = 1; i < n; ++i)
    mx = logits[i] > mx ? logits[i] : mx;
  double sum = 0.0;
  for (uint32_t i = 0; i < n; ++i)
    sum += std::exp((double)logits[i] - mx);
  return (double)logits[idx] - mx - std::log(sum);
}

uint32_t argmax(const float *v, uint32_t n) {
  uint32_t b = 0;
  for (uint32_t i = 1; i < n; ++i)
    if (v[i] > v[b])
      b = i;
  return b;
}

uint64_t now_us() {
  return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2)
    return usage(argv[0]);
  Opts o;
  o.prefix = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 >= argc) {
        usage(argv[0]);
        std::exit(2);
      }
      return argv[++i];
    };
    auto u32 = [&]() { return (uint32_t)strtoul(next(), nullptr, 10); };
    if (a == "--tokens")
      o.tokens = next();
    else if (a == "--chunk")
      o.chunk = u32();
    else if (a == "--steps")
      o.steps = u32();
    else if (a == "--eval")
      o.eval = true;
    else if (a == "--dump-op") {
      o.dump = true;
      o.dump_op = u32();
    } else if (a == "--dump-buf")
      o.dump_buf = u32();
    else if (a == "--dump-off")
      o.dump_off = u32();
    else if (a == "--dump-bytes")
      o.dump_bytes = u32();
    else if (a == "--dump-out")
      o.dump_out = next();
    else
      return usage(argv[0]);
  }
  if (o.tokens.empty() || (o.dump && (o.dump_out.empty() || !o.dump_bytes)))
    return usage(argv[0]);

  try {
    HexModelConfig cfg = read_hexcfg(o.prefix + ".hexcfg");
    HexLoweredGraph g = lower_qwen3(cfg);
    nntr_htp_oplist_header h;
    std::memcpy(&h, g.oplist.data(), sizeof(h));

    RpcmemBuffer weights(g.weights_size), kv(g.kv_size), act(g.act_size);
    if (!weights.valid() || !kv.valid() || !act.valid())
      throw std::runtime_error("rpcmem allocation failed");
    read_file_into(o.prefix + ".hexw", weights.data(), g.weights_size);
    std::memset(kv.data(), 0, g.kv_size);
    std::memset(act.data(), 0, g.act_size);

    auto runner = HexagonRunner::create();
    if (!runner)
      throw std::runtime_error("HexagonRunner::create failed");
    if (runner->init(g.oplist.data(), (uint32_t)g.oplist.size(), weights, kv,
                     act) != 0)
      throw std::runtime_error("HexagonRunner::init failed");
    std::printf("E2E init ok weights=%llu kv=%llu act=%llu n_ops=%u\n",
                (unsigned long long)g.weights_size,
                (unsigned long long)g.kv_size, (unsigned long long)g.act_size,
                h.n_ops);

    std::vector<uint8_t> tokbytes = read_file(o.tokens);
    const uint32_t n_tok = (uint32_t)(tokbytes.size() / 4);
    std::vector<int32_t> tokens(n_tok);
    std::memcpy(tokens.data(), tokbytes.data(), n_tok * 4);
    if (n_tok == 0)
      throw std::runtime_error("empty token file");
    for (int32_t t : tokens)
      if (t < 0 || (uint32_t)t >= cfg.vocab)
        throw std::runtime_error("token id out of vocab range");

    const uint32_t chunk = o.chunk ? o.chunk : cfg.max_chunk;
    if (chunk > cfg.max_chunk)
      throw std::runtime_error("--chunk exceeds max_chunk of the image");
    std::vector<float> logits(cfg.vocab);
    uint32_t step = 0;
    /* One forward = one RPC. Prints the per-step line the log parsers use. */
    auto fwd = [&](const int32_t *t, uint32_t n, uint32_t pos, int top1,
                   double logprob) {
      uint64_t pc = 0;
      const uint64_t t0 = now_us();
      int err = runner->forward(t, n, pos, logits.data(), cfg.vocab, &pc);
      const uint64_t us = now_us() - t0;
      if (err != 0)
        throw std::runtime_error("forward failed (0x" + std::to_string(err) +
                                 ")");
      std::printf("E2E step %u pos=%u n=%u pcycles=%llu us=%llu top1=%d "
                  "logprob=%.6f\n",
                  step++, pos, n, (unsigned long long)pc,
                  (unsigned long long)us, top1, logprob);
    };
    const uint64_t t_start = now_us();

    if (o.dump) {
      if (o.dump_op == 0 || o.dump_op > h.n_ops)
        throw std::runtime_error("--dump-op must be in [1, n_ops]");
      const uint32_t n = n_tok < chunk ? n_tok : chunk;
      std::vector<uint8_t> dump(o.dump_bytes);
      uint64_t pc = 0;
      int err =
        runner->forward_debug(tokens.data(), n, 0, o.dump_op, o.dump_buf,
                              o.dump_off, dump.data(), o.dump_bytes, &pc);
      if (err != 0)
        throw std::runtime_error("forward_debug failed (0x" +
                                 std::to_string(err) + ")");
      write_file(o.dump_out, dump.data(), dump.size());
      std::printf("E2E dump buf=%u off=%u bytes=%u pcycles=%llu -> %s\n",
                  o.dump_buf, o.dump_off, o.dump_bytes, (unsigned long long)pc,
                  o.dump_out.c_str());
    } else if (o.eval) {
      if (n_tok < 2)
        throw std::runtime_error("--eval needs at least 2 tokens");
      uint32_t steps = n_tok - 1;
      if (o.steps && o.steps < steps)
        steps = o.steps;
      double nll = 0.0;
      uint32_t top1_hits = 0;
      for (uint32_t p = 0; p < steps; ++p) {
        /* the per-step line reports the prediction for token p+1 */
        uint64_t pc = 0;
        const uint64_t t0 = now_us();
        int err =
          runner->forward(&tokens[p], 1, p, logits.data(), cfg.vocab, &pc);
        const uint64_t us = now_us() - t0;
        if (err != 0)
          throw std::runtime_error("forward failed (0x" + std::to_string(err) +
                                   ")");
        const double lp =
          log_softmax_at(logits.data(), cfg.vocab, (uint32_t)tokens[p + 1]);
        const uint32_t top1 = argmax(logits.data(), cfg.vocab);
        nll -= lp;
        top1_hits += top1 == (uint32_t)tokens[p + 1];
        std::printf("E2E step %u pos=%u n=1 pcycles=%llu us=%llu top1=%u "
                    "logprob=%.6f\n",
                    p, p, (unsigned long long)pc, (unsigned long long)us, top1,
                    lp);
      }
      std::printf("E2E ppl %.4f steps %u top1 %u wall_ms %llu\n",
                  std::exp(nll / steps), steps, top1_hits,
                  (unsigned long long)((now_us() - t_start) / 1000));
    } else {
      uint32_t pos = 0;
      while (pos < n_tok) {
        const uint32_t n = n_tok - pos < chunk ? n_tok - pos : chunk;
        fwd(&tokens[pos], n, pos, -1, 0.0);
        pos += n;
      }
      const uint32_t steps = o.steps ? o.steps : 16;
      std::string ids;
      int32_t next = (int32_t)argmax(logits.data(), cfg.vocab);
      for (uint32_t s = 0; s < steps; ++s) {
        ids += " " + std::to_string(next);
        if (s + 1 == steps || pos >= cfg.max_seq)
          break;
        fwd(&next, 1, pos++, next, 0.0);
        next = (int32_t)argmax(logits.data(), cfg.vocab);
      }
      std::printf("E2E gen%s\n", ids.c_str());
      std::printf("E2E wall_ms %llu\n",
                  (unsigned long long)((now_us() - t_start) / 1000));
    }
  } catch (const std::exception &e) {
    std::printf("E2E FAIL %s\n", e.what());
    return 1;
  }
  return 0;
}

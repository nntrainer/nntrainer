// SPDX-License-Identifier: Apache-2.0
/**
 * @file	hexagon_ref_run.cpp
 * @date	31 August 2026
 * @brief	x86 reference runner: interprets a packed qwen3 image
 *		(.hexw/.hexcfg from nntr_hexpack) with the scalar ref_* ops,
 *		i.e. the same fp16 + per-token int8 math as the DSP. Serves as
 *		the accuracy oracle for the device harness.
 *
 * Modes (exactly one of --eval / --dump-op / --list-ops / default):
 *   default   : prefill in chunks then greedy-decode --steps tokens, print ids
 *   --eval    : teacher-forced PPL over the token file (one token per step)
 *   --dump-op : run ops [0, i) for the first chunk at pos 0 and write the
 *               output tensor bytes of op i-1 to --dump-out
 *   --list-ops: print index/kind/layer/output ref of every op
 *
 * Usage: hexagon_ref_run <prefix> --tokens <f> [--chunk N] [--eval]
 *          [--steps N] [--dump-op i --dump-out <f>] [--list-ops]
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
#include "nntr_htp_common.h"
#include "qwen3_lowering.h"
#include "ref_ops.h"

using namespace nntrainer::hexagon;

namespace {

const char *kKindName[NNTR_HTP_OP_KIND_COUNT] = {
  "EMBED",    "RMSNORM", "MATMUL_W8A8",   "ROPE",        "ATTN",
  "SILU_MUL", "ADD",     "MATMUL_LOGITS", "MATMUL_W8A16"};

struct Opts {
  std::string prefix, tokens, dump_out;
  uint32_t chunk = 0, steps = 0, dump_op = 0;
  bool eval = false, dump = false, list = false;
};

int usage(const char *a0) {
  std::fprintf(stderr,
               "usage: %s <prefix> --tokens <f> [--chunk N] [--eval] "
               "[--steps N] [--dump-op i --dump-out <f>] [--list-ops]\n",
               a0);
  return 2;
}

uint8_t *alloc128(uint64_t bytes) {
  void *p = aligned_alloc(128, align128(bytes));
  if (!p)
    throw std::runtime_error("out of memory");
  std::memset(p, 0, align128(bytes));
  return static_cast<uint8_t *>(p);
}

/** Output tensor (buf, offset, bytes) of op d as the DSP would address it. */
void out_ref(const nntr_htp_oplist_header &h, const nntr_htp_op_desc &d,
             uint32_t n_tokens, uint32_t &buf, uint32_t &off, uint32_t &bytes) {
  const uint32_t m = d.m ? d.m : n_tokens;
  buf = d.out.buf;
  off = d.out.offset;
  switch (d.kind) {
  case NNTR_HTP_OP_EMBED:
    bytes = m * d.k * 2u;
    break;
  case NNTR_HTP_OP_ROPE: /* in place on q (in0) */
    buf = d.in0.buf;
    off = d.in0.offset;
    bytes = m * h.n_heads * 128u * 2u;
    break;
  case NNTR_HTP_OP_ATTN:
    bytes = m * h.n_heads * 128u * 2u;
    break;
  case NNTR_HTP_OP_MATMUL_LOGITS:
    bytes = d.n * 4u;
    break;
  default:
    bytes = m * d.n * 2u;
    break;
  }
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
    if (a == "--tokens")
      o.tokens = next();
    else if (a == "--chunk")
      o.chunk = (uint32_t)strtoul(next(), nullptr, 10);
    else if (a == "--steps")
      o.steps = (uint32_t)strtoul(next(), nullptr, 10);
    else if (a == "--eval")
      o.eval = true;
    else if (a == "--list-ops")
      o.list = true;
    else if (a == "--dump-op") {
      o.dump = true;
      o.dump_op = (uint32_t)strtoul(next(), nullptr, 10);
    } else if (a == "--dump-out")
      o.dump_out = next();
    else
      return usage(argv[0]);
  }
  if (!o.list && o.tokens.empty())
    return usage(argv[0]);
  if (o.dump && o.dump_out.empty())
    return usage(argv[0]);

  try {
    HexModelConfig cfg = read_hexcfg(o.prefix + ".hexcfg");
    HexLoweredGraph g = lower_qwen3(cfg);
    nntr_htp_oplist_header h;
    std::memcpy(&h, g.oplist.data(), sizeof(h));
    const nntr_htp_op_desc *ops =
      reinterpret_cast<const nntr_htp_op_desc *>(g.oplist.data() + sizeof(h));

    if (o.list) {
      for (uint32_t i = 0; i < h.n_ops; ++i) {
        uint32_t b, off, bytes;
        out_ref(h, ops[i], cfg.max_chunk, b, off, bytes);
        std::printf(
          "OP %u kind=%s layer=%u m=%u k=%u n=%u out=%u:%u bytes=%u\n", i,
          kKindName[ops[i].kind], ops[i].layer, ops[i].m, ops[i].k, ops[i].n, b,
          off, bytes);
      }
      return 0;
    }

    uint8_t *weights = alloc128(g.weights_size);
    read_file_into(o.prefix + ".hexw", weights, g.weights_size);
    uint8_t *kv = alloc128(g.kv_size);
    uint8_t *act = alloc128(g.act_size);

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
    auto fwd = [&](const int32_t *t, uint32_t n, uint32_t pos, uint32_t lim) {
      if (n == 0 || n > cfg.max_chunk || pos + n > cfg.max_seq)
        throw std::runtime_error("forward args out of range");
      ref_graph_forward_upto(g.oplist.data(), weights, kv, act, t, n, pos,
                             logits.data(), lim);
    };
    const auto t0 = std::chrono::steady_clock::now();
    auto wall_ms = [&]() {
      return (unsigned long long)
        std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - t0)
          .count();
    };

    if (o.dump) {
      if (o.dump_op == 0 || o.dump_op > h.n_ops)
        throw std::runtime_error("--dump-op must be in [1, n_ops]");
      const uint32_t n = n_tok < chunk ? n_tok : chunk;
      fwd(tokens.data(), n, 0, o.dump_op);
      uint32_t b, off, bytes;
      out_ref(h, ops[o.dump_op - 1], n, b, off, bytes);
      const uint8_t *src = b == NNTR_HTP_BUF_WEIGHTS ? weights
                           : b == NNTR_HTP_BUF_KV    ? kv
                           : b == NNTR_HTP_BUF_ACT   ? act
                                                     : nullptr;
      if (b == NNTR_HTP_BUF_LOGITS)
        src = reinterpret_cast<const uint8_t *>(logits.data());
      if (!src)
        throw std::runtime_error("op output is not in a dumpable buffer");
      write_file(o.dump_out, src + off, bytes);
      std::printf("DUMP op=%u kind=%s buf=%u off=%u bytes=%u -> %s\n",
                  o.dump_op - 1, kKindName[ops[o.dump_op - 1].kind], b, off,
                  bytes, o.dump_out.c_str());
    } else if (o.eval) {
      if (n_tok < 2)
        throw std::runtime_error("--eval needs at least 2 tokens");
      uint32_t steps = n_tok - 1;
      if (o.steps && o.steps < steps)
        steps = o.steps;
      double nll = 0.0;
      uint32_t top1 = 0;
      for (uint32_t p = 0; p < steps; ++p) {
        fwd(&tokens[p], 1, p, h.n_ops);
        nll -=
          log_softmax_at(logits.data(), cfg.vocab, (uint32_t)tokens[p + 1]);
        top1 += argmax(logits.data(), cfg.vocab) == (uint32_t)tokens[p + 1];
        if ((p + 1) % 16 == 0)
          std::fprintf(stderr, "step %u/%u ppl=%.4f wall_ms=%llu\n", p + 1,
                       steps, std::exp(nll / (p + 1)), wall_ms());
      }
      std::printf("PPL %.4f wall_ms %llu steps %u top1 %u\n",
                  std::exp(nll / steps), wall_ms(), steps, top1);
    } else {
      uint32_t pos = 0;
      while (pos < n_tok) {
        const uint32_t n = n_tok - pos < chunk ? n_tok - pos : chunk;
        fwd(&tokens[pos], n, pos, h.n_ops);
        pos += n;
      }
      const uint32_t steps = o.steps ? o.steps : 16;
      std::printf("GEN");
      int32_t next = (int32_t)argmax(logits.data(), cfg.vocab);
      for (uint32_t s = 0; s < steps; ++s) {
        std::printf(" %d", next);
        if (s + 1 == steps || pos >= cfg.max_seq)
          break;
        fwd(&next, 1, pos++, h.n_ops);
        next = (int32_t)argmax(logits.data(), cfg.vocab);
      }
      std::printf("\nwall_ms %llu\n", wall_ms());
    }
    free(weights);
    free(kv);
    free(act);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "hexagon_ref_run: %s\n", e.what());
    return 1;
  }
  return 0;
}

// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_lowering.cpp
 * @date	19 August 2026
 * @brief	x86 self-check for qwen3 graph_lowering: op-list shape,
 *		WEIGHTS/ACT layout, header fields, and pack_weights() byte
 *		packing (int8/scale memcpy, norm fp16 conversion, RoPE
 *		table), against a tiny config and a qwen3-0.6b-dims smoke
 *		test. Not gtest, mirrors the test_oplist_header.c
 *		self-contained main pattern.
 *
 * Compile:
 *   g++ -std=c++17 -Wall -Werror \
 *       -I nntrainer/tensor/hexagon/htp -I nntrainer/tensor/hexagon/host \
 *       -o /tmp/test_lowering test/hexagon/test_lowering.cpp \
 *       Applications/CausalLM/hexagon/qwen3_lowering.cpp \
 *       nntrainer/tensor/hexagon/host/graph_lowering.cpp \
 *   && /tmp/test_lowering
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "../../Applications/CausalLM/hexagon/hex_image.h"
#include "../../Applications/CausalLM/hexagon/qwen3_lowering.h"
#include "../../nntrainer/tensor/hexagon/host/graph_lowering.h"
#include "../../nntrainer/tensor/hexagon/htp/nntr_htp_common.h"
#include "sim/sim_test_util.h"

using nntrainer::hexagon::HexLayerWeights;
using nntrainer::hexagon::HexLoweredGraph;
using nntrainer::hexagon::HexModelConfig;
using nntrainer::hexagon::HexModelWeights;
using nntrainer::hexagon::HexWeightOffsets;
using nntrainer::hexagon::lower_qwen3;
using nntrainer::hexagon::pack_weights;
using nntrainer::hexagon::read_hexcfg;
using nntrainer::hexagon::write_hexcfg;

/** @brief Print the failing check and exit 1. */
#define FAIL(msg)                                                              \
  do {                                                                         \
    std::fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);       \
    std::exit(1);                                                              \
  } while (0)

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond))                                                               \
      FAIL(msg);                                                               \
  } while (0)

namespace {

uint32_t f32_bits(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

uint64_t align128(uint64_t x) {
  return (x + 127ull) & ~static_cast<uint64_t>(127ull);
}

nntr_htp_oplist_header read_header(const HexLoweredGraph &g) {
  nntr_htp_oplist_header h;
  std::memcpy(&h, g.oplist.data(), sizeof(h));
  return h;
}

nntr_htp_op_desc read_op(const HexLoweredGraph &g, uint32_t i) {
  nntr_htp_op_desc d;
  uint64_t off =
    sizeof(nntr_htp_oplist_header) + static_cast<uint64_t>(i) * sizeof(d);
  std::memcpy(&d, g.oplist.data() + off, sizeof(d));
  return d;
}

/** @brief 16 op kinds of one transformer layer block, plan table order. */
const uint32_t kLayerKinds[16] = {
  NNTR_HTP_OP_RMSNORM,     NNTR_HTP_OP_MATMUL_W8A8, NNTR_HTP_OP_MATMUL_W8A8,
  NNTR_HTP_OP_MATMUL_W8A8, NNTR_HTP_OP_RMSNORM,     NNTR_HTP_OP_RMSNORM,
  NNTR_HTP_OP_ROPE,        NNTR_HTP_OP_ATTN,        NNTR_HTP_OP_MATMUL_W8A8,
  NNTR_HTP_OP_ADD,         NNTR_HTP_OP_RMSNORM,     NNTR_HTP_OP_MATMUL_W8A8,
  NNTR_HTP_OP_MATMUL_W8A8, NNTR_HTP_OP_SILU_MUL,    NNTR_HTP_OP_MATMUL_W8A16,
  NNTR_HTP_OP_ADD,
};

/** @brief Check the op sequence kinds/params for one config's graph. */
void check_sequence(const HexLoweredGraph &g, const HexModelConfig &cfg) {
  const uint32_t n_q = cfg.n_heads * cfg.head_dim;
  const uint32_t n_kv = cfg.n_kv_heads * cfg.head_dim;
  const uint32_t eps_bits = f32_bits(cfg.rms_eps);
  const uint32_t n_ops = 1u + 16u * cfg.n_layers + 2u;

  CHECK(read_header(g).n_ops == n_ops, "n_ops mismatch");
  CHECK(g.oplist.size() == sizeof(nntr_htp_oplist_header) +
                             (uint64_t)n_ops * sizeof(nntr_htp_op_desc),
        "oplist byte size mismatch");

  nntr_htp_op_desc embed = read_op(g, 0);
  CHECK(embed.kind == NNTR_HTP_OP_EMBED, "op0 must be EMBED");
  CHECK(embed.k == cfg.hidden, "EMBED k != hidden");

  uint32_t idx = 1;
  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    for (uint32_t j = 0; j < 16; ++j, ++idx) {
      nntr_htp_op_desc d = read_op(g, idx);
      CHECK(d.kind == kLayerKinds[j], "layer op kind mismatch");

      switch (j) {
      case 0:  // L.1 RMSNORM (attn_norm)
      case 10: // L.11 RMSNORM (ffn_norm)
        CHECK(d.param0 == eps_bits, "RMSNORM param0 != eps bits");
        CHECK(!(d.flags & NNTR_HTP_FLAG_PER_HEAD), "unexpected PER_HEAD");
        break;
      case 4: // L.5 RMSNORM (q_norm)
        CHECK(d.param0 == eps_bits, "RMSNORM param0 != eps bits");
        CHECK(d.flags & NNTR_HTP_FLAG_PER_HEAD, "q_norm missing PER_HEAD");
        CHECK(d.n == n_q, "q_norm n != n_heads*head_dim");
        break;
      case 5: // L.6 RMSNORM (k_norm)
        CHECK(d.param0 == eps_bits, "RMSNORM param0 != eps bits");
        CHECK(d.flags & NNTR_HTP_FLAG_PER_HEAD, "k_norm missing PER_HEAD");
        CHECK(d.n == n_kv, "k_norm n != n_kv_heads*head_dim");
        break;
      case 1: // L.2 MATMUL wq
        CHECK(d.k == cfg.hidden && d.n == n_q, "wq k,n mismatch");
        break;
      case 2: // L.3 MATMUL wk
      case 3: // L.4 MATMUL wv
        CHECK(d.k == cfg.hidden && d.n == n_kv, "wk/wv k,n mismatch");
        break;
      case 7: // L.8 ATTN
        CHECK(d.layer == l, "ATTN layer field mismatch");
        break;
      case 8: // L.9 MATMUL wo
        CHECK(d.k == n_q && d.n == cfg.hidden, "wo k,n mismatch");
        break;
      case 11: // L.12 MATMUL gate
      case 12: // L.13 MATMUL up
        CHECK(d.k == cfg.hidden && d.n == cfg.ffn, "gate/up k,n mismatch");
        break;
      case 14: // L.15 MATMUL down
        CHECK(d.k == cfg.ffn && d.n == cfg.hidden, "down k,n mismatch");
        break;
      default:
        break;
      }
    }
  }

  nntr_htp_op_desc final_norm = read_op(g, idx++);
  CHECK(final_norm.kind == NNTR_HTP_OP_RMSNORM, "final norm kind");
  CHECK(final_norm.param0 == eps_bits, "final norm param0 != eps bits");

  nntr_htp_op_desc logits = read_op(g, idx++);
  CHECK(logits.kind == NNTR_HTP_OP_MATMUL_LOGITS, "last op != LOGITS");
  CHECK(logits.m == 1u, "MATMUL_LOGITS m != 1");
  CHECK(logits.k == cfg.hidden && logits.n == cfg.vocab, "LOGITS k,n");
  CHECK(logits.out.buf == NNTR_HTP_BUF_LOGITS, "LOGITS out buf");
  CHECK(idx == n_ops, "op count walked mismatch");

  // Tied sharing: LOGITS in1/in2 must reuse EMBED's in1/in2 offsets.
  CHECK(logits.in1.offset == embed.in1.offset, "tied embed offset");
  CHECK(logits.in2.offset == embed.in2.offset, "tied embed_scale offset");
}

/** @brief Check ACT slot offsets (read back from op refs) are disjoint. */
void check_act_disjoint(const HexLoweredGraph &g, const HexModelConfig &cfg) {
  const uint64_t mc = cfg.max_chunk;
  const uint64_t n_q = static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim;
  const uint64_t n_kv = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;

  // Slot offsets are read back from the first layer's op refs (op 1..16).
  nntr_htp_op_desc l1 = read_op(g, 1);    // out -> t
  nntr_htp_op_desc l2 = read_op(g, 2);    // out -> q
  nntr_htp_op_desc l3 = read_op(g, 3);    // out -> kb
  nntr_htp_op_desc l4 = read_op(g, 4);    // out -> vb
  nntr_htp_op_desc l8 = read_op(g, 8);    // out -> ao
  nntr_htp_op_desc l9 = read_op(g, 9);    // out -> h2
  nntr_htp_op_desc l12 = read_op(g, 12);  // out -> g
  nntr_htp_op_desc l13 = read_op(g, 13);  // out -> u
  nntr_htp_op_desc embed = read_op(g, 0); // out -> x

  struct Slot {
    uint64_t off, size;
  };
  Slot slots[9] = {
    {embed.out.offset, mc * cfg.hidden * 2u}, // x
    {l1.out.offset, mc * cfg.hidden * 2u},    // t
    {l2.out.offset, mc * n_q * 2u},           // q
    {l3.out.offset, mc * n_kv * 2u},          // kb
    {l4.out.offset, mc * n_kv * 2u},          // vb
    {l8.out.offset, mc * n_q * 2u},           // ao
    {l9.out.offset, mc * cfg.hidden * 2u},    // h2
    {l12.out.offset, mc * cfg.ffn * 2u},      // g
    {l13.out.offset, mc * cfg.ffn * 2u},      // u
  };

  for (int i = 0; i < 9; ++i) {
    CHECK(slots[i].off + slots[i].size <= g.act_size, "ACT slot exceeds size");
    for (int j = i + 1; j < 9; ++j) {
      bool disjoint = slots[i].off + slots[i].size <= slots[j].off ||
                      slots[j].off + slots[j].size <= slots[i].off;
      CHECK(disjoint, "ACT slots overlap");
    }
  }
}

/** @brief Recompute expected WEIGHTS byte size independently of the impl. */
uint64_t expected_weights_size(const HexModelConfig &cfg) {
  uint64_t cur = 0;
  auto add = [&](uint64_t bytes) { cur = align128(cur) + bytes; };
  const uint64_t n_q = static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim;
  const uint64_t n_kv = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;

  add(static_cast<uint64_t>(cfg.vocab) * cfg.hidden);  // embed
  add(static_cast<uint64_t>(cfg.vocab) * 4u);          // embed_scale
  add(static_cast<uint64_t>(cfg.max_seq) * 128u * 2u); // rope_table
  add(static_cast<uint64_t>(cfg.hidden) * 2u);         // final_norm

  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    add(n_q * cfg.hidden);                            // wq
    add(n_q * 4u);                                    // wq_s
    add(n_kv * cfg.hidden);                           // wk
    add(n_kv * 4u);                                   // wk_s
    add(n_kv * cfg.hidden);                           // wv
    add(n_kv * 4u);                                   // wv_s
    add(static_cast<uint64_t>(cfg.hidden) * n_q);     // wo
    add(static_cast<uint64_t>(cfg.hidden) * 4u);      // wo_s
    add(static_cast<uint64_t>(cfg.ffn) * cfg.hidden); // gate
    add(static_cast<uint64_t>(cfg.ffn) * 4u);         // gate_s
    add(static_cast<uint64_t>(cfg.ffn) * cfg.hidden); // up
    add(static_cast<uint64_t>(cfg.ffn) * 4u);         // up_s
    add(static_cast<uint64_t>(cfg.hidden) * cfg.ffn); // down
    add(static_cast<uint64_t>(cfg.hidden) * 4u);      // down_s
    add(static_cast<uint64_t>(cfg.hidden) * 2u);      // attn_norm
    add(static_cast<uint64_t>(cfg.hidden) * 2u);      // ffn_norm
    add(static_cast<uint64_t>(cfg.head_dim) * 2u);    // q_norm
    add(static_cast<uint64_t>(cfg.head_dim) * 2u);    // k_norm
  }
  return cur;
}

/** @brief Synthetic source weights for pack_weights() packing tests. */
struct SynthLayer {
  std::vector<int8_t> wq, wk, wv, wo, gate, up, down;
  std::vector<float> wq_s, wk_s, wv_s, wo_s, gate_s, up_s, down_s;
  std::vector<float> attn_norm, ffn_norm, q_norm, k_norm;
};

/** @brief Full synthetic model: embed + per-layer SynthLayer blocks. */
struct SynthWeights {
  std::vector<int8_t> embed;
  std::vector<float> embed_s;
  std::vector<float> final_norm;
  std::vector<SynthLayer> layers;
};

/** @brief proj int8 pattern: buf[i] = (int8_t)(i*31 + ord*7); ord tags
 *         a tensor so distinct tensors get distinct byte patterns. */
void fill_i8(int8_t *buf, uint64_t n, uint32_t ord) {
  for (uint64_t i = 0; i < n; ++i)
    buf[i] = static_cast<int8_t>(i * 31u + ord * 7u);
}

/** @brief Per-row (per-output-channel) scale pattern. */
void fill_scale(float *buf, uint64_t rows) {
  for (uint64_t r = 0; r < rows; ++r)
    buf[r] = 0.001f + 0.0001f * static_cast<float>(r);
}

/** @brief Norm gamma pattern: LCG-driven fp32 values in [0.5, 1.5). */
void fill_norm(float *buf, uint64_t n) {
  for (uint64_t i = 0; i < n; ++i)
    buf[i] = 1.0f + 0.5f * frand();
}

/** @brief Build deterministic synthetic weights matching cfg's shapes. */
SynthWeights make_synth_weights(const HexModelConfig &cfg) {
  SynthWeights s;
  const uint64_t n_q = static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim;
  const uint64_t n_kv = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
  uint32_t ord = 0;

  s.embed.resize(static_cast<uint64_t>(cfg.vocab) * cfg.hidden);
  fill_i8(s.embed.data(), s.embed.size(), ord++);
  s.embed_s.resize(cfg.vocab);
  fill_scale(s.embed_s.data(), s.embed_s.size());
  s.final_norm.resize(cfg.hidden);
  fill_norm(s.final_norm.data(), s.final_norm.size());

  s.layers.resize(cfg.n_layers);
  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    SynthLayer &ly = s.layers[l];

    ly.wq.resize(n_q * cfg.hidden);
    fill_i8(ly.wq.data(), ly.wq.size(), ord++);
    ly.wq_s.resize(n_q);
    fill_scale(ly.wq_s.data(), ly.wq_s.size());

    ly.wk.resize(n_kv * cfg.hidden);
    fill_i8(ly.wk.data(), ly.wk.size(), ord++);
    ly.wk_s.resize(n_kv);
    fill_scale(ly.wk_s.data(), ly.wk_s.size());

    ly.wv.resize(n_kv * cfg.hidden);
    fill_i8(ly.wv.data(), ly.wv.size(), ord++);
    ly.wv_s.resize(n_kv);
    fill_scale(ly.wv_s.data(), ly.wv_s.size());

    ly.wo.resize(static_cast<uint64_t>(cfg.hidden) * n_q);
    fill_i8(ly.wo.data(), ly.wo.size(), ord++);
    ly.wo_s.resize(cfg.hidden);
    fill_scale(ly.wo_s.data(), ly.wo_s.size());

    ly.gate.resize(static_cast<uint64_t>(cfg.ffn) * cfg.hidden);
    fill_i8(ly.gate.data(), ly.gate.size(), ord++);
    ly.gate_s.resize(cfg.ffn);
    fill_scale(ly.gate_s.data(), ly.gate_s.size());

    ly.up.resize(static_cast<uint64_t>(cfg.ffn) * cfg.hidden);
    fill_i8(ly.up.data(), ly.up.size(), ord++);
    ly.up_s.resize(cfg.ffn);
    fill_scale(ly.up_s.data(), ly.up_s.size());

    ly.down.resize(static_cast<uint64_t>(cfg.hidden) * cfg.ffn);
    fill_i8(ly.down.data(), ly.down.size(), ord++);
    ly.down_s.resize(cfg.hidden);
    fill_scale(ly.down_s.data(), ly.down_s.size());

    ly.attn_norm.resize(cfg.hidden);
    fill_norm(ly.attn_norm.data(), ly.attn_norm.size());
    ly.ffn_norm.resize(cfg.hidden);
    fill_norm(ly.ffn_norm.data(), ly.ffn_norm.size());
    ly.q_norm.resize(cfg.head_dim);
    fill_norm(ly.q_norm.data(), ly.q_norm.size());
    ly.k_norm.resize(cfg.head_dim);
    fill_norm(ly.k_norm.data(), ly.k_norm.size());
  }
  return s;
}

/** @brief Wire SynthWeights storage into non-owning HexModelWeights. */
HexModelWeights to_model_weights(const SynthWeights &s) {
  HexModelWeights w{};
  w.embed = s.embed.data();
  w.embed_s = s.embed_s.data();
  w.final_norm = s.final_norm.data();
  w.layers.resize(s.layers.size());
  for (size_t l = 0; l < s.layers.size(); ++l) {
    const SynthLayer &ly = s.layers[l];
    HexLayerWeights &lw = w.layers[l];
    lw.wq = ly.wq.data();
    lw.wq_s = ly.wq_s.data();
    lw.wk = ly.wk.data();
    lw.wk_s = ly.wk_s.data();
    lw.wv = ly.wv.data();
    lw.wv_s = ly.wv_s.data();
    lw.wo = ly.wo.data();
    lw.wo_s = ly.wo_s.data();
    lw.w_gate = ly.gate.data();
    lw.w_gate_s = ly.gate_s.data();
    lw.w_up = ly.up.data();
    lw.w_up_s = ly.up_s.data();
    lw.w_down = ly.down.data();
    lw.w_down_s = ly.down_s.data();
    lw.attn_norm = ly.attn_norm.data();
    lw.ffn_norm = ly.ffn_norm.data();
    lw.q_norm = ly.q_norm.data();
    lw.k_norm = ly.k_norm.data();
  }
  return w;
}

/** @brief One (offset, size) tensor extent, WEIGHTS layout order. */
using Extent = std::pair<uint64_t, uint64_t>;

/** @brief Ordered tensor extents (offset,size), mirroring the WEIGHTS
 *         layout section, for the full-coverage (check 5) sweep. */
std::vector<Extent> tensor_extents(const HexLoweredGraph &g,
                                   const HexModelConfig &cfg) {
  std::vector<Extent> v;
  const uint64_t n_q = static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim;
  const uint64_t n_kv = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;
  auto add = [&](uint32_t off, uint64_t size) { v.push_back({off, size}); };

  add(g.woff.embed, static_cast<uint64_t>(cfg.vocab) * cfg.hidden);
  add(g.woff.embed_scale, static_cast<uint64_t>(cfg.vocab) * 4u);
  add(g.woff.rope_table, static_cast<uint64_t>(cfg.max_seq) * 128u * 2u);
  add(g.woff.final_norm, static_cast<uint64_t>(cfg.hidden) * 2u);

  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    const HexWeightOffsets::PerLayer &pl = g.woff.layers[l];
    add(pl.wq, n_q * cfg.hidden);
    add(pl.wq_s, n_q * 4u);
    add(pl.wk, n_kv * cfg.hidden);
    add(pl.wk_s, n_kv * 4u);
    add(pl.wv, n_kv * cfg.hidden);
    add(pl.wv_s, n_kv * 4u);
    add(pl.wo, static_cast<uint64_t>(cfg.hidden) * n_q);
    add(pl.wo_s, static_cast<uint64_t>(cfg.hidden) * 4u);
    add(pl.gate, static_cast<uint64_t>(cfg.ffn) * cfg.hidden);
    add(pl.gate_s, static_cast<uint64_t>(cfg.ffn) * 4u);
    add(pl.up, static_cast<uint64_t>(cfg.ffn) * cfg.hidden);
    add(pl.up_s, static_cast<uint64_t>(cfg.ffn) * 4u);
    add(pl.down, static_cast<uint64_t>(cfg.hidden) * cfg.ffn);
    add(pl.down_s, static_cast<uint64_t>(cfg.hidden) * 4u);
    add(pl.attn_norm, static_cast<uint64_t>(cfg.hidden) * 2u);
    add(pl.ffn_norm, static_cast<uint64_t>(cfg.hidden) * 2u);
    add(pl.q_norm, static_cast<uint64_t>(cfg.head_dim) * 2u);
    add(pl.k_norm, static_cast<uint64_t>(cfg.head_dim) * 2u);
  }
  return v;
}

/** @brief True iff every byte in [p, p+n) equals 0xA5. */
bool all_0xA5(const uint8_t *p, uint64_t n) {
  for (uint64_t i = 0; i < n; ++i)
    if (p[i] != 0xA5u)
      return false;
  return true;
}

/** @brief memcmp check 1: byte-exact int8 blob / fp32 scale copy. */
void check_bytes(const uint8_t *dst, uint32_t off, const void *src,
                 uint64_t bytes, const char *msg) {
  CHECK(std::memcmp(dst + off, src, bytes) == 0, msg);
}

/** @brief Read one fp16 (as raw bits) back from the packed buffer. */
uint16_t read_u16(const uint8_t *dst, uint32_t off, uint64_t idx) {
  uint16_t v;
  std::memcpy(&v, dst + off + idx * 2u, 2);
  return v;
}

/** @brief IEEE fp16 bits -> fp32 (integer arithmetic, no _Float16). */
float f16_bits_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else { /* subnormal: normalize */
      int e = 127 - 15 + 1;
      while (!(mant & 0x400u)) {
        mant <<= 1;
        --e;
      }
      mant &= 0x3ffu;
      bits = sign | (static_cast<uint32_t>(e) << 23) | (mant << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

/** @brief fp16 bits vs fp32 ref within |d| <= 1e-3 + 1e-3*|ref|. */
bool close_norm(uint16_t bits, float ref) {
  float got = f16_bits_to_f32(bits);
  float d = std::fabs(got - ref);
  return d <= 1e-3f + 1e-3f * std::fabs(ref);
}

/** @brief check 2: a packed fp16 norm vector matches its fp32 source. */
void check_norm_vec(const uint8_t *dst, uint32_t off, const float *ref,
                    uint64_t n, const char *msg) {
  for (uint64_t i = 0; i < n; ++i)
    CHECK(close_norm(read_u16(dst, off, i), ref[i]), msg);
}

/** @brief fp16 bits vs fp32 ref within |d| <= 2e-3 + 5e-3*|ref|. */
bool close_rope(uint16_t bits, float ref) {
  float got = f16_bits_to_f32(bits);
  float d = std::fabs(got - ref);
  return d <= 2e-3f + 5e-3f * std::fabs(ref);
}

/** @brief check 3: RoPE cos/sin fp16 rows against cosf/sinf directly,
 *         for p in {0, 1, max_seq-1} and i in {0, 1, 63}. */
void check_rope_table(const uint8_t *dst, const HexLoweredGraph &g,
                      const HexModelConfig &cfg) {
  const uint32_t ps[3] = {0u, 1u, cfg.max_seq - 1u};
  const uint32_t is[3] = {0u, 1u, 63u};
  for (uint32_t p : ps) {
    for (uint32_t i : is) {
      float exponent = -2.0f * static_cast<float>(i) / 128.0f;
      float angle = static_cast<float>(p) * powf(cfg.rope_theta, exponent);
      uint64_t row = static_cast<uint64_t>(p) * 128u;
      uint16_t cos_bits = read_u16(dst, g.woff.rope_table, row + i);
      uint16_t sin_bits = read_u16(dst, g.woff.rope_table, row + 64u + i);
      CHECK(close_rope(cos_bits, cosf(angle)), "rope cos mismatch");
      CHECK(close_rope(sin_bits, sinf(angle)), "rope sin mismatch");
    }
  }
}

/** @brief pack_weights() checks 1-5: memcmp coverage, norm/RoPE
 *         conversion accuracy, tied-embed size accounting, and
 *         full-coverage writtenness against an 0xA5 prefill. */
void check_pack_weights(const HexLoweredGraph &g, const HexModelConfig &cfg) {
  SynthWeights synth = make_synth_weights(cfg);
  HexModelWeights w = to_model_weights(synth);

  std::vector<uint8_t> dst(g.weights_size, 0xA5u);
  pack_weights(g, cfg, w, dst.data());

  // 1. int8 blobs and fp32 scale arrays: byte-exact vs. source.
  check_bytes(dst.data(), g.woff.embed, synth.embed.data(), synth.embed.size(),
              "embed bytes");
  check_bytes(dst.data(), g.woff.embed_scale, synth.embed_s.data(),
              synth.embed_s.size() * 4u, "embed_scale bytes");
  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    const HexWeightOffsets::PerLayer &pl = g.woff.layers[l];
    const SynthLayer &ly = synth.layers[l];
    check_bytes(dst.data(), pl.wq, ly.wq.data(), ly.wq.size(), "wq");
    check_bytes(dst.data(), pl.wq_s, ly.wq_s.data(), ly.wq_s.size() * 4u,
                "wq_s");
    check_bytes(dst.data(), pl.wk, ly.wk.data(), ly.wk.size(), "wk");
    check_bytes(dst.data(), pl.wk_s, ly.wk_s.data(), ly.wk_s.size() * 4u,
                "wk_s");
    check_bytes(dst.data(), pl.wv, ly.wv.data(), ly.wv.size(), "wv");
    check_bytes(dst.data(), pl.wv_s, ly.wv_s.data(), ly.wv_s.size() * 4u,
                "wv_s");
    check_bytes(dst.data(), pl.wo, ly.wo.data(), ly.wo.size(), "wo");
    check_bytes(dst.data(), pl.wo_s, ly.wo_s.data(), ly.wo_s.size() * 4u,
                "wo_s");
    check_bytes(dst.data(), pl.gate, ly.gate.data(), ly.gate.size(), "gate");
    check_bytes(dst.data(), pl.gate_s, ly.gate_s.data(), ly.gate_s.size() * 4u,
                "gate_s");
    check_bytes(dst.data(), pl.up, ly.up.data(), ly.up.size(), "up");
    check_bytes(dst.data(), pl.up_s, ly.up_s.data(), ly.up_s.size() * 4u,
                "up_s");
    check_bytes(dst.data(), pl.down, ly.down.data(), ly.down.size(), "down");
    check_bytes(dst.data(), pl.down_s, ly.down_s.data(), ly.down_s.size() * 4u,
                "down_s");
  }

  // 2. norms: fp16 vs. fp32 source within tolerance.
  check_norm_vec(dst.data(), g.woff.final_norm, synth.final_norm.data(),
                 cfg.hidden, "final_norm value");
  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    const HexWeightOffsets::PerLayer &pl = g.woff.layers[l];
    const SynthLayer &ly = synth.layers[l];
    check_norm_vec(dst.data(), pl.attn_norm, ly.attn_norm.data(), cfg.hidden,
                   "attn_norm value");
    check_norm_vec(dst.data(), pl.ffn_norm, ly.ffn_norm.data(), cfg.hidden,
                   "ffn_norm value");
    check_norm_vec(dst.data(), pl.q_norm, ly.q_norm.data(), cfg.head_dim,
                   "q_norm value");
    check_norm_vec(dst.data(), pl.k_norm, ly.k_norm.data(), cfg.head_dim,
                   "k_norm value");
  }

  // 3. RoPE table values.
  check_rope_table(dst.data(), g, cfg);

  // 4. Tied embed: HexWeightOffsets carries a single 'embed' field (no
  // separate lm_head offset), and weights_size must equal the sum of
  // every 128B-aligned tensor extent with embed counted exactly once.
  CHECK(g.weights_size == expected_weights_size(cfg),
        "tied embed: weights_size counts embed more than once");

  // 5. Full coverage: every tensor range must have been written (no
  // longer reads as the untouched 0xA5 prefill pattern). The synthetic
  // generators never emit an all-0xA5 constant range, so this is safe.
  std::vector<Extent> ext = tensor_extents(g, cfg);
  for (const Extent &e : ext)
    CHECK(!all_0xA5(dst.data() + e.first, e.second),
          "tensor range still reads as untouched 0xA5 prefill");
}

} // namespace

int main(void) {
  // --- Tiny config (deliberately hidden != n_heads*head_dim). ---
  HexModelConfig cfg{};
  cfg.n_layers = 2;
  cfg.hidden = 256;
  cfg.n_heads = 4;
  cfg.n_kv_heads = 2;
  cfg.head_dim = 128;
  cfg.ffn = 512;
  cfg.vocab = 512;
  cfg.max_seq = 64;
  cfg.max_chunk = 8;
  cfg.rms_eps = 1e-6f;
  cfg.rope_theta = 1e6f;

  HexLoweredGraph g = lower_qwen3(cfg);

  // 1. validate() must accept the produced op-list against the sizes
  // lower_qwen3 itself reports.
  uint32_t buf_size[NNTR_HTP_BUF_COUNT];
  buf_size[NNTR_HTP_BUF_WEIGHTS] = static_cast<uint32_t>(g.weights_size);
  buf_size[NNTR_HTP_BUF_KV] = static_cast<uint32_t>(g.kv_size);
  buf_size[NNTR_HTP_BUF_ACT] = static_cast<uint32_t>(g.act_size);
  buf_size[NNTR_HTP_BUF_TOKENS] = cfg.max_chunk * 4u;
  buf_size[NNTR_HTP_BUF_LOGITS] = cfg.vocab * 4u;
  CHECK(nntr_htp_oplist_validate(g.oplist.data(),
                                 static_cast<uint32_t>(g.oplist.size()),
                                 buf_size) == 0,
        "validate() failed on tiny config");

  // 2/3/5: op count/sequence, tied sharing, per-op params.
  check_sequence(g, cfg);

  // 4. header fields match cfg.
  nntr_htp_oplist_header h = read_header(g);
  CHECK(h.magic == NNTR_HTP_OPLIST_MAGIC, "header magic");
  CHECK(h.version == NNTR_HTP_ABI_VERSION, "header version");
  CHECK(h.n_layers == cfg.n_layers, "header n_layers");
  CHECK(h.n_heads == cfg.n_heads, "header n_heads");
  CHECK(h.n_kv_heads == cfg.n_kv_heads, "header n_kv_heads");
  CHECK(h.head_dim == cfg.head_dim, "header head_dim");
  CHECK(h.hidden == cfg.hidden, "header hidden");
  CHECK(h.ffn == cfg.ffn, "header ffn");
  CHECK(h.vocab == cfg.vocab, "header vocab");
  CHECK(h.max_seq == cfg.max_seq, "header max_seq");
  CHECK(h.max_chunk == cfg.max_chunk, "header max_chunk");

  // 6. ACT slots pairwise disjoint and within act_size.
  check_act_disjoint(g, cfg);

  CHECK(g.weights_size == expected_weights_size(cfg),
        "tiny weights_size mismatch vs independent layout recompute");

  // 8. pack_weights(): int8/scale byte-exact copy, norm/RoPE fp16
  // conversion accuracy, tied-embed accounting, full write coverage.
  check_pack_weights(g, cfg);

  // 9. .hexcfg round trip: every field survives text serialization.
  {
    std::string path = std::string(P_tmpdir) + "/hexcfg_roundtrip.hexcfg";
    write_hexcfg(path, cfg);
    HexModelConfig back = read_hexcfg(path);
    CHECK(back.n_layers == cfg.n_layers && back.n_heads == cfg.n_heads &&
            back.n_kv_heads == cfg.n_kv_heads &&
            back.head_dim == cfg.head_dim && back.hidden == cfg.hidden &&
            back.ffn == cfg.ffn && back.vocab == cfg.vocab &&
            back.max_seq == cfg.max_seq && back.max_chunk == cfg.max_chunk,
          "hexcfg integer field round trip");
    CHECK(back.rms_eps == cfg.rms_eps && back.rope_theta == cfg.rope_theta,
          "hexcfg float field round trip");
    std::remove(path.c_str());
  }

  // 7. Real-dims smoke: qwen3-0.6b.
  HexModelConfig real{};
  real.n_layers = 28;
  real.hidden = 1024;
  real.n_heads = 16;
  real.n_kv_heads = 8;
  real.head_dim = 128;
  real.ffn = 3072;
  real.vocab = 151936;
  real.max_seq = 2048;
  real.max_chunk = 128;
  real.rms_eps = 1e-6f;
  real.rope_theta = 1e6f;

  HexLoweredGraph rg = lower_qwen3(real);
  CHECK(read_header(rg).n_ops == 451u, "qwen3-0.6b n_ops != 451");
  uint64_t exp_kv = 2ull * 28u * 8u * real.max_seq * 128u * 2u;
  CHECK(rg.kv_size == exp_kv, "qwen3-0.6b kv_size mismatch");

  uint32_t rbuf[NNTR_HTP_BUF_COUNT];
  rbuf[NNTR_HTP_BUF_WEIGHTS] = static_cast<uint32_t>(rg.weights_size);
  rbuf[NNTR_HTP_BUF_KV] = static_cast<uint32_t>(rg.kv_size);
  rbuf[NNTR_HTP_BUF_ACT] = static_cast<uint32_t>(rg.act_size);
  rbuf[NNTR_HTP_BUF_TOKENS] = real.max_chunk * 4u;
  rbuf[NNTR_HTP_BUF_LOGITS] = real.vocab * 4u;
  CHECK(nntr_htp_oplist_validate(
          rg.oplist.data(), static_cast<uint32_t>(rg.oplist.size()), rbuf) == 0,
        "validate() failed on qwen3-0.6b dims");

  CHECK(rg.weights_size == expected_weights_size(real),
        "qwen3-0.6b weights_size mismatch vs independent recompute");
  // Sanity band around the actual qwen3-0.6b parameter count (embed
  // 155.6MB + non-embedding 0.44B int8 bytes + small scale/norm/rope
  // overhead): observed ~598.6MB. See task-7-report.md for why this
  // replaces the plan's rough 655-700MB estimate.
  CHECK(rg.weights_size >= 590000000ull && rg.weights_size <= 610000000ull,
        "qwen3-0.6b weights_size out of sane band");

  std::puts("LOWER_TEST PASS");
  return 0;
}

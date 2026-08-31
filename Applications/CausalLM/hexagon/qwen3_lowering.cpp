// SPDX-License-Identifier: Apache-2.0
/**
 * @file	qwen3_lowering.cpp
 * @date	19 August 2026
 * @brief	qwen3 -> v2 op-list lowering (WEIGHTS/ACT layout + op-list
 *		bytes). No weight data is read here; see pack_weights()
 *		(Task 8, nntrainer/tensor/hexagon/host/graph_lowering.h) for
 *		the actual byte packing.
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include "qwen3_lowering.h"

#include <cmath>
#include <cstring>

#include "nntr_htp_common.h"

namespace nntrainer::hexagon {

namespace {

/** @brief Bump allocator used for both the WEIGHTS and ACT layouts. */
class Cursor {
public:
  /** @brief Reserve bytes, 128B-aligning the start first. */
  uint32_t alloc(uint64_t bytes) {
    cur_ = align128(cur_);
    uint64_t start = cur_;
    cur_ += bytes;
    return static_cast<uint32_t>(start);
  }
  uint64_t size() const { return cur_; }

private:
  uint64_t cur_ = 0;
};

/** @brief Reinterpret an fp32 value as its raw bit pattern. */
uint32_t f32_bits(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

/** @brief Build one tensor_ref from a buffer id and byte offset. */
nntr_htp_tensor_ref ref(uint32_t buf, uint32_t offset) {
  nntr_htp_tensor_ref r;
  r.buf = buf;
  r.offset = offset;
  return r;
}

} // namespace

HexLoweredGraph lower_qwen3(const HexModelConfig &cfg) {
  HexLoweredGraph g{};
  g.woff.layers.resize(cfg.n_layers);

  const uint64_t n_q = static_cast<uint64_t>(cfg.n_heads) * cfg.head_dim;
  const uint64_t n_kv = static_cast<uint64_t>(cfg.n_kv_heads) * cfg.head_dim;

  // --- WEIGHTS layout: embed, embed_scale, rope_table, final_norm, then
  // per layer wq/wq_s/.../q_norm/k_norm, each tensor 128B aligned. ---
  Cursor wcur;
  g.woff.embed = wcur.alloc(static_cast<uint64_t>(cfg.vocab) * cfg.hidden);
  g.woff.embed_scale = wcur.alloc(static_cast<uint64_t>(cfg.vocab) * 4u);
  g.woff.rope_table =
    wcur.alloc(static_cast<uint64_t>(cfg.max_seq) * 128u * 2u);
  g.woff.final_norm = wcur.alloc(static_cast<uint64_t>(cfg.hidden) * 2u);

  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    HexWeightOffsets::PerLayer &pl = g.woff.layers[l];
    pl.wq = wcur.alloc(n_q * cfg.hidden);
    pl.wq_s = wcur.alloc(n_q * 4u);
    pl.wk = wcur.alloc(n_kv * cfg.hidden);
    pl.wk_s = wcur.alloc(n_kv * 4u);
    pl.wv = wcur.alloc(n_kv * cfg.hidden);
    pl.wv_s = wcur.alloc(n_kv * 4u);
    pl.wo = wcur.alloc(static_cast<uint64_t>(cfg.hidden) * n_q);
    pl.wo_s = wcur.alloc(static_cast<uint64_t>(cfg.hidden) * 4u);
    pl.gate = wcur.alloc(static_cast<uint64_t>(cfg.ffn) * cfg.hidden);
    pl.gate_s = wcur.alloc(static_cast<uint64_t>(cfg.ffn) * 4u);
    pl.up = wcur.alloc(static_cast<uint64_t>(cfg.ffn) * cfg.hidden);
    pl.up_s = wcur.alloc(static_cast<uint64_t>(cfg.ffn) * 4u);
    pl.down = wcur.alloc(static_cast<uint64_t>(cfg.hidden) * cfg.ffn);
    pl.down_s = wcur.alloc(static_cast<uint64_t>(cfg.hidden) * 4u);
    pl.attn_norm = wcur.alloc(static_cast<uint64_t>(cfg.hidden) * 2u);
    pl.ffn_norm = wcur.alloc(static_cast<uint64_t>(cfg.hidden) * 2u);
    pl.q_norm = wcur.alloc(static_cast<uint64_t>(cfg.head_dim) * 2u);
    pl.k_norm = wcur.alloc(static_cast<uint64_t>(cfg.head_dim) * 2u);
  }
  g.weights_size = wcur.size();

  // --- ACT layout: x, t, q, kb, vb, ao, h2, g, u, each 128B aligned. ---
  Cursor acur;
  const uint64_t mc = cfg.max_chunk;
  uint32_t act_x = acur.alloc(mc * cfg.hidden * 2u);
  uint32_t act_t = acur.alloc(mc * cfg.hidden * 2u);
  uint32_t act_q = acur.alloc(mc * n_q * 2u);
  uint32_t act_kb = acur.alloc(mc * n_kv * 2u);
  uint32_t act_vb = acur.alloc(mc * n_kv * 2u);
  uint32_t act_ao = acur.alloc(mc * n_q * 2u);
  uint32_t act_h2 = acur.alloc(mc * cfg.hidden * 2u);
  uint32_t act_g = acur.alloc(mc * cfg.ffn * 2u);
  uint32_t act_u = acur.alloc(mc * cfg.ffn * 2u);
  g.act_size = acur.size();

  g.kv_size =
    2ull * cfg.n_layers * cfg.n_kv_heads * cfg.max_seq * cfg.head_dim * 2ull;

  const uint32_t eps_bits = f32_bits(cfg.rms_eps);
  const uint32_t inv_sqrt_hd_bits =
    f32_bits(1.0f / std::sqrt(static_cast<float>(cfg.head_dim)));

  const uint32_t n_ops = 1u + 16u * cfg.n_layers + 2u;
  std::vector<nntr_htp_op_desc> ops;
  ops.reserve(n_ops);

  // op 0: EMBED, tokens[TOKENS@0] * embed/embed_scale -> x
  {
    nntr_htp_op_desc op{};
    op.kind = NNTR_HTP_OP_EMBED;
    op.k = cfg.hidden;
    op.in0 = ref(NNTR_HTP_BUF_TOKENS, 0u);
    op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, g.woff.embed);
    op.in2 = ref(NNTR_HTP_BUF_WEIGHTS, g.woff.embed_scale);
    op.out = ref(NNTR_HTP_BUF_ACT, act_x);
    ops.push_back(op);
  }

  for (uint32_t l = 0; l < cfg.n_layers; ++l) {
    const HexWeightOffsets::PerLayer &pl = g.woff.layers[l];
    const uint32_t n_q32 = static_cast<uint32_t>(n_q);
    const uint32_t n_kv32 = static_cast<uint32_t>(n_kv);

    // L.1: RMSNORM x*attn_norm -> t
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_RMSNORM;
      op.n = cfg.hidden;
      op.param0 = eps_bits;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_x);
      op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, pl.attn_norm);
      op.out = ref(NNTR_HTP_BUF_ACT, act_t);
      ops.push_back(op);
    }
    // L.2: MATMUL_W8A8 t*wq -> q
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_MATMUL_W8A8;
      op.k = cfg.hidden;
      op.n = n_q32;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_t);
      op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, pl.wq);
      op.in2 = ref(NNTR_HTP_BUF_WEIGHTS, pl.wq_s);
      op.out = ref(NNTR_HTP_BUF_ACT, act_q);
      ops.push_back(op);
    }
    // L.3: MATMUL_W8A8 t*wk -> kb
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_MATMUL_W8A8;
      op.k = cfg.hidden;
      op.n = n_kv32;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_t);
      op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, pl.wk);
      op.in2 = ref(NNTR_HTP_BUF_WEIGHTS, pl.wk_s);
      op.out = ref(NNTR_HTP_BUF_ACT, act_kb);
      ops.push_back(op);
    }
    // L.4: MATMUL_W8A8 t*wv -> vb
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_MATMUL_W8A8;
      op.k = cfg.hidden;
      op.n = n_kv32;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_t);
      op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, pl.wv);
      op.in2 = ref(NNTR_HTP_BUF_WEIGHTS, pl.wv_s);
      op.out = ref(NNTR_HTP_BUF_ACT, act_vb);
      ops.push_back(op);
    }
    // L.5: RMSNORM q*q_norm -> q (per-head QK-norm)
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_RMSNORM;
      op.flags = NNTR_HTP_FLAG_PER_HEAD;
      op.n = n_q32;
      op.param0 = eps_bits;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_q);
      op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, pl.q_norm);
      op.out = ref(NNTR_HTP_BUF_ACT, act_q);
      ops.push_back(op);
    }
    // L.6: RMSNORM kb*k_norm -> kb (per-head QK-norm)
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_RMSNORM;
      op.flags = NNTR_HTP_FLAG_PER_HEAD;
      op.n = n_kv32;
      op.param0 = eps_bits;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_kb);
      op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, pl.k_norm);
      op.out = ref(NNTR_HTP_BUF_ACT, act_kb);
      ops.push_back(op);
    }
    // L.7: ROPE q,kb using rope_table, in-place (out == in0)
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_ROPE;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_q);
      op.in1 = ref(NNTR_HTP_BUF_ACT, act_kb);
      op.in2 = ref(NNTR_HTP_BUF_WEIGHTS, g.woff.rope_table);
      op.out = ref(NNTR_HTP_BUF_ACT, act_q);
      ops.push_back(op);
    }
    // L.8: ATTN q,kb,vb -> ao (K/V history addressed via layer into KV)
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_ATTN;
      op.layer = l;
      op.param0 = inv_sqrt_hd_bits;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_q);
      op.in1 = ref(NNTR_HTP_BUF_ACT, act_kb);
      op.in2 = ref(NNTR_HTP_BUF_ACT, act_vb);
      op.out = ref(NNTR_HTP_BUF_ACT, act_ao);
      ops.push_back(op);
    }
    // L.9: MATMUL_W8A8 ao*wo -> h2
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_MATMUL_W8A8;
      op.k = n_q32;
      op.n = cfg.hidden;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_ao);
      op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, pl.wo);
      op.in2 = ref(NNTR_HTP_BUF_WEIGHTS, pl.wo_s);
      op.out = ref(NNTR_HTP_BUF_ACT, act_h2);
      ops.push_back(op);
    }
    // L.10: ADD x+h2 -> x (attn residual)
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_ADD;
      op.n = cfg.hidden;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_x);
      op.in1 = ref(NNTR_HTP_BUF_ACT, act_h2);
      op.out = ref(NNTR_HTP_BUF_ACT, act_x);
      ops.push_back(op);
    }
    // L.11: RMSNORM x*ffn_norm -> t
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_RMSNORM;
      op.n = cfg.hidden;
      op.param0 = eps_bits;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_x);
      op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, pl.ffn_norm);
      op.out = ref(NNTR_HTP_BUF_ACT, act_t);
      ops.push_back(op);
    }
    // L.12: MATMUL_W8A8 t*gate -> g
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_MATMUL_W8A8;
      op.k = cfg.hidden;
      op.n = cfg.ffn;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_t);
      op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, pl.gate);
      op.in2 = ref(NNTR_HTP_BUF_WEIGHTS, pl.gate_s);
      op.out = ref(NNTR_HTP_BUF_ACT, act_g);
      ops.push_back(op);
    }
    // L.13: MATMUL_W8A8 t*up -> u
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_MATMUL_W8A8;
      op.k = cfg.hidden;
      op.n = cfg.ffn;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_t);
      op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, pl.up);
      op.in2 = ref(NNTR_HTP_BUF_WEIGHTS, pl.up_s);
      op.out = ref(NNTR_HTP_BUF_ACT, act_u);
      ops.push_back(op);
    }
    // L.14: SILU_MUL g,u -> g (in-place)
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_SILU_MUL;
      op.n = cfg.ffn;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_g);
      op.in1 = ref(NNTR_HTP_BUF_ACT, act_u);
      op.out = ref(NNTR_HTP_BUF_ACT, act_g);
      ops.push_back(op);
    }
    // L.15: MATMUL_W8A16 g*down -> h2. The SwiGLU output is outlier-heavy;
    // per-token int8 there alone costs ~6% PPL on qwen3-0.6b (M4 Task 5),
    // so this one matmul keeps the fp16 activation.
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_MATMUL_W8A16;
      op.k = cfg.ffn;
      op.n = cfg.hidden;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_g);
      op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, pl.down);
      op.in2 = ref(NNTR_HTP_BUF_WEIGHTS, pl.down_s);
      op.out = ref(NNTR_HTP_BUF_ACT, act_h2);
      ops.push_back(op);
    }
    // L.16: ADD x+h2 -> x (ffn residual)
    {
      nntr_htp_op_desc op{};
      op.kind = NNTR_HTP_OP_ADD;
      op.n = cfg.hidden;
      op.in0 = ref(NNTR_HTP_BUF_ACT, act_x);
      op.in1 = ref(NNTR_HTP_BUF_ACT, act_h2);
      op.out = ref(NNTR_HTP_BUF_ACT, act_x);
      ops.push_back(op);
    }
  }

  // final-2: RMSNORM x*final_norm -> t
  {
    nntr_htp_op_desc op{};
    op.kind = NNTR_HTP_OP_RMSNORM;
    op.n = cfg.hidden;
    op.param0 = eps_bits;
    op.in0 = ref(NNTR_HTP_BUF_ACT, act_x);
    op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, g.woff.final_norm);
    op.out = ref(NNTR_HTP_BUF_ACT, act_t);
    ops.push_back(op);
  }
  // final-1: MATMUL_LOGITS t*embed(tied)/embed_scale -> LOGITS
  {
    nntr_htp_op_desc op{};
    op.kind = NNTR_HTP_OP_MATMUL_LOGITS;
    op.m = 1u;
    op.k = cfg.hidden;
    op.n = cfg.vocab;
    op.in0 = ref(NNTR_HTP_BUF_ACT, act_t);
    op.in1 = ref(NNTR_HTP_BUF_WEIGHTS, g.woff.embed);
    op.in2 = ref(NNTR_HTP_BUF_WEIGHTS, g.woff.embed_scale);
    op.out = ref(NNTR_HTP_BUF_LOGITS, 0u);
    ops.push_back(op);
  }

  // --- Serialize header + ops into the wire-format byte buffer. ---
  nntr_htp_oplist_header header{};
  header.magic = NNTR_HTP_OPLIST_MAGIC;
  header.version = NNTR_HTP_ABI_VERSION;
  header.n_ops = static_cast<uint32_t>(ops.size());
  header.n_layers = cfg.n_layers;
  header.n_heads = cfg.n_heads;
  header.n_kv_heads = cfg.n_kv_heads;
  header.head_dim = cfg.head_dim;
  header.hidden = cfg.hidden;
  header.ffn = cfg.ffn;
  header.vocab = cfg.vocab;
  header.max_seq = cfg.max_seq;
  header.max_chunk = cfg.max_chunk;

  g.oplist.resize(sizeof(header) + ops.size() * sizeof(nntr_htp_op_desc));
  std::memcpy(g.oplist.data(), &header, sizeof(header));
  if (!ops.empty())
    std::memcpy(g.oplist.data() + sizeof(header), ops.data(),
                ops.size() * sizeof(nntr_htp_op_desc));

  return g;
}

} // namespace nntrainer::hexagon

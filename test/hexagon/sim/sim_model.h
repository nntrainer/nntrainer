// SPDX-License-Identifier: Apache-2.0
/**
 * @file	sim_model.h
 * @date	15 September 2026
 * @brief	Parameterized hand lowering of the qwen3 op sequence for
 *		hexagon-sim tests: WEIGHTS/ACT offset plan, deterministic
 *		weight fill and op-list build (the same 1 + 16 * L + 2 ops as
 *		lower_qwen3(), HEXAGON.md section 2.3)
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#ifndef SIM_MODEL_H
#define SIM_MODEL_H

#include <stdint.h>

struct sim_model_cfg {
  uint32_t n_layers, n_heads, n_kv_heads, head_dim;
  uint32_t hidden, ffn, vocab, max_seq, max_chunk;
  float eps, rope_theta;
};

struct sim_layer_off {
  uint32_t wq, sq, wk, sk, wv, sv, wo, so, wg, sg, wu, su, wd, sd;
  uint32_t attn_g, ffn_g, q_g, k_g;
};

/** Every offset is 128B aligned (bump allocator). WEIGHTS: embed, embed
 * scales, rope table, final gamma, then per layer the seven int8 matrices
 * with fp32 scales and four fp16 gammas. ACT: ten slots sized for
 * max_chunk rows, reused by every layer. */
struct sim_model_plan {
  struct sim_model_cfg cfg;
  uint32_t embed_w, embed_s, rope, final_g;
  struct sim_layer_off *l; /* [n_layers] */
  uint32_t wtotal;
  uint32_t resid, xn, q, kbuf, vbuf, attn, oout, gate, up, fout;
  uint32_t atotal;
  uint32_t kv_bytes, n_ops, oplist_len;
};

/* 0 ok, 1 allocation failure. */
int sim_model_plan_init(struct sim_model_plan *p,
                        const struct sim_model_cfg *cfg);
void sim_model_plan_free(struct sim_model_plan *p);

/** Deterministic pseudo-random int8 weights, magnitude-preserving fp32
 * scales, gammas near 1, and the real rope table (uses frand()). */
void sim_model_fill_weights(const struct sim_model_plan *p, uint8_t *w);

/** Writes header + p->n_ops descriptors into buf (>= p->oplist_len B) and
 * returns nntr_htp_oplist_validate() of the result against the plan's own
 * WEIGHTS/KV/ACT/TOKENS/LOGITS sizes: 0 ok, else the validator rc. */
int sim_model_build_oplist(const struct sim_model_plan *p, uint8_t *buf);

#endif /* SIM_MODEL_H */

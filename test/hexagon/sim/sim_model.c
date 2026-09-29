// SPDX-License-Identifier: Apache-2.0
/**
 * @file	sim_model.c
 * @date	15 September 2026
 * @brief	Parameterized hand lowering for hexagon-sim tests (see sim_model.h)
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nntr_htp_common.h"
#include "ref_ops.h"
#include "sim_model.h"
#include "sim_test_util.h"

static uint32_t bump(uint32_t *top, uint32_t bytes) {
  uint32_t o = *top;
  *top = (o + bytes + 127u) & ~127u;
  return o;
}

int sim_model_plan_init(struct sim_model_plan *p,
                        const struct sim_model_cfg *cfg) {
  const uint32_t qdim = cfg->n_heads * cfg->head_dim;
  const uint32_t kvdim = cfg->n_kv_heads * cfg->head_dim;
  uint32_t t = 0, l;

  memset(p, 0, sizeof(*p));
  p->cfg = *cfg;
  p->l = calloc(cfg->n_layers, sizeof(*p->l));
  if (!p->l)
    return 1;

  p->embed_w = bump(&t, cfg->vocab * cfg->hidden);
  p->embed_s = bump(&t, cfg->vocab * 4u);
  p->rope = bump(&t, cfg->max_seq * 128u * 2u);
  p->final_g = bump(&t, cfg->hidden * 2u);
  for (l = 0; l < cfg->n_layers; ++l) {
    struct sim_layer_off *w = &p->l[l];
    w->wq = bump(&t, qdim * cfg->hidden);
    w->sq = bump(&t, qdim * 4u);
    w->wk = bump(&t, kvdim * cfg->hidden);
    w->sk = bump(&t, kvdim * 4u);
    w->wv = bump(&t, kvdim * cfg->hidden);
    w->sv = bump(&t, kvdim * 4u);
    w->wo = bump(&t, cfg->hidden * qdim);
    w->so = bump(&t, cfg->hidden * 4u);
    w->wg = bump(&t, cfg->ffn * cfg->hidden);
    w->sg = bump(&t, cfg->ffn * 4u);
    w->wu = bump(&t, cfg->ffn * cfg->hidden);
    w->su = bump(&t, cfg->ffn * 4u);
    w->wd = bump(&t, cfg->hidden * cfg->ffn);
    w->sd = bump(&t, cfg->hidden * 4u);
    w->attn_g = bump(&t, cfg->hidden * 2u);
    w->ffn_g = bump(&t, cfg->hidden * 2u);
    w->q_g = bump(&t, cfg->head_dim * 2u);
    w->k_g = bump(&t, cfg->head_dim * 2u);
  }
  p->wtotal = t;

  t = 0;
  p->resid = bump(&t, cfg->max_chunk * cfg->hidden * 2u);
  p->xn = bump(&t, cfg->max_chunk * cfg->hidden * 2u);
  p->q = bump(&t, cfg->max_chunk * qdim * 2u);
  p->kbuf = bump(&t, cfg->max_chunk * kvdim * 2u);
  p->vbuf = bump(&t, cfg->max_chunk * kvdim * 2u);
  p->attn = bump(&t, cfg->max_chunk * qdim * 2u);
  p->oout = bump(&t, cfg->max_chunk * cfg->hidden * 2u);
  p->gate = bump(&t, cfg->max_chunk * cfg->ffn * 2u);
  p->up = bump(&t, cfg->max_chunk * cfg->ffn * 2u);
  p->fout = bump(&t, cfg->max_chunk * cfg->hidden * 2u);
  p->atotal = t;

  p->kv_bytes = (uint32_t)nntr_htp_kv_bytes(cfg->n_layers, cfg->n_kv_heads,
                                            cfg->max_seq, cfg->head_dim);
  p->n_ops = 1u + 16u * cfg->n_layers + 2u;
  p->oplist_len = (uint32_t)nntr_htp_oplist_bytes(p->n_ops);
  return 0;
}

void sim_model_plan_free(struct sim_model_plan *p) {
  free(p->l);
  p->l = 0;
}

static void fill_i8(uint8_t *w, uint32_t off, uint32_t count) {
  int8_t *p = (int8_t *)(w + off);
  for (uint32_t i = 0; i < count; ++i)
    p[i] = (int8_t)(frand() * 127.f);
}

/** Scale range keeps every projection roughly magnitude-preserving so the
 * residual stream stays O(1) like a trained model (larger scales blow the
 * activations up per layer and the coarse fp16 ulps then amplify benign
 * HVX-vs-scalar rounding past the golden tolerance). */
static void fill_scale(uint8_t *w, uint32_t off, uint32_t count) {
  float *p = (float *)(void *)(w + off);
  for (uint32_t i = 0; i < count; ++i)
    p[i] = 0.0001f + 0.0004f * (frand() * 0.5f + 0.5f);
}

static void fill_gamma(uint8_t *w, uint32_t off, uint32_t count) {
  __fp16 *p = (__fp16 *)(void *)(w + off);
  for (uint32_t i = 0; i < count; ++i)
    p[i] = (__fp16)(1.0f + 0.25f * frand());
}

/** The int8 areas are uniform random bytes, so the same fill is a valid
 * tiled32 image: kernel and reference both index it with
 * nntr_htp_tile_off, no repack needed here. */
void sim_model_fill_weights(const struct sim_model_plan *p, uint8_t *w) {
  const struct sim_model_cfg *c = &p->cfg;
  const uint32_t qdim = c->n_heads * c->head_dim;
  const uint32_t kvdim = c->n_kv_heads * c->head_dim;

  fill_i8(w, p->embed_w, c->vocab * c->hidden);
  fill_scale(w, p->embed_s, c->vocab);
  ref_rope_table_fill((__fp16 *)(void *)(w + p->rope), c->max_seq,
                      c->rope_theta);
  fill_gamma(w, p->final_g, c->hidden);
  for (uint32_t l = 0; l < c->n_layers; ++l) {
    const struct sim_layer_off *lw = &p->l[l];
    fill_i8(w, lw->wq, qdim * c->hidden);
    fill_scale(w, lw->sq, qdim);
    fill_i8(w, lw->wk, kvdim * c->hidden);
    fill_scale(w, lw->sk, kvdim);
    fill_i8(w, lw->wv, kvdim * c->hidden);
    fill_scale(w, lw->sv, kvdim);
    fill_i8(w, lw->wo, c->hidden * qdim);
    fill_scale(w, lw->so, c->hidden);
    fill_i8(w, lw->wg, c->ffn * c->hidden);
    fill_scale(w, lw->sg, c->ffn);
    fill_i8(w, lw->wu, c->ffn * c->hidden);
    fill_scale(w, lw->su, c->ffn);
    fill_i8(w, lw->wd, c->hidden * c->ffn);
    fill_scale(w, lw->sd, c->hidden);
    fill_gamma(w, lw->attn_g, c->hidden);
    fill_gamma(w, lw->ffn_g, c->hidden);
    fill_gamma(w, lw->q_g, c->head_dim);
    fill_gamma(w, lw->k_g, c->head_dim);
  }
}

static uint32_t f_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  return u;
}

static struct nntr_htp_tensor_ref R(uint32_t buf, uint32_t off) {
  struct nntr_htp_tensor_ref r = {buf, off};
  return r;
} static struct nntr_htp_tensor_ref W(uint32_t off) {
  return R(NNTR_HTP_BUF_WEIGHTS, off);
} static struct nntr_htp_tensor_ref A(uint32_t off) {
  return R(NNTR_HTP_BUF_ACT, off);
}

static void
emit(struct nntr_htp_op_desc **p, uint32_t kind, uint32_t flags, uint32_t layer,
     uint32_t m, uint32_t k, uint32_t n, struct nntr_htp_tensor_ref in0,
     struct nntr_htp_tensor_ref in1, struct nntr_htp_tensor_ref in2,
     struct nntr_htp_tensor_ref out, uint32_t param0) {
  struct nntr_htp_op_desc *d = (*p)++;
  memset(d, 0, sizeof(*d));
  d->kind = kind;
  d->flags = flags;
  d->layer = layer;
  d->m = m;
  d->k = k;
  d->n = n;
  d->in0 = in0;
  d->in1 = in1;
  d->in2 = in2;
  d->out = out;
  d->param0 = param0;
}

/** EMBED + 16 ops per layer + final RMSNORM + MATMUL_LOGITS: the op
 * sequence of HEXAGON.md section 2.3. The result is checked by the shared
 * validator against the plan's own buffer sizes, so this hand lowering is
 * bound to the same extent table (nntr_htp_op_extent) as lower_qwen3(). */
int sim_model_build_oplist(const struct sim_model_plan *p, uint8_t *buf) {
  const struct sim_model_cfg *c = &p->cfg;
  uint32_t sizes[NNTR_HTP_BUF_COUNT];
  const uint32_t qdim = c->n_heads * c->head_dim;
  const uint32_t kvdim = c->n_kv_heads * c->head_dim;
  struct nntr_htp_oplist_header *h = (struct nntr_htp_oplist_header *)buf;
  struct nntr_htp_op_desc *d =
    (struct nntr_htp_op_desc *)(void *)(buf + sizeof(*h));
  const uint32_t eps = f_bits(c->eps);
  const uint32_t attn_scale = f_bits(1.0f / sqrtf((float)c->head_dim));

  memset(h, 0, sizeof(*h));
  h->magic = NNTR_HTP_OPLIST_MAGIC;
  h->version = NNTR_HTP_ABI_VERSION;
  h->n_ops = p->n_ops;
  h->n_layers = c->n_layers;
  h->n_heads = c->n_heads;
  h->n_kv_heads = c->n_kv_heads;
  h->head_dim = c->head_dim;
  h->hidden = c->hidden;
  h->ffn = c->ffn;
  h->vocab = c->vocab;
  h->max_seq = c->max_seq;
  h->max_chunk = c->max_chunk;
  h->weight_layout = NNTR_HTP_WEIGHT_LAYOUT_TILED32;

  emit(&d, NNTR_HTP_OP_EMBED, 0, 0, 0, c->hidden, 0, R(NNTR_HTP_BUF_TOKENS, 0),
       W(p->embed_w), W(p->embed_s), A(p->resid), 0);

  for (uint32_t l = 0; l < c->n_layers; ++l) {
    const struct sim_layer_off *w = &p->l[l];
    emit(&d, NNTR_HTP_OP_RMSNORM, 0, l, 0, 0, c->hidden, A(p->resid),
         W(w->attn_g), W(0), A(p->xn), eps);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A8, 0, l, 0, c->hidden, qdim, A(p->xn),
         W(w->wq), W(w->sq), A(p->q), 0);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A8, 0, l, 0, c->hidden, kvdim, A(p->xn),
         W(w->wk), W(w->sk), A(p->kbuf), 0);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A8, 0, l, 0, c->hidden, kvdim, A(p->xn),
         W(w->wv), W(w->sv), A(p->vbuf), 0);
    emit(&d, NNTR_HTP_OP_RMSNORM, NNTR_HTP_FLAG_PER_HEAD, l, 0, 0, qdim,
         A(p->q), W(w->q_g), W(0), A(p->q), eps);
    emit(&d, NNTR_HTP_OP_RMSNORM, NNTR_HTP_FLAG_PER_HEAD, l, 0, 0, kvdim,
         A(p->kbuf), W(w->k_g), W(0), A(p->kbuf), eps);
    emit(&d, NNTR_HTP_OP_ROPE, 0, l, 0, 0, 0, A(p->q), A(p->kbuf), W(p->rope),
         A(p->q), 0);
    emit(&d, NNTR_HTP_OP_ATTN, 0, l, 0, 0, 0, A(p->q), A(p->kbuf), A(p->vbuf),
         A(p->attn), attn_scale);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A8, 0, l, 0, qdim, c->hidden, A(p->attn),
         W(w->wo), W(w->so), A(p->oout), 0);
    emit(&d, NNTR_HTP_OP_ADD, 0, l, 0, 0, c->hidden, A(p->resid), A(p->oout),
         W(0), A(p->resid), 0);
    emit(&d, NNTR_HTP_OP_RMSNORM, 0, l, 0, 0, c->hidden, A(p->resid),
         W(w->ffn_g), W(0), A(p->xn), eps);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A8, 0, l, 0, c->hidden, c->ffn, A(p->xn),
         W(w->wg), W(w->sg), A(p->gate), 0);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A8, 0, l, 0, c->hidden, c->ffn, A(p->xn),
         W(w->wu), W(w->su), A(p->up), 0);
    emit(&d, NNTR_HTP_OP_SILU_MUL, 0, l, 0, 0, c->ffn, A(p->gate), A(p->up),
         W(0), A(p->gate), 0);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A16, 0, l, 0, c->ffn, c->hidden, A(p->gate),
         W(w->wd), W(w->sd), A(p->fout), 0);
    emit(&d, NNTR_HTP_OP_ADD, 0, l, 0, 0, c->hidden, A(p->resid), A(p->fout),
         W(0), A(p->resid), 0);
  }

  emit(&d, NNTR_HTP_OP_RMSNORM, 0, 0, 0, 0, c->hidden, A(p->resid),
       W(p->final_g), W(0), A(p->xn), eps);
  /* Tied lm_head: reuse the embedding table and scales. */
  emit(&d, NNTR_HTP_OP_MATMUL_LOGITS, 0, 0, 1, c->hidden, c->vocab, A(p->xn),
       W(p->embed_w), W(p->embed_s), R(NNTR_HTP_BUF_LOGITS, 0), 0);

  sizes[NNTR_HTP_BUF_WEIGHTS] = p->wtotal;
  sizes[NNTR_HTP_BUF_KV] = p->kv_bytes;
  sizes[NNTR_HTP_BUF_ACT] = p->atotal;
  sizes[NNTR_HTP_BUF_TOKENS] = c->max_chunk * 4u;
  sizes[NNTR_HTP_BUF_LOGITS] = c->vocab * 4u;
  return nntr_htp_oplist_validate(buf, p->oplist_len, sizes);
}

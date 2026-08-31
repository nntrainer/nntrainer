// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_graph.c
 * @date	19 August 2026
 * @brief	Hexagon-sim golden test for the op-list graph executor: a
 *		synthetic 2-layer model (35 ops), prefill then decode compared
 *		against the scalar reference executor, plus negative cases
 *		for init validation and forward runtime-argument checks
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "htp_graph.h"
#include "ref_ops.h"
#include "sim_test_util.h"

#define N_LAYERS 2u
#define HIDDEN 256u
#define N_HEADS 4u
#define N_KV_HEADS 2u
#define HEAD_DIM 128u
#define FFN 512u
#define VOCAB 512u
#define MAX_SEQ 64u
#define MAX_CHUNK 8u
/* Note n_heads*head_dim = 512 != hidden = 256: Q/K/V project hidden ->
 * heads*head_dim and O projects back. */
#define QDIM (N_HEADS * HEAD_DIM)     /* 512 */
#define KVDIM (N_KV_HEADS * HEAD_DIM) /* 256 */
#define N_OPS 35u
#define OPLIST_LEN (64u + N_OPS * 64u)
#define KV_BYTES (2u * N_LAYERS * N_KV_HEADS * MAX_SEQ * HEAD_DIM * 2u)
#define EPS 1e-5f

/* Bump allocator: every returned offset and the next top stay 128B
 * aligned (all region sizes here are already multiples of 128B). */
static uint32_t bump(uint32_t *top, uint32_t bytes) {
  uint32_t o = *top;
  *top = (o + bytes + 127u) & ~127u;
  return o;
}

/*
 * WEIGHTS offset plan (bump order, all 128B aligned):
 *   embed_w int8[VOCAB][HIDDEN]  embed_s fp32[VOCAB]   (tied lm_head)
 *   rope    fp16[MAX_SEQ][128]   final_g fp16[HIDDEN]
 *   per layer l = 0..1:
 *     wq int8[QDIM][HIDDEN]   sq fp32[QDIM]
 *     wk int8[KVDIM][HIDDEN]  sk fp32[KVDIM]
 *     wv int8[KVDIM][HIDDEN]  sv fp32[KVDIM]
 *     wo int8[HIDDEN][QDIM]   so fp32[HIDDEN]
 *     wg int8[FFN][HIDDEN]    sg fp32[FFN]
 *     wu int8[FFN][HIDDEN]    su fp32[FFN]
 *     wd int8[HIDDEN][FFN]    sd fp32[HIDDEN]
 *     attn_g fp16[HIDDEN]  ffn_g fp16[HIDDEN]
 *     q_g fp16[HEAD_DIM]   k_g fp16[HEAD_DIM]
 *
 * ACT offset plan (rows sized for MAX_CHUNK tokens, reused per layer):
 *   resid fp16[8][HIDDEN]  - residual stream (EMBED out, ADD in/out)
 *   xn    fp16[8][HIDDEN]  - RMSNORM output feeding the matmuls
 *   q     fp16[8][QDIM]    kbuf fp16[8][KVDIM]  vbuf fp16[8][KVDIM]
 *   attn  fp16[8][QDIM]    oout fp16[8][HIDDEN]
 *   gate  fp16[8][FFN]     up   fp16[8][FFN]    fout fp16[8][HIDDEN]
 */
struct layer_w {
  uint32_t wq, sq, wk, sk, wv, sv, wo, so, wg, sg, wu, su, wd, sd;
  uint32_t attn_g, ffn_g, q_g, k_g;
};
static struct {
  uint32_t embed_w, embed_s, rope, final_g;
  struct layer_w l[N_LAYERS];
  uint32_t wtotal;
  uint32_t resid, xn, q, kbuf, vbuf, attn, oout, gate, up, fout;
  uint32_t atotal;
} P;

static void plan_offsets(void) {
  uint32_t t = 0;
  P.embed_w = bump(&t, VOCAB * HIDDEN);
  P.embed_s = bump(&t, VOCAB * 4u);
  P.rope = bump(&t, MAX_SEQ * 128u * 2u);
  P.final_g = bump(&t, HIDDEN * 2u);
  for (uint32_t l = 0; l < N_LAYERS; ++l) {
    struct layer_w *w = &P.l[l];
    w->wq = bump(&t, QDIM * HIDDEN);
    w->sq = bump(&t, QDIM * 4u);
    w->wk = bump(&t, KVDIM * HIDDEN);
    w->sk = bump(&t, KVDIM * 4u);
    w->wv = bump(&t, KVDIM * HIDDEN);
    w->sv = bump(&t, KVDIM * 4u);
    w->wo = bump(&t, HIDDEN * QDIM);
    w->so = bump(&t, HIDDEN * 4u);
    w->wg = bump(&t, FFN * HIDDEN);
    w->sg = bump(&t, FFN * 4u);
    w->wu = bump(&t, FFN * HIDDEN);
    w->su = bump(&t, FFN * 4u);
    w->wd = bump(&t, HIDDEN * FFN);
    w->sd = bump(&t, HIDDEN * 4u);
    w->attn_g = bump(&t, HIDDEN * 2u);
    w->ffn_g = bump(&t, HIDDEN * 2u);
    w->q_g = bump(&t, HEAD_DIM * 2u);
    w->k_g = bump(&t, HEAD_DIM * 2u);
  }
  P.wtotal = t;

  t = 0;
  P.resid = bump(&t, MAX_CHUNK * HIDDEN * 2u);
  P.xn = bump(&t, MAX_CHUNK * HIDDEN * 2u);
  P.q = bump(&t, MAX_CHUNK * QDIM * 2u);
  P.kbuf = bump(&t, MAX_CHUNK * KVDIM * 2u);
  P.vbuf = bump(&t, MAX_CHUNK * KVDIM * 2u);
  P.attn = bump(&t, MAX_CHUNK * QDIM * 2u);
  P.oout = bump(&t, MAX_CHUNK * HIDDEN * 2u);
  P.gate = bump(&t, MAX_CHUNK * FFN * 2u);
  P.up = bump(&t, MAX_CHUNK * FFN * 2u);
  P.fout = bump(&t, MAX_CHUNK * HIDDEN * 2u);
  P.atotal = t;
}

static void fill_i8(uint8_t *w, uint32_t off, uint32_t count) {
  int8_t *p = (int8_t *)(w + off);
  for (uint32_t i = 0; i < count; ++i)
    p[i] = (int8_t)(frand() * 127.f);
}

static void fill_scale(uint8_t *w, uint32_t off, uint32_t count) {
  /* Scale range keeps every projection roughly magnitude-preserving so the
   * residual stream stays O(1) like a trained model. Larger scales blow the
   * activations up ~17x per layer, and the resulting coarse fp16 ulps feed
   * the per-token quantizers, amplifying benign HVX-vs-scalar rounding
   * differences past the golden tolerance. */
  float *p = (float *)(void *)(w + off);
  for (uint32_t i = 0; i < count; ++i)
    p[i] = 0.0001f + 0.0004f * (frand() * 0.5f + 0.5f);
}

static void fill_gamma(uint8_t *w, uint32_t off, uint32_t count) {
  __fp16 *p = (__fp16 *)(void *)(w + off);
  for (uint32_t i = 0; i < count; ++i)
    p[i] = (__fp16)(1.0f + 0.25f * frand());
}

static void fill_weights(uint8_t *w) {
  fill_i8(w, P.embed_w, VOCAB * HIDDEN);
  fill_scale(w, P.embed_s, VOCAB);
  ref_rope_table_fill((__fp16 *)(void *)(w + P.rope), MAX_SEQ, 1e6f);
  fill_gamma(w, P.final_g, HIDDEN);
  for (uint32_t l = 0; l < N_LAYERS; ++l) {
    const struct layer_w *lw = &P.l[l];
    fill_i8(w, lw->wq, QDIM * HIDDEN);
    fill_scale(w, lw->sq, QDIM);
    fill_i8(w, lw->wk, KVDIM * HIDDEN);
    fill_scale(w, lw->sk, KVDIM);
    fill_i8(w, lw->wv, KVDIM * HIDDEN);
    fill_scale(w, lw->sv, KVDIM);
    fill_i8(w, lw->wo, HIDDEN * QDIM);
    fill_scale(w, lw->so, HIDDEN);
    fill_i8(w, lw->wg, FFN * HIDDEN);
    fill_scale(w, lw->sg, FFN);
    fill_i8(w, lw->wu, FFN * HIDDEN);
    fill_scale(w, lw->su, FFN);
    fill_i8(w, lw->wd, HIDDEN * FFN);
    fill_scale(w, lw->sd, HIDDEN);
    fill_gamma(w, lw->attn_g, HIDDEN);
    fill_gamma(w, lw->ffn_g, HIDDEN);
    fill_gamma(w, lw->q_g, HEAD_DIM);
    fill_gamma(w, lw->k_g, HEAD_DIM);
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
}

static struct nntr_htp_tensor_ref
W(uint32_t off) {
  return R(NNTR_HTP_BUF_WEIGHTS, off);
}

static struct nntr_htp_tensor_ref
A(uint32_t off) {
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

/* Hand lowering of the tiny 2-layer model: EMBED + 2 x 16 layer ops +
 * final RMSNORM + MATMUL_LOGITS = 35 ops. */
static void build_tiny_oplist(uint8_t *buf) {
  struct nntr_htp_oplist_header *h = (struct nntr_htp_oplist_header *)buf;
  struct nntr_htp_op_desc *d =
    (struct nntr_htp_op_desc *)(void *)(buf + sizeof(*h));
  const uint32_t eps = f_bits(EPS);
  const uint32_t attn_scale = f_bits(1.0f / sqrtf((float)HEAD_DIM));

  memset(h, 0, sizeof(*h));
  h->magic = NNTR_HTP_OPLIST_MAGIC;
  h->version = NNTR_HTP_ABI_VERSION;
  h->n_ops = N_OPS;
  h->n_layers = N_LAYERS;
  h->n_heads = N_HEADS;
  h->n_kv_heads = N_KV_HEADS;
  h->head_dim = HEAD_DIM;
  h->hidden = HIDDEN;
  h->ffn = FFN;
  h->vocab = VOCAB;
  h->max_seq = MAX_SEQ;
  h->max_chunk = MAX_CHUNK;

  emit(&d, NNTR_HTP_OP_EMBED, 0, 0, 0, HIDDEN, 0, R(NNTR_HTP_BUF_TOKENS, 0),
       W(P.embed_w), W(P.embed_s), A(P.resid), 0);

  for (uint32_t l = 0; l < N_LAYERS; ++l) {
    const struct layer_w *w = &P.l[l];
    emit(&d, NNTR_HTP_OP_RMSNORM, 0, l, 0, 0, HIDDEN, A(P.resid), W(w->attn_g),
         W(0), A(P.xn), eps);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A8, 0, l, 0, HIDDEN, QDIM, A(P.xn), W(w->wq),
         W(w->sq), A(P.q), 0);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A8, 0, l, 0, HIDDEN, KVDIM, A(P.xn), W(w->wk),
         W(w->sk), A(P.kbuf), 0);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A8, 0, l, 0, HIDDEN, KVDIM, A(P.xn), W(w->wv),
         W(w->sv), A(P.vbuf), 0);
    emit(&d, NNTR_HTP_OP_RMSNORM, NNTR_HTP_FLAG_PER_HEAD, l, 0, 0, QDIM, A(P.q),
         W(w->q_g), W(0), A(P.q), eps);
    emit(&d, NNTR_HTP_OP_RMSNORM, NNTR_HTP_FLAG_PER_HEAD, l, 0, 0, KVDIM,
         A(P.kbuf), W(w->k_g), W(0), A(P.kbuf), eps);
    emit(&d, NNTR_HTP_OP_ROPE, 0, l, 0, 0, 0, A(P.q), A(P.kbuf), W(P.rope),
         A(P.q), 0);
    emit(&d, NNTR_HTP_OP_ATTN, 0, l, 0, 0, 0, A(P.q), A(P.kbuf), A(P.vbuf),
         A(P.attn), attn_scale);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A8, 0, l, 0, QDIM, HIDDEN, A(P.attn),
         W(w->wo), W(w->so), A(P.oout), 0);
    emit(&d, NNTR_HTP_OP_ADD, 0, l, 0, 0, HIDDEN, A(P.resid), A(P.oout), W(0),
         A(P.resid), 0);
    emit(&d, NNTR_HTP_OP_RMSNORM, 0, l, 0, 0, HIDDEN, A(P.resid), W(w->ffn_g),
         W(0), A(P.xn), eps);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A8, 0, l, 0, HIDDEN, FFN, A(P.xn), W(w->wg),
         W(w->sg), A(P.gate), 0);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A8, 0, l, 0, HIDDEN, FFN, A(P.xn), W(w->wu),
         W(w->su), A(P.up), 0);
    emit(&d, NNTR_HTP_OP_SILU_MUL, 0, l, 0, 0, FFN, A(P.gate), A(P.up), W(0),
         A(P.gate), 0);
    emit(&d, NNTR_HTP_OP_MATMUL_W8A16, 0, l, 0, FFN, HIDDEN, A(P.gate),
         W(w->wd), W(w->sd), A(P.fout), 0);
    emit(&d, NNTR_HTP_OP_ADD, 0, l, 0, 0, HIDDEN, A(P.resid), A(P.fout), W(0),
         A(P.resid), 0);
  }

  emit(&d, NNTR_HTP_OP_RMSNORM, 0, 0, 0, 0, HIDDEN, A(P.resid), W(P.final_g),
       W(0), A(P.xn), eps);
  /* Tied lm_head: reuse the embedding table and scales. */
  emit(&d, NNTR_HTP_OP_MATMUL_LOGITS, 0, 0, 1, HIDDEN, VOCAB, A(P.xn),
       W(P.embed_w), W(P.embed_s), R(NNTR_HTP_BUF_LOGITS, 0), 0);
}

int test_graph(void) {
  static const int32_t prefill_tok[MAX_CHUNK] = {1,  17, 300, 511,
                                                 42, 7,  129, 256};
  static const int32_t decode_tok = 99;
  float logits[VOCAB], rlogits[VOCAB];
  struct htp_graph g;
  int rc = 0, inited = 0;

  plan_offsets();
  uint8_t *w = memalign(128, P.wtotal);
  uint8_t *act = memalign(128, P.atotal);
  uint8_t *kv = memalign(128, KV_BYTES);
  uint8_t *ol = memalign(128, OPLIST_LEN);
  /* Reference executor runs on its own copies of every buffer. */
  uint8_t *rw = memalign(128, P.wtotal);
  uint8_t *ract = memalign(128, P.atotal);
  uint8_t *rkv = memalign(128, KV_BYTES);

  fill_weights(w);
  memset(act, 0, P.atotal);
  memset(kv, 0, KV_BYTES);
  build_tiny_oplist(ol);
  memcpy(rw, w, P.wtotal);
  memcpy(ract, act, P.atotal);
  memcpy(rkv, kv, KV_BYTES);

  /* Negative: corrupted head_dim must be rejected by init validation. */
  {
    uint8_t *bad = malloc(OPLIST_LEN);
    memcpy(bad, ol, OPLIST_LEN);
    ((struct nntr_htp_oplist_header *)(void *)bad)->head_dim = 64u;
    if (htp_graph_init(&g, bad, OPLIST_LEN, w, P.wtotal, kv, KV_BYTES, act,
                       P.atotal) == 0) {
      printf("SIM_TEST graph FAIL corrupted head_dim accepted\n");
      rc = 1;
    }
    free(bad);
  }

  if (!rc) {
    if (htp_graph_init(&g, ol, OPLIST_LEN, w, P.wtotal, kv, KV_BYTES, act,
                       P.atotal)) {
      printf("SIM_TEST graph FAIL init\n");
      rc = 1;
    } else {
      inited = 1;
    }
  }

  /* Negative: forward runtime-argument violations must be rejected. */
  if (!rc &&
      (htp_graph_forward(&g, prefill_tok, 0, 0, logits, VOCAB) == 0 ||
       htp_graph_forward(&g, prefill_tok, MAX_CHUNK + 1u, 0, logits, VOCAB) ==
         0 ||
       htp_graph_forward(&g, prefill_tok, MAX_CHUNK, MAX_SEQ - MAX_CHUNK + 1u,
                         logits, VOCAB) == 0 ||
       htp_graph_forward(&g, prefill_tok, 1, 0, logits, VOCAB - 1u) == 0)) {
    printf("SIM_TEST graph FAIL bad forward args accepted\n");
    rc = 1;
  }

  /* Prefill 8 tokens at pos 0, compare final logits vs the reference. */
  if (!rc) {
    if (htp_graph_forward(&g, prefill_tok, MAX_CHUNK, 0, logits, VOCAB)) {
      printf("SIM_TEST graph FAIL prefill forward\n");
      rc = 1;
    } else {
      ref_graph_forward(ol, rw, rkv, ract, prefill_tok, MAX_CHUNK, 0, rlogits);
      rc = cmp_f("graph_prefill", rlogits, logits, VOCAB, 3e-2f, 5e-2f);
    }
  }

  /* Decode 1 token at pos 8 reusing the KV cache. */
  if (!rc) {
    if (htp_graph_forward(&g, &decode_tok, 1, MAX_CHUNK, logits, VOCAB)) {
      printf("SIM_TEST graph FAIL decode forward\n");
      rc = 1;
    } else {
      ref_graph_forward(ol, rw, rkv, ract, &decode_tok, 1, MAX_CHUNK, rlogits);
      rc = cmp_f("graph_decode", rlogits, logits, VOCAB, 3e-2f, 5e-2f);
    }
  }

  /* Partial execution matches the reference executor at the same cut. */
  if (!rc) {
    const uint32_t cut = 9u; /* right after layer 0's ATTN */
    const uint32_t n_attn = 3u * QDIM;
    uint64_t pc = 0;
    float *fa = malloc(n_attn * sizeof(float));
    float *fb = malloc(n_attn * sizeof(float));
    uint32_t i;
    memset(kv, 0, KV_BYTES);
    memset(act, 0, P.atotal);
    if (htp_graph_forward_upto(&g, prefill_tok, 3u, 0u, logits, VOCAB, cut,
                               &pc)) {
      printf("SIM_TEST graph FAIL partial forward\n");
      rc = 1;
    } else {
      memset(rkv, 0, KV_BYTES);
      memset(ract, 0, P.atotal);
      ref_graph_forward_upto(ol, rw, rkv, ract, prefill_tok, 3u, 0u, rlogits,
                             cut);
      for (i = 0; i < n_attn; ++i) {
        fa[i] = (float)((const __fp16 *)(const void *)(act + P.attn))[i];
        fb[i] = (float)((const __fp16 *)(const void *)(ract + P.attn))[i];
      }
      rc = cmp_f("graph_partial_attn", fb, fa, n_attn, 1e-2f, 1e-2f);
      if (!rc && pc == 0) {
        printf("SIM_TEST graph FAIL pcycles did not advance\n");
        rc = 1;
      }
    }
    free(fa);
    free(fb);
  }

  /* Out-of-range limits and dump refs are rejected, not clamped. */
  if (!rc) {
    uint64_t pc = 0;
    if (htp_graph_forward_upto(&g, prefill_tok, 3u, 0u, logits, VOCAB,
                               N_OPS + 1u, &pc) == 0 ||
        htp_graph_buf_ref(&g, NNTR_HTP_BUF_COUNT, 0u, 4u) != 0 ||
        htp_graph_buf_ref(&g, NNTR_HTP_BUF_TOKENS, 0u, 4u) != 0 ||
        htp_graph_buf_ref(&g, NNTR_HTP_BUF_ACT, P.atotal - 64u, 128u) != 0 ||
        htp_graph_buf_ref(&g, NNTR_HTP_BUF_ACT, P.attn, 128u) != act + P.attn) {
      printf("SIM_TEST graph FAIL bad limit/dump ref accepted\n");
      rc = 1;
    }
  }

  /* Partial execution: running ops [0, N_OPS) must equal a full run (the
   * executor keeps no per-call state besides KV), and a truncated run must
   * leave the logits buffer untouched. */
  if (!rc) {
    float la[VOCAB], lb[VOCAB];
    uint32_t i;
    memset(rkv, 0, KV_BYTES);
    ref_graph_forward(ol, rw, rkv, ract, prefill_tok, 3u, 0, la);
    memset(rkv, 0, KV_BYTES);
    ref_graph_forward_upto(ol, rw, rkv, ract, prefill_tok, 3u, 0, lb, N_OPS);
    for (i = 0; i < VOCAB && !rc; ++i)
      if (la[i] != lb[i])
        rc = 1;
    memset(rkv, 0, KV_BYTES);
    memset(lb, 0, sizeof(lb));
    ref_graph_forward_upto(ol, rw, rkv, ract, prefill_tok, 3u, 0, lb,
                           N_OPS - 1u);
    if (lb[0] != 0.0f)
      rc = 1;
    if (rc)
      printf("SIM_TEST graph FAIL ref partial execution\n");
  }

  if (inited)
    htp_graph_destroy(&g);
  free(rkv);
  free(ract);
  free(rw);
  free(ol);
  free(kv);
  free(act);
  free(w);

  if (rc)
    return 1;

  printf("SIM_TEST graph PASS\n");
  return 0;
}

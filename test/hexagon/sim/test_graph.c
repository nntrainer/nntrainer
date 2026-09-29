// SPDX-License-Identifier: Apache-2.0
/**
 * @file	test_graph.c
 * @date	19 August 2026
 * @brief	Hexagon-sim golden test for the op-list graph executor: a
 *		synthetic 2-layer model (35 ops), prefill then decode compared
 *		against the scalar reference executor, plus negative cases
 *		for init validation (header and per-op shape rules) and
 *		forward runtime-argument checks (counts, position, token ids)
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <qurt.h>

#include "htp_graph.h"
#include "ref_ops.h"
#include "sim_model.h"
#include "sim_test_util.h"
#include "worker_pool.h"

#define N_LAYERS 2u
#define HIDDEN 256u
#define N_HEADS 4u
#define N_KV_HEADS 2u
#define HEAD_DIM 128u
#define FFN 512u
#define VOCAB 512u
#define MAX_SEQ 64u
#define MAX_CHUNK 8u
/** Note n_heads*head_dim = 512 != hidden = 256: Q/K/V project hidden ->
 * heads*head_dim and O projects back. */
#define QDIM (N_HEADS * HEAD_DIM)     /* 512 */
#define KVDIM (N_KV_HEADS * HEAD_DIM) /* 256 */
#define N_OPS 35u
#define OPLIST_LEN                                                             \
  ((uint32_t)(sizeof(struct nntr_htp_oplist_header) +                          \
              N_OPS * sizeof(struct nntr_htp_op_desc)))
#define KV_BYTES (2u * N_LAYERS * N_KV_HEADS * MAX_SEQ * HEAD_DIM * 2u)
#define EPS 1e-5f

static struct sim_model_plan P;
static const struct sim_model_cfg TINY = {
  N_LAYERS, N_HEADS, N_KV_HEADS, HEAD_DIM, HIDDEN, FFN,
  VOCAB,    MAX_SEQ, MAX_CHUNK,  EPS,      1e6f,
};

/** One per-op mutation of the sim-model op-list that the validator must
 * refuse (rc 5): which descriptor, which field, and the value to write. */
enum bad_op_field { F_LAYER, F_M, F_K, F_N, F_IN0_OFF, F_IN2_OFF };
struct bad_op_case {
  const char *name;
  uint32_t op;   /* index into the 35-op list */
  uint32_t kind; /* expected kind at that index (guards the index map) */
  enum bad_op_field field;
  uint32_t value;
};

int test_graph(void) {
  static const int32_t prefill_tok[MAX_CHUNK] = {1,  17, 300, 511,
                                                 42, 7,  129, 256};
  static const int32_t decode_tok = 99;
  float logits[VOCAB], rlogits[VOCAB];
  struct htp_graph g;
  int rc = 0, inited = 0;

  if (sim_model_plan_init(&P, &TINY)) {
    printf("SIM_TEST graph FAIL plan alloc\n");
    return 1;
  }
  if (P.n_ops != N_OPS || P.oplist_len != OPLIST_LEN ||
      P.kv_bytes != KV_BYTES) {
    printf("SIM_TEST graph FAIL plan sizes n_ops=%u len=%u kv=%u\n",
           (unsigned)P.n_ops, (unsigned)P.oplist_len, (unsigned)P.kv_bytes);
    sim_model_plan_free(&P);
    return 1;
  }
  uint8_t *w = memalign(128, P.wtotal);
  uint8_t *act = memalign(128, P.atotal);
  uint8_t *kv = memalign(128, KV_BYTES);
  uint8_t *ol = memalign(128, OPLIST_LEN);
  /* Reference executor runs on its own copies of every buffer. */
  uint8_t *rw = memalign(128, P.wtotal);
  uint8_t *ract = memalign(128, P.atotal);
  uint8_t *rkv = memalign(128, KV_BYTES);

  sim_model_fill_weights(&P, w);
  memset(act, 0, P.atotal);
  memset(kv, 0, KV_BYTES);
  {
    int vrc = sim_model_build_oplist(&P, ol);
    if (vrc) {
      printf("SIM_TEST graph FAIL sim_model validate rc=%d\n", vrc);
      rc = 1;
      goto out;
    }
  }
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

  /** Negative: an op-list without the tiled32 weight_layout id (a v3-era
   * producer, or an unknown layout) must be rejected at init. */
  {
    uint8_t *bad = malloc(OPLIST_LEN);
    memcpy(bad, ol, OPLIST_LEN);
    ((struct nntr_htp_oplist_header *)(void *)bad)->weight_layout = 0u;
    if (htp_graph_init(&g, bad, OPLIST_LEN, w, P.wtotal, kv, KV_BYTES, act,
                       P.atotal) == 0) {
      printf("SIM_TEST graph FAIL weight_layout 0 accepted\n");
      rc = 1;
    }
    free(bad);
  }

  /** Negative: per-op shape rules no kernel can execute must be rejected
   * at init. Op indices follow sim_model_build_oplist: 0 EMBED, layer-0 ops
   * 1..16 (RMSNORM, W8A8 q/k/v, PER_HEAD RMSNORM q/k, ROPE, ATTN, W8A8 o,
   * ADD, RMSNORM, W8A8 gate/up, SILU_MUL, W8A16, ADD), 33 final RMSNORM,
   * 34 LOGITS. */
  {
    const struct bad_op_case cases[] = {
      {"attn_layer", 8u, NNTR_HTP_OP_ATTN, F_LAYER, N_LAYERS},
      {"k0_w8a8", 2u, NNTR_HTP_OP_MATMUL_W8A8, F_K, 0u},
      {"k0_w8a16", 15u, NNTR_HTP_OP_MATMUL_W8A16, F_K, 0u},
      {"k0_logits", 34u, NNTR_HTP_OP_MATMUL_LOGITS, F_K, 0u},
      {"k0_embed", 0u, NNTR_HTP_OP_EMBED, F_K, 0u},
      {"rmsnorm_n65", 1u, NNTR_HTP_OP_RMSNORM, F_N, 65u},
      /* a multiple of 64 that is not a multiple of head_dim */
      {"perhead_n192", 5u, NNTR_HTP_OP_RMSNORM, F_N, HEAD_DIM + 64u},
      {"add_n65", 10u, NNTR_HTP_OP_ADD, F_N, 65u},
      {"silu_n65", 14u, NNTR_HTP_OP_SILU_MUL, F_N, 65u},
      /* one hidden row fits at the end of ACT, MAX_CHUNK rows do not */
      {"logits_in0_short", 34u, NNTR_HTP_OP_MATMUL_LOGITS, F_IN0_OFF,
       P.atotal - HIDDEN * 2u},
      /* ATTN's fresh V rows (MAX_CHUNK * KVDIM halves) past the end of ACT */
      {"attn_in2_short", 8u, NNTR_HTP_OP_ATTN, F_IN2_OFF, P.atotal - 128u},
      /* a descriptor-fixed m on any kind but LOGITS */
      {"fixed_m", 1u, NNTR_HTP_OP_RMSNORM, F_M, MAX_CHUNK},
    };
    const uint32_t n_cases = (uint32_t)(sizeof(cases) / sizeof(cases[0]));
    const uint32_t hdr_len = (uint32_t)sizeof(struct nntr_htp_oplist_header);
    uint8_t *bad = malloc(OPLIST_LEN);
    uint32_t ci;
    for (ci = 0; ci < n_cases && !rc; ++ci) {
      const struct bad_op_case *bc = &cases[ci];
      struct nntr_htp_op_desc *ops;
      memcpy(bad, ol, OPLIST_LEN);
      ops = (struct nntr_htp_op_desc *)(void *)(bad + hdr_len);
      if (ops[bc->op].kind != bc->kind) {
        printf("SIM_TEST graph FAIL %s: op %u is kind %u\n", bc->name,
               (unsigned)bc->op, (unsigned)ops[bc->op].kind);
        rc = 1;
        break;
      }
      switch (bc->field) {
      case F_LAYER:
        ops[bc->op].layer = bc->value;
        break;
      case F_M:
        ops[bc->op].m = bc->value;
        break;
      case F_K:
        ops[bc->op].k = bc->value;
        break;
      case F_N:
        ops[bc->op].n = bc->value;
        break;
      case F_IN0_OFF:
        ops[bc->op].in0.offset = bc->value;
        break;
      case F_IN2_OFF:
        ops[bc->op].in2.offset = bc->value;
        break;
      }
      /* rc 5 (bad op), not merely nonzero: an older rule must not mask it */
      {
        const int irc = htp_graph_init(&g, bad, OPLIST_LEN, w, P.wtotal, kv,
                                       KV_BYTES, act, P.atotal);
        if (irc != 5) {
          printf("SIM_TEST graph FAIL %s: init rc %d, expected 5\n", bc->name,
                 irc);
          rc = 1;
        }
      }
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

  /** Negative: a token id >= vocab (a negative id lands there through the
   * uint32 cast) must be rejected before KV/ACT are touched. Both are still
   * all-zero here, as are their reference copies. VOCAB - 1 is accepted
   * below as the last id of prefill_tok. */
  if (!rc) {
    static const int32_t id_vocab = (int32_t)VOCAB;
    static const int32_t id_neg = -1; /* 0xFFFFFFFF as uint32 */
    int32_t bad_last[MAX_CHUNK];
    memcpy(bad_last, prefill_tok, sizeof(bad_last));
    bad_last[MAX_CHUNK - 1u] = (int32_t)VOCAB;
    if (htp_graph_forward(&g, &id_vocab, 1, 0, logits, VOCAB) == 0 ||
        htp_graph_forward(&g, &id_neg, 1, 0, logits, VOCAB) == 0 ||
        htp_graph_forward(&g, bad_last, MAX_CHUNK, 0, logits, VOCAB) == 0) {
      printf("SIM_TEST graph FAIL bad token id accepted\n");
      rc = 1;
    } else if (memcmp(kv, rkv, KV_BYTES) != 0 ||
               memcmp(act, ract, P.atotal) != 0) {
      printf("SIM_TEST graph FAIL rejected forward touched KV/ACT\n");
      rc = 1;
    }
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
    htp_graph_profile_reset(&g);
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
      if (!rc) {
        /** Per-kind profile: 9 ops ran, their cycles must account for the
         * whole loop (the loop's own timer reads are the only slack). */
        uint64_t pk[NNTR_HTP_OP_KIND_COUNT];
        uint32_t ck[NNTR_HTP_OP_KIND_COUNT];
        uint64_t sum = 0;
        uint32_t calls = 0, kk;
        htp_graph_profile_get(&g, pk, ck);
        for (kk = 0; kk < (uint32_t)NNTR_HTP_OP_KIND_COUNT; ++kk) {
          sum += pk[kk];
          calls += ck[kk];
        }
        /** cut == 9: EMBED, RMSNORM x3 (attn_norm, q_norm, k_norm),
         * MATMUL_W8A8 x3, ROPE, ATTN */
        if (calls != cut || ck[NNTR_HTP_OP_EMBED] != 1u ||
            ck[NNTR_HTP_OP_RMSNORM] != 3u ||
            ck[NNTR_HTP_OP_MATMUL_W8A8] != 3u || ck[NNTR_HTP_OP_ROPE] != 1u ||
            ck[NNTR_HTP_OP_ATTN] != 1u || ck[NNTR_HTP_OP_ADD] != 0u) {
          printf("SIM_TEST graph FAIL profile calls (total %u)\n",
                 (unsigned)calls);
          rc = 1;
        } else if (sum > pc || sum * 100u < pc * 99u) {
          printf("SIM_TEST graph FAIL profile cycles sum=%llu loop=%llu\n",
                 (unsigned long long)sum, (unsigned long long)pc);
          rc = 1;
        } else {
          /** Per-op counters: same window, sum equals the per-kind sum, op
           * `cut` (not run) stays 0, op 0 (EMBED, ran) is non-zero. */
          uint64_t po[N_OPS], osum = 0;
          if (htp_graph_profile_get_ops(&g, po, N_OPS)) {
            printf("SIM_TEST graph FAIL profile_get_ops\n");
            rc = 1;
          } else {
            for (kk = 0; kk < N_OPS; ++kk)
              osum += po[kk];
            if (osum != sum || po[cut] != 0u || po[0] == 0u) {
              printf("SIM_TEST graph FAIL profile per-op sum=%llu kinds=%llu\n",
                     (unsigned long long)osum, (unsigned long long)sum);
              rc = 1;
            }
          }
        }
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

  /** Partial execution: running ops [0, N_OPS) must equal a full run (the
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

  /** Worker-count independence: the same prefill with a forced 2-worker
   * pool must match the reference too (the sim has 4 HVX units; the
   * device count differs, so kernels may not bake either in). */
  if (!rc && inited) {
    htp_graph_destroy(&g);
    inited = 0;
    memset(kv, 0, KV_BYTES);
    memset(act, 0, P.atotal);
    memset(rkv, 0, KV_BYTES);
    memset(ract, 0, P.atotal);
    if (htp_graph_init_ex(&g, ol, OPLIST_LEN, w, P.wtotal, kv, KV_BYTES, act,
                          P.atotal, 2)) {
      printf("SIM_TEST graph FAIL init_ex(2)\n");
      rc = 1;
    } else {
      inited = 1;
      if (wp_size(g.ctx.pool) != 2) {
        printf("SIM_TEST graph FAIL init_ex pool size %d != 2\n",
               wp_size(g.ctx.pool));
        rc = 1;
      } else if (htp_graph_forward(&g, prefill_tok, MAX_CHUNK, 0, logits,
                                   VOCAB)) {
        printf("SIM_TEST graph FAIL prefill forward (2 workers)\n");
        rc = 1;
      } else {
        ref_graph_forward(ol, rw, rkv, ract, prefill_tok, MAX_CHUNK, 0,
                          rlogits);
        rc =
          cmp_f("graph_prefill_2workers", rlogits, logits, VOCAB, 3e-2f, 5e-2f);
      }
    }
  }

  /** Over-large request must be clamped to the 128B-mode HVX unit count: a
   * worker without a unit blocks forever in qurt_hvx_lock(). No forward
   * needed, the pool size is the whole check. */
  if (!rc && inited) {
    const int units = (qurt_hvx_get_units() >> 8) & 0xFF;
    const int expect = units > 0 ? units : 1;
    htp_graph_destroy(&g);
    inited = 0;
    memset(kv, 0, KV_BYTES);
    memset(act, 0, P.atotal);
    if (htp_graph_init_ex(&g, ol, OPLIST_LEN, w, P.wtotal, kv, KV_BYTES, act,
                          P.atotal, 64)) {
      printf("SIM_TEST graph FAIL init_ex(64)\n");
      rc = 1;
    } else {
      inited = 1;
      if (wp_size(g.ctx.pool) != expect) {
        printf("SIM_TEST graph FAIL init_ex clamp pool size %d != %d\n",
               wp_size(g.ctx.pool), expect);
        rc = 1;
      }
    }
  }

out:
  if (inited)
    htp_graph_destroy(&g);
  free(rkv);
  free(ract);
  free(rw);
  free(ol);
  free(kv);
  free(act);
  free(w);
  sim_model_plan_free(&P);

  if (rc)
    return 1;

  printf("SIM_TEST graph PASS\n");
  return 0;
}

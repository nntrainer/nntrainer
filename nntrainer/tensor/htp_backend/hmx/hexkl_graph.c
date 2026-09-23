// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hexkl_graph.c
 * @date   23 Sep 2026
 * @brief  The session's validated decode op table and its forward loop (#85)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Address space (LEDGER rule 8): one hexkl_graph is HTP_GRAPH_MAX_OPS x
 * (304 + 8) B of table, about 80 KiB, plus HTP_GRAPH_N_SLOTS x slot_words
 * f32 of activation slots (24 KiB with only MOE resident at hidden 2048),
 * all DSP heap, no arena, no VTCM, no DMA. That is noise against the
 * ~182 MiB heap and needs no arena accounting.
 */

#include "hexkl_graph.h"

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_perf.h>

/* htp_graph_desc.h restates the SDK's codes so it can be built with no
   SDK; here both are in scope, so a drift is a build error. */
#if HTP_GRAPH_E_CLASSNOTSUPPORT != AEE_ECLASSNOTSUPPORT ||                     \
  HTP_GRAPH_E_BADSTATE != AEE_EBADSTATE ||                                     \
  HTP_GRAPH_E_BADITEM != AEE_EBADITEM ||                                       \
  HTP_GRAPH_E_INVALIDFORMAT != AEE_EINVALIDFORMAT ||                           \
  HTP_GRAPH_E_INCOMPLETEITEM != AEE_EINCOMPLETEITEM ||                         \
  HTP_GRAPH_E_UNSUPPORTED != AEE_EUNSUPPORTED ||                               \
  HTP_GRAPH_E_NOTYPE != AEE_ENOTYPE ||                                         \
  HTP_GRAPH_E_INVALIDITEM != AEE_EINVALIDITEM ||                               \
  HTP_GRAPH_E_INVHANDLE != AEE_EINVHANDLE
#error "htp_graph_desc.h's error codes drifted from AEEStdErr.h"
#endif

/** @brief What one forward call carries to the kernels. */
typedef struct {
  const hexkl_graph_env *env;
  hexkl_graph_routing routing; /**< n_experts 0 once consumed */
} graph_call;

typedef int (*graph_kernel)(hexkl_graph *g, const htp_graph_op *op,
                            graph_call *call, const float *in, float *out);

/* ---- MOE: hexkl_mm_u8i4_moe_layer_run at M = 1, unchanged -------------- */

static int graph_op_moe(hexkl_graph *g, const htp_graph_op *op,
                        graph_call *call, const float *in, float *out) {
  const hexkl_graph_env *env = call->env;
  const hexkl_graph_routing *r = &call->routing;
  uint32_t e, sum = 0;
  (void)g;
  /* The same checks as the per-layer entry (nntr_hvx_mm_u8i4.c
     check_moe_layer_args / check_moe_row_totals), with M fixed at 1: every
     row index is 0 and no expert sees the token twice. */
  if (r->n_experts == 0u) {
    return AEE_EBADSTATE;
  }
  if (r->n_experts != op->n_experts || r->n_rows == 0u ||
      r->n_rows > op->top_k) {
    return AEE_EINVALIDFORMAT;
  }
  for (e = 0; e < r->n_experts; ++e) {
    if (r->row_count[e] > 1u) {
      return AEE_EINVALIDFORMAT;
    }
    sum += r->row_count[e];
  }
  if (sum != r->n_rows) {
    return AEE_EINVALIDFORMAT;
  }
  for (e = 0; e < r->n_rows; ++e) {
    if (r->row_index[e] != 0u) {
      return AEE_EINVALIDFORMAT;
    }
  }
  call->routing.n_experts = 0u; /* consumed */
  return hexkl_mm_u8i4_moe_layer_run(
    env->tbl, env->vtcm_base, env->vtcm_size, env->config_off, 1u, op->K, op->N,
    op->N_out, op->n_experts, op->h_gu, op->h_dn, r->row_index, r->row_count,
    r->row_weight, in, out, env->pool, env->scratch, env->moe_flags);
}

/** @brief The kernel table: a NULL slot is a kind this build does not run
 *  (hvx_impl's htp_op_table rule); the validator refuses a resident bit on
 *  it with AEE_ECLASSNOTSUPPORT, so forward never reaches a NULL. */
static const graph_kernel kernels[HTP_OP_KIND_N] = {
  NULL,         /* RMSNORM      #82 */
  NULL,         /* FC           plan 85 section 7 */
  NULL,         /* CONV1D_GATE  #82 */
  NULL,         /* QK_NORM      #82 */
  NULL,         /* ROPE         #82 */
  NULL,         /* ATTN_M1      #81 */
  NULL,         /* ADD */
  NULL,         /* ROUTER_TOPK */
  graph_op_moe, /* MOE */
  NULL,         /* DENSE_FFN */
  NULL,         /* LM_HEAD */
};

uint32_t hexkl_graph_resident_kinds(void) {
  uint32_t mask = 0, k;
  for (k = 0; k < HTP_OP_KIND_N; ++k) {
    if (kernels[k] != NULL) {
      mask |= HTP_GRAPH_KIND_BIT(k);
    }
  }
  return mask;
}

static int graph_check_handle(const hexkl_weight_u8i4_table *tbl, uint32_t h,
                              uint32_t K, uint32_t N) {
  if (h >= HEXKL_MM_U8I4_MAX_WEIGHTS || !tbl->slots[h].in_use ||
      tbl->slots[h].K != K || tbl->slots[h].N != N) {
    return AEE_EINVHANDLE;
  }
  return AEE_SUCCESS;
}

int hexkl_graph_init(const uint32_t *words, uint32_t n_words,
                     const hexkl_weight_u8i4_table *tbl, hexkl_graph **out) {
  uint32_t n_ops = 0, i, e, slot_words = 0;
  hexkl_graph *g;
  int rc;
  if (out == NULL || tbl == NULL) {
    return AEE_EBADSTATE;
  }
  rc = (int)htp_graph_validate(words, n_words, hexkl_graph_resident_kinds(),
                               &n_ops);
  if (rc != 0) {
    return rc;
  }
  for (i = 0; i < n_ops; ++i) {
    const htp_graph_op *op = htp_graph_op_cat(words, i);
    if (!op->resident) {
      continue;
    }
    if (op->kind == HTP_OP_MOE) {
      for (e = 0; e < op->n_experts; ++e) {
        if (graph_check_handle(tbl, op->h_gu[e], op->K, 2u * op->N) !=
              AEE_SUCCESS ||
            graph_check_handle(tbl, op->h_dn[e], op->N, op->N_out) !=
              AEE_SUCCESS) {
          return AEE_EINVHANDLE;
        }
      }
    }
    if (htp_graph_op_in_words(op) > slot_words) {
      slot_words = htp_graph_op_in_words(op);
    }
    if (htp_graph_op_out_words(op) > slot_words) {
      slot_words = htp_graph_op_out_words(op);
    }
  }
  g = (hexkl_graph *)calloc(1, sizeof(*g));
  if (g == NULL) {
    return AEE_ENOMEMORY;
  }
  g->n_layers = words[2];
  g->n_ops = n_ops;
  g->hidden = words[4];
  g->vocab = words[5];
  g->max_seq = words[6];
  g->slot_words = slot_words;
  if (slot_words != 0u) {
    g->slots =
      (float *)calloc((size_t)HTP_GRAPH_N_SLOTS * slot_words, sizeof(float));
    if (g->slots == NULL) {
      free(g);
      return AEE_ENOMEMORY;
    }
  }
  for (i = 0; i < n_ops; ++i) {
    g->ops[i] = *htp_graph_op_cat(words, i);
  }
  *out = g;
  return AEE_SUCCESS;
}

void hexkl_graph_free(hexkl_graph *g) {
  if (g == NULL) {
    return;
  }
  free(g->slots);
  free(g);
}

int hexkl_graph_uses_handle(const hexkl_graph *g, uint32_t handle) {
  uint32_t i, e;
  if (g == NULL) {
    return 0;
  }
  for (i = 0; i < g->n_ops; ++i) {
    const htp_graph_op *op = &g->ops[i];
    if (!op->resident || op->kind != HTP_OP_MOE) {
      continue;
    }
    for (e = 0; e < op->n_experts; ++e) {
      if (op->h_gu[e] == handle || op->h_dn[e] == handle) {
        return 1;
      }
    }
  }
  return 0;
}

int hexkl_graph_forward(hexkl_graph *g, const hexkl_graph_env *env,
                        uint32_t start_op, uint32_t n_ops_limit, uint32_t pos,
                        const hexkl_graph_routing *routing, const float *act_in,
                        uint32_t act_in_len, float *act_out,
                        uint32_t act_out_len, uint32_t *resume_at) {
  graph_call call;
  uint32_t i, n_run = 0;
  const htp_graph_op *last = NULL;
  if (g == NULL || env == NULL || resume_at == NULL) {
    return AEE_EBADSTATE;
  }
  memset(g->op_pcycles, 0, sizeof(g->op_pcycles));
  if (start_op >= g->n_ops || pos >= g->max_seq) {
    return AEE_EBADITEM;
  }
  if (act_in == NULL ||
      act_in_len != htp_graph_op_in_words(&g->ops[start_op])) {
    return AEE_EINVALIDFORMAT;
  }
  /* Which op will be the last one run is known before anything runs, so
     act_out is checked now rather than after the work was done. */
  for (i = start_op; i < g->n_ops && n_run < n_ops_limit && g->ops[i].resident;
       ++i, ++n_run) {
    last = &g->ops[i];
  }
  if (last != NULL &&
      (act_out == NULL || act_out_len != htp_graph_op_out_words(last))) {
    return AEE_EINVALIDFORMAT;
  }
  last = NULL;
  n_run = 0;
  call.env = env;
  if (routing != NULL) {
    call.routing = *routing;
  } else {
    memset(&call.routing, 0, sizeof(call.routing));
  }
  for (i = start_op; i < g->n_ops && n_run < n_ops_limit; ++i) {
    const htp_graph_op *op = &g->ops[i];
    const graph_kernel k = kernels[op->kind];
    float *in, *out;
    uint64_t t0;
    int rc;
    if (!op->resident) {
      break;
    }
    if (k == NULL) {
      return AEE_ECLASSNOTSUPPORT;
    }
    in = g->slots + (size_t)op->in_slot * g->slot_words;
    out = g->slots + (size_t)op->out_slot * g->slot_words;
    if (n_run == 0u) {
      memcpy(in, act_in, (size_t)act_in_len * sizeof(float));
    }
    t0 = HAP_perf_get_pcycles();
    rc = k(g, op, &call, in, out);
    g->op_pcycles[i] = HAP_perf_get_pcycles() - t0;
    if (rc != AEE_SUCCESS) {
      return rc;
    }
    last = op;
    ++n_run;
  }
  *resume_at = i;
  if (last != NULL) {
    if (act_out == NULL || act_out_len != htp_graph_op_out_words(last)) {
      return AEE_EINVALIDFORMAT;
    }
    memcpy(act_out, g->slots + (size_t)last->out_slot * g->slot_words,
           (size_t)act_out_len * sizeof(float));
  }
  return AEE_SUCCESS;
}

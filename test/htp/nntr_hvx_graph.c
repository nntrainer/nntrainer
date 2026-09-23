// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   nntr_hvx_graph.c
 * @date   23 Sep 2026
 * @brief  The per-token entry: graph_init / graph_release / forward /
 *         forward_debug over the session's hexkl_graph (#85)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>

#include "hexkl_graph.h"
#include "hexkl_probe.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

int nntr_hvx_graph_init(remote_handle64 handle, const uint32 *desc, int descLen,
                        uint32 *n_ops) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  int rc;
  if (!s || !desc || descLen <= 0 || !n_ops) {
    return AEE_EBADPARM;
  }
  if (s->graph != NULL) {
    FARF(ERROR, "graph_init: a graph is already live; release it first");
    return AEE_EBADSTATE;
  }
  rc = hexkl_graph_init(desc, (uint32_t)descLen, &s->weights_u8i4, &s->graph);
  if (rc != AEE_SUCCESS) {
    FARF(ERROR, "graph_init: %s (0x%08x), %d words", htp_graph_err_name(rc),
         (unsigned)rc, descLen);
    return rc;
  }
  *n_ops = s->graph->n_ops;
  FARF(HIGH,
       "[graph] init n_ops=%u layers=%u hidden=%u max_seq=%u resident=0x%x",
       (unsigned)s->graph->n_ops, (unsigned)s->graph->n_layers,
       (unsigned)s->graph->hidden, (unsigned)s->graph->max_seq,
       (unsigned)hexkl_graph_resident_kinds());
  return AEE_SUCCESS;
}

int nntr_hvx_graph_release(remote_handle64 handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_EBADPARM;
  }
  hexkl_graph_free(s->graph);
  s->graph = NULL;
  return AEE_SUCCESS;
}

/** @brief The sequences' shape checks the kernel cannot make (it sees
 *  pointers, not lengths), shared by both entries. */
static int graph_check_args(const nntr_hvx_session *s, int row_indexLen,
                            int row_countLen, int row_weightLen, int act_inLen,
                            int act_outLen, const uint32 *resume_at) {
  if (!s || !resume_at || row_indexLen < 0 || row_countLen < 0 ||
      act_inLen < 0 || act_outLen < 0) {
    return AEE_EBADPARM;
  }
  if (s->graph == NULL) {
    FARF(ERROR, "forward: no graph; call graph_init first");
    return AEE_EBADSTATE;
  }
  if (row_weightLen != row_indexLen) {
    FARF(ERROR, "forward: row_index %d vs row_weight %d", row_indexLen,
         row_weightLen);
    return AEE_EINVALIDFORMAT;
  }
  return AEE_SUCCESS;
}

static void graph_env_of(const nntr_hvx_session *s, hexkl_graph_env *env) {
  env->tbl = (hexkl_weight_u8i4_table *)&s->weights_u8i4;
  env->vtcm_base = s->vtcm_base;
  env->vtcm_size = s->vtcm_size;
  env->config_off = s->config_off;
  env->pool = s->quant_pool;
  env->scratch = (hexkl_moe_scratch *)&s->moe_scratch;
  env->moe_flags = s->moe_flags;
}

/** @brief One FARF line per call (HIGH: silent unless the mask enables
 *  it), the per-op pcycles summed and the MoE op's own. */
static void graph_farf(const hexkl_graph *g, uint32_t start, uint32_t resume) {
  uint64_t total = 0, moe = 0;
  uint32_t i;
  for (i = start; i < resume; ++i) {
    total += g->op_pcycles[i];
    if (g->ops[i].kind == HTP_OP_MOE) {
      moe += g->op_pcycles[i];
    }
  }
  FARF(HIGH, "[graph] start=%u resume=%u ops=%u pcyc=%llu moe=%llu",
       (unsigned)start, (unsigned)resume, (unsigned)(resume - start),
       (unsigned long long)total, (unsigned long long)moe);
  (void)total;
  (void)moe;
}

int nntr_hvx_forward(remote_handle64 handle, uint32 start_op, uint32 pos,
                     const uint32 *row_index, int row_indexLen,
                     const uint32 *row_count, int row_countLen,
                     const float *row_weight, int row_weightLen,
                     const float *act_in, int act_inLen, float *act_out,
                     int act_outLen, uint32 *resume_at) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  hexkl_graph_env env;
  hexkl_graph_routing routing;
  int rc = graph_check_args(s, row_indexLen, row_countLen, row_weightLen,
                            act_inLen, act_outLen, resume_at);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  graph_env_of(s, &env);
  routing.row_index = row_index;
  routing.row_count = row_count;
  routing.row_weight = row_weight;
  routing.n_rows = (uint32_t)row_indexLen;
  routing.n_experts = (uint32_t)row_countLen;
  rc = hexkl_graph_forward(s->graph, &env, start_op, HTP_GRAPH_MAX_OPS, pos,
                           &routing, act_in, (uint32_t)act_inLen, act_out,
                           (uint32_t)act_outLen, resume_at);
  if (rc != AEE_SUCCESS) {
    FARF(ERROR, "forward: start=%u pos=%u: %s (0x%08x)", (unsigned)start_op,
         (unsigned)pos, htp_graph_err_name(rc), (unsigned)rc);
    return rc;
  }
  graph_farf(s->graph, start_op, *resume_at);
  return AEE_SUCCESS;
}

int nntr_hvx_forward_debug(remote_handle64 handle, uint32 start_op,
                           uint32 n_ops_limit, uint32 pos,
                           const uint32 *row_index, int row_indexLen,
                           const uint32 *row_count, int row_countLen,
                           const float *row_weight, int row_weightLen,
                           const float *act_in, int act_inLen, float *act_out,
                           int act_outLen, uint32 *resume_at,
                           uint32 *op_pcycles, int op_pcyclesLen,
                           uint32 *stage_us, int stage_usLen) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  hexkl_graph_env env;
  hexkl_graph_routing routing;
  uint64_t t0, t1;
  uint32_t i;
  int rc = graph_check_args(s, row_indexLen, row_countLen, row_weightLen,
                            act_inLen, act_outLen, resume_at);
  if (rc != AEE_SUCCESS) {
    return rc;
  }
  if (!op_pcycles || op_pcyclesLen < 0 ||
      (uint32_t)op_pcyclesLen != n_ops_limit) {
    FARF(ERROR, "forward_debug: op_pcyclesLen %d, n_ops_limit %u",
         op_pcyclesLen, (unsigned)n_ops_limit);
    return AEE_EINVALIDFORMAT;
  }
  if (!stage_us || stage_usLen < 0 ||
      (uint32_t)stage_usLen != nntr_hvx_moe_stage_count()) {
    /* The same stale-count symptom as moe_layer_timed, and the same
       code, on purpose: the ARM side already names it. */
    FARF(ERROR, "forward_debug: stage_usLen %d, expected %u", stage_usLen,
         (unsigned)nntr_hvx_moe_stage_count());
    return AEE_EBADPARM;
  }
  graph_env_of(s, &env);
  routing.row_index = row_index;
  routing.row_count = row_count;
  routing.row_weight = row_weight;
  routing.n_rows = (uint32_t)row_indexLen;
  routing.n_experts = (uint32_t)row_countLen;

  hexkl_probe_reset(1);
  t0 = hexkl_probe_now();
  rc = hexkl_graph_forward(s->graph, &env, start_op, n_ops_limit, pos, &routing,
                           act_in, (uint32_t)act_inLen, act_out,
                           (uint32_t)act_outLen, resume_at);
  t1 = hexkl_probe_now();
  hexkl_probe_on = 0;
  if (rc != AEE_SUCCESS) {
    FARF(ERROR, "forward_debug: start=%u pos=%u: %s (0x%08x)",
         (unsigned)start_op, (unsigned)pos, htp_graph_err_name(rc),
         (unsigned)rc);
    return rc;
  }
  for (i = 0; i < n_ops_limit; ++i) {
    op_pcycles[i] = (start_op + i < s->graph->n_ops)
                      ? (uint32)s->graph->op_pcycles[start_op + i]
                      : 0u;
  }
  graph_farf(s->graph, start_op, *resume_at);
  return nntr_hvx_moe_stage_fill(stage_us, t0, t1, AEE_SUCCESS);
}

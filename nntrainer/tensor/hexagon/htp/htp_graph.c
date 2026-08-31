// SPDX-License-Identifier: Apache-2.0
/**
 * @file	htp_graph.c
 * @date	19 August 2026
 * @brief	Op-list graph executor: validates once at init, then forward is
 *		a plain sequential dispatch loop over the op table (each op
 *		runs its own parallel section and returns at the barrier)
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	dlwlzzero <dlwlzzero@gmail.com>
 * @bug		No known bugs except for NYI items
 */
#include <HAP_compute_res.h>
#include <HAP_perf.h>
#include <stdlib.h>
#include <string.h>

#include "htp_graph.h"

const htp_op_fn htp_op_table[NNTR_HTP_OP_KIND_COUNT] = {
  hvx_op_embed, hvx_op_rmsnorm,       hvx_op_matmul_w8a8,
  hvx_op_rope,  hvx_op_attn,          hvx_op_silu_mul,
  hvx_op_add,   hvx_op_matmul_logits, hvx_op_matmul_w8a16,
};

#define HTP_GRAPH_VTCM_BYTES (4u * 1024u * 1024u)

int htp_graph_init(struct htp_graph *g, const uint8_t *oplist, uint32_t len,
                   uint8_t *weights, uint32_t wsize, uint8_t *kv,
                   uint32_t kvsize, uint8_t *act, uint32_t actsize) {
  uint32_t buf_size[NNTR_HTP_BUF_COUNT];
  uint32_t k_max = 0, i;
  int rc;

  if (!g)
    return 1;
  memset(g, 0, sizeof(*g));
  if (nntr_htp_oplist_check(oplist, len))
    return 1;
  memcpy(&g->cfg, oplist, sizeof(g->cfg));

  /* TOKENS/LOGITS sizes are the forward() contract: at most max_chunk
   * int32 token ids and exactly vocab fp32 logits. */
  buf_size[NNTR_HTP_BUF_WEIGHTS] = wsize;
  buf_size[NNTR_HTP_BUF_KV] = kvsize;
  buf_size[NNTR_HTP_BUF_ACT] = actsize;
  buf_size[NNTR_HTP_BUF_TOKENS] = g->cfg.max_chunk * 4u;
  buf_size[NNTR_HTP_BUF_LOGITS] = g->cfg.vocab * 4u;
  rc = nntr_htp_oplist_validate(oplist, len, buf_size);
  if (rc)
    return rc;

  g->ops =
    (const struct nntr_htp_op_desc *)(const void *)(oplist + sizeof(g->cfg));
  memcpy(g->ctx.buf_size, buf_size, sizeof(buf_size));
  g->ctx.buf[NNTR_HTP_BUF_WEIGHTS] = weights;
  g->ctx.buf[NNTR_HTP_BUF_KV] = kv;
  g->ctx.buf[NNTR_HTP_BUF_ACT] = act;
  g->ctx.cfg = &g->cfg;

  g->ctx.pool = wp_create(0);
  if (!g->ctx.pool)
    return 1;

  /* Quant scratch sized by the widest matmul k in this op-list. */
  for (i = 0; i < g->cfg.n_ops; ++i)
    if ((g->ops[i].kind == (uint32_t)NNTR_HTP_OP_MATMUL_W8A8 ||
         g->ops[i].kind == (uint32_t)NNTR_HTP_OP_MATMUL_LOGITS) &&
        g->ops[i].k > k_max)
      k_max = g->ops[i].k;
  if (k_max) {
    g->ctx.xq = memalign(128, (size_t)g->cfg.max_chunk * k_max);
    g->ctx.xq_scale = malloc((size_t)g->cfg.max_chunk * sizeof(float));
  }
  /* [n_workers][max_seq] fp32 scores, +128B pad: hvx_exp_f32's tail path
   * reads one whole unaligned vector starting at the last elements. */
  g->ctx.attn_scratch = memalign(
    128, (size_t)wp_size(g->ctx.pool) * g->cfg.max_seq * sizeof(float) + 128u);
  if ((k_max && (!g->ctx.xq || !g->ctx.xq_scale)) || !g->ctx.attn_scratch) {
    htp_graph_destroy(g);
    return 1;
  }

  /* Best-effort VTCM: NULL keeps the matmul DDR direct-read fallback and
   * is not an error. */
  {
    compute_res_attr_t rattr;
    unsigned id;
    HAP_compute_res_attr_init(&rattr);
    HAP_compute_res_attr_set_vtcm_param(&rattr, HTP_GRAPH_VTCM_BYTES, 1);
    id = HAP_compute_res_acquire(&rattr, 10000 /*us*/);
    if (id) {
      void *p = HAP_compute_res_attr_get_vtcm_ptr(&rattr);
      if (p) {
        g->vtcm_ctx_id = id;
        g->ctx.vtcm = (uint8_t *)p;
        g->ctx.vtcm_size = HTP_GRAPH_VTCM_BYTES;
      } else {
        HAP_compute_res_release(id);
      }
    }
  }
  return 0;
}

int htp_graph_forward_upto(struct htp_graph *g, const int32_t *tokens,
                           uint32_t n_tokens, uint32_t pos, float *logits,
                           uint32_t n_logits, uint32_t n_ops_limit,
                           uint64_t *pcycles) {
  uint32_t i;
  uint64_t t0;

  if (!g || !tokens || !logits)
    return 1;
  /* Runtime-argument gate: init cannot validate these, and a violation
   * reads past the rope table / writes past the KV cache. */
  if (n_tokens == 0u || n_tokens > g->cfg.max_chunk ||
      (uint64_t)pos + n_tokens > (uint64_t)g->cfg.max_seq)
    return 1;
  if (n_logits != g->cfg.vocab)
    return 1;
  if (n_ops_limit > g->cfg.n_ops)
    return 1;

  g->ctx.buf[NNTR_HTP_BUF_TOKENS] = (uint8_t *)(uintptr_t)tokens;
  g->ctx.buf[NNTR_HTP_BUF_LOGITS] = (uint8_t *)logits;
  g->ctx.n_tokens = n_tokens;
  g->ctx.pos = pos;

  t0 = HAP_perf_get_pcycles();
  for (i = 0; i < n_ops_limit; ++i)
    htp_op_table[g->ops[i].kind](&g->ctx, &g->ops[i]);
  if (pcycles)
    *pcycles = HAP_perf_get_pcycles() - t0;
  return 0;
}

int htp_graph_forward(struct htp_graph *g, const int32_t *tokens,
                      uint32_t n_tokens, uint32_t pos, float *logits,
                      uint32_t n_logits) {
  if (!g)
    return 1;
  return htp_graph_forward_upto(g, tokens, n_tokens, pos, logits, n_logits,
                                g->cfg.n_ops, 0);
}

const uint8_t *htp_graph_buf_ref(const struct htp_graph *g, uint32_t buf,
                                 uint32_t offset, uint32_t bytes) {
  if (!g || buf > (uint32_t)NNTR_HTP_BUF_ACT)
    return 0;
  if ((uint64_t)offset + bytes > (uint64_t)g->ctx.buf_size[buf])
    return 0;
  return g->ctx.buf[buf] + offset;
}

void htp_graph_destroy(struct htp_graph *g) {
  if (!g)
    return;
  if (g->vtcm_ctx_id)
    HAP_compute_res_release(g->vtcm_ctx_id);
  free(g->ctx.xq);
  free(g->ctx.xq_scale);
  free(g->ctx.attn_scratch);
  if (g->ctx.pool)
    wp_destroy(g->ctx.pool);
  memset(g, 0, sizeof(*g));
}

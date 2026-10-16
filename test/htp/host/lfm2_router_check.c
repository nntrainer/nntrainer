// SPDX-License-Identifier: Apache-2.0
/** @file lfm2_router_check.c
 * @brief Check LFM2 64-expert sigmoid routing and the 24B graph geometry.
 * @author NNTrainer contributors
 * @bug Device validation pending.
 */
#include "htp_graph_desc.h"
#include "hvx_m1_ops_f32.h"
#include "m1_ops_det.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void router(unsigned E, hvx_worker_pool *pool) {
  const unsigned K = 64, stride = (E + 31u) / 32u * 32u;
  float x[64], *w = calloc(K * stride, sizeof(float));
  float *plain = calloc(K * E, sizeof(float));
  float bias[64], got[64], expected[64], weight[64], ref_weight[64];
  unsigned sel[64], ref_sel[64];
  assert(w && plain);
  for (unsigned k = 0; k < K; ++k) {
    x[k] = ((int)(k % 5u) - 2) / 32.0f;
    for (unsigned e = 0; e < E; ++e) {
      plain[k * E + e] = w[k * stride + e] =
        ((int)((k * 7u + e * 13u) % 31u) - 15) / 128.0f;
    }
  }
  for (unsigned e = 0; e < E; ++e)
    bias[e] = e / 16.0f;
  m1_router_cpu_det(x, plain, bias, K, E, E, expected, ref_sel, ref_weight);
  hvx_router_topk_f32(x, w, bias, K, E, E, got, sel, weight, pool);
  assert(memcmp(got, expected, E * sizeof(float)) == 0);
  assert(memcmp(sel, ref_sel, E * sizeof(unsigned)) == 0);
  assert(memcmp(weight, ref_weight, E * sizeof(float)) == 0);
  // Independent full selection: no expert may be selected twice, including
  // the second 32-bit word. Zero logits and increasing bias fix exact order.
  memset(w, 0, K * stride * sizeof(float));
  hvx_router_topk_f32(x, w, bias, K, E, E, got, sel, weight, pool);
  for (unsigned r = 0; r < E; ++r)
    assert(sel[r] == E - 1u - r);
  memset(bias, 0, sizeof(bias));
  hvx_router_topk_f32(x, w, bias, K, E, E, got, sel, weight, pool);
  for (unsigned r = 0; r < E; ++r)
    assert(sel[r] == r);
  free(w);
  free(plain);
}

int main(void) {
  hvx_worker_pool *pool = hvx_worker_pool_create(3u);
  for (unsigned E = 4; E <= 64; E *= 2) {
    router(E, NULL);
    router(E, pool);
  }
  hvx_worker_pool_destroy(pool);
  htp_graph_lfm2_shape s = {40, 2, 2048, 11776, 1536, 64,   4,
                            32, 8, 64,   65536, 2048, 1e-5f};
  uint8_t attn[40];
  uint32_t *w = calloc(htp_graph_words_for(40, HTP_GRAPH_MAX_OPS), sizeof(*w));
  assert(w);
  for (unsigned l = 0; l < 40; ++l)
    attn[l] = l % 4 == 2;
  unsigned count =
    htp_graph_lfm2_build(w, htp_graph_words_for(40, HTP_GRAPH_MAX_OPS), &s,
                         attn, HTP_GRAPH_KINDS_ALL);
  assert(count);
  unsigned first_dense = 0, ops = 0;
  for (unsigned i = 0; i < w[3]; ++i)
    if (htp_graph_op_at(w, i)->kind == HTP_OP_DENSE_FFN) {
      htp_graph_op_at(w, i)->feed |= HTP_GRAPH_FEED_WH;
      first_dense = i;
    }
  assert(htp_graph_validate(w, count, HTP_GRAPH_KINDS_ALL, &ops) == 0);
  assert(ops == 380);
  htp_graph_op_at(w, first_dense)->feed = 0;
  assert(htp_graph_validate(w, count, HTP_GRAPH_KINDS_ALL, &ops) ==
         HTP_GRAPH_E_SCHEMENOTSUPPORTED);
  free(w);
  puts("LFM2 router 4..64 experts, cross-word ties, worker parity and 24B WH "
       "graph PASS");
}

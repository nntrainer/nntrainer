// SPDX-License-Identifier: Apache-2.0
/** @file prefill_scratch_host_check.c
 * @brief DSP scratch lifecycle against the real session type and methods.
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nntr_hvx_session.h"
#include <AEEStdErr.h>
#include <remote.h>

#define MOE_SCRATCH_ALIGN 128u
#include "prefill_scratch_methods.inc"

int main(void) {
  nntr_hvx_session *s = (nntr_hvx_session *)calloc(1, sizeof(*s));
  assert(s);
  const remote_handle64 handle = (remote_handle64)(uintptr_t)s;
  uint64 bytes = 99;
  assert(nntr_hvx_release_prefill_scratch(0, &bytes) == AEE_EBADPARM);
  assert(nntr_hvx_release_prefill_scratch(handle, NULL) == AEE_EBADPARM);
  assert(nntr_hvx_release_prefill_scratch(handle, &bytes) == AEE_SUCCESS);
  assert(bytes == 0);

  // Shared graph, weights, recurrent/KV state and decode's FC feed survive.
  unsigned char persistent[128] = {0};
  s->graph = (hexkl_graph *)persistent;
  s->attn_m1 = (hvx_attn_m1_ctx *)persistent;
  s->fc_l2 = persistent;
  s->q4m1[0].w = persistent;
  s->arenas[0].va = persistent;
  assert(hexkl_moe_scratch_reserve(&s->moe_scratch, 2u << 20) == AEE_SUCCESS);
  s->norm_rows_n = 65536;
  s->norm_rows = (float *)malloc(s->norm_rows_n * sizeof(float));
  s->moe_res_n = 131072;
  s->moe_res = (float *)malloc(s->moe_res_n * sizeof(float));
  assert(s->norm_rows && s->moe_res);
  const uint64 expected =
    (2u << 20) + 128u + (uint64)(s->norm_rows_n + s->moe_res_n) * sizeof(float);
  nntr_hvx_session saved = *s;

  // Borrowed per-call parameters identify a call that has not completed.
  s->moe_scratch.pre_gamma = (float *)persistent;
  assert(nntr_hvx_release_prefill_scratch(handle, &bytes) == AEE_EBADSTATE);
  assert(bytes == 0 && s->moe_scratch.raw == saved.moe_scratch.raw);
  s->moe_scratch.pre_gamma = NULL;
  s->moe_scratch.out_heap = 1;
  assert(nntr_hvx_release_prefill_scratch(handle, &bytes) == AEE_EBADSTATE);
  assert(bytes == 0 && s->norm_rows == saved.norm_rows);
  s->moe_scratch.out_heap = 0;

  assert(nntr_hvx_release_prefill_scratch(handle, &bytes) == AEE_SUCCESS);
  assert(bytes == expected);
  saved.moe_scratch.raw = NULL;
  saved.moe_scratch.base = NULL;
  saved.moe_scratch.cap = 0;
  saved.norm_rows = NULL;
  saved.norm_rows_n = 0;
  saved.moe_res = NULL;
  saved.moe_res_n = 0;
  assert(memcmp(s, &saved, sizeof(*s)) == 0);
  assert(nntr_hvx_release_prefill_scratch(handle, &bytes) == AEE_SUCCESS);
  assert(bytes == 0);

  // Decode-sized buffers stay allocated, and another prefill grows from
  // them normally. Reclaiming the second prompt must not retain stale sizes.
  assert(hexkl_moe_scratch_reserve(&s->moe_scratch, 65536) == AEE_SUCCESS);
  s->norm_rows_n = 16384;
  s->norm_rows = (float *)malloc(s->norm_rows_n * sizeof(float));
  s->moe_res_n = 1024;
  s->moe_res = (float *)malloc(s->moe_res_n * sizeof(float));
  assert(s->norm_rows && s->moe_res);
  saved = *s;
  assert(nntr_hvx_release_prefill_scratch(handle, &bytes) == AEE_SUCCESS);
  assert(bytes == 0 && memcmp(s, &saved, sizeof(*s)) == 0);
  assert(hexkl_moe_scratch_reserve(&s->moe_scratch, 4u << 20) == AEE_SUCCESS);
  memset(s->moe_scratch.base, 0x5a, s->moe_scratch.cap);
  assert(nntr_hvx_release_prefill_scratch(handle, &bytes) == AEE_SUCCESS);
  assert(bytes == (4u << 20) + 128u);
  assert(!s->moe_scratch.raw && !s->moe_scratch.base && !s->moe_scratch.cap);
  assert(s->norm_rows == saved.norm_rows && s->moe_res == saved.moe_res);
  free(s->norm_rows);
  free(s->moe_res);
  free(s);
  puts("PASS: DSP scratch release, persistent state, active consumers, "
       "repeated prefill");
  return 0;
}

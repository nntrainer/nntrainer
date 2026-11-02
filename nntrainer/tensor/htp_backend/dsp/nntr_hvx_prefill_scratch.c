// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file nntr_hvx_prefill_scratch.c
 * @brief Release completed prefill workspace while retaining decode state.
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"
#include <AEEStdErr.h>
#include <stdlib.h>

/** Release only ephemeral workspace, between completed synchronous calls.
 * The host holds invoke_mutex_ across this RPC and each queue response; an
 * installed token driver may remain idle across independent prompts. */
int nntr_hvx_release_prefill_scratch(remote_handle64 handle, uint64 *bytes) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  const size_t keep_bytes = 64u << 10;
  if (!s || !bytes)
    return AEE_EBADPARM;
  *bytes = 0;
  if (s->moe_scratch.pre_gamma || s->moe_scratch.out_heap ||
      s->moe_scratch.rotated_input || s->moe_scratch.router_act)
    return AEE_EBADSTATE;
  if (s->moe_scratch.cap > keep_bytes) {
    *bytes += s->moe_scratch.cap + 128u;
    hexkl_moe_scratch_free(&s->moe_scratch);
  }
  if ((size_t)s->norm_rows_n * sizeof(float) > keep_bytes) {
    *bytes += (uint64)s->norm_rows_n * sizeof(float);
    free(s->norm_rows);
    s->norm_rows = NULL;
    s->norm_rows_n = 0;
  }
  if ((size_t)s->moe_res_n * sizeof(float) > keep_bytes) {
    *bytes += (uint64)s->moe_res_n * sizeof(float);
    free(s->moe_res);
    s->moe_res = NULL;
    s->moe_res_n = 0;
  }
  return AEE_SUCCESS;
}

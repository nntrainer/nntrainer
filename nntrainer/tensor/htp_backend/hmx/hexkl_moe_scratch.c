// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file hexkl_moe_scratch.c
 * @brief Allocation and invalidation of reusable MoE workspace.
 * @author NNTrainer contributors
 * @bug No known bugs.
 */
#include "hexkl_mm_u8i4_moe.h"
#include <AEEStdErr.h>
#include <stdlib.h>

#define MOE_SCRATCH_ALIGN 128u

void hexkl_moe_scratch_free(hexkl_moe_scratch *s) {
  if (s) {
    htp_router_handoff_scratch_invalidate(&s->router_handoff);
    s->router_act = NULL;
    s->router_used = 0;
    free(s->raw);
    s->raw = NULL;
    s->base = NULL;
    s->cap = 0;
  }
}

/** @brief Grows the scratch to @a bytes; a no-op once it is big enough,
 *         which after the first prefill call is every call. */
int hexkl_moe_scratch_reserve(hexkl_moe_scratch *s, size_t bytes) {
  htp_router_handoff_scratch_invalidate(&s->router_handoff);
  s->router_act = NULL;
  s->router_used = 0;
  if (s->cap >= bytes) {
    return AEE_SUCCESS;
  }
  hexkl_moe_scratch_free(s);
  s->raw = malloc(bytes + MOE_SCRATCH_ALIGN);
  if (!s->raw) {
    return AEE_ENOMEMORY;
  }
  s->base = (uint8_t *)(((uintptr_t)s->raw + (MOE_SCRATCH_ALIGN - 1u)) &
                        ~(uintptr_t)(MOE_SCRATCH_ALIGN - 1u));
  s->cap = bytes;
  return AEE_SUCCESS;
}

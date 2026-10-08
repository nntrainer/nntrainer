// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_expand_i2i4.c
 * @date   23 Sep 2026
 * @brief  2-bit expert codes -> int4 WH bytes, in VTCM, before the HMX
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The why, the budget and the unverified assumption are in the header.
 */

#include "hvx_expand_i2i4.h"

/* The header only forward-declares the pool type so a C++ check can include
   it; the pool call itself needs the real declaration. */
#include "hvx_worker_pool.h"

#include <stddef.h>

#if defined(__hexagon__)
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

/** @brief Source bytes one HVX vector covers. */
#define EXPAND_VEC 128u

void hvx_expand_i2i4_table(const int8_t *pal, uint8_t *table) {
  for (uint32_t i = 0; i < HVX_EXPAND_TABLE_BYTES; ++i) {
    table[i] = 0u;
  }
  for (uint32_t hi = 0; hi < 4u; ++hi) {
    for (uint32_t lo = 0; lo < 4u; ++lo) {
      /* At 2 * index, not at index. V6_vlutvvb fetches entry 2*i for index
         i -- measured, EXPAND_I2I4_MAP (doc 54 5.8), after the obvious
         reading turned out to be wrong. Sixteen indices land on the even
         bytes 0..30 of a 32-byte block, one each, so undoing it is a
         permutation of where the entries are written and costs nothing at
         run time: the loop is still three ops an output vector. */
      table[2u * ((hi << 2) | lo)] =
        (uint8_t)((pal[lo] & 0x0F) | ((pal[hi] & 0x0F) << 4));
    }
  }
  /* The 32-byte block repeated, so the lookup is right whether the hardware
     takes its table from the low bytes of the vector or from each 32-byte
     lane group -- the other half of the guess, which the map probe showed
     does not matter either way. The buffer is scratch: it is loaded into a
     vector register once and never read from memory again, so only the four
     palette bytes have to be kept per weight. */
  for (uint32_t i = 32u; i < HVX_EXPAND_TABLE_BYTES; ++i) {
    table[i] = table[i & 31u];
  }
}

/** @brief The index scaling V6_vlutvvb applies, so the scalar twin reads
 *         the same table the hardware does rather than a second one. */
#define EXPAND_LUT(table, idx) ((table)[2u * (uint32_t)(idx)])

void hvx_expand_i2i4_scalar(const uint8_t *src, uint32_t src_bytes,
                            const uint8_t *table, uint8_t *dst) {
  /* A group of EXPAND_VEC codes becomes two contiguous runs of int4: the
     low halves first, then the high halves. That is the whole reason the
     HVX form needs no shuffle -- and why whPack2 stores the codes the way
     whCodeByte2 says. */
  uint32_t g;
  for (g = 0; g + EXPAND_VEC <= src_bytes; g += EXPAND_VEC) {
    uint32_t k;
    for (k = 0; k < EXPAND_VEC; ++k) {
      const uint8_t s = src[g + k];
      dst[2u * g + k] = EXPAND_LUT(table, s & 0x0Fu);
      dst[2u * g + EXPAND_VEC + k] = EXPAND_LUT(table, (s >> 4) & 0x0Fu);
    }
  }
  /* A tail shorter than a group keeps the same shape over what is left. */
  if (g < src_bytes) {
    const uint32_t n = src_bytes - g;
    uint32_t k;
    for (k = 0; k < n; ++k) {
      const uint8_t s = src[g + k];
      dst[2u * g + k] = EXPAND_LUT(table, s & 0x0Fu);
      dst[2u * g + n + k] = EXPAND_LUT(table, (s >> 4) & 0x0Fu);
    }
  }
}

void hvx_expand_i2i4(const uint8_t *src, uint32_t src_bytes,
                     const uint8_t *table, uint8_t *dst) {
#if defined(__hexagon__)
  const uint32_t vecs = src_bytes / EXPAND_VEC;
  /* Both of these are loop-invariant and stay in vector registers for the
     whole call -- vlut32 takes its table as a register operand, so the
     lookup is not a memory access. That is the entire reason this costs
     three ops an output vector: one 128-byte load a call, not a gather a
     byte. Nine vectors are live in the loop against HVX's 32. */
  const HVX_Vector tab = *(const HVX_UVector *)table;
  const HVX_Vector m0f = Q6_V_vsplat_R((int)0x0F0F0F0Fu);

  for (uint32_t v = 0; v < vecs; ++v) {
    const HVX_Vector s = *(const HVX_UVector *)(src + (size_t)v * EXPAND_VEC);
    /* The two nibble planes are the two halves of every source byte, and
       each is already a complete table index -- no spreading, because
       whPack2 put the code pairs where the output byte pairs are. */
    const HVX_Vector lo = Q6_V_vand_VV(s, m0f);
    const HVX_Vector hi = Q6_V_vand_VV(Q6_Vuh_vlsr_VuhR(s, 4), m0f);
    /* Straight out, no interleave: whPack2 already put the codes in the
       order that makes each lookup a whole output vector (whCodeByte2).
       The shuffle this replaces was 26% of the time (doc 54 5.13). */
    HVX_UVector *d = (HVX_UVector *)(dst + (size_t)v * 2u * EXPAND_VEC);
    d[0] = Q6_Vb_vlut32_VbVbI(lo, tab, 0);
    d[1] = Q6_Vb_vlut32_VbVbI(hi, tab, 0);
  }
  /* src_bytes is a multiple of 128 for every shape this is called with (a
     WH tile is 256 code bytes); a tail would be a caller bug, so it is
     finished rather than ignored. */
  if (vecs * EXPAND_VEC != src_bytes) {
    const uint32_t done = vecs * EXPAND_VEC;
    hvx_expand_i2i4_scalar(src + done, src_bytes - done, table,
                           dst + 2u * done);
  }
#else
  hvx_expand_i2i4_scalar(src, src_bytes, table, dst);
#endif
}

typedef struct {
  const uint8_t *src;
  const uint8_t *table;
  uint8_t *dst;
  uint32_t src_bytes;
} expand_ctx;

static void expand_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  const expand_ctx *c = (const expand_ctx *)vctx;
  /* Whole vectors each, so no two workers produce bytes of the same
     shuffle. The last worker takes any remainder. */
  const uint32_t vecs = c->src_bytes / EXPAND_VEC;
  const uint32_t v0 = vecs * i / n_threads;
  const uint32_t v1 = vecs * (i + 1u) / n_threads;
  const uint32_t off = v0 * EXPAND_VEC;
  /* The last worker absorbs any tail past the whole vectors. */
  const uint32_t bytes =
    (i + 1u == n_threads) ? (c->src_bytes - off) : ((v1 - v0) * EXPAND_VEC);
  if (bytes) {
    hvx_expand_i2i4(c->src + off, bytes, c->table, c->dst + 2u * (size_t)off);
  }
}

typedef struct {
  uint8_t *base;
  const uint8_t *table;
  uint32_t stride;
  uint32_t src_bytes;
  uint32_t nrows;
} expand_rows_ctx;

static void expand_rows_worker(uint32_t n_threads, uint32_t i, void *vctx) {
  const expand_rows_ctx *c = (const expand_rows_ctx *)vctx;
  const uint32_t r0 = c->nrows * i / n_threads;
  const uint32_t r1 = c->nrows * (i + 1u) / n_threads;
  uint32_t r;
  for (r = r0; r < r1; ++r) {
    uint8_t *dst = c->base + (size_t)r * c->stride;
    hvx_expand_i2i4(dst + c->src_bytes, c->src_bytes, c->table, dst);
  }
}

void hvx_expand_i2i4_rows(uint8_t *base, uint32_t stride, uint32_t src_bytes,
                          uint32_t nrows, const uint8_t *table,
                          hvx_worker_pool *pool) {
  expand_rows_ctx ctx;
  ctx.base = base;
  ctx.table = table;
  ctx.stride = stride;
  ctx.src_bytes = src_bytes;
  ctx.nrows = nrows;
  if (!pool) {
    expand_rows_worker(1u, 0u, &ctx);
    return;
  }
  hvx_worker_pool_run(pool, expand_rows_worker, &ctx, nrows);
}

void hvx_expand_i2i4_pool(const uint8_t *src, uint32_t src_bytes,
                          const uint8_t *table, uint8_t *dst,
                          hvx_worker_pool *pool) {
  if (!pool) {
    hvx_expand_i2i4(src, src_bytes, table, dst);
    return;
  }
  expand_ctx ctx = {src, table, dst, src_bytes};
  /* One unit a vector, so the pool caps at its own width and a transfer
     too small to split runs inline. */
  hvx_worker_pool_run(pool, expand_worker, &ctx, src_bytes / EXPAND_VEC);
}

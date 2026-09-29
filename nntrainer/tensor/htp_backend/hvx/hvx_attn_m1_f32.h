// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_attn_m1_f32.h
 * @date   27 Sep 2026
 * @brief  Decode attention at m=1 on HVX with a DSP-resident f32 KV cache,
 *         bit-identical to nntrainer/tensor/attn_m1_det.h
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * The specification, its operation order and its domain live in
 * attn_m1_det.h; this file implements it in plain Vsf (no qf32, no sf FMA --
 * HVX has none) over a cache object the session owns (plan 81 section 3.1):
 *
 *   Kt [n_layers][n_kv][head_dim][max_seq]   one 128-B vector = 32 positions
 *   V  [n_layers][n_kv][max_seq][head_dim]   one position = head_dim/32 vectors
 *
 * both f32 on the DSP heap, 128-byte aligned, zero-filled at create so the
 * lanes past the context in the last block read finite values (they are
 * masked out of every reduction). Per kv head, one HVX_Vector accumulator
 * per q head of its GQA group shares the Kt stream, then the V stream --
 * the fusion changes no arithmetic, each q head's sequence is the spec's.
 *
 * ADDRESS BUDGET (doc 46 section 41: 3840 MiB arena + about 182 MiB heap).
 * The cache is n_layers * n_kv * head_dim * max_seq * 2 * 4 bytes: at the
 * LFM2.5 shape (6 attention layers, 8 kv heads, head_dim 64) that is 24 MiB
 * per 1024 of max_seq, so 48 MiB at nntr_config.json's max_seq_len 2048 --
 * 26 % of the heap, with no VTCM, no mapping and no DMA. The probability
 * scratch is n_kv * gqa * max_seq * 4 = 256 KiB at 2048. Growth policy:
 * none; the Kt stride is max_seq, so the size is fixed at create. fp16 K/V
 * would halve it and is the upgrade path once an sf -> hf rounding spec is
 * device-confirmed (plan 81 section 3.1).
 *
 * THREADS. forward runs one unit per (kv head, pair of its q heads) on
 * hvx_worker_pool_run when gqa is even -- 16 units at LFM2.5's shape, so
 * the busiest of 6 lanes runs 1.5 kv heads' work instead of 2 (#146) --
 * and one unit per kv head when gqa is odd. Lane i takes units i, i + n,
 * ...; each unit writes its q heads' probabilities into their own scratch
 * rows. The split is deterministic by construction: no reduction crosses
 * a q head, so the output is byte-equal at any worker count -- which the
 * host check proves at 0, 3 and 7 workers.
 *
 * ALIGNMENT. q, k, v, out and stats are the caller's FastRPC buffers and
 * carry no vector alignment: every access to them is HVX_UVector or
 * scalar. The cache and the scratch are memalign(128) and are read and
 * written with aligned vectors; every such pointer is a multiple of 32
 * floats from a 128-byte base (max_seq and head_dim are multiples of 32).
 *
 * ERRORS are AEEStdErr codes so the skel entries pass them through:
 * AEE_EINVALIDFORMAT for a shape or position out of range, AEE_EBADSTATE
 * for a hole (pos past the layer's length) or a missing cache,
 * AEE_ENOMEMORY when the heap refuses the cache. Never AEE_EBADPARM, which
 * stays the stale-skel symptom (rule 3).
 *
 * test/htp/host/attn_m1_host_check.c compiles THIS source against the
 * hvx_emu/ intrinsic emulation and the real worker pool on pthreads and
 * memcmp's it with the spec; unittest_hvx_attn.cpp's HvxAttnM1.* repeats
 * that on the device.
 */

#ifndef __NNTRAINER_HVX_ATTN_M1_F32_H__
#define __NNTRAINER_HVX_ATTN_M1_F32_H__

#include <stddef.h>
#include <stdint.h>

#include "hvx_worker_pool.h"

/**
 * @brief The cache object. Read-only for callers (the host check compares
 *        two caches byte for byte through kt / v / cache_floats); create,
 *        kv_append and forward are the writers.
 */
typedef struct {
  uint32_t n_layers;
  uint32_t n_kv;
  uint32_t gqa;
  uint32_t head_dim;   /**< a multiple of 32 */
  uint32_t max_seq;    /**< a multiple of 32; the Kt stride */
  uint32_t *kv_len;    /**< [n_layers] positions held, 0..max_seq */
  float *kt;           /**< [n_layers][n_kv][head_dim][max_seq], memalign 128 */
  float *v;            /**< [n_layers][n_kv][max_seq][head_dim], memalign 128 */
  float *scratch;      /**< [n_kv * gqa][max_seq] per q head, memalign 128 */
  size_t cache_floats; /**< floats in kt, and in v */
  hvx_worker_pool *pool; /**< borrowed; NULL runs every head on the caller */
} hvx_attn_m1_ctx;

/**
 * @brief Allocates and zero-fills the cache for a fixed shape.
 *
 * @param n_layers  attention layers (the layer ordinal space), >= 1
 * @param n_kv      kv heads, >= 1
 * @param gqa       q heads per kv head, 1..8 (the register file holds gqa
 *                  score accumulators plus gqa * head_dim/32 PV ones)
 * @param head_dim  a multiple of 32, at most 128
 * @param max_seq   a multiple of 32; the position bound
 * @param pool      the session's worker pool, borrowed; may be NULL
 * @param err       receives AEE_SUCCESS, AEE_EINVALIDFORMAT (shape, or a
 *                  cache past 2 GiB) or AEE_ENOMEMORY; may be NULL. Every
 *                  shape rule is checked here, so forward never rejects a
 *                  shape that registered
 * @return the cache, or NULL with *err set
 */
hvx_attn_m1_ctx *hvx_attn_m1_create(uint32_t n_layers, uint32_t n_kv,
                                    uint32_t gqa, uint32_t head_dim,
                                    uint32_t max_seq, hvx_worker_pool *pool,
                                    int *err);

/** @brief Frees the cache. Safe on NULL. Does not touch the pool. */
void hvx_attn_m1_free(hvx_attn_m1_ctx *ctx);

/**
 * @brief Writes n_rows positions [kv_from, kv_from + n_rows) of one layer
 *        from the CPU cache layout and sets the layer's length to their end.
 *
 * @param k_rows, v_rows  [n_rows][n_kv][head_dim] f32
 * @return AEE_SUCCESS; AEE_EINVALIDFORMAT if layer or the range is out of
 *         bounds; AEE_EBADSTATE if kv_from is past the layer's length (a
 *         hole) or ctx is NULL
 */
int hvx_attn_m1_kv_append(hvx_attn_m1_ctx *ctx, uint32_t layer,
                          uint32_t kv_from, uint32_t n_rows,
                          const float *k_rows, const float *v_rows);

/**
 * @brief Appends position @a pos (k, v of the new token) to @a layer and
 *        computes the attention output over positions 0..pos.
 *
 * pos <= kv_len[layer] is required: pos == kv_len appends, pos < kv_len
 * rewinds (the CPU's cache_index reset), pos > kv_len is a hole.
 *
 * @param q      [n_kv * gqa][head_dim], post-RoPE
 * @param k, v   [n_kv][head_dim], post-RoPE k
 * @param out    [n_kv * gqa][head_dim]
 * @param stats  2 * n_kv * gqa floats, (m, l) per q head, or NULL
 * @return AEE_SUCCESS; AEE_EINVALIDFORMAT if layer or pos is out of range;
 *         AEE_EBADSTATE for a hole or a NULL ctx
 */
int hvx_attn_m1_forward(hvx_attn_m1_ctx *ctx, uint32_t layer, uint32_t pos,
                        float scale, const float *q, const float *k,
                        const float *v, float *out, float *stats);

/**
 * @brief hvx_attn_m1_forward that also fills the phase words (#146).
 *
 * @param prof  ATTN_M1_PROF_WORDS uint32 words (attn_m1_det.h's
 *              ATTN_M1_PROF_* indices), or NULL -- then this is exactly
 *              hvx_attn_m1_forward and takes no timestamp. Written only on
 *              AEE_SUCCESS. The words are a measurement channel: the output
 *              and stats are byte-equal with and without them
 */
int hvx_attn_m1_forward_prof(hvx_attn_m1_ctx *ctx, uint32_t layer, uint32_t pos,
                             float scale, const float *q, const float *k,
                             const float *v, float *out, float *stats,
                             uint32_t *prof);

#endif /* __NNTRAINER_HVX_ATTN_M1_F32_H__ */

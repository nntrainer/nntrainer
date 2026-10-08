// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_attn_m1_f32.h
 * @date   27 Sep 2026
 * @brief  Decode attention at m=1 on HVX with a DSP-resident fp16 KV cache,
 *         bit-identical to nntrainer/tensor/attn_m1_det.h (the Android fp16
 *         CPU attention)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * The specification, its operation order and its domain live in
 * attn_m1_det.h; this file's kernel (hvx_attn_m1_f32.c; the name predates
 * the fp16 cache) runs it in fp16 lanes with hvx_attn_m1_hf.h's primitives
 * (plan 170) over a cache object the session owns:
 *
 *   Kt [n_layers][n_kv][seq / 64][head_dim][64]  fp16, one 8 KiB tile per
 *                                                64 positions (a vector per d)
 *   V  [n_layers][n_kv][seq][head_dim]           fp16, a vector per position
 *
 * seq = max_seq rounded up to 64. Every row is rounded to fp16 on append
 * (the CPU's seed rows already are), 128-byte aligned, zero-filled at
 * create so the lanes past the context in the last tile read finite values
 * (they are masked out of every reduction). head_dim is 64 (LFM2) or,
 * since plan 201 S4, any multiple of 64 up to 512 (Gemma 4's 256 and 512:
 * a Kt tile is then head_dim vectors, a V row head_dim / 64), n_kv * gqa <=
 * 64 (the sum vector's lanes). A sliding window (Gemma's sliding layers)
 * is a forward argument: the cache keeps every position, the call reads
 * the last min(L, window).
 *
 * ADDRESS BUDGET (doc 46 section 41: 3840 MiB arena + about 182 MiB heap).
 * The cache is n_layers * n_kv * head_dim * seq * 2 * 2 bytes: at the
 * LFM2.5 shape (6 attention layers, 8 kv heads, head_dim 64) 12 MiB per 1024
 * of max_seq, so 24 MiB at nntr_config.json's max_seq_len 2048 (48 MiB
 * with the f32 cache before #170). Scratch per cache: the scores /
 * probabilities n_q * seq * 2 (128 KiB), the transposed exps seq * 128
 * (256 KiB), the q rows n_q * 128 (4 KiB; round 2's splats were
 * n_q * head_dim * 128 = 256 KiB), the k
 * rows n_kv * head_dim * 2 (1 KiB) and the exp16 table ATTN_M1_DET_EXP_N *
 * 2 (37 KiB, #170 round 3) at LFM2.5 and 2048: about 426 KiB of the ~182
 * MiB heap (215 KiB less than round 2). No VTCM, no mapping, no DMA (the
 * l2fetch leads are hints). Growth policy: none; the size is fixed at
 * create. [plan 201 S4] At Gemma 4's shape the cache does not fit that
 * heap: 25 sliding layers x 8 kv heads x 256 are 200 MiB per 1024 of
 * max_seq (K and V), the 5 full layers x 2 x 512 another 20 MiB.
 * ponytail: a ring of the window's 1024 positions would cap the sliding
 * layers at 200 MiB whatever max_seq, and the cache must move into the
 * session's arena either way (S5's load hand-over); this file only
 * computes over it.
 *
 * THREADS. forward runs three pool runs (hvx_attn_m1_f32.c's header):
 * scores and exp by (kv head, 64-position tile), PV by a range of q-head
 * chains per lane (#170 round 2: 1.125x the mean lane at 6 lanes, from 1.5x
 * with (kv head, 4 heads) units), with the max and the sequential sum on
 * the caller between them. No reduction crosses a unit or a chain except
 * those two serial steps, so the output is byte-equal at any worker count
 * -- which the host check proves at 0, 3 and 7 workers.
 *
 * ALIGNMENT. q, k, v, out and stats are the caller's FastRPC buffers and
 * carry no vector alignment beyond a float's: they are read and written
 * with unaligned vectors (HVX_UVector). The cache and the scratch are
 * memalign(128) and are read and written with aligned vectors.
 *
 * ERRORS are AEEStdErr codes so the skel entries pass them through:
 * AEE_EINVALIDFORMAT for a shape, position or scale out of range (scale
 * must be an fp16 value), AEE_EBADSTATE for a hole (pos past the layer's
 * length) or a missing cache, AEE_ENOMEMORY when the heap refuses the
 * cache. Never AEE_EBADPARM, which stays the stale-skel symptom (rule 3).
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
 *        two caches byte for byte through kt / v / cache_halves); create,
 *        kv_append and forward are the writers.
 */
typedef struct {
  uint32_t n_layers;
  uint32_t n_kv;
  uint32_t gqa;
  uint32_t head_dim;   /**< 64, or a multiple of 64 up to 512 */
  uint32_t max_seq;    /**< a multiple of 32; the position bound */
  uint32_t seq;        /**< max_seq rounded up to 64: the tile count * 64 */
  uint32_t *kv_len;    /**< [n_layers] positions held, 0..max_seq */
  uint16_t *kt;        /**< fp16 [n_layers][n_kv][seq/64][head_dim][64] */
  uint16_t *v;         /**< fp16 [n_layers][n_kv][seq][head_dim] */
  uint16_t *s;         /**< fp16 [n_kv * gqa][seq]: scores, then probs */
  uint16_t *et;        /**< fp16 [seq][64]: the exps, q heads in lanes */
  uint16_t *qs;        /**< fp16 [n_kv * gqa][head_dim / 64][64]: q, zipped for
                            vlut16 per 64-wide chunk */
  uint16_t *kr;        /**< fp16 [n_kv][head_dim]: the new k rows */
  uint16_t *exp_tab;   /**< fp16 [ATTN_M1_DET_EXP_N]: exp16 by index */
  size_t cache_halves; /**< fp16 values in kt, and in v */
  hvx_worker_pool *pool; /**< borrowed; NULL runs every unit on the caller */
} hvx_attn_m1_ctx;

/**
 * @brief Allocates and zero-fills the cache for a fixed shape.
 *
 * @param n_layers  attention layers (the layer ordinal space), >= 1
 * @param n_kv      kv heads, >= 1
 * @param gqa       q heads per kv head, 1..8, with n_kv * gqa <= 64
 * @param head_dim  a multiple of 64 up to ATTN_M1_DET_MAX_HD (512)
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
 * @param window [plan 201 S4] attend to the last min(pos + 1, window)
 *               positions (attn_m1_det_lo); 0 = every position
 * @param scale  the score scale, an fp16 value (0.125 at LFM2's head_dim
 *               64; 1.0 for Gemma 4, whose q is not pre-scaled here)
 * @param q      [n_kv * gqa][head_dim], post-RoPE
 * @param k, v   [n_kv][head_dim], post-RoPE k
 * @param out    [n_kv * gqa][head_dim]
 * @param stats  2 * n_kv * gqa floats, (m, l) per q head, or NULL
 * @return AEE_SUCCESS; AEE_EINVALIDFORMAT if layer or pos is out of range
 *         or scale is not an fp16 value; AEE_EBADSTATE for a hole or a NULL
 *         ctx
 */
int hvx_attn_m1_forward(hvx_attn_m1_ctx *ctx, uint32_t layer, uint32_t pos,
                        uint32_t window, float scale, const float *q,
                        const float *k, const float *v, float *out,
                        float *stats);

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
                             uint32_t window, float scale, const float *q,
                             const float *k, const float *v, float *out,
                             float *stats, uint32_t *prof);

#endif /* __NNTRAINER_HVX_ATTN_M1_F32_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hexkl_mm_u8i4_dma.h
 * @date   06 Aug 2026
 * @brief  Persistent u8i4 weight registry and the cross-matmul DMA layer path
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Turns hexkl_mm_u8i4's accuracy-harness building blocks (plan / bake /
 * run, one matmul at a time, weight baked fresh every call) into the
 * performance path doc15 §4/§8 item 3 describes: a weight is baked once
 * and kept DSP-resident until released, and a "layer" of matmuls that share
 * one activation (Q/K/V, or gate/up) runs back to back with the next
 * weight prefetched into VTCM while the current one computes -- the
 * cross-matmul win measured in doc13 §3a (1.7-2x over SDKL on V79).
 */

#ifndef __NNTRAINER_HEXKL_MM_U8I4_DMA_H__
#define __NNTRAINER_HEXKL_MM_U8I4_DMA_H__

#include <stdint.h>

#include "hexkl_mm_opts.h"

/** @brief Upper bound on weights resident at once. A 28-layer qwen3-sized
 *         model registers 7 per layer (q,k,v,o,gate,up,down) = 196.
 *         LFM2.5-8B-A1B resident in full (doc 45) is 22 MoE layers x 64 +
 *         2 dense x 2 + 18 conv x 2 + 6 attention x 4 = 1472, so 512 --
 *         which bounded the MoE work to 8 layers at a time -- is not
 *         enough for the whole model. The table is ~48 bytes per slot;
 *         2048 costs ~100 KB of static DSP memory. */
#define HEXKL_MM_U8I4_MAX_WEIGHTS 2048

/**
 * @brief One registered weight: WH-baked bytes plus its dequant constants,
 *        resident in DSP heap memory (not VTCM -- VTCM is compute
 *        scratch) until hexkl_weight_u8i4_release.
 */
typedef struct {
  int in_use;
  uint8_t *wh_bytes; /**< k_tiles*n_tiles*512 bytes, WH layout */
  float *w_scale;    /**< N entries */
  int32_t *colsum_w; /**< N entries */
  float *bias;       /**< N entries */
  uint32_t K, N;
} hexkl_weight_u8i4;

typedef struct {
  hexkl_weight_u8i4 slots[HEXKL_MM_U8I4_MAX_WEIGHTS];
} hexkl_weight_u8i4_table;

/**
 * @brief Bakes a K x N int4 weight (int4 values in int8 containers,
 *        row-major) to WH layout and keeps the result resident until
 *        hexkl_weight_u8i4_release.
 *
 * Borrows the caller's VTCM arena as scratch for the bake -- the caller
 * must hold the HMX lock, and no other VTCM use may be in flight for the
 * duration of this call.
 *
 * The bake splits across @a pool when one is given (the session's quant
 * pool; every tile reads and writes disjoint memory, so the finished bytes
 * equal the serial bake's -- device-verified bit-identical, doc 43 §7).
 * NULL bakes serially on the calling thread, same bytes.
 *
 * @param[out] out_handle  index into @a tbl; pass back to
 *                         hexkl_mm_u8i4_layer_run / hexkl_weight_u8i4_release
 * @return AEE_SUCCESS, AEE_EBADPARM on a shape violation, AEE_ENOMEMORY if
 *         the table is full or a host allocation fails, or a HexKL error
 *         code from the bake itself.
 */
int hexkl_weight_u8i4_register(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                               uint32_t vtcm_size, uint32_t K, uint32_t N,
                               const int8_t *w_i4_rm, const float *w_scale,
                               const int32_t *colsum_w, const float *bias,
                               hvx_worker_pool *pool, uint32_t *out_handle);

/** @brief Frees a registered weight's resident bytes. */
int hexkl_weight_u8i4_release(hexkl_weight_u8i4_table *tbl, uint32_t handle);

/**
 * @brief Runs matmuls against several registered weights that share one
 *        quantized activation, double-buffering each weight into VTCM and
 *        prefetching the next handle's weight while the current one
 *        computes.
 *
 * Every handle must have been registered with K == @a K; this is checked,
 * not assumed. Output is dequantized f32, one contiguous M x handle[i].N
 * block per handle in call order (not interleaved row-by-row).
 *
 * @param config_off  session-constant HMX config region offset (from
 *                     hexkl_mm_u8i4_plan; the same for every M/K/N because
 *                     it depends only on vtcm_size). The caller must have
 *                     called hexkl_micro_hmx_setup_acc_read_int32 for it
 *                     once already -- this function does not repeat that
 *                     because it is session-scoped, not call-scoped.
 * @param[out] out_cat  sum(handles[i].N) * M f32 entries
 * @param opts  NULL for defaults; see hexkl_mm_opts.h
 */
int hexkl_mm_u8i4_layer_run(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                            uint32_t vtcm_size, uint32_t config_off, uint32_t M,
                            uint32_t K, const uint32_t *handles,
                            uint32_t n_handles, const float *act_f32,
                            float *out_cat, const hexkl_mm_opts *opts);

/**
 * @brief Runs ONE MoE expert FFN in a single call: gate_up matmul -> SwiGLU
 *        -> down matmul, with the SwiGLU intermediate never leaving the DSP.
 *
 * @a handles must be exactly two registered weights, [gate_up, down], with
 * gate_up baked at K x 2I and down at I x N_out -- down.K == gate_up.N / 2
 * is checked, not assumed (that ratio IS the SwiGLU contract: gate_up's
 * output row is [gate | up], silu(gate)*up feeds the down projection).
 * act_f32 is M x K f32; out is M x N_out f32, written (or accumulated into,
 * per opts->accumulate) directly -- there is no out_cat scatter, the fused
 * call has exactly one output matrix.
 *
 * The pipeline: quantize act into VTCM AH tiles, block on gate_up's weight,
 * stream down's weight in behind it (the cross-matmul prefetch pattern),
 * run gate_up's matmuls and dequantize each tile straight into two VTCM
 * regions (gate and up), SwiGLU them in place, requantize the gate half
 * into AH tiles, run down's matmuls, dequantize to out. What this deletes
 * versus two hexkl_mm_u8i4_layer_run calls is the M x 2I and M x I f32
 * intermediates crossing FastRPC twice, plus one round trip.
 *
 * opts: only pool and accumulate are honored. act_scale/act_zp/
 * act_ah_prepacked are rejected (AEE_EBADPARM) rather than silently
 * ignored -- the down matmul's quantization is inherently dynamic (it
 * quantizes the SwiGLU output, which the caller has never seen), so a
 * caller-supplied act_scale could only apply to stage 1, an accuracy
 * change dressed up as an option.
 *
 * Requires the in-place accumulator tile layout (hexkl_acc_layout usable);
 * returns AEE_EUNSUPPORTED otherwise -- the gate/up split of the dequant
 * has no vendor-copy fallback.
 */
int hexkl_mm_u8i4_fused_run(hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base,
                            uint32_t vtcm_size, uint32_t config_off, uint32_t M,
                            uint32_t K, const uint32_t *handles,
                            const float *act_f32, float *out,
                            const hexkl_mm_opts *opts);

/**
 * @brief [L2, split-call variant] gate_up matmul -> SwiGLU -> requantize to
 *        u8 AH, ONE weight -- see hexkl_mm_u8i4_dma.c's doc comment for why
 *        this exists alongside hexkl_mm_u8i4_fused_run rather than instead
 *        of it (smaller VTCM surface, reuses hexkl_mm_u8i4_layer_run's
 *        act_ah_prepacked path for the down matmul instead of reimplementing
 *        it). Feed out_ah/out_scale/out_zp straight into
 *        hexkl_mm_u8i4_layer_run via hexkl_mm_opts.act_ah_prepacked for the
 *        down matmul -- that path is already device-verified.
 *
 * @param[in]  handle_gate_up  registered handle; N must be even (gate|up
 *                            split) and N/2 must satisfy HexKL's own
 *                            32-divisibility (it becomes the down matmul's K)
 * @param[out] out_ah    m_pad(M) * (registered N / 2) bytes, AH-tiled --
 *                       same layout htp_act_quant.h's htp_quant_pack_u8_ah
 *                       produces and hexkl_mm_opts.act_ah_prepacked expects
 * @param[out] out_scale, out_zp  m_pad(M) entries each
 * @return AEE_SUCCESS, AEE_EBADPARM on a shape violation, AEE_ENOMEMORY if
 *         the block layout does not fit VTCM, AEE_EUNSUPPORTED if the
 *         in-place accumulator tile layout is unavailable
 */
int hexkl_mm_u8i4_gate_up_swiglu_run(hexkl_weight_u8i4_table *tbl,
                                     uint8_t *vtcm_base, uint32_t vtcm_size,
                                     uint32_t config_off, uint32_t M,
                                     uint32_t K, uint32_t handle_gate_up,
                                     const float *act_f32, uint8_t *out_ah,
                                     float *out_scale, int32_t *out_zp,
                                     hvx_worker_pool *pool);

#endif /* __NNTRAINER_HEXKL_MM_U8I4_DMA_H__ */

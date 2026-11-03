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

#include <stddef.h>
#include <stdint.h>

#include "hexkl_mm_opts.h"

/** @brief Upper bound on weights resident at once. A 28-layer qwen3-sized
 *         model registers 7 per layer (q,k,v,o,gate,up,down) = 196.
 *         LFM2.5-8B-A1B resident in full (doc 45) is 22 MoE layers x 64 +
 *         2 dense x 2 + 18 conv x 2 + 6 attention x 4 = 1472, so 512 --
 *         which bounded the MoE work to 8 layers at a time -- is not
 *         enough for the whole model. [plan 201 S1] Gemma-4-26B-A4B's
 *         expert pool in one PD is about 1300 slots of 2 weights each
 *         (plan 201 section 3.4), so 4096. The table is ~36 bytes per slot
 *         on the DSP; 4096 costs ~150 KB of the session struct (DSP heap,
 *         +75 KB over 2048). */
#define HEXKL_MM_U8I4_MAX_WEIGHTS 4096

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
  void *arrays;      /**< the one allocation the three arrays sit in */
  uint32_t K, N;
  int borrowed; /**< wh_bytes points into a host arena, not our heap */
  int clean;    /**< [#267 L1] a heap image: written once at registration
                     and flushed to DDR then, so no DSP cache holds a dirty
                     line of it and a DMA may read it around the L2 */
  /** 4 (QS4CX_WH) or 2 (QS2CX_WH); 0 in a slot never filled reads as 4.
      At 2, wh_bytes holds half as many bytes -- two bits a weight indexing
      @a pal -- which the M=1 GEMV looks up in a register
      (hvx_gemm_u8i2_wh_*) and the HMX path expands to int4 in VTCM
      (hvx_expand_i2i4.h). Both give the 4-bit path's int32 sums. */
  uint32_t bits;
  int8_t pal[4]; /**< the four int4 codes, ascending. Unused when bits == 4 */
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

/**
 * @brief Registers a weight whose WH bytes already sit in host memory the
 *        DSP has mapped (an arena), borrowing them in place.
 *
 * Nothing is copied and nothing is allocated for the bytes: the slot points
 * at @a wh and release() leaves it alone. The three N-sized arrays are
 * copied, they are small. With @a w_scale, @a colsum_w and @a bias all NULL
 * the scales and column sums are taken from the arena instead, where the
 * host laid them right after the WH bytes (N f32 scales, then N i32 column
 * sums; doc 52 section 10.30), and the bias is zero: @a wh must then be
 * followed by 8*N mapped bytes, which the caller checks against its arena.
 * The caller guarantees @a wh stays mapped and unchanged for the handle's
 * lifetime -- the arena code refuses to detach while a slot borrows from
 * it, which is the check that keeps that promise.
 *
 * @a wh must be WEIGHT_TILE_BYTES_U8I4-aligned: every DMA out of it starts
 * on a tile boundary and the ring's 2D descriptors assume it.
 */
int hexkl_weight_u8i4_register_arena(hexkl_weight_u8i4_table *tbl,
                                     uint32_t vtcm_size, uint32_t K, uint32_t N,
                                     const uint8_t *wh, const float *w_scale,
                                     const int32_t *colsum_w, const float *bias,
                                     uint32_t *out_handle);

/**
 * @brief [doc 52 section 10.25] Points a live arena-borrowed slot at new
 *        bytes of the same shape, in place: the handle keeps its number and
 *        its arrays, the scales and column sums are overwritten and the
 *        bias zeroed. What an expert swap does when the retired pair has
 *        the new expert's shape -- no allocation, no free, no slot search.
 *        NULL @a w_scale and @a colsum_w take them from the arena after
 *        @a wh, as hexkl_weight_u8i4_register_arena does. A non-NULL
 *        @a pal makes the slot 2-bit (QS2CX_WH codes at @a wh, the tail
 *        after whBytes2), NULL makes it 4-bit.
 *
 * @return AEE_SUCCESS, or AEE_EBADPARM (changing nothing) for a free,
 *         owned or differently shaped slot, or an unaligned @a wh.
 */
int hexkl_weight_u8i4_rebind_arena(hexkl_weight_u8i4_table *tbl, uint32_t h,
                                   uint32_t K, uint32_t N, const uint8_t *wh,
                                   const int8_t *pal, const float *w_scale,
                                   const int32_t *colsum_w);

/**
 * @brief hexkl_weight_u8i4_register_arena for QS2CX_WH: half the bytes,
 *        plus the palette.
 *
 * @a wh is whBytes2(K, N) = K*N/4 bytes of 2-bit codes in whPack's nibble
 * order, and @a pal the four int4 codes they index. NULL arrays take the
 * scales and column sums from the arena after the codes. The HMX path
 * expands a chunk in place after its DMA lands, so no VTCM is set aside for
 * it -- see hexkl_moe_expand_chunk.
 */
int hexkl_weight_u2i4_register_arena(hexkl_weight_u8i4_table *tbl,
                                     uint32_t vtcm_size, uint32_t K, uint32_t N,
                                     const uint8_t *wh, const int8_t *pal,
                                     const float *w_scale,
                                     const int32_t *colsum_w, const float *bias,
                                     uint32_t *out_handle);

/** @brief Whether any live slot borrows bytes inside [base, base+bytes). */
int hexkl_weight_u8i4_borrows(const hexkl_weight_u8i4_table *tbl,
                              const uint8_t *base, uint32_t bytes);

/**
 * @brief Copies a registered weight's baked WH bytes out, for caching.
 *
 * @a wh_out_len must equal the weight's (K/32)*(N/32)*512.
 * @return AEE_SUCCESS, or AEE_EBADPARM for an unknown handle, a free slot,
 *         or a length that does not match the weight's shape.
 */
int hexkl_weight_u8i4_export(const hexkl_weight_u8i4_table *tbl,
                             uint32_t handle, uint8_t *wh_out,
                             uint32_t wh_out_len);

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
 * @brief Integer-only FC: asymmetric UINT16 x symmetric INT4 -> INT32.
 * Activation is row-major M x K with one fixed encoding for the tensor;
 * K is a multiple of 64. Output consists of concatenated M x N_i blocks.
 * out[m,c] = sum_k((activation[m,k] - zero_point) * weight[k,c]) + bias[c].
 * bias is NULL, or n_handles pointers to N_i INT32 entries (a NULL entry
 * means zero). Bias uses accumulator units s_a * s_w[c], prepared offline.
 * The registry's float scales and biases are ignored. No float arithmetic,
 * dynamic quantization, residual or normalization occurs in this path.
 * Weights and column sums must already be prepared in offline WH layout;
 * this call only transfers the tiled weights to VTCM, without repacking.
 * Use hexkl_requantize_i32_u16 (hexkl_integer.h) to feed another integer FC.
 * Inputs/weights/bias/output/VTCM must not overlap. As with layer_run, the
 * caller owns the HMX lock and initializes accumulator readout beforehand.
 * Returns AEE_EFAILED on INT32 output overflow (discard the whole output),
 * AEE_EBADPARM for invalid shapes/weights, AEE_ENOMEMORY for insufficient
 * VTCM, AEE_EUNSUPPORTED for an unsupported accumulator layout, or a
 * HexKL error. Each byte sweep is checked to fit INT32; exact widened
 * intermediates are used wherever the dot/correction range is not proven.
 * Final overflow is always checked, including bias. No silent saturation.
 */
int hexkl_mm_u16i4_layer_run_i32(
  hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
  uint32_t config_off, uint32_t M, uint32_t K, const uint32_t *handles,
  uint32_t n_handles, const uint16_t *activation, uint16_t zero_point,
  const int32_t *const *bias, int32_t *out, hvx_worker_pool *pool);

struct hexkl_requant_u16;
/** @brief Same exact INT32 FC, plus fused UINT16 output for the next layer.
 * Both output buffers are required, disjoint, and concatenated M x N_i.
 * requant[i] has n_requant[i] entries: one or N_i (per channel), as defined
 * by hexkl_requant_u16 in hexkl_integer.h. Parameters are validated before
 * any output is written. On accumulator overflow discard BOTH outputs.
 * Requantization runs in the epilogue, overlapping subsequent HMX batches,
 * without rereading the INT32 output from DDR. Other i32 entry rules apply.
 */
int hexkl_mm_u16i4_layer_run_i32_requant(
  hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
  uint32_t config_off, uint32_t M, uint32_t K, const uint32_t *handles,
  uint32_t n_handles, const uint16_t *activation, uint16_t zero_point,
  const int32_t *const *bias, int32_t *out, hvx_worker_pool *pool,
  const struct hexkl_requant_u16 *const *requant, const uint32_t *n_requant,
  uint16_t *out_u16);

/** @brief Optional external AH byte planes for integer FC chaining.
 * Each image is [low plane][high plane], with ceil(M/64)*64*width bytes
 * per plane. Within a plane, (r,c) is at
 * (r/64)*64*width + (c/32)*2048 + (r%64)*32 + c%32.
 * Input width is K; output width is the sum of the handles' N values in
 * handle order, so output channels are interleaved into one logical tensor.
 * Both pointers, when supplied, must be 128-byte aligned, outside VTCM,
 * and disjoint from all other inputs/outputs. Capacities are in bytes.
 * The caller keeps buffers alive until return. Padding output rows are zero.
 */
typedef struct {
  const uint8_t *input;
  size_t input_bytes;
  uint8_t *output;
  size_t output_bytes;
} hexkl_u16_ah_io;

/** @brief Integer FC with optional packed input and directly packed output.
 * Supply exactly one of activation (row-major) or ah->input (AH planes).
 * Packed input is DMA-transferred to VTCM without an HVX packing pass.
 * out always receives the same concatenated INT32 blocks as the other APIs.
 * ah->output, if non-NULL, receives requantized AH planes directly from the
 * epilogue, without an intermediate row-major UINT16 array. requant and
 * n_requant are required only when producing this packed output.
 * To feed another FC, the caller must choose requant parameters defining
 * one common output encoding, and pass that encoding's zero-point there.
 * Per-channel multipliers can still reflect per-channel weight scales.
 * No conversion or float metadata processing is performed by this API.
 * The other integer FC error, overflow and ownership contracts apply.
 */
int hexkl_mm_u16i4_layer_run_i32_ah(
  hexkl_weight_u8i4_table *tbl, uint8_t *vtcm_base, uint32_t vtcm_size,
  uint32_t config_off, uint32_t M, uint32_t K, const uint32_t *handles,
  uint32_t n_handles, const uint16_t *activation, uint16_t zero_point,
  const int32_t *const *bias, int32_t *out, hvx_worker_pool *pool,
  const struct hexkl_requant_u16 *const *requant, const uint32_t *n_requant,
  const hexkl_u16_ah_io *ah);

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

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   hvx_tile_f16.h
 * @date   14 Sep 2026
 * @brief  HVX builders for the 32x32 fp16 HMX tile layouts attention needs
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * HexKL ships rm_to_ah_f16 / rm_to_wh_f16 / ah_to_rm_f16, but each is a
 * fixed 32x32 fp16 tile in and out, and the two "rm_to_*" variants that
 * take a DDR matrix pointer are ~400x slower when that pointer is VTCM
 * (measured on the prior attention branch). Attention needs four things
 * HexKL does not offer as one step: an f32 -> f16 conversion with a scale
 * fused into the activation interleave, a transposed weight tile so K
 * never needs a separate transpose pass, a diagonal tile for the
 * online-softmax rescale, and all of it reading rows that are already in
 * VTCM. These are those four, plus the inverse for the output store.
 *
 * Layout (the one HexKL's own fp16 functions produce, and the one the
 * tutorial and llama.cpp both rely on): a 32x32 fp16 tile is 16 HVX
 * vectors; vector i holds rows 2i and 2i+1 interleaved element by element,
 *
 *   vec[i] = { r[2i][0], r[2i+1][0], r[2i][1], r[2i+1][1], ... }
 *            (32 such pairs, ending with r[2i][31], r[2i+1][31])
 *
 * and for fp16 the activation (AH) and weight (WH) tiles share this
 * layout, so an accumulator read back as AH can be fed straight back in
 * as a weight. Nothing here is verified by construction: the probe entry
 * in test/htp/nntr_hvx_attn_f16.c compares every builder byte for byte
 * against HexKL's converters on the device, and the plan says to stop if
 * they differ.
 */

#ifndef __NNTRAINER_HVX_TILE_F16_H__
#define __NNTRAINER_HVX_TILE_F16_H__

#include <stdint.h>
#include <string.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

/** @brief Vectors per 32x32 fp16 tile. */
#define HVX_TILE_F16_VECS 16u
/** @brief fp16 elements per tile row. */
#define HVX_TILE_F16_COLS 32u
/** @brief Bytes per tile row (32 fp16). */
#define HVX_TILE_F16_ROW_BYTES 64u

/**
 * @brief Whole-vector load / store through memcpy.
 *
 * Casting a scalar array (float sc[64], uint32_t buf[16][32], ...) to
 * HVX_UVector* and dereferencing it is a strict-aliasing violation, and at
 * -O3 hexagon-clang does exploit it: the vector load can be scheduled
 * before the scalar stores that were meant to fill the array, so some
 * lanes read stale stack. Found on-device -- a softmax that was right in a
 * probe and wrong in the kernel, with nothing but inlining context
 * between them. memcpy has byte semantics, so the compiler must order it
 * after the stores, and it still emits a single vector access.
 */
static inline HVX_Vector hvx_tile_load_u(const void *p) {
  HVX_Vector v;
  memcpy(&v, p, sizeof(v));
  return v;
}
static inline void hvx_tile_store_u(void *p, HVX_Vector v) {
  memcpy(p, &v, sizeof(v));
}

/**
 * @brief Loads a 64-byte row (32 fp16) into the low half of a vector.
 *
 * Rows arrive at whatever stride the caller's matrix has, so this cannot
 * be a single aligned vector load; a memcpy into a zeroed vector is what
 * the compiler turns into an unaligned half-vector load without ever
 * reading past the row.
 */
static inline HVX_Vector hvx_tile_f16_load_row(const uint16_t *row) {
  HVX_Vector v = Q6_V_vzero();
  memcpy(&v, row, HVX_TILE_F16_ROW_BYTES);
  return v;
}

/**
 * @brief Interleaves two 32-element fp16 rows into one tile vector.
 *
 * vshuff with a -2 control shuffles at 2-byte granularity across the pair,
 * so the low half of the result is exactly the 64-element interleave the
 * tile layout wants. Argument order matters: vshuff(Vu=odd, Vv=even) puts
 * the even row's element first in each pair.
 */
static inline HVX_Vector hvx_tile_f16_interleave_rows(HVX_Vector even,
                                                      HVX_Vector odd) {
  return Q6_V_lo_W(Q6_W_vshuff_VVR(odd, even, -2));
}

/**
 * @brief Row-major fp16 rows -> one AH/WH tile.
 *
 * Reads 32 rows of 32 fp16 at @a src_stride elements between rows; writes
 * 16 vectors to @a dst (must be 2048-byte aligned VTCM). This is the same
 * transform as hexkl_micro_hmx_rm_to_ah_f16, but from arbitrary-stride
 * rows in either DDR or VTCM. Used for V (natural orientation is already
 * the weight orientation for P.V) and for f16 Q.
 */
static inline void hvx_tile_f16_rows_to_tile(HVX_Vector *dst,
                                             const uint16_t *src,
                                             uint32_t src_stride) {
  for (uint32_t i = 0; i < HVX_TILE_F16_VECS; ++i) {
    const HVX_Vector even = hvx_tile_f16_load_row(src + (2u * i) * src_stride);
    const HVX_Vector odd =
      hvx_tile_f16_load_row(src + (2u * i + 1u) * src_stride);
    dst[i] = hvx_tile_f16_interleave_rows(even, odd);
  }
}

/**
 * @brief Row-major f32 rows -> one fp16 AH tile, scaled.
 *
 * Converts on the way in and applies @a scale (attention folds
 * 1/sqrt(head_dim) * log2(e) here so the softmax can run base-2 and HMX
 * never needs its output-scale register, which HexKL does not expose).
 *
 * Fused conversion + interleave: a widening qf32 pair holds lanes 0..31
 * in its low vector and 32..63 in its high one, and the narrowing
 * Vhf = Wqf32 conversion writes low-vector lanes to EVEN fp16 lanes and
 * high-vector lanes to ODD ones -- the inverse of how Wqf32 = vmpy(Vhf,
 * Vhf) splits its input. So converting (row 2i, row 2i+1) as one pair
 * produces the interleaved tile vector directly, no vshuff. That lane
 * mapping is the one assumption in this file that is not spelled out in
 * the intrinsic name; the layout probe checks it against
 * hexkl_micro_hmx_rm_to_ah_f16 on the device.
 *
 * @param src         32 rows of at least 32 f32
 * @param src_stride  elements between consecutive rows
 */
static inline void hvx_tile_f16_rows_f32_to_tile(HVX_Vector *dst,
                                                 const float *src,
                                                 uint32_t src_stride,
                                                 float scale) {
  int bits;
  __builtin_memcpy(&bits, &scale, sizeof(bits));
  const HVX_Vector vscale = Q6_V_vsplat_R(bits);
  for (uint32_t i = 0; i < HVX_TILE_F16_VECS; ++i) {
    // FastRPC / DDR rows carry no vector alignment guarantee.
    const HVX_Vector r0 = hvx_tile_load_u(src + (2u * i) * src_stride);
    const HVX_Vector r1 = hvx_tile_load_u(src + (2u * i + 1u) * src_stride);
    const HVX_Vector q0 = Q6_Vqf32_vmpy_VsfVsf(r0, vscale);
    const HVX_Vector q1 = Q6_Vqf32_vmpy_VsfVsf(r1, vscale);
    dst[i] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0));
  }
}

/**
 * @brief Same as hvx_tile_f16_rows_f32_to_tile, but each of the 32 rows
 *        comes from its own pointer, and a NULL pointer means a zero row.
 *
 * Attention packs G query heads x Br tokens into one tile's rows, so
 * consecutive tile rows are not a constant stride apart in Q, and the rows
 * past G*Br (padding to 32) have no source at all.
 */
static inline void
hvx_tile_f16_rows_f32_gather_to_tile(HVX_Vector *dst, const float *const *rows,
                                     float scale) {
  int bits;
  __builtin_memcpy(&bits, &scale, sizeof(bits));
  const HVX_Vector vscale = Q6_V_vsplat_R(bits);
  for (uint32_t i = 0; i < HVX_TILE_F16_VECS; ++i) {
    const float *p0 = rows[2u * i];
    const float *p1 = rows[2u * i + 1u];
    const HVX_Vector r0 = p0 ? hvx_tile_load_u(p0) : Q6_V_vzero();
    const HVX_Vector r1 = p1 ? hvx_tile_load_u(p1) : Q6_V_vzero();
    const HVX_Vector q0 = Q6_Vqf32_vmpy_VsfVsf(r0, vscale);
    const HVX_Vector q1 = Q6_Vqf32_vmpy_VsfVsf(r1, vscale);
    dst[i] = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(q1, q0));
  }
}

/**
 * @brief Row-major fp16 rows -> the WH tile of their TRANSPOSE.
 *
 * For S = Q.K^T the weight operand is K^T, whose tile vector i must hold
 * K^T rows 2i and 2i+1 interleaved -- i.e. K columns 2i and 2i+1, taken
 * from every one of the 32 K rows in order. Two adjacent fp16 columns of
 * one K row are one 32-bit word, so tile vector i is simply word i of
 * every K row: the transform is a 32x16 -> 16x32 transpose at word
 * granularity, and no element ever moves inside its word.
 *
 * A vshuff network, no stack and no scalar access: the 32 rows pair up
 * at word granularity (rows 2j and 2j+1 interleaved: vector j holds word
 * i of both rows at lanes 2i, 2i+1), then four butterfly stages at 8, 16,
 * 32 and 64 bytes between vectors 1, 2, 4 and 8 apart bring every row's
 * word i together. Each stage's lo/hi are the lower/upper halves of the
 * pair, so vector k ends up holding word bitrev4(k): the final store
 * undoes that. Derived and checked on the lane emulation against the
 * word-transpose definition (tile_f16_host_check, bit for bit), and on
 * the device by ProbeLayoutsMatchHexkl.
 *
 * It replaced a scalar word transpose through a stack buffer: 256 scalar
 * loads and 512 scalar stores per tile, and the rows come from VTCM,
 * where a scalar access costs an L2 miss each -- measured at 91 ms of a
 * 93 ms attention call against 0.5 ms of HMX work (doc 20 section 1).
 * The row loads here read a whole vector of which the row is the first
 * 64 bytes, like hvx_tile_f16_rows_to_tile does.
 *
 * @param src         32 K rows of at least 32 fp16 (natural [kv][hd])
 * @param src_stride  elements between consecutive rows
 */
static inline void hvx_tile_f16_rows_to_tile_transposed(HVX_Vector *dst,
                                                        const uint16_t *src,
                                                        uint32_t src_stride) {
  HVX_Vector a[HVX_TILE_F16_VECS];
  for (uint32_t j = 0; j < HVX_TILE_F16_VECS; ++j) {
    const HVX_Vector r0 = hvx_tile_load_u(src + (2u * j) * src_stride);
    const HVX_Vector r1 = hvx_tile_load_u(src + (2u * j + 1u) * src_stride);
    a[j] = Q6_V_lo_W(Q6_W_vshuff_VVR(r1, r0, -4));
  }
#define HVX_TILE_F16_T_STAGE(g, d)                                             \
  for (uint32_t p = 0; p < HVX_TILE_F16_VECS; ++p) {                           \
    if ((p & (d)) == 0u) {                                                     \
      const HVX_VectorPair w = Q6_W_vshuff_VVR(a[p + (d)], a[p], -(g));        \
      a[p] = Q6_V_lo_W(w);                                                     \
      a[p + (d)] = Q6_V_hi_W(w);                                               \
    }                                                                          \
  }
  HVX_TILE_F16_T_STAGE(8, 1)
  HVX_TILE_F16_T_STAGE(16, 2)
  HVX_TILE_F16_T_STAGE(32, 4)
  HVX_TILE_F16_T_STAGE(64, 8)
#undef HVX_TILE_F16_T_STAGE
  for (uint32_t k = 0; k < HVX_TILE_F16_VECS; ++k) {
    const uint32_t i =
      ((k & 1u) << 3) | ((k & 2u) << 1) | ((k & 4u) >> 1) | ((k & 8u) >> 3);
    dst[i] = a[k];
  }
}

/**
 * @brief diag(v) as one AH tile, for the online-softmax rescale on HMX.
 *
 * Tile row r has v[r] at column r and zero elsewhere. In the interleaved
 * layout, row 2i's column 2i is fp16 lane 2*(2i) = 4i of vector i, and
 * row 2i+1's column 2i+1 is lane 2*(2i+1)+1 = 4i+3. Everything else is
 * zero. Built on the stack, stored as 16 full vectors.
 *
 * @param v  32 fp16 bit patterns, the diagonal
 */
static inline void hvx_tile_f16_diag(HVX_Vector *dst, const uint16_t *v) {
  uint16_t buf[HVX_TILE_F16_VECS][64];
  memset(buf, 0, sizeof(buf));
  for (uint32_t i = 0; i < HVX_TILE_F16_VECS; ++i) {
    buf[i][4u * i] = v[2u * i];
    buf[i][4u * i + 3u] = v[2u * i + 1u];
  }
  for (uint32_t i = 0; i < HVX_TILE_F16_VECS; ++i) {
    dst[i] = hvx_tile_load_u(buf[i]);
  }
}

/**
 * @brief One AH/WH tile -> 32 row-major fp16 rows.
 *
 * vdeal with a -2 control is the inverse of the vshuff above: the low
 * half of the result is the even row, the high half the odd row. Same as
 * hexkl_micro_hmx_ah_to_rm_f16 but to arbitrary-stride rows.
 *
 * @param dst_stride  elements between consecutive destination rows
 */
static inline void hvx_tile_f16_tile_to_rows(uint16_t *dst, uint32_t dst_stride,
                                             const HVX_Vector *src) {
  for (uint32_t i = 0; i < HVX_TILE_F16_VECS; ++i) {
    const HVX_VectorPair d = Q6_W_vdeal_VVR(Q6_V_vzero(), src[i], -2);
    const HVX_Vector even = Q6_V_lo_W(d);
    const HVX_Vector odd = Q6_V_hi_W(d);
    memcpy(dst + (2u * i) * dst_stride, &even, HVX_TILE_F16_ROW_BYTES);
    memcpy(dst + (2u * i + 1u) * dst_stride, &odd, HVX_TILE_F16_ROW_BYTES);
  }
}

#endif /* __NNTRAINER_HVX_TILE_F16_H__ */

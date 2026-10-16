// SPDX-License-Identifier: Apache-2.0
// TQ2CX ternary weight x int8 activation GEMV / GEMM kernels.
//
// Weight format (TQ2CX, one record per weight / per expert):
//   masks : N rows x U units, U = K / 32, unit = { uint pos; uint neg; }
//           bit b of pos (neg) set <=> w[n][32u + b] == +1 (-1); both clear
//           <=> 0; both set is illegal. 8 bytes per 32 input taps. Unit
//           (n, u) is stored at tq2_uidx(n, u, U) (stored layout, below).
//   scales: N fp32, directly after the masks (record tail).
//   Logical dim is the FC weight [K = in, N = out]; bytes are N rows.
// Activation format: int8 [M][K] row-major + per-row fp32 scale and int32
//   zero point, exactly what the v8c activation quantizer
//   (v8c_act_quant_f32 / _par in int8_int4_gemm_v8c.cl) produces:
//   x_fp ~= (x_q - zp) * x_scale.
//
// Math (all integer until the epilogue, so the result is order-independent
// and bit-exact against a scalar reference on the same int8 activation):
//   acc[m][n]  = sum_k x_q[m][k] * w[n][k]                  (w in {-1,0,+1})
//   wsum[n]    = sum_k w[n][k] = popcount(pos) - popcount(neg)
//   y[m][n]    = (float)(acc - zp[m] * wsum[n]) * x_scale[m] * scale[n]
//
// Unpack: each 4-bit slice of pos / neg covers 4 consecutive taps. A
// 16-entry table spreads the nibble into one 0/1 byte per tap (byte j = bit
// j, little endian, i.e. the same tap order as the packed activation dword),
// and the int8x4 weight is  w8 = SPREAD(pos4) | SPREAD(neg4) * 0xFF  (the
// two masks are disjoint, so the OR never mixes bytes and 1 * 0xFF = -1 as
// int8). The packed int8x4 product is dot_4x8packed_ss_int (dp4a); the XMX
// GEMM feeds the same dwords as the VNNI B operand of the int8 DPAS.
//
// TQ2_SPREAD_LUT selects a __constant 16-entry table; the default computes
// the identical value with one multiply ((v * 0x00204081) & 0x01010101),
// which avoids divergent constant-cache gathers on Intel.
//
// Requirements (checked by the host): K % 32 == 0, 4-byte aligned bases.
// The XMX GEMM additionally needs the 2D block-io surface rules: row bytes
// (K / 4) and K are multiples of 16 and >= 64, buffers 64-byte aligned.
//
// Compile with -cl-std=CL3.0 (Intel NEO exposes dot_4x8packed_* only there).

#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_khr_subgroup_clustered_reduce : enable

// ---- stored layout of the records (tq2cx_layout.h), set by the host ----
//   TQ2_LAYR == 0: row-major (v1), unit (n, u) at n * U + u, scales at N * U.
//   TQ2_LAYR == R, TQ2_LAYV == V: tiled (v2), rows in tiles of R padded to
//   Npad = ceil(N / R) * R, blocks of V units inside a tile, block b holding
//   R rows x w_b units row-major (w_b = min(V, U - b * V)); scales at Npad * U.
#ifndef TQ2_LAYR
#define TQ2_LAYR 0
#endif
#ifndef TQ2_LAYV
#define TQ2_LAYV 1
#endif
static inline long tq2_uidx(const int n, const int u, const int U) {
#if TQ2_LAYR == 0
  return (long)n * U + u;
#else
  const int t = n / TQ2_LAYR, r = n - t * TQ2_LAYR;
  const int b = u / TQ2_LAYV, ub = u - b * TQ2_LAYV;
  const int w = min(TQ2_LAYV, U - b * TQ2_LAYV);
  return (long)t * TQ2_LAYR * U + (long)b * TQ2_LAYV * TQ2_LAYR + (long)r * w +
         ub;
#endif
}
static inline int tq2_npad(const int N) {
#if TQ2_LAYR == 0
  return N;
#else
  return (N + TQ2_LAYR - 1) / TQ2_LAYR * TQ2_LAYR;
#endif
}
/* units u0 .. u0 + NU - 1 of one row are adjacent (u0 % NU == 0) */
#define TQ2_CONTIG(NU) (TQ2_LAYR == 0 || (TQ2_LAYV % (NU)) == 0)
/* NU units of row n from u0 (u0 % NU == 0) into dst[]; units >= U read 0 */
#define TQ2_LOADU(dst, W, n, u0, U, NU)                                        \
  do {                                                                         \
    if (TQ2_CONTIG(NU) && (u0) + (NU) <= (U)) {                                \
      __global const uint *p_ = (W) + 2 * tq2_uidx((n), (u0), (U));            \
      if ((NU) == 2) {                                                         \
        const uint4 v_ = vload4(0, p_);                                        \
        dst[0] = v_.s01;                                                       \
        dst[1 % (NU)] = v_.s23;                                                \
      } else {                                                                 \
        for (int i_ = 0; i_ < (NU); ++i_)                                      \
          dst[i_] = vload2(i_, p_);                                            \
      }                                                                        \
    } else {                                                                   \
      for (int i_ = 0; i_ < (NU); ++i_)                                        \
        dst[i_] = ((u0) + i_ < (U))                                            \
                    ? vload2(0, (W) + 2 * tq2_uidx((n), (u0) + i_, (U)))       \
                    : (uint2)(0u, 0u);                                         \
    }                                                                          \
  } while (0)
/* unit u of row n */
#define TQ2_UNIT(W, n, u, U) vload2(0, (W) + 2 * tq2_uidx((n), (u), (U)))

__constant uint TQ2_LUT[16] = {
  0x00000000u, 0x00000001u, 0x00000100u, 0x00000101u,
  0x00010000u, 0x00010001u, 0x00010100u, 0x00010101u,
  0x01000000u, 0x01000001u, 0x01000100u, 0x01000101u,
  0x01010000u, 0x01010001u, 0x01010100u, 0x01010101u};

#ifdef TQ2_SPREAD_LUT
#define TQ2_SPREAD(v) (TQ2_LUT[(v)])
#else
#define TQ2_SPREAD(v) (((v) * 0x00204081u) & 0x01010101u)
#endif

#ifndef TQ2_ACT_PERM
/** int8x4 weight dword for taps 4j..4j+3 of one 32-tap unit */
#define TQ2_W8(p, g, j)                                                        \
  (TQ2_SPREAD(((p) >> (4 * (j))) & 0xFu) |                                     \
   (TQ2_SPREAD(((g) >> (4 * (j))) & 0xFu) * 0xFFu))
#else
/** TQ2_ACT_PERM: the activation is stored in "TQ2 order" (tq2cx_act_perm):
 *  within every 32-tap block, dword j holds taps j, 8+j, 16+j, 24+j. The
 *  matching weight dword is then a plain shift + mask of each plane (bit
 *  8b+j of the mask lands in byte b), no nibble spread at all. */
#define TQ2_W8(p, g, j)                                                        \
  ((((p) >> (j)) & 0x01010101u) | ((((g) >> (j)) & 0x01010101u) * 0xFFu))
#endif

/** sum over one 32-tap unit: a = 32 int8 activations as 8 packed dwords */
static inline int tq2_unit_dot(const uint p, const uint g, const uint8 a) {
  return dot_4x8packed_ss_int(a.s0, TQ2_W8(p, g, 0)) +
         dot_4x8packed_ss_int(a.s1, TQ2_W8(p, g, 1)) +
         dot_4x8packed_ss_int(a.s2, TQ2_W8(p, g, 2)) +
         dot_4x8packed_ss_int(a.s3, TQ2_W8(p, g, 3)) +
         dot_4x8packed_ss_int(a.s4, TQ2_W8(p, g, 4)) +
         dot_4x8packed_ss_int(a.s5, TQ2_W8(p, g, 5)) +
         dot_4x8packed_ss_int(a.s6, TQ2_W8(p, g, 6)) +
         dot_4x8packed_ss_int(a.s7, TQ2_W8(p, g, 7));
}

static inline int tq2_unit_wsum(const uint p, const uint g) {
  return (int)popcount(p) - (int)popcount(g);
}

// =============================================================================
// GEMV variant A: one work-item per output channel, serial K loop.
// =============================================================================
__kernel void tq2cx_gemv_row(__global const uint2 *Wm,   // [N][U] mask units
                             __global const float *S,    // [N] scales
                             __global const char *X,     // [K] int8 act
                             __global const float *x_scale, // [1]
                             __global const int *x_zp,      // [1]
                             __global float *Y,             // [N]
                             const int N, const int K) {
  const int n = get_global_id(0);
  if (n >= N)
    return;
  const int U = K >> 5;
  __global const uint *wm = (__global const uint *)Wm;
  __global const uint *x = (__global const uint *)X;
  int acc = 0, ws = 0;
  for (int u = 0; u < U; ++u) {
    const uint2 m = TQ2_UNIT(wm, n, u, U);
    acc += tq2_unit_dot(m.x, m.y, vload8(u, x));
    ws += tq2_unit_wsum(m.x, m.y);
  }
  Y[n] = (float)(acc - x_zp[0] * ws) * x_scale[0] * S[n];
}

// =============================================================================
// GEMV variant B: subgroup-cooperative over K.
//   A subgroup of 16 lanes covers 16 / TQ2_LPC output channels; the TQ2_LPC
//   lanes of one channel stride over its units (lane kl takes units kl,
//   kl + LPC, ...), so one step of a channel reads LPC consecutive 8-byte
//   units = a contiguous 8*LPC-byte segment of the row. The partial int32
//   sums are combined with a clustered subgroup reduction (integer, exact).
//   A work-group holds TQ2_SGS subgroups; the activation row is optionally
//   staged in local memory (TQ2_ACT_LDS) and otherwise read through L1/L3.
//   LPC = 8 balances both target shapes: U = 88 (K = 2816) -> 11 units per
//   lane, U = 22 (K = 704) -> 3 / 2 units per lane.
// =============================================================================
#ifndef TQ2_LPC
#define TQ2_LPC 4
#endif
#ifndef TQ2_CPL
#define TQ2_CPL 1
#endif
#ifndef TQ2_SGS
#define TQ2_SGS 4
#endif
#ifndef TQ2_UPL
#define TQ2_UPL 2
#endif
#ifndef TQ2_MAX_K
#define TQ2_MAX_K 16384
#endif
#define TQ2_CPS (16 / TQ2_LPC * TQ2_CPL) /* channels per subgroup */
#define TQ2_CPW (TQ2_CPS * TQ2_SGS)       /* channels per work-group */

#if TQ2_LPC == 16
#define TQ2_REDUCE(v) sub_group_reduce_add(v)
#else
#define TQ2_REDUCE(v) sub_group_clustered_reduce_add((v), TQ2_LPC)
#endif

/** one work-group's worth of channels for one weight / one activation row.
 *  Each lane owns TQ2_CPL consecutive channels, so one activation load (32
 *  bytes per unit, 4x the 8 weight bytes it meets) is reused TQ2_CPL times. */
static inline void tq2_gemv_sg_body(__global const uint2 *Wm,
                                    __global const float *S,
                                    __global const char *X,
                                    const float xs, const int zp,
                                    __global float *Y, const int N,
                                    const int K, __local uint *xl) {
  const int U = K >> 5;
  const int lane = get_sub_group_local_id();
  const int kl = lane % TQ2_LPC;
  const int n0 = (int)get_group_id(0) * TQ2_CPW +
                 (int)get_sub_group_id() * TQ2_CPS +
                 (lane / TQ2_LPC) * TQ2_CPL;
#ifdef TQ2_ACT_LDS
  {
    __global const uint *xg = (__global const uint *)X;
    for (int i = get_local_id(0); i < (K >> 2); i += 16 * TQ2_SGS)
      xl[i] = xg[i];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
#define TQ2_XPTR xl
#else
  __global const uint *xg = (__global const uint *)X;
#define TQ2_XPTR xg
#endif
  int acc[TQ2_CPL], ws[TQ2_CPL];
#pragma unroll
  for (int c = 0; c < TQ2_CPL; ++c) {
    acc[c] = 0;
    ws[c] = 0;
  }
  if (n0 < N) {
    __global const uint *wm = (__global const uint *)Wm;
    /* each lane takes TQ2_UPL adjacent units per step (8 * UPL contiguous
     * bytes in either layout), all loads issued before the math; units past
     * U are skipped */
    for (int u0 = TQ2_UPL * kl; u0 < U; u0 += TQ2_UPL * TQ2_LPC) {
      uint2 m[TQ2_CPL][TQ2_UPL];
#pragma unroll
      for (int c = 0; c < TQ2_CPL; ++c) {
        if (n0 + c < N) {
          TQ2_LOADU(m[c], wm, n0 + c, u0, U, TQ2_UPL);
        } else {
#pragma unroll
          for (int i = 0; i < TQ2_UPL; ++i)
            m[c][i] = (uint2)(0u, 0u);
        }
      }
#pragma unroll
      for (int i = 0; i < TQ2_UPL; ++i) {
        if (u0 + i < U) {
          const uint8 a = vload8(u0 + i, TQ2_XPTR);
#pragma unroll
          for (int c = 0; c < TQ2_CPL; ++c) {
            acc[c] += tq2_unit_dot(m[c][i].x, m[c][i].y, a);
            ws[c] += tq2_unit_wsum(m[c][i].x, m[c][i].y);
          }
        }
      }
    }
  }
#undef TQ2_XPTR
#pragma unroll
  for (int c = 0; c < TQ2_CPL; ++c) {
    const int a = TQ2_REDUCE(acc[c]);
    const int s = TQ2_REDUCE(ws[c]);
    if (n0 + c < N && kl == 0)
      Y[n0 + c] = (float)(a - zp * s) * xs * S[n0 + c];
  }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16 * TQ2_SGS, 1, 1)))
__kernel void tq2cx_gemv(__global const uint2 *Wm,      // [N][U] mask units
                         __global const float *S,       // [N] scales
                         __global const char *X,        // [K] int8 act
                         __global const float *x_scale, // [1]
                         __global const int *x_zp,      // [1]
                         __global float *Y,             // [N]
                         const int N, const int K) {
#ifdef TQ2_ACT_LDS
  __local uint xl[TQ2_MAX_K / 4];
#else
  __local uint *xl = 0;
#endif
  tq2_gemv_sg_body(Wm, S, X, x_scale[0], x_zp[0], Y, N, K, xl);
}

// Expert batch: B weights of the same [K, N] shape in one launch, grid dim 1
// = weight index b. Weight b is the TQ2CX record starting at word offset
// rec_off[b] of Wbase (masks, then N fp32 scales), so a stacked expert
// tensor (records back to back, sec. 4.3 of the format) or any set of
// sub-allocations of one buffer can be addressed without copies. Its
// activation is row act_row[b] of the quantized activation matrix X
// [R][K] (scale / zero point per row) -- gate and up of the same token share
// one row, down of each expert has its own -- and its output is Y[b][N].
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16 * TQ2_SGS, 1, 1)))
__kernel void tq2cx_gemv_batch(__global const uint *Wbase,
                               __global const uint *rec_off, // [B] words
                               __global const char *X,       // [R][K]
                               __global const float *x_scale, // [R]
                               __global const int *x_zp,      // [R]
                               __global const int *act_row,   // [B]
                               __global float *Y,             // [B][N]
                               const int N, const int K) {
#ifdef TQ2_ACT_LDS
  __local uint xl[TQ2_MAX_K / 4];
#else
  __local uint *xl = 0;
#endif
  const int b = get_group_id(1);
  const int r = act_row[b];
  if (r < 0)
    return; // a row left to another launch (the whole group skips it)
  __global const uint *rec = Wbase + rec_off[b];
  __global const uint2 *Wm = (__global const uint2 *)rec;
  __global const float *S =
    (__global const float *)(rec + (long)tq2_npad(N) * (K >> 5) * 2);
  tq2_gemv_sg_body(Wm, S, X + (long)r * K, x_scale[r], x_zp[r],
                   Y + (long)b * N, N, K, xl);
}

// =============================================================================
// GEMM dp4a fallback (prefill, any device with integer dot product).
//   Work-item (n, m-block): one output channel x TQ2_TM rows. The unit is
//   loaded and unpacked once and reused for TQ2_TM activation rows; the
//   neighbouring work-items share the activation rows through the cache.
// =============================================================================
#ifndef TQ2_TM
#define TQ2_TM 8
#endif

__kernel void tq2cx_gemm_dp4a(__global const uint2 *Wm,     // [N][U]
                              __global const float *S,      // [N]
                              __global const char *X,       // [M][K]
                              __global const float *x_scale, // [M]
                              __global const int *x_zp,      // [M]
                              __global float *Y,             // [M][N]
                              const int M, const int N, const int K) {
  const int n = get_global_id(0);
  const int m0 = get_global_id(1) * TQ2_TM;
  if (n >= N)
    return;
  const int U = K >> 5;
  __global const uint *wm = (__global const uint *)Wm;
  int acc[TQ2_TM];
#pragma unroll
  for (int r = 0; r < TQ2_TM; ++r)
    acc[r] = 0;
  int ws = 0;
  for (int u = 0; u < U; ++u) {
    const uint2 m = TQ2_UNIT(wm, n, u, U);
    const uint p = m.x, g = m.y;
    const uint8 wv = (uint8)(TQ2_W8(p, g, 0), TQ2_W8(p, g, 1), TQ2_W8(p, g, 2),
                             TQ2_W8(p, g, 3), TQ2_W8(p, g, 4), TQ2_W8(p, g, 5),
                             TQ2_W8(p, g, 6), TQ2_W8(p, g, 7));
    ws += tq2_unit_wsum(p, g);
#pragma unroll
    for (int r = 0; r < TQ2_TM; ++r) {
      if (m0 + r < M) {
        const uint8 a =
          vload8(u, (__global const uint *)(X + (long)(m0 + r) * K));
        acc[r] += dot_4x8packed_ss_int(a.s0, wv.s0) +
                  dot_4x8packed_ss_int(a.s1, wv.s1) +
                  dot_4x8packed_ss_int(a.s2, wv.s2) +
                  dot_4x8packed_ss_int(a.s3, wv.s3) +
                  dot_4x8packed_ss_int(a.s4, wv.s4) +
                  dot_4x8packed_ss_int(a.s5, wv.s5) +
                  dot_4x8packed_ss_int(a.s6, wv.s6) +
                  dot_4x8packed_ss_int(a.s7, wv.s7);
      }
    }
  }
  const float s = S[n];
#pragma unroll
  for (int r = 0; r < TQ2_TM; ++r) {
    const int m = m0 + r;
    if (m < M)
      Y[(long)m * N + n] = (float)(acc[r] - x_zp[m] * ws) * x_scale[m] * s;
  }
}

// =============================================================================
// GEMM XMX / DPAS (Intel Xe2 / Xe3, SG16). Built like gemm_xmx_i4: a
// subgroup owns a (TQ2_MT * 8) x (TQ2_NT * 16) output tile, the K loop
// advances 4 units (128 taps) per step.
//   Weights: lane L takes the 8 words of channel n0 + L = 4 consecutive
//     (pos, neg) units. Row-major records: one
//     intel_sub_group_2d_block_read_transpose_32b_16r8x1c on the mask surface
//     (width = pitch = K / 4 bytes, height = N). Tiled records (16-row tiles,
//     2-unit blocks): two plain 16-byte loads 16 * 16 bytes apart, which the
//     16 lanes of a tile issue side by side (coalesced). Per unit the 8
//     int8x4 dwords TQ2_W8(p, g, 0..7) are exactly the VNNI B operand of one
//     k32 DPAS (dword g = taps 4g..4g+3 of column L). Out-of-range rows /
//     words read as 0, i.e. as zero weights.
//   Activations: intel_sub_group_2d_block_read_8b_8r32x1c, 8 rows x 32 taps
//     per DPAS, rows >= M read as 0 and are not stored.
//   wsum (for the zero-point term) is accumulated per lane from popcounts of
//   the same words.
// =============================================================================
#ifdef TQ2_XMX
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable

#ifndef TQ2_MT
#define TQ2_MT 4
#endif
#ifndef TQ2_NT
#define TQ2_NT 2
#endif
#ifndef TQ2_SG_M
#define TQ2_SG_M 1
#endif
#define TQ2_M_PER_SG (TQ2_MT * 8)
#define TQ2_N_PER_SG (TQ2_NT * 16)
#define TQ2_M_PER_WG (TQ2_M_PER_SG * TQ2_SG_M)

#if TQ2_LAYR != 0
/** the 4 units u0 .. u0 + 3 of row n as (p0, g0, .., p3, g3); rows past the
 *  padded rows and units past U read as 0 */
static inline uint8 tq2_xmx_wload(__global const uint *W, const int n,
                                  const int u0, const int N, const int U) {
  uint2 m[4];
  if (n >= tq2_npad(N)) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      m[i] = (uint2)(0u, 0u);
  } else if (TQ2_CONTIG(2) && u0 + 4 <= U) {
    const uint4 a = vload4(0, W + 2 * tq2_uidx(n, u0, U));
    const uint4 b = vload4(0, W + 2 * tq2_uidx(n, u0 + 2, U));
    m[0] = a.s01;
    m[1] = a.s23;
    m[2] = b.s01;
    m[3] = b.s23;
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i)
      m[i] = u0 + i < U ? TQ2_UNIT(W, n, u0 + i, U) : (uint2)(0u, 0u);
  }
  return (uint8)(m[0], m[1], m[2], m[3]);
}
#endif

/** one subgroup tile. Rows are [r0, r_end) of the activation surface X
 *  (height r_end, so rows past it read as 0); m0 is the tile's first row. */
static inline void tq2_xmx_tile(__global const char *X, const int r_end,
                                const int m0, __global const uint *Wm,
                                __global const float *S,
                                __global const float *x_scale,
                                __global const int *x_zp, __global float *Y,
                                const int N, const int K) {
  const int lane = get_sub_group_local_id();
  const int n0 = (int)get_group_id(0) * TQ2_N_PER_SG;
  const int row_bytes = K >> 2;

  int8 acc[TQ2_MT][TQ2_NT];
  int ws[TQ2_NT];
#pragma unroll
  for (int j = 0; j < TQ2_NT; ++j) {
    ws[j] = 0;
#pragma unroll
    for (int i = 0; i < TQ2_MT; ++i)
      acc[i][j] = (int8)(0);
  }

  for (int k0 = 0; k0 < K; k0 += 128) {
    uint8 wraw[TQ2_NT];
#pragma unroll
    for (int j = 0; j < TQ2_NT; ++j) {
#if TQ2_LAYR == 0
      uint draw[8];
      intel_sub_group_2d_block_read_transpose_32b_16r8x1c(
        (__global void *)Wm, row_bytes, N, row_bytes,
        (int2)(k0 >> 4, n0 + j * 16), draw);
      wraw[j] = *(uint8 *)draw;
#else
      wraw[j] = tq2_xmx_wload(Wm, n0 + j * 16 + lane, k0 >> 5, N, K >> 5);
#endif
    }

#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const int kk = k0 + s * 32;
      if (kk >= K)
        break;
      int8 b[TQ2_NT];
#pragma unroll
      for (int j = 0; j < TQ2_NT; ++j) {
        const uint p = (s == 0)   ? wraw[j].s0
                       : (s == 1) ? wraw[j].s2
                       : (s == 2) ? wraw[j].s4
                                  : wraw[j].s6;
        const uint g = (s == 0)   ? wraw[j].s1
                       : (s == 1) ? wraw[j].s3
                       : (s == 2) ? wraw[j].s5
                                  : wraw[j].s7;
        b[j] = as_int8((uint8)(TQ2_W8(p, g, 0), TQ2_W8(p, g, 1),
                               TQ2_W8(p, g, 2), TQ2_W8(p, g, 3),
                               TQ2_W8(p, g, 4), TQ2_W8(p, g, 5),
                               TQ2_W8(p, g, 6), TQ2_W8(p, g, 7)));
        ws[j] += tq2_unit_wsum(p, g);
      }
#pragma unroll
      for (int i = 0; i < TQ2_MT; ++i) {
        ushort a_raw[8];
        intel_sub_group_2d_block_read_8b_8r32x1c(
          (__global void *)X, K, r_end, K, (int2)(kk, m0 + i * 8), a_raw);
        const short8 a = *(short8 *)a_raw;
#pragma unroll
        for (int j = 0; j < TQ2_NT; ++j)
          acc[i][j] = intel_sub_group_i8_i8_matrix_mad_k32(a, b[j], acc[i][j]);
      }
    }
  }

#pragma unroll
  for (int j = 0; j < TQ2_NT; ++j) {
    const int col = n0 + j * 16 + lane;
    if (col >= N)
      continue;
    const float sw = S[col];
#pragma unroll
    for (int i = 0; i < TQ2_MT; ++i) {
      const int8 r = acc[i][j];
#pragma unroll
      for (int rr = 0; rr < 8; ++rr) {
        const int m = m0 + i * 8 + rr;
        if (m >= r_end)
          continue;
        Y[(long)m * N + col] =
          (float)(r[rr] - x_zp[m] * ws[j]) * x_scale[m] * sw;
      }
    }
  }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, TQ2_SG_M, 1)))
__kernel void tq2cx_gemm_xmx(__global const char *X,        // [M][K] int8
                             __global const uint *Wm,       // [N][K/16] words
                             __global const float *S,       // [N]
                             __global const float *x_scale, // [M]
                             __global const int *x_zp,      // [M]
                             __global float *Y,             // [M][N]
                             const int M, const int N, const int K) {
  const int m0 = (int)get_group_id(1) * TQ2_M_PER_WG +
                 (int)get_sub_group_id() * TQ2_M_PER_SG;
  tq2_xmx_tile(X, M, m0, Wm, S, x_scale, x_zp, Y, N, K);
}

// Grouped (MoE prefill): one launch for all experts, grid dim 2 = expert e.
// Expert e owns activation rows [row_off[e], row_off[e] + row_cnt[e]) of X
// (tokens already gathered per expert, scale / zero point per row) and
// writes the same rows of Y [rows][N]; its weight is the TQ2CX record at
// word offset rec_off[e] of Wbase (must be 64-byte aligned for the 2D block
// reads of a row-major record; true for stacked records of both target
// shapes). Grid dim 1 covers
// the largest row count; tiles past an expert's rows exit at once.
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, TQ2_SG_M, 1)))
__kernel void tq2cx_gemm_xmx_grouped(__global const char *X,
                                     __global const uint *Wbase,
                                     __global const uint *rec_off,
                                     __global const int *row_off,
                                     __global const int *row_cnt,
                                     __global const float *x_scale,
                                     __global const int *x_zp,
                                     __global float *Y, const int N,
                                     const int K) {
  const int e = get_group_id(2);
  const int cnt = row_cnt[e];
  const int m_local = (int)get_group_id(1) * TQ2_M_PER_WG +
                      (int)get_sub_group_id() * TQ2_M_PER_SG;
  if (m_local >= cnt)
    return;
  __global const uint *rec = Wbase + rec_off[e];
  __global const float *S =
    (__global const float *)(rec + (long)tq2_npad(N) * (K >> 5) * 2);
  const int r0 = row_off[e];
  tq2_xmx_tile(X, r0 + cnt, r0 + m_local, rec, S, x_scale, x_zp, Y, N, K);
}
#endif // TQ2_XMX

// Activation re-order for TQ2_ACT_PERM builds: per 32-tap block, output
// byte 4j + b = input tap 8b + j. Pure byte shuffle (the quantized values,
// scale and zero point are untouched), so results stay bit-identical; in a
// real pipeline this is the store index of the activation quantizer.
__kernel void tq2cx_act_perm(__global const uchar *src, __global uchar *dst,
                             const int n_blocks) {
  const int blk = get_global_id(0);
  if (blk >= n_blocks)
    return;
  const uint8 v = vload8(blk, (__global const uint *)src);
  const uint in[8] = {v.s0, v.s1, v.s2, v.s3, v.s4, v.s5, v.s6, v.s7};
  uint out[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    uint o = 0;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const int t = 8 * b + j; // input tap
      o |= ((in[t >> 2] >> (8 * (t & 3))) & 0xFFu) << (8 * b);
    }
    out[j] = o;
  }
  vstore8((uint8)(out[0], out[1], out[2], out[3], out[4], out[5], out[6],
                  out[7]),
          blk, (__global uint *)dst);
}

// Weight-read bandwidth probe (harness only): streams a buffer as uint4 and
// writes one value per work-item so the loads cannot be removed.
__kernel void tq2cx_bw_probe(__global const uint4 *src, __global uint *dst,
                             const int n_vec) {
  const int gid = get_global_id(0);
  const int gsz = get_global_size(0);
  uint4 s = (uint4)(0);
  for (int i = gid; i < n_vec; i += gsz)
    s ^= src[i];
  dst[gid] = s.x ^ s.y ^ s.z ^ s.w;
}

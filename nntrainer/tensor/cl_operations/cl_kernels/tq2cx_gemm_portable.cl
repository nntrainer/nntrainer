// SPDX-License-Identifier: Apache-2.0
// TQ2CX ternary weight x int8 activation kernels for devices with integer dot
// product but no Intel subgroup / matrix extensions (e.g. Adreno). Same
// record and activation formats and the same math as tq2cx_gemm.cl:
//   acc[m][n] = sum_k x_q[m][k] * w[n][k],  wsum[n] = sum_k w[n][k]
//   y[m][n]   = (float)(acc - zp[m] * wsum[n]) * x_scale[m] * scale[n]
// The sums are integer, so any summation order is exact; the epilogue is the
// reference expression, so results are bit-identical to the scalar reference
// on the same int8 activation.
//
// Only cl_khr_integer_dot_product (dot_4x8packed_ss_int) and plain work-group
// local memory are used. Requirements (checked by the host): K % 32 == 0,
// 8-byte aligned records. Units are read through tq2_uidx (stored layout).

#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#pragma OPENCL FP_CONTRACT OFF

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


#define TQ2P_SPREAD(v) (((v) * 0x00204081u) & 0x01010101u)
/** int8x4 weight dword for taps 4j..4j+3 of one 32-tap unit */
#define TQ2P_W8(p, g, j)                                                       \
  (TQ2P_SPREAD(((p) >> (4 * (j))) & 0xFu) |                                    \
   (TQ2P_SPREAD(((g) >> (4 * (j))) & 0xFu) * 0xFFu))

static inline int tq2p_unit_dot(const uint p, const uint g, const uint8 a) {
  return dot_4x8packed_ss_int(a.s0, TQ2P_W8(p, g, 0)) +
         dot_4x8packed_ss_int(a.s1, TQ2P_W8(p, g, 1)) +
         dot_4x8packed_ss_int(a.s2, TQ2P_W8(p, g, 2)) +
         dot_4x8packed_ss_int(a.s3, TQ2P_W8(p, g, 3)) +
         dot_4x8packed_ss_int(a.s4, TQ2P_W8(p, g, 4)) +
         dot_4x8packed_ss_int(a.s5, TQ2P_W8(p, g, 5)) +
         dot_4x8packed_ss_int(a.s6, TQ2P_W8(p, g, 6)) +
         dot_4x8packed_ss_int(a.s7, TQ2P_W8(p, g, 7));
}

static inline int tq2p_unit_wsum(const uint p, const uint g) {
  return (int)popcount(p) - (int)popcount(g);
}

// =============================================================================
// Expert batch GEMV (decode): B weights of one [K, N] shape in one launch,
// grid dim 1 = weight b. Weight b is the record at word offset rec_off[b] of
// Wbase (masks, then N fp32 scales); its activation is row act_row[b] of X
// [R][K] (scale / zero point per row); its output is Y[b][N].
//   The activation row is staged once per work-group in local memory,
//   PERMUTED per 32-tap block (dword j holds taps j, 8+j, 16+j, 24+j), so the
//   int8x4 weight dword is a shift and a mask of each plane instead of a
//   nibble spread. A work-group holds TQ2P_CPW channels x TQ2P_LPC lanes;
//   lane l of a channel takes units 2l, 2l+1, 2l+2*LPC, ... with one 16-byte
//   load per pair (adjacent in both layouts); the integer partial sums are
//   added in local memory.
//   Measured on Adreno 840 (decode shapes 2816 x 704 x 16 and 704 x 2816 x 8):
//   61 GB/s against 22 GB/s for the one-unit-per-lane form with the
//   activation read from global memory.
// =============================================================================
#ifndef TQ2P_LPC
#define TQ2P_LPC 4
#endif
#ifndef TQ2P_CPW
#define TQ2P_CPW 32
#endif
#define TQ2P_LWS (TQ2P_LPC * TQ2P_CPW)
#ifndef TQ2P_MAXK
#define TQ2P_MAXK 8192
#endif

/** int8x4 weight dword j of a unit against the permuted activation */
#define TQ2P_W8P(p, g, j)                                                      \
  ((((p) >> (j)) & 0x01010101u) | ((((g) >> (j)) & 0x01010101u) * 0xFFu))

static inline int tq2p_unit_dot_perm(const uint p, const uint g,
                                     const uint8 a) {
  return dot_4x8packed_ss_int(a.s0, TQ2P_W8P(p, g, 0)) +
         dot_4x8packed_ss_int(a.s1, TQ2P_W8P(p, g, 1)) +
         dot_4x8packed_ss_int(a.s2, TQ2P_W8P(p, g, 2)) +
         dot_4x8packed_ss_int(a.s3, TQ2P_W8P(p, g, 3)) +
         dot_4x8packed_ss_int(a.s4, TQ2P_W8P(p, g, 4)) +
         dot_4x8packed_ss_int(a.s5, TQ2P_W8P(p, g, 5)) +
         dot_4x8packed_ss_int(a.s6, TQ2P_W8P(p, g, 6)) +
         dot_4x8packed_ss_int(a.s7, TQ2P_W8P(p, g, 7));
}

// (No reqd_work_group_size: measured on Adreno 840, driver 0842.19.3, the
// attribute on this kernel compiles and enqueues cleanly but the kernel
// writes nothing. The host launches it with a TQ2P_LWS-wide work-group.)
__kernel void tq2cx_gemv_batch_portable(__global const uint *Wbase,
                                        __global const uint *rec_off,
                                        __global const char *X,
                                        __global const float *x_scale,
                                        __global const int *x_zp,
                                        __global const int *act_row,
                                        __global float *Y, const int N,
                                        const int K) {
  __local uint xl[TQ2P_MAXK / 4];
  __local int lacc[TQ2P_LWS];
  __local int lws[TQ2P_LWS];
  const int lid = get_local_id(0);
  const int c = lid / TQ2P_LPC, l = lid - c * TQ2P_LPC;
  const int n = get_group_id(0) * TQ2P_CPW + c;
  const int b = get_global_id(1);
  const int r = act_row[b];
  const int U = K >> 5;
  __global const uchar *xr = (__global const uchar *)(X + (long)r * K);
  for (int i = lid; i < (K >> 2); i += TQ2P_LWS) {
    __global const uchar *s = xr + (i >> 3) * 32;
    const int j = i & 7;
    xl[i] = (uint)s[j] | ((uint)s[8 + j] << 8) | ((uint)s[16 + j] << 16) |
            ((uint)s[24 + j] << 24);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  __global const uint *rec = Wbase + rec_off[b];
  __global const float *S =
    (__global const float *)(rec + (long)tq2_npad(N) * U * 2);
  int acc = 0, ws = 0;
  if (n < N) {
    for (int u = 2 * l; u < U; u += 2 * TQ2P_LPC) {
      if (u + 1 < U) {
        const uint4 m =
          TQ2_CONTIG(2) ? vload4(0, rec + 2 * tq2_uidx(n, u, U))
                        : (uint4)(TQ2_UNIT(rec, n, u, U),
                                  TQ2_UNIT(rec, n, u + 1, U));
        acc += tq2p_unit_dot_perm(m.x, m.y, vload8(u, xl)) +
               tq2p_unit_dot_perm(m.z, m.w, vload8(u + 1, xl));
        ws += tq2p_unit_wsum(m.x, m.y) + tq2p_unit_wsum(m.z, m.w);
      } else {
        const uint2 m = TQ2_UNIT(rec, n, u, U);
        acc += tq2p_unit_dot_perm(m.x, m.y, vload8(u, xl));
        ws += tq2p_unit_wsum(m.x, m.y);
      }
    }
  }
  lacc[lid] = acc;
  lws[lid] = ws;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (l == 0 && n < N) {
    int a = 0, s = 0;
    for (int i = 0; i < TQ2P_LPC; ++i) {
      a += lacc[c * TQ2P_LPC + i];
      s += lws[c * TQ2P_LPC + i];
    }
    Y[(long)b * N + n] = (float)(a - x_zp[r] * s) * x_scale[r] * S[n];
  }
}

// =============================================================================
// Grouped GEMM (MoE prefill): one launch for all experts, grid dim 2 = expert
// e. Expert e owns rows [row_off[e], row_off[e] + row_cnt[e]) of X [R][K] and
// writes the same rows of Y [R][N]; its weight is the record at word offset
// rec_off[e] of Wbase. Work-item (channel block, m-block): TQ2P_NB output
// channels x TQ2P_TM rows; each unit is loaded and unpacked once and reused
// for TQ2P_TM rows, and each 32-byte activation load feeds TQ2P_NB x 8 dot
// products. Channel c of the work-item at local id lid of work-group w is
// n = w * LWS * NB + c * LWS + lid (LWS = the work-group width), so a
// work-group still reads LWS adjacent channels per load instruction.
//   The sums are integer and the epilogue is the same expression, so every NB
// gives bit-identical outputs. Measured on Adreno 840 (E = 128 experts, a
// routed 334-token prompt, gate / up 704 x 2816 and down 2816 x 704): NB = 2
// runs the three launches 1.52x faster than one channel per work-item; the
// per-row activation load, not the dot rate or the weight stream, limits
// the one-channel form. TQ2P_NB == 1 keeps that form (one unit per step).
// =============================================================================
#ifndef TQ2P_TM
#define TQ2P_TM 8
#endif
#ifndef TQ2P_NB
#define TQ2P_NB 2
#endif

__kernel void tq2cx_gemm_grouped_portable(__global const char *X,
                                          __global const uint *Wbase,
                                          __global const uint *rec_off,
                                          __global const int *row_off,
                                          __global const int *row_cnt,
                                          __global const float *x_scale,
                                          __global const int *x_zp,
                                          __global float *Y, const int N,
                                          const int K) {
#if TQ2P_NB == 1
  const int n = get_global_id(0);
  const int e = get_global_id(2);
  const int cnt = row_cnt[e];
  const int m_local = get_global_id(1) * TQ2P_TM;
  if (n >= N || m_local >= cnt)
    return;
  const int r0 = row_off[e] + m_local;
  const int rows = min(TQ2P_TM, cnt - m_local);
  const int U = K >> 5;
  __global const uint *rec = Wbase + rec_off[e];
  __global const float *S =
    (__global const float *)(rec + (long)tq2_npad(N) * U * 2);
  int acc[TQ2P_TM];
  for (int i = 0; i < TQ2P_TM; ++i)
    acc[i] = 0;
  int ws = 0;
  for (int u = 0; u < U; ++u) {
    const uint2 m = TQ2_UNIT(rec, n, u, U);
    const uint p = m.x, g = m.y;
    const uint8 wv =
      (uint8)(TQ2P_W8(p, g, 0), TQ2P_W8(p, g, 1), TQ2P_W8(p, g, 2),
              TQ2P_W8(p, g, 3), TQ2P_W8(p, g, 4), TQ2P_W8(p, g, 5),
              TQ2P_W8(p, g, 6), TQ2P_W8(p, g, 7));
    ws += tq2p_unit_wsum(p, g);
    for (int i = 0; i < TQ2P_TM; ++i) {
      if (i < rows) {
        const uint8 a = vload8(u, (__global const uint *)(X + (long)(r0 + i) * K));
        acc[i] += dot_4x8packed_ss_int(a.s0, wv.s0) +
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
  for (int i = 0; i < rows; ++i) {
    const int m = r0 + i;
    Y[(long)m * N + n] = (float)(acc[i] - x_zp[m] * ws) * x_scale[m] * s;
  }
#else
  const int lid = get_local_id(0), lw = get_local_size(0);
  const int n0 = (int)get_group_id(0) * lw * TQ2P_NB + lid;
  const int e = get_global_id(2);
  const int cnt = row_cnt[e];
  const int m_local = get_global_id(1) * TQ2P_TM;
  if (n0 >= N || m_local >= cnt)
    return;
  const int r0 = row_off[e] + m_local;
  const int rows = min(TQ2P_TM, cnt - m_local);
  const int U = K >> 5;
  __global const uint *rec = Wbase + rec_off[e];
  __global const float *S =
    (__global const float *)(rec + (long)tq2_npad(N) * U * 2);
  __global const uint *xr = (__global const uint *)(X + (long)r0 * K);
  const int KD = K >> 2;
  /* channels past N read row N - 1 and are not stored */
  int nc[TQ2P_NB];
  for (int c = 0; c < TQ2P_NB; ++c)
    nc[c] = min(n0 + c * lw, N - 1);
  int acc[TQ2P_NB][TQ2P_TM], ws[TQ2P_NB];
  for (int c = 0; c < TQ2P_NB; ++c) {
    ws[c] = 0;
    for (int i = 0; i < TQ2P_TM; ++i)
      acc[c][i] = 0;
  }
  /* two units per step: one 16-byte load per channel where adjacent */
  for (int u = 0; u < U; u += 2) {
    uint4 m[TQ2P_NB];
    for (int c = 0; c < TQ2P_NB; ++c)
      m[c] = (TQ2_CONTIG(2) && u + 1 < U)
               ? vload4(0, rec + 2 * tq2_uidx(nc[c], u, U))
               : (uint4)(TQ2_UNIT(rec, nc[c], u, U),
                         u + 1 < U ? TQ2_UNIT(rec, nc[c], u + 1, U)
                                   : (uint2)(0u, 0u));
    for (int h = 0; h < 2; ++h) {
      uint8 wv[TQ2P_NB];
      for (int c = 0; c < TQ2P_NB; ++c) {
        const uint p = h ? m[c].z : m[c].x, g = h ? m[c].w : m[c].y;
        ws[c] += tq2p_unit_wsum(p, g);
        wv[c] = (uint8)(TQ2P_W8(p, g, 0), TQ2P_W8(p, g, 1), TQ2P_W8(p, g, 2),
                        TQ2P_W8(p, g, 3), TQ2P_W8(p, g, 4), TQ2P_W8(p, g, 5),
                        TQ2P_W8(p, g, 6), TQ2P_W8(p, g, 7));
      }
      /* the zero unit past an odd U adds 0 to both sums */
      for (int i = 0; i < TQ2P_TM; ++i) {
        if (i < rows && u + h < U) {
          const uint8 a = vload8(u + h, xr + (long)i * KD);
          for (int c = 0; c < TQ2P_NB; ++c)
            acc[c][i] += dot_4x8packed_ss_int(a.s0, wv[c].s0) +
                         dot_4x8packed_ss_int(a.s1, wv[c].s1) +
                         dot_4x8packed_ss_int(a.s2, wv[c].s2) +
                         dot_4x8packed_ss_int(a.s3, wv[c].s3) +
                         dot_4x8packed_ss_int(a.s4, wv[c].s4) +
                         dot_4x8packed_ss_int(a.s5, wv[c].s5) +
                         dot_4x8packed_ss_int(a.s6, wv[c].s6) +
                         dot_4x8packed_ss_int(a.s7, wv[c].s7);
        }
      }
    }
  }
  for (int c = 0; c < TQ2P_NB; ++c) {
    const int n = n0 + c * lw;
    if (n < N) {
      const float s = S[n];
      for (int i = 0; i < rows; ++i) {
        const int mr = r0 + i;
        Y[(long)mr * N + n] =
          (float)(acc[c][i] - x_zp[mr] * ws[c]) * x_scale[mr] * s;
      }
    }
  }
#endif
}

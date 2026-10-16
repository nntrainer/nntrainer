// SPDX-License-Identifier: Apache-2.0
// The MoE router on the device: the deterministic router of moe_route_det.h
// term for term (only IEEE add / multiply / fma and exact operations in a
// fixed order), so it selects the same experts with the same weights as the
// host router on any device.
//
// Decode (one row), three launches:
// tq2moe_route_norm: one work-group: the rms scale of the row, then the
// normalised, scaled row y[H] the projection reads.
// tq2moe_route_logits: one lane per (expert, RT_SEG-wide segment of the
// hidden axis): partial[seg][e] = the fma chain over the segment.
// tq2moe_route_select: one work-group of E lanes: logits = partial sums in
// segment order, softmax, top-k by rank, renormalisation, per-expert scale
// (tq2moe_route_select_ref: the same in its one-lane reference form);
// out ids[K] / w[K] in ascending expert id (the host combine order) and the
// selected experts' record offsets for the batched GEMVs (off_gu = gate
// offsets then up offsets, off_d = down offsets).
//
// Prefill (T rows), eight launches, no local memory and no barrier:
// tq2moe_route_sumsq_rows: lane l of 32 per row: the host's lane-l chain of
// the sum of squares, acc[t][l].
// tq2moe_route_y_rows: folds the 32 lane sums into the rms scale as the host
// does and writes y[t][f] = ((x * scale) * norm) * (rscale * hscale).
// tq2moe_route_seg_rows: one work-item per (expert, segment, four rows): the
// segment's fma chain, partial[t][s][e] (rows in chunks, see the host).
// tq2moe_route_fold_rows: one work-item per (expert, row): the segment sums
// added in order, logits[t][e].
// tq2moe_route_exp_rows / _prob_rows / _rank_rows: one work-item per (expert,
// row): exponential against the row maximum, probability against the row
// sum, rank placement of the selected K: out[t][0..K) = ids, out[t][K..2K) =
// probability bits, in rank order (as the host router returns them).
// tq2moe_route_fin_rows: one work-item per row: renormalisation and
// per-expert scale of the K weights, in place.
//
// Built with -DRT_EXT=r (streamed experts) the decode select kernels take one
// more argument, ext[K + r]: the ids of ranks 0 .. K + r - 1 in rank order
// (the selected K, then the next r), a recency hint for the expert slots. The
// selection itself is the same code either way. tq2moe_route_split: the
// hit / miss split of a streamed decode step.
//
// Built with -DRT_SOFT_FMA on a device without a hardware fp32 fma
// (DeviceCaps::fp32_fma), where fma() is a slow library routine: rt_fma then
// builds the same correctly rounded result from multiplies and adds.
//
// tq2moe_route_logits / _seg_rows built with -DRT_W_BF16 read the router
// weight as the high 16 bits of each fp32 value (tq2RouterWeightBf16: only
// when every low half is zero), widened back exactly: the same fp32
// operands, half the bytes.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF
#define RT_SEG 32
#define RT_MAXH 4096
#define RT_MAXE 256
#define RT_MAXK 16

// fma(a, b, c), correctly rounded, from fp32 multiplies and adds only (the
// Dekker product and two-sum are exact, the low parts are added rounded to
// odd, then the high part rounded to nearest: Boldo & Melquiond's emulated
// fma). Outside the range where those steps are exact (huge operands, a
// product below 2^-100) it calls fma().
inline float rt_fma(const float a, const float b, const float c) {
#ifdef RT_SOFT_FMA
  const float ph = a * b;
  const float m = fabs(ph);
  if (fabs(a) <= 0x1p100f && fabs(b) <= 0x1p100f && fabs(c) <= 0x1p120f &&
      m <= 0x1p100f && (m >= 0x1p-100f || a == 0.0f || b == 0.0f)) {
    const float ta = 4097.0f * a, tb = 4097.0f * b;
    const float ah = ta - (ta - a), al = a - ah;
    const float bh = tb - (tb - b), bl = b - bh;
    const float pl = ((ah * bh - ph) + ah * bl + al * bh) + al * bl;
    const float th = c + ph;
    const float tz = th - c;
    const float tl = (c - (th - tz)) + (ph - tz);
    const float v = tl + pl;
    if (v == 0.0f)
      return th;
    const float vz = v - tl;
    const float ve = (tl - (v - vz)) + (pl - vz);
    uint vu = as_uint(v);
    if (ve != 0.0f && (vu & 1u) == 0u)
      vu = ((ve > 0.0f) == (v > 0.0f)) ? vu + 1u : vu - 1u;
    return th + as_float(vu);
  }
#endif
  return fma(a, b, c);
}

// The deterministic primitives of moe_route_det.h, term for term: only IEEE
// add / multiply / fma and exact operations, so the host and every device
// produce the same bits whatever their division / sqrt / exp accuracy.
inline float rt_div(const float x, const float s) {
  float y = 1.0f / s;
  float e = rt_fma(-s, y, 1.0f);
  y = rt_fma(e, y, y);
  e = rt_fma(-s, y, 1.0f);
  y = rt_fma(e, y, y);
  const float q = x * y;
  const float r = rt_fma(-s, q, x);
  return r == 0.0f ? q : rt_fma(r, y, q);
}
inline float rt_rsqrt(const float a) {
  float y = as_float(0x5f375a86u - (as_uint(a) >> 1));
  const float h = 0.5f * a;
  for (int it = 0; it < 3; ++it) {
    const float hy = h * y;
    const float r = rt_fma(-hy, y, 1.5f);
    y = y * r;
  }
  return y;
}
inline float rt_exp(const float x) {
  if (x < -87.0f)
    return 0.0f;
  const float t = x * 1.44269504088896341f;
  const float n = rint(t);
  float r = rt_fma(-n, 0.693145751953125f, x);
  r = rt_fma(-n, 1.428606765330187e-06f, r);
  float p = 1.3888889e-3f;
  p = rt_fma(p, r, 8.3333333e-3f);
  p = rt_fma(p, r, 4.1666667e-2f);
  p = rt_fma(p, r, 1.6666667e-1f);
  p = rt_fma(p, r, 0.5f);
  p = rt_fma(p, r, 1.0f);
  p = rt_fma(p, r, 1.0f);
  return ldexp(p, (int)n);
}

__kernel void tq2moe_route_norm(__global const half *xh,
                                __global const float *xf, const int x_half,
                                __global const float *norm,
                                const int has_norm,
                                __global const float *rscale,
                                const float hscale, const float eps,
                                const int H, __global float *y) {
  __local float lx[RT_MAXH];
  __local float lacc[32];
  __local float lscale;
  const int tid = get_local_id(0);
  const int lws = get_local_size(0);
  for (int i = tid; 4 * i < H; i += lws) // H % 8 == 0
    vstore4(x_half ? vload_half4(i, xh) : vload4(i, xf), i, lx);
  barrier(CLK_LOCAL_MEM_FENCE);
  // sum of squares in the host order: lane l of 32 sums x*x over i = l,
  // l + 32, ... with fma; the 8-wide tail goes to lanes 0..7. The square of
  // an fp16 value is exact in fp32 (22 bits), so for a half row x*x + acc is
  // the same bits as fma(x, x, acc) without the fma.
  if (tid < 32) {
    const int n32 = H / 32 * 32;
    float acc = 0.0f;
    if (x_half) {
      for (int i = tid; i < n32; i += 32)
        acc = lx[i] * lx[i] + acc;
      if (tid < 8)
        for (int i = n32 + tid; i + 8 - tid <= H; i += 8)
          acc = lx[i] * lx[i] + acc;
    } else {
      for (int i = tid; i < n32; i += 32)
        acc = rt_fma(lx[i], lx[i], acc);
      if (tid < 8)
        for (int i = n32 + tid; i + 8 - tid <= H; i += 8)
          acc = rt_fma(lx[i], lx[i], acc);
    }
    lacc[tid] = acc;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid == 0) {
    float h[4];
    for (int j = 0; j < 4; ++j) {
      __local const float *v = lacc + 8 * j;
      h[j] = ((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7]));
    }
    const float sumsq = ((h[0] + h[1]) + h[2]) + h[3];
    const float mean = rt_div(sumsq, (float)H);
    lscale = rt_rsqrt(mean + eps);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float scale = lscale;
  for (int i = tid; i < H; i += lws) {
    float v = lx[i] * scale;
    if (has_norm)
      v = v * norm[i];
    y[i] = v * (rscale[i] * hscale);
  }
}

#ifdef RT_W_BF16
#define RT_WT ushort
#define RT_WLOAD(i) as_float(((uint)W[i]) << 16)
#else
#define RT_WT float
#define RT_WLOAD(i) W[i]
#endif

__kernel void tq2moe_route_logits(__global const float *y,
                                  __global const RT_WT *W, const int H,
                                  const int E, __global float *partial) {
  const int e = get_global_id(0);
  const int seg = get_global_id(1);
  if (e >= E)
    return;
  const int f0 = seg * RT_SEG;
  const int n = min(RT_SEG, H - f0);
  float acc = 0.0f;
#pragma unroll 8
  for (int f = 0; f < n; ++f)
    acc = rt_fma(y[f0 + f], RT_WLOAD((long)(f0 + f) * E + e), acc);
  partial[(long)seg * E + e] = acc;
}

__kernel void tq2moe_route_select_ref(__global const float *partial,
                                  const int nseg, __global const float *pes,
                                  __global const uint *off_tab, const int E,
                                  const int K, __global uint *ids,
                                  __global float *wout, __global uint *off_gu,
                                  __global uint *off_d
#ifdef RT_EXT
                                  , __global uint *ext
#endif
                                  ) {
  __local float lp[RT_MAXE];
  __local float lred;
  __local int rank_id[RT_MAXK];
  __local float rank_p[RT_MAXK];
  const int e = get_local_id(0);
  float l = 0.0f;
  if (e < E) {
    l = partial[e];
    for (int s = 1; s < nseg; ++s)
      l = l + partial[(long)s * E + e];
    lp[e] = l;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e == 0) {
    float mx = lp[0];
    for (int i = 1; i < E; ++i)
      mx = fmax(mx, lp[i]);
    lred = mx;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float ex = e < E ? rt_exp(l - lred) : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e < E)
    lp[e] = ex;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e == 0) {
    float sum = 0.0f;
    for (int i = 0; i < E; ++i)
      sum = sum + lp[i];
    lred = sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float p = e < E ? rt_div(ex, lred) : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e < E)
    lp[e] = p;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e < E) {
    // rank among the experts: larger first, ties to the lower id
    int rank = 0;
    for (int i = 0; i < E; ++i)
      rank += (lp[i] > p || (lp[i] == p && i < e)) ? 1 : 0;
    if (rank < K) {
      rank_id[rank] = e;
      rank_p[rank] = p;
    }
#ifdef RT_EXT
    else if (rank < K + RT_EXT)
      ext[rank] = (uint)e;
#endif
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e == 0) {
    int sel[RT_MAXK];
    float val[RT_MAXK];
    float s = 0.0f;
    for (int j = 0; j < K; ++j) {
      sel[j] = rank_id[j];
#ifdef RT_EXT
      ext[j] = (uint)rank_id[j];
#endif
      val[j] = rank_p[j];
      s = s + val[j];
    }
    for (int j = 0; j < K; ++j)
      val[j] = rt_div(val[j], s) * pes[sel[j]];
    // ascending expert id: the order the host combine sums in
    for (int a = 1; a < K; ++a)
      for (int b = a; b > 0 && sel[b - 1] > sel[b]; --b) {
        const int ti = sel[b]; sel[b] = sel[b - 1]; sel[b - 1] = ti;
        const float tv = val[b]; val[b] = val[b - 1]; val[b - 1] = tv;
      }
    for (int a = 0; a < K; ++a) {
      ids[a] = (uint)sel[a];
      wout[a] = val[a];
      off_gu[a] = off_tab[sel[a]];
      off_gu[K + a] = off_tab[E + sel[a]];
      off_d[a] = off_tab[2 * E + sel[a]];
    }
  }
}

// The same selection as tq2moe_route_select_ref, bit for bit, with less
// latency: the segment loads are issued eight ahead (the adds keep the
// segment order), the max is a two-level reduction (fmax is exact in any
// order), every lane runs the softmax sum itself (the same strict
// i = 0..E-1 chain, read four at a time from local memory, so no barrier
// waits on one lane), the rank reads four probabilities per local load and
// the K selected experts are renormalised and put in ascending id order by K
// lanes at once (no per-lane arrays).
__kernel void tq2moe_route_select(__global const float *partial,
                                  const int nseg, __global const float *pes,
                                  __global const uint *off_tab, const int E,
                                  const int K, __global uint *ids,
                                  __global float *wout, __global uint *off_gu,
                                  __global uint *off_d
#ifdef RT_EXT
                                  , __global uint *ext
#endif
                                  ) {
  __local float4 lp4[RT_MAXE / 4];
  __local float lm[16];
  __local int rank_id[RT_MAXK];
  __local float rank_p[RT_MAXK];
  __local float *lp = (__local float *)lp4;
  const int e = get_local_id(0);
  const int lws = get_local_size(0); // a multiple of 16, >= E
  float l = 0.0f;
  if (e < E) {
    __global const float *pe = partial + e;
    l = pe[0];
    int s = 1;
    for (; s + 8 <= nseg; s += 8) {
      const float v0 = pe[(long)(s + 0) * E], v1 = pe[(long)(s + 1) * E];
      const float v2 = pe[(long)(s + 2) * E], v3 = pe[(long)(s + 3) * E];
      const float v4 = pe[(long)(s + 4) * E], v5 = pe[(long)(s + 5) * E];
      const float v6 = pe[(long)(s + 6) * E], v7 = pe[(long)(s + 7) * E];
      l = l + v0;
      l = l + v1;
      l = l + v2;
      l = l + v3;
      l = l + v4;
      l = l + v5;
      l = l + v6;
      l = l + v7;
    }
    for (; s < nseg; ++s)
      l = l + pe[(long)s * E];
  }
  lp[e] = e < E ? l : -INFINITY;
  barrier(CLK_LOCAL_MEM_FENCE);
  // max: 16 lanes take lws / 16 values each, then every lane the 16
  if (e < 16) {
    const int g = lws / 16;
    float m = lp[e * g];
    for (int i = 1; i < g; ++i)
      m = fmax(m, lp[e * g + i]);
    lm[e] = m;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  float mx = lm[0];
  for (int i = 1; i < 16; ++i)
    mx = fmax(mx, lm[i]);
  const float ex = e < E ? rt_exp(l - mx) : 0.0f;
  lp[e] = ex; // lp was last read before the barrier above
  barrier(CLK_LOCAL_MEM_FENCE);
  float sum = 0.0f;
  {
    const int e4 = E / 4;
    for (int i = 0; i < e4; ++i) {
      const float4 v = lp4[i];
      sum = sum + v.s0;
      sum = sum + v.s1;
      sum = sum + v.s2;
      sum = sum + v.s3;
    }
    for (int i = 4 * e4; i < E; ++i)
      sum = sum + lp[i];
  }
  const float p = e < E ? rt_div(ex, sum) : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);
  lp[e] = e < E ? p : -INFINITY;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e < E) {
    // rank among the experts: larger first, ties to the lower id (the
    // padding is -inf and never counts)
    int rank = 0;
    const int n4 = (E + 3) / 4;
    for (int i = 0; i < n4; ++i) {
      const float4 v = lp4[i];
      const int b = 4 * i;
      rank += (v.s0 > p || (v.s0 == p && b < e)) ? 1 : 0;
      rank += (v.s1 > p || (v.s1 == p && b + 1 < e)) ? 1 : 0;
      rank += (v.s2 > p || (v.s2 == p && b + 2 < e)) ? 1 : 0;
      rank += (v.s3 > p || (v.s3 == p && b + 3 < e)) ? 1 : 0;
    }
    if (rank < K) {
      rank_id[rank] = e;
      rank_p[rank] = p;
    }
#ifdef RT_EXT
    else if (rank < K + RT_EXT)
      ext[rank] = (uint)e;
#endif
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (e < K) {
    float s = 0.0f;
    for (int j = 0; j < K; ++j)
      s = s + rank_p[j];
    const int id = rank_id[e];
#ifdef RT_EXT
    ext[e] = (uint)id;
#endif
    const float v = rt_div(rank_p[e], s) * pes[id];
    // position in ascending expert id (the ids are distinct)
    int pos = 0;
    for (int j = 0; j < K; ++j)
      pos += rank_id[j] < id ? 1 : 0;
    ids[pos] = (uint)id;
    wout[pos] = v;
    off_gu[pos] = off_tab[id];
    off_gu[K + pos] = off_tab[E + id];
    off_d[pos] = off_tab[2 * E + id];
  }
}

// The hit / miss split of a streamed decode step (one lane per selected
// expert): res[e] != 0 when expert e is resident. The GEMVs take an
// activation row per record and skip a record whose row is -1, so the
// launch over the hits gets the miss rows as -1 and the launch over the
// misses the hit rows: gate / up rows a and K + a read row 0, down row a
// reads row a.
__kernel void tq2moe_route_split(__global const uint *ids,
                                 __global const uint *res, const int K,
                                 __global int *gu_hit, __global int *gu_miss,
                                 __global int *d_hit, __global int *d_miss) {
  const int a = get_global_id(0);
  if (a >= K)
    return;
  const int hit = res[ids[a]] != 0u;
  gu_hit[a] = hit ? 0 : -1;
  gu_hit[K + a] = hit ? 0 : -1;
  gu_miss[a] = hit ? -1 : 0;
  gu_miss[K + a] = hit ? -1 : 0;
  d_hit[a] = hit ? a : -1;
  d_miss[a] = hit ? -1 : a;
}

// ---- prefill: T rows -------------------------------------------------------
// No local memory and no barrier (one Adreno driver returns wrong results for
// local tiles + barrier loops at two or more work-groups per SP; a row's
// cross-lane step goes through global memory and the next launch instead).

// acc[t][l] = lane l's sum of squares of row t, the order of
// moe_route_det_row: i = l, l + 32, ... below H / 32 * 32, then the 8-wide
// tail on lanes 0..7. gws = (32, T).
__kernel void tq2moe_route_sumsq_rows(__global const half *xh,
                                      __global const float *xf,
                                      const int x_half, const int H,
                                      const int T, __global float *acc) {
  const int l = get_global_id(0);
  const int t = get_global_id(1);
  if (l >= 32 || t >= T)
    return;
  const long base = (long)t * H;
  const int n32 = H / 32 * 32;
  float a = 0.0f;
  if (x_half) {
    // the square of an fp16 value is exact in fp32: x * x + a is fma's bits
    for (int i = l; i < n32; i += 32) {
      const float v = vload_half(base + i, xh);
      a = v * v + a;
    }
    if (l < 8)
      for (int i = n32 + l; i + 8 - l <= H; i += 8) {
        const float v = vload_half(base + i, xh);
        a = v * v + a;
      }
  } else {
    for (int i = l; i < n32; i += 32) {
      const float v = xf[base + i];
      a = rt_fma(v, v, a);
    }
    if (l < 8)
      for (int i = n32 + l; i + 8 - l <= H; i += 8) {
        const float v = xf[base + i];
        a = rt_fma(v, v, a);
      }
  }
  acc[(long)t * 32 + l] = a;
}

// y[t][f] = ((x * scale) * norm) * (rscale * hscale), scale folded from the
// 32 lane sums as the host does: ((v0+v4)+(v1+v5))+((v2+v6)+(v3+v7)) per 8,
// then ((h0+h1)+h2)+h3, rsqrt(sumsq / H + eps). Four elements per work-item.
// gws = (H / 4, T).
__kernel void tq2moe_route_y_rows(__global const half *xh,
                                  __global const float *xf, const int x_half,
                                  __global const float *acc,
                                  __global const float *norm,
                                  const int has_norm,
                                  __global const float *rscale,
                                  const float hscale, const float eps,
                                  const int H, const int T,
                                  __global float *y) {
  const int f4 = get_global_id(0);
  const int t = get_global_id(1);
  if (4 * f4 >= H || t >= T)
    return;
  __global const float *v = acc + (long)t * 32;
  float h[4];
  for (int j = 0; j < 4; ++j) {
    __global const float *u = v + 8 * j;
    h[j] = ((u[0] + u[4]) + (u[1] + u[5])) + ((u[2] + u[6]) + (u[3] + u[7]));
  }
  const float sumsq = ((h[0] + h[1]) + h[2]) + h[3];
  const float scale = rt_rsqrt(rt_div(sumsq, (float)H) + eps);
  const long base = (long)t * H;
  const float4 x = x_half ? vload_half4(0, xh + base + 4 * f4)
                          : vload4(0, xf + base + 4 * f4);
  const float4 rs = vload4(f4, rscale);
  float4 o = x * scale;
  if (has_norm)
    o = o * vload4(f4, norm);
  o = o * (rs * hscale);
  vstore4(o, f4, y + base);
}

// rt_fma(a, b, c) for a weight b shared by several rows, its Dekker halves
// bh + bl taken once (RT_WSPLIT): the same operations as rt_fma without the
// split of b. With RT_W_BF16 b has at most 8 significant bits, so its split
// is bh == b, bl == +0: the two products with bl are zeros and are left out
// (they can change only the sign of a zero low part, which the result does
// not depend on: v = tl + pl is then tl, or zero and th is returned).
inline float rt_fma_w(const float a, const float b, const float bh,
                      const float bl, const float c) {
#ifdef RT_SOFT_FMA
  const float ph = a * b;
  const float m = fabs(ph);
  if (fabs(a) <= 0x1p100f && fabs(b) <= 0x1p100f && fabs(c) <= 0x1p120f &&
      m <= 0x1p100f && (m >= 0x1p-100f || a == 0.0f || b == 0.0f)) {
    const float ta = 4097.0f * a;
    const float ah = ta - (ta - a), al = a - ah;
#ifdef RT_W_BF16
    const float pl = (ah * bh - ph) + al * bh;
#else
    const float pl = ((ah * bh - ph) + ah * bl + al * bh) + al * bl;
#endif
    const float th = c + ph;
    const float tz = th - c;
    const float tl = (c - (th - tz)) + (ph - tz);
    const float v = tl + pl;
    if (v == 0.0f)
      return th;
    const float vz = v - tl;
    const float ve = (tl - (v - vz)) + (pl - vz);
    uint vu = as_uint(v);
    if (ve != 0.0f && (vu & 1u) == 0u)
      vu = ((ve > 0.0f) == (v > 0.0f)) ? vu + 1u : vu - 1u;
    return th + as_float(vu);
  }
#endif
  return fma(a, b, c);
}

// the Dekker halves of a weight (rt_fma's split), once per weight load
#if defined(RT_SOFT_FMA) && !defined(RT_W_BF16)
#define RT_WSPLIT(w, h, l)                                                     \
  const float h##_t = 4097.0f * (w);                                           \
  const float h = h##_t - (h##_t - (w));                                       \
  const float l = (w)-h;
#else
#define RT_WSPLIT(w, h, l)                                                     \
  const float h = (w);                                                         \
  const float l = 0.0f;
#endif

// Four f of one row's segment chain: a = fma(y[f], w[f], a), f ascending.
#define RT_FMA4(a, yp, f)                                                      \
  {                                                                            \
    const float4 v_ = vload4(0, (yp) + (f));                                   \
    a = rt_fma_w(v_.s0, w0, wh0, wl0, a);                                      \
    a = rt_fma_w(v_.s1, w1, wh1, wl1, a);                                      \
    a = rt_fma_w(v_.s2, w2, wh2, wl2, a);                                      \
    a = rt_fma_w(v_.s3, w3, wh3, wl3, a);                                      \
  }
// the four weights of expert e at f .. f + 3 of the hidden axis, and their
// halves
#define RT_W4(f)                                                               \
  const float w0 = RT_WLOAD((long)(f)*E + e);                                  \
  const float w1 = RT_WLOAD((long)((f) + 1) * E + e);                          \
  const float w2 = RT_WLOAD((long)((f) + 2) * E + e);                          \
  const float w3 = RT_WLOAD((long)((f) + 3) * E + e);                          \
  RT_WSPLIT(w0, wh0, wl0)                                                      \
  RT_WSPLIT(w1, wh1, wl1)                                                      \
  RT_WSPLIT(w2, wh2, wl2)                                                      \
  RT_WSPLIT(w3, wh3, wl3)

// Segment partials of the logits: partial[t - row0][s][e] = the fma chain of
// segment s (f ascending from 0.0f) for eight rows t of one expert e. A lane
// per (expert, segment, eight rows), so a short prompt still fills the
// device: the chains are 32 fma long instead of H, and the lanes are E *
// nseg * ceil(rows / 8) instead of E * ceil(rows / 8). No arrays (named
// accumulators and row pointers), no local memory. Rows past the last are
// computed from the last row and not stored. gws = (E rounded up, nseg,
// ceil(n / 8)) over the rows [row0, row0 + n) of y.
__kernel void tq2moe_route_seg_rows(__global const float *y,
                                    __global const RT_WT *W, const int H,
                                    const int E, const int row0, const int n,
                                    const int nseg, __global float *partial) {
  const int e = get_global_id(0);
  const int s = get_global_id(1);
  const int r0 = get_global_id(2) * 8;
  if (e >= E || r0 >= n)
    return;
  const int f0 = s * RT_SEG;
  const int len = min(RT_SEG, H - f0); // a multiple of 8 (H % 8 == 0)
  __global const float *yb = y + (long)row0 * H + f0;
  __global const float *y0 = yb + (long)r0 * H;
  __global const float *y1 = yb + (long)min(r0 + 1, n - 1) * H;
  __global const float *y2 = yb + (long)min(r0 + 2, n - 1) * H;
  __global const float *y3 = yb + (long)min(r0 + 3, n - 1) * H;
  __global const float *y4 = yb + (long)min(r0 + 4, n - 1) * H;
  __global const float *y5 = yb + (long)min(r0 + 5, n - 1) * H;
  __global const float *y6 = yb + (long)min(r0 + 6, n - 1) * H;
  __global const float *y7 = yb + (long)min(r0 + 7, n - 1) * H;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  for (int f = 0; f < len; f += 4) {
    RT_W4(f0 + f)
    RT_FMA4(a0, y0, f)
    RT_FMA4(a1, y1, f)
    RT_FMA4(a2, y2, f)
    RT_FMA4(a3, y3, f)
    RT_FMA4(a4, y4, f)
    RT_FMA4(a5, y5, f)
    RT_FMA4(a6, y6, f)
    RT_FMA4(a7, y7, f)
  }
  __global float *p = partial + ((long)r0 * nseg + s) * E + e;
  const long rs = (long)nseg * E;
  p[0] = a0;
  if (r0 + 1 < n)
    p[rs] = a1;
  if (r0 + 2 < n)
    p[2 * rs] = a2;
  if (r0 + 3 < n)
    p[3 * rs] = a3;
  if (r0 + 4 < n)
    p[4 * rs] = a4;
  if (r0 + 5 < n)
    p[5 * rs] = a5;
  if (r0 + 6 < n)
    p[6 * rs] = a6;
  if (r0 + 7 < n)
    p[7 * rs] = a7;
}

// The logits of four rows of one expert in one lane: every segment's chain
// as tq2moe_route_seg_rows computes it, the segment sums added in order
// (tq2moe_route_seg_rows + tq2moe_route_fold_rows in one pass, without the
// partials), for a call with enough rows to fill the device this way.
// gws = (E rounded up, ceil(n / 4)) over the rows [row0, row0 + n).
__kernel void tq2moe_route_chain_rows(__global const float *y,
                                      __global const RT_WT *W, const int H,
                                      const int E, const int row0,
                                      const int n, __global float *logits) {
  const int e = get_global_id(0);
  const int r0 = get_global_id(1) * 4;
  if (e >= E || r0 >= n)
    return;
  __global const float *yb = y + (long)row0 * H;
  __global const float *y0 = yb + (long)r0 * H;
  __global const float *y1 = yb + (long)min(r0 + 1, n - 1) * H;
  __global const float *y2 = yb + (long)min(r0 + 2, n - 1) * H;
  __global const float *y3 = yb + (long)min(r0 + 3, n - 1) * H;
  float l0 = 0.0f, l1 = 0.0f, l2 = 0.0f, l3 = 0.0f;
  for (int f0 = 0; f0 < H; f0 += RT_SEG) {
    const int f1 = min(f0 + RT_SEG, H);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int f = f0; f < f1; f += 4) {
      RT_W4(f)
      RT_FMA4(a0, y0, f)
      RT_FMA4(a1, y1, f)
      RT_FMA4(a2, y2, f)
      RT_FMA4(a3, y3, f)
    }
    if (f0 == 0) {
      l0 = a0;
      l1 = a1;
      l2 = a2;
      l3 = a3;
    } else {
      l0 = l0 + a0;
      l1 = l1 + a1;
      l2 = l2 + a2;
      l3 = l3 + a3;
    }
  }
  __global float *o = logits + (long)(row0 + r0) * E + e;
  o[0] = l0;
  if (r0 + 1 < n)
    o[E] = l1;
  if (r0 + 2 < n)
    o[2 * E] = l2;
  if (r0 + 3 < n)
    o[3 * E] = l3;
}

// logits[row0 + t][e] = the segment partials of row t added in segment order
// (the host's l = a_0, then l = l + a_s). gws = (E rounded up, n).
__kernel void tq2moe_route_fold_rows(__global const float *partial,
                                     const int E, const int row0, const int n,
                                     const int nseg, __global float *logits) {
  const int e = get_global_id(0);
  const int t = get_global_id(1);
  if (e >= E || t >= n)
    return;
  __global const float *p = partial + (long)t * nseg * E + e;
  float l = p[0];
  for (int s = 1; s < nseg; ++s)
    l = l + p[(long)s * E];
  logits[(long)(row0 + t) * E + e] = l;
}

// ex[t][e] = rt_exp(logits[t][e] - max of row t). Every lane takes the row
// maximum itself, in the host's order (fmax is exact). gws = (E rounded up,
// T).
__kernel void tq2moe_route_exp_rows(__global const float *logits, const int E,
                                    const int T, __global float *ex) {
  const int e = get_global_id(0);
  const int t = get_global_id(1);
  if (e >= E || t >= T)
    return;
  __global const float *lp = logits + (long)t * E;
  float mx = lp[0];
  for (int i = 1; i < E; ++i)
    mx = fmax(mx, lp[i]);
  ex[(long)t * E + e] = rt_exp(lp[e] - mx);
}

// p[t][e] = rt_div(ex[t][e], sum of row t). Every lane adds the row itself,
// the host's strict i = 0 .. E - 1 chain, so every lane has the same sum.
// gws = (E rounded up, T).
__kernel void tq2moe_route_prob_rows(__global const float *ex, const int E,
                                     const int T, __global float *p) {
  const int e = get_global_id(0);
  const int t = get_global_id(1);
  if (e >= E || t >= T)
    return;
  __global const float *xp = ex + (long)t * E;
  float sum = 0.0f;
  for (int i = 0; i < E; ++i)
    sum = sum + xp[i];
  p[(long)t * E + e] = rt_div(xp[e], sum);
}

// The host's rank placement: rank of e = how many experts come before it
// (larger probability first, ties to the lower id); a rank below K writes
// out[t][rank] = e and out[t][K + rank] = its probability's bits (the ranks
// are distinct, so every slot is written once). gws = (E rounded up, T).
__kernel void tq2moe_route_rank_rows(__global const float *p, const int E,
                                     const int K, const int T,
                                     __global uint *out) {
  const int e = get_global_id(0);
  const int t = get_global_id(1);
  if (e >= E || t >= T)
    return;
  __global const float *pp = p + (long)t * E;
  const float pe = pp[e];
  int rank = 0;
  for (int i = 0; i < E; ++i) {
    const float pi = pp[i];
    rank += (pi > pe || (pi == pe && i < e)) ? 1 : 0;
  }
  if (rank < K) {
    __global uint *o = out + (long)t * 2 * K;
    o[rank] = (uint)e;
    o[K + rank] = as_uint(pe);
  }
}

// The selected weights renormalised over their sum (rank order) and scaled
// per expert, in place: out[t][K + j] = rt_div(p_j, s) * pes[id_j]. A lane
// per row, no private arrays. gws = (T rounded up).
__kernel void tq2moe_route_fin_rows(__global const float *pes, const int K,
                                    const int T, __global uint *out) {
  const int t = get_global_id(0);
  if (t >= T)
    return;
  __global uint *o = out + (long)t * 2 * K;
  float s = 0.0f;
  for (int j = 0; j < K; ++j)
    s = s + as_float(o[K + j]);
  for (int j = 0; j < K; ++j)
    o[K + j] = as_uint(rt_div(as_float(o[K + j]), s) * pes[o[j]]);
}

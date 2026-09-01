// SPDX-License-Identifier: Apache-2.0
// Two-1x1-conv attention for prefill, per ML Drift section 3.7
// (matmul-as-convolution pattern; K-cache OHWI, V-cache reversed-dims;
// avoids the per-WI register state that makes flash attention spill on
// mobile GPUs).
//
// Pipeline:
//   K1 qk_matmul   : Q[M, HD_Q]  @ K[N_kv, HD_KV]^T  -> scores[H, M, N_kv]
//   K2 softmax_row : row-wise softmax over the N_kv axis (per (h, m))
//   K3 sv_matmul   : scores[H, M, N_kv] @ V[N_kv, HD_KV] -> O[M, HD_Q]
//
// All three kernels treat each (head_q) tile independently. GQA is
// resolved at compile time inside each kernel via head_kv = head_q /
// gqa. Q/K/V/O are FP16-bit (uint16) in global memory; reductions and
// softmax run in FP32 register precision.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// =============================================================
// QK matmul: scores[h, m, n] = (Q[m, h, :] . K[n, h_kv, :]) * scale
//   Each WI computes a TM_QK x TN_QK tile of the (M, N_kv) score
//   matrix for a fixed head_q. The d-axis is fully reduced inside
//   the WI; d is small (=128 for Qwen3 / Llama-class models).
//
// gws = (N_kv / TN_QK, M / TM_QK, H)
// Each WI's private accumulator = TM_QK*TN_QK floats (e.g. 4*8 = 32);
// fits comfortably in the per-WI register file with room for K/Q
// staging.
// =============================================================
#ifndef TM_QK
#define TM_QK 4
#endif
#ifndef TN_QK
#define TN_QK 8
#endif

__kernel void
qk_matmul_f16(__global const half *Q, // [M, HD_Q] fp16, row-major
              __global const half *K, // [N_kv, HD_KV] fp16, row-major
              __global half *scores,  // [H, M, N_kv] fp16, row-major
              const int M, const int N_kv, const int d, const int HD_Q,
              const int HD_KV, const int gqa, const int causal,
              const float scale) {
  const int n0 = get_global_id(0) * TN_QK;
  const int m0 = get_global_id(1) * TM_QK;
  const int head_q = get_global_id(2);
  const int head_kv = head_q / gqa;

  if (m0 >= M || n0 >= N_kv)
    return;

  float acc[TM_QK][TN_QK];
#pragma unroll
  for (int i = 0; i < TM_QK; i++)
#pragma unroll
    for (int j = 0; j < TN_QK; j++)
      acc[i][j] = 0.0f;

  // Reduction over d.
  for (int x = 0; x < d; x++) {
    half q_col[TM_QK];
    half k_col[TN_QK];
#pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const int m = m0 + i;
      q_col[i] = (m < M) ? Q[(long)m * HD_Q + head_q * d + x] : (half)0.0f;
    }
#pragma unroll
    for (int j = 0; j < TN_QK; j++) {
      const int n = n0 + j;
      k_col[j] = (n < N_kv) ? K[(long)n * HD_KV + head_kv * d + x] : (half)0.0f;
    }
#pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const float qf = (float)q_col[i];
#pragma unroll
      for (int j = 0; j < TN_QK; j++)
        acc[i][j] += qf * (float)k_col[j];
    }
  }

  // Apply scale + causal mask + write out. The mask must be applied
  // before softmax sees the value, so writing -INFINITY here keeps the
  // softmax kernel mask-agnostic.
  const long score_base = (long)head_q * (long)M * (long)N_kv;
#pragma unroll
  for (int i = 0; i < TM_QK; i++) {
    const int m = m0 + i;
    if (m >= M)
      continue;
#pragma unroll
    for (int j = 0; j < TN_QK; j++) {
      const int n = n0 + j;
      if (n >= N_kv)
        continue;
      float v = acc[i][j] * scale;
      if (causal && n > m)
        v = -INFINITY;
      scores[score_base + (long)m * N_kv + n] = (half)v;
    }
  }
}

// =============================================================
// §3.8 OHWI K-cache variant of qk_matmul_f16.
// K is laid out as [H_kv, S_max, d] (per-head contiguous over the
// S/d axes — the OHWI "convolution weight" form), not the default
// row-major [N_kv, H_kv * d]. The kernel only reads N_kv rows out
// of the S_max allocated; S_max is the head stride.
//
// Element index:  K[(long)head_kv * (long)S_max * d + n * d + x]
//
// Per-head contiguous d-axis reads (stride 1 inside the inner d loop)
// are more cache-line friendly than the concat layout's strided reads
// (stride = HD_KV per token), so the scalar kernel should match or
// beat the concat variant. Image2d OHWI variant is a follow-up.
// =============================================================
__kernel void qk_matmul_f16_ohwi(
  __global const half *Q, // [M, HD_Q] fp16, row-major (unchanged)
  __global const half *K, // [H_kv, S_max, d] fp16, OHWI
  __global half *scores,  // [H, M, N_kv] fp16, row-major
  const int M, const int N_kv, const int d, const int HD_Q, const int S_max,
  const int gqa, const int causal, const float scale,
  const int local_window) { // >0: mask keys n with n+W <= q_pos
  const int n0 = get_global_id(0) * TN_QK;
  const int m0 = get_global_id(1) * TM_QK;
  const int head_q = get_global_id(2);
  const int head_kv = head_q / gqa;

  if (m0 >= M || n0 >= N_kv)
    return;

  float acc[TM_QK][TN_QK];
#pragma unroll
  for (int i = 0; i < TM_QK; i++)
#pragma unroll
    for (int j = 0; j < TN_QK; j++)
      acc[i][j] = 0.0f;

  const long k_head_base = (long)head_kv * (long)S_max * (long)d;

  for (int x = 0; x < d; x++) {
    half q_col[TM_QK];
    half k_col[TN_QK];
#pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const int m = m0 + i;
      q_col[i] = (m < M) ? Q[(long)m * HD_Q + head_q * d + x] : (half)0.0f;
    }
#pragma unroll
    for (int j = 0; j < TN_QK; j++) {
      const int n = n0 + j;
      k_col[j] = (n < N_kv) ? K[k_head_base + (long)n * d + x] : (half)0.0f;
    }
#pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const float qf = (float)q_col[i];
#pragma unroll
      for (int j = 0; j < TN_QK; j++)
        acc[i][j] += qf * (float)k_col[j];
    }
  }

  // Query row m's ABSOLUTE position is (N_kv - M) + m (same convention as
  // the _img variant): prefill M==N_kv gives q_off=0 (the old `n > m`);
  // decode M=1 puts the single query at N_kv-1.
  const int q_off = N_kv - M;
  const long score_base = (long)head_q * (long)M * (long)N_kv;
#pragma unroll
  for (int i = 0; i < TM_QK; i++) {
    const int m = m0 + i;
    if (m >= M)
      continue;
#pragma unroll
    for (int j = 0; j < TN_QK; j++) {
      const int n = n0 + j;
      if (n >= N_kv)
        continue;
      float v = acc[i][j] * scale;
      // Causal upper bound + sliding-window lower bound (flash formula:
      // key n is visible iff q_pos-W < n <= q_pos; W=0 means no window).
      if (causal && (n > q_off + m ||
                     (local_window > 0 && n + local_window <= q_off + m)))
        v = -INFINITY;
      scores[score_base + (long)m * N_kv + n] = (half)v;
    }
  }
}

// =============================================================
// qk_matmul_f16_ohwi_img — paper §3.7/§3.8 K via image2d.
// K image layout: width = d_h/8 texels (8 halves per texel along d),
//                 height = H_kv * S_max
//   texel(d_tex, head_kv*S_max + n) = K[head_kv, n, d_tex*8 .. +7]
// Q stays in SVM (scalar half loads); only K benefits from the
// texture cache. Same TM_QK/TN_QK tiling as the buffer variant —
// inner loop over d in 8-wide chunks via 1 K image read per (n, d).
// =============================================================
// Image-sampling kernel. Compiled out with -DTCA_BUFFER_ONLY on runtimes
// (e.g. Intel NEO) whose SPIR-V backend cannot compile integer-coordinate
// read_imageui — otherwise clBuildProgram fails for the whole program and the
// image-FREE attention kernels (qk_matmul_f16_ohwi / softmax_row_f16 /
// sv_matmul_f16) used by the NNTR_OHWI_IMG=0 path could not register either.
// Default (Adreno) builds with no option, keeping that path bit-identical.
#ifndef TCA_BUFFER_ONLY
__kernel void qk_matmul_f16_ohwi_img(
  __global const half *Q,      // [M, HD_Q] fp16, row-major
  __read_only image2d_t K_img, // see comment above
  __global half *scores,       // [H, M, N_kv] fp16, row-major
  const int M, const int N_kv, const int d, const int HD_Q, const int S_max,
  const int gqa, const int causal, const float scale,
  const float softcap,      // Gemma2-style: >0 => cap*tanh(s/cap)
  const int local_window) { // >0: mask keys n with n+W <= q_pos
  const int n0 = get_global_id(0) * TN_QK;
  const int m0 = get_global_id(1) * TM_QK;
  const int head_q = get_global_id(2);
  const int head_kv = head_q / gqa;

  if (m0 >= M || n0 >= N_kv)
    return;

  // Causal mask uses the query's ABSOLUTE position. Query row m holds the last
  // M positions of the N_kv context, so its absolute position is (N_kv-M)+m.
  // For prefill M==N_kv (q_off=0, the old `n > m`); for decode M=1, N_kv=N the
  // single query sits at N-1 and must see ALL keys (q_off=N-1). Without this
  // the decode query masks every key but n0=0 -> garbage.
  const int q_off = N_kv - M;

  // Causal whole-tile skip: if the smallest key index in this tile (n0)
  // exceeds the largest query absolute index (q_off+m0+TM_QK-1), every element
  // is masked. Write -INF and skip the d-reduction — ~half the tiles for a
  // square causal prefill (q_off=0).
  if (causal && n0 > q_off + m0 + (TM_QK - 1)) {
    const long sb = (long)head_q * (long)M * (long)N_kv;
#pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const int m = m0 + i;
      if (m >= M)
        continue;
#pragma unroll
      for (int j = 0; j < TN_QK; j++) {
        const int n = n0 + j;
        if (n < N_kv)
          scores[sb + (long)m * N_kv + n] = (half)(-INFINITY);
      }
    }
    return;
  }

  // Sliding-window whole-tile skip (mirror of the causal skip above): every
  // element is below the window when even the tile's LARGEST key index with
  // its SMALLEST query index satisfies n + W <= q_off + m (larger m only
  // masks more keys).
  if (causal && local_window > 0 &&
      (n0 + TN_QK - 1) + local_window <= q_off + m0) {
    const long sb = (long)head_q * (long)M * (long)N_kv;
#pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const int m = m0 + i;
      if (m >= M)
        continue;
#pragma unroll
      for (int j = 0; j < TN_QK; j++) {
        const int n = n0 + j;
        if (n < N_kv)
          scores[sb + (long)m * N_kv + n] = (half)(-INFINITY);
      }
    }
    return;
  }

  float acc[TM_QK][TN_QK];
#pragma unroll
  for (int i = 0; i < TM_QK; i++)
#pragma unroll
    for (int j = 0; j < TN_QK; j++)
      acc[i][j] = 0.0f;

  const sampler_t smp =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  const int d_tex_count = d >> 3;
  const int k_row_base = head_kv * S_max;

  for (int dt = 0; dt < d_tex_count; dt++) {
    half8 q_pack[TM_QK];
    half8 k_pack[TN_QK];
#pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const int m = m0 + i;
      if (m < M) {
        const long off = (long)m * HD_Q + (long)head_q * d + (long)dt * 8;
        q_pack[i] = vload8(0, Q + off);
      } else {
        q_pack[i] = (half8)((half)0.0h);
      }
    }
#pragma unroll
    for (int j = 0; j < TN_QK; j++) {
      const int n = n0 + j;
      const int ny = (n < N_kv) ? n : 0;
      const uint4 vv = read_imageui(K_img, smp, (int2)(dt, k_row_base + ny));
      const half8 hp = as_half8(vv);
      k_pack[j] = (n < N_kv) ? hp : (half8)((half)0.0h);
    }
#pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const float4 qlo = convert_float4(q_pack[i].s0123);
      const float4 qhi = convert_float4(q_pack[i].s4567);
#pragma unroll
      for (int j = 0; j < TN_QK; j++) {
        const float4 klo = convert_float4(k_pack[j].s0123);
        const float4 khi = convert_float4(k_pack[j].s4567);
        acc[i][j] += dot(qlo, klo) + dot(qhi, khi);
      }
    }
  }

  const long score_base = (long)head_q * (long)M * (long)N_kv;
#pragma unroll
  for (int i = 0; i < TM_QK; i++) {
    const int m = m0 + i;
    if (m >= M)
      continue;
#pragma unroll
    for (int j = 0; j < TN_QK; j++) {
      const int n = n0 + j;
      if (n >= N_kv)
        continue;
      float v = acc[i][j] * scale;
      if (softcap > 0.0f)
        v = softcap * tanh(v / softcap); // Gemma2-style score cap
      // Causal upper bound + sliding-window lower bound (flash formula:
      // key n is visible iff q_pos-W < n <= q_pos; W=0 means no window).
      if (causal && (n > q_off + m ||
                     (local_window > 0 && n + local_window <= q_off + m)))
        v = -INFINITY;
      scores[score_base + (long)m * N_kv + n] = (half)v;
    }
  }
}
#endif // TCA_BUFFER_ONLY (qk_matmul_f16_ohwi_img)

// =============================================================
// Row softmax (in-place): for each (h, m), softmax over N_kv axis.
//   One workgroup of LWS WIs per (h, m). Three local-memory
//   reductions: max, then exp + sum, then inverse-multiply.
// gws = (LWS, M, H), LWS = SOFTMAX_LWS.
// =============================================================
#ifndef SOFTMAX_LWS
#define SOFTMAX_LWS 64
#endif

__attribute__((reqd_work_group_size(SOFTMAX_LWS, 1, 1))) __kernel void
softmax_row_f16(__global half *scores, // [H, M, N_kv]
                const int M, const int N_kv) {
  const int tid = get_local_id(0);
  const int m = get_group_id(1);
  const int h = get_group_id(2);
  if (m >= M)
    return;

  __global half *row = scores + (long)h * (long)M * N_kv + (long)m * N_kv;

  __local float lscratch[SOFTMAX_LWS];

  // Pass 1: per-WI partial max.
  float pmax = -INFINITY;
  for (int n = tid; n < N_kv; n += SOFTMAX_LWS) {
    float v = (float)row[n];
    pmax = fmax(pmax, v);
  }
  lscratch[tid] = pmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = SOFTMAX_LWS / 2; s > 0; s >>= 1) {
    if (tid < s)
      lscratch[tid] = fmax(lscratch[tid], lscratch[tid + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float row_max = lscratch[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  // Pass 2: per-WI exp(x - max) + partial sum. Stash exp() back to row.
  float psum = 0.0f;
  for (int n = tid; n < N_kv; n += SOFTMAX_LWS) {
    float e = (row_max == -INFINITY) ? 0.0f : exp((float)row[n] - row_max);
    row[n] = (half)e;
    psum += e;
  }
  lscratch[tid] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = SOFTMAX_LWS / 2; s > 0; s >>= 1) {
    if (tid < s)
      lscratch[tid] += lscratch[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float row_sum = lscratch[0];
  const float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Pass 3: divide by sum (in-place).
  for (int n = tid; n < N_kv; n += SOFTMAX_LWS) {
    row[n] = (half)((float)row[n] * inv);
  }
}

// =============================================================
// SV matmul: O[m, h, x] = sum_n scores[h, m, n] * V[n, h_kv, x]
//   Mirror of qk_matmul: each WI computes a TM_SV x TD_SV tile of
//   (M, d) for fixed head_q. Reduction is over N_kv (1003 for our
//   1k-prefill workload, so larger than d but still small enough
//   that no online softmax is needed - softmax has already run).
//
// gws = (d / TD_SV, M / TM_SV, H)
// =============================================================
#ifndef TM_SV
#define TM_SV 4
#endif
#ifndef TD_SV
#define TD_SV 8
#endif

__kernel void
sv_matmul_f16(__global const half *scores, // [H, M, N_kv] fp16, post-softmax
              __global const half *V,      // [N_kv, HD_KV] fp16
              __global half *O,            // [M, HD_Q] fp16, row-major
              const int M, const int N_kv, const int d, const int HD_Q,
              const int HD_KV, const int gqa) {
  const int x0 = get_global_id(0) * TD_SV;
  const int m0 = get_global_id(1) * TM_SV;
  const int head_q = get_global_id(2);
  const int head_kv = head_q / gqa;

  if (m0 >= M || x0 >= d)
    return;

  float acc[TM_SV][TD_SV];
#pragma unroll
  for (int i = 0; i < TM_SV; i++)
#pragma unroll
    for (int j = 0; j < TD_SV; j++)
      acc[i][j] = 0.0f;

  const long score_base = (long)head_q * (long)M * (long)N_kv;

  // Reduction over N_kv. We re-read each (m, n) score for each (n, x)
  // V row, which is suboptimal but the score range is small (fp16) and
  // the bandwidth-amplification factor is bounded by TD_SV.
  for (int n = 0; n < N_kv; n++) {
    half s_col[TM_SV];
    half v_col[TD_SV];
#pragma unroll
    for (int i = 0; i < TM_SV; i++) {
      const int m = m0 + i;
      s_col[i] = (m < M) ? scores[score_base + (long)m * N_kv + n] : (half)0.0f;
    }
#pragma unroll
    for (int j = 0; j < TD_SV; j++) {
      const int x = x0 + j;
      v_col[j] = (x < d) ? V[(long)n * HD_KV + head_kv * d + x] : (half)0.0f;
    }
#pragma unroll
    for (int i = 0; i < TM_SV; i++) {
      const float sf = (float)s_col[i];
#pragma unroll
      for (int j = 0; j < TD_SV; j++)
        acc[i][j] += sf * (float)v_col[j];
    }
  }

#pragma unroll
  for (int i = 0; i < TM_SV; i++) {
    const int m = m0 + i;
    if (m >= M)
      continue;
#pragma unroll
    for (int j = 0; j < TD_SV; j++) {
      const int x = x0 + j;
      if (x >= d)
        continue;
      O[(long)m * HD_Q + head_q * d + x] = (half)acc[i][j];
    }
  }
}

// =============================================================
// §3.8 V OHWI-reversed variant of sv_matmul_f16.
// V is laid out as [H_kv, d, S_max] (per-head, per-d sequential
// over the S/cache axis) — the OHWI "reversed" form so the inner
// reduction over n reads V at stride-1 (cache-friendly) instead of
// stride HD_KV (concat layout).
//
// Element index: V[(long)head_kv * d * S_max + x * S_max + n]
//
// Trade-off: per-iteration we still read TD_SV halves with stride
// S_max (not coalesced within one n), but cross-n reads are
// sequential at stride 1 — the dominant pattern at large N_kv
// where the cache miss matters.
// =============================================================
__kernel void sv_matmul_f16_ohwi(
  __global const half *scores, // [H, M, N_kv] fp16, post-softmax
  __global const half *V,      // [H_kv, d, S_max] fp16 OHWI-reversed
  __global half *O,            // [M, HD_Q] fp16, row-major
  const int M, const int N_kv, const int d, const int HD_Q, const int S_max,
  const int gqa) {
  const int x0 = get_global_id(0) * TD_SV;
  const int m0 = get_global_id(1) * TM_SV;
  const int head_q = get_global_id(2);
  const int head_kv = head_q / gqa;

  if (m0 >= M || x0 >= d)
    return;

  float acc[TM_SV][TD_SV];
#pragma unroll
  for (int i = 0; i < TM_SV; i++)
#pragma unroll
    for (int j = 0; j < TD_SV; j++)
      acc[i][j] = 0.0f;

  const long score_base = (long)head_q * (long)M * (long)N_kv;
  const long v_head_base = (long)head_kv * (long)d * (long)S_max;

  for (int n = 0; n < N_kv; n++) {
    half s_col[TM_SV];
    half v_col[TD_SV];
#pragma unroll
    for (int i = 0; i < TM_SV; i++) {
      const int m = m0 + i;
      s_col[i] = (m < M) ? scores[score_base + (long)m * N_kv + n] : (half)0.0f;
    }
#pragma unroll
    for (int j = 0; j < TD_SV; j++) {
      const int x = x0 + j;
      v_col[j] = (x < d) ? V[v_head_base + (long)x * S_max + n] : (half)0.0f;
    }
#pragma unroll
    for (int i = 0; i < TM_SV; i++) {
      const float sf = (float)s_col[i];
#pragma unroll
      for (int j = 0; j < TD_SV; j++)
        acc[i][j] += sf * (float)v_col[j];
    }
  }

#pragma unroll
  for (int i = 0; i < TM_SV; i++) {
    const int m = m0 + i;
    if (m >= M)
      continue;
#pragma unroll
    for (int j = 0; j < TD_SV; j++) {
      const int x = x0 + j;
      if (x >= d)
        continue;
      O[(long)m * HD_Q + head_q * d + x] = (half)acc[i][j];
    }
  }
}

// =============================================================
// int8-KV variants (paper §3.7 int8 KV path).
// K/V cache is stored as signed int8 bytes; a per-(token, head)
// FP16 amax scale lifts the int8 values back to fp16 range.
// Same memory layout as the f16 kernels above:
//   K_i8[n, head_kv, x]   -> n*HD_KV + head_kv*d + x (1 byte each)
//   K_scale[n, head_kv]   -> n*num_heads_KV + head_kv (fp16)
// The scale is constant across the d-axis for a given (n, head_kv),
// so we multiply once per (n, head_kv) outside the inner d-loop.
// =============================================================

__kernel void
qk_matmul_f16_kvi8(__global const half *Q,       // [M, HD_Q] fp16
                   __global const char *K_i8,    // [N_kv, HD_KV] int8 (signed)
                   __global const half *K_scale, // [N_kv, num_heads_KV] fp16
                   __global half *scores,        // [H, M, N_kv] fp16
                   const int M, const int N_kv, const int d, const int HD_Q,
                   const int HD_KV, const int gqa, const int num_heads_KV,
                   const int causal, const float scale) {
  const int n0 = get_global_id(0) * TN_QK;
  const int m0 = get_global_id(1) * TM_QK;
  const int head_q = get_global_id(2);
  const int head_kv = head_q / gqa;

  if (m0 >= M || n0 >= N_kv)
    return;

  float acc[TM_QK][TN_QK];
#pragma unroll
  for (int i = 0; i < TM_QK; i++)
#pragma unroll
    for (int j = 0; j < TN_QK; j++)
      acc[i][j] = 0.0f;

  // Reduction over d. Q in fp16, K in int8; convert int8 -> fp32 at
  // load and accumulate in fp32 (matches CPU helper's precision).
  for (int x = 0; x < d; x++) {
    half q_col[TM_QK];
    float k_col[TN_QK];
#pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const int m = m0 + i;
      q_col[i] = (m < M) ? Q[(long)m * HD_Q + head_q * d + x] : (half)0.0f;
    }
#pragma unroll
    for (int j = 0; j < TN_QK; j++) {
      const int n = n0 + j;
      k_col[j] = (n < N_kv)
                   ? convert_float(K_i8[(long)n * HD_KV + head_kv * d + x])
                   : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < TM_QK; i++) {
      const float qf = (float)q_col[i];
#pragma unroll
      for (int j = 0; j < TN_QK; j++)
        acc[i][j] += qf * k_col[j];
    }
  }

  // Apply per-(n, head_kv) scale, the softmax scale, causal mask, and
  // write out. Scale is constant in d for each (n, head_kv) so it
  // factors out of the inner reduction.
  float ks[TN_QK];
#pragma unroll
  for (int j = 0; j < TN_QK; j++) {
    const int n = n0 + j;
    ks[j] =
      (n < N_kv) ? (float)K_scale[(long)n * num_heads_KV + head_kv] : 0.0f;
  }

  const long score_base = (long)head_q * (long)M * (long)N_kv;
#pragma unroll
  for (int i = 0; i < TM_QK; i++) {
    const int m = m0 + i;
    if (m >= M)
      continue;
#pragma unroll
    for (int j = 0; j < TN_QK; j++) {
      const int n = n0 + j;
      if (n >= N_kv)
        continue;
      float v = acc[i][j] * ks[j] * scale;
      if (causal && n > m)
        v = -INFINITY;
      scores[score_base + (long)m * N_kv + n] = (half)v;
    }
  }
}

__kernel void sv_matmul_f16_kvi8(
  __global const half *scores,  // [H, M, N_kv] fp16 post-softmax
  __global const char *V_i8,    // [N_kv, HD_KV] int8
  __global const half *V_scale, // [N_kv, num_heads_KV] fp16
  __global half *O,             // [M, HD_Q] fp16
  const int M, const int N_kv, const int d, const int HD_Q, const int HD_KV,
  const int gqa, const int num_heads_KV) {
  const int x0 = get_global_id(0) * TD_SV;
  const int m0 = get_global_id(1) * TM_SV;
  const int head_q = get_global_id(2);
  const int head_kv = head_q / gqa;

  if (m0 >= M || x0 >= d)
    return;

  float acc[TM_SV][TD_SV];
#pragma unroll
  for (int i = 0; i < TM_SV; i++)
#pragma unroll
    for (int j = 0; j < TD_SV; j++)
      acc[i][j] = 0.0f;

  const long score_base = (long)head_q * (long)M * (long)N_kv;

  // Reduction over N_kv. The V_scale is constant in d for a given
  // (n, head_kv) so we fold it into the score multiplier once per n.
  for (int n = 0; n < N_kv; n++) {
    const float vs = (float)V_scale[(long)n * num_heads_KV + head_kv];
    half s_col[TM_SV];
    float v_col[TD_SV];
#pragma unroll
    for (int i = 0; i < TM_SV; i++) {
      const int m = m0 + i;
      s_col[i] = (m < M) ? scores[score_base + (long)m * N_kv + n] : (half)0.0f;
    }
#pragma unroll
    for (int j = 0; j < TD_SV; j++) {
      const int x = x0 + j;
      v_col[j] =
        (x < d) ? convert_float(V_i8[(long)n * HD_KV + head_kv * d + x]) : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < TM_SV; i++) {
      const float sf = (float)s_col[i] * vs;
#pragma unroll
      for (int j = 0; j < TD_SV; j++)
        acc[i][j] += sf * v_col[j];
    }
  }

#pragma unroll
  for (int i = 0; i < TM_SV; i++) {
    const int m = m0 + i;
    if (m >= M)
      continue;
#pragma unroll
    for (int j = 0; j < TD_SV; j++) {
      const int x = x0 + j;
      if (x >= d)
        continue;
      O[(long)m * HD_Q + head_q * d + x] = (half)acc[i][j];
    }
  }
}

// =============================================================
// Packed (image2d_from_buffer) variants of qk_matmul / sv_matmul.
// Pattern reused from v8c FC kernel (87% Adreno 830 peak): view the
// fp16 buffer as RGBA UINT32 image2d (16 bytes = 8 halves per texel),
// read via read_imageui returning uint4, reinterpret as half8 with
// as_half8(). 8x fewer memory transactions than scalar half loads.
// d, HD_Q, HD_KV must be multiples of 8.
// Image-sampling kernels: compiled out with -DTCA_BUFFER_ONLY (Intel NEO).
// =============================================================
#ifndef TCA_BUFFER_ONLY

// Smaller tile (TM_IMG=2, TN_IMG=4 = 8 acc) to keep register pressure low.
// half8 staging × (TM_IMG+TN_IMG) = 6 half8 = 48 halves = 96 bytes plus
// the 8 float acc = 32 bytes. Fits comfortably in the per-WI register
// file even with private temporaries.
#ifndef TM_IMG
#define TM_IMG 2
#endif
#ifndef TN_IMG
#define TN_IMG 4
#endif

__kernel void qk_matmul_f16_img(
  __read_only image2d_t Q_img, // width=HD_Q/8 texels, height=M
  __read_only image2d_t K_img, // width=HD_KV/8 texels, height=N_kv
  __global half *scores,       // [H, M, N_kv] fp16, row-major
  const int M, const int N_kv, const int d, const int HD_Q, const int HD_KV,
  const int gqa, const int causal, const float scale) {
  const int n0 = get_global_id(0) * TN_IMG;
  const int m0 = get_global_id(1) * TM_IMG;
  const int head_q = get_global_id(2);
  const int head_kv = head_q / gqa;

  if (m0 >= M || n0 >= N_kv)
    return;

  float acc[TM_IMG][TN_IMG];
#pragma unroll
  for (int i = 0; i < TM_IMG; i++)
#pragma unroll
    for (int j = 0; j < TN_IMG; j++)
      acc[i][j] = 0.0f;

  const sampler_t smp =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  const int q_tx0 = (head_q * d) >> 3; // 1 texel = 8 halves
  const int k_tx0 = (head_kv * d) >> 3;
  const int d_tex = d >> 3;

  for (int xt = 0; xt < d_tex; xt++) {
    half8 q_pack[TM_IMG];
    half8 k_pack[TN_IMG];

#pragma unroll
    for (int i = 0; i < TM_IMG; i++) {
      const int m = m0 + i;
      const int my = (m < M) ? m : 0;
      const uint4 v = read_imageui(Q_img, smp, (int2)(q_tx0 + xt, my));
      const half8 hp = as_half8(v);
      q_pack[i] = (m < M) ? hp : (half8)((half)0.0h);
    }
#pragma unroll
    for (int j = 0; j < TN_IMG; j++) {
      const int n = n0 + j;
      const int ny = (n < N_kv) ? n : 0;
      const uint4 v = read_imageui(K_img, smp, (int2)(k_tx0 + xt, ny));
      const half8 hp = as_half8(v);
      k_pack[j] = (n < N_kv) ? hp : (half8)((half)0.0h);
    }

// Per (i, j): compute one half-precision dot via two half4 dot()
// builtins. The compiler maps each dot(float4,float4) to fma chains.
#pragma unroll
    for (int i = 0; i < TM_IMG; i++) {
      const float4 qlo = convert_float4(q_pack[i].s0123);
      const float4 qhi = convert_float4(q_pack[i].s4567);
#pragma unroll
      for (int j = 0; j < TN_IMG; j++) {
        const float4 klo = convert_float4(k_pack[j].s0123);
        const float4 khi = convert_float4(k_pack[j].s4567);
        acc[i][j] += dot(qlo, klo) + dot(qhi, khi);
      }
    }
  }

  // Apply scale + causal mask + write.
  const long score_base = (long)head_q * (long)M * (long)N_kv;
#pragma unroll
  for (int i = 0; i < TM_IMG; i++) {
    const int m = m0 + i;
    if (m >= M)
      continue;
#pragma unroll
    for (int j = 0; j < TN_IMG; j++) {
      const int n = n0 + j;
      if (n >= N_kv)
        continue;
      float v = acc[i][j] * scale;
      if (causal && n > m)
        v = -INFINITY;
      scores[score_base + (long)m * N_kv + n] = (half)v;
    }
  }
}

// SV with V packed as image2d. Score is small per-n (TM_SV scalars)
// and stays as a plain buffer; only V benefits from the texel pack.
// Each WI computes a TM_SV_IMG x 8 tile of (M, d) for fixed head_q.
#ifndef TM_SV_IMG
#define TM_SV_IMG 2
#endif
__kernel void sv_matmul_f16_img(
  __global const half *scores, // [H, M, N_kv] post-softmax fp16
  __read_only image2d_t V_img, // width=HD_KV/8 texels, height=N_kv
  __global half *O,            // [M, HD_Q] fp16
  const int M, const int N_kv, const int d, const int HD_Q, const int HD_KV,
  const int gqa) {
  const int x_tex = get_global_id(0); // 1 texel = 8 halves of d
  const int m0 = get_global_id(1) * TM_SV_IMG;
  const int head_q = get_global_id(2);
  const int head_kv = head_q / gqa;
  const int x0 = x_tex * 8;

  if (m0 >= M || x0 >= d)
    return;

  float8 acc[TM_SV_IMG];
#pragma unroll
  for (int i = 0; i < TM_SV_IMG; i++)
    acc[i] = (float8)(0.0f);

  const sampler_t smp =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  const long score_base = (long)head_q * (long)M * (long)N_kv;
  const int v_tx_x = ((head_kv * d) >> 3) + x_tex;

  // Reduction over N_kv. Per n: TM_SV_IMG score scalars + one 8-half V texel.
  for (int n = 0; n < N_kv; n++) {
    const uint4 vv = read_imageui(V_img, smp, (int2)(v_tx_x, n));
    const float8 v_pack = convert_float8(as_half8(vv));

#pragma unroll
    for (int i = 0; i < TM_SV_IMG; i++) {
      const int m = m0 + i;
      const float sf =
        (m < M) ? (float)scores[score_base + (long)m * N_kv + n] : 0.0f;
      acc[i] = mad((float8)sf, v_pack, acc[i]);
    }
  }

#pragma unroll
  for (int i = 0; i < TM_SV_IMG; i++) {
    const int m = m0 + i;
    if (m >= M)
      continue;
    const half8 oh = convert_half8(acc[i]);
    // Adreno's OpenCL compiler rejects subscript on vector types
    // (`oh[e]`); spill to a half[8] array and index that instead.
    half oh_arr[8];
    vstore8(oh, 0, oh_arr);
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const int x = x0 + e;
      if (x >= d)
        continue;
      O[(long)m * HD_Q + head_q * d + x] = oh_arr[e];
    }
  }
}

// =============================================================
// sv_matmul_f16_ohwi_img — paper §3.8 V OHWI-reversed via
// image2d_from_buffer. Each work-item computes one output element
// O[m, head_q*d + x] by reducing over n (cache-size axis) in chunks
// of 8. Both V and scores reads are stride-1 along the reduction
// axis (V is OHWI-reversed [H_kv, d, S_max], scores row-major
// [H_q, M, N_kv]). 1 V image read + 1 scores 8-half load per iter.
//
// V_image layout: width = S_max/8 texels (8 halves per texel),
//                 height = H_kv * d
//   texel(n_tex, head_kv*d + x) = V[head_kv, x, n_tex*8 .. n_tex*8+7]
//
// Workgroup tiled as (LWS_X=16 x's, LWS_Y=4 m's) so 4 WIs share each
// V row read within a wavefront → image-cache hit. Single accumulator
// per WI minimizes register pressure.
// =============================================================
__kernel void sv_matmul_f16_ohwi_img(
  __global const half *scores, // [H_q, M, N_kv] fp16, row-major
  __read_only image2d_t V_img, // see comment above
  __global half *O,            // [M, HD_Q] fp16, row-major
  const int M, const int N_kv, const int d, const int HD_Q, const int S_max,
  const int gqa, const int causal) {
  // Tiled over the output d/x axis: each WI computes TDX=8 consecutive
  // output channels x0..x0+7. The scores chunk depends only on (m, n_tex),
  // NOT on x, so it is loaded ONCE per n_tex and reused across all 8 x's —
  // cutting scores-buffer reads 8x vs the 1-output-per-WI version (the
  // scores row was previously re-read by all d WIs of a given m).
  const int x0 = get_global_id(0) * 8;
  const int m = get_global_id(1);
  const int head_q = get_global_id(2);
  if (m >= M || x0 >= d)
    return;
  const int head_kv = head_q / gqa;

  const sampler_t smp =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  const int v_row0 = head_kv * d + x0;
  const long score_base =
    (long)head_q * (long)M * (long)N_kv + (long)m * (long)N_kv;
  // Causal: scores[m][n]=0 for n > the query's ABSOLUTE position, so only the
  // first ceil((q_abs+1)/8) score chunks contribute. Query row m holds the last
  // M positions of the N_kv context -> q_abs = (N_kv-M)+m. For prefill M==N_kv
  // (q_off=0, the old (m>>3)+1); for decode M=1 the single query at N-1 must
  // sum ALL N_kv keys (else it sums only V[0..7] -> garbage). Paired with
  // qk_matmul_f16_ohwi_img's q_off causal mask. The cap is a work-skip over
  // scores the QK mask already zeroed, so a non-causal call simply keeps the
  // full range.
  int N_kv_tex = (N_kv + 7) >> 3;
  if (causal) {
    const int q_off = N_kv - M;
    const int N_kv_tex_causal = ((q_off + m) >> 3) + 1;
    if (N_kv_tex_causal < N_kv_tex)
      N_kv_tex = N_kv_tex_causal;
  }

  float acc[8];
#pragma unroll
  for (int t = 0; t < 8; t++)
    acc[t] = 0.0f;

  for (int n_tex = 0; n_tex < N_kv_tex; n_tex++) {
    const int n0 = n_tex * 8;
    half8 s_pack;
    if (n0 + 8 <= N_kv) {
      s_pack = vload8(0, scores + score_base + n0);
    } else {
      half tmp[8];
#pragma unroll
      for (int k = 0; k < 8; k++)
        tmp[k] = (n0 + k < N_kv) ? scores[score_base + n0 + k] : (half)0.0h;
      s_pack = vload8(0, tmp);
    }
    const float4 slo = convert_float4(s_pack.s0123);
    const float4 shi = convert_float4(s_pack.s4567);
#pragma unroll
    for (int t = 0; t < 8; t++) {
      const uint4 vv = read_imageui(V_img, smp, (int2)(n_tex, v_row0 + t));
      const half8 v_pack = as_half8(vv);
      acc[t] += dot(slo, convert_float4(v_pack.s0123)) +
                dot(shi, convert_float4(v_pack.s4567));
    }
  }

#pragma unroll
  for (int t = 0; t < 8; t++) {
    const int x = x0 + t;
    if (x < d)
      O[(long)m * HD_Q + head_q * d + x] = (half)acc[t];
  }
}

// M-tiled sv: each WI computes 2 adjacent query rows (m0, m0+1) for the
// same 8 output channels, loading each V texel ONCE and reusing it across both
// rows (the V image is independent of m). Halves the V re-fetch that bounds
// sv_matmul_f16_ohwi_img (it reloads V once per m). Causal: use the larger
// row's tile cap; the extra n for m0 have score 0 (softmax-masked) -> add 0.
__kernel void sv_matmul_f16_ohwi_img_tm2(__global const half *scores,
                                         __read_only image2d_t V_img,
                                         __global half *O, const int M,
                                         const int N_kv, const int d,
                                         const int HD_Q, const int S_max,
                                         const int gqa, const int causal) {
  const int x0 = get_global_id(0) * 8;
  const int m0 = get_global_id(1) * 2;
  const int head_q = get_global_id(2);
  if (m0 >= M || x0 >= d)
    return;
  const int m1 = m0 + 1;
  const int has1 = (m1 < M);
  const int head_kv = head_q / gqa;
  const sampler_t smp =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  const int v_row0 = head_kv * d + x0;
  const long sb0 = (long)head_q * (long)M * (long)N_kv + (long)m0 * (long)N_kv;
  const long sb1 = (long)head_q * (long)M * (long)N_kv + (long)m1 * (long)N_kv;
  // Causal cap on the query's ABSOLUTE position (q_off=N_kv-M); decode (M=1)
  // sits at N-1 and must sum ALL keys, not just V[0..7]. See the non-tm2
  // kernel; a non-causal call keeps the full range.
  int N_kv_tex = (N_kv + 7) >> 3;
  if (causal) {
    const int q_off = N_kv - M;
    const int cap_m = has1 ? m1 : m0;
    const int N_kv_tex_causal = ((q_off + cap_m) >> 3) + 1;
    if (N_kv_tex_causal < N_kv_tex)
      N_kv_tex = N_kv_tex_causal;
  }

  float acc0[8], acc1[8];
#pragma unroll
  for (int t = 0; t < 8; t++) {
    acc0[t] = 0.0f;
    acc1[t] = 0.0f;
  }

  for (int n_tex = 0; n_tex < N_kv_tex; n_tex++) {
    const int n0 = n_tex * 8;
    half8 s0, s1;
    if (n0 + 8 <= N_kv) {
      s0 = vload8(0, scores + sb0 + n0);
      s1 = has1 ? vload8(0, scores + sb1 + n0) : (half8)((half)0.0h);
    } else {
      half t0[8], t1[8];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        t0[k] = (n0 + k < N_kv) ? scores[sb0 + n0 + k] : (half)0.0h;
        t1[k] = (has1 && n0 + k < N_kv) ? scores[sb1 + n0 + k] : (half)0.0h;
      }
      s0 = vload8(0, t0);
      s1 = vload8(0, t1);
    }
    const float4 s0lo = convert_float4(s0.s0123);
    const float4 s0hi = convert_float4(s0.s4567);
    const float4 s1lo = convert_float4(s1.s0123);
    const float4 s1hi = convert_float4(s1.s4567);
#pragma unroll
    for (int t = 0; t < 8; t++) {
      const uint4 vv = read_imageui(V_img, smp, (int2)(n_tex, v_row0 + t));
      const half8 vp = as_half8(vv);
      const float4 vlo = convert_float4(vp.s0123);
      const float4 vhi = convert_float4(vp.s4567);
      acc0[t] += dot(s0lo, vlo) + dot(s0hi, vhi);
      acc1[t] += dot(s1lo, vlo) + dot(s1hi, vhi);
    }
  }
#pragma unroll
  for (int t = 0; t < 8; t++) {
    const int x = x0 + t;
    if (x < d) {
      O[(long)m0 * HD_Q + head_q * d + x] = (half)acc0[t];
      if (has1)
        O[(long)m1 * HD_Q + head_q * d + x] = (half)acc1[t];
    }
  }
}

// =============================================================
// FUSED single-kernel attention (paper §4.2 compute-bound regime).
// One workgroup per (head_q, query-row m) computes the WHOLE row in
// three in-kernel phases, reusing BOTH existing OHWI images and
// NEVER materializing the [H,M,N_kv] scores matrix to DRAM:
//   (A) QK   : scores[n] = scale * dot(Q[m,hq,:], K[hkv,n,:]) into LDS,
//              reading the K image (8 channels / texel) — same texels
//              as qk_matmul_f16_ohwi_img.
//   (B) soft : full-row fp32 softmax over the LDS scores (max, exp, sum;
//              normalize deferred to (C) as acc*inv).
//   (C) SV   : O[m,hq,x] = inv * sum_n p[n] * V[hkv,x,n], reading the
//              reversed-V image (8 keys / texel) — same texels as
//              sv_matmul_f16_ohwi_img, but scores come from LDS.
// Kills the 3-kernel path's ~128 MB/layer scores round-trip (qk write +
// softmax R/W + sv read) and the L2 thrash that caps it at large M, and
// collapses 3 enqueues -> 1. LDS: q_sh[d] + sc_sh[<=S_max] + red[LWS].
//
// K image : texel(dt,        head_kv*S_max + n) = K[head_kv, n, dt*8..+7]
// V image : texel(n>>3,      head_kv*d + x)     = V[head_kv, x, n0..n0+7]
// gws = (FUSED_ATTN_LWS, M, num_heads_Q), lws = (FUSED_ATTN_LWS, 1, 1)
// =============================================================
#ifndef FUSED_ATTN_LWS
#define FUSED_ATTN_LWS 64
#endif
#ifndef FUSED_ATTN_MAX_NKV
#define FUSED_ATTN_MAX_NKV 1024
#endif
#ifndef FUSED_ATTN_MAX_D
#define FUSED_ATTN_MAX_D 128
#endif

__attribute__((reqd_work_group_size(FUSED_ATTN_LWS, 1, 1))) __kernel void
fused_row_attention_f16_ohwi_img(
  __global const half *Q,      // [M, HD_Q] fp16, row-major (concat)
  __read_only image2d_t K_img, // OHWI [H_kv, S_max, d]
  __read_only image2d_t V_img, // reversed-dims [H_kv, d, S_max]
  __global half *O,            // [M, HD_Q] fp16, row-major (concat)
  const int M, const int N_kv, const int d, const int HD_Q, const int S_max,
  const int gqa, const int causal, const float scale) {
  const int lid = get_local_id(0);
  const int m = get_global_id(1);
  const int head_q = get_global_id(2);
  if (m >= M)
    return;
  const int head_kv = head_q / gqa;

  const sampler_t smp =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  __local float q_sh[FUSED_ATTN_MAX_D];
  __local float sc_sh[FUSED_ATTN_MAX_NKV];
  __local float red[FUSED_ATTN_LWS];

  // Causal: only keys n <= m contribute (matches the 3-kernel n>m -> -inf).
  const int n_last = causal ? min(N_kv - 1, m) : (N_kv - 1);
  const int n_tex_count = (n_last + 1 + 7) >> 3; // ceil((n_last+1)/8)

  // (0) cooperative load Q[m, head_q, :] into LDS (fp32 promotion).
  const long q_base = (long)m * (long)HD_Q + (long)head_q * (long)d;
  for (int x = lid; x < d; x += FUSED_ATTN_LWS)
    q_sh[x] = (float)Q[q_base + x];
  barrier(CLK_LOCAL_MEM_FENCE);

  // (A) QK: each WI fully reduces d for its keys n = lid, lid+LWS, ...
  const int d_tex = d >> 3;
  const int k_row_base = head_kv * S_max;
  for (int n = lid; n <= n_last; n += FUSED_ATTN_LWS) {
    float acc = 0.0f;
    for (int dt = 0; dt < d_tex; dt++) {
      const uint4 kv = read_imageui(K_img, smp, (int2)(dt, k_row_base + n));
      const half8 kp = as_half8(kv);
      const float8 q8 = vload8(dt, q_sh);
      acc += dot(q8.s0123, convert_float4(kp.s0123)) +
             dot(q8.s4567, convert_float4(kp.s4567));
    }
    sc_sh[n] = acc * scale;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // (B1) row max.
  float pmax = -INFINITY;
  for (int n = lid; n <= n_last; n += FUSED_ATTN_LWS)
    pmax = fmax(pmax, sc_sh[n]);
  red[lid] = pmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = FUSED_ATTN_LWS >> 1; off > 0; off >>= 1) {
    if (lid < off)
      red[lid] = fmax(red[lid], red[lid + off]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float rowmax = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  // (B2) exp + sum (store un-normalized p into sc_sh; normalize in (C)).
  float psum = 0.0f;
  for (int n = lid; n <= n_last; n += FUSED_ATTN_LWS) {
    const float e = exp(sc_sh[n] - rowmax);
    sc_sh[n] = e;
    psum += e;
  }
  red[lid] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = FUSED_ATTN_LWS >> 1; off > 0; off >>= 1) {
    if (lid < off)
      red[lid] += red[lid + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float inv = (red[0] > 0.0f) ? (1.0f / red[0]) : 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Zero the last texel's tail (keys in (n_last, n_tex_count*8)) so (C)
  // can vload8 the score row without a per-key causal mask.
  for (int n = n_last + 1 + lid; n < n_tex_count * 8; n += FUSED_ATTN_LWS)
    sc_sh[n] = 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);

  // (C) SV: each WI computes output channels x = lid, lid+LWS, ...
  //     reading the reversed-V image (8 keys per texel) and the LDS row.
  for (int x = lid; x < d; x += FUSED_ATTN_LWS) {
    const int v_row = head_kv * d + x;
    float acc = 0.0f;
    for (int nt = 0; nt < n_tex_count; nt++) {
      const uint4 vv = read_imageui(V_img, smp, (int2)(nt, v_row));
      const half8 vp = as_half8(vv);
      const float8 p8 = vload8(nt, sc_sh);
      acc += dot(p8.s0123, convert_float4(vp.s0123)) +
             dot(p8.s4567, convert_float4(vp.s4567));
    }
    O[(long)m * (long)HD_Q + (long)head_q * (long)d + x] = (half)(acc * inv);
  }
}

#endif // TCA_BUFFER_ONLY (image2d_from_buffer attention variants)

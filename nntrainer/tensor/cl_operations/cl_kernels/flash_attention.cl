// SPDX-License-Identifier: Apache-2.0
// Flash-attention-style single-kernel prefill / decode attention for
// Adreno (paper ML Drift §3.6 fusion + Dao et al. 2022 online softmax).
//
// REPLACES: the three-kernel two_conv_attention.cl path which is
// (a) slower than CPU NEON on Adreno 830 due to VGPR spill (each WI's
// TM_QK*TN_QK accumulator floods the register file), and (b) routes
// the full scores[H, M, N_kv] tensor through global memory between
// kernels, which is ~6.7 MB bandwidth per prefill that we don't need.
//
// PIPELINE (one kernel):
//   For each (head_q, query_row):
//     - online maximum / sum / output accumulator (head_dim FP32 reg)
//     - serial loop over K/V rows n=0..N_kv-1 (causal: n<=m):
//       * s = scale * dot(Q[m,head_q], K[n,head_kv])   (fp32 accum)
//       * m_new = max(m_i, s); alpha = exp(m_i - m_new); p = exp(s - m_new)
//       * l_i = alpha*l_i + p
//       * acc[:] = alpha*acc[:] + p*V[n,head_kv,:]
//       * m_i = m_new
//     - write O[query_row, head_q*d : (head_q+1)*d] = acc / l_i
//
// CORRECTNESS-FIRST DESIGN:
//   ONE work-item per (head_q, query_row). The full d=128 fp32 output
//   accumulator (512 B) + Q row (256 B fp16) live in private memory.
//   No LDS, no inter-WI reduction, no scores DRAM materialization — the
//   whole point (removing the [H,M,N_kv] global traffic) is achieved by
//   the single-WI serial form. Tiling/LDS cooperation is a follow-up.
//   gws = (num_heads_q * M,); lws chosen by host (small, e.g. 64).
//
// K-LAYOUT NOTE: the gpu_native NNTR_OHWI_IMG=0 path stores cache_k_svm
//   in OHWI form  K[head_kv * S_max * d + n * d + x]  (qk_matmul_f16_ohwi
//   layout, NOT the pure concat [N_kv, HD_KV]). V stays concat
//   V[n * HD_KV + head_kv * d + x] (sv_matmul_f16 layout). To feed the
//   EXACT SAME buffers as the 3-kernel _ohwi_cl fallback, this kernel
//   takes a k_stride param: if k_stride > 0, K is OHWI with that S_max
//   row-stride; if k_stride <= 0, K is pure concat (HD_KV stride). Q and
//   O are always concat. This keeps the flash path bit-comparable to the
//   baseline it replaces.
//
// QWEN3-0.6B SHAPES (for reference; kernel is parameterized):
//   M = step_size (prefill: input length; decode: 1)
//   N_kv = cache_to (running cache fill, <= MAX_SEQ_LEN)
//   d = head_dim = 128
//   num_heads_q = 16, num_heads_kv = 8 (GQA = 2)
//   HD_Q = num_heads_q * d = 2048
//   HD_KV = num_heads_kv * d = 1024

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef FLASH_BLOCK_KV
#define FLASH_BLOCK_KV 32
#endif

#ifndef FLASH_MAX_D
#define FLASH_MAX_D 128
#endif

// Single-WI-per-(head_q, query_row) flash attention prefill. Online
// softmax (Dao et al.) so scores are never materialized to global memory.
// Signature matches the host wrapper flash_attention_prefill_f16_cl.
__kernel void flash_attention_prefill_f16_skeleton(
  __global const half *Q, // [M, HD_Q] fp16, row-major (concat)
  __global const half *K, // OHWI [H_kv,S_max,d] or concat [N_kv,HD_KV]
  __global const half *V, // [N_kv, HD_KV] fp16, row-major (concat)
  __global half *O,       // [M, HD_Q] fp16, row-major (concat)
  const int M, const int N_kv,
  const int d,     // head_dim
  const int HD_Q,  // num_heads_q * d
  const int HD_KV, // num_heads_kv * d
  const int gqa,   // num_heads_q / num_heads_kv
  const int is_causal,
  const float scale, // 1 / sqrt(d), precomputed
  const int k_stride // >0: K OHWI S_max row-stride; <=0: concat
) {
  const int gid = get_global_id(0); // decodes to (head_q, query_row)
  const int head_q = gid / M;
  const int m = gid % M;
  if (m >= M)
    return;
  const int total = (HD_Q / d) * M; // num_heads_q * M
  if (gid >= total)
    return;

  const int head_kv = head_q / gqa;

  // K base offset for this (head_kv). OHWI: head_kv*S_max*d + n*d + x.
  // concat: n*HD_KV + head_kv*d + x. We fold the per-n stride into k_row.
  const long k_head_base = (k_stride > 0)
                             ? ((long)head_kv * (long)k_stride * (long)d)
                             : ((long)head_kv * (long)d);
  const long k_row_stride = (k_stride > 0) ? (long)d : (long)HD_KV;

  const long q_base = (long)m * HD_Q + (long)head_q * d;

  // Load this query row into private fp32 registers.
  float q_row[FLASH_MAX_D];
  for (int x = 0; x < d; ++x)
    q_row[x] = (float)Q[q_base + x];

  // Online-softmax state.
  float m_i = -INFINITY;
  float l_i = 0.0f;
  float acc[FLASH_MAX_D];
  for (int x = 0; x < d; ++x)
    acc[x] = 0.0f;

  // Causal: key n is masked when is_causal && n > m. So the valid range
  // is n in [0, n_last] where n_last = is_causal ? min(N_kv-1, m) : N_kv-1.
  const int n_last = is_causal ? min(N_kv - 1, m) : (N_kv - 1);

  for (int n = 0; n <= n_last; ++n) {
    const long k_base = k_head_base + (long)n * k_row_stride;
    const long v_base = (long)n * HD_KV + (long)head_kv * d;

    // s = scale * dot(Q[m,head_q], K[n,head_kv]) in fp32.
    float dot = 0.0f;
    for (int x = 0; x < d; ++x)
      dot += q_row[x] * (float)K[k_base + x];
#ifdef FLASH_FP16_SCORE
    // Match the 3-kernel baseline, which writes scores as fp16 before
    // softmax (qk_matmul_f16 stores (half)(acc*scale)). Truncating here
    // makes the flash path bit-comparable to that baseline.
    const float s = (float)((half)(scale * dot));
#else
    const float s = scale * dot;
#endif

    // Online softmax update (Dao et al.).
    const float m_new = fmax(m_i, s);
    const float alpha = exp(m_i - m_new); // m_i==-inf, m_new finite => 0
    const float p = exp(s - m_new);
    l_i = alpha * l_i + p;
    for (int x = 0; x < d; ++x)
      acc[x] = alpha * acc[x] + p * (float)V[v_base + x];
    m_i = m_new;
  }

  // Normalize and write out. l_i == 0 only when no key was attended
  // (e.g. causal row with N_kv==0); guard to avoid NaN.
  const float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
  const long o_base = (long)m * HD_Q + (long)head_q * d;
  for (int x = 0; x < d; ++x)
    O[o_base + x] = (half)(acc[x] * inv);
}

// ===========================================================================
// COOPERATIVE flash attention prefill — d-AXIS-TILED online softmax.
//
// Motivation (measured on Intel Arc 0x7d55): the naive 1-WI variant and the
// first "split-the-key-loop + tree-reduce" coop attempt BOTH keep a full
// private acc[d=128] + q_row[d=128] per work-item. clGetKernelWorkGroupInfo
// reported CL_KERNEL_PRIVATE_MEM_SIZE = 16384 B/WI for both => the compiler
// spills that to global scratch, and the kernel becomes scratch-bandwidth
// bound (8.2 s @ M=1024 vs the 3-kernel path's ~1.0 s). The key-split coop
// made it WORSE by stacking a 32 KB LDS acc reduction on top of the same
// 16 KB private spill (max_wg collapsed to 64, occupancy died).
//
// FIX: tile the head_dim across the work-group so NO work-item ever holds a
// full d-wide vector. One WORKGROUP per (head_q, query_row), LWS work-items:
//   - Q row -> LDS once (q_sh[d], cooperative load), reused for every key.
//   - acc[d] -> LDS (acc_sh[d]), shared online-softmax output accumulator;
//     WI `lid` owns the d-lanes  x = lid, lid+LWS, ...   (tiny private state).
//   - m_i, l_i -> LDS scalars (shared online-softmax running max / denom).
//   - For each key n (serial over the WG, all WIs in lockstep):
//       * each WI computes a PARTIAL dot over its d-lanes (q_sh[x]*K[..x]);
//       * tree-reduce the LWS partials in LDS -> scalar score s (red_sh[]);
//       * one online-softmax step: alpha=exp(m_i-m_new), p=exp(s-m_new);
//         each WI updates ITS acc lanes acc_sh[x]=alpha*acc_sh[x]+p*V[..x],
//         and l_i=alpha*l_i+p ; m_i=m_new (scalars updated by all, identical).
//   - Finally each WI writes O[.. x] = acc_sh[x]/l_i for its d-lanes.
//
// Private footprint per WI is now O(1) floats (a couple of scalars + a small
// strided loop index) => NO spill. LDS = q_sh[d] + acc_sh[d] + red_sh[LWS]
// + a few scalars = 128*4 + 128*4 + LWS*4 + 16 ≈ 1.3 KB (LWS=64). Fits both
// Intel (64 KB) and Adreno (32 KB) with huge occupancy headroom.
//
// Score-reduction tree is a portable log-step LDS reduction (NO subgroup-64
// assumption) => Adreno-portable. LWS must be a power of two and divide d
// reasonably (each WI owns ceil(d/LWS) lanes); LWS in {16,32,64,128}.
//
// Same signature / layout contract as the naive variant (K OHWI via k_stride,
// V concat, causal via n_last, FLASH_FP16_SCORE diag off by default).
// ===========================================================================
#ifndef FLASH_COOP_LWS
#define FLASH_COOP_LWS 64
#endif

// Keys processed per reduction phase (amortizes the log-step dot-reduction
// barriers over a tile of keys). LDS red_sh = BLOCK_KV*LWS*4 B
// (4*64*4 = 1 KB). Must keep total LDS <= 32 KB for Adreno. Default 4 is the
// measured Intel Arc sweet spot (BLOCK_KV 2-4 ~equal; 1 and >=8 slower).
#ifndef FLASH_COOP_BLOCK_KV
#define FLASH_COOP_BLOCK_KV 4
#endif

__attribute__((reqd_work_group_size(FLASH_COOP_LWS, 1, 1))) __kernel void
flash_attention_prefill_f16_coop(
  __global const half *Q, // [M, HD_Q] fp16, row-major (concat)
  __global const half *K, // OHWI [H_kv,S_max,d] or concat [N_kv,HD_KV]
  __global const half *V, // [N_kv, HD_KV] fp16, row-major (concat)
  __global half *O,       // [M, HD_Q] fp16, row-major (concat)
  const int M, const int N_kv,
  const int d,     // head_dim
  const int HD_Q,  // num_heads_q * d
  const int HD_KV, // num_heads_kv * d
  const int gqa,   // num_heads_q / num_heads_kv
  const int is_causal,
  const float scale, // 1 / sqrt(d), precomputed
  const int k_stride // >0: K OHWI S_max row-stride; <=0: concat
) {
  const int lid = get_local_id(0);
  const int grp = get_group_id(0); // decodes to (head_q, query_row)
  const int head_q = grp / M;
  const int m = grp % M;
  const int total_groups = (HD_Q / d) * M; // num_heads_q * M
  if (grp >= total_groups || m >= M)
    return;

  const int head_kv = head_q / gqa;

  const long k_head_base = (k_stride > 0)
                             ? ((long)head_kv * (long)k_stride * (long)d)
                             : ((long)head_kv * (long)d);
  const long k_row_stride = (k_stride > 0) ? (long)d : (long)HD_KV;
  const long q_base = (long)m * HD_Q + (long)head_q * d;

  // Shared per-(head_q,m) state in LDS. q_sh: query row; acc_sh: output
  // accumulator (WI lid owns disjoint d-lanes lid,lid+LWS,...). red_sh:
  // score-dot reduction scratch, sized [BLOCK_KV][LWS] so a whole key tile
  // reduces with ONE set of log-step barriers instead of one set per key.
  __local float q_sh[FLASH_MAX_D];
  __local float acc_sh[FLASH_MAX_D];
  __local float red_sh[FLASH_COOP_BLOCK_KV * FLASH_COOP_LWS];

  // m_i / l_i are kept PRIVATE in every WI and stay identical across the WG
  // (all WIs read the same reduced score) — this removes the per-key cross-WI
  // barrier the shared-scalar version needed. acc_sh lanes are WI-private
  // (disjoint), so consecutive keys need NO barrier between acc updates; the
  // only barriers are inside the dot reduction.
  float m_i = -INFINITY;
  float l_i = 0.0f;

  // Cooperative load of Q row + zero acc. WI lid owns d-lanes lid, lid+LWS...
  for (int x = lid; x < d; x += FLASH_COOP_LWS) {
    q_sh[x] = (float)Q[q_base + x];
    acc_sh[x] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const int n_last = is_causal ? min(N_kv - 1, m) : (N_kv - 1);

  // Key-blocked loop: process up to BLOCK_KV keys per reduction phase.
  for (int n0 = 0; n0 <= n_last; n0 += FLASH_COOP_BLOCK_KV) {
    const int nb = min(FLASH_COOP_BLOCK_KV, n_last - n0 + 1);

    // (1) Each WI computes its PARTIAL d-dot for every key in the tile and
    //     stages them into red_sh[j*LWS + lid].
    for (int j = 0; j < nb; ++j) {
      const long k_base = k_head_base + (long)(n0 + j) * k_row_stride;
      float part = 0.0f;
      for (int x = lid; x < d; x += FLASH_COOP_LWS)
        part += q_sh[x] * (float)K[k_base + x];
      red_sh[j * FLASH_COOP_LWS + lid] = part;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // (2) Tree-reduce ALL nb columns together: log2(LWS) barrier rounds for
    //     the whole tile (not per key). Each active WI folds nb partials.
    for (int off = FLASH_COOP_LWS >> 1; off > 0; off >>= 1) {
      if (lid < off)
        for (int j = 0; j < nb; ++j)
          red_sh[j * FLASH_COOP_LWS + lid] +=
            red_sh[j * FLASH_COOP_LWS + lid + off];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    // red_sh[j*LWS] now holds the full fp32 dot for key (n0+j).

    // (3) Serial online-softmax over the tile. No barrier between keys: each
    //     WI updates only its own acc_sh lanes and its private m_i/l_i.
    for (int j = 0; j < nb; ++j) {
#ifdef FLASH_FP16_SCORE
      const float s = (float)((half)(scale * red_sh[j * FLASH_COOP_LWS]));
#else
      const float s = scale * red_sh[j * FLASH_COOP_LWS];
#endif
      const float m_new = fmax(m_i, s);
      const float alpha = exp(m_i - m_new); // m_i=-inf, m_new finite => 0
      const float p = exp(s - m_new);
      const long v_base = (long)(n0 + j) * HD_KV + (long)head_kv * d;
      for (int x = lid; x < d; x += FLASH_COOP_LWS)
        acc_sh[x] = alpha * acc_sh[x] + p * (float)V[v_base + x];
      l_i = alpha * l_i + p;
      m_i = m_new;
    }
    // acc_sh lanes are WI-private; next tile's reduction barrier (step 1->2)
    // re-fences red_sh. A barrier here ensures the tile's acc writes are
    // settled before red_sh is overwritten by the next tile (red_sh and
    // acc_sh are distinct, but the staging write of step (1) must not race
    // a still-reading WI of step (3) — they read red_sh, step (1) writes it).
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Normalize and write O for this WI's d-lanes.
  const float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
  const long o_base = (long)m * HD_Q + (long)head_q * d;
  for (int x = lid; x < d; x += FLASH_COOP_LWS)
    O[o_base + x] = (half)(acc_sh[x] * inv);
}

// ===========================================================================
// VECTORIZED COOPERATIVE flash attention prefill — half8/half4 K/V loads.
//
// Same workgroup decomposition as flash_attention_prefill_f16_coop (one
// WG per (head_q, query_row), online softmax, acc[d] in LDS, BLOCK_KV blocked,
// causal n_last, K OHWI via k_stride / V concat) but with two changes that
// RAISE ARITHMETIC INTENSITY on Intel Arc (which cannot sample images, so the
// 3-kernel image dot4 path is unavailable — plain __global half* vload8/vload4
// coalesces just fine here):
//
//   (A) CONTIGUOUS d-lane ownership instead of strided. WI `lid` owns the
//       contiguous lane block [lid*VPL, lid*VPL+VPL) where VPL = d / LWS
//       (=2 for LWS64, 4 for LWS32, 8 for LWS16). Contiguous lanes let each
//       WI issue a SINGLE vload8/vload4/vload2 over its K/V/Q block instead
//       of VPL scalar half->float reads, cutting the load instruction count
//       by VPL and coalescing the global access.
//
//   (B) OPTIONAL LDS STAGING of each key's K[n,:] and V[n,:] row. All LWS WIs
//       of a WG attend the SAME keys, so each K/V row is otherwise re-read
//       LWS-strided from global by every WI. With FLASH_VEC_STAGE the WG
//       cooperatively vload8s K[n,:]/V[n,:] into LDS ONCE per key, then every
//       WI reads its VPL lanes from LDS — cutting global K/V traffic ~LWS x.
//       (LDS cost: 2*BLOCK_KV*d halfs = 2*4*128*2 = 2KB at defaults; stays
//       <=32KB for Adreno at LWS<=64, BLOCK_KV<=8.)
//
// REDUCTION-ORDER NOTE: the per-WI partial dot now sums a DIFFERENT (contig)
// set of d-lanes than the coop variant's strided set, so the fp32 tree-reduced
// score is a reassociation of its sum. fp32 add is non-associative => the score
// can differ by a few ULPs. This is FAR below the fp16 output / fp16-score
// granularity, so e2e greedy tokens are unchanged (verified) — the math is the
// same online-softmax merge, only the load layout changed.
//
// Q row, acc[d], m_i/l_i, the score tree-reduction, and the FLASH_FP16_SCORE
// diagnostic are all identical to it. d MUST be divisible by LWS (Qwen3 d=128,
// LWS in {16,32,64,128} all divide it). half8/half4/half2 picked from VPL.
// ===========================================================================
#ifndef FLASH_VEC_LWS
#define FLASH_VEC_LWS 64
#endif
#ifndef FLASH_VEC_BLOCK_KV
#define FLASH_VEC_BLOCK_KV 4
#endif
// FLASH_VEC_STAGE: 1 => stage K/V rows in LDS (lever B); 0 => vload direct
// from global (lever A only). Default OFF — measured on Intel Arc the LDS
// staging adds a per-tile barrier + LDS pressure that nets slower than direct
// vloads at d=128 (the K/V row is small and the texture/L2 already caches it).
#ifndef FLASH_VEC_STAGE
#define FLASH_VEC_STAGE 0
#endif

// VPL = vector lanes per WI = head_dim / LWS, made a COMPILE-TIME constant so
// the half2/half4/half8 vload + the float2/4/8 vector ops are true REGISTER
// vectors (no private-array spill — the main perf trap here; cf. the coop
// variant's priv=0).
// Qwen3 d=128: LWS 16->VPL8, 32->VPL4, 64->VPL2. (Host rejects LWS not | d.)
#ifndef FLASH_VEC_D
#define FLASH_VEC_D 128
#endif
#define FLASH_VEC_VPL (FLASH_VEC_D / FLASH_VEC_LWS)

#if FLASH_VEC_VPL == 8
#define FVHALF half8
#define FVFLOAT float8
#define FV_VLOAD(p, off) vload8((off), (p))
#define FV_VSTORE(v, off, p) vstore8((v), (off), (p))
#define FV_CVT_F(v) convert_float8(v)
#define FV_CVT_H(v) convert_half8(v)
#define FV_VLOAD_F(p, off) vload8((off), (p))
#define FV_VSTORE_F(v, off, p) vstore8((v), (off), (p))
#elif FLASH_VEC_VPL == 4
#define FVHALF half4
#define FVFLOAT float4
#define FV_VLOAD(p, off) vload4((off), (p))
#define FV_VSTORE(v, off, p) vstore4((v), (off), (p))
#define FV_CVT_F(v) convert_float4(v)
#define FV_CVT_H(v) convert_half4(v)
#define FV_VLOAD_F(p, off) vload4((off), (p))
#define FV_VSTORE_F(v, off, p) vstore4((v), (off), (p))
#elif FLASH_VEC_VPL == 2
#define FVHALF half2
#define FVFLOAT float2
#define FV_VLOAD(p, off) vload2((off), (p))
#define FV_VSTORE(v, off, p) vstore2((v), (off), (p))
#define FV_CVT_F(v) convert_float2(v)
#define FV_CVT_H(v) convert_half2(v)
#define FV_VLOAD_F(p, off) vload2((off), (p))
#define FV_VSTORE_F(v, off, p) vstore2((v), (off), (p))
#else
#error FLASH_VEC_VPL must be 2 4 or 8 set FLASH_VEC_LWS to half quarter eighth of d
#endif

// Horizontal fp32 sum of a FVFLOAT register (compile-time unrolled per VPL).
static inline float fv_hsum(FVFLOAT v) {
#if FLASH_VEC_VPL == 8
  return (v.s0 + v.s1) + (v.s2 + v.s3) + ((v.s4 + v.s5) + (v.s6 + v.s7));
#elif FLASH_VEC_VPL == 4
  return (v.s0 + v.s1) + (v.s2 + v.s3);
#else
  return v.s0 + v.s1;
#endif
}

__attribute__((reqd_work_group_size(FLASH_VEC_LWS, 1, 1))) __kernel void
flash_attention_prefill_f16_coop_vec(
  __global const half *Q, // [M, HD_Q] fp16, row-major (concat)
  __global const half *K, // OHWI [H_kv,S_max,d] or concat [N_kv,HD_KV]
  __global const half *V, // [N_kv, HD_KV] fp16, row-major (concat)
  __global half *O,       // [M, HD_Q] fp16, row-major (concat)
  const int M, const int N_kv,
  const int d,     // head_dim
  const int HD_Q,  // num_heads_q * d
  const int HD_KV, // num_heads_kv * d
  const int gqa,   // num_heads_q / num_heads_kv
  const int is_causal,
  const float scale, // 1 / sqrt(d), precomputed
  const int k_stride // >0: K OHWI S_max row-stride; <=0: concat
) {
  const int lid = get_local_id(0);
  const int grp = get_group_id(0); // decodes to (head_q, query_row)
  const int head_q = grp / M;
  const int m = grp % M;
  const int total_groups = (HD_Q / d) * M; // num_heads_q * M
  if (grp >= total_groups || m >= M)
    return;

  const int head_kv = head_q / gqa;

  const long k_head_base = (k_stride > 0)
                             ? ((long)head_kv * (long)k_stride * (long)d)
                             : ((long)head_kv * (long)d);
  const long k_row_stride = (k_stride > 0) ? (long)d : (long)HD_KV;
  const long q_base = (long)m * HD_Q + (long)head_q * d;

  // VPL contiguous lanes per WI (compile-time). WI lid owns lanes
  // [lane0, lane0+VPL). q (this WI's block) lives in a REGISTER vector; acc
  // (this WI's block) lives in a REGISTER vector across the whole key loop —
  // no acc_sh round-trip per key, no private array => priv stays ~0.
  const int VPL = FLASH_VEC_VPL;
  const int lane0 = lid * VPL;

  // LDS only for the cross-WI score reduction (red_sh). q/acc are registers.
  __local float red_sh[FLASH_VEC_BLOCK_KV * FLASH_VEC_LWS];
#if FLASH_VEC_STAGE
  // Optional staged K/V tile in LDS, half-vector aligned. [BLOCK_KV][d] each.
  __local half k_sh[FLASH_VEC_BLOCK_KV * FLASH_VEC_D];
  __local half v_sh[FLASH_VEC_BLOCK_KV * FLASH_VEC_D];
#endif

  float m_i = -INFINITY;
  float l_i = 0.0f;

  // Load this WI's contiguous Q block into a register vector (single vloadN,
  // half-vector -> float-vector promotion).
  const FVFLOAT q_reg = FV_CVT_F(FV_VLOAD(Q, (q_base + lane0) / VPL));
  // acc register vector, zero-initialized.
  FVFLOAT acc_reg = (FVFLOAT)(0.0f);

  const int n_last = is_causal ? min(N_kv - 1, m) : (N_kv - 1);

  for (int n0 = 0; n0 <= n_last; n0 += FLASH_VEC_BLOCK_KV) {
    const int nb = min(FLASH_VEC_BLOCK_KV, n_last - n0 + 1);

#if FLASH_VEC_STAGE
    // (0) Cooperatively stage this key tile's K/V rows into LDS, ONCE per WG.
    //     Each WI vstoreN's its own contiguous VPL block of each of the nb
    //     rows.
    for (int j = 0; j < nb; ++j) {
      const long k_base = k_head_base + (long)(n0 + j) * k_row_stride;
      const long v_base = (long)(n0 + j) * HD_KV + (long)head_kv * d;
      const FVHALF kblk = FV_VLOAD(K, (k_base + lane0) / VPL);
      const FVHALF vblk = FV_VLOAD(V, (v_base + lane0) / VPL);
      FV_VSTORE(kblk, (j * FLASH_VEC_D + lane0) / VPL, k_sh);
      FV_VSTORE(vblk, (j * FLASH_VEC_D + lane0) / VPL, v_sh);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif

    // (1) Each WI computes its PARTIAL d-dot for every key in the tile from a
    //     SINGLE vloadN of K (register vector FMA), stages into red_sh.
    for (int j = 0; j < nb; ++j) {
#if FLASH_VEC_STAGE
      const FVFLOAT k_reg =
        FV_CVT_F(FV_VLOAD(k_sh, (j * FLASH_VEC_D + lane0) / VPL));
#else
      const long k_base = k_head_base + (long)(n0 + j) * k_row_stride;
      const FVFLOAT k_reg = FV_CVT_F(FV_VLOAD(K, (k_base + lane0) / VPL));
#endif
      red_sh[j * FLASH_VEC_LWS + lid] = fv_hsum(q_reg * k_reg);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // (2) Tree-reduce ALL nb columns together (log2(LWS) barrier rounds).
    for (int off = FLASH_VEC_LWS >> 1; off > 0; off >>= 1) {
      if (lid < off)
        for (int j = 0; j < nb; ++j)
          red_sh[j * FLASH_VEC_LWS + lid] +=
            red_sh[j * FLASH_VEC_LWS + lid + off];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // (3) Serial online-softmax over the tile. Each WI updates only its own
    //     acc register vector (single vloadN of V) and its private m_i/l_i.
    for (int j = 0; j < nb; ++j) {
#ifdef FLASH_FP16_SCORE
      const float s = (float)((half)(scale * red_sh[j * FLASH_VEC_LWS]));
#else
      const float s = scale * red_sh[j * FLASH_VEC_LWS];
#endif
      const float m_new = fmax(m_i, s);
      const float alpha = exp(m_i - m_new);
      const float p = exp(s - m_new);
#if FLASH_VEC_STAGE
      const FVFLOAT v_reg =
        FV_CVT_F(FV_VLOAD(v_sh, (j * FLASH_VEC_D + lane0) / VPL));
#else
      const long v_base = (long)(n0 + j) * HD_KV + (long)head_kv * d;
      const FVFLOAT v_reg = FV_CVT_F(FV_VLOAD(V, (v_base + lane0) / VPL));
#endif
      acc_reg = alpha * acc_reg + p * v_reg;
      l_i = alpha * l_i + p;
      m_i = m_new;
    }
    // red_sh is overwritten by the next tile's step (1); fence before reuse.
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Normalize and write this WI's contiguous d-lanes (single vstoreN).
  const float inv = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
  const long o_base = (long)m * HD_Q + (long)head_q * d;
  const FVHALF o_reg = FV_CVT_H(acc_reg * inv);
  FV_VSTORE(o_reg, (o_base + lane0) / VPL, O);
}

// ===========================================================================
// BLOCK-Q vectorized flash prefill — reuse K/V across FBQ_TM query rows.
// Same d-axis WG cooperation as flash_attention_prefill_f16_coop_vec (WI lid
// owns contiguous lanes [lid*VPL, lid*VPL+VPL), VPL=d/LWS, vector vloads, fp32
// tree-reduced score, online softmax) — but ONE workgroup owns a TILE of
// FBQ_TM query rows of ONE head_q. Each K[n]/V[n] is loaded ONCE per WG and
// FMA'd into ALL FBQ_TM rows => cuts global K+V traffic ~FBQ_TM x (the measured
// Intel bottleneck: K/V are memory-bound, re-read ~gqa*(M-n) times by the
// 1-row-per-WG vec kernel). Each row keeps its OWN m_i,l_i,acc and its OWN
// causal cutoff (key n contributes only to rows m>=n). Reuses FLASH_VEC_LWS/
// _BLOCK_KV/_D and all FV_* macros above (same translation unit).
// NOTE register budget: acc+q = 2*FBQ_TM*VPL floats/WI. LWS=32/half4/TM4 = 56
// (safe); LWS=16/half8/TM4 = 96 (SPILLS on Intel — use LWS>=32 with TM>=4).
// ===========================================================================
#ifndef FBQ_TM
#define FBQ_TM 4
#endif

#ifdef FBQ_SG
// Subgroup-reduce variant: LWS == subgroup size, so the cross-lane d-dot
// reduction is a single sub_group_reduce_add per (key,row) — NO red_sh LDS, NO
// barriers (vs the log-step tree below). Gated by NNTR_FLASH_SG. Picks the
// platform subgroup attribute (Intel reqd size = LWS; Adreno qcom "half" = 64,
// so the host MUST dispatch LWS=64 there). Mirrors rmsnorm.cl.
#if defined(cl_intel_required_subgroup_size)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
__attribute__((intel_reqd_sub_group_size(FLASH_VEC_LWS)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
__attribute__((qcom_reqd_sub_group_size("half")))
#else
__attribute__((reqd_work_group_size(FLASH_VEC_LWS, 1, 1)))
#endif
#else
__attribute__((reqd_work_group_size(FLASH_VEC_LWS, 1, 1)))
#endif
__kernel void
flash_attention_prefill_f16_blockq(
  __global const half *Q, // [M, HD_Q]
  __global const half *K, // OHWI [H_kv,S_max,d] (k_stride>0) or concat
  __global const half *V, // [N_kv, HD_KV]
  __global half *O,       // [M, HD_Q]
  const int M, const int N_kv, const int d, const int HD_Q, const int HD_KV,
  const int gqa, const int is_causal, const float scale, const int k_stride,
  const float softcap, const int local_window,
  // [kv-window-ring] >0: K/V physical row = n % ring_cap
  const int ring_cap) {
  const int lid = get_local_id(0);
  const int grp = get_group_id(0); // -> (head_q, row-tile)
  const int n_row_tiles = (M + FBQ_TM - 1) / FBQ_TM;
  const int head_q = grp / n_row_tiles;
  const int tile = grp % n_row_tiles;
  const int m0 = tile * FBQ_TM;
  const int total_groups = (HD_Q / d) * n_row_tiles;
  if (grp >= total_groups || m0 >= M)
    return;

  const int head_kv = head_q / gqa;
  const long k_head_base = (k_stride > 0)
                             ? ((long)head_kv * (long)k_stride * (long)d)
                             : ((long)head_kv * (long)d);
  const long k_row_stride = (k_stride > 0) ? (long)d : (long)HD_KV;

  const int VPL = FLASH_VEC_VPL;
  const int lane0 = lid * VPL;

#ifndef FBQ_SG
  __local float red_sh[FLASH_VEC_BLOCK_KV * FBQ_TM * FLASH_VEC_LWS];
#endif

  FVFLOAT q_reg[FBQ_TM];
  FVFLOAT acc_reg[FBQ_TM];
  float m_i[FBQ_TM];
  float l_i[FBQ_TM];
  int row_valid[FBQ_TM];

#pragma unroll
  for (int r = 0; r < FBQ_TM; ++r) {
    const int m = m0 + r;
    row_valid[r] = (m < M) ? 1 : 0;
    m_i[r] = -INFINITY;
    l_i[r] = 0.0f;
    acc_reg[r] = (FVFLOAT)(0.0f);
    const long q_base = (long)(row_valid[r] ? m : 0) * HD_Q + (long)head_q * d;
    q_reg[r] = FV_CVT_F(FV_VLOAD(Q, (q_base + lane0) / VPL));
  }

  // Decode/chunked: the M query rows are the LAST M positions of the N_kv
  // context (cache_from = N_kv - M), so query row r maps to ABSOLUTE position
  // (N_kv - M) + (m0 + r). Causal/window masks must compare keys against the
  // absolute query position, not the local row index. Prefill big-step has
  // N_kv == M (q_pos_off = 0), so this is a no-op there.
  const int q_pos_off = N_kv - M;
  // Largest causal key any VALID row in this tile attends (absolute pos).
  const int last_row =
    (((m0 + FBQ_TM - 1) < M) ? (m0 + FBQ_TM - 1) : (M - 1)) + q_pos_off;
  const int n_last = is_causal ? min(N_kv - 1, last_row) : (N_kv - 1);
  // [window-skip] With a sliding window, keys below EVERY row's window bound
  // contribute to no row in this tile: row r's low bound is
  // (m0+q_pos_off+r) - local_window + 1 and only GROWS with r, so the tile
  // union's low bound is row 0's. Start the key walk there instead of 0 --
  // the flash-decode kernel already clips this way (win_start), but the
  // prefill walk did not, so every sliding layer ran O(M*N) instead of
  // O(M*(W+TM)) (sliding layers with W=1024 waste ~8x key visits at 16K,
  // worse at 32K). Bit-identical by construction: every
  // skipped key was already masked (`continue`) for every row of this tile,
  // contributing exactly nothing to acc/l/m.
  const int n_lo = (is_causal && local_window > 0)
                     ? max(0, m0 + q_pos_off - local_window + 1)
                     : 0;

#ifdef FBQ_SG
  // Subgroup-reduce path: one key at a time, full d-dot = sub_group_reduce_add
  // of each WI's fv_hsum partial (no LDS staging, no barriers). The reduce is
  // called uniformly across lanes (m0/n/M are WG-uniform, r is the loop var),
  // so the per-row causal `continue` does not break subgroup uniformity.
  for (int n = n_lo; n <= n_last; ++n) {
    // [kv-window-ring] physical cache row = n % ring_cap (ring_cap<=0: linear).
    const long pn = (ring_cap > 0) ? (long)(n % ring_cap) : (long)n;
    const long k_base = k_head_base + pn * k_row_stride;
    const FVFLOAT k_reg = FV_CVT_F(FV_VLOAD(K, (k_base + lane0) / VPL));
    float sdot[FBQ_TM];
#pragma unroll
    for (int r = 0; r < FBQ_TM; ++r)
      sdot[r] = sub_group_reduce_add(fv_hsum(q_reg[r] * k_reg));
    const long v_base = pn * HD_KV + (long)head_kv * d;
    const FVFLOAT v_reg = FV_CVT_F(FV_VLOAD(V, (v_base + lane0) / VPL));
#pragma unroll
    for (int r = 0; r < FBQ_TM; ++r) {
      const int m = m0 + r + q_pos_off; // ABSOLUTE query position (decode-safe)
      // Causal mask (n > m) + Gemma4 sliding-window mask (n + W <= m, i.e.
      // key n is older than the window of W keys ending at m). local_window<=0
      // disables the window (full causal attention).
      if (!row_valid[r] ||
          (is_causal && (n > m || (local_window > 0 && n + local_window <= m))))
        continue;
#ifdef FLASH_FP16_SCORE
      float s = (float)((half)(scale * sdot[r]));
#else
      float s = scale * sdot[r];
#endif
      // Gemma2 attention logit soft-cap: s = softcap * tanh(s / softcap).
      if (softcap > 0.0f)
        s = softcap * tanh(s / softcap);
      const float m_new = fmax(m_i[r], s);
      const float alpha = exp(m_i[r] - m_new);
      const float p = exp(s - m_new);
      acc_reg[r] = alpha * acc_reg[r] + p * v_reg;
      l_i[r] = alpha * l_i[r] + p;
      m_i[r] = m_new;
    }
  }
#else
  for (int n0 = n_lo; n0 <= n_last; n0 += FLASH_VEC_BLOCK_KV) {
    const int nb = min(FLASH_VEC_BLOCK_KV, n_last - n0 + 1);

    // (1) Load K[n] ONCE per key; partial d-dot for ALL TM rows -> red_sh.
    for (int j = 0; j < nb; ++j) {
      // [kv-window-ring] physical cache row = n % ring_cap.
      const long pnj =
          (ring_cap > 0) ? (long)((n0 + j) % ring_cap) : (long)(n0 + j);
      const long k_base = k_head_base + pnj * k_row_stride;
      const FVFLOAT k_reg = FV_CVT_F(FV_VLOAD(K, (k_base + lane0) / VPL));
#pragma unroll
      for (int r = 0; r < FBQ_TM; ++r)
        red_sh[(j * FBQ_TM + r) * FLASH_VEC_LWS + lid] =
          fv_hsum(q_reg[r] * k_reg);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // (2) Tree-reduce all nb*TM columns (log2(LWS) rounds).
    for (int off = FLASH_VEC_LWS >> 1; off > 0; off >>= 1) {
      if (lid < off)
        for (int c = 0; c < nb * FBQ_TM; ++c)
          red_sh[c * FLASH_VEC_LWS + lid] +=
            red_sh[c * FLASH_VEC_LWS + lid + off];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    // (3) Online-softmax. V[n] loaded ONCE per key, applied to all TM rows.
    for (int j = 0; j < nb; ++j) {
      const int n = n0 + j;
      const long pn = (ring_cap > 0) ? (long)(n % ring_cap) : (long)n;
      const long v_base = pn * HD_KV + (long)head_kv * d;
      const FVFLOAT v_reg = FV_CVT_F(FV_VLOAD(V, (v_base + lane0) / VPL));
#pragma unroll
      for (int r = 0; r < FBQ_TM; ++r) {
        const int m =
          m0 + r + q_pos_off; // ABSOLUTE query position (decode-safe)
        // PER-ROW causal mask (n > m) + Gemma4 sliding-window mask
        // (n + W <= m). local_window<=0 disables the window.
        if (!row_valid[r] || (is_causal && (n > m || (local_window > 0 &&
                                                      n + local_window <= m))))
          continue;
#ifdef FLASH_FP16_SCORE
        float s =
          (float)((half)(scale * red_sh[(j * FBQ_TM + r) * FLASH_VEC_LWS]));
#else
        float s = scale * red_sh[(j * FBQ_TM + r) * FLASH_VEC_LWS];
#endif
        // Gemma2 attention logit soft-cap: s = softcap * tanh(s / softcap).
        if (softcap > 0.0f)
          s = softcap * tanh(s / softcap);
        const float m_new = fmax(m_i[r], s);
        const float alpha = exp(m_i[r] - m_new); // PER-ROW rescale
        const float p = exp(s - m_new);
        acc_reg[r] = alpha * acc_reg[r] + p * v_reg;
        l_i[r] = alpha * l_i[r] + p;
        m_i[r] = m_new;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE); // protect red_sh before next tile's step (1)
  }
#endif // FBQ_SG

// Normalize + write each valid row (single vstoreN per row).
#pragma unroll
  for (int r = 0; r < FBQ_TM; ++r) {
    if (!row_valid[r])
      continue;
    const float inv = (l_i[r] > 0.0f) ? (1.0f / l_i[r]) : 0.0f;
    const long o_base = (long)(m0 + r) * HD_Q + (long)head_q * d;
    const FVHALF o_reg = FV_CVT_H(acc_reg[r] * inv);
    FV_VSTORE(o_reg, (o_base + lane0) / VPL, O);
  }
}

// ===========================================================================
// FLASH-DECODING (split-KV) for M=1 decode. The decode query is a single row,
// so blockq/coop_vec only spawn num_heads_q workgroups (the EU array starves).
// Here the KV axis is split into n_chunks: gws = num_heads_q * n_chunks groups,
// each running online softmax over its KV chunk and writing an UNNORMALIZED
// partial (acc[d] fp32 + running max m + denom l). flash_decode_reduce then
// combines the n_chunks partials per head. Mirrors vLLM / llama.cpp
// flash-decoding. M=1 query is row 0; the query is the LAST position so causal
// needs no mask (every key 0..N_kv-1 is valid); Gemma4 sliding window keeps
// only keys [N_kv-W, N_kv). Reuses FLASH_VEC_* (VPL = d / LWS half vloads).
__kernel void flash_decode_partial(
  __global const half *Q,   // [1, HD_Q]
  __global const half *K,   // OHWI [H_kv,S_max,d] or concat [N_kv,HD_KV]
  __global const half *V,   // [N_kv, HD_KV]
  __global float *part_acc, // [H_q][n_chunks][d] fp32 (unnormalized)
  __global float *part_ml,  // [H_q][n_chunks][2] fp32 (m, l)
  const int N_kv, const int d, const int HD_Q, const int HD_KV, const int gqa,
  const float scale, const int k_stride, const int local_window,
  const int chunk_kv, const int n_chunks,
  // [kv-window-ring] >0: physical row = n % ring_cap
  const int ring_cap) {
  const int lid = get_local_id(0);
  const int grp = get_group_id(0); // -> (head_q, chunk)
  const int head_q = grp / n_chunks;
  const int chunk = grp % n_chunks;
  if (head_q >= (HD_Q / d))
    return;
  const int head_kv = head_q / gqa;
  const long k_head_base = (k_stride > 0)
                             ? ((long)head_kv * (long)k_stride * (long)d)
                             : ((long)head_kv * (long)d);
  const long k_row_stride = (k_stride > 0) ? (long)d : (long)HD_KV;
  const long q_base = (long)head_q * d; // M=1: query row 0

  const int VPL = FLASH_VEC_VPL;
  const int lane0 = lid * VPL;
  const FVFLOAT q_reg = FV_CVT_F(FV_VLOAD(Q, (q_base + lane0) / VPL));
  FVFLOAT acc_reg = (FVFLOAT)(0.0f);
  float m_i = -INFINITY, l_i = 0.0f;

  // This chunk's KV range, clipped to the sliding window low bound.
  int n0 = chunk * chunk_kv;
  int n1 = min(n0 + chunk_kv, N_kv);
  const int win_start =
    (local_window > 0 && local_window < N_kv) ? (N_kv - local_window) : 0;
  if (n0 < win_start)
    n0 = win_start;

  __local float red_sh[FLASH_VEC_BLOCK_KV * FLASH_VEC_LWS];
  for (int nb0 = n0; nb0 < n1; nb0 += FLASH_VEC_BLOCK_KV) {
    const int nb = min(FLASH_VEC_BLOCK_KV, n1 - nb0);
    for (int j = 0; j < nb; ++j) {
      const long pnj =
          (ring_cap > 0) ? (long)((nb0 + j) % ring_cap) : (long)(nb0 + j);
      const long k_base = k_head_base + pnj * k_row_stride;
      const FVFLOAT k_reg = FV_CVT_F(FV_VLOAD(K, (k_base + lane0) / VPL));
      red_sh[j * FLASH_VEC_LWS + lid] = fv_hsum(q_reg * k_reg);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int off = FLASH_VEC_LWS >> 1; off > 0; off >>= 1) {
      if (lid < off)
        for (int j = 0; j < nb; ++j)
          red_sh[j * FLASH_VEC_LWS + lid] +=
            red_sh[j * FLASH_VEC_LWS + lid + off];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int j = 0; j < nb; ++j) {
      const float s = scale * red_sh[j * FLASH_VEC_LWS];
      const float m_new = fmax(m_i, s);
      const float alpha = exp(m_i - m_new);
      const float p = exp(s - m_new);
      const long pnj2 =
          (ring_cap > 0) ? (long)((nb0 + j) % ring_cap) : (long)(nb0 + j);
      const long v_base = pnj2 * HD_KV + (long)head_kv * d;
      const FVFLOAT v_reg = FV_CVT_F(FV_VLOAD(V, (v_base + lane0) / VPL));
      acc_reg = alpha * acc_reg + p * v_reg;
      l_i = alpha * l_i + p;
      m_i = m_new;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // Unnormalized partial. l_i==0 => empty chunk (fully window-masked).
  const long pa = ((long)head_q * n_chunks + chunk) * d;
  FV_VSTORE_F(acc_reg, (pa + lane0) / VPL, part_acc);
  if (lid == 0) {
    const long pm = ((long)head_q * n_chunks + chunk) * 2;
    part_ml[pm + 0] = (l_i > 0.0f) ? m_i : -INFINITY;
    part_ml[pm + 1] = l_i;
  }
}

// Flash-decoding REDUCE: combine the n_chunks partials per head -> O[1, HD_Q].
__kernel void
flash_decode_reduce(__global const float *part_acc, // [H_q][n_chunks][d]
                    __global const float *part_ml,  // [H_q][n_chunks][2]
                    __global half *O,               // [1, HD_Q]
                    const int d, const int HD_Q, const int n_chunks) {
  const int lid = get_local_id(0);
  const int head_q = get_group_id(0);
  if (head_q >= (HD_Q / d))
    return;
  const int VPL = FLASH_VEC_VPL;
  const int lane0 = lid * VPL;

  float m_g = -INFINITY;
  for (int c = 0; c < n_chunks; ++c)
    m_g = fmax(m_g, part_ml[((long)head_q * n_chunks + c) * 2 + 0]);

  FVFLOAT acc_g = (FVFLOAT)(0.0f);
  float l_g = 0.0f;
  for (int c = 0; c < n_chunks; ++c) {
    const float m_c = part_ml[((long)head_q * n_chunks + c) * 2 + 0];
    const float l_c = part_ml[((long)head_q * n_chunks + c) * 2 + 1];
    if (l_c <= 0.0f)
      continue; // empty (window-masked) chunk
    const float w = exp(m_c - m_g);
    const long pa = ((long)head_q * n_chunks + c) * d;
    const FVFLOAT acc_c = FV_VLOAD_F(part_acc, (pa + lane0) / VPL);
    acc_g += w * acc_c;
    l_g += w * l_c;
  }
  const float inv = (l_g > 0.0f) ? (1.0f / l_g) : 0.0f;
  const long o_base = (long)head_q * d;
  FV_VSTORE(FV_CVT_H(acc_g * inv), (o_base + lane0) / VPL, O);
}

// ===========================================================================
// XMX (DPAS) flash prefill (#r30-q4) — the full-attention long-N term.
// One SUBGROUP (16 lanes, one per WG) owns 8 query rows of one head_q and
// walks keys in 16-key tiles. QK^T and P*V run on the fp16 systolic array
// (intel_sub_group_f16_f16_matrix_mad_k16: A 8x16 fp16, B 16x16 fp16 VNNI,
// C 8x16 fp32), replacing the scalar per-(row,key) dot + subgroup-reduce
// that dominates the blockq kernel's O(M*N) full-attention cost.
//
// Operand layouts (SG16, K=16):
//   A short8: lane l element r = A[r][l]      (Q rows / P rows)
//   B int8  : lane l element k2 = VNNI pair (B[2k2][l], B[2k2+1][l])
//   C float8: lane l element r = C[r][l]      (l = key / d-column)
// K's B-fragment needs pairs along d — ADJACENT in the K cache row, so K is
// consumed with plain int8 loads, NO repack. V's B-fragment needs pairs
// across KEYS (different rows), so the 16-key V tile is staged in SLM and
// VNNI-packed on read. Softmax bookkeeping (m/l/alpha) stays fp32; P is
// truncated to fp16 for the DPAS A operand (values in [0,1]).
// Masking uses the -1e30f sentinel (not -INFINITY: -cl-finite-math-only is
// in the default copts); rows clamped for tails are never stored.
// Compiled ONLY when the host passes -DFLASH_XMX=1 (caps().dpas devices);
// other devices never see this code.
// ===========================================================================
#if defined(FLASH_XMX)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

#ifndef FXA_D
#define FXA_D 128
#endif
#define FXA_KT 16
#define FXA_KCH (FXA_D / 16)
// Query rows per subgroup tile = the DPAS M dimension.
#ifndef FXA_TM
#define FXA_TM 8
#endif
// FXA_NSG (v2): subgroups per WG, each owning a d-slice of FXA_D/FXA_NSG.
// For d=512 this halves the per-lane chunk count back to the d=256 register
// envelope (no spill at TM=8) and doubles lane residency per WG. QK^T
// partials are summed across subgroups via SLM once per key tile; the
// softmax bookkeeping is then recomputed identically (deterministically)
// in every subgroup, and P*V / O are d-slice-local. Barrier count per tile
// is unchanged (2).
#ifndef FXA_NSG
#define FXA_NSG 1
#endif
// FXA_XB: key-tiles per psum-exchange batch (NSG>1 only). The exchange's
// SLM round-trip + WG barrier serialize ~300ns per tile per WG; batching
// XB tiles' QK partials into one exchange cuts that stall frequency by XB
// at the cost of XB*NSG*TM*KT*4B extra SLM. V staging is subgroup-local
// (each subgroup reads only the slice it wrote), so it moves inside the
// per-tile phase under a cheap sub_group_barrier.
#ifndef FXA_XB
#define FXA_XB 1
#endif
// FXA_XRED: exchange-reduction mode (NSG>1 only). 0 (default) = the original
// all-to-all reduction: EVERY subgroup reads all NSG psum partials for all TM
// rows and sums (NSG*NSG*TM*KT SLM reads/WG per tile -- redundant NSG-fold).
// 1 = DISTRIBUTED reduction: each subgroup reduces only the rows it OWNS
// (r == sg mod NSG) once, writes the full score to a shared ssum buffer, then
// all subgroups read their rows back. Cuts the psum-read volume from NSG^2 to
// ~2*NSG per (row,key) (measured: the psum round-trip is ~56% of the d512
// full-attn kernel and is SLM-traffic-bound, not barrier-bound -- shrinking
// the traffic is the lever FXA_XB batching alone could not reach). The per-g
// sum order (g=0..NSG-1) is preserved, so the online-softmax input is
// BIT-IDENTICAL to FXA_XRED=0; +1 WG barrier and +XB*TM*KT*4B SLM.
#ifndef FXA_XRED
#define FXA_XRED 0
#endif
#define FXA_DSUB (FXA_D / FXA_NSG)
#define FXA_KCH_SUB (FXA_DSUB / 16)
// Row tiles are built from DPAS M<=8 fragments: FXA_FR rows per fragment,
// FXA_FRAG fragments per tile. TM=16 exists because the kernel is K/V
// MEMORY-BANDWIDTH-bound (TM=4 A/B: 2.24x slower = traffic-proportional):
// doubling the rows served per K/V pass halves the dominant traffic, and
// the K/V fragment loads are SHARED by both DPAS fragments.
#if FXA_TM == 16
#define FXA_FR 8
#define FXA_FRAG 2
#elif FXA_TM == 8
#define FXA_FR 8
#define FXA_FRAG 1
#elif FXA_TM == 4
#define FXA_FR 4
#define FXA_FRAG 1
#else
#error "FXA_TM must be 4, 8 or 16"
#endif
#if FXA_FR == 8
#define FXA_AV short8
#define FXA_CV float8
#define FXA_AV_LOAD(p) vload8(0, p)
#define FXA_CV_STORE(v, p) vstore8(v, 0, p)
#else
#define FXA_AV short4
#define FXA_CV float4
#define FXA_AV_LOAD(p) vload4(0, p)
#define FXA_CV_STORE(v, p) vstore4(v, 0, p)
#endif

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16 * FXA_NSG, 1, 1))) __kernel void
flash_attention_prefill_f16_xmx(
  __global const half *Q, // [M, HD_Q]
  __global const half *K, // OHWI (k_stride>0) or concat
  __global const half *V, // [N_kv, HD_KV]
  __global half *O,       // [M, HD_Q]
  const int M, const int N_kv, const int d, const int HD_Q, const int HD_KV,
  const int gqa, const int is_causal, const float scale, const int k_stride,
  const float softcap, const int local_window,
  // [kv-window-ring] >0: K/V physical row = n % ring_cap. Argument 15 of the
  // Block-Q family; the XMX kernel must declare it because the host binder
  // binds the slot for every flash_blockq kernel (use_xmx is a strict subset).
  // Without the declaration clSetKernelArg(15) returns CL_INVALID_ARG_INDEX,
  // the binder fails, and mha_core silently falls back to host attention on
  // every full-attention prefill call.
  const int ring_cap) {
  const int lane = get_sub_group_local_id();
  const int sg = get_sub_group_id(); // 0..FXA_NSG-1 (d-slice owner)
  const int dbase = sg * FXA_DSUB;
  const int grp = get_group_id(0);
  const int n_row_tiles = (M + FXA_TM - 1) / FXA_TM;
  const int head_q = grp / n_row_tiles;
  const int tile = grp % n_row_tiles;
  const int m0 = tile * FXA_TM;
  const int total_groups = (HD_Q / FXA_D) * n_row_tiles;
  if (grp >= total_groups || m0 >= M)
    return;

  const int head_kv = head_q / gqa;
  const long k_head_base = (k_stride > 0)
                             ? ((long)head_kv * (long)k_stride * (long)FXA_D)
                             : ((long)head_kv * (long)FXA_D);
  const long k_row_stride = (k_stride > 0) ? (long)FXA_D : (long)HD_KV;
  const long v_head_base = (long)head_kv * (long)FXA_D;

  // Q A-fragments for THIS subgroup's d-slice, loaded once:
  // qa[ch][r] = Q[m0+r][dbase + ch*16 + lane].
  short qa[FXA_KCH_SUB][FXA_TM];
#pragma unroll
  for (int ch = 0; ch < FXA_KCH_SUB; ++ch)
#pragma unroll
    for (int r = 0; r < FXA_TM; ++r) {
      const int m = (m0 + r < M) ? (m0 + r) : (M - 1); // clamp; never stored
      qa[ch][r] = as_short(
        Q[(long)m * HD_Q + (long)head_q * FXA_D + dbase + ch * 16 + lane]);
    }

  float m_i[FXA_TM], l_i[FXA_TM], alpha[FXA_TM], p[FXA_TM];
  float acc[FXA_TM][FXA_KCH_SUB];
#pragma unroll
  for (int r = 0; r < FXA_TM; ++r) {
    m_i[r] = -1e30f;
    l_i[r] = 0.0f;
#pragma unroll
    for (int ch = 0; ch < FXA_KCH_SUB; ++ch)
      acc[r][ch] = 0.0f;
  }

  const int q_pos_off = N_kv - M;
  const int last_row =
    (((m0 + FXA_TM - 1) < M) ? (m0 + FXA_TM - 1) : (M - 1)) + q_pos_off;
  const int n_last = is_causal ? min(N_kv - 1, last_row) : (N_kv - 1);

  __local half vtile[FXA_KT * FXA_D];
#if FXA_NSG > 1
  __local float psum[FXA_XB * FXA_NSG * FXA_TM * FXA_KT];
#if FXA_XRED
  // Full (NSG-summed) scores, written once by the owning subgroup, read by all.
  __local float ssum[FXA_XB * FXA_TM * FXA_KT];
#endif
#endif

  const int n0_start =
    (is_causal && local_window > 0)
      ? ((max(0, m0 + q_pos_off - local_window + 1) / FXA_KT) * FXA_KT)
      : 0;
  for (int n0g = n0_start; n0g <= n_last; n0g += FXA_XB * FXA_KT) {
    float scb[FXA_XB][FXA_TM];

    // ---- phase 1: QK^T partials for every tile in the batch
#pragma unroll
    for (int b = 0; b < FXA_XB; ++b) {
      const int n0 = n0g + b * FXA_KT;
      if (n0 <= n_last) {
        const int kt = min(FXA_KT, n_last - n0 + 1);
        const int nk = n0 + ((lane < kt) ? lane : (kt - 1));
        // physical cache row; the causal / window masks keep the LOGICAL index
        const long pnk = (ring_cap > 0) ? (long)(nk % ring_cap) : (long)nk;
        const __global half *krow = K + k_head_base + pnk * k_row_stride;
        FXA_CV c8[FXA_FRAG];
#pragma unroll
        for (int f = 0; f < FXA_FRAG; ++f)
          c8[f] = (FXA_CV)(0.0f);
#pragma unroll
        for (int ch = 0; ch < FXA_KCH_SUB; ++ch) {
          const int8 bv =
            vload8(0, (__global const int *)(krow + dbase + ch * 16));
#pragma unroll
          for (int f = 0; f < FXA_FRAG; ++f) {
            const FXA_AV av = FXA_AV_LOAD(qa[ch] + f * FXA_FR);
            c8[f] = intel_sub_group_f16_f16_matrix_mad_k16(av, bv, c8[f]);
          }
        }
#pragma unroll
        for (int f = 0; f < FXA_FRAG; ++f)
          FXA_CV_STORE(c8[f], scb[b] + f * FXA_FR);
      }
    }

    // ---- one exchange (write + WG barrier + read) per XB tiles
#if FXA_NSG > 1
    barrier(CLK_LOCAL_MEM_FENCE); // protect psum/ssum from prev-batch readers
#pragma unroll
    for (int b = 0; b < FXA_XB; ++b)
      if (n0g + b * FXA_KT <= n_last)
#pragma unroll
        for (int r = 0; r < FXA_TM; ++r)
          psum[((b * FXA_NSG + sg) * FXA_TM + r) * FXA_KT + lane] = scb[b][r];
    barrier(CLK_LOCAL_MEM_FENCE); // psum visible
#if FXA_XRED
    // Distributed reduction: this subgroup sums the NSG partials for the rows
    // it owns (r == sg mod NSG) ONCE and publishes the full score to ssum.
#pragma unroll
    for (int b = 0; b < FXA_XB; ++b)
      if (n0g + b * FXA_KT <= n_last)
#pragma unroll
        for (int r = sg; r < FXA_TM; r += FXA_NSG) {
          float s = 0.0f;
#pragma unroll
          for (int g = 0; g < FXA_NSG; ++g)
            s += psum[((b * FXA_NSG + g) * FXA_TM + r) * FXA_KT + lane];
          ssum[(b * FXA_TM + r) * FXA_KT + lane] = s;
        }
    barrier(CLK_LOCAL_MEM_FENCE); // ssum visible
#endif
#endif

    // ---- phase 2: per tile -- full scores, softmax, V stage, P*V
#pragma unroll
    for (int b = 0; b < FXA_XB; ++b) {
      const int n0 = n0g + b * FXA_KT;
      if (n0 > n_last)
        continue;
      const int kt = min(FXA_KT, n_last - n0 + 1);
      const int nk = n0 + ((lane < kt) ? lane : (kt - 1));
      const long pnk = (ring_cap > 0) ? (long)(nk % ring_cap) : (long)nk;
      float sc[FXA_TM];
#if FXA_NSG > 1 && FXA_XRED
#pragma unroll
      for (int r = 0; r < FXA_TM; ++r)
        sc[r] = ssum[(b * FXA_TM + r) * FXA_KT + lane];
#elif FXA_NSG > 1
#pragma unroll
      for (int r = 0; r < FXA_TM; ++r) {
        float s = 0.0f;
#pragma unroll
        for (int g = 0; g < FXA_NSG; ++g)
          s += psum[((b * FXA_NSG + g) * FXA_TM + r) * FXA_KT + lane];
        sc[r] = s;
      }
#else
#pragma unroll
      for (int r = 0; r < FXA_TM; ++r)
        sc[r] = scb[b][r];
#endif

      // ---- stage the V-tile d-slice (SUBGROUP-LOCAL: this subgroup writes
      // and reads only its own slice, so a sub_group barrier suffices)
      sub_group_barrier(CLK_LOCAL_MEM_FENCE); // protect from prev-tile reads
      {
        const __global half *vrow = V + v_head_base + pnk * (long)HD_KV;
#pragma unroll
        for (int c8i = 0; c8i < FXA_DSUB / 8; ++c8i)
          vstore8(vload8(c8i, vrow + dbase), c8i, vtile + lane * FXA_D + dbase);
      }
      sub_group_barrier(CLK_LOCAL_MEM_FENCE); // slice visible

      // ---- flash softmax update; every subgroup's lane owns the same key
      const int key = n0 + lane;
#pragma unroll
      for (int r = 0; r < FXA_TM; ++r) {
        float s = scale * sc[r];
        if (softcap > 0.0f)
          s = softcap * tanh(s / softcap);
        const int mabs = m0 + r + q_pos_off;
        const int ok =
          (lane < kt) && ((m0 + r) < M) &&
          (!is_causal ||
           (key <= mabs && (local_window <= 0 || key + local_window > mabs)));
        s = ok ? s : -1e30f;
        const float tmax = sub_group_reduce_max(s);
        const float m_new = fmax(m_i[r], tmax);
        alpha[r] = exp(m_i[r] - m_new);
        p[r] = exp(s - m_new);
        l_i[r] = alpha[r] * l_i[r] + sub_group_reduce_add(p[r]);
        m_i[r] = m_new;
      }

      // ---- P*V via DPAS over this subgroup's d-slice
      short parr[FXA_TM];
#pragma unroll
      for (int r = 0; r < FXA_TM; ++r)
        parr[r] = as_short(convert_half(p[r]));
#pragma unroll
      for (int ch = 0; ch < FXA_KCH_SUB; ++ch) {
        int vbarr[8];
#pragma unroll
        for (int k2 = 0; k2 < 8; ++k2) {
          const ushort lo =
            as_ushort(vtile[(2 * k2) * FXA_D + dbase + ch * 16 + lane]);
          const ushort hi =
            as_ushort(vtile[(2 * k2 + 1) * FXA_D + dbase + ch * 16 + lane]);
          vbarr[k2] = (int)(((uint)hi << 16) | (uint)lo);
        }
        const int8 vb = vload8(0, vbarr);
#pragma unroll
        for (int f = 0; f < FXA_FRAG; ++f) {
          const FXA_AV pa = FXA_AV_LOAD(parr + f * FXA_FR);
          const FXA_CV pv8 =
            intel_sub_group_f16_f16_matrix_mad_k16(pa, vb, (FXA_CV)(0.0f));
          float pvarr[FXA_FR];
          FXA_CV_STORE(pv8, pvarr);
#pragma unroll
          for (int rr = 0; rr < FXA_FR; ++rr)
            acc[f * FXA_FR + rr][ch] =
              alpha[f * FXA_FR + rr] * acc[f * FXA_FR + rr][ch] + pvarr[rr];
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < FXA_TM; ++r) {
    if (m0 + r >= M)
      continue;
    const float inv = (l_i[r] > 0.0f) ? (1.0f / l_i[r]) : 0.0f;
    const long o_base = (long)(m0 + r) * HD_Q + (long)head_q * FXA_D;
#pragma unroll
    for (int ch = 0; ch < FXA_KCH_SUB; ++ch)
      O[o_base + dbase + ch * 16 + lane] = convert_half(acc[r][ch] * inv);
  }
}
#endif // FLASH_XMX

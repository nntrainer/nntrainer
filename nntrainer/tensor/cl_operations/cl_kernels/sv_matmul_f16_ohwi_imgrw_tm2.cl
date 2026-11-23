
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void sv_matmul_f16_ohwi_imgrw_tm2(__global const half *scores,
                                         __read_write image2d_t V_img,
                                         __global half *O, const int M,
                                         const int N_kv, const int d,
                                         const int HD_Q, const int S_max,
                                         const int gqa, const int causal,
                                         const int local_window) {
  const int x0 = get_global_id(0) * 8;
  const int m0 = get_global_id(1) * 2;
  const int head_q = get_global_id(2);
  if (m0 >= M || x0 >= d)
    return;
  const int m1 = m0 + 1;
  const int has1 = (m1 < M);
  const int head_kv = head_q / gqa;
  const int v_row0 = head_kv * d + x0;
  const long sb0 = (long)head_q * (long)M * (long)N_kv + (long)m0 * (long)N_kv;
  const long sb1 = (long)head_q * (long)M * (long)N_kv + (long)m1 * (long)N_kv;
  // Causal cap on the query's ABSOLUTE position (q_off=N_kv-M); decode (M=1)
  // sits at N-1 and must sum ALL keys, not just V[0..7]. See the non-tm2
  // kernel; a non-causal call keeps the full range.
  int N_kv_tex = (N_kv + 7) >> 3;
  int n_tex0 = 0;
  if (causal) {
    const int q_off = N_kv - M;
    const int cap_m = has1 ? m1 : m0;
    const int N_kv_tex_causal = ((q_off + cap_m) >> 3) + 1;
    if (N_kv_tex_causal < N_kv_tex)
      N_kv_tex = N_kv_tex_causal;
    // Sliding-window LOWER bound (see sv_matmul_f16_ohwi_img above). This WI
    // owns rows m0 and m1=m0+1; m0 has the SMALLER absolute query position and
    // therefore the smaller window floor, so bounding on m0 is the safe (and
    // for m1 merely conservative) choice — the extra texels m1 does not need
    // are exactly the +0.0 ones it would have skipped.
    if (local_window > 0) {
      const int n_first = q_off + m0 - local_window + 1;
      if (n_first > 0)
        n_tex0 = n_first >> 3;
      if (n_tex0 > N_kv_tex)
        n_tex0 = N_kv_tex;
    }
  }

  float acc0[8], acc1[8];
#pragma unroll
  for (int t = 0; t < 8; t++) {
    acc0[t] = 0.0f;
    acc1[t] = 0.0f;
  }

  for (int n_tex = n_tex0; n_tex < N_kv_tex; n_tex++) {
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
      const uint4 vv = read_imageui(V_img, (int2)(n_tex, v_row0 + t));
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
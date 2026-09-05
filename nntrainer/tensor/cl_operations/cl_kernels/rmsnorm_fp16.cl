#pragma OPENCL EXTENSION cl_khr_fp16 : enable
// Optional subgroup-reduce path (RMSN_SG). When LWS == subgroup size the
// cross-lane sum is one sub_group_reduce_add (no __local, no barriers). Used for
// the Intel q/k-norm where W=head_dim=128 -> LWS=16 (W8=16, perfect occupancy;
// the default LWS=64 left 48/64 WIs idle and ran a 6-round LDS tree = 4x Adreno).
#ifdef RMSN_SG
#if defined(cl_intel_required_subgroup_size)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define RMSN_SG_ATTR __attribute__((intel_reqd_sub_group_size(RMSN_LWS)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define RMSN_SG_ATTR __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define RMSN_SG_ATTR
#endif
#endif
__kernel void
rmsnorm_cl_fp16(__global const half *input, // Input tensor
                __global half *output,      // Output tensor
                __global const half *alpha, // Alpha values (one for each width)
                half epsilon,
                int B, // Number of batches
                int C, // Number of channels
                int H, // Height of feature map
                int W  // Width of feature map
) {
  int global_id = get_global_id(0); // Get the global work item index

  // Compute the corresponding batch, height, and channel indices
  int n = global_id / C;    // Batch index
  int c = global_id % C;    // Height index
  int h = get_global_id(1); // Channel index
  int index = ((n * C + c) * H + h) * W;

  // Calculate RMS norm for the current channel, height, and batch.
  // Accumulate the sum of squares in fp32: half accumulation overflows / loses
  // precision once squared activations exceed the half range (+/-65504), which
  // garbles the norm (matches the fp32 accumulation used by the cooperative
  // rmsnorm kernels in this file).
  float sum_squares = 0.0f;
  for (int j = 0; j < W; ++j) {
    const float v = (float)input[index + j];
    sum_squares += v * v;
  }
  const float mean = sum_squares / (float)W;
  const half rms_norm = (half)sqrt(mean + (float)epsilon);
  // Each work item processes all width elements for its specific n, h, c
  for (int w = 0; w < W; ++w) {
    output[index + w] = (input[index + w] / rms_norm) * alpha[w];
  }
}

// Cooperative RMSNorm: one workgroup (RMSN_LWS WIs) per row, fp32
// accumulation, half8-vectorized. Replaces the 1-WI-per-row scalar kernel
// above which, for W=1024 with only n_rows work-items total, was both
// low-occupancy and serial (266 ms / 23% of M=1024 GPU time). Requires
// W % 8 == 0 (head_dim=128 and hidden=1024 both qualify). gws = RMSN_LWS *
// n_rows (1-D), lws = RMSN_LWS; get_group_id(0) selects the row.
#ifndef RMSN_LWS
#define RMSN_LWS 64
#endif
#ifdef RMSN_SG
RMSN_SG_ATTR
#endif
__attribute__((reqd_work_group_size(RMSN_LWS, 1, 1)))
__kernel void rmsnorm_cl_fp16_coop(__global const half *input,
                                   __global half *output,
                                   __global const half *alpha, half epsilon,
                                   int n_rows, int W) {
  const int row = get_group_id(0);
  const int tid = get_local_id(0);
  if (row >= n_rows)
    return;
  const long base = (long)row * (long)W;
  const int W8 = W >> 3;
  __global const half8 *in8 = (__global const half8 *)(input + base);

  float partial = 0.0f;
  for (int i = tid; i < W8; i += RMSN_LWS) {
    const float8 v = convert_float8(in8[i]);
    partial += dot(v.lo, v.lo) + dot(v.hi, v.hi);
  }

#ifdef RMSN_SG
  const float ssum = sub_group_reduce_add(partial); // no __local, no barrier
#else
  __local float lsum[RMSN_LWS];
  lsum[tid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = RMSN_LWS >> 1; s > 0; s >>= 1) {
    if (tid < s)
      lsum[tid] += lsum[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float ssum = lsum[0];
#endif

  const float mean = ssum / (float)W;
  const float scale = rsqrt(mean + (float)epsilon);
  __global half8 *out8 = (__global half8 *)(output + base);
  for (int i = tid; i < W8; i += RMSN_LWS) {
    // Gemma4 has very large RMSNorm gammas (absmax ~83-116 vs Gemma2 ~1), so
    // normalized*gamma can exceed FP16 max (65504) on small-RMS rows -> inf/NaN.
    // Compute the gamma multiply in FP32 and clamp to a FP16-safe range; the
    // downstream QINT4 FC re-quantizes the activation to int8 so the clamp is
    // numerically negligible. No-op for Gemma2 (values << 60000).
    // gamma (alpha) is a WEIGHT pointer with no SVM alignment guarantee (Gemma4
    // lands it 2-byte-aligned), so a half8 vector load (16-byte align required)
    // reads garbage -> wrong gamma -> 60000 overflow. Load gamma per-element.
    const float8 nv = convert_float8(in8[i]) * scale;
    const int gi = i << 3;
    const float8 a = (float8)(
      (float)alpha[gi + 0], (float)alpha[gi + 1], (float)alpha[gi + 2],
      (float)alpha[gi + 3], (float)alpha[gi + 4], (float)alpha[gi + 5],
      (float)alpha[gi + 6], (float)alpha[gi + 7]);
    const float8 o = clamp(nv * a, -60000.0f, 60000.0f);
    out8[i] = convert_half8(o);
  }
}

// Gamma-free cooperative RMSNorm (Gemma4 v_norm: use_gamma=false). Identical to
// rmsnorm_cl_fp16_coop but skips the per-element gamma fold (pure normalization),
// so the host fallback (cl_queue_finish + 2x blocking SVM map + FP32 intrinsic +
// unmap, ~0.18 ms/call x ~30 calls/token) is avoided when the attention path is
// GPU-resident (NNTR_MHA_GPU_DECODE). Keeps the SAME 6-arg signature as the coop
// kernel (input, output, alpha, epsilon, n_rows, W) so the host wrapper's dispatch
// is unchanged -- alpha is bound (to the input pointer) but never read here. The
// sum-of-squares is in fp32 (convert_float8 before squaring), matching the host
// rms_norm_wrt_width_fp32_intrinsic, so v_norm's |x|~674 rows do not overflow the
// fp16 range the way an in-half squaring would.
__attribute__((reqd_work_group_size(RMSN_LWS, 1, 1)))
__kernel void rmsnorm_cl_fp16_coop_ng(__global const half *input,
                                      __global half *output,
                                      __global const half *alpha, half epsilon,
                                      int n_rows, int W) {
  const int row = get_group_id(0);
  const int tid = get_local_id(0);
  if (row >= n_rows)
    return;
  const long base = (long)row * (long)W;
  const int W8 = W >> 3;
  __global const half8 *in8 = (__global const half8 *)(input + base);

  float partial = 0.0f;
  for (int i = tid; i < W8; i += RMSN_LWS) {
    const float8 v = convert_float8(in8[i]);
    partial += dot(v.lo, v.lo) + dot(v.hi, v.hi);
  }

  __local float lsum[RMSN_LWS];
  lsum[tid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = RMSN_LWS >> 1; s > 0; s >>= 1) {
    if (tid < s)
      lsum[tid] += lsum[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float ssum = lsum[0];

  const float mean = ssum / (float)W;
  const float scale = rsqrt(mean + (float)epsilon);
  __global half8 *out8 = (__global half8 *)(output + base);
  for (int i = tid; i < W8; i += RMSN_LWS) {
    const float8 nv = convert_float8(in8[i]) * scale;
    out8[i] = convert_half8(nv);
  }
}

// PLE reverse-RMSNorm (RMSReverseNormLayer GPU path):
//   out = out_scale * normalize(input * weight)
// The per-feature `weight` is applied BEFORE the RMS denominator (it couples all
// features, so this can NOT be expressed as a plain rmsnorm*gamma), and
// `out_scale` is a POST-norm SCALAR. Sum-of-squares in fp32 (matches the host
// rms_reverse_norm FP32 path). One workgroup (RMSN_LWS WIs) per row; each WI
// strides the row. `weight` is a model-weight pointer with NO 16-byte alignment
// guarantee, so it is loaded PER-ELEMENT (a half8 vload would read garbage --
// same caveat as the coop gamma above). W need not be %8 (scalar per-element).
__attribute__((reqd_work_group_size(RMSN_LWS, 1, 1)))
__kernel void rms_reverse_norm_cl_fp16_coop(__global const half *input,
                                            __global half *output,
                                            __global const half *weight,
                                            half out_scale, half epsilon,
                                            int n_rows, int W) {
  const int row = get_group_id(0);
  const int tid = get_local_id(0);
  if (row >= n_rows)
    return;
  const long base = (long)row * (long)W;

  float partial = 0.0f;
  for (int j = tid; j < W; j += RMSN_LWS) {
    const float t = (float)input[base + j] * (float)weight[j];
    partial += t * t;
  }
  __local float lsum[RMSN_LWS];
  lsum[tid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = RMSN_LWS >> 1; s > 0; s >>= 1) {
    if (tid < s)
      lsum[tid] += lsum[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float mean = lsum[0] / (float)W;
  const float scale = rsqrt(mean + (float)epsilon) * (float)out_scale;
  for (int j = tid; j < W; j += RMSN_LWS) {
    const float t = (float)input[base + j] * (float)weight[j];
    output[base + j] = (half)(t * scale);
  }
}

// Cooperative RMSNorm that also emits the int8 activation quantisation of the
// row it just normalised (v8c_act_quant_f16_parx), so a norm feeding a v8c FC
// costs ONE dispatch instead of two.
//
// Why this is worth a kernel of its own: on the Adreno decode step both halves
// are ~2-7 us of GPU under a ~7.4 us per-dispatch submission floor, so removing
// the second dispatch returns three to four times its GPU cost. The two halves
// already read the same row.
//
// Bit-exactness is the whole design constraint, and it forces the odd shape:
//
//  * The norm's reduction is a float sum tree, and float addition is not
//    associative -- folding the same row over 256 lanes instead of 64 changes
//    `mean` and with it every output element. So the NORM half runs on exactly
//    RMSN_LWS lanes, striding and folding exactly as rmsnorm_cl_fp16_coop
//    does, whatever the workgroup size is. The remaining lanes idle through it
//    (they still execute every barrier -- a barrier must be reached by the
//    whole group).
//  * The QUANT half then runs on ALL lanes, which is the width that made the
//    standalone quantiser fast, and it is bit-identical at any width by its
//    own argument (fmin/fmax and an integer sum fold the same per-lane
//    partials in any order).
//  * The quant half re-READS the normed row from global memory rather than
//    keeping it in registers, behind a global-memory barrier. That is not a
//    missed optimisation: it is what makes the result identical to running the
//    two kernels separately, because the separate quantiser reads FP16 from
//    memory and this way the rounding to half happens in exactly one place.
//    The re-read is W halves per row, negligible against the row's own traffic.
//
// scale_per_row / zp_per_row / row_sum_act / act_int8 are the v8c FC's
// per-fanout activation scratch; n_rows rows are written, matching the rows
// the FC's quant-direct path would have written.
#define RMSNQ_LWS_MAX 256
__kernel void rmsnorm_cl_fp16_coop_q(__global const half *input,
                                     __global half *output,
                                     __global const half *alpha, half epsilon,
                                     int n_rows, int W,
                                     __global char *act_int8,
                                     __global float *scale_per_row,
                                     __global int *zp_per_row,
                                     __global int *row_sum_act) {
  const int row = get_group_id(0);
  const int tid = get_local_id(0);
  const int lsz = (int)get_local_size(0);
  if (row >= n_rows)
    return;
  const long base = (long)row * (long)W;
  const int W8 = W >> 3;
  __global const half8 *in8 = (__global const half8 *)(input + base);
  __global half8 *out8 = (__global half8 *)(output + base);

  __local float lmin[RMSNQ_LWS_MAX];
  __local float lmax[RMSNQ_LWS_MAX];
  __local int lsum[RMSNQ_LWS_MAX];
  __local float l_scale_q;
  __local int l_zp;

  /* ---- norm half: RMSN_LWS lanes, folded exactly as the coop kernel ---- */
  float partial = 0.0f;
  if (tid < RMSN_LWS) {
    for (int i = tid; i < W8; i += RMSN_LWS) {
      const float8 v = convert_float8(in8[i]);
      partial += dot(v.lo, v.lo) + dot(v.hi, v.hi);
    }
  }
  lmin[tid] = partial; /* lmin doubles as the sum-of-squares scratch here */
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = RMSN_LWS >> 1; s > 0; s >>= 1) {
    if (tid < s)
      lmin[tid] += lmin[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  const float mean = lmin[0] / (float)W;
  const float nscale = rsqrt(mean + (float)epsilon);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid < RMSN_LWS) {
    for (int i = tid; i < W8; i += RMSN_LWS) {
      const float8 nv = convert_float8(in8[i]) * nscale;
      const int gi = i << 3;
      const float8 a = (float8)(
        (float)alpha[gi + 0], (float)alpha[gi + 1], (float)alpha[gi + 2],
        (float)alpha[gi + 3], (float)alpha[gi + 4], (float)alpha[gi + 5],
        (float)alpha[gi + 6], (float)alpha[gi + 7]);
      const float8 o = clamp(nv * a, -60000.0f, 60000.0f);
      out8[i] = convert_half8(o);
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  /* ---- quant half: every lane, v8c_act_quant_f16_parx verbatim ---- */
  __global const half *q_in = output + base;
  float pmin = 0.0f, pmax = 0.0f;
  for (int k = tid; k < W; k += lsz) {
    const float v = (float)q_in[k];
    pmin = fmin(pmin, v);
    pmax = fmax(pmax, v);
  }
  lmin[tid] = pmin;
  lmax[tid] = pmax;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsz / 2; s > 0; s >>= 1) {
    if (tid < s) {
      lmin[tid] = fmin(lmin[tid], lmin[tid + s]);
      lmax[tid] = fmax(lmax[tid], lmax[tid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0) {
    const float fmn = lmin[0], fmx = lmax[0];
    const float rmin = fmn < 0.0f ? fmn : 0.0f;
    const float rmax = fmx > 0.0f ? fmx : 0.0f;
    const float qmin = -128.0f, qmax = 127.0f;
    const float range = rmax - rmin;
    const float scale_q = range > 0.0f ? 255.0f / range : 1.0f;
    const float recip = range > 0.0f ? range / 255.0f : 1.0f;
    const float dmin = rmin * scale_q, dmax = rmax * scale_q;
    const float zp_lo = qmin - dmin, zp_hi = qmax - dmax;
    float zp_f = (qmin + dmin) + (qmax + dmax) > 0.0f ? zp_lo : zp_hi;
    if (zp_f < qmin)
      zp_f = qmin;
    if (zp_f > qmax)
      zp_f = qmax;
    l_scale_q = scale_q;
    l_zp = (int)rint(zp_f);
    scale_per_row[row] = recip;
    zp_per_row[row] = l_zp;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  const float scale_q = l_scale_q;
  const int zp = l_zp;
  int psum = 0;
  for (int k = tid; k < W; k += lsz) {
    int q = (int)rint((float)q_in[k] * scale_q) + zp;
    if (q < -128)
      q = -128;
    if (q > 127)
      q = 127;
    act_int8[(long)row * W + k] = (char)q;
    psum += q;
  }
  lsum[tid] = psum;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = lsz / 2; s > 0; s >>= 1) {
    if (tid < s)
      lsum[tid] += lsum[tid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (tid == 0)
    row_sum_act[row] = lsum[0];
}

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

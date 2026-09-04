#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif

// FP16 LayerNorm. Reductions and the affine transform are accumulated in FP32
// (half accumulation loses precision / overflows once activations are squared),
// matching the FP32-accumulation policy of rmsnorm_fp16.cl. One workgroup per
// row; each work item strides the row and the per-row sums are collapsed with
// sub_group_reduce_add. gamma/beta are loaded per element (weight pointers
// carry no 16-byte vector-load alignment guarantee).
__kernel void
layernorm_cl_fp16(__global const half *input, // Input tensor
                  __global half *output,      // Output tensor
                  __global const half *gamma, // Scale (one for each width)
                  __global const half *beta,  // Shift (one for each width)
                  float epsilon,
                  int H, // Height of feature map (batch*channel*height rows)
                  int W  // Width of feature map (normalized dimension)
) {
  int h = get_group_id(0);
  int index = h * W;

  // pass 1: mean = (1/W) * sum(x)
  float s = 0.0f;
  for (int i = get_local_id(0); i < W; i += get_local_size(0)) {
    s += (float)input[index + i];
  }
  s = sub_group_reduce_add(s);
  const float mean = s / W;

  // pass 2: variance = (1/W) * sum((x - mean)^2)
  float v = 0.0f;
  for (int i = get_local_id(0); i < W; i += get_local_size(0)) {
    const float d = (float)input[index + i] - mean;
    v += d * d;
  }
  v = sub_group_reduce_add(v);
  const float scale = rsqrt(v / W + epsilon);

  // out = (x - mean) * scale * gamma + beta
  for (int i = get_local_id(0); i < W; i += get_local_size(0)) {
    const float o = ((float)input[index + i] - mean) * scale * (float)gamma[i] +
                    (float)beta[i];
    output[index + i] = (half)o;
  }
}

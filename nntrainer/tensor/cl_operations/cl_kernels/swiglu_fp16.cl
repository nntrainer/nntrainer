#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void swiglu_cl_fp16(__global const half *in1, __global const half *in2,
                             __global half *out) {
  const int i = get_global_id(0);

  // SiLU in fp32 (32-bit partial accumulation) for numerical stability.
  // The half form x*exp(x)/(1+exp(x)) OVERFLOWS: x*exp(x) exceeds half-max
  // (65504) already for x > ~8.7, and exp(x) itself overflows for x > 11,
  // both giving inf/inf = NaN. On a 28-layer Qwen3 stack the growing
  // residual eventually drives a gate activation past this threshold and the
  // whole hidden state turns NaN (bug #3). The mathematically identical
  // sigmoid form x/(1+exp(-x)) computed in fp32 never overflows (for x << 0,
  // exp(-x) -> +inf -> x/inf = 0) and also removes the systematic bias of the
  // device's low-precision half exp().
  const float x = (float)in1[i];
  const float swish = x / (1.0f + exp(-x));
  out[i] = (half)(swish * (float)in2[i]);
}

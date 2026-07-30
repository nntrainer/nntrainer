#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void sigmoid_add_cl_fp16(__global const half *in1,
                                  __global const half *in2,
                                  __global half *out) {
  const int i = get_global_id(0);

  // sigmoid + combine computed in fp32 (upcast) for numerical stability.
  // Matches the CPU/CUDA sigmoid form 1/(1+exp(-x)). The PLE gating mix:
  // out = sigmoid(gate) + emb.
  const float x = (float)in1[i];
  const float s = 1.0f / (1.0f + exp(-x));

  out[i] = (half)(s + (float)in2[i]);
}

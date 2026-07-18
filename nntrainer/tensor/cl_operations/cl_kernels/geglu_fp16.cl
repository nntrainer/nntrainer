#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void geglu_cl_fp16(__global const half *in1, __global const half *in2,
                            __global half *out, const int row_off) {
  // row_off shifts the flat index so a whole-buffer (cl_mem, bound at index 0)
  // binding can process only the live row at decode: dispatch GWS = width and
  // pass row_off = (live_row * width). SVM/host paths offset their pointers
  // instead and pass row_off = 0 (geglu is element-wise per row, so processing
  // only the consumed row is token-identical to processing the whole prefix).
  const int i = get_global_id(0) + row_off;

  // Compute the gelu in fp32 for accuracy, multiply back in fp16.
  const float x = (float)in1[i];
  const float in2_val = (float)in2[i];

  const float k0 = 0.7978845608028654f; // sqrt(2/pi)
  const float k1 = 0.044715f;
  const float inner = k0 * (x + k1 * x * x * x);
  const float gelu = 0.5f * x * (1.0f + tanh(inner));

  out[i] = (half)(gelu * in2_val);
}


#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void scalar_mul_cl_fp16(__global const half *in, __global half *out,
                                 float s, int n, int row_off) {
  int i = get_global_id(0);
  if (i < n)
    out[i + row_off] = convert_half(convert_float(in[i + row_off]) * s);
}

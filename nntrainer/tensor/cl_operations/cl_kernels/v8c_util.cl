
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void v8c_cvt_h2f(__global const half *in, __global float *out,
                          const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = (float)in[i];
}
__kernel void v8c_copy_h2h(__global const half *in, __global half *out,
                           const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = in[i];
}
__kernel void v8c_copy_f2f(__global const float *in, __global float *out,
                           const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = in[i];
}
__kernel void v8c_add_h2h(__global const half *in, __global half *out,
                          const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] += in[i];
}
__kernel void v8c_add2_h2h(__global const half *a, __global const half *b,
                           __global half *out, const int n) {
  int i = get_global_id(0);
  if (i < n) out[i] = a[i] + b[i];
}

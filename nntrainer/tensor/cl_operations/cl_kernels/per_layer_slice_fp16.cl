
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void per_layer_slice_cl_fp16(__global const half *in,
                                      __global half *out, int fs, int in_width,
                                      int off, int rows) {
  int gid = get_global_id(0);
  int total = rows * fs;
  if (gid >= total)
    return;
  int r = gid / fs;
  int j = gid - r * fs;
  out[gid] = in[r * in_width + off + j];
}

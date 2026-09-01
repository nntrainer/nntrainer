/**
 * OpenCL kernels for the sentence-embedding pooling / normalize tail.
 *
 * Two row reductions that had no op_table entry before (see ComputeOps::
 * mean_rows / l2_normalize_rows). They are deliberately NOT rmsnorm.cl:
 * L2-normalize uses the SUM of squares with epsilon as a FLOOR ON THE NORM and
 * has no gamma, while rmsnorm_cl uses the MEAN of squares, adds epsilon under
 * the sqrt, and unconditionally multiplies by a mandatory alpha/gamma buffer.
 *
 * The reduction uses __local memory rather than sub_group_reduce_add (which
 * rmsnorm.cl uses behind cl_intel_/cl_qcom_ subgroup extensions) so these
 * kernels compile and run on any OpenCL 1.2+ device. Local size is chosen by
 * the host and passed as the __local buffer length.
 */

/**
 * Row-wise L2 normalize along the last dimension:
 *   out[r, i] = in[r, i] / max(sqrt(sum_i in[r, i]^2), epsilon)
 * One work-group per row; work-items cooperate over the width.
 */
__kernel void l2_normalize_rows_cl(__global const float *input,
                                   __global float *output, const float epsilon,
                                   const int W, __local float *scratch) {
  const int row = get_group_id(0);
  const int lid = get_local_id(0);
  const int lsz = get_local_size(0);
  const int base = row * W;

  float partial = 0.0f;
  for (int i = lid; i < W; i += lsz) {
    const float v = input[base + i];
    partial += v * v;
  }

  scratch[lid] = partial;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int off = lsz >> 1; off > 0; off >>= 1) {
    if (lid < off)
      scratch[lid] += scratch[lid + off];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // max(norm, epsilon): epsilon floors the NORM here, it is not added under
  // the sqrt. This matches FloatTensor::normalization_i exactly.
  const float norm = sqrt(scratch[0]);
  const float scale = 1.0f / (norm > epsilon ? norm : epsilon);

  for (int i = lid; i < W; i += lsz) {
    output[base + i] = input[base + i] * scale;
  }
}

/**
 * Mean over rows: out[i] = (1/rows) * sum_{r<rows} in[r, i].
 * One work-item per column; the per-column accumulation walks rows in order,
 * matching the host ones-vector GEMV reduction order.
 */
__kernel void mean_rows_cl(__global const float *input, __global float *output,
                           const int rows, const int W) {
  const int col = get_global_id(0);
  if (col >= W)
    return;

  float acc = 0.0f;
  for (int r = 0; r < rows; ++r) {
    acc += input[r * W + col];
  }
  output[col] = acc / (float)rows;
}

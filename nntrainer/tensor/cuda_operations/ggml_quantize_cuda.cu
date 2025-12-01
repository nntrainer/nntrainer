#include "ggml_quantize_cuda.h"
#include <cstdio>
#include <cuda_fp16.h>

#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128
#define WARP_SIZE 32

// Helper functions for warp reduction
template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset, width);
  }
  return x;
}

static __global__ void quantize_q8_1(const float *__restrict__ x,
                                     void *__restrict__ vy, const int64_t ne00,
                                     const int64_t s01, const int64_t s02,
                                     const int64_t s03, const int64_t ne0,
                                     const int ne1, const int ne2) {
  const int64_t i0 = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

  if (i0 >= ne0) {
    return;
  }

  const int64_t i1 = blockIdx.y;
  const int64_t i2 = blockIdx.z % ne2;
  const int64_t i3 = blockIdx.z / ne2;

  const int64_t &i00 = i0;
  const int64_t &i01 = i1;
  const int64_t &i02 = i2;
  const int64_t &i03 = i3;

  // Calculate contiguous index
  const int64_t i_cont = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

  block_q8_1 *y = (block_q8_1 *)vy;

  const int64_t ib = i_cont / QK8_1;  // block index
  const int64_t iqs = i_cont % QK8_1; // quant index

  const float xi =
    i0 < ne00 ? x[i03 * s03 + i02 * s02 + i01 * s01 + i00] : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max(amax);
  sum = warp_reduce_sum(sum);

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  reinterpret_cast<half &>(y[ib].ds.x) = __float2half(d);
  reinterpret_cast<half &>(y[ib].ds.y) = __float2half(sum);
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(const float *__restrict__ x,
                                         void *__restrict__ vy,
                                         const int64_t kx0, const int64_t kx1,
                                         const int64_t kx0_padded) {

  constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
  constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

  const int64_t ix0 = ((int64_t)blockDim.x * blockIdx.x + threadIdx.x) * 4;

  if (ix0 >= kx0_padded) {
    return;
  }

  const float4 *x4 = (const float4 *)x;

  const int64_t ix1 = kx1 * blockIdx.z + blockIdx.y;

  block_q8_1_mmq *y = (block_q8_1_mmq *)vy;

  const int64_t ib0 =
    blockIdx.z * ((int64_t)gridDim.y * gridDim.x * blockDim.x /
                  QK8_1); // first block of channel
  const int64_t ib =
    ib0 + (ix0 / (4 * QK8_1)) * kx1 + blockIdx.y; // block index in channel
  const int64_t iqs = ix0 % (4 * QK8_1);          // quant index in block

  // Load 4 floats per thread and calculate max. abs. value between them:
  const float4 xi =
    ix0 < kx0 ? x4[(ix1 * kx0 + ix0) / 4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float amax = fabsf(xi.x);
  amax = fmaxf(amax, fabsf(xi.y));
  amax = fmaxf(amax, fabsf(xi.z));
  amax = fmaxf(amax, fabsf(xi.w));

  // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
  for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
  }

  float sum;
  if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
    sum = xi.x + xi.y + xi.z + xi.w;

    // Exchange calculate sum across vals_per_sum/4 threads.
#pragma unroll
    for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }
  }

  const float d_inv = 127.0f / amax;
  char4 q;
  q.x = roundf(xi.x * d_inv);
  q.y = roundf(xi.y * d_inv);
  q.z = roundf(xi.z * d_inv);
  q.w = roundf(xi.w * d_inv);

  // Write back 4 int8 values as a single 32 bit value for better memroy
  // bandwidth:
  char4 *yqs4 = (char4 *)y[ib].qs;
  yqs4[iqs / 4] = q;

  if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
    if (iqs % 16 != 0 || iqs >= 96) {
      return;
    }

    y[ib].d2s6[2 + iqs / 16] = __float2half(sum);

    if (iqs % 64 != 0) {
      return;
    }

    const float d = 1.0f / d_inv;

    y[ib].d2s6[iqs / 64] = __float2half(d);

    return;
  }

  if (iqs % 32 != 0) {
    return;
  }

  const float d = 1.0f / d_inv;

  if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
    y[ib].ds4[iqs / 32] = make_half2(__float2half(d), __float2half(sum));
  } else {
    y[ib].d4[iqs / 32] = d;
  }
}

void quantize_row_q8_1_cuda(const float *x, void *vy, const int64_t ne00,
                            const int64_t s01, const int64_t s02,
                            const int64_t s03, const int64_t ne0,
                            const int64_t ne1, const int64_t ne2,
                            const int64_t ne3, cudaStream_t stream) {

  const int64_t block_num_x =
    (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  const dim3 num_blocks(block_num_x, ne1, ne2 * ne3);
  const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
  quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, s01, s02,
                                                       s03, ne0, ne1, ne2);
}

void quantize_row_q8_1_cuda(const float *x, void *vy, int64_t k,
                            cudaStream_t stream) {
  const int64_t ne0 = k;
  const int64_t ne1 = 1;
  const int64_t ne2 = 1;
  const int64_t ne3 = 1;
  const int64_t ne00 = k;
  const int64_t s01 =
    sizeof(float) * k;     // Stride for next row (not used for 1D)
  const int64_t s02 = s01; // Stride for next matrix (not used for 1D)
  const int64_t s03 = s01; // Stride for next batch (not used for 1D)

  quantize_row_q8_1_cuda(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2, ne3,
                         stream);
}

void quantize_mmq_q8_1_cuda(const float *x, void *vy, const ggml_type type_src0,
                            const int64_t ne00, const int64_t s01,
                            const int64_t s02, const int64_t s03,
                            const int64_t ne0, const int64_t ne1,
                            const int64_t ne2, const int64_t ne3,
                            cudaStream_t stream) {

  const int64_t block_num_x = (ne0 + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) /
                              (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
  const dim3 num_blocks(block_num_x, ne1, ne2 * ne3);
  const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
  switch (mmq_get_q8_1_ds_layout(type_src0)) {
  case MMQ_Q8_1_DS_LAYOUT_D4:
    quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
      <<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, ne1, ne0);
    break;
  case MMQ_Q8_1_DS_LAYOUT_DS4:
    quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
      <<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, ne1, ne0);
    break;
  case MMQ_Q8_1_DS_LAYOUT_D2S6:
    quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
      <<<num_blocks, block_size, 0, stream>>>(x, vy, ne00, ne1, ne0);
    break;
  default:
    break;
  }
}

// CUDA kernel for INT4 quantization with padding
static __global__ void quantize_input_int4_pad_kernel(
  const float *__restrict__ input, int8_t *__restrict__ quantized_input,
  half *__restrict__ scales, unsigned int M, unsigned int K,
  unsigned int quantization_group_size) {

  const unsigned int group_id = blockIdx.x;
  const unsigned int tid = threadIdx.x;

  const unsigned int align_k =
    ((K + quantization_group_size - 1) / quantization_group_size) *
    quantization_group_size;
  const unsigned int groups_in_row = align_k / quantization_group_size;
  const unsigned int row_id = group_id / groups_in_row;
  const unsigned int group_id_in_row = group_id % groups_in_row;
  const unsigned int input_offset =
    (row_id * K) + (group_id_in_row * quantization_group_size);
  const unsigned int output_offset = group_id * quantization_group_size;
  const unsigned int max_quantize_block = quantization_group_size;

  unsigned int quantize_block;
  if (group_id_in_row == groups_in_row - 1) {
    quantize_block = quantization_group_size - (align_k - K);
  } else {
    quantize_block = quantization_group_size;
  }

  // Shared memory for reduction
  __shared__ float shared_max[32];

  // Find maximum absolute value
  float local_max = 0.0f;
  for (unsigned int i = tid; i < quantize_block; i += blockDim.x) {
    unsigned int idx = input_offset + i;
    float val = (idx < row_id * K + K)
                  ? fabsf(__half2float(__float2half(input[idx])))
                  : 0.0f;
    local_max = fmaxf(local_max, val);
  }

  shared_max[tid] = local_max;

  // Reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
    }
  }

  float max_value = fmaxf(shared_max[0], 0.001f);

  // Calculate quantization scale
  float quan_scale = max_value / 127.0f;
  float quan_scale_1 = 1.0f / quan_scale;

  // Quantize the data
  for (unsigned int i = tid; i < quantize_block; i += blockDim.x) {
    unsigned int input_idx = input_offset + i;
    unsigned int output_idx = output_offset + i;
    float val = (input_idx < row_id * K + K)
                  ? __half2float(__float2half(input[input_idx]))
                  : 0.0f;
    float quantized_val = val * quan_scale_1;
    quantized_input[output_idx] = (int8_t)__float2int_rn(quantized_val);
  }

  // Pad with zeros if necessary
  for (unsigned int i = quantize_block + tid; i < max_quantize_block;
       i += blockDim.x) {
    unsigned int output_idx = output_offset + i;
    quantized_input[output_idx] = 0;
  }

  // Store the scale (thread 0 only)
  if (tid == 0) {
    scales[group_id * 2] = __float2half(quan_scale);
    scales[group_id * 2 + 1] = (__half)0.0f; // Placeholder for activation sum
  }
}

void quantize_input_int4_pad_cuda(const void *input, void *quantized_input,
                                  void *scales, unsigned int M, unsigned int K,
                                  unsigned int quantization_group_size,
                                  cudaStream_t stream) {
  const unsigned int align_k =
    ((K + quantization_group_size - 1) / quantization_group_size) *
    quantization_group_size;
  const unsigned int groups_in_row = align_k / quantization_group_size;
  const unsigned int total_groups = M * groups_in_row;

  const dim3 grid(total_groups);
  const dim3 block(32);

  quantize_input_int4_pad_kernel<<<grid, block, 0, stream>>>(
    (const float *)input, (int8_t *)quantized_input, (half *)scales, M, K,
    quantization_group_size);
}

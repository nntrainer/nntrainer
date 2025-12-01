/**
 * @file ggml_quantize_cuda.h
 * @brief Header file for CUDA quantization functions
 * @author Samsung R&D Institute
 * @bug No known bugs
 */
#pragma once

#include "ggml_cuda_common.h"
#include <cstdint>
#include <cuda_runtime.h>

// Struct for MMQ Q8_1 block (CUDA specific)
/**
 * @brief Structure for MMQ Q8_1 block (CUDA specific)
 */
struct block_q8_1_mmq {
  union {
    float d4[4];       // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
    ggml_half2 ds4[4]; // 1 16 bit scale + 1 16 bit partial sum per 32 values,
                       // stored as d0,s0,d1,s1,d2,s2,d3,s3
    ggml_half d2s6[8]; // 1 16 bit scale per 64 values + 1 16 bit partial sum
                       // per 16 values for the first 96 values,
                       //     stored as d0,d1,s1,s2,s3,s4,s5
  };
  int8_t qs[4 * QK8_1]; // 128 values quantized to 8 bit each
};

/**
 * @brief Quantizes a row of FP32 data to Q8_1 format on the GPU (CUDA).
 *
 * This function launches a CUDA kernel to convert FP32 data into Q8_1 format.
 * It handles the grid and block dimensions for the kernel launch.
 *
 * @param x Pointer to the input FP32 data array on the device.
 * @param vy Pointer to the output buffer on the device where Q8_1 blocks will
 * be stored.
 * @param ne00 The number of elements in the 0-th dimension of the source
 * tensor.
 * @param s01 Stride of the 1st dimension of the source tensor (in bytes).
 * @param s02 Stride of the 2nd dimension of the source tensor (in bytes).
 * @param s03 Stride of the 3rd dimension of the source tensor (in bytes).
 * @param ne0 The number of elements in the 0-th dimension.
 * @param ne1 The number of elements in the 1st dimension.
 * @param ne2 The number of elements in the 2nd dimension.
 * @param ne3 The number of elements in the 3rd dimension.
 * @param stream The CUDA stream to execute the kernel on.
 */
void quantize_row_q8_1_cuda(const float *x, void *vy, const int64_t ne00,
                            const int64_t s01, const int64_t s02,
                            const int64_t s03, const int64_t ne0,
                            const int64_t ne1, const int64_t ne2,
                            const int64_t ne3, cudaStream_t stream);

/**
 * @brief Simplified version of quantize_row_q8_1_cuda matching the host API.
 *
 * This overload assumes a contiguous 1D array (single row).
 *
 * @param x Pointer to the input FP32 data array on the device.
 * @param vy Pointer to the output buffer on the device.
 * @param k The number of elements in the array.
 * @param stream The CUDA stream to execute the kernel on.
 */
void quantize_row_q8_1_cuda(const float *x, void *vy, int64_t k,
                            cudaStream_t stream);

/**
 * @brief Quantizes data to MMQ-compatible Q8_1 format on the GPU (CUDA).
 *
 * This function quantizes data specifically for Matrix Multiplication Quantized
 * (MMQ) operations. It supports different data layouts (D4, DS4, D2S6)
 * depending on the source quantization type.
 *
 * @param x Pointer to the input FP32 data array on the device.
 * @param vy Pointer to the output buffer on the device.
 * @param type_src0 The GGML type of the source tensor, determining the target
 * MMQ layout.
 * @param ne00 The number of elements in the 0-th dimension of the source
 * tensor.
 * @param s01 Stride of the 1st dimension of the source tensor (in bytes).
 * @param s02 Stride of the 2nd dimension of the source tensor (in bytes).
 * @param s03 Stride of the 3rd dimension of the source tensor (in bytes).
 * @param ne0 The number of elements in the 0-th dimension.
 * @param ne1 The number of elements in the 1st dimension.
 * @param ne2 The number of elements in the 2nd dimension.
 * @param ne3 The number of elements in the 3rd dimension.
 * @param stream The CUDA stream to execute the kernel on.
 */
void quantize_mmq_q8_1_cuda(const float *x, void *vy, const ggml_type type_src0,
                            const int64_t ne00, const int64_t s01,
                            const int64_t s02, const int64_t s03,
                            const int64_t ne0, const int64_t ne1,
                            const int64_t ne2, const int64_t ne3,
                            cudaStream_t stream);

/**
 * @brief Determines the MMQ Q8_1 data layout based on the source quantization
 * type.
 *
 * Different source quantization types (e.g., Q4_0, Q5_0) require different
 * internal layouts (D4, DS4, D2S6) when converted to Q8_1 for efficient MMQ
 * kernel execution.
 *
 * @param type_x The GGML type of the source tensor.
 * @return The corresponding mmq_q8_1_ds_layout enum value.
 */
static mmq_q8_1_ds_layout mmq_get_q8_1_ds_layout(const ggml_type type_x) {
  switch (type_x) {
  case GGML_TYPE_Q4_0:
  case GGML_TYPE_Q4_1:
    return MMQ_Q8_1_DS_LAYOUT_DS4;
  case GGML_TYPE_Q5_0:
    return MMQ_Q8_1_DS_LAYOUT_D4;
  case GGML_TYPE_Q5_1:
    return MMQ_Q8_1_DS_LAYOUT_DS4;
  case GGML_TYPE_Q8_0:
    return MMQ_Q8_1_DS_LAYOUT_D4;
  // Add other types as needed, defaulting to D4 for safety if unknown
  default:
    return MMQ_Q8_1_DS_LAYOUT_D4;
  }
}

/**
 * @brief Quantizes FP16 input to INT4 format with padding support (CUDA).
 *
 * This function quantizes FP16 input data to INT4 format in groups,
 * matching the OpenCL openvino_quantize_input_int4_pad kernel behavior.
 * Each group is quantized independently with its own scale factor.
 *
 * @param input Pointer to the input FP16 data array on the device.
 * @param quantized_input Pointer to the output INT8 buffer on the device.
 * @param scales Pointer to the output UINT16 (FP16) scales buffer on the
 * device.
 * @param M Number of rows in the input matrix.
 * @param K Number of columns in the input matrix.
 * @param quantization_group_size Size of each quantization group (typically
 * 32).
 * @param stream The CUDA stream to execute the kernel on (default: 0).
 */
void quantize_input_int4_pad_cuda(const void *input, void *quantized_input,
                                  void *scales, unsigned int M, unsigned int K,
                                  unsigned int quantization_group_size,
                                  cudaStream_t stream = 0);

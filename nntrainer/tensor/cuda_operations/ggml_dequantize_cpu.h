/**
 * @file ggml_dequantize_cpu.h
 * @brief Header file for CPU dequantization functions
 * @author Samsung R&D Institute
 * @bug No known bugs
 */
#pragma once

#include <cstdint>

/**
 * @brief Dequantizes a row of Q8_1 data to FP32 format on the host (CPU).
 *
 * This function converts Q8_1 quantized blocks back to 32-bit floating point
 * values. It is the inverse operation of quantize_row_q8_1_host.
 *
 * @param vx Pointer to the input Q8_1 blocks.
 * @param y Pointer to the output FP32 data array.
 * @param k The number of elements to dequantize. Must be a multiple of 32
 * (QK8_1).
 */
void dequantize_row_q8_1_host(const void *vx, float *y, int64_t k);

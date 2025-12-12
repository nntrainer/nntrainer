/**
 * @file ggml_quantize_cpu.h
 * @brief Header file for CPU quantization functions
 * @author Samsung R&D Institute
 * @bug No known bugs
 */
#pragma once

#include <cstdint>

/**
 * @brief Quantizes a row of FP32 data to Q8_1 format on the host (CPU).
 *
 * This function converts a contiguous array of 32-bit floating point values
 * into the Q8_1 quantization format. The Q8_1 format uses blocks of 32 values,
 * storing 8-bit quantized weights and shared scaling factors.
 *
 * @param x Pointer to the input FP32 data array. Must be 32-byte aligned.
 * @param vy Pointer to the output buffer where Q8_1 blocks will be stored.
 * @param k The number of elements in the input array. Must be a multiple of 32
 * (QK8_1).
 */
void quantize_row_q8_1_host(const float *x, void *vy, int64_t k);

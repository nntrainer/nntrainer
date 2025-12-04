// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_util.cpp
 * @brief  Shared utility functions for unit tests
 * @author Samsung Electronics Co., Ltd.
 * @bug    No known bugs except for NYI items
 *
 */

#include "unittest_util.h"
#if defined(ENABLE_OPENCL)
#include <cl_context.h>
#endif
#include <engine.h>
#include <fp16.h>

namespace nntrainer {
#if defined(ENABLE_OPENCL)
void *allocateSVM(size_t size_bytes) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  void *ptr = blas_cc->context_inst_.createSVMRegion(size_bytes);
  if (!ptr) {
    throw std::runtime_error("Failed to allocate SVM for unit test.");
  }
  return ptr;
}

void freeSVM(void *ptr) {
  auto *blas_cc =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  blas_cc->context_inst_.releaseSVMRegion(ptr);
}
#endif
int8_t round_half_to_even(float x) {
  float r = roundf(x);
  float d = r - x;
  if (fabsf(d) != 0.5f) {
    return (int8_t)r;
  }
  // If exactly halfway, round to even
  int ir = (int)r;
  return (int8_t)((ir % 2 == 0) ? ir : ir - (ir > 0 ? 1 : -1));
}

void cpu_quantize_input_int8_pad(float *input, int8_t *quantized_input,
                                 uint16_t *scales, unsigned int M,
                                 unsigned int K,
                                 unsigned int quantization_group_size) {
  int alignK = (K + quantization_group_size - 1) / quantization_group_size *
               quantization_group_size;
  int groups_in_row = alignK / quantization_group_size;

  for (int group_id = 0; group_id < M * groups_in_row; ++group_id) {
    int row_id = group_id / groups_in_row;
    int group_id_in_row = group_id % groups_in_row;
    int input_offset =
      (row_id * K) + (group_id_in_row * quantization_group_size);
    int output_offset = group_id * quantization_group_size;
    int max_quantize_block = quantization_group_size;
    int quantize_block;

    if (group_id_in_row == groups_in_row - 1) {
      quantize_block = quantization_group_size - (alignK - K);
    } else {
      quantize_block = quantization_group_size;
    }

    // Find maximum absolute value in the block
    float max_value = 0.0f;
    for (int i = 0; i < quantize_block; ++i) {
      int idx = input_offset + i;
      // Simulate half precision for input
      float val = idx < row_id * K + K
                    ? compute_fp16_to_fp32(compute_fp32_to_fp16(input[idx]))
                    : 0.0f;
      float abs_val = fabsf(val);
      max_value = fmaxf(max_value, abs_val);
    }
    float epsilon = 0.001f;
    max_value = fmaxf(max_value, epsilon);

    // Calculate quantization scale
    float quan_scale = max_value / 127.0f;

    // Quantize the data
    for (int i = 0; i < quantize_block; ++i) {
      int input_idx = input_offset + i;
      int output_idx = output_offset + i;
      // Simulate half precision for input
      float val =
        (input_idx < row_id * K + K)
          ? compute_fp16_to_fp32(compute_fp32_to_fp16(input[input_idx]))
          : 0.0f;
      float quantized_val = val / quan_scale;
      // Round to nearest even (RTE)
      int8_t rounded_val = round_half_to_even(quantized_val);
      quantized_input[output_idx] = rounded_val;
    }

    // Pad with zeros if necessary
    for (int i = quantize_block; i < max_quantize_block; ++i) {
      int output_idx = output_offset + i;
      quantized_input[output_idx] = 0;
    }

    // Store the scale
    // Kernel writes to group_id * 2 (interleaved with activation sum)
    scales[group_id * 2] = compute_fp32_to_fp16(quan_scale);
    scales[group_id * 2 + 1] = 0; // Placeholder for activation sum
  }
}

void printMatrixI(const char *name, float *data, int Y, int X) {
  printf("%s :\n", name);
  for (int y = 0; y < Y; y++) {
    // printf("[");
    for (int x = 0; x < X; x++) {
      if (x % 10 == 0) {
        printf("| ");
      }
      std::cout << (int)(0.5f + data[y * X + x]) << " ";
    }
    printf("\n");
  }
}

std::vector<float> generate_vector(const size_t size, float min_val,
                                   float max_val) {
  const float step = (max_val - min_val) / (float)size;
  float current_value = min_val;
  std::vector<float> vec(size, 0.0f);

  for (int i = 0; i < vec.size(); ++i) {
    vec[i] = current_value;
    current_value += step;
  }

  return vec;
}

std::vector<float> generate_01_vector(const size_t size,
                                      const float ones_ratio) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, (float)size);
  if (ones_ratio >= 1.0) {
    std::vector<float> vec(size, 1.0f);
    return vec;
  } else {
    std::vector<float> vec(size, 0.0f);
    size_t ones_cnt = (size_t)(size * ones_ratio);
    for (size_t i = 0; i < ones_cnt; i++) {
      int pos = static_cast<int>(dist(gen));
      vec[pos] = 1.0f;
    }
    return vec;
  }
}

void gemm_fp32_ref(const float *input, const float *weights, float *output,
                   unsigned int M, unsigned int N, unsigned int K) {
  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (unsigned int k = 0; k < K; ++k) {
        sum += input[m * K + k] *
               weights[k * N + n]; // Assuming KxN weights (row-major)
      }
      output[m * N + n] = sum;
    }
  }
}

} // namespace nntrainer

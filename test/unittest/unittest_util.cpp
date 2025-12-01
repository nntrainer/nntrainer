#include "unittest_util.h"
#include <cl_context.h>
#include <engine.h>
#include <fp16.h>

namespace nntrainer {

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

void cpu_quantize_input_int4_pad(float *input, int8_t *quantized_input,
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

} // namespace nntrainer

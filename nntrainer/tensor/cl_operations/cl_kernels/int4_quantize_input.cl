#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define ALIGN_SIZE_K ALIGN(SIZE_K, SIZE_QUANTIZATION_GROUP)

kernel void quantize_input_int4(const __global half *restrict input,
                                __global char *restrict quantized_input,
                                __global half *restrict quan_var,
                                const int size_n,
                                const int size_k,
                                const int quantization_group_size) {
  const uint offset = get_global_id(0);
  const uint input_offset = offset * quantization_group_size;
  const uint quantize_block = quantization_group_size / 4;
  half4 input_0;
  char4 quantized_value;
  half max_vals[32] = {0}; // MAX_QUANTIZATION_GROUP_SIZE / 4 = 128 / 4 = 32

  for(uint i = 0; i < quantize_block; ++i) {
    input_0 = vload4(0, &input[input_offset + (i * 4)]);
    max_vals[i] = fmax(fmax(fabs(input_0[0]), fabs(input_0[1])),
                  fmax(fabs(input_0[2]), fabs(input_0[3])));
  }

  half max_value = 0.001h;
  for (uint i = 0; i < quantize_block; i += 8) {
    half temp = fmax(fmax(fmax(max_vals[i], max_vals[i + 1]), fmax(max_vals[i + 2], max_vals[i + 3])),
                     fmax(fmax(max_vals[i + 4], max_vals[i + 5]), fmax(max_vals[i + 6], max_vals[i + 7])));
    max_value = fmax(max_value, temp);
  }

  float quan_scale = convert_float(max_value) / 127.0f;

  for (uint i = 0; i < quantize_block; ++i) {
    input_0 = vload4(0, &input[input_offset + (i * 4)]);
    float4 buff = convert_float4(input_0) / quan_scale;
    quantized_value = convert_char4_rte(buff);
    vstore4(quantized_value, 0, &quantized_input[input_offset + (i * 4)]);
  }

  // Pair of quantizing_scale and quantized activation_sum for each group
  quan_var[offset * 2] = convert_half(quan_scale);
}

kernel void quantize_input_int4_pad(const __global half *restrict input,
                                    __global char *restrict quantized_input,
                                    __global half *restrict quan_var,
                                    const int size_n,
                                    const int size_k,
                                    const int quantization_group_size) {
  const uint group_id = get_global_id(0);
  const uint groups_in_row = ALIGN(size_k, quantization_group_size) / quantization_group_size;
  const uint row_id = group_id / groups_in_row;
  const uint group_id_in_row = group_id % groups_in_row;
  const uint input_offset =
    (row_id * size_k) + (group_id_in_row * quantization_group_size);
  const uint output_offset = group_id * quantization_group_size;
  const uint max_quantize_block = quantization_group_size / 4;
  uint quantize_block;

  if (group_id_in_row == groups_in_row - 1) {
    quantize_block = (quantization_group_size - (ALIGN(size_k, quantization_group_size) - size_k)) / 4;
  } else {
    quantize_block = quantization_group_size / 4;
  }

  half4 input_0;
  char4 quantized_value;
  half max_vals[32] = {0}; // MAX_QUANTIZATION_GROUP_SIZE / 4 = 128 / 4 = 32

  for(uint i = 0; i < quantize_block; ++i) {
    input_0 = vload4(0, &input[input_offset + (i * 4)]);
    max_vals[i] = fmax(fmax(fabs(input_0[0]), fabs(input_0[1])),
                  fmax(fabs(input_0[2]), fabs(input_0[3])));
  }

  half max_value = 0.001h;
  for (uint i = 0; i < quantize_block; i += 8) {
     half temp = fmax(fmax(fmax(max_vals[i], max_vals[i + 1]), fmax(max_vals[i + 2], max_vals[i + 3])),
                      fmax(fmax(max_vals[i + 4], max_vals[i + 5]), fmax(max_vals[i + 6], max_vals[i + 7])));
     max_value = fmax(max_value, temp);
  }

  float quan_scale = convert_float(max_value) / 127.0f;

  for (uint i = 0; i < quantize_block; ++i) {
    input_0 = vload4(0, &input[input_offset + (i * 4)]);
    float4 buff = convert_float4(input_0) / quan_scale;
    quantized_value = convert_char4_rte(buff);
    vstore4(quantized_value, 0, &quantized_input[output_offset + (i * 4)]);
  }

  char4 zero_value = 0;
  for (uint i = quantize_block; i < max_quantize_block; ++i) {
    vstore4(zero_value, 0, &quantized_input[output_offset + (i * 4)]);
  }

  // Pair of quantizing_scale and quantized activation_sum for each group
  quan_var[group_id * 2] = convert_half(quan_scale);
}

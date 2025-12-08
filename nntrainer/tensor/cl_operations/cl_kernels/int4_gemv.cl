// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Modifications made by Donghyeon Jeong on September 13 2025:
// - Limit its functionality exclusively to OS_IS_YX_OSV32_ISV2

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if !defined(cl_intel_subgroups) && defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(v, l, u) MAX((l), MIN((v), (u)))

// Creates vector type.
#define MAKE_VECTOR_TYPE_IMPL_1(elem_type) elem_type
#define MAKE_VECTOR_TYPE_IMPL_2(elem_type) CAT(elem_type, 2)
#define MAKE_VECTOR_TYPE_IMPL_3(elem_type) CAT(elem_type, 3)
#define MAKE_VECTOR_TYPE_IMPL_4(elem_type) CAT(elem_type, 4)
#define MAKE_VECTOR_TYPE_IMPL_8(elem_type) CAT(elem_type, 8)
#define MAKE_VECTOR_TYPE_IMPL_16(elem_type) CAT(elem_type, 16)
#define MAKE_VECTOR_TYPE(elem_type, size)                                      \
  CAT(MAKE_VECTOR_TYPE_IMPL_, size)(elem_type)

#define AS_TYPE(type, val) CAT(as_, type)(val)

#define TYPE_SIZE_uchar 1
#define TYPE_SIZE_char 1
#define TYPE_SIZE_ushort 2
#define TYPE_SIZE_short 2
#define TYPE_SIZE_half 2
#define TYPE_SIZE_int 4
#define TYPE_SIZE_uint 4
#define TYPE_SIZE_float 4
#define TYPE_SIZE_ulong 8
#define TYPE_SIZE_long 8
#define TYPE_SIZE(type) CAT(TYPE_SIZE_, type)

#ifdef cl_intel_required_subgroup_size
#define REQD_SUB_GROUP_SIZE(sg_size)                                           \
  __attribute__((intel_reqd_sub_group_size(sg_size)))
#else
#define REQD_SUB_GROUP_SIZE(sg_size)
#endif

#define BLOCK_READ_TYPE_size1 uchar
#define BLOCK_READ_TYPE_size2 ushort
#define BLOCK_READ_TYPE_size4 uint
#define BLOCK_READ_TYPE_size8 ulong
#define BLOCK_READ_TYPE(type_size) CAT(BLOCK_READ_TYPE_size, type_size)

#define BLOCK_READ_FUNC_size1 _sub_group_block_read_uc
#define BLOCK_READ_FUNC_size2 _sub_group_block_read_us
#define BLOCK_READ_FUNC_size4 _sub_group_block_read
#define BLOCK_READ_FUNC_size8 _sub_group_block_read_ul
#define BLOCK_READ_FUNC(type_size) CAT(BLOCK_READ_FUNC_size, type_size)

#define BLOCK_READN_FUNC_SIZE_DEF(type_size, vector_size)                      \
  MAKE_VECTOR_TYPE(BLOCK_READ_FUNC(type_size), vector_size)
#define BLOCK_READN_FUNC_size1(vector_size)                                    \
  BLOCK_READN_FUNC_SIZE_DEF(1, vector_size)
#define BLOCK_READN_FUNC_size2(vector_size)                                    \
  BLOCK_READN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_READN_FUNC_size4(vector_size)                                    \
  BLOCK_READN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_READN_FUNC_size8(vector_size)                                    \
  BLOCK_READN_FUNC_SIZE_DEF(8, vector_size)
#define BLOCK_READN_FUNC(type_size, vector_size)                               \
  CAT(BLOCK_READN_FUNC_size, type_size)(vector_size)

#define BLOCK_READN_RAW(type_size, vector_size, addr_space, ptr, offset)       \
  BLOCK_READN_FUNC(type_size, vector_size)                                     \
  ((const addr_space BLOCK_READ_TYPE(type_size) *)(ptr) + (offset))

#define BLOCK_READN(type, vector_size, ptr, offset)                            \
  AS_TYPE(                                                                     \
    MAKE_VECTOR_TYPE(type, vector_size),                                       \
    BLOCK_READN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset))

#define BLOCK_READN_SLM(type, vector_size, ptr, offset)                        \
  AS_TYPE(MAKE_VECTOR_TYPE(type, vector_size),                                 \
          BLOCK_READN_RAW(TYPE_SIZE(type), vector_size, __local, ptr, offset))

#define DT_INPUT_BLOCK_READ(ptr, offset) BLOCK_READN(half, 1, ptr, offset)
#define DT_INPUT_BLOCK_READ2(ptr, offset) BLOCK_READN(half, 2, ptr, offset)
#define DT_INPUT_BLOCK_READ4(ptr, offset) BLOCK_READN(half, 4, ptr, offset)
#define DT_INPUT_BLOCK_READ8(ptr, offset) BLOCK_READN(half, 8, ptr, offset)
#define DT_INPUT_BLOCK_READ16(ptr, offset) BLOCK_READN(half, 16, ptr, offset)

#define DT_FILTER_BLOCK_READ(ptr, offset) BLOCK_READN(char, 1, ptr, offset)
#define DT_FILTER_BLOCK_READ2(ptr, offset) BLOCK_READN(char, 2, ptr, offset)
#define DT_FILTER_BLOCK_READ4(ptr, offset) BLOCK_READN(char, 4, ptr, offset)
#define DT_FILTER_BLOCK_READ8(ptr, offset) BLOCK_READN(char, 8, ptr, offset)
#define DT_FILTER_BLOCK_READ16(ptr, offset) BLOCK_READN(char, 16, ptr, offset)

#define BLOCK_READ_IMPL(vec_size) CAT(BLOCK_READ_IMPL_, vec_size)
#define BLOCK_READ_FUNC_NAME(type_size, vec_size)                              \
  MAKE_VECTOR_TYPE(BLOCK_READ_FUNC(type_size), vec_size)
#define DECLARE_BLOCK_READ_EMULATION(type_size, vec_size)                      \
  inline MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size)                \
    BLOCK_READ_FUNC_NAME(type_size, vec_size)(                                 \
      const __global BLOCK_READ_TYPE(type_size) * ptr) {                       \
    uint idx = get_sub_group_local_id();                                       \
    MAKE_VECTOR_TYPE(BLOCK_READ_TYPE(type_size), vec_size) ret;                \
    BLOCK_READ_IMPL(vec_size)                                                  \
    return ret;                                                                \
  }

#if defined(cl_intel_subgroups_short)
#define _sub_group_block_read_us(ptr) intel_sub_group_block_read_us(ptr)
#define _sub_group_block_read_us2(ptr) intel_sub_group_block_read_us2(ptr)
#define _sub_group_block_read_us4(ptr) intel_sub_group_block_read_us4(ptr)
#define _sub_group_block_read_us8(ptr) intel_sub_group_block_read_us8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
DECLARE_BLOCK_READ_EMULATION(2, 1)
DECLARE_BLOCK_READ_EMULATION(2, 2)
DECLARE_BLOCK_READ_EMULATION(2, 4)
DECLARE_BLOCK_READ_EMULATION(2, 8)
#endif

#if defined(cl_intel_subgroups_char)
#define _sub_group_block_read_uc(ptr) intel_sub_group_block_read_uc(ptr)
#define _sub_group_block_read_uc2(ptr) intel_sub_group_block_read_uc2(ptr)
#define _sub_group_block_read_uc4(ptr) intel_sub_group_block_read_uc4(ptr)
#define _sub_group_block_read_uc8(ptr) intel_sub_group_block_read_uc8(ptr)
#define _sub_group_block_read_uc16(ptr) intel_sub_group_block_read_uc16(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
DECLARE_BLOCK_READ_EMULATION(1, 1)
DECLARE_BLOCK_READ_EMULATION(1, 2)
DECLARE_BLOCK_READ_EMULATION(1, 4)
DECLARE_BLOCK_READ_EMULATION(1, 8)
DECLARE_BLOCK_READ_EMULATION(1, 16)
#endif

#define SIMD 16
#define SUBGROUP_SIZE SIMD
#define DECOMPRESSION_GROUP_SIZE SIZE_QUANTIZATION_GROUP
#define INPUT_TILE_SIZE 1

#define GEMV_INPUT_VEC_TYPE MAKE_VECTOR_TYPE(half, INPUT_TILE_SIZE)
#define GEMV_ACCUMULATOR_VEC_TYPE MAKE_VECTOR_TYPE(float, 8)
#define GEMV_FILTER_VEC_TYPE MAKE_VECTOR_TYPE(half, 16)
#define GEMV_FILTER_PACKED_VEC_TYPE MAKE_VECTOR_TYPE(char, 16)
#define GEMV_OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(half, 1)
#define TO_GEMV_OUTPUT_VEC_TYPE(x) CAT(convert_, GEMV_OUTPUT_VEC_TYPE)(x)
#define TO_GEMV_FILTER_VEC_TYPE(x) CAT(convert_, GEMV_FILTER_VEC_TYPE)(x)
#define TO_GEMV_FILTER_PACKED_VEC_TYPE(x)                                      \
  CAT(convert_, GEMV_FILTER_PACKED_VEC_TYPE)(x)

#define GEMV_INPUT_BLOCK_READ(ptr, offset)                                     \
  BLOCK_READN(half, INPUT_TILE_SIZE, ptr, offset)
#define GEMV_FILTER_BLOCK_READ(ptr, offset) BLOCK_READN(char, 16, ptr, offset)

inline int get_4bit_weight_index(int k, int n, int K, int N, int OSV) {
  return (n / OSV) * (OSV * K / 2) + (n % OSV) + (k / 2) * OSV;
}

inline int get_4bit_weight_index_no_isv(int k, int n, int K, int N, int OSV) {
  return (n / OSV) * (OSV * K / 2) + (k / 2) * OSV;
}

inline void thread_task_splitter(const int group_num, const int thr_num,
                                 const int thr_id, int *n_start, int *n_end) {
  if (thr_num <= 1 || group_num == 0) {
    *n_start = 0;
    *n_end = group_num;
  } else {
    int num = (group_num + thr_num - 1) / thr_num;
    int num_minus = num - 1;
    int last = group_num - num_minus * thr_num;
    *n_end = thr_id < last ? num : num_minus;
    *n_start =
      thr_id <= last ? thr_id * num : last * num + (thr_id - last) * num_minus;
  }
  *n_end += *n_start;
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) kernel void
fully_connected_gpu_int4_gemv(__global half *input, const __global half *scales,
                              __global half *output,
                              const __global char *weights, const int WEIGHTS_K,
                              const int WEIGHTS_N,
                              const int quantization_group_size,
                              const int scale_row_major) {
  const int SCALE_GROUP_NUM = CEIL_DIV(WEIGHTS_K, quantization_group_size);
  int ALIGN_WEIGHTS_N = ALIGN(WEIGHTS_N, 32);
  int ALIGN_WEIGHTS_K = ALIGN(WEIGHTS_K, quantization_group_size);

  int n = get_global_id(0) * 2;         // N
  int thr_id = get_local_id(2);         // 0~15
  int thr_num = get_local_size(2);      // 16
  int wi_id = get_sub_group_local_id(); // 0~15

  int gk0, gk1;
  thread_task_splitter(SCALE_GROUP_NUM, thr_num, thr_id, &gk0, &gk1);

  __local float all_sum_even[16][16]; // [wi_id, thr_id]
  __local float all_sum_odd[16][16];

  if (scale_row_major) {
    scales += ((n / 32) * 32 + (n % 32) / 2) * SCALE_GROUP_NUM;
  } else {
    // Scale layout is fbyx
    scales += (n / 32) * 32 + (n % 32) / 2;
  }

  float2 sum_all = 0;
  for (int gk = gk0; gk < gk1; gk++) {
    __global half *A = input + gk * quantization_group_size;
    int w_id = get_4bit_weight_index(gk * quantization_group_size, n,
                                     ALIGN_WEIGHTS_K, ALIGN_WEIGHTS_N, 32);

    const __global char *B = weights + w_id;

    GEMV_ACCUMULATOR_VEC_TYPE sum = 0;

    float scale_0, scale_1;
    if (scale_row_major) {
      scale_0 = convert_float(scales[gk]);
      scale_1 = convert_float(scales[gk + 16 * SCALE_GROUP_NUM]);
    } else {
      scale_0 = convert_float(scales[gk * ALIGN_WEIGHTS_N]);
      scale_1 = convert_float(scales[gk * ALIGN_WEIGHTS_N + 16]);
    }

    __attribute__((opencl_unroll_hint(4))) for (int g = 0;
                                                g < quantization_group_size;
                                                g += 16, B += 16 * 16) {
      GEMV_INPUT_VEC_TYPE input_value =
        GEMV_INPUT_BLOCK_READ(A, g); // read 16 elements of A

      GEMV_FILTER_PACKED_VEC_TYPE bx16 = TO_GEMV_FILTER_PACKED_VEC_TYPE(
        GEMV_FILTER_BLOCK_READ(B, 0)); // read 16x16 int8 = (16x2)x16 int4

#if WEI_UINT4
      GEMV_FILTER_VEC_TYPE i4x16_even =
        TO_GEMV_FILTER_VEC_TYPE(bx16 & (char16)0xF);
      GEMV_FILTER_VEC_TYPE i4x16_odd =
        TO_GEMV_FILTER_VEC_TYPE(as_char16(as_uchar16(bx16) >> 4));
#else
      char16 i4x16_even_c16 = (bx16 & (char16)0xF);
      char16 i4x16_odd_c16 = (as_char16(as_uchar16(bx16) >> 4));
      i4x16_even_c16 = select(i4x16_even_c16, i4x16_even_c16 - (char16)16,
                              i4x16_even_c16 > (char16)7);
      i4x16_odd_c16 = select(i4x16_odd_c16, i4x16_odd_c16 - (char16)16,
                             i4x16_odd_c16 > (char16)7);
      GEMV_FILTER_VEC_TYPE i4x16_even = TO_GEMV_FILTER_VEC_TYPE(i4x16_even_c16);
      GEMV_FILTER_VEC_TYPE i4x16_odd = TO_GEMV_FILTER_VEC_TYPE(i4x16_odd_c16);
#endif

      sum[0] += as_half(sub_group_broadcast(input_value, 0)) * i4x16_even.s0 +
                as_half(sub_group_broadcast(input_value, 4)) * i4x16_even.s4 +
                as_half(sub_group_broadcast(input_value, 8)) * i4x16_even.s8 +
                as_half(sub_group_broadcast(input_value, 12)) * i4x16_even.sc;

      sum[1] += as_half(sub_group_broadcast(input_value, 0)) * i4x16_even.s1 +
                as_half(sub_group_broadcast(input_value, 4)) * i4x16_even.s5 +
                as_half(sub_group_broadcast(input_value, 8)) * i4x16_even.s9 +
                as_half(sub_group_broadcast(input_value, 12)) * i4x16_even.sd;

      sum[2] += as_half(sub_group_broadcast(input_value, 1)) * i4x16_odd.s0 +
                as_half(sub_group_broadcast(input_value, 5)) * i4x16_odd.s4 +
                as_half(sub_group_broadcast(input_value, 9)) * i4x16_odd.s8 +
                as_half(sub_group_broadcast(input_value, 13)) * i4x16_odd.sc;

      sum[3] += as_half(sub_group_broadcast(input_value, 1)) * i4x16_odd.s1 +
                as_half(sub_group_broadcast(input_value, 5)) * i4x16_odd.s5 +
                as_half(sub_group_broadcast(input_value, 9)) * i4x16_odd.s9 +
                as_half(sub_group_broadcast(input_value, 13)) * i4x16_odd.sd;

      sum[4] += as_half(sub_group_broadcast(input_value, 2)) * i4x16_even.s2 +
                as_half(sub_group_broadcast(input_value, 6)) * i4x16_even.s6 +
                as_half(sub_group_broadcast(input_value, 10)) * i4x16_even.sa +
                as_half(sub_group_broadcast(input_value, 14)) * i4x16_even.se;

      sum[5] += as_half(sub_group_broadcast(input_value, 2)) * i4x16_even.s3 +
                as_half(sub_group_broadcast(input_value, 6)) * i4x16_even.s7 +
                as_half(sub_group_broadcast(input_value, 10)) * i4x16_even.sb +
                as_half(sub_group_broadcast(input_value, 14)) * i4x16_even.sf;

      sum[6] += as_half(sub_group_broadcast(input_value, 3)) * i4x16_odd.s2 +
                as_half(sub_group_broadcast(input_value, 7)) * i4x16_odd.s6 +
                as_half(sub_group_broadcast(input_value, 11)) * i4x16_odd.sa +
                as_half(sub_group_broadcast(input_value, 15)) * i4x16_odd.se;

      sum[7] += as_half(sub_group_broadcast(input_value, 3)) * i4x16_odd.s3 +
                as_half(sub_group_broadcast(input_value, 7)) * i4x16_odd.s7 +
                as_half(sub_group_broadcast(input_value, 11)) * i4x16_odd.sb +
                as_half(sub_group_broadcast(input_value, 15)) * i4x16_odd.sf;
    }

    sum_all[0] += (sum[0] + sum[2] + sum[4] + sum[6]) * scale_0;
    sum_all[1] += (sum[1] + sum[3] + sum[5] + sum[7]) * scale_1;
  }

  all_sum_even[wi_id][thr_id] = sum_all[0];
  all_sum_odd[wi_id][thr_id] = sum_all[1];
  barrier(CLK_LOCAL_MEM_FENCE);

  float2 sum_value;
  sum_value[0] = as_float(
    intel_sub_group_block_read((const __local uint *)all_sum_even[thr_id]));
  sum_value[1] = as_float(
    intel_sub_group_block_read((const __local uint *)all_sum_odd[thr_id]));
  sum_value[0] = sub_group_reduce_add(sum_value[0]);
  sum_value[1] = sub_group_reduce_add(sum_value[1]);

  if (wi_id == 0) {
    int cur_n = n + thr_id;

    for (int i = 0; i < 2; i++) {
      output[cur_n + 16 * i] =
        TO_GEMV_OUTPUT_VEC_TYPE(convert_half(sum_value[i]));
    }
  }
}

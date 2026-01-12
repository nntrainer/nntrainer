// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Daekyoung Jung All Rights Reserved.
 *
 * @file   openvino_gemm_nomacro.cl
 * @date   12 1월 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Daekyoung Jung <dk11.jung@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file contains ...
 */

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_char : enable

typedef struct __attribute__((packed)) int4x2_t {
  char s0;
} int4x2_t;
typedef struct __attribute__((packed)) int4x4_t {
  int4x2_t s0;
  int4x2_t s1;
} int4x4_t;
typedef struct __attribute__((packed)) int4x8_t {
  int4x2_t s0;
  int4x2_t s1;
  int4x2_t s2;
  int4x2_t s3;
} int4x8_t;

typedef struct __attribute__((packed)) uint4x2_t {
  uchar s0;
} uint4x2_t;
typedef struct __attribute__((packed)) uint4x4_t {
  uint4x2_t s0;
  uint4x2_t s1;
} uint4x4_t;
typedef struct __attribute__((packed)) uint4x8_t {
  uint4x2_t s0;
  uint4x2_t s1;
  uint4x2_t s2;
  uint4x2_t s3;
} uint4x8_t;

inline uchar2 cvt_uint4x2_to_uint8x2(uint4x2_t v) __attribute__((overloadable)) {
  const uchar v0 = v.s0 & 0x0F;
  const uchar v1 = (v.s0 & 0xF0) >> 4;
  return (uchar2)(v0, v1);
}

inline char2 cvt_uint4x2_to_int8x2(uint4x2_t v) __attribute__((overloadable)) {
  const char v0 = convert_char(v.s0 & 0x0F);
  const char v1 = convert_char((v.s0 & 0xF0) >> 4);
  return (char2)(v0, v1);
}

inline char2 cvt_int4x2_to_int8x2(int4x2_t v) __attribute__((overloadable)) {
  const char s_bit = (v.s0 & convert_char(0x08));
  const char mask = s_bit > 0 ? convert_char(0xF0) : convert_char(0x00);
  const char v0 = (v.s0 & convert_char(0x0F)) | mask;
  const char v1 = v.s0 >> 4;
  return (char2)(v0, v1);
}

inline uchar2 unpack_to_uchar(uint4x2_t v) __attribute__((overloadable)) {
  return cvt_uint4x2_to_uint8x2(v);
}

inline char2 unpack_to_char(int4x2_t v) __attribute__((overloadable)) {
  return cvt_int4x2_to_int8x2(v);
}

inline char2 unpack_to_char(uint4x2_t v) __attribute__((overloadable)) {
  return convert_char2(cvt_uint4x2_to_uint8x2(v));
}

inline char4 unpack_to_char(int4x4_t v) __attribute__((overloadable)) {
  char2 v0 = unpack_to_char(v.s0);
  char2 v1 = unpack_to_char(v.s1);
  return (char4)(v0.s0, v0.s1, v1.s0, v1.s1);
}

inline char4 unpack_to_char(uint4x4_t v) __attribute__((overloadable)) {
  char2 v0 = unpack_to_char(v.s0);
  char2 v1 = unpack_to_char(v.s1);
  return (char4)(v0.s0, v0.s1, v1.s0, v1.s1);
}

inline uchar4 unpack_to_uchar(uint4x4_t v) __attribute__((overloadable)) {
  uchar2 v0 = unpack_to_uchar(v.s0);
  uchar2 v1 = unpack_to_uchar(v.s1);
  return (uchar4)(v0.s0, v0.s1, v1.s0, v1.s1);
}

inline int imad_SW(int acc, uchar4 input, char4 weight) __attribute__((overloadable)) {
  acc += input[0] * weight[0];
  acc += input[1] * weight[1];
  acc += input[2] * weight[2];
  acc += input[3] * weight[3];
  return acc;
}

inline int imad_SW(int acc, char4 input, char4 weight) __attribute__((overloadable)) {
  acc += input[0] * weight[0];
  acc += input[1] * weight[1];
  acc += input[2] * weight[2];
  acc += input[3] * weight[3];
  return acc;
}

inline int imad_SW(int acc, char4 input, uchar4 weight) __attribute__((overloadable)) {
  acc += input[0] * weight[0];
  acc += input[1] * weight[1];
  acc += input[2] * weight[2];
  acc += input[3] * weight[3];
  return acc;
}

inline int imad_SW(int acc, uchar4 input, uchar4 weight) __attribute__((overloadable)) {
  acc += input[0] * weight[0];
  acc += input[1] * weight[1];
  acc += input[2] * weight[2];
  acc += input[3] * weight[3];
  return acc;
}

inline char4 _sub_group_shuffle(char4 v, uint c) __attribute__((overloadable)) {
  return (char4)(
    as_char(intel_sub_group_shuffle(as_char(v.s0), c)),
    as_char(intel_sub_group_shuffle(as_char(v.s1), c)),
    as_char(intel_sub_group_shuffle(as_char(v.s2), c)),
    as_char(intel_sub_group_shuffle(as_char(v.s3), c)));
}

inline uchar4 _sub_group_shuffle(uchar4 v, uint c) __attribute__((overloadable)) {
  return (uchar4)(
    as_uchar(intel_sub_group_shuffle(as_uchar(v.s0), c)),
    as_uchar(intel_sub_group_shuffle(as_uchar(v.s1), c)),
    as_uchar(intel_sub_group_shuffle(as_uchar(v.s2), c)),
    as_uchar(intel_sub_group_shuffle(as_uchar(v.s3), c)));
}

inline char8 unpack_to_char_osv32_isv2(int4x8_t v) __attribute__((overloadable)) {
  char2 v0 = unpack_to_char(v.s0);
  char2 v1 = unpack_to_char(v.s2);
  char2 v2 = unpack_to_char(v.s1);
  char2 v3 = unpack_to_char(v.s3);
  return (char8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}

inline char8 unpack_to_char_osv32_isv2(uint4x8_t v) __attribute__((overloadable)) {
  char2 v0 = unpack_to_char(v.s0);
  char2 v1 = unpack_to_char(v.s2);
  char2 v2 = unpack_to_char(v.s1);
  char2 v3 = unpack_to_char(v.s3);
  return (char8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}

// Helper macros for dynamic parameters
#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define ALIGN_SIZE_K ALIGN(SIZE_K, SIZE_QUANTIZATION_GROUP)
#define DECOMPRESSION_SCALE_GROUPS_NUM CEIL_DIV(SIZE_K, SIZE_QUANTIZATION_GROUP)
#define DECOMPRESSION_SCALE_BATCH_NUM ALIGN(SIZE_N, 32)
#define DECOMPRESSION_SCALE_BATCH_PITCH DECOMPRESSION_SCALE_GROUPS_NUM
#define DECOMPRESSION_SCALE_FEATURE_PITCH 1
#define DECOMPRESSION_SCALE_LENGTH ((ALIGN(SIZE_N, 32)) * (DECOMPRESSION_SCALE_GROUPS_NUM))

inline void fc_bf_tiled_kernel_dyn_quan_nomacro(
  const __global half *input,
  __global char *quantized_input,
  __global half *quan_var,
  const __global half *decompression_scale,
  __global half *output,
  const __global char *weights,
  __local uint *wei_local_mem,
  const int BATCH_SIZE) {

  uint gid = (uint)get_group_id(0);
  uint local_id = (uint)get_local_id(1);
  uint sglid = (uint)get_sub_group_local_id();

  // DISPATCH_FSV = 1, DISPATCH_BSV = 1
  uint feature_mini_block = gid % 1;
  uint batch_mini_block = gid / 1 % 1;

  // TILE_OUT_F_NUM = SIZE_N
  // OUTER_OFM = 1, TILE_OFM = 2, SIMD = 16
  uint feature_mega_block = gid / 1 % (CEIL_DIV(SIZE_N, 32) / 1);
  uint batch_mega_block = gid / (1 * CEIL_DIV(SIZE_N, 32) / 1);

  // FILTER_VEC_TYPE = float8 (TILE_K_OFM = 8)
  float8 wei = 0;

  uint out_f = gid * 32;
  // LWS_BATCHES = 8, TILE_B = 8
  uint out_b = 64 * (uint)get_group_id(1) + local_id * 8;

  // OUTPUT_3D = 0
  uint input_offset = out_b * ALIGN_SIZE_K;

  // COMPRESSED_WEIGHTS_INT4 = 1
  // FILTER_LAYOUT_OS_IS_YX_OSV64_ISV2 = 0
  // weights_offset = out_f * (INPUT_ELEMENTS_COUNT / 2)
  uint weights_offset = out_f * (ALIGN_SIZE_K / 2);

  // ACCUMULATOR_VEC_TYPE = float2 (TILE_OFM=2)
  float2 acc[8] = {};

  // Dynamic Quantize
  // INPUT_LOAD_SIZE = 4
  char4 tiled_input_0[4] = {}; // HALF_TILE_B = 4
  uint packed_in_0[4] = {};
  half de_quantize_scale[8];

  // COMPRESSED_WEIGHTS = 1, OUTER_OFM = 1
#if DECOMPRESSION_SCALE_GROUPS_NUM == 1
  // DECOMPRESSION_SCALE_LENGTH > 1 && DECOMPRESSION_SCALE_LENGTH % 32 == 0
  // We assume SIZE_N is multiple of 32 or handled by padding?
  // ALIGN_SIZE_N is aligned to 32.
  // DECOMPRESSION_SCALE_LENGTH = ALIGN_SIZE_N * 1.
  // So it is multiple of 32.
  float2 d_scale = as_float2(intel_sub_group_block_read_us2((const __global ushort *)decompression_scale + out_f));
#else
  float2 d_scale = decompression_scale[0];
#endif

  float *d_scales = (float *)(&d_scale);

  // DECOMPRESSION_ZP_TERM = 0 (assumed)

  float2 activated[8] = {};

  // Main computation loop
  // MAIN_LOOP_ELEMENTS_COUNT = ALIGN_SIZE_K
  // TILE_IFM_ELEMENTS_SIZE = 32
  const uint iterations = CEIL_DIV(ALIGN_SIZE_K, 32);
  
  const uint idx_sglid = (sglid * 4) % 32;
  const uint batch_sglid = (sglid * 4) / 32;
  const uint scale_pitch = CEIL_DIV(ALIGN_SIZE_K, SIZE_QUANTIZATION_GROUP);

  // PER_TOKEN_SIZE_DYN_QUANTIZE = 0

  // COMPRESSED_WEIGHTS_INT8 = 0

  int8 acc_tmp[2] = {}; // TILE_OFM = 2, TILE_B = 8. int8 is vector of 8 ints.

  __attribute__((opencl_unroll_hint(1)))
  for (uint ni = 0; ni < iterations; ++ni) {
    uint in_offset = input_offset + (idx_sglid + batch_sglid * ALIGN_SIZE_K);
    uint scale_offset = CEIL_DIV(input_offset, SIZE_QUANTIZATION_GROUP);
    
    for (uint bi = 0; bi < 4; ++bi) {
      // Load quantizing info
      // tiled_input_0[bi] = vload4(0, &quantized_input[in_offset]);
      // quantized_input is char*. vload4 loads char4.
      tiled_input_0[bi] = vload4(0, &quantized_input[in_offset]);
      packed_in_0[bi] = as_uint(tiled_input_0[bi]);

      in_offset += (ALIGN_SIZE_K * 2);

      // NUM_LOOP_IN_DYN_QUAN_GROUP = QUANTIZE_GROUP_SIZE / (TILE_IFM * SIMD) = SIZE_QUANTIZATION_GROUP / 32
      // We assume SIZE_QUANTIZATION_GROUP >= 32.
#if (SIZE_QUANTIZATION_GROUP / 32) == 1
      de_quantize_scale[bi * 2] = quan_var[scale_offset * 2];
      de_quantize_scale[bi * 2 + 1] = quan_var[scale_offset * 2 + scale_pitch * 2];
      scale_offset += (scale_pitch * 2);
#endif
    }

#if (SIZE_QUANTIZATION_GROUP / 32) > 1
    if (ni % (SIZE_QUANTIZATION_GROUP / 32) == 0) {
      __attribute__((opencl_unroll_hint))
      for (uint bi = 0; bi < 8; ++bi) {
        de_quantize_scale[bi] = quan_var[scale_offset * 2];
        scale_offset += scale_pitch;
      }
    }
#endif

    input_offset += 32;

    // Skip first barrier if single iteration
    if (iterations > 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    __local uint *char_slm_weight = (__local uint *)wei_local_mem;

    // COMPRESSED_WEIGHTS_INT4 = 1
    // FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2 = 1
    // FILTER_ACTUAL_LOAD_BLOCK_SIZE = 4
    // FILTER_LOAD_ITERS = 1
    // TILE_K_OFM_PACKED = 4
    
    // weights_idx calculation
    // FILTER_LAYOUT_OS_IS_YX_OSV64_ISV2 = 0
    uint weights_idx = weights_offset + local_id * 16 * 1 * 4;
    
    uint wei_local_idx = local_id * 16 * 1 * (4 / 2) + sglid * 2;

    // FILTER_LOAD_ITERS = 1
    __attribute__((opencl_unroll_hint))
    for (uint load_iter = 0; load_iter < 1; ++load_iter) {
        // FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2 = 1
        // SLM_FILTER_PACKED_VEC = char4
        // BLOCK_READN(char, 4, weights, weights_idx)
        char4 wei_packed = as_char4(intel_sub_group_block_read((const __global uint *)weights + weights_idx/4)); 
        // Note: BLOCK_READN for char4 uses intel_sub_group_block_read which reads uint (4 bytes).
        // weights is char*. weights_idx is byte offset? No, index in FILTER_TYPE (char).
        // BLOCK_READN_RAW(1, 4, ...) -> BLOCK_READN_FUNC(1, 4) -> _sub_group_block_read_uc4
        // Wait, openvino_gemm.cl defines _sub_group_block_read_uc4.
        // But standard intel extension might not have it?
        // openvino_gemm.cl defines emulation if not present.
        // I will use `as_char4(intel_sub_group_block_read((const __global uint*)(weights + weights_idx)))` assuming 4-byte alignment.
        
        // UNPACK_TRANSPOSED_INT4 -> UNPACK_INT4x2_OSV32_ISV2
        // UNPACK_INT4x2_OSV32_ISV2 calls unpack_to_char_osv32_isv2.
        // INT4_PACKED_TYPE_PRELOAD = int4x8_t.
        // But wei_packed is char4 (4 bytes). int4x8_t is 4 bytes.
        // So cast is valid.
        char8 dq_wei_unpacked = unpack_to_char_osv32_isv2(*((int4x8_t *)&wei_packed));

        // FILTER_LOAD_BLOCK_SIZE = 4
        // SLM_WEIGHT_VEC = char4
        char4 wei_1 = (char4)(dq_wei_unpacked.s0, dq_wei_unpacked.s1, dq_wei_unpacked.s2, dq_wei_unpacked.s3);
        char_slm_weight[wei_local_idx] = as_uint(wei_1);
        char4 wei_2 = (char4)(dq_wei_unpacked.s4, dq_wei_unpacked.s5, dq_wei_unpacked.s6, dq_wei_unpacked.s7);
        char_slm_weight[wei_local_idx + 1] = as_uint(wei_2);

        wei_local_idx += 16 * 2;
        weights_idx += 16 * 4;
    }

    wei_local_idx = sglid * 2;
    barrier(CLK_LOCAL_MEM_FENCE);

    // TILE_IFM_ELEMENTS_SIZE / TILE_K = 32 / 4 = 8
    __attribute__((opencl_unroll_hint))
    for (uint ki = 0; ki < 8; ++ki) {
        // WEIGHT_VEC_TYPE = char8
        // vload8 from local memory
        char8 weight = vload8(0, (__local char *)(&char_slm_weight[wei_local_idx + 16 * 2 * ki]));
        char4 first_weight = (char4)(weight.s0, weight.s1, weight.s2, weight.s3);
        char4 second_weight = (char4)(weight.s4, weight.s5, weight.s6, weight.s7);

        __attribute__((opencl_unroll_hint))
        for (uint bi = 0; bi < 8; ++bi) {
            // MAKE_DQ_TYPE_VEC(4) -> char4
            // _sub_group_shuffle(char4, ...)
            char4 input_val = _sub_group_shuffle(as_char4(packed_in_0[bi / 2]), (bi % 2) * 8 + ki);
            
            // acc_tmp is int8 (vector of 8 ints).
            // acc_tmp[0] is int8? No. acc_tmp[TILE_OFM] where TILE_OFM=2.
            // acc_tmp is array of 2 int8 vectors.
            // acc_tmp[0][bi] access the bi-th element of first vector.
            
            // imad_SW(int, char4, char4)
            int val0 = acc_tmp[0][bi];
            val0 = imad_SW(val0, input_val, first_weight);
            acc_tmp[0][bi] = val0;

            int val1 = acc_tmp[1][bi];
            val1 = imad_SW(val1, input_val, second_weight);
            acc_tmp[1][bi] = val1;
        }

        weights_offset += 4 * 1 * 16;

        // DQ_DECOMPRESSION_SCALE_POST_OP = 1
        // TILE_IFM_ELEMENTS_SIZE (32) > DECOMPRESSION_SCALE_GROUP_SIZE (SIZE_QUANTIZATION_GROUP)
        // If SIZE_QUANTIZATION_GROUP < 32
#if 32 > SIZE_QUANTIZATION_GROUP
        __attribute__((opencl_unroll_hint))
        for (uint bi = 0; bi < 8; ++bi) {
            __attribute__((opencl_unroll_hint))
            for (uint fi = 0; fi < 2; ++fi) {
                const uint offset_ofm = out_f + fi * 16 + sglid;
#if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) * DECOMPRESSION_SCALE_BATCH_PITCH +
                    ((ni * 32 * 16 + ki * 4) / SIZE_QUANTIZATION_GROUP) * DECOMPRESSION_SCALE_FEATURE_PITCH;
                float ds = (float)decompression_scale[scale_offset];
#else
                float ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
#endif
                // COMPRESSED_WEIGHTS_INT8 = 0
                acc[bi][fi] += convert_half((float)acc_tmp[fi][bi] * ds * (float)de_quantize_scale[bi]);
                acc_tmp[fi][bi] = 0;
            }
        }
#endif
    }

    // TILE_IFM_ELEMENTS_SIZE (32) <= DECOMPRESSION_SCALE_GROUP_SIZE
#if 32 <= SIZE_QUANTIZATION_GROUP
    if ((ni % (SIZE_QUANTIZATION_GROUP / 32)) == ((SIZE_QUANTIZATION_GROUP / 32) - 1)) {
        const uint ni_offset = ((ni * 32 * 16) / SIZE_QUANTIZATION_GROUP) * DECOMPRESSION_SCALE_FEATURE_PITCH;
        __attribute__((opencl_unroll_hint))
        for (uint bi = 0; bi < 8; ++bi) {
            __attribute__((opencl_unroll_hint))
            for (uint fi = 0; fi < 2; ++fi) {
                const uint offset_ofm = out_f + fi * 16 + sglid;
#if DECOMPRESSION_SCALE_GROUPS_NUM > 1
                // SCALE_ROW_MAJOR = 0
                const uint scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) + ni_offset * ALIGN(SIZE_N, 32);
                float ds = (float)decompression_scale[scale_offset];
#else
                float ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
#endif
                acc[bi][fi] += convert_half((float)acc_tmp[fi][bi] * ds * (float)de_quantize_scale[bi]);
                acc_tmp[fi][bi] = 0;
            }
        }
    }
#endif

  } // Main loop

  // Post-processing
  for (uint bi = 0; bi < 8; ++bi) {
      activated[bi] = convert_float2(acc[bi]);
  }

  // BIAS_TERM = 0
  // HAS_FUSED_OPS = 0

  // Write results
  uint output_offset = out_f * 1 + out_b * SIZE_N + 0;

  // TILE_OUT_F_NUM = SIZE_N
  if ((SIZE_N % 32 == 0 || out_f + 32 <= SIZE_N)) {
      // CONST_LOOP(8, WRITE_OUTPUT)
      // Unroll 8 times
      #pragma unroll
      for(int bi=0; bi<8; ++bi) {
          // OUTPUT_BLOCK_WRITE(output, output_offset, result[bi])
          // BLOCK_WRITEN(half, 2, output, output_offset, result[bi])
          // intel_sub_group_block_write_us2
          intel_sub_group_block_write_us2((__global ushort *)output + output_offset, as_ushort2(convert_half2(activated[bi])));
          output_offset += SIZE_N;
      }
  } else {
      output_offset += sglid;
      for (uint bi = 0; bi < 8; ++bi) {
          for (uint fi = 0; fi < 2; ++fi) {
              bool should_write = (SIZE_N % 32 == 0 || out_f + fi * 16 + sglid < SIZE_N);
              if (should_write) {
                  output[output_offset] = ((half *)(&activated[bi]))[fi];
              }
              output_offset += 16;
          }
          output_offset += SIZE_N - 32;
      }
  }
}

__attribute__((intel_reqd_sub_group_size(16)))
kernel void fc_bf_tiled_kernel_default_nomacro(
  const __global half *input,
  const __global half *decompression_scale,
  __global half *output,
  const __global char *weights,
  __global char *quantized_input,
  __global half *quan_var,
  const int M) {
  
  __local uint dq_wei_local_mem[512];
  
  fc_bf_tiled_kernel_dyn_quan_nomacro(
      input,
      quantized_input,
      quan_var,
      decompression_scale,
      output,
      weights,
      dq_wei_local_mem,
      M
  );
}


#define OPTIONAL_SHAPE_INFO_ARG
#define OPTIONAL_SHAPE_INFO_TENSOR
#define COMPRESSED_WEIGHTS 1
#define COMPRESSED_WEIGHTS_INT4 1
#define FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2 1

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(v, l, u) MAX((l), MIN((v), (u)))

#define DECOMPRESSION_SCALE_TERM 1

#define TILE_IFM_ELEMENTS_SIZE 32

#define DECOMPRESSION_SCALE_FEATURE_PITCH 1

#define INPUT0_TYPE half
#define OUTPUT_TYPE half
#define FILTER_TYPE char
#define DECOMPRESSION_SCALE_TYPE half
#define OUTPUT_TYPE_SIZE 2
#define INPUT0_OFFSET 0
#define OUTPUT_OFFSET 0

#define ACCUMULATOR_TYPE float
#define ACTIVATION_TYPE float

#define INT4_PACKED_TYPE = int4x8_t
#define W_IDX (fi * TILE_K + kii)
#define TILE_OFM_PER_OSV_SIZE 1
#define W_DYN_QUAN_IDX (fi * TILE_K + kii)
#define USE_SLM 1
#define LWS_BATCHES 8
#define FILTER_LOAD_ITERS 1
#define FILTER_ACTUAL_LOAD_BLOCK_SIZE 4
#define INT4_PACKED_TYPE_PRELOAD int4x8_t
#define FILTER_LOAD_BLOCK_SIZE 4
#define FILTER_ELEMENTS_PER_LOAD 8
#define DYNAMIC_QUANTIZE 1
#define DQ_DECOMPRESSION_SCALE_POST_OP 1
#define PER_TOKEN_SIZE_DYN_QUANTIZE 0
#define INPUT_LOAD_SIZE 4
#define DQ_TYPE char
#define SIMD 16
#define TILE_B 8
#define HALF_TILE_B 4
#define TILE_OFM 2
#define TILE_IFM 2
#define TILE_K 4
#define TILE_K_OFM 8
#define TILE_K_OFM_PACKED 4
#define OUTER_OFM 1
#define DISPATCH_BSV 1
#define DISPATCH_FSV 1
#define REALIGN_FP16_OFFSET 0
#define TILE_OUT_F_PITCH 1

#define ACTIVATION_FUNC_TYPED(input, params) (input)
#define ACTIVATION_PARAMS_TYPED 0
#define ACTIVATION_TYPED(input, params) ACTIVATION_FUNC_TYPED(input, params)

#define CONST_LOOP_CALL(macro, idx) macro(idx)
#define CONST_LOOP_1(macro) CONST_LOOP_CALL(macro, 0)
#define CONST_LOOP_2(macro)                                                    \
  CONST_LOOP_1(macro);                                                         \
  CONST_LOOP_CALL(macro, 1)
#define CONST_LOOP_3(macro)                                                    \
  CONST_LOOP_2(macro);                                                         \
  CONST_LOOP_CALL(macro, 2)
#define CONST_LOOP_4(macro)                                                    \
  CONST_LOOP_3(macro);                                                         \
  CONST_LOOP_CALL(macro, 3)
#define CONST_LOOP_5(macro)                                                    \
  CONST_LOOP_4(macro);                                                         \
  CONST_LOOP_CALL(macro, 4)
#define CONST_LOOP_6(macro)                                                    \
  CONST_LOOP_5(macro);                                                         \
  CONST_LOOP_CALL(macro, 5)
#define CONST_LOOP_7(macro)                                                    \
  CONST_LOOP_6(macro);                                                         \
  CONST_LOOP_CALL(macro, 6)
#define CONST_LOOP_8(macro)                                                    \
  CONST_LOOP_7(macro);                                                         \
  CONST_LOOP_CALL(macro, 7)

#define CONST_LOOP(count, macro) CAT(CONST_LOOP_, count)(macro)

inline int imad_SW(int acc, uchar4 input, char4 weight)
  __attribute__((overloadable)) {
  acc += input[0] * weight[0];
  acc += input[1] * weight[1];
  acc += input[2] * weight[2];
  acc += input[3] * weight[3];
  return acc;
}

inline int imad_SW(int acc, char4 input, char4 weight)
  __attribute__((overloadable)) {
  acc += input[0] * weight[0];
  acc += input[1] * weight[1];
  acc += input[2] * weight[2];
  acc += input[3] * weight[3];
  return acc;
}

inline int imad_SW(int acc, char4 input, uchar4 weight)
  __attribute__((overloadable)) {
  acc += input[0] * weight[0];
  acc += input[1] * weight[1];
  acc += input[2] * weight[2];
  acc += input[3] * weight[3];
  return acc;
}

inline int imad_SW(int acc, uchar4 input, uchar4 weight)
  __attribute__((overloadable)) {
  acc += input[0] * weight[0];
  acc += input[1] * weight[1];
  acc += input[2] * weight[2];
  acc += input[3] * weight[3];
  return acc;
}

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#if !defined(cl_intel_subgroups) && defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define OFFSET_GLOBAL_PTR(elem_type, ptr, byte_offset)                         \
  ((__global elem_type *)((__global char *)(ptr) + (byte_offset)))
#define MULTIPLY_OFFSET(elem_type, byte_offset)                                \
  ((byte_offset) * sizeof(elem_type))

#if OPT_HINTS_SUPPORTED
#define ASSUME_HINT(x) __builtin_assume(x)
#else
#define ASSUME_HINT(x)                                                         \
  do {                                                                         \
  } while (0)
#endif

#define unroll_for __attribute__((opencl_unroll_hint)) for

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

// ====================================================================================================================
// TYPE_SIZE(type) - evaluates to size of "type" in bytes
// type [PP] - Must evaluate to non-vectorized type.
// ====================================================================================================================
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

// ====================================================================================================================
// BLOCK_READN(type, vector_size, ptr, offset)
//    - evaluates to intel_sub_group_block_read operation for specified "type"
//    and "vector size", reading
//      "vector_size" elements from memory starting at "ptr" + "offset"
//  For more details and description of intel_sub_group_block_read functions
//  please, refer to cl_intel_subgroups extension documentation.
//
// BLOCK_READN_SLM(type, vector_size, ptr, offset)
//    - performs same operation as BLOCK_READN, but with "ptr" being in __local
//    address space.
//
// type        [PP] - Must evaluate to non-vectorized type, ex. float, half,
// char, etc.. vector_size [PP] - Number of elements to read/write, ex 2 for
// intel_sub_group_block_read2. ptr              - Pointer to global memory
// where to read from/write to. offset           - Additional offset added to
// ptr in "type" elements, equivalent to passing ((ptr) + (offset)) as "ptr".
// val              - For write function vector of "vector_size" of "type"
// elements (or scalar) to write.
//
// ====================================================================================================================
// Pre-defined commonly used definitions:
//   DT_<tensor>_BLOCK_READ<n>(ptr, offset)
// Where:
//    <tensor> is one of: INPUT - referencing type jitted as INPUT0,
//                        BIAS,
//                        FILTER
//    <n> is a vector size, one of {2,4,8,16} or none, meaning the output will
//    be a scalar
//
// ====================================================================================================================

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

#define DT_INPUT_BLOCK_READ(ptr, offset)                                       \
  BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
#define DT_INPUT_BLOCK_READ2(ptr, offset)                                      \
  BLOCK_READN(INPUT0_TYPE, 2, ptr, offset)
#define DT_INPUT_BLOCK_READ4(ptr, offset)                                      \
  BLOCK_READN(INPUT0_TYPE, 4, ptr, offset)
#define DT_INPUT_BLOCK_READ8(ptr, offset)                                      \
  BLOCK_READN(INPUT0_TYPE, 8, ptr, offset)
#define DT_INPUT_BLOCK_READ16(ptr, offset)                                     \
  BLOCK_READN(INPUT0_TYPE, 16, ptr, offset)

#define DT_BIAS_BLOCK_READ(ptr, offset) BLOCK_READN(BIAS_TYPE, 1, ptr, offset)
#define DT_BIAS_BLOCK_READ2(ptr, offset) BLOCK_READN(BIAS_TYPE, 2, ptr, offset)
#define DT_BIAS_BLOCK_READ4(ptr, offset) BLOCK_READN(BIAS_TYPE, 4, ptr, offset)
#define DT_BIAS_BLOCK_READ8(ptr, offset) BLOCK_READN(BIAS_TYPE, 8, ptr, offset)
#define DT_BIAS_BLOCK_READ16(ptr, offset)                                      \
  BLOCK_READN(BIAS_TYPE, 16, ptr, offset)

#define DT_FILTER_BLOCK_READ(ptr, offset)                                      \
  BLOCK_READN(FILTER_TYPE, 1, ptr, offset)
#define DT_FILTER_BLOCK_READ2(ptr, offset)                                     \
  BLOCK_READN(FILTER_TYPE, 2, ptr, offset)
#define DT_FILTER_BLOCK_READ4(ptr, offset)                                     \
  BLOCK_READN(FILTER_TYPE, 4, ptr, offset)
#define DT_FILTER_BLOCK_READ8(ptr, offset)                                     \
  BLOCK_READN(FILTER_TYPE, 8, ptr, offset)
#define DT_FILTER_BLOCK_READ16(ptr, offset)                                    \
  BLOCK_READN(FILTER_TYPE, 16, ptr, offset)

#define BLOCK_READ_IMPL_1 ret = ptr[idx];

#define BLOCK_READ_IMPL_2                                                      \
  ret.s0 = ptr[idx];                                                           \
  idx += get_max_sub_group_size();                                             \
  ret.s1 = ptr[idx];                                                           \
  idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL_4                                                      \
  BLOCK_READ_IMPL_2                                                            \
  ret.s2 = ptr[idx];                                                           \
  idx += get_max_sub_group_size();                                             \
  ret.s3 = ptr[idx];                                                           \
  idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL_8                                                      \
  BLOCK_READ_IMPL_4                                                            \
  ret.s4 = ptr[idx];                                                           \
  idx += get_max_sub_group_size();                                             \
  ret.s5 = ptr[idx];                                                           \
  idx += get_max_sub_group_size();                                             \
  ret.s6 = ptr[idx];                                                           \
  idx += get_max_sub_group_size();                                             \
  ret.s7 = ptr[idx];                                                           \
  idx += get_max_sub_group_size();

#define BLOCK_READ_IMPL_16                                                     \
  BLOCK_READ_IMPL_8                                                            \
  ret.s8 = ptr[idx];                                                           \
  idx += get_max_sub_group_size();                                             \
  ret.s9 = ptr[idx];                                                           \
  idx += get_max_sub_group_size();                                             \
  ret.sa = ptr[idx];                                                           \
  idx += get_max_sub_group_size();                                             \
  ret.sb = ptr[idx];                                                           \
  idx += get_max_sub_group_size();                                             \
  ret.sc = ptr[idx];                                                           \
  idx += get_max_sub_group_size();                                             \
  ret.sd = ptr[idx];                                                           \
  idx += get_max_sub_group_size();                                             \
  ret.se = ptr[idx];                                                           \
  idx += get_max_sub_group_size();                                             \
  ret.sf = ptr[idx];                                                           \
  idx += get_max_sub_group_size();

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

#if defined(cl_intel_subgroups)
#define _sub_group_block_read(ptr) intel_sub_group_block_read(ptr)
#define _sub_group_block_read2(ptr) intel_sub_group_block_read2(ptr)
#define _sub_group_block_read4(ptr) intel_sub_group_block_read4(ptr)
#define _sub_group_block_read8(ptr) intel_sub_group_block_read8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
DECLARE_BLOCK_READ_EMULATION(4, 1)
DECLARE_BLOCK_READ_EMULATION(4, 2)
DECLARE_BLOCK_READ_EMULATION(4, 4)
DECLARE_BLOCK_READ_EMULATION(4, 8)
#endif

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

#if defined(cl_intel_subgroups_long)
#define _sub_group_block_read_ul(ptr) intel_sub_group_block_read_ul(ptr)
#define _sub_group_block_read_ul2(ptr) intel_sub_group_block_read_ul2(ptr)
#define _sub_group_block_read_ul4(ptr) intel_sub_group_block_read_ul4(ptr)
#define _sub_group_block_read_ul8(ptr) intel_sub_group_block_read_ul8(ptr)
#elif (__OPENCL_C_VERSION__ >= 200)
DECLARE_BLOCK_READ_EMULATION(8, 1)
DECLARE_BLOCK_READ_EMULATION(8, 2)
DECLARE_BLOCK_READ_EMULATION(8, 4)
DECLARE_BLOCK_READ_EMULATION(8, 8)
#endif

// ====================================================================================================================
// BLOCK_WRITEN(type, vector_size, ptr, offset, val)
//    - evaluates to intel_sub_group_block_write operation for specified "type"
//    and "vector size", writing
//      "vector_size"-element vector "val" to memory starting at "ptr" +
//      "offset"
//  For more details and description of intel_sub_group_block_read/write
//  functions please, refer to cl_intel_subgroups extension documentation.
//
// BLOCK_WRITEN_SLM(type, vector_size, ptr, offset, val)
//    - performs same operation as BLOCK_READN, but with "ptr" being in __local
//    address space.
//
// type        [PP] - Must evaluate to non-vectorized type, ex. float, half,
// char, etc.. vector_size [PP] - Number of elements to read/write, ex 2 for
// intel_sub_group_block_read2. ptr              - Pointer to global memory
// where to read from/write to. offset           - Additional offset added to
// ptr in "type" elements, equivalent to passing ((ptr) + (offset)) as "ptr".
// val              - For write function vector of "vector_size" of "type"
// elements (or scalar) to write.
//
// ====================================================================================================================
// Pre-defined commonly used definitions:
//   DT_<tensor>_BLOCK_WRITE<n>(ptr, offset, offset)
// Where:
//    <tensor> is usually OUTPUT,
//    <n> is a vector size, one of {2,4,8,16} or none, meaning the output will
//    be a scalar
//
// ====================================================================================================================

#define BLOCK_WRITE_TYPE_size1 uchar
#define BLOCK_WRITE_TYPE_size2 ushort
#define BLOCK_WRITE_TYPE_size4 uint
#define BLOCK_WRITE_TYPE_size8 ulong
#define BLOCK_WRITE_TYPE(type_size) CAT(BLOCK_WRITE_TYPE_size, type_size)

#define BLOCK_WRITE_FUNC_size1 _sub_group_block_write_uc
#define BLOCK_WRITE_FUNC_size2 _sub_group_block_write_us
#define BLOCK_WRITE_FUNC_size4 _sub_group_block_write
#define BLOCK_WRITE_FUNC_size8 _sub_group_block_write_ul
#define BLOCK_WRITE_FUNC(type_size) CAT(BLOCK_WRITE_FUNC_size, type_size)

#define BLOCK_WRITEN_FUNC_SIZE_DEF(type_size, vector_size)                     \
  MAKE_VECTOR_TYPE(BLOCK_WRITE_FUNC(type_size), vector_size)
#define BLOCK_WRITEN_FUNC_size1(vector_size)                                   \
  BLOCK_WRITEN_FUNC_SIZE_DEF(1, vector_size)
#define BLOCK_WRITEN_FUNC_size2(vector_size)                                   \
  BLOCK_WRITEN_FUNC_SIZE_DEF(2, vector_size)
#define BLOCK_WRITEN_FUNC_size4(vector_size)                                   \
  BLOCK_WRITEN_FUNC_SIZE_DEF(4, vector_size)
#define BLOCK_WRITEN_FUNC_size8(vector_size)                                   \
  BLOCK_WRITEN_FUNC_SIZE_DEF(8, vector_size)
#define BLOCK_WRITEN_FUNC(type_size, vector_size)                              \
  CAT(BLOCK_WRITEN_FUNC_size, type_size)(vector_size)

#define BLOCK_WRITEN_RAW(type_size, vector_size, addr_space, ptr, offset, val) \
  BLOCK_WRITEN_FUNC(type_size, vector_size)                                    \
  ((addr_space BLOCK_WRITE_TYPE(type_size) *)(ptr) + (offset),                 \
   AS_TYPE(MAKE_VECTOR_TYPE(BLOCK_WRITE_TYPE(type_size), vector_size), val))

#define BLOCK_WRITEN(type, vector_size, ptr, offset, val)                      \
  BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __global, ptr, offset, val)

#define BLOCK_WRITEN_SLM(type, vector_size, ptr, offset, val)                  \
  BLOCK_WRITEN_RAW(TYPE_SIZE(type), vector_size, __local, ptr, offset, val)

#define DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)                                \
  BLOCK_WRITEN(OUTPUT_TYPE, 1, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE2(ptr, offset, val)                               \
  BLOCK_WRITEN(OUTPUT_TYPE, 2, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE4(ptr, offset, val)                               \
  BLOCK_WRITEN(OUTPUT_TYPE, 4, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE8(ptr, offset, val)                               \
  BLOCK_WRITEN(OUTPUT_TYPE, 8, ptr, offset, val)
#define DT_OUTPUT_BLOCK_WRITE16(ptr, offset, val)                              \
  BLOCK_WRITEN(OUTPUT_TYPE, 16, ptr, offset, val)

#define BLOCK_WRITE_IMPL_1 out_ptr[idx] = v;
#define BLOCK_WRITE_IMPL_2                                                     \
  out_ptr[idx] = v.s0;                                                         \
  idx += get_max_sub_group_size();                                             \
  out_ptr[idx] = v.s1;                                                         \
  idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_4                                                     \
  BLOCK_WRITE_IMPL_2                                                           \
  out_ptr[idx] = v.s2;                                                         \
  idx += get_max_sub_group_size();                                             \
  out_ptr[idx] = v.s3;                                                         \
  idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_8                                                     \
  BLOCK_WRITE_IMPL_4                                                           \
  out_ptr[idx] = v.s4;                                                         \
  idx += get_max_sub_group_size();                                             \
  out_ptr[idx] = v.s5;                                                         \
  idx += get_max_sub_group_size();                                             \
  out_ptr[idx] = v.s6;                                                         \
  idx += get_max_sub_group_size();                                             \
  out_ptr[idx] = v.s7;                                                         \
  idx += get_max_sub_group_size();
#define BLOCK_WRITE_IMPL_16                                                    \
  BLOCK_WRITE_IMPL_8                                                           \
  out_ptr[idx] = v.s8;                                                         \
  idx += get_max_sub_group_size();                                             \
  out_ptr[idx] = v.s9;                                                         \
  idx += get_max_sub_group_size();                                             \
  out_ptr[idx] = v.sa;                                                         \
  idx += get_max_sub_group_size();                                             \
  out_ptr[idx] = v.sb;                                                         \
  idx += get_max_sub_group_size();                                             \
  out_ptr[idx] = v.sc;                                                         \
  idx += get_max_sub_group_size();                                             \
  out_ptr[idx] = v.sd;                                                         \
  idx += get_max_sub_group_size();                                             \
  out_ptr[idx] = v.se;                                                         \
  idx += get_max_sub_group_size();                                             \
  out_ptr[idx] = v.sf;                                                         \
  idx += get_max_sub_group_size();

#define BLOCK_WRITE_IMPL(vec_size) CAT(BLOCK_WRITE_IMPL_, vec_size)
#define BLOCK_WRITE_FUNC_NAME(type_size, vec_size)                             \
  MAKE_VECTOR_TYPE(BLOCK_WRITE_FUNC(type_size), vec_size)
#define DECLARE_BLOCK_WRITE_EMULATION(type_size, vec_size)                     \
  inline void BLOCK_WRITE_FUNC_NAME(type_size, vec_size)(                      \
    __global BLOCK_WRITE_TYPE(type_size) * out_ptr,                            \
    MAKE_VECTOR_TYPE(BLOCK_WRITE_TYPE(type_size), vec_size) v) {               \
    uint idx = get_sub_group_local_id();                                       \
    BLOCK_WRITE_IMPL(vec_size)                                                 \
  }

#if defined(cl_intel_subgroups)
#define _sub_group_block_write(ptr, v) intel_sub_group_block_write(ptr, v)
#define _sub_group_block_write2(ptr, v) intel_sub_group_block_write2(ptr, v)
#define _sub_group_block_write4(ptr, v) intel_sub_group_block_write4(ptr, v)
#define _sub_group_block_write8(ptr, v) intel_sub_group_block_write8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
DECLARE_BLOCK_WRITE_EMULATION(4, 1)
DECLARE_BLOCK_WRITE_EMULATION(4, 2)
DECLARE_BLOCK_WRITE_EMULATION(4, 4)
DECLARE_BLOCK_WRITE_EMULATION(4, 8)
#endif

#if defined(cl_intel_subgroups_short)
#define _sub_group_block_write_us(ptr, v) intel_sub_group_block_write_us(ptr, v)
#define _sub_group_block_write_us2(ptr, v)                                     \
  intel_sub_group_block_write_us2(ptr, v)
#define _sub_group_block_write_us4(ptr, v)                                     \
  intel_sub_group_block_write_us4(ptr, v)
#define _sub_group_block_write_us8(ptr, v)                                     \
  intel_sub_group_block_write_us8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
DECLARE_BLOCK_WRITE_EMULATION(2, 1)
DECLARE_BLOCK_WRITE_EMULATION(2, 2)
DECLARE_BLOCK_WRITE_EMULATION(2, 4)
DECLARE_BLOCK_WRITE_EMULATION(2, 8)
#endif

#if defined(cl_intel_subgroups_char)
#define _sub_group_block_write_uc(ptr, v) intel_sub_group_block_write_uc(ptr, v)
#define _sub_group_block_write_uc2(ptr, v)                                     \
  intel_sub_group_block_write_uc2(ptr, v)
#define _sub_group_block_write_uc4(ptr, v)                                     \
  intel_sub_group_block_write_uc4(ptr, v)
#define _sub_group_block_write_uc8(ptr, v)                                     \
  intel_sub_group_block_write_uc8(ptr, v)
#define _sub_group_block_write_uc16(ptr, v)                                    \
  intel_sub_group_block_write_uc16(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
DECLARE_BLOCK_WRITE_EMULATION(1, 1)
DECLARE_BLOCK_WRITE_EMULATION(1, 2)
DECLARE_BLOCK_WRITE_EMULATION(1, 4)
DECLARE_BLOCK_WRITE_EMULATION(1, 8)
DECLARE_BLOCK_WRITE_EMULATION(1, 16)
#endif

#if defined(cl_intel_subgroups_long)
#define _sub_group_block_write_ul(ptr, v) intel_sub_group_block_write_ul(ptr, v)
#define _sub_group_block_write_ul2(ptr, v)                                     \
  intel_sub_group_block_write_ul2(ptr, v)
#define _sub_group_block_write_ul4(ptr, v)                                     \
  intel_sub_group_block_write_ul4(ptr, v)
#define _sub_group_block_write_ul8(ptr, v)                                     \
  intel_sub_group_block_write_ul8(ptr, v)
#elif (__OPENCL_C_VERSION__ >= 200)
DECLARE_BLOCK_WRITE_EMULATION(8, 1)
DECLARE_BLOCK_WRITE_EMULATION(8, 2)
DECLARE_BLOCK_WRITE_EMULATION(8, 4)
DECLARE_BLOCK_WRITE_EMULATION(8, 8)
#endif

#ifdef cl_intel_subgroups
#define _sub_group_shuffle(v, c) intel_sub_group_shuffle(v, c)
#define _sub_group_shuffle_up(c, n, d) intel_sub_group_shuffle_up(c, n, d)
#define _sub_group_shuffle_down(c, n, d) intel_sub_group_shuffle_down(c, n, d)
#elif (__OPENCL_C_VERSION__ >= 200)

// The spec for intel_subgroup_shuffle says that index (c) need not be the same
// value for all work-items in a subgroup while sub_group_broadcast requires
// that. However, most of our kernels uses shuffle in a way that produces same
// index for all work-items, so for now we use this solution. In case of
// accuracy issues we may switch to something like this: #define MAX_SG_SIZE 32
// #define DECLARE_SUB_GROUP_SHUFFLE1(type, cast_type)
// inline type _sub_group_shuffle(type v, uint c) __attribute__((overloadable))
// {
//     type vals[MAX_SG_SIZE];
//     for (size_t i = 0; i < get_max_sub_group_size(); i++) {
//         vals[i] = AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v),
//         i));
//     }
//     return vals[c];
// }

#define DECLARE_SUB_GROUP_SHUFFLE1(type, cast_type)                            \
  inline type _sub_group_shuffle(type v, uint c)                               \
    __attribute__((overloadable)) {                                            \
    return AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v), c));       \
  }

#define DECLARE_SUB_GROUP_SHUFFLE2(type, cast_type)                            \
  inline CAT(type, 2) _sub_group_shuffle(CAT(type, 2) v, uint c)               \
    __attribute__((overloadable)) {                                            \
    return (CAT(type, 2))(                                                     \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)));        \
  }

#define DECLARE_SUB_GROUP_SHUFFLE4(type, cast_type)                            \
  inline CAT(type, 4) _sub_group_shuffle(CAT(type, 4) v, uint c)               \
    __attribute__((overloadable)) {                                            \
    return (CAT(type, 4))(                                                     \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s2), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s3), c)));        \
  }

#define DECLARE_SUB_GROUP_SHUFFLE8(type, cast_type)                            \
  inline CAT(type, 8) _sub_group_shuffle(CAT(type, 8) v, uint c)               \
    __attribute__((overloadable)) {                                            \
    return (CAT(type, 8))(                                                     \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s2), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s3), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s4), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s5), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s6), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s7), c)));        \
  }

#define DECLARE_SUB_GROUP_SHUFFLE16(type, cast_type)                           \
  inline CAT(type, 16) _sub_group_shuffle(CAT(type, 16) v, uint c)             \
    __attribute__((overloadable)) {                                            \
    return (CAT(type, 16))(                                                    \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s0), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s1), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s2), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s3), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s4), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s5), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s6), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s7), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s8), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.s9), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sa), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sb), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sc), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sd), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.se), c)),         \
      AS_TYPE(type, sub_group_broadcast(AS_TYPE(cast_type, v.sf), c)));        \
  }

#define DECLARE_SUB_GROUP_SHUFFLE(type)                                        \
  DECLARE_SUB_GROUP_SHUFFLE1(type, type)                                       \
  DECLARE_SUB_GROUP_SHUFFLE2(type, type)                                       \
  DECLARE_SUB_GROUP_SHUFFLE4(type, type)                                       \
  DECLARE_SUB_GROUP_SHUFFLE8(type, type)                                       \
  DECLARE_SUB_GROUP_SHUFFLE16(type, type)

#define DECLARE_SUB_GROUP_SHUFFLE_CASTED(type, cast_type)                      \
  DECLARE_SUB_GROUP_SHUFFLE1(type, cast_type)                                  \
  DECLARE_SUB_GROUP_SHUFFLE2(type, cast_type)                                  \
  DECLARE_SUB_GROUP_SHUFFLE4(type, cast_type)                                  \
  DECLARE_SUB_GROUP_SHUFFLE8(type, cast_type)                                  \
  DECLARE_SUB_GROUP_SHUFFLE16(type, cast_type)

DECLARE_SUB_GROUP_SHUFFLE(int)
DECLARE_SUB_GROUP_SHUFFLE(uint)
DECLARE_SUB_GROUP_SHUFFLE(float)

#if defined(cl_khr_fp16)
DECLARE_SUB_GROUP_SHUFFLE(half)
DECLARE_SUB_GROUP_SHUFFLE_CASTED(short, half)
DECLARE_SUB_GROUP_SHUFFLE_CASTED(ushort, half)
#endif

#endif

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
typedef struct __attribute__((packed)) int4x16_t {
  int4x2_t s0;
  int4x2_t s1;
  int4x2_t s2;
  int4x2_t s3;
  int4x2_t s4;
  int4x2_t s5;
  int4x2_t s6;
  int4x2_t s7;
} int4x16_t;

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
typedef struct __attribute__((packed)) uint4x16_t {
  uint4x2_t s0;
  uint4x2_t s1;
  uint4x2_t s2;
  uint4x2_t s3;
  uint4x2_t s4;
  uint4x2_t s5;
  uint4x2_t s6;
  uint4x2_t s7;
} uint4x16_t;

inline uchar2 cvt_uint4x2_to_uint8x2(uint4x2_t v)
  __attribute__((overloadable)) {
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

// 4bit x 4
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

inline char4 unpack_transposed_to_char(int4x4_t v)
  __attribute__((overloadable)) {
  char2 v0 = unpack_to_char(v.s0);
  char2 v1 = unpack_to_char(v.s1);
  return (char4)(v0.s0, v1.s0, v0.s1, v1.s1);
}

inline char4 unpack_transposed_to_char(uint4x4_t v)
  __attribute__((overloadable)) {
  char2 v0 = unpack_to_char(v.s0);
  char2 v1 = unpack_to_char(v.s1);
  return (char4)(v0.s0, v1.s0, v0.s1, v1.s1);
}

inline uchar4 unpack_transposed_to_uchar(uint4x4_t v)
  __attribute__((overloadable)) {
  uchar2 v0 = unpack_to_uchar(v.s0);
  uchar2 v1 = unpack_to_uchar(v.s1);
  return (uchar4)(v0.s0, v1.s0, v0.s1, v1.s1);
}

// 4bit x 8
inline uchar8 unpack_to_uchar(uint4x8_t v) __attribute__((overloadable)) {
  uchar2 v0 = unpack_to_uchar(v.s0);
  uchar2 v1 = unpack_to_uchar(v.s1);
  uchar2 v2 = unpack_to_uchar(v.s2);
  uchar2 v3 = unpack_to_uchar(v.s3);
  return (uchar8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}

inline char8 unpack_to_char(int4x8_t v) __attribute__((overloadable)) {
  char2 v0 = unpack_to_char(v.s0);
  char2 v1 = unpack_to_char(v.s1);
  char2 v2 = unpack_to_char(v.s2);
  char2 v3 = unpack_to_char(v.s3);
  return (char8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}

inline char8 unpack_to_char(uint4x8_t v) __attribute__((overloadable)) {
  char2 v0 = unpack_to_char(v.s0);
  char2 v1 = unpack_to_char(v.s1);
  char2 v2 = unpack_to_char(v.s2);
  char2 v3 = unpack_to_char(v.s3);
  return (char8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}

inline char8 unpack_transposed_to_char(int4x8_t v)
  __attribute__((overloadable)) {
  char2 v0 = unpack_to_char(v.s0);
  char2 v1 = unpack_to_char(v.s1);
  char2 v2 = unpack_to_char(v.s2);
  char2 v3 = unpack_to_char(v.s3);
  return (char8)(v0.s0, v1.s0, v2.s0, v3.s0, v0.s1, v1.s1, v2.s1, v3.s1);
}

inline char8 unpack_transposed_to_char(uint4x8_t v)
  __attribute__((overloadable)) {
  char2 v0 = unpack_to_char(v.s0);
  char2 v1 = unpack_to_char(v.s1);
  char2 v2 = unpack_to_char(v.s2);
  char2 v3 = unpack_to_char(v.s3);
  return (char8)(v0.s0, v1.s0, v2.s0, v3.s0, v0.s1, v1.s1, v2.s1, v3.s1);
}

inline uchar8 unpack_transposed_to_uchar(uint4x8_t v)
  __attribute__((overloadable)) {
  uchar2 v0 = unpack_to_uchar(v.s0);
  uchar2 v1 = unpack_to_uchar(v.s1);
  uchar2 v2 = unpack_to_uchar(v.s2);
  uchar2 v3 = unpack_to_uchar(v.s3);
  return (uchar8)(v0.s0, v1.s0, v2.s0, v3.s0, v0.s1, v1.s1, v2.s1, v3.s1);
}

// For float
inline float2 unpack_to_float(uint4x2_t v) __attribute__((overloadable)) {
  return convert_float2(cvt_uint4x2_to_uint8x2(v));
}

inline float2 unpack_to_float(int4x2_t v) __attribute__((overloadable)) {
  return convert_float2(cvt_int4x2_to_int8x2(v));
}

inline float4 unpack_to_float(uint4x4_t v) __attribute__((overloadable)) {
  float2 f0 = unpack_to_float(v.s0);
  float2 f1 = unpack_to_float(v.s1);
  return (float4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline float4 unpack_to_float(int4x4_t v) __attribute__((overloadable)) {
  float2 f0 = unpack_to_float(v.s0);
  float2 f1 = unpack_to_float(v.s1);
  return (float4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline float8 unpack_to_float(uint4x8_t v) __attribute__((overloadable)) {
  float2 f0 = unpack_to_float(v.s0);
  float2 f1 = unpack_to_float(v.s1);
  float2 f2 = unpack_to_float(v.s2);
  float2 f3 = unpack_to_float(v.s3);
  return (float8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

inline float8 unpack_to_float(int4x8_t v) __attribute__((overloadable)) {
  float2 f0 = unpack_to_float(v.s0);
  float2 f1 = unpack_to_float(v.s1);
  float2 f2 = unpack_to_float(v.s2);
  float2 f3 = unpack_to_float(v.s3);
  return (float8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

#if defined(cl_khr_fp16)
inline half2 unpack_to_half(uint4x2_t v) __attribute__((overloadable)) {
  return convert_half2(cvt_uint4x2_to_uint8x2(v));
}

inline half2 unpack_to_half(int4x2_t v) __attribute__((overloadable)) {
  return convert_half2(cvt_int4x2_to_int8x2(v));
}

inline half4 unpack_to_half(uint4x4_t v) __attribute__((overloadable)) {
  half2 f0 = unpack_to_half(v.s0);
  half2 f1 = unpack_to_half(v.s1);
  return (half4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline half4 unpack_to_half_osv32_isv2(uint4x4_t v)
  __attribute__((overloadable)) {
  half2 f0 = unpack_to_half(v.s0);
  half2 f1 = unpack_to_half(v.s1);
  return (half4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline half4 unpack_to_half(int4x4_t v) __attribute__((overloadable)) {
  half2 f0 = unpack_to_half(v.s0);
  half2 f1 = unpack_to_half(v.s1);
  return (half4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline half4 unpack_to_half_osv32_isv2(int4x4_t v)
  __attribute__((overloadable)) {
  half2 f0 = unpack_to_half(v.s0);
  half2 f1 = unpack_to_half(v.s1);
  return (half4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline half8 unpack_to_half(uint4x8_t v) __attribute__((overloadable)) {
  half2 f0 = unpack_to_half(v.s0);
  half2 f1 = unpack_to_half(v.s1);
  half2 f2 = unpack_to_half(v.s2);
  half2 f3 = unpack_to_half(v.s3);
  return (half8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

inline half8 unpack_to_half_osv32_isv2(uint4x8_t v)
  __attribute__((overloadable)) {
  half2 f0 = unpack_to_half(v.s0);
  half2 f1 = unpack_to_half(v.s2);
  half2 f2 = unpack_to_half(v.s1);
  half2 f3 = unpack_to_half(v.s3);
  return (half8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

inline half8 unpack_to_half(int4x8_t v) __attribute__((overloadable)) {
  half2 f0 = unpack_to_half(v.s0);
  half2 f1 = unpack_to_half(v.s1);
  half2 f2 = unpack_to_half(v.s2);
  half2 f3 = unpack_to_half(v.s3);
  return (half8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

inline half8 unpack_to_half_osv32_isv2(int4x8_t v)
  __attribute__((overloadable)) {
  half2 f0 = unpack_to_half(v.s0);
  half2 f1 = unpack_to_half(v.s2);
  half2 f2 = unpack_to_half(v.s1);
  half2 f3 = unpack_to_half(v.s3);
  return (half8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

inline char8 unpack_to_char_osv32_isv2(int4x8_t v)
  __attribute__((overloadable)) {
  char2 v0 = unpack_to_char(v.s0);
  char2 v1 = unpack_to_char(v.s2);
  char2 v2 = unpack_to_char(v.s1);
  char2 v3 = unpack_to_char(v.s3);
  return (char8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}

inline char8 unpack_to_char_osv32_isv2(uint4x8_t v)
  __attribute__((overloadable)) {
  char2 v0 = unpack_to_char(v.s0);
  char2 v1 = unpack_to_char(v.s2);
  char2 v2 = unpack_to_char(v.s1);
  char2 v3 = unpack_to_char(v.s3);
  return (char8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}

inline uchar8 unpack_to_uchar_osv32_isv2(uint4x8_t v)
  __attribute__((overloadable)) {
  uchar2 v0 = unpack_to_uchar(v.s0);
  uchar2 v1 = unpack_to_uchar(v.s2);
  uchar2 v2 = unpack_to_uchar(v.s1);
  uchar2 v3 = unpack_to_uchar(v.s3);
  return (uchar8)(v0.s0, v0.s1, v1.s0, v1.s1, v2.s0, v2.s1, v3.s0, v3.s1);
}

#endif // defined(cl_khr_fp16)

#define UNPACK_INT4x2(target_type, value) CAT(unpack_to_, target_type)(value)
#define UNPACK_INT4x2_OSV32_ISV2(target_type, value)                           \
  CAT(CAT(unpack_to_, target_type), _osv32_isv2)(value)
#define UNPACK_INT4x4_OSV32_ISV2(target_type, value)                           \
  CAT(CAT(unpack_to_, target_type), _osv32_isv2)(value)
#define UNPACK_TRANSPOSED_INT4x2(target_type, value)                           \
  CAT(unpack_transposed_to_, target_type)(value)

// JIT Parameters:
// SIMD         - sub-group size/simd width, one of {8, 16};
// TILE_B       - number of batches processed by each work-item;
// TILE_OFM     - number of output features calculated by work-item, one of {1,
// 2, 4, 8}; TILE_IFM     - number of input features loaded from input by
// work-item, one of {1, 2, 4, 8}; TILE_K       - number of input features
// loaded from weights, one of {1, 2, 4, 8}; TILE_K_OFM   - must be equal to
// TILE_OFM * TILE_K and less or equal to 8; DISPATCH_FSV - output coordinates
// for each sub-group are calculated from linearized coordinates DISPATCH_BSV as
// if they laid in bs_fs_bsv_fsv format, these macros describe fsv and bsv
// factors;

kernel void quantize_input(const __global INPUT0_TYPE *input,
                           __global DQ_TYPE *quantized_input,
                           __global INPUT0_TYPE *quan_var, const int size_n,
                           const int size_k,
                           const int quantization_group_size) {
  const uint offset = get_global_id(0);

  const uint input_offset = offset * quantization_group_size;
  const uint quantize_block = quantization_group_size / INPUT_LOAD_SIZE;
  MAKE_VECTOR_TYPE(INPUT0_TYPE, INPUT_LOAD_SIZE) input_0;
  MAKE_VECTOR_TYPE(DQ_TYPE, INPUT_LOAD_SIZE) quantized_value;
  INPUT0_TYPE max_vals[32]; // MAX_QUANTIZATION_GROUP_SIZE / INPUT_LOAD_SIZE =
                            // 128 / 4 = 32

  for (uint i = 0; i < quantize_block; ++i) {
    input_0 = vload4(0, &input[input_offset + i * 4]);
    max_vals[i] = fmax(fmax(fabs(input_0[0]), fabs(input_0[1])),
                       fmax(fabs(input_0[2]), fabs(input_0[3])));
  }

  INPUT0_TYPE max_value = 0.001h;
  for (uint i = 0; i < quantize_block; i += 8) {
    INPUT0_TYPE temp = fmax(fmax(fmax(max_vals[i], max_vals[i + 1]),
                                 fmax(max_vals[i + 2], max_vals[i + 3])),
                            fmax(fmax(max_vals[i + 4], max_vals[i + 5]),
                                 fmax(max_vals[i + 6], max_vals[i + 7])));
    max_value = fmax(max_value, temp);
  }

  float quan_scale = (float)max_value / 127.f;
#if COMPRESSED_WEIGHTS_INT8
  int quantized_sum = 0;
#endif
  for (uint i = 0; i < quantize_block; ++i) {
    input_0 = vload4(0, &input[input_offset + i * 4]);
    float4 buff = convert_float4(input_0) / quan_scale;
    quantized_value = CAT(
      CAT(convert_, MAKE_VECTOR_TYPE(DQ_TYPE, INPUT_LOAD_SIZE)), _rte)(buff);
#if COMPRESSED_WEIGHTS_INT8
    quantized_sum += quantized_value[0] + quantized_value[1] +
                     quantized_value[2] + quantized_value[3];
#endif
    vstore4(quantized_value, 0, &quantized_input[input_offset + i * 4]);
  }

  // Pair of quantizing_scale and quantized activation_sum for each group
  quan_var[offset * 2] = convert_half(quan_scale);
#if COMPRESSED_WEIGHTS_INT8
  quan_var[(offset * 2) + 1] = convert_half(quantized_sum);
#endif
}

// Verify JIT parameters.
#if SIMD != 8 && SIMD != 16
#error "fully_connected_gpu_bf_tiled.cl - SIMD must be one of {8, 16}"
#endif

#if TILE_OFM != 1 && TILE_OFM != 2 && TILE_OFM != 4 && TILE_OFM != 8
#error "fully_connected_gpu_bf_tiled.cl - TILE_OFM must be one of {1, 2, 4, 8}"
#endif

#if TILE_IFM != 1 && TILE_IFM != 2 && TILE_IFM != 4 && TILE_IFM != 8
#error "fully_connected_gpu_bf_tiled.cl - TILE_IFM must be one of {1, 2, 4, 8}"
#endif

#if TILE_K != 1 && TILE_K != 2 && TILE_K != 4 && TILE_K != 8
#error "fully_connected_gpu_bf_tiled.cl - TILE_K must be one of {1, 2, 4, 8}"
#endif

#if TILE_K_OFM != (TILE_K * TILE_OFM) || TILE_K_OFM > 8
#error                                                                         \
  "fully_connected_gpu_bf_tiled.cl - TILE_K_OFM must be equal to TILE_K * TILE_OFM and at most 8"
#endif

#if COMPRESSED_WEIGHTS_INT4
#if TILE_K_OFM != TILE_K_OFM_PACKED * 2
#error                                                                         \
  "fully_connected_gpu_bf_tiled.cl - TILE_K_OFM must be divisible by 2 for 4-bit compressed case"
#endif
#if FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2 && TILE_K != 4 && TILE_K != 2 &&         \
  TILE_K != 1
#error "fully_connected_gpu_bf_tiled.cl - TILE_K must be one of {1, 2, 4}"
#endif
#endif

#if TILE_K == 4 && COMPRESSED_WEIGHTS_INT4 && FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2
// Data stored in memory : f0k0k1|f16k0k1|f0k2k3|f16k2k3
// => unpack as f0k0k1|f0k2k3|f16k0k1|f16k2k3 so that the weight access order is
// preserved
#define UNPACK_INT4 UNPACK_INT4x2_OSV32_ISV2
// No need to apply transpose for dynamic quantizing. Weight values are located
// in order of tile_k : f0(k0,k1),f1(k2,k3)
#define UNPACK_TRANSPOSED_INT4 UNPACK_INT4x2_OSV32_ISV2
#else
#define UNPACK_INT4 UNPACK_INT4x2
#define UNPACK_TRANSPOSED_INT4 UNPACK_TRANSPOSED_INT4x2
#endif
// Macros for vectorized types.
#define INPUT_VEC_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, TILE_IFM)
#define ACCUMULATOR_VEC_TYPE MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_OFM)
#define FILTER_VEC_TYPE MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_K_OFM)
#define FILTER_PACKED_VEC_TYPE MAKE_VECTOR_TYPE(FILTER_TYPE, TILE_K_OFM_PACKED)
#define BIAS_VEC_TYPE MAKE_VECTOR_TYPE(BIAS_TYPE, TILE_OFM)
#define OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, TILE_OFM)
#define ACTIVATION_VEC_TYPE MAKE_VECTOR_TYPE(ACTIVATION_TYPE, TILE_OFM)
#define TO_OUTPUT_VEC_TYPE(x) CAT(convert_, OUTPUT_VEC_TYPE)(x)
#define TO_ACTIVATION_VEC_TYPE(x) CAT(convert_, ACTIVATION_VEC_TYPE)(x)
#define TO_FILTER_VEC_TYPE(x) CAT(convert_, FILTER_VEC_TYPE)(x)
#define TO_ACCUMULATOR_VEC_TYPE(x) CAT(convert_, ACCUMULATOR_VEC_TYPE)(x)

#define INPUT_BLOCK_READ(ptr, offset)                                          \
  BLOCK_READN(INPUT0_TYPE, TILE_IFM, ptr, offset)
#define FILTER_BLOCK_READ(ptr, offset)                                         \
  BLOCK_READN(FILTER_TYPE, TILE_K_OFM_PACKED, ptr, offset)
#define BIAS_BLOCK_READ(ptr, offset)                                           \
  BLOCK_READN(BIAS_TYPE, TILE_OFM, ptr, offset)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val)                                   \
  BLOCK_WRITEN(OUTPUT_TYPE, TILE_OFM, ptr, offset, val)

#define SLM_FILTER_VEC MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_OFM)
#define SLM_FILTER_PACKED_VEC                                                  \
  MAKE_VECTOR_TYPE(FILTER_TYPE, FILTER_ACTUAL_LOAD_BLOCK_SIZE)
#define SLM_FILTER_UNPACKED_VEC                                                \
  MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, FILTER_ELEMENTS_PER_LOAD)

// Check alignment restrictions for using block writes on output.
#define USE_BLOCK_WRITE                                                        \
  ((OUTPUT_TYPE_SIZE * TILE_OUT_B_PITCH) % 16 == 0 &&                          \
   (OUTPUT_TYPE_SIZE * OUTPUT_OFFSET) % 16 == 0)

#if !REALIGN_FP16_OFFSET
#define MAIN_LOOP_ELEMENTS_COUNT IFM_SIZE
#else
// For REALIGN_FP16_OFFSET one feature is processed separately before entering
// main loop to correct alignment.
#define MAIN_LOOP_ELEMENTS_COUNT (IFM_SIZE - 1)
#endif

#define INPUT_ELEMENTS_COUNT IFM_SIZE

// Dyc Quantize
#if USE_SLM && DYNAMIC_QUANTIZE

#if COMPRESSED_WEIGHTS_INT4
#define SLM_WEIGHT_TYPE DQ_TYPE
#else
#define SLM_WEIGHT_TYPE FILTER_TYPE
#endif

#define PACKED_DQ_TYPE uint
#define ACCUM_DQ_TYPE int
#define DQ_SLM_FILTER_PACKED_VEC                                               \
  MAKE_VECTOR_TYPE(FILTER_TYPE, FILTER_ACTUAL_LOAD_BLOCK_SIZE)
#define SLM_WEIGHT_VEC MAKE_VECTOR_TYPE(SLM_WEIGHT_TYPE, INPUT_LOAD_SIZE)
#define SLM_WEIGHT_UNPACKED_VEC                                                \
  MAKE_VECTOR_TYPE(SLM_WEIGHT_TYPE, FILTER_ELEMENTS_PER_LOAD)
#define WEIGHT_VEC_TYPE MAKE_VECTOR_TYPE(SLM_WEIGHT_TYPE, TILE_K_OFM)
#define MAKE_DQ_TYPE_VEC(x) MAKE_VECTOR_TYPE(DQ_TYPE, x)

#define TO_DQ_TYPE(x) CAT(CAT(convert_, DQ_TYPE), _sat)(x)
#define TO_DQ_VEC_TYPE(x) CAT(convert_, DQ_VEC_TYPE)(x)
#define TO_ACCUM_DQ_TYPE(x) CAT(convert_, ACCUM_DQ_TYPE)(x)
#define TO_SLM_WEIGHT_UNPACKED_VEC(x) CAT(convert_, SLM_WEIGHT_UNPACKED_VEC)(x)
#define TO_WEIGHT_VEC_TYPE(x) CAT(convert_, WEIGHT_VEC_TYPE)(x)

#define AS_TYPE_N_(type, n, x) as_##type##n(x)
#define AS_TYPE_N(type, n, x) AS_TYPE_N_(type, n, x)
#define AS_DQ_TYPE_4(x) AS_TYPE_N(DQ_TYPE, INPUT_LOAD_SIZE, x)

inline void fc_bf_tiled_kernel_dyn_quan(
  OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE *input,
  __global DQ_TYPE *quantized_input,
  __global INPUT0_TYPE *quan_var, // pair of params for each quantizing group :
                                  // scale, activation_sum
#if DECOMPRESSION_SCALE_TERM
  const __global DECOMPRESSION_SCALE_TYPE *decompression_scale,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
  const __global DECOMPRESSION_ZP_TYPE *decompression_zp,
#endif
  __global OUTPUT_TYPE *output, const __global FILTER_TYPE *weights,
  __local uint *wei_local_mem,
#if BIAS_TERM
  , const __global BIAS_TYPE *biases
#endif
#if HAS_FUSED_OPS_DECLS
  ,
  FUSED_OPS_DECLS
#endif
  const int BATCH_SIZE,
  const int size_n, const int size_k, const int quantization_group_size,
  const int scale_row_major) {
  uint gid = (uint)get_group_id(0);
  uint local_id = (uint)get_local_id(1);
  uint sglid = (uint)get_sub_group_local_id();

  const int ALIGN_SIZE_K = ALIGN(size_k, quantization_group_size);
  const int ALIGN_SIZE_N = ALIGN(size_n, TILE_IFM_ELEMENTS_SIZE);
  const int DECOMPRESSION_SCALE_GROUPS_NUM =
    CEIL_DIV(size_k, quantization_group_size);
  const int DECOMPRESSION_SCALE_BATCH_NUM = ALIGN_SIZE_N;
  const int DECOMPRESSION_SCALE_BATCH_PITCH = DECOMPRESSION_SCALE_GROUPS_NUM;
  const int DECOMPRESSION_SCALE_LENGTH =
    ALIGN_SIZE_N * DECOMPRESSION_SCALE_GROUPS_NUM;
  const int NUM_LOOP_IN_DYN_QUAN_GROUP =
    quantization_group_size / (TILE_IFM * SIMD);
  const int TILE_OUT_F_NUM = size_n;
  const int TILE_OUT_B_PITCH = size_n;
  const int TILE_IN_B_PITCH = ALIGN_SIZE_K;
  const int DECOMPRESSION_SCALE_GROUP_SIZE = quantization_group_size;
  const int QUANTIZE_GROUP_SIZE = quantization_group_size;
  const int IFM_SIZE = ALIGN_SIZE_K;

  // Dispatch as bs_fs_bsv_fsv, where bsv = DISPATCH_BSV and fsv = DISPATCH_FSV.
  // This allows more fine grained control over dispatch order than using
  // work-groups and avoids requirement of threads being available for whole
  // work-group. It could hovewer have some drawbacks like not providing
  // physical locality or not using full dispatch pipeline.
  uint feature_mini_block = gid % DISPATCH_FSV;
  uint batch_mini_block = gid / DISPATCH_FSV % DISPATCH_BSV;

  uint feature_mega_block =
    gid / (DISPATCH_FSV * DISPATCH_BSV) %
    (CEIL_DIV(TILE_OUT_F_NUM, OUTER_OFM * TILE_OFM * SIMD) / DISPATCH_FSV);
  uint batch_mega_block =
    gid /
    (DISPATCH_FSV * DISPATCH_BSV *
     CEIL_DIV(TILE_OUT_F_NUM, OUTER_OFM * TILE_OFM * SIMD) / DISPATCH_FSV);

  FILTER_VEC_TYPE wei = 0;

  uint out_f = gid * (TILE_OFM * SIMD);
  uint out_b = LWS_BATCHES * TILE_B * (uint)get_group_id(1) + local_id * TILE_B;

#if OUTPUT_3D
  uint out_b0 = out_b / OUTPUT_FEATURE_NUM;
  uint out_b1 = out_b % OUTPUT_FEATURE_NUM;
  uint input_offset =
    out_b0 * INPUT0_BATCH_PITCH + out_b1 * INPUT0_FEATURE_PITCH + INPUT0_OFFSET;
#else
  uint input_offset = out_b * TILE_IN_B_PITCH + INPUT0_OFFSET;
#endif

#if COMPRESSED_WEIGHTS_INT4
#if FILTER_LAYOUT_OS_IS_YX_OSV64_ISV2
  const int power_of_two_for_simd = 5;
  const int power_of_two_for_osv = 6;
  const uint osv64_weight_base =
    (((int)(out_f >> power_of_two_for_osv)) << power_of_two_for_osv);
  const uint osv_weight_stride = (INPUT_ELEMENTS_COUNT >> 1);
  const uint out_f_offset = (int)((out_f >> power_of_two_for_simd) & 0x1)
                            << power_of_two_for_simd;
  // out_f(32)  : 0  * osv_weight_stride + 32;
  // out_f(64)  : 64 * osv_weight_stride + 0;
  // out_f(128) : 64 * osv_weight_stride + 32;
  // ...
  uint weights_offset = osv64_weight_base * osv_weight_stride + out_f_offset;
#else
  uint weights_offset = out_f * (INPUT_ELEMENTS_COUNT / 2);
#endif
#else
  uint weights_offset = out_f * INPUT_ELEMENTS_COUNT;
#endif

  ACCUMULATOR_VEC_TYPE acc[TILE_B] = {};

  // Dynamic Quantize
  MAKE_VECTOR_TYPE(DQ_TYPE, INPUT_LOAD_SIZE)
  tiled_input_0[HALF_TILE_B] = {}; // Load 4 linear inputs for packing
  PACKED_DQ_TYPE packed_in_0[HALF_TILE_B] =
    {}; // Packing char4 inputs to 1 integer
  INPUT0_TYPE de_quantize_scale[TILE_B];

#if COMPRESSED_WEIGHTS_INT8
  INPUT0_TYPE activation_sum[TILE_B] = {};
#endif

#if COMPRESSED_WEIGHTS
  ACCUMULATOR_VEC_TYPE d_scale = 0;
  if (DECOMPRESSION_SCALE_GROUPS_NUM == 1 && OUTER_OFM == 1) {
    if (DECOMPRESSION_SCALE_LENGTH > 1 &&
        DECOMPRESSION_SCALE_LENGTH % (TILE_OFM * SIMD) == 0) {
      d_scale = TO_ACCUMULATOR_VEC_TYPE(BLOCK_READN(
        DECOMPRESSION_SCALE_TYPE, TILE_OFM, decompression_scale, out_f));
    } else if (DECOMPRESSION_SCALE_LENGTH > 1) {
      unroll_for(uint of = 0; of < TILE_OFM; ++of) {
        uint offset = out_f + of * SIMD + get_sub_group_local_id();
        if (offset < DECOMPRESSION_SCALE_LENGTH)
          ((ACCUMULATOR_TYPE *)(&d_scale))[of] = decompression_scale[offset];
      }
    } else {
      d_scale = decompression_scale[0];
    }
  }
  ACCUMULATOR_TYPE *d_scales = (ACCUMULATOR_TYPE *)(&d_scale);
#endif

#if COMPRESSED_WEIGHTS && DECOMPRESSION_ZP_TERM &&                             \
  DECOMPRESSION_ZP_GROUPS_NUM == 1 && !DECOMPRESSION_ZP_SCALAR &&              \
  OUTER_OFM == 1
#if DECOMPRESSION_ZP_LENGTH > 1 &&                                             \
  DECOMPRESSION_ZP_LENGTH % (TILE_OFM * SIMD) == 0
  ACCUMULATOR_VEC_TYPE d_zp = TO_ACCUMULATOR_VEC_TYPE(
    BLOCK_READN(DECOMPRESSION_ZP_TYPE, TILE_OFM, decompression_zp, out_f));
#elif DECOMPRESSION_ZP_LENGTH > 1 &&                                           \
  DECOMPRESSION_ZP_LENGTH % (TILE_OFM * SIMD) != 0
  ACCUMULATOR_VEC_TYPE d_zp = 0;
  unroll_for(uint of = 0; of < TILE_OFM; ++of) {
    uint offset = out_f + of * SIMD + get_sub_group_local_id();
    if (offset < DECOMPRESSION_ZP_LENGTH)
      ((ACCUMULATOR_TYPE *)(&d_zp))[of] = decompression_zp[offset];
  }
#else
  ACCUMULATOR_VEC_TYPE d_zp = decompression_zp[0];
#endif
  ACCUMULATOR_TYPE *d_zps = (ACCUMULATOR_TYPE *)(&d_zp);
#endif

  ACTIVATION_VEC_TYPE activated[TILE_B] = {};
#if OUTER_OFM > 1
  uint input_offset_init = input_offset;
  uint weights_offset_init = weights_offset;
  uint out_f_init = out_f;
  __attribute__((opencl_unroll_hint(1))) for (uint oi = 0; oi < OUTER_OFM;
                                              ++oi) {
    input_offset = input_offset_init;
    out_f += TILE_OFM * SIMD * oi;
#endif

    // =====================================================================================================================================
    // Main computation loop
    const uint iterations = CEIL_DIV(
      MAIN_LOOP_ELEMENTS_COUNT,
      TILE_IFM_ELEMENTS_SIZE); // TILE_IFM_ELEMENTS_SIZE : (TILE_IFM * SIMD)
    // Each sub-group loads 2 Batch
    const uint idx_sglid =
      (sglid * TILE_K) %
      TILE_IFM_ELEMENTS_SIZE; // same index for sglid 0~7 : to tile_k direction
    const uint batch_sglid =
      (sglid * TILE_K) / TILE_IFM_ELEMENTS_SIZE; // 0 to 1 : to batch direction
    const uint scale_pitch = CEIL_DIV(TILE_IN_B_PITCH, QUANTIZE_GROUP_SIZE);

#if PER_TOKEN_SIZE_DYN_QUANTIZE
    // Each token is quantized by once. So, all MAIN_LOOP_ELEMENTS_COUNT share
    // just one quantizing variable
    uint per_token_offset = input_offset / QUANTIZE_GROUP_SIZE;
    unroll_for(uint bi = 0; bi < TILE_B; ++bi) {
      de_quantize_scale[bi] = TO_INPUT0_TYPE(quan_var[per_token_offset * 2]);
#if COMPRESSED_WEIGHTS_INT8
      activation_sum[bi] = TO_INPUT0_TYPE(quan_var[per_token_offset * 2 + 1]);
#endif
      per_token_offset += scale_pitch;
    }
#endif

#if COMPRESSED_WEIGHTS_INT8
    ACCUMULATOR_TYPE wei_zp[TILE_OFM] = {};
    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
#if DECOMPRESSION_ZP_TERM
#if DECOMPRESSION_ZP_SCALAR
      wei_zp[fi] = (TO_ACCUMULATOR_TYPE)(DECOMPRESSION_ZP_VALUE);
#elif DECOMPRESSION_ZP_GROUPS_NUM == 1
      wei_zp[fi] = TO_ACCUMULATOR_TYPE(d_zps[fi % DECOMPRESSION_ZP_LENGTH]);
#endif
#else
      wei_zp[fi] = ACCUMULATOR_VAL_ZERO;
#endif
    }
#endif

    MAKE_VECTOR_TYPE(int, TILE_B) acc_tmp[TILE_OFM] = {};
    __attribute__((opencl_unroll_hint(1))) for (uint ni = 0; ni < iterations;
                                                ++ni) {
      uint in_offset =
        input_offset + (idx_sglid + batch_sglid * TILE_IN_B_PITCH);
      uint scale_offset = CEIL_DIV(input_offset, QUANTIZE_GROUP_SIZE);
      for (uint bi = 0; bi < HALF_TILE_B; ++bi) {
        // Load quantizing info from pre-quantizing kernel
        tiled_input_0[bi] = vload4(0, &quantized_input[in_offset]);
        // Packing : Get 4(B)x4(K) integer vector (packing to 4x1 vector)
        packed_in_0[bi] = as_uint(tiled_input_0[bi]);

        // Next batch
        in_offset += (TILE_IN_B_PITCH * 2);

#if !PER_TOKEN_SIZE_DYN_QUANTIZE
        if (NUM_LOOP_IN_DYN_QUAN_GROUP == 1) {
          de_quantize_scale[bi * 2] = quan_var[scale_offset * 2];
          de_quantize_scale[bi * 2 + 1] =
            quan_var[scale_offset * 2 + scale_pitch * 2];
#if COMPRESSED_WEIGHTS_INT8
          // Need additional accumulation of quantized activation along the
          // dyn-quan group
          //  to use i8 multiplier for int8 weight
          activation_sum[bi * 2] = quan_var[scale_offset * 2 + 1];
          activation_sum[bi * 2 + 1] =
            quan_var[scale_offset * 2 + 1 + scale_pitch * 2];
#endif
          scale_offset += (scale_pitch * 2);
        }
#endif
      }

#if !PER_TOKEN_SIZE_DYN_QUANTIZE
      if (NUM_LOOP_IN_DYN_QUAN_GROUP > 1) {
        if (ni % NUM_LOOP_IN_DYN_QUAN_GROUP == 0) {
          unroll_for(uint bi = 0; bi < TILE_B; ++bi) {
            de_quantize_scale[bi] = quan_var[scale_offset * 2];
#if COMPRESSED_WEIGHTS_INT8
            activation_sum[bi] = quan_var[scale_offset * 2 + 1];
#endif
            scale_offset += scale_pitch;
          }
        }
      }
#endif

      input_offset += TILE_IFM_ELEMENTS_SIZE;

#if TILE_OFM != 2
#error "FC bf_tiled kernel: can't use SLM optimization with TILE_OFM != 2"
#endif
#if FILTER_LAYOUT_OS_IYX_OSV16 && TILE_K != 4
#error                                                                         \
  "FC bf_tiled kernel: can't use SLM optimization with TILE_K != 2 && OS_IYX_OSV16 layout"
#endif

// Skip first barrier synchronization if there is only single outer loop
// iteration.
#if MAIN_LOOP_ELEMENTS_COUNT / TILE_IFM_ELEMENTS_SIZE > 1
      barrier(CLK_LOCAL_MEM_FENCE);
#endif

      __local uint *char_slm_weight = (__local uint *)wei_local_mem;

#if COMPRESSED_WEIGHTS_INT4
#if FILTER_LAYOUT_OS_IS_YX_OSV64_ISV2
      uint weights_idx = weights_offset + local_id * SIMD * FILTER_LOAD_ITERS *
                                            FILTER_LOAD_BLOCK_SIZE * 2;
#else
      uint weights_idx = weights_offset + local_id * SIMD * FILTER_LOAD_ITERS *
                                            FILTER_ACTUAL_LOAD_BLOCK_SIZE;
#endif
#else
    uint weights_idx =
      weights_offset + local_id * SIMD * FILTER_LOAD_ITERS * TILE_K_OFM_PACKED;
#endif
      uint wei_local_idx =
        local_id * SIMD * FILTER_LOAD_ITERS * (FILTER_LOAD_BLOCK_SIZE / 2) +
        sglid * 2;

      // DQ_DECOMPRESSION_SCALE_POST_OP SHOULD be enabled for dynamic quantize
      // FC : scale is ACCUMULATOR_VAL_ONE
      unroll_for(uint load_iter = 0; load_iter < FILTER_LOAD_ITERS;
                 ++load_iter) {
#if COMPRESSED_WEIGHTS_INT4
#if FILTER_LAYOUT_OS_IYX_OSV16
        SLM_FILTER_PACKED_VEC wei_packed0 = BLOCK_READN(
          FILTER_TYPE, FILTER_ACTUAL_LOAD_BLOCK_SIZE, weights, weights_idx);
        SLM_FILTER_PACKED_VEC wei_packed1 =
          BLOCK_READN(FILTER_TYPE, FILTER_ACTUAL_LOAD_BLOCK_SIZE, weights,
                      (weights_idx + ((IFM_SIZE / 2) * 16)));
        SLM_WEIGHT_UNPACKED_VEC dq_wei_unpacked;
        // loaded weights 'wei_packed' of os_iyx_osv16 format have continuous
        // values along TILE_K. So no need to transpose while unpacking
        dq_wei_unpacked.s0123 =
          (UNPACK_INT4(DQ_TYPE, *((INT4_PACKED_TYPE_PRELOAD *)&wei_packed0)));
        dq_wei_unpacked.s4567 =
          (UNPACK_INT4(DQ_TYPE, *((INT4_PACKED_TYPE_PRELOAD *)&wei_packed1)));
#elif FILTER_LAYOUT_OS_IS_YX_OSV64_ISV2
        SLM_FILTER_PACKED_VEC wei_packed0 = BLOCK_READN(
          FILTER_TYPE, FILTER_ACTUAL_LOAD_BLOCK_SIZE, weights, weights_idx);
        SLM_FILTER_PACKED_VEC wei_packed1 =
          BLOCK_READN(FILTER_TYPE, FILTER_ACTUAL_LOAD_BLOCK_SIZE, weights,
                      (weights_idx + (FILTER_LOAD_BLOCK_SIZE * SIMD)));
        SLM_WEIGHT_UNPACKED_VEC dq_wei_unpacked;
        SLM_WEIGHT_UNPACKED_VEC dq_wei_unpacked_tmp;
        dq_wei_unpacked_tmp.s0123 =
          (UNPACK_INT4(DQ_TYPE, *((INT4_PACKED_TYPE_PRELOAD *)&wei_packed0)));
        dq_wei_unpacked_tmp.s4567 =
          (UNPACK_INT4(DQ_TYPE, *((INT4_PACKED_TYPE_PRELOAD *)&wei_packed1)));
        dq_wei_unpacked.s01 = dq_wei_unpacked_tmp.s01;
        dq_wei_unpacked.s23 = dq_wei_unpacked_tmp.s45;
        dq_wei_unpacked.s45 = dq_wei_unpacked_tmp.s23;
        dq_wei_unpacked.s67 = dq_wei_unpacked_tmp.s67;
#else
        SLM_FILTER_PACKED_VEC wei_packed = BLOCK_READN(
          FILTER_TYPE, FILTER_LOAD_BLOCK_SIZE, weights, weights_idx);
        SLM_WEIGHT_UNPACKED_VEC dq_wei_unpacked = (UNPACK_TRANSPOSED_INT4(
          DQ_TYPE, *((INT4_PACKED_TYPE_PRELOAD *)&wei_packed)));
#endif
#else // COMPRESSED_WEIGHTS_INT8
      SLM_WEIGHT_UNPACKED_VEC dq_wei_unpacked;
      WEIGHT_VEC_TYPE wei_packed =
        TO_WEIGHT_VEC_TYPE(FILTER_BLOCK_READ(weights, weights_idx));
      dq_wei_unpacked.s0123 = wei_packed.s0246;
      dq_wei_unpacked.s4567 = wei_packed.s1357;
#endif

// Calculate zero-point and scale only for DQ_DECOMPRESSION_SCALE_POST_OP
// enabled Calculate weight : w = (w - dzp) * ds if DECOMPRESSION_ZP_TERM is not
// enabled, then dzp is ACCUMULATOR_VAL_ZERO.
#if DECOMPRESSION_ZP_TERM && !COMPRESSED_WEIGHTS_INT8
#if DECOMPRESSION_ZP_SCALAR
        SLM_WEIGHT_UNPACKED_VEC dzp =
          (SLM_WEIGHT_UNPACKED_VEC)(DECOMPRESSION_ZP_VALUE);
        dq_wei_unpacked -= dzp;
#elif DECOMPRESSION_ZP_GROUPS_NUM > 1
        SLM_WEIGHT_TYPE *w = (SLM_WEIGHT_TYPE *)(&dq_wei_unpacked);
        const uint ni_offset = ni * TILE_IFM * SIMD + local_id *
                                                        FILTER_LOAD_ITERS *
                                                        FILTER_LOAD_BLOCK_SIZE;
        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
          const uint offset_ofm = out_f + fi * SIMD + sglid;
          unroll_for(uint kii = 0; kii < FILTER_LOAD_BLOCK_SIZE; ++kii) {
            const uint offset_ifm =
              ni_offset + load_iter * FILTER_LOAD_BLOCK_SIZE + kii;
            const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) *
                                     DECOMPRESSION_ZP_BATCH_PITCH +
                                   (offset_ifm / DECOMPRESSION_ZP_GROUP_SIZE) *
                                     DECOMPRESSION_ZP_FEATURE_PITCH;
            w[W_DYN_QUAN_IDX] =
              w[W_DYN_QUAN_IDX] - CAT(CAT(convert_, SLM_WEIGHT_TYPE),
                                      _rte)(decompression_zp[zp_offset]);
          }
        }
#else
        SLM_WEIGHT_TYPE *w = (SLM_WEIGHT_TYPE *)(&dq_wei_unpacked);
        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
          unroll_for(uint kii = 0; kii < FILTER_LOAD_BLOCK_SIZE; ++kii) {
#if OUTER_OFM > 1
            w[W_DYN_QUAN_IDX] =
              w[W_DYN_QUAN_IDX] - decompression_zp[out_f + fi * SIMD + sglid];
#else
            w[W_DYN_QUAN_IDX] =
              w[W_DYN_QUAN_IDX] - d_zps[fi % DECOMPRESSION_ZP_LENGTH];
#endif
          }
        }
#endif
#endif

#if FILTER_LOAD_BLOCK_SIZE == 2
        SLM_WEIGHT_VEC wei_1 = {dq_wei_unpacked.s01, dq_wei_unpacked.s23};
        char_slm_weight[wei_local_idx] = as_uint(wei_1);
#elif FILTER_LOAD_BLOCK_SIZE == 4
      SLM_WEIGHT_VEC wei_1 = {dq_wei_unpacked.s01, dq_wei_unpacked.s23};
      char_slm_weight[wei_local_idx] = as_uint(wei_1);
      SLM_WEIGHT_VEC wei_2 = {dq_wei_unpacked.s45, dq_wei_unpacked.s67};
      char_slm_weight[wei_local_idx + 1] = as_uint(wei_2);
#elif FILTER_LOAD_BLOCK_SIZE == 8
      SLM_WEIGHT_VEC wei_1 = {dq_wei_unpacked.s01, dq_wei_unpacked.s23};
      char_slm_weight[wei_local_idx] = as_uint(wei_1);
      SLM_WEIGHT_VEC wei_2 = {dq_wei_unpacked.s45, dq_wei_unpacked.s67};
      char_slm_weight[wei_local_idx + 1] = as_uint(wei_2);
      SLM_WEIGHT_VEC wei_3 = {dq_wei_unpacked.s89, dq_wei_unpacked.sab};
      char_slm_weight[wei_local_idx + 2] = as_uint(wei_3);
      SLM_WEIGHT_VEC wei_4 = {dq_wei_unpacked.scd, dq_wei_unpacked.sef};
      char_slm_weight[wei_local_idx + 3] = as_uint(wei_4);
#else
#error "FC bf_tiled kernel: unsupported FILTER_LOAD_BLOCK_SIZE for SLM kernel"
#endif

        wei_local_idx += SIMD * (FILTER_LOAD_BLOCK_SIZE / 2);
#if COMPRESSED_WEIGHTS_INT8
        weights_idx += SIMD * TILE_K_OFM_PACKED;
#else
      weights_idx += SIMD * FILTER_ACTUAL_LOAD_BLOCK_SIZE;
#endif

#if COMPRESSED_WEIGHTS_INT8 && DECOMPRESSION_ZP_TERM &&                        \
  DECOMPRESSION_ZP_GROUPS_NUM > 1 && !DECOMPRESSION_ZP_SCALAR
        unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
#if FILTER_LOAD_BLOCK_SIZE > DECOMPRESSION_ZP_GROUP_SIZE &&                    \
  FILTER_LOAD_BLOCK_SIZE % DECOMPRESSION_ZP_GROUP_SIZE != 0
#error                                                                         \
  "FC bf_tiled kernel: Not support DECOMPRESSION_ZP_GROUPS_NUM > 1 with unaligned DECOMPRESSION_ZP_GROUP_SIZE"
#elif FILTER_LOAD_BLOCK_SIZE < DECOMPRESSION_ZP_GROUP_SIZE &&                  \
  DECOMPRESSION_ZP_GROUP_SIZE % FILTER_LOAD_BLOCK_SIZE != 0
#error                                                                         \
  "FC bf_tiled kernel: Not support DECOMPRESSION_ZP_GROUPS_NUM > 1 with unaligned FILTER_LOAD_BLOCK_SIZE"
#endif

          const uint ni_offset =
            ni * TILE_IFM * SIMD +
            local_id * FILTER_LOAD_ITERS * FILTER_LOAD_BLOCK_SIZE;
          const uint offset_ofm = out_f + fi * SIMD + sglid;
          const uint offset_ifm =
            ni_offset + load_iter * FILTER_LOAD_BLOCK_SIZE;
          const uint zp_offset = (offset_ofm % DECOMPRESSION_ZP_BATCH_NUM) *
                                   DECOMPRESSION_ZP_BATCH_PITCH +
                                 (offset_ifm / DECOMPRESSION_ZP_GROUP_SIZE) *
                                   DECOMPRESSION_ZP_FEATURE_PITCH;
          wei_zp[fi] = TO_ACCUMULATOR_TYPE(decompression_zp[zp_offset]);
        }
#endif
      }

      wei_local_idx = sglid * 2;

      barrier(CLK_LOCAL_MEM_FENCE);

      unroll_for(uint ki = 0; ki < TILE_IFM_ELEMENTS_SIZE / TILE_K; ++ki) {
#if TILE_K != 4
#error "FC bf_tiled kernel: unsupported TILE_K size for SLM kernel"
#endif

        // Compute input * weight : packed char4 type
        WEIGHT_VEC_TYPE weight =
          vload8(0, (__local SLM_WEIGHT_TYPE
                       *)(&char_slm_weight[wei_local_idx + 16 * 2 * ki]));
        SLM_WEIGHT_VEC first_weight = weight.s0123;
        SLM_WEIGHT_VEC second_weight = weight.s4567;
        unroll_for(uint bi = 0; bi < TILE_B; ++bi) {
          MAKE_DQ_TYPE_VEC(4)
          input_val = AS_DQ_TYPE_4(
            _sub_group_shuffle(packed_in_0[bi / 2], (bi % 2) * 8 + ki));
          acc_tmp[0][bi] = imad_SW(acc_tmp[0][bi], input_val, first_weight);
          acc_tmp[1][bi] = imad_SW(acc_tmp[1][bi], input_val, second_weight);
        }

        weights_offset += TILE_K_OFM_PACKED * TILE_OFM_PER_OSV_SIZE * SIMD;

#if DQ_DECOMPRESSION_SCALE_POST_OP
        if (TILE_IFM_ELEMENTS_SIZE > DECOMPRESSION_SCALE_GROUP_SIZE) {
          unroll_for(uint bi = 0; bi < TILE_B; ++bi) {
            unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
              const uint offset_ofm = out_f + fi * SIMD + sglid;
              ACCUMULATOR_TYPE ds;
              if (DECOMPRESSION_SCALE_GROUPS_NUM > 1) {
                const uint scale_offset =
                  (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) *
                    DECOMPRESSION_SCALE_BATCH_PITCH +
                  ((ni * TILE_IFM * SIMD + ki * TILE_K) /
                   DECOMPRESSION_SCALE_GROUP_SIZE) *
                    DECOMPRESSION_SCALE_FEATURE_PITCH;
                ds = decompression_scale[scale_offset];
              } else {
#if OUTER_OFM > 1
                ds = decompression_scale[offset_ofm];
#else
                ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
#endif
              }

#if COMPRESSED_WEIGHTS_INT8
              ACCUM_DQ_TYPE modified_calc_buff =
                ((int *)(&acc_tmp[fi]))[bi] -
                ((float)(wei_zp[fi]) * activation_sum[bi]);
              ((ACCUMULATOR_TYPE *)(&acc[bi]))[fi] +=
                (convert_half)(convert_float(modified_calc_buff) * (float)ds *
                               (float)de_quantize_scale[bi]);
#else
              ((ACCUMULATOR_TYPE *)(&acc[bi]))[fi] +=
                convert_half(((int *)(&acc_tmp[fi]))[bi]) * ds *
                de_quantize_scale[bi];
#endif
              acc_tmp[fi][bi] = 0;
            }
          }
        }
#endif
      } // Whole tile_k elements of each iteration : ki

#if !PER_TOKEN_SIZE_DYN_QUANTIZE && DQ_DECOMPRESSION_SCALE_POST_OP
      if (TILE_IFM_ELEMENTS_SIZE <= DECOMPRESSION_SCALE_GROUP_SIZE) {
        // Dynamic-quantizing group size set to same or smaller than scale group
        // size
        if ((ni % NUM_LOOP_IN_DYN_QUAN_GROUP) ==
            (NUM_LOOP_IN_DYN_QUAN_GROUP - 1)) {
          const uint ni_offset =
            ((ni * TILE_IFM * SIMD) / DECOMPRESSION_SCALE_GROUP_SIZE) *
            DECOMPRESSION_SCALE_FEATURE_PITCH;
          unroll_for(uint bi = 0; bi < TILE_B; ++bi) {
            unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
              const uint offset_ofm = out_f + fi * SIMD + sglid;
              ACCUMULATOR_TYPE ds;
              if (DECOMPRESSION_SCALE_GROUPS_NUM > 1) {
                uint scale_offset;
                if (scale_row_major) {
                  scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) *
                                   DECOMPRESSION_SCALE_BATCH_PITCH +
                                 ni_offset;
                } else {
                  scale_offset = (offset_ofm % DECOMPRESSION_SCALE_BATCH_NUM) +
                                 ni_offset * ALIGN_SIZE_N;
                }
                ds = decompression_scale[scale_offset];
              } else {
#if OUTER_OFM > 1
                ds = decompression_scale[offset_ofm];
#else
                ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
#endif
              }

#if COMPRESSED_WEIGHTS_INT8
              ACCUM_DQ_TYPE modified_calc_buff =
                ((float)((int *)(&acc_tmp[fi]))[bi]) -
                ((float)(wei_zp[fi]) * activation_sum[bi]);
              ((ACCUMULATOR_TYPE *)(&acc[bi]))[fi] +=
                (convert_half)(convert_float(modified_calc_buff) * (float)ds *
                               (float)de_quantize_scale[bi]);
#else
              ((ACCUMULATOR_TYPE *)(&acc[bi]))[fi] +=
                convert_float(((int *)(&acc_tmp[fi]))[bi]) * ds *
                de_quantize_scale[bi];
#endif
              acc_tmp[fi][bi] = 0;
            }
          }
        }
      }
#endif
    } // Main compute loop : ni

#if PER_TOKEN_SIZE_DYN_QUANTIZE
    unroll_for(uint bi = 0; bi < TILE_B; ++bi) {
      unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
        ACCUMULATOR_TYPE ds = d_scales[fi % DECOMPRESSION_SCALE_LENGTH];
#if COMPRESSED_WEIGHTS_INT8
        float modified_calc_buff = ((float)((int *)(&acc_tmp[fi]))[bi]) -
                                   ((float)(wei_zp[fi]) * activation_sum[bi]);
        ((ACCUMULATOR_TYPE *)(&acc[bi]))[fi] =
          (convert_half)(modified_calc_buff)*ds * de_quantize_scale[bi];
#else
        ((ACCUMULATOR_TYPE *)(&acc[bi]))[fi] =
          convert_half(((int *)(&acc_tmp[fi]))[bi]) * ds *
          de_quantize_scale[bi];
#endif
      }
    }
#endif

    // =====================================================================================================================================
    // Post-processing: bias, activation, fused-ops
    for (uint bi = 0; bi < TILE_B; ++bi) {
      activated[bi] = TO_ACTIVATION_VEC_TYPE(acc[bi]);
#if OUTER_OFM > 1
      acc[bi] = 0;
#endif
    }

#if BIAS_TERM
#if TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0
    BIAS_VEC_TYPE bias = BIAS_BLOCK_READ(biases, out_f);
#else
    BIAS_VEC_TYPE bias = 0;
    unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
      ((BIAS_TYPE *)(&bias))[fi] = biases[out_f + sglid + fi * SIMD];
    }
#endif
    unroll_for(uint bi = 0; bi < TILE_B; ++bi) {
      activated[bi] += TO_ACTIVATION_VEC_TYPE(bias);
    }
#endif

    OUTPUT_VEC_TYPE result[TILE_B] = {};
#if HAS_FUSED_OPS
    unroll_for(uint bi = 0; bi < TILE_B; ++bi) {
#if TILE_OFM > 1
      unroll_for(uint fi = 0; fi < TILE_OFM; ++fi) {
        FUSED_OPS_VEC;
        result[bi][fi] = FUSED_OPS_RESULT_VEC;
      }
#else
      FUSED_OPS_SCALAR;
      result[bi] = FUSED_OPS_RESULT_SCALAR;
#endif // TILE_OFM > 1
    }
#else
  unroll_for(uint bi = 0; bi < TILE_B; ++bi) {
    result[bi] = TO_OUTPUT_VEC_TYPE(
      ACTIVATION_TYPED(activated[bi], ACTIVATION_PARAMS_TYPED));
  }
#endif

    // =====================================================================================================================================
    // Write results
    uint output_offset =
      out_f * TILE_OUT_F_PITCH + out_b * TILE_OUT_B_PITCH + OUTPUT_OFFSET;

    if ((TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 ||
         out_f + (TILE_OFM * SIMD) <= TILE_OUT_F_NUM)) {
#if IS_DYNAMIC
#define WRITE_OUTPUT(bi)                                                       \
  do {                                                                         \
    if (bi + out_b < BATCH_SIZE)                                               \
      OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);                   \
    output_offset += TILE_OUT_B_PITCH;                                         \
  } while (false)
#else
#define WRITE_OUTPUT(bi)                                                       \
  do {                                                                         \
    OUTPUT_BLOCK_WRITE(output, output_offset, result[bi]);                     \
    output_offset += TILE_OUT_B_PITCH;                                         \
  } while (false)
#endif
      CONST_LOOP(TILE_B, WRITE_OUTPUT);
#undef WRITE_OUTPUT
    } else {
      output_offset += sglid;

      for (uint bi = 0; bi < TILE_B; ++bi) {
        for (uint fi = 0; fi < TILE_OFM; ++fi) {
          const bool should_write =
#if IS_DYNAMIC
            bi + out_b < BATCH_SIZE &&
#endif
            (TILE_OUT_F_NUM % (TILE_OFM * SIMD) == 0 ||
             out_f + fi * SIMD + sglid < TILE_OUT_F_NUM);
          if (should_write) {
            output[output_offset] = ((OUTPUT_TYPE *)(&result[bi]))[fi];
          }
          output_offset += SIMD;
        }
        output_offset += TILE_OUT_B_PITCH - TILE_OFM * SIMD;
      }
    }
#if OUTER_OFM > 1 && !defined(SWIGLU_LENGTH)
  }
#endif
  // =====================================================================================================================================
}
#endif

REQD_SUB_GROUP_SIZE(SIMD)
kernel void fc_bf_tiled_kernel_default(
  OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE *input,
#if DECOMPRESSION_SCALE_TERM
  const __global DECOMPRESSION_SCALE_TYPE *decompression_scale,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
  const __global DECOMPRESSION_ZP_TYPE *decompression_zp,
#endif
  __global OUTPUT_TYPE *output, const __global FILTER_TYPE *weights
#if BIAS_TERM
  ,
  const __global BIAS_TYPE *biases
#endif
#if DYNAMIC_QUANTIZE
  ,
  __global DQ_TYPE *quantized_input, __global INPUT0_TYPE *quan_var
#endif
  ,
  const int M, const int size_n, const int size_k,
  const int quantization_group_size, const int scale_row_major) {
  __local uint dq_wei_local_mem[SIMD * TILE_OFM * SIMD];
  fc_bf_tiled_kernel_dyn_quan(
    OPTIONAL_SHAPE_INFO_TENSOR input, quantized_input, quan_var,
#if DECOMPRESSION_SCALE_TERM
    decompression_scale,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    decompression_zp,
#endif
    output, weights, dq_wei_local_mem,
#if BIAS_TERM
    , biases
#endif
        M,
    size_n, size_k, quantization_group_size, scale_row_major);
}

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Portions of this file are derived from llama.cpp
 * (https://github.com/ggml-org/llama.cpp), licensed under the MIT License.
 * Copyright (c) Contributors to llama.cpp
 *
 * Modified by Sungsik Kong, 2025: Adapted for CPU backend integration
 *
 * @file   nntr_ggml_impl_neon.cpp
 * @date   20 August 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author  Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Custom-implemented functions to support ggml functions for internal
 * uses in nntrainer
 */

#include <algorithm>
#include <assert.h>
#include <cstring>
#include <math.h>
#include <stddef.h>
#include <stdexcept>
#include <stdint.h>
#include <tensor_dim.h>

#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_utils.h>

void nntr_gemv_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);
#if defined(__ARM_FEATURE_DOTPROD)
  const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx;
  for (int c = 0; c < nc; c += ncols_interleaved) {
    const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
    float32x4_t acc = vdupq_n_f32(0);
    for (int b = 0; b < nb; b++) {
      int8x16_t b0 = vld1q_s8((const int8_t *)b_ptr->qs);
      int8x16_t b1 = vld1q_s8((const int8_t *)b_ptr->qs + 16);
      int8x16_t b2 = vld1q_s8((const int8_t *)b_ptr->qs + 32);
      int8x16_t b3 = vld1q_s8((const int8_t *)b_ptr->qs + 48);
      float16x4_t bd = vld1_f16((const __fp16 *)b_ptr->d);

      int8x16_t a0 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs);
      int8x16_t a1 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs + 1);
      int8x16_t a2 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs + 2);
      int8x16_t a3 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs + 3);
      float16x4_t ad = vld1_dup_f16((const __fp16 *)&a_ptr->d);

      int32x4_t ret0 = vdupq_n_s32(0);
      int32x4_t ret1 = vdupq_n_s32(0);

      ret0 = vdotq_s32(ret0, b0 << 4, a0);
      ret1 = vdotq_s32(ret1, b1 << 4, a0);
      ret0 = vdotq_s32(ret0, b2 << 4, a1);
      ret1 = vdotq_s32(ret1, b3 << 4, a1);

      ret0 = vdotq_s32(ret0, b0 & 0xf0U, a2);
      ret1 = vdotq_s32(ret1, b1 & 0xf0U, a2);
      ret0 = vdotq_s32(ret0, b2 & 0xf0U, a3);
      ret1 = vdotq_s32(ret1, b3 & 0xf0U, a3);

      int32x4_t ret = vpaddq_s32(ret0, ret1);

      acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4),
                      vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
      a_ptr++;
      b_ptr++;
    }
    vst1q_f32(s, acc);
    s += ncols_interleaved;
  }
  return;

#else
  const void *b_ptr = vx;
  const void *a_ptr = vy;
  float *res_ptr = s;

  __asm__ __volatile__(
    "movi v2.16b, #0x4\n"
    "movi v1.16b, #0xf0\n"
    "add %x[b_ptr], %x[b_ptr], #0x8\n"
    "1:" // Column loop
    "add x23, %x[a_ptr], #0x2\n"
    "movi v0.16b, #0x0\n"
    "mov x22, %x[nb]\n"
    "2:" // Block loop
    "ldr q31, [%x[b_ptr], #0x0]\n"
    "ldr q30, [%x[b_ptr], #0x10]\n"
    "mov x21, x23\n"
    "movi v29.4s, #0x0\n"
    "ldr q28, [%x[b_ptr], #0x20]\n"
    "ldr q27, [%x[b_ptr], #0x30]\n"
    "movi v26.4s, #0x0\n"
    "sub x20, x23, #0x2\n"
    "ld1r { v25.8h }, [x20]\n"
    "ldr q24, [%x[b_ptr], #-0x8]\n"
    "sub x22, x22, #0x1\n"
    "add x23, x23, #0x22\n"
    "ld1r { v23.2d }, [x21], #0x8\n"
    "sshl v22.16b, v31.16b, v2.16b\n"
    "sshl v16.16b, v30.16b, v2.16b\n"
    "add %x[b_ptr], %x[b_ptr], #0x48\n"
    "ld1r { v21.2d }, [x21], #0x8\n"
    "sshl v20.16b, v28.16b, v2.16b\n"
    "sshl v19.16b, v27.16b, v2.16b\n"
    "ld1r { v18.2d }, [x21], #0x8\n"
    "ld1r { v17.2d }, [x21], #0x8\n"
    "and v31.16b, v31.16b, v1.16b\n"
    "and v30.16b, v30.16b, v1.16b\n"
    ".inst 0x4e9796dd  // sdot v29.4s, v22.16b, v23.16b\n"
    ".inst 0x4e97961a  // sdot v26.4s, v16.16b, v23.16b\n"
    "and v28.16b, v28.16b, v1.16b\n"
    "and v27.16b, v27.16b, v1.16b\n"
    "fcvtl v25.4s, v25.4h\n"
    "fcvtl v16.4s, v24.4h\n"
    ".inst 0x4e95969d  // sdot v29.4s, v20.16b, v21.16b\n"
    ".inst 0x4e95967a  // sdot v26.4s, v19.16b, v21.16b\n"
    "fmul v16.4s, v16.4s, v25.4s\n"
    ".inst 0x4e9297fd  // sdot v29.4s, v31.16b, v18.16b\n"
    ".inst 0x4e9297da  // sdot v26.4s, v30.16b, v18.16b\n"
    ".inst 0x4e91979d  // sdot v29.4s, v28.16b, v17.16b\n"
    ".inst 0x4e91977a  // sdot v26.4s, v27.16b, v17.16b\n"
    "addp v29.4s, v29.4s, v26.4s\n"
    "scvtf v29.4s, v29.4s, #0x4\n"
    "fmla v0.4s, v29.4s, v16.4s\n"
    "cbnz x22, 2b\n"
    "sub %x[nc], %x[nc], #0x4\n"
    "str q0, [%x[res_ptr], #0x0]\n"
    "add %x[res_ptr], %x[res_ptr], #0x10\n"
    "cbnz %x[nc], 1b\n"
    : [b_ptr] "+&r"(b_ptr), [res_ptr] "+&r"(res_ptr), [nc] "+&r"(nc)
    : [a_ptr] "r"(a_ptr), [nb] "r"(nb)
    : "memory", "v0", "v1", "v2", "v16", "v17", "v18", "v19", "v20", "v21",
      "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
      "x20", "x21", "x22", "x23");
  return;
#endif
}

#ifdef ENABLE_FP16
// FP16-output variant of nntr_gemv_q4_0_4x8_q8_0. Same kernel body, but the
// final store fuses an fcvtn (FP32 4-lane -> FP16 4-lane) and writes a 64-bit
// d-register; the per-column-tile res_ptr advance halves to 8 bytes.
void nntr_gemv_q4_0_4x8_q8_0_fp16(int n, __fp16 *__restrict s, size_t bs,
                                  const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);
#if defined(__ARM_FEATURE_DOTPROD)
  const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx;
  for (int c = 0; c < nc; c += ncols_interleaved) {
    const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
    float32x4_t acc = vdupq_n_f32(0);
    for (int b = 0; b < nb; b++) {
      int8x16_t b0 = vld1q_s8((const int8_t *)b_ptr->qs);
      int8x16_t b1 = vld1q_s8((const int8_t *)b_ptr->qs + 16);
      int8x16_t b2 = vld1q_s8((const int8_t *)b_ptr->qs + 32);
      int8x16_t b3 = vld1q_s8((const int8_t *)b_ptr->qs + 48);
      float16x4_t bd = vld1_f16((const __fp16 *)b_ptr->d);

      int8x16_t a0 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs);
      int8x16_t a1 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs + 1);
      int8x16_t a2 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs + 2);
      int8x16_t a3 = (int8x16_t)vld1q_dup_s64((const int64_t *)a_ptr->qs + 3);
      float16x4_t ad = vld1_dup_f16((const __fp16 *)&a_ptr->d);

      int32x4_t ret0 = vdupq_n_s32(0);
      int32x4_t ret1 = vdupq_n_s32(0);

      ret0 = vdotq_s32(ret0, b0 << 4, a0);
      ret1 = vdotq_s32(ret1, b1 << 4, a0);
      ret0 = vdotq_s32(ret0, b2 << 4, a1);
      ret1 = vdotq_s32(ret1, b3 << 4, a1);

      ret0 = vdotq_s32(ret0, b0 & 0xf0U, a2);
      ret1 = vdotq_s32(ret1, b1 & 0xf0U, a2);
      ret0 = vdotq_s32(ret0, b2 & 0xf0U, a3);
      ret1 = vdotq_s32(ret1, b3 & 0xf0U, a3);

      int32x4_t ret = vpaddq_s32(ret0, ret1);

      acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4),
                      vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
      a_ptr++;
      b_ptr++;
    }
    vst1_f16((__fp16 *)s, vcvt_f16_f32(acc));
    s += ncols_interleaved;
  }
  return;

#else
  const void *b_ptr = vx;
  const void *a_ptr = vy;
  __fp16 *res_ptr = s;

  __asm__ __volatile__(
    "movi v2.16b, #0x4\n"
    "movi v1.16b, #0xf0\n"
    "add %x[b_ptr], %x[b_ptr], #0x8\n"
    "1:" // Column loop
    "add x23, %x[a_ptr], #0x2\n"
    "movi v0.16b, #0x0\n"
    "mov x22, %x[nb]\n"
    "2:" // Block loop
    "ldr q31, [%x[b_ptr], #0x0]\n"
    "ldr q30, [%x[b_ptr], #0x10]\n"
    "mov x21, x23\n"
    "movi v29.4s, #0x0\n"
    "ldr q28, [%x[b_ptr], #0x20]\n"
    "ldr q27, [%x[b_ptr], #0x30]\n"
    "movi v26.4s, #0x0\n"
    "sub x20, x23, #0x2\n"
    "ld1r { v25.8h }, [x20]\n"
    "ldr q24, [%x[b_ptr], #-0x8]\n"
    "sub x22, x22, #0x1\n"
    "add x23, x23, #0x22\n"
    "ld1r { v23.2d }, [x21], #0x8\n"
    "sshl v22.16b, v31.16b, v2.16b\n"
    "sshl v16.16b, v30.16b, v2.16b\n"
    "add %x[b_ptr], %x[b_ptr], #0x48\n"
    "ld1r { v21.2d }, [x21], #0x8\n"
    "sshl v20.16b, v28.16b, v2.16b\n"
    "sshl v19.16b, v27.16b, v2.16b\n"
    "ld1r { v18.2d }, [x21], #0x8\n"
    "ld1r { v17.2d }, [x21], #0x8\n"
    "and v31.16b, v31.16b, v1.16b\n"
    "and v30.16b, v30.16b, v1.16b\n"
    ".inst 0x4e9796dd  // sdot v29.4s, v22.16b, v23.16b\n"
    ".inst 0x4e97961a  // sdot v26.4s, v16.16b, v23.16b\n"
    "and v28.16b, v28.16b, v1.16b\n"
    "and v27.16b, v27.16b, v1.16b\n"
    "fcvtl v25.4s, v25.4h\n"
    "fcvtl v16.4s, v24.4h\n"
    ".inst 0x4e95969d  // sdot v29.4s, v20.16b, v21.16b\n"
    ".inst 0x4e95967a  // sdot v26.4s, v19.16b, v21.16b\n"
    "fmul v16.4s, v16.4s, v25.4s\n"
    ".inst 0x4e9297fd  // sdot v29.4s, v31.16b, v18.16b\n"
    ".inst 0x4e9297da  // sdot v26.4s, v30.16b, v18.16b\n"
    ".inst 0x4e91979d  // sdot v29.4s, v28.16b, v17.16b\n"
    ".inst 0x4e91977a  // sdot v26.4s, v27.16b, v17.16b\n"
    "addp v29.4s, v29.4s, v26.4s\n"
    "scvtf v29.4s, v29.4s, #0x4\n"
    "fmla v0.4s, v29.4s, v16.4s\n"
    "cbnz x22, 2b\n"
    "sub %x[nc], %x[nc], #0x4\n"
    "fcvtn v0.4h, v0.4s\n"
    "str d0, [%x[res_ptr], #0x0]\n"
    "add %x[res_ptr], %x[res_ptr], #0x8\n"
    "cbnz %x[nc], 1b\n"
    : [b_ptr] "+&r"(b_ptr), [res_ptr] "+&r"(res_ptr), [nc] "+&r"(nc)
    : [a_ptr] "r"(a_ptr), [nb] "r"(nb)
    : "memory", "v0", "v1", "v2", "v16", "v17", "v18", "v19", "v20", "v21",
      "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
      "x20", "x21", "x22", "x23");
  return;
#endif
}
#endif // ENABLE_FP16

void nntr_gemm_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  const void *b_ptr = vx;
  const void *a_ptr = vy;
  float *res_ptr = s;
  size_t res_stride = bs * sizeof(float);

  __asm__ __volatile__("mov x10, %x[nr]\n"
                       "mov x9, #0x88\n"
                       "cmp x10, #0x10\n"
                       "mul x9, %x[nb], x9\n"
                       "blt 4f\n"
                       "1:" // Row loop
                       "add x28, %x[b_ptr], #0x8\n"
                       "mov x27, %x[nc]\n"
                       "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
                       "2:" // Column loop
                       "add x25, %x[a_ptr], #0x8\n"
                       "movi v2.16b, #0x0\n"
                       "movi v10.16b, #0x0\n"
                       "mov x24, %x[nb]\n"
                       "add x23, x25, x9\n"
                       "movi v12.16b, #0x0\n"
                       "movi v28.16b, #0x0\n"
                       "add x22, x23, x9\n"
                       "movi v11.16b, #0x0\n"
                       "movi v13.16b, #0x0\n"
                       "add x21, x22, x9\n"
                       "movi v22.16b, #0x0\n"
                       "movi v23.16b, #0x0\n"
                       "movi v25.16b, #0x0\n"
                       "movi v5.16b, #0x0\n"
                       "movi v7.16b, #0x0\n"
                       "movi v4.16b, #0x0\n"
                       "movi v6.16b, #0x0\n"
                       "movi v30.16b, #0x0\n"
                       "movi v24.16b, #0x0\n"
                       "movi v14.16b, #0x0\n"
                       "3:" // Block loop
                       "ldr q21, [x28, #0x0]\n"
                       "ldr q16, [x28, #0x10]\n"
                       "movi v1.16b, #0x4\n"
                       "movi v19.4s, #0x0\n"
                       "ldr q27, [x25, #0x0]\n"
                       "ldr q15, [x25, #0x10]\n"
                       "movi v26.4s, #0x0\n"
                       "movi v18.4s, #0x0\n"
                       "ldr q29, [x28, #0x20]\n"
                       "ldr q3, [x28, #0x30]\n"
                       "movi v17.4s, #0x0\n"
                       "movi v0.16b, #0xf0\n"
                       "ldr d20, [x25, #-0x8]\n"
                       "ldr d9, [x23, #-0x8]\n"
                       "sshl v8.16b, v21.16b, v1.16b\n"
                       "sshl v31.16b, v16.16b, v1.16b\n"
                       "and v21.16b, v21.16b, v0.16b\n"
                       "and v16.16b, v16.16b, v0.16b\n"
                       "sub x20, x28, #0x8\n"
                       "subs x24, x24, #0x1\n"
                       "add x28, x28, #0x48\n"
                       ".inst 0x4e88a773  // smmla v19.4s, v27.16b, v8.16b\n"
                       ".inst 0x4e9fa77a  // smmla v26.4s, v27.16b, v31.16b\n"
                       "ldr q27, [x25, #0x20]\n"
                       ".inst 0x4e88a5f2  // smmla v18.4s, v15.16b, v8.16b\n"
                       ".inst 0x4e9fa5f1  // smmla v17.4s, v15.16b, v31.16b\n"
                       "sshl v15.16b, v29.16b, v1.16b\n"
                       "sshl v1.16b, v3.16b, v1.16b\n"
                       "and v29.16b, v29.16b, v0.16b\n"
                       "and v3.16b, v3.16b, v0.16b\n"
                       "ldr q0, [x25, #0x30]\n"
                       "fcvtl v20.4s, v20.4h\n"
                       ".inst 0x4e8fa773  // smmla v19.4s, v27.16b, v15.16b\n"
                       "fcvtl v9.4s, v9.4h\n"
                       ".inst 0x4e81a77a  // smmla v26.4s, v27.16b, v1.16b\n"
                       "ldr q27, [x25, #0x40]\n"
                       ".inst 0x4e8fa412  // smmla v18.4s, v0.16b, v15.16b\n"
                       ".inst 0x4e81a411  // smmla v17.4s, v0.16b, v1.16b\n"
                       "ldr q0, [x25, #0x50]\n"
                       ".inst 0x4e95a773  // smmla v19.4s, v27.16b, v21.16b\n"
                       ".inst 0x4e90a77a  // smmla v26.4s, v27.16b, v16.16b\n"
                       "ldr q27, [x25, #0x60]\n"
                       ".inst 0x4e95a412  // smmla v18.4s, v0.16b, v21.16b\n"
                       ".inst 0x4e90a411  // smmla v17.4s, v0.16b, v16.16b\n"
                       "ldr q0, [x25, #0x70]\n"
                       "add x25, x25, #0x88\n"
                       ".inst 0x4e9da773  // smmla v19.4s, v27.16b, v29.16b\n"
                       ".inst 0x4e83a77a  // smmla v26.4s, v27.16b, v3.16b\n"
                       "ldr d27, [x20, #0x0]\n"
                       ".inst 0x4e9da412  // smmla v18.4s, v0.16b, v29.16b\n"
                       ".inst 0x4e83a411  // smmla v17.4s, v0.16b, v3.16b\n"
                       "fcvtl v27.4s, v27.4h\n"
                       "uzp1 v0.2d, v19.2d, v26.2d\n"
                       "uzp2 v26.2d, v19.2d, v26.2d\n"
                       "fmul v19.4s, v27.4s, v20.s[0]\n"
                       "scvtf v0.4s, v0.4s, #0x4\n"
                       "scvtf v26.4s, v26.4s, #0x4\n"
                       "fmla v2.4s, v0.4s, v19.4s\n"
                       "ldr q19, [x23, #0x0]\n"
                       "uzp1 v0.2d, v18.2d, v17.2d\n"
                       "uzp2 v18.2d, v18.2d, v17.2d\n"
                       "fmul v17.4s, v27.4s, v20.s[1]\n"
                       "scvtf v0.4s, v0.4s, #0x4\n"
                       "scvtf v18.4s, v18.4s, #0x4\n"
                       "fmla v10.4s, v26.4s, v17.4s\n"
                       "ldr q17, [x23, #0x10]\n"
                       "fmul v26.4s, v27.4s, v20.s[2]\n"
                       "fmul v20.4s, v27.4s, v20.s[3]\n"
                       "fmla v12.4s, v0.4s, v26.4s\n"
                       "ldr d0, [x22, #-0x8]\n"
                       "ldr d26, [x21, #-0x8]\n"
                       "fcvtl v0.4s, v0.4h\n"
                       "fmla v28.4s, v18.4s, v20.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v18.4s, #0x0\n"
                       ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
                       ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
                       "ldr q19, [x23, #0x20]\n"
                       "fcvtl v26.4s, v26.4h\n"
                       ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
                       ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
                       "ldr q19, [x23, #0x40]\n"
                       ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
                       ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
                       "ldr q19, [x23, #0x60]\n"
                       ".inst 0x4e9da674  // smmla v20.4s, v19.16b, v29.16b\n"
                       ".inst 0x4e83a672  // smmla v18.4s, v19.16b, v3.16b\n"
                       "uzp1 v19.2d, v20.2d, v18.2d\n"
                       "scvtf v19.4s, v19.4s, #0x4\n"
                       "uzp2 v20.2d, v20.2d, v18.2d\n"
                       "fmul v18.4s, v27.4s, v9.s[0]\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "fmla v11.4s, v19.4s, v18.4s\n"
                       "ldr q18, [x22, #0x0]\n"
                       "fmul v19.4s, v27.4s, v9.s[1]\n"
                       "fmla v13.4s, v20.4s, v19.4s\n"
                       "movi v19.4s, #0x0\n"
                       "movi v20.4s, #0x0\n"
                       ".inst 0x4e88a633  // smmla v19.4s, v17.16b, v8.16b\n"
                       ".inst 0x4e9fa634  // smmla v20.4s, v17.16b, v31.16b\n"
                       "ldr q17, [x23, #0x30]\n"
                       ".inst 0x4e8fa633  // smmla v19.4s, v17.16b, v15.16b\n"
                       ".inst 0x4e81a634  // smmla v20.4s, v17.16b, v1.16b\n"
                       "ldr q17, [x23, #0x50]\n"
                       ".inst 0x4e95a633  // smmla v19.4s, v17.16b, v21.16b\n"
                       ".inst 0x4e90a634  // smmla v20.4s, v17.16b, v16.16b\n"
                       "ldr q17, [x23, #0x70]\n"
                       "add x23, x23, #0x88\n"
                       ".inst 0x4e9da633  // smmla v19.4s, v17.16b, v29.16b\n"
                       ".inst 0x4e83a634  // smmla v20.4s, v17.16b, v3.16b\n"
                       "uzp1 v17.2d, v19.2d, v20.2d\n"
                       "scvtf v17.4s, v17.4s, #0x4\n"
                       "uzp2 v20.2d, v19.2d, v20.2d\n"
                       "fmul v19.4s, v27.4s, v9.s[2]\n"
                       "fmul v9.4s, v27.4s, v9.s[3]\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "fmla v22.4s, v17.4s, v19.4s\n"
                       "ldr q17, [x22, #0x10]\n"
                       "movi v19.4s, #0x0\n"
                       ".inst 0x4e88a653  // smmla v19.4s, v18.16b, v8.16b\n"
                       "fmla v23.4s, v20.4s, v9.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v9.4s, #0x0\n"
                       ".inst 0x4e9fa654  // smmla v20.4s, v18.16b, v31.16b\n"
                       "ldr q18, [x22, #0x20]\n"
                       ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
                       ".inst 0x4e8fa653  // smmla v19.4s, v18.16b, v15.16b\n"
                       ".inst 0x4e81a654  // smmla v20.4s, v18.16b, v1.16b\n"
                       "ldr q18, [x22, #0x40]\n"
                       ".inst 0x4e95a653  // smmla v19.4s, v18.16b, v21.16b\n"
                       ".inst 0x4e90a654  // smmla v20.4s, v18.16b, v16.16b\n"
                       "ldr q18, [x22, #0x60]\n"
                       ".inst 0x4e9da653  // smmla v19.4s, v18.16b, v29.16b\n"
                       ".inst 0x4e83a654  // smmla v20.4s, v18.16b, v3.16b\n"
                       "movi v18.4s, #0x0\n"
                       ".inst 0x4e9fa632  // smmla v18.4s, v17.16b, v31.16b\n"
                       "ldr q17, [x22, #0x30]\n"
                       ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
                       ".inst 0x4e81a632  // smmla v18.4s, v17.16b, v1.16b\n"
                       "ldr q17, [x22, #0x50]\n"
                       ".inst 0x4e95a629  // smmla v9.4s, v17.16b, v21.16b\n"
                       ".inst 0x4e90a632  // smmla v18.4s, v17.16b, v16.16b\n"
                       "ldr q17, [x22, #0x70]\n"
                       "add x22, x22, #0x88\n"
                       ".inst 0x4e9da629  // smmla v9.4s, v17.16b, v29.16b\n"
                       ".inst 0x4e83a632  // smmla v18.4s, v17.16b, v3.16b\n"
                       "uzp1 v17.2d, v19.2d, v20.2d\n"
                       "uzp2 v20.2d, v19.2d, v20.2d\n"
                       "fmul v19.4s, v27.4s, v0.s[0]\n"
                       "scvtf v17.4s, v17.4s, #0x4\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "fmla v25.4s, v17.4s, v19.4s\n"
                       "ldr q19, [x21, #0x0]\n"
                       "fmul v17.4s, v27.4s, v0.s[1]\n"
                       "fmla v5.4s, v20.4s, v17.4s\n"
                       "ldr q17, [x21, #0x10]\n"
                       "uzp1 v20.2d, v9.2d, v18.2d\n"
                       "uzp2 v9.2d, v9.2d, v18.2d\n"
                       "fmul v18.4s, v27.4s, v0.s[2]\n"
                       "fmul v0.4s, v27.4s, v0.s[3]\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "scvtf v9.4s, v9.4s, #0x4\n"
                       "fmla v7.4s, v20.4s, v18.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v18.4s, #0x0\n"
                       ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
                       ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
                       "ldr q19, [x21, #0x20]\n"
                       "fmla v4.4s, v9.4s, v0.4s\n"
                       "movi v9.4s, #0x0\n"
                       "movi v0.4s, #0x0\n"
                       ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
                       "fmul v8.4s, v27.4s, v26.s[0]\n"
                       ".inst 0x4e9fa620  // smmla v0.4s, v17.16b, v31.16b\n"
                       "ldr q17, [x21, #0x30]\n"
                       ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
                       "fmul v31.4s, v27.4s, v26.s[1]\n"
                       ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
                       "ldr q19, [x21, #0x40]\n"
                       ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
                       "fmul v15.4s, v27.4s, v26.s[2]\n"
                       "fmul v27.4s, v27.4s, v26.s[3]\n"
                       ".inst 0x4e81a620  // smmla v0.4s, v17.16b, v1.16b\n"
                       "ldr q1, [x21, #0x50]\n"
                       ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
                       ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
                       "ldr q26, [x21, #0x60]\n"
                       ".inst 0x4e95a429  // smmla v9.4s, v1.16b, v21.16b\n"
                       ".inst 0x4e90a420  // smmla v0.4s, v1.16b, v16.16b\n"
                       "ldr q21, [x21, #0x70]\n"
                       "add x21, x21, #0x88\n"
                       ".inst 0x4e9da754  // smmla v20.4s, v26.16b, v29.16b\n"
                       ".inst 0x4e83a752  // smmla v18.4s, v26.16b, v3.16b\n"
                       ".inst 0x4e9da6a9  // smmla v9.4s, v21.16b, v29.16b\n"
                       ".inst 0x4e83a6a0  // smmla v0.4s, v21.16b, v3.16b\n"
                       "uzp1 v29.2d, v20.2d, v18.2d\n"
                       "uzp2 v21.2d, v20.2d, v18.2d\n"
                       "scvtf v29.4s, v29.4s, #0x4\n"
                       "uzp1 v18.2d, v9.2d, v0.2d\n"
                       "uzp2 v16.2d, v9.2d, v0.2d\n"
                       "scvtf v21.4s, v21.4s, #0x4\n"
                       "fmla v6.4s, v29.4s, v8.4s\n"
                       "scvtf v18.4s, v18.4s, #0x4\n"
                       "scvtf v16.4s, v16.4s, #0x4\n"
                       "fmla v30.4s, v21.4s, v31.4s\n"
                       "fmla v24.4s, v18.4s, v15.4s\n"
                       "fmla v14.4s, v16.4s, v27.4s\n"
                       "bgt 3b\n"
                       "mov x20, %x[res_ptr]\n"
                       "subs x27, x27, #0x4\n"
                       "add %x[res_ptr], %x[res_ptr], #0x10\n"
                       "str q2, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q10, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q12, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q28, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q11, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q13, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q22, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q23, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q25, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q5, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q7, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q4, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q6, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q30, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q24, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "str q14, [x20, #0x0]\n"
                       "bne 2b\n"
                       "mov x20, #0x4\n"
                       "sub x10, x10, #0x10\n"
                       "cmp x10, #0x10\n"
                       "mov %x[res_ptr], x26\n"
                       "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
                       "bge 1b\n"
                       "4:" // Row loop skip
                       "cbz x10, 9f\n"
                       "5:" // Row tail: Row loop
                       "add x24, %x[b_ptr], #0x8\n"
                       "mov x23, %x[nc]\n"
                       "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
                       "6:" // Row tail: Column loop
                       "movi v2.16b, #0x0\n"
                       "movi v10.16b, #0x0\n"
                       "add x25, %x[a_ptr], #0x8\n"
                       "mov x21, %x[nb]\n"
                       "movi v12.16b, #0x0\n"
                       "movi v28.16b, #0x0\n"
                       "7:" // Row tail: Block loop
                       "ldr q6, [x24, #0x0]\n"
                       "ldr q5, [x24, #0x10]\n"
                       "movi v17.16b, #0x4\n"
                       "movi v8.4s, #0x0\n"
                       "ldr q4, [x25, #0x0]\n"
                       "ldr q13, [x25, #0x10]\n"
                       "movi v27.4s, #0x0\n"
                       "movi v0.4s, #0x0\n"
                       "ldr q31, [x24, #0x20]\n"
                       "ldr q14, [x24, #0x30]\n"
                       "movi v29.4s, #0x0\n"
                       "movi v22.16b, #0xf0\n"
                       "ldr q11, [x25, #0x20]\n"
                       "ldr q23, [x25, #0x30]\n"
                       "sshl v21.16b, v6.16b, v17.16b\n"
                       "sshl v16.16b, v5.16b, v17.16b\n"
                       "ldr q20, [x25, #0x40]\n"
                       "ldr q26, [x25, #0x50]\n"
                       "and v6.16b, v6.16b, v22.16b\n"
                       "and v5.16b, v5.16b, v22.16b\n"
                       "ldr q25, [x25, #0x60]\n"
                       "ldr q3, [x25, #0x70]\n"
                       "sshl v19.16b, v31.16b, v17.16b\n"
                       "sshl v18.16b, v14.16b, v17.16b\n"
                       "ldr d17, [x25, #-0x8]\n"
                       ".inst 0x4e95a488  // smmla v8.4s, v4.16b, v21.16b\n"
                       ".inst 0x4e90a49b  // smmla v27.4s, v4.16b, v16.16b\n"
                       "and v31.16b, v31.16b, v22.16b\n"
                       ".inst 0x4e95a5a0  // smmla v0.4s, v13.16b, v21.16b\n"
                       ".inst 0x4e90a5bd  // smmla v29.4s, v13.16b, v16.16b\n"
                       "and v14.16b, v14.16b, v22.16b\n"
                       "sub x20, x24, #0x8\n"
                       "ldr d16, [x20, #0x0]\n"
                       "subs x21, x21, #0x1\n"
                       "add x25, x25, #0x88\n"
                       "fcvtl v17.4s, v17.4h\n"
                       "add x24, x24, #0x48\n"
                       ".inst 0x4e93a568  // smmla v8.4s, v11.16b, v19.16b\n"
                       ".inst 0x4e92a57b  // smmla v27.4s, v11.16b, v18.16b\n"
                       ".inst 0x4e93a6e0  // smmla v0.4s, v23.16b, v19.16b\n"
                       ".inst 0x4e92a6fd  // smmla v29.4s, v23.16b, v18.16b\n"
                       "fcvtl v16.4s, v16.4h\n"
                       ".inst 0x4e86a688  // smmla v8.4s, v20.16b, v6.16b\n"
                       ".inst 0x4e85a69b  // smmla v27.4s, v20.16b, v5.16b\n"
                       "fmul v23.4s, v16.4s, v17.s[0]\n"
                       "fmul v21.4s, v16.4s, v17.s[1]\n"
                       "fmul v1.4s, v16.4s, v17.s[2]\n"
                       "fmul v20.4s, v16.4s, v17.s[3]\n"
                       ".inst 0x4e86a740  // smmla v0.4s, v26.16b, v6.16b\n"
                       ".inst 0x4e85a75d  // smmla v29.4s, v26.16b, v5.16b\n"
                       ".inst 0x4e9fa728  // smmla v8.4s, v25.16b, v31.16b\n"
                       ".inst 0x4e8ea73b  // smmla v27.4s, v25.16b, v14.16b\n"
                       ".inst 0x4e9fa460  // smmla v0.4s, v3.16b, v31.16b\n"
                       ".inst 0x4e8ea47d  // smmla v29.4s, v3.16b, v14.16b\n"
                       "uzp1 v19.2d, v8.2d, v27.2d\n"
                       "uzp2 v18.2d, v8.2d, v27.2d\n"
                       "scvtf v19.4s, v19.4s, #0x4\n"
                       "uzp1 v17.2d, v0.2d, v29.2d\n"
                       "uzp2 v16.2d, v0.2d, v29.2d\n"
                       "scvtf v18.4s, v18.4s, #0x4\n"
                       "fmla v2.4s, v19.4s, v23.4s\n"
                       "scvtf v17.4s, v17.4s, #0x4\n"
                       "scvtf v16.4s, v16.4s, #0x4\n"
                       "fmla v10.4s, v18.4s, v21.4s\n"
                       "fmla v12.4s, v17.4s, v1.4s\n"
                       "fmla v28.4s, v16.4s, v20.4s\n"
                       "bgt 7b\n"
                       "mov x20, %x[res_ptr]\n"
                       "cmp x10, #0x1\n"
                       "str q2, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "cmp x10, #0x2\n"
                       "str q10, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "cmp x10, #0x3\n"
                       "str q12, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "str q28, [x20, #0x0]\n"
                       "8:" // Row tail: Accumulator store skip
                       "subs x23, x23, #0x4\n"
                       "add %x[res_ptr], %x[res_ptr], #0x10\n"
                       "bne 6b\n"
                       "subs x10, x10, #0x4\n"
                       "add %x[a_ptr], %x[a_ptr], x9\n"
                       "mov %x[res_ptr], x22\n"
                       "bgt 5b\n"
                       "9:" // Row tail: Row loop skip
                       : [a_ptr] "+&r"(a_ptr), [res_ptr] "+&r"(res_ptr)
                       : [b_ptr] "r"(b_ptr), [nr] "r"(nr), [nb] "r"(nb),
                         [res_stride] "r"(res_stride), [nc] "r"(nc)
                       : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5",
                         "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
                         "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
                         "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                         "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23",
                         "x24", "x25", "x26", "x27", "x28");
  return;
}

#ifdef ENABLE_FP16
// FP16-output variant of nntr_gemm_q4_0_4x8_q8_0. Same kernel body, but every
// store to the result buffer is fused with an fcvtn (FP32 4-lane -> FP16
// 4-lane) and reduced to a 64-bit d-register write, and the per-column-tile
// res_ptr advance halves to 8 bytes. Caller passes `bs` as the FP16 element
// stride; res_stride is computed as bs * sizeof(_FP16) inside.
void nntr_gemm_q4_0_4x8_q8_0_fp16(int n, _FP16 *__restrict s, size_t bs,
                                  const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc) {
  const int qk = Q8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 4;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  const void *b_ptr = vx;
  const void *a_ptr = vy;
  _FP16 *res_ptr = s;
  size_t res_stride = bs * sizeof(_FP16);

  __asm__ __volatile__("mov x10, %x[nr]\n"
                       "mov x9, #0x88\n"
                       "cmp x10, #0x10\n"
                       "mul x9, %x[nb], x9\n"
                       "blt 4f\n"
                       "1:" // Row loop
                       "add x28, %x[b_ptr], #0x8\n"
                       "mov x27, %x[nc]\n"
                       "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
                       "2:" // Column loop
                       "add x25, %x[a_ptr], #0x8\n"
                       "movi v2.16b, #0x0\n"
                       "movi v10.16b, #0x0\n"
                       "mov x24, %x[nb]\n"
                       "add x23, x25, x9\n"
                       "movi v12.16b, #0x0\n"
                       "movi v28.16b, #0x0\n"
                       "add x22, x23, x9\n"
                       "movi v11.16b, #0x0\n"
                       "movi v13.16b, #0x0\n"
                       "add x21, x22, x9\n"
                       "movi v22.16b, #0x0\n"
                       "movi v23.16b, #0x0\n"
                       "movi v25.16b, #0x0\n"
                       "movi v5.16b, #0x0\n"
                       "movi v7.16b, #0x0\n"
                       "movi v4.16b, #0x0\n"
                       "movi v6.16b, #0x0\n"
                       "movi v30.16b, #0x0\n"
                       "movi v24.16b, #0x0\n"
                       "movi v14.16b, #0x0\n"
                       "3:" // Block loop
                       "ldr q21, [x28, #0x0]\n"
                       "ldr q16, [x28, #0x10]\n"
                       "movi v1.16b, #0x4\n"
                       "movi v19.4s, #0x0\n"
                       "ldr q27, [x25, #0x0]\n"
                       "ldr q15, [x25, #0x10]\n"
                       "movi v26.4s, #0x0\n"
                       "movi v18.4s, #0x0\n"
                       "ldr q29, [x28, #0x20]\n"
                       "ldr q3, [x28, #0x30]\n"
                       "movi v17.4s, #0x0\n"
                       "movi v0.16b, #0xf0\n"
                       "ldr d20, [x25, #-0x8]\n"
                       "ldr d9, [x23, #-0x8]\n"
                       "sshl v8.16b, v21.16b, v1.16b\n"
                       "sshl v31.16b, v16.16b, v1.16b\n"
                       "and v21.16b, v21.16b, v0.16b\n"
                       "and v16.16b, v16.16b, v0.16b\n"
                       "sub x20, x28, #0x8\n"
                       "subs x24, x24, #0x1\n"
                       "add x28, x28, #0x48\n"
                       ".inst 0x4e88a773  // smmla v19.4s, v27.16b, v8.16b\n"
                       ".inst 0x4e9fa77a  // smmla v26.4s, v27.16b, v31.16b\n"
                       "ldr q27, [x25, #0x20]\n"
                       ".inst 0x4e88a5f2  // smmla v18.4s, v15.16b, v8.16b\n"
                       ".inst 0x4e9fa5f1  // smmla v17.4s, v15.16b, v31.16b\n"
                       "sshl v15.16b, v29.16b, v1.16b\n"
                       "sshl v1.16b, v3.16b, v1.16b\n"
                       "and v29.16b, v29.16b, v0.16b\n"
                       "and v3.16b, v3.16b, v0.16b\n"
                       "ldr q0, [x25, #0x30]\n"
                       "fcvtl v20.4s, v20.4h\n"
                       ".inst 0x4e8fa773  // smmla v19.4s, v27.16b, v15.16b\n"
                       "fcvtl v9.4s, v9.4h\n"
                       ".inst 0x4e81a77a  // smmla v26.4s, v27.16b, v1.16b\n"
                       "ldr q27, [x25, #0x40]\n"
                       ".inst 0x4e8fa412  // smmla v18.4s, v0.16b, v15.16b\n"
                       ".inst 0x4e81a411  // smmla v17.4s, v0.16b, v1.16b\n"
                       "ldr q0, [x25, #0x50]\n"
                       ".inst 0x4e95a773  // smmla v19.4s, v27.16b, v21.16b\n"
                       ".inst 0x4e90a77a  // smmla v26.4s, v27.16b, v16.16b\n"
                       "ldr q27, [x25, #0x60]\n"
                       ".inst 0x4e95a412  // smmla v18.4s, v0.16b, v21.16b\n"
                       ".inst 0x4e90a411  // smmla v17.4s, v0.16b, v16.16b\n"
                       "ldr q0, [x25, #0x70]\n"
                       "add x25, x25, #0x88\n"
                       ".inst 0x4e9da773  // smmla v19.4s, v27.16b, v29.16b\n"
                       ".inst 0x4e83a77a  // smmla v26.4s, v27.16b, v3.16b\n"
                       "ldr d27, [x20, #0x0]\n"
                       ".inst 0x4e9da412  // smmla v18.4s, v0.16b, v29.16b\n"
                       ".inst 0x4e83a411  // smmla v17.4s, v0.16b, v3.16b\n"
                       "fcvtl v27.4s, v27.4h\n"
                       "uzp1 v0.2d, v19.2d, v26.2d\n"
                       "uzp2 v26.2d, v19.2d, v26.2d\n"
                       "fmul v19.4s, v27.4s, v20.s[0]\n"
                       "scvtf v0.4s, v0.4s, #0x4\n"
                       "scvtf v26.4s, v26.4s, #0x4\n"
                       "fmla v2.4s, v0.4s, v19.4s\n"
                       "ldr q19, [x23, #0x0]\n"
                       "uzp1 v0.2d, v18.2d, v17.2d\n"
                       "uzp2 v18.2d, v18.2d, v17.2d\n"
                       "fmul v17.4s, v27.4s, v20.s[1]\n"
                       "scvtf v0.4s, v0.4s, #0x4\n"
                       "scvtf v18.4s, v18.4s, #0x4\n"
                       "fmla v10.4s, v26.4s, v17.4s\n"
                       "ldr q17, [x23, #0x10]\n"
                       "fmul v26.4s, v27.4s, v20.s[2]\n"
                       "fmul v20.4s, v27.4s, v20.s[3]\n"
                       "fmla v12.4s, v0.4s, v26.4s\n"
                       "ldr d0, [x22, #-0x8]\n"
                       "ldr d26, [x21, #-0x8]\n"
                       "fcvtl v0.4s, v0.4h\n"
                       "fmla v28.4s, v18.4s, v20.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v18.4s, #0x0\n"
                       ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
                       ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
                       "ldr q19, [x23, #0x20]\n"
                       "fcvtl v26.4s, v26.4h\n"
                       ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
                       ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
                       "ldr q19, [x23, #0x40]\n"
                       ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
                       ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
                       "ldr q19, [x23, #0x60]\n"
                       ".inst 0x4e9da674  // smmla v20.4s, v19.16b, v29.16b\n"
                       ".inst 0x4e83a672  // smmla v18.4s, v19.16b, v3.16b\n"
                       "uzp1 v19.2d, v20.2d, v18.2d\n"
                       "scvtf v19.4s, v19.4s, #0x4\n"
                       "uzp2 v20.2d, v20.2d, v18.2d\n"
                       "fmul v18.4s, v27.4s, v9.s[0]\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "fmla v11.4s, v19.4s, v18.4s\n"
                       "ldr q18, [x22, #0x0]\n"
                       "fmul v19.4s, v27.4s, v9.s[1]\n"
                       "fmla v13.4s, v20.4s, v19.4s\n"
                       "movi v19.4s, #0x0\n"
                       "movi v20.4s, #0x0\n"
                       ".inst 0x4e88a633  // smmla v19.4s, v17.16b, v8.16b\n"
                       ".inst 0x4e9fa634  // smmla v20.4s, v17.16b, v31.16b\n"
                       "ldr q17, [x23, #0x30]\n"
                       ".inst 0x4e8fa633  // smmla v19.4s, v17.16b, v15.16b\n"
                       ".inst 0x4e81a634  // smmla v20.4s, v17.16b, v1.16b\n"
                       "ldr q17, [x23, #0x50]\n"
                       ".inst 0x4e95a633  // smmla v19.4s, v17.16b, v21.16b\n"
                       ".inst 0x4e90a634  // smmla v20.4s, v17.16b, v16.16b\n"
                       "ldr q17, [x23, #0x70]\n"
                       "add x23, x23, #0x88\n"
                       ".inst 0x4e9da633  // smmla v19.4s, v17.16b, v29.16b\n"
                       ".inst 0x4e83a634  // smmla v20.4s, v17.16b, v3.16b\n"
                       "uzp1 v17.2d, v19.2d, v20.2d\n"
                       "scvtf v17.4s, v17.4s, #0x4\n"
                       "uzp2 v20.2d, v19.2d, v20.2d\n"
                       "fmul v19.4s, v27.4s, v9.s[2]\n"
                       "fmul v9.4s, v27.4s, v9.s[3]\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "fmla v22.4s, v17.4s, v19.4s\n"
                       "ldr q17, [x22, #0x10]\n"
                       "movi v19.4s, #0x0\n"
                       ".inst 0x4e88a653  // smmla v19.4s, v18.16b, v8.16b\n"
                       "fmla v23.4s, v20.4s, v9.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v9.4s, #0x0\n"
                       ".inst 0x4e9fa654  // smmla v20.4s, v18.16b, v31.16b\n"
                       "ldr q18, [x22, #0x20]\n"
                       ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
                       ".inst 0x4e8fa653  // smmla v19.4s, v18.16b, v15.16b\n"
                       ".inst 0x4e81a654  // smmla v20.4s, v18.16b, v1.16b\n"
                       "ldr q18, [x22, #0x40]\n"
                       ".inst 0x4e95a653  // smmla v19.4s, v18.16b, v21.16b\n"
                       ".inst 0x4e90a654  // smmla v20.4s, v18.16b, v16.16b\n"
                       "ldr q18, [x22, #0x60]\n"
                       ".inst 0x4e9da653  // smmla v19.4s, v18.16b, v29.16b\n"
                       ".inst 0x4e83a654  // smmla v20.4s, v18.16b, v3.16b\n"
                       "movi v18.4s, #0x0\n"
                       ".inst 0x4e9fa632  // smmla v18.4s, v17.16b, v31.16b\n"
                       "ldr q17, [x22, #0x30]\n"
                       ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
                       ".inst 0x4e81a632  // smmla v18.4s, v17.16b, v1.16b\n"
                       "ldr q17, [x22, #0x50]\n"
                       ".inst 0x4e95a629  // smmla v9.4s, v17.16b, v21.16b\n"
                       ".inst 0x4e90a632  // smmla v18.4s, v17.16b, v16.16b\n"
                       "ldr q17, [x22, #0x70]\n"
                       "add x22, x22, #0x88\n"
                       ".inst 0x4e9da629  // smmla v9.4s, v17.16b, v29.16b\n"
                       ".inst 0x4e83a632  // smmla v18.4s, v17.16b, v3.16b\n"
                       "uzp1 v17.2d, v19.2d, v20.2d\n"
                       "uzp2 v20.2d, v19.2d, v20.2d\n"
                       "fmul v19.4s, v27.4s, v0.s[0]\n"
                       "scvtf v17.4s, v17.4s, #0x4\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "fmla v25.4s, v17.4s, v19.4s\n"
                       "ldr q19, [x21, #0x0]\n"
                       "fmul v17.4s, v27.4s, v0.s[1]\n"
                       "fmla v5.4s, v20.4s, v17.4s\n"
                       "ldr q17, [x21, #0x10]\n"
                       "uzp1 v20.2d, v9.2d, v18.2d\n"
                       "uzp2 v9.2d, v9.2d, v18.2d\n"
                       "fmul v18.4s, v27.4s, v0.s[2]\n"
                       "fmul v0.4s, v27.4s, v0.s[3]\n"
                       "scvtf v20.4s, v20.4s, #0x4\n"
                       "scvtf v9.4s, v9.4s, #0x4\n"
                       "fmla v7.4s, v20.4s, v18.4s\n"
                       "movi v20.4s, #0x0\n"
                       "movi v18.4s, #0x0\n"
                       ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
                       ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
                       "ldr q19, [x21, #0x20]\n"
                       "fmla v4.4s, v9.4s, v0.4s\n"
                       "movi v9.4s, #0x0\n"
                       "movi v0.4s, #0x0\n"
                       ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
                       "fmul v8.4s, v27.4s, v26.s[0]\n"
                       ".inst 0x4e9fa620  // smmla v0.4s, v17.16b, v31.16b\n"
                       "ldr q17, [x21, #0x30]\n"
                       ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
                       "fmul v31.4s, v27.4s, v26.s[1]\n"
                       ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
                       "ldr q19, [x21, #0x40]\n"
                       ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
                       "fmul v15.4s, v27.4s, v26.s[2]\n"
                       "fmul v27.4s, v27.4s, v26.s[3]\n"
                       ".inst 0x4e81a620  // smmla v0.4s, v17.16b, v1.16b\n"
                       "ldr q1, [x21, #0x50]\n"
                       ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
                       ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
                       "ldr q26, [x21, #0x60]\n"
                       ".inst 0x4e95a429  // smmla v9.4s, v1.16b, v21.16b\n"
                       ".inst 0x4e90a420  // smmla v0.4s, v1.16b, v16.16b\n"
                       "ldr q21, [x21, #0x70]\n"
                       "add x21, x21, #0x88\n"
                       ".inst 0x4e9da754  // smmla v20.4s, v26.16b, v29.16b\n"
                       ".inst 0x4e83a752  // smmla v18.4s, v26.16b, v3.16b\n"
                       ".inst 0x4e9da6a9  // smmla v9.4s, v21.16b, v29.16b\n"
                       ".inst 0x4e83a6a0  // smmla v0.4s, v21.16b, v3.16b\n"
                       "uzp1 v29.2d, v20.2d, v18.2d\n"
                       "uzp2 v21.2d, v20.2d, v18.2d\n"
                       "scvtf v29.4s, v29.4s, #0x4\n"
                       "uzp1 v18.2d, v9.2d, v0.2d\n"
                       "uzp2 v16.2d, v9.2d, v0.2d\n"
                       "scvtf v21.4s, v21.4s, #0x4\n"
                       "fmla v6.4s, v29.4s, v8.4s\n"
                       "scvtf v18.4s, v18.4s, #0x4\n"
                       "scvtf v16.4s, v16.4s, #0x4\n"
                       "fmla v30.4s, v21.4s, v31.4s\n"
                       "fmla v24.4s, v18.4s, v15.4s\n"
                       "fmla v14.4s, v16.4s, v27.4s\n"
                       "bgt 3b\n"
                       "mov x20, %x[res_ptr]\n"
                       "subs x27, x27, #0x4\n"
                       "add %x[res_ptr], %x[res_ptr], #0x8\n"
                       "fcvtn v2.4h, v2.4s\n"
                       "str d2, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v10.4h, v10.4s\n"
                       "str d10, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v12.4h, v12.4s\n"
                       "str d12, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v28.4h, v28.4s\n"
                       "str d28, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v11.4h, v11.4s\n"
                       "str d11, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v13.4h, v13.4s\n"
                       "str d13, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v22.4h, v22.4s\n"
                       "str d22, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v23.4h, v23.4s\n"
                       "str d23, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v25.4h, v25.4s\n"
                       "str d25, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v5.4h, v5.4s\n"
                       "str d5, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v7.4h, v7.4s\n"
                       "str d7, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v4.4h, v4.4s\n"
                       "str d4, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v6.4h, v6.4s\n"
                       "str d6, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v30.4h, v30.4s\n"
                       "str d30, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v24.4h, v24.4s\n"
                       "str d24, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "fcvtn v14.4h, v14.4s\n"
                       "str d14, [x20, #0x0]\n"
                       "bne 2b\n"
                       "mov x20, #0x4\n"
                       "sub x10, x10, #0x10\n"
                       "cmp x10, #0x10\n"
                       "mov %x[res_ptr], x26\n"
                       "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
                       "bge 1b\n"
                       "4:" // Row loop skip
                       "cbz x10, 9f\n"
                       "5:" // Row tail: Row loop
                       "add x24, %x[b_ptr], #0x8\n"
                       "mov x23, %x[nc]\n"
                       "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
                       "6:" // Row tail: Column loop
                       "movi v2.16b, #0x0\n"
                       "movi v10.16b, #0x0\n"
                       "add x25, %x[a_ptr], #0x8\n"
                       "mov x21, %x[nb]\n"
                       "movi v12.16b, #0x0\n"
                       "movi v28.16b, #0x0\n"
                       "7:" // Row tail: Block loop
                       "ldr q6, [x24, #0x0]\n"
                       "ldr q5, [x24, #0x10]\n"
                       "movi v17.16b, #0x4\n"
                       "movi v8.4s, #0x0\n"
                       "ldr q4, [x25, #0x0]\n"
                       "ldr q13, [x25, #0x10]\n"
                       "movi v27.4s, #0x0\n"
                       "movi v0.4s, #0x0\n"
                       "ldr q31, [x24, #0x20]\n"
                       "ldr q14, [x24, #0x30]\n"
                       "movi v29.4s, #0x0\n"
                       "movi v22.16b, #0xf0\n"
                       "ldr q11, [x25, #0x20]\n"
                       "ldr q23, [x25, #0x30]\n"
                       "sshl v21.16b, v6.16b, v17.16b\n"
                       "sshl v16.16b, v5.16b, v17.16b\n"
                       "ldr q20, [x25, #0x40]\n"
                       "ldr q26, [x25, #0x50]\n"
                       "and v6.16b, v6.16b, v22.16b\n"
                       "and v5.16b, v5.16b, v22.16b\n"
                       "ldr q25, [x25, #0x60]\n"
                       "ldr q3, [x25, #0x70]\n"
                       "sshl v19.16b, v31.16b, v17.16b\n"
                       "sshl v18.16b, v14.16b, v17.16b\n"
                       "ldr d17, [x25, #-0x8]\n"
                       ".inst 0x4e95a488  // smmla v8.4s, v4.16b, v21.16b\n"
                       ".inst 0x4e90a49b  // smmla v27.4s, v4.16b, v16.16b\n"
                       "and v31.16b, v31.16b, v22.16b\n"
                       ".inst 0x4e95a5a0  // smmla v0.4s, v13.16b, v21.16b\n"
                       ".inst 0x4e90a5bd  // smmla v29.4s, v13.16b, v16.16b\n"
                       "and v14.16b, v14.16b, v22.16b\n"
                       "sub x20, x24, #0x8\n"
                       "ldr d16, [x20, #0x0]\n"
                       "subs x21, x21, #0x1\n"
                       "add x25, x25, #0x88\n"
                       "fcvtl v17.4s, v17.4h\n"
                       "add x24, x24, #0x48\n"
                       ".inst 0x4e93a568  // smmla v8.4s, v11.16b, v19.16b\n"
                       ".inst 0x4e92a57b  // smmla v27.4s, v11.16b, v18.16b\n"
                       ".inst 0x4e93a6e0  // smmla v0.4s, v23.16b, v19.16b\n"
                       ".inst 0x4e92a6fd  // smmla v29.4s, v23.16b, v18.16b\n"
                       "fcvtl v16.4s, v16.4h\n"
                       ".inst 0x4e86a688  // smmla v8.4s, v20.16b, v6.16b\n"
                       ".inst 0x4e85a69b  // smmla v27.4s, v20.16b, v5.16b\n"
                       "fmul v23.4s, v16.4s, v17.s[0]\n"
                       "fmul v21.4s, v16.4s, v17.s[1]\n"
                       "fmul v1.4s, v16.4s, v17.s[2]\n"
                       "fmul v20.4s, v16.4s, v17.s[3]\n"
                       ".inst 0x4e86a740  // smmla v0.4s, v26.16b, v6.16b\n"
                       ".inst 0x4e85a75d  // smmla v29.4s, v26.16b, v5.16b\n"
                       ".inst 0x4e9fa728  // smmla v8.4s, v25.16b, v31.16b\n"
                       ".inst 0x4e8ea73b  // smmla v27.4s, v25.16b, v14.16b\n"
                       ".inst 0x4e9fa460  // smmla v0.4s, v3.16b, v31.16b\n"
                       ".inst 0x4e8ea47d  // smmla v29.4s, v3.16b, v14.16b\n"
                       "uzp1 v19.2d, v8.2d, v27.2d\n"
                       "uzp2 v18.2d, v8.2d, v27.2d\n"
                       "scvtf v19.4s, v19.4s, #0x4\n"
                       "uzp1 v17.2d, v0.2d, v29.2d\n"
                       "uzp2 v16.2d, v0.2d, v29.2d\n"
                       "scvtf v18.4s, v18.4s, #0x4\n"
                       "fmla v2.4s, v19.4s, v23.4s\n"
                       "scvtf v17.4s, v17.4s, #0x4\n"
                       "scvtf v16.4s, v16.4s, #0x4\n"
                       "fmla v10.4s, v18.4s, v21.4s\n"
                       "fmla v12.4s, v17.4s, v1.4s\n"
                       "fmla v28.4s, v16.4s, v20.4s\n"
                       "bgt 7b\n"
                       "mov x20, %x[res_ptr]\n"
                       "cmp x10, #0x1\n"
                       "fcvtn v2.4h, v2.4s\n"
                       "str d2, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "cmp x10, #0x2\n"
                       "fcvtn v10.4h, v10.4s\n"
                       "str d10, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "cmp x10, #0x3\n"
                       "fcvtn v12.4h, v12.4s\n"
                       "str d12, [x20, #0x0]\n"
                       "add x20, x20, %x[res_stride]\n"
                       "ble 8f\n"
                       "fcvtn v28.4h, v28.4s\n"
                       "str d28, [x20, #0x0]\n"
                       "8:" // Row tail: Accumulator store skip
                       "subs x23, x23, #0x4\n"
                       "add %x[res_ptr], %x[res_ptr], #0x8\n"
                       "bne 6b\n"
                       "subs x10, x10, #0x4\n"
                       "add %x[a_ptr], %x[a_ptr], x9\n"
                       "mov %x[res_ptr], x22\n"
                       "bgt 5b\n"
                       "9:" // Row tail: Row loop skip
                       : [a_ptr] "+&r"(a_ptr), [res_ptr] "+&r"(res_ptr)
                       : [b_ptr] "r"(b_ptr), [nr] "r"(nr), [nb] "r"(nb),
                         [res_stride] "r"(res_stride), [nc] "r"(nc)
                       : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5",
                         "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13",
                         "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
                         "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                         "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23",
                         "x24", "x25", "x26", "x27", "x28");
  return;
}
#endif // ENABLE_FP16

void nntr_gemm_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 8;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  float sumf[4][8];
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_0x8 *b_ptr = (const block_q4_0x8 *)vx + (x * nb);
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          sumf[m][j] = 0.0;
      }
      for (int l = 0; l < nb; l++) {
        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
          for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
              sumi = 0;
              for (int i = 0; i < blocklen; ++i) {
                const int v0 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i]
                           << 4);
                const int v1 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i] &
                           0xF0);
                sumi +=
                  ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                   (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i +
                                     qk / 2 * 4])) >>
                  4;
              }
              sumf[m][j] += sumi * nntr_fp16_to_fp32(b_ptr[l].d[j]) *
                            nntr_fp16_to_fp32(a_ptr[l].d[m]);
            }
          }
        }
      }
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++)
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
      }
    }
  }
}

void nntr_gemm_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = QK_K;
  const int nb = n / qk;
  const int ncols_interleaved = 8;
  const int blocklen = 8;
  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  assert(n % qk == 0);
  assert(nr % 4 == 0);
  assert(nc % ncols_interleaved == 0);

  float sumf[4][8];
  float sum_minf[4][8];
  uint32_t utmp[32];
  int sumi1;
  int sumi2;
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_Kx4 *a_ptr = (const block_q8_Kx4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_Kx8 *b_ptr = (const block_q4_Kx8 *)vx + (x * nb);
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          sumf[m][j] = 0.0;
          sum_minf[m][j] = 0.0;
        }
      }
      for (int l = 0; l < nb; l++) {
        for (int sb = 0; sb < 8; sb++) {
          memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
          utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) |
                             (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
          const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
          utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) |
                             (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
          utmp[sb * 4 + 2] = uaux_0;
          utmp[sb * 4 + 0] &= kmask1;
        }
        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
          uint8_t *scales_0 = (uint8_t *)utmp + (k / 4) * 32;
          uint8_t *scales_1 = (uint8_t *)utmp + (k / 4) * 32 + 16;
          for (int m = 0; m < 4; m++) {
            for (int j = 0; j < ncols_interleaved; j++) {
              sumi1 = 0;
              sumi2 = 0;
              sumi = 0;
              for (int i = 0; i < blocklen; ++i) {
                const int v0 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i] &
                           0xF);
                const int v1 =
                  (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                       j * blocklen + i] >>
                           4);
                sumi1 =
                  (v0 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen +
                                    m * blocklen + i]);
                sumi2 =
                  (v1 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen +
                                    m * blocklen + i + 128]);
                sumi1 = sumi1 * scales_0[j];
                sumi2 = sumi2 * scales_1[j];
                sumi += sumi1 + sumi2;
              }
              sumf[m][j] +=
                sumi * nntr_fp16_to_fp32(b_ptr[l].d[j]) * a_ptr[l].d[m];
            }
          }
        }
        for (int sb = 0; sb < 8; sb++) {
          uint8_t *mins = (uint8_t *)utmp + 8 + sb * 16;
          for (int m = 0; m < 4; m++) {
            const int16_t *bsums =
              a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
            for (int j = 0; j < ncols_interleaved; j++) {
              sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) *
                                nntr_fp16_to_fp32(b_ptr[l].dmin[j]) *
                                a_ptr[l].d[m];
            }
          }
        }
      }
      for (int m = 0; m < 4; m++) {
        for (int j = 0; j < ncols_interleaved; j++) {
          s[(y * 4 + m) * bs + x * ncols_interleaved + j] =
            sumf[m][j] - sum_minf[m][j];
        }
      }
    }
  }
}

void nntr_gemv_q4_K_8x8_q8_K(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = QK_K;
  const int nb = n / qk;
  const int ncols_interleaved = 8;
  const int blocklen = 8;
  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

  float sumf[8];
  float sum_minf[8];
  uint32_t utmp[32];
  int sumi1;
  int sumi2;
  int sumi;

  const block_q8_K *a_ptr = (const block_q8_K *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_q4_Kx8 *b_ptr = (const block_q4_Kx8 *)vx + (x * nb);

    for (int j = 0; j < ncols_interleaved; j++) {
      sumf[j] = 0.0;
      sum_minf[j] = 0.0;
    }
    for (int l = 0; l < nb; l++) {
      for (int sb = 0; sb < 8; sb++) {
        memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
        utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) |
                           (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
        const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
        utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) |
                           (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
        utmp[sb * 4 + 2] = uaux_0;
        utmp[sb * 4 + 0] &= kmask1;
      }
      for (int k = 0; k < (qk / (2 * blocklen)); k++) {
        uint8_t *scales_0 = (uint8_t *)utmp + (k / 4) * 32;
        uint8_t *scales_1 = (uint8_t *)utmp + (k / 4) * 32 + 16;
        for (int j = 0; j < ncols_interleaved; j++) {
          sumi1 = 0;
          sumi2 = 0;
          sumi = 0;
          for (int i = 0; i < blocklen; ++i) {
            const int v0 =
              (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                   j * blocklen + i] &
                       0xF);
            const int v1 =
              (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                   j * blocklen + i] >>
                       4);
            sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
            sumi2 =
              (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
            sumi1 = sumi1 * scales_0[j];
            sumi2 = sumi2 * scales_1[j];
            sumi += sumi1 + sumi2;
          }
          sumf[j] += sumi * nntr_fp16_to_fp32(b_ptr[l].d[j]) * a_ptr[l].d;
        }
      }
      for (int sb = 0; sb < 8; sb++) {
        uint8_t *mins = (uint8_t *)utmp + 8 + sb * 16;
        for (int j = 0; j < ncols_interleaved; j++) {
          sum_minf[j] += mins[j] *
                         (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) *
                         nntr_fp16_to_fp32(b_ptr[l].dmin[j]) * a_ptr[l].d;
        }
      }
    }
    for (int j = 0; j < ncols_interleaved; j++) {
      s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
    }
  }
}

void nntr_quantize_mat_q8_K_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  assert(QK_K == 256);
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  block_q8_Kx4 *__restrict y = (block_q8_Kx4 *)vy;

  // scalar
  const int blck_size_interleave = 8;
  float srcv[4][QK_K];
  float iscale[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      float amax = 0.0f; // absolute max
      float max = 0;

      for (int j = 0; j < QK_K; j++) {
        srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
        // Update the maximum value of the corresponding super block
        if (amax < fabsf(srcv[row_iter][j])) {
          amax = fabsf(srcv[row_iter][j]);
          max = srcv[row_iter][j];
        }
      }

      iscale[row_iter] = amax ? -127.f / max : 0;

      y[i].d[row_iter] = amax ? 1 / iscale[row_iter] : 0;
    }

    for (int j = 0; j < QK_K / 4; j++) {
      y[i].bsums[j] = 0;
    }

    // Quants values are interleaved in sequence of eight bytes from
    // corresponding super blocks Bsums values are interleaved in sequence of
    // four bsums from each super block taken for interleaving i.e first four
    // bsums from the first super block, followed by first four bsums from
    // second super block and so on
    for (int j = 0; j < QK_K * 4; j++) {
      int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
      int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
      src_offset += (j % blck_size_interleave);
      int index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

      float x0 = srcv[src_id][src_offset] * iscale[src_id];
      y[i].qs[j] = nearest_int(x0);
      y[i].bsums[index] += y[i].qs[j];
    }
  }
}

void nntr_quantize_mat_q8_0_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  assert(Q8_0 == 32);
  assert(k % Q8_0 == 0);
  const int nb = k / Q8_0;

  block_q8_0x4 *__restrict y = (block_q8_0x4 *)vy;

  float32x4_t srcv[4][8];
  float id[4];

  for (int i = 0; i < nb; i++) {
    float32x4_t asrcv[8];
    float32x4_t amaxv[8];

    for (int row_iter = 0; row_iter < 4; row_iter++) {
      for (int j = 0; j < 8; j++)
        srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
      for (int j = 0; j < 8; j++)
        asrcv[j] = vabsq_f32(srcv[row_iter][j]);

      for (int j = 0; j < 4; j++)
        amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
      for (int j = 0; j < 2; j++)
        amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
      for (int j = 0; j < 1; j++)
        amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

      const float amax = vmaxvq_f32(amaxv[0]);

      const float d = amax / ((1 << 7) - 1);
      id[row_iter] = d ? 1.0f / d : 0.0f;

      y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);
    }

    for (int j = 0; j < 4; j++) {
      float32x4_t v = vmulq_n_f32(srcv[0][2 * j], id[0]);
      int32x4_t vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 0] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 1] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 2] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 3] = vgetq_lane_s32(vi, 3);
      v = vmulq_n_f32(srcv[0][2 * j + 1], id[0]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 4] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 5] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 6] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 7] = vgetq_lane_s32(vi, 3);

      v = vmulq_n_f32(srcv[1][2 * j], id[1]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 8] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 9] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 10] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 11] = vgetq_lane_s32(vi, 3);
      v = vmulq_n_f32(srcv[1][2 * j + 1], id[1]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 12] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 13] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 14] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 15] = vgetq_lane_s32(vi, 3);

      v = vmulq_n_f32(srcv[2][2 * j], id[2]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 16] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 17] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 18] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 19] = vgetq_lane_s32(vi, 3);
      v = vmulq_n_f32(srcv[2][2 * j + 1], id[2]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 20] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 21] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 22] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 23] = vgetq_lane_s32(vi, 3);

      v = vmulq_n_f32(srcv[3][2 * j], id[3]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 24] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 25] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 26] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 27] = vgetq_lane_s32(vi, 3);
      v = vmulq_n_f32(srcv[3][2 * j + 1], id[3]);
      vi = vcvtnq_s32_f32(v);
      y[i].qs[32 * j + 28] = vgetq_lane_s32(vi, 0);
      y[i].qs[32 * j + 29] = vgetq_lane_s32(vi, 1);
      y[i].qs[32 * j + 30] = vgetq_lane_s32(vi, 2);
      y[i].qs[32 * j + 31] = vgetq_lane_s32(vi, 3);
    }
  }
}

static block_q4_0x4 nntr_make_block_q4_0x4(block_q4_0 *in,
                                           unsigned int blck_size_interleave) {
  block_q4_0x4 out;

  for (int i = 0; i < 4; i++) {
    out.d[i] = in[i].d;
  }

  const int end = Q4_0 * 2 / blck_size_interleave;

  if (blck_size_interleave == 8) {
    const uint64_t xor_mask = 0x8888888888888888ULL;
    for (int i = 0; i < end; ++i) {
      int src_id = i % 4;
      int src_offset = (i / 4) * blck_size_interleave;
      int dst_offset = i * blck_size_interleave;

      uint64_t elems;
      // Using memcpy to avoid unaligned memory accesses
      memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
      elems ^= xor_mask;
      memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }
  } else if (blck_size_interleave == 4) {
    const uint32_t xor_mask = 0x88888888;
    for (int i = 0; i < end; ++i) {
      int src_id = i % 4;
      int src_offset = (i / 4) * blck_size_interleave;
      int dst_offset = i * blck_size_interleave;

      uint32_t elems;
      memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint32_t));
      elems ^= xor_mask;
      memcpy(&out.qs[dst_offset], &elems, sizeof(uint32_t));
    }
  } else {
    assert(false);
  }

  return out;
}

static block_q4_0x8 nntr_make_block_q4_0x8(block_q4_0 *in,
                                           unsigned int blck_size_interleave) {
  block_q4_0x8 out;

  for (int i = 0; i < 8; i++) {
    out.d[i] = in[i].d;
  }

  const int end = QK_0<4>() * 4 / blck_size_interleave;
  const uint64_t xor_mask = 0x8888888888888888ULL;

  for (int i = 0; i < end; ++i) {
    int src_id = i % 8;
    int src_offset = (i / 8) * blck_size_interleave;
    int dst_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
    elems ^= xor_mask;
    memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
  }

  return out;
}

static block_q4_Kx8 make_block_q4_Kx8(block_q4_K *in,
                                      unsigned int blck_size_interleave) {
  block_q4_Kx8 out;
  // Delta(scale) and dmin values of the eight Q4_K structures are copied onto
  // the output interleaved structure
  for (int i = 0; i < 8; i++) {
    out.d[i] = in[i].data.data.d;
  }

  for (int i = 0; i < 8; i++) {
    out.dmin[i] = in[i].data.data.dmin;
  }

  const int end = QK_K * 4 / blck_size_interleave;

  // Interleave Q4_K quants by taking 8 bytes at a time
  for (int i = 0; i < end; ++i) {
    int src_id = i % 8;
    int src_offset = (i / 8) * blck_size_interleave;
    int dst_offset = i * blck_size_interleave;

    uint64_t elems;
    memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
    memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
  }

  // The below logic is designed so as to unpack and rearrange scales and mins
  // values in Q4_K Currently the Q4_K structure has 8 scales and 8 mins packed
  // in 12 bytes ( 6 bits for each value) The output Q4_Kx8 structure has 96
  // bytes Every 12 byte is packed such that it contains scales and mins for
  // corresponding sub blocks from Q4_K structure For eg - First 12 bytes
  // contains 8 scales and 8 mins - each of first sub block from different Q4_K
  // structures
  uint8_t s[8], m[8];

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 8; j++) {
      s[j] = in[j].scales[i] & 63;
      m[j] = in[j].scales[i + 4] & 63;
    }

    out.scales[i * 12] = (s[0] & 63) + ((s[4] & 48) << 2);
    out.scales[i * 12 + 1] = (s[1] & 63) + ((s[5] & 48) << 2);
    out.scales[i * 12 + 2] = (s[2] & 63) + ((s[6] & 48) << 2);
    out.scales[i * 12 + 3] = (s[3] & 63) + ((s[7] & 48) << 2);
    out.scales[i * 12 + 4] = (m[0] & 63) + ((m[4] & 48) << 2);
    out.scales[i * 12 + 5] = (m[1] & 63) + ((m[5] & 48) << 2);
    out.scales[i * 12 + 6] = (m[2] & 63) + ((m[6] & 48) << 2);
    out.scales[i * 12 + 7] = (m[3] & 63) + ((m[7] & 48) << 2);
    out.scales[i * 12 + 8] = (s[4] & 15) + ((m[4] & 15) << 4);
    out.scales[i * 12 + 9] = (s[5] & 15) + ((m[5] & 15) << 4);
    out.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
    out.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 8; j++) {
      s[j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15);
      m[j] =
        ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4);
    }

    out.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
    out.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
    out.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
    out.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
    out.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
    out.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
    out.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
    out.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
    out.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
    out.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
    out.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
    out.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
  }

  return out;
}

int nntr_repack_q4_0_to_q4_0_4_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 4 || interleave_block == 8);
  constexpr int nrows_interleaved = 4;

  block_q4_0x4 *dst_ = (block_q4_0x4 *)dst;
  const block_q4_0 *src = (const block_q4_0 *)data;
  block_q4_0 dst_tmp[4];
  int nblocks = k / Q4_0;

  assert(data_size == nrow * nblocks * sizeof(block_q4_0));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = nntr_make_block_q4_0x4(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}

int nntr_repack_q4_0_to_q4_0_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 8);
  constexpr size_t nrows_interleaved = 8;

  block_q4_0x8 *dst_ = (block_q4_0x8 *)dst;
  const block_q4_0 *src = (const block_q4_0 *)data;
  block_q4_0 dst_tmp[8];
  int nblocks = k / QK_0<4>();

  assert(data_size == nrow * nblocks * sizeof(block_q4_0));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = nntr_make_block_q4_0x8(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}

int nntr_repack_q4_K_to_q4_K_8_bl(void *__restrict dst, int interleave_block,
                                  const void *__restrict data, size_t data_size,
                                  size_t nrow, size_t k) {
  assert(interleave_block == 8);
  constexpr size_t nrows_interleaved = 8;

  block_q4_Kx8 *dst_ = (block_q4_Kx8 *)dst;
  const block_q4_K *src = (const block_q4_K *)data;
  block_q4_K dst_tmp[8];
  int nblocks = k / QK_K;

  assert(data_size == nrow * nblocks * sizeof(block_q4_K));

  if (nrow % nrows_interleaved != 0 || k % 8 != 0) {
    return -1;
  }

  for (size_t b = 0; b < nrow; b += nrows_interleaved) {
    for (int64_t x = 0; x < nblocks; x++) {
      for (size_t i = 0; i < nrows_interleaved; i++) {
        dst_tmp[i] = src[x + i * nblocks];
      }
      *dst_++ = make_block_q4_Kx8(dst_tmp, interleave_block);
    }
    src += nrows_interleaved * nblocks;
  }
  return 0;
}

//===================================== Dot products
//=================================

void nntr_vec_dot_q6_K_q8_K(int n, float *__restrict s, size_t bs,
                            const void *__restrict vx, size_t bx,
                            const void *__restrict vy, size_t by, int nrc) {
  assert(n % QK_K == 0);
  assert(nrc == 1);

  const block_q6_K *__restrict x = (const block_q6_K *)vx;
  const block_q8_K *__restrict y = (const block_q8_K *)vy;

  const int nb = n / QK_K;

  float sum = 0;

  const uint8x16_t m4b = vdupq_n_u8(0xF);
  const int32x4_t vzero = vdupq_n_s32(0);
  // const int8x16_t  m32s = vdupq_n_s8(32);

  const uint8x16_t mone = vdupq_n_u8(3);

  ggml_int8x16x4_t q6bytes;
  ggml_uint8x16x4_t q6h;

  for (int i = 0; i < nb; ++i) {

    const float d_all = nntr_compute_fp16_to_fp32(x[i].d);

    const uint8_t *__restrict q6 = x[i].ql;
    const uint8_t *__restrict qh = x[i].qh;
    const int8_t *__restrict q8 = y[i].qs;

    const int8_t *__restrict scale = x[i].scales;

    const ggml_int16x8x2_t q8sums = ggml_vld1q_s16_x2(y[i].bsums);
    const int8x16_t scales = vld1q_s8(scale);
    const ggml_int16x8x2_t q6scales = {
      {vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales))}};

    const int32x4_t prod =
      vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[0]),
                                    vget_low_s16(q6scales.val[0])),
                          vmull_s16(vget_high_s16(q8sums.val[0]),
                                    vget_high_s16(q6scales.val[0]))),
                vaddq_s32(vmull_s16(vget_low_s16(q8sums.val[1]),
                                    vget_low_s16(q6scales.val[1])),
                          vmull_s16(vget_high_s16(q8sums.val[1]),
                                    vget_high_s16(q6scales.val[1]))));
    int32_t isum_mins = vaddvq_s32(prod);

    int32_t isum = 0;

    for (int j = 0; j < QK_K / 128; ++j) {

      ggml_uint8x16x2_t qhbits = ggml_vld1q_u8_x2(qh);
      qh += 32;
      ggml_uint8x16x4_t q6bits = ggml_vld1q_u8_x4(q6);
      q6 += 64;
      ggml_int8x16x4_t q8bytes = ggml_vld1q_s8_x4(q8);
      q8 += 64;

      q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
      q6h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
      uint8x16_t shifted = vshrq_n_u8(qhbits.val[0], 2);
      q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 2);
      q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

      // q6bytes.val[0] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b),
      // q6h.val[0])), m32s); q6bytes.val[1] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b),
      // q6h.val[1])), m32s); q6bytes.val[2] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b),
      // q6h.val[2])), m32s); q6bytes.val[3] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b),
      // q6h.val[3])), m32s);
      q6bytes.val[0] =
        vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0]));
      q6bytes.val[1] =
        vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1]));
      q6bytes.val[2] =
        vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2]));
      q6bytes.val[3] =
        vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3]));

      isum +=
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) *
          scale[0] +
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) *
          scale[1] +
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) *
          scale[2] +
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) *
          scale[3];

      scale += 4;

      q8bytes = ggml_vld1q_s8_x4(q8);
      q8 += 64;

      shifted = vshrq_n_u8(qhbits.val[0], 4);
      q6h.val[0] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 4);
      q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[0], 6);
      q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
      shifted = vshrq_n_u8(qhbits.val[1], 6);
      q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

      // q6bytes.val[0] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4),
      // q6h.val[0])), m32s); q6bytes.val[1] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4),
      // q6h.val[1])), m32s); q6bytes.val[2] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4),
      // q6h.val[2])), m32s); q6bytes.val[3] =
      // vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4),
      // q6h.val[3])), m32s);
      q6bytes.val[0] =
        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0]));
      q6bytes.val[1] =
        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1]));
      q6bytes.val[2] =
        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2]));
      q6bytes.val[3] =
        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3]));

      isum +=
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) *
          scale[0] +
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) *
          scale[1] +
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) *
          scale[2] +
        vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) *
          scale[3];
      scale += 4;
    }
    // sum += isum * d_all * y[i].d;
    sum += d_all * y[i].d * (isum - 32 * isum_mins);
  }
  *s = sum;
}

void nntr_gemv_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs,
                             const void *__restrict vx,
                             const void *__restrict vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  const int ncols_interleaved = 8;
  const int blocklen = 8;

  assert(n % qk == 0);
  assert(nc % ncols_interleaved == 0);

  {
    float sumf[8];
    int sumi;
    const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_0x8 *b_ptr = (const block_q4_0x8 *)vx + (x * nb);

      for (int j = 0; j < ncols_interleaved; j++)
        sumf[j] = 0.0;
      for (int l = 0; l < nb; l++) {
        for (int k = 0; k < (qk / (2 * blocklen)); k++) {
          for (int j = 0; j < ncols_interleaved; j++) {
            sumi = 0;
            for (int i = 0; i < blocklen; ++i) {
              const int v0 =
                (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                     j * blocklen + i]
                         << 4);
              const int v1 =
                (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen +
                                     j * blocklen + i] &
                         0xF0);
              sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) +
                       (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >>
                      4;
            }
            sumf[j] += sumi * nntr_fp16_to_fp32(b_ptr[l].d[j]) *
                       nntr_fp16_to_fp32(a_ptr[l].d);
          }
        }
      }
      for (int j = 0; j < ncols_interleaved; j++)
        s[x * ncols_interleaved + j] = sumf[j];
    }
  }
}

// ============================================================================
// Q8_0 x Q8_0 GEMM (NEON) -- phase C-2 of native Q8_0 support.
//
// Mirror of the AVX2 nntr_gemm_q8_0_q8_0 in nntr_ggml_impl_avx.cpp. The int8
// dot product core is the SAME helper Q4_0's NEON kernel uses:
//
//   ggml_vdotq_s32(acc, a, b)
//     -> single `vdotq_s32` (NEON SDOT) when __ARM_FEATURE_DOTPROD is set,
//        i.e. armv8.2-a and later.
//     -> vmull_s8 / vpaddlq_s16 / vaddq_s32 fallback otherwise (armv8.0-a).
//
// So the inner-loop instruction mix matches the Q4_0 NEON path exactly. The
// only structural difference is that Q8_0 weights are already 32 int8s per
// block -- no 4-bit nibble unpack, no Q4_0x8 interleave repack.
//
// Layout / shape contract (identical to the AVX2 sibling):
//   n  : K (must be a multiple of QK8_0=32)
//   s  : output FP32, [nr x nc]
//   bs : ldc (in element units, >= nc)
//   vx : weights, const block_q8_0 *, packed [nc x K/32]
//   vy : activations, const block_q8_0 *, packed [nr x K/32]
//   nr : M (activation rows), nc : N (weight rows / output cols)
// ============================================================================
void nntr_gemm_q8_0_q8_0(int n, float *__restrict s, size_t bs,
                         const void *__restrict vx, const void *__restrict vy,
                         int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  assert(n % qk == 0);

  const block_q8_0 *a_base = (const block_q8_0 *)vy; // activations [nr x nb]
  const block_q8_0 *b_base = (const block_q8_0 *)vx; // weights     [nc x nb]

  // Scalar (SDOT) dot of one activation row against one weight row -- used for
  // the M%4 / N%4 edges and as the whole body on non-i8mm targets. Byte-for-
  // byte the inner core of the pre-SMMLA version of this kernel, and the
  // per-block accumulation order matches the 4x4 path below, so results are
  // bit-identical regardless of which path computes an output.
  auto dot_one = [&](const block_q8_0 *arow, const block_q8_0 *brow) -> float {
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
      int32x4_t sumi = vdupq_n_s32(0);
      sumi = ggml_vdotq_s32(sumi, vld1q_s8(arow[b].qs), vld1q_s8(brow[b].qs));
      sumi = ggml_vdotq_s32(sumi, vld1q_s8(arow[b].qs + 16),
                            vld1q_s8(brow[b].qs + 16));
      acc += nntr_fp16_to_fp32(arow[b].d) * nntr_fp16_to_fp32(brow[b].d) *
             (float)vaddvq_s32(sumi);
    }
    return acc;
  };

#if defined(__ARM_FEATURE_MATMUL_INT8)
  // Register-blocked 4x4 SMMLA tiles over plain block_q8_0 rows -- the FP32-
  // output port of nntr_gemm_q8_0_q8_0_fp16 (see nntr_ggml_impl_neon.cpp,
  // conv path); only the stores differ.
  const int nr4 = nr & ~3;
  const int nc4 = nc & ~3;

  for (int m = 0; m < nr4; m += 4) {
    const block_q8_0 *a0 = a_base + (size_t)(m + 0) * nb;
    const block_q8_0 *a1 = a_base + (size_t)(m + 1) * nb;
    const block_q8_0 *a2 = a_base + (size_t)(m + 2) * nb;
    const block_q8_0 *a3 = a_base + (size_t)(m + 3) * nb;

    for (int j = 0; j < nc4; j += 4) {
      const block_q8_0 *b0 = b_base + (size_t)(j + 0) * nb;
      const block_q8_0 *b1 = b_base + (size_t)(j + 1) * nb;
      const block_q8_0 *b2 = b_base + (size_t)(j + 2) * nb;
      const block_q8_0 *b3 = b_base + (size_t)(j + 3) * nb;

      float32x4_t fr0 = vdupq_n_f32(0.0f);
      float32x4_t fr1 = vdupq_n_f32(0.0f);
      float32x4_t fr2 = vdupq_n_f32(0.0f);
      float32x4_t fr3 = vdupq_n_f32(0.0f);

      for (int b = 0; b < nb; ++b) {
        // Per-block int32 accumulators (reset each block: block scales differ).
        int32x4_t acc00 = vdupq_n_s32(0); // rows(0,1) x cols(0,1)
        int32x4_t acc01 = vdupq_n_s32(0); // rows(0,1) x cols(2,3)
        int32x4_t acc10 = vdupq_n_s32(0); // rows(2,3) x cols(0,1)
        int32x4_t acc11 = vdupq_n_s32(0); // rows(2,3) x cols(2,3)

        for (int c = 0; c < qk; c += 8) {
          const int8x16_t ar01 =
            vcombine_s8(vld1_s8(a0[b].qs + c), vld1_s8(a1[b].qs + c));
          const int8x16_t ar23 =
            vcombine_s8(vld1_s8(a2[b].qs + c), vld1_s8(a3[b].qs + c));
          const int8x16_t bc01 =
            vcombine_s8(vld1_s8(b0[b].qs + c), vld1_s8(b1[b].qs + c));
          const int8x16_t bc23 =
            vcombine_s8(vld1_s8(b2[b].qs + c), vld1_s8(b3[b].qs + c));
          // vmmlaq_s32(r,a,b): r[0]+=a_lo.b_lo, r[1]+=a_lo.b_hi,
          //                    r[2]+=a_hi.b_lo, r[3]+=a_hi.b_hi
          acc00 = vmmlaq_s32(acc00, ar01, bc01);
          acc01 = vmmlaq_s32(acc01, ar01, bc23);
          acc10 = vmmlaq_s32(acc10, ar23, bc01);
          acc11 = vmmlaq_s32(acc11, ar23, bc23);
        }

        // Reassemble one int32x4 per row over cols j..j+3.
        const int32x4_t ri0 =
          vcombine_s32(vget_low_s32(acc00), vget_low_s32(acc01));
        const int32x4_t ri1 =
          vcombine_s32(vget_high_s32(acc00), vget_high_s32(acc01));
        const int32x4_t ri2 =
          vcombine_s32(vget_low_s32(acc10), vget_low_s32(acc11));
        const int32x4_t ri3 =
          vcombine_s32(vget_high_s32(acc10), vget_high_s32(acc11));

        const float da0 = nntr_fp16_to_fp32(a0[b].d);
        const float da1 = nntr_fp16_to_fp32(a1[b].d);
        const float da2 = nntr_fp16_to_fp32(a2[b].d);
        const float da3 = nntr_fp16_to_fp32(a3[b].d);
        const float db_arr[4] = {
          nntr_fp16_to_fp32(b0[b].d), nntr_fp16_to_fp32(b1[b].d),
          nntr_fp16_to_fp32(b2[b].d), nntr_fp16_to_fp32(b3[b].d)};
        const float32x4_t db = vld1q_f32(db_arr);

        fr0 = vfmaq_f32(fr0, vmulq_n_f32(vcvtq_f32_s32(ri0), da0), db);
        fr1 = vfmaq_f32(fr1, vmulq_n_f32(vcvtq_f32_s32(ri1), da1), db);
        fr2 = vfmaq_f32(fr2, vmulq_n_f32(vcvtq_f32_s32(ri2), da2), db);
        fr3 = vfmaq_f32(fr3, vmulq_n_f32(vcvtq_f32_s32(ri3), da3), db);
      }

      vst1q_f32(&s[(size_t)(m + 0) * bs + j], fr0);
      vst1q_f32(&s[(size_t)(m + 1) * bs + j], fr1);
      vst1q_f32(&s[(size_t)(m + 2) * bs + j], fr2);
      vst1q_f32(&s[(size_t)(m + 3) * bs + j], fr3);
    }

    // Column remainder (nc % 4) for this 4-row block.
    for (int j = nc4; j < nc; ++j) {
      const block_q8_0 *brow = b_base + (size_t)j * nb;
      s[(size_t)(m + 0) * bs + j] = dot_one(a0, brow);
      s[(size_t)(m + 1) * bs + j] = dot_one(a1, brow);
      s[(size_t)(m + 2) * bs + j] = dot_one(a2, brow);
      s[(size_t)(m + 3) * bs + j] = dot_one(a3, brow);
    }
  }

  // Row remainder (nr % 4).
  for (int m = nr4; m < nr; ++m) {
    const block_q8_0 *arow = a_base + (size_t)m * nb;
    for (int j = 0; j < nc; ++j)
      s[(size_t)m * bs + j] = dot_one(arow, b_base + (size_t)j * nb);
  }
#else
  for (int m = 0; m < nr; ++m) {
    const block_q8_0 *arow = a_base + (size_t)m * nb;
    for (int j = 0; j < nc; ++j)
      s[(size_t)m * bs + j] = dot_one(arow, b_base + (size_t)j * nb);
  }
#endif
}

void nntr_gemv_q8_0_q8_0(int n, float *__restrict s, size_t bs,
                         const void *__restrict vx, const void *__restrict vy,
                         int nr, int nc) {
  (void)nr;
  nntr_gemm_q8_0_q8_0(n, s, bs, vx, vy, 1, nc);
}

#ifdef ENABLE_FP16
// ============================================================================
// SMMLA (i8mm) FP16-output GEMM for plain block_q8_0 weights x plain block_q8_0
// activations. Same numerics as nntr_gemm_q8_0_q8_0 (acc += da*db*int_dot,
// per block), but register-blocked 4x4 so it recovers the throughput the
// scalar SDOT kernel leaves on the table (that kernel produces one output per
// full K-loop, with no M x N reuse -- the actual cause of the W8A16 slowdown,
// not the use of int8 itself). Both operands stay in the plain block_q8_0
// layout -- no offline repack, no interleave -- so the W8A16 activation gather/
// quantize front-end is reused verbatim; only the micro-kernel changes.
//
// Layout / shape contract (identical to nntr_gemm_q8_0_q8_0, FP16 output):
//   n  : K (multiple of QK8_0=32)   s : FP16 out [nr x bs]   bs : ldc (>= nc)
//   vx : weights, const block_q8_0 *, [nc x K/32]
//   vy : activations, const block_q8_0 *, [nr x K/32]
//   nr : M   nc : N
// ============================================================================
void nntr_gemm_q8_0_q8_0_fp16(int n, _FP16 *__restrict s, size_t bs,
                              const void *__restrict vx,
                              const void *__restrict vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  assert(n % qk == 0);

  const block_q8_0 *a_base = (const block_q8_0 *)vy; // activations [nr x nb]
  const block_q8_0 *b_base = (const block_q8_0 *)vx; // weights     [nc x nb]

  // Scalar (SDOT) dot of one activation row against one weight row -- used for
  // the M%4 / N%4 edges and as the whole body on non-i8mm targets. Byte-for-
  // byte the inner core of nntr_gemm_q8_0_q8_0.
  auto dot_one = [&](const block_q8_0 *arow, const block_q8_0 *brow) -> float {
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
      int32x4_t sumi = vdupq_n_s32(0);
      sumi = ggml_vdotq_s32(sumi, vld1q_s8(arow[b].qs), vld1q_s8(brow[b].qs));
      sumi = ggml_vdotq_s32(sumi, vld1q_s8(arow[b].qs + 16),
                            vld1q_s8(brow[b].qs + 16));
      acc += nntr_fp16_to_fp32(arow[b].d) * nntr_fp16_to_fp32(brow[b].d) *
             (float)vaddvq_s32(sumi);
    }
    return acc;
  };

#if defined(__ARM_FEATURE_MATMUL_INT8)
  const int nr4 = nr & ~3;
  const int nc4 = nc & ~3;

  for (int m = 0; m < nr4; m += 4) {
    const block_q8_0 *a0 = a_base + (size_t)(m + 0) * nb;
    const block_q8_0 *a1 = a_base + (size_t)(m + 1) * nb;
    const block_q8_0 *a2 = a_base + (size_t)(m + 2) * nb;
    const block_q8_0 *a3 = a_base + (size_t)(m + 3) * nb;

    for (int j = 0; j < nc4; j += 4) {
      const block_q8_0 *b0 = b_base + (size_t)(j + 0) * nb;
      const block_q8_0 *b1 = b_base + (size_t)(j + 1) * nb;
      const block_q8_0 *b2 = b_base + (size_t)(j + 2) * nb;
      const block_q8_0 *b3 = b_base + (size_t)(j + 3) * nb;

      float32x4_t fr0 = vdupq_n_f32(0.0f);
      float32x4_t fr1 = vdupq_n_f32(0.0f);
      float32x4_t fr2 = vdupq_n_f32(0.0f);
      float32x4_t fr3 = vdupq_n_f32(0.0f);

      for (int b = 0; b < nb; ++b) {
        // Per-block int32 accumulators (reset each block: block scales differ).
        int32x4_t acc00 = vdupq_n_s32(0); // rows(0,1) x cols(0,1)
        int32x4_t acc01 = vdupq_n_s32(0); // rows(0,1) x cols(2,3)
        int32x4_t acc10 = vdupq_n_s32(0); // rows(2,3) x cols(0,1)
        int32x4_t acc11 = vdupq_n_s32(0); // rows(2,3) x cols(2,3)

        for (int c = 0; c < qk; c += 8) {
          const int8x16_t ar01 =
            vcombine_s8(vld1_s8(a0[b].qs + c), vld1_s8(a1[b].qs + c));
          const int8x16_t ar23 =
            vcombine_s8(vld1_s8(a2[b].qs + c), vld1_s8(a3[b].qs + c));
          const int8x16_t bc01 =
            vcombine_s8(vld1_s8(b0[b].qs + c), vld1_s8(b1[b].qs + c));
          const int8x16_t bc23 =
            vcombine_s8(vld1_s8(b2[b].qs + c), vld1_s8(b3[b].qs + c));
          // vmmlaq_s32(r,a,b): r[0]+=a_lo.b_lo, r[1]+=a_lo.b_hi,
          //                    r[2]+=a_hi.b_lo, r[3]+=a_hi.b_hi
          acc00 = vmmlaq_s32(acc00, ar01, bc01);
          acc01 = vmmlaq_s32(acc01, ar01, bc23);
          acc10 = vmmlaq_s32(acc10, ar23, bc01);
          acc11 = vmmlaq_s32(acc11, ar23, bc23);
        }

        // Reassemble one int32x4 per row over cols j..j+3.
        const int32x4_t ri0 =
          vcombine_s32(vget_low_s32(acc00), vget_low_s32(acc01));
        const int32x4_t ri1 =
          vcombine_s32(vget_high_s32(acc00), vget_high_s32(acc01));
        const int32x4_t ri2 =
          vcombine_s32(vget_low_s32(acc10), vget_low_s32(acc11));
        const int32x4_t ri3 =
          vcombine_s32(vget_high_s32(acc10), vget_high_s32(acc11));

        const float da0 = nntr_fp16_to_fp32(a0[b].d);
        const float da1 = nntr_fp16_to_fp32(a1[b].d);
        const float da2 = nntr_fp16_to_fp32(a2[b].d);
        const float da3 = nntr_fp16_to_fp32(a3[b].d);
        const float db_arr[4] = {
          nntr_fp16_to_fp32(b0[b].d), nntr_fp16_to_fp32(b1[b].d),
          nntr_fp16_to_fp32(b2[b].d), nntr_fp16_to_fp32(b3[b].d)};
        const float32x4_t db = vld1q_f32(db_arr);

        fr0 = vfmaq_f32(fr0, vmulq_n_f32(vcvtq_f32_s32(ri0), da0), db);
        fr1 = vfmaq_f32(fr1, vmulq_n_f32(vcvtq_f32_s32(ri1), da1), db);
        fr2 = vfmaq_f32(fr2, vmulq_n_f32(vcvtq_f32_s32(ri2), da2), db);
        fr3 = vfmaq_f32(fr3, vmulq_n_f32(vcvtq_f32_s32(ri3), da3), db);
      }

      float tmp[4];
      vst1q_f32(tmp, fr0);
      for (int k = 0; k < 4; ++k)
        s[(size_t)(m + 0) * bs + j + k] = (_FP16)tmp[k];
      vst1q_f32(tmp, fr1);
      for (int k = 0; k < 4; ++k)
        s[(size_t)(m + 1) * bs + j + k] = (_FP16)tmp[k];
      vst1q_f32(tmp, fr2);
      for (int k = 0; k < 4; ++k)
        s[(size_t)(m + 2) * bs + j + k] = (_FP16)tmp[k];
      vst1q_f32(tmp, fr3);
      for (int k = 0; k < 4; ++k)
        s[(size_t)(m + 3) * bs + j + k] = (_FP16)tmp[k];
    }

    // Column remainder (nc % 4) for this 4-row block.
    for (int j = nc4; j < nc; ++j) {
      const block_q8_0 *brow = b_base + (size_t)j * nb;
      s[(size_t)(m + 0) * bs + j] = (_FP16)dot_one(a0, brow);
      s[(size_t)(m + 1) * bs + j] = (_FP16)dot_one(a1, brow);
      s[(size_t)(m + 2) * bs + j] = (_FP16)dot_one(a2, brow);
      s[(size_t)(m + 3) * bs + j] = (_FP16)dot_one(a3, brow);
    }
  }

  // Row remainder (nr % 4).
  for (int m = nr4; m < nr; ++m) {
    const block_q8_0 *arow = a_base + (size_t)m * nb;
    for (int j = 0; j < nc; ++j)
      s[(size_t)m * bs + j] = (_FP16)dot_one(arow, b_base + (size_t)j * nb);
  }
#else
  for (int m = 0; m < nr; ++m) {
    const block_q8_0 *arow = a_base + (size_t)m * nb;
    for (int j = 0; j < nc; ++j)
      s[(size_t)m * bs + j] = (_FP16)dot_one(arow, b_base + (size_t)j * nb);
  }
#endif
}

// Interleaved (q8_0x4) SMMLA GEMM: both weight and activation are pre-packed to
// block_q8_0x4 (4 rows interleaved, byte layout qs[32*sub + row*8 + c], mirroring
// __ggml_quantize_mat_q8_0_4x8 and the offline Python repack_q8_0). This is the
// speed variant of nntr_gemm_q8_0_q8_0_fp16: the register-blocked 4x4 compute is
// byte-for-byte the same, but each i8mm operand is now a single contiguous
// vld1q_s8 instead of a vcombine of two scattered per-row vld1_s8 loads -- fewer
// load micro-ops and cache-friendly streaming, matching the proven Q4_0 x8 path.
// Requires nr % 4 == 0 (front-end pads the M-tail); nc%4 handled scalar.
void nntr_gemm_q8_0_q8_0_4x4_fp16(int n, _FP16 *__restrict s, size_t bs,
                                  const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  assert(n % qk == 0);
  assert(nr % 4 == 0);

  const block_q8_0x4 *a_sbase = (const block_q8_0x4 *)vy; // act    [nr/4][nb]
  const block_q8_0x4 *b_sbase = (const block_q8_0x4 *)vx; // weight [nc/4][nb]

  // Scalar interleaved dot of one act row (ar in its super-block) against one
  // weight row (wr in its super-block) -- used only for the nc%4 column edge.
  auto dot_one = [&](const block_q8_0x4 *a, int ar, const block_q8_0x4 *b,
                     int wr) -> float {
    float acc = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
      int32_t si = 0;
      for (int sub = 0; sub < 4; ++sub)
        for (int c = 0; c < 8; ++c)
          si += (int32_t)a[bi].qs[32 * sub + ar * 8 + c] *
                (int32_t)b[bi].qs[32 * sub + wr * 8 + c];
      acc += nntr_fp16_to_fp32(a[bi].d[ar]) * nntr_fp16_to_fp32(b[bi].d[wr]) *
             (float)si;
    }
    return acc;
  };

#if defined(__ARM_FEATURE_MATMUL_INT8)
  const int nc4 = nc & ~3;
  const int nr16 = nr & ~15;

  // Fold a 4x4 int32 tile (two vmmla acc pairs for one act super-block) into the
  // four FP row accumulators. Shared by the 8-row main loop and the 4-row tail.
  auto fold4 = [](int32x4_t acc01_lo, int32x4_t acc01_hi, int32x4_t acc23_lo,
                  int32x4_t acc23_hi, float32x4_t da, float32x4_t db,
                  float32x4_t &f0, float32x4_t &f1, float32x4_t &f2,
                  float32x4_t &f3) {
    const int32x4_t ri0 =
      vcombine_s32(vget_low_s32(acc01_lo), vget_low_s32(acc01_hi));
    const int32x4_t ri1 =
      vcombine_s32(vget_high_s32(acc01_lo), vget_high_s32(acc01_hi));
    const int32x4_t ri2 =
      vcombine_s32(vget_low_s32(acc23_lo), vget_low_s32(acc23_hi));
    const int32x4_t ri3 =
      vcombine_s32(vget_high_s32(acc23_lo), vget_high_s32(acc23_hi));
    f0 = vfmaq_f32(f0, vmulq_n_f32(vcvtq_f32_s32(ri0), vgetq_lane_f32(da, 0)), db);
    f1 = vfmaq_f32(f1, vmulq_n_f32(vcvtq_f32_s32(ri1), vgetq_lane_f32(da, 1)), db);
    f2 = vfmaq_f32(f2, vmulq_n_f32(vcvtq_f32_s32(ri2), vgetq_lane_f32(da, 2)), db);
    f3 = vfmaq_f32(f3, vmulq_n_f32(vcvtq_f32_s32(ri3), vgetq_lane_f32(da, 3)), db);
  };
  auto store4 = [&](int row, int j, float32x4_t f) {
    float tmp[4];
    vst1q_f32(tmp, f);
    for (int k = 0; k < 4; ++k)
      s[(size_t)row * bs + j + k] = (_FP16)tmp[k];
  };

  // Main loop: 16 rows (four act super-blocks) share ONE set of 8 weight
  // registers per block, matching ggml's Q4 weight reuse (16x vs a 4-row tile's
  // 4x). Per super-block only 4 int32 accumulators are live, folded immediately,
  // so 16 FP accumulators + 8 weight regs stay resident without spilling.
  for (int m = 0; m < nr16; m += 16) {
    const block_q8_0x4 *aA = a_sbase + (size_t)(m / 4 + 0) * nb;
    const block_q8_0x4 *aB = a_sbase + (size_t)(m / 4 + 1) * nb;
    const block_q8_0x4 *aC = a_sbase + (size_t)(m / 4 + 2) * nb;
    const block_q8_0x4 *aD = a_sbase + (size_t)(m / 4 + 3) * nb;

    for (int j = 0; j < nc4; j += 4) {
      const block_q8_0x4 *b = b_sbase + (size_t)(j / 4) * nb;

      float32x4_t fA0 = vdupq_n_f32(0.0f), fA1 = vdupq_n_f32(0.0f);
      float32x4_t fA2 = vdupq_n_f32(0.0f), fA3 = vdupq_n_f32(0.0f);
      float32x4_t fB0 = vdupq_n_f32(0.0f), fB1 = vdupq_n_f32(0.0f);
      float32x4_t fB2 = vdupq_n_f32(0.0f), fB3 = vdupq_n_f32(0.0f);
      float32x4_t fC0 = vdupq_n_f32(0.0f), fC1 = vdupq_n_f32(0.0f);
      float32x4_t fC2 = vdupq_n_f32(0.0f), fC3 = vdupq_n_f32(0.0f);
      float32x4_t fD0 = vdupq_n_f32(0.0f), fD1 = vdupq_n_f32(0.0f);
      float32x4_t fD2 = vdupq_n_f32(0.0f), fD3 = vdupq_n_f32(0.0f);

      for (int bi = 0; bi < nb; ++bi) {
        // 8 weight registers: 4 k-subgroups x 2 col-pairs, loaded once, reused
        // across all four act super-blocks (16 rows).
        const int8x16_t w0a = vld1q_s8(b[bi].qs + 0);
        const int8x16_t w0b = vld1q_s8(b[bi].qs + 16);
        const int8x16_t w1a = vld1q_s8(b[bi].qs + 32);
        const int8x16_t w1b = vld1q_s8(b[bi].qs + 48);
        const int8x16_t w2a = vld1q_s8(b[bi].qs + 64);
        const int8x16_t w2b = vld1q_s8(b[bi].qs + 80);
        const int8x16_t w3a = vld1q_s8(b[bi].qs + 96);
        const int8x16_t w3b = vld1q_s8(b[bi].qs + 112);

        const float32x4_t db = vcvt_f32_f16(vld1_f16((const __fp16 *)b[bi].d));

        // Compute one 4-row super-block against the resident weights, then fold.
        auto do_sb = [&](const block_q8_0x4 *a, float32x4_t &f0, float32x4_t &f1,
                         float32x4_t &f2, float32x4_t &f3) {
          int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
          int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
          const int8x16_t r0a = vld1q_s8(a[bi].qs + 0);
          const int8x16_t r0b = vld1q_s8(a[bi].qs + 16);
          c00 = vmmlaq_s32(c00, r0a, w0a);
          c01 = vmmlaq_s32(c01, r0a, w0b);
          c10 = vmmlaq_s32(c10, r0b, w0a);
          c11 = vmmlaq_s32(c11, r0b, w0b);
          const int8x16_t r1a = vld1q_s8(a[bi].qs + 32);
          const int8x16_t r1b = vld1q_s8(a[bi].qs + 48);
          c00 = vmmlaq_s32(c00, r1a, w1a);
          c01 = vmmlaq_s32(c01, r1a, w1b);
          c10 = vmmlaq_s32(c10, r1b, w1a);
          c11 = vmmlaq_s32(c11, r1b, w1b);
          const int8x16_t r2a = vld1q_s8(a[bi].qs + 64);
          const int8x16_t r2b = vld1q_s8(a[bi].qs + 80);
          c00 = vmmlaq_s32(c00, r2a, w2a);
          c01 = vmmlaq_s32(c01, r2a, w2b);
          c10 = vmmlaq_s32(c10, r2b, w2a);
          c11 = vmmlaq_s32(c11, r2b, w2b);
          const int8x16_t r3a = vld1q_s8(a[bi].qs + 96);
          const int8x16_t r3b = vld1q_s8(a[bi].qs + 112);
          c00 = vmmlaq_s32(c00, r3a, w3a);
          c01 = vmmlaq_s32(c01, r3a, w3b);
          c10 = vmmlaq_s32(c10, r3b, w3a);
          c11 = vmmlaq_s32(c11, r3b, w3b);
          const float32x4_t da = vcvt_f32_f16(vld1_f16((const __fp16 *)a[bi].d));
          fold4(c00, c01, c10, c11, da, db, f0, f1, f2, f3);
        };

        do_sb(aA, fA0, fA1, fA2, fA3);
        do_sb(aB, fB0, fB1, fB2, fB3);
        do_sb(aC, fC0, fC1, fC2, fC3);
        do_sb(aD, fD0, fD1, fD2, fD3);
      }

      store4(m + 0, j, fA0);
      store4(m + 1, j, fA1);
      store4(m + 2, j, fA2);
      store4(m + 3, j, fA3);
      store4(m + 4, j, fB0);
      store4(m + 5, j, fB1);
      store4(m + 6, j, fB2);
      store4(m + 7, j, fB3);
      store4(m + 8, j, fC0);
      store4(m + 9, j, fC1);
      store4(m + 10, j, fC2);
      store4(m + 11, j, fC3);
      store4(m + 12, j, fD0);
      store4(m + 13, j, fD1);
      store4(m + 14, j, fD2);
      store4(m + 15, j, fD3);
    }

    // Column remainder (nc % 4) for these 16 rows.
    for (int j = nc4; j < nc; ++j) {
      const block_q8_0x4 *b = b_sbase + (size_t)(j / 4) * nb;
      const int wr = j % 4;
      const block_q8_0x4 *ap[4] = {aA, aB, aC, aD};
      for (int sb = 0; sb < 4; ++sb)
        for (int rr = 0; rr < 4; ++rr)
          s[(size_t)(m + sb * 4 + rr) * bs + j] = (_FP16)dot_one(ap[sb], rr, b, wr);
    }
  }

  // M-tail: remaining rows (nr - nr16), guaranteed multiple of 4, in 4-row tiles.
  for (int m = nr16; m < nr; m += 4) {
    const block_q8_0x4 *a = a_sbase + (size_t)(m / 4) * nb;

    for (int j = 0; j < nc4; j += 4) {
      const block_q8_0x4 *b = b_sbase + (size_t)(j / 4) * nb;

      float32x4_t fr0 = vdupq_n_f32(0.0f), fr1 = vdupq_n_f32(0.0f);
      float32x4_t fr2 = vdupq_n_f32(0.0f), fr3 = vdupq_n_f32(0.0f);

      for (int bi = 0; bi < nb; ++bi) {
        int32x4_t acc00 = vdupq_n_s32(0), acc01 = vdupq_n_s32(0);
        int32x4_t acc10 = vdupq_n_s32(0), acc11 = vdupq_n_s32(0);
        for (int sub = 0; sub < 4; ++sub) {
          const int8x16_t ar01 = vld1q_s8(a[bi].qs + 32 * sub);
          const int8x16_t ar23 = vld1q_s8(a[bi].qs + 32 * sub + 16);
          const int8x16_t bc01 = vld1q_s8(b[bi].qs + 32 * sub);
          const int8x16_t bc23 = vld1q_s8(b[bi].qs + 32 * sub + 16);
          acc00 = vmmlaq_s32(acc00, ar01, bc01);
          acc01 = vmmlaq_s32(acc01, ar01, bc23);
          acc10 = vmmlaq_s32(acc10, ar23, bc01);
          acc11 = vmmlaq_s32(acc11, ar23, bc23);
        }
        const float32x4_t db = vcvt_f32_f16(vld1_f16((const __fp16 *)b[bi].d));
        const float32x4_t da = vcvt_f32_f16(vld1_f16((const __fp16 *)a[bi].d));
        fold4(acc00, acc01, acc10, acc11, da, db, fr0, fr1, fr2, fr3);
      }

      store4(m + 0, j, fr0);
      store4(m + 1, j, fr1);
      store4(m + 2, j, fr2);
      store4(m + 3, j, fr3);
    }

    for (int j = nc4; j < nc; ++j) {
      const block_q8_0x4 *b = b_sbase + (size_t)(j / 4) * nb;
      const int wr = j % 4;
      for (int rr = 0; rr < 4; ++rr)
        s[(size_t)(m + rr) * bs + j] = (_FP16)dot_one(a, rr, b, wr);
    }
  }
#else
  for (int m = 0; m < nr; ++m) {
    const block_q8_0x4 *a = a_sbase + (size_t)(m / 4) * nb;
    const int ar = m % 4;
    for (int j = 0; j < nc; ++j)
      s[(size_t)m * bs + j] =
        (_FP16)dot_one(a, ar, b_sbase + (size_t)(j / 4) * nb, j % 4);
  }
#endif
}

void nntr_gemm_q8_0_4x8_q8_0_fp16(int n, NNTR_GGML_FP16 *__restrict s,
                                  size_t bs, const void *__restrict vx,
                                  const void *__restrict vy, int nr, int nc) {
  const int qk = QK8_0;
  const int nb = n / qk;
  assert(n % qk == 0);
  assert(nr % 4 == 0);

  const block_q8_0x4 *b_sbase = (const block_q8_0x4 *)vx; // weight [nc/4][nb]
  const block_q8_0x4 *a_sbase = (const block_q8_0x4 *)vy; // act    [nr/4][nb]

  // Scalar interleaved dot of one act row against one weight row -- used for
  // the nc%4 column edge and the no-i8mm fallback.
  auto dot_one = [&](const block_q8_0x4 *a, int ar, const block_q8_0x4 *b,
                     int wr) -> float {
    float acc = 0.0f;
    for (int bi = 0; bi < nb; ++bi) {
      int32_t si = 0;
      for (int sub = 0; sub < 4; ++sub)
        for (int c = 0; c < 8; ++c)
          si += (int32_t)a[bi].qs[32 * sub + ar * 8 + c] *
                (int32_t)b[bi].qs[32 * sub + wr * 8 + c];
      acc += nntr_fp16_to_fp32(a[bi].d[ar]) * nntr_fp16_to_fp32(b[bi].d[wr]) *
             (float)si;
    }
    return acc;
  };

#if defined(__ARM_FEATURE_MATMUL_INT8)
  const int nc4 = nc & ~3;
  const int nr8 = nr & ~7;

  // Fold one act super-block's 2x2 SMMLA pairs into its 4 fp32 accumulators.
  // Shared by the 8-row main loop and the 4-row tail below.
  auto fold4 = [](int32x4_t acc01_lo, int32x4_t acc01_hi, int32x4_t acc23_lo,
                  int32x4_t acc23_hi, float32x4_t da, float32x4_t db,
                  float32x4_t &f0, float32x4_t &f1, float32x4_t &f2,
                  float32x4_t &f3) {
    const int32x4_t ri0 =
      vcombine_s32(vget_low_s32(acc01_lo), vget_low_s32(acc01_hi));
    const int32x4_t ri1 =
      vcombine_s32(vget_high_s32(acc01_lo), vget_high_s32(acc01_hi));
    const int32x4_t ri2 =
      vcombine_s32(vget_low_s32(acc23_lo), vget_low_s32(acc23_hi));
    const int32x4_t ri3 =
      vcombine_s32(vget_high_s32(acc23_lo), vget_high_s32(acc23_hi));
    f0 = vfmaq_f32(f0, vmulq_n_f32(vcvtq_f32_s32(ri0), vgetq_lane_f32(da, 0)), db);
    f1 = vfmaq_f32(f1, vmulq_n_f32(vcvtq_f32_s32(ri1), vgetq_lane_f32(da, 1)), db);
    f2 = vfmaq_f32(f2, vmulq_n_f32(vcvtq_f32_s32(ri2), vgetq_lane_f32(da, 2)), db);
    f3 = vfmaq_f32(f3, vmulq_n_f32(vcvtq_f32_s32(ri3), vgetq_lane_f32(da, 3)), db);
  };
  auto store4 = [&](int row, int j, float32x4_t f) {
    float tmp[4];
    vst1q_f32(tmp, f);
    for (int k = 0; k < 4; ++k)
      s[(size_t)row * bs + j + k] = (NNTR_GGML_FP16)tmp[k];
  };

  // Q8_0 weights need no nibble unpack (unlike Q4_0): every block's 8 rows
  // are already plain int8, so they feed vmmlaq_s32 as direct loads -- no
  // shift/mask before and no /16 fixed-point rescale after (the Q4_0 kernel
  // needs both because it pre-shifts unpacked nibbles into the int8 high
  // bits). Main loop: 8 rows (two act super-blocks) share ONE set of 8
  // weight registers per block. NEON only has 32 vector registers -- with 4
  // super-blocks resident (16 accumulators + 8 weight regs + temporaries)
  // the compiler was forced to spill accumulators to the stack every block
  // iteration (measured: 137 `[sp]` accesses in the compiled hot loop vs 9
  // for the hand-written Q4_0 asm kernel), which dominated over any savings
  // from skipping the Q4_0 nibble unpack/rescale. Two super-blocks (8
  // accumulators + 8 weight regs) fits without spilling.
  for (int j = 0; j < nc4; j += 4) {
    const block_q8_0x4 *b = b_sbase + (size_t)(j / 4) * nb;

    for (int m = 0; m < nr8; m += 8) {
      const block_q8_0x4 *aA = a_sbase + (size_t)(m / 4 + 0) * nb;
      const block_q8_0x4 *aB = a_sbase + (size_t)(m / 4 + 1) * nb;

      float32x4_t fA0 = vdupq_n_f32(0.0f), fA1 = vdupq_n_f32(0.0f);
      float32x4_t fA2 = vdupq_n_f32(0.0f), fA3 = vdupq_n_f32(0.0f);
      float32x4_t fB0 = vdupq_n_f32(0.0f), fB1 = vdupq_n_f32(0.0f);
      float32x4_t fB2 = vdupq_n_f32(0.0f), fB3 = vdupq_n_f32(0.0f);

      for (int bi = 0; bi < nb; ++bi) {
        const int8x16_t w01_0 = vld1q_s8(b[bi].qs + 0);
        const int8x16_t w23_0 = vld1q_s8(b[bi].qs + 16);
        const int8x16_t w01_1 = vld1q_s8(b[bi].qs + 32);
        const int8x16_t w23_1 = vld1q_s8(b[bi].qs + 48);
        const int8x16_t w01_2 = vld1q_s8(b[bi].qs + 64);
        const int8x16_t w23_2 = vld1q_s8(b[bi].qs + 80);
        const int8x16_t w01_3 = vld1q_s8(b[bi].qs + 96);
        const int8x16_t w23_3 = vld1q_s8(b[bi].qs + 112);

        const float32x4_t db = vcvt_f32_f16(vld1_f16((const __fp16 *)b[bi].d));

        // Compute one 4-row act super-block against the resident weights.
        auto do_sb = [&](const block_q8_0x4 *a, float32x4_t &f0,
                         float32x4_t &f1, float32x4_t &f2, float32x4_t &f3) {
          int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
          int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
          const int8x16_t a01_0 = vld1q_s8(a[bi].qs + 0);
          const int8x16_t a23_0 = vld1q_s8(a[bi].qs + 16);
          c00 = vmmlaq_s32(c00, a01_0, w01_0);
          c01 = vmmlaq_s32(c01, a01_0, w23_0);
          c10 = vmmlaq_s32(c10, a23_0, w01_0);
          c11 = vmmlaq_s32(c11, a23_0, w23_0);
          const int8x16_t a01_1 = vld1q_s8(a[bi].qs + 32);
          const int8x16_t a23_1 = vld1q_s8(a[bi].qs + 48);
          c00 = vmmlaq_s32(c00, a01_1, w01_1);
          c01 = vmmlaq_s32(c01, a01_1, w23_1);
          c10 = vmmlaq_s32(c10, a23_1, w01_1);
          c11 = vmmlaq_s32(c11, a23_1, w23_1);
          const int8x16_t a01_2 = vld1q_s8(a[bi].qs + 64);
          const int8x16_t a23_2 = vld1q_s8(a[bi].qs + 80);
          c00 = vmmlaq_s32(c00, a01_2, w01_2);
          c01 = vmmlaq_s32(c01, a01_2, w23_2);
          c10 = vmmlaq_s32(c10, a23_2, w01_2);
          c11 = vmmlaq_s32(c11, a23_2, w23_2);
          const int8x16_t a01_3 = vld1q_s8(a[bi].qs + 96);
          const int8x16_t a23_3 = vld1q_s8(a[bi].qs + 112);
          c00 = vmmlaq_s32(c00, a01_3, w01_3);
          c01 = vmmlaq_s32(c01, a01_3, w23_3);
          c10 = vmmlaq_s32(c10, a23_3, w01_3);
          c11 = vmmlaq_s32(c11, a23_3, w23_3);
          const float32x4_t da = vcvt_f32_f16(vld1_f16((const __fp16 *)a[bi].d));
          fold4(c00, c01, c10, c11, da, db, f0, f1, f2, f3);
        };

        do_sb(aA, fA0, fA1, fA2, fA3);
        do_sb(aB, fB0, fB1, fB2, fB3);
      }

      store4(m + 0, j, fA0);
      store4(m + 1, j, fA1);
      store4(m + 2, j, fA2);
      store4(m + 3, j, fA3);
      store4(m + 4, j, fB0);
      store4(m + 5, j, fB1);
      store4(m + 6, j, fB2);
      store4(m + 7, j, fB3);
    }

    // M-tail: remaining rows (nr - nr8), guaranteed multiple of 4.
    for (int m = nr8; m < nr; m += 4) {
      const block_q8_0x4 *a = a_sbase + (size_t)(m / 4) * nb;

      float32x4_t fr0 = vdupq_n_f32(0.0f), fr1 = vdupq_n_f32(0.0f);
      float32x4_t fr2 = vdupq_n_f32(0.0f), fr3 = vdupq_n_f32(0.0f);

      for (int bi = 0; bi < nb; ++bi) {
        const int8x16_t w01_0 = vld1q_s8(b[bi].qs + 0);
        const int8x16_t w23_0 = vld1q_s8(b[bi].qs + 16);
        const int8x16_t w01_1 = vld1q_s8(b[bi].qs + 32);
        const int8x16_t w23_1 = vld1q_s8(b[bi].qs + 48);
        const int8x16_t w01_2 = vld1q_s8(b[bi].qs + 64);
        const int8x16_t w23_2 = vld1q_s8(b[bi].qs + 80);
        const int8x16_t w01_3 = vld1q_s8(b[bi].qs + 96);
        const int8x16_t w23_3 = vld1q_s8(b[bi].qs + 112);

        const int8x16_t a01_0 = vld1q_s8(a[bi].qs + 0);
        const int8x16_t a23_0 = vld1q_s8(a[bi].qs + 16);
        const int8x16_t a01_1 = vld1q_s8(a[bi].qs + 32);
        const int8x16_t a23_1 = vld1q_s8(a[bi].qs + 48);
        const int8x16_t a01_2 = vld1q_s8(a[bi].qs + 64);
        const int8x16_t a23_2 = vld1q_s8(a[bi].qs + 80);
        const int8x16_t a01_3 = vld1q_s8(a[bi].qs + 96);
        const int8x16_t a23_3 = vld1q_s8(a[bi].qs + 112);

        int32x4_t c00 = vdupq_n_s32(0), c01 = vdupq_n_s32(0);
        int32x4_t c10 = vdupq_n_s32(0), c11 = vdupq_n_s32(0);
        c00 = vmmlaq_s32(c00, a01_0, w01_0);
        c01 = vmmlaq_s32(c01, a01_0, w23_0);
        c10 = vmmlaq_s32(c10, a23_0, w01_0);
        c11 = vmmlaq_s32(c11, a23_0, w23_0);
        c00 = vmmlaq_s32(c00, a01_1, w01_1);
        c01 = vmmlaq_s32(c01, a01_1, w23_1);
        c10 = vmmlaq_s32(c10, a23_1, w01_1);
        c11 = vmmlaq_s32(c11, a23_1, w23_1);
        c00 = vmmlaq_s32(c00, a01_2, w01_2);
        c01 = vmmlaq_s32(c01, a01_2, w23_2);
        c10 = vmmlaq_s32(c10, a23_2, w01_2);
        c11 = vmmlaq_s32(c11, a23_2, w23_2);
        c00 = vmmlaq_s32(c00, a01_3, w01_3);
        c01 = vmmlaq_s32(c01, a01_3, w23_3);
        c10 = vmmlaq_s32(c10, a23_3, w01_3);
        c11 = vmmlaq_s32(c11, a23_3, w23_3);

        const float32x4_t db = vcvt_f32_f16(vld1_f16((const __fp16 *)b[bi].d));
        const float32x4_t da = vcvt_f32_f16(vld1_f16((const __fp16 *)a[bi].d));
        fold4(c00, c01, c10, c11, da, db, fr0, fr1, fr2, fr3);
      }

      store4(m + 0, j, fr0);
      store4(m + 1, j, fr1);
      store4(m + 2, j, fr2);
      store4(m + 3, j, fr3);
    }
  }

  // Column remainder (nc % 4), scalar.
  for (int j = nc4; j < nc; ++j) {
    const block_q8_0x4 *b = b_sbase + (size_t)(j / 4) * nb;
    const int wr = j % 4;
    for (int m = 0; m < nr; m += 4) {
      const block_q8_0x4 *a = a_sbase + (size_t)(m / 4) * nb;
      for (int rr = 0; rr < 4; ++rr)
        s[(size_t)(m + rr) * bs + j] = (NNTR_GGML_FP16)dot_one(a, rr, b, wr);
    }
  }
#else
  for (int m = 0; m < nr; ++m) {
    const block_q8_0x4 *a = a_sbase + (size_t)(m / 4) * nb;
    const int ar = m % 4;
    for (int j = 0; j < nc; ++j)
      s[(size_t)m * bs + j] =
        (NNTR_GGML_FP16)dot_one(a, ar, b_sbase + (size_t)(j / 4) * nb, j % 4);
  }
#endif
}
#endif // ENABLE_FP16

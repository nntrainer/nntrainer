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
 * @file   nntr_ggml_impl_internal.h
 * @date   20 August 2025
 * @see    https://github.com/nnstreamer/nntrainer
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
#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__) &&   \
  defined(__aarch64__)
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
#else
  float sumf[4];
  int sumi;

  const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
  for (int x = 0; x < nc / ncols_interleaved; x++) {
    const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);

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
          sumf[j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                     nntr_compute_fp16_to_fp32(a_ptr[l].d);
        }
      }
    }
    for (int j = 0; j < ncols_interleaved; j++)
      s[x * ncols_interleaved + j] = sumf[j];
  }
#endif
}

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

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__) &&   \
  defined(__aarch64__)
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
#else
  float sumf[4][4];
  int sumi;

  for (int y = 0; y < nr / 4; y++) {
    const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_0x4 *b_ptr = (const block_q4_0x4 *)vx + (x * nb);
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
              sumf[m][j] += sumi * nntr_compute_fp16_to_fp32(b_ptr[l].d[j]) *
                            nntr_compute_fp16_to_fp32(a_ptr[l].d[m]);
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
#endif
}

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

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__)
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
  {
    const void *b_ptr = vx;
    const void *a_ptr = vy;
    float *res_ptr = s;
    size_t res_stride = bs * sizeof(float);

    __asm__ __volatile__(
      "mov x20, #0x4\n"
      "mov x13, %x[nr]\n"
      "mov z28.s, #-0x4\n"
      "mov x12, #0x88\n"
      "ptrue p1.b\n"
      "whilelt p0.s, XZR, x20\n"
      "cmp x13, #0x10\n"
      "mul x12, %x[nb], x12\n"
      "blt 4f\n"
      "1:" // Row loop
      "add x11, %x[b_ptr], #0x10\n"
      "mov x10, %x[nc]\n"
      "add x9, %x[res_ptr], %x[res_stride], LSL #4\n"
      "2:" // Column loop
      "add x28, %x[a_ptr], #0x8\n"
      "mov z24.b, #0x0\n"
      "mov z15.b, #0x0\n"
      "mov x27, %x[nb]\n"
      "add x26, x28, x12\n"
      "mov z12.b, #0x0\n"
      "mov z0.b, #0x0\n"
      "add x25, x26, x12\n"
      "mov z13.b, #0x0\n"
      "mov z1.b, #0x0\n"
      "add x24, x25, x12\n"
      "mov z20.b, #0x0\n"
      "mov z25.b, #0x0\n"
      "mov z11.b, #0x0\n"
      "mov z16.b, #0x0\n"
      "mov z19.b, #0x0\n"
      "mov z26.b, #0x0\n"
      "mov z8.b, #0x0\n"
      "mov z29.b, #0x0\n"
      "mov z27.b, #0x0\n"
      "mov z10.b, #0x0\n"
      "3:" // Block loop
      "ld1b { z30.b }, p1/Z, [x11]\n"
      "ld1b { z21.b }, p1/Z, [x11, #1, MUL VL]\n"
      "mov z18.s, #0x0\n"
      "mov z7.s, #0x0\n"
      "ld1rqb { z3.b }, p1/Z, [x28]\n"
      "ld1rqb { z5.b }, p1/Z, [x28, #16]\n"
      "mov z9.s, #0x0\n"
      "mov z22.s, #0x0\n"
      "ld1b { z4.b }, p1/Z, [x11, #2, MUL VL]\n"
      "ld1b { z17.b }, p1/Z, [x11, #3, MUL VL]\n"
      "sub x20, x11, #0x10\n"
      "sub x23, x28, #0x8\n"
      "lsl z31.b, z30.b, #0x4\n"
      "lsl z6.b, z21.b, #0x4\n"
      "ld1h { z23.s }, p1/Z, [x20]\n"
      "sub x22, x26, #0x8\n"
      "and z30.b, z30.b, #0xf0\n"
      "and z21.b, z21.b, #0xf0\n"
      "sub x21, x25, #0x8\n"
      "sub x20, x24, #0x8\n"
      "lsl z14.b, z4.b, #0x4\n"
      "lsl z2.b, z17.b, #0x4\n"
      "subs x27, x27, #0x1\n"
      "add x11, x11, #0x90\n"
      ".inst 0x451f9872  // smmla z18.s, z3.b, z31.b\n"
      ".inst 0x45069867  // smmla z7.s, z3.b, z6.b\n"
      "ld1rqb { z3.b }, p1/Z, [x28, #32]\n"
      "and z4.b, z4.b, #0xf0\n"
      ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
      ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
      "ld1rqb { z5.b }, p1/Z, [x28, #48]\n"
      "and z17.b, z17.b, #0xf0\n"
      "fcvt z23.s, p1/m, z23.h\n"
      ".inst 0x450e9872  // smmla z18.s, z3.b, z14.b\n"
      ".inst 0x45029867  // smmla z7.s, z3.b, z2.b\n"
      "ld1rqb { z3.b }, p1/Z, [x28, #64]\n"
      ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
      ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
      "ld1rqb { z5.b }, p1/Z, [x28, #80]\n"
      "fscale z23.s, p1/m, z23.s, z28.s\n"
      ".inst 0x451e9872  // smmla z18.s, z3.b, z30.b\n"
      ".inst 0x45159867  // smmla z7.s, z3.b, z21.b\n"
      "ld1rqb { z3.b }, p1/Z, [x28, #96]\n"
      ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
      ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
      "ld1rqb { z5.b }, p1/Z, [x28, #112]\n"
      "add x28, x28, #0x88\n"
      ".inst 0x45049872  // smmla z18.s, z3.b, z4.b\n"
      ".inst 0x45119867  // smmla z7.s, z3.b, z17.b\n"
      "ld1h { z3.s }, p0/Z, [x23]\n"
      ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
      ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
      "fcvt z3.s, p1/m, z3.h\n"
      "uzp1 z5.d, z18.d, z7.d\n"
      "uzp2 z18.d, z18.d, z7.d\n"
      "mov z3.q, z3.q[0]\n"
      "uzp1 z7.d, z9.d, z22.d\n"
      "uzp2 z22.d, z9.d, z22.d\n"
      "fmul z9.s, z23.s, z3.s[0]\n"
      "scvtf z5.s, p1/m, z5.s\n"
      "scvtf z18.s, p1/m, z18.s\n"
      "scvtf z7.s, p1/m, z7.s\n"
      "scvtf z22.s, p1/m, z22.s\n"
      "fmla z24.s, p1/M, z5.s, z9.s\n"
      "ld1rqb { z5.b }, p1/Z, [x26]\n"
      "fmul z9.s, z23.s, z3.s[1]\n"
      "fmla z15.s, p1/M, z18.s, z9.s\n"
      "ld1rqb { z18.b }, p1/Z, [x26, #16]\n"
      "fmul z9.s, z23.s, z3.s[2]\n"
      "fmul z3.s, z23.s, z3.s[3]\n"
      "fmla z12.s, p1/M, z7.s, z9.s\n"
      "mov z9.s, #0x0\n"
      "ld1h { z7.s }, p0/Z, [x22]\n"
      ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
      "fmla z0.s, p1/M, z22.s, z3.s\n"
      "mov z22.s, #0x0\n"
      "ld1h { z3.s }, p0/Z, [x21]\n"
      ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
      "ld1rqb { z5.b }, p1/Z, [x26, #32]\n"
      "fcvt z7.s, p1/m, z7.h\n"
      "fcvt z3.s, p1/m, z3.h\n"
      ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
      ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
      "ld1rqb { z5.b }, p1/Z, [x26, #64]\n"
      "mov z7.q, z7.q[0]\n"
      "mov z3.q, z3.q[0]\n"
      ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
      ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
      "ld1rqb { z5.b }, p1/Z, [x26, #96]\n"
      ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
      ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
      "uzp1 z5.d, z9.d, z22.d\n"
      "scvtf z5.s, p1/m, z5.s\n"
      "uzp2 z22.d, z9.d, z22.d\n"
      "fmul z9.s, z23.s, z7.s[0]\n"
      "scvtf z22.s, p1/m, z22.s\n"
      "fmla z13.s, p1/M, z5.s, z9.s\n"
      "ld1rqb { z9.b }, p1/Z, [x25]\n"
      "fmul z5.s, z23.s, z7.s[1]\n"
      "fmla z1.s, p1/M, z22.s, z5.s\n"
      "mov z5.s, #0x0\n"
      "mov z22.s, #0x0\n"
      ".inst 0x451f9a45  // smmla z5.s, z18.b, z31.b\n"
      ".inst 0x45069a56  // smmla z22.s, z18.b, z6.b\n"
      "ld1rqb { z18.b }, p1/Z, [x26, #48]\n"
      ".inst 0x450e9a45  // smmla z5.s, z18.b, z14.b\n"
      ".inst 0x45029a56  // smmla z22.s, z18.b, z2.b\n"
      "ld1rqb { z18.b }, p1/Z, [x26, #80]\n"
      ".inst 0x451e9a45  // smmla z5.s, z18.b, z30.b\n"
      ".inst 0x45159a56  // smmla z22.s, z18.b, z21.b\n"
      "ld1rqb { z18.b }, p1/Z, [x26, #112]\n"
      "add x26, x26, #0x88\n"
      ".inst 0x45049a45  // smmla z5.s, z18.b, z4.b\n"
      ".inst 0x45119a56  // smmla z22.s, z18.b, z17.b\n"
      "uzp1 z18.d, z5.d, z22.d\n"
      "scvtf z18.s, p1/m, z18.s\n"
      "uzp2 z22.d, z5.d, z22.d\n"
      "fmul z5.s, z23.s, z7.s[2]\n"
      "fmul z7.s, z23.s, z7.s[3]\n"
      "scvtf z22.s, p1/m, z22.s\n"
      "fmla z20.s, p1/M, z18.s, z5.s\n"
      "ld1rqb { z18.b }, p1/Z, [x25, #16]\n"
      "ld1h { z5.s }, p0/Z, [x20]\n"
      "fcvt z5.s, p1/m, z5.h\n"
      "fmla z25.s, p1/M, z22.s, z7.s\n"
      "mov z22.s, #0x0\n"
      "mov z7.s, #0x0\n"
      ".inst 0x451f9936  // smmla z22.s, z9.b, z31.b\n"
      ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
      "ld1rqb { z9.b }, p1/Z, [x25, #32]\n"
      "mov z5.q, z5.q[0]\n"
      ".inst 0x450e9936  // smmla z22.s, z9.b, z14.b\n"
      ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
      "ld1rqb { z9.b }, p1/Z, [x25, #64]\n"
      ".inst 0x451e9936  // smmla z22.s, z9.b, z30.b\n"
      ".inst 0x45159927  // smmla z7.s, z9.b, z21.b\n"
      "ld1rqb { z9.b }, p1/Z, [x25, #96]\n"
      ".inst 0x45049936  // smmla z22.s, z9.b, z4.b\n"
      ".inst 0x45119927  // smmla z7.s, z9.b, z17.b\n"
      "uzp1 z9.d, z22.d, z7.d\n"
      "scvtf z9.s, p1/m, z9.s\n"
      "uzp2 z22.d, z22.d, z7.d\n"
      "fmul z7.s, z23.s, z3.s[0]\n"
      "scvtf z22.s, p1/m, z22.s\n"
      "fmla z11.s, p1/M, z9.s, z7.s\n"
      "ld1rqb { z9.b }, p1/Z, [x24]\n"
      "fmul z7.s, z23.s, z3.s[1]\n"
      "fmla z16.s, p1/M, z22.s, z7.s\n"
      "mov z22.s, #0x0\n"
      "mov z7.s, #0x0\n"
      ".inst 0x451f9a56  // smmla z22.s, z18.b, z31.b\n"
      ".inst 0x45069a47  // smmla z7.s, z18.b, z6.b\n"
      "ld1rqb { z18.b }, p1/Z, [x25, #48]\n"
      ".inst 0x450e9a56  // smmla z22.s, z18.b, z14.b\n"
      ".inst 0x45029a47  // smmla z7.s, z18.b, z2.b\n"
      "ld1rqb { z18.b }, p1/Z, [x25, #80]\n"
      ".inst 0x451e9a56  // smmla z22.s, z18.b, z30.b\n"
      ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
      "ld1rqb { z18.b }, p1/Z, [x25, #112]\n"
      "add x25, x25, #0x88\n"
      ".inst 0x45049a56  // smmla z22.s, z18.b, z4.b\n"
      ".inst 0x45119a47  // smmla z7.s, z18.b, z17.b\n"
      "uzp1 z18.d, z22.d, z7.d\n"
      "scvtf z18.s, p1/m, z18.s\n"
      "uzp2 z7.d, z22.d, z7.d\n"
      "fmul z22.s, z23.s, z3.s[2]\n"
      "fmul z3.s, z23.s, z3.s[3]\n"
      "scvtf z7.s, p1/m, z7.s\n"
      "fmla z19.s, p1/M, z18.s, z22.s\n"
      "ld1rqb { z18.b }, p1/Z, [x24, #16]\n"
      "fmul z22.s, z23.s, z5.s[0]\n"
      "fmla z26.s, p1/M, z7.s, z3.s\n"
      "mov z3.s, #0x0\n"
      "mov z7.s, #0x0\n"
      ".inst 0x451f9923  // smmla z3.s, z9.b, z31.b\n"
      ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
      "ld1rqb { z9.b }, p1/Z, [x24, #32]\n"
      ".inst 0x450e9923  // smmla z3.s, z9.b, z14.b\n"
      ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
      "mov z9.s, #0x0\n"
      ".inst 0x451f9a49  // smmla z9.s, z18.b, z31.b\n"
      "mov z31.s, #0x0\n"
      ".inst 0x45069a5f  // smmla z31.s, z18.b, z6.b\n"
      "ld1rqb { z6.b }, p1/Z, [x24, #48]\n"
      "ld1rqb { z18.b }, p1/Z, [x24, #64]\n"
      ".inst 0x450e98c9  // smmla z9.s, z6.b, z14.b\n"
      "fmul z14.s, z23.s, z5.s[1]\n"
      ".inst 0x450298df  // smmla z31.s, z6.b, z2.b\n"
      "ld1rqb { z6.b }, p1/Z, [x24, #80]\n"
      "fmul z2.s, z23.s, z5.s[2]\n"
      "fmul z23.s, z23.s, z5.s[3]\n"
      ".inst 0x451e9a43  // smmla z3.s, z18.b, z30.b\n"
      ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
      "ld1rqb { z5.b }, p1/Z, [x24, #96]\n"
      ".inst 0x451e98c9  // smmla z9.s, z6.b, z30.b\n"
      ".inst 0x451598df  // smmla z31.s, z6.b, z21.b\n"
      "ld1rqb { z18.b }, p1/Z, [x24, #112]\n"
      "add x24, x24, #0x88\n"
      ".inst 0x450498a3  // smmla z3.s, z5.b, z4.b\n"
      ".inst 0x451198a7  // smmla z7.s, z5.b, z17.b\n"
      ".inst 0x45049a49  // smmla z9.s, z18.b, z4.b\n"
      ".inst 0x45119a5f  // smmla z31.s, z18.b, z17.b\n"
      "uzp1 z18.d, z3.d, z7.d\n"
      "uzp2 z5.d, z3.d, z7.d\n"
      "scvtf z18.s, p1/m, z18.s\n"
      "uzp1 z6.d, z9.d, z31.d\n"
      "uzp2 z9.d, z9.d, z31.d\n"
      "scvtf z5.s, p1/m, z5.s\n"
      "fmla z8.s, p1/M, z18.s, z22.s\n"
      "scvtf z6.s, p1/m, z6.s\n"
      "scvtf z9.s, p1/m, z9.s\n"
      "fmla z29.s, p1/M, z5.s, z14.s\n"
      "fmla z27.s, p1/M, z6.s, z2.s\n"
      "fmla z10.s, p1/M, z9.s, z23.s\n"
      "bgt 3b\n"
      "mov x20, %x[res_ptr]\n"
      "subs x10, x10, #0x8\n"
      "add %x[res_ptr], %x[res_ptr], #0x20\n"
      "st1w { z24.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z15.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z12.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z0.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z13.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z1.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z20.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z25.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z11.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z16.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z19.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z26.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z8.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z29.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z27.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "st1w { z10.s }, p1, [x20]\n"
      "bne 2b\n"
      "mov x20, #0x4\n"
      "sub x13, x13, #0x10\n"
      "cmp x13, #0x10\n"
      "mov %x[res_ptr], x9\n"
      "madd %x[a_ptr], x20, x12, %x[a_ptr]\n"
      "bge 1b\n"
      "4:" // Row loop skip
      "cbz x13, 9f\n"
      "5:" // Row tail: Row loop
      "add x25, %x[b_ptr], #0x10\n"
      "mov x24, %x[nc]\n"
      "add x23, %x[res_ptr], %x[res_stride], LSL #2\n"
      "6:" // Row tail: Column loop
      "mov z24.b, #0x0\n"
      "mov z15.b, #0x0\n"
      "add x28, %x[a_ptr], #0x8\n"
      "mov x22, %x[nb]\n"
      "mov z12.b, #0x0\n"
      "mov z0.b, #0x0\n"
      "7:" // Row tail: Block loop
      "ld1b { z3.b }, p1/Z, [x25]\n"
      "ld1b { z6.b }, p1/Z, [x25, #1, MUL VL]\n"
      "mov z2.s, #0x0\n"
      "mov z25.s, #0x0\n"
      "ld1rqb { z26.b }, p1/Z, [x28]\n"
      "ld1rqb { z21.b }, p1/Z, [x28, #16]\n"
      "mov z27.s, #0x0\n"
      "mov z19.s, #0x0\n"
      "ld1b { z29.b }, p1/Z, [x25, #2, MUL VL]\n"
      "ld1b { z16.b }, p1/Z, [x25, #3, MUL VL]\n"
      "sub x21, x25, #0x10\n"
      "sub x20, x28, #0x8\n"
      "lsl z20.b, z3.b, #0x4\n"
      "lsl z4.b, z6.b, #0x4\n"
      "ld1rqb { z10.b }, p1/Z, [x28, #32]\n"
      "ld1rqb { z23.b }, p1/Z, [x28, #48]\n"
      "and z3.b, z3.b, #0xf0\n"
      "and z6.b, z6.b, #0xf0\n"
      "ld1rqb { z11.b }, p1/Z, [x28, #64]\n"
      "ld1rqb { z7.b }, p1/Z, [x28, #80]\n"
      "lsl z8.b, z29.b, #0x4\n"
      "lsl z14.b, z16.b, #0x4\n"
      "ld1rqb { z18.b }, p1/Z, [x28, #96]\n"
      "ld1rqb { z30.b }, p1/Z, [x28, #112]\n"
      ".inst 0x45149b42  // smmla z2.s, z26.b, z20.b\n"
      ".inst 0x45049b59  // smmla z25.s, z26.b, z4.b\n"
      "and z29.b, z29.b, #0xf0\n"
      "ld1h { z17.s }, p1/Z, [x21]\n"
      ".inst 0x45149abb  // smmla z27.s, z21.b, z20.b\n"
      ".inst 0x45049ab3  // smmla z19.s, z21.b, z4.b\n"
      "and z16.b, z16.b, #0xf0\n"
      "ld1h { z4.s }, p0/Z, [x20]\n"
      "subs x22, x22, #0x1\n"
      "add x28, x28, #0x88\n"
      "fcvt z17.s, p1/m, z17.h\n"
      "add x25, x25, #0x90\n"
      ".inst 0x45089942  // smmla z2.s, z10.b, z8.b\n"
      ".inst 0x450e9959  // smmla z25.s, z10.b, z14.b\n"
      "fcvt z4.s, p1/m, z4.h\n"
      ".inst 0x45089afb  // smmla z27.s, z23.b, z8.b\n"
      ".inst 0x450e9af3  // smmla z19.s, z23.b, z14.b\n"
      "fscale z17.s, p1/m, z17.s, z28.s\n"
      "mov z4.q, z4.q[0]\n"
      ".inst 0x45039962  // smmla z2.s, z11.b, z3.b\n"
      ".inst 0x45069979  // smmla z25.s, z11.b, z6.b\n"
      "fmul z23.s, z17.s, z4.s[0]\n"
      "fmul z9.s, z17.s, z4.s[1]\n"
      "fmul z21.s, z17.s, z4.s[2]\n"
      "fmul z4.s, z17.s, z4.s[3]\n"
      ".inst 0x450398fb  // smmla z27.s, z7.b, z3.b\n"
      ".inst 0x450698f3  // smmla z19.s, z7.b, z6.b\n"
      ".inst 0x451d9a42  // smmla z2.s, z18.b, z29.b\n"
      ".inst 0x45109a59  // smmla z25.s, z18.b, z16.b\n"
      ".inst 0x451d9bdb  // smmla z27.s, z30.b, z29.b\n"
      ".inst 0x45109bd3  // smmla z19.s, z30.b, z16.b\n"
      "uzp1 z31.d, z2.d, z25.d\n"
      "uzp2 z13.d, z2.d, z25.d\n"
      "scvtf z31.s, p1/m, z31.s\n"
      "uzp1 z17.d, z27.d, z19.d\n"
      "uzp2 z18.d, z27.d, z19.d\n"
      "scvtf z13.s, p1/m, z13.s\n"
      "fmla z24.s, p1/M, z31.s, z23.s\n"
      "scvtf z17.s, p1/m, z17.s\n"
      "scvtf z18.s, p1/m, z18.s\n"
      "fmla z15.s, p1/M, z13.s, z9.s\n"
      "fmla z12.s, p1/M, z17.s, z21.s\n"
      "fmla z0.s, p1/M, z18.s, z4.s\n"
      "bgt 7b\n"
      "mov x20, %x[res_ptr]\n"
      "cmp x13, #0x1\n"
      "st1w { z24.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "ble 8f\n"
      "cmp x13, #0x2\n"
      "st1w { z15.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "ble 8f\n"
      "cmp x13, #0x3\n"
      "st1w { z12.s }, p1, [x20]\n"
      "add x20, x20, %x[res_stride]\n"
      "ble 8f\n"
      "st1w { z0.s }, p1, [x20]\n"
      "8:" // Row tail: Accumulator store skip
      "subs x24, x24, #0x8\n"
      "add %x[res_ptr], %x[res_ptr], #0x20\n"
      "bne 6b\n"
      "subs x13, x13, #0x4\n"
      "add %x[a_ptr], %x[a_ptr], x12\n"
      "mov %x[res_ptr], x23\n"
      "bgt 5b\n"
      "9:" // Row tail: Row loop skip
      : [a_ptr] "+&r"(a_ptr), [res_ptr] "+&r"(res_ptr)
      : [b_ptr] "r"(b_ptr), [nr] "r"(nr), [nb] "r"(nb),
        [res_stride] "r"(res_stride), [nc] "r"(nc)
      : "cc", "memory", "p0", "p1", "x9", "x10", "x11", "x12", "x13", "x20",
        "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1",
        "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12",
        "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22",
        "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");
    return;
  }
#endif // #if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
#elif defined(__AVX2__) || defined(__AVX512F__)
  {
    const block_q4_0x8 *b_ptr_start = (const block_q4_0x8 *)vx;
    const block_q8_0x4 *a_ptr_start = (const block_q8_0x4 *)vy;
    int64_t b_nb = n / QK4_0;
    int64_t y = 0;
    // Mask to mask out nibbles from packed bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    const __m128i loadMask =
      _mm_blend_epi32(_mm_setzero_si128(), _mm_set1_epi32(0xFFFFFFFF), 3);
    // Lookup table to convert signed nibbles to signed bytes
    __m256i signextendlut = _mm256_castsi128_si256(
      _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
    // Permute mask used for easier vector processing at later stages
    __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    int64_t xstart = 0;
    int anr = nr - nr % 16; // Used to align nr with boundary of 16
#ifdef __AVX512F__
    int anc = nc - nc % 16; // Used to align nc with boundary of 16
    // Mask to mask out nibbles from packed bytes expanded to 512 bit length
    const __m512i m4bexpanded = _mm512_set1_epi8(0x0F);
    // Lookup table to convert signed nibbles to signed bytes expanded to 512
    // bit length
    __m512i signextendlutexpanded = _mm512_inserti32x8(
      _mm512_castsi256_si512(signextendlut), signextendlut, 1);

    // Take group of four block_q8_0x4 structures at each pass of the loop and
    // perform dot product operation
    for (; y < anr / 4; y += 4) {

      const block_q8_0x4 *a_ptrs[4];

      a_ptrs[0] = a_ptr_start + (y * nb);
      for (int i = 0; i < 3; ++i) {
        a_ptrs[i + 1] = a_ptrs[i] + nb;
      }

      // Take group of two block_q4_0x8 structures at each pass of the loop and
      // perform dot product operation
      for (int64_t x = 0; x < anc / 8; x += 2) {

        const block_q4_0x8 *b_ptr_0 = b_ptr_start + ((x)*b_nb);
        const block_q4_0x8 *b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

        // Master FP accumulators
        __m512 acc_rows[16];
        for (int i = 0; i < 16; i++) {
          acc_rows[i] = _mm512_setzero_ps();
        }

        for (int64_t b = 0; b < nb; b++) {
          // Load the sixteen block_q4_0 quantized values interleaved with each
          // other in chunks of eight - B0,B1 ....BE,BF
          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs));
          const __m256i rhs_raw_mat_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 32));
          const __m256i rhs_raw_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 64));
          const __m256i rhs_raw_mat_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 96));

          const __m256i rhs_raw_mat_89AB_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs));
          const __m256i rhs_raw_mat_CDEF_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 32));
          const __m256i rhs_raw_mat_89AB_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 64));
          const __m256i rhs_raw_mat_CDEF_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 96));

          // Save the values in the following vectors in the formats
          // B0B1B4B5B8B9BCBD, B2B3B6B7BABBBEBF for further processing and
          // storing of values
          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);

          const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(
            rhs_raw_mat_89AB_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder),
            rhs_raw_mat_CDEF_0, 240);
          const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(
            rhs_raw_mat_89AB_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder),
            rhs_raw_mat_CDEF_1, 240);

          const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
          const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
          const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
          const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

          // 4-bit -> 8-bit - Sign is maintained
          const __m512i rhs_mat_014589CD_0 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(rhs_raw_mat_014589CD_0,
                             m4bexpanded)); // B0(0-7) B1(0-7) B4(0-7) B5(0-7)
                                            // B8(0-7) B9(0-7) BC(0-7) BD(0-7)
          const __m512i rhs_mat_2367ABEF_0 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(rhs_raw_mat_2367ABEF_0,
                             m4bexpanded)); // B2(0-7) B3(0-7) B6(0-7) B7(0-7)
                                            // BA(0-7) BB(0-7) BE(0-7) BF(0-7)

          const __m512i rhs_mat_014589CD_1 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(
              rhs_raw_mat_014589CD_1,
              m4bexpanded)); // B0(8-15) B1(8-15) B4(8-15) B5(8-15) B8(8-15)
                             // B9(8-15) BC(8-15) BD(8-15)
          const __m512i rhs_mat_2367ABEF_1 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(
              rhs_raw_mat_2367ABEF_1,
              m4bexpanded)); // B2(8-15) B3(8-15) B6(8-15) B7(8-15) BA(8-15)
                             // BB(8-15) BE(8-15) BF(8-15)

          const __m512i rhs_mat_014589CD_2 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(
              _mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4),
              m4bexpanded)); // B0(16-23) B1(16-23) B4(16-23) B5(16-23)
                             // B8(16-23) B9(16-23) BC(16-23) BD(16-23)
          const __m512i rhs_mat_2367ABEF_2 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(
              _mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4),
              m4bexpanded)); // B2(16-23) B3(16-23) B6(16-23) B7(16-23)
                             // BA(16-23) BB(16-23) BE(16-23) BF(16-23)

          const __m512i rhs_mat_014589CD_3 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(
              _mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4),
              m4bexpanded)); // B0(24-31) B1(24-31) B4(24-31) B5(24-31)
                             // B8(24-31) B9(24-31) BC(24-31) BD(24-31)
          const __m512i rhs_mat_2367ABEF_3 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(
              _mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4),
              m4bexpanded)); // B2(24-31) B3(24-31) B6(24-31) B7(24-31)
                             // BA(24-31) BB(24-31) BE(24-31) BF(24-31)

          // Shuffle pattern one - right side input
          const __m512i rhs_mat_014589CD_0_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_0,
            (_MM_PERM_ENUM)136); // B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3)
                                 // B5(0-3) B4(0-3) B5(0-3) B8(0-3) B9(0-3)
                                 // B8(0-3) B9(0-3) BC(0-3) BD(0-3) BC(0-3)
                                 // BD(0-3)
          const __m512i rhs_mat_2367ABEF_0_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_0,
            (_MM_PERM_ENUM)136); // B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3)
                                 // B7(0-3) B6(0-3) B7(0-3) BA(0-3) BB(0-3)
                                 // BA(0-3) BB(0-3) BE(0-3) BF(0-3) BE(0-3)
                                 // BF(0-3)

          const __m512i rhs_mat_014589CD_1_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_1,
            (_MM_PERM_ENUM)136); // B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11)
                                 // B5(8-11) B4(8-11) B5(8-11) B8(8-11) B9(8-11)
                                 // B8(8-11) B9(8-11) BC(8-11) BD(8-11) BC(8-11)
                                 // BD(8-11)
          const __m512i rhs_mat_2367ABEF_1_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_1,
            (_MM_PERM_ENUM)136); // B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11)
                                 // B7(8-11) B6(8-11) B7(8-11) BA(8-11) BB(8-11)
                                 // BA(8-11) BB(8-11) BE(8-11) BF(8-11) BE(8-11)
                                 // BF(8-11)

          const __m512i rhs_mat_014589CD_2_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_2,
            (_MM_PERM_ENUM)136); // B0(16-19) B1(16-19) B0(16-19) B1(16-19)
                                 // B4(16-19) B5(16-19) B4(16-19) B5(16-19)
                                 // B8(16-19) B9(16-19) B8(16-19) B9(16-19)
                                 // BC(16-19) BD(16-19) BC(16-19) BD(16-19)
          const __m512i rhs_mat_2367ABEF_2_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_2,
            (_MM_PERM_ENUM)136); // B2(16-19) B3(16-19) B2(16-19) B3(16-19)
                                 // B6(16-19) B7(16-19) B6(16-19) B7(16-19)
                                 // BA(16-19) BB(16-19) BA(16-19) BB(16-19)
                                 // BE(16-19) BF(16-19) BE(16-19) BF(16-19)

          const __m512i rhs_mat_014589CD_3_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_3,
            (_MM_PERM_ENUM)136); // B0(24-27) B1(24-27) B0(24-27) B1(24-27)
                                 // B4(24-27) B5(24-27) B4(24-27) B5(24-27)
                                 // B8(24-27) B9(24-27) B8(24-27) B9(24-27)
                                 // BC(24-27) BD(24-27) BC(24-27) BD(24-27)
          const __m512i rhs_mat_2367ABEF_3_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_3,
            (_MM_PERM_ENUM)136); // B2(24-27) B3(24-27) B2(24-27) B3(24-27)
                                 // B6(24-27) B7(24-27) B6(24-27) B7(24-27)
                                 // BA(24-27) BB(24-27) BA(24-27) BB(24-27)
                                 // BE(24-27) BF(24-27) BE(24-27) BF(24-27)

          // Shuffle pattern two - right side input

          const __m512i rhs_mat_014589CD_0_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_0,
            (_MM_PERM_ENUM)221); // B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7)
                                 // B5(4-7) B4(4-7) B5(4-7) B8(4-7) B9(4-7)
                                 // B8(4-7) B9(4-7) BC(4-7) BD(4-7) BC(4-7)
                                 // BD(4-7)
          const __m512i rhs_mat_2367ABEF_0_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_0,
            (_MM_PERM_ENUM)221); // B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7)
                                 // B7(4-7) B6(4-7) B7(4-7) BA(4-7) BB(4-7)
                                 // BA(4-7) BB(4-7) BE(4-7) BF(4-7) BE(4-7)
                                 // BF(4-7)

          const __m512i rhs_mat_014589CD_1_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_1,
            (_MM_PERM_ENUM)221); // B0(12-15) B1(12-15) B0(12-15) B1(12-15)
                                 // B4(12-15) B5(12-15) B4(12-15) B5(12-15)
                                 // B8(12-15) B9(12-15) B8(12-15) B9(12-15)
                                 // BC(12-15) BD(12-15) BC(12-15) BD(12-15)
          const __m512i rhs_mat_2367ABEF_1_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_1,
            (_MM_PERM_ENUM)221); // B2(12-15) B3(12-15) B2(12-15) B3(12-15)
                                 // B6(12-15) B7(12-15) B6(12-15) B7(12-15)
                                 // BA(12-15) BB(12-15) BA(12-15) BB(12-15)
                                 // BE(12-15) BF(12-15) BE(12-15) BF(12-15)

          const __m512i rhs_mat_014589CD_2_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_2,
            (_MM_PERM_ENUM)221); // B0(20-23) B1(20-23) B0(20-23) B1(20-23)
                                 // B4(20-23) B5(20-23) B4(20-23) B5(20-23)
                                 // B8(20-23) B9(20-23) B8(20-23) B9(20-23)
                                 // BC(20-23) BD(20-23) BC(20-23) BD(20-23)
          const __m512i rhs_mat_2367ABEF_2_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_2,
            (_MM_PERM_ENUM)221); // B2(20-23) B3(20-23) B2(20-23) B3(20-23)
                                 // B6(20-23) B7(20-23) B6(20-23) B7(20-23)
                                 // BA(20-23) BB(20-23) BA(20-23) BB(20-23)
                                 // BE(20-23) BF(20-23) BE(20-23) BF(20-23)

          const __m512i rhs_mat_014589CD_3_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_3,
            (_MM_PERM_ENUM)221); // B0(28-31) B1(28-31) B0(28-31) B1(28-31)
                                 // B4(28-31) B5(28-31) B4(28-31) B5(28-31)
                                 // B8(28-31) B9(28-31) B8(28-31) B9(28-31)
                                 // BC(28-31) BD(28-31) BC(28-31) BD(28-31)
          const __m512i rhs_mat_2367ABEF_3_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_3,
            (_MM_PERM_ENUM)221); // B2(28-31) B3(28-31) B2(28-31) B3(28-31)
                                 // B6(28-31) B7(28-31) B6(28-31) B7(28-31)
                                 // BA(28-31) BB(28-31) BA(28-31) BB(28-31)
                                 // BE(28-31) BF(28-31) BE(28-31) BF(28-31)

          // Scale values - Load the weight scale values of two block_q4_0x8
          const __m512 col_scale_f32 =
            GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

          // Process LHS in pairs of rows
          for (int rp = 0; rp < 4; rp++) {

            // Load the four block_q4_0 quantized values interleaved with each
            // other in chunks of eight - A0,A1,A2,A3 Loaded as set of 128 bit
            // vectors and repeated and stored into a 256 bit vector before
            // again repeating into 512 bit vector
            __m256i lhs_mat_ymm_0123_0 =
              _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs)));
            __m256i lhs_mat_ymm_01_0 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 0);
            __m256i lhs_mat_ymm_23_0 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 17);
            __m256i lhs_mat_ymm_0123_1 =
              _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 32)));
            __m256i lhs_mat_ymm_01_1 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 0);
            __m256i lhs_mat_ymm_23_1 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 17);
            __m256i lhs_mat_ymm_0123_2 =
              _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 64)));
            __m256i lhs_mat_ymm_01_2 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 0);
            __m256i lhs_mat_ymm_23_2 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 17);
            __m256i lhs_mat_ymm_0123_3 =
              _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 96)));
            __m256i lhs_mat_ymm_01_3 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 0);
            __m256i lhs_mat_ymm_23_3 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 17);

            __m512i lhs_mat_01_0 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_01_0), lhs_mat_ymm_01_0, 1);
            __m512i lhs_mat_23_0 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_23_0), lhs_mat_ymm_23_0, 1);
            __m512i lhs_mat_01_1 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_01_1), lhs_mat_ymm_01_1, 1);
            __m512i lhs_mat_23_1 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_23_1), lhs_mat_ymm_23_1, 1);
            __m512i lhs_mat_01_2 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_01_2), lhs_mat_ymm_01_2, 1);
            __m512i lhs_mat_23_2 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_23_2), lhs_mat_ymm_23_2, 1);
            __m512i lhs_mat_01_3 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_01_3), lhs_mat_ymm_01_3, 1);
            __m512i lhs_mat_23_3 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_23_3), lhs_mat_ymm_23_3, 1);

            // Shuffle pattern one - left side input

            const __m512i lhs_mat_01_0_sp1 = _mm512_shuffle_epi32(
              lhs_mat_01_0,
              (_MM_PERM_ENUM)160); // A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3)
                                   // A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3)
                                   // A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3)
                                   // A1(0-3)
            const __m512i lhs_mat_23_0_sp1 = _mm512_shuffle_epi32(
              lhs_mat_23_0,
              (_MM_PERM_ENUM)160); // A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3)
                                   // A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3)
                                   // A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3)
                                   // A3(0-3)

            const __m512i lhs_mat_01_1_sp1 = _mm512_shuffle_epi32(
              lhs_mat_01_1,
              (_MM_PERM_ENUM)160); // A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                                   // A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                                   // A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                                   // A0(8-11) A0(8-11) A1(8-11) A1(8-11)
            const __m512i lhs_mat_23_1_sp1 = _mm512_shuffle_epi32(
              lhs_mat_23_1,
              (_MM_PERM_ENUM)160); // A2(8-11) A2(8-11) A3(8-11) A3(8-11)
                                   // A2(8-11) A2(8-11) A3(8-11) A3(8-11)
                                   // A2(8-11) A2(8-11) A3(8-11) A3(8-11)
                                   // A2(8-11) A2(8-11) A3(8-11) A3(8-11)

            const __m512i lhs_mat_01_2_sp1 = _mm512_shuffle_epi32(
              lhs_mat_01_2,
              (_MM_PERM_ENUM)160); // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                                   // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                                   // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                                   // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
            const __m512i lhs_mat_23_2_sp1 = _mm512_shuffle_epi32(
              lhs_mat_23_2,
              (_MM_PERM_ENUM)160); // A2(16-19) A2(16-19) A3(16-19) A3(16-19)
                                   // A2(16-19) A2(16-19) A3(16-19) A3(16-19)
                                   // A2(16-19) A2(16-19) A3(16-19) A3(16-19)
                                   // A2(16-19) A2(16-19) A3(16-19) A3(16-19)

            const __m512i lhs_mat_01_3_sp1 = _mm512_shuffle_epi32(
              lhs_mat_01_3,
              (_MM_PERM_ENUM)160); // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                                   // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                                   // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                                   // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
            const __m512i lhs_mat_23_3_sp1 = _mm512_shuffle_epi32(
              lhs_mat_23_3,
              (_MM_PERM_ENUM)160); // A2(24-27) A2(24-27) A3(24-27) A3(24-27)
                                   // A2(24-27) A2(24-27) A3(24-27) A3(24-27)
                                   // A2(24-27) A2(24-27) A3(24-27) A3(24-27)
                                   // A2(24-27) A2(24-27) A3(24-27) A3(24-27)

            // Shuffle pattern two - left side input

            const __m512i lhs_mat_01_0_sp2 = _mm512_shuffle_epi32(
              lhs_mat_01_0,
              (_MM_PERM_ENUM)245); // A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7)
                                   // A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7)
                                   // A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7)
                                   // A1(4-7)
            const __m512i lhs_mat_23_0_sp2 = _mm512_shuffle_epi32(
              lhs_mat_23_0,
              (_MM_PERM_ENUM)245); // A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7)
                                   // A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7)
                                   // A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7)
                                   // A3(4-7)

            const __m512i lhs_mat_01_1_sp2 = _mm512_shuffle_epi32(
              lhs_mat_01_1,
              (_MM_PERM_ENUM)245); // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                                   // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                                   // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                                   // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
            const __m512i lhs_mat_23_1_sp2 = _mm512_shuffle_epi32(
              lhs_mat_23_1,
              (_MM_PERM_ENUM)245); // A2(12-15) A2(12-15) A3(12-15) A3(12-15)
                                   // A2(12-15) A2(12-15) A3(12-15) A3(12-15)
                                   // A2(12-15) A2(12-15) A3(12-15) A3(12-15)
                                   // A2(12-15) A2(12-15) A3(12-15) A3(12-15)

            const __m512i lhs_mat_01_2_sp2 = _mm512_shuffle_epi32(
              lhs_mat_01_2,
              (_MM_PERM_ENUM)245); // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                                   // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                                   // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                                   // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
            const __m512i lhs_mat_23_2_sp2 = _mm512_shuffle_epi32(
              lhs_mat_23_2,
              (_MM_PERM_ENUM)245); // A2(20-23) A2(20-23) A3(20-23) A3(20-23)
                                   // A2(20-23) A2(20-23) A3(20-23) A3(20-23)
                                   // A2(20-23) A2(20-23) A3(20-23) A3(20-23)
                                   // A2(20-23) A2(20-23) A3(20-23) A3(20-23)

            const __m512i lhs_mat_01_3_sp2 = _mm512_shuffle_epi32(
              lhs_mat_01_3,
              (_MM_PERM_ENUM)245); // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                                   // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                                   // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                                   // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
            const __m512i lhs_mat_23_3_sp2 = _mm512_shuffle_epi32(
              lhs_mat_23_3,
              (_MM_PERM_ENUM)245); // A2(28-31) A2(28-31) A3(28-31) A3(28-31)
                                   // A2(28-31) A2(28-31) A3(28-31) A3(28-31)
                                   // A2(28-31) A2(28-31) A3(28-31) A3(28-31)
                                   // A2(28-31) A2(28-31) A3(28-31) A3(28-31)

            // The values arranged in shuffle patterns are operated with dot
            // product operation within 32 bit lane i.e corresponding bytes and
            // multiplied and added into 32 bit integers within 32 bit lane
            // Resembles MMLAs into 2x2 matrices in ARM Version
            const __m512i zero = _mm512_setzero_epi32();
            __m512i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(
                  mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1,
                                                rhs_mat_014589CD_3_sp1),
                  lhs_mat_01_2_sp1, rhs_mat_014589CD_2_sp1),
                lhs_mat_01_1_sp1, rhs_mat_014589CD_1_sp1),
              lhs_mat_01_0_sp1, rhs_mat_014589CD_0_sp1);
            __m512i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(
                  mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1,
                                                rhs_mat_2367ABEF_3_sp1),
                  lhs_mat_01_2_sp1, rhs_mat_2367ABEF_2_sp1),
                lhs_mat_01_1_sp1, rhs_mat_2367ABEF_1_sp1),
              lhs_mat_01_0_sp1, rhs_mat_2367ABEF_0_sp1);
            __m512i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(
                  mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1,
                                                rhs_mat_014589CD_3_sp1),
                  lhs_mat_23_2_sp1, rhs_mat_014589CD_2_sp1),
                lhs_mat_23_1_sp1, rhs_mat_014589CD_1_sp1),
              lhs_mat_23_0_sp1, rhs_mat_014589CD_0_sp1);
            __m512i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(
                  mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1,
                                                rhs_mat_2367ABEF_3_sp1),
                  lhs_mat_23_2_sp1, rhs_mat_2367ABEF_2_sp1),
                lhs_mat_23_1_sp1, rhs_mat_2367ABEF_1_sp1),
              lhs_mat_23_0_sp1, rhs_mat_2367ABEF_0_sp1);
            __m512i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(
                  mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2,
                                                rhs_mat_014589CD_3_sp2),
                  lhs_mat_01_2_sp2, rhs_mat_014589CD_2_sp2),
                lhs_mat_01_1_sp2, rhs_mat_014589CD_1_sp2),
              lhs_mat_01_0_sp2, rhs_mat_014589CD_0_sp2);
            __m512i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(
                  mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2,
                                                rhs_mat_2367ABEF_3_sp2),
                  lhs_mat_01_2_sp2, rhs_mat_2367ABEF_2_sp2),
                lhs_mat_01_1_sp2, rhs_mat_2367ABEF_1_sp2),
              lhs_mat_01_0_sp2, rhs_mat_2367ABEF_0_sp2);
            __m512i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(
                  mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2,
                                                rhs_mat_014589CD_3_sp2),
                  lhs_mat_23_2_sp2, rhs_mat_014589CD_2_sp2),
                lhs_mat_23_1_sp2, rhs_mat_014589CD_1_sp2),
              lhs_mat_23_0_sp2, rhs_mat_014589CD_0_sp2);
            __m512i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(
                  mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2,
                                                rhs_mat_2367ABEF_3_sp2),
                  lhs_mat_23_2_sp2, rhs_mat_2367ABEF_2_sp2),
                lhs_mat_23_1_sp2, rhs_mat_2367ABEF_1_sp2),
              lhs_mat_23_0_sp2, rhs_mat_2367ABEF_0_sp2);

            // Output of both shuffle patterns are added in order to sum dot
            // product outputs of all 32 values in block
            __m512i iacc_mat_00 =
              _mm512_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
            __m512i iacc_mat_01 =
              _mm512_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
            __m512i iacc_mat_10 =
              _mm512_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
            __m512i iacc_mat_11 =
              _mm512_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);

            // Straighten out to make 4 row vectors
            __m512i iacc_row_0 = _mm512_mask_blend_epi32(
              0xCCCC, iacc_mat_00,
              _mm512_shuffle_epi32(iacc_mat_01, (_MM_PERM_ENUM)78));
            __m512i iacc_row_1 = _mm512_mask_blend_epi32(
              0xCCCC, _mm512_shuffle_epi32(iacc_mat_00, (_MM_PERM_ENUM)78),
              iacc_mat_01);
            __m512i iacc_row_2 = _mm512_mask_blend_epi32(
              0xCCCC, iacc_mat_10,
              _mm512_shuffle_epi32(iacc_mat_11, (_MM_PERM_ENUM)78));
            __m512i iacc_row_3 = _mm512_mask_blend_epi32(
              0xCCCC, _mm512_shuffle_epi32(iacc_mat_10, (_MM_PERM_ENUM)78),
              iacc_mat_11);

            // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it
            // across lanes
            const __m128i row_scale_f16 = _mm_shuffle_epi32(
              _mm_maskload_epi32((int const *)(a_ptrs[rp][b].d), loadMask), 68);
            const __m512 row_scale_f32 =
              GGML_F32Cx16_REPEAT_LOAD(row_scale_f16);

            // Multiply with appropiate scales and accumulate
            acc_rows[rp * 4] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(iacc_row_0),
              _mm512_mul_ps(col_scale_f32,
                            _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
              acc_rows[rp * 4]);
            acc_rows[rp * 4 + 1] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(iacc_row_1),
              _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(
                                             row_scale_f32, row_scale_f32, 85)),
              acc_rows[rp * 4 + 1]);
            acc_rows[rp * 4 + 2] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(iacc_row_2),
              _mm512_mul_ps(
                col_scale_f32,
                _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
              acc_rows[rp * 4 + 2]);
            acc_rows[rp * 4 + 3] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(iacc_row_3),
              _mm512_mul_ps(
                col_scale_f32,
                _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
              acc_rows[rp * 4 + 3]);
          }
        }

        // Store the accumulated values
        for (int i = 0; i < 16; i++) {
          _mm512_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                           acc_rows[i]);
        }
      }
    }
    // Take a block_q8_0x4 structures at each pass of the loop and perform dot
    // product operation
    for (; y < nr / 4; y++) {

      const block_q8_0x4 *a_ptr = a_ptr_start + (y * nb);

      // Take group of two block_q4_0x8 structures at each pass of the loop and
      // perform dot product operation
      for (int64_t x = 0; x < anc / 8; x += 2) {

        const block_q4_0x8 *b_ptr_0 = b_ptr_start + ((x)*b_nb);
        const block_q4_0x8 *b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

        // Master FP accumulators
        __m512 acc_rows[4];
        for (int i = 0; i < 4; i++) {
          acc_rows[i] = _mm512_setzero_ps();
        }

        for (int64_t b = 0; b < nb; b++) {
          // Load the sixteen block_q4_0 quantized values interleaved with each
          // other in chunks of eight - B0,B1 ....BE,BF
          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs));
          const __m256i rhs_raw_mat_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 32));
          const __m256i rhs_raw_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 64));
          const __m256i rhs_raw_mat_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 96));

          const __m256i rhs_raw_mat_89AB_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs));
          const __m256i rhs_raw_mat_CDEF_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 32));
          const __m256i rhs_raw_mat_89AB_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 64));
          const __m256i rhs_raw_mat_CDEF_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 96));

          // Save the values in the following vectors in the formats B0B1B4B5,
          // B2B3B6B7 for further processing and storing of valuess
          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);

          const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(
            rhs_raw_mat_89AB_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder),
            rhs_raw_mat_CDEF_0, 240);
          const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(
            rhs_raw_mat_89AB_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder),
            rhs_raw_mat_CDEF_1, 240);

          const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
          const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
          const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
          const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

          // 4-bit -> 8-bit - Sign is maintained
          const __m512i rhs_mat_014589CD_0 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(rhs_raw_mat_014589CD_0,
                             m4bexpanded)); // B0(0-7) B1(0-7) B4(0-7) B5(0-7)
                                            // B8(0-7) B9(0-7) BC(0-7) BD(0-7)
          const __m512i rhs_mat_2367ABEF_0 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(rhs_raw_mat_2367ABEF_0,
                             m4bexpanded)); // B2(0-7) B3(0-7) B6(0-7) B7(0-7)
                                            // BA(0-7) BB(0-7) BE(0-7) BF(0-7)

          const __m512i rhs_mat_014589CD_1 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(
              rhs_raw_mat_014589CD_1,
              m4bexpanded)); // B0(8-15) B1(8-15) B4(8-15) B5(8-15) B8(8-15)
                             // B9(8-15) BC(8-15) BD(8-15)
          const __m512i rhs_mat_2367ABEF_1 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(
              rhs_raw_mat_2367ABEF_1,
              m4bexpanded)); // B2(8-15) B3(8-15) B6(8-15) B7(8-15) BA(8-15)
                             // BB(8-15) BE(8-15) BF(8-15)

          const __m512i rhs_mat_014589CD_2 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(
              _mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4),
              m4bexpanded)); // B0(16-23) B1(16-23) B4(16-23) B5(16-23)
                             // B8(16-23) B9(16-23) BC(16-23) BD(16-23)
          const __m512i rhs_mat_2367ABEF_2 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(
              _mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4),
              m4bexpanded)); // B2(16-23) B3(16-23) B6(16-23) B7(16-23)
                             // BA(16-23) BB(16-23) BE(16-23) BF(16-23)

          const __m512i rhs_mat_014589CD_3 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(
              _mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4),
              m4bexpanded)); // B0(24-31) B1(24-31) B4(24-31) B5(24-31)
                             // B8(24-31) B9(24-31) BC(24-31) BD(24-31)
          const __m512i rhs_mat_2367ABEF_3 = _mm512_shuffle_epi8(
            signextendlutexpanded,
            _mm512_and_si512(
              _mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4),
              m4bexpanded)); // B2(24-31) B3(24-31) B6(24-31) B7(24-31)
                             // BA(24-31) BB(24-31) BE(24-31) BF(24-31)

          // Shuffle pattern one - right side input
          const __m512i rhs_mat_014589CD_0_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_0,
            (_MM_PERM_ENUM)136); // B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3)
                                 // B5(0-3) B4(0-3) B5(0-3) B8(0-3) B9(0-3)
                                 // B8(0-3) B9(0-3) BC(0-3) BD(0-3) BC(0-3)
                                 // BD(0-3)
          const __m512i rhs_mat_2367ABEF_0_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_0,
            (_MM_PERM_ENUM)136); // B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3)
                                 // B7(0-3) B6(0-3) B7(0-3) BA(0-3) BB(0-3)
                                 // BA(0-3) BB(0-3) BE(0-3) BF(0-3) BE(0-3)
                                 // BF(0-3)

          const __m512i rhs_mat_014589CD_1_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_1,
            (_MM_PERM_ENUM)136); // B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11)
                                 // B5(8-11) B4(8-11) B5(8-11) B8(8-11) B9(8-11)
                                 // B8(8-11) B9(8-11) BC(8-11) BD(8-11) BC(8-11)
                                 // BD(8-11)
          const __m512i rhs_mat_2367ABEF_1_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_1,
            (_MM_PERM_ENUM)136); // B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11)
                                 // B7(8-11) B6(8-11) B7(8-11) BA(8-11) BB(8-11)
                                 // BA(8-11) BB(8-11) BE(8-11) BF(8-11) BE(8-11)
                                 // BF(8-11)

          const __m512i rhs_mat_014589CD_2_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_2,
            (_MM_PERM_ENUM)136); // B0(16-19) B1(16-19) B0(16-19) B1(16-19)
                                 // B4(16-19) B5(16-19) B4(16-19) B5(16-19)
                                 // B8(16-19) B9(16-19) B8(16-19) B9(16-19)
                                 // BC(16-19) BD(16-19) BC(16-19) BD(16-19)
          const __m512i rhs_mat_2367ABEF_2_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_2,
            (_MM_PERM_ENUM)136); // B2(16-19) B3(16-19) B2(16-19) B3(16-19)
                                 // B6(16-19) B7(16-19) B6(16-19) B7(16-19)
                                 // BA(16-19) BB(16-19) BA(16-19) BB(16-19)
                                 // BE(16-19) BF(16-19) BE(16-19) BF(16-19)

          const __m512i rhs_mat_014589CD_3_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_3,
            (_MM_PERM_ENUM)136); // B0(24-27) B1(24-27) B0(24-27) B1(24-27)
                                 // B4(24-27) B5(24-27) B4(24-27) B5(24-27)
                                 // B8(24-27) B9(24-27) B8(24-27) B9(24-27)
                                 // BC(24-27) BD(24-27) BC(24-27) BD(24-27)
          const __m512i rhs_mat_2367ABEF_3_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_3,
            (_MM_PERM_ENUM)136); // B2(24-27) B3(24-27) B2(24-27) B3(24-27)
                                 // B6(24-27) B7(24-27) B6(24-27) B7(24-27)
                                 // BA(24-27) BB(24-27) BA(24-27) BB(24-27)
                                 // BE(24-27) BF(24-27) BE(24-27) BF(24-27)

          // Shuffle pattern two - right side input

          const __m512i rhs_mat_014589CD_0_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_0,
            (_MM_PERM_ENUM)221); // B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7)
                                 // B5(4-7) B4(4-7) B5(4-7) B8(4-7) B9(4-7)
                                 // B8(4-7) B9(4-7) BC(4-7) BD(4-7) BC(4-7)
                                 // BD(4-7)
          const __m512i rhs_mat_2367ABEF_0_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_0,
            (_MM_PERM_ENUM)221); // B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7)
                                 // B7(4-7) B6(4-7) B7(4-7) BA(4-7) BB(4-7)
                                 // BA(4-7) BB(4-7) BE(4-7) BF(4-7) BE(4-7)
                                 // BF(4-7)

          const __m512i rhs_mat_014589CD_1_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_1,
            (_MM_PERM_ENUM)221); // B0(12-15) B1(12-15) B0(12-15) B1(12-15)
                                 // B4(12-15) B5(12-15) B4(12-15) B5(12-15)
                                 // B8(12-15) B9(12-15) B8(12-15) B9(12-15)
                                 // BC(12-15) BD(12-15) BC(12-15) BD(12-15)
          const __m512i rhs_mat_2367ABEF_1_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_1,
            (_MM_PERM_ENUM)221); // B2(12-15) B3(12-15) B2(12-15) B3(12-15)
                                 // B6(12-15) B7(12-15) B6(12-15) B7(12-15)
                                 // BA(12-15) BB(12-15) BA(12-15) BB(12-15)
                                 // BE(12-15) BF(12-15) BE(12-15) BF(12-15)

          const __m512i rhs_mat_014589CD_2_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_2,
            (_MM_PERM_ENUM)221); // B0(20-23) B1(20-23) B0(20-23) B1(20-23)
                                 // B4(20-23) B5(20-23) B4(20-23) B5(20-23)
                                 // B8(20-23) B9(20-23) B8(20-23) B9(20-23)
                                 // BC(20-23) BD(20-23) BC(20-23) BD(20-23)
          const __m512i rhs_mat_2367ABEF_2_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_2,
            (_MM_PERM_ENUM)221); // B2(20-23) B3(20-23) B2(20-23) B3(20-23)
                                 // B6(20-23) B7(20-23) B6(20-23) B7(20-23)
                                 // BA(20-23) BB(20-23) BA(20-23) BB(20-23)
                                 // BE(20-23) BF(20-23) BE(20-23) BF(20-23)

          const __m512i rhs_mat_014589CD_3_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_3,
            (_MM_PERM_ENUM)221); // B0(28-31) B1(28-31) B0(28-31) B1(28-31)
                                 // B4(28-31) B5(28-31) B4(28-31) B5(28-31)
                                 // B8(28-31) B9(28-31) B8(28-31) B9(28-31)
                                 // BC(28-31) BD(28-31) BC(28-31) BD(28-31)
          const __m512i rhs_mat_2367ABEF_3_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_3,
            (_MM_PERM_ENUM)221); // B2(28-31) B3(28-31) B2(28-31) B3(28-31)
                                 // B6(28-31) B7(28-31) B6(28-31) B7(28-31)
                                 // BA(28-31) BB(28-31) BA(28-31) BB(28-31)
                                 // BE(28-31) BF(28-31) BE(28-31) BF(28-31)

          // Scale values - Load the weight scale values of two block_q4_0x8
          const __m512 col_scale_f32 =
            GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

          // Load the four block_q4_0 quantized values interleaved with each
          // other in chunks of eight - A0,A1,A2,A3 Loaded as set of 128 bit
          // vectors and repeated and stored into a 256 bit vector before again
          // repeating into 512 bit vector
          __m256i lhs_mat_ymm_0123_0 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs)));
          __m256i lhs_mat_ymm_01_0 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 0);
          __m256i lhs_mat_ymm_23_0 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 17);
          __m256i lhs_mat_ymm_0123_1 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 32)));
          __m256i lhs_mat_ymm_01_1 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 0);
          __m256i lhs_mat_ymm_23_1 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 17);
          __m256i lhs_mat_ymm_0123_2 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 64)));
          __m256i lhs_mat_ymm_01_2 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 0);
          __m256i lhs_mat_ymm_23_2 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 17);
          __m256i lhs_mat_ymm_0123_3 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 96)));
          __m256i lhs_mat_ymm_01_3 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 0);
          __m256i lhs_mat_ymm_23_3 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 17);

          __m512i lhs_mat_01_0 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_01_0), lhs_mat_ymm_01_0, 1);
          __m512i lhs_mat_23_0 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_23_0), lhs_mat_ymm_23_0, 1);
          __m512i lhs_mat_01_1 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_01_1), lhs_mat_ymm_01_1, 1);
          __m512i lhs_mat_23_1 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_23_1), lhs_mat_ymm_23_1, 1);
          __m512i lhs_mat_01_2 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_01_2), lhs_mat_ymm_01_2, 1);
          __m512i lhs_mat_23_2 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_23_2), lhs_mat_ymm_23_2, 1);
          __m512i lhs_mat_01_3 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_01_3), lhs_mat_ymm_01_3, 1);
          __m512i lhs_mat_23_3 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_23_3), lhs_mat_ymm_23_3, 1);

          // Shuffle pattern one - left side input

          const __m512i lhs_mat_01_0_sp1 = _mm512_shuffle_epi32(
            lhs_mat_01_0,
            (_MM_PERM_ENUM)160); // A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3)
                                 // A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3)
                                 // A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3)
                                 // A1(0-3)
          const __m512i lhs_mat_23_0_sp1 = _mm512_shuffle_epi32(
            lhs_mat_23_0,
            (_MM_PERM_ENUM)160); // A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3)
                                 // A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3)
                                 // A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3)
                                 // A3(0-3)

          const __m512i lhs_mat_01_1_sp1 = _mm512_shuffle_epi32(
            lhs_mat_01_1,
            (_MM_PERM_ENUM)160); // A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11)
                                 // A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11)
                                 // A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11)
                                 // A1(8-11)
          const __m512i lhs_mat_23_1_sp1 = _mm512_shuffle_epi32(
            lhs_mat_23_1,
            (_MM_PERM_ENUM)160); // A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11)
                                 // A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11)
                                 // A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11)
                                 // A3(8-11)

          const __m512i lhs_mat_01_2_sp1 = _mm512_shuffle_epi32(
            lhs_mat_01_2,
            (_MM_PERM_ENUM)160); // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                                 // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                                 // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                                 // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
          const __m512i lhs_mat_23_2_sp1 = _mm512_shuffle_epi32(
            lhs_mat_23_2,
            (_MM_PERM_ENUM)160); // A2(16-19) A2(16-19) A3(16-19) A3(16-19)
                                 // A2(16-19) A2(16-19) A3(16-19) A3(16-19)
                                 // A2(16-19) A2(16-19) A3(16-19) A3(16-19)
                                 // A2(16-19) A2(16-19) A3(16-19) A3(16-19)

          const __m512i lhs_mat_01_3_sp1 = _mm512_shuffle_epi32(
            lhs_mat_01_3,
            (_MM_PERM_ENUM)160); // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                                 // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                                 // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                                 // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
          const __m512i lhs_mat_23_3_sp1 = _mm512_shuffle_epi32(
            lhs_mat_23_3,
            (_MM_PERM_ENUM)160); // A2(24-27) A2(24-27) A3(24-27) A3(24-27)
                                 // A2(24-27) A2(24-27) A3(24-27) A3(24-27)
                                 // A2(24-27) A2(24-27) A3(24-27) A3(24-27)
                                 // A2(24-27) A2(24-27) A3(24-27) A3(24-27)

          // Shuffle pattern two - left side input

          const __m512i lhs_mat_01_0_sp2 = _mm512_shuffle_epi32(
            lhs_mat_01_0,
            (_MM_PERM_ENUM)245); // A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7)
                                 // A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7)
                                 // A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7)
                                 // A1(4-7)
          const __m512i lhs_mat_23_0_sp2 = _mm512_shuffle_epi32(
            lhs_mat_23_0,
            (_MM_PERM_ENUM)245); // A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7)
                                 // A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7)
                                 // A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7)
                                 // A3(4-7)

          const __m512i lhs_mat_01_1_sp2 = _mm512_shuffle_epi32(
            lhs_mat_01_1,
            (_MM_PERM_ENUM)245); // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                                 // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                                 // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                                 // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
          const __m512i lhs_mat_23_1_sp2 = _mm512_shuffle_epi32(
            lhs_mat_23_1,
            (_MM_PERM_ENUM)245); // A2(12-15) A2(12-15) A3(12-15) A3(12-15)
                                 // A2(12-15) A2(12-15) A3(12-15) A3(12-15)
                                 // A2(12-15) A2(12-15) A3(12-15) A3(12-15)
                                 // A2(12-15) A2(12-15) A3(12-15) A3(12-15)

          const __m512i lhs_mat_01_2_sp2 = _mm512_shuffle_epi32(
            lhs_mat_01_2,
            (_MM_PERM_ENUM)245); // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                                 // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                                 // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                                 // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
          const __m512i lhs_mat_23_2_sp2 = _mm512_shuffle_epi32(
            lhs_mat_23_2,
            (_MM_PERM_ENUM)245); // A2(20-23) A2(20-23) A3(20-23) A3(20-23)
                                 // A2(20-23) A2(20-23) A3(20-23) A3(20-23)
                                 // A2(20-23) A2(20-23) A3(20-23) A3(20-23)
                                 // A2(20-23) A2(20-23) A3(20-23) A3(20-23)

          const __m512i lhs_mat_01_3_sp2 = _mm512_shuffle_epi32(
            lhs_mat_01_3,
            (_MM_PERM_ENUM)245); // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                                 // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                                 // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                                 // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
          const __m512i lhs_mat_23_3_sp2 = _mm512_shuffle_epi32(
            lhs_mat_23_3,
            (_MM_PERM_ENUM)245); // A2(28-31) A2(28-31) A3(28-31) A3(28-31)
                                 // A2(28-31) A2(28-31) A3(28-31) A3(28-31)
                                 // A2(28-31) A2(28-31) A3(28-31) A3(28-31)
                                 // A2(28-31) A2(28-31) A3(28-31) A3(28-31)

          // The values arranged in shuffle patterns are operated with dot
          // product operation within 32 bit lane i.e corresponding bytes and
          // multiplied and added into 32 bit integers within 32 bit lane
          // Resembles MMLAs into 2x2 matrices in ARM Version
          const __m512i zero = _mm512_setzero_epi32();
          __m512i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x16(
            mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1,
                                              rhs_mat_014589CD_3_sp1),
                lhs_mat_01_2_sp1, rhs_mat_014589CD_2_sp1),
              lhs_mat_01_1_sp1, rhs_mat_014589CD_1_sp1),
            lhs_mat_01_0_sp1, rhs_mat_014589CD_0_sp1);
          __m512i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x16(
            mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1,
                                              rhs_mat_2367ABEF_3_sp1),
                lhs_mat_01_2_sp1, rhs_mat_2367ABEF_2_sp1),
              lhs_mat_01_1_sp1, rhs_mat_2367ABEF_1_sp1),
            lhs_mat_01_0_sp1, rhs_mat_2367ABEF_0_sp1);
          __m512i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x16(
            mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1,
                                              rhs_mat_014589CD_3_sp1),
                lhs_mat_23_2_sp1, rhs_mat_014589CD_2_sp1),
              lhs_mat_23_1_sp1, rhs_mat_014589CD_1_sp1),
            lhs_mat_23_0_sp1, rhs_mat_014589CD_0_sp1);
          __m512i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x16(
            mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1,
                                              rhs_mat_2367ABEF_3_sp1),
                lhs_mat_23_2_sp1, rhs_mat_2367ABEF_2_sp1),
              lhs_mat_23_1_sp1, rhs_mat_2367ABEF_1_sp1),
            lhs_mat_23_0_sp1, rhs_mat_2367ABEF_0_sp1);
          __m512i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x16(
            mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2,
                                              rhs_mat_014589CD_3_sp2),
                lhs_mat_01_2_sp2, rhs_mat_014589CD_2_sp2),
              lhs_mat_01_1_sp2, rhs_mat_014589CD_1_sp2),
            lhs_mat_01_0_sp2, rhs_mat_014589CD_0_sp2);
          __m512i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x16(
            mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2,
                                              rhs_mat_2367ABEF_3_sp2),
                lhs_mat_01_2_sp2, rhs_mat_2367ABEF_2_sp2),
              lhs_mat_01_1_sp2, rhs_mat_2367ABEF_1_sp2),
            lhs_mat_01_0_sp2, rhs_mat_2367ABEF_0_sp2);
          __m512i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x16(
            mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2,
                                              rhs_mat_014589CD_3_sp2),
                lhs_mat_23_2_sp2, rhs_mat_014589CD_2_sp2),
              lhs_mat_23_1_sp2, rhs_mat_014589CD_1_sp2),
            lhs_mat_23_0_sp2, rhs_mat_014589CD_0_sp2);
          __m512i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x16(
            mul_sum_i8_pairs_acc_int32x16(
              mul_sum_i8_pairs_acc_int32x16(
                mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2,
                                              rhs_mat_2367ABEF_3_sp2),
                lhs_mat_23_2_sp2, rhs_mat_2367ABEF_2_sp2),
              lhs_mat_23_1_sp2, rhs_mat_2367ABEF_1_sp2),
            lhs_mat_23_0_sp2, rhs_mat_2367ABEF_0_sp2);

          // Output of both shuffle patterns are added in order to sum dot
          // product outputs of all 32 values in block
          __m512i iacc_mat_00 =
            _mm512_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
          __m512i iacc_mat_01 =
            _mm512_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
          __m512i iacc_mat_10 =
            _mm512_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
          __m512i iacc_mat_11 =
            _mm512_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);

          // Straighten out to make 4 row vectors
          __m512i iacc_row_0 = _mm512_mask_blend_epi32(
            0xCCCC, iacc_mat_00,
            _mm512_shuffle_epi32(iacc_mat_01, (_MM_PERM_ENUM)78));
          __m512i iacc_row_1 = _mm512_mask_blend_epi32(
            0xCCCC, _mm512_shuffle_epi32(iacc_mat_00, (_MM_PERM_ENUM)78),
            iacc_mat_01);
          __m512i iacc_row_2 = _mm512_mask_blend_epi32(
            0xCCCC, iacc_mat_10,
            _mm512_shuffle_epi32(iacc_mat_11, (_MM_PERM_ENUM)78));
          __m512i iacc_row_3 = _mm512_mask_blend_epi32(
            0xCCCC, _mm512_shuffle_epi32(iacc_mat_10, (_MM_PERM_ENUM)78),
            iacc_mat_11);

          // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it
          // across lanes
          const __m128i row_scale_f16 = _mm_shuffle_epi32(
            _mm_maskload_epi32((int const *)(a_ptr[b].d), loadMask), 68);
          const __m512 row_scale_f32 = GGML_F32Cx16_REPEAT_LOAD(row_scale_f16);

          // Multiply with appropiate scales and accumulate
          acc_rows[0] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(iacc_row_0),
            _mm512_mul_ps(col_scale_f32,
                          _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
            acc_rows[0]);
          acc_rows[1] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(iacc_row_1),
            _mm512_mul_ps(col_scale_f32,
                          _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)),
            acc_rows[1]);
          acc_rows[2] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(iacc_row_2),
            _mm512_mul_ps(col_scale_f32,
                          _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
            acc_rows[2]);
          acc_rows[3] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(iacc_row_3),
            _mm512_mul_ps(col_scale_f32,
                          _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
            acc_rows[3]);
        }

        // Store the accumulated values
        for (int i = 0; i < 4; i++) {
          _mm512_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                           acc_rows[i]);
        }
      }
    }
    if (anc != nc) {
      xstart = anc / 8;
      y = 0;
    }
#endif // __AVX512F__

    // Take group of four block_q8_0x4 structures at each pass of the loop and
    // perform dot product operation

    for (; y < anr / 4; y += 4) {
      const block_q8_0x4 *a_ptrs[4];

      a_ptrs[0] = a_ptr_start + (y * nb);
      for (int i = 0; i < 3; ++i) {
        a_ptrs[i + 1] = a_ptrs[i] + nb;
      }

      // Take group of eight block_q4_0x8 structures at each pass of the loop
      // and perform dot product operation
      for (int64_t x = xstart; x < nc / 8; x++) {

        const block_q4_0x8 *b_ptr = b_ptr_start + (x * b_nb);

        // Master FP accumulators
        __m256 acc_rows[16];
        for (int i = 0; i < 16; i++) {
          acc_rows[i] = _mm256_setzero_ps();
        }

        for (int64_t b = 0; b < nb; b++) {
          // Load the eight block_q4_0 quantized values interleaved with each
          // other in chunks of eight - B0,B1 ....B6,B7
          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
          const __m256i rhs_raw_mat_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32));
          const __m256i rhs_raw_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64));
          const __m256i rhs_raw_mat_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96));

          // Save the values in the following vectors in the formats B0B1B4B5,
          // B2B3B6B7 for further processing and storing of values
          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);

          // 4-bit -> 8-bit - Sign is maintained
          const __m256i rhs_mat_0145_0 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_0145_0,
                             m4b)); // B0(0-7) B1(0-7) B4(0-7) B5(0-7)
          const __m256i rhs_mat_2367_0 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_2367_0,
                             m4b)); // B2(0-7) B3(0-7) B6(0-7) B7(0-7)

          const __m256i rhs_mat_0145_1 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_0145_1,
                             m4b)); // B0(8-15) B1(8-15) B4(8-15) B5(8-15)
          const __m256i rhs_mat_2367_1 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_2367_1,
                             m4b)); // B2(8-15) B3(8-15) B6(8-15) B7(8-15)

          const __m256i rhs_mat_0145_2 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4),
                             m4b)); // B0(16-23) B1(16-23) B4(16-23) B5(16-23)
          const __m256i rhs_mat_2367_2 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4),
                             m4b)); // B2(16-23) B3(16-23) B6(16-23) B7(16-23)

          const __m256i rhs_mat_0145_3 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4),
                             m4b)); // B0(24-31) B1(24-31) B4(24-31) B5(24-31)
          const __m256i rhs_mat_2367_3 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4),
                             m4b)); // B2(24-31) B3(24-31) B6(24-31) B7(24-31)

          // Shuffle pattern one - right side input
          const __m256i rhs_mat_0145_0_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_0, 136); // B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3)
                                  // B5(0-3) B4(0-3) B5(0-3)
          const __m256i rhs_mat_2367_0_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_0, 136); // B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3)
                                  // B7(0-3) B6(0-3) B7(0-3)

          const __m256i rhs_mat_0145_1_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_1, 136); // B0(8-11) B1(8-11) B0(8-11) B1(8-11)
                                  // B4(8-11) B5(8-11) B4(8-11) B5(8-11)
          const __m256i rhs_mat_2367_1_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_1, 136); // B2(8-11) B3(8-11) B2(8-11) B3(8-11)
                                  // B6(8-11) B7(8-11) B6(8-11) B7(8-11)

          const __m256i rhs_mat_0145_2_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_2, 136); // B0(16-19) B1(16-19) B0(16-19) B1(16-19)
                                  // B4(16-19) B5(16-19) B4(16-19) B5(16-19)
          const __m256i rhs_mat_2367_2_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_2, 136); // B2(16-19) B3(16-19) B2(16-19) B3(16-19)
                                  // B6(16-19) B7(16-19) B6(16-19) B7(16-19)

          const __m256i rhs_mat_0145_3_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_3, 136); // B0(24-27) B1(24-27) B0(24-27) B1(24-27)
                                  // B4(24-27) B5(24-27) B4(24-27) B5(24-27)
          const __m256i rhs_mat_2367_3_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_3, 136); // B2(24-27) B3(24-27) B2(24-27) B3(24-27)
                                  // B6(24-27) B7(24-27) B6(24-27) B7(24-27)

          // Shuffle pattern two - right side input

          const __m256i rhs_mat_0145_0_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_0, 221); // B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7)
                                  // B5(4-7) B4(4-7) B5(4-7)
          const __m256i rhs_mat_2367_0_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_0, 221); // B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7)
                                  // B7(4-7) B6(4-7) B7(4-7)

          const __m256i rhs_mat_0145_1_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_1, 221); // B0(12-15) B1(12-15) B0(12-15) B1(12-15)
                                  // B4(12-15) B5(12-15) B4(12-15) B5(12-15)
          const __m256i rhs_mat_2367_1_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_1, 221); // B2(12-15) B3(12-15) B2(12-15) B3(12-15)
                                  // B6(12-15) B7(12-15) B6(12-15) B7(12-15)

          const __m256i rhs_mat_0145_2_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_2, 221); // B0(20-23) B1(20-23) B0(20-23) B1(20-23)
                                  // B4(20-23) B5(20-23) B4(20-23) B5(20-23)
          const __m256i rhs_mat_2367_2_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_2, 221); // B2(20-23) B3(20-23) B2(20-23) B3(20-23)
                                  // B6(20-23) B7(20-23) B6(20-23) B7(20-23)

          const __m256i rhs_mat_0145_3_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_3, 221); // B0(28-31) B1(28-31) B0(28-31) B1(28-31)
                                  // B4(28-31) B5(28-31) B4(28-31) B5(28-31)
          const __m256i rhs_mat_2367_3_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_3, 221); // B2(28-31) B3(28-31) B2(28-31) B3(28-31)
                                  // B6(28-31) B7(28-31) B6(28-31) B7(28-31)

          // Scale values - Load the wight scale values of block_q4_0x8
          const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

          // Process LHS in groups of four
          for (int rp = 0; rp < 4; rp++) {
            // Load the four block_q4_0 quantized values interleaved with each
            // other in chunks of eight - A0,A1,A2,A3 Loaded as set of 128 bit
            // vectors and repeated into a 256 bit vector
            __m256i lhs_mat_0123_0 =
              _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs)));
            __m256i lhs_mat_01_0 =
              _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 0);
            __m256i lhs_mat_23_0 =
              _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 17);
            __m256i lhs_mat_0123_1 =
              _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 32)));
            __m256i lhs_mat_01_1 =
              _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 0);
            __m256i lhs_mat_23_1 =
              _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 17);
            __m256i lhs_mat_0123_2 =
              _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 64)));
            __m256i lhs_mat_01_2 =
              _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 0);
            __m256i lhs_mat_23_2 =
              _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 17);
            __m256i lhs_mat_0123_3 =
              _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 96)));
            __m256i lhs_mat_01_3 =
              _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 0);
            __m256i lhs_mat_23_3 =
              _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 17);

            // Shuffle pattern one - left side input
            const __m256i lhs_mat_01_0_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_0, 160); // A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3)
                                  // A0(0-3) A1(0-3) A1(0-3)
            const __m256i lhs_mat_23_0_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_0, 160); // A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3)
                                  // A2(0-3) A3(0-3) A3(0-3)

            const __m256i lhs_mat_01_1_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_1, 160); // A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                                  // A0(8-11) A0(8-11) A1(8-11) A1(8-11)
            const __m256i lhs_mat_23_1_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_1, 160); // A2(8-11) A2(8-11) A3(8-11) A3(8-11)
                                  // A2(8-11) A2(8-11) A3(8-11) A3(8-11)

            const __m256i lhs_mat_01_2_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_2, 160); // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                                  // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
            const __m256i lhs_mat_23_2_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_2, 160); // A2(16-19) A2(16-19) A3(16-19) A3(16-19)
                                  // A2(16-19) A2(16-19) A3(16-19) A3(16-19)

            const __m256i lhs_mat_01_3_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_3, 160); // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                                  // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
            const __m256i lhs_mat_23_3_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_3, 160); // A2(24-27) A2(24-27) A3(24-27) A3(24-27)
                                  // A2(24-27) A2(24-27) A3(24-27) A3(24-27)

            // Shuffle pattern two - left side input
            const __m256i lhs_mat_01_0_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_0, 245); // A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7)
                                  // A0(4-7) A1(4-7) A1(4-7)
            const __m256i lhs_mat_23_0_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_0, 245); // A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7)
                                  // A2(4-7) A3(4-7) A3(4-7)

            const __m256i lhs_mat_01_1_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_1, 245); // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                                  // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
            const __m256i lhs_mat_23_1_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_1, 245); // A2(12-15) A2(12-15) A3(12-15) A3(12-15)
                                  // A2(12-15) A2(12-15) A3(12-15) A3(12-15)

            const __m256i lhs_mat_01_2_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_2, 245); // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                                  // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
            const __m256i lhs_mat_23_2_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_2, 245); // A2(20-23) A2(20-23) A3(20-23) A3(20-23)
                                  // A2(20-23) A2(20-23) A3(20-23) A3(20-23)

            const __m256i lhs_mat_01_3_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_3, 245); // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                                  // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
            const __m256i lhs_mat_23_3_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_3, 245); // A2(28-31) A2(28-31) A3(28-31) A3(28-31)
                                  // A2(28-31) A2(28-31) A3(28-31) A3(28-31)

            // The values arranged in shuffle patterns are operated with dot
            // product operation within 32 bit lane i.e corresponding bytes and
            // multiplied and added into 32 bit integers within 32 bit lane
            // Resembles MMLAs into 2x2 matrices in ARM Version
            const __m256i zero = _mm256_setzero_si256();
            __m256i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1,
                                               rhs_mat_0145_3_sp1),
                  lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1),
                lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1),
              lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1);
            __m256i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1,
                                               rhs_mat_2367_3_sp1),
                  lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1),
                lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1),
              lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1);
            __m256i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1,
                                               rhs_mat_0145_3_sp1),
                  lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1),
                lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1),
              lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1);
            __m256i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1,
                                               rhs_mat_2367_3_sp1),
                  lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1),
                lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1),
              lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1);
            __m256i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2,
                                               rhs_mat_0145_3_sp2),
                  lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2),
                lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2),
              lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2);
            __m256i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2,
                                               rhs_mat_2367_3_sp2),
                  lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2),
                lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2),
              lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2);
            __m256i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2,
                                               rhs_mat_0145_3_sp2),
                  lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2),
                lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2),
              lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2);
            __m256i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(
                  mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2,
                                               rhs_mat_2367_3_sp2),
                  lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2),
                lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2),
              lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2);

            // Output of both shuffle patterns are added in order to sum dot
            // product outputs of all 32 values in block
            __m256i iacc_mat_00 =
              _mm256_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
            __m256i iacc_mat_01 =
              _mm256_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
            __m256i iacc_mat_10 =
              _mm256_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
            __m256i iacc_mat_11 =
              _mm256_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);

            // Straighten out to make 4 row vectors
            __m256i iacc_row_0 = _mm256_blend_epi32(
              iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
            __m256i iacc_row_1 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
            __m256i iacc_row_2 = _mm256_blend_epi32(
              iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
            __m256i iacc_row_3 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

            // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it
            // across lanes
            const __m256 row_scale_f32 =
              GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

            // Multiply with appropiate scales and accumulate
            acc_rows[rp * 4] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_0),
              _mm256_mul_ps(col_scale_f32,
                            _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
              acc_rows[rp * 4]);
            acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_1),
              _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(
                                             row_scale_f32, row_scale_f32, 85)),
              acc_rows[rp * 4 + 1]);
            acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_2),
              _mm256_mul_ps(
                col_scale_f32,
                _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
              acc_rows[rp * 4 + 2]);
            acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_3),
              _mm256_mul_ps(
                col_scale_f32,
                _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
              acc_rows[rp * 4 + 3]);
          }
        }

        // Store the accumulated values
        for (int i = 0; i < 16; i++) {
          _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                           acc_rows[i]);
        }
      }
    }

    // Take a block_q8_0x4 structures at each pass of the loop and perform dot
    // product operation
    for (; y < nr / 4; y++) {

      const block_q8_0x4 *a_ptr = a_ptr_start + (y * nb);

      // Load the eight block_q4_0 quantized values interleaved with each other
      // in chunks of eight - B0,B1 ....B6,B7
      for (int64_t x = xstart; x < nc / 8; x++) {

        const block_q4_0x8 *b_ptr = b_ptr_start + (x * b_nb);

        // Master FP accumulators
        __m256 acc_rows[4];
        for (int i = 0; i < 4; i++) {
          acc_rows[i] = _mm256_setzero_ps();
        }

        for (int64_t b = 0; b < nb; b++) {
          // Load the eight block_q8_0 quantized values interleaved with each
          // other in chunks of eight - B0,B1 ....B6,B7
          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
          const __m256i rhs_raw_mat_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32));
          const __m256i rhs_raw_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64));
          const __m256i rhs_raw_mat_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96));

          // Save the values in the following vectors in the formats B0B1B4B5,
          // B2B3B6B7 for further processing and storing of valuess
          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);

          // 4-bit -> 8-bit - Sign is maintained
          const __m256i rhs_mat_0145_0 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_0145_0,
                             m4b)); // B0(0-7) B1(0-7) B4(0-7) B5(0-7)
          const __m256i rhs_mat_2367_0 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_2367_0,
                             m4b)); // B2(0-7) B3(0-7) B6(0-7) B7(0-7)

          const __m256i rhs_mat_0145_1 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_0145_1,
                             m4b)); // B0(8-15) B1(8-15) B4(8-15) B5(8-15)
          const __m256i rhs_mat_2367_1 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(rhs_raw_mat_2367_1,
                             m4b)); // B2(8-15) B3(8-15) B6(8-15) B7(8-15)

          const __m256i rhs_mat_0145_2 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4),
                             m4b)); // B0(16-23) B1(16-23) B4(16-23) B5(16-23)
          const __m256i rhs_mat_2367_2 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4),
                             m4b)); // B2(16-23) B3(16-23) B6(16-23) B7(16-23)

          const __m256i rhs_mat_0145_3 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4),
                             m4b)); // B0(24-31) B1(24-31) B4(24-31) B5(24-31)
          const __m256i rhs_mat_2367_3 = _mm256_shuffle_epi8(
            signextendlut,
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4),
                             m4b)); // B2(24-31) B3(24-31) B6(24-31) B7(24-31)

          // Shuffle pattern one - right side input
          const __m256i rhs_mat_0145_0_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_0, 136); // B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3)
                                  // B5(0-3) B4(0-3) B5(0-3)
          const __m256i rhs_mat_2367_0_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_0, 136); // B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3)
                                  // B7(0-3) B6(0-3) B7(0-3)

          const __m256i rhs_mat_0145_1_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_1, 136); // B0(8-11) B1(8-11) B0(8-11) B1(8-11)
                                  // B4(8-11) B5(8-11) B4(8-11) B5(8-11)
          const __m256i rhs_mat_2367_1_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_1, 136); // B2(8-11) B3(8-11) B2(8-11) B3(8-11)
                                  // B6(8-11) B7(8-11) B6(8-11) B7(8-11)

          const __m256i rhs_mat_0145_2_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_2, 136); // B0(16-19) B1(16-19) B0(16-19) B1(16-19)
                                  // B4(16-19) B5(16-19) B4(16-19) B5(16-19)
          const __m256i rhs_mat_2367_2_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_2, 136); // B2(16-19) B3(16-19) B2(16-19) B3(16-19)
                                  // B6(16-19) B7(16-19) B6(16-19) B7(16-19)

          const __m256i rhs_mat_0145_3_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_3, 136); // B0(24-27) B1(24-27) B0(24-27) B1(24-27)
                                  // B4(24-27) B5(24-27) B4(24-27) B5(24-27)
          const __m256i rhs_mat_2367_3_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_3, 136); // B2(24-27) B3(24-27) B2(24-27) B3(24-27)
                                  // B6(24-27) B7(24-27) B6(24-27) B7(24-27)

          // Shuffle pattern two - right side input

          const __m256i rhs_mat_0145_0_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_0, 221); // B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7)
                                  // B5(4-7) B4(4-7) B5(4-7)
          const __m256i rhs_mat_2367_0_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_0, 221); // B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7)
                                  // B7(4-7) B6(4-7) B7(4-7)

          const __m256i rhs_mat_0145_1_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_1, 221); // B0(12-15) B1(12-15) B0(12-15) B1(12-15)
                                  // B4(12-15) B5(12-15) B4(12-15) B5(12-15)
          const __m256i rhs_mat_2367_1_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_1, 221); // B2(12-15) B3(12-15) B2(12-15) B3(12-15)
                                  // B6(12-15) B7(12-15) B6(12-15) B7(12-15)

          const __m256i rhs_mat_0145_2_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_2, 221); // B0(20-23) B1(20-23) B0(20-23) B1(20-23)
                                  // B4(20-23) B5(20-23) B4(20-23) B5(20-23)
          const __m256i rhs_mat_2367_2_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_2, 221); // B2(20-23) B3(20-23) B2(20-23) B3(20-23)
                                  // B6(20-23) B7(20-23) B6(20-23) B7(20-23)

          const __m256i rhs_mat_0145_3_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_3, 221); // B0(28-31) B1(28-31) B0(28-31) B1(28-31)
                                  // B4(28-31) B5(28-31) B4(28-31) B5(28-31)
          const __m256i rhs_mat_2367_3_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_3, 221); // B2(28-31) B3(28-31) B2(28-31) B3(28-31)
                                  // B6(28-31) B7(28-31) B6(28-31) B7(28-31)

          // Scale values - Load the wight scale values of block_q4_0x8
          const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

          // Load the four block_q4_0 quantized values interleaved with each
          // other in chunks of eight - A0,A1,A2,A3 Loaded as set of 128 bit
          // vectors and repeated into a 256 bit vector
          __m256i lhs_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs)));
          __m256i lhs_mat_01_0 =
            _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 0);
          __m256i lhs_mat_23_0 =
            _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 17);
          __m256i lhs_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 32)));
          __m256i lhs_mat_01_1 =
            _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 0);
          __m256i lhs_mat_23_1 =
            _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 17);
          __m256i lhs_mat_0123_2 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 64)));
          __m256i lhs_mat_01_2 =
            _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 0);
          __m256i lhs_mat_23_2 =
            _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 17);
          __m256i lhs_mat_0123_3 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 96)));
          __m256i lhs_mat_01_3 =
            _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 0);
          __m256i lhs_mat_23_3 =
            _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 17);

          // Shuffle pattern one - left side input

          const __m256i lhs_mat_01_0_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_0, 160); // A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3)
                                // A0(0-3) A1(0-3) A1(0-3)
          const __m256i lhs_mat_23_0_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_0, 160); // A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3)
                                // A2(0-3) A3(0-3) A3(0-3)

          const __m256i lhs_mat_01_1_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_1, 160); // A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11)
                                // A0(8-11) A1(8-11) A1(8-11)
          const __m256i lhs_mat_23_1_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_1, 160); // A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11)
                                // A2(8-11) A3(8-11) A3(8-11)

          const __m256i lhs_mat_01_2_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_2, 160); // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                                // A0(16-19) A0(16-19) A1(16-19) A1(16-19)
          const __m256i lhs_mat_23_2_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_2, 160); // A2(16-19) A2(16-19) A3(16-19) A3(16-19)
                                // A2(16-19) A2(16-19) A3(16-19) A3(16-19)

          const __m256i lhs_mat_01_3_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_3, 160); // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                                // A0(24-27) A0(24-27) A1(24-27) A1(24-27)
          const __m256i lhs_mat_23_3_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_3, 160); // A2(24-27) A2(24-27) A3(24-27) A3(24-27)
                                // A2(24-27) A2(24-27) A3(24-27) A3(24-27)

          // Shuffle pattern two - left side input

          const __m256i lhs_mat_01_0_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_0, 245); // A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7)
                                // A0(4-7) A1(4-7) A1(4-7)
          const __m256i lhs_mat_23_0_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_0, 245); // A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7)
                                // A2(4-7) A3(4-7) A3(4-7)

          const __m256i lhs_mat_01_1_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_1, 245); // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                                // A0(12-15) A0(12-15) A1(12-15) A1(12-15)
          const __m256i lhs_mat_23_1_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_1, 245); // A2(12-15) A2(12-15) A3(12-15) A3(12-15)
                                // A2(12-15) A2(12-15) A3(12-15) A3(12-15)

          const __m256i lhs_mat_01_2_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_2, 245); // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                                // A0(20-23) A0(20-23) A1(20-23) A1(20-23)
          const __m256i lhs_mat_23_2_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_2, 245); // A2(20-23) A2(20-23) A3(20-23) A3(20-23)
                                // A2(20-23) A2(20-23) A3(20-23) A3(20-23)

          const __m256i lhs_mat_01_3_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_3, 245); // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                                // A0(28-31) A0(28-31) A1(28-31) A1(28-31)
          const __m256i lhs_mat_23_3_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_3, 245); // A2(28-31) A2(28-31) A3(28-31) A3(28-31)
                                // A2(28-31) A2(28-31) A3(28-31) A3(28-31)

          // The values arranged in shuffle patterns are operated with dot
          // product operation within 32 bit lane i.e corresponding bytes and
          // multiplied and added into 32 bit integers within 32 bit lane
          // Resembles MMLAs into 2x2 matrices in ARM Version
          const __m256i zero = _mm256_setzero_si256();
          __m256i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1,
                                             rhs_mat_0145_3_sp1),
                lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1),
              lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1),
            lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1);
          __m256i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1,
                                             rhs_mat_2367_3_sp1),
                lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1),
              lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1),
            lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1);
          __m256i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1,
                                             rhs_mat_0145_3_sp1),
                lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1),
              lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1),
            lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1);
          __m256i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1,
                                             rhs_mat_2367_3_sp1),
                lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1),
              lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1),
            lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1);
          __m256i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2,
                                             rhs_mat_0145_3_sp2),
                lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2),
              lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2),
            lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2);
          __m256i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2,
                                             rhs_mat_2367_3_sp2),
                lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2),
              lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2),
            lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2);
          __m256i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2,
                                             rhs_mat_0145_3_sp2),
                lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2),
              lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2),
            lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2);
          __m256i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x8(
            mul_sum_i8_pairs_acc_int32x8(
              mul_sum_i8_pairs_acc_int32x8(
                mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2,
                                             rhs_mat_2367_3_sp2),
                lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2),
              lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2),
            lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2);

          // Output of both shuffle patterns are added in order to sum dot
          // product outputs of all 32 values in block
          __m256i iacc_mat_00 =
            _mm256_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
          __m256i iacc_mat_01 =
            _mm256_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
          __m256i iacc_mat_10 =
            _mm256_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
          __m256i iacc_mat_11 =
            _mm256_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);

          // Straighten out to make 4 row vectors
          __m256i iacc_row_0 = _mm256_blend_epi32(
            iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
          __m256i iacc_row_1 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
          __m256i iacc_row_2 = _mm256_blend_epi32(
            iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
          __m256i iacc_row_3 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

          // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it
          // across lanes
          const __m256 row_scale_f32 =
            GGML_F32Cx8_REPEAT_LOAD(a_ptr[b].d, loadMask);

          // Multiply with appropiate scales and accumulate
          acc_rows[0] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_0),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
            acc_rows[0]);
          acc_rows[1] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_1),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)),
            acc_rows[1]);
          acc_rows[2] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_2),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
            acc_rows[2]);
          acc_rows[3] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_3),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
            acc_rows[3]);
        }

        // Store the accumulated values
        for (int i = 0; i < 4; i++) {
          _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                           acc_rows[i]);
        }
      }
    }
    return;
  }
#elif defined(__riscv_v_intrinsic)
  if (__riscv_vlenb() >= QK4_0) {
    const size_t vl = QK4_0;

    for (int y = 0; y < nr / 4; y++) {
      const block_q8_0x4 *a_ptr = (const block_q8_0x4 *)vy + (y * nb);
      for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x8 *b_ptr = (const block_q4_0x8 *)vx + (x * nb);
        vfloat32m1_t sumf0 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
        vfloat32m1_t sumf1 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
        vfloat32m1_t sumf2 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
        vfloat32m1_t sumf3 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
        for (int l = 0; l < nb; l++) {
          const vint8m4_t rhs_raw_vec =
            __riscv_vle8_v_i8m4((const int8_t *)b_ptr[l].qs, vl * 4);
          const vint8m4_t rhs_vec_lo = __riscv_vsra_vx_i8m4(
            __riscv_vsll_vx_i8m4(rhs_raw_vec, 4, vl * 4), 4, vl * 4);
          const vint8m4_t rhs_vec_hi =
            __riscv_vsra_vx_i8m4(rhs_raw_vec, 4, vl * 4);
          const vint8m2_t rhs_vec_lo_0 =
            __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 0);
          const vint8m2_t rhs_vec_lo_1 =
            __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 1);
          const vint8m2_t rhs_vec_hi_0 =
            __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 0);
          const vint8m2_t rhs_vec_hi_1 =
            __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 1);

          // vector version needs Zvfhmin extension
          const float a_scales[4] = {
            nntr_fp16_to_fp32(a_ptr[l].d[0]), nntr_fp16_to_fp32(a_ptr[l].d[1]),
            nntr_fp16_to_fp32(a_ptr[l].d[2]), nntr_fp16_to_fp32(a_ptr[l].d[3])};
          const float b_scales[8] = {
            nntr_fp16_to_fp32(b_ptr[l].d[0]), nntr_fp16_to_fp32(b_ptr[l].d[1]),
            nntr_fp16_to_fp32(b_ptr[l].d[2]), nntr_fp16_to_fp32(b_ptr[l].d[3]),
            nntr_fp16_to_fp32(b_ptr[l].d[4]), nntr_fp16_to_fp32(b_ptr[l].d[5]),
            nntr_fp16_to_fp32(b_ptr[l].d[6]), nntr_fp16_to_fp32(b_ptr[l].d[7])};
          const vfloat32m1_t b_scales_vec =
            __riscv_vle32_v_f32m1(b_scales, vl / 4);

          const int64_t A0 = *(const int64_t *)&a_ptr[l].qs[0];
          const int64_t A4 = *(const int64_t *)&a_ptr[l].qs[32];
          const int64_t A8 = *(const int64_t *)&a_ptr[l].qs[64];
          const int64_t Ac = *(const int64_t *)&a_ptr[l].qs[96];
          __asm__ __volatile__("" ::
                                 : "memory"); // prevent gcc from emitting fused
                                              // vlse64, violating alignment
          vint16m4_t sumi_l0;
          {
            const vint8m2_t lhs_0_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(A0, vl / 4));
            const vint8m2_t lhs_1_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(A4, vl / 4));
            const vint8m2_t lhs_2_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(A8, vl / 4));
            const vint8m2_t lhs_3_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(Ac, vl / 4));
            const vint16m4_t sumi_lo_0 =
              __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
            const vint16m4_t sumi_lo_1 =
              __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
            const vint16m4_t sumi_hi_0 =
              __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
            const vint16m4_t sumi_hi_m =
              __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

            sumi_l0 = sumi_hi_m;
          }

          {
            const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(
              __riscv_vreinterpret_v_i16m4_i32m4(sumi_l0));
            const vuint16m2_t sumi_h2_0 =
              __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
            const vuint16m2_t sumi_h2_1 =
              __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
            const vuint16m2_t sumi_h2 =
              __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
            const vuint32m2_t sumi_h2_i32 =
              __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
            const vuint16m1_t sumi_h4_0 =
              __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
            const vuint16m1_t sumi_h4_1 =
              __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
            const vuint16m1_t sumi_h4 =
              __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
            const vuint32m1_t sumi_h4_i32 =
              __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
            const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(
              __riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
            const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(
              __riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
            const vint32m1_t sumi_h8 =
              __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
            const vfloat32m1_t facc =
              __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

            const vfloat32m1_t tmp1 =
              __riscv_vfmul_vf_f32m1(facc, a_scales[0], vl / 4);
            sumf0 = __riscv_vfmacc_vv_f32m1(sumf0, tmp1, b_scales_vec, vl / 4);
          }

          const int64_t A1 = *(const int64_t *)&a_ptr[l].qs[8];
          const int64_t A5 = *(const int64_t *)&a_ptr[l].qs[40];
          const int64_t A9 = *(const int64_t *)&a_ptr[l].qs[72];
          const int64_t Ad = *(const int64_t *)&a_ptr[l].qs[104];
          __asm__ __volatile__("" ::
                                 : "memory"); // prevent gcc from emitting fused
                                              // vlse64, violating alignment
          vint16m4_t sumi_l1;
          {
            const vint8m2_t lhs_0_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(A1, vl / 4));
            const vint8m2_t lhs_1_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(A5, vl / 4));
            const vint8m2_t lhs_2_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(A9, vl / 4));
            const vint8m2_t lhs_3_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(Ad, vl / 4));
            const vint16m4_t sumi_lo_0 =
              __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
            const vint16m4_t sumi_lo_1 =
              __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
            const vint16m4_t sumi_hi_0 =
              __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
            const vint16m4_t sumi_hi_m =
              __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

            sumi_l1 = sumi_hi_m;
          }

          {
            const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(
              __riscv_vreinterpret_v_i16m4_i32m4(sumi_l1));
            const vuint16m2_t sumi_h2_0 =
              __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
            const vuint16m2_t sumi_h2_1 =
              __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
            const vuint16m2_t sumi_h2 =
              __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
            const vuint32m2_t sumi_h2_i32 =
              __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
            const vuint16m1_t sumi_h4_0 =
              __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
            const vuint16m1_t sumi_h4_1 =
              __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
            const vuint16m1_t sumi_h4 =
              __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
            const vuint32m1_t sumi_h4_i32 =
              __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
            const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(
              __riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
            const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(
              __riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
            const vint32m1_t sumi_h8 =
              __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
            const vfloat32m1_t facc =
              __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

            const vfloat32m1_t tmp1 =
              __riscv_vfmul_vf_f32m1(facc, a_scales[1], vl / 4);
            sumf1 = __riscv_vfmacc_vv_f32m1(sumf1, tmp1, b_scales_vec, vl / 4);
          }

          const int64_t A2 = *(const int64_t *)&a_ptr[l].qs[16];
          const int64_t A6 = *(const int64_t *)&a_ptr[l].qs[48];
          const int64_t Aa = *(const int64_t *)&a_ptr[l].qs[80];
          const int64_t Ae = *(const int64_t *)&a_ptr[l].qs[112];
          __asm__ __volatile__("" ::
                                 : "memory"); // prevent gcc from emitting fused
                                              // vlse64, violating alignment
          vint16m4_t sumi_l2;
          {
            const vint8m2_t lhs_0_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(A2, vl / 4));
            const vint8m2_t lhs_1_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(A6, vl / 4));
            const vint8m2_t lhs_2_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(Aa, vl / 4));
            const vint8m2_t lhs_3_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(Ae, vl / 4));
            const vint16m4_t sumi_lo_0 =
              __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
            const vint16m4_t sumi_lo_1 =
              __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
            const vint16m4_t sumi_hi_0 =
              __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
            const vint16m4_t sumi_hi_m =
              __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

            sumi_l2 = sumi_hi_m;
          }

          {
            const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(
              __riscv_vreinterpret_v_i16m4_i32m4(sumi_l2));
            const vuint16m2_t sumi_h2_0 =
              __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
            const vuint16m2_t sumi_h2_1 =
              __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
            const vuint16m2_t sumi_h2 =
              __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
            const vuint32m2_t sumi_h2_i32 =
              __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
            const vuint16m1_t sumi_h4_0 =
              __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
            const vuint16m1_t sumi_h4_1 =
              __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
            const vuint16m1_t sumi_h4 =
              __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
            const vuint32m1_t sumi_h4_i32 =
              __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
            const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(
              __riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
            const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(
              __riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
            const vint32m1_t sumi_h8 =
              __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
            const vfloat32m1_t facc =
              __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

            const vfloat32m1_t tmp1 =
              __riscv_vfmul_vf_f32m1(facc, a_scales[2], vl / 4);
            sumf2 = __riscv_vfmacc_vv_f32m1(sumf2, tmp1, b_scales_vec, vl / 4);
          }

          const int64_t A3 = *(const int64_t *)&a_ptr[l].qs[24];
          const int64_t A7 = *(const int64_t *)&a_ptr[l].qs[56];
          const int64_t Ab = *(const int64_t *)&a_ptr[l].qs[88];
          const int64_t Af = *(const int64_t *)&a_ptr[l].qs[120];
          __asm__ __volatile__("" ::
                                 : "memory"); // prevent gcc from emitting fused
                                              // vlse64, violating alignment
          vint16m4_t sumi_l3;
          {
            const vint8m2_t lhs_0_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(A3, vl / 4));
            const vint8m2_t lhs_1_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(A7, vl / 4));
            const vint8m2_t lhs_2_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(Ab, vl / 4));
            const vint8m2_t lhs_3_8 = __riscv_vreinterpret_v_i64m2_i8m2(
              __riscv_vmv_v_x_i64m2(Af, vl / 4));
            const vint16m4_t sumi_lo_0 =
              __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
            const vint16m4_t sumi_lo_1 =
              __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
            const vint16m4_t sumi_hi_0 =
              __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
            const vint16m4_t sumi_hi_m =
              __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

            sumi_l3 = sumi_hi_m;
          }

          {
            const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(
              __riscv_vreinterpret_v_i16m4_i32m4(sumi_l3));
            const vuint16m2_t sumi_h2_0 =
              __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
            const vuint16m2_t sumi_h2_1 =
              __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
            const vuint16m2_t sumi_h2 =
              __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
            const vuint32m2_t sumi_h2_i32 =
              __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
            const vuint16m1_t sumi_h4_0 =
              __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
            const vuint16m1_t sumi_h4_1 =
              __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
            const vuint16m1_t sumi_h4 =
              __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
            const vuint32m1_t sumi_h4_i32 =
              __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
            const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(
              __riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
            const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(
              __riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
            const vint32m1_t sumi_h8 =
              __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
            const vfloat32m1_t facc =
              __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

            const vfloat32m1_t tmp1 =
              __riscv_vfmul_vf_f32m1(facc, a_scales[3], vl / 4);
            sumf3 = __riscv_vfmacc_vv_f32m1(sumf3, tmp1, b_scales_vec, vl / 4);
          }
        }
        __riscv_vse32_v_f32m1(&s[(y * 4 + 0) * bs + x * ncols_interleaved],
                              sumf0, vl / 4);
        __riscv_vse32_v_f32m1(&s[(y * 4 + 1) * bs + x * ncols_interleaved],
                              sumf1, vl / 4);
        __riscv_vse32_v_f32m1(&s[(y * 4 + 2) * bs + x * ncols_interleaved],
                              sumf2, vl / 4);
        __riscv_vse32_v_f32m1(&s[(y * 4 + 3) * bs + x * ncols_interleaved],
                              sumf3, vl / 4);
      }
    }

    return;
  }
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) &&
       // defined(__aarch64__)
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

#if defined(__AVX2__) || defined(__AVX512F__)
  const block_q4_Kx8 *b_ptr_start = (const block_q4_Kx8 *)vx;
  const block_q8_Kx4 *a_ptr_start = (const block_q8_Kx4 *)vy;
  int64_t b_nb = n / QK_K;
  int64_t y = 0;

  // Mask to mask out nibbles from packed bytes
  const __m256i m4b = _mm256_set1_epi8(0x0F);
  // Permute mask used for easier vector processing at later stages
  __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
  int64_t xstart = 0;
  int anr = nr - nr % 16;
  ; // Used to align nr with boundary of 16
#ifdef __AVX512F__
  int anc = nc - nc % 16; // Used to align nc with boundary of 16
  // Mask to mask out nibbles from packed bytes expanded to 512 bit length
  const __m512i m4bexpanded = _mm512_set1_epi8(0x0F);
  // Take group of four block_q8_Kx4 structures at each pass of the loop and
  // perform dot product operation
  for (; y < anr / 4; y += 4) {

    const block_q8_Kx4 *a_ptrs[4];

    a_ptrs[0] = a_ptr_start + (y * nb);
    for (int i = 0; i < 3; ++i) {
      a_ptrs[i + 1] = a_ptrs[i] + nb;
    }

    // Take group of eight block_q4_kx8 structures at each pass of the loop and
    // perform dot product operation
    for (int64_t x = 0; x < anc / 8; x += 2) {

      const block_q4_Kx8 *b_ptr_0 = b_ptr_start + ((x)*b_nb);
      const block_q4_Kx8 *b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

      // Master FP accumulators
      __m512 acc_rows[16];
      for (int i = 0; i < 16; i++) {
        acc_rows[i] = _mm512_setzero_ps();
      }

      __m512 acc_min_rows[16];
      for (int i = 0; i < 16; i++) {
        acc_min_rows[i] = _mm512_setzero_ps();
      }

      // For super block
      for (int64_t b = 0; b < nb; b++) {
        // Scale values - Load the sixteen scale values from two block_q4_kx8
        // structures
        const __m512 col_scale_f32 =
          GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

        // dmin values - Load the sixteen dmin values from two block_q4_kx8
        // structures
        const __m512 col_dmin_f32 =
          GGML_F32Cx8x2_LOAD(b_ptr_0[b].dmin, b_ptr_1[b].dmin);

        // Loop to iterate over the eight sub blocks of a super block - two sub
        // blocks are processed per iteration
        for (int sb = 0; sb < QK_K / 64; sb++) {

          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + sb * 256));
          const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 32 + sb * 256));
          const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 64 + sb * 256));
          const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 96 + sb * 256));
          const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 128 + sb * 256));
          const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 160 + sb * 256));
          const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 192 + sb * 256));
          const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 224 + sb * 256));

          const __m256i rhs_raw_mat_89AB_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + sb * 256));
          const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 32 + sb * 256));
          const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 64 + sb * 256));
          const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 96 + sb * 256));
          const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 128 + sb * 256));
          const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 160 + sb * 256));
          const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 192 + sb * 256));
          const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 224 + sb * 256));

          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);
          const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(
            rhs_raw_mat_0123_2,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder),
            rhs_raw_mat_4567_2, 240);
          const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(
            rhs_raw_mat_0123_3,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder),
            rhs_raw_mat_4567_3, 240);

          const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(
            rhs_raw_mat_89AB_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder),
            rhs_raw_mat_CDEF_0, 240);
          const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(
            rhs_raw_mat_89AB_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder),
            rhs_raw_mat_CDEF_1, 240);
          const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(
            rhs_raw_mat_89AB_2,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder),
            240);
          const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder),
            rhs_raw_mat_CDEF_2, 240);
          const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(
            rhs_raw_mat_89AB_3,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder),
            240);
          const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder),
            rhs_raw_mat_CDEF_3, 240);

          const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
          const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
          const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
          const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

          const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
          const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
          const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
          const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

          // 4-bit -> 8-bit
          const __m512i rhs_mat_014589CD_00 = _mm512_and_si512(
            rhs_raw_mat_014589CD_0,
            m4bexpanded); // B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7)
                          // B09(0-7) B0C(0-7) B0D(0-7)
          const __m512i rhs_mat_2367ABEF_00 = _mm512_and_si512(
            rhs_raw_mat_2367ABEF_0,
            m4bexpanded); // B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7)
                          // B0B(0-7) B0E(0-7) B0F(0-7)
          const __m512i rhs_mat_014589CD_01 = _mm512_and_si512(
            rhs_raw_mat_014589CD_1,
            m4bexpanded); // B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15)
                          // B09(8-15) B0C(8-15) B0D(8-15)
          const __m512i rhs_mat_2367ABEF_01 = _mm512_and_si512(
            rhs_raw_mat_2367ABEF_1,
            m4bexpanded); // B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15)
                          // B0B(8-15) B0E(8-15) B0F(8-15)

          const __m512i rhs_mat_014589CD_02 = _mm512_and_si512(
            rhs_raw_mat_014589CD_2,
            m4bexpanded); // B00(16-23) B01(16-23) B04(16-23) B05(16-23)
                          // B08(16-23) B09(16-23) B0C(16-23) B0D(16-23)
          const __m512i rhs_mat_2367ABEF_02 = _mm512_and_si512(
            rhs_raw_mat_2367ABEF_2,
            m4bexpanded); // B02(16-23) B03(16-23) B06(16-23) B07(16-23)
                          // B0A(16-23) B0B(16-23) B0E(16-23) B0F(16-23)
          const __m512i rhs_mat_014589CD_03 = _mm512_and_si512(
            rhs_raw_mat_014589CD_3,
            m4bexpanded); // B00(24-31) B01(24-31) B04(24-31) B05(24-31)
                          // B08(24-31) B09(24-31) B0C(24-31) B0D(24-31)
          const __m512i rhs_mat_2367ABEF_03 = _mm512_and_si512(
            rhs_raw_mat_2367ABEF_3,
            m4bexpanded); // B02(24-31) B03(24-31) B06(24-31) B07(24-31)
                          // B0A(24-31) B0B(24-31) B0E(24-31) B0F(24-31)

          const __m512i rhs_mat_014589CD_10 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4),
            m4bexpanded); // B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7)
                          // B19(0-7) B1C(0-7) B1D(0-7)
          const __m512i rhs_mat_2367ABEF_10 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4),
            m4bexpanded); // B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7)
                          // B1B(0-7) B1E(0-7) B1F(0-7)
          const __m512i rhs_mat_014589CD_11 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4),
            m4bexpanded); // B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15)
                          // B19(8-15) B1C(8-15) B1D(8-15)
          const __m512i rhs_mat_2367ABEF_11 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4),
            m4bexpanded); // B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15)
                          // B1B(8-15) B1E(8-15) B1F(8-15)

          const __m512i rhs_mat_014589CD_12 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4),
            m4bexpanded); // B10(16-23) B11(16-23) B14(16-23) B15(16-23)
                          // B18(16-23) B19(16-23) B1C(16-23) B1D(16-23)
          const __m512i rhs_mat_2367ABEF_12 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4),
            m4bexpanded); // B12(16-23) B13(16-23) B16(16-23) B17(16-23)
                          // B1A(16-23) B1B(16-23) B1E(16-23) B1F(16-23)
          const __m512i rhs_mat_014589CD_13 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4),
            m4bexpanded); // B10(24-31) B11(24-31) B14(24-31) B15(24-31)
                          // B18(24-31) B19(24-31) B1C(24-31) B1D(24-31)
          const __m512i rhs_mat_2367ABEF_13 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4),
            m4bexpanded); // B12(24-31) B13(24-31) B16(24-31) B17(24-31)
                          // B1A(24-31) B1B(24-31) B1E(24-31) B1F(24-31)

          // Shuffle pattern one - right side input
          const __m512i rhs_mat_014589CD_00_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_00,
            (_MM_PERM_ENUM)136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3)
                                 // B05(0-3) B04(0-3) B05(0-3) B08(0-3) B09(0-3)
                                 // B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3)
                                 // B0D(0-3)
          const __m512i rhs_mat_2367ABEF_00_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_00,
            (_MM_PERM_ENUM)136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3)
                                 // B07(0-3) B06(0-3) B07(0-3) B0A(0-3) B0B(0-3)
                                 // B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3)
                                 // B0F(0-3)
          const __m512i rhs_mat_014589CD_01_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_01,
            (_MM_PERM_ENUM)136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11)
                                 // B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                                 // B08(8-11) B09(8-11) B08(8-11) B09(8-11)
                                 // B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
          const __m512i rhs_mat_2367ABEF_01_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_01,
            (_MM_PERM_ENUM)136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11)
                                 // B06(8-11) B07(8-11) B06(8-11) B07(8-11)
                                 // B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11)
                                 // B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)
          const __m512i rhs_mat_014589CD_02_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_02,
            (_MM_PERM_ENUM)136); // B00(16-19) B01(16-19) B00(16-19) B01(16-19)
                                 // B04(16-19) B05(16-19) B04(16-19) B05(16-19)
                                 // B08(16-19) B09(16-19) B08(16-19) B09(16-19)
                                 // B0C(16-19) B0D(16-19) B0C(16-19) B0D(16-19)
          const __m512i rhs_mat_2367ABEF_02_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_02,
            (_MM_PERM_ENUM)136); // B02(16-19) B03(16-19) B02(16-19) B03(16-19)
                                 // B06(16-19) B07(16-19) B06(16-19) B07(16-19)
                                 // B0A(16-19) B0B(16-19) B0A(16-19) B0B(16-19)
                                 // B0E(16-19) B0F(16-19) B0E(16-19) B0F(16-19)
          const __m512i rhs_mat_014589CD_03_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_03,
            (_MM_PERM_ENUM)136); // B00(24-27) B01(24-27) B00(24-27) B01(24-27)
                                 // B04(24-27) B05(24-27) B04(24-27) B05(24-27)
                                 // B08(24-27) B09(24-27) B08(24-27) B09(24-27)
                                 // B0C(24-27) B0D(24-27) B0C(24-27) B0D(24-27)
          const __m512i rhs_mat_2367ABEF_03_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_03,
            (_MM_PERM_ENUM)136); // B02(24-27) B03(24-27) B02(24-27) B03(24-27)
                                 // B06(24-27) B07(24-27) B06(24-27) B07(24-27)
                                 // B0A(24-27) B0B(24-27) B0A(24-27) B0B(24-27)
                                 // B0E(24-27) B0F(24-27) B0E(24-27) B0F(24-27)

          const __m512i rhs_mat_014589CD_10_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_10,
            (_MM_PERM_ENUM)136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3)
                                 // B15(0-3) B14(0-3) B15(0-3) B18(0-3) B19(0-3)
                                 // B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3)
                                 // B1D(0-3)
          const __m512i rhs_mat_2367ABEF_10_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_10,
            (_MM_PERM_ENUM)136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3)
                                 // B17(0-3) B16(0-3) B17(0-3) B1A(0-3) B1B(0-3)
                                 // B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3)
                                 // B1F(0-3)
          const __m512i rhs_mat_014589CD_11_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_11,
            (_MM_PERM_ENUM)136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11)
                                 // B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                                 // B18(8-11) B19(8-11) B18(8-11) B19(8-11)
                                 // B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
          const __m512i rhs_mat_2367ABEF_11_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_11,
            (_MM_PERM_ENUM)136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11)
                                 // B16(8-11) B17(8-11) B16(8-11) B17(8-11)
                                 // B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11)
                                 // B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)
          const __m512i rhs_mat_014589CD_12_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_12,
            (_MM_PERM_ENUM)136); // B10(16-19) B11(16-19) B10(16-19) B11(16-19)
                                 // B14(16-19) B15(16-19) B14(16-19) B15(16-19)
                                 // B18(16-19) B19(16-19) B18(16-19) B19(16-19)
                                 // B1C(16-19) B1D(16-19) B1C(16-19) B1D(16-19)
          const __m512i rhs_mat_2367ABEF_12_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_12,
            (_MM_PERM_ENUM)136); // B12(16-19) B13(16-19) B12(16-19) B13(16-19)
                                 // B16(16-19) B17(16-19) B16(16-19) B17(16-19)
                                 // B1A(16-19) B1B(16-19) B1A(16-19) B1B(16-19)
                                 // B1E(16-19) B1F(16-19) B1E(16-19) B1F(16-19)
          const __m512i rhs_mat_014589CD_13_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_13,
            (_MM_PERM_ENUM)136); // B10(24-27) B11(24-27) B10(24-27) B11(24-27)
                                 // B14(24-27) B15(24-27) B14(24-27) B15(24-27)
                                 // B18(24-27) B19(24-27) B18(24-27) B19(24-27)
                                 // B1C(24-27) B1D(24-27) B1C(24-27) B1D(24-27)
          const __m512i rhs_mat_2367ABEF_13_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_13,
            (_MM_PERM_ENUM)136); // B12(24-27) B13(24-27) B12(24-27) B13(24-27)
                                 // B16(24-27) B17(24-27) B16(24-27) B17(24-27)
                                 // B1A(24-27) B1B(24-27) B1A(24-27) B1B(24-27)
                                 // B1E(24-27) B1F(24-27) B1E(24-27) B1F(24-27)

          // Shuffle pattern two - right side input
          const __m512i rhs_mat_014589CD_00_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_00,
            (_MM_PERM_ENUM)221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7)
                                 // B05(4-7) B04(4-7) B05(4-7) B08(4-7) B09(4-7)
                                 // B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7)
                                 // B0D(4-7)
          const __m512i rhs_mat_2367ABEF_00_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_00,
            (_MM_PERM_ENUM)221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7)
                                 // B07(4-7) B06(4-7) B07(4-7) B0A(4-7) B0B(4-7)
                                 // B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7)
                                 // B0F(4-7)
          const __m512i rhs_mat_014589CD_01_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_01,
            (_MM_PERM_ENUM)221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15)
                                 // B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                                 // B08(12-15) B09(12-15) B08(12-15) B09(12-15)
                                 // B0C(12-15) B0D(12-15) B0C(12-15) B0D(12-15)
          const __m512i rhs_mat_2367ABEF_01_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_01,
            (_MM_PERM_ENUM)221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15)
                                 // B06(12-15) B07(12-15) B06(12-15) B07(12-15)
                                 // B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15)
                                 // B0E(12-15) B0F(12-15) B0E(12-15) B0F(12-15)
          const __m512i rhs_mat_014589CD_02_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_02,
            (_MM_PERM_ENUM)221); // B00(20-23) B01(20-23) B00(20-23) B01(20-23)
                                 // B04(20-23) B05(20-23) B04(20-23) B05(20-23)
                                 // B08(20-23) B09(20-23) B08(20-23) B09(20-23)
                                 // B0C(20-23) B0D(20-23) B0C(20-23) B0D(20-23)
          const __m512i rhs_mat_2367ABEF_02_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_02,
            (_MM_PERM_ENUM)221); // B02(20-23) B03(20-23) B02(20-23) B03(20-23)
                                 // B06(20-23) B07(20-23) B06(20-23) B07(20-23)
                                 // B0A(20-23) B0B(20-23) B0A(20-23) B0B(20-23)
                                 // B0E(20-23) B0F(20-23) B0E(20-23) B0F(20-23)
          const __m512i rhs_mat_014589CD_03_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_03,
            (_MM_PERM_ENUM)221); // B00(28-31) B01(28-31) B00(28-31) B01(28-31)
                                 // B04(28-31) B05(28-31) B04(28-31) B05(28-31)
                                 // B08(28-31) B09(28-31) B08(28-31) B09(28-31)
                                 // B0C(28-31) B0D(28-31) B0C(28-31) 0BD(28-31)
          const __m512i rhs_mat_2367ABEF_03_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_03,
            (_MM_PERM_ENUM)221); // B02(28-31) B03(28-31) B02(28-31) B03(28-31)
                                 // B06(28-31) B07(28-31) B06(28-31) B07(28-31)
                                 // B0A(28-31) B0B(28-31) B0A(28-31) B0B(28-31)
                                 // B0E(28-31) B0F(28-31) B0E(28-31) B0F(28-31)

          const __m512i rhs_mat_014589CD_10_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_10,
            (_MM_PERM_ENUM)221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7)
                                 // B15(4-7) B14(4-7) B15(4-7) B18(4-7) B19(4-7)
                                 // B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7)
                                 // B1D(4-7)
          const __m512i rhs_mat_2367ABEF_10_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_10,
            (_MM_PERM_ENUM)221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7)
                                 // B17(4-7) B16(4-7) B17(4-7) B1A(4-7) B1B(4-7)
                                 // B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7)
                                 // B1F(4-7)
          const __m512i rhs_mat_014589CD_11_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_11,
            (_MM_PERM_ENUM)221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15)
                                 // B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                                 // B18(12-15) B19(12-15) B18(12-15) B19(12-15)
                                 // B1C(12-15) B1D(12-15) B1C(12-15) B1D(12-15)
          const __m512i rhs_mat_2367ABEF_11_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_11,
            (_MM_PERM_ENUM)221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15)
                                 // B16(12-15) B17(12-15) B16(12-15) B17(12-15)
                                 // B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15)
                                 // B1E(12-15) B1F(12-15) B1E(12-15) B1F(12-15)
          const __m512i rhs_mat_014589CD_12_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_12,
            (_MM_PERM_ENUM)221); // B10(20-23) B11(20-23) B10(20-23) B11(20-23)
                                 // B14(20-23) B15(20-23) B14(20-23) B15(20-23)
                                 // B18(20-23) B19(20-23) B18(20-23) B19(20-23)
                                 // B1C(20-23) B1D(20-23) B1C(20-23) B1D(20-23)
          const __m512i rhs_mat_2367ABEF_12_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_12,
            (_MM_PERM_ENUM)221); // B12(20-23) B13(20-23) B12(20-23) B13(20-23)
                                 // B16(20-23) B17(20-23) B16(20-23) B17(20-23)
                                 // B1A(20-23) B1B(20-23) B1A(20-23) B1B(20-23)
                                 // B1E(20-23) B1F(20-23) B1E(20-23) B1F(20-23)
          const __m512i rhs_mat_014589CD_13_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_13,
            (_MM_PERM_ENUM)221); // B10(28-31) B11(28-31) B10(28-31) B11(28-31)
                                 // B14(28-31) B15(28-31) B14(28-31) B15(28-31)
                                 // B18(28-31) B19(28-31) B18(28-31) B19(28-31)
                                 // B1C(28-31) B1D(28-31) B1C(28-31) B1D(28-31)
          const __m512i rhs_mat_2367ABEF_13_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_13,
            (_MM_PERM_ENUM)221); // B12(28-31) B13(28-31) B12(28-31) B13(28-31)
                                 // B16(28-31) B17(28-31) B16(28-31) B17(28-31)
                                 // B1A(28-31) B1B(28-31) B1A(28-31) B1B(28-31)
                                 // B1E(28-31) B1F(28-31) B1E(28-31) B1F(28-31)

          uint32_t utmp_00[4], utmp_01[4], utmp_10[4], utmp_11[4];

          // Scales and Mins of corresponding sub blocks from different Q4_K
          // structures are stored together The below block is for eg to extract
          // first sub block's scales and mins from different Q4_K structures
          // for the sb loop
          memcpy(utmp_00, b_ptr_0[b].scales + 24 * sb, 12);
          utmp_00[3] =
            ((utmp_00[2] >> 4) & kmask2) | (((utmp_00[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_00 = utmp_00[1] & kmask1;
          utmp_00[1] =
            (utmp_00[2] & kmask2) | (((utmp_00[0] >> 6) & kmask3) << 4);
          utmp_00[2] = uaux_00;
          utmp_00[0] &= kmask1;

          // The below block is for eg to extract second sub block's scales and
          // mins from different Q4_K structures for the sb loop
          memcpy(utmp_01, b_ptr_0[b].scales + 12 + sb * 24, 12);
          utmp_01[3] =
            ((utmp_01[2] >> 4) & kmask2) | (((utmp_01[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_01 = utmp_01[1] & kmask1;
          utmp_01[1] =
            (utmp_01[2] & kmask2) | (((utmp_01[0] >> 6) & kmask3) << 4);
          utmp_01[2] = uaux_01;
          utmp_01[0] &= kmask1;

          memcpy(utmp_10, b_ptr_1[b].scales + sb * 24, 12);
          utmp_10[3] =
            ((utmp_10[2] >> 4) & kmask2) | (((utmp_10[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_10 = utmp_10[1] & kmask1;
          utmp_10[1] =
            (utmp_10[2] & kmask2) | (((utmp_10[0] >> 6) & kmask3) << 4);
          utmp_10[2] = uaux_10;
          utmp_10[0] &= kmask1;

          // The below block is for eg to extract second sub block's scales and
          // mins from different Q4_K structures for the sb loop
          memcpy(utmp_11, b_ptr_1[b].scales + 12 + sb * 24, 12);
          utmp_11[3] =
            ((utmp_11[2] >> 4) & kmask2) | (((utmp_11[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_11 = utmp_11[1] & kmask1;
          utmp_11[1] =
            (utmp_11[2] & kmask2) | (((utmp_11[0] >> 6) & kmask3) << 4);
          utmp_11[2] = uaux_11;
          utmp_11[0] &= kmask1;

          // Scales of first sub block in the sb loop
          const __m256i mins_and_scales_0 =
            _mm256_set_epi32(utmp_10[3], utmp_10[2], utmp_10[1], utmp_10[0],
                             utmp_00[3], utmp_00[2], utmp_00[1], utmp_00[0]);
          const __m512i scales_0 = _mm512_cvtepu8_epi16(
            _mm256_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

          // Scales of second sub block in the sb loop
          const __m256i mins_and_scales_1 =
            _mm256_set_epi32(utmp_11[3], utmp_11[2], utmp_11[1], utmp_11[0],
                             utmp_01[3], utmp_01[2], utmp_01[1], utmp_01[0]);
          const __m512i scales_1 = _mm512_cvtepu8_epi16(
            _mm256_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

          // Mins of first and second sub block of Q4_K block are arranged side
          // by side
          const __m512i mins_01 = _mm512_cvtepu8_epi16(
            _mm256_unpacklo_epi8(_mm256_shuffle_epi32(mins_and_scales_0, 78),
                                 _mm256_shuffle_epi32(mins_and_scales_1, 78)));

          const __m512i scale_014589CD_0 =
            _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)68);
          const __m512i scale_2367ABEF_0 =
            _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)238);

          const __m512i scale_014589CD_1 =
            _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)68);
          const __m512i scale_2367ABEF_1 =
            _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)238);

          for (int rp = 0; rp < 4; rp++) {

            // Load the four block_q8_k quantized values interleaved with each
            // other in chunks of eight bytes - A0,A1,A2,A3 Loaded as set of 128
            // bit vectors and repeated and stored into a 256 bit vector before
            // again repeating into 512 bit vector
            __m256i lhs_mat_ymm_0123_00 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 256 * sb)));
            __m256i lhs_mat_ymm_01_00 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
            __m256i lhs_mat_ymm_23_00 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
            __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 32 + 256 * sb)));
            __m256i lhs_mat_ymm_01_01 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
            __m256i lhs_mat_ymm_23_01 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
            __m256i lhs_mat_ymm_0123_02 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 64 + 256 * sb)));
            __m256i lhs_mat_ymm_01_02 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 0);
            __m256i lhs_mat_ymm_23_02 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 17);
            __m256i lhs_mat_ymm_0123_03 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 96 + 256 * sb)));
            __m256i lhs_mat_ymm_01_03 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 0);
            __m256i lhs_mat_ymm_23_03 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 17);
            __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 128 + 256 * sb)));
            __m256i lhs_mat_ymm_01_10 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
            __m256i lhs_mat_ymm_23_10 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
            __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 160 + 256 * sb)));
            __m256i lhs_mat_ymm_01_11 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
            __m256i lhs_mat_ymm_23_11 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
            __m256i lhs_mat_ymm_0123_12 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 192 + 256 * sb)));
            __m256i lhs_mat_ymm_01_12 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 0);
            __m256i lhs_mat_ymm_23_12 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 17);
            __m256i lhs_mat_ymm_0123_13 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 224 + 256 * sb)));
            __m256i lhs_mat_ymm_01_13 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 0);
            __m256i lhs_mat_ymm_23_13 = _mm256_permute2f128_si256(
              lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 17);

            __m512i lhs_mat_01_00 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
            __m512i lhs_mat_23_00 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
            __m512i lhs_mat_01_01 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
            __m512i lhs_mat_23_01 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);
            __m512i lhs_mat_01_02 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_01_02), lhs_mat_ymm_01_02, 1);
            __m512i lhs_mat_23_02 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_23_02), lhs_mat_ymm_23_02, 1);
            __m512i lhs_mat_01_03 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_01_03), lhs_mat_ymm_01_03, 1);
            __m512i lhs_mat_23_03 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_23_03), lhs_mat_ymm_23_03, 1);

            __m512i lhs_mat_01_10 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
            __m512i lhs_mat_23_10 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
            __m512i lhs_mat_01_11 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
            __m512i lhs_mat_23_11 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);
            __m512i lhs_mat_01_12 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_01_12), lhs_mat_ymm_01_12, 1);
            __m512i lhs_mat_23_12 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_23_12), lhs_mat_ymm_23_12, 1);
            __m512i lhs_mat_01_13 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_01_13), lhs_mat_ymm_01_13, 1);
            __m512i lhs_mat_23_13 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_mat_ymm_23_13), lhs_mat_ymm_23_13, 1);

            // Bsums are loaded - four bsums are loaded (for two sub blocks) for
            // the different Q8_K blocks
            __m256i lhs_bsums_0123_01 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].bsums + 16 * sb)));
            __m256i lhs_bsums_hsum_ymm_0123_01 = _mm256_castsi128_si256(
              _mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01),
                             _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
            lhs_bsums_hsum_ymm_0123_01 = _mm256_permute2x128_si256(
              lhs_bsums_hsum_ymm_0123_01, lhs_bsums_hsum_ymm_0123_01, 0);
            __m512i lhs_bsums_hsum_0123_01 = _mm512_inserti32x8(
              _mm512_castsi256_si512(lhs_bsums_hsum_ymm_0123_01),
              lhs_bsums_hsum_ymm_0123_01, 1);

            // Shuffle pattern one - left side input
            const __m512i lhs_mat_01_00_sp1 = _mm512_shuffle_epi32(
              lhs_mat_01_00,
              (_MM_PERM_ENUM)160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                                   // A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                                   // A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                                   // A00(0-3) A00(0-3) A01(0-3) A01(0-3)
            const __m512i lhs_mat_23_00_sp1 = _mm512_shuffle_epi32(
              lhs_mat_23_00,
              (_MM_PERM_ENUM)160); // A02(0-3) A02(0-3) A03(0-3) A03(0-3)
                                   // A02(0-3) A02(0-3) A03(0-3) A03(0-3)
                                   // A02(0-3) A02(0-3) A03(0-3) A03(0-3)
                                   // A02(0-3) A02(0-3) A03(0-3) A03(0-3)
            const __m512i lhs_mat_01_01_sp1 = _mm512_shuffle_epi32(
              lhs_mat_01_01,
              (_MM_PERM_ENUM)160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                   // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                   // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                   // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
            const __m512i lhs_mat_23_01_sp1 = _mm512_shuffle_epi32(
              lhs_mat_23_01,
              (_MM_PERM_ENUM)160); // A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                                   // A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                                   // A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                                   // A02(8-11) A02(8-11) A03(8-11) A03(8-11)
            const __m512i lhs_mat_01_02_sp1 = _mm512_shuffle_epi32(
              lhs_mat_01_02,
              (_MM_PERM_ENUM)160); // A00(16-19) A00(16-19) A01(16-19)
                                   // A01(16-19) A00(16-19) A00(16-19)
                                   // A01(16-19) A01(16-19) A00(16-19)
                                   // A00(16-19) A01(16-19) A01(16-19)
                                   // A00(16-19) A00(16-19) A01(16-19)
                                   // A01(16-19)
            const __m512i lhs_mat_23_02_sp1 = _mm512_shuffle_epi32(
              lhs_mat_23_02,
              (_MM_PERM_ENUM)160); // A02(16-19) A02(16-19) A03(16-19)
                                   // A03(16-19) A02(16-19) A02(16-19)
                                   // A03(16-19) A03(16-19) A02(16-19)
                                   // A02(16-19) A03(16-19) A03(16-19)
                                   // A02(16-19) A02(16-19) A03(16-19)
                                   // A03(16-19)
            const __m512i lhs_mat_01_03_sp1 = _mm512_shuffle_epi32(
              lhs_mat_01_03,
              (_MM_PERM_ENUM)160); // A00(24-27) A00(24-27) A01(24-27)
                                   // A01(24-27) A00(24-27) A00(24-27)
                                   // A01(24-27) A01(24-27) A00(24-27)
                                   // A00(24-27) A01(24-27) A01(24-27)
                                   // A00(24-27) A00(24-27) A01(24-27)
                                   // A01(24-27)
            const __m512i lhs_mat_23_03_sp1 = _mm512_shuffle_epi32(
              lhs_mat_23_03,
              (_MM_PERM_ENUM)160); // A02(24-27) A02(24-27) A03(24-27)
                                   // A03(24-27) A02(24-27) A02(24-27)
                                   // A03(24-27) A03(24-27) A02(24-27)
                                   // A02(24-27) A03(24-27) A03(24-27)
                                   // A02(24-27) A02(24-27) A03(24-27)
                                   // A03(24-27)

            const __m512i lhs_mat_01_10_sp1 = _mm512_shuffle_epi32(
              lhs_mat_01_10,
              (_MM_PERM_ENUM)160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                                   // A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                                   // A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                                   // A10(0-3) A10(0-3) A11(0-3) A11(0-3)
            const __m512i lhs_mat_23_10_sp1 = _mm512_shuffle_epi32(
              lhs_mat_23_10,
              (_MM_PERM_ENUM)160); // A12(0-3) A12(0-3) A13(0-3) A13(0-3)
                                   // A12(0-3) A12(0-3) A13(0-3) A13(0-3)
                                   // A12(0-3) A12(0-3) A13(0-3) A13(0-3)
                                   // A12(0-3) A12(0-3) A13(0-3) A13(0-3)
            const __m512i lhs_mat_01_11_sp1 = _mm512_shuffle_epi32(
              lhs_mat_01_11,
              (_MM_PERM_ENUM)160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                   // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                   // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                   // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
            const __m512i lhs_mat_23_11_sp1 = _mm512_shuffle_epi32(
              lhs_mat_23_11,
              (_MM_PERM_ENUM)160); // A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                                   // A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                                   // A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                                   // A12(8-11) A12(8-11) A13(8-11) A13(8-11)
            const __m512i lhs_mat_01_12_sp1 = _mm512_shuffle_epi32(
              lhs_mat_01_12,
              (_MM_PERM_ENUM)160); // A10(16-19) A10(16-19) A11(16-19)
                                   // A11(16-19) A10(16-19) A10(16-19)
                                   // A11(16-19) A11(16-19) A10(16-19)
                                   // A10(16-19) A11(16-19) A11(16-19)
                                   // A10(16-19) A10(16-19) A11(16-19)
                                   // A11(16-19)
            const __m512i lhs_mat_23_12_sp1 = _mm512_shuffle_epi32(
              lhs_mat_23_12,
              (_MM_PERM_ENUM)160); // A12(16-19) A12(16-19) A13(16-19)
                                   // A13(16-19) A12(16-19) A12(16-19)
                                   // A13(16-19) A13(16-19) A12(16-19)
                                   // A12(16-19) A13(16-19) A13(16-19)
                                   // A12(16-19) A12(16-19) A13(16-19)
                                   // A13(16-19)
            const __m512i lhs_mat_01_13_sp1 = _mm512_shuffle_epi32(
              lhs_mat_01_13,
              (_MM_PERM_ENUM)160); // A10(24-27) A10(24-27) A11(24-27)
                                   // A11(24-27) A10(24-27) A10(24-27)
                                   // A11(24-27) A11(24-27) A10(24-27)
                                   // A10(24-27) A11(24-27) A11(24-27)
                                   // A10(24-27) A10(24-27) A11(24-27)
                                   // A11(24-27)
            const __m512i lhs_mat_23_13_sp1 = _mm512_shuffle_epi32(
              lhs_mat_23_13,
              (_MM_PERM_ENUM)160); // A12(24-27) A12(24-27) A13(24-27)
                                   // A13(24-27) A12(24-27) A12(24-27)
                                   // A13(24-27) A13(24-27) A12(24-27)
                                   // A12(24-27) A13(24-27) A13(24-27)
                                   // A12(24-27) A12(24-27) A13(24-27)
                                   // A13(24-27)

            const __m512i lhs_mat_01_00_sp2 = _mm512_shuffle_epi32(
              lhs_mat_01_00,
              (_MM_PERM_ENUM)245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                                   // A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                                   // A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                                   // A00(4-7) A00(4-7) A01(4-7) A01(4-7)
            const __m512i lhs_mat_23_00_sp2 = _mm512_shuffle_epi32(
              lhs_mat_23_00,
              (_MM_PERM_ENUM)245); // A02(4-7) A02(4-7) A03(4-7) A03(4-7)
                                   // A02(4-7) A02(4-7) A03(4-7) A03(4-7)
                                   // A02(4-7) A02(4-7) A03(4-7) A03(4-7)
                                   // A02(4-7) A02(4-7) A03(4-7) A03(4-7)
            const __m512i lhs_mat_01_01_sp2 = _mm512_shuffle_epi32(
              lhs_mat_01_01,
              (_MM_PERM_ENUM)245); // A00(12-15) A00(12-15) A01(12-15)
                                   // A01(12-15) A00(12-15) A00(12-15)
                                   // A01(12-15) A01(12-15) A00(12-15)
                                   // A00(12-15) A01(12-15) A01(12-15)
                                   // A00(12-15) A00(12-15) A01(12-15)
                                   // A01(12-15)
            const __m512i lhs_mat_23_01_sp2 = _mm512_shuffle_epi32(
              lhs_mat_23_01,
              (_MM_PERM_ENUM)245); // A02(12-15) A02(12-15) A03(12-15)
                                   // A03(12-15) A02(12-15) A02(12-15)
                                   // A03(12-15) A03(12-15) A02(12-15)
                                   // A02(12-15) A03(12-15) A03(12-15)
                                   // A02(12-15) A02(12-15) A03(12-15)
                                   // A03(12-15)
            const __m512i lhs_mat_01_02_sp2 = _mm512_shuffle_epi32(
              lhs_mat_01_02,
              (_MM_PERM_ENUM)245); // A00(20-23) A00(20-23) A01(20-23)
                                   // A01(20-23) A00(20-23) A00(20-23)
                                   // A01(20-23) A01(20-23) A00(20-23)
                                   // A00(20-23) A01(20-23) A01(20-23)
                                   // A00(20-23) A00(20-23) A01(20-23)
                                   // A01(20-23)
            const __m512i lhs_mat_23_02_sp2 = _mm512_shuffle_epi32(
              lhs_mat_23_02,
              (_MM_PERM_ENUM)245); // A02(20-23) A02(20-23) A03(20-23)
                                   // A03(20-23) A02(20-23) A02(20-23)
                                   // A03(20-23) A03(20-23) A02(20-23)
                                   // A02(20-23) A03(20-23) A03(20-23)
                                   // A02(20-23) A02(20-23) A03(20-23)
                                   // A03(20-23)
            const __m512i lhs_mat_01_03_sp2 = _mm512_shuffle_epi32(
              lhs_mat_01_03,
              (_MM_PERM_ENUM)245); // A00(28-31) A00(28-31) A01(28-31)
                                   // A01(28-31) A00(28-31) A00(28-31)
                                   // A01(28-31) A01(28-31) A00(28-31)
                                   // A00(28-31) A01(28-31) A01(28-31)
                                   // A00(28-31) A00(28-31) A01(28-31)
                                   // A01(28-31)
            const __m512i lhs_mat_23_03_sp2 = _mm512_shuffle_epi32(
              lhs_mat_23_03,
              (_MM_PERM_ENUM)245); // A02(28-31) A02(28-31) A03(28-31)
                                   // A03(28-31) A02(28-31) A02(28-31)
                                   // A03(28-31) A03(28-31) A02(28-31)
                                   // A02(28-31) A03(28-31) A03(28-31)
                                   // A02(28-31) A02(28-31) A03(28-31)
                                   // A03(28-31)

            const __m512i lhs_mat_01_10_sp2 = _mm512_shuffle_epi32(
              lhs_mat_01_10,
              (_MM_PERM_ENUM)245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                                   // A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                                   // A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                                   // A10(4-7) A10(4-7) A11(4-7) A11(4-7)
            const __m512i lhs_mat_23_10_sp2 = _mm512_shuffle_epi32(
              lhs_mat_23_10,
              (_MM_PERM_ENUM)245); // A12(4-7) A12(4-7) A13(4-7) A13(4-7)
                                   // A12(4-7) A12(4-7) A13(4-7) A13(4-7)
                                   // A12(4-7) A12(4-7) A13(4-7) A13(4-7)
                                   // A12(4-7) A12(4-7) A13(4-7) A13(4-7)
            const __m512i lhs_mat_01_11_sp2 = _mm512_shuffle_epi32(
              lhs_mat_01_11,
              (_MM_PERM_ENUM)245); // A10(12-15) A10(12-15) A11(12-15)
                                   // A11(12-15) A10(12-15) A10(12-15)
                                   // A11(12-15) A11(12-15) A10(12-15)
                                   // A10(12-15) A11(12-15) A11(12-15)
                                   // A10(12-15) A10(12-15) A11(12-15)
                                   // A11(12-15)
            const __m512i lhs_mat_23_11_sp2 = _mm512_shuffle_epi32(
              lhs_mat_23_11,
              (_MM_PERM_ENUM)245); // A12(12-15) A12(12-15) A13(12-15)
                                   // A13(12-15) A12(12-15) A12(12-15)
                                   // A13(12-15) A13(12-15) A12(12-15)
                                   // A12(12-15) A13(12-15) A13(12-15)
                                   // A12(12-15) A12(12-15) A13(12-15)
                                   // A13(12-15)
            const __m512i lhs_mat_01_12_sp2 = _mm512_shuffle_epi32(
              lhs_mat_01_12,
              (_MM_PERM_ENUM)245); // A10(20-23) A10(20-23) A11(20-23)
                                   // A11(20-23) A10(20-23) A10(20-23)
                                   // A11(20-23) A11(20-23) A10(20-23)
                                   // A10(20-23) A11(20-23) A11(20-23)
                                   // A10(20-23) A10(20-23) A11(20-23)
                                   // A11(20-23)
            const __m512i lhs_mat_23_12_sp2 = _mm512_shuffle_epi32(
              lhs_mat_23_12,
              (_MM_PERM_ENUM)245); // A12(20-23) A12(20-23) A13(20-23)
                                   // A13(20-23) A12(20-23) A12(20-23)
                                   // A13(20-23) A13(20-23) A12(20-23)
                                   // A12(20-23) A13(20-23) A13(20-23)
                                   // A12(20-23) A12(20-23) A13(20-23)
                                   // A13(20-23)
            const __m512i lhs_mat_01_13_sp2 = _mm512_shuffle_epi32(
              lhs_mat_01_13,
              (_MM_PERM_ENUM)245); // A10(28-31) A10(28-31) A11(28-31)
                                   // A11(28-31) A10(28-31) A10(28-31)
                                   // A11(28-31) A11(28-31) A10(28-31)
                                   // A10(28-31) A11(28-31) A11(28-31)
                                   // A10(28-31) A10(28-31) A11(28-31)
                                   // A11(28-31)
            const __m512i lhs_mat_23_13_sp2 = _mm512_shuffle_epi32(
              lhs_mat_23_13,
              (_MM_PERM_ENUM)245); // A12(28-31) A12(28-31) A13(28-31)
                                   // A13(28-31) A12(28-31) A12(28-31)
                                   // A13(28-31) A13(28-31) A12(28-31)
                                   // A12(28-31) A13(28-31) A13(28-31)
                                   // A12(28-31) A12(28-31) A13(28-31)
                                   // A13(28-31)

            // The values arranged in shuffle patterns are operated with dot
            // product operation within 32 bit lane i.e corresponding bytes and
            // multiplied and added into 32 bit integers within 32 bit lane
            __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1,
                                                      lhs_mat_01_03_sp1),
                                 _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1,
                                                      lhs_mat_01_02_sp1)),
                _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1,
                                     lhs_mat_01_01_sp1)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1));
            __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1,
                                                      lhs_mat_01_03_sp1),
                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1,
                                                      lhs_mat_01_02_sp1)),
                _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1,
                                     lhs_mat_01_01_sp1)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1));
            __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1,
                                                      lhs_mat_23_03_sp1),
                                 _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1,
                                                      lhs_mat_23_02_sp1)),
                _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1,
                                     lhs_mat_23_01_sp1)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1));
            __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1,
                                                      lhs_mat_23_03_sp1),
                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1,
                                                      lhs_mat_23_02_sp1)),
                _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1,
                                     lhs_mat_23_01_sp1)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1));
            __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1,
                                                      lhs_mat_01_13_sp1),
                                 _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1,
                                                      lhs_mat_01_12_sp1)),
                _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1,
                                     lhs_mat_01_11_sp1)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1));
            __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1,
                                                      lhs_mat_01_13_sp1),
                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1,
                                                      lhs_mat_01_12_sp1)),
                _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1,
                                     lhs_mat_01_11_sp1)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1));
            __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1,
                                                      lhs_mat_23_13_sp1),
                                 _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1,
                                                      lhs_mat_23_12_sp1)),
                _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1,
                                     lhs_mat_23_11_sp1)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1));
            __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1,
                                                      lhs_mat_23_13_sp1),
                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1,
                                                      lhs_mat_23_12_sp1)),
                _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1,
                                     lhs_mat_23_11_sp1)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1));

            __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2,
                                                      lhs_mat_01_03_sp2),
                                 _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2,
                                                      lhs_mat_01_02_sp2)),
                _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2,
                                     lhs_mat_01_01_sp2)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2));
            __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2,
                                                      lhs_mat_01_03_sp2),
                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2,
                                                      lhs_mat_01_02_sp2)),
                _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2,
                                     lhs_mat_01_01_sp2)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2));
            __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2,
                                                      lhs_mat_23_03_sp2),
                                 _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2,
                                                      lhs_mat_23_02_sp2)),
                _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2,
                                     lhs_mat_23_01_sp2)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2));
            __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2,
                                                      lhs_mat_23_03_sp2),
                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2,
                                                      lhs_mat_23_02_sp2)),
                _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2,
                                     lhs_mat_23_01_sp2)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2));
            __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2,
                                                      lhs_mat_01_13_sp2),
                                 _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2,
                                                      lhs_mat_01_12_sp2)),
                _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2,
                                     lhs_mat_01_11_sp2)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2));
            __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2,
                                                      lhs_mat_01_13_sp2),
                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2,
                                                      lhs_mat_01_12_sp2)),
                _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2,
                                     lhs_mat_01_11_sp2)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2));
            __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2,
                                                      lhs_mat_23_13_sp2),
                                 _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2,
                                                      lhs_mat_23_12_sp2)),
                _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2,
                                     lhs_mat_23_11_sp2)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2));
            __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(
              _mm512_add_epi16(
                _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2,
                                                      lhs_mat_23_13_sp2),
                                 _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2,
                                                      lhs_mat_23_12_sp2)),
                _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2,
                                     lhs_mat_23_11_sp2)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2));

            // Output of both shuffle patterns are added in order to sum dot
            // product outputs of all 32 values in block
            __m512i iacc_mat_00_0 =
              _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
            __m512i iacc_mat_01_0 =
              _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
            __m512i iacc_mat_10_0 =
              _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
            __m512i iacc_mat_11_0 =
              _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

            __m512i iacc_mat_00_1 =
              _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
            __m512i iacc_mat_01_1 =
              _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
            __m512i iacc_mat_10_1 =
              _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
            __m512i iacc_mat_11_1 =
              _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

            iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
            iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
            iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
            iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

            iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
            iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
            iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
            iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

            // Straighten out to make 4 row vectors (4 for each sub block which
            // are accumulated together in the next step)
            __m512i iacc_row_0_0 = _mm512_mask_blend_epi32(
              0xCCCC, iacc_mat_00_0,
              _mm512_shuffle_epi32(iacc_mat_01_0, (_MM_PERM_ENUM)78));
            __m512i iacc_row_1_0 = _mm512_mask_blend_epi32(
              0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_0, (_MM_PERM_ENUM)78),
              iacc_mat_01_0);
            __m512i iacc_row_2_0 = _mm512_mask_blend_epi32(
              0xCCCC, iacc_mat_10_0,
              _mm512_shuffle_epi32(iacc_mat_11_0, (_MM_PERM_ENUM)78));
            __m512i iacc_row_3_0 = _mm512_mask_blend_epi32(
              0xCCCC, _mm512_shuffle_epi32(iacc_mat_10_0, (_MM_PERM_ENUM)78),
              iacc_mat_11_0);
            __m512i iacc_row_0_1 = _mm512_mask_blend_epi32(
              0xCCCC, iacc_mat_00_1,
              _mm512_shuffle_epi32(iacc_mat_01_1, (_MM_PERM_ENUM)78));
            __m512i iacc_row_1_1 = _mm512_mask_blend_epi32(
              0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_1, (_MM_PERM_ENUM)78),
              iacc_mat_01_1);
            __m512i iacc_row_2_1 = _mm512_mask_blend_epi32(
              0xCCCC, iacc_mat_10_1,
              _mm512_shuffle_epi32(iacc_mat_11_1, (_MM_PERM_ENUM)78));
            __m512i iacc_row_3_1 = _mm512_mask_blend_epi32(
              0xCCCC, _mm512_shuffle_epi32(iacc_mat_10_1, (_MM_PERM_ENUM)78),
              iacc_mat_11_1);

            __m512i iacc_row_0 = _mm512_add_epi32(iacc_row_0_0, iacc_row_0_1);
            __m512i iacc_row_1 = _mm512_add_epi32(iacc_row_1_0, iacc_row_1_1);
            __m512i iacc_row_2 = _mm512_add_epi32(iacc_row_2_0, iacc_row_2_1);
            __m512i iacc_row_3 = _mm512_add_epi32(iacc_row_3_0, iacc_row_3_1);

            // Load the scale(d) values for all the 4 Q8_k blocks and repeat it
            // across lanes
            const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
            const __m256 row_scale_f32_ymm =
              _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
            const __m512 row_scale_f32 = _mm512_insertf32x8(
              _mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

            // Multiply with appropiate scales and accumulate (for both d and
            // dmin) below
            acc_rows[rp * 4] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(iacc_row_0),
              _mm512_mul_ps(col_scale_f32,
                            _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
              acc_rows[rp * 4]);
            acc_rows[rp * 4 + 1] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(iacc_row_1),
              _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(
                                             row_scale_f32, row_scale_f32, 85)),
              acc_rows[rp * 4 + 1]);
            acc_rows[rp * 4 + 2] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(iacc_row_2),
              _mm512_mul_ps(
                col_scale_f32,
                _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
              acc_rows[rp * 4 + 2]);
            acc_rows[rp * 4 + 3] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(iacc_row_3),
              _mm512_mul_ps(
                col_scale_f32,
                _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
              acc_rows[rp * 4 + 3]);

            __m512i iacc_row_min_0 = _mm512_madd_epi16(
              _mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)0),
              mins_01);
            __m512i iacc_row_min_1 = _mm512_madd_epi16(
              _mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)85),
              mins_01);
            __m512i iacc_row_min_2 = _mm512_madd_epi16(
              _mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)170),
              mins_01);
            __m512i iacc_row_min_3 = _mm512_madd_epi16(
              _mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)255),
              mins_01);

            acc_min_rows[rp * 4] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(iacc_row_min_0),
              _mm512_mul_ps(col_dmin_f32,
                            _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
              acc_min_rows[rp * 4]);
            acc_min_rows[rp * 4 + 1] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(iacc_row_min_1),
              _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32,
                                                            row_scale_f32, 85)),
              acc_min_rows[rp * 4 + 1]);
            acc_min_rows[rp * 4 + 2] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(iacc_row_min_2),
              _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(
                                            row_scale_f32, row_scale_f32, 170)),
              acc_min_rows[rp * 4 + 2]);
            acc_min_rows[rp * 4 + 3] = _mm512_fmadd_ps(
              _mm512_cvtepi32_ps(iacc_row_min_3),
              _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(
                                            row_scale_f32, row_scale_f32, 255)),
              acc_min_rows[rp * 4 + 3]);
          }
        }
      }
      // Store the accumulated values
      for (int i = 0; i < 16; i++) {
        _mm512_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                         _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
      }
    }
  }

  for (; y < nr / 4; y++) {

    const block_q8_Kx4 *a_ptr = a_ptr_start + (y * nb);

    // Take group of eight block_q4_kx8 structures at each pass of the loop and
    // perform dot product operation
    for (int64_t x = 0; x < anc / 8; x += 2) {

      const block_q4_Kx8 *b_ptr_0 = b_ptr_start + ((x)*b_nb);
      const block_q4_Kx8 *b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

      // Master FP accumulators
      __m512 acc_rows[4];
      for (int i = 0; i < 4; i++) {
        acc_rows[i] = _mm512_setzero_ps();
      }

      __m512 acc_min_rows[4];
      for (int i = 0; i < 4; i++) {
        acc_min_rows[i] = _mm512_setzero_ps();
      }

      // For super block
      for (int64_t b = 0; b < nb; b++) {
        // Scale values - Load the sixteen scale values from two block_q4_kx8
        // structures
        const __m512 col_scale_f32 =
          GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

        // dmin values - Load the sixteen dmin values from two block_q4_kx8
        // structures
        const __m512 col_dmin_f32 =
          GGML_F32Cx8x2_LOAD(b_ptr_0[b].dmin, b_ptr_1[b].dmin);

        // Loop to iterate over the eight sub blocks of a super block - two sub
        // blocks are processed per iteration
        for (int sb = 0; sb < QK_K / 64; sb++) {

          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + sb * 256));
          const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 32 + sb * 256));
          const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 64 + sb * 256));
          const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 96 + sb * 256));
          const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 128 + sb * 256));
          const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 160 + sb * 256));
          const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 192 + sb * 256));
          const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_0[b].qs + 224 + sb * 256));

          const __m256i rhs_raw_mat_89AB_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + sb * 256));
          const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 32 + sb * 256));
          const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 64 + sb * 256));
          const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 96 + sb * 256));
          const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 128 + sb * 256));
          const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 160 + sb * 256));
          const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 192 + sb * 256));
          const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256(
            (const __m256i *)(b_ptr_1[b].qs + 224 + sb * 256));

          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);
          const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(
            rhs_raw_mat_0123_2,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder),
            rhs_raw_mat_4567_2, 240);
          const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(
            rhs_raw_mat_0123_3,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder),
            rhs_raw_mat_4567_3, 240);

          const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(
            rhs_raw_mat_89AB_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder),
            rhs_raw_mat_CDEF_0, 240);
          const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(
            rhs_raw_mat_89AB_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder),
            rhs_raw_mat_CDEF_1, 240);
          const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(
            rhs_raw_mat_89AB_2,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder),
            240);
          const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder),
            rhs_raw_mat_CDEF_2, 240);
          const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(
            rhs_raw_mat_89AB_3,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder),
            240);
          const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder),
            rhs_raw_mat_CDEF_3, 240);

          const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
          const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
          const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
          const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

          const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
          const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
          const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
          const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(
            _mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

          // 4-bit -> 8-bit
          const __m512i rhs_mat_014589CD_00 = _mm512_and_si512(
            rhs_raw_mat_014589CD_0,
            m4bexpanded); // B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7)
                          // B09(0-7) B0C(0-7) B0D(0-7)
          const __m512i rhs_mat_2367ABEF_00 = _mm512_and_si512(
            rhs_raw_mat_2367ABEF_0,
            m4bexpanded); // B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7)
                          // B0B(0-7) B0E(0-7) B0F(0-7)
          const __m512i rhs_mat_014589CD_01 = _mm512_and_si512(
            rhs_raw_mat_014589CD_1,
            m4bexpanded); // B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15)
                          // B09(8-15) B0C(8-15) B0D(8-15)
          const __m512i rhs_mat_2367ABEF_01 = _mm512_and_si512(
            rhs_raw_mat_2367ABEF_1,
            m4bexpanded); // B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15)
                          // B0B(8-15) B0E(8-15) B0F(8-15)

          const __m512i rhs_mat_014589CD_02 = _mm512_and_si512(
            rhs_raw_mat_014589CD_2,
            m4bexpanded); // B00(16-23) B01(16-23) B04(16-23) B05(16-23)
                          // B08(16-23) B09(16-23) B0C(16-23) B0D(16-23)
          const __m512i rhs_mat_2367ABEF_02 = _mm512_and_si512(
            rhs_raw_mat_2367ABEF_2,
            m4bexpanded); // B02(16-23) B03(16-23) B06(16-23) B07(16-23)
                          // B0A(16-23) B0B(16-23) B0E(16-23) B0F(16-23)
          const __m512i rhs_mat_014589CD_03 = _mm512_and_si512(
            rhs_raw_mat_014589CD_3,
            m4bexpanded); // B00(24-31) B01(24-31) B04(24-31) B05(24-31)
                          // B08(24-31) B09(24-31) B0C(24-31) B0D(24-31)
          const __m512i rhs_mat_2367ABEF_03 = _mm512_and_si512(
            rhs_raw_mat_2367ABEF_3,
            m4bexpanded); // B02(24-31) B03(24-31) B06(24-31) B07(24-31)
                          // B0A(24-31) B0B(24-31) B0E(24-31) B0F(24-31)

          const __m512i rhs_mat_014589CD_10 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4),
            m4bexpanded); // B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7)
                          // B19(0-7) B1C(0-7) B1D(0-7)
          const __m512i rhs_mat_2367ABEF_10 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4),
            m4bexpanded); // B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7)
                          // B1B(0-7) B1E(0-7) B1F(0-7)
          const __m512i rhs_mat_014589CD_11 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4),
            m4bexpanded); // B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15)
                          // B19(8-15) B1C(8-15) B1D(8-15)
          const __m512i rhs_mat_2367ABEF_11 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4),
            m4bexpanded); // B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15)
                          // B1B(8-15) B1E(8-15) B1F(8-15)

          const __m512i rhs_mat_014589CD_12 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4),
            m4bexpanded); // B10(16-23) B11(16-23) B14(16-23) B15(16-23)
                          // B18(16-23) B19(16-23) B1C(16-23) B1D(16-23)
          const __m512i rhs_mat_2367ABEF_12 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4),
            m4bexpanded); // B12(16-23) B13(16-23) B16(16-23) B17(16-23)
                          // B1A(16-23) B1B(16-23) B1E(16-23) B1F(16-23)
          const __m512i rhs_mat_014589CD_13 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4),
            m4bexpanded); // B10(24-31) B11(24-31) B14(24-31) B15(24-31)
                          // B18(24-31) B19(24-31) B1C(24-31) B1D(24-31)
          const __m512i rhs_mat_2367ABEF_13 = _mm512_and_si512(
            _mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4),
            m4bexpanded); // B12(24-31) B13(24-31) B16(24-31) B17(24-31)
                          // B1A(24-31) B1B(24-31) B1E(24-31) B1F(24-31)

          // Shuffle pattern one - right side input
          const __m512i rhs_mat_014589CD_00_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_00,
            (_MM_PERM_ENUM)136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3)
                                 // B05(0-3) B04(0-3) B05(0-3) B08(0-3) B09(0-3)
                                 // B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3)
                                 // B0D(0-3)
          const __m512i rhs_mat_2367ABEF_00_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_00,
            (_MM_PERM_ENUM)136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3)
                                 // B07(0-3) B06(0-3) B07(0-3) B0A(0-3) B0B(0-3)
                                 // B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3)
                                 // B0F(0-3)
          const __m512i rhs_mat_014589CD_01_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_01,
            (_MM_PERM_ENUM)136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11)
                                 // B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                                 // B08(8-11) B09(8-11) B08(8-11) B09(8-11)
                                 // B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
          const __m512i rhs_mat_2367ABEF_01_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_01,
            (_MM_PERM_ENUM)136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11)
                                 // B06(8-11) B07(8-11) B06(8-11) B07(8-11)
                                 // B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11)
                                 // B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)
          const __m512i rhs_mat_014589CD_02_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_02,
            (_MM_PERM_ENUM)136); // B00(16-19) B01(16-19) B00(16-19) B01(16-19)
                                 // B04(16-19) B05(16-19) B04(16-19) B05(16-19)
                                 // B08(16-19) B09(16-19) B08(16-19) B09(16-19)
                                 // B0C(16-19) B0D(16-19) B0C(16-19) B0D(16-19)
          const __m512i rhs_mat_2367ABEF_02_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_02,
            (_MM_PERM_ENUM)136); // B02(16-19) B03(16-19) B02(16-19) B03(16-19)
                                 // B06(16-19) B07(16-19) B06(16-19) B07(16-19)
                                 // B0A(16-19) B0B(16-19) B0A(16-19) B0B(16-19)
                                 // B0E(16-19) B0F(16-19) B0E(16-19) B0F(16-19)
          const __m512i rhs_mat_014589CD_03_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_03,
            (_MM_PERM_ENUM)136); // B00(24-27) B01(24-27) B00(24-27) B01(24-27)
                                 // B04(24-27) B05(24-27) B04(24-27) B05(24-27)
                                 // B08(24-27) B09(24-27) B08(24-27) B09(24-27)
                                 // B0C(24-27) B0D(24-27) B0C(24-27) B0D(24-27)
          const __m512i rhs_mat_2367ABEF_03_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_03,
            (_MM_PERM_ENUM)136); // B02(24-27) B03(24-27) B02(24-27) B03(24-27)
                                 // B06(24-27) B07(24-27) B06(24-27) B07(24-27)
                                 // B0A(24-27) B0B(24-27) B0A(24-27) B0B(24-27)
                                 // B0E(24-27) B0F(24-27) B0E(24-27) B0F(24-27)

          const __m512i rhs_mat_014589CD_10_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_10,
            (_MM_PERM_ENUM)136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3)
                                 // B15(0-3) B14(0-3) B15(0-3) B18(0-3) B19(0-3)
                                 // B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3)
                                 // B1D(0-3)
          const __m512i rhs_mat_2367ABEF_10_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_10,
            (_MM_PERM_ENUM)136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3)
                                 // B17(0-3) B16(0-3) B17(0-3) B1A(0-3) B1B(0-3)
                                 // B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3)
                                 // B1F(0-3)
          const __m512i rhs_mat_014589CD_11_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_11,
            (_MM_PERM_ENUM)136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11)
                                 // B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                                 // B18(8-11) B19(8-11) B18(8-11) B19(8-11)
                                 // B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
          const __m512i rhs_mat_2367ABEF_11_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_11,
            (_MM_PERM_ENUM)136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11)
                                 // B16(8-11) B17(8-11) B16(8-11) B17(8-11)
                                 // B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11)
                                 // B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)
          const __m512i rhs_mat_014589CD_12_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_12,
            (_MM_PERM_ENUM)136); // B10(16-19) B11(16-19) B10(16-19) B11(16-19)
                                 // B14(16-19) B15(16-19) B14(16-19) B15(16-19)
                                 // B18(16-19) B19(16-19) B18(16-19) B19(16-19)
                                 // B1C(16-19) B1D(16-19) B1C(16-19) B1D(16-19)
          const __m512i rhs_mat_2367ABEF_12_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_12,
            (_MM_PERM_ENUM)136); // B12(16-19) B13(16-19) B12(16-19) B13(16-19)
                                 // B16(16-19) B17(16-19) B16(16-19) B17(16-19)
                                 // B1A(16-19) B1B(16-19) B1A(16-19) B1B(16-19)
                                 // B1E(16-19) B1F(16-19) B1E(16-19) B1F(16-19)
          const __m512i rhs_mat_014589CD_13_sp1 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_13,
            (_MM_PERM_ENUM)136); // B10(24-27) B11(24-27) B10(24-27) B11(24-27)
                                 // B14(24-27) B15(24-27) B14(24-27) B15(24-27)
                                 // B18(24-27) B19(24-27) B18(24-27) B19(24-27)
                                 // B1C(24-27) B1D(24-27) B1C(24-27) B1D(24-27)
          const __m512i rhs_mat_2367ABEF_13_sp1 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_13,
            (_MM_PERM_ENUM)136); // B12(24-27) B13(24-27) B12(24-27) B13(24-27)
                                 // B16(24-27) B17(24-27) B16(24-27) B17(24-27)
                                 // B1A(24-27) B1B(24-27) B1A(24-27) B1B(24-27)
                                 // B1E(24-27) B1F(24-27) B1E(24-27) B1F(24-27)

          // Shuffle pattern two - right side input
          const __m512i rhs_mat_014589CD_00_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_00,
            (_MM_PERM_ENUM)221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7)
                                 // B05(4-7) B04(4-7) B05(4-7) B08(4-7) B09(4-7)
                                 // B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7)
                                 // B0D(4-7)
          const __m512i rhs_mat_2367ABEF_00_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_00,
            (_MM_PERM_ENUM)221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7)
                                 // B07(4-7) B06(4-7) B07(4-7) B0A(4-7) B0B(4-7)
                                 // B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7)
                                 // B0F(4-7)
          const __m512i rhs_mat_014589CD_01_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_01,
            (_MM_PERM_ENUM)221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15)
                                 // B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                                 // B08(12-15) B09(12-15) B08(12-15) B09(12-15)
                                 // B0C(12-15) B0D(12-15) B0C(12-15) B0D(12-15)
          const __m512i rhs_mat_2367ABEF_01_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_01,
            (_MM_PERM_ENUM)221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15)
                                 // B06(12-15) B07(12-15) B06(12-15) B07(12-15)
                                 // B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15)
                                 // B0E(12-15) B0F(12-15) B0E(12-15) B0F(12-15)
          const __m512i rhs_mat_014589CD_02_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_02,
            (_MM_PERM_ENUM)221); // B00(20-23) B01(20-23) B00(20-23) B01(20-23)
                                 // B04(20-23) B05(20-23) B04(20-23) B05(20-23)
                                 // B08(20-23) B09(20-23) B08(20-23) B09(20-23)
                                 // B0C(20-23) B0D(20-23) B0C(20-23) B0D(20-23)
          const __m512i rhs_mat_2367ABEF_02_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_02,
            (_MM_PERM_ENUM)221); // B02(20-23) B03(20-23) B02(20-23) B03(20-23)
                                 // B06(20-23) B07(20-23) B06(20-23) B07(20-23)
                                 // B0A(20-23) B0B(20-23) B0A(20-23) B0B(20-23)
                                 // B0E(20-23) B0F(20-23) B0E(20-23) B0F(20-23)
          const __m512i rhs_mat_014589CD_03_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_03,
            (_MM_PERM_ENUM)221); // B00(28-31) B01(28-31) B00(28-31) B01(28-31)
                                 // B04(28-31) B05(28-31) B04(28-31) B05(28-31)
                                 // B08(28-31) B09(28-31) B08(28-31) B09(28-31)
                                 // B0C(28-31) B0D(28-31) B0C(28-31) 0BD(28-31)
          const __m512i rhs_mat_2367ABEF_03_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_03,
            (_MM_PERM_ENUM)221); // B02(28-31) B03(28-31) B02(28-31) B03(28-31)
                                 // B06(28-31) B07(28-31) B06(28-31) B07(28-31)
                                 // B0A(28-31) B0B(28-31) B0A(28-31) B0B(28-31)
                                 // B0E(28-31) B0F(28-31) B0E(28-31) B0F(28-31)

          const __m512i rhs_mat_014589CD_10_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_10,
            (_MM_PERM_ENUM)221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7)
                                 // B15(4-7) B14(4-7) B15(4-7) B18(4-7) B19(4-7)
                                 // B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7)
                                 // B1D(4-7)
          const __m512i rhs_mat_2367ABEF_10_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_10,
            (_MM_PERM_ENUM)221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7)
                                 // B17(4-7) B16(4-7) B17(4-7) B1A(4-7) B1B(4-7)
                                 // B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7)
                                 // B1F(4-7)
          const __m512i rhs_mat_014589CD_11_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_11,
            (_MM_PERM_ENUM)221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15)
                                 // B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                                 // B18(12-15) B19(12-15) B18(12-15) B19(12-15)
                                 // B1C(12-15) B1D(12-15) B1C(12-15) B1D(12-15)
          const __m512i rhs_mat_2367ABEF_11_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_11,
            (_MM_PERM_ENUM)221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15)
                                 // B16(12-15) B17(12-15) B16(12-15) B17(12-15)
                                 // B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15)
                                 // B1E(12-15) B1F(12-15) B1E(12-15) B1F(12-15)
          const __m512i rhs_mat_014589CD_12_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_12,
            (_MM_PERM_ENUM)221); // B10(20-23) B11(20-23) B10(20-23) B11(20-23)
                                 // B14(20-23) B15(20-23) B14(20-23) B15(20-23)
                                 // B18(20-23) B19(20-23) B18(20-23) B19(20-23)
                                 // B1C(20-23) B1D(20-23) B1C(20-23) B1D(20-23)
          const __m512i rhs_mat_2367ABEF_12_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_12,
            (_MM_PERM_ENUM)221); // B12(20-23) B13(20-23) B12(20-23) B13(20-23)
                                 // B16(20-23) B17(20-23) B16(20-23) B17(20-23)
                                 // B1A(20-23) B1B(20-23) B1A(20-23) B1B(20-23)
                                 // B1E(20-23) B1F(20-23) B1E(20-23) B1F(20-23)
          const __m512i rhs_mat_014589CD_13_sp2 = _mm512_shuffle_epi32(
            rhs_mat_014589CD_13,
            (_MM_PERM_ENUM)221); // B10(28-31) B11(28-31) B10(28-31) B11(28-31)
                                 // B14(28-31) B15(28-31) B14(28-31) B15(28-31)
                                 // B18(28-31) B19(28-31) B18(28-31) B19(28-31)
                                 // B1C(28-31) B1D(28-31) B1C(28-31) B1D(28-31)
          const __m512i rhs_mat_2367ABEF_13_sp2 = _mm512_shuffle_epi32(
            rhs_mat_2367ABEF_13,
            (_MM_PERM_ENUM)221); // B12(28-31) B13(28-31) B12(28-31) B13(28-31)
                                 // B16(28-31) B17(28-31) B16(28-31) B17(28-31)
                                 // B1A(28-31) B1B(28-31) B1A(28-31) B1B(28-31)
                                 // B1E(28-31) B1F(28-31) B1E(28-31) B1F(28-31)

          uint32_t utmp_00[4], utmp_01[4], utmp_10[4], utmp_11[4];

          // Scales and Mins of corresponding sub blocks from different Q4_K
          // structures are stored together The below block is for eg to extract
          // first sub block's scales and mins from different Q4_K structures
          // for the sb loop
          memcpy(utmp_00, b_ptr_0[b].scales + 24 * sb, 12);
          utmp_00[3] =
            ((utmp_00[2] >> 4) & kmask2) | (((utmp_00[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_00 = utmp_00[1] & kmask1;
          utmp_00[1] =
            (utmp_00[2] & kmask2) | (((utmp_00[0] >> 6) & kmask3) << 4);
          utmp_00[2] = uaux_00;
          utmp_00[0] &= kmask1;

          // The below block is for eg to extract second sub block's scales and
          // mins from different Q4_K structures for the sb loop
          memcpy(utmp_01, b_ptr_0[b].scales + 12 + sb * 24, 12);
          utmp_01[3] =
            ((utmp_01[2] >> 4) & kmask2) | (((utmp_01[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_01 = utmp_01[1] & kmask1;
          utmp_01[1] =
            (utmp_01[2] & kmask2) | (((utmp_01[0] >> 6) & kmask3) << 4);
          utmp_01[2] = uaux_01;
          utmp_01[0] &= kmask1;

          // The below block is for eg to extract first sub block's scales and
          // mins from different Q4_K structures for the sb loop
          memcpy(utmp_10, b_ptr_1[b].scales + sb * 24, 12);
          utmp_10[3] =
            ((utmp_10[2] >> 4) & kmask2) | (((utmp_10[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_10 = utmp_10[1] & kmask1;
          utmp_10[1] =
            (utmp_10[2] & kmask2) | (((utmp_10[0] >> 6) & kmask3) << 4);
          utmp_10[2] = uaux_10;
          utmp_10[0] &= kmask1;

          // The below block is for eg to extract second sub block's scales and
          // mins from different Q4_K structures for the sb loop
          memcpy(utmp_11, b_ptr_1[b].scales + 12 + sb * 24, 12);
          utmp_11[3] =
            ((utmp_11[2] >> 4) & kmask2) | (((utmp_11[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_11 = utmp_11[1] & kmask1;
          utmp_11[1] =
            (utmp_11[2] & kmask2) | (((utmp_11[0] >> 6) & kmask3) << 4);
          utmp_11[2] = uaux_11;
          utmp_11[0] &= kmask1;

          // Scales of first sub block in the sb loop
          const __m256i mins_and_scales_0 =
            _mm256_set_epi32(utmp_10[3], utmp_10[2], utmp_10[1], utmp_10[0],
                             utmp_00[3], utmp_00[2], utmp_00[1], utmp_00[0]);
          const __m512i scales_0 = _mm512_cvtepu8_epi16(
            _mm256_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

          // Scales of second sub block in the sb loop
          const __m256i mins_and_scales_1 =
            _mm256_set_epi32(utmp_11[3], utmp_11[2], utmp_11[1], utmp_11[0],
                             utmp_01[3], utmp_01[2], utmp_01[1], utmp_01[0]);
          const __m512i scales_1 = _mm512_cvtepu8_epi16(
            _mm256_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

          // Mins of first and second sub block of Q4_K block are arranged side
          // by side
          const __m512i mins_01 = _mm512_cvtepu8_epi16(
            _mm256_unpacklo_epi8(_mm256_shuffle_epi32(mins_and_scales_0, 78),
                                 _mm256_shuffle_epi32(mins_and_scales_1, 78)));

          const __m512i scale_014589CD_0 =
            _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)68);
          const __m512i scale_2367ABEF_0 =
            _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)238);

          const __m512i scale_014589CD_1 =
            _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)68);
          const __m512i scale_2367ABEF_1 =
            _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)238);

          // Load the four block_q8_k quantized values interleaved with each
          // other in chunks of eight bytes - A0,A1,A2,A3 Loaded as set of 128
          // bit vectors and repeated into a 256 bit vector
          __m256i lhs_mat_ymm_0123_00 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 256 * sb)));
          __m256i lhs_mat_ymm_01_00 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
          __m256i lhs_mat_ymm_23_00 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
          __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 32 + 256 * sb)));
          __m256i lhs_mat_ymm_01_01 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
          __m256i lhs_mat_ymm_23_01 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
          __m256i lhs_mat_ymm_0123_02 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 64 + 256 * sb)));
          __m256i lhs_mat_ymm_01_02 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 0);
          __m256i lhs_mat_ymm_23_02 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 17);
          __m256i lhs_mat_ymm_0123_03 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 96 + 256 * sb)));
          __m256i lhs_mat_ymm_01_03 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 0);
          __m256i lhs_mat_ymm_23_03 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 17);
          __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 128 + 256 * sb)));
          __m256i lhs_mat_ymm_01_10 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
          __m256i lhs_mat_ymm_23_10 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
          __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 160 + 256 * sb)));
          __m256i lhs_mat_ymm_01_11 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
          __m256i lhs_mat_ymm_23_11 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
          __m256i lhs_mat_ymm_0123_12 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 192 + 256 * sb)));
          __m256i lhs_mat_ymm_01_12 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 0);
          __m256i lhs_mat_ymm_23_12 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 17);
          __m256i lhs_mat_ymm_0123_13 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 224 + 256 * sb)));
          __m256i lhs_mat_ymm_01_13 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 0);
          __m256i lhs_mat_ymm_23_13 = _mm256_permute2f128_si256(
            lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 17);

          // Loaded as set of 128 bit vectors and repeated and stored into a 256
          // bit vector before again repeating into a 512 bit vector
          __m512i lhs_mat_01_00 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
          __m512i lhs_mat_23_00 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
          __m512i lhs_mat_01_01 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
          __m512i lhs_mat_23_01 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);
          __m512i lhs_mat_01_02 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_01_02), lhs_mat_ymm_01_02, 1);
          __m512i lhs_mat_23_02 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_23_02), lhs_mat_ymm_23_02, 1);
          __m512i lhs_mat_01_03 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_01_03), lhs_mat_ymm_01_03, 1);
          __m512i lhs_mat_23_03 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_23_03), lhs_mat_ymm_23_03, 1);

          __m512i lhs_mat_01_10 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
          __m512i lhs_mat_23_10 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
          __m512i lhs_mat_01_11 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
          __m512i lhs_mat_23_11 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);
          __m512i lhs_mat_01_12 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_01_12), lhs_mat_ymm_01_12, 1);
          __m512i lhs_mat_23_12 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_23_12), lhs_mat_ymm_23_12, 1);
          __m512i lhs_mat_01_13 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_01_13), lhs_mat_ymm_01_13, 1);
          __m512i lhs_mat_23_13 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_mat_ymm_23_13), lhs_mat_ymm_23_13, 1);

          // Bsums are loaded - four bsums are loaded (for two sub blocks) for
          // the different Q8_K blocks
          __m256i lhs_bsums_0123_01 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].bsums + 16 * sb)));
          __m256i lhs_bsums_hsum_ymm_0123_01 = _mm256_castsi128_si256(
            _mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01),
                           _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
          lhs_bsums_hsum_ymm_0123_01 = _mm256_permute2x128_si256(
            lhs_bsums_hsum_ymm_0123_01, lhs_bsums_hsum_ymm_0123_01, 0);
          __m512i lhs_bsums_hsum_0123_01 = _mm512_inserti32x8(
            _mm512_castsi256_si512(lhs_bsums_hsum_ymm_0123_01),
            lhs_bsums_hsum_ymm_0123_01, 1);

          // Shuffle pattern one - left side input
          const __m512i lhs_mat_01_00_sp1 = _mm512_shuffle_epi32(
            lhs_mat_01_00,
            (_MM_PERM_ENUM)160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3)
                                 // A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3)
                                 // A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3)
                                 // A01(0-3)
          const __m512i lhs_mat_23_00_sp1 = _mm512_shuffle_epi32(
            lhs_mat_23_00,
            (_MM_PERM_ENUM)160); // A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3)
                                 // A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3)
                                 // A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3)
                                 // A03(0-3)
          const __m512i lhs_mat_01_01_sp1 = _mm512_shuffle_epi32(
            lhs_mat_01_01,
            (_MM_PERM_ENUM)160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                 // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                 // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                 // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
          const __m512i lhs_mat_23_01_sp1 = _mm512_shuffle_epi32(
            lhs_mat_23_01,
            (_MM_PERM_ENUM)160); // A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                                 // A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                                 // A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                                 // A02(8-11) A02(8-11) A03(8-11) A03(8-11)
          const __m512i lhs_mat_01_02_sp1 = _mm512_shuffle_epi32(
            lhs_mat_01_02,
            (_MM_PERM_ENUM)160); // A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                                 // A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                                 // A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                                 // A00(16-19) A00(16-19) A01(16-19) A01(16-19)
          const __m512i lhs_mat_23_02_sp1 = _mm512_shuffle_epi32(
            lhs_mat_23_02,
            (_MM_PERM_ENUM)160); // A02(16-19) A02(16-19) A03(16-19) A03(16-19)
                                 // A02(16-19) A02(16-19) A03(16-19) A03(16-19)
                                 // A02(16-19) A02(16-19) A03(16-19) A03(16-19)
                                 // A02(16-19) A02(16-19) A03(16-19) A03(16-19)
          const __m512i lhs_mat_01_03_sp1 = _mm512_shuffle_epi32(
            lhs_mat_01_03,
            (_MM_PERM_ENUM)160); // A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                                 // A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                                 // A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                                 // A00(24-27) A00(24-27) A01(24-27) A01(24-27)
          const __m512i lhs_mat_23_03_sp1 = _mm512_shuffle_epi32(
            lhs_mat_23_03,
            (_MM_PERM_ENUM)160); // A02(24-27) A02(24-27) A03(24-27) A03(24-27)
                                 // A02(24-27) A02(24-27) A03(24-27) A03(24-27)
                                 // A02(24-27) A02(24-27) A03(24-27) A03(24-27)
                                 // A02(24-27) A02(24-27) A03(24-27) A03(24-27)

          const __m512i lhs_mat_01_10_sp1 = _mm512_shuffle_epi32(
            lhs_mat_01_10,
            (_MM_PERM_ENUM)160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3)
                                 // A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3)
                                 // A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3)
                                 // A11(0-3)
          const __m512i lhs_mat_23_10_sp1 = _mm512_shuffle_epi32(
            lhs_mat_23_10,
            (_MM_PERM_ENUM)160); // A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3)
                                 // A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3)
                                 // A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3)
                                 // A13(0-3)
          const __m512i lhs_mat_01_11_sp1 = _mm512_shuffle_epi32(
            lhs_mat_01_11,
            (_MM_PERM_ENUM)160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                 // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                 // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                 // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
          const __m512i lhs_mat_23_11_sp1 = _mm512_shuffle_epi32(
            lhs_mat_23_11,
            (_MM_PERM_ENUM)160); // A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                                 // A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                                 // A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                                 // A12(8-11) A12(8-11) A13(8-11) A13(8-11)
          const __m512i lhs_mat_01_12_sp1 = _mm512_shuffle_epi32(
            lhs_mat_01_12,
            (_MM_PERM_ENUM)160); // A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                                 // A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                                 // A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                                 // A10(16-19) A10(16-19) A11(16-19) A11(16-19)
          const __m512i lhs_mat_23_12_sp1 = _mm512_shuffle_epi32(
            lhs_mat_23_12,
            (_MM_PERM_ENUM)160); // A12(16-19) A12(16-19) A13(16-19) A13(16-19)
                                 // A12(16-19) A12(16-19) A13(16-19) A13(16-19)
                                 // A12(16-19) A12(16-19) A13(16-19) A13(16-19)
                                 // A12(16-19) A12(16-19) A13(16-19) A13(16-19)
          const __m512i lhs_mat_01_13_sp1 = _mm512_shuffle_epi32(
            lhs_mat_01_13,
            (_MM_PERM_ENUM)160); // A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                                 // A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                                 // A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                                 // A10(24-27) A10(24-27) A11(24-27) A11(24-27)
          const __m512i lhs_mat_23_13_sp1 = _mm512_shuffle_epi32(
            lhs_mat_23_13,
            (_MM_PERM_ENUM)160); // A12(24-27) A12(24-27) A13(24-27) A13(24-27)
                                 // A12(24-27) A12(24-27) A13(24-27) A13(24-27)
                                 // A12(24-27) A12(24-27) A13(24-27) A13(24-27)
                                 // A12(24-27) A12(24-27) A13(24-27) A13(24-27)

          const __m512i lhs_mat_01_00_sp2 = _mm512_shuffle_epi32(
            lhs_mat_01_00,
            (_MM_PERM_ENUM)245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7)
                                 // A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7)
                                 // A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7)
                                 // A01(4-7)
          const __m512i lhs_mat_23_00_sp2 = _mm512_shuffle_epi32(
            lhs_mat_23_00,
            (_MM_PERM_ENUM)245); // A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7)
                                 // A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7)
                                 // A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7)
                                 // A03(4-7)
          const __m512i lhs_mat_01_01_sp2 = _mm512_shuffle_epi32(
            lhs_mat_01_01,
            (_MM_PERM_ENUM)245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                                 // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                                 // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                                 // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
          const __m512i lhs_mat_23_01_sp2 = _mm512_shuffle_epi32(
            lhs_mat_23_01,
            (_MM_PERM_ENUM)245); // A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                                 // A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                                 // A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                                 // A02(12-15) A02(12-15) A03(12-15) A03(12-15)
          const __m512i lhs_mat_01_02_sp2 = _mm512_shuffle_epi32(
            lhs_mat_01_02,
            (_MM_PERM_ENUM)245); // A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                                 // A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                                 // A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                                 // A00(20-23) A00(20-23) A01(20-23) A01(20-23)
          const __m512i lhs_mat_23_02_sp2 = _mm512_shuffle_epi32(
            lhs_mat_23_02,
            (_MM_PERM_ENUM)245); // A02(20-23) A02(20-23) A03(20-23) A03(20-23)
                                 // A02(20-23) A02(20-23) A03(20-23) A03(20-23)
                                 // A02(20-23) A02(20-23) A03(20-23) A03(20-23)
                                 // A02(20-23) A02(20-23) A03(20-23) A03(20-23)
          const __m512i lhs_mat_01_03_sp2 = _mm512_shuffle_epi32(
            lhs_mat_01_03,
            (_MM_PERM_ENUM)245); // A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                                 // A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                                 // A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                                 // A00(28-31) A00(28-31) A01(28-31) A01(28-31)
          const __m512i lhs_mat_23_03_sp2 = _mm512_shuffle_epi32(
            lhs_mat_23_03,
            (_MM_PERM_ENUM)245); // A02(28-31) A02(28-31) A03(28-31) A03(28-31)
                                 // A02(28-31) A02(28-31) A03(28-31) A03(28-31)
                                 // A02(28-31) A02(28-31) A03(28-31) A03(28-31)
                                 // A02(28-31) A02(28-31) A03(28-31) A03(28-31)

          const __m512i lhs_mat_01_10_sp2 = _mm512_shuffle_epi32(
            lhs_mat_01_10,
            (_MM_PERM_ENUM)245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7)
                                 // A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7)
                                 // A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7)
                                 // A11(4-7)
          const __m512i lhs_mat_23_10_sp2 = _mm512_shuffle_epi32(
            lhs_mat_23_10,
            (_MM_PERM_ENUM)245); // A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7)
                                 // A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7)
                                 // A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7)
                                 // A13(4-7)
          const __m512i lhs_mat_01_11_sp2 = _mm512_shuffle_epi32(
            lhs_mat_01_11,
            (_MM_PERM_ENUM)245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                                 // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                                 // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                                 // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
          const __m512i lhs_mat_23_11_sp2 = _mm512_shuffle_epi32(
            lhs_mat_23_11,
            (_MM_PERM_ENUM)245); // A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                                 // A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                                 // A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                                 // A12(12-15) A12(12-15) A13(12-15) A13(12-15)
          const __m512i lhs_mat_01_12_sp2 = _mm512_shuffle_epi32(
            lhs_mat_01_12,
            (_MM_PERM_ENUM)245); // A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                                 // A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                                 // A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                                 // A10(20-23) A10(20-23) A11(20-23) A11(20-23)
          const __m512i lhs_mat_23_12_sp2 = _mm512_shuffle_epi32(
            lhs_mat_23_12,
            (_MM_PERM_ENUM)245); // A12(20-23) A12(20-23) A13(20-23) A13(20-23)
                                 // A12(20-23) A12(20-23) A13(20-23) A13(20-23)
                                 // A12(20-23) A12(20-23) A13(20-23) A13(20-23)
                                 // A12(20-23) A12(20-23) A13(20-23) A13(20-23)
          const __m512i lhs_mat_01_13_sp2 = _mm512_shuffle_epi32(
            lhs_mat_01_13,
            (_MM_PERM_ENUM)245); // A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                                 // A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                                 // A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                                 // A10(28-31) A10(28-31) A11(28-31) A11(28-31)
          const __m512i lhs_mat_23_13_sp2 = _mm512_shuffle_epi32(
            lhs_mat_23_13,
            (_MM_PERM_ENUM)245); // A12(28-31) A12(28-31) A13(28-31) A13(28-31)
                                 // A12(28-31) A12(28-31) A13(28-31) A13(28-31)
                                 // A12(28-31) A12(28-31) A13(28-31) A13(28-31)
                                 // A12(28-31) A12(28-31) A13(28-31) A13(28-31)

          // The values arranged in shuffle patterns are operated with dot
          // product operation within 32 bit lane i.e corresponding bytes and
          // multiplied and added into 32 bit integers within 32 bit lane
          __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1,
                                                    lhs_mat_01_03_sp1),
                               _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1,
                                                    lhs_mat_01_02_sp1)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_01_01_sp1)),
            _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1));
          __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1,
                                                    lhs_mat_01_03_sp1),
                               _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1,
                                                    lhs_mat_01_02_sp1)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_01_01_sp1)),
            _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1));
          __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1,
                                                    lhs_mat_23_03_sp1),
                               _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1,
                                                    lhs_mat_23_02_sp1)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_23_01_sp1)),
            _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1));
          __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1,
                                                    lhs_mat_23_03_sp1),
                               _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1,
                                                    lhs_mat_23_02_sp1)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_23_01_sp1)),
            _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1));
          __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1,
                                                    lhs_mat_01_13_sp1),
                               _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1,
                                                    lhs_mat_01_12_sp1)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_01_11_sp1)),
            _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1));
          __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1,
                                                    lhs_mat_01_13_sp1),
                               _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1,
                                                    lhs_mat_01_12_sp1)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_01_11_sp1)),
            _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1));
          __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1,
                                                    lhs_mat_23_13_sp1),
                               _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1,
                                                    lhs_mat_23_12_sp1)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_23_11_sp1)),
            _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1));
          __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1,
                                                    lhs_mat_23_13_sp1),
                               _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1,
                                                    lhs_mat_23_12_sp1)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_23_11_sp1)),
            _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1));

          __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2,
                                                    lhs_mat_01_03_sp2),
                               _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2,
                                                    lhs_mat_01_02_sp2)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_01_01_sp2)),
            _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2));
          __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2,
                                                    lhs_mat_01_03_sp2),
                               _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2,
                                                    lhs_mat_01_02_sp2)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_01_01_sp2)),
            _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2));
          __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2,
                                                    lhs_mat_23_03_sp2),
                               _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2,
                                                    lhs_mat_23_02_sp2)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_23_01_sp2)),
            _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2));
          __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2,
                                                    lhs_mat_23_03_sp2),
                               _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2,
                                                    lhs_mat_23_02_sp2)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_23_01_sp2)),
            _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2));
          __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2,
                                                    lhs_mat_01_13_sp2),
                               _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2,
                                                    lhs_mat_01_12_sp2)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_01_11_sp2)),
            _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2));
          __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2,
                                                    lhs_mat_01_13_sp2),
                               _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2,
                                                    lhs_mat_01_12_sp2)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_01_11_sp2)),
            _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2));
          __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2,
                                                    lhs_mat_23_13_sp2),
                               _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2,
                                                    lhs_mat_23_12_sp2)),
              _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_23_11_sp2)),
            _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2));
          __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(
            _mm512_add_epi16(
              _mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2,
                                                    lhs_mat_23_13_sp2),
                               _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2,
                                                    lhs_mat_23_12_sp2)),
              _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_23_11_sp2)),
            _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2));

          // Output of both shuffle patterns are added in order to sum dot
          // product outputs of all 32 values in block
          __m512i iacc_mat_00_0 =
            _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
          __m512i iacc_mat_01_0 =
            _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
          __m512i iacc_mat_10_0 =
            _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
          __m512i iacc_mat_11_0 =
            _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

          __m512i iacc_mat_00_1 =
            _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
          __m512i iacc_mat_01_1 =
            _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
          __m512i iacc_mat_10_1 =
            _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
          __m512i iacc_mat_11_1 =
            _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

          iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
          iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
          iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
          iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

          iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
          iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
          iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
          iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

          // Straighten out to make 4 row vectors (4 for each sub block which
          // are accumulated together in the next step)
          __m512i iacc_row_0_0 = _mm512_mask_blend_epi32(
            0xCCCC, iacc_mat_00_0,
            _mm512_shuffle_epi32(iacc_mat_01_0, (_MM_PERM_ENUM)78));
          __m512i iacc_row_1_0 = _mm512_mask_blend_epi32(
            0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_0, (_MM_PERM_ENUM)78),
            iacc_mat_01_0);
          __m512i iacc_row_2_0 = _mm512_mask_blend_epi32(
            0xCCCC, iacc_mat_10_0,
            _mm512_shuffle_epi32(iacc_mat_11_0, (_MM_PERM_ENUM)78));
          __m512i iacc_row_3_0 = _mm512_mask_blend_epi32(
            0xCCCC, _mm512_shuffle_epi32(iacc_mat_10_0, (_MM_PERM_ENUM)78),
            iacc_mat_11_0);
          __m512i iacc_row_0_1 = _mm512_mask_blend_epi32(
            0xCCCC, iacc_mat_00_1,
            _mm512_shuffle_epi32(iacc_mat_01_1, (_MM_PERM_ENUM)78));
          __m512i iacc_row_1_1 = _mm512_mask_blend_epi32(
            0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_1, (_MM_PERM_ENUM)78),
            iacc_mat_01_1);
          __m512i iacc_row_2_1 = _mm512_mask_blend_epi32(
            0xCCCC, iacc_mat_10_1,
            _mm512_shuffle_epi32(iacc_mat_11_1, (_MM_PERM_ENUM)78));
          __m512i iacc_row_3_1 = _mm512_mask_blend_epi32(
            0xCCCC, _mm512_shuffle_epi32(iacc_mat_10_1, (_MM_PERM_ENUM)78),
            iacc_mat_11_1);

          __m512i iacc_row_0 = _mm512_add_epi32(iacc_row_0_0, iacc_row_0_1);
          __m512i iacc_row_1 = _mm512_add_epi32(iacc_row_1_0, iacc_row_1_1);
          __m512i iacc_row_2 = _mm512_add_epi32(iacc_row_2_0, iacc_row_2_1);
          __m512i iacc_row_3 = _mm512_add_epi32(iacc_row_3_0, iacc_row_3_1);

          // Load the scale(d) values for all the 4 Q8_k blocks and repeat it
          // across lanes
          const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
          const __m256 row_scale_f32_ymm =
            _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
          const __m512 row_scale_f32 = _mm512_insertf32x8(
            _mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

          // Multiply with appropiate scales and accumulate (for both d and
          // dmin) below
          acc_rows[0] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(iacc_row_0),
            _mm512_mul_ps(col_scale_f32,
                          _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
            acc_rows[0]);
          acc_rows[1] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(iacc_row_1),
            _mm512_mul_ps(col_scale_f32,
                          _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)),
            acc_rows[1]);
          acc_rows[2] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(iacc_row_2),
            _mm512_mul_ps(col_scale_f32,
                          _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
            acc_rows[2]);
          acc_rows[3] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(iacc_row_3),
            _mm512_mul_ps(col_scale_f32,
                          _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
            acc_rows[3]);

          __m512i iacc_row_min_0 = _mm512_madd_epi16(
            _mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)0),
            mins_01);
          __m512i iacc_row_min_1 = _mm512_madd_epi16(
            _mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)85),
            mins_01);
          __m512i iacc_row_min_2 = _mm512_madd_epi16(
            _mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)170),
            mins_01);
          __m512i iacc_row_min_3 = _mm512_madd_epi16(
            _mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)255),
            mins_01);

          acc_min_rows[0] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(iacc_row_min_0),
            _mm512_mul_ps(col_dmin_f32,
                          _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
            acc_min_rows[0]);
          acc_min_rows[1] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(iacc_row_min_1),
            _mm512_mul_ps(col_dmin_f32,
                          _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)),
            acc_min_rows[1]);
          acc_min_rows[2] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(iacc_row_min_2),
            _mm512_mul_ps(col_dmin_f32,
                          _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
            acc_min_rows[2]);
          acc_min_rows[3] = _mm512_fmadd_ps(
            _mm512_cvtepi32_ps(iacc_row_min_3),
            _mm512_mul_ps(col_dmin_f32,
                          _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
            acc_min_rows[3]);
        }
      }
      // Store accumlated values
      for (int i = 0; i < 4; i++) {
        _mm512_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                         _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
      }
    }
  }
  if (anc != nc) {
    xstart = anc / 8;
    y = 0;
  }
#endif // AVX512F

  // Take group of four block_q8_Kx4 structures at each pass of the loop and
  // perform dot product operation
  for (; y < anr / 4; y += 4) {

    const block_q8_Kx4 *a_ptrs[4];

    a_ptrs[0] = a_ptr_start + (y * nb);
    for (int i = 0; i < 3; ++i) {
      a_ptrs[i + 1] = a_ptrs[i] + nb;
    }

    // Take group of eight block_q4_kx8 structures at each pass of the loop and
    // perform dot product operation
    for (int64_t x = xstart; x < nc / 8; x++) {

      const block_q4_Kx8 *b_ptr = b_ptr_start + (x * b_nb);

      // Master FP accumulators
      __m256 acc_rows[16];
      for (int i = 0; i < 16; i++) {
        acc_rows[i] = _mm256_setzero_ps();
      }

      __m256 acc_min_rows[16];
      for (int i = 0; i < 16; i++) {
        acc_min_rows[i] = _mm256_setzero_ps();
      }

      // For super block
      for (int64_t b = 0; b < nb; b++) {

        // Scale values - Load the eight scale values of block_q4_kx8
        const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

        // dmin values - Load the eight dmin values of block_q4_kx8
        const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

        // Loop to iterate over the eight sub blocks of a super block - two sub
        // blocks are processed per iteration
        for (int sb = 0; sb < QK_K / 64; sb++) {

          // Load the eight block_q4_K for two sub blocks quantized values
          // interleaved with each other in chunks of eight bytes - B0,B1
          // ....B6,B7
          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
          const __m256i rhs_raw_mat_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
          const __m256i rhs_raw_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
          const __m256i rhs_raw_mat_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
          const __m256i rhs_raw_mat_0123_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
          const __m256i rhs_raw_mat_4567_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
          const __m256i rhs_raw_mat_0123_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
          const __m256i rhs_raw_mat_4567_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

          // Save the values in the following vectors in the formats B0B1B4B5,
          // B2B3B6B7 for further processing and storing of values
          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);
          const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(
            rhs_raw_mat_0123_2,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder),
            rhs_raw_mat_4567_2, 240);
          const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(
            rhs_raw_mat_0123_3,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder),
            rhs_raw_mat_4567_3, 240);

          // 4-bit -> 8-bit
          // First sub block of the two sub blocks processed in the iteration
          const __m256i rhs_mat_0145_00 = _mm256_and_si256(
            rhs_raw_mat_0145_0, m4b); // B00(0-7) B01(0-7) B04(0-7) B05(0-7)
          const __m256i rhs_mat_2367_00 = _mm256_and_si256(
            rhs_raw_mat_2367_0, m4b); // B02(0-7) B03(0-7) B06(0-7) B07(0-7)

          const __m256i rhs_mat_0145_01 = _mm256_and_si256(
            rhs_raw_mat_0145_1, m4b); // B00(8-15) B01(8-15) B04(8-15) B05(8-15)
          const __m256i rhs_mat_2367_01 = _mm256_and_si256(
            rhs_raw_mat_2367_1, m4b); // B02(8-15) B03(8-15) B06(8-15) B07(8-15)

          const __m256i rhs_mat_0145_02 = _mm256_and_si256(
            rhs_raw_mat_0145_2,
            m4b); // B00(16-23) B01(16-23) B04(16-23) B05(16-23)
          const __m256i rhs_mat_2367_02 = _mm256_and_si256(
            rhs_raw_mat_2367_2,
            m4b); // B02(16-23) B03(16-23) B06(16-23) B07(16-23)

          const __m256i rhs_mat_0145_03 = _mm256_and_si256(
            rhs_raw_mat_0145_3,
            m4b); // B00(24-31) B01(24-31) B04(24-31) B05(24-31)
          const __m256i rhs_mat_2367_03 = _mm256_and_si256(
            rhs_raw_mat_2367_3,
            m4b); // B02(24-31) B03(24-31) B06(24-31) B07(24-31)

          // Second sub block of the two sub blocks processed in the iteration
          const __m256i rhs_mat_0145_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4),
                             m4b); // B10(0-7) B11(0-7) B14(0-7) B15(0-7)
          const __m256i rhs_mat_2367_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4),
                             m4b); // B12(0-7) B13(0-7) B16(0-7) B17(0-7)

          const __m256i rhs_mat_0145_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4),
                             m4b); // B10(8-15) B11(8-15) B14(8-15) B15(8-15)
          const __m256i rhs_mat_2367_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4),
                             m4b); // B12(8-15) B13(8-15) B16(8-15) B17(8-15)

          const __m256i rhs_mat_0145_12 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_0145_2, 4),
            m4b); // B10(16-23) B11(16-23) B14(16-23) B15(16-23)
          const __m256i rhs_mat_2367_12 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_2367_2, 4),
            m4b); // B12(16-23) B13(16-23) B16(16-23) B17(16-23)

          const __m256i rhs_mat_0145_13 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_0145_3, 4),
            m4b); // B10(24-31) B11(24-31) B14(24-31) B15(24-31)
          const __m256i rhs_mat_2367_13 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_2367_3, 4),
            m4b); // B12(24-31) B13(24-31) B16(24-31) B17(24-31)

          // Shuffle pattern one - right side input
          const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_00, 136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3)
                                   // B04(0-3) B05(0-3) B04(0-3) B05(0-3)
          const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_00, 136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3)
                                   // B06(0-3) B07(0-3) B06(0-3) B07(0-3)

          const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_01, 136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11)
                                   // B04(8-11) B05(8-11) B04(8-11) B05(8-11)
          const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_01, 136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11)
                                   // B06(8-11) B07(8-11) B06(8-11) B07(8-11)

          const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_02,
            136); // B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19)
                  // B05(16-19) B04(16-19) B05(16-19)
          const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_02,
            136); // B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19)
                  // B07(16-19) B06(16-19) B07(16-19)

          const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_03,
            136); // B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27)
                  // B05(24-27) B04(24-27) B05(24-27)
          const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_03,
            136); // B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27)
                  // B07(24-27) B06(24-27) B07(24-27)

          const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_10, 136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3)
                                   // B14(0-3) B15(0-3) B14(0-3) B15(0-3)
          const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_10, 136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3)
                                   // B16(0-3) B17(0-3) B16(0-3) B17(0-3)

          const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_11, 136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11)
                                   // B14(8-11) B15(8-11) B14(8-11) B15(8-11)
          const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_11, 136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11)
                                   // B16(8-11) B17(8-11) B16(8-11) B17(8-11)

          const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_12,
            136); // B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19)
                  // B15(16-19) B14(16-19) B15(16-19)
          const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_12,
            136); // B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19)
                  // B17(16-19) B16(16-19) B17(16-19)

          const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_13,
            136); // B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27)
                  // B15(24-27) B14(24-27) B15(24-27)
          const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_13,
            136); // B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27)
                  // B17(24-27) B16(24-27) B17(24-27)

          // Shuffle pattern two - right side input
          const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_00, 221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7)
                                   // B04(4-7) B05(4-7) B04(4-7) B05(4-7)
          const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_00, 221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7)
                                   // B06(4-7) B07(4-7) B06(4-7) B07(4-7)

          const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_01,
            221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15)
                  // B05(12-15) B04(12-15) B05(12-15)
          const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_01,
            221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15)
                  // B07(12-15) B06(12-15) B07(12-15)

          const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_02,
            221); // B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23)
                  // B05(20-23) B04(20-23) B05(20-23)
          const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_02,
            221); // B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23)
                  // B07(20-23) B06(20-23) B07(20-23)

          const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_03,
            221); // B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31)
                  // B05(28-31) B04(28-31) B05(28-31)
          const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_03,
            221); // B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31)
                  // B07(28-31) B06(28-31) B07(28-31)

          const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_10, 221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7)
                                   // B14(4-7) B15(4-7) B14(4-7) B15(4-7)
          const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_10, 221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7)
                                   // B16(4-7) B17(4-7) B16(4-7) B17(4-7)

          const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_11,
            221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15)
                  // B15(12-15) B14(12-15) B15(12-15)
          const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_11,
            221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15)
                  // B17(12-15) B16(12-15) B17(12-15)

          const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_12,
            221); // B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23)
                  // B15(20-23) B14(20-23) B15(20-23)
          const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_12,
            221); // B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23)
                  // B17(20-23) B16(20-23) B17(20-23)

          const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_13,
            221); // B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31)
                  // B15(28-31) B14(28-31) B15(28-31)
          const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_13,
            221); // B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31)
                  // B17(28-31) B16(28-31) B17(28-31)

          uint32_t utmp_0[4], utmp_1[4];

          // Scales and Mins of corresponding sub blocks from different Q4_K
          // structures are stored together The below block is for eg to extract
          // first sub block's scales and mins from different Q4_K structures
          // for the sb loop
          memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
          utmp_0[3] =
            ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_0 = utmp_0[1] & kmask1;
          utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
          utmp_0[2] = uaux_0;
          utmp_0[0] &= kmask1;

          // The below block is for eg to extract second sub block's scales and
          // mins from different Q4_K structures for the sb loop
          memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
          utmp_1[3] =
            ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_1 = utmp_1[1] & kmask1;
          utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
          utmp_1[2] = uaux_1;
          utmp_1[0] &= kmask1;

          // Scales of first sub block in the sb loop
          const __m128i mins_and_scales_0 =
            _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
          const __m256i scales_0 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

          // Scales of second sub block in the sb loop
          const __m128i mins_and_scales_1 =
            _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
          const __m256i scales_1 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

          // Mins of first and second sub block of Q4_K block are arranged side
          // by side
          const __m256i mins_01 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78),
                              _mm_shuffle_epi32(mins_and_scales_1, 78)));

          const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
          const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

          const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
          const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

          for (int rp = 0; rp < 4; rp++) {

            // Load the four block_q8_k quantized values interleaved with each
            // other in chunks of eight bytes - A0,A1,A2,A3 Loaded as set of 128
            // bit vectors and repeated into a 256 bit vector
            __m256i lhs_mat_0123_00 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 256 * sb)));
            __m256i lhs_mat_01_00 =
              _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
            __m256i lhs_mat_23_00 =
              _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
            __m256i lhs_mat_0123_01 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 32 + 256 * sb)));
            __m256i lhs_mat_01_01 =
              _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
            __m256i lhs_mat_23_01 =
              _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
            __m256i lhs_mat_0123_02 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 64 + 256 * sb)));
            __m256i lhs_mat_01_02 =
              _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
            __m256i lhs_mat_23_02 =
              _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
            __m256i lhs_mat_0123_03 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 96 + 256 * sb)));
            __m256i lhs_mat_01_03 =
              _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
            __m256i lhs_mat_23_03 =
              _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
            __m256i lhs_mat_0123_10 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 128 + 256 * sb)));
            __m256i lhs_mat_01_10 =
              _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
            __m256i lhs_mat_23_10 =
              _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
            __m256i lhs_mat_0123_11 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 160 + 256 * sb)));
            __m256i lhs_mat_01_11 =
              _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
            __m256i lhs_mat_23_11 =
              _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
            __m256i lhs_mat_0123_12 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 192 + 256 * sb)));
            __m256i lhs_mat_01_12 =
              _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
            __m256i lhs_mat_23_12 =
              _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
            __m256i lhs_mat_0123_13 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].qs + 224 + 256 * sb)));
            __m256i lhs_mat_01_13 =
              _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
            __m256i lhs_mat_23_13 =
              _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

            // Bsums are loaded - four bsums are loaded (for two sub blocks) for
            // the different Q8_K blocks
            __m256i lhs_bsums_0123_01 = _mm256_loadu_si256(
              (const __m256i *)((a_ptrs[rp][b].bsums + 16 * sb)));
            __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(
              _mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01),
                             _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
            lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(
              lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

            // Shuffle pattern one - left side input
            const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_00, 160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                                   // A00(0-3) A00(0-3) A01(0-3) A01(0-3)
            const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_00, 160); // A02(0-3) A03(0-3) A02(0-3) A03(0-3)
                                   // A02(0-3) A03(0-3) A02(0-3) A03(0-3)

            const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_01, 160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                   // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
            const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_01, 160); // A02(8-11) A03(8-11) A02(8-11) A03(8-11)
                                   // A02(8-11) A03(8-11) A02(8-11) A03(8-11)

            const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_02,
              160); // A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19)
                    // A00(16-19) A01(16-19) A01(16-19)
            const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_02,
              160); // A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19)
                    // A03(16-19) A02(16-19) A03(16-19)

            const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_03,
              160); // A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27)
                    // A00(24-27) A01(24-27) A01(24-27)
            const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_03,
              160); // A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27)
                    // A03(24-27) A02(24-27) A03(24-27)

            const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_10, 160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                                   // A10(0-3) A10(0-3) A11(0-3) A11(0-3)
            const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_10, 160); // A12(0-3) A13(0-3) A12(0-3) A13(0-3)
                                   // A12(0-3) A13(0-3) A12(0-3) A13(0-3)

            const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_11, 160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                   // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
            const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_11, 160); // A12(8-11) A13(8-11) A12(8-11) A13(8-11)
                                   // A12(8-11) A13(8-11) A12(8-11) A13(8-11)

            const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_12,
              160); // A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19)
                    // A10(16-19) A11(16-19) A11(16-19)
            const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_12,
              160); // A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19)
                    // A13(16-19) A12(16-19) A13(16-19)

            const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(
              lhs_mat_01_13,
              160); // A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27)
                    // A10(24-27) A11(24-27) A11(24-27)
            const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(
              lhs_mat_23_13,
              160); // A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27)
                    // A13(24-27) A12(24-27) A13(24-27)

            // Shuffle pattern two- left side input
            const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_00, 245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                                   // A00(4-7) A00(4-7) A01(4-7) A01(4-7)
            const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_00, 245); // A02(4-7) A03(4-7) A02(4-7) A03(4-7)
                                   // A02(4-7) A03(4-7) A02(4-7) A03(4-7)

            const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_01,
              245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15)
                    // A00(12-15) A01(12-15) A01(12-15)
            const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_01,
              245); // A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15)
                    // A03(12-15) A02(12-15) A03(12-15)

            const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_02,
              245); // A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23)
                    // A00(20-23) A01(20-23) A01(20-23)
            const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_02,
              245); // A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23)
                    // A03(20-23) A02(20-23) A03(20-23)

            const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_03,
              245); // A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31)
                    // A00(28-31) A01(28-31) A01(28-31)
            const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_03,
              245); // A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31)
                    // A03(28-31) A02(28-31) A03(28-31)

            const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_10, 245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                                   // A10(4-7) A10(4-7) A11(4-7) A11(4-7)
            const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_10, 245); // A12(4-7) A13(4-7) A12(4-7) A13(4-7)
                                   // A12(4-7) A13(4-7) A12(4-7) A13(4-7)

            const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_11,
              245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15)
                    // A10(12-15) A11(12-15) A11(12-15)
            const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_11,
              245); // A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15)
                    // A13(12-15) A12(12-15) A13(12-15)

            const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_12,
              245); // A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23)
                    // A10(20-23) A11(20-23) A11(20-23)
            const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_12,
              245); // A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23)
                    // A13(20-23) A12(20-23) A13(20-23)

            const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(
              lhs_mat_01_13,
              245); // A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31)
                    // A10(28-31) A11(28-31) A11(28-31)
            const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(
              lhs_mat_23_13,
              245); // A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31)
                    // A13(28-31) A12(28-31) A13(28-31)

            // The values arranged in shuffle patterns are operated with dot
            // product operation within 32 bit lane i.e corresponding bytes and
            // multiplied and added into 32 bit integers within 32 bit lane
            __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1),
                  _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)),
                _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
            __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1),
                  _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)),
                _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
            __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1),
                  _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)),
                _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
            __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1),
                  _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)),
                _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
            __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1),
                  _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)),
                _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
            __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1),
                  _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)),
                _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
            __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1),
                  _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)),
                _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
            __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1),
                  _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)),
                _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

            __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2),
                  _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)),
                _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
            __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2),
                  _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)),
                _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
            __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2),
                  _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)),
                _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
            __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2),
                  _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)),
                _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
            __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2),
                  _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)),
                _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
            __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2),
                  _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)),
                _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
            __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2),
                  _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)),
                _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
            __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_add_epi16(
                  _mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2),
                  _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)),
                _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

            // Output of both shuffle patterns are added in order to sum dot
            // product outputs of all 32 values in block
            __m256i iacc_mat_00_0 =
              _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
            __m256i iacc_mat_01_0 =
              _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
            __m256i iacc_mat_10_0 =
              _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
            __m256i iacc_mat_11_0 =
              _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

            __m256i iacc_mat_00_1 =
              _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
            __m256i iacc_mat_01_1 =
              _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
            __m256i iacc_mat_10_1 =
              _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
            __m256i iacc_mat_11_1 =
              _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

            // Output of both shuffle patterns are added in order to sum dot
            // product outputs of all 32 values in block
            iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
            iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
            iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
            iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

            iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
            iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
            iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
            iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

            // Straighten out to make 4 row vectors (4 for each sub block which
            // are accumulated together in the next step)
            __m256i iacc_row_0_0 = _mm256_blend_epi32(
              iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
            __m256i iacc_row_1_0 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
            __m256i iacc_row_2_0 = _mm256_blend_epi32(
              iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
            __m256i iacc_row_3_0 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
            __m256i iacc_row_0_1 = _mm256_blend_epi32(
              iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
            __m256i iacc_row_1_1 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
            __m256i iacc_row_2_1 = _mm256_blend_epi32(
              iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
            __m256i iacc_row_3_1 = _mm256_blend_epi32(
              _mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

            __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
            __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
            __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
            __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

            // Load the scale(d) values for all the 4 Q8_k blocks and repeat it
            // across lanes
            const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
            const __m256 row_scale_f32 = _mm256_set_m128(
              row_scale_f32_sse,
              row_scale_f32_sse); // GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d,
                                  // loadMask);

            // Multiply with appropiate scales and accumulate (for both d and
            // dmin) below
            acc_rows[rp * 4] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_0),
              _mm256_mul_ps(col_scale_f32,
                            _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
              acc_rows[rp * 4]);
            acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_1),
              _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(
                                             row_scale_f32, row_scale_f32, 85)),
              acc_rows[rp * 4 + 1]);
            acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_2),
              _mm256_mul_ps(
                col_scale_f32,
                _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
              acc_rows[rp * 4 + 2]);
            acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_3),
              _mm256_mul_ps(
                col_scale_f32,
                _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
              acc_rows[rp * 4 + 3]);

            __m256i iacc_row_min_0 = _mm256_madd_epi16(
              _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
            __m256i iacc_row_min_1 = _mm256_madd_epi16(
              _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
            __m256i iacc_row_min_2 = _mm256_madd_epi16(
              _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
            __m256i iacc_row_min_3 = _mm256_madd_epi16(
              _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

            acc_min_rows[rp * 4] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_min_0),
              _mm256_mul_ps(col_dmin_f32,
                            _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
              acc_min_rows[rp * 4]);
            acc_min_rows[rp * 4 + 1] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_min_1),
              _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32,
                                                            row_scale_f32, 85)),
              acc_min_rows[rp * 4 + 1]);
            acc_min_rows[rp * 4 + 2] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_min_2),
              _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(
                                            row_scale_f32, row_scale_f32, 170)),
              acc_min_rows[rp * 4 + 2]);
            acc_min_rows[rp * 4 + 3] = _mm256_fmadd_ps(
              _mm256_cvtepi32_ps(iacc_row_min_3),
              _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(
                                            row_scale_f32, row_scale_f32, 255)),
              acc_min_rows[rp * 4 + 3]);
          }
        }
      }
      // Store the accumulated values
      for (int i = 0; i < 16; i++) {
        _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                         _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
      }
    }
  }
  for (; y < nr / 4; y++) {

    const block_q8_Kx4 *a_ptr = a_ptr_start + (y * nb);

    for (int64_t x = xstart; x < nc / 8; x++) {

      const block_q4_Kx8 *b_ptr = b_ptr_start + (x * b_nb);

      // Master FP accumulators
      __m256 acc_rows[4];
      for (int i = 0; i < 4; i++) {
        acc_rows[i] = _mm256_setzero_ps();
      }

      __m256 acc_min_rows[4];
      for (int i = 0; i < 4; i++) {
        acc_min_rows[i] = _mm256_setzero_ps();
      }

      for (int64_t b = 0; b < nb; b++) {

        // Scale values - Load the eight scale values of block_q4_Kx8
        const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

        // dmin values - Load the eight dmin values of block_q4_Kx8
        const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

        // Loop to iterate over the eight sub blocks of a super block - two sub
        // blocks are processed per iteration
        for (int sb = 0; sb < QK_K / 64; sb++) {

          // Load the eight block_q4_k for two sub blocks quantized values
          // interleaved with each other in chunks of eight bytes - B0,B1
          // ....B6,B7
          const __m256i rhs_raw_mat_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
          const __m256i rhs_raw_mat_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
          const __m256i rhs_raw_mat_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
          const __m256i rhs_raw_mat_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
          const __m256i rhs_raw_mat_0123_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
          const __m256i rhs_raw_mat_4567_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
          const __m256i rhs_raw_mat_0123_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
          const __m256i rhs_raw_mat_4567_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

          // Save the values in the following vectors in the formats B0B1B4B5,
          // B2B3B6B7 for further processing and storing of values
          const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(
            rhs_raw_mat_0123_0,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder),
            rhs_raw_mat_4567_0, 240);
          const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(
            rhs_raw_mat_0123_1,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder),
            rhs_raw_mat_4567_1, 240);
          const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(
            rhs_raw_mat_0123_2,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder),
            rhs_raw_mat_4567_2, 240);
          const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(
            rhs_raw_mat_0123_3,
            _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder),
            240);
          const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(
            _mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder),
            rhs_raw_mat_4567_3, 240);

          // 4-bit -> 8-bit
          // First sub block of the two sub blocks processed in the iteration
          const __m256i rhs_mat_0145_00 = _mm256_and_si256(
            rhs_raw_mat_0145_0, m4b); // B00(0-7) B01(0-7) B04(0-7) B05(0-7)
          const __m256i rhs_mat_2367_00 = _mm256_and_si256(
            rhs_raw_mat_2367_0, m4b); // B02(0-7) B03(0-7) B06(0-7) B07(0-7)

          const __m256i rhs_mat_0145_01 = _mm256_and_si256(
            rhs_raw_mat_0145_1, m4b); // B00(8-15) B01(8-15) B04(8-15) B05(8-15)
          const __m256i rhs_mat_2367_01 = _mm256_and_si256(
            rhs_raw_mat_2367_1, m4b); // B02(8-15) B03(8-15) B06(8-15) B07(8-15)

          const __m256i rhs_mat_0145_02 = _mm256_and_si256(
            rhs_raw_mat_0145_2,
            m4b); // B00(16-23) B01(16-23) B04(16-23) B05(16-23)
          const __m256i rhs_mat_2367_02 = _mm256_and_si256(
            rhs_raw_mat_2367_2,
            m4b); // B02(16-23) B03(16-23) B06(16-23) B07(16-23)

          const __m256i rhs_mat_0145_03 = _mm256_and_si256(
            rhs_raw_mat_0145_3,
            m4b); // B00(24-31) B01(24-31) B04(24-31) B05(24-31)
          const __m256i rhs_mat_2367_03 = _mm256_and_si256(
            rhs_raw_mat_2367_3,
            m4b); // B02(24-31) B03(24-31) B06(24-31) B07(24-31)

          // Second sub block of the two sub blocks processed in the iteration
          const __m256i rhs_mat_0145_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4),
                             m4b); // B10(0-7) B11(0-7) B14(0-7) B15(0-7)
          const __m256i rhs_mat_2367_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4),
                             m4b); // B12(0-7) B13(0-7) B16(0-7) B17(0-7)

          const __m256i rhs_mat_0145_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4),
                             m4b); // B10(8-15) B11(8-15) B14(8-15) B15(8-15)
          const __m256i rhs_mat_2367_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4),
                             m4b); // B12(8-15) B13(8-15) B16(8-15) B17(8-15)

          const __m256i rhs_mat_0145_12 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_0145_2, 4),
            m4b); // B10(16-23) B11(16-23) B14(16-23) B15(16-23)
          const __m256i rhs_mat_2367_12 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_2367_2, 4),
            m4b); // B12(16-23) B13(16-23) B16(16-23) B17(16-23)

          const __m256i rhs_mat_0145_13 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_0145_3, 4),
            m4b); // B10(24-31) B11(24-31) B14(24-31) B15(24-31)
          const __m256i rhs_mat_2367_13 = _mm256_and_si256(
            _mm256_srli_epi16(rhs_raw_mat_2367_3, 4),
            m4b); // B12(24-31) B13(24-31) B16(24-31) B17(24-31)

          // Shuffle pattern one - right side input
          const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_00, 136); // B00(0-3) B01(0-3) B00(0-3) B01(0-3)
                                   // B04(0-3) B05(0-3) B04(0-3) B05(0-3)
          const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_00, 136); // B02(0-3) B03(0-3) B02(0-3) B03(0-3)
                                   // B06(0-3) B07(0-3) B06(0-3) B07(0-3)

          const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_01, 136); // B00(8-11) B01(8-11) B00(8-11) B01(8-11)
                                   // B04(8-11) B05(8-11) B04(8-11) B05(8-11)
          const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_01, 136); // B02(8-11) B03(8-11) B02(8-11) B03(8-11)
                                   // B06(8-11) B07(8-11) B06(8-11) B07(8-11)

          const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_02,
            136); // B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19)
                  // B05(16-19) B04(16-19) B05(16-19)
          const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_02,
            136); // B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19)
                  // B07(16-19) B06(16-19) B07(16-19)

          const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_03,
            136); // B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27)
                  // B05(24-27) B04(24-27) B05(24-27)
          const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_03,
            136); // B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27)
                  // B07(24-27) B06(24-27) B07(24-27)

          const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_10, 136); // B10(0-3) B11(0-3) B10(0-3) B11(0-3)
                                   // B14(0-3) B15(0-3) B14(0-3) B15(0-3)
          const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_10, 136); // B12(0-3) B13(0-3) B12(0-3) B13(0-3)
                                   // B16(0-3) B17(0-3) B16(0-3) B17(0-3)

          const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_11, 136); // B10(8-11) B11(8-11) B10(8-11) B11(8-11)
                                   // B14(8-11) B15(8-11) B14(8-11) B15(8-11)
          const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_11, 136); // B12(8-11) B13(8-11) B12(8-11) B13(8-11)
                                   // B16(8-11) B17(8-11) B16(8-11) B17(8-11)

          const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_12,
            136); // B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19)
                  // B15(16-19) B14(16-19) B15(16-19)
          const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_12,
            136); // B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19)
                  // B17(16-19) B16(16-19) B17(16-19)

          const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(
            rhs_mat_0145_13,
            136); // B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27)
                  // B15(24-27) B14(24-27) B15(24-27)
          const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(
            rhs_mat_2367_13,
            136); // B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27)
                  // B17(24-27) B16(24-27) B17(24-27)

          // Shuffle pattern two - right side input
          const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_00, 221); // B00(4-7) B01(4-7) B00(4-7) B01(4-7)
                                   // B04(4-7) B05(4-7) B04(4-7) B05(4-7)
          const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_00, 221); // B02(4-7) B03(4-7) B02(4-7) B03(4-7)
                                   // B06(4-7) B07(4-7) B06(4-7) B07(4-7)

          const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_01,
            221); // B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15)
                  // B05(12-15) B04(12-15) B05(12-15)
          const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_01,
            221); // B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15)
                  // B07(12-15) B06(12-15) B07(12-15)

          const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_02,
            221); // B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23)
                  // B05(20-23) B04(20-23) B05(20-23)
          const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_02,
            221); // B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23)
                  // B07(20-23) B06(20-23) B07(20-23)

          const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_03,
            221); // B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31)
                  // B05(28-31) B04(28-31) B05(28-31)
          const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_03,
            221); // B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31)
                  // B07(28-31) B06(28-31) B07(28-31)

          const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_10, 221); // B10(4-7) B11(4-7) B10(4-7) B11(4-7)
                                   // B14(4-7) B15(4-7) B14(4-7) B15(4-7)
          const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_10, 221); // B12(4-7) B13(4-7) B12(4-7) B13(4-7)
                                   // B16(4-7) B17(4-7) B16(4-7) B17(4-7)

          const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_11,
            221); // B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15)
                  // B15(12-15) B14(12-15) B15(12-15)
          const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_11,
            221); // B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15)
                  // B17(12-15) B16(12-15) B17(12-15)

          const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_12,
            221); // B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23)
                  // B15(20-23) B14(20-23) B15(20-23)
          const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_12,
            221); // B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23)
                  // B17(20-23) B16(20-23) B17(20-23)

          const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(
            rhs_mat_0145_13,
            221); // B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31)
                  // B15(28-31) B14(28-31) B15(28-31)
          const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(
            rhs_mat_2367_13,
            221); // B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31)
                  // B17(28-31) B16(28-31) B17(28-31)

          uint32_t utmp_0[4], utmp_1[4];

          // Scales and Mins of corresponding sub blocks from different Q4_K
          // structures are stored together The below block is for eg to extract
          // first sub block's scales and mins from different Q4_K structures
          // for the sb loop
          memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
          utmp_0[3] =
            ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_0 = utmp_0[1] & kmask1;
          utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
          utmp_0[2] = uaux_0;
          utmp_0[0] &= kmask1;

          // The below block is for eg to extract second sub block's scales and
          // mins from different Q4_K structures when sb = 1
          memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
          utmp_1[3] =
            ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_1 = utmp_1[1] & kmask1;
          utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
          utmp_1[2] = uaux_1;
          utmp_1[0] &= kmask1;

          // Scales of first sub block in the sb loop
          const __m128i mins_and_scales_0 =
            _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
          const __m256i scales_0 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

          // Scales of second sub block in the sb loop
          const __m128i mins_and_scales_1 =
            _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
          const __m256i scales_1 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

          // Mins of first and second sub block of Q4_K block are arranged side
          // by side
          const __m256i mins_01 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78),
                              _mm_shuffle_epi32(mins_and_scales_1, 78)));

          const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
          const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

          const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
          const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

          // Load the four block_q8_k quantized values interleaved with each
          // other in chunks of eight bytes - A0,A1,A2,A3 Loaded as set of 128
          // bit vectors and repeated into a 256 bit vector
          __m256i lhs_mat_0123_00 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 256 * sb)));
          __m256i lhs_mat_01_00 =
            _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
          __m256i lhs_mat_23_00 =
            _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
          __m256i lhs_mat_0123_01 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 32 + 256 * sb)));
          __m256i lhs_mat_01_01 =
            _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
          __m256i lhs_mat_23_01 =
            _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
          __m256i lhs_mat_0123_02 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 64 + 256 * sb)));
          __m256i lhs_mat_01_02 =
            _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
          __m256i lhs_mat_23_02 =
            _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
          __m256i lhs_mat_0123_03 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 96 + 256 * sb)));
          __m256i lhs_mat_01_03 =
            _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
          __m256i lhs_mat_23_03 =
            _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
          __m256i lhs_mat_0123_10 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 128 + 256 * sb)));
          __m256i lhs_mat_01_10 =
            _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
          __m256i lhs_mat_23_10 =
            _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
          __m256i lhs_mat_0123_11 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 160 + 256 * sb)));
          __m256i lhs_mat_01_11 =
            _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
          __m256i lhs_mat_23_11 =
            _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
          __m256i lhs_mat_0123_12 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 192 + 256 * sb)));
          __m256i lhs_mat_01_12 =
            _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
          __m256i lhs_mat_23_12 =
            _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
          __m256i lhs_mat_0123_13 = _mm256_loadu_si256(
            (const __m256i *)((a_ptr[b].qs + 224 + 256 * sb)));
          __m256i lhs_mat_01_13 =
            _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
          __m256i lhs_mat_23_13 =
            _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

          // Bsums are loaded - four bsums are loaded (for two sub blocks) for
          // the different Q8_K blocks
          __m256i lhs_bsums_0123_01 =
            _mm256_loadu_si256((const __m256i *)((a_ptr[b].bsums + 16 * sb)));
          __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(
            _mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01),
                           _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
          lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(
            lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

          // Shuffle pattern one - left side input
          const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_00, 160); // A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3)
                                 // A00(0-3) A01(0-3) A01(0-3)
          const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_00, 160); // A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3)
                                 // A03(0-3) A02(0-3) A03(0-3)

          const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_01, 160); // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                                 // A00(8-11) A00(8-11) A01(8-11) A01(8-11)
          const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_01, 160); // A02(8-11) A03(8-11) A02(8-11) A03(8-11)
                                 // A02(8-11) A03(8-11) A02(8-11) A03(8-11)

          const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_02, 160); // A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                                 // A00(16-19) A00(16-19) A01(16-19) A01(16-19)
          const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_02, 160); // A02(16-19) A03(16-19) A02(16-19) A03(16-19)
                                 // A02(16-19) A03(16-19) A02(16-19) A03(16-19)

          const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_03, 160); // A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                                 // A00(24-27) A00(24-27) A01(24-27) A01(24-27)
          const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_03, 160); // A02(24-27) A03(24-27) A02(24-27) A03(24-27)
                                 // A02(24-27) A03(24-27) A02(24-27) A03(24-27)

          const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_10, 160); // A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3)
                                 // A10(0-3) A11(0-3) A11(0-3)
          const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_10, 160); // A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3)
                                 // A13(0-3) A12(0-3) A13(0-3)

          const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_11, 160); // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                                 // A10(8-11) A10(8-11) A11(8-11) A11(8-11)
          const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_11, 160); // A12(8-11) A13(8-11) A12(8-11) A13(8-11)
                                 // A12(8-11) A13(8-11) A12(8-11) A13(8-11)

          const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_12, 160); // A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                                 // A10(16-19) A10(16-19) A11(16-19) A11(16-19)
          const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_12, 160); // A12(16-19) A13(16-19) A12(16-19) A13(16-19)
                                 // A12(16-19) A13(16-19) A12(16-19) A13(16-19)

          const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(
            lhs_mat_01_13, 160); // A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                                 // A10(24-27) A10(24-27) A11(24-27) A11(24-27)
          const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(
            lhs_mat_23_13, 160); // A12(24-27) A13(24-27) A12(24-27) A13(24-27)
                                 // A12(24-27) A13(24-27) A12(24-27) A13(24-27)

          // Shuffle pattern two- left side input
          const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_00, 245); // A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7)
                                 // A00(4-7) A01(4-7) A01(4-7)
          const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_00, 245); // A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7)
                                 // A03(4-7) A02(4-7) A03(4-7)

          const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_01, 245); // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                                 // A00(12-15) A00(12-15) A01(12-15) A01(12-15)
          const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_01, 245); // A02(12-15) A03(12-15) A02(12-15) A03(12-15)
                                 // A02(12-15) A03(12-15) A02(12-15) A03(12-15)

          const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_02, 245); // A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                                 // A00(20-23) A00(20-23) A01(20-23) A01(20-23)
          const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_02, 245); // A02(20-23) A03(20-23) A02(20-23) A03(20-23)
                                 // A02(20-23) A03(20-23) A02(20-23) A03(20-23)

          const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_03, 245); // A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                                 // A00(28-31) A00(28-31) A01(28-31) A01(28-31)
          const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_03, 245); // A02(28-31) A03(28-31) A02(28-31) A03(28-31)
                                 // A02(28-31) A03(28-31) A02(28-31) A03(28-31)

          const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_10, 245); // A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7)
                                 // A10(4-7) A11(4-7) A11(4-7)
          const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_10, 245); // A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7)
                                 // A13(4-7) A12(4-7) A13(4-7)

          const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_11, 245); // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                                 // A10(12-15) A10(12-15) A11(12-15) A11(12-15)
          const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_11, 245); // A12(12-15) A13(12-15) A12(12-15) A13(12-15)
                                 // A12(12-15) A13(12-15) A12(12-15) A13(12-15)

          const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_12, 245); // A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                                 // A10(20-23) A10(20-23) A11(20-23) A11(20-23)
          const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_12, 245); // A12(20-23) A13(20-23) A12(20-23) A13(20-23)
                                 // A12(20-23) A13(20-23) A12(20-23) A13(20-23)

          const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(
            lhs_mat_01_13, 245); // A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                                 // A10(28-31) A10(28-31) A11(28-31) A11(28-31)
          const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(
            lhs_mat_23_13, 245); // A12(28-31) A13(28-31) A12(28-31) A13(28-31)
                                 // A12(28-31) A13(28-31) A12(28-31) A13(28-31)

          // The values arranged in shuffle patterns are operated with dot
          // product operation within 32 bit lane i.e corresponding bytes and
          // multiplied and added into 32 bit integers within 32 bit lane
          __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1),
                _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)),
            _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
          __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1),
                _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)),
            _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
          __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1),
                _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)),
            _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
          __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1),
                _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)),
            _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
          __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1),
                _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)),
            _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
          __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1),
                _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)),
            _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
          __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1),
                _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)),
              _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)),
            _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
          __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1),
                _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)),
              _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)),
            _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

          __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2),
                _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)),
            _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
          __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2),
                _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)),
            _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
          __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2),
                _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)),
            _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
          __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2),
                _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)),
            _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
          __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2),
                _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)),
            _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
          __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2),
                _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)),
            _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
          __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2),
                _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)),
              _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)),
            _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
          __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(
            _mm256_add_epi16(
              _mm256_add_epi16(
                _mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2),
                _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)),
              _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)),
            _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

          // Output of both shuffle patterns are added in order to sum dot
          // product outputs of all 32 values in block
          __m256i iacc_mat_00_0 =
            _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
          __m256i iacc_mat_01_0 =
            _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
          __m256i iacc_mat_10_0 =
            _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
          __m256i iacc_mat_11_0 =
            _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

          __m256i iacc_mat_00_1 =
            _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
          __m256i iacc_mat_01_1 =
            _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
          __m256i iacc_mat_10_1 =
            _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
          __m256i iacc_mat_11_1 =
            _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

          // Output of both shuffle patterns are added in order to sum dot
          // product outputs of all 32 values in block
          iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
          iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
          iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
          iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

          iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
          iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
          iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
          iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

          // Straighten out to make 4 row vectors (4 for each sub block which
          // are accumulated together in the next step)
          __m256i iacc_row_0_0 = _mm256_blend_epi32(
            iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
          __m256i iacc_row_1_0 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
          __m256i iacc_row_2_0 = _mm256_blend_epi32(
            iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
          __m256i iacc_row_3_0 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
          __m256i iacc_row_0_1 = _mm256_blend_epi32(
            iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
          __m256i iacc_row_1_1 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
          __m256i iacc_row_2_1 = _mm256_blend_epi32(
            iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
          __m256i iacc_row_3_1 = _mm256_blend_epi32(
            _mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

          __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
          __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
          __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
          __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

          // Load the scale(d) values for all the 4 Q8_k blocks and repeat it
          // across lanes
          const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
          const __m256 row_scale_f32 = _mm256_set_m128(
            row_scale_f32_sse,
            row_scale_f32_sse); // GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d,
                                // loadMask);

          // Multiply with appropiate scales and accumulate (for both d and
          // dmin) below
          acc_rows[0] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_0),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
            acc_rows[0]);
          acc_rows[1] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_1),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)),
            acc_rows[1]);
          acc_rows[2] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_2),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
            acc_rows[2]);
          acc_rows[3] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_3),
            _mm256_mul_ps(col_scale_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
            acc_rows[3]);

          __m256i iacc_row_min_0 = _mm256_madd_epi16(
            _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
          __m256i iacc_row_min_1 = _mm256_madd_epi16(
            _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
          __m256i iacc_row_min_2 = _mm256_madd_epi16(
            _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
          __m256i iacc_row_min_3 = _mm256_madd_epi16(
            _mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

          acc_min_rows[0] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_min_0),
            _mm256_mul_ps(col_dmin_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)),
            acc_min_rows[0]);
          acc_min_rows[1] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_min_1),
            _mm256_mul_ps(col_dmin_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)),
            acc_min_rows[1]);
          acc_min_rows[2] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_min_2),
            _mm256_mul_ps(col_dmin_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)),
            acc_min_rows[2]);
          acc_min_rows[3] = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_row_min_3),
            _mm256_mul_ps(col_dmin_f32,
                          _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)),
            acc_min_rows[3]);
        }
      }

      // Store the accumulated values
      for (int i = 0; i < 4; i++) {
        _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)),
                         _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
      }
    }
  }

#else

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
#endif
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

#if defined(__AVX2__)
  // Lookup table to convert signed nibbles to signed bytes
  __m256i signextendlut = _mm256_castsi128_si256(
    _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
  signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
  // Shuffle masks to rearrange delta and scale values to multiply with
  // appropriate scales
  __m128i deltamask =
    _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
  __m128i scalemask =
    _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
  // Permute mask used for easier vector processing at later stages
  __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

  // Mask to extract nibbles from bytes
  const __m256i m4b = _mm256_set1_epi8(0x0F);

  int64_t b_nb = n / QK_K;

  const block_q4_Kx8 *b_ptr_start = (const block_q4_Kx8 *)vx;
  const block_q8_K *a_ptr_start = (const block_q8_K *)vy;

  // Process Q8_K blocks one by one
  for (int64_t y = 0; y < nr; y++) {

    // Pointers to LHS blocks of block_q8_K format
    const block_q8_K *a_ptr = a_ptr_start + (y * nb);

    // Take group of eight interleaved block_q4_K structures at each pass of the
    // loop and perform dot product operation
    for (int64_t x = 0; x < nc / 8; x++) {

      // Pointers to RHS blocks
      const block_q4_Kx8 *b_ptr = b_ptr_start + (x * b_nb);

      // Master FP accumulators
      __m256 acc_row = _mm256_setzero_ps();
      __m256 acc_min_rows = _mm256_setzero_ps();

      for (int64_t b = 0; b < nb; b++) {

        // Load and convert to FP32 scale from block_q8_K
        const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));

        // Load the scale values for the 8 blocks interleaved in block_q4_Kx8
        // col_scale_f32 rearranged so as to multiply with appropriate quants
        const __m256 col_scale_f32 =
          GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, deltamask);
        const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

        __m256i iacc_b = _mm256_setzero_si256();
        __m256i iacc_min_b = _mm256_setzero_si256();

        const __m256i q8sums =
          _mm256_loadu_si256((const __m256i *)(a_ptr[b].bsums));
        __m256i q8s = _mm256_castsi128_si256(_mm_hadd_epi16(
          _mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1)));
        q8s = _mm256_permute2f128_si256(q8s, q8s, 0);

        // Processes two sub blocks from each Q4_K in each iteration
        for (int sb = 0; sb < QK_K / 64; sb++) {

          // Load the eight block_q4_K for two sub blocks quantized values
          // interleaved with each other in chunks of eight - B0,B1 ....B6,B7
          const __m256i rhs_raw_vec_0123_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + sb * 256));
          const __m256i rhs_raw_vec_4567_0 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32 + sb * 256));
          const __m256i rhs_raw_vec_0123_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64 + sb * 256));
          const __m256i rhs_raw_vec_4567_1 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96 + sb * 256));
          const __m256i rhs_raw_vec_0123_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 128 + sb * 256));
          const __m256i rhs_raw_vec_4567_2 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 160 + sb * 256));
          const __m256i rhs_raw_vec_0123_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 192 + sb * 256));
          const __m256i rhs_raw_vec_4567_3 =
            _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 224 + sb * 256));

          // 4-bit -> 8-bit
          // Values of the first sub block of eight block_q4_K structures for
          // the sb loop
          const __m256i rhs_vec_0123_00 =
            _mm256_and_si256(rhs_raw_vec_0123_0, m4b);
          const __m256i rhs_vec_4567_00 =
            _mm256_and_si256(rhs_raw_vec_4567_0, m4b);
          const __m256i rhs_vec_0123_01 =
            _mm256_and_si256(rhs_raw_vec_0123_1, m4b);
          const __m256i rhs_vec_4567_01 =
            _mm256_and_si256(rhs_raw_vec_4567_1, m4b);
          const __m256i rhs_vec_0123_02 =
            _mm256_and_si256(rhs_raw_vec_0123_2, m4b);
          const __m256i rhs_vec_4567_02 =
            _mm256_and_si256(rhs_raw_vec_4567_2, m4b);
          const __m256i rhs_vec_0123_03 =
            _mm256_and_si256(rhs_raw_vec_0123_3, m4b);
          const __m256i rhs_vec_4567_03 =
            _mm256_and_si256(rhs_raw_vec_4567_3, m4b);

          // Values of the second sub block of eight block_q4_K structures when
          // sb = 1
          const __m256i rhs_vec_0123_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
          const __m256i rhs_vec_4567_10 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
          const __m256i rhs_vec_0123_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
          const __m256i rhs_vec_4567_11 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
          const __m256i rhs_vec_0123_12 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
          const __m256i rhs_vec_4567_12 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
          const __m256i rhs_vec_0123_13 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
          const __m256i rhs_vec_4567_13 =
            _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

          uint32_t utmp_0[4], utmp_1[4];

          // Scales and Mins of corresponding sub blocks from different Q8_K
          // structures are stored together The below block is for eg to extract
          // first sub block's scales and mins from different Q4_K structures
          // for the sb loop
          memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
          utmp_0[3] =
            ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_0 = utmp_0[1] & kmask1;
          utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
          utmp_0[2] = uaux_0;
          utmp_0[0] &= kmask1;

          // The below block is for eg to extract second sub block's scales and
          // mins from different Q4_K structures for the sb loop
          memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
          utmp_1[3] =
            ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
          const uint32_t uaux_1 = utmp_1[1] & kmask1;
          utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
          utmp_1[2] = uaux_1;
          utmp_1[0] &= kmask1;

          // Scales of first sub block in the sb loop
          const __m128i mins_and_scales_0 =
            _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
          __m128i scales_rearrange_0 =
            _mm_shuffle_epi8(mins_and_scales_0, scalemask);
          __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

          // Scales of second sub block in the sb loop
          __m128i mins_and_scales_1 =
            _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
          __m128i scales_rearrange_1 =
            _mm_shuffle_epi8(mins_and_scales_1, scalemask);
          __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

          // Mins of first and second sub block of Q4_K block are arranged side
          // by side
          __m256i mins_01 = _mm256_cvtepu8_epi16(
            _mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78),
                              _mm_shuffle_epi32(mins_and_scales_1, 78)));

          // Load the two sub block values corresponding to sb in block_q8_K in
          // batches of 16 bytes and replicate the same across 256 bit vector
          __m256i lhs_vec_00 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 64)));
          __m256i lhs_vec_01 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16 + sb * 64)));
          __m256i lhs_vec_10 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 32 + sb * 64)));
          __m256i lhs_vec_11 = _mm256_castsi128_si256(
            _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 48 + sb * 64)));

          lhs_vec_00 = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
          lhs_vec_01 = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
          lhs_vec_10 = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
          lhs_vec_11 = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

          // Dot product done within 32 bit lanes and accumulated in the same
          // vector First done for first sub block and thenn for second sub
          // block in each sb B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3)
          // B3(0-3) B7(0-3) with A0(0-3) B0(4-7) B4(4-7) B1(4-7) B5(4-7)
          // B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
          // ...........................................................................
          // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31)
          // B3(28-31) B7(28-31) with A0(28-31)

          __m256i iacc_0 = _mm256_setzero_si256();
          __m256i iacc_1 = _mm256_setzero_si256();

          iacc_0 = _mm256_add_epi16(
            iacc_0, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_00,
                        _mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_00, 0)));
          iacc_0 = _mm256_add_epi16(
            iacc_0,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177),
                                 rhs_vec_4567_00, 170),
              _mm256_shuffle_epi32(lhs_vec_00, 85)));

          iacc_0 = _mm256_add_epi16(
            iacc_0, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_01,
                        _mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_00, 170)));
          iacc_0 = _mm256_add_epi16(
            iacc_0,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177),
                                 rhs_vec_4567_01, 170),
              _mm256_shuffle_epi32(lhs_vec_00, 255)));

          iacc_0 = _mm256_add_epi16(
            iacc_0, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_02,
                        _mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_01, 0)));
          iacc_0 = _mm256_add_epi16(
            iacc_0,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177),
                                 rhs_vec_4567_02, 170),
              _mm256_shuffle_epi32(lhs_vec_01, 85)));

          iacc_0 = _mm256_add_epi16(
            iacc_0, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_03,
                        _mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_01, 170)));
          iacc_0 = _mm256_add_epi16(
            iacc_0,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177),
                                 rhs_vec_4567_03, 170),
              _mm256_shuffle_epi32(lhs_vec_01, 255)));

          iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

          iacc_1 = _mm256_add_epi16(
            iacc_1, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_10,
                        _mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_10, 0)));
          iacc_1 = _mm256_add_epi16(
            iacc_1,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177),
                                 rhs_vec_4567_10, 170),
              _mm256_shuffle_epi32(lhs_vec_10, 85)));

          iacc_1 = _mm256_add_epi16(
            iacc_1, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_11,
                        _mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_10, 170)));
          iacc_1 = _mm256_add_epi16(
            iacc_1,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177),
                                 rhs_vec_4567_11, 170),
              _mm256_shuffle_epi32(lhs_vec_10, 255)));

          iacc_1 = _mm256_add_epi16(
            iacc_1, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_12,
                        _mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_11, 0)));
          iacc_1 = _mm256_add_epi16(
            iacc_1,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177),
                                 rhs_vec_4567_12, 170),
              _mm256_shuffle_epi32(lhs_vec_11, 85)));

          iacc_1 = _mm256_add_epi16(
            iacc_1, _mm256_maddubs_epi16(
                      _mm256_blend_epi32(
                        rhs_vec_0123_13,
                        _mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170),
                      _mm256_shuffle_epi32(lhs_vec_11, 170)));
          iacc_1 = _mm256_add_epi16(
            iacc_1,
            _mm256_maddubs_epi16(
              _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177),
                                 rhs_vec_4567_13, 170),
              _mm256_shuffle_epi32(lhs_vec_11, 255)));

          iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

          // Accumulate the iacc value for one sb
          __m256i iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

          // Broadcast the bsums of the two sub blocks  of the iteration of Q8_K
          // across the vector Multiply-Add with corresponding mins of Q4_Kx8
          // with bsums
          __m256i q8s_sb = _mm256_shuffle_epi32(q8s, 0);
          __m256i iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
          q8s = _mm256_bsrli_epi128(q8s, 4);

          // Accumulate for the complete block
          iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
          iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
        }

        // Multiply-Add with scale values for the complete super block
        acc_row =
          _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b),
                          _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
        acc_min_rows = _mm256_fmadd_ps(
          _mm256_cvtepi32_ps(iacc_min_b),
          _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);
      }

      // Accumulated output values permuted so as to be stored in appropriate
      // order post accumulation
      acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
      _mm256_storeu_ps(s + (y * nr + x * 8),
                       _mm256_sub_ps(acc_row, acc_min_rows));
    }
  }

#else

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
#endif
}

void nntr_quantize_mat_q8_K_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  assert(QK_K == 256);
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  block_q8_Kx4 *__restrict y = (block_q8_Kx4 *)vy;

#if defined(__AVX2__)
  float iscale[4];
  __m256 srcv[4][32];
  __m256 iscale_vec[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      // Load elements into 4 AVX vectors
      __m256 v0 = _mm256_loadu_ps(x + row_iter * k + i * 256);
      __m256 v1 = _mm256_loadu_ps(x + row_iter * k + i * 256 + 8);
      __m256 v2 = _mm256_loadu_ps(x + row_iter * k + i * 256 + 16);
      __m256 v3 = _mm256_loadu_ps(x + row_iter * k + i * 256 + 24);

      // Compute max(abs(e)) for the block
      const __m256 signBit = _mm256_set1_ps(-0.0f);
      __m256 abs0 = _mm256_andnot_ps(signBit, v0);
      __m256 abs1 = _mm256_andnot_ps(signBit, v1);
      __m256 abs2 = _mm256_andnot_ps(signBit, v2);
      __m256 abs3 = _mm256_andnot_ps(signBit, v3);

      __m256 maxAbs = _mm256_max_ps(abs0, abs1);
      maxAbs = _mm256_max_ps(maxAbs, abs2);
      maxAbs = _mm256_max_ps(maxAbs, abs3);

      __m256 mask0 = _mm256_cmp_ps(maxAbs, v0, _CMP_EQ_OQ);
      __m256 mask1 = _mm256_cmp_ps(maxAbs, v1, _CMP_EQ_OQ);
      __m256 mask2 = _mm256_cmp_ps(maxAbs, v2, _CMP_EQ_OQ);
      __m256 mask3 = _mm256_cmp_ps(maxAbs, v3, _CMP_EQ_OQ);

      __m256 maskAbs =
        _mm256_or_ps(_mm256_or_ps(mask0, mask1), _mm256_or_ps(mask2, mask3));

      srcv[row_iter][0] = v0;
      srcv[row_iter][1] = v1;
      srcv[row_iter][2] = v2;
      srcv[row_iter][3] = v3;

      for (int sb = 1; sb < 8; sb++) {
        // Temporarily stores absolute quant values
        __m256 tempAbs = maxAbs;

        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps(x + row_iter * k + i * 256 + sb * 32);
        __m256 v1 = _mm256_loadu_ps(x + row_iter * k + i * 256 + sb * 32 + 8);
        __m256 v2 = _mm256_loadu_ps(x + row_iter * k + i * 256 + sb * 32 + 16);
        __m256 v3 = _mm256_loadu_ps(x + row_iter * k + i * 256 + sb * 32 + 24);

        // Compute max(abs(e)) for the block
        __m256 abs0 = _mm256_andnot_ps(signBit, v0);
        __m256 abs1 = _mm256_andnot_ps(signBit, v1);
        __m256 abs2 = _mm256_andnot_ps(signBit, v2);
        __m256 abs3 = _mm256_andnot_ps(signBit, v3);

        maxAbs = _mm256_max_ps(maxAbs, abs0);
        maxAbs = _mm256_max_ps(maxAbs, abs1);
        maxAbs = _mm256_max_ps(maxAbs, abs2);
        maxAbs = _mm256_max_ps(maxAbs, abs3);

        __m256 mask_prev = _mm256_cmp_ps(tempAbs, maxAbs, _CMP_EQ_OQ);
        maskAbs = _mm256_and_ps(maskAbs, mask_prev);

        mask0 = _mm256_cmp_ps(maxAbs, v0, _CMP_EQ_OQ);
        mask1 = _mm256_cmp_ps(maxAbs, v1, _CMP_EQ_OQ);
        mask2 = _mm256_cmp_ps(maxAbs, v2, _CMP_EQ_OQ);
        mask3 = _mm256_cmp_ps(maxAbs, v3, _CMP_EQ_OQ);

        __m256 mask_curr =
          _mm256_or_ps(_mm256_or_ps(mask0, mask1), _mm256_or_ps(mask2, mask3));
        maskAbs = _mm256_or_ps(maskAbs, mask_curr);

        srcv[row_iter][sb * 4] = v0;
        srcv[row_iter][sb * 4 + 1] = v1;
        srcv[row_iter][sb * 4 + 2] = v2;
        srcv[row_iter][sb * 4 + 3] = v3;
      }

      __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1),
                               _mm256_castps256_ps128(maxAbs));
      max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
      max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
      const float maxScalar = _mm_cvtss_f32(max4);

      __m256 maxScalarVec = _mm256_set1_ps(maxScalar);

      __m256 mask_next = _mm256_cmp_ps(maxScalarVec, maxAbs, _CMP_EQ_OQ);
      __m256 finalMask = _mm256_and_ps(maskAbs, mask_next);

      const int mask = _mm256_movemask_ps(finalMask);
      iscale[row_iter] = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;

      if (mask) {
        iscale[row_iter] = (maxScalar != 0.0f) ? -127.f / maxScalar : 0.0f;
      }

      y[i].d[row_iter] = maxScalar ? 1 / iscale[row_iter] : 0;
      iscale_vec[row_iter] = _mm256_set1_ps(iscale[row_iter]);
    }

    __m256i quants_interleaved[32];
    for (int j = 0; j < 32; j++) {
      // Apply the multiplier
      __m256 v0 = _mm256_mul_ps(srcv[0][j], iscale_vec[0]);
      __m256 v1 = _mm256_mul_ps(srcv[1][j], iscale_vec[1]);
      __m256 v2 = _mm256_mul_ps(srcv[2][j], iscale_vec[2]);
      __m256 v3 = _mm256_mul_ps(srcv[3][j], iscale_vec[3]);

      // Round to nearest integer
      v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
      v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
      v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
      v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

      // Convert floats to integers
      __m256i i0 = _mm256_cvtps_epi32(v0);
      __m256i i1 = _mm256_cvtps_epi32(v1);
      __m256i i2 = _mm256_cvtps_epi32(v2);
      __m256i i3 = _mm256_cvtps_epi32(v3);

      // Convert int32 to int16
      i0 = _mm256_packs_epi32(i0, i1);
      i2 = _mm256_packs_epi32(i2, i3);
      // Convert int16 to int8
      i0 = _mm256_packs_epi16(i0, i2);

      //  Permute and store the quantized weights in the required order after
      //  the pack instruction
      const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
      i0 = _mm256_permutevar8x32_epi32(i0, perm);

      _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
      quants_interleaved[j] = i0;
    }

    // Masks to shuffle the quants of corresonding sub blocks for rearraning
    // quants for vectorized bsums computation
    __m256i shuffle_mask_sb2 = _mm256_castsi128_si256(
      _mm_setr_epi8(0, 1, 0, 1, 4, 5, 6, 7, 8, 9, 8, 9, 12, 13, 14, 15));
    shuffle_mask_sb2 =
      _mm256_permute2f128_si256(shuffle_mask_sb2, shuffle_mask_sb2, 0);
    __m256i shuffle_mask_sb3 = _mm256_castsi128_si256(
      _mm_setr_epi8(0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 8, 9, 14, 15));
    shuffle_mask_sb3 =
      _mm256_permute2f128_si256(shuffle_mask_sb3, shuffle_mask_sb3, 0);
    __m256i shuffle_mask_sb4 = _mm256_castsi128_si256(
      _mm_setr_epi8(0, 1, 2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 12, 13, 8, 9));
    shuffle_mask_sb4 =
      _mm256_permute2f128_si256(shuffle_mask_sb4, shuffle_mask_sb4, 0);

    for (int k = 0; k < 4; k++) {
      // Quants from four different sub blocks are taken
      __m256i q0 = quants_interleaved[k * 8 + 0];
      __m256i q1 = quants_interleaved[k * 8 + 1];
      __m256i q2 = quants_interleaved[k * 8 + 2];
      __m256i q3 = quants_interleaved[k * 8 + 3];
      __m256i q4 = quants_interleaved[k * 8 + 4];
      __m256i q5 = quants_interleaved[k * 8 + 5];
      __m256i q6 = quants_interleaved[k * 8 + 6];
      __m256i q7 = quants_interleaved[k * 8 + 7];

      // The below code block has the first half of different sub blocks
      // shuffled and blended so as to process 2 values from each sub block at a
      // time
      __m256i sb2_h1_shuffled = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
      __m256i sb_h1_interleaved = _mm256_blend_epi16(q0, sb2_h1_shuffled, 34);
      __m256i sb3_h1_shuffled = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
      sb_h1_interleaved =
        _mm256_blend_epi16(sb_h1_interleaved, sb3_h1_shuffled, 68);
      __m256i sb4_h1_shuffled = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
      sb_h1_interleaved =
        _mm256_blend_epi16(sb_h1_interleaved, sb4_h1_shuffled, 136);

      __m256i one = _mm256_set1_epi8(1);
      __m256i bsums_r1 = _mm256_maddubs_epi16(one, sb_h1_interleaved);

      for (int l = 0; l < 3; l++) {
        // Quants value shifted to process next two values from each sub block
        q0 = _mm256_srli_epi64(q0, 16);
        q2 = _mm256_srli_epi64(q2, 16);
        q4 = _mm256_srli_epi64(q4, 16);
        q6 = _mm256_srli_epi64(q6, 16);

        sb2_h1_shuffled = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
        sb_h1_interleaved = _mm256_blend_epi16(q0, sb2_h1_shuffled, 34);
        sb3_h1_shuffled = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
        sb_h1_interleaved =
          _mm256_blend_epi16(sb_h1_interleaved, sb3_h1_shuffled, 68);
        sb4_h1_shuffled = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
        sb_h1_interleaved =
          _mm256_blend_epi16(sb_h1_interleaved, sb4_h1_shuffled, 136);

        bsums_r1 = _mm256_add_epi16(
          bsums_r1, _mm256_maddubs_epi16(one, sb_h1_interleaved));
      }

      // The below code block has the second half of different sub blocks
      // shuffled and blended so as to process 2 values from each sub block at a
      // time
      __m256i sb2_h2_shuffled = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
      __m256i sb_h2_interleaved = _mm256_blend_epi16(q1, sb2_h2_shuffled, 34);
      __m256i sb3_h2_shuffled = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
      sb_h2_interleaved =
        _mm256_blend_epi16(sb_h2_interleaved, sb3_h2_shuffled, 68);
      __m256i sb4_h2_shuffled = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
      sb_h2_interleaved =
        _mm256_blend_epi16(sb_h2_interleaved, sb4_h2_shuffled, 136);

      __m256i bsums_r2 = _mm256_maddubs_epi16(one, sb_h2_interleaved);

      for (int l = 0; l < 3; l++) {
        // Quants value shifted to process next two values from each sub block
        q1 = _mm256_srli_epi64(q1, 16);
        q3 = _mm256_srli_epi64(q3, 16);
        q5 = _mm256_srli_epi64(q5, 16);
        q7 = _mm256_srli_epi64(q7, 16);

        sb2_h2_shuffled = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
        sb_h2_interleaved = _mm256_blend_epi16(q1, sb2_h2_shuffled, 34);
        sb3_h2_shuffled = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
        sb_h2_interleaved =
          _mm256_blend_epi16(sb_h2_interleaved, sb3_h2_shuffled, 68);
        sb4_h2_shuffled = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
        sb_h2_interleaved =
          _mm256_blend_epi16(sb_h2_interleaved, sb4_h2_shuffled, 136);

        bsums_r2 = _mm256_add_epi16(
          bsums_r2, _mm256_maddubs_epi16(one, sb_h2_interleaved));
      }

      // Overall bsums in interleaved fashion computed by adding results of both
      // halves
      __m256i bsums_r = _mm256_add_epi16(bsums_r1, bsums_r2);
      _mm256_storeu_si256((__m256i *)(y[i].bsums + 16 * k), bsums_r);
    }
  }

#else

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
#endif
}

void nntr_quantize_mat_q8_0_4x8(const float *__restrict x, void *__restrict vy,
                                int64_t k) {
  assert(Q8_0 == 32);
  assert(k % Q8_0 == 0);
  const int nb = k / Q8_0;

  block_q8_0x4 *__restrict y = (block_q8_0x4 *)vy;

#if defined(__ARM_NEON)
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
#elif defined(__AVX2__) || defined(__AVX__)
  float id[4];
  __m256 srcv[4][4];
  __m256 idvec[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      // Load elements into 4 AVX vectors
      __m256 v0 = _mm256_loadu_ps(x + row_iter * k + i * 32);
      __m256 v1 = _mm256_loadu_ps(x + row_iter * k + i * 32 + 8);
      __m256 v2 = _mm256_loadu_ps(x + row_iter * k + i * 32 + 16);
      __m256 v3 = _mm256_loadu_ps(x + row_iter * k + i * 32 + 24);

      // Compute max(abs(e)) for the block
      const __m256 signBit = _mm256_set1_ps(-0.0f);
      __m256 maxAbs = _mm256_andnot_ps(signBit, v0);
      maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v1));
      maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v2));
      maxAbs = _mm256_max_ps(maxAbs, _mm256_andnot_ps(signBit, v3));

      __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maxAbs, 1),
                               _mm256_castps256_ps128(maxAbs));
      max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
      max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
      const float maxScalar = _mm_cvtss_f32(max4);

      // Divided by 127.f to mirror results in quantize_row_q8_0
      const float d = maxScalar / 127.f;
      id[row_iter] =
        (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f; // d ? 1.0f / d : 0.0f;

      // Store the scale for the individual block
      y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);

      // Store the values in blocks of eight values - Aim is to use these later
      // for block interleaving
      srcv[row_iter][0] = v0;
      srcv[row_iter][1] = v1;
      srcv[row_iter][2] = v2;
      srcv[row_iter][3] = v3;
      idvec[row_iter] = _mm256_set1_ps(id[row_iter]);
    }

    // The loop iterates four times - The aim is to get 4 corresponding chunks
    // of eight bytes from the original weight blocks that are interleaved
    for (int j = 0; j < 4; j++) {
      // Apply the multiplier
      __m256 v0 = _mm256_mul_ps(srcv[0][j], idvec[0]);
      __m256 v1 = _mm256_mul_ps(srcv[1][j], idvec[1]);
      __m256 v2 = _mm256_mul_ps(srcv[2][j], idvec[2]);
      __m256 v3 = _mm256_mul_ps(srcv[3][j], idvec[3]);

      // Round to nearest integer
      v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
      v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);
      v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST);
      v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST);

      // Convert floats to integers
      __m256i i0 = _mm256_cvtps_epi32(v0);
      __m256i i1 = _mm256_cvtps_epi32(v1);
      __m256i i2 = _mm256_cvtps_epi32(v2);
      __m256i i3 = _mm256_cvtps_epi32(v3);

#if defined(__AVX2__)
      // Convert int32 to int16
      i0 = _mm256_packs_epi32(i0, i1);
      i2 = _mm256_packs_epi32(i2, i3);
      // Convert int16 to int8
      i0 = _mm256_packs_epi16(i0, i2);

      //  Permute and store the quantized weights in the required order after
      //  the pack instruction
      const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
      i0 = _mm256_permutevar8x32_epi32(i0, perm);

      _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
#else
      // Since we don't have in AVX some necessary functions,
      // we split the registers in half and call AVX2 analogs from SSE
      __m128i ni0 = _mm256_castsi256_si128(i0);
      __m128i ni1 = _mm256_extractf128_si256(i0, 1);
      __m128i ni2 = _mm256_castsi256_si128(i1);
      __m128i ni3 = _mm256_extractf128_si256(i1, 1);
      __m128i ni4 = _mm256_castsi256_si128(i2);
      __m128i ni5 = _mm256_extractf128_si256(i2, 1);
      __m128i ni6 = _mm256_castsi256_si128(i3);
      __m128i ni7 = _mm256_extractf128_si256(i3, 1);

      // Convert int32 to int16
      ni0 = _mm_packs_epi32(ni0, ni1);
      ni2 = _mm_packs_epi32(ni2, ni3);
      ni4 = _mm_packs_epi32(ni4, ni5);
      ni6 = _mm_packs_epi32(ni6, ni7);
      // Convert int16 to int8
      ni0 = _mm_packs_epi16(ni0, ni2);
      ni4 = _mm_packs_epi16(ni4, ni6);
      _mm_storeu_si128((__m128i *)(y[i].qs + 32 * j), ni0);
      _mm_storeu_si128((__m128i *)(y[i].qs + 32 * j + 16), ni4);
#endif
    }
  }
#else
  // scalar
  const int blck_size_interleave = 8;
  float srcv[4][Q8_0];
  float id[4];

  for (int i = 0; i < nb; i++) {
    for (int row_iter = 0; row_iter < 4; row_iter++) {
      float amax = 0.0f; // absolute max

      for (int j = 0; j < Q8_0; j++) {
        srcv[row_iter][j] = x[row_iter * k + i * Q8_0 + j];
        amax = MAX(amax, fabsf(srcv[row_iter][j]));
      }

      const float d = amax / ((1 << 7) - 1);
      id[row_iter] = d ? 1.0f / d : 0.0f;

      y[i].d[row_iter] = nntr_compute_fp32_to_fp16(d);
    }

    for (int j = 0; j < Q8_0 * 4; j++) {
      int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
      int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
      src_offset += (j % blck_size_interleave);

      float x0 = srcv[src_id][src_offset] * id[src_id];
      y[i].qs[j] = roundf(x0);
    }
  }
#endif
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

//
// Helper functions
//
#if __AVX__ || __AVX2__ || __AVX512F__

static inline __m128i get_scale_shuffle(int i) {
  static const uint8_t k_shuffle[128] = {
    0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,
    2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,
    4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,
    7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,
    9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11,
    11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14,
    14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
  return _mm_loadu_si128((const __m128i *)k_shuffle + i);
}
#elif defined(__loongarch_asx)
static inline __m128i get_scale_shuffle(int i) {
  static const uint8_t k_shuffle[128] = {
    0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,
    2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,  4,  4,  4,  4,  4,  4,
    4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  7,
    7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,
    9,  9,  9,  9,  10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11,
    11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14,
    14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15};
  return __lsx_vld((const __m128i *)k_shuffle + i, 0);
}
#endif

void nntr_vec_dot_q6_K_q8_K(int n, float *__restrict s, size_t bs,
                            const void *__restrict vx, size_t bx,
                            const void *__restrict vy, size_t by, int nrc) {
  assert(n % QK_K == 0);
  assert(nrc == 1);

  const block_q6_K *__restrict x = (const block_q6_K *)vx;
  const block_q8_K *__restrict y = (const block_q8_K *)vy;

  const int nb = n / QK_K;

#ifdef __ARM_FEATURE_SVE
  const int vector_length = QK8_0 * 8;
  float sum = 0;
  svuint8_t m4b = svdup_n_u8(0xf);
  svint32_t vzero = svdup_n_s32(0);
  svuint8_t mone = svdup_n_u8(0x30);
  svint8_t q6bytes_1, q6bytes_2, q6bytes_3, q6bytes_4;
  svuint8_t q6h_1, q6h_2, q6h_3, q6h_4;

  for (int i = 0; i < nb; ++i) {
    const float d_all = nntr_compute_fp16_to_fp32(x[i].d);

    const uint8_t *__restrict q6 = x[i].ql;
    const uint8_t *__restrict qh = x[i].qh;
    const int8_t *__restrict q8 = y[i].qs;

    const int8_t *__restrict scale = x[i].scales;

    const svbool_t pg16_8 = svptrue_pat_b16(SV_VL8);
    const svint16_t q8sums_1 = svld1_s16(pg16_8, y[i].bsums);
    const svint16_t q8sums_2 = svld1_s16(pg16_8, y[i].bsums + 8);
    const svint16_t q6scales_1 =
      svunpklo_s16(svld1_s8(svptrue_pat_b8(SV_VL8), scale));
    const svint16_t q6scales_2 =
      svunpklo_s16(svld1_s8(svptrue_pat_b8(SV_VL8), scale + 8));
    const svint64_t prod = svdup_n_s64(0);
    int32_t isum_mins = svaddv_s64(
      svptrue_b64(),
      svadd_s64_x(svptrue_b64(), svdot_s64(prod, q8sums_1, q6scales_1),
                  svdot_s64(prod, q8sums_2, q6scales_2)));
    int32_t isum = 0;

    switch (vector_length) {
    case 128: {
      const svbool_t pg32_4 = svptrue_pat_b32(SV_VL4);
      const svbool_t pg8_16 = svptrue_pat_b8(SV_VL16);
      svint32_t isum_tmp = svdup_n_s32(0);
      for (int j = 0; j < QK_K / 128; ++j) {
        svuint8_t qhbits_1 = svld1_u8(pg8_16, qh);
        svuint8_t qhbits_2 = svld1_u8(pg8_16, qh + 16);
        qh += 32;
        svuint8_t q6bits_1 = svld1_u8(pg8_16, q6);
        svuint8_t q6bits_2 = svld1_u8(pg8_16, q6 + 16);
        svuint8_t q6bits_3 = svld1_u8(pg8_16, q6 + 32);
        svuint8_t q6bits_4 = svld1_u8(pg8_16, q6 + 48);
        q6 += 64;
        svint8_t q8bytes_1 = svld1_s8(pg8_16, q8);
        svint8_t q8bytes_2 = svld1_s8(pg8_16, q8 + 16);
        svint8_t q8bytes_3 = svld1_s8(pg8_16, q8 + 32);
        svint8_t q8bytes_4 = svld1_s8(pg8_16, q8 + 48);
        q8 += 64;

        q6h_1 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_1, 4));
        q6h_2 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_2, 4));
        q6h_3 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_1, 2));
        q6h_4 = svand_u8_x(pg16_8, mone, svlsl_n_u8_x(pg16_8, qhbits_2, 2));
        q6bytes_1 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_1, m4b), q6h_1));
        q6bytes_2 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_2, m4b), q6h_2));
        q6bytes_3 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_3, m4b), q6h_3));
        q6bytes_4 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svand_u8_x(pg8_16, q6bits_4, m4b), q6h_4));
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_1, q8bytes_1), scale[0]);
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_2, q8bytes_2), scale[1]);
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_3, q8bytes_3), scale[2]);
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_4, q8bytes_4), scale[3]);

        scale += 4;
        q8bytes_1 = svld1_s8(pg8_16, q8);
        q8bytes_2 = svld1_s8(pg8_16, q8 + 16);
        q8bytes_3 = svld1_s8(pg8_16, q8 + 32);
        q8bytes_4 = svld1_s8(pg8_16, q8 + 48);
        q8 += 64;

        q6h_1 = svand_u8_x(pg16_8, mone, qhbits_1);
        q6h_2 = svand_u8_x(pg16_8, mone, qhbits_2);
        q6h_3 = svand_u8_x(pg16_8, mone, svlsr_n_u8_x(pg16_8, qhbits_1, 2));
        q6h_4 = svand_u8_x(pg16_8, mone, svlsr_n_u8_x(pg16_8, qhbits_2, 2));
        q6bytes_1 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_1, 4), q6h_1));
        q6bytes_2 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_2, 4), q6h_2));
        q6bytes_3 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_3, 4), q6h_3));
        q6bytes_4 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_16, svlsr_n_u8_x(pg8_16, q6bits_4, 4), q6h_4));
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_1, q8bytes_1), scale[0]);
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_2, q8bytes_2), scale[1]);
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_3, q8bytes_3), scale[2]);
        isum_tmp = svmla_n_s32_x(
          pg32_4, isum_tmp, svdot_s32(vzero, q6bytes_4, q8bytes_4), scale[3]);
        scale += 4;
      }
      isum += svaddv_s32(pg32_4, isum_tmp);
      sum += d_all * y[i].d * (isum - 32 * isum_mins);
    } break;
    case 256:
    case 512: {
      const svbool_t pg8_2 = svptrue_pat_b8(SV_VL2);
      const svbool_t pg32_8 = svptrue_pat_b32(SV_VL8);
      const svbool_t pg8_32 = svptrue_pat_b8(SV_VL32);
      svint32_t isum_tmp = svdup_n_s32(0);
      for (int j = 0; j < QK_K / 128; j++) {
        svuint8_t qhbits_1 = svld1_u8(pg8_32, qh);
        qh += 32;
        svuint8_t q6bits_1 = svld1_u8(pg8_32, q6);
        svuint8_t q6bits_2 = svld1_u8(pg8_32, q6 + 32);
        q6 += 64;
        svint8_t q8bytes_1 = svld1_s8(pg8_32, q8);
        svint8_t q8bytes_2 = svld1_s8(pg8_32, q8 + 32);
        svint8_t q8bytes_3 = svld1_s8(pg8_32, q8 + 64);
        svint8_t q8bytes_4 = svld1_s8(pg8_32, q8 + 96);
        q8 += 128;
        q6h_1 = svand_u8_x(pg8_32, mone, svlsl_n_u8_x(pg8_32, qhbits_1, 4));
        q6h_2 = svand_u8_x(pg8_32, mone, svlsl_n_u8_x(pg8_32, qhbits_1, 2));
        q6h_3 = svand_u8_x(pg8_32, mone, qhbits_1);
        q6h_4 = svand_u8_x(pg8_32, mone, svlsr_n_u8_x(pg8_32, qhbits_1, 2));
        q6bytes_1 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_32, svand_u8_x(pg8_32, q6bits_1, m4b), q6h_1));
        q6bytes_2 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_32, svand_u8_x(pg8_32, q6bits_2, m4b), q6h_2));
        q6bytes_3 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_32, svlsr_n_u8_x(pg8_32, q6bits_1, 4), q6h_3));
        q6bytes_4 = svreinterpret_s8_u8(
          svorr_u8_x(pg8_32, svlsr_n_u8_x(pg8_32, q6bits_2, 4), q6h_4));

        svint8_t scale_lane_1_tmp = svld1_s8(pg8_2, scale);
        scale_lane_1_tmp = svzip1_s8(scale_lane_1_tmp, scale_lane_1_tmp);
        scale_lane_1_tmp = svzip1_s8(scale_lane_1_tmp, scale_lane_1_tmp);
        svint8_t scale_lane_2_tmp = svld1_s8(pg8_2, scale + 2);
        scale_lane_2_tmp = svzip1_s8(scale_lane_2_tmp, scale_lane_2_tmp);
        scale_lane_2_tmp = svzip1_s8(scale_lane_2_tmp, scale_lane_2_tmp);
        svint8_t scale_lane_3_tmp = svld1_s8(pg8_2, scale + 4);
        scale_lane_3_tmp = svzip1_s8(scale_lane_3_tmp, scale_lane_3_tmp);
        scale_lane_3_tmp = svzip1_s8(scale_lane_3_tmp, scale_lane_3_tmp);
        svint8_t scale_lane_4_tmp = svld1_s8(pg8_2, scale + 6);
        scale_lane_4_tmp = svzip1_s8(scale_lane_4_tmp, scale_lane_4_tmp);
        scale_lane_4_tmp = svzip1_s8(scale_lane_4_tmp, scale_lane_4_tmp);
        svint32_t scale_lane_1 = svunpklo_s32(svunpklo_s16(scale_lane_1_tmp));
        svint32_t scale_lane_2 = svunpklo_s32(svunpklo_s16(scale_lane_2_tmp));
        svint32_t scale_lane_3 = svunpklo_s32(svunpklo_s16(scale_lane_3_tmp));
        svint32_t scale_lane_4 = svunpklo_s32(svunpklo_s16(scale_lane_4_tmp));

        isum_tmp =
          svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_1, q8bytes_1),
                      scale_lane_1);
        isum_tmp =
          svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_2, q8bytes_2),
                      scale_lane_2);
        isum_tmp =
          svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_3, q8bytes_3),
                      scale_lane_3);
        isum_tmp =
          svmla_s32_x(pg32_8, isum_tmp, svdot_s32(vzero, q6bytes_4, q8bytes_4),
                      scale_lane_4);
        scale += 8;
      }
      isum += svaddv_s32(pg32_8, isum_tmp);
      sum += d_all * y[i].d * (isum - 32 * isum_mins);
    } break;
    default:
      assert(false && "Unsupported vector length");
      break;
    }
  }

  *s = sum;

#elif __ARM_NEON
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

#elif defined __AVX2__

  const __m256i m4 = _mm256_set1_epi8(0xF);
  const __m256i m2 = _mm256_set1_epi8(3);
  const __m256i m32s = _mm256_set1_epi8(32);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {

    const float d = y[i].d * nntr_compute_fp16_to_fp32(x[i].d);

    const uint8_t *__restrict q4 = x[i].ql;
    const uint8_t *__restrict qh = x[i].qh;
    const int8_t *__restrict q8 = y[i].qs;

    const __m128i scales = _mm_loadu_si128((const __m128i *)x[i].scales);

    __m256i sumi = _mm256_setzero_si256();

    int is = 0;

    for (int j = 0; j < QK_K / 128; ++j) {

      const __m128i scale_0 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
      const __m128i scale_1 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
      const __m128i scale_2 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
      const __m128i scale_3 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
      is += 4;

      const __m256i q4bits1 = _mm256_loadu_si256((const __m256i *)q4);
      q4 += 32;
      const __m256i q4bits2 = _mm256_loadu_si256((const __m256i *)q4);
      q4 += 32;
      const __m256i q4bitsH = _mm256_loadu_si256((const __m256i *)qh);
      qh += 32;

      const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
      const __m256i q4h_1 = _mm256_slli_epi16(
        _mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
      const __m256i q4h_2 = _mm256_slli_epi16(
        _mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
      const __m256i q4h_3 = _mm256_slli_epi16(
        _mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

      const __m256i q4_0 =
        _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
      const __m256i q4_1 =
        _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
      const __m256i q4_2 = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
      const __m256i q4_3 = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

      const __m256i q8_0 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;
      const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;
      const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;
      const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8);
      q8 += 32;

      __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
      __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
      __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
      __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

      __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
      __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
      __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
      __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

      p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
      p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
      p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
      p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

      p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
      p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
      p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
      p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
      sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
    }

    acc =
      _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
  }

  *s = hsum_float_8(acc);

#elif defined __AVX__

  const __m128i m3 = _mm_set1_epi8(3);
  const __m128i m15 = _mm_set1_epi8(15);

  __m256 acc = _mm256_setzero_ps();

  for (int i = 0; i < nb; ++i) {

    const float d = y[i].d * nntr_compute_fp16_to_fp32(x[i].d);

    const uint8_t *__restrict q4 = x[i].ql;
    const uint8_t *__restrict qh = x[i].qh;
    const int8_t *__restrict q8 = y[i].qs;

    // handle the q6_k -32 offset separately using bsums
    const __m128i q8sums_0 = _mm_loadu_si128((const __m128i *)y[i].bsums);
    const __m128i q8sums_1 = _mm_loadu_si128((const __m128i *)y[i].bsums + 1);
    const __m128i scales = _mm_loadu_si128((const __m128i *)x[i].scales);
    const __m128i scales_16_0 = _mm_cvtepi8_epi16(scales);
    const __m128i scales_16_1 = _mm_cvtepi8_epi16(_mm_bsrli_si128(scales, 8));
    const __m128i q8sclsub_0 =
      _mm_slli_epi32(_mm_madd_epi16(q8sums_0, scales_16_0), 5);
    const __m128i q8sclsub_1 =
      _mm_slli_epi32(_mm_madd_epi16(q8sums_1, scales_16_1), 5);

    __m128i sumi_0 = _mm_setzero_si128();
    __m128i sumi_1 = _mm_setzero_si128();

    int is = 0;

    for (int j = 0; j < QK_K / 128; ++j) {

      const __m128i q4bitsH_0 = _mm_loadu_si128((const __m128i *)qh);
      qh += 16;
      const __m128i q4bitsH_1 = _mm_loadu_si128((const __m128i *)qh);
      qh += 16;

      const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m3), 4);
      const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m3), 4);
      const __m128i q4h_2 =
        _mm_slli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(12)), 2);
      const __m128i q4h_3 =
        _mm_slli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(12)), 2);
      const __m128i q4h_4 = _mm_and_si128(q4bitsH_0, _mm_set1_epi8(48));
      const __m128i q4h_5 = _mm_and_si128(q4bitsH_1, _mm_set1_epi8(48));
      const __m128i q4h_6 =
        _mm_srli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(-64)), 2);
      const __m128i q4h_7 =
        _mm_srli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(-64)), 2);

      const __m128i q4bits1_0 = _mm_loadu_si128((const __m128i *)q4);
      q4 += 16;
      const __m128i q4bits1_1 = _mm_loadu_si128((const __m128i *)q4);
      q4 += 16;
      const __m128i q4bits2_0 = _mm_loadu_si128((const __m128i *)q4);
      q4 += 16;
      const __m128i q4bits2_1 = _mm_loadu_si128((const __m128i *)q4);
      q4 += 16;

      const __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m15), q4h_0);
      const __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m15), q4h_1);
      const __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m15), q4h_2);
      const __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m15), q4h_3);
      const __m128i q4_4 =
        _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m15), q4h_4);
      const __m128i q4_5 =
        _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m15), q4h_5);
      const __m128i q4_6 =
        _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m15), q4h_6);
      const __m128i q4_7 =
        _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m15), q4h_7);

      const __m128i q8_0 = _mm_loadu_si128((const __m128i *)q8);
      q8 += 16;
      const __m128i q8_1 = _mm_loadu_si128((const __m128i *)q8);
      q8 += 16;
      const __m128i q8_2 = _mm_loadu_si128((const __m128i *)q8);
      q8 += 16;
      const __m128i q8_3 = _mm_loadu_si128((const __m128i *)q8);
      q8 += 16;
      const __m128i q8_4 = _mm_loadu_si128((const __m128i *)q8);
      q8 += 16;
      const __m128i q8_5 = _mm_loadu_si128((const __m128i *)q8);
      q8 += 16;
      const __m128i q8_6 = _mm_loadu_si128((const __m128i *)q8);
      q8 += 16;
      const __m128i q8_7 = _mm_loadu_si128((const __m128i *)q8);
      q8 += 16;

      __m128i p16_0 = _mm_maddubs_epi16(q4_0, q8_0);
      __m128i p16_1 = _mm_maddubs_epi16(q4_1, q8_1);
      __m128i p16_2 = _mm_maddubs_epi16(q4_2, q8_2);
      __m128i p16_3 = _mm_maddubs_epi16(q4_3, q8_3);
      __m128i p16_4 = _mm_maddubs_epi16(q4_4, q8_4);
      __m128i p16_5 = _mm_maddubs_epi16(q4_5, q8_5);
      __m128i p16_6 = _mm_maddubs_epi16(q4_6, q8_6);
      __m128i p16_7 = _mm_maddubs_epi16(q4_7, q8_7);

      const __m128i scale_0 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
      const __m128i scale_1 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
      const __m128i scale_2 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
      const __m128i scale_3 =
        _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
      is += 4;

      p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
      p16_1 =
        _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_0, 8)), p16_1);
      p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
      p16_3 =
        _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_1, 8)), p16_3);
      p16_4 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_2), p16_4);
      p16_5 =
        _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_2, 8)), p16_5);
      p16_6 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_3), p16_6);
      p16_7 =
        _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_3, 8)), p16_7);

      sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
      sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));
      sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_4, p16_6));
      sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_5, p16_7));
    }

    sumi_0 = _mm_sub_epi32(sumi_0, q8sclsub_0);
    sumi_1 = _mm_sub_epi32(sumi_1, q8sclsub_1);
    const __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
    acc = _mm256_add_ps(
      _mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), acc);
  }

  *s = hsum_float_8(acc);

#elif defined __wasm_simd128__
  int8_t aux8[QK_K] __attribute__((aligned(16)));
  int32_t aux32[8] __attribute__((aligned(16))) = {0};
  float sums[8] __attribute__((aligned(16))) = {0};

  for (int i = 0; i < nb; ++i) {
    // Unpack 6-bit quantized data into aux8 (unchanged)
    const uint8_t *__restrict q4 = x[i].ql;
    const uint8_t *__restrict qh = x[i].qh;
    int8_t *a = aux8;
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        a[l + 0] = (int8_t)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        a[l + 32] =
          (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        a[l + 64] = (int8_t)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        a[l + 96] =
          (int8_t)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      }
      a += 128;
      q4 += 64;
      qh += 32;
    }

    const int8_t *__restrict a_ptr = aux8;
    const int8_t *__restrict q8 = y[i].qs;
    v128_t acc0 = wasm_i32x4_splat(0);
    v128_t acc1 = wasm_i32x4_splat(0);

    for (int j = 0; j < QK_K / 16; ++j) {
      const int scale = x[i].scales[j];
      const v128_t vscale = wasm_i32x4_splat(scale);

      // Load 16 elements from a and q8
      const v128_t a_vec = wasm_v128_load(a_ptr);
      const v128_t q8_vec = wasm_v128_load(q8);

      // Process low 8 elements
      v128_t a_low = wasm_i16x8_extend_low_i8x16(a_vec);
      v128_t q8_low = wasm_i16x8_extend_low_i8x16(q8_vec);
      v128_t prod_low = wasm_i16x8_mul(a_low, q8_low);
      v128_t prod_lo_lo = wasm_i32x4_extend_low_i16x8(prod_low);
      v128_t prod_lo_hi = wasm_i32x4_extend_high_i16x8(prod_low);

      // Process high 8 elements
      v128_t a_high = wasm_i16x8_extend_high_i8x16(a_vec);
      v128_t q8_high = wasm_i16x8_extend_high_i8x16(q8_vec);
      v128_t prod_high = wasm_i16x8_mul(a_high, q8_high);
      v128_t prod_hi_lo = wasm_i32x4_extend_low_i16x8(prod_high);
      v128_t prod_hi_hi = wasm_i32x4_extend_high_i16x8(prod_high);

      // Scale and accumulate
      prod_lo_lo = wasm_i32x4_mul(prod_lo_lo, vscale);
      prod_lo_hi = wasm_i32x4_mul(prod_lo_hi, vscale);
      prod_hi_lo = wasm_i32x4_mul(prod_hi_lo, vscale);
      prod_hi_hi = wasm_i32x4_mul(prod_hi_hi, vscale);

      acc0 = wasm_i32x4_add(acc0, wasm_i32x4_add(prod_lo_lo, prod_hi_lo));
      acc1 = wasm_i32x4_add(acc1, wasm_i32x4_add(prod_lo_hi, prod_hi_hi));

      a_ptr += 16;
      q8 += 16;
    }

    // Store accumulated results
    wasm_v128_store(&aux32[0], acc0);
    wasm_v128_store(&aux32[4], acc1);

    const float d = nntr_compute_fp16_to_fp32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) {
      sums[l] += d * aux32[l];
    }
  }

  // Sum final results
  float sumf = 0;
  for (int l = 0; l < 8; ++l) {
    sumf += sums[l];
  }
  *s = sumf;

#elif defined __riscv_v_intrinsic

  const int vector_length = __riscv_vlenb() * 8;
  float sumf = 0;

  switch (vector_length) {
  case 256:
    for (int i = 0; i < nb; ++i) {

      const float d = nntr_compute_fp16_to_fp32(x[i].d) * y[i].d;

      const uint8_t *__restrict q6 = x[i].ql;
      const uint8_t *__restrict qh = x[i].qh;
      const int8_t *__restrict q8 = y[i].qs;

      const int8_t *__restrict scale = x[i].scales;

      size_t vl;

      vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

      int sum_t = 0;
      int is = 0;

      for (int j = 0; j < QK_K / 128; ++j) {

        vl = 32;

        // load qh
        vuint8m1_t qh_x = __riscv_vle8_v_u8m1(qh, vl);

        // load Q6
        vuint8m1_t q6_0 = __riscv_vle8_v_u8m1(q6, vl);
        vuint8m1_t q6_1 = __riscv_vle8_v_u8m1(q6 + 32, vl);

        vuint8m1_t q6a_0 = __riscv_vand_vx_u8m1(q6_0, 0x0F, vl);
        vuint8m1_t q6a_1 = __riscv_vand_vx_u8m1(q6_1, 0x0F, vl);
        vuint8m1_t q6s_0 = __riscv_vsrl_vx_u8m1(q6_0, 0x04, vl);
        vuint8m1_t q6s_1 = __riscv_vsrl_vx_u8m1(q6_1, 0x04, vl);

        vuint8m1_t qh_0 = __riscv_vand_vx_u8m1(qh_x, 0x03, vl);
        vuint8m1_t qh_1 =
          __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x2, vl), 0x03, vl);
        vuint8m1_t qh_2 =
          __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x4, vl), 0x03, vl);
        vuint8m1_t qh_3 =
          __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x6, vl), 0x03, vl);

        vuint8m1_t qhi_0 =
          __riscv_vor_vv_u8m1(q6a_0, __riscv_vsll_vx_u8m1(qh_0, 0x04, vl), vl);
        vuint8m1_t qhi_1 =
          __riscv_vor_vv_u8m1(q6a_1, __riscv_vsll_vx_u8m1(qh_1, 0x04, vl), vl);
        vuint8m1_t qhi_2 =
          __riscv_vor_vv_u8m1(q6s_0, __riscv_vsll_vx_u8m1(qh_2, 0x04, vl), vl);
        vuint8m1_t qhi_3 =
          __riscv_vor_vv_u8m1(q6s_1, __riscv_vsll_vx_u8m1(qh_3, 0x04, vl), vl);

        vint8m1_t a_0 =
          __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_0), 32, vl);
        vint8m1_t a_1 =
          __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_1), 32, vl);
        vint8m1_t a_2 =
          __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_2), 32, vl);
        vint8m1_t a_3 =
          __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_3), 32, vl);

        // load Q8 and take product
        vint16m2_t va_q_0 =
          __riscv_vwmul_vv_i16m2(a_0, __riscv_vle8_v_i8m1(q8, vl), vl);
        vint16m2_t va_q_1 =
          __riscv_vwmul_vv_i16m2(a_1, __riscv_vle8_v_i8m1(q8 + 32, vl), vl);
        vint16m2_t va_q_2 =
          __riscv_vwmul_vv_i16m2(a_2, __riscv_vle8_v_i8m1(q8 + 64, vl), vl);
        vint16m2_t va_q_3 =
          __riscv_vwmul_vv_i16m2(a_3, __riscv_vle8_v_i8m1(q8 + 96, vl), vl);

        vl = 16;

        vint32m2_t vaux_0 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(va_q_0, 0), scale[is + 0], vl);
        vint32m2_t vaux_1 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(va_q_0, 1), scale[is + 1], vl);
        vint32m2_t vaux_2 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(va_q_1, 0), scale[is + 2], vl);
        vint32m2_t vaux_3 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(va_q_1, 1), scale[is + 3], vl);
        vint32m2_t vaux_4 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(va_q_2, 0), scale[is + 4], vl);
        vint32m2_t vaux_5 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(va_q_2, 1), scale[is + 5], vl);
        vint32m2_t vaux_6 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(va_q_3, 0), scale[is + 6], vl);
        vint32m2_t vaux_7 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(va_q_3, 1), scale[is + 7], vl);

        vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(
          __riscv_vadd_vv_i32m2(vaux_0, vaux_1, vl), vzero, vl);
        vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(
          __riscv_vadd_vv_i32m2(vaux_2, vaux_3, vl), isum0, vl);
        vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(
          __riscv_vadd_vv_i32m2(vaux_4, vaux_5, vl), isum1, vl);
        vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(
          __riscv_vadd_vv_i32m2(vaux_6, vaux_7, vl), isum2, vl);

        sum_t += __riscv_vmv_x_s_i32m1_i32(isum3);

        q6 += 64;
        qh += 32;
        q8 += 128;
        is = 8;
      }

      sumf += d * sum_t;
    }
    break;
  case 128:
    for (int i = 0; i < nb; ++i) {

      const float d = nntr_compute_fp16_to_fp32(x[i].d) * y[i].d;

      const uint8_t *restrict q6 = x[i].ql;
      const uint8_t *restrict qh = x[i].qh;
      const int8_t *restrict q8 = y[i].qs;

      const int8_t *restrict scale = x[i].scales;

      int sum_t = 0;
      int t0;

      for (int j = 0; j < QK_K / 128; ++j) {
        __asm__ __volatile__(
          "vsetvli zero, %[vl32], e8, m2\n\t"
          "vle8.v v4, (%[qh])\n\t"
          "vsll.vi v0, v4, 4\n\t"
          "vsll.vi v2, v4, 2\n\t"
          "vsrl.vi v6, v4, 2\n\t"
          "vsetvli zero, %[vl64], e8, m4\n\t"
          "vle8.v v8, (%[q6])\n\t"
          "vsrl.vi v12, v8, 4\n\t"
          "vand.vi v8, v8, 0xF\n\t"
          "vsetvli zero, %[vl128], e8, m8\n\t"
          "vand.vx v0, v0, %[mask]\n\t"
          "vor.vv v8, v8, v0\n\t"
          "vle8.v v0, (%[q8])\n\t"
          "vsub.vx v8, v8, %[vl32]\n\t"
          "vsetvli zero, %[vl64], e8, m4\n\t"
          "vwmul.vv v16, v0, v8\n\t"
          "vwmul.vv v24, v4, v12\n\t"
          "vsetivli zero, 16, e16, m2\n\t"
          "vmv.v.x v0, zero\n\t"
          "vwredsum.vs v10, v16, v0\n\t"
          "vwredsum.vs v9, v18, v0\n\t"
          "vwredsum.vs v8, v20, v0\n\t"
          "vwredsum.vs v7, v22, v0\n\t"
          "vwredsum.vs v11, v24, v0\n\t"
          "vwredsum.vs v12, v26, v0\n\t"
          "vwredsum.vs v13, v28, v0\n\t"
          "vwredsum.vs v14, v30, v0\n\t"
          "vsetivli zero, 4, e32, m1\n\t"
          "vslideup.vi v10, v9, 1\n\t"
          "vslideup.vi v8, v7, 1\n\t"
          "vslideup.vi v11, v12, 1\n\t"
          "vslideup.vi v13, v14, 1\n\t"
          "vslideup.vi v10, v8, 2\n\t"
          "vslideup.vi v11, v13, 2\n\t"
          "vsetivli zero, 8, e32, m2\n\t"
          "vle8.v v2, (%[scale])\n\t"
          "vsext.vf4 v4, v2\n\t"
          "vmul.vv v2, v4, v10\n\t"
          "vredsum.vs v0, v2, v0\n\t"
          "vmv.x.s %[t0], v0\n\t"
          "add %[sumi], %[sumi], %[t0]"
          : [sumi] "+&r"(sum_t), [t0] "=&r"(t0)
          : [qh] "r"(qh), [q6] "r"(q6), [q8] "r"(q8), [scale] "r"(scale),
            [vl32] "r"(32), [vl64] "r"(64), [vl128] "r"(128), [mask] "r"(0x30)
          : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
            "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
            "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
            "v28", "v29", "v30", "v31");
        q6 += 64;
        qh += 32;
        q8 += 128;
        scale += 8;
      }

      sumf += d * sum_t;
    }
    break;
  default:
    assert(false && "Unsupported vector length");
    break;
  }

  *s = sumf;

#elif defined(__POWER9_VECTOR__)
  const vector signed char lowMask = vec_splats((signed char)0xF);
  const vector int v0 = vec_splats((int32_t)0);
  const vector unsigned char v2 = vec_splats((unsigned char)0x2);
  const vector unsigned char v3 = vec_splats((unsigned char)0x3);
  const vector unsigned char v4 = vec_splats((unsigned char)0x4);
  const vector unsigned char v6 = vec_splats((unsigned char)0x6);
  const vector signed char off = vec_splats((signed char)0x20);

  vector float vsumf0 = vec_splats(0.0f);
  vector float vsumf1 = vec_splats(0.0f);
  vector float vsumf2 = vec_splats(0.0f);
  vector float vsumf3 = vec_splats(0.0f);

  for (int i = 0; i < nb; ++i) {
    vector float vxd = vec_splats(nntr_compute_fp16_to_fp32(x[i].d));
    vector float vyd = vec_splats(y[i].d);
    vector float vd = vec_mul(vxd, vyd);

    vector signed int vsumi0 = v0;
    vector signed int vsumi1 = v0;
    vector signed int vsumi2 = v0;
    vector signed int vsumi3 = v0;
    vector signed int vsumi4 = v0;
    vector signed int vsumi5 = v0;
    vector signed int vsumi6 = v0;
    vector signed int vsumi7 = v0;

    const uint8_t *__restrict q6 = x[i].ql;
    const uint8_t *__restrict qh = x[i].qh;
    const int8_t *__restrict qs = x[i].scales;
    const int8_t *__restrict q8 = y[i].qs;

    for (int j = 0; j < QK_K / 128; ++j) {
      __builtin_prefetch(q6, 0, 0);
      __builtin_prefetch(qh, 0, 0);
      __builtin_prefetch(q8, 0, 0);

      vector signed char qxs0 = (vector signed char)vec_xl(0, q6);
      vector signed char qxs1 = (vector signed char)vec_xl(16, q6);
      vector signed char qxs2 = (vector signed char)vec_xl(32, q6);
      vector signed char qxs3 = (vector signed char)vec_xl(48, q6);
      q6 += 64;

      vector signed char qxs00 = vec_and(qxs0, lowMask);
      vector signed char qxs01 = vec_sr(qxs0, v4);
      vector signed char qxs10 = vec_and(qxs1, lowMask);
      vector signed char qxs11 = vec_sr(qxs1, v4);
      vector signed char qxs20 = vec_and(qxs2, lowMask);
      vector signed char qxs21 = vec_sr(qxs2, v4);
      vector signed char qxs30 = vec_and(qxs3, lowMask);
      vector signed char qxs31 = vec_sr(qxs3, v4);

      vector signed char qxhs0 = (vector signed char)vec_xl(0, qh);
      vector signed char qxhs1 = (vector signed char)vec_xl(16, qh);
      qh += 32;

      vector signed char qxh00 =
        vec_sl(vec_and((vector signed char)v3, qxhs0), v4);
      vector signed char qxh01 =
        vec_sl(vec_and((vector signed char)v3, vec_sr(qxhs0, v4)), v4);
      vector signed char qxh10 =
        vec_sl(vec_and((vector signed char)v3, qxhs1), v4);
      vector signed char qxh11 =
        vec_sl(vec_and((vector signed char)v3, vec_sr(qxhs1, v4)), v4);
      vector signed char qxh20 =
        vec_sl(vec_and((vector signed char)v3, vec_sr(qxhs0, v2)), v4);
      vector signed char qxh21 =
        vec_sl(vec_and((vector signed char)v3, vec_sr(qxhs0, v6)), v4);
      vector signed char qxh30 =
        vec_sl(vec_and((vector signed char)v3, vec_sr(qxhs1, v2)), v4);
      vector signed char qxh31 =
        vec_sl(vec_and((vector signed char)v3, vec_sr(qxhs1, v6)), v4);

      vector signed char q6x00 = vec_sub(vec_or(qxh00, qxs00), off);
      vector signed char q6x01 = vec_sub(vec_or(qxh01, qxs01), off);
      vector signed char q6x10 = vec_sub(vec_or(qxh10, qxs10), off);
      vector signed char q6x11 = vec_sub(vec_or(qxh11, qxs11), off);
      vector signed char q6x20 = vec_sub(vec_or(qxh20, qxs20), off);
      vector signed char q6x21 = vec_sub(vec_or(qxh21, qxs21), off);
      vector signed char q6x30 = vec_sub(vec_or(qxh30, qxs30), off);
      vector signed char q6x31 = vec_sub(vec_or(qxh31, qxs31), off);

      vector signed char q8y00 = vec_xl(0, q8);
      vector signed char q8y10 = vec_xl(16, q8);
      vector signed char q8y20 = vec_xl(32, q8);
      vector signed char q8y30 = vec_xl(48, q8);
      vector signed char q8y01 = vec_xl(64, q8);
      vector signed char q8y11 = vec_xl(80, q8);
      vector signed char q8y21 = vec_xl(96, q8);
      vector signed char q8y31 = vec_xl(112, q8);
      q8 += 128;

      vector signed short qv00 =
        vec_add(vec_mule(q6x00, q8y00), vec_mulo(q6x00, q8y00));
      vector signed short qv10 =
        vec_add(vec_mule(q6x10, q8y10), vec_mulo(q6x10, q8y10));
      vector signed short qv20 =
        vec_add(vec_mule(q6x20, q8y20), vec_mulo(q6x20, q8y20));
      vector signed short qv30 =
        vec_add(vec_mule(q6x30, q8y30), vec_mulo(q6x30, q8y30));
      vector signed short qv01 =
        vec_add(vec_mule(q6x01, q8y01), vec_mulo(q6x01, q8y01));
      vector signed short qv11 =
        vec_add(vec_mule(q6x11, q8y11), vec_mulo(q6x11, q8y11));
      vector signed short qv21 =
        vec_add(vec_mule(q6x21, q8y21), vec_mulo(q6x21, q8y21));
      vector signed short qv31 =
        vec_add(vec_mule(q6x31, q8y31), vec_mulo(q6x31, q8y31));

      vector signed short vscales = vec_unpackh(vec_xl_len(qs, 8));
      qs += 8;

      vector signed short vs0 = vec_splat(vscales, 0);
      vector signed short vs1 = vec_splat(vscales, 1);
      vector signed short vs2 = vec_splat(vscales, 2);
      vector signed short vs3 = vec_splat(vscales, 3);
      vector signed short vs4 = vec_splat(vscales, 4);
      vector signed short vs5 = vec_splat(vscales, 5);
      vector signed short vs6 = vec_splat(vscales, 6);
      vector signed short vs7 = vec_splat(vscales, 7);

      vsumi0 = vec_msum(qv00, vs0, vsumi0);
      vsumi1 = vec_msum(qv01, vs4, vsumi1);
      vsumi2 = vec_msum(qv10, vs1, vsumi2);
      vsumi3 = vec_msum(qv11, vs5, vsumi3);
      vsumi4 = vec_msum(qv20, vs2, vsumi4);
      vsumi5 = vec_msum(qv21, vs6, vsumi5);
      vsumi6 = vec_msum(qv30, vs3, vsumi6);
      vsumi7 = vec_msum(qv31, vs7, vsumi7);
    }

    vsumi0 = vec_add(vsumi0, vsumi4);
    vsumi1 = vec_add(vsumi1, vsumi5);
    vsumi2 = vec_add(vsumi2, vsumi6);
    vsumi3 = vec_add(vsumi3, vsumi7);

    vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
    vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
    vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
  }

  vsumf0 = vec_add(vsumf0, vsumf2);
  vsumf1 = vec_add(vsumf1, vsumf3);

  vsumf0 = vec_add(vsumf0, vsumf1);

  vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
  vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

  *s = vec_extract(vsumf0, 0);

#elif defined __loongarch_asx

  const __m256i m32s = __lasx_xvreplgr2vr_b(32);

  __m256 acc = (__m256)__lasx_xvldi(0);

  for (int i = 0; i < nb; ++i) {

    const float d = y[i].d * nntr_compute_fp16_to_fp32(x[i].d);

    const uint8_t *__restrict q4 = x[i].ql;
    const uint8_t *__restrict qh = x[i].qh;
    const int8_t *__restrict q8 = y[i].qs;

    const __m128i scales128 = __lsx_vld((const __m128i *)x[i].scales, 0);
    const v16i8 shuffle_mask = {0, 2, 4, 6, 8, 10, 12, 14,
                                1, 3, 5, 7, 9, 11, 13, 15};
    const __m256i scales_shuffled =
      lasx_ext8_16(__lsx_vshuf_b(scales128, scales128, (__m128i)shuffle_mask));

    __m256i sumi = __lasx_xvldi(0);

    for (int j = 0; j < QK_K / 128; ++j) {

      const __m256i q4bits1 = __lasx_xvld((const __m256i *)q4, 0);
      q4 += 32;
      const __m256i q4bits2 = __lasx_xvld((const __m256i *)q4, 0);
      q4 += 32;
      const __m256i q4bitsH = __lasx_xvld((const __m256i *)qh, 0);
      qh += 32;

      const __m256i q4h_0 = __lasx_xvslli_b(__lasx_xvandi_b(q4bitsH, 3), 4);
      const __m256i q4h_1 =
        __lasx_xvslli_b(__lasx_xvandi_b(q4bitsH, 3 << 2), 2);
      const __m256i q4h_2 = __lasx_xvandi_b(q4bitsH, 3 << 4);
      const __m256i q4h_3 =
        __lasx_xvsrli_b(__lasx_xvandi_b(q4bitsH, 3 << 6), 2);

      const __m256i q4_0 = __lasx_xvor_v(__lasx_xvandi_b(q4bits1, 0xf), q4h_0);
      const __m256i q4_1 = __lasx_xvor_v(__lasx_xvandi_b(q4bits2, 0xf), q4h_1);
      const __m256i q4_2 = __lasx_xvor_v(__lasx_xvsrli_b(q4bits1, 4), q4h_2);
      const __m256i q4_3 = __lasx_xvor_v(__lasx_xvsrli_b(q4bits2, 4), q4h_3);

      const __m256i q8_0 = __lasx_xvld((const __m256i *)q8, 0);
      q8 += 32;
      const __m256i q8_1 = __lasx_xvld((const __m256i *)q8, 0);
      q8 += 32;
      const __m256i q8_2 = __lasx_xvld((const __m256i *)q8, 0);
      q8 += 32;
      const __m256i q8_3 = __lasx_xvld((const __m256i *)q8, 0);
      q8 += 32;

      __m256i p16_0 = lasx_madd_h_b(__lasx_xvsub_b(q4_0, m32s), q8_0);
      __m256i p16_1 = lasx_madd_h_b(__lasx_xvsub_b(q4_1, m32s), q8_1);
      __m256i p16_2 = lasx_madd_h_b(__lasx_xvsub_b(q4_2, m32s), q8_2);
      __m256i p16_3 = lasx_madd_h_b(__lasx_xvsub_b(q4_3, m32s), q8_3);

      p16_0 =
        lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 0), p16_0);
      p16_1 =
        lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 1), p16_1);
      p16_2 =
        lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 2), p16_2);
      p16_3 =
        lasx_madd_h(lasx_xvrepl128vei_h(scales_shuffled, 4 * j + 3), p16_3);

      sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p16_0, p16_1));
      sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p16_2, p16_3));
    }

    acc = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(d),
                           __lasx_xvffint_s_w(sumi), acc);
  }

  *s = hsum_float_8(acc);
#elif defined(__VXE__) || defined(__VXE2__)
  float sum = 0;

  // Lower 4-bit and upper 2-bit masks
  const uint8x16_t v_lm = vec_splat_u8(0x0F);
  const uint8x16_t v_um = vec_splat_u8(0x03);

  const int32x4_t v_z = vec_splat_s32(0);

  int8x16_t q6b[4];
  uint8x16_t q6h[4];

  uint8x16_t v_xl[4];
  uint8x16_t v_xh[2];
  int8x16_t v_y[4];

  for (int i = 0; i < nb; ++i) {
    const float d_all = nntr_compute_fp16_to_fp32(x[i].d);

    const uint8_t *__restrict x0l = x[i].ql;
    const uint8_t *__restrict x0h = x[i].qh;
    const int8_t *__restrict y0 = y[i].qs;

    const int8_t *__restrict scale = x[i].scales;

    const int16x8_t v_ysumsl = vec_xl(0, y[i].bsums);
    const int16x8_t v_ysumsh = vec_xl(16, y[i].bsums);

    const int8x16_t v_scale = vec_xl(0, scale);
    const int16x8_t v_scalel = vec_unpackh(v_scale);
    const int16x8_t v_scaleh = vec_unpackl(v_scale);

    const int32x4_t v_minslo = vec_mulo(v_ysumsl, v_scalel);
    const int32x4_t v_minsle = vec_mule(v_ysumsl, v_scalel);
    const int32x4_t v_minsho = vec_mulo(v_ysumsh, v_scaleh);
    const int32x4_t v_minshe = vec_mule(v_ysumsh, v_scaleh);
    const int32x4_t v_mins = v_minslo + v_minsle + v_minsho + v_minshe;

    const int32_t mins = v_mins[0] + v_mins[1] + v_mins[2] + v_mins[3];

    int32_t isum = 0;
    for (int j = 0; j < QK_K / 128; ++j) {
      // Load model upper 2 bits
      v_xh[0] = vec_xl(0, x0h);
      v_xh[1] = vec_xl(16, x0h);
      x0h += 32;

      // Load model lower 4 bits
      v_xl[0] = vec_xl(0, x0l);
      v_xl[1] = vec_xl(16, x0l);
      v_xl[2] = vec_xl(32, x0l);
      v_xl[3] = vec_xl(48, x0l);
      x0l += 64;

      // Load activation quants
      v_y[0] = vec_xl(0, y0);
      v_y[1] = vec_xl(16, y0);
      v_y[2] = vec_xl(32, y0);
      v_y[3] = vec_xl(48, y0);
      y0 += 64;

      q6h[0] = vec_sl(vec_and(v_um, v_xh[0]), 4);
      q6h[1] = vec_sl(vec_and(v_um, v_xh[1]), 4);
      uint8x16_t shifted = vec_sr(v_xh[0], 2);
      q6h[2] = vec_sl(vec_and(v_um, shifted), 4);
      shifted = vec_sr(v_xh[1], 2);
      q6h[3] = vec_sl(vec_and(v_um, shifted), 4);

      q6b[0] = (int8x16_t)(vec_or(vec_and(v_xl[0], v_lm), q6h[0]));
      q6b[1] = (int8x16_t)(vec_or(vec_and(v_xl[1], v_lm), q6h[1]));
      q6b[2] = (int8x16_t)(vec_or(vec_and(v_xl[2], v_lm), q6h[2]));
      q6b[3] = (int8x16_t)(vec_or(vec_and(v_xl[3], v_lm), q6h[3]));

      int32x4_t summs0 = ggml_vec_dot(v_z, q6b[0], v_y[0]);
      int32x4_t summs1 = ggml_vec_dot(v_z, q6b[1], v_y[1]);
      int32x4_t summs2 = ggml_vec_dot(v_z, q6b[2], v_y[2]);
      int32x4_t summs3 = ggml_vec_dot(v_z, q6b[3], v_y[3]);

      isum += (summs0[0] + summs0[1] + summs0[2] + summs0[3]) * scale[0] +
              (summs1[0] + summs1[1] + summs1[2] + summs1[3]) * scale[1] +
              (summs2[0] + summs2[1] + summs2[2] + summs2[3]) * scale[2] +
              (summs3[0] + summs3[1] + summs3[2] + summs3[3]) * scale[3];

      scale += 4;

      // Load activation quants
      v_y[0] = vec_xl(0, y0);
      v_y[1] = vec_xl(16, y0);
      v_y[2] = vec_xl(32, y0);
      v_y[3] = vec_xl(48, y0);
      y0 += 64;

      shifted = vec_sr(v_xh[0], 4);
      q6h[0] = vec_sl(vec_and(v_um, shifted), 4);
      shifted = vec_sr(v_xh[1], 4);
      q6h[1] = vec_sl(vec_and(v_um, shifted), 4);
      shifted = vec_sr(v_xh[0], 6);
      q6h[2] = vec_sl(vec_and(v_um, shifted), 4);
      shifted = vec_sr(v_xh[1], 6);
      q6h[3] = vec_sl(vec_and(v_um, shifted), 4);

      q6b[0] = (int8x16_t)(vec_or(vec_sr(v_xl[0], 4), q6h[0]));
      q6b[1] = (int8x16_t)(vec_or(vec_sr(v_xl[1], 4), q6h[1]));
      q6b[2] = (int8x16_t)(vec_or(vec_sr(v_xl[2], 4), q6h[2]));
      q6b[3] = (int8x16_t)(vec_or(vec_sr(v_xl[3], 4), q6h[3]));

      summs0 = ggml_vec_dot(v_z, q6b[0], v_y[0]);
      summs1 = ggml_vec_dot(v_z, q6b[1], v_y[1]);
      summs2 = ggml_vec_dot(v_z, q6b[2], v_y[2]);
      summs3 = ggml_vec_dot(v_z, q6b[3], v_y[3]);

      isum += (summs0[0] + summs0[1] + summs0[2] + summs0[3]) * scale[0] +
              (summs1[0] + summs1[1] + summs1[2] + summs1[3]) * scale[1] +
              (summs2[0] + summs2[1] + summs2[2] + summs2[3]) * scale[2] +
              (summs3[0] + summs3[1] + summs3[2] + summs3[3]) * scale[3];

      scale += 4;
    }

    sum += d_all * y[i].d * (isum - 32 * mins);
  }

  *s = sum;
#else

  int8_t aux8[QK_K];
  int16_t aux16[8];
  float sums[8];
  int32_t aux32[8];
  memset(sums, 0, 8 * sizeof(float));

  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    const uint8_t *__restrict q4 = x[i].ql;
    const uint8_t *__restrict qh = x[i].qh;
    const int8_t *__restrict q8 = y[i].qs;
    memset(aux32, 0, 8 * sizeof(int32_t));
    int8_t *__restrict a = aux8;
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        a[l + 0] = (int8_t)((q4[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        a[l + 32] =
          (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        a[l + 64] = (int8_t)((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        a[l + 96] =
          (int8_t)((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      }
      a += 128;
      q4 += 64;
      qh += 32;
    }
    a = aux8;
    int is = 0;
    for (int j = 0; j < QK_K / 16; ++j) {
      int scale = x[i].scales[is++];
      for (int l = 0; l < 8; ++l)
        aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l)
        aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
      for (int l = 0; l < 8; ++l)
        aux16[l] = q8[l] * a[l];
      for (int l = 0; l < 8; ++l)
        aux32[l] += scale * aux16[l];
      q8 += 8;
      a += 8;
    }
    const float d = nntr_compute_fp16_to_fp32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l)
      sums[l] += d * aux32[l];
  }
  for (int l = 0; l < 8; ++l)
    sumf += sums[l];
  *s = sumf;
#endif
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

#if !((defined(_MSC_VER)) && !defined(__clang__)) && defined(__aarch64__)
#if defined(__ARM_FEATURE_SVE)
  {
    const void *b_ptr = vx;
    const void *a_ptr = vy;
    float *res_ptr = s;

    __asm__ __volatile__(
      "ptrue p0.b\n"
      "add %x[b_ptr], %x[b_ptr], #0x10\n"
      "1:" // Column loop
      "add x22, %x[a_ptr], #0x2\n"
      "mov z31.b, #0x0\n"
      "mov x21, %x[nb]\n"
      "2:" // Block loop
      "ld1b { z30.b }, p0/Z, [%x[b_ptr]]\n"
      "ld1b { z29.b }, p0/Z, [%x[b_ptr], #1, MUL VL]\n"
      "mov z28.s, #0x0\n"
      "mov z27.s, #0x0\n"
      "ld1rd { z26.d }, p0/Z, [x22]\n"
      "ld1b { z25.b }, p0/Z, [%x[b_ptr], #2, MUL VL]\n"
      "sub x20, x22, #0x2\n"
      "sub x21, x21, #0x1\n"
      "ld1b { z24.b }, p0/Z, [%x[b_ptr], #3, MUL VL]\n"
      "ld1rd { z23.d }, p0/Z, [x22, #8]\n"
      "lsl z22.b, z30.b, #0x4\n"
      "lsl z16.b, z29.b, #0x4\n"
      "and z30.b, z30.b, #0xf0\n"
      "and z29.b, z29.b, #0xf0\n"
      "ld1rd { z21.d }, p0/Z, [x22, #16]\n"
      "ld1rd { z20.d }, p0/Z, [x22, #24]\n"
      "lsl z19.b, z25.b, #0x4\n"
      "and z25.b, z25.b, #0xf0\n"
      "ld1rh { z17.h }, p0/Z, [x20]\n"
      "ld1h { z18.s }, p0/Z, [%x[b_ptr], #-1, MUL VL]\n"
      "sdot z28.s, z22.b, z26.b\n"
      "sdot z27.s, z16.b, z26.b\n"
      "lsl z16.b, z24.b, #0x4\n"
      "add x22, x22, #0x22\n"
      "and z24.b, z24.b, #0xf0\n"
      "add %x[b_ptr], %x[b_ptr], #0x90\n"
      "fcvt z17.s, p0/m, z17.h\n"
      "fcvt z18.s, p0/m, z18.h\n"
      "sdot z28.s, z19.b, z23.b\n"
      "sdot z27.s, z16.b, z23.b\n"
      "fmul z18.s, z18.s, z17.s\n"
      "sdot z28.s, z30.b, z21.b\n"
      "sdot z27.s, z29.b, z21.b\n"
      "sdot z28.s, z25.b, z20.b\n"
      "sdot z27.s, z24.b, z20.b\n"
      "uzp1 z17.s, z28.s, z27.s\n"
      "uzp2 z16.s, z28.s, z27.s\n"
      "add z17.s, z17.s, z16.s\n"
      "asr z17.s, z17.s, #0x4\n"
      "scvtf z17.s, p0/m, z17.s\n"
      "fmla z31.s, p0/M, z17.s, z18.s\n"
      "cbnz x21, 2b\n"
      "sub %x[nc], %x[nc], #0x8\n"
      "st1w { z31.s }, p0, [%x[res_ptr]]\n"
      "add %x[res_ptr], %x[res_ptr], #0x20\n"
      "cbnz %x[nc], 1b\n"
      : [b_ptr] "+&r"(b_ptr), [res_ptr] "+&r"(res_ptr), [nc] "+&r"(nc)
      : [a_ptr] "r"(a_ptr), [nb] "r"(nb)
      : "memory", "p0", "x20", "x21", "x22", "z16", "z17", "z18", "z19", "z20",
        "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30",
        "z31");
    return;
  }
#endif // #if defined(__ARM_FEATURE_SVE)
#elif defined(__AVX2__)
  // Lookup table to convert signed nibbles to signed bytes
  __m256i signextendlut = _mm256_castsi128_si256(
    _mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
  signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
  __m128i changemask =
    _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
  __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

  // Permute mask used for easier vector processing at later stages
  const __m256i m4b = _mm256_set1_epi8(0x0F);

  int64_t b_nb = n / QK4_0;

  const block_q4_0x8 *b_ptr_start = (const block_q4_0x8 *)vx;
  const block_q8_0 *a_ptr_start = (const block_q8_0 *)vy;

  // Process Q8_0 blocks one by one
  for (int64_t y = 0; y < nr; y++) {

    // Pointers to LHS blocks of block_q8_0 format
    const block_q8_0 *a_ptr = a_ptr_start + (y * nb);

    // Take group of eight block_q4_0x8 structures at each pass of the loop and
    // perform dot product operation
    for (int64_t x = 0; x < nc / 8; x++) {

      // Pointers to RHS blocks
      const block_q4_0x8 *b_ptr = b_ptr_start + (x * b_nb);

      // Master FP accumulator
      __m256 acc_row = _mm256_setzero_ps();

      for (int64_t b = 0; b < nb; b++) {
        // Load 8 blocks of Q4_0 interleaved as 8 bytes (B0 - B7)
        const __m256i rhs_raw_vec_0123_0 =
          _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
        const __m256i rhs_raw_vec_4567_0 =
          _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 1);
        const __m256i rhs_raw_vec_0123_1 =
          _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 2);
        const __m256i rhs_raw_vec_4567_1 =
          _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 3);

        // 4-bit -> 8-bit - Sign is maintained
        const __m256i rhs_vec_0123_0 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(rhs_raw_vec_0123_0,
                           m4b)); // B0(0-7) B1(0-7) B2(0-7) B3(0-7)
        const __m256i rhs_vec_4567_0 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(rhs_raw_vec_4567_0,
                           m4b)); // B4(0-7) B5(0-7) B6(0-7) B7(0-7)
        const __m256i rhs_vec_0123_1 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(rhs_raw_vec_0123_1,
                           m4b)); // B0(8-15) B1(8-15) B2(8-15) B3(8-15)
        const __m256i rhs_vec_4567_1 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(rhs_raw_vec_4567_1,
                           m4b)); // B0(8-15) B1(8-15) B2(8-15) B3(8-15)

        const __m256i rhs_vec_0123_2 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4),
                           m4b)); // B0(16-23) B1(16-23) B2(16-23) B3(16-23)
        const __m256i rhs_vec_4567_2 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4),
                           m4b)); // B4(16-23) B5(16-23) B6(16-23) B7(16-23)
        const __m256i rhs_vec_0123_3 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4),
                           m4b)); // B0(24-31) B1(24-31) B2(24-31) B3(24-31)
        const __m256i rhs_vec_4567_3 = _mm256_shuffle_epi8(
          signextendlut,
          _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4),
                           m4b)); // B4(24-31) B5(24-31) B6(24-31) B7(24-31)

        // Load the scale values for the 8 blocks interleaved in block_q4_0x8
        const __m256 col_scale_f32 =
          GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, changemask);

        // Load and convert to FP32 scale from block_q8_0
        const __m256 row_scale_f32 =
          _mm256_set1_ps(nntr_fp16_to_fp32(a_ptr[b].d));

        // Load the block values in block_q8_0 in batches of 16 bytes and
        // replicate the same across 256 bit vector
        __m256i lhs_vec_0 =
          _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)a_ptr[b].qs));
        __m256i lhs_vec_1 = _mm256_castsi128_si256(
          _mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16)));

        lhs_vec_0 = _mm256_permute2f128_si256(lhs_vec_0, lhs_vec_0,
                                              0); // A0 (0-15) A0(0-15)
        lhs_vec_1 = _mm256_permute2f128_si256(lhs_vec_1, lhs_vec_1,
                                              0); // A0 (16-31) A0(16-31))

        __m256i iacc = _mm256_setzero_si256();

        // Dot product done within 32 bit lanes and accumulated in the same
        // vector B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3)
        // B7(0-3) with A0(0-3) B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7)
        // B3(4-7) B7(4-7) with A0(4-7)
        // ...........................................................................
        // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31) B3(28-31)
        // B7(28-31) with A0(28-31)

        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(rhs_vec_0123_0,
                             _mm256_shuffle_epi32(rhs_vec_4567_0, 177), 170),
          _mm256_shuffle_epi32(lhs_vec_0, 0));
        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_0, 177),
                             rhs_vec_4567_0, 170),
          _mm256_shuffle_epi32(lhs_vec_0, 85));

        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(rhs_vec_0123_1,
                             _mm256_shuffle_epi32(rhs_vec_4567_1, 177), 170),
          _mm256_shuffle_epi32(lhs_vec_0, 170));
        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_1, 177),
                             rhs_vec_4567_1, 170),
          _mm256_shuffle_epi32(lhs_vec_0, 255));

        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(rhs_vec_0123_2,
                             _mm256_shuffle_epi32(rhs_vec_4567_2, 177), 170),
          _mm256_shuffle_epi32(lhs_vec_1, 0));
        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_2, 177),
                             rhs_vec_4567_2, 170),
          _mm256_shuffle_epi32(lhs_vec_1, 85));

        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(rhs_vec_0123_3,
                             _mm256_shuffle_epi32(rhs_vec_4567_3, 177), 170),
          _mm256_shuffle_epi32(lhs_vec_1, 170));
        iacc = mul_sum_i8_pairs_acc_int32x8(
          iacc,
          _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_3, 177),
                             rhs_vec_4567_3, 170),
          _mm256_shuffle_epi32(lhs_vec_1, 255));

        // Accumulated values multipled with appropriate scales
        acc_row =
          _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc),
                          _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
      }

      // Accumulated output values permuted so as to be stored in appropriate
      // order post accumulation
      acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
      _mm256_storeu_ps(s + (y * nr + x * 8), acc_row);
    }
  }
  return;
#elif defined(__riscv_v_intrinsic)
  if (__riscv_vlenb() >= QK4_0) {
    const size_t vl = QK4_0;

    const block_q8_0 *a_ptr = (const block_q8_0 *)vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
      const block_q4_0x8 *b_ptr = (const block_q4_0x8 *)vx + (x * nb);

      vfloat32m1_t sumf = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
      for (int l = 0; l < nb; l++) {
        const int64_t a0 = *(const int64_t *)&a_ptr[l].qs[0];
        const int64_t a1 = *(const int64_t *)&a_ptr[l].qs[8];
        const int64_t a2 = *(const int64_t *)&a_ptr[l].qs[16];
        const int64_t a3 = *(const int64_t *)&a_ptr[l].qs[24];
        __asm__ __volatile__("" ::
                               : "memory"); // prevent gcc from emitting fused
                                            // vlse64, violating alignment
        const vint8m2_t lhs_0_8 =
          __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a0, vl / 4));
        const vint8m2_t lhs_1_8 =
          __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a1, vl / 4));
        const vint8m2_t lhs_2_8 =
          __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a2, vl / 4));
        const vint8m2_t lhs_3_8 =
          __riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a3, vl / 4));

        const vint8m4_t rhs_raw_vec =
          __riscv_vle8_v_i8m4((const int8_t *)b_ptr[l].qs, vl * 4);
        const vint8m4_t rhs_vec_lo = __riscv_vsra_vx_i8m4(
          __riscv_vsll_vx_i8m4(rhs_raw_vec, 4, vl * 4), 4, vl * 4);
        const vint8m4_t rhs_vec_hi =
          __riscv_vsra_vx_i8m4(rhs_raw_vec, 4, vl * 4);
        const vint8m2_t rhs_vec_lo_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 0);
        const vint8m2_t rhs_vec_lo_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 1);
        const vint8m2_t rhs_vec_hi_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 0);
        const vint8m2_t rhs_vec_hi_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 1);

        const vint16m4_t sumi_lo_0 =
          __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
        const vint16m4_t sumi_lo_1 =
          __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
        const vint16m4_t sumi_hi_0 =
          __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
        const vint16m4_t sumi_hi_m =
          __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

        const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(
          __riscv_vreinterpret_v_i16m4_i32m4(sumi_hi_m));
        const vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
        const vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
        const vuint16m2_t sumi_h2 =
          __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
        const vuint32m2_t sumi_h2_i32 =
          __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
        const vuint16m1_t sumi_h4_0 =
          __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
        const vuint16m1_t sumi_h4_1 =
          __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
        const vuint16m1_t sumi_h4 =
          __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
        const vuint32m1_t sumi_h4_i32 =
          __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
        const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(
          __riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
        const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(
          __riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
        const vint32m1_t sumi_h8 =
          __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
        const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

        // vector version needs Zvfhmin extension
        const float a_scale = nntr_fp16_to_fp32(a_ptr[l].d);
        const float b_scales[8] = {
          nntr_fp16_to_fp32(b_ptr[l].d[0]), nntr_fp16_to_fp32(b_ptr[l].d[1]),
          nntr_fp16_to_fp32(b_ptr[l].d[2]), nntr_fp16_to_fp32(b_ptr[l].d[3]),
          nntr_fp16_to_fp32(b_ptr[l].d[4]), nntr_fp16_to_fp32(b_ptr[l].d[5]),
          nntr_fp16_to_fp32(b_ptr[l].d[6]), nntr_fp16_to_fp32(b_ptr[l].d[7])};
        const vfloat32m1_t b_scales_vec =
          __riscv_vle32_v_f32m1(b_scales, vl / 4);
        const vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scale, vl / 4);
        sumf = __riscv_vfmacc_vv_f32m1(sumf, tmp1, b_scales_vec, vl / 4);
      }
      __riscv_vse32_v_f32m1(s + x * ncols_interleaved, sumf, vl / 4);
    }
    return;
  }
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) &&
       // defined(__aarch64__)
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

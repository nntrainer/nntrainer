// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Portions of this file are derived from llama.cpp
 * (https://github.com/ggml-org/llama.cpp), licensed under the MIT License.
 * Copyright (c) Contributors to llama.cpp
 *
 * Modified by Pawel Debski, 2024: Adapted for CPU backend integration
 *
 * @file   nntr_ggml_impl_fp16_fp32.cpp
 * @date   20 August 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author  Pawel Debski <p.debski2@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Implementations of GGML init & fp16-fp32 conversions
 */

#include <nntr_ggml_impl.h>
#include <nntr_ggml_impl_utils.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>

#include <cstring>
#include <math.h>
#include <time.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <mutex>

// FP16 to FP32 conversion

// 16-bit float
// on Arm, we use __fp16
// on x86, we use uint16_t
//
// for old CUDA compilers (<= 11), we use uint16_t: ref
// https://github.com/ggml-org/llama.cpp/pull/10616 for     MUSA compilers , we
// use uint16_t: ref https://github.com/ggml-org/llama.cpp/pull/11843
//
#if defined(__ARM_NEON) && defined(ENABLE_FP16) &&                             \
  !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__ <= 11) && !defined(__MUSACC__)
#define NNTR_COMPUTE_FP16_TO_FP32(x) nntr_compute_fp16_to_fp32_impl(x)
#define NNTR_COMPUTE_FP32_TO_FP16(x) nntr_compute_fp32_to_fp16_impl(x)

#define NNTR_FP16_TO_FP32(x) nntr_compute_fp16_to_fp32_impl(x)

static inline float nntr_compute_fp16_to_fp32_impl(nntr_fp16_t h) {
  __fp16 tmp;
  memcpy(&tmp, &h, sizeof(nntr_fp16_t));
  return (float)tmp;
}

static inline nntr_fp16_t nntr_compute_fp32_to_fp16_impl(float f) {
  nntr_fp16_t res;
  __fp16 tmp = f;
  memcpy(&res, &tmp, sizeof(nntr_fp16_t));
  return res;
}

#elif defined(__F16C__)

// Use _mm_cvtph_ps intrinsic for all compilers to avoid GCC generating
// AVX-512FP16 vmovw instructions from _cvtsh_ss on AVX-512VL-capable CPUs
// that lack AVX-512FP16 support (GCC 13+ bug).
#define NNTR_COMPUTE_FP16_TO_FP32(x)                                           \
  _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)))
#define NNTR_COMPUTE_FP32_TO_FP16(x)                                           \
  _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0)

#elif defined(__POWER9_VECTOR__)

#define NNTR_COMPUTE_FP16_TO_FP32(x) nntr_compute_fp16_to_fp32_impl(x)
#define NNTR_COMPUTE_FP32_TO_FP16(x) nntr_compute_fp32_to_fp16_impl(x)
/** the inline asm below is about 12% faster than the lookup method */
#define NNTR_FP16_TO_FP32(x) NNTR_COMPUTE_FP16_TO_FP32(x)
#define NNTR_FP32_TO_FP16(x) NNTR_COMPUTE_FP32_TO_FP16(x)

static inline float nntr_compute_fp16_to_fp32_impl(nntr_fp16_t h) {
  float f;
  double d;
  __asm__("mtfprd %0,%2\n"
          "xscvhpdp %0,%0\n"
          "frsp %1,%0\n"
          :
          /** temp */ "=d"(d),
          /** out */ "=f"(f)
          :
          /** in */ "r"(h));
  return f;
}

static inline nntr_fp16_t nntr_compute_fp32_to_fp16_impl(float f) {
  double d;
  nntr_fp16_t r;
  __asm__(/** xscvdphp can work on double or single precision */
          "xscvdphp %0,%2\n"
          "mffprd %1,%0\n"
          :
          /** temp */ "=d"(d),
          /** out */ "=r"(r)
          :
          /** in */ "f"(f));
  return r;
}

#elif defined(__riscv) && defined(GGML_RV_ZFH)

static inline float nntr_compute_fp16_to_fp32_impl(nntr_fp16_t h) {
  float f;
  __asm__("fmv.h.x %[f], %[h]\n\t"
          "fcvt.s.h %[f], %[f]"
          : [f] "=&f"(f)
          : [h] "r"(h));
  return f;
}

static inline nntr_fp16_t nntr_compute_fp32_to_fp16_impl(float f) {
  nntr_fp16_t res;
  __asm__("fcvt.h.s %[f], %[f]\n\t"
          "fmv.x.h %[h], %[f]"
          : [h] "=&r"(res)
          : [f] "f"(f));
  return res;
}

#define NNTR_COMPUTE_FP16_TO_FP32(x) nntr_compute_fp16_to_fp32_impl(x)
#define NNTR_COMPUTE_FP32_TO_FP16(x) nntr_compute_fp32_to_fp16_impl(x)
#define NNTR_FP16_TO_FP32(x) NNTR_COMPUTE_FP16_TO_FP32(x)
#define NNTR_FP32_TO_FP16(x) NNTR_COMPUTE_FP32_TO_FP16(x)

#else

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
  union {
    uint32_t as_bits;
    float as_value;
  } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
  union {
    float as_value;
    uint32_t as_bits;
  } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}

static inline float nntr_compute_fp16_to_fp32_impl(nntr_fp16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) ||             \
     defined(__GNUC__) && !defined(__STRICT_ANSI__)) &&                        \
  (!defined(__cplusplus) || __cplusplus >= 201703L)
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float normalized_value =
    fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
    fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
    sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                        : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

static inline nntr_fp16_t nntr_compute_fp32_to_fp16_impl(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) ||             \
     defined(__GNUC__) && !defined(__STRICT_ANSI__)) &&                        \
  (!defined(__cplusplus) || __cplusplus >= 201703L)
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
#else
  const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
  const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (sign >> 16) |
         (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define NNTR_COMPUTE_FP16_TO_FP32(x) nntr_compute_fp16_to_fp32_impl(x)
#define NNTR_COMPUTE_FP32_TO_FP16(x) nntr_compute_fp32_to_fp16_impl(x)

#endif // defined(__ARM_NEON) && !(defined(__CUDACC__) && __CUDACC_VER_MAJOR__
       // <= 11) && !defined(__MUSACC__)

// precomputed f32 table for f16 (256 KB)
// defined in ggml.c, initialized in ggml_init()
static float ggml_table_f32_f16[1 << 16];

// On ARM NEON, it's quicker to directly convert x -> x instead of calling into
// nntr_lookup_fp16_to_fp32, so we define NNTR_FP16_TO_FP32 and
// NNTR_FP32_TO_FP16 elsewhere for NEON. This is also true for POWER9.
#if !defined(NNTR_FP16_TO_FP32)
inline static float nntr_lookup_fp16_to_fp32(nntr_fp16_t f) {
  uint16_t s;
  memcpy(&s, &f, sizeof(uint16_t));
  return ggml_table_f32_f16[s];
}

#define NNTR_FP16_TO_FP32(x) nntr_lookup_fp16_to_fp32(x)
#endif

#if !defined(NNTR_FP32_TO_FP16)
#define NNTR_FP32_TO_FP16(x) NNTR_COMPUTE_FP32_TO_FP16(x)
#endif

// --------------------------------------------------------------------

float nntr_fp16_to_fp32(nntr_fp16_t h) { return NNTR_FP16_TO_FP32(h); }

nntr_fp16_t nntr_fp32_to_fp16(float f) { return NNTR_FP32_TO_FP16(f); }

float nntr_compute_fp16_to_fp32(nntr_fp16_t h) {
  return NNTR_COMPUTE_FP16_TO_FP32(h);
}

nntr_fp16_t nntr_compute_fp32_to_fp16(float f) {
  return NNTR_COMPUTE_FP32_TO_FP16(f);
}

// ---------------------------- INIT ----------------------------------

//
// ggml object
//

/**
 * @brief Internal enum for GGML object type
 */
enum ggml_object_type {
  GGML_OBJECT_TYPE_TENSOR,
  GGML_OBJECT_TYPE_GRAPH,
  GGML_OBJECT_TYPE_WORK_BUFFER
};

/**
 * @brief Internal structure definition for ggml_context struct
 */
struct ggml_object {
  size_t offs;
  size_t size;

  struct ggml_object *next;

  enum ggml_object_type type;

  char padding[4];
};

static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);

//
// ggml context
//

/**
 * @brief Internal structure returned by GGML initialization function
 */
struct ggml_context {
  size_t mem_size;
  void *mem_buffer;
  bool mem_buffer_owned;
  bool no_alloc;

  int n_objects;

  struct ggml_object *objects_begin;
  struct ggml_object *objects_end;
};

/**
 * @brief Internal structure for GGML initialization function
 */
struct ggml_init_params {
  // memory pool
  size_t mem_size;  // bytes
  void *mem_buffer; // if NULL, memory will be allocated internally
  bool no_alloc;    // don't allocate memory for the tensor data
};

// ----- THREADING -----

std::mutex ggml_critical_section_mutex;

void ggml_critical_section_start() { ggml_critical_section_mutex.lock(); }

void ggml_critical_section_end(void) { ggml_critical_section_mutex.unlock(); }

// ---------------------

void ggml_aligned_free(void *ptr, size_t size) {
  (void)(size);
#if defined(_MSC_VER) || defined(__MINGW32__)
  _aligned_free(ptr);
#elif GGML_USE_CPU_HBM
  if (ptr != NULL) {
    hbw_free(ptr);
  }
#elif TARGET_OS_OSX
  if (ptr != NULL) {
    vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)ptr, size);
  }
#else
  free(ptr);
#endif
}

#if defined(_MSC_VER) || defined(__MINGW32__)
static int64_t timer_freq, timer_start;
void ggml_time_init(void) {
  LARGE_INTEGER t;
  QueryPerformanceFrequency(&t);
  timer_freq = t.QuadPart;

  // The multiplication by 1000 or 1000000 below can cause an overflow if
  // timer_freq and the uptime is high enough. We subtract the program start
  // time to reduce the likelihood of that happening.
  QueryPerformanceCounter(&t);
  timer_start = t.QuadPart;
}
int64_t ggml_time_ms(void) {
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return ((t.QuadPart - timer_start) * 1000) / timer_freq;
}
int64_t ggml_time_us(void) {
  LARGE_INTEGER t;
  QueryPerformanceCounter(&t);
  return ((t.QuadPart - timer_start) * 1000000) / timer_freq;
}
#else
void ggml_time_init(void) {}
int64_t ggml_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
}

int64_t ggml_time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}
#endif

struct ggml_context *ggml_init(struct ggml_init_params params) {
  static bool is_first_call = true;

  ggml_critical_section_start();

  if (is_first_call) {
    // initialize time system (required on Windows)
    ggml_time_init();

    for (int i = 0; i < (1 << 16); ++i) {
      nntr_fp16_t fp16 = (nntr_fp16_t)(i);
      ggml_table_f32_f16[i] = NNTR_COMPUTE_FP16_TO_FP32(fp16);
    }

    is_first_call = false;
  }

  ggml_critical_section_end();

  struct ggml_context *ctx = nullptr;

  return ctx;
}

void ggml_free(struct ggml_context *ctx) {
  if (ctx == NULL) {
    return;
  }

  if (ctx->mem_buffer_owned) {
    ggml_aligned_free(ctx->mem_buffer, ctx->mem_size);
  }

  free(ctx);
}

void nntr_ggml_init() {
  // needed to initialize f16 tables
  struct ggml_init_params params = {0, NULL, false};
  struct ggml_context *ctx = ggml_init(params);
  ggml_free(ctx);
}

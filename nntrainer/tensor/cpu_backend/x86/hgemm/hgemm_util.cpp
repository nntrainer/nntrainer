// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Yonghyeon Cho <dyddyd8574@gmail.com>
 *
 * @file   hgemm_util.cpp
 * @date   15 May 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Yonghyeon Cho <dyddyd8574@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  Utility helpers for x86 FP16 GEMM
 */

#include "hgemm_util.h"

#include "hgemm_common.h"

#include <avx2_impl.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <initializer_list>
#include <new>
#include <type_traits>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

#if defined(_WIN32)
#include <malloc.h> // _aligned_malloc / _aligned_free
#endif

namespace nntrainer::hgemm::internal {

namespace {

/** @brief Decoded x86 CPU vendor string and family/model identification */
struct X86CpuInfo {
  char vendor[13] = {};
  unsigned int family = 0;
  unsigned int model = 0;
};

bool read_cpuid(unsigned int leaf, unsigned int subleaf, unsigned int &eax,
                unsigned int &ebx, unsigned int &ecx, unsigned int &edx) {
#if defined(_MSC_VER)
  int regs[4] = {};
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  eax = static_cast<unsigned int>(regs[0]);
  ebx = static_cast<unsigned int>(regs[1]);
  ecx = static_cast<unsigned int>(regs[2]);
  edx = static_cast<unsigned int>(regs[3]);
  return true;
#elif defined(__GNUC__) || defined(__clang__)
  return __get_cpuid_count(leaf, subleaf, &eax, &ebx, &ecx, &edx) != 0;
#else
  (void)leaf;
  (void)subleaf;
  eax = ebx = ecx = edx = 0;
  return false;
#endif
}

X86CpuInfo detect_x86_cpu() {
  X86CpuInfo info;
  unsigned int eax = 0;
  unsigned int ebx = 0;
  unsigned int ecx = 0;
  unsigned int edx = 0;

  if (!read_cpuid(0, 0, eax, ebx, ecx, edx)) {
    return info;
  }

  std::memcpy(info.vendor + 0, &ebx, sizeof(ebx));
  std::memcpy(info.vendor + 4, &edx, sizeof(edx));
  std::memcpy(info.vendor + 8, &ecx, sizeof(ecx));

  if (!read_cpuid(1, 0, eax, ebx, ecx, edx)) {
    return info;
  }

  const unsigned int base_family = (eax >> 8) & 0xf;
  const unsigned int base_model = (eax >> 4) & 0xf;
  const unsigned int ext_family = (eax >> 20) & 0xff;
  const unsigned int ext_model = (eax >> 16) & 0xf;

  info.family = base_family;
  if (base_family == 0xf) {
    info.family += ext_family;
  }
  info.model = base_model;
  if (base_family == 0x6 || base_family == 0xf) {
    info.model += ext_model << 4;
  }

  return info;
}

bool is_vendor(const X86CpuInfo &info, const char *vendor) {
  return std::strncmp(info.vendor, vendor, 12) == 0;
}

bool is_intel_model(unsigned int model,
                    std::initializer_list<unsigned int> ms) {
  for (unsigned int m : ms) {
    if (model == m) {
      return true;
    }
  }
  return false;
}

HgemmBlockSizes tune_block_sizes() {
  const X86CpuInfo cpu = detect_x86_cpu();

  if (is_vendor(cpu, "GenuineIntel") && cpu.family == 0x6) {
    if (is_intel_model(cpu.model, {0x3c, 0x3f, 0x45, 0x46, 0x47, 0x4f, 0x56})) {
      return {256, 384, 256, 96, 128}; // Haswell/Broadwell-era private L2.
    }
    if (is_intel_model(cpu.model, {0x4e, 0x5e, 0x8e, 0x9e})) {
      return {384, 512, 256, 96, 128}; // Skylake/Kaby/Coffee client.
    }
    if (is_intel_model(cpu.model, {0x55, 0x6a, 0x6c, 0x8f})) {
      return {768, 512, 256, 192, 128}; // Skylake-X/Ice Lake server.
    }
    if (is_intel_model(cpu.model, {0x97, 0x9a, 0xb7, 0xba, 0xbf})) {
      return {1024, 512, 256, 192, 128}; // Alder/Raptor Lake class cores.
    }
  }

  if (is_vendor(cpu, "AuthenticAMD")) {
    if (cpu.family == 0x17) {
      return {384, 384, 256, 128, 128}; // Zen/Zen+/Zen2, 512KB L2.
    }
    if (cpu.family == 0x19) {
      return {512, 512, 256, 192, 128}; // Zen3/Zen4, 512KB+ L2.
    }
  }

  return {X86_HGEMM_M_BLOCKING, X86_HGEMM_N_BLOCKING, X86_HGEMM_K_BLOCKING,
          X86_HGEMM_C_M_BLOCKING, X86_HGEMM_C_N_BLOCKING};
}

} // namespace

const HgemmBlockSizes &get_hgemm_block_sizes() {
  static const HgemmBlockSizes sizes = tune_block_sizes();
  return sizes;
}

float *aligned_alloc_f32(std::size_t n_floats) {
  // Round up so the byte count is a multiple of the 64-byte alignment, which
  // std::aligned_alloc requires (C11 size must be a multiple of alignment).
  const std::size_t bytes = ((n_floats * sizeof(float) + 63u) / 64u) * 64u;
#if defined(_WIN32)
  void *p = _aligned_malloc(bytes, 64);
#else
  void *p = std::aligned_alloc(64, bytes);
#endif
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return static_cast<float *>(p);
}

void aligned_free(float *p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

namespace {

/// Widen one C row into the FP32 accumulator. F16C for _FP16, plain copy for
/// float so the FP32-output GEMMs avoid a needless conversion pass.
template <typename CType>
inline void widen_C_row(const CType *src, float *dst, unsigned int N) {
  if constexpr (std::is_same_v<CType, float>) {
    std::memcpy(dst, src, static_cast<std::size_t>(N) * sizeof(float));
  } else {
    nntrainer::avx2::vcvt_f16_f32(N, src, dst);
  }
}

/// Narrow one FP32 accumulator row back into C (F16C for _FP16, copy for
/// float).
template <typename CType>
inline void narrow_C_row(const float *src, CType *dst, unsigned int N) {
  if constexpr (std::is_same_v<CType, float>) {
    std::memcpy(dst, src, static_cast<std::size_t>(N) * sizeof(float));
  } else {
    nntrainer::avx2::vcvt_f32_f16(N, src, dst);
  }
}

} // namespace

template <typename CType>
void copy_C_to_C32(const CType *C, float *C32, unsigned int M, unsigned int N,
                   unsigned int c_stride, unsigned int c32_stride, float beta) {
  if (std::fpclassify(beta) == FP_ZERO) {
    const std::size_t row_bytes = static_cast<std::size_t>(N) * sizeof(float);
    for (unsigned int m = 0; m < M; ++m) {
      std::memset(C32 + m * c32_stride, 0, row_bytes);
    }
    return;
  }

  const bool beta_one = (beta == 1.0F);
  for (unsigned int m = 0; m < M; ++m) {
    float *row = C32 + m * c32_stride;
    widen_C_row<CType>(C + m * c_stride, row, N);
    if (!beta_one) {
      const __m256 vbeta = _mm256_set1_ps(beta);
      unsigned int n = 0;
      for (; n + 8 <= N; n += 8) {
        __m256 v = _mm256_loadu_ps(row + n);
        _mm256_storeu_ps(row + n, _mm256_mul_ps(vbeta, v));
      }
      for (; n < N; ++n) {
        row[n] *= beta;
      }
    }
  }
}

template <typename CType>
void copy_C32_to_C(const float *C32, CType *C, unsigned int M, unsigned int N,
                   unsigned int c32_stride, unsigned int c_stride) {
  for (unsigned int m = 0; m < M; ++m) {
    narrow_C_row<CType>(C32 + m * c32_stride, C + m * c_stride, N);
  }
}

template <typename CType>
void apply_beta_to_C(CType *C, unsigned int M, unsigned int N,
                     unsigned int c_stride, float beta) {
  if (std::fpclassify(beta) == FP_ZERO) {
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int n = 0; n < N; ++n) {
        C[m * c_stride + n] = static_cast<CType>(0.0F);
      }
    }
    return;
  }

  if (beta == 1.0F) {
    return;
  }

  for (unsigned int m = 0; m < M; ++m) {
    for (unsigned int n = 0; n < N; ++n) {
      const std::size_t idx = static_cast<std::size_t>(m) * c_stride + n;
      C[idx] = static_cast<CType>(beta * static_cast<float>(C[idx]));
    }
  }
}

template void copy_C_to_C32<_FP16>(const _FP16 *, float *, unsigned int,
                                   unsigned int, unsigned int, unsigned int,
                                   float);
template void copy_C_to_C32<float>(const float *, float *, unsigned int,
                                   unsigned int, unsigned int, unsigned int,
                                   float);
template void copy_C32_to_C<_FP16>(const float *, _FP16 *, unsigned int,
                                   unsigned int, unsigned int, unsigned int);
template void copy_C32_to_C<float>(const float *, float *, unsigned int,
                                   unsigned int, unsigned int, unsigned int);
template void apply_beta_to_C<_FP16>(_FP16 *, unsigned int, unsigned int,
                                     unsigned int, float);
template void apply_beta_to_C<float>(float *, unsigned int, unsigned int,
                                     unsigned int, float);

} /* namespace nntrainer::hgemm::internal */

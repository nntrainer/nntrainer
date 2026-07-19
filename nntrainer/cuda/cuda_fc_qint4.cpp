// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_fc_qint4.cpp
 * @date    23 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   Fused QS4CX dequant-GEMM implementation (NVRTC kernel).
 */

#include "cuda_fc_qint4.h"

#include <cuda_blas_manager.h>
#include <cuda_context.h>
#include <cuda_context_manager.h>
#include <cuda_stream_manager.h>

#include <nntrainer_log.h>

#include <cstdlib>
#include <map>
#include <mutex>
#include <unordered_map>

#include <cuda_runtime.h>
#include <fp16.h>

namespace nntrainer::cuda {

// One thread per output element Y[m,n]; loops K reading the int4 weight from
// the QS4CX PLAIN payload (row-major [N][Kh] bytes, even k = low nibble, stored
// uint4 = int4+8), dequantizing inline and scaling by the per-channel fp16
// scale. float accumulation. cvt_h2f / cvt_f2h are the fp16<->fp32 element
// converters the fp16-activation path stages through.
static const char *FC_QINT4_PLAIN_SRC = R"CU(
extern "C" {

__device__ __forceinline__ float plain_h2f(unsigned short h) {
  unsigned int sign = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int exp = (h >> 10) & 0x1Fu;
  unsigned int mant = h & 0x3FFu;
  unsigned int out;
  if (exp == 0u) {
    if (mant == 0u) {
      out = sign;
    } else {
      int e = -1;
      do { mant <<= 1; e++; } while ((mant & 0x400u) == 0u);
      mant &= 0x3FFu;
      out = sign | ((unsigned int)(127 - 15 - e) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    out = sign | 0x7F800000u | (mant << 13);
  } else {
    out = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
  }
  return __int_as_float((int)out);
}

__device__ __forceinline__ unsigned short plain_f2h(float f) {
  unsigned int x = (unsigned int)__float_as_int(f), s = (x >> 16) & 0x8000u,
               mant = x & 0x7FFFFFu;
  int e = (int)((x >> 23) & 0xFFu);
  if (e == 0xFF) return (unsigned short)(s | 0x7C00u | (mant ? 0x200u : 0u));
  int exp = e - 127 + 15;
  if (exp >= 0x1F) return (unsigned short)(s | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10) return (unsigned short)s;
    mant |= 0x800000u; int sh = 14 - exp;
    unsigned int hh = mant >> sh, rem = mant & ((1u << sh) - 1u),
                 half = 1u << (sh - 1);
    if (rem > half || (rem == half && (hh & 1u))) hh++;
    return (unsigned short)(s | hh);
  }
  unsigned int hh = ((unsigned int)exp << 10) | (mant >> 13), rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (hh & 1u))) hh++;
  return (unsigned short)(s | hh);
}

__global__ void fc_qint4_plain_gemm(const float *X, const unsigned char *W,
                                    const unsigned short *sc, float *Y, int M,
                                    int N, int K, int Kh) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  const unsigned char *wrow = W + (long)n * Kh;
  const float *xr = X + (long)m * K;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    unsigned char b = wrow[k >> 1];
    int nib = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
    acc += xr[k] * (float)(nib - 8);
  }
  Y[(long)m * N + n] = acc * plain_h2f(sc[n]);
}

__global__ void cvt_f2h(const float *src, unsigned short *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = plain_f2h(src[i]);
}

__global__ void cvt_h2f(const unsigned short *src, float *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = plain_h2f(src[i]);
}

}
)CU";

static const char *FC_QINT4_DP4A_SRC =
  R"CU(
extern "C" {

__device__ __forceinline__ float dp4a_h2f(unsigned short h) {
  unsigned int sign = ((unsigned int)(h & 0x8000u)) << 16;
  unsigned int exp = (h >> 10) & 0x1Fu;
  unsigned int mant = h & 0x3FFu;
  unsigned int out;
  if (exp == 0u) {
    if (mant == 0u) {
      out = sign;
    } else {
      int e = -1;
      do { mant <<= 1; e++; } while ((mant & 0x400u) == 0u);
      mant &= 0x3FFu;
      out = sign | ((unsigned int)(127 - 15 - e) << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    out = sign | 0x7F800000u | (mant << 13);
  } else {
    out = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
  }
  return __int_as_float((int)out);
}

// float -> fp16 (IEEE half), round to nearest even.
__device__ __forceinline__ unsigned short dp4a_f2h(float f) {
  unsigned int x = (unsigned int)__float_as_int(f);
  unsigned int sign = (x >> 16) & 0x8000u;
  int e = (int)((x >> 23) & 0xFFu);
  unsigned int mant = x & 0x7FFFFFu;
  if (e == 0xFF)
    return (unsigned short)(sign | 0x7C00u | (mant ? 0x200u : 0u)); // inf/nan
  int exp = e - 127 + 15;
  if (exp >= 0x1F)
    return (unsigned short)(sign | 0x7C00u); // overflow -> inf
  if (exp <= 0) {
    if (exp < -10)
      return (unsigned short)sign; // underflow -> 0
    mant |= 0x800000u;
    int shift = 14 - exp;
    unsigned int h = mant >> shift;
    unsigned int rem = mant & ((1u << shift) - 1u);
    unsigned int half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1u)))
      h++;
    return (unsigned short)(sign | h);
  }
  unsigned int h = ((unsigned int)exp << 10) | (mant >> 13);
  unsigned int rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u)))
    h++;
  return (unsigned short)(sign | h);
}

// asymmetric int8 quant params for a row's [min,max] (range forced to include
// 0, nudged zero-point) -- mirrors the OpenCL v8c act-quant. Returns recip
// (dequant scale) and zp; sets scale_q (quant multiplier) by reference.
__device__ __forceinline__ void asym_qparams(float fmn, float fmx,
                                             float &scale_q, float &recip,
                                             int &zp) {
  float rmin = fminf(0.f, fmn), rmax = fmaxf(0.f, fmx);
  float range = rmax - rmin;
  scale_q = range > 0.f ? 255.f / range : 1.f;
  recip = range > 0.f ? range / 255.f : 1.f;
  float dmin = rmin * scale_q, dmax = rmax * scale_q;
  float zp_lo = -128.f - dmin, zp_hi = 127.f - dmax;
  float zp_f = ((-128.f + dmin) + (127.f + dmax) > 0.f) ? zp_lo : zp_hi;
  zp_f = fmaxf(-128.f, fminf(127.f, zp_f));
  zp = (int)rintf(zp_f);
}

// per-row asymmetric int8 quant of an fp16 activation (one block per row).
// stores recip in ascale[m], zero-point in azp[m].
__global__ void act_quant_i8_h(const unsigned short *Xh, signed char *q8,
                               float *ascale, int *azp, int M, int K) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  __shared__ float smn[256];
  __shared__ float smx[256];
  const unsigned short *xr = Xh + (long)m * K;
  float lmn = 0.f, lmx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = dp4a_h2f(xr[k]);
    lmn = fminf(lmn, v);
    lmx = fmaxf(lmx, v);
  }
  smn[threadIdx.x] = lmn;
  smx[threadIdx.x] = lmx;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x + s]);
      smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x + s]);
    }
    __syncthreads();
  }
  float scale_q, recip;
  int zp;
  asym_qparams(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) {
    ascale[m] = recip;
    azp[m] = zp;
  }
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = (int)rintf(dp4a_h2f(xr[k]) * scale_q) + zp;
    q = max(-128, min(127, q));
    q8[(long)m * K + k] = (signed char)q;
  }
}

// per-output-channel weight row-sum (sum of signed int4) for the activation
// zero-point correction: Y -= recip[m]*scale_w[n]*zp[m]*rowsum_w[n].
__global__ void weight_rowsum(const signed char *plain, int *rowsum, int N,
                              int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N)
    return;
  int Kh = (K + 1) >> 1;
  const signed char *wrow = plain + (long)n * Kh;
  int s = 0;
  for (int kb = 0; kb < Kh; ++kb) {
    int b = (unsigned char)wrow[kb];
    int k0 = 2 * kb, k1 = 2 * kb + 1;
    if (k0 < K)
      s += ((int)(signed char)(b << 4)) >> 4;
    if (k1 < K)
      s += ((int)(signed char)b) >> 4;
  }
  rowsum[n] = s;
}

// float buffer -> fp16 buffer.
__global__ void cvt_f2h(const float *src, unsigned short *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = dp4a_f2h(src[i]);
}

// fp16 buffer -> float buffer.
__global__ void cvt_h2f(const unsigned short *src, float *dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = dp4a_h2f(src[i]);
}

// signed int4 weight for (output n, input k) from the QS4CX plain payload
// (row-major [N][Kh] bytes, even k = low nibble, stored uint4 = int4+8).
__device__ __forceinline__ int plain_decode(const unsigned char *qw, int n,
                                            int k, int Kh) {
  unsigned char b = qw[(long)n * Kh + (k >> 1)];
  int nib = (k & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
  return nib - 8;
}

// per-row asymmetric int8 quant of the activation (one block per row).
__global__ void act_quant_i8(const float *X, signed char *q8, float *ascale,
                             int *azp, int M, int K) {
  int m = blockIdx.x;
  if (m >= M)
    return;
  __shared__ float smn[256];
  __shared__ float smx[256];
  const float *xr = X + (long)m * K;
  float lmn = 0.f, lmx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = xr[k];
    lmn = fminf(lmn, v);
    lmx = fmaxf(lmx, v);
  }
  smn[threadIdx.x] = lmn;
  smx[threadIdx.x] = lmx;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      smn[threadIdx.x] = fminf(smn[threadIdx.x], smn[threadIdx.x + s]);
      smx[threadIdx.x] = fmaxf(smx[threadIdx.x], smx[threadIdx.x + s]);
    }
    __syncthreads();
  }
  float scale_q, recip;
  int zp;
  asym_qparams(smn[0], smx[0], scale_q, recip, zp);
  if (threadIdx.x == 0) {
    ascale[m] = recip;
    azp[m] = zp;
  }
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = (int)rintf(xr[k] * scale_q) + zp;
    q = max(-128, min(127, q));
    q8[(long)m * K + k] = (signed char)q;
  }
}

// QS4CX plain -> signed packed int4 [N, ceil(K/2)]: byte[n][kb] low nibble =
// int4(n, 2kb), high nibble = int4(n, 2kb+1), each stored two's-complement.
// The source has the SAME [N][Kh] byte indexing with uint4 = int4+8 nibbles,
// and (x-8)&0xF == x^8 on a 4-bit lane, so the whole repack is one byte-wise
// XOR with 0x88 (an odd-K pad nibble 8 becomes signed 0, as before).
__global__ void repack_plain_i4(const unsigned char *qw, signed char *packed,
                                int N, int Kh) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (long long)N * Kh)
    packed[i] = (signed char)(qw[i] ^ 0x88);
}

)CU"
  // NOTE: split here into two adjacent raw-string literals — MSVC caps a single
  // string literal at 16380 bytes (C2026); the two concatenate
  // byte-identically.
  R"CU(
// Y[m,n] = recip[m]*w_scale[n]*(sum_k q8[m,k]*int4(n,k) - zp[m]*rowsum_w[n]),
// the asymmetric-activation dequant (zp from act_quant, rowsum_w from the
// weight). via __dp4a.
__global__ void dp4a_gemm(const signed char *q8, const signed char *plain,
                          const float *ascale, const int *azp,
                          const int *wrowsum, const unsigned short *wscale,
                          float *Y, int M, int N, int K, int out_fp16) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  int Kh = (K + 1) >> 1;
  const signed char *qrow = q8 + (long)m * K;
  const signed char *wrow = plain + (long)n * Kh;
  int acc = 0, k = 0;
  for (; k + 4 <= K; k += 4) {
    int a = *(const int *)(qrow + k); // lanes = act k,k+1,k+2,k+3
    int kb = k >> 1;
    int b0 = (unsigned char)wrow[kb];     // k(low), k+1(high)
    int b1 = (unsigned char)wrow[kb + 1]; // k+2(low), k+3(high)
    int w0 = ((int)(signed char)(b0 << 4)) >> 4;
    int w1 = ((int)(signed char)b0) >> 4;
    int w2 = ((int)(signed char)(b1 << 4)) >> 4;
    int w3 = ((int)(signed char)b1) >> 4;
    int w = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) |
            ((w3 & 0xFF) << 24);
    acc = __dp4a(a, w, acc);
  }
  for (; k < K; ++k) { // tail (none when K%32==0)
    int kb = k >> 1;
    int b = (unsigned char)wrow[kb];
    int wv = (k & 1) ? (((int)(signed char)b) >> 4)
                     : (((int)(signed char)(b << 4)) >> 4);
    acc += (int)qrow[k] * wv;
  }
  float r = (float)(acc - azp[m] * wrowsum[n]) * ascale[m] * dp4a_h2f(wscale[n]);
  if (out_fp16)
    ((unsigned short *)Y)[(long)m * N + n] = dp4a_f2h(r);
  else
    Y[(long)m * N + n] = r;
}

// Dedicated M=1 decode GEMV: one block per output n, threads split K and
// block-reduce. The tiled dp4a_gemm wastes 15/16 rows of its 16x16 block at M=1
// (94% idle) and reads weight rows with a stride; here every thread is active
// and reads a contiguous K-slice of one weight row (coalesced). Activation row
// is row 0 (q8). out_fp16 folds the fp16 conversion in.
__global__ void dp4a_gemv(const signed char *q8, const signed char *plain,
                          const float *ascale, const int *azp,
                          const int *wrowsum, const unsigned short *wscale,
                          float *Y, int N, int K, int out_fp16) {
  // One WARP per output n (warps_per_block outputs per block) -> N/warps_per_block
  // blocks instead of N, amortizing the per-block launch/epilogue overhead that
  // dominated the old one-block-per-tiny-output design. No shared memory, no
  // __syncthreads: each lane reads a coalesced K-slice of the weight row and the
  // warp-shuffle reduces. dp4a int32 accumulate is integer-associative so the
  // result is BIT-IDENTICAL to the block-reduce version. (llama.cpp MMVQ shape.)
  const int warps_per_block = blockDim.x >> 5;
  int n = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  if (n >= N)
    return;
  const int lane = threadIdx.x & 31;
  int Kh = (K + 1) >> 1;
  const signed char *wrow = plain + (long)n * Kh;
  int acc = 0;
  for (int k = lane * 4; k + 4 <= K; k += 32 * 4) {
    int a = *(const int *)(q8 + k);
    int kb = k >> 1; // = lane*2 -> 2-byte aligned for the short load (K even)
    unsigned int w16 = *(const unsigned short *)(wrow + kb);
    int b0 = w16 & 0xFF;
    int b1 = (w16 >> 8) & 0xFF;
    int w0 = ((int)(signed char)(b0 << 4)) >> 4;
    int w1 = ((int)(signed char)b0) >> 4;
    int w2 = ((int)(signed char)(b1 << 4)) >> 4;
    int w3 = ((int)(signed char)b1) >> 4;
    int w = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) |
            ((w3 & 0xFF) << 24);
    acc = __dp4a(a, w, acc);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) {
    float r = (float)(acc - azp[0] * wrowsum[n]) * ascale[0] *
              dp4a_h2f(wscale[n]);
    if (out_fp16)
      ((unsigned short *)Y)[n] = dp4a_f2h(r);
    else
      Y[n] = r;
  }
}

// Register-blocked dp4a GEMM: a 64x64 output tile per block; each of the 256
// threads accumulates a 4x4 micro-tile in registers, so a K-chunk of 32 staged
// once into shared memory feeds 16 dp4a per thread before the next load -- much
// higher arithmetic intensity than the 1-output-per-thread tiled kernel.
#define RB_BM 64
#define RB_BN 64
#define RB_BK 32
#define RB_TM 4
#define RB_TN 4
__global__ void dp4a_gemm_reg(const signed char *q8, const signed char *plain,
                              const float *ascale, const int *azp,
                              const int *wrowsum, const unsigned short *wscale,
                              float *Y, int M, int N, int K, int out_fp16) {
  __shared__ signed char As[RB_BM][RB_BK];
  __shared__ signed char Ws[RB_BN][RB_BK];
  int tx = threadIdx.x, ty = threadIdx.y; // 0..15 each
  int tid = ty * 16 + tx;
  int blockM = blockIdx.y * RB_BM, blockN = blockIdx.x * RB_BN;
  int Kh = (K + 1) >> 1;
  int acc[RB_TM][RB_TN];
#pragma unroll
  for (int i = 0; i < RB_TM; i++)
#pragma unroll
    for (int j = 0; j < RB_TN; j++)
      acc[i][j] = 0;
  for (int k0 = 0; k0 < K; k0 += RB_BK) {
    for (int e = tid; e < RB_BM * RB_BK; e += 256) {
      int i = e / RB_BK, j = e % RB_BK;
      int mm = blockM + i, kk = k0 + j;
      As[i][j] = (mm < M && kk < K) ? q8[(long)mm * K + kk] : (signed char)0;
    }
    for (int e = tid; e < RB_BN * RB_BK; e += 256) {
      int i = e / RB_BK, j = e % RB_BK;
      int nn = blockN + i, kk = k0 + j;
      signed char wv = 0;
      if (nn < N && kk < K) {
        unsigned char b = (unsigned char)plain[(long)nn * Kh + (kk >> 1)];
        wv = (kk & 1) ? (((signed char)b) >> 4)
                      : (((signed char)(b << 4)) >> 4);
      }
      Ws[i][j] = wv;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < RB_BK; kk += 4) {
      int af[RB_TM], wf[RB_TN];
#pragma unroll
      for (int i = 0; i < RB_TM; i++)
        af[i] = *(const int *)&As[ty * RB_TM + i][kk];
#pragma unroll
      for (int j = 0; j < RB_TN; j++)
        wf[j] = *(const int *)&Ws[tx * RB_TN + j][kk];
#pragma unroll
      for (int i = 0; i < RB_TM; i++)
#pragma unroll
        for (int j = 0; j < RB_TN; j++)
          acc[i][j] = __dp4a(af[i], wf[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < RB_TM; i++) {
    int row = blockM + ty * RB_TM + i;
    if (row >= M)
      continue;
    float as = ascale[row];
    int zp = azp[row];
#pragma unroll
    for (int j = 0; j < RB_TN; j++) {
      int col = blockN + tx * RB_TN + j;
      if (col < N) {
        float r =
          (float)(acc[i][j] - zp * wrowsum[col]) * as * dp4a_h2f(wscale[col]);
        if (out_fp16)
          ((unsigned short *)Y)[(long)row * N + col] = dp4a_f2h(r);
        else
          Y[(long)row * N + col] = r;
      }
    }
  }
}

// === cuBLAS INT8 IMMA (Tensor Core) prefill FC support ===
// The __dp4a kernels run on the int ALU (ceiling ~21 TOPS on Ada). cuBLAS int8
// IMMA runs on the Tensor Cores (~30 TOPS measured, ~10x our dp4a GEMM). These
// three kernels feed it: unpack the int4 weight -> int8 ONCE (cached), and the
// int32 GEMM result is bit-identical to the __dp4a acc, so the SAME dequant
// applies in the epilogue.

// int4 plain weight -> int8 [K,N] (w8[k*N+n] = int4(n,k)). Unpacked once and
// cached (weights are static), so cuBLAS reads contiguous int8 -- doing this per
// call would add a memory pass that erases the Tensor-Core win.
__global__ void repack_plain_i8_kn(const unsigned char *qw, signed char *w8,
                                   int N, int K, int Kh) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (n >= N || k >= K)
    return;
  w8[(long)k * N + n] = (signed char)plain_decode(qw, n, k, Kh);
}

// per-output-channel sum of the int8 weight column (k-strided), for the
// activation zero-point correction. one thread per output channel n.
__global__ void weight_rowsum_kn(const signed char *w8, int *rowsum, int N,
                                 int K) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N)
    return;
  long s = 0;
  for (int k = 0; k < K; ++k)
    s += (int)w8[(long)k * N + n];
  rowsum[n] = (int)s;
}

// dequant epilogue for the int8 IMMA GEMM: C is the int32 dot-product (== the
// __dp4a acc, bit-identical). Y[m,n]=(C - zp[m]*rowsum[n])*recip[m]*wscale[n].
__global__ void dequant_i32_fp16(const int *C, const float *ascale,
                                 const int *azp, const int *wrowsum,
                                 const unsigned short *wscale, unsigned short *Y,
                                 int M, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N)
    return;
  float r = (float)(C[(long)m * N + n] - azp[m] * wrowsum[n]) * ascale[m] *
            dp4a_h2f(wscale[n]);
  Y[(long)m * N + n] = dp4a_f2h(r);
}

}
)CU";

// [single weight copy] The QS4CX plain payload is consumed by the CUDA FC path
// directly (the OpenCL v8c kernel consumes it the same way) -- no host/UVM copy
// of the nibble payload. The only per-weight side allocation is this N-entry
// fp16 scale buffer: the dequant kernel reads the per-channel scale on device
// every call, while the tensor stores fp32 scales. Built once at first use,
// cached by the fp32-scale pointer with no erase (weights live for the process
// lifetime), never under a graph capture (a cudaMallocManaged inside capture
// invalidates it).
bool cuda_fc_qs4cx_scales_to_uvm_fp16(const float *fp32_scales, unsigned int N,
                                      const unsigned short **out_sc) {
  static std::map<const void *, unsigned short *> cache;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = cache.find(fp32_scales);
  if (it == cache.end()) {
    if (StreamManager::Global().isCapturing())
      return false;
    unsigned short *usc = nullptr;
    // [WDDM coherence] This buffer is host-WRITTEN once and device-READ every
    // FC call -- the pattern that is incoherent on cMA==0 managed memory. Use
    // pinned host-mapped (zero-copy, UVA same-pointer) there; managed
    // elsewhere.
    static const bool host_mapped = []() {
      const char *e = std::getenv("NNTR_CUDA_HOST_MAPPED");
      if (e != nullptr)
        return e[0] == '1';
      return !ContextManager::Global().concurrentManagedAccess();
    }();
    if (host_mapped) {
      if (cudaHostAlloc(&usc, sizeof(unsigned short) * (size_t)N,
                        cudaHostAllocMapped) != cudaSuccess)
        return false;
    } else if (cudaMallocManaged(&usc, sizeof(unsigned short) * (size_t)N) !=
               cudaSuccess)
      return false;
    for (unsigned int n = 0; n < N; ++n)
      usc[n] = compute_fp32_to_fp16(fp32_scales[n]);
    it = cache.emplace(fp32_scales, usc).first;
  }
  *out_sc = it->second;
  return true;
}

namespace {
// Reusable fp32 activation/output staging for the fp16-naive path (the plain
// GEMM is fp32-in/fp32-out). Grown on demand, kept for reuse.
float *g_stage_xf = nullptr;
size_t g_stage_xf_cap = 0;
float *g_stage_yf = nullptr;
size_t g_stage_yf_cap = 0;
std::mutex g_stage_mtx;

bool ensure_buf(void **buf, size_t *cap, size_t bytes) {
  if (bytes <= *cap)
    return true;
  // cudaMalloc/cudaFree inside a CUDA-graph stream capture invalidates it; bail
  // so the caller falls back rather than corrupting the graph.
  if (StreamManager::Global().isCapturing())
    return false;
  if (*buf)
    cudaFree(*buf);
  if (cudaMalloc(buf, bytes) != cudaSuccess) {
    *buf = nullptr;
    *cap = 0;
    return false;
  }
  *cap = bytes;
  return true;
}

// --- w4a8 dp4a fast path (the default int4 FC decode path) ---------------
/**
 * @brief Cached signed-packed-int4 repack of a QS4CX weight, keyed by the
 * plain host/UVM payload pointer (weight.getData()). Weights are static for
 * the model lifetime, so the derived device cache is built once and never
 * erased.
 */
struct DevWeightQ {
  signed char *plain = nullptr; // signed packed int4 [N, ceil(K/2)]
  int *rowsum = nullptr;        // per-channel sum of signed int4 [N]
};
std::unordered_map<const void *, DevWeightQ> g_dp4a_plain_cache;
// per-row int8 activation quant scratch (q8 + recip scale + zero-point).
signed char *g_dp4a_q8 = nullptr;
size_t g_dp4a_q8_cap = 0;
float *g_dp4a_ascale = nullptr; // per-row recip (dequant scale)
size_t g_dp4a_ascale_cap = 0;
int *g_dp4a_azp = nullptr; // per-row activation zero-point
size_t g_dp4a_azp_cap = 0;
std::mutex g_dp4a_mtx;
// +256B tail pad on the int8 activation scratch: the cuBLAS int8 IMMA GEMM (a
// later change) reads A with wide (>=16B) Tensor-Core loads that can run past
// the last element; sizing the shared scratch with the pad here keeps that
// change a pure add. The __dp4a path itself does not over-read.
static constexpr size_t FC_I8_TAIL_PAD = 256;

DevWeightQ *ensure_dp4a_cache_locked(const unsigned char *plain_w,
                                     unsigned int N, unsigned int K) {
  auto it = g_dp4a_plain_cache.find(plain_w);
  if (it != g_dp4a_plain_cache.end())
    return &it->second;
  const int n = (int)N, k = (int)K;
  const size_t Kh = (K + 1u) / 2u;
  auto kr = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                     "repack_plain_i4");
  auto krs = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "weight_rowsum");
  if (!kr || !krs)
    return nullptr;
  DevWeightQ dw;
  if (cudaMalloc(&dw.plain, (size_t)N * Kh) != cudaSuccess)
    return nullptr;
  if (cudaMalloc(&dw.rowsum, sizeof(int) * (size_t)N) != cudaSuccess) {
    cudaFree(dw.plain);
    return nullptr;
  }
  const int khi = (int)Kh;
  kr->SetKernelArguments(0, &plain_w, sizeof(plain_w));
  kr->SetKernelArguments(1, &dw.plain, sizeof(dw.plain));
  kr->SetKernelArguments(2, &n, sizeof(n));
  kr->SetKernelArguments(3, &khi, sizeof(khi));
  const int rb[3] = {256, 1, 1};
  const int rg[3] = {(int)(((size_t)N * Kh + 255) / 256), 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kr, rg, rb)) {
    cudaFree(dw.plain);
    cudaFree(dw.rowsum);
    return nullptr;
  }
  // per-channel weight row-sum (for the activation zero-point correction).
  krs->SetKernelArguments(0, &dw.plain, sizeof(dw.plain));
  krs->SetKernelArguments(1, &dw.rowsum, sizeof(dw.rowsum));
  krs->SetKernelArguments(2, &n, sizeof(n));
  krs->SetKernelArguments(3, &k, sizeof(k));
  const int sb[3] = {128, 1, 1};
  const int sg[3] = {((int)N + 127) / 128, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*krs, sg, sb)) {
    cudaFree(dw.plain);
    cudaFree(dw.rowsum);
    return nullptr;
  }
  it = g_dp4a_plain_cache.emplace(plain_w, dw).first;
  return &it->second;
}

// repack (cached) + GEMM into a device float Y, using the already-staged
// q8/ascale scratch. Caller holds g_dp4a_mtx and has run act-quant.
bool dp4a_repack_and_gemm(const unsigned char *plain_w,
                          const unsigned short *scales_fp16, float *Yf,
                          unsigned int M, unsigned int N, unsigned int K,
                          int out_fp16 = 0) {
  const int n = (int)N, k = (int)K;
  const bool gemv = (M == 1);
  const bool tiled = (M >= 8);
  auto kg = CudaContext::Global().registerCudaKernel(
    FC_QINT4_DP4A_SRC,
    gemv ? "dp4a_gemv" : (tiled ? "dp4a_gemm_reg" : "dp4a_gemm"));
  if (!kg)
    return false;

  DevWeightQ *dwp = ensure_dp4a_cache_locked(plain_w, N, K);
  if (!dwp)
    return false;
  signed char *plain = dwp->plain;
  int *wrowsum = dwp->rowsum;

  const int mm = (int)M;
  kg->SetKernelArguments(0, &g_dp4a_q8, sizeof(g_dp4a_q8));
  kg->SetKernelArguments(1, &plain, sizeof(plain));
  kg->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kg->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
  kg->SetKernelArguments(4, &wrowsum, sizeof(wrowsum));
  kg->SetKernelArguments(5, &scales_fp16, sizeof(scales_fp16));
  kg->SetKernelArguments(6, &Yf, sizeof(Yf));
  if (gemv) {
    // dp4a_gemv: one WARP per output, 4 warps (128 threads) per block ->
    // ceil(N/4) blocks instead of N (4x fewer per-block launch/epilogue
    // overheads).
    kg->SetKernelArguments(7, &n, sizeof(n));
    kg->SetKernelArguments(8, &k, sizeof(k));
    kg->SetKernelArguments(9, &out_fp16, sizeof(out_fp16));
    const int gvb[3] = {128, 1, 1};
    const int gvg[3] = {((int)N + 3) / 4, 1, 1};
    return StreamManager::Global().DispatchCommand(*kg, gvg, gvb);
  }
  kg->SetKernelArguments(7, &mm, sizeof(mm));
  kg->SetKernelArguments(8, &n, sizeof(n));
  kg->SetKernelArguments(9, &k, sizeof(k));
  kg->SetKernelArguments(10, &out_fp16, sizeof(out_fp16));
  const int gb[3] = {16, 16, 1};
  const int tile = tiled ? 64 : 16;
  const int gg[3] = {((int)N + tile - 1) / tile, ((int)M + tile - 1) / tile, 1};
  return StreamManager::Global().DispatchCommand(*kg, gg, gb);
}

static bool dp4a_stage_scratch(unsigned int M, unsigned int K) {
  return ensure_buf((void **)&g_dp4a_q8, &g_dp4a_q8_cap,
                    (size_t)M * K + FC_I8_TAIL_PAD) &&
         ensure_buf((void **)&g_dp4a_ascale, &g_dp4a_ascale_cap,
                    sizeof(float) * (size_t)M) &&
         ensure_buf((void **)&g_dp4a_azp, &g_dp4a_azp_cap,
                    sizeof(int) * (size_t)M);
}

// --- cuBLAS int8 IMMA (Tensor Core) prefill weight cache ------------------
/**
 * @brief int8-unpacked weight [K,N] + per-channel rowsum for the cuBLAS int8
 * path, keyed by the QS4CX plain payload pointer (unpacked once, weights are
 * static).
 */
struct DevWeightI8 {
  signed char *w8 = nullptr; // int8 weight [K,N] (w8[k*N+n] = int4(n,k))
  int *rowsum = nullptr;     // per-channel sum of int8 weight [N]
};
std::unordered_map<const void *, DevWeightI8> g_i8_weight_cache;
int *g_i8_c = nullptr; // int32 GEMM output scratch [Mpad,N]
size_t g_i8_c_cap = 0;
// act-quant dedup (opt-in NNTR_QUANT_DEDUP): sibling FCs sharing an input
// activation reuse the first's int8 quant in g_dp4a_q8.
const void *g_last_quant_xh = nullptr;
int g_last_quant_k = 0;

static DevWeightI8 *ensure_i8_cache_locked(const unsigned char *plain_w,
                                           unsigned int N, unsigned int K) {
  auto it = g_i8_weight_cache.find(plain_w);
  if (it != g_i8_weight_cache.end())
    return &it->second;
  const int n = (int)N, k = (int)K, kh = (int)((K + 1u) / 2u);
  auto krp = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "repack_plain_i8_kn");
  auto krs = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "weight_rowsum_kn");
  if (!krp || !krs)
    return nullptr;
  DevWeightI8 dw;
  if (cudaMalloc(&dw.w8, (size_t)N * K + FC_I8_TAIL_PAD) != cudaSuccess)
    return nullptr;
  if (cudaMalloc(&dw.rowsum, sizeof(int) * (size_t)N) != cudaSuccess) {
    cudaFree(dw.w8);
    return nullptr;
  }
  krp->SetKernelArguments(0, &plain_w, sizeof(plain_w));
  krp->SetKernelArguments(1, &dw.w8, sizeof(dw.w8));
  krp->SetKernelArguments(2, &n, sizeof(n));
  krp->SetKernelArguments(3, &k, sizeof(k));
  krp->SetKernelArguments(4, &kh, sizeof(kh));
  const int pb[3] = {16, 16, 1};
  const int pg[3] = {((int)N + 15) / 16, ((int)K + 15) / 16, 1};
  if (!StreamManager::Global().DispatchCommand(*krp, pg, pb)) {
    cudaFree(dw.w8);
    cudaFree(dw.rowsum);
    return nullptr;
  }
  krs->SetKernelArguments(0, &dw.w8, sizeof(dw.w8));
  krs->SetKernelArguments(1, &dw.rowsum, sizeof(dw.rowsum));
  krs->SetKernelArguments(2, &n, sizeof(n));
  krs->SetKernelArguments(3, &k, sizeof(k));
  const int sb[3] = {128, 1, 1};
  const int sg[3] = {((int)N + 127) / 128, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*krs, sg, sb)) {
    cudaFree(dw.w8);
    cudaFree(dw.rowsum);
    return nullptr;
  }
  it = g_i8_weight_cache.emplace(plain_w, dw).first;
  return &it->second;
}

} // namespace

bool cuda_fc_qs4cx_gemm_fp32(const float *X, const unsigned char *plain_w,
                             const unsigned short *scales_fp16, float *Y,
                             unsigned int M, unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;

  auto kernel = CudaContext::Global().registerCudaKernel(FC_QINT4_PLAIN_SRC,
                                                         "fc_qint4_plain_gemm");
  if (!kernel) {
    ml_loge("[CUDA] fc_qint4_plain: kernel registration failed");
    return false;
  }

  int m = (int)M, n = (int)N, k = (int)K;
  int kh = (int)((K + 1u) / 2u);
  kernel->SetKernelArguments(0, &X, sizeof(X));
  kernel->SetKernelArguments(1, &plain_w, sizeof(plain_w));
  kernel->SetKernelArguments(2, &scales_fp16, sizeof(scales_fp16));
  kernel->SetKernelArguments(3, &Y, sizeof(Y));
  kernel->SetKernelArguments(4, &m, sizeof(m));
  kernel->SetKernelArguments(5, &n, sizeof(n));
  kernel->SetKernelArguments(6, &k, sizeof(k));
  kernel->SetKernelArguments(7, &kh, sizeof(kh));

  const int block[3] = {16, 16, 1};
  const int grid[3] = {((int)N + 15) / 16, ((int)M + 15) / 16, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

bool cuda_fc_qs4cx_gemm_fp16_naive(const unsigned short *Xh,
                                   const unsigned char *plain_w,
                                   const unsigned short *scales_fp16,
                                   unsigned short *Yh, unsigned int M,
                                   unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  auto kh2f =
    CudaContext::Global().registerCudaKernel(FC_QINT4_PLAIN_SRC, "cvt_h2f");
  auto kf2h =
    CudaContext::Global().registerCudaKernel(FC_QINT4_PLAIN_SRC, "cvt_f2h");
  if (!kh2f || !kf2h)
    return false;
  std::lock_guard<std::mutex> lk(g_stage_mtx);
  const size_t xn = (size_t)M * K, yn = (size_t)M * N;
  if (!ensure_buf((void **)&g_stage_xf, &g_stage_xf_cap, sizeof(float) * xn) ||
      !ensure_buf((void **)&g_stage_yf, &g_stage_yf_cap, sizeof(float) * yn))
    return false;
  int xni = (int)xn, yni = (int)yn;
  const int cb[3] = {256, 1, 1};
  kh2f->SetKernelArguments(0, &Xh, sizeof(Xh));
  kh2f->SetKernelArguments(1, &g_stage_xf, sizeof(g_stage_xf));
  kh2f->SetKernelArguments(2, &xni, sizeof(xni));
  const int xg[3] = {((int)xn + 255) / 256, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kh2f, xg, cb))
    return false;
  // naive plain-decode FP32-act GEMM (its own dispatch + drain).
  if (!cuda_fc_qs4cx_gemm_fp32(g_stage_xf, plain_w, scales_fp16, g_stage_yf, M,
                               N, K))
    return false;
  kf2h->SetKernelArguments(0, &g_stage_yf, sizeof(g_stage_yf));
  kf2h->SetKernelArguments(1, &Yh, sizeof(Yh));
  kf2h->SetKernelArguments(2, &yni, sizeof(yni));
  const int yg[3] = {((int)yn + 255) / 256, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kf2h, yg, cb))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

bool cuda_fc_qs4cx_dp4a_gemm_fp32(const float *X, const unsigned char *plain_w,
                                  const unsigned short *scales_fp16, float *Y,
                                  unsigned int M, unsigned int N,
                                  unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  auto kq =
    CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC, "act_quant_i8");
  if (!kq) {
    ml_loge("[CUDA] fc_qint4 dp4a: kernel registration failed");
    return false;
  }
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  if (!dp4a_stage_scratch(M, K))
    return false;
  int m = (int)M, k = (int)K;
  kq->SetKernelArguments(0, &X, sizeof(X));
  kq->SetKernelArguments(1, &g_dp4a_q8, sizeof(g_dp4a_q8));
  kq->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kq->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
  kq->SetKernelArguments(4, &m, sizeof(m));
  kq->SetKernelArguments(5, &k, sizeof(k));
  const int qb[3] = {256, 1, 1};
  const int qg[3] = {(int)M, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kq, qg, qb))
    return false;
  if (!dp4a_repack_and_gemm(plain_w, scales_fp16, Y, M, N, K))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

bool cuda_fc_qs4cx_dp4a_gemm_fp16(const unsigned short *Xh,
                                  const unsigned char *plain_w,
                                  const unsigned short *scales_fp16,
                                  unsigned short *Yh, unsigned int M,
                                  unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  auto kqh = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "act_quant_i8_h");
  auto kc =
    CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC, "cvt_f2h");
  if (!kqh || !kc) {
    ml_loge("[CUDA] fc_qint4 dp4a fp16: kernel registration failed");
    return false;
  }
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // No float Y staging here: the GEMM writes fp16 directly (out_fp16=1 below),
  // so g_dp4a_yf is unused on this path. Allocating it lazily would cudaMalloc
  // inside a CUDA-graph capture (NNTR_CUDA_GRAPH) on the first captured decode
  // token and invalidate the graph -- so it is deliberately NOT sized here.
  if (!dp4a_stage_scratch(M, K))
    return false;
  int m = (int)M, k = (int)K;
  // 1) int8 activation quant from the fp16 input.
  kqh->SetKernelArguments(0, &Xh, sizeof(Xh));
  kqh->SetKernelArguments(1, &g_dp4a_q8, sizeof(g_dp4a_q8));
  kqh->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kqh->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
  kqh->SetKernelArguments(4, &m, sizeof(m));
  kqh->SetKernelArguments(5, &k, sizeof(k));
  const int qb[3] = {256, 1, 1};
  const int qg[3] = {(int)M, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kqh, qg, qb))
    return false;
  // 2) repack + GEMM writing fp16 directly: the float->fp16 conversion is
  // folded into the GEMM epilogue (out_fp16=1), removing the separate cvt_f2h
  // kernel + the FP32 staging buffer (one fewer kernel per FC -- a decode
  // launch-overhead win). (void)kc keeps the registration check above harmless.
  (void)kc;
  if (!dp4a_repack_and_gemm(plain_w, scales_fp16, reinterpret_cast<float *>(Yh),
                            M, N, K,
                            /** out_fp16= */ 1))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

// [i8-jit] Optional transient JIT int8 weight unpack (NNTR_CUDA_I8_JIT): unpack
// the resident dp4a packed-int4 weight to int8 on the GPU per-prefill into a
// reusable scratch, instead of keeping a persistent per-weight int8 cache --
// trades a small per-call unpack cost for the cache's VRAM. Opt-in, default
// off.
static inline bool i8_jit_on() {
  static const bool v = []() {
    const char *e = std::getenv("NNTR_CUDA_I8_JIT");
    return e != nullptr && e[0] == '1';
  }();
  return v;
}

// Tiled transpose-unpack: dp4a packed [N, Kh] (byte = plain^0x88, nibbles =
// two's-complement signed 4-bit) -> int8 [K, N]. Reads coalesced along Kh,
// writes coalesced along N via the shared tile.
static const char *I8_JIT_SRC = R"CU(
extern "C" __global__ void i8_jit_unpack(const signed char *q4,
                                         signed char *w8, int N, int K,
                                         int Kh) {
  __shared__ signed char t[32][65];
  int nn0 = blockIdx.y * 32, kh0 = blockIdx.x * 32;
  int nn = nn0 + threadIdx.y, kh = kh0 + threadIdx.x;
  if (nn < N && kh < Kh) {
    unsigned char b = (unsigned char)q4[(long long)nn * Kh + kh];
    t[threadIdx.y][2 * threadIdx.x] =
      (signed char)((((b & 0xF) ^ 8) & 0xF) - 8);
    t[threadIdx.y][2 * threadIdx.x + 1] =
      (signed char)(((((b >> 4) & 0xF) ^ 8) & 0xF) - 8);
  }
  __syncthreads();
  int k0 = kh0 * 2, wn = nn0 + threadIdx.x;
  for (int kk = threadIdx.y; kk < 64; kk += 32) {
    int k = k0 + kk;
    if (k < K && wn < N)
      w8[(long long)k * N + wn] = t[threadIdx.x][kk];
  }
}

// Vectorized variant (K%8==0 && N%4==0 -- every FC shape this path accepts):
// 64n x 64k tile, 256 threads; uint (4-byte) global loads along Kh and int
// (4-byte) coalesced global stores along N -- runs the ~1.8GB/prefill unpack
// traffic at near-memcpy bandwidth instead of byte-granular transactions.
extern "C" __global__ void i8_jit_unpack_v4(const unsigned char *q4,
                                            signed char *w8, int N, int K,
                                            int Kh) {
  __shared__ signed char t[64][68]; // [k_local][n_local], row stride 68 (4B)
  const int nn0 = blockIdx.y * 64;
  const int kh0 = blockIdx.x * 32; // bytes of Kh covered by this tile
  const int tid = threadIdx.x;     // 256 threads
  for (int rep = 0; rep < 2; ++rep) {
    int idx = tid + rep * 256;
    int nn = idx >> 3;   // 0..63
    int kb4 = idx & 7;   // which 4-byte group in the 32-byte span
    int n = nn0 + nn;
    int khb = kh0 + kb4 * 4;
    if (n < N && khb + 3 < Kh) {
      unsigned int v = *reinterpret_cast<const unsigned int *>(
        q4 + (long long)n * Kh + khb);
      int kl = kb4 * 8;
      for (int j = 0; j < 4; ++j) {
        unsigned int b = (v >> (8 * j)) & 0xFFu;
        t[kl + 2 * j][nn] = (signed char)((((b & 0xF) ^ 8) & 0xF) - 8);
        t[kl + 2 * j + 1][nn] =
          (signed char)(((((b >> 4) & 0xF) ^ 8) & 0xF) - 8);
      }
    } else if (n < N) { // Kh tail (unused when K%8==0, kept for safety)
      for (int j = 0; j < 4; ++j) {
        int kb = khb + j;
        if (kb < Kh) {
          unsigned char b = q4[(long long)n * Kh + kb];
          int kl = kb4 * 8 + 2 * j;
          t[kl][nn] = (signed char)((((b & 0xF) ^ 8) & 0xF) - 8);
          t[kl + 1][nn] = (signed char)(((((b >> 4) & 0xF) ^ 8) & 0xF) - 8);
        }
      }
    }
  }
  __syncthreads();
  const int k0 = kh0 * 2;
  for (int rep = 0; rep < 4; ++rep) {
    int idx = tid + rep * 256;
    int kl = idx >> 4; // 0..63
    int ni = idx & 15; // 16 ints cover 64 n
    int k = k0 + kl;
    int n = nn0 + ni * 4;
    if (k < K && n + 3 < N) {
      int val = *reinterpret_cast<const int *>(&t[kl][ni * 4]);
      *reinterpret_cast<int *>(w8 + (long long)k * N + n) = val;
    } else if (k < K) {
      for (int j = 0; j < 4; ++j)
        if (n + j < N)
          w8[(long long)k * N + n + j] = t[kl][ni * 4 + j];
    }
  }
}
)CU";

// w4a8 on the INT8 Tensor Cores via cuBLAS (prefill FC). Same quant scheme as
// the dp4a path -- per-row asym int8 activation + symmetric int4 weight -- but
// the int8xint8->int32 GEMM runs on IMMA Tensor Cores instead of __dp4a on the
// int ALU (~10x the GEMM throughput at prefill M). The int32 accumulate is
// exact so the result is bit-identical to dp4a; the int4->int8 weight unpack is
// cached (one-time) to keep it off the per-call critical path.

bool cuda_fc_qs4cx_cublas_i8_gemm_fp16(const unsigned short *Xh,
                                       const unsigned char *plain_w,
                                       const unsigned short *scales_fp16,
                                       unsigned short *Yh, unsigned int M,
                                       unsigned int N, unsigned int K) {
  if (M == 0 || N == 0 || K == 0)
    return true;
  auto kqh = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "act_quant_i8_h");
  auto kde = CudaContext::Global().registerCudaKernel(FC_QINT4_DP4A_SRC,
                                                      "dequant_i32_fp16");
  if (!kqh || !kde) {
    ml_loge("[CUDA] fc_qint4 cublas-i8: kernel registration failed");
    return false;
  }
  std::lock_guard<std::mutex> lk(g_dp4a_mtx);
  // cuBLAS int8 IMMA requires the GEMM dims to be multiples of 32 (measured:
  // M=260/272 -> CUBLAS_STATUS_NOT_SUPPORTED, 256/320/512 OK). The prefill
  // token count M is arbitrary (e.g. 511), so pad the activation row count up
  // to a multiple of 32 for the GEMM only -- the extra rows are computed from
  // (harmless int8) scratch and ignored by the epilogue, which writes just the
  // real M rows. N and K are multiples of 32 by the load invariant.
  const unsigned Mpad = ((M + 31u) / 32u) * 32u;
  if (!dp4a_stage_scratch(Mpad, K))
    return false;
  const int m = (int)M, n = (int)N, k = (int)K, mpad = (int)Mpad;

  // 1) int8 activation quant from the fp16 input (reuse the dp4a quantizer).
  // Skip when this exact (Xh,K) was just quantized into g_dp4a_q8 by a sibling
  // FC (q/k/v share attention_norm; gate/up share ffn_norm) -- the buffer still
  // holds it. See g_last_quant_xh above.
  // Opt-in: measured gain is within the thermal noise floor on Orin (act_quant
  // is not on the critical path -- the GEMM is), so default OFF; correct +
  // ready if a less-throttled host or a power budget makes the redundant
  // launches matter.
  static const bool quant_dedup = []() {
    const char *e = std::getenv("NNTR_QUANT_DEDUP");
    return e != nullptr && e[0] == '1';
  }();
  const bool reuse_quant =
    quant_dedup && Xh == g_last_quant_xh && k == g_last_quant_k;
  if (!reuse_quant) {
    kqh->SetKernelArguments(0, &Xh, sizeof(Xh));
    kqh->SetKernelArguments(1, &g_dp4a_q8, sizeof(g_dp4a_q8));
    kqh->SetKernelArguments(2, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
    kqh->SetKernelArguments(3, &g_dp4a_azp, sizeof(g_dp4a_azp));
    kqh->SetKernelArguments(4, &m, sizeof(m));
    kqh->SetKernelArguments(5, &k, sizeof(k));
    const int qb[3] = {256, 1, 1};
    const int qg[3] = {(int)M, 1, 1};
    if (!StreamManager::Global().DispatchCommand(*kqh, qg, qb))
      return false;
    g_last_quant_xh = Xh;
    g_last_quant_k = k;
  }

  // 2) int8 weight [K,N] + per-channel rowsum. [i8-jit] JIT mode transpose-
  // unpacks the RESIDENT dp4a packed copy into a reusable scratch (nothing
  // stays resident; rowsum shared with the dp4a cache -- same values); else
  // the persistent per-weight cache (one-time unpack).
  signed char *w8src = nullptr;
  int *rowsum = nullptr;
  if (i8_jit_on()) {
    DevWeightQ *dw4 = ensure_dp4a_cache_locked(plain_w, N, K);
    if (!dw4)
      return false;
    static signed char *jit_w8 = nullptr;
    static size_t jit_cap = 0;
    if (!ensure_buf((void **)&jit_w8, &jit_cap, (size_t)K * N))
      return false;
    // Vectorized transpose for 8|K && 4|N (every eligible FC); byte-granular
    // fallback otherwise.
    const bool vec_ok = ((K & 7u) == 0u) && ((N & 3u) == 0u);
    auto ku = CudaContext::Global().registerCudaKernel(
      I8_JIT_SRC, vec_ok ? "i8_jit_unpack_v4" : "i8_jit_unpack");
    if (!ku)
      return false;
    const int khi = (int)((K + 1u) / 2u);
    ku->SetKernelArguments(0, &dw4->plain, sizeof(dw4->plain));
    ku->SetKernelArguments(1, &jit_w8, sizeof(jit_w8));
    ku->SetKernelArguments(2, &n, sizeof(n));
    ku->SetKernelArguments(3, &k, sizeof(k));
    ku->SetKernelArguments(4, &khi, sizeof(khi));
    const int ub[3] = {vec_ok ? 256 : 32, vec_ok ? 1 : 32, 1};
    const int ug[3] = {(khi + 31) / 32,
                       vec_ok ? ((int)N + 63) / 64 : ((int)N + 31) / 32, 1};
    if (!StreamManager::Global().DispatchCommand(*ku, ug, ub))
      return false;
    w8src = jit_w8;
    rowsum = dw4->rowsum;
  } else {
    // int8 weight [K,N] + per-channel rowsum from the persistent per-weight
    // cache (one-time unpack; weights are static).
    DevWeightI8 *dw8 = ensure_i8_cache_locked(plain_w, N, K);
    if (!dw8)
      return false;
    w8src = dw8->w8;
    rowsum = dw8->rowsum;
  }

  // 3) int32 GEMM output scratch [Mpad,N] (+tail pad: IMMA can write/read C in
  // wide vectorized tiles past the last element on large shapes).
  if (!ensure_buf((void **)&g_i8_c, &g_i8_c_cap,
                  sizeof(int) * (size_t)Mpad * N + FC_I8_TAIL_PAD))
    return false;

  // 4) INT8 IMMA GEMM on the Tensor Cores (Mpad rows; same backend stream as
  // the kernels). C is [Mpad,N] row-major; the real M rows are at the same
  // offsets so the epilogue reads C[m*N+n] for m<M directly.
  if (!BlasManager::Global().igemmRowMajor(mpad, n, k, g_dp4a_q8, w8src,
                                           g_i8_c))
    return false;

  // 5) dequant epilogue (bit-identical math to the dp4a kernel) -> fp16 Y.
  kde->SetKernelArguments(0, &g_i8_c, sizeof(g_i8_c));
  kde->SetKernelArguments(1, &g_dp4a_ascale, sizeof(g_dp4a_ascale));
  kde->SetKernelArguments(2, &g_dp4a_azp, sizeof(g_dp4a_azp));
  kde->SetKernelArguments(3, &rowsum, sizeof(rowsum));
  kde->SetKernelArguments(4, &scales_fp16, sizeof(scales_fp16));
  kde->SetKernelArguments(5, &Yh, sizeof(Yh));
  kde->SetKernelArguments(6, &m, sizeof(m));
  kde->SetKernelArguments(7, &n, sizeof(n));
  const int db[3] = {16, 16, 1};
  const int dg[3] = {((int)N + 15) / 16, ((int)M + 15) / 16, 1};
  if (!StreamManager::Global().DispatchCommand(*kde, dg, db))
    return false;
  StreamManager::Global().maybeFinish();
  // Catch an ASYNC failure in the cuBLAS IMMA GEMM / epilogue (the sync cuBLAS
  // status was already checked). On Orin a large-M IMMA can fault at runtime
  // and leave a STICKY cuda error -- which then makes the NEXT layer's
  // cudaPointerGetAttributes (rms_norm dev_ok gate) fail, dropping rms_norm to
  // its host path that reads device/managed activations under cMA=0 -> SIGSEGV.
  // Clearing + returning false makes the caller fall back to the (correct) dp4a
  // GEMM cleanly instead of corrupting the rest of the forward.
  {
    cudaError_t _e = cudaGetLastError();
    if (_e != cudaSuccess) {
      if (std::getenv("NNTR_IGEMM_DBG"))
        std::fprintf(
          stderr,
          "[IGEMM] async error after GEMM M=%d N=%d K=%d: %s -> dp4a "
          "fallback\n",
          m, n, k, cudaGetErrorString(_e));
      return false;
    }
  }
  return true;
}
// [wprefetch] Migrate a QS4CX weight's managed plain payload (+ fp32 scale
// tail) to the device with cudaMemPrefetchAsync, so the FC bytes leave host
// RSS and the GEMM reads them from VRAM. Discrete only (managed pages migrate).
bool cuda_fc_qs4cx_prefetch_weight(const unsigned char *plain_w, unsigned int N,
                                   unsigned int K) {
  if (plain_w == nullptr || N == 0 || K == 0)
    return false;
  if (ContextManager::Global().isIntegrated())
    return false;
  cudaPointerAttributes attr{};
  if (cudaPointerGetAttributes(&attr, plain_w) != cudaSuccess ||
      attr.type != cudaMemoryTypeManaged) {
    cudaGetLastError();
    return false;
  }
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  const size_t bytes = (size_t)N * ((K + 1u) / 2u) + (size_t)N * sizeof(float);
  // CUDA 13 signature (cudaMemLocation + flags).
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeDevice;
  loc.id = dev;
  if (cudaMemPrefetchAsync(plain_w, bytes, loc, /** flags */ 0,
                           StreamManager::Global().GetStream()) !=
      cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return true;
}

// Q6_K lm_head GEMV: FP16 hidden x (managed) Q6_K weight -> FP16 logits, all on
// the device. this is the GPU path gemma4 gets from its QINT4 untied lm_head.
static const char *Q6K_GEMV_SRC = R"CU(
#define QK_K 256
typedef unsigned char  u8;
typedef signed char    s8;
typedef unsigned short u16;

typedef struct { u8 ql[128]; u8 qh[64]; s8 scales[16]; u16 d; } block_q6_K;

__device__ __forceinline__ float h2f(u16 h) {
  unsigned int s = (unsigned int)(h & 0x8000) << 16;
  unsigned int e = (h >> 10) & 0x1F;
  unsigned int m = h & 0x3FF;
  unsigned int out;
  if (e == 0) {
    if (m == 0) { out = s; }
    else { e = 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF;
           out = s | ((e + 112) << 23) | (m << 13); }
  } else if (e == 31) { out = s | 0x7F800000u | (m << 13); }
  else { out = s | ((e + 112) << 23) | (m << 13); }
  return __int_as_float((int)out);
}

__device__ __forceinline__ u16 f2h(float f) {
  unsigned int x = (unsigned int)__float_as_int(f);
  unsigned int sign = (x >> 16) & 0x8000u;
  int exp = (int)((x >> 23) & 0xFF) - 127 + 15;
  unsigned int man = x & 0x7FFFFFu;
  if (exp <= 0) {
    if (exp < -10) return (u16)sign;
    man |= 0x800000u;
    unsigned int shift = (unsigned int)(14 - exp);
    unsigned int half = man >> shift;
    if ((man >> (shift - 1)) & 1u) half += 1; // round to nearest
    return (u16)(sign | half);
  } else if (exp >= 31) {
    return (u16)(sign | 0x7C00u);
  }
  unsigned int half = (unsigned int)(exp << 10) | (man >> 13);
  if ((man >> 12) & 1u) half += 1; // round to nearest
  return (u16)(sign | half);
}

extern "C" __global__ void q6k_gemv(const void *src0, const u16 *src1, u16 *dst,
                                    int ne00, int ne01) {
  const int N_SIMDWIDTH = 16;
  __shared__ float red[2][16];
  int nb = ne00 / QK_K;
  int row_group = threadIdx.x / N_SIMDWIDTH;
  int lane = threadIdx.x % N_SIMDWIDTH;
  int row = blockIdx.x * 2 + row_group;
  const block_q6_K *x = (const block_q6_K *)src0 + (long)row * nb;
  const u16 *yy = src1;
  u8 kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
  int tid = lane;
  int ip = tid / 8, il = tid % 8, l0 = 4 * il;
  int is = 8 * ip + l0 / 16;
  int y_offset = 128 * ip + l0;
  int q_offset_l = 64 * ip + l0;
  int q_offset_h = 32 * ip + l0;
  float sumf = 0.0f;
  if (row < ne01) {
    for (int i = 0; i < nb; i++) {
      const u8 *q1 = x[i].ql + q_offset_l;
      const u8 *q2 = q1 + QK_K / 8;
      const u8 *qh = x[i].qh + q_offset_h;
      const s8 *sc = x[i].scales + is;
      const u16 *y = yy + i * QK_K + y_offset;
      float dall = h2f(x[i].d);
      float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (int j = 0; j < 4; j++) {
        s0 += h2f(y[j + 0])  * ((float)((q1[j] & 0xF) | ((qh[j] & kmask1) << 4)) - 32.f);
        s1 += h2f(y[j + 32]) * ((float)((q2[j] & 0xF) | ((qh[j] & kmask2) << 2)) - 32.f);
        s2 += h2f(y[j + 64]) * ((float)((q1[j] >> 4)  | ((qh[j] & kmask3) >> 0)) - 32.f);
        s3 += h2f(y[j + 96]) * ((float)((q2[j] >> 4)  | ((qh[j] & kmask4) >> 2)) - 32.f);
      }
      sumf += dall * (s0 * sc[0] + s1 * sc[2] + s2 * sc[4] + s3 * sc[6]);
    }
  }
  red[row_group][lane] = sumf;
  __syncthreads();
  for (int off = N_SIMDWIDTH / 2; off > 0; off >>= 1) {
    if (lane < off) red[row_group][lane] += red[row_group][lane + off];
    __syncthreads();
  }
  if (lane == 0 && row < ne01)
    dst[row] = f2h(red[row_group][0]);
}
)CU";

bool lmhead_gemv_q6_k_cuda(const void *w_q6k_dev,
                           const unsigned short *hidden_fp16_dev,
                           unsigned short *logits_fp16_dev, int vocab,
                           int hidden) {
  if (vocab <= 0 || hidden <= 0 || (hidden % 256) != 0)
    return false;
  auto kernel =
    CudaContext::Global().registerCudaKernel(Q6K_GEMV_SRC, "q6k_gemv");
  if (!kernel)
    return false;
  kernel->SetKernelArguments(0, &w_q6k_dev, sizeof(w_q6k_dev));
  kernel->SetKernelArguments(1, &hidden_fp16_dev, sizeof(hidden_fp16_dev));
  kernel->SetKernelArguments(2, &logits_fp16_dev, sizeof(logits_fp16_dev));
  kernel->SetKernelArguments(3, &hidden, sizeof(hidden));
  kernel->SetKernelArguments(4, &vocab, sizeof(vocab));
  const int block[3] = {32, 1, 1};
  const int grid[3] = {(vocab + 1) / 2, 1, 1};
  if (!StreamManager::Global().DispatchCommand(*kernel, grid, block))
    return false;
  StreamManager::Global().maybeFinish();
  return true;
}

} // namespace nntrainer::cuda

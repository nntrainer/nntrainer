// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Portions of this file are derived from llama.cpp
 * (https://github.com/ggml-org/llama.cpp), licensed under the MIT License.
 * Copyright (c) Contributors to llama.cpp
 *
 * Modified by Pawel Debski, 2025: Adapted for CPU backend integration
 *
 * @file   nntr_ggml_impl_common.h
 * @date   20 August 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author  Pawel Debski <p.debski2@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Common structutures & functions definitions to be used in ggml
 * functions implementations
 */

#ifndef __NNTR_GGML_IMPL_COMMON__
#define __NNTR_GGML_IMPL_COMMON__

#include <assert.h>
#include <cmath>
#include <cstring>
#include <stdint.h>

typedef uint16_t nntr_fp16_t;

typedef uint16_t nntr_half;
typedef uint32_t nntr_half2;

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#define GROUP_MAX_EPS 1e-15f

#define K_SCALE_SIZE 12

#define Q4_0 32
#define Q8_0 32

#define QK_K 256

#define QK4_0 32
#define QK8_0 32

/**
 * @brief struct template for q4_0 and q8_0
 *
 * @tparam K 4 or 8
 * @return constexpr int number of elements in the quantized block
 */
template <int K> constexpr int QK_0() {
  if constexpr (K == 4) {
    return Q4_0;
  }
  if constexpr (K == 8) {
    return Q8_0;
  }
  return -1;
}

/**
 * @brief block_q4_K
 */
typedef struct {
  union {
    struct {
      uint16_t d;    // super-block scale for quantized scales
      uint16_t dmin; // super-block scale for quantized mins
    } data;
    uint32_t dm;
  } data;
  uint8_t scales[12];   // scales and mins, quantized with 6 bits
  uint8_t qs[QK_K / 2]; // 4--bit quants
} block_q4_K;

/**
 * @brief block_q6_K
 *
 */
typedef struct {
  uint8_t ql[QK_K / 2];     // quants, lower 4 bits
  uint8_t qh[QK_K / 4];     // quants, upper 2 bits
  int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
  nntr_half d;              // super-block scale
} block_q6_K;
/**
 * @brief block_q8_K
 *
 */
typedef struct {
  float d;                  // delta
  int8_t qs[QK_K];          // quants
  int16_t bsums[QK_K / 16]; // sum of quants in groups of 16
} block_q8_K;

/**
 * @brief block_q4_0
 */
typedef struct {
  nntr_half d;           // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

typedef struct {
  nntr_half d;      // delta
  int8_t qs[QK8_0]; // quants
} block_q8_0;

/**
 * @brief block of q4_0 or q8_0 block
 *
 * @tparam K 4 or 8
 * @tparam N number of blocks to be packed
 */
template <int K, int N> struct block {
  nntr_half d[N];                     // deltas for N qK_0 blocks
  int8_t qs[(QK_0<K>() * N * K) / 8]; // quants for N qK_0 blocks
};

using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;

/**
 * @brief block_q4_Kx8
 */
struct block_q4_Kx8 {
  nntr_half d[8];     // super-block scale for quantized scales
  nntr_half dmin[8];  // super-block scale for quantized mins
  uint8_t scales[96]; // scales and mins, quantized with 6 bits
  uint8_t qs[1024];   // 4--bit quants
};

/**
 * @brief block_q8_Kx4
 */
struct block_q8_Kx4 {
  float d[4];              // delta
  int8_t qs[QK_K * 4];     // quants
  int16_t bsums[QK_K / 4]; // sum of quants in groups of 16
};

float nntr_fp16_to_fp32(nntr_fp16_t h);
nntr_fp16_t nntr_fp32_to_fp16(float f);

float nntr_compute_fp16_to_fp32(nntr_fp16_t h);
nntr_fp16_t nntr_compute_fp32_to_fp16(float f);

inline int nearest_int(float fval) {
  assert(fabsf(fval) <= 4194303.f);
  float val = fval + 12582912.f;
  int i;
  memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

/**
 * @brief enum of ggml tensor type
 */
enum ggml_type {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_Q8_K = 15,
  GGML_TYPE_COUNT = 39,
};

inline int64_t ggml_blck_size(enum ggml_type type) {

  switch (type) {
  case GGML_TYPE_Q4_0: {
    return QK4_0;
  }
  case GGML_TYPE_Q8_0: {
    return QK8_0;
  }
  case GGML_TYPE_Q4_K: {
    return QK_K;
  }
  case GGML_TYPE_Q6_K: {
    return QK_K;
  }
  case GGML_TYPE_Q8_K: {
    return QK_K;
  }
  default: {
    break;
  }
  }

  return -1;
}

inline size_t ggml_type_size(enum ggml_type type) {

  switch (type) {
  case GGML_TYPE_Q4_0: {
    return sizeof(block_q4_0);
  }
  case GGML_TYPE_Q8_0: {
    return sizeof(block_q8_0);
  }
  case GGML_TYPE_Q4_K: {
    return sizeof(block_q4_K);
  }
  case GGML_TYPE_Q6_K: {
    return sizeof(block_q6_K);
  }
  case GGML_TYPE_Q8_K: {
    return sizeof(block_q8_K);
  }
  default: {
    break;
  }
  }

  return -1;
}

inline size_t ggml_row_size(enum ggml_type type, int64_t ne) {
  assert(ne % ggml_blck_size(type) == 0);
  return ggml_type_size(type) * ne / ggml_blck_size(type);
}

#endif

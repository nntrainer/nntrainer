/**
 * @file ggml_cuda_common.h
 * @brief Common definitions and structures for CUDA operations in GGML
 * @author Samsung R&D Institute
 * @bug No known bugs
 */
#pragma once

#include <cstdint>

#ifdef __CUDACC__
#include <cuda_fp16.h>
typedef half ggml_half;
typedef half2 ggml_half2;
#else
typedef uint16_t ggml_half;
typedef struct {
  uint16_t x;
  uint16_t y;
} ggml_half2;
#endif

#define QK8_1 32

// Macros for anonymous unions/structs
#ifdef _MSC_VER
#define GGML_EXTENSION
#else
#define GGML_EXTENSION __extension__
#endif

#define GGML_COMMON_AGGR_U
#define GGML_COMMON_AGGR_S data

typedef struct {
  GGML_EXTENSION union {
    struct {
      ggml_half d; // delta
      ggml_half s; // d * sum(qs[i])
    } GGML_COMMON_AGGR_S;
    ggml_half2 ds;
  } GGML_COMMON_AGGR_U;
  int8_t qs[QK8_1]; // quants
} block_q8_1;

enum ggml_type {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  GGML_TYPE_Q5_0 = 6,
  GGML_TYPE_Q5_1 = 7,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q8_1 = 9,
  // ... add others if needed
};

// Enum for MMQ Q8_1 data layout
enum mmq_q8_1_ds_layout {
  MMQ_Q8_1_DS_LAYOUT_D4,
  MMQ_Q8_1_DS_LAYOUT_DS4,
  MMQ_Q8_1_DS_LAYOUT_D2S6,
};

//==============================================================================
//
// Copyright (c) 2024 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  A header which defines the QNN GPU specialization of the QnnMem.h interface.
 */

#ifndef QNN_GPU_MEM_H
#define QNN_GPU_MEM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* QnnGpuMem_Buffer_t;

/**
 * @brief This enum defines QNN GPU memory type
 */
typedef enum { QNN_GPU_MEM_OPENCL = 0, QNN_GPU_MEM_UNDEFINED = 0x7FFFFFF } QnnGpu_MemType_t;

/**
 * @brief A struct which defines the QNN GPU memory preallocated by the client.
 *        Objects of this type are to be referenced through Qnn_MemInfoCustom_t.
 */
typedef struct {
  QnnGpu_MemType_t memType;
  union {
    QnnGpuMem_Buffer_t buffer;
  };
} QnnGpu_MemInfoCustom_t;

// clang-format off
/// QnnGpu_MemInfoCustom_t initializer macro
#define QNN_GPU_MEMINFO_CUSTOM_INIT                               \
  {                                                               \
    QNN_GPU_MEM_UNDEFINED, /*memType*/                            \
    NULL /* buffer*/                                              \
  }
// clang-format on

#ifdef __cplusplus
}  // extern "C"
#endif

#endif

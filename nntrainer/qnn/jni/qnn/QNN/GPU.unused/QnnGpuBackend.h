//==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  A header which defines the QNN GPU specialization of the QnnBackend.h interface.
 */

#ifndef QNN_GPU_BACKEND_H
#define QNN_GPU_BACKEND_H

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#include "QnnBackend.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* @brief This enum defines QNN GPU custom Backend config options.
*/
typedef enum {
  /// If non-zero, tuning mode will be enabled
  QNN_GPU_BACKEND_CONFIG_OPTION_ENABLE_TUNING_MODE = 0,
  /// The Performance cache directory. Must be non-null
  QNN_GPU_BACKEND_CONFIG_OPTION_PERFORMANCE_CACHE_DIR = 1,
  /// If non-zero, the performance cache will be ignored when initializing
  QNN_GPU_BACKEND_CONFIG_OPTION_INVALIDATE_PERFORMANCE_CACHE = 2,
  /// Unused, present to ensure 32 bits.
  QNN_GPU_BACKEND_CONFIG_OPTION_UNDEFINED = 0x7FFFFFFF,
} QnnGpuBackend_ConfigOption_t;

/**
 * @brief A struct which defines the QNN GPU Backend custom configuration options.
 *        Objects of this type are to be referenced through QnnBackend_CustomConfig_t.
 */
typedef struct {
   QnnGpuBackend_ConfigOption_t option;
   union UNNAMED {
      uint8_t enableTuningMode;
      const char* performanceCacheDir;
      uint8_t invalidatePerformanceCache;
   };
} QnnGpuBackend_CustomConfig_t;

// clang-format off
/// QnnGpuBackend_CustomConfig_t initializer macro
#define QNN_GPU_BACKEND_CUSTOM_CONFIG_INIT                        \
  {                                                               \
    QNN_GPU_BACKEND_CONFIG_OPTION_UNDEFINED, /*option*/           \
    {                                                             \
    false                            /*enableTuningMode*/         \
    }                                                             \
  }
// clang-format on

#ifdef __cplusplus
}  // extern "C"
#endif

#endif

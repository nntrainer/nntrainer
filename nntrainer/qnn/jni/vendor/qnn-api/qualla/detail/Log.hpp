//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <stdio.h>

// FIXME: Use logger from qualla::Env

#define QNN_INFO(fmt, ...)  fprintf(stderr, "[INFO]  " #fmt "\n", ##__VA_ARGS__)
#define QNN_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " #fmt "\n", ##__VA_ARGS__)
#define QNN_WARN(fmt, ...)  fprintf(stderr, "[WARN]  " #fmt "\n", ##__VA_ARGS__)

#if 0
    // #define NSP_LOG_LEVEL 2
#define QNN_DEBUG(fmt, ...) fprintf(stderr, "[DEBUG] " #fmt "\n", ##__VA_ARGS__)
#else
#define QNN_DEBUG(fmt, ...)
#endif

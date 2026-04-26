//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>

#include "QnnLog.h"

namespace qnn {
namespace log {
namespace utils {

// In non-hexagon app stdout is used and for hexagon farf logging is used
void logDefaultCallback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list argp);

static std::mutex sg_logUtilMutex;

}  // namespace utils
}  // namespace log
}  // namespace qnn

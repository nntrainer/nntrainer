//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <filesystem>
#include <string>

#include "PAL/DynamicLoading.hpp"

namespace genie {
namespace util {

inline std::filesystem::path getGenieLibDir() {
  static int s_anchor = 0;
  std::string libPath;
  if (pal::dynamicloading::dlAddrToLibName(reinterpret_cast<void *>(&s_anchor),
                                           libPath) != 0) {
    return std::filesystem::path(libPath).parent_path();
  }
  return {};
}

/// dlOpen wrapper that retries loading from the Genie library's directory
/// when the initial load fails and the path is relative.
inline void *dlOpenWrapper(const char *path, int flags) {
  void *handle = pal::dynamicloading::dlOpen(path, flags);
  if (nullptr == handle && !std::filesystem::path(path).is_absolute()) {
    auto genieDir = getGenieLibDir();
    if (!genieDir.empty()) {
      auto retryPath = genieDir / std::filesystem::path(path).filename();
      handle = pal::dynamicloading::dlOpen(retryPath.string().c_str(), flags);
    }
  }
  return handle;
}

} // namespace util
} // namespace genie

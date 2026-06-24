//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <string>
#include <vector>

#include "QnnGraph.h"
#include "QnnTypes.h"

struct BackendExtensionsConfigs {
  std::string sharedLibraryPath;
  std::string configFilePath;

  BackendExtensionsConfigs() : sharedLibraryPath(""), configFilePath("") {}

  BackendExtensionsConfigs(const std::string& _sharedLibraryPath,
                           const std::string& _configFilePath)
      : sharedLibraryPath(_sharedLibraryPath), configFilePath(_configFilePath) {}
};

struct GraphConfigs {
  std::string graphName;
  bool priorityPresent;
  Qnn_Priority_t priority;
  GraphConfigs() : graphName(), priorityPresent(false), priority(QNN_PRIORITY_UNDEFINED) {}
};

struct ConfigOptions {
  BackendExtensionsConfigs backendExtensionsConfigs;
  std::vector<GraphConfigs> graphConfigs;
  ConfigOptions() : backendExtensionsConfigs(), graphConfigs() {}
};

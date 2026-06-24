//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "IBackend.hpp"
#include "QnnConfig.hpp"
#include "ResourceManager.hpp"
#include "System/QnnSystemDlc.h"

class BackendExtensions final {
 public:
  BackendExtensions(BackendExtensionsConfigs backendExtensionsConfig,
                    void* backendLibHandle,
                    bool debug_qnn                                            = false,
                    QnnLog_Callback_t registeredLogCallback                   = nullptr,
                    QnnLog_Level_t qnnLogLevel                                = QNN_LOG_LEVEL_ERROR,
                    std::shared_ptr<genie::ResourceManager> m_resourceManager = nullptr);
  ~BackendExtensions();
  qnn::tools::netrun::IBackend* interface();

 private:
  qnn::tools::netrun::IBackend* m_backendInterface;
  qnn::tools::netrun::DestroyBackendInterfaceFnType_t m_destroyBackendInterfaceFn;
};

//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <stdexcept>

#include "BackendExtensions.hpp"
#include "Log/Logger.hpp"
#include "dlwrap.hpp"

BackendExtensions::BackendExtensions(BackendExtensionsConfigs backendExtensionsConfig,
                                     void* backendLibHandle,
                                     qnn::tools::netrun::PerfProfile perfProfile,
                                     bool debug_qnn,
                                     QnnLog_Callback_t registeredLogCallback,
                                     QnnLog_Level_t qnnLogLevel)
    : m_backendInterface(nullptr), m_destroyBackendInterfaceFn(nullptr) {
  QNN_DEBUG("DEBUG: backendExtensionsConfig.sharedLibraryPath=%s\n",
            backendExtensionsConfig.sharedLibraryPath.c_str());
  if (backendExtensionsConfig.sharedLibraryPath.empty()) {
    throw std::runtime_error("Empty backend extensions library path.");
  }

  QNN_DEBUG("DEBUG: backendExtensionsConfig.configFilePath=%s\n",
            backendExtensionsConfig.configFilePath.c_str());
  if (backendExtensionsConfig.configFilePath.empty()) {
    throw std::runtime_error("Empty backend extensions config path.");
  }

  void* libHandle =
      dlopen(backendExtensionsConfig.sharedLibraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (nullptr == libHandle) {
    QNN_ERROR("Unable to load backend extensions lib: [%s]. dlerror(): [%s]",
              backendExtensionsConfig.sharedLibraryPath.c_str(),
              dlerror());
    throw std::runtime_error("Unable to open backend extension library.");
  }

  auto createBackendInterfaceFn = (qnn::tools::netrun::CreateBackendInterfaceFnType_t)dlsym(
      libHandle, "createBackendInterface");
  if (nullptr == createBackendInterfaceFn) {
    throw std::runtime_error("Unable to resolve createBackendInterface.");
  }

  m_destroyBackendInterfaceFn = (qnn::tools::netrun::DestroyBackendInterfaceFnType_t)dlsym(
      libHandle, "destroyBackendInterface");
  if (nullptr == m_destroyBackendInterfaceFn) {
    throw std::runtime_error("Unable to resolve destroyBackendInterface.");
  }

  m_backendInterface = createBackendInterfaceFn();
  if (nullptr == m_backendInterface) {
    throw std::runtime_error("Unable to load backend extensions interface.");
  }

  if (debug_qnn) {
    if (!(m_backendInterface->setupLogging(registeredLogCallback, qnnLogLevel))) {
      throw std::runtime_error("Unable to initialize logging in backend extensions.");
    }
  }

  if (!m_backendInterface->initialize(backendLibHandle)) {
    throw std::runtime_error("Unable to initialize backend extensions interface.");
  }

  if (!m_backendInterface->setPerfProfile(perfProfile)) {
    QNN_WARN("Unable to set perf profile in  backend extensions interface.");
    // Do not throw
    // TODO: is this correct?
  }

  if (!m_backendInterface->loadConfig(backendExtensionsConfig.configFilePath)) {
    throw std::runtime_error("Unable to load backend extensions config.");
  }
}

BackendExtensions::~BackendExtensions() { m_destroyBackendInterfaceFn(m_backendInterface); }

qnn::tools::netrun::IBackend* BackendExtensions::interface() { return m_backendInterface; }

//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "PAL/DynamicLoading.hpp"
#include "System/QnnSystemInterface.h"
#include "qualla/detail/Log.hpp"

namespace genie {

class ResourceManager {
 public:
  ResourceManager();
  ResourceManager(const std::string& dlcPath);
  ~ResourceManager();

  // Accessors for the stored handle and interface (used by QnnApi injection).
  QnnSystemDlc_Handle_t getDlcHandle() const { return m_dlcHandle; }
  QNN_SYSTEM_INTERFACE_VER_TYPE getQnnSystemInterface() const { return m_qnnSystemInterface; }

  bool isDLCRecord(const std::string& name) const;

  size_t getFileSize(std::string filePath);

  size_t getDLCRecordSize(std::string name);

  size_t getBufferSize(std::string name);

  bool readBinaryFromFile(std::string filePath,
                          std::shared_ptr<uint8_t>& buffer,
                          size_t bufferSize);

  bool readBinaryFromDLC(const std::string& recordName, std::shared_ptr<uint8_t>& buffer);

  bool getBuffer(const std::string& name, std::shared_ptr<uint8_t>& buffer, size_t bufferSize);
  bool getBuffer(const std::string& name, std::shared_ptr<uint8_t>& buffer);

  /// dlOpen wrapper that retries loading from the Genie library's directory
  /// when the initial load fails and the path is relative.
  void* dlOpen(const char* path, int flags) const;

 private:
  std::string m_dlcPath;
  QnnSystemDlc_Handle_t m_dlcHandle{nullptr};
  Qnn_LogHandle_t m_logHandle{nullptr};
  void* m_systemLibraryHandle{nullptr};
  QNN_SYSTEM_INTERFACE_VER_TYPE m_qnnSystemInterface{};
  mutable std::mutex m_dlcMutex;
  typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t*** providerList,
                                                            uint32_t* numProviders);
  typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
      const QnnSystemInterface_t*** providerList, uint32_t* numProviders);
  // private helper function to create QNN System Interface from a library path.
  bool createQnnSystemInterface(std::string systemLibraryPath);
};

}  // namespace genie
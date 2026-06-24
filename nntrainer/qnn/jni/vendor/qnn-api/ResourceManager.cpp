//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <mutex>

#include "ResourceManager.hpp"
#include "qualla/detail/dlOpenWrapper.hpp"

namespace genie {

//=============================================================================
// ResourceManager functions
//=============================================================================
ResourceManager::ResourceManager() {
#ifdef _WIN32
  const std::string systemLibraryPath = "QnnSystem.dll";
#else
  const std::string systemLibraryPath = "libQnnSystem.so";
#endif
  if (!createQnnSystemInterface(systemLibraryPath)) {
    throw std::runtime_error("Resource manager not able to create QnnSystemInterface");
  }
}

ResourceManager::ResourceManager(const std::string& dlcPath) : m_dlcPath(dlcPath) {
#ifdef _WIN32
  const std::string systemLibraryPath = "QnnSystem.dll";
#else
  const std::string systemLibraryPath = "libQnnSystem.so";
#endif
  if (!createQnnSystemInterface(systemLibraryPath)) {
    throw std::runtime_error("Resource manager not able to create QnnSystemInterface");
  }

  if (!m_dlcHandle) {
    auto qnnError =
        m_qnnSystemInterface.systemDlcCreateFromFile(m_logHandle, m_dlcPath.c_str(), &m_dlcHandle);
    if (QNN_SUCCESS != QNN_GET_ERROR_CODE(qnnError) || !m_dlcHandle) {
      m_qnnSystemInterface.systemDlcFree(m_dlcHandle);
      if (m_systemLibraryHandle) {
        pal::dynamicloading::dlClose(m_systemLibraryHandle);
        m_systemLibraryHandle = nullptr;
      }
      throw std::runtime_error("Issue with retrieving the handle for dlc " + m_dlcPath);
    }
  }
}

bool ResourceManager::createQnnSystemInterface(std::string systemLibraryPath) {
  QnnSystemInterfaceGetProvidersFn_t getSystemInterfaceProviders{nullptr};

  m_systemLibraryHandle = dlOpen(systemLibraryPath.c_str(), pal::dynamicloading::DL_NOW);
  if (nullptr == m_systemLibraryHandle) {
    QNN_ERROR("Unable to load system library. pal::dynamicloading::dlError(): %s",
              pal::dynamicloading::dlError());
    return false;
  }

  // Get QNN System Interface
  getSystemInterfaceProviders = reinterpret_cast<QnnSystemInterfaceGetProvidersFn_t>(
      pal::dynamicloading::dlSym(m_systemLibraryHandle, "QnnSystemInterface_getProviders"));
  if (nullptr == getSystemInterfaceProviders) {
    return false;
  }

  uint32_t numProviders{0};
  QnnSystemInterface_t** systemInterfaceProviders{nullptr};
  if (QNN_SUCCESS !=
      getSystemInterfaceProviders(
          const_cast<const QnnSystemInterface_t***>(&systemInterfaceProviders), &numProviders)) {
    QNN_ERROR("Failed to get system interface providers.");
    return false;
  }
  if (nullptr == systemInterfaceProviders) {
    QNN_ERROR(
        "Failed to get system interface providers: null system interface providers received.");
    return false;
  }
  if (0 == numProviders) {
    QNN_ERROR("Failed to get system interface providers: 0 system interface providers.");
    return false;
  }

  bool foundValidSystemInterface{false};
  for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
    const Qnn_Version_t& systemApiVersion = systemInterfaceProviders[pIdx]->systemApiVersion;
    if (QNN_SYSTEM_API_VERSION_MAJOR == systemApiVersion.major &&
        QNN_SYSTEM_API_VERSION_MINOR <= systemApiVersion.minor) {
      foundValidSystemInterface = true;
      m_qnnSystemInterface      = systemInterfaceProviders[pIdx]->QNN_SYSTEM_INTERFACE_VER_NAME;
      break;
    }
  }
  if (!foundValidSystemInterface) {
    QNN_ERROR("Unable to find a valid system interface.");
    return false;
  }

  if (nullptr == m_qnnSystemInterface.systemContextCreate ||
      nullptr == m_qnnSystemInterface.systemContextGetBinaryInfo ||
      nullptr == m_qnnSystemInterface.systemContextFree) {
    QNN_ERROR("QNN System function pointers are not populated.");
    return false;
  }

  return true;
}

bool ResourceManager::isDLCRecord(const std::string& name) const {
  return name.substr(0, 9) == "record://";
}

size_t ResourceManager::getFileSize(std::string filePath) {
  std::ifstream in(filePath, std::ifstream::binary);
  if (!in) {
    QNN_ERROR("Failed to open input file: %s", filePath.c_str());
    return 0;
  }
  in.seekg(0, in.end);
  const size_t length = static_cast<size_t>(in.tellg());
  in.seekg(0, in.beg);
  return length;
}

size_t ResourceManager::getDLCRecordSize(std::string name) {
  std::lock_guard<std::mutex> lock(m_dlcMutex);
  if (nullptr == m_dlcHandle) {
    throw std::runtime_error("Failed to get DLC record size for record '" + name +
                             "': DLC is not configured.");
  }

  QnnSystemDlc_RecordHandle_t recordHandle{nullptr};
  std::string recordName = "genai.artifact." + name;
  auto qnnError =
      m_qnnSystemInterface.systemDlcGetRecordByName(m_dlcHandle, recordName.c_str(), &recordHandle);
  if (QNN_SUCCESS != QNN_GET_ERROR_CODE(qnnError) || !recordHandle) {
    QNN_ERROR("Failed to retrieve cache %s from dlc", recordName.c_str());
    return 0;
  }

  uint64_t recordBufferSize{0};
  QNN_DEBUG("Cache %s found, trying to get cache size.", recordName.c_str());
  qnnError = m_qnnSystemInterface.systemDlcGetRecordDataSize(recordHandle, &recordBufferSize);
  if (QNN_SUCCESS != QNN_GET_ERROR_CODE(qnnError)) {
    QNN_ERROR("Failed to get record size from record handle.");
    return 0;
  }
  return static_cast<size_t>(recordBufferSize);
}

size_t ResourceManager::getBufferSize(std::string name) {
  if (isDLCRecord(name)) {
    size_t pos = name.find("://");
    if (pos != std::string::npos) {
      std::string recordName = name.substr(pos + 3);
      return getDLCRecordSize(recordName);
    } else {
      QNN_ERROR("Malformed record name.");
      return 0;
    }
  } else {
    return getFileSize(name);
  }
}

bool ResourceManager::readBinaryFromFile(std::string filePath,
                                         std::shared_ptr<uint8_t>& buffer,
                                         size_t bufferSize) {
  if (nullptr == buffer) {
    QNN_ERROR("buffer is nullptr");
    return false;
  }
  std::ifstream in(filePath, std::ifstream::binary);
  if (!in) {
    QNN_ERROR("Failed to open input file: %s", filePath.c_str());
    return false;
  }
  if (!in.read(reinterpret_cast<char*>(buffer.get()), static_cast<std::streamsize>(bufferSize))) {
    QNN_ERROR("Failed to read the contents of: %s", filePath.c_str());
    return false;
  }
  return true;
}

bool ResourceManager::readBinaryFromDLC(const std::string& name, std::shared_ptr<uint8_t>& buffer) {
  std::lock_guard<std::mutex> lock(m_dlcMutex);
  if (nullptr == m_dlcHandle) {
    throw std::runtime_error("Failed to load DLC record '" + name + "': DLC is not configured.");
  }

  QnnSystemDlc_RecordHandle_t recordHandle{nullptr};
  std::string recordName = "genai.artifact." + name;

  auto qnnError =
      m_qnnSystemInterface.systemDlcGetRecordByName(m_dlcHandle, recordName.c_str(), &recordHandle);
  if (QNN_SUCCESS != QNN_GET_ERROR_CODE(qnnError) || !recordHandle) {
    QNN_ERROR("Failed to retrieve cache %s from dlc", recordName.c_str());
    return false;
  }

  QNN_DEBUG("Cache %s found, trying to read cache data.", recordName.c_str());
  const uint8_t* cacheRecordBuf;
  uint64_t recordBufferSize{0};

  qnnError = m_qnnSystemInterface.systemDlcReadRecordDataMemoryMapped(
      recordHandle, &cacheRecordBuf, &recordBufferSize);
  if (QNN_SUCCESS != QNN_GET_ERROR_CODE(qnnError) || recordBufferSize == 0 ||
      cacheRecordBuf == nullptr) {
    QNN_ERROR("Failed to read record data from record handle.");
    return false;
  }
  buffer = std::shared_ptr<uint8_t>(const_cast<uint8_t*>(cacheRecordBuf), [](uint8_t*) {});
  QNN_DEBUG("Cache %s data successfully read to a buffer", recordName.c_str());
  return true;
}

bool ResourceManager::getBuffer(const std::string& name, std::shared_ptr<uint8_t>& buffer) {
  const size_t bufferSize = getBufferSize(name);
  if (bufferSize == 0) {
    QNN_ERROR("getBufferSize returned 0 for: %s", name.c_str());
    return false;
  }
  return getBuffer(name, buffer, bufferSize);
}

bool ResourceManager::getBuffer(const std::string& name,
                                std::shared_ptr<uint8_t>& buffer,
                                size_t bufferSize) {
  bool success = false;
  if (isDLCRecord(name)) {
    size_t pos = name.find("://");
    if (pos != std::string::npos) {
      std::string recordName = name.substr(pos + 3);
      success                = readBinaryFromDLC(name.substr(9), buffer);
    } else {
      QNN_ERROR("Malformed record name.");
      return false;
    }
  } else {
    if (!buffer) {
      buffer = std::shared_ptr<uint8_t>(new uint8_t[bufferSize], std::default_delete<uint8_t[]>());
    }
    success = readBinaryFromFile(name, buffer, bufferSize);
  }
  if (!success) {
    QNN_ERROR("Unsuccessful read operation for: %s", name.c_str());
    return false;
  }
  if (!buffer) {
    QNN_ERROR("Buffer is null after successful read operation for: %s", name.c_str());
    return false;
  }
  return true;
}

void* ResourceManager::dlOpen(const char* path, int flags) const {
  return genie::util::dlOpenWrapper(path, flags);
}

ResourceManager::~ResourceManager() {
  if (m_dlcHandle) {
    m_qnnSystemInterface.systemDlcFree(m_dlcHandle);
  }
  if (m_logHandle) {
    m_qnnSystemInterface.systemLogFree(m_logHandle);
  }
  if (m_systemLibraryHandle) {
    pal::dynamicloading::dlClose(m_systemLibraryHandle);
  }
}

}  // namespace genie
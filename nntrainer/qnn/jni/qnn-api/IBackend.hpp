//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <map>
#include <memory>

#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnDevice.h"
#include "QnnGraph.h"
#include "QnnLog.h"
#include "QnnProfile.h"
#include "QnnWrapperUtils.hpp"
#include "QnnTypes.h"

namespace qnn {
    namespace commandline2 {
        class ICommandLineManager;
    }
namespace tools {
    namespace iotensor {
        class IBufferAlloc;
    }
namespace netrun {

const uint32_t g_profilingLevelNotSet = 0;

enum class PerfProfile {
  LOW_BALANCED,
  BALANCED,
  DEFAULT,
  HIGH_PERFORMANCE,
  SUSTAINED_HIGH_PERFORMANCE,
  BURST,
  EXTREME_POWER_SAVER,
  LOW_POWER_SAVER,
  POWER_SAVER,
  HIGH_POWER_SAVER,
  SYSTEM_SETTINGS,
  NO_USER_INPUT,
  CUSTOM,
  INVALID
};

enum class AppType {
  QNN_APP_NETRUN                   = 0,
  QNN_APP_CONTEXT_BINARY_GENERATOR = 1,
  // Value selected to ensure 32 bits.
  QNN_APP_UNKNOWN = 0x7FFFFFFF
};

// This is the interface that enables backend specific extensions in qnn-net-run.
// It is designed as hooks in the timeline of various events in NetRun.
// Backends that intend to implement custom features through qnn-net-run will have
// to implement this interface and add functionality in appropriate methods depending
// on where/when the custom functionality needs to be exercised.
// These functions/hooks will be called through the IBackend interface from within
// qnn-net-run wherever necessary.
class IBackend {
 public:
  virtual ~IBackend() {}

  virtual bool setupLogging(QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel) = 0;

  virtual bool initialize(void* backendLibHandle) = 0;

  virtual bool setPerfProfile(PerfProfile perfProfile) = 0;

  virtual QnnProfile_Level_t getProfilingLevel() = 0;

  virtual bool loadConfig(std::string configFile) = 0;

  virtual bool loadCommandLineArgs(
      std::shared_ptr<commandline2::ICommandLineManager> clManager) = 0;

  virtual bool beforeBackendInitialize(QnnBackend_Config_t*** customConfigs,
                                       uint32_t* configCount) = 0;

  virtual bool afterBackendInitialize() = 0;

  virtual bool beforeContextCreate(QnnContext_Config_t*** customConfigs, uint32_t* configCount) = 0;

  virtual bool afterContextCreate() = 0;

  virtual bool beforeComposeGraphs(qnn_wrapper_api::GraphConfigInfo_t*** customGraphConfigs,
                                   uint32_t* graphCount) = 0;

  virtual bool afterComposeGraphs() = 0;

  virtual bool beforeGraphFinalizeUpdateConfig(const char* graphName,
                                               Qnn_GraphHandle_t graphHandle,
                                               QnnGraph_Config_t*** customConfigs,
                                               uint32_t* configCount) = 0;

  virtual bool beforeGraphFinalize() = 0;

  virtual bool afterGraphFinalize() = 0;

  virtual bool beforeRegisterOpPackages() = 0;

  virtual bool afterRegisterOpPackages() = 0;

  virtual bool beforeExecute(const char* graphName,
                             QnnGraph_Config_t*** customConfigs,
                             uint32_t* configCount) = 0;

  virtual bool afterExecute() = 0;

  virtual bool beforeContextFree() = 0;

  virtual bool afterContextFree() = 0;

  virtual bool beforeBackendTerminate() = 0;

  virtual bool afterBackendTerminate() = 0;

  virtual bool beforeCreateFromBinary(QnnContext_Config_t*** customConfigs,
                                      uint32_t* configCount) = 0;

  virtual bool afterCreateFromBinary() = 0;

  virtual bool beforeCreateContextsFromBinaryList(
      std::map<std::string, std::tuple<QnnContext_Config_t**, uint32_t>>*
          contextKeyToCustomConfigsMap,
      QnnContext_Config_t*** commonCustomConfigs,
      uint32_t* commonConfigCount) = 0;

  virtual bool afterCreateContextsFromBinaryList() = 0;

  virtual bool beforeCreateDevice(QnnDevice_Config_t*** deviceConfigs, uint32_t* configCount) = 0;

  virtual bool afterCreateDevice() = 0;

  virtual bool beforeFreeDevice() = 0;

  virtual bool afterFreeDevice() = 0;

  virtual bool beforeActivateContext(QnnContext_Config_t*** customConfigs,
                                     uint32_t* configCount) = 0;

  virtual bool afterActivateContext() = 0;

  virtual bool beforeDeactivateContext(QnnContext_Config_t*** customConfigs,
                                       uint32_t* configCount) = 0;

  virtual bool afterDeactivateContext() = 0;

  virtual std::unique_ptr<uint8_t[]> allocateBinaryBuffer(uint32_t bufferSize) = 0;

  virtual void releaseBinaryBuffer(std::unique_ptr<uint8_t[]> buffer) = 0;

  virtual std::unique_ptr<iotensor::IBufferAlloc> getBufferAllocator() = 0;

  virtual bool setParentAppType(AppType appType) = 0;

  virtual bool beforeContextApplyBinarySection() = 0;

  virtual bool afterContextApplyBinarySection() = 0;

  virtual bool isOpMappingsRequired() = 0;

  virtual bool prepareSoc(std::int32_t curDeviceId,
                  std::string dspArch, int vtcmMem, std::string name) = 0;

  virtual void provideOpMappings(Qnn_OpMapping_t* opMappings, uint32_t numOpMappings) = 0;
};

// These are the function types that the backend extensions shared library is
// expected to expose. The first function helps NetRun obtain a valid implementation
// of IBackend interface and the second is used to destroy the same interface at the end.
// The function names themselves are expected to be these strings:
//      1. "createBackendInterface"
//      2. "destroyBackendInterface"
// These functions need to be tagged with extern "C" and their symbols need to be exposed.
typedef IBackend* (*CreateBackendInterfaceFnType_t)();
typedef void (*DestroyBackendInterfaceFnType_t)(IBackend*);

}  // namespace netrun
}  // namespace tools
}  // namespace qnn
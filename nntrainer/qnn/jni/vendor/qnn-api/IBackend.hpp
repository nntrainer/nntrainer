//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnDevice.h"
#include "QnnGraph.h"
#include "QnnInterface.h"
#include "QnnLog.h"
#include "QnnProfile.h"
#include "QnnTypes.h"
#include "QnnWrapperUtils.hpp"

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
  LOW_BALANCED                     = 0,
  BALANCED                         = 1,
  DEFAULT                          = 2,
  HIGH_PERFORMANCE                 = 3,
  SUSTAINED_HIGH_PERFORMANCE       = 4,
  BURST                            = 5,
  GENAI_BURST                      = 6,
  GENAI_LOW_BALANCED               = 7,
  GENAI_BALANCED                   = 8,
  GENAI_DEFAULT                    = 9,
  GENAI_HIGH_PERFORMANCE           = 10,
  GENAI_SUSTAINED_HIGH_PERFORMANCE = 11,
  GENAI_EXTREME_POWER_SAVER        = 12,
  GENAI_LOW_POWER_SAVER            = 13,
  GENAI_POWER_SAVER                = 14,
  GENAI_HIGH_POWER_SAVER           = 15,
  EXTREME_POWER_SAVER              = 16,
  LOW_POWER_SAVER                  = 17,
  POWER_SAVER                      = 18,
  HIGH_POWER_SAVER                 = 19,
  SYSTEM_SETTINGS                  = 20,
  NO_USER_INPUT                    = 21,
  CUSTOM                           = 22,
  INVALID                          = 23
};

enum class AppType {
  QNN_APP_NETRUN                   = 0,
  QNN_APP_CONTEXT_BINARY_GENERATOR = 1,
  // Value selected to ensure 32 bits.
  QNN_APP_UNKNOWN = 0x7FFFFFFF
};

typedef Qnn_ErrorHandle_t (*QnnInterface_GetProvidersFn_t)(const QnnInterface_t*** providerList,
                                                           uint32_t* numProviders);

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

  virtual bool initializeWithProviders(QnnInterface_GetProvidersFn_t getProvidersFn) = 0;

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

  virtual bool beforeContextFree(const std::vector<Qnn_ContextHandle_t>& contextHandle) = 0;

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

  virtual bool beforeCreateDevice(QnnDevice_Config_t*** deviceConfigs,
                                  uint32_t* configCount,
                                  uint32_t socModel) = 0;

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
                          std::string dspArch,
                          int vtcmMem,
                          std::string name,
                          int optimizationLevel             = 0,
                          int dlbcEnable                    = 0,
                          int referenceWeightSharingEnabled = 0,
                          int hvxThreads                    = 0) = 0;

  virtual bool allocateExternalBuffers(void* contextHandle,
                                       const std::string& spillFillBuffer,
                                       const std::string& vtcmBackupBuffer,
                                       const std::string& weightsBuffer) = 0;

  // Finalize all shared buffer pools: allocate shared buffers and memRegister per context.
  // Called after all contexts in the pool have registered via allocateExternalBuffers.
  // Default no-op returns true (success). HTPBackend overrides with pool logic.
  virtual bool finalizeExternalBufferPools() { return true; }

  virtual void provideOpMappings(Qnn_OpMapping_t* opMappings, uint32_t numOpMappings) = 0;

  virtual bool detachableBuffersEnabled() = 0;

  virtual bool detachBuffers(Qnn_ContextHandle_t contextHandle) = 0;

  virtual bool attachBuffers(Qnn_ContextHandle_t contextHandle) = 0;
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

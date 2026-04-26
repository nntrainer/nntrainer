// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    qnn_context_var.h
 * @date    08 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains app context data related functions and classes
 * that manages the global configuration of the current QNN environment.
 */

#ifndef __QNN_CONTEXT_VAR_H__
#define __QNN_CONTEXT_VAR_H__

#include "BackendExtensions.hpp"
#include "IOTensor.hpp"
#include "Log/Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QNN/HTP/QnnHtpContext.h"
#include "QNN/QnnTypes.h"
#include "iotensor_wrapper.hpp"
#include "qnn_rpc_manager.h"

#include <algorithm>
#include <cstdint>
#include <fcntl.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

#include <context.h>
#include <layer.h>
#include <layer_devel.h>

#include <nntrainer_log.h>

using namespace qnn;
using namespace qnn::tools;

namespace nntrainer {

/**
 * @enum    StatusCode
 * @brief   Coarse status codes returned across the QNN integration layer.
 */
enum class StatusCode {
  SUCCESS,
  FAILURE,
  FAILURE_INPUT_LIST_EXHAUSTED,
  FAILURE_SYSTEM_ERROR,
  FAILURE_SYSTEM_COMMUNICATION_ERROR,
  QNN_FEATURE_UNSUPPORTED
};

/**
 * @struct  Qnn_Context_Graph_t
 * @brief   Bundles a QNN context handle with the set of compiled graphs that
 *          live inside it (graph_map / graph_idx keyed by graph name).
 */
struct Qnn_Context_Graph_t {
  Qnn_ContextHandle_t m_context;
  qnn_wrapper_api::GraphInfo_t **m_graphsInfo;
  std::map<std::string, qnn_wrapper_api::GraphInfo_t *>
    graph_map; /** graph name in Context - graph map **/
  std::map<std::string, uint32_t>
    graph_idx; /** graph name in Context - graph map **/

  uint32_t m_graphsCount;

  QnnContext_Config_t **m_contextConfig = nullptr;

  void setGraphInfoMap() {
    for (uint32_t i = 0; i < m_graphsCount; ++i) {
      std::string n((m_graphsInfo)[i]->graphName);
      graph_map.insert(std::make_pair(n, (m_graphsInfo)[i]));
    }
  }

  qnn_wrapper_api::GraphInfo_t *getGraphPtr(std::string graph_name) {
    auto mapIt = graph_map.find(graph_name);
    if (mapIt != graph_map.end()) {
      return mapIt->second;
    } else {
      ml_loge("cannot find graph");
      return nullptr;
    }
  }

  int getGraphIdx(std::string graph_name) {
    auto mapIt = graph_idx.find(graph_name);
    if (mapIt != graph_idx.end()) {
      return mapIt->second;
    } else {
      ml_loge("cannot find graph");
      return -1;
    }
  }
};

/**
 * @struct  QNNVar
 * @brief   Per-process QNN runtime state (backend handle, device handle,
 *          extensions, profiling level, IO data types). Held by
 *          QNNBackendVar and reachable from any tensor whose ContextData
 *          points at a QNN context.
 */
struct QNNVar {
  QnnBackend_Config_t **m_backendConfig = nullptr;
  Qnn_BackendHandle_t m_backendHandle = nullptr;
  BackendExtensions *m_backendExtensions = nullptr;
  Qnn_DeviceHandle_t m_deviceHandle = nullptr;
  iotensor::OutputDataType m_outputDataType;
  iotensor::InputDataType m_inputDataType;
  sample_app::ProfilingLevel m_profilingLevel;
  bool m_isBackendInitialized;
  void *m_backendLibraryHandle = nullptr;
  Qnn_LogHandle_t m_logHandle = nullptr;
  Qnn_ProfileHandle_t m_profileBackendHandle = nullptr;
  sample_app::QnnFunctionPointers m_qnnFunctionPointers;
  std::shared_ptr<QNNRpcManager> RpcMem;
  IOTensorWrapper m_ioTensor;
  std::string name = "qnn_backend_param";
  std::map<std::string, Qnn_Context_Graph_t>
    ct_map; /** bin file name - Context map **/

  std::optional<std::reference_wrapper<Qnn_Context_Graph_t>>
  findContext(std::string bin_path) {
    auto mapIt = ct_map.find(bin_path);
    if (mapIt != ct_map.end()) {
      return mapIt->second;
    }
    return std::nullopt;
  }

  StatusCode makeContext(props::FilePath bin) {

    if (findContext(bin.get())) {
      ml_logw("context is already exists");
      return StatusCode::SUCCESS;
    };

    // Let backendExtensions populate configs
    QnnContext_Config_t **customConfigs{nullptr};
    uint32_t customConfigCount{0};
    if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
      if (!m_backendExtensions->interface()->beforeCreateFromBinary(
            &customConfigs, &customConfigCount)) {
        QNN_ERROR("Extensions Failure in beforeCreateFromBinary()");
        return StatusCode::FAILURE;
      }
    }

    if (nullptr ==
          m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
        nullptr ==
          m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ||
        nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextFree) {

      ml_loge("QNN System function pointers are not populated.");
    }

    uint64_t bufferSize{0};
    bufferSize = bin.file_size();
    std::shared_ptr<uint8_t> buffer{nullptr};

    void *mappedBuffer = nullptr;
    if (true != mmapBinaryFile(bin.get(), &mappedBuffer, bufferSize)) {
      ml_loge("Failed to read binary data");
      return StatusCode::FAILURE;
    }

    buffer = std::shared_ptr<uint8_t>(static_cast<uint8_t *>(mappedBuffer),
                                      [&bufferSize](uint8_t *ptr) {
                                        if (munmap(ptr, bufferSize)) {
                                          ml_loge("Failed to unmap buffer");
                                        }
                                      });

    auto returnStatus = StatusCode::SUCCESS;
    Qnn_Context_Graph_t context_i;

    QnnSystemContext_Handle_t sysCtxHandle{nullptr};
    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate(
          &sysCtxHandle)) {
      ml_loge("Could not create system handle");
      returnStatus = StatusCode::FAILURE;
    }
    const QnnSystemContext_BinaryInfo_t *binaryInfo{nullptr};
    Qnn_ContextBinarySize_t binaryInfoSize{0};
    if (StatusCode::SUCCESS == returnStatus &&
        QNN_SUCCESS !=
          m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
            sysCtxHandle, static_cast<void *>(buffer.get()), bin.file_size(),
            &binaryInfo, &binaryInfoSize)) {
      ml_loge("Fail to get context binary info");
      returnStatus = StatusCode::FAILURE;
    }

    if (StatusCode::SUCCESS == returnStatus &&
        !qnn::tools::sample_app::copyMetadataToGraphsInfo(
          binaryInfo, context_i.m_graphsInfo, context_i.m_graphsCount)) {
      ml_loge("Failed to copy metadata.");
      returnStatus = StatusCode::FAILURE;
    }

    m_qnnFunctionPointers.qnnSystemInterface.systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;

    if (StatusCode::SUCCESS == returnStatus &&
        nullptr == m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary) {
      ml_loge("contextCreateFromBinaryFnHandle is nullptr.");
      returnStatus = StatusCode::FAILURE;
    }

    QnnHtpContext_CustomConfig_t ioMemEstimation;
    ioMemEstimation.option = QnnHtpContext_ConfigOption_t::
      QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
    ioMemEstimation.ioMemEstimation = true;

    unsigned int customConfigCountIOMemEstimate = 1;

    context_i.m_contextConfig = (QnnContext_Config_t **)malloc(
      (customConfigCountIOMemEstimate + customConfigCount + 1) *
      sizeof(QnnContext_Config_t *));
    context_i.m_contextConfig[0] =
      (QnnContext_Config_t *)malloc(sizeof(QnnContext_Config_t));
    context_i.m_contextConfig[0]->option =
      QnnContext_ConfigOption_t::QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    context_i.m_contextConfig[0]->customConfig =
      reinterpret_cast<QnnContext_CustomConfig_t>(&ioMemEstimation);

    for (int i = 0; i < customConfigCount; i++)
      context_i.m_contextConfig[i + 1] = customConfigs[i];
    context_i.m_contextConfig[customConfigCount + 1] = nullptr;

    if (StatusCode::SUCCESS == returnStatus &&
        m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
          m_backendHandle, m_deviceHandle,
          (const QnnContext_Config_t **)context_i.m_contextConfig,
          static_cast<void *>(buffer.get()), bin.file_size(),
          &(context_i.m_context), m_profileBackendHandle)) {
      ml_loge("Could not create context from binary.");
      returnStatus = StatusCode::FAILURE;
    }

    if (nullptr != m_backendExtensions && m_backendExtensions->interface()) {
      if (!m_backendExtensions->interface()->afterCreateFromBinary()) {
        QNN_ERROR("Extensions Failure in afterCreateFromBinary()");
        return StatusCode::FAILURE;
      }
    }

    if (sample_app::ProfilingLevel::OFF != m_profilingLevel) {
      extractBackendProfilingInfo();
    }
    context_i.setGraphInfoMap();

    ct_map.insert(std::make_pair(bin.get(), context_i));
    return StatusCode::SUCCESS;
  }

  qnn_wrapper_api::GraphInfo_t *graphRetrieve(std::string bin_path,
                                              std::string graphName) {

    std::optional<std::reference_wrapper<Qnn_Context_Graph_t>> op =
      findContext(bin_path);

    if (!op) {
      ml_loge("Cannot find context");
      return nullptr;
    }

    Qnn_Context_Graph_t &context_i = *op;

    qnn_wrapper_api::GraphInfo_t *graphInfo = context_i.getGraphPtr(graphName);

    if (nullptr == m_qnnFunctionPointers.qnnInterface.graphRetrieve) {
      ml_loge("graphRetrieveFnHandle is nullptr.");
      return nullptr;
    }

    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnInterface.graphRetrieve(
          context_i.m_context, graphName.c_str(), &(graphInfo->graph))) {
      ml_loge("Unable to retrieve graph handle for graph name : %s",
              graphName.c_str());
      return nullptr;
    }

    return graphInfo;
  }

  StatusCode extractBackendProfilingInfo() {
    Qnn_ProfileHandle_t profileHandle = m_profileBackendHandle;

    if (nullptr == m_profileBackendHandle) {
      ml_loge("Backend Profile handle is nullptr; may not be initialized.");
      return StatusCode::FAILURE;
    }
    const QnnProfile_EventId_t *profileEvents{nullptr};
    uint32_t numEvents{0};
    if (QNN_PROFILE_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.profileGetEvents(
          profileHandle, &profileEvents, &numEvents)) {
      ml_loge("Failure in profile get events.");
      return StatusCode::FAILURE;
    }
    ml_loge("ProfileEvents: [%p], numEvents: [%d]", profileEvents, numEvents);
    for (size_t event = 0; event < numEvents; event++) {
      extractProfilingEvent(*(profileEvents + event));
      extractProfilingSubEvents(*(profileEvents + event));
    }
    return StatusCode::SUCCESS;
  }

  StatusCode extractProfilingSubEvents(QnnProfile_EventId_t profileEventId) {
    const QnnProfile_EventId_t *profileSubEvents{nullptr};
    uint32_t numSubEvents{0};
    if (QNN_PROFILE_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.profileGetSubEvents(
          profileEventId, &profileSubEvents, &numSubEvents)) {
      ml_loge("Failure in profile get sub events.");
      return StatusCode::FAILURE;
    }
    ml_logd("ProfileSubEvents: [%p], numSubEvents: [%d]", profileSubEvents,
            numSubEvents);
    for (size_t subEvent = 0; subEvent < numSubEvents; subEvent++) {
      extractProfilingEvent(*(profileSubEvents + subEvent));
      extractProfilingSubEvents(*(profileSubEvents + subEvent));
    }
    return StatusCode::SUCCESS;
  }

  StatusCode extractProfilingEvent(QnnProfile_EventId_t profileEventId) {
    QnnProfile_EventData_t eventData;
    if (QNN_PROFILE_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.profileGetEventData(profileEventId,
                                                               &eventData)) {
      ml_loge("Failure in profile get event type.");
      return StatusCode::FAILURE;
    }
    ml_logd("Printing Event Info - Event Type: [%d], Event Value: [%lu], Event "
            "Identifier: [%s], Event Unit: [%d]",
            eventData.type, eventData.value, eventData.identifier,
            eventData.unit);
    return StatusCode::SUCCESS;
  }

  bool mmapBinaryFile(std::string filePath, void **buffer, size_t bufferSize) {
    int fd = open(filePath.c_str(), O_RDONLY);
    int OFFSET = 0;

    // read the binary file as memory map
    *buffer = mmap(nullptr, bufferSize, PROT_READ, MAP_PRIVATE, fd, OFFSET);
    close(fd);
    if (madvise(*buffer, bufferSize, MADV_NOHUGEPAGE)) {
      ml_loge("Failed to advise OS on memory usage err: %s", strerror(errno));
    }
    return true;
  }
};

/**
 * @class   QNNBackendVar
 * @brief   ContextData subclass that carries QNN-specific runtime state
 *          (QNNVar) so layers and tensors bound to the QNN Context can
 *          reach the backend/device handles via
 * ContextData::as<QNNBackendVar>().
 */
class QNNBackendVar : public ContextData {
public:
  QNNBackendVar() : data(std::make_shared<QNNVar>()) {}

  const char *getType() const override { return "qnn"; }

  std::shared_ptr<QNNVar> &getVar() { return data; }

private:
  std::shared_ptr<QNNVar> data;
};
} // namespace nntrainer
#endif /* __QNN_CONTEXT_VAR_H__ */
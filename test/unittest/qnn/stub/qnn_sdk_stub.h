// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    qnn_sdk_stub.h
 * @date    30 Sep 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug     No known bugs
 * @brief   Minimal stand-ins for the QNN SDK and nntrainer declarations that
 * nntrainer/qnn/jni/qnn_context_var.h uses, so that the header can be unit
 * tested on a host without the Qualcomm SDK. Only the names and members the
 * header touches are declared; layouts do not match the real SDK.
 */

#ifndef __QNN_SDK_STUB_H__
#define __QNN_SDK_STUB_H__

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <sys/stat.h>

typedef uint64_t Qnn_ErrorHandle_t;
#define QNN_SUCCESS 0
#define QNN_CONTEXT_NO_ERROR 0
#define QNN_PROFILE_NO_ERROR 0

typedef void *Qnn_ContextHandle_t;
typedef void *Qnn_BackendHandle_t;
typedef void *Qnn_DeviceHandle_t;
typedef void *Qnn_LogHandle_t;
typedef void *Qnn_ProfileHandle_t;
typedef void *Qnn_GraphHandle_t;
typedef void *QnnSystemContext_Handle_t;
typedef uint64_t Qnn_ContextBinarySize_t;
typedef uint64_t QnnProfile_EventId_t;
typedef void *QnnContext_CustomConfig_t;

/** @brief backend config (unused by the header beyond its pointer type) */
typedef struct {
  int unused;
} QnnBackend_Config_t;

/** @brief binary info returned by systemContextGetBinaryInfo */
typedef struct {
  int version;
} QnnSystemContext_BinaryInfo_t;

/** @brief tensor descriptor */
typedef struct {
  int unused;
} Qnn_Tensor_t;

/** @brief context config option */
typedef enum { QNN_CONTEXT_CONFIG_OPTION_CUSTOM = 0 } QnnContext_ConfigOption_t;

/** @brief context config entry */
typedef struct {
  QnnContext_ConfigOption_t option;
  QnnContext_CustomConfig_t customConfig;
} QnnContext_Config_t;

/** @brief profile event data */
typedef struct {
  int type;
  unsigned long value;
  const char *identifier;
  int unit;
} QnnProfile_EventData_t;

/** @brief HTP context config option */
typedef enum {
  QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION = 1
} QnnHtpContext_ConfigOption_t;

/** @brief HTP custom context config */
typedef struct {
  QnnHtpContext_ConfigOption_t option;
  bool ioMemEstimation;
} QnnHtpContext_CustomConfig_t;

#define QNN_ERROR(...) (fprintf(stderr, __VA_ARGS__), fputc('\n', stderr))

namespace qnn_wrapper_api {
/** @brief graph info as filled by copyMetadataToGraphsInfo */
typedef struct {
  Qnn_GraphHandle_t graph;
  char *graphName;
  Qnn_Tensor_t *inputTensors;
  uint32_t numInputTensors;
  Qnn_Tensor_t *outputTensors;
  uint32_t numOutputTensors;
} GraphInfo_t;

/**
 * @brief release a tensor array allocated by copyMetadataToGraphsInfo
 * @note defined by the test
 */
bool freeQnnTensors(Qnn_Tensor_t *&tensors, uint32_t numTensors);
} // namespace qnn_wrapper_api

/** @brief backend extension interface */
class QnnBackendExtensionStub {
public:
  /** @brief destructor */
  virtual ~QnnBackendExtensionStub() = default;
  /** @brief hook called before contextCreateFromBinary */
  virtual bool beforeCreateFromBinary(QnnContext_Config_t ***customConfigs,
                                      uint32_t *customConfigCount) = 0;
  /** @brief hook called after contextCreateFromBinary */
  virtual bool afterCreateFromBinary() = 0;
};

/** @brief backend extension holder */
class BackendExtensions {
public:
  /** @brief return the extension interface, or nullptr */
  QnnBackendExtensionStub *interface() { return iface; }
  QnnBackendExtensionStub *iface = nullptr; /**< interface under test */
};

namespace qnn {
namespace tools {
namespace iotensor {
/** @brief output data type */
enum class OutputDataType { FLOAT_ONLY };
/** @brief input data type */
enum class InputDataType { FLOAT };
} // namespace iotensor

namespace sample_app {
/** @brief profiling level */
enum class ProfilingLevel { OFF, BASIC };

/** @brief QNN backend function table (the members the header calls) */
struct QnnInterfaceStub {
  Qnn_ErrorHandle_t (*contextCreateFromBinary)(
    Qnn_BackendHandle_t, Qnn_DeviceHandle_t, const QnnContext_Config_t **,
    const void *, Qnn_ContextBinarySize_t, Qnn_ContextHandle_t *,
    Qnn_ProfileHandle_t);
  Qnn_ErrorHandle_t (*contextFree)(Qnn_ContextHandle_t, Qnn_ProfileHandle_t);
  Qnn_ErrorHandle_t (*graphRetrieve)(Qnn_ContextHandle_t, const char *,
                                     Qnn_GraphHandle_t *);
  Qnn_ErrorHandle_t (*profileGetEvents)(Qnn_ProfileHandle_t,
                                        const QnnProfile_EventId_t **,
                                        uint32_t *);
  Qnn_ErrorHandle_t (*profileGetSubEvents)(QnnProfile_EventId_t,
                                           const QnnProfile_EventId_t **,
                                           uint32_t *);
  Qnn_ErrorHandle_t (*profileGetEventData)(QnnProfile_EventId_t,
                                           QnnProfile_EventData_t *);
};

/** @brief QNN system function table (the members the header calls) */
struct QnnSystemInterfaceStub {
  Qnn_ErrorHandle_t (*systemContextCreate)(QnnSystemContext_Handle_t *);
  Qnn_ErrorHandle_t (*systemContextGetBinaryInfo)(
    QnnSystemContext_Handle_t, void *, uint64_t,
    const QnnSystemContext_BinaryInfo_t **, Qnn_ContextBinarySize_t *);
  Qnn_ErrorHandle_t (*systemContextFree)(QnnSystemContext_Handle_t);
};

/** @brief function tables loaded from the QNN libraries */
struct QnnFunctionPointers {
  QnnInterfaceStub qnnInterface{};             /**< backend functions */
  QnnSystemInterfaceStub qnnSystemInterface{}; /**< system functions */
};

/**
 * @brief copy graph metadata out of a context binary
 * @note defined by the test
 */
bool copyMetadataToGraphsInfo(const QnnSystemContext_BinaryInfo_t *binaryInfo,
                              qnn_wrapper_api::GraphInfo_t **&graphsInfo,
                              uint32_t &graphsCount);
} // namespace sample_app
} // namespace tools
} // namespace qnn

/** @brief placeholder for the real IO tensor helper */
class IOTensorWrapper {};

namespace nntrainer {
/** @brief placeholder for the real rpcmem manager */
class QNNRpcManager {};

/** @brief placeholder for the real context data base class */
class ContextData {
public:
  /** @brief destructor */
  virtual ~ContextData() = default;
};

namespace props {
/** @brief path property with the accessors makeContext() uses */
class FilePath {
public:
  /** @brief construct from a path */
  FilePath(const std::string &p) : path(p) {}
  /** @brief return the path */
  const std::string &get() const { return path; }
  /** @brief return the file size, or 0 if it cannot be read */
  std::uint64_t file_size() const {
    struct stat st;
    return stat(path.c_str(), &st) == 0 ? st.st_size : 0;
  }

private:
  std::string path;
};
} // namespace props
} // namespace nntrainer

#define ml_loge(...) (fprintf(stderr, __VA_ARGS__), fputc('\n', stderr))
#define ml_logw(...) (fprintf(stderr, __VA_ARGS__), fputc('\n', stderr))
#define ml_logi(...) ((void)0)
#define ml_logd(...) ((void)0)

#endif /* __QNN_SDK_STUB_H__ */

//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

/**
 *  @file
 *  @brief  QNN System Context API.
 *
 *          This is a system API header to provide
 *          Deep Learning Container (DLC) services to users.
 */

#ifndef QNN_SYSTEM_DLC_H
#define QNN_SYSTEM_DLC_H

#include "QnnInterface.h"
#include "QnnTypes.h"
#include "System/QnnSystemCommon.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemLog.h"

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Error Codes
//=============================================================================

/**
 * @brief QNN System Context API result / error codes.
 */
typedef enum {
  QNN_SYSTEM_DLC_MINERROR = QNN_MIN_ERROR_SYSTEM,
  //////////////////////////////////////////

  /// Qnn System Context success
  QNN_SYSTEM_DLC_NO_ERROR = QNN_SYSTEM_COMMON_NO_ERROR,
  /// There is optional API component that is not supported yet.
  QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE = QNN_SYSTEM_COMMON_ERROR_UNSUPPORTED_FEATURE,
  /// QNN System DLC invalid handle
  QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE = QNN_SYSTEM_COMMON_ERROR_INVALID_HANDLE,
  /// One or more arguments to a System DLC API is/are NULL/invalid.
  QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT = QNN_SYSTEM_COMMON_ERROR_INVALID_ARGUMENT,
  /// Generic Failure in achieving the objective of a System DLC API
  QNN_SYSTEM_DLC_ERROR_OPERATION_FAILED = QNN_SYSTEM_DLC_MINERROR + 2,

  /// Malformed DLC Binary
  QNN_SYSTEM_DLC_ERROR_MALFORMED_BINARY = QNN_SYSTEM_DLC_MINERROR + 10,
  //////////////////////////////////////////
  QNN_SYSTEM_DLC_MAXERROR = QNN_MAX_ERROR_SYSTEM
} QnnSystemDlc_Error_t;

//=============================================================================
// Data Types
//=============================================================================

/// Version of the graph config info
typedef enum {
  QNN_SYSTEM_DLC_GRAPH_CONFIG_INFO_VERSION_1 = 0x01,
  // Unused, present to ensure 32 bits.
  QNN_SYSTEM_DLC_GRAPH_CONFIG_INFO_UNDEFINED = 0x7FFFFFFF
} QnnSystemContext_GraphConfigInfoVersion_t;

typedef struct {
  const char* graphName;
  const QnnGraph_Config_t** graphConfigs;
  uint32_t numConfigs;
} QnnSystemDlc_GraphConfigInfoV1_t;

/// @brief structure to define
typedef struct {
  QnnSystemContext_GraphConfigInfoVersion_t version;
  union UNNAMED {
    QnnSystemDlc_GraphConfigInfoV1_t v1;
  };
} QnnSystemDlc_GraphConfigInfo_t;

//=============================================================================
// Record Types
//=============================================================================
typedef enum {
  // DLC.metadata that stores DLC meta info
  QNN_SYSTEM_DLC_RECORD_TYPE_DLC_METADATA = 0x01,
  // The irGraph topology
  QNN_SYSTEM_DLC_RECORD_TYPE_MODEL_TOPOLOGY = 0x02,
  // The meta info of all the weights of the models
  QNN_SYSTEM_DLC_RECORD_TYPE_MODEL_PARAMS = 0x03,
  // The raw weights of the models
  QNN_SYSTEM_DLC_RECORD_TYPE_MODEL_RAW_WIEGHTS = 0x04,
  // The encodings of all the weights of the models
  QNN_SYSTEM_DLC_RECORD_TYPE_MODEL_ENCODINGS = 0x05,
  // The metatdata of transforms occurring on lora static tensors
  QNN_SYSTEM_DLC_RECORD_TYPE_LORA_CONVERTER_METADATA = 0x06,
  QNN_SYSTEM_DLC_RECORD_PREFIX_HTP_CACHE_RECORD      = 0x07,
  QNN_SYSTEM_DLC_RECORD_PREFIX_AIP_CACHE_RECORD      = 0x08,
  QNN_SYSTEM_DLC_RECORD_PREFIX_HTA_CACHE_RECORD      = 0x09,
  QNN_SYSTEM_DLC_RECORD_PREFIX_AIX_CACHE_RECORD      = 0x0A,
  QNN_SYSTEM_DLC_RECORD_PREFIX_GPU_CACHE_RECORD      = 0x0B,
  QNN_SYSTEM_DLC_RECORD_PREFIX_HEXNNV2_CACHE_RECORD  = 0x0C,
  QNN_SYSTEM_DLC_RECORD_TYPE_SOURCE_MAPPING          = 0x0D,
  QNN_SYSTEM_DLC_RECORD_TYPE_SOURCE_TOPOLOGY         = 0x0E,
  // Record to store the schematic of graph composed on HTP backend
  QNN_SYSTEM_DLC_RECORD_TYPE_HTP_SCHEMATIC_BIN = 0x0F,
  // Record to store the mapping information of ops in irGraph and HTP backend
  QNN_SYSTEM_DLC_RECORD_TYPE_HTP_GRAPH_MAPPING = 0x10,
  QNN_SYSTEM_DLC_RECORD_TYPE_GENAI_METADATA    = 0x11,
  QNN_SYSTEM_DLC_RECORD_TYPE_GENAI_ARTIFACT    = 0x12,
  // Record to store the HTP shared weights
  QNN_SYSTEM_DLC_RECORD_TYPE_HTP_SHARED_WEIGHTS_RECORD = 0x13,
  // Record to store the HTP shared weights metadata
  QNN_SYSTEM_DLC_RECORD_TYPE_HTP_SHARED_WEIGHTS_METADATA_RECORD = 0x14,
  // Unused, present to ensure 32 bits.
  QNN_SYSTEM_DLC_RECORD_NAME_UNKNOWN = 0x7FFFFFFF
} QnnSystemDlc_RecordType_t;

/**
 * @brief A typedef to indicate a QNN System DLC Record handle
 */
typedef void* QnnSystemDlc_RecordHandle_t;

//=============================================================================
// Public Functions
//=============================================================================

/**
 * @brief A function to create an instance of the DLC from a file
 *
 * @param[in] dlcPath path the DLC
 * @param[in] logger a log handle produced from QnnSystemLog_create(). Can be NULL
 * @param[out] dlcHandle A handle to the created instance of a systemDlc entity
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully created a systemDlc entity
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: sysCtxHandle is NULL
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Error encountered in allocating memory for
 *           systemDlc instance
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: system context features not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_createFromFile(Qnn_LogHandle_t logger,
                                              const char* dlcPath,
                                              QnnSystemDlc_Handle_t* dlcHandle);

/**
 * @brief A function to create an instance of the DLC from a file with a destination directory hint
 *
 * @param[in] logger a log handle produced from QnnSystemLog_create(). Can be NULL
 * @param[in] dlcPath path to the DLC
 * @param[in] destinationDirectoryHint path to a directory where the DLC will be saved after
 *            changes. Must not be NULL.
 * @param[out] dlcHandle A handle to the created instance of a systemDlc entity
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully created a systemDlc entity
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: dlcPath, destinationDirectoryHint, or
 *           dlcHandle is NULL
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Error encountered in allocating memory for
 *           systemDlc instance
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: system context features not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_createFromFileWithDestinationDir(
    Qnn_LogHandle_t logger,
    const char* dlcPath,
    const char* destinationDirectoryHint,
    QnnSystemDlc_Handle_t* dlcHandle);

/**
 * @brief A function to create an instance of the DLC from a binary buffer
 *
 * @param[in]  buffer pointer to buffer representing the DLC
 * @param[in]  logger a log handle produced from QnnSystemLog_create(). Can be NULL
 * @param[in]  bufferSize size of the binary buffer
 * @param[out] dlcHandle A handle to the created instance of a systemDlc entity
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully created a systemDlc entity
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: sysCtxHandle is NULL
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Error encountered in allocating memory for
 *           systemDlc instance
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: system context features not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_createFromBinary(Qnn_LogHandle_t logger,
                                                const uint8_t* buffer,
                                                const Qnn_ContextBinarySize_t bufferSize,
                                                QnnSystemDlc_Handle_t* dlcHandle);

/**
 * @brief A function to create an instance of the DLC from a binary buffer with a destination
 *        directory hint
 *
 * @param[in]  logger a log handle produced from QnnSystemLog_create(). Can be NULL
 * @param[in]  buffer pointer to buffer representing the DLC
 * @param[in]  bufferSize size of the binary buffer
 * @param[in]  destinationDirectoryHint path to a directory where the DLC will be saved after
 *             changes. Must not be NULL.
 * @param[out] dlcHandle A handle to the created instance of a systemDlc entity
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully created a systemDlc entity
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: buffer, destinationDirectoryHint, or
 *           dlcHandle is NULL
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Error encountered in allocating memory for
 *           systemDlc instance
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: system context features not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_createFromBinaryWithDestinationDir(
    Qnn_LogHandle_t logger,
    const uint8_t* buffer,
    const Qnn_ContextBinarySize_t bufferSize,
    const char* destinationDirectoryHint,
    QnnSystemDlc_Handle_t* dlcHandle);

/**
 * @brief A function to compose graphs from a DLC on a particular backend, __backend__, through
 *        an backendInterface __backendInterface__. Memory allocated in __graphs__ is owned by clients and may
 *        be released with calls to free().
 *
 * @param[in]  dlcHandle the DLC to retrieve graphs from
 * @param[in]  graphConfigs the graph configuration information for a particular graph
 * @param[in]  numGraphConfigs number of graph configurations
 * @param[in] backend the backend on which to compose the graphs
 * @param[in]  context the context on which to compose the graphs
 * @param[in]  backendInterface the interface used to compose the graph.
 * @param[in] graphVersion version of the graph info structure to be returned
 * @param[out] graphs An array of graph information representing what was created with the backend
 * @param[out] numGraphs the number of created graphs
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully composed graphs
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: Argument is NULL
 *         - QNN_COMMON_ERROR_MEM_ALLOC: Error encountered in allocating memory for
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid Dlc handle to free
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: DLC features not supported
 *
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_composeGraphs(QnnSystemDlc_Handle_t dlcHandle,
                                             const QnnSystemDlc_GraphConfigInfo_t** graphConfigs,
                                             const uint32_t numGraphConfigs,
                                             Qnn_BackendHandle_t backend,
                                             Qnn_ContextHandle_t context,
                                             QnnInterface_t backendInterface,
                                             QnnSystemContext_GraphInfoVersion_t graphVersion,
                                             QnnSystemContext_GraphInfo_t** graphs,
                                             uint32_t* numGraphs);
/**
 * @brief A function to retrieve Op Mapping information from a DLC
 *
 * @param[in]  dlcHandle Handle to the DLC
 * @param[out] opMappings a list of op mappings. The memory allocated here is owned by the System
 *             library and is released when the corresponding DLC Handle is freed
 * @param[out] numOpMappings the number of opMappings
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully freed instance of System Context
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid Dlc handle to free
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getOpMappings(QnnSystemDlc_Handle_t dlcHandle,
                                             const Qnn_OpMapping_t** opMappings,
                                             uint32_t* numOpMappings);

/**
 * @brief A function to retrieve a record associated with handle __dlcHandle__
 *        of name __recordName__
 *
 * @param[in] dlcHandle handle to the DLC to which this record is associated
 * @param[in] recordName the name of the record to retrieve
 * @param[out] recordHandle record handle matching __recordName__
 *
 * @note If there are no records that match __recordName__, then recordHandle will
 *       be set to nullptr. Record names are unique within a DLC so this API will only
 *       return at most one record handle
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved record of __recordName__
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid dlcHandle passed
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: Invalid recordName or recordHandle passed
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getRecordByName(QnnSystemDlc_Handle_t dlcHandle,
                                               const char* recordName,
                                               QnnSystemDlc_RecordHandle_t* recordHandle);

/**
 * @brief A function to retrieve records associated with handle __dlcHandle__ of type
 *        __recordType__
 *
 * @param[in] dlcHandle handle to the DLC to which the records are associated
 * @param[in] recordType the type of the records to retrieve
 * @param[in] getMostOptimalContextBinary option to retrieve the most-compatible context binary on
 *                                        current SoC. This option is only useful when retrieving
 *                                        context binaries. If set to 1, no more than one context
 *                                        binary will be returned even if the DLC has multiple
 *                                        context binaries that match the provided __recordType__.
 *                                        If set to 0, all context binaries that match __recordType__
 *                                        will be returned
 * @param[out] recordHandles array of record handles matching __recordType__
 * @param[out] numRecordHandles number of record handles in recordHandles array
 *
 * @note If there are no records of __recordType__, then recordHandles will be set
 *       to nullptr and numRecordHandles will be set to 0
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieving records of type __recordType__
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid dlcHandle passed
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: invalid recordHandles or numRecordHandles passed
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getRecordsByType(QnnSystemDlc_Handle_t dlcHandle,
                                                QnnSystemDlc_RecordType_t recordType,
                                                uint8_t getMostOptimalContextBinary,
                                                QnnSystemDlc_RecordHandle_t** recordHandles,
                                                uint32_t* numRecordHandles);

/**
 * @brief A function to retrieve the size of the data in __recordHandle__
 *
 * @param[in] recordHandle record handle to retrieve the size of the data from
 * @param[out] dataSize size of the data in the record
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved the size of the record
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid record handle
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: invalid dataSize pointer passed
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getRecordDataSize(QnnSystemDlc_RecordHandle_t recordHandle,
                                                 uint64_t* dataSize);

/**
 * @brief A function to retrieve the content of __recordHandle__ as a memory mapped buffer __data__
 *
 * @param[in] recordHandle record handle to read from
 * @param[out] data data read from the record
 * @param[out] dataSize size of the data read
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully read the data of the record to user buffer
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid record handle
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: invalid user data pointer or dataSize
 *                                                  pointer passed
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_readRecordDataMemoryMapped(QnnSystemDlc_RecordHandle_t recordHandle,
                                                          const uint8_t** data,
                                                          uint64_t* dataSize);

/**
 * @brief A function to free a record. Record will also be freed if the associated
 *        DLC handle is freed
 *
 * @param[in] recordHandle handle to the record to be freed
 *
 * @note If the record is associated with a DLC, this API does not remove the record
 *       from the DLC. It will only free the record handle. To remove the record from
 *       the DLC, use the QnnSystemDlc_removeRecordByType() API
 *
 * @note This API will fail if the record is associated with a DLC and its handle has
 *       already been freed. Since freeing the DLC handle also frees all handles of
 *       the records associated with it, this operation will result in failure
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully freed instance of record handle
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid record handle to free
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_freeRecord(QnnSystemDlc_RecordHandle_t recordHandle);

/**
 * @brief A function to free the instance of the System Context object
 *        This API clears any intermediate memory allocated and associated
 *        with a valid handle
 *
 * @param[in] sysCtxHandle Handle to the System Context object
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully freed instance of System Context
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: Invalid System Context handle to free
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_free(QnnSystemDlc_Handle_t dlcHandle);

/**
 * @brief A function to save the DLC __dlcHandle__ to the location path.
 *
 * @param[in] dlcHandle the handle to which the record is associated
 * @param[in] path path to save the DLC handle
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_save(QnnSystemDlc_Handle_t dlcHandle, const char* path);

/**
 * @brief A function to get the size of memory needed to hold the DLC in binary form.
 *        Must be called before QnnSystemDlc_getBinary() to determine the required buffer size.
 *
 * @param[in]  dlcHandle        A handle to the DLC instance.
 * @param[out] binaryBufferSize The number of bytes the client must allocate to hold the
 *                              serialized DLC binary.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully retrieved the binary size
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: dlcHandle is not a valid handle
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: binaryBufferSize is NULL
 *         - QNN_SYSTEM_DLC_ERROR_OPERATION_FAILED: failed to compute binary size
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: feature not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getBinarySize(QnnSystemDlc_Handle_t dlcHandle,
                                             Qnn_SystemDlcBinarySize_t* binaryBufferSize);

/**
 * @brief A function to serialize the DLC into a client-provided buffer.
 *        The client must first call QnnSystemDlc_getBinarySize() to determine the required
 *        buffer size and allocate sufficient memory before calling this function.
 *
 * @param[in]  dlcHandle         A handle to the DLC instance.
 * @param[in]  binaryBuffer      Pointer to the client-allocated buffer to write the DLC into.
 * @param[in]  binaryBufferSize  Size of binaryBuffer in bytes.
 * @param[out] writtenBufferSize Number of bytes actually written into binaryBuffer.
 *
 * @return Error code
 *         - QNN_SUCCESS: Successfully serialized the DLC into binaryBuffer
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_HANDLE: dlcHandle is not a valid handle
 *         - QNN_SYSTEM_DLC_ERROR_INVALID_ARGUMENT: binaryBuffer or writtenBufferSize is NULL
 *         - QNN_SYSTEM_DLC_ERROR_OPERATION_FAILED: failed to serialize the DLC
 *         - QNN_SYSTEM_DLC_ERROR_UNSUPPORTED_FEATURE: feature not supported
 */
QNN_SYSTEM_API
Qnn_ErrorHandle_t QnnSystemDlc_getBinary(QnnSystemDlc_Handle_t dlcHandle,
                                         uint8_t* binaryBuffer,
                                         Qnn_SystemDlcBinarySize_t binaryBufferSize,
                                         Qnn_SystemDlcBinarySize_t* writtenBufferSize);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // QNN_SYSTEM_DLC_H

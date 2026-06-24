// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    qnn_context.h
 * @date    10 Dec 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains qnn context related functions and classes that
 * manages the global configuration of the current QNN environment.
 */

#include "Log/Logger.hpp"
#include "QNN/HTP/QnnHtpGraph.h"
#include "QnnTypes.h"
#include "Utils/BuildId.hpp"
#include "Utils/DynamicLoadUtil.hpp"
#include "Utils/QnnSampleAppUtils.hpp"
#include "WrapperUtils/QnnWrapperUtils.hpp"
#include "iotensor_wrapper.hpp"

#include "BackendExtensions.hpp"

#include "qnn_context.h"
#include <QNNGraph.h>
#include <cstdlib>
#include <iostream>
#include <limits.h>
#include <unistd.h>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "QNNContext"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(fmt, ...) fprintf(stdout, "[DEBUG] " fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#endif

using namespace qnn;
using namespace qnn::tools;
using namespace qnn::tools::sample_app;

namespace nntrainer {

static std::string g_default_backend_ext_config_path;

static std::string trim_trailing_slashes(std::string path) {
  while (path.size() > 1 && path.back() == '/') {
    path.pop_back();
  }
  return path;
}

static bool is_absolute_path(const std::string &path) {
  return !path.empty() && path[0] == '/';
}

static std::string resolve_quick_dot_ai_base_dir() {
  const char *override_base_dir = std::getenv("QUICK_DOT_AI_BASE_DIR");
  if (override_base_dir != nullptr && override_base_dir[0] != '\0') {
    std::string resolved = trim_trailing_slashes(override_base_dir);
    LOGD("resolve_quick_dot_ai_base_dir: using QUICK_DOT_AI_BASE_DIR=%s",
         resolved.c_str());
    return resolved;
  }

  char cwd[PATH_MAX] = {
    0,
  };
  if (getcwd(cwd, sizeof(cwd)) != nullptr && cwd[0] != '\0') {
    std::string resolved = trim_trailing_slashes(cwd);
    LOGD("resolve_quick_dot_ai_base_dir: using cwd=%s", resolved.c_str());
    return resolved;
  }

  std::string fallback = "/sdcard/Download/aistudio-mobile/";
  LOGD("resolve_quick_dot_ai_base_dir: using fallback=%s", fallback.c_str());
  return fallback;
}

static std::string
resolve_backend_extensions_config_value(const std::string &path) {
  if (path.empty() || is_absolute_path(path)) {
    return path;
  }
  return resolve_quick_dot_ai_base_dir() + "/" + path;
}

static std::string resolve_backend_extensions_config_path() {
  const char *override_config_path =
    std::getenv("QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH");
  if (override_config_path != nullptr && override_config_path[0] != '\0') {
    std::string config_path =
      resolve_backend_extensions_config_value(override_config_path);
    LOGD("resolve_backend_extensions_config_path: using "
         "QUICK_DOT_AI_QNN_BACKEND_EXT_CONFIG_PATH=%s",
         config_path.c_str());
    return config_path;
  }

  std::string config_path =
    resolve_quick_dot_ai_base_dir() + "/htp_backend_ext_config.json";
  LOGD("resolve_backend_extensions_config_path: %s", config_path.c_str());
  return config_path;
}

void QNNContext::setDefaultBackendExtConfigPath(const std::string &path) {
  g_default_backend_ext_config_path = path;
  LOGD("setDefaultBackendExtConfigPath: %s", path.c_str());
}

std::mutex qnn_factory_mutex;

void QNNContext::initialize() noexcept {
  LOGD("initialize: START");
  try {
    LOGD("initialize: calling init()");
    init();
    LOGD("initialize: init() completed");
    ml_logi("qnn init done");
    LOGD("initialize: creating QNNRpcManager");
    setMemAllocator(std::make_shared<QNNRpcManager>());

    std::static_pointer_cast<QNNBackendVar>(getContextData())
      ->getVar()
      ->RpcMem = std::static_pointer_cast<QNNRpcManager>(getMemAllocator());
    LOGD("initialize: QNNRpcManager set");

    LOGD("initialize: registering QNN layers");
    registerFactory(nntrainer::createLayer<QNNGraph>, QNNGraph::type, -1);
    ml_logi("qnn registerFactory done");
    LOGD("initialize: registerFactory done");
  } catch (std::exception &e) {
    LOGE("initialize: registering qnn layers failed!!, reason: %s", e.what());
    ml_loge("registering qnn layers failed!!, reason: %s", e.what());
  } catch (...) {
    LOGE("initialize: registering qnn layer failed due to unknown reason");
    ml_loge("registering qnn layer failed due to unknown reason");
  }
  LOGD("initialize: END");
}

int QNNContext::init() {
  LOGD("init: START");
  std::cout << "qnncontext::init called" << std::endl;
  if (!log::initializeLogging()) {
    LOGE("init: Unable to initialize logging!");
    ml_loge("ERROR: Unable to initialize logging!");
    return -1;
  }
  LOGD("init: logging initialized");
  log::setLogLevel(QnnLog_Level_t::QNN_LOG_LEVEL_ERROR);

  std::string backEndPath = "libQnnHtp.so";
  std::string systemLibraryPath = "libQnnSystem.so";
  LOGD("init: backEndPath=%s", backEndPath.c_str());
  LOGD("init: systemLibraryPath=%s", systemLibraryPath.c_str());

  std::string opPackagePaths = "";

  auto qnn_data = getQnnData();
  LOGD("init: qnn_data obtained");

  qnn_data->m_outputDataType = iotensor::OutputDataType::FLOAT_AND_NATIVE;
  qnn_data->m_inputDataType = iotensor::InputDataType::NATIVE;
  qnn_data->m_profilingLevel = ProfilingLevel::OFF;

  qnn_data->m_isBackendInitialized = false;
  m_isContextCreated = false;

  qnn::tools::sample_app::split(m_opPackagePaths, opPackagePaths, ',');

  if (backEndPath.empty()) {
    LOGE("init: Cannot find backend Path : libQnnHtp.so");
    ml_loge("ERROR: Cannot fine backend Path : libQnnHtp.so");
    return -1;
  }
  LOGD("init: calling dynamicloadutil::getQnnFunctionPointers");

  auto statusCode = dynamicloadutil::getQnnFunctionPointers(
    backEndPath, "", &qnn_data->m_qnnFunctionPointers,
    &qnn_data->m_backendLibraryHandle, false, nullptr);
  LOGD("init: getQnnFunctionPointers returned status=%d", (int)statusCode);
  if (dynamicloadutil::StatusCode::SUCCESS != statusCode) {
    if (dynamicloadutil::StatusCode::FAIL_LOAD_BACKEND == statusCode) {
      LOGE("init: could not load backend");
      ml_loge(
        "Error: initializing QNN Function Pointers: could not load backend");
    } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == statusCode) {
      LOGE("init: could not load model");
      ml_loge("Error initializing QNN Function Pointers: could not load");
    } else {
      LOGE("init: unknown error initializing QNN Function Pointers");
      ml_loge("Error initializing QNN Function Pointers");
    }
  }

  LOGD("init: calling getQnnSystemFunctionPointers");
  statusCode = qnn::tools::dynamicloadutil::getQnnSystemFunctionPointers(
    systemLibraryPath, &qnn_data->m_qnnFunctionPointers);
  LOGD("init: getQnnSystemFunctionPointers returned status=%d",
       (int)statusCode);
  if (qnn::tools::dynamicloadutil::StatusCode::SUCCESS != statusCode) {
    LOGE("init: Error initializing QNN System Function Pointers");
    ml_loge("Error initializing QNN System Function Pointers", EXIT_FAILURE);
  }

  if (log::isLogInitialized()) {
    auto logCallback = log::getLogCallback();
    auto logLevel = log::getLogLevel();

    if (QNN_SUCCESS != qnn_data->m_qnnFunctionPointers.qnnInterface.logCreate(
                         logCallback, logLevel, &qnn_data->m_logHandle)) {
      LOGE("init: Unable to initialize logging in the backend");
      ml_logw("Unable to initialize logging in the backend.");
    } else {
      LOGD("init: Logging initialized in the backend");
      ml_logw("Logging not available in the backend.");
    }
  }

  LOGD("init: Creating backend extensions");
  BackendExtensionsConfigs backend_extensions_config;
  std::string config_path;
  if (!m_backendExtConfigPath.empty()) {
    config_path = m_backendExtConfigPath;
  } else if (!g_default_backend_ext_config_path.empty()) {
    config_path = g_default_backend_ext_config_path;
  } else {
    config_path = resolve_backend_extensions_config_path();
  }
  config_path = resolve_backend_extensions_config_value(config_path);
  LOGD("init: backend_extensions_config.configFilePath = %s",
       config_path.c_str());
  backend_extensions_config.configFilePath = config_path;
  backend_extensions_config.sharedLibraryPath = "libQnnHtpNetRunExtensions.so";

  BackendExtensions *backend_extensions = new BackendExtensions(
    backend_extensions_config, qnn_data->m_backendLibraryHandle, false, nullptr,
    QNN_LOG_LEVEL_ERROR);
  qnn_data->m_backendExtensions = backend_extensions;
  LOGD("init: Backend extensions created");

  QnnBackend_Config_t **customConfigs{nullptr};
  uint32_t customConfigCount{0};
  if (backend_extensions->interface()) {
    if (!backend_extensions->interface()->beforeBackendInitialize(
          &customConfigs, &customConfigCount)) {
      LOGE("init: Extensions Failure in beforeBackendInitialize()");
      QNN_ERROR("Extensions Failure in beforeBackendInitialize()");
      return false;
    }
    LOGD("init: beforeBackendInitialize done, customConfigCount=%u",
         customConfigCount);
  }
  if ((customConfigCount) > 0) {
    qnn_data->m_backendConfig = (QnnBackend_Config_t **)calloc(
      (customConfigCount + 1), sizeof(QnnBackend_Config_t *));
    if (nullptr == qnn_data->m_backendConfig) {
      LOGE("init: Could not allocate memory for allBackendConfigs");
      QNN_ERROR("Could not allocate memory for allBackendConfigs");
      return false;
    }
    for (size_t cnt = 0; cnt < customConfigCount; cnt++) {
      qnn_data->m_backendConfig[cnt] = customConfigs[cnt];
    }
  }

  LOGD("init: Calling backendCreate");
  auto qnnStatus = qnn_data->m_qnnFunctionPointers.qnnInterface.backendCreate(
    qnn_data->m_logHandle,
    (const QnnBackend_Config_t **)qnn_data->m_backendConfig,
    &qnn_data->m_backendHandle);
  LOGD("init: backendCreate returned status=%lu", qnnStatus);
  if (QNN_BACKEND_NO_ERROR != qnnStatus) {
    LOGE("init: Could not initialize backend, error=%d",
         (unsigned int)qnnStatus);
    ml_loge("Could not initialize backend due to error = %d",
            (unsigned int)qnnStatus);
    if (qnn_data->m_backendConfig) {
      free(qnn_data->m_backendConfig);
    }
    return -1;
  }
  ml_logi("Initialize BAckend Returned Status = %lu", qnnStatus);
  qnn_data->m_isBackendInitialized = true;
  LOGD("init: Backend initialized successfully");

  if (qnn_data->m_backendConfig) {
    free(qnn_data->m_backendConfig);
  }

  if (backend_extensions->interface()) {
    if (!backend_extensions->interface()->afterBackendInitialize()) {
      LOGE("init: Extensions Failure in afterBackendInitialize()");
      QNN_ERROR("Extensions Failure in afterBackendInitialize()");
      return false;
    }
    LOGD("init: afterBackendInitialize done");
  }

  LOGD("init: Creating device");
  auto devicePropertySupportStatus = this->isDevicePropertySupported();
  LOGD("init: isDevicePropertySupported returned %d",
       (int)devicePropertySupportStatus);
  if (StatusCode::FAILURE != devicePropertySupportStatus) {
    auto createDeviceStatus = this->createDevice();
    LOGD("init: createDevice returned %d", (int)createDeviceStatus);
    if (StatusCode::SUCCESS != createDeviceStatus) {
      LOGE("init: Device Creation failure");
      ml_loge("Device Creation failure");
      return -1;
    }
  }

  LOGD("init: Initializing profiling");
  if (StatusCode::SUCCESS != this->initializeProfiling()) {
    LOGE("init: Profiling Initialization failure");
    ml_loge("Profiling Initialization failure");
    return -1;
  }

  LOGD("init: Registering Op Packages");
  if (StatusCode::SUCCESS != this->registerOpPackages()) {
    LOGE("init: Register Op Packages failure");
    ml_loge("Register Op Packages failure");
    return -1;
  }

  LOGD("init: END (returning 0)");
  return 0;
}

template <typename T>
const int QNNContext::registerFactory(const FactoryType<T> factory,
                                      const std::string &key,
                                      const int int_key) {
  static_assert(isSupported<T>::value,
                "qnn_context: given type is not supported for current context");

  auto &index = std::get<IndexType<T>>(factory_map);
  auto &str_map = std::get<StrIndexType<T>>(index);
  auto &int_map = std::get<IntIndexType>(index);

  std::string assigned_key = key == "" ? factory({})->getType() : key;

  std::transform(assigned_key.begin(), assigned_key.end(), assigned_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::lock_guard<std::mutex> lock(qnn_factory_mutex);
  if (str_map.find(assigned_key) != str_map.end()) {
    // std::stringstream ss;
    // ss << "qnn_context: cannot register factory with already taken key: "
    //    << key;
    // throw std::invalid_argument(ss.str().c_str());
    for (const auto &[ik, sk] : int_map) {
      if (sk == assigned_key)
        return ik;
    }

    return -1;
  }
  if (int_key != -1 && int_map.find(int_key) != int_map.end()) {
    // std::stringstream ss;
    // ss << "qnn_context: cannot register factory with already taken int key: "
    //    << int_key;
    // throw std::invalid_argument(ss.str().c_str());
    return int_key;
  }

  int assigned_int_key = int_key == -1 ? str_map.size() + 1 : int_key;

  str_map[assigned_key] = factory;
  int_map[assigned_int_key] = assigned_key;

  ml_logd("qnn_context: factory has registered with key: %s, int_key: %d",
          assigned_key.c_str(), assigned_int_key);

  return assigned_int_key;
}

StatusCode QNNContext::isDevicePropertySupported() {
  auto qnn_data = getQnnData();
  if (nullptr !=
      qnn_data->m_qnnFunctionPointers.qnnInterface.propertyHasCapability) {
    auto qnnStatus =
      qnn_data->m_qnnFunctionPointers.qnnInterface.propertyHasCapability(
        QNN_PROPERTY_GROUP_DEVICE);
    if (QNN_PROPERTY_NOT_SUPPORTED == qnnStatus) {
      ml_logw("Device property is not supported");
    }
    if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnStatus) {
      ml_loge("Device property is not known to backend");
      return StatusCode::FAILURE;
    }
  }
  return StatusCode::SUCCESS;
}

StatusCode QNNContext::createDevice() {
  auto qnn_data = getQnnData();
  QnnDevice_Config_t **deviceConfigs{nullptr};
  uint32_t configCount{0};
  uint32_t socModel{0};
  auto backend_extensions = qnn_data->m_backendExtensions;

  if (nullptr != backend_extensions && backend_extensions->interface()) {
    if (!backend_extensions->interface()->beforeCreateDevice(
          &deviceConfigs, &configCount, socModel)) {
      QNN_ERROR("Extensions Failure in beforeCreateDevice()");
      return StatusCode::FAILURE;
    }
  }
  std::vector<const QnnDevice_Config_t *> deviceConfigPointers(configCount + 1,
                                                               nullptr);
  for (size_t idx = 0u; idx < configCount; idx++) {
    deviceConfigPointers[idx] = deviceConfigs[idx];
  }
  if (nullptr != qnn_data->m_qnnFunctionPointers.qnnInterface.deviceCreate) {
    auto qnnStatus = qnn_data->m_qnnFunctionPointers.qnnInterface.deviceCreate(
      qnn_data->m_logHandle, deviceConfigPointers.data(),
      &qnn_data->m_deviceHandle);
    if (QNN_SUCCESS != qnnStatus &&
        QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
      ml_loge("Failed to create device (QNN error=0x%x)", (unsigned)qnnStatus);
      return verifyFailReturnStatus(qnnStatus);
    }
  }
  if (nullptr != backend_extensions && backend_extensions->interface()) {
    if (!backend_extensions->interface()->afterCreateDevice()) {
      QNN_ERROR("Extensions Failure in afterCreateDevice()");
      return StatusCode::FAILURE;
    }
  }
  return StatusCode::SUCCESS;
}

StatusCode QNNContext::verifyFailReturnStatus(Qnn_ErrorHandle_t errCode) {
  auto returnStatus = StatusCode::FAILURE;
  switch (errCode) {
  case QNN_COMMON_ERROR_SYSTEM_COMMUNICATION:
    returnStatus = StatusCode::FAILURE_SYSTEM_COMMUNICATION_ERROR;
    break;
  case QNN_COMMON_ERROR_SYSTEM:
    returnStatus = StatusCode::FAILURE_SYSTEM_ERROR;
    break;
  case QNN_COMMON_ERROR_NOT_SUPPORTED:
    returnStatus = StatusCode::QNN_FEATURE_UNSUPPORTED;
    break;
  default:
    break;
  }
  return returnStatus;
}

StatusCode QNNContext::initializeProfiling() {
  auto qnn_data = getQnnData();
  if (ProfilingLevel::OFF != qnn_data->m_profilingLevel) {
    ml_logi("Profiling turned on; level = %d", (int)qnn_data->m_profilingLevel);
    if (ProfilingLevel::BASIC == qnn_data->m_profilingLevel) {
      ml_logi("Basic profiling requested. Creating Qnn Profile object.");
      if (QNN_PROFILE_NO_ERROR !=
          qnn_data->m_qnnFunctionPointers.qnnInterface.profileCreate(
            qnn_data->m_backendHandle, QNN_PROFILE_LEVEL_BASIC,
            &qnn_data->m_profileBackendHandle)) {
        ml_logw("Unable to create profile handle in the backend.");
        return StatusCode::FAILURE;
      }
    } else if (ProfilingLevel::DETAILED == qnn_data->m_profilingLevel) {
      ml_logi("Detailed profiling requested. Creating Qnn Profile object.");
      if (QNN_PROFILE_NO_ERROR !=
          qnn_data->m_qnnFunctionPointers.qnnInterface.profileCreate(
            qnn_data->m_backendHandle, QNN_PROFILE_LEVEL_DETAILED,
            &qnn_data->m_profileBackendHandle)) {
        ml_loge("Unable to create profile handle in the backend.");
        return StatusCode::FAILURE;
      }
    }
  }
  return StatusCode::SUCCESS;
}

StatusCode QNNContext::registerOpPackages() {
  const size_t pathIdx = 0;
  const size_t interfaceProviderIdx = 1;
  auto qnn_data = getQnnData();
  std::cout << qnn_data->name << std::endl;

  for (auto const &opPackagePath : m_opPackagePaths) {
    std::vector<std::string> opPackage;
    qnn::tools::sample_app::split(opPackage, opPackagePath, ':');
    ml_logi("opPackagePath: %s", opPackagePath.c_str());
    const char *target = nullptr;
    const size_t targetIdx = 2;
    if (opPackage.size() != 2 && opPackage.size() != 3) {
      ml_loge("Malformed opPackageString provided: %s", opPackagePath.c_str());
      return StatusCode::FAILURE;
    }
    if (opPackage.size() == 3) {
      target = (char *)opPackage[targetIdx].c_str();
    }
    if (nullptr ==
        qnn_data->m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage) {
      ml_loge("backendRegisterOpPackageFnHandle is nullptr.");
      return StatusCode::FAILURE;
    }
    if (QNN_BACKEND_NO_ERROR !=
        qnn_data->m_qnnFunctionPointers.qnnInterface.backendRegisterOpPackage(
          qnn_data->m_backendHandle, (char *)opPackage[pathIdx].c_str(),
          (char *)opPackage[interfaceProviderIdx].c_str(), target)) {
      ml_loge("Could not register Op Package: %s and interface provider: %s",
              opPackage[pathIdx].c_str(),
              opPackage[interfaceProviderIdx].c_str());
      return StatusCode::FAILURE;
    }
    ml_logi("Registered Op Package: %s and interface provider: %s",
            opPackage[pathIdx].c_str(),
            opPackage[interfaceProviderIdx].c_str());
  }
  return StatusCode::SUCCESS;
}

void QNNContext::release() {
  auto devicePropertySupportStatus = this->isDevicePropertySupported();
  if (StatusCode::FAILURE != devicePropertySupportStatus) {
    auto freeDevcieStatus = this->freeDevice();
    if (StatusCode::SUCCESS != freeDevcieStatus) {
      ml_loge("Device Free Failure");
    }
  }
}

StatusCode QNNContext::freeDevice() {
  auto qnn_data = getQnnData();

  if (nullptr != qnn_data->m_qnnFunctionPointers.qnnInterface.deviceFree) {
    auto qnnStatus = qnn_data->m_qnnFunctionPointers.qnnInterface.deviceFree(
      qnn_data->m_deviceHandle);
    if (QNN_SUCCESS != qnnStatus &&
        QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
      ml_loge("Failed to free devcie");
      return verifyFailReturnStatus(qnnStatus);
    }
  }
  return StatusCode::SUCCESS;
}

/**
 * @copydoc const int QNNContext::registerFactory
 */
template const int QNNContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

#ifdef PLUGGABLE
nntrainer::Context *create_qnn_context() {
  nntrainer::QNNContext *qnn_context = new nntrainer::QNNContext();
  if (!g_default_backend_ext_config_path.empty()) {
    qnn_context->setBackendExtConfigPath(g_default_backend_ext_config_path);
  }
  qnn_context->initializeOnce();
  return qnn_context;
}

void destory_qnn_context(nntrainer::Context *ct) { delete ct; }

extern "C" {
nntrainer::ContextPluggable ml_train_context_pluggable{create_qnn_context,
                                                       destory_qnn_context};
}
#endif
} // namespace nntrainer

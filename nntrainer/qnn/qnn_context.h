/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    qnn_context.h
 * @date    10 Dec 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains app context related functions and classes that
 * manages the global configuration of the current QNN environment.
 */

#ifndef __QNN_CONTEXT_H__
#define __QNN_CONTEXT_H__

#include "Log/Logger.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QNN/QnnTypes.h"
#include "Utils/IOTensor.hpp"
#include "qnn_rpc_manager.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <context.h>
#include <layer.h>
#include <layer_devel.h>

#include <nntrainer_log.h>
#include <qnn_context_var.h>

/* LOGD macro for QNN context if not defined */
#if defined(__ANDROID__)
#include <android/log.h>
#ifndef LOG_TAG
#define LOG_TAG "qnn_context"
#endif
#ifndef LOGD
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#endif
#else
#ifndef LOG_TAG
#define LOG_TAG "qnn_context"
#endif
#ifndef LOGD
#include <cstdio>
#define LOGD(...)                                                              \
  do {                                                                         \
    fprintf(stderr, "[DEBUG][%s] ", LOG_TAG);                                  \
    fprintf(stderr, __VA_ARGS__);                                              \
    fprintf(stderr, "\n");                                                     \
  } while (0)
#endif
#endif

#include "singleton.h"

using namespace qnn;
using namespace qnn::tools;

namespace nntrainer {

extern std::mutex qnn_factory_mutex;

/**
 * @class QNNContext contains user-dependent configuration for QNN support
 * @brief QNN support for app context
 */

class QNNContext : public Context, public Singleton<QNNContext> {

public:
  /**
   * @brief   Default constructor
   */
  QNNContext() : Context(std::make_shared<QNNBackendVar>()) {}

  ~QNNContext() {
    auto qnn_data = getQnnData();
    // Free all remaining QNN contexts in ct_map before releasing backend
    if (qnn_data) {
      qnn_data->freeAllContexts();
    }
    if ((qnn_data->m_isBackendInitialized &&
         nullptr != qnn_data->m_qnnFunctionPointers.qnnInterface.backendFree) &&
        QNN_BACKEND_NO_ERROR !=
          qnn_data->m_qnnFunctionPointers.qnnInterface.backendFree(
            qnn_data->m_backendHandle)) {
      ml_loge("Could not terminate backed");
    }
    qnn_data->m_isBackendInitialized = false;
    this->release();
    if (qnn_data->m_backendLibraryHandle) {
      pal::dynamicloading::dlClose(qnn_data->m_backendLibraryHandle);
    }
  }

  int init() override;

  /**
   * @brief Factory register function, use this function to register custom
   * object
   *
   * @tparam T object to create. Currently Layer is supported
   * @param factory factory function that creates std::unique_ptr<T>
   * @param key key to access the factory, if key is empty, try to find key by
   * calling factory({})->getType();
   * @param int_key key to access the factory by integer, if it is -1(default),
   * the function automatically unsigned the key and return
   * @return const int unique integer value to access the current factory
   * @throw invalid argument when key and/or int_key is already taken
   */
  template <typename T>
  const int registerFactory(const PtrFactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1) {
    FactoryType<T> f = factory;
    return registerFactory(f, key, int_key);
  }

  /**
   * @brief Factory register function, use this function to register custom
   * object
   *
   * @tparam T object to create. Currently Layer is supported
   * @param factory factory function that creates std::unique_ptr<T>
   * @param key key to access the factory, if key is empty, try to find key by
   * calling factory({})->getType();
   * @param int_key key to access the factory by integer, if it is -1(default),
   * the function automatically unsigned the key and return
   * @return const int unique integer value to access the current factory
   * @throw invalid argument when key and/or int_key is already taken
   */
  template <typename T>
  const int registerFactory(const FactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1);

  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &properties) override {
    return createObject<nntrainer::Layer>(type, properties);
  }

  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(int_key, properties);
  }

  /**
   * @brief Create an Object from the integer key
   *
   * @tparam T Type of Object, currently, Only Layer is supported
   * @param int_key integer key
   * @param props property
   * @return PtrType<T> unique pointer to the object
   */
  template <typename T>
  PtrType<T> createObject(const int int_key,
                          const PropsType &props = {}) const {
    static_assert(isSupported<T>::value,
                  "given type is not supported for current app context");
    auto &index = std::get<IndexType<T>>(factory_map);
    auto &int_map = std::get<IntIndexType>(index);

    LOGD("%s:%d", __FILE__, __LINE__);

    const auto &entry = int_map.find(int_key);

    if (entry == int_map.end()) {
      std::stringstream ss;
      ss << "Int Key is not found for the object. Key: " << int_key;
      throw exception::not_supported(ss.str().c_str());
    }

    // entry is an object of int_map which is an unordered_map<int, std::string>
    return createObject<T>(entry->second, props);
  }

  /**
   * @brief Create an Object from the string key
   *
   * @tparam T Type of object, currently, only Layer is supported
   * @param key integer key
   * @param props property
   * @return PtrType<T> unique pointer to the object
   */
  template <typename T>
  PtrType<T> createObject(const std::string &key,
                          const PropsType &props = {}) const {
    auto &index = std::get<IndexType<T>>(factory_map);
    auto &str_map = std::get<StrIndexType<T>>(index);

    // LOGD("%s:%d", __FILE__, __LINE__);

    std::string lower_key;
    lower_key.resize(key.size());

    // LOGD("%s:%d", __FILE__, __LINE__);

    std::transform(key.begin(), key.end(), lower_key.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    //    LOGD("%s:%d", __FILE__, __LINE__);

    const auto &entry = str_map.find(lower_key);

    if (entry == str_map.end()) {
      std::stringstream ss;
      ss << "Key is not found for the object. Key: " << lower_key;
      //    LOGD("%s:%d, key is not found", __FILE__, __LINE__);

      throw exception::not_supported(ss.str().c_str());
    }

    // LOGD("%s:%d", __FILE__, __LINE__);

    // entry -> object of str_map -> unordered_map<std::string, FactoryType<T>>
    return entry->second(props);
  }

  std::string getName() override { return "qnn"; }

  /**
   * @brief   Set the default backend extension config path before singleton
   * initialization. Must be called before QNNContext::Global() or any
   * operation that triggers context creation. Relative paths are resolved from
   * QUICK_DOT_AI_BASE_DIR when set, otherwise from the current working
   * directory.
   */
  static void setDefaultBackendExtConfigPath(const std::string &path);

  void setBackendExtConfigPath(const std::string &path) {
    m_backendExtConfigPath = path;
  }

  void setMemAllocator(std::shared_ptr<QNNRpcManager> mem) {
    getContextData()->setMemAllocator(mem);
  }

  std::shared_ptr<QNNVar> getQnnData() {
    std::shared_ptr<QNNBackendVar> d =
      std::static_pointer_cast<QNNBackendVar>(this->getContextData());
    return d->getVar();
  }

  int load(const std::string &file_path) override {

    StatusCode ret = getQnnData()->makeContext(file_path);
    return (int)ret;
  }

private:
  void initialize() noexcept override;

  // flag to check predefined qnn context is resistered
  bool qnn_initialized = false;

  FactoryMap<nntrainer::Layer> factory_map;

  template <typename Args, typename T> struct isSupportedHelper;

  /**
   * @brief supportHelper to check if given type is supported within cl context
   */
  template <typename T, typename... Args>
  struct isSupportedHelper<T, QNNContext::FactoryMap<Args...>> {
    static constexpr bool value =
      (std::is_same_v<std::decay_t<T>, std::decay_t<Args>> || ...);
  };

  /**
   * @brief supportHelper to check if given type is supported within cl context
   */
  template <typename T>
  struct isSupported : isSupportedHelper<T, decltype(factory_map)> {};

  std::vector<std::string> m_opPackagePaths;

  std::string m_backendExtConfigPath;

  bool m_isContextCreated;

  static StatusCode
  QnnModel_freeGraphsInfo(qnn_wrapper_api::GraphInfoPtr_t **graphsInfo,
                          uint32_t numGraphsInfo) {
    if (graphsInfo == nullptr || *graphsInfo == nullptr) {
      PRINT_ERROR("freeGraphsInfo() invalid graphsInfo.");
      return StatusCode::FAILURE;
    }
    for (uint32_t i = 0; i < numGraphsInfo; i++) {
      PRINT_INFO("Freeing graph in freeGraphInfo");
      free((*graphsInfo)[i]->graphName);
      qnn_wrapper_api::freeQnnTensors((*graphsInfo)[i]->inputTensors,
                                      (*graphsInfo)[i]->numInputTensors);
      qnn_wrapper_api::freeQnnTensors((*graphsInfo)[i]->outputTensors,
                                      (*graphsInfo)[i]->numOutputTensors);
    }

    free(**graphsInfo);
    free(*graphsInfo);
    *graphsInfo = nullptr;

    return StatusCode::SUCCESS;
  }

  StatusCode isDevicePropertySupported();

  StatusCode createDevice();

  StatusCode verifyFailReturnStatus(Qnn_ErrorHandle_t errCode);

  StatusCode initializeProfiling();

  StatusCode registerOpPackages();

  void release();

  StatusCode freeDevice();
};

/**
 * @copydoc const int QNNContext::registerFactory
 */
extern template const int QNNContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

} // namespace nntrainer

#endif /* __QNN_CONTEXT_H__ */

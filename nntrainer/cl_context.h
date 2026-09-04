// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    cl_context.h
 * @date    23 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains app context related functions and classes that
 * manages the global configuration of the current OpenCL environment. It also
 * creates the OpenCL command queue and context.
 */

#ifndef __CL_CONTEXT_H__
#define __CL_CONTEXT_H__

#include <algorithm>
#include <cstdlib>
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
#include <mem_allocator.h>

#include <opencl_buffer_manager.h>
#include <opencl_command_queue_manager.h>
#include <opencl_context_manager.h>
#include <opencl_kernel.h>
#include <opencl_program.h>

#include "singleton.h"

namespace nntrainer {

extern std::mutex cl_factory_mutex;

/**
 * @class ClContext contains user-dependent configuration for OpenCL support
 * @brief OpenCL support for app context
 */

class ClContext : public Context, public Singleton<ClContext> {
public:
  using SharedPtrClKernel = std::shared_ptr<opencl::Kernel>;

  /** string to kernel pointer map*/
  using OclKernelMap = std::unordered_map<std::string, SharedPtrClKernel>;

  // getting static instance of commandqueue, opencl context and buffermanager
  opencl::CommandQueueManager &command_queue_inst_ =
    opencl::CommandQueueManager::Global();

  opencl::ContextManager &context_inst_ = opencl::ContextManager::Global();

  ClBufferManager &clbuffInstance = ClBufferManager::Global();

  /**
   * @brief   Default constructor
   */
  ClContext() : Context(std::make_shared<ContextData>()) {}

  /**
   * @brief destructor to release opencl commandQueue
   */
  ~ClContext() override { releaseClResources(); };

  /**
   * @brief release the buffers, command queue and context owned here
   *
   * @note Idempotent: clInit() arms this as an exit handler, and the
   * destructor calls it again for the paths that never got there.
   */
  void releaseClResources() {
    if (!cl_initialized)
      return;
    cl_initialized = false;

    // SVM regions and device buffers belong to the context below, so they
    // have to go back first.
    clbuffInstance.releaseBuffers();
    command_queue_inst_.ReleaseCommandQueue();
    // getContext() is called by clCreateKernel
    context_inst_.ReleaseContext();
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

    std::string lower_key;
    lower_key.resize(key.size());

    std::transform(key.begin(), key.end(), lower_key.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    const auto &entry = str_map.find(lower_key);

    if (entry == str_map.end()) {
      std::stringstream ss;
      ss << "Key is not found for the object. Key: " << lower_key;
      throw exception::not_supported(ss.str().c_str());
    }

    // entry -> object of str_map -> unordered_map<std::string, FactoryType<T>>
    return entry->second(props);
  }

  /**
   * @brief Create a Layer object from the string key
   *
   * @param type string key
   * @param properties property
   * @return std::unique_ptr<nntrainer::Layer> unique pointer to the object
   */
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(type, properties);
  }

  /**
   * @brief Create a Layer object from the integer key
   *
   * @param type integer key
   * @param properties property
   * @return std::unique_ptr<nntrainer::Layer> unique pointer to the object
   */
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(int_key, properties);
  }

  /**
   * @brief register or return already present OpenCl kernel pointer
   * @param kernel_string kernel implementation string
   * @param kernel_name kernel name
   * @return std::shared_ptr<opencl::Kernel>
   */
  const SharedPtrClKernel registerClKernel(std::string kernel_string,
                                           std::string kernel_name,
                                           std::string compile_options = {});

  /**
   * @brief Initialize and register all blas OpenCl kernels
   */
  void initBlasClKernels();

  /**
   * @brief Initialize and register all attention OpenCl kernels
   */
  void initAttentionClKernels();

  /**
   * @brief Get the name of the context
   */
  std::string getName() override { return "gpu"; }

  /**
   * @brief Set the Mem Allocator object
   *
   * @param mem Memory allocator object
   */
  void setMemAllocator(std::shared_ptr<MemAllocator> mem) {
    getContextData()->setMemAllocator(mem);
  }

private:
  /**
   * @brief   Overriden initialization function
   */
  void initialize() noexcept override;

  void add_default_object();

  // flag to check opencl commandqueue and context inititalization
  bool cl_initialized = false;

  // flag to check default blas kernels registered or not
  bool blas_kernels_initialized = false;

  // flag to check default attention kernels registered or not
  bool attention_kernels_initialized = false;

  FactoryMap<nntrainer::Layer> factory_map;

  template <typename Args, typename T> struct isSupportedHelper;

  // global map to store initialized opencl::Kernel
  inline static OclKernelMap ocl_kernel_map;

  /**
   * @brief supportHelper to check if given type is supported within cl context
   */
  template <typename T, typename... Args>
  struct isSupportedHelper<T, ClContext::FactoryMap<Args...>> {
    static constexpr bool value =
      (std::is_same_v<std::decay_t<T>, std::decay_t<Args>> || ...);
  };

  /**
   * @brief supportHelper to check if given type is supported within cl context
   */
  template <typename T>
  struct isSupported : isSupportedHelper<T, decltype(factory_map)> {};

  /**
   * @brief Initialize opencl commandqueue and context
   * @return true if OpenCL context and command queue creation is successful,
   * false otherwise
   */

  bool clInit() {
    // if commandqueue already created
    if (cl_initialized)
      return true;

    // getContext() called inside createCommandQueue which creates clContext
    bool result = command_queue_inst_.CreateCommandQueue();

    // Return false when creating command queue fails
    if (!result)
      return result;

    // initialize device buffers
    clbuffInstance.initBuffers();
    cl_initialized = result;

    // CreateCommandQueue() above dlopen()s the OpenCL driver, so the driver
    // registered its own exit handlers *after* the singletons here were
    // constructed. Exit handlers run in reverse registration order, so the
    // driver tears itself down first and the singleton destructors then call
    // clSVMFree()/clRelease*() on finalized driver state. Registering the
    // teardown from here, after the driver is loaded, puts it ahead of the
    // driver's own handlers.
    std::atexit([]() { ClContext::Global().releaseClResources(); });

    return cl_initialized;
  };

  /**
   * @brief create OpenCl kernel
   * @param kernel_string reference of implementation string
   * @param kernel_name reference of kernel_name
   * @param kernel_ptr_ reference of shared_ptr of Kernel
   * @return true if successful, false otherwise
   */
  bool clCreateKernel(std::string &kernel_string, std::string &kernel_name,
                      std::string &compile_options,
                      const SharedPtrClKernel &kernel_ptr_);
};

/**
 * @copydoc const int ClContext::registerFactory
 */
extern template const int ClContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

} // namespace nntrainer

#endif /* __CL_CONTEXT_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   engine.h
 * @date   27 December 2024
 * @brief  This file contains engine context related functions and classes that
 * manages the engines (NPU, GPU, CPU) of the current environment
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __ENGINE_H__
#define __ENGINE_H__

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
#include <mem_allocator.h>
#include <nntrainer_error.h>

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
#include <cl_context.h>
#endif

// QNN context is loaded as a plugin .so (libqnn_context.so)
// No header dependency needed here.

#include "singleton.h"

#if defined(__GNUC__) || defined(__clang__)
#define NNTR_VISIBILITY __attribute__((visibility("default")))
#else
#define NNTR_VISIBILITY
#endif

namespace nntrainer {

extern std::mutex engine_mutex;
namespace {} // namespace

/**
 * @class Engine contains user-dependent configuration
 * @brief App
 */
class NNTR_VISIBILITY Engine : public Singleton<Engine> {
protected:
  static const int RegisterContextMax = 16;
  static nntrainer::Context *nntrainerRegisteredContext[RegisterContextMax];
  /// Valgrind complains memory leaks with context registered because
  /// every context is alive during the whole application lifecycle
  /// and we do not free them. It can be amended by using unique_ptr;
  /// however, as we use container and function calls with context,
  /// let's not bother modify all the related functions, but waste
  /// a few words.

  void initialize() noexcept override;

  void add_default_object();

  void registerContext(std::string name, nntrainer::Context *context,
                       void *library_handle = nullptr,
                       DestroyContextFunc destroy_func = nullptr) {
    const std::lock_guard<std::mutex> lock(engine_mutex);
    static int registerCount = 0;

    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (engines.find(name) != engines.end()) {
      // Re-registering an already-registered Context is a no-op, not an error.
      // QuickAI QNN models register the process-global "qnn" Context once per
      // model in Quick_Dot_AI_QNN::initialize(); a multi-model handle (e.g. the
      // multimodal [vision, LLM] pair) therefore calls this more than once.
      // Upstream main throws here, but pr/3963 (the working QNN reference)
      // returns early so the 2nd+ models reuse the existing Context. Carried
      // forward from pr/3963.
      return;
    }
    engines.insert(std::make_pair(name, context));

    if (registerCount < RegisterContextMax) {
      nntrainerRegisteredContext[registerCount] = context;
      registerCount++;
    }

    auto alloc = context->getMemAllocator();

    allocator.insert(std::make_pair(name, alloc));
  }

public:
  /**
   * @brief   Get the single process-wide Engine instance.
   * @note    Overrides Singleton<Engine>::Global() with an out-of-line
   *          definition in engine.cpp so there is exactly ONE Engine instance
   *          across all shared libraries. The inherited template Global() is an
   *          inline method, which under -fvisibility=hidden / per-namespace
   *          loading gets instantiated separately in each consumer .so
   *          (libcausallm, libquick_dot_ai, libqnn_context, ...). That produced
   *          multiple Engine instances: a context registered into one (e.g.
   *          "qnn" via Quick_Dot_AI_QNN in libquick_dot_ai) was invisible to
   *          another (NetworkGraph in libnntrainer), surfacing as
   *          std::invalid_argument "[Engine] qnn Context is not registered".
   *          A single out-of-line definition in libnntrainer.so makes every
   *          Engine::Global() caller share the same instance.
   */
  static Engine &Global();

  /**
   * @brief   Default constructor
   */
  Engine() = default;

  /**
   * @brief   Default Destructor
   */
  ~Engine() = default;

  /**
   * @brief   Release resources allocated by Engine
   *
   */
  void release();

  /**
   * @brief register a Context from a shared library
   * plugin must have **extern "C" LayerPluggable *ml_train_context_pluggable**
   * defined else error
   *
   * @param library_path a file name of the library
   * @param base_path    base path to make a full path (optional)
   * @throws std::invalid_parameter if library_path is invalid or library is
   * invalid
   */
  int registerContext(const std::string &library_path,
                      const std::string &base_path = "");

  /**
   * @brief get registered a Context
   *
   * @param name Registered Context Name
   * @throws std::invalid_parameter if no context with name
   * @return Context Pointer : for register Object factory, casting might be
   * needed.
   */
  nntrainer::Context *getRegisteredContext(std::string name) const {

    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (engines.find(name) == engines.end()) {
      throw std::invalid_argument("[Engine] " + name +
                                  " Context is not registered");
    }
    return engines.at(name);
  }

  std::unordered_map<std::string, std::shared_ptr<nntrainer::MemAllocator>>
  getAllocators() {
    return allocator;
  }

  /**
   *
   * @brief Parse compute Engine keywords in properties : eg) engine = cpu
   *  default is "cpu"
   * @return Context name
   */
  std::string parseComputeEngine(const std::vector<std::string> &props) const;

  /**
   * @brief Create an Layer Object with Layer name
   *
   * @param type layer name
   * @param props property
   * @return unitque_ptr<T> unique pointer to the Layer object
   */
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &properties = {}) const {
    auto ct = getRegisteredContext(parseComputeEngine(properties));
    return ct->createLayerObject(type);
  }

  /**
   * @brief Create an Layer Object with Layer key
   *
   * @param int_key key
   * @param props property
   * @return unitque_ptr<T> unique pointer to the Layer object
   */
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &properties = {}) const {
    auto ct = getRegisteredContext(parseComputeEngine(properties));
    return ct->createLayerObject(int_key);
  }

  /**
   * @brief Create an Optimizer Object with Optimizer name
   *
   * @param type Optimizer name
   * @param props property
   * @return unitque_ptr<T> unique pointer to the Optimizer object
   */
  std::unique_ptr<nntrainer::Optimizer>
  createOptimizerObject(const std::string &type,
                        const std::vector<std::string> &properties = {}) const {
    auto ct = getRegisteredContext(parseComputeEngine(properties));
    return ct->createOptimizerObject(type);
  }

  /**
   * @brief Create an Optimizer Object with Optimizer key
   *
   * @param int_key key
   * @param props property
   * @return unitque_ptr<T> unique pointer to the Optimizer object
   */
  std::unique_ptr<nntrainer::Optimizer>
  createOptimizerObject(const int int_key,
                        const std::vector<std::string> &properties = {}) const {
    auto ct = getRegisteredContext(parseComputeEngine(properties));
    return ct->createOptimizerObject(int_key);
  }

  /**
   * @brief Create an LearningRateScheduler Object with type
   *
   * @param type type of LearningRateScheduler
   * @param props property
   * @return unitque_ptr<T> unique pointer to the LearningRateScheduler object
   */
  std::unique_ptr<ml::train::LearningRateScheduler>
  createLearningRateSchedulerObject(
    const std::string &type,
    const std::vector<std::string> &properties = {}) const {
    auto ct = getRegisteredContext(parseComputeEngine(properties));
    return ct->createLearningRateSchedulerObject(type, properties);
  }

  /**
   * @brief Create an LearningRateScheduler Object with key
   *
   * @param int_key key
   * @param props property
   * @return unitque_ptr<T> unique pointer to the LearningRateScheduler object
   */
  std::unique_ptr<ml::train::LearningRateScheduler>
  createLearningRateSchedulerObject(
    const int int_key, const std::vector<std::string> &properties = {}) {
    auto ct = getRegisteredContext(parseComputeEngine(properties));
    return ct->createLearningRateSchedulerObject(int_key, properties);
  }

  /**
   * @brief Get Working Path from a relative or representation of a path
   * starting from @a working_path_base.
   * @param[in] path to make full path
   * @return If absolute path is given, returns @a path
   * If relative path is given and working_path_base is not set, return
   * relative path.
   * If relative path is given and working_path_base has set, return absolute
   * path from current working directory
   */
  const std::string getWorkingPath(const std::string &path = "") const;

  /**
   * @brief Set Working Directory for a relative path. working directory is set
   * canonically
   * @param[in] base base directory
   * @throw std::invalid_argument if path is not valid for current system
   */
  void setWorkingDirectory(const std::string &base);

  /**
   * @brief unset working directory
   *
   */
  void unsetWorkingDirectory() { working_path_base = ""; }

  /**
   * @brief query if the appcontext has working directory set
   *
   * @retval true working path base is set
   * @retval false working path base is not set
   */
  bool hasWorkingDirectory() { return !working_path_base.empty(); }

private:
  /**
   * @brief map for Context and Context name
   *
   */
  std::unordered_map<std::string, nntrainer::Context *> engines;

  std::unordered_map<std::string, std::shared_ptr<nntrainer::MemAllocator>>
    allocator;

  std::string working_path_base;
};

namespace plugin {}

} // namespace nntrainer

#endif /* __ENGINE_H__ */

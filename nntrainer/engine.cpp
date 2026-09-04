// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   engine.cpp
 * @date   27 December 2024
 * @brief  This file contains engine context related functions and classes that
 * manages the engines (NPU, GPU, CPU) of the current environment
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <app_context.h>
#include <base_properties.h>
#include <compute_ops.h>
#include <context.h>
#include <dynamic_library_loader.h>
#include <engine.h>
#if defined(ENABLE_HEXKL) && ENABLE_HEXKL == 1
#include <htp_context.h>
#endif

static std::string solib_suffix = ".so";
static std::string contextlib_suffix = "context.so";
static const std::string func_tag = "[Engine] ";

namespace nntrainer {

std::mutex engine_mutex;

std::once_flag global_engine_init_flag;

nntrainer::Context
  *Engine::nntrainerRegisteredContext[Engine::RegisterContextMax];

Engine &Engine::Global() {
  // Single definition in libnntrainer.so → one Engine instance shared by every
  // consumer .so (see declaration in engine.h). initializeOnce() registers the
  // default contexts (cpu/gpu, and qnn when ENABLE_NPU) exactly once.
  static Engine instance;
  instance.initializeOnce();
  return instance;
}

// Guarded on exactly the backends that call bringUpWanted() below, which today
// is OpenCL alone. Widening this to ENABLE_CUDA before a CUDA call site exists
// leaves two unused statics and a -Wunused-function in a CUDA-on/OpenCL-off
// build; the guard widens in the change that adds the second caller.
#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
namespace {

/**
 * @brief The compute engine this process asked for, lowercased, or "" if unset.
 *
 * Read from NNTR_ENGINE. This is the process-wide backend selector: a consumer
 * that runs one engine exports it before the first Engine::Global(), and that
 * ordering is the contract -- the value is latched on first use, so a later
 * setenv() has no effect. Unset means "no preference", which keeps the
 * historical default (see bringUpWanted's on_by_default).
 *
 * @note This is the only reader of NNTR_ENGINE in the tree. A backend that adds
 *       its own read must route through this function rather than comparing
 *       getenv() exactly, or NNTR_ENGINE=GPU registers a context that the
 *       backend's own gate then declines.
 */
const std::string &requestedEngine() {
  static const std::string eng = []() -> std::string {
    const char *e = std::getenv("NNTR_ENGINE");
    if (e == nullptr)
      return std::string();
    std::string s(e);
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
  }();
  return eng;
}

/**
 * @brief Should the named backend's Context be brought up in this process?
 *
 * SYMMETRY IS THE POINT. A process runs ONE compute engine (the consumer
 * resolves exactly one from NNTR_ENGINE), but this translation unit used to be
 * the place where every compiled-in backend was constructed unconditionally, so
 * each lane paid the other's device bring-up:
 *
 *   - a CUDA run enumerated OpenCL and eagerly built ~50 CL programs that no
 *     CL kernel would ever execute (measured 200-218 ms warm on an RTX 5060
 *     box, and ~8.5 s on a launch where the OpenCL driver's compiler cache is
 *     cold)
 *   - a cpu-engine run paid that same OpenCL bring-up, plus the ~2.4 s wake of
 *     a runtime-PM-suspended discrete GPU that the ICD loader triggers
 *
 * @param name       registry name of the backend ("gpu", "cuda")
 * @param eager_env  env var that force-restores the unconditional bring-up
 *                   (kept so the gate is A/B-able and so a genuinely mixed
 *                   process can opt back in)
 * @param on_by_default true for the backend that an unset NNTR_ENGINE selects
 *                   (OpenCL "gpu": the consumer's own default; CUDA is opt-in)
 * @return true when this run should construct that Context
 */
bool bringUpWanted(const char *name, const char *eager_env,
                   bool on_by_default) {
  // Not latched, unlike requestedEngine(): this is the per-backend A/B escape
  // hatch, read once per backend at bring-up, so a static per call site would
  // buy nothing and would need one static per name.
  const char *eager = std::getenv(eager_env);
  if (eager != nullptr && eager[0] != '0')
    return true;
  const std::string &eng = requestedEngine();
  if (eng.empty())
    return on_by_default;
  return eng == name;
}

} // namespace
#endif

void Engine::add_default_object() {
  /// @note all layers should be added to the app_context to guarantee that
  /// createLayer/createOptimizer class is created

  auto &app_context = nntrainer::AppContext::Global();

  // Ensure CPU backend compute-ops table is bound. ensureComputeOps() is
  // std::call_once-guarded, so this call is safe even if AppContext or
  // another Context already initialized it.
  ensureComputeOps();
  registerContext("cpu", &app_context);

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // Engine-conditional, the mirror of the CUDA gate below (see bringUpWanted).
  // ClContext::Global() is leaked-by-design (never destroyed, cl_context.h), so
  // on a non-OpenCL run it would leave a cl_context + command queue and ~50
  // compiled programs alive for the whole process for nothing. Skipping the
  // registration is safe because an engine name that is not registered resolves
  // to "cpu" in parseComputeEngine, and every "gpu" consumer on the production
  // path already try/catches a missing context.
  // NNTR_CL_EAGER_CTX=1 restores the unconditional bring-up.
  if (bringUpWanted("gpu", "NNTR_CL_EAGER_CTX", /*on_by_default=*/true)) {
    auto &cl_context = nntrainer::ClContext::Global();

    registerContext("gpu", &cl_context);
  } else {
    // Warn, not info: a model that ran on the GPU yesterday now runs on the
    // CPU, and this line is the only trace of it.
    ml_logw("OpenCL/gpu backend compiled in but not brought up "
            "(NNTR_ENGINE=%s); engine=gpu layers fall back to cpu.",
            requestedEngine().c_str());
  }
#endif

#if defined(ENABLE_HEXKL) && ENABLE_HEXKL == 1
  auto &htp_context = nntrainer::HtpContext::Global();
  registerContext("htp", &htp_context);
#endif

#if defined(ENABLE_NPU) && ENABLE_NPU == 1
  // QNN context is loaded as a plugin .so for decoupling from QNN SDK.
  // libqnn_context.so exports ml_train_context_pluggable symbol.
  try {
    registerContext("libqnn_context.so", "");
  } catch (std::exception &e) {
    ml_logw("QNN context plugin not available: %s", e.what());
  }
#endif
}

void Engine::initialize() noexcept {
  try {
    add_default_object();
  } catch (std::exception &e) {
    ml_loge("registering layers failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("registering layer failed due to unknown reason");
  }
};

void Engine::release() {}

std::string
Engine::parseComputeEngine(const std::vector<std::string> &props) const {
  for (auto &prop : props) {
    std::string key, value;
    int status = nntrainer::getKeyValue(prop, key, value);
    if (nntrainer::istrequal(key, "engine")) {
      // Validate against the LIVE registered-context name set, not the closed
      // LayerComputeEngine enum + string list. A vendor backend that
      // self-registers a Context (e.g. "npu", "exynos") then resolves with no
      // enum edit; an unknown/unavailable engine falls back to "cpu" instead of
      // resolving to a name that getRegisteredContext would later reject.
      // An engine tag is therefore a registered Context NAME, not a device
      // taxonomy: names are flat keys bound one-per-Context, so "gpu" and
      // "cuda" are disjoint by construction and no tag is a superset of
      // another. Nothing downstream should read a hierarchy into them.
      std::string name = value;
      std::transform(name.begin(), name.end(), name.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (engines.find(name) != engines.end())
        return name;

      // An open registry cannot treat an unknown name as an error, but it must
      // not swallow it either. A typo ("gpuu"), a backend compiled out, and a
      // backend the bring-up gate declined all arrive here and all resolve to
      // the CPU; without this line the only trace is a once-per-process
      // bring-up warning that names no layer. Report the requested and the
      // resolved engine together so the fallback is readable from the log.
      ml_logw("engine=%s is not a backend registered in this process; "
              "resolving this layer to engine=cpu instead",
              value.c_str());
      return "cpu";
    }
  }

  return "cpu";
}

/**
 * @brief Get the Full Path from given string
 * @details path is resolved in the following order
 * 1) if @a path is absolute, return path
 * ----------------------------------------
 * 2) if @a base == "" && @a path == "", return "."
 * 3) if @a base == "" && @a path != "", return @a path
 * 4) if @a base != "" && @a path == "", return @a base
 * 5) if @a base != "" && @a path != "", return @a base + "/" + path
 *
 * @param path path to calculate from base
 * @param base base path
 * @return const std::string
 */
const std::string getFullPath(const std::string &path,
                              const std::string &base) {
  /// if path is absolute, return path
  if (path[0] == '/') {
    return path;
  }

  if (base == std::string()) {
    return path == std::string() ? "." : path;
  }

  return path == std::string() ? base : base + "/" + path;
}

const std::string Engine::getWorkingPath(const std::string &path) const {
  return getFullPath(path, working_path_base);
}

void Engine::setWorkingDirectory(const std::string &base) {
  std::filesystem::path base_path(base);

  if (!std::filesystem::is_directory(base_path)) {
    std::stringstream ss;
    ss << func_tag << "path is not directory or has no permission: " << base;
    throw std::invalid_argument(ss.str().c_str());
  }

  char *ret = getRealpath(base.c_str(), nullptr);

  if (ret == nullptr) {
    std::stringstream ss;
    ss << func_tag << "failed to get canonical path for the path: ";
    throw std::invalid_argument(ss.str().c_str());
  }

  working_path_base = std::string(ret);
  ml_logd("working path base has set: %s", working_path_base.c_str());
  free(ret);
}

int Engine::registerContext(const std::string &library_path,
                            const std::string &base_path) {
  const std::string full_path = getFullPath(library_path, base_path);

  void *handle = DynamicLibraryLoader::loadLibrary(full_path.c_str(),
                                                   RTLD_LAZY | RTLD_LOCAL);
  const char *error_msg = DynamicLibraryLoader::getLastError();

  NNTR_THROW_IF(handle == nullptr, std::invalid_argument)
    << func_tag << "open plugin failed, reason: " << error_msg;

  nntrainer::ContextPluggable *pluggable =
    reinterpret_cast<nntrainer::ContextPluggable *>(
      DynamicLibraryLoader::loadSymbol(handle, "ml_train_context_pluggable"));

  error_msg = DynamicLibraryLoader::getLastError();
  auto close_dl = [handle] { DynamicLibraryLoader::freeLibrary(handle); };
  NNTR_THROW_IF_CLEANUP(error_msg != nullptr || pluggable == nullptr,
                        std::invalid_argument, close_dl)
    << func_tag << "loading symbol failed, reason: " << error_msg;

  auto context = pluggable->createfunc();
  NNTR_THROW_IF_CLEANUP(context == nullptr, std::invalid_argument, close_dl)
    << func_tag << "created pluggable context is null";
  auto type = context->getName();
  NNTR_THROW_IF_CLEANUP(type == "", std::invalid_argument, close_dl)
    << func_tag << "custom layer must specify type name, but it is empty";

  // If this type is already registered (e.g. called again for a second
  // sub-model in a multi-model handle), free the newly-created context
  // immediately rather than leaking it. The name-based overload is the
  // authoritative synchronized check; this is just an early-exit path.
  if (engines.find(type) != engines.end()) {
    pluggable->destroyfunc(context);
    DynamicLibraryLoader::freeLibrary(handle);
    return 0;
  }

  registerContext(type, context);

  return 0;
}

} // namespace nntrainer

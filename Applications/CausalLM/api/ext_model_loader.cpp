// SPDX-License-Identifier: Apache-2.0
/**
 * @file   ext_model_loader.cpp
 * @brief  Host-side loader for the optional model-extension plugin.
 */
#include "ext_model_loader.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <mutex>

#ifdef NNTR_STATIC_MODEL_EXTENSION
extern "C" const QdaExtAbiInfo *qai_ext_model_abi_info(void);
extern "C" int qai_ext_model_register(void);
#endif

#if defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)
#include <dlfcn.h>
#define QDA_EXT_HAVE_DLOPEN 1
#endif

#ifdef __ANDROID__
#include <android/log.h>
#define QDA_EXT_LOGI(...)                                                      \
  __android_log_print(ANDROID_LOG_INFO, "QuickAI", __VA_ARGS__)
#define QDA_EXT_LOGE(...)                                                      \
  __android_log_print(ANDROID_LOG_ERROR, "QuickAI", __VA_ARGS__)
#else
#define QDA_EXT_LOGI(fmt, ...) std::fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#define QDA_EXT_LOGE(fmt, ...) std::fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif

namespace quick_dot_ai {

bool extDlopenErrorMeansAbsent(const std::string &error,
                               const std::string &library) {
  if (error.find(library) == std::string::npos)
    return false;
  // A dependency of the plugin that is missing names the plugin as the
  // requester ("needed by", "needed or dlopened by", "referenced by"): the
  // plugin is there, so that is a failure.
  if (error.find("needed by") != std::string::npos ||
      error.find("dlopened by") != std::string::npos ||
      error.find("referenced by") != std::string::npos)
    return false;
  // bionic: library "libqai_ext_model.so" not found
  // glibc : libqai_ext_model.so: cannot open shared object file: No such ...
  return error.find("not found") != std::string::npos ||
         error.find("No such file") != std::string::npos ||
         error.find("cannot open shared object file") != std::string::npos;
}

bool checkExtAbi(const QdaExtAbiInfo *plugin, const QdaExtAbiInfo &host,
                 std::string *why) {
  auto describe = [](const QdaExtAbiInfo &i) {
    return "abi " + std::to_string(i.abi_version) +
           ", sizeof(ModelDescriptor) " +
           std::to_string(i.sizeof_model_descriptor) +
           ", sizeof(ModelCallbacks) " +
           std::to_string(i.sizeof_model_callbacks);
  };
  if (plugin != nullptr && plugin->abi_version == host.abi_version &&
      plugin->sizeof_model_descriptor == host.sizeof_model_descriptor &&
      plugin->sizeof_model_callbacks == host.sizeof_model_callbacks)
    return true;
  if (why != nullptr) {
    *why = plugin == nullptr
             ? "the plugin returned no ABI info"
             : "ABI mismatch: plugin was built for (" + describe(*plugin) +
                 "), this host is (" + describe(host) +
                 "). Rebuild the plugin against the host's headers.";
  }
  return false;
}

static ExtModelStatus failed(const std::string &library,
                             const std::string &reason) {
  ExtModelStatus s;
  s.state = ExtModelState::FAILED;
  s.message = "failed: " + library + ": " + reason;
  QDA_EXT_LOGE("model extension %s", s.message.c_str());
  return s;
}

ExtModelStatus loadExtensionModelsFrom(const char *library,
                                       bool explicit_path) {
  const std::string lib = library != nullptr ? library : QDA_EXT_LIBRARY_NAME;
  ExtModelStatus status;
#ifndef QDA_EXT_HAVE_DLOPEN
  (void)explicit_path;
  status.state = ExtModelState::ABSENT;
  status.message =
    "absent: model extensions are not supported on this platform";
  return status;
#else
  // Already mapped (System.loadLibrary from Kotlin, a DT_NEEDED edge)? Reuse
  // it; otherwise map it now. Neither runs registration: a plugin has no
  // registering static constructors, so loading is harmless until the ABI
  // check.
  void *handle = dlopen(lib.c_str(), RTLD_NOW | RTLD_NOLOAD);
  if (handle == nullptr) {
    (void)dlerror();
    handle = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
  }
  if (handle == nullptr) {
    const char *e = dlerror();
    const std::string err = e != nullptr ? e : "dlopen failed";
    if (!explicit_path && extDlopenErrorMeansAbsent(err, lib)) {
      status.state = ExtModelState::ABSENT;
      status.message =
        "absent: no " + lib + " on the library path (public model set only)";
      QDA_EXT_LOGI("model extension %s", status.message.c_str());
      return status;
    }
    return failed(lib, "dlopen: " + err);
  }

  auto abi_fn =
    reinterpret_cast<QdaExtAbiInfoFn>(dlsym(handle, QDA_EXT_ABI_INFO_SYMBOL));
  if (abi_fn == nullptr) {
    dlclose(handle);
    return failed(lib, "missing symbol " QDA_EXT_ABI_INFO_SYMBOL
                       " (a plugin built before the ABI check existed, or not "
                       "a model-extension plugin)");
  }
  std::string why;
  if (!checkExtAbi(abi_fn(), currentExtAbiInfo(), &why)) {
    dlclose(handle);
    return failed(lib, why);
  }
  auto reg_fn =
    reinterpret_cast<QdaExtRegisterFn>(dlsym(handle, QDA_EXT_REGISTER_SYMBOL));
  if (reg_fn == nullptr) {
    dlclose(handle);
    return failed(lib, "missing symbol " QDA_EXT_REGISTER_SYMBOL);
  }

  int n = -1;
  try {
    n = reg_fn();
  } catch (const std::exception &e) {
    return failed(lib, std::string("registration threw: ") + e.what());
  } catch (...) {
    return failed(lib, "registration threw an unknown exception");
  }
  if (n < 0)
    return failed(lib, "registration returned " + std::to_string(n));

  status.state = ExtModelState::LOADED;
  status.registrars = n;
  status.message = "loaded: " + lib +
                   " abi=" + std::to_string(QDA_EXT_ABI_VERSION) +
                   " registrars=" + std::to_string(n);
  QDA_EXT_LOGI("model extension %s", status.message.c_str());
  return status;
#endif
}

const ExtModelStatus &ensureExtensionModels() {
  static std::once_flag once;
  static ExtModelStatus status;
  std::call_once(once, []() {
#ifdef NNTR_STATIC_MODEL_EXTENSION
    std::string why;
    if (!checkExtAbi(qai_ext_model_abi_info(), currentExtAbiInfo(), &why)) {
      status = failed("static archive", why);
      return;
    }
    try {
      int count = qai_ext_model_register();
      if (count < 0) {
        status = failed("static archive",
                        "registration returned " + std::to_string(count));
        return;
      }
      status.state = ExtModelState::LOADED;
      status.registrars = count;
      status.message =
        "loaded: static archive abi=" + std::to_string(QDA_EXT_ABI_VERSION) +
        " registrars=" + std::to_string(count);
    } catch (const std::exception &e) {
      status = failed("static archive",
                      std::string("registration threw: ") + e.what());
    } catch (...) {
      status =
        failed("static archive", "registration threw an unknown exception");
    }
#else
    const char *path = std::getenv(QDA_EXT_PATH_ENV);
    if (path != nullptr && path[0] != '\0')
      status = loadExtensionModelsFrom(path, /*explicit_path=*/true);
    else
      status = loadExtensionModelsFrom(QDA_EXT_LIBRARY_NAME, false);
#endif
  });
  return status;
}

} // namespace quick_dot_ai

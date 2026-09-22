// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   dynamic_library_loader.h
 * @date   14 January 2025
 * @brief  Wrapper for loading dynamic libraries on multiple operating systems
 * @see    https://github.com/nntrainer/nntrainer
 * @author Grzegorz Kisala <g.kisala@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __DYNAMIC_LIBRARY_LOADER__
#define __DYNAMIC_LIBRARY_LOADER__

#include <string>

#ifdef _WIN32
#include "windows.h"

// This flags are not used on windows. Defining those symbols for windows make
// possible using the same external interface for loadLibrary function
#define RTLD_LAZY 0
#define RTLD_NOW 0
#define RTLD_BINDING_MASK 0
#define RTLD_NOLOAD 0
#define RTLD_DEEPBIND 0
#define RTLD_GLOBAL 0
#define RTLD_LOCAL 0
#define RTLD_NODELETE 0

#else
#include <dlfcn.h>
#endif

namespace nntrainer {

/**
 * @brief DynamicLibraryLoader wrap process of loading dynamic libraries for
 * multiple operating system
 *
 */
class DynamicLibraryLoader {
public:
  static void *loadLibrary(const char *path, [[maybe_unused]] const int flag) {
#if defined(_WIN32)
    HMODULE handle = LoadLibraryA(path);
    if (handle != nullptr) {
      SetLastError(ERROR_SUCCESS);
    }
    return handle;
#else
    return dlopen(path, flag);
#endif
  }

  static int freeLibrary(void *handle) {
#if defined(_WIN32)
    return FreeLibrary((HMODULE)handle);
#else
    return dlclose(handle);
#endif
  }

  /**
   * @brief Error reported by the most recent loader call
   *
   * @return error description, empty when the last call did not fail
   *
   * @note Follows the dlerror() contract: an empty result means "no error" and
   * reading the error consumes it, so a second call returns an empty string
   * until another loader call fails. On Windows the thread error is sticky and
   * process-wide, so loadLibrary()/loadSymbol() clear it on success to keep an
   * error raised by unrelated code from being reported as a loader failure.
   */
  static std::string getLastError() {
#if defined(_WIN32)
    const DWORD code = GetLastError();
    if (code == ERROR_SUCCESS) {
      return std::string();
    }
    SetLastError(ERROR_SUCCESS);
    return std::to_string(code);
#else
    const char *error = dlerror();
    return error == nullptr ? std::string() : std::string(error);
#endif
  }

  static void *loadSymbol(void *handle, const char *symbol_name) {
#if defined(_WIN32)
    FARPROC symbol = GetProcAddress((HMODULE)handle, symbol_name);
    if (symbol != nullptr) {
      SetLastError(ERROR_SUCCESS);
    }
    return symbol;
#else
    return dlsym(handle, symbol_name);
#endif
  }
};

} // namespace nntrainer

#endif // __DYNAMIC_LIBRARY_LOADER__

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file        dl_error_fixture.cpp
 * @date        02 October 2026
 * @brief       Windows test module whose DllMain raises a Win32 error
 * @see         https://github.com/nntrainer/nntrainer
 * @author      MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 * @note Loaded by unittest_dynamic_library_loader to check that an error
 * raised inside a successful LoadLibraryA() call is not reported by
 * DynamicLibraryLoader::getLastError().
 */

#include <windows.h>

/**
 * @brief Leave a Win32 error behind while the module is being loaded
 *
 * @note extern "C" keeps the entry point unmangled on every toolchain, so the
 * CRT calls this one rather than its default stub.
 */
extern "C" BOOL WINAPI DllMain(HINSTANCE /* instance */, DWORD reason,
                               LPVOID /* reserved */) {
  if (reason == DLL_PROCESS_ATTACH) {
    SetLastError(ERROR_ACCESS_DENIED);
  }
  return TRUE;
}

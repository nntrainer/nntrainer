//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifdef _WIN32

#pragma warning(disable : 4133 4996)

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <windows.h>

#include "dlwrap.hpp"

static const char* last_func;
static long last_err;

void* dlopen(const char* dll, int flags) {
  HINSTANCE h = LoadLibraryA(dll);
  if (h == NULL) {
    last_err  = GetLastError();
    last_func = "dlopen";
  }

  return h;
}

int dlclose(void* h) {
  if (!FreeLibrary((HINSTANCE)h)) {
    last_err  = GetLastError();
    last_func = "dlclose";
    return -1;
  }

  return 0;
}

void* dlsym(void* h, const char* name) {
  FARPROC p = GetProcAddress((HINSTANCE)h, name);
  if (!p) {
    last_err  = GetLastError();
    last_func = "dlsym";
  }
  return (void*)(intptr_t)p;
}

const char* dlerror(void) {
  static char str[88];

  if (!last_err) return NULL;

  sprintf(str, "%s error #%ld", last_func, last_err);
  last_err  = 0;
  last_func = NULL;

  return str;
}

#endif  // _WIN32

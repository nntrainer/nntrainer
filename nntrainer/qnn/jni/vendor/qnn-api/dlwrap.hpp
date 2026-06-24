//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef DLWRAP_HPP
#define DLWRAP_HPP

#ifndef _WIN32

// Just include regular dlfcn
#include <dlfcn.h>

#else  // _WIN32

// Define basic set dl functions and flags

#define RTLD_GLOBAL 0x100
#define RTLD_LOCAL  0x000
#define RTLD_LAZY   0x000
#define RTLD_NOW    0x001

void* dlopen(const char* filename, int flag);
int dlclose(void* handle);
void* dlsym(void* handle, const char* name);
const char* dlerror(void);

#endif  // _WIN32

#endif  // DLWRAP_HPP

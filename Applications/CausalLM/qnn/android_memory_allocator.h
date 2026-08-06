// SPDX-License-Identifier: Apache-2.0
/**
 * @file   android_memory_allocator.h
 * @brief  RPCMEM-backed shared memory allocator for QNN buffers on Android.
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __ANDROID_MEMORY_ALLOCATOR_H__
#define __ANDROID_MEMORY_ALLOCATOR_H__

#include <cstring>
#include <iostream>
#include <map>

void *allocate(size_t fileSize);
void deallocate(void *pointer);

#endif // __ANDROID_MEMORY_ALLOCATOR_H__
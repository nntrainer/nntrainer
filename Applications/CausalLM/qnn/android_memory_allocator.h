#ifndef __ANDROID_MEMORY_ALLOCATOR_H__
#define __ANDROID_MEMORY_ALLOCATOR_H__

#include <cstring>
#include <iostream>
#include <map>

void *allocate(size_t fileSize);
void deallocate(void *pointer);

#endif // __ANDROID_MEMORY_ALLOCATOR_H__
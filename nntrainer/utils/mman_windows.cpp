
// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   mman_windows.h
 * @date   8 April 2025
 * @brief  Windows implementation of posix mman
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Grzegorz Kisala <g.kisala@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#if defined(_WIN32)

#include "mman_windows.h"

#include <errno.h>
#include <io.h>
#include <windows.h>

namespace {

DWORD selectProtect(const int prot) {
  DWORD protect = 0;

  if (prot == PROT_NONE) {
    return protect;
  }

  if (prot & PROT_EXEC) {
    protect = (prot & PROT_WRITE) ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
  } else {
    protect = (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY;
  }

  return protect;
}

DWORD selectDesiredAccess(const int prot) {
  DWORD desiredAccess = 0;

  if (prot == PROT_NONE) {
    return desiredAccess;
  }

  if (prot & PROT_READ) {
    desiredAccess |= FILE_MAP_READ;
  }

  if (prot & PROT_WRITE) {
    desiredAccess |= FILE_MAP_WRITE;
  }

  if (prot & PROT_EXEC) {
    desiredAccess |= FILE_MAP_EXECUTE;
  }

  return desiredAccess;
}

} // namespace

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
  const DWORD protect = selectProtect(prot);
  const DWORD desired_access = selectDesiredAccess(prot);
  const DWORD file_offset_low =
    (sizeof(off_t) <= sizeof(DWORD)) ? (DWORD)off : (DWORD)(off & 0xFFFFFFFFL);
  const DWORD file_offset_high = (sizeof(off_t) <= sizeof(DWORD))
                                   ? (DWORD)0
                                   : (DWORD)(((uint64_t)off >> 32) & 0xFFFFFFFFL);
  const off_t max_size = off + (off_t)len;
  const DWORD max_size_low = (sizeof(off_t) <= sizeof(DWORD))
                               ? (DWORD)max_size
                               : (DWORD)(max_size & 0xFFFFFFFFL);
  const DWORD max_size_high = (sizeof(off_t) <= sizeof(DWORD))
                                ? (DWORD)0
                                : (DWORD)(((uint64_t)max_size >> 32) & 0xFFFFFFFFL);
  errno = 0;

  if (len == 0 || prot == PROT_EXEC) {
    errno = EINVAL;
    return MAP_FAILED;
  }

  HANDLE file_handle =
    (flags & MAP_ANONYMOUS) ? INVALID_HANDLE_VALUE : (HANDLE)_get_osfhandle(fd);

  if ((flags & MAP_ANONYMOUS) && (file_handle == INVALID_HANDLE_VALUE)) {
    errno = EBADF;
    return MAP_FAILED;
  }

  HANDLE file_mapping =
    CreateFileMapping(file_handle, 0, protect, max_size_high, max_size_low, 0);

  if (!file_mapping) {
    errno = GetLastError();
    return MAP_FAILED;
  }

  void *map_view = nullptr;

  if (flags & MAP_FIXED) {
    map_view = MapViewOfFile(file_mapping, desired_access, file_offset_high,
                             file_offset_low, len);
  } else {
    map_view = MapViewOfFileEx(file_mapping, desired_access, file_offset_high,
                               file_offset_low, len, addr);
  }

  if (!map_view) {
    errno = GetLastError();
    return MAP_FAILED;
  }

  CloseHandle(file_mapping);

  return map_view;
}

int munmap(void *addr, size_t len) {
  if (UnmapViewOfFile(addr)) {
    return 0;
  }

  return -1;
}

int msync(void *addr, size_t len, int flags) {
  if (FlushViewOfFile(addr, len)) {
    return 0;
  }

  return -1;
}

int mlock(const void *addr, size_t len) {
  if (VirtualLock((LPVOID)addr, len)) {
    return 0;
  }

  return -1;
}

int munlock(const void *addr, size_t len) {
  if (VirtualUnlock((LPVOID)addr, len)) {
    return 0;
  }

  return -1;
}

#endif // _WIN32

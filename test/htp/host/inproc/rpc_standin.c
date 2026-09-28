// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   rpc_standin.c
 * @date   27 Sep 2026
 * @brief  Host stand-in for the FastRPC runtime on both sides of the call:
 *         rpcmem and fastrpc_mmap (ARM side), HAP_mmap_get, HAP_power and
 *         FARF's sink (DSP side)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * An "ION buffer" is a page-aligned malloc with a made-up fd; the DSP's
 * HAP_mmap_get(fd) returns the same pointer, which is how the arena the
 * ARM writes is the arena the kernel DMAs from, as on the device. Every
 * symbol is exported with default visibility: HtpRpcMemApi::get() finds
 * the rpcmem entries with dlsym(RTLD_DEFAULT), exactly as it finds
 * libcdsprpc.so's on the phone.
 */

#include <AEEStdErr.h>
#include <HAP_debug.h>
#include <HAP_mem.h>
#include <HAP_power.h>
#include <remote.h>

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXPORT __attribute__((visibility("default")))

/* rpcmem.h's entries, as HtpRpcMemApi::get() casts them (the SDK header
   lives outside incs/, and nothing here needs more than these). */
EXPORT void rpcmem_init(void);
EXPORT void rpcmem_deinit(void);
EXPORT void *rpcmem_alloc(int heap, uint32_t flags, int size);
EXPORT void rpcmem_free(void *p);
EXPORT int rpcmem_to_fd(void *p);

/* ---- rpcmem: the ARM side's ION buffers ---- */
#define RPC_MAX_BUFS 4096
#define RPC_FD_BASE 100000
typedef struct {
  void *ptr;
  size_t bytes;
  int mapped; /**< fastrpc_mmap'd: only then does the DSP side see it */
} rpc_buf;
static rpc_buf g_bufs[RPC_MAX_BUFS];
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

static int buf_find(const void *p) {
  for (int i = 0; i < RPC_MAX_BUFS; ++i)
    if (g_bufs[i].ptr == p && p)
      return i;
  return -1;
}

EXPORT void rpcmem_init(void) {}
EXPORT void rpcmem_deinit(void) {}
EXPORT void *rpcmem_alloc(int heap, uint32_t flags, int size) {
  (void)heap;
  (void)flags;
  if (size <= 0)
    return NULL;
  const size_t bytes = ((size_t)size + 4095u) & ~(size_t)4095u;
  void *p = aligned_alloc(4096u, bytes);
  if (!p)
    return NULL;
  pthread_mutex_lock(&g_mu);
  int slot = buf_find(NULL);
  if (slot < 0) {
    for (slot = 0; slot < RPC_MAX_BUFS && g_bufs[slot].ptr; ++slot) {
    }
  }
  if (slot >= RPC_MAX_BUFS) {
    pthread_mutex_unlock(&g_mu);
    free(p);
    return NULL;
  }
  g_bufs[slot].ptr = p;
  g_bufs[slot].bytes = bytes;
  g_bufs[slot].mapped = 0;
  pthread_mutex_unlock(&g_mu);
  return p;
}
EXPORT void rpcmem_free(void *p) {
  if (!p)
    return;
  pthread_mutex_lock(&g_mu);
  const int slot = buf_find(p);
  if (slot >= 0)
    g_bufs[slot].ptr = NULL;
  pthread_mutex_unlock(&g_mu);
  free(p);
}
EXPORT int rpcmem_to_fd(void *p) {
  pthread_mutex_lock(&g_mu);
  const int slot = buf_find(p);
  pthread_mutex_unlock(&g_mu);
  return slot < 0 ? -1 : RPC_FD_BASE + slot;
}
static rpc_buf *buf_of_fd(int fd) {
  if (fd < RPC_FD_BASE || fd >= RPC_FD_BASE + RPC_MAX_BUFS)
    return NULL;
  return g_bufs[fd - RPC_FD_BASE].ptr ? &g_bufs[fd - RPC_FD_BASE] : NULL;
}

/** @brief [#141] The address behind a fastrpc_mmap'd fd, NULL when the fd
 *  is unknown or not mapped: dspqueue_standin.c's buffer references
 *  resolve through this, with the device's own rule. */
void *rpc_standin_mapped_ptr(int fd);
void *rpc_standin_mapped_ptr(int fd) {
  const rpc_buf *b = buf_of_fd(fd);
  return (b && b->mapped) ? b->ptr : NULL;
}

/* ---- remote.h: the session controls and the fd mapping ---- */
EXPORT int remote_session_control(uint32_t req, void *data, uint32_t len) {
  (void)req, (void)data, (void)len;
  return AEE_SUCCESS;
}
EXPORT int remote_handle64_control(remote_handle64 h, uint32_t req, void *data,
                                   uint32_t len) {
  (void)h, (void)req, (void)data, (void)len;
  return AEE_SUCCESS;
}
EXPORT int fastrpc_mmap(int domain, int fd, void *addr, int offset,
                        size_t length, enum fastrpc_map_flags flags) {
  (void)domain, (void)addr, (void)offset, (void)length, (void)flags;
  rpc_buf *b = buf_of_fd(fd);
  if (!b)
    return AEE_EBADPARM;
  b->mapped = 1;
  return AEE_SUCCESS;
}
EXPORT int fastrpc_munmap(int domain, int fd, void *addr, size_t length) {
  (void)domain, (void)addr, (void)length;
  rpc_buf *b = buf_of_fd(fd);
  if (!b)
    return AEE_EBADPARM;
  b->mapped = 0;
  return AEE_SUCCESS;
}

/* ---- HAP_mem.h: the DSP side of the same fds. An fd the ARM side has
   not fastrpc_mmap'd is refused, as the device refuses it, so a dropped
   or reordered mapping fails here and not on the phone. ---- */
EXPORT int HAP_mmap_get(int fd, void **vaddr, uint64 *paddr) {
  const rpc_buf *b = buf_of_fd(fd);
  if (!b || !b->mapped)
    return AEE_EBADPARM;
  *vaddr = b->ptr;
  if (paddr)
    *paddr = (uint64)(uintptr_t)b->ptr;
  return AEE_SUCCESS;
}
EXPORT int HAP_mmap_put(int fd) {
  return buf_of_fd(fd) ? AEE_SUCCESS : AEE_EBADPARM;
}
/* The probe entries' explicit mapping: not a model path, refused. */
EXPORT void *HAP_mmap(void *addr, int len, int prot, int flags, int fd,
                      long offset) {
  (void)addr, (void)len, (void)prot, (void)flags, (void)fd, (void)offset;
  return (void *)-1;
}
EXPORT int HAP_munmap(void *addr, int len) {
  (void)addr, (void)len;
  return AEE_EFAILED;
}

/* ---- HAP_power.h: every vote is accepted and does nothing ---- */
EXPORT int HAP_power_set(void *context, HAP_power_request_t *request) {
  (void)context, (void)request;
  return AEE_SUCCESS;
}

/* ---- HAP_debug.h: FARF's sink. Errors always reach stderr; the rest
   only with NNTR_HTP_FARF set, so a gate's output stays readable. ---- */
static int farf_all(void) {
  static int on = -1;
  if (on < 0)
    on = getenv("NNTR_HTP_FARF") != NULL;
  return on;
}
EXPORT void HAP_debug_v2(int level, const char *file, int line,
                         const char *format, ...) {
  if (level < HAP_LEVEL_ERROR && !farf_all())
    return;
  va_list ap;
  va_start(ap, format);
  fprintf(stderr, "[FARF %d %s:%d] ", level, file, line);
  vfprintf(stderr, format, ap);
  fputc('\n', stderr);
  va_end(ap);
}
EXPORT void HAP_debug(const char *msg, int level, const char *filename,
                      int line) {
  if (level < HAP_LEVEL_ERROR && !farf_all())
    return;
  fprintf(stderr, "[FARF %d %s:%d] %s\n", level, filename, line, msg);
}
EXPORT void HAP_debug_runtime(int level, const char *file, int line,
                              const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  fprintf(stderr, "[FARF %d %s:%d] ", level, file, line);
  vfprintf(stderr, format, ap);
  fputc('\n', stderr);
  va_end(ap);
}

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Haehun Yang <haehun.yang@ax.samsung.com>
 *
 * @file   htp_stub_tunnel.c
 * @date   11 Nov 2026
 * @brief  The generated FastRPC client stub with its invoke routed through
 *         nntr_htp_tunnel_remote_handle64_invoke (htp_compute_ops.cpp),
 *         which sends a method over the dspqueue when it can.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Haehun Yang <haehun.yang@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <remote.h>

int nntr_htp_tunnel_remote_handle64_invoke(remote_handle64 h, uint32_t sc,
                                           remote_arg *pra);
#define nntr_htp_tunnel_remote_handle64_open remote_handle64_open
#define nntr_htp_tunnel_remote_handle64_close remote_handle64_close
#undef __QAIC_REMOTE
#define __QAIC_REMOTE(ff) nntr_htp_tunnel_##ff
#include "nntr_hvx_stub.c"

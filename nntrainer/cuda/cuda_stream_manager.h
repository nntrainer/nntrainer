// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_stream_manager.h
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   CUDA wrapper for stream/dispatch management. Peer of
 *          nntrainer::opencl::CommandQueueManager: owns one cudaStream_t and
 *          launches kernels (cuLaunchKernel) / copies (cudaMemcpyAsync) on it.
 */

#ifndef __CUDA_STREAM_MANAGER_H__
#define __CUDA_STREAM_MANAGER_H__

#include <cstddef>

#include <cuda_runtime.h>

#include "singleton.h"

namespace nntrainer::cuda {

class Kernel;

/**
 * @class StreamManager
 * @brief Singleton owning the backend's primary CUDA stream + dispatch helpers.
 */
class StreamManager : public Singleton<StreamManager> {
public:
  /**
   * @brief Get the backend stream
   */
  cudaStream_t GetStream() const { return stream_; }

  /**
   * @brief host->device copy on the backend stream (sync unless async)
   */
  bool EnqueueWriteBuffer(void *dst_dev, size_t size, const void *src_host,
                          bool async = false);

  /**
   * @brief device->host copy on the backend stream (sync unless async)
   */
  bool EnqueueReadBuffer(const void *src_dev, size_t size, void *dst_host,
                         bool async = false);

  /**
   * @brief Launch @p kernel with the given 3D grid (blocks) and block (threads)
   *        dims and optional dynamic shared memory.
   */
  bool DispatchCommand(Kernel &kernel, const int (&grid)[3],
                       const int (&block)[3], unsigned int shared_bytes = 0);

  /**
   * @brief Block until the stream drains (cudaStreamSynchronize).
   */
  void finish();

  /**
   * @brief Conditional drain: finish() unless NNTR_CUDA_ASYNC=1. Per-op
   *        cudaStreamSynchronize is ~90% of decode wall time (it serializes
   *        CPU/GPU); once every decode op is on-GPU (no host op reads UVM
   *        mid-chain), NNTR_CUDA_ASYNC=1 turns these into no-ops so the GPU
   *        pipeline fills and only the final host read (sampling) drains once
   *        per token. Until then, default (sync) keeps coherence.
   */
  void maybeFinish();

  /**
   * @brief Inverse of maybeFinish: drain ONLY when NNTR_CUDA_ASYNC=1. Call this
   *        right before a HOST op reads GPU output on a path that stays on the
   *        host (e.g. the prefill RoPE fallback): in async mode the GPU ops did
   *        not drain, so the host read must sync first; in default mode the
   *        stream is already drained so it is a cheap no-op.
   */
  void finishIfAsync();

  /**
   * @brief Begin CUDA-graph stream capture on the backend stream (Relaxed mode,
   *        which allows the driver-API cuLaunchKernel + cuBLAS sub-launches).
   *        Drains the stream first (start from idle), then enters capture.
   * While capturing, kernel/cuBLAS calls are RECORDED into a graph rather than
   *        executed, and finish()/maybeFinish()/finishIfAsync() become no-ops
   *        (an in-capture cudaStreamSynchronize is illegal -- drains are
   * deferred to after the graph replay). Returns false if the stream is missing
   * / begin fails.
   * @note  Decode CUDA-graph foundation. Capturing the whole
   *        per-token forward additionally needs the embedding host-staging
   *        buffers (embedding_layer.cpp / tie_word_embedding.cpp `emb_stage`)
   * to be PERSISTENT + PINNED (a local std::vector is freed before the graph
   *        replays, and a pageable cudaMemcpyAsync is not capturable). TODO.
   */
  bool beginCapture();

  /**
   * @brief End stream capture; returns the captured graph in @p graph. After
   *        this, isCapturing() is false again.
   */
  bool endCapture(cudaGraph_t *graph);

  /**
   * @brief True while a capture is in progress (drains are suppressed).
   */
  bool isCapturing() const { return capturing_; }

  /**
   * @brief Destroy the stream
   */
  ~StreamManager() override;

protected:
  /**
   * @brief Singleton hook: ensure device/context then create the stream.
   */
  void initialize() noexcept override;

private:
  cudaStream_t stream_{nullptr};
  bool capturing_{false};
};

/**
 * @brief Process-lifetime device int[2] holding the per-token DECODE position:
 *        [0] = pos (== cache_index / RoPE `from`), [1] = N_kv (== pos+1). The
 *        M2-B single-capture decode graph bakes this FIXED device pointer into
 *        its RoPE / attention / KV-write kernel nodes, so cross-token replay
 * only rewrites these 8 bytes (cuda_set_pos) instead of re-recording the graph.
 *        Allocated once on first use.
 */
int *cuda_pos_buffer();

/**
 * @brief Update the per-token decode position (8-byte H2D on the backend
 * stream, issued OUTSIDE graph capture, ordered before the cudaGraphLaunch that
 *        reads it). pos == cache_index for the token; n_kv == pos+1.
 */
void cuda_set_pos(int pos, int n_kv);

} // namespace nntrainer::cuda

#endif // __CUDA_STREAM_MANAGER_H__

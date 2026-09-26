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
   * @note  Decode CUDA-graph (NNTR_CUDA_GRAPH) foundation. Capturing the whole
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
   *
   * Reads the PROCESS-wide flag, not a member: with separately-linked modules
   * the guard has to see the capture another module started (see
   * initialize()). capture_flag_ points at this object's own int until
   * initialize() re-points it at the shared state, so it is safe to read
   * before initialization.
   */
  bool isCapturing() const { return *capture_flag_ != 0; }

  /**
   * @brief Monotonic count of kernels dispatched on the backend stream.
   *
   * Producer/consumer ops that want to hand a derived buffer straight to the
   * next op (rather than recomputing it) can only do so while NOTHING ELSE
   * touched the source in between. A raw pointer equality test is not enough
   * for that -- the activation pool recycles buffers, so a later, unrelated
   * tensor can land on the very same address. Stamping the handoff with this
   * counter turns "same pointer" into "same pointer AND not one kernel ran
   * since", which the pool cannot forge.
   */
  unsigned long long dispatchSeq() const { return dispatch_seq_; }

  /**
   * @brief Destroy the stream
   */
  ~StreamManager() override;

  /**
   * @brief Get the process-wide instance (out-of-line override of
   *        Singleton<T>::Global(), intentionally-leaked heap instance --
   *        never destroyed). See cuda::ContextManager::Global() for the
   *        2026-07-20 field-crash rationale; ~StreamManager calls
   *        cudaStreamDestroy.
   */
  static StreamManager &Global();

protected:
  /**
   * @brief Singleton hook: ensure device/context then create the stream.
   */
  void initialize() noexcept override;

private:
  cudaStream_t stream_{nullptr};
  /// true only in the module that CREATED the process stream; the adopters
  /// borrowed the handle and must not destroy it (see ~StreamManager).
  bool owns_stream_{false};
  /// Fallback storage used until initialize() points capture_flag_ at the
  /// process-wide SharedCudaState::capturing.
  int own_capture_flag_{0};
  int *capture_flag_{&own_capture_flag_};
  unsigned long long dispatch_seq_{0};
};

/**
 * @brief Process-lifetime device int[2] holding the per-token DECODE position:
 *        [0] = pos (== cache_index / RoPE `from`), [1] = N_kv (== pos+1). The
 *        the single-capture decode graph bakes this FIXED device pointer into
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

/**
 * @brief Record that a host-side dispatch decision compared the live KV length
 *        against @p threshold, and answer that comparison (@p kv_len >
 *        @p threshold).
 *
 * A capture freezes every host decision taken while it recorded: the kernel a
 * dispatch selected, the grid it launched, the arm an `if` chose. Kernels can
 * be taught to read the moving values from cuda_pos_buffer(), but a host branch
 * cannot -- it already happened. So the branches are made REPORTABLE instead:
 * every site that compares the key count against a fixed bound routes the
 * comparison through here, and the capture keeps the answers. Before each
 * replay, kv_regime_holds() re-asks them at the current key count; when one has
 * flipped, the graph is retired and recaptured against the arm that is now
 * correct (nntrainer::CudaContext::runDecode).
 *
 * What this catches, measured on a discrete part: the split-KV decode engages
 * at `N_kv > NNTR_CUDA_FLASH_DECODE` (default 64 keys). A 46-token prompt
 * captures at ~48 keys -- below it -- so the replay keeps launching the dense
 * per-key kernel while the eager path crosses into split-KV at absolute
 * position 64, and the two reduce in different orders: the answer stays fluent
 * and diverges from the golden at exactly that token. Forcing either arm for
 * the whole run makes replay byte-identical to eager again, which is what named
 * the arm flip as the cause.
 *
 * Only comparisons taken while recording (i.e. inside the decode capture) are
 * kept, so a call on a prefill-only path costs one comparison and nothing else.
 * What is stored is (kv_at_site - threshold), not the two values: every key
 * count in a decode step advances by exactly one per token, so one delta
 * re-asks every recorded question regardless of which count each site saw.
 */
bool kv_regime_gt(int kv_len, int threshold);

/**
 * @brief Record that a decision consumed the live KV length as a VALUE rather
 *        than through a bound -- a copy whose byte count is the key count, say.
 *        No later key count can satisfy it, so the capture is valid for this
 *        step only and every following step recaptures.
 */
void kv_regime_exact(int kv_len);

/**
 * @brief Clear the table and start recording, with @p kv_len as the key count
 *        the recorded answers belong to. Called immediately before a capture.
 */
void kv_regime_begin(int kv_len);

/**
 * @brief Stop recording. The table now describes the captured graph.
 */
void kv_regime_seal();

/**
 * @brief True while every recorded comparison still answers the same way at
 *        @p kv_len -- i.e. while the captured graph's host decisions are still
 *        the ones this step would take. An empty table holds (nothing recorded,
 *        nothing to invalidate).
 */
bool kv_regime_holds(int kv_len);

/**
 * @brief Number of comparisons in the table (diagnostics only).
 */
int kv_regime_size();

} // namespace nntrainer::cuda

#endif // __CUDA_STREAM_MANAGER_H__

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_emb_gather.h
 * @date    19 Aug 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   On-GPU embedding-LUT gather+dequant for the decode path.
 *
 * A sidecar embedding LUT (packed signed-4-bit rows + fp32 scales) lives in
 * host memory: the payload is a read-only file mmap, the scales a heap
 * vector. Per decode token the host used to dequantize the token's row(s)
 * (embd + folded-PLE) and push them H2D into the device-only activation
 * pool. These helpers move that lookup+dequant onto the GPU, inside the M2-B
 * decode graph, with ZERO extra copies of the tables:
 *
 *  - The GPU reads the LUT bytes IN PLACE over HMM pageable access
 *    (cudaDevAttrPageableMemoryAccess, open kernel module). cudaHostRegister
 *    is NOT used: registering a file-backed PROT_READ mmap is rejected by
 *    the driver (probed: "operation not supported" / "invalid argument"
 *    even with cudaHostRegisterReadOnly), and pinning would lock the whole
 *    table resident anyway. HMM maps exactly the pages a row touches --
 *    the same pages the host dequant faulted in before, so residency is
 *    unchanged.
 *  - The token id reaches the captured graph through a process-lifetime
 *    device int fed by a captured 4-byte H2D from a pinned host slot (the
 *    cuda_set_pos pattern): the per-token feed is ONE host store. The id is
 *    NOT taken from the on-GPU argmax scratch: the sampler legitimately
 *    falls back to the host (do_sample, fp32 logits, over-cap penalty
 *    windows), which would leave that scratch stale -- the pinned slot is
 *    correct on every path, and costs nothing extra since the 4-byte D2H
 *    for EOS/detok already syncs each token.
 *  - First GPU access to a row's pages pays an HMM mapping fault (measured
 *    ~0.1-0.7 ms). emb_gather_notify_token() -- called where the argmax D2H
 *    first learns the next id -- warms those pages via a small
 *    cudaMemPrefetchAsync on a side stream, overlapping the host's
 *    detok/EOS/feed window; a per-row bitmap makes it once-per-id.
 *
 * All device state is allocated at registration time (NEVER under graph
 * capture); dispatch is capture-safe (one H2D + one kernel on the backend
 * stream). Unsupported platforms (no HMM: Windows/WDDM, older kernels)
 * simply fail registration and the host dequant+staging path is kept.
 */

#ifndef __CUDA_EMB_GATHER_H__
#define __CUDA_EMB_GATHER_H__

#include <cstddef>

namespace nntrainer::cuda {

/**
 * @brief Register one sidecar LUT for on-GPU gather. Compiles the kernels,
 *        allocates the shared id slot, advises the payload/scale ranges for
 *        device access and sets up the prefetch bitmap. Call OUTSIDE graph
 *        capture (returns -1 under capture; retry later).
 * @param payload  packed 4-bit row table (host pointer; file mmap ok)
 * @param payload_bytes payload size in bytes
 * @param scales   per-(row,block) fp32 scales, scale_count entries
 * @param n_rows   rows (vocab)
 * @param out_dim  columns per row (nibbles)
 * @param nblocks  scale blocks per row (1 = one scale per row)
 * @return handle >= 0 on success, -1 when unavailable (no HMM, env off,
 *         geometry mismatch, allocation/compile failure)
 */
int emb_gather_register_lut(const void *payload, size_t payload_bytes,
                            const float *scales, size_t scale_count,
                            unsigned n_rows, unsigned out_dim,
                            unsigned nblocks);

/**
 * @brief Per-token feed: publish the token id to the pinned slot the
 *        captured 4-byte H2D reads on every replay, and (safety net for
 *        host-sampled ids) warm the row's pages if still cold. One host
 *        store on the hot path.
 */
void emb_gather_set_token(int handle, int tok);

/**
 * @brief Prefill-time page warm: batch-fault the GPU mappings for a chunk's
 *        token rows on the side stream (deduplicated by the same per-row
 *        bitmap the notify hook uses). The prompt vocabulary dominates the
 *        decode vocabulary, so warming it during prefill removes most of the
 *        decode-time HMM cold faults. The pages are the very ones the host
 *        prefill dequant touches anyway, so residency does not change.
 *        Best-effort: skipped under capture or while a previous warm batch
 *        is still in flight.
 * @param ids_host  the prefill chunk's token ids as the input tensor's floats
 */
void emb_gather_warm_ids(int handle, const float *ids_host, unsigned n);

/**
 * @brief Early page-warm hook: called right after the on-GPU argmax D2H
 *        learns the next token id, so the HMM prefetch of that id's rows
 *        overlaps the host detok/EOS/feed window. No-op with no LUTs
 *        registered, under capture, or when the id's pages are warm.
 */
void emb_gather_notify_token(unsigned tok);

/**
 * @brief Enqueue the id H2D + gather kernel on the backend stream (capture-
 *        safe; captured into the decode graph when recording). Writes the
 *        dequantized row where the host staging H2D used to land.
 * @param fp16_out true: fp16 row (host _FP16 cast parity), false: fp32
 */
bool emb_gather_dispatch_s4(int handle, float layer_scale, void *out,
                            bool fp16_out);

/**
 * @brief M2-B cached-graph lifecycle: true while a captured decode graph is
 *        live (its replay performs the gathers, so the per-token feed must
 *        NOT re-dispatch them eagerly); false drops it (and bumps the epoch
 *        below) when the cached exec is destroyed or capture fails.
 */
void emb_gather_set_graph_live(bool live);
bool emb_gather_graph_live();

/**
 * @brief Capture epoch: incremented whenever a cached graph is dropped. A
 *        layer records the epoch at capture-time dispatch and skips its feed
 *        work only while (graph_live && epoch unchanged) -- so a graph that
 *        was captured WITHOUT the gather (dispatch refused mid-capture)
 *        keeps the host staging refresh it depends on.
 */
unsigned emb_gather_epoch();

} // namespace nntrainer::cuda

#endif // __CUDA_EMB_GATHER_H__

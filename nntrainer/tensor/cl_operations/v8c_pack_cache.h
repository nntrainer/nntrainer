// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   v8c_pack_cache.h
 * @date   31 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Derive-once disk cache for the v8c GEMM weight pack.
 *
 * The v8c weight build (make_v8c_weight_backing_from_qs4cx) permutes the
 * plain QS4CX nibbles into the KAI-order byte layout and folds the
 * per-channel int4 row sums -- a deterministic, pure function of the weight
 * bytes. For the biggest weight (an untied lm-head class N) that permute is
 * the longest single-node span of the whole model load. This cache persists
 * the packed payload + row sums next to the model .bin
 * (<model.bin>.v8cpack; falls back to $XDG_CACHE_HOME/nntrainer/v8cpack
 * when the model dir is not writable), so later launches mmap the pack and
 * skip the permute entirely -- the device upload still happens, from the
 * mapped pages, with no staging copy.
 *
 * Identity/keying (hard rule -- never pointer-keyed): the pack file is bound
 * to the source .bin by (size, mtime-ns, format version); each record is
 * keyed by (weight name FNV-1a, N, K, row_bytes, payload length) and guarded
 * by a sampled payload FNV (both 64 KB ends + 16 interior pages) plus a full
 * FNV over the row-sum block. Any mismatch is a silent per-record miss
 * (derive as before); a stale/absent/corrupt header invalidates the whole
 * file and the first launch rewrites it (temp file + fsync + atomic rename,
 * finalized on a background thread that is exit-joined). NNTR_V8C_PACK_CACHE=0
 * opts out; NNTR_V8C_PACK_CACHE_MIN_MB (default 64) bounds which weights are
 * cached so the disk cost stays at the few giant packs unless asked for.
 * POSIX-only (no-op stubs elsewhere).
 */

#ifndef __V8C_PACK_CACHE_H__
#define __V8C_PACK_CACHE_H__

#include <cstddef>
#include <cstdint>

namespace nntrainer {
namespace v8c_pack {

/**
 * @brief A validated cache hit: pointers into the pack file mmap. Valid until
 *        the next set_source() (i.e. for the whole model lifetime in the
 *        one-model-per-process apps).
 */
struct Hit {
  const uint8_t *payload = nullptr; /**< packed v8c bytes, payload_len long */
  const int32_t *rowsum = nullptr;  /**< per-channel int4 row sums, N entries */
  size_t payload_len = 0;
};

/**
 * @brief Opaque per-record writer handle (miss path tee). Chunk writes go to
 *        disjoint offsets, so concurrent loader workers may write their own
 *        records in parallel.
 */
struct RecordWriter;

/**
 * @brief Bind the cache to a source weight file. Validates/maps an existing
 *        pack for it; a stale or corrupt pack arms rewrite mode. Safe to call
 *        again (model switch): joins any in-flight finalize first.
 */
void set_source(const char *model_bin_path);

/**
 * @brief All load-time builds are done: finalize a pending rewrite (index +
 *        header + fsync + rename) on a background thread. The thread is
 *        joined at static destruction (exit) so one-shot CLI runs land it.
 */
void load_complete();

/**
 * @brief Look up a record. Returns true and fills @p out only when every key
 *        field and both checksums match. Thread-safe (immutable after
 *        set_source).
 */
bool lookup(const char *name, unsigned int N, unsigned int K, size_t row_bytes,
            size_t payload_len, uint64_t src_fnv, Hit &out);

/**
 * @brief Fingerprint of the SOURCE bytes a record is derived from, to be
 *        passed to lookup() and begin_record(). The file-level identity
 *        (size + mtime) cannot see a weight file replaced in place with one
 *        of the same size and timestamp -- a restored archive, a regenerated
 *        build artefact, or any filesystem with a coarse mtime -- and serving
 *        a stale pack there would infer with silently wrong weights. Sampled
 *        (both ends plus interior pages), so it costs nothing on a hit.
 */
uint64_t source_fingerprint(const void *data, size_t len);

/**
 * @brief Drop the (clean, file-backed) payload pages of a consumed hit.
 */
void payload_consumed(const Hit &hit);

/**
 * @brief Start teeing a record derive to the pack temp file. Returns nullptr
 *        when the cache is off, the weight is below the size floor, or a
 *        valid pack already exists (partial rewrites never clobber a good
 *        pack). Must be paired with commit_record or abort_record.
 */
RecordWriter *begin_record(const char *name, unsigned int N, unsigned int K,
                           size_t row_bytes, size_t payload_len,
                           uint64_t src_fnv);

/**
 * @brief Tee one packed chunk at @p payload_off (bytes into this record's
 *        payload region).
 */
void record_write(RecordWriter *rw, size_t payload_off, const void *data,
                  size_t len);

/**
 * @brief Payload fully written: append the row sums, checksum, and index the
 *        record.
 */
void commit_record(RecordWriter *rw, const int32_t *rowsum, size_t count);

/**
 * @brief Derive failed mid-way: forget the record (its blob region is simply
 *        left unreferenced by the index).
 */
void abort_record(RecordWriter *rw);

} // namespace v8c_pack
} // namespace nntrainer

#endif /* __V8C_PACK_CACHE_H__ */

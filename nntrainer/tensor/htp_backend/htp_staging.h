// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_staging.h
 * @brief Owns HTP staging buffers and the validity of retained host views.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items
 */
#ifndef NNTR_HTP_STAGING_H
#define NNTR_HTP_STAGING_H
#include "htp_runtime_config.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <htp_router_handoff.h>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nntrainer {
/** Internal ownership component; the session invocation mutex serializes
 * access. Buffer and Transfer bind the production ION/copy policy without
 * virtual calls.
 */
template <typename Buffer, typename Transfer> class HtpStaging {
public:
  /** @brief Grows @a buf to at least @a bytes, reusing it otherwise.
   *
   *  One buffer for activation, one for output, reused across every call
   *  instead of one rpcmem alloc/free per matmul -- the buffers this
   *  replaces (the tensor pool's plain heap pointers, passed straight
   *  through) are pinned and mapped by the FastRPC driver on EVERY call
   *  (htp_rpcmem.h's own doc comment; measured at ~155 MB/s vs ION's
   *  44.6 GB/s, docs/htp_attention/34_fc_measured.md section4 item F). The
   *  memcpy this adds is the trade: cheap relative to a per-call pin+map,
   *  but that trade is exactly what htp.profile's transport column is
   *  for confirming on device -- do not assume the win without it. */
  static void ensureCapacity(std::unique_ptr<Buffer> &buf, size_t bytes) {
    if (!buf || buf->size() < bytes) {
      buf = std::make_unique<Buffer>(bytes);
    }
  }

  /** @brief ION scratch by size class, one buffer per power of two from
   *  64 KiB up, so a call stages through a buffer near its own size.
   *
   * One pair grown to the largest shape (ensureCapacity) made every call
   * pay for that shape: the driver's cache maintenance on a cached ION
   * buffer covers the whole dma-buf, not the bytes passed, so the MoE
   * decode call's 8 KB rode a 3.6 MB pair at 553 us of transport, a
   * 10.9 MB out buffer at 814 and a 12.7 MB pair at 1633 (doc 50 section
   * 3.6: 36-59 us/MB, cache-flush speed), and prefill's calls paid the
   * same. With classes, decode stays on the 64 KiB pair. */
  /** @brief The host range a staging buffer stands for. kept: its call
   *  left only sampledSame's windows on the host (copyOutOf's to_host
   *  false), so a copy of any part of it must come from the buffer
   *  (trueSource); a model tensor, alive as long as the model. */
  struct Held {
    const void *first;
    size_t second;
    bool kept = false;
  };
  struct StagingPool {
    std::map<size_t, std::unique_ptr<Buffer>> by_class;
    /** What stageInput last copied into a class's buffer, cleared by any
     *  other stage() of it (doc 57 section 9.31). */
    std::map<size_t, Held> holds;
    /** The host tensor a class's buffer was last copied out to, whole
     *  (copyOutOf), cleared by any other stage() of it. */
    std::map<size_t, Held> copied_out;
  };
  static size_t stageClass(size_t bytes) {
    size_t cls = size_t(64) << 10;
    while (cls < bytes)
      cls <<= 1;
    return cls;
  }
  /** A writable output may first restore a kept host tensor. Reject the
   * handoff if that restore could overwrite its actual ION input bytes. */
  static bool stagingRestoreOverlaps(const StagingPool &pool, size_t bytes,
                                     const void *input, size_t input_bytes) {
    const size_t cls = stageClass(bytes);
    for (const auto *m : {&pool.holds, &pool.copied_out}) {
      auto it = m->find(cls);
      if (it == m->end() || !it->second.kept)
        continue;
      if (htp_router_handoff_overlap(input, input_bytes, it->second.first,
                                     it->second.second))
        return true;
    }
    return false;
  }
  /** @brief @a slot made large enough for @a bytes. The class only names
   *  the slot: the buffer is as large as the largest tensor it has held
   *  (plus a page), not the class, because the driver's cache maintenance
   *  covers the whole dma-buf on every call it rides (11.5 MB of rows rode
   *  16 MiB, 23 MB rode 32 MiB). Growing drops the old buffer, so the
   *  callers do it only where the old bytes are no longer wanted. */
  void fit(std::unique_ptr<Buffer> &slot, size_t bytes) {
    if (slot && slot->size() >= bytes)
      return;
    htp_router_handoff_invalidate(&router_handoff_);
    if (slot) {
      const uint8_t *lo = slot->data(), *hi = lo + slot->size();
      for (IonAlias &a : ion_alias_)
        if (a.ion >= static_cast<const void *>(lo) &&
            a.ion < static_cast<const void *>(hi))
          a = IonAlias{};
    }
    slot = std::make_unique<Buffer>(
      ((bytes + 4095) & ~static_cast<size_t>(4095)) + 4096);
  }
  Buffer &stage(StagingPool &pool, size_t bytes) {
    // Handing out a writable buffer ends any pending router ownership.
    htp_router_handoff_invalidate(&router_handoff_);
    const size_t cls = stageClass(bytes);
    auto &slot = pool.by_class[cls];
    // A tensor a call kept in this buffer (copyOutOf's to_host false) has
    // only its windows on the host; once the buffer is reused nobody can
    // find its bytes any more (trueSource / ionAlias see the new bytes),
    // and a later reader of the host tensor would take the stale rows. So
    // the host tensor is made whole first -- when its windows still match
    // the buffer, i.e. the host has not rewritten it since. Only a step
    // whose sizes collide in one class pays (a 441-row chunk after 2048-row
    // ones: 2048 rows put every tensor in its own class).
    for (auto *m : {&pool.copied_out, &pool.holds}) {
      auto it = m->find(cls);
      if (it != m->end() && it->second.kept && slot &&
          sampledSame(it->second.first, slot->data(), it->second.second))
        Transfer::copy(const_cast<void *>(it->second.first), slot->data(),
                       it->second.second);
    }
    pool.holds.erase(cls);
    pool.copied_out.erase(cls);
    fit(slot, bytes);
    return *slot;
  }
  /** @brief Whether @a b still holds the copy of @a a: 32 spread 64-byte
   *  windows and the tail compared. a rewrite that keeps every
   *  sampled window is missed; a full compare costs about what the copy
   *  does. */
  static bool sampledSame(const void *a_, const void *b_, size_t bytes) {
    const uint8_t *a = static_cast<const uint8_t *>(a_);
    const uint8_t *b = static_cast<const uint8_t *>(b_);
    bool same =
      bytes < 64 || std::memcmp(a + bytes - 64, b + bytes - 64, 64) == 0;
    for (size_t i = 0; same && i < 32 && bytes >= 64; ++i) {
      const size_t off = (bytes - 64) / 32 * i;
      same = std::memcmp(a + off, b + off, 64) == 0;
    }
    return same;
  }
  /** @brief stage() plus the copy of @a bytes from @a src, skipped when the
   *  buffer still holds exactly that copy: the router, dense and MoE calls
   *  of a Gemma-4 layer each stage the same 11.5 MB post-attention rows.
   *  Any other stage() of the buffer forgets the copy, and before skipping
   *  the copy is compared (sampledSame) with @a src, so a host tensor
   *  rewritten (or another one placed at the same address) is copied again.
   *
   *  With @a from, a buffer of that pool that @a src was copied out of
   *  whole (copyOutOf) and still matches is swapped in instead of copied:
   *  the dense output is the MoE call's x2 and the MoE output the next
   *  qkv call's input (doc 57 section 9.34). Stage this before the call's
   *  own output buffer, which may be the one swapped in. */
  Buffer &stageInput(StagingPool &pool, const void *src, size_t bytes,
                     StagingPool *from = nullptr, int line = __builtin_LINE(),
                     bool parallel = false) {
    const size_t cls = stageClass(bytes);
    auto &slot = pool.by_class[cls];
    auto it = pool.holds.find(cls);
    if (it != pool.holds.end() && slot && it->second.first == src &&
        it->second.second == bytes && sampledSame(src, slot->data(), bytes))
      return *slot;
    // No-write hits above keep their ION epoch. Copies/swaps below do not.
    htp_router_handoff_invalidate(&router_handoff_);
    for (StagingPool *source_pool : {from, &attn_out_pool_}) {
      if (!source_pool || source_pool == &pool)
        continue;
      // what from's buffer of this size is: a call's output (copied_out),
      // or an input staged there (holds: the qkv call's activation rows are
      // the o-proj call's residual)
      auto c = source_pool->copied_out.find(cls);
      const Held *was =
        c != source_pool->copied_out.end() ? &c->second : nullptr;
      if (!was) {
        auto h = source_pool->holds.find(cls);
        was = h != source_pool->holds.end() ? &h->second : nullptr;
      }
      auto &fslot = source_pool->by_class[cls];
      if (was && was->first == src && was->second == bytes && fslot &&
          sampledSame(src, fslot->data(), bytes)) {
        const bool kept = was->kept;
        // the buffer handed to from: a kept tensor of pool's class goes
        // with it. In a layer's own order that tensor has had its last
        // reader (the previous dense output, the layer input, the o-proj
        // output: 3 a layer, 11.5 MB each at 1024 rows), so it is not
        // copied to the host; it is remembered as lost, and a later read of
        // it throws instead of taking the windows-only host bytes.
        for (auto *m : {&pool.copied_out, &pool.holds}) {
          auto k = m->find(cls);
          if (k != m->end() && k->second.kept && slot &&
              k->second.first != src &&
              sampledSame(k->second.first, slot->data(), k->second.second))
            noteLost(k->second.first, k->second.second);
        }
        std::swap(slot, fslot);
        source_pool->holds.erase(cls);
        source_pool->copied_out.erase(cls);
        pool.copied_out.erase(cls);
        pool.holds[cls] = {src, bytes, kept};
        return *slot;
      }
    }
    // A copy after all (a row chunk of a tensor, a size the pools hold no
    // buffer of): from where the bytes really are -- a tensor a call kept
    // (copyOutOf's to_host false) has only its windows on the host.
    const void *from_buf = trueSource(src, bytes);
    const bool stage_debug = htpRuntimeConfig().htp.stage_debug;
    if (stage_debug && from_buf == src) {
      // a host range that a kept record covers, whose buffer no longer
      // matches: the host bytes are the stale windows-only copy
      for (StagingPool *pl : {&out_pool_, &act_pool_, &x2_pool_})
        for (auto *m : {&pl->copied_out, &pl->holds})
          for (const auto &rec : *m) {
            const char *h = static_cast<const char *>(rec.second.first);
            const char *sc = static_cast<const char *>(src);
            if (rec.second.kept && h && sc >= h &&
                sc + bytes <= h + rec.second.second)
              Transfer::warn(
                "HTP stage: STALE host read of %zu B at line %d (kept "
                "record pool=%p cls=%zu bytes=%zu)",
                bytes, line, static_cast<void *>(pl), rec.first,
                rec.second.second);
          }
    }
    if (from_buf == src)
      throwIfLost(src, bytes, line);
    // a source inside this very buffer means it is large enough already
    fit(slot, bytes);
    if (from_buf != slot->data() &&
        (!parallel ||
         !Transfer::parallelCopy(slot->data(), from_buf, bytes, line)))
      Transfer::copy(slot->data(), from_buf, bytes, line);
    pool.holds[cls] = {src, bytes, false};
    return *slot;
  }
  /** @brief A kept host tensor whose staging buffer was handed on
   *  (stageInput's swap) without the copy to the host: its range and the
   *  windows sampledSame compares, as they were. */
  struct Lost {
    const void *host = nullptr;
    size_t bytes = 0;
    std::array<uint8_t, 33 * 64> win{};
  };
  std::array<Lost, 8> lost_{};
  size_t lost_next_ = 0;
  static void sampledPack(uint8_t *win, const void *src_, size_t bytes) {
    const uint8_t *s = static_cast<const uint8_t *>(src_);
    std::memcpy(win, s + bytes - 64, 64);
    for (size_t i = 0; i < 32; ++i)
      std::memcpy(win + 64 * (i + 1), s + (bytes - 64) / 32 * i, 64);
  }
  void noteLost(const void *host, size_t bytes) {
    if (bytes < 64)
      return;
    Lost &l = lost_[lost_next_++ % lost_.size()];
    l.host = host;
    l.bytes = bytes;
    sampledPack(l.win.data(), host, bytes);
  }
  /** @brief Throws when [src, src + bytes) is inside a lost tensor the host
   *  has not rewritten since: those host bytes are only the windows.
   *  the last 8 lost tensors are remembered (3 are lost a layer);
   *  a reader further behind than that is not caught. */
  void throwIfLost(const void *src, size_t bytes, int line) const {
    const char *s = static_cast<const char *>(src);
    for (const Lost &l : lost_) {
      const char *h = static_cast<const char *>(l.host);
      if (!h || s < h || s + bytes > h + l.bytes)
        continue;
      std::array<uint8_t, 33 * 64> now;
      sampledPack(now.data(), l.host, l.bytes);
      if (now == l.win)
        throw std::runtime_error(
          "HTP stage: host read (line " + std::to_string(line) +
          ") of a kept tensor whose staging buffer was reused; "
          "htp.keep_out=0 copies every output to the host");
    }
  }
  /** @brief Where [src, src + bytes) really is: inside a staging buffer
   *  that holds (by sampledSame over its whole tensor) a kept host tensor
   *  containing the range, else @a src itself. Only kept ones: any other
   *  record's host range may be memory freed since. */
  const void *trueSource(const void *src, size_t bytes) {
    const char *s = static_cast<const char *>(src);
    for (StagingPool *pool : {&out_pool_, &act_pool_, &x2_pool_}) {
      for (auto *m : {&pool->copied_out, &pool->holds}) {
        for (const auto &rec : *m) {
          const char *h = static_cast<const char *>(rec.second.first);
          const size_t hb = rec.second.second;
          auto slot = pool->by_class.find(rec.first);
          if (!rec.second.kept || !h || s < h || s + bytes > h + hb ||
              slot == pool->by_class.end() || !slot->second)
            continue;
          if (sampledSame(h, slot->second->data(), hb))
            return slot->second->data() + (s - h);
        }
      }
    }
    return src;
  }
  /** @brief The windows sampledSame compares, copied: what makes a host
   *  tensor that was NOT copied out still recognizable as this buffer's. */
  static void sampledCopy(void *dst_, const void *src_, size_t bytes) {
    uint8_t *d = static_cast<uint8_t *>(dst_);
    const uint8_t *s = static_cast<const uint8_t *>(src_);
    if (bytes < 64) {
      std::memcpy(d, s, bytes);
      return;
    }
    std::memcpy(d + bytes - 64, s + bytes - 64, 64);
    for (size_t i = 0; i < 32; ++i) {
      const size_t off = (bytes - 64) / 32 * i;
      std::memcpy(d + off, s + off, 64);
    }
  }
  /** @brief The copy of a call's whole output out of @a pool's buffer for
   *  @a bytes, remembered for stageInput's @a from.
   *
   *  @a to_host false: the layer says only accelerator calls read this
   *  output at prefill, so only sampledSame's windows go to the host (2 KB
   *  of an 11.5 MB tensor): the calls that take it find the buffer by the
   *  same content check as ever and swap it in, and a different tensor a
   *  CPU layer writes at that address still fails the check and is copied.
   *  A host READER of this tensor sees stale bytes between the windows: the
   *  builder sets the flag only where every consumer runs here, the layers'
   *  CPU fallbacks call sync_to_host first, and htp.profile >= 2 (its
   *  host norms read these tensors) copies in full. */
  void copyOutOf(StagingPool &pool, void *dst, size_t bytes,
                 bool to_host = true, int line = __builtin_LINE()) {
    const size_t cls = stageClass(bytes);
    const void *buf = pool.by_class[cls]->data();
    forgetIonAliases(dst, buf, bytes);
    // htp.keep_out=0: every output copied to the host whole (a
    // reference for the kept path; never set in a tok/s run)
    const bool keep_off = !htpRuntimeConfig().htp.keep_out;
    bool kept = !to_host && !keep_off && Transfer::profileLevel() < 2;
    if (kept) {
      sampledCopy(dst, buf, bytes);
      // the windows must be what sampledSame compares, or the readers
      // would take the stale host tensor: checked, the full copy otherwise
      kept = sampledSame(dst, buf, bytes);
    }
    if (!kept)
      Transfer::copy(dst, buf, bytes, line);
    pool.copied_out[cls] = {dst, bytes, kept};
  }

  /** @brief A host range last copied out of a staging buffer, so a later
   *  call can hand the DSP that buffer's bytes in place of a fresh copy:
   *  the qkv call's Q block is the attention's Q, the attention's output
   *  the o-proj's input. The staging buffers live as long as the ops, so
   *  the pointer stays valid; whether its bytes still are the host's is
   *  sampledSame's to say at use. */
  struct IonAlias {
    const void *host = nullptr;
    const void *ion = nullptr;
    size_t bytes = 0;
    bool host_complete = false;
  };
  std::array<IonAlias, 16> ion_alias_{};
  size_t ion_alias_next_ = 0;
  void forgetIonAliases(const void *host, const void *ion, size_t bytes) {
    const auto overlaps = [](const void *a, size_t an, const void *b,
                             size_t bn) {
      const uintptr_t x = reinterpret_cast<uintptr_t>(a);
      const uintptr_t y = reinterpret_cast<uintptr_t>(b);
      return x <= y ? y - x < an : x - y < bn;
    };
    // Materialization belongs to one output generation. Reusing its host
    // or staging storage must not inherit an older complete-copy record.
    for (auto &a : ion_alias_)
      if (a.host && (overlaps(host, bytes, a.host, a.bytes) ||
                     overlaps(ion, bytes, a.ion, a.bytes)))
        a = {};
  }
  void noteIonAlias(const void *host, const void *ion, size_t bytes,
                    bool host_complete) {
    forgetIonAliases(host, ion, bytes);
    ion_alias_[ion_alias_next_++ % ion_alias_.size()] = {host, ion, bytes,
                                                         host_complete};
  }
  /** @brief The staging bytes that hold [host, host + bytes), or null. */
  const void *ionAlias(const void *host, size_t bytes,
                       bool *host_complete = nullptr) const {
    const char *h = static_cast<const char *>(host);
    for (const IonAlias &a : ion_alias_) {
      const char *b = static_cast<const char *>(a.host);
      if (!b || h < b || h + bytes > b + a.bytes)
        continue;
      const void *ion = static_cast<const char *>(a.ion) + (h - b);
      if (sampledSame(host, ion, bytes)) {
        if (host_complete)
          *host_complete = a.host_complete;
        return ion;
      }
    }
    return nullptr;
  }

  /** @brief The attention's u16 context left in its staging buffer for the
   *  o-proj (sdpa_q2_kvcache_f32 -> gemm_qs4cx_res_add_fp32): the host
   *  tensor it stands for, the buffer, its shape and the per-head
   *  encodings. One at a time: the graph runs a layer's attention, then its
   *  o-proj. a CPU consumer of that f32 tensor, or another
   *  attention call before the o-proj, is not covered -- the builder sets
   *  mha_core's attn_out_u16 only for the folded o-proj on this engine. */
  struct AttnOut16 {
    const float *host = nullptr;
    const uint16_t *ion = nullptr;
    unsigned int rows = 0, width = 0, head_dim = 0;
    std::vector<float> enc;
  } attn_out16_;
  htp_router_handoff router_handoff_{};
  StagingPool act_pool_;
  StagingPool out_pool_;
  StagingPool rope_pool_;     /**< the fused projection call's RoPE rows */
  StagingPool x2_pool_;       /**< the MoE call's post-FFN second addend */
  StagingPool attn_q_pool_;   /**< the row-blocked attention's f32 Q */
  StagingPool attn_out_pool_; /**< and its f32 output */
  std::unique_ptr<Buffer> act_ah_buf_;
  std::vector<float> act_scale_scratch_;
  std::vector<int32_t> act_zp_scratch_;

  /** Release prefill/seed temporaries between completed synchronous calls.
   * The public phase-boundary hook publishes kept outputs first. The token
   * driver calls here after the seed's last reader has completed. */
  void releasePrefillStaging() {
    htp_router_handoff_invalidate(&router_handoff_);
    constexpr size_t keep_bytes = size_t(64) << 10;
    size_t freed = 0;
    for (StagingPool *pool : {&act_pool_, &out_pool_, &rope_pool_, &x2_pool_,
                              &attn_q_pool_, &attn_out_pool_}) {
      for (auto it = pool->by_class.upper_bound(keep_bytes);
           it != pool->by_class.end();) {
        freed += it->second ? it->second->size() : 0;
        it = pool->by_class.erase(it);
      }
      pool->holds.clear();
      pool->copied_out.clear();
    }
    // These aliases can point inside the allocations just returned above.
    ion_alias_ = {};
    ion_alias_next_ = 0;
    lost_ = {};
    lost_next_ = 0;
    attn_out16_ = {};
    if (act_ah_buf_ && act_ah_buf_->size() > keep_bytes) {
      freed += act_ah_buf_->size();
      act_ah_buf_.reset();
      std::vector<float>().swap(act_scale_scratch_);
      std::vector<int32_t>().swap(act_zp_scratch_);
    }
    if (freed)
      std::fprintf(stderr, "[HTP-MEM] released_prefill_scratch_bytes=%zu\n",
                   freed);
  }
};
} // namespace nntrainer
#endif

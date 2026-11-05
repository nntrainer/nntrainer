// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   expert_fsu_flags.h
 * @date   10 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  The streamed-expert decode step without a host round trip
 *         (NNTR_G4MOE_FSU_FLAGS): a miss ring and ready flags in
 *         fine-grain SVM between the GPU and the expert reader threads.
 *
 * @details Without it, a streamed MoE layer's decode step goes
 *
 *   select kernel -> marker callback -> host reads the selected ids back ->
 *   host acquire (victims) -> reads -> slot writes -> offset tables ->
 *   host waits / user event -> expert GEMVs
 *
 * and every arrow is a host <-> device hand-off. With it:
 *
 *  1. Before a layer's step (and before the previous layer reads ahead for
 *     it) the host writes the layer's slot table and a victim queue into
 *     the layer's block of fine-grain SVM (prepare()): expert -> slot, slot
 *     -> expert, each slot's content tag, and the slots in the order the
 *     residency policy would evict them (ExpertResidency::victimOrder).
 *  2. The device routes the step and, in a one-lane publish kernel, finds
 *     the misses, gives each the first queued slot whose expert the step
 *     does not need, updates the slot table, writes each selected expert's
 *     slot offsets for the GEMVs, and publishes one ring entry -- the
 *     selected experts with their slots, tags and a "read" mark for the
 *     misses -- by storing the entry's sequence number last (release,
 *     all-SVM-devices scope). A read ahead for the next layer (the step's
 *     prediction of it) is published the same way as a speculative entry
 *     placed from that layer's queue (at most until `keep` queued slots
 *     remain, so the layer's own step always finds room).
 *  3. A host poller thread takes the entries in sequence order (acquire),
 *     hands every marked read to the reader threads -- straight into its
 *     slot, O_DIRECT -- and returns the ring slot (the tail). A reader that
 *     finished stores the slot's ready word = the entry's tag (release); a
 *     failed read sets the failed word instead.
 *  4. A one-work-group gate kernel ahead of the expert GEMVs spins until
 *     every selected slot's ready word equals its tag (acquire), or the
 *     failed / error word is set, or its spin budget runs out; on failure it
 *     writes the error word and makes the failed lane's routing (combine)
 *     weight wts[a] NaN. Lane 0 then adds the slowest lane's polls to the
 *     control block's spin account (refGateAccount), which the host reads
 *     for the stats.
 *  5. The residency policy is replayed from the entries on the host, off
 *     the step's path, before the layer's next prepare (place / admit /
 *     forget / refresh): the policy ends where acquire() would have.
 *
 * Nothing on the step's path waits for the host thread: no callback, no
 * readback, no user event, no flush from a reader, no buffer write.
 *
 * Invariants (each checked where it can be):
 *  - Sequence numbers are assigned by the host in enqueue order and the
 *    publishes run in that order (one in-order queue), so ring entry seq
 *    sits at seq % kRingEntries and the poller takes them one by one. A
 *    stale entry (seq - kRingEntries) never matches the expected seq; the
 *    device never overwrites an entry the host has not consumed (it waits
 *    for the tail, bounded).
 *  - A tag names one placement: tagOf(seq, i) carries seq reduced to 27
 *    bits (1 .. 2^27 - 1) and the item, so it is unique per (publish, item)
 *    over 2^27 - 1 consecutive publishes, never 0 and never has bit 31 set,
 *    whatever seq is; host-filled slots get tags with bit 31 set. A ready
 *    word holds the tag of the last read that finished
 *    into the slot, so an older placement's ready word never satisfies a
 *    newer one (no ABA), and 0 (never a tag) is "nothing read".
 *  - A slot is taken at most once per prepare (the device marks its queue
 *    entry taken) and a slot with a read in flight is never queued, so two
 *    reads never write one slot at once; a read's ready word is stored
 *    before the slot leaves the in-flight set.
 *  - A slot's previous readers (the layer's GEMVs of an earlier step) are
 *    ahead of the publish that hands it out on the same in-order queue, and
 *    its read starts only after that publish ran: no last-use fence needed.
 *  - Between a layer's prepare and the replay of its entries only the
 *    device changes its table, and the host mirror replays exactly the
 *    entries: at the next prepare the device table equals the mirror
 *    (checked with check = true), except for the speculative placements
 *    the host cancelled before their read started, which the prepare
 *    undoes in both.
 *  - A gate never waits for something nobody will produce: every published
 *    read is issued (or failed), the failed word stops every gate, and the
 *    spin budget bounds the rest.
 *
 * The reference forms of the three kernels (refPublish, refSpec, refGate)
 * are the CPU emulation the unit tests drive; tq2moe_fsu_flags.cl follows
 * them line for line and takes the layout below through clDefines().
 */

#ifndef __EXPERT_FSU_FLAGS_H__
#define __EXPERT_FSU_FLAGS_H__

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/prctl.h>
#include <time.h>
#endif

namespace nntrainer {
namespace fsuflags {

// ---------------------------------------------------------------------------
// Layout (32-bit words), shared with tq2moe_fsu_flags.cl through clDefines()
// ---------------------------------------------------------------------------

constexpr uint32_t kNone = 0xFFFFFFFFu; /**< no slot / no expert / taken */
constexpr uint32_t kLine = 64;          /**< words of a 256-byte line */

// control block: one word per line where host and device both write
constexpr uint32_t kCtlTail = 0;      /**< host: last seq consumed */
constexpr uint32_t kCtlFailed = 64;   /**< host: != 0, a read failed */
constexpr uint32_t kCtlError = 128;   /**< device: first error code */
constexpr uint32_t kCtlErrInfo = 129; /**< device: layer << 16 | item */
constexpr uint32_t kCtlDevSeq = 192;  /**< device: last seq published */
// the gates' spin account (device, lane 0 of each gate; the host reads it)
constexpr uint32_t kCtlSpinLo = 200;    /**< failed polls of the slowest */
constexpr uint32_t kCtlSpinHi = 201;    /**<   lane, summed over gates */
constexpr uint32_t kCtlGates = 202;     /**< gates run */
constexpr uint32_t kCtlSpinGates = 203; /**< gates that waited (polls > 0) */
constexpr uint32_t kCtlSpinMax = 204;   /**< the most polls of one gate */
constexpr uint32_t kGateLanes = 16;     /**< a gate's work-group, at most */
constexpr uint32_t kCtlRing = 256;      /**< ring entries start here */
constexpr uint32_t kRingEntries = 128;  /**< power of two */
constexpr uint32_t kEntryWords = 128;   /**< words of one ring entry */
constexpr uint32_t kCtlWords = kCtlRing + kRingEntries * kEntryWords;

// ring entry
constexpr uint32_t kEnSeq = 0;   /**< written last (release) */
constexpr uint32_t kEnKind = 1;  /**< kKindStep or kKindSpec */
constexpr uint32_t kEnLayer = 2; /**< the layer the entry places into */
constexpr uint32_t kEnCount = 3; /**< items */
constexpr uint32_t kEnX = 4;     /**< ext ids that follow */
constexpr uint32_t kEnExt = 8;   /**< step: ranked ids; spec: predicted */
constexpr uint32_t kMaxExt = 32;
constexpr uint32_t kEnItems = 40; /**< items: expert, slot, tag, flags */
constexpr uint32_t kItemWords = 4;
constexpr uint32_t kMaxItems = 16;
static_assert(kEnItems + kItemWords * kMaxItems <= kEntryWords, "entry");
static_assert(kGateLanes >= kMaxItems, "a gate has a lane per item");
static_assert(kCtlDevSeq < kCtlSpinLo && kCtlSpinMax < kCtlRing,
              "the spin account shares the device-written line");
static_assert(kEnExt + kMaxExt <= kEnItems, "entry");

constexpr uint32_t kKindStep = 1; /**< a layer's routed decode step */
constexpr uint32_t kKindSpec = 2; /**< a read ahead for a later step */

constexpr uint32_t kItemRead = 1;  /**< a read was assigned (miss, spec) */
constexpr uint32_t kItemReady = 2; /**< its slot was ready at publish */

// layer block: ready words one per 64 bytes, then the table
constexpr uint32_t kHaveStride = 16; /**< words between two ready words */
constexpr uint32_t kHdrGen = 0;      /**< host: release after a table write */
constexpr uint32_t kHdrVqLen = 1;    /**< host: entries of the victim queue */
constexpr uint32_t kHdrWords = 16;

/** error codes of the error word */
enum Error : uint32_t {
  kOk = 0,
  kErrTimeout = 1,  /**< a gate's spin budget ran out */
  kErrFailed = 2,   /**< a gate saw the failed word (a read failed) */
  kErrNoSlot = 3,   /**< a publish found no slot for a miss */
  kErrRingFull = 4, /**< a publish waited for the tail too long */
  kErrBadArg = 5,   /**< an expert id or a count out of range */
};

/** a short name of error code @a e */
inline const char *errorName(uint32_t e) {
  switch (e) {
  case kOk:
    return "ok";
  case kErrTimeout:
    return "gate timeout";
  case kErrFailed:
    return "read failed";
  case kErrNoSlot:
    return "no slot for a miss";
  case kErrRingFull:
    return "miss ring full";
  case kErrBadArg:
    return "bad argument";
  default:
    return "unknown";
  }
}

/** where the parts of a layer block of @a C slots and @a E experts are */
struct LayerLayout {
  uint32_t C = 0, E = 0;
  uint32_t hdr = 0, slot_of = 0, expert_in = 0, tag = 0, vq = 0, words = 0;
  LayerLayout() = default;
  LayerLayout(uint32_t c, uint32_t e) : C(c), E(e) {
    hdr = kHaveStride * C;
    slot_of = hdr + kHdrWords;
    expert_in = slot_of + E;
    tag = expert_in + C;
    vq = tag + C;
    words = vq + C;
  }
  uint32_t have(uint32_t slot) const { return slot * kHaveStride; }
};

/** the publishes a tag tells apart: 2^27 - 1 (seq reduced to 27 bits) */
constexpr uint32_t kTagSeqPeriod = 0x07FFFFFFu;

/** seq reduced to 27 bits for its tags: 1 .. 2^27 - 1, never 0 (a plain
 *  27-bit mask would give tag 0, "nothing read", to item 0 of every 2^27th
 *  publish) */
inline uint32_t tagSeq(uint32_t seq) { return seq % kTagSeqPeriod + 1u; }

/** the tag of item @a i (< kMaxItems) of publish @a seq: never 0, bit 31
 *  clear (never a host tag) for any seq, unique over 2^27 - 1 consecutive
 *  publishes (tagSeq). tq2moe_fsu_flags.cl's fsu_tag is the same formula. */
inline uint32_t tagOf(uint32_t seq, uint32_t i) {
  return (tagSeq(seq) << 4) | (i & 0xFu);
}

/** a host tag (a slot filled by the host, not by a publish): bit 31 set */
inline uint32_t hostTag(uint32_t n) { return 0x80000000u | (n & 0x7FFFFFFFu); }

/** -D options that give the kernels this layout */
inline std::string clDefines() {
  char b[2048];
  std::snprintf(
    b, sizeof(b),
    " -DFSU_NONE=0x%xu -DFSU_CTL_TAIL=%u -DFSU_CTL_FAILED=%u "
    "-DFSU_CTL_ERROR=%u -DFSU_CTL_ERRINFO=%u -DFSU_CTL_DEVSEQ=%u "
    "-DFSU_CTL_RING=%u -DFSU_RING=%u -DFSU_ENTRY=%u -DFSU_EN_KIND=%u "
    "-DFSU_EN_LAYER=%u -DFSU_EN_COUNT=%u -DFSU_EN_X=%u -DFSU_EN_EXT=%u "
    "-DFSU_MAX_EXT=%u -DFSU_EN_ITEMS=%u -DFSU_MAX_ITEMS=%u -DFSU_KIND_STEP=%u "
    "-DFSU_KIND_SPEC=%u -DFSU_ITEM_READ=%u -DFSU_ITEM_READY=%u "
    "-DFSU_HAVE_STRIDE=%u -DFSU_HDR_GEN=%u -DFSU_HDR_VQLEN=%u "
    "-DFSU_HDR_WORDS=%u -DFSU_ERR_TIMEOUT=%u -DFSU_ERR_FAILED=%u "
    "-DFSU_ERR_NOSLOT=%u -DFSU_ERR_RINGFULL=%u -DFSU_ERR_BADARG=%u "
    "-DFSU_TAG_PERIOD=0x%xu -DFSU_CTL_SPIN_LO=%u -DFSU_CTL_SPIN_HI=%u "
    "-DFSU_CTL_GATES=%u -DFSU_CTL_SPIN_GATES=%u -DFSU_CTL_SPIN_MAX=%u "
    "-DFSU_GATE_LANES=%u",
    kNone, kCtlTail, kCtlFailed, kCtlError, kCtlErrInfo, kCtlDevSeq, kCtlRing,
    kRingEntries, kEntryWords, kEnKind, kEnLayer, kEnCount, kEnX, kEnExt,
    kMaxExt, kEnItems, kMaxItems, kKindStep, kKindSpec, kItemRead, kItemReady,
    kHaveStride, kHdrGen, kHdrVqLen, kHdrWords, (uint32_t)kErrTimeout,
    (uint32_t)kErrFailed, (uint32_t)kErrNoSlot, (uint32_t)kErrRingFull,
    (uint32_t)kErrBadArg, kTagSeqPeriod, kCtlSpinLo, kCtlSpinHi, kCtlGates,
    kCtlSpinGates, kCtlSpinMax, kGateLanes);
  return b;
}

// ---------------------------------------------------------------------------
// Atomics on shared words (fine-grain SVM with SVM atomics)
// ---------------------------------------------------------------------------

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) &&
                std::atomic<uint32_t>::is_always_lock_free,
              "a shared word is a plain lock-free 32-bit atomic");

inline uint32_t loadAcquire(const uint32_t *p) {
  return reinterpret_cast<const std::atomic<uint32_t> *>(p)->load(
    std::memory_order_acquire);
}
inline void storeRelease(uint32_t *p, uint32_t v) {
  reinterpret_cast<std::atomic<uint32_t> *>(p)->store(
    v, std::memory_order_release);
}
/** a plain (relaxed) access: the device-owned table words */
inline uint32_t loadRelaxed(const uint32_t *p) {
  return reinterpret_cast<const std::atomic<uint32_t> *>(p)->load(
    std::memory_order_relaxed);
}
inline void storeRelaxed(uint32_t *p, uint32_t v) {
  reinterpret_cast<std::atomic<uint32_t> *>(p)->store(
    v, std::memory_order_relaxed);
}

/** the first error of the error word, with @a info: set when it is still
 *  clear (a load, then two stores -- no read-modify-write; two failures at
 *  once may both write, either is kept) */
inline void setError(uint32_t *ctl, uint32_t code, uint32_t info) {
  if (loadAcquire(ctl + kCtlError) == kOk) {
    storeRelaxed(ctl + kCtlErrInfo, info);
    storeRelease(ctl + kCtlError, code);
  }
}

// ---------------------------------------------------------------------------
// Reference forms of the kernels (the CPU emulation of the device side)
// ---------------------------------------------------------------------------

/**
 * @brief Wait (bounded by @a cap polls) until ring entry @a seq may be
 *        written: the host consumed seq - kRingEntries.
 * @return false when the budget ran out (the error word is set)
 */
inline bool refRingRoom(uint32_t *ctl, uint32_t seq, uint32_t layer,
                        uint64_t cap) {
  uint64_t n = 0;
  while ((uint32_t)(seq - loadAcquire(ctl + kCtlTail)) > kRingEntries) {
    if (++n > cap) {
      setError(ctl, kErrRingFull, layer << 16);
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

/** @a x in @a ids[0, n) */
inline bool refHas(const uint32_t *ids, uint32_t n, uint32_t x) {
  for (uint32_t i = 0; i < n; ++i)
    if (ids[i] == x)
      return true;
  return false;
}

/**
 * @brief Take the first queued slot of @a lay whose expert is not in
 *        @a avoid[0, navoid): marks the queue entry taken, returns the slot
 *        or kNone.
 */
inline uint32_t refTake(uint32_t *lay, const LayerLayout &L,
                        const uint32_t *avoid, uint32_t navoid) {
  const uint32_t n = loadRelaxed(lay + L.hdr + kHdrVqLen);
  for (uint32_t i = 0; i < n && i < L.C; ++i) {
    const uint32_t v = loadRelaxed(lay + L.vq + i);
    if (v == kNone || v >= L.C)
      continue;
    const uint32_t occ = loadRelaxed(lay + L.expert_in + v);
    if (occ != kNone && refHas(avoid, navoid, occ))
      continue;
    storeRelaxed(lay + L.vq + i, kNone);
    return v;
  }
  return kNone;
}

/** queued slots not taken yet */
inline uint32_t refQueued(const uint32_t *lay, const LayerLayout &L) {
  const uint32_t n = loadRelaxed(lay + L.hdr + kHdrVqLen);
  uint32_t q = 0;
  for (uint32_t i = 0; i < n && i < L.C; ++i)
    q += loadRelaxed(lay + L.vq + i) != kNone ? 1 : 0;
  return q;
}

/** expert @a e into slot @a v of @a lay with tag @a tg (evicts) */
inline void refPlace(uint32_t *lay, const LayerLayout &L, uint32_t e,
                     uint32_t v, uint32_t tg) {
  const uint32_t occ = loadRelaxed(lay + L.expert_in + v);
  if (occ != kNone && occ < L.E)
    storeRelaxed(lay + L.slot_of + occ, kNone);
  storeRelaxed(lay + L.slot_of + e, v);
  storeRelaxed(lay + L.expert_in + v, e);
  storeRelaxed(lay + L.tag + v, tg);
}

/**
 * @brief The publish kernel (tq2moe_fsu_publish): a routed decode step of
 *        layer @a layer selected @a ids[0, K) (ascending ids); place its
 *        misses, publish entry @a seq with the ranked @a ext[0, X).
 * @param[out] slot, tag per selected expert (what the kernel turns into the
 *        GEMVs' offsets and the gate's wait list)
 * @return the error code (also in the error word)
 */
inline uint32_t refPublish(uint32_t *ctl, uint32_t *lay, const LayerLayout &L,
                           uint32_t seq, uint32_t layer, const uint32_t *ids,
                           uint32_t K, const uint32_t *ext, uint32_t X,
                           uint32_t *slot, uint32_t *tag, uint64_t cap) {
  // a failed publish leaves no slot to wait on: its gate fails
  for (uint32_t a = 0; a < K && a < kMaxItems; ++a)
    slot[a] = kNone;
  (void)loadAcquire(lay + L.hdr + kHdrGen); // the host's table writes
  if (K == 0 || K > kMaxItems || X > kMaxExt) {
    setError(ctl, kErrBadArg, layer << 16);
    return kErrBadArg;
  }
  if (!refRingRoom(ctl, seq, layer, cap))
    return kErrRingFull;
  uint32_t *en = ctl + kCtlRing + (seq % kRingEntries) * kEntryWords;
  for (uint32_t a = 0; a < K; ++a) {
    const uint32_t e = ids[a];
    if (e >= L.E) {
      setError(ctl, kErrBadArg, layer << 16 | a);
      return kErrBadArg;
    }
    uint32_t s = loadRelaxed(lay + L.slot_of + e), tg, fl;
    if (s != kNone) {
      tg = loadRelaxed(lay + L.tag + s);
      fl = loadAcquire(lay + L.have(s)) == tg ? kItemReady : 0;
    } else {
      s = refTake(lay, L, ids, K);
      if (s == kNone) {
        setError(ctl, kErrNoSlot, layer << 16 | a);
        return kErrNoSlot;
      }
      tg = tagOf(seq, a);
      refPlace(lay, L, e, s, tg);
      fl = kItemRead;
    }
    slot[a] = s;
    tag[a] = tg;
    uint32_t *it = en + kEnItems + a * kItemWords;
    it[0] = e;
    it[1] = s;
    it[2] = tg;
    it[3] = fl;
  }
  for (uint32_t i = 0; i < X; ++i)
    en[kEnExt + i] = ext[i];
  en[kEnKind] = kKindStep;
  en[kEnLayer] = layer;
  en[kEnCount] = K;
  en[kEnX] = X;
  storeRelease(en + 0, seq);
  storeRelaxed(ctl + kCtlDevSeq, seq);
  return kOk;
}

/**
 * @brief The speculative publish kernel (tq2moe_fsu_spec): a read ahead of
 *        @a pred[0, m) (predicted experts of layer @a layer, ranked) placed
 *        from the layer's queue while more than @a keep queued slots
 *        remain, avoiding the slots of predicted experts; entry @a seq.
 * @return the error code
 */
inline uint32_t refSpec(uint32_t *ctl, uint32_t *lay, const LayerLayout &L,
                        uint32_t seq, uint32_t layer, const uint32_t *pred,
                        uint32_t m, uint32_t keep, uint64_t cap) {
  (void)loadAcquire(lay + L.hdr + kHdrGen);
  if (m > kMaxItems || m > kMaxExt) {
    setError(ctl, kErrBadArg, layer << 16);
    return kErrBadArg;
  }
  if (!refRingRoom(ctl, seq, layer, cap))
    return kErrRingFull;
  uint32_t *en = ctl + kCtlRing + (seq % kRingEntries) * kEntryWords;
  uint32_t queued = refQueued(lay, L), n = 0;
  for (uint32_t j = 0; j < m; ++j) {
    const uint32_t e = pred[j];
    if (e >= L.E) {
      setError(ctl, kErrBadArg, layer << 16 | j);
      return kErrBadArg;
    }
    if (loadRelaxed(lay + L.slot_of + e) != kNone)
      continue; // resident, or placed by an earlier read ahead
    if (queued <= keep)
      break;
    const uint32_t s = refTake(lay, L, pred, m);
    if (s == kNone)
      break;
    --queued;
    const uint32_t tg = tagOf(seq, n);
    refPlace(lay, L, e, s, tg);
    uint32_t *it = en + kEnItems + n * kItemWords;
    it[0] = e;
    it[1] = s;
    it[2] = tg;
    it[3] = kItemRead;
    ++n;
  }
  for (uint32_t i = 0; i < m; ++i)
    en[kEnExt + i] = pred[i];
  en[kEnKind] = kKindSpec;
  en[kEnLayer] = layer;
  en[kEnCount] = n;
  en[kEnX] = m;
  storeRelease(en + 0, seq);
  storeRelaxed(ctl + kCtlDevSeq, seq);
  return kOk;
}

/**
 * @brief The gate kernel (tq2moe_fsu_gate), lane @a a: spin until slot
 *        @a slot's ready word is @a tag; stop on the failed or error word
 *        (checked every 64 polls) or after @a cap polls. On failure: the
 *        error word (first error kept) and the lane's routing weight NaN.
 * @return the error code
 */
inline uint32_t refGateLane(uint32_t *ctl, const uint32_t *lay,
                            const LayerLayout &L, uint32_t layer, uint32_t a,
                            uint32_t slot, uint32_t tag, uint64_t cap, float *w,
                            uint64_t *polls = nullptr) {
  uint32_t err = kOk;
  uint64_t n = 0;
  if (slot >= L.C) {
    err = kErrBadArg;
  } else {
    while (loadAcquire(lay + L.have(slot)) != tag) {
      if ((++n & 63) == 0 && (loadAcquire(ctl + kCtlFailed) != 0 ||
                              loadAcquire(ctl + kCtlError) != 0)) {
        err = kErrFailed;
        break;
      }
      if (n > cap) {
        err = kErrTimeout;
        break;
      }
    }
  }
  if (err != kOk) {
    setError(ctl, err, layer << 16 | a);
    if (w != nullptr)
      w[a] = std::numeric_limits<float>::quiet_NaN();
  }
  if (polls != nullptr)
    *polls = n;
  return err;
}

/**
 * @brief The gate kernel's spin account, after its lanes: lane 0 adds
 *        @a slowest (the most failed polls of a lane, saturated at 2^32 - 1)
 *        to the control block -- polls (64-bit, lo / hi), gates, gates that
 *        waited, the most polls of one gate. One gate at a time.
 */
inline void refGateAccount(uint32_t *ctl, uint64_t slowest) {
  const uint32_t m = (uint32_t)std::min<uint64_t>(slowest, 0xFFFFFFFFu);
  const uint32_t lo = ctl[kCtlSpinLo], nlo = lo + m;
  ctl[kCtlSpinLo] = nlo;
  if (nlo < lo)
    ctl[kCtlSpinHi] += 1u;
  ctl[kCtlGates] += 1u;
  if (m != 0u)
    ctl[kCtlSpinGates] += 1u;
  if (m > ctl[kCtlSpinMax])
    ctl[kCtlSpinMax] = m;
}

/** the control block's spin account (refGateAccount / the gate kernel) */
struct SpinAccount {
  uint64_t polls = 0, gates = 0, waited = 0;
  uint32_t max_polls = 0;
};

/** read the spin account of @a ctl (host; relaxed: a total) */
inline SpinAccount spinAccount(const uint32_t *ctl) {
  SpinAccount a;
  a.polls = (uint64_t)loadRelaxed(ctl + kCtlSpinHi) << 32 |
            loadRelaxed(ctl + kCtlSpinLo);
  a.gates = loadRelaxed(ctl + kCtlGates);
  a.waited = loadRelaxed(ctl + kCtlSpinGates);
  a.max_polls = loadRelaxed(ctl + kCtlSpinMax);
  return a;
}

// ---------------------------------------------------------------------------
// The host side
// ---------------------------------------------------------------------------

/**
 * @brief The host side of the protocol for every streamed layer: sequence
 *        numbers, the layer tables and victim queues (prepare), the ring
 *        poller, the reads it starts, the ready words, the replay of the
 *        residency policy, and drain().
 *
 * Threads: the host thread (beginPublish, prepare, drain, addLayer), one
 * poller (pollOnce, or startPoller's thread) and the readers (readDone).
 * The mirror and the hooks are touched by the host thread only; the poller
 * and the readers share the in-flight reads and the pending replay under
 * one mutex.
 */
class FsuFlagsHost {
public:
  /** the residency policy of one layer (MoeExpertFsu's flags hooks) */
  struct Hooks {
    /** ExpertResidency::victimOrder of the layer, without @a exclude */
    std::function<std::vector<unsigned int>(
      const std::vector<unsigned int> &exclude)>
      victim_order;
    /** ExpertResidency::place: a step of @a need whose misses went to
     *  @a placed */
    std::function<void(
      const std::vector<unsigned int> &need,
      const std::vector<std::pair<unsigned int, unsigned int>> &placed)>
      place;
    /** ExpertResidency::admit: a read ahead put @a expert into @a slot */
    std::function<void(unsigned int expert, unsigned int slot)> admit;
    /** ExpertResidency::forget: a read ahead was dropped before it read */
    std::function<void(unsigned int expert)> forget;
    /** ExpertResidency::refresh: the step's ranks, least to most recent */
    std::function<void(const std::vector<unsigned int> &recency)> refresh;
    /** the layer's expert -> slot map as the policy has it now (read on the
     *  host thread after the hooks above) */
    std::function<std::vector<unsigned int>()> slot_map;
    /** optional: a step never replayed, counted (tallyUnreplayed) */
    std::function<void(unsigned int hits, unsigned int misses)> tally;
  };

  /** one read a published entry asks for */
  struct Read {
    unsigned int layer = 0, expert = 0, slot = 0;
    uint32_t tag = 0;
    bool spec = false; /**< a read ahead (issued at low priority) */
  };

  /**
   * @brief Start @a r: read expert r.expert's record straight into slot
   *        r.slot, at low priority when @a low; the reader exchanges
   *        @a once to true before it reads (true already: skip, cancelled)
   *        and calls readDone(r, errno) at the end in every case (ECANCELED
   *        when skipped). Called by the poller, outside the host's lock.
   */
  using Issue = std::function<void(
    const Read &r, std::shared_ptr<std::atomic<bool>> once, bool low)>;

  /** totals (printed by the backend at exit) */
  struct Stats {
    uint64_t steps = 0, step_items = 0, misses = 0, ready_at_publish = 0;
    uint64_t specs = 0, spec_placed = 0, spec_hits = 0, spec_promoted = 0,
             spec_cancelled = 0;
    /** the poller: per entry it waited for, the time from its last empty
     *  poll to the poll that took it (an upper bound of the time the entry
     *  sat in the ring), us; its sleeps; its thread's CPU time */
    std::vector<uint32_t> poll_gap_us;
    uint64_t poll_sleeps = 0, poller_cpu_us = 0, poller_wall_us = 0;
    uint64_t reads_done = 0, reads_failed = 0;
    uint64_t prepares = 0, prepare_us = 0, prepare_wait_us = 0, adopts = 0,
             room_waits = 0, drains = 0;
    std::vector<uint32_t> miss_read_us; /**< issue -> ready, demand reads */
  };

  /**
   * @param ctl the control block (kCtlWords, zeroed, shared with the device)
   * @param issue the reads' starter
   * @param check compare the device table with the mirror at every prepare
   *        (a protocol self-check; throws std::logic_error on a mismatch)
   */
  FsuFlagsHost(uint32_t *ctl, Issue issue, bool check = false) :
    ctl_(ctl), issue_(std::move(issue)), check_(check) {}
  ~FsuFlagsHost() { stopPoller(); }
  FsuFlagsHost(const FsuFlagsHost &) = delete;
  FsuFlagsHost &operator=(const FsuFlagsHost &) = delete;

  /**
   * @brief Register layer @a layer: its block @a lay (LayerLayout(C, E)
   *        words, shared with the device; written empty here), @a keep the
   *        experts a step selects (the queue keeps at least that many
   *        slots). Before any publish of the layer.
   */
  void addLayer(unsigned int layer, uint32_t *lay, unsigned int C,
                unsigned int E, unsigned int keep, Hooks hooks) {
    if (C == 0 || E == 0 || keep == 0 || keep > C || keep > kMaxItems)
      throw std::invalid_argument("FsuFlagsHost: layer " +
                                  std::to_string(layer) + " shape");
    std::lock_guard<std::mutex> lk(mu_);
    auto &p = layers_[layer];
    if (p)
      throw std::invalid_argument("FsuFlagsHost: layer " +
                                  std::to_string(layer) + " registered twice");
    p = std::make_unique<Layer>();
    p->lay = lay;
    p->L = LayerLayout(C, E);
    p->keep = keep;
    p->hooks = std::move(hooks);
    p->m_slot_of.assign(E, kNone);
    p->m_expert_in.assign(C, kNone);
    p->m_tag.assign(C, 0);
    p->inflight.resize(C);
    // the block starts empty: nothing resident, nothing read, no queue
    for (uint32_t s = 0; s < C; ++s)
      storeRelaxed(lay + p->L.have(s), 0);
    writeTable(*p, {});
  }

  bool hasLayer(unsigned int layer) const {
    std::lock_guard<std::mutex> lk(mu_);
    return layers_.count(layer) != 0;
  }
  const LayerLayout &layout(unsigned int layer) const { return get(layer).L; }
  uint32_t *layerBlock(unsigned int layer) const { return get(layer).lay; }

  /** the layer's table is not current: prepare() before its next publish
   *  (a step of it was published since its last prepare, or drain ran) */
  bool needsPrepare(unsigned int layer) const { return get(layer).dirty; }

  /**
   * @brief The sequence number of the next publish (host thread), which the
   *        caller enqueues now for layer @a layer: a step (@a spec false)
   *        or a read ahead. Publishes must run in this order.
   */
  uint32_t beginPublish(unsigned int layer, bool spec) {
    Layer &l = get(layer);
    const uint32_t seq = ++host_seq_;
    l.last_seq = seq;
    if (!spec)
      l.dirty = true;
    {
      std::lock_guard<std::mutex> lk(work_mu_);
      published_.store(seq, std::memory_order_release);
    }
    work_cv_.notify_one();
    return seq;
  }

  /** the last sequence number handed out */
  uint32_t lastSeq() const { return host_seq_; }

  /** publish @a seq has been consumed */
  bool consumed(uint32_t seq) const {
    return (int32_t)(consumed_.load(std::memory_order_acquire) - seq) >= 0;
  }

  /**
   * @brief Make layer @a layer's table current for its next publishes:
   *        wait (bounded by @a timeout) until its last publish is consumed,
   *        replay what its entries did into the policy and the mirror, take
   *        back the read-ahead placements cancelled before they read, adopt
   *        what the policy changed outside the protocol (a prefill), and
   *        write the slot table and the victim queue. Host thread.
   * @throw std::runtime_error on a failed read, a device error or a timeout
   */
  void prepare(unsigned int layer,
               std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
    const auto t0 = std::chrono::steady_clock::now();
    Layer &l = get(layer);
    waitConsumed(l.last_seq, timeout, "layer " + std::to_string(layer));
    const auto t1 = std::chrono::steady_clock::now();
    throwIfFailed();
    replay(l);
    // the device table equals the mirror (the cancelled placements aside,
    // which replay() undid in the mirror only)
    if (check_)
      checkTable(layer, l);
    l.undone.clear();
    adoptIfChanged(layer, l);
    std::vector<unsigned int> order;
    for (int tries = 0;; ++tries) {
      std::vector<unsigned int> busy = inflightSlots(l);
      order = l.hooks.victim_order(busy);
      if (order.size() >= l.keep)
        break;
      // reads ahead still in flight hold too many slots: wait for them
      // (they end: every read issued ends or fails)
      stats_.room_waits++;
      waitInflightBelow(l, l.L.C - l.keep, timeout);
      if (tries > 2)
        throw std::runtime_error("FsuFlagsHost: layer " +
                                 std::to_string(layer) +
                                 " has fewer free slots than a step needs");
    }
    writeTable(l, order);
    l.dirty = false;
    const auto t2 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mu_);
    stats_.prepares++;
    stats_.prepare_wait_us += usBetween(t0, t1);
    stats_.prepare_us += usBetween(t1, t2);
  }

  /**
   * @brief Every publish consumed, every read ahead not started dropped,
   *        every read finished, every entry replayed into the policy: then
   *        the slots may be written by others (a prefill) and the policy
   *        changed. Host thread. Every layer then needs a prepare (which
   *        adopts what changed).
   */
  void drain(std::chrono::milliseconds timeout = std::chrono::seconds(10)) {
    waitConsumed(host_seq_, timeout, "drain");
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (auto &kv : layers_) {
        Layer &l = *kv.second;
        for (unsigned int s = 0; s < l.L.C; ++s) {
          auto &f = l.inflight[s];
          if (f && f->read.spec && !f->once->exchange(true)) {
            l.pending.push_back(Pending::dead(f->read));
            stats_.spec_cancelled++;
            f.reset();
          }
        }
      }
    }
    waitAllInflight(timeout);
    for (auto &kv : layers_) {
      replay(*kv.second);
      kv.second->dirty = true;
    }
    std::lock_guard<std::mutex> lk(mu_);
    stats_.drains++;
  }

  /**
   * @brief Take the next ring entry if the device published it: start its
   *        reads, drop the reads ahead the step did not need, queue it for
   *        the replay, return the ring slot. Poller thread (one).
   * @return true when an entry was taken
   */
  bool pollOnce() {
    const uint32_t seq = consumed_.load(std::memory_order_relaxed) + 1;
    const uint32_t *en = ctl_ + kCtlRing + (seq % kRingEntries) * kEntryWords;
    if (loadAcquire(en + 0) != seq)
      return false;
    Entry e;
    e.seq = seq;
    e.kind = en[kEnKind];
    e.layer = en[kEnLayer];
    e.n = std::min(en[kEnCount], kMaxItems);
    e.x = std::min(en[kEnX], kMaxExt);
    for (uint32_t i = 0; i < e.x; ++i)
      e.ext[i] = en[kEnExt + i];
    for (uint32_t i = 0; i < e.n; ++i)
      for (uint32_t k = 0; k < kItemWords; ++k)
        e.item[i][k] = en[kEnItems + i * kItemWords + k];
    // the copy is taken: the device may reuse the ring slot
    storeRelease(ctl_ + kCtlTail, seq);
    last_kind_ = e.kind;
    if (on_take_)
      on_take_(seq);
    handle(e);
    consumed_.store(seq, std::memory_order_release);
    if (consume_waiters_.load(std::memory_order_acquire) != 0) {
      std::lock_guard<std::mutex> lk(wait_mu_);
      wait_cv_.notify_all();
    }
    return true;
  }

  /** how the poller waits while a publish is outstanding */
  struct PollConfig {
    /** false (spin): spin for spin_us after the last entry, then poll every
     *  20 us. true (adaptive): spin for spin_us after an entry, then sleep
     *  min_sleep_us, doubling up to max_sleep_us (timer slack 1 ns where the
     *  platform has it), and spin again from lead x the steps' usual
     *  interval after the last step entry (an EWMA of step to step) until
     *  the next one or 1.5 intervals */
    bool adaptive = false;
    unsigned int spin_us = 2000;
    unsigned int min_sleep_us = 1, max_sleep_us = 50;
    float lead = 0.8f;
  };

  /**
   * @brief Run pollOnce on a thread of its own: it waits as @a c says while
   *        a publish is outstanding and sleeps on a condition variable
   *        while none is.
   */
  void startPoller(PollConfig c) {
    if (poller_.joinable())
      return;
    stop_ = false;
    stop_flag_.store(false);
    poller_ = std::thread([this, c]() { pollLoop(c); });
  }
  /** the spin poller with a spin window of @a spin_us */
  void startPoller(unsigned int spin_us = 2000) {
    PollConfig c;
    c.spin_us = spin_us;
    startPoller(c);
  }
  /** tests: called by the poller with each sequence number it took */
  void onTake(std::function<void(uint32_t)> f) { on_take_ = std::move(f); }
  void stopPoller() {
    stop_flag_.store(true);
    {
      std::lock_guard<std::mutex> lk(work_mu_);
      stop_ = true;
    }
    work_cv_.notify_all();
    if (poller_.joinable())
      poller_.join();
  }

  /**
   * @brief Read @a r ended with @a err (0, an errno; ECANCELED: skipped
   *        through its once flag). Success stores the slot's ready word =
   *        the tag (release) before the slot leaves the in-flight set; a
   *        failure sets the failed word, which stops every gate. Reader
   *        threads.
   */
  void readDone(const Read &r, int err) {
    if (err == ECANCELED)
      return; // whoever cancelled it accounted for it
    std::lock_guard<std::mutex> lk(mu_);
    auto it = layers_.find(r.layer);
    if (it == layers_.end())
      return;
    Layer &l = *it->second;
    auto &f = r.slot < l.inflight.size() ? l.inflight[r.slot] : dummy_;
    const bool mine = f && f->read.tag == r.tag;
    if (err != 0) {
      if (error_.empty()) {
        error_ = "reading expert " + std::to_string(r.expert) + " of layer " +
                 std::to_string(r.layer) + " into slot " +
                 std::to_string(r.slot) + " failed (" +
                 std::string(err > 0 ? std::strerror(err) : "end of file") +
                 ")";
        std::fprintf(stderr, "[TQ2CX-FSU-FLAGS] ERROR: %s\n", error_.c_str());
      }
      storeRelease(ctl_ + kCtlFailed, 1);
      stats_.reads_failed++;
    } else {
      storeRelease(l.lay + l.L.have(r.slot), r.tag);
      stats_.reads_done++;
      if (mine && f->demand)
        stats_.miss_read_us.push_back(
          (uint32_t)std::min<uint64_t>(usSince(f->issued), 0xFFFFFFFFu));
    }
    if (mine)
      f.reset();
    inflight_cv_.notify_all();
  }

  /**
   * @brief Wait (bounded) until publish @a seq of layer @a layer is consumed
   *        and the reads its step waits for are done: a host-side join, for
   *        a device whose kernels do not see the host's stores while they
   *        run (the coarse-grain test mode). Never on the protocol's own
   *        path.
   */
  void
  waitStepReads(unsigned int layer, uint32_t seq,
                std::chrono::milliseconds timeout = std::chrono::seconds(10)) {
    waitConsumed(seq, timeout, "layer " + std::to_string(layer));
    Layer &l = get(layer);
    std::unique_lock<std::mutex> lk(mu_);
    const bool ok = inflight_cv_.wait_for(lk, timeout, [&] {
      if (loadAcquire(ctl_ + kCtlFailed) != 0)
        return true;
      for (const auto &f : l.inflight)
        if (f && !f->read.spec)
          return false;
      return true;
    });
    lk.unlock();
    throwIfFailed();
    if (!ok)
      throw std::runtime_error("NNTR_G4MOE_FSU_FLAGS: layer " +
                               std::to_string(layer) + ": step reads not done");
  }

  /** the error word, or kErrFailed when a read failed */
  uint32_t errorCode() const {
    const uint32_t e = loadAcquire(ctl_ + kCtlError);
    if (e != kOk)
      return e;
    return loadAcquire(ctl_ + kCtlFailed) != 0 ? (uint32_t)kErrFailed : kOk;
  }

  /** @throw std::runtime_error when a read failed or a kernel reported an
   *         error */
  void throwIfFailed() const {
    const uint32_t e = errorCode();
    if (e == kOk)
      return;
    std::string what;
    {
      std::lock_guard<std::mutex> lk(mu_);
      what = error_;
    }
    const uint32_t info = loadAcquire(ctl_ + kCtlErrInfo);
    throw std::runtime_error(
      std::string("NNTR_G4MOE_FSU_FLAGS: ") + errorName(e) + " (layer " +
      std::to_string(info >> 16) + ", item " + std::to_string(info & 0xFFFF) +
      ")" + (what.empty() ? std::string() : ": " + what));
  }

  Stats stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stats_;
  }

  /**
   * @brief Count every step entry taken and not replayed yet (a layer's
   *        steps are replayed at its next prepare, so at the end of a
   *        decode the last token's are not) through the layers' tally
   *        hooks, without replaying them: the totals then cover every
   *        step. At exit, once. Any thread.
   * @return the steps counted
   */
  uint64_t tallyUnreplayed() {
    std::vector<std::pair<std::function<void(unsigned int, unsigned int)>,
                          std::pair<unsigned int, unsigned int>>>
      calls;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (auto &kv : layers_) {
        Layer &l = *kv.second;
        if (!l.hooks.tally)
          continue;
        for (const Pending &p : l.pending) {
          if (p.is_dead || p.entry.kind != kKindStep || p.tallied)
            continue;
          unsigned int hits = 0, misses = 0;
          for (uint32_t i = 0; i < p.entry.n; ++i)
            (p.entry.item[i][3] & kItemRead ? misses : hits)++;
          calls.push_back({l.hooks.tally, {hits, misses}});
        }
        for (Pending &p : l.pending)
          p.tallied = true;
      }
    }
    for (auto &c : calls)
      c.first(c.second.first, c.second.second);
    return calls.size();
  }

  /** reads in flight (tests) */
  size_t inflight() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = 0;
    for (const auto &kv : layers_)
      for (const auto &f : kv.second->inflight)
        n += f ? 1 : 0;
    return n;
  }

  /** the mirror's expert -> slot map of @a layer (tests) */
  std::vector<uint32_t> mirror(unsigned int layer) const {
    return get(layer).m_slot_of;
  }

private:
  struct Inflight {
    Read read;
    std::shared_ptr<std::atomic<bool>> once;
    std::chrono::steady_clock::time_point issued;
    bool demand = false; /**< issued for a step (a miss or a promotion) */
  };

  struct Entry {
    uint32_t seq = 0, kind = 0, layer = 0, n = 0, x = 0;
    uint32_t ext[kMaxExt] = {};
    uint32_t item[kMaxItems][kItemWords] = {};
  };

  /** one replay step: an entry, or a read ahead dropped before it read */
  struct Pending {
    bool is_dead = false;
    bool tallied = false; /**< counted by tallyUnreplayed */
    Entry entry;
    Read dead_read;
    static Pending dead(const Read &r) {
      Pending p;
      p.is_dead = true;
      p.dead_read = r;
      return p;
    }
  };

  struct Layer {
    uint32_t *lay = nullptr;
    LayerLayout L;
    unsigned int keep = 0;
    Hooks hooks;
    // host thread only
    std::vector<uint32_t> m_slot_of, m_expert_in, m_tag; /**< the mirror */
    uint32_t last_seq = 0; /**< its last publish */
    bool dirty = true;
    uint32_t gen = 0;
    /** spec placements cancelled (expert, slot) since the last prepare:
     *  the device table still has them */
    std::vector<std::pair<uint32_t, uint32_t>> undone;
    // under mu_
    std::vector<std::shared_ptr<Inflight>> inflight; /**< per slot */
    std::vector<Pending> pending;
    /** per slot: the tag of its last read-ahead placement (spec hits) */
    std::unordered_map<uint32_t, uint32_t> spec_tag;
  };

  static uint64_t usBetween(std::chrono::steady_clock::time_point a,
                            std::chrono::steady_clock::time_point b) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(b -
                                                                           a)
      .count();
  }
  static uint64_t usSince(std::chrono::steady_clock::time_point a) {
    return usBetween(a, std::chrono::steady_clock::now());
  }

  Layer &get(unsigned int layer) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = layers_.find(layer);
    if (it == layers_.end())
      throw std::out_of_range("FsuFlagsHost: unknown layer " +
                              std::to_string(layer));
    return *it->second;
  }

  /** wait until entry @a seq is consumed (bounded) */
  void waitConsumed(uint32_t seq, std::chrono::milliseconds timeout,
                    const std::string &who) {
    if ((int32_t)(consumed_.load(std::memory_order_acquire) - seq) >= 0)
      return;
    const auto t_end = std::chrono::steady_clock::now() + timeout;
    consume_waiters_.fetch_add(1, std::memory_order_acq_rel);
    std::unique_lock<std::mutex> lk(wait_mu_);
    bool ok = true;
    while ((int32_t)(consumed_.load(std::memory_order_acquire) - seq) < 0) {
      if (errorCode() != kOk)
        break;
      if (std::chrono::steady_clock::now() > t_end) {
        ok = false;
        break;
      }
      wait_cv_.wait_for(lk, std::chrono::microseconds(200));
    }
    lk.unlock();
    consume_waiters_.fetch_sub(1, std::memory_order_acq_rel);
    throwIfFailed();
    if (!ok)
      throw std::runtime_error(
        "NNTR_G4MOE_FSU_FLAGS: " + who + ": publish " + std::to_string(seq) +
        " not seen in the miss ring (last consumed " +
        std::to_string(consumed_.load()) + ", device " +
        std::to_string(loadAcquire(ctl_ + kCtlDevSeq)) + ")");
  }

  std::vector<unsigned int> inflightSlots(Layer &l) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<unsigned int> s;
    for (unsigned int i = 0; i < l.L.C; ++i)
      if (l.inflight[i])
        s.push_back(i);
    return s;
  }

  void waitInflightBelow(Layer &l, unsigned int n,
                         std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    inflight_cv_.wait_for(lk, timeout, [&] {
      unsigned int c = 0;
      for (const auto &f : l.inflight)
        c += f ? 1 : 0;
      return c <= n || loadAcquire(ctl_ + kCtlFailed) != 0;
    });
  }

  void waitAllInflight(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    const bool ok = inflight_cv_.wait_for(lk, timeout, [&] {
      for (const auto &kv : layers_)
        for (const auto &f : kv.second->inflight)
          if (f)
            return false;
      return true;
    });
    lk.unlock();
    throwIfFailed();
    if (!ok)
      throw std::runtime_error(
        "NNTR_G4MOE_FSU_FLAGS: drain: expert reads still in flight");
  }

  /** start @a reads (outside the lock) */
  void issueAll(std::vector<std::pair<Read, std::shared_ptr<Inflight>>> &v) {
    for (auto &r : v)
      issue_(r.first, r.second->once, r.first.spec);
  }

  /** register a read of @a r (caller holds mu_) */
  std::shared_ptr<Inflight> addRead(Layer &l, const Read &r) {
    auto f = std::make_shared<Inflight>();
    f->read = r;
    f->once = std::make_shared<std::atomic<bool>>(false);
    f->issued = std::chrono::steady_clock::now();
    f->demand = !r.spec;
    if (l.inflight[r.slot]) {
      // two reads into one slot: a protocol violation; stop every gate
      if (error_.empty())
        error_ = "two reads into slot " + std::to_string(r.slot) +
                 " of layer " + std::to_string(r.layer);
      storeRelease(ctl_ + kCtlFailed, 1);
    }
    l.inflight[r.slot] = f;
    return f;
  }

  /** the poller's part of an entry */
  void handle(const Entry &e) {
    std::vector<std::pair<Read, std::shared_ptr<Inflight>>> start;
    {
      std::lock_guard<std::mutex> lk(mu_);
      auto it = layers_.find(e.layer);
      if (it == layers_.end()) {
        if (error_.empty())
          error_ = "a ring entry for unknown layer " + std::to_string(e.layer);
        storeRelease(ctl_ + kCtlFailed, 1);
        return;
      }
      Layer &l = *it->second;
      std::vector<uint32_t> need;
      for (uint32_t i = 0; i < e.n; ++i) {
        const uint32_t ex = e.item[i][0], s = e.item[i][1], tg = e.item[i][2],
                       fl = e.item[i][3];
        if (s >= l.L.C || ex >= l.L.E) {
          if (error_.empty())
            error_ = "a ring entry out of range";
          storeRelease(ctl_ + kCtlFailed, 1);
          return;
        }
        need.push_back(ex);
        if (fl & kItemRead) {
          Read r;
          r.layer = e.layer;
          r.expert = ex;
          r.slot = s;
          r.tag = tg;
          r.spec = e.kind == kKindSpec;
          start.emplace_back(r, addRead(l, r));
          if (r.spec) {
            l.spec_tag[s] = tg;
            stats_.spec_placed++;
          } else {
            stats_.misses++;
          }
          continue;
        }
        // a hit; on a read ahead of this step?
        if (fl & kItemReady)
          stats_.ready_at_publish++;
        auto st = l.spec_tag.find(s);
        if (st != l.spec_tag.end() && st->second == tg)
          stats_.spec_hits++;
        auto &f = l.inflight[s];
        if (f && f->read.tag == tg && f->read.spec) {
          // the step waits for it: if nobody started it, read it now at
          // demand priority (the same placement, the same tag)
          if (!f->once->exchange(true)) {
            Read r = f->read;
            r.spec = false;
            f.reset();
            start.emplace_back(r, addRead(l, r));
            stats_.spec_promoted++;
          } else {
            f->read.spec = false; // running: the step waits for it, it is
                                  // never dropped
          }
        }
      }
      // the reads ahead of this step that it did not need and nobody
      // started: dropped, their placements undone at the next prepare
      // (replayed after this step's entry, which did not need them)
      std::vector<Pending> dead;
      if (e.kind == kKindStep) {
        stats_.steps++;
        stats_.step_items += e.n;
        for (unsigned int s = 0; s < l.L.C; ++s) {
          auto &f = l.inflight[s];
          if (!f || !f->read.spec ||
              std::find(need.begin(), need.end(), f->read.expert) != need.end())
            continue;
          if (!f->once->exchange(true)) {
            dead.push_back(Pending::dead(f->read));
            stats_.spec_cancelled++;
            f.reset();
          }
        }
        l.spec_tag.clear();
      } else {
        stats_.specs++;
      }
      Pending p;
      p.entry = e;
      l.pending.push_back(p);
      for (Pending &d : dead)
        l.pending.push_back(d);
    }
    issueAll(start);
  }

  /** replay @a l's pending entries into the mirror and the policy (host
   *  thread) */
  void replay(Layer &l) {
    std::vector<Pending> todo;
    {
      std::lock_guard<std::mutex> lk(mu_);
      todo.swap(l.pending);
    }
    for (const Pending &p : todo) {
      if (p.is_dead) {
        const Read &r = p.dead_read;
        if (l.m_slot_of[r.expert] == r.slot && l.m_tag[r.slot] == r.tag) {
          l.m_slot_of[r.expert] = kNone;
          l.m_expert_in[r.slot] = kNone;
          l.undone.emplace_back(r.expert, r.slot);
          if (l.hooks.forget)
            l.hooks.forget(r.expert);
        }
        continue;
      }
      const Entry &e = p.entry;
      std::vector<unsigned int> need;
      std::vector<std::pair<unsigned int, unsigned int>> placed;
      for (uint32_t i = 0; i < e.n; ++i) {
        const uint32_t ex = e.item[i][0], s = e.item[i][1], tg = e.item[i][2];
        need.push_back(ex);
        if (!(e.item[i][3] & kItemRead))
          continue;
        const uint32_t occ = l.m_expert_in[s];
        if (occ != kNone)
          l.m_slot_of[occ] = kNone;
        l.m_slot_of[ex] = s;
        l.m_expert_in[s] = ex;
        l.m_tag[s] = tg;
        if (e.kind == kKindSpec) {
          if (l.hooks.admit)
            l.hooks.admit(ex, s);
        } else {
          placed.emplace_back(ex, s);
        }
      }
      if (e.kind == kKindStep) {
        if (l.hooks.place)
          l.hooks.place(need, placed);
        if (l.hooks.refresh) {
          std::vector<unsigned int> rec(e.ext, e.ext + e.x);
          std::reverse(rec.begin(), rec.end());
          l.hooks.refresh(rec);
        }
      }
    }
  }

  /** the device table equals the mirror, the undone placements aside */
  void checkTable(unsigned int layer, Layer &l) {
    const uint32_t *t = l.lay;
    auto undone = [&](uint32_t e, uint32_t s) {
      for (const auto &u : l.undone)
        if ((e == kNone || u.first == e) && (s == kNone || u.second == s))
          return true;
      return false;
    };
    for (uint32_t e = 0; e < l.L.E; ++e) {
      const uint32_t d = loadRelaxed(t + l.L.slot_of + e);
      if (d != l.m_slot_of[e] && !undone(e, kNone))
        throw std::logic_error("FsuFlagsHost: layer " + std::to_string(layer) +
                               " expert " + std::to_string(e) +
                               ": device slot " + std::to_string(d) +
                               ", mirror " + std::to_string(l.m_slot_of[e]));
    }
    for (uint32_t s = 0; s < l.L.C; ++s) {
      const uint32_t d = loadRelaxed(t + l.L.expert_in + s);
      const uint32_t dt = loadRelaxed(t + l.L.tag + s);
      if (undone(kNone, s))
        continue;
      if (d != l.m_expert_in[s] || (d != kNone && dt != l.m_tag[s]))
        throw std::logic_error(
          "FsuFlagsHost: layer " + std::to_string(layer) + " slot " +
          std::to_string(s) + ": device expert " + std::to_string(d) + " tag " +
          std::to_string(dt) + ", mirror " + std::to_string(l.m_expert_in[s]) +
          " tag " + std::to_string(l.m_tag[s]));
    }
  }

  /** the policy's map changed outside the protocol (a prefill, a blocking
   *  step): take it over; a slot whose expert changed was filled by the
   *  host (synchronously): a host tag, ready at once */
  void adoptIfChanged(unsigned int layer, Layer &l) {
    if (!l.hooks.slot_map)
      return;
    const std::vector<unsigned int> map = l.hooks.slot_map();
    if (map.size() != l.L.E)
      throw std::logic_error("FsuFlagsHost: layer " + std::to_string(layer) +
                             " slot map size");
    bool same = true;
    for (uint32_t e = 0; e < l.L.E && same; ++e)
      same = map[e] == l.m_slot_of[e];
    if (same)
      return;
    if (!inflightSlots(l).empty())
      throw std::logic_error("FsuFlagsHost: layer " + std::to_string(layer) +
                             " changed outside the protocol with reads in "
                             "flight (no drain)");
    std::vector<uint32_t> in(l.L.C, kNone);
    for (uint32_t e = 0; e < l.L.E; ++e)
      if (map[e] != kNone) {
        if (map[e] >= l.L.C || in[map[e]] != kNone)
          throw std::logic_error("FsuFlagsHost: layer " +
                                 std::to_string(layer) + " slot map");
        in[map[e]] = e;
      }
    for (uint32_t s = 0; s < l.L.C; ++s) {
      if (in[s] == l.m_expert_in[s] && in[s] != kNone)
        continue; // the same expert: its bytes and tag stay
      l.m_expert_in[s] = in[s];
      if (in[s] != kNone) {
        const uint32_t tg = hostTag(++host_tags_);
        l.m_tag[s] = tg;
        storeRelease(l.lay + l.L.have(s), tg);
      } else {
        l.m_tag[s] = 0;
      }
    }
    for (uint32_t e = 0; e < l.L.E; ++e)
      l.m_slot_of[e] = map[e];
    std::lock_guard<std::mutex> lk(mu_);
    stats_.adopts++;
  }

  /** write the mirror and the victim queue @a order into the device table,
   *  then release the generation word the kernels acquire first */
  void writeTable(Layer &l, const std::vector<unsigned int> &order) {
    uint32_t *t = l.lay;
    for (uint32_t e = 0; e < l.L.E; ++e)
      storeRelaxed(t + l.L.slot_of + e, l.m_slot_of[e]);
    for (uint32_t s = 0; s < l.L.C; ++s) {
      storeRelaxed(t + l.L.expert_in + s, l.m_expert_in[s]);
      storeRelaxed(t + l.L.tag + s, l.m_tag[s]);
    }
    uint32_t n = 0;
    for (unsigned int s : order)
      if (s < l.L.C && n < l.L.C)
        storeRelaxed(t + l.L.vq + n++, s);
    for (uint32_t i = n; i < l.L.C; ++i)
      storeRelaxed(t + l.L.vq + i, kNone);
    storeRelaxed(t + l.L.hdr + kHdrVqLen, n);
    storeRelease(t + l.L.hdr + kHdrGen, ++l.gen);
  }

  /** this thread's CPU time in us (0 where unknown) */
  static uint64_t threadCpuUs() {
#if defined(__linux__)
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0)
      return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
#endif
    return 0;
  }

  static void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
  }

  void pollLoop(PollConfig c) {
    using clk = std::chrono::steady_clock;
#if defined(__linux__)
    // the adaptive poller's short sleeps: no timer slack (default 50 us)
    if (c.adaptive)
      (void)prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL);
#endif
    const auto t_start = clk::now();
    const uint64_t cpu0 = threadCpuUs();
    auto last = clk::now(); // the last entry taken
    auto last_empty = last; // the last poll that found nothing
    bool waiting = false;   // a poll found nothing since the last take
    auto last_step = last;  // the last step entry taken
    double step_us = 0.0;   // EWMA of step to step
    unsigned int sleep_us = c.min_sleep_us;
    uint64_t sleeps = 0;
    std::vector<uint32_t> gaps;
    auto flush = [&]() {
      // the poller's own account, merged under the lock now and then
      std::lock_guard<std::mutex> lk(mu_);
      stats_.poll_gap_us.insert(stats_.poll_gap_us.end(), gaps.begin(),
                                gaps.end());
      stats_.poll_sleeps += sleeps;
      stats_.poller_cpu_us = threadCpuUs() - cpu0;
      stats_.poller_wall_us = usBetween(t_start, clk::now());
      gaps.clear();
      sleeps = 0;
    };
    for (;;) {
      if (pollOnce()) {
        const auto t = clk::now();
        if (waiting)
          gaps.push_back((uint32_t)std::min<uint64_t>(usBetween(last_empty, t),
                                                      0xFFFFFFFFu));
        waiting = false;
        last = t;
        sleep_us = c.min_sleep_us;
        if (last_kind_ == kKindStep) {
          const double d = (double)usBetween(last_step, t);
          // an EWMA of the step interval; a long gap (a token boundary, a
          // prefill) counts at most 4 intervals
          step_us =
            step_us == 0.0 ? d : 0.8 * step_us + 0.2 * std::min(d, 4 * step_us);
          last_step = t;
        }
        if (gaps.size() >= 256)
          flush();
        continue;
      }
      last_empty = clk::now();
      waiting = true;
      const uint32_t cs = consumed_.load(std::memory_order_relaxed);
      if (published_.load(std::memory_order_acquire) == cs) {
        // nothing outstanding: sleep until a publish is announced
        flush();
        std::unique_lock<std::mutex> lk(work_mu_);
        work_cv_.wait(lk, [&] {
          return stop_ || published_.load(std::memory_order_acquire) !=
                            consumed_.load(std::memory_order_relaxed);
        });
        if (stop_)
          return;
        last = clk::now();
        waiting = false;
        continue;
      }
      if (stop_flag_.load(std::memory_order_relaxed)) {
        flush();
        return;
      }
      const uint64_t since = usBetween(last, last_empty);
      if (!c.adaptive) {
        if (since > c.spin_us) {
          std::this_thread::sleep_for(std::chrono::microseconds(20));
          ++sleeps;
        } else {
          cpuRelax();
        }
        continue;
      }
      bool spin = since < c.spin_us;
      if (!spin && step_us > 0.0) {
        // the next step entry is due: spin through its usual arrival
        const double ss = (double)usBetween(last_step, last_empty);
        spin = ss >= c.lead * step_us && ss <= 1.5 * step_us + c.spin_us;
      }
      if (spin) {
        cpuRelax();
        sleep_us = c.min_sleep_us;
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        ++sleeps;
        sleep_us = std::min(c.max_sleep_us, 2 * sleep_us);
      }
    }
  }

  uint32_t *ctl_;
  Issue issue_;
  bool check_;
  mutable std::mutex mu_;
  std::condition_variable inflight_cv_;
  std::unordered_map<unsigned int, std::unique_ptr<Layer>> layers_;
  std::shared_ptr<Inflight> dummy_;
  std::string error_;
  Stats stats_;
  uint32_t host_seq_ = 0;  /**< host thread */
  uint32_t host_tags_ = 0; /**< host thread */
  std::atomic<uint32_t> consumed_{0};
  std::atomic<uint32_t> published_{0};
  std::atomic<int> consume_waiters_{0};
  std::mutex wait_mu_;
  std::condition_variable wait_cv_;
  std::mutex work_mu_;
  std::condition_variable work_cv_;
  bool stop_ = false;
  std::atomic<bool> stop_flag_{false}; /**< stop_, for the poller's spin */
  std::function<void(uint32_t)> on_take_;
  uint32_t last_kind_ = 0; /**< poller: the kind of the entry it took */
  std::thread poller_;
};

} // namespace fsuflags
} // namespace nntrainer

#endif // __EXPERT_FSU_FLAGS_H__

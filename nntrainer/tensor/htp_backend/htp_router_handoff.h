// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_router_handoff.h
 * @brief One-shot ownership tickets for the LFM router's normalized input.
 * @author NNTrainer contributors
 * @bug No known bugs except for NYI items.
 */
#ifndef __HTP_ROUTER_HANDOFF_H__
#define __HTP_ROUTER_HANDOFF_H__

#include <stddef.h>
#include <stdint.h>

/** Same wire word in router n_sel (only top_k=0) and plain MoE act.
 * Bit 0 remains the activation selector; handoff is SwiGLU-only. */
#define HTP_ROUTER_HANDOFF_MARK 0x80000000u
#define HTP_ROUTER_HANDOFF_MAX 0x3fffffffu

static inline uint32_t htp_router_handoff_wire(uint32_t ticket) {
  return ticket && ticket <= HTP_ROUTER_HANDOFF_MAX
           ? HTP_ROUTER_HANDOFF_MARK | (ticket << 1u)
           : 0u;
}
static inline uint32_t htp_router_handoff_ticket(uint32_t wire) {
  return (wire & HTP_ROUTER_HANDOFF_MARK) && !(wire & 1u)
           ? (wire & ~HTP_ROUTER_HANDOFF_MARK) >> 1u
           : 0u;
}
static inline int htp_router_handoff_shape(uint32_t M, uint32_t K) {
  return M >= 128u && M <= 4096u && K == 2048u;
}

/** Alias rejection only; overlap is never evidence of equal contents. */
static inline int htp_router_handoff_overlap(const void *a, size_t an,
                                             const void *b, size_t bn) {
  const uintptr_t aa = (uintptr_t)a, bb = (uintptr_t)b;
  return an && bn && (aa <= bb ? bb - aa < an : aa - bb < bn);
}

/** ARM source ownership, serialized by invoke_mutex_. No owned allocation or
 * content cache. epoch changes before any ION write or ownership replacement;
 * ticket identifies a producer even when buffer/session addresses repeat. */
typedef struct {
  uint64_t epoch, published_epoch, owner;
  const void *source;
  size_t bytes;
  uint32_t next_ticket, ticket;
} htp_router_handoff;

static inline void htp_router_handoff_invalidate(htp_router_handoff *s) {
  ++s->epoch;
  s->ticket = 0u;
  s->source = NULL;
  s->bytes = 0u;
  s->owner = 0u;
}

/** Tickets never wrap. Exhaustion disables the optimization for this owner. */
static inline uint32_t htp_router_handoff_next(htp_router_handoff *s) {
  htp_router_handoff_invalidate(s);
  if (s->next_ticket == HTP_ROUTER_HANDOFF_MAX)
    return 0u;
  return ++s->next_ticket;
}

static inline void htp_router_handoff_publish(htp_router_handoff *s,
                                              uint32_t ticket, uint64_t owner,
                                              const void *source,
                                              size_t bytes) {
  htp_router_handoff_invalidate(s);
  if (!ticket || ticket > HTP_ROUTER_HANDOFF_MAX || !owner || !source || !bytes)
    return;
  s->owner = owner;
  s->source = source;
  s->bytes = bytes;
  s->published_epoch = s->epoch;
  s->ticket = ticket;
}

/** ARM only: authorize sending ticket in the plain MoE RPC only after the
 * actual ION source, session, extent and write epoch match. Every attempt
 * consumes the stamp, including mismatches and failures. */
static inline uint32_t htp_router_handoff_take(htp_router_handoff *s,
                                               uint64_t owner,
                                               const void *source, size_t bytes,
                                               uint32_t ticket) {
  const uint32_t hit =
    s->ticket && s->ticket == ticket && s->epoch == s->published_epoch &&
        s->owner == owner && s->source == source && s->bytes == bytes
      ? ticket
      : 0u;
  htp_router_handoff_invalidate(s);
  return hit;
}

/** DSP scratch ownership is separate from ARM source ownership. FastRPC may
 * map one ION buffer at different DSP VAs in the router and MoE RPCs, so no RPC
 * input VA is kept here. The paired ARM client sends a ticket only after its
 * source-bound take above succeeds. This session's DSP must independently
 * match that ticket, its retained allocation, extent and scratch write epoch.
 * This is a trusted paired-client protocol, not authentication of arbitrary
 * callers that fabricate ticket words. */
typedef struct {
  uint64_t epoch, published_epoch;
  const void *owner;
  size_t bytes;
  uint32_t ticket;
} htp_router_handoff_scratch;

static inline void
htp_router_handoff_scratch_invalidate(htp_router_handoff_scratch *s) {
  ++s->epoch;
  s->ticket = 0u;
  s->owner = NULL;
  s->bytes = 0u;
}

/** Publish only after synchronous router input DMA and its readers finish. */
static inline void
htp_router_handoff_scratch_publish(htp_router_handoff_scratch *s,
                                   uint32_t ticket, const void *owner,
                                   size_t bytes) {
  htp_router_handoff_scratch_invalidate(s);
  if (!ticket || ticket > HTP_ROUTER_HANDOFF_MAX || !owner || !bytes)
    return;
  s->owner = owner;
  s->bytes = bytes;
  s->published_epoch = s->epoch;
  s->ticket = ticket;
}

/** The ARM-authorized ticket is consumed once even on a DSP-side mismatch. */
static inline uint32_t
htp_router_handoff_scratch_take(htp_router_handoff_scratch *s,
                                const void *owner, size_t bytes,
                                uint32_t ticket) {
  const uint32_t hit = s->ticket && s->ticket == ticket &&
                           s->epoch == s->published_epoch &&
                           s->owner == owner && s->bytes == bytes
                         ? ticket
                         : 0u;
  htp_router_handoff_scratch_invalidate(s);
  return hit;
}
#endif

// SPDX-License-Identifier: Apache-2.0
// The device side of the streamed-expert decode step without a host round
// trip (NNTR_G4MOE_FSU_FLAGS, expert_fsu_flags.h): line for line the
// reference forms refPublish / refSpec / refGateLane there, which the host
// unit tests drive. Built with OpenCL C 2.0 atomics (-cl-std=CL2.0, or
// CL3.0 with the all-devices scope and acquire / release) on fine-grain SVM
// with SVM atomics; the layout comes as -D options (fsuflags::clDefines()).
// The control block (ctl) and the layer block (lay) are SVM pointers; the
// routing buffers are plain device buffers.
//
// tq2moe_fsu_publish: one work-item. A routed decode step selected ids[K]
// (ascending, tq2moe_route_select): a resident expert keeps its slot; a
// miss takes the first queued slot whose expert the step does not need, and
// the slot table is updated. Per selected expert: the GEMVs' record offsets
// (off_gu = gate offsets then up offsets, off_d = down offsets, in words of
// the layer's partition: slot * stride + the record's shift in its slot)
// and the gate's wait list (slot, tag). Then one ring entry, its sequence
// number stored last (release).
// tq2moe_fsu_spec: one work-item. A read ahead of the predicted experts
// pred[m] of a later step, placed from that layer's queue while more than
// `keep` slots stay queued, avoiding the slots of predicted experts.
// tq2moe_fsu_gate: one work-group of K lanes ahead of the expert GEMVs:
// lane a spins until its slot's ready word holds its tag (acquire), stops
// on the failed / error word (every 64 polls) or after `cap` polls; on
// failure the error word and a NaN routing weight wts[a]. Then lane 0 adds
// the slowest lane's failed polls to the control block's spin account
// (gates, gates that waited, polls, the largest gate's polls).
//
// The spinners are narrow (one work-group, K lanes, each on its own word)
// and no word is counted into by many groups (a probe on Adreno 840: system-
// scope RMWs serialise at ~0.33 us; legacy atomic_* / mem_fence were not
// coherent with the host on Adreno 840).

#define FSU_ACQ(p)                                                             \
  atomic_load_explicit((volatile __global atomic_uint *)(p),                   \
                       memory_order_acquire, memory_scope_all_svm_devices)
#define FSU_REL(p, v)                                                          \
  atomic_store_explicit((volatile __global atomic_uint *)(p), (v),             \
                        memory_order_release, memory_scope_all_svm_devices)

// the parts of a layer block of C slots and E experts
#define L_HAVE(s) ((s) * FSU_HAVE_STRIDE)
#define L_HDR(C) ((C) * FSU_HAVE_STRIDE)
#define L_SLOTOF(C) (L_HDR(C) + FSU_HDR_WORDS)
#define L_EXPIN(C, E) (L_SLOTOF(C) + (E))
#define L_TAG(C, E) (L_EXPIN(C, E) + (C))
#define L_VQ(C, E) (L_TAG(C, E) + (C))

// fsuflags::tagOf: seq reduced to 27 bits (1 .. 2^27 - 1), so a tag is
// never 0 ("nothing read") and never has bit 31 set (the host tags)
inline uint fsu_tag(uint seq, uint i) {
  return ((seq % FSU_TAG_PERIOD + 1u) << 4) | (i & 0xFu);
}

// the first error of the error word is kept (a load, then two stores: no
// read-modify-write)
inline void fsu_set_error(__global uint *ctl, uint code, uint info) {
  if (FSU_ACQ(ctl + FSU_CTL_ERROR) == 0u) {
    ctl[FSU_CTL_ERRINFO] = info;
    FSU_REL(ctl + FSU_CTL_ERROR, code);
  }
}

// ring entry seq may be written once the host consumed seq - FSU_RING
inline int fsu_ring_room(__global uint *ctl, uint seq, uint layer, ulong cap) {
  ulong n = 0;
  while ((uint)(seq - FSU_ACQ(ctl + FSU_CTL_TAIL)) > (uint)FSU_RING) {
    if (++n > cap) {
      fsu_set_error(ctl, FSU_ERR_RINGFULL, layer << 16);
      return 0;
    }
  }
  return 1;
}

inline int fsu_has(__global const uint *ids, uint n, uint x) {
  for (uint i = 0; i < n; ++i)
    if (ids[i] == x)
      return 1;
  return 0;
}

// the first queued slot whose expert is not in avoid[navoid]: taken
inline uint fsu_take(__global uint *lay, uint C, uint E,
                     __global const uint *avoid, uint navoid) {
  const uint n = lay[L_HDR(C) + FSU_HDR_VQLEN];
  for (uint i = 0; i < n && i < C; ++i) {
    const uint v = lay[L_VQ(C, E) + i];
    if (v == FSU_NONE || v >= C)
      continue;
    const uint occ = lay[L_EXPIN(C, E) + v];
    if (occ != FSU_NONE && fsu_has(avoid, navoid, occ))
      continue;
    lay[L_VQ(C, E) + i] = FSU_NONE;
    return v;
  }
  return FSU_NONE;
}

inline uint fsu_queued(__global const uint *lay, uint C, uint E) {
  const uint n = lay[L_HDR(C) + FSU_HDR_VQLEN];
  uint q = 0;
  for (uint i = 0; i < n && i < C; ++i)
    q += lay[L_VQ(C, E) + i] != FSU_NONE ? 1u : 0u;
  return q;
}

inline void fsu_place(__global uint *lay, uint C, uint E, uint e, uint v,
                      uint tg) {
  const uint occ = lay[L_EXPIN(C, E) + v];
  if (occ != FSU_NONE && occ < E)
    lay[L_SLOTOF(C) + occ] = FSU_NONE;
  lay[L_SLOTOF(C) + e] = v;
  lay[L_EXPIN(C, E) + v] = e;
  lay[L_TAG(C, E) + v] = tg;
}

__kernel void tq2moe_fsu_publish(
  __global uint *ctl, __global uint *lay, __global const uint *ids,
  __global const uint *ext, __global uint *off_gu, __global uint *off_d,
  __global const uint *shift_w, __global uint *wait, const uint seq,
  const uint layer, const uint C, const uint E, const uint K, const uint X,
  const uint stride_w, const uint gu_w, const ulong cap) {
  if (get_global_id(0) != 0)
    return;
  // a failed publish leaves no slot to wait on: its gate fails
  for (uint a = 0; a < K && a < FSU_MAX_ITEMS; ++a)
    wait[a] = FSU_NONE;
  (void)FSU_ACQ(lay + L_HDR(C) + FSU_HDR_GEN); // the host's table writes
  if (K == 0u || K > FSU_MAX_ITEMS || X > FSU_MAX_EXT) {
    fsu_set_error(ctl, FSU_ERR_BADARG, layer << 16);
    return;
  }
  if (!fsu_ring_room(ctl, seq, layer, cap))
    return;
  __global uint *en = ctl + FSU_CTL_RING + (seq % FSU_RING) * FSU_ENTRY;
  for (uint a = 0; a < K; ++a) {
    const uint e = ids[a];
    if (e >= E) {
      fsu_set_error(ctl, FSU_ERR_BADARG, layer << 16 | a);
      return;
    }
    uint s = lay[L_SLOTOF(C) + e], tg, fl;
    if (s != FSU_NONE) {
      tg = lay[L_TAG(C, E) + s];
      fl = FSU_ACQ(lay + L_HAVE(s)) == tg ? FSU_ITEM_READY : 0u;
    } else {
      s = fsu_take(lay, C, E, ids, K);
      if (s == FSU_NONE) {
        fsu_set_error(ctl, FSU_ERR_NOSLOT, layer << 16 | a);
        return;
      }
      tg = fsu_tag(seq, a);
      fsu_place(lay, C, E, e, s, tg);
      fl = FSU_ITEM_READ;
    }
    const uint base = s * stride_w + shift_w[e];
    off_gu[a] = base;
    off_gu[K + a] = base + gu_w;
    off_d[a] = base + 2u * gu_w;
    wait[K + a] = tg;
    wait[a] = s;
    __global uint *it = en + FSU_EN_ITEMS + a * 4u;
    it[0] = e;
    it[1] = s;
    it[2] = tg;
    it[3] = fl;
  }
  for (uint i = 0; i < X; ++i)
    en[FSU_EN_EXT + i] = ext[i];
  en[FSU_EN_KIND] = FSU_KIND_STEP;
  en[FSU_EN_LAYER] = layer;
  en[FSU_EN_COUNT] = K;
  en[FSU_EN_X] = X;
  FSU_REL(en, seq);
  ctl[FSU_CTL_DEVSEQ] = seq;
}

__kernel void tq2moe_fsu_spec(__global uint *ctl, __global uint *lay,
                              __global const uint *pred, const uint seq,
                              const uint layer, const uint C, const uint E,
                              const uint m, const uint keep, const ulong cap) {
  if (get_global_id(0) != 0)
    return;
  (void)FSU_ACQ(lay + L_HDR(C) + FSU_HDR_GEN);
  if (m > FSU_MAX_ITEMS || m > FSU_MAX_EXT) {
    fsu_set_error(ctl, FSU_ERR_BADARG, layer << 16);
    return;
  }
  if (!fsu_ring_room(ctl, seq, layer, cap))
    return;
  __global uint *en = ctl + FSU_CTL_RING + (seq % FSU_RING) * FSU_ENTRY;
  uint queued = fsu_queued(lay, C, E), n = 0;
  for (uint j = 0; j < m; ++j) {
    const uint e = pred[j];
    if (e >= E) {
      fsu_set_error(ctl, FSU_ERR_BADARG, layer << 16 | j);
      return;
    }
    if (lay[L_SLOTOF(C) + e] != FSU_NONE)
      continue; // resident, or placed by an earlier read ahead
    if (queued <= keep)
      break;
    const uint s = fsu_take(lay, C, E, pred, m);
    if (s == FSU_NONE)
      break;
    --queued;
    const uint tg = fsu_tag(seq, n);
    fsu_place(lay, C, E, e, s, tg);
    __global uint *it = en + FSU_EN_ITEMS + n * 4u;
    it[0] = e;
    it[1] = s;
    it[2] = tg;
    it[3] = FSU_ITEM_READ;
    ++n;
  }
  for (uint i = 0; i < m; ++i)
    en[FSU_EN_EXT + i] = pred[i];
  en[FSU_EN_KIND] = FSU_KIND_SPEC;
  en[FSU_EN_LAYER] = layer;
  en[FSU_EN_COUNT] = n;
  en[FSU_EN_X] = m;
  FSU_REL(en, seq);
  ctl[FSU_CTL_DEVSEQ] = seq;
}

__kernel void tq2moe_fsu_gate(__global uint *ctl, __global const uint *lay,
                              __global const uint *wait, __global float *wts,
                              const uint layer, const uint C, const uint K,
                              const ulong cap) {
  // the spin account (fsuflags::refGateAccount): each lane's failed polls,
  // the slowest lane's added to the control block by lane 0
  __local uint spins[FSU_GATE_LANES];
  const uint a = get_global_id(0), lid = get_local_id(0);
  ulong n = 0;
  if (a < K) {
    const uint s = wait[a], tg = wait[K + a];
    uint err = 0u;
    if (s >= C) {
      err = FSU_ERR_BADARG;
    } else {
      while (FSU_ACQ(lay + L_HAVE(s)) != tg) {
        if ((++n & 63ul) == 0ul && (FSU_ACQ(ctl + FSU_CTL_FAILED) != 0u ||
                                    FSU_ACQ(ctl + FSU_CTL_ERROR) != 0u)) {
          err = FSU_ERR_FAILED;
          break;
        }
        if (n > cap) {
          err = FSU_ERR_TIMEOUT;
          break;
        }
      }
    }
    if (err != 0u) {
      fsu_set_error(ctl, err, layer << 16 | a);
      wts[a] = NAN;
    }
  }
  if (lid < FSU_GATE_LANES)
    spins[lid] = (uint)min(n, (ulong)0xFFFFFFFFu);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    uint m = 0u;
    for (uint i = 0; i < FSU_GATE_LANES && i < get_local_size(0); ++i)
      m = max(m, spins[i]);
    // one gate at a time (one work-group, one in-order queue): plain
    // read-modify-writes of device-only words
    const uint lo = ctl[FSU_CTL_SPIN_LO], nlo = lo + m;
    ctl[FSU_CTL_SPIN_LO] = nlo;
    if (nlo < lo)
      ctl[FSU_CTL_SPIN_HI] += 1u;
    ctl[FSU_CTL_GATES] += 1u;
    if (m != 0u)
      ctl[FSU_CTL_SPIN_GATES] += 1u;
    if (m > ctl[FSU_CTL_SPIN_MAX])
      ctl[FSU_CTL_SPIN_MAX] = m;
  }
}

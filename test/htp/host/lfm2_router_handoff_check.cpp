// SPDX-License-Identifier: Apache-2.0
/**
 * @file lfm2_router_handoff_check.cpp
 * @brief Actual staging/RPC/scratch lifecycle checks with host arithmetic
 * stubs.
 * @author NNTrainer contributors
 * @bug Hardware cache coherence and model arithmetic need device validation.
 */
#include "state.h"
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

static unsigned tests, ion_allocs, dsp_allocs;
static size_t ion_copy_bytes, dsp_copy_bytes;
static bool fail_ion_alloc, fail_dsp_alloc;
static void check(bool ok, const char *why) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", why);
    std::abort();
  }
}
static void stagedMemcpy(void *d, const void *s, size_t n, int = 0) {
  ion_copy_bytes += n;
  std::memcpy(d, s, n);
}
struct HtpRpcBuffer {
  std::vector<uint8_t> memory;
  uint8_t *mapped = nullptr; // Fixture-only alternate mapping of the same ION.
  explicit HtpRpcBuffer(size_t n) {
    if (fail_ion_alloc)
      throw std::bad_alloc();
    memory.resize(n);
    ++ion_allocs;
  }
  uint8_t *data() { return mapped ? mapped : memory.data(); }
  size_t size() const { return memory.size(); }
};
#define ml_logw(...) ((void)0)
struct HtpProfile {
  int value = 0;
  static HtpProfile &global() {
    static HtpProfile p;
    return p;
  }
  int level() const { return value; }
};
static int direct_rpc_failure;
static unsigned direct_rpc_calls;
static const float *direct_q_input;
struct HtpBackend {
  static HtpBackend &global() {
    static HtpBackend h;
    return h;
  }
  bool enabled() const { return true; }
  uint64_t handle() const { return 7; }
};
using remote_handle64 = uintptr_t;
static int64_t now_us() { return 0; }
static bool attn_trace_enabled() { return false; }
static constexpr unsigned kDecodeMaxRows = 8, kDecodeMaxHeadDim = 128;
#define ml_logi(...) ((void)0)
template <size_t output_index, typename... Args>
static int direct_rpc(Args... args) {
  auto t = std::make_tuple(args...);
  float *out = std::get<output_index>(t);
  const int len = std::get<output_index + 1>(t);
  for (int i = 0; i < len; ++i)
    out[i] = 7.25f + float(i % 13);
  ++direct_rpc_calls;
  return direct_rpc_failure;
}
template <typename... Args> static int nntr_hvx_attn_q_step(Args... args) {
  direct_q_input = std::get<13>(std::make_tuple(args...));
  return direct_rpc<17>(args...);
}
template <typename... Args> static int nntr_hvx_attn_f16_decode(Args... args) {
  return direct_rpc<17>(args...);
}
template <typename... Args> static int nntr_hvx_attn_f16_prefill(Args... args) {
  return direct_rpc<19>(args...);
}
struct Backend {
  struct IonAlias {
    const void *host = nullptr;
    const void *ion = nullptr;
    size_t bytes = 0;
  };
  std::array<IonAlias, 16> ion_alias_{};
  htp_router_handoff router_handoff_{};
  std::mutex invoke_mutex_;
// clang-format off
#include "pool.inc"
#include "ion_alias.inc"
#include "sync.inc"
  // clang-format on
  StagingPool act_pool_, out_pool_, x2_pool_, attn_q_pool_, attn_out_pool_;
  bool have_id_ = false;
  unsigned pending_id_ = 17;
  enum { AEE_SUCCESS = 0 };
  // clang-format off
#include "attn_q.inc"
#include "attn_f16.inc"
#include "take_id.inc"
  // clang-format on
  void large_writer_guard(unsigned which) {
    if (which == 0) {
#include "release_prefill_scratch.inc"
    } else if (which == 1) {
#include "tokenForward.inc"
    } else {
#include "oprojection_materialize.inc"
    }
  }
  struct Queue {
    std::unique_ptr<HtpRpcBuffer> act, out;
  };
  std::unique_ptr<Queue> dspq_;
  void beginPrompt() {
#include "beginPrompt.inc"
  }
  void releasePrefillStaging() {
#include "releasePrefillStaging.inc"
    act_pool_ = StagingPool{};
    out_pool_ = StagingPool{};
  }
  uint32_t publish(uint64_t session, const void *source, size_t bytes) {
    uint32_t t = htp_router_handoff_next(&router_handoff_);
    htp_router_handoff_publish(&router_handoff_, t, session, source, bytes);
    return t;
  }
  uint32_t producer(const void *x, const void *act, void *logits,
                    unsigned M = 1024, unsigned K = 2048, unsigned E = 64,
                    bool handoff_enabled = true, const float *gamma = nullptr,
                    unsigned top_k = 0, uint32_t h8 = UINT32_MAX) {
    const size_t x_len = size_t(M) * K, out_len = size_t(M) * E;
#include "producer.inc"
    return ticket;
  }
  uint32_t select(uint64_t session, const float *act, unsigned M = 1024,
                  unsigned K = 2048, unsigned inter = 1536,
                  unsigned N_out = 2048, int kind = 0, uint32_t glu = 0,
                  const float *pre_gamma = nullptr,
                  const float *post_gamma = nullptr,
                  const float *add_x2 = nullptr) {
    const bool via_dspq = false, host_prenorm = false;
    const bool with_norms = pre_gamma || post_gamma;
    const size_t act_bytes = size_t(M) * K * sizeof(float);
    const size_t out_bytes = size_t(M) * N_out * sizeof(float);
#include "select.inc"
    const int rep = 0;
    return
#include "wire.inc"
      ;
  }
};

typedef uint32_t uint32;
enum {
  AEE_SUCCESS,
  AEE_EBADPARM,
  AEE_ENOMEMORY,
  AEE_EINVALIDFORMAT,
  AEE_EUNSUPPORTED
};
#define FARF(level, ...) ((void)0)
#define MOE_SCRATCH_ALIGN 128u
#define ROUND_UP_SZ(x, a) (((x) + (a)-1u) & ~(size_t)((a)-1u))
#define ROUND_UP_U32(x, a) (((x) + (a)-1u) & ~((a)-1u))
#define HEXKL_PROBE_T0(x) ((void)0)
#define HEXKL_PROBE_ADD(x, y) ((void)0)
#define HEXKL_MOE_FLAG_GELU_TANH 128u
#include "had_flags.inc"
#define HVX_GLU_SILU 0u
#define HVX_GLU_GELU_TANH 1u
// Tile dimensions mirror hexkl_micro.h; this host test does not need HexKL.
#define HEXKL_HMX_INT8_BLOCK_N_ROW 64u
#define HEXKL_HMX_INT8_BLOCK_N_COL 32u
#define HEXKL_HMX_INT8_BLOCK_N_INNER 32u
#define HEXKL_HMX_ACTIVATION_ALIGNMENT 2048u
#define HEXKL_MOE_PACK_UNIT_ROWS 16u
#define MOE_U8_GUARD 128u
#define MOE_MAX_CHUNKS 16u
#define MOE_TAIL_MAX_ROWS 0u
#define MOE_TAIL_TILE_BYTES 0u
#define MOE_M1_MAX_ROWS 4u
#define MOE_M1_MAX_EXPERTS 16u
#define MOE_M1_TILE_BYTES 512u
#define HEXKL_DMA_SITE_COPY_IN 1u
// Background job/context stand-ins reserve 64/256 bytes here (v81: 32/64).
// The extracted carve code tests ordering and input byte parity with those
// same sizes on both paths; exact DSP capacity uses a separate v81 ABI check.
struct hvx_bg_job {
  uint8_t bytes[64];
};
struct moe_tail_ctx {
  uint8_t bytes[256];
};
struct moe_tail_shared {
  int32_t *acc_gu, *acc_dn, *rq_zp;
  float *gate_f32, *rq_scale;
  uint8_t *mid_ah;
};
#include "scratch.inc"
static void *fixture_malloc(size_t n) {
  if (fail_dsp_alloc)
    return nullptr;
  ++dsp_allocs;
  return std::malloc(n);
}
#define malloc fixture_malloc
#include "reserve.inc"
#undef malloc
struct WeightTable {
  bool compact = true;
};
struct nntr_hvx_session {
  hexkl_moe_scratch moe_scratch{};
  WeightTable weights_u8i4;
  uint8_t *vtcm_base = nullptr;
  uint32_t vtcm_size = 0, config_off = 0, moe_flags = 0;
  void *quant_pool = nullptr;
  int hmx_locked = 1;
};
static bool fail_router, fail_core;
static unsigned core_calls, core_hits;
static size_t last_need, last_carved;
static void hexkl_dma_ring_reset() {}
static void hexkl_moe_dma_copy(void *d, const void *s, size_t n, int, int) {
  dsp_copy_bytes += n;
  std::memcpy(d, s, n);
}
static void moe_dma_copy(void *d, const void *s, size_t n, int a, int b, int) {
  hexkl_moe_dma_copy(d, s, n, a, b);
}
static int hvx_worker_pool_workers(void *) { return 6; }
static int hvx_rmsnorm_rows_f32(const float *x, float *y, uint32_t M,
                                uint32_t K, uint32_t, const float *, float,
                                void *) {
  std::memcpy(y, x, size_t(M) * K * sizeof(float));
  return 0;
}
static int hvx_router_rows_norm_f32(const float *x, const float *, float,
                                    const float *, float *, float *y,
                                    uint32_t M, uint32_t K, uint32_t E,
                                    void *) {
  if (fail_router)
    return 1;
  for (uint32_t i = 0; i < M * E; ++i)
    y[i] = x[size_t(i / E) * K + i % E];
  return 0;
}
static int hvx_router_topk_rows_f32(const float *, const float *, uint32_t *,
                                    float *, uint32_t, uint32_t, uint32_t,
                                    uint32_t, void *) {
  return 0;
}
static int hexkl_mm_u8i4_moe_layer_run(
  WeightTable *tbl, uint8_t *, uint32_t, uint32_t, uint32_t M, uint32_t K,
  uint32_t inter, uint32_t N_out, uint32_t n_experts, const uint32_t *,
  const uint32_t *, const uint32_t *row_index, const uint32_t *row_count,
  const float *, const float *act_f32, float *out_f32, void *pool,
  hexkl_moe_scratch *scratch, uint32_t flags) {
  ++core_calls;
#include "core_enter.inc"
  if (fail_core)
    return AEE_EINVALIDFORMAT;
  const uint32_t act =
    flags & HEXKL_MOE_FLAG_GELU_TANH ? HVX_GLU_GELU_TANH : HVX_GLU_SILU;
  const uint32_t BR = HEXKL_HMX_INT8_BLOCK_N_ROW, m_pad = ROUND_UP_U32(M, BR),
                 inter_ntiles = inter / HEXKL_HMX_INT8_BLOCK_N_COL;
  const uint32_t dn_ntiles = N_out / HEXKL_HMX_INT8_BLOCK_N_COL,
                 inter_ktiles = inter / HEXKL_HMX_INT8_BLOCK_N_INNER;
  const int use_m1 = M == 1u;
  // This fixture isolates the unrotated native LFM handoff path.
  const int had_up = 0, use_had_hmx = 0;
  const bool use_tl2 = tbl->compact;
  uint32_t n_rows = 0, n_with_rows = 0;
  for (uint32_t e = 0; e < n_experts; ++e) {
    n_rows += row_count[e];
    n_with_rows += row_count[e] != 0;
  }
  const uint32_t n_compact_tl2 = tbl->compact ? n_with_rows : 0;
  int rc = AEE_SUCCESS;
#include "core_layout.inc"
  last_need = need;
  last_carved = size_t(cur - scratch->base);
  check(last_carved == need, "carve total differs from same reserve sum");
  check(!router_reuse || act_c == router_act, "router prefix moved");
  ++tests;
  // Real expert order/slot writes must not overwrite the retained prefix.
  uint32_t n_active = 0;
  // clang-format off
#include "core_slots.inc"
#include "core_copy.inc"
  // clang-format on
  check(!std::memcmp(act_direct ? act_f32 : act_c, act_f32,
                     size_t(M) * K * sizeof(float)),
        "retained input changed");
  // Arithmetic stand-in: copy the exact selected input, so baseline and
  // reuse byte equality is checked across the full activation, not samples.
  std::memcpy(out_f32, act_direct ? act_f32 : act_c,
              size_t(M) * N_out * sizeof(float));
  core_hits += router_reuse;
out:
  return rc;
}
#include "rpc.inc"

struct DspFixture {
  unsigned M;
  size_t bytes;
  nntr_hvx_session s;
  std::vector<float> input, other, output, logits, w;
  uint32_t gu[2] = {0, 2}, dn[2] = {1, 3};
  uint32_t rows[2] = {0, 1}, counts[2] = {1, 1};
  float rw[2] = {0.25f, 0.75f};
  explicit DspFixture(unsigned m = 1024) :
    M(m),
    bytes(size_t(M) * 2048 * sizeof(float)),
    input(bytes / 4),
    other(input.size()),
    output(input.size()),
    logits(size_t(M) * 128),
    w(size_t(2048) * 128) {
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = float(int(i % 257) - 128) * 0.125f;
    other = input;
    check(hexkl_moe_scratch_reserve(&s.moe_scratch,
                                    3 * bytes + (size_t(8) << 20)) == 0,
          "scratch setup");
  }
  ~DspFixture() { hexkl_moe_scratch_free(&s.moe_scratch); }
  int router(uint32_t t = 1, const float *in = nullptr, int gamma_len = 0,
             unsigned E = 64, bool alias_logits = false) {
    return nntr_hvx_router_logits_f32(
      reinterpret_cast<remote_handle64>(&s), M, 2048, E, 0.0f,
      gamma_len ? w.data() : nullptr, gamma_len, w.data(), int(2048 * E),
      in ? in : input.data(), int(input.size()), 0, htp_router_handoff_wire(t),
      nullptr, 0, alias_logits ? input.data() : logits.data(), int(M * E),
      nullptr, 0, nullptr, 0);
  }
  int moe(uint32_t t = 1, const float *in = nullptr, int bad_len = 0,
          unsigned I = 1536, unsigned act_override = 0) {
    return nntr_hvx_mm_u8i4_moe_layer(
      reinterpret_cast<remote_handle64>(&s), M, 2048, I, 2048,
      act_override ? act_override : htp_router_handoff_wire(t), gu, 2, dn, 2,
      rows, 2, counts, 2, rw, 2, in ? in : input.data(),
      int(input.size()) - bad_len, output.data(), int(output.size()));
  }
};

static void host_checks() {
  constexpr unsigned M = 1024;
  const size_t bytes = size_t(M) * 2048 * sizeof(float);
  std::vector<float> x(bytes / 4, 1.25f), y(x.size(), 2.5f);
  for (unsigned mode = 0; mode < 16; ++mode) {
    Backend b;
    b.stage(
      b.x2_pool_,
      64); // A later writable request must invalidate even without growth.
    auto &first = b.stageInput(b.act_pool_, x.data(), bytes);
    auto *ion = first.data();
    const uint32_t t = b.publish(7, ion, bytes);
    const unsigned alloc_before = ion_allocs;
    const size_t copy_before = ion_copy_bytes;
    if (mode == 1)
      x[0] += 1.0f; // sampled CPU mutation triggers real ION copy.
    if (mode == 2)
      x[100] += 1.0f; // unsampled mutation leaves baseline ION unchanged.
    if (mode == 3)
      b.stage(b.x2_pool_, 64); // interposed writable owner.
    if (mode == 4)
      b.beginPrompt();
    if (mode == 5)
      b.releasePrefillStaging();
    if (mode == 6)
      b.router_handoff_.epoch++;
    if (mode == 7)
      b.router_handoff_.source = y.data();
    if (mode == 8)
      b.router_handoff_.bytes--;
    if (mode == 9)
      b.router_handoff_.owner++;
    if (mode == 10) {
      auto &o = b.stage(b.out_pool_, bytes);
      std::memcpy(o.data(), y.data(), bytes);
      b.copyOutOf(b.out_pool_, y.data(), bytes);
      b.publish(7, ion, bytes);
    }
    if (mode == 11)
      HtpProfile::global().value = 2;
    if (mode == 12) {
      auto &o = b.stage(b.out_pool_, bytes);
      std::memcpy(o.data(), ion, bytes);
      b.out_pool_.copied_out[Backend::stageClass(bytes)] = {ion, bytes, true};
      b.publish(7, ion, bytes);
    }
    if (mode == 13)
      b.fit(b.act_pool_.by_class[Backend::stageClass(bytes)], bytes + 8192);
    if (mode == 14)
      b.router_handoff_.ticket++;
    uint32_t wire = b.select(7, mode == 10 ? y.data() : x.data(), M, 2048,
                             mode == 15 ? 1472 : 1536);
    const bool hit = mode == 0 || mode == 2 || mode == 14;
    // A different valid producer ticket is permitted on host; DSP must
    // reject its mismatch. The host never authenticates DSP's stamp alone.
    check((htp_router_handoff_ticket(wire) != 0) == hit, "ARM lifecycle guard");
    if (hit) {
      auto *actual = b.act_pool_.by_class[Backend::stageClass(bytes)]->data();
      check(actual == ion, "no-write handoff moved ION");
      check(ion_copy_bytes == copy_before, "handoff added a host input copy");
      check(ion_allocs <= alloc_before + 1,
            "handoff added activation allocation");
      if (mode == 2)
        check(reinterpret_cast<float *>(actual)[100] != x[100],
              "fixture missed unsampled baseline behavior");
    }
    check(!b.router_handoff_.ticket && !b.select(7, x.data()),
          "host stamp reused twice");
    HtpProfile::global().value = 0;
    x.assign(x.size(), 1.25f);
    ++tests;
  }
  {
    Backend b;
    auto &a = b.stageInput(b.act_pool_, x.data(), bytes);
    check(b.producer(x.data(), a.data(), y.data()) != 0,
          "producer normal guard");
    check(!b.producer(x.data(), a.data(), a.data()),
          "router output would overwrite ION input");
    check(!b.producer(x.data(), a.data(), a.data() + bytes - 4),
          "partial router output alias");
    check(!b.producer(x.data(), a.data(), y.data(), 1024, 2816, 128),
          "Gemma producer requested reuse");
    check(!b.producer(x.data(), a.data(), y.data(), 1024, 2048, 64, false),
          "opt-in off requested reuse");
    check(
      !b.producer(x.data(), a.data(), y.data(), 1024, 2048, 64, true, x.data()),
      "fused norm producer requested reuse");
    check(!b.producer(x.data(), a.data(), y.data(), 1024, 2048, 64, true,
                      nullptr, 4),
          "softmax producer requested reuse");
    check(!b.producer(x.data(), a.data(), y.data(), 1024, 2048, 64, true,
                      nullptr, 0, 5),
          "u8i8 producer requested reuse");
    tests += 8;
  }
  // Existing kept-output restoration can write ION; overlap must decline.
  {
    Backend b;
    auto &a = b.stageInput(b.act_pool_, x.data(), bytes);
    auto &o = b.stage(b.out_pool_, bytes);
    std::memcpy(o.data(), a.data(), bytes);
    reinterpret_cast<float *>(o.data())[100] = 7.0f;
    b.out_pool_.holds[Backend::stageClass(bytes)] = {a.data(), bytes, true};
    b.publish(7, a.data(), bytes);
    check(!b.select(7, x.data()), "kept restore alias did not decline");
    check(reinterpret_cast<float *>(a.data())[100] == 7.0f,
          "alias fixture did not exercise real ION write");
    ++tests;
  }
  {
    Backend b;
    auto &a = b.stageInput(b.act_pool_, x.data(), bytes);
    b.publish(7, a.data(), bytes);
    fail_ion_alloc = true;
    try {
      b.stage(b.out_pool_, bytes);
      check(false, "missing allocation exception");
    } catch (const std::bad_alloc &) {
    }
    fail_ion_alloc = false;
    check(!b.router_handoff_.ticket, "failed writer retained stamp");
    ++tests;
  }
}
static void state_checks() {
  nntr_hvx_session flags_session;
  flags_session.moe_flags = HEXKL_MOE_FLAG_HAD_MASK;
  check(moe_call_flags(&flags_session, 0u) == HEXKL_MOE_FLAG_HAD_MASK,
        "expert call lost rotation flags");
  check(moe_call_flags(&flags_session, 2u) == 0u,
        "dense silu call retained rotation flags");
  check(moe_call_flags(&flags_session, 3u) == HEXKL_MOE_FLAG_GELU_TANH,
        "dense gelu call lost activation or retained rotation flags");
  tests += 3;
  htp_router_handoff s{};
  uint32_t data[4] = {0};
  const uint32_t t = htp_router_handoff_next(&s);
  htp_router_handoff_publish(&s, t, 7, data, sizeof(data));
  check(!htp_router_handoff_take(&s, 7, data, sizeof(data), t + 1),
        "wrong ticket accepted");
  check(!htp_router_handoff_take(&s, 7, data, sizeof(data), t),
        "mismatch did not consume");
  htp_router_handoff_publish(&s, t, 7, data, sizeof(data));
  check(htp_router_handoff_take(&s, 7, data, sizeof(data), t),
        "valid state refused");
  check(!htp_router_handoff_take(&s, 7, data, sizeof(data), t),
        "state reused twice");
  s.next_ticket = HTP_ROUTER_HANDOFF_MAX - 1;
  check(htp_router_handoff_next(&s) == HTP_ROUTER_HANDOFF_MAX &&
          !htp_router_handoff_next(&s) && !htp_router_handoff_next(&s),
        "ticket wrap");
  for (uint32_t n : {0u, 1u, 2u, HTP_ROUTER_HANDOFF_MAX})
    check(htp_router_handoff_ticket(htp_router_handoff_wire(n)) == n,
          "wire roundtrip");
  check(!htp_router_handoff_ticket(1) &&
          !htp_router_handoff_ticket(0x80000003u),
        "bad wire");
  check(!htp_router_handoff_shape(1, 2048) &&
          !htp_router_handoff_shape(127, 2048) &&
          !htp_router_handoff_shape(4097, 2048) &&
          !htp_router_handoff_shape(1024, 2816),
        "decode/Gemma/long shape guard");
  tests += 9;
  htp_router_handoff_scratch scratch{};
  for (unsigned mode = 0; mode < 5; ++mode) {
    htp_router_handoff_scratch_publish(&scratch, t, data, sizeof(data));
    if (mode == 0)
      scratch.epoch++;
    check(!htp_router_handoff_scratch_take(
            &scratch, mode == 1 ? data + 1 : data,
            mode == 2 ? sizeof(data) - 1 : sizeof(data),
            mode == 3   ? t + 1
            : mode == 4 ? 0
                        : t),
          "DSP scratch mismatch accepted");
    check(!htp_router_handoff_scratch_take(&scratch, data, sizeof(data), t),
          "DSP scratch mismatch did not consume");
    ++tests;
  }
}

#if defined(__linux__)
/** Real shared host mappings model three addresses for one logical ION:
 * ARM, router RPC and plain MoE RPC. They share bytes, not copied snapshots.
 * This exercises mapping-independent ownership; it is not DSP coherency proof.
 */
struct SharedMappings {
  FILE *file = nullptr;
  size_t bytes;
  float *arm = nullptr, *producer = nullptr, *consumer = nullptr;
  explicit SharedMappings(size_t n) : bytes(n) {
    file = std::tmpfile();
    check(file && !ftruncate(fileno(file), off_t(bytes)), "mapping backing");
    auto map = [&]() {
      void *p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fileno(file), 0);
      check(p != MAP_FAILED, "shared mapping failed");
      return static_cast<float *>(p);
    };
    arm = map();
    producer = map();
    consumer = map();
    check(arm != producer && arm != consumer && producer != consumer,
          "RPC mappings must have distinct virtual addresses");
  }
  ~SharedMappings() {
    munmap(consumer, bytes);
    munmap(producer, bytes);
    munmap(arm, bytes);
    std::fclose(file);
  }
};
static void remapped_ion_checks() {
  for (unsigned M : {128u, 1024u, 2048u}) {
    for (unsigned mode = 0; mode < 7; ++mode) {
      DspFixture f(M);
      SharedMappings mappings(f.bytes);
      Backend b;
      auto &a = b.stageInput(b.act_pool_, f.input.data(), f.bytes);
      std::memcpy(mappings.arm, a.data(), f.bytes);
      a.mapped = reinterpret_cast<uint8_t *>(mappings.arm);
      const uint32_t t =
        b.producer(f.input.data(), a.data(), f.logits.data(), f.M);
      check(t && !f.router(t, mappings.producer), "mapped producer setup");
      htp_router_handoff_publish(&b.router_handoff_, t, 7, a.data(), f.bytes);
      if (mode == 1)
        f.input[0] += 1.0f; // Actual staging copy writes all shared mappings.
      if (mode == 2)
        b.router_handoff_.source = f.other.data(); // Source mismatch alone.
      if (mode == 3)
        hexkl_moe_scratch_reserve(&f.s.moe_scratch, f.s.moe_scratch.cap);
      if (mode == 4)
        b.beginPrompt();
      if (mode == 5) {
        auto &o = b.stage(b.out_pool_, f.bytes);
        std::memcpy(o.data(), f.other.data(), f.bytes);
        b.copyOutOf(b.out_pool_, f.other.data(), f.bytes);
        htp_router_handoff_publish(&b.router_handoff_, t, 7, a.data(), f.bytes);
      }
      const uint32_t wire = b.select(
        mode == 6 ? 8 : 7, mode == 5 ? f.other.data() : f.input.data(), M);
      const unsigned hits = core_hits;
      const unsigned allocations = dsp_allocs;
      const size_t copied = dsp_copy_bytes;
      const auto *actual =
        b.act_pool_.by_class[Backend::stageClass(f.bytes)]->data();
      const float *rpc_input =
        mode == 5 ? reinterpret_cast<const float *>(actual) : mappings.consumer;
      const bool hit = mode == 0;
      check(!f.moe(htp_router_handoff_ticket(wire), rpc_input) &&
              core_hits == hits + unsigned(hit) && dsp_allocs == allocations &&
              dsp_copy_bytes == copied + (hit ? 0 : f.bytes) &&
              !std::memcmp(f.output.data(), actual, f.bytes),
            "remapped ION ownership/input parity");
      check((wire != 0) == (mode == 0 || mode == 3),
            "ARM authorized a different/mutated ION source");
      check(!f.moe(htp_router_handoff_ticket(wire), rpc_input) &&
              core_hits == hits + unsigned(hit),
            "remapped ION ticket reused twice");
      ++tests;
    }
  }
}
#endif
static void direct_writer_checks() {
  for (unsigned mode = 0; mode < 8; ++mode) {
    DspFixture f(128);
    Backend b;
    auto &a = b.stageInput(b.act_pool_, f.input.data(), f.bytes);
    float *ion = reinterpret_cast<float *>(a.data());
    const uint32_t t = b.producer(f.input.data(), ion, f.logits.data(), f.M);
    check(t && !f.router(t, ion), "direct writer producer setup");
    htp_router_handoff_publish(&b.router_handoff_, t, 7, ion, f.bytes);
    std::vector<uint16_t> kv(64u * 32u, 0);
    direct_rpc_failure = mode == 3 ? 1 : 0;
    if (mode == 0 || mode == 3) {
      const bool ok =
        b.sdpa_q_kvcache(1, 0, 0, 32, kv.data(), kv.data(), f.input.data(), 32,
                         1, 0, 1, 1, 1, 32, 0, 0, nullptr, ion + 100, 32);
      check(ok == (mode == 0), "quant attention return changed");
    } else if (mode < 3) {
      const unsigned nq = mode == 1 ? 1 : 64;
      check(b.sdpa_fp16_kvcache(f.input.data(), 32, kv.data(), kv.data(), 32,
                                nq, 0, nq, 1, 1, 32, 0, 0, nullptr, ion + 100,
                                32),
            "fp16 attention failed");
    } else if (mode == 4) {
      b.have_id_ = true;
      check(b.take_decode_token_id(reinterpret_cast<unsigned *>(ion + 100)),
            "token id writer failed");
    } else {
      b.large_writer_guard(mode - 5);
      check(!b.router_handoff_.ticket, "large writer boundary retained stamp");
      ion[100] += 1.0f; // Simulate its already-authorized destination write.
    }
    direct_rpc_failure = 0;
    const unsigned hits = core_hits;
    const uint32_t wire = b.select(7, f.input.data(), f.M);
    check(!f.moe(htp_router_handoff_ticket(wire), ion) && core_hits == hits &&
            !std::memcmp(f.output.data(), ion, f.bytes),
          "direct output writer retained stale router prefix");
    check(!wire, "interposed writer requested stale stamp");
    ++tests;
  }
}
static void quant_attention_staging_checks() {
  for (unsigned rows : {1u, 127u, 128u, 129u, 1024u}) {
    Backend b;
    const size_t count = size_t(rows) * 32, bytes = count * sizeof(float);
    std::vector<float> q(count), out(count, -1.0f);
    for (size_t i = 0; i < count; ++i)
      q[i] = float(i % 47) / 13.0f;
    const unsigned alloc_before = ion_allocs;
    for (unsigned pass = 0; pass < 3; ++pass) {
      const bool fail = pass == 2;
      direct_rpc_failure = fail ? 1 : 0;
      std::fill(out.begin(), out.end(), -1.0f);
      const size_t copied = ion_copy_bytes;
      check(b.sdpa_q_kvcache(1, 0, 0, 32, nullptr, nullptr, q.data(), 32, rows,
                             0, rows, 1, 1, 32, 0, 0, nullptr, out.data(),
                             32) == !fail,
            "quant attention staging return");
      check(!std::memcmp(direct_q_input, q.data(), bytes),
            "quant attention input copy differs");
      check((direct_q_input != q.data()) == (rows >= 128),
            "quant attention staging threshold");
      check(ion_allocs == alloc_before + (rows >= 128 ? 2u : 0u),
            "quant attention repeated staging allocated again");
      check(ion_copy_bytes - copied ==
              (rows >= 128 ? bytes * (fail ? 1 : 2) : 0),
            "quant attention input/output copy count");
      for (size_t i = 0; i < count; ++i)
        check(out[i] == (fail && rows >= 128 ? -1.0f : 7.25f + float(i % 13)),
              "quant attention output/failure copy differs");
      ++tests;
    }
    direct_rpc_failure = 0;
  }
}
static void sync_ion_checks() {
  for (unsigned M : {128u, 1024u}) {
    DspFixture f(M);
    Backend b;
    auto &a = b.stageInput(b.act_pool_, f.input.data(), f.bytes);
    auto &o = b.stage(b.out_pool_, f.bytes);
    float *ion = reinterpret_cast<float *>(a.data());
    std::memcpy(o.data(), ion, f.bytes);
    reinterpret_cast<float *>(o.data())[100] += 1.0f;
    // A retained alias can publish into an address that is itself ION.
    b.ion_alias_[0] = {ion, o.data(), f.bytes};
    const uint32_t ticket = b.producer(f.input.data(), ion, f.logits.data(), M);
    check(ticket && !f.router(ticket, ion), "sync alias router setup");
    htp_router_handoff_publish(&b.router_handoff_, ticket, 7, ion, f.bytes);
    b.sync_to_host(ion, f.bytes);
    check(ion[100] != f.input[100], "sync did not exercise actual ION write");
    const uint32_t wire = b.select(7, f.input.data(), M);
    const unsigned hits = core_hits;
    check(!f.moe(htp_router_handoff_ticket(wire), ion) && core_hits == hits &&
            !std::memcmp(f.output.data(), ion, f.bytes),
          "sync_to_host ION write retained stale router bytes");
    check(wire == 0, "sync ION write retained ticket");
    ++tests;
  }
}
static void direct_ion_checks() {
  for (unsigned M : {128u, 1024u}) {
    DspFixture f(M);
    Backend b;
    auto &a = b.stageInput(b.act_pool_, f.input.data(), f.bytes);
    float *ion = reinterpret_cast<float *>(a.data());
    b.stageInput(b.act_pool_, ion,
                 f.bytes); // Source is the caller-writable ION itself.
    const uint32_t ticket = b.producer(ion, ion, f.logits.data(), M);
    check(!f.router(ticket, ion), "direct ION router failed");
    htp_router_handoff_publish(&b.router_handoff_, ticket, 7, ion, f.bytes);
    ion[100] += 1.0f; // An actual ION write outside staging methods.
    const uint32_t wire = b.select(7, ion, M);
    const unsigned hits = core_hits;
    check(!f.moe(htp_router_handoff_ticket(wire), ion) && core_hits == hits &&
            !std::memcmp(f.output.data(), ion, f.bytes),
          "caller-writable ION reused stale router bytes");
    check(ticket == 0 && wire == 0, "direct ION requested a handoff ticket");
    check(!b.producer(a.data() + f.bytes - 4, a.data(), f.logits.data(), M),
          "partial caller ION source alias requested a ticket");
    tests += 2;
  }
}
static void dsp_checks() {
  for (unsigned m : {128u, 1024u, 2048u}) {
    DspFixture f(m);
    check(!f.router(21), "cached producer failed");
    const unsigned a = dsp_allocs, h = core_hits;
    const size_t before = dsp_copy_bytes;
    check(!f.moe(21) && core_hits == h + 1 && dsp_copy_bytes == before &&
            dsp_allocs == a,
          "actual router->plain RPC did not reuse");
    check(!std::memcmp(f.output.data(), f.input.data(), f.bytes),
          "full reused bytes differ");
    const size_t need = last_need;
    check(!f.moe(21) && core_hits == h + 1 && last_need == need &&
            dsp_copy_bytes == before + f.bytes,
          "second consumer did not copy");
    ++tests;
  }
  for (unsigned mode = 0; mode < 17; ++mode) {
    DspFixture f(128);
    check(!f.router(71), "producer fixture");
    if (mode == 0)
      f.s.moe_scratch.router_handoff.epoch++;
    if (mode == 1)
      hexkl_moe_scratch_reserve(&f.s.moe_scratch, f.s.moe_scratch.cap);
    if (mode == 2)
      f.s.moe_scratch.router_handoff.owner = f.other.data();
    if (mode == 3)
      f.s.moe_scratch.router_handoff.bytes--;
    if (mode == 4)
      f.s.weights_u8i4.compact = false;
    if (mode == 5)
      f.s.moe_scratch.cap = f.bytes + size_t(128) * 64 * 4;
    if (mode == 6) {
      fail_router = true;
      check(f.router(72) == AEE_EINVALIDFORMAT, "router fail");
      fail_router = false;
    }
    if (mode == 7) {
      check(f.moe(71, nullptr, 1) == AEE_EBADPARM, "length failure");
    }
    if (mode == 8) {
      fail_core = true;
      check(f.moe(71) == AEE_EINVALIDFORMAT, "kernel failure");
      fail_core = false;
    }
    if (mode == 9) {
      check(!f.router(72), "second producer");
    }
    if (mode == 10)
      check(!f.router(0), "unticketed producer");
    if (mode == 11) {
      check(check_moe_layer_args(&f.s, 0, 2048, 1536, 2048, 2, 2, 2, 2, 2, 0,
                                 0) == AEE_EBADPARM,
            "interposed invalid RPC");
    }
    if (mode == 12)
      check(!f.router(72, nullptr, 2048), "norm producer fallback");
    if (mode == 13) {
      hexkl_moe_scratch_free(&f.s.moe_scratch);
      fail_dsp_alloc = true;
      check(!f.router(72), "uncached router allocation failure fallback");
      fail_dsp_alloc = false;
    }
    if (mode == 16)
      check(!f.router(71, nullptr, 0, 128), "E128 producer");
    if (mode == 15)
      f.s.moe_scratch.router_handoff.ticket = 0; // No ARM-authorized producer.
    const unsigned h = core_hits;
    const int rc =
      f.moe(mode == 14 ? 72 : 71, mode == 15 ? f.other.data() : nullptr);
    check(rc == 0 && core_hits == h && !f.s.moe_scratch.router_handoff.ticket,
          "DSP fallback/mutation reused old scratch");
    check(!std::memcmp(f.output.data(), f.input.data(), f.bytes),
          "fallback bytes differ");
    ++tests;
  }
  {
    DspFixture f(128);
    check(!f.router(41, nullptr, 0, 64, true), "aliased router output");
    check(!f.s.moe_scratch.router_handoff.ticket,
          "aliased RPC output published stamp");
    const unsigned hits = core_hits;
    check(!f.moe(41) && core_hits == hits &&
            !std::memcmp(f.output.data(), f.input.data(), f.bytes),
          "aliased producer did not preserve baseline source bytes");
    ++tests;
  }
  {
    DspFixture f(128);
    check(!f.router(11), "bad activation setup");
    check(f.moe(11, nullptr, 0, 1536, 0x80000003u) == AEE_EBADPARM &&
            !f.s.moe_scratch.router_handoff.ticket,
          "bad activation did not consume");
    check(!f.moe(0, nullptr, 0, 1536, 1), "GELU fallback rejected");
    ++tests;
  }
}
int main() {
  state_checks();
  host_checks();
  dsp_checks();
  direct_ion_checks();
  sync_ion_checks();
  direct_writer_checks();
  quant_attention_staging_checks();
#if defined(__linux__)
  remapped_ion_checks();
#endif
  std::printf(
    "ROUTER_HANDOFF %u cases PASS; actual DSP hits=%u; all mutations/failures "
    "consumed; no new handoff activation allocation\n",
    tests, core_hits);
}

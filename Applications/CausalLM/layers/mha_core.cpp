// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   mha_core.cpp
 * @date   11 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This code is based on custom_multi_head_attention_layer.cpp.
 *         This code is a part of the break down version of the mha layer.
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <engine.h>
#include <env_compat.h>
#include <kv_ring.h> // causallm::kvRingCap (shared with models/transformer.h)
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_elementwise.h>
#include <cuda_rope.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

static std::mutex rope_init_mtx;

// Minimum prefill step_size for routing attention/RoPE onto the GPU paths.
// Below the threshold the prefill falls to the host path. The threshold no
// longer depends on head_dim (the parameter is ignored): x86 returns 1 for
// every head_dim (see the note below on the former head_dim=128 issue), and
// ARM returns 1 with NNTR_MHA_GPU on, else 32. Env override:
// NNTR_MIN_PREFILL.
static unsigned int min_prefill_thr(unsigned int head_dim) {
  static const int env = []() {
    const char *e = std::getenv("NNTR_MIN_PREFILL");
    return e ? std::atoi(e) : -1;
  }();
  if (env >= 0)
    return (unsigned int)env;

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||             \
  defined(_M_IX86)
  // Intel/CUDA: no host NEON, so the GPU wins even at tiny step_size, for every
  // head_dim. The head_dim=128 (qwen3) degeneration under sampling was NOT the
  // flash kernel -- it was the GPU RoPE doing the cos/sin rotation in fp16
  // (rope_inplace_f16), which distorted the softmax distribution; fixed to fp32
  // rotation. So GPU flash + GPU RoPE is correct for all head_dims now.
  (void)head_dim;
  return 1u;
#else
  // ARM (Adreno): the image attention path is coherent (incl. qwen3); the
  // host-NEON crossover / GPU-request behaviour is unchanged from before.
  (void)head_dim;
  // value-checked (=0 disables), so an explicit NNTR_MHA_GPU=0 turns it off
  // (env_compat.h trap).
  if (nntr_env_on("NNTR_MHA_GPU"))
    return 1u;
  return 32u;
#endif
}

// Whether a device attention arm can run in this process: the OpenCL arm
// (NNTR_MHA_GPU on the gpu engine, whose context is registered) or the CUDA
// arm (NNTR_CUDA_ATTN on the cuda engine). False on the cpu engine, whatever
// the env and the model properties say. Latched like the env gates it combines
// (the engine is fixed for the process).
static bool mha_device_attention_on() {
  static const bool on = []() {
    const std::string eng = nntr_engine_env();
    if (eng == "cpu")
      return false;
    if (eng == "cuda")
      return nntr_env_on("NNTR_CUDA_ATTN");
#if defined(ENABLE_OPENCL)
    return nntr_env_on("NNTR_MHA_GPU") &&
           nntrainer::Engine::Global().isContextRegistered("gpu");
#else
    return false;
#endif
  }();
  return on;
}

#if defined(ENABLE_OPENCL)
// OpenCL attention/blas kernel interfaces (GPU attention + clmem residency).
// Guarded so the no-OpenCL CPU build compiles the host attention path.
#include <attention_kernels.h>
#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#include <opencl_kernel.h>
#endif
#include <fp16.h>
#include <layer_context.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <thread_manager.h>
#include <util_func.h>

#include <cstdint>

#if defined(ENABLE_OPENCL)
namespace {

/**
 * @brief The FP16 operand planes an FP32-activation layer narrowed on the
 *        device, while no host reader has waited for them yet.
 * @details MHACoreLayer::incremental_forwarding fills Q/K/V there with
 * kernels it does not drain, because the device attention arms that normally
 * read them are on the same in-order queue. A host arm of
 * one_batch_incremental_forwarding (host RoPE, host attention) is the one
 * reader that is not, and it calls settle_act_stage_for_host() first.
 */
struct ActStageHostState {
  void *ptr[4] = {nullptr, nullptr, nullptr, nullptr};
  size_t bytes[4] = {0, 0, 0, 0};
  bool pending = false; ///< written on the device, not yet mapped for the host
  bool mapped = false;  ///< mapped for the host, unmap after the layer
};

thread_local ActStageHostState s_act_stage;

/**
 * @brief Map the pending FP16 operand planes for the host. Blocking, so the
 *        narrowing kernels have finished when it returns. A no-op unless an
 *        FP32-activation layer staged its operands on the device.
 */
void settle_act_stage_for_host() {
  if (!s_act_stage.pending)
    return;
  for (unsigned int i = 0; i < 4; ++i)
    nntrainer::cl_svm_map_force(s_act_stage.ptr[i], s_act_stage.bytes[i],
                                /*read_only=*/false);
  s_act_stage.pending = false;
  s_act_stage.mapped = true;
}

/**
 * @brief Holds shared-virtual-memory operands for a host reader or writer.
 * @details add() maps an operand that lives in shared virtual memory,
 * blocking, so the kernels that write it have finished and the host sees
 * their bytes; the destructor hands every mapped operand back to the device.
 * Operands in host memory are left alone, so this is free on the cpu engine.
 */
class HostSvmAccess {
public:
  HostSvmAccess() = default;
  HostSvmAccess(const HostSvmAccess &) = delete;
  HostSvmAccess &operator=(const HostSvmAccess &) = delete;
  ~HostSvmAccess() {
    for (void *p : mapped_)
      nntrainer::cl_svm_unmap_force(p);
  }

  /**
   * @brief Map @a t for the host when it is in shared virtual memory.
   * @param t operand
   * @param read_only true when the host only reads it
   */
  void add(const nntrainer::Tensor &t, bool read_only) {
    const auto md = t.getMemoryData();
    if (!md || !md->isSVM() || t.isClMem())
      return;
    void *p = t.getData();
    nntrainer::cl_svm_map_force(p, t.bytes(), read_only);
    mapped_.push_back(p);
  }

private:
  std::vector<void *> mapped_;
};

} // namespace
#endif // ENABLE_OPENCL

inline float convert_scalar(uint16_t h) {
  return nntrainer::compute_fp16_to_fp32(h);
}

// =============================================================================
// §3.8 KV OHWI Phase 1 — env-gated K-cache layout switch.
// =============================================================================
// Default K cache layout: [B, max_seq_len, num_heads_KV, head_dim] row-major
// (concat over [t, h, d]). Paper §3.8 OHWI layout reinterprets the SAME buffer
// as [B, num_heads_KV, max_seq_len, head_dim] (per-head contiguous), which is
// the convolution-weight form that attention's QK matmul wants for 1×1-conv
// kernel access. The layout is the layer's kv_layout, fixed in finalize(), and
// the host write and read go through the ops with it (KVCacheView::layout).
// The gather below stays for the OpenCL two_conv arm, which reads concat,
// until that arm moves behind the op. The GPU paths follow kv_layout too.

// Gather an OHWI K cache slice [0, cache_to) into concat layout
// [cache_to, num_heads_kv * head_dim]. dst is a freshly-allocated buffer of
// size cache_to * num_heads_kv * head_dim elements.
static inline void gather_k_ohwi_to_concat_fp16(
  const uint16_t *cache_base, // raw cache buffer
  uint16_t *dst,              // [cache_to, num_heads_kv * head_dim]
  unsigned int batch, unsigned int cache_to, unsigned int num_heads_kv,
  unsigned int head_dim, unsigned int max_seq_len) {
  const size_t HD = (size_t)num_heads_kv * head_dim;
  const size_t per_head = (size_t)max_seq_len * head_dim;
  const size_t batch_off = (size_t)batch * num_heads_kv * per_head;
  for (unsigned int h = 0; h < num_heads_kv; ++h) {
    const uint16_t *src_head = cache_base + batch_off + (size_t)h * per_head;
    uint16_t *dst_head = dst + (size_t)h * head_dim;
    for (unsigned int t = 0; t < cache_to; ++t) {
      std::memcpy(dst_head + (size_t)t * HD, src_head + (size_t)t * head_dim,
                  (size_t)head_dim * sizeof(uint16_t));
    }
  }
}

namespace causallm {

/**
 * @brief Whether this pack asked for the prefill K rotation to be drained
 * (nntr_config.json "prefill_kv_drain", default false). Set once at model
 * setup, read on the forward path; a relaxed atomic because the two are
 * ordered by the model being built before it is run, not by this variable.
 */
static std::atomic<bool> g_prefill_kv_drain{false};

void setPrefillKvDrain(bool on) {
  g_prefill_kv_drain.store(on, std::memory_order_relaxed);
}

bool prefillKvDrain() {
  return g_prefill_kv_drain.load(std::memory_order_relaxed);
}

/**
 * @brief [prefill-long] Total key span the prefill about to run will reach.
 * @details The tight-stride V image (NNTR_KV_VTIGHT) is sized from `cache_to`,
 * which under chunked prefill grows one chunk at a time. Each growth crosses
 * the 256-half image pitch quantum, and a stride change invalidates EVERY row
 * already scattered -- so a 4-chunk prefill re-scatters [0,1024), [0,2048) and
 * [0,3072) of V on top of the rows it actually writes, and blocks the host on
 * the gather+drain that precedes each one. Measured: NNTR_KV_VTIGHT=0 (no
 * tight image at all, no re-scatter) is worth 659 -> 603 ms at 1 606 tokens
 * and 1 817 -> 1 619 ms at 3 595 tokens with a 2 048 chunk.
 *
 * The prefill driver knows the whole span before the first chunk. Handing it
 * over lets the tight image be laid out ONCE, at the stride the last chunk
 * would have chosen: the texture-cache win of a tight pitch is kept and the
 * re-scatters disappear. 0 = unknown (decode, or a caller that does not set
 * it), which restores the grow-as-you-go behaviour exactly.
 */
static std::atomic<unsigned int> g_prefill_span_hint{0};

void setPrefillSpanHint(unsigned int span) {
  g_prefill_span_hint.store(span, std::memory_order_relaxed);
}

unsigned int prefillSpanHint() {
  return g_prefill_span_hint.load(std::memory_order_relaxed);
}

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
// Upload a vector<vector<_FP16>> RoPE LUT to a flat device buffer ONCE (cached
// by table identity), [num_positions * half] row-major. The per-call host LUT
// row would otherwise force a blocking host->device cudaMemcpy in cuda_rope --
// nsys showed that at 74% of API time under NNTR_CUDA_ASYNC. There are only a
// couple of distinct tables (one per head_dim), so this is a few MB, uploaded
// once. Returns a device pointer; cuda_rope's dev check then skips the mirror.
static const unsigned short *
rope_lut_device(std::vector<std::vector<_FP16>> *table, int half) {
  static std::unordered_map<const void *, unsigned short *> cache;
  static std::mutex mtx;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = cache.find(table);
  if (it != cache.end())
    return it->second;
  const size_t npos = table->size();
  std::vector<unsigned short> flat(npos * (size_t)half);
  for (size_t p = 0; p < npos; ++p)
    std::memcpy(&flat[p * half], (*table)[p].data(),
                (size_t)half * sizeof(unsigned short));
  unsigned short *dev = nullptr;
  if (cudaMalloc(&dev, flat.size() * sizeof(unsigned short)) != cudaSuccess)
    dev = nullptr;
  else
    cudaMemcpy(dev, flat.data(), flat.size() * sizeof(unsigned short),
               cudaMemcpyHostToDevice);
  cache[table] = dev;
  return dev;
}
#endif

/************************************************************** */

/**
 * @brief constructor of MHACoreLayer
 */
MHACoreLayer::MHACoreLayer() :
  mha_core_props(
    nntrainer::props::NumHeads(), props::NumHeads_KV(),
    nntrainer::props::ProjectedKeyDim(), nntrainer::props::ProjectedValueDim(),
    nntrainer::props::OutputShape(), nntrainer::props::DropOutRate(),
    nntrainer::props::ReturnAttentionWeight(),
    nntrainer::props::AverageAttentionWeight(), nntrainer::props::MaxTimestep(),
    props::SlidingWindow(), props::InitSeqLen(), props::MaxNewTokens(),
    props::RopeTheta(), props::UseRope(), props::MaxPositionEmbeddings(),
    props::UseSink(), props::RopeScalingType(), props::RopeScalingFactor(),
    props::RopePartialRotaryFactor(), props::RopeScalingMaxPositionEmbeddings(),
    props::AttnLogitSoftcapping(), props::IsCausal(), props::UseGemmAttention(),
    props::GpuDecodeAttn(), props::GpuDecodeRope(), props::GpuOhwiRope(),
    props::KvWindowRing()),
  sm(nntrainer::ActivationType::ACT_SOFTMAX),
  epsilon(1e-3),
  cache_index(0),
  num_heads_Q(0),
  num_heads_KV(0),
  head_dim(0),
  cache_shift(false) {
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

MHACoreLayer::~MHACoreLayer() {
  // Release the Adreno image-attention OHWI mirrors / image views, if any.
  // Routed through libnntrainer (release_cl_mem) so this layer needn't link
  // OpenCL directly.
#if defined(ENABLE_OPENCL)
  // release_cl_mem is OpenCL-only; the mirror handles stay null without it.
  nntrainer::release_cl_mem(k_image_ohwi);
  nntrainer::release_cl_mem(v_image_ohwi);
  nntrainer::release_cl_mem(v_image_ohwi_pp);
  nntrainer::release_cl_mem(v_image_tight);
  nntrainer::release_cl_mem(k_buf_ohwi);
  nntrainer::release_cl_mem(v_buf_ohwi);
#endif
}

/************************************************************** */

// [attn-timer] NNTR_MHA_ATTN_TIMER=1: true device time of the attention arm,
// bucketed decode/prefill x sliding/full, so "a sliding layer costs O(window),
// a full layer O(context)" is a measurement instead of a reading of the
// kernels. The queue is drained before and after every timed launch, so this
// serializes the device -- diagnostics only, and it must run with the CUDA
// decode graph off (a replayed graph does not execute this host code).
// Totals are printed at process exit.
namespace {
struct MhaAttnTimer {
  // [0]=decode-sliding [1]=decode-full [2]=prefill-sliding [3]=prefill-full
  long long ns[4] = {0, 0, 0, 0}, calls[4] = {0, 0, 0, 0};
  unsigned int last_kv = 0;
  static bool on() {
    static const bool v = nntr_env_on("NNTR_MHA_ATTN_TIMER");
    return v;
  }
  static void drain() {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    if (nntr_engine_env() == "cuda") {
      nntrainer::cuda::StreamManager::Global().finish();
      return;
    }
#endif
#if defined(ENABLE_OPENCL)
    nntrainer::cl_queue_finish();
#endif
  }
  ~MhaAttnTimer() {
    static const char *name[4] = {"decode-sliding", "decode-full",
                                  "prefill-sliding", "prefill-full"};
    for (int i = 0; i < 4; ++i)
      if (calls[i])
        std::fprintf(stderr,
                     "[MHA-ATTN-TIMER] %s %.3f ms / %lld calls = %.4f ms/call "
                     "(last N_kv=%u)\n",
                     name[i], ns[i] / 1.0e6, calls[i],
                     ns[i] / 1.0e6 / (double)calls[i], last_kv);
  }
};
MhaAttnTimer g_mha_attn_timer;
struct MhaAttnTimerScope {
  int bucket = -1;
  std::chrono::steady_clock::time_point t0;
  MhaAttnTimerScope(unsigned int n_q, unsigned int n_kv, bool sliding) {
    if (!MhaAttnTimer::on())
      return;
    bucket = (n_q == 1 ? 0 : 2) + (sliding ? 0 : 1);
    g_mha_attn_timer.last_kv = n_kv;
    MhaAttnTimer::drain();
    t0 = std::chrono::steady_clock::now();
  }
  ~MhaAttnTimerScope() {
    if (bucket < 0)
      return;
    MhaAttnTimer::drain();
    g_mha_attn_timer.ns[bucket] +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0)
        .count();
    ++g_mha_attn_timer.calls[bucket];
  }
};
} // namespace

// [kv-window-ring] Guard for an attention arm that indexes the KV cache
// LINEARLY from the logical key count. When the ring is on the cache holds only
// kv_ring_cap physical rows, so such an arm would read past the buffer (the
// GPU two_conv / image kernels) or read physical rows as if they were absolute
// (the host compute path). Only three arms modulo-map the row:
// flash_attention_prefill_f16_cl, flash_decode_f16_cl and
// cuda_attention_interleaved_fp16.
//
// causallm::kvRingArmAvailable() already refuses to turn the ring on unless one
// of those is selectable, so this is the second line: it catches the case where
// the selected arm fails at RUNTIME and the cascade would otherwise walk down
// into a linear arm. Returns true when the arm must be skipped.
// (maybe_unused: every caller sits inside an ENABLE_OPENCL block.)
[[maybe_unused]] static bool mha_ring_refuses_arm(unsigned int ring_cap,
                                                  const char *arm) {
  if (ring_cap == 0)
    return false; // ring off: every arm is correct, nothing to refuse
  static bool logged = false;
  if (!logged) {
    logged = true;
    ml_loge("[kv-window-ring] skipping the '%s' attention arm: it indexes the "
            "KV cache linearly, but the cache holds only %u physical rows. "
            "Set NNTR_KV_WINDOW_RING=0 to keep the linear full-height cache.",
            arm, ring_cap);
  }
  return true;
}

// The Adreno image attention needs a K/V mirror per layer, and the mirror is an
// image2d whose size the device bounds. When it cannot be built the layer runs
// the buffer (flash) arm instead, which is several times slower; that used to
// happen with nothing logged at all. Once per process, on stderr as well as
// the log, with the numbers that decide it.
[[maybe_unused]] static void
mha_report_image_mirror_refused(size_t num_heads_KV, size_t head_dim,
                                unsigned int S_max) {
  static bool reported = false;
  if (reported)
    return;
  reported = true;
  char msg[512];
  std::snprintf(
    msg, sizeof(msg),
    "[IMG-ATTN] the K/V image mirror could not be created (KV heads=%zu, "
    "head_dim=%zu, rows=%u -> K image %zu rows before packing); the device "
    "image limits or the allocation refused it. Image attention is OFF for "
    "this process: attention runs the buffer (flash) arm -- correct, but "
    "several times slower. Lower max_seq_len, or check "
    "CL_DEVICE_IMAGE2D_MAX_HEIGHT/WIDTH.",
    num_heads_KV, head_dim, S_max, num_heads_KV * (size_t)S_max);
  ml_loge("%s", msg);
  std::fprintf(stderr, "%s\n", msg);
  std::fflush(stderr);
}

void MHACoreLayer::finalize(nntrainer::InitLayerContext &context) {

  NNTR_THROW_IF(context.getNumInputs() < 3 || context.getNumInputs() > 5,
                std::invalid_argument)
    << "Multi head Attention layer needs 3, 4, or 5 inputs. "
       "(query, key, value; mask is optional; external cache_key + cache_value "
       "for external cache mode)";

  use_external_cache = (context.getNumInputs() >= 5);
  ml::train::TensorDim::TensorType activation_type = {
    context.getFormat(), context.getActivationDataType()};
  ml::train::TensorDim empty_dim(activation_type);

  const std::vector<ml::train::TensorDim> &input_dims =
    context.getInputDimensions();
  const ml::train::TensorDim &query_dim = input_dims[INOUT_INDEX::QUERY];
  const ml::train::TensorDim &key_dim = input_dims[INOUT_INDEX::KEY];

  /** max time step of this model */
  const unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  /** max position embeddings */
  max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mha_core_props).get();

  /** local window size */
  local_window_size = std::get<props::SlidingWindow>(mha_core_props).get();

  /** [kv-window-ring] physical ring capacity for this (sliding) layer. The ring
   * is fp16 external-cache only for now; the int8 internal cache (allocated at
   * full max_seq below) is NOT ring-sized, so keep it linear. */
  const unsigned int init_seq_len =
    std::get<props::InitSeqLen>(mha_core_props).get()
      ? std::get<props::InitSeqLen>(mha_core_props).get()
      : query_dim.height(); // unset -> the plane we were handed
  // The model's default for the ring, the same boolean it sized the plane with.
  const bool ring_model_default =
    std::get<props::KvWindowRing>(mha_core_props).get();
  const unsigned int prefill_chunk =
    causallm::effectivePrefillChunk(init_seq_len, ring_model_default);
  kv_ring_cap =
    causallm::kvRingLayerEligible(
      std::get<props::UseSink>(mha_core_props).get(), use_external_cache)
      ? causallm::kvRingCap((unsigned int)local_window_size, max_timestep,
                            prefill_chunk, ring_model_default)
      : 0u;
  /** [kvring-decode] The mirror is the linear window the Adreno OHWI image
   * kernels address, not the ring: it only has to hold the rows ONE launch
   * reads, so size it from W and the chunk instead of from Wcap (which is a
   * multiple of the chunk and so up to 4x taller than any launch needs). 0 for
   * a linear layer -- the mirror sites keep deriving that from the KV cache
   * height. */
  kv_mirror_rows_want = causallm::kvMirrorRows(
    kv_ring_cap, (unsigned int)local_window_size, prefill_chunk);

  /** [determinism] This layer's attention arm, decided ONCE here from facts the
   * layer already has. The fast default puts every layer on the image path;
   * the reproducible arm (NNTR_DETERMINISTIC=1) routes a layer to the
   * buffer/flash kernels by a per-layer rule;
   * causallm::imageAttnLayer() holds the rule and the measurement behind it.
   * Every later stage of THIS layer reads this one field, so the mirror
   * prebuild, the Q staging, the two RoPE gates and the engage cannot disagree
   * -- which is the only reason the arm had to be a per-process decision
   * before. */
  img_arm_want = causallm::imageAttnLayer(local_window_size, max_timestep);
  causallm::announceDetArmOnce();

  /** attention scaling computation */
  rope_scaling_type = std::get<props::RopeScalingType>(mha_core_props).get();
  scale = std::get<props::RopeScalingFactor>(mha_core_props).get();
  rope_partial_rotary_factor =
    std::get<props::RopePartialRotaryFactor>(mha_core_props).get();
  if (rope_scaling_type == "yarn")
    original_max_position_embeddings =
      std::get<props::RopeScalingMaxPositionEmbeddings>(mha_core_props).get();

  /** query_dim = (B, 1, seq_len, H_Q * Head_Dim ) */
  const unsigned int batch_size = query_dim.batch();
  const unsigned int query_width = query_dim.width();
  /** key_dim = (B, 1, max_seq_len, H_KV * Head_Dim ) */
  const unsigned int key_width = key_dim.width();

  /**
   *  @note If NumHeads_KV is set, then use the value. Otherwise,
   *        we initialize num_heads_KV with num_heads_Q.
   */
  num_heads_Q = static_cast<size_t>(
    std::get<nntrainer::props::NumHeads>(mha_core_props).get());
  num_heads_KV =
    std::get<props::NumHeads_KV>(mha_core_props).empty()
      ? num_heads_Q
      : static_cast<size_t>(std::get<props::NumHeads_KV>(mha_core_props).get());

  // head_dim
  head_dim = static_cast<size_t>(query_width) / num_heads_Q;
  NNTR_THROW_IF(head_dim != key_width / num_heads_KV, std::invalid_argument)
    << "num_heads_Q and num_heads_KV are not properly given. Please check the "
       "num_heads_* are set correctly so that the `head_dim`s are all same for "
       "query / key / value";

  /** Weight for Sink */
  use_sink = std::get<props::UseSink>(mha_core_props).get();
  if (use_sink) {
#if ENABLE_FP16 && defined(__ANDROID__)
    nntrainer::TensorDim sink_dim(
      1, 1, 1, num_heads_Q,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       ml::train::TensorDim::DataType::FP16));
#else
    nntrainer::TensorDim sink_dim(
      1, 1, 1, num_heads_Q,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getActivationDataType()));
#endif
    sink_idx = context.requestWeight(sink_dim, nntrainer::Initializer::ZEROS,
                                     nntrainer::WeightRegularizer::NONE, 0.0f,
                                     0.0f, "sink");
  }

  attn_logit_softcapping =
    std::get<props::AttnLogitSoftcapping>(mha_core_props).get();

  /** Is Causal */
  is_causal = std::get<props::IsCausal>(mha_core_props).get();
  use_gemm_attention = std::get<props::UseGemmAttention>(mha_core_props).get();

  // Honor the LayerImpl `skip_prefill` property (parsed into layer_impl_props
  // by LayerImpl::setProperty). When set, mha_core writes its KV cache during
  // the prefill big-step but skips the (unused) prefill attention compute --
  // see the skip_prefill member doc in mha_core.h.
  if (!std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).get();

  // Paper section 3.7 int8 KV cache path. Reduces KV cache memory + read
  // bandwidth ~2x. The byte buffer is stored as UINT8; we treat the
  // bytes as signed int8 in the read/write code paths. Per-(token,
  // head) FP16 scale captures the row's amax. Env-gated so the FP16
  // baseline stays default.
  kv_int8 = std::getenv("NNTR_KV_INT8") != nullptr;
#ifndef ENABLE_FP16
  NNTR_THROW_IF(kv_int8, std::invalid_argument)
    << "NNTR_KV_INT8 requires ENABLE_FP16";
#endif

  // The K cache layout (§3.8 OHWI, NNTR_KV_OHWI): per-head rows on an fp16
  // cache that is not int8. Fixed here, once, instead of re-derived from the
  // environment and the step's dtype on every write, so a write and the read
  // that follows agree (the former write condition also asked for an FP16
  // step, which an FP32-activation x86 run never has, while the read did not).
  // Not on a KV window ring: the read gather below copies cache_to rows with
  // the linear per-head stride into a plane only min(cache_to, Wcap) rows
  // tall, so a ringed layer keeps the concat cache.
  {
    const bool cache_fp16 = use_external_cache
                              ? context.getInputDimensions()[3].getDataType() ==
                                  ml::train::TensorDim::DataType::FP16
                              : !kv_int8;
    kv_layout = (std::getenv("NNTR_KV_OHWI") != nullptr && !kv_int8 &&
                 cache_fp16 && kv_ring_cap == 0)
                  ? nntrainer::KVCacheLayout::KOhwi
                  : nntrainer::KVCacheLayout::Concat;
  }

  /** Tensor for KV-Cache (only allocate internally when not using external
   * cache) */
  if (!use_external_cache) {
    if (kv_int8) {
      ml::train::TensorDim cache_key_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::UINT8});
      ml::train::TensorDim cache_value_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::UINT8});
      ml::train::TensorDim cache_key_scale_dim(
        {batch_size, 1, max_timestep, num_heads_KV},
        {context.getFormat(), ml::train::TensorDim::DataType::FP16});
      ml::train::TensorDim cache_value_scale_dim(
        {batch_size, 1, max_timestep, num_heads_KV},
        {context.getFormat(), ml::train::TensorDim::DataType::FP16});
      tensor_idx[AttentionParams::cache_key] = context.requestTensor(
        cache_key_dim, "cache_key", nntrainer::Initializer::NONE, false,
        nntrainer::TensorLifespan::MAX_LIFESPAN);
      tensor_idx[AttentionParams::cache_value] = context.requestTensor(
        cache_value_dim, "cache_value", nntrainer::Initializer::NONE, false,
        nntrainer::TensorLifespan::MAX_LIFESPAN);
      tensor_idx[AttentionParams::cache_key_scale] = context.requestTensor(
        cache_key_scale_dim, "cache_key_scale", nntrainer::Initializer::NONE,
        false, nntrainer::TensorLifespan::MAX_LIFESPAN);
      tensor_idx[AttentionParams::cache_value_scale] =
        context.requestTensor(cache_value_scale_dim, "cache_value_scale",
                              nntrainer::Initializer::NONE, false,
                              nntrainer::TensorLifespan::MAX_LIFESPAN);
    } else {
#ifdef ENABLE_FP16
      ml::train::TensorDim cache_key_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::FP16});
      ml::train::TensorDim cache_value_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::FP16});
#else
      ml::train::TensorDim cache_key_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::UINT16});
      ml::train::TensorDim cache_value_dim(
        {batch_size, 1, max_timestep, num_heads_KV * head_dim},
        {context.getFormat(), ml::train::TensorDim::DataType::UINT16});
#endif

      tensor_idx[AttentionParams::cache_key] = context.requestTensor(
        cache_key_dim, "cache_key", nntrainer::Initializer::NONE, false,
        nntrainer::TensorLifespan::MAX_LIFESPAN);
      tensor_idx[AttentionParams::cache_value] = context.requestTensor(
        cache_value_dim, "cache_value", nntrainer::Initializer::NONE, false,
        nntrainer::TensorLifespan::MAX_LIFESPAN);
    }
  }

  theta = (float)std::get<props::RopeTheta>(mha_core_props).get();

  /** set Output dimension! - one output */
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = input_dims[0];
  output_dims[0].width(head_dim * num_heads_Q);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);

#if ENABLE_FP16 && defined(__ANDROID__)
  // Pre-build the one-time lazy state the first timed prefill otherwise pays
  // for (measured with NNTR_KV_STAGE_TPROF / clprof):
  //  - the process-wide flat RoPE LUT (~40ms, the KVST "lutcheck" segment),
  //  - the per-layer OHWI K/V mirrors + image views + the pitch-aligned
  //    tight V view (~0.5ms/layer of clCreateBuffer/clCreateImage).
  // Both are idempotent; refinalize re-entry is a no-op. Host/load-time
  // only -- no forward-path command ordering changes.
  if (theta > 0.0f && head_dim > 0 && head_dim % 2 == 0)
    ensure_rope_flat_lut();
#if defined(ENABLE_OPENCL)
  // OHWI K/V mirror + image-view prebuild (create_ohwi_kv_mirror etc.) is
  // OpenCL-only.
  // NOTE: the arm is all-or-nothing per LAYER, never per call. use_image_attn
  // is the switch the whole pipeline keys on (concat-RoPE drain mode, Q
  // staging, OHWI decode RoPE, engage); flipping it per call desyncs those
  // stages — empirically garbage even at short context, because this step's
  // K/V is written into the image rather than the buffer that aliases it, so a
  // call served by the flash kernels would read a mirror the buffer does not
  // have. Per LAYER is safe for exactly the reason per call is not: each layer
  // owns its own K/V cache and its own mirror, so the two arms never share a
  // store. Sliding windows are handled IN the OHWI kernels (local_window
  // arg).
  // [determinism] img_arm_want (= causallm::imageAttnLayer above) is the ONE
  // place this layer's arm is decided, so the mirror prebuild, the Q staging,
  // the decode-RoPE gate, the engage and the ring rule cannot disagree about
  // which arm this LAYER is on. A layer that answers false takes the
  // reproducible buffer/flash kernels and no mirror is built for it, which is
  // why the reproducible arm is also a small memory win over the fast default.
  if (img_arm_want && !kv_int8 && head_dim % 8 == 0 && !kv_mirror_init) {
    static const unsigned int mirror_cap = []() {
      const char *e = std::getenv("NNTR_KV_MIRROR_CAP");
      return e ? (unsigned int)std::atoi(e) : 0u;
    }();
    // Ring-aware (port of ed271ee7d): a W-bounded layer stores only
    // kv_ring_cap rows, so mirroring max_timestep both over-allocates and
    // disagrees with the lazy mirror-init site, which derives S_max from the
    // KV cache height (= the ring cap when ringed). Constant for the run.
    // [kvring-decode] and a ringed layer's mirror is tighter still -- one
    // launch's span, kv_mirror_rows_want -- which the lazy site reads from the
    // same field so the two still cannot disagree.
    unsigned int S_max = nntrainer::ohwi_mirror_capacity(
      kv_mirror_rows_want ? kv_mirror_rows_want
                          : (kv_ring_cap ? kv_ring_cap : max_timestep));
    if (mirror_cap >= 8 && mirror_cap < S_max)
      S_max = (mirror_cap + 7u) & ~7u;
    kv_mirror_S_max = S_max;
    // [kimg-pack] Derived from max_timestep, NOT this layer's S_max: the
    // kernels carry a single compile-time KIMG_GSH, so every K mirror in the
    // process must be packed the same way, and max_timestep is the one value
    // all layers share. 0 (today's view, instruction for instruction) whenever
    // KV heads x max_timestep already fits the device image height.
    kv_kimg_gsh = nntrainer::kimg_gsh_for(
      num_heads_KV, nntrainer::ohwi_mirror_capacity(max_timestep));
    bool m_ok = nntrainer::create_ohwi_kv_mirror(
                  /*is_v=*/false, num_heads_KV, head_dim, S_max,
                  reinterpret_cast<cl_mem *>(&k_buf_ohwi),
                  reinterpret_cast<cl_mem *>(&k_image_ohwi), kv_kimg_gsh) &&
                nntrainer::create_ohwi_kv_mirror(
                  /*is_v=*/true, num_heads_KV, head_dim, S_max,
                  reinterpret_cast<cl_mem *>(&v_buf_ohwi),
                  reinterpret_cast<cl_mem *>(&v_image_ohwi));
    kv_mirror_init = m_ok;
    if (m_ok) {
      // The env probe normally happens lazily at the first engage; with the
      // mirrors prebuilt the earlier GPU-RoPE block needs the answer too
      // (the q_stage gate runs before the engage block on the same call).
      use_image_attn = 1;
      // NNTR_KV_VIMG_PP=1: a second full-size image view over
      // the same V buffer, alternated per attention call (with the tight view
      // off), so two consecutive layers/calls never bind the same image object.
      static const bool vimg_pp = []() {
        const char *e = std::getenv("NNTR_KV_VIMG_PP");
        return e != nullptr && e[0] == '1';
      }();
      if (vimg_pp) {
        unsigned int s_full = S_max;
        void *img2 = nullptr;
        if (nntrainer::create_ohwi_v_image_view(v_buf_ohwi, num_heads_KV,
                                                head_dim, &s_full, &img2) &&
            s_full == S_max)
          v_image_ohwi_pp = img2;
        else if (img2)
          nntrainer::release_cl_mem(img2);
      }
      // Tight V view at the typical prefill capacity (the engage path grows
      // it on demand if the live sequence exceeds the guess).
      unsigned int s_tight = S_max < 1024u ? S_max : 1024u;
      void *nimg = nullptr;
      // (linear layers only: a ringed layer keeps one stride, see the engage)
      if (kv_ring_cap == 0 &&
          nntrainer::create_ohwi_v_image_view(v_buf_ohwi, num_heads_KV,
                                              head_dim, &s_tight, &nimg) &&
          s_tight < S_max) {
        v_image_tight = nimg;
        kv_v_img_S = s_tight;
      } else if (nimg) {
        nntrainer::release_cl_mem(nimg);
      }
    } else {
      use_image_attn = 0; // permanent disable; flash takes over (same as
                          // the lazy-init failure path)
      mha_report_image_mirror_refused(num_heads_KV, head_dim, S_max);
      // With the ring granted the layers no longer share one mirror size, so a
      // refusal can leave image and buffer layers mixed in one process -- the
      // combination the note above rules out -- and a ringed layer has no
      // validated buffer arm on this bundle. Fail the load, precisely.
      NNTR_THROW_IF(causallm::kvRingEnabled(ring_model_default),
                    std::runtime_error)
        << "[IMG-ATTN] the K/V image mirror could not be created while the "
           "sliding-window KV ring is on (KV heads="
        << num_heads_KV << ", head_dim=" << head_dim << ", rows=" << S_max
        << "). Lower max_seq_len, or set NNTR_KV_IMG_RING=0 to run the linear "
           "cache (the buffer attention arm then serves every layer).";
    }
  }
#endif // ENABLE_OPENCL (OHWI mirror prebuild)
#endif
}

/************************************************************** */

#ifdef ENABLE_FP16
// The RoPE LUT only needs to cover positions [0, max sequence). Models expose
// max_position_embeddings (the theoretical RoPE max, e.g. Gemma4 = 131072) but
// the live KV cache is only max_seq_len (e.g. 1024); sizing/uploading the LUT
// at 131072 is a 128x waste (the rope angle theta_j is position-independent, so
// a shorter LUT is exact). Cap to MaxTimestep (the model's max sequence).
// NNTR_ROPE_LUT_CAP overrides for experiments. (Capping the
// per-layer-transition re-upload from ~33-67MB to ~256-512KB was worth ~+500
// TPS at M=1024.)
unsigned int MHACoreLayer::rope_lut_positions() const {
  unsigned int cap =
    (unsigned int)std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();
  if (const char *e = std::getenv("NNTR_ROPE_LUT_CAP"))
    cap = (unsigned int)std::atoi(e);
  if (cap == 0 || cap > max_position_embeddings)
    cap = max_position_embeddings;
  return cap;
}

void MHACoreLayer::ensure_rope_flat_lut() {
  if (rope_cos_flat_cur != nullptr)
    return;
  // rope_freq_cache is process-wide and shared by every layer instance; the
  // lookup and the build run under rope_init_mtx.
  const std::lock_guard<std::mutex> lock(rope_init_mtx);
  const unsigned int mp = rope_lut_positions();
  precompute_freqs((int)head_dim, mp, theta, true);
  rope_cos_flat_cur =
    reinterpret_cast<const uint16_t *>(freqs_cos_fp16_flat->data());
  rope_sin_flat_cur =
    reinterpret_cast<const uint16_t *>(freqs_sin_fp16_flat->data());
  rope_flat_positions = mp;
}
#endif

/**
 * @note In external KV cache mode (use_external_cache == true), this
 *       implements the inference forward pass using cache tensors supplied
 *       as input[3] (cache_key) and input[4] (cache_value). The host (e.g.
 *       KVCacheManager, bound by CausalLM::allocateAndBindKVCache) is
 *       responsible for owning these buffers and for calling setCacheIndex()
 *       before each step to set the write position. After this call
 *       cache_index is advanced by step_size (incremental_step_size when set,
 *       else input.height()).
 *
 *       In legacy 3/4-input mode (use_external_cache == false) training is
 *       NYI and incremental_forwarding() is the inference path.
 *
 *       Input layout for external cache mode:
 *         input[0] = Q   (B, 1, step_size, num_heads_Q  * head_dim)
 *         input[1] = K   (B, 1, step_size, num_heads_KV * head_dim)
 *         input[2] = V   (B, 1, step_size, num_heads_KV * head_dim)
 *         input[3] = cache_key   (B, 1, max_seq_len, num_heads_KV * head_dim)
 *         input[4] = cache_value (B, 1, max_seq_len, num_heads_KV * head_dim)
 */
void MHACoreLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {
  if (!use_external_cache) {
    // 3/4-input (internal KV cache) mode: full-sequence self-attention.
    //
    // incremental_forwarding() is being phased out as the inference entry
    // point, so attention must run through forwarding(). The internal-cache
    // attention worker already lives in incremental_forwarding(): it sources
    // the pool-backed cache_key/cache_value via context.getTensor() — which,
    // under the GPU SVM pool, are GPU-resident — and dispatches the same
    // GPU/CPU one_batch_incremental_forwarding paths. Drive it here over the
    // whole current sequence [0, seq). This is a stateless prefill: reset the
    // write base to 0 so RoPE positions and the cache write start at the
    // sequence origin, and do not carry cache_index across forwarding() calls.
    // (Decode / cache reuse remains incremental_forwarding's job — Step 8.)
    //
    // Training for the internal-cache mode is NYI (calcGradient/calcDerivative
    // are no-ops); preserve the previous training no-op and run attention only
    // for inference, so this change adds inference support without silently
    // changing training behaviour.
    if (training)
      return;
    // Precondition: the internal cache is sized to max_timestep, and
    // incremental_forwarding rejects to > max_timestep, so the model must set
    // max_timestep > prompt length (it already does: MAX_SEQ_LEN + new tokens).
    cache_index = 0;
    const unsigned int seq =
      (unsigned int)context.getInput(INOUT_INDEX::QUERY).height();
    incremental_forwarding(context, 0, seq, training);
    return;
  }
  if (kv_int8) {
    // The internal cache_key_scale/cache_value_scale tensors are
    // allocated only when use_external_cache is false. In the Qwen3
    // setup the cache_k/cache_v inputs come from
    // Transformer::createKVCachePlaceholders, which still emits FP16
    // placeholders. Until the placeholder helper is taught to emit
    // int8 + scale (or the mha layer is rewired with 7 inputs), the
    // external-cache + kv_int8 combination is unsupported.
    throw std::runtime_error(
      "NNTR_KV_INT8 requires internal cache mode (3-input mha). The "
      "current model uses external cache placeholders (5-input mha) "
      "that are still FP16. See project_kv_int8_plan memory for the "
      "wiring work needed in transformer.cpp.");
  }

  nntrainer::Tensor &query = context.getInput(INOUT_INDEX::QUERY);
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY);
  nntrainer::Tensor &value = context.getInput(INOUT_INDEX::VALUE);
  nntrainer::Tensor &output = context.getOutput(INOUT_INDEX::OUTPUT);

  nntrainer::Tensor &cache_key = context.getInput(3);
  nntrainer::Tensor &cache_value = context.getInput(4);

  nntrainer::Tensor sink;
  if (use_sink) {
    sink = context.getWeight(sink_idx);
  }

  unsigned int step_size = (incremental_step_size > 0)
                             ? incremental_step_size
                             : (unsigned int)query.height();
  unsigned int from = cache_index;
  unsigned int to = cache_index + step_size;

  auto get_step_dim = [step_size](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(step_size);
    return step_dim;
  };

  ml::train::TensorDim query_dim = query.getDim();
  ml::train::TensorDim key_dim = key.getDim();
  ml::train::TensorDim value_dim = value.getDim();
  ml::train::TensorDim output_dim = output.getDim();
  ml::train::TensorDim cache_key_dim = cache_key.getDim();
  ml::train::TensorDim cache_value_dim = cache_value.getDim();

  ml::train::TensorDim query_step_dim = get_step_dim(query_dim);
  ml::train::TensorDim key_step_dim = get_step_dim(key_dim);
  ml::train::TensorDim value_step_dim = get_step_dim(value_dim);
  ml::train::TensorDim output_step_dim = get_step_dim(output_dim);
  ml::train::TensorDim cache_key_step_dim = get_step_dim(cache_key_dim);
  ml::train::TensorDim cache_value_step_dim = get_step_dim(cache_value_dim);

  unsigned int batch_size = query_dim.batch();
  for (unsigned int batch = 0; batch < batch_size; ++batch) {
    nntrainer::Tensor query_step = query.getSharedDataTensor(
      query_step_dim, batch * query_dim.getFeatureLen(), true);
    nntrainer::Tensor key_step = key.getSharedDataTensor(
      key_step_dim, batch * key_dim.getFeatureLen(), true);
    nntrainer::Tensor value_step = value.getSharedDataTensor(
      value_step_dim, batch * value_dim.getFeatureLen(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, batch * output_dim.getFeatureLen(), true);

    if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
      // FP32 operands: the device route, else the host arms with their
      // operands held for the host (see incremental_forwarding()).
      if (attention_fp32_on_device(
            batch, from, from, to, query_step, key_step, value_step,
            output_step, cache_key, cache_value, cache_key_dim,
            cache_key_step_dim, cache_value_dim, cache_value_step_dim))
        continue;
      // The host arms below read Q/K/V and write O on the host: wait for their
      // producers and hold the planes for the host until the layer is done.
      HostSvmAccess host_io;
      host_io.add(query_step, /*read_only=*/true);
      host_io.add(key_step, /*read_only=*/true);
      host_io.add(value_step, /*read_only=*/true);
      host_io.add(output_step, /*read_only=*/false);
#endif
#if ENABLE_FP16 && defined(__ANDROID__)
      nntrainer::TensorDim Q_step_dim = query_step_dim;
      nntrainer::TensorDim K_step_dim = key_step_dim;
      nntrainer::TensorDim V_step_dim = value_step_dim;
      nntrainer::TensorDim O_step_dim = output_step_dim;
      Q_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      K_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      V_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      O_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);

      nntrainer::Tensor Q_step = nntrainer::Tensor(Q_step_dim, true);
      nntrainer::Tensor K_step = nntrainer::Tensor(K_step_dim, true);
      nntrainer::Tensor V_step = nntrainer::Tensor(V_step_dim, true);
      nntrainer::Tensor O_step = nntrainer::Tensor(O_step_dim, true);

      Q_step.copyData(query_step);
      K_step.copyData(key_step);
      V_step.copyData(value_step);

      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, from, from, to, Q_step, K_step, V_step, O_step, cache_key,
          cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
          cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(batch, from, from, to, Q_step, K_step,
                                         V_step, O_step, cache_key, cache_value,
                                         cache_key_dim, cache_key_step_dim,
                                         cache_value_dim, cache_value_step_dim);
      }
      output_step.copyData(O_step);
#else
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(
          batch, from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim);
      }
#endif
    } else {
      one_batch_incremental_forwarding(
        batch, from, from, to, query_step, key_step, value_step, output_step,
        cache_key, cache_value, cache_key_dim, cache_key_step_dim,
        cache_value_dim, cache_value_step_dim);
    }
  }

  cache_index += step_size;
}

/**
 * @note This incremental_forwarding method is invoked for inference mode.
 *       It processes the step rows [_from, _to), one row (decode) or
 *       several (prefill chunk).
 */
// kv_int8 (internal cache mode only): the INT8 cache + FP16 scale tensors are
// allocated in finalize() and consumed by the int8 write/read paths below.
void MHACoreLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int _from, unsigned int _to,
                                          bool training) {
  // External KV cache path: from/to are interpreted as the absolute write
  // position; route through forwarding() which reads cache_key/cache_value
  // from input slots 3/4. forwarding() advances cache_index internally.
  if (use_external_cache) {
    cache_index = _from;
    incremental_step_size = _to - _from;
    forwarding(context, training);
    incremental_step_size = 0;
    return;
  }
  if (kv_int8) {
    // Internal cache mode: scale tensors were allocated in finalize().
    // Plumb them through to one_batch_incremental_forwarding via member
    // pointers (no signature churn on the per-batch entry point).
    kv_int8_key_scale =
      &context.getTensor(tensor_idx[AttentionParams::cache_key_scale]);
    kv_int8_value_scale =
      &context.getTensor(tensor_idx[AttentionParams::cache_value_scale]);
  }

  /// @todo replace step_size into input height
  unsigned int step_size = _to - _from;

  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  unsigned int from = _from;
  unsigned int to = _to;

  // `to` is an exclusive end index, so to == max_timestep exactly fills the
  // cache and is legal; only to > max_timestep overflows it.
  if (to > max_timestep) {
    // initial forwarding
    if (!_from) {
      throw std::invalid_argument(
        "to shouldn't greater than max_timestep for initial forwarding");
    } else {
      throw std::runtime_error("NYI: cache shift is not available");
      // exceeds the kv_cache size
      // KV_cache is shifted!
      cache_shift = true;
      from = max_timestep - 1;
      to = max_timestep;
    }
  }

  // util fn to compute tensor dimension for one step.
  auto get_step_dim = [step_size](const ml::train::TensorDim &dim) {
    auto step_dim = dim;
    step_dim.batch(1);
    step_dim.height(step_size);
    return step_dim;
  };

  /** incremental forwarding for each batch */
  nntrainer::Tensor &query =
    context.getInput(INOUT_INDEX::QUERY); // projected query
  nntrainer::Tensor &key = context.getInput(INOUT_INDEX::KEY); // projected key
  nntrainer::Tensor &value =
    context.getInput(INOUT_INDEX::VALUE); // projected value
  nntrainer::Tensor &output =
    context.getOutput(INOUT_INDEX::OUTPUT); // output to be projected

  nntrainer::Tensor &cache_key =
    context.getTensor(tensor_idx[AttentionParams::cache_key]);
  nntrainer::Tensor &cache_value =
    context.getTensor(tensor_idx[AttentionParams::cache_value]);

  nntrainer::Tensor sink;
  if (use_sink) {
    sink = context.getWeight(sink_idx);
  }

  ml::train::TensorDim query_dim =
    query.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim key_dim =
    key.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim value_dim =
    value.getDim(); // (B, 1, seq_len, n_heads_KV * head_dim)
  ml::train::TensorDim output_dim =
    output.getDim(); // (B, 1, seq_len, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_dim =
    cache_key.getDim(); // (B, 1, max_timestep, n_heads_KV * head_dim)
  ml::train::TensorDim cache_value_dim =
    cache_value.getDim(); // (B, 1, max_timestep, n_heads_KV * head_dim)

  ml::train::TensorDim query_step_dim =
    get_step_dim(query_dim); // (1, 1, step_size, n_heads_Q * head_dim)
  ml::train::TensorDim key_step_dim = get_step_dim(key_dim);
  ml::train::TensorDim value_step_dim = get_step_dim(value_dim);
  ml::train::TensorDim output_step_dim =
    get_step_dim(output_dim); // (1, 1, step_size, n_heads_Q * head_dim)
  ml::train::TensorDim cache_key_step_dim =
    get_step_dim(cache_key_dim); // (1, 1, step_size, n_heads_KV * head_dim)

  ml::train::TensorDim cache_value_step_dim =
    get_step_dim(cache_value_dim); // (1, 1, step_size, n_heads_KV * head_dim)

  unsigned int batch_size = query_dim.batch();
  // do the incremental forwarding
  for (unsigned int batch = 0; batch < batch_size; ++batch) {

    // preparing step tensors
    nntrainer::Tensor query_step = query.getSharedDataTensor(
      query_step_dim, batch * query_dim.getFeatureLen(), true);
    nntrainer::Tensor key_step = key.getSharedDataTensor(
      key_step_dim, batch * key_dim.getFeatureLen(), true);
    nntrainer::Tensor value_step = value.getSharedDataTensor(
      value_step_dim, batch * value_dim.getFeatureLen(), true);
    nntrainer::Tensor output_step = output.getSharedDataTensor(
      output_step_dim, batch * output_dim.getFeatureLen(), true);

    if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
      // FP32-activation model with the device attention on. The device arms
      // (RoPE, image / flash attention) take FP16 operands in shared virtual
      // memory, and this layer's operands are FP32. The host path below reads
      // Q/K/V while the projections that write them can still be in the queue
      // (V comes straight from its projection, so it was read stale), and on
      // ARM its FP16 copies are host heap that no device arm can take, so
      // decode also lands on the slow host arm. Narrow on the device into
      // FP16 planes instead, run the same path as an FP16-activation model and
      // widen O back on the device.
      if (attention_fp32_on_device(
            batch, _from, from, to, query_step, key_step, value_step,
            output_step, cache_key, cache_value, cache_key_dim,
            cache_key_step_dim, cache_value_dim, cache_value_step_dim))
        continue;
      // The host arms below read Q/K/V and write O on the host: wait for their
      // producers and hold the planes for the host until the layer is done.
      HostSvmAccess host_io;
      host_io.add(query_step, /*read_only=*/true);
      host_io.add(key_step, /*read_only=*/true);
      host_io.add(value_step, /*read_only=*/true);
      host_io.add(output_step, /*read_only=*/false);
#endif
#if ENABLE_FP16 && defined(__ANDROID__)
      nntrainer::TensorDim Q_step_dim = query_step_dim;
      nntrainer::TensorDim K_step_dim = key_step_dim;
      nntrainer::TensorDim V_step_dim = value_step_dim;
      nntrainer::TensorDim O_step_dim = output_step_dim;
      Q_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      K_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      V_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      O_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);

      nntrainer::Tensor Q_step = nntrainer::Tensor(Q_step_dim, true);
      nntrainer::Tensor K_step = nntrainer::Tensor(K_step_dim, true);
      nntrainer::Tensor V_step = nntrainer::Tensor(V_step_dim, true);
      nntrainer::Tensor O_step = nntrainer::Tensor(O_step_dim, true);

      Q_step.copyData(query_step);
      K_step.copyData(key_step);
      V_step.copyData(value_step);
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, _from, from, to, Q_step, K_step, V_step, O_step, cache_key,
          cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
          cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(batch, _from, from, to, Q_step, K_step,
                                         V_step, O_step, cache_key, cache_value,
                                         cache_key_dim, cache_key_step_dim,
                                         cache_value_dim, cache_value_step_dim);
      }
      output_step.copyData(O_step);
#else
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, _from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(
          batch, _from, from, to, query_step, key_step, value_step, output_step,
          cache_key, cache_value, cache_key_dim, cache_key_step_dim,
          cache_value_dim, cache_value_step_dim);
      }
#endif
    } else {
      one_batch_incremental_forwarding(
        batch, _from, from, to, query_step, key_step, value_step, output_step,
        cache_key, cache_value, cache_key_dim, cache_key_step_dim,
        cache_value_dim, cache_value_step_dim);
    }
  }

  // increase cache size
  cache_index += step_size;
}

#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
bool MHACoreLayer::attention_fp32_on_device(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query_step,
  nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
  nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
  ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim) {
  if (!use_gemm_attention || kv_int8 || use_sink ||
      !mha_device_attention_on() || nntr_engine_env() == "cuda")
    return false;
  auto svm_plane = [](const nntrainer::Tensor &t) {
    const auto md = t.getMemoryData();
    return md && md->isSVM() && !t.isClMem();
  };
  if (!svm_plane(query_step) || !svm_plane(key_step) ||
      !svm_plane(value_step) || !svm_plane(attention_output_step) ||
      !svm_plane(cache_key) || !svm_plane(cache_value) ||
      cache_key.getDataType() != ml::train::TensorDim::DataType::FP16 ||
      cache_value.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false;

  // One FP16 plane per operand, grow-only: the prefill step is the largest
  // and comes first, so decode steps reuse it.
  const nntrainer::Tensor *src[4] = {&query_step, &key_step, &value_step,
                                     &attention_output_step};
  nntrainer::Tensor half[4];
  std::shared_ptr<nntrainer::MemoryData> plane[4];
  for (unsigned int i = 0; i < 4; ++i) {
    nntrainer::TensorDim d = src[i]->getDim();
    d.setDataType(ml::train::TensorDim::DataType::FP16);
    plane[i] =
      nntrainer::fp16_act_stage_svm(i, d.getDataLen() * sizeof(uint16_t));
    if (!plane[i])
      return false;
    half[i] = nntrainer::Tensor(d, false);
    half[i].setData(plane[i], 0);
    // The op table of the source: ComputeOps::attention on these planes must
    // reach ClComputeOps (the device arms), not the CPU table a bare Tensor
    // gets.
    half[i].setContextData(src[i]->getContextData());
  }
  for (unsigned int i = 0; i < 3; ++i) {
    if (!nntrainer::gpu_cast_f32_to_f16_cl(
          src[i]->getData<float>(), half[i].getData<uint16_t>(),
          static_cast<unsigned int>(src[i]->size())))
      return false;
  }

  for (unsigned int i = 0; i < 4; ++i) {
    s_act_stage.ptr[i] = half[i].getData<uint16_t>();
    s_act_stage.bytes[i] = half[i].bytes();
  }
  s_act_stage.pending = true;
  s_act_stage.mapped = false;
  one_batch_incremental_forwarding(batch, _from, from, to, half[0], half[1],
                                   half[2], half[3], cache_key, cache_value,
                                   cache_key_dim, cache_key_step_dim,
                                   cache_value_dim, cache_value_step_dim);
  s_act_stage.pending = false;
  if (s_act_stage.mapped) {
    // A host arm took this call and mapped the planes: give them back
    // before the device reads O.
    for (unsigned int i = 0; i < 4; ++i)
      nntrainer::cl_svm_unmap_force(s_act_stage.ptr[i]);
    s_act_stage.mapped = false;
  }
  NNTR_THROW_IF(!nntrainer::gpu_cast_f16_to_f32_cl(
                  half[3].getData<uint16_t>(),
                  attention_output_step.getData<float>(),
                  static_cast<unsigned int>(attention_output_step.size())),
                std::runtime_error)
    << "mha_core: widening the FP16 attention output on the device failed";
  return true;
}
#endif

// NNTR_KV_STAGE_TPROF=1: host-side wall timing of the V-copy -> k_scatter ->
// attention-return window (attributes the 39ms/layer GPU idle the CL-event
// profiler pins between scatter_copy_f16 and k_scatter_ohwi). Prints
// accumulated per-segment totals at each multiple of 26 windows (= 1 prefill
// forward of Gemma2-2B).
static bool _kvst_on() {
  static const bool on = std::getenv("NNTR_KV_STAGE_TPROF") != nullptr;
  return on;
}
static double _kvst_now() {
#ifdef _WIN32
  // No clock_gettime on MSVC; steady_clock is the same monotonic source.
  return std::chrono::duration<double, std::milli>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
#endif
}
// _kvst_mark_scatter and its accumulator state are only fed from the
// ENABLE_FP16 && ENABLE_OPENCL GPU scatter paths below (the only call site
// is inside that guard); mirror the same guard here or the state variables
// go unused (and the function undefined-but-unused) in other configs.
#if defined(ENABLE_FP16) && defined(ENABLE_OPENCL)
static double _kvst_t0 = 0;
static double _kvst_acc01 = 0, _kvst_acc_k = 0, _kvst_acc_v = 0,
              _kvst_acc_a = 0;
static int _kvst_n = 0;
static void _kvst_mark_scatter(double t1, double tk, double tv, double t2) {
  if (_kvst_t0 == 0)
    return;
  _kvst_acc01 += t1 - _kvst_t0;
  _kvst_acc_k += tk - t1;
  _kvst_acc_v += tv - tk;
  _kvst_acc_a += t2 - tv;
  _kvst_t0 = 0;
  if (++_kvst_n % 26 == 0) {
    std::fprintf(stderr,
                 "[KVST] n=%d vcopy->scatter %.2fms k_scatter %.2fms "
                 "v_scatter %.2fms attn-call %.2fms\n",
                 _kvst_n, _kvst_acc01, _kvst_acc_k, _kvst_acc_v, _kvst_acc_a);
    std::fflush(stderr);
    _kvst_acc01 = _kvst_acc_k = _kvst_acc_v = _kvst_acc_a = 0;
  }
}
#endif

void MHACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query_step,
  nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
  nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
  ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim) {

  /**
   *
   *  cache_key
   *  +------------------------------------------+
   *  |<--cache_index-->|<--b_cache_value_step-->|
   *  +------------------------------------------+
   *                    |<-------key_step------->|
   *  |<-------------b_cached_key--------------->|
   */

  // [kv-window-ring] Fail loud if a step write would straddle the ring seam.
  // A ringed layer takes one contiguous slice per call, so the PRODUCER must
  // never issue a call that crosses an absolute multiple of the prefill chunk
  // C (Wcap is a multiple of C): CausalLM::run phases its chunks on the
  // absolute grid for exactly this reason, and decode is step == 1. cache_index
  // is the ABSOLUTE position, so it is C-aligned only on a first turn; a
  // resumed session starts anywhere. Reaching this throw means a producer fed
  // a multi-row step that was not phased -- throw rather than silently corrupt
  // the neighbouring pool region (SVM writes do not bounds-check). Covers
  // every cache-row write below (same cache_index).
  if (kv_ring_cap &&
      cacheRow(cache_index) + (size_t)cache_key_step_dim.height() >
        (size_t)kv_ring_cap) {
    throw std::runtime_error(
      "mha_core kv-window-ring: step write straddles the ring seam (absolute "
      "position " +
      std::to_string(cache_index) + " + " +
      std::to_string(cache_key_step_dim.height()) + " rows on a " +
      std::to_string(kv_ring_cap) +
      "-row ring). The caller must split multi-row steps at absolute multiples "
      "of the prefill chunk; set NNTR_KV_WINDOW_RING=0 to keep the linear "
      "cache.");
  }

  // Load Input Tensors of this batch : b_ denotes a Tensor for this batch
  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() +
      cacheRow(cache_index) * cache_key_dim.width(),
    true);
  nntrainer::Tensor b_cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim,
    batch * cache_value_dim.getFeatureLen() +
      cacheRow(cache_index) * cache_value_dim.width(),
    true);

  // [rq-scalar-off] The SAME address, expressed as (stable base, scalar row
  // offset) instead of an offset-baked pointer.
  //
  // b_cache_key_step / b_cache_value_step above are views whose OFFSET moves
  // with cache_index, so `.getData()` hands the GPU a DIFFERENT SVM pointer
  // every decode token even though the allocation never moved. A recorded
  // command queue (cl_qcom_recordable_queues) captures SVM pointer arguments
  // at record time and cannot re-bind them per replay, while it CAN override
  // scalar arguments -- so a per-token SVM pointer is a dispatch that can
  // never be replayed. The kernels that take the offset as a scalar already
  // exist (rope_inplace_f16 write_off, scatter_copy_f16_row,
  // k_scatter_ohwi / v_scatter_ohwi_t src_off); these are the stable bases and
  // scalar offsets their call sites need. Byte-identical by construction:
  // base + off is the very expression the view was built from.
  //
  // NNTR_KV_SCALAR_OFF=0 restores the offset-baked pointers (A/B arm).
  static const bool _kv_scalar_off = [] {
    const char *e = std::getenv("NNTR_KV_SCALAR_OFF");
    return !(e && e[0] == '0');
  }();
  const size_t _kc_step_off =
    (size_t)batch * cache_key_dim.getFeatureLen() +
    (size_t)cacheRow(cache_index) * cache_key_dim.width();
  const size_t _vc_step_off =
    (size_t)batch * cache_value_dim.getFeatureLen() +
    (size_t)cacheRow(cache_index) * cache_value_dim.width();

  // Static GPU_CLMEM residency: the wq/wk/wv FC outputs and the attention
  // output may live in planner cl_mem sub-buffers (class GPU_CLMEM). The GPU
  // stages bind these handles directly (RoPE / V-copy / image attention); any
  // path that touches them on the HOST (host RoPE, kv_int8 quant, CPU/NEON
  // attention, SVM-pointer kernels) must LOWER first (one blocking readback
  // into the SVM shadow) -- after a lower the handle is nulled so every later
  // consumer uses the now-fresh SVM plane. The output is RAISED back into its
  // sub-buffer whenever a non-cl_mem path wrote it (the wo FC reads cl_mem).
  // Offset-0 views only (batch 0; b_size==1 on the live path).
  // NNTR_CLMEM_MHA_OFF=1 (bisect): ignore residency handles entirely -- the
  // mha consumes the legacy SVM plane.
  // NNTR_KV_STAGE_TPROF: host time from mha entry to the rope-Q enqueue
  // (decomposes the copy_h2h->rope GPU-idle gap: executor/plumbing vs rope
  // wrapper).
  const double _mha_entry_t = _kvst_on() ? _kvst_now() : 0;

  static const bool mha_handles_off =
    std::getenv("NNTR_CLMEM_MHA_OFF") != nullptr;
  void *q_cl =
    (!mha_handles_off && query_step.getOffset() == 0 && query_step.isClMem())
      ? query_step.getClMem()
      : nullptr;
  void *k_cl =
    (!mha_handles_off && key_step.getOffset() == 0 && key_step.isClMem())
      ? key_step.getClMem()
      : nullptr;
  void *v_cl =
    (!mha_handles_off && value_step.getOffset() == 0 && value_step.isClMem())
      ? value_step.getClMem()
      : nullptr;
  void *o_cl =
    (attention_output_step.getOffset() == 0 && attention_output_step.isClMem())
      ? attention_output_step.getClMem()
      : nullptr;
  bool o_written_clmem = false;
  // Rotated-Q handle for the image attention's qk kernel. Defaults to the
  // tensor's own cl_mem (q_cl); the staged GPU-RoPE path below redirects it
  // to a dedicated q_stage temp -- writing the rotation IN-PLACE into q_cl
  // and reading the SAME handle from the next kernel without a drain flips
  // tokens on this driver (K and V survive precisely because their rope/
  // scatter chains write a DIFFERENT handle than they read; bisected
  // 2026-06-12: Q-only CLMEM flips, Q-only + a qk-side drain is clean).
  void *q_attn_clmem = q_cl;
  void *q_rope_staged = nullptr;
  // Set when the GPU RoPE read the FC output from q_cl but wrote the ROTATED Q
  // into the SVM plane (the image-attention form, see the rope call below).
  // From then on q_cl holds the UN-rotated Q and must never be lowered over
  // the SVM plane: the fallbacks under the image arm consume the plane as is.
  bool q_rotated_in_svm_plane = false;
  auto lower_q = [&]() {
#if defined(ENABLE_OPENCL)
    if (q_cl) {
      nntrainer::clmem_lower_cl(query_step, 0);
      q_cl = nullptr;
    }
#endif // clmem_lower_cl is OpenCL-only; q_cl is null without it
  };
  auto lower_kv = [&]() {
#if defined(ENABLE_OPENCL)
    if (k_cl) {
      nntrainer::clmem_lower_cl(key_step, 0);
      k_cl = nullptr;
    }
    if (v_cl) {
      nntrainer::clmem_lower_cl(value_step, 0);
      v_cl = nullptr;
    }
#endif //
  };

  // NNTR_CLMEM_MHA_LOWER=1 (bisect): lower Q/K/V at entry unconditionally --
  // the mha then consumes the legacy SVM plane (read side = baseline) while
  // the FC write side stays cl_mem. Divergence persisting under this flag
  // indicts the FC out-copy; divergence vanishing indicts the mha bindings.
  static const bool mha_lower_all =
    std::getenv("NNTR_CLMEM_MHA_LOWER") != nullptr;
  if (mha_lower_all) {
    // After a lower the bytes live in the HOST view; the GPU stages below
    // read the SVM device-side, so hand the buffers back (unmap_force) --
    // the same protocol the embedding raise path uses.
    bool any = q_cl || k_cl || v_cl;
    lower_q();
    lower_kv();
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
    // cl_svm_unmap_force lives in the FP16 OpenCL kernel block.
    if (any) {
      nntrainer::cl_svm_unmap_force(query_step.getData<uint8_t>());
      nntrainer::cl_svm_unmap_force(key_step.getData<uint8_t>());
      nntrainer::cl_svm_unmap_force(value_step.getData<uint8_t>());
    }
#else
    (void)any;
#endif
  }

  // Kernel-chain K/V staging (drain removal). When the Adreno image attention
  // consumes this step's K/V (NNTR_KV_IMG_ATTN), the GPU RoPE/copy below route
  // through a persistent cl_mem staging temp instead of draining per op:
  //   rope-K -> k_stage (cl_mem, no clFinish) -> k_scatter(src_clmem=k_stage)
  //                                        \-> side-fill k_stage -> SVM cache
  //   V copy -> SVM cache (no clFinish); v_scatter reads value_step directly.
  // The per-op clFinish drains here measured 1289ms of GPU idle per 1K
  // prefill (V-copy->k_scatter 1010ms + gemm->rope 279ms). The SVM cache
  // writes are consumed only by host decode after the lm_head lower (a full
  // queue drain), so no per-op drain is needed; the scatters and the image
  // attention read pure cl_mem chains the in-order queue serializes.
  // NNTR_MHA_CLMEM: the OHWI mirrors are the ONLY store during the prefill
  // window -- the K side-fill and the V SVM cache write (the last undrained
  // SVM writes in the mha window) are skipped, and any host/SVM slab reader
  // first gathers the missing rows back from the mirrors (sync_kv_slab,
  // one drained boundary sync at decode entry). v1 limit: prefill must fit
  // the mirror capacity (engage-gate misses beyond S_max leave the slab
  // stale) and the save_kvcache path is not synced.
  static const bool mha_clmem_mode = []() {
    // 2026-06-12 re-baseline: DEFAULT ON (inert without NNTR_KV_IMG_ATTN
    // staging; NNTR_MHA_CLMEM=0 restores the SVM side-fills).
    const char *e = std::getenv("NNTR_MHA_CLMEM");
    return !(e && e[0] == '0');
  }();
  // The cl_mem staging temps (q_stage/k_stage, tca_ensure device buffers) are
  // NOT coherent with the in-order SVM queue on Adreno: scatter/qk read garbage
  // from them at M>=32 (bisected 2026-06-18; the non-staged path, which routes
  // the GPU RoPE through the planner cl_mem q_cl in-place + the SVM cache, IS
  // coherent -- e.g. gemma4 "what is the capital of South Korea" -> "Seoul").
  // So staging is DISABLED BY DEFAULT. The non-staged path is still
  // GPU-resident (no host RoPE bounce) and, after the RoPE-LUT cap fix
  // (rope_lut_positions), is within ~2% of the broken-staged TPS at M=1024
  // (1520 vs 1556). Opt back into staging with NNTR_KV_STAGE=1 (to test an
  // SVM-backed staging fix); NNTR_NO_KV_STAGE still force-disables.
  static const bool kv_stage_on = causallm::imageAttnRequested() &&
                                  std::getenv("NNTR_KV_STAGE") != nullptr &&
                                  std::getenv("NNTR_NO_KV_STAGE") == nullptr;
  void *k_stage = nullptr;               // rope-K wrote the staging temp
  const uint16_t *v_stage_svm = nullptr; // v_scatter source (value_step)
  void *v_stage_clmem = nullptr;         // v_scatter cl_mem source (v_cl)
#if defined(ENABLE_OPENCL)
  // Set when a K/V cache write below was enqueued with a submission flush and
  // NO trailing clFinish. On the image-attention path that is coherent -- every
  // consumer is a same-queue GPU kernel -- but the SVM/host fallbacks further
  // down are not on that queue, so they drain once when this is set.
  bool kv_write_undrained = false;
#endif

  const double _mha_t_prelude = _kvst_on() ? _kvst_now() : 0;

  bool use_rope = theta > 0.0f;
  if (kv_int8) {
    // Host RoPE + host int8 quant read Q/K/V on the host: lower first.
    lower_q();
    lower_kv();
  }
  if (kv_int8) {
    // KV int8 path (paper section 3.7): Q RoPE in fp16 as usual; K (rotated
    // when RoPE is on) and V go through kv_cache_write, which quantizes each
    // (row, head) into the int8 planes with its fp16 scale.
    if (use_rope)
      apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, cache_index,
                                 true);
    write_kv_cache(batch, key_step, value_step, cache_key, cache_value,
                   cache_index, /*write_key=*/true, /*write_value=*/true,
                   /*rotate_key=*/use_rope);

    // Stash per-batch scale base pointers (row 0 of this batch) for the
    // read path: the host arms get them through KVCacheView (attend).
    const size_t scale_batch_offset =
      batch * kv_int8_key_scale->getDim().getFeatureLen();
    cur_kv_int8_key_scale_batch =
      kv_int8_key_scale->getData<uint16_t>() + scale_batch_offset;
    cur_kv_int8_value_scale_batch =
      kv_int8_value_scale->getData<uint16_t>() + scale_batch_offset;
  } else if (use_rope) {
    bool gpu_rope_done = false;
#ifdef ENABLE_FP16
#if defined(ENABLE_OPENCL) // GPU concat-RoPE (rope_inplace_f16_cl) path
    // GPU-resident RoPE (residency): rotate Q in place and K into its cache
    // slice on the device, so the activation never bounces to the host. Active
    // only when the GPU attention path will follow (same NNTR_MHA_GPU +
    // use_gemm_attention + prefill-length + FP16 gate) on the standard concat
    // FP16 path. int8/OHWI keep their host RoPE (separate scatter/layout).
    // Kill-switch: NNTR_NO_GPU_ROPE. cos/sin LUT is uploaded as a small
    // constant; Q/K stay SVM-direct when the pool is SVM.
    static const bool _gpu_rope_off =
      std::getenv("NNTR_NO_GPU_ROPE") != nullptr;
    static const bool _mha_gpu_on = nntr_env_on("NNTR_MHA_GPU");
    const bool kv_ohwi_now =
      kv_layout == nntrainer::KVCacheLayout::KOhwi &&
      key_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
      cache_key.getDataType() == ml::train::TensorDim::DataType::FP16;
    const unsigned int ROPE_MIN_PREFILL =
      min_prefill_thr((unsigned int)head_dim); // env NNTR_MIN_PREFILL
    // S3 decode: NNTR_MHA_GPU_DECODE moves the M=1 decode RoPE onto the GPU so
    // Q stays SVM-resident (q_cl handled in-place) and lower_q/lower_kv become
    // no-ops -- those are clEnqueueReadBuffer(CL_TRUE) blocking drains that
    // cost ~65 ms/token (35 layers x 2) on the host-RoPE decode path. (A)
    // GPU-RoPE-decode gate. Enabled by the NNTR_MHA_GPU_DECODE env flag (global
    // testing override) OR per-LAYER via the gpu_decode_rope property
    // (DEFAULT-ON only for models where decode GPU-RoPE is token-identical,
    // e.g. gemma4). The env read stays process-wide static; the property is
    // read PER-CALL (per-layer state) and OR'd in. The NNTR_NO_GPU_ROPE
    // kill-switch (_gpu_rope_off, checked at the if() below) still wins.
    static const bool _gpu_rope_decode_env =
      std::getenv("NNTR_MHA_GPU_DECODE") != nullptr;
    // The gpu_decode_rope property default is validated ONLY on the Intel
    // buffer/flash decode path. On the Adreno image-attention path
    // (NNTR_KV_IMG_ATTN) GPU-RoPE-at-decode produces garbage (the image KV/attn
    // consumes host-RoPE'd Q/K in a different layout), so suppress the property
    // there. The explicit NNTR_MHA_GPU_DECODE env override is unaffected.
    // [determinism] PER-LAYER, not the process env: under the default arm a
    // full-attention layer is on the buffer/flash path, where the
    // gpu_decode_rope property IS validated, so suppressing it there would cost
    // the layer its GPU decode RoPE for a hazard it does not have.
    const bool _img_arm = img_arm_want;
    const bool _gpu_rope_decode =
      _gpu_rope_decode_env ||
      (std::get<props::GpuDecodeRope>(mha_core_props).get() && !_img_arm);
    // Image path (Adreno): a GPU RoPE kernel dispatch per layer for the M=1
    // decode step is SLOWER than the trivial host rotation of a single row
    // (measured ~14.7 vs ~18 TPS); keep the GPU path for prefill (the big win)
    // and let decode fall to the host RoPE. Non-image (Intel/CUDA) unchanged.
    // NNTR_MHA_GPU_DECODE still forces the GPU decode path (_gpu_rope_decode).
    const bool _rope_len_ok =
      ((to - from) >= ROPE_MIN_PREFILL && !(_img_arm && (to - from) == 1)) ||
      (_gpu_rope_decode && (to - from) == 1);
    // The kernels bind Q and K as shared or device memory; a host-heap operand
    // (the ARM host copy of an FP32-activation layer) is neither, and binding
    // it as a shared pointer rotated nothing the attention then read.
    auto device_addressable = [](const nntrainer::Tensor &t) {
      const auto md = t.getMemoryData();
      return t.isClMem() || (md && md->isSVM());
    };
    if (!_gpu_rope_off && _mha_gpu_on && use_gemm_attention && !kv_int8 &&
        !kv_ohwi_now && _rope_len_ok &&
        query_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        device_addressable(query_step) && device_addressable(key_step)) {
      // Flat [max_pos, half_d] cos/sin LUT (fp16 bits) from the cached trig
      // table. Built once (cached by every RoPE parameter) so the GPU RoPE
      // path has no per-call flatten; rope_inplace_f16_cl uploads it once.
      const unsigned int mp = rope_lut_positions();
      if (mp >= cache_index + (to - from)) {
        ensure_rope_flat_lut();
        const double _mha_t_lut = _kvst_on() ? _kvst_now() : 0;
        // Current-slot stable pointers (one table per RoPE configuration): for
        // Gemma4 the two attention slots each keep their own resident table,
        // so the sliding<->full transition no longer rebuilds + re-uploads the
        // shared LUT every layer. The pointers are rope_freq_cache-node-stable,
        // which keeps the rope_inplace_f16_cl device-LUT upload cached per
        // slot.
        const uint16_t *cos_lut = rope_cos_flat_cur;
        const uint16_t *sin_lut = rope_sin_flat_cur;
        uint16_t *q_p =
          reinterpret_cast<uint16_t *>(query_step.getData<_FP16>());
        const uint16_t *k_p =
          reinterpret_cast<const uint16_t *>(key_step.getData<_FP16>());
        uint16_t *kc_p =
          reinterpret_cast<uint16_t *>(b_cache_key_step.getData<_FP16>());
        const bool q_svm =
          query_step.getMemoryData() && query_step.getMemoryData()->isSVM();
        const bool kc_svm = b_cache_key_step.getMemoryData() &&
                            b_cache_key_step.getMemoryData()->isSVM();
        // Rotated-K staging temp: rope-K writes a cl_mem temp (no trailing
        // drain) which feeds k_scatter + the SVM-cache side-fill as a pure
        // kernel chain (see kv_stage_on decl). Grow-only, shared across
        // layers (in-order queue serializes reuse).
        static void *k_stage_buf = nullptr;
        static size_t k_stage_cap = 0;
        const size_t k_stage_bytes =
          (size_t)(to - from) * num_heads_KV * head_dim * sizeof(uint16_t);
        void *k_out_stage = nullptr;
        if (kv_stage_on && kc_svm &&
            nntrainer::ensure_cl_stage_buf(&k_stage_buf, &k_stage_cap,
                                           k_stage_bytes))
          k_out_stage = k_stage_buf;
        // Rotated-Q staging temp (CLMEM Q only): rope-Q writes q_stage and
        // the qk kernel reads q_stage -- write-handle != read-handle, the
        // same shape that keeps the K and V chains coherent without drains
        // (see q_attn_clmem). Only when the image attention is certain to
        // engage (mirrors prebuilt, capacity check) -- otherwise rope-Q
        // keeps its in-place + SVM-out (drained) form for the fallbacks.
        static void *q_stage_buf = nullptr;
        static size_t q_stage_cap = 0;
        void *q_out_stage = nullptr;
        if (kv_stage_on && q_cl != nullptr && use_image_attn == 1 &&
            kv_mirror_init && cache_index + (to - from) <= kv_mirror_S_max) {
          const size_t q_stage_bytes =
            (size_t)(to - from) * num_heads_Q * head_dim * sizeof(uint16_t);
          if (nntrainer::ensure_cl_stage_buf(&q_stage_buf, &q_stage_cap,
                                             q_stage_bytes))
            q_out_stage = q_stage_buf;
        }
        if (_kvst_on() && _mha_entry_t > 0) {
          static double acc_pre = 0, acc_mid = 0, acc_lut = 0;
          static int n_e2r = 0;
          const double tnow = _kvst_now();
          acc_pre += _mha_t_prelude - _mha_entry_t;
          acc_mid += _mha_t_lut - _mha_t_prelude;
          acc_lut += tnow - _mha_t_lut;
          if (++n_e2r % 26 == 0) {
            std::fprintf(stderr,
                         "[KVST] entry->ropeQ n=%d prelude=%.2f "
                         "lutcheck=%.2f tail=%.2f ms\n",
                         n_e2r, acc_pre, acc_mid, acc_lut);
            std::fflush(stderr);
            acc_pre = acc_mid = acc_lut = 0;
          }
        }
        // rope-Q's trailing SVM-out drain is LOAD-BEARING when the rotation
        // lands in the same plane the consumer reads (drain_svm_out=false
        // flipped " itſelf" -> " tubes", 2026-06-12). The staged form writes
        // a DIFFERENT cl_mem handle (q_stage) than it reads, which is
        // coherent without a drain (the K/V chain shape); the qk kernel then
        // binds q_stage. Non-staged keeps the drained in-place form.
        // On the Adreno image-attention path the qk kernel reads Q from the SVM
        // plane (Q_p, q_clmem=null; see the OHWI dispatch below). Rotating INTO
        // the FC's planner cl_mem (q_cl) leaves that SVM plane stale on this
        // driver (a cl_mem write is not visible through the aliased SVM
        // pointer), so the qk reads the UN-rotated Q -> garbage. Rotate
        // straight into the SVM plane (out_clmem=null -> writes Q_p) instead:
        // read the FC output from q_cl, write the rotation to the SVM Q_p the
        // qk actually consumes. GPU-resident, no cl_mem->SVM copy.
        // (Non-image/flash keeps q_cl.)
        // The K rotation and the V copy below land in the SVM KV-cache
        // slices, and while the image attention is the live path their ONLY
        // consumers are same-queue GPU kernels (k_scatter_ohwi /
        // v_scatter_ohwi_t, then the image attention itself); the host reads
        // those slices no earlier than the lm_head lower, which drains the
        // whole queue anyway. The trailing clFinish each of them takes is
        // therefore pure GPU idle: 2 per layer, 70 per 1K prefill, 19.0 ms =
        // 5.4% of the prefill window (measured on an Adreno 840 cell). Keep
        // the submission flush, drop the drain. This is the shape the OHWI
        // DECODE branch below already ships (drain_svm_out=false / drain=false,
        // 0 steady-decode clFinish), and it is a weaker claim than rope-Q's:
        // both writes go to a handle they do not read. Non-image lanes
        // (Intel/CUDA flash, host attention) keep the drain.
        // This site -- the prefill K rotation, which writes
        // b_cache_key_step's SVM cache slice and lets k_scatter_ohwi_cl read
        // that slice out of a LATER submission -- is where one model on this
        // chain stops being reproducible under greedy decode. Without the
        // drain the run produces a different logits row for the first sampled
        // token about one time in three, and the token sequence is a
        // deterministic function of that row, so the generated text differs
        // run to run at temperature 0. With it, 10 runs of 10 give one row and
        // one answer.
        //
        // The drain is therefore DEFAULT-ON only where that has been observed,
        // and the pack SAYS SO: nntr_config.json "prefill_kv_drain": true.
        // Elsewhere the no-drain above stands, because the drain is expensive
        // on the prefill it was removed from -- measured -41 % on gemma4
        // (4274 -> 2521 TPS, interleaved, one binary) while gemma4's golden
        // held 21 of 21 runs without it.
        //
        // Why a declaration and not an inference. The first gate this took was
        // skip_prefill, on the argument that it is a configuration property
        // rather than a model name -- but it is a property about something
        // else (whether the first sampled token runs as a decode-shaped step),
        // and packs that do not need the drain set it too, so the proxy
        // silently spent their prefill gain. A flag whose only meaning is
        // "this pack was observed to need the drain" cannot mis-fire that way:
        // it says exactly what it knows and nothing more.
        //
        // It is still a MANIFESTATION record, not the risk condition. The risk
        // is a consumer reading an SVM write from an earlier submission, which
        // every model on this chain does; a pack that does not set the flag is
        // unobserved, not proven safe. Deciding the risk condition
        // structurally, where the chain is built, is the per-dispatch hazard
        // check (Kernel::noteUndrainedSvmPlane, opt-in via
        // NNTR_SVM_HAZARD_DRAIN=1) and is what should retire this flag.
        // NNTR_KV_PREFILL_NODRAIN overrides either way and outranks the pack:
        // =1 no drain, =0 drain.
        static const int _kv_nodrain_override = [] {
          const char *e = std::getenv("NNTR_KV_PREFILL_NODRAIN");
          return e == nullptr ? -1 : (std::atoi(e) != 0 ? 1 : 0);
        }();
        const bool _kv_nodrain_env = _kv_nodrain_override >= 0
                                       ? (_kv_nodrain_override == 1)
                                       : !prefillKvDrain();
        const bool kv_chain_gpu_only =
          _kv_nodrain_env && _img_arm && use_image_attn == 1;
        void *q_rope_out =
          q_out_stage != nullptr ? q_out_stage : (_img_arm ? nullptr : q_cl);
        bool ok =
          nntrainer::rope_inplace_f16_cl(
            q_p, q_p, cos_lut, sin_lut, to - from, num_heads_Q, head_dim,
            cache_index, mp, q_svm, /*in_clmem=*/q_cl,
            /*out_clmem=*/q_rope_out,
            // Image path: rotation lands in the SVM plane the same-queue
            // qk kernel reads next; the in-order queue serializes it, so
            // only a submission flush is needed (no full clFinish drain).
            // This is the pre-regression cost (a drain here cost ~5 TPS
            // of decode). Non-image keeps the default drain.
            /*drain_svm_out=*/!_img_arm) &&
          nntrainer::rope_inplace_f16_cl(
            k_p, kc_p, cos_lut, sin_lut, to - from, num_heads_KV, head_dim,
            cache_index, mp, kc_svm, /*in_clmem=*/k_cl,
            /*out_clmem=*/k_out_stage,
            // Staged (k_out_stage != null) already skips the drain by writing
            // a cl_mem; unstaged writes the SVM cache slice and, on the image
            // chain, has the same all-GPU consumer set.
            /*drain_svm_out=*/!kv_chain_gpu_only);
        if (ok && k_out_stage == nullptr && kv_chain_gpu_only) {
          kv_write_undrained = true;
          // The rotation above bound an OFFSET-BAKED destination (kc_p = the
          // step's slice), while the scatter that consumes it binds the cache
          // tensor's STABLE BASE with the row offset as a kernel scalar
          // (NNTR_KV_SCALAR_OFF). The generic declaration inside
          // rope_inplace_f16_cl therefore names a plane that starts BELOW the
          // consumer's pointer and would not contain it. Widen it to the whole
          // cache tensor, which both addresses lie in. This is the site the
          // non-determinism was bisected to, so it is declared here even at
          // `site` scope.
          nntrainer::opencl::Kernel::noteUndrainedSvmPlane(
            cache_key.getData<_FP16>(), cache_key.bytes(),
            /*declared_by_consumer_site=*/true);
        }
        if (ok && q_out_stage != nullptr) {
          q_attn_clmem = q_out_stage;
          q_rope_staged = q_out_stage;
        }
        if (ok && q_rope_out == nullptr && q_cl != nullptr)
          q_rotated_in_svm_plane = true;
        if (ok && k_out_stage != nullptr) {
          // Side-fill the SVM cache slice for host decode (consumed only
          // after the lm_head lower drains the queue): no per-op drain.
          const unsigned int kc_n =
            (to - from) * (unsigned int)num_heads_KV * (unsigned int)head_dim;
          if (mha_clmem_mode) {
            // MHA_CLMEM: mirror-only store; the boundary gather syncs the
            // slab for host decode/save. No SVM write here.
            k_stage = k_stage_buf;
          } else if (nntrainer::gpu_copy_f16_cl(
                       k_p, kc_p, kc_n, /*svm_inputs=*/true,
                       /*in_clmem=*/k_stage_buf, /*out_clmem=*/nullptr,
                       /*drain=*/false)) {
            k_stage = k_stage_buf;
            static int _kv_stage_logged = 0;
            if (!_kv_stage_logged) {
              _kv_stage_logged = 1;
              ml_logd("[KV-STAGE] engaged M=%u kbytes=%zu", to - from,
                      k_stage_bytes);
            }
          } else {
            // Side-fill kernel unavailable: redo rope-K straight into the
            // SVM cache slice (drained), abandoning the staging chain.
            ok = nntrainer::rope_inplace_f16_cl(
              k_p, kc_p, cos_lut, sin_lut, to - from, num_heads_KV, head_dim,
              cache_index, mp, kc_svm, /*in_clmem=*/k_cl,
              /*out_clmem=*/nullptr);
          }
        }
        if (ok) {
          // V: no RoPE — scatter the V slice into its cache window on the GPU
          // (residency: keep V on the device, no host copy). Falls back to the
          // host copy if the GPU copy is unsupported.
          const bool v_svm = value_step.getMemoryData() &&
                             value_step.getMemoryData()->isSVM() &&
                             b_cache_value_step.getMemoryData() &&
                             b_cache_value_step.getMemoryData()->isSVM();
          const uint16_t *v_in =
            reinterpret_cast<const uint16_t *>(value_step.getData<_FP16>());
          uint16_t *v_out =
            reinterpret_cast<uint16_t *>(b_cache_value_step.getData<_FP16>());
          const unsigned int v_n =
            (to - from) * (unsigned int)num_heads_KV * (unsigned int)head_dim;
          // Staged chain: the cache write is a decode side-fill (drain
          // skipped); v_scatter reads this step's V straight from value_step
          // (same bytes, no dependency on the cache write).
          const bool v_stage = kv_stage_on && v_svm;
          if (mha_clmem_mode && v_stage) {
            // MHA_CLMEM: mirror-only store; v_scatter reads value_step.
            v_stage_svm = v_in;
            v_stage_clmem = v_cl;
            if (_kvst_on())
              _kvst_t0 = _kvst_now();
          } else if (nntrainer::gpu_copy_f16_cl(
                       v_in, v_out, v_n, v_svm, /*in_clmem=*/v_cl,
                       /*out_clmem=*/nullptr,
                       /*drain=*/!(v_stage || kv_chain_gpu_only))) {
            if (v_stage) {
              v_stage_svm = v_in;
              v_stage_clmem = v_cl;
            }
            if (!v_stage && kv_chain_gpu_only) {
              kv_write_undrained = true;
              // Same widening as the K site: an offset-baked destination, a
              // base-plus-scalar consumer.
              nntrainer::opencl::Kernel::noteUndrainedSvmPlane(
                cache_value.getData<_FP16>(), cache_value.bytes(),
                /*declared_by_consumer_site=*/true);
            }
            if (_kvst_on())
              _kvst_t0 = _kvst_now();
          } else {
            // Host fallback reads V on the host: lower first.
            if (v_cl) {
              nntrainer::clmem_lower_cl(value_step, 0);
              v_cl = nullptr;
            }
            b_cache_value_step.copyData(value_step); // host fallback
          }
          gpu_rope_done = true;
        }
      }
    }
#endif // ENABLE_OPENCL (GPU concat-RoPE)
#endif
#ifdef ENABLE_FP16
#if defined(ENABLE_OPENCL) // OHWI image GPU-RoPE (rope_inplace_f16_cl)
    // ---- (C) Adreno OHWI image-attention DECODE GPU-RoPE
    // --------------------- The concat GPU-RoPE branch above is gated
    // `!kv_ohwi_now`, but the live Adreno gemma4 decode path is the
    // IMAGE-attention path (NNTR_KV_IMG_ATTN), which is a SEPARATE env from
    // NNTR_KV_OHWI: kv_ohwi_now is FALSE there, so the
    // concat branch is also skipped and gpu_rope_done stays false -> the host
    // RoPE fallback (the !gpu_rope_done branch) runs lower_q()+lower_kv() (two
    // BLOCKING
    // clEnqueueReadBuffer drains, ~16-35 ms/token over 35 layers) and host
    // apply_rotary_emb. Rotate Q/K/V ON THE GPU instead, landing the bytes in
    // exactly the SVM buffers the existing GPU OHWI scatter + image attention
    // already consume:
    //   * Q  -> rotate IN-PLACE into query_step's SVM plane (Q_p). The OHWI
    //          decode qk kernel reads Q_p with q_clmem=nullptr, so
    //          the rotation must land in the SVM shadow. When the FC parked Q
    //          in a planner cl_mem (q_cl != null) we read cl_mem -> SVM
    //          (drained once, then null q_cl) so Q_p is fresh.
    //   * K  -> rotate from key_step into b_cache_key_step's SVM concat cache
    //          slice (kc_p). k_scatter_ohwi_cl reads b_cache_key_step.
    //          This is the SAME destination the host RoPE fallback writes
    //          and the SAME destination the concat GPU-RoPE writes (kc_p).
    //   * V  -> flat copy value_step -> b_cache_value_step's SVM slice (no
    //   RoPE).
    //          v_scatter_ohwi_t_cl reads b_cache_value_step when
    //          v_stage_svm is null (the unstaged decode path here). Matches the
    //          host fallback's V copy byte-for-byte.
    // Token-identical: rope_inplace_f16_cl == apply_rotary_emb_tensor_v2 for
    // the rotated bytes (the concat GPU-RoPE already depends on this
    // equivalence) and the destinations are identical to the host fallback's.
    // drain_svm_out=false / drain=false keep the writes as in-order enqueues
    // consumed only by the same-queue GPU OHWI scatter + image attention (the
    // SVM cache is read on the host only after the lm_head lower drains the
    // queue) -- the residency win. Gate: NNTR_OHWI_GPU_ROPE (default OFF ->
    // host path unchanged, A/B-able). Falls back (leaves gpu_rope_done=false ->
    // host RoPE) if: not the IMG-ATTN env, not a decode step (M!=1), not FP16,
    // kv_int8, the rope LUT does not cover the position, the cache slices are
    // not SVM, or any GPU op fails. Enabled by the NNTR_OHWI_GPU_ROPE env
    // (global override) OR per-LAYER via the gpu_ohwi_rope property (default-on
    // where token-identical: gemma4 +32%, gemma2 +8%; qwen3 stays false --
    // head_dim=128/q-k-norm diverges).
    static const bool _ohwi_gpu_rope_env =
      std::getenv("NNTR_OHWI_GPU_ROPE") != nullptr;
    const bool _ohwi_gpu_rope =
      _ohwi_gpu_rope_env || std::get<props::GpuOhwiRope>(mha_core_props).get();
    // Invariant: OHWI GPU-RoPE only feeds the image-attention layout — run
    // it only when this layer's attention actually takes the image path
    // (use_image_attn == 1; decided per layer by img_arm_want, see the
    // prebuild note).
    if (!gpu_rope_done && _ohwi_gpu_rope && !_gpu_rope_off && _mha_gpu_on &&
        _img_arm && use_image_attn == 1 && use_gemm_attention && !kv_int8 &&
        (to - from) == 1 &&
        query_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        key_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        value_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        cache_key.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        cache_value.getDataType() == ml::train::TensorDim::DataType::FP16) {
      const unsigned int mp = rope_lut_positions();
      // SVM cache slices are the scatter sources; require them SVM-backed (they
      // are on the live NNTR_GPU_SVM_POOL path). Q may be cl_mem (FC output) or
      // SVM; rope_inplace handles cl_mem-in -> SVM-out.
      const bool kc_svm = b_cache_key_step.getMemoryData() &&
                          b_cache_key_step.getMemoryData()->isSVM();
      const bool vc_svm = b_cache_value_step.getMemoryData() &&
                          b_cache_value_step.getMemoryData()->isSVM();
      const bool q_svm =
        query_step.getMemoryData() && query_step.getMemoryData()->isSVM();
      const bool v_svm =
        value_step.getMemoryData() && value_step.getMemoryData()->isSVM();
      if (mp >= cache_index + (to - from) && kc_svm && vc_svm) {
        ensure_rope_flat_lut();
        const uint16_t *cos_lut = rope_cos_flat_cur;
        const uint16_t *sin_lut = rope_sin_flat_cur;
        uint16_t *q_p =
          reinterpret_cast<uint16_t *>(query_step.getData<_FP16>());
        const uint16_t *k_p =
          reinterpret_cast<const uint16_t *>(key_step.getData<_FP16>());
        uint16_t *kc_p =
          reinterpret_cast<uint16_t *>(b_cache_key_step.getData<_FP16>());
        const uint16_t *v_in =
          reinterpret_cast<const uint16_t *>(value_step.getData<_FP16>());
        uint16_t *v_out =
          reinterpret_cast<uint16_t *>(b_cache_value_step.getData<_FP16>());
        // [rq-scalar-off] The recordable form of the same two destinations:
        // the cache tensor's OWN base (fixed for the life of the graph) plus
        // the step's row offset as a kernel SCALAR. kc_base + _kc_step_off is
        // kc_p; vc_base + _vc_step_off is v_out (assert-equal below).
        uint16_t *kc_base =
          reinterpret_cast<uint16_t *>(cache_key.getData<_FP16>());
        uint16_t *vc_base =
          reinterpret_cast<uint16_t *>(cache_value.getData<_FP16>());
        const bool _kv_soff_k = _kv_scalar_off && kc_base != nullptr &&
                                kc_svm && (kc_base + _kc_step_off) == kc_p;
        const bool _kv_soff_v = _kv_scalar_off && vc_base != nullptr &&
                                vc_svm && v_svm && v_in != nullptr &&
                                (vc_base + _vc_step_off) == v_out;
        const unsigned int kv_n =
          (to - from) * (unsigned int)num_heads_KV * (unsigned int)head_dim;
        // ORDER MATTERS for safe partial-failure fallback: rotate K and V into
        // the cache slices FIRST (these read the UNMODIFIED key_step/value_step
        // and write b_cache_key/value_step), and rotate Q IN-PLACE LAST. If any
        // op fails, gpu_rope_done stays false and the host fallback re-reads
        // the still-untouched key_step/value_step/query_step (the partial GPU
        // writes into the cache slices get overwritten by the host RoPE; Q is
        // untouched unless every op already succeeded). So Q is never
        // double-rotated. K: rotate key_step -> b_cache_key_step SVM slice (the
        // OHWI scatter source). cl_mem-in (k_cl) -> SVM-out. key_step left
        // unmodified.
        // [qkv-commit] The three launches below (rope-K, V copy, rope-Q) in
        // ONE kernel, byte-identical: on an Adreno 840 decode step they were
        // 3 of ~24 launches per layer plus 2 submissions, ~0.5 ms of host
        // time per token (1.5B model). Falls through to the separate
        // launches when a source is neither SVM nor device-resident (nothing
        // has been enqueued at that point). NNTR_QKV_COMMIT_FUSE=0 disables;
        // NNTR_QKV_COMMIT_FLUSH=1 submits right after it.
        static const bool _qkv_fuse = []() {
          const char *e = std::getenv("NNTR_QKV_COMMIT_FUSE");
          return e == nullptr || e[0] != '0';
        }();
        static const bool _qkv_flush = []() {
          const char *e = std::getenv("NNTR_QKV_COMMIT_FLUSH");
          return e != nullptr && e[0] == '1';
        }();
        const bool k_svm =
          key_step.getMemoryData() && key_step.getMemoryData()->isSVM();
        bool fused = false;
        if (_qkv_fuse && (q_cl != nullptr || q_svm) &&
            (k_cl != nullptr || k_svm) && (v_cl != nullptr || v_svm)) {
          fused = nntrainer::qkv_rope_commit_f16_cl(
            q_p, q_cl, k_p, k_cl, _kv_soff_k ? kc_base : kc_p,
            _kv_soff_k ? (unsigned int)_kc_step_off : 0u, v_in, v_cl,
            _kv_soff_v ? vc_base : v_out,
            _kv_soff_v ? (unsigned int)_vc_step_off : 0u, cos_lut, sin_lut,
            num_heads_Q, num_heads_KV, head_dim, cache_index, mp, _qkv_flush);
        }
        bool ok = fused;
        if (!fused)
          ok = nntrainer::rope_inplace_f16_cl(
            k_p, _kv_soff_k ? kc_base : kc_p, cos_lut, sin_lut, to - from,
            num_heads_KV, head_dim, cache_index, mp, kc_svm, /*in_clmem=*/k_cl,
            /*out_clmem=*/nullptr, /*drain_svm_out=*/false,
            /*write_off=*/_kv_soff_k ? (unsigned int)_kc_step_off : 0u);
        // Enqueued without a drain: the SVM/host fallbacks below must drain
        // once if the image attention ends up missing this step.
        if (ok)
          kv_write_undrained = true;
        // V: flat copy value_step -> b_cache_value_step SVM slice (no RoPE, the
        // OHWI v-scatter source). value_step left unmodified.
        if (ok && !fused)
          ok = _kv_soff_v ? nntrainer::gpu_copy_f16_row_cl(
                              v_in, vc_base, kv_n, (int)_vc_step_off,
                              /*svm_inputs=*/true,
                              /*in_clmem=*/v_cl, /*out_base_clmem=*/nullptr,
                              /*drain=*/false)
                          : nntrainer::gpu_copy_f16_cl(v_in, v_out, kv_n, v_svm,
                                                       /*in_clmem=*/v_cl,
                                                       /*out_clmem=*/nullptr,
                                                       /*drain=*/false);
        // Q LAST: in-place into the SVM shadow (the OHWI decode qk reads Q_p
        // with q_clmem=null). cl_mem-in -> SVM-out when the FC parked Q in
        // cl_mem. Only run once K+V are committed so a Q failure cannot leave a
        // half-rotated, then re-rotated, Q.
        if (ok && !fused)
          ok = nntrainer::rope_inplace_f16_cl(
            q_p, q_p, cos_lut, sin_lut, to - from, num_heads_Q, head_dim,
            cache_index, mp, q_svm, /*in_clmem=*/q_cl, /*out_clmem=*/nullptr,
            /*drain_svm_out=*/false);
        if (ok) {
          // GPU-side hashes of what the commit left behind
          // (inert unless NNTR_ATTN_GPUHASH).
          nntrainer::attn_gpuhash_note(
            "qc", q_p, nullptr,
            (size_t)num_heads_Q * head_dim * sizeof(uint16_t), to - from);
          nntrainer::attn_gpuhash_note(
            "kc", kc_p, nullptr, (size_t)kv_n * sizeof(uint16_t), to - from);
          nntrainer::attn_gpuhash_note(
            "vc", v_out, nullptr, (size_t)kv_n * sizeof(uint16_t), to - from);
          // All committed. Q/K/V no longer read on the host: null their cl_mem
          // handles (the GPU rotation already read them, SVM shadows are fresh)
          // so no later lower_q/lower_kv fires.
          q_cl = nullptr;
          k_cl = nullptr;
          v_cl = nullptr;
          gpu_rope_done = true;
          static int _ohwi_rope_logged = 0;
          if (!_ohwi_rope_logged) {
            _ohwi_rope_logged = 1;
            ml_logd("[OHWI-GPU-ROPE] engaged M=%u cache_index=%u mp=%u",
                    to - from, cache_index, mp);
          }
        }
        // ok==false -> gpu_rope_done stays false: host fallback runs unchanged
        // (the partial GPU writes above are harmless -- the host RoPE rewrites
        // Q/K/V over them after lower_q/lower_kv).
      }
    }
#endif // ENABLE_OPENCL (OHWI image GPU-RoPE)
#endif
// kv_ohwi_now / gpu_rope_done are declared inside the ENABLE_FP16 +
// ENABLE_OPENCL concat-RoPE block above, so this diagnostic must carry the
// same pair of guards (an enable-fp16=false OpenCL build otherwise fails to
// compile on the undeclared kv_ohwi_now).
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
    if (std::getenv("NNTR_ROPE_TPROF")) {
      static int _dbg_pf = 0, _dbg_dec = 0;
      const bool _dec = (to - from) == 1;
      if ((_dec && _dbg_dec < 1) || (!_dec && _dbg_pf < 1)) {
        if (_dec)
          ++_dbg_dec;
        else
          ++_dbg_pf;
        std::fprintf(
          stderr,
          "[ROPE-PATH] %s M=%u gpu_rope_done=%d kv_ohwi=%d -> %s RoPE\n",
          _dec ? "DECODE" : "PREFILL", to - from, (int)gpu_rope_done,
          (int)kv_ohwi_now, gpu_rope_done ? "GPU" : "HOST(lower)");
        std::fflush(stderr);
      }
    }
#endif
    if (!gpu_rope_done) {
      // Host RoPE reads/writes Q/K/V on the host: lower first (decode and any
      // GPU-RoPE-ineligible path).
#if defined(ENABLE_OPENCL)
      settle_act_stage_for_host();
#endif
      // NNTR_ROPE_TPROF: per-token wall time of the host-RoPE lower_q+lower_kv
      // drains (decode only). Accumulated across layers; printed each decode
      // token (reset). Measures whether GPU-RoPE-for-OHWI is worth the work.
      static const bool _rope_tprof = std::getenv("NNTR_ROPE_TPROF") != nullptr;
      static double _rope_acc = 0.0;
      static int _rope_layers = 0;
      static unsigned int _rope_M = 0;
      std::chrono::steady_clock::time_point _rt0;
      if (_rope_tprof)
        _rt0 = std::chrono::steady_clock::now();
      lower_q();
      lower_kv();
      if (_rope_tprof) {
        const double _dt = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - _rt0)
                             .count();
        if ((to - from) > 1) {
          // Prefill: accumulate across its (skip_prefill) producing layers and
          // print a running cumulative — the last line before decode = the
          // total prefill host-RoPE lower cost.
          static double _pf_acc = 0.0;
          static int _pf_n = 0;
          _pf_acc += _dt;
          ++_pf_n;
          std::fprintf(
            stderr,
            "[ROPE-TPROF-PREFILL] layer lower=%.2fms (M=%u) cum=%.2fms "
            "(%d layers)\n",
            _dt, to - from, _pf_acc, _pf_n);
          std::fflush(stderr);
        } else {
          _rope_acc += _dt;
          if (++_rope_layers >= 35) {
            std::fprintf(stderr,
                         "[ROPE-TPROF-DECODE] host lower_q+lower_kv = %.2f "
                         "ms/token (%d layers)\n",
                         _rope_acc, _rope_layers);
            std::fflush(stderr);
            _rope_acc = 0.0;
            _rope_layers = 0;
          }
        }
      }
      // apply rotary embedding for query.
      //
      // [skip-prefill] ...except on a KV-shared layer's prefill step. Such a
      // layer returns below as soon as K/V are in its cache, because its
      // attention output is unused, and the rotated query is read by NOTHING
      // ELSE -- only that attention. So the rotation is pure waste there.
      // Property-gated (skip_prefill) and prefill-only: no other layer, step
      // or engine changes behaviour. K/V still rotate and scatter -- the cache
      // is the whole point of the layer.
      const bool skip_q_rope = skip_prefill && (to - from) > 1;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
      // [m2b] Publish THIS step's absolute position before anything below reads
      // it. Under the decode-graph pair the K and V writes further down compute
      // their destination row on the DEVICE from d_pos[0]
      // (cuda_rope_fp16_dpos / cuda_scalar_mul_fp16_slot with out_slot_dpos),
      // so a step that does not publish scatters its keys at whatever row the
      // PREVIOUS step left there.
      //
      // Several paths reached those writes without publishing, because the only
      // publish was inside the query-RoPE arm below:
      //   - a skip-prefill model's prefill (skip_q_rope) leaves that arm before
      //     the publish;
      //   - so does a prefill whose query is not device-resident, or whose trig
      //     table is shorter than cache_index + nrows;
      //   - NNTR_CUDA_ROPE off skips the arm entirely.
      // The first is the one that bites. A process's FIRST prefill starts at 0
      // and inherits an accidentally-correct 0 from fresh device memory; a
      // SECOND generation prefills at from > 0 and inherits the previous run's
      // last decode position instead.
      //
      // Measured, Gemma-4 on a discrete part (skip_prefill = true), 1K prompt
      // cut at 400 tokens, 64 asked for: continuing in the same process
      // produced 5 tokens of unrelated text where the uninterrupted run
      // produced 64, and with NNTR_CUDA_DECODE_GRAPH=0 -- which turns the slot
      // writes off -- the two agreed. Plain multi-turn shows it with no KV
      // resume involved, so this is not about resume: it is about any second
      // prefill.
      //
      // Publishing here is an 8-byte pinned H2D on the backend stream,
      // idempotent, and the value is the one the arm below used to set. Not
      // while capturing: under capture CudaContext::runDecode publishes once
      // per token before the replay, and a write inside the capture would
      // bake it.
      {
        static const bool m2b_pos = nntrainer::cuda::decodeGraphEnabled();
        if (m2b_pos && !nntrainer::cuda::StreamManager::Global().isCapturing())
          nntrainer::cuda::cuda_set_pos((int)cache_index, (int)cache_index + 1);
      }
      // [m2b] ...and the device-slot writes themselves belong to a step that
      // can be REPLAYED, which is a single-token decode step and nothing else.
      // The decode graph is only ever captured for (to - from) == 1 (the
      // prefill capture is a separate lever and only at from == 0, where it
      // records the direct writes), so on a multi-token step the slot form buys
      // nothing and costs the guarantee that the destination is known on the
      // host.
      //
      // It cost exactly that. A resumed prefill (from > 0, many rows) went
      // through the slot kernels, whose destination row is d_pos[0] + row/width
      // read on the DEVICE, and landed its keys outside the range the run then
      // attended: dumping the cache after a 400 -> 842 resumed prefill showed
      // the planes from layer 3 onward still ZERO at rows 400..841 (17.6 MB of
      // 36 MB differing from the same resume done in a fresh process, and all
      // of it zero on the failing side). The answer stayed fluent for five
      // tokens and then stopped. With NNTR_CUDA_DECODE_GRAPH=0 -- i.e. with the
      // direct writes -- the cache came out byte-identical to the uninterrupted
      // run.
      //
      // A process's FIRST prefill hid this: it starts at row 0, and the direct
      // and slot forms then agree for as long as d_pos still holds the 0 it was
      // allocated with. Only a second prefill in the same process (multi-turn,
      // or a resume) starts anywhere else.
      const bool m2b_replayable_step = (to - from) == 1;
      // GPU RoPE (decode, device-resident query): split-half rotation on the
      // device matching apply_rotary_emb_tensor_v2, keeping the query off the
      // host. Opt-in (NNTR_CUDA_ROPE) until the whole decode chain is on-GPU.
      bool q_rope_gpu = false;
      {
        static const bool cuda_rope = nntr_env_on("NNTR_CUDA_ROPE");
        if (cuda_rope &&
            query_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
          if (cached_freqs_cos_fp16 == nullptr ||
              cached_freqs_sin_fp16 == nullptr) {
            const std::lock_guard<std::mutex> lock(rope_init_mtx);
            // Cap the RoPE trig table to the live max sequence length
            // (rope_lut_positions() = MaxTimestep ~= max_seq_len) instead of
            // the model's max_position_embeddings (131072 for gemma4). The
            // table is already shared across layers via precompute_freqs'
            // rope_freq_cache (only the 2 distinct (head_dim, theta) configs
            // are built), but each build was the full 128K positions => ~376ms
            // of prefill host time for the 2 trig builds + ~113ms flatten/H2D,
            // when prefill only touches <=max_seq_len positions. The OpenCL
            // GPU-RoPE path already caps via rope_lut_positions(); this mirrors
            // it for the CUDA path.
            precompute_freqs(head_dim, rope_lut_positions(), theta, true);
            cached_freqs_cos_fp16 = freqs_cos_fp16;
            cached_freqs_sin_fp16 = freqs_sin_fp16;
          }
          const unsigned int nrows = query_step.height();
          if ((size_t)cache_index + nrows <= (*cached_freqs_cos_fp16).size()) {
            unsigned short *q =
              reinterpret_cast<unsigned short *>(query_step.getData<_FP16>());
            const bool dev = nntrainer::cuda::dev_accessible(q);
            if (dev) {
              const int half = head_dim / 2;
              const unsigned short *cosd =
                rope_lut_device(cached_freqs_cos_fp16, half);
              const unsigned short *sind =
                rope_lut_device(cached_freqs_sin_fp16, half);
              if (cosd && sind && skip_q_rope) {
                // [skip-prefill] everything above is warm-up that must keep
                // happening on the prefill step -- the trig table build and
                // its one-time device upload. Only the rotation itself is
                // dropped, and only for this layer's dead prefill query.
                q_rope_gpu = true;
              } else if (cosd && sind) {
                // M2-B: read the RoPE position from the device d_pos buffer so
                // a captured decode graph stays valid across tokens. The
                // publish moved up to the top of this block, where every path
                // that reaches the K/V slot writes passes through it.
                static const bool m2b = nntrainer::cuda::decodeGraphEnabled();
                if (m2b && m2b_replayable_step) {
                  q_rope_gpu = nntrainer::cuda::cuda_rope_fp16_dpos(
                    q, q, cosd, sind, query_step.width() / head_dim, head_dim,
                    (int)nrows, /*out_slot_dpos=*/0);
                } else {
                  q_rope_gpu = nntrainer::cuda::cuda_rope_fp16(
                    q, q, cosd, sind, query_step.width() / head_dim, head_dim,
                    (int)nrows, (int)cache_index);
                }
              }
            }
          }
        }
      }
      if (!q_rope_gpu) {
        // host RoPE fallback (prefill height>1): sync first so the host read of
        // GPU-produced q is coherent under NNTR_CUDA_ASYNC.
        nntrainer::cuda::drain_if_async();
#endif
        apply_rotary_emb_tensor_v2(query_step, query_step, head_dim,
                                   cache_index, true);
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
      }
#endif

      // append kcache with rotary embedding. A KOhwi cache (§3.8 OHWI,
      // kv_layout) is written by the host through the op, which rotates K and
      // scatters it per head; the device arms below write a concat cache only.
      // Whether K still has to be written on the host, after a device arm.
      bool k_on_host = false;
      if (kv_layout == nntrainer::KVCacheLayout::KOhwi) {
        k_on_host = true;
      } else {
        // GPU RoPE for K straight into the (UVM) cache slice + GPU V copy:
        // keeps the whole KV-cache write off the host. Requires NNTR_CUDA_ROPE
        // and a device-resident cache (NNTR_CUDA_KV_UVM); the dev checks gate
        // it.
        bool k_rope_gpu = false;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
        {
          static const bool cuda_rope = nntr_env_on("NNTR_CUDA_ROPE");
          auto dev = [](const void *p) {
            return nntrainer::cuda::dev_accessible(p);
          };
          const unsigned int knrows = key_step.height();
          if (cuda_rope &&
              key_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
              cached_freqs_cos_fp16 != nullptr &&
              (size_t)cache_index + knrows <= (*cached_freqs_cos_fp16).size()) {
            auto *kin =
              reinterpret_cast<unsigned short *>(key_step.getData<_FP16>());
            auto *kout = reinterpret_cast<unsigned short *>(
              b_cache_key_step.getData<_FP16>());
            if (dev(kin) && dev(kout)) {
              const int half = head_dim / 2;
              const unsigned short *cosd =
                rope_lut_device(cached_freqs_cos_fp16, half);
              const unsigned short *sind =
                rope_lut_device(cached_freqs_sin_fp16, half);
              static const bool m2b_k = nntrainer::cuda::decodeGraphEnabled();
              if (cosd && sind && m2b_k && m2b_replayable_step) {
                // M2-B: write RoPE'd K into the cache at the live slot computed
                // on-device from d_pos[0] (kbase = cache BASE for this batch,
                // not the host pre-offset b_cache_key_step) -> correct slot on
                // replay.
                unsigned short *kbase =
                  reinterpret_cast<unsigned short *>(
                    cache_key.getData<_FP16>()) +
                  (size_t)batch * cache_key_dim.getFeatureLen();
                k_rope_gpu = nntrainer::cuda::cuda_rope_fp16_dpos(
                  kin, kbase, cosd, sind, key_step.width() / head_dim, head_dim,
                  (int)knrows, /*out_slot_dpos=*/1,
                  /*ring_cap=*/(int)kv_ring_cap);
              } else if (cosd && sind) {
                k_rope_gpu = nntrainer::cuda::cuda_rope_fp16(
                  kin, kout, cosd, sind, key_step.width() / head_dim, head_dim,
                  (int)knrows, (int)cache_index);
              }
            }
          }
        }
#endif
        k_on_host = !k_rope_gpu;
      }

      // V has no RoPE. On the cuda engine an FP16 V may be copied into its
      // cache slice on the device; otherwise it is written on the host below.
      bool v_on_host = true;
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
      if (query_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
        bool v_copy_gpu = false;
        {
          static const bool cuda_elt = nntr_env_on("NNTR_CUDA_ROPE");
          // V-copy historically stayed host for the prefill big-step (height>1)
          // because a host attention path could read the V cache unsynced. With
          // GPU attention (NNTR_CUDA_ATTN) + UVM KV cache (NNTR_CUDA_KV_UVM),
          // prefill attention reads the cache on the GPU (same stream), so a
          // GPU V copy is a GPU->GPU handoff -- no host read, no drain. Gated
          // by NNTR_CUDA_VCOPY_PREFILL (defaulted on by the CudaContext device
          // profiles; =0 restores the host copy); removes the per-layer
          // finishIfAsync bubble that made async-on prefill slow.
          static const bool vcopy_prefill =
            nntr_env_on("NNTR_CUDA_VCOPY_PREFILL");
          auto *vin =
            reinterpret_cast<unsigned short *>(value_step.getData<_FP16>());
          auto *vout = reinterpret_cast<unsigned short *>(
            b_cache_value_step.getData<_FP16>());
          const bool dev = nntrainer::cuda::dev_accessible(vout);
          // Device-only KV (NNTR_CUDA_KV_DEV): the host copyData fallback
          // below would dereference a cudaMalloc pointer -- always take the
          // GPU copy for a device-only destination, independent of the
          // VCOPY_PREFILL opt-in. Asked via the MemoryData residency stamp
          // (pool-bind time), not a per-call driver query -- layering rule.
          const auto v_md = b_cache_value_step.getMemoryData();
          const bool v_dev_only = v_md && !v_md->isHostAddressable();
          static const bool m2b_v = nntrainer::cuda::decodeGraphEnabled();
          if (cuda_elt && dev &&
              (value_step.height() == 1 || vcopy_prefill || v_dev_only)) {
            if (m2b_v && m2b_replayable_step) {
              // M2-B: write V into the cache at the live slot d_pos[0] computed
              // on-device (vbase = cache BASE for this batch) -> correct on
              // replay.
              unsigned short *vbase =
                reinterpret_cast<unsigned short *>(
                  cache_value.getData<_FP16>()) +
                (size_t)batch * cache_value_dim.getFeatureLen();
              if (nntrainer::cuda::cuda_scalar_mul_fp16_slot(
                    vin, vbase, (unsigned int)value_step.size(), 1.0f,
                    (int)cache_value_dim.width(),
                    /*ring_cap=*/(int)kv_ring_cap))
                v_copy_gpu = true;
            } else if (nntrainer::cuda::cuda_scalar_mul_fp16(
                         vin, vout, (unsigned int)value_step.size(), 1.0f)) {
              v_copy_gpu = true;
            }
          }
        }
        v_on_host = !v_copy_gpu;
      }
#endif

      // The host writes the planes no device arm took: K rotated, V as is.
      if (k_on_host || v_on_host) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
        nntrainer::cuda::drain_if_async();
#if defined(ENABLE_FP16)
        NNTR_THROW_IF(k_on_host && b_cache_key_step.getMemoryData() &&
                        !b_cache_key_step.getMemoryData()->isHostAddressable(),
                      std::runtime_error)
          << "device-only KV (NNTR_CUDA_KV_DEV) requires GPU RoPE for K; the "
             "host fallback would fault (NNTR_CUDA_ROPE off or RoPE LUT "
             "range miss at cache_index="
          << cache_index << ")";
        NNTR_THROW_IF(
          v_on_host &&
            query_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
            b_cache_value_step.getMemoryData() &&
            !b_cache_value_step.getMemoryData()->isHostAddressable(),
          std::runtime_error)
          << "device-only KV (NNTR_CUDA_KV_DEV) requires the GPU V-copy; the "
             "host copyData fallback would fault -- check NNTR_CUDA_ELTWISE";
#endif
#endif
        write_kv_cache(batch, key_step, value_step, cache_key, cache_value,
                       cache_index, k_on_host, v_on_host, /*rotate_key=*/true);
      }
    }
  } else {
    // No-RoPE path: host scatters/copies read K/V on the host -- lower first.
    lower_kv();
    write_kv_cache(batch, key_step, value_step, cache_key, cache_value,
                   cache_index, /*write_key=*/true, /*write_value=*/true,
                   /*rotate_key=*/false);
  }

  /// @todo replace step_size into input height
  unsigned int step_size = to - from;
  unsigned int cache_from = cache_index;
  unsigned int cache_to = cache_from + step_size;

  // NNTR_KVHASH: hash this layer's SVM K/V cache so two runs of
  // the same prompt can be compared without pulling the caches off the device.
  //   1 = at the prefill/decode boundary only (one drain, at the point decode
  //       drains anyway, so what it hashes is an un-perturbed prefill);
  //   2 = after every prefill chunk as well (one drain per chunk).
  // Rows are hashed in 512-row blocks so the report names the first differing
  // block, not just the layer. Inert unless the env is set.
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
  {
    static const int kvhash_mode = []() {
      const char *e = std::getenv("NNTR_KVHASH");
      return e ? std::atoi(e) : 0;
    }();
    if (kvhash_mode > 0 && batch == 0 &&
        cache_key.getDataType() == ml::train::TensorDim::DataType::FP16) {
      static std::mutex kvh_mu;
      static std::unordered_map<const void *, int> kvh_id;
      static std::unordered_map<const void *, char> kvh_decoded;
      std::lock_guard<std::mutex> _kvh_lock(kvh_mu);
      auto it = kvh_id.find((const void *)this);
      if (it == kvh_id.end())
        it = kvh_id.emplace((const void *)this, (int)kvh_id.size()).first;
      const bool boundary =
        step_size == 1 &&
        kvh_decoded.find((const void *)this) == kvh_decoded.end();
      if (step_size == 1)
        kvh_decoded[(const void *)this] = 1;
      if (boundary || (kvhash_mode >= 2 && step_size > 1)) {
        nntrainer::cl_queue_finish();
        const size_t hd = (size_t)num_heads_KV * head_dim;
        const unsigned int rows =
          kv_ring_cap ? std::min<unsigned int>(cache_from, kv_ring_cap)
                      : cache_from;
        const uint16_t *kb =
          reinterpret_cast<const uint16_t *>(cache_key.getData<_FP16>());
        const uint16_t *vb =
          reinterpret_cast<const uint16_t *>(cache_value.getData<_FP16>());
        // Hash, and alongside it the magnitude summary that distinguishes a
        // 1-ULP numeric difference from a garbage row.
        struct Blk {
          uint64_t h;
          double absum;
          double amax;
          unsigned int nonfinite;
        };
        auto blk = [&](const uint16_t *base, unsigned int r0, unsigned int r1) {
          Blk b{1469598103934665603ull, 0.0, 0.0, 0u};
          for (size_t i = (size_t)r0 * hd; i < (size_t)r1 * hd; ++i) {
            b.h ^= base[i];
            b.h *= 1099511628211ull;
            const float f = (float)*reinterpret_cast<const _FP16 *>(base + i);
            if (!std::isfinite(f))
              ++b.nonfinite;
            else {
              const double a = std::fabs((double)f);
              b.absum += a;
              if (a > b.amax)
                b.amax = a;
            }
          }
          return b;
        };
        for (unsigned int r0 = 0; r0 < rows; r0 += 512) {
          const unsigned int r1 = std::min(rows, r0 + 512u);
          const Blk bk = kb ? blk(kb, r0, r1) : Blk{0, 0, 0, 0};
          const Blk bv = vb ? blk(vb, r0, r1) : Blk{0, 0, 0, 0};
          std::fprintf(stderr,
                       "[KVHASH] L%02d %s from=%u rows=%u blk=%u-%u "
                       "k=%016llx v=%016llx ks=%.6e ka=%.6e kn=%u vs=%.6e "
                       "va=%.6e vn=%u\n",
                       it->second, boundary ? "bnd" : "chk", cache_from, rows,
                       r0, r1, (unsigned long long)bk.h,
                       (unsigned long long)bv.h, bk.absum, bk.amax,
                       bk.nonfinite, bv.absum, bv.amax, bv.nonfinite);
        }
      }
    }
  }
#endif

  // skip_prefill (Gemma4 KV-shared layers): K/V are now written + scattered
  // into this layer's own cache slab above, which is all decode needs (decode
  // re-derives cache_index from the absolute `from` each step, and attends to
  // these cached positions). The prefill attention OUTPUT is unused -- the O
  // FC and every downstream per-layer op for a KV-shared layer also skip the
  // prefill big-step -- so skip the heavy attention compute + output write.
  // Decode (step_size == 1) always runs the full path. The CLMEM output raise
  // is skipped intentionally: the downstream wo FC does not read O in prefill.
  if (skip_prefill && step_size > 1)
    return;

  // NNTR_MHA_CLMEM boundary sync (see mha_clmem_mode): gather the mirror
  // rows the slab is missing back into the concat SVM slab, capped by the
  // mirror's valid range, then drain once so the host may read immediately.
  auto sync_kv_slab = [&](unsigned int upto) {
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
    // OHWI mirror gather (k_gather_ohwi_cl / v_gather_ohwi_t_cl, cl_mem) is
    // OpenCL-only; no mirror exists without it.
    // ENABLE_FP16 is required too: the KV mirror is an FP16-only layout and the
    // gather reinterprets cache_key/cache_value as _FP16, a type that only
    // exists under ENABLE_FP16.
    if (!mha_clmem_mode || !kv_mirror_init)
      return;
    if (upto > kv_k_valid_to)
      upto = kv_k_valid_to;
    if (upto > kv_v_valid_to)
      upto = kv_v_valid_to;
    if (kv_slab_synced_to >= upto)
      return;
    const size_t hd = (size_t)num_heads_KV * head_dim;
    // [kv-slab-gather] The mirror is the ONLY K/V store just for the staged
    // chain (NNTR_KV_STAGE): there the mha_clmem_mode branches above skip the
    // SVM side-fill, and only there does the slab need gathering back. Unstaged
    // -- the default -- the RoPE side-fill and the decode host writes have
    // already made the slab complete, which the mirror-repair site right below
    // states outright, so this gather is a no-op BY VALUE.
    //
    // It stopped being a no-op by MEMORY PATH the moment the mirror became
    // image-written: it READS the mirror through the aliased BUFFER handle
    // while the scatters WRITE it through the image -- the original hazard
    // exactly reversed. A stale texture/buffer line then lands in the concat
    // slab, and mirror_backfill re-scatters those rows straight back into the
    // mirror. That is contributor (iii): it fires only where the mirror has
    // slid (cache_from > kv_mirror_base), i.e. only past ~4K keys, at a chunk
    // boundary -- which is exactly where the residual 12-15-node event was
    // seen, and why 0.3B/2047 and E2B's 923-token prefill (mirror never slides)
    // were already clean. So: with the mirror image-written, do not read it
    // back through the buffer at all.
    //
    // Staged + image-written is the one combination that still needs the rows:
    // there the gather must go through the IMAGE (kv_img_gather below), because
    // the buffer read is unsound and the slab is genuinely stale.
    // NNTR_KV_SLAB_GATHER=1 forces the old buffer gather back (A/B only).
    static const bool _slab_gather_force =
      std::getenv("NNTR_KV_SLAB_GATHER") != nullptr &&
      std::atoi(std::getenv("NNTR_KV_SLAB_GATHER")) != 0;
    const bool slab_authoritative = !kv_stage_on && !_slab_gather_force;
    if (nntrainer::kv_img_write_mode() >= 2 ||
        (nntrainer::kv_img_write_enabled() && slab_authoritative)) {
      kv_slab_synced_to = upto;
      return;
    }
    const unsigned int v_gs =
      kv_v_cur_stride != 0 ? kv_v_cur_stride : kv_mirror_S_max;
    // [kv-slab-ring] This gather used to take the ABSOLUTE range to both sides:
    // it wrote the concat slab at row `abs` and read the mirror at row `abs`.
    // The slab of a RINGED layer is only kv_ring_cap rows tall, so the write
    // ran past the layer's allocation as soon as abs >= kv_ring_cap; and a
    // mirror that has slid holds [kv_mirror_base, +rows), so the read picked
    // the wrong rows. Nothing consumed the result on today's image path, which
    // is why it stayed latent -- it is still an out-of-allocation write. Map
    // both sides the way every other consumer does (slab through cacheRow,
    // mirror through
    // - kv_mirror_base), split at the ring seam, and copy only rows the mirror
    // actually holds. Ring off with an unslid mirror this is one segment with
    // the old arguments, instruction for instruction.
    const unsigned int mirror_rows = std::min(kv_mirror_S_max, v_gs);
    const auto segs = causallm::kvSlabSyncSegments(
      kv_slab_synced_to, upto, kv_ring_cap, kv_mirror_base, mirror_rows);
    if (segs.empty()) {
      kv_slab_synced_to = upto;
      return;
    }
    // [kv-img-write] This boundary sync stays on the BUFFER read even when the
    // mirror is written through the image, and that is a measurement, not an
    // oversight. NNTR_KV_IMG_GATHER=1 routes it through the image instead
    // (k/v_gather_ohwi_img) and 0.3B/2047 then leaves the reference ids
    // with the KV slab bit-identical at the boundary --
    // i.e. the gather's own image READ is what breaks the run: it pulls the
    // mirror through the texture cache once per step, and the lines it leaves
    // there are what the NEXT chunk's attention is served. An image read is a
    // cache polluter on this driver; an image write is not. The gather is a
    // drained boundary sync on the same in-order queue, so reading the buffer
    // after image writes is ordered by the clFinish it already does.
    static const bool _kvimg_gather_img =
      std::getenv("NNTR_KV_IMG_GATHER") != nullptr &&
      std::atoi(std::getenv("NNTR_KV_IMG_GATHER")) != 0;
    void *v_img_g = (v_image_tight != nullptr && kv_v_img_S == v_gs)
                      ? v_image_tight
                      : v_image_ohwi;
    // [kv-slab-gather] Reaching here with the mirror image-written means the
    // staged chain really does need these rows (see above); the buffer read
    // would be the reverse alias, so take the image.
    const bool gather_img =
      (_kvimg_gather_img ||
       (nntrainer::kv_img_write_enabled() && !slab_authoritative)) &&
      k_image_ohwi != nullptr;
    uint16_t *const k_base = reinterpret_cast<uint16_t *>(
      cache_key.getData<_FP16>() +
      (size_t)batch * cache_key_dim.getFeatureLen());
    uint16_t *const v_base = reinterpret_cast<uint16_t *>(
      cache_value.getData<_FP16>() +
      (size_t)batch * cache_value_dim.getFeatureLen());
    for (size_t si = 0; si < segs.size(); ++si) {
      const auto &s = segs[si];
      const bool last = (si + 1 == segs.size());
      uint16_t *k_dst = k_base + (size_t)s.slab_row * hd;
      uint16_t *v_dst = v_base + (size_t)s.slab_row * hd;
      if (gather_img) {
        nntrainer::k_gather_ohwi_img_cl(k_image_ohwi, k_dst, s.rows,
                                        num_heads_KV, head_dim, kv_mirror_S_max,
                                        s.mirror_row,
                                        /*drain=*/false);
        nntrainer::v_gather_ohwi_t_img_cl(v_img_g, v_dst, s.rows, num_heads_KV,
                                          head_dim, v_gs, s.mirror_row,
                                          /*drain=*/last);
      } else {
        nntrainer::k_gather_ohwi_cl(reinterpret_cast<cl_mem>(k_buf_ohwi), k_dst,
                                    s.rows, num_heads_KV, head_dim,
                                    kv_mirror_S_max, s.mirror_row,
                                    /*drain=*/false);
        nntrainer::v_gather_ohwi_t_cl(reinterpret_cast<cl_mem>(v_buf_ohwi),
                                      v_dst, s.rows, num_heads_KV, head_dim,
                                      v_gs, s.mirror_row, /*drain=*/last);
      }
    }
    kv_slab_synced_to = upto;
#else
    (void)upto;
#endif // ENABLE_OPENCL (sync_kv_slab)
  };

  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  // [kv-window-ring] The cache buffer is only kv_ring_cap rows, so the read
  // VIEW must fit it. The GPU flash kernel walks the LOGICAL range (N_kv =
  // cache_to, passed separately) and modulo-maps to the physical ring, so a
  // Wcap-high view is sufficient. (Ring off: full cache_to, unchanged.)
  const unsigned int read_rows =
    kv_ring_cap ? std::min<unsigned int>(cache_to, kv_ring_cap) : cache_to;
  cached_key_dim.height(read_rows);
  cached_value_dim.height(read_rows);

  // §3.8 Phase 2: OHWI-direct GPU prefill. When both NNTR_KV_OHWI=1 and
  // NNTR_MHA_GPU=1 are set, the K cache is already laid out as
  // [H_kv, S_max, d] (Phase 1 write) and we can dispatch the
  // qk_matmul_f16_ohwi kernel against the raw SVM cache_key buffer
  // directly — skipping the Phase 1 gather entirely. V is still concat
  // (Phase 3 will move V); sv_matmul_f16 is reused. Returns early on
  // success; on failure falls through to the Phase 1 gather + concat
  // path below. Opt-in by both env vars together; no broken-gate.
  {
    const unsigned int FLASH_MIN_PREFILL =
      min_prefill_thr((unsigned int)head_dim); // env NNTR_MIN_PREFILL
    static const bool _mha_gpu_env = nntr_env_on("NNTR_MHA_GPU");
    const bool _ohwi_gpu_on =
      kv_layout == nntrainer::KVCacheLayout::KOhwi && _mha_gpu_env;
    // NNTR_MHA_GPU_DECODE=1 also routes DECODE (step_size==1) through this
    // OHWI image-attention path. The decode CPU NEON path (compute_kcaches) is
    // the long-context bottleneck; the OHWI qk/sv image kernels + KV scatter
    // already handle M=1. Falls through to CPU on not-ok.
    // (B) flash/OHWI decode-attention gate. Enabled by the NNTR_MHA_GPU_DECODE
    // env flag (global testing override) OR per-LAYER via the gpu_decode_attn
    // property (DEFAULT-ON for models where decode flash attention is
    // token-identical, e.g. gemma4, gemma2). Env read stays static; property
    // read PER-CALL and OR'd in.
    static const bool _ohwi_decode_env =
      std::getenv("NNTR_MHA_GPU_DECODE") != nullptr;
    const bool _ohwi_decode_on =
      _ohwi_decode_env || std::get<props::GpuDecodeAttn>(mha_core_props).get();
    const unsigned int step_size_p2 = to - from;
    const unsigned int cache_to_p2 = cache_index + step_size_p2;
    if (_ohwi_gpu_on && use_gemm_attention &&
        (step_size_p2 >= FLASH_MIN_PREFILL || _ohwi_decode_on) &&
        query_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        cache_key.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        attention_output_step.getDataType() ==
          ml::train::TensorDim::DataType::FP16 &&
        head_dim > 0 && num_heads_KV > 0 && num_heads_Q % num_heads_KV == 0) {
#ifdef ENABLE_FP16
#if defined(ENABLE_OPENCL) // OHWI-direct GPU prefill attention
      const uint16_t *Q_p =
        reinterpret_cast<const uint16_t *>(query_step.getData<_FP16>());
      uint16_t *O_p =
        reinterpret_cast<uint16_t *>(attention_output_step.getData<_FP16>());
      const size_t kv_per_batch =
        (size_t)num_heads_KV * cache_key_dim.height() * head_dim;
      const uint16_t *K_ohwi =
        reinterpret_cast<const uint16_t *>(cache_key.getData<_FP16>()) +
        (size_t)batch * kv_per_batch;
      const uint16_t *V_concat =
        reinterpret_cast<const uint16_t *>(cache_value.getData<_FP16>()) +
        (size_t)batch * cache_value_dim.getFeatureLen();
      const bool svm_ok =
        query_step.getMemoryData() && cache_key.getMemoryData() &&
        cache_value.getMemoryData() && attention_output_step.getMemoryData() &&
        query_step.getMemoryData()->isSVM() &&
        cache_key.getMemoryData()->isSVM() &&
        cache_value.getMemoryData()->isSVM() &&
        attention_output_step.getMemoryData()->isSVM();
      // SVM-gated by default. KVCacheManager allocates the KV cache
      // from the SVM pool on the gpu engine (unless NNTR_GPU_SVM_POOL=0);
      // with a host-tensor cache svm_ok is false and the non-SVM wrapper
      // path would upload the full H_kv*S_max*d slab per layer per
      // step (4MB at S_max=2048, ~7x more than the live N_kv slice
      // would need). Opting into NNTR_KV_OHWI_GPU_FORCE=1 exercises the
      // non-SVM path anyway (slow + drift-prone, for kernel validation
      // only).
      static const bool _ohwi_force =
        std::getenv("NNTR_KV_OHWI_GPU_FORCE") != nullptr;
      static int _ohwi_logged = 0;
      static int _ohwi_dec_logged = 0;
      const bool _is_dec = (step_size_p2 == 1);
      if (!_ohwi_logged || (_is_dec && _ohwi_dec_logged < 3)) {
        _ohwi_logged = 1;
        if (_is_dec)
          ++_ohwi_dec_logged;
        ml_logd("[OHWI-P2] %s M=%u N_kv=%u S_max=%zu H_q=%zu H_kv=%zu "
                "d=%zu svm=%d force=%d qSVM=%d kSVM=%d vSVM=%d oSVM=%d",
                _is_dec ? "DECODE" : "PREFILL", step_size_p2, cache_to_p2,
                cache_key_dim.height(), num_heads_Q, num_heads_KV, head_dim,
                (int)svm_ok, (int)_ohwi_force,
                (int)(query_step.getMemoryData() &&
                      query_step.getMemoryData()->isSVM()),
                (int)(cache_key.getMemoryData() &&
                      cache_key.getMemoryData()->isSVM()),
                (int)(cache_value.getMemoryData() &&
                      cache_value.getMemoryData()->isSVM()),
                (int)(attention_output_step.getMemoryData() &&
                      attention_output_step.getMemoryData()->isSVM()));
      }
      if (svm_ok || _ohwi_force) {
        const unsigned int win_p2 = (local_window_size >= (size_t)cache_to_p2)
                                      ? 0u
                                      : (unsigned int)local_window_size;
        bool ok = nntrainer::two_conv_attention_prefill_f16_ohwi_cl(
          Q_p, K_ohwi, V_concat, O_p, step_size_p2, cache_to_p2, num_heads_Q,
          num_heads_KV, head_dim, cache_key_dim.height(), is_causal,
          /*svm_inputs=*/svm_ok, win_p2);
        if (ok)
          return;
      }
#endif // ENABLE_OPENCL (OHWI-direct GPU prefill)
#endif
    }
  }

  // §3.8 OHWI read path: a KOhwi cache_key holds [B, H_kv, S, D]. The op
  // reads that layout itself (KVCacheView::layout), but the OpenCL two_conv
  // arm below reads concat and takes a host copy, so until it moves behind
  // the op the layer gathers into a fresh concat scratch here and hands that
  // to every reader (the op then sees a Concat plane). The fresh tensor is
  // non-SVM, so the GPU prefill SVM check below falls through to the CPU as
  // before; the OHWI-direct path above handles OHWI+MHA_GPU.
  const bool kv_ohwi_read_active = kv_layout == nntrainer::KVCacheLayout::KOhwi;
  nntrainer::Tensor b_cached_key;
  if (kv_ohwi_read_active) {
    // fp16 bits either way: an FP16 cache, or UINT16 without ENABLE_FP16.
    ml::train::TensorDim per_batch_key_dim = cached_key_dim;
    per_batch_key_dim.batch(1);
    b_cached_key = nntrainer::Tensor(per_batch_key_dim, true);
    gather_k_ohwi_to_concat_fp16(
      cache_key.getData<uint16_t>(), b_cached_key.getData<uint16_t>(), batch,
      cache_to, num_heads_KV, head_dim, cache_key_dim.height());
  } else {
    b_cached_key = cache_key.getSharedDataTensor(
      cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  }
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  // The device attention arms (OpenCL), for a layer with use_gemm_attention.
  // Gated on a minimum prefill length. Decode (step_size == 1) enters when a
  // device attention arm can run (mha_device_attention_on()). A step no device
  // arm here takes falls through to the op call below, where the backend picks
  // the OpenCL flash / two_conv arms (ClComputeOps::attention), the CUDA arm,
  // or a host arm (CpuComputeOps::attention: per-row or flash).
  const unsigned int FLASH_MIN_PREFILL =
    min_prefill_thr((unsigned int)head_dim); // env NNTR_MIN_PREFILL
  // S3 (decode GPU attention): NNTR_MHA_GPU_DECODE lets step_size==1 (decode)
  // enter the GPU attention block too. The KV-image (kvimg_view) path scatters
  // the single new token into the mirrors and reads the full N_kv context, so
  // M=1 is handled; otherwise decode falls to the host compute_kcaches NEON
  // path (a per-layer GPU->host queue drain). Pays off only with the host
  // RoPE/v_norm drains also removed (land S1+S2+S3 as a SET).
  // (B) flash decode-attention gate. Enabled by NNTR_MHA_GPU_DECODE env
  // (global testing override) OR per-LAYER via the gpu_decode_attn property
  // (DEFAULT-ON for gemma4 / gemma2). Env read static; property read PER-CALL.
  static const bool _mha_gpu_decode_env =
    std::getenv("NNTR_MHA_GPU_DECODE") != nullptr;
  const bool _mha_gpu_decode =
    _mha_gpu_decode_env || std::get<props::GpuDecodeAttn>(mha_core_props).get();
  // The decode term above (gpu_decode_attn, a model property) says nothing
  // about the engine: without a device arm a decode step skips this block.
  // (The CUDA arm is CudaComputeOps::attention now and takes every step size
  // from the op call below, so NNTR_CUDA_ATTN no longer routes here.)
  if (use_gemm_attention && (step_size > 1 || mha_device_attention_on()) &&
      (step_size >= FLASH_MIN_PREFILL || (_mha_gpu_decode && step_size == 1))) {
    // GPU two-1x1-conv attention path (paper section 3.7). Env-gated via
    // NNTR_MHA_GPU=1. FP16-Q + FP16-out only; K/V is either FP16 or, when
    // kv_int8 is set, int8 + per-(token, head) FP16 scale. Falls back to
    // the CPU path on any shape mismatch.
    static const bool _tca_on = nntr_env_on("NNTR_MHA_GPU");
    if (_tca_on &&
        query_step.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        attention_output_step.getDataType() ==
          ml::train::TensorDim::DataType::FP16 &&
        head_dim > 0 && num_heads_KV > 0 && num_heads_Q % num_heads_KV == 0) {
#ifdef ENABLE_FP16
#if defined(ENABLE_OPENCL) // GPU int8 two_conv / image attention path
      const uint16_t *Q_p =
        reinterpret_cast<const uint16_t *>(query_step.getData<_FP16>());
      uint16_t *O_p =
        reinterpret_cast<uint16_t *>(attention_output_step.getData<_FP16>());
      const bool svm_ok = query_step.getMemoryData() &&
                          b_cached_key.getMemoryData() &&
                          b_cached_value.getMemoryData() &&
                          attention_output_step.getMemoryData() &&
                          query_step.getMemoryData()->isSVM() &&
                          b_cached_key.getMemoryData()->isSVM() &&
                          b_cached_value.getMemoryData()->isSVM() &&
                          attention_output_step.getMemoryData()->isSVM();
      bool ok = false;
      // int8 GPU prefill is wired but currently produces degraded output
      // (model generates early-EOS); kept behind a second env var
      // (NNTR_KV_INT8_GPU=1) until the numerical issue is isolated. Default
      // for kv_int8 + NNTR_MHA_GPU is to fall through to the CPU NEON path
      // which is verified working.
      static const bool _tca_int8_on =
        std::getenv("NNTR_KV_INT8_GPU") != nullptr;
      if (kv_int8 && _tca_int8_on) {
        const int8_t *K_i8 =
          reinterpret_cast<const int8_t *>(b_cached_key.getData<uint8_t>());
        const int8_t *V_i8 =
          reinterpret_cast<const int8_t *>(b_cached_value.getData<uint8_t>());
        NNTR_THROW_IF(cur_kv_int8_key_scale_batch == nullptr ||
                        cur_kv_int8_value_scale_batch == nullptr,
                      std::invalid_argument)
          << "kv_int8 GPU prefill missing per-batch scale pointers";
        ok = nntrainer::two_conv_attention_prefill_f16_kvi8_cl(
          Q_p, K_i8, V_i8, cur_kv_int8_key_scale_batch,
          cur_kv_int8_value_scale_batch, O_p, step_size, cache_to, num_heads_Q,
          num_heads_KV, head_dim, is_causal, svm_ok);
      } else if (kv_int8) {
        // kv_int8 + GPU prefill not enabled: fall through to CPU path
        // (the host flash arm behind ComputeOps::attention dequantizes on
        // the fly).
        ok = false;
      } else {
        // Adreno image attention (the texture-cache prefill path).
        // Read K/V via image2d_from_buffer (read_imageui texture cache) instead
        // of the SVM buffer flash kernel. The SVM KV cache can't back an image
        // (clCreateImage needs a cl_mem handle), so keep per-layer cl_mem OHWI
        // mirrors (lazy-init), scatter this step's rotated K / raw V from the
        // SVM cache slice into them, then attention reads the images. Gated by
        // NNTR_KV_IMG_ATTN (Adreno only — read_imageui won't build on Intel
        // NEO, which keeps the flash path below). Preempts flash on success.
        if (use_image_attn < 0) {
          // Value-checked (imageAttnRequested in kv_ring.h) so
          // NNTR_KV_IMG_ATTN=0 really disables the image path.
          use_image_attn = img_arm_want ? 1 : 0;
        }
        // Sliding-window layers: the OHWI image kernels take the effective
        // window (win_img below) and mask keys older than it (n + W <= m).
        // Without that lower bound, once cache_to exceeds the window they
        // would compute full causal attention over evicted keys (gemma4
        // W=512: a 999-tok Adreno prefill degenerated into word salad).
        // The OHWI image kernels take a window but not a ring capacity. A
        // ringed layer reaches them through a SLIDING mirror instead (see
        // kv_mirror_base): linear addressing over a ring-cap-high mirror.
        if (use_image_attn == 1 && svm_ok && !kv_int8 && head_dim % 8 == 0) {
          // NNTR_KV_MIRROR_CAP (experiment): clamp the OHWI mirror S_max.
          // A reference build at S_max=1024 ran qk_matmul_f16_ohwi_img
          // 4.7x faster than S_max=2048 (59.8 vs ~281ms at M=1024, same
          // kernel/wrapper) -- this isolates whether the K-image height
          // (hKV*S_max rows) is the texture-cache culprit. NOT safe for
          // sequences beyond the cap (attention reads garbage rows).
          static const unsigned int mirror_cap = []() {
            const char *e = std::getenv("NNTR_KV_MIRROR_CAP");
            return e ? (unsigned int)std::atoi(e) : 0u;
          }();
          // [kvring-decode] A ringed layer's mirror is one launch's span, not
          // the whole ring (kv_mirror_rows_want, set at finalize from W and the
          // prefill chunk); a linear layer's is the KV cache height, as before.
          unsigned int S_max = nntrainer::ohwi_mirror_capacity(
            kv_mirror_rows_want ? kv_mirror_rows_want : cache_key_dim.height());
          if (mirror_cap >= 8 && mirror_cap < S_max)
            S_max = (mirror_cap + 7u) & ~7u;
          if (!kv_mirror_init) {
            kv_mirror_S_max = S_max;
            // Same derivation as the finalize prebuild (which is Android-only).
            kv_kimg_gsh = nntrainer::kimg_gsh_for(
              num_heads_KV,
              nntrainer::ohwi_mirror_capacity(
                (unsigned int)std::get<nntrainer::props::MaxTimestep>(
                  mha_core_props)
                  .get()));
            bool m_ok =
              nntrainer::create_ohwi_kv_mirror(
                /*is_v=*/false, num_heads_KV, head_dim, S_max,
                reinterpret_cast<cl_mem *>(&k_buf_ohwi),
                reinterpret_cast<cl_mem *>(&k_image_ohwi), kv_kimg_gsh) &&
              nntrainer::create_ohwi_kv_mirror(
                /*is_v=*/true, num_heads_KV, head_dim, S_max,
                reinterpret_cast<cl_mem *>(&v_buf_ohwi),
                reinterpret_cast<cl_mem *>(&v_image_ohwi));
            kv_mirror_init = m_ok;
            if (!m_ok) {
              use_image_attn = 0; // permanent disable; flash takes over
              mha_report_image_mirror_refused(num_heads_KV, head_dim, S_max);
            }
          }
          // [kv-window-ring] Where the mirror window sits for this call. A
          // linear layer's mirror starts at absolute row 0 and never moves.
          // A ringed layer's is only ring-cap rows high, so its base slides
          // forward to the oldest key this step can attend (cache_from-W+1):
          //  - prefill slides EAGERLY, which also keeps N_kv -- and with it the
          //    [hQ, M, N_kv] scores scratch -- near W + M instead of the cap;
          //  - decode slides only when the mirror is full (once per rows-W
          //    tokens), since a slide back-fills ~W rows from the cache.
          // The base is a multiple of 64 whenever that fits: the score band
          // and the row softmax partition the key axis on 8/64-aligned
          // boundaries, so a 64-aligned base makes the mirror-local grid the
          // absolute one shifted, and ring on == ring off bit for bit.
          //
          // [kvring-decode] This occupancy is NOT what a big prefill block
          // costs decode, though the shape of the code invites that reading:
          // the kernels below are handed N_kv = cache_to - base, so a mirror
          // that slides only when full does let N_kv track the mirror height
          // (~1536..2048 at a 1024 block, ~4608..8192 at a 4096 one). Measured
          // on Adreno with the timer below, that costs almost nothing: decode
          // per-call 0.2949 ms at N_kv 512 vs 0.3145 at 4608 vs 0.2985 at 1536,
          // i.e. 0.3 ms of a 10 ms/step gap. Bounding it moved 0.3B 16K decode
          // 31.6 -> 32.5 TPS and cost 3% at 4095 tokens, so it is not done.
          // What the block really charges decode is the prefill GPU work it
          // leaves queued -- see the commit message: end to end over 128 tokens
          // the 1024- and 4096-row blocks are within 1%.
          bool mirror_fits = false;
          if (kv_mirror_init && S_max == kv_mirror_S_max) {
            if (kv_ring_cap == 0) {
              mirror_fits = cache_to <= kv_mirror_S_max;
            } else {
              const unsigned int W = (unsigned int)local_window_size;
              const unsigned int need_lo =
                cache_from + 1u > W ? cache_from + 1u - W : 0u;
              const unsigned int al64 = need_lo & ~63u;
              unsigned int nb = kv_mirror_base;
              if (cache_from < nb)
                nb = al64; // rewind / a new request
              else if (step_size > 1 && al64 > nb)
                nb = al64;
              else if (cache_to - nb > kv_mirror_S_max)
                nb = al64;
              if (cache_to - nb > kv_mirror_S_max)
                nb = need_lo; // 64-alignment does not fit: exact floor
              mirror_fits = cache_to - nb <= kv_mirror_S_max;
              if (mirror_fits && nb != kv_mirror_base) {
                // Everything the mirror held sits at the old offsets: drop it
                // and let the back-fill below re-scatter [nb, cache_from)
                // from the (ringed, complete) SVM cache.
                kv_mirror_base = nb;
                kv_k_valid_to = nb;
                kv_v_valid_to = nb;
              }
            }
          }
          // mirror-local positions (== absolute for a linear layer)
          const unsigned int m_from = cache_from - kv_mirror_base;
          const unsigned int m_to = cache_to - kv_mirror_base;
          // Back-fill absolute rows [abs_from, abs_to) of the SVM cache into
          // the mirror. The cache is ring-indexed for a ringed layer, so the
          // span may cross the seam: one scatter per contiguous segment. With
          // no ring this is the single scatter it always was.
          // the full V view this call binds (alternates under
          // NNTR_KV_VIMG_PP)
          void *v_img_full_sel =
            (v_image_ohwi_pp != nullptr && ((v_img_pp_n++) & 1u))
              ? v_image_ohwi_pp
              : v_image_ohwi;
          auto mirror_backfill = [&](bool is_v, unsigned int abs_from,
                                     unsigned int abs_to, unsigned int stride) {
            const size_t hd = (size_t)num_heads_KV * head_dim;
            // [kv-img-write] The image the V rows must be written through is
            // the one the sv kernels read at this stride (the tight view once
            // adopted); K always goes to the full-capacity K image.
            const bool img_w =
              nntrainer::kv_img_write_enabled() && v_image_ohwi != nullptr;
            void *v_img_bf = (v_image_tight != nullptr && kv_v_img_S == stride)
                               ? v_image_tight
                               : v_img_full_sel;
            unsigned int pos = abs_from;
            while (pos < abs_to) {
              const size_t phys = cacheRow(pos);
              unsigned int seg = abs_to - pos;
              if (kv_ring_cap && (size_t)seg > (size_t)kv_ring_cap - phys)
                seg = (unsigned int)((size_t)kv_ring_cap - phys);
              // [kv-img-write] Image write first (the default); a false return
              // means the driver refused the CL2.0 program, which has already
              // disarmed the path, so fall THROUGH to the buffer scatter rather
              // than leave these rows unwritten.
              bool written = false;
              if (is_v && img_w && nntrainer::kv_img_write_mode() >= 2)
                written = nntrainer::v_scatter_ohwi_t_img_slab_cl(
                  reinterpret_cast<const uint16_t *>(
                    cache_value.getData<_FP16>() +
                    (size_t)batch * cache_value_dim.getFeatureLen()),
                  v_img_bf, num_heads_KV, head_dim, pos - kv_mirror_base,
                  pos - kv_mirror_base + seg, kv_mirror_base, kv_ring_cap,
                  abs_to);
              else if (is_v && img_w)
                written = nntrainer::v_scatter_ohwi_t_img_cl(
                  reinterpret_cast<const uint16_t *>(
                    cache_value.getData<_FP16>() +
                    (size_t)batch * cache_value_dim.getFeatureLen() +
                    phys * hd),
                  v_img_bf, seg, num_heads_KV, head_dim, stride,
                  pos - kv_mirror_base);
              else if (!is_v && img_w && k_image_ohwi != nullptr)
                written = nntrainer::k_scatter_ohwi_img_cl(
                  reinterpret_cast<const uint16_t *>(
                    cache_key.getData<_FP16>() +
                    (size_t)batch * cache_key_dim.getFeatureLen() + phys * hd),
                  k_image_ohwi, seg, num_heads_KV, head_dim, kv_mirror_S_max,
                  pos - kv_mirror_base);
              if (!written && is_v)
                nntrainer::v_scatter_ohwi_t_cl(
                  reinterpret_cast<const uint16_t *>(
                    cache_value.getData<_FP16>() +
                    (size_t)batch * cache_value_dim.getFeatureLen() +
                    phys * hd),
                  reinterpret_cast<cl_mem>(v_buf_ohwi), seg, num_heads_KV,
                  head_dim, stride, pos - kv_mirror_base);
              else if (!written)
                nntrainer::k_scatter_ohwi_cl(
                  reinterpret_cast<const uint16_t *>(
                    cache_key.getData<_FP16>() +
                    (size_t)batch * cache_key_dim.getFeatureLen() + phys * hd),
                  reinterpret_cast<cl_mem>(k_buf_ohwi), seg, num_heads_KV,
                  head_dim, kv_mirror_S_max, pos - kv_mirror_base);
              pos += seg;
            }
          };
          // A ringed layer has no other validated arm on this bundle: the
          // buffer (flash) arm under the image bundle is exact on a LINEAR
          // cache but measured wrong on a 2048-row ring on Adreno 840, and
          // the arm is fixed per layer at finalize (img_arm_want). Stop with
          // the way out rather than emit plausible text.
          NNTR_THROW_IF(kv_ring_cap != 0 && !mirror_fits, std::runtime_error)
            << "[kv-window-ring] the Adreno image-attention mirror cannot "
               "serve this step of a ringed layer (mirror_init="
            << (int)kv_mirror_init << " rows=" << kv_mirror_S_max << " step=["
            << cache_from << "," << cache_to << ") ring=" << kv_ring_cap
            << "). Set NNTR_KV_IMG_RING=0 to keep the linear KV cache under "
               "NNTR_KV_IMG_ATTN.";
          if (mirror_fits) {
            // --- Tight-stride V image (texture-cache cliff). The sv kernel
            // walks V texels along the sequence axis; a pitch sized to the
            // 2048 allocation cap instead of the live sequence wastes the
            // texture cache on padding (sv_matmul 63 -> 41ms at M=843 tight).
            // The view shares v_buf_ohwi; only the scatter stride must match
            // (the sv kernels never address V via their S_max argument).
            // NNTR_KV_VTIGHT=0 restores the full-stride image.
            static const bool v_tight_on = []() {
              const char *e = std::getenv("NNTR_KV_VTIGHT");
              return !(e && e[0] == '0');
            }();
            void *v_img_use = v_img_full_sel;
            unsigned int v_stride = kv_mirror_S_max;
            // A ringed layer keeps ONE stride for the whole run: its mirror is
            // already small (ring-cap rows) and constant, so there is no cliff
            // for a tight view to avoid, and a stride that never changes never
            // re-lays V (port of ed271ee7d's "the restride cannot recur"). A
            // tight view there also measured WRONG on Adreno 840 once the
            // stride shrank mid-run (2048 -> 1536 at the last 1024-chunk:
            // repetition loop, NNTR_KV_VTIGHT=0 exact) -- not root-caused, so
            // not offered.
            if (v_tight_on && kv_ring_cap == 0) {
              unsigned int need = (m_to + 7u) & ~7u;
              // [prefill-long] Lay the tight image out for the WHOLE prefill
              // span, not just this chunk, so the stride never changes mid
              // prefill (a change re-scatters every row already written).
              {
                const unsigned int hint = prefillSpanHint();
                if (hint > need)
                  need = (hint + 7u) & ~7u;
              }
              if (need > kv_v_img_S) {
                void *nimg = nullptr;
                // The helper rounds `need` up to the device image pitch
                // alignment (Adreno: multiples of 256 halves) and reports the
                // stride actually used; only adopt it while it still fits
                // the full-capacity buffer (else the full image is as good).
                if (nntrainer::create_ohwi_v_image_view(
                      v_buf_ohwi, num_heads_KV, head_dim, &need, &nimg)) {
                  if (need < kv_mirror_S_max) {
                    if (v_image_tight)
                      nntrainer::release_cl_mem(v_image_tight);
                    v_image_tight = nimg;
                    kv_v_img_S = need;
                  } else {
                    nntrainer::release_cl_mem(nimg);
                  }
                }
              }
              if (v_image_tight != nullptr && m_to <= kv_v_img_S) {
                v_img_use = v_image_tight;
                v_stride = kv_v_img_S;
              }
            }
            // --- Mirror content repair (stride change / decode gap). The
            // SVM K/V caches are complete (RoPE side-fill + decode host
            // writes), so re-scatter whatever the mirrors are missing:
            // a V stride change invalidates ALL prior rows; decode tokens
            // land only in the SVM cache, leaving [valid_to, cache_from)
            // missing from both mirrors on a follow-up prefill.
            static const bool _ring_trace =
              std::getenv("NNTR_IMG_RING_TRACE") != nullptr;
            if (_ring_trace)
              std::fprintf(stderr,
                           "[IMG-RING] W=%zu cap=%u from=%u to=%u base=%u "
                           "m_to=%u S_max=%u v_stride=%u cur_stride=%u "
                           "kvalid=%u vvalid=%u tightS=%u\n",
                           local_window_size, kv_ring_cap, cache_from, cache_to,
                           kv_mirror_base, m_to, kv_mirror_S_max, v_stride,
                           kv_v_cur_stride, kv_k_valid_to, kv_v_valid_to,
                           kv_v_img_S);
            if (v_stride != kv_v_cur_stride) {
              if (cache_from > kv_mirror_base) {
                // MHA_CLMEM: the re-scatter source is the concat slab --
                // gather the mirror-only rows back first (no-op otherwise).
                sync_kv_slab(cache_from);
                mirror_backfill(/*is_v=*/true, kv_mirror_base, cache_from,
                                v_stride);
              }
              kv_v_cur_stride = v_stride;
              kv_v_valid_to = cache_from;
            } else if (cache_from > kv_v_valid_to) {
              mirror_backfill(/*is_v=*/true, kv_v_valid_to, cache_from,
                              v_stride);
              kv_v_valid_to = cache_from;
            }
            if (cache_from > kv_k_valid_to) {
              mirror_backfill(/*is_v=*/false, kv_k_valid_to, cache_from,
                              kv_mirror_S_max);
              kv_k_valid_to = cache_from;
            }
            // Scatter this step's rotated K into the OHWI K mirror at row
            // cache_from, and V into the reversed-OHWI V mirror at column
            // cache_from. Staged chain (kv_stage_on): K reads the cl_mem
            // staging temp the GPU RoPE wrote (src_clmem) and V reads
            // value_step directly -- pure kernel chains, no host drain.
            // Unstaged: both read the SVM cache slices the RoPE/copy wrote
            // (left undrained on the image chain). The in-order queue keeps
            // RoPE -> scatter -> attention ordered with no explicit sync.
            const double _kvst_t1 = _kvst_on() ? _kvst_now() : 0;
            // [rq-scalar-off] Unstaged (k_stage == null) the source is the
            // step slice of the SVM K cache -- an offset-baked pointer that
            // moves every token. Read the cache's stable base at the scalar
            // row offset instead (same address, recordable).
            const uint16_t *k_sc_src = reinterpret_cast<const uint16_t *>(
              b_cache_key_step.getData<_FP16>());
            unsigned int k_sc_off = 0u;
            if (k_stage == nullptr && _kv_scalar_off) {
              const uint16_t *b =
                reinterpret_cast<const uint16_t *>(cache_key.getData<_FP16>());
              if (b != nullptr && b + _kc_step_off == k_sc_src) {
                k_sc_src = b;
                k_sc_off = (unsigned int)_kc_step_off;
              }
            }
            // [kv-img-write] Write the rows through the IMAGE the qk kernel
            // reads (write_imageui) instead of the buffer that aliases it --
            // same rows, same folded mapping, one memory path (the mirror's
            // read/write path split is what makes a just-scattered row come
            // back pre-write from the texture cache at a random chunk).
            const bool kv_img_write =
              nntrainer::kv_img_write_enabled() && k_image_ohwi != nullptr;
            if (kv_img_write) {
              static int _kvimg_logged = 0;
              if (!_kvimg_logged) {
                _kvimg_logged = 1;
                std::fprintf(stderr,
                             "[KV-IMG-WRITE] KVIMG_MARKER_A engaged: "
                             "mirrors written with write_imageui (gsh=%u "
                             "S_max=%u ring=%u vtightS=%u)\n",
                             kv_kimg_gsh, kv_mirror_S_max, kv_ring_cap,
                             kv_v_img_S);
              }
            }
            // A false return means the driver refused the CL2.0 image-write
            // program (which disarms the path process-wide); fall through to
            // the buffer scatter so the rows are still written. [rq-scalar-off]
            // Same as the K scatter below: when the source is the V cache's
            // step slice (no stage), address the stable base with a scalar row
            // offset.
            const uint16_t *v_sc_src =
              v_stage_svm != nullptr ? v_stage_svm
                                     : reinterpret_cast<const uint16_t *>(
                                         b_cache_value_step.getData<_FP16>());
            unsigned int v_sc_off = 0u;
            if (v_stage_svm == nullptr && v_stage_clmem == nullptr &&
                _kv_scalar_off) {
              const uint16_t *b = reinterpret_cast<const uint16_t *>(
                cache_value.getData<_FP16>());
              if (b != nullptr && b + _vc_step_off == v_sc_src) {
                v_sc_src = b;
                v_sc_off = (unsigned int)_vc_step_off;
              }
            }
            // [kv-scatter-fuse] A one-row decode step writes one K texel row
            // and one V texel per head; the two image scatters below are the
            // same reads and stores in ONE launch (byte-identical mirrors),
            // one launch per layer less. Only the mode-1 V write path (the
            // slab rebuild of mode 2 stays separate). NNTR_KV_SCATTER_FUSE=0
            // disables.
            static const bool _kv_scatter_fuse = []() {
              const char *e = std::getenv("NNTR_KV_SCATTER_FUSE");
              return e == nullptr || e[0] != '0';
            }();
            bool kv_fused = false;
            if (_kv_scatter_fuse && kv_img_write && step_size == 1 &&
                nntrainer::kv_img_write_mode() < 2 && v_img_use != nullptr)
              kv_fused = nntrainer::kv_scatter_ohwi_img_cl(
                k_sc_src, /*k_src_clmem=*/k_stage, k_sc_off, k_image_ohwi,
                kv_mirror_S_max, v_sc_src, /*v_src_clmem=*/v_stage_clmem,
                v_sc_off, v_img_use, v_stride, num_heads_KV, head_dim, m_from);
            if (kv_fused) {
              // both mirrors written; the separate scatters are skipped below
            } else if (!kv_img_write ||
                       !nntrainer::k_scatter_ohwi_img_cl(
                         k_sc_src, k_image_ohwi, step_size, num_heads_KV,
                         head_dim, kv_mirror_S_max, m_from,
                         /*src_clmem=*/k_stage, /*src_off=*/k_sc_off))
              nntrainer::k_scatter_ohwi_cl(
                k_sc_src, reinterpret_cast<cl_mem>(k_buf_ohwi), step_size,
                num_heads_KV, head_dim, kv_mirror_S_max, m_from,
                /*src_clmem=*/k_stage, /*src_off=*/k_sc_off);
            kv_k_valid_to = cache_to;
            const double _kvst_tk = _kvst_on() ? _kvst_now() : 0;
            // [kv-img-write] V through the image in use (the tight view once it
            // is adopted -- the image the sv kernels read IS the image the
            // scatter must write), same source and same rows as the buffer
            // scatter below.
            bool v_written = kv_fused;
            if (kv_fused) {
              // written by the fused launch above
            } else if (kv_img_write && nntrainer::kv_img_write_mode() >= 2)
              // mode 2: no image read in the write path -- the partial texels
              // come from the slab, not from the mirror.
              v_written = nntrainer::v_scatter_ohwi_t_img_slab_cl(
                reinterpret_cast<const uint16_t *>(
                  cache_value.getData<_FP16>() +
                  (size_t)batch * cache_value_dim.getFeatureLen()),
                v_img_use, num_heads_KV, head_dim, m_from, m_to, kv_mirror_base,
                kv_ring_cap, cache_to);
            else if (kv_img_write)
              v_written = nntrainer::v_scatter_ohwi_t_img_cl(
                v_sc_src, v_img_use, step_size, num_heads_KV, head_dim,
                v_stride, m_from,
                /*src_clmem=*/v_stage_clmem, /*src_off=*/v_sc_off);
            if (!v_written)
              nntrainer::v_scatter_ohwi_t_cl(
                v_sc_src, reinterpret_cast<cl_mem>(v_buf_ohwi), step_size,
                num_heads_KV, head_dim, v_stride, m_from,
                /*src_clmem=*/v_stage_clmem, /*src_off=*/v_sc_off);
            kv_v_valid_to = cache_to;
            const double _kvst_tv = _kvst_on() ? _kvst_now() : 0;
            // S3 decode: Q is rotated in place in its SVM plane (Q_p), by the
            // OHWI GPU RoPE or the host fallback; q_attn_clmem (= q_cl, the FC
            // output cl_mem) is NOT rotated. Binding it would feed the qk
            // kernel an UNROTATED Q -> degenerate decode
            // attention. For decode (step_size==1) pass null so the kernel
            // reads Q_p (the SVM query). Prefill: the concat GPU-RoPE now
            // rotates straight into the SVM plane Q_p on the image path
            // (out_clmem=null above), so the qk must ALSO read Q_p (null) --
            // binding q_cl would feed the UN-rotated FC output. Only the
            // (opt-in) staged path writes a separate cl_mem handle the qk
            // should bind (q_rope_staged).
            void *q_clmem_use = (step_size == 1 || q_rope_staged == nullptr)
                                  ? nullptr
                                  : q_attn_clmem;
            // Sliding-window layers: pass the effective window so the OHWI
            // kernels mask keys older than the window (n + W <= q_pos) —
            // same convention as the flash call below (0 = no window).
            const unsigned int win_img = (local_window_size >= (size_t)m_to)
                                           ? 0u
                                           : (unsigned int)local_window_size;
            // [prefill-long] The 3-kernel chain materializes [hQ, M, N_kv]
            // fp16 scores: 16 heads x 4096 x 16384 is 2 GiB for ONE launch.
            // Past a budget, run the query rows in sub-blocks instead; each
            // row's scores, softmax and output are independent of the other
            // rows, and a 64-row-aligned cut keeps the qk m-tiles and the sv
            // row pairs where they were, so the output is bit-identical. A
            // sub-block addresses Q/O through the SVM plane (interior
            // pointers), so O is raised to its cl_mem afterwards like any
            // other SVM-written arm.
            //
            // The scratch buffer is kept at the largest launch it ever served,
            // so the budget IS resident footprint. Two regimes, and a single
            // MiB number is the wrong answer for both (measured on Adreno,
            // 1300 MHz, warm kernel cache, ids bit-identical in every pair):
            //
            //  - SINGLE-SHOT prefill, the whole prompt in ONE launch (the key
            //    span this prefill reaches is this launch's M). Splitting it is
            //    pure loss: at a 256 MiB budget the 0.3B 4095-token cell drops
            //    18784 -> 14677 TPS (-22%) and the 1.5B 3925-token cell 3533 ->
            //    3153 (-11%). And what it needs is BOUNDED -- hQ * M^2 * 2,
            //    i.e. 268 MiB for the 0.3B at 4096 rows and 537 MiB for the
            //    1.5B -- because the plane (and so M) is now sized to the
            //    prompt and capped at kPrefillBlockCap. So never split it.
            //
            //  - CHUNKED prefill, M rows against a span that keeps growing:
            //    [hQ, 4096, 16384] fp16 is 2 GiB for one launch, so this one
            //    must be bounded -- and the SMALLER the budget, the faster AND
            //    the cheaper, monotonically, all the way down to 48 MiB. The
            //    earlier 256-vs-512 reading was right about the direction and
            //    stopped one step too early; it also divided by a
            //    prefill/decode split taken before the GPU queue drained.
            //    Re-swept end to end with the boundary drain armed
            //    (NNTR_PERF_SPLIT_DRAIN=1), 128 tokens, Adreno 840 at 1300 MHz,
            //    warm kernel cache, budget 48 / 96 / 256 / 512 MiB, e2e ms and
            //    honest MiB:
            //      0.3B  8191   3843  3840  4020  4102     960  958 1031 1287
            //      0.3B 16383   7180  7116  7800  8157     961 1059 1032 1288
            //      1.5B  8103   9056  9314  9746  9833    2073 2078 2151 2408
            //      1.5B 16203  19111 20062 21977 22804    2129 2179 2153 2409
            //    and at the plane-1024 block that prefillBlockPays() now
            //    defaults to, where the same order holds and the footprint gap
            //    widens (1.5B 16203, 32 tokens, repeated: 15535/15533 at 48 vs
            //    16731 at 96 vs 18270/18289 at 256 ms; honest 1596 vs 1708 vs
            //    2028 MiB). 48 vs 96 is a tie on the 0.3B (16383: 7032, 7011 vs
            //    7002 ms; run-to-run spread is 0.5%) and a 1.2 s win on the
            //    1.5B, so 48 it is.
            //
            // NNTR_ATTN_SCORES_MB is that chunked-regime budget, default 48.
            // The regimes are told apart by the driver's span hint, and only a
            // hint that PROVES this launch is the whole prefill skips the
            // split -- with no hint (span 0) the old bounded behaviour stands,
            // because the first chunk of a chunked prefill is indistinguishable
            // from a single-shot one by its own arguments alone. That exemption
            // is what keeps the smaller budget from touching a single-shot
            // prefill: at 999 and 919 tokens the budget changes neither TTFT
            // (189.5 vs 185.7 ms, 503.4 vs 489.7 ms) nor the footprint (631.5
            // vs 631.3, 1570.4 vs 1570.1 MiB) -- the scratch is never taken.
            static const size_t kScoresBudgetMbDefault = 48;
            static const size_t scores_budget = []() -> size_t {
              const char *e = std::getenv("NNTR_ATTN_SCORES_MB");
              const long mb = e ? std::atol(e) : (long)kScoresBudgetMbDefault;
              return (size_t)(mb > 0 ? (size_t)mb : kScoresBudgetMbDefault)
                     << 20;
            }();
            // [attn-nondet] The split is the SECOND nondeterminism contributor
            // on this driver: running a chunk's query rows in sub-blocks means
            // many launches per chunk against the same K/V images, and a
            // texture line one launch left is reused by the next (measured at
            // 0.3B 16383: every run pair differs by 545-1138 of 5408 node
            // hashes with the split, 2 of 3 runs identical without it; 1.5B
            // 8103: 705 vs 15 of 4064). One launch per chunk is therefore the
            // determinism arm, and NNTR_DETERMINISTIC -- the cross-lane knob
            // the API can set -- selects it. It is NOT free here, unlike on the
            // other lanes: it costs +24.6% prefill / +17.5% e2e / +500 MiB on
            // 0.3B 16383 and +10.4% / +6.6% / +182 MiB on 1.5B 8103 (nothing on
            // a single-shot prefill like E2B's 923 tokens, which never splits),
            // so it stays opt-in and the throughput default keeps the budget.
            static const bool scores_no_split = []() {
              const char *d = std::getenv("NNTR_DETERMINISTIC");
              return d != nullptr && d[0] != '0';
            }();
            const unsigned int _span = prefillSpanHint();
            const bool one_block =
              scores_no_split || (_span != 0u && _span <= step_size);
            const size_t row_bytes =
              (size_t)num_heads_Q * (size_t)m_to * sizeof(uint16_t);
            unsigned int sub = step_size;
            if (!one_block && step_size > 64u && q_clmem_use == nullptr &&
                row_bytes * step_size > scores_budget) {
              sub = (unsigned int)(scores_budget / row_bytes) & ~63u;
              if (sub < 64u)
                sub = 64u;
            }
            // [attn-timer] The image arm was the one arm the timer did not
            // cover, which is why "which decode op grows with the ring cap" had
            // to be read off the kernels instead of measured. N_kv here is the
            // mirror OCCUPANCY (m_to), the number the cost actually follows.
            {
              MhaAttnTimerScope _img_attn_timer(step_size, m_to,
                                                local_window_size != UINT_MAX);
              if (sub >= step_size) {
                ok =
                  nntrainer::two_conv_attention_prefill_f16_ohwi_kvimg_view_cl(
                    Q_p, reinterpret_cast<cl_mem>(k_image_ohwi),
                    reinterpret_cast<cl_mem>(v_img_use), O_p, step_size, m_to,
                    num_heads_Q, num_heads_KV, head_dim, kv_mirror_S_max,
                    is_causal, attn_logit_softcapping, /*q_clmem=*/q_clmem_use,
                    /*o_clmem=*/o_cl, win_img, v_buf_ohwi, v_stride);
                if (ok && o_cl != nullptr)
                  o_written_clmem = true;
              } else {
                const size_t q_row = (size_t)num_heads_Q * head_dim;
                ok = true;
                for (unsigned int a = 0; ok && a < step_size; a += sub) {
                  const unsigned int b = std::min(a + sub, step_size);
                  ok = nntrainer::
                    two_conv_attention_prefill_f16_ohwi_kvimg_view_cl(
                      Q_p + (size_t)a * q_row,
                      reinterpret_cast<cl_mem>(k_image_ohwi),
                      reinterpret_cast<cl_mem>(v_img_use),
                      O_p + (size_t)a * q_row, b - a, m_from + b, num_heads_Q,
                      num_heads_KV, head_dim, kv_mirror_S_max, is_causal,
                      attn_logit_softcapping,
                      /*q_clmem=*/nullptr, /*o_clmem=*/nullptr, win_img,
                      v_buf_ohwi, v_stride);
                }
                // O went to the SVM plane: the raise below lands it in o_cl.
              }
            }
            if (_kvst_on())
              _kvst_mark_scatter(_kvst_t1, _kvst_tk, _kvst_tv, _kvst_now());
            static int _img_attn_logged = 0;
            if (!_img_attn_logged) {
              _img_attn_logged = 1;
              ml_logd("[IMG-ATTN] engaged ok=%d M=%u N_kv=%u S_max=%u "
                      "hQ=%zu hKV=%zu d=%zu softcap=%.1f ring_cap=%u gsh=%u",
                      (int)ok, step_size, cache_to, kv_mirror_S_max,
                      num_heads_Q, num_heads_KV, head_dim,
                      attn_logit_softcapping, kv_ring_cap, kv_kimg_gsh);
            }
          }
        }

        // d=256 attention: register-tiled, barrier-free, scores-free flash
        // (FBQ_SG Block-Q when NNTR_V8C_BUF). The layer-graph previously only
        // had the scalar two_conv path (materializes a [hQ,M,N_kv] fp16 scores
        // tensor in DRAM, ~820ms at M=1024 d=256) and otherwise fell to host
        // flash attention (~8.5s, M=1024). Concat K (non-OHWI b_cached_key) =>
        // max_seq_len=0. SVM-only; falls through to two_conv/host when svm_ok
        // is false. The SVM-pointer fallbacks below read Q via its SVM shadow:
        // lower first when Q is GPU_CLMEM-resident and the image path did not
        // run.
        if (!ok && q_rope_staged != nullptr) {
          // The rotated Q lives only in the q_stage temp; the SVM fallbacks
          // below read Q's SVM plane. Land it there (drained) and drop the
          // (stale, un-rotated) cl_mem handle so lower_q stays a no-op.
          nntrainer::gpu_copy_f16_cl(
            reinterpret_cast<const uint16_t *>(query_step.getData<_FP16>()),
            reinterpret_cast<uint16_t *>(query_step.getData<_FP16>()),
            step_size * (unsigned int)num_heads_Q * (unsigned int)head_dim,
            /*svm_inputs=*/true, /*in_clmem=*/q_rope_staged,
            /*out_clmem=*/nullptr, /*drain=*/true);
          q_cl = nullptr;
        }
        if (!ok && q_rotated_in_svm_plane && q_cl != nullptr) {
          // The image arm did not take this call (no mirror -- the K image did
          // not fit the device, or the call is past the mirror capacity), but
          // NNTR_KV_IMG_ATTN still shaped the RoPE above: the rotated Q is in
          // the SVM plane and q_cl still holds the FC's UN-rotated output.
          // lower_q() below would copy that over the plane, and every arm
          // under this one would then attend with a position-less Q against
          // rotated keys -- fluent for a sentence, then a repetition loop, and
          // a first token that moves with the chunk size. That was the whole
          // of "the non-image fallback is wrong". Drop the stale handle
          // instead (one drain: that rotation was enqueued without one).
          nntrainer::cl_queue_finish();
          q_cl = nullptr;
        }
        if (!ok && (kv_stage_on || k_stage != nullptr ||
                    v_stage_svm != nullptr || kv_write_undrained)) {
          // The staged K/V cache side-fills (and, in staged mode, the
          // undrained rope-Q SVM output) were never drained; the
          // flash/two_conv SVM readers below depend on them. One drain.
          nntrainer::cl_queue_finish();
        }
        // MHA_CLMEM: the SVM fallbacks below read the concat slab, which
        // the mirror-only prefill window left stale -- gather it back.
        if (!ok)
          sync_kv_slab(cache_to);
        if (!ok)
          lower_q();
        // The flash and two_conv arms over the concat cache are
        // ClComputeOps::attention's (step 7a): a step the image arm did not
        // take falls through to the op call below, which runs them first.
      }
      if (ok) {
        // The wo FC consumes O through its static plane: raise it into the
        // planner sub-buffer when a non-cl_mem path here (the image arm's
        // sub-blocks, kvi8) wrote the SVM shadow instead. The flash/two_conv
        // arms' O is ClComputeOps::attention's to raise, the same way.
        if (o_cl != nullptr && !o_written_clmem) {
          // The flash/two_conv prefill kernel wrote O into the SVM shadow and,
          // on the in-order SVM-pool queue, deliberately SKIPS its post-kernel
          // drain (attention_kernels.cpp) on the assumption that the next GPU
          // op orders behind it. But clmem_raise_cl below is NOT a GPU->GPU
          // dependency: it reads the SVM shadow as a plain HOST pointer
          // (the clEnqueueWriteBuffer source). The host view of a kernel-
          // written coarse-grained SVM region is not coherent until the
          // kernel completes, so at large prefill M (sliding-window prefill at
          // ctx > window is the first config that runs a long flash kernel on
          // this path) the raise can snapshot a partially-written O => corrupt
          // last-position hidden state => non-deterministic immediate EOS /
          // garbage. Drain once here so the host source is coherent before the
          // raise. Measured no prefill regression (M=1024 ~737 TPS, identical
          // to the undrained path within noise) because the flash kernel has
          // effectively completed by the time this layer's O is consumed; the
          // host gemm path below wrote O on the host and needs no such drain.
          // GPU-side O raise (o_svm -> o_cl, GPU->GPU on the in-order queue):
          // the host clmem_raise_cl reads the O SVM shadow as a HOST pointer
          // (clEnqueueWriteBuffer source), which is NOT coherent on
          // coarse-grain SVM (Xe3 / Panther Lake) even after cl_queue_finish --
          // a finish drains the queue but does NOT map the SVM for host access,
          // so the raise snapshots a stale/partially-written O. A short prefill
          // (M~4) exposes this: the flash completes just before the raise ->
          // corrupt last-position hidden -> garbage (large-M survived because
          // the flash finished long before the raise). The GPU copy needs no
          // host coherence and is perf-neutral (no drain, no host bounce).
          // DEFAULT-ON; override with NNTR_NO_GPU_OBRIDGE for the legacy
          // host-raise path.
          static const bool _gpu_obridge =
            std::getenv("NNTR_NO_GPU_OBRIDGE") == nullptr;
          if (_gpu_obridge) {
            uint16_t *o_svm = reinterpret_cast<uint16_t *>(
              attention_output_step.getData<_FP16>());
            nntrainer::gpu_copy_f16_cl(
              o_svm, o_svm, (unsigned int)attention_output_step.size(),
              /*svm_inputs=*/true, /*in_clmem=*/nullptr, /*out_clmem=*/o_cl,
              /*drain=*/false);
          } else {
            nntrainer::cl_queue_finish();
            nntrainer::clmem_raise_cl(attention_output_step, 0);
          }
        }
        return;
      }
#endif // ENABLE_OPENCL (GPU two-conv/image/flash attention)
#endif
    }
  }

  // The host arms, or the CUDA arm inside CudaComputeOps::attention, through
  // the op. The host arms read Q and the K/V cache and write O on the host.
#if defined(ENABLE_OPENCL)
  // A K/V cache write above may have been enqueued with a submission flush
  // and no drain (the image-attention GPU chain). This reader is on the
  // host, not on that queue: settle the queue once first.
  if (kv_write_undrained)
    nntrainer::cl_queue_finish();
  settle_act_stage_for_host();
#endif
  lower_q();
  // MHA_CLMEM: the host reads the whole prefix from the concat slab; gather
  // the prefill rows (mirror-only during the prefill window) back once.
  // This step's own K/V were host-written into the slab above.
  sync_kv_slab(cache_from);
  if (mha_clmem_mode && cache_to > kv_slab_synced_to)
    kv_slab_synced_to = cache_to;
  // A planner cl_mem O is raised by the op: on the gpu engine either a
  // device kernel or a host arm wrote it, and the two need different raises.
  attend(query_step, b_cached_key, b_cached_value, attention_output_step,
         cache_from, cache_to, nullptr);
}

void MHACoreLayer::one_batch_incremental_forwarding(
  const unsigned int batch, const unsigned int _from, const unsigned int from,
  const unsigned int to, nntrainer::Tensor &query_step,
  nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
  nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
  nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
  ml::train::TensorDim &cache_key_step_dim,
  ml::train::TensorDim &cache_value_dim,
  ml::train::TensorDim &cache_value_step_dim, nntrainer::Tensor &sink_step) {
  /// @todo replace from, to into cache_index, input height
  /// @note currently, only gpt-oss uses this method

  // Static GPU_CLMEM residency, defensive bridge: this (host-heavy) sink
  // variant is not cl_mem-converted -- lower any resident I/O up front and
  // raise the output at exit so a GPU_CLMEM classification cannot leave it
  // reading/writing a stale plane. No-ops when the classes are SVM.
#if defined(ENABLE_OPENCL)
  // GPU_CLMEM lower/raise bridge is OpenCL-only (I/O is plain host memory
  // without it).
  if (query_step.isClMem())
    nntrainer::clmem_lower_cl(query_step, 0);
  if (key_step.isClMem())
    nntrainer::clmem_lower_cl(key_step, 0);
  if (value_step.isClMem())
    nntrainer::clmem_lower_cl(value_step, 0);
  struct ORaise {
    nntrainer::Tensor &t;
    bool by_op = false; ///< the attention op raised it
    ~ORaise() {
      if (!by_op && t.isClMem())
        nntrainer::clmem_raise_cl(t, 0);
    }
  } _oraise{attention_output_step};
#endif

  /**
   *  cache_key
   *  +--------+                        ->
   *  |        |                        ->
   *  |        |                        ->
   *  |........| from                   ->
   *  |........| to -> b_cache_key_step -> b_cached_key
   *  |        |
   *  +--------+
   *
   */

  // NNTR_KV_INT8 is wired through the Qwen3 entry point only. The
  // gpt-oss/sink_step variant has no int8 write/read plumbing yet.
  NNTR_THROW_IF(kv_int8, std::invalid_argument)
    << "NNTR_KV_INT8 is not supported for the sink_step (gpt-oss) variant of "
       "one_batch_incremental_forwarding";

  apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, _from, true);

  // K rotated and V into rows [from, to). from is _from: a step past the cache
  // end (the only case where they differ) is rejected before this.
  write_kv_cache(batch, key_step, value_step, cache_key, cache_value, _from,
                 /*write_key=*/true, /*write_value=*/true,
                 /*rotate_key=*/true);

  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  // [kv-window-ring] No clamp here: this is the attention-sink overload, and
  // causallm::kvRingLayerEligible() returns false for a use_sink layer, so
  // kv_ring_cap is 0 by construction and the cache is the full max_seq height.
  // A clamp would read as though the ring could reach this path.
  cached_key_dim.height(to);
  cached_value_dim.height(to);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  // skip_prefill (see the non-sink overload above): K/V are written into the
  // cache; skip the unused prefill attention compute + output write. The
  // _oraise guard above raises the (untouched) output on return -- a no-op for
  // the SVM/host case and harmless for cl_mem since the downstream wo FC also
  // skips the prefill big-step.
  if (skip_prefill && (to - from) > 1)
    return;

  attend(query_step, b_cached_key, b_cached_value, attention_output_step, from,
         to, &sink_step);
#if defined(ENABLE_OPENCL)
  _oraise.by_op = true;
#endif
}

/************************************************************** */

/**
 * @brief rotary embedding-related member function
 * @note seq_len -> max_position_embeddings
 */
void MHACoreLayer::precompute_freqs(int head_dim, unsigned int seq_len,
                                    float theta, bool is_fp16) {
  const std::string cache_key =
    std::string(is_fp16 ? "fp16:" : "fp32:") + std::to_string(head_dim) + ":" +
    std::to_string(seq_len) + ":" + std::to_string(theta) + ":" +
    rope_scaling_type + ":" + std::to_string(scale) + ":" +
    std::to_string(original_max_position_embeddings) + ":" +
    std::to_string(rope_partial_rotary_factor);

  auto cache_iter = rope_freq_cache.find(cache_key);
  if (cache_iter != rope_freq_cache.end()) {
    if (is_fp16) {
#ifdef ENABLE_FP16
      freqs_cos_fp16 = &cache_iter->second.cos_fp16;
      freqs_sin_fp16 = &cache_iter->second.sin_fp16;
      freqs_cos_fp16_flat = &cache_iter->second.cos_fp16_flat;
      freqs_sin_fp16_flat = &cache_iter->second.sin_fp16_flat;
#endif
    } else {
      freqs_cos = &cache_iter->second.cos;
      freqs_sin = &cache_iter->second.sin;
    }
    return;
  }

  thetas.clear();
  if (rope_scaling_type == "yarn")
    _compute_yarn_parameters(head_dim, theta);
  else if (rope_scaling_type == "proportional" ||
           rope_partial_rotary_factor != 1.0f)
    // Proportional rope (Gemma3n/Gemma4 E2B). Also routes here when a
    // partial_rotary_factor < 1 is configured even under the "default" type,
    // since _compute_proportional_parameters is the one that zeroes the
    // non-rotary tail of the frequency table.
    _compute_proportional_parameters(head_dim, theta);
  else {
    // "default" plus any model-specific type we don't special-case (e.g.
    // Gemma4/Gemma3n "linear"/local rope) -> standard RoPE frequencies. Warn
    // once so the unhandled type is visible without aborting the run.
    if (rope_scaling_type != "default") {
      static bool warned = false;
      if (!warned) {
        warned = true;
        ml_logw("[mha_core] rope_scaling_type='%s' not special-cased; "
                "using default RoPE.",
                rope_scaling_type.c_str());
      }
    }
    _compute_default_parameters(head_dim, theta);
  }

  unsigned int half_ = head_dim / 2;
  auto &cache = rope_freq_cache[cache_key];

  if (!is_fp16) {
    // cos / sin as [seq_len, half_] rows, the nntrainer::RotaryTable layout.
    // calc_trigonometric_vals_dup writes a head_dim-wide row whose second half
    // repeats the first; the rotary kernels read only the first half.
    cache.cos.assign((size_t)seq_len * half_, 0.0f);
    cache.sin.assign((size_t)seq_len * half_, 0.0f);
    std::vector<float> cos_row(head_dim);
    std::vector<float> sin_row(head_dim);

    // update cos / sin frequency
    for (unsigned int i = 0; i < seq_len; ++i) {

      // cpu_backend provides calc_trigonometric_vals_dup on every arch (NEON
      // on ARM, scalar fallback elsewhere), so the app-side USE_NEON fork is
      // gone — one code path, one rounding behaviour across backends.
      nntrainer::calc_trigonometric_vals_dup(half_, thetas.data(),
                                             cos_row.data(), sin_row.data(), i,
                                             attention_scaling);
      std::copy_n(cos_row.data(), half_, &cache.cos[(size_t)i * half_]);
      std::copy_n(sin_row.data(), half_, &cache.sin[(size_t)i * half_]);
    }
    freqs_cos = &cache.cos;
    freqs_sin = &cache.sin;
  }

#ifdef ENABLE_FP16
  if (is_fp16) {
    // cos / sin for FP16
    cache.cos_fp16.assign(seq_len, std::vector<_FP16>(head_dim, 0));
    cache.sin_fp16.assign(seq_len, std::vector<_FP16>(head_dim, 0));
    cache.cos_fp16_flat.assign((size_t)seq_len * half_, 0);
    cache.sin_fp16_flat.assign((size_t)seq_len * half_, 0);

    std::vector<float> cos_tmp(head_dim);
    std::vector<float> sin_tmp(head_dim);

    for (unsigned int i = 0; i < seq_len; ++i) {
      // Same as the FP32 branch: one arch-neutral cpu_backend call, no
      // app-side USE_NEON fork.
      nntrainer::calc_trigonometric_vals_dup(half_, thetas.data(),
                                             cos_tmp.data(), sin_tmp.data(), i,
                                             attention_scaling);
      for (unsigned int j = 0; j < head_dim; ++j) {
        cache.cos_fp16[i][j] = (_FP16)cos_tmp[j];
        cache.sin_fp16[i][j] = (_FP16)sin_tmp[j];
      }
      std::copy_n(cache.cos_fp16[i].data(), half_,
                  &cache.cos_fp16_flat[(size_t)i * half_]);
      std::copy_n(cache.sin_fp16[i].data(), half_,
                  &cache.sin_fp16_flat[(size_t)i * half_]);
    }
    freqs_cos_fp16 = &cache.cos_fp16;
    freqs_sin_fp16 = &cache.sin_fp16;
    freqs_cos_fp16_flat = &cache.cos_fp16_flat;
    freqs_sin_fp16_flat = &cache.sin_fp16_flat;
  }
#endif
};

void MHACoreLayer::_compute_default_parameters(int head_dim, float theta) {

  // no attention scaling
  attention_scaling = 1.0f;

  // theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... , dim/2]
  // head_dim should be divisible by 2
  unsigned int half_ = head_dim / 2;
  for (unsigned int i = 0; i < half_; ++i) {
    thetas.push_back(1.0 /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }
}

void MHACoreLayer::_compute_proportional_parameters(int head_dim, float theta) {

  // no attention scaling for proportional rope
  attention_scaling = 1.0f;

  // Partial rotary: only the first rope_angles frequencies receive rotary
  // embedding; the rest of the head_dim/2 entries are zeroed so cos=1/sin=0,
  // i.e. those channels pass through unrotated. With
  // rope_partial_rotary_factor == 1.0 this reduces to default RoPE scaled by
  // 1/scale.
  const int half_dim = static_cast<int>(head_dim / 2);
  const int rope_angles =
    static_cast<int>((rope_partial_rotary_factor * head_dim) / 2.0f);
  thetas.reserve(half_dim);
  for (int i = 0; i < rope_angles; ++i)
    thetas.push_back(1.0f /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  for (int i = rope_angles; i < half_dim; ++i)
    thetas.push_back(0.0f);
  for (auto &val : thetas)
    val /= scale;
}

void MHACoreLayer::_compute_yarn_parameters(int head_dim, float theta) {

  // Config parameters
  ///@todo partial_rotary_factor should be generalized to fully support
  /// transformers's implementation
  // const float partial_rotary_factor = has_partial_rotary_factor ?
  // config_partial_rotary_factor : 1.0f;
  const float partial_rotary_factor = 1.0f;
  const int dim = static_cast<int>(head_dim * partial_rotary_factor);
  const float base = theta;

  // Handle max position embeddings

  // Attention scaling calculation (simplified from Python version)
  auto get_mscale = [](float scale, float mscale = 1.0f) {
    return (scale <= 1.0f) ? 1.0f : (0.1f * mscale * std::log(scale) + 1.0f);
  };

  ///@todo attention_scaling should be generalized to fully support
  /// transformers's implementation
  // if (has_mscale && has_mscale_all_dim) {
  // attention_scaling = get_mscale(factor, mscale) / get_mscale(factor,
  // mscale_all_dim);
  // } else {
  // attention_scaling = get_mscale(factor);
  // }
  attention_scaling = get_mscale(scale);

  ///@todo attention_scaling should be generalized to fully support
  /// transformers's implementation
  // const float beta_fast = has_beta_fast ? config_beta_fast : 32.0f;
  // const float beta_slow = has_beta_slow ? config_beta_slow : 1.0f;
  // const bool truncate = has_truncate ? config_truncate : true;
  // Beta parameters
  const float beta_fast = 32.0f;
  const float beta_slow = 1.0f;
  const bool truncate = false;

  // Helper functions
  auto find_correction_dim = [&](float num_rotations) {
    return (dim * std::log(original_max_position_embeddings /
                           (num_rotations * 2 * M_PI))) /
           (2 * std::log(base));
  };

  auto [low, high] = [&]() {
    float low_val = find_correction_dim(beta_fast);
    float high_val = find_correction_dim(beta_slow);
    if (truncate) {
      low_val = std::floor(low_val);
      high_val = std::ceil(high_val);
    }
    return std::make_pair(low_val, high_val);
  }();

  // Compute position frequencies
  thetas.resize(dim / 2);

  // Compute interpolation and extrapolation frequencies
  std::vector<float> inv_freq_interpolation;
  std::vector<float> inv_freq_extrapolation;
  for (size_t i = 0; i < dim / 2; ++i) {
    inv_freq_extrapolation.push_back(
      1.0 / (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
    inv_freq_interpolation.push_back(
      1.0 / (scale * std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }

  auto linear_ramp_factor = [](float min, float max, int size) {
    if (min == max) {
      max += 0.001f; // Prevent singularity
    }
    std::vector<float> ramp(size);
    for (int i = 0; i < size; ++i) {
      float val = (i - min) / (max - min);
      ramp[i] = std::clamp(val, 0.0f, 1.0f);
    }
    return ramp;
  };

  std::vector<float> inv_freq_extrapolation_factor =
    linear_ramp_factor(low, high, dim / 2);
  for (auto &val : inv_freq_extrapolation_factor) {
    val = 1.0f - val;
  }

  // Combine frequencies
  for (size_t i = 0; i < thetas.size(); ++i) {
    thetas[i] =
      inv_freq_extrapolation[i] * inv_freq_extrapolation_factor[i] +
      inv_freq_interpolation[i] * (1.0f - inv_freq_extrapolation_factor[i]);
  }
}

void MHACoreLayer::apply_rotary_emb_tensor_v2(nntrainer::Tensor &in,
                                              nntrainer::Tensor &out,
                                              unsigned int dim,
                                              unsigned int from,
                                              bool apply_rope) {
  if (in.getDataType() != ml::train::TensorDim::DataType::FP32 &&
      in.getDataType() != ml::train::TensorDim::DataType::FP16)
    return;
  in.getOps()->rotary_embedding(
    in, out, dim, rotary_table(in.getDataType(), dim), from, apply_rope);
}

nntrainer::RotaryTable
MHACoreLayer::rotary_table(ml::train::TensorDim::DataType dt,
                           unsigned int dim) {
  nntrainer::RotaryTable table;
  table.half = dim / 2;
  if (dt == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    // The same table the OpenCL RoPE uses, capped at rope_lut_positions()
    // (the live max sequence length, not max_position_embeddings).
    ensure_rope_flat_lut();
    table.cos_fp16 = rope_cos_flat_cur;
    table.sin_fp16 = rope_sin_flat_cur;
    table.positions = rope_flat_positions;
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
    return table;
  }
  if (cached_freqs_cos == nullptr || cached_freqs_sin == nullptr) {
    const std::lock_guard<std::mutex> lock(rope_init_mtx);
    precompute_freqs(head_dim, max_position_embeddings, theta, false);
    cached_freqs_cos = freqs_cos;
    cached_freqs_sin = freqs_sin;
  }
  table.cos = cached_freqs_cos->data();
  table.sin = cached_freqs_sin->data();
  table.positions = (unsigned int)(cached_freqs_cos->size() / table.half);
  return table;
}

void MHACoreLayer::write_kv_cache(unsigned int batch,
                                  nntrainer::Tensor &key_step,
                                  nntrainer::Tensor &value_step,
                                  nntrainer::Tensor &cache_key,
                                  nntrainer::Tensor &cache_value,
                                  unsigned int pos, bool write_key,
                                  bool write_value, bool rotate_key) {
  auto batch_plane = [batch](nntrainer::Tensor &cache) {
    ml::train::TensorDim dim = cache.getDim();
    dim.batch(1);
    return cache.getSharedDataTensor(dim, batch * dim.getFeatureLen(), true);
  };
  nntrainer::Tensor key_plane, value_plane;
  nntrainer::KVCacheView kv;
  if (write_key) {
    key_plane = batch_plane(cache_key);
    kv.key = &key_plane;
  }
  if (write_value) {
    value_plane = batch_plane(cache_value);
    kv.value = &value_plane;
  }
  kv.head_dim = head_dim;
  kv.ring_cap = kv_ring_cap;
  kv.layout = kv_layout;
  if (kv_int8) {
    // The int8 cache's fp16 scales, from row 0 of this batch's plane.
    const size_t scale_batch_offset =
      batch * kv_int8_key_scale->getDim().getFeatureLen();
    kv.key_scale = kv_int8_key_scale->getData<uint16_t>() + scale_batch_offset;
    kv.value_scale =
      kv_int8_value_scale->getData<uint16_t>() + scale_batch_offset;
  }

  nntrainer::RotaryTable rope;
  if (rotate_key && write_key)
    rope = rotary_table(key_step.getDataType(), head_dim);
  key_step.getOps()->kv_cache_write(key_step, value_step, kv, pos,
                                    rotate_key && write_key ? &rope : nullptr);
}

void MHACoreLayer::attend(nntrainer::Tensor &query_step,
                          nntrainer::Tensor &cached_key,
                          nntrainer::Tensor &cached_value,
                          nntrainer::Tensor &output, unsigned int from,
                          unsigned int to, nntrainer::Tensor *sink) {
  nntrainer::KVCacheView kv;
  kv.key = &cached_key;
  kv.value = &cached_value;
  kv.head_dim = head_dim;
  kv.ring_cap = kv_ring_cap;
  kv.key_scale = cur_kv_int8_key_scale_batch;
  kv.value_scale = cur_kv_int8_value_scale_batch;

  nntrainer::AttentionOpParams p;
  p.num_heads_q = num_heads_Q;
  p.q_pos = from;
  p.kv_len = to;
  p.causal = is_causal;
  p.window = local_window_size;
  p.softcap = attn_logit_softcapping;
  p.sink = sink;
  p.flash = use_gemm_attention;
  static const bool gpu_decode_env =
    std::getenv("NNTR_MHA_GPU_DECODE") != nullptr;
  p.device_decode =
    gpu_decode_env || std::get<props::GpuDecodeAttn>(mha_core_props).get();
  query_step.getOps()->attention(query_step, kv, output, p);
}

void MHACoreLayer::setBatch(nntrainer::RunLayerContext &context,
                            unsigned int batch) {

  const float dropout_rate =
    std::get<nntrainer::props::DropOutRate>(mha_core_props).get();
  context.updateTensor(tensor_idx[AttentionParams::cache_key], batch);
  context.updateTensor(tensor_idx[AttentionParams::cache_value], batch);
  // context.updateTensor(tensor_idx[AttentionParams::attention_weight], batch);
  if (dropout_rate > epsilon) {
    context.updateTensor(tensor_idx[AttentionParams::dropout_mask], batch);
  }
}

void MHACoreLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  unsigned int height = input_dimensions[0].height();
  unsigned int &max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();
  unsigned int &max_new_tokens =
    std::get<props::MaxNewTokens>(mha_core_props).get();
  max_position_embeddings =
    std::get<props::MaxPositionEmbeddings>(mha_core_props).get();
  max_timestep = height + max_new_tokens;

  ml::train::TensorDim kv_dim = input_dimensions[0];
  kv_dim.width(kv_dim.width() / (num_heads_Q / num_heads_KV));

  ml::train::TensorDim kv_cache_dim = kv_dim;
#ifdef ENABLE_FP16
  kv_cache_dim.setDataType(ml::train::TensorDim::DataType::FP16);
#else
  kv_cache_dim.setDataType(ml::train::TensorDim::DataType::UINT16);
#endif
  kv_cache_dim.height(max_timestep);

  context.updateInput(INOUT_INDEX::QUERY, input_dimensions[0]);
  context.updateInput(INOUT_INDEX::KEY, kv_dim);
  context.updateInput(INOUT_INDEX::VALUE, kv_dim);
  context.updateOutput(0, input_dimensions[0]);

  context.updateTensor(tensor_idx[AttentionParams::cache_key], kv_cache_dim);
  context.updateTensor(tensor_idx[AttentionParams::cache_value], kv_cache_dim);
}

void MHACoreLayer::calcDerivative(nntrainer::RunLayerContext &context) {}

void MHACoreLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void MHACoreLayer::exportTo(nntrainer::Exporter &exporter,
                            const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(mha_core_props, method, this);
}

void MHACoreLayer::setProperty(const std::vector<std::string> &values) {
  std::vector<std::string> props;
  props.reserve(values.size());
  for (const auto &value : values) {
    std::string key;
    std::string parsed_value;
    if (nntrainer::getKeyValue(value, key, parsed_value) == ML_ERROR_NONE &&
        key == "cache_index") {
      setCacheIndex(static_cast<unsigned int>(std::stoul(parsed_value)));
    } else {
      props.push_back(value);
    }
  }

  auto remain_props = loadProperties(props, mha_core_props);
  LayerImpl::setProperty(remain_props);
  cached_freqs_cos = nullptr;
  cached_freqs_sin = nullptr;
#ifdef ENABLE_FP16
  cached_freqs_cos_fp16 = nullptr;
  cached_freqs_sin_fp16 = nullptr;
  rope_cos_flat_cur = nullptr;
  rope_sin_flat_cur = nullptr;
#endif
}

size_t MHACoreLayer::calc_attn_index(size_t i) { return (i * (i + 1)) / 2; };

#ifdef PLUGGABLE

nntrainer::Layer *create_mha_core_layer() {
  auto layer = new MHACoreLayer();
  return layer;
}

void destroy_mha_core_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_mha_core_layer,
                                                   destroy_mha_core_layer};
}

#endif

} // namespace causallm

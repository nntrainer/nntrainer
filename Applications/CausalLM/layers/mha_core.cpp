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
#include <cmath>
#include <cstring>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

static std::mutex rope_init_mtx;

#include <fp16.h>
#include <layer_context.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <thread_manager.h>
#include <util_func.h>

#include <atomic>
#include <cstdint>
#include <dlfcn.h>
#include <unordered_map>



inline float convert_scalar(uint16_t h) {
  return nntrainer::compute_fp16_to_fp32(h);
}

namespace causallm {

namespace {
/**
 * @brief Flash attention bridge function type.
 * Dispatches Q/K/V/mask to DSP via nntr_htp_bridge_flash_attn.
 */
using flash_attn_fn = int (*)(const void *, const void *, const void *,
                              const void *, void *, unsigned int, unsigned int,
                              unsigned int, unsigned int, unsigned int, float,
                              int, int);

/**
 * @brief Lazily dlopen libggml-hexagon.so and dlsym nntr_htp_bridge_flash_attn.
 * Cached for process lifetime - same pattern as hexagon_compute_ops.cpp.
 */
flash_attn_fn get_flash_attn_bridge() {
  static flash_attn_fn fn = []() -> flash_attn_fn {
    fprintf(stderr, "[FLASH_ATTN] attempting dlopen(libggml-hexagon.so)\n");
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      const char *err = dlerror();
      fprintf(stderr, "[FLASH_ATTN] dlopen FAILED: %s\n", err);
      ml_logw("MHACore: dlopen(libggml-hexagon.so) failed: %s "
              "(flash attention disabled, using CPU path)",
              err);
      return nullptr;
    }
    fprintf(stderr, "[FLASH_ATTN] dlopen SUCCESS, handle=%p\n", handle);

    fprintf(stderr, "[FLASH_ATTN] attempting dlsym(handle, nntr_htp_bridge_flash_attn)\n");
    void *s = dlsym(handle, "nntr_htp_bridge_flash_attn");
    if (!s) {
      const char *err = dlerror();
      fprintf(stderr, "[FLASH_ATTN] dlsym FAILED: %s\n", err);
      ml_logw("MHACore: dlsym(nntr_htp_bridge_flash_attn) failed: %s "
              "(flash attention disabled, using CPU path)",
              err);
      return nullptr;
    }
    fprintf(stderr, "[FLASH_ATTN] dlsym SUCCESS, function ptr=%p\n", s);

    fprintf(stderr, "[FLASH_ATTN] bridge loaded successfully\n");
    ml_logi("MHACore: flash attention bridge loaded successfully");
    return reinterpret_cast<flash_attn_fn>(s);
  }();
  return fn;
}

/**
 * @brief RoPE DSP bridge function type.
 * Dispatches in-place rotary position embedding to the cDSP via
 * nntr_htp_bridge_rope (HTP_OP_ROPE). F32-only.
 */
using rope_fn = int (*)(float *, const int32_t *, unsigned int,
                        unsigned int, unsigned int, float, int);

/**
 * @brief Lazily dlopen libggml-hexagon.so and dlsym nntr_htp_bridge_rope.
 * Same pattern as get_flash_attn_bridge().
 */
rope_fn get_rope_bridge() {
  static rope_fn fn = []() -> rope_fn {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      ml_logw("MHACore: dlopen(libggml-hexagon.so) failed: %s "
              "(RoPE DSP offload disabled, using CPU path)", dlerror());
      return nullptr;
    }
    void *s = dlsym(handle, "nntr_htp_bridge_rope");
    if (!s) {
      ml_logw("MHACore: dlsym(nntr_htp_bridge_rope) failed: %s "
              "(RoPE DSP offload disabled, using CPU path)", dlerror());
      return nullptr;
    }
    ml_logi("MHACore: RoPE DSP bridge loaded successfully");
    return reinterpret_cast<rope_fn>(s);
  }();
  return fn;
}

/**
 * @brief FP16 RoPE DSP bridge function type.
 * Dispatches in-place rotary position embedding on __fp16 tensors via
 * nntr_htp_bridge_rope_f16. The DSP kernel casts F16->F32 internally,
 * runs F32 rope, and casts back — all in one op (replaces the 3-op
 * cast-rotate-cast chain).
 */
using rope_f16_fn = int (*)(__fp16 *, const int32_t *, unsigned int,
                            unsigned int, unsigned int, float, int);

/**
 * @brief Lazily dlopen libggml-hexagon.so and dlsym nntr_htp_bridge_rope_f16.
 */
rope_f16_fn get_rope_f16_bridge() {
  static rope_f16_fn fn = []() -> rope_f16_fn {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      return nullptr;
    }
    void *s = dlsym(handle, "nntr_htp_bridge_rope_f16");
    if (!s) {
      ml_logw("MHACore: dlsym(nntr_htp_bridge_rope_f16) failed: %s "
              "(FP16 RoPE will use cast-rotate-cast chain)", dlerror());
      return nullptr;
    }
    ml_logi("MHACore: FP16 RoPE DSP bridge loaded successfully");
    return reinterpret_cast<rope_f16_fn>(s);
  }();
  return fn;
}


/**
 * @brief Force a DSP sync if a begin_batch/end_batch scope is currently
 * open, otherwise do nothing.
 *
 * Every nntr_htp_bridge_* call above only enqueues without executing when
 * a batch is open (that's the whole point of batching) - the actual DSP
 * computation doesn't happen until something flushes. If this layer is
 * about to read a tensor's raw CPU memory that an earlier enqueued-but-
 * unflushed op (a Q/K/V projection GEMM, or the RoPE dispatch above) was
 * supposed to have written, that read sees whatever was there before, not
 * the result - silently wrong attention, not just slow attention. Call
 * this immediately before any such read (see its call site in
 * one_batch_incremental_forwarding, right before the K/V cache-append
 * copies).
 */
void flush_if_batch_active() {
  using IsBatchActiveFn = int (*)();
  using FlushFn = void (*)();
  static IsBatchActiveFn is_batch_active = nullptr;
  static FlushFn do_flush = nullptr;
  static bool tried = false;
  if (!tried) {
    tried = true;
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (handle) {
      is_batch_active =
        (IsBatchActiveFn)dlsym(handle, "nntr_htp_bridge_is_batch_active");
      do_flush = (FlushFn)dlsym(handle, "nntr_htp_bridge_flush");
    }
  }
  if (is_batch_active && do_flush && is_batch_active()) {
    do_flush();
  }
}

/**
 * @brief §6.2: DSP-side copy bridge for KV-cache append.
 * Dispatches HTP_OP_CPY to the cDSP, allowing K/V cache append to happen
 * on the NPU without forcing a CPU-side flush+copy. This eliminates the
 * 5/block explicit flush_if_batch_active() calls in mha_core.
 * dst_is_fp16=1 downcasts F32 src to the cache's actual F16 storage dtype
 * (op_cpy's existing same-shape F32->F16 kernel, htp/cpy-ops.c).
 */
using cpy_fn = int (*)(const void *, void *, unsigned int, int, int);

cpy_fn get_cpy_bridge() {

  static cpy_fn fn = []() -> cpy_fn {
    void *handle = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) return nullptr;
    return reinterpret_cast<cpy_fn>(dlsym(handle, "nntr_htp_bridge_cpy"));
  }();
  return fn;
}

/**
 * @brief Try a straight (no-rotation) DSP copy for KV-cache append via
 * HTP_OP_CPY. `src` and `dst` must have the same element count and be a
 * contiguous view (true for the n_tokens-row cache-append slices this is
 * called with - see one_batch_incremental_forwarding). Handles FP32/FP16 on
 * either side (the DSP kernel supports all four combinations - see
 * htp/cpy-ops.c). Any other dtype (e.g. UINT16 cache), non-cdsp engine, or
 * missing bridge falls back to the caller's CPU path, unchanged.
 *
 * No flush_if_batch_active() needed before this call, unlike the CPU
 * fallback: the DSP enqueue just chains after whatever DSP op already
 * produced `src`, instead of forcing the host to wait for it first.
 *
 * On this model's actual on-device dtypes (Q/K/V activations and the KV
 * cache are both FP16 - see Applications/CausalLM/models/transformer.cpp,
 * cache_dtype="FP16" on ARM/Android), this only ever fires for the V-cache
 * append: V needs no rotation, so it's a pure copy every time. K's append
 * is NOT a pure copy on this model - key_step.getDataType()==FP32 (the
 * DSP RoPE bridge's dtype gate) is false for FP16 activations, so K's
 * rotation never dispatches to the DSP RoPE kernel and always falls
 * through to the CPU path, which computes the actual rotation (not just a
 * copy) and therefore must flush and read real data - this helper can't
 * remove that flush without an FP16-capable DSP RoPE kernel, which does
 * not exist today (htp/rope-ops.c is F32-only). The K call site below is
 * kept for the case where DSP RoPE does succeed (F32 activations, or a
 * future FP16 RoPE kernel).
 */
bool try_dsp_cache_copy(bool is_cdsp_engine, nntrainer::Tensor &src,
                        nntrainer::Tensor &dst) {
  if (!is_cdsp_engine)
    return false;
  auto to_flag = [](ml::train::TensorDim::DataType t) -> int {
    if (t == ml::train::TensorDim::DataType::FP32)
      return 0;
#ifdef ENABLE_FP16
    if (t == ml::train::TensorDim::DataType::FP16)
      return 1;
#endif
    return -1;
  };
  int src_is_fp16 = to_flag(src.getDataType());
  int dst_is_fp16 = to_flag(dst.getDataType());
  if (src_is_fp16 < 0 || dst_is_fp16 < 0)
    return false;
  cpy_fn fn = get_cpy_bridge();
  if (!fn)
    return false;
  unsigned int n_elems = static_cast<unsigned int>(src.size());
  const void *src_ptr = src_is_fp16
#ifdef ENABLE_FP16
                          ? static_cast<const void *>(src.getData<_FP16>())
#else
                          ? nullptr
#endif
                          : static_cast<const void *>(src.getData<float>());
  void *dst_ptr = dst_is_fp16
#ifdef ENABLE_FP16
                    ? static_cast<void *>(dst.getData<_FP16>())
#else
                    ? nullptr
#endif
                    : static_cast<void *>(dst.getData<float>());
  int rc = fn(src_ptr, dst_ptr, n_elems, src_is_fp16, dst_is_fp16);
  if (rc != 0) {
    ml_logw("MHACore: DSP cache-copy bridge failed (rc=%d), falling back to CPU", rc);
    return false;
  }
  return true;
}

/**
 * @brief Reusable FP16 scratch tensor for the Q/K/V/O FP32->FP16 staging
 * cast in forwarding()'s `#if ENABLE_FP16 && __ANDROID__` block.
 *
 * Measured root cause of a regression: that block used to do
 * `nntrainer::Tensor(dim, true)` - a brand-new heap allocation - for
 * Q_step/K_step/V_step/O_step on *every* call, i.e. every one of the 28
 * transformer blocks, every prefill. Since that memory was never part of
 * any rpcmem region the DSP already knows about, every
 * `nntr_htp_bridge_cpy()` touching it was a pool "miss" - a fresh,
 * synchronous rpcmem registration RPC that can't be deferred into the open
 * batch. Measured: this alone produced 112 of 141 real FastRPC
 * round-trips in a 28-layer/909-token prefill
 * (`pool_stats cpy:dst/src: 140 hit(s), 112 miss(es)`) - see
 * docs/backend_guide/AGENT_HANDOFF_2026-08-20.md.
 *
 * Every layer in this model shares the same head_dim/num_heads, so one
 * grow-once (never-shrink) scratch tensor per role, held in the caller's
 * function-static storage (shared across all 28 MHACoreLayer instances,
 * since `forwarding()` has exactly one compiled body), means only the
 * very first layer's call ever registers a new buffer - the other 27 get
 * a plain `getSharedDataTensor()` view into memory the bridge has already
 * seen (a pool hit, no RPC).
 */
nntrainer::Tensor
get_reusable_fp16_scratch(nntrainer::Tensor &owner,
                          const nntrainer::TensorDim &want_dim) {
  if (owner.empty() || owner.size() < want_dim.getDataLen()) {
    owner = nntrainer::Tensor(want_dim, true);
    // Reusing the same C++ Tensor/pointer across layers is necessary but
    // NOT sufficient: nntr_htp_bridge_cpy's hit/miss check
    // (nntr_htp_bridge_find_ext_pool) only checks the bridge's OWN
    // registered-pool table, not "have I seen this address before" - a
    // stable pointer that was never registered still misses on every
    // single call. Register it once here, same as RopeScratchRpcMem does
    // for its own scratch buffer, so all subsequent reuses are real hits.
    using RegisterFn = int (*)(const void *, size_t);
    static RegisterFn register_pool = []() -> RegisterFn {
      void *bridge = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
      if (!bridge) return nullptr;
      return reinterpret_cast<RegisterFn>(
        dlsym(bridge, "nntr_htp_bridge_register_activation_pool"));
    }();
    if (register_pool) {
      int rc = register_pool(owner.getData<char>(), owner.bytes());
      if (rc != 0) {
        ml_logw("MHACore: register_activation_pool failed (rc=%d) for FP16 "
                "staging scratch (%zu bytes) - this buffer will miss the "
                "pool on every use",
                rc, owner.bytes());
      }
    }
    return owner;
  }
  return owner.getSharedDataTensor(want_dim, 0, true);
}

// ---------------------------------------------------------------------------
// FP16 RoPE via cast-rotate-cast: the DSP RoPE kernel (htp/rope-ops.c) is
// F32-only, but this model's Q/K activations are FP16 - so instead of a new
// FP16 RoPE kernel, chain three DSP ops that already exist and enqueue them
// into the same batch (FIFO-ordered, no host flush needed between or before
// them - same chaining guarantee nntr_htp_bridge_ffn_swiglu already relies
// on for its 5 sequentially-dependent ops):
//   1. cpy (F16->F32): cast the FP16 activation into a scratch F32 buffer
//   2. rope (F32, in-place): rotate the scratch buffer
//   3. cpy (F32->F16): cast the rotated result into its FP16 destination
//      (the KV cache for K, or back into the activation tensor for Q)
// A single scratch rpcmem buffer is allocated and registered ONCE (not per
// layer/per call - registering many small pools was found to hang the DSP
// after ~12-13 registrations, see RMSNormLayer's disabled gamma-rpcmem
// attempt in rms_norm.cpp) and reused for every block's Q and K calls in
// turn, since prefill processes one block, one tensor, at a time.
// ---------------------------------------------------------------------------
namespace {

class RopeScratchRpcMem {
public:
  static RopeScratchRpcMem &global() {
    static RopeScratchRpcMem inst;
    return inst;
  }

  // Returns a scratch F32 buffer with room for at least `n_elems` floats,
  // or nullptr if rpcmem/the bridge is unavailable or n_elems exceeds the
  // buffer's fixed capacity (sized generously for this model at
  // construction; see kMaxElems below).
  float *get(unsigned int n_elems) {
    if (!usable() || n_elems > kMaxElems) {
      return nullptr;
    }
    return buf_;
  }

  bool usable() const { return buf_ != nullptr; }

private:
  // Sized for one block's largest single Q or K call: num_heads * head_dim
  // per token, up to 1024 tokens - matching the existing gemm_q4_0/
  // ffn_swiglu "M>1024 activation rows" bridge limit (docs/backend_guide:
  // "add graceful CPU fallback for M>1024 activation rows"), since prefill
  // already falls back to CPU wholesale past that length anyway. 32 heads *
  // 128 head_dim * 1024 tokens covers Qwen3-0.6B (16 Q heads) with margin
  // for larger head counts.
  static constexpr unsigned int kMaxElems = 32u * 128u * 1024u;

  using AllocFn = void *(*)(int, uint32_t, int);
  using RegisterFn = int (*)(const void *, size_t);

  static constexpr int kHeapIdSystem = 25;
  static constexpr int kDefaultFlags = 1;

  RopeScratchRpcMem() {
    void *rpc = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_GLOBAL);
    if (!rpc) {
      ml_logw("MHACore: dlopen(libcdsprpc.so) failed: %s", dlerror());
      return;
    }
    auto alloc = reinterpret_cast<AllocFn>(dlsym(rpc, "rpcmem_alloc"));
    if (!alloc) {
      ml_logw("MHACore: dlsym(rpcmem_alloc) failed: %s", dlerror());
      return;
    }

    void *bridge = dlopen("libggml-hexagon.so", RTLD_NOW | RTLD_GLOBAL);
    if (!bridge) {
      ml_logw("MHACore: dlopen(libggml-hexagon.so) failed: %s", dlerror());
      return;
    }
    auto register_pool = reinterpret_cast<RegisterFn>(
      dlsym(bridge, "nntr_htp_bridge_register_activation_pool"));
    if (!register_pool) {
      ml_logw("MHACore: dlsym(nntr_htp_bridge_register_activation_pool) "
              "failed: %s",
              dlerror());
      return;
    }

    size_t bytes = static_cast<size_t>(kMaxElems) * sizeof(float);
    void *p = alloc(kHeapIdSystem, kDefaultFlags, static_cast<int>(bytes));
    if (!p) {
      ml_logw("MHACore: rpcmem_alloc(%zu) failed for RoPE scratch buffer; "
              "FP16 RoPE will stay on CPU",
              bytes);
      return;
    }
    if (register_pool(p, bytes) != 0) {
      ml_logw("MHACore: bridge rejected RoPE scratch pool %p (%zu bytes); "
              "FP16 RoPE will stay on CPU",
              p, bytes);
      return;
    }
    buf_ = static_cast<float *>(p);
  }

  float *buf_ = nullptr;
};

} // namespace

/**
 * @brief Try FP16 RoPE via the cast-rotate-cast chain described above.
 * `dst` may alias `src.getData()` for Q (in-place rotation, no cache
 * involved) or be a separate KV-cache view for K. Returns false (caller
 * falls back to CPU) if any step can't be dispatched - `src` is never
 * mutated by this function (only the scratch buffer is), so the CPU
 * fallback path re-reads unmodified original data, exactly as if this had
 * never been attempted.
 *
 * Re-enabled: the original 14% regression was measured when the
 * flush_if_batch_active() calls in the CPU fallback path were no-ops
 * (nothing was pending on the DSP at that point). Now that the QKV input
 * copies use try_dsp_cache_copy (Op 2 fix), there ARE pending DSP ops
 * when K-RoPE runs, making the CPU fallback's flush a real round-trip.
 * The cast-chain eliminates that real flush, so the trade-off is now:
 * 3 extra HVX dispatches vs 1 real FastRPC flush per block. Re-measure
 * to confirm whether this is now a net win.
 */
bool try_dsp_fp16_rope(bool is_cdsp_engine, nntrainer::Tensor &src,
                       nntrainer::Tensor &dst, const int32_t *positions,
                       unsigned int n_tokens, unsigned int n_heads,
                       unsigned int head_dim, float theta) {

  if (!is_cdsp_engine)
    return false;
#ifdef ENABLE_FP16
  if (src.getDataType() != ml::train::TensorDim::DataType::FP16 ||
      dst.getDataType() != ml::train::TensorDim::DataType::FP16)
    return false;

  // Try the single-op FP16 RoPE bridge first (nntr_htp_bridge_rope_f16).
  // The DSP kernel (rope_job_f16 in rope-ops.c) casts F16->F32 internally,
  // runs F32 rope, and casts back to F16 — all in one DSP op. This replaces
  // the old 3-op cast-rotate-cast chain (cpy F16->F32 + rope F32 + cpy
  // F32->F16) that produced 3× as many FastRPC round-trips.
  const rope_f16_fn &rope_f16_dsp = get_rope_f16_bridge();
  if (rope_f16_dsp) {
    int rc = rope_f16_dsp(src.getData<_FP16>(), positions, n_tokens, n_heads,
                          head_dim, theta, 2);
    if (rc == 0) {
      // Single-op F16 RoPE succeeded. If src != dst, copy result to dst.
      // (For Q rotation, src==dst and this is a no-op; for K rotation,
      // rope_f16_dsp writes in-place to src, so copy to dst/cache.)
      if (&src != &dst) {
        if (!try_dsp_cache_copy(is_cdsp_engine, src, dst)) {
          flush_if_batch_active();
          dst.copyData(src);
        }
      }
      return true;
    }
    ml_logw("MHACore: FP16 RoPE DSP bridge failed (rc=%d), falling back to "
            "cast-rotate-cast chain",
            rc);
  }

  // Fallback: 3-op cast-rotate-cast chain (for older DSP libs without
  // nntr_htp_bridge_rope_f16).
  cpy_fn cpy = get_cpy_bridge();
  const rope_fn &rope_dsp = get_rope_bridge();
  if (!cpy || !rope_dsp)
    return false;
  unsigned int n_elems = static_cast<unsigned int>(src.size());
  float *scratch = RopeScratchRpcMem::global().get(n_elems);
  if (!scratch)
    return false;

  int rc = cpy(static_cast<const void *>(src.getData<_FP16>()),
              static_cast<void *>(scratch), n_elems, /*src_is_fp16=*/1,
              /*dst_is_fp16=*/0);
  if (rc != 0) {
    ml_logw("MHACore: DSP cast-in (F16->F32) failed (rc=%d) for FP16 RoPE, "
            "falling back to CPU",
            rc);
    return false;
  }

  rc = rope_dsp(scratch, positions, n_tokens, n_heads, head_dim, theta, 2);
  if (rc != 0) {
    ml_logw("MHACore: DSP RoPE on scratch buffer failed (rc=%d), falling "
            "back to CPU",
            rc);
    return false;
  }

  rc = cpy(static_cast<const void *>(scratch),
          static_cast<void *>(dst.getData<_FP16>()), n_elems,
          /*src_is_fp16=*/0, /*dst_is_fp16=*/1);
  if (rc != 0) {
    ml_logw("MHACore: DSP cast-out (F32->F16) failed (rc=%d) for FP16 RoPE, "
            "falling back to CPU",
            rc);
    return false;
  }
  return true;
#else
  return false;
#endif
}


// Forward declaration — build_causal_mask is defined below in this namespace.
void build_causal_mask(std::vector<uint16_t> &mask, unsigned int n_tokens,
                       unsigned int n_kv, unsigned int cache_from);

/**
 * @brief §6.3: Cached causal mask to avoid rebuilding it on every layer.
 * The causal mask for a given (n_tokens, n_kv, cache_from) is identical
 * across all 28 transformer blocks during prefill. Cache it so the CPU
 * build_causal_mask loop runs once instead of 28 times.
 */
static std::unordered_map<uint64_t, std::vector<uint16_t>> g_causal_mask_cache;

const std::vector<uint16_t> & get_cached_causal_mask(
    unsigned int n_tokens, unsigned int n_kv, unsigned int cache_from) {
  // Cache key: pack (cache_from, n_kv, n_tokens) into 64 bits
  uint64_t key = ((uint64_t)cache_from << 40) | ((uint64_t)n_kv << 20) | (uint64_t)n_tokens;
  auto it = g_causal_mask_cache.find(key);
  if (it != g_causal_mask_cache.end()) {
    return it->second;
  }
  // Build and cache
  std::vector<uint16_t> & mask = g_causal_mask_cache[key];
  build_causal_mask(mask, n_tokens, n_kv, cache_from);
  return mask;
}



/**
 * @brief Check if flash attention should be used.
 * Enabled for prefill (step_size > 1) with head_dim a multiple of 64

 * (HMX fast path). The bridge (nntr_htp_bridge_flash_attn) and the DSP
 * kernel (hmx_flash_attn_ext in flash-attn-ops.c) both require
 * head_dim % 64 == 0 — so head_dim=64 (Qwen3-0.6B) and head_dim=128
 * (Qwen3-4B/8B) are both supported.
 */
bool should_use_flash_attn(unsigned int step_size, unsigned int head_dim,
                           bool is_prefill) {

  static const char *env = std::getenv("NNTR_HEXAGON_FLASH_ATTN");
  bool enabled = (env && std::atoi(env) == 1);

  if (!enabled) {
    return false;
  }

  // Verbose mode: log every gate evaluation
  static const char *verbose_env = std::getenv("NNTR_HEXAGON_FLASH_ATTN_VERBOSE");
  static const bool verbose = (verbose_env && std::atoi(verbose_env) == 1);

  if (!is_prefill || step_size <= 1) {
    if (verbose)
      fprintf(stderr, "[FLASH_ATTN] gate: REJECT (not prefill or step_size<=1)\n");
    return false;
  }

  // Below this, the per-layer FastRPC round trip (~1.6-1.8ms/layer measured
  // on S25/HTP79 at 137-203 tokens, see HEXAGON_NPU_OBSERVATION_LOG.md S31)
  // is not amortized by the O(step_size^2) CPU attention cost it replaces:
  // measured net loss below ~150 tokens, net win from ~200 tokens on.
  // Overridable for re-measurement on other devices/models.
  static const char *min_tok_env = std::getenv("NNTR_HEXAGON_FLASH_ATTN_MIN_TOKENS");
  static const unsigned int min_tokens = min_tok_env ? (unsigned int)std::atoi(min_tok_env) : 160;
  if (step_size < min_tokens) {
    if (verbose)
      fprintf(stderr, "[FLASH_ATTN] gate: REJECT (step_size=%u < min_tokens=%u)\n",
              step_size, min_tokens);
    return false;
  }

  // HMX requires head_dim to be a multiple of 64. Both the bridge
  // (nntr_htp_bridge_flash_attn: head_dim % 64 != 0 check) and the DSP kernel
  // (hmx_flash_attn_ext: k->ne[0] % 64 == 0 check) enforce this.
  // Qwen3-0.6B has head_dim=64, Qwen3-4B/8B has head_dim=128 — both pass.
  if (head_dim % 64 != 0 || head_dim == 0) {
    if (verbose)
      fprintf(stderr, "[FLASH_ATTN] gate: REJECT (head_dim=%u not multiple of 64)\n", head_dim);
    return false;
  }


  // Log ACCEPT only once per forward pass (first layer)
  static std::atomic<bool> logged_accept{false};
  if (!logged_accept.exchange(true)) {
    fprintf(stderr, "[FLASH_ATTN] gate: ACCEPT (step_size=%u head_dim=%u), loading bridge...\n",
            step_size, head_dim);
  }

  const flash_attn_fn &fn = get_flash_attn_bridge();
  bool result = (fn != nullptr);
  if (verbose)
    fprintf(stderr, "[FLASH_ATTN] gate: bridge=%p, result=%d\n", fn, (int)result);

  return result;
}

/**
 * @brief Build causal mask for flash attention.
 * mask[i][j] = 0 if j < cache_from + i + 1, else -INF (0xFC00).
 * Layout: [n_tokens, n_kv] F16, row-major.
 */
void build_causal_mask(std::vector<uint16_t> &mask, unsigned int n_tokens,
                       unsigned int n_kv, unsigned int cache_from) {
  mask.resize(n_tokens * n_kv);
  for (unsigned int i = 0; i < n_tokens; ++i) {
    const unsigned int valid_to = cache_from + i + 1;
    for (unsigned int j = 0; j < n_kv; ++j) {
      mask[i * n_kv + j] = (j < valid_to) ? 0x0000 : 0xFC00;
    }
  }
}

} // namespace

#define tile_size 4

static void compute_kcaches_fp32_reference(
  const float *in, const float *kcache, float *output, int num_rows,
  int num_cache_head, int head_dim, int gqa_size, size_t local_window_size,
  int head_start = 0, int head_end = -1) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  const int window = static_cast<int>(
    std::min(static_cast<size_t>(num_rows), local_window_size));
  const int start_row = num_rows - window;
  const float inv_sqrt_head_dim =
    1.0f / std::sqrt(static_cast<float>(head_dim));

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int g = 0; g < gqa_size; ++g) {
      const float *query = in + (n * gqa_size + g) * head_dim;
      for (int row = start_row; row < num_rows; ++row) {
        const float *key = kcache + (row * num_cache_head + n) * head_dim;
        float sum = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          sum += query[d] * key[d];
        }
        output[(row - start_row) * num_cache_head * gqa_size + n * gqa_size +
               g] = sum * inv_sqrt_head_dim;
      }
    }
  }
}

static void compute_vcache_fp32_transposed_reference(
  int row_num, const float *in, const float *vcache, float *output,
  int num_cache_head, int gqa_size, int head_dim, size_t local_window_size,
  int head_start = 0, int head_end = -1) {
  const int actual_head_end = (head_end < 0) ? num_cache_head : head_end;
  NNTR_THROW_IF(head_start >= actual_head_end, std::invalid_argument)
    << "head_start (" << head_start << ") must be less than head_end ("
    << actual_head_end << ")";

  const int window = static_cast<int>(
    std::min(static_cast<size_t>(row_num + 1), local_window_size));
  const int start_row = row_num + 1 - window;

  for (int n = head_start; n < actual_head_end; ++n) {
    for (int h = 0; h < gqa_size; ++h) {
      float *out = output + (n * gqa_size + h) * head_dim;
      std::fill(out, out + head_dim, 0.0f);

      for (int row = start_row; row <= row_num; ++row) {
        const int attn_row = row - start_row;
        const float a_val =
          in[attn_row * (num_cache_head * gqa_size) + n * gqa_size + h];
        const float *value = vcache + (row * num_cache_head + n) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          out[d] += a_val * value[d];
        }
      }
    }
  }
}

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
    props::SlidingWindow(), props::MaxNewTokens(), props::RopeTheta(),
    props::UseRope(), props::MaxPositionEmbeddings(), props::UseSink(),
    props::RopeScalingType(), props::RopeScalingFactor(),
    props::RopePartialRotaryFactor(), props::RopeScalingMaxPositionEmbeddings(),
    props::AttnLogitSoftcapping(), props::IsCausal()),
  sm(nntrainer::ActivationType::ACT_SOFTMAX),
  epsilon(1e-3),
  cache_index(0),
  num_heads_Q(0),
  num_heads_KV(0),
  head_dim(0),
  cache_shift(false) {
  tensor_idx.fill(std::numeric_limits<unsigned>::max());
}

MHACoreLayer::~MHACoreLayer() {}

/************************************************************** */

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

  /** use rope */
  use_rope = std::get<props::UseRope>(mha_core_props).get();

  /** cache compute_engine for one_batch_incremental_forwarding(), which has
   * no RunLayerContext of its own to query it from directly */
  is_cdsp_engine =
    context.getComputeEngineType() == ml::train::LayerComputeEngine::CDSP;

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

  if (!std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill =
      std::get<nntrainer::props::SkipPrefill>(*layer_impl_props).get();

  /** Tensor for KV-Cache (only allocate internally when not using external
   * cache) */
  if (!use_external_cache) {
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

  theta = (float)std::get<props::RopeTheta>(mha_core_props).get();

  /** set Output dimension! - one output */
  std::vector<nntrainer::TensorDim> output_dims(1);
  output_dims[0] = input_dims[0];
  output_dims[0].width(head_dim * num_heads_Q);
  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions(output_dims);
}

/************************************************************** */

/**
 * @note In external KV cache mode (use_external_cache == true), this
 *       implements the inference forward pass using cache tensors supplied
 *       as input[3] (cache_key) and input[4] (cache_value). The host (e.g.
 *       KVCacheManager via setExternalTensors) is responsible for owning
 *       these buffers and for calling setCacheIndex() before each step to
 *       set the write position. After this call cache_index is advanced by
 *       input.height().
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
    return;
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
#if ENABLE_FP16 && defined(__ANDROID__)
      nntrainer::TensorDim Q_step_dim = query_step_dim;
      nntrainer::TensorDim K_step_dim = key_step_dim;
      nntrainer::TensorDim V_step_dim = value_step_dim;
      nntrainer::TensorDim O_step_dim = output_step_dim;
      Q_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      K_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      V_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      O_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);

      // Shared across all 28 MHACoreLayer instances - see
      // get_reusable_fp16_scratch()'s doc comment for why.
      static nntrainer::Tensor Q_scratch_owner, K_scratch_owner,
        V_scratch_owner, O_scratch_owner;
      nntrainer::Tensor Q_step =
        get_reusable_fp16_scratch(Q_scratch_owner, Q_step_dim);
      nntrainer::Tensor K_step =
        get_reusable_fp16_scratch(K_scratch_owner, K_step_dim);
      nntrainer::Tensor V_step =
        get_reusable_fp16_scratch(V_scratch_owner, V_step_dim);
      nntrainer::Tensor O_step =
        get_reusable_fp16_scratch(O_scratch_owner, O_step_dim);

      bool cdsp = context.getComputeEngineType() ==
                  ml::train::LayerComputeEngine::CDSP;
      if (!try_dsp_cache_copy(cdsp, query_step, Q_step))
        Q_step.copyData(query_step);
      if (!try_dsp_cache_copy(cdsp, key_step, K_step))
        K_step.copyData(key_step);
      if (!try_dsp_cache_copy(cdsp, value_step, V_step))
        V_step.copyData(value_step);

      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, from, from, to, Q_step, K_step, V_step, O_step, cache_key,
          cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
          cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(
          batch, from, from, to, Q_step, K_step, V_step, O_step, cache_key,
          cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
          cache_value_step_dim);
      }
      if (!try_dsp_cache_copy(cdsp, O_step, output_step))
        output_step.copyData(O_step);
#else
      {
        if (use_sink) {
          one_batch_incremental_forwarding(
            batch, from, from, to, query_step, key_step, value_step,
            output_step, cache_key, cache_value, cache_key_dim,
            cache_key_step_dim, cache_value_dim, cache_value_step_dim, sink);
        } else {
          one_batch_incremental_forwarding(
            batch, from, from, to, query_step, key_step, value_step,
            output_step, cache_key, cache_value, cache_key_dim,
            cache_key_step_dim, cache_value_dim, cache_value_step_dim);
        }
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
 *       Please note that Transformer Decoder's MHA takes only one sequence at a
 * step. Incremental forwarding function is used for this.
 */
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

  /// @todo replace step_size into input height
  unsigned int step_size = _to - _from;

  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  unsigned int from = _from;
  unsigned int to = _to;

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
#if ENABLE_FP16 && defined(__ANDROID__)
      nntrainer::TensorDim Q_step_dim = query_step_dim;
      nntrainer::TensorDim K_step_dim = key_step_dim;
      nntrainer::TensorDim V_step_dim = value_step_dim;
      nntrainer::TensorDim O_step_dim = output_step_dim;
      Q_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      K_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      V_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);
      O_step_dim.setDataType(ml::train::TensorDim::DataType::FP16);

      // Shared across all 28 MHACoreLayer instances - see
      // get_reusable_fp16_scratch()'s doc comment for why.
      static nntrainer::Tensor Q_scratch_owner, K_scratch_owner,
        V_scratch_owner, O_scratch_owner;
      nntrainer::Tensor Q_step =
        get_reusable_fp16_scratch(Q_scratch_owner, Q_step_dim);
      nntrainer::Tensor K_step =
        get_reusable_fp16_scratch(K_scratch_owner, K_step_dim);
      nntrainer::Tensor V_step =
        get_reusable_fp16_scratch(V_scratch_owner, V_step_dim);
      nntrainer::Tensor O_step =
        get_reusable_fp16_scratch(O_scratch_owner, O_step_dim);

      bool cdsp = context.getComputeEngineType() ==
                  ml::train::LayerComputeEngine::CDSP;
      if (!try_dsp_cache_copy(cdsp, query_step, Q_step))
        Q_step.copyData(query_step);
      if (!try_dsp_cache_copy(cdsp, key_step, K_step))
        K_step.copyData(key_step);
      if (!try_dsp_cache_copy(cdsp, value_step, V_step))
        V_step.copyData(value_step);
      if (use_sink) {
        one_batch_incremental_forwarding(
          batch, _from, from, to, Q_step, K_step, V_step, O_step, cache_key,
          cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
          cache_value_step_dim, sink);
      } else {
        one_batch_incremental_forwarding(
          batch, _from, from, to, Q_step, K_step, V_step, O_step, cache_key,
          cache_value, cache_key_dim, cache_key_step_dim, cache_value_dim,
          cache_value_step_dim);
      }
      if (!try_dsp_cache_copy(cdsp, O_step, output_step))
        output_step.copyData(O_step);
#else
      {
        if (use_sink) {
          one_batch_incremental_forwarding(
            batch, _from, from, to, query_step, key_step, value_step,
            output_step, cache_key, cache_value, cache_key_dim,
            cache_key_step_dim, cache_value_dim, cache_value_step_dim, sink);
        } else {
          one_batch_incremental_forwarding(
            batch, _from, from, to, query_step, key_step, value_step,
            output_step, cache_key, cache_value, cache_key_dim,
            cache_key_step_dim, cache_value_dim, cache_value_step_dim);
        }
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

/**
 * @brief Function to compute Attention Scores using Tensor inputs. Wrapper
 * around nntrainer::compute_kcaches with multi-threading support
 *
 * Expected Input Shapes:
 * @param in (Query): [Batch, 1, sequence_len, Num_Heads_Q * Head_Dim]
 * @param cache (Key Cache): [Batch, 1, Max_Timestep, Num_Heads_KV * Head_Dim]
 * @param out (Attention Score): [Batch, 1, 1, Num_Heads_Q * Context_Len]
 *            where Context_Len is usually the current timestep 'to'.
 *
 */
void MHACoreLayer::compute_kcaches(nntrainer::Tensor &in,
                                   nntrainer::Tensor &cache,
                                   nntrainer::Tensor &out, unsigned int from,
                                   size_t sequence_len, unsigned int num_head,
                                   unsigned int group_size,
                                   unsigned int head_dim) {

  // Dispatch based on data type (FP32 or FP16)
  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (sequence_len == 1) {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_to_compute = is_causal ? from + 1 : from + sequence_len;
      unsigned int num_cache_head = num_head / group_size;

      // Use ThreadManager for lower overhead parallelization during decoding
      const float *in_data = in.getData<float>();
      float *out_data = out.getData<float>();

      auto &tm = nntrainer::ThreadManager::Global();
      if (cache.getDataType() == ml::train::TensorDim::DataType::FP32) {
        const float *cache_data = cache.getData<float>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            compute_kcaches_fp32_reference(
              in_data, cache_data, out_data, row_to_compute, num_cache_head,
              head_dim, group_size, local_window_size, head_kv, head_kv + 1);
          });
      } else {
        const uint16_t *cache_data = cache.getData<uint16_t>();
        tm.parallel_for(0, static_cast<size_t>(num_cache_head),
                        [=](size_t head_kv) {
                          nntrainer::compute_kcaches<uint16_t>(
                            in_data, cache_data, out_data, row_to_compute,
                            num_cache_head, head_dim, group_size, tile_size,
                            local_window_size, head_kv, head_kv + 1);
                        });
      }

    } else {
      // Sequence processing (prefill or chunked)
      // Iterate over ALL query rows so that no row is skipped even when
      // sequence_len > local_window_size.
      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(sequence_len), [=](size_t i) {
        float *input_addr = in.getData<float>() + num_head * head_dim * i;
        int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
        // Windowed cumulative offset so that each row's scores are placed
        // contiguously after the previous row's scores (respecting the window).
        size_t out_start_row = is_causal ? calc_windowed_attn_index(from + i) -
                                             calc_windowed_attn_index(from)
                                         : i * (from + sequence_len);
        float *output_addr = out.getData<float>() + out_start_row * num_head;

        if (cache.getDataType() == ml::train::TensorDim::DataType::FP32) {
          float *cache_addr = cache.getData<float>();
          compute_kcaches_fp32_reference(
            input_addr, cache_addr, output_addr, row_to_compute,
            num_head / group_size, head_dim, group_size, local_window_size);
        } else {
          uint16_t *cache_addr = cache.getData<uint16_t>();
          nntrainer::compute_kcaches<uint16_t>(
            input_addr, cache_addr, output_addr, row_to_compute,
            num_head / group_size, head_dim, group_size, tile_size,
            local_window_size);
        }
      });
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (sequence_len == 1) {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int num_rows = is_causal ? from + 1 : from + sequence_len;
      unsigned int num_cache_head = num_head / group_size;

      // Use ThreadManager for lower overhead parallelization during decoding
      const _FP16 *in_data = in.getData<_FP16>();
      const _FP16 *cache_data = cache.getData<_FP16>();
      _FP16 *out_data = out.getData<_FP16>();

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(
        0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
          nntrainer::compute_kcaches(
            in_data, cache_data, out_data, num_rows, num_cache_head, head_dim,
            group_size, tile_size, local_window_size, head_kv, head_kv + 1);
        });
    } else {
      // Iterate over ALL query rows so that no row is skipped even when
      // sequence_len > local_window_size.
      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(sequence_len), [=](size_t i) {
        _FP16 *input_addr = in.getData<_FP16>() + num_head * head_dim * i;
        _FP16 *cache_addr = cache.getData<_FP16>();
        int row_to_compute = is_causal ? from + i + 1 : from + sequence_len;
        // Windowed cumulative offset so that each row's scores are placed
        // contiguously after the previous row's scores (respecting the window).
        size_t out_start_row = is_causal ? calc_windowed_attn_index(from + i) -
                                             calc_windowed_attn_index(from)
                                         : i * (from + sequence_len);

        _FP16 *output_addr = out.getData<_FP16>() + out_start_row * num_head;

        nntrainer::compute_kcaches(input_addr, cache_addr, output_addr,
                                   row_to_compute, num_head / group_size,
                                   head_dim, group_size, tile_size,
                                   local_window_size);
      });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

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

  // Load Input Tensors of this batch : b_ denotes a Tensor for this batch
  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + cache_index * cache_key_dim.width(),
    true);
  nntrainer::Tensor b_cache_value_step =
    cache_value.getSharedDataTensor(cache_value_step_dim,
                                    batch * cache_value_dim.getFeatureLen() +
                                      cache_index * cache_value_dim.width(),
                                    true);

  // step_size/is_prefill computed here (rather than after, where it used to
  // be computed) because the DSP RoPE dispatch below needs it: decode
  // (step_size==1) must never hit the bridge - a single-row RoPE is a GEMV-
  // scale op where the FastRPC round trip costs far more than the CPU
  // compute it replaces, exactly like every other NPU dispatch gate in this
  // codebase (should_use_flash_attn, should_use_fused_ffn).
  unsigned int step_size = to - from;
  bool is_prefill = !from || step_size > 1;

  // append kcache with or without rotary embedding
  // For prefill with flash_attn: try DSP RoPE on K, then copy already-
  // rotated K into cache. F32 key_step rotates in-place directly; FP16
  // key_step (this model's actual dtype) goes through the cast-rotate-cast
  // chain in try_dsp_fp16_rope (the F32-only DSP RoPE kernel by way of a
  // scratch buffer) since there's no FP16 RoPE kernel on the DSP.
  bool k_rope_done = false;
  if (is_prefill && use_rope && is_cdsp_engine &&
      !getenv("NNTR_HEXAGON_NO_ELEM_OPS") &&
      (key_step.getDataType() == ml::train::TensorDim::DataType::FP32 ||
       key_step.getDataType() == ml::train::TensorDim::DataType::FP16)) {

    unsigned int n_tokens = key_step.height();
    unsigned int n_heads = num_heads_KV;
    std::vector<int32_t> positions(n_tokens);
    for (unsigned int i = 0; i < n_tokens; i++) {
      positions[i] = (int32_t)(cache_index + i);
    }

    if (key_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const rope_fn &rope_dsp = get_rope_bridge();
      if (rope_dsp) {
        int rc = rope_dsp(key_step.getData<float>(), positions.data(),
                          n_tokens, n_heads, (unsigned int)head_dim,
                          theta, 2);
        if (rc == 0) {
          // K is now actually rotated (rotation was just enqueued, not yet
          // run) - copy it into the cache without re-rotating. Try the DSP
          // copy bridge first: it enqueues right after the pending RoPE op
          // with no host-side flush at all. Only fall back to
          // flush_if_batch_active()+CPU copy (which forces the RoPE result
          // to materialize on the host first) if the DSP copy path isn't
          // available for this tensor (see try_dsp_cache_copy).
          if (!try_dsp_cache_copy(is_cdsp_engine, key_step, b_cache_key_step)) {
            flush_if_batch_active();
            apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim,
                                       cache_index, true);
          }
          k_rope_done = true;
        } else {
          ml_logw("MHACore: K RoPE DSP bridge failed (rc=%d), falling back to CPU", rc);
        }
      }
    } else {
      // FP16: cast-rotate-cast straight into the cache - no separate copy
      // step needed, try_dsp_fp16_rope's last cast writes directly to dst.
      if (try_dsp_fp16_rope(is_cdsp_engine, key_step, b_cache_key_step,
                            positions.data(), n_tokens, n_heads,
                            (unsigned int)head_dim, theta)) {
        k_rope_done = true;
      }
    }
  }
  if (!k_rope_done) {
    // key_step may be the output of an enqueued-but-not-yet-executed K
    // projection GEMM (dispatched via HexagonComputeOps from the FC layer
    // above, inside the same open batch) - sync before reading it here too.
    flush_if_batch_active();
    apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim, cache_index,
                               !use_rope);
  }


  // append vcache without rotary embedding.
  // value_step is the V projection GEMM's output, which may itself be an
  // enqueued-but-not-yet-executed op if it went through HexagonComputeOps
  // inside the same open batch. Try the DSP copy bridge first - it chains
  // after whatever produced value_step with no host-side flush at all
  // (V never needs rotation, unlike K, so this doesn't depend on RoPE
  // dispatch succeeding); only the fallback CPU paths below need to force
  // it to materialize first.
  if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (!try_dsp_cache_copy(is_cdsp_engine, value_step, b_cache_value_step)) {
      flush_if_batch_active();
      apply_rotary_emb_tensor_v2(value_step, b_cache_value_step, head_dim,
                                 cache_index, true);
    }
  } else if (query_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (!try_dsp_cache_copy(is_cdsp_engine, value_step, b_cache_value_step)) {
      flush_if_batch_active();
      b_cache_value_step.copyData(value_step);
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }

  if (skip_prefill && is_prefill)
    return;

  // apply rotary embedding for query
  if (use_rope) {
    // Try DSP RoPE bridge for prefill (in-place rotation on cDSP). F32
    // rotates directly; FP16 (this model's actual dtype) goes through the
    // cast-rotate-cast chain in try_dsp_fp16_rope - see the K-RoPE dispatch
    // above for the full explanation. Prefill-only - see the is_prefill
    // comment above the K-RoPE dispatch.
    bool rope_done = false;
    if (is_prefill && is_cdsp_engine &&
        !getenv("NNTR_HEXAGON_NO_ELEM_OPS") &&
        (query_step.getDataType() == ml::train::TensorDim::DataType::FP32 ||
         query_step.getDataType() == ml::train::TensorDim::DataType::FP16)) {
      unsigned int n_tokens = query_step.height();
      unsigned int n_heads = num_heads_Q;

      // Build position indices [cache_index, cache_index+1, ..., cache_index+n_tokens-1]
      std::vector<int32_t> positions(n_tokens);
      for (unsigned int i = 0; i < n_tokens; i++) {
        positions[i] = (int32_t)(cache_index + i);
      }

      if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
        const rope_fn &rope_dsp = get_rope_bridge();
        if (rope_dsp) {
          // mode=2 is NEOX (split-half) — matches ggml/llama.cpp Qwen3 RoPE
          int rc = rope_dsp(query_step.getData<float>(), positions.data(),
                            n_tokens, n_heads, (unsigned int)head_dim,
                            theta, 2);
          if (rc == 0) {
            rope_done = true;
          } else {
            ml_logw("MHACore: RoPE DSP bridge failed (rc=%d), falling back to CPU", rc);
          }
        }
      } else {
        // FP16, in-place: src and dst are the same tensor - safe, since the
        // cast-in (reads query_step) always executes before the cast-out
        // (writes query_step) in DSP FIFO order, with the rotation done on
        // the scratch buffer in between.
        if (try_dsp_fp16_rope(is_cdsp_engine, query_step, query_step,
                              positions.data(), n_tokens, n_heads,
                              (unsigned int)head_dim, theta)) {
          rope_done = true;
        }
      }
    }
    if (!rope_done) {
      // Same hazard as the K fallback above: query_step may be an
      // enqueued-but-not-yet-executed Q projection GEMM's output.
      flush_if_batch_active();
      apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, cache_index,
                                 false);
    }
  }


  /// @todo replace step_size into input height
  unsigned int cache_from = cache_index;
  unsigned int cache_to = cache_from + step_size;

  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  cached_key_dim.height(cache_to);
  cached_value_dim.height(cache_to);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  // out_ stores the output of Q * K
  nntrainer::Tensor out_(1, 1,
                         is_causal ? (calc_windowed_attn_index(cache_to) -
                                      calc_windowed_attn_index(cache_from))
                                   : (step_size * cache_to),
                         num_heads_Q, query_step.getTensorType());

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  // Try flash attention dispatch (DSP offload) for prefill
  bool use_flash = should_use_flash_attn(step_size, head_dim, is_prefill);

  if (use_flash) {
    // §6.3: Use cached causal mask instead of rebuilding per-layer.
    // The mask for a given (step_size, cache_to, cache_from) is identical
    // across all 28 transformer blocks during prefill, so build it once
    // and reuse. Falls back to build_causal_mask on cache miss.
    const std::vector<uint16_t> & mask =
      get_cached_causal_mask(step_size, cache_to, cache_from);


    // Determine Q/out dtype - FP16 is the common case for CausalLM
    bool q_is_fp16 = (query_step.getDataType() ==
                      ml::train::TensorDim::DataType::FP16);
    bool out_is_fp16 = (attention_output_step.getDataType() ==
                        ml::train::TensorDim::DataType::FP16);

    const flash_attn_fn &fn = get_flash_attn_bridge();
    int rc = fn(query_step.getData(), b_cached_key.getData(),
                b_cached_value.getData(), mask.data(),
                attention_output_step.getData(), step_size, num_heads_Q,
                num_heads_KV, head_dim, cache_to,
                1.0f / sqrtf((float)head_dim), q_is_fp16, out_is_fp16);

    if (rc != 0) {
      ml_logw("MHACore: flash_attn returned %d, falling back to CPU", rc);
      use_flash = false;
    }
  }

  if (!use_flash) {
    // query_step/b_cached_key can be enqueued-but-not-yet-executed DSP
    // output here: RoPE-DSP is gated on is_prefill alone (see above), while
    // flash_attn additionally requires step_size >= 160 - so for
    // 2 <= step_size < 160, RoPE may have been dispatched to the DSP
    // (leaving query_step's rotation pending) while attention still falls
    // through to this CPU path. Sync before it reads anything.
    flush_if_batch_active();
    // Original CPU path
    compute_kcaches(query_step, b_cached_key, out_, cache_from,
                    cache_to - cache_from, num_heads_Q, gqa_size, head_dim);

    softmax_triangle(out_, step_size, num_heads_Q, cache_from);

    compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                  cache_from, num_heads_KV, gqa_size, head_dim,
                                  cache_to);
  }
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

  /** 1. Load Input Tensors of this batch : b_ denotes a Tensor for this batch
   * **/
  nntrainer::Tensor b_cache_key_step = cache_key.getSharedDataTensor(
    cache_key_step_dim,
    batch * cache_key_dim.getFeatureLen() + from * cache_key_dim.width(), true);
  nntrainer::Tensor b_cache_value_step = cache_value.getSharedDataTensor(
    cache_value_step_dim,
    batch * cache_value_dim.getFeatureLen() + from * cache_value_dim.width(),
    true);

  if (use_rope) {
    apply_rotary_emb_tensor_v2(query_step, query_step, head_dim, _from, false);
  }

  apply_rotary_emb_tensor_v2(key_step, b_cache_key_step, head_dim, _from,
                             !use_rope);

  if (query_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    apply_rotary_emb_tensor_v2(value_step, b_cache_value_step, head_dim, _from,
                               true);
  } else if (query_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (!try_dsp_cache_copy(is_cdsp_engine, value_step, b_cache_value_step)) {
      flush_if_batch_active();
      b_cache_value_step.copyData(value_step);
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }


  ml::train::TensorDim cached_key_dim = cache_key_dim;
  ml::train::TensorDim cached_value_dim = cache_value_dim;
  cached_key_dim.height(to);
  cached_value_dim.height(to);

  nntrainer::Tensor b_cached_key = cache_key.getSharedDataTensor(
    cached_key_dim, batch * cache_key_dim.getFeatureLen(), true);
  nntrainer::Tensor b_cached_value = cache_value.getSharedDataTensor(
    cached_value_dim, batch * cache_value_dim.getFeatureLen(), true);

  nntrainer::Tensor out_(1, 1,
                         is_causal ? (((to - from) == 1)
                                        ? to
                                        : calc_windowed_attn_index(to) -
                                            calc_windowed_attn_index(from))
                                   : ((to - from) * to),
                         num_heads_Q, query_step.getTensorType());

  unsigned int gqa_size = num_heads_Q / num_heads_KV;

  compute_kcaches(query_step, b_cached_key, out_, _from, to - from, num_heads_Q,
                  gqa_size, head_dim);

  softmax_triangle(out_, to - from, num_heads_Q, from, sink_step);

  compute_fp16vcache_transposed(out_, b_cached_value, attention_output_step,
                                from, num_heads_KV, gqa_size, head_dim, to);
}

/************************************************************** */

/**
 * @brief rotary embedding-related member function
 * @note seq_len -> max_position_embeddings
 */
void MHACoreLayer::precompute_freqs(int head_dim, unsigned int seq_len,
                                    float theta, bool is_fp16) {
  const std::string rope_cache_key = getRopeCacheKey(head_dim, seq_len, theta);
  thetas.clear();
  if (rope_scaling_type == "default")
    _compute_default_parameters(head_dim, theta);
  else if (rope_scaling_type == "yarn")
    _compute_yarn_parameters(head_dim, theta);
  else if (rope_scaling_type == "proportional")
    _compute_proportional_parameters(head_dim, theta);
  else
    NNTR_THROW_IF(true, std::invalid_argument) << "Unsupported rope type!";

  unsigned int half_ = head_dim / 2;

  if (!is_fp16) {
    auto it = rope_cache_fp32.find(rope_cache_key);
    if (it != rope_cache_fp32.end()) {
      freqs_fp32 = it->second;
      return;
    }

    auto cached = std::make_shared<RopeCacheFP32>();
    cached->cos.assign(seq_len, std::vector<float>(head_dim, 0));
    cached->sin.assign(seq_len, std::vector<float>(head_dim, 0));

    for (unsigned int i = 0; i < seq_len; ++i) {
      nntrainer::calc_trigonometric_vals_dup(
        half_, thetas.data(), cached->cos[i].data(), cached->sin[i].data(), i,
        attention_scaling);
    }
    rope_cache_fp32[rope_cache_key] = cached;
    freqs_fp32 = cached;
  }

#ifdef ENABLE_FP16
  if (is_fp16) {
    auto it = rope_cache_fp16.find(rope_cache_key);
    if (it != rope_cache_fp16.end()) {
      freqs_fp16 = it->second;
      return;
    }

    auto cached = std::make_shared<RopeCacheFP16>();
    cached->cos.assign(seq_len, std::vector<_FP16>(head_dim, 0));
    cached->sin.assign(seq_len, std::vector<_FP16>(head_dim, 0));

    std::vector<float> cos_tmp(head_dim);
    std::vector<float> sin_tmp(head_dim);

    for (unsigned int i = 0; i < seq_len; ++i) {
      nntrainer::calc_trigonometric_vals_dup(half_, thetas.data(),
                                             cos_tmp.data(), sin_tmp.data(), i,
                                             attention_scaling);
      for (unsigned int j = 0; j < head_dim; ++j) {
        cached->cos[i][j] = (_FP16)cos_tmp[j];
        cached->sin[i][j] = (_FP16)sin_tmp[j];
      }
    }
    rope_cache_fp16[rope_cache_key] = cached;
    freqs_fp16 = cached;
  }
#endif
}

std::string MHACoreLayer::getRopeCacheKey(int head_dim, unsigned int seq_len,
                                          float theta) const {
  std::ostringstream ss;
  ss << rope_scaling_type << "|" << head_dim << "|" << seq_len << "|" << theta
     << "|" << scale << "|" << rope_partial_rotary_factor << "|"
     << original_max_position_embeddings;
  return ss.str();
}

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
  attention_scaling = 1.0f;
  const int half_dim = static_cast<int>(head_dim / 2);
  const int rope_angles =
    static_cast<int>((rope_partial_rotary_factor * head_dim) / 2.0f);

  thetas.reserve(half_dim);
  for (int i = 0; i < rope_angles; ++i) {
    thetas.push_back(1.0f /
                     (std::pow(theta, (2 * i) / static_cast<float>(head_dim))));
  }

  for (int i = rope_angles; i < half_dim; ++i) {
    thetas.push_back(0.0f);
  }

  for (auto &val : thetas) {
    val /= scale;
  }
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
                                              bool convert_only) {
  if (!use_rope) {
    if (&in != &out) {
      out.copyData(in);
    }
    return;
  }
  unsigned int half_ = dim / 2;
  unsigned int max_timestep =
    std::get<nntrainer::props::MaxTimestep>(mha_core_props).get();

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if (freqs_fp32 == nullptr) {
      const std::lock_guard<std::mutex> lock(rope_init_mtx);
      if (freqs_fp32 == nullptr) {
        precompute_freqs(head_dim, max_position_embeddings, theta, false);
      }
    }
    std::vector<float> *cos_ = nullptr;
    std::vector<float> *sin_ = nullptr;

    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &freqs_fp32->cos[from + h];
            sin_ = &freqs_fp32->sin[from + h];
          }
          float *in_ptr = in.getData<float>() +
                          b * in.channel() * in.height() * in.width() +
                          c * in.height() * in.width() + h * in.width();

          if (out.getDataType() == ml::train::TensorDim::DataType::FP32) {
            float *out_ptr = out.getData<float>() +
                             b * out.channel() * out.height() * out.width() +
                             c * out.height() * out.width() + h * out.width();

            if (out_ptr != in_ptr) {
              std::memcpy(out_ptr, in_ptr, sizeof(float) * in.width());
            }
            if (!convert_only) {
              nntrainer::compute_rotary_emb_value(
                in.width(), dim, half_, out_ptr, nullptr, cos_->data(),
                sin_->data(), false);
            }
          } else if (out.getDataType() ==
                       ml::train::TensorDim::DataType::UINT16 ||
                     out.getDataType() ==
                       ml::train::TensorDim::DataType::FP16) {
            uint16_t *out_ptr = out.getData<uint16_t>() +
                                b * out.channel() * out.height() * out.width() +
                                c * out.height() * out.width() +
                                h * out.width();

            nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                                out_ptr, cos_->data(),
                                                sin_->data(), convert_only);
          }
        }
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if (freqs_fp16 == nullptr) {
      const std::lock_guard<std::mutex> lock(rope_init_mtx);
      if (freqs_fp16 == nullptr) {
        precompute_freqs(head_dim, max_position_embeddings, theta, true);
      }
    }
    std::vector<_FP16> *cos_ = nullptr;
    std::vector<_FP16> *sin_ = nullptr;

    for (unsigned int b = 0; b < in.batch(); b++) {
      for (unsigned int c = 0; c < in.channel(); c++) {
        for (unsigned int h = 0; h < in.height(); h++) {
          if (from < max_timestep) {
            cos_ = &freqs_fp16->cos[from + h];
            sin_ = &freqs_fp16->sin[from + h];
          }
          _FP16 *in_ptr = in.getData<_FP16>() +
                          b * in.channel() * in.height() * in.width() +
                          c * in.height() * in.width() + h * in.width();
          _FP16 *out_ptr = out.getData<_FP16>() +
                           b * out.channel() * out.height() * out.width() +
                           c * out.height() * out.width() + h * out.width();

          nntrainer::compute_rotary_emb_value(in.width(), dim, half_, in_ptr,
                                              out_ptr, cos_->data(),
                                              sin_->data());
        }
      }
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
                                    size_t num_head, unsigned int from) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] =
          std::tanh(qk_out_[i] * inv_softcapping) * attn_logit_softcapping;
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      // Iterate over ALL rows (not just min(row, window)) so that every query
      // row in a long prefill gets softmaxed over the correct windowed range.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head);
      });
    }
  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] = (_FP16)(std::tanh((float)qk_out_[i] * inv_softcapping) *
                             attn_logit_softcapping);
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
    } else {
      // Iterate over ALL rows (not just min(row, window)) so that every query
      // row in a long prefill gets softmaxed over the correct windowed range.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head);
      });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::softmax_triangle(nntrainer::Tensor &qk_out, size_t row,
                                    size_t num_head, unsigned int from,
                                    nntrainer::Tensor &sink_step) {
  if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *qk_out_ = qk_out.getData<float>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] =
          std::tanh(qk_out_[i] * inv_softcapping) * attn_logit_softcapping;
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        unsigned int to = from + row;
        end_row = to;
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head,
                                     sink_step.getData());
    } else {
      // Iterate over ALL rows (not just min(row, window)) for correct windowed
      // prefill when sequence_len > local_window_size.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(i + from) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head,
                               sink_step.getData());
      });
    }
  } else if (qk_out.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *qk_out_ = qk_out.getData<_FP16>();
    _FP16 *sink_step_ = sink_step.getData<_FP16>();

    if (attn_logit_softcapping > 0.0f) {
      size_t len =
        qk_out.batch() * qk_out.height() * qk_out.width() * qk_out.channel();
      float inv_softcapping = 1.0f / attn_logit_softcapping;
      for (size_t i = 0; i < len; ++i) {
        qk_out_[i] = (_FP16)(std::tanh((float)qk_out_[i] * inv_softcapping) *
                             attn_logit_softcapping);
      }
    }

    if (row == 1) {
      size_t start_row = 0;
      size_t end_row = 0;
      if (is_causal) {
        end_row = from < local_window_size ? from + 1 : local_window_size;
      } else {
        end_row = from + row; // end_row = to
      }
      nntrainer::softmax_row_inplace(qk_out_, start_row, end_row, num_head,
                                     sink_step_);
    } else {
      // Iterate over ALL rows (not just min(row, window)) for correct windowed
      // prefill when sequence_len > local_window_size.
      size_t total_rows = row;
      if (!is_causal)
        total_rows = row;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, total_rows, [=](size_t i) {
        size_t start_row, end_row;
        if (is_causal) {
          start_row =
            calc_windowed_attn_index(i + from) - calc_windowed_attn_index(from);
          end_row = calc_windowed_attn_index(from + i + 1) -
                    calc_windowed_attn_index(from);
        } else {
          unsigned int to = from + row;
          start_row = i * to;
          end_row = (i + 1) * to;
        }
        nntrainer::softmax_row(qk_out_, start_row, end_row, num_head,
                               sink_step_);
      });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
}

void MHACoreLayer::compute_fp16vcache_transposed(
  nntrainer::Tensor &in, nntrainer::Tensor &vcache, nntrainer::Tensor &output,
  int from, int num_cache_head, int gqa_size, int head_dim, int to) {

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    if ((to - from) != 1) {
      // Iterate over ALL output rows so every query row gets an output even
      // when (to - from) > local_window_size.
      int total = to - from;
      if (!is_causal)
        total = to - from;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(total), [=](size_t i) {
        size_t start_idx;
        if (is_causal) {
          start_idx =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
        } else {
          start_idx = i * to; // linear index
        }
        const float *input =
          in.getData<float>() + start_idx * num_cache_head * gqa_size;
        float *out =
          output.getData<float>() + i * (num_cache_head * gqa_size * head_dim);

        int row_num = is_causal ? (from + (int)i) : to - 1;
        if (vcache.getDataType() == ml::train::TensorDim::DataType::FP32) {
          compute_vcache_fp32_transposed_reference(
            row_num, input, vcache.getData<float>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        } else {
          nntrainer::compute_fp16vcache_fp32_transposed(
            row_num, input, vcache.getData<uint16_t>(), out, num_cache_head,
            gqa_size, head_dim, local_window_size);
        }
      });
    } else {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_num = to - 1;

      // Use OpenMP for lower overhead parallelization during decoding
      const float *in_data = in.getData<float>();
      float *output_data = output.getData<float>();

      auto &tm = nntrainer::ThreadManager::Global();
      if (vcache.getDataType() == ml::train::TensorDim::DataType::FP32) {
        const float *vcache_data = vcache.getData<float>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            compute_vcache_fp32_transposed_reference(
              row_num, in_data, vcache_data, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
          });
      } else {
        const uint16_t *vcache_data = vcache.getData<uint16_t>();
        tm.parallel_for(
          0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
            nntrainer::compute_fp16vcache_fp32_transposed(
              row_num, in_data, vcache_data, output_data, num_cache_head,
              gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
          });
      }
    }
  } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    if ((to - from) != 1) {
      // Iterate over ALL output rows so every query row gets an output even
      // when (to - from) > local_window_size.
      int total = to - from;
      if (!is_causal)
        total = to - from;

      auto &tm = nntrainer::ThreadManager::Global();
      tm.parallel_for(0, static_cast<size_t>(total), [=](size_t i) {
        size_t start_idx;
        if (is_causal) {
          start_idx =
            calc_windowed_attn_index(from + i) - calc_windowed_attn_index(from);
        } else {
          start_idx = i * to;
        }
        const _FP16 *input =
          in.getData<_FP16>() + start_idx * num_cache_head * gqa_size;
        _FP16 *out =
          output.getData<_FP16>() + i * (num_cache_head * gqa_size * head_dim);
        int row_num = is_causal ? (from + (int)i) : to - 1;
        nntrainer::compute_fp16vcache_transposed(
          row_num, input, vcache.getData<_FP16>(), out, num_cache_head,
          gqa_size, head_dim, local_window_size);
      });
    } else {
      // Single token processing (common during generation)
      // Parallelize over KV heads for decoding since Q direction is always 1
      int row_num = to - 1;

      // Use OpenMP for lower overhead parallelization during decoding
      const _FP16 *in_data = in.getData<_FP16>();
      const _FP16 *vcache_data = vcache.getData<_FP16>();
      _FP16 *output_data = output.getData<_FP16>();

      auto &tm_fp16 = nntrainer::ThreadManager::Global();
      tm_fp16.parallel_for(
        0, static_cast<size_t>(num_cache_head), [=](size_t head_kv) {
          nntrainer::compute_fp16vcache_transposed(
            row_num, in_data, vcache_data, output_data, num_cache_head,
            gqa_size, head_dim, local_window_size, head_kv, head_kv + 1);
        });
    }
#else
    NNTR_THROW_IF(true, std::invalid_argument) << "enable-fp16 is not set!";
#endif
  }
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
}

size_t MHACoreLayer::calc_attn_index(size_t i) { return (i * (i + 1)) / 2; };

size_t MHACoreLayer::calc_windowed_attn_index(size_t i) {
  // S(i) = sum_{k=0}^{i-1} min(k+1, W)
  // For i <= W:  S(i) = i*(i+1)/2   (same as full-attention triangular index)
  // For i >  W:  S(i) = W*(W+1)/2 + (i - W)*W
  // When W == UINT_MAX, i <= W is always true, so we never evaluate
  // W*(W+1)/2 and there is no overflow.
  if (i <= local_window_size) {
    return (i * (i + 1)) / 2;
  } else {
    return (local_window_size * (local_window_size + 1)) / 2 +
           (i - local_window_size) * local_window_size;
  }
};

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

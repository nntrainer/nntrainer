// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   compute_ops.h
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  ComputeOps abstract interface for backend-agnostic dispatch
 *
 * Each Context (CPU/GPU/NPU) provides a concrete ComputeOps subclass.
 * Tensor operations call through this interface, enabling runtime
 * dispatch to the correct backend (ARM NEON, x86 AVX, OpenCL, CUDA,
 * QNN/HMX, ...) without #ifdef and — crucially — letting backend
 * subclasses carry their own state (cl_command_queue, npu_session,
 * kernel cache, ...) as member variables. That is the difference
 * between this and a function-pointer table: virtual dispatch lets
 * the impl reach back into per-backend resources without leaking a
 * `this` pointer through every call.
 *
 * Default method bodies throw std::runtime_error("not implemented").
 * Concrete subclasses override every op they want to support. For
 * accelerator-only ops (GPU batch/accel variants), pair the op with
 * a supports_*() predicate so callers can pick a CPU path on backends
 * that don't have an accelerated impl.
 */

#ifndef __COMPUTE_OPS_H__
#define __COMPUTE_OPS_H__
#ifdef __cplusplus

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef ENABLE_FP16
#include <tensor_dim.h>
#endif

namespace nntrainer {

class Tensor;

/**
 * @brief The router of a MoE layer, for a backend that routes a decode row on
 *        the device (see MoeExpertGroup::router).
 * @details The routing it computes: rms_norm(row, epsilon), times norm (when
 * given), times scale * hidden^-1/2, the projection by weight, softmax over
 * the experts, the topk largest (ties: lowest expert id), renormalised over
 * the topk, times per_expert_scale of each selected expert.
 */
struct MoeRouter {
  const Tensor *input = nullptr;            /**< [1, H] router row */
  const Tensor *weight = nullptr;           /**< [H, E] fp32 projection */
  const Tensor *scale = nullptr;            /**< [H] fp32 router scale */
  const Tensor *norm = nullptr;             /**< [H] fp32, or null */
  const Tensor *per_expert_scale = nullptr; /**< [E] fp32 */
  float epsilon = 1.0e-6f;                  /**< rms_norm epsilon */
  unsigned int topk = 0;                    /**< experts per token */
};

/**
 * @brief Where a MoE layer's experts come from when the backend reads them
 *        from the weight file into its own expert slots (see
 *        MoeExpertGroup::fsu).
 * @details One expert's record is its gate, up and down weights back to back
 * in the file (record_bytes in all). The layer owns the residency policy: it
 * decides which expert sits in which of its @a slots slots; the backend owns
 * the bytes (the slot memory, the reads, and the fences that keep a slot from
 * being overwritten while a queued kernel still reads it).
 */
struct MoeExpertFsu {
  int fd = -1; /**< the weight file, kept open by the caller */
  /** per expert id: file offset of its record */
  const std::vector<uint64_t> *offset = nullptr;
  size_t record_bytes = 0; /**< bytes of one record (the record stride) */
  unsigned int slots = 0;  /**< this layer's slots */
  /** per expert id: its slot, or UINT_MAX when not resident; read after
   *  each acquire below */
  const std::vector<unsigned int> *slot_of = nullptr;
  /** (expert, slot) pairs to read before use, for a caller that acquired
   *  them up front (no acquire hook) */
  const std::vector<std::pair<unsigned int, unsigned int>> *fill = nullptr;
  /** make @a experts resident (the backend calls it with at most @a slots
   *  experts at a time, before it uses them) and append the (expert, slot)
   *  pairs that must be read first to @a fill; may be empty when @a fill
   *  above already covers the call */
  std::function<void(const std::vector<unsigned int> &experts,
                     std::vector<std::pair<unsigned int, unsigned int>> &fill)>
    acquire;
  /** called once the (expert, slot) pairs @a fill that the acquire hook
   *  returned are ordered ahead of every later reader and writer of their
   *  slots (@a ok), or could not be read (!@a ok: those slots do not hold
   *  the experts); the policy holds those slots from its acquire until
   *  then, so no other acquire takes them while their reads are in flight */
  std::function<void(
    const std::vector<std::pair<unsigned int, unsigned int>> &fill, bool ok)>
    filled;
  /** every expert is resident and stays in its slot (slot_of no longer
   *  changes): the backend may route a decode row on the device */
  bool complete = false;
  /** a decode row routed on the device: the experts ranked after the
   *  selected ones, with them, least to most recent (the top-1 last); a
   *  hint for the policy (it loads and evicts nothing) */
  std::function<void(const std::vector<unsigned int> &recency)> refresh;

  // ---- prefill: whole-layer block stream ----
  /** position of the layer among the streamed MoE layers (0, 1, ...; every
   *  layer's partition is created at load under this index), or UINT_MAX */
  unsigned int layer_index = UINT32_MAX;
  /** the next streamed MoE layer's descriptor (fd, offset, record_bytes,
   *  slots, layer_index), or null for the last layer */
  const MoeExpertFsu *next = nullptr;
  /** a call of at least this many rows on a layer with fewer slots than
   *  experts streams whole layers: its experts run from a block read
   *  ahead (during the previous layer) into partitions lent by the arena,
   *  and it reads the next layer's block ahead the same way. The layer has
   *  forgotten its resident experts before such a call (its slots may be
   *  lent). Fewer rows: only the routed experts are read, most rows first */
  unsigned int spec_min_tokens = 64;
  /** drop the page cache of a streamed block's file range once it is
   *  consumed (posix_fadvise DONTNEED) */
  bool fadvise_dontneed = true;

  // ---- short prefill: next-layer read-ahead ----
  /** a call of fewer than spec_min_tokens rows (not streamed): the experts
   *  the next layer (next) is predicted to need -- its router run on this
   *  call's router rows, every expert in some row's top-k -- most
   *  predicted rows first, or null. Once this call's experts are enqueued
   *  the backend may read the ones the next layer does not hold into that
   *  layer's own slots (next->reserve; next->release with admit once
   *  written), so the reads run while this layer computes and the next
   *  layer's acquire finds them resident. A hint only: no routing and no
   *  value changes; a wrong prediction is a record evicted later
   *  (NNTR_G4MOE_FSU_PREFILL_LOOKAHEAD=0: never read ahead) */
  const std::vector<unsigned int> *next_ahead = nullptr;

  // ---- decode: next-layer prefetch (pre-gating) ----
  /** the next streamed layer's router (weight, scale, norm,
   *  per_expert_scale, epsilon; input and topk unused), or null. A backend
   *  may run it on this layer's router row to predict the next layer's
   *  experts and read the ones not resident ahead into a host record cache
   *  (NNTR_G4MOE_FSU_PREGATE); the next layer's misses found there are then
   *  copied, not read. Changes no routing and no residency. */
  const MoeRouter *next_router = nullptr;
  /** the layer's residency hooks, set on the descriptor the previous layer
   *  names as next (a short prefill's read ahead, next_ahead): reserve a
   *  slot for @a expert ahead of the layer's next call (a free slot or a victim
   *  outside @a avoid evicted now; the slot is held for it and the
   *  reservation counts no access), or UINT_MAX; release(expert, admit)
   *  ends an unused reservation (admit: the written record stays as a
   *  resident expert, no access counted); resident(expert). The layer's
   *  next acquire places a reserved expert in its slot. Called from reader
   *  threads. resident may also be set on the call's own descriptor: a
   *  short prefill's chunks then take the resident experts first. */
  std::function<unsigned int(unsigned int expert,
                             const std::vector<unsigned int> &avoid)>
    reserve;
  std::function<void(unsigned int expert, bool admit)> release;
  std::function<bool(unsigned int expert)> resident;

  // ---- decode without the host round trip (NNTR_G4MOE_FSU_FLAGS) ----
  /** the device places a decode step's misses itself, from a victim queue
   *  the backend publishes ahead of the step, and the backend replays what
   *  it did into the policy afterwards (off the step's path). Set on the
   *  call's descriptor and on the one the previous layer names as next.
   *  victim_order: the slots in the order successive acquires would take
   *  them for misses, without @a exclude (ExpertResidency::victimOrder);
   *  place: a step of @a need whose misses went to @a placed (acquire()'s
   *  history and recency); admit: a read ahead put @a expert into @a slot
   *  (no access); forget: a read ahead dropped before it read. Host thread
   *  only; each leaves slot_of current. */
  std::function<std::vector<unsigned int>(
    const std::vector<unsigned int> &exclude)>
    victim_order;
  std::function<void(
    const std::vector<unsigned int> &need,
    const std::vector<std::pair<unsigned int, unsigned int>> &placed)>
    place;
  std::function<void(unsigned int expert, unsigned int slot)> admit;
  std::function<void(unsigned int expert)> forget;
  /** optional: count a decode step the device placed that was never
   *  replayed (the last token's, at exit: a step is replayed at its layer's
   *  next prepare) into the model's totals: @a hits experts were resident,
   *  @a misses were read. Changes no policy; may run at exit, after the
   *  layer itself is gone, so it must not reach the layer */
  std::function<void(unsigned int hits, unsigned int misses)> tally;
};

/**
 * @brief All selected experts of one MoE forward call, for a backend that
 *        runs them together (see ComputeOps::moe_experts).
 */
struct MoeExpertGroup {
  const void *layer_key = nullptr;  /**< identity of the calling layer, for
                                         state a backend keeps per layer */
  const Tensor *input = nullptr;    /**< [T, H] expert input rows */
  Tensor *output = nullptr;         /**< [T, H] written (not accumulated) */
  unsigned int tokens = 0;          /**< T */
  std::vector<Tensor *> gate;       /**< every expert's gate weight, by id */
  std::vector<Tensor *> up;         /**< every expert's up weight, by id */
  std::vector<Tensor *> down;       /**< every expert's down weight, by id */
  std::vector<unsigned int> active; /**< selected experts, ascending id */
  /** per expert id: (token row, routing weight), ascending token */
  const std::vector<std::vector<std::pair<unsigned int, float>>> *assignments =
    nullptr;
  bool exact_gelu = false; /**< erf gelu gate instead of the tanh form */
  /** moe_experts_prebuild only: where each record's bytes are read from,
   *  [gate_0, up_0, down_0, gate_1, ...] (e.g. the weight file mapping);
   *  null reads the weight tensors themselves */
  const std::vector<const void *> *records = nullptr;
  /** a one-token call may leave active / assignments empty and give the
   *  router instead: the backend routes on the device (or declines) */
  const MoeRouter *router = nullptr;
  /** non-null: the experts are not resident in the weights (or are read
   *  past them); the backend reads them from the file into its slots */
  const MoeExpertFsu *fsu = nullptr;
  /** set by the backend (when non-null) to true when the kernels that write
   *  @a output were left deferred until its first reader asks for them
   *  (ComputeOps::flush_deferred); the caller then hands the output over
   *  as the backend left it and does not touch it */
  bool *output_deferred = nullptr;
};

/**
 * @class ComputeOps
 * @brief Abstract dispatch interface for tensor compute kernels.
 */
class ComputeOps {
public:
  virtual ~ComputeOps() = default;

  /** Release temporary prefill storage between completed graph traversals.
   * The caller keeps tensors alive until this returns; backend-only outputs
   * are materialized before their temporary storage is released. */
  virtual void release_prefill_scratch() {}

  // ===========================================================================
  // FP32 BLAS
  // ===========================================================================
  virtual void sgemm_fp32(const unsigned int TStorageOrder, bool TransA,
                          bool TransB, const unsigned int M,
                          const unsigned int N, const unsigned int K,
                          const float alpha, const float *A,
                          const unsigned int lda, const float *B,
                          const unsigned int ldb, const float beta, float *C,
                          const unsigned int ldc);

  virtual void sgemv_fp32(const unsigned int TStorageOrder, bool TransA,
                          const unsigned int M, const unsigned int N,
                          const float alpha, const float *A,
                          const unsigned int lda, const float *X,
                          const unsigned int incX, const float beta, float *Y,
                          const unsigned int incY);

  virtual float sdot_fp32(const unsigned int N, const float *X,
                          const unsigned int incX, const float *Y,
                          const unsigned int incY);

  virtual void saxpy_fp32(const unsigned int N, const float alpha,
                          const float *X, const unsigned int incX, float *Y,
                          const unsigned int incY);

  virtual void scopy_fp32(const unsigned int N, const float *X,
                          const unsigned int incX, float *Y,
                          const unsigned int incY);

  virtual void sscal_fp32(const unsigned int N, const float alpha, float *X,
                          const unsigned int incX);

  virtual float snrm2_fp32(const unsigned int N, const float *X,
                           const unsigned int incX);

  virtual unsigned int isamax_fp32(const unsigned int N, const float *X,
                                   const unsigned int incX);

  // ===========================================================================
  // FP32 Element-wise
  // ===========================================================================
  virtual void ele_mul_fp32(const unsigned int N, const float *X,
                            const float *Y, float *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_add_fp32(const unsigned int N, const float *X,
                            const float *Y, float *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_sub_fp32(const unsigned int N, const float *X,
                            const float *Y, float *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_div_fp32(const unsigned int N, const float *X,
                            const float *Y, float *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);

  // ===========================================================================
  // FP32 Activation / Special
  // ===========================================================================
  virtual void swiglu_fp32(const unsigned int N, float *X, float *Y, float *Z);
  virtual void swiglu_alpha_fp32(const unsigned int N, float *X, float *Y,
                                 float *Z, float alpha);
  virtual void tanh_gelu_fp32(const unsigned int N, const float *X, float *Y);
  virtual void gelu_v2_fp32(const unsigned int N, const float *X, float *Y);
  virtual void tanh_gelu_v2_fp32(const unsigned int N, const float *X,
                                 float *Y);
  virtual void tanh_gelu_mul_fp32(const unsigned int N, float *X, float *Y,
                                  float *Z);
  virtual void tanh_gelu_v2_mul_fp32(const unsigned int N, float *X, float *Y,
                                     float *Z);
  virtual float max_val_fp32(const unsigned int N, float *X);
  virtual void softmax_fp32(const unsigned int N, float *X, float *Y);
  virtual bool is_valid_fp32(const unsigned int N, const float *X);

  // ===========================================================================
  // FP32 Matrix ops
  // ===========================================================================
  virtual void transpose_matrix_fp32(const unsigned int M, const unsigned int N,
                                     const float *src, unsigned int ld_src,
                                     float *dst, unsigned int ld_dst);

  // ===========================================================================
  // FP32 Data conversion / Copy
  // ===========================================================================
  virtual void scopy_u8(const unsigned int N, const uint8_t *X,
                        const unsigned int incX, uint8_t *Y,
                        const unsigned int incY);
  virtual void scopy_s8(const unsigned int N, const int8_t *X,
                        const unsigned int incX, int8_t *Y,
                        const unsigned int incY);
  virtual void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                                     const unsigned int incX, float *Y,
                                     const unsigned int incY);
  virtual void copy_s16_fp32(const unsigned int N, const int16_t *X, float *Y);
  virtual void copy_u16_fp32(const unsigned int N, const uint16_t *X, float *Y);
  virtual void copy_fp32_u32(const unsigned int N, const float *X, uint32_t *Y);
  virtual void copy_fp32_u16(const unsigned int N, const float *X, uint16_t *Y);
  virtual void copy_fp32_u8(const unsigned int N, const float *X, uint8_t *Y);
  virtual void copy_fp32_s16(const unsigned int N, const float *X, int16_t *Y);
  virtual void copy_fp32_s8(const unsigned int N, const float *X, int8_t *Y);

  // ===========================================================================
  // Quantized GEMM (GGUF format)
  // ===========================================================================
  virtual void gemm_q4_0_fp32(const unsigned int M, const unsigned int N,
                              const unsigned int K, const float *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, float *C,
                              const unsigned int ldc);
  virtual void gemm_q4_K_fp32(const unsigned int M, const unsigned int N,
                              const unsigned int K, const float *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, float *C,
                              const unsigned int ldc);
  virtual void gemm_q6_K_fp32(const unsigned int M, const unsigned int N,
                              const unsigned int K, const float *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, float *C,
                              const unsigned int ldc);

  /**
   * @brief One activation against several q4_0 weights in a single call.
   *
   * Shares the online q8_0 quantization of the activation and a single
   * thread barrier across all weights, instead of repeating both per weight.
   * Only worthwhile for M > 1; callers must query the predicate first.
   *
   * @note @a matAdata holds the weights and @a matBdata the activation --
   * the reverse of gemm_q4_0_fp32's naming, kept for source compatibility
   * with the accelerator backends that introduced this entry point.
   */
  virtual bool supports_gemm_q4_0_batch_fp32() const { return false; }
  virtual void gemm_q4_0_batch_fp32(std::vector<void *> matAdata,
                                    float *matBdata,
                                    std::vector<float *> matCdata,
                                    unsigned int M, std::vector<unsigned int> N,
                                    unsigned int K);

  /**
   * @brief One fp32 activation against several TQ2CX (ternary) weights.
   *
   * Same argument order as gemm_q4_0_batch_fp32: @a matAdata holds the
   * TQ2CX records (N_i x K) and @a matBdata the M x K activation, used in
   * fp32 without quantization. Valid for every M. C_i = A * W_i^T is
   * written (no accumulation into C).
   */
  virtual bool supports_gemm_tq2cx_batch_fp32() const { return false; }
  virtual void
  gemm_tq2cx_batch_fp32(std::vector<void *> matAdata, float *matBdata,
                        std::vector<float *> matCdata, unsigned int M,
                        std::vector<unsigned int> N, unsigned int K);

  // ===========================================================================
  // Quantized weight packing / quantization
  // ===========================================================================
  virtual void unpack_q4_0(const void *in_q4_0x, void *out_q4_0,
                           size_t data_size, const unsigned int M,
                           const unsigned int N);
  virtual void unpack_q4_0x8_transpose16(const void *src, uint16_t *d_out,
                                         uint16_t *qs_out, int N, int K);
  virtual size_t quantize_q4_0(const float *src, void *dst, int64_t nrow,
                               int64_t n_per_row, const float *quant_weights);
  virtual void dequantize_row_q4_0(const void *x, float *y, int64_t k);
  virtual void repack_q4_0(void *dst, void *src, size_t data_size,
                           const unsigned int M, const unsigned int N);

  // ===========================================================================
  // Clamp
  // ===========================================================================
  virtual void clamp_fp32(const float *input, float *output, size_t length,
                          float lower_bound, float upper_bound);

  // ===========================================================================
  // Data conversion (int8 → FP32)
  // ===========================================================================
  virtual void scopy_int8_to_fp32_u(const unsigned int N, const uint8_t *X,
                                    const unsigned int incX, float *Y,
                                    const unsigned int incY);
  virtual void scopy_int8_to_fp32_s(const unsigned int N, const int8_t *X,
                                    const unsigned int incX, float *Y,
                                    const unsigned int incY);

  // ===========================================================================
  // Accelerator-only (GPU/NPU) ops — query supports_* before calling.
  // CPU subclasses leave both the impl (default-throw) and predicate
  // (default false) untouched. Accelerator subclasses override both.
  // ===========================================================================
  // gemm_q4_0_batch_fp32 with the RMSNorms a block runs around its
  // projections folded into the same accelerator call (doc 57 section 5
  // step 4), so the rows do not leave the accelerator for them: pre_gamma
  // (K floats, or nullptr) norms every activation row before the
  // quantizer; post_chunk[i] (one per weight; 0 = none) and post_gamma
  // (the chunks of every normed weight in call order, concatenated) norm
  // weight i's output rows per post_chunk[i]-wide piece, in place. eps
  // for both. A backend without it says so and the layer norms itself.
  // matAscale: empty for Q4_0 weights, else one QS4CX per-channel scale
  // region per weight (gemm_qs4cx_accel_fp32's form: quantized offline,
  // no Q4_0 detour). rope_cs / rope_hd / rope_weights: after the norms,
  // RoPE per head of rope_hd on the outputs of the first rope_weights
  // weights (q and k), rope_cs being M rows of 2*rope_hd floats -- the
  // CPU table's cos row then sin row for each row's position; rope_hd 0
  // for none.
  virtual bool supports_gemm_q4_0_batch_norm_fp32() const { return false; }
  virtual void gemm_q4_0_batch_norm_fp32(
    std::vector<void *> matAdata, std::vector<float *> matAscale,
    float *matBdata, std::vector<float *> matCdata, unsigned int M,
    std::vector<unsigned int> N, unsigned int K, const float *pre_gamma,
    const std::vector<unsigned int> &post_chunk, const float *post_gamma,
    float eps, const float *rope_cs = nullptr, unsigned int rope_hd = 0,
    unsigned int rope_weights = 0, bool out_to_host = true);

  // The decoder block's epilogue as one accelerator call (doc 57 section
  // 5 step 4): out = scale * (resid + rmsnorm(x [+ x2]) * gamma) over M
  // rows of N floats, the whole-row norm of the summed addend. x2 nullptr
  // for one addend, gamma nullptr for 1. A backend without it says so and
  // the layer runs the three ops itself.
  virtual bool supports_rmsnorm_add_fp32() const { return false; }
  virtual void rmsnorm_add_fp32(unsigned int M, unsigned int N,
                                const float *resid, const float *x,
                                const float *x2, const float *gamma, float eps,
                                float scale, float *out);

  // The MoE router's logits at prefill shapes (doc 57 section 5 step 5):
  // logits[M][E] = rmsnorm(x)[M][K] . w[K][E] in f32, gamma (K floats)
  // for the router's own norm or nullptr for none. E a multiple of 32 up
  // to 128. top_k > 0 (the softmax router) adds the selection: logits
  // then holds the softmax probabilities, sel the n_sel largest per row
  // (descending, an exact tie to the lower index, the layer's own rule)
  // and weight the first top_k's p * (1 / their sum) * scale[e].
  virtual bool supports_router_logits_fp32() const { return false; }
  virtual void
  router_logits_fp32(unsigned int M, unsigned int K, unsigned int E,
                     const float *x, const float *gamma, float eps,
                     const float *w, float *logits, unsigned int top_k = 0,
                     unsigned int n_sel = 0, const float *scale = nullptr,
                     unsigned int *sel = nullptr, float *weight = nullptr);

  // The lm_head of one row (doc 57 section 9.7): y (N floats) = the Q4_0
  // weight w (N rows of K, canonical block_q4_0 -- a tied embedding's own
  // bytes) times rmsnorm(x) * gamma (gamma nullptr: x as it is), then
  // softcap * tanh(y / softcap) (softcap 0: none). False when the backend
  // cannot run it (no such call, a shape it does not take, or a weight it
  // could not place); the caller then runs the same three on the CPU.
  virtual bool lm_head_q4_0_fp32(const void *w, unsigned int K, unsigned int N,
                                 const float *x, const float *gamma, float eps,
                                 float softcap, float *y) {
    (void)w, (void)K, (void)N, (void)x, (void)gamma, (void)eps, (void)softcap,
      (void)y;
    return false;
  }
  /** @brief Places the lm_head weight for lm_head_q4_0_fp32 ahead of the
   *  first call (at load, after every other weight: the arena order), so
   *  the first prefill does not pay for it. True when the calls will run
   *  here; false leaves the caller on its own path. */
  virtual bool lm_head_q4_0_prepare(const void *w, unsigned int K,
                                    unsigned int N) {
    (void)w, (void)K, (void)N;
    return false;
  }

  /** Share an untied LM head with the complete decode graph. The caller has
   * checked that no other CPU layer reads the source. The backend may release
   * its host pages once registered; both prefill and decode use its handles. */
  virtual void share_decode_lm_head(const void *w) { (void)w; }

  /** Run a previously shared head, or return false without reading its weight.
   */
  virtual bool lm_head_shared_fp32(const void *w, unsigned K, unsigned N,
                                   const float *x, float *y) {
    (void)w, (void)K, (void)N, (void)x, (void)y;
    return false;
  }

  virtual bool supports_gemm_q4_0_accel_fp32() const { return false; }
  virtual void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata,
                                    float *matCdata, unsigned int M,
                                    unsigned int N, unsigned int K);

  // The M > 1 gate FloatTensor::dot() applies around both calls above
  // exists because, for CPU/GPU, a single-row GEMM is already fast
  // without going through the batch/accel path -- that reasoning does
  // not hold for HTP, where M == 1 (decode) is the FastRPC-amortization
  // shape these kernels exist for (see docs/htp_attention/
  // 40_moe_ffn_htp_task.md section2.2, 41_moe_ffn_e2e_and_perf_task.md
  // sectionB3). This predicate lets a backend opt in at M == 1 without
  // touching CPU/GPU's existing M > 1 behavior, which is intentional and
  // must not change.
  virtual bool accelerates_q4_0_at_m1() const { return false; }
  // The same question for QS4CX weights, answered separately: a backend
  // that holds them in its own format may prefer every row, decode's one
  // included, over a CPU path in another format (doc 57 section 5).
  virtual bool accelerates_qs4cx_at_m1() const { return false; }

  // QS4CX weights reach an accelerator without the Q4_0 detour: a model
  // quantized straight from FP32 into QS4CX carries the same int4 values
  // HexKL's registry wants, so the conversion at the seam is a bit
  // rearrangement plus a colsum rather than a second quantization (see
  // htp_qs4cx_from_packed). matAdata / matAscale are QS4CX_Tensor's
  // getData()/getScale() regions.
  virtual bool supports_gemm_qs4cx_accel_fp32() const { return false; }
  virtual void gemm_qs4cx_accel_fp32(void *matAdata, float *matAscale,
                                     float *matBdata, float *matCdata,
                                     unsigned int M, unsigned int N,
                                     unsigned int K);

  // Same grouping as gemm_q4_0_batch_fp32 (several weights sharing one
  // activation, e.g. decode's top-K experts' gate_up projections), for
  // QS4CX weights. Without this override, FloatTensor::dot's vector
  // overload had no fast path for QS4CX at all and fell through to a
  // per-weight loop of individual dotQs4cx() calls -- one FastRPC call per
  // expert instead of one per call, same shape of miss as the Q4_0 batch
  // override being written but the caller-side grouping shipping first
  // (htp_compute_ops.cpp's gemm_q4_0_batch_fp32 comment).
  virtual bool supports_gemm_qs4cx_batch_fp32() const { return false; }
  virtual void
  gemm_qs4cx_batch_fp32(std::vector<void *> matAdata,
                        std::vector<float *> matAscale, float *matBdata,
                        std::vector<float *> matCdata, unsigned int M,
                        std::vector<unsigned int> N, unsigned int K);

  // Fused MoE expert FFN (doc 43 §[L2]): ONE accelerator call computes
  // gate_up @ act -> SwiGLU -> down @, the SwiGLU intermediate never
  // leaving the accelerator. matAdata/matAscale/N carry exactly TWO QS4CX
  // weights, [gate_up (K x 2I), down (I x N_out)] -- N[0] == 2 * down.K is
  // the SwiGLU contract the impl must check. Callers gate on M > 1
  // themselves (decode's single token cannot amortize the fused call's
  // 64-row pad tax); without an override the caller keeps its two-dot +
  // host-activation fallback.
  virtual bool supports_gemm_qs4cx_fused_swiglu_fp32() const { return false; }
  virtual void gemm_qs4cx_fused_swiglu_fp32(std::vector<void *> matAdata,
                                            std::vector<float *> matAscale,
                                            float *matBdata, float *matCdata,
                                            unsigned int M,
                                            std::vector<unsigned int> N,
                                            unsigned int K);

  // A whole MoE FFN layer -- routing, every expert, and the scatter-add --
  // in one accelerator call (doc 46). Where gemm_qs4cx_fused_swiglu_fp32
  // above is one expert and the caller loops, this takes the routing table
  // and does the looping on the accelerator, so the token gather, the
  // routing multiply and the scatter-add stop crossing the boundary too.
  //
  // gate_up[e] is K x 2*inter and down[e] is inter x N_out. row_index holds
  // the token row each (expert, slot) works on, grouped by expert in expert
  // order; row_count says how many belong to each and must sum to
  // row_index's size; row_weight is the routing weight per entry. out is
  // M x N_out, zeroed by the implementation and accumulated into, because
  // experts share token rows.
  //
  // By const reference, unlike the vector-by-value overloads above: a
  // prefill layer's row_index is one entry per token per expert -- 1776 for
  // this model -- and copying that per layer is the kind of cost this call
  // exists to remove.
  //
  // w_bits says how the nibbles are stored: 0 = plain QS4CX (the backend
  // converts), 4 = QS4CX_WH (already in the HMX weight-tile layout),
  // 2 = QS2CX_WH (half that, expanded on the DSP -- doc 54); 1 reads as 4
  // (#4415's lfm2_moe passes its bool weights_wh here)
  // (htp_wh_layout.h) with a per-output-channel column sum after the scales,
  // so the implementation registers them as they are instead of converting
  // and baking. It is a flag rather than a colsum pointer because the sums
  // sit immediately after the scales and the callee already knows N.
  //
  // gelu says the gated activation between gate_up and down is
  // gelu_tanh(g)*u (Gemma-4) rather than silu(g)*u (LFM2).
  /**
   * @brief out = scale * (res + rmsnorm(act . W) * gamma): an FC over a
   *        QS4CX weight (data @a w, scales @a w_scale, K x N) with the
   *        decoder block's post-attention epilogue on its output, in one
   *        accelerator call (doc 57 section 9.32). False when not taken:
   *        the caller then runs the FC and the add itself.
   */
  /**
   * @brief out_to_host false on gemm_qs4cx_res_add_fp32, the dense FFN, the
   *        MoE layer call and gemm_q4_0_batch_norm_fp32 (whose reader takes
   *        host_view): the layer says only this engine's calls read
   *        that output at prefill, so the engine may keep it and leave the
   *        host tensor (mostly) unwritten. A caller about to read such a
   *        tensor on the host -- a CPU fallback -- calls sync_to_host on it
   *        first; a no-op where the host already has it.
   */
  virtual void sync_to_host(void *host, size_t bytes) {
    (void)host, (void)bytes;
  }
  /** @brief Where a tensor's bytes are for a host reader: the engine's own
   *  (host-mapped) buffer when it kept them (gemm_q4_0_batch_norm_fp32 with
   *  out_to_host false), else @a host. Valid until the engine's next call. */
  virtual const void *host_view(const void *host, size_t bytes) {
    (void)bytes;
    return host;
  }
  virtual bool gemm_qs4cx_res_add_fp32(void *w, float *w_scale,
                                       const float *act, unsigned int M,
                                       unsigned int K, unsigned int N,
                                       const float *res, const float *gamma,
                                       float eps, float scale, float *out,
                                       bool out_to_host = true) {
    (void)w, (void)out_to_host;
    (void)w_scale;
    (void)act;
    (void)M;
    (void)K;
    (void)N;
    (void)res;
    (void)gamma;
    (void)eps;
    (void)scale;
    (void)out;
    return false;
  }

  // add_x2 (doc 57 section 9.30): the block's post-FFN epilogue on the
  // output, out = add_scale * (act + rmsnorm(moe + add_x2) * add_gamma),
  // with act the raw rows (pre_gamma set, K == N_out).
  virtual bool supports_gemm_qs4cx_moe_layer_fp32() const { return false; }
  virtual void gemm_qs4cx_moe_layer_fp32(
    const std::vector<void *> &gate_up_data,
    const std::vector<float *> &gate_up_scale,
    const std::vector<void *> &down_data,
    const std::vector<float *> &down_scale,
    const std::vector<unsigned int> &row_index,
    const std::vector<unsigned int> &row_count,
    const std::vector<float> &row_weight, const float *act, float *out,
    unsigned int M, unsigned int K, unsigned int inter, unsigned int N_out,
    unsigned int w_bits, bool gelu = false, const float *pre_gamma = nullptr,
    const float *post_gamma = nullptr, float eps = 0.0f,
    const float *add_x2 = nullptr, const float *add_gamma = nullptr,
    float add_scale = 1.0f, bool out_to_host = true);

  // [#85] Hands the accelerator the decode step's op list (the words of
  // htp_backend/htp_graph_desc.h, built by the model) so it can validate
  // the list once and run resident ops from one entry per token. Words,
  // not a struct: core gains no accelerator type. A backend without a
  // per-token entry returns false and the model runs as before.
  virtual bool set_decode_graph_desc(const std::vector<uint32_t> &words) {
    (void)words;
    return false;
  }

  // [#132 Part B] One Q4_0 weight of the decode list's FC, dense FFN or
  // lm_head ops, handed by the model at load in list order: K x N as the
  // FC reads it, canonical block_q4_0 rows when @a canonical (the tied
  // embedding table), else this ISA's repack. The backend keeps the
  // pointer until its graph init. Returns false when it takes none.
  virtual bool add_decode_graph_q4_0(const void *data, unsigned K, unsigned N,
                                     bool canonical) {
    (void)data;
    (void)K;
    (void)N;
    (void)canonical;
    return false;
  }

  // [#260] add_decode_graph_q4_0 for a QS4CX FC or dense FFN weight (K x N
  // codes, @a scale its N per-column f32 scales), in the same list order.
  // The backend binds the handles its prefill registers from the same bytes
  // (one image set); no Q4M1 or CPU fallback. Returns false when it takes
  // none.
  // Register an in-file QS4CX_WH record as the source for FC/dense kernels.
  // The pointer remains valid until the backend has placed its handles.
  virtual bool set_fc_wh_source(const void *data, unsigned K, unsigned N,
                                bool affine = false) {
    (void)data;
    (void)K;
    (void)N;
    (void)affine;
    return false;
  }

  virtual bool add_decode_graph_qs4cx(const void *data, const float *scale,
                                      unsigned K, unsigned N) {
    (void)data;
    (void)scale;
    (void)K;
    (void)N;
    return false;
  }

  // The untied QS4CX lm_head's WH slices as a file beside the model
  // (<model file>.lmhead_wh): read straight into the accelerator's arena at
  // load when it matches the weight, written after the first conversion
  // otherwise, so no later load converts the head. No-op without HTP.
  virtual void set_lm_head_wh_cache(const char *path) { (void)path; }

  // [plan 201 S4] One f32 parameter of the decode list's op @a op (which:
  // HTP_GRAPH_PARAM_*, n floats), handed by the model at load by the
  // weight's name instead of by its layer's hook order (Gemma 4 runs its
  // two FFN branches in the other order than its list). The backend checks
  // the length now, keeps the pointer and binds it at graph init, so the
  // data must stay where it is until the first decode token (a resident
  // weight, or a buffer the model owns); the op's hook then binds
  // nothing. False when the backend takes none.
  virtual bool set_decode_graph_param(unsigned op, unsigned which,
                                      const float *data, unsigned n) {
    (void)op;
    (void)which;
    (void)data;
    (void)n;
    return false;
  }
  // [plan 201 S4] The model's gated activation for the accelerator's MoE
  // experts and dense FFN: true = GeGLU (gelu_tanh(gate) * up, Gemma 4),
  // false = SwiGLU (the default). Set before the first MoE call.
  virtual void set_moe_geglu(bool on) { (void)on; }

  // [#130] The per-token hook a CPU layer calls at one decode row before
  // running its own kernel: kind is the op kind of htp_graph_desc.h, pos
  // the absolute token position, in / out the row (in may be null when
  // the accelerator already holds the stretch's input), param the op's
  // f32 parameter (a norm's gamma, a conv op's w0 | w1 | w2, the RoPE
  // table for attention), state the conv state x_{t-2} | x_{t-1}. Returns
  // 0 when the op is not resident (the layer runs its CPU path), 1 when
  // out was written by the accelerator, 2 when the attention cache must
  // be seeded first: the layer then hands rows [0, pos) of its KV cache to
  // decode_kv_seed_fp32 and calls again; [plan 201 S4] 3 when an attention
  // hook brought no RoPE table and its ROPE op has none bound: the layer
  // builds its table and calls again. Plain scalars and pointers: core
  // gains no accelerator type.
  virtual int decode_op_fp32(unsigned kind, unsigned pos, const float *in,
                             unsigned in_len, float *out, unsigned out_len,
                             const float *param, unsigned param_len,
                             const float *state, unsigned state_len,
                             float eps) {
    (void)kind;
    (void)pos;
    (void)in;
    (void)in_len;
    (void)out;
    (void)out_len;
    (void)param;
    (void)param_len;
    (void)state;
    (void)state_len;
    (void)eps;
    return 0;
  }
  // [#132 Part B] True while the accelerator runs the whole decode row at
  // pos (every kind resident: the token is one stretch, and its first
  // hook has handed the row over): every CPU FC result of the row is then
  // discarded, so the layers skip their GEMVs. False everywhere else, so
  // the CPU path is unchanged.
  virtual bool decode_row_resident(unsigned pos) {
    (void)pos;
    return false;
  }
  // [#132 Part B E3] The two-session decode (NNTR_HTP_E2E=1). finish: the
  // model handed its Q4_0 weights and mapped everything else it maps at
  // load (after repack_weight): the backend may open its second session
  // and register them now. set_decode_logits(false): the caller
  // takes each decode token's id (take_decode_token_id) instead of its
  // logits, so only the id travels back; true (the default) keeps the
  // logits. take_decode_token_id: the id of the last decode token whose
  // logits did not come back, once; false when there is none.
  virtual bool finish_decode_graph_q4_0() { return false; }
  // [#234 P4] True when the one-PD decode list (NNTR_HTP_E2E=1) holds Q4_0
  // weights it has not bound yet: the loader then hands this backend the
  // FC WH sidecar (set_fc_wh_file) even when no layer keyed an FC to it.
  virtual bool has_decode_graph_q4_0() { return false; }
  virtual void set_decode_logits(bool want) { (void)want; }
  // [#132 Part B E3] The ids the caller's greedy pick sets to -inf (its
  // bad words): an id taken with take_decode_token_id skips them too.
  virtual void set_decode_ban(const unsigned *ids, unsigned n) {
    (void)ids;
    (void)n;
  }
  virtual bool take_decode_token_id(unsigned *id) {
    (void)id;
    return false;
  }
  virtual bool decode_kv_seed_fp32(unsigned n_rows, const float *k_rows,
                                   const float *v_rows) {
    (void)n_rows;
    (void)k_rows;
    (void)v_rows;
    return false;
  }
  // The host's KV caches of the windowed (sliding) attention layers hold
  // a window of the context: absolute position a is their row a - base and
  // they have `rows` rows. Told before the first decode row, so the
  // accelerator's own caches of those layers are that size and its
  // attention reads row pos - base; the seed carries the rows from the
  // base on. No-op where no decode graph is resident.
  virtual void set_decode_kv_window(unsigned rows, unsigned base) {
    (void)rows;
    (void)base;
  }
  // [#4416 D6] The same seed from the cache's fp16 bits, widened on the
  // accelerator (exactly, as compute_fp16_to_fp32): no CPU conversion.
  virtual bool decode_kv_seed_f16(unsigned n_rows, const uint16_t *k_rows,
                                  const uint16_t *v_rows) {
    (void)n_rows;
    (void)k_rows;
    (void)v_rows;
    return false;
  }
  // [#289] The same seed with V from the layer's own quantized prefill
  // cache (kv_cache_q_register's handle, fixed scales) on the accelerator
  // and K from k_rows ([n_rows][n_kv x head_dim] f32); the handle is
  // released on success. False: not taken, seed with decode_kv_seed_fp32.
  virtual bool decode_kv_seed_kvq(unsigned n_rows, int kv_handle,
                                  const float *k_rows) {
    (void)n_rows;
    (void)kv_handle;
    (void)k_rows;
    return false;
  }

  /** Whether the complete decode graph consumes this registered source.
   * Per-op M=1 capability can be false when the whole token runs on device.
   */
  virtual bool uses_device_weight(const void *data) const {
    (void)data;
    return false;
  }

  /** Release an exclusively accelerated weight's host pages after registration.
   * The caller guarantees that no CPU consumer or fallback reads this range.
   * The address remains a registry key; its backing allocation still owns it.
   */
  virtual void discard_weight_source(void *data, size_t bytes) {
    (void)data;
    (void)bytes;
  }

  // Registers one K x N expert weight with the accelerator ahead of its
  // first use, so a model's load pays that cost rather than its first
  // prefill: for the 1408 weights of LFM2-8B-A1B it is 747 ms, 31% of the
  // measured prefill (doc 46 section 50.2 P6). The same call at forward time
  // then finds the weight already registered. A backend with nothing to
  // register returns false and the caller moves on.
  virtual bool register_qs4cx_weight(void *data, const float *scale,
                                     unsigned int K, unsigned int N,
                                     unsigned int w_bits) {
    (void)data;
    (void)scale;
    (void)K;
    (void)N;
    (void)w_bits;
    return false;
  }

  // The Q4_0 twin, for a fully_connected layer under engine=htp: its
  // weight is otherwise converted and registered by the first dot() that
  // reaches gemm_q4_0_accel_fp32, which is inside the first prefill (doc
  // 50). Called from Transformer::repack_weight with the data pointer that
  // dot() will pass, so the forward-time lookup is a cache hit.
  virtual bool register_q4_0_weight(void *data, unsigned int K,
                                    unsigned int N) {
    (void)data;
    (void)K;
    (void)N;
    return false;
  }

  // [#225] The FC WH sidecar (htp_wh_layout.h) the model names in
  // nntr_config.json's fc_wh_file_name: the three register_q4_0_* hooks
  // then take each Q4_0 weight's image from it instead of re-quantizing.
  // Called once, before them. false: this backend does not read one.
  virtual bool set_fc_wh_file(const char *path) {
    (void)path;
    return false;
  }

  // A QS4CX_WH expert pair the loader never read (a virtual weight, doc
  // 52), and where its bytes are: gate_up [K, 2 * inter] at off_gu and down
  // [inter, N_out] at off_dn in the model file behind fd, each laid out as
  // QS4CX_WH_Tensor writes it -- [WH nibbles][N scales][N column sums].
  // The keys are what the caller passes as gate_up_data / down_data, with
  // a null scale, to gemm_qs4cx_moe_layer_fp32 once the expert is in.
  struct ExpertFileDesc {
    const void *key_gu;
    const void *key_dn;
    int fd;
    size_t off_gu, off_dn;
    unsigned int K, inter, N_out;
    unsigned int w_bits = 4; /**< [plan 229] 4 QS4CX_WH, 2 QS2CX_WH */
    /** [PR 4428] QS4CX_WH_AFFINE: 4-bit codes with N i32 offsets after the
     *  scales and sums; the file's values go to the DSP as they are. */
    bool affine = false;
  };

  // How many expert slots the caller will ever hold at once (the LRU's
  // capacity), so the backend can size its memory to that instead of to a
  // fixed chunk. Advisory; call before the first register. Doc 52 section
  // 10.14: at NNTR_MOE_CACHE_EXPERTS=1 the pool is 116 MiB, and a 256 MiB
  // chunk would hide the saving.
  // Supplying a shape also allocates the slots, without reading weights or
  // registering experts. This keeps arena mapping out of the first prefill
  // even when the caller chooses a smaller load-time preload budget.
  virtual void
  reserve_qs4cx_wh_expert_slots(size_t n,
                                const ExpertFileDesc *shape = nullptr) {
    (void)n;
    (void)shape;
  }

  // Reads one expert from the model file into a slot the backend owns and
  // registers it. Idempotent for a key already resident. at_load says
  // whether the profile counts it as load-time registration or as a cache
  // miss. A backend without a slot pool returns false.
  virtual bool register_qs4cx_wh_expert_file(const ExpertFileDesc &d,
                                             bool at_load) {
    (void)d;
    (void)at_load;
    return false;
  }

  // The same for several experts, none at load (doc 52 section 10.23): a
  // backend with a batched register makes one round trip for all of them.
  virtual bool
  register_qs4cx_wh_expert_files(const std::vector<ExpertFileDesc> &ds) {
    for (const ExpertFileDesc &d : ds)
      if (!register_qs4cx_wh_expert_file(d, /*at_load=*/false))
        return false;
    return true;
  }

  // The same, split so the file reads overlap other work (doc 52 sections
  // 10.10, 10.20): _begin queues one batch -- a slot per expert, read in the
  // background -- and returns; several batches may be in flight. _end waits
  // for the OLDEST batch, registers it, and returns the key_gu of each of
  // its experts now resident (empty when none is queued). The caller must
  // _end a batch before any ComputeOps call that touches its experts, only
  // between accelerator calls, and must have made room -- _begin takes
  // only free or new slots. False / empty from a backend without a slot
  // pool.
  virtual bool
  prefetch_qs4cx_wh_experts_begin(const std::vector<ExpertFileDesc> &ds) {
    (void)ds;
    return false;
  }
  virtual std::vector<const void *> prefetch_qs4cx_wh_experts_end() {
    return {};
  }

  // [plan 201 S1] The pool's policy for the per-token entry's miss path:
  // makes every key of need resident, calling evict for each key that
  // leaves (before any load) and load for each that must come in -- the
  // layer's ExpertLru::acquire.
  using ExpertPoolFn =
    std::function<void(const std::vector<const void *> &need,
                       const std::function<void(const void *)> &load,
                       const std::function<void(const void *)> &evict)>;

  // [plan 201 S1] One MoE layer's experts (all of them, expert order), in
  // layer order across calls, and the pool's policy: with NNTR_HTP_E2E=1
  // the backend serves S1's misses from these while a decode token runs.
  // A backend without the per-token entry ignores it.
  virtual void set_decode_moe_experts(const std::vector<ExpertFileDesc> &all,
                                      const ExpertPoolFn &pool) {
    (void)all;
    (void)pool;
  }

  // [#4416] How the backend raises the pool's capacity by n keys once the
  // prefill's buffers are released (NNTR_MOE_CACHE_GROW): the layer's
  // ExpertLru::addCapacity. Ignored by a backend without the pool.
  virtual void set_decode_pool_grow(const std::function<void(size_t)> &grow) {
    (void)grow;
  }

  // Undoes the above for one expert: both handles released, the slot back
  // in the pool for the next register_qs4cx_wh_expert_file. False when the
  // key is not resident.
  virtual bool release_qs4cx_wh_expert(const void *key_gu) {
    (void)key_gu;
    return false;
  }

  // The dense SwiGLU FFN as ONE accelerator call (doc 51): up and gate
  // [K x I] and down [I x N], all Q4_0x4 as loaded; act [M x K] f32 ->
  // out [M x N] f32 = (silu(act . gate) * (act . up)) . down, or with
  // gelu, (gelu_tanh(act . gate) * (act . up)) . down. The HTP answers it
  // with its MoE layer kernel over column chunks of I, each chunk a whole
  // "expert" whose down output is summed into out.
  // register_q4_0_dense_ffn is the load-time twin, like the two above.
  virtual bool supports_gemm_q4_0_dense_ffn_fp32() const { return false; }
  // pre_gamma (K floats) / post_gamma (N floats), nullptr for none, and
  // eps: the RMSNorms before and after the block, folded into the call as
  // gemm_q4_0_batch_norm_fp32 folds them. up_scale / gate_scale /
  // down_scale: non-null when the three weights are QS4CX (their
  // per-channel scales), null for Q4_0.
  virtual void gemm_q4_0_dense_ffn_fp32(
    void *up, void *gate, void *down, const float *act, float *out,
    unsigned int M, unsigned int K, unsigned int I, unsigned int N,
    bool gelu = false, const float *pre_gamma = nullptr,
    const float *post_gamma = nullptr, float eps = 0.0f,
    const float *up_scale = nullptr, const float *gate_scale = nullptr,
    const float *down_scale = nullptr, bool out_to_host = true) {
    (void)gelu, (void)out_to_host;
    (void)pre_gamma;
    (void)post_gamma;
    (void)eps;
    (void)up_scale;
    (void)gate_scale;
    (void)down_scale;
    (void)up;
    (void)gate;
    (void)down;
    (void)act;
    (void)out;
    (void)M;
    (void)K;
    (void)I;
    (void)N;
    throw std::runtime_error(
      "ComputeOps::gemm_q4_0_dense_ffn_fp32 not implemented by this backend");
  }
  virtual bool register_q4_0_dense_ffn(void *up, void *gate, void *down,
                                       unsigned int K, unsigned int I,
                                       unsigned int N,
                                       const float *up_scale = nullptr,
                                       const float *gate_scale = nullptr,
                                       const float *down_scale = nullptr) {
    (void)up_scale;
    (void)gate_scale;
    (void)down_scale;
    (void)up;
    (void)gate;
    (void)down;
    (void)K;
    (void)I;
    (void)N;
    return false;
  }

  // An LFM2 conv block as ONE accelerator call (doc 51 section 2):
  // in_proj [K x 3C] and out_proj [C x N] Q4_0x4 as loaded, conv_w [3 x C]
  // f32 (w0 for row t, w1 for t-1, w2 for t-2); act [M x K] f32 ->
  // out [M x N] f32 = ((b * conv1d(a * c)) . out_proj) with a | b | c the
  // column thirds of act . in_proj, and state [2 x C] f32 <- the conv
  // input's rows M-2 and M-1 (zero where M is shorter), the state the
  // CPU decode path continues from. register_q4_0_conv_block is the
  // load-time twin.
  virtual bool supports_gemm_q4_0_conv_block_fp32() const { return false; }
  virtual void gemm_q4_0_conv_block_fp32(void *in_proj, const float *conv_w,
                                         void *out_proj, const float *act,
                                         float *out, float *state,
                                         unsigned int M, unsigned int K,
                                         unsigned int C, unsigned int N) {
    (void)in_proj;
    (void)conv_w;
    (void)out_proj;
    (void)act;
    (void)out;
    (void)state;
    (void)M;
    (void)K;
    (void)C;
    (void)N;
    throw std::runtime_error(
      "ComputeOps::gemm_q4_0_conv_block_fp32 not implemented by this backend");
  }
  virtual bool register_q4_0_conv_block(void *in_proj, void *out_proj,
                                        unsigned int K, unsigned int C,
                                        unsigned int N) {
    (void)in_proj;
    (void)out_proj;
    (void)K;
    (void)C;
    (void)N;
    return false;
  }

  /**
   * @brief Fused causal attention over an fp16 KV cache (MHACoreLayer's
   *        compute_kcaches + softmax_triangle + compute_fp16vcache in one
   *        call, no materialized logits).
   *
   * out[q][(n*G+g)*hd + d] = softmax_k(q_row . k_row / sqrt(hd)) . v over
   * cache rows [max(0, pos+1-window), pos], pos = cache_from + q, with
   * G = n_head_q / n_head_kv query heads per KV head. Optional logit
   * softcap tanh(s/softcap)*softcap (0 = off) and per-head sink logits
   * (nullptr = none), both with MHACoreLayer's semantics.
   *
   * @param q          f32 [n_q][q_stride], post-RoPE, head h at column h*hd
   * @param k_cache    fp16 bit patterns [cache_to][kv_stride], post-RoPE
   * @param v_cache    fp16 bit patterns [cache_to][kv_stride]
   * @param window     sliding window; 0 means unlimited
   * @param sinks      n_head_q floats or nullptr
   * @param out        f32 [n_q][out_stride]
   * @return true on success; false means the accelerator could not run
   *         this call (shape or transport) and the caller should take the
   *         CPU path for it. Unlike the gemm ops above this never throws
   *         for a runtime failure: attention sits on the per-token hot
   *         path and a fallback is always available.
   */
  virtual bool supports_sdpa_fp16_kvcache() const { return false; }
  virtual bool sdpa_fp16_kvcache(
    const float *q, unsigned int q_stride, const uint16_t *k_cache,
    const uint16_t *v_cache, unsigned int kv_stride, unsigned int n_q,
    unsigned int cache_from, unsigned int cache_to, unsigned int n_head_q,
    unsigned int n_head_kv, unsigned int head_dim, unsigned int window,
    float softcap, const float *sinks, float *out, unsigned int out_stride);

  /**
   * @brief Memory the backend's accelerator reads in place.
   *
   * sdpa_fp16_kvcache's K/V cache reaches the accelerator without a copy
   * only when it lives in memory obtained here: on the HTP this is rpcmem,
   * a dma-buf FastRPC maps into the DSP and cache-maintains per call, where
   * a cache in ordinary memory is copied into a scratch buffer on every
   * call (the whole used range, every token). The default backend has no
   * such memory and returns nullptr; a caller then allocates as usual and
   * still gets correct results, just with the copy.
   *
   * @param bytes block size
   * @return a block at least 16-byte aligned, or nullptr when this backend
   *         has no shared memory or the allocation failed
   */
  virtual void *alloc_shared(size_t bytes) {
    (void)bytes;
    return nullptr;
  }

  /**
   * @brief Releases a block from alloc_shared(). nullptr is a no-op.
   */
  virtual void free_shared(void *block) { (void)block; }

  /**
   * @brief [#289] Frees an alloc_shared() block before its owner does; the
   *        owner's free_shared() of it is then a no-op. Not a live block:
   *        ignored.
   */
  virtual void drop_shared(void *block) { (void)block; }

  /**
   * @brief A quantized copy of an attention layer's KV cache that lives on
   *        the accelerator, appended row by row and attended over in place.
   *
   * The layer keeps its fp16 cache as the source of truth and mirrors the
   * rows it writes: kv_cache_q_register() once per (layer, batch) with the
   * cache's capacity, kv_cache_q_append() with the fp16 rows of each step
   * (or of any range that changed, e.g. after a cache load or rewind --
   * rows may be rewritten), then sdpa_q_kvcache() by handle with the same
   * contract as sdpa_fp16_kvcache() minus the cache pointers. The
   * accelerator quantizes at append time (per-token scales) and never
   * revisits a row; how it stores the cache is its own business. The
   * default backend has none: register returns -1 and the layer stays on
   * its fp16 path.
   *
   * @param kind      0 = int8 (A8W8), 1 = int4 (A8W4)
   * @param max_rows  cache capacity in rows
   * @return a handle >= 0, or -1 when this backend has no quantized cache
   *         or the registration failed
   */
  virtual bool supports_kv_cache_q() const { return false; }
  virtual int kv_cache_q_register(unsigned int kind, unsigned int max_rows,
                                  unsigned int n_head_kv,
                                  unsigned int head_dim) {
    (void)kind;
    (void)max_rows;
    (void)n_head_kv;
    (void)head_dim;
    return -1;
  }

  /**
   * @brief Writes cache rows [row0, row0 + n_rows) of the fp16 cache into
   *        the accelerator's copy.
   *
   * @param kv_stride  elements per cache row (n_head_kv * head_dim); rows
   *                   are dense fp16 bit patterns, K post-RoPE, V raw
   * @return false on a transport failure; the caller then drops the
   *         handle and takes the fp16 path
   */
  virtual bool kv_cache_q_append(int handle, unsigned int row0,
                                 unsigned int n_rows, unsigned int kv_stride,
                                 const uint16_t *k_rows,
                                 const uint16_t *v_rows) {
    (void)handle;
    (void)row0;
    (void)n_rows;
    (void)kv_stride;
    (void)k_rows;
    (void)v_rows;
    return false;
  }

  virtual void kv_cache_q_release(int handle) { (void)handle; }

  /**
   * @brief The row-blocked int8 attention (hexkl_attn_q2) over a cache in
   *        fixed-scale mode: K one scale per KV head, V one per (KV head,
   *        dim), Q one per query head, all given by the caller as a
   *        quantized model's encodings give them. kv_cache_q_set_fixed_scales
   *        must be called once after kv_cache_q_register and before the
   *        first append; sdpa_q2_kvcache then has sdpa_q_kvcache's contract
   *        with the per-head Q scales added and softcap / sinks removed
   *        (neither exists in the models this path serves). head_dim up to
   *        512.
   */
  virtual bool supports_kv_cache_q2() const { return false; }
  virtual bool kv_cache_q_set_fixed_scales(int handle, unsigned int n_head_kv,
                                           unsigned int head_dim,
                                           const float *s_k, const float *s_v) {
    (void)handle;
    (void)n_head_kv;
    (void)head_dim;
    (void)s_k;
    (void)s_v;
    return false;
  }
  virtual bool sdpa_q2_kvcache(int handle, unsigned int append_row0,
                               unsigned int append_rows, unsigned int kv_stride,
                               const uint16_t *k_rows, const uint16_t *v_rows,
                               const uint16_t *q, const float *q_enc,
                               unsigned int q_stride, unsigned int n_q,
                               unsigned int cache_from, unsigned int cache_to,
                               unsigned int n_head_q, unsigned int n_head_kv,
                               unsigned int head_dim, unsigned int window,
                               uint16_t *out, const float *out_enc,
                               unsigned int out_stride) {
    (void)handle;
    (void)append_row0;
    (void)append_rows;
    (void)kv_stride;
    (void)k_rows;
    (void)v_rows;
    (void)q;
    (void)q_enc;
    (void)q_stride;
    (void)n_q;
    (void)cache_from;
    (void)cache_to;
    (void)n_head_q;
    (void)n_head_kv;
    (void)head_dim;
    (void)window;
    (void)out;
    (void)out_enc;
    (void)out_stride;
    return false;
  }
  /**
   * @brief sdpa_q2_kvcache with f32 Q and the output left on the
   *        accelerator: Q is quantized there (quant_u16_f32 with q_inv[h]
   *        = 1 / q_enc[2h], the same u16) and the u16 context stays in the
   *        accelerator's buffer for the o-proj that follows
   *        (gemm_qs4cx_res_add_fp32 with @a out as its activation reads it
   *        from there and dequantizes it on the way in). @a out is NOT
   *        written: the caller must know the o-proj runs on this engine.
   *        @a out_u16_copy, when given, receives the u16 context (n_q x
   *        out_stride) for a host check. false = not available, the caller
   *        converts.
   */
  /**
   * @brief [P2] sdpa_q2_kvcache_f32 with the layer's calibration folded
   *        in: the accelerator scans this step's fp16 K / V rows and f32 Q
   *        rows for the maxima, derives the fixed int8 scales and the u16
   *        encodings exactly as the layer's calibrate_q2_scales does, sets
   *        the scales on the cache and runs the step. s_k[n_head_kv],
   *        s_v[n_head_kv * head_dim], q_enc[2 n_head_q], out_enc[2
   *        n_head_q] receive them. false = not available, the caller
   *        calibrates.
   */
  virtual bool sdpa_q2_kvcache_f32_calib(
    int handle, unsigned int append_row0, unsigned int append_rows,
    unsigned int kv_stride, const uint16_t *k_rows, const uint16_t *v_rows,
    const float *q, unsigned int q_stride, unsigned int n_q,
    unsigned int cache_from, unsigned int cache_to, unsigned int n_head_q,
    unsigned int n_head_kv, unsigned int head_dim, unsigned int window,
    float *out, float *s_k, float *s_v, float *q_enc, float *out_enc,
    unsigned int out_stride) {
    (void)handle, (void)append_row0, (void)append_rows, (void)kv_stride;
    (void)k_rows, (void)v_rows, (void)q, (void)q_stride, (void)n_q;
    (void)cache_from, (void)cache_to, (void)n_head_q, (void)n_head_kv;
    (void)head_dim, (void)window, (void)out, (void)s_k, (void)s_v;
    (void)q_enc, (void)out_enc, (void)out_stride;
    return false;
  }

  virtual bool sdpa_q2_kvcache_f32(
    int handle, unsigned int append_row0, unsigned int append_rows,
    unsigned int kv_stride, const uint16_t *k_rows, const uint16_t *v_rows,
    const float *q, const float *q_inv, const float *q_enc,
    unsigned int q_stride, unsigned int n_q, unsigned int cache_from,
    unsigned int cache_to, unsigned int n_head_q, unsigned int n_head_kv,
    unsigned int head_dim, unsigned int window, float *out,
    uint16_t *out_u16_copy, const float *out_enc, unsigned int out_stride) {
    (void)handle, (void)append_row0, (void)append_rows, (void)kv_stride;
    (void)k_rows, (void)v_rows, (void)q, (void)q_inv, (void)q_enc;
    (void)q_stride, (void)n_q, (void)cache_from, (void)cache_to;
    (void)n_head_q, (void)n_head_kv, (void)head_dim, (void)window;
    (void)out, (void)out_u16_copy, (void)out_enc, (void)out_stride;
    return false;
  }

  /**
   * @brief sdpa_fp16_kvcache() over a registered quantized cache, with the
   *        rows the cache is missing appended in the same round trip:
   *        rows [append_row0, append_row0 + append_rows) are written from
   *        k_rows / v_rows (dense, kv_stride elements per row) before the
   *        attention runs. append_rows == 0 attends over the rows already
   *        there. One accelerator call per layer per step is what keeps
   *        the transport cost at the fp16 path's.
   */
  virtual bool sdpa_q_kvcache(
    int handle, unsigned int append_row0, unsigned int append_rows,
    unsigned int kv_stride, const uint16_t *k_rows, const uint16_t *v_rows,
    const float *q, unsigned int q_stride, unsigned int n_q,
    unsigned int cache_from, unsigned int cache_to, unsigned int n_head_q,
    unsigned int n_head_kv, unsigned int head_dim, unsigned int window,
    float softcap, const float *sinks, float *out, unsigned int out_stride) {
    (void)append_row0;
    (void)append_rows;
    (void)kv_stride;
    (void)k_rows;
    (void)v_rows;
    (void)handle;
    (void)q;
    (void)q_stride;
    (void)n_q;
    (void)cache_from;
    (void)cache_to;
    (void)n_head_q;
    (void)n_head_kv;
    (void)head_dim;
    (void)window;
    (void)softcap;
    (void)sinks;
    (void)out;
    (void)out_stride;
    return false;
  }

  virtual bool supports_gemv_int4_batch_fp32() const { return false; }
  virtual void gemv_int4_batch_fp32(std::vector<void *> weights,
                                    std::vector<uint16_t *> scales,
                                    float *input, std::vector<float *> outputs,
                                    unsigned int K,
                                    std::vector<unsigned int> Ns,
                                    unsigned int group_size);

  virtual bool supports_gemm_int4_batch_fp32() const { return false; }
  virtual void gemm_int4_batch_fp32(float *input, std::vector<void *> weights,
                                    std::vector<uint16_t *> scales,
                                    std::vector<float *> matCdata,
                                    unsigned int M,
                                    std::vector<unsigned int> Ns,
                                    unsigned int K, unsigned int group_size);

  virtual bool supports_gemv_int4_accel_fp32() const { return false; }
  virtual void gemv_int4_accel_fp32(char *weight, uint16_t *scale, float *input,
                                    float *output, unsigned int K,
                                    unsigned int N, unsigned int group_size);

  virtual bool supports_sgemm_int4_accel_fp32() const { return false; }
  virtual void sgemm_int4_accel_fp32(float *input, char *weight,
                                     uint16_t *scale, float *output,
                                     unsigned int M, unsigned int N,
                                     unsigned int K, unsigned int group_size);

#ifdef ENABLE_FP16
  // ===========================================================================
  // FP16 BLAS
  // ===========================================================================
  virtual void sgemm_fp16(const unsigned int TStorageOrder, bool TransA,
                          bool TransB, const unsigned int M,
                          const unsigned int N, const unsigned int K,
                          const float alpha, const _FP16 *A,
                          const unsigned int lda, const _FP16 *B,
                          const unsigned int ldb, const float beta, _FP16 *C,
                          const unsigned int ldc);
  virtual void sgemv_fp16(const unsigned int TStorageOrder, bool TransA,
                          const unsigned int M, const unsigned int N,
                          const float alpha, const _FP16 *A,
                          const unsigned int lda, const _FP16 *X,
                          const unsigned int incX, const float beta, _FP16 *Y,
                          const unsigned int incY);
  virtual _FP16 sdot_fp16(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, const _FP16 *Y,
                          const unsigned int incY);
  virtual void saxpy_fp16(const unsigned int N, const float alpha,
                          const _FP16 *X, const unsigned int incX, _FP16 *Y,
                          const unsigned int incY);
  virtual void scopy_fp16(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, _FP16 *Y,
                          const unsigned int incY);
  virtual void scopy_fp32_to_fp16(const unsigned int N, const float *X,
                                  const unsigned int incX, _FP16 *Y,
                                  const unsigned int incY);
  virtual void scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                                  const unsigned int incX, float *Y,
                                  const unsigned int incY);
  virtual void sscal_fp16(const unsigned int N, const float alpha, _FP16 *X,
                          const unsigned int incX);
  virtual _FP16 snrm2_fp16(const unsigned int N, const _FP16 *X,
                           const unsigned int incX);
  virtual unsigned int isamax_fp16(const unsigned int N, const _FP16 *X,
                                   const unsigned int incX);

  // ===========================================================================
  // FP16 Element-wise
  // ===========================================================================
  virtual void ele_mul_fp16(const unsigned int N, const _FP16 *X,
                            const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_add_fp16(const unsigned int N, const _FP16 *X,
                            const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_sub_fp16(const unsigned int N, const _FP16 *X,
                            const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_div_fp16(const unsigned int N, const _FP16 *X,
                            const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);

  // ===========================================================================
  // FP16 Activation / Special
  // ===========================================================================
  virtual void swiglu_fp16(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z);
  virtual _FP16 max_val_fp16(const unsigned int N, _FP16 *X);
  virtual void softmax_fp16(const unsigned int N, _FP16 *X, _FP16 *Y);
  virtual bool is_valid_fp16(const unsigned int N, const _FP16 *X);
  virtual void inv_sqrt_inplace_fp16(const unsigned int N, _FP16 *X);

  // ===========================================================================
  // FP16 Matrix ops
  // ===========================================================================
  virtual void transpose_matrix_fp16(const unsigned int M, const unsigned int N,
                                     const _FP16 *src, unsigned int ld_src,
                                     _FP16 *dst, unsigned int ld_dst);

  // ===========================================================================
  // FP16 Data conversion
  // ===========================================================================
  virtual void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                                     const unsigned int incX, _FP16 *Y,
                                     const unsigned int incY);
  virtual void scopy_int8_to_float16_u(const unsigned int N, const uint8_t *X,
                                       const unsigned int incX, _FP16 *Y,
                                       const unsigned int incY);
  virtual void scopy_int8_to_float16_s(const unsigned int N, const int8_t *X,
                                       const unsigned int incX, _FP16 *Y,
                                       const unsigned int incY);

  // ===========================================================================
  // Mixed precision BLAS
  // ===========================================================================
  virtual void shgemm(const unsigned int TStorageOrder, bool TransA,
                      bool TransB, const unsigned int M, const unsigned int N,
                      const unsigned int K, const float alpha, const float *A,
                      const unsigned int lda, const _FP16 *B,
                      const unsigned int ldb, const float beta, float *C,
                      const unsigned int ldc);
  virtual void shgemv(const unsigned int TStorageOrder, bool TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const float *A, const unsigned int lda,
                      const _FP16 *X, const unsigned int incX, const float beta,
                      float *Y, const unsigned int incY);
  virtual void hsgemm(const unsigned int TStorageOrder, bool TransA,
                      bool TransB, const unsigned int M, const unsigned int N,
                      const unsigned int K, const float alpha, const _FP16 *A,
                      const unsigned int lda, const float *B,
                      const unsigned int ldb, const float beta, float *C,
                      const unsigned int ldc);
  virtual void hsgemv(const unsigned int TStorageOrder, bool TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const _FP16 *A, const unsigned int lda,
                      const float *X, const unsigned int incX, const float beta,
                      float *Y, const unsigned int incY);

  // ===========================================================================
  // Quantized GEMM (FP16 variants)
  // ===========================================================================
  virtual void gemm_q4_0_fp16(const unsigned int M, const unsigned int N,
                              const unsigned int K, const _FP16 *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, _FP16 *C,
                              const unsigned int ldc);
  virtual void gemm_q6_K_fp16(const unsigned int M, const unsigned int N,
                              const unsigned int K, const _FP16 *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, _FP16 *C,
                              const unsigned int ldc);

  // ===========================================================================
  // Rotary embedding
  // ===========================================================================
  virtual void compute_rotary_embedding_value(unsigned int dim,
                                              unsigned int half_,
                                              unsigned int w, _FP16 *in,
                                              _FP16 *out, float *cos_,
                                              float *sin_);
#endif // ENABLE_FP16

  // ===========================================================================
  // Whole-op (Tensor-level) ops.
  //
  // Every op above takes raw pointers and a length, which is all a CPU kernel
  // needs. An accelerator needs more: it has to know whether the operand
  // already lives in device memory so it can bind that buffer instead of
  // staging the bytes back and forth. A raw pointer cannot answer that, so the
  // ops below take Tensors and the backend implementation introspects them.
  //
  // The division of labour is: a backend-neutral Layer owns structure, shapes
  // and orchestration, ComputeOps owns the whole kernel. That is what lets one
  // Layer class serve every backend instead of a per-backend Layer fork.
  //
  // The row window is expressed as two numbers, (active_rows, row_offset),
  // rather than as sub-Tensor views, because a view is a fresh Tensor and does
  // not necessarily carry the residency state of the tensor it was cut from.
  // Rows are the (batch, channel, height) axes flattened; width() is the
  // per-row element count.
  //
  // Every default below throws, so a backend that does not implement an op
  // reports it instead of silently computing something else.
  // ===========================================================================

  /**
   * @brief GeGLU activation over the `active_rows` rows starting at
   *        `row_offset`: out = gelu_tanh(in1) * in2 ({gate, up} -> result).
   *        in1/in2/out share shape.
   */
  virtual void geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                     unsigned int active_rows, unsigned int row_offset);
  /**
   * @brief out = gelu(in1) * in2 with the exact (erf) gelu -- the HF "gelu"
   *        / approximate="none" GeGLU -- over the same row window as geglu().
   */
  virtual void geglu_erf(const Tensor &in1, const Tensor &in2, Tensor &out,
                         unsigned int active_rows, unsigned int row_offset);

  /**
   * @brief SwiGLU activation over the `active_rows` rows starting at
   *        `row_offset`: out = silu(in1) * in2 = (in1 * sigmoid(in1)) * in2
   *        ({gate, up} -> result). in1/in2/out share shape.
   */
  virtual void swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
                      unsigned int active_rows, unsigned int row_offset);

  /**
   * @brief Sigmoid-gated linear unit over the `active_rows` rows starting at
   *        `row_offset`: out = sigmoid(in1) * in2 ({gate, up} -> result).
   *        in1/in2/out share shape; width() is the per-row element count.
   */
  virtual void sigmoid_glu(const Tensor &in1, const Tensor &in2, Tensor &out,
                           unsigned int active_rows, unsigned int row_offset);

  /**
   * @brief Sigmoid-gated add over the `active_rows` rows starting at
   *        `row_offset`: out = sigmoid(in1) + in2 ({gate, addend} -> result).
   *        The additive counterpart of sigmoid_glu, for architectures that mix
   *        a side signal into the hidden state rather than scaling it.
   *        in1/in2/out share shape; width() is the per-row element count.
   */
  virtual void sigmoid_add(const Tensor &in1, const Tensor &in2, Tensor &out,
                           unsigned int active_rows, unsigned int row_offset);

  /**
   * @brief Row-wise layer normalization over the `active_rows` rows starting at
   *        `row_offset`: out = (x - mean(x)) * rsqrt(var(x) + epsilon) * gamma
   *        + beta, with mean/variance taken over the LAST (width) axis and
   *        accumulated in FP32. in/out share shape; width() is the normalized
   *        dimension. gamma/beta are [1,1,1,W] and are BOTH required; they may
   *        carry the weight dtype, which need not equal the activation dtype.
   *        in/out may alias -- the op is row-independent and reads each element
   *        before it writes it.
   * @note  There is deliberately no `axis` parameter: the op normalizes over
   *        width and nothing else. The Layer owns the axis contract and only
   *        dispatches here when its property matches.
   */
  virtual void layer_norm(const Tensor &in, Tensor &out, const Tensor &gamma,
                          const Tensor &beta, float epsilon,
                          unsigned int active_rows, unsigned int row_offset);

  /**
   * @brief Element-wise activation over the `active_rows` rows starting at
   *        `row_offset`: out = f_act(in). @p act_type is an
   *        nntrainer::ActivationType cast to int, kept as an int so this header
   *        stays free of the layers headers. ACT_NONE copies in -> out. in/out
   *        share shape and MAY alias. A backend implements only the modes it
   *        can accelerate and throws for the rest, so there is no silent host
   *        fallback on a device-resident tensor; query support without
   *        provoking the throw via supports_activation().
   */
  virtual void activation(const Tensor &in, Tensor &out, int act_type,
                          unsigned int active_rows, unsigned int row_offset);

  /**
   * @brief Whether activation() can service @p act_type on this backend (an
   *        nntrainer::ActivationType cast to int). Capability predicate in the
   *        style of supports_shgemm(): a backend that throws for the modes it
   *        cannot accelerate overrides this so callers and tests can ask
   *        first. Default true -- the CPU table services every mode.
   */
  virtual bool supports_activation(int act_type) const {
    (void)act_type;
    return true;
  }

  /**
   * @brief One residual-add operand: hidden = input (accumulate == false, the
   *        first operand) or hidden += input (accumulate == true). The neutral
   *        AdditionLayer calls this once per input so an accelerator can keep
   *        the residual stream device-resident while the CPU table runs the
   *        host Tensor copy/add.
   */
  virtual void residual_op(Tensor &hidden, const Tensor &input,
                           bool accumulate);

  /**
   * @brief The whole two-operand residual add in one go: hidden = a + b.
   *
   * The per-operand residual_op above costs one device dispatch per operand --
   * a copy and then an accumulate -- and on a backend whose per-dispatch
   * submission floor is several times the kernel's own runtime that pair is
   * dominated by the floor, not by the add. A backend that can do both
   * operands in one kernel overrides this; the default declines and the
   * caller keeps the per-operand loop.
   *
   * @param[out] hidden destination
   * @param[in] a first operand
   * @param[in] b second operand
   * @return true when the fused add was issued
   */
  virtual bool residual_op2(Tensor &hidden, const Tensor &a, const Tensor &b) {
    (void)hidden;
    (void)a;
    (void)b;
    return false;
  }

  /**
   * @brief Fully-connected matmul: output = input * weight. The neutral
   *        FullyConnectedLayer owns the weight/bias binding and calls this for
   *        the matmul, so a quantized accelerator GEMM lives in the op table
   *        instead of in a forked Layer. input/weight may carry residency
   *        state the implementation reads, hence non-const.
   */
  virtual void fc(Tensor &input, Tensor &weight, Tensor &output);

  /**
   * @brief Optional one-time weight transform at load time, e.g. an
   *        accelerator repack. Default no-op; called from the layer's read()
   *        once the weights are in memory.
   */
  virtual void fc_prebuild_weight(Tensor &weight) { (void)weight; }

  /**
   * @brief Same one-time weight transform, but reading the weight's payload
   *        from @p src rather than from the tensor.
   *
   * @details The zero-copy weight load: the layer hands the weight file's
   * mapping to the backend so the payload is never copied into host memory at
   * all. Default declines, and a caller that gets false must read the payload
   * the ordinary way before using the weight.
   *
   * @param[in] weight the weight tensor (identity, shape and scales)
   * @param[in] src    the weight's payload bytes in the file mapping
   * @return true when the backend built its weight from @p src
   */
  virtual bool fc_prebuild_weight_from(Tensor &weight, const void *src) {
    (void)weight;
    (void)src;
    return false;
  }

  /**
   * @brief Fused activation epilogue: apply an element-wise activation in
   *        place on a compute layer's output, after GEMM+bias. This is the
   *        in-place special case of activation(): one tensor, whole tensor, so
   *        a backend can fuse it into the GEMM epilogue. @p act_type is an
   *        nntrainer::ActivationType cast to int; ACT_NONE is a no-op.
   */
  virtual void apply_activation(Tensor &out, int act_type);

  /**
   * @brief Mean over rows: out[0, w) is the average over
   *        r in [row_offset, row_offset + active_rows) of in[r, w).
   *        `in` is a [.., rows, W] tensor and `out` a [.., 1, W] tensor.
   * @note  Distinct from the width-axis reductions above: this one reduces
   *        ACROSS rows, not within one. Backs mean-token pooling.
   */
  virtual void mean_rows(const Tensor &in, Tensor &out,
                         unsigned int active_rows, unsigned int row_offset);

  /**
   * @brief Row-wise L2 normalize along the last dimension:
   *        out[r, :] = in[r, :] / max(||in[r, :]||_2, epsilon).
   * @note  The epsilon is a FLOOR ON THE NORM and the denominator uses the SUM
   *        of squares. This is Tensor::normalization_i(3) semantics, which is
   *        NOT RMSNorm (mean of squares, epsilon under the sqrt, mandatory
   *        gamma).
   */
  virtual void l2_normalize_rows(const Tensor &in, Tensor &out, float epsilon);

  // ===========================================================================
  // Mixture-of-experts data movement and expert projection
  //
  // A MoE layer runs each selected expert over the token rows routed to it:
  // gather those rows, project them (gate / up / down FCs), and add the result
  // back into the output rows with the routing weights. The three movement
  // ops below keep those rows on the accelerator; they RETURN whether they ran
  // instead of throwing, and enqueue nothing when they decline, so a layer can
  // keep its host loop for operands an accelerator cannot address. The base
  // (and the CPU table) decline everything.
  // ===========================================================================

  /**
   * @brief out[r, :] = in[src_rows[r], :] for r in [0, rows).
   * @return true when the accelerator ran it
   */
  virtual bool moe_gather_rows(const Tensor &in, Tensor &out,
                               const unsigned int *src_rows,
                               unsigned int rows) {
    (void)in;
    (void)out;
    (void)src_rows;
    (void)rows;
    return false;
  }

  /**
   * @brief out[dst_rows[r], :] += weights[r] * expert_out[r, :] for r in
   *        [0, rows). The destination rows must be distinct within a call;
   *        calls are ordered, so a caller that visits the experts in a fixed
   *        order gets the same sums on every run.
   * @return true when the accelerator ran it
   */
  virtual bool moe_scatter_add_rows(const Tensor &expert_out, Tensor &out,
                                    const unsigned int *dst_rows,
                                    const float *weights, unsigned int rows) {
    (void)expert_out;
    (void)out;
    (void)dst_rows;
    (void)weights;
    (void)rows;
    return false;
  }

  /**
   * @brief out[0 .. rows, :] = 0.
   * @return true when the accelerator ran it
   */
  virtual bool moe_zero_rows(Tensor &out, unsigned int rows) {
    (void)out;
    (void)rows;
    return false;
  }

  /**
   * @brief Run every selected expert of a MoE call at once: GeGLU expert
   *        FFN of each routed row and the weighted combine into the output.
   * @details For expert formats whose kernels batch several experts in one
   * launch (decode: all selected experts' gate / up as one batched GEMV;
   * prefill: one grouped GEMM over all experts' gathered rows). Returns
   * false, having enqueued nothing, when the backend does not serve the
   * expert dtype; the layer then runs its per-expert loop.
   */
  virtual bool moe_experts(const MoeExpertGroup &group) {
    (void)group;
    return false;
  }

  /**
   * @brief Route @a tokens rows of @a router.input on the accelerator: the
   *        deterministic router (moe_route_det.h), so the selected experts
   *        and their weights are the host router's, bit for bit.
   * @details Blocks until the routing is read back (one read of
   * tokens x topk ids and weights; the router rows themselves stay on the
   * device). Returns false, having enqueued nothing, when the backend does
   * not route on the device; the caller then routes on the host.
   * @param[out] ids     [tokens][topk] expert ids, in rank order (larger
   *                     probability first, ties to the lower id)
   * @param[out] weights [tokens][topk] their routing weights
   */
  virtual bool moe_route(const MoeRouter &router, unsigned int tokens,
                         uint32_t *ids, float *weights) {
    (void)router;
    (void)tokens;
    (void)ids;
    (void)weights;
    return false;
  }

  /**
   * @brief Build the backend's own copy of every expert weight of a MoE
   *        layer now (at load), the copy moe_experts() reads.
   * @details Only the weights and layer_key of @a group are read (plus
   * group.records when given). Returns true when the backend holds the
   * experts from now on, so the caller may release (or never fill) the host
   * payloads; false, having built nothing, when it does not serve them.
   */
  virtual bool moe_experts_prebuild(const MoeExpertGroup &group) {
    (void)group;
    return false;
  }

  /**
   * @brief Start reading what the next moe_experts() call of a streamed MoE
   *        layer (group.fsu) will need, so the read overlaps the work queued
   *        before that call (the router). Only group.fsu, group.layer_key,
   *        the weights and group.tokens are read. Returns false when nothing
   *        was started (the call then reads what it needs itself).
   */
  virtual bool moe_experts_prefetch(const MoeExpertGroup &group) {
    (void)group;
    return false;
  }

  /**
   * @brief Wait until nothing of an earlier moe_experts() call of the layer
   *        group.layer_key still runs off the calling thread (a streamed
   *        decode step makes its experts resident on reader threads, which
   *        call the layer's acquire / filled / refresh hooks), and rethrow
   *        what failed there. A layer calls it before it touches the state
   *        those hooks share (its residency policy). Only group.layer_key
   *        and group.tokens are read: a call of more than one row (a
   *        prefill, which may change the policy and write the slots itself)
   *        also waits until the decode steps of every streamed layer are
   *        done and replayed into the policies (NNTR_G4MOE_FSU_FLAGS).
   */
  virtual void moe_experts_join(const MoeExpertGroup &group) { (void)group; }

  /**
   * @brief A producer may leave the last kernels that write its output
   *        deferred (a streamed MoE decode step: its expert half waits for
   *        reads, and work enqueued in between - the dense MLP - runs
   *        meanwhile; MoeExpertGroup::output_deferred). Enqueue them now if
   *        @a t is such an output: the graph calls this for every input of
   *        a node before the node runs.
   */
  virtual void flush_deferred(const Tensor &t) { (void)t; }

  /**
   * @brief One expert projection: output = input x weight.
   * @details The dispatch point for expert weights. It is fc() unless a
   * backend serves an expert-only weight format here (see
   * registerMoeExpertFcCl on the OpenCL table), so such a format plugs in
   * without touching the MoE layers.
   */
  virtual void fc_expert(Tensor &input, Tensor &weight, Tensor &output) {
    fc(input, weight, output);
  }

  // True after the accelerator completed this row and returned only its
  // token id. CPU logit postprocessing is then unused. This does not
  // consume the id: take_decode_token_id must still return it once.
  virtual bool has_decode_token_id(unsigned pos) const {
    (void)pos;
    return false;
  }

protected:
  /**
   * @brief Helper used by default impls to throw a uniform "not
   *        implemented" runtime_error tagged with the op name.
   */
  [[noreturn]] static void throwNotImplemented(const char *op);
};

/**
 * @brief Global compute ops pointer.
 *
 * Set once during init_backend(). When a Context-specific ops table is
 * available (via ContextData), that takes precedence.
 */
extern ComputeOps *g_compute_ops;

/**
 * @brief Ensure the global compute ops is initialized.
 */
void ensureComputeOps();

/**
 * @brief Is any producer's output deferred (ComputeOps::flush_deferred)?
 *        The graph asks the inputs' tables to flush only while it is set.
 */
bool deferredOutputsPending();

/** @brief set by a backend when it defers / has flushed every output */
void setDeferredOutputsPending(bool pending);

/**
 * @brief Get the active compute ops with lazy initialization.
 */
inline ComputeOps *getComputeOps() {
#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_expect(g_compute_ops == nullptr, 0))
#else
  if (g_compute_ops == nullptr)
#endif
    ensureComputeOps();
  return g_compute_ops;
}

/**
 * @brief Initialize the CPU compute backend.
 *
 * Sets up architecture-specific resources (e.g., GGML, OpenBLAS threads)
 * and assigns g_compute_ops to the matching concrete ComputeOps
 * subclass for the current CPU architecture.
 */
void init_backend();

/**
 * @brief Backend-specific compute ops getters.
 *
 * `get_cpu_ops()` returns a process-wide singleton of the unified
 * `CpuComputeOps` subclass. The same singleton works for ARM / x86 /
 * fallback because each arch's compute_backend.cpp provides its own
 * specialised body for `nntrainer::sgemm` etc.; the wrapper class is
 * arch-agnostic and only needs to be defined once.
 */
ComputeOps *get_cpu_ops();
#ifdef ENABLE_OPENCL
/** @brief OpenCL accelerator ComputeOps singleton. Defined when
 *  enable-opencl is on, in cl_operations/cl_compute_ops.cpp. */
ComputeOps *get_cl_ops();
#endif
#ifdef ENABLE_CUDA
/** @brief CUDA accelerator ComputeOps singleton. Defined when enable-cuda is
 *  on, in cuda/cuda_compute_ops.cpp. */
ComputeOps *get_cuda_ops();
#endif
#ifdef ENABLE_HEXKL
/** @brief HTP (Hexagon/HMX) accelerator ComputeOps singleton. Defined
 *  when enable-htp is on, in htp_backend/htp_compute_ops.cpp. */
ComputeOps *get_htp_ops();
#endif

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __COMPUTE_OPS_H__ */

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
  virtual bool supports_gemm_q4_0_accel_fp32() const { return false; }
  virtual void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata,
                                    float *matCdata, unsigned int M,
                                    unsigned int N, unsigned int K);

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

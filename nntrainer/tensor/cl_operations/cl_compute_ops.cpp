// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   cl_compute_ops.cpp
 * @date   25 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  OpenCL ComputeOps subclass — provides accelerated quantized
 *         GEMM/GEMV variants on top of the existing nntrainer
 *         OpenCL kernels in cl_operations/blas_kernels.cpp.
 *
 * Two families live here. The accelerator-specific ops (Q4_0 batch / accel,
 * INT4 batch / accel) are overridden with their supports_*() predicates
 * returning true; callers rely on supports_*() to decide whether to use this
 * path or fall back to a CPU ops table — exactly the contract
 * float_tensor.cpp's dispatch sites already follow.
 *
 * The second family is the whole-op table a backend-neutral layer calls
 * without asking a predicate first: fc, apply_activation, layer_norm,
 * activation, residual_op / residual_op2, geglu, swiglu, sigmoid_glu,
 * sigmoid_add, mean_rows, l2_normalize_rows, rotary_embedding and the scopy
 * family. These have no supports_*() escape hatch, so every op a layer
 * registered on the gpu engine can reach has to resolve here — the base
 * ComputeOps default throws, and a layer has nowhere to catch that. The list
 * follows the layers this context registers, among them FullyConnectedLayerCl
 * (fc, apply_activation, fc_prebuild_weight), AdditionLayer (residual_op),
 * SwiGLULayer (swiglu), GeGLULayer (geglu), SigmoidGluLayer / SigmoidAddLayer,
 * LayerNormalizationLayer (layer_norm), ActivationLayer (activation), plus
 * Tensor::copy (scopy) and CausalLM's MHACoreLayer (rotary_embedding).
 * ActivationLayer does not throw either: a mode with no
 * OpenCL kernel runs on the host table, and supports_activation() reports
 * which modes are accelerated.
 *
 * This file is what unblocks GPU dispatch end-to-end:
 *   ClContext (Engine-registered) -> ContextData -> ClComputeOps
 *   -> nntrainer::gemm_q4_0_async_cl(...) -> OpenCL kernel queue.
 */

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <attention_kernels.h>     // gpu_copy_f16_cl
#include <blas_kernel_interface.h> // add_i_cl, dotCl, dotCl_v8c
#include <blas_kernels.h>
#include <common_properties.h> // ActivationType, the act_type int encoding
#include <compute_ops.h>
#include <embedding_pool_cl_op.h>
#include <geglu_cl_op.h>
#include <gelu_cl_op.h>
#include <layernorm_cl_op.h>
#include <qs4cx_tensor.h> // refuseIfQs4cxPayloadDropped
#include <sigmoid_add_cl_op.h>
#include <sigmoid_glu_cl_op.h>
#include <swiglu_cl_op.h>
#include <tensor.h>

namespace nntrainer {

namespace {

/**
 * @brief Whether a Q4_0 tensor in this build has the q4_0x8 layout the
 *        device Q4_0 GEMM unpacks. The CPU backend repacks Q4_0 per ISA:
 *        q4_0x4 on ARM, q4_0x8 on x86 and in the generic fallback.
 */
constexpr bool q4_0_device_layout() {
#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64)
  return false;
#else
  return true;
#endif
}

/**
 * @brief Holds an operand of a host computation on the gpu engine.
 * @details The operands of the host arms here (an FC dot the device does not
 * cover, the FP32 residual add) usually live in shared virtual memory, and the
 * ones a kernel produced may still be in the queue: a projection or a norm
 * maps its output back for the host without waiting. While a kernel may be in
 * flight, the operand is mapped first -- the map blocks until the queue has
 * run up to it -- and released when the host is done. An output the host
 * writes is left mapped for the host afterwards, as a kernel maps its output
 * back, so the device op that consumes it finds it mapped and releases it.
 * Operands in host memory are left alone.
 */
class HostOperand {
public:
  /**
   * @param t operand
   * @param host_writes true for an output the host writes
   */
  HostOperand(const Tensor &t, bool host_writes) {
    const auto md = t.getMemoryData();
    if (!md || !md->isSVM() || t.isClMem())
      return;
    ptr_ = t.getData();
    bytes_ = t.bytes();
    host_writes_ = host_writes;
    // A kernel still in the queue may write this region (its producer) or
    // read it (a pool region the planner gave an earlier tensor too): wait
    // for the queue before the host touches it.
    if (cl_queue_may_be_busy()) {
      cl_svm_map_force(ptr_, bytes_, /*read_only=*/!host_writes);
      mapped_ = true;
    }
    if (host_writes)
      cl_svm_host_write_begin(ptr_, bytes_);
  }
  HostOperand(const HostOperand &) = delete;
  HostOperand &operator=(const HostOperand &) = delete;
  ~HostOperand() {
    if (mapped_)
      cl_svm_unmap_force(ptr_);
    if (host_writes_)
      cl_svm_host_write_end(ptr_, bytes_);
  }

private:
  void *ptr_ = nullptr;
  size_t bytes_ = 0;
  bool mapped_ = false;
  bool host_writes_ = false;
};

} // namespace

/**
 * @brief OpenCL ComputeOps table: the accelerator-backed subset of the
 *        ComputeOps interface, dispatched through ClContext's ContextData.
 */
class ClComputeOps : public ComputeOps {
public:
  // ── Accelerator-only Q4_0 / INT4 GEMM/GEMV ────────────────
  // The device Q4_0 GEMM unpacks the weight from the q4_0x8 interleave
  // (unpack_q4_0x8_transpose16). That is the layout a Q4_0 tensor has in an
  // x86 (and generic) build; an ARM build stores q4_0x4, which the unpack
  // misreads into non-finite outputs. Leave Q4_0 to the CPU table there.
  bool supports_gemm_q4_0_batch_fp32() const override {
    return q4_0_device_layout();
  }
  void gemm_q4_0_batch_fp32(std::vector<void *> matAdata, float *matBdata,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> N,
                            unsigned int K) override {
    nntrainer::gemm_q4_0_async_cl(matAdata, matBdata, matCdata, M, N, K);
  }

  bool supports_gemm_q4_0_accel_fp32() const override {
    return q4_0_device_layout();
  }
  void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata, float *matCdata,
                            unsigned int M, unsigned int N,
                            unsigned int K) override {
    nntrainer::gemm_q4_0_cl(matAdata, matBdata, matCdata, M, N, K);
  }

  bool supports_gemv_int4_batch_fp32() const override { return true; }
  void gemv_int4_batch_fp32(std::vector<void *> weights,
                            std::vector<uint16_t *> scales, float *input,
                            std::vector<float *> outputs, unsigned int K,
                            std::vector<unsigned int> Ns,
                            unsigned int group_size) override {
    nntrainer::gemv_int4_async_cl(weights, scales, input, outputs, K, Ns,
                                  group_size);
  }

  bool supports_gemm_int4_batch_fp32() const override { return true; }
  void gemm_int4_batch_fp32(float *input, std::vector<void *> weights,
                            std::vector<uint16_t *> scales,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> Ns, unsigned int K,
                            unsigned int group_size) override {
    nntrainer::gemm_int4_async_cl(input, weights, scales, matCdata, M, Ns, K,
                                  group_size);
  }

  bool supports_gemv_int4_accel_fp32() const override { return true; }
  void gemv_int4_accel_fp32(char *weight, uint16_t *scale, float *input,
                            float *output, unsigned int K, unsigned int N,
                            unsigned int group_size) override {
    nntrainer::gemv_int4_cl(weight, scale, input, output, K, N, group_size);
  }

  bool supports_sgemm_int4_accel_fp32() const override { return true; }
  void sgemm_int4_accel_fp32(float *input, char *weight, uint16_t *scale,
                             float *output, unsigned int M, unsigned int N,
                             unsigned int K, unsigned int group_size) override {
    nntrainer::sgemm_int4_cl(input, weight, scale, output, M, N, K, group_size);
  }

  // ── Whole-ops (Tensor level) ──────────────────────────────────
  // Fully-connected matmul, for the neutral fully-connected layer: the layer
  // keeps the weight and bias binding and the shape logic, and only the GEMM
  // comes here. dotCl picks the dot, GEMV or GEMM kernel from the operand
  // shapes, exactly as the layer used to do inline.
  //
  // dotCl is contracted to PRODUCE its output, not accumulate into it, which
  // is why no zero-fill precedes this call. The GEMV and GEMM shapes store,
  // in both the FP32 and the FP16 branch (the FP32 general GEMM goes to
  // sgemm_cl). One shape does not honour it: the 1 x 1 dot (M == 1 and
  // N == 1) adds dot_cl's result to the value already in the destination.
  // Do not add a zero-fill here to paper over it: that would cost a
  // full-output memset on every forward for one scalar shape.
  //
  // A quantized weight goes to the v8c int8xint4 GEMM first. dotCl_v8c
  // returns false rather than throwing when it declines the call (the weight
  // is not int4, the shape is outside what the kernel covers, or the path is
  // switched off), so the fallbacks below stay reachable: a quantized weight
  // the kernel declined is dotted on the host, and everything else keeps the
  // dotCl route this layer has always used. Neither fallback accumulates
  // either -- Tensor::dot takes beta = 0 by default and every quantized
  // FloatTensor::dot kernel stores -- so the no-zero-fill contract above holds
  // unchanged for all three routes out of this function.
  void fc(Tensor &input, Tensor &weight, Tensor &output) override {
    if (nntrainer::dotCl_v8c(input, weight, output))
      return;

    switch (weight.getDataType()) {
    case ml::train::TensorDim::DataType::QS4CX: {
      // This is the fallback boundary the plain-payload release is guarded
      // at: once the v8c repack exists, NNTR_V8C_DROP_PLAIN may have handed
      // the nibbles' pages back, and the host dot below would then multiply
      // zeros and return a well-formed wrong output with rc = 0. Refuse
      // loudly instead. Free when nothing was ever dropped (one atomic load).
      refuseIfQs4cxPayloadDropped(weight.getData<uint8_t>(),
                                  "ClComputeOps::fc host fallback");
      HostOperand host_in(input, /*host_writes=*/false);
      HostOperand host_out(output, /*host_writes=*/true);
      input.dot(weight, output, false, false);
      break;
    }
    case ml::train::TensorDim::DataType::QINT4:
    case ml::train::TensorDim::DataType::Q4_0:
    case ml::train::TensorDim::DataType::Q4_K:
    case ml::train::TensorDim::DataType::Q6_K: {
      // A quantized weight has no GPU kernel once v8c declines it: dispatch
      // the host dot, which knows every quantization the CPU backend does.
      HostOperand host_in(input, /*host_writes=*/false);
      HostOperand host_out(output, /*host_writes=*/true);
      input.dot(weight, output, false, false);
      break;
    }
    default:
      nntrainer::dotCl(input, weight, output);
      break;
    }
  }

  // Build the device-side v8c backing for this weight at load time, so the
  // first prefill does not pay the one-time repack. A no-op when the weight
  // is not one the v8c GEMM can take.
  void fc_prebuild_weight(Tensor &weight) override {
    nntrainer::dotCl_v8c_prebuild_weight(weight);
  }

  // Same build, sourcing the nibbles from the loader's file mapping so the
  // plain payload is never copied into the tensor (see the header).
  bool fc_prebuild_weight_from(Tensor &weight, const void *src) override {
    return nntrainer::dotCl_v8c_prebuild_weight_from(weight, src);
  }

  // The fused activation epilogue, which the fully-connected layer applies to
  // its own output in place. It runs on the host table: the operand is the
  // tensor the GEMM just read back, so the CPU table works on exactly the
  // memory the kernels staged and yields the values a standalone
  // ActivationLayer would. Unlike ::activation below, this one accepts every
  // mode, because a fused epilogue has no way to decline one. Replacing it
  // with a kernel is a residency refinement and touches no caller.
  void apply_activation(Tensor &out, int act_type) override {
    get_cpu_ops()->apply_activation(out, act_type);
  }

  // The gated pairs: (gate, up) -> out, element-wise, one kernel each.
  void geglu(const Tensor &in1, const Tensor &in2, Tensor &out,
             unsigned int active_rows, unsigned int row_offset) override {
    nntrainer::geglu_cl_op(in1, in2, out, active_rows, row_offset);
  }
  void swiglu(const Tensor &in1, const Tensor &in2, Tensor &out,
              unsigned int active_rows, unsigned int row_offset) override {
    // FP32 operands the host just produced, with nothing left in the queue
    // (an FP32 model whose projections run on the host): gate them on the
    // host too, where the down projection that reads the result runs.
    const auto fp32 = ml::train::TensorDim::DataType::FP32;
    if (in1.getDataType() == fp32 && in2.getDataType() == fp32 &&
        out.getDataType() == fp32 && !cl_queue_may_be_busy()) {
      HostOperand h1(in1, /*host_writes=*/false);
      HostOperand h2(in2, /*host_writes=*/false);
      HostOperand ho(out, /*host_writes=*/true);
      get_cpu_ops()->swiglu(in1, in2, out, active_rows, row_offset);
      return;
    }
    nntrainer::swiglu_cl_op(in1, in2, out, active_rows, row_offset);
  }
  // The two sigmoid-gated members of the family: sigmoid_glu multiplies the
  // second operand by the gate, sigmoid_add adds it.
  void sigmoid_glu(const Tensor &in1, const Tensor &in2, Tensor &out,
                   unsigned int active_rows, unsigned int row_offset) override {
    nntrainer::sigmoid_glu_cl_op(in1, in2, out, active_rows, row_offset);
  }
  void sigmoid_add(const Tensor &in1, const Tensor &in2, Tensor &out,
                   unsigned int active_rows, unsigned int row_offset) override {
    nntrainer::sigmoid_add_cl_op(in1, in2, out, active_rows, row_offset);
  }

  // LayerNorm over the last axis. The neutral LayerNormalizationLayer owns
  // the axis contract and only dispatches here when its property matches, so
  // this op never sees a property.
  void layer_norm(const Tensor &in, Tensor &out, const Tensor &gamma,
                  const Tensor &beta, float epsilon, unsigned int active_rows,
                  unsigned int row_offset) override {
    nntrainer::layernorm_cl_op(in, out, gamma, beta, epsilon, active_rows,
                               row_offset);
  }

  // Element-wise activation. Only gelu and tanh_gelu have OpenCL kernels; the
  // remaining modes run on the host table.
  //
  // That is the same answer apply_activation() above gives, and it has to be:
  // both are handed the same tensors on the same context, so they cannot hold
  // different beliefs about whether a host loop is valid on them. This
  // function used to throw instead, arguing that a tensor here may live in
  // device memory the host has unmapped -- but a device-resident (GPU_CLMEM)
  // tensor still keeps its slice of the shared plane unless the application
  // sets ResidencyPolicy::skip_shared_slice, which no model in tree does, and
  // while that stays true the throw only produces a worse outcome than the
  // loop. ClContext registers the core ActivationLayer gated
  // on the gelu kernels building, not on the mode, so engine=gpu with
  // activation=relu constructed successfully and then failed at the first
  // forward, mid-inference, while the fused epilogue computed the identical
  // value for the identical tensor.
  //
  // supports_activation() below still reports only the accelerated pair: it
  // answers "is this mode accelerated here", which is what a caller choosing
  // between paths wants to know. This function answers "can this mode be
  // served", and now it always can. If that shared slice is ever dropped,
  // this host branch and apply_activation() change together.
  void activation(const Tensor &in, Tensor &out, int act_type,
                  unsigned int active_rows, unsigned int row_offset) override {
    switch (static_cast<ActivationType>(act_type)) {
    case ActivationType::ACT_GELU:
      nntrainer::gelu_cl_op(in, out, /*mode=*/0, active_rows, row_offset);
      return;
    case ActivationType::ACT_TANH_GELU:
      nntrainer::gelu_cl_op(in, out, /*mode=*/1, active_rows, row_offset);
      return;
    default:
      get_cpu_ops()->activation(in, out, act_type, active_rows, row_offset);
      return;
    }
  }

  bool supports_activation(int act_type) const override {
    const auto type = static_cast<ActivationType>(act_type);
    return type == ActivationType::ACT_GELU ||
           type == ActivationType::ACT_TANH_GELU;
  }

  // One residual-add operand, for the neutral AdditionLayer. FP32 same-size
  // operands take a host copy/add: the FP32 addition kernel reads its result
  // back into the caller's pointer, which is the very read into shared memory
  // that does not land (see the FP32 GEMM read-back), and both operands are
  // host-addressable here anyway.
  // Both operands of the residual join in one kernel. Declines (and the
  // neutral AdditionLayer falls back to the per-operand loop) whenever the
  // three tensors are not all device-plane FP16.
  bool residual_op2(Tensor &hidden, const Tensor &a, const Tensor &b) override {
#ifdef ENABLE_FP16
    return nntrainer::clmem_residual_add2_cl(hidden, a, b);
#else
    (void)hidden;
    (void)a;
    (void)b;
    return false;
#endif
  }

  void residual_op(Tensor &hidden, const Tensor &input,
                   bool accumulate) override {
    const auto fp32 = ml::train::TensorDim::DataType::FP32;
    if (hidden.getDataType() == fp32 && input.getDataType() == fp32 &&
        hidden.size() == input.size()) {
      const size_t n = hidden.size();
      // Both on the shared plane (the activations of an FP32 model) with a
      // producer still in the queue: add on the device, in queue order, so
      // the host does not wait for it. Otherwise the host loop: its operands
      // are complete (or it waits for them), and an add the host runs next to
      // its own projections needs no device round trip.
      const auto hmd = hidden.getMemoryData();
      const auto imd = input.getMemoryData();
      if (hmd && imd && hmd->isSVM() && imd->isSVM() && !hidden.isClMem() &&
          !input.isClMem() && cl_queue_may_be_busy() &&
          nntrainer::residual_f32_svm_cl(
            input.getData<float>(), hidden.getData<float>(),
            static_cast<unsigned int>(n), accumulate))
        return;
      HostOperand host_in(input, /*host_writes=*/false);
      HostOperand host_out(hidden, /*host_writes=*/true);
      float *out = hidden.getData<float>();
      const float *in = input.getData<float>();
      if (!accumulate) {
        std::memcpy(out, in, n * sizeof(float));
      } else {
        for (size_t i = 0; i < n; ++i)
          out[i] += in[i];
      }
      return;
    }

    // Device plane first: when either operand lives there, the shared-plane
    // paths below cannot address it.
#ifdef ENABLE_FP16
    if (nntrainer::clmem_residual_op_cl(hidden, input, accumulate))
      return;
#endif

    if (!accumulate) {
      // First operand: hidden = input. This has to happen ON THE DEVICE when
      // both live on the shared plane, because the accumulate branch below is
      // a GPU kernel. A host copy followed by a device add reads whatever the
      // device last saw in `hidden` -- the host write was never handed back --
      // so the second operand lands on stale bytes and the residual ends up
      // carrying only one of its two inputs. Measured on the 1K cell: the
      // decoder-add output was the embedding alone, with the whole attention
      // contribution missing, and every later layer inherited it.
#ifdef ENABLE_FP16
      const bool svm16 =
        hidden.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        input.getDataType() == ml::train::TensorDim::DataType::FP16 &&
        hidden.getMemoryData() && hidden.getMemoryData()->isSVM() &&
        input.getMemoryData() && input.getMemoryData()->isSVM() &&
        hidden.size() == input.size();
      // NNTR_ADD_DRAIN set (any value) drains after the copy; the in-order
      // queue already sequences the add behind it, so the default is not to.
      static const bool add_drain = std::getenv("NNTR_ADD_DRAIN") != nullptr;
      if (svm16 &&
          nntrainer::gpu_copy_f16_cl(
            reinterpret_cast<const uint16_t *>(input.getData<_FP16>()),
            reinterpret_cast<uint16_t *>(hidden.getData<_FP16>()),
            (unsigned int)hidden.size(), /*svm_inputs=*/true,
            /*in_clmem=*/nullptr, /*out_clmem=*/nullptr, /*drain=*/add_drain))
        return;
#endif
      hidden.copy(input);
    } else {
      nntrainer::add_i_cl(hidden, input);
    }
  }

  // Tensor::add() and Tensor::multiply() reach the table the same way
  // Tensor::copy() does -- straight through, with no supports_*() guard for a
  // caller to consult -- so leaving these off the table is not a missing
  // acceleration but a throw: an addition between two tensors on this context
  // raised "ComputeOps::ele_add_fp32 not implemented by this backend" from the
  // base default. Serve them on the host table, for the same reason
  // residual_op's FP32 path does: the addition kernel reads its result back
  // into the caller's pointer, which is exactly the read into shared memory
  // that does not land, and both operands are host-addressable here anyway.
  // Putting these on a kernel is blocked behind that read-back fix, not behind
  // a missing kernel.
  void ele_add_fp32(const unsigned int N, const float *X, const float *Y,
                    float *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    get_cpu_ops()->ele_add_fp32(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }
  void ele_mul_fp32(const unsigned int N, const float *X, const float *Y,
                    float *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    get_cpu_ops()->ele_mul_fp32(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }
  void ele_sub_fp32(const unsigned int N, const float *X, const float *Y,
                    float *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    get_cpu_ops()->ele_sub_fp32(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }
  void ele_div_fp32(const unsigned int N, const float *X, const float *Y,
                    float *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    get_cpu_ops()->ele_div_fp32(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }

#ifdef ENABLE_FP16
  void ele_add_fp16(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                    _FP16 *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    get_cpu_ops()->ele_add_fp16(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }
  void ele_mul_fp16(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                    _FP16 *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    get_cpu_ops()->ele_mul_fp16(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }
  void ele_sub_fp16(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                    _FP16 *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    get_cpu_ops()->ele_sub_fp16(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }
  void ele_div_fp16(const unsigned int N, const _FP16 *X, const _FP16 *Y,
                    _FP16 *Z, float alpha, float beta, unsigned int i_stride,
                    unsigned int o_stride) override {
    get_cpu_ops()->ele_div_fp16(N, X, Y, Z, alpha, beta, i_stride, o_stride);
  }
#endif

  // Tensor::copy() reaches the table with no supports_*() guard, so without
  // these a copy of a tensor on this context would throw "not implemented" --
  // which the residual copy above does on the FP16 path. A host loop is
  // correct for host pointers and for host-coherent shared memory; moving the
  // copy onto a kernel is a residency refinement, not a correctness one.
  void scopy_fp32(const unsigned int N, const float *X, const unsigned int incX,
                  float *Y, const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }

  // The two row reductions of the sentence-embedding tail. Their callers are
  // backend-neutral layers reaching them through in.getOps(), and before these
  // overrides the same maths arrived as Tensor::average(2) -> sgemv_fp32 and
  // normalization_i(3) -> snrm2_fp32 + sscal_fp32, none of which this table
  // implements: the tail threw on a gpu-context tensor rather than running.
  void mean_rows(const Tensor &in, Tensor &out, unsigned int active_rows,
                 unsigned int row_offset) override {
    mean_rows_cl_op(in, out, active_rows, row_offset);
  }
  void l2_normalize_rows(const Tensor &in, Tensor &out,
                         float epsilon) override {
    l2_normalize_rows_cl_op(in, out, epsilon);
  }
  // Host rotation for now: the device RoPE (rope_inplace_f16_cl) is still
  // selected inside the attention layer, which settles the queue before it
  // reaches the host path this serves. Moving that device arm here is the
  // next step; until then this keeps the host path's behaviour unchanged.
  void rotary_embedding(const Tensor &in, Tensor &out, unsigned int head_dim,
                        const RotaryTable &table, unsigned int start_pos,
                        bool rotate) override {
    get_cpu_ops()->rotary_embedding(in, out, head_dim, table, start_pos,
                                    rotate);
  }

#ifdef ENABLE_FP16
  void scopy_fp16(const unsigned int N, const _FP16 *X, const unsigned int incX,
                  _FP16 *Y, const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = X[i * incX];
  }
  // Mixed precision: an FP32 source feeding an FP16 graph, or an FP16 result
  // read back as FP32, both route here on this backend.
  void scopy_fp32_to_fp16(const unsigned int N, const float *X,
                          const unsigned int incX, _FP16 *Y,
                          const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = static_cast<_FP16>(X[i * incX]);
  }
  void scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, float *Y,
                          const unsigned int incY) override {
    for (unsigned int i = 0; i < N; ++i)
      Y[i * incY] = static_cast<float>(X[i * incX]);
  }
#endif
};

ComputeOps *get_cl_ops() {
  static ClComputeOps instance;
  return &instance;
}

} // namespace nntrainer

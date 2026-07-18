// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   geglu_cl_op.cpp
 * @date   29 June 2026
 * @brief  OpenCL GeGLU whole-op kernel dispatch (gelu_tanh(gate) * up).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Relocated verbatim from geglu_cl.cpp (GeGLULayerCl::gegluProcess /
 * geglu_cl / geglu_cl_fp16 / registerClKernels) into free functions so the
 * GeGLU layer can be a single backend-neutral Layer.
 */

#include "geglu_cl_op.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <blas_kernel_interface.h> // get_or_create_resident_backing (residency)
#include <cl_kernels/geglu.h>
#include <engine.h> // Engine::Global().getRegisteredContext("gpu")
#include <nntrainer_log.h>
#include <tensor.h>
#ifdef ENABLE_FP16
#include <cl_kernels/geglu_fp16.h>
#endif

namespace nntrainer {

namespace {

enum Kernels { GEGLU_CL, GEGLU_CL_FP16 }; /** kernels enum */

std::vector<ClContext::SharedPtrClKernel> &getLayerKernelPtrs() {
  /**< kernel list relevant with this layer */
  static std::vector<ClContext::SharedPtrClKernel> layer_kernel_ptrs;
  return layer_kernel_ptrs;
}

void geglu_cl(float *matAdata, float *vecXdata, float *vecYdata,
              unsigned int dim1, unsigned int dim2, bool svm,
              void *resident_out) {
  (void)resident_out; // FP32 path: residency overlay is FP16-only (no-op here)
  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {
    const auto &kernel_geglu_ptr = getLayerKernelPtrs()[Kernels::GEGLU_CL];
    int dim = int(dim1 * dim2);

    if (!svm) {
      bool write_result = true;

      write_result &= clbuffInstance.getInBufferA()->WriteDataRegion(
        global_cl_context->command_queue_inst_, dim * sizeof(float), matAdata);
      write_result &= clbuffInstance.getInBufferB()->WriteDataRegion(
        global_cl_context->command_queue_inst_, dim * sizeof(float), vecXdata);
      if (!write_result) {
        break;
      }

      auto bufferInA = clbuffInstance.getInBufferA()->GetBuffer();
      auto bufferInB = clbuffInstance.getInBufferB()->GetBuffer();
      auto bufferOutA = clbuffInstance.getOutBufferA()->GetBuffer();

      bool set_result = true;
      set_result &=
        kernel_geglu_ptr->SetKernelArguments(0, &bufferInA, sizeof(cl_mem));
      set_result &=
        kernel_geglu_ptr->SetKernelArguments(1, &bufferInB, sizeof(cl_mem));
      set_result &=
        kernel_geglu_ptr->SetKernelArguments(2, &bufferOutA, sizeof(cl_mem));
      if (!set_result) {
        break;
      }
    } else {
      bool map_result = true;
      map_result &=
        global_cl_context->command_queue_inst_.enqueueSVMUnmap(matAdata);
      map_result &=
        global_cl_context->command_queue_inst_.enqueueSVMUnmap(vecXdata);
      if (!map_result) {
        ml_loge("Failed to map svm");
        break;
      }

      bool set_svm_result = true;
      set_svm_result &= kernel_geglu_ptr->SetKernelSVMArguments(0, matAdata);
      set_svm_result &= kernel_geglu_ptr->SetKernelSVMArguments(1, vecXdata);
      // [resident-act] output direct to a cl_mem backing (device-resident, no
      // SVM intermediate) when the caller passed one; else SVM output.
      if (resident_out != nullptr) {
        cl_mem out_buf = static_cast<cl_mem>(resident_out);
        set_svm_result &=
          kernel_geglu_ptr->SetKernelArguments(2, &out_buf, sizeof(cl_mem));
      } else {
        set_svm_result &= kernel_geglu_ptr->SetKernelSVMArguments(2, vecYdata);
      }
      if (!set_svm_result) {
        ml_loge("Failed to set svm");
        break;
      }
    }

    // NOTE(mwlasiuk) : local size can not be larger than global
    const int32_t desired_local = 64;
    const bool can_use_desired = dim >= desired_local;
    const int32_t chosen_local = can_use_desired ? desired_local : dim;

    const int work_groups_count[3] = {dim, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {chosen_local, 1, 1}; // test-value

    if (!global_cl_context->command_queue_inst_.DispatchCommand(
          kernel_geglu_ptr, work_groups_count, work_group_size)) {
      ml_loge("Failed to run");
      break;
    }

    if (!svm) {
      if (!clbuffInstance.getOutBufferA()->ReadDataRegion(
            global_cl_context->command_queue_inst_, dim * sizeof(float),
            vecYdata)) {
        break;
      }
    } else {
      if (!global_cl_context->command_queue_inst_.enqueueSVMMap(
            vecYdata, dim * sizeof(float), true)) {
        ml_loge("Failed to unmap svm");
        break;
      }
    }

  } while (false);
}

#ifdef ENABLE_FP16
void geglu_cl_fp16(_FP16 *matAdata, _FP16 *vecXdata, _FP16 *vecYdata,
                   unsigned int dim1, unsigned int dim2, bool svm,
                   void *resident_out, bool skip_out_map, void *in1_clmem,
                   void *in2_clmem, unsigned int row_off) {

  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {
    const auto &kernel_geglu_ptr = getLayerKernelPtrs()[Kernels::GEGLU_CL_FP16];

    int dim = int(dim1 * dim2);

    if (!svm) {
      bool write_result = true;
      write_result &= clbuffInstance.getInBufferA()->WriteDataRegion(
        global_cl_context->command_queue_inst_, dim * sizeof(_FP16), matAdata);
      write_result &= clbuffInstance.getInBufferB()->WriteDataRegion(
        global_cl_context->command_queue_inst_, dim * sizeof(_FP16), vecXdata);
      if (!write_result) {
        break;
      }

      auto bufferInA = clbuffInstance.getInBufferA()->GetBuffer();
      auto bufferInB = clbuffInstance.getInBufferB()->GetBuffer();
      auto bufferOutA = clbuffInstance.getOutBufferA()->GetBuffer();

      bool set_result = true;
      set_result &=
        kernel_geglu_ptr->SetKernelArguments(0, &bufferInA, sizeof(cl_mem));
      set_result &=
        kernel_geglu_ptr->SetKernelArguments(1, &bufferInB, sizeof(cl_mem));
      set_result &=
        kernel_geglu_ptr->SetKernelArguments(2, &bufferOutA, sizeof(cl_mem));
      if (!set_result) {
        break;
      }
    } else {
      // Static cl_mem residency: a cl_mem-bound input (the gate/up FC wrote
      // the planner sub-buffer) needs NO SVM unmap -- its SVM shadow was never
      // written and unmapping it would flip SVM state one-sidedly.
      bool map_result = true;
      if (in1_clmem == nullptr)
        map_result &=
          global_cl_context->command_queue_inst_.enqueueSVMUnmap(matAdata);
      if (in2_clmem == nullptr)
        map_result &=
          global_cl_context->command_queue_inst_.enqueueSVMUnmap(vecXdata);
      if (!map_result) {
        ml_loge("Failed to map svm");
        break;
      }

      bool set_svm_result = true;
      if (in1_clmem != nullptr) {
        cl_mem in1_buf = static_cast<cl_mem>(in1_clmem);
        set_svm_result &=
          kernel_geglu_ptr->SetKernelArguments(0, &in1_buf, sizeof(cl_mem));
      } else {
        set_svm_result &= kernel_geglu_ptr->SetKernelSVMArguments(0, matAdata);
      }
      if (in2_clmem != nullptr) {
        cl_mem in2_buf = static_cast<cl_mem>(in2_clmem);
        set_svm_result &=
          kernel_geglu_ptr->SetKernelArguments(1, &in2_buf, sizeof(cl_mem));
      } else {
        set_svm_result &= kernel_geglu_ptr->SetKernelSVMArguments(1, vecXdata);
      }
      // [resident-act / static GPU_CLMEM] output direct to a cl_mem (the
      // tensor's planner sub-buffer): no SVM intermediate; else SVM output.
      if (resident_out != nullptr) {
        cl_mem out_buf = static_cast<cl_mem>(resident_out);
        set_svm_result &=
          kernel_geglu_ptr->SetKernelArguments(2, &out_buf, sizeof(cl_mem));
      } else {
        set_svm_result &= kernel_geglu_ptr->SetKernelSVMArguments(2, vecYdata);
      }
      if (!set_svm_result) {
        ml_loge("Failed to set svm");
        break;
      }
    }

    // Flat-index shift (arg 3): nonzero only for the all-cl_mem live-row decode
    // path, where every buffer is bound whole at index 0 and the kernel reads/
    // writes row (row_off/dim2). SVM/host paths carry the offset on their
    // pointer and pass 0. GWS below is dim1*dim2, so exactly dim1 rows run.
    if (!kernel_geglu_ptr->SetKernelArguments(3, &row_off, sizeof(int))) {
      ml_loge("Failed to set geglu row_off");
      break;
    }

    // NOTE(mwlasiuk) : local size can not be larger than global
    const int32_t desired_local = 64;
    const bool can_use_desired = dim >= desired_local;
    const int32_t chosen_local = can_use_desired ? desired_local : dim;

    const int work_groups_count[3] = {dim, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {chosen_local, 1, 1}; // test-value

    if (!global_cl_context->command_queue_inst_.DispatchCommand(
          kernel_geglu_ptr, work_groups_count, work_group_size)) {
      ml_loge("Failed to run");
      break;
    }
    // NNTR_GEGLU_VERIFY=1 (Xe3): out == gelu(in1) * in2 from the SAME buffers
    // the kernel used. Correct => geglu fine (any garbage is a stale INPUT).
    if (std::getenv("NNTR_GEGLU_VERIFY")) {
      cl_command_queue qq =
        global_cl_context->command_queue_inst_.GetCommandQueue();
      clFinish(qq);
      const size_t cnt = std::min((size_t)dim, (size_t)2048);
      std::vector<uint16_t> a(cnt), b(cnt), o(cnt);
      auto rd = [&](void *clm, _FP16 *svm, std::vector<uint16_t> &v) {
        if (clm)
          clEnqueueReadBuffer(qq, static_cast<cl_mem>(clm), CL_TRUE, 0, cnt * 2,
                              v.data(), 0, nullptr, nullptr);
        else
          std::memcpy(v.data(), svm, cnt * 2);
      };
      rd(in1_clmem, matAdata, a);
      rd(in2_clmem, vecXdata, b);
      rd(resident_out, vecYdata, o);
      auto h2f = [](uint16_t hh) {
        _FP16 t;
        std::memcpy(&t, &hh, 2);
        return (float)t;
      };
      float maxd = 0;
      for (size_t i = 0; i < cnt; ++i) {
        float x = h2f(a[i]);
        float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        float ref = 0.5f * x * (1.0f + std::tanh(inner)) * h2f(b[i]);
        maxd = std::max(maxd, std::fabs(h2f(o[i]) - ref));
      }
      std::fprintf(stderr, "[GEGLUVERIFY] dim1=%u dim2=%u maxdiff=%.4f\n", dim1,
                   dim2, maxd);
      std::fflush(stderr);
    }

    if (!svm) {
      if (!clbuffInstance.getOutBufferA()->ReadDataRegion(
            global_cl_context->command_queue_inst_, dim * sizeof(_FP16),
            vecYdata)) {
        break;
      }
    } else if (resident_out == nullptr && !skip_out_map) {
      // async: geglu output is always consumed by the next GPU op (ffn_down
      // FC); in-order queue preserves ordering, no host read between.
      if (!global_cl_context->command_queue_inst_.enqueueSVMMap(
            vecYdata, dim * sizeof(_FP16), true, /** async */ true)) {
        ml_loge("Failed to unmap svm");
        break;
      }
    }
    // NNTR_DEVRES Step 4: skip_out_map leaves vecYdata GPU-owned (unmapped) for
    // the device-resident ffn_down FC, which skips its matching unmap (the FC's
    // v8c_copy_svm_to_clmem) — the map/unmap PAIR is removed together (a one-
    // sided skip would crash on asymmetric SVM state). Coordinated via the
    // device_valid bit the producer set below.
    // resident_out: output is a cl_mem backing — no SVM map (consumer reads
    // it as cl_mem via the residency overlay).

  } while (false);
}
#endif

} // namespace

bool registerGeGLUClKernels(ClContext &cl_context) {
  auto &layer_kernel_ptrs = getLayerKernelPtrs();

  // check if the kernels are already registered.
  if (!layer_kernel_ptrs.empty()) {
    ml_loge("kernels for geglu_cl are already registered.");
    return false;
  }

  do {
    ClContext::SharedPtrClKernel kernel_geglu_ptr = nullptr;

    kernel_geglu_ptr = cl_context.registerClKernel(geglu_kernel, "geglu_cl");

    if (!kernel_geglu_ptr) {
      ml_loge("OpenCL Error: Fail to register geglu_cl kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_geglu_ptr);

#ifdef ENABLE_FP16
    kernel_geglu_ptr =
      cl_context.registerClKernel(geglu_fp16_kernel, "geglu_cl_fp16");

    if (!kernel_geglu_ptr) {
      ml_loge("OpenCL Error: Fail to register geglu_cl_fp16 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_geglu_ptr);
#endif

    return true;
  } while (false);

  // clear all registered kernels if any error occurs during registration
  layer_kernel_ptrs.clear();

  return false;
}

void geglu_cl_op(const Tensor &in1, const Tensor &in2, Tensor &result,
                 unsigned int active_rows, unsigned int row_offset) {

  unsigned int dim1, dim2;
  dim1 = active_rows;
  dim2 = in1.width();

  // Element offset into the SVM pointers so the kernel processes rows
  // [row_offset, row_offset+active_rows). row_offset must be 0 for the cl_mem
  // path (a whole-buffer cl_mem carries no pointer offset -- the
  // caller passes row_offset=0 + active_rows=to in that case).
  const size_t elem_off = (size_t)row_offset * dim2;

  // SVM-direct only when the tensors are GPU-resident (SVM pool). On the
  // default host (cpu) pool getData() returns host pointers, so use the
  // host-bounce path; passing host pointers as SVM kernel arguments produces
  // garbage. When the graph opts into the SVM pool (NNTR_GPU_SVM_POOL), this
  // becomes a zero-copy SVM-direct dispatch (residency).
  const auto md = in1.getMemoryData();
  const bool use_svm = md && md->isSVM();

  if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data1 = in1.getData() + elem_off;
    float *data2 = in2.getData() + elem_off;
    float *rdata = result.getData() + elem_off;
    geglu_cl(data1, data2, rdata, dim1, dim2, use_svm,
             /** resident_out */ nullptr);
  } else if (in1.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *data1 = in1.getData<_FP16>() + elem_off;
    _FP16 *data2 = in2.getData<_FP16>() + elem_off;
    _FP16 *rdata = result.getData<_FP16>() + elem_off;
    // Planner-decided STATIC residency: each of in1/in2/out binds the plane
    // its ResidencyClass picked at allocation -- the gate/up FC outputs and
    // the geglu output are GPU_CLMEM on the live path, so the kernel reads
    // and writes the planner cl_mem sub-buffers directly (no SVM map at all
    // on this op). Mixed cl_mem/SVM args are valid for partial classification.
    void *in1_cl = (use_svm && in1.isClMem()) ? in1.getClMem() : nullptr;
    void *in2_cl = (use_svm && in2.isClMem()) ? in2.getClMem() : nullptr;
    void *out_cl = (use_svm && result.isClMem()) ? result.getClMem() : nullptr;
    // Legacy runtime overlay (env-off by default): only when the static class
    // did not already place the output in cl_mem.
    void *res_backing = out_cl;
    if (res_backing == nullptr) {
      static const bool resident_act =
        std::getenv("NNTR_RESIDENT_ACT") != nullptr;
      const auto rmd = result.getMemoryData();
      if (resident_act && use_svm && rmd && rmd->isSVM())
        res_backing =
          static_cast<void *>(nntrainer::get_or_create_resident_backing(
            result.getName(), (unsigned int)result.size(), /** fp16 */ true));
    }
    static const bool devres_geglu = std::getenv("NNTR_DEVRES") != nullptr;
    const bool resident_edge =
      devres_geglu && use_svm && res_backing == nullptr;
    // When every buffer is bound as a whole cl_mem (index 0, no pointer
    // offset), the kernel applies elem_off itself so a single live row at
    // row_offset is processed in place. SVM/host args carry the offset on
    // their pointer, so row_off stays 0 there.
    const unsigned int kern_row_off =
      (in1_cl && in2_cl && res_backing) ? (unsigned int)elem_off : 0u;
    geglu_cl_fp16(data1, data2, rdata, dim1, dim2, use_svm, res_backing,
                  /** skip_out_map */ resident_edge, in1_cl, in2_cl,
                  kern_row_off);
    if (devres_geglu)
      if (auto rmd = result.getMemoryData())
        rmd->setDeviceValid(
          resident_edge, resident_edge ? result.getData<uint8_t>() : nullptr);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

} // namespace nntrainer

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Thummala Pallavi <t.pallavi@samsung.com>
 *
 * @file        rmsnorm_layer_cl.cpp
 * @date        8 June 2024
 * @brief       This is RMSNorm Layer Class for Neural Network with
 * OpenCl implementation
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Thummala Pallavi <t.pallavi@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

#include <blas_kernels.h>
#include <cl_kernels/rmsnorm.h>
#ifdef ENABLE_FP16
#include <cl_kernels/rmsnorm_fp16.h>
#endif
#include <common_properties.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <rmsnorm_layer_cl.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum RMSParams { gamma };

RMSNormLayerCl::RMSNormLayerCl() : LayerImplCl() { wt_idx.fill(0); }

bool RMSNormLayerCl::registerClKernels(ClContext &cl_context) {
  auto &layer_kernel_ptrs = getLayerKernelPtrs();

  // check if already registered
  if (!layer_kernel_ptrs.empty()) {
    ml_loge("kernels for reshape layer are already registered");
    return false;
  }

  do {

    ClContext::SharedPtrClKernel kernel_rmsnorm_ptr =
      cl_context.registerClKernel(rmsnorm_kernel, "rmsnorm_cl");
    if (!kernel_rmsnorm_ptr) {
      ml_loge("OpenCL Error: Fail to register rmsnorm_cl kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_rmsnorm_ptr);

#ifdef ENABLE_FP16
    ClContext::SharedPtrClKernel kernel_rmsnorm_fp16_ptr =
      cl_context.registerClKernel(rmsnorm_fp16_kernel, "rmsnorm_cl_fp16");
    if (!kernel_rmsnorm_fp16_ptr) {
      ml_loge("OpenCL Error: Fail to register rmsnorm_cl_fp16 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_rmsnorm_fp16_ptr);
#endif

    return true;

  } while (false);

  // clear all registered kernels if any error occurs during registration
  // layer_kernel_ptrs.clear();

  return false;
}

void RMSNormLayerCl::finalize(InitLayerContext &context) {
  std::vector<TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  auto &rmsparams_gamma = std::get<props::GammaInitializer>(rmsnorm_props);

  TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()));
  wt_idx[RMSParams::gamma] =
    context.requestWeight(gamma_dim, rmsparams_gamma, WeightRegularizer::NONE,
                          1.0f, 0.0f, "gamma", false);
}

void RMSNormLayerCl::forwarding(RunLayerContext &context, bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  auto &epsilon = std::get<props::Epsilon>(rmsnorm_props).get();
  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
    rmsnormProcess(in, out, gamma, epsilon);
  } else {
#ifdef ENABLE_FP16
    rmsnormProcess_fp16(in, out, gamma, epsilon);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void RMSNormLayerCl::rmsnormProcess(Tensor const &input, Tensor &result,
                                    Tensor const &gamma, const float epsilon) {
  // Bind device memory directly (SVM-direct) only when the tensors are
  // GPU-resident, i.e. allocated from the SVM pool. On the default host (cpu)
  // pool getData() returns HOST pointers, and passing those as SVM kernel
  // arguments produces garbage -- so take the host-bounce path there.
  // rmsnorm_cl's use_svm defaults to true, so omitting it is not neutral.
  const auto md = input.getMemoryData();
  const bool use_svm = md && md->isSVM();
  rmsnorm_cl(input.getData<float>(), gamma.getData<float>(),
             result.getData<float>(), epsilon,
             input.batch() * input.channel() * input.height(), input.width(),
             use_svm);
}

#ifdef ENABLE_FP16
void RMSNormLayerCl::rmsnormProcess_fp16(Tensor const &input, Tensor &result,
                                         Tensor const &gamma,
                                         const float epsilon) {

  bool ret = false;
  int dim1 = input.batch() * input.height() * input.width() * input.channel();
  CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(), input.height(),
                       input.width(), input.getTensorType());
  int b = input.batch();
  int c = input.channel();
  int h = input.height();
  int w = input.width();

  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {
    auto kernel_rmsnorm_ptr = getLayerKernelPtrs()[Kernels::RMSNORM_CL_FP16];

    const _FP16 *data = input.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();
    const _FP16 *gdata = gamma.getData<_FP16>();

    ret = clbuffInstance.getInBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_, dim1 * sizeof(cl_half), data);
    if (!ret) {
      break;
    }

    ret = clbuffInstance.getInBufferB()->WriteDataRegion(
      global_cl_context->command_queue_inst_, input.width() * sizeof(cl_half),
      gdata);
    if (!ret) {
      break;
    }

    // SetKernelArguments takes a POINTER to the value (clSetKernelArg
    // semantics), so a cl_mem arg must be passed by address. Passing
    // GetBuffer() (a cl_mem) directly bound a garbage handle =>
    // CL_INVALID_MEM_OBJECT, which silently broke this dispatch and left the
    // output stale.
    ret = kernel_rmsnorm_ptr->SetKernelArguments(
      0, &clbuffInstance.getInBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!ret) {
      break;
    }

    ret = kernel_rmsnorm_ptr->SetKernelArguments(
      1, &clbuffInstance.getOutBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!ret) {
      break;
    }

    ret = kernel_rmsnorm_ptr->SetKernelArguments(
      2, &clbuffInstance.getInBufferB()->GetBuffer(), sizeof(cl_mem));
    if (!ret) {
      break;
    }
    ret = kernel_rmsnorm_ptr->SetKernelArguments(4, &b, sizeof(int));

    if (!ret) {
      break;
    }

    // epsilon is a float; the kernel arg is half, so convert it. Passing the
    // low 2 bytes of the float as a half gave a wrong epsilon.
    const _FP16 epsilon_h = static_cast<_FP16>(epsilon);
    ret =
      kernel_rmsnorm_ptr->SetKernelArguments(3, &epsilon_h, sizeof(cl_half));
    if (!ret) {
      break;
    }

    ret = kernel_rmsnorm_ptr->SetKernelArguments(5, &c, sizeof(int));
    if (!ret) {
      break;
    }

    ret = kernel_rmsnorm_ptr->SetKernelArguments(6, &h, sizeof(int));
    if (!ret) {
      break;
    }
    ret = kernel_rmsnorm_ptr->SetKernelArguments(7, &w, sizeof(int));
    if (!ret) {
      break;
    }
    const int work_groups_count[3] = {b * c, h, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {w, 1, 1}; // test-value

    ret = global_cl_context->command_queue_inst_.DispatchCommand(
      kernel_rmsnorm_ptr, work_groups_count, work_group_size);
    if (!ret) {
      break;
    }

    ret = clbuffInstance.getOutBufferA()->ReadDataRegion(
      global_cl_context->command_queue_inst_, dim1 * sizeof(cl_half), rdata);
    if (!ret) {
      break;
    }
  } while (false);
}
#endif

void RMSNormLayerCl::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  in_step_dim.height(to - from);
  out_step_dim.height(to - from);

  Tensor in_step = in.getSharedDataTensor(in_step_dim, 0, true);
  Tensor out_step = out.getSharedDataTensor(out_step_dim, 0, true);

  auto &epsilon = std::get<props::Epsilon>(rmsnorm_props).get();

  if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
    rmsnormProcess(in, out, gamma, epsilon);
  } else {
#ifdef ENABLE_FP16
    rmsnormProcess_fp16(in, out, gamma, epsilon);
#else
    throw std::runtime_error("enable-fp16 is not enabled");
#endif
  }
}

void RMSNormLayerCl::calcDerivative(RunLayerContext &context) {
  ml_logi("Training not supported");
}

void RMSNormLayerCl::calcGradient(RunLayerContext &context) {
  ml_logi("Training not supported");
}

void RMSNormLayerCl::exportTo(Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(rmsnorm_props, method, this);
}

void RMSNormLayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, rmsnorm_props);
  LayerImpl::setProperty(remain_props);
}

std::vector<ClContext::SharedPtrClKernel> &
RMSNormLayerCl::getLayerKernelPtrs() {
  /**< kernel list relevant with this layer */
  static std::vector<ClContext::SharedPtrClKernel> layer_kernel_ptrs;
  return layer_kernel_ptrs;
}

} // namespace nntrainer

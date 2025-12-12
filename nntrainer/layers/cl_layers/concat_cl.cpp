// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024  Niket Agarwal <niket.a@samsung.com>
 *
 * @file   concat_cl.cpp
 * @date   2 July 2024
 * @brief  Implementation of Concat Layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cstring>
#include <iostream>
#include <vector>

#include <cl_kernels/concat_axis_1.h>
#include <cl_kernels/concat_axis_2.h>
#include <cl_kernels/concat_axis_3.h>
#ifdef ENABLE_FP16
#include <cl_kernels/concat_axis_1_fp16.h>
#include <cl_kernels/concat_axis_2_fp16.h>
#include <cl_kernels/concat_axis_3_fp16.h>
#endif
#include <concat_cl.h>
#include <layer_context.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor_dim.h>
#include <util_func.h>

namespace nntrainer {
ConcatLayerCl::ConcatLayerCl() : LayerImplCl() {}

static constexpr size_t SINGLE_INOUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

bool ConcatLayerCl::registerClKernels(ClContext &cl_context) {
  auto &layer_kernel_ptrs = getLayerKernelPtrs();

  // check if already registered
  if (!layer_kernel_ptrs.empty()) {
    ml_loge("kernels for concat layer are already registered");
    return false;
  }

  do {

    ClContext::SharedPtrClKernel kernel_concat_ptr = nullptr;

    kernel_concat_ptr =
      cl_context.registerClKernel(concat_axis_1_kernel, "concat_cl_axis1");
    if (!kernel_concat_ptr) {
      ml_loge("OpenCL Error: Fail to register concat_cl_axis1 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_concat_ptr);

    kernel_concat_ptr =
      cl_context.registerClKernel(concat_axis_2_kernel, "concat_cl_axis2");
    if (!kernel_concat_ptr) {
      ml_loge("OpenCL Error: Fail to register concat_cl_axis2 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_concat_ptr);

    kernel_concat_ptr =
      cl_context.registerClKernel(concat_axis_3_kernel, "concat_cl_axis3");
    if (!kernel_concat_ptr) {
      ml_loge("OpenCL Error: Fail to register concat_cl_axis3 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_concat_ptr);

#ifdef ENABLE_FP16
    kernel_concat_ptr = cl_context.registerClKernel(concat_axis_1_fp16_kernel,
                                                    "concat_cl_axis1_fp16");
    if (!kernel_concat_ptr) {
      ml_loge("OpenCL Error: Fail to register concat_cl_axis1_fp16 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_concat_ptr);

    kernel_concat_ptr = cl_context.registerClKernel(concat_axis_2_fp16_kernel,
                                                    "concat_cl_axis2_fp16");
    if (!kernel_concat_ptr) {
      ml_loge("OpenCL Error: Fail to register concat_cl_axis2_fp16 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_concat_ptr);

    kernel_concat_ptr = cl_context.registerClKernel(concat_axis_3_fp16_kernel,
                                                    "concat_cl_axis3_fp16");
    if (!kernel_concat_ptr) {
      ml_loge("OpenCL Error: Fail to register concat_cl_axis3_fp16 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_concat_ptr);
#endif

    return true;

  } while (false);

  // clear all registered kernels if any error occurs during registration
  layer_kernel_ptrs.clear();

  return false;
}

void ConcatLayerCl::finalize(InitLayerContext &context) {
  auto &concat_dimension_prop = std::get<props::ConcatDimension>(concat_props);
  /** for backward compatibility, default concat dimension will be channel */
  /// @todo this is hacky way to force concat dimension to width if channel
  /// dimension is taken, this is because recurrent realizer, return sequence
  /// exploits concat layer but have no control over where to stack/axis
  unsigned int concat_dimension =
    context.getInputDimensions().front().channel() > 1 ? 3 : 1;
  if (!concat_dimension_prop.empty())
    concat_dimension = concat_dimension_prop.get();

  /**
   * The concat is only done along the axis dimension.
   * For example, consider 2 inputs a, b with dimensions [b,c,h,w] each
   * 1. concat_dimension = 1, output_dim = [b,c_a+c_b,h,w]
   * 2. concat_dimension = 2, output_dim = [b,c,h_a+h_b,w]
   * 3. concat_dimension = 3, output_dim = [b,c,h,w_a+w_b]
   */
  auto const &input_dims = context.getInputDimensions();
  const TensorDim &input_dim_0 = input_dims[SINGLE_INOUT_IDX];
  unsigned int concat_dim_val = input_dim_0.getTensorDim(concat_dimension);

  for (unsigned int idx = 1; idx < input_dims.size(); ++idx) {
    const TensorDim &dim = input_dims[idx];

    for (unsigned int i = 0; i < ml::train::TensorDim::getNumDim(); ++i) {
      if (i == concat_dimension)
        continue;
      NNTR_THROW_IF(input_dim_0[i] != dim[i], std::runtime_error)
        << "Error: concat layer requires same shape from all input layers "
           "along non-concat dimension";
    }
    concat_dim_val += dim[concat_dimension];
  }

  TensorDim output_dim = input_dim_0;
  output_dim.setTensorDim(concat_dimension, concat_dim_val);

  context.setOutputDimensions({output_dim});
}

void ConcatLayerCl::forwarding(RunLayerContext &context, bool training) {
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  const Tensor &in1 = context.getInput(INPUT_IDX_1);
  const Tensor &in2 = context.getInput(INPUT_IDX_2);
  ConcatProcess(in1, in2, out);
}

void ConcatLayerCl::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  const Tensor &in1 = context.getInput(INPUT_IDX_1);
  const Tensor &in2 = context.getInput(INPUT_IDX_2);
  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }
  ConcatProcess(in1, in2, out);
}

void ConcatLayerCl::ConcatProcess(Tensor const &input1, Tensor const &input2,
                                  Tensor &result) {
  auto dim1 = input1.getDim();
  auto dim2 = input2.getDim();

  unsigned int input1_batch_size = dim1.batch();
  unsigned int input1_height = dim1.height();
  unsigned int input1_channels = dim1.channel();
  unsigned int input1_width = dim1.width();
  unsigned int input2_batch_size = dim2.batch();
  unsigned int input2_height = dim2.height();
  unsigned int input2_channels = dim2.channel();
  unsigned int input2_width = dim2.width();

  if (input1.getDataType() == ml::train::TensorDim::DataType::FP32) {
    const float *data1 = input1.getData();
    const float *data2 = input2.getData();
    float *rdata = result.getData();
    if (input1_width != input2_width) {
      concat_cl_axis3(data1, data2, rdata, input1_batch_size, input1_channels,
                      input1_height, input1_width, input2_width);
    } else if (input1_height != input2_height) {
      concat_cl_axis2(data1, data2, rdata, input1_batch_size, input1_channels,
                      input1_width, input1_height, input2_height);
    } else if (input1_channels != input2_channels) {
      concat_cl_axis1(data1, data2, rdata, input1_batch_size, input1_height,
                      input1_width, input1_channels, input2_channels);
    }
  } else if (input1.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    const _FP16 *data1 = input1.getData<_FP16>();
    const _FP16 *data2 = input2.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();
    if (input1_width != input2_width) {
      concat_cl_axis3_fp16(data1, data2, rdata, input1_batch_size,
                           input1_channels, input1_height, input1_width,
                           input2_width);
    } else if (input1_height != input2_height) {
      concat_cl_axis2_fp16(data1, data2, rdata, input1_batch_size,
                           input1_channels, input1_width, input1_height,
                           input2_height);
    } else if (input1_channels != input2_channels) {
      concat_cl_axis1_fp16(data1, data2, rdata, input1_batch_size,
                           input1_height, input1_width, input1_channels,
                           input2_channels);
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void ConcatLayerCl::concat_cl_axis3(const float *matAdata,
                                    const float *vecXdata, float *vecYdata,
                                    unsigned int input1_batch_size,
                                    unsigned int input1_channels,
                                    unsigned int input1_height,
                                    unsigned int input1_width,
                                    unsigned int input2_width) {

  bool result = false;

  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {

    const auto &kernel_concat_ptr =
      getLayerKernelPtrs()[Kernels::CONCAT_CL_AXIS3];

    int dim = int(input1_batch_size * input1_channels * input1_height *
                  (input1_width + input2_width));

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(float) * input1_batch_size * input1_channels * input1_height *
        input1_width,
      matAdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(float) * input1_batch_size * input1_channels * input1_height *
        input2_width,
      vecXdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(float) * input1_batch_size * input1_channels * input1_height *
        (input1_width + input2_width),
      vecYdata);
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      0, &clbuffInstance.getInBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      1, &clbuffInstance.getInBufferB()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      2, &clbuffInstance.getOutBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(3, &input1_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(4, &input1_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(5, &input1_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(6, &input1_width, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(7, &input2_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {dim, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {1, 1, 1}; // test-value

    result = global_cl_context->command_queue_inst_.DispatchCommand(
      kernel_concat_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(float) * input1_batch_size * input1_channels * input1_height *
        (input1_width + input2_width),
      vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

void ConcatLayerCl::concat_cl_axis2(const float *matAdata,
                                    const float *vecXdata, float *vecYdata,
                                    unsigned int input1_batch_size,
                                    unsigned int input1_channels,
                                    unsigned int input1_width,
                                    unsigned int input1_height,
                                    unsigned int input2_height) {

  bool result = false;

  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {

    const auto &kernel_concat_ptr =
      getLayerKernelPtrs()[Kernels::CONCAT_CL_AXIS2];

    int dim = int(input1_batch_size * input1_channels * input1_width *
                  (input1_height + input2_height));

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(float) * input1_batch_size * input1_channels * input1_height *
        input1_width,
      matAdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(float) * input1_batch_size * input1_channels * input2_height *
        input1_width,
      vecXdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(float) * input1_batch_size * input1_channels *
        (input1_height + input2_height) * input1_width,
      vecYdata);
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      0, &clbuffInstance.getInBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      1, &clbuffInstance.getInBufferB()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      2, &clbuffInstance.getOutBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(3, &input1_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(4, &input1_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(5, &input1_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(6, &input2_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(7, &input1_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {dim, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {1, 1, 1}; // test-value

    result = global_cl_context->command_queue_inst_.DispatchCommand(
      kernel_concat_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(float) * input1_batch_size * input1_channels *
        (input1_height + input2_height) * input1_width,
      vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

void ConcatLayerCl::concat_cl_axis1(const float *matAdata,
                                    const float *vecXdata, float *vecYdata,
                                    unsigned int input1_batch_size,
                                    unsigned int input1_height,
                                    unsigned int input1_width,
                                    unsigned int input1_channels,
                                    unsigned int input2_channels) {

  bool result = false;

  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {
    const auto &kernel_concat_ptr =
      getLayerKernelPtrs()[Kernels::CONCAT_CL_AXIS1];

    int dim = int(input1_batch_size * input1_width * input1_height *
                  (input1_channels + input2_channels));

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(float) * input1_batch_size * input1_channels * input1_height *
        input1_width,
      matAdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(float) * input1_batch_size * input2_channels * input1_height *
        input1_width,
      vecXdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(float) * input1_batch_size * input1_width * input1_height *
        (input1_channels + input2_channels),
      vecYdata);
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      0, &clbuffInstance.getInBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      1, &clbuffInstance.getInBufferB()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      2, &clbuffInstance.getOutBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(3, &input1_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(4, &input1_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(5, &input2_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(6, &input1_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(7, &input1_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {dim, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {1, 1, 1}; // test-value

    result = global_cl_context->command_queue_inst_.DispatchCommand(
      kernel_concat_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(float) * input1_batch_size * input1_width * input1_height *
        (input1_channels + input2_channels),
      vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

#ifdef ENABLE_FP16
void ConcatLayerCl::concat_cl_axis3_fp16(const _FP16 *matAdata,
                                         const _FP16 *vecXdata, _FP16 *vecYdata,
                                         unsigned int input1_batch_size,
                                         unsigned int input1_channels,
                                         unsigned int input1_height,
                                         unsigned int input1_width,
                                         unsigned int input2_width) {

  bool result = false;

  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {

    const auto &kernel_concat_ptr =
      getLayerKernelPtrs()[Kernels::CONCAT_CL_AXIS3_FP16];

    int dim = int(input1_batch_size * input1_channels * input1_height *
                  (input1_width + input2_width));

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(_FP16) * input1_batch_size * input1_channels * input1_height *
        input1_width,
      matAdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(_FP16) * input1_batch_size * input1_channels * input1_height *
        input2_width,
      vecXdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(_FP16) * input1_batch_size * input1_channels * input1_height *
        (input1_width + input2_width),
      vecYdata);
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      0, &clbuffInstance.getInBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      1, &clbuffInstance.getInBufferB()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      2, &clbuffInstance.getOutBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(3, &input1_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(4, &input1_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(5, &input1_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(6, &input1_width, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(7, &input2_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {dim, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {1, 1, 1}; // test-value

    result = global_cl_context->command_queue_inst_.DispatchCommand(
      kernel_concat_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(_FP16) * input1_batch_size * input1_channels * input1_height *
        (input1_width + input2_width),
      vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

void ConcatLayerCl::concat_cl_axis2_fp16(const _FP16 *matAdata,
                                         const _FP16 *vecXdata, _FP16 *vecYdata,
                                         unsigned int input1_batch_size,
                                         unsigned int input1_channels,
                                         unsigned int input1_width,
                                         unsigned int input1_height,
                                         unsigned int input2_height) {

  bool result = false;

  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {
    const auto &kernel_concat_ptr =
      getLayerKernelPtrs()[Kernels::CONCAT_CL_AXIS2_FP16];

    int dim = int(input1_batch_size * input1_channels * input1_width *
                  (input1_height + input2_height));

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(_FP16) * input1_batch_size * input1_channels * input1_height *
        input1_width,
      matAdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(_FP16) * input1_batch_size * input1_channels * input2_height *
        input1_width,
      vecXdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(_FP16) * input1_batch_size * input1_channels *
        (input1_height + input2_height) * input1_width,
      vecYdata);
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      0, &clbuffInstance.getInBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      1, &clbuffInstance.getInBufferB()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      2, &clbuffInstance.getOutBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(3, &input1_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(4, &input1_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(5, &input1_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(6, &input2_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(7, &input1_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {dim, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {1, 1, 1}; // test-value

    result = global_cl_context->command_queue_inst_.DispatchCommand(
      kernel_concat_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(_FP16) * input1_batch_size * input1_channels *
        (input1_height + input2_height) * input1_width,
      vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}

void ConcatLayerCl::concat_cl_axis1_fp16(const _FP16 *matAdata,
                                         const _FP16 *vecXdata, _FP16 *vecYdata,
                                         unsigned int input1_batch_size,
                                         unsigned int input1_height,
                                         unsigned int input1_width,
                                         unsigned int input1_channels,
                                         unsigned int input2_channels) {

  bool result = false;

  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {

    const auto &kernel_concat_ptr =
      getLayerKernelPtrs()[Kernels::CONCAT_CL_AXIS1_FP16];

    int dim = int(input1_batch_size * input1_width * input1_height *
                  (input1_channels + input2_channels));

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(_FP16) * input1_batch_size * input1_channels * input1_height *
        input1_width,
      matAdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(_FP16) * input1_batch_size * input2_channels * input1_height *
        input1_width,
      vecXdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(_FP16) * input1_batch_size * input1_width * input1_height *
        (input1_channels + input2_channels),
      vecYdata);
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      0, &clbuffInstance.getInBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      1, &clbuffInstance.getInBufferB()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result = kernel_concat_ptr->SetKernelArguments(
      2, &clbuffInstance.getOutBufferA()->GetBuffer(), sizeof(cl_mem));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(3, &input1_batch_size, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(4, &input1_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(5, &input2_channels, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(6, &input1_height, sizeof(int));
    if (!result) {
      break;
    }

    result =
      kernel_concat_ptr->SetKernelArguments(7, &input1_width, sizeof(int));
    if (!result) {
      break;
    }

    const int work_groups_count[3] = {dim, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {1, 1, 1}; // test-value

    result = global_cl_context->command_queue_inst_.DispatchCommand(
      kernel_concat_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      global_cl_context->command_queue_inst_,
      sizeof(_FP16) * input1_batch_size * input1_width * input1_height *
        (input1_channels + input2_channels),
      vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}
#endif

void ConcatLayerCl::calcDerivative(RunLayerContext &context) {
  //   /**
  //    * @todo skipping calcDerivative, support yet to be added
  //    */
}

void ConcatLayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, concat_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[ConcatLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void ConcatLayerCl::exportTo(Exporter &exporter,
                             const ml::train::ExportMethods &method) const {
  Layer::exportTo(exporter, method);
  exporter.saveResult(concat_props, method, this);
}

std::vector<ClContext::SharedPtrClKernel> &ConcatLayerCl::getLayerKernelPtrs() {
  /**< kernel list relevant with this layer */
  static std::vector<ClContext::SharedPtrClKernel> layer_kernel_ptrs;
  return layer_kernel_ptrs;
}

} /* namespace nntrainer */

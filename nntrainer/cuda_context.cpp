// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    cuda_context.cpp
 * @date    22 Jun 2026
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   NVIDIA CUDA application context implementation (mirror of
 * ClContext).
 */

#include <cuda_context.h>
#include <env_compat.h>

#include <mutex>

#include <addition_layer.h>
#include <compute_ops.h>
#include <cuda_mem_allocator.h>
#include <fc_layer_cl.h>
#include <geglu_layer.h>
#include <logit_softcapping.h>
#include <rms_norm_layer.h>
#include <scalar_multiply.h>
#include <swiglu_layer.h>
#include <tie_word_embedding.h>

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace nntrainer {

std::mutex cuda_factory_mutex;

void CudaContext::initialize() noexcept {
  try {
    // On a dual-backend build this runs at the FIRST
    // Engine::Global() touch of ANY run — including engine=cpu/gpu — and
    // cudaInit()'s cuInit wakes a runtime-PM-suspended dGPU over PCIe
    // (measured: the D3cold wake alone costs seconds). Defer the bring-up when
    // CUDA is not the active engine: explicit NNTR_ENGINE != cuda, or
    // NNTR_ENGINE unset on an OpenCL-enabled build (where the engine default is
    // "gpu", mirroring the app engine default). Non-cuda runs never
    // legitimately touch this context (prewarm/StreamManager gate on the engine
    // string). NNTR_CUDA_EAGER_INIT=1 restores the old eager behavior.
    {
      const char *eng = std::getenv("NNTR_ENGINE");
      const char *eager = std::getenv("NNTR_CUDA_EAGER_INIT");
      const bool eager_on = eager && eager[0] == '1';
#if defined(ENABLE_OPENCL)
      const bool cuda_active = eng && std::string(eng) == "cuda";
#else
      const bool cuda_active = !eng || std::string(eng) == "cuda";
#endif
      if (!cuda_active && !eager_on) {
        ml_logi("[CudaContext] bring-up deferred (engine=%s)",
                eng ? eng : "(unset; OpenCL default)");
        return;
      }
    }
    if (!cudaInit()) {
      ml_loge(
        "Error: CudaContext::initialize() failed (no usable CUDA device)");
      return;
    }

    // Probe device capabilities once (log-only; the ExecPlan resolver consumes
    // this later, docs/ARCHITECTURE_REFACTOR.md §10 T1). Truth from the
    // existing cuda::ContextManager queries — adds no decision site, so
    // byte-identical.
    caps_.backend = "cuda";
    caps_.device_name = context_inst_.GetDeviceName();
    caps_.arch = context_inst_.GetComputeArch();
    caps_.integrated = context_inst_.isIntegrated();
    caps_.unified_memory = true; // cudaMallocManaged (UVM) is the default pool
    ml_logi("[CudaContext] %s", caps_.toString().c_str());

    add_default_object();

    // Unified-Memory allocator: MemoryPool buffers for engine=cuda tensors are
    // cudaMallocManaged -> host-addressable AND device-accessible (the SVM
    // analogue), so a tensor on this context is device-resident with no
    // separate copy step. Falls back to host memory if UVM is unavailable.
    setMemAllocator(std::make_shared<CudaMemAllocator>());

    // Install the CUDA ComputeOps: the element-wise decode ops (swiglu /
    // scalar_mul / softcap) take the device kernels under their residency
    // gates; everything else inherits the CpuComputeOps bodies, which are
    // correct over the host-coherent UVM tensors.
    getContextData()->setComputeOps(get_cuda_ops());

  } catch (std::exception &e) {
    ml_loge("cuda_context: initialization failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("cuda_context: initialization failed due to unknown reason");
  }
}

void CudaContext::add_default_object() {
  // FC: the backend-neutral FullyConnectedLayerCl dispatches its GEMM through
  // the installed CudaComputeOps table — at this change the inherited
  // CpuComputeOps host dot on UVM (a CUDA quantized-GEMM fc() override is a
  // later change). Same class as the gpu context.
  registerFactory(nntrainer::createLayer<FullyConnectedLayerCl>,
                  FullyConnectedLayerCl::type, ml::train::LayerType::LAYER_FC);
  // addition: the core CPU AdditionLayer is pure host Tensor ops -> correct on
  // the host-coherent UVM tensors (do NOT use the OpenCL AdditionLayerCL).
  registerFactory(nntrainer::createLayer<AdditionLayer>, AdditionLayer::type,
                  ml::train::LayerType::LAYER_ADDITION);
  // rms_norm: the backend-neutral RMSNormLayer dispatches via
  // CudaComputeOps::rms_norm — the fp16 device kernel for decode-sized row
  // counts, else this backend's fused host fallback. Both halves accumulate
  // the sum of squares in FP32 (an fp16 activation with a large residual
  // element squares past the fp16 max -> the row zeroes -> garbage).
  registerFactory(nntrainer::createLayer<RMSNormLayer>, RMSNormLayer::type,
                  ml::train::LayerType::LAYER_RMSNORM);
  // geglu: the backend-neutral GeGLULayer dispatches via the installed table;
  // no CUDA geglu override exists at this change, so it resolves to the
  // inherited CpuComputeOps host body on UVM.
  registerFactory(nntrainer::createLayer<GeGLULayer>, GeGLULayer::type);
  // swiglu: the merged backend-neutral SwiGLULayer dispatches via
  // CudaComputeOps::swiglu — the device-resident fp16 one-kernel decode fast
  // path (cuda_swiglu_fp16) under its residency gates, else the inherited
  // host body. Replaces the former app-side SwiGLU fork.
  registerFactory(nntrainer::createLayer<SwiGLULayer>, SwiGLULayer::type);
  // logit_softcapping (promoted to core): dispatches via
  // CudaComputeOps::softcap — the fp16 device kernel on device-accessible
  // logits (carrying the terminal pipeline drain), else the inherited host
  // body on UVM.
  registerFactory(nntrainer::createLayer<LogitSoftCappingLayer>,
                  LogitSoftCappingLayer::type);
  // scalar_multiply (promoted): dispatches via CudaComputeOps::scalar_mul —
  // the opt-in (NNTR_CUDA_ELTWISE) fp16 device kernel, else the inherited
  // host body on UVM (the _gpu variant is OpenCL-only).
  registerFactory(nntrainer::createLayer<ScalarMultiplyLayer>,
                  ScalarMultiplyLayer::type);
  // tie_word_embedding (promoted): host lm_head on UVM (the GPU GEMV is the
  // OpenCL path, now #if ENABLE_OPENCL-guarded inside the layer so it builds
  // without OpenCL). Register unconditionally.
  registerFactory(nntrainer::createLayer<TieWordEmbedding>,
                  TieWordEmbedding::type);
}

template <typename T>
const int CudaContext::registerFactory(const FactoryType<T> factory,
                                       const std::string &key,
                                       const int int_key) {
  static_assert(
    isSupported<T>::value,
    "cuda_context: given type is not supported for current context");

  auto &index = std::get<IndexType<T>>(factory_map);
  auto &str_map = std::get<StrIndexType<T>>(index);
  auto &int_map = std::get<IntIndexType>(index);

  std::string assigned_key = key == "" ? factory({})->getType() : key;

  std::transform(assigned_key.begin(), assigned_key.end(), assigned_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::lock_guard<std::mutex> lock(cuda_factory_mutex);
  if (str_map.find(assigned_key) != str_map.end()) {
    std::stringstream ss;
    ss << "cuda_context: cannot register factory with already taken key: "
       << key;
    throw std::invalid_argument(ss.str().c_str());
  }

  if (int_key != -1 && int_map.find(int_key) != int_map.end()) {
    std::stringstream ss;
    ss << "cuda_context: cannot register factory with already taken int key: "
       << int_key;
    throw std::invalid_argument(ss.str().c_str());
  }

  int assigned_int_key = int_key == -1 ? str_map.size() + 1 : int_key;

  str_map[assigned_key] = factory;
  int_map[assigned_int_key] = assigned_key;

  ml_logd("cuda_context: factory has registered with key: %s, int_key: %d",
          assigned_key.c_str(), assigned_int_key);

  return assigned_int_key;
}

const CudaContext::SharedPtrCudaKernel
CudaContext::registerCudaKernel(const std::string &kernel_source,
                                const std::string &kernel_name,
                                const std::string &compile_options) {
  // hot path: a single key + lookup, no copy of the (multi-KB) source string.
  const std::string kkey = kernel_name + compile_options;
  auto it = cuda_kernel_map.find(kkey);
  if (it != cuda_kernel_map.end())
    return it->second;

  // owning module cache: kernels sharing one (source, options) reuse the
  // compiled+loaded CUmodule (and its on-disk PTX cache, see cuda_module.cpp).
  const std::string mkey =
    std::to_string(cuda::Module::GetKernelHash(kernel_source, compile_options));
  std::shared_ptr<cuda::Module> module;
  auto mit = cuda_module_map.find(mkey);
  if (mit != cuda_module_map.end()) {
    module = mit->second;
  } else {
    module = std::make_shared<cuda::Module>();
    if (!module->CreateModuleFromSource(kernel_source, kernel_name,
                                        compile_options)) {
      ml_loge("Failed to compile CUDA module for kernel %s",
              kernel_name.c_str());
      return nullptr;
    }
    cuda_module_map.emplace(mkey, module);
  }

  SharedPtrCudaKernel kernelPtr = std::make_shared<cuda::Kernel>();
  if (!kernelPtr->CreateKernelFromModule(*module, kernel_name)) {
    ml_loge("Failed to resolve CUDA kernel %s", kernel_name.c_str());
    return nullptr;
  }
  cuda_kernel_map.emplace(kkey, kernelPtr);
  return cuda_kernel_map[kkey];
}

/**
 * @copydoc const int CudaContext::registerFactory
 */
template const int CudaContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

// Non-template seam (Context::registerLayerFactory override): forwards to the
// per-class registerFactory<Layer> here in the same TU so the explicit
// instantiation is used and no template crosses the .so boundary.
int CudaContext::registerLayerFactory(PtrFactoryType<nntrainer::Layer> factory,
                                      const std::string &key,
                                      const int int_key) {
  return registerFactory<nntrainer::Layer>(factory, key, int_key);
}

} // namespace nntrainer

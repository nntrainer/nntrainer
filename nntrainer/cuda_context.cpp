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

#include <activation_layer.h>
#include <addition_layer.h>
#include <compute_ops.h>
#include <cuda_mem_allocator.h>
#include <cuda_rmsnorm_layer.h>
#include <fc_layer_cl.h>
#include <layer_normalization_layer.h>
#include <lm_head.h>
#include <logit_softcapping.h>
#include <qkv_layer.h>
#include <scalar_multiply.h>
#include <swiglu_layer.h>
#include <tie_word_embedding.h>

// The decode/prefill graph state machine needs the model walk and the CUDA
// graph API (cuda_context.h already pulls in the stream/context managers).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <neuralnet.h>

namespace nntrainer {

std::mutex cuda_factory_mutex;

CudaContext &CudaContext::Global() {
  // Out-of-line + intentionally leaked (see header note): matches the
  // never-destroy convention adopted for the whole GPU-context singleton
  // family (ClContext::Global(), cuda::ContextManager/StreamManager/
  // BlasManager::Global()) after the 2026-07-20 shared+cuda exit crash.
  static CudaContext *instance = new CudaContext();
  instance->initializeOnce();
  return *instance;
}

void CudaContext::initialize() noexcept {
  try {
    // [r20 fresh-init tax] On a dual-backend build this runs at the FIRST
    // Engine::Global() touch of ANY run — including engine=cpu/gpu — and
    // cudaInit()'s cuInit wakes a runtime-PM-suspended dGPU over PCIe
    // (measured: nvidia-smi-alone D3cold wake 2.27s on RTX 5060 = the whole
    // "fresh intel init +2.4s" constant; waking the card first drops a fresh
    // intel init from 3451 to 1133 ms). Defer the bring-up when CUDA is not
    // the active engine: explicit NNTR_ENGINE != cuda, or NNTR_ENGINE unset
    // on an OpenCL-enabled build (where the engine default is "gpu",
    // mirroring causallm_engine()). Non-cuda runs never legitimately touch
    // this context (prewarm/StreamManager gate on the engine string).
    // NNTR_CUDA_EAGER_INIT=1 restores the old eager behavior.
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

    const bool integrated = context_inst_.isIntegrated();
    ml_logi("[CudaContext] device=\"%s\" arch=%s integrated=%d "
            "concurrentManagedAccess=%d",
            context_inst_.GetDeviceName().c_str(),
            context_inst_.GetComputeArch().c_str(), (int)integrated,
            (int)context_inst_.concurrentManagedAccess());

    // Hardware-derived defaults. The device kernels this backend adds are
    // individually switchable, which is useful while bringing a new part up
    // but is a bad deal for a user: nobody should have to export a list of
    // flags to get the backend they asked for. So the context fills in the
    // profile that is right for the device it just probed, with
    // setenv(..., overwrite=0) so an explicit setting from the environment
    // always wins (including "=0", which every consumer treats as off -- see
    // nntr_env_on()).
    setenv("NNTR_CUDA_GEGLU", "1", 0);
    setenv("NNTR_CUDA_ELTWISE", "1", 0);
    setenv("NNTR_CUDA_ROPE", "1", 0);
    setenv("NNTR_FC_CUDA_CUBLAS", "1", 0);
    setenv("NNTR_CUDA_PREWARM", "1", 0);
    setenv("NNTR_CUDA_ATTN", "1", 0);
    setenv("NNTR_CUDA_FLASH_DECODE", "64", 0);
    setenv("NNTR_CUDA_BLOCKQ", "1", 0);
    if (!integrated && context_inst_.concurrentManagedAccess()) {
      // Discrete-GPU profile: let work queue up instead of draining after
      // every op. This is only legal when the driver reports concurrent
      // managed access -- without it (notably the Windows WDDM model) a host
      // touch of managed memory with kernels in flight is an access violation
      // rather than a race, so an integrated or WDDM device keeps the
      // conservative profile.
      setenv("NNTR_CUDA_ASYNC", "1", 0);
      setenv("NNTR_CUDA_GRAPH", "1", 0);
      // The row cap reads "=all" as RAISE, not disable: the device norm kernel
      // synchronizes per call, so on a wide (prefill-shaped) row window the
      // multi-threaded host loop wins and the default caps the device path at
      // 32 rows. On a discrete part the launch is cheap enough that uncapping
      // wins everywhere.
      setenv("NNTR_RMSNORM_CUDA_OFF", "all", 0);
      setenv("NNTR_LAYERNORM_CUDA_OFF", "all", 0);
    }

    add_default_object();

    // Unified-Memory allocator: MemoryPool buffers for engine=cuda tensors are
    // cudaMallocManaged -> host-addressable AND device-accessible (the SVM
    // analogue), so a tensor on this context is device-resident with no
    // separate copy step. Falls back to host memory if UVM is unavailable.
    setMemAllocator(std::make_shared<CudaMemAllocator>());

    // ComputeOps = the CUDA op table. CudaComputeOps derives from CpuComputeOps
    // rather than from the abstract base, because engine=cuda tensors are
    // Unified Memory and therefore host-coherent: every op the CUDA table has
    // not overridden yet still computes the right answer by running the CPU
    // implementation over the managed buffer. That is what lets the table be
    // filled in one op at a time instead of having to cover the whole surface
    // before anything can run. A neutral Layer calling
    // in.getOps()->layer_norm(...) lands here with no #ifdef anywhere in
    // nntrainer/layers.
    getContextData()->setComputeOps(get_cuda_ops());

  } catch (std::exception &e) {
    ml_loge("cuda_context: initialization failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("cuda_context: initialization failed due to unknown reason");
  }
}

void CudaContext::add_default_object() {
  // RMS normalization is the one CUDA-specific Layer class here, and it exists
  // for a numerical reason rather than a performance one: the host FP16 path
  // squares the row in FP16, so a residual element of |x| ~ 1700 -- which real
  // transformer blocks do produce -- overflows the sum of squares to +Inf and
  // zeroes the row. This class accumulates in FP32 and hands the row window to
  // a device kernel. It registers under the same type string as the OpenCL
  // RMSNormLayerCl, so a graph moves between backends by changing engine= and
  // nothing else.
  registerFactory(nntrainer::createLayer<CudaRMSNormLayer>,
                  CudaRMSNormLayer::type, ml::train::LayerType::LAYER_RMSNORM);

  // Everything below is a BACKEND-NEUTRAL core class, registered
  // unchanged -- literally the same objects the CPU context registers. They
  // reach the device through the CUDA op table (CudaComputeOps), not through a
  // per-backend Layer fork, which is the entire point of the Tensor-level
  // whole-op surface.
  //
  // fully connected: the same backend-neutral class the OpenCL context
  // registers. Its GEMM goes out through in.getOps()->fc(), which lands on
  // CudaComputeOps::fc -- the quantized device path (QS4CX dp4a / cuBLAS) with
  // the inherited host implementation as the fallback. The layer itself
  // contains no CUDA code and no #ifdef.
  registerFactory(nntrainer::createLayer<FullyConnectedLayerCl>,
                  FullyConnectedLayerCl::type, ml::train::LayerType::LAYER_FC);
  // addition: host Tensor ops, correct on the host-coherent managed buffers;
  // its residual_op dispatch is where the residual stream can stay in place.
  registerFactory(nntrainer::createLayer<AdditionLayer>, AdditionLayer::type,
                  ml::train::LayerType::LAYER_ADDITION);
  // layer normalization / activation: dispatch to CudaComputeOps::layer_norm
  // and ::activation, which run device kernels for the shapes and dtypes they
  // cover and fall back to the inherited host implementation for the rest.
  registerFactory(nntrainer::createLayer<LayerNormalizationLayer>,
                  LayerNormalizationLayer::type,
                  ml::train::LayerType::LAYER_LAYER_NORMALIZATION);
  registerFactory(nntrainer::createLayer<ActivationLayer>,
                  ActivationLayer::type,
                  ml::train::LayerType::LAYER_ACTIVATION);

  // The promoted LLM layers, again the same core classes. This table has no
  // whole-op entry for them yet, so they run the inherited host implementation
  // over the host-coherent managed buffer: unaccelerated, but correct. They
  // are registered for the same reason as on the OpenCL context -- an
  // unregistered type makes createLayer() throw, so a model could not build
  // its graph under engine=cuda even for the layers this backend does
  // accelerate.
  registerFactory(nntrainer::createLayer<LmHeadLayer>, LmHeadLayer::type);
  registerFactory(nntrainer::createLayer<LogitSoftCappingLayer>,
                  LogitSoftCappingLayer::type);
  registerFactory(nntrainer::createLayer<QKVLayer>, QKVLayer::type);
  registerFactory(nntrainer::createLayer<ScalarMultiplyLayer>,
                  ScalarMultiplyLayer::type);
  // swiglu dispatches to CudaComputeOps::swiglu, which does have a device
  // kernel for it.
  registerFactory(nntrainer::createLayer<SwiGLULayer>, SwiGLULayer::type);
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

// CUDA override of the decode/prefill step.
//
// A single-token decode step issues on the order of a thousand tiny kernels,
// and the CPU-side launch between them -- not the kernels -- is what limits it:
// the GPU sits idle for most of the step. Capturing the step into a CUDA graph
// once and replaying it collapses that launch cost.
//
// Two capture points, both off unless asked for:
//
//  * the DECODE graph, captured on the first single-token step and replayed for
//    every later one. Between replays only the nodes the model declared as feed
//    nodes re-run on the host (to refresh what the graph reads through fixed
//    device pointers) and the token position is updated in device memory. A
//    model that declares no feed nodes does not get this path at all, because
//    replaying without the refresh would silently reuse the previous step's
//    embedding.
//  * the PREFILL graph, which is the same machinery applied to the multi-token
//    step. Default on for an integrated GPU, where the per-op drain the eager
//    path needs is expensive; a discrete GPU keeps the eager path.
//
// The flag is value-checked rather than presence-checked, because the context
// auto-fills it on hardware where it belongs. A presence check would make "=0"
// mean "on" here while the attention layer's own value-checked gate turned OFF
// -- and that split state is not a slowdown, it is corruption: the replay
// rewrites K/V at the first captured slot every step.
sharedConstTensors CudaContext::runDecode(NeuralNetwork &nn, unsigned int from,
                                          unsigned int to,
                                          const sharedConstTensors &input,
                                          const sharedConstTensors &label) {
  sharedConstTensors out;

  static const bool decode_graph = nntr_env_on("NNTR_CUDA_GRAPH");
  static const bool prefill_graph = []() {
    const char *e = std::getenv("NNTR_CUDA_PREFILL_GRAPH");
    if (e != nullptr)
      return e[0] != '0';
    return nntrainer::cuda::ContextManager::Global().isIntegrated();
  }();
  static const bool graph_dbg = std::getenv("NNTR_CUDA_GRAPH_DBG") != nullptr;

  static cudaGraphExec_t cached_exec = nullptr;
  static sharedConstTensors cached_out;
  bool captured = false;

  const bool feed_declared = !nn.getGraphReplayFeedNodes().empty();
  const bool single_token = (to - from) == 1 && from != 0;

  if (decode_graph && feed_declared && !single_token &&
      cached_exec != nullptr) {
    // A new sequence, or a resumed multi-token step, is about to run eagerly.
    // That forward may free and reallocate the scratch the captured graph holds
    // pointers into, so the graph has to go now; replaying it afterwards would
    // launch with dangling device pointers. The next decode step recaptures
    // against the fresh ones.
    cudaGraphExecDestroy(cached_exec);
    cached_exec = nullptr;
    cached_out = {};
  }

  if (decode_graph && feed_declared && single_token) {
    auto &sm = nntrainer::cuda::StreamManager::Global();
    if (cached_exec != nullptr) {
      nn.setStepFeedOnly(true);
      out = nn.incremental_forwarding(from, to, input, label, false);
      nn.setStepFeedOnly(false);
      nntrainer::cuda::cuda_set_pos((int)from, (int)from + 1);
      cudaGraphLaunch(cached_exec, sm.GetStream());
      cudaStreamSynchronize(sm.GetStream());
      out = cached_out;
      captured = true;
    } else if (sm.beginCapture()) {
      nntrainer::cuda::cuda_set_pos((int)from, (int)from + 1);
      out = nn.incremental_forwarding(from, to, input, label, false);
      cudaGraph_t graph = nullptr;
      if (sm.endCapture(&graph) && graph != nullptr) {
        if (graph_dbg) {
          size_t n_nodes = 0;
          cudaGraphGetNodes(graph, nullptr, &n_nodes);
          std::fprintf(stderr, "[CUDA_GRAPH] decode graph: %zu nodes\n",
                       n_nodes);
        }
        if (cudaGraphInstantiate(&cached_exec, graph, 0) == cudaSuccess) {
          cudaGraphLaunch(cached_exec, sm.GetStream());
          cudaStreamSynchronize(sm.GetStream());
          cached_out = out;
          captured = true;
        }
        cudaGraphDestroy(graph);
      } else {
        // The capture was invalidated (an allocation inside it, typically).
        // Clear the sticky error so the eager fallback below is not blamed.
        cudaGetLastError();
      }
    }
  }

  if (!captured && prefill_graph && !nn.isPrefillCaptureDisabled() &&
      from == 0 && (to - from) > 1) {
    auto &sm = nntrainer::cuda::StreamManager::Global();
    if (sm.beginCapture()) {
      out = nn.incremental_forwarding(from, to, input, label, false);
      cudaGraph_t graph = nullptr;
      if (sm.endCapture(&graph) && graph != nullptr) {
        cudaGraphExec_t exec = nullptr;
        if (cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess) {
          cudaGraphLaunch(exec, sm.GetStream());
          cudaStreamSynchronize(sm.GetStream());
          cudaGraphExecDestroy(exec);
          captured = true;
        }
        cudaGraphDestroy(graph);
      } else {
        cudaGetLastError();
      }
    }
  }

  if (!captured)
    out = nn.incremental_forwarding(from, to, input, label, false);

  return out;
}

} // namespace nntrainer

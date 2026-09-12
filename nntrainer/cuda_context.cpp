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

#include <algorithm>
#include <cctype>
#include <mutex>

#include <activation_layer.h>
#include <addition_layer.h>
#include <compute_ops.h>
#include <cuda_mem_allocator.h>
#include <cuda_rmsnorm_layer.h>
#include <fc_layer_cl.h>
#include <geglu_layer.h>
#include <layer_normalization_layer.h>
#include <sigmoid_add_layer.h>
#include <sigmoid_glu_layer.h>

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
    //
    // This test MUST match Engine::add_default_object()'s gate for "cuda"
    // exactly: same environment variable, same escape hatch, same default.
    // Two gates that merely resemble each other is the bug -- one of them
    // then registers a context the other declines to bring up, or brings up a
    // device nothing will use. NNTR_ENGINE is the single selector (lowercased
    // there, so compare case-insensitively here), the default is off because
    // CUDA is opt-in, and NNTR_CUDA_EAGER_CTX is the one flag that restores
    // the unconditional bring-up on both sides.
    {
      const char *eng = std::getenv("NNTR_ENGINE");
      const char *eager = std::getenv("NNTR_CUDA_EAGER_CTX");
      const bool eager_on = eager != nullptr && eager[0] != '0';
      std::string eng_l = eng != nullptr ? std::string(eng) : std::string();
      std::transform(eng_l.begin(), eng_l.end(), eng_l.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (eng_l != "cuda" && !eager_on) {
        ml_logi("[CudaContext] bring-up deferred (NNTR_ENGINE=%s)",
                eng ? eng : "(unset)");
        return;
      }
    }
    if (!cudaInit()) {
      ml_loge(
        "Error: CudaContext::initialize() failed (no usable CUDA device)");
      return;
    }

    // Default-ON opt-out. nntr_env_on() cannot express this: it answers false
    // for an unset variable, which is the right default for a lever but the
    // wrong one for a profile that must apply unless someone says not to.
    const auto wddmProfileWanted = []() {
      const char *e = std::getenv("NNTR_CUDA_WDDM_PROFILE");
      return !(e != nullptr && e[0] == '0');
    };

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
    // Only flags this library actually reads are defaulted here. A default for
    // a flag whose read site lives in a module that does not exist yet would
    // be a shim nothing consumes; the PR that adds the reader derives the
    // default at the read site instead.
    setenv("NNTR_CUDA_GEGLU", "1", 0);
    setenv("NNTR_FC_CUDA_CUBLAS", "1", 0);
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
      // Async submission (no per-op drain) is worth roughly +75 % prefill and
      // +100 % decode on a 1K cell, but it is only COHERENT when no host op
      // sits in the middle of the device chain: dropping the drains means a
      // host read of a managed activation races the kernel still writing it,
      // and concurrentManagedAccess page faults do not order the two.
      //
      // What makes it safe is the device-only activation pool
      // (NNTR_CUDA_DEV_ACT): with activations in real device memory a host
      // fallback cannot silently engage -- it faults -- so every op is on the
      // device and one drain per token is enough. The switch is read in this
      // library (Manager::activationAllocator gives the tensor pool the
      // device-only allocator while the weights keep UVM), so a discrete part
      // can be given the pool the async profile needs right here.
      //
      // NNTR_CUDA_VCOPY_PREFILL is part of the same profile rather than a
      // separate decision: it moves the prefill V-cache copy onto the GPU, and
      // that copy is the last host touch left inside the prefill chain -- kept
      // on the host it reinstates a per-layer drain and hands most of the
      // prefill win back. Its read site is in the attention layer this backend
      // is driven by rather than in this library, so on a library-only build
      // the value is simply never read; it is defaulted here because a profile
      // describes the DEVICE it probed, not one directory.
      //
      // Measured on a discrete laptop part (cMA=1) with a 1K summarisation
      // cell, both models: prefill 7.26k -> 8.96k TPS (+23 %), decode
      // 68.5 -> 132.6 TPS (+94 %), same token count, same end-of-turn stop,
      // byte-identical output, VmHWM unchanged, GPU residency +6 MiB.
      //
      // Opting out is per lever and explicit, since setenv(overwrite=0) never
      // overwrites a value the environment already carries:
      // NNTR_CUDA_DEV_ACT=0 restores the managed activation pool and with it
      // drained submission (async is only auto-enabled alongside the pool),
      // NNTR_CUDA_VCOPY_PREFILL=0 puts the prefill V-copy back on the host,
      // and NNTR_DETERMINISTIC=1 keeps drained submission with the pool on.
      setenv("NNTR_CUDA_DEV_ACT", "1", 0);
      setenv("NNTR_CUDA_VCOPY_PREFILL", "1", 0);
      // Async submission is auto-enabled alongside the device-only pool; the
      // derivation now lives after this block, because its precondition is the
      // pool -- not the device probe that happens to set the pool here.
      //
      // The decode-graph pair (NNTR_CUDA_GRAPH + NNTR_CUDA_M2B) is not decided
      // here either: it applies to a discrete part of EITHER memory model and
      // is defaulted in its own block below. What keeps it from reaching an
      // unverified model is a runtime gate rather than this probe -- capture
      // and replay are only correct with the FEED half wired up (between two
      // replays the host has to re-run the nodes that produce this token's
      // inputs, so the replay reads new bytes instead of the ones frozen into
      // the capture), and a model that declares no such nodes never reaches the
      // capture at all. The same capture also makes any in-capture host FC
      // freeze its output.
      //
      // The row cap reads "=all" as RAISE, not disable: the device norm kernel
      // synchronizes per call, so on a wide (prefill-shaped) row window the
      // multi-threaded host loop wins and the default caps the device path at
      // 32 rows. On a discrete part the launch is cheap enough that uncapping
      // wins everywhere.
      setenv("NNTR_RMSNORM_CUDA_OFF", "all", 0);
      setenv("NNTR_LAYERNORM_CUDA_OFF", "all", 0);
    } else if (!integrated && wddmProfileWanted()) {
      // Discrete part whose driver does not report concurrent managed access:
      // in practice the Windows WDDM model. It used to fall through with only
      // the unconditional flags above, which left it on drained submission --
      // measured on an RTX 5070 Laptop, gemma4 1K decode 40.6 TPS against 134
      // for the same tree on a cMA=1 Linux part.
      //
      // The reason the levers were withheld does not apply to this device the
      // way the comment above states it. That text reasons about a MANAGED
      // activation pool, but on cMA=0 there is no managed pool to touch:
      // CudaMemAllocator::use_host_mapped() already defaults to true here and
      // hands out pinned host-mapped (zero-copy) memory instead, whose pages
      // never migrate -- a host touch mid-kernel is a data race, not the access
      // violation the comment describes. And NNTR_CUDA_DEV_ACT bypasses that
      // pool into real cudaMalloc device memory, where a host fallback faults
      // loudly instead of racing, which is what makes the profile safe.
      //
      // So this device gets the same two levers, for the same reason, and they
      // are not new on Windows: an application layered on this engine can set
      // exactly these from its own `#if defined(_WIN32)` environment block, and
      // that pairing is the second half of an OS-divergence defect -- one gate
      // turning the levers off by device probe, another turning them back on by
      // platform macro, neither configuration tested as a pair. Defaulting them
      // here makes a plain NNTR_ENGINE=cuda run on Windows behave the same way,
      // so a caller does not have to carry device policy.
      //
      // NOT included: NNTR_CUDA_KV_DEV / CUBLAS_WS_MB / QS4CX_DECOMMIT, which
      // such an environment block usually sets too -- those are memory-budget
      // choices, not this profile's subject, and the measurement that motivated
      // them folded them together. The decode-graph pair is not part of this
      // profile either; it is decided for both memory models below.
      //
      // NNTR_CUDA_WDDM_PROFILE=0 disables this block wholesale; each lever
      // remains individually opt-out via its own =0, as everywhere else.
      setenv("NNTR_CUDA_DEV_ACT", "1", 0);
      setenv("NNTR_CUDA_VCOPY_PREFILL", "1", 0);
      setenv("NNTR_RMSNORM_CUDA_OFF", "all", 0);
      setenv("NNTR_LAYERNORM_CUDA_OFF", "all", 0);
      ml_logi("[CudaContext] WDDM discrete profile applied "
              "(concurrentManagedAccess=0); NNTR_CUDA_WDDM_PROFILE=0 to skip");
    }

    // Async submission follows the device-only activation pool, wherever that
    // pool came from -- this profile, an embedding application's environment
    // block, or the caller's shell. It used to be derived INSIDE the cMA guard,
    // so a device that had NNTR_CUDA_DEV_ACT set by someone else never got the
    // matching NNTR_CUDA_ASYNC written and ran drained: precisely the Windows
    // case, where such a block sets the pool and the guard skipped the
    // derivation.
    // Moving it out changes nothing for a cMA=1 part (the guard above sets the
    // pool, then this fires exactly as before) and grants nothing on its own:
    // with no pool, async_ok is false and this writes "0", as today.
    //
    // The rule itself is unchanged: async is coherent only with the pool (a
    // host op in the middle of an undrained chain reads bytes a kernel is still
    // writing), and NNTR_DETERMINISTIC=1 keeps the drains. The runtime gate in
    // cuda_stream_manager additionally requires a non-integrated part.
    {
      const char *dev_act = getenv("NNTR_CUDA_DEV_ACT");
      const bool pool_on = dev_act != nullptr && dev_act[0] == '1';
      // Determinism does NOT have to cost the async profile, and it used to:
      // this derivation also required !NNTR_DETERMINISTIC, so asking for
      // reproducible output halved decode (measured on WDDM, 80 -> 41.8 TPS on
      // one model and 67 -> 36.6 on a second; on a Linux discrete part the same
      // ~2x).
      //
      // The reason for the old coupling was real but narrower than the gate:
      // the hazard of undrained submission is a HOST op reading a buffer a
      // kernel is still writing, and async ALONE (no device-only pool) was
      // measurably non-deterministic -- one run 214 tokens, another 267, on the
      // same input. With the pool on there is no host op left in the chain to
      // race: an activation lives in real device memory, so a host fallback
      // faults instead of quietly participating. The drains then order nothing
      // that is not already ordered.
      //
      // Evidence for dropping it: 5 runs per model on a discrete part with the
      // profile on, byte-identical every time on both models (one of them also
      // with the decode graph enabled), and 3 runs per arm on WDDM likewise.
      // So determinism now keeps the pool's async, and only turns off the
      // levers whose ordering it actually pins (see cuda_blas_manager).
      //
      // An explicit NNTR_CUDA_ASYNC=0 still forces the drains back -- this
      // write is overwrite=0 like every other default -- so a machine that
      // disagrees has a one-variable way out and a bug report worth having.
      setenv("NNTR_CUDA_ASYNC", pool_on ? "1" : "0", 0);
    }

    // The captured decode graph, for a discrete part of either memory model.
    //
    // It was opt-in for exactly one reason: a capture froze host decisions that
    // depend on the key count, so a replay could keep launching an arm the step
    // had outgrown. The dispatch sites now report those comparisons while a
    // capture records (nntrainer::cuda::kv_regime_gt) and the capture is
    // retired and retaken when one of them flips, which closes the half this
    // library can see by itself; a model that knows its own attention window
    // can name the position where its regime changes, which is the same defect
    // from the model side.
    //
    // What still gates the path is the model's own declaration of the nodes a
    // replay must re-run on the host (NeuralNetwork::getGraphReplayFeedNodes):
    // a model that declares none never reaches the capture at all, so this
    // default cannot reach an unverified model. In this tree nothing declares
    // them yet -- the declaration is written by the model, so the default is
    // inert here and becomes effective when a model opts in.
    //
    // Why it is worth defaulting: the replay's saving is per-launch submission
    // cost, which is ~nothing on a Linux discrete part (measured 133 vs 133
    // TPS) and most of the decode on the Windows WDDM model, where the same
    // tree goes 64 -> 102 TPS on one model and 80 -> 134 on another, with
    // byte-identical output. One default for both, rather than a platform macro
    // that sets the levers on one OS only -- that shape is the OS-divergence
    // defect this profile block already had to undo once.
    //
    // Evidence for the default: 2 models x {46-token prompt + 600 tokens, 1K
    // cell} x {off, on} x 3 runs, every output equal to the eager arm's and
    // decode within noise; plus a 5083-token cell with and without the KV ring,
    // all byte-identical.
    //
    // Opting out: NNTR_CUDA_DECODE_GRAPH=0 turns off the pair. Either lever's
    // own =0 does too, and is written through to the other, because HALF the
    // pair is corruption rather than a configuration: the capture and the
    // model-side slot writes have to agree. An explicit 1/0 pair from the
    // caller is left exactly as given, and refused loudly where it is read.
    if (!integrated) {
      const char *pair = std::getenv("NNTR_CUDA_DECODE_GRAPH");
      const char *capture = std::getenv("NNTR_CUDA_GRAPH");
      const char *slots = std::getenv("NNTR_CUDA_M2B");
      const auto declined = [](const char *e) {
        return e != nullptr && e[0] == '0';
      };
      const char *v =
        (declined(pair) || declined(capture) || declined(slots)) ? "0" : "1";
      setenv("NNTR_CUDA_GRAPH", v, 0);
      setenv("NNTR_CUDA_M2B", v, 0);
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
  // geglu: gelu_tanh(gate) * up, dispatching to CudaComputeOps::geglu -- a
  // device FP16 kernel where the inputs are device-resident, and the inherited
  // host implementation over the managed buffer otherwise, so the op table
  // side of this layer is complete on this backend. The registration is what
  // was missing: ml::train::LayerType has no GeGLU enumerator, so this is a
  // string-keyed factory with an auto-assigned integer key, the same shape the
  // application context uses. Without it createLayer("geglu", {engine=cuda})
  // throws "Key is not found for the object", and a gemma-family graph -- the
  // only in-tree consumer of this type -- cannot be built under engine=cuda at
  // all.
  registerFactory(nntrainer::createLayer<GeGLULayer>, GeGLULayer::type);

  // The sigmoid-gated pair likewise: CudaComputeOps has a device kernel for
  // each of them, so the neutral layers reach the device through the table.
  registerFactory(nntrainer::createLayer<SigmoidGluLayer>,
                  SigmoidGluLayer::type);
  registerFactory(nntrainer::createLayer<SigmoidAddLayer>,
                  SigmoidAddLayer::type);
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

  // A capture freezes every host decision taken while it recorded, and some of
  // those depend on the CURRENT key count rather than on the token: a dispatch
  // site picks its kernel, its grid and its masking arm by comparing the live
  // KV length against a fixed bound (the split-KV decode engages at
  // NNTR_CUDA_FLASH_DECODE keys, the GEMM prefill arm at its key floor, the
  // sliding window at its width). The kernels can be taught to read the moving
  // values from cuda_pos_buffer(), but the `if` that picked them already
  // happened and no replay re-asks it, so the answer stays fluent and quietly
  // becomes wrong from the token the bound was crossed on.
  //
  // So the sites REPORT their comparisons while the capture records
  // (nntrainer::cuda::kv_regime_gt) and this asks them again, at this step's
  // key count, before every replay; when one has flipped the graph is retired
  // and recaptured against the arm that is now correct.
  //
  // Measured on a discrete part, 46-token prompt, 600 tokens: the capture lands
  // at 48 keys, below the 64-key split-KV threshold, so the replay kept the
  // dense per-key kernel while the eager path crossed into the split reduce at
  // absolute position 64 -- the two texts diverge at exactly that token.
  // Pinning either arm for the whole run (NNTR_CUDA_FLASH_DECODE=8 or =999999)
  // made replay byte-identical to eager again, which is the evidence that the
  // ARM, not the arithmetic, was the difference.
  const int step_kv = (int)from + 1; // == cuda_set_pos()'s n_kv for this token
  const bool regime_holds =
    cached_exec == nullptr || nntrainer::cuda::kv_regime_holds(step_kv);
  if (cached_exec != nullptr && !regime_holds) {
    if (graph_dbg)
      std::fprintf(stderr,
                   "[CUDA_GRAPH] captured decode graph retired at kv=%d "
                   "(a dispatch threshold flipped); recapturing\n",
                   step_kv);
    cudaGraphExecDestroy(cached_exec);
    cached_exec = nullptr;
    cached_out = {};
  }

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

  /** [capture prewarm] How many decode tokens run EAGER before the capture
   *  token. Default 1, and that 1 is load-bearing rather than a safety margin.
   *
   *  The decode path builds several device-side caches LAZILY, on first use,
   *  and every one of those builders refuses to allocate inside a stream
   *  capture -- a cudaMalloc there invalidates the graph -- so under capture
   *  they return false and their caller falls back to host math. The
   *  fp32->fp16 gamma conversion every RMSNorm needs is the chief example, and
   *  first use is the first M=1 forward, because the norms' device path is
   *  gated to narrow row counts and the whole prefill never touches those
   *  caches.
   *
   *  So capturing ON the first decode token makes "first use" and "capture"
   *  the same call: every cache is cold, every norm declines its device path,
   *  and it either throws or silently vanishes from the recorded graph --
   *  which on a host-addressable pinned pool no residency probe can catch. One
   *  eager token first moves every lazy build outside the capture, at the cost
   *  of exactly one un-graphed token per sequence.
   *
   *  It is also the only place the split-KV decode can publish its fixed
   *  stride: that prewarm early-returns while capturing, and without it the
   *  decode binds no device position buffer and freezes the host KV length at
   *  the capture token -- the signature being an answer that is right for a
   *  while and then degrades the further generation runs past the capture
   *  point.
   *
   *  The eager token is a normal decode: the attention layer sets the device
   *  RoPE/KV position itself when not capturing, so the slot writes stay
   *  correct. NNTR_CUDA_M2B_WARM overrides the count; 0 restores the old
   *  capture-on-first-token behaviour.
   */
  static const unsigned int m2b_warm = []() {
    const char *e = std::getenv("NNTR_CUDA_M2B_WARM");
    return e != nullptr ? (unsigned int)(std::max)(0, std::atoi(e)) : 1u;
  }();
  static unsigned int warm_left = m2b_warm;
  if (from == 0 || (to - from) > 1)
    warm_left = m2b_warm; /* new sequence or a multi-token step */

  if (decode_graph && feed_declared && single_token && cached_exec == nullptr &&
      warm_left > 0) {
    --warm_left;
    if (graph_dbg)
      std::fprintf(stderr,
                   "[CUDA_GRAPH] prewarm: running token at %u eagerly "
                   "(%u left before capture)\n",
                   from, warm_left);
  } else if (decode_graph && feed_declared && single_token) {
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
      // Collect the host comparisons this forward takes, so a later step can
      // ask whether they still hold.
      nntrainer::cuda::kv_regime_begin(step_kv);
      out = nn.incremental_forwarding(from, to, input, label, false);
      nntrainer::cuda::kv_regime_seal();
      cudaGraph_t graph = nullptr;
      if (sm.endCapture(&graph) && graph != nullptr) {
        if (graph_dbg) {
          size_t n_nodes = 0;
          cudaGraphGetNodes(graph, nullptr, &n_nodes);
          std::fprintf(stderr,
                       "[CUDA_GRAPH] decode graph: %zu nodes, %d kv-length "
                       "comparison(s) recorded at kv=%d\n",
                       n_nodes, nntrainer::cuda::kv_regime_size(), step_kv);
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

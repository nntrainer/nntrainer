// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    context.h
 * @date    10 Dec 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains app context related functions and classes that
 * manages the global configuration of the current environment.
 */

#ifndef __CONTEXT_H__
#define __CONTEXT_H__

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <context.h>
#include <context_data.h>
#include <layer.h>
#include <layer_devel.h>
#include <mem_allocator.h>
#include <model_features.h>
#include <optimizer.h>
#include <optimizer_devel.h>

#include <nntrainer_log.h>

namespace nntrainer {

// Forward declarations for the decode seam below — kept as forward
// declarations so context.h does NOT pull in the heavy tensor.h or
// neuralnet.h. The seam's tensor type is
// std::vector<std::shared_ptr<const Tensor>>, which only needs Tensor to be
// declared: a shared_ptr of an incomplete type is fine.
class NeuralNetwork;
class Tensor;

// ContextData lives in its own header so that layer_context.h / layer_node.h
// can pull it in without triggering the context.h → layer_devel.h cycle.

/**
 * @struct DeviceCaps
 * @brief Read-only snapshot of device capabilities, probed ONCE per backend at
 *        Context init from real device queries (clGetDeviceInfo /
 *        cudaGetDeviceProperties via the per-backend ContextManagers) rather
 *        than from NNTR_* env flags. Currently LOG-ONLY — no decision site
 *        reads it yet; it is the input the ExecPlan resolver consumes (see
 *        docs/backend_guide/ARCHITECTURE_REFACTOR.md §6). Fields describe
 *        attributes (what the device can do), never identity (who it is);
 *        unknown values stay at the defaults below.
 */
struct DeviceCaps {
  std::string backend = "cpu";  /**< "cpu" / "gpu" (OpenCL) / "cuda" */
  std::string device_name = ""; /**< human-readable device name */
  std::string arch = "";        /**< backend arch tag, e.g. "compute_120" */
  uint32_t vendor_id = 0;       /**< OpenCL CL_DEVICE_VENDOR_ID; 0 = n/a */
  bool integrated = true;       /**< host+device share one physical pool
                                     (host-coherent); CPU = true */
  bool unified_memory = false;  /**< single-pointer SVM/UVM available */
  bool subgroups = false;       /**< OpenCL cl_intel_subgroups (XMX/DPAS) */
  uint32_t compute_units = 0;   /**< OpenCL CL_DEVICE_MAX_COMPUTE_UNITS */
  uint64_t max_alloc_bytes = 0; /**< per-alloc cap (CL MAX_MEM_ALLOC_SIZE);
                                     0 = unknown/unbounded */
  bool image_v8c = true;        /**< device uses the image2d v8c path (FC GEMM +
                                     KV attention) rather than the cl_mem buffer
                                     path. Both report CL_DEVICE_IMAGE_SUPPORT, so
                                     this is not a clean query — it is set from
                                     vendor_id at init (Intel NEO's compiler
                                     rejects the integer-coord read_imageui v8c
                                     kernel ⇒ buffer; Adreno/unknown ⇒ image). */
  bool dpas = false; /**< OpenCL cl_intel_subgroup_matrix_multiply_accumulate
                          — the actual systolic-array/DPAS matrix engine
                          (Xe2/Xe3 "Arc"/"Battlemage" and later). NOT the
                          same as `subgroups`: cl_intel_subgroups is
                          advertised by every Intel GPU since Gen9
                          (Meteor-Lake Xe-LPG included) and has no matrix
                          unit, so gating XMX on it silently ropes
                          non-DPAS Intel iGPUs into the DPAS kernel
                          (IGC emulates it — catastrophic slowdown). This
                          is the real XMX-capability gate. Declared LAST
                          so appending it leaves the other field offsets
                          unmoved (ABI-safe for an app built against the
                          old DeviceCaps). */

  /**
   * @brief One-line human-readable dump for the init-time log.
   */
  std::string toString() const {
    std::ostringstream os;
    os << "DeviceCaps{backend=" << backend << ", device=\"" << device_name
       << "\", arch=" << (arch.empty() ? "-" : arch) << ", vendor_id=0x"
       << std::hex << vendor_id << std::dec << ", integrated=" << integrated
       << ", unified_memory=" << unified_memory << ", subgroups=" << subgroups
       << ", image_v8c=" << image_v8c << ", dpas=" << dpas
       << ", compute_units=" << compute_units
       << ", max_alloc_bytes=" << max_alloc_bytes << "}";
    return os.str();
  }
};

/**
 * @brief Which quantized FC GEMM family this device should run. Derived from
 *        DeviceCaps (attributes), never from env flags or a device name.
 */
enum class GemmPath {
  CPU,    /**< host CPU backend */
  DP4A,   /**< OpenCL dp4a / buffer int8xint4 (Adreno, non-XMX Intel) */
  XMX,    /**< Intel Xe2/Xe3 systolic DPAS
             (cl_intel_subgroup_matrix_multiply_accumulate present) */
  CUBLAS, /**< CUDA cuBLAS int8 + dp4a */
};

/**
 * @brief Convert a GemmPath to its log string.
 */
inline const char *toString(GemmPath p) {
  switch (p) {
  case GemmPath::CPU:
    return "CPU";
  case GemmPath::DP4A:
    return "DP4A";
  case GemmPath::XMX:
    return "XMX";
  case GemmPath::CUBLAS:
    return "CUBLAS";
  }
  return "?";
}

/**
 * @struct ExecPlan
 * @brief The backend's resolved execution decisions, derived from DeviceCaps.
 *        This is the output of the ExecPlan resolver — the single place where
 *        device attributes (and later ModelFeatures) decide which kernels run,
 *        replacing scattered NNTR_* env flags. Currently a SHADOW
 *        (docs/backend_guide/ARCHITECTURE_REFACTOR.md §6): resolved, logged
 *        and asserted equal to the current env-driven choice, but NOT yet
 *        authoritative — no decision site reads it, so it is byte-identical.
 *
 *        Only cleanly caps-derivable cells are resolved here. Cells that are
 *        NOT a pure function of caps stay env overrides for now and are NOT
 *        shadowed:
 *        - kv_backing (image2d vs cl_mem buffer) — both kinds of device
 *          advertise image2d; the split is a compiler quirk (one vendor's
 *          driver rejects the integer-coordinate read_imageui kernel), which
 *          has no direct probe and is derived from the vendor id instead.
 *        - the SVM coherence drain — a coherence regression whose failure mode
 *          is wrong output, so it stays a conservative decision. It does have
 *          a probe (the device's fine-grain-SVM capability), and is the first
 *          cell scheduled to become resolver-authoritative.
 *        Model-dependent cells (head_dim attention path, KV-share/skip-prefill)
 *        arrive with ModelFeatures.
 *
 *        Plan ownership: once the resolver is authoritative, the resolved
 *        ExecPlan belongs to the compiled MODEL (one plan per model instance,
 *        resolved at compile from that model's ModelFeatures); a Context owns
 *        only its DeviceCaps. Two models in one process then get two plans
 *        over one caps.
 */
struct ExecPlan {
  GemmPath gemm_path = GemmPath::CPU;
  bool host_coherent = true; /**< host+device share one pool (no copy needed) */
  bool decode_gpu = false; /**< run attention/RoPE on the GPU at the M=1 decode
                                step (model-dependent: gemma2/gemma4 yes, qwen3
                                no — d=128 diverges). Filled by the
                              ModelFeatures matcher overload; default off. */

  /**
   * @brief One-line dump for the shadow log.
   */
  std::string toString() const {
    std::ostringstream os;
    os << "ExecPlan{gemm_path=" << nntrainer::toString(gemm_path)
       << ", host_coherent=" << host_coherent << ", decode_gpu=" << decode_gpu
       << "}";
    return os.str();
  }
};

/**
 * @brief Resolve the ExecPlan from device capabilities alone (no env, no
 *        device-name branch). Pure function — the seam the resolver owns.
 */
inline ExecPlan resolveExecPlan(const DeviceCaps &c) {
  ExecPlan p;
  p.host_coherent = c.integrated;
  if (c.backend == "cuda")
    p.gemm_path = GemmPath::CUBLAS;
  else if (c.backend == "gpu")
    p.gemm_path = c.dpas ? GemmPath::XMX : GemmPath::DP4A;
  else
    p.gemm_path = GemmPath::CPU;
  return p;
}

/**
 * @brief The matcher: resolve the ExecPlan from device caps AND the model's
 *        declared features. The caps-only cells (gemm_path, host_coherent) come
 *        from resolveExecPlan(caps); the model-dependent cells (decode_gpu, and
 *        later the head_dim attention path / KV-share) come from ModelFeatures.
 *        SHADOW for now (no decision site reads it) — byte-identical.
 */
inline ExecPlan resolveExecPlan(const DeviceCaps &c, const ModelFeatures &m) {
  ExecPlan p = resolveExecPlan(c);
  // A model that wants GPU decode only gets it on a device that can actually
  // run the resident decode path (a real GPU backend, not the host CPU).
  p.decode_gpu = m.decode_gpu && (c.backend == "gpu" || c.backend == "cuda");
  return p;
}

/**
 * @class Context contains user-dependent configuration for  support
 * @brief  support for app context
 */

class Context {
public:
  using PropsType = std::vector<std::string>;

  template <typename T> using PtrType = std::unique_ptr<T>;

  template <typename T>
  using FactoryType = std::function<PtrType<T>(const PropsType &)>;

  template <typename T>
  using PtrFactoryType = PtrType<T> (*)(const PropsType &);

  template <typename T>
  using StrIndexType = std::unordered_map<std::string, FactoryType<T>>;

  /** integer to string key */
  using IntIndexType = std::unordered_map<int, std::string>;

  /**
   * This type contains tuple of
   * 1) integer -> string index
   * 2) string -> factory index
   */
  template <typename T>
  using IndexType = std::tuple<StrIndexType<T>, IntIndexType>;

  template <typename... Ts> using FactoryMap = std::tuple<IndexType<Ts>...>;

  /**
   * @brief   Default constructor
   */
  Context(std::shared_ptr<ContextData> data_ = nullptr) : data(data_) {}

  /**
   * @brief   Destructor
   */
  virtual ~Context() = default;

  /**
   *
   * @brief Initialization of Context.
   *
   * @return status &
   */
  virtual int init() { return 0; };

  /**
   * @brief Create an Layer Object from the type (string)
   *
   * @param type type of layer
   * @param props property
   * @return PtrType<nntrainer::Layer> unique pointer to the object
   */
  virtual PtrType<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &props = {}) {
    ml_logw(
      "[Warning] Implement createLayerObject for the concrete context class to "
      "properly create the layer");
    return nullptr;
  };

  /**
   * @brief Create an Layer Object from the integer key
   *
   * @param int_key integer key
   * @param props property
   * @return PtrType<nntrainer::Layer> unique pointer to the object
   */
  virtual PtrType<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &props = {}) {
    ml_logw(
      "[Warning] Implement createLayerObject for the concrete context class to "
      "properly create the layer");
    return nullptr;
  };

  /**
   * @brief Create an Optimizer Object from the type (string)
   *
   * @param type type of optimizer
   * @param props property
   * @return PtrType<nntrainer::Optimizer> unique pointer to the object
   */
  virtual PtrType<nntrainer::Optimizer>
  createOptimizerObject(const std::string &type,
                        const std::vector<std::string> &props = {}) {
    return nullptr;
  };

  /**
   * @brief Create an Layer Object from the integer key
   *
   * @param int_key integer key
   * @param props property
   * @return PtrType<nntrainer::Optimizer> unique pointer to the object
   */
  virtual PtrType<nntrainer::Optimizer>
  createOptimizerObject(const int int_key,
                        const std::vector<std::string> &properties = {}) {
    return nullptr;
  };

  /**
   * @brief Create an LearningRateScheduler Object from the type (stirng)
   *
   * @param type type of optimizer
   * @param props property
   * @return PtrType<ml::train::LearningRateScheduler> unique pointer to the
   * object
   */
  virtual PtrType<ml::train::LearningRateScheduler>
  createLearningRateSchedulerObject(
    const std::string &type, const std::vector<std::string> &propeties = {}) {
    return nullptr;
  }

  /**
   * @brief Create an LearningRateScheduler Object from the integer key
   *
   * @param int_key integer key
   * @param props property
   * @return PtrType<ml::train::LearningRateScheduler> unique pointer to the
   * object
   */
  virtual std::unique_ptr<ml::train::LearningRateScheduler>
  createLearningRateSchedulerObject(
    const int int_key, const std::vector<std::string> &propeties = {}) {
    return nullptr;
  }

  /**
   * @brief getter of context name
   *
   * @return string name of the context
   */
  virtual std::string getName() = 0;

  std::shared_ptr<ContextData> getContextData() { return data; }

  std::shared_ptr<MemAllocator> getMemAllocator() {
    return getContextData()->getMemAllocator();
  };

  /**
   * @brief load weight and graph for the specific context
   *
   * @return return 0 for success
   */
  virtual int load(const std::string &file_path) { return 0; };

  /**
   * @brief Read-only device capability snapshot for this backend, probed once
   *        at init. The base returns CPU caps (host-coherent, no accelerator);
   *        ClContext / CudaContext override with a probed snapshot. LOG-ONLY
   *        for now (see docs/backend_guide/ARCHITECTURE_REFACTOR.md §6) — no
   *        decision site reads it yet, so adding or overriding it is
   *        byte-identical.
   *
   * @return const DeviceCaps& capabilities of the device backing this context
   */
  virtual const DeviceCaps &caps() const {
    static const DeviceCaps
      cpu_caps; // backend="cpu", integrated=true, defaults
    return cpu_caps;
  }

  /**
   * @brief Register a layer factory under this backend. The base default is
   *        "unsupported" (returns -1); each concrete Context overrides it to
   *        forward to its own registerFactory<Layer>. This is the non-template
   *        seam that lets callers register a layer on any backend through
   *        Engine::registerLayerFactory(engine, creator) WITHOUT a
   *        static_cast to a concrete ClContext/CudaContext (whose
   *        registerFactory is a per-class explicit template instantiation, an
   *        ABI hazard across the .so boundary). See
   *        docs/backend_guide/ARCHITECTURE_REFACTOR.md §5.
   *
   * @param factory layer creator (createLayer<T> result)
   * @param key string key (empty ⇒ derived from the layer's getType())
   * @param int_key integer key (-1 ⇒ auto-assigned)
   * @return registered integer key, or -1 if this backend declines to
   *         register a layer factory
   */
  virtual int registerLayerFactory(PtrFactoryType<nntrainer::Layer>,
                                   const std::string & = "", const int = -1) {
    ml_logw("[Context] this backend does not support layer registration");
    return -1;
  }

  /**
   * @brief Which residency plane (LayerComputeEngine) this backend's tensors
   *        live on. This is the single authority the layer graph consults to
   *        map a registered engine NAME onto the tensor/residency-plane enum
   *        (toLayerComputeEngine, layer_node.cpp) — retiring the central
   *        name→enum string table (ComputeEngineTypeInfo::EnumStr) in favour of
   *        a per-context declaration. The base is CPU (host residency);
   *        ClContext→GPU, CudaContext→CUDA, QNNContext→QNN override it. A new
   *        aliased/added backend just declares its plane here, with no central
   *        table to edit (add-only). See
   *        docs/backend_guide/ARCHITECTURE_REFACTOR.md §5 step 1.
   * @note  NEW vtable tail: appended after every pre-existing slot so a rebuilt
   *        libnntrainer.so stays ABI-compatible with an app/ccapi built against
   *        the old vtable. The QNN context lives in the libqnn_context.so
   *        plugin, which subclasses Context, so that plugin must be rebuilt
   *        alongside libnntrainer.so (an old plugin lacks this slot entirely —
   *        calling residencyEngine() on it is UB).
   * @return residency-plane enum backing this context's tensors
   */
  virtual ml::train::LayerComputeEngine residencyEngine() const {
    return ml::train::LayerComputeEngine::CPU;
  }

  /**
   * @brief Run one decode or prefill forward step for the model — the
   *        execution seam a backend needs when its decode strategy is not a
   *        plain graph walk. The base IS the plain walk
   *        (nn.incremental_forwarding(...)), so CPU and OpenCL are
   *        byte-identical to the pre-seam code; a backend whose decode is a
   *        capture/replay state machine overrides this instead of adding a
   *        compile-guarded block to neuralnet.cpp.
   *        docs/backend_guide/ARCHITECTURE_REFACTOR.md §6 states the rule: a
   *        backend-specific decode strategy belongs behind one Context hook.
   * @note  Appended at the vtable tail, after residencyEngine(), so every
   *        pre-existing slot keeps its index and a rebuilt libnntrainer.so
   *        stays ABI-compatible with an app built against the old vtable.
   * @note  The parameter and return type is spelled out as
   *        std::vector<std::shared_ptr<const Tensor>> (sharedConstTensors) to
   *        keep tensor.h out of this header; defined out of line in
   *        neuralnet.cpp.
   *
   * @param nn    the model to step
   * @param from  first token position of this step
   * @param to    one past the last token position of this step
   * @param input model inputs for this step
   * @param label model labels for this step
   * @return the model outputs for this step
   */
  virtual std::vector<std::shared_ptr<const Tensor>>
  runDecode(NeuralNetwork &nn, unsigned int from, unsigned int to,
            const std::vector<std::shared_ptr<const Tensor>> &input,
            const std::vector<std::shared_ptr<const Tensor>> &label);

private:
  /**
   * @brief map of context
   */
  static inline std::unordered_map<std::string, Context *> ContextMap;

  std::shared_ptr<ContextData> data = nullptr;
};

using CreateContextFunc = nntrainer::Context *(*)();
using DestroyContextFunc = void (*)(nntrainer::Context *);

/**
 * @brief  Context Pluggable struct that enables pluggable layer
 *
 */
typedef struct {
  CreateContextFunc createfunc;   /**< create layer function */
  DestroyContextFunc destroyfunc; /**< destory function */
} ContextPluggable;

/**
 * @brief pluggable Context must have this structure defined
 */
extern "C" ContextPluggable ml_train_context_pluggable;

} // namespace nntrainer

#endif /* __CONTEXT_H__ */

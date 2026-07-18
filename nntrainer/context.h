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
#include <optimizer.h>
#include <optimizer_devel.h>

#include <nntrainer_log.h>

namespace nntrainer {

// Forward decls for the runDecode seam — kept as forward declarations so
// context.h does NOT pull in the heavy tensor.h / neuralnet.h. The seam's
// tensor type is sharedConstTensors == std::vector<std::shared_ptr<const
// Tensor>>, which only needs Tensor to be *declared* (shared_ptr of an
// incomplete type is fine).
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
 *        reads it yet; it is the input the ExecPlan resolver will consume (see
 *        docs/ARCHITECTURE_REFACTOR.md §10 T1/T4). Fields describe attributes
 *        (what the device can do), never identity (who it is); unknown values
 *        stay at the defaults below.
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
                                     kernel ⇒ buffer; Adreno/unknown ⇒ image). The
                                     V8C_BUF cell of the resolver. */
  bool dpas = false; /**< OpenCL cl_intel_subgroup_matrix_multiply_accumulate
                          — the actual systolic-array/DPAS matrix engine
                          (Xe2/Xe3 "Arc"/"Battlemage" and later). NOT the
                          same as `subgroups`: cl_intel_subgroups is
                          advertised by every Intel GPU since Gen9
                          (Meteor-Lake Xe-LPG included) and has no matrix
                          unit, so gating XMX on it silently ropes
                          non-DPAS Intel iGPUs into the DPAS kernel
                          (IGC emulates it — catastrophic slowdown). This
                          is the real XMX-capability gate. Appended after
                          the pre-existing fields so their offsets stay
                          unmoved (ABI-safe for an app built against the
                          old DeviceCaps). */
  bool svm_fine_grain = false; /**< CL_DEVICE_SVM_FINE_GRAIN_BUFFER: the device
                                    keeps an SVM allocation coherent across a
                                    kernel→kernel handoff on its own. A
                                    coarse-grain device needs a host-side
                                    drain between the producing and consuming
                                    dispatch instead; the backend decides that
                                    once at init from this field and pushes the
                                    decision down. Appended last for the same
                                    ABI reason as `dpas`. */

  /**
   * @brief OpenCL CL_DEVICE_VENDOR_ID of Intel parts.
   *
   * Kept here, beside the fields derived from it, so that backend plumbing
   * (nntrainer/opencl/*) never has to name a vendor: a device quirk enters the
   * codebase as a caps field, and only this seam knows which vendor implies
   * it.
   */
  static constexpr uint32_t VENDOR_INTEL = 0x8086;

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
       << ", svm_fine_grain=" << svm_fine_grain
       << ", compute_units=" << compute_units
       << ", max_alloc_bytes=" << max_alloc_bytes << "}";
    return os.str();
  }
};

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
   * for now (docs/ARCHITECTURE_REFACTOR.md §10 T1) — no decision site reads it
   * yet, so adding/overriding it is byte-identical.
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
   *        ABI hazard across the .so boundary). [docs/ARCHITECTURE_REFACTOR.md
   *        §10 T3 / §11 S1]
   *
   * @param factory layer creator (createLayer<T> result)
   * @param key string key (empty ⇒ derived from the layer's getType())
   * @param int_key integer key (-1 ⇒ auto-assigned)
   * @return registered integer key, or -1 if the backend cannot register
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
   *        table to edit (add-only). [docs/ARCHITECTURE_REFACTOR.md §10 T3]
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
   * @brief Run ONE decode/prefill forward step for the model — the
   *        exec-engine seam. The base is a plain graph walk
   *        (`nn.incremental_forwarding(...)`), so CPU and OpenCL are
   *        byte-identical; a backend with its own decode engine (e.g. a
   *        CUDA-graph capture/replay state machine) overrides it.
   * @note  Appended at the vtable tail (its slot follows every pre-existing
   *        one) so a rebuilt libnntrainer.so stays ABI-compatible with an
   *        app/ccapi built against the old vtable.
   * @note  Return/param type is `sharedConstTensors`
   *        (std::vector<std::shared_ptr<const Tensor>>); spelled out to keep
   *        context.h free of tensor.h. Defined out-of-line in neuralnet.cpp.
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

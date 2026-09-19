// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    cl_context.h
 * @date    23 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @author  Niket Agarwal <niket.a@samsung.com>
 * @author  Thummala Pallavi <t.pallavi@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains app context related functions and classes that
 * manages the global configuration of the current OpenCL environment. It also
 * creates the OpenCL command queue and context.
 */

#include <activation_layer.h>
#include <addition_layer.h>
#include <attention_kernels.h>
#include <blas_kernels.h>
#include <cerrno>
#include <cl_context.h>
#include <cl_kernels/cl_kernels.h>
#include <cl_svm_allocator.h>
#include <compute_ops.h>
#include <concat_cl.h>
#include <cstdlib>
#include <fc_layer_cl.h>
#include <geglu_cl_op.h>
#include <geglu_layer.h>
#include <gelu_cl_op.h>
#include <layer_normalization_layer.h>
#include <layernorm_cl_op.h>
#include <load_trace.h>
#include <opencl_context_manager.h>
#include <opencl_loader.h>
#include <reshape_cl.h>
#include <rmsnorm_layer_cl.h>
#include <string>
#include <swiglu_cl_op.h>
#include <swiglu_layer.h>
#include <transpose_cl.h>

#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <system_error>
#include <unordered_map>
#include <unordered_set>

#if defined(_WIN32)
#include <windows.h>
#else
// For the kernel cache directory ownership/permission check below.
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nntrainer {
#if KERNEL_CACHE
static constexpr bool KERNEL_CACHE_ENABLED = true;
#else
static constexpr bool KERNEL_CACHE_ENABLED = false;
#endif
std::mutex cl_factory_mutex;

// Guards ClContext::ocl_kernel_map, the process-wide compiled-kernel cache.
// Separate from cl_factory_mutex, which guards the layer factory registry: the
// two are unrelated maps taken on unrelated paths, and sharing one lock would
// put a cold kernel compile in front of every layer registration.
static std::mutex ocl_kernel_map_mutex;

// Whether the resolved cache directory is safe to load GPU program binaries
// from. Set once at init and read on every cached lookup; false leaves the
// cache disabled for the process without disabling anything else.
static bool kernel_cache_usable = false;

/**
 * @brief Is the kernel cache directory writable only by its owner, and owned
 *        by the user this process runs as?
 *
 * A cached binary goes straight to clCreateProgramWithBinary, so anyone who
 * can write the file chooses what the GPU driver compiles. The file name is no
 * protection: the key is a hash over kernel sources that ship in this
 * repository plus device strings any local user can read. Decline the cache
 * rather than trust a directory that somebody else can write into -- the cost
 * is one cold kernel compile, and the alternative is a local code-execution
 * surface that appears the moment a process is started from a shared
 * directory.
 *
 * @param path cache directory
 * @return true when the directory may be read from and written to
 */
static bool kernelCacheDirIsPrivate(const std::string &path) {
#if defined(_WIN32) || defined(__ANDROID__)
  // Windows permissions do not map onto the POSIX bits this checks, and on
  // Android the directory is inside the app's own private storage.
  (void)path;
  return true;
#else
  struct stat st;
  if (::stat(path.c_str(), &st) != 0) {
    ml_logw("Cannot stat the kernel cache directory %s; not caching kernels",
            path.c_str());
    return false;
  }
  if (st.st_uid != ::geteuid()) {
    ml_logw("The kernel cache directory %s is owned by another user; not "
            "caching kernels",
            path.c_str());
    return false;
  }
  if ((st.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    ml_logw("The kernel cache directory %s is writable by other users; not "
            "caching kernels",
            path.c_str());
    return false;
  }
  return true;
#endif
}

std::vector<std::byte> readBinaryFile(const std::string &path) {
  // reading binary
  std::ifstream fs(path, std::ios::binary | std::ios::in);

  if (fs.good()) {
    fs.seekg(0, std::ios::end);
    size_t binary_size = fs.tellg();
    fs.seekg(0, std::ios::beg);

    std::vector<std::byte> data(binary_size);
    fs.read(reinterpret_cast<char *>(data.data()), binary_size);
    return data;
  } else {
    return {};
  }
}

/**
 * @brief Directory the kernel binary cache lives in.
 *
 * NNTR_KERNEL_CACHE_DIR names it outright; with the variable unset or empty,
 * Program::DEFAULT_KERNEL_PATH stands, which is the configured path already
 * resolved under the user's own cache directory on every platform that has
 * one.
 *
 * The override is the integration point for an embedder that knows its own
 * private storage and cannot rely on either -- an Android application passing
 * Context.getCacheDir() is the case that motivates it, since the per-user
 * resolution deliberately does not apply there and the configured path stays
 * relative to a working directory the library does not choose.
 *
 * Program::DEFAULT_KERNEL_PATH is still READ when the override points
 * somewhere else, so setting it does not orphan a cache that is already there;
 * new entries go only to the resolved directory.
 */
static const std::string &kernelCacheDir() {
  static const std::string dir = []() -> std::string {
    if (const char *e = std::getenv("NNTR_KERNEL_CACHE_DIR"); e && *e)
      return std::string(e);
    return opencl::Program::DEFAULT_KERNEL_PATH;
  }();
  return dir;
}

bool writeBinaryFile(const std::string &path,
                     const std::vector<std::byte> &data) {
  std::ofstream fs(path, std::ios::out | std::ios::binary);
  if (!fs) {
    ml_loge("Failed to open file for writing: %s", path.c_str());
    return false;
  }

  fs.write(reinterpret_cast<const char *>(data.data()), data.size());
  return true;
}

void ClContext::initialize() noexcept {
  try {
    {
      // Platform + device enumeration, cl_context and the command queue. It
      // is the first thing a GPU run pays and, unlike everything after it,
      // there is no cache that can make it cheaper.
      nntrainer::load_trace::Scope _lt(nntrainer::load_trace::CTX_CREATE);
      if (!clInit()) {
        ml_loge("Error: ClContext::initialize() failed");
        return;
      }
    }

    // Probe the device's capabilities once, here, where the device has just
    // been enumerated. The struct is a plain POD of attributes (what the
    // device can do), never identity, and every consumer reads it through
    // Context::caps() rather than re-querying CL or matching a device name.
    if (const auto *di = context_inst_.getDeviceInfo()) {
      // CL_DEVICE_VENDOR_ID for Intel. Used only to derive the two attributes
      // below whose real signal is a compiler/driver trait no CL query
      // reports; it is never compared against a device name.
      constexpr uint32_t INTEL_VENDOR_ID = 0x8086;

      caps_.backend = "gpu";
      caps_.device_name = di->getDeviceName();
      // CL_DEVICE_NAME is stored sized to include the query's trailing NUL; an
      // embedded NUL would truncate the %s log line, so strip trailing NUL/ws.
      while (!caps_.device_name.empty()) {
        const char c = caps_.device_name.back();
        if (c == '\0' || c == ' ' || c == '\n' || c == '\r' || c == '\t')
          caps_.device_name.pop_back();
        else
          break;
      }
      caps_.vendor_id = di->getDeviceVendorId();
      caps_.compute_units = di->getDeviceMaxComputeUnits();
      caps_.max_alloc_bytes = di->getDeviceMaxMemAllocSize();
      caps_.unified_memory = di->getDeviceSVMCapabilities() != 0;
      caps_.subgroups = di->getDeviceExtensions().find("cl_intel_subgroups") !=
                        std::string::npos;
      // cl_intel_subgroups is advertised by every Intel GPU since Gen9
      // (including non-DPAS Xe-LPG parts), so it cannot gate a DPAS/XMX
      // matrix-engine kernel. The matrix-multiply-accumulate extension is
      // DPAS-specific, so it is the real capability signal.
      caps_.dpas =
        di->getDeviceExtensions().find(
          "cl_intel_subgroup_matrix_multiply_accumulate") != std::string::npos;
      // image_v8c: whether the device should prefer an image2d-based path over
      // a cl_mem buffer path. No clean device query distinguishes the two
      // (both report CL_DEVICE_IMAGE_SUPPORT); the practical split is that
      // Intel NEO's compiler rejects integer-coordinate read_imageui kernels.
      // Keyed off vendor_id -- a stable, queryable, vendor-wide attribute (the
      // quirk is a compiler trait, not a per-model one), not the brittle
      // device_name. Intel => buffer; others keep the image default.
      caps_.image_v8c = (caps_.vendor_id != INTEL_VENDOR_ID);
      cl_bool host_unified = CL_FALSE;
      caps_.integrated =
        (opencl::clGetDeviceInfo(
           context_inst_.GetDeviceId(), CL_DEVICE_HOST_UNIFIED_MEMORY,
           sizeof(host_unified), &host_unified, nullptr) == CL_SUCCESS) &&
        (host_unified == CL_TRUE);
      ml_logi("[ClContext] %s", caps_.toString().c_str());
    }

    if (KERNEL_CACHE_ENABLED) {
      // Best effort: the binary cache is an optimisation. A read-only or
      // otherwise unwritable directory must not take the whole context down
      // with it -- create_directories throws std::filesystem::filesystem_error,
      // and everything below (the kernel registration, the memory allocator,
      // the ops table) would then be skipped by the catch at the end of this
      // function, leaving a context that registers no layer and hands out no
      // allocator.
      std::error_code ec;
      std::filesystem::create_directories(kernelCacheDir(), ec);
      if (ec) {
        ml_logw("Could not create the kernel cache directory %s (%s); "
                "compiling kernels from source without caching them",
                kernelCacheDir().c_str(), ec.message().c_str());
      } else {
        kernel_cache_usable = kernelCacheDirIsPrivate(kernelCacheDir());
      }
    }

    {
      nntrainer::load_trace::Scope _lt(nntrainer::load_trace::KRN_BLAS);
      initBlasClKernels();
    }
    {
      nntrainer::load_trace::Scope _lt(nntrainer::load_trace::KRN_ATTN);
      initAttentionClKernels();
    }

    // The allocator and the ops table are installed BEFORE the layer
    // registrations, not after. add_default_object() throws on a duplicate
    // registration key, and the catch at the end of this function swallows
    // that -- so with the old order a single bad key left the context alive
    // with a null MemAllocator, and the failure surfaced much later as a
    // segfault in TensorPool's constructor (allocator_->makePool on a null
    // shared_ptr) for every model on this engine, with nothing in the log to
    // connect the two. Installing them first bounds the damage of a failed
    // registration to the layers that did not register.
    //
    // SVM-backed allocator so MemoryPool buffers are device-visible
    // without an explicit copy. Falls back to host memory inside
    // ClSVMAllocator when the driver lacks SVM support.
    setMemAllocator(
      std::make_shared<ClSVMAllocator>(opencl::ContextManager::Global()));

    // Install the OpenCL ComputeOps subclass so tensors created from
    // this Context dispatch the whole-op table and the accelerator-only ops
    // (Q4_0/INT4 batch & accel GEMM/GEMV) to the OpenCL kernels instead of
    // throwing or silently taking the CPU path. CPU-only ops on a CL-attached
    // tensor still throw via base default — by design, those stay on a CPU
    // context.
    getContextData()->setComputeOps(get_cl_ops());

    add_default_object();

  } catch (std::exception &e) {
    ml_loge("cl_context: registering layers failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("cl_context: registering layer failed due to unknown reason");
  }
};

void ClContext::add_default_object() {
  // The FC layer is now backend-neutral (it dispatches its GEMM through
  // ComputeOps::fc), so it registers no kernels of its own here.
  registerFactory(nntrainer::createLayer<FullyConnectedLayerCl>,
                  FullyConnectedLayerCl::type, ml::train::LayerType::LAYER_FC);

  // The core AdditionLayer is backend-neutral: its per-input copy and add
  // dispatch through ComputeOps::residual_op, so the residual stream can stay
  // device-resident without forking the layer. The former AdditionLayerCL is
  // gone.
  registerFactory(nntrainer::createLayer<AdditionLayer>, AdditionLayer::type,
                  ml::train::LayerType::LAYER_ADDITION);

  // Likewise SwiGLU: one neutral layer dispatching ComputeOps::swiglu, in
  // place of the former SwiGLULayerCl.
  if (registerSwiGLUClKernels(*this)) {
    registerFactory(nntrainer::createLayer<SwiGLULayer>, SwiGLULayer::type,
                    ml::train::LayerType::LAYER_SWIGLU);
  }

  if (ReshapeLayerCl::registerClKernels(*this)) {
    registerFactory(nntrainer::createLayer<ReshapeLayerCl>,
                    ReshapeLayerCl::type, ml::train::LayerType::LAYER_RESHAPE);
  }

  if (RMSNormLayerCl::registerClKernels(*this)) {
    registerFactory(nntrainer::createLayer<RMSNormLayerCl>,
                    RMSNormLayerCl::type, ml::train::LayerType::LAYER_RMSNORM);
  }

  if (ConcatLayerCl::registerClKernels(*this)) {
    registerFactory(nntrainer::createLayer<ConcatLayerCl>, ConcatLayerCl::type,
                    ml::train::LayerType::LAYER_CONCAT);
  }

  if (TransposeLayerCl::registerClKernels(*this)) {
    registerFactory(nntrainer::createLayer<TransposeLayerCl>,
                    TransposeLayerCl::type,
                    ml::train::LayerType::LAYER_TRANSPOSE);
  }

  // LayerNormalization and Activation are the SAME core classes the cpu
  // context registers, under the same type strings -- there is no
  // LayerNormLayerCl or ActivationLayerCl. Both dispatch their maths through
  // the tensor's ComputeOps, so createLayer("layer_normalization",
  // {engine=gpu}) and createLayer("activation", {activation=gelu,
  // engine=gpu}) land on ClComputeOps::layer_norm / ::activation. Registration
  // is gated on the kernels building, so a device that cannot compile them
  // leaves the type unregistered rather than accepting the layer and throwing
  // at the first forward. Both keys are explicit: the auto-assigned key is
  // str_map.size() + 1, which silently collides with an enum key once the
  // registration list grows.
  if (registerLayerNormClKernels(*this)) {
    registerFactory(nntrainer::createLayer<LayerNormalizationLayer>,
                    LayerNormalizationLayer::type,
                    ml::train::LayerType::LAYER_LAYER_NORMALIZATION);
  }
  if (registerGeluClKernels(*this)) {
    registerFactory(nntrainer::createLayer<ActivationLayer>,
                    ActivationLayer::type,
                    ml::train::LayerType::LAYER_ACTIVATION);
  }

  // GeGLU: gelu_tanh(gate) * up, dispatching to ClComputeOps::geglu -- a
  // device kernel where the gate and up rows are device-resident, and the
  // inherited host implementation over the SVM-coherent buffer otherwise, so
  // the op table side of this layer is complete on this backend once its
  // kernel compiles. What was missing was the factory: ml::train::LayerType
  // has no GeGLU enumerator, so, matching how the application context
  // registers the same class, this is a string-keyed factory with an
  // auto-assigned integer key. Without it, createLayer("geglu", {engine=gpu})
  // throws "Key is not found for the object", and a gemma-family graph -- the
  // only in-tree consumer of this type -- cannot be built under engine=gpu at
  // all.
  if (registerGeGLUClKernels(*this)) {
    registerFactory(nntrainer::createLayer<GeGLULayer>, GeGLULayer::type);
  } else {
    ml_logw("failed to register the OpenCL GeGLU kernels");
  }
}

template <typename T>
const int ClContext::registerFactory(const FactoryType<T> factory,
                                     const std::string &key,
                                     const int int_key) {
  static_assert(isSupported<T>::value,
                "cl_context: given type is not supported for current context");

  auto &index = std::get<IndexType<T>>(factory_map);
  auto &str_map = std::get<StrIndexType<T>>(index);
  auto &int_map = std::get<IntIndexType>(index);

  std::string assigned_key = key == "" ? factory({})->getType() : key;

  std::transform(assigned_key.begin(), assigned_key.end(), assigned_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::lock_guard<std::mutex> lock(cl_factory_mutex);
  if (str_map.find(assigned_key) != str_map.end()) {
    std::stringstream ss;
    ss << "cl_context: cannot register factory with already taken key: " << key;
    throw std::invalid_argument(ss.str().c_str());
  }

  if (int_key != -1 && int_map.find(int_key) != int_map.end()) {
    std::stringstream ss;
    ss << "cl_context: cannot register factory with already taken int key: "
       << int_key;
    throw std::invalid_argument(ss.str().c_str());
  }

  // An auto-assigned key is str_map.size() + 1, which is not free: an explicit
  // key taken from ml::train::LayerType sits in the same map, so inserting a
  // string-keyed factory ahead of the explicit ones shifts every later
  // auto-key onto one of them. The int_map write then silently replaced a
  // registration instead of failing, and the type it displaced simply stopped
  // resolving. Skip past what is taken rather than overwrite it, and keep the
  // explicit-key branch throwing, which is the caller's own mistake.
  int assigned_int_key = int_key;
  if (assigned_int_key == -1) {
    assigned_int_key = static_cast<int>(str_map.size()) + 1;
    while (int_map.find(assigned_int_key) != int_map.end())
      ++assigned_int_key;
  }

  str_map[assigned_key] = factory;
  int_map[assigned_int_key] = assigned_key;

  ml_logd("cl_context: factory has registered with key: %s, int_key: %d",
          assigned_key.c_str(), assigned_int_key);

  return assigned_int_key;
}

void ClContext::initBlasClKernels() {
  if (blas_kernels_initialized) {
    ml_logi(
      "ClContext: Default blas kernels already registered and initialized");
    return;
  }

  registerClKernel(sgemv_kernel, "sgemv_cl");
  registerClKernel(sgemv_no_trans_kernel, "sgemv_cl_noTrans");
  registerClKernel(dot_kernel, "dot_cl");
  registerClKernel(sgemm_no_trans_kernel, "sgemm_cl_noTrans");
  registerClKernel(sgemm_trans_a_kernel, "sgemm_cl_transA");
  registerClKernel(sgemm_trans_b_kernel, "sgemm_cl_transB");
  registerClKernel(sgemm_trans_ab_kernel, "sgemm_cl_transAB");
  registerClKernel(addition_kernel, "addition_cl");
  registerClKernel(sscal_kernel, "sscal_cl");
  registerClKernel(q6_k_sgemv_kernel, "kernel_mul_mv_q6_K_f32");

  // register Q4_0 kernels
  registerClKernel(convert_block_q4_0_kernel,
                   "kernel_convert_block_q4_0_noshuffle");
  registerClKernel(restore_block_q4_0_kernel, "kernel_restore_block_q4_0");
  registerClKernel(transpose_16bit_kernel, "kernel_transpose_16");
  registerClKernel(transpose_32bit_16bit_kernel, "kernel_transpose_32_16");
  registerClKernel(q4_0_ab_bi_8x4_kernel, "kernel_mul_mat_Ab_Bi_8x4");

  // register INT4 computation kernels
  registerClKernel(int4_gemv_kernel, "fully_connected_gpu_int4_gemv");
  registerClKernel(int4_quantize_input_kernel, "quantize_input_int4");
  registerClKernel(int4_quantize_input_kernel, "quantize_input_int4_pad");

#ifdef ENABLE_FP16
  registerClKernel(hgemv_kernel, "sgemv_cl_fp16");
  registerClKernel(hgemv_no_trans_kernel, "sgemv_cl_noTrans_fp16");
  registerClKernel(dot_fp16_kernel, "dot_cl_fp16");
  registerClKernel(hgemm_no_trans_kernel, "sgemm_cl_noTrans_fp16");
  registerClKernel(hgemm_trans_a_kernel, "sgemm_cl_transA_fp16");
  registerClKernel(hgemm_trans_b_kernel, "sgemm_cl_transB_fp16");
  registerClKernel(hgemm_trans_ab_kernel, "sgemm_cl_transAB_fp16");
  registerClKernel(addition_fp16_kernel, "addition_cl_fp16");
  registerClKernel(hscal_kernel, "sscal_cl_fp16");
#endif
  blas_kernels_initialized = true;
}

void ClContext::initAttentionClKernels() {
  if (attention_kernels_initialized) {
    ml_logi("ClContext: Default attention kernels already registered and "
            "initialized");
    return;
  }

  registerClKernel(rotary_emb_kernel, "rotary_emb_cl");

#ifdef ENABLE_FP16
  registerClKernel(rotary_emb_fp16_kernel, "rotary_emb_cl_fp16");
#endif

  // Programs that would otherwise be built on their first dispatch, i.e.
  // inside the first prefill and the first decode step. Collected by the
  // translation unit that owns their sources AND the compile options its
  // dispatch passes, because a prewarm with the wrong options builds a
  // program the hot path never looks up: it pays the compile twice and
  // removes nothing from the critical path.
  {
    std::vector<std::function<void()>> lazy_tasks;
    v8c_collect_lazy_program_tasks(*this, lazy_tasks);
    for (auto &t : lazy_tasks)
      t();
  }

  // The attention dispatchers own a second family of lazily built programs:
  // the in-place RoPE source also hosts the KV scatter/copy kernels, so one
  // registration builds the program all of them share.
  attention_prewarm_programs(*this);

  attention_kernels_initialized = true;
}

const ClContext::SharedPtrClKernel
ClContext::registerClKernel(const std::string &kernel_string,
                            const std::string &kernel_name,
                            const std::string &compile_options) {
  // check if created before. One key construction, one lookup: the previous
  // by-value parameters copied the whole kernel source -- tens of KB -- on
  // every cached lookup, and the attention path takes this route once per
  // kernel per layer.
  const std::string key = kernel_name + compile_options;

  // Kernel ring-rotation: hand out one of K rotating CLONES of each kernel
  // rather than a single process-global object. Every dispatcher re-binds
  // arguments on the object this returns, and with a singleton that re-bind
  // can land on an object whose previous enqueue the driver has not locked
  // in yet -- a token-altering hazard we have measured on this stack, which
  // a per-dispatch flush only makes rarer. Rotating guarantees the object
  // being re-bound is the one enqueued K calls ago. The cost is K-1 extra
  // clCreateKernel per kernel, which the driver's program cache makes cheap,
  // and nothing per call. A call site that caches the returned pointer in a
  // static keeps singleton behaviour; that gap is deliberate and documented
  // here rather than papered over.
  //
  // This is a correctness fix, so it is unconditional and does not sit under
  // the NNTR_DETERMINISTIC opt-out, which relaxes only the ordering and
  // reduction-shape half of the determinism contract.
  // NNTR_CL_KERNEL_RING=K is the explicit diagnostic override; K=1
  // reproduces the old singleton deliberately, as a bisection aid.
  {
    static const int ring_k = []() {
      const char *r = std::getenv("NNTR_CL_KERNEL_RING");
      if (r == nullptr)
        return 8;
      // strtol, not atoi: atoi maps every non-numeric string to 0 and has no
      // way to report one, so a typo would silently select the singleton.
      char *end = nullptr;
      errno = 0;
      const long v = std::strtol(r, &end, 10);
      if (errno != 0 || end == r || *end != '\0' || v < 1 || v > 64) {
        ml_logw("Ignoring NNTR_CL_KERNEL_RING=%s: expected an integer in "
                "[1, 64]",
                r);
        return 8;
      }
      return (int)v;
    }();
    if (ring_k > 1) {
      static std::unordered_map<
        std::string, std::pair<std::vector<SharedPtrClKernel>, size_t>>
        ring_map;
      auto &slot = ring_map[key];
      auto &clones = slot.first;
      if ((int)clones.size() < ring_k) {
        std::string ks = kernel_string, kn = kernel_name, co = compile_options;
        SharedPtrClKernel kp = std::make_shared<opencl::Kernel>();
        if (clCreateKernel(ks, kn, co, kp)) {
          clones.push_back(kp);
          return clones.back();
        }
        // Clone creation failed: fall through to the single-object path.
      } else {
        slot.second = (slot.second + 1) % clones.size();
        return clones[slot.second];
      }
    }
  }

  // ocl_kernel_map is a process-wide static reached from the per-op dispatch
  // path, not only from init, so it takes the same treatment clCreateKernel
  // gives program_cache one frame below. Leaving the outer map unguarded while
  // the inner one is locked is not a lost cache hit but a data race on an
  // unordered_map, and therefore undefined behaviour. It gets its own lock
  // because clCreateKernel takes program_cache_mtx while this one is not held,
  // and clCreateKernel never calls back in here, so there is no reentrancy.
  {
    const std::lock_guard<std::mutex> lock(ocl_kernel_map_mutex);
    auto it = ocl_kernel_map.find(key);
    if (it != ocl_kernel_map.end())
      return it->second;
  }

  // Built outside the lock: a cold compile can take hundreds of milliseconds
  // and holding the map across it would serialise every other lookup behind
  // it. Two threads racing on the same cold key both compile, and try_emplace
  // below keeps the first one home -- wasted work on a rare race, never a torn
  // map.
  //
  // clCreateKernel takes mutable references, so the cold path makes the copies
  // it needs.
  std::string source = kernel_string;
  std::string name = kernel_name;
  std::string options = compile_options;
  SharedPtrClKernel kernelPtr = std::make_shared<opencl::Kernel>();
  if (!clCreateKernel(source, name, options, kernelPtr)) {
    ml_loge("Failed to register kernel %s", kernel_name.c_str());
    return nullptr;
  }

  const std::lock_guard<std::mutex> lock(ocl_kernel_map_mutex);
  return ocl_kernel_map.try_emplace(key, kernelPtr).first->second;
}

bool ClContext::clCreateKernel(std::string &kernel_string,
                               std::string &kernel_name,
                               std::string &compile_options,
                               const SharedPtrClKernel &kernel_ptr_) {

  ml_logi("Kernel initializing: %s", kernel_name.c_str());

  bool result = false;

  opencl::Program program;

  // In-memory program cache: kernels that share one source and one option
  // string share the built cl_program. Without it every kernel of a
  // multi-kernel source repeats the binary read and the
  // clCreateProgramWithBinary that goes with it, all of it inside the first
  // forward pass.
  static std::unordered_map<std::string, opencl::Program> program_cache;
  static std::mutex program_cache_mtx;
  const std::string pc_key =
    std::to_string(program.GetKernelHash(kernel_string, "")) + "|" +
    compile_options;
  {
    std::lock_guard<std::mutex> lk(program_cache_mtx);
    auto it = program_cache.find(pc_key);
    if (it != program_cache.end()) {
      nntrainer::load_trace::Scope _lt(nntrainer::load_trace::KRN_OBJ);
      return kernel_ptr_->CreateKernelFromProgram(it->second, kernel_name);
    }
  }

  // On-disk kernel binary cache. The key folds in the per-kernel
  // compile_options and the device signature (name + driver version): a stored
  // binary is only valid for the exact source and options it was built from,
  // and only on the same GPU and driver. Keying on the source alone hands a
  // binary built for another device to clCreateProgramWithBinary.
  static const std::string device_sig =
    opencl::ContextManager::Global().GetDeviceSignature();
  const std::string binary_file_name =
    std::to_string(program.GetKernelHash(kernel_string,
                                         compile_options + "|" + device_sig)) +
    ".cl.bin";
  const std::string binary_file_path =
    kernelCacheDir() + "/" + binary_file_name;

  /** ---- Negative cache: a kernel that cannot be built is not free. --------
   *
   *  Measured on a mobile GPU handset with a FULLY warm kernel cache: 76-114 ms
   *  of every init -- a third of the whole non-load init -- is six
   *  clCreateProgramWithSource+clBuildProgram calls that all end in "Failed to
   *  register kernel". They are three legacy INT4 kernels
   *  (fully_connected_gpu_int4_gemv, quantize_input_int4,
   *  quantize_input_int4_pad) that this device declines, prewarmed by
   *  initBlasClKernels for a dispatch path the quantized GEMM lane never takes,
   *  and they are six rather than three because the kernel ring's clone attempt
   *  and the singleton path below it each pay the same compile for the same
   *  deterministic failure. Nothing observed it, because a null kernel is only
   *  ever a silent decline here.
   *
   *  A failure is as cacheable as a success and on exactly the same key -- the
   *  source, its compile options, the device and its driver -- plus the kernel
   *  NAME, because clCreateKernel can fail for one name in a program that
   *  built. Record it in-process (which is what collapses the ring's double
   *  attempt) and as an empty marker file next to the binary cache (which is
   *  what makes the next launch free). A driver update, a changed source or a
   *  changed option string all move the key, so a marker can never outlive the
   *  thing it describes; wiping the cache directory clears them like any other
   *  entry.
   *
   *  NNTR_KERNEL_NEG_CACHE=0 turns it off and restores the retry-every-launch
   *  behaviour in the same binary.
   *  ---------------------------------------------------------------------- */
  static const bool neg_cache_on = []() {
    const char *e = std::getenv("NNTR_KERNEL_NEG_CACHE");
    return !(e != nullptr && e[0] == '0');
  }();
  const std::string neg_key = binary_file_name + "|" + kernel_name;
  const std::string neg_file_path =
    kernelCacheDir() + "/" + binary_file_name + "." + kernel_name + ".fail";
  static std::unordered_set<std::string> failed_kernels;
  static std::mutex failed_kernels_mtx;
  auto note_failure = [&]() {
    if (!neg_cache_on)
      return;
    {
      const std::lock_guard<std::mutex> lk(failed_kernels_mtx);
      failed_kernels.insert(neg_key);
    }
    if (KERNEL_CACHE_ENABLED && kernel_cache_usable) {
      std::ofstream fs(neg_file_path, std::ios::out | std::ios::binary);
      if (!fs)
        ml_logw("Could not record the failed build of %s at %s; it will be "
                "retried on the next launch",
                kernel_name.c_str(), neg_file_path.c_str());
    }
  };
  if (neg_cache_on) {
    bool known_bad = false;
    {
      const std::lock_guard<std::mutex> lk(failed_kernels_mtx);
      known_bad = failed_kernels.count(neg_key) != 0;
    }
    if (!known_bad && KERNEL_CACHE_ENABLED && kernel_cache_usable &&
        std::filesystem::exists(neg_file_path)) {
      known_bad = true;
      const std::lock_guard<std::mutex> lk(failed_kernels_mtx);
      failed_kernels.insert(neg_key);
    }
    if (known_bad) {
      ml_logd("Kernel %s failed to build here before; not rebuilding it",
              kernel_name.c_str());
      return false;
    }
  }
  std::vector<std::byte> binary_data;
  if (KERNEL_CACHE_ENABLED && kernel_cache_usable) {
    nntrainer::load_trace::Scope _lt(nntrainer::load_trace::KRN_BIN_READ);
    binary_data = readBinaryFile(binary_file_path);
    _lt.bytes(binary_data.size());
  }
  // Where the bytes actually came from. The log line used to name the resolved
  // directory whatever the answer was, which reads as a cache that is whole
  // when it is in fact split across two directories.
  std::string binary_read_path = binary_file_path;
  bool served_from_legacy_dir = false;
  if (KERNEL_CACHE_ENABLED && kernel_cache_usable && binary_data.empty() &&
      kernelCacheDir() != opencl::Program::DEFAULT_KERNEL_PATH) {
    // Fall back to the legacy working-directory location so a cache written
    // before the directory was resolvable is still used.
    const std::string legacy_path =
      opencl::Program::DEFAULT_KERNEL_PATH + "/" + binary_file_name;
    binary_data = readBinaryFile(legacy_path);
    if (!binary_data.empty()) {
      binary_read_path = legacy_path;
      served_from_legacy_dir = true;
    }
  }

  bool loaded_from_binary = false;
  if (KERNEL_CACHE_ENABLED && kernel_cache_usable && !binary_data.empty()) {
    ml_logi("Using cached version of kernel: %s at path %s",
            kernel_name.c_str(), binary_read_path.c_str());
    {
      nntrainer::load_trace::Scope _lt(nntrainer::load_trace::KRN_PROG_BIN);
      _lt.bytes(binary_data.size());
      loaded_from_binary = program.CreateCLProgramWithBinary(
        opencl::ContextManager::Global().GetContext(),
        opencl::ContextManager::Global().GetDeviceId(), binary_data,
        binary_read_path, "");
    }
    if (!loaded_from_binary) {
      ml_logw("Cached kernel binary %s was rejected; recompiling from source",
              binary_read_path.c_str());
    } else if (served_from_legacy_dir) {
      // Promote the hit into the resolved directory. The legacy location is
      // the process working directory, so an entry that lives only there is
      // lost the moment the application is started from anywhere else -- and
      // nothing ever writes it back, because the fallback is read-only by
      // design. Copying the accepted binary across costs one small write on
      // the launch that found it and keeps the cache whole afterwards. The
      // key already folds in the device name and driver version, so this
      // moves a binary that this device just accepted, never a foreign one.
      if (!writeBinaryFile(binary_file_path, binary_data))
        ml_logw("Could not promote kernel cache entry %s to %s; continuing",
                binary_read_path.c_str(), binary_file_path.c_str());
    }
  }

  if (loaded_from_binary) {
    result = true;
  } else {
    ml_logi("Binary for kernel %s not found, compiling from source...",
            kernel_name.c_str());
    {
      nntrainer::load_trace::Scope _lt(nntrainer::load_trace::KRN_PROG_SRC);
      result =
        program.CreateCLProgram(opencl::ContextManager::Global().GetContext(),
                                opencl::ContextManager::Global().GetDeviceId(),
                                kernel_string, compile_options);
    }

    if (KERNEL_CACHE_ENABLED && kernel_cache_usable && result) {
      // Best-effort cache write: the freshly compiled program is already
      // usable, so failing to persist it is a warning, not a build failure.
      auto binary = program.GetProgramBinary(
        opencl::ContextManager::Global().GetDeviceId());

      if (binary.empty()) {
        ml_logw("Failed retrieving binary for kernel %s; skipping cache write",
                kernel_name.c_str());
      } else if (!writeBinaryFile(binary_file_path, binary)) {
        ml_logw("Failed writing kernel cache %s; continuing",
                binary_file_path.c_str());
      }
    }
  }

  if (!result) {
    note_failure();
    return false;
  }

  {
    std::lock_guard<std::mutex> lk(program_cache_mtx);
    program_cache.emplace(pc_key, program);
  }

  {
    nntrainer::load_trace::Scope _lt(nntrainer::load_trace::KRN_OBJ);
    result = kernel_ptr_->CreateKernelFromProgram(program, kernel_name);
  }

  // The program built; this NAME is not in it, or the device refused the
  // kernel object. Either way the next attempt gets the same answer.
  if (!result)
    note_failure();

  return result;
}

/**
 * @copydoc const int ClContext::registerFactory
 */
template const int ClContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

// Non-template seam (Context::registerLayerFactory override): forwards to the
// per-class registerFactory<Layer> here in the same translation unit, so the
// explicit instantiation above is the one used and no template crosses the .so
// boundary.
int ClContext::registerLayerFactory(PtrFactoryType<nntrainer::Layer> factory,
                                    const std::string &key, const int int_key) {
  return registerFactory<nntrainer::Layer>(factory, key, int_key);
}

} // namespace nntrainer

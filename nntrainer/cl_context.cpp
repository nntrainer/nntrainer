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
#include <cl_context.h>
#include <cl_kernels/cl_kernels.h>
#include <cl_svm_allocator.h>
#include <compute_ops.h>
#include <concat_cl.h>
#include <fc_layer_cl.h>
#include <geglu_cl_op.h>
#include <gelu_cl_op.h>
#include <layer_normalization_layer.h>
#include <layernorm_cl_op.h>
#include <opencl_context_manager.h>
#include <reshape_cl.h>
#include <rmsnorm_layer_cl.h>
#include <swiglu_cl_op.h>
#include <swiglu_layer.h>
#include <transpose_cl.h>

#include <filesystem>
#include <mutex>
#include <system_error>
#include <unordered_map>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace nntrainer {
#if KERNEL_CACHE
static constexpr bool KERNEL_CACHE_ENABLED = true;
#else
static constexpr bool KERNEL_CACHE_ENABLED = false;
#endif
std::mutex cl_factory_mutex;

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
    if (!clInit()) {
      ml_loge("Error: ClContext::initialize() failed");
      return;
    }
    if (KERNEL_CACHE_ENABLED) {
      // Best effort: the binary cache is an optimisation. A read-only or
      // otherwise unwritable working directory must not take the whole
      // context down with it -- create_directories throws
      // std::filesystem::filesystem_error, and everything below (the kernel
      // registration, the memory allocator, the ops table) would then be
      // skipped by the catch at the end of this function, leaving a context
      // that registers no layer and hands out no allocator.
      std::error_code ec;
      std::filesystem::create_directories(opencl::Program::DEFAULT_KERNEL_PATH,
                                          ec);
      if (ec) {
        ml_logw("Could not create the kernel cache directory %s (%s); "
                "compiling kernels from source without caching them",
                opencl::Program::DEFAULT_KERNEL_PATH.c_str(),
                ec.message().c_str());
      }
    }

    initBlasClKernels();
    initAttentionClKernels();

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

  // GeGLU registers its kernels and no factory. The neutral GeGLULayer is
  // registered on the application context under a string key only, and it
  // reaches ComputeOps::geglu through its input tensor, so all this context
  // owes it is a compiled kernel. Adding a factory here would need an explicit
  // integer key as well -- the auto-assigned one is str_map.size() + 1 and
  // collides with an enum key -- so that belongs with the layer's promotion,
  // not with its kernel.
  if (!registerGeGLUClKernels(*this))
    ml_logw("failed to register the OpenCL GeGLU kernels");
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

  auto it = ocl_kernel_map.find(key);
  if (it != ocl_kernel_map.end())
    return it->second;

  // creating shared_ptr for kernel object. clCreateKernel takes mutable
  // references, so the cold path makes the copies it needs.
  std::string source = kernel_string;
  std::string name = kernel_name;
  std::string options = compile_options;
  SharedPtrClKernel kernelPtr = std::make_shared<opencl::Kernel>();
  if (!clCreateKernel(source, name, options, kernelPtr)) {
    ml_loge("Failed to register kernel %s", kernel_name.c_str());
    return nullptr;
  }
  // add to map
  ocl_kernel_map.emplace(key, kernelPtr);
  return ocl_kernel_map[key];
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
    if (it != program_cache.end())
      return kernel_ptr_->CreateKernelFromProgram(it->second, kernel_name);
  }

  // On-disk kernel binary cache. The key folds in the per-kernel
  // compile_options and the device signature (name + driver version): a stored
  // binary is only valid for the exact source and options it was built from,
  // and only on the same GPU and driver. Keying on the source alone hands a
  // binary built for another device to clCreateProgramWithBinary.
  static const std::string device_sig =
    opencl::ContextManager::Global().GetDeviceSignature();
  std::string binary_file_path =
    opencl::Program::DEFAULT_KERNEL_PATH + "/" +
    std::to_string(program.GetKernelHash(kernel_string,
                                         compile_options + "|" + device_sig)) +
    ".cl.bin";
  auto binary_data = KERNEL_CACHE_ENABLED ? readBinaryFile(binary_file_path)
                                          : std::vector<std::byte>();

  bool loaded_from_binary = false;
  if (KERNEL_CACHE_ENABLED && !binary_data.empty()) {
    ml_logi("Using cached version of kernel: %s at path %s",
            kernel_name.c_str(), binary_file_path.c_str());
    loaded_from_binary = program.CreateCLProgramWithBinary(
      opencl::ContextManager::Global().GetContext(),
      opencl::ContextManager::Global().GetDeviceId(), binary_data,
      binary_file_path, "");
    if (!loaded_from_binary)
      ml_logw("Cached kernel binary %s was rejected; recompiling from source",
              binary_file_path.c_str());
  }

  if (loaded_from_binary) {
    result = true;
  } else {
    ml_logi("Binary for kernel %s not found, compiling from source...",
            kernel_name.c_str());
    result =
      program.CreateCLProgram(opencl::ContextManager::Global().GetContext(),
                              opencl::ContextManager::Global().GetDeviceId(),
                              kernel_string, compile_options);

    if (KERNEL_CACHE_ENABLED && result) {
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
    return false;
  }

  {
    std::lock_guard<std::mutex> lk(program_cache_mtx);
    program_cache.emplace(pc_key, program);
  }

  result = kernel_ptr_->CreateKernelFromProgram(program, kernel_name);

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

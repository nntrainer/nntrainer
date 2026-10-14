/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	neuralnet.cpp
 * @date	04 December 2019
 * @brief	This is Neural Network Class
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "layer_context.h"
#include "model.h"
#include "model_common_properties.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <compute_ops.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <load_trace.h>
#if defined(ENABLE_OPENCL)
#include <blas_kernel_interface.h> // cl_fc_release_caches
#endif
#include <mutex>
#include <sstream>
#include <thread>

#if defined(__GLIBC__) && !defined(_WIN32)
#include <malloc.h> // malloc_trim: return the loader's freed transients to the OS
#endif

#if !defined(_WIN32)
#include <fcntl.h> // posix_fadvise: drop the weight file's page cache after the load
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h> // cuda::drain_if_async
#include <cuda_fc_qs4cx.h>        // [wprefetch] cuda_fc_qs4cx_prefetch_weight
#endif

#include <activation_realizer.h>
#include <adamw.h>
#include <common_properties.h>
#include <databuffer.h>
#include <flatten_realizer.h>
#include <fp16.h>
#include <fusion_realizer.h>
#include <ini_interpreter.h>
#include <ini_wrapper.h>
#include <input_realizer.h>
#include <int4_utils.h>
#include <model_loader.h>
#include <multiout_realizer.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <optimizer_context.h>
#include <optional>
#include <previous_input_realizer.h>
#include <profiler.h>
#include <qs4cx_tensor.h>
#include <recurrent_realizer.h>
#include <remap_realizer.h>
#include <safetensors_util.h>
#include <slice_realizer.h>
#include <tq2cx_layout.h>
#include <util_func.h>

#ifdef ENABLE_TFLITE_INTERPRETER
#include <tflite_interpreter.h>
#endif

/**
 * @brief Internal enum values for nntrainer to summarize model accuracy & loss
 */
#define ML_TRAIN_SUMMARY_MODEL_TRAIN_LOSS 101
#define ML_TRAIN_SUMMARY_MODEL_VALID_LOSS 102
#define ML_TRAIN_SUMMARY_MODEL_VALID_ACCURACY 103

namespace nntrainer {

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
// Derive-once GPU weight-pack cache
// (tensor/cl_operations/v8c_pack_cache.h). Declared here rather than included
// so this file stays free of the OpenCL headers; only these two calls, both
// of which take plain arguments, are needed from the loader.
namespace v8c_pack {
void set_source(const char *model_bin_path);
void load_complete();
} // namespace v8c_pack
// Submit-and-go upload staging queued by the v8c weight build
// (tensor/cl_operations/blas_kernels.h). Declared here for the same reason.
void v8c_flush_pending_uploads();
void v8c_flush_aux_arena();
void v8c_open_aux_arena();
void debug_dump_rows_cl(const std::string &name, const Tensor &t,
                        unsigned int rows, const char *dir, unsigned int seq);
#endif

namespace {

Tensor mapExternalTensor(float *buf, const TensorDim &dim) {
  const unsigned int bytes = static_cast<unsigned int>(
    static_cast<size_t>(dim.getDataLen()) * dim.getDataTypeSize());

  switch (dim.getDataType()) {
  case TensorDim::DataType::FP16:
  case TensorDim::DataType::UINT16:
  case TensorDim::DataType::QINT16:
    return Tensor::Map<uint16_t>(reinterpret_cast<uint16_t *>(buf), bytes, dim,
                                 0);
  case TensorDim::DataType::UINT8:
  case TensorDim::DataType::UINT4:
  case TensorDim::DataType::QINT8:
  case TensorDim::DataType::QINT4:
  case TensorDim::DataType::QS4CX:
  case TensorDim::DataType::Q4_K:
  case TensorDim::DataType::Q6_K:
  case TensorDim::DataType::Q4_0:
  case TensorDim::DataType::TQ2CX:
  case TensorDim::DataType::QS4ACX:
    return Tensor::Map<uint8_t>(reinterpret_cast<uint8_t *>(buf), bytes, dim,
                                0);
  case TensorDim::DataType::UINT32:
  case TensorDim::DataType::BCQ:
    return Tensor::Map<uint32_t>(reinterpret_cast<uint32_t *>(buf), bytes, dim,
                                 0);
  case TensorDim::DataType::FP32:
  case TensorDim::DataType::NONE:
  default:
    return Tensor::Map<float>(buf, bytes, dim, 0);
  }
}

/**
 * @brief [#150] NNTR_OP_TIME=1: wall time of every node's decode call
 *        (to - from == 1), one line per node at exit, for
 *        tools/htp/op_time_report.py. Off, it costs one predicted branch
 *        per node and never touches a tensor. Unlike the --profile build it
 *        builds no string per call and starts no sampler thread, so it can
 *        ride the TPS binary.
 *
 * ponytail: indexed by execution order, one process-wide table. Two models
 * in one process would add into the same rows; the CausalLM app runs one.
 */
struct OpTimeTable {
  struct Row {
    std::string name, type;
    uint64_t calls = 0, sum_ns = 0, min_ns = UINT64_MAX, max_ns = 0;
    size_t wbytes = 0;
  };
  std::vector<Row> rows;

  static bool on() {
    static const bool v = [] {
      const char *e = std::getenv("NNTR_OP_TIME");
      return e != nullptr && std::strcmp(e, "1") == 0;
    }();
    return v;
  }

  void add(size_t f, LayerNode &node, uint64_t ns) {
    if (f >= rows.size())
      rows.resize(f + 1);
    Row &r = rows[f];
    if (r.calls++ == 0) {
      r.name = node.getName();
      r.type = node.getType();
      for (unsigned int i = 0; i < node.getNumWeights(); ++i)
        r.wbytes += node.getWeight(i).getMemoryBytes();
    }
    r.sum_ns += ns;
    r.min_ns = std::min(r.min_ns, ns);
    r.max_ns = std::max(r.max_ns, ns);
  }

  ~OpTimeTable() {
    for (size_t f = 0; f < rows.size(); ++f) {
      const Row &r = rows[f];
      if (r.calls == 0)
        continue;
      std::fprintf(stderr,
                   "[OP-TIME] node f=%zu name=%s type=%s calls=%llu "
                   "sum_us=%.3f min_us=%.3f max_us=%.3f wbytes=%zu\n",
                   f, r.name.c_str(), r.type.c_str(),
                   static_cast<unsigned long long>(r.calls), r.sum_ns / 1e3,
                   r.min_ns / 1e3, r.max_ns / 1e3, r.wbytes);
    }
  }
};

OpTimeTable g_op_time;

/**
 * @brief NNTR_ACT_STATS=1: after every node of the first prefill (from == 0,
 *        to - from > 1), one stderr line per FP32 output -- NaN and Inf
 *        counts, max |x| and rms over the rows that call wrote -- so an
 *        all-NPU run diffed against the device's own CPU (NEON) run names
 *        the first node whose output leaves it (doc 57 section 9.20).
 *        Never in a tok/s run: a full read of every output.
 */
void actStats(LayerNode &node, unsigned int rows) {
  static const bool on = [] {
    const char *e = std::getenv("NNTR_ACT_STATS");
    return e != nullptr && std::strcmp(e, "1") == 0;
  }();
  if (!on)
    return;
  for (unsigned int i = 0; i < node.getNumOutputs(); ++i) {
    const Tensor &t = node.getOutput(i);
    if (t.empty() || t.getDataType() != TensorDim::DataType::FP32)
      continue;
    const TensorDim &d = t.getDim();
    const size_t n =
      std::min<size_t>(rows, d.height()) * d.width() * d.channel();
    const float *p = t.getData<float>();
    // rms over the finite values and over the first row alone, so a CPU
    // run and an accelerator run of the same prompt line up node by node
    const size_t row = static_cast<size_t>(d.width()) * d.channel();
    size_t nan = 0, inf = 0;
    float mx = 0.0f;
    double ss = 0.0, ss0 = 0.0;
    for (size_t k = 0; k < n; ++k) {
      const float v = p[k];
      if (std::isnan(v)) {
        ++nan;
      } else if (std::isinf(v)) {
        ++inf;
      } else {
        mx = std::max(mx, std::fabs(v));
        ss += static_cast<double>(v) * v;
        if (k < row)
          ss0 += static_cast<double>(v) * v;
      }
    }
    const size_t fin = n - nan - inf;
    std::fprintf(stderr,
                 "[ACT] %s out%u n=%zu nan=%zu inf=%zu maxabs=%g rms=%g "
                 "rms_row0=%g\n",
                 node.getName().c_str(), i, n, nan, inf, mx,
                 fin ? std::sqrt(ss / fin) : 0.0,
                 row ? std::sqrt(ss0 / std::min(row, n ? n : 1)) : 0.0);
  }
}

} // namespace

NeuralNetwork::NeuralNetwork() :
  model_props(props::LossType(), {}, {}, props::ClipGradByGlobalNorm(),
              props::LossScale()),
  model_flex_props(props::Epochs(), props::TrainingBatchSize(),
                   props::SavePath(), props::ContinueTrain(),
                   props::SaveBestPath(), props::MemoryOptimization(),
                   props::Fsu(), props::FsuPath(), props::FsuLookahead(),
                   props::TensorFormat(), props::ModelTensorDataType()),
  load_path(std::string()),
  epoch_idx(0),
  iter(0),
  loss(0.0f),
  data_buffers({nullptr, nullptr, nullptr}),
  initialized(false),
  compiled(false),
  loadedFromConfig(false),
  exec_mode(ExecutionMode::TRAIN),
  ct_engine(&Engine::Global()) {}

NeuralNetwork::NeuralNetwork(const Engine *ct_engine_) :
  model_props(props::LossType(), {}, {}, props::ClipGradByGlobalNorm(),
              props::LossScale()),
  model_flex_props(props::Epochs(), props::TrainingBatchSize(),
                   props::SavePath(), props::ContinueTrain(),
                   props::SaveBestPath(), props::MemoryOptimization(),
                   props::Fsu(), props::FsuPath(), props::FsuLookahead(),
                   props::TensorFormat(), props::ModelTensorDataType()),
  load_path(std::string()),
  epoch_idx(0),
  iter(0),
  loss(0.0f),
  data_buffers({nullptr, nullptr, nullptr}),
  initialized(false),
  compiled(false),
  loadedFromConfig(false),
  exec_mode(ExecutionMode::TRAIN),
  ct_engine(ct_engine_) {}

int NeuralNetwork::loadFromConfig(const std::string &config) {
  if (loadedFromConfig == true) {
    ml_loge("can not do loadFromConfig twice");
    return ML_ERROR_INVALID_PARAMETER;
  }

  ModelLoader loader(ct_engine);
  NeuralNetwork tempNet(*this);

  int status = loader.loadFromContext(tempNet);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  status = loader.loadFromConfig(config, tempNet);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  tempNet.loadedFromConfig = true;
  swap(tempNet, *this);

  return ML_ERROR_NONE;
}

unsigned int NeuralNetwork::getCurrentEpoch() {
#ifdef DEBUG
  ml_logd("[NNTrainer] Current epoch: %d", epoch_idx);
#endif
  return epoch_idx;
};

void NeuralNetwork::setProperty(const std::vector<std::string> &values) {
  auto left_props = loadProperties(values, model_props);
  setTrainConfig(left_props);
}

void NeuralNetwork::setTrainConfig(const std::vector<std::string> &values) {
  auto left_props = loadProperties(values, model_flex_props);
  NNTR_THROW_IF(left_props.size(), std::invalid_argument)
    << "Model has unparsed properties, size: " << left_props.size()
    << " of first element: " << left_props.front();
}

int NeuralNetwork::compile(ExecutionMode mode) {

  exec_mode = mode;

  // The decode seam caches the Context it resolved from the graph's engines.
  // Compiling replaces the graph, so a cached answer from a previous compile
  // may name a backend this graph does not run on. Drop it and let the next
  // decode step resolve against the graph that actually exists.
  decode_ctx_ = nullptr;

  std::string loss_type = std::get<props::LossType>(model_props).empty()
                            ? std::string()
                            : std::get<props::LossType>(model_props);

  auto &input_conn = std::get<std::vector<props::InputConnection>>(model_props);
  /// @note label layer might need to be treated in the similar way as well

  /// @todo make NetworkGraph compiled at the construction instead of having
  /// graph.compile(), neuralnetwork have ownership of list of layer nodes,
  /// which will be passed at compile time.

  std::vector<std::unique_ptr<GraphRealizer>> realizers;

  realizers.emplace_back(new PreviousInputRealizer(
    std::vector<Connection>(input_conn.begin(), input_conn.end())));
  realizers.emplace_back(new MultioutRealizer());
  realizers.emplace_back(new FlattenRealizer());
  // Fuse a compute layer's activation epilogue (conv+relu / fc+act) BEFORE
  // ActivationRealizer would split it into a node, so it stays inline. Gated
  // by NNTR_FUSE_ACT (default on) and restricted to inference: the fused
  // forward is value-identical to the standalone node, but there is no fused
  // backward, so a training graph keeps the separate ActivationLayer that
  // contributes the activation derivative. A no-op for graphs whose compute
  // layers carry no activation property.
  if (mode == ExecutionMode::INFERENCE)
    realizers.emplace_back(new FusionRealizer());
  realizers.emplace_back(new ActivationRealizer());

  for (auto &realizer : realizers) {
    graph_representation = realizer->realize(graph_representation);
  }

  bool fsu = std::get<props::Fsu>(model_flex_props);
  const std::string fsu_path = std::get<props::FsuPath>(model_flex_props);
  unsigned int lookahead = std::get<props::FsuLookahead>(model_flex_props);

  const std::string tensor_format =
    to_string(std::get<props::TensorFormat>(model_flex_props));

  const std::string tensor_type =
    to_string(std::get<props::ModelTensorDataType>(model_flex_props));

  bool has_qnn_engine = false;
  for (auto &node : graph_representation) {
    if (node->getComputeEngineType() == "qnn" ||
        node->getType() == "qnn_graph") {
      has_qnn_engine = true;
      break;
    }
  }

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  bool has_gpu_engine = false;
  for (auto &node : graph_representation) {
    if (node->isComputeEngineGPU()) {
      has_gpu_engine = true;
      break;
    }
  }
#endif

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  bool has_cuda_engine = false;
  for (auto &node : graph_representation) {
    if (node->isComputeEngineCUDA()) {
      has_cuda_engine = true;
      break;
    }
  }
#endif

  model_graph =
    NetworkGraph(fsu, mode, fsu_path, lookahead, tensor_format, tensor_type);

  // QNN/HTP graphs register their I/O tensors with the DSP via rpcmem_to_fd(),
  // which requires those buffers to be rpcmem (DMA/ION). Route ONLY the
  // activation pool to the "qnn" (rpcmem) allocator; the weight pool stays on
  // CPU. The nntrainer weight pool is NOT DSP-registered (QNN graph weights are
  // loaded by the QNN context loader), so routing weights to rpcmem too would
  // needlessly exhaust the scarce CMA pool — observed as rpcmem_to_fd failures
  // after a few generated tokens once the app UI's GPU dmabuf also draws on
  // CMA. Mirrors the upstream setComputeBackend("", "npu") tensor-only design
  // (here the QNN context registers under the name "qnn").
  if (has_qnn_engine)
    model_graph.setComputeBackend("", "qnn");
#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  // A graph with an OpenCL layer in it reads its tensors from the GPU, so
  // allocate them where the GPU can read them: the OpenCL allocator hands out
  // memory both sides address, and it is the allocator that can additionally
  // place a tensor in device memory when the memory planner asks for it.
  // Without this the pools stay on host memory and every GPU layer pays a copy
  // in and a copy out, whatever the planner decided.
  else if (has_gpu_engine)
    model_graph.setComputeBackend("gpu", "gpu");
#endif
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // Same argument for a graph with a CUDA layer in it: the CUDA allocator hands
  // out Unified Memory (cudaMallocManaged), addressable by host and device
  // alike, which is what lets CudaComputeOps::fc take its device arms at all.
  // Without this the pools stay on plain host memory, dev_accessible() is false
  // for input, weight and output, every device arm declines, and the FC falls
  // to the host Tensor::dot -- which has no QS4CX/int4 implementation, so a
  // quantized model does not merely run slowly on engine=cuda, it throws
  // "unsupported datatype" on its first FC.
  // NNTR_CUDA_UVM_POOL=0 forces the host allocator back (correct, host-only).
  else if (has_cuda_engine) {
    const char *uvm = std::getenv("NNTR_CUDA_UVM_POOL");
    if (!(uvm != nullptr && uvm[0] == '0'))
      model_graph.setComputeBackend("cuda", "cuda");
  }
#endif

  // QNN activation tensors are rpcmem-backed and registered with the DSP, so
  // their addresses must stay stable across decode tokens. Let inference()
  // reuse the pool (allocate once) instead of reallocating per call. CPU/GPU
  // keep the realloc-per-call behavior they need for correct tensor state.
  reuse_inference_tensor_pool_ = has_qnn_engine;

  model_graph.setMemoryOptimizations(
    std::get<props::MemoryOptimization>(model_flex_props));
  for (auto &node : graph_representation) {
    if (auto &prop = std::get<props::ClipGradByGlobalNorm>(model_props);
        !prop.empty()) {
      node->setProperty({"clip_grad_by_norm=" + to_string(prop)});
    }
    if (auto &prop = std::get<props::LossScale>(model_props); !prop.empty()) {
      node->setProperty({"loss_scale=" + to_string(prop)});
    }
    model_graph.addLayer(node);
  }

  int status = model_graph.compile(loss_type);
  NN_RETURN_STATUS();

  compiled = true;

  return status;
}

int NeuralNetwork::initialize(ExecutionMode mode) {
  int status = ML_ERROR_NONE;

  if (mode != exec_mode) {
    if (mode == ExecutionMode::INFERENCE) {
      ml_logd("Execution mode mismatch : train mode @compile & inference mode "
              "@ initialize");
      exec_mode = mode;
    } else {
      NNTR_THROW_IF(exec_mode == ExecutionMode::TRAIN, std::invalid_argument)
        << "Execution mode mismatch : trying to train with compiled for "
           "inference";
    }
  }

  if (initialized) {
    ml_loge("Error: Initializing the model again");
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (!compiled) {
    ml_loge("Error: Need to compile first");
    return ML_ERROR_NOT_SUPPORTED;
  }

  unsigned int n_layers = (unsigned int)model_graph.size();

  ml_logd("initializing neural network, layer size: %d", n_layers);
  PROFILE_MEM_ANNOTATE("Initialize");

  auto &input_conn_prop =
    std::get<std::vector<props::InputConnection>>(model_props);
  auto &label_layer_prop =
    std::get<std::vector<props::LabelLayer>>(model_props);

  std::vector<Connection> input_conn(input_conn_prop.begin(),
                                     input_conn_prop.end());
  std::vector<std::string> label_layers;

  if (!label_layer_prop.empty()) {
    label_layers = std::vector<std::string>(label_layer_prop.begin(),
                                            label_layer_prop.end());
  }

  status = model_graph.initialize(
    exec_mode, input_conn,
    std::vector<Connection>(label_layers.begin(), label_layers.end()));
  NN_RETURN_STATUS();

  model_graph.setBatchSize(
    std::get<props::TrainingBatchSize>(model_flex_props));

  // If the execution mode is `train`, the optimizer and its relevant variables
  // are initialized. Throws an error if the optimizer is not set for training;
  // otherwise, it initializes
  if (exec_mode == ExecutionMode::TRAIN) {

    if (!opt) {
      ml_loge("Optimizer should be set before initialization for training.");
      return ML_ERROR_INVALID_PARAMETER;
    }
    /** TODO: update request of optimizer to be of same format as
     * Layer::requestTensor */
    opt->finalize();
    std::function<std::vector<TensorDim>(const TensorDim &)> cb =
      [this](const TensorDim &dim) {
        return opt->getOptimizerVariableDim(dim);
      };
    model_graph.requestOptimizerVariable(cb, true);
  }

  // Allocate weights
  model_graph.allocateWeights(exec_mode != ExecutionMode::INFERENCE);
  // enable this to save initialized weights for INFERENCE
  // model_graph.allocateWeights(true);

  initialized = true;

  if (!load_path.empty()) {
    load(load_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  }

  return status;
}

int NeuralNetwork::reinitialize() {
  int status = ML_ERROR_NONE;

  if (!initialized) {
    ml_loge("Error: Need to initialize first");
    return ML_ERROR_NOT_SUPPORTED;
  }

  unsigned int n_layers = (unsigned int)model_graph.size();

  // Same reason as in compile(): reinitialize() re-runs every node's
  // refinalize(), so the resolved decode Context is stale from here on.
  decode_ctx_ = nullptr;

  ml_logd("reinitializing neural network, layer size: %d", n_layers);
  PROFILE_MEM_ANNOTATE("Reinitialize");

  auto &input_conn_prop =
    std::get<std::vector<props::InputConnection>>(model_props);
  auto &label_layer_prop =
    std::get<std::vector<props::LabelLayer>>(model_props);

  std::vector<Connection> input_conn(input_conn_prop.begin(),
                                     input_conn_prop.end());
  std::vector<std::string> label_layers;

  if (!label_layer_prop.empty()) {
    label_layers = std::vector<std::string>(label_layer_prop.begin(),
                                            label_layer_prop.end());
  }

  status = model_graph.reinitialize(
    input_conn,
    std::vector<Connection>(label_layers.begin(), label_layers.end()));
  NN_RETURN_STATUS();

  return status;
}

/**
 * @brief     free layers
 */
NeuralNetwork::~NeuralNetwork() {
  try {
    deallocate();
  } catch (const std::runtime_error &e) {
    std::cerr << "Error occurred during destroying NeuralNetwork: " << e.what()
              << std::endl;
  }

  /** if neuralnet open fd */
  if (model_file_fd != -1)
    close(model_file_fd);
}

/**
 * @brief     forward propagation using layers object which has layer
 */
sharedConstTensors NeuralNetwork::forwarding(
  bool training, std::function<bool(void *userdata)> stop_cb, void *userdata) {

  unsigned int lookahead = std::get<props::FsuLookahead>(model_flex_props);
  bool fsu_mode = std::get<props::Fsu>(model_flex_props);
  if (fsu_mode) {
    for (unsigned int i = 0; i < lookahead; ++i) {
      model_graph.LoadTensors(i);
    }
  }
  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op =
    [this, stop_cb, lookahead, fsu_mode](std::shared_ptr<LayerNode> node,
                                         bool training) -> void {
    (void)this;
    PROFILE_MEM_ANNOTATE("Forwarding for layer: " + node->getName());

    auto f = std::get<0>(node->getExecutionOrder());
    if (exec_mode == ExecutionMode::TRAIN or
        (exec_mode == ExecutionMode::INFERENCE and !fsu_mode)) {
      model_graph.flushCacheExcept(f);
      node->forwarding(training);
    } else {
      /**
         currently, it supports FSU asynch mode for inference. The prcedure of
         FSU is below,

         Prerequests : This function is called node by node at the forwarding
         function in network graph.

         Step 1. If the execution order is the first (f==0) then, it will try
       to load tensors which used at layer 0.

         Step 2. It check whether these tensors from Step 1, then do the
                 forwarding of the first node.

         Step 3. Then check the look a head which says how many layer weights
       need to be loaded before running to hide overehad due to FSU,

         Step 4. Try to get the tesors by asking tensors for layers which is
       done by thread pool

         Step 5. Try to release the weights which has execution order less
       then f.

         Step n. repeat next layer starting with checking the tenosrs are
       loaded, and if it is loaded, then run forwarding. Every time it
       finishes the forwarding, ask load tensors for next n layers.

      **/
      model_graph.checkLoadComplete(f);
      node->forwarding(training);
      model_graph.inActive(f);
      model_graph.LoadTensors(f + lookahead);
    }
  };

  return model_graph.forwarding(training, forwarding_op, stop_cb, userdata);
}

/**
 * @brief     forward propagation using layers object which has layer
 */
sharedConstTensors NeuralNetwork::forwarding(sharedConstTensors input,
                                             sharedConstTensors label,
                                             bool training) {
  auto current_batch = model_graph.getBatchSize();
  if (current_batch != input[0]->batch()) {
    model_graph.setBatchSize(input[0]->batch());
    current_batch = model_graph.getBatchSize();
  }

  NNTR_THROW_IF(input[0]->batch() != current_batch ||
                  (!label.empty() && label[0]->batch() != current_batch),
                std::logic_error)
    << "Error: mismatch in batchsize for data and model."
    << " input_batch: " << input[0]->batch()
    << " label_batch: " << label[0]->batch()
    << " target_batch: " << current_batch;

  model_graph.setInputsLabels(input, label);

  return forwarding(training);
}

// The base execution seam: one decode or prefill forward step is a plain graph
// walk. CPU and OpenCL use this unchanged, so they are byte-identical to the
// pre-seam code; a backend whose decode is a capture/replay state machine
// overrides it on its own Context.
sharedConstTensors Context::runDecode(NeuralNetwork &nn, unsigned int from,
                                      unsigned int to,
                                      const sharedConstTensors &input,
                                      const sharedConstTensors &label) {
  return nn.incremental_forwarding(from, to, input, label, false);
}

// Resolve, once and then cache, the Context whose runDecode() drives a decode
// step. Every backend's base runDecode is the same plain walk, which dispatches
// each layer to the context named by its own engine= property, so the seam must
// drive through a backend only when the graph's layers actually run there.
//
// A build with two GPU backends compiled in registers both of their contexts. A
// blind "prefer the accelerator" would then hijack a run whose graph is on the
// other one — prefill walked on one backend, decode captured on the other, with
// the KV and hidden-state handoff crossing backends incoherently. Decide from
// the authoritative per-layer engine property instead, the same one the graph
// uses to route each node's context, and never from a hardcoded backend order.
//
// "The graph's engine" is a property of the WHOLE graph, so it is read from
// every node rather than sampled from one. Sampling the first node cannot work:
// the head of a realistic graph is an Input layer that carries no engine=
// property at all, so it always answers "cpu" and the seam would resolve to the
// host on a graph running entirely on an accelerator. The rule is instead: if
// exactly one non-cpu engine appears among the nodes, that backend drives the
// step; if the nodes disagree (two accelerators in one graph), no single
// backend can own the walk and the host does.
Context *NeuralNetwork::getDecodeContext() {
  if (decode_ctx_ != nullptr)
    return decode_ctx_;
  if (ct_engine == nullptr)
    return nullptr;

  std::string engine_name = "cpu";
  for (auto it = model_graph.cbegin(); it != model_graph.cend(); ++it) {
    const std::string node_engine = (*it)->getComputeEngineType();
    if (node_engine.empty() || node_engine == "cpu")
      continue;
    if (engine_name == "cpu") {
      engine_name = node_engine;
    } else if (engine_name != node_engine) {
      // Two different accelerators in one graph: neither owns the step.
      ml_logw("graph mixes engine=%s and engine=%s; the decode step stays on "
              "the host",
              engine_name.c_str(), node_engine.c_str());
      engine_name = "cpu";
      break;
    }
  }

  for (const std::string &name : {engine_name, std::string("cpu")}) {
    try {
      decode_ctx_ = ct_engine->getRegisteredContext(name);
      if (decode_ctx_ != nullptr)
        return decode_ctx_;
    } catch (...) {
      // not registered in this build; try the next candidate
    }
  }
  return decode_ctx_;
}

sharedConstTensors NeuralNetwork::incremental_forwarding(
  unsigned int from, unsigned int to, bool training,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {

  unsigned int lookahead = std::get<props::FsuLookahead>(model_flex_props);
  bool fsu_mode = std::get<props::Fsu>(model_flex_props);

  if (fsu_mode) {
    for (unsigned int i = 0; i < lookahead; ++i) {
      model_graph.LoadTensors(i);
    }
  }

  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op =
    [this, from, to, stop_cb, fsu_mode,
     lookahead](std::shared_ptr<LayerNode> node, bool training) -> void {
    PROFILE_MEM_ANNOTATE("Forwarding for layer: " + node->getName());

    // Feed-only step: a captured step graph is about to be replayed, so the
    // device work of every node is already in that graph. Only the nodes the
    // model declared as feeds still have to run, because their HOST side writes
    // into a buffer the graph reads through a fixed device pointer -- skip the
    // rest. This is the consumer half of setStepFeedOnly() /
    // setGraphReplayFeedNodes(); without it the backend can capture a graph but
    // the walk still runs every node on every token, which is slower than not
    // capturing at all.
    //
    // Ahead of getExecutionOrder()/flushCacheExcept deliberately: a skipped
    // node must not touch the cache plan either. Inert unless a backend has
    // entered feed-only mode for this one forward, so the CPU and OpenCL walks
    // are bit-identical.
    if (step_feed_only_ &&
        std::find(graph_replay_feed_nodes_.begin(),
                  graph_replay_feed_nodes_.end(),
                  node->getName()) == graph_replay_feed_nodes_.end())
      return;

    auto f = std::get<0>(node->getExecutionOrder());
    if (exec_mode == ExecutionMode::TRAIN or
        (exec_mode == ExecutionMode::INFERENCE and !fsu_mode)) {
      // Per-layer timing, opt-in via NNTR_LAYER_PROFILE so it costs nothing
      // when unset. Emits one TSV line per layer per call:
      //   LAYERPROF <span> <layer name> <ns>
      // where <span> is (to - from), i.e. > 1 for prefill and 1 for decode.
      static const bool layer_profile =
        std::getenv("NNTR_LAYER_PROFILE") != nullptr;
      if (layer_profile) {
        auto start_layer = std::chrono::high_resolution_clock::now();
        model_graph.flushCacheExcept(f);
        node->incremental_forwarding(from, to, training);
        auto end_layer = std::chrono::high_resolution_clock::now();
        auto duration_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
          end_layer - start_layer);
        std::cout << "LAYERPROF\t" << (to - from) << "\t" << node->getName()
                  << "\t" << duration_.count() << "\n";
        if (to - from == 1 && OpTimeTable::on())
          g_op_time.add(f, *node, duration_.count());
      } else if (to - from == 1 && OpTimeTable::on()) {
        model_graph.flushCacheExcept(f);
        const auto t0 = std::chrono::steady_clock::now();
        node->incremental_forwarding(from, to, training);
        g_op_time.add(f, *node,
                      std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count());
      } else {
        model_graph.flushCacheExcept(f);
        node->incremental_forwarding(from, to, training);
        if (from == 0 && to - from > 1)
          actStats(*node, to - from);
      }
#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
      // Debug: NNTR_DUMP_ROWS_DIR=<dir> writes each node's output rows of the
      // prefill step (to - from > 1). Inert when unset.
      // NNTR_DUMP_ROWS_HASH prints a per-node hash instead of
      // writing files, so it needs no directory; =2 covers the decode steps
      // (to - from == 1) as well.
      static const char *dump_hash = std::getenv("NNTR_DUMP_ROWS_HASH");
      static const char *dump_dir_env = std::getenv("NNTR_DUMP_ROWS_DIR");
      static const char *dump_dir = dump_dir_env != nullptr
                                      ? dump_dir_env
                                      : (dump_hash != nullptr ? "." : nullptr);
      static const bool dump_decode =
        dump_hash != nullptr && std::atoi(dump_hash) >= 2;
      if (dump_dir != nullptr && (to - from > 1 || dump_decode) &&
          node->getNumOutputs() > 0) {
        static const char *match = std::getenv("NNTR_DUMP_ROWS_MATCH");
        static unsigned int seq = 0;
        if (match == nullptr ||
            node->getName().find(match) != std::string::npos) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
          // On the cuda engine the dump reads the node output on the host, so
          // the stream this node's kernels were submitted on has to have
          // drained first or the hash is taken on rows the GPU has not written
          // yet -- which reads as nondeterminism that belongs to the
          // instrument, not to the model. Engine-gated: a no-op, and no CUDA
          // context created, on every non-cuda run.
          nntrainer::cuda::drain_if_async();
#endif
          debug_dump_rows_cl(node->getName(), node->getOutput(0), to - from,
                             dump_dir, seq);
        }
        ++seq;
      }
#endif
    } else {
      model_graph.checkLoadComplete(f);
      node->incremental_forwarding(from, to, training);
      model_graph.inActive(f);
      model_graph.LoadTensors(f + lookahead);
    }
  };

  return model_graph.incremental_forwarding(from, to, training, forwarding_op,
                                            stop_cb, userdata);
}

sharedConstTensors
NeuralNetwork::incremental_forwarding(unsigned int from, unsigned int to,
                                      sharedConstTensors input,
                                      sharedConstTensors label, bool training) {
  auto current_batch = model_graph.getBatchSize();
  NNTR_THROW_IF(input[0]->batch() != current_batch ||
                  (!label.empty() && label[0]->batch() != current_batch),
                std::logic_error)
    << "Error: mismatch in batchsize for data and model."
    << " input_batch: " << input[0]->batch()
    << " label_batch: " << label[0]->batch()
    << " target_batch: " << current_batch;

  model_graph.setInputsLabels(input, label);

  return incremental_forwarding(from, to, training);
}

/**
 * @brief     back propagation
 *            Call backwarding function of layer in reverse order
 *            No need to call at first Input Layer (No data to be updated)
 */
void NeuralNetwork::backwarding(int iteration,
                                std::function<bool(void *userdata)> stop_cb,
                                void *userdata) {

#ifdef DEBUG
  NNTR_THROW_IF(!opt, std::invalid_argument) << "optimizer is null!";
#endif

  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op =
    [this, stop_cb](std::shared_ptr<LayerNode> node, bool training) -> void {
    (void)this;
    PROFILE_MEM_ANNOTATE("Forwarding for layer: " + node->getName());

    auto f = std::get<0>(node->getExecutionOrder());
    model_graph.flushCacheExcept(f);

    node->forwarding(training);
  };

  std::function<bool(std::shared_ptr<LayerNode>, int)> backwarding_op =
    [this, stop_cb, userdata](std::shared_ptr<LayerNode> node,
                              int iteration) -> bool {
    /**
     * Do not change this order:
     * 1. calcGradient
     * 2. calcDerivative
     * 3. applyGradient
     * 4. gradientClippingOnLastAccess
     */

    model_graph.flushCacheExcept(std::get<1>(node->getExecutionOrder()));
    PROFILE_MEM_ANNOTATE("CalcGradient: " + node->getName());

    bool apply_gradient = true;
    if (node->getTrainable()) {
      /** If gradient optimization mode, then calculate gradient first */
      if (dynamic_training_opt.isGradientMode())
        node->calcGradient();

      /**
       * If optimization off, or gradient must be applied, then this will be
       * true
       * @todo This apply gradient should be passed to the each weight and later
       * be queried when updating gradient at once. (after moving apply_gradient
       * out of this function)
       *
       */
      // auto &layer = node->getObject();
      // apply_gradient = dynamic_training_opt.checkIfApply(
      //   layer->getWeightsRef(), layer->net_input[0], layer->net_hidden[0],
      //   opt, iteration);

      /** If gradient must be applied and its not gradient mode, calculate
       * gradient
       */
      if (!dynamic_training_opt.isGradientMode() && apply_gradient) {
        node->calcGradient();

        RunLayerContext &rc = node->getRunContext();
        if (model_graph.isMixedPrecision()) {
          for (auto w : rc.getWeights()) {
            if (w->hasGradient())
              if (!w->getGradientRef().isValid())
                return false;
          }
        }
      }
    }

    model_graph.flushCacheExcept(std::get<2>(node->getExecutionOrder()));
    PROFILE_MEM_ANNOTATE("CalcDerivative: " + node->getName());

    if (stop_cb(userdata)) {
      return true;
    }

    if (node->needsCalcDerivative()) {
      node->calcDerivative();
    }

    model_graph.flushCacheExcept(std::get<3>(node->getExecutionOrder()));
    PROFILE_MEM_ANNOTATE("ApplyGradient: " + node->getName());

    if (apply_gradient) {
      /// Apply gradient only at the end of the last shared weight access
      model_graph.applyGradients(
        node.get(), [iteration, opt_ = opt.get()](Weight &w) {
          w.calcRegularizationGradient();
          if (opt_->getType() != AdamW::type) {
            w.calcWeightDecayGradient();
          }
          RunOptimizerContext opt_context(&w, iteration,
                                          opt_->getLearningRate(iteration));
          opt_->applyGradient(opt_context);
        });
    }
    return true;
  };

  std::function<void(Weight &, int)> lazy_apply_grad_op =
    [opt_ = opt.get()](Weight &w, int iteration) -> void {
    w.calcRegularizationGradient();
    w.calcWeightDecayGradient();
    RunOptimizerContext opt_context(&w, iteration,
                                    opt_->getLearningRate(iteration));
    opt_->applyGradient(opt_context);
  };

  // return false if the gradient is not valid
  bool ret = false;

  while (!ret) {
    ret = model_graph.backwarding(iteration, forwarding_op, backwarding_op,
                                  lazy_apply_grad_op, stop_cb, userdata);
  }
}

namespace {

#if !defined(_WIN32) && defined(POSIX_FADV_WILLNEED)
/**
 * @brief Read the weight file ahead of the mapped load workers.
 *
 * The mapped loader copies every weight out of a file mapping, so on a cold
 * page cache each worker pulls the file in through page faults: one readahead
 * window per fault, one outstanding request per worker. A stream read of the
 * same file hands the kernel one large request per tensor instead, and on a
 * flash-backed handset that difference is most of the mapped load's time:
 * measured on a 3.4 GB weight file, 20-30 K major faults per load against
 * 0.1-6 K for the stream read of the same bytes.
 *
 * One helper thread therefore walks the file front to back -- the order the
 * workers consume it in -- and asks the kernel to start reading each step
 * with posix_fadvise(POSIX_FADV_WILLNEED). The call only queues reads into the
 * page cache: it maps nothing into this process (no RSS), never changes a byte
 * that is read, and the workers' faults on those pages become minor ones.
 *
 * The step is small on purpose. The kernel serves one WILLNEED call with at
 * most one readahead window (max(ra_pages, io_pages)) and silently drops the
 * rest of the range: measured on the same file, 32 MiB steps left the load
 * exactly as slow as no read-ahead, and 512 KiB steps recovered only part of
 * it. 128 KiB is the common default window; a step no larger than the window
 * covers every byte, and the call does not wait for the read, so many small
 * calls keep the device queue deep.
 *
 * The walk is bounded to half of the memory the system reports as available
 * when the load starts, so a file larger than the page cache can hold is not
 * read ahead only to be evicted again before its reader gets there; past the
 * bound the workers fall back to plain fault-driven readahead. It stops as
 * soon as the load's workers are done. NNTR_LOAD_PREFETCH=0 turns it off.
 */
class WeightFileReadAhead {
public:
  explicit WeightFileReadAhead(int fd) {
    static const bool enabled = []() {
      const char *e = std::getenv("NNTR_LOAD_PREFETCH");
      return !(e != nullptr && e[0] == '0');
    }();
    if (!enabled || fd < 0)
      return;
    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0)
      return;
    const uint64_t limit =
      std::min<uint64_t>(static_cast<uint64_t>(st.st_size), budgetBytes());
    if (limit == 0)
      return;
    worker_ = std::thread([this, fd, limit]() {
      constexpr uint64_t step = 128ull << 10;
      const auto t0 = std::chrono::steady_clock::now();
      uint64_t off = 0;
      for (; off < limit && !stop_.load(std::memory_order_relaxed); off += step)
        (void)::posix_fadvise(fd, static_cast<off_t>(off),
                              static_cast<off_t>(std::min(step, limit - off)),
                              POSIX_FADV_WILLNEED);
      ml_logi("load read-ahead: %llu of %llu MiB queued in %.1f ms",
              static_cast<unsigned long long>(std::min(off, limit) >> 20),
              static_cast<unsigned long long>(limit >> 20),
              std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0)
                .count());
    });
  }

  ~WeightFileReadAhead() {
    stop_.store(true, std::memory_order_relaxed);
    if (worker_.joinable())
      worker_.join();
  }

  WeightFileReadAhead(const WeightFileReadAhead &) = delete;
  WeightFileReadAhead &operator=(const WeightFileReadAhead &) = delete;

private:
  /** @brief half of MemAvailable, else a quarter of physical memory */
  static uint64_t budgetBytes() {
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    uint64_t kib = 0;
    while (meminfo >> key >> kib) {
      if (key == "MemAvailable:")
        return (kib << 10) / 2;
      meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    const long pages = ::sysconf(_SC_PHYS_PAGES);
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (pages <= 0 || page_size <= 0)
      return 0;
    return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size) / 4;
  }

  std::atomic<bool> stop_{false};
  std::thread worker_;
};
#else
/** @brief no read-ahead where posix_fadvise is not available */
class WeightFileReadAhead {
public:
  explicit WeightFileReadAhead(int) {}
};
#endif

/**
 * @brief Resolve the data type a weight will actually be stored as.
 *
 * Mirrors the per-weight policy of the layer save overrides: bias-like tensors
 * (height == 1) are not block-quantized and stay in their original type.
 */
TensorDim::DataType resolveStoredDtype(const Tensor &weight,
                                       TensorDim::DataType requested) {
  if (requested == TensorDim::DataType::NONE ||
      requested == weight.getDataType())
    return weight.getDataType();

  if (nntrainer::safetensors::isQuantized(requested) &&
      weight.getDim().height() == 1)
    return weight.getDataType();

  return requested;
}

} // namespace

void NeuralNetwork::save(
  const std::string &file_path, ml::train::ModelFormat format,
  TensorDim::DataType dtype,
  const std::map<std::string, TensorDim::DataType> &layer_dtype_map,
  ml::train::ISA target_isa) {
  NNTR_THROW_IF(!initialized, std::runtime_error)
    << "Cannot save model if not initialized yet, path: " << file_path
    << " format: " << static_cast<unsigned>(format);

  NNTR_THROW_IF(format != ml::train::ModelFormat::MODEL_FORMAT_BIN &&
                  dtype != TensorDim::DataType::NONE,
                std::runtime_error)
    << "Cannot save the model with a specific data type unless the model "
       "format is `MODEL_FORMAT_BIN`.";

  /// @todo this switch case should be delegating the function call only. It's
  /// not delegating for now as required logics are manageable for now.
  switch (format) {
  case ml::train::ModelFormat::MODEL_FORMAT_BIN: {
    auto model_file = checkedOpenStream<std::ofstream>(
      file_path, std::ios::out | std::ios::binary | std::ios::trunc);

    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      const auto &layer_node = *iter;
      auto it = layer_dtype_map.find(layer_node->getName());
      auto target_dtype = (it != layer_dtype_map.end()) ? it->second : dtype;
      layer_node->save(model_file, false, exec_mode, target_dtype, target_isa);
    }

    if (opt && istrequal(opt->getType(), "adam")) {
      std::string adam = "adam";
      model_file.write(adam.c_str(), 4);
      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           iter++) {
        (*iter)->save(model_file, true);
      }
    }

    if (exec_mode == ml::train::ExecutionMode::TRAIN) {
      model_file.write((char *)&epoch_idx, sizeof(epoch_idx));
      model_file.write((char *)&iter, sizeof(iter));
    }

    model_file.close();
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_INI:
    saveModelIni(file_path);
    break;
  case ml::train::ModelFormat::MODEL_FORMAT_INI_WITH_BIN: {
    auto old_save_path = std::get<props::SavePath>(model_flex_props);
    auto bin_file_name =
      file_path.substr(0, file_path.find_last_of('.')) + ".bin";

    std::get<props::SavePath>(model_flex_props).set(bin_file_name);
    save(file_path, ml::train::ModelFormat::MODEL_FORMAT_INI);
    save(bin_file_name, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    std::get<props::SavePath>(model_flex_props) = old_save_path;
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_ONNX: {
    throw nntrainer::exception::not_supported(
      "saving with ONNX format is not supported yet.");
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS: {
    // Delegate the data section to the same per-layer save() the BIN path
    // uses so the quantized bytes are byte-identical: each layer override
    // applies its own quantization policy (e.g. embedding/tie-word-embedding
    // do not transpose, shared weights are written once on first access),
    // which a generic quantizer here could not replicate. Bytes go to a temp
    // file first so per-weight sizes are known before the header is written.
    const std::string tmp_path = file_path + ".nntrtmp";
    std::vector<safetensors::TensorEntry> entries;

    {
      auto tmp_file = checkedOpenStream<std::ofstream>(
        tmp_path, std::ios::out | std::ios::binary | std::ios::trunc);

      std::unordered_set<const Tensor *> visited_st;
      size_t data_offset = 0;

      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           iter++) {
        const auto &layer_node = *iter;
        auto it = layer_dtype_map.find(layer_node->getName());
        const auto requested =
          (it != layer_dtype_map.end()) ? it->second : dtype;
        auto &rc = layer_node->getRunContext();

        // Collect the weights this layer will actually write: first-access
        // only (shared weights are saved once), deduped across the graph.
        struct WInfo {
          const Tensor *t;
          TensorDim::DataType stored;
        };
        std::vector<WInfo> wlist;
        for (unsigned int i = 0; i < rc.getNumWeights(); ++i) {
          if (!rc.isGradientFirstAccess(i))
            continue;
          const Tensor &t = rc.getWeight(i);
          if (!visited_st.insert(&t).second)
            continue;
          wlist.push_back({&t, resolveStoredDtype(t, requested)});
        }

        // Write this layer's weights exactly as the BIN path would.
        const auto start = static_cast<size_t>(tmp_file.tellp());
        layer_node->save(tmp_file, false, exec_mode, requested, target_isa);
        const auto layer_bytes = static_cast<size_t>(tmp_file.tellp()) - start;

        // Map the written bytes back to per-weight header entries. At most one
        // weight per layer is block-quantized; the rest are stored as-is, so
        // the quantized weight's size is whatever remains.
        size_t known = 0;
        int quant_count = 0;
        for (const auto &w : wlist) {
          if (safetensors::isQuantized(w.stored))
            ++quant_count;
          else
            known += w.t->getMemoryBytes();
        }
        NNTR_THROW_IF(quant_count > 1, std::runtime_error)
          << "safetensors save: layer '" << layer_node->getName()
          << "' has multiple quantized weights, which is not supported.";

        size_t assigned = 0;
        for (const auto &w : wlist) {
          const auto &dim = w.t->getDim();
          const bool is_quant = safetensors::isQuantized(w.stored);
          const size_t wsize =
            is_quant ? (layer_bytes - known) : w.t->getMemoryBytes();

          safetensors::TensorEntry entry;
          entry.name = w.t->getName();
          entry.offset_start = data_offset;
          entry.offset_end = data_offset + wsize;
          if (is_quant) {
            // Quantized blobs are opaque bytes (U8) with a 1-D byte shape;
            // the native type and logical shape live in extension fields.
            entry.dtype = safetensors::dtypeToString(w.stored); // "U8"
            entry.shape = {wsize};
            entry.nntr_dtype = safetensors::nntrDtypeName(w.stored);
            entry.nntr_shape = {dim.batch(), dim.channel(), dim.height(),
                                dim.width()};
          } else {
            entry.dtype = safetensors::dtypeToString(w.stored);
            entry.shape = {dim.batch(), dim.channel(), dim.height(),
                           dim.width()};
          }
          entries.push_back(std::move(entry));
          data_offset += wsize;
          assigned += wsize;
        }

        NNTR_THROW_IF(assigned != layer_bytes, std::runtime_error)
          << "safetensors save: byte accounting mismatch for layer '"
          << layer_node->getName() << "' (wrote " << layer_bytes << ", mapped "
          << assigned << ").";
      }

      tmp_file.close();
    }

    // Embed an nntrainer dtype summary so a quantized file can be inspected
    // and identified without an accompanying nntr_config.json.
    std::map<std::string, std::string> metadata;
    bool any_quant = false;
    bool any_q4_0 = false;
    for (const auto &e : entries) {
      any_quant = any_quant || !e.nntr_dtype.empty();
      any_q4_0 = any_q4_0 || e.nntr_dtype == "Q4_0";
    }
    if (any_quant)
      metadata["nntr_format"] = "nntr-safetensors-v1";
    // Q4_0 is repacked into an ISA-specific layout (x86: q4_0x8, ARM: q4_0x4)
    // that is indistinguishable from the header alone, so record which one was
    // produced. DEFAULT resolves to the build platform's layout. Only emitted
    // when a Q4_0 tensor is present, since no other type depends on the ISA.
    if (any_q4_0) {
      const char *isa_str;
      switch (target_isa) {
      case ml::train::ISA::X86:
        isa_str = "x86";
        break;
      case ml::train::ISA::ARM:
        isa_str = "arm";
        break;
      default: // DEFAULT -> the compiled backend's layout
#if defined(__aarch64__) || defined(__arm__)
        isa_str = "arm";
#else
        isa_str = "x86";
#endif
        break;
      }
      metadata["nntr_q4_0_isa"] = isa_str;
    }

    // Write: [8-byte header_size][header (padded to 8)][raw weight data]
    const std::string header_json = safetensors::buildHeader(entries, metadata);
    const uint64_t header_size = static_cast<uint64_t>(header_json.size());
    // safetensors layout: [8-byte header length][header JSON][tensor raw data]
    auto st_file = checkedOpenStream<std::ofstream>(
      file_path, std::ios::out | std::ios::binary | std::ios::trunc);
    // [8-byte header length]
    st_file.write(reinterpret_cast<const char *>(&header_size),
                  sizeof(header_size));
    // [header JSON: per-tensor dtype/shape/offsets + __metadata__]
    st_file.write(header_json.data(),
                  static_cast<std::streamsize>(header_json.size()));
    // [tensor raw data]
    {
      std::ifstream data_in(tmp_path, std::ios::in | std::ios::binary);
      st_file << data_in.rdbuf();
    }
    st_file.close();
    std::remove(tmp_path.c_str());
    break;
  }
  default:
    throw nntrainer::exception::not_supported(
      "saving with given format is not supported yet");
  }
}

/**
 * @brief What a single QS4CX record says about the stride it was written at.
 */
enum class Qs4cxRecordStride {
  PADDED,    /**< the record reads as padded */
  TRIMMED,   /**< the record reads as trimmed */
  UNDECIDED, /**< this record cannot tell the two apart */
};

/**
 * @brief Which stride the QS4CX record at @a offset was written at, read off
 *        the record itself rather than off the file's size.
 * @param path weight file to peek into
 * @param offset start of the record in @a path
 * @param K record height (input channels)
 * @param N record width (output channels, one fp32 scale each)
 * @return PADDED or TRIMMED when the record settles it, UNDECIDED when it
 *         cannot
 *
 * The two strides differ only by the floor(N/2) pad bytes the padded layout
 * keeps between a record's nibbles and its scales. The writer never touches
 * that pad -- QS4CX_Tensor zero-fills the record while the heap bypass is
 * resolved off (a writer running with it on, e.g. on Android, leaves the pad
 * uninitialized and this probe can misread it) and the quantiser writes
 * strictly inside the N * ceil(K/2) nibbles -- so under the padded layout the
 * bytes right after the nibbles are zero, while under the trimmed layout they
 * are the first per-channel scale, which is a strictly positive finite
 * reciprocal and never zero. Reading four bytes there therefore tells the two
 * apart outright, with no dependency on what save() appended after the weight
 * body. Fewer than four pad bytes (N < 8) leaves the probe reading the low
 * bytes of a real scale under the padded layout too, which can pass for a
 * float of its own; it answers UNDECIDED there rather than guessing, and the
 * caller says out loud which stride it fell back to.
 */
static Qs4cxRecordStride qs4cxRecordStride(const std::string &path,
                                           size_t offset, size_t K, size_t N) {
  const size_t trimmed_nibbles = QS4CX_Tensor::nibbleBytes(K, N, false);
  const size_t padded_nibbles = QS4CX_Tensor::nibbleBytes(K, N, true);
  if (padded_nibbles - trimmed_nibbles < sizeof(float))
    return Qs4cxRecordStride::UNDECIDED;

  std::ifstream probe_file(path, std::ios::in | std::ios::binary);
  if (!probe_file.is_open())
    return Qs4cxRecordStride::UNDECIDED;
  probe_file.seekg(static_cast<std::streamoff>(offset + trimmed_nibbles),
                   std::ios::beg);
  float probe = 0.0f;
  probe_file.read(reinterpret_cast<char *>(&probe), sizeof(probe));
  if (!probe_file)
    return Qs4cxRecordStride::UNDECIDED;

  return (std::isnormal(probe) && probe > 0.0f) ? Qs4cxRecordStride::TRIMMED
                                                : Qs4cxRecordStride::PADDED;
}

void NeuralNetwork::load(const std::string &file_path,
                         ml::train::ModelFormat format) {
  /// @todo this switch case should be delegating the function call only. It's
  /// not delegating for now as required logics are manageable for now.

  // [load-trace] The record walk below is the one single-threaded stretch
  // ahead of the worker fan-out; time it separately so a slow header parse
  // cannot hide inside the load's wall clock.
  const auto _lt_load_t0 = std::chrono::steady_clock::now();

  bool fsu_mode = std::get<props::Fsu>(model_flex_props);

  const std::regex reg_("\\s*\\;\\s*");
  auto v = split(file_path, reg_);

  size_t start_from = 0;
  std::vector<std::pair<size_t, size_t>> file_offset;
  std::unordered_set<const Tensor *> visited_weights;
  // A QINT4 record's on-disk size depends on its container: the shared
  // plain container (qscheme PER_CHANNEL_AFFINE at the record head; the
  // PR#3978 form) carries fp32 scales plus KAI pad and is NOT the in-memory
  // size that getMemoryBytes() reports. QINT4 is an on-disk format only
  // (Tensor refuses to build a QINT4 tensor), so int4 records reach this walk
  // as QS4CX weights and the legacy_int4_model branch below peeks each
  // record's qscheme so the running offset matches the actual file layout.
  std::ifstream qint4_peek_stream;
  bool qint4_peek_tried = false;
  // A legacy QINT4 model (model_tensor_type "QINT4-*") now builds QS4CX
  // weights; its int4 records are still the legacy on-disk form (u16 header +
  // Section A fp16 scales, or plain container) whose size differs from the
  // QS4CX in-memory getMemoryBytes(). Size them explicitly and flag the
  // tensors so QS4CX_Tensor::read() transcodes.
  const std::string model_tensor_type_str =
    to_string(std::get<props::ModelTensorDataType>(model_flex_props));
  const bool legacy_int4_model = model_tensor_type_str.rfind("QINT4", 0) == 0;
  // Whether the running offset below is a faithful model of the file, i.e.
  // whether every record it accounts for is the size actually on disk. Cleared
  // by the dtypes for which it is not -- see the validation after the loop.
  bool exact_record_sizes = true;
  // What the same walk would total under the OTHER QS4CX record stride, and
  // whether any weight is a QS4CX record the two strides size differently.
  // Both feed the per-file stride choice after the walk.
  size_t alt_total = 0;
  bool qs4cx_stride_differs = false;
  // Where the file's first QS4CX record starts, and its shape. Every record
  // ahead of it is stride-independent, so this offset is the same under either
  // walk -- which makes it the one QS4CX record that can be read straight out
  // of the file to settle the stride.
  bool first_qs4cx_found = false;
  size_t first_qs4cx_offset = 0;
  size_t first_qs4cx_height = 0;
  size_t first_qs4cx_width = 0;
  // Lay every weight out over the file: these record sizes, accumulated in
  // order, ARE the offsets. Parameterised by the QS4CX record stride (the file
  // names neither of the two) and re-runnable, so the other stride can be
  // tried once the file size says which one the exporter used.
  auto walk_records = [&](bool qs4cx_padded) {
    start_from = 0;
    alt_total = 0;
    file_offset.clear();
    visited_weights.clear();
    exact_record_sizes = true;
    qs4cx_stride_differs = false;
    first_qs4cx_found = false;
    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      auto weights = (*iter)->getRunContext().getWeights();
      for (auto weight : weights) {
        // Shared weights (e.g., TieWordEmbedding) reference the same Tensor
        // object via requestOrExtend. Calling setFileOffset on the second
        // occurrence overwrites the correct offset by the first.
        // Skip duplicates so that:
        // 1. file_offset is only set once (at the position where save writes)
        // 2. start_from is only advanced once (matching actual file layout)
        if (!visited_weights.insert(&weight->getVariableRef()).second) {
          continue;
        }
        auto tensor_data_type = weight->getDim().getDataType();
        // A QS4CX record is read at the stride this walk assumes, and
        // getMemoryBytes() below reports whichever stride the tensor carries --
        // so choose before asking. (A legacy QINT4 model's QS4CX weights are
        // not QS4CX records at all; they keep the default and are sized
        // explicitly further down.)
        if (tensor_data_type == TensorDim::DataType::QS4CX &&
            !legacy_int4_model)
          weight->getVariableRef().setQs4cxRecordPadded(qs4cx_padded);
        // by reference: getVariable() hands back a copy, and the record size
        // has to come from the very tensor the stride was just set on.
        size_t size = weight->getVariableRef().getMemoryBytes();
        weight->getVariableRef().setFileOffset(start_from);
        ///@todo instead of checking the data type,
        /// we may need to create a common parent class for
        /// quantized tensors, requiring qparam to be saved
        /// and creating a common interface to check if qparam is needed
        /// this kind of type checking should be avoided
        if (tensor_data_type != TensorDim::DataType::FP32 &&
            tensor_data_type != TensorDim::DataType::FP16 &&
            tensor_data_type != TensorDim::DataType::Q6_K &&
            tensor_data_type != TensorDim::DataType::Q4_0 &&
            tensor_data_type != TensorDim::DataType::QS4CX &&
            tensor_data_type != TensorDim::DataType::TQ2CX &&
            tensor_data_type != TensorDim::DataType::QS4ACX &&
            tensor_data_type != TensorDim::DataType::QS4CX_WH &&
            tensor_data_type != TensorDim::DataType::QS2CX_WH) {
          // for tensor with qparam
          size += sizeof(uint16_t);
          // ... but only CharTensor, ShortTensor, UIntTensor (UINT8/16/32),
          // Uint4QTensor and BCQTensor actually override save()/read() to
          // write and consume that byte pair. Q4_K has no override, so its
          // records carry no qparam header and are 2 bytes smaller on disk
          // than accounted here. UINT4 is ambiguous: with qscheme Q4_Kx8 it
          // materialises as a Q4_K_Tensor and behaves like Q4_K. UINT8/16/32
          // are excluded from the exact check as well, conservatively. Track
          // that so the size check below only speaks where it can be
          // believed.
          if (tensor_data_type == TensorDim::DataType::Q4_K ||
              tensor_data_type == TensorDim::DataType::UINT4 ||
              tensor_data_type == TensorDim::DataType::UINT8 ||
              tensor_data_type == TensorDim::DataType::UINT16 ||
              tensor_data_type == TensorDim::DataType::UINT32)
            exact_record_sizes = false;
        }
        if (tensor_data_type == TensorDim::DataType::QINT4) {
          if (!qint4_peek_tried) {
            qint4_peek_tried = true;
            qint4_peek_stream.open((v.size() == 2) ? v[1] : v[0],
                                   std::ios::in | std::ios::binary);
          }
          uint16_t disk_scheme = 0xFFFF;
          if (qint4_peek_stream.is_open()) {
            qint4_peek_stream.clear();
            qint4_peek_stream.seekg(static_cast<std::streamoff>(start_from),
                                    std::ios::beg);
            qint4_peek_stream.read(reinterpret_cast<char *>(&disk_scheme),
                                   sizeof(uint16_t));
          }
          if (qint4_peek_stream.is_open() && qint4_peek_stream.good() &&
              disk_scheme ==
                static_cast<uint16_t>(QScheme::PER_CHANNEL_AFFINE)) {
            const TensorDim &d = weight->getDim();
            size = sizeof(uint16_t) +
                   Int4Utils::plainRecordPayloadBytes(d.width(), d.height());
          }
        }
        if (legacy_int4_model &&
            tensor_data_type == TensorDim::DataType::QS4CX) {
          // Legacy QINT4 record loaded as a QS4CX tensor: the on-disk record
          // is the legacy form, not the QS4CX in-memory size. Flag the tensor
          // so read() transcodes, and size by the peeked container.
          weight->getVariableRef().setOnDiskLegacyQint4(true);
          if (!qint4_peek_tried) {
            qint4_peek_tried = true;
            qint4_peek_stream.open((v.size() == 2) ? v[1] : v[0],
                                   std::ios::in | std::ios::binary);
          }
          uint16_t disk_scheme = 0xFFFF;
          if (qint4_peek_stream.is_open()) {
            qint4_peek_stream.clear();
            qint4_peek_stream.seekg(static_cast<std::streamoff>(start_from),
                                    std::ios::beg);
            qint4_peek_stream.read(reinterpret_cast<char *>(&disk_scheme),
                                   sizeof(uint16_t));
          }
          const TensorDim &d = weight->getDim();
          if (disk_scheme == static_cast<uint16_t>(QScheme::PER_CHANNEL_AFFINE))
            size = sizeof(uint16_t) +
                   Int4Utils::plainRecordPayloadBytes(d.width(), d.height());
          else
            size = sizeof(uint16_t) +
                   Int4Utils::kaiNibblePayloadBytes(d.width(), d.height()) +
                   static_cast<size_t>(d.width()) * sizeof(uint16_t);
        }
        // Same record under the other QS4CX stride, so one walk yields both
        // candidate totals. Every other record is stride-independent.
        size_t alt_size = size;
        if (tensor_data_type == TensorDim::DataType::QS4CX &&
            !legacy_int4_model) {
          const TensorDim &d = weight->getDim();
          alt_size =
            QS4CX_Tensor::recordBytes(d.height(), d.width(), !qs4cx_padded);
          if (alt_size != size)
            qs4cx_stride_differs = true;
          if (!first_qs4cx_found) {
            first_qs4cx_found = true;
            first_qs4cx_offset = start_from;
            first_qs4cx_height = d.height();
            first_qs4cx_width = d.width();
          }
        }
        file_offset.emplace_back(std::make_pair(start_from, size));
        start_from += size;
        alt_total += alt_size;
      }
    }
  };
  // Start from the padded stride: it is what every package exported before the
  // writer switched holds, and it is never the smaller of the two, so the
  // memory already planned from it stays an upper bound if the file turns out
  // to be trimmed.
  walk_records(true);

  // Tell the two QS4CX record strides apart, per FILE. The record has no
  // version field and the .bin no per-tensor dtype, so nothing in the file
  // says which stride the exporter used -- but a file is written by one
  // exporter, so one stride holds throughout. Mixed-stride files are not
  // representable here and are not supported: a file that pairs records from
  // two exporters is read wholly at whichever stride wins below, and every
  // record written at the other one is misread. Two signals settle it:
  //
  //  - Size, when it can speak. total_trimmed <= actual < total_padded can
  //    only be the trimmed layout, because no tail makes a padded file that
  //    short.
  //  - The first record's own bytes, when size cannot. save() appends a tail
  //    after the weight body (the adam optimiser block, then 8 bytes of
  //    training metadata), and the two totals differ by only sum(floor(N/2))
  //    -- 8 bytes already for a single 16-channel weight -- so a trimmed file
  //    with a tail that long reaches total_padded and its size stops meaning
  //    anything. qs4cxRecordStride() reads the record instead, which the tail
  //    cannot touch.
  //
  // Anything neither signal can judge keeps reading as padded, as before -- but
  // it says so, because that is the one outcome that can be silently wrong: a
  // trimmed file read as padded keeps its nibbles in place and slides every
  // per-channel scale by floor(N/2) bytes, so the output stays plausible while
  // each channel is scaled by a neighbour's factor. The gap the two signals
  // leave is a file whose first QS4CX record is narrow (N < 8, so fewer than
  // four pad bytes for the probe to read) and whose strides differ in total by
  // no more than save()'s tail.
  const auto weight_path = (v.size() == 2) ? v[1] : v[0];
  const size_t total_padded = start_from;
  const size_t total_trimmed = alt_total;
  if (format == ml::train::ModelFormat::MODEL_FORMAT_BIN &&
      qs4cx_stride_differs) {
    std::error_code ec;
    const auto file_bytes = std::filesystem::file_size(weight_path, ec);
    if (!ec) {
      const size_t actual = static_cast<size_t>(file_bytes);
      bool trimmed = (total_trimmed <= actual && actual < total_padded);
      if (!trimmed && total_padded <= actual && first_qs4cx_found) {
        switch (qs4cxRecordStride(weight_path, first_qs4cx_offset,
                                  first_qs4cx_height, first_qs4cx_width)) {
        case Qs4cxRecordStride::TRIMMED:
          trimmed = true;
          break;
        case Qs4cxRecordStride::PADDED:
          break;
        case Qs4cxRecordStride::UNDECIDED:
          ml_logw(
            "weight file '%s' (%zu bytes) fits neither QS4CX record stride "
            "exclusively (padded total %zu, trimmed total %zu) and its first "
            "QS4CX record (%zu x %zu) is too narrow to read the stride off, "
            "so it is being read at the PADDED stride. If this file was "
            "exported by a build that wrote the trimmed stride, its "
            "per-channel scales are read from the wrong offset and the output "
            "will be plausible but wrong",
            weight_path.c_str(), actual, total_padded, total_trimmed,
            first_qs4cx_height, first_qs4cx_width);
          break;
        }
      }
      if (trimmed) {
        walk_records(false);
        ml_logi("weight file '%s' matches the trimmed QS4CX record stride "
                "(%zu bytes) rather than the padded one (%zu); reading it as "
                "trimmed",
                weight_path.c_str(), total_trimmed, total_padded);
      }
    }
  }

  // Check the on-disk contract the offsets above assume. The .bin carries no
  // per-tensor dtype, so a weight's position is nothing but the running sum of
  // the record sizes just computed: if any layer requests a dtype or stride
  // other than the one the exporting graph wrote, every following weight is
  // read at the wrong offset. That surfaces as garbage output or a crash far
  // from the cause, while the file itself disagrees with the graph by a plain
  // byte count -- so compare the two and say so here instead.
  if (format == ml::train::ModelFormat::MODEL_FORMAT_BIN && start_from != 0 &&
      exec_mode == ExecutionMode::INFERENCE && exact_record_sizes) {
    std::error_code ec;
    const auto file_bytes = std::filesystem::file_size(weight_path, ec);
    // A path this code cannot stat is left to checkedOpenStream below, which
    // reports a missing/unreadable file far better than a size mismatch would.
    if (!ec) {
      const size_t actual = static_cast<size_t>(file_bytes);
      // Reaching the throw with two QS4CX candidates means both fell short (a
      // file between them was read as trimmed above), so name both: there is
      // no single expected size to quote, and the smaller one is the floor.
      const std::string accounts_for =
        qs4cx_stride_differs
          ? std::to_string(total_trimmed) +
              " bytes with the trimmed QS4CX record stride (" +
              std::to_string(total_padded) + " with the padded one)"
          : std::to_string(start_from) + " bytes";
      const size_t min_total =
        qs4cx_stride_differs ? total_trimmed : start_from;
      NNTR_THROW_IF(actual < start_from, std::runtime_error)
        << "[NeuralNetwork::load] weight file is smaller than this graph "
           "expects: '"
        << weight_path << "' holds " << actual << " bytes, but the "
        << visited_weights.size() << " weights of this graph account for "
        << accounts_for << " (short by " << (min_total - actual)
        << "). The model file records no per-tensor dtype, so the first thing "
           "to suspect is a layer requesting a weight at a different dtype or "
           "stride than the package was exported with -- that shifts every "
           "later weight's offset. A truncated or mismatched file does it too.";
      if (actual > start_from)
        ml_logw("weight file '%s' is %zu bytes longer than the %zu bytes this "
                "graph accounts for; reading the leading %zu and ignoring the "
                "tail",
                weight_path.c_str(), actual - start_from, start_from,
                start_from);
    }
  }

  nntrainer::load_trace::add(
    nntrainer::load_trace::PRESCAN,
    (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - _lt_load_t0)
      .count());

  if (exec_mode == ExecutionMode::INFERENCE && fsu_mode) {
    model_graph.setFsuWeightPath((v.size() == 2) ? v[1] : v[0]);
    model_graph.setWeightOffset(file_offset);
  }

  switch (format) {
  case ml::train::ModelFormat::MODEL_FORMAT_BIN: {
    NNTR_THROW_IF(!initialized, std::runtime_error)
      << "Cannot load if not initialized yet, path: " << file_path
      << " format: " << static_cast<unsigned>(format);
    auto f_path = (v.size() == 2) ? v[1] : v[0];

    auto model_file =
      checkedOpenStream<std::ifstream>(f_path, std::ios::in | std::ios::binary);

#if defined(_WIN32)
    HANDLE hFile, hMap;
#endif

    if (exec_mode == ml::train::ExecutionMode::INFERENCE) {
      // Always keep a long-lived fd open during inference. Virtual (slim)
      // tensors capture this fd at read-time and use it later in activate()
      // to mmap their backing region on demand. Without it, virtual tensors
      // end up with fd=-1 and activate() returns MAP_FAILED, segfaulting on
      // first use (e.g. SlimMoE expert weights when MMAP_READ=true).
      model_file_fd = weightPageCachePolicy() >= 0
                        ? openWeightFile(f_path.c_str())
                        : open(f_path.c_str(), O_RDONLY);
      NNTR_THROW_IF((model_file_fd == -1), std::invalid_argument)
        << "Cannot open file : " << f_path;

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
      // Bind the derive-once GPU weight-pack cache to this weight file, by
      // file identity (size and modification time). Either maps a pack the
      // load workers below can consume, or arms a one-time rewrite. A no-op
      // when the cache is opted out of, and on every non-OpenCL build.
      {
        nntrainer::load_trace::Scope _lt(nntrainer::load_trace::PACK_OPEN);
        v8c_pack::set_source(f_path.c_str());
      }
      // Open the shared device arena the per-weight scale/row-sum pairs are
      // carved from. It stays open until the flush at the end of this load,
      // which is the only thing that guarantees a staged chunk reaches the
      // device; outside that window every carve writes itself.
      v8c_open_aux_arena();
#endif

      // Load weights with bounded thread number not to exceed mmap limits
      constexpr size_t MAX_LOAD_THREADS = 8;
      std::vector<std::shared_ptr<LayerNode>> load_nodes(model_graph.cbegin(),
                                                         model_graph.cend());
      const size_t num_load_nodes = load_nodes.size();
      const size_t num_load_threads =
        std::min<size_t>(num_load_nodes, MAX_LOAD_THREADS);

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // [wprefetch level 2] NNTR_CUDA_WPREFETCH>=2 on a cuda graph: migrate
      // each QS4CX weight's plain payload to the device AS IT IS READ, so the
      // FC bytes never accumulate in host RSS during the load. Opt-in and
      // default-off: the migration only has anything to move when the payload
      // is in managed memory, and cuda_fc_qs4cx_prefetch_weight declines
      // otherwise. The engine gate keeps this off OpenCL/CPU runs of a
      // dual-enabled binary, where a stray CUDA call would create a CUDA
      // context. Scale conversion at the first forward briefly faults the
      // small fp32-scale tail back -- expected, and tiny.
      bool cuda_wprefetch_load = false;
      {
        static const int _wpf = []() {
          const char *e = std::getenv("NNTR_CUDA_WPREFETCH");
          return e ? atoi(e) : 0;
        }();
        if (_wpf >= 2)
          for (auto &n : load_nodes)
            if (n->isComputeEngineCUDA()) {
              cuda_wprefetch_load = true;
              break;
            }
      }
#endif

      std::atomic<size_t> next_load_index{0};

#if !defined(_WIN32)
      // [page cache] Load-mapping reaper. Each worker maps the WHOLE weight
      // file and drops it only when its node is done, so at any instant the
      // resident file pages are (workers) x (one node's weights) -- measured
      // on gemma4 E2B / Adreno 840 as ~190 MiB of the 1 520 MiB RssFile peak,
      // and the peak is the load, not the steady state.
      //
      // The drop is nearly free here and that is the point: the mapping is
      // read-only MAP_PRIVATE, so MADV_DONTNEED only tears down THIS process's
      // PTEs. The bytes stay in the page cache, so a page the reader still
      // wants comes back as a MINOR fault, not a trip to flash. Reaping every
      // few milliseconds therefore bounds the resident window without turning
      // the streaming read back into the fault-per-page disaster
      // POSIX_MADV_RANDOM used to make of it.
      //
      // NNTR_LOAD_REAP_MS sets the period; 0 turns the reaper off and restores
      // the previous drop-at-node-end behaviour exactly.
      static const int reap_ms = []() {
        const char *e = std::getenv("NNTR_LOAD_REAP_MS");
        const int v = (e != nullptr) ? std::atoi(e) : 20;
        return (v > 0) ? v : 0;
      }();
      std::mutex reap_mtx;
      std::vector<std::pair<void *, size_t>> reap_live;
      std::atomic<bool> reap_stop{false};
#endif

      // Serializes weight_load_hook invocations across the load workers: the
      // hook body may run parallel_for derives and device uploads of its own,
      // and the interleave's win is overlapping THAT work with the other
      // workers' file reads -- not running several hook bodies at once.
      std::mutex load_hook_mtx;

      std::exception_ptr load_error;
      std::mutex load_error_mutex;
      auto load_worker = [&]() {
        try {
          for (size_t idx =
                 next_load_index.fetch_add(1, std::memory_order_relaxed);
               idx < num_load_nodes;
               idx = next_load_index.fetch_add(1, std::memory_order_relaxed)) {
            auto node = load_nodes[idx];

            if (weightPageCachePolicy() >= 0) {
              node->read(FileReadSource{model_file_fd}, false, exec_mode,
                         fsu_mode, std::numeric_limits<size_t>::max(), true,
                         model_file_fd);
            } else if (!MMAP_READ) {
              auto local_model_file = checkedOpenStream<std::ifstream>(
                (v.size() == 2) ? v[1] : v[0], std::ios::in | std::ios::binary);
              node->read(local_model_file, false, exec_mode, fsu_mode,
                         std::numeric_limits<size_t>::max(), true,
                         model_file_fd);
            } else {
#if defined(_WIN32)
              // Map per-ask, then unmap immediately after: enables early
              // release of pages
              HANDLE hFile =
                CreateFileA(f_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
              NNTR_THROW_IF((hFile == INVALID_HANDLE_VALUE), std::runtime_error)
                << "CreateFileA failed";

              HANDLE hMap =
                CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
              NNTR_THROW_IF((hMap == NULL), std::runtime_error)
                << "CreateFileMapping failed";

              char *view = static_cast<char *>(
                MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
              NNTR_THROW_IF((view == nullptr), std::runtime_error)
                << "MapViewOfFile failed";

              node->read(view, false, exec_mode, fsu_mode,
                         std::numeric_limits<size_t>::max(), true,
                         model_file_fd);

              // Early unmap: let the OS reclaim the working set ASAP
              UnmapViewOfFile(view);
              CloseHandle(hMap);
              CloseHandle(hFile);
#else
              // POSIX: map per-task, advise kernel, drop pages, unmap
              size_t f_size = 0;
              void *mmap_ptr = nullptr;
              {
                nntrainer::load_trace::Scope _lt(nntrainer::load_trace::MAP);
                int fd = ::open(f_path.c_str(), O_RDONLY);
                NNTR_THROW_IF((fd == -1), std::invalid_argument)
                  << "Cannot open file : " << f_path;

                struct stat st {};
                NNTR_THROW_IF((::fstat(fd, &st) == -1), std::invalid_argument)
                  << "Cannot get file info (fstat): " << f_path;

                f_size = static_cast<size_t>(st.st_size);
                mmap_ptr =
                  ::mmap(nullptr, f_size, PROT_READ, MAP_PRIVATE, fd, 0);
                ::close(fd); // fd not needed after mmap
                NNTR_THROW_IF((mmap_ptr == MAP_FAILED), std::runtime_error)
                  << "mmap failed";
              }

              // A node's weights are one contiguous run and the nodes are
              // handed out by an atomic counter, so this walks the file front
              // to back -- it is a streaming read, not a scattered one. RANDOM
              // was the advice here, and it turns off BOTH kernel readahead AND
              // fault-around, which makes every 4 KiB page its own synchronous
              // flash trip at queue depth 1. Measured on Adreno 840 with a
              // 3.17 GiB weight file: 830 803 major faults for 830 767 file
              // pages (every page faulted exactly once, nothing re-read) moving
              // 176 MB/s, on a device that reads the same file at 2.2-2.5 GB/s.
              // SEQUENTIAL restores the readahead window and adds drop-behind,
              // which also bounds the page-cache footprint of the load.
              {
                nntrainer::load_trace::Scope _lt(nntrainer::load_trace::MADV);
                (void)::posix_madvise(mmap_ptr, f_size, POSIX_MADV_SEQUENTIAL);
              }

              // Publish the mapping to the reaper for the duration of the read.
              if (reap_ms > 0) {
                std::lock_guard<std::mutex> reap_lk(reap_mtx);
                reap_live.emplace_back(mmap_ptr, f_size);
              }

              char *view = static_cast<char *>(mmap_ptr);
              {
                nntrainer::load_trace::Scope _lt(
                  nntrainer::load_trace::NODE_READ);
                node->read(view, false, exec_mode, fsu_mode,
                           std::numeric_limits<size_t>::max(), true,
                           model_file_fd);
              }

              // Retract it BEFORE the unmap below, so the reaper can never
              // advise a region this worker has already given back.
              if (reap_ms > 0) {
                std::lock_guard<std::mutex> reap_lk(reap_mtx);
                for (auto it = reap_live.begin(); it != reap_live.end(); ++it)
                  if (it->first == mmap_ptr) {
                    reap_live.erase(it);
                    break;
                  }
              }

              // Early drop: pages no longer needed; bounds peak RSS during the
              // overlap. NOTE posix_madvise(POSIX_MADV_DONTNEED) is a
              // documented NO-OP in glibc -- it validates its arguments and
              // returns 0 without freeing a page -- so the raw madvise is
              // required for the drop to actually happen. The mapping is
              // read-only MAP_PRIVATE, so dropped pages simply re-fault from
              // the page cache if touched again: bytes read are identical
              // either way.
              {
                nntrainer::load_trace::Scope _lt(nntrainer::load_trace::DROP);
                (void)::madvise(mmap_ptr, f_size, MADV_DONTNEED);

                ::munmap(mmap_ptr, f_size);
              }
#endif
            }

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
            if (cuda_wprefetch_load) {
              for (unsigned int wi = 0; wi < node->getNumWeights(); ++wi) {
                nntrainer::Tensor &wt = node->getWeight(wi);
                if (wt.getDataType() == ml::train::TensorDim::DataType::QS4CX)
                  (void)nntrainer::cuda::cuda_fc_qs4cx_prefetch_weight(
                    wt.getData<uint8_t>(), wt.width(), wt.height());
              }
            }
#endif

            // [load-drop interleave] this node's weight bytes (payload + any
            // trailing scales) are fully materialized in their final tensor
            // storage -- on BOTH the stream and the mmap arm above: hand the
            // node to the registered per-weight hook NOW, so derived-cache
            // builds and plain-payload drops overlap the other workers' reads
            // instead of waiting for the join. Failures are contained -- the
            // application's post-load sweep is the fallback for anything the
            // hook missed.
            if (weight_load_hook) {
              std::lock_guard<std::mutex> hook_lk(load_hook_mtx);
              try {
                weight_load_hook(*node, node->getRunContext(),
                                 weight_load_hook_data);
              } catch (const std::exception &e) {
                ml_logw("weight-load hook failed on node %s: %s",
                        node->getName().c_str(), e.what());
              } catch (...) {
                ml_logw(
                  "weight-load hook failed on node %s (non-std exception)",
                  node->getName().c_str());
              }
            }
          }
        } catch (...) {
          std::lock_guard<std::mutex> lock(load_error_mutex);
          if (!load_error)
            load_error = std::current_exception();
        }
      };

      std::vector<std::thread> threads;
      threads.reserve(num_load_threads);
#if !defined(_WIN32)
      std::thread reaper;
      if (reap_ms > 0)
        reaper = std::thread([&]() {
          while (!reap_stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(reap_ms));
            std::lock_guard<std::mutex> reap_lk(reap_mtx);
            for (auto &e : reap_live)
              (void)::madvise(e.first, e.second, MADV_DONTNEED);
          }
        });
#endif
      // The mapped arm reads the file through page faults; start the kernel
      // reading it ahead of the workers (see WeightFileReadAhead). FSU keeps
      // the file as its weight store and reads only what it needs.
      std::optional<WeightFileReadAhead> read_ahead;
      if (MMAP_READ && !fsu_mode)
        read_ahead.emplace(model_file_fd);
      const auto _lt_fanout_t0 = std::chrono::steady_clock::now();
      for (size_t t = 0; t < num_load_threads; ++t) {
        threads.emplace_back(load_worker);
      }
      for (auto &t : threads) {
        if (t.joinable())
          t.join();
      }
      read_ahead.reset();
      nntrainer::load_trace::add(
        nntrainer::load_trace::WALL,
        (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - _lt_fanout_t0)
          .count());
#if !defined(_WIN32)
      reap_stop.store(true, std::memory_order_relaxed);
      if (reaper.joinable())
        reaper.join();
#endif

#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
      // Every load-time weight pack has committed its record by now: finalize
      // a pending rewrite off-thread, so writing the pack costs the load
      // nothing. The writer is joined at exit.
      v8c_pack::load_complete();

      // The v8c build hands each non-blocking chunk upload's host staging to
      // a registry that the NEXT prebuild drains -- so the last builds of the
      // load are never drained by anything, and their staging stays resident
      // for the life of the process. Measured on an Adreno 840 handset with
      // gemma4 E2B: five 9.0 MiB staging vectors, 45.0 MiB, still fully
      // resident in the middle of decode and visible in smaps as
      // `[anon:scudo:secondary]` blocks of exactly N x v8c_row_bytes. Drain
      // them here, where the load -- and therefore every prebuild -- is over.
      // Waiting is free at this point: the uploads were enqueued on an
      // in-order queue long ago and completed while the rest of the load ran.
      v8c_flush_pending_uploads();

      // The per-weight scale/row-sum pairs are carved out of a shared device
      // arena and staged host-side while the load runs, so the model's aux
      // plane costs a handful of transfers instead of one per weight. Write
      // the chunk still open and seal the arena: after this a weight built
      // lazily at its first dispatch writes its own pair, so nothing can read
      // an aux buffer whose bytes have not landed.
      v8c_flush_aux_arena();
#endif

#if !defined(_WIN32)
      // [page cache] The per-node madvise(MADV_DONTNEED) above drops this
      // process's PTEs; it does not touch the SYSTEM page cache the read
      // filled. A 1.09 GiB weight pack therefore leaves 1.09 GiB of `Cached`
      // behind for the rest of the process, and the load also evicted whatever
      // was there before it. posix_fadvise(DONTNEED) hands those clean pages
      // back to the free list now that every weight is materialized in its
      // tensor storage.
      //
      // It is non-destructive by construction -- the pages are clean and
      // file-backed, so any later reader (an FSU/slim tensor's on-demand mmap
      // through this same fd) re-reads identical bytes, it just pays the trip.
      // FSU is excluded anyway, since there the file IS the weight store.
      //
      // MEASURED AND NOT ADOPTED AS A DEFAULT (Adreno 840, gemma4 E2B):
      // it returns 1.09 GiB of `Cached` and moves nothing a
      // caller can spend. VmHWM, RssFile peak and the MemAvailable drawdown
      // are all unchanged -- 1 772.8 / 1 525.3 / 1 749.9 MiB against the
      // control's 1 771.3 / 1 523.9 / 1 763.6 -- because clean page cache
      // already counts as available memory. What it does change is the next
      // load of the same pack, which is now cold: init 1 237 ms median
      // against the control's 788 ms. So it is OFF unless
      // NNTR_LOAD_FADV_DROP=1 asks for it, which is the right arm for a
      // one-shot process on a memory-tight system that will not load the same
      // model again.
      {
        static const bool fadv_drop = []() {
          const char *e = std::getenv("NNTR_LOAD_FADV_DROP");
          return (e != nullptr && e[0] == '1');
        }();
        if (fadv_drop && !fsu_mode && model_file_fd >= 0)
          (void)::posix_fadvise(model_file_fd, 0, 0, POSIX_FADV_DONTNEED);
      }
#endif

#if defined(__GLIBC__) && !defined(_WIN32)
      // Give the loader's freed transients (staging vectors, per-chunk derive
      // buffers) back to the OS: glibc keeps brk/arena tops cached after
      // free(), so they stay in this process's RSS until a trim. One call, at
      // the end of the positional load -- past the worker join above, and past
      // any per-weight cache build that ran inside the load, so the trim sees
      // everything the load transiently allocated. Opt out with
      // NNTR_MALLOC_TRIM=0.
      {
        static const bool malloc_trim_on = []() {
          const char *e = std::getenv("NNTR_MALLOC_TRIM");
          return !(e != nullptr && e[0] == '0');
        }();
        if (malloc_trim_on)
          ::malloc_trim(0);
      }
#endif

      nntrainer::load_trace::dump("model->load_weight");
      if (load_error)
        std::rethrow_exception(load_error);

    } else {
      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           ++iter) {
        (*iter)->read(model_file, false, exec_mode, fsu_mode);
      }

      try {
        /// this is assuming that the failure is allowed at the end of the file
        /// read. so, after this line, additional read shouldn't be called
        if (opt && istrequal(opt->getType(), "adam")) {
          std::string opt_type;
          opt_type.resize(4);
          model_file.read((char *)&opt_type[0], 4);

          if (istrequal(opt_type, "adam")) {
            for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
                 iter++) {
              (*iter)->read(model_file, true, exec_mode);
            }
          }
        }

        if (!fsu_mode && exec_mode == ml::train::ExecutionMode::TRAIN) {

          checkedRead(model_file, (char *)&epoch_idx, sizeof(epoch_idx),
                      "[NeuralNetwork::readModel] failed to read epoch_idx");
          checkedRead(model_file, (char *)&iter, sizeof(iter),
                      "[NeuralNetwork::readModel] failed to read iteration");
        }
      } catch (...) {
        std::cerr << "failed to read additional data like optimizer variable, "
                     "iteration, proceeding with default\n";
      }
    }

    ml_logi("read modelfile: %s",
            (v.size() == 2) ? v[1].c_str() : v[0].c_str());
    break;
  }

  case ml::train::ModelFormat::MODEL_FORMAT_INI_WITH_BIN: {
    int ret = loadFromConfig((v.size() == 2) ? v[1] : v[0]);
    throw_status(ret);
    auto &save_path = std::get<props::SavePath>(model_flex_props);
    if (!save_path.empty()) {
      checkedOpenStream<std::ifstream>(save_path,
                                       std::ios::in | std::ios::binary);
      load_path = save_path;
    }
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_INI: {
    int ret = loadFromConfig((v.size() == 2) ? v[1] : v[0]);
    throw_status(ret);
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_FLATBUFFER: {
    break;
  }

  case ml::train::ModelFormat::MODEL_FORMAT_ONNX: {
    int ret = loadFromConfig((v.size() == 2) ? v[1] : v[0]);
    throw_status(ret);
    break;
  }

  case ml::train::ModelFormat::MODEL_FORMAT_QNN: {
    // for now, we only support to QNN binary format for Inference mode.
    // expect to have the file path for qnn bin and nntrainer bin seperated by
    // ":" QNN bin ( graph ) : NNTrainer bin (weight)
    NNTR_THROW_IF(exec_mode != ExecutionMode::INFERENCE, std::invalid_argument)
      << "Only support QNN biarny for Infernece";
    NNTR_THROW_IF(!isFileExist(props::FilePath(v[0])), std::invalid_argument)
      << "Cannot open QNN context bin file";

    std::thread qnn_load([this, &v]() {
      int ret =
        ct_engine->getRegisteredContext("qnn")->load(props::FilePath(v[0]));
      throw_status(ret);
    });

    if (!fsu_mode && v.size() > 1) {
      NNTR_THROW_IF(!isFileExist(props::FilePath(v[1])), std::invalid_argument)
        << "Cannot open weight bin file";
      load(props::FilePath(v[1]), ml::train::ModelFormat::MODEL_FORMAT_BIN);
    } else if (fsu_mode) {
      NNTR_THROW_IF(v.size() <= 1, std::invalid_argument)
        << "Swap mode should run with loading a weight-bin file";
      NNTR_THROW_IF(!isFileExist(props::FilePath(v[1])), std::invalid_argument)
        << "Cannot open weight bin file";
      // model_graph.setFsuWeightPath(v[1]);
    }

    qnn_load.join();
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS: {
    NNTR_THROW_IF(!initialized, std::runtime_error)
      << "Cannot load safetensors if not initialized yet, path: " << file_path;

    const auto f_path = (v.size() == 2) ? v[1] : v[0];

    // Read header_size (8 bytes) + header JSON
    std::ifstream st_file(f_path, std::ios::in | std::ios::binary);
    NNTR_THROW_IF(!st_file.is_open(), std::runtime_error)
      << "Cannot open safetensors file: " << f_path;

    uint64_t header_size = 0;
    st_file.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
    NNTR_THROW_IF(!st_file, std::runtime_error)
      << "Failed to read safetensors header length from: " << f_path;

    std::string header_json(header_size, '\0');
    st_file.read(header_json.data(), static_cast<std::streamsize>(header_size));
    NNTR_THROW_IF(!st_file, std::runtime_error)
      << "Failed to read safetensors header from: " << f_path;
    st_file.close();

    // data_base: byte offset in file where the data section starts
    const size_t data_base =
      sizeof(uint64_t) + static_cast<size_t>(header_size);

    // Parse header: name -> entry (dtype, data offsets)
    const auto entries = safetensors::parseHeaderEntries(header_json);
    std::unordered_map<std::string, const safetensors::TensorEntry *> by_name;
    size_t data_size = 0;
    for (const auto &e : entries) {
      by_name.emplace(e.name, &e);
      data_size = std::max(data_size, static_cast<size_t>(e.offset_end));
    }

    /**
     * A tensor stored as F16 / BF16 for a weight requested in FP32 (e.g. a
     * norm gamma under FP32 activations) is converted on load instead of
     * being read as raw bytes of the wrong size. Such a weight is pointed at
     * the start of the data section for the bulk read below (any in-bounds
     * bytes) and filled with the converted values afterwards.
     */
    struct HalfToFloat {
      Tensor *var;
      size_t file_off;
      size_t count;
      bool bf16;
    };
    std::vector<HalfToFloat> half_to_float;
    std::vector<std::string> missing;

    // TQ2CX records are stored in the layout the header names (absent: the
    // row-major "rm"); the weights were sized and every kernel indexes them
    // in the process's stored layout, so the two must agree.
    std::string pack_tq2cx_layout;
    bool pack_tq2cx_layout_read = false;
    auto check_tq2cx_layout = [&](const std::string &name) {
      if (!pack_tq2cx_layout_read) {
        const auto meta = safetensors::parseMetadata(header_json);
        auto m = meta.find(TQ2CX_LAYOUT_META_KEY);
        pack_tq2cx_layout = m == meta.end() ? "rm" : m->second;
        pack_tq2cx_layout_read = true;
      }
      Tq2cxLayout pack;
      NNTR_THROW_IF(!tq2cxLayoutFromName(pack_tq2cx_layout, pack),
                    std::invalid_argument)
        << "safetensors: " << f_path << " stores TQ2CX in layout '"
        << pack_tq2cx_layout << "', which this build does not implement";
      const Tq2cxLayout kernel = tq2cxStoredLayout();
      NNTR_THROW_IF(pack != kernel, std::invalid_argument)
        << "safetensors: '" << name << "': pack layout "
        << tq2cxLayoutName(pack) << ", kernel layout "
        << tq2cxLayoutName(kernel) << " (the layout is set from the header's "
        << TQ2CX_LAYOUT_META_KEY << " / the nntr_config "
        << TQ2CX_LAYOUT_CONFIG_KEY << " before the model is initialized)";
    };

    // A Q4_0 tensor is repacked into an ISA-specific layout that is
    // indistinguishable from the header alone (save() records which one at
    // nntr_q4_0_isa, see above). Loading the wrong one is not an error at
    // the repack/dequant call site -- it silently reads garbage. Refuse it
    // here instead.
    auto metadata = safetensors::parseMetadata(header_json);
    auto isa_it = metadata.find("nntr_q4_0_isa");
    if (isa_it != metadata.end()) {
#if defined(__aarch64__) || defined(__arm__)
      const char *expected_isa = "arm";
#else
      const char *expected_isa = "x86";
#endif
      NNTR_THROW_IF(isa_it->second != expected_isa, std::runtime_error)
        << "safetensors file " << f_path << " has Q4_0 weights packed for '"
        << isa_it->second << "' but this build expects '" << expected_isa
        << "'. Re-quantize with --isa " << expected_isa
        << " (nntr_quantize), or run on a matching target.";
    }

    // Assign file offsets to each weight by name
    std::unordered_set<const Tensor *> visited_st;
    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      auto weights = (*iter)->getRunContext().getWeights();
      for (auto weight : weights) {
        if (!visited_st.insert(&weight->getVariableRef()).second)
          continue;
        const std::string &name = weight->getName();
        auto it = by_name.find(name);
        if (it == by_name.end()) {
          missing.push_back(name);
          continue;
        }
        const safetensors::TensorEntry &e = *it->second;
        Tensor &var = weight->getVariableRef();
        const size_t file_off = data_base + e.offset_start;
        if (var.getDataType() == TensorDim::DataType::TQ2CX)
          check_tq2cx_layout(name);
        const size_t stored = e.offset_end - e.offset_start;

        // Resolve the QS4CX record stride from the header, which is the one
        // place a stride is written down: the .bin has to infer it (from the
        // whole file's size, then from the first QS4CX record's bytes) because
        // a record names neither of the two, but a
        // safetensors entry carries this tensor's own byte extent. So no
        // inference and no per-file assumption here -- ask the record that was
        // actually written. A weight whose extent matches neither stride is
        // left alone; the read below reports it far better than a guess would.
        if (var.getDataType() == TensorDim::DataType::QS4CX &&
            !legacy_int4_model) {
          const TensorDim &d = var.getDim();
          if (stored == QS4CX_Tensor::recordBytes(d.height(), d.width(), false))
            var.setQs4cxRecordPadded(false);
          else if (stored ==
                   QS4CX_Tensor::recordBytes(d.height(), d.width(), true))
            var.setQs4cxRecordPadded(true);
        }

        const size_t needed = var.getMemoryBytes();
        // A legacy QINT4 record is transcoded on read and has its own size
        // rule (see the .bin walk); take its extent as written.
        if (stored == needed || legacy_int4_model) {
          var.setFileOffset(file_off);
          continue;
        }
        const bool half = e.dtype == "F16" || e.dtype == "BF16";
        NNTR_THROW_IF(!half || var.getDataType() != TensorDim::DataType::FP32 ||
                        stored * 2 != needed || var.isVirtual(),
                      std::invalid_argument)
          << "safetensors: '" << name << "' is stored as " << e.dtype << " ("
          << stored << " bytes) but the weight is " << var.getDim() << " ("
          << needed << " bytes)";
        NNTR_THROW_IF(needed > data_size, std::invalid_argument)
          << "safetensors: data section too small to stage '" << name << "'";
        var.setFileOffset(data_base);
        half_to_float.push_back(
          {&var, file_off, stored / 2, e.dtype == "BF16"});
      }
    }
    if (!missing.empty()) {
      std::ostringstream names;
      for (size_t i = 0; i < missing.size() && i < 8; ++i)
        names << (i ? ", " : "") << missing[i];
      ml_logw("safetensors: %zu weight(s) not found in %s, left as "
              "initialized: %s%s",
              missing.size(), f_path.c_str(), names.str().c_str(),
              missing.size() > 8 ? ", ..." : "");
    }
    NNTR_THROW_IF(!half_to_float.empty() &&
                    exec_mode != ml::train::ExecutionMode::INFERENCE,
                  std::invalid_argument)
      << "safetensors: F16/BF16 to FP32 conversion on load is inference only";

    auto convert_half_weights = [&]() {
      std::ifstream in(f_path, std::ios::in | std::ios::binary);
      NNTR_THROW_IF(!in.is_open(), std::runtime_error)
        << "Cannot open safetensors file: " << f_path;
      std::vector<uint16_t> buf;
      for (const auto &h : half_to_float) {
        buf.resize(h.count);
        in.seekg(static_cast<std::streamoff>(h.file_off));
        in.read(reinterpret_cast<char *>(buf.data()),
                static_cast<std::streamsize>(h.count * sizeof(uint16_t)));
        NNTR_THROW_IF(!in, std::runtime_error)
          << "safetensors: failed to read " << h.var->getName();
        float *dst = h.var->getData<float>();
        for (size_t i = 0; i < h.count; ++i) {
          if (h.bf16) {
            const uint32_t bits = static_cast<uint32_t>(buf[i]) << 16;
            std::memcpy(dst + i, &bits, sizeof(float));
          } else {
            dst[i] = compute_fp16_to_fp32(buf[i]);
          }
        }
      }
    };

    if (exec_mode == ml::train::ExecutionMode::INFERENCE) {
      model_file_fd = ::open(f_path.c_str(), O_RDONLY);
      NNTR_THROW_IF((model_file_fd == -1), std::invalid_argument)
        << "Cannot open safetensors file: " << f_path;

      // A bounded pool of workers takes the nodes in graph order, as the .bin
      // loader does. One thread per node used to map the whole file once per
      // node AT THE SAME TIME: a 7 GiB file times a hundred-odd nodes exceeds
      // the 39-bit user address space of an arm64 Android process, and mmap
      // fails there. The worker count bounds the live mappings.
      constexpr size_t MAX_ST_LOAD_THREADS = 8;
      std::vector<std::shared_ptr<LayerNode>> st_nodes(model_graph.cbegin(),
                                                       model_graph.cend());
      std::atomic<size_t> st_next{0};
      std::vector<std::thread> threads;
      const size_t st_workers =
        std::min<size_t>(st_nodes.size(), MAX_ST_LOAD_THREADS);
      threads.reserve(st_workers);
      for (size_t w = 0; w < st_workers; ++w) {
        threads.emplace_back([&]() {
          for (size_t idx = st_next.fetch_add(1, std::memory_order_relaxed);
               idx < st_nodes.size();
               idx = st_next.fetch_add(1, std::memory_order_relaxed)) {
            auto node = st_nodes[idx];
            if (!MMAP_READ) {
              auto local_file = checkedOpenStream<std::ifstream>(
                f_path, std::ios::in | std::ios::binary);
              node->read(local_file, false, exec_mode, fsu_mode,
                         std::numeric_limits<size_t>::max(), true,
                         model_file_fd);
            } else {
#if defined(_WIN32)
              HANDLE hFile =
                CreateFileA(f_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
              NNTR_THROW_IF((hFile == INVALID_HANDLE_VALUE), std::runtime_error)
                << "CreateFileA failed for safetensors file: " << f_path;

              HANDLE hMap =
                CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
              NNTR_THROW_IF((hMap == NULL), std::runtime_error)
                << "CreateFileMapping failed for safetensors file: " << f_path;

              char *view = static_cast<char *>(
                MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
              NNTR_THROW_IF((view == nullptr), std::runtime_error)
                << "MapViewOfFile failed for safetensors file: " << f_path;

              node->read(view, false, exec_mode, fsu_mode,
                         std::numeric_limits<size_t>::max(), true,
                         model_file_fd);

              UnmapViewOfFile(view);
              CloseHandle(hMap);
              CloseHandle(hFile);
#else
              int fd = ::open(f_path.c_str(), O_RDONLY);
              NNTR_THROW_IF((fd == -1), std::invalid_argument)
                << "Cannot open safetensors file: " << f_path;

              struct stat st {};
              NNTR_THROW_IF((::fstat(fd, &st) == -1), std::invalid_argument)
                << "Cannot stat safetensors file: " << f_path;

              const size_t f_size = static_cast<size_t>(st.st_size);
              void *mmap_ptr =
                ::mmap(nullptr, f_size, PROT_READ, MAP_PRIVATE, fd, 0);
              ::close(fd);
              NNTR_THROW_IF((mmap_ptr == MAP_FAILED), std::runtime_error)
                << "mmap failed for safetensors file: " << f_path;

              // Streaming read, not a scattered one -- see the sibling site
              // in the positional .bin arm above for the fault census that
              // retired POSIX_MADV_RANDOM here.
              (void)::posix_madvise(mmap_ptr, f_size, POSIX_MADV_SEQUENTIAL);

              char *view = static_cast<char *>(mmap_ptr);
              node->read(view, false, exec_mode, fsu_mode,
                         std::numeric_limits<size_t>::max(), true,
                         model_file_fd);

              (void)::posix_madvise(mmap_ptr, f_size, POSIX_MADV_DONTNEED);
              ::munmap(mmap_ptr, f_size);
#endif
            }
          }
        });
      }
      for (auto &t : threads) {
        if (t.joinable())
          t.join();
      }
      convert_half_weights();
    } else {
      // TRAINING mode: sequential read
      std::ifstream st_in(f_path, std::ios::in | std::ios::binary);
      NNTR_THROW_IF(!st_in.is_open(), std::runtime_error)
        << "Cannot open safetensors file for training load: " << f_path;
      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           ++iter) {
        (*iter)->read(st_in, false, exec_mode, fsu_mode);
      }
    }

    ml_logi("read safetensors model file: %s", f_path.c_str());
    break;
  }
  default:
    throw nntrainer::exception::not_supported(
      "loading with given format is not supported yet");
  }
}

float NeuralNetwork::getLoss() {
  loss = 0.0f;

  for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
    loss += (*iter)->getLoss();
  }
  return loss;
}

void NeuralNetwork::setLoss(float l) { loss = l; }

NeuralNetwork &NeuralNetwork::copy(NeuralNetwork &from) {
  if (this != &from) {
    model_props = from.model_props;
    model_flex_props = from.model_flex_props;
    loss = from.loss;
    opt = from.opt;

    model_graph.copy(from.model_graph);
  }
  return *this;
}

void NeuralNetwork::saveModelIni(const std::string &file_path) {
  NNTR_THROW_IF(isFileExist(file_path), std::invalid_argument)
    << "There is already a file, overriding to the existing file is not "
       "permitted, path: "
    << file_path;

  std::vector<IniSection> sections;

  IniSection model_section = IniSection::FromExportable("model", *this);
  model_section.setEntry("type", "NeuralNetwork");
  sections.push_back(model_section);

  auto add_section_if_any = [&sections](const std::string &section_name,
                                        auto obj_ptr, auto pred) {
    if (pred(obj_ptr)) {
      IniSection s = IniSection::FromExportable(section_name, *obj_ptr);
      s.setEntry("type", obj_ptr->getType());
      sections.push_back(s);
    }
  };

  add_section_if_any("optimizer", opt,
                     [](const auto &obj) { return static_cast<bool>(obj); });

  auto &[train_buffer, valid_buffer, test_buffer] = data_buffers;
  auto data_buffer_valid = [](const auto &buffer) {
    return buffer && buffer->isSerializable(
                       ml::train::ExportMethods::METHOD_STRINGVECTOR);
  };

  add_section_if_any("train_set", train_buffer, data_buffer_valid);
  add_section_if_any("valid_set", valid_buffer, data_buffer_valid);
  add_section_if_any("test_set", test_buffer, data_buffer_valid);

  IniWrapper wrapper("model_saver", sections);
  wrapper.save_ini(file_path);

  IniGraphInterpreter interpreter;
  interpreter.serialize(graph_representation, file_path);
}

/**
 * @brief Name of a tensor data type for an error message
 */
static std::string dtypeName(TensorDim::DataType type) {
  try {
    return str_converter<enum_class_prop_tag, TensorDataTypeInfo>::to_string(
      type);
  } catch (std::invalid_argument &) {
    return "dtype " + std::to_string(static_cast<int>(type));
  }
}

bool NeuralNetwork::validateInput(sharedConstTensors X) {
  auto input_dim = getInputDimension();
  if (X.size() != input_dim.size()) {
    ml_loge("Error: provided number of inputs %d, required %d", (int)X.size(),
            (int)input_dim.size());
    return false;
  }

  const auto &input_names = model_graph.getInputList();
  for (unsigned int dim = 0; dim < input_dim.size(); dim++) {
    /// A dtype mismatch is not converted: the tensor would be bound with the
    /// model's element size and read as garbage, so name both dtypes.
    const auto expected = input_dim[dim].getDataType();
    const auto given = X[dim]->getDataType();
    NNTR_THROW_IF(expected != given, std::invalid_argument)
      << "input " << dim << " ("
      << (dim < input_names.size() ? input_names[dim] : std::string("?"))
      << "): the model expects " << dtypeName(expected) << " data but "
      << dtypeName(given) << " was given";

    if (input_dim[dim] != X[dim]->getDim()) {
      ml_loge("Error: provided input shape does not match required shape");
      std::stringstream ss;
      ss << X[dim]->getDim();
      ml_loge("Provided tensor summary : %s", ss.str().c_str());

      ss.str(std::string());
      ss << input_dim[dim];
      ml_loge("Required tensor summary : %s", ss.str().c_str());
      return false;
    }
  }

  return true;
}

sharedConstTensors NeuralNetwork::inference(sharedConstTensors X,
                                            bool free_mem) {
  return inference(X, {}, free_mem);
}

sharedConstTensors NeuralNetwork::inference(sharedConstTensors X,
                                            sharedConstTensors label,
                                            bool free_mem) {
  if (model_graph.getBatchSize() != X[0]->batch()) {
    model_graph.setBatchSize(X[0]->batch());
  }

  sharedConstTensors out;
  if (!validateInput(X))
    throw std::invalid_argument("Input validation failed.");

  // QNN activation tensors are rpcmem-backed and registered with the DSP, so
  // their address must stay stable across decode tokens. For QNN graphs reuse
  // the already-allocated pool (allocate once) instead of the per-call
  // deallocateTensors()+allocateTensors() that allocate() does: reallocating
  // every token hands out NEW rpcmem addresses, defeats registerQnnTensor()'s
  // findMatchingPtr cache (every token re-runs rpcmem_to_fd()+memRegister()),
  // and churns the scarce contiguous CMA pool until rpcmem_to_fd fails under
  // the app UI's GPU dmabuf pressure. allocateTensors() no-ops when already
  // allocated; a free_mem=true caller deallocates at the end so the next call
  // re-allocates. CPU/GPU keep the realloc-per-call behavior — reusing the pool
  // there yields degenerate output (verified: gemma4 CPU loops "... is ...
  // is").
  if (reuse_inference_tensor_pool_)
    model_graph.allocateTensors(ExecutionMode::INFERENCE);
  else
    allocate(ExecutionMode::INFERENCE);

  int nn_foward;
  PROFILE_TIME_REGISTER_EVENT(nn_foward, "nn_forward");
  PROFILE_TIME_START(nn_foward);
  out = forwarding(X, label, false);
  PROFILE_TIME_END(nn_foward);

  if (free_mem)
    /**
     * Free the memory needed for training before exiting.
     * Note that this does not free the weights for the model.
     * Weights of the model will be freed when the model is destroyed.
     */
    model_graph.deallocateTensors(false);

  /** Clear the set inputs and labels */
  model_graph.setInputsLabels({}, {});

  return out;
}

std::vector<float *>
NeuralNetwork::inference(unsigned int batch_size,
                         const std::vector<float *> &input,
                         const std::vector<float *> &label) {
  sharedConstTensors input_tensors, output_tensors;
  auto in_dim = getInputDimension();

  input_tensors.reserve(input.size());
  for (unsigned int idx = 0; idx < in_dim.size(); idx++) {
    in_dim[idx].batch(batch_size);
    input_tensors.emplace_back(
      MAKE_SHARED_TENSOR(mapExternalTensor(input[idx], in_dim[idx])));
  }

  if (!label.empty()) {
    sharedConstTensors label_tensors;
    auto label_dim = getOutputDimension();
    label_tensors.reserve(label.size());
    for (unsigned int idx = 0; idx < label_dim.size(); idx++) {
      label_dim[idx].batch(batch_size);
      label_tensors.emplace_back(
        MAKE_SHARED_TENSOR(mapExternalTensor(label[idx], label_dim[idx])));
    }
    output_tensors = inference(input_tensors, label_tensors, false);
  } else {
    output_tensors = inference(input_tensors, false);
  }

  std::vector<float *> output;
  output.reserve(output_tensors.size());

  for (auto &out : output_tensors) {
    auto out_t = *out.get();
    output.push_back(out_t.getData());
  }

  return output;
}

sharedConstTensors
NeuralNetwork::incremental_inference(sharedConstTensors X,
                                     unsigned int init_seq_len,
                                     unsigned int from, unsigned int to) {
  return incremental_inference(X, {}, init_seq_len, from, to);
}

sharedConstTensors NeuralNetwork::incremental_inference(
  sharedConstTensors X, sharedConstTensors label, unsigned int init_seq_len,
  unsigned int from, unsigned int to) {
  if (model_graph.getBatchSize() != X[0]->batch()) {
    model_graph.setBatchSize(X[0]->batch());
  }

  sharedConstTensors out;
  if (!validateInput(X))
    throw std::invalid_argument("Input validation failed.");

  if (!from) {
    model_graph.allocateTensors(ExecutionMode::INFERENCE);
  }

  int nn_foward;
  PROFILE_TIME_REGISTER_EVENT(nn_foward, "nn_forward");
  PROFILE_TIME_START(nn_foward);

  // The execution seam: one decode or prefill step is dispatched through the
  // resolved Context. The base runDecode is the plain walk below, so CPU and
  // OpenCL are byte-identical; a backend with its own decode strategy overrides
  // it instead of adding a compile-guarded block here.
  if (Context *dctx = getDecodeContext())
    out = dctx->runDecode(*this, from, to, X, label);
  else
    out = incremental_forwarding(from, to, X, label, false);

  PROFILE_TIME_END(nn_foward);

  /** @todo: deallocate tensor after incremental inference **/
  /** Clear the set inputs and labels */
  model_graph.setInputsLabels({}, {});

  return out;
}

std::vector<float *> NeuralNetwork::incremental_inference(
  unsigned int batch_size, const std::vector<float *> &input,
  const std::vector<float *> &label, unsigned int init_seq_len,
  unsigned int from, unsigned int to, bool output_hidden_state) {

  // auto start_in_neuralnet = std::chrono::high_resolution_clock::now();

  sharedConstTensors input_tensors, output_tensors;
  auto in_dim = getInputDimension();

  input_tensors.reserve(input.size());
  for (unsigned int idx = 0; idx < in_dim.size(); idx++) {
    in_dim[idx].batch(batch_size);
    input_tensors.emplace_back(
      MAKE_SHARED_TENSOR(mapExternalTensor(input[idx], in_dim[idx])));
  }

  // auto start_increment = std::chrono::high_resolution_clock::now();
  if (!label.empty()) {
    sharedConstTensors label_tensors;
    auto label_dim = getOutputDimension();
    label_tensors.reserve(label.size());
    for (unsigned int idx = 0; idx < label_dim.size(); idx++) {
      label_dim[idx].batch(batch_size);
      label_tensors.emplace_back(
        MAKE_SHARED_TENSOR(mapExternalTensor(label[idx], label_dim[idx])));
    }
    output_tensors = incremental_inference(input_tensors, label_tensors,
                                           init_seq_len, from, to);
  } else {
    output_tensors =
      incremental_inference(input_tensors, init_seq_len, from, to);
  }
  // auto end_increment = std::chrono::high_resolution_clock::now();
  std::vector<float *> output;

  for (auto &out : output_tensors) {
    auto out_t = *out.get();
    float *last_out_buf_data;

    if (output_hidden_state) {
      std::cout << "Warning: output_hidden_state is not supported yet.\n"
                << "Returning last hidden state only...\n"
                << "Please free output memory after use!";
    }
    const size_t buf_size = batch_size * out_t.getDim().getFeatureLen();
    last_out_buf_data = new float[buf_size];

    if (out->getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16

      nntrainer::getComputeOps()->scopy_fp16_to_fp32(
        buf_size, out_t.getData<_FP16>(), 1, last_out_buf_data, 1);
#else
      throw std::invalid_argument("Error: enable-fp16 is not set");
#endif
    } else if (out->getDataType() == ml::train::TensorDim::DataType::FP32) {

      std::memcpy(last_out_buf_data, out_t.getData(), sizeof(float) * buf_size);
    }

    output.push_back(last_out_buf_data);
  }
  // auto end_net_inference = std::chrono::high_resolution_clock::now();
  // auto prepare =
  // std::chrono::duration_cast<std::chrono::nanoseconds>(start_increment-start_in_neuralnet);
  // auto run_inf =
  // std::chrono::duration_cast<std::chrono::nanoseconds>(end_increment-start_increment);;
  // auto out_gen =
  // std::chrono::duration_cast<std::chrono::nanoseconds>(end_net_inference-end_increment);;
  // auto net_gen =
  // std::chrono::duration_cast<std::chrono::nanoseconds>(end_net_inference-start_in_neuralnet);

  // std::cout <<"prepare : "<< prepare.count() << " run_inf : "<<
  // run_inf.count() << " out_gen : "<< out_gen.count()<<std::endl; std::cout <<
  // "-------- net_inference: "<< net_gen.count() << std::endl;

  return output;
}

void NeuralNetwork::resetInputDimension(std::vector<TensorDim> dims) {
  model_graph.resetInputDimension(dims);
}

int NeuralNetwork::setDataset(const DatasetModeType &mode,
                              std::shared_ptr<ml::train::Dataset> dataset) {
  return setDataBuffer(mode, std::static_pointer_cast<DataBuffer>(dataset));
}

int NeuralNetwork::allocate(ExecutionMode mode) {
  model_graph.deallocateTensors();
  model_graph.allocateTensors(mode);

  return ML_ERROR_NONE;
}

/**
 * @brief Hand this model's cached GPU residency back before the tensors go.
 *
 * The OpenCL v8c FC path keeps a process-lifetime cache of packed int4 weight
 * buffers keyed on each weight's host address, and nothing released it when a
 * model was destroyed -- the plane stayed on the device until exit (measured
 * 566 MiB per load of the dense 1.5B; the GPU ledger showed 619 -> 1110 -> 1599
 * MiB live over three loads of the SAME model in one process). The next load
 * then built a second plane beside the first, and the Intel driver's buffer
 * create + upload cost grows with the live total, which is what made loads
 * 2..N ~580 ms slower than load 1.
 *
 * The release itself is cl_fc_release_caches() -- the ONE hook that owns the
 * FC path's process-lifetime state (the weight packs, the norm-site strike
 * counters, the quantisation handoffs in the shared scratch, the handle epoch).
 * It is the same hook CausalLM::releaseDeviceCaches() calls after the last
 * model of a handle is gone; this is the engine-side entry point, so a consumer
 * that only ever destroys a model still gets the device memory back. Both entry
 * points are idempotent and a second call finds an empty state.
 *
 * Whole-cache, not per-weight: the arena release that follows in the app hook
 * requires the pack cache to be EMPTY (every pack holds sub-buffers carved from
 * the arena's chunks), which a per-weight sweep cannot promise for an entry
 * whose tensor is already gone. The cost of the wider granularity is that a
 * process holding two models rebuilds the surviving model's packs lazily; the
 * app hook already clears at this granularity, and a pack is pure derived
 * state.
 *
 * Still called BEFORE deallocateTensors(), where the per-weight release had to
 * be: the packs' host-pointer keys name memory the tensors own.
 */
void NeuralNetwork::releaseGpuWeightCaches() {
#if defined(ENABLE_OPENCL)
  try {
    cl_fc_release_caches();
  } catch (...) {
    // A teardown must not throw on a best-effort cache release.
  }
#endif
}

int NeuralNetwork::deallocate() {
  try {
    releaseGpuWeightCaches();
    model_graph.deallocateTensors(true);
    return ML_ERROR_NONE;
  } catch (const std::exception &e) {
    std::cerr << "Error occurred during deallocation of NeuralNetwork: "
              << e.what() << std::endl;
    return ML_ERROR_UNKNOWN;
  }
}

int NeuralNetwork::train(const std::vector<std::string> &values,
                         std::function<bool(void *)> stop_cb,
                         void *stop_user_data,
                         std::function<void(void *)> epoch_complete_cb,
                         void *epoch_user_data) {
  int status = ML_ERROR_NONE;

  if (data_buffers[static_cast<int>(DatasetModeType::MODE_TRAIN)] == nullptr) {
    ml_loge("Cannot initialize the model without the train data buffer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!opt) {
    ml_loge("Cannot train network without optimizer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  setTrainConfig(values);

  /** set batch size just before training */
  model_graph.setBatchSize(
    std::get<props::TrainingBatchSize>(model_flex_props));

  status = allocate(ExecutionMode::TRAIN);
  NN_RETURN_STATUS();

  status =
    train_run(stop_cb, stop_user_data, epoch_complete_cb, epoch_user_data);
  NN_RETURN_STATUS();

  /**
   * Free the memory needed for training before exiting.
   * Note that this does not free the weights for the model.
   * Weights of the model will be freed when the model is destroyed.
   */
  model_graph.deallocateTensors(false);
  return status;
}

/**
 * @brief     Run NeuralNetwork train with callback function by user
 */
int NeuralNetwork::train_run(
  std::function<bool(void *userdata)> stop_cb, void *stop_user_data,
  std::function<void(void *userdata)> epoch_complete_cb,
  void *epoch_user_data) {
  int status = ML_ERROR_NONE;

  if (!std::get<props::ContinueTrain>(model_flex_props)) {
    epoch_idx = 0;
    iter = 0;
    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      (*iter)->clearOptVar();
    }
  }

  auto batch_size = std::get<props::TrainingBatchSize>(model_flex_props);

  auto const &outputs = model_graph.getOutputTensors();
  auto in_dims = model_graph.getInputDimension();
  auto label_dims = model_graph.getOutputDimension();

  auto &[train_buffer, valid_buffer, test_buffer] = data_buffers;

  if (train_buffer == nullptr) {
    ml_loge("[NeuralNetworks] there is no train dataset!");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /**
   * @brief run a single epoch with given callback, @a auto is used instead of
   * std::function for performance measure
   * @param buffer buffer to run
   * @param shuffle whether to shuffle or not
   * @param on_iteration_fetch function that will receive reference to stat,
   * buffer which will be called every time data is fetched and set
   * @param on_epoch_end function that will receive reference to stat,
   * buffer which will be called on the epoch end
   */
  auto run_epoch = [this, &in_dims, &label_dims, &outputs, batch_size](
                     DataBuffer *buffer, bool shuffle,
                     auto &&on_iteration_fetch, auto &&on_iteration_update_stat,
                     auto &&on_epoch_end, RunStats &stat) {
    /// @todo managing metrics must be handled here as well!! for now it is
    /// handled in individual callbacks
    // RunStats stat;

    stat.accuracy = 0.0;
    stat.loss = 0.0;
    stat.num_iterations = 0;
    stat.num_correct_predictions = 0;
    stat.max_epoch = getEpochs();
    stat.epoch_idx = epoch_idx;

    std::future<std::shared_ptr<IterationQueue>> future_iq =
      buffer->startFetchWorker(in_dims, label_dims, shuffle);
    while (true) {
      ScopedView<Iteration> iter_view = buffer->fetch();
      if (iter_view.isEmpty()) {
        break;
      }
      auto &iteration = iter_view.get();
      if (iteration.batch() != static_cast<unsigned int>(batch_size)) {
        /// @todo support partial batch
        continue;
      }

      auto const &labels = iteration.getLabelsRef();
      auto const &inputs = iteration.getInputsRef();
      model_graph.setInputsLabels(inputs, labels);

      on_iteration_fetch(stat, *buffer);
      on_iteration_update_stat(stat, outputs, labels);
    }
    future_iq.get();
    on_epoch_end(stat, *buffer);

    if (stat.num_iterations == 0) {
      throw std::runtime_error("No data came while buffer ran");
    }

    return stat;
  };

  auto train_for_iteration =
    [this, stop_cb, stop_user_data](RunStats &stat, DataBuffer &buffer) {
      ml_logi("train for iteration");
      forwarding(true, stop_cb, stop_user_data);
      backwarding(iter++, stop_cb, stop_user_data);

      // To avoid unconsidered memory leak, we need to clear the cache
      model_graph.flushCache();

      if (!stop_cb(stop_user_data)) {
        std::cout << "#" << epoch_idx << "/" << getEpochs();
        ml_logi("# %d / %d", epoch_idx, getEpochs());
        auto loss = getLoss();
        buffer.displayProgress(stat.num_iterations, loss);
      }
    };

  auto update_train_stat = [this](RunStats &stat,
                                  const std::vector<Tensor> &outputs,
                                  const std::vector<Tensor> &labels) {
    stat.loss += getLoss();
    stat.num_iterations++;
  };

  auto train_epoch_end = [this, stop_cb, stop_user_data](RunStats &stat,
                                                         DataBuffer &buffer) {
    if (stat.num_iterations != 0) {
      stat.loss /= static_cast<float>(stat.num_iterations);
    } else {
      std::cerr << "stat.num_iterations is 0" << std::endl;
      return;
    }
    auto &save_path = std::get<props::SavePath>(model_flex_props);
    if (!stop_cb(stop_user_data)) {
      if (!save_path.empty()) {
        save(save_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
      }

      std::cout << "#" << epoch_idx << "/" << getEpochs()
                << " - Training Loss: " << stat.loss;
      ml_logi("# %d / %d - Training Loss: %f", epoch_idx, getEpochs(),
              stat.loss);
      ml_logd("[NNTrainer] Training epoch %d / %d finished successfully.",
              epoch_idx, getEpochs());
    } else {
      ml_logd("[NNTrainer] Training stopped by stop callback function during "
              "epoch %d.",
              epoch_idx);
    }
  };

  auto eval_for_iteration = [this, batch_size, stop_cb, stop_user_data](
                              RunStats &stat, DataBuffer &buffer) {
    forwarding(false, stop_cb, stop_user_data);
  };

  auto update_eval_stat = [batch_size, &update_train_stat](
                            RunStats &stat, const std::vector<Tensor> &outputs,
                            const std::vector<Tensor> &labels) {
    auto model_out = outputs[0].argmax();
    auto label_out = labels[0].argmax();

    for (unsigned int b = 0; b < batch_size; b++) {
      if (model_out[b] == label_out[b])
        stat.num_correct_predictions++;
    }

    update_train_stat(stat, outputs, labels);
  };

  auto eval_epoch_end = [this, batch_size, max_acc = 0.0f,
                         min_loss = std::numeric_limits<float>::max()](
                          RunStats &stat, DataBuffer &buffer) mutable {
    if (stat.num_iterations != 0) {
      stat.loss /= static_cast<float>(stat.num_iterations);
    } else {
      std::cerr << "stat.num_iterations is 0" << std::endl;
      return;
    }
    stat.accuracy = stat.num_correct_predictions /
                    static_cast<float>(stat.num_iterations * batch_size) *
                    100.0f;

    if (stat.accuracy > max_acc ||
        (stat.accuracy == max_acc && stat.loss < min_loss)) {
      max_acc = stat.accuracy;
      /// @note this is not actually 'the' min loss for whole time but records
      /// when data change
      min_loss = stat.loss;
      auto &save_best_path = std::get<props::SaveBestPath>(model_flex_props);
      if (!save_best_path.empty()) {
        save(save_best_path);
      }
    }
    std::cout << " >> [ Accuracy: " << stat.accuracy
              << "% - Validation Loss : " << stat.loss << " ]";
    ml_logi("[ Accuracy: %.2f %% - Validation Loss: %.5f", stat.accuracy,
            stat.loss);
  };

  PROFILE_MEM_ANNOTATE("TRAIN START");
  auto epochs = getEpochs();
  ml_logd("[NNTrainer] Starts training. Current epoch: %d. Total epochs: %d.",
          epoch_idx + 1, getEpochs());
  for (epoch_idx = epoch_idx + 1; epoch_idx <= epochs; ++epoch_idx) {
    if (stop_cb(stop_user_data)) {
      --epoch_idx;
      break;
    }
    training = run_epoch(train_buffer.get(), true, train_for_iteration,
                         update_train_stat, train_epoch_end, training);
    if (valid_buffer) {
      validation = run_epoch(valid_buffer.get(), false, eval_for_iteration,
                             update_eval_stat, eval_epoch_end, validation);
    }
    std::cout << '\n';
    epoch_complete_cb(epoch_user_data);
  }
  PROFILE_MEM_ANNOTATE("TRAIN END");

  if (test_buffer) {
    std::cout << "Evaluation with test data...\n";
    testing = run_epoch(test_buffer.get(), false, eval_for_iteration,
                        update_eval_stat, eval_epoch_end, testing);
  }

  /** Clear the set inputs and labels */
  model_graph.setInputsLabels({}, {});

  return status;
}

void swap(NeuralNetwork &lhs, NeuralNetwork &rhs) {
  {
    using std::swap;

    swap(lhs.model_props, rhs.model_props);
    swap(lhs.model_flex_props, rhs.model_flex_props);
    swap(lhs.load_path, rhs.load_path);
    swap(lhs.epoch_idx, rhs.epoch_idx);
    swap(lhs.iter, rhs.iter);
    swap(lhs.loss, rhs.loss);
    swap(lhs.opt, rhs.opt);
    swap(lhs.data_buffers, rhs.data_buffers);
    swap(lhs.initialized, rhs.initialized);
    swap(lhs.model_graph, rhs.model_graph);
    swap(lhs.graph_representation, rhs.graph_representation);
    swap(lhs.compiled, rhs.compiled);
    swap(lhs.loadedFromConfig, rhs.loadedFromConfig);
  }
}

int NeuralNetwork::addLayer(NodeType layer) {
  int status = ML_ERROR_NONE;

  if (initialized) {
    return ML_ERROR_NOT_SUPPORTED;
  }

  /** Insert the layer to the graph */
  model_graph.addLayer(layer);
  graph_representation.push_back(layer);

  return status;
}

NeuralNetwork &NeuralNetwork::copyConfiguration(NeuralNetwork &from) {
  if (this != &from) {
    model_props = from.model_props;
    model_flex_props = from.model_flex_props;
    loss = from.loss;
    opt = from.opt;

    NetworkGraph f_graph = from.getNetworkGraph();
    for (auto &l_node : f_graph.getLayerNodes()) {
      addLayer(static_cast<std::shared_ptr<ml::train::Layer>>(
        l_node->cloneConfiguration()));
    }
  }
  return *this;
}

NeuralNetwork::GraphType
NeuralNetwork::getUnsortedLayers(const std::string &input_layer,
                                 const std::string &output_layer) {
  return model_graph.getUnsortedLayers(input_layer, output_layer);
}

int NeuralNetwork::setOptimizer(
  std::shared_ptr<ml::train::Optimizer> optimizer) {
  if (initialized) {
    ml_loge("Cannot set optimizer if already initialized");
    return ML_ERROR_NOT_SUPPORTED;
  }

  opt = std::static_pointer_cast<OptimizerWrapped>(optimizer);

  return ML_ERROR_NONE;
}

int NeuralNetwork::setDataBuffer(const DatasetModeType &mode,
                                 std::shared_ptr<DataBuffer> data_buffer) {
  if (data_buffer == nullptr) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->data_buffers[static_cast<int>(mode)] = data_buffer;

  return ML_ERROR_NONE;
}

int NeuralNetwork::getLayer(const char *name,
                            std::shared_ptr<ml::train::Layer> *layer) {
  // We provide the layer change through the api with user's responsibility.
  //
  // if (compiled) {
  //   ml_loge("Cannot get compiled layer.");
  //   return ML_ERROR_NOT_SUPPORTED;
  // }

  *layer = std::static_pointer_cast<ml::train::Layer>(
    model_graph.getLayerNode(std::string(name)));
  return ML_ERROR_NONE;
}

void NeuralNetwork::printMetrics(std::ostream &out, unsigned int flags) {
  switch (flags) {
  case ML_TRAIN_SUMMARY_MODEL_TRAIN_LOSS:
    out << training.loss << std::endl;
    break;

  case ML_TRAIN_SUMMARY_MODEL_VALID_LOSS:
    out << validation.loss << std::endl;
    break;

  case ML_TRAIN_SUMMARY_MODEL_VALID_ACCURACY:
    out << validation.accuracy << std::endl;
    break;

  default:
    break;
  }
}

void NeuralNetwork::printPreset(std::ostream &out, unsigned int preset) {
  /** print neuralnet metrics */
  printMetrics(out, preset);
  if (preset > ML_TRAIN_SUMMARY_TENSOR)
    return;

  LayerNode::PrintPreset layer_preset = LayerNode::PrintPreset::PRINT_NONE;

  ///@todo match flags with preset
  unsigned int flags = PRINT_INST_INFO | PRINT_GRAPH_INFO | PRINT_PROP |
                       PRINT_OPTIMIZER | PRINT_METRIC;

  switch (preset) {
  case ML_TRAIN_SUMMARY_TENSOR:
    layer_preset = LayerNode::PrintPreset::PRINT_ALL;
    break;
  case ML_TRAIN_SUMMARY_LAYER:
    layer_preset = initialized ? LayerNode::PrintPreset::PRINT_SUMMARY
                               : LayerNode::PrintPreset::PRINT_SUMMARY_META;
    break;
  case ML_TRAIN_SUMMARY_MODEL:
    break;
  default:
    throw std::invalid_argument("given verbosity is invalid");
  }

  print(out, flags, layer_preset);
}

void NeuralNetwork::addWithReferenceLayers(
  const std::vector<std::shared_ptr<ml::train::Layer>> &reference,
  const std::string &scope, const std::vector<std::string> &input_layers,
  const std::vector<std::string> &start_layers,
  const std::vector<std::string> &end_layers,
  ml::train::ReferenceLayersType type,
  const std::vector<std::string> &type_properties) {
  std::vector<NodeType> casted_reference;
  casted_reference.reserve(reference.size());
  for (auto &node : reference) {
    casted_reference.emplace_back(std::static_pointer_cast<LayerNode>(node));
  }

  addWithReferenceLayers(casted_reference, scope, input_layers, start_layers,
                         end_layers, type, type_properties);
}

void NeuralNetwork::addWithReferenceLayers(
  const std::vector<std::shared_ptr<LayerNode>> &reference,
  const std::string &scope, const std::vector<std::string> &input_layers,
  const std::vector<std::string> &start_layers,
  const std::vector<std::string> &end_layers,
  ml::train::ReferenceLayersType type,
  const std::vector<std::string> &type_properties) {
  /// @todo below configuration should be extracted as a free function to make
  /// it more testable, and reused inside graph interpreter

  /// @note we can exploit connection to connection more fine grained, for now
  /// it is not supported but we can easily make this supported
  std::vector<std::shared_ptr<LayerNode>> nodes;
  nodes.reserve(reference.size());
  for (auto &node : reference) {
    nodes.push_back(node->cloneConfiguration());
  }

  auto start_conns =
    std::vector<Connection>(start_layers.begin(), start_layers.end());
  auto input_conns =
    std::vector<Connection>(input_layers.begin(), input_layers.end());
  auto end_conns =
    std::vector<Connection>(end_layers.begin(), end_layers.end());

  std::vector<std::unique_ptr<GraphRealizer>> realizers;

  realizers.emplace_back(new PreviousInputRealizer(start_conns));
  realizers.emplace_back(new SliceRealizer(start_conns, end_conns));

  if (!input_conns.empty()) {
    realizers.emplace_back(new InputRealizer(start_conns, input_conns));
  }

  if (type == ml::train::ReferenceLayersType::RECURRENT) {
    realizers.emplace_back(
      new RecurrentRealizer(type_properties, input_conns, end_conns));
  }

  if (!scope.empty()) {
    realizers.emplace_back(
      new RemapRealizer([&scope, &input_conns](std::string &name) {
        for (auto &i : input_conns) {
          if (i.getName() == name) {
            return;
          }
        }
        name = scope + "/" + name;
      }));
  }

  for (auto &realizer : realizers) {
    nodes = realizer->realize(nodes);
  }

  for (auto &node : nodes) {
    addLayer(node);
  }
}

void NeuralNetwork::exportTo(Exporter &exporter,
                             const ml::train::ExportMethods &method) const {
  exporter.saveResult(model_props, method, this);
  exporter.saveResult(model_flex_props, method, this);
}

void NeuralNetwork::print(std::ostream &out, unsigned int flags,
                          LayerNode::PrintPreset layerPrintPreset) {
  if (flags & PRINT_INST_INFO) {
    /// @todo uncomment this after implement getProperty (#1875)
    // out << "===================";
    // printInstance(out, this);
  }

  if (flags & PRINT_GRAPH_INFO) {
    unsigned int total_col_size = 80;
    std::vector<unsigned int> column_size = {20, 20, 20, 20};
    auto print_graph_layer_info =
      [column_size](std::ostream &out, std::vector<std::string> layer_info) {
        const auto &trim_string = [](std::string str,
                                     unsigned int column_width) {
          return str.size() < column_width ? str
                                           : str.substr(0, column_width - 1);
        };

        for (unsigned int i = 0; i < column_size.size(); ++i) {
          out << std::setw(column_size[i])
              << trim_string(layer_info[i], column_size[i]);
        }
        out << "\n";
      };

    out << std::string(total_col_size, '=') << '\n';
    print_graph_layer_info(
      out, {"Layer name", "Layer type", "Output dimension", "Input layer"});
    out << std::string(total_col_size, '=') << '\n';
    if (compiled) {
      props::GenericShape dim_property;

      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           iter++) {
        std::string first_dim;
        if (iter->getOutputDimensions().empty()) {
          first_dim = "";
        } else {
          dim_property.set(iter->getOutputDimensions()[0]);
          first_dim = to_string(dim_property);
        }
        const std::vector<std::string> &input_layer_names =
          iter->getInputConnections();
        std::string first_input_name =
          input_layer_names.empty() ? "" : input_layer_names[0];
        print_graph_layer_info(
          out, {iter->getName(), iter->getType(), first_dim, first_input_name});
        for (unsigned int i = 1; i < input_layer_names.size(); ++i) {
          dim_property.set(iter->getInputDimensions()[i]);
          print_graph_layer_info(out, {"", "", "", input_layer_names[i]});
        }
        out << std::string(total_col_size,
                           iter == model_graph.cend() - 1 ? '=' : '-')
            << '\n';
      }
    } else {
      auto &input_connection =
        std::get<std::vector<props::InputConnection>>(model_props);
      auto model_input = std::vector<Connection>(input_connection.begin(),
                                                 input_connection.end());
      auto is_actually_an_input_node =
        [model_input](graph_const_iterator<LayerNode> node) {
          return node->hasInputShapeProperty() or
                 std::any_of(model_input.begin(), model_input.end(),
                             [node](auto &conn) {
                               return node->getName() == conn.getName();
                             });
        };

      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           iter++) {
        const std::vector<std::string> &input_layer_names =
          iter->getInputConnections();

        /// @brief connection information.
        // Intended comment.
        // std::string first_input_name =
        //   input_layer_names.empty()
        //     ? (is_actually_an_input_node(iter) || iter ==
        //     model_graph.cbegin()
        //          ? ""
        //          : (iter - 1)->getName())
        //     : input_layer_names[0];
        print_graph_layer_info(out, {iter->getName(), iter->getType(), "", ""});
        for (unsigned int i = 1; i < input_layer_names.size(); ++i) {
          print_graph_layer_info(out, {"", "", "", ""});
        }
        out << std::string(total_col_size,
                           iter == model_graph.cend() - 1 ? '=' : '-')
            << '\n';
      }
    }
  }

  if (flags & PRINT_PROP) {
    /// @todo print neuralnet property
    /// @todo print mode (if it is eval or training)
  }

  if (flags & PRINT_OPTIMIZER) {
    /// @todo print optimizer (with print optimizer prop)
  }

  if (flags & PRINT_METRIC) {
    /// @todo print metric (currently it is done at printPreset as a
    /// workaround)
    /// @todo print loss function when it is not initialized. (if it is
    /// initialized, loss layer will be printed)
  }

  if (model_graph.empty()) {
    out << "model is empty!" << std::endl;
    return;
  }

  /** print layer properties */
  for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++)
    (*iter)->printPreset(out, layerPrintPreset);

  /// @todo Add status to check neuralnet has been run. #290
}

void NeuralNetwork::forEachLayer(
  std::function<void(ml::train::Layer &, RunLayerContext &, void *)> fn,
  void *user_data) {
  for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
    auto ln = std::static_pointer_cast<LayerNode>(*iter).get();
    fn(*ln, std::forward<RunLayerContext &>(ln->getRunContext()), user_data);
  };
}

void NeuralNetwork::exports(const ml::train::ExportMethods &method,
                            const std::string file_path) {
  switch (method) {
  case ml::train::ExportMethods::METHOD_TFLITE: {
#ifdef ENABLE_TFLITE_INTERPRETER
    nntrainer::TfliteInterpreter interpreter;

    /// We will call "serialize" method for the model which is already trained
    /// or allocated. So, we need to call deallocateTensors first to make sure
    /// `dealloc_weights == false`
    model_graph.deallocateTensors();
    model_graph.allocateTensors(ExecutionMode::INFERENCE);
    model_graph.setBatchSize(1); // For now, to inference batch size to be 1
    interpreter.serialize(graph_representation, file_path);
    model_graph.deallocateTensors();
#else
    throw std::runtime_error{
      "Export methods METHOD_TFLITE is not supported. Please enable tflite "
      "interpreter by set ENABLE_TFLITE_INTERPRETER=1"};
#endif
    break;
  }
  case ml::train::ExportMethods::METHOD_FLATBUFFER: {

    /**
     * @todo The current FLATBUFFER exporter only supports TRAIN execution mode.
     * It should be updated to support both train and inference mode.
     * It would be more natural to support inference by default since tflite is
     * typically used solely for inference
     */
    model_graph.deallocateTensors();
    model_graph.allocateTensors(ExecutionMode::TRAIN);
    break;
  }
  default:
    throw std::runtime_error{"Unsupported export method"};
  }
}
} /* namespace nntrainer */

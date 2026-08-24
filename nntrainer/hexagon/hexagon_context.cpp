// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   hexagon_context.cpp
 * @date   23 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  See hexagon_context.h.
 */

#include <hexagon_compute_ops.h>
#include <hexagon_context.h>
#include <hexagon_rpc_allocator.h>

#include <addition_layer.h>
#include <fc_layer.h>
#include <gate_up_layer.h>
#include <input_layer.h>
#include <multiout_layer.h>
#include <qkv_layer.h>

namespace nntrainer {

std::mutex hexagon_factory_mutex;

template <typename T>
const int HexagonContext::registerFactory(const FactoryType<T> factory,
                                          const std::string &key,
                                          const int int_key) {
  static_assert(isSupported<T>::value,
                "hexagon_context: given type is not supported for current "
                "context");

  auto &index = std::get<IndexType<T>>(factory_map);
  auto &str_map = std::get<StrIndexType<T>>(index);
  auto &int_map = std::get<IntIndexType>(index);

  std::string assigned_key = key == "" ? factory({})->getType() : key;

  std::transform(assigned_key.begin(), assigned_key.end(), assigned_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::lock_guard<std::mutex> lock(hexagon_factory_mutex);
  if (str_map.find(assigned_key) != str_map.end()) {
    std::stringstream ss;
    ss << "hexagon_context: cannot register factory with already taken key: "
       << key;
    throw std::invalid_argument(ss.str().c_str());
  }

  if (int_key != -1 && int_map.find(int_key) != int_map.end()) {
    std::stringstream ss;
    ss << "hexagon_context: cannot register factory with already taken int "
          "key: "
       << int_key;
    throw std::invalid_argument(ss.str().c_str());
  }

  int assigned_int_key = int_key == -1 ? str_map.size() + 1 : int_key;

  str_map[assigned_key] = factory;
  int_map[assigned_int_key] = assigned_key;

  return assigned_int_key;
}

void HexagonContext::add_default_object() {
  // Reuse the CPU FullyConnectedLayer as-is - the difference between
  // "engine=cpu" and "engine=cdsp" is only which ComputeOps the layer's
  // weight Tensor is handed (see initialize() below), not the layer class.
  registerFactory(nntrainer::createLayer<FullyConnectedLayer>,
                  FullyConnectedLayer::type, ml::train::LayerType::LAYER_FC);
  // QKVLayer/GateUpLayer batch several Q4_0 weights sharing one activation
  // through Tensor::dot(vector<Tensor*>, ...), which dispatches to
  // gemm_q4_0_batch_fp32 - collapsing what would be 3 (or 2) separate cDSP
  // round trips into 1. Engine::createLayerObject resolves engine=cdsp by
  // looking up the layer's type string in *this* context's own factory map
  // (not AppContext's), so these need their own registration here even
  // though they are already registered under "cpu" in app_context.cpp -
  // without this, tagging a qkv_layer/gate_up_layer with engine=cdsp would
  // throw "Key is not found for the object".
  registerFactory(nntrainer::createLayer<QKVLayer>, QKVLayer::type);
  registerFactory(nntrainer::createLayer<GateUpLayer>, GateUpLayer::type);
  // AdditionLayer's forwarding()/incremental_forwarding() dispatches
  // residual-add to the cDSP bridge itself (see addition_layer.cpp) when its
  // RunLayerContext reports engine=cdsp - it needs to be resolvable under
  // this context (same "Key is not found" reasoning as QKV/GateUp above) for
  // that tag to ever reach it via withHexagonEngine().
  registerFactory(nntrainer::createLayer<AdditionLayer>, AdditionLayer::type,
                  ml::train::LayerType::LAYER_ADDITION);
  // MultioutRealizer (compiler/multiout_realizer.cpp) auto-generates a
  // "multiout" node for any connection consumed by more than one downstream
  // layer (e.g. the residual stream after an "addition" tagged engine=cdsp)
  // and now propagates engine=cdsp onto it when the source node is also
  // cdsp - needs to be resolvable under this context for that tag, same
  // "Key is not found" reasoning as above. MultiOutLayer::forwarding() is a
  // true no-op when running in-place (the normal case, forced by
  // NetworkGraph's InPlaceType::RESTRICTING for this layer type), so this is
  // purely closing a guard-overhead gap, not adding new dispatch behavior.
  registerFactory(nntrainer::createLayer<MultiOutLayer>, MultiOutLayer::type);
  // KV-cache placeholder tensors (createKVCachePlaceholders() in
  // transformer.cpp) are plain "input" layers tagged engine=cdsp so the
  // LayerNode flush guard (layer_node.cpp) doesn't force a flush before
  // them - InputLayer::forwarding() is a no-op (the tensor is bound
  // externally by KVCacheManager), so this is the same guard-overhead-only
  // registration as MultiOutLayer above, not new dispatch behavior. Needs
  // to be resolvable under this context for the same "Key is not found"
  // reasoning as every other type registered above.
  registerFactory(nntrainer::createLayer<InputLayer>, InputLayer::type);
}

void HexagonContext::initialize() noexcept {
  try {
    add_default_object();

    // Every Q4_0 weight Tensor attached to this context's ContextData will
    // dispatch its GEMM through HexagonComputeOps::gemm_q4_0_accel_fp32
    // (float_tensor.cpp's dotQnK already checks supports_gemm_q4_0_accel_fp32()
    // before falling back to the CPU NEON/AVX kernel - see compute_ops.h).
    getContextData()->setComputeOps(get_hexagon_ops());

    // Route this context's activation tensor pool through rpcmem
    // (NeuralNetwork::load's has_cdsp_engine check -> setComputeBackend("",
    // "cdsp") is what actually activates this - see neuralnet.cpp). Lets the
    // bridge hand the DSP a pointer it can map directly instead of
    // memcpy-ing the activation into a separate staging buffer on every
    // accelerated GEMM call.
    setMemAllocator(std::make_shared<HexagonRpcAllocator>());
  } catch (std::exception &e) {
    ml_loge("hexagon_context: registering layers failed!!, reason: %s",
            e.what());
  } catch (...) {
    ml_loge("hexagon_context: registering layer failed due to unknown reason");
  }
}

template const int HexagonContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

} // namespace nntrainer

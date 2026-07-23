// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   hexagon_context.h
 * @date   23 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @brief  Context for the Hexagon cDSP compute engine ("engine=cdsp").
 *
 * Deliberately NOT built on nntrainer/qnn/ - that code loads a whole
 * pre-compiled QNN graph binary as one opaque "qnn_graph" layer, handing the
 * model architecture to QNN's graph compiler. HexagonContext instead
 * registers the ordinary FullyConnectedLayer (same C++ class the CPU engine
 * uses, unmodified) and only swaps out the Q4_0 GEMM's ComputeOps, exactly
 * like ClContext does for GPU. nntrainer stays in charge of the graph, KV
 * cache, and sampling; only the matmul kernel moves to the cDSP.
 */

#ifndef __HEXAGON_CONTEXT_H__
#define __HEXAGON_CONTEXT_H__

#include <algorithm>
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
#include <layer.h>
#include <layer_devel.h>
#include <mem_allocator.h>

#include "singleton.h"

namespace nntrainer {

extern std::mutex hexagon_factory_mutex;

/**
 * @class HexagonContext
 * @brief Context that routes Q4_0 FC layers to the Hexagon cDSP.
 */
class HexagonContext : public Context, public Singleton<HexagonContext> {
public:
  /**
   * @brief   Default constructor
   */
  HexagonContext() : Context(std::make_shared<ContextData>()) {}

  ~HexagonContext() override = default;

  template <typename T>
  const int registerFactory(const PtrFactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1) {
    FactoryType<T> f = factory;
    return registerFactory(f, key, int_key);
  }

  template <typename T>
  const int registerFactory(const FactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1);

  template <typename T>
  PtrType<T> createObject(const int int_key,
                          const PropsType &props = {}) const {
    static_assert(isSupported<T>::value,
                  "given type is not supported for hexagon context");
    auto &index = std::get<IndexType<T>>(factory_map);
    auto &int_map = std::get<IntIndexType>(index);

    const auto &entry = int_map.find(int_key);

    if (entry == int_map.end()) {
      std::stringstream ss;
      ss << "Int Key is not found for the object. Key: " << int_key;
      throw std::invalid_argument(ss.str().c_str());
    }

    return createObject<T>(entry->second, props);
  }

  template <typename T>
  PtrType<T> createObject(const std::string &key,
                          const PropsType &props = {}) const {
    auto &index = std::get<IndexType<T>>(factory_map);
    auto &str_map = std::get<StrIndexType<T>>(index);

    std::string lower_key;
    lower_key.resize(key.size());

    std::transform(key.begin(), key.end(), lower_key.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    const auto &entry = str_map.find(lower_key);

    if (entry == str_map.end()) {
      std::stringstream ss;
      ss << "Key is not found for the object. Key: " << lower_key;
      throw std::invalid_argument(ss.str().c_str());
    }

    return entry->second(props);
  }

  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(type, properties);
  }

  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &properties = {}) override {
    return createObject<nntrainer::Layer>(int_key, properties);
  }

  std::string getName() override { return "cdsp"; }

  void setMemAllocator(std::shared_ptr<MemAllocator> mem) {
    getContextData()->setMemAllocator(mem);
  }

private:
  void initialize() noexcept override;

  void add_default_object();

  FactoryMap<nntrainer::Layer> factory_map;

  template <typename Args, typename T> struct isSupportedHelper;

  template <typename T, typename... Args>
  struct isSupportedHelper<T, HexagonContext::FactoryMap<Args...>> {
    static constexpr bool value =
      (std::is_same_v<std::decay_t<T>, std::decay_t<Args>> || ...);
  };

  template <typename T>
  struct isSupported : isSupportedHelper<T, decltype(factory_map)> {};
};

} // namespace nntrainer

#endif /* __HEXAGON_CONTEXT_H__ */

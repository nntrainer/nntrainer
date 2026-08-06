// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   causal_lm.h
 * @date   22 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  CausalLM Factory to support registration and creation of various
 * CausalLM models
 */

#ifndef __CAUSALLM_FACTORY_H__
#define __CAUSALLM_FACTORY_H__

#include <ostream>
#include <transformer.h>
#include <unordered_map>

namespace causallm {

/**
 * @brief Factory class
 */
class Factory {
public:
  using Creator =
    std::function<std::unique_ptr<Transformer>(json &, json &, json &)>;

  /**
   * @brief   Get the single process-wide Factory instance.
   * @note    Declared here but DEFINED OUT-OF-LINE in factory.cpp (compiled
   *          into libcausallm.so) so there is exactly ONE Factory across all
   *          shared libraries. An inline definition (function-local static in
   *          this header) gets instantiated separately in each consumer .so
   *          under Android's per-namespace loading / -fvisibility=hidden, i.e.
   *          libcausallm, libquick_dot_ai_api, and the optional model plugin
   *          libqai_ext_model each got their OWN Factory. A model registered
   *          into one (e.g. a gauss model self-registering from the plugin's
   *          __attribute__((constructor))) was then invisible to another (the
   *          api's load_into_handle calling create()), surfacing as
   *          "Factory::create returned nullptr". A single out-of-line
   *          definition makes every Factory::Instance() caller share one map.
   *          Mirrors Engine::Global() (see nntrainer/engine.h / engine.cpp).
   */
  static Factory &Instance();

  void registerModel(const std::string &key, Creator creator) {
    creators[key] = creator;
  }

  std::unique_ptr<Transformer> create(const std::string &key, json &cfg,
                                      json &generation_cfg,
                                      json &nntr_cfg) const {
    auto it = creators.find(key);
    if (it != creators.end()) {
      return (it->second)(cfg, generation_cfg, nntr_cfg);
    }
    return nullptr;
  }

  void printRegistered(std::ostream &os) const {
    for (const auto &pair : creators) {
      os << "\n\t" << pair.first;
    }
  }

private:
  std::unordered_map<std::string, Creator> creators;
};

} // namespace causallm

#endif

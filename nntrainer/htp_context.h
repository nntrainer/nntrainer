// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 j2z0.lee <j2z0.lee@ax.samsung.com>
 *
 * @file   htp_context.h
 * @date   19 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author j2z0.lee <j2z0.lee@ax.samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP (Hexagon/HMX) compute Context — installs HtpComputeOps.
 *
 * Mirrors ClContext: a Singleton Context registered with the Engine
 * under the name "htp". A model selects it with the engine="htp"
 * property. HTP adds no custom layers, so layer creation is delegated
 * to the CPU AppContext; the only difference from "cpu" is the
 * ComputeOps installed into this context's ContextData — HtpComputeOps
 * when the NPU initialized, else the CPU ops as a transparent fallback.
 *
 * Compiled only when ENABLE_HEXKL is defined (meson: -Denable-htp=true).
 */

#ifndef __HTP_CONTEXT_H__
#define __HTP_CONTEXT_H__
#ifdef __cplusplus
#ifdef ENABLE_HEXKL

#include <memory>
#include <string>
#include <vector>

#include <context.h>
#include <singleton.h>

namespace nntrainer {

/**
 * @class HtpContext
 * @brief Engine-registered context that installs the HTP ComputeOps.
 */
class HtpContext : public Context, public Singleton<HtpContext> {
public:
  HtpContext() : Context(std::make_shared<ContextData>()) {}
  ~HtpContext() override = default;

  std::string getName() override { return "htp"; }

  ml::train::LayerComputeEngine residencyEngine() const override {
    return ml::train::LayerComputeEngine::HTP;
  }

  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &properties = {}) override;

  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &properties = {}) override;

protected:
  void initialize() noexcept override;
};

} // namespace nntrainer

#endif // ENABLE_HEXKL
#endif /* __cplusplus */
#endif /* __HTP_CONTEXT_H__ */

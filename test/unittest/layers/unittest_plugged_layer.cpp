// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   unittest_plugged_layer.cpp
 * @date   29 October 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs
 * @brief  PluggedLayer forwards the input-lifespan traits to the wrapped
 *         plugin layer instead of silently defaulting to std::nullopt.
 */

#include <memory>
#include <optional>

#include <gtest/gtest.h>

#include <layer_devel.h>
#include <plugged_layer.h>

namespace {

using nntrainer::InitLayerContext;
using nntrainer::RunLayerContext;
using nntrainer::TensorLifespan;

/**
 * @brief minimal concrete Layer that declares known input lifespans, used
 * only to prove PluggedLayer forwards to layerImpl instead of defaulting
 */
class FakePluginLayer : public nntrainer::Layer {
public:
  const std::string getType() const override { return "fake_plugin"; }
  void finalize(InitLayerContext &) override {}
  void forwarding(RunLayerContext &, bool) override {}
  void calcDerivative(RunLayerContext &) override {}
  void setProperty(const std::vector<std::string> &) override {}
  bool supportBackwarding() const override { return false; }

  std::optional<TensorLifespan> getInputLifespan() const override {
    return TensorLifespan::FORWARD_DERIV_LIFESPAN;
  }
  std::optional<TensorLifespan> getInputGradLifespan() const override {
    return TensorLifespan::CALC_GRAD_DERIV_LIFESPAN;
  }
};

nntrainer::Layer *createFakePluginLayer() { return new FakePluginLayer(); }
void destroyFakePluginLayer(nntrainer::Layer *l) { delete l; }

} // namespace

/**
 * @brief PluggedLayer forwards getInputLifespan()/getInputGradLifespan() to
 * the plugin layer it wraps, instead of defaulting to std::nullopt
 */
TEST(PluggedLayer, forwards_input_lifespan_traits_p) {
  nntrainer::LayerPluggable pluggable{createFakePluginLayer,
                                      destroyFakePluginLayer};
  nntrainer::internal::PluggedLayer plugged(&pluggable);

  EXPECT_EQ(plugged.getInputLifespan(), TensorLifespan::FORWARD_DERIV_LIFESPAN);
  EXPECT_EQ(plugged.getInputGradLifespan(),
            TensorLifespan::CALC_GRAD_DERIV_LIFESPAN);
}

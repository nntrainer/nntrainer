// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   shared_fully_connected_layer.h
 * @date   20 January 2026
 * @brief  Shared Fully Connected Layer Class
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __SHARED_FULLY_CONNECTED_LAYER_H__
#define __SHARED_FULLY_CONNECTED_LAYER_H__

#pragma once
#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_impl.h>

#include <array>

/**
 * @brief Namespace for CausalLM application components
 */
namespace causallm {

/**
 * @brief Namespace for CausalLM layer properties
 */
namespace props {

/**
 * @brief Mode property
 */
class SharedMode : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "shared_mode";
  using prop_tag = nntrainer::bool_prop_tag;
  SharedMode(bool value = false) { set(value); }
};

/**
 * @brief Full Input Range property
 */
class FullInputRange : public nntrainer::Property<bool> {
public:
  static constexpr const char *key = "full_input_range";
  using prop_tag = nntrainer::bool_prop_tag;
  FullInputRange(bool value = false) { set(value); }
};

} // namespace props

/**
 * @class   SharedFullyConnectedLayer
 * @brief   A fully connected layer that skips reading weights from file
 *          (intended for use with shared_from weights)
 */
class WIN_EXPORT SharedFullyConnectedLayer : public nntrainer::LayerImpl {
public:
  SharedFullyConnectedLayer();

  ~SharedFullyConnectedLayer() = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::read()
   */
  void read(std::ifstream &file, nntrainer::RunLayerContext &context,
            bool opt_var, ml::train::ExecutionMode mode, bool trainable,
            nntrainer::TensorDim::DataType definedWeightDataType,
            bool fsu = false, size_t start_offset = 0,
            bool read_from_offset = false, int file_fd = -1) override;

  /**
   * @copydoc Layer::save()
   */
  void save(std::ofstream &file, nntrainer::RunLayerContext &run_context,
            bool opt_var, ml::train::ExecutionMode mode, bool trainable,
            nntrainer::TensorDim::DataType definedWeightDataType =
              nntrainer::TensorDim::DataType::NONE,
            ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ExportMethods &method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return SharedFullyConnectedLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::pack(RunLayerContext &context)
   */
  void pack(nntrainer::RunLayerContext &context) override;

  static constexpr const char *type = "shared_fully_connected";

private:
  std::tuple<nntrainer::props::Unit, nntrainer::props::DisableBias,
             nntrainer::props::WeightInitializer,
             nntrainer::props::BiasInitializer, props::SharedMode,
             props::FullInputRange>
    fc_props;
  bool shared_mode_ = false;
  std::array<unsigned int, 2> weight_idx;
};

} // namespace causallm

#endif /* __SHARED_FULLY_CONNECTED_LAYER_H__ */

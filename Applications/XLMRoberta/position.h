// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   position.h
 * @date   7 January 2026
 * @brief  Implementation of position layer for XLMRoberta
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __POSITION_H__
#define __POSITION_H__

#include <complex>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

namespace xlmroberta {

/**
 * @brief A position layer for XLMRoberta.
 *
 */
class Position final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new position layer object
   *
   */
  Position() : Layer() {}

  /**
   * @brief Destroy the position layer object
   *
   */
  ~Position() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return Position::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override{};

  static constexpr const char *type = "position";
};
} // namespace xlmroberta

#endif /* __POSITION_H__ */

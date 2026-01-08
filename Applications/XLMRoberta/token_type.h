// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   token_type.h
 * @date   7 January 2026
 * @brief  Implementation of token type layer for XLMRoberta
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __TOKEN_TYPE_H__
#define __TOKEN_TYPE_H__

#include <complex>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>
#include <utility>

namespace xlmroberta {

/**
 * @brief A token type layer for XLMRoberta.
 *
 */
class TokenType final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new token type layer object
   *
   */
  TokenType() : Layer() {}

  /**
   * @brief Destroy the token type layer object
   *
   */
  ~TokenType() {}

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
  const std::string getType() const override { return TokenType::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override{};

  static constexpr const char *type = "token_type";
};
} // namespace xlmroberta

#endif /* __TOKEN_TYPE_H__ */

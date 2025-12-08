// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   split_layer.h
 * @date   09 Dec 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is split layer class (operation layer)
 */

#ifndef __CAUSALLM_SPLIT_LAYER_H__
#define __CAUSALLM_SPLIT_LAYER_H__
#ifdef __cplusplus

#include <causallm_common_properties.h>
#include <common_properties.h>
#include <layer_devel.h>
#include <layer_impl.h>

namespace causallm {

/**
 * @class Split Layer
 * @brief Split Layer
 */
class SplitLayer : public nntrainer::Layer {
public:
  /**
   * @brief Constructor of Split Layer
   */
  SplitLayer() :
    nntrainer::Layer(),
    split_props(nntrainer::props::Print(), causallm::props::StartIndex(), causallm::props::EndIndex(),
                causallm::props::Axis()),
    axis(0), start(0) {}

  /**
   * @brief Destructor of Split Layer
   */
  ~SplitLayer() = default;

  /**
   *  @brief  Move constructor of Split Layer.
   *  @param[in] SplitLayer &&
   */
  SplitLayer(SplitLayer &&rhs) noexcept = default;

  /**
   * @brief Move assignment operator.
   * @parma[in] rhs SplitLayer to be moved.
   */
  SplitLayer &operator=(SplitLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) final;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) final;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) final;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) final;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const final { return SplitLayer::type; }

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const final { return true; }

  /**
   * @brief     Incremental forward propagation of a layer
   * @param[in] context Context of the layer
   * @param[in] from Start step
   * @param[in] to End step
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  static constexpr const char *type = "causallm_split";

private:
  std::tuple<nntrainer::props::Print, causallm::props::StartIndex, causallm::props::EndIndex, causallm::props::Axis>
    split_props;
  unsigned int axis;
  unsigned int start;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __CAUSALLM_SPLIT_LAYER_H__ */

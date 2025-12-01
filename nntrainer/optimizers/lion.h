// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jeonghun Park <top231902@naver.com>
 *
 * @file   lion.h
 * @date   1 December 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jeonghun Park <top231902@naver.com>
 * @author Minseo Kim <ms05251@naver.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Lion Optimizer Header.
 */

#ifndef __LION_H__
#define __LION_H__
#ifdef __cplusplus

#include <adam.h>
#include <base_properties.h>
#include <optimizer_devel.h>
#include <tuple>

namespace nntrainer {

/**
 * @brief weight decay property
 *
 */
class PropsWeightDecayLion : public Property<double> {
public:
  static constexpr const char *key =
    "weight_decay";                 /**< unique key to access */
  using prop_tag = double_prop_tag; /**< property type */
};

/**
 * @class   Lion Optimizer class
 * @brief   Lion Optimizer (E. Chen et al., 2023)
 */
class Lion : public Optimizer {
public:
  /**
   * @brief Construct a new Lion object
   */
  Lion();

  /**
   * @brief Destroy the Lion object
   */
  ~Lion();

  /**
   * @copydoc Optimizer::getDefaultLearningRate()
   */
  double getDefaultLearningRate() const override { return 1e-4; }

  /**
   * @copydoc Optimizer::applyGradient(RunOptimizerContext &context)
   */
  void applyGradient(RunOptimizerContext &context) override;

  /**
   * @copydoc Optimizer::getType()
   */
  const std::string getType() const override { return Lion::type; }

  /**
   * @copydoc Optimizer::getOptimizerVariableDim(const TensorDim &dim)
   */
  std::vector<TensorDim> getOptimizerVariableDim(const TensorDim &dim) override;

  /**
   * @copydoc Optimizer::exportTo(Exporter &exporter,
   * const ml::train::ExportMethods &method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Optimizer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "lion";

private:
  std::tuple<PropsB1, PropsB2, PropsWeightDecayLion> lion_props;
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LION_H__ */

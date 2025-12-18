// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   adam.h
 * @date   6 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Adam optimizer.
 */
#ifndef __ADAM_H__
#define __ADAM_H__
#ifdef __cplusplus

#include <tuple>

#include <base_properties.h>
#include <optimizer_devel.h>

namespace nntrainer {

/**
 * @brief Beta 1 props
 *
 */
class PropsB1 : public Property<double> {
public:
  static constexpr const char *key = "beta1"; /**< unique key to access */
  using prop_tag = double_prop_tag;           /**< property type */
};

/**
 * @brief Beta 2 props
 *
 */
class PropsB2 : public Property<double> {
public:
  static constexpr const char *key = "beta2"; /**< unique key to access */
  using prop_tag = double_prop_tag;           /**< property type */
};

/**
 * @brief epsilon props
 * @todo move this to common props
 *
 */
class PropsEpsilon : public Property<double> {
public:
  static constexpr const char *key = "epsilon"; /**< unique key to access */
  using prop_tag = double_prop_tag;             /**< property type */
};

/**
 * @brief pytorch reference implementation
 *
 */
class TorchRef : public Property<bool> {
public:
  static constexpr const char *key = "torch_ref"; /**< unique key to access */
  using prop_tag = bool_prop_tag;                 /**< property type */
};

/**
 * @class   Adam optimizer class
 * @brief   Adam optimizer
 */
class Adam : public Optimizer {
public:
  /**
   * @brief Construct a new Adam object
   *
   */
  Adam();

  /**
   * @brief Destroy the Adam object
   *
   */
  ~Adam();

  /**
   * @copydoc Optimizer::getDefaultLearningRate()
   *
   */
  double getDefaultLearningRate() const override { return 0.001; }

  /**
   * @copydoc applyGradient(RunOptimizerContext &context)
   */
  void applyGradient(RunOptimizerContext &context) override;

  /**
   * @copydoc Optimizer::getType()
   */
  const std::string getType() const override { return Adam::type; }

  /**
   * @copydoc Optimizer::getOptimizerVariableDim(const TensorDim &dim)
   */
  std::vector<TensorDim> getOptimizerVariableDim(const TensorDim &dim) override;

  /**
   * @copydoc Optimizer::exportTo(Exporter &exporter, const
   * ml::train::ExportMethods& method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  static constexpr const char *type = "adam";

  /**
   * @copydoc Optimizer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

private:
  std::tuple<PropsB1, PropsB2, PropsEpsilon, TorchRef> adam_props;

  /**
   * @brief Get updated learning rate
   *
   * @param lr learning rate
   *
   * @return updated learning rate
   */
  double getUpdatedLearningRate(unsigned int iteration, double lr) const;
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __ADAM_H__ */

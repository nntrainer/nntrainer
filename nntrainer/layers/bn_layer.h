/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file	bn_layer.h
 * @date	14 May 2020
 * @brief	This is Batch Normalization Layer Class of Neural Network
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BN_LAYER_H__
#define __BN_LAYER_H__
#ifdef __cplusplus

#include <array>
#include <functional>
#include <vector>

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   BatchNormalizationLayer
 * @brief   Batch Noramlization Layer
 */
class BatchNormalizationLayer : public Layer {
public:
  /**
   * @brief     Constructor of Batch Normalization Layer
   */
  BatchNormalizationLayer();

  /**
   * @brief     Destructor of BatchNormalizationLayer
   */
  ~BatchNormalizationLayer() {}

  /**
   *  @brief  Move constructor of Pooling 2D Layer.
   *  @param[in] BatchNormalization &&
   */
  BatchNormalizationLayer(BatchNormalizationLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs BatchNormalizationLayer to be moved.
   */
  BatchNormalizationLayer &operator=(BatchNormalizationLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return BatchNormalizationLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  InPlaceType initializeInPlace() final {
    is_inplace = true;
    return InPlaceType::NON_RESTRICTING;
  }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override;

  static constexpr const char *type = "batch_normalization";

  /**
   * @copydoc Layer::save(std::ofstream &file,
   *      RunLayerContext &run_context,
   *      bool opt_var,
   *      ml::train::ExecutionMode mode,
   *      bool trainable,
   *      TensorDim::DataType definedWeightDataType,
   *      ml::train::ISA target_isa)
   */
  void save(std::ofstream &file, RunLayerContext &run_context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType definedWeightDataType,
            ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const override;

  /**
   * @copydoc Layer::read(std::ifstream &file, RunLayerContext &context, bool
   * opt_var, ml::train::ExecutionMode mode, bool trainable, TensorDim::DataType
   * definedWeightDataType, bool fsu)
   */
  void read(std::ifstream &file, RunLayerContext &context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType definedWeightDataType, bool fsu = false,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1, const std::string &lora_path = "") override;

  /**
   * @copydoc Layer::read(ReadSource src, RunLayerContext &context, bool
   * opt_var, ml::train::ExecutionMode mode, bool trainable, TensorDim::DataType
   * definedWeightDataType, bool fsu)
   */
  void read(ReadSource src, RunLayerContext &context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType definedWeightDataType, bool fsu = false,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1,
            const std::string &lora_path = "") override;

private:
  float divider; /**< size of the axes of the reduced */

  std::vector<unsigned int> axes_to_reduce; /**< target axes to reduce */
  std::array<unsigned int, 11>
    wt_idx; /**< indices of the weights and tensors */
  std::tuple<props::Epsilon, props::MuInitializer, props::VarInitializer,
             props::BetaInitializer, props::GammaInitializer, props::Momentum,
             props::Axis, props::WeightDecay, props::BiasDecay>
    bn_props;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __BN_LAYER_H__ */

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv2d_layer.h
 * @date   01 June 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Convolution Layer Class for Neural Network
 *
 */

#ifndef __CONV2D_LAYER_H_
#define __CONV2D_LAYER_H_
#ifdef __cplusplus

#include <memory.h>

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

constexpr const unsigned int CONV2D_DIM = 2;

/**
 * @class   Convolution 2D Layer
 * @brief   Convolution 2D Layer
 */
class Conv2DLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Conv 2D Layer
   */
  Conv2DLayer(const std::array<unsigned int, CONV2D_DIM * 2> &padding_ = {
                0, 0, 0, 0});

  /**
   * @brief     Destructor of Conv 2D Layer
   */
  ~Conv2DLayer() = default;

  /**
   *  @brief  Move constructor of Conv 2D Layer.
   *  @param[in] Conv2dLayer &&
   */
  Conv2DLayer(Conv2DLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Conv2DLayer to be moved.
   */
  Conv2DLayer &operator=(Conv2DLayer &&rhs) = default;

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
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::save(std::ofstream &file, RunLayerContext &run_context,
   * bool opt_var, ml::train::ExecutionMode mode, bool trainable,
   * TensorDim::DataType dtype, ml::train::ISA target_isa)
   *
   * @note Overridden so a conv filter can be quantized to Q4_0 as a matmul
   * weight. The FP32 filter is stored [out_ch, in_ch, kh, kw], which is already
   * row-major [out_ch, CRS] (CRS = in_ch*kh*kw) = [N, K] with N=out_ch rows of
   * K=CRS — exactly the layout quantize_q4_0 wants, so (unlike the FC path in
   * the base class) it needs no transpose. Ineligible filters (out_ch or CRS
   * not 32-aligned) and the bias stay FP32.
   */
  void save(
    std::ofstream &file, RunLayerContext &run_context, bool opt_var,
    ml::train::ExecutionMode mode, bool trainable,
    ml::train::TensorDim::DataType dtype = ml::train::TensorDim::DataType::NONE,
    ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return Conv2DLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /* TO DO : support keras type of padding */
  /* enum class PaddingType { */
  /*   full = 0, */
  /*   same = 1, */
  /*   valid = 2, */
  /*   unknown = 3, */
  /* }; */

  static constexpr const char *type = "conv2d";

private:
  std::array<unsigned int, CONV2D_DIM * 2> padding;
  std::tuple<props::FilterSize, std::array<props::KernelSize, CONV2D_DIM>,
             std::array<props::Stride, CONV2D_DIM>, props::Padding2D,
             std::array<props::Dilation, CONV2D_DIM>, props::ConvGroups,
             props::FusedActivation>
    conv_props;

#ifdef ENABLE_FP16
  std::vector<_FP16> repacked_conv0_weights_fp16;
#endif
  std::array<unsigned int, 5> wt_idx; /**< indices of the weights and tensors */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONV2D_LAYER_H__ */

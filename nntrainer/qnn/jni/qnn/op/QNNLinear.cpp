#include "QNNLinear.h"
#include "QnnTypes.h"
#include <cstdint>
#include <memory>

#include <common_properties.h>
#include <layer_context.h>
// #include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { weight, bias };
enum LORAParams { loraA, loraB, loraTmp, loraOut };

QNNLinear::QNNLinear() {
  // weight_.setBackend(bn);
  // bias_.setBackend(bn);

  // weightScale_.setBackend(bn);
  // biasScale_.setBackend(bn);
  // outputScale_.setBackend(bn);
  // inputScale_.setBackend(bn);
}

void QNNLinear::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  const auto &unit = std::get<props::Unit>(fc_props).get();
  const auto &lora_rank = (std::get<props::LoraRank>(fc_props).empty())
                            ? 0
                            : std::get<props::LoraRank>(fc_props).get();
  lora_scaling = (lora_rank && !std::get<props::LoraAlpha>(fc_props).empty())
                   ? (float)std::get<props::LoraAlpha>(fc_props) / lora_rank
                   : 1;

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  std::vector<TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  // @todo : This NCHW format setting is just temporal, it needs to be set by
  // global configuration

  /** Bias Dimension : (1, 1, 1, unit) */
  TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  /** Weight Dimension : (1, 1, in_dim.width(), unit)*/
  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[FCParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FCParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", true);
  }

  /** create weights for LoRA */
  if (lora_rank) {

    /** loraA Dimension : (1, 1, in_dim.width, lora_rank) */
    TensorDim loraA_dim(
      1, is_nchw ? 1 : lora_rank, is_nchw ? in_dim.width() : 1,
      is_nchw ? lora_rank : in_dim.channel(),
      TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
      is_nchw ? 0b0011 : 0b0101);

    /** loraB Dimension : (1, 1, lora_rank, unit) */
    TensorDim loraB_dim(
      1, is_nchw ? 1 : unit, is_nchw ? lora_rank : 1,
      is_nchw ? unit : lora_rank,
      TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
      is_nchw ? 0b0011 : 0b0101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), lora_rank) */
    TensorDim loraTmp_dim(
      in_dim.batch(), is_nchw ? 1 : lora_rank, is_nchw ? in_dim.height() : 1,
      is_nchw ? lora_rank : in_dim.width(),
      TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
      is_nchw ? 0b1011 : 0b1101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), unit) */
    TensorDim loraOut_dim(
      in_dim.batch(), is_nchw ? 1 : unit, is_nchw ? in_dim.height() : 1,
      is_nchw ? unit : in_dim.width(),
      TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
      is_nchw ? 0b1011 : 0b1101);

    lora_idx[LORAParams::loraA] = context.requestWeight(
      loraA_dim, Initializer::ZEROS, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraA", true);

    lora_idx[LORAParams::loraB] = context.requestWeight(
      loraB_dim, Initializer::LECUN_NORMAL, weight_regularizer,
      weight_regularizer_constant, weight_decay, "loraB", true);

    lora_idx[LORAParams::loraTmp] =
      context.requestTensor(loraTmp_dim, "hidden_tmp_lora", Initializer::NONE,
                            true, TensorLifespan::FORWARD_GRAD_LIFESPAN);

    lora_idx[LORAParams::loraOut] =
      context.requestTensor(loraOut_dim, "hidden_lora", Initializer::NONE, true,
                            TensorLifespan::FORWARD_FUNC_LIFESPAN);
  }
}

void QNNLinear::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  LayerImpl::setProperty(remain_props);
}

void QNNLinear::exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

} // namespace nntrainer

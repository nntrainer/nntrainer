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
 *
 * @file	bn_layer.cpp
 * @date	14 May 2020
 * @brief	This is Batch Normalization Layer Class for Neural Network
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <bn_layer.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum BNParams {
  mu,
  var,
  gamma,
  beta,
  mu_b,
  var_b,
  deviation,
  invstd,
  cvar,
  t_reduced,
  t_full
};

BatchNormalizationLayer::BatchNormalizationLayer() :
  Layer(),
  divider(0),
  bn_props(props::Epsilon(), props::MuInitializer(), props::VarInitializer(),
           props::BetaInitializer(), props::GammaInitializer(),
           props::Momentum(), props::Axis(), props::WeightDecay(),
           props::BiasDecay()) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

/// @todo add multiple axis support
void BatchNormalizationLayer::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Only one input is allowed for batch normalization layer";

  auto &bnparams_mu = std::get<props::MuInitializer>(bn_props);
  auto &bnparams_var = std::get<props::VarInitializer>(bn_props);
  auto &bnparams_beta = std::get<props::BetaInitializer>(bn_props);
  auto &bnparams_gamma = std::get<props::GammaInitializer>(bn_props);
  auto &weight_decay = std::get<props::WeightDecay>(bn_props);
  auto &bias_decay = std::get<props::BiasDecay>(bn_props);

  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  context.setOutputDimensions(context.getInputDimensions());

  TensorDim dim(context.getFormat(), context.getWeightDataType());

  if (context.getExecutionMode() == ml::train::ExecutionMode::TRAIN) {
    dim.setDataType(TensorDim::DataType::FP32);
  }

  /// @note this logic cannot tell channel is actually 1 or it is just not used.
  auto &axis_prop = std::get<props::Axis>(bn_props);
  unsigned int axis;
  if (axis_prop.empty())
    axis = in_dim.channel() > 1 ? 1 : 3;
  else
    axis = axis_prop.get();

  /**
   * @todo This can be speedup by employing transpose for convolution. With
   * transpose, the channel dimension can be made last, and the remaining
   * dimensions can be squeezed. This would allow the sum and average to be
   * faster, and no temporary allocations inside them.
   */

  dim.setTensorDim(axis, in_dim.getTensorDim(axis));

  divider = 1;
  for (unsigned int i = 0; i < 4; ++i) {
    if (axis != i) {
      axes_to_reduce.push_back(i);
      divider *= in_dim.getTensorDim(i);
    }
  }

  wt_idx[BNParams::mu] =
    context.requestWeight(dim, dim, bnparams_mu, WeightRegularizer::NONE, 1.0f,
                          0.0f, "moving_mean", false);
  wt_idx[BNParams::var] =
    context.requestWeight(dim, dim, bnparams_var, WeightRegularizer::NONE, 1.0f,
                          0.0f, "moving_variance", false);
  wt_idx[BNParams::gamma] =
    context.requestWeight(dim, dim, bnparams_gamma, WeightRegularizer::NONE,
                          1.0f, weight_decay, "gamma", true);
  wt_idx[BNParams::beta] =
    context.requestWeight(dim, dim, bnparams_beta, WeightRegularizer::NONE,
                          1.0f, bias_decay, "beta", true);

  wt_idx[BNParams::mu_b] =
    context.requestTensor(dim, "moviing_mean_backup", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);

  wt_idx[BNParams::var_b] =
    context.requestTensor(dim, "moviing_variance_backup", Initializer::NONE,
                          false, TensorLifespan::ITERATION_LIFESPAN);

  /**
   * caches the deviation -> input - avg(input)
   * @todo check if avoiding this storage and adding dependency on input (no
   * more in-place calculation) can save memory during memory optimization.
   */
  TensorDim in_dim_ = in_dim;

  if (context.getExecutionMode() == ml::train::ExecutionMode::TRAIN) {
    in_dim_.setDataType(TensorDim::DataType::FP32);
  }

  wt_idx[BNParams::deviation] =
    context.requestTensor(in_dim_, "deviation", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  /** caches the inverse standard deviation */
  wt_idx[BNParams::invstd] =
    context.requestTensor(dim, "invstd", Initializer::NONE, false,
                          TensorLifespan::ITERATION_LIFESPAN);
  /**
   * Temporary tensor to store the full sized tensors in order to allow batch
   * norm to execute in-place. Running in-place leads to same memory footprint
   * for this layer in its backwarding, but reduces the peak memory requirement
   * as the output of this layer need not be stored all the time.
   */
  wt_idx[BNParams::t_full] =
    context.requestTensor(in_dim_, "tensor_full", Initializer::NONE, false,
                          TensorLifespan::CALC_DERIV_LIFESPAN);
  /**
   * caches variance + epsilon as well.
   */
  wt_idx[BNParams::cvar] = context.requestTensor(
    dim, "cvar", Initializer::NONE, false, TensorLifespan::ITERATION_LIFESPAN);
  /**
   * Temporary tensor to store the reduced tensors along the axes_to_reduce.
   */
  wt_idx[BNParams::t_reduced] =
    context.requestTensor(dim, "tensor_reduced", Initializer::NONE, false,
                          TensorLifespan::FORWARD_DERIV_LIFESPAN);
}

void BatchNormalizationLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, bn_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[BNLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void BatchNormalizationLayer::forwarding(RunLayerContext &context,
                                         bool training) {
  float epsilon = std::get<props::Epsilon>(bn_props);
  float momentum = std::get<props::Momentum>(bn_props);

  Tensor &mu = context.getWeight(wt_idx[BNParams::mu]);
  Tensor &var = context.getWeight(wt_idx[BNParams::var]);
  Tensor &gamma = context.getWeight(wt_idx[BNParams::gamma]);
  Tensor &beta = context.getWeight(wt_idx[BNParams::beta]);

  Tensor em_input, em_hidden;

  Tensor &input_ = em_input;
  Tensor &hidden_ = em_hidden;

  if (training) {
    if (context.getInput(SINGLE_INOUT_IDX).getDataType() !=
        TensorDim::DataType::FP32) {
      input_ =
        context.getInput(SINGLE_INOUT_IDX).clone(TensorDim::DataType::FP32);
    } else {
      input_ = context.getInput(SINGLE_INOUT_IDX);
    }

    if (context.getOutput(SINGLE_INOUT_IDX).getDataType() !=
        TensorDim::DataType::FP32) {
      hidden_ =
        context.getOutput(SINGLE_INOUT_IDX).clone(TensorDim::DataType::FP32);
    } else {
      hidden_ = context.getOutput(SINGLE_INOUT_IDX);
    }
  } else {
    input_ = context.getInput(SINGLE_INOUT_IDX);
    hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  }

  Tensor &deviation = context.getTensor(wt_idx[BNParams::deviation]);
  Tensor &invstd = context.getTensor(wt_idx[BNParams::invstd]);

  /** @todo these are not needed for inference, support optimizing these */
  Tensor &t_reduced = context.getTensor(wt_idx[BNParams::t_reduced]);
  /** use hidden_ as temporary tensor before setting the result in hidden */
  Tensor t_full = hidden_;
  Tensor &cvar = context.getTensor(wt_idx[BNParams::cvar]);

  if (training) {

    Tensor &mu_b = context.getTensor(wt_idx[BNParams::mu_b]);
    Tensor &var_b = context.getTensor(wt_idx[BNParams::var_b]);

    if (context.reStoreData()) {
      mu.copyData(mu_b);
      var.copyData(var_b);
      deviation.setZero();
      invstd.setZero();
      t_reduced.setZero();
      cvar.setZero();
    } else {
      mu_b.copyData(mu);
      var_b.copyData(var);
    }

    input_.average(axes_to_reduce, t_reduced);
    input_.subtract(t_reduced, deviation);

    mu.multiply_i(momentum);
    mu.add_i(t_reduced, 1 - momentum);

    deviation.pow(2.0f, t_full);
    t_full.average(axes_to_reduce, cvar);

    var.multiply_i(momentum);
    var.add_i(cvar, 1 - momentum);

    cvar.add_i(epsilon);
    cvar.pow(-0.5f, invstd);
  } else {
    input_.subtract(mu, deviation);
    /** @todo do below 2 lines only for first iteration */
    var.add(epsilon, invstd);
    invstd.pow_i(-0.5f);
  }

  deviation.multiply(invstd, hidden_);
  hidden_.multiply_i(gamma);
  hidden_.add_i(beta);

  if (training && hidden_.getDataType() !=
                    context.getOutput(SINGLE_INOUT_IDX).getDataType())
    context.getOutput(SINGLE_INOUT_IDX).copyData(hidden_);
}

void BatchNormalizationLayer::calcDerivative(RunLayerContext &context) {

  Tensor &gamma = context.getWeight(wt_idx[BNParams::gamma]);

  Tensor em_dx, deriv32;
  bool deriv_copyed = false;

  const Tensor deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  if (deriv.getDataType() != TensorDim::DataType::FP32) {
    deriv_copyed = true;
    TensorDim dim = deriv.getDim();
    dim.setDataType(TensorDim::DataType::FP32);
    deriv32 = Tensor(dim, true);
    deriv32.copyData(deriv);
  }

  Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX).getDataType() ==
                   TensorDim::DataType::FP32
                 ? context.getOutgoingDerivative(SINGLE_INOUT_IDX)
                 : em_dx;

  if (dx.empty())
    dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX)
           .clone(TensorDim::DataType::FP32);

  Tensor &deviation = context.getTensor(wt_idx[BNParams::deviation]);
  Tensor &invstd = context.getTensor(wt_idx[BNParams::invstd]);
  Tensor &cvar = context.getTensor(wt_idx[BNParams::cvar]);

  Tensor &t_reduced = context.getTensor(wt_idx[BNParams::t_reduced]);
  Tensor &t_full = context.getTensor(wt_idx[BNParams::t_full]);

  t_full.setZero();

  deviation.multiply((deriv_copyed ? deriv32 : deriv), t_full);
  t_full.average(axes_to_reduce, t_reduced);
  t_reduced.divide_i(cvar);
  deviation.multiply_i(t_reduced);

  if (context.getTrainable()) {
    /**
     * This calculates dgamma tensor.
     */
    Tensor &dgamma = context.getWeightGrad(wt_idx[BNParams::gamma]);
    t_full.multiply_i(invstd);
    t_full.sum(axes_to_reduce, dgamma);

    /**
     * This implementation depends on the pre-calculated dbeta calculated.
     */
    Tensor &dbeta = context.getWeightGrad(wt_idx[BNParams::beta]);
    dbeta.divide(divider, t_reduced);
  } else {
    (deriv_copyed ? deriv32 : deriv).average(axes_to_reduce, t_reduced);
  }

  (deriv_copyed ? deriv32 : deriv).subtract(t_reduced, dx);
  dx.subtract_i(deviation);

  invstd.multiply_i(gamma);
  dx.multiply_i(invstd);

  if (dx.getDataType() !=
      context.getOutgoingDerivative(SINGLE_INOUT_IDX).getDataType())
    context.getOutgoingDerivative(SINGLE_INOUT_IDX).copyData(dx);
}

void BatchNormalizationLayer::calcGradient(RunLayerContext &context) {
  /** dgamma is calculated in calcDerivative. dbeta is calculated here */
  Tensor &dbeta = context.getWeightGrad(wt_idx[BNParams::beta]);
  dbeta.setZero();

  Tensor deriv32;
  bool deriv_copyed = false;

  const Tensor deriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  if (deriv.getDataType() != TensorDim::DataType::FP32) {
    deriv_copyed = true;
    TensorDim dim = deriv.getDim();
    dim.setDataType(TensorDim::DataType::FP32);
    deriv32 = Tensor(dim, true);
    deriv32.copyData(deriv);
  }

  (deriv_copyed ? deriv32 : deriv).sum(axes_to_reduce, dbeta);
}

void BatchNormalizationLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(bn_props, method, this);
}

void BatchNormalizationLayer::setBatch(RunLayerContext &context,
                                       unsigned int batch) {
  context.updateTensor(wt_idx[BNParams::deviation], batch);
  context.updateTensor(wt_idx[BNParams::t_full], batch);

  /// reset divider
  divider = 1;
  auto input_dim = context.getInput(0).getDim();
  for (auto axis : axes_to_reduce) {
    if (axis == 0) {
      /// @note input dim batch is not updated, it will be more sensible we
      /// update batch before any node comes to this spot
      divider *= batch;
    }
    divider *= input_dim.getTensorDim(axis);
  }
}

void BatchNormalizationLayer::save(std::ofstream &file,
                                   RunLayerContext &run_context, bool opt_var,
                                   ml::train::ExecutionMode mode,
                                   bool trainable,
                                   TensorDim::DataType definedWeightDataType,
                                   ml::train::ISA target_isa) const {
  if (opt_var) {
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      if (run_context.isGradientFirstAccess(i) && trainable) {
        // @note save optimizer variables
        if (run_context.weightHasGradient(i)) {
          for (unsigned int j = 0; j < run_context.getNumWeightOptVar(i); ++j) {
            run_context.getWeightOptVar(i, j).save(file);
          }
        }
      }
    }
  } else {
    // @note shared weights are only be saved at the first access
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      if (run_context.isGradientFirstAccess(i)) {

        // @note For batch normalization layer, we do need full precision for
        // training and the data type of weight is full precision. But for
        // inference, We do have to save them as activation data type.
        if ((mode == ml::train::ExecutionMode::TRAIN) &&
            (definedWeightDataType != TensorDim::DataType::FP32)) {
          TensorDim dim = run_context.getWeight(i).getDim();

          dim.setDataType(definedWeightDataType);

          Tensor T_save(dim, true);

          T_save.copyData(run_context.getWeight(i));

          T_save.save(file);
        } else {
          run_context.getWeight(i).save(file);
        }
      }
    }
  }
}

void BatchNormalizationLayer::read(std::ifstream &file,
                                   RunLayerContext &run_context, bool opt_var,
                                   ml::train::ExecutionMode mode,
                                   bool trainable,
                                   TensorDim::DataType definedWeightDataType,
                                   bool fsu, size_t start_offset,
                                   bool read_from_offset, int file_fd) {
  if (opt_var) {
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      if (run_context.isGradientLastAccess(i) && trainable) {
        /// @note read optimizer variables
        for (unsigned int j = 0; j < run_context.getNumWeightOptVar(i); ++j) {
          run_context.getWeightOptVar(i, j).read(file);
        }
      }
    }
  } else {
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      /// @note shared weights are only be read at the first acecss
      //      if (run_context->isGradientLastAccess(i)) {
      if (run_context.isGradientFirstAccess(i)) {
        if ((mode == ml::train::ExecutionMode::TRAIN) &&
            (definedWeightDataType != TensorDim::DataType::FP32)) {

          /** @note for batch normalization layer, we do need full
          precision
           * for training. but weight can be saved with other type. for
           * training, bn weight type is fixed with full precsion */

          TensorDim dim = run_context.getWeight(i).getDim();
          dim.setDataType(definedWeightDataType);
          Tensor T_read(dim, true);
          T_read.read(file);
          run_context.getWeight(i).copyData(T_read);
        } else {
          run_context.getWeight(i).read(file, start_offset, read_from_offset,
                                        file_fd);
        }

        if (run_context.isMixedPrecision(i) && trainable &&
            !run_context.getWeightFP32(i).empty()) {
          run_context.getWeightFP32(i).copyData(run_context.getWeight(i));
        }
      }
    }
  }
}

void BatchNormalizationLayer::read(ReadSource src, RunLayerContext &run_context,
                                   bool opt_var, ml::train::ExecutionMode mode,
                                   bool trainable,
                                   TensorDim::DataType definedWeightDataType,
                                   bool fsu, size_t start_offset,
                                   bool read_from_offset, int file_fd) {
  if (opt_var) {
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      if (run_context.isGradientLastAccess(i) && trainable) {
        /// @note read optimizer variables
        for (unsigned int j = 0; j < run_context.getNumWeightOptVar(i); ++j) {
          run_context.getWeightOptVar(i, j).read(src);
        }
      }
    }
  } else {
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      /// @note shared weights are only be read at the first acecss
      //      if (run_context->isGradientLastAccess(i)) {
      if (run_context.isGradientFirstAccess(i)) {
        if ((mode == ml::train::ExecutionMode::TRAIN) &&
            (definedWeightDataType != TensorDim::DataType::FP32)) {

          /** @note for batch normalization layer, we do need full
          precision
           * for training. but weight can be saved with other type. for
           * training, bn weight type is fixed with full precsion */

          TensorDim dim = run_context.getWeight(i).getDim();
          dim.setDataType(definedWeightDataType);
          Tensor T_read(dim, true);
          T_read.read(src);
          run_context.getWeight(i).copyData(T_read);
        } else {
          run_context.getWeight(i).read(src, start_offset, read_from_offset,
                                        file_fd);
        }

        if (run_context.isMixedPrecision(i) && trainable &&
            !run_context.getWeightFP32(i).empty()) {
          run_context.getWeightFP32(i).copyData(run_context.getWeight(i));
        }
      }
    }
  }
}

} /* namespace nntrainer */

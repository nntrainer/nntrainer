// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   fc_layer_cl.h
 * @date   7 May 2024
 * @brief  Backend-neutral quantized Fully Connected layer (op-table dispatch).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details A thin Layer that owns the weight/bias binding and dispatches the
 * matmul through the op table: input.getOps()->fc(...) lands on
 * ClComputeOps::fc (the OpenCL GEMM) on the gpu engine and on
 * CpuComputeOps::fc (host Tensor::dot) with no accelerator ContextData
 * attached. The eager weight transform at load is fc_prebuild_weight(), a
 * no-op on a backend that needs none. Registered for the "gpu" engine; the
 * general FullyConnectedLayer (LoRA/quantizer) stays separate for cpu.
 */

#ifndef __FC_LAYER_CL_H__
#define __FC_LAYER_CL_H__
#ifdef __cplusplus

#include <array>
#include <tuple>

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   FullyConnectedLayerCl
 * @brief   backend-neutral quantized fully connected layer (op-table dispatch)
 */
class FullyConnectedLayerCl : public LayerImpl {
public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  FullyConnectedLayerCl();

  /**
   * @brief     Destructor of Fully Connected Layer
   */
  ~FullyConnectedLayerCl() = default;

  /**
   *  @brief  Move constructor.
   */
  FullyConnectedLayerCl(FullyConnectedLayerCl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   */
  FullyConnectedLayerCl &operator=(FullyConnectedLayerCl &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::read(std::ifstream &file, RunLayerContext &run_context,
   * ...)
   * @note after the base read, eagerly builds the backend's GPU weight entry
   *       (getOps()->fc_prebuild_weight) so the first prefill does not pay the
   *       lazy per-weight transform. Skipped under FSU (the weight data may be
   *       streamed out again); a no-op on backends that need no prebuild.
   */
  void read(std::ifstream &file, RunLayerContext &run_context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType defineWeightDataType, bool fsu,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1) override;

  /**
   * @copydoc Layer::read(ReadSource src, RunLayerContext &run_context, ...)
   */
  void read(ReadSource src, RunLayerContext &run_context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType defineWeightDataType, bool fsu,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1) override;

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
    return FullyConnectedLayerCl::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "fully_connected";

private:
  bool skip_prefill =
    false; /**< skip compute during prefill (Gemma4 KV-share) */
  std::tuple<props::Unit, props::FusedActivation, props::PlanLastRowOnly>
    fc_props; /**< fc layer properties : unit - number of output neurons;
                   fused_activation - inline activation epilogue;
                   plan_last_row_only - plan the output at height 1 because
                   only the last row is ever produced */
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FC_LAYER_CL_H__ */

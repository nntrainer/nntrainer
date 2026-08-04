// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   NeuronGraph.h
 * @date   30 Jul 2026
 * @brief  Layer that wraps execution of one precompiled MediaTek NeuroPilot
 *         (.dla) network via the Neuron Runtime.
 * @see    https://github.com/nnstreamer/nntrainer
 *
 * @details Structurally mirrors nntrainer/qnn/jni/QNNGraph.h, but I/O is
 * bound by position (nntrainer input/output index i <-> Neuron frontend IO
 * handle i) rather than by name — Neuron addresses tensors purely by
 * `uint64_t` index, there is no per-tensor name in the runtime API. If a
 * DLA's compiled I/O order ever needs to differ from nntrainer's layer
 * input/output order, an explicit index-map property should be added; not
 * needed for the current single/simple-graph use case.
 */
#ifndef __NNTR_NEURONGRAPH_H__
#define __NNTR_NEURONGRAPH_H__

#include <layer_impl.h>
#include <neuron_context_var.h>
#include <neuron_properties.h>

namespace nntrainer {

/** @brief Layer that wraps one MediaTek NeuroPilot .dla network. */
class NeuronGraph : public LayerImpl {
public:
  NeuronGraph();
  ~NeuronGraph();

  inline static const std::string type = "neuron_graph";

  const std::string getType() const override { return NeuronGraph::type; };

  void finalize(InitLayerContext &context) override;

  bool supportBackwarding() const override { return false; }

  void calcDerivative(RunLayerContext &context) override{};

  void forwarding(RunLayerContext &context, bool training) override;

  void setProperty(const std::vector<std::string> &values) override;

  /** @brief nntrainer weight-bin reader hook; a no-op since the network's
   * weights live inside the .dla, loaded by NeuronRuntime_loadNetworkFromFile
   * during finalize()/forwarding(), not by nntrainer's weight loader. */
  void read(std::ifstream &file, RunLayerContext &run_context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType defineWeightDataType, bool fsu = false,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1) override;

private:
  std::tuple<std::vector<props::TensorDimension>,
             std::vector<props::TensorDataType>, std::vector<props::TensorType>,
             props::FilePath, std::vector<props::NeuronInputQuantParam>,
             std::vector<props::NeuronOutputQuantParam>>
    graph_props;

  std::string dla_path;
};

} // namespace nntrainer

#endif /* __NNTR_NEURONGRAPH_H__ */

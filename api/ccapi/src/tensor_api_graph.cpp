// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@@samsung.com>
 *
 * @file   tensor_api_graph.cpp
 * @date   11 December 2023
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Symbolic graph construction and Model::compile overloads
 *         for the ml::train::Tensor API
 *
 * @note This is experimental API and not stable.
 */

#include "tensor_api_impl.h"

#include <model.h>

#include <layer_context.h>
#include <tensor.h>

#include <functional>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace ml {
namespace train {

std::shared_ptr<Layer> Tensor::getProducingLayer() const {
  return (impl_ && impl_->graph_edge) ? impl_->graph_edge->producing_layer
                                      : nullptr;
}

Tensor Tensor::output(unsigned int index) const {
  Tensor result;
  result.impl_ = std::make_unique<Impl>();
  result.impl_->valid = true;

  // Create a new graph edge that shares the same producing layer but with index
  auto indexed_edge = std::make_shared<SymbolicGraphNode>();
  if (impl_ && impl_->graph_edge) {
    indexed_edge->producing_layer = impl_->graph_edge->producing_layer;
    indexed_edge->inputs = impl_->graph_edge->inputs;
    indexed_edge->dim = impl_->graph_edge->dim;
    indexed_edge->name = impl_->graph_edge->name;
  }
  indexed_edge->output_index = static_cast<int>(index);
  result.impl_->graph_edge = indexed_edge;

  if (impl_) {
    result.impl_->src_layer = impl_->src_layer;
    result.impl_->dim = impl_->dim;
  }

  return result;
}

std::vector<Tensor> Tensor::getInputTensors() const {
  if (!impl_ || !impl_->graph_edge) {
    return {};
  }
  std::vector<Tensor> result;
  for (auto &edge : impl_->graph_edge->inputs) {
    Tensor t;
    t.impl_->graph_edge = edge;
    if (edge) {
      t.impl_->dim = edge->dim;
      t.impl_->name = edge->name;
      t.impl_->valid = true;
    }
    result.push_back(std::move(t));
  }
  return result;
}

// --- Symbolic tensor operations (implicit layers) ---

static unsigned int implicit_layer_counter = 0;

static std::string nextImplicitName(const std::string &prefix) {
  return prefix + "_auto_" + std::to_string(implicit_layer_counter++);
}

Tensor Tensor::add(const Tensor &other) const {
  LayerHandle layer =
    createLayer("Addition", {"name=" + nextImplicitName("add")});
  return layer({*this, other});
}

Tensor Tensor::multiply(const Tensor &other) const {
  LayerHandle layer =
    createLayer("Multiply", {"name=" + nextImplicitName("mul")});
  return layer({*this, other});
}

Tensor Tensor::reshape(const TensorDim &new_shape) const {
  std::string target = std::to_string(new_shape.channel()) + ":" +
                       std::to_string(new_shape.height()) + ":" +
                       std::to_string(new_shape.width());
  LayerHandle layer = createLayer(
    "reshape", {"name=" + nextImplicitName("reshape"),
                "target_shape=" + target});
  Tensor output = layer({*this});
  // Override shape to match requested dimensions
  output.impl_->dim = TensorDim({shape().batch(), new_shape.channel(),
                                  new_shape.height(), new_shape.width()});
  return output;
}

// --- LayerHandle graph construction ---

/**
 * @brief Try to infer output dimensions from layer type and properties.
 *
 * This is a best-effort inference for common layer types.
 * Full shape inference happens during model.compile().
 */
static TensorDim inferOutputDim(const std::shared_ptr<Layer> &layer,
                                const std::vector<Tensor> &inputs) {
  std::string layer_type = layer->getType();

  // Input layer: parse shape from input_shape property
  if (layer_type == "input") {
    try {
      std::string shape_str = layer->getProperty("input_shape");
      if (!shape_str.empty()) {
        // Parse "B:C:H:W" or "C:H:W" format
        std::vector<unsigned int> dims;
        std::stringstream ss(shape_str);
        std::string token;
        while (std::getline(ss, token, ':')) {
          dims.push_back(static_cast<unsigned int>(std::stoul(token)));
        }
        if (dims.size() == 4) {
          return TensorDim({dims[0], dims[1], dims[2], dims[3]});
        } else if (dims.size() == 3) {
          return TensorDim({1, dims[0], dims[1], dims[2]});
        }
      }
    } catch (...) {
      // Fall through
    }
  }

  if (inputs.empty() || !inputs[0].isValid()) {
    return TensorDim();
  }

  const TensorDim &in_dim = inputs[0].shape();

  // Fully connected: output = {batch, 1, 1, unit}
  if (layer_type == "fully_connected") {
    try {
      std::string unit_str = layer->getProperty("unit");
      if (!unit_str.empty()) {
        unsigned int unit = static_cast<unsigned int>(std::stoul(unit_str));
        return TensorDim({in_dim.batch(), 1, 1, unit});
      }
    } catch (...) {
      // Fall through to default
    }
  }

  // Most layers preserve shape (activation, normalization, dropout, etc.)
  return in_dim;
}

Tensor LayerHandle::operator()(const Tensor &input) {
  return (*this)(std::vector<Tensor>{input});
}

Tensor LayerHandle::operator()(const std::vector<Tensor> &inputs) {
  if (!ptr_) {
    throw std::runtime_error("LayerHandle: layer is null");
  }
  if (inputs.empty()) {
    throw std::invalid_argument("LayerHandle: at least one input required");
  }

  // Infer output dimensions
  TensorDim out_dim = inferOutputDim(ptr_, inputs);

  // Build output name from layer name
  std::string out_name;
  try {
    out_name = ptr_->getName();
    if (!out_name.empty()) {
      out_name += ":output";
    }
  } catch (...) {
    // getName() might throw if layer isn't fully initialized
  }

  // Create symbolic output tensor
  Tensor output;
  if (out_dim.batch() > 0 && out_dim.width() > 0) {
    output = Tensor(out_dim, out_name);
  } else {
    // Couldn't infer shape — create a valid but shapeless tensor
    output = Tensor(TensorDim(), out_name);
  }

  // Record graph edge (shared, no deep copies)
  auto edge = std::make_shared<SymbolicGraphNode>();
  edge->producing_layer = ptr_;
  edge->dim = out_dim;
  edge->name = out_name;
  for (auto &inp : inputs) {
    if (inp.impl_ && inp.impl_->graph_edge) {
      edge->inputs.push_back(inp.impl_->graph_edge);
    } else {
      // Leaf tensor — create a leaf edge with no producer
      auto leaf = std::make_shared<SymbolicGraphNode>();
      if (inp.isValid()) {
        leaf->dim = inp.shape();
        leaf->name = inp.name();
      }
      edge->inputs.push_back(leaf);
    }
  }
  output.impl_->graph_edge = edge;

  return output;
}

// --- Model::compile(Tensor, Tensor) — graph extraction ---

int Model::compile(Tensor &input, Tensor &output, ExecutionMode mode) {
  std::vector<Tensor> inputs = {input};
  std::vector<Tensor> outputs = {output};
  int status = compile(inputs, outputs, mode);
  input = inputs[0];
  output = outputs[0];
  return status;
}

int Model::compile(Tensor &input, std::vector<Tensor> &outputs,
                   ExecutionMode mode) {
  std::vector<Tensor> inputs = {input};
  int status = compile(inputs, outputs, mode);
  input = inputs[0];
  return status;
}

int Model::compile(std::vector<Tensor> &inputs, std::vector<Tensor> &outputs,
                   ExecutionMode mode) {
  struct LayerInfo {
    std::shared_ptr<Layer> layer;
    std::vector<std::string> input_layer_names;
  };

  std::vector<LayerInfo> layers_in_order;
  std::unordered_set<Layer *> visited;

  // Collect input leaf names
  std::set<std::string> input_leaf_names;
  for (size_t i = 0; i < inputs.size(); ++i) {
    std::string name = inputs[i].name();
    if (name.empty()) {
      name = (inputs.size() == 1) ? "graph_input"
                                  : "graph_input_" + std::to_string(i);
    }
    input_leaf_names.insert(name);
  }

  // Track additional leaf tensors (e.g., fromData external caches)
  struct LeafInfo {
    TensorDim dim;
  };
  std::map<std::string, LeafInfo> additional_leaves;
  int unnamed_leaf_counter = 0;

  // DFS on SymbolicGraphNode (post-order = topological order)
  std::function<void(const std::shared_ptr<SymbolicGraphNode> &)> dfs =
    [&](const std::shared_ptr<SymbolicGraphNode> &edge) {
      if (!edge || !edge->producing_layer) {
        return; // leaf
      }
      if (visited.count(edge->producing_layer.get())) {
        return;
      }
      visited.insert(edge->producing_layer.get());

      for (auto &inp : edge->inputs) {
        dfs(inp);
      }

      // Build input_layers names
      std::vector<std::string> input_names;
      for (auto &inp : edge->inputs) {
        if (inp && inp->producing_layer) {
          std::string layer_ref = inp->producing_layer->getName();
          if (inp->output_index >= 0) {
            layer_ref += "(" + std::to_string(inp->output_index) + ")";
          }
          input_names.push_back(layer_ref);
        } else {
          // Leaf tensor
          std::string leaf_name = inp ? inp->name : "";
          if (leaf_name.empty()) {
            // Empty leaf from Tensor() — this is the sentinel input to an
            // input layer. Check if the current edge's producing layer is an
            // input layer; if so, skip this leaf entirely since the input
            // layer itself serves as the graph entry point (no input_layers).
            if (edge->producing_layer) {
              try {
                if (edge->producing_layer->getType() == "input") {
                  // Input layer's sentinel leaf — skip, no input_layers needed
                  continue;
                }
              } catch (...) {}
            }
            leaf_name = "ext_input_" + std::to_string(unnamed_leaf_counter++);
          }
          if (input_leaf_names.count(leaf_name)) {
            input_names.push_back(leaf_name);
          } else {
            if (additional_leaves.find(leaf_name) == additional_leaves.end()) {
              additional_leaves[leaf_name] = {inp ? inp->dim : TensorDim()};
            }
            input_names.push_back(leaf_name);
          }
        }
      }

      layers_in_order.push_back({edge->producing_layer, input_names});
    };

  // Walk backwards from each output
  for (auto &output : outputs) {
    if (output.impl_ && output.impl_->graph_edge) {
      dfs(output.impl_->graph_edge);
    }
  }

  // 1. Create and add input layers (skip if already in DFS-discovered graph)
  //    When constructModel() already creates input layers via
  //    LayerHandle(createLayer("input",...)), the DFS walk will have found
  //    them. In that case, skip creating duplicate input layers.
  std::set<std::string> discovered_layer_names;
  for (auto &li : layers_in_order) {
    try {
      discovered_layer_names.insert(li.layer->getName());
    } catch (...) {}
  }

  int status;
  for (auto &inp : inputs) {
    std::string inp_name = inp.name();
    if (inp_name.empty()) {
      inp_name =
        (inputs.size() == 1)
          ? "graph_input"
          : "graph_input_" + std::to_string(&inp - &inputs[0]);
    }

    // If this input's producing layer was already discovered by DFS, skip
    if (inp.impl_ && inp.impl_->graph_edge &&
        inp.impl_->graph_edge->producing_layer) {
      std::string prod_name;
      try {
        prod_name = inp.impl_->graph_edge->producing_layer->getName();
      } catch (...) {}
      if (!prod_name.empty() && discovered_layer_names.count(prod_name)) {
        continue;  // Input layer already in the graph
      }
    }

    // Also skip if a layer with the same name was already discovered
    // (handles the case where inp_name matches an output name like "input0:output")
    bool already_has_input = false;
    for (auto &li : layers_in_order) {
      try {
        if (li.layer->getType() == "input" &&
            li.layer->getName() == inp_name) {
          already_has_input = true;
          break;
        }
      } catch (...) {}
    }
    if (already_has_input) continue;

    if (!inp.isValid()) continue;  // skip invalid tensors

    const TensorDim &dim = inp.shape();
    std::string shape_str = std::to_string(dim.channel()) + ":" +
                             std::to_string(dim.height()) + ":" +
                             std::to_string(dim.width());
    auto input_layer = createLayer(
      "input", {"name=" + inp_name, "input_shape=" + shape_str});
    status = addLayer(std::move(input_layer));
    if (status != ML_ERROR_NONE) {
      return status;
    }
  }

  // 1b. Create input layers for additional leaf tensors
  for (auto &[leaf_name, leaf_info] : additional_leaves) {
    std::string leaf_shape = std::to_string(leaf_info.dim.channel()) + ":" +
                              std::to_string(leaf_info.dim.height()) + ":" +
                              std::to_string(leaf_info.dim.width());
    auto leaf_layer = createLayer(
      "input", {"name=" + leaf_name, "input_shape=" + leaf_shape});
    status = addLayer(std::move(leaf_layer));
    if (status != ML_ERROR_NONE) {
      return status;
    }
  }

  // 2. Add each layer in topological order with input_layers set
  for (auto &info : layers_in_order) {
    if (!info.input_layer_names.empty()) {
      std::string input_layers_str;
      for (size_t i = 0; i < info.input_layer_names.size(); ++i) {
        if (i > 0)
          input_layers_str += ",";
        input_layers_str += info.input_layer_names[i];
      }
      info.layer->setProperty({"input_layers=" + input_layers_str});
    }
    status = addLayer(info.layer);
    if (status != ML_ERROR_NONE) {
      return status;
    }
  }

  // 3. Compile the model
  status = compile(mode);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  // 4. Initialize the model
  status = initialize(mode);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  // 5. Allocate tensor memory
  status = allocate(mode);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  // 6. Materialize API tensors with allocated buffers
  for (auto &inp : inputs) {
    std::string inp_name = inp.name();
    if (inp_name.empty()) {
      inp_name =
        (inputs.size() == 1)
          ? "graph_input"
          : "graph_input_" + std::to_string(&inp - &inputs[0]);
    }
    const TensorDim &in_dim = inp.shape();
    inp.impl_->eager_data = std::make_shared<nntrainer::Tensor>(
      in_dim, true, nntrainer::Initializer::ZEROS, inp_name);
    inp.impl_->eager_data->initialize();
  }
  for (auto &output : outputs) {
    auto output_producer = output.getProducingLayer();
    if (!output_producer)
      continue;
    std::string output_layer_name = output_producer->getName();
    TensorDim out_dim;
    forEachLayer(
      [&](Layer &layer, nntrainer::RunLayerContext &rc, void *) {
        if (layer.getName() == output_layer_name && rc.getNumOutputs() > 0) {
          out_dim = rc.getOutput(0).getDim();
        }
      },
      nullptr);
    if (out_dim.getDataLen() > 0) {
      output.impl_->eager_data = std::make_shared<nntrainer::Tensor>(
        out_dim, true, nntrainer::Initializer::ZEROS, output_layer_name);
      output.impl_->eager_data->initialize();
      output.impl_->dim = out_dim;
    }
  }

  return ML_ERROR_NONE;
}

} // namespace train
} // namespace ml

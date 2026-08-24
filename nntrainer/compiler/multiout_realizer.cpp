// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file multiout_realizer.h
 * @date 17 November 2021
 * @brief NNTrainer graph realizer which realizes multiout to actual node
 * @see	https://github.com/nntrainer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <common_properties.h>
#include <compiler_fwd.h>
#include <connection.h>
#include <layer_node.h>
#include <multiout_realizer.h>
#include <remap_realizer.h>

namespace nntrainer {
MultioutRealizer::~MultioutRealizer() {}

GraphRepresentation
MultioutRealizer::realize(const GraphRepresentation &reference) {
  GraphRepresentation processed(reference.begin(), reference.end());

  std::unordered_map<Connection, unsigned> freq_map;
  std::unordered_set<std::string> node_names;
  std::vector<Connection> connections;
  // Looked up when creating each multiout node below, so it can inherit the
  // source node's compute_engine (see §2) - a node tagged engine=cdsp
  // (e.g. an "addition" layer whose output fans out to two consumers) would
  // otherwise always force a flush before its auto-generated multiout
  // successor, even though the successor does zero real compute (it's a
  // pure tensor-pool/in-place-aliasing bookkeeping node, see
  // MultiOutLayer::forwarding()).
  std::unordered_map<std::string, std::shared_ptr<LayerNode>> node_by_name;

  /// 1. build frequency map and connection names
  for (auto &node : reference) {
    NNTR_THROW_IF(node_names.count(node->getName()), std::invalid_argument)
      << "node name clashes: " << node->getName();
    node_names.emplace(node->getName());
    node_by_name.emplace(node->getName(), node);

    for (unsigned int i = 0, num_nodes = node->getNumInputConnections();
         i < num_nodes; ++i) {
      Connection c(node->getInputConnectionName(i),
                   node->getInputConnectionIndex(i));
      [[maybe_unused]] auto [iter, result] = freq_map.try_emplace(c, 0);
      if (result)
        connections.push_back(c);
      iter->second++;
    }
  }

  /// 2. for each connection names, if a connection is referenced multiple
  /// times, create multioutput node and remap to multi output node index
  std::unordered_map<
    std::string /**< original id */,
    std::vector<std::shared_ptr<LayerNode>> /**< created node */>
    multiout_nodes;

  for (auto &con : connections) {
    unsigned freq = freq_map[con];
    /// @note freq < 1 should never happen as the map entry is not created.
    /// but if it happens multiout realizer is not interested in checking if it
    /// is a dangled or actually an output. So there is no assurance done at
    /// this point. Some other class must check if the given graph is formed in
    /// a correct way.
    if (freq <= 1) {
      continue;
    }

    std::string id = con.getName();
    auto idx = con.getIndex();

    std::stringstream ss;
    /// {connection_name}/generated_out_{index}
    ss << id << "/generated_out_" << idx;
    while (node_names.count(ss.str()) != 0) {
      ss << "_";
    }
    auto multiout_name = ss.str();

    std::vector<std::string> multiout_props = {
      "name=" + multiout_name, "input_layers=" + con.toString()};
    // NOTE: LayerNode::getComputeEngineType() reads the `compute_engine`
    // member, which is only populated by LayerNode::finalize() - realizers
    // (this one included) run on the GraphRepresentation *before*
    // finalize(), so that member is always still the default "cpu" here.
    // getProperty("engine") reads layer_node_props directly instead, which
    // is populated at construction time (withHexagonEngine() sets it via
    // the "engine=cdsp" key), so it's actually visible at this point.
    auto src_it = node_by_name.find(id);
    if (src_it != node_by_name.end() &&
        src_it->second->getProperty("engine") == "cdsp") {
      multiout_props.push_back("engine=cdsp");
    }
    multiout_nodes[id].push_back(
      createLayerNode("multiout", multiout_props));
    node_names.emplace(multiout_name);

    unsigned input_count = 0;
    RemapRealizer remapper([&id, &multiout_name, idx,
                            &input_count](std::string &id_, unsigned &idx_) {
      if (id_ == id && idx_ == idx) {
        id_ = multiout_name;
        idx_ = input_count++;
      }
    });

    processed = remapper.realize(processed);
  }

  /// 3. insert multiout_nodes close to the original node to make the
  /// realization more sensible
  GraphRepresentation ret;
  ret.reserve(processed.size());
  for (auto &node : processed) {
    ret.push_back(node);
    auto ranges = multiout_nodes[node->getName()];
    for (auto &it : ranges) {
      ret.push_back(it);
    }
  }

  return ret;
}

} // namespace nntrainer

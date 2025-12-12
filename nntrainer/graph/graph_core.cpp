// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file    network_graph.h
 * @date    12 May 2020
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @author  Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is Graph Core Class for Neural Network
 *
 */

#include <algorithm>
#include <sstream>

#include <graph_core.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer {

void GraphCore::addGraphNode(std::shared_ptr<GraphNode> node) {
  node_list.push_back(node);
  node_map[node->getName()] = node_list.size() - 1;
}

const std::shared_ptr<GraphNode> &GraphCore::getNode(unsigned int ith) const {
  return node_list.at(ith);
}

const std::shared_ptr<GraphNode> &
GraphCore::getSortedNode(unsigned int ith) const {
  return Sorted.at(ith);
}

const unsigned int GraphCore::getSortedNodeIdx(const std::string &name) const {
  return sorted_node_map.at(name);
}

void GraphCore::makeAdjacencyList(
  std::vector<std::list<std::shared_ptr<GraphNode>>> &adj) {
  /** initialize the adj list */
  for (auto &node : node_list) {
    adj.push_back(std::list<std::shared_ptr<GraphNode>>({node}));
  }

  /** make the connections */
  for (unsigned int i = node_list.size() - 1; i != -1; --i) {
    auto &node = node_list[i];
    for (auto const &in_conn : node->getInputConnections()) {
      unsigned int to_node_id = getNodeIdx(in_conn);
      adj[to_node_id].push_back(node);
    }
  }
}

void GraphCore::topologicalSortUtil(
  std::vector<std::list<std::shared_ptr<GraphNode>>> &adj, unsigned int ith,
  std::vector<bool> &visited,
  std::stack<std::shared_ptr<GraphNode>> &dfs_stack) {
  visited[ith] = true;

  std::list<std::shared_ptr<GraphNode>>::iterator i;
  for (i = adj[ith].begin(); i != adj[ith].end(); ++i) {
    auto index = getNodeIdx((*i)->getName());
    if (!visited[index])
      topologicalSortUtil(adj, index, visited, dfs_stack);
  }

  dfs_stack.push(getNode(ith));
}

void GraphCore::topologicalSort() {
  std::vector<std::list<std::shared_ptr<GraphNode>>> adj;
  std::stack<std::shared_ptr<GraphNode>> dfs_stack;
  std::vector<bool> visited(node_list.size(), false);

  makeAdjacencyList(adj);
  Sorted.clear();

  // Quite likely this is not needed - verify this
  // TODO : After make node list of graph, we have to find root. (That means it
  // should be the only one input for now.). Need to support multiple input and
  // support search.

  for (unsigned int i = adj.size() - 1; i != -1; --i) {
    if (visited[i] == false) {
      topologicalSortUtil(adj, i, visited, dfs_stack);
    }
  }

  while (dfs_stack.empty() == false) {
    Sorted.push_back(dfs_stack.top());
    dfs_stack.pop();
  }

  if (Sorted.size() != node_list.size())
    throw std::runtime_error("Internal error in topologicalSort");
  unsigned int idx = 0;
  for (auto &n : Sorted) {
    sorted_node_map[n->getName()] = idx;
    idx++;
  }
}

const std::shared_ptr<GraphNode> &
GraphCore::getNode(const std::string &name) const {
  return node_list.at(node_map.at(name));
}

void GraphCore::addNode(std::shared_ptr<GraphNode> node, bool ensure_name) {
  /** Ensure that the node has a name and is unique */
  if (ensure_name)
    ensureName(*node);

  /** Insert the node to the graph */
  addGraphNode(node);
}

void GraphCore::ensureName(GraphNode &node, const std::string &prefix_,
                           const std::string &postfix_, bool force_rename) {
  auto to_lower = [](const std::string &str) -> std::string {
    std::string ret = str;
    std::transform(ret.begin(), ret.end(), ret.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return ret;
  };

  std::string orig_name = to_lower(node.getName());
  std::string prefix = to_lower(prefix_);
  std::string postfix = to_lower(postfix_);

  bool orig_name_empty = orig_name.empty();
  /** If node already has name which is unique and valid, and force is
   * disabled, then nothing to do.
   */
  if (!orig_name_empty && !force_rename && !verifyNode(orig_name)) {
    node.setName(orig_name);
    node_names.emplace(orig_name);
    return;
  }

  /** If just prefix with node name makes it unique - directly set the name */
  if (!orig_name_empty) {
    std::string direct_name = prefix + orig_name + postfix;
    if (!verifyNode(direct_name)) {
      node.setName(direct_name);
      node_names.emplace(direct_name);
      return;
    }
  }

  std::unordered_set<std::string>::iterator iter;
  std::string name;
  if (orig_name_empty) {
    orig_name = node.getType();
  }

  std::string direct_name = prefix + orig_name + postfix;

  do {
    name = direct_name + std::to_string(def_name_count++);
    iter = node_names.find(name);
  } while (iter != node_names.end());

  node.setName(name);
  node_names.emplace(name);
}

void GraphCore::replaceNode(std::shared_ptr<GraphNode> from,
                            std::shared_ptr<GraphNode> to) {
  if (node_map.find(from->getName()) == node_map.end())
    throw std::invalid_argument("Graph node to be replaced is missing");
  if (node_map.find(to->getName()) != node_map.end())
    throw std::invalid_argument("Nodes in the graph must be unique");

  unsigned int from_idx = getNodeIdx(from->getName());
  node_list[from_idx] = to;
  node_map.erase(from->getName());
  node_map[to->getName()] = from_idx;
}

void GraphCore::realizeInputOutputNode() {
  for (auto iter = cbegin(); iter != cend(); ++iter) {
    if (iter->getInputConnections().size() == 0) {
      input_list.push_back(*iter);
    }
    if (iter->getOutputConnections().size() == 0) {
      output_list.push_back(*iter);
    }
  }
}

unsigned int GraphCore::getNodeIdx(const std::string &name) {
  return node_map.at(name);
}

} /* namespace nntrainer */

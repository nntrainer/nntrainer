// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file    graph_core.cpp
 * @date    12 May 2020
 * @see     https://github.com/nntrainer/nntrainer
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
  for (auto it = node_list.rbegin(); it != node_list.rend(); ++it) {
    auto &node = *it;
    for (const auto &in_conn : node->getInputConnections()) {
      unsigned int to_node_id = getNodeIdx(in_conn);
      if (to_node_id < adj.size()) {
        adj[to_node_id].push_back(node);
      }
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

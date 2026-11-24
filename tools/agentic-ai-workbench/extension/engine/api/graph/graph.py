# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file graph.py
# @brief The compiler IR graph: its nodes and edges.
# @author prachi-t17 <prachi.t@samsung.com>

"""The compiler IR graph: its nodes and edges."""

from .edge import GraphEdge
from .node import GraphNode


class Graph:
    """
    Compiler Intermediate Representation (IR).
    """

    def __init__(self):

        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

        # Graph metadata
        self.metadata: dict = {}

        # Model metadata
        self.model_name: str | None = None
        self.architecture: str | None = None

        # Weight information
        self.parameter_count: int = 0
        self.memory_mb: float = 0.0

    # ---------------------------------------------------------

    def add_node(self, node: GraphNode):
        """Add a node."""

        self.nodes[node.id] = node

    # ---------------------------------------------------------

    def add_edge(self, edge: GraphEdge):
        """Add an edge between two nodes that are already in the graph."""

        if edge.source not in self.nodes:
            raise ValueError(f"Unknown node {edge.source}")

        if edge.target not in self.nodes:
            raise ValueError(f"Unknown node {edge.target}")

        self.edges.append(edge)

        self.nodes[edge.source].add_output(edge.target)
        self.nodes[edge.target].add_input(edge.source)

    # ---------------------------------------------------------

    def connect(
        self,
        source: GraphNode,
        target: GraphNode,
    ):
        """Connect two nodes with an edge."""

        self.add_edge(
            GraphEdge(
                source.id,
                target.id,
            )
        )

    # ---------------------------------------------------------

    def get_node(self, node_id: str):
        """The node with this id, or None."""

        return self.nodes.get(node_id)

    # ---------------------------------------------------------

    def get_nodes(self):
        """All nodes, in the order they were added."""

        return list(self.nodes.values())

    # ---------------------------------------------------------

    def get_edges(self):
        """All edges."""

        return self.edges

    # ---------------------------------------------------------

    def statistics(self):
        """Node counts: total, supported and unsupported."""

        supported = sum(
            node.supported
            for node in self.nodes.values()
        )

        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "supported_nodes": supported,
            "unsupported_nodes": len(self.nodes) - supported,
            "parameter_count": self.parameter_count,
            "memory_mb": self.memory_mb,
            "operator_types": sorted(
                {
                    node.node_type
                    for node in self.nodes.values()
                }
            ),
        }

    # ---------------------------------------------------------

    def export(self):
        """The graph as JSON-ready data (summary, metadata, nodes and edges)."""

        return {
            "summary": self.statistics(),
            "metadata": self.metadata,
            "nodes": [
                node.to_dict()
                for node in self.nodes.values()
            ],
            "edges": [
                edge.to_dict()
                for edge in self.edges
            ],
        }

    # ---------------------------------------------------------

    def __len__(self):

        return len(self.nodes)

    # ---------------------------------------------------------

    def __repr__(self):

        return (
            f"Graph("
            f"model={self.model_name}, "
            f"architecture={self.architecture}, "
            f"nodes={len(self.nodes)}, "
            f"edges={len(self.edges)})"
        )

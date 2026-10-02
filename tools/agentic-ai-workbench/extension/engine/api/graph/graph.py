import json

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

        self.nodes[node.id] = node

    # ---------------------------------------------------------

    def add_edge(self, edge: GraphEdge):

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

        self.add_edge(
            GraphEdge(
                source.id,
                target.id,
            )
        )

    # ---------------------------------------------------------

    def remove_node(self, node_id: str):

        if node_id not in self.nodes:
            return

        self.edges = [
            edge
            for edge in self.edges
            if edge.source != node_id
            and edge.target != node_id
        ]

        del self.nodes[node_id]

    # ---------------------------------------------------------

    def remove_edge(
        self,
        source: str,
        target: str,
    ):

        self.edges = [
            edge
            for edge in self.edges
            if not (
                edge.source == source
                and edge.target == target
            )
        ]

    # ---------------------------------------------------------

    def get_node(self, node_id: str):

        return self.nodes.get(node_id)

    # ---------------------------------------------------------

    def get_nodes(self):

        return list(self.nodes.values())

    # ---------------------------------------------------------

    def get_edges(self):

        return self.edges

    # ---------------------------------------------------------

    def find_by_type(self, node_type: str):

        return [
            node
            for node in self.nodes.values()
            if node.node_type == node_type
        ]

    # ---------------------------------------------------------

    def find_by_name(self, name: str):

        return [
            node
            for node in self.nodes.values()
            if name.lower() in node.name.lower()
        ]

    # ---------------------------------------------------------

    def successors(self, node: GraphNode):

        return [
            self.nodes[n]
            for n in node.outputs
            if n in self.nodes
        ]

    # ---------------------------------------------------------

    def predecessors(self, node: GraphNode):

        return [
            self.nodes[n]
            for n in node.inputs
            if n in self.nodes
        ]

    # ---------------------------------------------------------

    def roots(self):

        return [
            node
            for node in self.nodes.values()
            if not node.inputs
        ]

    # ---------------------------------------------------------

    def leaves(self):

        return [
            node
            for node in self.nodes.values()
            if not node.outputs
        ]

    # ---------------------------------------------------------

    def statistics(self):

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

    def export_json(self):

        return json.dumps(
            self.export(),
            indent=4,
        )

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
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file node.py
# @brief Node of the compiler IR graph.
# @author prachi-t17 <prachi.t@samsung.com>

"""Node of the compiler IR graph."""

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
# pylint: disable-next=too-many-instance-attributes
class GraphNode:
    """
    Represents a node in the Compiler Intermediate Representation (IR).
    """

    name: str
    node_type: str

    # "id" is the key the graph JSON and the webview use
    id: str = field(default_factory=lambda: str(uuid4()))  # pylint: disable=invalid-name

    attributes: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    # ---------------------------------------
    # Weight information
    # ---------------------------------------

    weight_name: str | None = None
    weight_shape: tuple | None = None
    weight_dtype: str | None = None
    parameter_count: int = 0

    # ---------------------------------------
    # Compiler compatibility
    # ---------------------------------------

    supported: bool = False
    compatibility_reason: str = ""

    # ---------------------------------------
    # Optimization
    # ---------------------------------------

    optimizations: list[str] = field(default_factory=list)
    compiler_passes: list[str] = field(default_factory=list)

    fused: bool = False
    constant_folded: bool = False
    dead: bool = False

    # ---------------------------------------
    # Quantization
    # ---------------------------------------

    quantized: bool = False
    precision: str = "FP32"

    # ---------------------------------------
    # Profiling
    # ---------------------------------------

    latency: float | None = None
    memory: float | None = None

    # ---------------------------------------
    # Tensor information
    # ---------------------------------------

    input_shape: tuple | None = None
    output_shape: tuple | None = None

    # ---------------------------------------
    # Semantic / target-graph metadata
    #
    # Populated by the semantic adapters and the nntrainer lowering
    # stage (see api/semantic and api/lowering/nntrainer). Left at their
    # defaults for nodes coming out of GenericFxParser, so this is a
    # strict superset that doesn't change existing behavior.
    # ---------------------------------------

    #: e.g. "attention", "mlp", "norm", "embedding", "lm_head", "kv_cache"
    semantic_type: str = ""

    #: groups nodes belonging to one repeated block, e.g. "decoder_3", so
    #: the webview can collapse/expand a whole block as one unit.
    group_id: str = ""

    #: template this node was instantiated from, e.g. "qwen3_decoder" --
    #: lets the webview recognise structurally-identical repeated blocks.
    template_id: str = ""

    #: ids of the node(s) in the *other* graph (source <-> target) this
    #: node was produced from / lowered into. Drives click-to-highlight.
    source_node_ids: list[str] = field(default_factory=list)

    #: one of: supported, fused, custom_layer, host_managed, unsupported,
    #: passthrough. Independent of `supported` (bool), which is kept for
    #: backward compatibility with the op-table compatibility report.
    status: str = "supported"

    #: the C++ variable/symbol this node lowers to, once known.
    cpp_symbol: str = ""

    #: index within a repeated block (e.g. decoder layer number).
    repeat_index: int | None = None

    # ---------------------------------------
    # Graph helpers
    # ---------------------------------------

    def add_input(self, node_id: str):
        """Record an input node (once)."""
        if node_id not in self.inputs:
            self.inputs.append(node_id)

    def add_output(self, node_id: str):
        """Record an output node (once)."""
        if node_id not in self.outputs:
            self.outputs.append(node_id)


    # ---------------------------------------
    # Serialization
    # ---------------------------------------

    def to_dict(self):
        """The node as JSON-ready data."""

        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type,
            "attributes": self.attributes,
            "metadata": self.metadata,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "supported": self.supported,
            "compatibility_reason": self.compatibility_reason,
            "optimizations": self.optimizations,
            "compiler_passes": self.compiler_passes,
            "latency": self.latency,
            "memory": self.memory,
            "precision": self.precision,
            "quantized": self.quantized,
            "fused": self.fused,
            "constant_folded": self.constant_folded,
            "dead": self.dead,
            "weight_name": self.weight_name,
            "weight_shape": self.weight_shape,
            "weight_dtype": self.weight_dtype,
            "parameter_count": self.parameter_count,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "semantic_type": self.semantic_type,
            "group_id": self.group_id,
            "template_id": self.template_id,
            "source_node_ids": self.source_node_ids,
            "status": self.status,
            "cpp_symbol": self.cpp_symbol,
            "repeat_index": self.repeat_index,
        }

    def __repr__(self):

        return (
            f"GraphNode("
            f"name={self.name}, "
            f"type={self.node_type}, "
            f"supported={self.supported})"
        )

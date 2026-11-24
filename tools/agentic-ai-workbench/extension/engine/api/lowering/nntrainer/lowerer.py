# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

##
# @file lowerer.py
# @brief NNTrainerLowerer.
# @author prachi-t17 <prachi.t@samsung.com>

"""
NNTrainerLowerer.

Converts a framework-neutral `api.semantic.model.CausalLMIR` into the
actual target graph the C++ generator consumes: real nntrainer layer
types (`fully_connected`, `reshaped_rms_norm`, `mha_core`, `addition`,
...), wired with genuine tensor dataflow -- Q/K/V all reading the same
normalized input, the attention output feeding the residual add, etc.
-- instead of whatever declaration order the source model happened to
use.

This graph is the authoritative model structure: nothing downstream re-derives
structure from generated text (see converters/cpp_generator.py and
agents/graph_views.py, both of which consume this graph directly).
"""
from __future__ import annotations

from api.graph.graph import Graph
from api.graph.node import GraphNode
from api.semantic.model import AttentionIR, CausalLMIR, DecoderLayerIR, MLPIR, NormIR

from .concurrency import ThreadSafeGraphBuilder
from .nntrainer_names import (
    bool_property,
    disable_bias,
    nntrainer_activation,
    reshaped_rms_norm_properties,
)

#: Symbolic token-count dimension for the display shapes recorded on each
#: node. The lowered graph is generic in sequence length, so shapes are
#: [seq, features]. These shapes, weight_shape and parameter_count are for
#: the webview inspector only -- unlike `attributes`, nothing emits them as
#: nntrainer layer properties.
SEQ = "seq"


def _projection_weight(proj):
    """HF nn.Linear weight layout ([out_features, in_features]) plus the
    parameter count (weight + optional bias), or (None, 0) when the input
    size isn't known."""
    if proj.input_size is None:
        return None, 0
    params = proj.output_size * proj.input_size + (proj.output_size if proj.bias else 0)
    return (proj.output_size, proj.input_size), params


# pylint: disable-next=too-few-public-methods
class NNTrainerLowerer:
    """Lowers a CausalLMIR to the nntrainer layer graph."""
    def __init__(self, model_ir: CausalLMIR):
        self.model_ir = model_ir

    def lower(self) -> Graph:
        """Build the nntrainer graph of the model."""
        graph = Graph()
        graph.model_name = self.model_ir.architecture
        graph.architecture = self.model_ir.architecture
        graph.metadata["emission_mode"] = "causallm_component"
        graph.metadata["uniform_layers"] = self.model_ir.uniform_layer_signature() is not None
        graph.metadata["architecture"] = self.model_ir.architecture
        graph.metadata["num_layers"] = len(self.model_ir.decoder_layers)

        builder = ThreadSafeGraphBuilder(graph)

        hidden = self._lower_embedding(builder)
        for layer in self.model_ir.decoder_layers:
            hidden = self._lower_decoder_layer(builder, layer, hidden)
        hidden = self._lower_final_norm(builder, hidden)
        self._lower_lm_head(builder, hidden)

        return graph

    # ------------------------------------------------------------- embedding
    def _lower_embedding(self, builder: ThreadSafeGraphBuilder) -> GraphNode:
        node = GraphNode(
            name="embedding",
            node_type="embedding",
            semantic_type="embedding",
            status="supported",
            supported=True,
            attributes={"out_dim": self.model_ir.hidden_size},
            weight_name=f"{self.model_ir.embedding_name}.weight",
            weight_shape=(self.model_ir.vocab_size, self.model_ir.hidden_size),
            parameter_count=self.model_ir.vocab_size * self.model_ir.hidden_size,
            input_shape=(SEQ,),
            output_shape=(SEQ, self.model_ir.hidden_size),
            source_node_ids=[self.model_ir.embedding_name],
        )
        return builder.add_node(node)

    # ---------------------------------------------------------- decoder layer
    def _lower_decoder_layer(
        self, builder: ThreadSafeGraphBuilder, layer: DecoderLayerIR, hidden: GraphNode
    ) -> GraphNode:
        group_id = f"decoder_{layer.index}"
        template_id = f"{self.model_ir.architecture}_decoder"
        residual = hidden

        normed = self._lower_norm(
            builder, layer.input_norm, hidden, group_id, template_id, "input_norm"
        )
        attn_out = self._lower_attention(builder, layer.attention, normed, group_id, template_id)
        after_attn = self._lower_residual_add(
            builder, residual, attn_out, group_id, template_id, "attention_residual"
        )

        residual = after_attn
        post_normed = self._lower_norm(
            builder,
            layer.post_attention_norm,
            after_attn,
            group_id,
            template_id,
            "post_attention_norm",
        )
        mlp_out = self._lower_mlp(builder, layer.mlp, post_normed, group_id, template_id)
        after_mlp = self._lower_residual_add(
            builder, residual, mlp_out, group_id, template_id, "mlp_residual"
        )
        return after_mlp

    @staticmethod
    def _lower_norm(  # pylint: disable=too-many-arguments
        builder, norm: NormIR, inp: GraphNode, group_id, template_id, tag
    ) -> GraphNode:
        if norm.reshaped:
            layer_type = "reshaped_rms_norm"
            attributes = reshaped_rms_norm_properties(norm.epsilon, norm.feature_size)
        else:
            layer_type = "rms_norm"
            attributes = {"epsilon": norm.epsilon, "feature_size": norm.feature_size}
        # A norm keeps its input's shape. Q/K norms read the projection
        # output ([seq, heads * head_dim]) even though they normalise per
        # head (feature_size = head_dim), so take the shape from the input.
        shape = inp.output_shape or (SEQ, norm.feature_size)
        node = builder.add_node(GraphNode(
            name=norm.source_name,
            node_type=layer_type,
            semantic_type="normalization",
            group_id=group_id,
            template_id=f"{template_id}.{tag}",
            status="supported",
            supported=True,
            attributes=attributes,
            weight_name=f"{norm.source_name}.weight",
            weight_shape=(norm.feature_size,),
            parameter_count=norm.feature_size,
            input_shape=shape,
            output_shape=shape,
            source_node_ids=[norm.source_name],
        ))
        builder.connect(inp, node)
        return node

    @staticmethod
    # pylint: disable-next=too-many-arguments
    def _lower_projection(
        builder, proj, inp: GraphNode, group_id, template_id, tag, layer_type="fully_connected"
    ) -> GraphNode:
        weight_shape, params = _projection_weight(proj)
        node = builder.add_node(GraphNode(
            name=proj.source_name,
            node_type=layer_type,
            semantic_type="attention" if "attn" in proj.source_name else "mlp",
            group_id=group_id,
            template_id=f"{template_id}.{tag}",
            status="supported",
            supported=True,
            attributes={"unit": proj.output_size, "disable_bias": disable_bias(proj.bias)},
            weight_name=proj.weight_name,
            weight_shape=weight_shape,
            parameter_count=params,
            input_shape=(SEQ, proj.input_size) if proj.input_size is not None else inp.output_shape,
            output_shape=(SEQ, proj.output_size),
            source_node_ids=[proj.source_name],
        ))
        builder.connect(inp, node)
        return node

    # -------------------------------------------------------------- attention
    # pylint: disable-next=too-many-arguments
    def _lower_attention(
        self,
        builder: ThreadSafeGraphBuilder,
        attn: AttentionIR,
        normed: GraphNode,
        group_id,
        template_id,
    ) -> GraphNode:
        # Q, K and V are independent of each other -- all three read
        # `normed` and none depends on another's output -- so they're
        # built concurrently on the shared, lock-guarded graph builder.
        q_node, k_node, v_node = builder.run_concurrent_branches(
            [
                lambda: self._lower_projection(
                    builder, attn.q_proj, normed, group_id, template_id, "wq"
                ),
                lambda: self._lower_projection(
                    builder, attn.k_proj, normed, group_id, template_id, "wk"
                ),
                lambda: self._lower_projection(
                    builder, attn.v_proj, normed, group_id, template_id, "wv"
                ),
            ]
        )

        if attn.q_norm is not None:
            q_node = self._lower_norm(builder, attn.q_norm, q_node, group_id, template_id, "q_norm")
        if attn.k_norm is not None:
            k_node = self._lower_norm(builder, attn.k_norm, k_node, group_id, template_id, "k_norm")

        cache = builder.add_node(GraphNode(
            name=f"{attn.source_name}.kv_cache",
            node_type="kv_cache_placeholders",
            semantic_type="kv_cache",
            group_id=group_id,
            template_id=f"{template_id}.kv_cache",
            status="host_managed",
            supported=True,
            attributes={"num_kv_heads": attn.num_kv_heads},
        ))

        mha_attributes = {
            "num_heads": attn.num_heads,
            "num_heads_kv": attn.num_kv_heads,
            "rope_theta": attn.rope_theta or 10000.0,
            "is_causal": bool_property(attn.causal),
        }
        if attn.max_position_embeddings is not None:
            mha_attributes["max_position_embeddings"] = attn.max_position_embeddings
            # nntrainer's KV-cache capacity property; distinct from the
            # rope base's max_position_embeddings even though they're
            # the same source value for every architecture this lowerer
            # currently supports.
            mha_attributes["max_timestep"] = attn.max_position_embeddings
        if attn.sliding_window is not None:
            # Only emitted when the model actually uses one -- inventing
            # "sliding_window=0" for a model with no sliding window isn't
            # verified to be nntrainer's documented "disabled" value, so
            # omitting the property entirely is the safer default.
            mha_attributes["sliding_window"] = attn.sliding_window

        mha = builder.add_node(GraphNode(
            name=f"{attn.source_name}.mha_core",
            node_type="mha_core",
            semantic_type="attention",
            group_id=group_id,
            template_id=f"{template_id}.mha_core",
            status="fused",
            supported=True,
            attributes=mha_attributes,
            input_shape=q_node.output_shape,
            output_shape=(SEQ, attn.num_heads * attn.head_dim),
            source_node_ids=[attn.source_name, f"{attn.source_name.rsplit('.', 1)[0]}.rotary_emb"],
        ))
        for src in (q_node, k_node, v_node, cache):
            builder.connect(src, mha)

        return self._lower_projection(builder, attn.o_proj, mha, group_id, template_id, "wo")

    # -------------------------------------------------------------------- mlp
    # pylint: disable-next=too-many-arguments
    def _lower_mlp(
        self, builder: ThreadSafeGraphBuilder, mlp: MLPIR, inp: GraphNode, group_id, template_id
    ) -> GraphNode:
        activation_name = nntrainer_activation(mlp.activation)

        if mlp.gated:
            # gate/up are independent of each other, both read `inp`.
            gate, up_proj = builder.run_concurrent_branches(
                [
                    lambda: self._lower_projection(
                        builder, mlp.gate_proj, inp, group_id, template_id, "gate"
                    ),
                    lambda: self._lower_projection(
                        builder, mlp.up_proj, inp, group_id, template_id, "up"
                    ),
                ]
            )
            activated = builder.add_node(GraphNode(
                name=f"{mlp.source_name}.activation", node_type="activation",
                semantic_type="mlp", group_id=group_id, template_id=f"{template_id}.act",
                status="supported", supported=True,
                attributes={"activation": activation_name},
                input_shape=gate.output_shape, output_shape=gate.output_shape,
            ))
            builder.connect(gate, activated)
            gated = builder.add_node(GraphNode(
                name=f"{mlp.source_name}.gate_mul", node_type="multiply",
                semantic_type="mlp", group_id=group_id, template_id=f"{template_id}.mul",
                status="supported", supported=True,
                input_shape=up_proj.output_shape, output_shape=up_proj.output_shape,
            ))
            builder.connect(activated, gated)
            builder.connect(up_proj, gated)
            return self._lower_projection(
                builder, mlp.down_proj, gated, group_id, template_id, "down"
            )

        up_proj = self._lower_projection(builder, mlp.up_proj, inp, group_id, template_id, "up")
        activated = builder.add_node(GraphNode(
            name=f"{mlp.source_name}.activation", node_type="activation",
            semantic_type="mlp", group_id=group_id, template_id=f"{template_id}.act",
            status="supported", supported=True,
            attributes={"activation": activation_name},
            input_shape=up_proj.output_shape, output_shape=up_proj.output_shape,
        ))
        builder.connect(up_proj, activated)
        return self._lower_projection(
            builder, mlp.down_proj, activated, group_id, template_id, "down"
        )

    @staticmethod
    # pylint: disable-next=too-many-arguments
    def _lower_residual_add(
        builder, residual: GraphNode, branch: GraphNode, group_id, template_id, tag
    ) -> GraphNode:
        node = builder.add_node(GraphNode(
            name=f"{group_id}.{tag}", node_type="addition",
            semantic_type="residual", group_id=group_id, template_id=f"{template_id}.{tag}",
            status="supported", supported=True,
            input_shape=residual.output_shape, output_shape=residual.output_shape,
        ))
        builder.connect(residual, node)
        builder.connect(branch, node)
        return node

    # ------------------------------------------------------------- final / lm
    def _lower_final_norm(self, builder, hidden: GraphNode) -> GraphNode:
        return self._lower_norm(
            builder, self.model_ir.final_norm, hidden, "final", "final", "final_norm"
        )

    def _lower_lm_head(self, builder, hidden: GraphNode) -> GraphNode:
        node = builder.add_node(GraphNode(
            name=self.model_ir.lm_head_name,
            node_type="fully_connected",
            semantic_type="lm_head",
            group_id="lm_head",
            template_id="lm_head",
            status="supported" if not self.model_ir.tied_embeddings else "fused",
            supported=True,
            attributes={"unit": self.model_ir.vocab_size, "disable_bias": disable_bias(False)},
            weight_name=(
                f"{self.model_ir.embedding_name}.weight" if self.model_ir.tied_embeddings
                else f"{self.model_ir.lm_head_name}.weight"
            ),
            weight_shape=(self.model_ir.vocab_size, self.model_ir.hidden_size),
            # Tied: the weight is the embedding's, already counted there.
            parameter_count=0 if self.model_ir.tied_embeddings
            else self.model_ir.vocab_size * self.model_ir.hidden_size,
            input_shape=(SEQ, self.model_ir.hidden_size),
            output_shape=(SEQ, self.model_ir.vocab_size),
            source_node_ids=[self.model_ir.lm_head_name],
        ))
        builder.connect(hidden, node)
        return node

"""
NNTrainerLowerer.

Converts a framework-neutral `api.semantic.model.CausalLMIR` into the
actual target graph the C++ generator consumes: real nntrainer layer
types (`fully_connected`, `reshaped_rms_norm`, `mha_core`, `addition`,
...), wired with genuine tensor dataflow -- Q/K/V all reading the same
normalized input, the attention output feeding the residual add, etc.
-- instead of whatever declaration order the source model happened to
use.

This is the "authoritative model understanding layer" the semantic
lowering redesign is built around: nothing downstream re-derives
structure from generated text (see converters/cpp_generator.py and
agents/graph_views.py, both of which consume this graph directly).
"""
from __future__ import annotations

from api.graph.graph import Graph
from api.graph.node import GraphNode
from api.semantic.model import AttentionIR, CausalLMIR, DecoderLayerIR, MLPIR, NormIR

from .concurrency import ThreadSafeGraphBuilder
from .nntrainer_names import bool_property, disable_bias, nntrainer_activation, reshaped_rms_norm_properties


class NNTrainerLowerer:
    def __init__(self, model_ir: CausalLMIR):
        self.model_ir = model_ir

    def lower(self) -> Graph:
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
    def _lower_embedding(self, b: ThreadSafeGraphBuilder) -> GraphNode:
        node = GraphNode(
            name="embedding",
            node_type="embedding",
            semantic_type="embedding",
            status="supported",
            supported=True,
            attributes={"out_dim": self.model_ir.hidden_size},
            weight_name=f"{self.model_ir.embedding_name}.weight",
            source_node_ids=[self.model_ir.embedding_name],
        )
        return b.add_node(node)

    # ---------------------------------------------------------- decoder layer
    def _lower_decoder_layer(self, b: ThreadSafeGraphBuilder, layer: DecoderLayerIR, hidden: GraphNode) -> GraphNode:
        group_id = f"decoder_{layer.index}"
        template_id = f"{self.model_ir.architecture}_decoder"
        residual = hidden

        normed = self._lower_norm(b, layer.input_norm, hidden, group_id, template_id, "input_norm")
        attn_out = self._lower_attention(b, layer.attention, normed, group_id, template_id)
        after_attn = self._lower_residual_add(b, residual, attn_out, group_id, template_id, "attention_residual")

        residual = after_attn
        post_normed = self._lower_norm(b, layer.post_attention_norm, after_attn, group_id, template_id, "post_attention_norm")
        mlp_out = self._lower_mlp(b, layer.mlp, post_normed, group_id, template_id)
        after_mlp = self._lower_residual_add(b, residual, mlp_out, group_id, template_id, "mlp_residual")
        return after_mlp

    def _lower_norm(self, b, norm: NormIR, inp: GraphNode, group_id, template_id, tag) -> GraphNode:
        if norm.reshaped:
            layer_type = "reshaped_rms_norm"
            attributes = reshaped_rms_norm_properties(norm.epsilon, norm.feature_size)
        else:
            layer_type = "rms_norm"
            attributes = {"epsilon": norm.epsilon, "feature_size": norm.feature_size}
        node = b.add_node(GraphNode(
            name=norm.source_name,
            node_type=layer_type,
            semantic_type="normalization",
            group_id=group_id,
            template_id=f"{template_id}.{tag}",
            status="supported",
            supported=True,
            attributes=attributes,
            weight_name=f"{norm.source_name}.weight",
            source_node_ids=[norm.source_name],
        ))
        b.connect(inp, node)
        return node

    def _lower_projection(self, b, proj, inp: GraphNode, group_id, template_id, tag, layer_type="fully_connected") -> GraphNode:
        node = b.add_node(GraphNode(
            name=proj.source_name,
            node_type=layer_type,
            semantic_type="attention" if "attn" in proj.source_name else "mlp",
            group_id=group_id,
            template_id=f"{template_id}.{tag}",
            status="supported",
            supported=True,
            attributes={"unit": proj.output_size, "disable_bias": disable_bias(proj.bias)},
            weight_name=proj.weight_name,
            source_node_ids=[proj.source_name],
        ))
        b.connect(inp, node)
        return node

    # -------------------------------------------------------------- attention
    def _lower_attention(self, b: ThreadSafeGraphBuilder, attn: AttentionIR, normed: GraphNode, group_id, template_id) -> GraphNode:
        # Q, K and V are independent of each other -- all three read
        # `normed` and none depends on another's output -- so they're
        # built concurrently on the shared, lock-guarded graph builder.
        q, k, v = b.run_concurrent_branches([
            lambda: self._lower_projection(b, attn.q_proj, normed, group_id, template_id, "wq"),
            lambda: self._lower_projection(b, attn.k_proj, normed, group_id, template_id, "wk"),
            lambda: self._lower_projection(b, attn.v_proj, normed, group_id, template_id, "wv"),
        ])

        if attn.q_norm is not None:
            q = self._lower_norm(b, attn.q_norm, q, group_id, template_id, "q_norm")
        if attn.k_norm is not None:
            k = self._lower_norm(b, attn.k_norm, k, group_id, template_id, "k_norm")

        cache = b.add_node(GraphNode(
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

        mha = b.add_node(GraphNode(
            name=f"{attn.source_name}.mha_core",
            node_type="mha_core",
            semantic_type="attention",
            group_id=group_id,
            template_id=f"{template_id}.mha_core",
            status="fused",
            supported=True,
            attributes=mha_attributes,
            source_node_ids=[attn.source_name, f"{attn.source_name.rsplit('.', 1)[0]}.rotary_emb"],
        ))
        for src in (q, k, v, cache):
            b.connect(src, mha)

        return self._lower_projection(b, attn.o_proj, mha, group_id, template_id, "wo")

    # -------------------------------------------------------------------- mlp
    def _lower_mlp(self, b: ThreadSafeGraphBuilder, mlp: MLPIR, inp: GraphNode, group_id, template_id) -> GraphNode:
        activation_name = nntrainer_activation(mlp.activation)

        if mlp.gated:
            # gate/up are independent of each other, both read `inp`.
            gate, up = b.run_concurrent_branches([
                lambda: self._lower_projection(b, mlp.gate_proj, inp, group_id, template_id, "gate"),
                lambda: self._lower_projection(b, mlp.up_proj, inp, group_id, template_id, "up"),
            ])
            activated = b.add_node(GraphNode(
                name=f"{mlp.source_name}.activation", node_type="activation",
                semantic_type="mlp", group_id=group_id, template_id=f"{template_id}.act",
                status="supported", supported=True,
                attributes={"activation": activation_name},
            ))
            b.connect(gate, activated)
            gated = b.add_node(GraphNode(
                name=f"{mlp.source_name}.gate_mul", node_type="multiply",
                semantic_type="mlp", group_id=group_id, template_id=f"{template_id}.mul",
                status="supported", supported=True,
            ))
            b.connect(activated, gated)
            b.connect(up, gated)
            return self._lower_projection(b, mlp.down_proj, gated, group_id, template_id, "down")

        up = self._lower_projection(b, mlp.up_proj, inp, group_id, template_id, "up")
        activated = b.add_node(GraphNode(
            name=f"{mlp.source_name}.activation", node_type="activation",
            semantic_type="mlp", group_id=group_id, template_id=f"{template_id}.act",
            status="supported", supported=True,
            attributes={"activation": activation_name},
        ))
        b.connect(up, activated)
        return self._lower_projection(b, mlp.down_proj, activated, group_id, template_id, "down")

    def _lower_residual_add(self, b, residual: GraphNode, branch: GraphNode, group_id, template_id, tag) -> GraphNode:
        node = b.add_node(GraphNode(
            name=f"{group_id}.{tag}", node_type="addition",
            semantic_type="residual", group_id=group_id, template_id=f"{template_id}.{tag}",
            status="supported", supported=True,
        ))
        b.connect(residual, node)
        b.connect(branch, node)
        return node

    # ------------------------------------------------------------- final / lm
    def _lower_final_norm(self, b, hidden: GraphNode) -> GraphNode:
        return self._lower_norm(b, self.model_ir.final_norm, hidden, "final", "final", "final_norm")

    def _lower_lm_head(self, b, hidden: GraphNode) -> GraphNode:
        node = b.add_node(GraphNode(
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
            source_node_ids=[self.model_ir.lm_head_name],
        ))
        b.connect(hidden, node)
        return node

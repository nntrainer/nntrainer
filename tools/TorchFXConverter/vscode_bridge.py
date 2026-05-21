"""
VS Code bridge utilities for TorchFXConverter.

Provides serialization of TorchFX graphs and node mapping between
FX graph nodes and NNTrainer layers for visualization.
"""

import json


def serialize_fx_graph(graph):
    """Serialize torch.fx Graph to list of node dicts.

    Args:
        graph: torch.fx.Graph object from the tracer.

    Returns:
        List of dicts, each describing one FX graph node with its
        name, op type, target, arguments, and metadata.
    """
    nodes = []
    for node in graph.nodes:
        entry = {
            "name": node.name,
            "op": node.op,
            "target": str(node.target),
            "args": [str(a) for a in node.args if hasattr(a, 'name')],
            "output_shape": node.meta.get("output_shape"),
            "module_type": node.meta.get("module_type"),
            "scope": node.meta.get("scope", ""),
            "meta": {}
        }
        safe_keys = [
            "leaf_module", "has_weight", "has_bias",
            "in_features", "out_features", "num_embeddings", "embedding_dim",
            "normalized_shape", "eps", "hidden_size",
            "is_rmsnorm", "is_rnn_module"
        ]
        for key in safe_keys:
            if key in node.meta:
                val = node.meta[key]
                try:
                    json.dumps(val)
                    entry["meta"][key] = val
                except (TypeError, ValueError):
                    entry["meta"][key] = str(val)
        nodes.append(entry)
    return nodes


def serialize_layer(layer):
    """Serialize NNTrainerLayerDef to dict.

    Args:
        layer: NNTrainerLayerDef instance.

    Returns:
        Dict with all layer properties serialized for JSON output.
    """
    props = {}
    for k, v in layer.properties.items():
        try:
            json.dumps(v)
            props[k] = v
        except (TypeError, ValueError):
            props[k] = str(v)

    return {
        "name": layer.name,
        "layer_type": layer.layer_type,
        "properties": props,
        "input_layers": list(layer.input_layers),
        "fx_node_name": layer.fx_node_name,
        "hf_module_name": layer.hf_module_name,
        "hf_module_type": layer.hf_module_type,
        "has_weight": layer.has_weight,
        "has_bias": layer.has_bias,
        "weight_hf_key": layer.weight_hf_key,
        "bias_hf_key": layer.bias_hf_key,
        "transpose_weight": layer.transpose_weight,
        "shared_from": layer.shared_from,
    }


def serialize_structure(structure):
    """Serialize ModelStructure to dict.

    Args:
        structure: ModelStructure instance (or None).

    Returns:
        Dict with model architecture info, or None.
    """
    if structure is None:
        return None

    blocks = []
    for b in (structure.blocks or []):
        block = {
            "block_idx": b.block_idx,
            "pre_attn_norm": b.pre_attn_norm,
            "attn_residual": b.attn_residual,
            "pre_ffn_norm": b.pre_ffn_norm,
            "ffn_residual": b.ffn_residual,
            "norm_type": b.norm_type,
            "attention": None,
            "ffn": None,
        }
        if b.attention:
            a = b.attention
            block["attention"] = {
                "block_idx": a.block_idx,
                "q_proj": a.q_proj,
                "k_proj": a.k_proj,
                "v_proj": a.v_proj,
                "o_proj": a.o_proj,
                "attention_type": a.attention_type,
                "has_rope": a.has_rope,
                "num_heads": getattr(a, "num_heads", 0),
                "num_kv_heads": getattr(a, "num_kv_heads", 0),
                "head_dim": getattr(a, "head_dim", 0),
            }
        if b.ffn:
            f = b.ffn
            block["ffn"] = {
                "block_idx": f.block_idx,
                "ffn_type": f.ffn_type,
                "gate_proj": getattr(f, "gate_proj", ""),
                "up_proj": getattr(f, "up_proj", ""),
                "down_proj": getattr(f, "down_proj", ""),
                "intermediate_size": getattr(f, "intermediate_size", 0),
            }
        blocks.append(block)

    return {
        "model_type": structure.model_type,
        "arch_type": structure.arch_type,
        "vocab_size": structure.vocab_size,
        "hidden_size": structure.hidden_size,
        "num_layers": structure.num_layers,
        "num_heads": structure.num_heads,
        "num_kv_heads": structure.num_kv_heads,
        "head_dim": structure.head_dim,
        "intermediate_size": structure.intermediate_size,
        "norm_eps": structure.norm_eps,
        "tie_word_embeddings": structure.tie_word_embeddings,
        "rope_theta": structure.rope_theta,
        "embedding": structure.embedding,
        "final_norm": structure.final_norm,
        "lm_head": structure.lm_head,
        "blocks": blocks,
    }


def build_node_mapping(layers, fx_graph):
    """Build mapping between FX graph nodes and NNTrainer layers.

    Args:
        layers: List of NNTrainerLayerDef instances.
        fx_graph: torch.fx.Graph from the tracer.

    Returns:
        List of mapping dicts with fxNodeName, nntrainerLayerName,
        hfModuleName, and mappingType.
    """
    mappings = []
    fx_node_names = {node.name for node in fx_graph.nodes}
    mapped_fx = set()

    for layer in layers:
        if layer.fx_node_name and layer.fx_node_name in fx_node_names:
            mappings.append({
                "fxNodeName": layer.fx_node_name,
                "nntrainerLayerName": layer.name,
                "hfModuleName": layer.hf_module_name,
                "mappingType": "direct"
            })
            mapped_fx.add(layer.fx_node_name)

    for node in fx_graph.nodes:
        if node.name not in mapped_fx and node.op not in ("placeholder", "output"):
            mappings.append({
                "fxNodeName": node.name,
                "nntrainerLayerName": "",
                "hfModuleName": "",
                "mappingType": "skipped"
            })

    return mappings

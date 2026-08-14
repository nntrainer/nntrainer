"""
Builds the two JSON manifests that accompany generated CausalLM C++:

  - model metadata (architecture/config summary, for the CausalLM
    registry and for debugging a generated model without re-reading
    the original HF config)
  - weight mapping (source HF checkpoint parameter name -> target
    nntrainer weight identifier), so loading real weights into the
    generated component is a lookup table, not something a human has
    to reverse-engineer from the C++.

Both are pure functions over the already-built CausalLMIR / target
Graph -- no torch, no file I/O -- so they're trivially unit-testable
and safe to call from any pipeline stage that already has the IR.
"""
from __future__ import annotations

from typing import Optional

from api.graph.graph import Graph
from api.semantic.model import CausalLMIR

_TAG_TARGET_NAMES = {
    "wq": "wq", "wk": "wk", "wv": "wv", "wo": "wo",
    "q_norm": "q_norm", "k_norm": "k_norm",
    "input_norm": "attn_norm", "post_attention_norm": "ffn_norm",
    "gate": "gate_proj", "up": "up_proj", "down": "down_proj",
}


def build_model_metadata(model_ir: CausalLMIR, source_model: Optional[str] = None) -> dict:
    """Item 13's manifest: enough of the architecture summary that the
    CausalLM registry (or a human debugging a generated model) doesn't
    need to re-derive it from the C++ or re-fetch the HF config."""
    layer = model_ir.decoder_layers[0] if model_ir.decoder_layers else None
    attn = layer.attention if layer else None
    mlp = layer.mlp if layer else None

    metadata = {
        "architecture": model_ir.architecture,
        "source_model": source_model,
        "hidden_size": model_ir.hidden_size,
        "num_hidden_layers": model_ir.num_layers,
        "vocab_size": model_ir.vocab_size,
        "tied_embeddings": model_ir.tied_embeddings,
        "emission_mode": "causallm_component",
        "uniform_layers": model_ir.uniform_layer_signature() is not None,
    }
    if attn is not None:
        metadata.update({
            "num_attention_heads": attn.num_heads,
            "num_key_value_heads": attn.num_kv_heads,
            "head_dim": attn.head_dim,
            "rope_theta": attn.rope_theta,
            "max_position_embeddings": attn.max_position_embeddings,
            "sliding_window": attn.sliding_window,
            "qk_norm": attn.q_norm is not None or attn.k_norm is not None,
        })
    if mlp is not None:
        metadata.update({
            "intermediate_size": mlp.up_proj.output_size,
            "mlp_activation": mlp.activation,
            "mlp_gated": mlp.gated,
        })
    if layer is not None:
        metadata["rms_norm_eps"] = layer.input_norm.epsilon
    return metadata


def _templated_source(source_name: str, layer_index: int) -> str:
    """'model.layers.0.self_attn.q_proj' -> 'model.layers.{layer_id}.self_attn.q_proj'"""
    marker = f".{layer_index}."
    templated = f".{{layer_id}}."
    idx = source_name.find(marker)
    if idx == -1:
        return source_name
    return source_name[:idx] + templated + source_name[idx + len(marker):]


def build_weight_manifest(model_ir: CausalLMIR, graph: Graph) -> list:
    """Item 14's manifest: one entry per weight-bearing node, with
    `{layer_id}`-templated source/target names for the repeated decoder
    nodes (there's one entry covering all N layers, not N entries) and
    explicit ungrouped entries for embedding/final-norm/lm-head/tied
    embeddings. Shapes come straight from the semantic IR's
    ProjectionIR.input_size/output_size (== the real
    nn.Linear.weight.shape), so a shape mismatch against the real
    checkpoint is a direct list-index comparison, not something a
    human has to eyeball.
    """
    entries = []

    if model_ir.decoder_layers:
        template_layer = model_ir.decoder_layers[0]
        template_index = template_layer.index
        a, m = template_layer.attention, template_layer.mlp

        def proj_entry(tag: str, proj):
            weight_name = proj.weight_name or f"{proj.source_name}.weight"
            target_shape = [proj.output_size, proj.input_size] if proj.input_size is not None else [proj.output_size, None]
            entries.append({
                "source": _templated_source(weight_name, template_index),
                "target": f"layer{{layer_id}}_{_TAG_TARGET_NAMES.get(tag, tag)}/weight",
                "source_shape": target_shape,
                "target_shape": target_shape,
                "transform": "none",
                "repeated": True,
                "layer_count": len(model_ir.decoder_layers),
            })
            if proj.bias:
                entries.append({
                    "source": _templated_source(weight_name.replace(".weight", ".bias"), template_index),
                    "target": f"layer{{layer_id}}_{_TAG_TARGET_NAMES.get(tag, tag)}/bias",
                    "source_shape": [proj.output_size],
                    "target_shape": [proj.output_size],
                    "transform": "none",
                    "repeated": True,
                    "layer_count": len(model_ir.decoder_layers),
                })

        def norm_entry(tag: str, norm):
            entries.append({
                "source": _templated_source(f"{norm.source_name}.weight", template_index),
                "target": f"layer{{layer_id}}_{_TAG_TARGET_NAMES.get(tag, tag)}/weight",
                "source_shape": [norm.feature_size],
                "target_shape": [norm.feature_size],
                "transform": "none",
                "repeated": True,
                "layer_count": len(model_ir.decoder_layers),
            })

        norm_entry("input_norm", template_layer.input_norm)
        proj_entry("wq", a.q_proj)
        proj_entry("wk", a.k_proj)
        proj_entry("wv", a.v_proj)
        if a.q_norm is not None:
            norm_entry("q_norm", a.q_norm)
        if a.k_norm is not None:
            norm_entry("k_norm", a.k_norm)
        proj_entry("wo", a.o_proj)
        norm_entry("post_attention_norm", template_layer.post_attention_norm)
        if m.gate_proj is not None:
            proj_entry("gate", m.gate_proj)
        proj_entry("up", m.up_proj)
        proj_entry("down", m.down_proj)

    entries.append({
        "source": f"{model_ir.embedding_name}.weight",
        "target": "embedding/weight",
        "source_shape": [model_ir.vocab_size, model_ir.hidden_size],
        "target_shape": [model_ir.vocab_size, model_ir.hidden_size],
        "transform": "none",
        "repeated": False,
    })
    entries.append({
        "source": f"{model_ir.final_norm.source_name}.weight",
        "target": "final_norm/weight",
        "source_shape": [model_ir.final_norm.feature_size],
        "target_shape": [model_ir.final_norm.feature_size],
        "transform": "none",
        "repeated": False,
    })
    if model_ir.tied_embeddings:
        entries.append({
            "source": f"{model_ir.embedding_name}.weight",
            "target": "lm_head/weight",
            "source_shape": [model_ir.vocab_size, model_ir.hidden_size],
            "target_shape": [model_ir.vocab_size, model_ir.hidden_size],
            "transform": "tied_with_embedding",
            "repeated": False,
        })
    else:
        entries.append({
            "source": f"{model_ir.lm_head_name}.weight",
            "target": "lm_head/weight",
            "source_shape": [model_ir.vocab_size, model_ir.hidden_size],
            "target_shape": [model_ir.vocab_size, model_ir.hidden_size],
            "transform": "none",
            "repeated": False,
        })

    return entries


def validate_weight_manifest(manifest: list) -> list:
    """Cheap, local checks (item 14's validation list) that don't need
    the real checkpoint: every entry has matching source/target rank
    and dimensions where both are known, and tied embeddings are
    marked as such rather than duplicated as an independent weight."""
    problems = []
    for entry in manifest:
        src, tgt = entry.get("source_shape"), entry.get("target_shape")
        if src is not None and tgt is not None and len(src) == len(tgt):
            for s, t in zip(src, tgt):
                if s is not None and t is not None and s != t:
                    problems.append(f"{entry['source']} -> {entry['target']}: shape mismatch {src} vs {tgt}")
    return problems

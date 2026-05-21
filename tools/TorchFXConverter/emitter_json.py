"""
JSON configuration emitter for NNTrainer TorchFX converter.

Generates a JSON configuration file describing the model structure,
layer definitions, and weight mapping. This is useful for:
  - Tool-based processing and automation
  - Weight conversion scripts
  - Cross-platform model exchange
  - Debugging and visualization

Phase 4.3 of the TorchFX converter pipeline (DESIGN.md).
"""

import json


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that converts non-serializable objects to strings."""
    def default(self, o):
        return str(o)


def _json_safe(obj):
    """Convert non-JSON-serializable objects to strings."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _safe_properties(props):
    """Return a copy of properties dict with all values JSON-serializable."""
    return {k: _json_safe(v) for k, v in props.items()}


from emitter_base import BaseEmitter
from pattern_detector import ModelStructure
from nntrainer_layers import NNTrainerLayerDef


# =============================================================================
# JSON Emitter
# =============================================================================

class JsonEmitter(BaseEmitter):
    """Generates JSON model configuration from converter output.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        structure: ModelStructure from pattern detection.
    """

    def __init__(self, layers, structure, model_name=None):
        super().__init__(layers, structure, model_name=model_name)

    def emit(self):
        """Generate JSON configuration dict.

        Returns:
            dict: Complete model configuration.
        """
        config = {}
        config["model"] = self._emit_model_info()
        config["layers"] = self._emit_layers()
        config["weight_map"] = self._emit_weight_map()
        if self.structure and self.structure.blocks:
            config["structure"] = self._emit_structure()
        return config

    def emit_string(self, indent=2):
        """Generate JSON string.

        Returns:
            str: JSON-formatted string.
        """
        return json.dumps(self.emit(), indent=indent, ensure_ascii=False,
                          cls=_SafeEncoder)

    # =========================================================================
    # Model Info
    # =========================================================================

    def _emit_model_info(self):
        s = self.structure
        info = {
            "model_type": s.model_type if s else "",
            "arch_type": s.arch_type if s else "",
            "vocab_size": s.vocab_size if s else 0,
            "hidden_size": s.hidden_size if s else 0,
            "num_layers": s.num_layers if s else 0,
            "num_heads": s.num_heads if s else 0,
            "num_kv_heads": s.num_kv_heads if s else 0,
            "head_dim": s.head_dim if s else 0,
            "intermediate_size": s.intermediate_size if s else 0,
            "norm_eps": s.norm_eps if s else 0,
            "tie_word_embeddings": s.tie_word_embeddings if s else False,
        }
        if s and s.rope_theta:
            info["rope_theta"] = s.rope_theta
        return info

    # =========================================================================
    # Layer Definitions
    # =========================================================================

    def _emit_layers(self):
        """Emit all layers as JSON objects."""
        result = []
        for layer in self.layers:
            entry = {
                "name": layer.name,
                "type": layer.layer_type,
            }
            if layer.input_layers:
                entry["input_layers"] = layer.input_layers
            if layer.properties:
                entry["properties"] = _safe_properties(layer.properties)
            if layer.hf_module_name:
                entry["hf_module_name"] = layer.hf_module_name
            if layer.hf_module_type:
                entry["hf_module_type"] = layer.hf_module_type
            result.append(entry)
        return result

    # =========================================================================
    # Weight Map
    # =========================================================================

    def _emit_weight_map(self):
        """Emit HF state_dict key -> NNTrainer layer name mapping.

        Includes safetensors-compatible tensor names (layer_name:weight,
        layer_name:bias) for cross-format weight loading.
        """
        weight_map = []
        for layer in self.layers:
            if not layer.has_weight and not layer.has_bias:
                continue
            entry = {
                "layer_name": layer.name,
                "layer_type": layer.layer_type,
            }
            if layer.has_weight:
                entry["weight_key"] = layer.weight_hf_key
                entry["transpose_weight"] = layer.transpose_weight
                entry["safetensors_name"] = layer.name + ":weight"
            if layer.has_bias:
                entry["bias_key"] = layer.bias_hf_key
                entry["safetensors_bias_name"] = layer.name + ":bias"
            if layer.shared_from:
                entry["shared_from"] = layer.shared_from
            weight_map.append(entry)
        return weight_map

    # =========================================================================
    # Structure (from pattern detection)
    # =========================================================================

    def _emit_structure(self):
        """Emit detected model structure summary."""
        s = self.structure
        result = {
            "embedding": s.embedding,
            "final_norm": s.final_norm,
            "lm_head": s.lm_head,
            "blocks": [],
        }

        for block in s.blocks:
            b = {
                "block_idx": block.block_idx,
                "norm_type": block.norm_type,
                "pre_attn_norm": block.pre_attn_norm,
                "attn_residual": block.attn_residual,
                "pre_ffn_norm": block.pre_ffn_norm,
                "ffn_residual": block.ffn_residual,
            }
            if block.post_attn_norm:
                b["post_attn_norm"] = block.post_attn_norm
            if block.post_ffn_norm:
                b["post_ffn_norm"] = block.post_ffn_norm

            if block.attention:
                attn = block.attention
                b["attention"] = {
                    "type": attn.attention_type,
                    "q_proj": attn.q_proj,
                    "k_proj": attn.k_proj,
                    "v_proj": attn.v_proj,
                    "o_proj": attn.o_proj,
                    "has_qk_norm": attn.has_qk_norm,
                    "has_rope": attn.has_rope,
                    "num_heads": attn.num_heads,
                    "num_kv_heads": attn.num_kv_heads,
                    "head_dim": attn.head_dim,
                }
                if attn.q_norm:
                    b["attention"]["q_norm"] = attn.q_norm
                if attn.k_norm:
                    b["attention"]["k_norm"] = attn.k_norm

            if block.cross_attention:
                xattn = block.cross_attention
                b["cross_attention"] = {
                    "type": xattn.attention_type,
                    "q_proj": xattn.q_proj,
                    "k_proj": xattn.k_proj,
                    "v_proj": xattn.v_proj,
                    "o_proj": xattn.o_proj,
                }

            if block.ffn:
                ffn = block.ffn
                b["ffn"] = {
                    "type": ffn.ffn_type,
                    "intermediate_size": ffn.intermediate_size,
                }
                if ffn.ffn_type in ("swiglu", "geglu") or ffn.ffn_type.startswith("gated_"):
                    b["ffn"]["gate_proj"] = ffn.gate_proj
                    b["ffn"]["up_proj"] = ffn.up_proj
                    b["ffn"]["down_proj"] = ffn.down_proj
                else:
                    b["ffn"]["fc1"] = ffn.up_proj
                    b["ffn"]["fc2"] = ffn.down_proj

            result["blocks"].append(b)

        return result


# =============================================================================
# Convenience function
# =============================================================================

def emit_json(layers, structure):
    """Generate JSON model configuration.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        structure: ModelStructure from pattern detection.

    Returns:
        dict: Complete model configuration.
    """
    emitter = JsonEmitter(layers, structure)
    return emitter.emit()


def emit_json_string(layers, structure, indent=2):
    """Generate JSON string.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        structure: ModelStructure from pattern detection.
        indent: JSON indentation.

    Returns:
        str: JSON-formatted string.
    """
    emitter = JsonEmitter(layers, structure)
    return emitter.emit_string(indent=indent)

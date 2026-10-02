"""
Layer Registry and Detection

Detects available layers in nntrainer codebase and maintains
a registry of layer implementations with their properties.

Scans:
- /Applications/CausalLM/layers/ - Custom transformer layers
- /nntrainer/layers/ - Built-in nntrainer layers
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LayerInfo:
    """Information about a layer implementation."""
    name: str                    # Layer identifier (e.g., "rms_norm")
    type: str                    # "builtin", "custom", "generated"
    file_path: Optional[str]     # Path to implementation
    header_file: Optional[str]   # Header file if separate
    usage_string: str            # How to use: createLayer("name", {...})
    properties: Dict[str, str]   # Property name -> description
    inputs: List[str]            # Input tensor names
    outputs: List[str]           # Output tensor names
    description: str
    reference: Optional[str]     # URL or reference documentation


class LayerRegistry:
    """Registry of available layer implementations."""

    # Built-in layers from nntrainer (standard)
    BUILTIN_LAYERS = {
        "embedding": LayerInfo(
            name="embedding",
            type="builtin",
            file_path=None,
            header_file="<layer.h>",
            usage_string='createLayer("embedding")',
            properties={
                "name": "Layer name",
                "num_embed": "Vocabulary size",
                "dim": "Embedding dimension",
            },
            inputs=["input_ids"],
            outputs=["embedded"],
            description="Token to vector projection",
            reference="nntrainer built-in",
        ),
        "fully_connected": LayerInfo(
            name="fully_connected",
            type="builtin",
            file_path=None,
            header_file="<layer.h>",
            usage_string='createLayer("fully_connected")',
            properties={
                "name": "Layer name",
                "unit": "Output dimension",
                "disable_bias": "Disable bias (true/false)",
                "weight_initializer": "Weight initialization",
            },
            inputs=["input"],
            outputs=["output"],
            description="Linear projection / Dense layer",
            reference="nntrainer built-in",
        ),
        "addition": LayerInfo(
            name="addition",
            type="builtin",
            file_path=None,
            header_file="<layer.h>",
            usage_string='createLayer("addition")',
            properties={
                "name": "Layer name",
            },
            inputs=["input1", "input2"],
            outputs=["sum"],
            description="Element-wise addition (residual connection)",
            reference="nntrainer built-in",
        ),
        "multiply": LayerInfo(
            name="multiply",
            type="builtin",
            file_path=None,
            header_file="<layer.h>",
            usage_string='createLayer("multiply")',
            properties={
                "name": "Layer name",
            },
            inputs=["input1", "input2"],
            outputs=["product"],
            description="Element-wise multiplication",
            reference="nntrainer built-in",
        ),
        "activation": LayerInfo(
            name="activation",
            type="builtin",
            file_path=None,
            header_file="<layer.h>",
            usage_string='createLayer("activation")',
            properties={
                "name": "Layer name",
                "activation": "Activation type (relu, swish, gelu, etc.)",
            },
            inputs=["input"],
            outputs=["output"],
            description="Non-linear activation function",
            reference="nntrainer built-in",
        ),
        "layer_norm": LayerInfo(
            name="layer_norm",
            type="builtin",
            file_path=None,
            header_file="<layer.h>",
            usage_string='createLayer("layer_norm")',
            properties={
                "name": "Layer name",
                "epsilon": "Normalization epsilon",
            },
            inputs=["input"],
            outputs=["normalized"],
            description="Layer normalization",
            reference="nntrainer built-in",
        ),
    }

    # Custom layers in CausalLM/layers/
    CUSTOM_LAYERS = {
        "rms_norm": LayerInfo(
            name="rms_norm",
            type="custom",
            file_path="Applications/CausalLM/layers/rms_norm.cpp",
            header_file="<rms_norm.h>",
            usage_string='createLayer("rms_norm")',
            properties={
                "name": "Layer name",
                "epsilon": "Epsilon for numerical stability (default: 1e-6)",
                "skip_prefill": "Skip during prefill (true/false)",
            },
            inputs=["input"],
            outputs=["normalized"],
            description="RMSNorm: Root Mean Square normalization (replaces LayerNorm)",
            reference="Applications/CausalLM/layers/rms_norm.cpp",
        ),
        "reshaped_rms_norm": LayerInfo(
            name="reshaped_rms_norm",
            type="custom",
            file_path="Applications/CausalLM/layers/reshaped_rms_norm.cpp",
            header_file="<reshaped_rms_norm.h>",
            usage_string='createLayer("reshaped_rms_norm")',
            properties={
                "name": "Layer name",
                "epsilon": "Epsilon (default: 1e-6)",
                "packed": "Packed format (true/false)",
                "feature_size": "Feature dimension for reshaping",
            },
            inputs=["input"],
            outputs=["normalized"],
            description="Reshaped RMSNorm: For Q/K normalization before attention",
            reference="Applications/CausalLM/layers/reshaped_rms_norm.cpp",
        ),
        "mha_core": LayerInfo(
            name="mha_core",
            type="custom",
            file_path="Applications/CausalLM/layers/mha_core.cpp",
            header_file="<mha_core.h>",
            usage_string='createLayer("mha_core")',
            properties={
                "name": "Layer name",
                "num_heads": "Number of query heads",
                "num_heads_kv": "Number of KV heads (for GQA)",
                "rope_theta": "RoPE theta parameter",
                "is_causal": "Apply causal masking (true/false)",
                "sliding_window": "Sliding window size (optional)",
                "max_timestep": "Maximum sequence length",
                "use_rope": "Use Rotary Position Embeddings",
                "use_sink": "Use sink tokens",
                "attn_logit_softcapping": "Logit softcapping value",
            },
            inputs=["query", "key", "value", "cache_k", "cache_v"],
            outputs=["attention_output"],
            description="Multi-head attention with RoPE and KV cache",
            reference="Applications/CausalLM/layers/mha_core.cpp",
        ),
        "swiglu": LayerInfo(
            name="swiglu",
            type="custom",
            file_path="Applications/CausalLM/layers/swiglu.cpp",
            header_file="<swiglu.h>",
            usage_string='createLayer("swiglu")',
            properties={
                "name": "Layer name",
                "skip_prefill": "Skip during prefill (true/false)",
            },
            inputs=["gate", "value"],
            outputs=["output"],
            description="SwiGLU activation: output = gate * swish(value)",
            reference="Applications/CausalLM/layers/swiglu.cpp",
        ),
        "lm_head": LayerInfo(
            name="lm_head",
            type="custom",
            file_path="Applications/CausalLM/layers/lm_head.cpp",
            header_file="<lm_head.h>",
            usage_string='createLayer("lm_head")',
            properties={
                "name": "Layer name",
                "vocab_size": "Vocabulary size",
                "tie_word_embedding": "Tie with word embedding (true/false)",
            },
            inputs=["hidden_states"],
            outputs=["logits"],
            description="Final vocabulary projection (LM Head)",
            reference="Applications/CausalLM/layers/lm_head.cpp",
        ),
        "embedding_pooling": LayerInfo(
            name="embedding_pooling",
            type="custom",
            file_path="Applications/CausalLM/layers/embedding_pooling_layer.cpp",
            header_file="<embedding_pooling_layer.h>",
            usage_string='createLayer("embedding_pooling")',
            properties={
                "name": "Layer name",
                "pooling_type": "Pooling strategy (mean, max, etc.)",
            },
            inputs=["embeddings"],
            outputs=["pooled"],
            description="Embedding pooling for sequence representation",
            reference="Applications/CausalLM/layers/embedding_pooling_layer.cpp",
        ),
        "kv_cache_placeholders": LayerInfo(
            name="kv_cache_placeholders",
            type="custom",
            file_path="Applications/CausalLM/layers/kv_cache_manager.cpp",
            header_file="<kv_cache_manager.h>",
            usage_string='createKVCachePlaceholders(layer_id, num_heads_kv)',
            properties={
                "layer_id": "Layer index",
                "num_heads_kv": "Number of KV heads",
            },
            inputs=[],
            outputs=["cache_k", "cache_v"],
            description="KV cache placeholder for incremental generation",
            reference="Applications/CausalLM/layers/kv_cache_manager.cpp",
        ),
        "causal_conv1d": LayerInfo(
            name="causal_conv1d",
            type="custom",
            file_path="Applications/CausalLM/layers/causal_conv1d_layer.cpp",
            header_file="<causal_conv1d_layer.h>",
            usage_string='createLayer("causal_conv1d")',
            properties={
                "name": "Layer name",
                "kernel_size": "Convolution kernel size",
                "dilation": "Dilation factor",
            },
            inputs=["input"],
            outputs=["output"],
            description="Causal 1D convolution (for selective state space models)",
            reference="Applications/CausalLM/layers/causal_conv1d_layer.cpp",
        ),
    }

    def __init__(self):
        self.registry: Dict[str, LayerInfo] = {}
        self.generated_layers: Dict[str, LayerInfo] = {}

        # Initialize with known layers
        self.registry.update(self.BUILTIN_LAYERS)
        self.registry.update(self.CUSTOM_LAYERS)

    @staticmethod
    def _normalise_name(value: str) -> str:
        """Normalise HF operation, C++ class, and filename spellings."""
        return re.sub(r"[^a-z0-9]", "", value.lower().replace("layer", ""))

    def scan_causallm_layers(self, nntrainer_root: str) -> List[str]:
        """Discover real CausalLM layer files so this registry cannot go stale.

        The hand-curated catalog remains useful for documented properties.
        Scanning is additive: it only supplies filename/type candidates that
        are missing from that catalog and never replaces an existing entry.
        """
        layers_dir = Path(nntrainer_root) / "Applications" / "CausalLM" / "layers"
        if not layers_dir.is_dir():
            return []

        discovered: List[str] = []
        for header in sorted(layers_dir.glob("*.h")):
            source = header.with_suffix(".cpp")
            text = header.read_text(encoding="utf-8", errors="ignore")
            if source.exists():
                text += "\n" + source.read_text(encoding="utf-8", errors="ignore")
            types = set(re.findall(r'(?:type|type_str)\s*=\s*"([A-Za-z0-9_]+)"', text))
            names = types or {header.stem.removesuffix("_layer")}
            for name in names:
                if name not in self.registry:
                    self.registry[name] = LayerInfo(
                        name=name,
                        type="custom",
                        file_path=str(source) if source.exists() else None,
                        header_file=f"<{header.name}>",
                        usage_string=f'createLayer("{name}")',
                        properties={},
                        inputs=[],
                        outputs=[],
                        description=f"Discovered from {header.name}",
                        reference=str(header),
                    )
                discovered.append(name)
        return sorted(set(discovered))

    def fuzzy_find_layers(self, query: str, limit: int = 5) -> List[Tuple[LayerInfo, float]]:
        """Rank existing layers by a normalised filename/type match.

        A ranking is deliberately not an automatic substitution. Generation
        must still verify properties and input/output contracts before using a
        candidate, preventing a plausible filename match from changing model
        semantics.
        """
        needle = self._normalise_name(query)
        ranked = []
        for name, layer in self.registry.items():
            candidate = self._normalise_name(name)
            score = SequenceMatcher(None, needle, candidate).ratio()
            if needle and (needle in candidate or candidate in needle):
                score = max(score, 0.85)
            if score >= 0.35:
                ranked.append((layer, score))
        return sorted(ranked, key=lambda item: (-item[1], item[0].name))[:limit]

    def get_layer(self, name: str) -> Optional[LayerInfo]:
        """Get layer info by name."""
        return self.registry.get(name) or self.generated_layers.get(name)

    def list_available_layers(self) -> List[str]:
        """List all available layer names."""
        return sorted(self.registry.keys())

    def list_builtin_layers(self) -> List[str]:
        """List built-in nntrainer layers."""
        return sorted([k for k, v in self.registry.items() if v.type == "builtin"])

    def list_custom_layers(self) -> List[str]:
        """List custom CausalLM layers."""
        return sorted([k for k, v in self.registry.items() if v.type == "custom"])

    def list_generated_layers(self) -> List[str]:
        """List AI-generated layers."""
        return sorted(self.generated_layers.keys())

    def register_generated_layer(self, layer_info: LayerInfo) -> None:
        """Register a newly generated layer."""
        self.generated_layers[layer_info.name] = layer_info
        self.registry[layer_info.name] = layer_info

    def find_missing_layers(self, required_layers: List[str]) -> List[str]:
        """Find which required layers are not available."""
        available = set(self.registry.keys())
        required = set(required_layers)
        return sorted(required - available)

    def get_layer_usage_code(self, layer_name: str) -> Optional[str]:
        """Get template C++ code for using a layer."""
        layer = self.get_layer(layer_name)
        if not layer:
            return None

        code = f'''
  // {layer.description}
  LayerHandle {layer_name}_layer = createLayer(
    "{layer_name}",
    {{'''

        for prop_name, prop_desc in layer.properties.items():
            code += f'\n      withKey("{prop_name}", "value"),  // {prop_desc}'

        code += '\n    }\n  );\n'

        if len(layer.inputs) == 1:
            code += f'  Tensor {layer_name}_out = {layer_name}_layer({layer.inputs[0]});\n'
        elif len(layer.inputs) > 1:
            inputs_str = ", ".join(layer.inputs)
            code += f'  Tensor {layer_name}_out = {layer_name}_layer({{{inputs_str}}});\n'

        return code

    def to_dict(self) -> Dict:
        """Export registry as dictionary."""
        return {
            "builtin_layers": {
                name: self._layer_to_dict(info)
                for name, info in self.registry.items()
                if info.type == "builtin"
            },
            "custom_layers": {
                name: self._layer_to_dict(info)
                for name, info in self.registry.items()
                if info.type == "custom"
            },
            "generated_layers": {
                name: self._layer_to_dict(info)
                for name, info in self.generated_layers.items()
            },
        }

    @staticmethod
    def _layer_to_dict(layer: LayerInfo) -> Dict:
        """Convert layer info to dictionary."""
        return {
            "name": layer.name,
            "type": layer.type,
            "file_path": layer.file_path,
            "header_file": layer.header_file,
            "usage_string": layer.usage_string,
            "properties": layer.properties,
            "inputs": layer.inputs,
            "outputs": layer.outputs,
            "description": layer.description,
            "reference": layer.reference,
        }

    def print_summary(self):
        """Print a summary of the registry."""
        print("\n" + "=" * 70)
        print("LAYER REGISTRY SUMMARY")
        print("=" * 70)

        print(f"\n📦 Built-in Layers ({len(self.list_builtin_layers())}):")
        for name in self.list_builtin_layers():
            layer = self.get_layer(name)
            print(f"  • {name:<20} - {layer.description}")

        print(f"\n🔧 Custom Layers ({len(self.list_custom_layers())}):")
        for name in self.list_custom_layers():
            layer = self.get_layer(name)
            print(f"  • {name:<20} - {layer.description}")

        if self.list_generated_layers():
            print(f"\n✨ Generated Layers ({len(self.list_generated_layers())}):")
            for name in self.list_generated_layers():
                layer = self.get_layer(name)
                print(f"  • {name:<20} - {layer.description}")

        print("\n" + "=" * 70)


# Global registry instance
_global_registry = None


def get_layer_registry() -> LayerRegistry:
    """Get the global layer registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = LayerRegistry()
    return _global_registry


if __name__ == "__main__":
    registry = LayerRegistry()
    registry.print_summary()

    # Example: Check missing layers
    print("\n" + "-" * 70)
    print("EXAMPLE: Checking missing layers")
    print("-" * 70)

    required = ["embedding", "rms_norm", "mha_core", "swiglu", "fully_connected", "custom_layer"]
    missing = registry.find_missing_layers(required)

    print(f"Required layers: {required}")
    print(f"Missing layers: {missing}")

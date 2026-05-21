"""Plugin registry for custom PyTorch → NNTrainer layer mappings.

Allows users to register their own custom layer converters so the
TorchFXConverter pipeline can handle arbitrary PyTorch modules and
map them to NNTrainer LayerPluggable custom layers.

Usage:
    from plugin_registry import PluginRegistry, CustomLayerSpec

    # Define a custom layer spec
    spec = CustomLayerSpec(
        nntrainer_type="custom_pow",
        property_mapper=lambda module: {"exponent": module.exponent},
    )

    # Register by PyTorch class
    registry = PluginRegistry()
    registry.register(MyPowLayer, spec)

    # Or load from a YAML/JSON config file
    registry = PluginRegistry.from_config("custom_layers.yaml")
"""

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from nntrainer_layers import NNTrainerLayerDef


@dataclass
class CustomLayerSpec:
    """Specification for mapping a custom PyTorch module to NNTrainer.

    Attributes:
        nntrainer_type: Target NNTrainer layer type string (e.g. "custom_pow").
        property_mapper: Optional callable (module) -> dict of NNTrainer
            properties. If None, no extra properties are emitted.
        weight_keys: Optional dict mapping NNTrainer weight names to
            PyTorch state_dict key templates. Use "{name}" as placeholder
            for the module's full name in the model.
            E.g. {"weight": "{name}.weight", "bias": "{name}.bias"}
        has_weight: Whether the layer has trainable weights.
        has_bias: Whether the layer has a bias parameter.
        transpose_weight: Whether to transpose weight matrix.
        supports_training: Whether the custom layer supports backprop.
        description: Human-readable description of the custom layer.
        pluggable_so: Optional path to the compiled .so plugin library.
    """
    nntrainer_type: str
    property_mapper: Optional[Callable] = None
    weight_keys: dict = field(default_factory=dict)
    has_weight: bool = False
    has_bias: bool = False
    transpose_weight: bool = False
    supports_training: bool = True
    description: str = ""
    pluggable_so: str = ""


class PluginRegistry:
    """Registry for custom PyTorch module → NNTrainer layer mappings.

    Maintains a list of (matcher, spec) pairs. Matchers can be:
    - A PyTorch module class (isinstance check)
    - A string matching the class name (for classes not importable)
    - A callable predicate (module) -> bool
    """

    def __init__(self):
        self._entries = []  # List of (matcher, CustomLayerSpec)

    def register(self, matcher, spec):
        """Register a custom layer mapping.

        Args:
            matcher: One of:
                - A class (uses isinstance check)
                - A string (matches type(module).__name__)
                - A callable (module) -> bool
            spec: CustomLayerSpec defining the NNTrainer mapping.
        """
        if not isinstance(spec, CustomLayerSpec):
            raise TypeError(f"spec must be CustomLayerSpec, got {type(spec)}")
        self._entries.append((matcher, spec))

    def register_simple(self, matcher, nntrainer_type, **kwargs):
        """Convenience method to register with inline spec creation.

        Args:
            matcher: Class, string, or callable matcher.
            nntrainer_type: NNTrainer layer type string.
            **kwargs: Additional CustomLayerSpec fields.
        """
        spec = CustomLayerSpec(nntrainer_type=nntrainer_type, **kwargs)
        self.register(matcher, spec)

    def lookup(self, module):
        """Find a matching CustomLayerSpec for a PyTorch module.

        Args:
            module: A PyTorch nn.Module instance.

        Returns:
            CustomLayerSpec or None if no match found.
        """
        module_cls = type(module)
        module_cls_name = module_cls.__name__

        for matcher, spec in self._entries:
            if isinstance(matcher, type):
                if isinstance(module, matcher):
                    return spec
            elif isinstance(matcher, str):
                if module_cls_name == matcher:
                    return spec
            elif callable(matcher):
                if matcher(module):
                    return spec
        return None

    def map_module(self, module, module_name, input_names):
        """Map a custom module to NNTrainerLayerDef using registered specs.

        Args:
            module: PyTorch nn.Module instance.
            module_name: Full dotted name in model hierarchy.
            input_names: List of input layer names from FX graph.

        Returns:
            NNTrainerLayerDef or None if no match.
        """
        spec = self.lookup(module)
        if spec is None:
            return None

        # Build properties
        props = {}
        if spec.property_mapper is not None:
            props = spec.property_mapper(module)

        # Build weight keys
        weight_key = ""
        bias_key = ""
        if spec.weight_keys:
            wk = spec.weight_keys.get("weight", "")
            if wk:
                weight_key = wk.replace("{name}", module_name)
            bk = spec.weight_keys.get("bias", "")
            if bk:
                bias_key = bk.replace("{name}", module_name)

        sanitized_name = module_name.replace(".", "_")

        return NNTrainerLayerDef(
            layer_type=spec.nntrainer_type,
            name=sanitized_name,
            properties=props,
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=type(module).__name__,
            has_weight=spec.has_weight,
            has_bias=spec.has_bias,
            weight_hf_key=weight_key,
            bias_hf_key=bias_key,
            transpose_weight=spec.transpose_weight,
        )

    @property
    def registered_types(self):
        """Return list of registered NNTrainer type names."""
        return [spec.nntrainer_type for _, spec in self._entries]

    def __len__(self):
        return len(self._entries)

    def __bool__(self):
        return len(self._entries) > 0

    @classmethod
    def from_config(cls, config_path):
        """Load plugin registry from a JSON or YAML config file.

        Config format (JSON):
        {
          "custom_layers": [
            {
              "match_class_name": "MyCustomLayer",
              "nntrainer_type": "custom_pow",
              "properties": {"exponent": 2},
              "has_weight": false,
              "description": "Custom power layer"
            },
            {
              "match_class_name": "MyNormLayer",
              "nntrainer_type": "my_norm",
              "has_weight": true,
              "has_bias": true,
              "weight_keys": {
                "weight": "{name}.weight",
                "bias": "{name}.bias"
              },
              "pluggable_so": "libmy_norm_layer.so"
            }
          ]
        }
        """
        registry = cls()

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Plugin config not found: {config_path}")

        with open(config_path) as f:
            if config_path.endswith((".yaml", ".yml")):
                try:
                    import yaml
                    config = yaml.safe_load(f)
                except ImportError:
                    raise ImportError(
                        "PyYAML required for YAML config files. "
                        "Install with: pip install pyyaml")
            else:
                config = json.load(f)

        for entry in config.get("custom_layers", []):
            matcher = entry.get("match_class_name", "")
            if not matcher:
                continue

            # Build static property mapper from config
            static_props = entry.get("properties", {})
            prop_mapper = (lambda p: lambda _mod: dict(p))(static_props) \
                if static_props else None

            spec = CustomLayerSpec(
                nntrainer_type=entry["nntrainer_type"],
                property_mapper=prop_mapper,
                weight_keys=entry.get("weight_keys", {}),
                has_weight=entry.get("has_weight", False),
                has_bias=entry.get("has_bias", False),
                transpose_weight=entry.get("transpose_weight", False),
                supports_training=entry.get("supports_training", True),
                description=entry.get("description", ""),
                pluggable_so=entry.get("pluggable_so", ""),
            )
            registry.register(matcher, spec)

        return registry


# Global plugin registry instance (can be populated before conversion)
_global_registry = PluginRegistry()


def get_global_registry():
    """Get the global plugin registry."""
    return _global_registry


def register_custom_layer(matcher, nntrainer_type, **kwargs):
    """Register a custom layer in the global registry.

    Convenience function for simple use cases:
        register_custom_layer(MyPowLayer, "custom_pow",
                              property_mapper=lambda m: {"exponent": m.exp})
    """
    _global_registry.register_simple(matcher, nntrainer_type, **kwargs)

"""Node mapper: Maps FX graph nodes to NNTrainer layer definitions.

This module takes the traced FX graph and converts each node into
an NNTrainerLayerDef. It dispatches to specialized mappers:
  1. module_mapper  - call_module nodes (nn.Module leaf nodes)
  2. function_mapper - call_function nodes (torch.*/operator.* functions)
  3. method_mapper  - call_method nodes (Tensor.* methods)

The mapper is architecture-agnostic. It maps individual nodes based
on their type, not based on what model they came from. Pattern detection
(attention blocks, FFN blocks, etc.) is handled separately in pattern_detector.py.
"""

from typing import Optional

from nntrainer_layers import NNTrainerLayerDef, OP_UNSUPPORTED
from module_mapper import map_module_node, MULTI_OUTPUT_LAYER_TYPES
from function_mapper import map_function_node
from method_mapper import map_method_node
from mapper_helpers import get_input_node_names, sanitize_name

# Backward-compatible aliases
_get_input_node_names = get_input_node_names
_sanitize_name = sanitize_name


class NodeMapper:
    """Maps FX graph nodes to NNTrainer layer definitions.

    Usage:
        mapper = NodeMapper(model, graph, config)
        layers = mapper.map_all()
    """

    def __init__(self, model, graph, model_config=None):
        """
        Args:
            model: The traced nn.Module
            graph: The FX graph from tracing
            model_config: HuggingFace model config (optional, for extracting params)
        """
        self.model = model
        self.graph = graph
        self.config = model_config
        self._modules = dict(model.named_modules())
        self._node_to_layer = {}  # node.name -> NNTrainerLayerDef

    def map_all(self):
        """Map all graph nodes to NNTrainer layer definitions.

        Returns a list of NNTrainerLayerDef objects in graph order.
        """
        layers = []
        seen_names = {}  # layer_name -> count (for deduplication)

        for node in self.graph.nodes:
            layer_def = self._map_node(node)
            if layer_def is not None:
                # Store original FX node name for shape metadata lookup
                layer_def.fx_node_name = node.name

                # Deduplicate layer names: sanitize_name("a.b") and "a_b"
                # can collide. Append _1, _2, etc. to make names unique.
                if layer_def.name in seen_names:
                    seen_names[layer_def.name] += 1
                    layer_def.name = f"{layer_def.name}_{seen_names[layer_def.name]}"
                else:
                    seen_names[layer_def.name] = 0

                layers.append(layer_def)
                self._node_to_layer[node.name] = layer_def

        # Remap input references: FX node names -> actual layer names.
        # Layer names may differ from FX node names due to scoping
        # (e.g. _make_scoped_name adds module scope prefix) or dedup.
        fx_to_layer = {node_name: ldef.name
                       for node_name, ldef in self._node_to_layer.items()
                       if node_name != ldef.name}
        if fx_to_layer:
            for layer in layers:
                if layer.input_layers:
                    layer.input_layers = [
                        fx_to_layer.get(inp, inp) for inp in layer.input_layers
                    ]

        return layers

    def get_unknown_layers(self, layers=None):
        """Return layers that could not be mapped to NNTrainer types.

        Returns:
            List of NNTrainerLayerDef with unknown layer_type.
        """
        if layers is None:
            layers = self.map_all()
        return [l for l in layers if l.layer_type.startswith("unknown")
                or l.layer_type == OP_UNSUPPORTED]

    def get_unknown_module_types(self, layers=None):
        """Return set of module class names that could not be mapped.

        These modules should be excluded from LEAF_MODULES on re-trace,
        allowing the tracer to decompose them into tensor ops.

        Returns:
            Set of class name strings (e.g. {"CustomAttention", "MyNorm"}).
        """
        unknowns = self.get_unknown_layers(layers)
        module_types = set()
        for layer in unknowns:
            if layer.layer_type.startswith("unknown("):
                # Extract module type from "unknown(SomeModule)"
                module_types.add(layer.layer_type[8:-1])
        return module_types

    def _map_node(self, node) -> Optional[NNTrainerLayerDef]:
        """Map a single FX node to an NNTrainer layer definition."""
        if node.op == "call_module":
            return map_module_node(node, self._modules, self._node_to_layer)
        elif node.op == "call_function":
            return map_function_node(node, self._node_to_layer)
        elif node.op == "call_method":
            return map_method_node(node)
        elif node.op in ("placeholder", "output", "get_attr"):
            return None
        return None

    # Kept for backward compatibility with external code that may reference it
    _MULTI_OUTPUT_LAYER_TYPES = MULTI_OUTPUT_LAYER_TYPES

    @staticmethod
    def _extract_clamp_params(node, op_name, props):
        """Extract min/max parameters from clamp/clip FX node.

        Kept as static method for backward compatibility.
        """
        from function_mapper import _extract_clamp_params
        _extract_clamp_params(node, op_name, props)

    def _make_activation(self, module_name, module_type, act_type, input_names):
        """Helper to create an activation layer def.

        Kept for backward compatibility.
        """
        from module_mapper import _make_activation
        return _make_activation(module_name, module_type, act_type, input_names)

"""Method mapper: Maps call_method FX nodes to NNTrainer layer definitions.

Handles Tensor.* method calls such as add, mul, view, reshape, permute,
transpose, softmax, etc.
"""

from nntrainer_layers import (
    NNTrainerLayerDef,
    LAYER_ACTIVATION, LAYER_CLAMP,
    OP_NOOP, OP_RESHAPE, OP_UNSUPPORTED,
)
from op_registry import (
    METHOD_SIMPLE_OPS, METHOD_ACTIVATION_OPS, METHOD_SHAPE_OPS,
    METHOD_RESHAPE_NAMES, METHOD_DECOMPOSE_OPS, METHOD_CLAMP_NAMES,
    METHOD_NOOP_NAMES, METHOD_SPLIT_NAMES,
)
from mapper_helpers import (
    get_input_node_names, sanitize_name, make_scoped_name,
    extract_clamp_params,
)

# Backward-compatible aliases
_get_input_node_names = get_input_node_names
_sanitize_name = sanitize_name
_make_scoped_name = make_scoped_name
_extract_clamp_params = extract_clamp_params


def map_method_node(node):
    """Map a call_method node to NNTrainerLayerDef.

    Args:
        node: FX graph node with op == "call_method"

    Returns:
        NNTrainerLayerDef or None
    """
    method_name = node.target  # string
    scope = node.meta.get("scope", "")
    input_names = get_input_node_names(node)

    # === Simple ops (table lookup) ===
    layer_type = METHOD_SIMPLE_OPS.get(method_name)
    if layer_type is not None:
        return NNTrainerLayerDef(
            layer_type=layer_type,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope if layer_type.endswith("_op") else "",
        )

    # === Activation methods (table lookup) ===
    act_type = METHOD_ACTIVATION_OPS.get(method_name)
    if act_type is not None:
        return NNTrainerLayerDef(
            layer_type=LAYER_ACTIVATION,
            name=sanitize_name(scope) if scope else node.name,
            properties={"activation": act_type},
            input_layers=input_names,
            hf_module_name=scope,
            hf_module_type="Tensor",
        )

    # === Shape ops (view, reshape, permute, transpose) ===
    shape_type = METHOD_SHAPE_OPS.get(method_name)
    if shape_type is not None:
        return NNTrainerLayerDef(
            layer_type=shape_type,
            name=node.name,
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Reshape names (unsqueeze, squeeze, repeat, expand_as) ===
    if method_name in METHOD_RESHAPE_NAMES:
        return NNTrainerLayerDef(
            layer_type=OP_RESHAPE,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Decompose ops ===
    decompose_props = METHOD_DECOMPOSE_OPS.get(method_name)
    if decompose_props is not None:
        return NNTrainerLayerDef(
            layer_type=OP_UNSUPPORTED,
            name=make_scoped_name(scope, node),
            properties=dict(decompose_props),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Clamp ===
    if method_name in METHOD_CLAMP_NAMES:
        props = {"original_op": method_name}
        extract_clamp_params(node, method_name, props)
        return NNTrainerLayerDef(
            layer_type=LAYER_CLAMP,
            name=make_scoped_name(scope, node),
            properties=props,
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Split/chunk ===
    if method_name in METHOD_SPLIT_NAMES:
        return NNTrainerLayerDef(
            layer_type="split",
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === No-ops ===
    if method_name in METHOD_NOOP_NAMES:
        return NNTrainerLayerDef(
            layer_type=OP_NOOP,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Unknown method ===
    print(f"  [WARNING] Unmapped method: {method_name} in scope={scope}")
    return NNTrainerLayerDef(
        layer_type=f"unknown_method({method_name})",
        name=make_scoped_name(scope, node),
        input_layers=input_names,
        hf_module_name=scope,
    )

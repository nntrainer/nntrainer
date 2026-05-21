"""Shared helper functions for FX node mappers.

Centralizes utilities that were previously duplicated across
function_mapper.py, method_mapper.py, module_mapper.py, and node_mapper.py.
"""

import math


def get_input_node_names(node):
    """Get names of input FX nodes from a node's args.

    Args:
        node: FX graph node.

    Returns:
        List of input node name strings.
    """
    names = []
    for arg in node.args:
        if hasattr(arg, 'name'):
            names.append(arg.name)
    return names


def sanitize_name(name: str) -> str:
    """Convert HF module path to NNTrainer-compatible layer name.

    e.g. 'model.layers.0.self_attn.q_proj' -> 'model_layers_0_self_attn_q_proj'
    """
    return name.replace(".", "_")


def make_scoped_name(scope, node, suffix=""):
    """Build a scoped layer name from HF module scope and FX node name.

    Args:
        scope: HF module scope string (e.g. "model.layers.0.self_attn").
        node: FX graph node.
        suffix: Optional suffix to append.

    Returns:
        str: Scoped layer name.
    """
    if scope:
        name = f"{sanitize_name(scope)}_{node.name}"
    else:
        name = node.name
    if suffix:
        name = f"{name}_{suffix}"
    return name


def extract_clamp_params(node, op_name, props):
    """Extract min/max parameters from clamp/clip FX node.

    Populates props dict with "min" and "max" string values.

    Args:
        node: FX graph node for clamp/clip/clamp_min/clamp_max.
        op_name: Operation name string.
        props: Dict to populate with min/max values.
    """
    args = node.args
    kwargs = node.kwargs or {}

    if op_name == "clamp_min":
        props["min"] = str(args[1] if len(args) > 1 else kwargs.get("min", 0))
        props["max"] = str(math.inf)
    elif op_name == "clamp_max":
        props["min"] = str(-math.inf)
        props["max"] = str(args[1] if len(args) > 1 else kwargs.get("max", 0))
    else:
        if len(args) > 1:
            props["min"] = str(args[1])
        elif "min" in kwargs and kwargs["min"] is not None:
            props["min"] = str(kwargs["min"])
        else:
            props["min"] = str(-math.inf)

        if len(args) > 2:
            props["max"] = str(args[2])
        elif "max" in kwargs and kwargs["max"] is not None:
            props["max"] = str(kwargs["max"])
        else:
            props["max"] = str(math.inf)

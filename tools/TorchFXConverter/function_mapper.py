"""Function mapper: Maps call_function FX nodes to NNTrainer layer definitions.

Handles torch.* functions (torch.add, torch.matmul, etc.), operator.*
(operator.add, operator.mul, etc.), and F.* functional calls.
"""

import operator

import torch
import torch.nn.functional as F

from nntrainer_layers import (
    NNTrainerLayerDef,
    LAYER_ACTIVATION, LAYER_DROPOUT, LAYER_CONCAT, LAYER_MATMUL,
    LAYER_SPLIT, LAYER_CLAMP, LAYER_IDENTITY, LAYER_GATHER, LAYER_POOLING2D,
    LAYER_UPSAMPLE2D, LAYER_L2NORM, LAYER_FC,
    ACT_SWISH,
    OP_SDPA, OP_NOOP, OP_RESHAPE, OP_PERMUTE, OP_TRANSPOSE, OP_UNSUPPORTED,
)
from op_registry import (
    FUNCTION_SIMPLE_OPS, FUNCTION_NAME_SIMPLE_OPS,
    FUNCTION_NOOP_NAMES, FUNCTION_RESHAPE_NAMES, FUNCTION_DECOMPOSE_OPS,
    FUNCTION_PERMUTE_NAMES, FUNCTION_TRANSPOSE_NAMES,
    FUNCTION_ACTIVATION_OPS, FUNCTION_ACTIVATION_NAMES,
    FUNCTION_IDENTITY_OPS, FUNCTION_CLAMP_NAMES,
    FUNCTION_POOLING_NAMES, FUNCTION_INTERPOLATE_NAMES,
    FUNCTION_NORMALIZE_NAMES,
    FUNCTION_LOSS_OPS, FUNCTION_LOSS_NAMES,
    MULTI_OUTPUT_LAYER_TYPES,
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

# Layer types that produce tuple outputs (for operator.getitem handling)
_MULTI_OUTPUT_LAYER_TYPES = MULTI_OUTPUT_LAYER_TYPES


def map_function_node(node, node_to_layer):
    """Map a call_function node to NNTrainerLayerDef.

    Args:
        node: FX graph node with op == "call_function"
        node_to_layer: dict mapping node.name -> NNTrainerLayerDef

    Returns:
        NNTrainerLayerDef or None
    """
    func = node.target
    scope = node.meta.get("scope", "")
    input_names = get_input_node_names(node)
    func_name = getattr(func, "__name__", str(func))

    # === No-ops ===
    if func_name in FUNCTION_NOOP_NAMES:
        return NNTrainerLayerDef(
            layer_type=OP_NOOP,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Simple ops (table lookup by callable identity) ===
    layer_type = FUNCTION_SIMPLE_OPS.get(func)
    if layer_type is not None:
        return NNTrainerLayerDef(
            layer_type=layer_type,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope if layer_type.endswith("_op") else "",
        )

    # === Simple ops (table lookup by name) ===
    layer_type = FUNCTION_NAME_SIMPLE_OPS.get(func_name)
    if layer_type is not None:
        return NNTrainerLayerDef(
            layer_type=layer_type,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Permute / transpose by name (torch.permute, torch.swapaxes, etc.) ===
    if func_name in FUNCTION_PERMUTE_NAMES:
        return NNTrainerLayerDef(
            layer_type=OP_PERMUTE,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    if func_name in FUNCTION_TRANSPOSE_NAMES:
        return NNTrainerLayerDef(
            layer_type=OP_TRANSPOSE,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Reshape by name ===
    if func_name in FUNCTION_RESHAPE_NAMES:
        return NNTrainerLayerDef(
            layer_type=OP_RESHAPE,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Decompose ops ===
    decompose_props = FUNCTION_DECOMPOSE_OPS.get(func_name)
    if decompose_props is not None:
        # Use torch.abs identity check for abs specifically
        if func_name == "abs" and func is not torch.abs:
            pass  # fall through if name matches but not the expected func
        else:
            return NNTrainerLayerDef(
                layer_type=OP_UNSUPPORTED,
                name=make_scoped_name(scope, node),
                properties=dict(decompose_props),
                input_layers=input_names,
                hf_module_name=scope,
            )

    # torch.abs by identity (in case func_name didn't match above)
    if func is torch.abs:
        return NNTrainerLayerDef(
            layer_type=OP_UNSUPPORTED,
            name=make_scoped_name(scope, node),
            properties={"original_op": "abs", "decompose_to": "sqrt(pow(x, 2))"},
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Clamp (registry-based) ===
    if func_name in FUNCTION_CLAMP_NAMES:
        props = {"original_op": func_name}
        extract_clamp_params(node, func_name, props)
        return NNTrainerLayerDef(
            layer_type=LAYER_CLAMP,
            name=make_scoped_name(scope, node),
            properties=props,
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Activation functions (table lookup) ===
    act_type = FUNCTION_ACTIVATION_OPS.get(func)
    if act_type is not None:
        return NNTrainerLayerDef(
            layer_type=LAYER_ACTIVATION,
            name=make_scoped_name(scope, node, act_type),
            properties={"activation": act_type},
            input_layers=input_names,
        )

    # Activation by name (torch.tanh / F.tanh)
    act_type = FUNCTION_ACTIVATION_NAMES.get(func_name)
    if act_type is not None:
        return NNTrainerLayerDef(
            layer_type=LAYER_ACTIVATION,
            name=make_scoped_name(scope, node, func_name),
            properties={"activation": act_type},
            input_layers=input_names,
        )

    # === Loss functions (table lookup by callable) ===
    loss_type = FUNCTION_LOSS_OPS.get(func)
    if loss_type is not None:
        return NNTrainerLayerDef(
            layer_type=loss_type,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Loss functions (by name) ===
    loss_type = FUNCTION_LOSS_NAMES.get(func_name)
    if loss_type is not None:
        return NNTrainerLayerDef(
            layer_type=loss_type,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Identity-based ops (F.dropout, F.pad, etc.) ===
    identity_type = FUNCTION_IDENTITY_OPS.get(func)
    if identity_type is not None:
        return NNTrainerLayerDef(
            layer_type=identity_type,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope if identity_type == OP_NOOP else "",
        )

    # === torch.cat ===
    if func is torch.cat:
        return _map_cat(node, scope, input_names)

    # === torch.stack ===
    if func is torch.stack:
        return _map_stack(node, scope, input_names)

    # === torch.gather ===
    if func is torch.gather:
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim', 0)
        return NNTrainerLayerDef(
            layer_type=LAYER_GATHER,
            name=make_scoped_name(scope, node),
            properties={"axis": dim},
            input_layers=input_names,
        )

    # === torch.einsum ===
    if func is torch.einsum:
        equation = node.args[0] if len(node.args) > 0 else ""
        return NNTrainerLayerDef(
            layer_type=LAYER_MATMUL,
            name=make_scoped_name(scope, node),
            properties={"equation": equation},
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === F.scaled_dot_product_attention ===
    if func_name == "scaled_dot_product_attention":
        return NNTrainerLayerDef(
            layer_type=OP_SDPA,
            name=make_scoped_name(scope, node, "sdpa"),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === Pooling functions (F.max_pool2d, F.adaptive_avg_pool2d, etc.) ===
    if func_name in FUNCTION_POOLING_NAMES:
        return _map_pooling(node, scope, func_name, input_names)

    # === F.interpolate -> upsample2d ===
    if func_name in FUNCTION_INTERPOLATE_NAMES or func is F.interpolate:
        return _map_interpolate(node, scope, input_names)

    # === F.normalize -> preprocess_l2norm ===
    if func_name in FUNCTION_NORMALIZE_NAMES or func is F.normalize:
        return _map_normalize(node, scope, input_names)

    # === F.linear (functional linear / fully connected) ===
    if func_name == "linear":
        return NNTrainerLayerDef(
            layer_type=LAYER_FC,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === torch.chunk / torch.split ===
    if func_name in ("chunk", "split"):
        return NNTrainerLayerDef(
            layer_type=LAYER_SPLIT,
            name=make_scoped_name(scope, node),
            input_layers=input_names,
            hf_module_name=scope,
        )

    # === operator.getitem (tuple unpacking for multi-output modules) ===
    if func is operator.getitem:
        return _map_getitem(node, scope, node_to_layer)

    # === Unknown function ===
    print(f"  [WARNING] Unmapped function: {func_name} in scope={scope}")
    return NNTrainerLayerDef(
        layer_type=f"unknown_func({func_name})",
        name=make_scoped_name(scope, node),
        input_layers=input_names,
        hf_module_name=scope,
    )


def _map_cat(node, scope, input_names):
    """Map torch.cat to concat layer."""
    cat_inputs = []
    if node.args:
        tensor_list = node.args[0]
        if isinstance(tensor_list, (list, tuple)):
            cat_inputs = [a.name for a in tensor_list if hasattr(a, 'name')]
    if not cat_inputs:
        cat_inputs = input_names
    dim = node.kwargs.get('dim', 0)
    if len(node.args) > 1 and isinstance(node.args[1], int):
        dim = node.args[1]
    props = {}
    if dim != 0:
        # NNTrainer ConcatDimension uses key "axis" with NCHW indices (1-3).
        # Convert negative PyTorch dims: -1 → 3 (width), -2 → 2 (height), etc.
        nn_dim = dim if dim > 0 else (4 + dim)
        props["axis"] = nn_dim
    return NNTrainerLayerDef(
        layer_type=LAYER_CONCAT,
        name=make_scoped_name(scope, node),
        properties=props,
        input_layers=cat_inputs,
    )


def _map_stack(node, scope, input_names):
    """Map torch.stack to concat layer."""
    stack_inputs = []
    if node.args:
        tensor_list = node.args[0]
        if isinstance(tensor_list, (list, tuple)):
            stack_inputs = [a.name for a in tensor_list if hasattr(a, 'name')]
    if not stack_inputs:
        stack_inputs = input_names
    dim = node.kwargs.get('dim', 0)
    if len(node.args) > 1 and isinstance(node.args[1], int):
        dim = node.args[1]
    nn_dim = dim if dim > 0 else (4 + dim)
    return NNTrainerLayerDef(
        layer_type=LAYER_CONCAT,
        name=make_scoped_name(scope, node),
        properties={"axis": nn_dim},
        input_layers=stack_inputs,
    )


def _map_pooling(node, scope, func_name, input_names):
    """Map F.max_pool2d, F.adaptive_avg_pool2d, etc. to pooling2d layer."""
    props = {}

    if "max" in func_name:
        props["pooling"] = "max"
    else:
        props["pooling"] = "average"

    if "adaptive" in func_name:
        props["adaptive"] = True
        # output_size is arg[1]
        out = node.args[1] if len(node.args) > 1 else node.kwargs.get("output_size", (1, 1))
        if isinstance(out, int):
            out = (out, out)
        props["pool_size"] = f"{out[0]},{out[1]}"
    else:
        # kernel_size is arg[1], stride is arg[2], padding is arg[3]
        ks = node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size", 1)
        if isinstance(ks, int):
            ks = (ks, ks)
        props["pool_size"] = f"{ks[0]},{ks[1]}"

        st = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride", ks)
        if isinstance(st, int):
            st = (st, st)
        props["stride"] = f"{st[0]},{st[1]}"

        pd = node.args[3] if len(node.args) > 3 else node.kwargs.get("padding", 0)
        if isinstance(pd, int):
            pd = (pd, pd)
        props["padding"] = f"{pd[0]},{pd[1]}"

    return NNTrainerLayerDef(
        layer_type=LAYER_POOLING2D,
        name=make_scoped_name(scope, node),
        properties=props,
        input_layers=input_names,
        hf_module_name=scope,
    )


def _map_interpolate(node, scope, input_names):
    """Map F.interpolate to NNTrainer upsample2d layer."""
    # F.interpolate(input, size=None, scale_factor=None, mode='nearest', ...)
    mode = node.kwargs.get('mode', 'nearest')
    if len(node.args) > 3:
        mode = node.args[3]
    if mode not in ("nearest", "bilinear"):
        mode = "nearest"

    props = {"upsample": mode}

    scale_factor = node.kwargs.get('scale_factor')
    if len(node.args) > 2 and node.args[2] is not None:
        scale_factor = node.args[2]

    size = node.kwargs.get('size')
    if len(node.args) > 1 and node.args[1] is not None:
        size = node.args[1]

    if scale_factor is not None:
        if isinstance(scale_factor, (tuple, list)):
            props["kernel_size"] = f"{int(scale_factor[0])},{int(scale_factor[1])}"
        else:
            props["kernel_size"] = f"{int(scale_factor)},{int(scale_factor)}"
    elif size is not None:
        if isinstance(size, int):
            props["kernel_size"] = f"{size},{size}"
        elif isinstance(size, (tuple, list)) and len(size) >= 2:
            props["kernel_size"] = f"{size[0]},{size[1]}"

    return NNTrainerLayerDef(
        layer_type=LAYER_UPSAMPLE2D,
        name=make_scoped_name(scope, node),
        properties=props,
        input_layers=input_names,
        hf_module_name=scope,
    )


def _map_normalize(node, scope, input_names):
    """Map F.normalize to NNTrainer preprocess_l2norm layer."""
    # F.normalize(input, p=2.0, dim=1, eps=1e-12)
    eps = node.kwargs.get('eps', 1e-12)
    if len(node.args) > 3:
        eps = node.args[3]
    return NNTrainerLayerDef(
        layer_type=LAYER_L2NORM,
        name=make_scoped_name(scope, node),
        properties={"epsilon": eps},
        input_layers=input_names,
        hf_module_name=scope,
    )


def _map_getitem(node, scope, node_to_layer):
    """Map operator.getitem for multi-output module tuple unpacking.

    For RNN-family layers (gru, lstm, etc.) only index 0 is the main output;
    other indices (hidden state) are typically unused in inference.

    For split/chunk layers, ALL indices are meaningful outputs that downstream
    layers reference, so every index produces an identity layer connected to
    the parent split.
    """
    if len(node.args) >= 2 and hasattr(node.args[0], 'name'):
        parent_name = node.args[0].name
        parent_layer = node_to_layer.get(parent_name)
        idx = node.args[1]
        if parent_layer and parent_layer.layer_type in _MULTI_OUTPUT_LAYER_TYPES:
            # Split: every indexed output is used
            if parent_layer.layer_type == "split":
                return NNTrainerLayerDef(
                    layer_type=LAYER_IDENTITY,
                    name=node.name,
                    input_layers=[parent_name],
                    hf_module_name=scope,
                    properties={"split_index": idx},
                )
            # RNN-family: only index 0 (main output)
            if idx == 0:
                return NNTrainerLayerDef(
                    layer_type=LAYER_IDENTITY,
                    name=node.name,
                    input_layers=[parent_name],
                    hf_module_name=scope,
                )
    return None

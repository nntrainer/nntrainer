"""Module mapper: Maps call_module FX nodes to NNTrainer layer definitions.

Handles nn.Module leaf nodes such as Linear, Embedding, LayerNorm, RMSNorm,
Conv1d/Conv2d, BatchNorm, GRU/LSTM/RNN, activation modules, and Dropout.
"""

import torch.nn as nn

from nntrainer_layers import (
    NNTrainerLayerDef,
    LAYER_FC, LAYER_EMBEDDING, LAYER_RMS_NORM, LAYER_LAYER_NORM,
    LAYER_ACTIVATION, LAYER_DROPOUT, LAYER_IDENTITY,
    LAYER_CONV1D, LAYER_CONV2D, LAYER_CONV2D_TRANSPOSE,
    LAYER_DEPTHWISE_CONV1D, LAYER_DEPTHWISE_CONV2D,
    LAYER_POOLING2D, LAYER_UPSAMPLE2D, LAYER_BATCH_NORM,
    LAYER_GROUP_NORM, LAYER_INSTANCE_NORM,
    LAYER_CHANNEL_SHUFFLE, LAYER_L2NORM, LAYER_MHA,
    LAYER_GRU, LAYER_LSTM, LAYER_RNN,
    LAYER_GRUCELL, LAYER_LSTMCELL, LAYER_RNNCELL,
    LAYER_SSM,
    LAYER_LOSS_MSE, LAYER_LOSS_CROSS_ENTROPY_SOFTMAX,
    LAYER_LOSS_CROSS_ENTROPY_SIGMOID, LAYER_LOSS_KLD,
    ACT_RELU, ACT_GELU, ACT_SWISH, ACT_SIGMOID, ACT_TANH, ACT_SOFTMAX,
)
from tracer import _is_rmsnorm, _is_gelu_variant
from mapper_helpers import get_input_node_names, sanitize_name

# Backward-compatible aliases
_sanitize_name = sanitize_name
_get_input_node_names = get_input_node_names


# Set of layer types that return tuple outputs (output, hidden_state)
MULTI_OUTPUT_LAYER_TYPES = frozenset({
    LAYER_GRU, LAYER_LSTM, LAYER_RNN,
    LAYER_LSTMCELL,  # LSTMCell returns (h, c) tuple
})

# Activation module type -> activation constant
_MODULE_ACTIVATIONS = [
    (nn.ReLU,    ACT_RELU),
    (nn.GELU,    ACT_GELU),
    (nn.SiLU,    ACT_SWISH),
    (nn.Sigmoid, ACT_SIGMOID),
    (nn.Tanh,    ACT_TANH),
    (nn.Softmax, ACT_SOFTMAX),
]


def map_module_node(node, modules, node_to_layer):
    """Map a call_module node to NNTrainerLayerDef.

    Args:
        node: FX graph node with op == "call_module"
        modules: dict from model.named_modules()
        node_to_layer: dict mapping node.name -> NNTrainerLayerDef (for lookups)

    Returns:
        NNTrainerLayerDef or None
    """
    module_name = node.target
    module = modules.get(module_name)
    if module is None:
        return None

    module_type = type(module).__name__
    input_names = _get_input_node_names(node)

    # nn.Linear -> fully_connected
    if isinstance(module, nn.Linear):
        return NNTrainerLayerDef(
            layer_type=LAYER_FC,
            name=_sanitize_name(module_name),
            properties={
                "unit": module.out_features,
                "disable_bias": module.bias is None,
            },
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
            has_weight=True,
            has_bias=module.bias is not None,
            weight_hf_key=f"{module_name}.weight",
            bias_hf_key=f"{module_name}.bias" if module.bias is not None else "",
            transpose_weight=True,
        )

    # nn.Embedding -> embedding_layer
    if isinstance(module, nn.Embedding):
        props = {
            "in_dim": module.num_embeddings,
            "out_dim": module.embedding_dim,
        }
        if hasattr(module, "scalar_embed_scale"):
            val = module.scalar_embed_scale
            if isinstance(val, (int, float)):
                props["embed_scale"] = float(val)
            elif hasattr(val, 'item'):
                props["embed_scale"] = float(val.item())
            else:
                props["embed_scale"] = str(val)
        return NNTrainerLayerDef(
            layer_type=LAYER_EMBEDDING,
            name=_sanitize_name(module_name),
            properties=props,
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
            has_weight=True,
            weight_hf_key=f"{module_name}.weight",
            transpose_weight=False,
        )

    # nn.LayerNorm -> layer_normalization
    if isinstance(module, nn.LayerNorm):
        return NNTrainerLayerDef(
            layer_type=LAYER_LAYER_NORM,
            name=_sanitize_name(module_name),
            properties={"epsilon": module.eps},
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
            has_weight=module.elementwise_affine,
            has_bias=module.elementwise_affine,
            weight_hf_key=f"{module_name}.weight" if module.elementwise_affine else "",
            bias_hf_key=f"{module_name}.bias" if module.elementwise_affine else "",
        )

    # *RMSNorm (any HF variant) -> rms_norm
    if _is_rmsnorm(module):
        eps = getattr(module, "eps", None) or getattr(module, "variance_epsilon", 1e-6)
        return NNTrainerLayerDef(
            layer_type=LAYER_RMS_NORM,
            name=_sanitize_name(module_name),
            properties={"epsilon": eps, "packed": False},
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
            has_weight=hasattr(module, "weight"),
            weight_hf_key=f"{module_name}.weight" if hasattr(module, "weight") else "",
        )

    # Standard activation modules
    for module_cls, act_type in _MODULE_ACTIVATIONS:
        if isinstance(module, module_cls):
            return _make_activation(module_name, module_type, act_type, input_names)

    # HuggingFace custom GELU variants
    if _is_gelu_variant(module):
        return _make_activation(module_name, module_type, ACT_GELU, input_names)

    # Dropout
    if isinstance(module, nn.Dropout):
        return NNTrainerLayerDef(
            layer_type=LAYER_DROPOUT,
            name=_sanitize_name(module_name),
            properties={"dropout_rate": module.p},
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
        )

    # Conv layers
    if isinstance(module, nn.Conv1d):
        # Depthwise conv1d: groups == in_channels and out_channels == in_channels
        if module.groups == module.in_channels and module.out_channels == module.in_channels:
            return _map_depthwise_conv1d(module, module_name, module_type, input_names)
        return _map_conv1d(module, module_name, module_type, input_names)

    if isinstance(module, nn.ConvTranspose2d):
        return _map_conv2d_transpose(module, module_name, module_type, input_names)

    if isinstance(module, nn.Conv2d):
        # Depthwise conv: groups == in_channels and out_channels == in_channels
        if module.groups == module.in_channels and module.out_channels == module.in_channels:
            return _map_depthwise_conv2d(module, module_name, module_type, input_names)
        return _map_conv2d(module, module_name, module_type, input_names)

    # Upsample
    if isinstance(module, nn.Upsample):
        return _map_upsample2d(module, module_name, module_type, input_names)

    # Pooling layers
    if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
        ks = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        st = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
        pd = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
        pool_type = "max" if isinstance(module, nn.MaxPool2d) else "average"
        return NNTrainerLayerDef(
            layer_type=LAYER_POOLING2D,
            name=_sanitize_name(module_name),
            properties={
                "pooling": pool_type,
                "pool_size": f"{ks[0]},{ks[1]}",
                "stride": f"{st[0]},{st[1]}",
                "padding": f"{pd[0]},{pd[1]}",
            },
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
        )

    if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
        out = module.output_size
        if isinstance(out, int):
            out = (out, out)
        pool_type = "average" if isinstance(module, nn.AdaptiveAvgPool2d) else "max"
        return NNTrainerLayerDef(
            layer_type=LAYER_POOLING2D,
            name=_sanitize_name(module_name),
            properties={
                "pooling": pool_type,
                "pool_size": f"{out[0]},{out[1]}",
                "adaptive": True,
            },
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
        )

    # BatchNorm
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return NNTrainerLayerDef(
            layer_type=LAYER_BATCH_NORM,
            name=_sanitize_name(module_name),
            properties={"epsilon": module.eps, "momentum": module.momentum},
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
            has_weight=module.affine,
            has_bias=module.affine,
            weight_hf_key=f"{module_name}.weight" if module.affine else "",
            bias_hf_key=f"{module_name}.bias" if module.affine else "",
        )

    # GroupNorm
    if isinstance(module, nn.GroupNorm):
        return NNTrainerLayerDef(
            layer_type=LAYER_GROUP_NORM,
            name=_sanitize_name(module_name),
            properties={
                "num_groups": module.num_groups,
                "epsilon": module.eps,
            },
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
            has_weight=module.affine,
            has_bias=module.affine,
            weight_hf_key=f"{module_name}.weight" if module.affine else "",
            bias_hf_key=f"{module_name}.bias" if module.affine else "",
        )

    # InstanceNorm
    if isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d)):
        affine = module.affine if hasattr(module, 'affine') else False
        return NNTrainerLayerDef(
            layer_type=LAYER_INSTANCE_NORM,
            name=_sanitize_name(module_name),
            properties={"epsilon": module.eps},
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
            has_weight=affine,
            has_bias=affine,
            weight_hf_key=f"{module_name}.weight" if affine else "",
            bias_hf_key=f"{module_name}.bias" if affine else "",
        )

    # MultiheadAttention
    if isinstance(module, nn.MultiheadAttention):
        return _map_multihead_attention(module, module_name, module_type, input_names)

    # ChannelShuffle
    if isinstance(module, nn.ChannelShuffle):
        return NNTrainerLayerDef(
            layer_type=LAYER_CHANNEL_SHUFFLE,
            name=_sanitize_name(module_name),
            properties={"split_number": module.groups},
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
        )

    # RNN family
    if isinstance(module, nn.GRU):
        return _map_rnn_module(module, module_name, module_type, input_names, LAYER_GRU)
    if isinstance(module, nn.LSTM):
        return _map_rnn_module(module, module_name, module_type, input_names, LAYER_LSTM)
    if isinstance(module, nn.RNN):
        return _map_rnn_module(module, module_name, module_type, input_names, LAYER_RNN)

    # RNN cell family
    if isinstance(module, nn.GRUCell):
        return _map_rnn_cell(module, module_name, module_type, input_names, LAYER_GRUCELL)
    if isinstance(module, nn.LSTMCell):
        return _map_rnn_cell(module, module_name, module_type, input_names, LAYER_LSTMCELL)
    if isinstance(module, nn.RNNCell):
        return _map_rnn_cell(module, module_name, module_type, input_names, LAYER_RNNCELL)

    # Identity (passthrough)
    if isinstance(module, nn.Identity):
        return NNTrainerLayerDef(
            layer_type=LAYER_IDENTITY,
            name=_sanitize_name(module_name),
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
        )

    # Loss layers
    if isinstance(module, nn.CrossEntropyLoss):
        return NNTrainerLayerDef(
            layer_type=LAYER_LOSS_CROSS_ENTROPY_SOFTMAX,
            name=_sanitize_name(module_name),
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
        )
    if isinstance(module, nn.MSELoss):
        return NNTrainerLayerDef(
            layer_type=LAYER_LOSS_MSE,
            name=_sanitize_name(module_name),
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
        )
    if isinstance(module, nn.KLDivLoss):
        return NNTrainerLayerDef(
            layer_type=LAYER_LOSS_KLD,
            name=_sanitize_name(module_name),
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
        )
    if isinstance(module, nn.BCEWithLogitsLoss):
        return NNTrainerLayerDef(
            layer_type=LAYER_LOSS_CROSS_ENTROPY_SIGMOID,
            name=_sanitize_name(module_name),
            input_layers=input_names,
            hf_module_name=module_name,
            hf_module_type=module_type,
        )

    # MambaMixer / SSM (when used as leaf module)
    if _is_mamba_mixer(module):
        return _map_mamba_mixer(module, module_name, module_type, input_names)

    # Unknown module type -- try plugin registry as last resort
    from plugin_registry import get_global_registry
    plugin_registry = get_global_registry()
    plugin_result = plugin_registry.map_module(module, module_name, input_names)
    if plugin_result is not None:
        return plugin_result

    print(f"  [WARNING] Unknown module type: {module_type} at {module_name}")
    return NNTrainerLayerDef(
        layer_type=f"unknown({module_type})",
        name=_sanitize_name(module_name),
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
    )


def _make_activation(module_name, module_type, act_type, input_names):
    return NNTrainerLayerDef(
        layer_type=LAYER_ACTIVATION,
        name=_sanitize_name(module_name),
        properties={"activation": act_type},
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
    )


def _map_conv1d(module, module_name, module_type, input_names):
    return NNTrainerLayerDef(
        layer_type=LAYER_CONV1D,
        name=_sanitize_name(module_name),
        properties={
            "filters": module.out_channels,
            "kernel_size": module.kernel_size[0],
            "stride": module.stride[0],
            "padding": module.padding[0],
        },
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
        has_weight=True,
        has_bias=module.bias is not None,
        weight_hf_key=f"{module_name}.weight",
        bias_hf_key=f"{module_name}.bias" if module.bias is not None else "",
    )


def _map_depthwise_conv1d(module, module_name, module_type, input_names):
    """Map nn.Conv1d with groups==in_channels to NNTrainer depthwiseconv1d layer."""
    return NNTrainerLayerDef(
        layer_type=LAYER_DEPTHWISE_CONV1D,
        name=_sanitize_name(module_name),
        properties={
            "filters": module.out_channels,
            "kernel_size": module.kernel_size[0],
            "stride": module.stride[0],
            "padding": module.padding[0],
            "dilation": module.dilation[0],
        },
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
        has_weight=True,
        has_bias=module.bias is not None,
        weight_hf_key=f"{module_name}.weight",
        bias_hf_key=f"{module_name}.bias" if module.bias is not None else "",
    )


def _map_conv2d(module, module_name, module_type, input_names):
    return NNTrainerLayerDef(
        layer_type=LAYER_CONV2D,
        name=_sanitize_name(module_name),
        properties={
            "filters": module.out_channels,
            "kernel_size": f"{module.kernel_size[0]},{module.kernel_size[1]}",
            "stride": f"{module.stride[0]},{module.stride[1]}",
            "padding": f"{module.padding[0]},{module.padding[1]}",
        },
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
        has_weight=True,
        has_bias=module.bias is not None,
        weight_hf_key=f"{module_name}.weight",
        bias_hf_key=f"{module_name}.bias" if module.bias is not None else "",
    )


def _map_conv2d_transpose(module, module_name, module_type, input_names):
    """Map nn.ConvTranspose2d to NNTrainer conv2dtranspose layer."""
    return NNTrainerLayerDef(
        layer_type=LAYER_CONV2D_TRANSPOSE,
        name=_sanitize_name(module_name),
        properties={
            "filters": module.out_channels,
            "kernel_size": f"{module.kernel_size[0]},{module.kernel_size[1]}",
            "stride": f"{module.stride[0]},{module.stride[1]}",
            "padding": f"{module.padding[0]},{module.padding[1]}",
            "dilation": f"{module.dilation[0]},{module.dilation[1]}",
        },
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
        has_weight=True,
        has_bias=module.bias is not None,
        weight_hf_key=f"{module_name}.weight",
        bias_hf_key=f"{module_name}.bias" if module.bias is not None else "",
    )


def _map_depthwise_conv2d(module, module_name, module_type, input_names):
    """Map nn.Conv2d with groups==in_channels to NNTrainer depthwiseconv2d layer."""
    return NNTrainerLayerDef(
        layer_type=LAYER_DEPTHWISE_CONV2D,
        name=_sanitize_name(module_name),
        properties={
            "filters": module.out_channels,
            "kernel_size": f"{module.kernel_size[0]},{module.kernel_size[1]}",
            "stride": f"{module.stride[0]},{module.stride[1]}",
            "padding": f"{module.padding[0]},{module.padding[1]}",
            "dilation": f"{module.dilation[0]},{module.dilation[1]}",
        },
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
        has_weight=True,
        has_bias=module.bias is not None,
        weight_hf_key=f"{module_name}.weight",
        bias_hf_key=f"{module_name}.bias" if module.bias is not None else "",
    )


def _map_upsample2d(module, module_name, module_type, input_names):
    """Map nn.Upsample to NNTrainer upsample2d layer."""
    mode = module.mode if module.mode in ("nearest", "bilinear") else "nearest"
    props = {"upsample": mode}
    if module.scale_factor is not None:
        sf = module.scale_factor
        if isinstance(sf, (tuple, list)):
            props["kernel_size"] = f"{int(sf[0])},{int(sf[1])}"
        else:
            props["kernel_size"] = f"{int(sf)},{int(sf)}"
    elif module.size is not None:
        # For fixed output size, encode as kernel_size (best-effort)
        sz = module.size
        if isinstance(sz, int):
            props["kernel_size"] = f"{sz},{sz}"
        else:
            props["kernel_size"] = f"{sz[0]},{sz[1]}"
    return NNTrainerLayerDef(
        layer_type=LAYER_UPSAMPLE2D,
        name=_sanitize_name(module_name),
        properties=props,
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
    )


def _map_multihead_attention(module, module_name, module_type, input_names):
    """Map nn.MultiheadAttention to NNTrainer multi_head_attention layer."""
    props = {
        "num_heads": module.num_heads,
        "projected_key_dim": module.kdim // module.num_heads,
        "projected_value_dim": module.vdim // module.num_heads,
        "output_shape": module.embed_dim,
    }
    if module.dropout > 0:
        props["dropout_rate"] = module.dropout
    return NNTrainerLayerDef(
        layer_type=LAYER_MHA,
        name=_sanitize_name(module_name),
        properties=props,
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
        has_weight=True,
        has_bias=module.in_proj_bias is not None,
        weight_hf_key=f"{module_name}.in_proj_weight",
        bias_hf_key=f"{module_name}.in_proj_bias" if module.in_proj_bias is not None else "",
    )


def _map_rnn_module(module, module_name, module_type, input_names, layer_type):
    """Map nn.GRU/LSTM/RNN to NNTrainer layer def."""
    if module.num_layers != 1:
        print(f"  [WARNING] Multi-layer {module_type} (num_layers="
              f"{module.num_layers}) not yet supported, mapping layer 0 only")

    props = {
        "unit": module.hidden_size,
        "return_sequences": True,
    }

    if hasattr(module, 'dropout') and module.dropout > 0:
        props["dropout_rate"] = module.dropout

    if module.bidirectional:
        if layer_type == LAYER_LSTM:
            props["bidirectional"] = True
        else:
            print(f"  [WARNING] Bidirectional {module_type} not supported "
                  f"in NNTrainer, using forward-only")

    weight_keys = {
        "weight_ih": f"{module_name}.weight_ih_l0",
        "weight_hh": f"{module_name}.weight_hh_l0",
    }
    bias_keys = {}
    if module.bias:
        bias_keys["bias_ih"] = f"{module_name}.bias_ih_l0"
        bias_keys["bias_hh"] = f"{module_name}.bias_hh_l0"

    layer_def = NNTrainerLayerDef(
        layer_type=layer_type,
        name=_sanitize_name(module_name),
        properties=props,
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
        has_weight=True,
        has_bias=module.bias,
        weight_hf_key=weight_keys["weight_ih"],
        bias_hf_key=bias_keys.get("bias_ih", ""),
        transpose_weight=True,
    )
    layer_def.properties["_weight_hh_key"] = weight_keys["weight_hh"]
    if module.bias:
        layer_def.properties["_bias_hh_key"] = bias_keys["bias_hh"]

    return layer_def


def _map_rnn_cell(module, module_name, module_type, input_names, layer_type):
    """Map nn.GRUCell/LSTMCell/RNNCell to NNTrainer cell layer def."""
    props = {
        "unit": module.hidden_size,
    }

    has_bias = module.bias
    weight_keys = {
        "weight_ih": f"{module_name}.weight_ih",
        "weight_hh": f"{module_name}.weight_hh",
    }
    bias_keys = {}
    if has_bias:
        bias_keys["bias_ih"] = f"{module_name}.bias_ih"
        bias_keys["bias_hh"] = f"{module_name}.bias_hh"

    layer_def = NNTrainerLayerDef(
        layer_type=layer_type,
        name=_sanitize_name(module_name),
        properties=props,
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
        has_weight=True,
        has_bias=has_bias,
        weight_hf_key=weight_keys["weight_ih"],
        bias_hf_key=bias_keys.get("bias_ih", ""),
        transpose_weight=True,
    )
    layer_def.properties["_weight_hh_key"] = weight_keys["weight_hh"]
    if has_bias:
        layer_def.properties["_bias_hh_key"] = bias_keys["bias_hh"]

    return layer_def


def _is_mamba_mixer(module):
    """Check if a module is a Mamba mixer (SSM) module.

    HuggingFace Mamba uses classes named MambaMixer, Mamba2Mixer, etc.
    """
    cls_name = type(module).__name__
    return "MambaMixer" in cls_name or "Mamba2Mixer" in cls_name


def _map_mamba_mixer(module, module_name, module_type, input_names):
    """Map MambaMixer to NNTrainer SSM layer def.

    This is used when MambaMixer is kept as a leaf module.
    Extracts SSM configuration from the module attributes.
    """
    props = {}
    # Extract SSM parameters from the module
    if hasattr(module, "d_state"):
        props["state_size"] = module.d_state
    if hasattr(module, "d_conv"):
        props["conv_kernel"] = module.d_conv
    if hasattr(module, "expand"):
        props["expand"] = module.expand
    if hasattr(module, "d_inner"):
        props["inner_dim"] = module.d_inner
    if hasattr(module, "dt_rank"):
        dt_rank = module.dt_rank
        if isinstance(dt_rank, str):  # "auto"
            dt_rank = max(1, getattr(module, "d_model", 0) // 16)
        props["dt_rank"] = dt_rank

    return NNTrainerLayerDef(
        layer_type=LAYER_SSM,
        name=_sanitize_name(module_name),
        properties=props,
        input_layers=input_names,
        hf_module_name=module_name,
        hf_module_type=module_type,
        has_weight=True,
        has_bias=False,
        weight_hf_key=f"{module_name}.in_proj.weight",
    )

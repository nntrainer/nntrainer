"""FFN pattern detection."""

from nntrainer_layers import (
    NNTrainerLayerDef, LAYER_FC, LAYER_ACTIVATION, LAYER_MULTIPLY,
)
from .data_types import FFNPattern


def detect_ffn(block_idx, ffn_scope, block_layers, by_name):
    """Detect FFN pattern within the given scope.

    Args:
        block_idx: Block number (0-based)
        ffn_scope: HF module scope for FFN
        block_layers: layers in this block
        by_name: dict mapping layer name -> NNTrainerLayerDef

    Returns:
        FFNPattern
    """
    ffn = FFNPattern(block_idx=block_idx)

    ffn_layers = [l for l in block_layers
                  if l.hf_module_name.startswith(ffn_scope)]

    fc_layers = [(l, l.hf_module_name[len(ffn_scope):].lstrip("."))
                 for l in ffn_layers if l.layer_type == LAYER_FC]

    gate, up, down = _match_fc_roles(fc_layers)

    if gate and up and down:
        _build_gated_ffn(ffn, gate, up, down, ffn_scope, block_layers)
    elif len(fc_layers) >= 2 and not gate:
        _build_standard_ffn(ffn, fc_layers, up, down, ffn_scope,
                            block_layers, by_name)

    return ffn


def _match_fc_roles(fc_layers):
    """Match FC layers to gate/up/down roles by name suffix."""
    gate = up = down = None
    w1_layer = None

    for layer, suffix in fc_layers:
        if suffix in ("gate_proj", "gate"):
            gate = layer
        elif suffix in ("up_proj", "up", "wi_1", "w1"):
            up = layer
            if suffix == "w1":
                w1_layer = layer
        elif suffix in ("down_proj", "down", "wo", "w2"):
            down = layer
        elif suffix in ("w3",):
            up = layer
        elif suffix in ("wi_0",):
            gate = layer
        elif suffix in ("wi",):
            up = layer
        elif suffix in ("intermediate.dense",):
            up = layer
        elif suffix in ("output.dense",):
            down = layer

    # LFM2-style remap: w1 is gate, w3 is up, w2 is down
    if w1_layer and not gate and up and up is not w1_layer and down:
        gate = w1_layer

    return gate, up, down


def _build_gated_ffn(ffn, gate, up, down, ffn_scope, block_layers):
    """Build SwiGLU/GeGLU/gated FFN pattern."""
    ffn.ffn_type = "swiglu"
    ffn.gate_proj = gate.name
    ffn.up_proj = up.name
    ffn.down_proj = down.name
    ffn.intermediate_size = int(gate.properties.get("unit", 0))
    ffn.layer_names = [gate.name, up.name, down.name]

    for layer in block_layers:
        if (layer.layer_type == LAYER_ACTIVATION
            and layer.hf_module_name.startswith(ffn_scope)):
            ffn.activation = layer.name
            ffn.layer_names.append(layer.name)
            act_type = layer.properties.get("activation", "")
            if act_type == "gelu":
                ffn.ffn_type = "geglu"
            elif act_type in ("relu", "tanh", "sigmoid"):
                ffn.ffn_type = f"gated_{act_type}"
        elif (layer.layer_type == LAYER_MULTIPLY
              and layer.hf_module_name.startswith(ffn_scope)):
            ffn.gate_multiply = layer.name
            ffn.layer_names.append(layer.name)


def _build_standard_ffn(ffn, fc_layers, up, down, ffn_scope,
                        block_layers, by_name):
    """Build GELU-FFN or standard FFN pattern."""
    ffn.ffn_type = "gelu_ffn"
    if up and down:
        ffn.up_proj = up.name
        ffn.down_proj = down.name
    else:
        ffn.up_proj = fc_layers[0][0].name
        ffn.down_proj = fc_layers[-1][0].name
    ffn.intermediate_size = int(
        by_name.get(ffn.up_proj, NNTrainerLayerDef(
            layer_type="", name="")).properties.get("unit", 0))
    ffn.layer_names = [ffn.up_proj, ffn.down_proj]

    for layer in block_layers:
        if (layer.layer_type == LAYER_ACTIVATION
            and layer.hf_module_name.startswith(ffn_scope)):
            ffn.activation = layer.name
            ffn.layer_names.append(layer.name)
            act_type = layer.properties.get("activation", "")
            if act_type == "gelu":
                ffn.ffn_type = "gelu_ffn"
            elif act_type == "relu":
                ffn.ffn_type = "standard"

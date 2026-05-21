"""SSM (Mamba) pattern detection.

Detects MambaMixer patterns within a block scope:
  - in_proj: Linear projection expanding input dimension
  - conv1d: Causal 1D convolution
  - x_proj: Linear projection for selection parameters (B, C)
  - dt_proj: Linear projection for time step delta
  - out_proj: Linear projection contracting back to hidden dim
"""

from .data_types import SSMPattern


def detect_ssm(block_idx, ssm_scope, block_layers, config=None):
    """Detect SSM (Mamba) pattern within a block.

    Args:
        block_idx: Block number (0-based)
        ssm_scope: HF module scope (e.g. "model.layers.0.mixer")
        block_layers: All layers in the parent block
        config: HF model config (optional, for SSM parameters)

    Returns:
        SSMPattern or None
    """
    ssm_layers = [l for l in block_layers
                  if l.hf_module_name.startswith(ssm_scope + ".")
                  or l.hf_module_name == ssm_scope]
    if not ssm_layers:
        return None

    pattern = SSMPattern(block_idx=block_idx)
    layer_names = []

    for layer in ssm_layers:
        suffix = layer.hf_module_name[len(ssm_scope):].lstrip(".")
        name = layer.name
        layer_names.append(name)

        if suffix == "in_proj":
            pattern.in_proj = name
        elif suffix == "conv1d":
            pattern.conv1d = name
        elif suffix == "x_proj":
            pattern.x_proj = name
        elif suffix == "dt_proj":
            pattern.dt_proj = name
        elif suffix == "out_proj":
            pattern.out_proj = name
        elif suffix in ("norm", "dt_layernorm", "B_layernorm",
                        "inner_layernorm"):
            pattern.norm = name

    pattern.layer_names = layer_names

    # Extract SSM parameters from config
    if config is not None:
        pattern.state_size = getattr(config, "state_size",
                             getattr(config, "ssm_state_size", 0))
        pattern.conv_kernel = getattr(config, "conv_kernel",
                              getattr(config, "d_conv", 0))
        pattern.expand = getattr(config, "expand",
                         getattr(config, "ssm_expand", 0))
        pattern.dt_rank = getattr(config, "time_step_rank",
                          getattr(config, "dt_rank", 0))
        # Handle "auto" dt_rank
        if isinstance(pattern.dt_rank, str) and pattern.dt_rank == "auto":
            hidden = getattr(config, "hidden_size", 0)
            pattern.dt_rank = max(1, hidden // 16) if hidden else 0

    # Detect SSM type
    model_type = getattr(config, "model_type", "") if config else ""
    if "mamba2" in model_type:
        pattern.ssm_type = "mamba2"
    else:
        pattern.ssm_type = "mamba"

    return pattern

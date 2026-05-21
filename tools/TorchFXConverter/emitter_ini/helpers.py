"""Shared helpers for INI emitter."""


def norm_type_for_model(model_type):
    """Determine norm type string for INI config.

    Args:
        model_type: HF model type string (e.g. "bert", "llama").

    Returns:
        str: NNTrainer norm type ("layer_normalization" or "rms_norm").
    """
    if model_type in ("bert", "roberta", "distilbert", "albert"):
        return "layer_normalization"
    return "rms_norm"


def format_property(key, value):
    """Format a single layer property as an INI key-value line."""
    if isinstance(value, bool):
        return f"{key} = {'true' if value else 'false'}"
    elif isinstance(value, (list, tuple)):
        return f"{key} = {','.join(str(x) for x in value)}"
    else:
        return f"{key} = {value}"

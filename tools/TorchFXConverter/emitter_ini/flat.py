"""Flat mode INI emission: verbatim layer list output."""

from .helpers import format_property


def emit_flat(layers, batch_size):
    """Emit every layer from the flat list as-is.

    Args:
        layers: List of NNTrainerLayerDef.
        batch_size: Batch size for [Model] section.

    Returns:
        str: Complete INI file content in flat mode.
    """
    sections = []
    sections.append("# Auto-generated NNTrainer configuration (flat mode)")
    sections.append("")

    sections.append("[Model]")
    sections.append("Type = NeuralNetwork")
    sections.append(f"batch_size = {batch_size}")
    sections.append("")

    for layer in layers:
        sections.append(f"[{layer.name}]")
        sections.append(f"Type = {layer.layer_type}")
        if layer.input_layers:
            sections.append(
                f"input_layers = {','.join(layer.input_layers)}")
        for k, v in layer.properties.items():
            sections.append(format_property(k, v))
        sections.append("")

    return "\n".join(sections)

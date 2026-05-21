"""INI configuration emitter package for NNTrainer TorchFX converter.

Generates .ini configuration files that can be loaded directly by NNTrainer
via model->load("model.ini", MODEL_FORMAT_INI).

Sub-modules:
  - helpers.py    : Norm type detection, property formatting
  - structured.py : Structured (pattern-based) emission
  - flat.py       : Flat (verbatim layer list) emission
"""

from emitter_base import BaseEmitter
from .structured import emit_structured
from .flat import emit_flat


class IniEmitter(BaseEmitter):
    """Generates NNTrainer INI configuration from converter output.

    Two modes:
      1. Flat mode: Emit every layer from the flat layer list (verbose, exact).
      2. Structured mode: Use ModelStructure to emit a clean, readable config
         with block structure and proper naming.
    """

    def __init__(self, layers, structure, batch_size=1, model_name=None):
        super().__init__(layers, structure, model_name=model_name)
        self.batch_size = batch_size

    def emit(self, mode="structured"):
        """Generate INI configuration string.

        Args:
            mode: "structured" uses pattern-detected structure for clean output.
                  "flat" emits every layer from the flat list verbatim.
                  When structured mode is requested but no transformer blocks
                  are detected, falls back to flat mode automatically.

        Returns:
            str: Complete INI file content.
        """
        if mode == "flat":
            return emit_flat(self.layers, self.batch_size)
        # Fall back to flat mode for non-transformer models
        if not self.structure.blocks:
            return emit_flat(self.layers, self.batch_size)
        return emit_structured(self.layers, self.structure, self.batch_size)


# =============================================================================
# Convenience function
# =============================================================================

def emit_ini(layers, structure, batch_size=1, mode="structured"):
    """Generate NNTrainer INI configuration.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        structure: ModelStructure from pattern detection.
        batch_size: Batch size for the model.
        mode: "structured" or "flat".

    Returns:
        str: Complete INI file content.
    """
    emitter = IniEmitter(layers, structure, batch_size)
    return emitter.emit(mode=mode)

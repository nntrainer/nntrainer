"""Base emitter interface for NNTrainer code generators.

Provides the common abstract interface that all emitters (C++, INI, JSON)
should implement, plus shared utility functions.
"""

from abc import ABC, abstractmethod


class BaseEmitter(ABC):
    """Abstract base class for NNTrainer emitters.

    All emitters receive the converter output (layers + structure) and
    produce a string representation in their target format.
    """

    def __init__(self, layers, structure, model_name=None):
        """
        Args:
            layers: List of NNTrainerLayerDef from converter pipeline.
            structure: ModelStructure from pattern detection.
            model_name: Optional model name for file naming.
        """
        self.layers = layers
        self.structure = structure
        self._by_name = {l.name: l for l in layers}
        self._model_name = model_name

    @abstractmethod
    def emit(self) -> str:
        """Generate the complete output string."""
        ...

    @staticmethod
    def format_property(key, value):
        """Format a layer property value for text output (INI/JSON).

        Handles bool, list/tuple, and scalar types.
        """
        if isinstance(value, bool):
            return f"{key} = {'true' if value else 'false'}"
        elif isinstance(value, (list, tuple)):
            return f"{key} = {','.join(str(x) for x in value)}"
        else:
            return f"{key} = {value}"

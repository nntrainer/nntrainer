"""
ArchitectureAdapter -- Strategy interface.

Each concrete adapter knows how to read one family of HuggingFace
causal-LM configs/module-trees and turn them into the framework-neutral
`api.semantic.model.CausalLMIR`. Nothing downstream (the nntrainer
lowerer, the C++ generator, the webview) needs to know which adapter
produced the IR -- that's the whole point of the indirection.

Adapters are stateless and safe to share across threads: `matches()`
and `build_semantic_ir()` only read from the config/model they're
given and allocate fresh IR objects, they never mutate shared state.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from api.semantic.model import CausalLMIR


class ArchitectureAdapter(ABC):
    #: HF `config.model_type` values this adapter handles.
    model_types: tuple[str, ...] = ()

    def matches(self, config: Any, model: Any) -> bool:
        return getattr(config, "model_type", None) in self.model_types

    @abstractmethod
    def build_semantic_ir(self, config: Any, model: Any) -> CausalLMIR:
        """Build a CausalLMIR from a live (from_config-constructed)
        model + its HF config. Must not download or touch real weights
        -- callers only ever pass AutoModel.from_config() instances."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_types={self.model_types})"

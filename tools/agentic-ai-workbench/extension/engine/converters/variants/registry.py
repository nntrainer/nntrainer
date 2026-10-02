"""
Registry of architecture-specific "variant" C++ generators.

Not every architecture is best generated from scratch via the generic
uniform-decoder CPPGenerator (converters/cpp_generator.py) -- some are a
small delta on top of an architecture nntrainer already has a real,
working implementation for: inherit from that class in C++, override
only the handful of methods that actually differ. Qwen3-MoE is the
motivating example (see converters/variants/qwen_moe.py): the real
hand-written qwen3_moe_causallm.cpp inherits from Qwen3CausalLM and only
swaps createMlp() for a custom expert-routing layer plus a couple of
extra setupParameters() fields -- it does not re-derive attention or the
decoder loop at all.

This registry lets each such family register its own detector ("is this
the architecture I handle?") and generator (produces
converters.cpp_generator.ComponentFiles, so callers don't need to know
which path produced a result), so adding support for a new family means
adding a new module here, not editing cpp_generator.py or the agent's
dispatch logic.
"""
from typing import Callable, List, Optional, Tuple

from ..cpp_generator import ComponentFiles

Detector = Callable[[str, dict], bool]
Generator = Callable[[str, dict], ComponentFiles]

_VARIANTS: List[Tuple[str, Detector, Generator]] = []


def register_variant(name: str, detector: Detector, generator: Generator) -> None:
    _VARIANTS.append((name, detector, generator))


def find_variant(architecture: str, hf_config: dict) -> Tuple[Optional[str], Optional[Generator]]:
    """
    Returns (variant_name, generator_fn) for the first registered variant
    whose detector matches, or (None, None) if none do -- callers should
    fall back to the generic CPPGenerator path in that case.
    """
    hf_config = hf_config or {}
    for name, detector, generator in _VARIANTS:
        if detector(architecture, hf_config):
            return name, generator
    return None, None
